"""#363 stage-flip corridor admission -- pricing and the gate. Hermetic.

The hole this covers, stated as the three facts that made it a hole:

* ``RegimeObserver._act_interlocks`` runs four interlocks and none is a
  memory admission;
* ``RegimeActuator.apply`` calls the #330 dial directly;
* the dial's GROW path checks only its own floor and the VA ceiling, and
  spends the corridor ladder only when the budget SHRINKS.

So a stage flip that grew a budget was the one direction that consumed free
VRAM without ever being priced against the corridor law. These tests pin the
gate that closes it, and -- more importantly -- pin that every way of NOT
knowing the price is a refusal rather than a zero. A move priced at zero is
always affordable, and a move that is always affordable is not gated.

Both duties again: the gate must OPEN on a funded move (a gate that cannot
open hides a broken path behind a permanent refusal) and CLOSE on each of the
seven ways a price can be unknown or unaffordable.
"""

import unittest

from sglang.srt.managers.regime_admission import (
    MIB,
    CorridorAdmission,
    price_stage_flip,
)
from sglang.srt.managers.regime_classifier import REGIME_MIXED, Stage
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_stage(name, *, vram=(8000, 8000), kv=(1, 1)):
    return Stage(
        name=name,
        regime=REGIME_MIXED,
        weight_vector=None,
        kv_token_vector=tuple(kv),
        vram_budget_mib=tuple(vram),
        max_total_num_tokens=100_000,
        measured_gain_pct=10.0,
        measured_band_pct=1.0,
        flip_cost_s=1.0,
    )


CUR = make_stage("balanced", vram=(8000, 8000))
#: 512 MiB more residency on rank 0 than the incumbent.
BIGGER = make_stage("split-heavy", vram=(8512, 8512))
#: Same budget vector: a KV-vector-only move, residency delta 0.
SAME_VRAM = make_stage("kv-only", vram=(8000, 8000), kv=(2, 1))

CENSUS = {"prefill_heavy": 1900.0, "decode_heavy": 700.0}


class StubGuardResult:
    def __init__(self, ok, free_before=20_000 * MIB, reclaimed=0):
        self.ok = ok
        self.free_before = free_before
        self.free_after = free_before - reclaimed
        self.reclaimed = reclaimed
        self.detail = f"stub guard ok={ok}"


class StubGuard:
    """Records what it was asked for, so the PRICE can be pinned."""

    def __init__(self, ok=True, raises=None):
        self.ok = ok
        self.raises = raises
        self.calls = []

    def ensure_headroom(self, want_bytes, *, reason="", **_kw):
        self.calls.append((want_bytes, reason))
        if self.raises is not None:
            raise self.raises
        return StubGuardResult(self.ok)


def build(
    *,
    guard=None,
    census=CENSUS,
    load_state="prefill_heavy",
    tp_size=1,
    collective_min=None,
    rank=0,
    transient_fn_wired=True,
):
    return CorridorAdmission(
        guard_fn=(lambda: guard) if guard is not None else (lambda: None),
        collective_min=collective_min,
        load_state_fn=lambda: load_state,
        transient_fn=(lambda stage: census) if transient_fn_wired else None,
        rank=rank,
        tp_size=tp_size,
    )


class TestPricing(unittest.TestCase):
    def test_price_is_residency_delta_plus_transient(self):
        price, why = price_stage_flip(
            current=CUR,
            target=BIGGER,
            rank=0,
            load_state="prefill_heavy",
            transient_by_load_state=CENSUS,
        )
        self.assertEqual(why, "")
        self.assertAlmostEqual(price.residency_delta_mib, 512.0)
        self.assertAlmostEqual(price.transient_mib, 1900.0)
        self.assertAlmostEqual(price.want_mib, 2412.0)
        self.assertEqual(price.want_bytes, int(2412.0 * MIB))

    def test_the_load_state_selects_the_transient(self):
        """Law: never a foreign load state's transient."""
        prefill, _ = price_stage_flip(
            current=CUR,
            target=BIGGER,
            rank=0,
            load_state="prefill_heavy",
            transient_by_load_state=CENSUS,
        )
        decode, _ = price_stage_flip(
            current=CUR,
            target=BIGGER,
            rank=0,
            load_state="decode_heavy",
            transient_by_load_state=CENSUS,
        )
        self.assertGreater(prefill.want_bytes, decode.want_bytes)
        self.assertAlmostEqual(prefill.transient_mib, 1900.0)
        self.assertAlmostEqual(decode.transient_mib, 700.0)

    def test_kv_only_move_still_pays_its_transient(self):
        """Residency delta 0 does not mean the flip draws nothing."""
        price, why = price_stage_flip(
            current=CUR,
            target=SAME_VRAM,
            rank=0,
            load_state="prefill_heavy",
            transient_by_load_state=CENSUS,
        )
        self.assertEqual(why, "")
        self.assertAlmostEqual(price.residency_delta_mib, 0.0)
        self.assertAlmostEqual(price.want_mib, 1900.0)

    def test_per_rank_residency_delta(self):
        lopsided = make_stage("lopsided", vram=(8000, 9000))
        price, _ = price_stage_flip(
            current=CUR,
            target=lopsided,
            rank=1,
            load_state="decode_heavy",
            transient_by_load_state=CENSUS,
        )
        self.assertAlmostEqual(price.residency_delta_mib, 1000.0)


class TestPricingRefusals(unittest.TestCase):
    """Seven ways not to know the price. All seven refuse; none returns 0."""

    def _refused(self, **kw):
        base = dict(
            current=CUR,
            target=BIGGER,
            rank=0,
            load_state="prefill_heavy",
            transient_by_load_state=CENSUS,
        )
        base.update(kw)
        price, why = price_stage_flip(**base)
        self.assertIsNone(price)
        self.assertTrue(why)
        return why

    def test_absent_census_refuses(self):
        why = self._refused(transient_by_load_state=None)
        self.assertIn("UNMEASURED", why)
        self.assertIn("reads as free memory", why)

    def test_empty_census_refuses(self):
        why = self._refused(transient_by_load_state={})
        self.assertIn("EMPTY table is not", why)

    def test_unknown_load_state_refuses_and_names_the_known_ones(self):
        why = self._refused(load_state="mixed")
        self.assertIn("'mixed'", why)
        self.assertIn("decode_heavy", why)
        self.assertIn("prefill_heavy", why)

    def test_unknown_load_state_refuses_rather_than_substituting(self):
        why = self._refused(load_state="mixed")
        self.assertIn("Substituting", why)

    def test_absent_load_state_refuses(self):
        why = self._refused(load_state=None)
        self.assertIn("load state is unknown", why)

    def test_negative_transient_refuses(self):
        why = self._refused(transient_by_load_state={"prefill_heavy": -5.0})
        self.assertIn("broken census", why)

    def test_geometry_mismatch_refuses(self):
        three_rank = make_stage("tp3", vram=(8000, 8000, 8000), kv=(1, 1, 1))
        why = self._refused(target=three_rank)
        self.assertIn("not two stages of one table", why)

    def test_rank_outside_the_vector_refuses(self):
        why = self._refused(rank=7)
        self.assertIn("outside", why)


class TestAdmission(unittest.TestCase):
    def test_the_gate_opens_on_a_funded_move(self):
        """A gate that cannot open hides a broken path behind a refusal."""
        guard = StubGuard(ok=True)
        gate = build(guard=guard)
        verdict = gate.admit(CUR, BIGGER)
        self.assertTrue(verdict.ok)
        self.assertTrue(verdict.local_ok)
        self.assertFalse(verdict.group_refused)
        self.assertEqual(gate.admitted, 1)

    def test_the_guard_is_asked_for_the_priced_bytes(self):
        """Pins that the number reaching the corridor is the priced one."""
        guard = StubGuard(ok=True)
        build(guard=guard).admit(CUR, BIGGER)
        self.assertEqual(len(guard.calls), 1)
        want_bytes, reason = guard.calls[0]
        self.assertEqual(want_bytes, int(2412.0 * MIB))
        self.assertIn("split-heavy", reason)

    def test_an_unfunded_move_is_refused_with_numbers(self):
        guard = StubGuard(ok=False)
        gate = build(guard=guard)
        verdict = gate.admit(CUR, BIGGER)
        self.assertFalse(verdict.ok)
        self.assertEqual(gate.refused, 1)
        self.assertIn("2412.0 MiB wanted", verdict.reason)
        self.assertIn("residency delta +512.0 MiB", verdict.reason)
        self.assertIn("transient 1900.0 MiB", verdict.reason)

    def test_no_guard_abstains_loudly_rather_than_passing(self):
        gate = build(guard=None)
        verdict = gate.admit(CUR, BIGGER)
        self.assertFalse(verdict.ok)
        self.assertIn("not an admitted move", verdict.reason)

    def test_no_census_reader_refuses_before_pricing(self):
        gate = build(guard=StubGuard(ok=True), transient_fn_wired=False)
        verdict = gate.admit(CUR, BIGGER)
        self.assertFalse(verdict.ok)
        self.assertIn("priced at zero", verdict.reason)

    def test_a_raising_guard_does_not_escape_into_the_loop(self):
        guard = StubGuard(raises=RuntimeError("probe exploded"))
        gate = build(guard=guard)
        verdict = gate.admit(CUR, BIGGER)
        self.assertFalse(verdict.ok)
        self.assertIn("RuntimeError", verdict.reason)
        self.assertIn("probe exploded", verdict.reason)

    def test_missing_census_refuses_even_with_a_clear_card(self):
        """The C30-family lesson: an unpriced move is not an affordable one."""
        gate = build(guard=StubGuard(ok=True), census={})
        verdict = gate.admit(CUR, BIGGER)
        self.assertFalse(verdict.ok)
        self.assertIn("EMPTY table is not", verdict.reason)


class TestGroupUniformity(unittest.TestCase):
    def test_unanimous_group_admits(self):
        gate = build(
            guard=StubGuard(ok=True),
            tp_size=2,
            collective_min=lambda payload: payload,
        )
        self.assertTrue(gate.admit(CUR, BIGGER).ok)

    def test_one_tight_rank_refuses_the_whole_group(self):
        """A rank-local admission would half-flip the group."""
        gate = build(
            guard=StubGuard(ok=True),
            tp_size=2,
            # This rank voted 1; the peer voted 0, so the MIN is 0.
            collective_min=lambda payload: [0, 0],
        )
        verdict = gate.admit(CUR, BIGGER)
        self.assertFalse(verdict.ok)
        self.assertTrue(verdict.local_ok)
        self.assertTrue(verdict.group_refused)
        self.assertIn("half-flip the group", verdict.reason)

    def test_local_refusal_is_not_reported_as_a_group_refusal(self):
        """The operator fix differs: this card needs the room, not another."""
        gate = build(
            guard=StubGuard(ok=False),
            tp_size=2,
            collective_min=lambda payload: [0, 0],
        )
        verdict = gate.admit(CUR, BIGGER)
        self.assertFalse(verdict.ok)
        self.assertFalse(verdict.local_ok)
        self.assertFalse(verdict.group_refused)

    def test_multi_rank_without_a_channel_refuses(self):
        """Unlike the observer, this path is about to spend memory."""
        gate = build(guard=StubGuard(ok=True), tp_size=2, collective_min=None)
        self.assertFalse(gate.admit(CUR, BIGGER).ok)

    def test_single_rank_needs_no_channel(self):
        gate = build(guard=StubGuard(ok=True), tp_size=1, collective_min=None)
        self.assertTrue(gate.admit(CUR, BIGGER).ok)

    def test_malformed_reduction_refuses(self):
        gate = build(
            guard=StubGuard(ok=True),
            tp_size=2,
            collective_min=lambda payload: [1],
        )
        self.assertFalse(gate.admit(CUR, BIGGER).ok)


if __name__ == "__main__":
    unittest.main()
