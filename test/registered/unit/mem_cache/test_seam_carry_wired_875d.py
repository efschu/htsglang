"""#875d: the layer carry stops being a library and becomes the restore path.

WHAT THE THREE COMMITS BEFORE THIS ONE LEFT. `seam_layer_carry.py` is exact and
proven on the layer axis, and its own header says "NOTHING IN THIS MODULE IS
WIRED INTO THE RESTORE PATH YET". A module nobody imports outside its own test
is the PRESENT-BUT-UNWIRED state, which is the expensive one in both directions:
it reads as delivered and behaves as absent. This file is the wiring, and it is
deliberately narrower than the ticket's ambition.

THE VERDICT THIS DOES NOT OVERTURN. `ec1717491f` answered axis 3 with DO NOT
BUILD, and that verdict is about a COLLECTIVE -- an all-to-all inside the
cutover's no-return region, milliseconds spent to save microseconds, in the one
place a collective is the #630 wedge shape. Nothing here issues a collective.

WHAT IS AVAILABLE WITHOUT ONE, and it is exactly the dangerous half. The refusal
guards two failure shapes that are not symmetric:

  PP copy -> TP pool   found(8) < expected(16). The loop indexes past the end:
                       `IndexError`, dead scheduler. LOUD -- this is W40. The
                       source is MISSING layers the destination needs, so the
                       data is on a peer and no rank-local answer exists. This
                       still refuses, and now refuses by NAME.

  TP copy -> PP pool   found(16) > expected(8). "NOT AN ERROR AT ALL" -- the
                       loop simply runs fewer iterations and writes the copy's
                       global layers 0..7 into the destination's global 8..15.
                       SILENT wrong-layer KV under a prefix the tree reports as
                       restored. And here the source covers EVERY layer the
                       destination needs. The carry is a rank-local SLICE. No
                       collective, no peer, no exchange.

So the direction that has no crash to make anyone look is the direction that was
always rank-locally fixable, and the refusal was paying a recompute for it.

THE ROOT FIX THIS NEEDED FIRST, and it is why this file also tests `MambaPool`.
`MambaPool.cpu_copy_layout` reported `start_layer=0` UNCONDITIONALLY: nothing in
the tree ever assigns `MambaPool.start_layer` (`getattr(self, "start_layer", 0)`
had no setter to find), while `__init__` is handed `mamba_layer_ids` -- the exact
global identity -- and kept only `len()` of it. Two consequences, one of them a
defect that predates this ticket:

  * a carry keyed on that identity would slice by a start_layer that is a
    default rather than a fact -- the silent wrong-layer write, rebuilt;
  * two mamba stages of EQUAL SIZE at different global offsets compared EQUAL,
    so the #861c drift check passed them and the wrong-layer write happened
    anyway. The identity was dishonest, and the guard above it could only be as
    good as the identity.

The pool is handed its identity and throws it away. That is the root; slicing
around it would have been the effect.

MAMBA LAYERS ARE NOT A RANGE, which is why the identity is an ID LIST and not a
second integer. In a hybrid checkpoint the mamba layers are a subset of the
global layer numbering, so `start_layer + i` is not their id. `CpuCopyLayout`
therefore carries an optional explicit `layer_ids`, and pools that genuinely are
a contiguous range (every `KVCache`) leave it None and compare exactly as before.

Hermetic: no CUDA, no pool construction, stand-ins throughout.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest

import torch

from sglang.srt.mem_cache.memory_pool import CpuCopyLayout, MambaPool
from sglang.srt.mem_cache.seam_layer_carry import (
    SeamCarryError,
    carry_payload,
    layout_global_ids,
    plan_rank_local_carry,
)
from sglang.test.test_utils import CustomTestCase

# This rig, from the boot log's own server args: 16 attention layers,
# pp_attn_stage_ratio 8/4/4.
TP_KV = CpuCopyLayout(kind="kv", layer_num=16, start_layer=0)
PP0_KV = CpuCopyLayout(kind="kv", layer_num=8, start_layer=0)
PP1_KV = CpuCopyLayout(kind="kv", layer_num=4, start_layer=8)
PP2_KV = CpuCopyLayout(kind="kv", layer_num=4, start_layer=12)


def _kv_payload(start, n):
    """One entry per layer, NAMING the global layer it holds -- so a wrong-layer
    placement is visible as a VALUE and not merely as a length."""
    return [f"L{start + i}" for i in range(n)]


class TestTheMambaIdentityWasHandedOverAndDiscarded(CustomTestCase):
    """THE ROOT. Everything else in this file is downstream of the pool being
    able to say which global layers it holds."""

    def test_the_pool_is_handed_its_global_layer_ids(self):
        """CONTROL. If this ever stops being true the root fix below is about
        something that no longer exists."""
        import inspect

        sig = inspect.signature(MambaPool.__init__)
        self.assertIn("mamba_layer_ids", sig.parameters)

    def test_the_pool_records_the_ids_rather_than_only_their_count(self):
        """RED before the fix: `__init__` computed `num_mamba_layers =
        len(mamba_layer_ids)` and never stored the ids themselves."""
        import inspect

        src = inspect.getsource(MambaPool.__init__)
        self.assertIn(
            "self.mamba_layer_ids",
            src,
            "the pool is handed its global identity and must keep it; a count "
            "cannot distinguish stage 0 from stage 2",
        )

    def test_the_declared_layout_names_the_ids_not_a_defaulted_zero(self):
        """RED. `start_layer` had no setter anywhere in the tree, so every mamba
        layout claimed to start at global 0."""
        stage = types.SimpleNamespace(
            mamba_cache=types.SimpleNamespace(conv=[None] * 3),
            mamba_layer_ids=(12, 16, 20),
        )
        layout = MambaPool.cpu_copy_layout(stage)
        self.assertEqual((12, 16, 20), tuple(layout.layer_ids))

    def test_two_equal_sized_stages_no_longer_compare_EQUAL(self):
        """THE DEFECT THAT PREDATES THIS TICKET. #861c's drift check compares
        layouts for equality; two mamba stages of the same size at different
        global offsets both reported `("mamba", n, 0)` and passed it. The
        wrong-layer write then happened with the guard green."""
        a = MambaPool.cpu_copy_layout(
            types.SimpleNamespace(
                mamba_cache=types.SimpleNamespace(conv=[None] * 3),
                mamba_layer_ids=(0, 4, 8),
            )
        )
        b = MambaPool.cpu_copy_layout(
            types.SimpleNamespace(
                mamba_cache=types.SimpleNamespace(conv=[None] * 3),
                mamba_layer_ids=(12, 16, 20),
            )
        )
        self.assertNotEqual(a, b)

    def test_a_pool_that_cannot_name_its_ids_still_answers_as_before(self):
        """The `getattr` default is load-bearing for pools that have no ids to
        give. They must keep comparing exactly as they did, or this fix breaks
        the families it was not about."""
        anon = MambaPool.cpu_copy_layout(
            types.SimpleNamespace(mamba_cache=types.SimpleNamespace(conv=[None] * 3))
        )
        self.assertIsNone(anon.layer_ids)
        self.assertEqual(3, anon.layer_num)


class TestTheLayoutsAnswerWhichGlobalLayersTheyHold(CustomTestCase):
    def test_a_kv_layout_is_a_contiguous_range(self):
        self.assertEqual([8, 9, 10, 11], list(layout_global_ids(PP1_KV)))

    def test_an_id_list_wins_over_the_range_arithmetic(self):
        """A mamba stage's ids are not `start + i`, and reading them that way is
        the wrong-layer write with extra steps."""
        lay = CpuCopyLayout(
            kind="mamba", layer_num=3, start_layer=0, layer_ids=(12, 16, 20)
        )
        self.assertEqual([12, 16, 20], list(layout_global_ids(lay)))

    def test_an_identity_the_module_does_not_recognise_is_refused(self):
        """THE FUTURE CHECK. A new pool family with a new layout shape must land
        on a refusal, never on a guess. #861c's contract test passes opaque
        tuples for exactly this reason and must keep getting the old behaviour."""
        with self.assertRaises(SeamCarryError):
            layout_global_ids(("kv", 16, 0))


class TestThePlanIsRankLocalOrItIsNothing(CustomTestCase):
    def test_a_tp_copy_covers_every_layer_a_pp_stage_needs(self):
        """THE SILENT DIRECTION, and the whole reason this is buildable without
        a collective: the source is a SUPERSET."""
        plan = plan_rank_local_carry(TP_KV, PP1_KV)
        self.assertEqual([8, 9, 10, 11], list(plan.take))

    def test_a_pp_stage_cannot_fill_a_tp_pool_and_says_which_layers_are_missing(self):
        """THE LOUD DIRECTION. It still refuses -- the data is on a peer, and
        fetching it is the collective `ec1717491f` said DO NOT BUILD."""
        with self.assertRaises(SeamCarryError) as caught:
            plan_rank_local_carry(PP0_KV, TP_KV)
        msg = str(caught.exception)
        self.assertIn("8", msg)
        self.assertIn("15", msg)

    def test_a_same_layout_plan_is_the_identity_and_is_not_special_cased(self):
        plan = plan_rank_local_carry(TP_KV, TP_KV)
        self.assertEqual(list(range(16)), list(plan.take))

    def test_a_kind_change_is_refused_rather_than_carried(self):
        """A KV layout and a mamba layout must never be reconciled, whatever
        their counts. `kind` exists for this."""
        with self.assertRaises(SeamCarryError):
            plan_rank_local_carry(
                TP_KV, CpuCopyLayout(kind="mamba", layer_num=16, start_layer=0)
            )

    def test_the_plan_selects_by_id_when_the_ids_are_not_a_range(self):
        """The mamba case. Source holds global 0,4,8,12; the destination stage
        holds 8,12. Positional slicing from the front would take 0 and 4."""
        src = CpuCopyLayout(
            kind="mamba", layer_num=4, start_layer=0, layer_ids=(0, 4, 8, 12)
        )
        dst = CpuCopyLayout(kind="mamba", layer_num=2, start_layer=0, layer_ids=(8, 12))
        self.assertEqual([2, 3], list(plan_rank_local_carry(src, dst).take))

    def test_the_three_stages_tile_the_model_exactly_once(self):
        """A per-stage assertion cannot see a scheme that double-counts a layer
        or drops one."""
        seen = []
        for dst in (PP0_KV, PP1_KV, PP2_KV):
            seen.extend(plan_rank_local_carry(TP_KV, dst).take)
        self.assertEqual(list(range(16)), seen)


class TestThePayloadIsSlicedByShapeAndNeverGuessed(CustomTestCase):
    def test_a_per_layer_list_is_sliced_to_the_destination(self):
        got = carry_payload(TP_KV, PP1_KV, _kv_payload(0, 16))
        self.assertEqual(["L8", "L9", "L10", "L11"], got)

    def test_the_row_axis_is_untouched(self):
        """The carry moves NOTHING on the row axis -- the extent contract has
        already settled that one, and touching it here would be the second
        mechanism for one question."""
        rows = [[f"L{i}-chunk0", f"L{i}-chunk1"] for i in range(16)]
        got = carry_payload(TP_KV, PP1_KV, rows)
        self.assertEqual([["L8-chunk0", "L8-chunk1"]], got[:1])
        self.assertIs(rows[8], got[0])

    def test_a_payload_whose_length_contradicts_its_layout_is_refused(self):
        with self.assertRaises(SeamCarryError):
            carry_payload(TP_KV, PP1_KV, _kv_payload(0, 15))

    def test_a_composite_hybrid_payload_carries_both_halves(self):
        """`HybridLinearKVPool`'s copy is `(kv_cpu, mamba_cpu)` and its layout is
        the pair. Carrying the KV half and leaving the mamba half is the partial
        restore `SeamCarryError` exists to forbid."""
        src_m = CpuCopyLayout(
            kind="mamba", layer_num=4, start_layer=0, layer_ids=(0, 4, 8, 12)
        )
        dst_m = CpuCopyLayout(
            kind="mamba", layer_num=2, start_layer=0, layer_ids=(8, 12)
        )
        src = ("hybrid", TP_KV, src_m)
        dst = ("hybrid", PP1_KV, dst_m)
        conv = [f"C{i}" for i in (0, 4, 8, 12)]
        temporal = torch.arange(4).reshape(4, 1)
        got_kv, (got_conv, got_temporal) = carry_payload(
            src, dst, (_kv_payload(0, 16), (conv, temporal))
        )
        self.assertEqual(["L8", "L9", "L10", "L11"], got_kv)
        self.assertEqual(["C8", "C12"], got_conv)
        self.assertEqual([2, 3], got_temporal.flatten().tolist())

    def test_a_hybrid_copy_without_a_mamba_half_stays_without_one(self):
        src = ("hybrid", TP_KV, None)
        dst = ("hybrid", PP1_KV, None)
        got_kv, got_mamba = carry_payload(src, dst, (_kv_payload(0, 16), None))
        self.assertEqual(["L8", "L9", "L10", "L11"], got_kv)
        self.assertIsNone(got_mamba)

    def test_the_layout_path_and_the_integer_path_agree(self):
        """ANTI-DRIFT. `carry_across` (the collective-shaped entry point) and
        `carry_payload` (the rank-local one) answer the same question by
        different routes. Two routes to one answer is how two answers appear, so
        the agreement is asserted rather than assumed."""
        from sglang.srt.mem_cache.seam_layer_carry import carry_across

        for dst in (PP0_KV, PP1_KV, PP2_KV, TP_KV):
            src = _kv_payload(0, 16)
            self.assertEqual(
                carry_across(16, 0, src, dst.layer_num, dst.start_layer),
                list(carry_payload(TP_KV, dst, src)),
                f"the two routes disagree for {dst.describe()}",
            )

    def test_a_dict_payload_is_refused_and_named(self):
        """DSA's copy is `{"kv": ..., "index_k": ...}`. It is not on this rig and
        it is NOT guessed at: a shape this module cannot name lands on the
        refusal that was there before it."""
        with self.assertRaises(SeamCarryError) as caught:
            carry_payload(TP_KV, PP1_KV, {"kv": _kv_payload(0, 16)})
        self.assertIn("dict", str(caught.exception))


class _CarryAlloc:
    """A destination pool that is FAITHFUL about what the real one does with a
    list it did not build: `range(self.layer_num)` positionally, whatever the
    copy's own geometry was. No exception in the surplus direction -- that is
    the point."""

    def __init__(self, layout, *, start=None, count=None):
        self.layout = layout
        # Held apart from `layout` so an OPAQUE layout (a tuple this tree cannot
        # parse) can still be handed to a working pool -- which is exactly the
        # case the last test in this file needs.
        self.start = layout.start_layer if start is None else start
        self.count = layout.layer_num if count is None else count
        self.restored = None
        #: Called INSIDE `load_cpu_copy`. The stamp a carried copy carries is
        #: only observable at that instant -- `Req.load_kv_cache` clears it on
        #: the way out -- so a test that reads it afterwards measures the
        #: clearing and not the carry. (That is exactly what the first version
        #: of `test_the_stamp_...` did: mutant M8 removed the assignment and the
        #: test stayed green.)
        self.watch = None

    def cpu_copy_layout(self):
        return self.layout

    def supports_mamba_cpu_copy(self):
        return True

    def get_cpu_copy(self, indices, mamba_indices=None):
        return _kv_payload(self.start, self.count)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        if self.watch is not None:
            self.watch()
        if len(kv_cache_cpu) < self.count:
            raise IndexError("list index out of range")  # the W40 arm
        self.restored = [kv_cache_cpu[i] for i in range(self.count)]


def _req(*, allocated=20, logical_len=21):
    from sglang.srt.managers.schedule_batch import Req

    req = types.SimpleNamespace(
        rid="rid-875d",
        req_pool_idx=0,
        seqlen=logical_len,
        kv_allocated_len=allocated,
        mamba_pool_idx=None,
        mamba_state_cpu=None,
        mamba_state_cpu_layout=None,
        kv_cache_cpu=None,
        kv_cache_cpu_extent=None,
        kv_cache_cpu_layout=None,
    )
    req.offload_kv_cache = types.MethodType(Req.offload_kv_cache, req)
    req.load_kv_cache = types.MethodType(Req.load_kv_cache, req)
    req._mamba_cpu_copy_is_mine = types.MethodType(Req._mamba_cpu_copy_is_mine, req)
    rtp = types.SimpleNamespace(
        req_to_token=torch.zeros((1, 64), dtype=torch.int64),
        mamba_pool=None,
        translate_mamba_indices=lambda ids: ids,
    )
    return req, rtp


class TestTheNaivePositionalRestoreIsTheDefect(CustomTestCase):
    """THE FALSIFIER FOR THE SILENT ARM. There is no exception to assert on, so
    this asserts CONTENT: what the destination ends up holding. A test that only
    checks 'no crash' passes the defect."""

    def test_without_the_carry_the_stage_holds_the_wrong_global_layers(self):
        pp = _CarryAlloc(PP1_KV)  # global 8..11
        pp.load_cpu_copy(_kv_payload(0, 16), None)
        self.assertEqual(
            ["L0", "L1", "L2", "L3"],
            pp.restored,
            "control: this IS the silent wrong-layer write, pinned so the "
            "assertion below is measured against the real defect",
        )


class TestRestoreSeamStateCarriesInsteadOfRefusing(CustomTestCase):
    def test_a_tp_copy_is_CARRIED_into_a_pp_stage(self):
        """RED before the wiring: `restore_seam_state` returned False and
        dropped the copy. The prefix was recomputed on every flip."""
        from sglang.srt.managers import schedule_batch as sb

        tp = _CarryAlloc(TP_KV)
        req, rtp = _req()
        req.offload_kv_cache(rtp, tp)

        pp = _CarryAlloc(PP1_KV)
        before = sb._SEAM_STATE_COUNTS.get("carried", 0)
        self.assertTrue(sb.restore_seam_state(req, rtp, pp))
        self.assertEqual(["L8", "L9", "L10", "L11"], pp.restored)
        self.assertEqual(before + 1, sb._SEAM_STATE_COUNTS.get("carried", 0))

    def test_the_carry_does_not_count_as_a_refusal(self):
        """A carried restore that still increments `refused_layout` would make
        the operator read a working seam as a losing one."""
        from sglang.srt.managers import schedule_batch as sb

        tp = _CarryAlloc(TP_KV)
        req, rtp = _req()
        req.offload_kv_cache(rtp, tp)
        before = sb._SEAM_STATE_COUNTS.get("refused_layout", 0)
        sb.restore_seam_state(req, rtp, _CarryAlloc(PP1_KV))
        self.assertEqual(before, sb._SEAM_STATE_COUNTS.get("refused_layout", 0))

    def test_the_stamp_describes_the_payload_AT_THE_MOMENT_IT_IS_HANDED_OVER(self):
        """A carried copy is no longer the geometry it was taken from, and the
        stamp must say so while the copy still exists.

        MEASURED AT THE HANDOVER, not afterwards. `Req.load_kv_cache` clears
        `kv_cache_cpu_layout` on its way out, so an assertion taken after the
        restore returns measures the CLEARING and passes whether the carry
        rewrote the stamp or not -- which is precisely what the first version of
        this test did, and mutant M8 (drop the rewrite) survived it."""
        from sglang.srt.managers import schedule_batch as sb

        tp = _CarryAlloc(TP_KV)
        req, rtp = _req()
        req.offload_kv_cache(rtp, tp)
        pp = _CarryAlloc(PP1_KV)
        seen = {}
        pp.watch = lambda: seen.setdefault("layout", req.kv_cache_cpu_layout)
        sb.restore_seam_state(req, rtp, pp)
        self.assertEqual(
            PP1_KV,
            seen.get("layout"),
            "the copy handed to the pool is the destination's geometry now; a "
            "stamp still naming the source describes a payload that no longer "
            "exists",
        )

    def test_a_pp_copy_into_a_tp_pool_STILL_refuses(self):
        """The loud arm is not fixed by this and must not appear to be. The
        missing layers are on a peer; fetching them is the collective the ticket
        determined DO NOT BUILD."""
        from sglang.srt.managers import schedule_batch as sb

        pp = _CarryAlloc(PP0_KV)
        req, rtp = _req()
        req.offload_kv_cache(rtp, pp)

        tp = _CarryAlloc(TP_KV)
        before = sb._SEAM_STATE_COUNTS.get("refused_layout", 0)
        self.assertFalse(sb.restore_seam_state(req, rtp, tp))
        self.assertIsNone(tp.restored)
        self.assertEqual(before + 1, sb._SEAM_STATE_COUNTS.get("refused_layout", 0))
        self.assertIsNone(req.kv_cache_cpu)

    def test_the_refusal_now_names_the_layers_it_could_not_get(self):
        """'Refused' without 'which layers' is what sent this ticket round three
        analysis passes."""
        from sglang.srt.managers import schedule_batch as sb

        pp = _CarryAlloc(PP0_KV)
        req, rtp = _req()
        req.offload_kv_cache(rtp, pp)
        with self.assertLogs("sglang.srt.managers.schedule_batch", "WARNING") as logs:
            sb.restore_seam_state(req, rtp, _CarryAlloc(TP_KV))
        blob = "\n".join(logs.output)
        self.assertIn("rid-875d", blob)
        self.assertIn("15", blob)

    def test_an_extent_drift_still_refuses_BEFORE_any_carry_is_attempted(self):
        """ORDER. The row axis is settled first and independently; a carry that
        ran ahead of it would slice layers off a copy whose rows are already
        wrong."""
        from sglang.srt.managers import schedule_batch as sb

        tp = _CarryAlloc(TP_KV)
        req, rtp = _req()
        req.offload_kv_cache(rtp, tp)
        req.seqlen += 4  # the request has advanced since the copy
        pp = _CarryAlloc(PP1_KV)
        before = sb._SEAM_STATE_COUNTS.get("carried", 0)
        self.assertFalse(sb.restore_seam_state(req, rtp, pp))
        self.assertIsNone(pp.restored)
        self.assertEqual(before, sb._SEAM_STATE_COUNTS.get("carried", 0))

    def test_an_opaque_layout_keeps_the_old_refusal_exactly(self):
        """#861c's contract test hands `restore_seam_state` tuples it cannot
        parse, on purpose. Those must land on the refusal, not on a guess --
        this is the regression guard for that file."""
        from sglang.srt.managers import schedule_batch as sb

        class _Opaque(tuple):
            pass

        a = _CarryAlloc(_Opaque(("kv", 16, 0)), start=0, count=16)
        req, rtp = _req()
        req.offload_kv_cache(rtp, a)
        b = _CarryAlloc(_Opaque(("kv", 4, 8)), start=8, count=4)
        self.assertFalse(sb.restore_seam_state(req, rtp, b))
        self.assertIsNone(b.restored)


if __name__ == "__main__":
    unittest.main()
