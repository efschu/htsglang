"""#1176 (round 4): THE WITNESS FRAME, AND THE DELETED CONTRADICTION CARRIER.

TWO CLASSES, ONE FILE, because they are the two halves of one decision: what
the witness measures, and who is allowed to act on the measurement.

CLASS 1 -- THE ARITHMETIC FRAME (round 3 read a SPAN-RELATIVE quantity as an
ABSOLUTE one).

The prefetch's span is chosen at REGISTRATION from the match this request
already held (scheduler.py:5323 `_matched_len = len(req.prefix_indices) +
req.host_hit_length`, :5350 `full_untruncated_fill_ids[_matched_len:
_match_end]`), and the insert is rooted at `last_host_node`
(unified_radix_cache.py:4022), so `insert_result.prefix_len` -> `matched` is
the part of THAT SPAN the tree already held and `loaded = min_completed_tokens
- prefix_len` (:4140-4145) is the rest of it. `matched + loaded ==
min_completed_tokens` is therefore an ALGEBRAIC IDENTITY OVER THE SPAN, and
comparing it against the WHOLE retract stamp under-reads the prefix by exactly
the registration-time resident. That under-read is a FALSE
StoreWitnessContradiction, and a false contradiction is a PP0 group STOP --
a boot killer, the very shape #1176 was filed for.

The corrected reading adds the device-resident prefix back:

    presence  = len(req.prefix_indices) + matched + loaded
    shortfall = stamp - presence

`host_hit_length` is NEVER added. It is disjoint from `prefix_indices` at the
match (schedule_policy.py:262-278 unpacks the two from ONE match_result) and
OVERLAPS it once `init_load_back` has run (schedule_batch.py:3637-3645 states
the identity verbatim; nothing clears `host_hit_length`), and no field at the
witness site says which regime the request is in. Crediting it would
over-credit in the load-back regime, and over-credit is the #939 direction.

CLASS 2 -- THE CARRIER IS GONE. Round 2+3 built a follower -> PP0 report so a
follower could ADMIT on a contradiction while PP0 raised later. It cannot
deliver, on three independent axes, all three re-verified in the tree before
this file was written:

  (a) ONE-LAP LIFETIME. `pp_note_prefetch_completion` (scheduler_pp_mixin.py
      :2315-2317) wipes the reporting rank's ENTIRE table slice on every lap
      and re-adds only what arrived. The r3 CONTRADICTION exemption lives in
      `pp_prefetch_completion_stamp` (the RELAY), not in the absorber, so
      PP0's own table still loses the entry the moment the follower stops
      reporting -- which is the pass after it admits.
  (b) FIRST-PASS UNREACHABLE. A follower's report can only exist on a LATER
      ring lap. On the first pass PP0's table is empty by construction and
      `_admit_under_group_completion` returns True at `want <= 0`; the rid
      then leaves both queues, so `pp_prefetch_completion_own` never produces
      the report at all.
  (c) THE PREMISE ADMITS ANYWAY. phase_purity.py counted 'contradiction' as
      `restored`, so `seam_transport_premise_holds` returned True on the very
      rank that had just measured one.

Under upstream-minimal-statt-eigenbau a defect INSIDE a compensation layer is
a DELETION candidate. Every rank now raises on its own true contradiction.
That is raenge-nie-uneins-compliant: a raise is a CRASH, not a state-changing
verdict, so it does not touch #968 PP0 admission authority.
"""

import types
import unittest

from sglang.srt.managers import phase_purity as phase_purity_mod
from sglang.srt.managers.phase_purity import (
    SEAM_GRANT_CONSUMED_ATTR,
    SEAM_READMIT_ATTR,
    StoreWitnessContradiction,
    assert_store_witness_at_admission,
    store_witness,
)
from sglang.srt.mem_cache.hicache_storage import PrefetchOutcome
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ALLOWANCE = 4096


def _req(rid, *, stamp, prefix_indices=0, host_hit_length=0, tokens=None):
    r = types.SimpleNamespace(
        rid=rid,
        cached_prompt_tokens_at_retract=stamp,
        cache_protected_len=0,
        origin_input_ids=list(range(int(tokens if tokens is not None else stamp))),
        storage_hit_length=0,
        prefix_indices=list(range(int(prefix_indices))),
        host_hit_length=int(host_hit_length),
    )
    setattr(r, SEAM_READMIT_ATTR, 3)
    setattr(r, SEAM_GRANT_CONSUMED_ATTR, False)
    return r


def _sched(req, outcome, *, allowance=ALLOWANCE):
    pool = types.SimpleNamespace(size=100)
    pool.available_size = lambda: 50
    tree = types.SimpleNamespace(
        root_node=types.SimpleNamespace(children={}),
        cache_controller=types.SimpleNamespace(mem_pool_host=pool),
        enable_storage=True,
        ongoing_prefetch={},
        prefetch_loaded_tokens_by_reqid={req.rid: outcome},
        prefetch_threshold=256,
        _prefetch_chunk_tokens=allowance,
    )
    return types.SimpleNamespace(tree_cache=tree, waiting_queue=[req])


class F1_ATruncatedPrefixSelfInvalidatesAndRaises(CustomTestCase):
    """F1 (round-2 truncation). stamp 80009, registration match 79000, span
    1009, delivered 1008 -- and PP0 then tells told=0, so `truncate_prefix_to`
    empties `prefix_indices` AND the #965 co-derived group before the witness
    reads them. Presence collapses to what the prefetch itself delivered, and
    79001 owed tokens is far past one chunk: RAISE. Crediting the pre-clamp
    residency here would license a 79001-token re-prefill on a stamp."""

    def test_the_emptied_prefix_leaves_only_the_delivered_span(self):
        r = _req("trunc", stamp=80_009, prefix_indices=0, tokens=80_009)
        out = PrefetchOutcome(1008, hit_tokens=1009, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(_sched(r, out), r)
        msg = str(cm.exception)
        self.assertIn("stamped=80009", msg)
        self.assertIn("shortfall=79001", msg)

    def test_truncate_prefix_to_really_clears_both_halves(self):
        """The tree fact the case above rests on: the witness-time read
        self-invalidates because `truncate_prefix_to` slices `prefix_indices`
        AND zeroes `host_hit_length` in the same block."""
        import inspect

        from sglang.srt.managers.schedule_batch import Req

        src = inspect.getsource(Req.truncate_prefix_to)
        self.assertIn("self.prefix_indices = self.prefix_indices[:told]", src)
        self.assertIn("self.host_hit_length = 0", src)


class F2_TheHostHalfIsNeverAddedToResidency(CustomTestCase):
    """F2 (round-2 double count). prefix_indices 15000 + host_hit_length 15000
    AFTER load-back: the real resident is 15000, because
    `len(prefix_indices) = device_original + host_loaded` and nothing clears
    `host_hit_length` (schedule_batch.py:3637-3645). Presence 15000 against
    stamp 30000 -> shortfall 15000 > allowance: RAISE."""

    def test_the_sum_of_both_match_halves_is_not_the_residency(self):
        r = _req("dbl", stamp=30_000, prefix_indices=15_000, host_hit_length=15_000)
        out = PrefetchOutcome(0, hit_tokens=40, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(_sched(r, out), r)
        msg = str(cm.exception)
        self.assertIn("shortfall=15000", msg)
        self.assertIn("device_resident=15000", msg)


class F3_AHostOnlyHitIsNotDevicePresence(CustomTestCase):
    """F3 (round-2 host-only). prefix_indices empty, host_hit_length 30000,
    nothing materialized: presence 0. The `presence > 0` requirement stays
    INSIDE the gate -- there is no short-circuit above it, so a credited host
    hit alone can never buy a pass."""

    def test_it_raises_with_nothing_present(self):
        r = _req("hostonly", stamp=30_000, prefix_indices=0, host_hit_length=30_000)
        out = PrefetchOutcome(0, hit_tokens=40, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(_sched(r, out), r)
        self.assertIn("nothing present", str(cm.exception))


class F4_TheReviewedBreakingInputIsAHit(CustomTestCase):
    """F4 (the round-3 breaking input, re-run on 8e73b2a9cc by the reviewer:
    it RAISED 'shortfall=5966'). stamp 6008, device holds 5966, the prefetch's
    span was the remaining 42 and delivered all of it. The WHOLE prefix is
    present; 8e73b2a9cc read 42 against 6008."""

    def test_the_device_resident_head_plus_the_delivered_span_covers_it(self):
        r = _req("b4", stamp=6008, prefix_indices=5966)
        out = PrefetchOutcome(42, hit_tokens=42, probed=True, matched=0)
        self.assertEqual(store_witness(_sched(r, out), r), "hit")

    def test_sibling_the_span_head_was_already_in_the_tree(self):
        """matched=42 / loaded=0: the tree already held the whole span, the
        prefetch transferred nothing. Same presence, same verdict."""
        r = _req("b4a", stamp=6008, prefix_indices=5966)
        out = PrefetchOutcome(0, hit_tokens=42, probed=True, matched=42)
        self.assertEqual(store_witness(_sched(r, out), r), "hit")

    def test_sibling_the_head_was_reached_through_a_host_hit(self):
        """The host hit has been loaded back, so it lives INSIDE
        prefix_indices and `host_hit_length` still reports it. Adding the two
        would double-count (F2); reading `prefix_indices` alone is exact."""
        r = _req("b4b", stamp=6008, prefix_indices=5966, host_hit_length=5966)
        out = PrefetchOutcome(42, hit_tokens=42, probed=True, matched=0)
        self.assertEqual(store_witness(_sched(r, out), r), "hit")

    def test_the_admission_site_agrees(self):
        r = _req("b4c", stamp=6008, prefix_indices=5966)
        out = PrefetchOutcome(42, hit_tokens=42, probed=True, matched=0)
        assert_store_witness_at_admission(r, out, _sched(r, out).tree_cache)


class F5_TheBoot6FollowerLineStaysAHit(CustomTestCase):
    """F5 (weg1b6 PP1/PP2, verbatim): matched 5966 + loaded 42 vs stamp 6008,
    with the fresh match reporting the span host-side (prefix_indices 0)."""

    def test_hit(self):
        r = _req("b6ok", stamp=6008, prefix_indices=0)
        out = PrefetchOutcome(42, hit_tokens=6008, probed=True, matched=5966)
        self.assertEqual(store_witness(_sched(r, out), r), "hit")


class F6_TheReapedRankIsASanctionedBoundedRePrefill(CustomTestCase):
    """F6 (weg1b6 PP0, the metal killer): REAPED at its 7.87 s budget with
    matched 3456 loaded 0 against stamp 6008 -- shortfall 2552 <= allowance
    4096, the #939-sanctioned one-chunk re-prefill, never a STOP."""

    def test_hit_within_the_one_chunk_allowance(self):
        r = _req("b6reaped", stamp=6008, prefix_indices=0)
        out = PrefetchOutcome(0, hit_tokens=3456, probed=True, matched=3456)
        self.assertEqual(store_witness(_sched(r, out), r), "hit")

    def test_one_token_past_the_allowance_still_raises(self):
        """CAN-FAIL companion: the allowance is a real bound, not a pass."""
        r = _req("b6over", stamp=6008, prefix_indices=0)
        out = PrefetchOutcome(0, hit_tokens=1911, probed=True, matched=1911)
        with self.assertRaises(StoreWitnessContradiction):
            store_witness(_sched(r, out), r)


class G_TheChatHeaderFalseHitStillStopsTheGroup(CustomTestCase):
    """The adversarial case the whole witness exists for, unchanged by the new
    frame: a 40-token chat-template probe answer beside stamp 80009, nothing
    resident, nothing materialized."""

    def test_it_raises(self):
        r = _req("hdr", stamp=80_009, prefix_indices=0)
        out = PrefetchOutcome(0, hit_tokens=40, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(_sched(r, out), r)
        self.assertIn("stamped=80009", str(cm.exception))


class H_EveryRankRaisesOnItsOwnTrueContradiction(CustomTestCase):
    """CLASS 2. The follower/authority split is DELETED: `is_authority`,
    `witness_stop_authority` and `_note_follower_contradiction_deferred` are
    gone, the state 'contradiction' is no longer producible, and a follower
    raises exactly like PP0."""

    def _contradicting(self):
        r = _req("stop", stamp=80_009, prefix_indices=0)
        out = PrefetchOutcome(0, hit_tokens=40, probed=True, matched=0)
        return r, out

    def test_a_pp_follower_raises_too(self):
        r, out = self._contradicting()
        s = _sched(r, out)
        s.ps = types.SimpleNamespace(pp_rank=2, pp_size=3)
        with self.assertRaises(StoreWitnessContradiction):
            store_witness(s, r)

    def test_pp0_raises(self):
        r, out = self._contradicting()
        s = _sched(r, out)
        s.ps = types.SimpleNamespace(pp_rank=0, pp_size=3)
        with self.assertRaises(StoreWitnessContradiction):
            store_witness(s, r)

    def test_the_admission_site_takes_no_authority_argument(self):
        import inspect

        sig = inspect.signature(assert_store_witness_at_admission)
        self.assertNotIn("is_authority", sig.parameters)

    def test_the_authority_helpers_are_gone(self):
        for name in (
            "witness_stop_authority",
            "_note_follower_contradiction_deferred",
        ):
            self.assertFalse(
                hasattr(phase_purity_mod, name),
                f"{name} must not survive the carrier deletion",
            )


class I_TheCarrierSymbolsAreGone(CustomTestCase):
    """The wire half of the deletion: no CONTRADICTION sentinel, no
    peers_reporting_contradiction, no report-side probe."""

    def test_pp_prefetch_completion_has_no_contradiction_sentinel(self):
        from sglang.srt.managers import pp_prefetch_completion as mod

        self.assertFalse(hasattr(mod, "CONTRADICTION"))
        self.assertFalse(hasattr(mod, "peers_reporting_contradiction"))

    def test_the_mixin_has_no_report_probe(self):
        from sglang.srt.managers import scheduler_pp_mixin as mod

        self.assertFalse(hasattr(mod, "_pp_store_witness_contradicts"))
        self.assertFalse(hasattr(mod, "PREFETCH_CONTRADICTION"))

    def test_no_source_file_still_names_the_deleted_carrier(self):
        import pathlib

        root = pathlib.Path(phase_purity_mod.__file__).parents[3]
        hits = []
        for path in root.rglob("*.py"):
            if "/test/" in str(path):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in (
                "PREFETCH_CONTRADICTION",
                "peers_reporting_contradiction",
                "_pp_store_witness_contradicts",
            ):
                if token in text:
                    hits.append(f"{path}:{token}")
        self.assertEqual(hits, [], f"orphan carrier references: {hits}")


if __name__ == "__main__":
    unittest.main()
