"""#1176 -- the store witness measures MATERIALIZED PRESENCE, not the transfer.

THE SPECIMEN (boot weg1b6 @ 06a605fc20, log
/spinning/evidence-665-f1/boot_855_weg1b6_06a605fc20_0903_160236.log:134985-134996).
rid 1e95e023, stamp 6008, allowance 4096. Three ranks re-admitted the same
6008-token prefix after a cutover:

    PP1/PP2  completed_local=6008  matched=5966  loaded=42
    PP0      REAPED at its 7.87 s budget (policy=timeout, queue-lagged behind
             its siblings in the cutover burst -- the mirror of #1175):
             completed_local=3456  matched=3456  loaded=0

`matched + loaded == completed_local` on every one of those lines: `matched`
is the prefix the tree ALREADY HELD on device (`insert_result.prefix_len`) and
`loaded` is what this prefetch transferred on top of it
(`min_completed_tokens - prefix_len`, unified_radix_cache.py:4140-4145). The
store was NOT empty -- sibling ce45cd48 seconds earlier read 5971 tokens out
of it (matched=40 loaded=5971), which is path B working.

THE DEFECT: `_witness_from_outcome` read `loaded` ALONE. PP0's loaded was 0
because its 3456 materialized tokens were all already resident, so the witness
called a request whose prefix WAS present a contradiction and STOPped the
group. Measured presence 3456 against stamp 6008 is a shortfall of 2552, which
is INSIDE the one-chunk allowance of 4096 -- a sanctioned #939 bounded
re-prefill, not a disagreement.

CLASS: the instrument measured the TRANSFER instead of the PRESENCE. The #841
comment at unified_radix_cache.py:4136-4139 names this same class for `loaded`
("Reporting the fetched tail as `loaded` would make the metric measure the
transfer instead of the retention"), and #631c retracted a defect axis built
on `matched` for the same reason in the other direction.

THE CASES (each red on the parent 06a605fc20 unless noted):
 (a) the exact Boot-6 PP0 line   -> "hit"      (RED on parent: raised)
 (b) the exact Boot-6 PP1 line   -> "hit"      (green on parent, kept as a pin)
 (c) CAN-FAIL: the 40-token chat-template header beside stamp 80009 must
     STILL raise (kein-doppel-prefill) -- materialized 40 << 80009.
 (d) nothing materialized beside a stamp        -> raises
 (e) pickle/deepcopy preserve `matched` (the review-B1 lesson: a keyword-only
     ctor survived dumps and died in loads inside the detokenizer)
 (f) Zukunfts-Check: BOTH PrefetchOutcome construction sites in
     unified_radix_cache.py must pass `matched=` -- a future writer that
     forgets it fails a test instead of silently regressing to the transfer.

MUTANTS: m1 `materialized` ignores `matched` -> (a) red; m2 the allowance
comparison removed -> (c) red.
"""

import ast
import copy
import inspect
import pickle
import types
import unittest

from sglang.srt.managers.phase_purity import (
    SEAM_GRANT_CONSUMED_ATTR,
    SEAM_READMIT_ATTR,
    StoreWitnessContradiction,
    assert_store_witness_at_admission,
    store_witness,
    store_witness_census,
)
from sglang.srt.mem_cache import unified_radix_cache as unified_mod
from sglang.srt.mem_cache.hicache_storage import PrefetchOutcome
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# The weg1b6 numbers, verbatim.
B6_STAMP = 6008
B6_ALLOWANCE = 4096  # chunked_prefill_size on that boot
PP0_MATCHED = 3456  # REAPED completed=3456, all of it already resident
PP0_LOADED = 0
PP1_MATCHED = 5966
PP1_LOADED = 42

# The adversarial case that must keep stopping the group.
HEADER_STAMP = 80_009
HEADER_HIT = 40


def _req(rid="1e95e023", *, stamp=B6_STAMP, tokens=B6_STAMP, seam_epoch=3):
    r = types.SimpleNamespace(
        rid=rid,
        cached_prompt_tokens_at_retract=stamp,
        cache_protected_len=0,
        origin_input_ids=list(range(tokens)),
        storage_hit_length=0,
    )
    setattr(r, SEAM_READMIT_ATTR, seam_epoch)
    setattr(r, SEAM_GRANT_CONSUMED_ATTR, False)
    return r


def _sched(reqs, *, outcomes=None, allowance=B6_ALLOWANCE):
    pool = types.SimpleNamespace(size=100)
    pool.available_size = lambda: 50
    tree = types.SimpleNamespace(
        root_node=types.SimpleNamespace(children={}),
        cache_controller=types.SimpleNamespace(mem_pool_host=pool),
        enable_storage=True,
        ongoing_prefetch={},
        prefetch_loaded_tokens_by_reqid=dict(outcomes or {}),
        prefetch_threshold=256,
        _prefetch_chunk_tokens=allowance,
    )
    return types.SimpleNamespace(tree_cache=tree, waiting_queue=list(reqs))


class A_TheBoot6ReapedRankIsABoundedRePrefill(CustomTestCase):
    """(a) THE RED-FIRST CASE. PP0's record: 3456 tokens materialized, none of
    them transferred by this prefetch. On the parent the witness read
    loaded=0 and raised."""

    def _record(self):
        return PrefetchOutcome(
            PP0_LOADED, hit_tokens=B6_STAMP, probed=True, matched=PP0_MATCHED
        )

    def test_the_witness_calls_it_a_hit(self):
        r = _req()
        s = _sched([r], outcomes={r.rid: self._record()})
        self.assertEqual(store_witness(s, r), "hit")

    def test_the_admission_site_does_not_stop_the_group(self):
        r = _req()
        s = _sched([r])
        assert_store_witness_at_admission(r, self._record(), s.tree_cache)

    def test_materialized_is_matched_plus_loaded(self):
        out = self._record()
        self.assertEqual(int(out), PP0_LOADED)
        self.assertEqual(out.matched, PP0_MATCHED)
        self.assertEqual(out.materialized, PP0_MATCHED + PP0_LOADED)
        # completed_local on the emitter line, by the arithmetic at
        # unified_radix_cache.py:4140-4145.
        self.assertEqual(out.materialized, 3456)


class B_TheBoot6FollowerRanksStayHits(CustomTestCase):
    """(b) PP1/PP2: matched=5966 loaded=42, materialized 6008 == the stamp."""

    def test_the_witness_calls_it_a_hit(self):
        r = _req()
        out = PrefetchOutcome(
            PP1_LOADED, hit_tokens=B6_STAMP, probed=True, matched=PP1_MATCHED
        )
        self.assertEqual(out.materialized, B6_STAMP)
        self.assertEqual(store_witness(_sched([r], outcomes={r.rid: out}), r), "hit")


class C_TheChatHeaderFalseHitStillStopsTheGroup(CustomTestCase):
    """(c) CAN-FAIL COMPANION. Every store probe answers the ~40-token chat
    template header. Beside stamp 80009 that is 40 materialized tokens against
    a stamped 80009-token span: a P=0 re-admission would recompute 79969
    tokens on a stamp (kein-doppel-prefill). The allowance term is what stops
    it, and this test is what a mutant removing that term must kill."""

    def test_it_raises_with_the_measured_terms_named(self):
        r = _req(rid="header", stamp=HEADER_STAMP, tokens=HEADER_STAMP)
        out = PrefetchOutcome(0, hit_tokens=HEADER_HIT, probed=True, matched=HEADER_HIT)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(_sched([r], outcomes={r.rid: out}), r)
        msg = str(cm.exception)
        self.assertIn(f"matched={HEADER_HIT}", msg)
        self.assertIn(f"materialized={HEADER_HIT}", msg)
        self.assertIn(f"shortfall={HEADER_STAMP - HEADER_HIT}", msg)
        # #1176 (review b): the demand is now the span the prefetch was
        # asked for; with no span stamp it stays the whole stamped prefix.
        self.assertIn("span=None", msg)
        self.assertIn("fell short of that span by more than one chunk", msg)

    def test_the_admission_site_raises_too(self):
        r = _req(rid="header", stamp=HEADER_STAMP, tokens=HEADER_STAMP)
        s = _sched([r])
        out = PrefetchOutcome(0, hit_tokens=HEADER_HIT, probed=True, matched=HEADER_HIT)
        with self.assertRaises(StoreWitnessContradiction):
            assert_store_witness_at_admission(r, out, s.tree_cache)


class D_NothingMaterializedBesideAStampRaises(CustomTestCase):
    """(d) The probe answered, nothing is resident and nothing was loaded."""

    def test_it_raises_and_says_nothing_materialized(self):
        r = _req()
        out = PrefetchOutcome(0, hit_tokens=0, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(_sched([r], outcomes={r.rid: out}), r)
        msg = str(cm.exception)
        self.assertIn("materialized=0", msg)
        self.assertIn("nothing materialized", msg)

    def test_a_bare_int_record_keeps_the_old_reading(self):
        """A tree that records only the loaded count (hiradix_cache.py:1937,
        hi_mamba_radix_cache.py:2364) has no annotation: the fallback reading
        is unchanged, so those writers stay on the plain path."""
        r = _req()
        self.assertEqual(store_witness(_sched([r], outcomes={r.rid: B6_STAMP}), r), "hit")
        # A bare count far below the stamp is still a contradiction (the count
        # is both the loaded and the presence reading on this record).
        with self.assertRaises(StoreWitnessContradiction):
            store_witness(_sched([r], outcomes={r.rid: HEADER_HIT}), r)
        # A bare ZERO says nothing about whether the store was asked, so it
        # stays 'unprobed' -- unchanged by #1176.
        self.assertEqual(store_witness(_sched([r], outcomes={r.rid: 0}), r), "unprobed")


class E_TheAnnotationSurvivesSerialization(CustomTestCase):
    """(e) The review-B1 lesson: the record rides req.storage_hit_length into
    the pickled detokenizer output, and int subclasses are rebuilt
    POSITIONALLY by copyreg. A keyword-only ctor died in pickle.loads."""

    def test_pickle_round_trip_preserves_matched(self):
        out = PrefetchOutcome(PP1_LOADED, hit_tokens=B6_STAMP, probed=True, matched=PP1_MATCHED)
        back = pickle.loads(pickle.dumps(out))
        self.assertEqual(int(back), PP1_LOADED)
        self.assertEqual(back.hit_tokens, B6_STAMP)
        self.assertTrue(back.probed)
        self.assertEqual(back.matched, PP1_MATCHED)
        self.assertEqual(back.materialized, B6_STAMP)

    def test_deepcopy_preserves_matched(self):
        out = PrefetchOutcome(PP0_LOADED, hit_tokens=B6_STAMP, probed=True, matched=PP0_MATCHED)
        back = copy.deepcopy(out)
        self.assertEqual(back.matched, PP0_MATCHED)
        self.assertEqual(back.materialized, PP0_MATCHED)

    def test_the_constructor_is_positional(self):
        out = PrefetchOutcome(7, 9, True, 11)
        self.assertEqual((int(out), out.hit_tokens, out.probed, out.matched), (7, 9, True, 11))
        self.assertEqual(PrefetchOutcome(7).matched, 0)

    def test_the_repr_names_matched(self):
        self.assertIn("matched=11", repr(PrefetchOutcome(7, 9, True, 11)))


class F_EveryWriterMustPassMatched(CustomTestCase):
    """(f) ZUKUNFTS-CHECK. A future writer that forgets `matched` regresses the
    witness back to measuring the transfer. That must fail a test, not run."""

    def test_both_unified_radix_cache_sites_pass_matched(self):
        src = inspect.getsource(unified_mod)
        tree = ast.parse(src)
        sites = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "PrefetchOutcome"
        ]
        self.assertEqual(
            len(sites),
            2,
            "expected exactly the two known PrefetchOutcome writers in "
            "unified_radix_cache.py (the terminate-prefetch site around :4148 "
            "and the revoke drain around :4617); found "
            f"{[n.lineno for n in sites]}",
        )
        for node in sites:
            named = {kw.arg for kw in node.keywords}
            self.assertIn(
                "matched",
                named,
                f"unified_radix_cache.py:{node.lineno} builds a PrefetchOutcome "
                "without matched= -- the store witness would measure the "
                "TRANSFER again and re-open #1176 (the two sites are the "
                "terminate path near :4148 and the revoke drain near :4617)",
            )


# ---------------------------------------------------------------------------
# #1176 follow-up (adversarial review, 2026-09-03). Three blocking findings on
# 4b277fff254f954725626b888045c96d6c34b4ff, each red on that parent:
#
#  (b) PRESENCE UNDER-COUNTS BY THE REGISTRATION-TIME MATCH. `matched` is
#      `insert_result.prefix_len`, the prefix of the FETCHED SPAN the tree
#      already held -- NOT this request's device-resident prefix. The span
#      itself excludes what was matched at registration
#      (scheduler.py:5322 `_matched_len = len(req.prefix_indices) +
#      req.host_hit_length`, :5350 `_new_input_tokens =
#      full_untruncated_fill_ids[_matched_len:_match_end]`), while the stamp
#      is the WHOLE prompt prefix (schedule_batch.py:2593). So
#      `matched + loaded <= stamp - _matched_len` BY CONSTRUCTION, and the
#      witness raised on a prefix that was 19984/20000 present. Boot 6 hid it
#      because the tree was reset at the cutover (_matched_len 0, span ==
#      stamp); it becomes MORE reachable exactly as path B starts serving,
#      because `host_hit_length` feeds `_matched_len`.
#
#  (f) TWO SURVIVING MUTANTS on the axis the commit is about: nothing
#      separated "the probe ANSWERED" from "the prefix is MATERIALIZED"
#      (MUT-B), and case (d) only ever raised through the shortfall arm, so
#      the "nothing materialized" term was unpinned (MUT-C).
#
#  (d) RANK DIVERGENCE ON THE VERDICT. Under --tp-size 1 --pp-size 3 the
#      packed MIN all_reduce is not taken (unified_radix_cache.py:3879-3907,
#      `if self.tp_world_size > 1`; the boot log reads `synced=no
#      attn_reduce_world=1`), so every rank evaluates its own record. Boot 6
#      MEASURED the split: PP0 raised, PP1/PP2 read "hit". The remedy uses the
#      authority that already exists (#968/#969Z: followers of a PP group
#      credit and decide nothing) instead of building a second channel.
# ---------------------------------------------------------------------------

# The demonstrated (b) case, from the review: a request whose prompt prefix is
# 19984/20000 present after a prefetch that completed in full.
SPAN_STAMP = 20_000
SPAN_MATCHED_AT_REGISTRATION = 10_000
SPAN_TOKENS = 9_999  # _match_end - _matched_len
SPAN_LOADED = 9_984  # what the completed prefetch materialized of that span


def _pp(scheduler, *, pp_rank, pp_size=3):
    """Give a stand-in scheduler a pipeline identity (`self.ps.pp_rank` /
    `pp_size`, the shape scheduler.py reads at :11264 and :11288)."""
    scheduler.ps = types.SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size)
    return scheduler


class G_PresenceIsJudgedAgainstTheSpanThePrefetchWasAskedFor(CustomTestCase):
    """(b) The prefetch is only ever asked for `_new_input_tokens`; the rest of
    the stamped prefix was already matched at registration and is not this
    operation's to deliver. Judging `matched + loaded` against the WHOLE stamp
    charges the operation for tokens nobody asked it to fetch."""

    def _delivered(self):
        return PrefetchOutcome(
            SPAN_LOADED, hit_tokens=SPAN_LOADED, probed=True, matched=0
        )

    def _req_with_span(self):
        r = _req(rid="span", stamp=SPAN_STAMP, tokens=SPAN_STAMP)
        r._prefetch_span_tokens = SPAN_TOKENS
        return r

    def test_a_fully_delivered_span_is_a_hit_not_a_contradiction(self):
        r = self._req_with_span()
        s = _sched([r], outcomes={r.rid: self._delivered()})
        self.assertEqual(store_witness(s, r), "hit")

    def test_the_admission_site_agrees(self):
        r = self._req_with_span()
        s = _sched([r])
        assert_store_witness_at_admission(r, self._delivered(), s.tree_cache)

    def test_a_span_short_by_more_than_one_chunk_still_raises(self):
        """The span narrows the DEMAND; it does not remove the allowance term.
        A prefetch that delivered 1000 of a 9999-token span is still a
        kein-doppel-prefill contradiction."""
        r = self._req_with_span()
        out = PrefetchOutcome(1000, hit_tokens=SPAN_TOKENS, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(_sched([r], outcomes={r.rid: out}), r)
        msg = str(cm.exception)
        self.assertIn(f"demand={SPAN_TOKENS}", msg)
        self.assertIn(f"span={SPAN_TOKENS}", msg)

    def test_without_a_span_stamp_the_whole_prefix_is_the_demand(self):
        """A record whose request carries no span stamp keeps the stamp as the
        demand -- the conservative reading, unchanged."""
        r = _req(rid="nospan", stamp=SPAN_STAMP, tokens=SPAN_STAMP)
        out = PrefetchOutcome(SPAN_LOADED, hit_tokens=SPAN_LOADED, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(_sched([r], outcomes={r.rid: out}), r)
        self.assertIn(f"demand={SPAN_STAMP}", str(cm.exception))

    def test_a_span_wider_than_the_stamp_never_widens_the_demand(self):
        """`_match_end` is derived from full_untruncated_fill_ids, which grows
        with the OUTPUT; the stamp is the prompt prefix. The demand is the
        smaller of the two, so an output-bearing request cannot be charged for
        tokens outside its stamped prefix."""
        r = _req(rid="wide", stamp=4_000, tokens=4_000)
        r._prefetch_span_tokens = 40_000
        out = PrefetchOutcome(4_000, hit_tokens=4_000, probed=True, matched=0)
        self.assertEqual(store_witness(_sched([r], outcomes={r.rid: out}), r), "hit")


class H_TheProbeAnsweringIsNotThePrefixBeingPresent(CustomTestCase):
    """(f) MUT-B. `shortfall = demand - presence` and `demand - hit` coincide
    on every case the suite had, because those cases all had hit == presence.
    This is the input that separates them: the probe answered the FULL span
    and 100 tokens are materialized. Reading the probe's answer as presence
    would admit at P=100 against a 6008-token stamp."""

    def test_a_full_probe_answer_over_an_empty_prefix_still_raises(self):
        r = _req(rid="probe-vs-presence")
        r._prefetch_span_tokens = B6_STAMP
        out = PrefetchOutcome(100, hit_tokens=B6_STAMP, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(_sched([r], outcomes={r.rid: out}), r)
        msg = str(cm.exception)
        self.assertIn("probed_hit=6008", msg)
        self.assertIn("materialized=100", msg)
        self.assertIn(f"shortfall={B6_STAMP - 100}", msg)
        self.assertIn("fell short", msg)


class I_NothingMaterializedIsItsOwnContradiction(CustomTestCase):
    """(f) MUT-C. Case (d) reaches the raise through the SHORTFALL arm, so
    `presence > 0` was never load-bearing there. This input is inside the
    allowance (stamp 100 <= 4096) and reaches the raise ONLY through the
    'nothing materialized' term."""

    def test_a_stamp_inside_the_allowance_with_zero_presence_raises(self):
        r = _req(rid="tiny-stamp", stamp=100, tokens=100)
        r._prefetch_span_tokens = 100
        out = PrefetchOutcome(0, hit_tokens=40, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(_sched([r], outcomes={r.rid: out}), r)
        msg = str(cm.exception)
        self.assertIn("nothing materialized", msg)
        # The shortfall arm cannot be what raised here.
        self.assertIn("shortfall=100", msg)
        self.assertLessEqual(100, B6_ALLOWANCE)


class J_OnlyTheAuthorityTurnsAWitnessContradictionIntoAGroupStop(CustomTestCase):
    """(d) RANK DIVERGENCE. Boot 6 measured it: PP0 raised while PP1/PP2 read
    'hit' off their own records. A follower that dies while its peers admit
    walks the group into a collective with a missing rank -- so the raise
    belongs to the rank that already owns the admission verdict (#969Z: "no
    rank of a PP group withholds admission for its own prefetch; PP0's
    standing verdict is TAKE WITHOUT WAITING"). A follower REPORTS the
    contradiction and returns a state that licenses nothing."""

    def _contradictory(self):
        return PrefetchOutcome(0, hit_tokens=HEADER_HIT, probed=True, matched=HEADER_HIT)

    def _req(self):
        r = _req(rid="divergent", stamp=HEADER_STAMP, tokens=HEADER_STAMP)
        r._prefetch_span_tokens = HEADER_STAMP
        return r

    def test_pp0_still_stops_the_group(self):
        r = self._req()
        s = _pp(_sched([r], outcomes={r.rid: self._contradictory()}), pp_rank=0)
        with self.assertRaises(StoreWitnessContradiction):
            store_witness(s, r)

    def test_a_follower_reports_instead_of_dying(self):
        r = self._req()
        s = _pp(_sched([r], outcomes={r.rid: self._contradictory()}), pp_rank=1)
        self.assertEqual(store_witness(s, r), "contradiction")

    def test_the_follower_state_licenses_nothing(self):
        """'contradiction' must not join the states seam_transport_premise_holds
        accepts -- otherwise the follower would admit on a premise it just
        refuted."""
        self.assertNotIn("contradiction", ("pending", "hit", "bounded"))

    def test_the_follower_admission_site_does_not_raise(self):
        r = self._req()
        s = _sched([r])
        assert_store_witness_at_admission(
            r, self._contradictory(), s.tree_cache, is_authority=False
        )

    def test_a_single_rank_world_is_its_own_authority(self):
        r = self._req()
        s = _pp(_sched([r], outcomes={r.rid: self._contradictory()}), pp_rank=0, pp_size=1)
        with self.assertRaises(StoreWitnessContradiction):
            store_witness(s, r)

    def test_a_stand_in_without_a_pipeline_identity_still_raises(self):
        """Every existing #1157/#1176 stand-in has no `ps` attribute; absence
        of a pipeline identity must not silence the witness."""
        r = self._req()
        s = _sched([r], outcomes={r.rid: self._contradictory()})
        self.assertFalse(hasattr(s, "ps"))
        with self.assertRaises(StoreWitnessContradiction):
            store_witness(s, r)

    def test_the_census_counts_the_follower_state(self):
        """store_witness_census reads the same predicate; a follower's
        contradiction must be COUNTED, not swallowed as UNREADABLE."""
        r = self._req()
        setattr(r, SEAM_READMIT_ATTR, 3)
        s = _pp(_sched([r], outcomes={r.rid: self._contradictory()}), pp_rank=2)
        s.last_seam_readmit_generation = 3
        census = store_witness_census(s)
        self.assertIn("contradiction=1", census)


if __name__ == "__main__":
    unittest.main()
