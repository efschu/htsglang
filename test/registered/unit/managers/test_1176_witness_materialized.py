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

from sglang.srt.managers import phase_purity as phase_purity_mod
from sglang.srt.managers.phase_purity import (
    SEAM_GRANT_CONSUMED_ATTR,
    SEAM_READMIT_ATTR,
    StoreWitnessContradiction,
    assert_store_witness_at_admission,
    seam_transport_premise_holds,
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


def _req(
    rid="1e95e023",
    *,
    stamp=B6_STAMP,
    tokens=B6_STAMP,
    seam_epoch=3,
    prefix_indices=0,
    host_hit_length=0,
):
    """A re-admitted request.

    `prefix_indices` / `host_hit_length` are THIS request's CURRENT match --
    the pair scheduler.py:5322 reads when it decides the prefetch span, and
    the pair the witness reads at witness time (review B1). Zero on both is
    the shape every pre-#1176 stand-in had: nothing matched, so the whole
    stamp is still owed.
    """
    r = types.SimpleNamespace(
        rid=rid,
        cached_prompt_tokens_at_retract=stamp,
        cache_protected_len=0,
        origin_input_ids=list(range(tokens)),
        storage_hit_length=0,
        prefix_indices=list(range(int(prefix_indices))),
        host_hit_length=int(host_hit_length),
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


def _carrier():
    """The #1176 (review B3) carrier symbols, imported LAZILY.

    They do not exist on the parent be3ec1760b. A module-level import would
    turn every case in this file into a collection error there and destroy the
    per-case red-first evidence for findings B1 and B2, which are about code
    that DOES exist on the parent.
    """
    from sglang.srt.managers import pp_prefetch_completion as mod
    from sglang.srt.managers.scheduler_pp_mixin import pp_prefetch_completion_own

    return types.SimpleNamespace(
        CONTRADICTION=mod.CONTRADICTION,
        PENDING=mod.PENDING,
        group_completion_verdict=mod.group_completion_verdict,
        peers_reporting_contradiction=mod.peers_reporting_contradiction,
        own=pp_prefetch_completion_own,
    )


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
        # #1176 (review r3): the presence is compared against the WHOLE
        # stamped prefix; `device_resident` is printed as a diagnostic and is
        # deliberately NOT netted off it (the 1634bc3d28 `demand=` term is
        # deleted -- it double-subtracted this request's own match).
        self.assertIn("device_resident=0", msg)
        self.assertNotIn("demand=", msg)
        self.assertIn("fell short of the stamped prefix by more than one chunk", msg)

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
# 19984/20000 present after a prefetch that completed in full. `RESIDENT` is
# what this request's OWN match holds at witness time -- the pair
# scheduler.py:5322 reads (`len(req.prefix_indices) + req.host_hit_length`)
# when it decides the span, and the pair the witness reads back.
SPAN_STAMP = 20_000
SPAN_RESIDENT = 10_000  # this request's current match
SPAN_DEMAND = SPAN_STAMP - SPAN_RESIDENT  # what the store still owes: 10000
SPAN_LOADED = 9_984  # what the completed prefetch materialized of that demand


def _pp(scheduler, *, pp_rank, pp_size=3):
    """Give a stand-in scheduler a pipeline identity (`self.ps.pp_rank` /
    `pp_size`, the shape scheduler.py reads at :11264 and :11288)."""
    scheduler.ps = types.SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size)
    return scheduler


class G_PresenceIsJudgedAgainstTheWholeStampedPrefix(CustomTestCase):
    """#1176 (review r3): NOTHING IS NETTED OFF THE STAMP.

    1634bc3d28 subtracted `len(prefix_indices) + host_hit_length` from the
    stamp BEFORE subtracting the presence, on the premise that the two describe
    disjoint spans. Three reviewed breaking inputs showed they do not, and each
    of them read "hit" where the parent raised. The witness now compares the
    prefetch's own whole-prefix presence (`matched + loaded`, ONE reader) with
    the whole stamp; these tests are what a re-introduced `resident`/`demand`
    net must kill.
    """

    def _delivered(self):
        return PrefetchOutcome(
            SPAN_LOADED, hit_tokens=SPAN_LOADED, probed=True, matched=0
        )

    def _req_with_match(self):
        return _req(
            rid="span",
            stamp=SPAN_STAMP,
            tokens=SPAN_STAMP,
            prefix_indices=SPAN_RESIDENT,
        )

    def test_the_current_match_is_not_netted_off_the_stamp(self):
        """RED on 1634bc3d28. `matched` (0 here) is the prefetch's own reading
        of the tree match; a request claiming 10000 resident rows alongside
        matched=0 is not a state the runtime produces, and crediting BOTH is
        the double-subtraction reviewed as B4."""
        r = self._req_with_match()
        s = _sched([r], outcomes={r.rid: self._delivered()})
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(s, r)
        msg = str(cm.exception)
        self.assertIn(f"stamped={SPAN_STAMP}", msg)
        self.assertIn(f"shortfall={SPAN_STAMP - SPAN_LOADED}", msg)
        self.assertIn(f"device_resident={SPAN_RESIDENT}", msg)

    def test_the_admission_site_agrees(self):
        r = self._req_with_match()
        s = _sched([r])
        with self.assertRaises(StoreWitnessContradiction):
            assert_store_witness_at_admission(r, self._delivered(), s.tree_cache)

    def test_the_host_hit_half_is_not_credited_either(self):
        """RED on 1634bc3d28 (reviewed as B2). `len(prefix_indices)` and
        `host_hit_length` OVERLAP once `init_load_back` has run
        (schedule_batch.py:3637-3645 states the identity; nothing clears
        `host_hit_length`), so their sum reads STALE-LARGER and covered a
        30000-token stamp with 15000 real tokens. Neither is netted now."""
        r = _req(
            rid="hosthalf",
            stamp=SPAN_STAMP,
            tokens=SPAN_STAMP,
            prefix_indices=4_000,
            host_hit_length=6_000,
        )
        s = _sched([r], outcomes={r.rid: self._delivered()})
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(s, r)
        self.assertIn("device_resident=4000", str(cm.exception))

    def test_a_presence_short_by_more_than_one_chunk_still_raises(self):
        r = self._req_with_match()
        out = PrefetchOutcome(1000, hit_tokens=SPAN_DEMAND, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(_sched([r], outcomes={r.rid: out}), r)
        msg = str(cm.exception)
        self.assertIn(f"stamped={SPAN_STAMP}", msg)
        self.assertIn(f"shortfall={SPAN_STAMP - 1000}", msg)

    def test_without_a_match_the_whole_prefix_is_still_the_measure(self):
        """A request holding nothing keeps the stamp as the measure -- the
        conservative reading, unchanged."""
        r = _req(rid="nomatch", stamp=SPAN_STAMP, tokens=SPAN_STAMP)
        out = PrefetchOutcome(SPAN_LOADED, hit_tokens=SPAN_LOADED, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(_sched([r], outcomes={r.rid: out}), r)
        self.assertIn(f"stamped={SPAN_STAMP}", str(cm.exception))

    def test_a_presence_that_covers_the_stamp_owes_nothing(self):
        """The prefetch's own reading covers the stamped prefix: hit. This is
        the #1176 metal shape (PP1/PP2 matched=5966 loaded=42 vs stamp 6008)
        and it must stay a hit -- the false contradiction stays closed."""
        r = _req(rid="covered", stamp=6008, tokens=6008, prefix_indices=5966)
        out = PrefetchOutcome(42, hit_tokens=6008, probed=True, matched=5966)
        self.assertEqual(store_witness(_sched([r], outcomes={r.rid: out}), r), "hit")

    def test_the_reaped_pp0_record_stays_a_sanctioned_bounded_reprefill(self):
        """The #1176 metal killer: PP0 REAPED with matched=3456 loaded=0
        against stamp 6008 -- shortfall 2552 <= allowance 4096, a SANCTIONED
        one-chunk re-prefill (#939), never a STOP."""
        r = _req(rid="reaped", stamp=6008, tokens=6008, prefix_indices=3456)
        out = PrefetchOutcome(0, hit_tokens=3456, probed=True, matched=3456)
        self.assertEqual(store_witness(_sched([r], outcomes={r.rid: out}), r), "hit")

    def test_nothing_materialized_raises_even_when_residency_covers_the_stamp(self):
        """RED on 1634bc3d28 (reviewed as B3). That commit put a
        `demand <= 0 -> "hit"` short-circuit ABOVE the `presence > 0` test, so a
        request whose CREDITED residency reached the stamp was called a hit
        with nothing materialized at all -- and the credit could come entirely
        from `host_hit_length`, a HOST-tier hit that may never reach the
        device. There is no short-circuit now."""
        r = _req(rid="hostonly", stamp=30_000, tokens=30_000, host_hit_length=30_000)
        out = PrefetchOutcome(0, hit_tokens=40, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(_sched([r], outcomes={r.rid: out}), r)
        self.assertIn("nothing materialized", str(cm.exception))

    def test_device_residency_alone_never_short_circuits_the_gate(self):
        """The other half of the same short-circuit, found by a MUTANT that
        SURVIVED the first cut of this class: the reviewed `demand` netted BOTH
        `prefix_indices` and `host_hit_length` off the stamp, so a credited
        residency from EITHER term alone reached the early `return "hit"`.
        The test above covers the host-tier half; this covers the device half,
        where the prefix indices cover the stamp and the prefetch record still
        says nothing was materialised."""
        r = _req(rid="deviceonly", stamp=30_000, tokens=30_000,
                 prefix_indices=30_000)
        out = PrefetchOutcome(0, hit_tokens=40, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(_sched([r], outcomes={r.rid: out}), r)
        msg = str(cm.exception)
        self.assertIn("nothing materialized", msg)
        self.assertIn("device_resident=30000", msg)


class H_TheProbeAnsweringIsNotThePrefixBeingPresent(CustomTestCase):
    """(f) MUT-B. `shortfall = demand - presence` and `demand - hit` coincide
    on every case the suite had, because those cases all had hit == presence.
    This is the input that separates them: the probe answered the FULL span
    and 100 tokens are materialized. Reading the probe's answer as presence
    would admit at P=100 against a 6008-token stamp."""

    def test_a_full_probe_answer_over_an_empty_prefix_still_raises(self):
        r = _req(rid="probe-vs-presence")
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
        return _req(rid="divergent", stamp=HEADER_STAMP, tokens=HEADER_STAMP)

    def test_pp0_still_stops_the_group(self):
        r = self._req()
        s = _pp(_sched([r], outcomes={r.rid: self._contradictory()}), pp_rank=0)
        with self.assertRaises(StoreWitnessContradiction):
            store_witness(s, r)

    def test_a_follower_reports_instead_of_dying(self):
        r = self._req()
        s = _pp(_sched([r], outcomes={r.rid: self._contradictory()}), pp_rank=1)
        self.assertEqual(store_witness(s, r), "contradiction")

    def test_the_follower_state_is_not_one_of_the_accepted_readings(self):
        """'contradiction' must never join the tuple of states that READ as a
        restore. It is handled by its own named branch instead (review B3,
        class M): the follower counts the candidate, states the divergence and
        leaves the verdict to PP0 -- what it must not do is quietly pass as a
        'hit'."""
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



# ---------------------------------------------------------------------------
# The three blocking findings of the be3ec1760b review, each with the input
# that separates the defect from the fix.
# ---------------------------------------------------------------------------

# The worked breaking input of review B1, verbatim.
TRUNC_STAMP = 80_009
TRUNC_REGISTRATION_MATCH = 79_000
TRUNC_SPAN = TRUNC_STAMP - TRUNC_REGISTRATION_MATCH  # 1009 -> stamped as 1008
TRUNC_DELIVERED = 1_008


class K_ATruncatedMatchRestoresTheWholeDemand(CustomTestCase):
    """(B1) THE LAW-WEAKENING CASE. RED on be3ec1760b, which read the demand
    off `req._prefetch_span_tokens` -- a number stamped ONCE at registration
    (scheduler.py:5355) that NOTHING in the tree ever clears.

    `Req.truncate_prefix_to` (schedule_batch.py:2357, called from
    scheduler.py:11495 and :10975 under the #791/#930 PP-told rule) empties
    `prefix_indices` AND `host_hit_length` and leaves the span standing. The
    premise path re-reads the record non-destructively on every pass
    (phase_purity.py:1631), so the next pass read a STALE-SMALLER span:

        stamp 80009, registration match 79000  -> span stamped 1008
        prefetch delivers matched+loaded = 1008
        PP0 tells told=0 -> truncate_prefix_to(0) -> the prefix is GONE
        parent: demand = min(80009, 1008) = 1008, presence 1008, shortfall 0
                -> "hit" -> seam-transport exemption -> P=0 re-admission
                -> 79001 recomputed tokens licensed by a stamp

    That is the exact kein-doppel-prefill violation the witness exists to
    prevent (#939 licenses at most ONE chunk). Reading the CURRENT match makes
    the record self-invalidating: the demand returns to the full stamp the
    moment the match does -- no new state, no clearer, no lifecycle."""

    def _truncated(self, *, span_field):
        """The record as the witness sees it AFTER the truncation."""
        r = _req(
            rid="truncated",
            stamp=TRUNC_STAMP,
            tokens=TRUNC_STAMP,
            prefix_indices=0,  # truncate_prefix_to(0) emptied it
            host_hit_length=0,  # ... and the #965 co-derived group with it
        )
        if span_field:
            # The stale registration-time stamp the parent trusted.
            r._prefetch_span_tokens = TRUNC_SPAN
        out = PrefetchOutcome(
            0, hit_tokens=TRUNC_DELIVERED, probed=True, matched=TRUNC_DELIVERED
        )
        return r, out

    def test_a_stale_smaller_span_cannot_shrink_the_demand(self):
        """RED-FIRST. On the parent this returned 'hit'."""
        r, out = self._truncated(span_field=True)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(_sched([r], outcomes={r.rid: out}), r)
        msg = str(cm.exception)
        self.assertIn(f"stamped={TRUNC_STAMP}", msg)
        self.assertIn("device_resident=0", msg)
        self.assertIn(f"shortfall={TRUNC_STAMP - TRUNC_DELIVERED}", msg)

    def test_the_admission_site_raises_on_the_same_record(self):
        r, out = self._truncated(span_field=True)
        s = _sched([r])
        with self.assertRaises(StoreWitnessContradiction):
            assert_store_witness_at_admission(r, out, s.tree_cache)

    def test_the_verdict_does_not_depend_on_the_stale_field_at_all(self):
        """Same record without the span stamp: identical verdict. The witness
        reads the live match, so the presence or absence of the retired
        registration snapshot cannot change the reading."""
        r_with, out = self._truncated(span_field=True)
        r_without, _ = self._truncated(span_field=False)
        msgs = []
        for req in (r_with, r_without):
            with self.assertRaises(StoreWitnessContradiction) as cm:
                store_witness(_sched([req], outcomes={req.rid: out}), req)
            msgs.append(str(cm.exception).split("rid=")[1].split(" ", 1)[1])
        self.assertEqual(msgs[0], msgs[1])

    def test_the_witness_module_no_longer_reads_the_registration_snapshot(self):
        """ZUKUNFTS-CHECK. `_prefetch_span_tokens` has ONE writer and NO
        clearer; a future reader of it inside the witness re-opens exactly this
        hole. The field keeps its one legitimate consumer
        (`_apply_prefetch_deferral`, scheduler.py) -- the witness must not
        become a second, lifecycle-blind one."""
        src = inspect.getsource(phase_purity_mod)
        tree = ast.parse(src)
        reads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "_prefetch_span_tokens"
        ]
        self.assertEqual(
            reads,
            [],
            "phase_purity must not read the registration-time span snapshot; "
            "the demand is derived from the CURRENT match (review B1)",
        )


# The input that separates matched+loaded from max(matched, loaded).
SUM_STAMP = 20_000
SUM_MATCHED = 9_000
SUM_LOADED = 9_000


class L_ThePresenceIsTheSumNotTheLarger(CustomTestCase):
    """(B2) THE UNCAUGHT MUTANT. `materialized` is `matched + loaded` -- the
    single arithmetic the whole #1176 fix turns on -- and every case the suite
    had left `max(matched, loaded)` alive: they all have matched==0, or
    loaded==0, or a split whose larger half still lands inside the allowance
    (the Boot-6 followers are 5966/42, and max 5966 is a shortfall of 42).

    This is the input that separates them. Under the sum, 18000 of a
    20000-token demand are present: shortfall 2000, inside the 4096 allowance,
    a sanctioned #939 bounded re-prefill. Under max(), presence reads 9000 and
    the witness raises -- the false contradiction that killed weg1b6, rebuilt
    from a different direction."""

    def _split(self):
        return PrefetchOutcome(
            SUM_LOADED, hit_tokens=SUM_STAMP, probed=True, matched=SUM_MATCHED
        )

    def test_materialized_is_the_sum_of_both_halves(self):
        out = self._split()
        self.assertEqual(out.materialized, SUM_MATCHED + SUM_LOADED)
        self.assertNotEqual(out.materialized, max(SUM_MATCHED, SUM_LOADED))

    def test_the_witness_calls_the_split_record_a_hit(self):
        r = _req(rid="sum-vs-max", stamp=SUM_STAMP, tokens=SUM_STAMP)
        out = self._split()
        self.assertEqual(store_witness(_sched([r], outcomes={r.rid: out}), r), "hit")
        # The two readings must not merely differ -- they must fall on OPPOSITE
        # sides of the allowance, or this input separates nothing.
        self.assertLessEqual(SUM_STAMP - out.materialized, B6_ALLOWANCE)
        self.assertGreater(SUM_STAMP - max(SUM_MATCHED, SUM_LOADED), B6_ALLOWANCE)

    def test_the_admission_site_agrees_on_the_split_record(self):
        r = _req(rid="sum-vs-max", stamp=SUM_STAMP, tokens=SUM_STAMP)
        s = _sched([r])
        assert_store_witness_at_admission(r, self._split(), s.tree_cache)

    def test_a_record_without_the_property_still_sums_both_halves(self):
        """The witness reads `getattr(outcome, 'materialized', matched +
        loaded)`. A duck-typed record from another writer has no property, and
        the DEFAULT must be the sum too -- otherwise the fallback path carries
        the mutant the annotated path rejects."""

        class _BareRecord(int):
            hit_tokens = SUM_STAMP
            probed = True
            matched = SUM_MATCHED

        r = _req(rid="ducktyped", stamp=SUM_STAMP, tokens=SUM_STAMP)
        out = _BareRecord(SUM_LOADED)
        self.assertFalse(hasattr(out, "materialized"))
        self.assertEqual(store_witness(_sched([r], outcomes={r.rid: out}), r), "hit")


class M_AFollowersContradictionNeverSplitsTheGroupSilently(CustomTestCase):
    """(B3) THE SILENT RANK DIVERGENCE. be3ec1760b made a follower stop RAISING
    (correct: #968/#969Z, a PP follower holds no admission verdict) but left
    the returned state outside the tuple `seam_transport_premise_holds`
    accepts -- so the follower still WITHHELD the seam premise. That boolean
    gates the WHOLE prefill-batch build one level up (`prefill_blocked_here`,
    phase_purity.py:1009 -> scheduler.py:8926 `new_batch = None`), and
    store_witness reads RANK-LOCAL records: under `--tp-size 1 --pp-size 3` the
    packed MIN all_reduce (unified_radix_cache.py:3879-3907) is never taken.

    THE EXACT INPUT, from weg1b6: stamp 6008 on both ranks. PP0 measured
    matched=5966/loaded=42 -> hit -> premise True -> PP0 builds a TP prefill
    batch. PP1, reaped early, measured matched=100/loaded=0 -> contradiction ->
    restored 0 -> premise False -> PP1 builds none. Mismatched collectives.

    raenge-nie-uneins forbids that trade: a LOUD stop was swapped for a SILENT
    split, which is the worse of the two. The follower therefore COUNTS the
    candidate (taking PP0's standing verdict, #969Z) and REPORTS the
    contradiction on the EXISTING follower -> PP0 completion carrier; PP0
    raises on the report, once, loudly, naming the peer."""

    STAMP = 6008
    PP0_MATCHED = 5966
    PP0_LOADED = 42
    FOLLOWER_MATCHED = 100

    def _authority_record(self):
        return PrefetchOutcome(
            self.PP0_LOADED, hit_tokens=self.STAMP, probed=True, matched=self.PP0_MATCHED
        )

    def _follower_record(self):
        return PrefetchOutcome(
            0, hit_tokens=self.STAMP, probed=True, matched=self.FOLLOWER_MATCHED
        )

    def _world(self, pp_rank):
        r = _req(rid="b3", stamp=self.STAMP, tokens=self.STAMP)
        out = self._authority_record() if pp_rank == 0 else self._follower_record()
        s = _pp(_sched([r], outcomes={r.rid: out}), pp_rank=pp_rank)
        s.phase_policy_cfg = None
        s.last_seam_readmit_generation = 3
        return s, r

    def test_the_two_ranks_really_do_read_different_states(self):
        """The premise of the finding: without this asymmetry there is no
        divergence to close."""
        s0, r0 = self._world(0)
        s1, r1 = self._world(1)
        self.assertEqual(store_witness(s0, r0), "hit")
        self.assertEqual(store_witness(s1, r1), "contradiction")

    def test_the_authority_holds_the_premise(self):
        s, _ = self._world(0)
        self.assertTrue(seam_transport_premise_holds(s))

    def test_the_follower_holds_the_SAME_premise(self):
        """RED-FIRST. On the parent this returned False while PP0 returned
        True -- a rank-uniform gate answered two ways."""
        s, _ = self._world(1)
        self.assertTrue(seam_transport_premise_holds(s))

    def test_the_follower_reports_the_contradiction_on_the_carrier(self):
        """The stop is not lost, it MOVES. The follower's per-rid report on the
        #1175 completion lap carries CONTRADICTION instead of a token count --
        a count could not carry the fact, because PP0 cannot compute a peer's
        match (`prefix_indices`/`host_hit_length` are rank-local)."""
        s, r = self._world(1)
        s.tree_cache.completed_prefetch_tokens = lambda rid: self.FOLLOWER_MATCHED
        s.tree_cache.prefetch_is_ongoing = lambda rid: False
        c = _carrier()
        reports = c.own(s)
        self.assertEqual(reports, ((r.rid, c.CONTRADICTION, 1),))

    def test_pp0_is_the_only_rank_that_reports_nothing(self):
        """PP0 is the CONSUMER of this fact; a self-report would make the wire
        disagree with the verdict, whose peer set excludes the decider."""
        s, _ = self._world(0)
        s.tree_cache.completed_prefetch_tokens = lambda rid: self.PP0_MATCHED
        s.tree_cache.prefetch_is_ongoing = lambda rid: False
        self.assertEqual(_carrier().own(s), ())

    def test_the_helper_names_the_contradicting_peers(self):
        c = _carrier()
        table = {("b3", 1): c.CONTRADICTION, ("b3", 2): 4096}
        self.assertEqual(c.peers_reporting_contradiction(table, "b3", 3), (1,))
        self.assertEqual(c.peers_reporting_contradiction(table, "other", 3), ())
        # A single-rank world has no peers and therefore no reports.
        self.assertEqual(c.peers_reporting_contradiction(table, "b3", 1), ())

    def test_pp0_raises_on_a_peer_report_even_with_no_span_of_its_own(self):
        """The dangerous combination is exactly 'PP0 fetched nothing while a
        peer measured a contradiction': the `want <= 0` early-out would swallow
        the only rank that saw the problem, so the report is read BEFORE it."""
        from sglang.srt.managers.scheduler import Scheduler

        holder = types.SimpleNamespace(
            ps=types.SimpleNamespace(pp_rank=0, pp_size=3),
            tree_cache=types.SimpleNamespace(completed_prefetch_tokens=lambda rid: 0),
            _pp_prefetch_completion={("b3", 1): _carrier().CONTRADICTION},
            _pp_group_completion_since={},
        )
        req = types.SimpleNamespace(rid="b3")
        with self.assertRaises(StoreWitnessContradiction) as cm:
            Scheduler._admit_under_group_completion(holder, req, lambda *a, **k: None)
        msg = str(cm.exception)
        self.assertIn("rid=b3", msg)
        self.assertIn("[1]", msg)  # the peer is NAMED
        self.assertIn("kein-doppel-prefill", msg)
        self.assertIn("raenge-nie-uneins", msg)

    def test_a_clean_group_still_admits(self):
        """CAN-FAIL companion: with no peer reporting a contradiction the gate
        must behave exactly as before (want <= 0 -> admit)."""
        from sglang.srt.managers.scheduler import Scheduler

        holder = types.SimpleNamespace(
            ps=types.SimpleNamespace(pp_rank=0, pp_size=3),
            tree_cache=types.SimpleNamespace(completed_prefetch_tokens=lambda rid: 0),
            _pp_prefetch_completion={("b3", 1): 4096},
            _pp_group_completion_since={},
        )
        req = types.SimpleNamespace(rid="b3")
        self.assertTrue(
            Scheduler._admit_under_group_completion(holder, req, lambda *a, **k: None)
        )

    def test_the_verdict_helper_never_arithmetics_the_sentinel(self):
        """A kill-switched or stand-in caller must degrade to 'this peer
        produced no usable reading', never die in int("contradiction")."""
        c = _carrier()
        table = {("b3", 1): c.CONTRADICTION, ("b3", 2): c.PENDING}
        verdict = c.group_completion_verdict(table, "b3", 4096, 3)
        self.assertFalse(verdict.admit)
        self.assertIn(1, verdict.missing)
        self.assertIn((1, c.CONTRADICTION), verdict.reports)

if __name__ == "__main__":
    unittest.main()
