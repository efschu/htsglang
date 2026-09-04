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
import inspect
import types

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


class G_PresenceIsTheResidentHeadPlusTheSpanReading(CustomTestCase):
    """#1176 (review r4): THE FRAME, CORRECTED -- AND THE r3 PINS INVERTED.

    Round 3 made ``presence = matched + loaded`` and compared it with the WHOLE
    retract stamp. That reads a SPAN-RELATIVE quantity as an absolute one:
    ``matched + loaded == min_completed_tokens`` is an algebraic identity over
    the span the prefetch was REGISTERED for (unified_radix_cache.py:4022 and
    :4140-4145), and that span starts at the match the registration saw --
    everything below it is device-resident and invisible to the record. Boot
    weg1b6 measured the consequence: a rank holding 5966 resident rows with a
    42-token top-up against a 6008 stamp read shortfall 6008-42 and STOPped the
    group on a prefix that was fully present.

    The frame is ``presence = len(req.prefix_indices) + matched + loaded``,
    with ``host_hit_length`` deliberately NOT added (it overlaps the device
    read after ``init_load_back``, schedule_batch.py:3637-3645, and is never
    cleared).

    Four tests below are the r3 pins INVERTED -- they asserted the defect and
    now assert the fix; each is annotated with what it used to claim.
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

    def test_the_resident_head_counts_as_presence(self):
        """INVERTED r3 pin ``test_the_current_match_is_not_netted_off_the
        stamp``, which demanded a RAISE here. This is the weg1b6 metal shape
        scaled up: 10000 resident rows plus a 9984-token top-up against a
        20000 stamp is 19984 present, shortfall 16, well inside one chunk."""
        r = self._req_with_match()
        s = _sched([r], outcomes={r.rid: self._delivered()})
        self.assertEqual(store_witness(s, r), "hit")

    def test_the_admission_site_agrees(self):
        """INVERTED with its sibling: the two sites read ONE function, so the
        admission arm must reach the same verdict on the same record."""
        r = self._req_with_match()
        s = _sched([r])
        assert_store_witness_at_admission(r, self._delivered(), s.tree_cache)

    def test_the_host_hit_half_is_not_credited_either(self):
        """UNCHANGED from r3 and still the load-bearing exclusion.
        ``len(prefix_indices)`` and ``host_hit_length`` OVERLAP once
        ``init_load_back`` has run (schedule_batch.py:3637-3645 states the
        identity; nothing clears ``host_hit_length``), so crediting both reads
        STALE-LARGER. Only the device head is credited: 4000 + 9984 = 13984
        against 20000 is a 6016 shortfall, beyond one chunk."""
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
        msg = str(cm.exception)
        self.assertIn("device_resident=4000", msg)
        self.assertIn("presence=13984", msg)
        self.assertIn(f"shortfall={SPAN_STAMP - 13984}", msg)

    def test_a_presence_short_by_more_than_one_chunk_still_raises(self):
        """The gate still bites: 10000 resident + 1000 loaded = 11000 against
        a 20000 stamp is a 9000 shortfall, twice the one-chunk allowance."""
        r = self._req_with_match()
        out = PrefetchOutcome(1000, hit_tokens=SPAN_DEMAND, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(_sched([r], outcomes={r.rid: out}), r)
        msg = str(cm.exception)
        self.assertIn(f"stamped={SPAN_STAMP}", msg)
        self.assertIn(f"shortfall={SPAN_STAMP - 11_000}", msg)

    def test_without_a_resident_head_the_span_reading_is_the_whole_measure(self):
        """A request holding nothing device-side keeps the stamp as the
        measure -- the conservative reading, unchanged from r3."""
        r = _req(rid="nomatch", stamp=SPAN_STAMP, tokens=SPAN_STAMP)
        out = PrefetchOutcome(SPAN_LOADED, hit_tokens=SPAN_LOADED, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(_sched([r], outcomes={r.rid: out}), r)
        self.assertIn(f"stamped={SPAN_STAMP}", str(cm.exception))

    def test_a_presence_that_covers_the_stamp_owes_nothing(self):
        """The weg1b6 sibling line verbatim (PP1/PP2 matched=5966 loaded=42
        against stamp 6008): a hit, and it must stay one."""
        r = _req(rid="covered", stamp=6008, tokens=6008, prefix_indices=5966)
        out = PrefetchOutcome(42, hit_tokens=6008, probed=True, matched=5966)
        self.assertEqual(store_witness(_sched([r], outcomes={r.rid: out}), r), "hit")

    def test_the_reaped_pp0_record_stays_a_sanctioned_bounded_reprefill(self):
        """The #1176 metal killer, with the resident head deliberately at 0 so
        the arithmetic the docstring names is the arithmetic that runs: PP0 was
        REAPED with matched=3456 loaded=0 against stamp 6008 -- presence 3456,
        shortfall 2552 <= allowance 4096, a SANCTIONED one-chunk re-prefill
        (#939), never a STOP."""
        r = _req(rid="reaped", stamp=6008, tokens=6008)
        out = PrefetchOutcome(0, hit_tokens=3456, probed=True, matched=3456)
        self.assertEqual(store_witness(_sched([r], outcomes={r.rid: out}), r), "hit")

    def test_nothing_present_at_all_raises_even_with_a_host_tier_hit(self):
        """UNCHANGED from r3 and the reason ``presence > 0`` stays INSIDE the
        gate rather than short-circuiting above it: a request whose whole
        credit is ``host_hit_length`` -- a HOST-tier hit that may never have
        reached the device -- has nothing present on this device."""
        r = _req(rid="hostonly", stamp=30_000, tokens=30_000, host_hit_length=30_000)
        out = PrefetchOutcome(0, hit_tokens=40, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(_sched([r], outcomes={r.rid: out}), r)
        msg = str(cm.exception)
        self.assertIn("shows nothing present", msg)
        self.assertIn("presence=0", msg)

    def test_device_residency_that_covers_the_stamp_is_a_hit(self):
        """INVERTED r3 pin ``test_device_residency_alone_never_short_circuits
        the_gate``, which demanded a RAISE when the device head alone covered
        the stamp. That is exactly backwards under the corrected frame: rows
        held in ``prefix_indices`` ARE present, so nothing is recomputed and
        there is nothing for #939 to forbid. The host-tier half above keeps the
        ``presence > 0`` requirement honest."""
        r = _req(rid="deviceonly", stamp=30_000, tokens=30_000,
                 prefix_indices=30_000)
        out = PrefetchOutcome(0, hit_tokens=40, probed=True, matched=0)
        self.assertEqual(store_witness(_sched([r], outcomes={r.rid: out}), r), "hit")


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


# #1176 (review r4): CLASS J WAS DELETED. It pinned the follower->PP0
# CONTRADICTION CARRIER -- "only the authority turns a witness contradiction
# into a group STOP" -- and that carrier is gone. Three tree facts proved it
# could not deliver the STOP it was traded for (one-lap table lifetime at
# scheduler_pp_mixin.py:2315-2317, unreachable on the first admission pass,
# and `seam_transport_premise_holds` counting 'contradiction' as `restored`
# so the reporting rank ADMITTED at P=0 meanwhile). Under
# upstream-minimal-statt-eigenbau a defect inside a compensation layer is a
# DELETION candidate, and this layer licensed the #939 violation it existed to
# stop. Every rank now raises on its OWN true contradiction -- a crash, not a
# state-changing verdict, so #968's PP0 admission authority is untouched and
# raenge-nie-uneins is satisfied. Class H below (in the r4 fixture file
# test_1176_presence_frame_r4.py) pins the replacement.


# The #1157 B1 truncation shape (constants restored here after the class-J
# deletion moved them out of scope, #1176 review r4).
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


# #1176 (review r4): CLASS M WAS DELETED for the same reason as class J. It
# pinned "a follower's contradiction never splits the group silently" against
# the report channel; with the channel deleted the follower does not split the
# group because it RAISES, and the group stops together. The corrected
# presence frame (device residency + matched + loaded) is what makes such a
# raise rare enough to be a true STOP rather than the weg1b6 false alarm.
