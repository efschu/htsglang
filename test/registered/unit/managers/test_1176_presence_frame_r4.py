"""#1176 (round 5): THE FRAME THE SPAN WAS REGISTERED AGAINST, AND THE ONE
RANK THAT MAY STOP THE GROUP.

TWO CLASSES, ONE FILE, because they are the two halves of one decision: what
the witness measures, and who is allowed to act on the measurement. Both
halves were WRONG in round 4 and both were measured wrong on metal.

CLASS 1 -- THE ARITHMETIC FRAME.

The prefetch's span is chosen at REGISTRATION out of the match the request
held at that moment (scheduler.py:5321 `_matched_len = len(req.prefix_indices)
+ req.host_hit_length`, :5349 `full_untruncated_fill_ids[_matched_len:
_match_end]`), and the insert is rooted at that same walk's `last_host_node`
(unified_radix_cache.py:4025), so `insert_result.prefix_len` -> `matched` is
the part of THAT SPAN the tree already held and `loaded = min_completed_tokens
- prefix_len` (:4140-4145) is the rest of it. `matched + loaded` is therefore
an identity OVER THE SPAN, and the head below the span is `_matched_len` --
DEVICE **AND** HOST.

Round 3 compared the span reading against the whole stamp and under-read by
the entire head (false STOP). Round 4 added `len(prefix_indices)` and got two
new defects at once: it still dropped the HOST half (stamp 6008,
prefix_indices 0, host_hit_length 6008 RAISED "shortfall=6008" for a prefix
that was entirely present -- the boot-killer), and where the device tree had
grown into the registered span it counted the same tokens twice, which
UNDER-reports the shortfall and licenses exactly the recompute #939 forbids
(prefix_indices 40000 + matched 40000 against stamp 80009 returned 'hit' with
40000 tokens present).

The frame that has neither defect reads the head from the REGISTRATION:

    head      = min(registered_head, len(prefix_indices) + host_hit_length)
    presence  = max(len(prefix_indices), head + matched + loaded)

`registered_head` is stamped by the same registration that produced the record
(scheduler.py:5377), before any load-back, so it never double counts the host
half against the device half. The `min` is a CAP, never a credit: a WITHDRAWN
match (`truncate_prefix_to` slices `prefix_indices` and zeroes
`host_hit_length` in one block) collapses the head, so a stale stamp cannot
claim a prefix that is gone. The outer `max` carries the second independent
lower bound -- device rows held right now -- for the case where the span was
registered at head 0 and the device tree grew into it afterwards.

CLASS 2 -- ONE RANK OWNS THE STOP.

Every input to the reading is rank-LOCAL on the shipping form (--tp-size 1
--pp-size 3): the packed MIN all_reduce that would make `matched`/`loaded`
uniform runs only under tp_world_size > 1, `prefix_indices` is this rank's own
match, and `host_hit_length` is documented as NOT rank-uniform under a
layer-partitioned host tier (schedule_policy.py:2244-2246). Round 4 let EVERY
rank raise on its own reading; weg1b6 measured the consequence on ONE rid --
PP0 raised while PP1/PP2 read 'hit' and admitted, i.e. one rank dead inside a
collective its peers had entered. `witness_stop_authority` restores the single
verdict point (#968: PP0 decides, followers credit and decide nothing, #969Z),
and the census/premise reader never raises at all -- it returns
"contradiction", which no caller counts as restored, so the premise refuses
instead of crashing one rank inside a group-uniform gate (scheduler.py:8926).

The follower -> PP0 report CARRIER stays deleted; three tree facts proved it
could not deliver a STOP. A follower does not report and does not stop; it
admits, exactly as it does for every other prefetch fact.
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


def _req(
    rid,
    *,
    stamp,
    prefix_indices=0,
    host_hit_length=0,
    tokens=None,
    registered=None,
):
    r = types.SimpleNamespace(
        rid=rid,
        cached_prompt_tokens_at_retract=stamp,
        cache_protected_len=0,
        origin_input_ids=list(range(int(tokens if tokens is not None else stamp))),
        storage_hit_length=0,
        prefix_indices=list(range(int(prefix_indices))),
        host_hit_length=int(host_hit_length),
    )
    if registered is not None:
        r._prefetch_registered_prefix_len = int(registered)
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


def _admit(req, outcome, *, may_stop=True, allowance=ALLOWANCE):
    """The ONLY path that may raise: the admission assert on the authority."""
    assert_store_witness_at_admission(
        req, outcome, _sched(req, outcome, allowance=allowance).tree_cache,
        may_stop=may_stop,
    )


class F1_ATruncatedPrefixSelfInvalidatesAndRaises(CustomTestCase):
    """F1 (round-2 truncation, PRESERVED under the new frame). stamp 80009,
    registration head 79000, span 1009, delivered 1008 -- and PP0 then tells
    told=0, so `truncate_prefix_to` empties `prefix_indices` AND zeroes
    `host_hit_length` before the witness reads them. The registration stamp is
    NOT cleared by that block, which is exactly why it is a CAP: the current
    match sum is 0, so the head collapses to 0 and presence falls back to what
    the prefetch itself delivered. 79001 owed tokens is far past one chunk."""

    def test_the_withdrawn_match_caps_the_registration_head(self):
        r = _req("trunc", stamp=80_009, prefix_indices=0, tokens=80_009, registered=79_000)
        out = PrefetchOutcome(1008, hit_tokens=1009, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            _admit(r, out)
        msg = str(cm.exception)
        self.assertIn("stamped=80009", msg)
        self.assertIn("shortfall=79001", msg)
        self.assertIn("registered_head=0", msg)

    def test_truncate_prefix_to_really_clears_both_halves(self):
        """The tree fact the cap rests on: `truncate_prefix_to` slices
        `prefix_indices` AND zeroes `host_hit_length` in the same block, so the
        cap `len(prefix_indices) + host_hit_length` really does collapse."""
        import inspect

        from sglang.srt.managers.schedule_batch import Req

        src = inspect.getsource(Req.truncate_prefix_to)
        self.assertIn("self.prefix_indices = self.prefix_indices[:told]", src)
        self.assertIn("self.host_hit_length = 0", src)


class F2_TheHostHalfIsCreditedAtTheRegistrationHead(CustomTestCase):
    """F2 (INVERTED from round 4). Round 4 asserted that stamp 30000 with
    prefix_indices 15000 + host_hit_length 15000 must RAISE 'shortfall=15000'
    because the host half is never credited. That was the boot-killer: at the
    witness site the two halves are DISJOINT (the same site round 4's own
    docstring proved fires before `init_load_back`), so the registration head
    is 30000 and the whole stamped prefix is present."""

    def test_both_match_halves_are_the_registered_head(self):
        r = _req(
            "dbl",
            stamp=30_000,
            prefix_indices=15_000,
            host_hit_length=15_000,
            registered=30_000,
        )
        out = PrefetchOutcome(0, hit_tokens=40, probed=True, matched=0)
        self.assertEqual(store_witness(_sched(r, out), r), "hit")

    def test_the_cap_still_binds_when_the_match_shrank(self):
        """CAN-FAIL companion: the same registration head against a match that
        has since shrunk to the device half alone still credits only what the
        current match holds -- 15000 against stamp 30000 is a real shortfall."""
        r = _req(
            "dbl-shrunk",
            stamp=30_000,
            prefix_indices=15_000,
            host_hit_length=0,
            registered=30_000,
        )
        out = PrefetchOutcome(0, hit_tokens=40, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            _admit(r, out)
        self.assertIn("shortfall=15000", str(cm.exception))


class F3_AHostOnlyHitIsPresence(CustomTestCase):
    """F3 (INVERTED from round 4). Round 4 asserted that prefix_indices empty
    with host_hit_length 30000 shows 'nothing present' and must RAISE. That is
    the exact shape the mission produces after a cutover -- the prefix lives in
    the HOST tier, the device match is 0 -- and it was the weg1b6 boot killer.
    A host hit is presence: `init_load_back` brings those tokens onto the
    device in `add_one_req`, so nothing about them is recomputed."""

    def test_a_host_only_head_covers_the_stamp(self):
        r = _req(
            "hostonly",
            stamp=30_000,
            prefix_indices=0,
            host_hit_length=30_000,
            registered=30_000,
        )
        out = PrefetchOutcome(0, hit_tokens=40, probed=True, matched=0)
        self.assertEqual(store_witness(_sched(r, out), r), "hit")

    def test_nothing_present_at_all_still_raises(self):
        """CAN-FAIL companion: the `presence > 0` requirement is untouched.
        No head, no span, a 40-token chat-template probe answer."""
        r = _req("empty", stamp=30_000, prefix_indices=0, registered=0)
        out = PrefetchOutcome(0, hit_tokens=40, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            _admit(r, out)
        self.assertIn("nothing present", str(cm.exception))


class F4_TheOverlapIsNeverDoubleCounted(CustomTestCase):
    """F4 (the round-4 LICENSING defect). The span was registered at head 0 --
    it covers the whole prompt -- and the device tree then grew 40000 tokens
    INTO that span, which the insert reports as `matched`. Round 4 computed
    `len(prefix_indices) + matched + loaded` = 80000 against stamp 80009 and
    returned 'hit', licensing a ~40000-token recompute on a stamp. The two
    terms describe THE SAME TOKENS; the honest reading is the larger of the two
    bounds, not their sum."""

    def test_the_grown_device_match_is_not_added_to_the_span(self):
        r = _req("overlap", stamp=80_009, prefix_indices=40_000, registered=0)
        out = PrefetchOutcome(0, hit_tokens=80_009, probed=True, matched=40_000)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            _admit(r, out)
        msg = str(cm.exception)
        self.assertIn("shortfall=40009", msg)
        self.assertIn("presence=40000", msg)

    def test_the_same_shape_inside_the_allowance_is_a_hit(self):
        """CAN-FAIL companion: the refusal above is the ALLOWANCE biting, not a
        blanket refusal of the overlap shape."""
        r = _req("overlap-ok", stamp=42_000, prefix_indices=40_000, registered=0)
        out = PrefetchOutcome(0, hit_tokens=42_000, probed=True, matched=40_000)
        self.assertEqual(store_witness(_sched(r, out), r), "hit")


class F5_TheBoot6LinesAreAllHits(CustomTestCase):
    """The weg1b6 records verbatim, all three ranks, one rid, stamp 6008."""

    def test_pp1_pp2_matched_5966_loaded_42(self):
        r = _req("b6ok", stamp=6008, prefix_indices=5966, tokens=6009, registered=0)
        out = PrefetchOutcome(42, hit_tokens=6008, probed=True, matched=5966)
        self.assertEqual(store_witness(_sched(r, out), r), "hit")

    def test_pp0_reaped_matched_3456_is_a_sanctioned_bounded_reprefill(self):
        """PP0 REAPED at its 7.87 s budget: matched 3456, loaded 0, shortfall
        2552 <= allowance 4096 -- the #939-sanctioned one-chunk re-prefill."""
        r = _req("b6reaped", stamp=6008, prefix_indices=0, tokens=6009, registered=0)
        out = PrefetchOutcome(0, hit_tokens=3456, probed=True, matched=3456)
        self.assertEqual(store_witness(_sched(r, out), r), "hit")

    def test_the_metal_killer_input_is_a_hit(self):
        """The line that actually killed weg1b6: stamp 6008, device match 0,
        the whole prefix in the host tier, nothing materialized by the
        prefetch. Round 4 raised 'shortfall=6008' here."""
        r = _req(
            "b6killer",
            stamp=6008,
            prefix_indices=0,
            host_hit_length=6008,
            tokens=6009,
            registered=6008,
        )
        out = PrefetchOutcome(0, hit_tokens=6008, probed=True, matched=0)
        self.assertEqual(store_witness(_sched(r, out), r), "hit")

    def test_one_token_past_the_allowance_still_raises(self):
        """CAN-FAIL companion: the allowance is a real bound, not a pass."""
        r = _req("b6over", stamp=6008, prefix_indices=0, tokens=6009, registered=0)
        out = PrefetchOutcome(0, hit_tokens=1911, probed=True, matched=1911)
        with self.assertRaises(StoreWitnessContradiction):
            _admit(r, out)


class G_TheChatHeaderFalseHitStillStopsTheGroup(CustomTestCase):
    """The adversarial case the whole witness exists for, unchanged by the new
    frame: a 40-token chat-template probe answer beside stamp 80009, nothing
    registered, nothing resident, nothing materialized."""

    def test_it_raises(self):
        r = _req("hdr", stamp=80_009, prefix_indices=0, registered=0)
        out = PrefetchOutcome(0, hit_tokens=40, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            _admit(r, out)
        self.assertIn("stamped=80009", str(cm.exception))


class H_OnlyTheAuthorityTurnsAContradictionIntoASTOP(CustomTestCase):
    """CLASS 2. Round 4 deleted the authority split and let every rank raise on
    a rank-local reading. Restored: exactly one rank, and exactly one call
    site, may raise."""

    def _contradicting(self):
        r = _req("stop", stamp=80_009, prefix_indices=0, registered=0)
        out = PrefetchOutcome(0, hit_tokens=40, probed=True, matched=0)
        return r, out

    def test_pp0_is_the_authority_and_followers_are_not(self):
        witness_stop_authority = self._authority()
        for rank, expected in ((0, True), (1, False), (2, False)):
            s = types.SimpleNamespace(
                ps=types.SimpleNamespace(pp_rank=rank, pp_size=3)
            )
            self.assertIs(witness_stop_authority(s), expected, f"pp_rank={rank}")

    def _authority(self):
        """Resolved through the module, not from-imported: on a tree without
        the helper this fails ONE assertion instead of aborting collection for
        the whole file, so the arithmetic classes still report."""
        fn = getattr(phase_purity_mod, "witness_stop_authority", None)
        self.assertIsNotNone(fn, "witness_stop_authority must exist")
        return fn

    def test_a_world_without_a_pipeline_is_its_own_authority(self):
        witness_stop_authority = self._authority()
        s = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0, pp_size=1))
        self.assertIs(witness_stop_authority(s), True)
        self.assertIs(witness_stop_authority(types.SimpleNamespace()), True)

    def test_the_authority_raises_at_admission(self):
        r, out = self._contradicting()
        with self.assertRaises(StoreWitnessContradiction):
            _admit(r, out, may_stop=True)

    def test_a_follower_reports_and_admits(self):
        r, out = self._contradicting()
        _admit(r, out, may_stop=False)  # must not raise

    def test_the_admission_site_takes_the_authority_argument(self):
        import inspect

        sig = inspect.signature(assert_store_witness_at_admission)
        self.assertIn("may_stop", sig.parameters)
        self.assertTrue(sig.parameters["may_stop"].kind.name == "KEYWORD_ONLY")

    def test_both_admission_call_sites_pass_the_resolved_authority(self):
        """The two sites must resolve the authority the same way or they drift.
        Neither may hardcode True."""
        import inspect

        from sglang.srt.managers import scheduler as scheduler_mod

        src = inspect.getsource(scheduler_mod)
        calls = src.count("assert_store_witness_at_admission(")
        self.assertEqual(calls, 2, "expected exactly two admission call sites")
        self.assertEqual(
            src.count("may_stop=witness_stop_authority(self)"),
            2,
            "both sites must resolve the authority through the helper",
        )


class I_TheCensusAndThePremiseNeverRaise(CustomTestCase):
    """`store_witness` is consulted by the census ('Reported only: this NEVER
    decides') and by `seam_transport_premise_holds`, which catches only
    (TypeError, ValueError) -- a raise from there escaped into the
    group-uniform gate at scheduler.py:8926. It returns 'contradiction'
    instead, which is not in the restored set."""

    def test_store_witness_returns_contradiction_on_every_rank(self):
        for rank in (0, 1, 2):
            r = _req(f"c{rank}", stamp=80_009, prefix_indices=0, registered=0)
            out = PrefetchOutcome(0, hit_tokens=40, probed=True, matched=0)
            s = _sched(r, out)
            s.ps = types.SimpleNamespace(pp_rank=rank, pp_size=3)
            self.assertEqual(store_witness(s, r), "contradiction", f"rank {rank}")

    def test_contradiction_is_not_in_the_restored_set(self):
        import inspect

        src = inspect.getsource(phase_purity_mod.seam_transport_premise_holds)
        self.assertIn('_state in ("pending", "hit", "bounded")', src)
        self.assertNotIn('"contradiction"', src.split("_state in")[1][:120])

    def test_unprobed_is_not_restored_either(self):
        """MUT-F coverage gap (review, non-blocking): nothing pinned that
        'unprobed' -- the operation terminated before the probe ran -- is
        excluded from `restored`. Counting it would re-open the #1157 hole
        (boot weg1b3 rid 679e4568: P=0 admission and 6 recomputed TP chunks on
        a premise that had never been probed)."""
        import inspect

        src = inspect.getsource(phase_purity_mod.seam_transport_premise_holds)
        head = src.split("_state in")[1][:120]
        self.assertNotIn("unprobed", head)


class J_TheUnprobedDiscriminatorIsPinned(CustomTestCase):
    """MUT-6 coverage gap (review, non-blocking): nothing pinned the `hit == 0`
    conjunct of the unprobed branch. Widening it to `if not probed:` would send
    every reaped-BUT-answered record to 'unprobed' and make the contradiction
    structurally unreachable on exactly the shape #1176 is about."""

    def test_reaped_without_an_answer_withholds_the_premise(self):
        r = _req("unpro", stamp=6008, prefix_indices=0, registered=0)
        out = PrefetchOutcome(0, hit_tokens=0, probed=False, matched=0)
        self.assertEqual(store_witness(_sched(r, out), r), "unprobed")

    def test_reaped_WITH_an_answer_falls_through_to_the_presence_gate(self):
        """probed=False but hit_tokens>0: the record's probe DID answer, so it
        must be judged on presence, not withheld. Here presence is 0 against
        stamp 6008 -- a contradiction, which `if not probed:` would hide."""
        r = _req("unpro2", stamp=6008, prefix_indices=0, registered=0)
        out = PrefetchOutcome(0, hit_tokens=40, probed=False, matched=0)
        with self.assertRaises(StoreWitnessContradiction):
            _admit(r, out)


class K_TheRegistrationHeadIsStampedByTheRegistrar(CustomTestCase):
    """The frame is only sound if the one registrar that can produce an
    annotated record stamps the head. Pinned at the source, beside the span."""

    def test_prefetch_kvcache_stamps_the_head_beside_the_span(self):
        import inspect

        from sglang.srt.managers import scheduler as scheduler_mod

        src = inspect.getsource(scheduler_mod)
        self.assertIn("req._prefetch_span_tokens = len(_new_input_tokens)", src)
        self.assertIn("req._prefetch_registered_prefix_len = int(_matched_len)", src)

    def test_an_unframed_record_never_over_credits(self):
        """No registration stamp: the two readings become independent lower
        bounds and the larger is used -- never their sum. Here device 40000 and
        span 40000 describe the same tokens, so presence is 40000, not 80000."""
        r = _req("unframed", stamp=80_009, prefix_indices=40_000)
        out = PrefetchOutcome(0, hit_tokens=80_009, probed=True, matched=40_000)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            _admit(r, out)
        msg = str(cm.exception)
        self.assertIn("frame=unframed", msg)
        self.assertIn("presence=40000", msg)


class M_TheResidentRowsAreAnIndependentLowerBound(CustomTestCase):
    """THE OUTER ``max`` IS LOAD-BEARING, and nothing pinned it until a mutant
    that deleted it stayed green (own mutant M5, this round).

    THE SHAPE. ``_prefetch_registered_prefix_len`` is stamped ONCE, when the
    prefetch is registered (scheduler.py:5377), and nothing in the tree
    ever raises it. The device match is NOT frozen with it: the admission loop
    re-matches on later passes, and a sibling request inserting an overlapping
    prefix makes rows matchable that were not matchable at registration time.
    So a request can be registered at head 0 -- the whole prompt was the span
    -- and be 30000 rows device-resident by the time the witness reads it,
    with only a 40-token tail delivered over the span.

    WITHOUT the outer bound the reading is ``head + matched + loaded`` = 40
    against a 30000 stamp: a group STOP on a prefix that is fully present on
    this device. That is the weg1b6 boot-killer direction reconstructed from a
    different input, which is why the bound is a MAX of two independent lower
    bounds and not a sum: ``len(prefix_indices)`` cannot double count anything
    -- those rows are held right now -- and the span reading cannot either.

    The deletion candidate is therefore REFUSED with a reason: dropping the
    bound is not a simplification, it is the licence to kill the group on a
    present prefix."""

    def test_a_grown_device_match_beats_a_stale_zero_registration_head(self):
        r = _req(
            rid="grown-match",
            stamp=30_000,
            tokens=30_000,
            prefix_indices=30_000,
            registered=0,  # registered before the tree grew into this prefix
        )
        out = PrefetchOutcome(40, hit_tokens=40, probed=True, matched=0)
        self.assertEqual(store_witness(_sched(r, out), r), "hit")

    def test_the_can_fail_companion_still_raises_when_nothing_is_resident(self):
        """Same stale head, same tiny span, but the device holds nothing: the
        bound has no second term to offer and the contradiction stands."""
        r = _req(rid="grown-match-empty", stamp=30_000, tokens=30_000, registered=0)
        out = PrefetchOutcome(40, hit_tokens=40, probed=True, matched=0)
        with self.assertRaises(StoreWitnessContradiction) as cm:
            _admit(r, out)
        msg = str(cm.exception)
        self.assertIn("device_resident=0", msg)
        self.assertIn("presence=40", msg)


class L_TheCarrierSymbolsAreGone(CustomTestCase):
    """The wire half of the round-4 deletion stands: no CONTRADICTION
    sentinel, no peers_reporting_contradiction, no report-side probe. The
    authority split is a LOCAL predicate, not a restored carrier."""

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
