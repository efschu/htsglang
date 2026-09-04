"""#1157 -- the seam premise is a MEASURED WITNESS, not a stamp.

THE SPECIMEN (boot weg1b3 @ 6980c75eac). `phase_purity.seam_transport_premise_
holds` counted `cached_prompt_tokens_at_retract > 0` as "restored" and the
policy held on it: 'premise verified on the retract credit' (log 69244,
23:56:17). The store read for that very request (rid 679e4568) had been reaped
before its probe (`#1157 PREFETCH REAPED probed=False`, the F1a/F1c file), so
the "verified" premise licensed a P=0 admission and six recomputed 4096-token
TP chunks (log 72593 -> 98337..100501). A stamp is a fact about the past; the
witness is the re-admission's OWN prefetch state on the tree -- the record the
admission loop already drains (`_prefetch_done_for`) and pops
(`pop_prefetch_loaded_tokens`). No second bookkeeping.

THE FOUR RED-FIRST CASES the operator named, plus the propagation of the STOP:

 (i)   pending store read       -> held (exemption open, admission waits)
 (ii)  probed hit > 0           -> admitted with P = hit
 (iii) probed 0, stamp > 0      -> `StoreWitnessContradiction` (group STOP)
 (iv)  probed 0, stamp 0        -> cold, admitted at P=0 as today

REVIEW FIX (operator decisions, 2026-09-03):
 B1  the record must survive pickle/deepcopy (it rides req.storage_hit_length
     into the pickled detokenizer output) and the admission sites store a
     bare int.
 B2  beside a stamp the witness MEASURES the stamped span: 'hit' only if
     loaded > 0 and stamp - hit_tokens <= one HiCache chunk (the tree's
     `_prefetch_chunk_tokens` = chunked_prefill_size); a revoke (loaded=0,
     e.g. the 40-token chat-template header answering every probe) or a
     shortfall beyond the allowance beside a stamp raises.
 B3  the raise text is '#1157 STORE WITNESS CONTRADICTION' (the former
     'SEAM RESTORE' pair tripped the #1068 seam-copy zombie gate).

MUTANTS: restore the stamp premise -> (iii) turns into a silent True; drop the
PP0 hold -> (i)'s source pin goes red; drop the shortfall term -> B2 red.
"""

import copy
import inspect
import pickle
import re
import types
import unittest

from sglang.srt.managers import phase_purity, scheduler as scheduler_mod
from sglang.srt.managers.phase_purity import (
    SEAM_GRANT_CONSUMED_ATTR,
    SEAM_READMIT_ATTR,
    StoreWitnessContradiction,
    assert_store_witness_at_admission,
    store_witness,
    seam_transport_premise_holds,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.hicache_storage import PrefetchOutcome
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

LONG = 84_027  # the weg1b3 LONG re-admission
STAMP = 80_009  # its `cached_prompt_tokens_at_retract` (#1036 protected_hwm)
CHUNK = 4_096  # chunked_prefill_size on weg1b3 = the one-chunk allowance
HEADER = 40  # the chat-template header every probe answers with (#1028B kv=40)


def _req(rid="679e4568", *, stamp=STAMP, tokens=LONG, seam_epoch=3):
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


def _sched(reqs, *, pending=(), outcomes=None, storage_on=True):
    """The fields the premise reads: the replicated queue and the tree's own
    prefetch state (a registered record per pending rid, the popped-at-
    admission outcome record, the threshold and the tier probes)."""
    pool = types.SimpleNamespace(size=100)
    pool.available_size = lambda: 50  # host content: the tier probe never refutes
    tree = types.SimpleNamespace(
        root_node=types.SimpleNamespace(children={}),
        cache_controller=types.SimpleNamespace(mem_pool_host=pool),
        enable_storage=storage_on,
        ongoing_prefetch={rid: object() for rid in pending},
        prefetch_loaded_tokens_by_reqid=dict(outcomes or {}),
        prefetch_threshold=256,
        _prefetch_chunk_tokens=CHUNK,
    )
    return types.SimpleNamespace(tree_cache=tree, waiting_queue=list(reqs))


class I_APendingStoreReadIsHeld(CustomTestCase):
    def test_the_witness_says_pending_and_the_premise_holds_on_it(self):
        """Red on the parent for a request WITHOUT a stamp: the old premise
        read only stamps, so a pending read with stamp 0 was 'cold'."""
        r = _req(stamp=0)
        s = _sched([r], pending=[r.rid])
        self.assertEqual(store_witness(s, r), "pending")
        self.assertTrue(seam_transport_premise_holds(s))

    def test_a_deferred_issue_is_pending_too(self):
        r = _req(stamp=0)
        r.prefetch_deferred = "rate_limited"
        self.assertEqual(store_witness(_sched([r]), r), "pending")

    def test_admission_waits_on_the_drained_verdict(self):
        """The hold the witness reuses: `_prefetch_done_for` answers False
        for a pending record, and the PP0 arm of the admission loop skips
        the request on that answer (`continue`), never admitting at P=0."""
        stub = types.SimpleNamespace(
            ps=types.SimpleNamespace(tp_size=1),
            tree_cache=types.SimpleNamespace(check_prefetch_progress=lambda rid: True),
        )
        done = Scheduler._prefetch_done_for.__get__(stub)
        self.assertFalse(done(types.SimpleNamespace(rid="r"), {"r": False}))
        src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
        self.assertRegex(
            src,
            r"if not _local_prefetch_done:\s*\n\s*_note_skip\(\"prefetch_pending_pp0\", req\.rid\)\s*\n\s*continue",
            "the PP0 hold on a pending prefetch is gone: a re-admission would "
            "be admitted at P=0 before its store read has answered",
        )


class II_AProbedHitIsAdmittedWithItsPrefix(CustomTestCase):
    def test_the_witness_says_hit_and_the_premise_holds(self):
        """The probe covered the stamped span within one chunk and the
        read-through loaded it (B2: 80000 beside 80009, shortfall 9)."""
        r = _req()
        out = PrefetchOutcome(80_000, hit_tokens=80_000, probed=True)
        s = _sched([r], outcomes={r.rid: out})
        self.assertEqual(store_witness(s, r), "hit")
        self.assertTrue(seam_transport_premise_holds(s))

    def test_the_admission_site_credits_the_loaded_prefix(self):
        """What the two admission sites do with the popped record, driven
        exactly as they are written: the witness assert, then the credit."""
        r = _req()
        s = _sched([r], outcomes={r.rid: PrefetchOutcome(80_000, hit_tokens=80_000, probed=True)})
        loaded = s.tree_cache.prefetch_loaded_tokens_by_reqid.pop(r.rid, 0)
        assert_store_witness_at_admission(r, loaded, s.tree_cache)  # no raise
        if loaded > 0:
            r.storage_hit_length = int(loaded)
        self.assertEqual(r.storage_hit_length, 80_000)
        # B1: the credit is a bare int, never the annotated record.
        self.assertIs(type(r.storage_hit_length), int)

    def test_the_admission_sites_consult_the_witness_right_after_the_pop(self):
        src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
        # #1176 (review r4): the `is_authority=` argument is DELETED -- every
        # rank raises on its own true contradiction, so there is no authority
        # parameter for the two sites to drift on. The pin still demands BOTH
        # arms, the tree, and the int(...) credit.
        hits = re.findall(
            r"loaded_tokens = self\.tree_cache\.pop_prefetch_loaded_tokens\(req\.rid\)"
            r"(?:\s*\n\s*#[^\n]*)*"
            r"\s*\n\s*assert_store_witness_at_admission\(\s*\n"
            r"\s*req, loaded_tokens, self\.tree_cache\s*\n\s*\)"
            r"\s*\n\s*if loaded_tokens > 0:\s*\n\s*req\.storage_hit_length = int\(loaded_tokens\)",
            src,
        )
        self.assertEqual(
            len(hits),
            2,
            "both admission arms (PP0 and TP) must assert with the tree and "
            "store the credit as int(...) (B1)",
        )
        self.assertNotIn(
            "is_authority",
            src,
            "the follower-vs-authority split is deleted (#1176 review r4)",
        )


class III_AProbedMissBesideAStampStopsTheGroup(CustomTestCase):
    def _contradiction(self):
        r = _req()
        return r, _sched([r], outcomes={r.rid: PrefetchOutcome(0, hit_tokens=0, probed=True)})

    def test_the_premise_raises_the_named_line(self):
        """THE RED-FIRST CASE. On the parent this returned True (the stamp)."""
        r, s = self._contradiction()
        with self.assertRaises(StoreWitnessContradiction) as cm:
            seam_transport_premise_holds(s)
        msg = str(cm.exception)
        # #1176 (review r3): the line carries the stamp and the presence and
        # nothing netted off either. `device_resident` is a printed diagnostic
        # at the END of the field list; the `demand=` term of 1634bc3d28 is
        # deleted (it double-subtracted this request's own match).
        self.assertIn(
            f"#1157 STORE WITNESS CONTRADICTION rid={r.rid} stamped={STAMP} "
            f"probed_hit=0 loaded=0 "
            f"allowance={CHUNK} shortfall={STAMP} requested={LONG}",
            msg,
        )
        self.assertIn("device_resident=0", msg)
        self.assertNotIn("demand=", msg)
        self.assertNotIn("SEAM RESTORE", msg)
        self.assertIsInstance(cm.exception, RuntimeError)

    def test_a_revoke_on_the_chat_header_beside_a_stamp_raises(self):
        """THE ADVERSARIAL RED CASE (review B2): every probe answers the
        ~40-token chat-template header, so a revoked re-admission
        (loaded=0, hit_tokens=40) beside stamp=80009 used to read 'hit' and
        license P=0. It is a contradiction."""
        r = _req()
        s = _sched([r], outcomes={r.rid: PrefetchOutcome(0, hit_tokens=HEADER, probed=True)})
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(s, r)
        self.assertIn(f"probed_hit={HEADER} loaded=0 allowance={CHUNK}", str(cm.exception))
        with self.assertRaises(StoreWitnessContradiction):
            assert_store_witness_at_admission(
                r, PrefetchOutcome(0, hit_tokens=HEADER, probed=True), s.tree_cache
            )

    def test_a_probed_shortfall_beyond_one_chunk_beside_a_stamp_raises(self):
        """4096 loaded and probed beside 80009: shortfall 75913 > allowance."""
        r = _req()
        s = _sched([r], outcomes={r.rid: PrefetchOutcome(4096, hit_tokens=4096, probed=True)})
        with self.assertRaises(StoreWitnessContradiction) as cm:
            store_witness(s, r)
        self.assertIn(f"shortfall={STAMP - 4096}", str(cm.exception))

    def test_a_shortfall_within_one_chunk_is_a_hit(self):
        """stamp - hit_tokens <= allowance, loaded > 0 -> 'hit' (boundary)."""
        r = _req()
        out = PrefetchOutcome(STAMP - CHUNK, hit_tokens=STAMP - CHUNK, probed=True)
        self.assertEqual(store_witness(_sched([r], outcomes={r.rid: out}), r), "hit")
        over = PrefetchOutcome(STAMP - CHUNK - 1, hit_tokens=STAMP - CHUNK - 1, probed=True)
        with self.assertRaises(StoreWitnessContradiction):
            store_witness(_sched([r], outcomes={r.rid: over}), r)

    def test_the_allowance_is_the_trees_one_chunk_term(self):
        """The allowance is `_prefetch_chunk_tokens` (chunked_prefill_size);
        a stand-in tree without it falls back to the prefetch threshold."""
        from sglang.srt.managers.phase_purity import _store_witness_allowance

        self.assertEqual(_store_witness_allowance(_sched([]).tree_cache), CHUNK)
        bare = types.SimpleNamespace(prefetch_threshold=256)
        self.assertEqual(_store_witness_allowance(bare), 256)
        disabled = types.SimpleNamespace(_prefetch_chunk_tokens=-1, prefetch_threshold=256)
        self.assertEqual(_store_witness_allowance(disabled), 256)

    def test_the_admission_site_raises_on_the_popped_record(self):
        r, s = self._contradiction()
        loaded = s.tree_cache.prefetch_loaded_tokens_by_reqid.pop(r.rid, 0)
        with self.assertRaises(StoreWitnessContradiction):
            assert_store_witness_at_admission(r, loaded, s.tree_cache)

    def test_the_stop_passes_through_the_fail_open_arm_probe(self):
        """`_1040_seam_readmit_ready` is fail-open (`except Exception: return
        True`); the STOP is not a probe error and must not be swallowed
        into a 'ready' verdict that arms the flip."""
        r, s = self._contradiction()
        ready = Scheduler._1040_seam_readmit_ready.__get__(s)
        with self.assertRaises(StoreWitnessContradiction):
            ready()

    def test_every_fail_open_probe_on_the_path_re_raises(self):
        src = inspect.getsource(scheduler_mod)
        self.assertEqual(
            len(re.findall(r"except StoreWitnessContradiction:\s*\n\s*raise", src)),
            3,
            "the three fail-open probes (_purity_allows, _1040_seam_readmit_ready, "
            "the policy-input premise read) must each re-raise the STOP",
        )

    def test_an_unannotated_zero_is_not_a_contradiction(self):
        """A tree that records only the loaded count (hiradix, flexkv) says
        nothing about the probe: no raise, no false STOP."""
        r = _req()
        assert_store_witness_at_admission(r, 0, _sched([r]).tree_cache)
        self.assertEqual(store_witness(_sched([r], outcomes={r.rid: 0}), r), "unprobed")


class IV_AGenuinelyColdRequestIsAdmittedAtZeroAsToday(CustomTestCase):
    def test_probed_zero_with_stamp_zero_is_cold_not_a_stop(self):
        r = _req(stamp=0)
        s = _sched([r], outcomes={r.rid: PrefetchOutcome(0, hit_tokens=0, probed=True)})
        self.assertEqual(store_witness(s, r), "cold")
        loaded = s.tree_cache.prefetch_loaded_tokens_by_reqid.pop(r.rid, 0)
        assert_store_witness_at_admission(r, loaded, s.tree_cache)  # no raise
        self.assertEqual(loaded, 0)
        self.assertFalse(seam_transport_premise_holds(s))

    def test_a_cold_request_with_a_header_hit_is_still_a_hit_as_today(self):
        """Stamp 0 keeps today's plain reading: the probe answered, so the
        read-through serves what it answered (the B2 rule is stamp-scoped)."""
        r = _req(stamp=0)
        s = _sched([r], outcomes={r.rid: PrefetchOutcome(HEADER, hit_tokens=HEADER, probed=True)})
        self.assertEqual(store_witness(s, r), "hit")

    def test_a_stamp_with_no_store_read_at_all_is_cold(self):
        """The withdrawn licence: a long stamped request whose read was
        never registered is not a restore."""
        r = _req()
        self.assertEqual(store_witness(_sched([r]), r), "cold")
        self.assertFalse(seam_transport_premise_holds(_sched([r])))

    def test_a_prompt_below_the_prefetch_threshold_is_bounded(self):
        """No store read can exist for it (`too_short` at the #915 gate);
        the recompute is bounded below the threshold, under the one-chunk
        allowance -- the /health_generate probe shape."""
        r = _req(stamp=40, tokens=40)
        self.assertEqual(store_witness(_sched([r]), r), "bounded")
        self.assertTrue(seam_transport_premise_holds(_sched([r])))

    def test_a_mixed_queue_opens_on_the_witnessed_one_and_the_cold_one_rides(self):
        warm, cold = _req(rid="warm"), _req(rid="cold", stamp=0)
        s = _sched([warm, cold], pending=["warm"])
        self.assertTrue(seam_transport_premise_holds(s))


class B1TheRecordSurvivesSerialization(CustomTestCase):
    def test_pickle_and_deepcopy_round_trip(self):
        """THE MATCHED CHECK (review B1): the record is rebuilt with its
        annotation by pickle.loads and copy.deepcopy. Red on the R1 build:
        TypeError (__new__ missing keyword-only arguments)."""
        rec = PrefetchOutcome(1912, hit_tokens=6008, probed=True)
        for name, back in (
            ("pickle", pickle.loads(pickle.dumps(rec, pickle.HIGHEST_PROTOCOL))),
            ("deepcopy", copy.deepcopy(rec)),
            ("copy", copy.copy(rec)),
        ):
            with self.subTest(name):
                self.assertEqual(back, rec)
                self.assertIsInstance(back, PrefetchOutcome)
                self.assertEqual(back.hit_tokens, 6008)
                self.assertTrue(back.probed)
        # The detokenizer payload shape: a dict with the record inside.
        payload = pickle.dumps({"storage": rec}, pickle.HIGHEST_PROTOCOL)
        self.assertEqual(pickle.loads(payload)["storage"].hit_tokens, 6008)

    def test_positional_construction_keeps_the_keyword_form(self):
        a = PrefetchOutcome(0, 40, True)
        b = PrefetchOutcome(0, hit_tokens=40, probed=True)
        self.assertEqual((a.hit_tokens, a.probed), (b.hit_tokens, b.probed))
        self.assertEqual((PrefetchOutcome(3).hit_tokens, PrefetchOutcome(3).probed), (0, False))


class TheStampIsNoLongerReadByThePremise(CustomTestCase):
    def test_the_premise_source_consults_the_witness_not_the_stamp(self):
        src = inspect.getsource(phase_purity.seam_transport_premise_holds)
        self.assertIn("store_witness(scheduler, req)", src)
        self.assertNotIn("SEAM RESTORE", inspect.getsource(phase_purity), "B3: the token trips the #1068 zombie gate")
        body = src.split('"""', 2)[2]  # past the docstring
        self.assertNotRegex(
            body,
            r"getattr\(req, \"cached_prompt_tokens_at_retract\"",
            "the premise reads the retract stamp again -- the weg1b3 licence",
        )


if __name__ == "__main__":
    unittest.main()
