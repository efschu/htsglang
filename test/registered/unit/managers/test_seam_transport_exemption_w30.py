"""W30: the seam re-admission is FLIP TRANSPORT, and only that.

THE SPECIMEN. /spinning/evidence-665-f1/SPECIMEN_w30_a1_purity_nocarry_livelock.log
(2026-08-24, pin 9b9b6d1f81, `--phase-flip-purity strict`): 150 flips in 17
minutes, 129 prefill batches, **zero decode batches**, all 28 client requests
timing out at 600 s with no completions. The scheduler's own arm auditor named
it 12 times, unprompted:

    PHASE-POLICY ARM-VERDICT-WRONG: armed pp_to_tp (DRAINED: 0 tok remaining
    (<= one chunk of 4096), 1 req decoding -- exit condition: drained), the
    cutover COMMITTED into the target layout, and it still built no batch in
    8 rounds ... target_can_admit=False ... if this repeats in alternating
    directions it is the 2026-08-16 10:24 ping-pong

It did repeat in alternating directions: 72 pp_to_tp against 69 tp_to_pp.

THE CHAIN, every link a shipped design decision:
  1. the policy arms pp_to_tp BECAUSE a request drained prefill and is ready
     to decode;
  2. the #856 seam RETRACTS that very request (no-carry);
  3. re-admitting it in TP needs a read-through PREFILL batch;
  4. strict purity forbids prefill in TP absolutely -- `Prefill batch
     phase=tp` was logged exactly 0 times in the whole arm;
  5. TP builds nothing, the policy flips back, the request re-prefills in PP,
     drains, and arms the same flip again. For ever.

WHY AN EXEMPTION AND NOT A STAND-DOWN. A purity stand-down lets ORDINARY
prefill run in TP, which both the #838 detector and w29_score.py count as
wrong-layout work -- it would make the acceptance unpassable by our own
scorers and would be dishonest about the user's rule. What crosses the seam
here is different in kind: the tokens were prefilled in the PP window, their
KV is in the canonical store from the #703 fence, and the re-admission
recomputes nothing. It is a cache restore -- seam mechanics, the same category
as the KV the flip moves.

THE DANGEROUS DIRECTION IS THE POINT OF THIS FILE. Every test below that ends
in `_is_still_blocked` is a can-fail: if the exemption ever widens to a
genuine new request, or to an OOM-preempted request's re-prefill, it has
stopped being transport and become a hole in the purity rule.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import types
import unittest

from sglang.srt.managers.phase_purity import (
    MODE_PREFILL_IN_TP,
    MODE_STRICT,
    SEAM_READMIT_ATTR,
    SEAM_TRANSPORT_ROUND_ATTR,
    PhasePurity,
    prefill_blocked_here,
    seam_readmit_candidates,
    seam_transport_exempt,
)
from sglang.srt.managers.phase_policy import PHASE_PP, PHASE_TP
from sglang.test.test_utils import CustomTestCase


def _req(rid, *, seam_epoch=None, oom_retracted=False, cached_prefix=4096):
    """A request double carrying only the marks the gate reads.

    #861d: `cached_prefix` is now part of what the gate reads, and it DEFAULTS
    TO A RESTORED PREFIX because that is what these tests were always about --
    a re-admission whose KV comes back from the canonical store. W37-D showed
    the other case exists on metal (258 batches at #cached-token 0), so it gets
    its own explicit tests below rather than being the silent default here.

    #1157: the gate no longer reads a STAMP (`cache_protected_len` /
    `cached_prompt_tokens_at_retract`) as restore evidence -- it reads the
    re-admission's own store prefetch state off the tree
    (`store_witness`). `cached_prefix > 0` therefore means "this
    request's store read is REGISTERED" (`_Sched` puts the rid into
    `tree_cache.ongoing_prefetch`), and `cached_prefix=0` means no read exists
    for a prompt long enough to need one: cold.
    """
    return types.SimpleNamespace(
        rid=rid,
        seam_readmit_epoch=seam_epoch,
        # The prefix length inserted into the tree cache: >0 means the
        # re-admission genuinely restores rather than recomputes.
        cache_protected_len=cached_prefix,
        # #1157: a long prompt, so the absence of a store read is COLD
        # rather than "bounded below the prefetch threshold".
        origin_input_ids=list(range(4096)),
        _store_read_registered=cached_prefix > 0,
        # What `Req.reset_for_retract` sets -- from ANY retraction path,
        # including plain decode-OOM preemption. Deliberately set on the
        # non-seam doubles so the can-fail tests are about the right thing.
        is_retracted=oom_retracted,
        retracted_stain=oom_retracted,
    )


class _Sched:
    """The fields the purity gate reads, plus the replicated waiting queue."""

    def __init__(self, phase, purity, queue=()):
        self.server_args = type(
            "A", (), {"enable_phase_flip": True, "phase_flip_purity": None}
        )()
        self.phase_flip_active_stack = phase
        self._phase_purity = purity
        self.waiting_queue = list(queue)
        # #1157: the witness the premise reads -- a registered store read per
        # restored double, nothing for a cold one.
        self.tree_cache = types.SimpleNamespace(
            ongoing_prefetch={
                r.rid: object()
                for r in self.waiting_queue
                if getattr(r, "_store_read_registered", False)
            },
            prefetch_loaded_tokens_by_reqid={},
            prefetch_threshold=256,
        )


class TestTheExemptionOpensExactlyWhenTheSeamOwesAReadmission(CustomTestCase):
    def test_a_stamped_request_makes_tp_prefill_permitted(self):
        sched = _Sched(
            PHASE_TP, PhasePurity(mode=MODE_STRICT), [_req("a", seam_epoch=7)]
        )
        self.assertFalse(
            prefill_blocked_here(sched),
            "the cutover's own re-admission must be able to land",
        )

    def test_the_round_flag_is_set_so_the_builder_can_filter(self):
        sched = _Sched(
            PHASE_TP, PhasePurity(mode=MODE_STRICT), [_req("a", seam_epoch=7)]
        )
        prefill_blocked_here(sched)
        self.assertTrue(getattr(sched, SEAM_TRANSPORT_ROUND_ATTR, False))

    def test_the_named_predicate_answers_directly(self):
        # `prefill_blocked_here` is the gate; `seam_transport_exempt` is the
        # NAMED reason it opens. Tested on its own so the carve-out has an
        # identity a reader can grep for, rather than only existing as a
        # branch inside a boolean.
        stamped = _req("a", seam_epoch=4)
        self.assertTrue(
            seam_transport_exempt(
                _Sched(PHASE_TP, PhasePurity(mode=MODE_STRICT), [stamped])
            )
        )
        self.assertFalse(
            seam_transport_exempt(_Sched(PHASE_TP, PhasePurity(mode=MODE_STRICT), []))
        )
        # and it reads the stamp by its published attribute name, so a rename
        # cannot silently disconnect the seam from the gate
        setattr(stamped, SEAM_READMIT_ATTR, None)
        self.assertFalse(
            seam_transport_exempt(
                _Sched(PHASE_TP, PhasePurity(mode=MODE_STRICT), [stamped])
            )
        )

    def test_candidates_are_read_off_the_replicated_queue(self):
        # RANK-UNIFORMITY. The gate's inputs must be the ones that are
        # identical on every rank -- the stamp (from a group-unanimous
        # cutover) and `waiting_queue` (replicated). A rank-local input here
        # splits the group across branches with mismatched collectives.
        q = [_req("a", seam_epoch=1), _req("b"), _req("c", seam_epoch=1)]
        sched = _Sched(PHASE_TP, PhasePurity(mode=MODE_STRICT), q)
        self.assertEqual([r.rid for r in seam_readmit_candidates(sched)], ["a", "c"])


class TestTheDangerousDirection(CustomTestCase):
    """Every one of these is a CAN-FAIL. If any stops blocking, the exemption
    has become a hole in the purity rule rather than a seam carve-out."""

    def test_a_genuine_new_request_is_still_blocked(self):
        sched = _Sched(PHASE_TP, PhasePurity(mode=MODE_STRICT), [_req("fresh")])
        self.assertTrue(
            prefill_blocked_here(sched),
            "a never-retracted request prefilling in TP is real wrong-layout work",
        )

    def test_an_oom_preempted_request_is_still_blocked(self):
        # THE TEST THIS FILE EXISTS FOR. `Req.reset_for_retract` sets
        # `is_retracted`/`retracted_stain` for FOUR different paths, only one
        # of which is the #856 seam: decode-OOM preemption (`retract_decode`),
        # the PD prefill path and the PP void path set the identical booleans.
        # An exemption keyed on `is_retracted` would silently exempt every
        # OOM-preempted request's re-prefill, which is real work. It must not.
        sched = _Sched(
            PHASE_TP,
            PhasePurity(mode=MODE_STRICT),
            [_req("preempted", oom_retracted=True)],
        )
        self.assertFalse(seam_readmit_candidates(sched))
        self.assertTrue(prefill_blocked_here(sched))

    def test_an_empty_queue_is_still_blocked(self):
        sched = _Sched(PHASE_TP, PhasePurity(mode=MODE_STRICT), [])
        self.assertTrue(prefill_blocked_here(sched))

    def test_a_mixed_queue_still_opens_but_the_builder_must_filter(self):
        # The gate opens because a genuine re-admission is owed; the batch is
        # then kept to transport only by the builder. Both halves are
        # required -- this asserts the gate half and names the other, which
        # `TestTheBuilderKeepsAnExemptBatchToTransportOnly` pins.
        q = [_req("readmit", seam_epoch=2), _req("fresh")]
        sched = _Sched(PHASE_TP, PhasePurity(mode=MODE_STRICT), q)
        self.assertFalse(prefill_blocked_here(sched))
        self.assertTrue(getattr(sched, SEAM_TRANSPORT_ROUND_ATTR, False))


class TestTheExemptionChangesNothingElse(CustomTestCase):
    def test_pp_is_untouched(self):
        sched = _Sched(
            PHASE_PP, PhasePurity(mode=MODE_STRICT), [_req("a", seam_epoch=1)]
        )
        self.assertFalse(prefill_blocked_here(sched), "prefill belongs in PP")

    def test_prefill_in_tp_mode_never_reaches_the_exemption(self):
        # The permissive mode returns earlier; the exemption is strict-only
        # machinery and must not start deciding anything under it.
        sched = _Sched(PHASE_TP, PhasePurity(mode=MODE_PREFILL_IN_TP), [_req("a")])
        self.assertFalse(prefill_blocked_here(sched))
        self.assertFalse(getattr(sched, SEAM_TRANSPORT_ROUND_ATTR, False))

    def test_the_round_flag_does_not_latch(self):
        # If it latched, a later round would filter the builder down to
        # stamped requests that no longer exist and build nothing -- the W30
        # livelock traded for a quieter one.
        sched = _Sched(
            PHASE_TP, PhasePurity(mode=MODE_STRICT), [_req("a", seam_epoch=1)]
        )
        prefill_blocked_here(sched)
        self.assertTrue(getattr(sched, SEAM_TRANSPORT_ROUND_ATTR, False))
        sched.waiting_queue = []  # the debt is paid
        prefill_blocked_here(sched)
        self.assertFalse(getattr(sched, SEAM_TRANSPORT_ROUND_ATTR, True))


class TestTheStampIsSeamOnlyAndOneShot(CustomTestCase):
    def test_only_the_cutover_closure_sets_the_stamp(self):
        # Keyed on the SOURCE because the property is "nowhere else writes
        # it". A behavioural test can show one writer; only this shows the
        # absence of the others, which is the whole safety argument.
        import subprocess

        out = subprocess.run(
            [
                "grep",
                "-rn",
                "seam_readmit_epoch *=",
                "--include=*.py",
                "python/sglang/",
            ],
            capture_output=True,
            text=True,
            cwd="/spinning/wt-851c",
        ).stdout
        writers = [
            ln
            for ln in out.splitlines()
            # the field's own declaration and the one-shot clear are not
            # "writers" in the sense that matters here
            if "schedule_batch.py" not in ln
        ]
        self.assertEqual(
            len(writers),
            1,
            f"exactly one site may stamp a seam re-admission; found:\n{out}",
        )
        self.assertIn("phase_flip_runtime.py", writers[0])

    def test_the_declared_field_defaults_to_none(self):
        # A real declared field, not a magic dynamic attribute, so a reader
        # that asks for it on a never-retracted request gets a defined answer.
        import inspect

        from sglang.srt.managers.schedule_batch import Req

        self.assertIn("self.seam_readmit_epoch = None", inspect.getsource(Req.__init__))


class TestTheBuilderKeepsAnExemptBatchToTransportOnly(CustomTestCase):
    """The other half of the carve-out: the gate says WHETHER, this says WHAT."""

    def test_the_builder_skips_unstamped_requests_on_an_exempt_round(self):
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
        self.assertIn("transport_only", src)
        self.assertIn("seam_transport_only", src, "the skip must be named/counted")
        self.assertIn("SEAM_READMIT_ATTR", src, "read via the named constant")

    def test_the_stamp_is_spent_on_admission(self):
        # One-shot. Left on the request it would exempt every later prefill
        # that request ever needs, turning a seam carve-out into a permanent
        # hole in the purity rule.
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler._get_new_batch_prefill_raw)
        self.assertIn("setattr(req, SEAM_READMIT_ATTR, None)", src)


if __name__ == "__main__":
    unittest.main()


class TestTheExemptionOutranksDrainModeSuppression(CustomTestCase):
    """W31 arm 1: THE FIX WAS INSTALLED AND UNREACHABLE.

    The exemption first sat after `prefill_allowed_in_tp`, which is BELOW the
    #677 drain-mode suppression. The W31 recipe runs
    `--phase-policy-drain-mode`, so `prefill_suppressed_in_tp` returned True
    and `prefill_blocked_here` returned before the exemption was ever
    evaluated. Measured (SPECIMEN_w31_a1_exemption_below_drain_gate.log): the
    seam retracted 87 requests across 39 pp_to_tp flips and logged
    `SEAM TRANSPORT ADMITTED` **0** times and `Prefill batch phase=tp` **0**
    times -- the W30 livelock reproduced exactly, with its own fix in the
    tree.

    Same defect shape this very function already carries a note about: "What
    broke was ORDER -- suppression was checked FIRST and returned True, so the
    valve never ran."

    AND THE ORDER IS SUBSTANTIVE, not cosmetic. Drain mode forbids TP prefill
    because "a TP window entered to finish a bundle must not admit the work it
    was entered to escape". A request the cutover itself retracted is not that
    work -- it IS the bundle this window was entered to finish. Suppressing it
    makes the drain contract unsatisfiable rather than defending it.
    """

    def _drain_sched(self, queue):
        from sglang.srt.managers.phase_policy import PhasePolicyConfig

        sched = _Sched(PHASE_TP, PhasePurity(mode=MODE_STRICT), queue)
        sched.phase_policy_cfg = PhasePolicyConfig(
            enabled=True, drain_mode=True, flip_tokens=7000
        )
        return sched

    def test_a_stamped_request_is_admitted_even_under_drain_mode(self):
        sched = self._drain_sched([_req("readmit", seam_epoch=5)])
        self.assertFalse(
            prefill_blocked_here(sched, running_bs=1),
            "the seam's own re-admission IS the bundle drain mode is finishing",
        )

    def test_drain_mode_still_suppresses_ordinary_prefill(self):
        # CAN-FAIL: the exemption must not become a way around drain mode for
        # work the seam did not retract. If this stops blocking, #677's
        # contract has been dissolved rather than qualified.
        sched = self._drain_sched([_req("fresh")])
        self.assertTrue(prefill_blocked_here(sched, running_bs=1))

    def test_the_exemption_is_checked_before_the_drain_gate(self):
        # Keyed on ORDER in the source, because that is precisely what W31
        # arm 1 got wrong and no behavioural test on a passing path can see.
        import inspect

        from sglang.srt.managers import phase_purity

        src = inspect.getsource(phase_purity.prefill_blocked_here)
        exempt_at = src.find("seam_transport_exempt(scheduler)")
        drain_at = src.find("prefill_suppressed_in_tp(")
        self.assertGreater(exempt_at, -1, "the exemption must be in this function")
        self.assertGreater(drain_at, -1)
        self.assertLess(
            exempt_at, drain_at, "seam transport must outrank drain suppression"
        )

    def test_the_exemption_is_single_sited(self):
        # It was moved, not copied. Two call sites would drift.
        import inspect

        from sglang.srt.managers import phase_purity

        src = inspect.getsource(phase_purity.prefill_blocked_here)
        self.assertEqual(src.count("seam_transport_exempt(scheduler)"), 1)


class TestOnePredicateBothCallers(CustomTestCase):
    """W32: THE EXEMPTION MUST NOT EXIST AS TWO COPIES.

    Measured, in the policy's own words, 23 times
    (SPECIMEN_w32_policy_purity_copy_pulls_back_to_pp.log):

        PHASE-POLICY arming tp_to_pp: pending prefill 1 tok > 0
          (purity: prefill cannot run in tp)

    `prefill_blocked_here` had been taught the seam-transport exemption. The
    POLICY kept an independent copy of the same rule, so the moment the seam
    re-admitted its residents those tokens read as "pending prefill" and the
    policy armed tp_to_pp -- leaving TP before the exemption at the batch
    builder could be consulted. The exemption fired ONCE in 144 pp_to_tp
    flips.

    That is the same shape as W31 arm 1 one level DOWN (the exemption sat
    below the drain gate and never ran). Twice now a correct mechanism has
    been overridden by a second site enforcing the same payload. So both
    callers must derive from ONE function, and this class is what stops them
    drifting apart again.

    MANDATORY INVENTORY (Ein-Job-ein-Mover), recorded here because the rule
    says a new canonical authority owes one. Sites enforcing "may prefill run
    in the TP layout":
      1. phase_purity.prefill_blocked_here      -- the batch gate
      2. the phase policy, via pending-token accounting -- fixed here
      3. w29_score.py                            -- the scorer (fixed at W32)
      4. scheduler._phase_admits("prefill_in_tp") -- a runtime probe
    and two that are NOT this payload: scheduler.py's boot-time
    `prefill_runs_in_tp` config collapse (static, derived from the MODE) and
    model_runner_kv_cache_mixin's `survivable` (KV sizing).
    """

    def test_both_answers_come_from_the_same_candidate_function(self):
        import inspect

        from sglang.srt.managers import phase_purity

        for fn in (
            phase_purity.seam_transport_exempt,
            phase_purity.seam_transport_pending_tokens,
        ):
            self.assertIn(
                "seam_readmit_candidates",
                inspect.getsource(fn),
                f"{fn.__name__} must derive from the one authority",
            )

    def test_a_mutation_on_the_shared_function_reds_both_callers(self):
        # THE DIVERGENCE PROOF. If the two ever stop sharing a source of
        # truth, this passes while the real system contradicts itself -- which
        # is precisely what W32 measured.
        from sglang.srt.managers import phase_purity

        stamped = _req("a", seam_epoch=1)
        stamped.origin_input_ids = list(range(100))
        stamped.cache_protected_len = 0
        sched = _Sched(PHASE_TP, PhasePurity(mode=MODE_STRICT), [stamped])

        self.assertTrue(phase_purity.seam_transport_exempt(sched))
        self.assertEqual(phase_purity.seam_transport_pending_tokens(sched), 100)

        original = phase_purity.seam_readmit_candidates
        try:
            phase_purity.seam_readmit_candidates = lambda s: []
            self.assertFalse(
                phase_purity.seam_transport_exempt(sched),
                "the gate must follow the shared function",
            )
            self.assertEqual(
                phase_purity.seam_transport_pending_tokens(sched),
                0,
                "the policy's accounting must follow the SAME shared function",
            )
        finally:
            phase_purity.seam_readmit_candidates = original

    def test_the_policy_input_boundary_subtracts_transport(self):
        # Pinned at the boundary, not at each trigger: a per-trigger
        # subtraction would be a fourth copy of the same judgement.
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler)
        self.assertIn("seam_transport_pending_tokens", src)

    def test_unstamped_pending_is_still_pp_work(self):
        # CAN-FAIL: ordinary queued prefill must still count toward the
        # tp_to_pp arm, or the policy stops returning to PP at all.
        from sglang.srt.managers import phase_purity

        fresh = _req("fresh")
        fresh.origin_input_ids = list(range(500))
        fresh.cache_protected_len = 0
        sched = _Sched(PHASE_TP, PhasePurity(mode=MODE_STRICT), [fresh])
        self.assertEqual(phase_purity.seam_transport_pending_tokens(sched), 0)


class TheExemptionsPremiseIsCheckedNotAsserted(unittest.TestCase):
    """#861d: W37-D falsified the exemption's factual claim on metal.

    The exemption permits prefill in the TP layout because "the re-admission
    recomputes nothing -- it is a cache restore". W37-D ran 258 such batches at
    ``#new-token: 4096, #cached-token: 0`` (and #cached-token was 0 on ALL 1441
    occurrences in the boot, storage_hit 0), i.e. cold prefill of real work in
    the decode layout -- the user's strict-batch law broken by the exemption
    written to respect it.

    CLASS: a guard or exemption that ASSERTS a premise in prose and never
    verifies it at runtime. Same class as #861c/F1, which assumed two host
    pools have equal slot counts. Both silent for a whole window; both fixed by
    checking the premise where it is relied upon.
    """

    def setUp(self):
        import inspect as _inspect

        from sglang.srt.managers import phase_purity

        self.pp = phase_purity
        self.inspect = _inspect

    @staticmethod
    def _s(queue):
        return _Sched(PHASE_TP, PhasePurity(mode=MODE_STRICT), queue)

    def test_a_restored_readmission_still_opens_the_exemption(self):
        """The legitimate case: KV really comes back -- since #1157 measured
        as a REGISTERED store read for the rid, not a stamp."""
        s = self._s([_req("a", seam_epoch=1, cached_prefix=4096)])
        self.assertTrue(self.pp.seam_transport_premise_holds(s))

    def test_a_stamp_alone_no_longer_opens_the_exemption(self):
        """INVERTED under #1157: before it, `cached_prompt_tokens_at_retract
        = 4096` (or `cache_protected_len > 0`) alone made this True. WITHDRAWN
        on boot weg1b3: the stamp licensed a P=0 TP prefill of 84k tokens
        while the store read for that request had been reaped unprobed.
        A stamped long request with no registered or answered store read is
        COLD; the premise refuses it."""
        r = _req("a", seam_epoch=1, cached_prefix=0)
        r.cached_prompt_tokens_at_retract = 4096
        r.cache_protected_len = 4096
        s = self._s([r])
        self.assertFalse(self.pp.seam_transport_premise_holds(s))

    def test_a_cold_readmission_is_REFUSED(self):
        """THE W37-D SPECIMEN. Stamped, queued, and nothing cached: admitting
        it in TP would be a 4096-token cold prefill wearing transport's
        clothes."""
        s = self._s([_req("a", seam_epoch=1, cached_prefix=0)])
        self.assertFalse(self.pp.seam_transport_premise_holds(s))

    def test_a_mixed_queue_opens_on_the_restored_one(self):
        """One genuine restore is enough ground for the round; the builder's
        own filter still keeps unstamped work out of the batch."""
        s = self._s(
            [
                _req("cold", seam_epoch=1, cached_prefix=0),
                _req("warm", seam_epoch=1, cached_prefix=2048),
            ]
        )
        self.assertTrue(self.pp.seam_transport_premise_holds(s))

    def test_an_empty_queue_holds(self):
        self.assertFalse(self.pp.seam_transport_premise_holds(self._s([])))

    def test_the_gate_consults_the_premise(self):
        """FUTURE-CHECK: the call site must AND the two, or the premise check
        is installed and unreachable -- the shape that cost W32."""
        src = self.inspect.getsource(self.pp.prefill_blocked_here)
        self.assertIn("seam_transport_premise_holds", src)
        self.assertIn("seam_transport_exempt(scheduler) and", src)
