"""#890: the seam-transport exemption is checked at the GRANT and must be REVOKED at the EXECUTION.

THE HOLE, IN ONE SENTENCE. `seam_transport_premise_holds` verifies -- before
the batch is built -- that a re-admitted request's tokens "were computed in the
PP window and the fence persisted them", i.e. that the re-admission is a cache
RESTORE and therefore transport rather than work. Nothing then checks whether
the restore ACTUALLY HAPPENED, and `restore_seam_state` has two branches that
refuse it outright and say so in their own log line: *"Dropped; these tokens are
recomputed."* A recompute is exactly the thing the premise promised would not
occur -- real prefill work, in the TP decode layout, under a permission granted
on the opposite claim.

MEASURED, W38: 90 and 21 `SEAM RESTORE REFUSED (LAYOUT)` in two boots
(schedule_batch.py:2176). Not a corner: the layout axis compares the copy's
per-layer geometry against the live pool's, and a phase flip REPARTITIONS the
layers, so once a copy fails this way it fails for every request the same
cutover stamped, on every rank, for as long as the two geometries differ.

WHY REVOCATION AND NOT LOUDNESS. The current design's answer is measurement:
the premise docstring says a transport batch that recomputes "is loud, not
silent", via `layout_conformance.work_in_wrong_layout` scoring the batch's own
`cached_tokens`. That instrument is not usable as the revocation signal -- the
`#cached-token` counter reads 0 for unrelated reasons before #873 -- and, more
fundamentally, an alarm does not close a permission. The NEXT cutover stamps
the same request again, the premise reads the same surviving evidence
(`cached_prompt_tokens_at_retract`, re-stamped by the recompute the refusal
forced), and the exemption is re-issued on a claim this very request has
already falsified on metal.

THE CLASS. Grant checked at issue, never revoked at execution -- the shape of
#501 (`kv_overallocated_freed` set before the declines that make it false).

RANK-UNIFORMITY, which the purity gate requires of every input it takes. Both
refusal branches are uniform across the group:
  * the EXTENT branch compares `kv_cache_cpu_extent` against `seqlen - 1`, and
    the extent field is stamped in `offload_kv_cache` from `token_indices
    .numel()` over `[: seqlen - 1]` -- the same logical length, replicated.
  * the LAYOUT branch compares the copy's `layer_num`/`start_layer` against the
    live pool's. A flip that repartitions layers changes `layer_num` on EVERY
    rank (a stage's slice against the whole), so the verdict is a property of
    the flip, not of the rank; a flip that repartitions nothing leaves them
    equal on every rank. The `None` tolerance is likewise uniform: whether a
    pool can state a layout is a property of the pool CLASS, which is the same
    on every rank.

THE DANGEROUS DIRECTION IS PINNED HERE TOO. A revocation that fired on the
ordinary population would disarm the W30 exemption and restore the livelock it
exists to prevent (150 flips in 17 minutes, zero decode batches). Every test
below whose name ends in `_still_holds` is that can-fail.

Hermetic: no CUDA, no pool, no scheduler -- stubs only.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest

import torch

from sglang.srt.managers.phase_policy import PHASE_TP
from sglang.srt.managers.phase_purity import (
    MODE_STRICT,
    SEAM_READMIT_ATTR,
    SEAM_RESTORE_REFUSED_ATTR,
    PhasePurity,
    prefill_blocked_here,
    seam_transport_premise_holds,
)
from sglang.test.test_utils import CustomTestCase

MAMBA_SLOT = torch.tensor([3], dtype=torch.int64)

#: Two layer geometries the flip moves between. Shape borrowed from
#: test_seam_layout_contract_861c.py so both files describe the same axis.
PP_LAYOUT = ("kv", 18, 32)
TP_LAYOUT = ("kv", 64, 0)


class _Alloc:
    """A pool that states its layout and is faithful about the consequence of
    being asked to restore into a different one."""

    def __init__(self, layout):
        self.layout = layout
        self.loaded = False

    def cpu_copy_layout(self):
        return self.layout

    def supports_mamba_cpu_copy(self):
        return True

    def get_cpu_copy(self, indices, mamba_indices=None):
        return {"n": int(indices.numel()), "layout": self.layout}

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        if kv_cache_cpu["layout"] != self.layout:
            raise IndexError("list index out of range")
        self.loaded = True


def _req(*, allocated=20, logical_len=21, rid="rid-890"):
    """A request stand-in that carries the real `Req` copy/restore methods."""
    from sglang.srt.managers.schedule_batch import Req

    req = types.SimpleNamespace(
        rid=rid,
        req_pool_idx=0,
        seqlen=logical_len,
        kv_allocated_len=allocated,
        mamba_pool_idx=MAMBA_SLOT,
        mamba_state_cpu=None,
        mamba_state_cpu_layout=None,
        kv_cache_cpu=None,
        kv_cache_cpu_extent=None,
        kv_cache_cpu_layout=None,
        # The seam's own stamp, and the restore evidence the premise reads.
        seam_readmit_epoch=7,
        cache_protected_len=0,
        cached_prompt_tokens_at_retract=logical_len - 1,
        origin_input_ids=list(range(logical_len - 1)),
    )
    setattr(req, SEAM_RESTORE_REFUSED_ATTR, False)
    req.offload_kv_cache = types.MethodType(Req.offload_kv_cache, req)
    req.load_kv_cache = types.MethodType(Req.load_kv_cache, req)
    req._mamba_cpu_copy_is_mine = types.MethodType(Req._mamba_cpu_copy_is_mine, req)
    rtp = types.SimpleNamespace(
        req_to_token=torch.zeros((1, 64), dtype=torch.int64),
        mamba_pool=None,
        translate_mamba_indices=lambda ids: ids,
    )
    return req, rtp


def _copy_then_restore(req, rtp, *, from_layout, into_layout, seqlen_at_restore=None):
    """Take the copy in one layout and offer it back in another. Returns the
    restore's own verdict."""
    from sglang.srt.managers.schedule_batch import restore_seam_state

    req.offload_kv_cache(rtp, _Alloc(from_layout))
    if seqlen_at_restore is not None:
        req.seqlen = seqlen_at_restore
        req.kv_allocated_len = seqlen_at_restore - 1
    return restore_seam_state(req, rtp, _Alloc(into_layout))


def _scheduler(reqs, *, purity=None):
    """The minimum a purity gate reads: a queue, a phase, and a rule. Shaped
    like `_Sched` in test_seam_transport_exemption_w30.py so both files put the
    gate in the same state."""
    return types.SimpleNamespace(
        waiting_queue=list(reqs),
        _phase_purity=purity or PhasePurity(mode=MODE_STRICT),
        phase_flip_active_stack=PHASE_TP,
        phase_policy_cfg=None,
        server_args=types.SimpleNamespace(
            enable_phase_flip=True,
            phase_flip_purity=None,
            chunked_prefill_size=4096,
        ),
    )


# ---------------------------------------------------------------- THE FIX


class TestARefusedRestoreRevokesTheExemption(CustomTestCase):
    """The grant rested on 'this recomputes nothing'. The refusal is the proof
    that it does."""

    def test_a_layout_refusal_marks_the_request(self):
        req, rtp = _req()
        self.assertFalse(
            _copy_then_restore(req, rtp, from_layout=PP_LAYOUT, into_layout=TP_LAYOUT)
        )
        self.assertTrue(getattr(req, SEAM_RESTORE_REFUSED_ATTR))

    def test_an_extent_refusal_marks_the_request(self):
        req, rtp = _req()
        self.assertFalse(
            _copy_then_restore(
                req,
                rtp,
                from_layout=TP_LAYOUT,
                into_layout=TP_LAYOUT,
                seqlen_at_restore=31,
            )
        )
        self.assertTrue(getattr(req, SEAM_RESTORE_REFUSED_ATTR))

    def test_the_premise_no_longer_holds_for_a_refused_request(self):
        """THE REVOCATION. The evidence field survives the refusal untouched --
        the recompute the refusal forces even re-stamps it -- so the premise
        must key on the OUTCOME, not only on the evidence."""
        req, rtp = _req()
        _copy_then_restore(req, rtp, from_layout=PP_LAYOUT, into_layout=TP_LAYOUT)
        # The evidence the grant reads is still there. That is the point.
        self.assertGreater(int(req.cached_prompt_tokens_at_retract), 0)
        self.assertFalse(seam_transport_premise_holds(_scheduler([req])))

    def test_the_gate_closes_on_the_next_round_after_a_refused_restore(self):
        """End to end through the real gate: strict purity, TP layout, a
        stamped request whose restore was refused -> prefill stays blocked."""
        req, rtp = _req()
        _copy_then_restore(req, rtp, from_layout=PP_LAYOUT, into_layout=TP_LAYOUT)
        # Re-stamped by the next cutover, as the seam does unconditionally.
        setattr(req, SEAM_READMIT_ATTR, 8)
        self.assertTrue(prefill_blocked_here(_scheduler([req]), running_bs=0))


class TestTheRevocationIsNotPermanent(CustomTestCase):
    """A restore that works re-establishes the premise. Otherwise one bad flip
    would exile a request from the exemption for the rest of its life."""

    def test_a_successful_restore_clears_an_earlier_refusal(self):
        req, rtp = _req()
        _copy_then_restore(req, rtp, from_layout=PP_LAYOUT, into_layout=TP_LAYOUT)
        self.assertTrue(getattr(req, SEAM_RESTORE_REFUSED_ATTR))
        req.seqlen = 21
        req.kv_allocated_len = 20
        self.assertTrue(
            _copy_then_restore(req, rtp, from_layout=TP_LAYOUT, into_layout=TP_LAYOUT)
        )
        self.assertFalse(getattr(req, SEAM_RESTORE_REFUSED_ATTR))
        self.assertTrue(seam_transport_premise_holds(_scheduler([req])))


# ------------------------------------------------- THE DANGEROUS DIRECTION


class TestTheW30ExemptionStillWorks(CustomTestCase):
    """CAN-FAIL. A revocation that fired on the ordinary population would put
    the W30 livelock back: 150 flips, zero decode batches, every request timing
    out at 600 s."""

    def test_an_unrefused_restore_premise_still_holds(self):
        req, _ = _req()
        self.assertTrue(seam_transport_premise_holds(_scheduler([req])))

    def test_a_successful_restore_never_sets_the_mark(self):
        req, rtp = _req()
        self.assertTrue(
            _copy_then_restore(req, rtp, from_layout=TP_LAYOUT, into_layout=TP_LAYOUT)
        )
        self.assertFalse(getattr(req, SEAM_RESTORE_REFUSED_ATTR))

    def test_a_request_that_never_carried_a_copy_is_not_marked(self):
        """`restore_seam_state` runs for EVERY request in an extend batch, and
        for almost all of them `kv_cache_cpu` is None. Marking that path would
        revoke the exemption for the whole world."""
        from sglang.srt.managers.schedule_batch import restore_seam_state

        req, rtp = _req()
        self.assertIsNone(req.kv_cache_cpu)
        self.assertFalse(restore_seam_state(req, rtp, _Alloc(TP_LAYOUT)))
        self.assertFalse(getattr(req, SEAM_RESTORE_REFUSED_ATTR))
        self.assertTrue(seam_transport_premise_holds(_scheduler([req])))

    def test_one_refused_request_does_not_revoke_the_others(self):
        """NOT A BLANKET DISARM. The premise is an OR over the stamped
        population; only the request that actually failed loses its evidence."""
        bad, rtp_bad = _req(rid="bad")
        good, _ = _req(rid="good")
        _copy_then_restore(bad, rtp_bad, from_layout=PP_LAYOUT, into_layout=TP_LAYOUT)
        self.assertTrue(seam_transport_premise_holds(_scheduler([bad, good])))

    def test_a_population_of_only_refused_requests_revokes(self):
        bad_a, rtp_a = _req(rid="bad-a")
        bad_b, rtp_b = _req(rid="bad-b")
        _copy_then_restore(bad_a, rtp_a, from_layout=PP_LAYOUT, into_layout=TP_LAYOUT)
        _copy_then_restore(bad_b, rtp_b, from_layout=PP_LAYOUT, into_layout=TP_LAYOUT)
        self.assertFalse(seam_transport_premise_holds(_scheduler([bad_a, bad_b])))


class TestTheRefusalAnnouncementIsEdgeTriggered(CustomTestCase):
    """The module's own rule, stated at `_drain_yield_announced`: *"Cleared on
    recovery, so a flapping rig logs each engagement rather than only the first
    in the process's life."* The premise refusal latched instead -- which would
    have made this very fix observable exactly once per process."""

    def test_the_refusal_flag_re_arms_once_the_premise_holds_again(self):
        bad, rtp = _req(rid="bad")
        _copy_then_restore(bad, rtp, from_layout=PP_LAYOUT, into_layout=TP_LAYOUT)
        sched = _scheduler([bad])
        self.assertFalse(seam_transport_premise_holds(sched))
        self.assertTrue(getattr(sched, "_seam_premise_refused_announced", False))
        good, _ = _req(rid="good")
        sched.waiting_queue = [good]
        self.assertTrue(seam_transport_premise_holds(sched))
        self.assertFalse(getattr(sched, "_seam_premise_refused_announced", True))


class TestTheHypotheticalProbeAsksTheSameTwoQuestions(CustomTestCase):
    """SIBLING SWEEP (#890 §3). `Scheduler._purity_allows("prefill_in_tp")` is
    the fourth caller of this judgement -- the hypothetical `target_can_admit`
    the policy uses to decide whether the OTHER layout could take the work. The
    real gate opens on `seam_transport_exempt AND seam_transport_premise_holds`;
    the probe asked only for the stamp. A probe more permissive than the gate
    reports `target_can_admit=True` for a round in which nothing can be built,
    which is the arm auditor's wedge signature and the W33 defect this site was
    already fixed for once."""

    @staticmethod
    def _probe_sched(reqs):
        from sglang.srt.managers.scheduler import Scheduler

        sched = _scheduler(reqs)
        sched.phase_policy_cfg = None
        sched._purity_allows = types.MethodType(Scheduler._purity_allows, sched)
        return sched

    def test_a_stamped_population_with_a_LIVE_premise_still_reads_admissible(self):
        """CAN-FAIL: the W33 fix must survive. A genuine transport population
        still makes the TP layout hypothetically admissible."""
        good, _ = _req(rid="good")
        self.assertTrue(
            self._probe_sched([good])._purity_allows("prefill_in_tp", running_bs=0)
        )

    def test_a_stamped_population_whose_restore_was_REFUSED_is_not_admissible(self):
        bad, rtp = _req(rid="bad")
        _copy_then_restore(bad, rtp, from_layout=PP_LAYOUT, into_layout=TP_LAYOUT)
        self.assertFalse(
            self._probe_sched([bad])._purity_allows("prefill_in_tp", running_bs=0)
        )

    def test_the_probe_and_the_gate_never_disagree(self):
        """THE PROPERTY, stated directly: for the same scheduler state, a probe
        that says the layout can admit must not sit beside a gate that blocks.
        This is the invariant, not the two cases above."""
        for reqs, label in (
            ([_req(rid="live")[0]], "premise holds"),
            ([self._refused("dead")], "premise revoked"),
        ):
            with self.subTest(label):
                probe = self._probe_sched(reqs)._purity_allows(
                    "prefill_in_tp", running_bs=0
                )
                gate = not prefill_blocked_here(_scheduler(reqs), running_bs=0)
                self.assertEqual(probe, gate, label)

    @staticmethod
    def _refused(rid):
        req, rtp = _req(rid=rid)
        _copy_then_restore(req, rtp, from_layout=PP_LAYOUT, into_layout=TP_LAYOUT)
        return req


class TestTheAttributeNameCannotDriftApart(CustomTestCase):
    """The write site names the attribute by literal, exactly as the seam stamp
    does. A getattr default turns a rename into a feature that silently never
    fires -- the #684 NameError shape this module records."""

    def test_the_field_is_declared_on_req(self):
        import inspect

        from sglang.srt.managers.schedule_batch import Req

        self.assertIn(
            f"self.{SEAM_RESTORE_REFUSED_ATTR} = False",
            inspect.getsource(Req.__init__),
        )

    def test_the_write_site_uses_the_same_name(self):
        import inspect

        from sglang.srt.managers.schedule_batch import restore_seam_state

        src = inspect.getsource(restore_seam_state)
        self.assertIn(SEAM_RESTORE_REFUSED_ATTR, src)

    def test_the_premise_reads_it(self):
        import inspect

        src = inspect.getsource(seam_transport_premise_holds)
        self.assertIn("SEAM_RESTORE_REFUSED_ATTR", src)


if __name__ == "__main__":
    unittest.main()
