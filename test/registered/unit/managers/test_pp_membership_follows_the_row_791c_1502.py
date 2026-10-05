"""#791c, as it stands after #1072: a downstream rank EXECUTES PP0's row.

Replaces ``test_pp_proxy_retracted_pass_mispair_791c.py`` (auftrag 1502), whose
five cases drove ``_pp_note_output_expectation`` and
``_pp_pass_retraction_reason``. Both are gone: eed2b1fce2 (#969 CUT V, "the
travelling batch is the verdict") split the first, 6120d63bd4 (#1072c) deleted
the told/local congruence check the second read.

THE INVARIANT THE OLD FILE PROTECTED, in one sentence: a downstream rank's
batch MEMBERSHIP must equal the membership of the batch its upstream already
launched, because a batch in flight cannot be amended and its proxy is then
unpairable (boot instr17 2026-08-21: PP0 126 rows = 22 + 104, PP1 22 tokens,
PP1 had dropped the 104-token rid whose told prefix 16896 it did not hold).

THE OLD MECHANISM guarded it by REFUSING the proxy after the rank retracted.
TODAY'S MECHANISM removes the cause: ``reconcile_pp_admission_decision``
(pp_admission_congruence.py:2170) no longer retracts. A rank that holds less of
the prefix than PP0 told (``local < told``: the pipeline stagger, #1072c) or
cannot find the rid at all (a lookup miss) still puts the rid into
``effective`` with ``told`` (:2334, :2398). Membership is therefore the
decision's own, on every rank, and the proxy width matches by construction. The
remaining width guard is ``model_runner.forward``'s row check (model_runner.py,
the ``#631 PP proxy/batch mismatch`` ValueError).

WHAT THIS FILE PINS (no gloo, no process group: the question is a pure
decision-agreement question, as test_pp_admission_congruence_791.py says):

  1. the specimen's own decision, through the real wire codec and the real
     reconcile, loses no rid on the victim rank for either cause
     (``local < told`` and a lookup miss) and the victim's token count equals
     the upstream's proxy rows (126);
  2. an entry an EARLIER rank excluded is passed through verbatim and is not
     resurrected (the old "retraction by another rank does not refuse" half);
  3. the proxy receive guard in ``_pp_recv_proxy_tensors`` delivers a proxy
     whose identity is right on the shipped class (no ``_pp_pass_retraction_
     reason`` accessor: nothing to refuse), and still refuses with
     ``#791c PROXY BATCH DIVERGED`` when a holder reports a reason (the live
     raise path, kept for holders that supply the accessor).

CPU-only.
"""

import types
import unittest
from unittest import mock

import torch

from sglang.srt.managers import scheduler_pp_mixin as ppm
from sglang.srt.managers.pp_admission_congruence import (
    PPAdmissionDecision,
    PPAdmissionEntry,
    reconcile_pp_admission_decision,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

WORLD = 3
VICTIM = 1
LIVE_MB = 1
FLIP_EPOCH = 66
PROXY_SEQ = 4181

# The specimen's own rids and widths (boot instr17, see the module docstring).
RID_HONOURABLE = "51a294650b8b464495eda568e42530d7"
RID_UNHONOURABLE = "5e744c29f8de41fe96cb2c673b8582e5"
TOLD_PREFIX = 16896
HONOURABLE_EXTEND = 22
UNHONOURABLE_EXTEND = 104
UPSTREAM_ROWS = HONOURABLE_EXTEND + UNHONOURABLE_EXTEND  # 126


def _pp0_decision():
    return PPAdmissionDecision(
        mb_id=LIVE_MB,
        entries=(
            PPAdmissionEntry(
                rid=RID_HONOURABLE, prefix_len=0, extend_len=HONOURABLE_EXTEND
            ),
            PPAdmissionEntry(
                rid=RID_UNHONOURABLE,
                prefix_len=TOLD_PREFIX,
                extend_len=UNHONOURABLE_EXTEND,
            ),
        ),
    )


def _over_the_wire(decision):
    """PP0's decision as the victim receives it: the real codec both ways."""
    msg = ppm.pp_admission_decision_to_wire(decision)
    return ppm.pp_admission_decision_from_wire(msg)


def _victim_tokens(decision, effective):
    """The token count of the batch the victim builds: the extend lengths of
    exactly the rids it executes this pass."""
    return sum(e.extend_len for e in decision.entries if e.rid in effective)


class DownstreamRankExecutesTheRowTest(CustomTestCase):
    def _assert_victim_batch_is_the_upstream_batch(self, local_match_lens):
        decision = _over_the_wire(_pp0_decision())
        effective, amended = reconcile_pp_admission_decision(
            decision, local_match_lens, rank=VICTIM, pp_size=WORLD
        )
        self.assertEqual(
            effective,
            {RID_HONOURABLE: 0, RID_UNHONOURABLE: TOLD_PREFIX},
            "the victim dropped or re-measured a rid PP0 had already launched; "
            "its batch is then a strict subset of the upstream's (instr17)",
        )
        self.assertEqual(
            _victim_tokens(decision, effective),
            UPSTREAM_ROWS,
            "the victim's token count must equal the 126 rows the upstream's "
            "proxy carries",
        )
        for entry in amended.entries:
            self.assertTrue(entry.admitted, entry)
            self.assertFalse(entry.retracted, entry)
            self.assertIsNone(entry.retracted_by_rank, entry)

    def test_a_cache_shorter_than_told_still_executes_the_row(self):
        """BUG REGRESSION (instr17): the victim measured local=0 against
        told=16896 and dropped the rid, leaving a 22-token batch for a 126-row
        proxy. ``local < told`` is the pipeline stagger, not a shortfall."""
        self._assert_victim_batch_is_the_upstream_batch(
            {RID_HONOURABLE: 0, RID_UNHONOURABLE: 0}
        )

    def test_a_lookup_miss_still_executes_the_row(self):
        """The other drop path: the rid cannot be found on this rank at all
        (an empty map reads as UNKNOWN_MATCH). A miss is not a measurement."""
        self._assert_victim_batch_is_the_upstream_batch({})

    def test_an_entry_excluded_by_an_earlier_rank_is_passed_through_verbatim(self):
        """An entry an earlier rank (or PP0) already excluded is carried
        unchanged: not resurrected into ``effective``, not re-derived, not
        re-logged. Refusing a pass over it would break every pass downstream of
        any exclusion on the ring."""
        decision = _over_the_wire(_pp0_decision())
        excluded = PPAdmissionEntry(
            rid=RID_UNHONOURABLE,
            prefix_len=TOLD_PREFIX,
            extend_len=UNHONOURABLE_EXTEND,
            admitted=False,
            retracted=True,
            retracted_by_rank=2,
        )
        decision = PPAdmissionDecision(
            mb_id=LIVE_MB, entries=(decision.entries[0], excluded)
        )
        with self.assertNoLogs(
            "sglang.srt.managers.pp_admission_congruence", level="WARNING"
        ):
            effective, amended = reconcile_pp_admission_decision(
                decision, {}, rank=VICTIM, pp_size=WORLD
            )
        self.assertEqual(effective, {RID_HONOURABLE: 0})
        self.assertEqual(amended.entries[1], excluded)


def _proxy_holder(*, reason=None):
    """The shipped ``_pp_recv_proxy_tensors`` bound to a holder whose receive is
    one stamped proxy of the specimen's width. The stamp is right in every
    element (this slot, a sequence number, the true rows, this epoch): the
    guard under test is the one that runs when IDENTITY is perfect."""
    frame = {
        "__msg_type__": "proxy",
        "__stamp__": (LIVE_MB, PROXY_SEQ, UPSTREAM_ROWS, FLIP_EPOCH),
        "hidden_states": torch.zeros(UPSTREAM_ROWS, 4),
    }
    holder = types.SimpleNamespace(
        pp_group=types.SimpleNamespace(is_first_rank=False),
        _pp_gapped_wire=False,
        require_attn_tp_allgather=False,
        attn_tp_group=None,
        forward_ct=0,
        _pp_proxy_drops=0,
        _pp_flip_epoch=lambda: FLIP_EPOCH,
        _pp_wait_for_proxy_readiness=lambda mb_id: None,
        _pp_recv_typed_dict=lambda **kw: dict(frame),
    )
    if reason is not None:
        holder._pp_pass_retraction_reason = lambda mb_id: reason
    return holder


class ProxyReceiveGuardTest(CustomTestCase):
    def _recv(self, holder):
        with mock.patch.object(ppm, "_999_geom", return_value=None):
            return ppm.SchedulerPPMixin._pp_recv_proxy_tensors(holder, LIVE_MB)

    def test_the_shipped_class_has_no_reason_to_refuse_a_perfectly_stamped_proxy(self):
        """DEFAULT PATH. Nothing on the shipped class says this rank narrowed
        its pass, so the proxy is delivered; the width question is left to
        ``model_runner.forward``."""
        holder = _proxy_holder()
        proxy = self._recv(holder)
        self.assertEqual(int(proxy["hidden_states"].shape[0]), UPSTREAM_ROWS)
        self.assertFalse(hasattr(holder, "_pp_proxy_batch_divergences"))

    def test_a_reported_reason_refuses_with_the_791c_message(self):
        """The raise path stays live for any holder that reports a reason."""
        holder = _proxy_holder(reason="rid X told=16896 local=0")
        with self.assertRaises(RuntimeError) as cm:
            self._recv(holder)
        self.assertIn("#791c PROXY BATCH DIVERGED", str(cm.exception))
        self.assertIn("told=16896 local=0", str(cm.exception))
        self.assertEqual(holder._pp_proxy_batch_divergences, 1)


if __name__ == "__main__":
    unittest.main()
