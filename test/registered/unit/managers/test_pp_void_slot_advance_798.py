"""#798 -> #1015 / #1015c: the proxy channel's one-message-per-pass debt.

HISTORY, kept because the next reader will look for it. #798 (07744abcb7,
934239d931, 2026-08-22) found that whether a proxy message existed for a pass
was decided independently on the two ends of the wire (sender: iff
``self.mbs[mb_id]``; receiver: iff ``self.mbs[mb_id]``, else a drain gated on
the upstream's ``launched`` statement from the admission decision). A rank
holding resident work while its upstream launched nothing entered a blocking
receive for a message nobody posted and consumed the next pass's proxy. The
fix voided such a pass (``_pp_void_pass_without_upstream_launch``) before the
admission decision was forwarded. This file used to reproduce that with three
gloo processes driving ``_pp_send_admission_decision`` /
``_pp_drain_voided_proxy``, and to wire-check the void in the real loop.

WHAT REPLACED IT, and why those cases were retired rather than repaired.

* #1015 (01a391fa03, 2026-08-29) DELETED the admission-decision arc
  (``_pp_send_admission_decision``, ``_pp_recv_admission_decision``,
  ``_pp_try_recv_admission_decision``, ``_pp_void_retracted_pass``, ...) and
  made the proxy channel UNCONDITIONAL on both ends: a rank with no batch, or
  a prebuilt one, posts a void frame, so every pass has exactly one frame to
  take and a receive cannot block on a message nobody sent. The #798 defect is
  structurally impossible; the harness bound methods that no longer exist.
* #1015c (206661a28b) retired the downstream refusal itself: with its only
  writer deleted, ``_pp_upstream_launched_incoming`` was always False and
  ``pp_upstream_void_pending`` always True, so every downstream rank refused
  every pass (boot_pp3solo_01a391fa03_0829_112833: 0 prefill batches, 0
  served). It now answers False; the void decision stays with the rank that
  owns it (PP0-authoritative direction).

WHAT THIS FILE PINS NOW: the retirement, in both of its halves, so neither the
deleted arc nor the downstream refusal can come back unnoticed.
"""

import types
import unittest

from sglang.srt.managers.scheduler_pp_mixin import (
    SchedulerPPMixin,
    pp_upstream_void_pending,
)

WORLD = 3

#: deleted by #1015 (01a391fa03), named in its commit message. Not listed:
#: `_pp_reconcile_incoming_admission`, which #631 (287d5d3946) RESTORED for the
#: row authority -- it now reconciles the row arriving on the proxy frame,
#: whose receive is non-blocking by construction.
DELETED_BY_1015 = (
    "_pp_send_admission_decision",
    "_pp_recv_admission_decision",
    "_pp_try_recv_admission_decision",
    "_pp_commit_admission_send_work",
    "_pp_void_retracted_pass",
    "_pp_era_ring_live",
    "_pp_admission_send_work",
    "_pp_drain_voided_proxy",
)


def _downstream(batch, upstream_launched=False, streak=0):
    """A middle rank holding resident work -- #798's fourth combination."""
    h = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=1, pp_size=WORLD),
        pp_group=types.SimpleNamespace(is_first_rank=False, is_last_rank=False),
        _pp_gapped_wire=False,
        _pp_upstream_launched_incoming=upstream_launched,
        _pp_upstream_void_withheld_work=batch is not None,
        _pp_upstream_idle_void_streak=streak,
        _pp_upstream_idle_voids=0,
        _pp_idle_void_suppress_log=False,
        _pp_admission_pass_voided=False,
        _pp_admission_incoming_effective=None,
        _pp_admission_amended_to_forward=None,
        _pp_admission_incoming_schedule=None,
        mbs=[batch] * WORLD,
        mb_metadata=[None] * WORLD,
        running_mbs=[None] * WORLD,
        chunked_req=None,
        waiting_queue=[],
        _pp_chunked_req_before_by_slot=[None] * WORLD,
    )
    for name in ("_pp_void_pass_without_upstream_launch", "_pp_void_own_batch"):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


class TheAdmissionArcStaysDeleted1015(unittest.TestCase):
    def test_no_deleted_arc_member_is_back(self):
        back = [n for n in DELETED_BY_1015 if hasattr(SchedulerPPMixin, n)]
        self.assertEqual(
            back, [],
            "#1015 deleted the admission-decision arc and made the proxy "
            "channel unconditional; a member coming back re-opens the #798 "
            "surplus-receive family this file used to reproduce",
        )


class TheDownstreamRefusalIsRetired1015c(unittest.TestCase):
    def test_resident_work_without_an_upstream_launch_is_not_voided(self):
        """#798's fourth combination: this rank has a batch, the upstream flag
        reads False. Since #1015c the pass is not voided here -- the frame on
        the unconditional channel carries the upstream's statement."""
        batch = object()
        h = _downstream(batch, upstream_launched=False, streak=3)
        self.assertFalse(pp_upstream_void_pending(h))
        self.assertFalse(h._pp_void_pass_without_upstream_launch(0))
        self.assertIs(h.mbs[0], batch, "the slot was cleared by a retired void")
        self.assertFalse(h._pp_admission_pass_voided)
        self.assertEqual(h._pp_upstream_idle_void_streak, 0)

    def test_it_stays_inert_when_the_upstream_did_launch(self):
        batch = object()
        h = _downstream(batch, upstream_launched=True)
        self.assertFalse(h._pp_void_pass_without_upstream_launch(0))
        self.assertIs(h.mbs[0], batch)


if __name__ == "__main__":
    unittest.main()
