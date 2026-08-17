# SPDX-License-Identifier: Apache-2.0
"""PP stages forward control requests before handling them (#633).

Ported from upstream sgl-project/sglang#33934, which was still OPEN with red
CI when this landed, so the fix is re-derived and re-tested here rather than
cherry-picked.

The defect: some control requests block until every rank joins.
``InitWeightsUpdateGroupReqInput`` waits for the new process group to come up.
A stage that runs its LOCAL handler before forwarding the request onward
starts waiting for peers that were never told to join, while the downstream
stage never receives the request because its upstream is parked inside the
handler. Adjacent stages wait on each other and the boot hangs instead of
failing.

Two things have to be true for the fix to hold, and the second is the one a
port gets wrong:

1. the helper forwards before it processes, and
2. every PP event loop actually goes THROUGH the helper.

Upstream's test covers (1) only. A port that adds the helper and converts two
of the three loops would pass (1) and still deadlock on the third. That
matters here specifically: our PP prefill group runs the DISAGGREGATED loops,
not the standard one, so the disagg loops are the ones we cannot afford to
miss. The source-level test below covers (2) for all three.
"""

import inspect
import re
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    InitWeightsUpdateGroupReqInput,
)
from sglang.srt.managers.scheduler_pp_mixin import (  # noqa: E402
    SchedulerPPMixin,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_LOOPS = (
    "event_loop_pp",
    "event_loop_pp_disagg_prefill",
    "event_loop_pp_disagg_decode",
)


def _loop_source(name: str) -> str:
    """A loop's source, PLUS the body it delegates to.

    #679 follow-up, and it repairs a pin rather than relaxing one.
    ``event_loop_pp`` was refactored to set ``_defer_flip_round_to_pp_loop``
    and call ``_event_loop_pp_body``; the helper call the pin looks for moved
    one frame down with it. The property -- every PP loop routes received
    requests through the forward-first helper, or adjacent stages deadlock --
    was never lost, so the fix is to follow the delegation, not to drop the
    assertion. A pin that stops reaching the code it guards is worse than no
    pin, because it reads as protection while protecting nothing.
    """
    src = inspect.getsource(getattr(SchedulerPPMixin, name))
    for delegate in re.findall(r"self\.(_event_loop_\w*body)\(", src):
        fn = getattr(SchedulerPPMixin, delegate, None)
        if fn is not None:
            src += "\n" + inspect.getsource(fn)
    return src


def _blocking_req():
    """The request whose handler blocks on peers -- the deadlock's trigger."""
    return InitWeightsUpdateGroupReqInput(
        master_address="127.0.0.1",
        master_port=12345,
        rank_offset=0,
        world_size=2,
    )


def _pp_stub() -> SchedulerPPMixin:
    """A mixin instance with the flags the helper reads pinned OFF.

    #679 follow-up. ``_pp_forward_and_process_input_requests`` now consults
    ``pp_phase_flip_armed()``, which reads ``server_args.enable_phase_flip``,
    so a bare ``SchedulerPPMixin()`` raises AttributeError before the ordering
    this class exists to assert is ever exercised. The flip is irrelevant here
    -- the contract under test is forward-before-process -- so it is pinned
    off rather than simulated, which is what the sibling gate test does with
    the same flag for the same reason.
    """
    scheduler = SchedulerPPMixin()
    scheduler.server_args = SimpleNamespace(enable_phase_flip=False)
    return scheduler


class PpForwardBeforeProcessTest(CustomTestCase):
    """(1) The helper's ordering contract."""

    def test_non_last_stage_forwards_then_processes(self):
        scheduler = _pp_stub()
        scheduler.pp_group = SimpleNamespace(is_last_rank=False)
        previous_send_work = ["previous-send"]
        current_send_work = ["current-send"]
        scheduler.send_req_work = previous_send_work
        recv_reqs = [_blocking_req()]

        # One Mock as the recorder, so ORDER across the three calls is
        # asserted rather than each call merely having happened.
        calls = Mock()
        scheduler._pp_commit_comm_work = calls.commit
        scheduler._pp_send_pyobj_to_next_stage = calls.forward
        scheduler.process_input_requests = calls.process
        calls.forward.return_value = current_send_work

        scheduler._pp_forward_and_process_input_requests(recv_reqs)

        self.assertEqual(
            calls.mock_calls,
            [
                call.commit(previous_send_work),
                call.forward(recv_reqs, async_send=True),
                call.process(recv_reqs),
            ],
            "the local handler must run AFTER the forward, or adjacent stages "
            "wait on each other",
        )
        self.assertIs(scheduler.send_req_work, current_send_work)

    def test_last_stage_has_nobody_to_forward_to(self):
        scheduler = _pp_stub()
        scheduler.pp_group = SimpleNamespace(is_last_rank=True)
        scheduler.send_req_work = []

        calls = Mock()
        scheduler._pp_commit_comm_work = calls.commit
        scheduler._pp_send_pyobj_to_next_stage = calls.forward
        scheduler.process_input_requests = calls.process

        recv_reqs = [_blocking_req()]
        scheduler._pp_forward_and_process_input_requests(recv_reqs)

        self.assertEqual(calls.mock_calls, [call.process(recv_reqs)])


class PpLoopsUseTheHelperTest(CustomTestCase):
    """(2) Mechanism reach: the loops actually route through the helper.

    Read from the source of each loop, because that is the property a port
    can silently lose. Calling the loops for real is not an option -- they are
    unbounded ``while True`` event loops over live process groups.
    """

    def test_every_pp_loop_calls_the_helper(self):
        for name in _LOOPS:
            with self.subTest(loop=name):
                src = _loop_source(name)
                self.assertIn(
                    "_pp_forward_and_process_input_requests(recv_reqs)",
                    src,
                    f"{name} does not route its received requests through the "
                    "forward-first helper, so it can still deadlock",
                )

    def test_no_pp_loop_processes_requests_directly(self):
        """The old call site must be GONE, not merely accompanied.

        Leaving a direct ``process_input_requests(recv_reqs)`` behind in a loop
        would both re-introduce the early block and double-process the batch.
        """
        for name in _LOOPS:
            with self.subTest(loop=name):
                src = _loop_source(name)
                self.assertNotRegex(
                    src,
                    r"(?<!_)\bself\.process_input_requests\(",
                    f"{name} still calls process_input_requests directly",
                )

    def test_recv_reqs_are_not_forwarded_a_second_time(self):
        """The disagg loops used to re-send recv_reqs later in the body.

        The helper now owns that send. If the later send survived the port,
        every control request would cross the stage boundary twice.
        """
        for name in ("event_loop_pp_disagg_prefill", "event_loop_pp_disagg_decode"):
            with self.subTest(loop=name):
                src = inspect.getsource(getattr(SchedulerPPMixin, name))
                self.assertIsNone(
                    re.search(
                        r"_pp_send_pyobj_to_next_stage\(\s*recv_reqs", src, re.MULTILINE
                    ),
                    f"{name} forwards recv_reqs a second time; the helper "
                    "already sent them",
                )


if __name__ == "__main__":
    unittest.main()
