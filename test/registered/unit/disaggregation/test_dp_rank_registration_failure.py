"""CPU unit tests: a failed prefill dp_rank registration must fail the room.

``_register_prefill_dp_rank`` posts this request's dp_rank to the bootstrap
server so the decode side can find it. A failure was only logged; the sender
then continued in Bootstrapping as if the decode side could resolve it, and the
request stalled until a timeout. The routing-conflict branch a few lines above
already records the failure and moves the room to ``KVPoll.Failed``.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVSender
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeKVManager:
    def __init__(self):
        self.attn_dp_rank = 2
        self.failures = []
        self.statuses = []

    def record_failure(self, room, message):
        self.failures.append((room, message))

    def update_status(self, room, status):
        self.statuses.append((room, status))


def _make_sender():
    sender = CommonKVSender.__new__(CommonKVSender)
    sender.kv_mgr = FakeKVManager()
    sender.bootstrap_room = 4711
    sender.bootstrap_server_url = "127.0.0.1:9999"
    return sender


class TestRegisterPrefillDpRank(CustomTestCase):
    def test_success_returns_true_and_does_not_fail_room(self):
        sender = _make_sender()
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.post",
            return_value=SimpleNamespace(status_code=200, text="ok"),
        ):
            self.assertTrue(sender._register_prefill_dp_rank())

        self.assertEqual(sender.kv_mgr.failures, [])
        self.assertEqual(sender.kv_mgr.statuses, [])

    def test_non_200_returns_false(self):
        sender = _make_sender()
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.post",
            return_value=SimpleNamespace(status_code=503, text="unavailable"),
        ):
            self.assertFalse(sender._register_prefill_dp_rank())

    def test_exception_returns_false(self):
        sender = _make_sender()
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.post",
            side_effect=OSError("connection refused"),
        ):
            self.assertFalse(sender._register_prefill_dp_rank())


class TestRegisterPrefillDpRankOrFail(CustomTestCase):
    def test_failure_records_failure_and_marks_room_failed(self):
        sender = _make_sender()
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.post",
            side_effect=OSError("connection refused"),
        ):
            ok = sender._register_prefill_dp_rank_or_fail()

        self.assertFalse(ok)
        self.assertEqual(len(sender.kv_mgr.failures), 1)
        room, message = sender.kv_mgr.failures[0]
        self.assertEqual(room, 4711)
        # The message must name the cause, not just that something failed.
        self.assertIn("dp_rank", message)
        self.assertIn("2", message)
        self.assertEqual(sender.kv_mgr.statuses, [(4711, KVPoll.Failed)])

    def test_success_leaves_room_untouched(self):
        sender = _make_sender()
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.post",
            return_value=SimpleNamespace(status_code=200, text="ok"),
        ):
            self.assertTrue(sender._register_prefill_dp_rank_or_fail())

        self.assertEqual(sender.kv_mgr.failures, [])
        self.assertEqual(sender.kv_mgr.statuses, [])


if __name__ == "__main__":
    unittest.main()
