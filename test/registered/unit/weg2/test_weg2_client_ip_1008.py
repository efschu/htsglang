"""08.10. (user: "ja einbauen"): the dashboard's session overview shows the ORIGIN IP of a session.  The front never recorded the peer;
now ``request_done.client_ip`` = the LAST address of ``X-Forwarded-For`` (the owui proxy appends the peer it saw; an XFF the client sent itself stays in front), else
``request.remote``.  Additive instrument next to the session hash (``_sess_note``): no routing, flip, admission or scheduler path.

RED before: ``session_trace.client_ip`` and ``RequestBook.client`` do not exist, ``request_done`` has no ``client_ip``.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.srt.weg2 import session_trace as st  # noqa: E402
from sglang.srt.weg2.front_requests import RequestBook  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeRequest:
    def __init__(self, headers=None, remote=None):
        self.headers = headers or {}
        self.remote = remote


def _front():
    return front_mod.Front(
        prefill="http://p", decode="http://d", awake="D", tag="clientip1008",
        store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
        weight_chunks=2, flip_min_work_tokens=1)


class TestClientIp(CustomTestCase):
    def test_last_forwarded_address_wins_then_the_peer(self):
        self.assertEqual(st.client_ip({"X-Forwarded-For": "6.6.6.6, 10.1.2.3"}, "127.0.0.1"), "10.1.2.3")
        self.assertEqual(st.client_ip({"x-forwarded-for": " 192.168.0.7 "}, "127.0.0.1"), "192.168.0.7")
        self.assertEqual(st.client_ip({}, "192.168.0.9"), "192.168.0.9")
        self.assertIsNone(st.client_ip({}, None))
        self.assertIsNone(st.client_ip(None, ""))
        self.assertEqual(len(st.client_ip({"X-Forwarded-For": "9" * 500}, None)), 64)         # bounded

    def test_request_done_carries_the_client_ip(self):
        f = _front()
        book = front_mod.Front._req_book(f)
        book.arrive("weg2-1-1", 100.0, 0)
        f._sess_note("weg2-1-1", FakeRequest({"X-Forwarded-For": "6.6.6.6, 10.1.2.3", "X-Claude-Code-Session-Id": "abc"}, "127.0.0.1"), {})
        rec, _ = book.done("weg2-1-1", 101.0, 200, 0)
        self.assertEqual(rec["client_ip"], "10.1.2.3")
        self.assertIsNotNone(rec["session_id"])                                           # the session note is untouched

    def test_peer_fallback_and_no_crash_without_headers(self):
        f = _front()
        book = front_mod.Front._req_book(f)
        for rid, req, want in (("weg2-1-2", FakeRequest({}, "192.168.0.5"), "192.168.0.5"),
                               ("weg2-1-3", FakeRequest({}, None), None),
                               ("weg2-1-4", object(), None),
                               ("weg2-1-5", None, None)):
            book.arrive(rid, 100.0, 0)
            f._sess_note(rid, req, {})                       # an instrument: never raises
            rec, _ = book.done(rid, 101.0, 200, 0)
            self.assertEqual(rec["client_ip"], want, rid)

    def test_the_book_alone_without_an_ip_writes_none(self):
        b = RequestBook(99.0)
        b.arrive("weg2-2-1", 100.0, 0)
        self.assertIsNone(b.done("weg2-2-1", 101.0, 200, 0)[0]["client_ip"])


if __name__ == "__main__":
    unittest.main()
