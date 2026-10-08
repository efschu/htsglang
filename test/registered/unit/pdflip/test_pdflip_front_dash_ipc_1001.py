"""DASHBOARD-IPC 01.10. (dashboard redesign, user 01.10. ~07:20Z / ~07:40Z: phase
live, flip time in the user's sense, TTFT per route with its parts, a session
view -- "Daten in die Zeitreihen-DB statt grep"): the front's IPC fields.

A. ``front.flip`` (Vorlauf / Layer / Nachlauf) and ``front.d_activity``, live.
   BUG (NF + 27B, 3-12 per boot): a P->D ``flip_first_work`` (decode_token)
   fired 0.02-0.99 s after ``flip_begin``, before ``flip_done``: ANY chunk of
   ANY open D stream counted -- NF 01.10. 05:27:47.638 epoch 26: pdflip-18-137
   (a wait-bound-parked stream of epoch 18) 153 ms after the begin, done at
   +2.2 s. Now D content before ``done`` counts only for a leg 2 dispatched
   in this flip (the dormant-admit hand-off: pdflip-3-10 D-ADMIT 64 ms after
   the begin, first token 2.9 s after P's end); the event carries
   ``leg2_dispatch_ts``, ``p_end_ts`` and ``flip_user_ms``.
B. ``front.ttft_by_via`` = {after_p|d_direct|d_single: n, ms_sum, ms_max,
   last_ms, last_ts, queue/p_prefill/flip_wait/d_first_token/other _ms_sum}.
C. ``request_done`` per finished request (events.jsonl, IPC thread), optional
   Influx point ``pdflip_req`` (FLLIPER_PDFLIP_METRICS_PUSH_URL, default off).
D. ``park`` per park episode.

RED on 7b8a2a41a1: the stale chunk fires ``flip_first_work``; the rest does
not exist (front_requests, FirstWorkClock.seen(leg2_dispatch_ts=...)).
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import front as front_mod  # noqa: E402
from flliper.srt.pdflip import front_state_ipc as fsi  # noqa: E402
from flliper.srt.pdflip import state_file  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402
from flliper.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _front():
    return front_mod.Front(
        prefill="http://p", decode="http://d", awake="D", tag="dashipc1001",
        store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
        weight_chunks=2, flip_min_work_tokens=1)


def _boot(root):
    return state_file.init(root, "nfdash-boot-20261001T090000Z-1001", "boot", {})


def _of(sd, typ, n=1, wait=3.0):
    deadline = time.time() + wait
    while True:
        got = [e for e in state_file.events(sd) if e["type"] == typ]
        if len(got) >= n or time.time() > deadline:
            return got
        time.sleep(0.02)


class TestFirstWorkStale(CustomTestCase):
    """A. the P->D first work is the flip's work, never a stale chunk."""

    def test_a_stale_chunk_before_done_is_not_the_flips_first_work(self):
        # NF 01.10. epoch 26: a P->D flip begins; 153 ms later a chunk of the
        # wait-bound-parked pdflip-18-137 (leg 2 of epoch 18) reaches the front.
        f = _front()
        pub = []
        f._ipc_publish = lambda typ, data: pub.append((typ, data))
        t0 = time.time()
        f._ipc_first_work_clock().arm(26, "P", "D", t0)
        f._ipc_first_work_seen("D", "decode_token", "pdflip-18-137")
        self.assertEqual([p for p in pub if p[0] == "flip_first_work"], [])

    def test_the_handoff_dispatched_in_the_flip_counts_before_done(self):
        c = fsi.FirstWorkClock()
        c.note_p_end(100.0)                       # P's last leg 1 (SERVED group=P)
        c.arm(4, "P", "D", 100.001)               # begin 1 ms after
        self.assertIsNone(c.seen("D", "decode_token", "old", 100.15, leg2_dispatch_ts=90.0))
        self.assertIsNone(c.seen("D", "decode_token", "unknown", 100.2))  # dispatch unknown
        ev = c.seen("D", "decode_token", "pdflip-3-10", 102.888, leg2_dispatch_ts=100.065)
        self.assertEqual((ev["rid"], ev["before_done"], ev["stale_skipped"]), ("pdflip-3-10", True, 2))
        self.assertEqual((ev["p_end_ts"], ev["p_end_source"], ev["leg2_dispatch_ts"]),
                         (100.0, "p_leg1_end", 100.065))
        self.assertEqual((ev["flip_user_ms"], ev["flip_time_ms"]), (2888, 2887))
        self.assertEqual(c.stale_skipped, 2)

    def test_after_done_any_d_content_is_the_first_work(self):
        c = fsi.FirstWorkClock()
        c.arm(26, "P", "D", 200.0)
        self.assertIsNone(c.seen("D", "decode_token", "pdflip-18-137", 200.153, leg2_dispatch_ts=150.0))
        c.done(202.2)
        ev = c.seen("D", "decode_token", "pdflip-18-137", 202.5, leg2_dispatch_ts=150.0)
        self.assertEqual((ev["before_done"], ev["done_ts"], ev["p_end_source"]), (False, 202.2, "flip_begin"))
        self.assertEqual(ev["flip_user_ms"], 2500)

    def test_p_end_of_an_earlier_p_phase_is_not_this_flips(self):
        c = fsi.FirstWorkClock()
        c.note_p_end(10.0)
        c.arm(2, "P", "D", 11.0)
        c.done(13.0)
        c.seen("D", "decode_token", "r", 14.0, leg2_dispatch_ts=11.5)
        c.arm(3, "D", "P", 20.0)
        c.done(22.0)
        c.seen("P", "p_leg1_dispatch", "r2", 22.5)
        c.arm(4, "P", "D", 40.0)                  # no leg 1 in this P phase
        c.done(42.0)
        ev = c.seen("D", "decode_token", "r3", 43.0, leg2_dispatch_ts=40.5)
        self.assertEqual((ev["p_end_ts"], ev["p_end_source"], ev["flip_user_ms"]), (None, "flip_begin", 3000))


#: FW-PING (NF fqnsdm 01.10.): D's keepalives and envelope, and two work chunks
_PING = b'event: ping\ndata: {"type": "ping"}\n\n'
_OAI_KEEPALIVE = b": keepalive\n\n"
_MSG_START = (b'event: message_start\ndata: {"type": "message_start", "message": '
              b'{"id": "msg_1", "usage": {"input_tokens": 0}}}\n\n')
_DELTA = (b'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, '
          b'"delta": {"type": "text_delta", "text": "Hi"}}\n\n')
_STOP = b'event: message_stop\ndata: {"type": "message_stop"}\n\n'


def _d_chunk(f, rid, chunk, path="/v1/messages"):
    """What leg 2's ``_write_client`` does with each D chunk it writes. Base
    1d3e0cd940 handed the hook no chunk (any 200 chunk counted) -- that form is
    the fallback, so a hook that loses the chunk again is red here."""
    try:
        f._ipc_first_work_seen("D", "decode_token", rid, chunk=chunk, path=path)
    except TypeError:
        f._ipc_first_work_seen("D", "decode_token", rid)


class TestFirstWorkNoPing(CustomTestCase):
    """FW-PING: a keepalive is never the flip's first work (P->D flips 28/30 on
    NF fqnsdm read 5.3 s from the 5 s ping of a stream parked in a long extend;
    D's first decode round came 0.4-0.5 s after done)."""

    def _armed_done(self):
        f = _front()
        pub = []
        f._ipc_publish = lambda typ, data: pub.append((typ, data))
        f._ipc_live_kick = lambda: None
        t0 = time.time()
        f._ipc_first_work_clock().note_p_end(t0 - 0.01)
        f._ipc_first_work_clock().arm(28, "P", "D", t0)
        f._ipc_first_work_clock().done(t0 + 0.2)
        return f, pub

    def test_ping_and_envelope_are_not_first_work_the_delta_is(self):
        f, pub = self._armed_done()
        for ch, path in ((_PING, "/v1/messages"), (_MSG_START, "/v1/messages"),
                         (_OAI_KEEPALIVE, "/v1/chat/completions")):
            _d_chunk(f, "pdflip-28-4", ch, path)
        self.assertEqual([p for p in pub if p[0] == "flip_first_work"], [])
        _d_chunk(f, "pdflip-28-4", _PING + _DELTA)
        fw = [p[1] for p in pub if p[0] == "flip_first_work"]
        self.assertEqual(len(fw), 1)
        self.assertEqual((fw[0]["epoch"], fw[0]["rid"], fw[0]["before_done"]), (28, "pdflip-28-4", False))

    def test_the_end_of_the_answer_counts(self):
        f, pub = self._armed_done()
        _d_chunk(f, "pdflip-28-5", _STOP)
        self.assertEqual(len([p for p in pub if p[0] == "flip_first_work"]), 1)

    def test_chunk_is_work_per_wire(self):
        w = front_mod.stream_chunk_is_work
        self.assertFalse(w(_PING, "/v1/messages"))
        self.assertFalse(w(_MSG_START, "/v1/messages"))
        self.assertFalse(w(_OAI_KEEPALIVE, "/v1/chat/completions"))
        self.assertFalse(w(b"", "/v1/messages"))
        self.assertTrue(w(_DELTA, "/v1/messages"))
        self.assertTrue(w(_STOP, "/v1/messages"))
        self.assertTrue(w(b'data: {"choices": [{"delta": {"content": "x"}}]}\n\n', "/v1/chat/completions"))


class TestFlipPhase(CustomTestCase):
    def test_vorlauf_layer_nachlauf_and_last(self):
        from flliper.srt.pdflip.front_requests import FlipPhase

        fp = FlipPhase()
        self.assertTrue(fp.vorlauf("D>P", "park:wait_bound", 100.0))
        self.assertFalse(fp.vorlauf("D>P", "backlog", 100.4))          # the park's decision stands
        self.assertEqual((fp.snap["phase"], fp.snap["reason"]), ("vorlauf", "park:wait_bound"))
        fp.layer("D>P", 100.66, "backlog")
        self.assertEqual((fp.snap["phase"], fp.snap["begin_ts"], fp.snap["decision_ts"], fp.snap["reason"]),
                         ("layer", 100.66, 100.0, "park:wait_bound"))
        fp.done(103.2)
        self.assertEqual((fp.snap["phase"], fp.snap["since_ts"]), ("nachlauf", 103.2))
        fp.first_work(103.5, "p_leg1_dispatch")
        s = fp.snap
        self.assertIsNone(s["phase"])
        self.assertEqual((s["last"]["vorlauf_ms"], s["last"]["layer_ms"], s["last"]["nachlauf_ms"],
                          s["last"]["decision_to_first_work_ms"]), (660, 2540, 300, 3500))

    def test_first_work_before_done_closes_at_done(self):
        from flliper.srt.pdflip.front_requests import FlipPhase

        fp = FlipPhase()
        fp.layer("P>D", 10.0, "handoff")                              # no vorlauf: zero-length
        fp.first_work(11.5, "decode_token")
        self.assertEqual(fp.snap["phase"], "layer")
        fp.done(12.0)
        self.assertIsNone(fp.snap["phase"])
        self.assertEqual((fp.snap["last"]["vorlauf_ms"], fp.snap["last"]["nachlauf_ms"]), (0, 0))

    def test_abort_names_why(self):
        from flliper.srt.pdflip.front_requests import FlipPhase

        fp = FlipPhase()
        fp.vorlauf("D>P", "park:x", 1.0)
        fp.abort(2.0, "front_stop")
        self.assertIsNone(fp.snap["phase"])
        self.assertEqual(fp.snap["last"]["aborted"], "front_stop")


class TestRequestBook(CustomTestCase):
    def _after_p(self, b):
        b.arrive("pdflip-3-10", 100.0, 3)
        b.stream("pdflip-3-10", True)
        b.session("pdflip-3-10", "853b45dbf2")
        b.leg1_dispatch("pdflip-3-10", 106.5)
        b.leg1_done("pdflip-3-10", 113.2, 15022, 13824, {"device": 13824, "host": 0, "storage": 0}, 1.1)
        b.leg2_dispatch("pdflip-3-10", 113.28)
        return b

    def test_ttft_parts_sum_to_the_ttft_per_via(self):
        from flliper.srt.pdflip.front_requests import RequestBook

        b = self._after_p(RequestBook(99.0))
        self.assertEqual(b.activity_block()["value"], "prefill")
        parts = b.first_token("pdflip-3-10", 117.65, "after_p")
        self.assertEqual(parts, {"ttft_ms": 17650, "queue_ms": 6500, "p_prefill_ms": 6700,
                                 "flip_wait_ms": 80, "d_first_token_ms": 4370, "other_ms": 0})
        self.assertIsNone(b.first_token("pdflip-3-10", 118.0, "after_p"))   # once
        v = b.ttft_block()["after_p"]
        self.assertEqual((v["n"], v["ms_sum"], v["ms_max"], v["last_ms"], v["last_ts"]),
                         (1, 17650.0, 17650.0, 17650.0, 117.65))
        self.assertEqual((v["queue_ms_sum"], v["p_prefill_ms_sum"], v["flip_wait_ms_sum"],
                          v["d_first_token_ms_sum"]), (6500.0, 6700.0, 80.0, 4370.0))
        self.assertEqual(b.ttft_block()["d_direct"]["n"], 0)
        self.assertEqual(b.activity_block()["value"], "decode")

    def test_request_done_record(self):
        from flliper.srt.pdflip.front_requests import RequestBook

        b = self._after_p(RequestBook(99.0))
        b.page_size = 64
        b.first_token("pdflip-3-10", 117.65, "after_p")
        b.d_served("pdflip-3-10", 15022, 15020, 72, {"device": 15020, "host": 0, "storage": 0}, handoff=True)
        b.d_prefill_s("pdflip-3-10", 0.004)
        b.leg2_end("pdflip-3-10", 118.985)
        rec, park = b.done("pdflip-3-10", 118.99, 200, 4, ("pdflip-3-8", 13866))
        self.assertIsNone(park)
        self.assertEqual((rec["session_id"], rec["turn"], rec["via"], rec["status"]),
                         ("853b45dbf2", 1, "after_p", 200))
        self.assertEqual((rec["ttft_ms"], rec["queue_ms"], rec["p_prefill_ms"], rec["flip_wait_ms"],
                          rec["d_first_token_ms"]), (17650, 6500, 6700, 80, 4370))
        self.assertEqual(rec["prefill"]["P"], {"ms": 1100, "ms_src": "pdflip_prefill_s", "wall_ms": 6700,
                                               "tokens": 1198, "prompt": 15022, "cached": 13824})
        self.assertEqual((rec["prefill"]["D"]["tokens"], rec["prefill"]["D"]["ms"]), (2, 4))
        self.assertEqual(rec["cached"], {"total": 15020, "device": 15020, "host": 0, "storage": 0,
                                         "told": 15020})
        self.assertEqual((rec["common_prefix"], rec["prev_rid"]), (13866, "pdflip-3-8"))
        self.assertEqual((rec["decode_tokens"], rec["decode_ms"], rec["context_tokens"], rec["kv_pages"]),
                         (72, 1340, 15094, 236))
        self.assertEqual((rec["flip_epochs"], rec["wall_s"], rec["legs"]), (1, 18.99, {"p": 1, "d": 1}))
        self.assertLess(len(json.dumps(rec)), state_file.EVENT_MAX - 500)
        self.assertEqual(b.activity_block()["value"], "idle")
        self.assertEqual(b.done("pdflip-3-10", 119.0, 200, 4), (None, None))   # gone

    def test_d_direct_and_turns(self):
        from flliper.srt.pdflip.front_requests import RequestBook

        b = RequestBook()
        for i, rid in enumerate(("a", "b")):
            b.arrive(rid, 10.0 + i, 5)
            b.session(rid, "s1")
            b.leg2_dispatch(rid, 10.5 + i)
            b.first_token(rid, 11.0 + i, "d_direct")
        v = b.ttft_block()["d_direct"]
        self.assertEqual((v["n"], v["ms_sum"], v["queue_ms_sum"], v["d_first_token_ms_sum"],
                          v["p_prefill_ms_sum"]), (2, 2000.0, 1000.0, 1000.0, 0.0))
        rec, _ = b.done("b", 12.0, 200, 5)
        self.assertEqual((rec["turn"], rec["via"], rec["flip_epochs"]), (2, "d_direct", 0))

    def test_park_episode_and_open_park_at_end(self):
        from flliper.srt.pdflip.front_requests import RequestBook

        b = RequestBook()
        b.arrive("r", 0.0, 1)
        b.est_prompt("r", 1000)
        b.page_size = 64
        b.leg2_dispatch("r", 1.0)
        b.first_token("r", 2.0, "d_direct")
        self.assertTrue(b.park("r", 5.0, "wait_bound:wait_bound", 1))
        self.assertFalse(b.park("r", 5.5, "again", 1))
        self.assertEqual(b.activity_block(), {"value": "idle", "since_ts": 5.0,
                                              "n": {"prefill": 0, "decode": 0, "parked": 1}})
        ev = b.resume("r", 9.0, "flip_to_d", 3)
        self.assertEqual((ev["park_ts"], ev["resume_ts"], ev["park_ms"], ev["pages"], ev["end"]),
                         (5.0, 9.0, 4000, 16, "flip_to_d"))
        self.assertIsNone(b.resume("r", 9.5, "again", 3))
        b.park("r", 10.0, "arrival_seat:youngest", 3)
        rec, open_park = b.done("r", 12.0, "cancelled", 3)
        self.assertEqual((open_park["resume_ts"], open_park["end"]), (None, "request_end"))
        self.assertEqual((rec["parks"], rec["resumes"], rec["park_ms"], rec["flip_epochs"]),
                         (2, 1, 6000, 2))

    def test_bounded(self):
        from flliper.srt.pdflip.front_requests import RequestBook

        b = RequestBook()
        b.MAX_ROWS = 3
        for i in range(5):
            b.arrive(f"r{i}", float(i), 0)
        self.assertEqual((list(b.rows), b.dropped), (["r2", "r3", "r4"], 2))


class TestFrontWiring(CustomTestCase):
    def test_handler_end_writes_request_done_and_park(self):
        sd = _boot(tempfile.mkdtemp(prefix="dash1001-"))
        f = _front()
        b = f._req_book()
        b.arrive("pdflip-1-1", time.time() - 2.0, 1)
        b.leg2_dispatch("pdflip-1-1", time.time() - 1.5)
        b.first_token("pdflip-1-1", time.time() - 1.0, "d_direct")
        b.park("pdflip-1-1", time.time() - 0.5, "wait_bound:x", 1)

        class _Req(dict):
            pass

        req = _Req()
        req[front_mod._hs.RID_KEY] = "pdflip-1-1"

        async def handler(request):
            return front_mod.web.Response(status=200)

        with mock.patch.dict(os.environ, {"PDFLIP_STATE_DIR": sd}):
            asyncio.run(f.ipc_out_wrap(handler)(req))
        done = _of(sd, "request_done")
        self.assertEqual(len(done), 1)
        d = done[0]["data"]
        self.assertEqual((d["rid"], d["status"], d["via"], d["parks"]), ("pdflip-1-1", 200, "d_direct", 1))
        park = _of(sd, "park")
        self.assertEqual((park[0]["data"]["rid"], park[0]["data"]["end"]), ("pdflip-1-1", "request_end"))

    def test_an_exception_is_a_named_status(self):
        f = _front()
        f._req_book().arrive("pdflip-1-2", time.time(), 1)
        got = []
        f._ipc_publish = lambda typ, data: got.append((typ, data))
        req = {front_mod._hs.RID_KEY: "pdflip-1-2"}

        async def handler(request):
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            asyncio.run(f.ipc_out_wrap(handler)(req))
        self.assertEqual([d["status"] for t, d in got if t == "request_done"], ["error:RuntimeError"])

    def test_front_fields_carry_flip_activity_ttft(self):
        f = _front()
        f._req_book().arrive("x", time.time() - 1, 0)
        f._req_book().leg2_dispatch("x", time.time() - 0.5)
        f._req_book().first_token("x", time.time(), "d_single")
        f._flip_phase().vorlauf("D>P", "park:x", time.time())
        fields = f._ipc_front_fields()
        self.assertFalse(set(fields) & set(fsi.HOST_FRONT_KEYS))
        self.assertEqual(fields["flip"]["phase"], "vorlauf")
        self.assertEqual(fields["d_activity"]["value"], "decode")
        self.assertEqual(fields["ttft_by_via"]["d_single"]["n"], 1)
        self.assertEqual(set(fields["ttft_by_via"]), {"after_p", "d_direct", "d_single"})
        if "arrival_seat" in fields:
            self.assertIs(fields["arrival_seat"]["ttft_by_via"], fields["ttft_by_via"])

    def test_live_kick_writes_flip_and_activity_without_ts(self):
        sd = _boot(tempfile.mkdtemp(prefix="dash1001-l-"))
        f = _front()
        with mock.patch.dict(os.environ, {"PDFLIP_STATE_DIR": sd}):
            f._flip_phase().layer("P>D", time.time(), "handoff")
            f._ipc_live_kick()
            f._ipc_live_kick()                          # coalesced
            deadline = time.time() + 3.0
            while time.time() < deadline and "flip" not in (state_file.read(sd).get("front") or {}):
                time.sleep(0.02)
        fr = state_file.read(sd)["front"]
        self.assertEqual(fr["flip"]["phase"], "layer")
        self.assertEqual(fr["d_activity"]["value"], "idle")
        self.assertNotIn("ts", fr)   # acc_state.stream_token_ts pairs front.ts with the rows' ages

    def test_push_only_with_env_and_rid_is_a_field(self):
        from flliper.srt.pdflip import front_metrics
        from flliper.srt.pdflip.front_requests import RequestBook, influx_req_fields

        b = RequestBook()
        b.arrive("pdflip-9-9", 1.0, 0)
        b.leg2_dispatch("pdflip-9-9", 1.2)
        b.first_token("pdflip-9-9", 1.5, "d_direct")
        rec, _ = b.done("pdflip-9-9", 2.0, 200, 0)
        tags, fields = influx_req_fields(rec)
        line = front_metrics.influx_line("pdflip_req", dict(tags, model="NF"), fields, ts_ns=2_000_000_000)
        head = line.split(" ", 1)[0]
        self.assertEqual(head, "pdflip_req,model=NF,status=200,via=d_direct")
        self.assertIn('rid="pdflip-9-9"', line)
        self.assertIn("ttft_ms=500i", line)

        f = _front()
        with envs.FLLIPER_PDFLIP_METRICS_PUSH_URL.override(None):
            self.assertIsNone(f._ipc_pusher())
        posted = []
        f2 = _front()
        with envs.FLLIPER_PDFLIP_METRICS_PUSH_URL.override("http://vm:8428/write"):
            p = f2._ipc_pusher()
            p._post = lambda url, body, t: posted.append((url, body))
            p.every_s = 0.0
            f2._ipc_req_push(rec)
        deadline = time.time() + 3.0
        while time.time() < deadline and not posted:
            time.sleep(0.02)
        self.assertEqual(posted[0][0], "http://vm:8428/write")
        self.assertTrue(posted[0][1].startswith(b"pdflip_req,"))


if __name__ == "__main__":
    unittest.main()
