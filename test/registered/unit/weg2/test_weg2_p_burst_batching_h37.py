"""fnFL2 H37 (Task #118): many small prefills on group P.

Two halves, both hermetic (no GPU, no model, no boot):

1. THE LAUNCHER LEVER. Stock, the scheduler derives group P's per-forward
   request cap as ``max_running_requests // pp_size`` -- ONE request per forward
   for ``--p-bs`` 1..5 on PP3, so ``--p-bs 3`` pipelines three requests over
   three stages but never puts two in one forward, and every END-ANCHOR tail
   (1-4 tokens) runs as a forward of its own. With
   ``SGLANG_WEG2_ENABLE_P_UNDIVIDED_MICRO_BATCH`` the launcher emits
   ``--pp-max-micro-batch-size`` = P's EFFECTIVE ``--max-running-requests``
   (the value argparse keeps). Off: the argv is unchanged. The flag is an
   admission cap only and stays out of the ring form key.

2. THE PROBE (``weg2/tools/nf_burst_probe.py``): deterministic requests with
   one needle each, the log readers against the real line shapes (x144/x136),
   the verdict ladder NEIN / SCHWANZ / JA, and an end-to-end burst against a
   fake streaming front that checks the requests really were in flight
   together and that a swapped answer is a MISS.
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

try:
    from sglang.srt.environ import envs
    from sglang.srt.weg2 import launcher as L
    from sglang.srt.weg2 import ring_table
    from sglang.srt.weg2.tools import nf_burst_probe as bp
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

#: arm_fnFL2_long.sh EXTRA_P, shape-exact for the scheduling part.
NF_EXTRA_P = ["--max-total-tokens", "262144", "--kv-cache-dtype", "fp8_e4m3",
              "--page-size", "64", "--max-running-requests", "1", "--disable-cuda-graph"]


def _p(p_bs, extra):
    return L.argv_p(py="/nonexistent/python", model="/nonexistent/model",
                    budgets=[28208, 17840, 17168], s_gb=1, m_mib=600,
                    store_cfg=json.dumps({"max_size": "1"}), extra=list(extra),
                    p_bs=p_bs, p_max_total_tokens=1277631, draft_kv_on_p=False)


def _last(argv, flag):
    return L._last_flag_value(argv, flag)


class TestLauncherMicroBatch(unittest.TestCase):
    def test_off_emits_nothing(self):
        with envs.SGLANG_WEG2_ENABLE_P_UNDIVIDED_MICRO_BATCH.override(False):
            argv = _p(3, [])
        self.assertNotIn(L.P_MICRO_BATCH_FLAG, argv)
        self.assertIn("mode=stock width=1 ", L.p_micro_batch_line(argv))

    def test_on_width_is_p_bs_undivided(self):
        with envs.SGLANG_WEG2_ENABLE_P_UNDIVIDED_MICRO_BATCH.override(True):
            argv = _p(3, [])
        # the mutant `p_bs // pp_size` would ship 1 here -- the stock cap again
        self.assertEqual(_last(argv, L.P_MICRO_BATCH_FLAG), "3")
        self.assertEqual(_last(argv, "--max-running-requests"), "3")
        line = L.p_micro_batch_line(argv)
        self.assertIn("mode=undivided width=3 ", line)
        self.assertIn("stock would be 1", line)

    def test_width_follows_the_effective_max_running_requests(self):
        # the NF arm doubles --max-running-requests in --extra-p; argparse keeps
        # the LAST, so the width must follow that one, in both spellings.
        with envs.SGLANG_WEG2_ENABLE_P_UNDIVIDED_MICRO_BATCH.override(True):
            a = _p(1, ["--max-running-requests", "8"])
            b = _p(1, ["--max-running-requests=6"])
            c = _p(4, NF_EXTRA_P)
        self.assertEqual(_last(a, L.P_MICRO_BATCH_FLAG), "8")
        self.assertEqual(_last(b, L.P_MICRO_BATCH_FLAG), "6")
        self.assertEqual(_last(c, L.P_MICRO_BATCH_FLAG), "1")
        self.assertIn("DOUBLED --max-running-requests 4/1: argparse keeps 1", L.p_micro_batch_line(c))

    def test_extra_value_wins_and_is_not_doubled(self):
        with envs.SGLANG_WEG2_ENABLE_P_UNDIVIDED_MICRO_BATCH.override(True):
            argv = _p(8, ["--pp-max-micro-batch-size", "2"])
        self.assertEqual([t for t in argv if t == L.P_MICRO_BATCH_FLAG], [L.P_MICRO_BATCH_FLAG])
        self.assertEqual(_last(argv, L.P_MICRO_BATCH_FLAG), "2")

    def test_form_key_is_blind_to_the_flag(self):
        with envs.SGLANG_WEG2_ENABLE_P_UNDIVIDED_MICRO_BATCH.override(False):
            off = _p(3, [])
        with envs.SGLANG_WEG2_ENABLE_P_UNDIVIDED_MICRO_BATCH.override(True):
            on = _p(3, [])
        self.assertNotEqual(off, on)
        self.assertEqual(ring_table.p_form_key(off)[0], ring_table.p_form_key(on)[0])
        self.assertIn("--chunked-prefill-size", " ".join(ring_table.p_form_key(on)[1:]))

    def test_stock_line_names_the_trap(self):
        with envs.SGLANG_WEG2_ENABLE_P_UNDIVIDED_MICRO_BATCH.override(False):
            for p_bs, width in ((1, 1), (3, 1), (5, 1), (6, 2), (8, 2)):
                line = L.p_micro_batch_line(_p(p_bs, []))
                self.assertIn(f"mode=stock width={width} ", line, line)


# ------------------------------------------------------------------- the probe

X144_LINES = """\
[2026-09-24 10:31:46 PP0] Prefill batch, #new-seq: 1, #new-token: 12660, #cached-token: 0, full token usage: 0.05, mamba usage: 0.12, #running-req: 0, #queue-req: 0, #pending-token: 3, cuda graph: False, input throughput (token/s): 501.36
[2026-09-24 10:31:46 PP0] Prefill rank batch, #new-token: 12660, #cached-token: 0, #chunks: 1, gpu-ms: 6850.7 (compute 6850.7, wait 0.0) (wait by family: tp.all_reduce 0.0/29x) bubble_ms=108.2 (between forwards, mb=0)
[2026-09-24 10:31:46 PP1] Prefill batch, #new-seq: 1, #new-token: 12660, #cached-token: 0, full token usage: 0.05, mamba usage: 0.12, #running-req: 0, #queue-req: 0, #pending-token: 3, cuda graph: False, input throughput (token/s): 502.77
[2026-09-24 10:31:46 PP0] Prefill batch, #new-seq: 1, #new-token: 3, #cached-token: 0, full token usage: 0.00, mamba usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 35.23
[2026-09-24 10:31:46 PP0] Prefill rank batch, #new-token: 3, #cached-token: 0, #chunks: 1, gpu-ms: 202.5 (compute 202.2, wait 0.3) (wait by family: tp.all_reduce 0.3/29x)
[2026-09-24 10:32:08 PP0] Prefill batch, #new-seq: 1, #new-token: 8360, #cached-token: 0, full token usage: 0.03, mamba usage: 0.12, #running-req: 0, #queue-req: 0, #pending-token: 1, cuda graph: False, input throughput (token/s): 386.34
[2026-09-24 10:32:08 PP0] Prefill rank batch, #new-token: 8360, #cached-token: 0, #chunks: 1, gpu-ms: 2870.9 (compute 2870.9, wait 0.0) (wait by family: tp.all_reduce 0.0/29x) bubble_ms=217.1 (between forwards, mb=0)
[2026-09-24 10:32:08 PP0] Prefill batch, #new-seq: 1, #new-token: 1, #cached-token: 0, full token usage: 0.00, mamba usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 17.04
[2026-09-24 10:32:08 PP0] Prefill rank batch, #new-token: 1, #cached-token: 0, #chunks: 1, gpu-ms: 141.1 (compute 140.8, wait 0.3) (wait by family: tp.all_reduce 0.3/29x)
[2026-09-24 07:48:09 PP0] FWD-TIMING-PREFILL forward=1 tokens=16384 layers=29 embed_ms=2.3 ple_ms=1438.9 hc_ms=319.2 dense_ms=407.3 qsa_idx_ms=56.9 attn_ms=126.3 linear_ms=148.8 shared_ms=27.6 gate_ms=184.3 moe_plan_ms=1497.2 moe_fetch_ms=1607.3 moe_apply_ms=470.6 other_ms=0.2 total_ms=6287.0 marks=1194 (CUDA-event timeline around model.forward; segments telescope, sum == total; compare total_ms with the rank's gpu-ms)
"""

FRONT_LINES = """\
[2026-09-24 10:31:46,100] INFO weg2.front: WEG2-SERVED group=P leg=1 rid=weg2-2-6 prompt_tokens=12663 cached_tokens=0 wall=10.96s epoch=3
[2026-09-24 10:31:50,100] INFO weg2.front: WEG2-SERVED group=D leg=2 rid=weg2-2-6 status=200 prompt_tokens=12663 cached_tokens=12660 completion_tokens=48 uncached=3 verdict=serve wall=1.0s epoch=4
[2026-09-24 10:32:08,100] INFO weg2.front: WEG2-SERVED group=P leg=1 rid=weg2-3-7 prompt_tokens=8361 cached_tokens=0 wall=5.83s epoch=5
[2026-09-24 10:32:20,100] INFO weg2.front: WEG2-SERVED group=D leg=2 rid=weg2-4-8 status=200 prompt_tokens=3100 cached_tokens=0 completion_tokens=48 uncached=3100 verdict=serve wall=3.0s epoch=6
[2026-09-24 10:32:21,000] INFO weg2.front: WEG2-FLIP begin P->D
"""


def _res(idx, pt, code="C", text=None, status=200):
    return bp.RequestResult(idx, pt, code, status=status, t_send=10.0, t_first=11.0, t_done=12.0,
                            prompt_tokens=pt, text=code if text is None else text)


def _batch(seq, tok):
    return ("[2026-09-24 10:00:00 PP0] Prefill batch, #new-seq: %d, #new-token: %d, #cached-token: 0, "
            "full token usage: 0.01" % (seq, tok))


class TestProbeRequests(unittest.TestCase):
    def test_deterministic_unique_and_in_range(self):
        a = bp.build_requests(8, 37, 3000, 8000, "m", 48)
        b = bp.build_requests(8, 37, 3000, 8000, "m", 48)
        c = bp.build_requests(8, 38, 3000, 8000, "m", 48)
        self.assertEqual([(r.code, r.target_tokens, r.body) for r in a],
                         [(r.code, r.target_tokens, r.body) for r in b])
        self.assertNotEqual([r.code for r in a], [r.code for r in c])
        self.assertEqual(len({r.code for r in a}), 8)
        heads = {r.body["messages"][0]["content"].split("\n", 1)[0] for r in a}
        self.assertEqual(len(heads), 8)  # no shared radix prefix between requests
        sizes = {r.body["messages"][0]["content"].count("Satz ") for r in a}
        self.assertEqual(len(sizes), 8)  # distinct lengths -> front lines join by prompt_tokens
        for r in a:
            self.assertTrue(3000 <= r.target_tokens <= 8000)
            content = r.body["messages"][0]["content"]
            self.assertEqual(content.count(r.code), 1)
            self.assertTrue(r.body["stream"])
            # ~17.1 tokens per sentence: the body lands near its target
            est = content.count("Satz ") * bp.TOKENS_PER_SENTENCE + bp.PROMPT_OVERHEAD_TOKENS
            self.assertLess(abs(est - r.target_tokens), 60)


class TestProbeLogs(unittest.TestCase):
    def test_real_line_shapes(self):
        pw = bp.parse_p_log(X144_LINES)
        self.assertEqual(pw.batches[0], [(1, 12660, 0), (1, 3, 0), (1, 8360, 0), (1, 1, 0)])
        self.assertEqual(len(pw.batches[1]), 1)
        self.assertEqual(pw.rank_ms[0][0], (12660, 6850.7))
        self.assertEqual(pw.fwd[0][0]["moe_fetch"], 1607.3)
        fw = bp.parse_front_log(FRONT_LINES)
        self.assertEqual([(pt, ct, w) for _, pt, ct, w in fw.p_served], [(12663, 0, 10.96), (8361, 0, 5.83)])
        self.assertEqual(fw.flips, 1)

    def test_join_by_prompt_tokens_and_short_route(self):
        fw = bp.parse_front_log(FRONT_LINES)
        rs = [_res(0, 12663), _res(1, 8361), _res(2, 3100), _res(3, 999)]
        bp.attach_front(rs, fw)
        self.assertEqual([r.route for r in rs], ["P", "P", "D", "?"])
        self.assertEqual(rs[0].p_wall_s, 10.96)

    def test_verdict_nein_today(self):
        # x144 shape: body + tail per request, each its own forward
        pw = bp.parse_p_log(X144_LINES)
        fw = bp.parse_front_log(FRONT_LINES)
        rs = [_res(0, 12663), _res(1, 8361)]
        bp.attach_front(rs, fw)
        s = bp.summarize(rs, pw, fw, "burst")
        self.assertEqual((s["batches"], s["multi_seq_batches"], s["fwd_per_req"]), (4, 0, 2.0))
        self.assertEqual(s["verdict"], "NEIN")
        self.assertIsNotNone(s["sockel_ms"])
        self.assertEqual(s["tail_ms"][0], 171.8)
        self.assertEqual(s["moe_fetch_ms"], 1607.3)
        self.assertIsNotNone(s["serial_tok_s_est"])

    def test_verdict_tail_merged_is_not_batching(self):
        # END-ANCHOR with an undivided micro batch: tail_k + body_k+1 per forward
        text = "\n".join([_batch(1, 4000), _batch(2, 5003), _batch(2, 6002), _batch(1, 2)])
        s = bp.summarize([_res(i, 4000 + i) for i in range(3)], bp.parse_p_log(text), None, "burst")
        self.assertEqual((s["multi_seq_batches"], s["max_new_seq"]), (2, 2))
        self.assertGreaterEqual(s["fwd_per_req"], 1.0)
        self.assertEqual(s["verdict"], "SCHWANZ")

    def test_verdict_ja_needs_fewer_forwards_than_requests(self):
        text = "\n".join([_batch(3, 16384), _batch(4, 12000), _batch(1, 4)])
        rs = [_res(i, 4000 + i) for i in range(6)]
        s = bp.summarize(rs, bp.parse_p_log(text), None, "burst")
        self.assertEqual(s["verdict"], "JA")
        self.assertEqual(s["tok_per_fwd"], round((16384 + 12000 + 4) / 3, 1))
        line = bp.format_report(s, rs)[0]
        for key in ("BURST-PROBE n=6 ", " tokens=", " wall=", " agg_tok_s=", " batches=3 ",
                    " multi_seq_batches=2 ", " tok_per_fwd="):
            self.assertIn(key, line)

    def test_no_p_lines_is_named(self):
        s = bp.summarize([_res(0, 3100)], bp.parse_p_log(""), None, "burst")
        self.assertEqual(s["verdict"], "KEINE-P-ZEILEN")


class _FakeFront(BaseHTTPRequestHandler):
    """SSE front: answers with the code in the prompt (or a neighbour's)."""

    lock = threading.Lock()
    inflight = 0
    peak = 0
    gate = None
    swap = {}

    def log_message(self, *a):  # noqa: D401 - silence
        pass

    def do_POST(self):
        cls = type(self)
        n = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(n))
        content = body["messages"][0]["content"]
        code = content.split("Geheimcode dieser Anfrage: ", 1)[1].split(".", 1)[0]
        code = cls.swap.get(code, code)
        with cls.lock:
            cls.inflight += 1
            cls.peak = max(cls.peak, cls.inflight)
        cls.gate.wait(timeout=10)  # every request must be in flight before any answers
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for obj in ({"choices": [{"delta": {"role": "assistant"}}]},
                    {"choices": [{"delta": {"content": code}}]},
                    {"choices": [{"delta": {"content": " - ein Text."}}]},
                    {"choices": [], "usage": {"prompt_tokens": len(content) // 3, "completion_tokens": 5}}):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        with cls.lock:
            cls.inflight -= 1


class TestProbeEndToEnd(unittest.TestCase):
    def _run(self, n, swap, mode="burst"):
        _FakeFront.inflight = _FakeFront.peak = 0
        _FakeFront.gate = threading.Barrier(n if mode == "burst" else 1)
        _FakeFront.swap = swap
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeFront)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            reqs = bp.build_requests(n, 5, 300, 900, "m", 16)
            res = bp.run_requests("http://127.0.0.1:%d" % srv.server_address[1], reqs, mode, 30)
        finally:
            srv.shutdown()
            srv.server_close()
        return reqs, res

    def test_burst_is_concurrent_and_needles_match(self):
        reqs, res = self._run(4, {})
        self.assertEqual(_FakeFront.peak, 4)
        self.assertEqual([r.needle for r in res], ["MATCH"] * 4)
        self.assertTrue(all(r.ttft_s is not None and r.prompt_tokens for r in res))

    def test_swapped_answer_is_a_miss(self):
        codes = [r.code for r in bp.build_requests(3, 5, 300, 900, "m", 16)]
        _, res = self._run(3, {codes[0]: codes[1]})
        self.assertEqual([r.needle for r in res], ["MISS", "MATCH", "MATCH"])

    def test_serial_is_one_at_a_time(self):
        _, res = self._run(3, {}, mode="serial")
        self.assertEqual(_FakeFront.peak, 1)
        self.assertEqual([r.needle for r in res], ["MATCH"] * 3)


if __name__ == "__main__":
    unittest.main()
