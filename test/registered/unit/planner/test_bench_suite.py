# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Unit tests for bench_suite.py -- the #151 benchmark/quality suite core.

Hermetic: NO network, NO server. The HTTP transport is a fake fed with
canned OpenAI-style responses / SSE streams / Prometheus scrapes. Covers:

  * the 16-test catalog + presets + run ordering (Cliff-2 tests LAST),
  * the chat-template gating matrix (each missing dependency -> the right
    skip / warn / blocked outcome; force lifts only the tool-parser block),
  * verifier correctness per test on canned responses: a passing one, a
    <tool_call>-in-content cascade one, a repetition-cascade one, a
    truncated/empty one,
  * tri-state needle handling (recall miss on HTTP 200 -> info, never fail),
  * long-ctx graceful skip (max_model_len pre-check AND the HTTP 400 path),
  * MTP acceptance length off the pinned Prometheus gauge names,
  * engine-health recheck after crash-prone probes (remaining tests skip).
"""

import json
import random
import re
import unittest

from sglang.srt.planner.bench_suite import (
    CLIFF2_TESTS,
    Capabilities,
    HttpResult,
    PRESETS,
    TEST_CATALOG,
    analyze_output_quality,
    gate_test,
    make_needle_secret,
    needle_recall_ok,
    order_selected,
    probe_capabilities,
    reassemble_stream,
    request_agentic_turn,
    request_basic,
    request_needle,
    request_quality,
    request_reasoning_heavy,
    request_throughput,
    request_tool_call,
    request_tool_prefill,
    run_suite,
)
from sglang.srt.planner.live_metrics import SPEC_EMA_ACCEPT_LEN_METRIC


# ---------------------------------------------------------------------------
# Canned-response helpers.
# ---------------------------------------------------------------------------
def chat_response(content=None, tool_calls=None, reasoning=None,
                  finish="stop", prompt_tokens=50, completion_tokens=20):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return HttpResult(status=200, body=json.dumps({
        "choices": [{"message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_tokens,
                  "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }), wall_ms=100.0)


def sse_lines(deltas, usage=None, finish="stop"):
    """Build SSE lines from a list of delta dicts."""
    lines = []
    for i, delta in enumerate(deltas):
        chunk = {"choices": [{"delta": delta,
                              "finish_reason":
                                  finish if i == len(deltas) - 1 else None}]}
        lines.append("data: " + json.dumps(chunk))
    if usage is not None:
        lines.append("data: " + json.dumps({"choices": [], "usage": usage}))
    lines.append("data: [DONE]")
    return lines


def stream_response(deltas, usage=None, ttft_ms=80.0, wall_ms=1000.0):
    return HttpResult(status=200, sse_lines=sse_lines(deltas, usage=usage),
                      ttft_ms=ttft_ms, wall_ms=wall_ms)


class FakeHttp:
    """Route by URL path; chat responses are served by a handler that may
    inspect the request body (needle echo). Records every call."""

    def __init__(self, chat_handler=None, metrics_text="", models_ok=True):
        self.chat_handler = chat_handler
        self.metrics_text = metrics_text
        self.models_ok = models_ok
        self.calls = []

    def __call__(self, method, url, body=None, stream=False, timeout=None):
        self.calls.append({"method": method, "url": url, "body": body,
                           "stream": stream})
        if url.endswith("/v1/models"):
            if not self.models_ok:
                return HttpResult(status=0, error="connection refused")
            return HttpResult(status=200, body=json.dumps({
                "data": [{"id": "test-model", "max_model_len": 262144}]}))
        if url.endswith("/metrics"):
            return HttpResult(status=200, body=self.metrics_text)
        if url.endswith("/v1/chat/completions"):
            assert self.chat_handler is not None, "no chat handler configured"
            return self.chat_handler(body, stream)
        return HttpResult(status=404, body="not found")


def full_caps(**kw):
    base = dict(chat_template_basic=True, tool_parser="qwen3_coder",
                reasoning_parser="qwen3", streaming=True, spec_decode=True,
                spec_mode="mtp-adaptive-k", max_model_len=262144,
                model="test-model")
    base.update(kw)
    return Capabilities(**base)


def run_one(test_id, caps, http, **kw):
    results = list(run_suite("http://x:1", "test-model", [test_id],
                             capabilities=caps, http=http, **kw))
    assert len(results) == 1
    return results[0]


# ---------------------------------------------------------------------------
# Catalog / presets / ordering.
# ---------------------------------------------------------------------------
class TestCatalog(unittest.TestCase):
    def test_sixteen_tests(self):
        self.assertEqual(sorted(TEST_CATALOG), list(range(1, 17)))
        for spec in TEST_CATALOG.values():
            self.assertTrue(spec.label)
            self.assertIn(spec.chat_template,
                          ("basic", "tool-aware", "thinking-aware"))

    def test_presets(self):
        self.assertEqual(PRESETS["functional"], (1, 2, 3, 4, 5, 6, 7))
        self.assertEqual(PRESETS["stress"], (8, 9, 10, 11, 12, 13, 14, 15))
        self.assertEqual(PRESETS["throughput"], (16,))
        self.assertEqual(PRESETS["full"], tuple(range(1, 17)))

    def test_cliff2_ordered_last(self):
        self.assertEqual(order_selected(PRESETS["stress"]),
                         [8, 9, 10, 11, 12, 13, 14, 15])
        # In the FULL preset the throughput test must run BEFORE the
        # crash-capable Cliff-2 rungs.
        self.assertEqual(order_selected(PRESETS["full"]),
                         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                          16, 14, 15])
        # Cliff-2 goes last regardless of the selection order given.
        self.assertEqual(order_selected([14, 8, 15, 9]), [8, 9, 14, 15])
        self.assertEqual(CLIFF2_TESTS, frozenset({14, 15}))

    def test_deps_json_schema(self):
        deps = TEST_CATALOG[8].deps_json()
        self.assertEqual(
            set(deps),
            {"chat_template", "tools", "streaming", "thinking",
             "longctx_rung"},
        )
        self.assertEqual(deps["longctx_rung"], [10000, 30000])


# ---------------------------------------------------------------------------
# Verbatim request bodies (spot checks against the club-3090 scripts).
# ---------------------------------------------------------------------------
class TestRequestBodies(unittest.TestCase):
    def test_basic(self):
        b = request_basic("m")
        self.assertEqual(b["messages"][0]["content"],
                         "What is the capital of France? One short sentence.")
        self.assertEqual(b["max_tokens"], 30)
        self.assertEqual(b["temperature"], 0.6)
        self.assertEqual(b["chat_template_kwargs"],
                         {"enable_thinking": False})

    def test_tool_call(self):
        b = request_tool_call("m")
        self.assertEqual(b["tools"][0]["function"]["name"], "get_weather")
        self.assertEqual(b["tool_choice"], "auto")
        self.assertEqual(b["max_tokens"], 200)
        self.assertEqual(b["temperature"], 0.3)

    def test_agentic_turn(self):
        b = request_agentic_turn("m")
        self.assertEqual(len(b["tools"]), 10)
        self.assertEqual(b["tool_choice"], "required")
        self.assertEqual(b["max_tokens"], 150)
        self.assertTrue(b["stream"])

    def test_needle_secret_placement(self):
        b = request_needle("m", 150, "crimson otter 42")
        content = b["messages"][0]["content"]
        self.assertIn("The hidden phrase is 'crimson otter 42'", content)
        self.assertEqual(b["max_tokens"], 30)
        self.assertEqual(b["temperature"], 0.0)
        # secret sits mid-document, question at the end
        self.assertLess(content.find("hidden phrase is 'crimson"),
                        content.find("Question:"))

    def test_tool_prefill_size(self):
        b = request_tool_prefill("m", target_chars=100000)
        tool_msg = b["messages"][2]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertGreaterEqual(len(tool_msg["content"]), 100000)
        self.assertEqual(b["max_tokens"], 500)

    def test_reasoning_heavy(self):
        b = request_reasoning_heavy("m")
        self.assertEqual(b["max_tokens"], 8192)
        self.assertEqual(b["temperature"], 0.0)

    def test_throughput_prompts(self):
        narr = request_throughput("m", "narrative")
        code = request_throughput("m", "code")
        self.assertEqual(narr["max_tokens"], 1000)
        self.assertEqual(code["max_tokens"], 800)
        with self.assertRaises(ValueError):
            request_throughput("m", "nope")


# ---------------------------------------------------------------------------
# Gating matrix.
# ---------------------------------------------------------------------------
class TestGating(unittest.TestCase):
    def test_no_basic_template_blocks_everything(self):
        caps = full_caps(chat_template_basic=False)
        for spec in TEST_CATALOG.values():
            d = gate_test(spec, caps)
            self.assertEqual(d.status, "blocked", spec.key)

    def test_missing_tool_parser_blocks_tool_tests(self):
        caps = full_caps(tool_parser=None)
        for tid in (2, 4, 9, 10, 11):
            d = gate_test(TEST_CATALOG[tid], caps)
            self.assertEqual(d.status, "blocked", tid)
            self.assertIn("tool-call-parser", d.reason)
        # non-tool tests unaffected
        for tid in (1, 3, 6, 12, 13):
            self.assertIsNone(gate_test(TEST_CATALOG[tid], caps).status, tid)

    def test_force_lifts_tool_parser_block_only(self):
        caps = full_caps(tool_parser=None, chat_template_basic=False)
        # basic-template block holds even under force
        self.assertEqual(gate_test(TEST_CATALOG[2], caps, force=True).status,
                         "blocked")
        caps = full_caps(tool_parser=None)
        self.assertIsNone(gate_test(TEST_CATALOG[2], caps, force=True).status)

    def test_missing_reasoning_parser_is_expected_fail_not_block(self):
        caps = full_caps(reasoning_parser=None)
        d = gate_test(TEST_CATALOG[5], caps)
        self.assertIsNone(d.status)  # runs
        self.assertIn("reasoning-parser", d.expected_fail_note)

    def test_spec_off_skips_test_7(self):
        d = gate_test(TEST_CATALOG[7], full_caps(spec_decode=False))
        self.assertEqual(d.status, "skip")
        self.assertIsNone(gate_test(TEST_CATALOG[7], full_caps()).status)

    def test_longctx_rungs_above_max_model_len_skip_gracefully(self):
        caps = full_caps(max_model_len=32768)
        d14 = gate_test(TEST_CATALOG[14], caps)  # 60K/90K rungs
        self.assertEqual(d14.status, "skip")
        self.assertIn("graceful", d14.reason)
        d8 = gate_test(TEST_CATALOG[8], caps)    # 10K/30K rungs fit
        self.assertIsNone(d8.status)
        self.assertEqual(d8.rungs, [10000, 30000])
        # partial fit: only the 10K rung survives
        d8b = gate_test(TEST_CATALOG[8], full_caps(max_model_len=20000))
        self.assertIsNone(d8b.status)
        self.assertEqual(d8b.rungs, [10000])

    def test_no_caps_means_ungated(self):
        self.assertIsNone(gate_test(TEST_CATALOG[2], None).status)


# ---------------------------------------------------------------------------
# Verifier primitives.
# ---------------------------------------------------------------------------
class TestVerifierPrimitives(unittest.TestCase):
    def test_quality_scan_clean(self):
        # Unique ALPHABETIC words (the variety scan tokenizes [A-Za-z']+,
        # so digits would split words and collapse the variety).
        words = [a + b + c for a in "abcdefg" for b in "hijklmn"
                 for c in "opqrstu"]
        text = "\n".join(
            " ".join(words[i * 12:(i + 1) * 12]) for i in range(20)
        )
        qa = analyze_output_quality(text)
        self.assertFalse(qa["tool_call_cascade"])
        self.assertLess(qa["max_line_repeat"], 5)
        self.assertGreaterEqual(qa["lexical_variety"], 0.30)

    def test_quality_scan_cascade(self):
        qa = analyze_output_quality("Sure. <tool_call>{}</tool_call> done")
        self.assertTrue(qa["tool_call_cascade"])

    def test_quality_scan_repetition(self):
        text = "intro line\n" + "the same line\n" * 6
        qa = analyze_output_quality(text)
        self.assertGreaterEqual(qa["max_line_repeat"], 5)

    def test_quality_scan_low_variety(self):
        qa = analyze_output_quality("buy now " * 150)
        self.assertLess(qa["lexical_variety"], 0.30)

    def test_reassemble_stream(self):
        lines = sse_lines(
            [{"content": "Hel"}, {"content": "lo"},
             {"tool_calls": [{"index": 0, "id": "c1",
                              "function": {"name": "Read",
                                           "arguments": '{"pa'}}]},
             {"tool_calls": [{"index": 0,
                              "function": {"arguments": 'th": "x"}'}}]}],
            usage={"prompt_tokens": 10, "completion_tokens": 4},
        )
        st = reassemble_stream(lines)
        self.assertEqual(st["content"], "Hello")
        self.assertEqual(st["chunks"], 2)
        self.assertEqual(st["tool_calls"],
                         [{"id": "c1", "name": "Read",
                           "arguments": '{"path": "x"}'}])
        self.assertEqual(st["usage"]["prompt_tokens"], 10)

    def test_needle_recall(self):
        self.assertTrue(needle_recall_ok("It was Crimson Otter 42.",
                                         "crimson otter 42"))
        self.assertFalse(needle_recall_ok("crimson otter", "crimson otter 42"))
        secret = make_needle_secret(random.Random(7))
        self.assertRegex(secret, r"^\w+ \w+ \d\d$")


# ---------------------------------------------------------------------------
# Per-test verdicts through run_suite (canned responses).
# ---------------------------------------------------------------------------
class TestVerdicts(unittest.TestCase):
    def test_basic_pass_and_fail(self):
        http = FakeHttp(lambda b, s: chat_response(
            "The capital of France is Paris."))
        self.assertEqual(run_one(1, full_caps(), http)["status"], "pass")
        http = FakeHttp(lambda b, s: chat_response("I do not know."))
        self.assertEqual(run_one(1, full_caps(), http)["status"], "fail")

    def test_tool_call_pass(self):
        http = FakeHttp(lambda b, s: chat_response(None, tool_calls=[
            {"id": "c1", "type": "function",
             "function": {"name": "get_weather",
                          "arguments": '{"city": "San Francisco"}'}}]))
        r = run_one(2, full_caps(), http)
        self.assertEqual(r["status"], "pass")

    def test_tool_call_cascade_fails(self):
        http = FakeHttp(lambda b, s: chat_response(
            '<tool_call>{"name": "get_weather"}</tool_call>'))
        r = run_one(2, full_caps(), http)
        self.assertEqual(r["status"], "fail")
        self.assertIn("cascade", r["metric"]["name"])

    def test_streaming_pass_and_too_few_chunks(self):
        deltas = [{"content": f"chunk {i} words here. "} for i in range(8)]
        http = FakeHttp(lambda b, s: stream_response(deltas))
        self.assertEqual(run_one(3, full_caps(), http)["status"], "pass")
        http = FakeHttp(lambda b, s: stream_response(
            [{"content": "one long single buffered blob of text"}]))
        self.assertEqual(run_one(3, full_caps(), http)["status"], "fail")

    def test_agentic_stream_tool_call(self):
        deltas = [
            {"tool_calls": [{"index": 0, "id": "c1",
                             "function": {"name": "Bash",
                                          "arguments": '{"command": "ls"}'}}]},
        ]
        http = FakeHttp(lambda b, s: stream_response(
            deltas, usage={"prompt_tokens": 900, "completion_tokens": 30}))
        r = run_one(4, full_caps(), http)
        self.assertEqual(r["status"], "pass")
        self.assertIn("decode_tps", r["detail"])
        # no tool call despite required -> fail
        http = FakeHttp(lambda b, s: stream_response(
            [{"content": "I will not call a tool."}] * 6))
        self.assertEqual(run_one(4, full_caps(), http)["status"], "fail")

    def test_thinking_pass_fail_and_expected_fail_warn(self):
        good = chat_response("4", reasoning="r" * 120)
        http = FakeHttp(lambda b, s: good)
        self.assertEqual(run_one(5, full_caps(), http)["status"], "pass")
        empty = FakeHttp(lambda b, s: chat_response("4", reasoning=""))
        self.assertEqual(run_one(5, full_caps(), empty)["status"], "fail")
        # no reasoning parser: same failure is pre-flagged -> warn
        empty = FakeHttp(lambda b, s: chat_response("4", reasoning=""))
        r = run_one(5, full_caps(reasoning_parser=None), empty)
        self.assertEqual(r["status"], "warn")
        self.assertIn("reasoning-parser", r["reason"])

    def test_quality_verdicts(self):
        wordlist = [a + b + c for a in "abcdefg" for b in "hijklmn"
                    for c in "opqrstu"]
        clean = "\n".join(
            " ".join(wordlist[i * 13:(i + 1) * 13]) for i in range(25)
        )
        http = FakeHttp(lambda b, s: chat_response(clean))
        self.assertEqual(run_one(6, full_caps(), http)["status"], "pass")
        # repetition cascade
        rep = "start\n" + "same line again\n" * 7
        http = FakeHttp(lambda b, s: chat_response(rep))
        r = run_one(6, full_caps(), http)
        self.assertEqual(r["status"], "fail")
        self.assertEqual(r["metric"]["name"], "max_line_repeat")
        # tool-call cascade
        http = FakeHttp(lambda b, s: chat_response("x <tool_call> y"))
        self.assertEqual(run_one(6, full_caps(), http)["metric"]["name"],
                         "cascade")
        # truncated / empty
        http = FakeHttp(lambda b, s: chat_response("", finish="length"))
        self.assertEqual(run_one(6, full_caps(), http)["status"], "fail")

    def test_mtp_acceptance_from_pinned_gauge(self):
        metrics = (f"{SPEC_EMA_ACCEPT_LEN_METRIC} 2.85\n"
                   "sglang:spec_accept_rate 0.71\n")
        http = FakeHttp(lambda b, s: chat_response("1\n2\n3"),
                        metrics_text=metrics)
        r = run_one(7, full_caps(), http)
        self.assertEqual(r["status"], "pass")
        self.assertAlmostEqual(r["metric"]["numeric"], 2.85)
        self.assertEqual(r["detail"]["spec_mode"], "mtp-adaptive-k")
        # collapsed AL -> fail
        http = FakeHttp(lambda b, s: chat_response("1"),
                        metrics_text=f"{SPEC_EMA_ACCEPT_LEN_METRIC} 1.2\n")
        self.assertEqual(run_one(7, full_caps(), http)["status"], "fail")
        # gauge absent -> skip, not fail
        http = FakeHttp(lambda b, s: chat_response("1"),
                        metrics_text="sglang:gen_throughput 12\n")
        self.assertEqual(run_one(7, full_caps(), http)["status"], "skip")

    def test_reasoning_heavy_token_gate(self):
        http = FakeHttp(lambda b, s: chat_response(
            "proof...", completion_tokens=1800))
        self.assertEqual(run_one(13, full_caps(), http)["status"], "pass")
        http = FakeHttp(lambda b, s: chat_response(
            "4", completion_tokens=120))
        r = run_one(13, full_caps(), http)
        self.assertEqual(r["status"], "fail")

    def test_tool_prefill_verdicts(self):
        http = FakeHttp(lambda b, s: chat_response("t" * 200))
        self.assertEqual(run_one(9, full_caps(), http)["status"], "pass")
        http = FakeHttp(lambda b, s: chat_response(""))
        self.assertEqual(run_one(9, full_caps(), http)["status"], "fail")
        http = FakeHttp(lambda b, s: HttpResult(status=500, body="OOM"))
        self.assertEqual(run_one(9, full_caps(), http)["status"], "fail")

    def test_throughput_informational(self):
        deltas = [{"content": "words "} for _ in range(10)]
        http = FakeHttp(lambda b, s: stream_response(
            deltas, usage={"prompt_tokens": 20, "completion_tokens": 500},
            ttft_ms=200.0, wall_ms=10200.0))
        r = run_one(16, full_caps(), http)
        self.assertEqual(r["status"], "info")
        self.assertEqual(r["metric"]["name"], "decode_tps_mean")
        self.assertAlmostEqual(r["metric"]["numeric"], 50.0)
        self.assertEqual(set(r["detail"]["runs"]), {"narrative", "code"})


# ---------------------------------------------------------------------------
# Needle tri-state + long-ctx graceful skip.
# ---------------------------------------------------------------------------
def echo_needle_handler(recall=True):
    """Chat handler that extracts the secret from the request and answers
    with it (or with a wrong phrase for the recall-miss case)."""
    def handler(body, stream):
        content = body["messages"][0]["content"]
        m = re.search(r"The hidden phrase is '([^']+)'", content)
        answer = m.group(1) if (m and recall) else "plaid walrus 00"
        return stream_response(
            [{"content": answer}],
            usage={"prompt_tokens": len(content) // 4, "completion_tokens": 8},
        )
    return handler


class TestNeedleTriState(unittest.TestCase):
    def test_recall_pass(self):
        http = FakeHttp(echo_needle_handler(recall=True))
        r = run_one(8, full_caps(), http, rng=random.Random(1))
        self.assertEqual(r["status"], "pass")
        self.assertEqual(len(r["detail"]["rungs"]), 2)

    def test_recall_miss_is_info_never_fail(self):
        http = FakeHttp(echo_needle_handler(recall=False))
        r = run_one(8, full_caps(), http, rng=random.Random(1))
        self.assertEqual(r["status"], "info")
        self.assertEqual(r["detail"]["rungs"][0]["outcome"], "info")

    def test_http_400_is_graceful_skip(self):
        http = FakeHttp(lambda b, s: HttpResult(status=400,
                                                body="context too long"))
        r = run_one(8, full_caps(), http, rng=random.Random(1))
        self.assertEqual(r["status"], "skip")

    def test_http_500_is_system_fail(self):
        http = FakeHttp(lambda b, s: HttpResult(status=500, body="boom"))
        r = run_one(14, full_caps(), http, rng=random.Random(1))
        self.assertEqual(r["status"], "fail")

    def test_precheck_skip_result_through_run_suite(self):
        # max_model_len below every rung of test 14 -> gated skip, engine
        # never called with a needle request.
        http = FakeHttp(lambda b, s: self.fail("must not reach the engine"))
        r = run_one(14, full_caps(max_model_len=32768), http)
        self.assertEqual(r["status"], "skip")
        chat_calls = [c for c in http.calls
                      if c["url"].endswith("chat/completions")]
        self.assertEqual(chat_calls, [])


# ---------------------------------------------------------------------------
# Suite driver: gating in-run, ordering, engine-health, progress_cb, schema.
# ---------------------------------------------------------------------------
class TestRunSuite(unittest.TestCase):
    def test_result_schema(self):
        http = FakeHttp(lambda b, s: chat_response("Paris."))
        r = run_one(1, full_caps(), http)
        self.assertEqual(
            set(r),
            {"test_id", "label", "status", "metric", "detail", "deps"},
        )
        self.assertEqual(set(r["metric"]),
                         {"name", "value", "numeric", "unit"})
        for key in ("prompt_tokens", "prefill_tps", "ttft_ms", "http_code",
                    "finish"):
            self.assertIn(key, r["detail"])
        self.assertEqual(
            set(r["deps"]),
            {"chat_template", "tools", "streaming", "thinking",
             "longctx_rung"},
        )

    def test_blocked_and_skip_flow_through(self):
        http = FakeHttp(lambda b, s: chat_response("Paris."))
        caps = full_caps(tool_parser=None, spec_decode=False)
        results = {r["test_id"]: r for r in run_suite(
            "http://x:1", "test-model", [1, 2, 7], capabilities=caps,
            http=http)}
        self.assertEqual(results[1]["status"], "pass")
        self.assertEqual(results[2]["status"], "blocked")
        self.assertEqual(results[7]["status"], "skip")

    def test_engine_health_recheck_skips_rest(self):
        # Test 9 (crash-prone) kills the engine; 12 and 13 must be skipped,
        # not run against a dead server.
        state = {"dead": False}

        def chat(body, stream):
            state["dead"] = True
            return HttpResult(status=0, error="died")

        http = FakeHttp(chat)
        orig = http.__call__

        def call(method, url, body=None, stream=False, timeout=None):
            if url.endswith("/v1/models") and state["dead"]:
                http.calls.append({"method": method, "url": url,
                                   "body": body, "stream": stream})
                return HttpResult(status=0, error="down")
            return orig(method, url, body=body, stream=stream,
                        timeout=timeout)

        results = list(run_suite("http://x:1", "test-model", [9, 12, 13],
                                 capabilities=full_caps(), http=call))
        by_id = {r["test_id"]: r for r in results}
        self.assertEqual(by_id[9]["status"], "fail")
        self.assertEqual(by_id[12]["status"], "skip")
        self.assertIn("engine unhealthy after test 9", by_id[12]["reason"])
        self.assertEqual(by_id[13]["status"], "skip")

    def test_progress_cb_matches_yield(self):
        http = FakeHttp(lambda b, s: chat_response("Paris."))
        seen = []
        results = list(run_suite("http://x:1", "test-model", [1],
                                 capabilities=full_caps(), http=http,
                                 progress_cb=seen.append))
        self.assertEqual(seen, results)

    def test_runner_exception_becomes_fail_not_crash(self):
        def boom(body, stream):
            raise RuntimeError("handler exploded")
        http = FakeHttp(boom)
        r = run_one(1, full_caps(), http)
        self.assertEqual(r["status"], "fail")


# ---------------------------------------------------------------------------
# Capability probe.
# ---------------------------------------------------------------------------
class TestProbeCapabilities(unittest.TestCase):
    def _http(self, server_info, ping_status=200):
        def call(method, url, body=None, stream=False, timeout=None):
            if url.endswith("/v1/models"):
                return HttpResult(status=200, body=json.dumps({
                    "data": [{"id": "qwen", "max_model_len": 262144}]}))
            if url.endswith("/get_server_info"):
                return HttpResult(status=200, body=json.dumps(server_info))
            if url.endswith("/v1/chat/completions"):
                return HttpResult(status=ping_status, body=json.dumps(
                    {"choices": [{"message": {"content": "x"}}]}))
            return HttpResult(status=404)
        return call

    def test_full_probe(self):
        caps = probe_capabilities("http://x:1", http=self._http({
            "tool_call_parser": "qwen3_coder",
            "reasoning_parser": "qwen3",
            "speculative_algorithm": "NEXTN",
            "speculative_adaptive": True,
        }))
        self.assertEqual(caps.model, "qwen")
        self.assertEqual(caps.max_model_len, 262144)
        self.assertEqual(caps.tool_parser, "qwen3_coder")
        self.assertEqual(caps.reasoning_parser, "qwen3")
        self.assertTrue(caps.spec_decode)
        self.assertEqual(caps.spec_mode, "mtp-adaptive-k")
        self.assertTrue(caps.chat_template_basic)

    def test_bare_server(self):
        caps = probe_capabilities("http://x:1",
                                  http=self._http({}, ping_status=400))
        self.assertIsNone(caps.tool_parser)
        self.assertIsNone(caps.reasoning_parser)
        self.assertFalse(caps.spec_decode)
        self.assertEqual(caps.spec_mode, "off")
        self.assertFalse(caps.chat_template_basic)

    def test_fixed_k_mode(self):
        caps = probe_capabilities("http://x:1", http=self._http({
            "speculative_algorithm": "NEXTN",
            "speculative_adaptive": False,
        }))
        self.assertEqual(caps.spec_mode, "mtp-fixed-k")


if __name__ == "__main__":
    unittest.main()
