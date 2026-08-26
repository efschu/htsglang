# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Benchmark / quality suite core for the dashboard "Benchmark Run" panel
(design #151).

Backend-driven port of the club-3090 verification suite:

  * Tests 1-7  = verify-full.sh   checks 3-8   (functional smoke)
  * Tests 8-15 = verify-stress.sh checks 1-8   (stress / boundary)
  * Test  4    = bench-agentic.sh              (streaming tool-calls, 1 turn)
  * Test 16    = bench.sh                      (throughput, narrative + code)

All request bodies are ported VERBATIM from the scripts (same prompts, same
max_tokens / temperature / tool schemas), and the verifiers are ports of the
script heredocs (Paris-grep, tool_calls-parse, cascade / lexical-variety scan,
needle-recall match, completion-token gates). The BROWSER NEVER CALLS THE
MODEL: the webui layer calls :func:`run_suite` server-side and streams the
per-test results back (CORS-safe, design section "BACKEND-DRIVEN").

Chat-template capability GATING (the design's core point):

  * no basic chat template        -> ALL chat tests blocked
  * no --tool-call-parser         -> tool tests (2,4,9,10,11) warn + BLOCKED
                                     by default; ``force=True`` runs them
                                     anyway (surfaces the cascade fail)
  * no --reasoning-parser         -> test 5 runs but a failure is reported as
                                     ``warn`` (pre-flagged expected-fail)
  * spec-decode off               -> test 7 skipped
  * long-ctx rung > max_model_len -> graceful skip (pre-check AND the
                                     engine's HTTP 400 rejection), NOT a fail

TRI-STATE verdicts: ``pass`` / ``info`` / ``fail``. A needle recall MISS on a
healthy HTTP 200 is attention-quality information (``info``), never a hard
failure -- only system-level failures (HTTP 5xx / timeout / crash) fail.
Gating adds the ``skip`` / ``warn`` / ``blocked`` statuses from the design's
result schema.

Result schema per test (design "Result-Schema pro Test"):

    {test_id, label, status: pass|info|fail|skip|warn|blocked,
     metric: {name, value, numeric, unit},
     detail: {prompt_tokens, prefill_tps, ttft_ms, http_code, finish, ...},
     deps:   {chat_template, tools, streaming, thinking, longctx_rung}}

MTP acceptance length (test 7) is read from Prometheus ``/metrics`` using the
metric names PINNED in ``live_metrics.py`` (imported, not re-guessed):
``sglang:spec_accept_rate`` / ``sglang:spec_num_steps`` /
``sglang:spec_ema_accept_len``.

HONEST NOTES (need a live server to close):
  * The AL gauge names are pinned against this fork's metrics_collector.py via
    live_metrics.py, but the end-to-end AL flow (drive decode -> scrape ->
    EMA moved) has not been exercised against a live htsglang build from THIS
    module yet (the running server may not be disturbed while this was built).
  * Long-ctx filler sizing uses the verify-stress.sh FALLBACK ratio of ~65
    tokens per filler-scale unit (Qwen tokenizers); the script's live
    calibration probe is not replicated. Oversized rungs degrade gracefully
    via the max_model_len pre-check + the HTTP 400 path.
  * TTFT is measured at the first SSE data line (transport level), a close
    upper bound of the scripts' first-content-delta timing.
  * Test 16 does ONE measured run per prompt (bench.sh does 3 warmup + 5
    measured); numbers are labelled informational accordingly.
"""

from __future__ import annotations

import dataclasses
import json
import random
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from sglang.srt.planner.energy import parse_prometheus_metrics
from sglang.srt.planner.live_metrics import (
    SPEC_ACCEPT_RATE_METRIC,
    SPEC_EMA_ACCEPT_LEN_METRIC,
    SPEC_NUM_STEPS_METRIC,
)

__all__ = [
    "Capabilities",
    "GateDecision",
    "HttpResult",
    "TEST_CATALOG",
    "TestSpec",
    "PRESETS",
    "STATUSES",
    "gate_test",
    "order_selected",
    "probe_capabilities",
    "run_suite",
    # request builders (exact bodies, reused by the UI layer / tests)
    "request_basic",
    "request_tool_call",
    "request_streaming",
    "request_agentic_turn",
    "request_thinking",
    "request_quality",
    "request_mtp_trigger",
    "request_needle",
    "request_tool_prefill",
    "request_ide_agent",
    "request_multiturn_agent",
    "request_lcb_coding",
    "request_reasoning_heavy",
    "request_throughput",
    # verifier primitives (pure, unit-tested)
    "analyze_output_quality",
    "reassemble_stream",
    "make_needle_secret",
    "needle_recall_ok",
]

#: Legal ``status`` values. ``pass``/``info``/``fail`` are the tri-state
#: verdicts of an EXECUTED test; ``skip``/``warn``/``blocked`` come from
#: gating (or, for ``warn``, a pre-flagged expected-fail).
STATUSES = ("pass", "info", "fail", "skip", "warn", "blocked")

#: Tokens per needle filler-scale unit (verify-stress.sh fallback for Qwen
#: tokenizers; the script's live calibration probe is not replicated here).
TOKENS_PER_FILLER_SCALE = 65

#: Cliff-2 tests (can crash the engine) -- always ordered LAST in a run.
CLIFF2_TESTS = frozenset({14, 15})

#: Tests that can plausibly kill the engine; an engine-health recheck runs
#: after each of these before the next test starts.
CRASH_PRONE_TESTS = frozenset({9, 10, 11, 14, 15})


# ===========================================================================
# HTTP transport (injectable; tests pass canned HttpResults, no network).
# ===========================================================================
@dataclasses.dataclass
class HttpResult:
    """One HTTP exchange. ``status == 0`` means no HTTP response at all
    (timeout / connection refused / engine died) -- the scripts' "000"."""

    status: int
    body: str = ""
    sse_lines: Optional[List[str]] = None
    ttft_ms: Optional[float] = None
    wall_ms: Optional[float] = None
    error: Optional[str] = None

    def json(self) -> Optional[dict]:
        try:
            d = json.loads(self.body)
            return d if isinstance(d, dict) else None
        except Exception:
            return None


def _default_http(
    method: str,
    url: str,
    body: Optional[dict] = None,
    stream: bool = False,
    timeout: float = 120.0,
) -> HttpResult:
    """urllib transport. For ``stream=True`` the SSE lines are collected and
    ``ttft_ms`` is stamped at the FIRST ``data: `` line received."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status = r.getcode() or 200
            if stream:
                lines: List[str] = []
                ttft = None
                for raw in r:
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    if ttft is None and line.startswith("data: "):
                        ttft = (time.time() - t0) * 1000.0
                    lines.append(line)
                return HttpResult(
                    status=status,
                    sse_lines=lines,
                    ttft_ms=ttft,
                    wall_ms=(time.time() - t0) * 1000.0,
                )
            text = r.read().decode("utf-8", errors="replace")
            return HttpResult(
                status=status, body=text, wall_ms=(time.time() - t0) * 1000.0
            )
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")[:1000]
        except Exception:
            err_body = ""
        return HttpResult(
            status=e.code,
            body=err_body,
            error=str(e),
            wall_ms=(time.time() - t0) * 1000.0,
        )
    except Exception as e:
        return HttpResult(status=0, error=str(e), wall_ms=(time.time() - t0) * 1000.0)


# ===========================================================================
# Capabilities: what the deployed server can do (gating input).
# ===========================================================================
@dataclasses.dataclass
class Capabilities:
    """Probed / user-declared server capabilities that gate the catalog.

    ``spec_mode`` records WHICH spec mode the server runs ("off" /
    "mtp-fixed-k" / "mtp-adaptive-k") -- a server-launch property, not a
    request parameter; it is copied into test 7's result for provenance.
    """

    chat_template_basic: bool = True
    tool_parser: Optional[str] = None  # None -> not tool-aware
    reasoning_parser: Optional[str] = None  # None -> not thinking-aware
    streaming: bool = True
    spec_decode: bool = False
    spec_mode: str = "off"
    max_model_len: Optional[int] = None
    model: Optional[str] = None

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def probe_capabilities(
    endpoint: str,
    *,
    http: Optional[Callable[..., HttpResult]] = None,
    timeout: float = 10.0,
    chat_ping: bool = True,
) -> Capabilities:
    """Probe a running OpenAI-compatible sglang server:

      * ``/v1/models``       -> model id + max_model_len
      * ``/get_server_info`` -> launch flags (--tool-call-parser /
                                --reasoning-parser / speculative_algorithm /
                                speculative_adaptive)
      * a 1-token chat ping  -> "basic chat template works" signal

    Launch flags are the RELIABLE template-capability source (design honest
    note: probing responses confounds "template missing" with "model bad at
    tools"); the chat ping only covers the basic-template axis.
    """
    http = http or _default_http
    base = endpoint.rstrip("/")
    caps = Capabilities(chat_template_basic=False)

    models = http("GET", base + "/v1/models", timeout=timeout)
    md = models.json() or {}
    data = md.get("data") or []
    if data and isinstance(data[0], dict):
        caps.model = data[0].get("id")
        mml = data[0].get("max_model_len")
        if isinstance(mml, (int, float)) and mml > 0:
            caps.max_model_len = int(mml)

    si_res = http("GET", base + "/get_server_info", timeout=timeout)
    si = si_res.json() or {}
    tool_parser = si.get("tool_call_parser")
    reasoning_parser = si.get("reasoning_parser")
    caps.tool_parser = tool_parser or None
    caps.reasoning_parser = reasoning_parser or None
    if caps.max_model_len is None:
        for key in ("max_model_len", "context_length", "max_total_num_tokens"):
            v = si.get(key)
            if isinstance(v, (int, float)) and v > 0:
                caps.max_model_len = int(v)
                break
    spec_algo = si.get("speculative_algorithm")
    caps.spec_decode = bool(spec_algo)
    if spec_algo:
        caps.spec_mode = (
            "mtp-adaptive-k" if si.get("speculative_adaptive") else "mtp-fixed-k"
        )

    if chat_ping and caps.model:
        ping = http(
            "POST",
            base + "/v1/chat/completions",
            body={
                "model": caps.model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=max(timeout, 180.0),  # doubles as cold-start warmup
        )
        caps.chat_template_basic = ping.status == 200
    return caps


# ===========================================================================
# Request builders -- exact bodies, verbatim from the club-3090 scripts.
# ===========================================================================
def _no_think(body: dict) -> dict:
    body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def request_basic(model: str) -> dict:
    """Test 1 -- verify-full.sh [3/8]."""
    return _no_think(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France? One short sentence.",
                }
            ],
            "max_tokens": 30,
            "temperature": 0.6,
        }
    )


def request_tool_call(model: str) -> dict:
    """Test 2 -- verify-full.sh [4/8]."""
    return _no_think(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in San Francisco? "
                    "Use the get_weather tool.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "max_tokens": 200,
            "temperature": 0.3,
        }
    )


def request_streaming(model: str) -> dict:
    """Test 3 -- verify-full.sh [5/8]."""
    return _no_think(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Write a three-sentence haiku about debugging.",
                }
            ],
            "max_tokens": 120,
            "temperature": 0.6,
            "stream": True,
        }
    )


#: bench-agentic.sh system prompt (verbatim; fixed so prefix caching warms).
AGENTIC_SYSTEM = (
    "You are an autonomous coding assistant working inside a Python repository. "
    "The user is investigating a performance regression. When file contents, "
    "search results, or command output would materially change your answer, "
    "call the appropriate tool — don't speculate. After each tool call, "
    "briefly state what you learned and what your next planned step is. "
    "Keep responses concise (under 100 words); defer to tools for raw data.\n\n"
    "Repository layout:\n"
    "  scripts/         — bench, verify, soak, launch helper scripts\n"
    "  models/          — per-model compose configs + patches\n"
    "  docs/            — architecture and cliff notes\n"
    "  BENCHMARKS.md    — measured performance numbers\n"
    "  CHANGELOG.md     — version history\n"
)

#: bench-agentic.sh 10-tool schema set (verbatim).
AGENTIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": n,
            "description": d,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "command": {"type": "string"},
                    "pattern": {"type": "string"},
                    "recursive": {"type": "boolean"},
                },
                "required": [],
            },
        },
    }
    for n, d in [
        ("Read", "Read a UTF-8 file from the repository."),
        ("Bash", "Execute a shell command and return stdout+stderr."),
        ("Edit", "Apply a string replacement edit to a file."),
        ("Write", "Write or overwrite a file."),
        ("Grep", "Search for a regex pattern across the codebase."),
        ("LS", "List files in a directory."),
        ("TodoRead", "Read the current task/todo list."),
        ("TodoWrite", "Create or update a task/todo list."),
        ("WebSearch", "Search the web for information."),
        ("WebFetch", "Fetch a URL and return the HTML/text."),
    ]
]

#: First fixture turn of scripts/fixtures/agentic-bench-fixture.json (the full
#: 15-turn fixture is ~100K tokens and stays in the club-3090 repo; the panel
#: runs ONE streamed tool-choice=required turn, which is the per-turn
#: measurement bench-agentic.sh reports).
AGENTIC_FIXTURE_USER = (
    'Run: ls <repo>/ && echo "---" && ls <repo>/scripts/ 2>/dev/null '
    '&& echo "---" && ls <'
)


def request_agentic_turn(model: str) -> dict:
    """Test 4 -- bench-agentic.sh single streamed turn."""
    return _no_think(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": AGENTIC_SYSTEM},
                {"role": "user", "content": AGENTIC_FIXTURE_USER},
            ],
            "tools": AGENTIC_TOOLS,
            "tool_choice": "required",
            "max_tokens": 150,
            "temperature": 0.3,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )


def request_thinking(model: str) -> dict:
    """Test 5 -- verify-full.sh [6/8]."""
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "What is 2+2? One-line answer.",
            }
        ],
        "max_tokens": 4000,
        "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def request_quality(model: str) -> dict:
    """Test 6 -- verify-full.sh [7/8] (cascade / degeneracy scan)."""
    return _no_think(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Write a detailed 1500-word essay explaining how "
                    "transformer attention works. Cover: query/key/value "
                    "projections, scaled dot-product attention, softmax, "
                    "multi-head attention, positional encodings, and a "
                    "brief comparison with RNN-based attention.",
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.6,
        }
    )


def request_mtp_trigger(model: str) -> dict:
    """Test 7 -- verify-full.sh [8/8] decode driver (AL then read from
    /metrics, see the module docstring)."""
    return _no_think(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Count from 1 to 80, one number per line.",
                }
            ],
            "max_tokens": 500,
            "temperature": 0.0,
        }
    )


#: verify-stress.sh needle filler block (verbatim).
NEEDLE_BLOCK = (
    "This section describes the history of computing in detail. "
    "Transistors were invented in 1947 at Bell Labs. The integrated circuit "
    "came a decade later. "
    "Microprocessors emerged in the 1970s and changed the world. "
    "Personal computing followed, then networking, then the web, then cloud "
    "and AI. "
)

_NEEDLE_ANIMALS = [
    "otter",
    "falcon",
    "platypus",
    "iguana",
    "narwhal",
    "chinchilla",
    "capybara",
    "axolotl",
]
_NEEDLE_COLORS = [
    "crimson",
    "turquoise",
    "amber",
    "violet",
    "emerald",
    "sapphire",
    "silver",
    "golden",
]


def make_needle_secret(rng: Optional[random.Random] = None) -> str:
    """Fresh '{color} {animal} {NN}' secret per rung (defeats prefix caching)."""
    rng = rng or random.Random()
    return (
        f"{rng.choice(_NEEDLE_COLORS)} {rng.choice(_NEEDLE_ANIMALS)} "
        f"{rng.randint(10, 99)}"
    )


def request_needle(model: str, filler_scale: int, secret: str) -> dict:
    """Tests 8 / 14 / 15 -- verify-stress.sh NIAH request (verbatim shape).
    Secret at ~50% depth, retrieval question at the end, max 30 tokens t0."""
    half = filler_scale // 2
    content = (
        NEEDLE_BLOCK * half + f"\n\nIMPORTANT MEMORY: The hidden phrase is '{secret}'. "
        "Remember this exactly.\n\n"
        + NEEDLE_BLOCK
        * (filler_scale - half)
        + "\n\nQuestion: In the middle of the document above I wrote "
        "'The hidden phrase is ___'. What was the hidden phrase? "
        "Reply with only the phrase, no other text."
    )
    return _no_think(
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 30,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )


#: verify-stress.sh check 2 mock tool-response news blocks (verbatim subset --
#: the payload is opaque filler repeated to PREFILL_TARGET_CHARS).
_PREFILL_BLOCKS = [
    "Federal Reserve Chair Jerome Powell stated today that interest rates "
    "would remain steady amid mixed economic signals. The central bank's "
    "decision came after months of debate about inflation trajectories and "
    "labor market resilience. Treasury yields responded modestly, with the "
    "10-year note ticking down two basis points by late trading.",
    "European markets opened higher on news that German industrial output "
    "rebounded sharply in March. The DAX gained 0.8% in morning trading "
    "while the Stoxx 600 added 0.5%. Analysts cited improved manufacturing "
    "PMI readings and stabilizing energy prices as primary drivers behind "
    "the optimistic open.",
    "Tech sector earnings season kicked into high gear this week with "
    "several major firms reporting better-than-expected quarterly results. "
    "Cloud computing revenues grew across the board, with AI infrastructure "
    "demand cited as a key catalyst. Margin pressure remained a concern in "
    "semiconductor names due to inventory adjustments.",
    "Crude oil prices edged higher after OPEC announced extended production "
    "cuts through the third quarter. Brent crude rose 1.2% to settle near "
    "$84 per barrel, while WTI gained similarly to $79. Geopolitical "
    "tensions in the Middle East continued to lend support to prices "
    "despite weakening demand signals from China.",
]


def request_tool_prefill(model: str, target_chars: int = 100000) -> dict:
    """Test 9 -- verify-stress.sh [2/8]: ~25K-token mock tool response
    (100000 chars default) in a 4-message tool conversation."""
    content = ""
    i = 0
    while len(content) < target_chars:
        content += _PREFILL_BLOCKS[i % len(_PREFILL_BLOCKS)] + "\n\n"
        i += 1
    tool_def = {
        "type": "function",
        "function": {
            "name": "fetch_news",
            "description": "Fetch latest news on a topic.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
    }
    return _no_think(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "What's happening in financial markets today?",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_news_1",
                            "type": "function",
                            "function": {
                                "name": "fetch_news",
                                "arguments": json.dumps({"topic": "markets"}),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_news_1", "content": content},
                {
                    "role": "user",
                    "content": "Summarize the top 3 themes from this news data in "
                    "about 100 words.",
                },
            ],
            "tools": [tool_def],
            "tool_choice": "auto",
            "max_tokens": 500,
            "temperature": 0.6,
        }
    )


def request_ide_agent(model: str) -> dict:
    """Test 10 -- verify-stress.sh [3/8]: IDE-agent one-shot (Cliff-1 probe).
    tool_choice='none' deliberately forces the content-only reasoning path."""
    sys_text = (
        "You are a helpful AI coding assistant operating inside an IDE. You "
        "have access to a set of tools to read, write, search, and execute "
        "commands in the user's project. Always use the appropriate tool "
        "when the user requests file operations or code execution. Be "
        "concise in your reasoning, prefer minimal edits, and verify your "
        "changes by reading the file back after writing. When refactoring, "
        "preserve existing behavior unless explicitly asked to change it. "
        "When reasoning through complex changes, think step by step but "
        "keep the explanation focused on the specific change being made. "
        "Avoid restating the user's request. If a request is ambiguous, ask "
        "one focused clarifying question rather than guessing. When a task "
        "requires multiple file edits, plan the edits first, then execute "
        "them in order, verifying each before moving to the next. Never "
        "modify files outside the user's project root. Never run "
        "destructive commands without explicit confirmation. "
    ) * 5
    tools = [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": d,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "pattern": {"type": "string"},
                        "command": {"type": "string"},
                        "content": {"type": "string"},
                        "recursive": {"type": "boolean"},
                        "encoding": {"type": "string", "default": "utf-8"},
                    },
                    "required": ["path"],
                },
            },
        }
        for n, d in [
            ("read_file", "Read the contents of a file at the given path."),
            ("write_file", "Write content to a file at the given path."),
            ("list_directory", "List files at the given path, optionally recursive."),
            ("search_code", "Search for a regex pattern across the codebase."),
            ("run_command", "Execute a shell command in the project directory."),
            ("get_file_metadata", "Get metadata for a file."),
            ("create_directory", "Create a directory."),
            ("delete_file", "Delete a file."),
            ("git_status", "Get the current git status."),
            ("git_diff", "Get the diff for current changes."),
        ]
    ]
    user_text = (
        "I have a Python function `compute_metrics` in "
        "`src/analytics/metrics.py` that currently calculates running "
        "statistics by re-iterating the entire data list every call. "
        "Refactor it to maintain a streaming aggregation state that updates "
        "incrementally. Preserve the public API. Show me the diff before "
        "applying it."
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_text},
        ],
        "tools": tools,
        "tool_choice": "none",
        "max_tokens": 2000,
        "temperature": 0.0,
        "stream": False,
    }


def request_multiturn_agent(model: str) -> dict:
    """Test 11 -- verify-stress.sh [4/8]: 4-turn agent history."""
    sys_text = (
        "You are a coding assistant inside an IDE. Use the provided tools to "
        "read and edit files. Be concise. After each tool call, verify the "
        "result before proceeding to the next step. "
    ) * 8
    tools = [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": d,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "pattern": {"type": "string"},
                    },
                    "required": ["path"],
                },
            },
        }
        for n, d in [
            ("read_file", "Read a file."),
            ("write_file", "Write a file."),
            ("search_code", "Search for a regex pattern."),
            ("list_directory", "List a directory."),
        ]
    ]
    mock_file = "\n".join(
        f"def function_{i}(arg{i}): return arg{i} * {i + 1}  # line {i}"
        for i in range(80)
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_text},
            {
                "role": "user",
                "content": "Read src/utils.py and tell me what functions are defined.",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_read_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "src/utils.py"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_read_1", "content": mock_file},
            {
                "role": "user",
                "content": "Now refactor function_5 to use a different multiplier.",
            },
        ],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 1500,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": False,
    }


def request_lcb_coding(model: str) -> dict:
    """Test 12 -- verify-stress.sh [5/8]: LCB-coding shape."""
    problem = (
        "You are given an integer array nums. Return the length of the "
        "longest subarray with a sum equal to a target value k. If no such "
        "subarray exists, return 0.\n\n"
        "Example 1:\n"
        "Input: nums = [1, -1, 5, -2, 3], k = 3\n"
        "Output: 4\n"
        "Explanation: The subarray [1, -1, 5, -2] sums to 3 and has "
        "length 4.\n\n"
        "Example 2:\n"
        "Input: nums = [-2, -1, 2, 1], k = 1\n"
        "Output: 2\n\n"
        "Constraints:\n"
        "- 1 <= nums.length <= 2 * 10^5\n"
        "- -10^4 <= nums[i] <= 10^4\n"
        "- -10^9 <= k <= 10^9\n\n"
        "Plan your approach in the format:\n"
        "GOAL: <one-line restatement>\n"
        "STATE: <data structures>\n"
        "ALGO: <key steps>\n"
        "EDGE: <edge cases>\n"
        "VERIFY: <how to test>\n\n"
        "Then implement the solution as "
        "`class Solution: def maxSubArrayLen(...)`."
    )
    return {
        "model": model,
        "messages": [{"role": "user", "content": problem}],
        "max_tokens": 4096,
        "temperature": 0.0,
        "stream": False,
    }


def request_reasoning_heavy(model: str) -> dict:
    """Test 13 -- verify-stress.sh [6/8]: sum-of-cubes induction + sigma n^4."""
    problem = (
        "Prove that for any positive integer n, the sum 1^3 + 2^3 + 3^3 + "
        "... + n^3 equals (n(n+1)/2)^2. Show every step of your reasoning, "
        "including:\n"
        "1. The base case verification.\n"
        "2. The inductive hypothesis.\n"
        "3. The full algebraic manipulation in the inductive step.\n"
        "4. A geometric or visual interpretation if you can think of one.\n"
        "5. A verification by computing both sides for n=1, 2, 3, 4, 5.\n\n"
        "Be thorough; show every algebraic step rather than skipping any. "
        "After the proof, also derive a closed-form expression for the sum "
        "1^4 + 2^4 + ... + n^4 using the same induction technique, and "
        "verify it for n=1, 2, 3."
    )
    return {
        "model": model,
        "messages": [{"role": "user", "content": problem}],
        "max_tokens": 8192,
        "temperature": 0.0,
        "stream": False,
    }


#: bench.sh canonical prompts (verbatim).
THROUGHPUT_PROMPTS = (
    (
        "narrative",
        "Write a detailed 800-word essay explaining transformer attention.",
        1000,
    ),
    (
        "code",
        "Write a Python implementation of quicksort with comments explaining "
        "each step.",
        800,
    ),
)


def request_throughput(model: str, kind: str) -> dict:
    """Test 16 -- bench.sh measured run (kind in {'narrative', 'code'})."""
    for name, prompt, max_tokens in THROUGHPUT_PROMPTS:
        if name == kind:
            return {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.6,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
    raise ValueError(f"unknown throughput prompt kind: {kind!r}")


# ===========================================================================
# Verifier primitives (pure -- ports of the script heredocs).
# ===========================================================================
def reassemble_stream(sse_lines: Iterable[str]) -> Dict[str, Any]:
    """Reassemble an SSE chat stream: content, reasoning, per-index tool
    calls, usage from the final chunk, chunk count, finish_reason.
    Port of the bench-agentic.sh / verify-full.sh stream readers."""
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls_acc: Dict[int, Dict[str, str]] = {}
    usage: Optional[dict] = None
    finish: Optional[str] = None
    content_chunks = 0
    for line in sse_lines or []:
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except Exception:
            continue
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                content_parts.append(delta["content"])
                content_chunks += 1
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = tool_calls_acc.setdefault(
                    idx, {"id": "", "name": "", "args": ""}
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
            if choices[0].get("finish_reason"):
                finish = choices[0]["finish_reason"]
        if chunk.get("usage"):
            usage = chunk["usage"]
    tool_calls = [
        {"id": s["id"], "name": s["name"], "arguments": s["args"] or "{}"}
        for _, s in sorted(tool_calls_acc.items())
        if s["name"]
    ]
    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "tool_calls": tool_calls,
        "usage": usage or {},
        "chunks": content_chunks,
        "finish": finish,
    }


def analyze_output_quality(content: str) -> Dict[str, Any]:
    """The verify-full.sh [7/8] cascade / degeneracy scan:

    * tool-call cascade: literal ``<tool_call>`` in normal content
    * repetition cascade: same non-empty line >= 5x consecutively
    * lexical variety over the first 200 words (>= 0.30 is healthy)
    """
    cascade = "<tool_call>" in content
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    max_repeat, cur_line, cur_count = 0, "", 0
    for l in lines:
        if l == cur_line:
            cur_count += 1
            max_repeat = max(max_repeat, cur_count)
        else:
            cur_line, cur_count = l, 1
    words = re.findall(r"[A-Za-z']+", content.lower())
    sample = words[:200]
    variety = (len(set(sample)) / len(sample)) if sample else 0.0
    return {
        "chars": len(content),
        "tool_call_cascade": cascade,
        "max_line_repeat": max_repeat,
        "lexical_variety": round(variety, 3),
    }


def needle_recall_ok(content: str, secret: str) -> bool:
    """All secret tokens (color, animal, number) present, case-insensitive."""
    low = (content or "").lower()
    return all(tok.lower() in low for tok in secret.split())


def _parse_chat_json(res: HttpResult) -> Tuple[dict, dict, Optional[str]]:
    """(message, usage, finish_reason) out of a non-streaming chat response."""
    d = res.json() or {}
    try:
        choice = (d.get("choices") or [{}])[0]
    except Exception:
        choice = {}
    return (
        choice.get("message") or {},
        d.get("usage") or {},
        choice.get("finish_reason"),
    )


# ===========================================================================
# Test catalog.
# ===========================================================================
@dataclasses.dataclass(frozen=True)
class TestSpec:
    test_id: int
    key: str
    label: str
    #: chat-template dependency: "basic" | "tool-aware" | "thinking-aware"
    chat_template: str
    tools: bool = False
    streaming: bool = False
    thinking: bool = False
    spec_decode: bool = False
    #: approximate prompt-token rung targets for long-ctx tests
    longctx_rungs: Optional[Tuple[int, ...]] = None
    optional: bool = False

    @property
    def crash_prone(self) -> bool:
        return self.test_id in CRASH_PRONE_TESTS

    @property
    def cliff2(self) -> bool:
        return self.test_id in CLIFF2_TESTS

    def deps_json(self) -> dict:
        return {
            "chat_template": self.chat_template,
            "tools": self.tools,
            "streaming": self.streaming,
            "thinking": self.thinking,
            "longctx_rung": list(self.longctx_rungs) if self.longctx_rungs else None,
        }


TEST_CATALOG: Dict[int, TestSpec] = {
    s.test_id: s
    for s in [
        TestSpec(1, "basic", "Basic completion (Paris)", "basic"),
        TestSpec(2, "tool_call", "Tool calling", "tool-aware", tools=True),
        TestSpec(3, "streaming", "Streaming SSE", "basic", streaming=True),
        TestSpec(
            4,
            "agentic_stream",
            "Streaming tool-calls (agentic turn)",
            "tool-aware",
            tools=True,
            streaming=True,
        ),
        TestSpec(
            5, "thinking", "Thinking / reasoning mode", "thinking-aware", thinking=True
        ),
        TestSpec(6, "output_quality", "Output quality / cascade detection", "basic"),
        TestSpec(
            7, "mtp_acceptance", "MTP acceptance length", "basic", spec_decode=True
        ),
        TestSpec(
            8,
            "needle_small",
            "Needle small rungs (10K / 30K)",
            "basic",
            streaming=True,
            longctx_rungs=(10000, 30000),
        ),
        TestSpec(
            9,
            "tool_prefill",
            "Tool-response prefill OOM (~25K tokens)",
            "tool-aware",
            tools=True,
        ),
        TestSpec(
            10,
            "ide_agent",
            "IDE-agent one-shot (Cliff-1 probe)",
            "tool-aware",
            tools=True,
        ),
        TestSpec(11, "multiturn_agent", "Multi-turn agent", "tool-aware", tools=True),
        TestSpec(12, "lcb_coding", "LCB-coding shape", "basic"),
        TestSpec(13, "reasoning_heavy", "Reasoning-heavy (8K budget)", "basic"),
        TestSpec(
            14,
            "needle_large",
            "Needle large rungs (60K / 90K, Cliff-2)",
            "basic",
            streaming=True,
            longctx_rungs=(60000, 90000),
        ),
        TestSpec(
            15,
            "ceiling_ladder",
            "Context ceiling ladder (NIAH)",
            "basic",
            streaming=True,
            longctx_rungs=(95000,),
            optional=True,
        ),
        TestSpec(
            16,
            "throughput",
            "Throughput (narrative + code)",
            "basic",
            streaming=True,
            optional=True,
        ),
    ]
}

#: Preset -> test ids (run order comes from :func:`order_selected`).
PRESETS: Dict[str, Tuple[int, ...]] = {
    "functional": (1, 2, 3, 4, 5, 6, 7),
    "stress": (8, 9, 10, 11, 12, 13, 14, 15),
    "throughput": (16,),
    "full": tuple(range(1, 17)),
}


def order_selected(selected: Iterable[int]) -> List[int]:
    """Run order: ascending ids, with the Cliff-2 tests (14, 15) forced to
    the very END regardless of selection order -- they can crash the engine
    and must not cascade-fail the healthy tests (verify-stress.sh probe
    ordering)."""
    ids = sorted(set(int(i) for i in selected))
    safe = [i for i in ids if i not in CLIFF2_TESTS]
    cliff = [i for i in ids if i in CLIFF2_TESTS]
    return safe + cliff


# ===========================================================================
# Gating (the chat-template matrix).
# ===========================================================================
@dataclasses.dataclass
class GateDecision:
    """Outcome of gating one test against the capabilities.

    ``status`` None -> run the test. Otherwise a terminal skip/blocked
    result with ``reason``. ``expected_fail_note`` marks a test that RUNS
    but whose failure is pre-flagged (missing reasoning parser -> warn)."""

    status: Optional[str] = None
    reason: Optional[str] = None
    expected_fail_note: Optional[str] = None
    #: long-ctx rungs that survive the max_model_len pre-check
    rungs: Optional[List[int]] = None


def gate_test(
    spec: TestSpec, caps: Optional[Capabilities], force: bool = False
) -> GateDecision:
    """Apply the design's chat-template gating matrix. ``force=True`` lifts
    ONLY the tool-parser warn+block (to deliberately surface the cascade
    fail); the basic-template block and the spec-off skip always hold."""
    if caps is None:
        return GateDecision(rungs=list(spec.longctx_rungs or []) or None)

    # Missing basic template blocks ALL chat tests (everything in the
    # catalog goes through /v1/chat/completions).
    if not caps.chat_template_basic:
        return GateDecision(
            status="blocked",
            reason="no working basic chat template (chat ping failed / "
            "template 'none') -- all chat tests blocked",
        )

    if spec.chat_template == "tool-aware" and not caps.tool_parser:
        if not force:
            return GateDecision(
                status="blocked",
                reason="no --tool-call-parser on the server: tool-aware "
                "template missing, test would show the <tool_call> "
                "cascade. Blocked by default; force-run to surface it.",
            )

    note = None
    if spec.chat_template == "thinking-aware" and not caps.reasoning_parser:
        note = (
            "no --reasoning-parser on the server: reasoning_content will "
            "be empty and the >=50-char gate fails. Pre-flagged "
            "expected-fail (reported as warn, not fail)."
        )

    if spec.spec_decode and not caps.spec_decode:
        return GateDecision(
            status="skip",
            reason="speculative decoding is off on this server",
        )

    rungs: Optional[List[int]] = None
    if spec.longctx_rungs:
        rungs = list(spec.longctx_rungs)
        if caps.max_model_len:
            in_budget = [r for r in rungs if r <= caps.max_model_len]
            if not in_budget:
                return GateDecision(
                    status="skip",
                    reason=f"all rungs {rungs} exceed max_model_len="
                    f"{caps.max_model_len} (graceful skip, not a fail)",
                )
            rungs = in_budget

    return GateDecision(expected_fail_note=note, rungs=rungs)


# ===========================================================================
# Runners.
# ===========================================================================
@dataclasses.dataclass
class _RunCtx:
    endpoint: str
    model: str
    caps: Optional[Capabilities]
    http: Callable[..., HttpResult]
    rng: random.Random
    vram_free_mib: Optional[Callable[[], Optional[int]]]
    timeout: float
    #: Every model exchange this run made, in order: the exact request body
    #: sent and the exact answer that came back. A pass/fail verdict is not
    #: reviewable on its own -- the reader has to be able to see what was
    #: asked and what the model actually said. Recorded here because
    #: ``chat()`` is the single funnel for every call the suite makes.
    transcript: List[dict] = dataclasses.field(default_factory=list)
    #: Which test the exchanges being recorded belong to; set by run_suite.
    test_id: Optional[int] = None

    @property
    def base(self) -> str:
        return self.endpoint.rstrip("/")

    def chat(
        self, body: dict, stream: bool = False, timeout: Optional[float] = None
    ) -> HttpResult:
        res = self.http(
            "POST",
            self.base + "/v1/chat/completions",
            body=body,
            stream=stream,
            timeout=timeout or self.timeout,
        )
        self._record(body, res, stream)
        return res

    def _record(self, body: dict, res: HttpResult, stream: bool) -> None:
        """Append one exchange to the transcript. Bodies are stored verbatim;
        an SSE answer is reassembled into the text the client would have seen,
        with the raw lines kept alongside."""
        answer = None
        if stream and res.sse_lines:
            asm = reassemble_stream(res.sse_lines)
            answer = asm.get("content") or None
        elif res.body:
            msg, _usage, _finish = _parse_chat_json(res)
            answer = (msg or {}).get("content") or res.body
        self.transcript.append(
            {
                "test_id": self.test_id,
                "request": body,
                "http_code": res.status,
                "answer": answer,
                "raw_body": None if stream else (res.body or None),
                "sse_lines": list(res.sse_lines) if stream and res.sse_lines else None,
                "ttft_ms": res.ttft_ms,
                "wall_ms": res.wall_ms,
                "error": res.error,
            }
        )


def _metric(
    name: str, value: Any, numeric: Optional[float] = None, unit: Optional[str] = None
) -> dict:
    return {"name": name, "value": value, "numeric": numeric, "unit": unit}


def _detail(res: Optional[HttpResult] = None, **kw: Any) -> dict:
    d: Dict[str, Any] = {
        "prompt_tokens": None,
        "prefill_tps": None,
        "ttft_ms": None,
        "http_code": None,
        "finish": None,
    }
    if res is not None:
        d["http_code"] = res.status
        d["ttft_ms"] = res.ttft_ms
    d.update(kw)
    return d


def _run_basic(ctx: _RunCtx) -> Tuple[str, dict, dict]:
    res = ctx.chat(request_basic(ctx.model), timeout=60)
    msg, usage, finish = _parse_chat_json(res)
    content = msg.get("content") or ""
    detail = _detail(res, finish=finish, prompt_tokens=usage.get("prompt_tokens"))
    if res.status != 200:
        return "fail", _metric("http", res.status), detail
    if re.search(r"paris", content, re.I):
        return "pass", _metric("contains", "Paris"), detail
    detail["content_head"] = content[:80]
    return "fail", _metric("contains", "no 'Paris' in reply"), detail


def _run_tool_call(ctx: _RunCtx) -> Tuple[str, dict, dict]:
    res = ctx.chat(request_tool_call(ctx.model), timeout=90)
    msg, usage, finish = _parse_chat_json(res)
    detail = _detail(res, finish=finish, prompt_tokens=usage.get("prompt_tokens"))
    if res.status != 200:
        return "fail", _metric("http", res.status), detail
    tool_calls = msg.get("tool_calls") or []
    content = msg.get("content") or ""
    if tool_calls:
        names = [(tc.get("function") or {}).get("name") for tc in tool_calls]
        if "get_weather" in names:
            return "pass", _metric("tool_calls", len(tool_calls)), detail
        detail["tool_call_names"] = names
        return "fail", _metric("tool_calls", "unexpected tool"), detail
    if "<tool_call>" in content:
        return (
            "fail",
            _metric(
                "cascade",
                "tool-call cascade: literal <tool_call> in content, tool_calls[] empty",
            ),
            detail,
        )
    detail["content_head"] = content[:120]
    return "fail", _metric("tool_calls", "empty"), detail


def _run_streaming(ctx: _RunCtx) -> Tuple[str, dict, dict]:
    res = ctx.chat(request_streaming(ctx.model), stream=True, timeout=90)
    detail = _detail(res)
    if res.status != 200:
        return "fail", _metric("http", res.status), detail
    st = reassemble_stream(res.sse_lines or [])
    detail["finish"] = st["finish"]
    detail["chunks"] = st["chunks"]
    text = st["content"]
    if st["chunks"] == 0 or not text:
        return "fail", _metric("chunks", 0, 0.0), detail
    if st["chunks"] < 5:
        return (
            "fail",
            _metric(
                "chunks", f"only {st['chunks']} (SSE buffering?)", float(st["chunks"])
            ),
            detail,
        )
    if len(text) < 20:
        return "fail", _metric("chars", len(text), float(len(text))), detail
    return ("pass", _metric("chunks", st["chunks"], float(st["chunks"])), detail)


def _run_agentic_stream(ctx: _RunCtx) -> Tuple[str, dict, dict]:
    res = ctx.chat(request_agentic_turn(ctx.model), stream=True, timeout=180)
    detail = _detail(res)
    if res.status != 200:
        return "fail", _metric("http", res.status), detail
    st = reassemble_stream(res.sse_lines or [])
    usage = st["usage"]
    detail["finish"] = st["finish"]
    detail["prompt_tokens"] = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens") or 0
    if res.ttft_ms and res.wall_ms and completion:
        decode_s = max((res.wall_ms - res.ttft_ms) / 1000.0, 1e-6)
        detail["decode_tps"] = round(completion / decode_s, 1)
    parseable = []
    for tc in st["tool_calls"]:
        try:
            json.loads(tc["arguments"])
            parseable.append(tc["name"])
        except Exception:
            continue
    detail["tool_call_names"] = parseable
    if parseable:
        return (
            "pass",
            _metric("tool_calls", len(parseable), float(len(parseable))),
            detail,
        )
    return (
        "fail",
        _metric("tool_calls", "no parseable tool call despite tool_choice=required"),
        detail,
    )


def _run_thinking(ctx: _RunCtx) -> Tuple[str, dict, dict]:
    res = ctx.chat(request_thinking(ctx.model), timeout=180)
    msg, usage, finish = _parse_chat_json(res)
    detail = _detail(res, finish=finish, prompt_tokens=usage.get("prompt_tokens"))
    if res.status != 200:
        return "fail", _metric("http", res.status), detail
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    detail["reasoning_chars"] = len(reasoning)
    detail["content_chars"] = len(content)
    if len(reasoning) < 50:
        return (
            "fail",
            _metric("reasoning_chars", len(reasoning), float(len(reasoning))),
            detail,
        )
    if not content and finish != "length":
        return ("fail", _metric("content", f"empty with finish={finish}"), detail)
    return (
        "pass",
        _metric("reasoning_chars", len(reasoning), float(len(reasoning))),
        detail,
    )


def _run_quality(ctx: _RunCtx) -> Tuple[str, dict, dict]:
    res = ctx.chat(request_quality(ctx.model), timeout=240)
    msg, usage, finish = _parse_chat_json(res)
    detail = _detail(res, finish=finish, prompt_tokens=usage.get("prompt_tokens"))
    if res.status != 200:
        return "fail", _metric("http", res.status), detail
    content = msg.get("content") or ""
    qa = analyze_output_quality(content)
    detail.update(qa)
    if qa["chars"] == 0:
        return "fail", _metric("chars", 0, 0.0), detail
    if qa["tool_call_cascade"]:
        return (
            "fail",
            _metric("cascade", "tool-call cascade: <tool_call> emitted in normal text"),
            detail,
        )
    if qa["max_line_repeat"] >= 5:
        return (
            "fail",
            _metric(
                "max_line_repeat", qa["max_line_repeat"], float(qa["max_line_repeat"])
            ),
            detail,
        )
    if qa["lexical_variety"] < 0.30:
        return (
            "fail",
            _metric("lexical_variety", qa["lexical_variety"], qa["lexical_variety"]),
            detail,
        )
    return (
        "pass",
        _metric("lexical_variety", qa["lexical_variety"], qa["lexical_variety"]),
        detail,
    )


def _run_mtp_acceptance(ctx: _RunCtx) -> Tuple[str, dict, dict]:
    """Drive a fresh decode, then read the acceptance-length EMA gauge from
    Prometheus /metrics (metric names pinned via live_metrics.py)."""
    res = ctx.chat(request_mtp_trigger(ctx.model), timeout=120)
    detail = _detail(res)
    if res.status != 200:
        return "fail", _metric("http", res.status), detail
    _, usage, finish = _parse_chat_json(res)
    detail["finish"] = finish
    detail["prompt_tokens"] = usage.get("prompt_tokens")

    mres = ctx.http("GET", ctx.base + "/metrics", timeout=15)
    if mres.status != 200:
        detail["metrics_http_code"] = mres.status
        return (
            "skip",
            _metric(
                "acceptance_length",
                "/metrics unreachable (server without --enable-metrics?)",
            ),
            detail,
        )
    flat = parse_prometheus_metrics(mres.body)
    if SPEC_EMA_ACCEPT_LEN_METRIC not in flat:
        return (
            "skip",
            _metric(
                "acceptance_length",
                f"{SPEC_EMA_ACCEPT_LEN_METRIC} absent from /metrics "
                "(spec metrics not exported)",
            ),
            detail,
        )
    al = flat[SPEC_EMA_ACCEPT_LEN_METRIC]
    detail["spec_accept_rate"] = flat.get(SPEC_ACCEPT_RATE_METRIC)
    detail["spec_num_steps"] = flat.get(SPEC_NUM_STEPS_METRIC)
    if ctx.caps is not None:
        detail["spec_mode"] = ctx.caps.spec_mode
    status = "pass" if al >= 2.0 else "fail"
    return status, _metric("acceptance_length", round(al, 2), al, "tokens"), detail


def _needle_rung(ctx: _RunCtx, target_tokens: int, timeout: float) -> Dict[str, Any]:
    """One NIAH rung: build request, stream it, judge recall. Returns a
    per-rung record for the test detail."""
    secret = make_needle_secret(ctx.rng)
    scale = max(100, target_tokens // TOKENS_PER_FILLER_SCALE)
    res = ctx.chat(
        request_needle(ctx.model, scale, secret), stream=True, timeout=timeout
    )
    rec: Dict[str, Any] = {
        "target_tokens": target_tokens,
        "http_code": res.status,
        "ttft_ms": res.ttft_ms,
    }
    if res.status == 400:
        rec["outcome"] = "skip"
        rec["note"] = "HTTP 400 (exceeds engine limit -- clean rejection)"
        return rec
    if res.status != 200:
        rec["outcome"] = "fail"
        rec["note"] = f"HTTP {res.status} (system failure)"
        return rec
    st = reassemble_stream(res.sse_lines or [])
    usage = st["usage"]
    rec["prompt_tokens"] = usage.get("prompt_tokens")
    if res.ttft_ms and usage.get("prompt_tokens"):
        rec["prefill_tps"] = round(usage["prompt_tokens"] / (res.ttft_ms / 1000.0), 1)
    if needle_recall_ok(st["content"], secret):
        rec["outcome"] = "pass"
    else:
        # Recall miss at HTTP 200 = attention-quality info, NOT a failure.
        rec["outcome"] = "info"
        rec["note"] = "recall MISS -- system OK, quality ceiling reached"
        rec["content_head"] = st["content"][:80]
    return rec


def _needle_result(rungs_out: List[dict], detail: dict) -> Tuple[str, dict, dict]:
    """Aggregate rung outcomes with the tri-state discipline."""
    detail["rungs"] = rungs_out
    if rungs_out:
        last = rungs_out[-1]
        detail["http_code"] = last.get("http_code")
        detail["prompt_tokens"] = last.get("prompt_tokens")
        detail["prefill_tps"] = last.get("prefill_tps")
        detail["ttft_ms"] = last.get("ttft_ms")
    outcomes = [r["outcome"] for r in rungs_out]
    passed = [r for r in rungs_out if r["outcome"] == "pass"]
    deepest = max(
        (r.get("prompt_tokens") or r["target_tokens"] for r in passed),
        default=0,
    )
    if "fail" in outcomes:
        return (
            "fail",
            _metric("needle", "system failure (HTTP 5xx / timeout)"),
            detail,
        )
    if "info" in outcomes:
        return (
            "info",
            _metric(
                "recall_ceiling_tokens",
                deepest or None,
                float(deepest) if deepest else None,
                "tokens",
            ),
            detail,
        )
    if passed:
        return (
            "pass",
            _metric("deepest_recall_tokens", deepest, float(deepest), "tokens"),
            detail,
        )
    return (
        "skip",
        _metric("needle", "all rungs rejected by engine pre-check (HTTP 400)"),
        detail,
    )


def _run_needle(ctx: _RunCtx, rungs: List[int]) -> Tuple[str, dict, dict]:
    out: List[dict] = []
    for target in rungs:
        rec = _needle_rung(ctx, target, timeout=600)
        out.append(rec)
        if rec["outcome"] in ("info", "fail", "skip"):
            # deeper rungs will only be worse / also rejected
            break
    return _needle_result(out, _detail())


def _run_tool_prefill(ctx: _RunCtx) -> Tuple[str, dict, dict]:
    res = ctx.chat(request_tool_prefill(ctx.model), timeout=480)
    msg, usage, finish = _parse_chat_json(res)
    detail = _detail(res, finish=finish, prompt_tokens=usage.get("prompt_tokens"))
    if res.status == 200:
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []
        detail["content_chars"] = len(content)
        detail["tool_calls"] = len(tool_calls)
        if len(content) >= 50 or tool_calls:
            return "pass", _metric("prefill", "survived ~25K-token prefill"), detail
        return (
            "fail",
            _metric(
                "prefill", "HTTP 200 but empty response (silent prefill truncation)"
            ),
            detail,
        )
    if res.status == 500:
        return (
            "fail",
            _metric(
                "prefill", "HTTP 500 -- OOM during ~25K-token tool-response prefill"
            ),
            detail,
        )
    if res.status == 0:
        return (
            "fail",
            _metric("prefill", "no HTTP response (timeout or engine died)"),
            detail,
        )
    return "fail", _metric("http", res.status), detail


def _http200_probe(
    ctx: _RunCtx, body: dict, timeout: float, fail_hint_500: str
) -> Tuple[str, dict, dict]:
    """Shared shape for the crash probes 10/11/12: any HTTP 200 passes."""
    res = ctx.chat(body, timeout=timeout)
    msg, usage, finish = _parse_chat_json(res)
    detail = _detail(res, finish=finish, prompt_tokens=usage.get("prompt_tokens"))
    detail["completion_tokens"] = usage.get("completion_tokens")
    if res.status == 200:
        return "pass", _metric("http", 200, 200.0), detail
    if res.status == 500:
        return "fail", _metric("http", f"500 -- {fail_hint_500}", 500.0), detail
    if res.status == 0:
        return (
            "fail",
            _metric("http", "no response (timeout or engine died)", 0.0),
            detail,
        )
    return "fail", _metric("http", res.status, float(res.status)), detail


def _run_ide_agent(ctx: _RunCtx) -> Tuple[str, dict, dict]:
    return _http200_probe(
        ctx,
        request_ide_agent(ctx.model),
        240,
        "likely Cliff-1 mech B (inductor FFN intermediate OOM)",
    )


def _run_multiturn_agent(ctx: _RunCtx) -> Tuple[str, dict, dict]:
    return _http200_probe(
        ctx,
        request_multiturn_agent(ctx.model),
        240,
        "multi-turn prefill crashed the engine",
    )


def _run_lcb_coding(ctx: _RunCtx) -> Tuple[str, dict, dict]:
    return _http200_probe(
        ctx,
        request_lcb_coding(ctx.model),
        300,
        "LCB-coding shape crashed the engine (DS conv-state class)",
    )


def _run_reasoning_heavy(ctx: _RunCtx) -> Tuple[str, dict, dict]:
    res = ctx.chat(request_reasoning_heavy(ctx.model), timeout=700)
    msg, usage, finish = _parse_chat_json(res)
    detail = _detail(res, finish=finish, prompt_tokens=usage.get("prompt_tokens"))
    if res.status != 200:
        if res.status == 500:
            return (
                "fail",
                _metric("http", "500 -- long generation crashed the engine", 500.0),
                detail,
            )
        return "fail", _metric("http", res.status, float(res.status)), detail
    completion = usage.get("completion_tokens") or 0
    detail["completion_tokens"] = completion
    if completion < 500:
        return (
            "fail",
            _metric("completion_tokens", completion, float(completion), "tokens"),
            detail,
        )
    return (
        "pass",
        _metric("completion_tokens", completion, float(completion), "tokens"),
        detail,
    )


def _ceiling_margin_level(free_mib: int):
    """``(phase, level_name, level_mib, net_mib, band_floor_mib)`` for the
    ceiling-ladder margin note.

    THE DEFECT THIS REPLACES (#784/#602). The line used to read
    ``vram_after < 1024``: a hardcoded bound on FREE VRAM, which is the
    phase-dependent quantity, measured as a single reading after the ladder.
    #602 named that exact number as overstating the gap on flip boots -- the
    binding level there was the arming floor at 1728/1825/2467 MiB, not 1024.
    A warning that fires systematically wrongly devalues the warnings that are
    right, which is the #739 alarm-noise economy.

    So the bound is now the BINDING LEVEL: the arming floor where the boot has
    one, the band floor where it does not. Both come from
    ``corridor_guard`` -- imported, never re-derived. A second authority for
    the same quantity is how the verdict and the runtime came to disagree
    about the band ceiling by 1 MiB, and it would be tomorrow's root.

    The comparison is corridor_guard's own: NET free (free minus the arming
    reserve the flip has committed) against the band floor. That is
    algebraically the same test as ``free < arming_floor`` and it is written
    the way the shipped verdict writes it, so the two cannot drift.

    Presence of a phase marker is what says whether an arming floor applies:
    a boot with no flip has no seam to enter and is graded on the band alone.
    """
    band_floor = 819
    phase = "unknown"
    level_name, level = "band_floor", band_floor
    try:
        from sglang.srt.managers.corridor_guard import (
            arming_floor_mib,
            corridor_band_floor_mib,
            net_free_mib,
        )

        band_floor = corridor_band_floor_mib()
        level_name, level = "band_floor", band_floor
        try:
            from sglang.srt.managers.phase_flip_presence import read_active_phase

            live = read_active_phase()
        except Exception:  # noqa: BLE001 - an instrument must not raise
            live = None
        if live:
            phase = live
            level_name, level = "arming_floor", arming_floor_mib()
        return phase, level_name, level, net_free_mib(free_mib, level), band_floor
    except Exception:  # noqa: BLE001
        # corridor_guard unavailable: grade on the band alone and SAY so,
        # rather than falling back to the 1024 this function exists to remove.
        return phase, "band_floor(fallback)", band_floor, int(free_mib), band_floor


def _run_ceiling_ladder(ctx: _RunCtx, start_rungs: List[int]) -> Tuple[str, dict, dict]:
    """Test 15 -- staggered NIAH from ~95K up to ~92% of n_ctx in 30K steps,
    with a free-VRAM margin check against the BINDING LEVEL after the
    ladder -- the arming floor where the boot has one, the band floor where it
    does not. Never a hardcoded 1024: see _ceiling_margin_level."""
    detail = _detail()
    n_ctx = ctx.caps.max_model_len if ctx.caps else None
    start = start_rungs[0] if start_rungs else 95000
    if not n_ctx:
        return ("skip", _metric("ceiling", "n_ctx unknown (no max_model_len)"), detail)
    top = int(n_ctx * 0.92)
    if top <= start:
        return (
            "skip",
            _metric(
                "ceiling",
                f"ceiling target {top} <= start {start} "
                "(needle-large already covers this range)",
            ),
            detail,
        )
    rungs = list(range(start, top, 30000))
    if not rungs or rungs[-1] != top:
        rungs.append(top)
    detail["ladder"] = rungs

    vram_before = ctx.vram_free_mib() if ctx.vram_free_mib else None
    out: List[dict] = []
    for target in rungs:
        rec = _needle_rung(ctx, target, timeout=900)
        out.append(rec)
        if rec["outcome"] != "pass":
            break
    status, metric, detail = _needle_result(out, detail)
    vram_after = ctx.vram_free_mib() if ctx.vram_free_mib else None
    detail["vram_free_mib_before"] = vram_before
    detail["vram_free_mib_after"] = vram_after
    if status == "pass" and vram_after is not None:
        phase, level_name, level, net, floor = _ceiling_margin_level(vram_after)
        if net < floor:
            # Recall fine but margin thin: informational warning, not a hard
            # fail. WARN-only is deliberate -- this is an instrument, not the
            # runtime path.
            detail["vram_margin_note"] = (
                f"phase={phase} level={level_name}({level} MiB): free "
                f"{vram_after} MiB, net {net} MiB < band floor {floor} MiB "
                f"at ceiling"
            )
            return "warn", metric, detail
    return status, metric, detail


def _run_throughput(ctx: _RunCtx) -> Tuple[str, dict, dict]:
    """Test 16 -- one measured streamed run per canonical prompt.
    Informational: reports decode TPS, never fails on speed."""
    detail = _detail()
    runs: Dict[str, dict] = {}
    decode_vals: List[float] = []
    for kind, _, _ in THROUGHPUT_PROMPTS:
        res = ctx.chat(request_throughput(ctx.model, kind), stream=True, timeout=600)
        rec: Dict[str, Any] = {
            "http_code": res.status,
            "ttft_ms": res.ttft_ms,
            "wall_ms": res.wall_ms,
        }
        if res.status != 200:
            runs[kind] = rec
            detail["runs"] = runs
            return ("fail", _metric("http", f"{kind}: HTTP {res.status}"), detail)
        st = reassemble_stream(res.sse_lines or [])
        usage = st["usage"]
        rec["prompt_tokens"] = usage.get("prompt_tokens")
        rec["completion_tokens"] = usage.get("completion_tokens")
        if res.wall_ms and usage.get("completion_tokens"):
            rec["wall_tps"] = round(
                usage["completion_tokens"] / (res.wall_ms / 1000.0), 1
            )
            if res.ttft_ms is not None:
                decode_s = max((res.wall_ms - res.ttft_ms) / 1000.0, 1e-6)
                rec["decode_tps"] = round(usage["completion_tokens"] / decode_s, 1)
                decode_vals.append(rec["decode_tps"])
        runs[kind] = rec
    detail["runs"] = runs
    detail["http_code"] = 200
    if decode_vals:
        mean = round(sum(decode_vals) / len(decode_vals), 1)
        return ("info", _metric("decode_tps_mean", mean, float(mean), "tok/s"), detail)
    return ("info", _metric("decode_tps_mean", "no usage data from stream"), detail)


# ===========================================================================
# Suite driver.
# ===========================================================================
def _engine_healthy(ctx: _RunCtx) -> bool:
    res = ctx.http("GET", ctx.base + "/v1/models", timeout=10)
    return res.status == 200


def _result_dict(
    spec: TestSpec,
    status: str,
    metric: Optional[dict] = None,
    detail: Optional[dict] = None,
    reason: Optional[str] = None,
) -> dict:
    out = {
        "test_id": spec.test_id,
        "label": spec.label,
        "status": status,
        "metric": metric or _metric("none", None),
        "detail": detail or _detail(),
        "deps": spec.deps_json(),
    }
    if reason:
        out["reason"] = reason
    return out


def run_suite(
    endpoint: str,
    model: str,
    selected: Optional[Iterable[int]] = None,
    capabilities: Optional[Capabilities] = None,
    progress_cb: Optional[Callable[[dict], None]] = None,
    *,
    preset: Optional[str] = None,
    force: bool = False,
    http: Optional[Callable[..., HttpResult]] = None,
    rng: Optional[random.Random] = None,
    vram_free_mib: Optional[Callable[[], Optional[int]]] = None,
    probe: bool = True,
    timeout: float = 120.0,
    transcript_sink: Optional[list] = None,
) -> Iterator[dict]:
    """Run the selected tests against a live OpenAI-compatible sglang server,
    YIELDING one result dict per test as it finishes (the webui layer streams
    these to the browser via SSE; ``progress_cb`` gets the same dicts).

    ``selected`` is a list of test ids (or use ``preset`` from
    :data:`PRESETS`). Execution order comes from :func:`order_selected`
    (Cliff-2 tests always last). ``capabilities`` gates the run; when None
    and ``probe=True`` they are probed from the server first.

    After every crash-prone probe an engine-health recheck runs; if the
    engine died, all remaining tests are yielded as ``skip`` -- this module
    NEVER restarts a server.

    All network access goes through the injectable ``http`` transport, so the
    whole suite is unit-testable with canned :class:`HttpResult` objects.
    """
    http = http or _default_http
    if selected is None:
        selected = PRESETS[preset or "full"]
    ids = order_selected(selected)

    caps = capabilities
    if caps is None and probe:
        caps = probe_capabilities(endpoint, http=http)
        if model and caps.model is None:
            caps.model = model

    ctx = _RunCtx(
        endpoint=endpoint,
        model=model,
        caps=caps,
        http=http,
        rng=rng or random.Random(),
        vram_free_mib=vram_free_mib,
        timeout=timeout,
    )
    # The caller (the webui run route) reads ctx.transcript after the
    # generator is exhausted and persists it with the run.
    if transcript_sink is not None:
        transcript_sink.append(ctx)

    runners: Dict[int, Callable[..., Tuple[str, dict, dict]]] = {
        1: _run_basic,
        2: _run_tool_call,
        3: _run_streaming,
        4: _run_agentic_stream,
        5: _run_thinking,
        6: _run_quality,
        7: _run_mtp_acceptance,
        9: _run_tool_prefill,
        10: _run_ide_agent,
        11: _run_multiturn_agent,
        12: _run_lcb_coding,
        13: _run_reasoning_heavy,
        16: _run_throughput,
    }

    engine_down_after: Optional[int] = None
    for tid in ids:
        spec = TEST_CATALOG[tid]

        if engine_down_after is not None:
            result = _result_dict(
                spec,
                "skip",
                reason=f"engine unhealthy after test {engine_down_after}; "
                "not restarting (manual restart required)",
            )
            if progress_cb:
                progress_cb(result)
            yield result
            continue

        decision = gate_test(spec, caps, force=force)
        if decision.status is not None:
            result = _result_dict(spec, decision.status, reason=decision.reason)
            if progress_cb:
                progress_cb(result)
            yield result
            continue

        ctx.test_id = tid
        try:
            if tid in (8, 14):
                status, metric, detail = _run_needle(
                    ctx, decision.rungs or list(spec.longctx_rungs or [])
                )
            elif tid == 15:
                status, metric, detail = _run_ceiling_ladder(
                    ctx, decision.rungs or list(spec.longctx_rungs or [])
                )
            else:
                status, metric, detail = runners[tid](ctx)
        except Exception as e:  # defensive: a runner bug must not kill the SSE
            status, metric, detail = ("fail", _metric("exception", str(e)), _detail())

        reason = None
        if status == "fail" and decision.expected_fail_note:
            status = "warn"
            reason = decision.expected_fail_note
        result = _result_dict(spec, status, metric, detail, reason=reason)
        if progress_cb:
            progress_cb(result)
        yield result

        if spec.crash_prone and not _engine_healthy(ctx):
            engine_down_after = tid
