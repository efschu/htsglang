#!/usr/bin/env python3
"""s16 -- one STRUCTURED-OUTPUT point against a live server, task #285.

Why this exists next to s14_decode_punkt.py. That file pins a synthetic decode
loop: one shared 2048-token prose prefix, ``ignore_eos``, nothing ever
finishes, and the working point is held perfectly still. That is the right
instrument for a TRANSPORT question, where the content must not move. It is
the wrong instrument for the question this step asks, because the question IS
about content:

    DFLASH is weak on prose and is supposed to earn its acceptance on
    format-constrained text -- code, JSON, tables. Does it, measured against
    NEXTN on identical prompts?

So the load here is real requests with real, finite, checkable answers, and
three things follow from that:

* THE CONTENT AXIS IS A COLUMN, NEVER AN AVERAGE. One point is one
  (arm, batch size, content class) triple. Averaging a code number and a prose
  number produces a figure no workload ever sees. The class is written into
  every record and the analysis groups by it.
* EVERY POINT IS OUTPUT-VALIDATED. A fast run that emits garbage looks
  excellent in a throughput table. Each completed request is parsed against
  its class's structural contract (Python/Bash parse, strict JSON plus
  required keys, row count of the requested kind); a point whose valid share
  falls under ``--min-valid-ratio`` is written out with ``counted: false`` and
  the analysis drops it.
* NOTHING IS FED BACK IN. The prompt set is independent by construction (see
  prompts/structured_v1.json) and this driver never uses a model output as a
  follow-up prompt. That is the #156 self-conditioning trap, and it is the one
  error that cannot be repaired after the fact.

TWO MEASUREMENT LEVELS, both kept and both labelled, as in s14:

* THE TICK LEVEL is primary for rate and accept. The scheduler's own
  ``Decode batch`` line is windowed by construction and carries ``accept len``
  (which is ``spec_num_accept_tokens / spec_num_forward_ct`` over one log
  interval -- NOT the EMA the same line can also print) and ``gen throughput``.
  Ticks whose ``#running-req`` is not the point's batch size are COUNTED and
  reported, never silently dropped: a window that did not hold its working
  point has to say so in the point.
* THE CLIENT LEVEL is the independent second opinion, and here it can do
  something s14's client level could not: because these requests FINISH inside
  the window, their final ``meta_info`` carries ``spec_accept_length`` and
  ``spec_verify_ct``. The point's client accept is the pooled
  ``sum(completion_tokens) / sum(spec_verify_ct)`` over the requests that both
  started and ended inside the window -- a genuine windowed accept at the
  point's own batch size, not a bs=1 probe afterwards.

ON ``ms/Verify``. Kept bit-for-bit as s12/s14 define it,
``accept_len * 1000 / gen_tok_s``, so the numbers of this window can be laid
next to the numbers of the transport windows. ``gen_tok_s`` is the WHOLE
batch's rate, so this is the verify step time PER REQUEST; the wall time of one
step is ``bs`` times larger and is reported separately as ``ms_per_step``.

WHY THE BATCH IS A POOL AND NOT A FIXED SET OF STREAMS. Structured answers are
short and finish; a fixed set of ``bs`` streams would drain and the tail of the
window would be measured at a smaller batch. A pool of ``bs`` workers that
immediately starts the next prompt keeps ``#running-req`` at ``bs`` for almost
the whole window, and the ticks that still land at another batch size are
reported rather than hidden.

Standard library only: this runs on the PVE host, outside any venv.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from s12_log_analyse import im_fenster, parse_decode  # noqa: E402  one parser

KIND = "s16_structured"
SCHEMA_VERSION = 1

# Warmup content, deliberately NOT from the prompt set. Warming on the measured
# prompts would put them in the radix cache before the window opens and the
# point would measure a cache hit; warming on something else leaves the first
# pass over the set as honest as any first pass can be. The ramp requests that
# follow the flush do populate the cache for later repetitions inside the same
# window -- identically on every arm, since prompt order and concurrency are
# pinned -- and that is stated here rather than pretended away.
WARMUP_PROMPT = (
    "Beschreibe in drei Saetzen, wie ein Dekodierschritt eines "
    "Transformer-Modells ablaeuft."
)
WARMUP_MAX_NEW_TOKENS = 96

# How many raw answers per point land on disk. Enough to read, small enough to
# keep in the run directory next to everything else.
SAMPLES_PER_POINT = 3


# ---------------------------------------------------------------------------
# http
# ---------------------------------------------------------------------------


def _get(port: int, path: str, timeout: float) -> str:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(errors="replace")


def _post(port: int, path: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode(errors="replace")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body[:500]}


def _flush_cache(port: int) -> str:
    """Empty the radix cache between warmup and window. Best effort.

    Not fatal when it fails: a stale cache makes both arms equally fast at
    prefill, so it costs realism, not comparability. It IS reported, so a run
    that silently kept a warm cache can be recognised afterwards.
    """
    last = "no attempt"
    for method in ("POST", "GET"):
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/flush_cache",
                data=b"{}" if method == "POST" else None,
                headers={"Content-Type": "application/json"},
                method=method,
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return f"{method} {resp.status}"
        except (urllib.error.URLError, OSError) as exc:
            last = f"{method}: {type(exc).__name__}"
    return last


def _metrics_snapshot(port: int) -> dict:
    keep = (
        "sglang:num_running_reqs",
        "sglang:num_queue_reqs",
        "sglang:gen_throughput",
        "sglang:spec_accept_length",
        "sglang:token_usage",
        "sglang:cache_hit_rate",
    )
    out: dict = {}
    try:
        body = _get(port, "/metrics", timeout=20)
    except (urllib.error.URLError, OSError) as exc:
        return {"metrics_error": f"{type(exc).__name__}: {exc}"}
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name not in keep:
            continue
        try:
            out[name] = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# output validation
# ---------------------------------------------------------------------------
#
# The contract per class, and nothing beyond it. These validators ask "is this
# the SHAPE that was requested", not "is the content good": a syntactically
# valid function that computes the wrong thing still proves the model held the
# format, which is the property the DFLASH claim is about. Judging content
# would need a second model and would make the gate non-deterministic.

_FENCE = re.compile(r"```(?P<lang>[a-zA-Z0-9_+-]*)\s*\n(?P<body>.*?)(?:```|\Z)", re.S)
_ROW_BULLET = re.compile(r"^\s*[-*]\s+\S")
_ROW_NUMBERED = re.compile(r"^\s*\d+[.)]\s+\S")
_TABLE_SEPARATOR = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def extract_code(text: str) -> str:
    """The first fenced block, or the whole answer when there is no fence."""
    m = _FENCE.search(text or "")
    if m:
        return m.group("body")
    return text or ""


def extract_json(text: str) -> str:
    """The first balanced JSON value in the answer.

    Fences are stripped first, then the span from the first ``{``/``[`` to the
    matching close is cut out. Models that add a sentence before or after the
    object are still counted as valid: the prompt asks for bare JSON, but a
    leading "Here is the JSON:" is a politeness failure, not a structure
    failure, and this step measures structure.
    """
    body = extract_code(text)
    starts = [i for i in (body.find("{"), body.find("[")) if i >= 0]
    if not starts:
        return body.strip()
    start = min(starts)
    opener = body[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(body)):
        ch = body[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return body[start : i + 1]
    return body[start:].strip()


def count_rows(text: str, row_kind: str) -> int:
    lines = (text or "").splitlines()
    if row_kind == "bullet":
        return sum(1 for line in lines if _ROW_BULLET.match(line))
    if row_kind == "numbered":
        return sum(1 for line in lines if _ROW_NUMBERED.match(line))
    if row_kind == "table":
        pipes = [line for line in lines if line.strip().startswith("|")]
        separators = [line for line in pipes if _TABLE_SEPARATOR.match(line)]
        # header + separator are not data. Without a separator line the block
        # is not a markdown table at all, and every pipe row is counted as
        # data so the shortfall shows up as a row count, not as a crash.
        if separators:
            return max(0, len(pipes) - len(separators) - 1)
        return len(pipes)
    return 0


def _bash_syntax_ok(code: str) -> tuple:
    fd, path = tempfile.mkstemp(suffix=".sh")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)
        try:
            proc = subprocess.run(
                ["bash", "-n", path], capture_output=True, text=True, timeout=20
            )
        except FileNotFoundError:
            return False, "bash_unavailable"
        except subprocess.TimeoutExpired:
            return False, "bash_timeout"
        if proc.returncode == 0:
            return True, ""
        return False, "bash_syntax: " + (proc.stderr or "").strip()[:160]
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def validate_output(text: str, validator: dict) -> tuple:
    """(ok, reason). ``reason`` is empty exactly when ok is True."""
    kind = (validator or {}).get("kind")
    if not kind:
        return False, "no_validator"

    if kind in ("python_syntax", "bash_syntax"):
        code = extract_code(text).strip()
        min_chars = int(validator.get("min_chars", 0))
        if len(code) < min_chars:
            return False, f"too_short: {len(code)} < {min_chars}"
        if kind == "python_syntax":
            try:
                ast.parse(code)
            except SyntaxError as exc:
                return False, f"python_syntax: {exc.msg} (line {exc.lineno})"
            except ValueError as exc:
                return False, f"python_syntax: {type(exc).__name__}"
            return True, ""
        return _bash_syntax_ok(code)

    if kind == "json_object":
        blob = extract_json(text)
        if not blob:
            return False, "json_empty"
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError as exc:
            return False, f"json_parse: {exc.msg} (pos {exc.pos})"
        want = validator.get("top_level", "object")
        if want == "object" and not isinstance(obj, dict):
            return False, f"json_toplevel: {type(obj).__name__} != object"
        if want == "array" and not isinstance(obj, list):
            return False, f"json_toplevel: {type(obj).__name__} != array"
        if isinstance(obj, dict):
            missing = [k for k in validator.get("required_keys", []) if k not in obj]
            if missing:
                return False, "json_missing_keys: " + ",".join(missing)
        if isinstance(obj, list):
            min_items = int(validator.get("min_items", 0))
            if len(obj) < min_items:
                return False, f"json_items: {len(obj)} < {min_items}"
            item_keys = validator.get("item_required_keys", [])
            for i, item in enumerate(obj):
                if not isinstance(item, dict):
                    return False, f"json_item{i}: {type(item).__name__} != object"
                missing = [k for k in item_keys if k not in item]
                if missing:
                    return False, f"json_item{i}_missing: " + ",".join(missing)
        return True, ""

    if kind == "rows":
        row_kind = validator.get("row_kind", "bullet")
        min_rows = int(validator.get("min_rows", 1))
        rows = count_rows(text, row_kind)
        if rows < min_rows:
            return False, f"rows_{row_kind}: {rows} < {min_rows}"
        return True, ""

    return False, f"unknown_validator: {kind}"


# ---------------------------------------------------------------------------
# tick aggregation
# ---------------------------------------------------------------------------


def tick_aggregate(ticks: list, bs: int, drop_edges: int = 1) -> dict:
    """Per-tick decode metrics of ONE window, at exactly this batch size.

    Same two rules as s14: the edge ticks are dropped because their log
    interval is cut by the window boundary and is therefore systematically
    slower, and ticks at another batch size are COUNTED rather than quietly
    removed -- a window that did not hold its working point must show that in
    the point instead of in a smaller sample.
    """
    matching = [t for t in ticks if t["running_req"] == bs]
    other = len(ticks) - len(matching)
    core = matching[drop_edges:-drop_edges] if drop_edges else matching
    if len(core) < 2:
        core = matching
    if not core:
        return {"ticks_window": len(ticks), "ticks_bs": 0, "ticks_counted": 0}
    rate = [t["gen_tok_s"] for t in core]
    accept = [t["accept_len"] for t in core]
    rate_med = statistics.median(rate)
    accept_med = statistics.median(accept)
    out = {
        "ticks_window": len(ticks),
        "ticks_bs": len(matching),
        "ticks_other_bs": other,
        "ticks_counted": len(core),
        "gen_tok_s_median": rate_med,
        "gen_tok_s_min": min(rate),
        "gen_tok_s_max": max(rate),
        "accept_len_median": accept_med,
        "cuda_graph": all(t["cuda_graph"] for t in core),
    }
    if len(rate) > 2:
        out["gen_tok_s_stdev"] = statistics.stdev(rate)
    if rate_med:
        out["ms_per_token"] = 1000.0 / rate_med
        out["ms_per_verify"] = 1000.0 / rate_med * accept_med
        out["ms_per_step"] = 1000.0 / rate_med * accept_med * bs
    return out


def _harvest_ticks(server_log: str, start: float, end: float, bs: int) -> dict:
    if not server_log or not os.path.exists(server_log):
        return {"tick_error": "no server log", "tick_source": server_log or None}
    try:
        with open(server_log, errors="replace") as f:
            ticks = parse_decode(f)
    except OSError as exc:
        return {"tick_error": f"{type(exc).__name__}: {exc}"}
    window = im_fenster(ticks, start, end)
    out = {"tick_source": server_log}
    out.update({f"tick_{k}": v for k, v in tick_aggregate(window, bs).items()})
    return out


# ---------------------------------------------------------------------------
# prompt set
# ---------------------------------------------------------------------------


def load_prompts(path: str, content_class: str) -> tuple:
    with open(path, errors="replace") as f:
        doc = json.load(f)
    prompts = [p for p in doc.get("prompts", []) if p.get("class") == content_class]
    prompts.sort(key=lambda p: p["id"])  # pinned order, identical on every arm
    return doc.get("name", os.path.basename(path)), prompts


# ---------------------------------------------------------------------------
# the load
# ---------------------------------------------------------------------------


class Pool:
    """``bs`` workers walking the class's prompts in a pinned round-robin.

    The counter is shared and monotone, so the SEQUENCE of prompts issued is a
    function of (class, bs, how many requests fit in the window) and of nothing
    else. Two arms that reach the same request count therefore ran the same
    prompts in the same order; an arm that is slower runs a prefix of what the
    faster one ran, and the per-prompt breakdown in the record makes that
    visible rather than letting it slide into the average.
    """

    def __init__(self, port: int, prompts: list, bs: int, default_max_new: int):
        self.port = port
        self.prompts = prompts
        self.bs = bs
        self.default_max_new = default_max_new
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self._next = 0
        self.records: list = []
        self.errors: list = []

    def _take(self) -> dict:
        with self._lock:
            p = self.prompts[self._next % len(self.prompts)]
            self._next += 1
        return p

    def _worker(self) -> None:
        while not self.stop.is_set():
            p = self._take()
            t0 = time.monotonic()
            try:
                answer = _post(
                    self.port,
                    "/generate",
                    {
                        "text": p["prompt"],
                        "sampling_params": {
                            "max_new_tokens": int(
                                p.get("max_new_tokens", self.default_max_new)
                            ),
                            "temperature": 0,
                        },
                    },
                    timeout=300,
                )
            except (urllib.error.URLError, OSError) as exc:
                self.errors.append(f"{p['id']}: {type(exc).__name__}: {exc}"[:200])
                continue
            t1 = time.monotonic()
            if isinstance(answer, dict) and isinstance(answer.get("error"), dict):
                self.errors.append(
                    f"{p['id']}: {str(answer['error'].get('message'))[:150]}"
                )
                continue
            meta = (answer.get("meta_info") or {}) if isinstance(answer, dict) else {}
            self.records.append(
                {
                    "prompt_id": p["id"],
                    "lang": p.get("lang"),
                    "t_start": t0,
                    "t_end": t1,
                    "e2e_ms": (t1 - t0) * 1000.0,
                    "text": answer.get("text") if isinstance(answer, dict) else None,
                    "completion_tokens": meta.get("completion_tokens"),
                    "prompt_tokens": meta.get("prompt_tokens"),
                    "cached_tokens": meta.get("cached_tokens"),
                    "spec_accept_length": meta.get("spec_accept_length"),
                    "spec_verify_ct": meta.get("spec_verify_ct"),
                    "finish_reason": (
                        (meta.get("finish_reason") or {}).get("type")
                        if isinstance(meta.get("finish_reason"), dict)
                        else meta.get("finish_reason")
                    ),
                    "validator": p.get("validator"),
                }
            )

    def run_for(self, seconds: float) -> None:
        threads = [
            threading.Thread(target=self._worker, daemon=True) for _ in range(self.bs)
        ]
        for t in threads:
            t.start()
        self._threads = threads
        time.sleep(seconds)

    def finish(self, join_timeout: float = 300.0) -> None:
        self.stop.set()
        for t in getattr(self, "_threads", []):
            t.join(timeout=join_timeout)


def measure_point(args) -> dict:
    set_name, prompts = load_prompts(args.prompt_file, args.content_class)
    if not prompts:
        return {
            "kind": KIND,
            "schema": SCHEMA_VERSION,
            "arm": args.arm,
            "bs": args.bs,
            "content_class": args.content_class,
            "counted": False,
            "error": f"no prompts of class {args.content_class} in {args.prompt_file}",
        }

    # Warmup on foreign content, then flush. See WARMUP_PROMPT.
    warm = Pool(
        args.port,
        [{"id": "_warmup", "prompt": WARMUP_PROMPT}],
        args.bs,
        WARMUP_MAX_NEW_TOKENS,
    )
    warm.run_for(args.warmup_seconds)
    warm.finish(join_timeout=120)
    flush = _flush_cache(args.port)

    pool = Pool(args.port, prompts, args.bs, args.default_max_new_tokens)
    pool.run_for(args.ramp_seconds)
    m0 = _metrics_snapshot(args.port)
    t0_mono, t0_wall = time.monotonic(), time.time()
    time.sleep(args.window_seconds)
    t1_mono, t1_wall = time.monotonic(), time.time()
    m1 = _metrics_snapshot(args.port)
    pool.finish()
    span = t1_mono - t0_mono

    # Only requests that both STARTED and ENDED inside the window are the
    # window's own work. A request straddling t0 carries ramp tokens, one
    # straddling t1 was cut off by the stop event.
    inside = [
        r for r in pool.records if t0_mono <= r["t_start"] and r["t_end"] <= t1_mono
    ]

    validated = []
    reasons: dict = {}
    for r in inside:
        ok, reason = validate_output(r.get("text") or "", r.get("validator") or {})
        r["valid"] = ok
        r["invalid_reason"] = reason
        validated.append(r)
        if not ok:
            key = reason.split(":", 1)[0] or "unknown"
            reasons[key] = reasons.get(key, 0) + 1

    valid = [r for r in validated if r["valid"]]
    valid_ratio = (len(valid) / len(validated)) if validated else 0.0

    tokens = sum(r.get("completion_tokens") or 0 for r in validated)
    verify = sum(r.get("spec_verify_ct") or 0 for r in validated)
    tokens_valid = sum(r.get("completion_tokens") or 0 for r in valid)
    accepts = [
        r["spec_accept_length"]
        for r in validated
        if isinstance(r.get("spec_accept_length"), (int, float))
    ]

    per_prompt: dict = {}
    for r in validated:
        slot = per_prompt.setdefault(
            r["prompt_id"], {"n": 0, "valid": 0, "tokens": 0, "e2e_ms": []}
        )
        slot["n"] += 1
        slot["valid"] += 1 if r["valid"] else 0
        slot["tokens"] += r.get("completion_tokens") or 0
        slot["e2e_ms"].append(r["e2e_ms"])
    for slot in per_prompt.values():
        slot["e2e_ms_median"] = (
            statistics.median(slot["e2e_ms"]) if slot["e2e_ms"] else None
        )
        del slot["e2e_ms"]

    point = {
        "kind": KIND,
        "schema": SCHEMA_VERSION,
        "seq": args.seq,
        "arm": args.arm,
        "algo": args.algo,
        "bs": args.bs,
        "content_class": args.content_class,
        "prompt_set": set_name,
        "prompt_file": args.prompt_file,
        "prompts_in_class": [p["id"] for p in prompts],
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "warmup_seconds": args.warmup_seconds,
        "ramp_seconds": args.ramp_seconds,
        "window_seconds": args.window_seconds,
        "window_span_s": span,
        "window_from": t0_wall,
        "window_to": t1_wall,
        "flush_cache": flush,
        "metrics_before": m0,
        "metrics_after": m1,
        "requests_total": len(pool.records),
        "requests_in_window": len(validated),
        "requests_valid": len(valid),
        "valid_ratio": valid_ratio,
        "invalid_reasons": reasons,
        "min_valid_ratio": args.min_valid_ratio,
        # The client level. Tokens and verify counts come from the final
        # meta_info of requests wholly inside the window, so this accept is a
        # windowed accept AT THE POINT'S OWN BATCH SIZE.
        "client_tokens": tokens,
        "client_tokens_valid": tokens_valid,
        "client_verify_ct": verify,
        "client_tok_s": (tokens / span) if span > 0 and tokens else None,
        "client_accept_len_pooled": (tokens / verify) if verify else None,
        "client_accept_len_median": statistics.median(accepts) if accepts else None,
        "per_prompt": per_prompt,
        "request_errors": pool.errors[:5],
        "request_error_count": len(pool.errors),
    }
    point.update(_harvest_ticks(args.server_log, t0_wall, t1_wall, args.bs))

    if point.get("tick_gen_tok_s_median") and point.get("client_accept_len_pooled"):
        # The same quantity as tick_ms_per_verify, but with the accept the
        # CLIENT measured. Two independent accepts against one rate: if these
        # two ms/Verify disagree by more than the noise floor, the point is
        # telling on itself.
        point["client_ms_per_verify"] = (
            1000.0 / point["tick_gen_tok_s_median"] * point["client_accept_len_pooled"]
        )

    # THE GATE. A point without a tick rate has no primary measure, and a point
    # whose outputs did not hold their shape is a fast garbage run. Either way
    # it is written out -- the record of a failure is worth more than a gap --
    # but it is marked so the analysis drops it.
    counted = True
    not_counted: list = []
    if point.get("tick_gen_tok_s_median") is None:
        counted = False
        not_counted.append("no tick rate in window")
    if not validated:
        counted = False
        not_counted.append("no request completed inside the window")
    elif valid_ratio < args.min_valid_ratio:
        counted = False
        not_counted.append(
            f"valid share {valid_ratio:.2f} < {args.min_valid_ratio:.2f}"
        )
    point["counted"] = counted
    point["not_counted_because"] = not_counted

    if args.samples_dir:
        os.makedirs(args.samples_dir, exist_ok=True)
        name = f"{args.arm}.bs{args.bs}.{args.content_class}.txt"
        with open(os.path.join(args.samples_dir, name), "w") as f:
            for r in validated[:SAMPLES_PER_POINT]:
                f.write(f"=== {r['prompt_id']} valid={r['valid']} ")
                f.write(f"reason={r['invalid_reason']!r} ")
                f.write(f"tokens={r.get('completion_tokens')} ===\n")
                f.write((r.get("text") or "") + "\n\n")
        point["sample_file"] = name

    time.sleep(args.drain_seconds)
    return point


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--algo", default="", help="NEXTN / DFLASH, for the record only")
    p.add_argument("--bs", type=int, required=True)
    p.add_argument("--content-class", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--seq", type=int, default=0)
    p.add_argument("--warmup-seconds", type=float, default=8.0)
    p.add_argument("--ramp-seconds", type=float, default=6.0)
    p.add_argument("--window-seconds", type=float, default=20.0)
    p.add_argument("--drain-seconds", type=float, default=5.0)
    p.add_argument("--default-max-new-tokens", type=int, default=320)
    p.add_argument("--min-valid-ratio", type=float, default=0.75)
    p.add_argument("--server-log", default="")
    p.add_argument("--samples-dir", default="")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    point = measure_point(args)
    with open(os.path.join(args.out_dir, "structured_points.jsonl"), "a") as f:
        f.write(json.dumps(point) + "\n")

    print(
        f"point {args.arm}/bs={args.bs}/{args.content_class}: "
        f"tick {point.get('tick_gen_tok_s_median')} tok/s, "
        f"accept tick {point.get('tick_accept_len_median')} "
        f"client {point.get('client_accept_len_pooled')}, "
        f"ms/Verify {point.get('tick_ms_per_verify')}, "
        f"ticks {point.get('tick_ticks_counted')}/{point.get('tick_ticks_bs')} "
        f"(other bs: {point.get('tick_ticks_other_bs')}), "
        f"valid {point.get('requests_valid')}/{point.get('requests_in_window')}, "
        f"counted={point.get('counted')}"
    )
    if not point.get("counted"):
        for why in point.get("not_counted_because") or []:
            print(f"  NOT COUNTED: {why}", file=sys.stderr)
        for err in point.get("request_errors") or []:
            print(f"  request error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
