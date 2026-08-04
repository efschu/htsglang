#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probes for the 2026-08-04 DSV4F window (#478 quant swap / #470 DSpark / #462 F2).

Run with the venv interpreter and the worktree on PYTHONPATH::

    export PYTHONPATH=/spinning/wt-dsv4f-window/python
    /spinning/htsglang-gpu/.venv/bin/python probes.py <mode> --port <p> --arm <a>

DESIGN RULES ENCODED HERE (they are house law, not preferences)
---------------------------------------------------------------
1. **Native /generate only for measurement.** ``/v1/chat/completions`` carries
   no ``meta_info`` on this fork, so every accept-rate and timing number comes
   from ``/generate``. Chat is used for exactly one thing: proving the chat
   template was applied (mode ``chatprobe``).
2. **ms/round is the measuring stick, not tok/s.** Every timing record
   reports ``ms_per_round`` (ms per verify round for a speculative arm, ms per
   token otherwise) and ``ms_per_prefill``. tok/s is carried along as a
   secondary, clearly-labelled figure.
3. **Time-bounded decode, never token-bounded.** A decode point runs for a
   wall-clock window (default 15 s) via a streamed request that is cut off,
   because a token budget makes the measurement duration depend on the very
   quantity being measured.
4. **A-vs-A floor before any delta.** ``avsa`` measures the same-boot noise
   floor from two identical greedy runs with the first discarded.
   ``report_delta`` REFUSES to print a delta whose magnitude does not exceed
   its own point's floor -- it prints "inside the floor", which is the result.
5. **Accept length is ``meta_info.spec_accept_length``.** Never
   ``spec_ema_accept_len``: that is a server-lifetime EMA gauge and is not
   this request's accept length. It is recorded next to the real number,
   explicitly labelled provenance-only.
6. **Greedy everywhere** (``temperature 0``). The DSpark solo placement
   refuses non-greedy rounds by name; that is the v1 limit, not a bug to work
   around, so every arm measures on the same greedy footing.
7. **INSTRUMENT PRECONDITION.** No probe's verdict counts until the
   instrument has passed a can-discriminate check on known-different inputs.
   ``--selftest`` (hermetic: no GPU, no server, no network) proves every
   scorer separates a known-good from a known-bad input, and proves the
   floor gate and the divergence comparator can both actually fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable

DEFAULT_RUN = "/spinning/gpu-battery-results/2026-08-04_dsv4f_window"
DEFAULT_TIMEOUT = 300.0

# ---------------------------------------------------------------------------
# HTTP -- bounded timeouts on every call, no exceptions.
# ---------------------------------------------------------------------------


def _post(base: str, path: str, body: dict, timeout: float = DEFAULT_TIMEOUT) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed localhost base
            raw = resp.read().decode("utf-8", "replace")
            return {"status": resp.status, "body": raw, "wall_ms": (time.time() - t0) * 1000}
    except urllib.error.HTTPError as exc:
        return {
            "status": exc.code,
            "body": exc.read().decode("utf-8", "replace"),
            "wall_ms": (time.time() - t0) * 1000,
            "error": f"HTTP {exc.code}",
        }
    except Exception as exc:  # noqa: BLE001 - a probe reports, it never crashes the arm
        return {"status": 0, "body": "", "wall_ms": (time.time() - t0) * 1000, "error": str(exc)}


def _get(base: str, path: str, timeout: float = 15.0) -> dict:
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as resp:  # noqa: S310
            return {"status": resp.status, "body": resp.read().decode("utf-8", "replace")}
    except urllib.error.HTTPError as exc:
        # Report the REAL status. Collapsing this to 0 made a 503 "still
        # initialising" indistinguishable from a dead socket, which is exactly
        # the ambiguity that hid a premature-readiness bug for two boots.
        return {
            "status": exc.code,
            "body": exc.read().decode("utf-8", "replace"),
            "error": f"HTTP {exc.code}",
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": 0, "body": "", "error": str(exc)}


def generate(base: str, text: str, max_new_tokens: int, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """One greedy native /generate call. Greedy is mandatory (see rule 6)."""
    body = {
        "text": text,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
    }
    res = _post(base, "/generate", body, timeout)
    out: dict[str, Any] = {"http_code": res["status"], "wall_ms": round(res["wall_ms"], 1)}
    if res["status"] != 200:
        out["error"] = res.get("error") or res["body"][:400]
        return out
    try:
        doc = json.loads(res["body"])
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"unparsable body: {exc}"
        return out
    out["text"] = doc.get("text", "")
    out["meta_info"] = doc.get("meta_info") or {}
    return out


# ---------------------------------------------------------------------------
# Streaming, time-bounded (rule 3)
# ---------------------------------------------------------------------------

_SSE_PREFIX = "data:"


def parse_sse_line(line: str) -> dict | None:
    """One SSE payload, or None for keep-alive lines / terminators.

    Split out so ``--selftest`` can drive it from a synthetic stream: an
    instrument whose parser has never been exercised on a known input is not
    an instrument.
    """
    line = line.strip()
    if not line or not line.startswith(_SSE_PREFIX):
        return None
    payload = line[len(_SSE_PREFIX) :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except Exception:  # noqa: BLE001 - a malformed chunk is skipped, not fatal
        return None


def classify_rounds(
    cut_off: bool,
    tokens: int,
    spec_verify_ct: Any,
    completion_tokens: int,
) -> dict:
    """Decide what a "round" was, from the FINAL meta_info only.

    Pure, so the decision is testable without a server. See ``stream_bounded``
    for why only the final chunk can carry ``spec_verify_ct``.
    """
    if cut_off:
        # The request never finished, so no verify count exists anywhere. Say
        # so rather than reporting ms/token under a verify label.
        return {
            "rounds": tokens,
            "round_kind": "token (stream cut off before finish: no verify count exists)",
        }
    if spec_verify_ct is not None and int(spec_verify_ct) > 0:
        rounds = int(spec_verify_ct)
        out = {"rounds": rounds, "round_kind": "verify"}
        # completion_tokens / verify_ct is the accept length INCLUDING the
        # bonus token -- the same definition as tokenizer_manager.py:2421.
        out["accept_length"] = round(completion_tokens / rounds, 4)
        return out
    # Non-speculative arm: one round IS one token. Stated, not silently
    # assumed, so a speculative arm that lost its counter stays visible.
    return {"rounds": tokens, "round_kind": "token (no spec_verify_ct in meta_info)"}


def stream_bounded(
    base: str, text: str, window_seconds: float, max_new_tokens: int = 128,
    connect_timeout: float = 120.0,
) -> dict:
    """Stream a greedy generation and CUT IT OFF after ``window_seconds``.

    Returns the phase split the house cares about:
      ``ms_prefill``  wall ms from request send to the FIRST streamed chunk
                      (time to first token = the prefill phase),
      ``decode_s``    wall seconds from the first chunk to the last one,
      ``tokens``      completion tokens produced inside that decode span,
      ``rounds``      verify rounds when the arm is speculative and the request
                      RAN TO COMPLETION, else tokens.

    On ``spec_verify_ct`` and why a time-bounded cut-off cannot price a verify
    round (verified at desk, tokenizer_manager.py:2145-2153): the speculative
    counters are attached to ``meta_info`` only inside ``if state.finished:``,
    via ``_calculate_spec_decoding_metrics``. They therefore appear on the
    FINAL chunk and on no other. Two consequences:

      * A per-chunk delta (``last - first``) can never fire -- the first chunk
        never carries the counter -- so it would silently and permanently
        report token-kind rounds even on a healthy speculative arm.
      * A stream that is cut off by the time bound never finishes, so it
        carries no counter at all.

    So ms/verify needs a request that COMPLETES. Size ``max_new_tokens`` to
    land inside the time budget at the arm's expected decode rate and leave
    ``window_seconds`` as the safety cut-off rather than the intended stop;
    ``rounds`` is then the final ``spec_verify_ct`` for the whole generation.
    That total includes the verify round overlapping prefill, which is why
    ``ms_prefill`` is reported separately and ``ms_round`` is derived over the
    decode span only.
    """
    body = {
        "text": text,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
        "stream": True,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base + "/generate", data=data, headers={"Content-Type": "application/json"}
    )
    rec: dict[str, Any] = {"window_seconds": window_seconds, "cut_off": False}
    t_send = time.time()
    first_t: float | None = None
    last_t: float | None = None
    first_meta: dict | None = None
    last_meta: dict | None = None
    chunks = 0
    try:
        with urllib.request.urlopen(req, timeout=connect_timeout) as resp:  # noqa: S310
            for raw in resp:
                chunk = parse_sse_line(raw.decode("utf-8", "replace"))
                if chunk is None:
                    continue
                now = time.time()
                chunks += 1
                meta = chunk.get("meta_info") or {}
                if first_t is None:
                    first_t, first_meta = now, dict(meta)
                last_t, last_meta = now, dict(meta)
                if now - first_t >= window_seconds:
                    rec["cut_off"] = True
                    break
    except Exception as exc:  # noqa: BLE001
        rec["error"] = str(exc)

    if first_t is None or last_t is None or last_meta is None or first_meta is None:
        rec["error"] = rec.get("error", "no streamed chunk arrived")
        return rec

    rec["ms_prefill"] = round((first_t - t_send) * 1000, 2)
    rec["decode_s"] = round(last_t - first_t, 4)
    rec["chunks"] = chunks
    rec["prompt_tokens"] = last_meta.get("prompt_tokens")

    tok_first = first_meta.get("completion_tokens") or 0
    tok_last = last_meta.get("completion_tokens") or 0
    rec["tokens"] = max(0, int(tok_last) - int(tok_first))

    rec.update(
        classify_rounds(
            cut_off=bool(rec["cut_off"]),
            tokens=int(rec["tokens"]),
            spec_verify_ct=last_meta.get("spec_verify_ct"),
            completion_tokens=int(tok_last),
        )
    )
    rec["last_meta_info"] = last_meta
    return derive_rates(rec)


def derive_rates(rec: dict) -> dict:
    """ms/round and ms/token from a phase-split record. ms/round is the headline."""
    decode_s = rec.get("decode_s") or 0.0
    tokens = rec.get("tokens") or 0
    rounds = rec.get("rounds") or 0
    rec["ms_per_token"] = round(decode_s * 1000 / tokens, 3) if tokens else None
    rec["ms_per_round"] = round(decode_s * 1000 / rounds, 3) if rounds else None
    # tok/s is SECONDARY and labelled as such -- reports quote ms/round.
    rec["tok_per_s_secondary"] = round(tokens / decode_s, 2) if decode_s else None
    return rec


# ---------------------------------------------------------------------------
# The A-vs-A floor and the delta gate (rule 4)
# ---------------------------------------------------------------------------


def spread_pct(a: float, b: float) -> float:
    """Percent spread of two readings of the same thing."""
    mean = (a + b) / 2.0
    if mean == 0:
        return 0.0
    return abs(a - b) / mean * 100.0


def report_delta(name: str, baseline: float, arm: float, floor_pct: float) -> dict:
    """The gate. A delta inside its own point's floor is NOT reported as a delta.

    Returns a record whose ``verdict`` is either ``"below floor"`` (and
    ``delta_pct`` is present but explicitly not quotable) or ``"above floor"``.
    """
    delta = spread_pct(baseline, arm)
    signed = ((arm - baseline) / baseline * 100.0) if baseline else 0.0
    above = delta > floor_pct
    return {
        "name": name,
        "baseline": baseline,
        "arm": arm,
        "delta_pct_abs": round(delta, 3),
        "delta_pct_signed": round(signed, 3),
        "floor_pct": round(floor_pct, 3),
        "verdict": "above floor" if above else "below floor",
        "quotable": above,
        "note": (
            ""
            if above
            else "magnitude does not exceed this point's own A-vs-A floor; "
            "the result is 'inside the floor', not a delta"
        ),
    }


# ---------------------------------------------------------------------------
# Scorers for the determined-answer quality gate (#478)
# ---------------------------------------------------------------------------

_THINK_END = "</think>"


def strip_reasoning(text: str) -> str:
    """Drop any reasoning prefix.

    ``--reasoning-parser`` only acts on the OpenAI chat path; native
    ``/generate`` returns raw text, so a thinking-mode answer can carry the
    tags inline. Everything up to and including the LAST ``</think>`` is
    reasoning.
    """
    if _THINK_END in text:
        text = text.rsplit(_THINK_END, 1)[1]
    return text.strip()


def _norm(text: str) -> str:
    return re.sub(r"[\s,]+", " ", strip_reasoning(text or "")).strip().lower().rstrip(".")


def exact_scorer(expected: str) -> Callable[[str], tuple[float, str]]:
    want = _norm(expected)

    def score(answer: str) -> tuple[float, str]:
        got = _norm(answer)
        if got == want:
            return 1.0, ""
        # A determined answer may arrive with a short lead-in; accept it only
        # when the expected string is the whole of the FINAL line.
        tail = got.split("\n")[-1].strip() if "\n" in got else got
        if tail == want:
            return 1.0, "matched on final line"
        return 0.0, f"want {want!r}, got {got[:80]!r}"

    return score


def number_scorer(expected: float, tol: float = 0.0) -> Callable[[str], tuple[float, str]]:
    def score(answer: str) -> tuple[float, str]:
        nums = re.findall(r"-?\d+(?:\.\d+)?", strip_reasoning(answer or ""))
        if not nums:
            return 0.0, f"no number in {(answer or '')[:80]!r}"
        got = float(nums[-1])
        if abs(got - expected) <= tol:
            return 1.0, ""
        return 0.0, f"want {expected}, got {got}"

    return score


def json_scorer(expected: dict) -> Callable[[str], tuple[float, str]]:
    def score(answer: str) -> tuple[float, str]:
        text = strip_reasoning(answer or "")
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return 0.0, f"no JSON object in {text[:80]!r}"
        try:
            got = json.loads(match.group(0))
        except Exception as exc:  # noqa: BLE001
            return 0.0, f"unparsable JSON: {exc}"
        return (1.0, "") if got == expected else (0.0, f"want {expected}, got {got}")

    return score


# Each spec carries its own known-good and known-bad sample, so --selftest can
# prove the scorer discriminates instead of merely running.
DETERMINED: list[dict[str, Any]] = [
    {
        "id": "arith_mul",
        "prompt": "Compute 17 * 23. Reply with only the number, nothing else.\n",
        "max_new_tokens": 48,
        "scorer": number_scorer(391),
        "good": "391",
        "bad": "402",
    },
    {
        "id": "capital_au",
        "prompt": "What is the capital city of Australia? Reply with only the city name, nothing else.\n",
        "max_new_tokens": 32,
        "scorer": exact_scorer("Canberra"),
        "good": "Canberra",
        "bad": "Sydney",
    },
    {
        "id": "letter_count",
        "prompt": "How many times does the letter r appear in the word strawberry? Reply with only the digit.\n",
        "max_new_tokens": 32,
        "scorer": number_scorer(3),
        "good": "3",
        "bad": "2",
    },
    {
        "id": "reverse_word",
        "prompt": "Write the word stack spelled backwards. Reply with only the reversed word, nothing else.\n",
        "max_new_tokens": 32,
        "scorer": exact_scorer("kcats"),
        "good": "kcats",
        "bad": "stcak",
    },
    {
        "id": "unit_convert",
        "prompt": "How many centimetres are in 2.5 metres? Reply with only the number, nothing else.\n",
        "max_new_tokens": 32,
        "scorer": number_scorer(250),
        "good": "250",
        "bad": "25",
    },
    {
        "id": "geom_seq",
        "prompt": "Continue the sequence 2, 4, 8, 16 with the next three numbers, comma separated, nothing else.\n",
        "max_new_tokens": 40,
        "scorer": exact_scorer("32 64 128"),
        "good": "32, 64, 128",
        "bad": "32, 48, 64",
    },
    {
        "id": "json_echo",
        "prompt": 'Reply with exactly this JSON object and nothing else: {"a": 1, "b": "two"}\n',
        "max_new_tokens": 48,
        "scorer": json_scorer({"a": 1, "b": "two"}),
        "good": '{"a": 1, "b": "two"}',
        "bad": '{"a": 2, "b": "two"}',
    },
    {
        "id": "date_arith",
        "prompt": "If today is Monday, what day of the week is it in 10 days? Reply with only the weekday name.\n",
        "max_new_tokens": 32,
        "scorer": exact_scorer("Thursday"),
        "good": "Thursday",
        "bad": "Wednesday",
    },
]


# ---------------------------------------------------------------------------
# Accept-length reader (rule 5)
# ---------------------------------------------------------------------------

_EMA_KEY = "spec_ema_accept_len"


def read_accept(meta_info: dict) -> dict:
    """Accept numbers from ONE request's ``meta_info``.

    ``spec_accept_length`` is the number. ``spec_ema_accept_len`` is a
    server-lifetime EMA gauge and MUST NOT be substituted for it -- this
    function refuses to, and ``--selftest`` proves the refusal by feeding a
    meta_info that carries ONLY the EMA key.
    """
    if _EMA_KEY in meta_info and "spec_accept_length" not in meta_info:
        return {
            "spec_accept_length": None,
            "error": (
                f"meta_info carries only {_EMA_KEY} (a server-lifetime EMA gauge), "
                "not spec_accept_length. That is not this request's accept "
                "length and is not substituted for it."
            ),
            "prom_spec_ema_accept_len_provenance_only": meta_info.get(_EMA_KEY),
        }
    return {
        "spec_accept_length": meta_info.get("spec_accept_length"),
        "spec_verify_ct": meta_info.get("spec_verify_ct"),
        # Fallback with named provenance: if the server did not attach
        # spec_accept_length, derive it the way the server itself would
        # (completion_tokens / spec_verify_ct, tokenizer_manager.py:2421)
        # rather than reporting None and losing the arm's headline number.
        "spec_accept_length_derived": (
            round(meta_info["completion_tokens"] / meta_info["spec_verify_ct"], 4)
            if meta_info.get("spec_verify_ct") and meta_info.get("completion_tokens")
            else None
        ),
        "completion_tokens": meta_info.get("completion_tokens"),
        "e2e_latency": meta_info.get("e2e_latency"),
        "prom_spec_ema_accept_len_provenance_only": meta_info.get(_EMA_KEY),
    }


# ---------------------------------------------------------------------------
# Prompt construction for the prefill curve
# ---------------------------------------------------------------------------

_FILLER = (
    "The maintenance log records the condition of each subsystem after every "
    "scheduled inspection, together with the tolerances that were measured and "
    "the corrective actions that the technician applied. Entries are ordered by "
    "date and each one names the responsible engineer, the part numbers that "
    "were replaced, and the observed deviation from the reference value. "
)

# Matches the prior window's points so the shape is comparable.
PREFILL_TARGETS = (240, 480, 940, 1850)


def build_prompt(target_tokens: int) -> str:
    """A prompt of roughly ``target_tokens``.

    Approximate on purpose: the ACTUAL length is read back from
    ``meta_info.prompt_tokens`` and recorded next to the target, so a curve is
    always plotted against measured lengths rather than intended ones.
    """
    approx_tokens_per_char = 1 / 4.0
    need_chars = int(target_tokens / approx_tokens_per_char)
    body = (_FILLER * (need_chars // len(_FILLER) + 1))[:need_chars]
    return (
        "Read the following maintenance log excerpt and then answer.\n\n"
        + body
        + "\n\nIn one short sentence, what kind of document is this?\n"
    )


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def mode_prefill(base: str, args) -> dict:
    rows = []
    for target in PREFILL_TARGETS:
        prompt = build_prompt(target)
        # s=1: a single stream, so the point is the per-request prefill cost.
        rec = stream_bounded(base, prompt, window_seconds=args.window_seconds, max_new_tokens=64)
        rec["target_prompt_tokens"] = target
        actual = rec.get("prompt_tokens")
        if actual:
            ms = rec.get("ms_prefill")
            rec["ms_per_prefill_token"] = round(ms / actual, 4) if ms else None
            rec["prefill_tok_per_s_secondary"] = (
                round(actual / (ms / 1000.0), 1) if ms else None
            )
        rows.append(rec)
        print(
            f"  prefill target={target} actual={actual} "
            f"ms_prefill={rec.get('ms_prefill')} "
            f"ms/prefill-token={rec.get('ms_per_prefill_token')}",
            flush=True,
        )
    return {"mode": "prefill", "streams": 1, "points": rows}


def mode_decode(base: str, args) -> dict:
    """bs=1 decode, TIME-bounded. Headline is ms/round, then ms/token."""
    rows = []
    for target in (args.context_tokens,):
        prompt = build_prompt(target)
        rec = stream_bounded(base, prompt, window_seconds=args.window_seconds,
                             max_new_tokens=args.max_new_tokens)
        rec["context_tokens_target"] = target
        rows.append(rec)
        print(
            f"  decode ctx={target} decode_s={rec.get('decode_s')} "
            f"tokens={rec.get('tokens')} rounds={rec.get('rounds')} "
            f"({rec.get('round_kind')}) ms/round={rec.get('ms_per_round')} "
            f"ms/token={rec.get('ms_per_token')} "
            f"ms/prefill={rec.get('ms_prefill')}",
            flush=True,
        )
    return {"mode": "decode", "bs": 1, "points": rows}


def mode_avsa(base: str, args) -> dict:
    """The same-boot A-vs-A floor. MUST run before any delta in every arm.

    Warmup discarded, then two identical greedy runs back to back. The floor
    is the percent spread of ms/round between them; a delta smaller than this
    is not a delta.
    """
    prompt = build_prompt(args.context_tokens)
    runs = []
    for i in range(3):  # run 0 is the discarded warmup
        rec = stream_bounded(base, prompt, window_seconds=args.window_seconds,
                             max_new_tokens=args.max_new_tokens)
        rec["run_index"] = i
        rec["role"] = "warmup (DISCARDED)" if i == 0 else f"A{i}"
        runs.append(rec)
        print(
            f"  avsa run{i} ({rec['role']}) ms/round={rec.get('ms_per_round')} "
            f"ms/token={rec.get('ms_per_token')}",
            flush=True,
        )
    a1, a2 = runs[1], runs[2]
    out: dict[str, Any] = {"mode": "avsa", "runs": runs, "warmup_discarded": True}
    for key in ("ms_per_round", "ms_per_token", "ms_prefill"):
        v1, v2 = a1.get(key), a2.get(key)
        out[f"floor_pct_{key}"] = round(spread_pct(v1, v2), 3) if v1 and v2 else None
    print(
        f"  A-vs-A FLOOR: ms/round {out.get('floor_pct_ms_per_round')}% "
        f"ms/token {out.get('floor_pct_ms_per_token')}% "
        f"ms/prefill {out.get('floor_pct_ms_prefill')}%",
        flush=True,
    )
    return out


def mode_determined(base: str, args) -> dict:
    """The #478 quality gate. Fully-determined answers, greedy, scored.

    Sent through ``/v1/chat/completions``, NOT native ``/generate``. This is the
    one probe where that is right: a determined-answer prompt is an
    INSTRUCTION, and an instruction handed to a base-completion endpoint gets
    CONTINUED rather than obeyed. Measured on the 1a arm, which ran this over
    /generate: 1 of 8 scored, and the failures were the model continuing web
    boilerplate ("</p> <p>Your response must only contain the JSON object...")
    or echoing the prompt back. That is a probe artefact and says nothing about
    the checkpoint -- reporting it as a quality number would have been a
    fabricated regression.

    The measurement probes stay on /generate because chat carries no
    meta_info; this one needs no meta_info, only the text.
    """
    rows = []
    for spec in DETERMINED:
        body = {
            "model": args.model or "dsv4f",
            "messages": [{"role": "user", "content": spec["prompt"]}],
            "temperature": 0.0,
            "max_tokens": spec["max_new_tokens"],
        }
        res = _post(base, "/v1/chat/completions", body, args.timeout)
        # _post hands back the RAW body plus the status; it does not parse.
        answer = ""
        parse_error = None
        try:
            doc = json.loads(res.get("body") or "{}")
            answer = doc["choices"][0]["message"]["content"] or ""
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
        try:
            score, note = spec["scorer"](answer)
        except Exception as exc:  # noqa: BLE001 - a scorer never takes the arm down
            score, note = 0.0, f"scorer raised: {exc}"
        rows.append(
            {
                "id": spec["id"],
                "score": score,
                "note": note,
                "answer": answer,
                "http_code": res.get("status"),
                "parse_error": parse_error,
            }
        )
        # An unparsable or non-200 answer is an INSTRUMENT failure, not a
        # quality result -- say which, so a broken endpoint can never be
        # reported as a model regression.
        if parse_error or res.get("status") != 200:
            print(f"  determined {spec['id']}: INSTRUMENT FAILURE "
                  f"status={res.get('status')} {parse_error or ''}", flush=True)
        else:
            print(f"  determined {spec['id']}: {score} {note}", flush=True)
    scored = [r["score"] for r in rows]
    return {
        "mode": "determined",
        "n": len(rows),
        "passed": int(sum(scored)),
        "score_pct": round(100.0 * sum(scored) / len(scored), 2) if scored else None,
        "rows": rows,
    }


def mode_accept(base: str, args) -> dict:
    """Accept length for the DSpark arm. Native /generate only.

    Prompt set: /spinning/gpu-battery-results/2026-08-03_447_dspark/prompts.json
    when present (its own provenance block warns it must NOT be compared 1:1
    against the llama.cpp 0.49-0.77 band -- those are other domains).
    """
    # Schema of that file (verified): {"prompts": [{"name", "domain", "text",
    # "max_new_tokens"}, ...]}, three domains, narrative / code /
    # math-reasoning. Its `usage` block prescribes exactly this call shape.
    prompts = []
    if args.prompts and os.path.exists(args.prompts):
        with open(args.prompts) as fh:
            doc = json.load(fh)
        for item in doc.get("prompts", []):
            text = item.get("text")
            if text:
                prompts.append(
                    {
                        "id": item.get("name", f"p{len(prompts)}"),
                        "domain": item.get("domain"),
                        "text": text,
                        "max_new_tokens": item.get("max_new_tokens", 400),
                    }
                )
    if not prompts:
        prompts = [
            {
                "id": "counting",
                "domain": "fallback-single",
                "text": "Count from 1 to 200, one number per line, nothing else.\n",
                "max_new_tokens": 600,
            }
        ]
        print(
            f"  NOTE: no usable prompt set at {args.prompts!r}; falling back to the "
            "single counting prompt. Record this -- the accept number is then "
            "single-domain and not comparable to a mixed-domain figure.",
            flush=True,
        )

    rows = []
    for item in prompts:
        res = generate(base, item["text"], item["max_new_tokens"], timeout=args.timeout)
        meta = res.get("meta_info") or {}
        rec = {
            "id": item["id"],
            "domain": item.get("domain"),
            "http_code": res.get("http_code"),
            **read_accept(meta),
        }
        decode_s = meta.get("e2e_latency")
        verify_ct = rec.get("spec_verify_ct")
        if decode_s and verify_ct:
            rec["ms_per_verify_round_incl_prefill"] = round(decode_s * 1000 / verify_ct, 3)
            rec["caveat"] = (
                "e2e_latency includes prefill; use the `decode` mode's "
                "ms/round for a clean decode figure"
            )
        rows.append(rec)
        print(
            f"  accept {item['id']}: spec_accept_length={rec.get('spec_accept_length')} "
            f"spec_verify_ct={rec.get('spec_verify_ct')} "
            f"completion_tokens={rec.get('completion_tokens')}",
            flush=True,
        )
    lengths = [r["spec_accept_length"] for r in rows if r.get("spec_accept_length")]
    return {
        "mode": "accept",
        "rows": rows,
        "mean_spec_accept_length": round(statistics.fmean(lengths), 4) if lengths else None,
        "reference_band_note": (
            "llama.cpp PR #25784 reports 0.49-0.77 on THEIR domains with no "
            "prompt file published. Order of magnitude only -- never a 1:1 "
            "comparison against this set."
        ),
    }


def mode_chatprobe(base: str, args) -> dict:
    """Prove the chat TEMPLATE was applied, not merely that the server answered.

    The discriminating construction: render the same conversation locally with
    the template file the boot passed, send that rendered string through native
    ``/generate``, and send the raw messages through ``/v1/chat/completions``.
    Greedy on both sides, short output. If the server applied the template,
    the two prompts are the same string and the greedy outputs agree.

    The probe's own can-discriminate arm runs in the same call: a NEGATIVE
    control sends the naive role-less concatenation through ``/generate``. Its
    output must DIFFER from the chat output. If negative and positive both
    agree, the probe cannot discriminate and its verdict is void -- which is
    reported as ``"instrument void"``, not as a pass.
    """
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Name the largest planet in the Solar System."},
        {"role": "assistant", "content": "Jupiter."},
        {"role": "user", "content": "And the smallest one?"},
    ]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from extract_chat_template import _render  # noqa: PLC0415 - local helper

    with open(args.chat_template, encoding="utf-8") as fh:
        template_src = fh.read()
    rendered = _render(template_src, messages, add_generation_prompt=True)
    naive = "\n".join(m["content"] for m in messages) + "\n"

    max_tok = args.chat_probe_tokens
    gen_templated = generate(base, rendered, max_tok, timeout=args.timeout)
    gen_naive = generate(base, naive, max_tok, timeout=args.timeout)

    chat_body = {
        "model": args.model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tok,
    }
    chat_res = _post(base, "/v1/chat/completions", chat_body, args.timeout)
    chat_text = ""
    if chat_res["status"] == 200:
        try:
            doc = json.loads(chat_res["body"])
            chat_text = (doc["choices"][0]["message"].get("content") or "").strip()
        except Exception as exc:  # noqa: BLE001
            chat_text = f"<unparsable: {exc}>"

    t_text = strip_reasoning(gen_templated.get("text", ""))
    n_text = strip_reasoning(gen_naive.get("text", ""))
    c_text = strip_reasoning(chat_text)

    positive = bool(c_text) and c_text == t_text
    discriminates = n_text != t_text

    if not discriminates:
        verdict = "instrument void"
        note = (
            "the naive role-less concatenation produced the SAME output as the "
            "templated prompt, so this probe cannot tell an applied template "
            "from an unapplied one. Its result proves nothing; lengthen "
            "--chat-probe-tokens or change the conversation and re-run."
        )
    elif positive:
        verdict = "template applied"
        note = "chat output == locally-rendered-template output, and the negative control differs"
    else:
        verdict = "TEMPLATE NOT APPLIED (or applied differently)"
        note = "chat output does not match the template this boot was given"

    print(f"  chatprobe: {verdict} -- {note}", flush=True)
    return {
        "mode": "chatprobe",
        "verdict": verdict,
        "note": note,
        "discriminates": discriminates,
        "chat_http_code": chat_res["status"],
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "rendered_prompt_tail": rendered[-160:],
        "templated_generate_text": t_text,
        "naive_generate_text": n_text,
        "chat_text": c_text,
    }


# ---------------------------------------------------------------------------
# Idempotence probe (TICKET_470 §3.2 / ANALYSE_447 §2.4)
# ---------------------------------------------------------------------------

IDEM_PROMPTS = [
    "List the first 12 prime numbers, comma separated, nothing else.\n",
    "Write exactly four short sentences about tidal power. No preamble.\n",
    "Explain, in exactly three numbered steps, how to boil an egg.\n",
]
IDEM_MAX_TOKENS = 220


def _idem_runs(base: str, timeout: float, repeats: int = 3) -> list[dict]:
    rows = []
    for prompt in IDEM_PROMPTS:
        texts = []
        for _ in range(repeats):
            res = generate(base, prompt, IDEM_MAX_TOKENS, timeout=timeout)
            texts.append(strip_reasoning(res.get("text", "")))
        digests = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        rows.append(
            {
                "prompt": prompt,
                "texts": texts,
                "sha256": digests,
                "self_identical": len(set(digests)) == 1,
            }
        )
    return rows


def mode_idem_record(base: str, args) -> dict:
    """Boot A half: the no-draft greedy reference.

    Under a correct speculative implementation, greedy verification makes
    speculative decoding OUTPUT-IDENTICAL to non-speculative greedy: a
    rejected draft token must leave no trace. If the CSA/HCA/LID compressor
    writes in ``compressor_v2.forward_unified``
    (python/sglang/srt/layers/attention/dsv4/compressor_v2.py:516-596, which
    writes ``state_pool.kv_score_buffer.kv_score`` and, when
    ``online_c128_mtp`` is present, ``write_prefix_states``) are NOT idempotent
    under a re-run at the same positions, rejected drafts corrupt that state
    and the outputs diverge.

    So the correctness question is answered by comparison across Boot A and
    Boot B -- which is also why the ticket's boot ORDER is mandatory.

    PRECONDITION, checked here: each boot must first be SELF-identical
    (``repeats`` identical hashes). A boot that is not internally
    deterministic cannot support a cross-boot comparison, and the probe says
    so rather than reporting a divergence it cannot attribute. (This host has
    a known GDN-prefill nondeterminism above roughly 109 tokens, so the
    max_new_tokens here is deliberately modest and the self-identity check is
    not a formality.)
    """
    rows = _idem_runs(base, args.timeout, args.idem_repeats)
    all_self = all(r["self_identical"] for r in rows)
    for r in rows:
        print(
            f"  idem-record {'SELF-IDENTICAL' if r['self_identical'] else 'NOT SELF-IDENTICAL'}"
            f" :: {r['prompt'][:50]!r}",
            flush=True,
        )
    out = {
        "mode": "idem-record",
        "arm": args.arm,
        "self_identical_all": all_self,
        "rows": rows,
    }
    path = os.path.join(args.run, f"idem_reference_{args.arm}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"  reference written -> {path}", flush=True)
    return out


def mode_idem_compare(base: str, args) -> dict:
    """Boot B half: compare the draft arm against Boot A's greedy reference."""
    if not args.reference or not os.path.exists(args.reference):
        return {
            "mode": "idem-compare",
            "verdict": "NOT RUN",
            "note": (
                f"no Boot A reference at {args.reference!r}. TICKET_470 §5: "
                "'If Boot A cannot be run at all, do not run Boot B: an "
                "unattributed multiplier is not a result.' The same applies "
                "here -- without the reference there is nothing to compare."
            ),
        }
    with open(args.reference) as fh:
        ref = json.load(fh)
    rows = _idem_runs(base, args.timeout, args.idem_repeats)

    if not all(r["self_identical"] for r in rows):
        return {
            "mode": "idem-compare",
            "verdict": "INCONCLUSIVE",
            "note": (
                "this boot is not internally deterministic across repeats, so a "
                "cross-boot difference cannot be attributed to the draft arm. "
                "Report the self-identity failure; do not report a divergence."
            ),
            "rows": rows,
        }
    if not ref.get("self_identical_all"):
        return {
            "mode": "idem-compare",
            "verdict": "INCONCLUSIVE",
            "note": "the Boot A reference was not internally deterministic either",
            "rows": rows,
        }

    comparisons = []
    for ref_row, arm_row in zip(ref["rows"], rows):
        same = ref_row["sha256"][0] == arm_row["sha256"][0]
        comparisons.append(
            {
                "prompt": arm_row["prompt"],
                "reference_sha256": ref_row["sha256"][0],
                "arm_sha256": arm_row["sha256"][0],
                "identical": same,
            }
        )
        print(
            f"  idem-compare {'IDENTICAL' if same else 'DIVERGED'} :: {arm_row['prompt'][:50]!r}",
            flush=True,
        )
    all_same = all(c["identical"] for c in comparisons)
    return {
        "mode": "idem-compare",
        "verdict": "idempotent (greedy spec == greedy no-spec)" if all_same else "DIVERGENCE -- STOP AND REPORT",
        "note": (
            "Greedy speculative decoding must be output-identical to greedy "
            "non-speculative decoding. A divergence here is the ANALYSE_447 "
            "§2.4 failure mode: non-idempotent compressor writes at re-run "
            "positions after a rejected draft token. This is a correctness "
            "question and it OUTRANKS the perf numbers in this window."
        ),
        "comparisons": comparisons,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Selftest (rule 7) -- hermetic: no GPU, no server, no network.
# ---------------------------------------------------------------------------


def selftest() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    print("probes.py selftest (hermetic: no GPU, no server, no network)")

    # --- 1. every determined-answer scorer must SEPARATE good from bad ------
    print("\n scorers: can-discriminate on known-different inputs")
    for spec in DETERMINED:
        good_score, _ = spec["scorer"](spec["good"])
        bad_score, bad_note = spec["scorer"](spec["bad"])
        check(f"{spec['id']}: accepts the known-good answer", good_score == 1.0)
        check(f"{spec['id']}: rejects the known-bad answer", bad_score == 0.0, bad_note)

    # --- 2. reasoning stripping ---------------------------------------------
    print("\n reasoning stripping")
    check(
        "strips everything up to the last </think>",
        strip_reasoning("<think>ramble</think>  391 ") == "391",
    )
    check("leaves a plain answer alone", strip_reasoning(" 391 ") == "391")
    check(
        "a scorer sees through a reasoning prefix",
        DETERMINED[0]["scorer"]("<think>17*23...</think>391")[0] == 1.0,
    )

    # --- 3. the accept reader must refuse the EMA ---------------------------
    print("\n accept reader")
    real = read_accept({"spec_accept_length": 2.4, "spec_verify_ct": 10, "completion_tokens": 24})
    check("reads spec_accept_length", real["spec_accept_length"] == 2.4)
    check("reads spec_verify_ct", real["spec_verify_ct"] == 10)
    ema_only = read_accept({_EMA_KEY: 3.9})
    check(
        "REFUSES to substitute spec_ema_accept_len for the accept length",
        ema_only["spec_accept_length"] is None and "error" in ema_only,
        ema_only.get("error", "")[:70],
    )
    check(
        "still records the EMA as provenance",
        ema_only["prom_spec_ema_accept_len_provenance_only"] == 3.9,
    )

    # --- 4. the floor gate must be able to REFUSE ---------------------------
    print("\n A-vs-A floor gate")
    check("spread_pct on a known pair", abs(spread_pct(100.0, 110.0) - 9.5238) < 1e-3)
    check("spread_pct of identical readings is 0", spread_pct(42.0, 42.0) == 0.0)
    small = report_delta("small", 100.0, 101.0, floor_pct=5.0)
    check(
        "a 1% delta under a 5% floor is REFUSED",
        small["verdict"] == "below floor" and small["quotable"] is False,
    )
    big = report_delta("big", 100.0, 140.0, floor_pct=5.0)
    check(
        "a 40% delta over a 5% floor is admitted",
        big["verdict"] == "above floor" and big["quotable"] is True,
    )
    check("the signed delta keeps its sign", report_delta("s", 100.0, 90.0, 1.0)["delta_pct_signed"] < 0)

    # --- 5. ms/round arithmetic --------------------------------------------
    print("\n rate derivation")
    r = derive_rates({"decode_s": 2.0, "tokens": 40, "rounds": 10})
    check("ms/round from rounds, not tokens", r["ms_per_round"] == 200.0)
    check("ms/token stays separate", r["ms_per_token"] == 50.0)
    check("tok/s is present but labelled secondary", r["tok_per_s_secondary"] == 20.0)
    r0 = derive_rates({"decode_s": 1.0, "tokens": 0, "rounds": 0})
    check("zero tokens yields None, never a divide-by-zero", r0["ms_per_round"] is None)

    # --- 6. the SSE parser must have seen a known input ---------------------
    print("\n SSE parsing")
    check(
        "parses a data: payload",
        (parse_sse_line('data: {"text":"hi","meta_info":{"completion_tokens":3}}') or {}).get("text")
        == "hi",
    )
    check("ignores [DONE]", parse_sse_line("data: [DONE]") is None)
    check("ignores keep-alive lines", parse_sse_line("") is None and parse_sse_line(": ping") is None)
    check("ignores a malformed payload instead of raising", parse_sse_line("data: {not json") is None)

    # --- 7. the divergence comparator must be able to SEE a difference ------
    print("\n idempotence comparator")
    same = hashlib.sha256(b"abc").hexdigest()
    other = hashlib.sha256(b"abz").hexdigest()
    check("identical texts hash the same", same == hashlib.sha256(b"abc").hexdigest())
    check("different texts hash differently (can-fail arm)", same != other)

    # --- 7b. round classification -------------------------------------------
    # The counters live only on the final chunk (tokenizer_manager.py:2145-2153),
    # so these arms pin the three cases apart. The can-fail arm is the cut-off
    # one: a truncated stream must NOT be allowed to report verify rounds.
    print("\n round classification")
    fin = classify_rounds(False, 40, 20, 40)
    check("a finished spec run reports verify rounds", fin["round_kind"] == "verify")
    check("and rounds is the final verify count", fin["rounds"] == 20)
    check("and accept length is tokens/rounds", fin["accept_length"] == 2.0)
    cut = classify_rounds(True, 40, 20, 40)
    check(
        "a cut-off stream refuses verify rounds (can-fail arm)",
        cut["round_kind"].startswith("token (stream cut off"),
    )
    check("and falls back to token rounds", cut["rounds"] == 40)
    plain = classify_rounds(False, 40, None, 40)
    check(
        "a non-spec arm names the missing counter",
        plain["round_kind"] == "token (no spec_verify_ct in meta_info)",
    )
    check(
        "the three cases are distinguishable",
        len({fin["round_kind"], cut["round_kind"], plain["round_kind"]}) == 3,
    )
    check(
        "a zero verify count does not divide",
        classify_rounds(False, 40, 0, 40)["rounds"] == 40,
    )

    # --- 8. prompt construction --------------------------------------------
    print("\n prompt construction")
    lengths = [len(build_prompt(t)) for t in PREFILL_TARGETS]
    check("prompt length grows monotonically with the target", lengths == sorted(lengths))
    check("the targets match the prior window's points", PREFILL_TARGETS == (240, 480, 940, 1850))

    print(f"\n{'SELFTEST PASSED' if not failures else 'SELFTEST FAILED: ' + ', '.join(failures)}")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MODES: dict[str, Callable[[str, Any], dict]] = {
    "prefill": mode_prefill,
    "decode": mode_decode,
    "avsa": mode_avsa,
    "determined": mode_determined,
    "accept": mode_accept,
    "chatprobe": mode_chatprobe,
    "idem-record": mode_idem_record,
    "idem-compare": mode_idem_compare,
}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mode", nargs="?", choices=sorted(MODES) + ["all"], help="which probe to run")
    ap.add_argument("--selftest", action="store_true", help="hermetic instrument checks; needs no server")
    ap.add_argument("--port", type=int, default=30478)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--arm", default="unnamed")
    ap.add_argument("--run", default=os.environ.get("RUN", DEFAULT_RUN))
    ap.add_argument("--model", default="default", help="model name for the chat probe only")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument(
        "--window-seconds",
        type=float,
        default=15.0,
        help="wall-clock bound per decode point (10-20 s; GPU probes are bounded by TIME)",
    )
    ap.add_argument("--context-tokens", type=int, default=940)
    # Sized so a bs=1 decode COMPLETES inside the time budget rather than being
    # cut off: the speculative counters only exist on the final chunk, so a
    # truncated stream can never yield a verify round. At the ~7 tok/s this
    # checkpoint decodes at, 96 tokens lands around 14 s.
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--chat-probe-tokens", type=int, default=24)
    ap.add_argument(
        "--chat-template",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsv4f_chat_template.jinja"),
    )
    ap.add_argument(
        "--prompts",
        default="/spinning/gpu-battery-results/2026-08-03_447_dspark/prompts.json",
    )
    ap.add_argument("--reference", default=None, help="Boot A reference json for idem-compare")
    ap.add_argument("--idem-repeats", type=int, default=3)
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.mode:
        ap.error("give a mode, or --selftest")

    if not 10.0 <= args.window_seconds <= 20.0:
        print(
            f"NOTE: --window-seconds {args.window_seconds} is outside the 10-20 s "
            "band the house rule names for GPU probes. Recording it anyway; say "
            "so in the report.",
            file=sys.stderr,
        )

    base = f"http://{args.host}:{args.port}"
    os.makedirs(args.run, exist_ok=True)

    # /health, NOT /health_generate. The latter runs a REAL generation, and
    # this server boots with --max-running-requests 1, so the readiness call
    # that wait_ready already made can still be occupying the only slot when
    # the probes start. Measured on the 1a baseline arm: the server was alive
    # and idle in uvicorn (py-spy confirmed) while five consecutive
    # /health_generate probes returned 0, the whole arm was skipped, and the
    # boot was torn down having produced nothing. /health is a cheap liveness
    # check and cannot queue behind a generation.
    health = _get(base, "/health", timeout=30.0)
    if health["status"] != 200:
        print(f"REFUSED: {base}/health answered {health['status']}", file=sys.stderr)
        return 2

    if args.mode == "all":
        # The floor comes FIRST: rule 4 says no delta is reported before its
        # own point's floor exists.
        #
        # idem-record / idem-compare are deliberately NOT in "all". They are a
        # PAIR across two boots -- one arm records the reference, the other
        # compares against it -- so which of the two a given arm should run is
        # a property of the arm, not of the probe set. Running both blindly
        # would have b_dspark write a reference nothing reads and have a_base
        # compare against a reference that does not exist yet. Each boot
        # script calls the correct half explicitly.
        modes = ["avsa"] + [
            m for m in sorted(MODES) if m != "avsa" and not m.startswith("idem-")
        ]
    else:
        modes = [args.mode]

    results: dict[str, Any] = {
        "arm": args.arm,
        "port": args.port,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "window_seconds": args.window_seconds,
        "results": {},
    }
    for mode in modes:
        print(f"[{mode}] arm={args.arm}", flush=True)
        try:
            results["results"][mode] = MODES[mode](base, args)
        except Exception as exc:  # noqa: BLE001 - one bad probe must not lose the others
            results["results"][mode] = {"mode": mode, "error": repr(exc)}
            print(f"  {mode} raised: {exc!r}", file=sys.stderr, flush=True)

    out = os.path.join(args.run, f"probes_{args.arm}_{args.mode}.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
