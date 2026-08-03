#!/usr/bin/env python3
"""s12 -- the multi-session prefill curve, BAR1 against the baseline.

Two modes, because the measurement runs where the server runs (the PVE host,
standard library only, no venv) while the summary runs in the container next to
the rest of the battery:

  --mode messen           one point against a LIVE server: prefill throughput
                          at N sessions, plus one decode point at bs=1 and one
                          at bs=16. Appends to punkte.jsonl.
  --mode zusammenfassen   punkte.jsonl -> prefill_kurve.json + the live table.

Why the measurement looks the way it does:

* THE NUMERATOR IS WHAT THE SERVER COUNTED. prompt_tokens from the response,
  never an estimate from the prompt string. The tokenizer is the authority on
  how many tokens went in.
* EVERY PROMPT IS UNIQUE, at the FRONT. A shared prefix would be served out of
  the radix cache and the second round would measure the cache, not the
  prefill. /flush_cache is called on top.
* WARMUP BEFORE EVERY POINT, AND THE ARTIFACT SAYS SO. 726 us against 95 us
  purely from the P-state ramp is the measured cost of skipping it (05_FALLEN,
  rule 4). The warm-up draw ran before #459 too, but it vanished without a
  trace, so a warmed-up point and a cold one produced the same file. It is now
  draw 0, recorded WITH its own rate under ``floor_series.warmup_draw`` and
  flagged discarded.
* A FLOOR ROUND IS ONE INVOCATION, NOT THREE. ``--floor-draws N`` runs the N
  A-vs-A draws back to back inside this process and records the measured idle
  gap before each one. Repeating the whole invocation instead put ~50 s of idle
  between draws, and #475 SS6 measured what that costs: a monotone clock ramp
  reported as a 13 % noise floor where the same instrument reports 3 %.
* TIME-BOXED, 10-20 s per point, and the raw per-request rows are persisted
  rather than only the aggregate (rule 6).
* ONE CONTENT CLASS. The content axis belongs to the accept boots (s02-s05);
  here every arm sees byte-identical prompts, because the comparison is between
  transports and content variance would be noise in it.

What this file does NOT do: judge. Whether the curve stays flat or starts to
rise is the finding the whole exercise is for, and it is written down, not
graded.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from s11_bar1_e2e import RE_GROUP  # noqa: E402  one regex, one place
from s12_log_analyse import (  # noqa: E402  one parser, one place
    decode_tick_aggregat,
    im_fenster,
    parse_decode,
)

KIND = "bar1_prefill_kurve"
#: 2: the per-point transport evidence under `groups` renamed its keys from
#: German to English (`gruppe`/`angefordert`/`erreicht` ->
#: `group`/`requested`/`achieved`), in step with s11_bar1_e2e.RE_GROUP and
#: barlink.py's report_state(). A schema-1 artifact spells them in German, so
#: every point would read back as having no barlink group at all -- which the
#: arm check would misread as "baseline". Rejecting it by version is the
#: point: re-run the step rather than read a stale artifact.
#: 3: task #358 put the remaining German keys of the per-point transport
#: evidence into English -- `beleg_vorhanden` -> `evidence_present`,
#: `gruppen` -> `groups`. A schema-2 artifact reads back as no evidence
#: at all through the new names, so it is rejected by version instead.
SCHEMA_VERSION = 3

# The baseline over the host path, from MESSUNG_PREFILL_ANTEIL.md and
# 02_WAS_ERREICHT_IST.md: "1190/1097/1144/1105/1122 tok/s" over 1, 2, 4, 8 and
# 16 sessions. The three published points are 1190,7 / 1143,7 / 1122,4 at 1, 4
# and 16; 1097 and 1105 are the two that were filled in afterwards, which fixes
# the mapping of the five numbers to the five session counts.
BASELINE_KNOWN = {1: 1190.7, 2: 1097.0, 4: 1143.7, 8: 1105.0, 16: 1122.4}
BASELINE_SOURCE = (
    "MESSUNG_PREFILL_ANTEIL.md / 02_WAS_ERREICHT_IST.md (host path, NCCL, "
    "Qwen3.6-27B-FP8 tp3, noise floor 0.47 %)"
)

ARMS = ("bar1", "grundlinie")
DECODE_BATCHES = (1, 16)

# ---------------------------------------------------------------------------
# floor draws (#459 / #475 SS6)
# ---------------------------------------------------------------------------
#
# A "floor draw" is one repetition of an identical recipe; the spread over the
# draws of one arm IS the A-vs-A noise floor that every later delta is measured
# against. #475 SS6 took that floor apart: sub-arm B2 of #435 drew
# 1597.7 / 1720.2 / 1820.2 tok/s and reported 13.0 %, where #424 reported 3.0 %
# on the same instrument. From the CollectiveClock lines the collective axis is
# flat to 1.6 % across those three draws and the ENTIRE spread is compute, and
# monotone. The draws were 48-51 s apart with ~12 s of work each, so the cards
# idled ~75 % of every cycle and every draw paid part of a clock ramp.
#
# Three draws with a monotone trend are not exchangeable samples, so their
# spread is drift reported as noise -- and it is loose in exactly the direction
# that lets a real regression pass as "within the floor" (#435 sub-arm B scored
# -8.5 % as parity against that 13.0 % floor). The remedy is procedural, not a
# wider floor: ONE warm-up draw that is discarded explicitly, then the measured
# draws BACK TO BACK with no idle gap.
#
# Both halves have to be visible in the artifact. A run that cannot show which
# draw it discarded, and cannot show what the gaps between its draws were, is
# not evidence for either property -- it is the same file as a run that did
# neither.

#: Two consecutive draws count as back-to-back when the second one starts
#: within this many seconds of the previous one's end. The gap is MEASURED and
#: recorded per draw; this constant only turns it into a verdict. It is far
#: below the 48-51 s of the #475 SS6 arms and above the sub-second
#: ``/flush_cache`` that has to run between two prefill draws.
BACK_TO_BACK_MAX_GAP_S = 5.0

#: The warm-up draw is draw 0 and is never a measurement. The measured draws
#: are numbered from 1, which is also the ``_floorPn`` suffix they are filed
#: under (``scripts/dev/475_prefill_barrier/window_accounting.py`` already
#: strips exactly that suffix to map a draw back to its arm).
WARMUP_DRAW_INDEX = 0

WARMUP_DISCARD_REASON = (
    "clock ramp: the first draw against idle cards measures the P-state ramp "
    "rather than the steady state (#475 SS6, 726 us against 95 us in "
    "05_FALLEN rule 4). Discarded by construction, never averaged in."
)


def _decode_batches(args) -> tuple:
    """Decode batch sizes for this point.

    The default stays (1, 16) so every earlier step measures byte-identically;
    --decode-batches lets a caller ask for the batch sizes ITS question is
    about. s15 asks for 1 and 8, because its prefill points are 1 and 8
    sessions and a phase comparison that reads prefill at 8 and decode at 16 is
    comparing two different load points.
    """
    raw = getattr(args, "decode_batches", None)
    if not raw:
        return DECODE_BATCHES
    return tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())

# The prompt and the filler corpus stay GERMAN on purpose: they are
# measurement input, not prose. Both arms have to see byte-identical prompts,
# and every recorded output_sample of every past run was generated from
# exactly these bytes -- translating them would silently change the workload.
PROSE = (
    "Erklaere in ruhigem, sachlichem Ton, wie ein Rechner mehrere Aufgaben "
    "gleichzeitig bearbeitet, und woran man merkt, dass er dabei an eine "
    "Grenze stoesst."
)

_WORDS = (
    "die messung zeigt einen deutlichen unterschied zwischen der erwartung und "
    "dem befund auf dieser maschine denn der weg ueber den hauptspeicher kostet "
    "zeit die niemand sieht solange die karten wenig zu tun haben und erst unter "
    "last wird aus dem umweg ein deckel der sich in jeder zahl niederschlaegt"
).split()


def _post(port: int, path: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode(errors="replace"))


def _post_stream(port: int, payload: dict, timeout: float, deadline: float) -> dict:
    """Returns {"chunks": [monotonic timestamps], "text": str}."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks = []
    text_parts = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            if time.monotonic() > deadline:
                break
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if not body or body == "[DONE]":
                continue
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            try:
                delta = obj["choices"][0].get("delta") or {}
            except (KeyError, IndexError, TypeError):
                continue
            piece = delta.get("content")
            if piece:
                chunks.append(time.monotonic())
                text_parts.append(piece)
    return {"chunks": chunks, "text": "".join(text_parts)}


def _flush_cache(port: int) -> str:
    for path in ("/flush_cache",):
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}", method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return f"{path}:{resp.status}"
        except (urllib.error.URLError, OSError) as exc:
            return f"{path}:{exc}"
    return "not called"


def _prompt(unique: str, target_tokens: int) -> str:
    """Unique marker FIRST, then deterministic filler.

    Deterministic so both arms see byte-identical prompts for the same
    (session, round); unique so the radix cache cannot serve any of it."""
    rng = random.Random(unique)
    words = int(target_tokens / 1.3)
    body = " ".join(rng.choice(_WORDS) for _ in range(words))
    return f"[{unique}] {body}"


# ---------------------------------------------------------------------------
# measuring one point
# ---------------------------------------------------------------------------


def measure_prefill(
    port: int, arm: str, sessions: int, seconds: float, target_tokens: int
) -> dict:
    rows: list = []
    lock = threading.Lock()
    stop_at = time.monotonic() + seconds
    errors: list = []

    def worker(slot: int) -> None:
        i = 0
        while time.monotonic() < stop_at:
            i += 1
            unique = f"{arm[:1]}{sessions}-{slot}-{i}-{int(stop_at)}"
            payload = {
                "model": "default",
                "messages": [
                    {"role": "user", "content": _prompt(unique, target_tokens)}
                ],
                "max_tokens": 1,
                "temperature": 0,
            }
            t0 = time.monotonic()
            try:
                answer = _post(port, "/v1/chat/completions", payload, timeout=180)
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
                with lock:
                    errors.append(f"{type(exc).__name__}: {exc}")
                return
            t1 = time.monotonic()
            usage = answer.get("usage") or {}
            with lock:
                rows.append(
                    {
                        "slot": slot,
                        "runde": i,
                        "start": t0,
                        "ende": t1,
                        "prompt_tokens": usage.get("prompt_tokens"),
                    }
                )

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(sessions)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=seconds + 240)

    good = [r for r in rows if isinstance(r.get("prompt_tokens"), int)]
    if not good:
        return {
            "requests": 0,
            "prefill_tok_s": None,
            "fehler": errors[:3] or ["no answer carrying prompt_tokens"],
        }
    first = min(r["start"] for r in good)
    last = max(r["ende"] for r in good)
    wall = last - first
    tokens = sum(r["prompt_tokens"] for r in good)
    latencies = sorted((r["ende"] - r["start"]) * 1000.0 for r in good)
    return {
        "requests": len(good),
        "prompt_tokens_total": tokens,
        "prompt_tokens_mean": tokens / len(good),
        "wall_s": wall,
        "prefill_tok_s": (tokens / wall) if wall > 0 else None,
        "latenz_ms_p50": statistics.median(latencies),
        "latenz_ms_max": latencies[-1],
        "fehler": errors[:3],
        "roh": good,
    }


def run_draw_series(draw, draws: int, *, warmup: bool, between=None) -> tuple:
    """One discarded warm-up draw, then ``draws`` measured draws back-to-back.

    ``draw(index)`` runs a single draw and returns its prefill dict; index 0 is
    the warm-up. ``between()`` runs in the GAP between two draws (the harness
    passes ``/flush_cache`` there), so whatever it costs is counted into the
    measured gap instead of disappearing inside a draw.

    Returns ``(series, records)`` where ``series`` is the block the artifact
    carries -- which draw was discarded, why, and the measured idle gap before
    every draw -- and ``records`` is ``[(index, gap_before_s, prefill), ...]``
    for the measured draws only.

    Nothing here is asserted: ``back_to_back`` is derived from gaps this
    function timed itself, so an artifact that claims it also carries the
    numbers to refute it.
    """
    series = {
        "draws": int(draws),
        "warmup_draw": None,
        "discarded_draws": [],
        "gap_before_s": [],
        "back_to_back": None,
        "back_to_back_max_gap_s": BACK_TO_BACK_MAX_GAP_S,
    }
    records: list = []
    last_end = None

    if warmup:
        t0 = time.monotonic()
        result = draw(WARMUP_DRAW_INDEX)
        last_end = time.monotonic()
        series["warmup_draw"] = {
            "draw": WARMUP_DRAW_INDEX,
            "discarded": True,
            "reason": WARMUP_DISCARD_REASON,
            "seconds": round(last_end - t0, 3),
            "prefill_tok_s": result.get("prefill_tok_s"),
            "requests": result.get("requests"),
        }
        series["discarded_draws"].append(WARMUP_DRAW_INDEX)

    for index in range(1, int(draws) + 1):
        if between is not None and last_end is not None:
            between()
        t0 = time.monotonic()
        gap = None if last_end is None else round(t0 - last_end, 3)
        series["gap_before_s"].append(gap)
        result = draw(index)
        last_end = time.monotonic()
        records.append((index, gap, result))

    gaps = [g for g in series["gap_before_s"] if g is not None]
    series["max_gap_s"] = max(gaps) if gaps else None
    # None, not True: with no warm-up and a single draw there is no gap to
    # judge, and "no gap measured" must not read as "no gap existed".
    series["back_to_back"] = (
        all(g <= BACK_TO_BACK_MAX_GAP_S for g in gaps) if gaps else None
    )
    return series, records


def floor_from_points(points: list) -> list:
    """The A-vs-A floor of every multi-draw arm, out of the persisted points.

    Reports the spread AND the two properties that decide whether the spread is
    a noise floor at all: were the draws back-to-back, and is the sequence
    MONOTONE. A monotone series is a drift, not exchangeable samples, and its
    spread must not be used as a floor even when it looks small (#475 SS6).
    """
    by_arm: dict = {}
    for p in points:
        series = p.get("floor_series") or {}
        if not series.get("draws") or series["draws"] < 2:
            continue
        key = (p.get("base_arm") or p.get("arm"), p.get("sessions"))
        by_arm.setdefault(key, []).append(p)

    out = []
    for (arm, sessions), group in sorted(by_arm.items(), key=lambda kv: str(kv[0])):
        group.sort(key=lambda p: p.get("draw") or 0)
        rates = [(p.get("prefill") or {}).get("prefill_tok_s") for p in group]
        good = [r for r in rates if isinstance(r, (int, float)) and r > 0]
        rising = all(b > a for a, b in zip(good, good[1:])) if len(good) > 1 else False
        falling = all(b < a for a, b in zip(good, good[1:])) if len(good) > 1 else False
        series = group[0].get("floor_series") or {}
        out.append(
            {
                "arm": arm,
                "sessions": sessions,
                "draws": len(group),
                "prefill_tok_s": rates,
                "spread_pct": (
                    (max(good) - min(good)) / min(good) * 100.0 if len(good) > 1 else None
                ),
                "monotone": rising or falling,
                "back_to_back": series.get("back_to_back"),
                "max_gap_s": series.get("max_gap_s"),
                "discarded_draws": series.get("discarded_draws"),
                "warmup_prefill_tok_s": (series.get("warmup_draw") or {}).get(
                    "prefill_tok_s"
                ),
            }
        )
    return out


def ernte_ticks(server_log: str, start: float, end: float, batch: int) -> dict:
    """The scheduler's own decode ticks for the window a decode point ran in.

    Why this exists at all. The request level and the tick level answer two
    different questions and the 2026-07-30 run only ever recorded the first:

    * ACCEPT was None in every one of the eight 2026-07-30 points, because the
      probe asked `/v1/chat/completions`, where `meta_info` is opt-in
      (`protocol.py` `return_meta_info: bool = False`) and the request never
      set it. Fixed in #326 by moving `_probe_accept_length` to `/generate`,
      which carries meta_info on every response; a probe that still comes back
      with no meta_info at all is now `accept_probe_fatal`, not a silent None.
      The scheduler logs `accept len` on every tick no matter who asked, which
      is why the tick source below exists independently of the probe.
    * THE RATE was the request level's: tokens of a stream over the wall clock
      of that stream, prefill and queue included. That reported 32 tok/s at
      bs=1 where the decode loop was turning over 77-83.

    Neither number replaces the other, so both are kept and both are labelled.
    """
    out = {"tick_quelle": server_log or None}
    if not server_log or not os.path.exists(server_log):
        out["tick_fehler"] = "kein Serverlog"
        return out
    try:
        with open(server_log, errors="replace") as f:
            ticks = parse_decode(f)
    except OSError as exc:
        out["tick_fehler"] = f"{type(exc).__name__}: {exc}"
        return out
    # Prefixed, because `ms_pro_token` exists on both levels and an unprefixed
    # merge would silently replace the request-level number with the tick one --
    # exactly the conflation this fix is about.
    agg = decode_tick_aggregat(im_fenster(ticks, start, end), batch)
    out.update({f"tick_{k}": v for k, v in agg.items()})
    return out


# The accept probe hits the NATIVE /generate endpoint, not
# /v1/chat/completions. `meta_info` is opt-in on the chat endpoint --
# `return_meta_info: bool = False` in protocol.py -- and the 2026-07-30 run
# never set it, so `spec_accept_length` came back None on all eight of that
# run's arms, silently, and every ms_pro_verify derived from those arms was
# void (task #326). `/generate` carries meta_info on every response with no
# flag to opt into -- the same reason s14_decode_punkt.py's Stream and its
# accept probe use it instead of the chat endpoint.
ACCEPT_PROBE_PATH = "/generate"
#: Named constants, not inline literals, so the marker-coupling test (#315
#: lesson: couple assertions to the real source, never retype a literal by
#: hand) imports these instead of guessing at the wording. A None
#: `spec_accept_length` must always carry exactly one of these in
#: `accept_probe_note` -- a bare None with no explanation is the silent shape
#: that let eight void arms pass as measurements on 2026-07-30.
ACCEPT_PROBE_NOTE_GENERATE_ERROR = "generate error: {message}"
ACCEPT_PROBE_NOTE_NO_META_INFO = (
    "no meta_info in /generate response -- structurally impossible on this "
    "endpoint, treat as a probe bug, not a content result"
)
ACCEPT_PROBE_NOTE_NO_SPEC_ACCEPT_LENGTH = (
    "meta_info carried no spec_accept_length (no speculative decoding on this boot)"
)
ACCEPT_PROBE_NOTE_EXCEPTION = "{exc_type}: {exc}"


def _probe_accept_length(port: int, timeout: float = 180) -> dict:
    """One bounded, non-streaming request on /generate, after the streamed
    batch: `spec_accept_length` is the only honest accept number (never
    `spec_ema_accept_len` from the log), and it rides on `meta_info`, which
    `/generate` attaches to every response without an opt-in flag.

    Returns a dict with:
    * `spec_accept_length` -- the value, or None.
    * `accept_probe_note` -- None when the value was read cleanly, otherwise
      one of the ACCEPT_PROBE_NOTE_* constants above, naming WHY it is None.
    * `accept_probe_fatal` -- True only for the class of bug #326 fixed: the
      probe could not be answered at all (network/JSON failure, an `error`
      object, or a response with no `meta_info` whatsoever -- structurally
      impossible on this endpoint). A `meta_info` that legitimately carries
      no spec block, because this boot runs without speculative decoding, is
      noted but NOT fatal: that is a true absence, not a broken probe.
    * `output_sample` -- the generated text, if any, for output_sample
      fallback.
    """
    try:
        answer = _post(
            port,
            ACCEPT_PROBE_PATH,
            {
                "text": PROSE,
                "sampling_params": {"max_new_tokens": 64, "temperature": 0},
            },
            timeout=timeout,
        )
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        return {
            "spec_accept_length": None,
            "accept_probe_note": ACCEPT_PROBE_NOTE_EXCEPTION.format(
                exc_type=type(exc).__name__, exc=exc
            ),
            "accept_probe_fatal": True,
            "output_sample": None,
        }

    if isinstance(answer, dict) and isinstance(answer.get("error"), dict):
        return {
            "spec_accept_length": None,
            "accept_probe_note": ACCEPT_PROBE_NOTE_GENERATE_ERROR.format(
                message=str(answer["error"].get("message"))[:200]
            ),
            "accept_probe_fatal": True,
            "output_sample": None,
        }

    meta = (answer.get("meta_info") or {}) if isinstance(answer, dict) else {}
    sample = (answer.get("text") or "")[:400] if isinstance(answer, dict) else None
    if not meta:
        return {
            "spec_accept_length": None,
            "accept_probe_note": ACCEPT_PROBE_NOTE_NO_META_INFO,
            "accept_probe_fatal": True,
            "output_sample": sample,
        }
    if "spec_accept_length" not in meta:
        return {
            "spec_accept_length": None,
            "accept_probe_note": ACCEPT_PROBE_NOTE_NO_SPEC_ACCEPT_LENGTH,
            "accept_probe_fatal": False,
            "output_sample": sample,
        }
    return {
        "spec_accept_length": meta.get("spec_accept_length"),
        "accept_probe_note": None,
        "accept_probe_fatal": False,
        "output_sample": sample,
    }


def measure_decode(port: int, batch: int, seconds: float) -> dict:
    results: list = []
    lock = threading.Lock()
    deadline = time.monotonic() + seconds

    def worker() -> None:
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": PROSE}],
            "max_tokens": 256,
            "temperature": 0,
            "stream": True,
        }
        try:
            out = _post_stream(port, payload, timeout=180, deadline=deadline)
        except (urllib.error.URLError, OSError) as exc:
            with lock:
                results.append({"fehler": f"{type(exc).__name__}: {exc}"})
            return
        with lock:
            results.append(out)

    threads = [threading.Thread(target=worker) for _ in range(batch)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=seconds + 240)

    good = [r for r in results if r.get("chunks")]
    if not good:
        return {
            "batch": batch,
            "decode_tok_s": None,
            "fehler": [r.get("fehler") for r in results][:3] or ["no streamed answer"],
        }
    first = min(r["chunks"][0] for r in good)
    last = max(r["chunks"][-1] for r in good)
    tokens = sum(len(r["chunks"]) - 1 for r in good)
    span = last - first
    rate = (tokens / span) if span > 0 and tokens > 0 else None

    probe = _probe_accept_length(port)
    accept = probe["spec_accept_length"]
    sample = good[0]["text"][:400]
    if not sample:
        if probe.get("output_sample"):
            sample = probe["output_sample"]
        elif probe.get("accept_probe_note"):
            sample = f"(no answer: {probe['accept_probe_note']})"

    return {
        "batch": batch,
        "requests": len(good),
        "tokens": tokens,
        "span_s": span,
        "decode_tok_s": rate,
        "ms_pro_token": (1000.0 / rate) if rate else None,
        "ms_pro_verify": (1000.0 / rate * accept) if rate and accept else None,
        "spec_accept_length": accept,
        # Never silently None: see ACCEPT_PROBE_NOTE_* above.
        "accept_probe_note": probe.get("accept_probe_note"),
        "accept_probe_fatal": probe.get("accept_probe_fatal", False),
        "output_sample": sample,
        "fehler": [r.get("fehler") for r in results if r.get("fehler")][:3],
    }


def draw_label(arm: str, index: int, draws: int) -> str:
    """The arm name a single draw is filed under.

    A single-draw point keeps the plain arm name, so every earlier step reads
    back byte-identically. A floor series files draw n as ``<arm>_floorPn`` --
    the suffix ``window_accounting.py`` already strips to map a draw back to
    its arm.
    """
    return arm if draws < 2 else f"{arm}_floorP{index}"


def mode_measure(args) -> int:
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    draws = max(1, int(getattr(args, "floor_draws", 1) or 1))

    flush = _flush_cache(args.port)

    def one_draw(index: int) -> dict:
        seconds = (
            args.warmup_seconds if index == WARMUP_DRAW_INDEX else args.point_seconds
        )
        return measure_prefill(
            args.port, args.arm, args.sessions, seconds, args.prompt_tokens
        )

    # The flush runs BETWEEN the draws, not inside one: every draw has to
    # prefill rather than hit the radix cache, and what that costs belongs in
    # the measured gap. With draws=1 the sequence is flush / warm-up / flush /
    # draw -- exactly what this step did before the series existed.
    series, records = run_draw_series(
        one_draw,
        draws,
        warmup=args.warmup_seconds > 0,
        between=lambda: _flush_cache(args.port),
    )

    # The decode phase runs ONCE, after the whole series: it is minutes of a
    # different load, and running it between two prefill draws would insert
    # exactly the idle gap the series exists to remove.
    decode = []
    if args.with_decode:
        for batch in _decode_batches(args):
            start = time.time()
            point_decode = measure_decode(args.port, batch, args.point_seconds)
            point_decode.update(ernte_ticks(args.server_log, start, time.time(), batch))
            decode.append(point_decode)

    rc = 0
    for index, gap, prefill in records:
        raw = prefill.pop("roh", [])
        arm_label = draw_label(args.arm, index, draws)
        point = {
            "folge": args.folge,
            "arm": arm_label,
            "base_arm": args.arm,
            "draw": index,
            "sessions": args.sessions,
            "zeit": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "flush_cache": flush,
            "point_seconds": args.point_seconds,
            "warmup_seconds": args.warmup_seconds,
            "prompt_tokens_ziel": args.prompt_tokens,
            "inhalt": "filler prose, unique head per request",
            # The evidence for both harness properties, per point: which draw
            # was discarded and why, and the measured idle gap before this one.
            "floor_series": series,
            "gap_before_s": gap,
            "prefill": prefill,
            # Attached to the last draw only -- one decode phase ran, and
            # copying it onto every draw would read as one phase per draw.
            "decode": decode if index == records[-1][0] else [],
        }
        with open(os.path.join(out_dir, "punkte.jsonl"), "a") as f:
            f.write(json.dumps(point) + "\n")
        with open(
            os.path.join(out_dir, f"roh_{arm_label}_{args.sessions}.jsonl"), "w"
        ) as f:
            for row in raw:
                f.write(json.dumps(row) + "\n")
        print(
            f"point {arm_label}/{args.sessions}: "
            f"{prefill.get('prefill_tok_s')} tok/s from "
            f"{prefill.get('requests')} requests"
        )
        # A point without a rate is not a point. Returning 0 here would let the
        # orchestrator keep booting arms against a server that answers nothing.
        if prefill.get("prefill_tok_s") is None:
            for err in prefill.get("fehler") or []:
                print(f"  error: {err}", file=sys.stderr)
            rc = 1

    if series.get("warmup_draw"):
        w = series["warmup_draw"]
        print(
            f"  draw {w['draw']} DISCARDED (warm-up, "
            f"{w.get('prefill_tok_s')} tok/s), draws back-to-back: "
            f"{series.get('back_to_back')} (max gap {series.get('max_gap_s')} s)"
        )

    # A decode point whose accept probe failed structurally (bug #326: the
    # old probe hit /v1/chat/completions, which never attaches meta_info, so
    # spec_accept_length was None on every arm and every ms_pro_verify derived
    # from it was void, silently). Failing loudly here is the point of the
    # fix: a None that could not possibly have been anything else must not
    # pass through as a measurement.
    fatal_decode = [d for d in decode if d.get("accept_probe_fatal")]
    if fatal_decode:
        for d in fatal_decode:
            print(
                f"  error: accept probe at bs={d.get('batch')} failed "
                f"structurally: {d.get('accept_probe_note')}",
                file=sys.stderr,
            )
        return 1
    return rc


# ---------------------------------------------------------------------------
# summary and the live table
# ---------------------------------------------------------------------------


def load_points(step_dir: str) -> list:
    path = os.path.join(step_dir, "punkte.jsonl")
    points: list = []
    if not os.path.exists(path):
        return points
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                points.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    points.sort(key=lambda p: p.get("folge", 0))
    return points


def load_evidence(step_dir: str, folge, arm, sessions) -> dict:
    """What the boot behind a point REALLY ran.

    The transport name in the log says bar1 either way -- that cost a whole
    measurement once already. So each boot leaves the ERREICHT lines behind and
    the point carries them: a bar1 point whose dcp group fell back to gloo is
    not a bar1 point, and a baseline point with any barlink group is not a
    baseline point.
    """
    path = os.path.join(step_dir, "belege", f"{folge}_{arm}_{sessions}.txt")
    out: dict = {"evidence_present": os.path.exists(path), "groups": []}
    if not out["evidence_present"]:
        return out
    with open(path, errors="replace") as f:
        for line in f:
            m = RE_GROUP.search(line)
            if m:
                out["groups"].append(
                    {
                        "group": m.group("group"),
                        "requested": m.group("requested"),
                        "achieved": m.group("achieved"),
                    }
                )
    return out


def load_fatal(step_dir: str, arm, sessions) -> dict:
    """The fatal harvest of ONE boot.

    s12_prefill_kurve.sh greps every boot's host log for OOM / NCCL error /
    traceback into logs/<arm>_<n>.fatal.txt. Nothing read that file, so eight
    boots that each died in a prefill OOM could still hand in throughput
    numbers and pass -- the only step in the battery without a fatal gate.
    An EMPTY file is the healthy case: the grep found nothing.

    Lines the server QUOTED from a helper subprocess it caught and recovered
    from (the stage-0 hardware probe) are skipped: a traceback in there is
    evidence about the helper, not about this boot -- see
    check_common.QUOTED_SUBLOG_PREFIX, keep the literals in step.
    """
    quoted_prefix = "[probe-subprocess] "
    name = f"logs/{arm}_{sessions}.fatal.txt"
    path = os.path.join(step_dir, name)
    if not os.path.exists(path):
        return {"fatal_erhoben": False, "fatal": None}
    with open(path, errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip() and quoted_prefix not in line:
                return {
                    "fatal_erhoben": True,
                    "fatal": f"{name}:{lineno}: {' '.join(line.split())[:200]}",
                }
    return {"fatal_erhoben": True, "fatal": None}


def zusammenfassen(step_dir: str, tol_pct: float, plan: list) -> dict:
    points = load_points(step_dir)
    curves: dict = {arm: {} for arm in ARMS}
    decode_points: list = []
    order = []
    for p in points:
        arm = p.get("arm")
        sessions = p.get("sessions")
        rate = (p.get("prefill") or {}).get("prefill_tok_s")
        if arm in curves and isinstance(sessions, int):
            curves[arm][sessions] = rate
        record = {
            "folge": p.get("folge"),
            "arm": arm,
            "sessions": sessions,
            "zeit": p.get("zeit"),
            "prefill_tok_s": rate,
            # Per point, so the order table shows for every single draw which
            # draw was discarded and how long the cards idled before it.
            "draw": p.get("draw"),
            "gap_before_s": p.get("gap_before_s"),
            "floor_series": p.get("floor_series"),
        }
        record.update(load_evidence(step_dir, p.get("folge"), arm, sessions))
        record.update(load_fatal(step_dir, arm, sessions))
        order.append(record)
        for d in p.get("decode") or []:
            entry = dict(d)
            entry["arm"] = arm
            entry["sessions_des_boots"] = sessions
            decode_points.append(entry)

    deviation = {}
    for sessions, rate in curves.get("grundlinie", {}).items():
        known = BASELINE_KNOWN.get(sessions)
        if known and isinstance(rate, (int, float)):
            deviation[str(sessions)] = (rate - known) / known * 100.0

    ratio = {}
    for sessions, rate in curves.get("bar1", {}).items():
        base = curves.get("grundlinie", {}).get(sessions)
        if base and isinstance(rate, (int, float)) and base > 0:
            ratio[str(sessions)] = rate / base

    def _short(name: str):
        path = os.path.join(step_dir, name)
        if not os.path.exists(path):
            return None
        with open(path, errors="replace") as f:
            return " ".join(f.read().split())[:200] or "no reason given"

    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "arme": list(ARMS),
        "sessions_geplant": plan,
        "abbruch": _short("abbruch.txt"),
        # A step that never got the cards must not be diagnosed through the
        # empty artifacts it left behind.
        "blockiert": _short("blocked.txt"),
        "host_erreichbar": not os.path.exists(
            os.path.join(step_dir, "host_unreachable.txt")
        ),
        "integration_vorhanden": not os.path.exists(
            os.path.join(step_dir, "integration_missing.txt")
        ),
        "punkte": len(points),
        "reihenfolge": order,
        # The A-vs-A floor of every multi-draw arm, with the two properties
        # that decide whether its spread may be used as a floor at all
        # (back-to-back, and not monotone) -- #475 SS6.
        "floor": floor_from_points(points),
        # One list, so the check does not have to walk the boots to find out
        # whether any of them died. Empty means every harvest came back clean.
        "fatal": [
            {
                "folge": e.get("folge"),
                "arm": e.get("arm"),
                "sessions": e.get("sessions"),
                "line": e["fatal"],
            }
            for e in order
            if e.get("fatal")
        ],
        "fatal_ungeprueft": [
            {
                "folge": e.get("folge"),
                "arm": e.get("arm"),
                "sessions": e.get("sessions"),
            }
            for e in order
            if not e.get("fatal_erhoben")
        ],
        "kurven": {
            arm: {str(k): v for k, v in sorted(d.items())} for arm, d in curves.items()
        },
        "decode": decode_points,
        "grundlinie_bekannt": {str(k): v for k, v in BASELINE_KNOWN.items()},
        "grundlinie_quelle": BASELINE_SOURCE,
        "grundlinie_abweichung_pct": deviation,
        "toleranz_pct": tol_pct,
        "verhaeltnis_bar1_zu_grundlinie": ratio,
        "output_samples": {
            arm: next(
                (
                    d.get("output_sample")
                    for d in decode_points
                    if d.get("arm") == arm and d.get("output_sample")
                ),
                None,
            )
            for arm in ARMS
        },
        "rohdaten": (
            sorted(f for f in os.listdir(step_dir) if f.startswith("roh_"))
            if os.path.isdir(step_dir)
            else []
        ),
    }


def _num(v, nk: int) -> str:
    return "None" if not isinstance(v, (int, float)) else format(v, f".{nk}f")


def tabelle(payload: dict) -> str:
    """The live table, rendered from the persisted points. Pure presentation:
    no marking, no comparison with an expectation, no verdict."""
    lines = [
        "| sessions | bar1 tok/s | grundlinie tok/s | bar1/grundlinie |",
        "|---:|---:|---:|---:|",
    ]
    sessions = sorted({int(s) for arm in ARMS for s in payload["kurven"].get(arm, {})})
    for s in sessions:
        a = payload["kurven"].get("bar1", {}).get(str(s))
        b = payload["kurven"].get("grundlinie", {}).get(str(s))
        r = payload["verhaeltnis_bar1_zu_grundlinie"].get(str(s))
        lines.append(
            f"| {s} | {a if a is None else format(a, '.1f')} "
            f"| {b if b is None else format(b, '.1f')} "
            f"| {r if r is None else format(r, '.3f')} |"
        )
    dec = payload.get("decode") or []
    if dec:
        # Two levels, both named. The tick columns are what the decode loop
        # turns over; the request column is tokens over the wall clock of a
        # whole stream and therefore carries prefill and queue with it. They
        # differ by more than a factor of two, so an unlabelled "decode tok/s"
        # column is not a number anybody can use.
        #
        # The column labels stay verbatim: test_s12_log_analyse.py asserts
        # "tok/s (Tick)" and "tok/s (Anfrage)", and "ms/Token" is the name
        # s12 defines for the whole battery (s14 re-reads it bit-for-bit).
        lines += [
            "",
            "| arm | bs | tok/s (Tick) | ms/Token (Tick) | accept (Tick) "
            "| Ticks | tok/s (Anfrage) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for d in dec[-8:]:
            lines.append(
                f"| {d.get('arm')} | {d.get('batch')} "
                f"| {_num(d.get('tick_gen_tok_s_median'), 1)} "
                f"| {_num(d.get('tick_ms_pro_token'), 2)} "
                f"| {_num(d.get('tick_accept_len_median'), 2)} "
                f"| {_num(d.get('tick_ticks_gewertet'), 0)} "
                f"| {_num(d.get('decode_tok_s'), 1)} |"
            )
    return "\n".join(lines) + "\n"


def mode_summarize(args) -> int:
    os.makedirs(args.step_dir, exist_ok=True)
    plan = [int(x) for x in str(args.sessions_plan).replace(",", " ").split()]
    payload = zusammenfassen(args.step_dir, args.tol_pct, plan)
    with open(os.path.join(args.step_dir, "prefill_kurve.json"), "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    text = tabelle(payload)
    with open(os.path.join(args.step_dir, "zwischentabelle.md"), "w") as f:
        f.write(text)
    print(text, end="")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("messen", "zusammenfassen"), required=True)
    ap.add_argument("--port", type=int, default=30030)
    # Free string, not `choices=ARME`. s12 itself only ever passes the two
    # names in ARME, and its summary still keys off ARME -- but the same
    # measurement driver is what #293 step 2 (s13) points at seven arms with,
    # and a closed choice list there would mean a second copy of this file.
    ap.add_argument("--arm", default="bar1")
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--folge", type=int, default=0)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--step-dir", default=".")
    ap.add_argument("--point-seconds", type=float, default=15.0)
    ap.add_argument("--warmup-seconds", type=float, default=8.0)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--with-decode", type=int, default=1)
    ap.add_argument(
        "--floor-draws",
        type=int,
        default=int(os.environ.get("S12_FLOOR_DRAWS", "1")),
        help="Number of A-vs-A floor draws of this arm, run BACK TO BACK "
        "inside one invocation after one discarded warm-up draw. 1 (the "
        "default) is the single measured point every earlier step took. "
        "Draw n is filed as <arm>_floorPn. Repeating the whole invocation "
        "instead puts ~50 s of idle between the draws, and #475 SS6 measured "
        "what that does: a monotone clock ramp reported as a 13 % noise "
        "floor where the same instrument reports 3 %.",
    )
    ap.add_argument(
        "--decode-batches",
        default="",
        help="Comma-separated decode batch sizes; empty keeps the default 1,16.",
    )
    ap.add_argument(
        "--server-log",
        default="",
        help="path of the server log ON THE HOST -- accept len and the decode "
        "tick rate come from there, neither is available at request level",
    )
    ap.add_argument(
        "--tol-pct",
        type=float,
        default=float(os.environ.get("S12_BASELINE_TOL_PCT", "5")),
    )
    ap.add_argument(
        "--sessions-plan",
        default=os.environ.get("S12_SESSIONS", "1 4 8 16"),
        help="which session counts the run is SUPPOSED to walk -- the check "
        "measures completeness against this",
    )
    args = ap.parse_args()
    if args.point_seconds > 60:
        print("point-seconds > 60 s -- against the time-box rule", file=sys.stderr)
        return 2
    if args.mode == "messen":
        return mode_measure(args)
    return mode_summarize(args)


if __name__ == "__main__":
    sys.exit(main())
