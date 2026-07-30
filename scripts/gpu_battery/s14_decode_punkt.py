#!/usr/bin/env python3
"""s14 -- one DECODE point against a live server, task #294.

Why this exists next to s12_prefill_kurve.py. That file measures a prefill
curve and carries a decode point as a by-product: it fires ``batch`` streaming
chat requests with ``max_tokens: 256``, lets them run for the point's seconds
and reads whatever ticks the scheduler happened to log. For a prefill window
that is enough. For a decode verdict it is not, and the 2026-07-30 run shows
every reason why:

* THREE TICKS PER POINT. At ``decode_log_interval`` 40 a 15 s window at bs=16
  produces three log lines, and the batch size is not the same in all of them:
  the nccl arms have one tick at #running-req 15 and two at 16, because
  requests finish inside the window. ``decode_tick_aggregat`` buckets on exact
  ``running_req``, so the good tick of an arm can land in the 15 bucket and its
  two ramp-contaminated ticks in the 16 bucket. That is a bucketing artefact,
  not a transport difference, and with three samples nothing separates the two.
* NO STEADY WORKING POINT. The requests prefill, ramp and finish inside the
  measured seconds, so the window contains the transition and not the loop.
* NO CONTEXT. The prosa prompt is ~50 tokens, so decode runs at a KV length no
  real workload has.

This file measures the decode loop instead:

* THE WORKING POINT IS PREFILLED, not grown into. A shared prefix of
  ``--context-tokens`` tokens is warmed once, then ``bs`` streams start on it;
  the radix cache serves the prefix, so every stream is in decode within
  seconds and all of them sit at the same KV length.
* NOTHING FINISHES INSIDE THE WINDOW. ``ignore_eos`` plus a max_new_tokens far
  above what the window can produce keeps ``#running-req`` pinned at ``bs``, so
  every tick in the window belongs to the same batch size.
* THE WINDOW IS CUT OUT OF THE MIDDLE. Ramp first, then t0, then the measured
  seconds, then t1; the first and last tick inside it are dropped as partial
  log intervals.
* BOTH LEVELS ARE KEPT AND LABELLED. The tick level -- the scheduler's own
  ``Decode batch`` line, whose ``accept len`` is ``spec_num_accept_tokens /
  spec_num_forward_ct`` over one log interval and whose ``gen throughput`` is
  that interval's token rate -- is the primary one, because it is windowed by
  construction. Note which field that is: ``accept len``, not the EMA the same
  line can also carry. The client level (tokens counted out of ``meta_info``
  between the two marks) is an independent second opinion on the rate.
* ACCEPT HAS A THIRD SOURCE, and it needs one. ``meta_info`` carries its spec
  block only on the FINAL chunk of a request -- mid-stream there is no
  ``spec_accept_length`` and no ``spec_verify_ct`` -- so a windowed accept
  cannot be built from the streams. One bounded probe request runs after the
  batch is released -- at bs=1, because the recipe's max_running_requests would
  otherwise queue it behind a full batch -- and its ``spec_accept_length`` is a
  cross-check that meta_info and the tick line agree, not the point's accept.

ON THE NAME ``ms/Verify``. It is kept bit-for-bit as s12 defines it,
``accept_len * 1000 / gen_tok_s``, so the numbers of this window can be laid
next to the numbers of the prefill window. ``gen_tok_s`` is the WHOLE batch's
rate, so this quantity is the verify step time PER REQUEST, not the wall time
of a step; the wall time is ``bs`` times larger and is reported separately as
``ms_pro_schritt``.

Standard library only: this runs on the PVE host, outside any venv.
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

from s12_log_analyse import im_fenster, parse_decode  # noqa: E402  one parser

KIND = "bar1_decode_verif"
SCHEMA_VERSION = 1

# The filler vocabulary is fixed and the RNG is seeded from a constant, so the
# prompt is byte-identical for every arm, every round and every batch size. The
# content axis is not what this window varies.
_WORDS = (
    "die messung zeigt einen deutlichen unterschied zwischen der erwartung und "
    "dem befund auf dieser maschine denn der weg ueber den hauptspeicher kostet "
    "zeit die niemand sieht solange die karten wenig zu tun haben und erst unter "
    "last wird aus dem umweg ein deckel der sich in jeder zahl niederschlaegt"
).split()

PREFIX_SEED = "s14-decode-verif"

# Ceiling for the streams' max_new_tokens, and a ceiling is all it is: what
# actually gets requested is the smaller of this and what the context leaves
# free. Nothing may finish inside the measured seconds -- a finished request
# changes #running-req and every tick of that interval becomes a different
# working point -- and at the fastest rate this rig reaches (bs=1, ~200 tok/s)
# a whole point including its ramp cannot produce 20000 tokens.
#
# It is a MINIMUM as much as a maximum. Asking for more than the context has
# room for is not clamped by the server: /generate answers 200 and streams a
# single `data: {"error": ...}` object with code 400, which a parser looking
# for meta_info drops without a word. The first attempt of this window spent
# three points that way -- sixteen streams, no chunks, no exception, an empty
# measurement. Hence both the derived cap below and the explicit error branch
# in Stream.run.
MAX_NEW_TOKENS_CAP = 20000
MAX_NEW_TOKENS_MIN = 2000
# What the boot recipe pins. Overridden by the server's own value when it can
# be read, so a recipe change cannot silently invalidate the cap.
CONTEXT_TOKENS_DEFAULT = 32768
# Accept probe: one bounded request AFTER the measured window, whose final
# meta_info is the only place spec_accept_length appears. Mid-stream chunks do
# not carry the spec block -- it is attached when the request finishes.
ACCEPT_PROBE_TOKENS = 128


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
        return json.loads(resp.read().decode(errors="replace"))


def _prompt(context_tokens: int, slot: int) -> str:
    """Shared prefix first, per-slot tail last.

    The prefix is what the radix cache serves, so it has to be identical across
    slots AND come first. The tail only has to make the sequences distinct so
    the batch is not one deduplicated request; it is short enough that its
    prefill costs nothing.
    """
    rng = random.Random(PREFIX_SEED)
    words = int(context_tokens / 1.3)
    body = " ".join(rng.choice(_WORDS) for _ in range(words))
    return f"{body} [slot {slot}] "


def _metrics_snapshot(port: int) -> dict:
    """The handful of gauges worth having at a window boundary."""
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
        return {"metrics_fehler": f"{type(exc).__name__}: {exc}"}
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


def _server_context_tokens(port: int, vorgabe: int) -> int:
    """The server's own context length, or the recipe's value.

    Read rather than assumed because the cap on max_new_tokens hangs off it,
    and a cap that is too large produces an empty measurement (see above).
    """
    for pfad in ("/get_server_info", "/get_model_info"):
        try:
            body = json.loads(_get(port, pfad, timeout=20))
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            continue
        stapel = [body]
        while stapel:
            knoten = stapel.pop()
            if isinstance(knoten, dict):
                wert = knoten.get("context_length") or knoten.get("max_context_length")
                if isinstance(wert, int) and wert > 0:
                    return wert
                stapel.extend(v for v in knoten.values() if isinstance(v, (dict, list)))
            elif isinstance(knoten, list):
                stapel.extend(v for v in knoten if isinstance(v, (dict, list)))
    return vorgabe


class Stream:
    """One streaming /generate request, sampled per chunk.

    The native endpoint is used rather than the chat one because ``meta_info``
    rides on every chunk here without an opt-in flag. What it does NOT carry
    mid-stream is the spec block: ``spec_accept_length`` and ``spec_verify_ct``
    are attached when a request FINISHES, so the accept length of the window
    comes from the scheduler's own tick line and the accept probe, not from
    here. What these chunks do give is ``completion_tokens``, and that is what
    makes a windowed client-side token count possible at all.
    """

    def __init__(
        self, port: int, prompt: str, stop: threading.Event, max_new: int
    ) -> None:
        self.port = port
        self.prompt = prompt
        self.stop = stop
        self.max_new = max_new
        self.samples: list = []  # (monotonic, completion_tokens, verify_ct)
        self.error: str | None = None
        self.meta_seen = False

    def run(self) -> None:
        payload = {
            "text": self.prompt,
            "sampling_params": {
                "max_new_tokens": self.max_new,
                "temperature": 0,
                "ignore_eos": True,
            },
            "stream": True,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                for raw in resp:
                    if self.stop.is_set():
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
                    # An error arrives as a normal 200 chunk with no meta_info.
                    # Silently skipping it is what turned a rejected request
                    # into an empty measurement, so it ends the stream loudly.
                    if isinstance(obj.get("error"), dict):
                        self.error = str(obj["error"].get("message"))[:300]
                        break
                    meta = obj.get("meta_info") or {}
                    if not meta:
                        continue
                    self.meta_seen = True
                    self.samples.append(
                        (
                            time.monotonic(),
                            meta.get("completion_tokens"),
                            meta.get("spec_verify_ct"),
                        )
                    )
        except (urllib.error.URLError, OSError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def between(self, t0: float, t1: float) -> dict:
        """Token and verify deltas between the two window marks.

        The sample nearest INSIDE each boundary is taken, never the nearest
        overall: a sample from before t0 would charge the window with tokens
        the ramp produced.
        """
        inner = [s for s in self.samples if t0 <= s[0] <= t1]
        if len(inner) < 2:
            return {"n": len(inner)}
        first, last = inner[0], inner[-1]
        out = {"n": len(inner), "spanne_s": last[0] - first[0]}
        if isinstance(first[1], int) and isinstance(last[1], int):
            out["d_tokens"] = last[1] - first[1]
        if isinstance(first[2], int) and isinstance(last[2], int):
            out["d_verify_ct"] = last[2] - first[2]
        if out.get("d_tokens") and out.get("d_verify_ct"):
            out["accept_len"] = out["d_tokens"] / out["d_verify_ct"]
        return out


def _tick_aggregat(ticks: list, bs: int, rand_verwerfen: int = 1) -> dict:
    """Per-tick decode metrics of ONE window, at exactly this batch size.

    Two differences to ``decode_tick_aggregat`` in s12, and both are what makes
    a decode verdict possible:

    * THE EDGE TICKS ARE DROPPED. The first log line inside the window covers a
      log interval that started before t0 and the last one covers an interval
      that is cut off by t1; both are partial and both are systematically
      slower. With three ticks per point that was half the sample.
    * TICKS AT THE WRONG BATCH SIZE ARE COUNTED, NOT SILENTLY DROPPED. A window
      that contains any of them did not hold its working point, and that has to
      show up in the point rather than in a quietly smaller sample.
    """
    passend = [t for t in ticks if t["running_req"] == bs]
    fremd = len(ticks) - len(passend)
    kern = passend[rand_verwerfen:-rand_verwerfen] if rand_verwerfen else passend
    if len(kern) < 2:
        kern = passend
    if not kern:
        return {"ticks_fenster": len(ticks), "ticks_bs": 0, "ticks_gewertet": 0}
    rate = [t["gen_tok_s"] for t in kern]
    accept = [t["accept_len"] for t in kern]
    rate_med = statistics.median(rate)
    accept_med = statistics.median(accept)
    out = {
        "ticks_fenster": len(ticks),
        "ticks_bs": len(passend),
        "ticks_fremde_bs": fremd,
        "ticks_gewertet": len(kern),
        "gen_tok_s_median": rate_med,
        "gen_tok_s_min": min(rate),
        "gen_tok_s_max": max(rate),
        "accept_len_median": accept_med,
        "cuda_graph": all(t["cuda_graph"] for t in kern),
    }
    if len(rate) > 2:
        out["gen_tok_s_stdev"] = statistics.stdev(rate)
    if rate_med:
        out["ms_pro_token"] = 1000.0 / rate_med
        out["ms_pro_verify"] = 1000.0 / rate_med * accept_med
        # The wall clock of one verify step. ``ms_pro_verify`` is that divided
        # by bs, because the scheduler's gen throughput counts the whole batch;
        # both are kept so neither can be mistaken for the other.
        out["ms_pro_schritt"] = 1000.0 / rate_med * accept_med * bs
    return out


def _ernte_ticks(server_log: str, von: float, bis: float, bs: int) -> dict:
    if not server_log or not os.path.exists(server_log):
        return {"tick_fehler": "kein Serverlog", "tick_quelle": server_log or None}
    try:
        with open(server_log, errors="replace") as f:
            ticks = parse_decode(f)
    except OSError as exc:
        return {"tick_fehler": f"{type(exc).__name__}: {exc}"}
    fenster = im_fenster(ticks, von, bis)
    out = {"tick_quelle": server_log}
    out.update({f"tick_{k}": v for k, v in _tick_aggregat(fenster, bs).items()})
    return out


def messe_punkt(args) -> dict:
    stop = threading.Event()
    prompt0 = _prompt(args.context_tokens, 0)

    # Warm the shared prefix into the radix cache. Without it the bs streams
    # all prefill the same 2048 tokens at once and the ramp is dominated by a
    # prefill burst that has nothing to do with the decode loop.
    warm = {}
    try:
        warm = _post(
            args.port,
            "/generate",
            {
                "text": prompt0,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            },
            timeout=300,
        )
        warm = (warm.get("meta_info") or {}) if isinstance(warm, dict) else {}
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        warm = {"fehler": f"{type(exc).__name__}: {exc}"}

    # The cap the streams run under, derived from what the context really has
    # left after the prompt. 512 tokens of margin cover the difference between
    # the warm request's prompt and a slot prompt's own tail.
    ctx = _server_context_tokens(args.port, args.model_context_tokens)
    prompt_tokens = warm.get("prompt_tokens") or args.context_tokens
    max_new = min(MAX_NEW_TOKENS_CAP, max(MAX_NEW_TOKENS_MIN, ctx - prompt_tokens - 512))

    streams = [
        Stream(args.port, _prompt(args.context_tokens, i), stop, max_new)
        for i in range(args.bs)
    ]
    threads = [threading.Thread(target=s.run, daemon=True) for s in streams]
    for t in threads:
        t.start()

    # Ramp: bounded wait until every stream is really in decode, then a fixed
    # settle. Both halves are needed -- the first says the batch is complete,
    # the second lets the clock rates settle before the window opens.
    ramp_deadline = time.monotonic() + args.ramp_seconds + 60
    while time.monotonic() < ramp_deadline:
        if all(len(s.samples) >= 25 or s.error for s in streams):
            break
        time.sleep(0.5)
    ramp_ok = all(len(s.samples) >= 25 for s in streams)
    time.sleep(args.ramp_seconds)

    m0 = _metrics_snapshot(args.port)
    t0_mono, t0_wall = time.monotonic(), time.time()
    time.sleep(args.window_seconds)
    t1_mono, t1_wall = time.monotonic(), time.time()
    m1 = _metrics_snapshot(args.port)

    stop.set()
    for t in threads:
        t.join(timeout=30)

    # The accept probe, and it runs AFTER the batch is gone, not alongside it.
    # Alongside was the first design and it deadlocked: the recipe pins
    # --max-running-requests 16, so at bs=16 the batch is full and an extra
    # request sits in the QUEUE until something finishes -- which nothing does,
    # because the streams are only released after the probe returns. py-spy put
    # the driver in _post at that exact line
    # (befunde/pyspy-driver-probe-hang.txt).
    #
    # The price is that this number is an accept length at bs=1, not at the
    # point's batch size, and it is named that way. The accept OF THE WINDOW
    # comes from the scheduler tick line, which measures it at the working
    # point by construction; the probe only says whether meta_info and the tick
    # line tell the same story about the same server.
    time.sleep(2.0)
    probe: dict = {}
    try:
        antwort = _post(
            args.port,
            "/generate",
            {
                "text": _prompt(args.context_tokens, 9999),
                "sampling_params": {
                    "max_new_tokens": ACCEPT_PROBE_TOKENS,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            },
            timeout=120,
        )
        meta = (antwort.get("meta_info") or {}) if isinstance(antwort, dict) else {}
        probe = {
            k: meta.get(k)
            for k in (
                "spec_accept_length",
                "spec_accept_rate",
                "spec_verify_ct",
                "completion_tokens",
            )
        }
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        probe = {"fehler": f"{type(exc).__name__}: {exc}"}

    je_strom = [s.between(t0_mono, t1_mono) for s in streams]
    tokens = sum(d.get("d_tokens") or 0 for d in je_strom)
    verify = sum(d.get("d_verify_ct") or 0 for d in je_strom)
    accepts = [d["accept_len"] for d in je_strom if d.get("accept_len")]
    spanne = t1_mono - t0_mono

    punkt = {
        "kind": KIND,
        "schema": SCHEMA_VERSION,
        "folge": args.folge,
        "arm": args.arm,
        "bs": args.bs,
        "zeit": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "context_tokens_ziel": args.context_tokens,
        "prompt_tokens": warm.get("prompt_tokens"),
        "cached_tokens_warm": warm.get("cached_tokens"),
        "server_context_tokens": ctx,
        "max_new_tokens": max_new,
        "ramp_seconds": args.ramp_seconds,
        "ramp_vollstaendig": ramp_ok,
        "window_seconds": args.window_seconds,
        "fenster_spanne_s": spanne,
        "fenster_von": t0_wall,
        "fenster_bis": t1_wall,
        "metriken_vor": m0,
        "metriken_nach": m1,
        # The client level: tokens counted out of meta_info between the two
        # marks. Independent of the scheduler's own accounting and of the log.
        "klient_tokens": tokens,
        "klient_verify_ct": verify,
        "klient_tok_s": (tokens / spanne) if spanne > 0 and tokens else None,
        "klient_accept_len_median": statistics.median(accepts) if accepts else None,
        "klient_accept_len_gesamt": (tokens / verify) if verify else None,
        "klient_stroeme": len([d for d in je_strom if d.get("n", 0) >= 2]),
        "probe_accept_bs1": probe,
        "meta_info_getragen": all(s.meta_seen for s in streams),
        "fehler": [s.error for s in streams if s.error][:3],
        "je_strom": je_strom,
    }
    punkt.update(_ernte_ticks(args.server_log, t0_wall, t1_wall, args.bs))

    # Let the batch drain before the next point. A point that starts while the
    # previous streams are still being torn down measures the teardown.
    time.sleep(args.drain_seconds)
    return punkt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--bs", type=int, required=True)
    p.add_argument("--folge", type=int, default=0)
    p.add_argument("--context-tokens", type=int, default=2048)
    p.add_argument("--model-context-tokens", type=int, default=CONTEXT_TOKENS_DEFAULT)
    p.add_argument("--ramp-seconds", type=float, default=6.0)
    p.add_argument("--window-seconds", type=float, default=15.0)
    p.add_argument("--drain-seconds", type=float, default=5.0)
    p.add_argument("--server-log", default="")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    punkt = messe_punkt(args)
    with open(os.path.join(args.out_dir, "decode_punkte.jsonl"), "a") as f:
        f.write(json.dumps(punkt) + "\n")

    print(
        f"punkt {args.arm}/bs={args.bs}: "
        f"tick {punkt.get('tick_gen_tok_s_median')} tok/s, "
        f"accept {punkt.get('tick_accept_len_median')}, "
        f"ms/Verify {punkt.get('tick_ms_pro_verify')}, "
        f"ticks {punkt.get('tick_ticks_gewertet')}/{punkt.get('tick_ticks_bs')} "
        f"(fremde bs: {punkt.get('tick_ticks_fremde_bs')}), "
        f"klient {punkt.get('klient_tok_s')} tok/s, "
        f"probe accept {(punkt.get('probe_accept_bs1') or {}).get('spec_accept_length')}"
    )
    # A point without a tick rate is not a point: the window held no usable
    # scheduler line, and reporting the client rate alone would hide that.
    if punkt.get("tick_gen_tok_s_median") is None:
        for err in punkt.get("fehler") or []:
            print(f"  Fehler: {err}", file=sys.stderr)
        print(f"  tick_fehler: {punkt.get('tick_fehler')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
