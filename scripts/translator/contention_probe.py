# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""A second conversation, to answer whether the tenant slows down under load.

DESIGN §17.8.1 left one hypothesis standing: the turn that "produced nothing"
during soak #1 coincided with the user driving a second conversation from a
phone, and soak #2 -- identical idle gaps, no second conversation -- passed
every turn. Two runs differing in load is a coincidence, not a measurement.

This probe is the deliberate version of that accident. It opens its OWN
session over the public front door and drives turns at a fixed cadence, and it
reads the same per-stage numbers the server already publishes on ``turn.done``
(``Stopwatch``), so a run of ``client_gate.py`` alone and a run of the gate
WITH this probe alongside are comparable stage by stage:

    # alone
    scripts/translator/client_gate.py --turns 6 --gap-s 30 --json-out alone.json
    # under load: start the probe first, then the gate, and compare
    scripts/translator/contention_probe.py --turns 12 --gap-s 20 &
    scripts/translator/client_gate.py --turns 6 --gap-s 30 --json-out loaded.json

It deliberately does NOT score the audio: this is a load generator and a
stopwatch, and loading an ASR to grade the reply would put the harness itself
on the critical path it is trying to measure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

DEFAULT_URL = "wss://efeu.ddnss.de/translate"
DEFAULT_CLIP = Path(
    "/spinning/llm_stuff/translator-models/xtts-v2/samples/de_sample.wav"
)
STAGE_KEYS = (
    "asr_ms",
    "embed_ms",
    "mt_first_token_ms",
    "mt_total_ms",
    "tts_first_audio_ms",
    "tts_wait_ms",
    "tts_total_ms",
    "first_audio_ms",
    "total_ms",
)


def load_clip(clip: Path):
    import soundfile as sf

    samples, rate = sf.read(str(clip), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, rate


def median(values: list) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


async def drive(args) -> int:
    import websockets

    from sglang.srt.translator.audio import negotiate_codec
    from sglang.srt.translator.backends import AudioChunk

    samples, rate = load_clip(args.clip)
    codec = negotiate_codec(["pcm16"])
    endpoint = f"{args.url.rstrip('/')}/api/translator/stream"
    chunk = max(1, int(args.chunk_ms * rate / 1000))
    turns: list = []

    async with websockets.connect(endpoint, max_size=None) as ws:
        await ws.send(
            json.dumps(
                {
                    "kind": "hello",
                    "codecs": ["pcm16"],
                    "participants": args.participants.split(","),
                }
            )
        )
        ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=args.timeout_s))
        if ready.get("kind") != "ready":
            print(f"[probe] expected a ready frame, got {ready!r}")
            return 1
        print(f"[probe] {args.label}: session {ready['session_id']}")

        for index in range(args.turns):
            if index:
                await asyncio.sleep(args.gap_s)
            # Push the utterance in client-sized pieces: the segmenter's VAD
            # decides where a turn ends, and one blob would bypass the path a
            # phone actually takes.
            for start in range(0, len(samples), chunk):
                piece = AudioChunk(samples[start : start + chunk], rate)
                for frame in codec.encode(piece):
                    await ws.send(frame)
            released = time.monotonic()
            await ws.send(json.dumps({"kind": "release"}))

            first_audio_at = None
            stages: list = []
            deadline = released + args.timeout_s
            while time.monotonic() < deadline:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue
                if isinstance(message, bytes):
                    if first_audio_at is None:
                        first_audio_at = time.monotonic()
                    continue
                event = json.loads(message)
                if event.get("kind") == "turn.done":
                    if event.get("timings"):
                        stages.append(event["timings"])
                    # An empty turn (nothing recognized) also ends here; stop
                    # waiting either way rather than burning the timeout.
                    break
            record = {
                "turn": index + 1,
                # Wall time as the OTHER side sees it -- transport and queueing
                # included, which is exactly what the server-side stopwatch
                # cannot see.
                "client_first_audio_s": (
                    None if first_audio_at is None
                    else round(first_audio_at - released, 2)
                ),
                "stages": stages,
            }
            turns.append(record)
            head = stages[0] if stages else {}
            print(
                f"[probe] {args.label} turn {index + 1}: "
                f"client first audio {record['client_first_audio_s']}s | "
                + " ".join(
                    f"{key[:-3]} {head.get(key, 0.0) / 1000.0:.2f}s"
                    for key in STAGE_KEYS
                )
            )

        # Release the slot instead of leaving it to the resume grace: a probe
        # that holds a session for two minutes after it finishes is the same
        # accumulation the collector exists to clean up, self-inflicted.
        await ws.send(json.dumps({"kind": "close"}))

    print()
    stages = [s for t in turns for s in t["stages"]]
    print(f"[probe] {args.label}: {len(turns)} turns, "
          f"{len(stages)} with server timings")
    for key in STAGE_KEYS:
        values = [s.get(key, 0.0) / 1000.0 for s in stages]
        if values:
            print(f"          {key[:-3]:<16} med {median(values):6.2f}s  "
                  f"min {min(values):6.2f}s  max {max(values):6.2f}s")
    client = [t["client_first_audio_s"] for t in turns
              if t["client_first_audio_s"] is not None]
    if client:
        print(f"          {'client first audio':<16} med {median(client):6.2f}s  "
              f"min {min(client):6.2f}s  max {max(client):6.2f}s")
    missing = [t["turn"] for t in turns if t["client_first_audio_s"] is None]
    if missing:
        print(f"[probe] {args.label}: NO AUDIO on turns {missing}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps({"label": args.label, "url": args.url, "turns": turns},
                       indent=2),
            encoding="utf-8",
        )
    return 1 if missing else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--gap-s", type=float, default=20.0)
    parser.add_argument("--participants", default="de,es")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--label", default="probe")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    return asyncio.run(drive(args))


if __name__ == "__main__":
    sys.exit(main())
