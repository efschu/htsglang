# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The whole tenant, through its own front door: German audio in, Spanish out.

Every other harness in this directory drives ONE stage in process. This one
touches nothing but the WebSocket a phone would use -- handshake, binary audio
frames, release, journal events, binary audio back -- so it exercises the
uvicorn stack, the codec negotiation, the session, the journal pump and all
four backends the way the client does, and nothing else.

That distinction is load-bearing here. This project has already shipped a
green hermetic suite over a WebSocket route that answered 404 in production,
because Starlette's TestClient implements the protocol in process and never
touches uvicorn (see ``launch.require_websocket_library``). A front-door test
is the only kind that could have caught it.

The verdict is not "no exception". The returned Spanish audio is transcribed
by the recognizer and scored against the translation the pipeline itself
produced, so a turn that returns plausible-sounding noise fails. The gate is
the same instrument and threshold as ``asr_roundtrip_gate.py``: WER <= 0.15.

    PYTHONPATH=<repo>/python /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/front_door_test.py --url ws://127.0.0.1:30800 \\
      --audio /spinning/llm_stuff/translator-models/xtts-v2/samples/de_sample.wav
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

ASR_LIB = Path("/spinning/llm_stuff/translator-models/asr-lib")
ASR_MODELS = Path("/spinning/llm_stuff/translator-models/asr-models")
DEFAULT_AUDIO = Path(
    "/spinning/llm_stuff/translator-models/xtts-v2/samples/de_sample.wav"
)
DEFAULT_THRESHOLD = 0.15


async def one_turn(args, samples: np.ndarray, rate: int) -> dict:
    """Drive one push-to-talk turn and return what came back."""
    import websockets

    from sglang.srt.translator.audio import negotiate_codec
    from sglang.srt.translator.backends import AudioChunk

    codec = negotiate_codec(["pcm16"])
    endpoint = f"{args.url.rstrip('/')}/api/translator/stream"
    events: list = []
    audio_frames: list = []
    started = time.monotonic()

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
            raise SystemExit(f"expected a ready frame, got {ready!r}")
        print(f"[door] session {ready['session_id']} codec {ready['codec']['name']}")
        matrix = ready.get("languages", {})
        print(
            f"[door] server offers {len(matrix.get('bidirectional', []))} "
            f"bidirectional languages "
            f"(asr {len(matrix.get('stages', {}).get('asr') or [])}, "
            f"tts {len(matrix.get('stages', {}).get('tts') or [])})"
        )

        # Push the utterance in client-sized chunks rather than one blob: the
        # segmenter's VAD is what decides where a turn ends, and handing it the
        # whole file at once would bypass the path a phone actually takes.
        #
        # `--repeats` exists for the clone path specifically. A speaker's
        # reference buffer fills from their OWN completed turns, so the first
        # utterance of a session can never be cloned -- it downgrades to a
        # preset voice by design. Speaking again in the same session is how a
        # real conversation reaches the clone path, and it is the only way to
        # exercise it from outside.
        chunk = max(1, int(args.chunk_ms * rate / 1000))
        for _ in range(max(1, args.repeats)):
            for start in range(0, len(samples), chunk):
                piece = AudioChunk(samples[start : start + chunk], rate)
                for frame in codec.encode(piece):
                    await ws.send(frame)
            # Push-to-talk release: end the turn without waiting for hangover.
            await ws.send(json.dumps({"kind": "release"}))

        # A 3 s clip segments into more than one turn, so frames are bound to
        # the turn that announced them. Scoring audio from one turn against
        # another turn's text would be an instrument that cannot tell a
        # correct pipeline from a scrambled one.
        deadline = time.monotonic() + args.timeout_s
        current_turn = None
        finished: list = []
        while time.monotonic() < deadline:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                # Idle is only an end condition once every pushed utterance has
                # come back. The tenant answers turns one at a time and a turn
                # costs seconds, so an idle window shorter than that would stop
                # the harness in the middle of the conversation it just started.
                if len(finished) >= max(1, args.repeats) and any(
                    t in finished for t, _ in audio_frames
                ):
                    break
                continue
            if isinstance(message, bytes):
                # One announcement can be followed by SEVERAL binary frames --
                # the codec decides the framing, not the event. The turn stays
                # current until the next JSON frame replaces it.
                audio_frames.append((current_turn, message))
                continue
            event = json.loads(message)
            events.append(event)
            current_turn = event.get("turn_id") if event.get("audio_follows") else None
            print(f"[door]   <- {event.get('kind')} {_summary(event)}")
            if event.get("kind") == "turn.done":
                finished.append(event.get("turn_id"))
        await ws.send(json.dumps({"kind": "close"}))

    # The LAST completed turn that carried audio: with --repeats that is the
    # one whose speaker already has a reference buffer, i.e. the clone path.
    scored = next(
        (t for t in reversed(finished) if any(turn == t for turn, _ in audio_frames)),
        None,
    )
    payload = b"".join(frame for turn, frame in audio_frames if turn == scored)
    return {
        "events": events,
        "turn_id": scored,
        "turns_completed": len(finished),
        "audio": codec.decode(payload) if payload else None,
        "wall_s": time.monotonic() - started,
    }


def _summary(event: dict) -> str:
    # The speaker event's reference bookkeeping is printed because it is the
    # only thing that says WHY a turn used a preset voice: the clone path opens
    # when a speaker's own admitted reference reaches the configured minimum,
    # and without these two numbers a downgrade looks like a failure rather
    # than a buffer that is still filling.
    if event.get("kind") == "turn.speaker":
        return (
            f"id={event.get('speaker_id')} sim={event.get('similarity')} "
            f"admitted={event.get('reference_admitted')} "
            f"reference={event.get('reference_seconds')}s"
        )
    for key in ("text", "translation", "reason", "state"):
        if key in event and isinstance(event[key], str):
            return f"{key}={event[key][:90]!r}"
    if "samples" in event:
        return f"{event['samples']} samples @ {event.get('sample_rate')} Hz"
    return ""


async def run(args) -> int:
    if str(ASR_LIB) not in sys.path:
        sys.path.append(str(ASR_LIB))

    import soundfile as sf

    from sglang.srt.translator.asr_backends import FasterWhisperAsr
    from sglang.srt.translator.backends import AudioChunk
    from sglang.srt.translator.scoring import word_error_rate

    samples, rate = sf.read(str(args.audio), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = np.ascontiguousarray(samples)
    print(f"[door] source {args.audio.name}: {len(samples) / rate:.2f}s @ {rate} Hz")

    turn = await one_turn(args, samples, rate)
    if turn["audio"] is None:
        print("[door] FAILED: the turn produced no audio at all")
        return 1

    audio = turn["audio"]
    duration = len(audio.samples) / audio.sample_rate
    peak = float(np.abs(audio.samples).max())
    print(
        f"[door] returned {duration:.2f}s of audio, peak {peak:.3f}, "
        f"turn wall time {turn['wall_s']:.2f}s"
    )
    if not np.isfinite(audio.samples).all() or peak < 1e-3:
        print("[door] FAILED: the returned audio is silence or non-finite")
        return 1

    # What the pipeline itself said it was going to say. Scoring the audio
    # against this rather than against a hand-written string keeps the gate
    # about SYNTHESIS -- a translation the operator dislikes is a different
    # question from audio that does not carry the words it was given.
    # Both event kinds carry their string under "text"; the KIND is what tells
    # the recognized source from the translation, and `partial` marks the
    # clause-by-clause deltas the tenant emits while it is still speaking.
    spoken = ""
    heard_source = ""
    voice_note = ""
    for event in turn["events"]:
        if event.get("turn_id") != turn["turn_id"]:
            continue
        if event.get("kind") == "turn.voice":
            voice_note = str(event.get("reason") or event.get("mode") or "")
        text = event.get("text")
        if not isinstance(text, str) or not text:
            continue
        if event.get("kind") == "turn.transcript":
            heard_source = text
        elif event.get("kind") == "turn.translation" and not event.get("partial"):
            spoken = text
    if not spoken:
        print("[door] FAILED: no translation text came back, so the returned "
              "audio cannot be scored against anything the pipeline promised")
        return 1
    print(f"[door] turn              : {turn['turn_id']} "
          f"(of {turn['turns_completed']} completed)")
    print(f"[door] recognized source : {heard_source!r}")
    print(f"[door] translation       : {spoken!r}")
    print(f"[door] voice             : {voice_note or 'clone'}")

    if args.out:
        sf.write(str(args.out), audio.samples, audio.sample_rate)
        print(f"[door] wrote {args.out}")

    asr = FasterWhisperAsr(
        model=args.asr_model,
        device=args.asr_device,
        compute_type=args.asr_compute_type,
        download_root=ASR_MODELS,
        beam_size=1,
    )
    transcript = await asr.transcribe(AudioChunk(audio.samples, audio.sample_rate))
    result = word_error_rate(spoken, transcript.text)
    print(f"[door] heard back        : {transcript.text!r}")
    print(f"[door] language          : {transcript.language} "
          f"({transcript.language_confidence:.3f})")
    print(
        f"[door] WER {result.rate:.3f} (sub {result.substitutions} "
        f"del {result.deletions} ins {result.insertions}, "
        f"ref {result.reference_words} words)"
    )

    # The can-fail control: the same gate run against the SOURCE audio, which
    # is the right words in the wrong language. If that passes, the instrument
    # is not discriminating and the number above means nothing.
    control = await asr.transcribe(AudioChunk(samples, rate))
    control_result = word_error_rate(spoken, control.text)
    print(
        f"[door] control (source audio vs the Spanish text): "
        f"WER {control_result.rate:.3f}"
    )
    if control_result.rate <= args.threshold:
        print("[door] THE GATE IS BROKEN: German audio scored as the Spanish "
              "translation. Fix the instrument before trusting the arm above.")
        return 1

    passed = result.rate <= args.threshold
    print(f"[door] threshold {args.threshold:.2f}: "
          f"{'PASS' if passed else 'FAIL'} at WER {result.rate:.3f}")
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:30800")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--participants", default="de,es")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="speak the clip this many times in ONE session; >1 reaches the "
             "clone path, which the first utterance of a session never can",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--asr-model", default="large-v3-turbo")
    parser.add_argument("--asr-device", default="cpu")
    parser.add_argument("--asr-compute-type", default="int8")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
