# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Cross-stack proof for the Opus uplink: Chromium encodes, the server decodes.

**What this closes.** Every other test of this change lives on one side of the
wire. The hermetic suite drives the server with packets Python produced; the
client tests read the shipped HTML as text. Neither can answer the question the
change actually turns on -- whether the bytes a BROWSER's ``AudioEncoder``
emits are the bytes ``OpusCodec.decode`` reads. That is a framing question
between two independent implementations of Opus, and the failure mode if they
disagree is not an exception: it is a decoder that resynchronises on garbage
and a recognizer that transcribes noise as words.

So this drives a real Chromium with the SAME encoder configuration the client
ships, takes the raw ``EncodedAudioChunk`` payloads out of the page, and feeds
them to the same :class:`OpusCodec` the server runs. It asserts the pitch
survives, the duration survives, and the bitrate is what the link budget was
planned around.

**Not in the hermetic suite, on purpose:** it needs a browser and a local HTTP
origin (WebCodecs is secure-context only, so ``file://`` and ``about:blank``
cannot see ``AudioEncoder`` at all -- which is itself worth knowing, because a
deployment served over plain HTTP would silently fall back to PCM16).

    PYTHONPATH=<repo>/python /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/probe_opus_uplink.py
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import http.server
import json
import socket
import sys
import threading
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from sglang.srt.translator.audio import (  # noqa: E402
    PIPELINE_SAMPLE_RATE,
    OpusCodec,
    Pcm16Codec,
    available_codecs,
    resample,
)
from sglang.srt.translator.backends import AudioChunk  # noqa: E402

#: Mirrors `opusEncoderConfig()` in client/index.html. Kept as a literal rather
#: than parsed out of the page: a probe that reads its own expectation from the
#: thing under test proves only that the file is self-consistent.
CLIENT_CONFIG = {
    "codec": "opus",
    "sampleRate": PIPELINE_SAMPLE_RATE,
    "numberOfChannels": 1,
    "bitrate": 24000,
    "opus": {"frameDuration": 20000},
}
FRAME = PIPELINE_SAMPLE_RATE * 20 // 1000
TONE_HZ = 220.0

PAGE = "<!doctype html><meta charset=utf-8><title>opus uplink probe</title>"

ENCODE_JS = """
async (cfg) => {
  const RATE = cfg.sampleRate, FRAME = cfg.frame, SECONDS = cfg.seconds;
  const config = cfg.config;
  const supported = (await AudioEncoder.isConfigSupported(config)).supported;
  if (!supported) return {supported: false};
  const packets = [];
  const errors = [];
  const encoder = new AudioEncoder({
    output: (chunk) => {
      const buf = new Uint8Array(chunk.byteLength);
      chunk.copyTo(buf);
      let s = "";
      for (let i = 0; i < buf.length; i++) s += String.fromCharCode(buf[i]);
      packets.push(btoa(s));
    },
    error: (err) => errors.push(String(err && err.message || err)),
  });
  encoder.configure(config);
  let ts = 0;
  const frames = Math.round(SECONDS * RATE / FRAME);
  for (let f = 0; f < frames; f++) {
    const s = new Float32Array(FRAME);
    for (let i = 0; i < FRAME; i++) {
      const t = (f * FRAME + i) / RATE;
      s[i] = 0.4 * Math.sin(2 * Math.PI * cfg.hz * t);
    }
    encoder.encode(new AudioData({
      format: "f32", sampleRate: RATE, numberOfFrames: FRAME,
      numberOfChannels: 1, timestamp: ts, data: s,
    }));
    ts += Math.round(FRAME * 1e6 / RATE);
  }
  await encoder.flush();
  encoder.close();
  return {supported: true, packets: packets, errors: errors,
          secure: window.isSecureContext, frames_fed: frames};
}
"""


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D102
        pass


def _serve(directory: Path):
    """A localhost origin, because WebCodecs is secure-context only."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    handler = functools.partial(_Quiet, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{port}/index.html"


async def _encode_in_browser(url: str, seconds: float) -> dict:
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        result = await page.evaluate(
            ENCODE_JS,
            {
                "config": CLIENT_CONFIG,
                "sampleRate": PIPELINE_SAMPLE_RATE,
                "frame": FRAME,
                "seconds": seconds,
                "hz": TONE_HZ,
            },
        )
        await browser.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()

    if "opus" not in available_codecs():
        print("[probe] FAIL: this process has no Opus decoder (PyAV missing)")
        return 2

    tmp = Path("/tmp/translator-opus-probe")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "index.html").write_text(PAGE, encoding="utf-8")
    server, url = _serve(tmp)
    try:
        result = asyncio.run(_encode_in_browser(url, args.seconds))
    finally:
        server.shutdown()

    if not result.get("supported"):
        print("[probe] FAIL: this browser refused the client's encoder config")
        return 2
    if result.get("errors"):
        print(f"[probe] FAIL: encoder errors {result['errors']}")
        return 2

    packets = [base64.b64decode(p) for p in result["packets"]]
    total = sum(len(p) for p in packets)
    print(f"[probe] secure context: {result['secure']}")
    print(f"[probe] fed {result['frames_fed']} frames of {FRAME} samples "
          f"at {PIPELINE_SAMPLE_RATE} Hz")
    print(f"[probe] browser produced {len(packets)} raw Opus packets, "
          f"{total} bytes -> {total * 8 / args.seconds / 1000:.1f} kbit/s")

    # THE ASSERTION THIS PROBE EXISTS FOR: the server's own decoder, unchanged,
    # reading what the browser wrote. No container was stripped on the way.
    codec = OpusCodec()
    parts = [codec.decode(p).samples for p in packets]
    decoded = AudioChunk(np.concatenate(parts), codec.sample_rate)
    chunk = resample(decoded, PIPELINE_SAMPLE_RATE)

    spectrum = np.abs(np.fft.rfft(chunk.samples))
    freqs = np.fft.rfftfreq(len(chunk.samples), 1.0 / PIPELINE_SAMPLE_RATE)
    dominant = float(freqs[int(np.argmax(spectrum))])
    duration = len(chunk.samples) / PIPELINE_SAMPLE_RATE
    peak = float(np.abs(chunk.samples).max())

    pcm_bytes = sum(
        len(f) for f in Pcm16Codec().encode(
            AudioChunk(np.zeros(int(args.seconds * PIPELINE_SAMPLE_RATE),
                                dtype=np.float32), PIPELINE_SAMPLE_RATE))
    )
    print(f"[probe] decoded {duration:.2f} s at {codec.sample_rate} Hz, "
          f"dominant {dominant:.1f} Hz (fed {TONE_HZ:.0f} Hz), peak {peak:.3f}")
    print(f"[probe] PCM16 would have been {pcm_bytes} bytes for the same "
          f"audio -- a factor of {pcm_bytes / max(total, 1):.1f}")

    ok = True
    for label, condition in (
        ("pitch survives the cross-stack trip",
         abs(dominant - TONE_HZ) < 15.0),
        ("duration survives (encoder delay aside)",
         abs(duration - args.seconds) < 0.10),
        ("the signal is not silence", peak > 0.1),
        ("the bitrate is inside the link budget",
         total * 8 / args.seconds < 40000),
        ("Opus is at least eight times cheaper than PCM16",
         pcm_bytes > 8 * total),
    ):
        print(f"[probe] {'ok  ' if condition else 'FAIL'} {label}")
        ok = ok and condition

    print("[probe]", json.dumps({
        "packets": len(packets), "bytes": total,
        "kbit_per_second": round(total * 8 / args.seconds / 1000, 1),
        "dominant_hz": round(dominant, 1), "seconds": round(duration, 3),
    }))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
