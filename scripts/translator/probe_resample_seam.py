# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Does per-frame scheduling survive the browser's resampler? (§466 click)

`probe_audio_onset.py` falsified all three original candidates for the click
the user hears at the start of speech: the talker's waveform rises naturally
over ~11.6 ms, the framing seams are not the outlier, and of 268 buffer
schedules exactly one started in the past -- the benign initialization.

That probe reads the samples the client is HANDED. What it cannot see, by
construction, is an artefact created after that point: the client builds one
``AudioBufferSource`` per 20 ms frame at the WIRE rate
(``createBuffer(1, float32.length, 16000)``) while the output context runs at
44100 (measured) or 48000 on a phone. Every 320-sample frame is therefore
resampled INDEPENDENTLY, and a resampler given 320 samples with no
neighbours has to invent its edges.

This is the falsifier for that, and it needs no server: render the SAME
signal twice in an ``OfflineAudioContext`` at the output rate --

  arm A: one source per 320-sample frame, scheduled back to back, exactly
         as the client does it today;
  arm B: one continuous buffer holding the whole signal.

Both arms carry identical input samples, so any difference in the rendered
output is manufactured by the per-frame path. A sine is used because a
discontinuity at a frame edge shows up as broadband splatter against an
otherwise single-frequency signal, which is unmistakable; the residual is
then localized against the frame grid to show WHERE it lives.

    /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/probe_resample_seam.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import numpy as np

#: Renders both arms and hands back the two rendered buffers. Everything
#: numeric is done in Python; the page only produces the audio.
RENDER_JS = """
async ({wireRate, outRate, frame, seconds, freq}) => {
  const n = Math.round(wireRate * seconds);
  const input = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    input[i] = 0.5 * Math.sin(2 * Math.PI * freq * i / wireRate);
  }
  const renderFrames = async () => {
    const ctx = new OfflineAudioContext(1, Math.ceil(outRate * seconds), outRate);
    let cursor = 0;
    for (let start = 0; start + frame <= n; start += frame) {
      const buf = ctx.createBuffer(1, frame, wireRate);
      buf.copyToChannel(input.slice(start, start + frame), 0);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      src.start(cursor);
      cursor += buf.duration;
    }
    const rendered = await ctx.startRendering();
    return Array.from(rendered.getChannelData(0));
  };
  const renderWhole = async () => {
    const ctx = new OfflineAudioContext(1, Math.ceil(outRate * seconds), outRate);
    const usable = Math.floor(n / frame) * frame;
    const buf = ctx.createBuffer(1, usable, wireRate);
    buf.copyToChannel(input.slice(0, usable), 0);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.start(0);
    const rendered = await ctx.startRendering();
    return Array.from(rendered.getChannelData(0));
  };
  // ARM C -- THE FIX, verified by the same construction. A stateful linear
  // resampler carries `prev` and a fractional read position ACROSS frames, so
  // the browser is only ever handed buffers that already sit at the context
  // rate and never has to resample a 320-sample island. Framed and whole must
  // then come out bit-identical: same algorithm, same input, only the
  // chunking differs. Comparing arm C against arm B would compare linear
  // interpolation against the browser's own resampler and measure the
  // ALGORITHM, which is not the question.
  const makeResampler = () => ({
    prev: 0.0, k: 0, consumed: 0, primed: false,
    // `inRate === outRate` is a pass-through, which is what keeps this
    // rate-agnostic: when the §19.5 ladder moves the wire to 48 kHz on a
    // 48 kHz context, this stops doing anything without being rebuilt.
    process(x, inRate, outRate) {
      if (inRate === outRate) return x;
      const ratio = inRate / outRate;
      if (!this.primed) { this.prev = x.length ? x[0] : 0.0; this.primed = true; }
      const out = [];
      // The read position is DERIVED from the global output index, never
      // accumulated. Adding `ratio` to a running position drifts, and the
      // drift differs between "one call of 16000 samples" and "50 calls of
      // 320" -- which showed up as an 0.086 peak at 48 kHz out, where the
      // ratio is 1/3 and the rounding is systematic. Deriving it means the
      // framed and whole paths compute the SAME position for the same output
      // sample, by construction rather than by luck.
      while (true) {
        const pos = this.k * ratio - this.consumed;
        const i = Math.floor(pos);
        if (i + 1 >= x.length) break;
        const f = pos - i;
        const s0 = (i < 0) ? this.prev : x[i];
        out.push(s0 * (1 - f) + x[i + 1] * f);
        this.k += 1;
      }
      this.consumed += x.length;
      if (x.length) this.prev = x[x.length - 1];
      return Float32Array.from(out);
    },
  });
  const renderResampled = async (chunked) => {
    const ctx = new OfflineAudioContext(1, Math.ceil(outRate * seconds), outRate);
    const rs = makeResampler();
    const usable = Math.floor(n / frame) * frame;
    const pieces = [];
    if (chunked) {
      for (let start = 0; start + frame <= n; start += frame) {
        pieces.push(rs.process(input.slice(start, start + frame), wireRate, outRate));
      }
    } else {
      pieces.push(rs.process(input.slice(0, usable), wireRate, outRate));
    }
    let cursor = 0;
    for (const piece of pieces) {
      if (!piece.length) continue;
      const buf = ctx.createBuffer(1, piece.length, outRate);
      buf.copyToChannel(piece, 0);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      src.start(cursor);
      cursor += buf.duration;
    }
    const rendered = await ctx.startRendering();
    return Array.from(rendered.getChannelData(0));
  };
  return {
    frames: await renderFrames(),
    whole: await renderWhole(),
    fixed_framed: await renderResampled(true),
    fixed_whole: await renderResampled(false),
  };
}
"""


async def run(args) -> int:
    from playwright.async_api import async_playwright

    params = {
        "wireRate": args.wire_rate,
        "outRate": args.out_rate,
        "frame": args.frame,
        "seconds": args.seconds,
        "freq": args.freq,
    }
    print(f"[seam] {args.wire_rate} Hz wire -> {args.out_rate} Hz context, "
          f"{args.frame}-sample frames ({args.frame / args.wire_rate * 1000:.0f} ms), "
          f"{args.freq} Hz tone")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page()
        await page.goto("about:blank")
        result = await page.evaluate(RENDER_JS, params)
        await browser.close()

    a = np.asarray(result["frames"], dtype=np.float64)
    b = np.asarray(result["whole"], dtype=np.float64)
    length = min(a.size, b.size)
    a, b = a[:length], b[:length]
    residual = a - b

    # Where the frame edges land in the OUTPUT domain.
    step = args.frame * args.out_rate / args.wire_rate
    edges = np.arange(step, length - 1, step).astype(int)
    guard = max(2, int(args.out_rate * 0.0005))       # +-0.5 ms around an edge
    near_edge = np.zeros(length, dtype=bool)
    for edge in edges:
        near_edge[max(0, edge - guard):min(length, edge + guard)] = True

    def rms(x):
        return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0

    signal_rms = rms(b)
    report = {
        "samples": int(length),
        "signal_rms": signal_rms,
        "residual_rms": rms(residual),
        "residual_peak": float(np.max(np.abs(residual))) if length else 0.0,
        "residual_rms_at_edges": rms(residual[near_edge]),
        "residual_rms_between_edges": rms(residual[~near_edge]),
        "edges": int(edges.size),
    }
    report["residual_vs_signal_db"] = (
        20 * np.log10(report["residual_rms"] / signal_rms)
        if report["residual_rms"] > 0 and signal_rms > 0 else float("-inf")
    )
    ratio = (report["residual_rms_at_edges"]
             / report["residual_rms_between_edges"]
             if report["residual_rms_between_edges"] > 0 else float("inf"))
    report["edge_concentration"] = ratio

    print(f"[seam] signal rms          {report['signal_rms']:.6f}")
    print(f"[seam] residual rms        {report['residual_rms']:.6f} "
          f"({report['residual_vs_signal_db']:.1f} dB below signal)")
    print(f"[seam] residual peak       {report['residual_peak']:.6f}")
    print(f"[seam] residual at edges   {report['residual_rms_at_edges']:.6f}")
    print(f"[seam] residual between    {report['residual_rms_between_edges']:.6f}")
    print(f"[seam] edge concentration  {ratio:.2f}x over {report['edges']} edges")

    # The verdict has to be a THRESHOLD, not an eyeball. Audible splatter
    # against a 0.5-amplitude tone needs to be within roughly 60 dB of it;
    # a residual 100 dB down is float noise and means the arms agree.
    proven = (report["residual_vs_signal_db"] > -60.0 and ratio > 2.0)
    print("[seam] verdict: "
          + ("PER-FRAME RESAMPLING MANUFACTURES ENERGY AT THE FRAME EDGES "
             "-- root proven" if proven else
             "the two arms agree; per-frame resampling is NOT the source"))
    # ARM C. The fix is verified against ITSELF, framed versus whole: same
    # resampler, same input, only the chunking differs. Anything above float
    # noise here is the fix failing to carry state across a frame boundary,
    # which is the entire defect it exists to remove.
    c_framed = np.asarray(result["fixed_framed"], dtype=np.float64)
    c_whole = np.asarray(result["fixed_whole"], dtype=np.float64)
    c_len = min(c_framed.size, c_whole.size)
    c_residual = c_framed[:c_len] - c_whole[:c_len]
    fixed = {
        "samples": int(c_len),
        "residual_rms": rms(c_residual),
        "residual_peak": float(np.max(np.abs(c_residual))) if c_len else 0.0,
    }
    report["fixed"] = fixed
    print(f"[seam] FIX framed-vs-whole: rms {fixed['residual_rms']:.9f} "
          f"peak {fixed['residual_peak']:.9f}")
    fix_ok = fixed["residual_peak"] < 1e-6
    print("[seam] fix verdict: "
          + ("the continuous resampler is chunking-INVARIANT -- framed output "
             "is bit-identical to whole" if fix_ok else
             "the fix still depends on how the stream was chunked"))

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if proven else 1


def main() -> int:
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--wire-rate", type=int, default=16000)
    parser.add_argument("--out-rate", type=int, default=44100)
    parser.add_argument("--frame", type=int, default=320,
                        help="samples per wire frame; 320 = 20 ms at 16 kHz")
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--freq", type=float, default=440.0)
    parser.add_argument("--json-out", type=Path, default=None)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
