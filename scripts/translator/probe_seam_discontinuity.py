# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Measure the discontinuity at every chunk seam, against what speech does there.

WHY THIS EXISTS. The user reports "ganz leichte knackser meist bei
wortanfängen" on the streaming build. Two causes were in play. The first --
`schedule()` re-applying its onset ramp on a mid-word re-anchor -- was
FALSIFIED from the field telemetry before this probe was written: in session
84ff8dd17a91, the build the report came from, there are zero mid-unit
re-anchors across three turns and 35 audio buffers, and every seam the client
did record begins within 0.008 of zero with a smooth head. There is no step at
any instrumented seam.

That leaves the boundaries the client never sampled. Since streaming shipped,
chunk N is cut from the decode of an N-frame prefix and chunk N+1 from the
decode of a longer one, and the codec's transformer reaches 25-50 frames
FORWARD in time -- so the same sample position has different values in the two
decodes and consecutive chunks need not join. Before streaming, a unit was
decoded once and sliced, and the seams were exact.

WHAT THIS MEASURES, and why not the number that was already reported.
`MEASURE_TTS_LATENCY.md` recorded "seams of 0.0427 against the one-shot
decode's own 0.0411" and read 3.7 % as "not a click". That is an AVERAGE of
absolute steps, and it is the wrong statistic twice over:

* absolute -- the same step is inaudible under a vowel and a click at a word
  onset, and "at word starts" is exactly what the user reports. The audible
  quantity is the step RELATIVE to the local signal;
* average -- one bad seam in twenty is what is heard. The distribution and
  its maximum are the evidence; the mean hides them.

So every seam is reported individually, normalized by the local RMS, with the
one-shot decode's own step at the SAME sample position as the control -- the
question is never "is there a step" (speech is full of them) but "is there a
step the waveform would not have had".

The spectral arm is the independent witness: a genuine discontinuity splatters
energy across the whole band, so it shows up ABOVE the speech band where a
real onset puts almost nothing. It is measured on the same windows.

WHAT IT FOUND, 2026-08-04, RTX 5090, two texts x two seeds, the live
translator and the 27B resident and untouched:

| text       | seed | seams | ONSET rel_max / control | STEADY rel_max / control |
|------------|-----:|------:|------------------------:|-------------------------:|
| field_58   |   11 |     7 |         0.07 / 0.07     |         0.41 / 0.40      |
| field_58   |   12 |     7 |         0.00 / 0.00     |         0.11 / 0.13      |
| field_long |   11 |    22 |         0.00 / 0.00     |         2.08 / 2.07      |
| field_long |   12 |    23 |         0.28 / 0.28     |         1.27 / 1.26      |

**THE CHUNK SEAM IS NOT THE CLICK.** The step at a seam is indistinguishable
from the step the one-shot decode already has at the SAME sample position --
0.41 against 0.40, 1.27 against 1.26. Seams that cross the audibility
threshold cross it because the CONTROL crosses it there too (3 of 22 in both
arms): that is speech moving fast, not a boundary artefact. At rising
envelopes -- the case the user's "bei wortanfängen" points at, separated out
here on purpose -- the step is 0.00 to 0.30 against a threshold of 1.0.

The arithmetic behind that, which the earlier round should have done: the two
decodes disagree by ~1 % of RMS (39.4 dB), while consecutive samples of 24 kHz
speech routinely differ by far more. A 1 % perturbation cannot dominate a step
that is already large. The previously reported "0.0427 against 0.0411" was not
hiding a tail in its average -- the ratio holds across the whole distribution.

A 5 ms overlap-add across the seam WAS built and measured against this and is
not shipped, because it changed nothing it was meant to change (0.41 -> 0.41,
1.27 -> 1.27, 2.08 -> 2.10) and slightly smeared what it touched: the
in-fade peak ratio against the one-shot decode moved 0.962 -> 0.926 on one
arm. Buying a perturbation for no measured benefit is worse than leaving it
alone. The lesson is recorded rather than the code.

So both candidates for the click are now falsified -- the re-anchor ramp from
the field telemetry, the chunk seam from here -- and the artefact lives
somewhere between these samples and the loudspeaker. The client's own
`audio.chunk_seam` instrument, which measures the RENDERED samples after
resampling, is what can see that stretch; this probe cannot.

Run with the venv interpreter:
  /spinning/htsglang-gpu/.venv/bin/python scripts/translator/probe_seam_discontinuity.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.srt.translator.audio import AudioChunk  # noqa: E402
from sglang.srt.translator.backends import TurnPacing  # noqa: E402
from sglang.srt.translator.inprocess_tts import (  # noqa: E402
    InProcessQwen3Tts,
    InProcessTtsConfig,
)

VOICES = Path("/spinning/llm_stuff/translator-models/preset-voices")
RATE = 24000
REF_S = 3.22

# The field shape. The 8.5 s turn is the one both field underruns sat in, and
# it is long enough to carry ~21 chunk seams; the 58-character turn is the
# field turn every earlier round was measured on.
TEXTS = {
    "field_58": (
        "Buenos días, me alegro mucho de verte hoy por aquí."
    ),
    "field_long": (
        "Buenos días, me alegro mucho de verte hoy por aquí. Hace mucho "
        "tiempo que no hablamos y tengo bastantes cosas que contarte sobre "
        "el viaje que hicimos el verano pasado."
    ),
}


class RecordingSink:
    """Keeps the chunks exactly as the emitter handed them over."""

    def __init__(self) -> None:
        self.chunks: List[AudioChunk] = []

    def push(self, chunk: Optional[AudioChunk]) -> None:
        if chunk is not None:
            self.chunks.append(chunk)

    def close(self) -> None:
        pass

    def audio(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([c.samples for c in self.chunks])

    def seam_positions(self) -> List[int]:
        """Sample index of the FIRST sample of every chunk after the first.

        These are the only positions where two decodes meet. Boundaries inside
        one `offer` are contiguous slices of a single array and cannot step.
        """
        positions: List[int] = []
        total = 0
        for chunk in self.chunks[:-1]:
            total += len(chunk.samples)
            positions.append(total)
        return positions


def load_reference() -> AudioChunk:
    path = VOICES / "man" / "man-03.de.wav"
    data, rate = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != RATE:
        raise SystemExit(f"{path} is {rate} Hz, expected {RATE}")
    return AudioChunk(data[: int(REF_S * RATE)], RATE)


def hf_ratio(window: np.ndarray, cutoff_hz: float = 6000.0) -> float:
    """Share of the window's energy above `cutoff_hz`.

    A step discontinuity is broadband: it puts energy everywhere, including
    where speech has almost none. A real onset -- even a plosive -- is shaped
    by the vocal tract and rolls off. So this separates "the waveform moved
    fast" from "the waveform jumped", which the time-domain step alone cannot.
    """
    if len(window) < 8:
        return 0.0
    taper = np.hanning(len(window))
    spectrum = np.abs(np.fft.rfft(window * taper)) ** 2
    freqs = np.fft.rfftfreq(len(window), d=1.0 / RATE)
    total = float(spectrum.sum())
    if total <= 0.0:
        return 0.0
    return float(spectrum[freqs >= cutoff_hz].sum() / total)


def seam_table(
    streamed: np.ndarray, control: np.ndarray, positions: List[int]
) -> List[Dict[str, float]]:
    """One row per seam: the step, the control step, and the splatter.

    `control` is the one-shot decode of the same generation -- the waveform the
    burst path would have produced. Every metric is computed on both at the
    SAME sample position, so the comparison isolates the seam rather than the
    speech.
    """
    rows: List[Dict[str, float]] = []
    # 20 ms of local signal, centred. Long enough to be an amplitude rather
    # than a sample, short enough that a word onset is not averaged together
    # with the vowel that follows it.
    half = int(0.010 * RATE)
    # 512 samples ~ 21 ms, the window the splatter is read from.
    fft_half = 256
    for position in positions:
        if position <= 0 or position >= len(streamed) or position >= len(control):
            continue
        local_window = control[max(0, position - half): position + half]
        local = float(np.sqrt(np.mean(np.square(local_window)))) if local_window.size else 0.0
        step = abs(float(streamed[position]) - float(streamed[position - 1]))
        control_step = abs(float(control[position]) - float(control[position - 1]))
        lo = max(0, position - fft_half)
        hi = min(len(streamed), position + fft_half)
        # THE ENVELOPE, because a pooled number cannot answer the question the
        # user is actually asking. He hears the artefact at WORD STARTS, which
        # are rising envelopes: quiet before, loud after. Those boundaries are
        # where a step is unmasked, and they are also the only place a 5 ms
        # crossfade between two decodes could SMEAR a transient rather than
        # join it. A seam in the middle of a vowel proves nothing either way.
        pre = control[max(0, position - half): position]
        post = control[position: position + half]
        pre_rms = float(np.sqrt(np.mean(np.square(pre)))) if pre.size else 0.0
        post_rms = float(np.sqrt(np.mean(np.square(post)))) if post.size else 0.0
        rise = post_rms / pre_rms if pre_rms > 1e-6 else (
            float("inf") if post_rms > 1e-4 else 1.0
        )
        # Inside the crossfade span: how far the emitted signal sits from the
        # one-shot decode. If the join SMEARS an onset, this is where it shows
        # -- the joined signal would be further from the control than the
        # unjoined one, not closer, precisely in a rising envelope.
        fade = min(int(0.005 * RATE), len(streamed) - position,
                   len(control) - position)
        if fade > 0:
            err = streamed[position: position + fade] - control[position: position + fade]
            fade_err = float(np.sqrt(np.mean(np.square(err))))
            peak_ratio = (
                float(np.abs(streamed[position: position + fade]).max()
                      / max(float(np.abs(control[position: position + fade]).max()), 1e-9))
            )
        else:
            fade_err = 0.0
            peak_ratio = 1.0
        rows.append({
            # Rising by at least 6 dB across the seam. Chosen as a doubling of
            # amplitude rather than a tuned value: that is the point at which
            # the pre-seam signal can no longer mask a step in the post-seam
            # one, which is the whole reason onsets are the audible case.
            "onset": bool(rise >= 2.0),
            "rise": min(rise, 999.0),
            "pre_rms": pre_rms,
            "post_rms": post_rms,
            "fade_err": fade_err,
            "fade_peak_ratio": peak_ratio,
            "position": int(position),
            "at_s": position / float(RATE),
            "local_rms": local,
            "step": step,
            "control_step": control_step,
            # THE AUDIBILITY NUMBER. A step of 0.04 under a 0.4 vowel is
            # nothing; the same step where the local RMS is 0.005 is the click.
            "rel": (step / local) if local > 1e-6 else 0.0,
            "control_rel": (control_step / local) if local > 1e-6 else 0.0,
            "hf": hf_ratio(streamed[lo:hi]),
            "control_hf": hf_ratio(control[lo:hi]),
        })
    return rows


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """The distribution, not the mean -- the mean is what hid this.

    Reported THREE times: all seams, the rising-envelope ones, and the rest.
    Pooling them is what makes a fix look good on average while the listener
    hears it, because the audible seams are a minority by construction.
    """
    if not rows:
        return {"seams": 0}
    out = _block(rows)
    onsets = [r for r in rows if r["onset"]]
    steady = [r for r in rows if not r["onset"]]
    out["onset"] = _block(onsets)
    out["steady"] = _block(steady)
    return out


def _block(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"seams": 0}
    rel = np.array([r["rel"] for r in rows])
    control_rel = np.array([r["control_rel"] for r in rows])
    hf = np.array([r["hf"] for r in rows])
    control_hf = np.array([r["control_hf"] for r in rows])
    return {
        "seams": len(rows),
        "rel_max": float(rel.max()),
        "rel_median": float(np.median(rel)),
        "control_rel_max": float(control_rel.max()),
        "control_rel_median": float(np.median(control_rel)),
        # Seams whose step is at least the local RMS. That is the same
        # threshold the client's own chunk-seam instrument logs on, so a field
        # package and this probe report the same quantity.
        "audible": int((rel >= 1.0).sum()),
        "control_audible": int((control_rel >= 1.0).sum()),
        "hf_max": float(hf.max()),
        "control_hf_max": float(control_hf.max()),
        "hf_excess_max": float((hf - control_hf).max()),
        # The smear test. Inside the crossfade the emitted signal should sit
        # CLOSER to the one-shot decode, not further: the join fades towards
        # the decode that saw more future. A join that smeared a transient
        # would show a larger error and a peak ratio below 1.
        "fade_err_max": float(max(r["fade_err"] for r in rows)),
        "fade_peak_ratio_min": float(min(r["fade_peak_ratio"] for r in rows)),
    }


def run_one(
    backend, torch, text: str, reference, seed: int, streaming: bool = True,
) -> Dict:
    """One generation.

    With `streaming=False` this is the BURST path, whose `_generate` returns
    the reference's own one-shot decode of the finished unit -- the control.
    With `streaming=True` it returns `emitter.assembled()`, which is what went
    on the wire and therefore what the listener heard.
    """
    backend.config = dataclasses.replace(
        backend.config, stream_within_unit=streaming,
    )
    sink = RecordingSink()
    pacing = TurnPacing()
    # Same seed on every arm: the talker samples from the global generator, so
    # this is what makes the arms decode the IDENTICAL token sequence. Without
    # it the arms differ by a draw whose step count has a 15 % standard
    # deviation, which is far larger than the effect being measured.
    torch.manual_seed(seed)
    started = time.monotonic()
    waveform = backend._generate(text, "es", reference, None, sink, pacing)
    total = time.monotonic() - started
    streamed = sink.audio()
    return {
        "seed": seed,
        "chars": len(text),
        "total_ms": total * 1000.0,
        "chunks": len(sink.chunks),
        "streamed_samples": int(len(streamed)),
        "result_samples": int(len(waveform)),
        "streamed": streamed,
        "positions": sink.seam_positions(),
        "waveform": waveform,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--out", type=Path,
                        default=Path("/spinning/466-client-logs/seam_discontinuity.json"))
    parser.add_argument("--wav-dir", type=Path, default=None,
                        help="write the streamed waveforms here for listening")
    args = parser.parse_args()

    import torch

    backend = InProcessQwen3Tts(InProcessTtsConfig())
    backend.load()
    reference = load_reference()

    records: List[Dict] = []
    for name, text in TEXTS.items():
        for seed in args.seeds:
            # THE CONTROL is the burst path on the same seed: one generation,
            # one decode of the whole unit, sliced by nobody. Its step at a
            # given sample position is what the waveform does there naturally,
            # which is the only fair thing to judge a seam against.
            burst = run_one(backend, torch, text, reference, seed,
                            streaming=False)
            control = burst["waveform"]
            arms = {0.0: run_one(backend, torch, text, reference, seed)}
            for _key, arm in arms.items():
                rows = seam_table(arm["streamed"], control, arm["positions"])
                record = {
                    "text": name,
                    "seed": seed,
                                "chunks": arm["chunks"],
                    "streamed_samples": arm["streamed_samples"],
                    "result_samples": arm["result_samples"],
                    # The stream and the burst must cover the same sample
                    # positions; if they do not, every seam metric below is
                    # comparing two different points in the utterance and the
                    # run is void rather than merely noisy.
                    "control_samples": int(len(control)),
                    "lengths_match": bool(len(control) == arm["streamed_samples"]),
                    "total_ms": arm["total_ms"],
                    "summary": summarize(rows),
                    "seams": rows,
                }
                records.append(record)
                s = record["summary"]
                on = s.get("onset", {})
                st = s.get("steady", {})
                print(
                    f"{name:<11} seed={seed} "
                    f"chunks={arm['chunks']:>3} seams={s.get('seams', 0):>3} "
                    f"| ONSET n={on.get('seams', 0):>2} "
                    f"rel_max={on.get('rel_max', 0):.2f} "
                    f"(ctl {on.get('control_rel_max', 0):.2f}) "
                    f"audible={on.get('audible', 0)}/{on.get('control_audible', 0)} "
                    f"| STEADY n={st.get('seams', 0):>2} "
                    f"rel_max={st.get('rel_max', 0):.2f} "
                    f"(ctl {st.get('control_rel_max', 0):.2f}) "
                    f"| fade_err={s.get('fade_err_max', 0):.4f} "
                    f"peak_ratio={s.get('fade_peak_ratio_min', 0):.3f}",
                    flush=True,
                )
                if args.wav_dir is not None:
                    args.wav_dir.mkdir(parents=True, exist_ok=True)
                    sf.write(
                        str(args.wav_dir /
                            f"{name}-s{seed}.wav"),
                        arm["streamed"], RATE,
                    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
