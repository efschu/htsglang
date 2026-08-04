# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Where the time-to-first-audio of one synthesis goes, on the card.

FIELD EVIDENCE (2026-08-04, session ea7531b498c6). A healthy turn spent
151 ms in ASR, 57 ms in diarization, 325 ms in MT -- and 6894 ms in TTS,
93 % of the wait between the user finishing speaking and hearing a word.
The whole utterance then arrived in a 199 ms burst, because the reference
generates a complete unit before it emits anything: time-to-first-audio is
full generation time BY CONSTRUCTION.

Every existing number for the internals of that 6.9 s is CPU-only. This probe
opens the shipped call on the GPU it actually runs on, and separates:

* the conditioning prefill (reference encode, speaker encoder, prompt);
* the autoregressive talker steps (one codec frame each);
* the nested code predictor (16 residual codes per frame);
* the codec decode that turns frames into a waveform.

WHY HOOKS AND NOT A REASSEMBLED PIPELINE. `probe_stage_timing.py` times three
stages by calling `create_voice_clone_prompt` / `generate` / `decode` itself.
That is a DIFFERENT call sequence from the one the server runs -- the
inprocess_tts docstring records that the prompt shapes diverge -- so its split
cannot be attributed to the field number. Here the shipped
`InProcessQwen3Tts._generate` is driven unmodified and the attribution comes
from forward hooks, so what is measured is what the user waited for.

WHY CUDA EVENTS AND NOT synchronize(). A synchronize in a per-step hook makes
the host wait on every one of ~70 steps, which measures the instrument as much
as the talker. Events are recorded on the stream and resolved once at the end.
The host wall clock of the whole call is taken as well, and the difference
between it and the summed device time is reported as host overhead rather than
silently absorbed into a stage.

MEASUREMENT DISCIPLINE. Arms are INTERLEAVED, not run in blocks, so a drift in
card state cannot be read as an arm effect. The first `--warmup` draws of the
whole run are discarded. Two arms are byte-identical on purpose (a1/a2): the
spread between those two is the noise floor, and no other delta in this run
means anything unless it clears that floor.

    CUDA_VISIBLE_DEVICES=<uuid-of-the-card> PYTHONPATH=<repo>/python \\
      /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/probe_tts_latency_gpu.py --draws 6
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.srt.translator.audio import AudioChunk  # noqa: E402
from sglang.srt.translator.inprocess_tts import (  # noqa: E402
    InProcessQwen3Tts,
    InProcessTtsConfig,
)

VOICES = Path("/spinning/llm_stuff/translator-models/preset-voices")
RATE = 24000

#: The field durations, kept exactly: the reference the incident session
#: carried before and after a speaker merge.
REF_SHORT_S = 3.22
REF_LONG_S = 7.74

#: Real translations from the server transcript of 2026-08-04, so the
#: character counts are the ones the step budget was fitted against rather
#: than invented strings.
TEXT_SHORT = "Para nada."
TEXT_FIELD = "Hola, soy Matthias y estoy de vacaciones aqui. Como estas?"
TEXT_LONG = (
    "No es mas rapido en absoluto. Para nada. Que pasa? "
    "Hola, soy Matthias y estoy de vacaciones aqui, como estas hoy?"
)


class _MethodRestore:
    """Undo a `wrap_method`, with the same `.remove()` shape as a hook handle."""

    def __init__(self, holder, name: str, original) -> None:
        self._holder = holder
        self._name = name
        self._original = original

    def remove(self) -> None:
        setattr(self._holder, self._name, self._original)


class StageProbe:
    """Forward hooks that attribute one generation to its stages.

    Segmentation rule: a talker-trunk call whose query length is > 1 is a
    PREFILL and opens a new generation; every length-1 call after it is one
    autoregressive step. That is the same branch the talker itself takes on
    `cache_position`, so the split cannot drift from the model's own notion
    of prefill versus decode.
    """

    def __init__(self, torch) -> None:
        self.torch = torch
        self.handles: List = []
        self.reset()

    def reset(self) -> None:
        # Per generation: list of (label, start_event, end_event)
        self.spans: List = []
        self.generations: List[Dict] = []
        self._open: Dict[int, object] = {}

    # -- hook plumbing ---------------------------------------------------

    def _pre(self, label: str):
        def hook(module, args, kwargs=None):
            event = self.torch.cuda.Event(enable_timing=True)
            event.record()
            self._open[id(module)] = (label, event, time.perf_counter(), args, kwargs)
            return None

        return hook

    def _post(self, label: str):
        def hook(module, args, output):
            opened = self._open.pop(id(module), None)
            if opened is None:
                return output
            _label, start, host_started, in_args, in_kwargs = opened
            end = self.torch.cuda.Event(enable_timing=True)
            end.record()
            self.spans.append(
                {
                    "label": label,
                    "start": start,
                    "end": end,
                    "host_s": time.perf_counter() - host_started,
                    "qlen": _query_length(in_args, in_kwargs),
                }
            )
            return output

        return hook

    def attach(self, module, label: str) -> bool:
        if module is None or not hasattr(module, "register_forward_hook"):
            return False
        self.handles.append(
            module.register_forward_pre_hook(self._pre(label), with_kwargs=True)
        )
        self.handles.append(module.register_forward_hook(self._post(label)))
        return True

    def wrap_method(self, holder, name: str, label: str) -> bool:
        """Time a plain METHOD, for the stages that are not module forwards.

        The codec is reached as ``speech_tokenizer.decode`` -> ``model.decode``
        (inference/qwen3_tts_tokenizer.py:259,354), never through ``forward``,
        so a forward hook on it binds and never fires -- which is exactly how
        the pilot run reported ``codec_ms 0.0`` while the same time sat in the
        host residual. A stage that is timed at zero is worse than one that is
        not timed at all, so it is wrapped where it is actually called.
        """
        original = getattr(holder, name, None)
        if original is None or not callable(original):
            return False

        def wrapped(*args, **kwargs):
            start = self.torch.cuda.Event(enable_timing=True)
            start.record()
            host_started = time.perf_counter()
            result = original(*args, **kwargs)
            end = self.torch.cuda.Event(enable_timing=True)
            end.record()
            self.spans.append(
                {
                    "label": label,
                    "start": start,
                    "end": end,
                    "host_s": time.perf_counter() - host_started,
                    "qlen": None,
                }
            )
            return result

        setattr(holder, name, wrapped)
        self.handles.append(_MethodRestore(holder, name, original))
        return True

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    # -- resolution ------------------------------------------------------

    def resolve(self) -> List[Dict]:
        """Read every event back after ONE synchronize. Order is preserved."""
        self.torch.cuda.synchronize()
        out = []
        for span in self.spans:
            out.append(
                {
                    "label": span["label"],
                    "device_ms": span["start"].elapsed_time(span["end"]),
                    "host_ms": span["host_s"] * 1000.0,
                    "qlen": span["qlen"],
                }
            )
        return out


def _query_length(args, kwargs) -> Optional[int]:
    """Sequence length of a forward call, from whichever argument carries it."""
    import torch

    candidates = list(args or ())
    if kwargs:
        for key in ("input_ids", "inputs_embeds", "hidden_states"):
            if key in kwargs and kwargs[key] is not None:
                candidates.insert(0, kwargs[key])
    for item in candidates:
        if torch.is_tensor(item) and item.dim() >= 2:
            return int(item.shape[1])
    return None


def summarize(spans: List[Dict]) -> Dict:
    """Fold a span list into per-generation stage totals.

    A trunk call with qlen > 1 opens a generation. Everything up to the next
    such call belongs to it. Spans before the first prefill are conditioning
    (speaker encoder, reference codec encode) and are attributed to the
    generation that follows.
    """
    generations: List[Dict] = []
    current: Optional[Dict] = None
    pending_pre: List[Dict] = []

    def fresh() -> Dict:
        return {
            "prefill_ms": 0.0,
            "prefill_qlen": None,
            "steps": 0,
            "trunk_step_ms": [],
            "code_predictor_ms": [],
            "codec_ms": 0.0,
            "speaker_encoder_ms": 0.0,
            "codec_encode_ms": 0.0,
            "other_ms": 0.0,
        }

    for span in spans:
        label = span["label"]
        if label == "trunk":
            if span["qlen"] is not None and span["qlen"] > 1:
                if current is not None:
                    generations.append(current)
                current = fresh()
                for held in pending_pre:
                    _absorb(current, held)
                pending_pre = []
                current["prefill_ms"] = span["device_ms"]
                current["prefill_qlen"] = span["qlen"]
            else:
                if current is None:
                    current = fresh()
                current["steps"] += 1
                current["trunk_step_ms"].append(span["device_ms"])
            continue
        if current is None:
            pending_pre.append(span)
        else:
            _absorb(current, span)
    if current is not None:
        generations.append(current)
    return {"generations": generations}


def _absorb(generation: Dict, span: Dict) -> None:
    label = span["label"]
    if label == "code_predictor":
        generation["code_predictor_ms"].append(span["device_ms"])
    elif label == "codec_decode":
        generation["codec_ms"] += span["device_ms"]
    elif label == "codec_encode":
        generation["codec_encode_ms"] += span["device_ms"]
    elif label == "speaker_encoder":
        generation["speaker_encoder_ms"] += span["device_ms"]
    else:
        generation["other_ms"] += span["device_ms"]


def load_reference(path: Path, seconds: float) -> np.ndarray:
    data, rate = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != RATE:
        raise SystemExit(f"{path} is {rate} Hz, expected {RATE}")
    want = int(seconds * RATE)
    if len(data) < want:
        raise SystemExit(f"{path} is {len(data)/RATE:.2f}s, need {seconds}s")
    return data[:want]


def build_arms(args) -> List[Dict]:
    voice = VOICES / "man" / "man-03.de.wav"
    short_ref = load_reference(voice, REF_SHORT_S)
    long_ref = load_reference(voice, REF_LONG_S)
    arms = [
        # a1 and a2 are IDENTICAL. Their spread is the noise floor.
        {"name": "a1_field_ref3.2", "text": TEXT_FIELD, "ref": short_ref},
        {"name": "a2_field_ref3.2", "text": TEXT_FIELD, "ref": short_ref},
        {"name": "b_field_ref7.7", "text": TEXT_FIELD, "ref": long_ref},
        {"name": "c_short_ref3.2", "text": TEXT_SHORT, "ref": short_ref},
        {"name": "d_long_ref3.2", "text": TEXT_LONG, "ref": short_ref},
    ]
    if args.only:
        keep = {name.strip() for name in args.only.split(",")}
        arms = [arm for arm in arms if arm["name"] in keep]
    return arms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path,
                        default=Path("/spinning/llm_stuff/translator-models/"
                                     "qwen3-tts-0.6b-base"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--draws", type=int, default=6,
                        help="draws per arm, interleaved")
    parser.add_argument("--warmup", type=int, default=2,
                        help="leading generations discarded as JIT outliers")
    parser.add_argument("--idle-gap-s", type=float, default=0.0,
                        help="sleep before each draw, to expose a restore cost")
    parser.add_argument("--park-when-idle", action="store_true")
    parser.add_argument(
        "--park-each-draw", action="store_true",
        help="park the modules before every draw, so `ensure_resident` has "
             "something to restore. Needed because `park()` is called from "
             "`synthesize()`, which this probe bypasses -- setting only "
             "--park-when-idle therefore measures a restore that never "
             "happened, and reports 0.01 ms of it.",
    )
    parser.add_argument("--only", default="")
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", default="")
    parser.add_argument("--dump-tree", action="store_true")
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.WARNING)
    import torch

    backend = InProcessQwen3Tts(
        InProcessTtsConfig(
            model_dir=args.model_dir,
            device=args.device,
            dtype=args.dtype,
            park_when_idle=args.park_when_idle,
        )
    )
    load_started = time.perf_counter()
    backend.load()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - load_started
    inner = getattr(backend._model, "model", backend._model)

    if args.dump_tree:
        for name, _module in inner.named_children():
            print("child:", name, type(_module).__name__)
        for name, held in vars(inner).items():
            if not name.startswith("_") and not isinstance(held, torch.nn.Module):
                print("holder:", name, type(held).__name__,
                      "has .model" if hasattr(held, "model") else "")
        talker = getattr(inner, "talker", None)
        if talker is not None:
            for name, _module in talker.named_children():
                print("talker child:", name, type(_module).__name__)
        return 0

    probe = StageProbe(torch)
    attached = {
        "trunk": probe.attach(backend._resolve(inner, "talker.model"), "trunk"),
        "code_predictor": probe.attach(
            backend._resolve(inner, "talker.code_predictor"), "code_predictor"
        ),
        "speaker_encoder": probe.attach(
            backend._resolve(inner, "speaker_encoder"), "speaker_encoder"
        ),
        "codec_decode": probe.wrap_method(
            backend._resolve(inner, "speech_tokenizer"), "decode", "codec_decode"
        ),
        "codec_encode": probe.wrap_method(
            backend._resolve(inner, "speech_tokenizer"), "encode", "codec_encode"
        ),
    }
    print(json.dumps({"event": "hooks", "attached": attached,
                      "load_s": round(load_s, 2)}))
    if not attached["trunk"]:
        raise SystemExit("could not hook the talker trunk; attribution impossible")

    arms = build_arms(args)
    geometry = backend.geometry
    records: List[Dict] = []
    order = 0
    for draw in range(args.draws):
        for arm in arms:
            if args.idle_gap_s > 0:
                time.sleep(args.idle_gap_s)
            reference = AudioChunk(arm["ref"], RATE)
            probe.reset()
            parked_bytes = 0
            if args.park_each_draw:
                torch.cuda.synchronize()
                parked_bytes = backend.park()
                torch.cuda.synchronize()
            torch.cuda.synchronize()
            resident_started = time.perf_counter()
            backend.ensure_resident()
            torch.cuda.synchronize()
            resident_ms = (time.perf_counter() - resident_started) * 1000.0

            call_started = time.perf_counter()
            waveform = backend._generate(
                arm["text"], "es", reference, None
            )
            torch.cuda.synchronize()
            call_ms = (time.perf_counter() - call_started) * 1000.0

            spans = probe.resolve()
            folded = summarize(spans)
            audio_s = len(waveform) / RATE
            record = {
                "order": order,
                "draw": draw,
                "arm": arm["name"],
                "chars": len(arm["text"]),
                "ref_s": round(len(arm["ref"]) / RATE, 2),
                "call_ms": round(call_ms, 1),
                "ensure_resident_ms": round(resident_ms, 2),
                "parked_bytes": parked_bytes,
                "audio_s": round(audio_s, 3),
                "rtf": round(call_ms / 1000.0 / max(audio_s, 1e-6), 3),
                "generations": [],
            }
            for generation in folded["generations"]:
                steps = generation["steps"]
                trunk_total = sum(generation["trunk_step_ms"])
                predictor_total = sum(generation["code_predictor_ms"])
                record["generations"].append(
                    {
                        "steps": steps,
                        "prefill_ms": round(generation["prefill_ms"], 2),
                        "prefill_qlen": generation["prefill_qlen"],
                        "trunk_step_ms_total": round(trunk_total, 1),
                        "trunk_step_ms_mean": round(trunk_total / steps, 3)
                        if steps
                        else None,
                        "code_predictor_ms_total": round(predictor_total, 1),
                        "code_predictor_calls": len(generation["code_predictor_ms"]),
                        "codec_ms": round(generation["codec_ms"], 1),
                        "codec_encode_ms": round(generation["codec_encode_ms"], 2),
                        "speaker_encoder_ms": round(
                            generation["speaker_encoder_ms"], 2
                        ),
                        "other_ms": round(generation["other_ms"], 1),
                    }
                )
            device_ms = sum(
                g["prefill_ms"]
                + g["trunk_step_ms_total"]
                + g["code_predictor_ms_total"]
                + g["codec_ms"]
                + g["codec_encode_ms"]
                + g["speaker_encoder_ms"]
                + g["other_ms"]
                for g in record["generations"]
            )
            record["device_ms_sum"] = round(device_ms, 1)
            record["host_overhead_ms"] = round(call_ms - device_ms, 1)
            total_steps = sum(g["steps"] for g in record["generations"])
            record["total_steps"] = total_steps
            record["steps_per_s"] = (
                round(total_steps / (call_ms / 1000.0), 2) if call_ms else None
            )
            # The only honest read on the codec's real frame rate: how much
            # audio came out per talker step actually taken.
            record["measured_frame_hz"] = (
                round(total_steps / audio_s, 3) if audio_s > 0 else None
            )
            records.append(record)
            print(json.dumps(record))
            sys.stdout.flush()
            order += 1

    probe.detach()
    kept = [r for r in records if r["order"] >= args.warmup]
    report = {
        "event": "summary",
        "label": args.label,
        "device": args.device,
        "dtype": args.dtype,
        "park_when_idle": args.park_when_idle,
        "idle_gap_s": args.idle_gap_s,
        "config_frame_hz": geometry.frame_rate_hz,
        "num_code_groups": geometry.num_code_groups,
        "load_s": round(load_s, 2),
        "draws": args.draws,
        "warmup_discarded": args.warmup,
        "arms": {},
    }
    for arm in arms:
        rows = [r for r in kept if r["arm"] == arm["name"]]
        if not rows:
            continue
        report["arms"][arm["name"]] = _arm_summary(rows)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(
            json.dumps({"summary": report, "records": records}, indent=2)
        )
    return 0


def _stat(values: List[float]) -> Dict:
    if not values:
        return {}
    return {
        "n": len(values),
        "median": round(statistics.median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
    }


def _arm_summary(rows: List[Dict]) -> Dict:
    step_ms = []
    prefill_ms = []
    codec_ms = []
    codec_encode_ms = []
    predictor_ms = []
    predictor_per_step_ms = []
    predictor_calls_per_step = []
    speaker_ms = []
    for row in rows:
        for generation in row["generations"]:
            if generation["trunk_step_ms_mean"] is not None:
                step_ms.append(generation["trunk_step_ms_mean"])
            prefill_ms.append(generation["prefill_ms"])
            codec_ms.append(generation["codec_ms"])
            codec_encode_ms.append(generation["codec_encode_ms"])
            predictor_ms.append(generation["code_predictor_ms_total"])
            speaker_ms.append(generation["speaker_encoder_ms"])
            steps = generation["steps"]
            if steps:
                predictor_per_step_ms.append(
                    generation["code_predictor_ms_total"] / steps
                )
                predictor_calls_per_step.append(
                    generation["code_predictor_calls"] / steps
                )
    return {
        "chars": rows[0]["chars"],
        "ref_s": rows[0]["ref_s"],
        "call_ms": _stat([r["call_ms"] for r in rows]),
        "audio_s": _stat([r["audio_s"] for r in rows]),
        "rtf": _stat([r["rtf"] for r in rows]),
        "total_steps": _stat([float(r["total_steps"]) for r in rows]),
        "steps_per_s": _stat([r["steps_per_s"] for r in rows if r["steps_per_s"]]),
        "measured_frame_hz": _stat(
            [r["measured_frame_hz"] for r in rows if r["measured_frame_hz"]]
        ),
        "trunk_step_ms": _stat(step_ms),
        "prefill_ms": _stat(prefill_ms),
        "codec_decode_ms": _stat(codec_ms),
        "codec_encode_ms": _stat(codec_encode_ms),
        "code_predictor_ms_total": _stat(predictor_ms),
        "code_predictor_ms_per_step": _stat(predictor_per_step_ms),
        "code_predictor_calls_per_step": _stat(predictor_calls_per_step),
        "speaker_encoder_ms": _stat(speaker_ms),
        "ensure_resident_ms": _stat([r["ensure_resident_ms"] for r in rows]),
        "parked_bytes": _stat([float(r.get("parked_bytes", 0)) for r in rows]),
        "host_overhead_ms": _stat([r["host_overhead_ms"] for r in rows]),
        "device_ms_sum": _stat([r["device_ms_sum"] for r in rows]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
