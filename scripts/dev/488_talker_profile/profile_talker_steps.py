#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#488 precursor: where do the 6.4 ms per talker forward actually go?

THE QUESTION
------------
``ANALYSE_488_talker_lane_layout.md`` rests on one premise: the Qwen3-TTS
talker is **launch- and Python-overhead bound at batch 1**, not bandwidth
bound. The arithmetic says one audio-second is 192 forward passes and 37.0 GiB
of weight reads, i.e. a bandwidth floor of RTF 0.022 on a 5090 against a
measured 1.23 -- so 24-55x of the cost is somewhere other than weight traffic.
This script says WHERE, in minutes, against today's in-process module, and it
does not need the native lane to exist.

If the premise holds, the decomposition shows a large ``gap`` (wall time not
covered by any kernel) and a kernel count per forward in the hundreds. If it
does not hold -- if the kernels really do account for the wall clock -- then
the whole redirect in ANALYSE_488 is wrong and TP=3 deserves a second look.
The script is written so that outcome is reportable, not just the expected one.

THE INSTRUMENT PROVES ITSELF FIRST
----------------------------------
CLAUDE.md: "an INSTRUMENT's verdict counts only after the instrument passes a
can-discriminate check on known-different inputs". So before it measures
anything real, this script runs two synthetic arms whose answer is known:

* ``gpu_bound``   -- a handful of large matmuls. Kernels must cover most of the
  wall clock; the gap fraction must be SMALL.
* ``launch_bound`` -- thousands of tiny elementwise ops. The gap fraction must
  be LARGE.

If those two do not separate by at least ``_MIN_SEPARATION``, the script
REFUSES and measures nothing: an instrument that cannot tell the two regimes
apart cannot testify about which one the talker is in. That refusal is the
point -- a degenerate profiler reporting "overhead bound" would confirm the
hypothesis for free.

RUNNING IT
----------
Standalone (loads its own copy of the module -- needs ~2 GB free)::

    CUDA_VISIBLE_DEVICES=<5090> python profile_talker_steps.py --json out.json

Inside a live tenant process, with NO new allocation, which is the preferred
shape while the serving instance holds the cards::

    from profile_talker_steps import profile_loaded_model
    report = profile_loaded_model(tts._model)   # the InProcessQwen3Tts wrapper

Every loop here is bounded by iteration count AND by a wall deadline, so a
wedged kernel ends the arm instead of the window.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import sys
import time
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("talker-profile")

#: The gap fraction of the launch-bound arm must exceed the gpu-bound arm's by
#: at least this much, or the instrument is not discriminating.
_MIN_SEPARATION = 0.40
#: A gpu-bound arm whose gap fraction is above this is not gpu-bound; the box
#: is contended and no verdict may be drawn from it.
_MAX_GPU_BOUND_GAP = 0.35
#: Per-arm wall deadline. Bounded waits only.
_ARM_DEADLINE_S = 60.0
#: Whole-run deadline, so the caller can promise the operator a number.
_RUN_DEADLINE_S = 360.0
#: Transient the calibration arms allocate: two 4096^2 fp16 operands plus the
#: matmul output. Measured, not guessed -- see ``_CALIB_SHAPE``.
_CALIB_SHAPE = (2048, 2048)
_CALIB_MIB = 3 * _CALIB_SHAPE[0] * _CALIB_SHAPE[1] * 2 / (1024 * 1024)
#: The standing VRAM corridor: at least this much must remain free on the card
#: AFTER our transient. Non-negotiable -- the tenant on this card is serving a
#: live conversation while this runs.
_MIN_FREE_MIB_AFTER = 400.0
#: What a STANDALONE second copy costs: 1745 MiB of checkpoint (measured from
#: the safetensors header) plus CUDA context, cuBLAS workspaces, the codec and
#: the speaker encoder. Only the standalone entry point pays this; the
#: in-process one reuses the tenant's already-resident modules.
_STANDALONE_FOOTPRINT_MIB = 2600.0


def check_headroom(free_mib: float, need_mib: float = _CALIB_MIB) -> Optional[str]:
    """``None`` when there is room, else the refusal text.

    Pure, so the arithmetic is testable without a card. The instrument is run
    INSIDE a process that is serving a live conversation: an OOM here is not a
    failed measurement, it is a dropped turn in front of the user.
    """
    remaining = free_mib - need_mib
    if remaining < _MIN_FREE_MIB_AFTER:
        return (
            f"only {free_mib:.0f} MiB free on the card; the calibration arms "
            f"need {need_mib:.0f} MiB and the standing corridor requires "
            f"{_MIN_FREE_MIB_AFTER:.0f} MiB to remain free afterwards "
            f"({remaining:.0f} MiB would). Refusing: this runs inside a "
            f"process serving a live conversation, where an OOM is a dropped "
            f"turn, not a failed measurement."
        )
    return None


@dataclasses.dataclass
class ArmResult:
    """One measured region, decomposed."""

    name: str
    iterations: int
    wall_s: float
    kernel_s: float
    kernel_count: int
    #: Device-to-host copies and explicit syncs seen by the profiler. Each one
    #: is a pipeline bubble the size of everything queued behind it.
    sync_count: int
    note: str = ""

    @property
    def wall_per_iter_ms(self) -> float:
        return 1000.0 * self.wall_s / max(1, self.iterations)

    @property
    def kernel_per_iter_ms(self) -> float:
        return 1000.0 * self.kernel_s / max(1, self.iterations)

    @property
    def gap_ms(self) -> float:
        """Wall time no kernel accounts for: launch latency + Python + syncs."""
        return max(0.0, self.wall_per_iter_ms - self.kernel_per_iter_ms)

    @property
    def gap_fraction(self) -> float:
        if self.wall_per_iter_ms <= 0.0:
            return 0.0
        return self.gap_ms / self.wall_per_iter_ms

    @property
    def kernels_per_iter(self) -> float:
        return self.kernel_count / max(1, self.iterations)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "wall_per_iter_ms": round(self.wall_per_iter_ms, 4),
            "kernel_per_iter_ms": round(self.kernel_per_iter_ms, 4),
            "gap_ms": round(self.gap_ms, 4),
            "gap_fraction": round(self.gap_fraction, 4),
            "kernels_per_iter": round(self.kernels_per_iter, 1),
            "sync_count": self.sync_count,
            "note": self.note,
        }


@dataclasses.dataclass
class Discrimination:
    """Whether the instrument may testify at all."""

    ok: bool
    reason: str
    gpu_bound_gap: float
    launch_bound_gap: float

    @property
    def separation(self) -> float:
        return self.launch_bound_gap - self.gpu_bound_gap

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "gpu_bound_gap_fraction": round(self.gpu_bound_gap, 4),
            "launch_bound_gap_fraction": round(self.launch_bound_gap, 4),
            "separation": round(self.separation, 4),
            "min_separation_required": _MIN_SEPARATION,
        }


def check_discrimination(gpu_bound: ArmResult, launch_bound: ArmResult) -> Discrimination:
    """The spread precondition, as a pure function so it is testable off-GPU.

    Two ways to fail, and they mean different things:

    * the arms do not separate -> the profiler is not resolving what it claims
      to resolve (wrong activity set, clock too coarse, everything inlined);
    * the gpu-bound arm itself has a large gap -> the box is contended, and on
      a contended box every arm looks launch-bound. Refusing here is what
      stops a busy serving instance from manufacturing the expected answer.
    """
    gpu_gap = gpu_bound.gap_fraction
    launch_gap = launch_bound.gap_fraction
    if gpu_gap > _MAX_GPU_BOUND_GAP:
        return Discrimination(
            ok=False,
            reason=(
                f"the known GPU-BOUND arm shows a gap fraction of {gpu_gap:.3f}, "
                f"above the {_MAX_GPU_BOUND_GAP} ceiling. Something outside this "
                f"process is taking the device, so every arm would look "
                f"overhead-bound and no verdict about the talker may be drawn. "
                f"Re-run when the card is quieter."
            ),
            gpu_bound_gap=gpu_gap,
            launch_bound_gap=launch_gap,
        )
    if launch_gap - gpu_gap < _MIN_SEPARATION:
        return Discrimination(
            ok=False,
            reason=(
                f"the two calibration arms separate by only "
                f"{launch_gap - gpu_gap:.3f} (< {_MIN_SEPARATION}); the "
                f"instrument cannot distinguish a launch-bound region from a "
                f"GPU-bound one, so its reading of the talker means nothing."
            ),
            gpu_bound_gap=gpu_gap,
            launch_bound_gap=launch_gap,
        )
    return Discrimination(
        ok=True,
        reason=(
            f"calibration arms separate by {launch_gap - gpu_gap:.3f} "
            f"(gpu-bound {gpu_gap:.3f} vs launch-bound {launch_gap:.3f})"
        ),
        gpu_bound_gap=gpu_gap,
        launch_bound_gap=launch_gap,
    )


def project_rtf(frame_ms: float, frame_hz: float = 12.0) -> float:
    """Real-time factor implied by a measured per-FRAME cost.

    One frame is one trunk step plus its residual groups, i.e. everything the
    talker does for 1/frame_hz seconds of audio.
    """
    return frame_ms * frame_hz / 1000.0


def recoverable_rtf(arms: Dict[str, ArmResult], frame_hz: float = 12.0) -> Optional[dict]:
    """What the measured frame would cost if the gap went to zero.

    This is the number the whole precursor exists for: it is the ceiling a
    CUDA-graph + tight-loop implementation is aiming at, computed from THIS
    box's own kernel times rather than from a bandwidth model. Reported
    alongside the bandwidth floor from ANALYSE_488 §3 so the two can be
    compared -- if they disagree badly, one of them is wrong and that is worth
    knowing before anyone builds.
    """
    frame = arms.get("frame")
    if frame is None:
        return None
    measured = project_rtf(frame.wall_per_iter_ms, frame_hz)
    kernels_only = project_rtf(frame.kernel_per_iter_ms, frame_hz)
    return {
        "frame_hz": frame_hz,
        "measured_rtf": round(measured, 4),
        "kernel_only_rtf": round(kernels_only, 4),
        "recoverable_factor": (
            round(measured / kernels_only, 2) if kernels_only > 0 else None
        ),
        "bandwidth_floor_rtf_5090_from_analyse_488": 0.022,
        "bandwidth_floor_rtf_3080_from_analyse_488": 0.052,
    }


# ---------------------------------------------------------------------------
# CUDA-side measurement. Everything above this line runs anywhere.
# ---------------------------------------------------------------------------


def _profile_region(
    name: str,
    fn: Callable[[], None],
    iterations: int,
    warmup: int = 3,
    note: str = "",
) -> ArmResult:
    """Run ``fn`` ``iterations`` times under the torch profiler and decompose.

    Warmup is discarded (benchmark-harness rule); the deadline is checked
    between iterations so a wedged call ends the arm, not the window.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    deadline = time.monotonic() + _ARM_DEADLINE_S
    done = 0
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
            done += 1
            if time.monotonic() > deadline:
                note = (note + " " if note else "") + (
                    f"DEADLINE: stopped after {done}/{iterations} iterations"
                )
                break
        torch.cuda.synchronize()
        wall = time.perf_counter() - start

    kernel_s = 0.0
    kernel_count = 0
    sync_count = 0
    for event in prof.key_averages():
        device_us = getattr(event, "self_device_time_total", 0) or 0
        if device_us:
            kernel_s += device_us / 1e6
            kernel_count += event.count
        lowered = event.key.lower()
        if "memcpy" in lowered and "dtoh" in lowered:
            sync_count += event.count
        elif "synchronize" in lowered:
            sync_count += event.count

    return ArmResult(
        name=name,
        iterations=done,
        wall_s=wall,
        kernel_s=kernel_s,
        kernel_count=kernel_count,
        sync_count=sync_count,
        note=note,
    )


def _calibration_arms(device: str) -> Dict[str, ArmResult]:
    """The two known-answer arms. Small and short-lived: this must be runnable
    inside a serving process without moving the VRAM needle."""
    import torch

    arms: Dict[str, ArmResult] = {}

    # Each arm owns its operands in its own scope, so the tensors are gone
    # before the next arm allocates. Matters here: this runs inside a serving
    # process whose card is already ~85 % full.
    def run_gpu_bound() -> ArmResult:
        # ~64 MiB of fp16 operands; large enough to be unambiguously
        # kernel-bound, small enough that a contended card still has room.
        a = torch.randn(*_CALIB_SHAPE, device=device, dtype=torch.float16)
        b = torch.randn(*_CALIB_SHAPE, device=device, dtype=torch.float16)

        def body() -> None:
            for _ in range(4):
                torch.mm(a, b)

        return _profile_region(
            "calib_gpu_bound",
            body,
            iterations=20,
            note="4x 2048^3 fp16 matmul -- kernels must cover the wall clock",
        )

    def run_launch_bound() -> ArmResult:
        tiny = torch.randn(64, device=device, dtype=torch.float16)

        def body() -> None:
            x = tiny
            for _ in range(400):
                x = x + 1.0

        return _profile_region(
            "calib_launch_bound",
            body,
            iterations=20,
            note="400 tiny adds -- wall clock must exceed the kernels",
        )

    # The operands die with each helper's frame; empty_cache after the frame
    # is gone is what actually returns the bytes to the driver.
    arms["calib_gpu_bound"] = run_gpu_bound()
    torch.cuda.empty_cache()
    arms["calib_launch_bound"] = run_launch_bound()
    torch.cuda.empty_cache()
    return arms


def profile_loaded_model(model, device: Optional[str] = None) -> dict:
    """Decompose an ALREADY LOADED Qwen3-TTS module. No new weights.

    ``model`` is whatever ``InProcessQwen3Tts._model`` holds -- the reference
    ``Qwen3TTSModel`` wrapper. Only its talker halves are touched, and only
    with decode-shaped inputs, so nothing here synthesises audio or moves the
    conversation state.
    """
    import torch

    inner = getattr(model, "model", model)
    talker = getattr(inner, "talker", None)
    if talker is None:
        raise ValueError(
            "no .talker on the given model; expected the reference "
            "Qwen3TTSModel wrapper (see translator/inprocess_tts.py)"
        )
    trunk = talker.model
    predictor = talker.code_predictor
    if device is None:
        device = str(next(trunk.parameters()).device)
    dtype = next(trunk.parameters()).dtype
    hidden = trunk.config.hidden_size

    report: Dict[str, object] = {"device": device, "dtype": str(dtype)}

    # Headroom precondition, BEFORE anything is allocated. On 2026-08-04 the
    # 5090 carried rank 0 (22436 MiB) plus this tenant (5910 MiB) with 3605 MiB
    # free, so the 96 MiB calibration transient is comfortable -- but the
    # margin is a fact about that moment, not a property of the box, so it is
    # re-read every run.
    free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device(device).index or 0)
    free_mib = free_bytes / (1024 * 1024)
    report["headroom"] = {
        "free_mib": round(free_mib, 1),
        "total_mib": round(total_bytes / (1024 * 1024), 1),
        "calibration_transient_mib": round(_CALIB_MIB, 1),
        "corridor_floor_mib": _MIN_FREE_MIB_AFTER,
    }
    refusal = check_headroom(free_mib)
    if refusal is not None:
        report["verdict"] = "REFUSED -- " + refusal
        return report

    run_deadline = time.monotonic() + _RUN_DEADLINE_S
    arms = _calibration_arms(device)

    discrimination = check_discrimination(
        arms["calib_gpu_bound"], arms["calib_launch_bound"]
    )
    report["discrimination"] = discrimination.to_json()
    if not discrimination.ok:
        report["arms"] = {k: v.to_json() for k, v in arms.items()}
        report["verdict"] = "REFUSED -- " + discrimination.reason
        return report

    # Decode-shaped inputs. batch 1, one position: exactly what a frame does.
    embeds = torch.randn(1, 1, hidden, device=device, dtype=dtype)

    def trunk_step() -> None:
        with torch.inference_mode():
            trunk(inputs_embeds=embeds, use_cache=False)

    def predictor_step() -> None:
        with torch.inference_mode():
            predictor.model(inputs_embeds=embeds, use_cache=False)

    def predictor_generate() -> None:
        with torch.inference_mode():
            predictor.generate(
                inputs_embeds=torch.cat([embeds, embeds], dim=1),
                max_new_tokens=predictor.config.num_code_groups - 1,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

    measurement_arms = (
        ("trunk_step", trunk_step, 30,
         "one 28-layer trunk forward, batch 1, one position = one frame"),
        ("predictor_step", predictor_step, 30,
         "one 5-layer code-predictor forward = one residual group"),
        ("predictor_generate", predictor_generate, 10,
         "the HF generate() envelope the reference re-enters per frame"),
    )
    for name, fn, iterations, note in measurement_arms:
        if time.monotonic() > run_deadline:
            report[f"{name}_skipped"] = "run deadline reached"
            continue
        try:
            arms[name] = _profile_region(name, fn, iterations=iterations, note=note)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            logger.warning("%s arm failed: %s", name, exc)
            report[f"{name}_error"] = str(exc)

    if "trunk_step" not in arms or "predictor_step" not in arms:
        report["verdict"] = (
            "REFUSED -- the trunk and predictor step arms are both required "
            "and at least one did not run; see the *_error entries."
        )
        report["arms"] = {k: v.to_json() for k, v in arms.items()}
        return report

    # The frame arm is derived rather than run, because running it needs the
    # real prompt state. Stated as a derivation so nobody reads it as measured.
    groups = predictor.config.num_code_groups - 1
    envelope = arms.get("predictor_generate")
    if envelope is not None:
        frame_ms = arms["trunk_step"].wall_per_iter_ms + envelope.wall_per_iter_ms
        frame_kernel_ms = (
            arms["trunk_step"].kernel_per_iter_ms + envelope.kernel_per_iter_ms
        )
    else:
        frame_ms = (
            arms["trunk_step"].wall_per_iter_ms
            + groups * arms["predictor_step"].wall_per_iter_ms
        )
        frame_kernel_ms = (
            arms["trunk_step"].kernel_per_iter_ms
            + groups * arms["predictor_step"].kernel_per_iter_ms
        )
    arms["frame"] = ArmResult(
        name="frame",
        iterations=1,
        wall_s=frame_ms / 1000.0,
        kernel_s=frame_kernel_ms / 1000.0,
        kernel_count=int(
            arms["trunk_step"].kernels_per_iter
            + (
                envelope.kernels_per_iter
                if envelope is not None
                else groups * arms["predictor_step"].kernels_per_iter
            )
        ),
        sync_count=arms["trunk_step"].sync_count,
        note="DERIVED from the step arms, not measured end to end",
    )

    report["arms"] = {k: v.to_json() for k, v in arms.items()}
    report["rtf"] = recoverable_rtf(arms)
    report["verdict"] = _verdict(arms)
    return report


def _verdict(arms: Dict[str, ArmResult]) -> str:
    frame = arms["frame"]
    if frame.gap_fraction >= 0.7:
        return (
            f"OVERHEAD-BOUND confirmed: {frame.gap_fraction:.1%} of the frame's "
            f"{frame.wall_per_iter_ms:.2f} ms is covered by no kernel "
            f"({frame.kernels_per_iter:.0f} kernels per frame). A graph-captured "
            f"tight loop is the lever; TP is not."
        )
    if frame.gap_fraction <= 0.3:
        return (
            f"PREMISE FALSIFIED: kernels account for "
            f"{1 - frame.gap_fraction:.1%} of the frame's "
            f"{frame.wall_per_iter_ms:.2f} ms. The talker is NOT overhead-bound "
            f"on this box, and ANALYSE_488's redirect needs revisiting -- "
            f"dividing real kernel work across cards may pay after all."
        )
    return (
        f"MIXED: gap fraction {frame.gap_fraction:.2f} on a "
        f"{frame.wall_per_iter_ms:.2f} ms frame. Neither reading is clean; "
        f"report the bands, do not pick a side."
    )


#: What this process calls itself in `ps` / `py-spy` output. Deliberately
#: carries the ticket AND the word GUEST: the two questions a triage asks are
#: "whose is this" and "is it supposed to be on this card".
PROCESS_TAG = "sglang::488-talker-profile-GUEST"


def _tag_process() -> None:
    """Rename this process so a VRAM triage can attribute it immediately.

    Best-effort by design: it is a diagnostic aid, and a profiling run must
    not die because a process-title library is missing. Both mechanisms are
    tried because they show up in different tools -- ``setproctitle`` rewrites
    argv (``ps``, ``/proc/<pid>/cmdline``), ``prctl`` sets the comm name
    (``/proc/<pid>/comm``, ``top``), and ``nvidia-smi`` resolves names through
    the former.
    """
    try:
        import setproctitle  # noqa: PLC0415

        setproctitle.setproctitle(PROCESS_TAG)
    except Exception:  # noqa: BLE001 - diagnostic only
        pass
    try:
        import ctypes  # noqa: PLC0415

        # PR_SET_NAME = 15; the kernel truncates comm to 15 bytes + NUL.
        ctypes.CDLL("libc.so.6").prctl(
            15, PROCESS_TAG.encode()[:15] + b"\0", 0, 0, 0
        )
    except Exception:  # noqa: BLE001 - diagnostic only
        pass


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base",
    )
    parser.add_argument("--json", default=None, help="write the report here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # PROCESS MARKER, before anything slow happens. This path puts a SECOND
    # copy of the talker on a card that a live tenant and a serving rank are
    # already using, so a triage looking at `nvidia-smi` sees an unexplained
    # ~2.6 GiB and an unfamiliar pid. On 2026-08-04 that cost minutes of
    # triage on pid 4020715 -- this run. A process whose argv says what it is
    # gets attributed in seconds instead.
    _tag_process()

    # STANDALONE ONLY: this path loads its OWN copy of the talker onto a card
    # that is already carrying a serving rank and a live tenant, so the
    # headroom check has to happen BEFORE the weights land -- the in-process
    # entry point checks after, because there is nothing to load there.
    import torch  # noqa: PLC0415

    free_bytes, _ = torch.cuda.mem_get_info(0)
    free_mib = free_bytes / (1024 * 1024)
    need_mib = _STANDALONE_FOOTPRINT_MIB + _CALIB_MIB
    if free_mib - need_mib < _MIN_FREE_MIB_AFTER:
        logger.error(
            "REFUSED: %.0f MiB free, a standalone copy needs ~%.0f MiB and the "
            "corridor requires %.0f MiB to remain. Use the in-process entry "
            "point (profile_loaded_model) inside the tenant instead.",
            free_mib, need_mib, _MIN_FREE_MIB_AFTER,
        )
        return 2
    logger.info(
        "headroom ok: %.0f MiB free, standalone copy needs ~%.0f MiB",
        free_mib, need_mib,
    )

    from sglang.srt.translator.inprocess_tts import (  # noqa: PLC0415
        InProcessQwen3Tts,
        InProcessTtsConfig,
    )

    tts = InProcessQwen3Tts(
        InProcessTtsConfig(model_dir=pathlib.Path(args.model_dir))
    )
    tts.load()
    report = profile_loaded_model(tts._model)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
