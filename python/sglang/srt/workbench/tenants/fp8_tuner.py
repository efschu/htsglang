# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Block-FP8 GEMM shape tuning as idle work (DESIGN #347 W6, the #255 remnant).

#255 committed two tuned RTX 5090 (sm120) block-FP8 shard configs and left
three shapes queued, on the observation that per-shape tuned tiles are
microbench-verified and cannot regress -- worst case they equal the untuned
default. The mechanism that ran them was a shell loop that took a card lock
when free, ran exactly one combination, released, and slept. That loop is the
idle-tuner protocol; this module is it, as a workbench tenant.

Three properties are carried over deliberately:

**One combination per segment.** A work item is one ``(N, K, M)`` -- one
weight shape at one batch size -- because that is the granularity the tuning
script writes at, so a preempted segment loses at most one combination and
never leaves a half-written config.

**Nothing is committed.** Results land in the tenant's artifact directory and
the operator is told the repo path they would be copied to. A kernel config
that appears in the tree without a human reading the A/B is a silent
performance change with no provenance.

**Shapes are input, not code.** They come from a queue file or the enqueue
endpoint. Which shapes matter is a fact about a deployment's model and TP
split; a hardcoded list would be exactly the rig-only assumption ANALYSE #347
excludes.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from sglang.srt.workbench.tenant import (
    DEFAULT_CUDA_CONTEXT_BYTES,
    MIB,
    EventSink,
    IdleWorkTenant,
    SubprocessSegment,
    WorkEstimate,
    WorkEvent,
    WorkGrant,
    WorkSegment,
)

logger = logging.getLogger(__name__)

#: The two operating points a serving engine actually runs these GEMMs at:
#: one decode-shaped batch and one prefill-chunk-shaped batch. Overridable
#: per work item; they are defaults, not a claim that no other M matters.
DEFAULT_BATCH_SIZES = (4, 2048)

#: Bytes per element the tuning script allocates, by role. ``input_type=fp8``
#: keeps both the fp32 draft it samples and the cast-down tensor alive for the
#: whole sweep, which is why both are posts.
_FP32 = 4
_QUANT = 1

#: Triton compiles and benchmarks every config in the search space; the
#: output tensor is reallocated per call and the caching allocator keeps the
#: high-water mark. An estimate, labelled as one, not a safety factor: it is
#: a post the operator can see and argue with.
TRITON_SWEEP_BYTES = 256 * MIB

#: Where a tuned config belongs once a human has read the A/B.
REPO_CONFIG_DIR = "python/sglang/srt/layers/quantization/configs"


@dataclass(frozen=True)
class TunerCombo:
    """One weight shape at one batch size: the unit of work."""

    n: int
    k: int
    batch_size: int
    input_type: str = "fp8"
    out_dtype: str = "bfloat16"
    block_n: int = 128
    block_k: int = 128

    @property
    def key(self) -> str:
        return f"N={self.n},K={self.k},M={self.batch_size},{self.input_type}"

    def config_filename(self, device_name: str) -> str:
        """The name the tuning script's ``save_configs`` writes."""
        return (
            f"N={self.n},K={self.k},device_name={device_name},"
            f"dtype={self.input_type}_w8a8,"
            f"block_shape=[{self.block_n}, {self.block_k}].json"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "k": self.k,
            "batch_size": self.batch_size,
            "input_type": self.input_type,
            "out_dtype": self.out_dtype,
            "block_n": self.block_n,
            "block_k": self.block_k,
        }


def parse_queue(text: str) -> list[TunerCombo]:
    """``<N> <K> [M,M,...]`` per line, ``#`` comments, blanks ignored.

    Deliberately the shape of the shell tuner's ``queue.txt`` so an existing
    queue is usable as-is. A line without batch sizes expands to
    :data:`DEFAULT_BATCH_SIZES`.
    """
    out: list[TunerCombo] = []
    for raw in (text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"tuner queue line {raw!r} needs at least '<N> <K>'")
        try:
            n, k = int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(
                f"tuner queue line {raw!r}: N and K must be integers"
            ) from None
        batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES
        if len(parts) > 2:
            batch_sizes = [int(b) for b in parts[2].split(",") if b.strip()]
        for batch in batch_sizes:
            out.append(TunerCombo(n=n, k=k, batch_size=int(batch)))
    return out


def repo_root() -> Path:
    """The checkout this package was imported from, or its parent tree."""
    import sglang

    return Path(sglang.__file__).resolve().parents[2]


class Fp8BlockTunerTenant(IdleWorkTenant):
    """Tunes one block-quantized GEMM combination per idle segment."""

    name = "fp8_tuner"
    priority = 50

    def __init__(
        self,
        *,
        artifact_root: Path,
        queue_path: Optional[Path] = None,
        combos: Sequence[TunerCombo] = (),
        script_path: Optional[Path] = None,
        python_executable: str = "",
        card_selector: str = "largest",
        device_resolver: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.artifact_root = Path(artifact_root)
        self.config_dir = self.artifact_root / "configs"
        self.queue_path = Path(queue_path) if queue_path else None
        self.script_path = (
            Path(script_path)
            if script_path
            else (
                repo_root()
                / "benchmark"
                / "kernels"
                / "quantization"
                / "tuning_block_wise_kernel.py"
            )
        )
        self.python_executable = python_executable or sys.executable
        #: ``largest`` (most total VRAM), an NVML index, or a name fragment.
        #: Resolved on every call, never cached: NVML enumeration order can
        #: shift between boots and a cached index is a wrong-card bug.
        self.card_selector = card_selector
        self._device_resolver = device_resolver or _nvml_devices
        self._queue: list[TunerCombo] = list(combos)
        self._loaded_queue_mtime: Optional[float] = None
        self._failed: dict[str, str] = {}

    # -- availability -------------------------------------------------------

    def available(self) -> tuple[bool, str]:
        if not self.script_path.is_file():
            return False, (
                f"the tuning script is not in this tree ({self.script_path}); "
                "it ships with the source checkout, not with the wheel. Pass "
                "an explicit path or run the workbench from a checkout."
            )
        try:
            self._card()
        except LookupError as exc:
            return False, str(exc)
        return True, ""

    def describe(self) -> str:
        try:
            card = self._card()
            where = f"card {card['index']} ({card['name']})"
        except LookupError:
            where = "no resolvable card"
        return (
            f"block-quantized GEMM shape tuning on {where}; results to "
            f"{self.config_dir}"
        )

    # -- the queue ----------------------------------------------------------

    def _card(self) -> dict[str, Any]:
        devices = self._device_resolver()
        if not devices:
            raise LookupError("no GPU is visible to this process")
        selector = (self.card_selector or "largest").strip()
        if selector.lower() == "largest":
            # Re-derived every call from NVML rather than stored: the biggest
            # card is where the widest shard lives, and which index that is
            # can change between boots.
            return max(devices, key=lambda d: (d["total_bytes"], -d["index"]))
        if re.fullmatch(r"\d+", selector):
            for device in devices:
                if device["index"] == int(selector):
                    return device
            # Fall through rather than raise: a digit string can also be a
            # name fragment ("3090"), and refusing it because no card carries
            # that index would be a surprising way to lose a valid selector.
        matches = [d for d in devices if selector.lower() in d["name"].lower()]
        if not matches:
            raise LookupError(
                f"--workbench-tuner-card {selector!r} matches no visible card "
                f"by index or by name; visible: "
                f"{[(d['index'], d['name']) for d in devices]}"
            )
        return matches[0]

    def device_name(self) -> str:
        """The device-name token the config filename is keyed on."""
        return self._card()["name"].replace(" ", "_")

    def _refresh_queue(self) -> None:
        if self.queue_path is None or not self.queue_path.is_file():
            return
        mtime = self.queue_path.stat().st_mtime
        if self._loaded_queue_mtime == mtime:
            return
        try:
            parsed = parse_queue(self.queue_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("workbench fp8_tuner: queue file unusable: %s", exc)
            return
        self._loaded_queue_mtime = mtime
        known = {c.key for c in self._queue}
        self._queue.extend(c for c in parsed if c.key not in known)

    def _is_done(self, combo: TunerCombo, device_name: str) -> bool:
        """A combination is done when its batch size is in the config file.

        Idempotent by construction: rerunning after a driver change is a
        delete-and-requeue, not a code edit.
        """
        path = self.config_dir / combo.config_filename(device_name)
        if not path.is_file():
            return False
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return str(combo.batch_size) in {str(k) for k in body}

    def remaining(self) -> list[TunerCombo]:
        self._refresh_queue()
        try:
            device_name = self.device_name()
        except LookupError:
            return []
        return [
            c
            for c in self._queue
            if c.key not in self._failed and not self._is_done(c, device_name)
        ]

    def pending(self) -> int:
        return len(self.remaining())

    def enqueue(self, item: Mapping[str, Any]) -> str:
        try:
            combo = TunerCombo(
                n=int(item["n"]),
                k=int(item["k"]),
                batch_size=int(item.get("batch_size", DEFAULT_BATCH_SIZES[0])),
                input_type=str(item.get("input_type", "fp8")),
                out_dtype=str(item.get("out_dtype", "bfloat16")),
                block_n=int(item.get("block_n", 128)),
                block_k=int(item.get("block_k", 128)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "a tuner work item needs integer 'n' and 'k' and may carry "
                f"'batch_size', 'input_type', 'out_dtype', 'block_n', "
                f"'block_k' ({exc})"
            ) from None
        if combo.input_type not in ("fp8", "int8"):
            raise ValueError(
                f"input_type {combo.input_type!r} is not one the tuning script "
                "accepts; use 'fp8' or 'int8'"
            )
        if combo.key not in {c.key for c in self._queue}:
            self._queue.append(combo)
        self._failed.pop(combo.key, None)
        return combo.key

    # -- pricing ------------------------------------------------------------

    def estimate(self) -> WorkEstimate:
        remaining = self.remaining()
        if not remaining:
            return WorkEstimate(per_card_bytes=0, cards_wanted=1)
        combo = remaining[0]
        card = self._card()
        posts = combo_posts(combo)
        return WorkEstimate(
            per_card_bytes=sum(posts.values()),
            posts=posts,
            card_uuids=(card["uuid"],),
            cards_wanted=1,
            # One combination took 35-70 s in the #255 runs; the estimate is
            # reported, never enforced.
            expected_seconds=90.0,
        )

    # -- running ------------------------------------------------------------

    async def start_segment(self, grant: WorkGrant, sink: EventSink) -> WorkSegment:
        remaining = self.remaining()
        if not remaining:
            raise RuntimeError("the tuner queue is empty")
        combo = remaining[0]
        self.config_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            self.python_executable,
            str(self.script_path),
            "--N",
            str(combo.n),
            "--K",
            str(combo.k),
            "--input-type",
            combo.input_type,
            "--out-dtype",
            combo.out_dtype,
            "--block-n",
            str(combo.block_n),
            "--block-k",
            str(combo.block_k),
            "--batch-sizes",
            str(combo.batch_size),
            "--save-path",
            str(self.config_dir),
        ]
        device_name = self.device_name()
        artifact = str(self.config_dir / combo.config_filename(device_name))
        sink(
            WorkEvent(
                "info",
                f"tuning {combo.key} on card {grant.card_indices} "
                f"({device_name}); result goes to {artifact}, and belongs in "
                f"{REPO_CONFIG_DIR}/ only after a human has read the A/B",
                data=combo.to_json(),
            )
        )
        segment = _TunerSegment(
            tenant=self,
            combo=combo,
            argv=argv,
            cwd=repo_root(),
            env={
                "CUDA_VISIBLE_DEVICES": grant.visible_devices,
                "PYTHONPATH": _pythonpath(),
            },
            sink=sink,
            label=f"fp8_tuner {combo.key}",
            artifact_path=artifact,
            line_filter=_tuner_line,
        )
        return await segment.start()

    def note_failure(self, combo: TunerCombo, error: str) -> None:
        """Stop retrying a combination that the script itself rejects.

        The #255 queue had a shape that failed twice with the same exit code;
        a scheduler that retries forever turns one bad shape into a rig that
        never does anything else.
        """
        self._failed[combo.key] = error

    def snapshot(self) -> dict[str, Any]:
        body = super().snapshot()
        body.update(
            {
                "queue": [c.to_json() for c in self._queue],
                "remaining": [c.key for c in self.remaining()],
                "failed": dict(self._failed),
                "config_dir": str(self.config_dir),
                "repo_config_dir": REPO_CONFIG_DIR,
                "queue_path": str(self.queue_path) if self.queue_path else None,
            }
        )
        return body


class _TunerSegment(SubprocessSegment):
    """One combination. Marks the combination failed when the script rejects it."""

    def __init__(
        self, *, tenant: Fp8BlockTunerTenant, combo: TunerCombo, **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.tenant = tenant
        self.combo = combo

    async def wait(self):
        outcome = await super().wait()
        if outcome.status.value == "failed":
            self.tenant.note_failure(self.combo, outcome.error or "unknown")
        return outcome


def combo_posts(combo: TunerCombo) -> dict[str, int]:
    """The named posts of one tuning combination. No safety factor.

    The script samples an fp32 draft of A and B, casts each down, and keeps
    both alive for the whole sweep; the per-block scales are fp32; the output
    is reallocated per benchmarked config and the caching allocator holds the
    high-water mark.
    """
    m, n, k = combo.batch_size, combo.n, combo.k
    out_bytes = 4 if combo.out_dtype == "float32" else 2
    k_tiles = -(-k // combo.block_k)
    n_tiles = -(-n // combo.block_n)
    return {
        "cuda_context": DEFAULT_CUDA_CONTEXT_BYTES,
        "a_fp32_draft": m * k * _FP32,
        "a_quantized": m * k * _QUANT,
        "b_fp32_draft": n * k * _FP32,
        "b_quantized": n * k * _QUANT,
        "scales": (m * k_tiles + n_tiles * k_tiles) * _FP32,
        "output": m * n * out_bytes,
        "triton_sweep": TRITON_SWEEP_BYTES,
    }


def _tuner_line(line: str) -> Optional[WorkEvent]:
    """Only the lines that mean something to an operator become events."""
    lowered = line.lower()
    if "completed tuning" in lowered or "writing best config" in lowered:
        return WorkEvent("info", line, type="metrics")
    if "error" in lowered or "traceback" in lowered:
        return WorkEvent("error", line)
    return None


def _pythonpath() -> str:
    """Keep the child on the same checkout this process was imported from."""
    import sglang

    package_root = str(Path(sglang.__file__).resolve().parents[1])
    existing = [p for p in sys.path if p and p != package_root]
    return ":".join([package_root, *existing[:1]]) if existing else package_root


def _nvml_devices() -> list[dict[str, Any]]:
    from sglang.srt.registry import nvml

    return [
        {
            "uuid": d.uuid,
            "index": d.index,
            "name": d.name,
            "total_bytes": d.total_bytes,
        }
        for d in nvml.list_devices()
    ]


__all__ = [
    "DEFAULT_BATCH_SIZES",
    "Fp8BlockTunerTenant",
    "REPO_CONFIG_DIR",
    "TunerCombo",
    "combo_posts",
    "parse_queue",
]
