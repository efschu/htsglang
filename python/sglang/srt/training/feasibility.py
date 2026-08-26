# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Whether a training job fits, computed against this machine (DESIGN #341 D2).

The rule this module exists to enforce: **feasibility is a formula, not a rig
constant.** Nothing here knows how many cards the development rig has or how
big they are. Every number comes from NVML, from the VRAM ledger, from
``/proc/meminfo``, from ``statvfs``, and from the model's own config and
weight files. A 27B full finetune is a perfectly ordinary request that fails
on a 20 GB card and succeeds on a node of H100s, and the same code says so in
both cases.

The second rule: **a rejection must be actionable.** "Out of memory" tells the
caller nothing they can use. So a rejection carries the arithmetic -- every
named post, the per-card total, the shortfall -- and the method ladder: which
cheaper method *would* have fit, with its own arithmetic beside it. A user
whose full finetune is refused should be able to read off that QLoRA fits with
3.1 GiB to spare and resubmit.

## The posts

Per card, for a training step:

===================  =====================================================
``weights``          resident base weights, at the method's storage dtype
``gradients``        trainable parameters x gradient dtype
``optimizer``        trainable parameters x optimizer state bytes
``activations``      the standard transformer activation estimate below
``logits``           batch x seq x vocab x 4, the fp32 cross-entropy upcast
``cuda_context``     the driver context and allocator arena of one process
===================  =====================================================

``activations`` uses the published per-layer-per-token estimate (Korthikanti
et al., *Reducing Activation Recomputation in Large Transformer Models*,
2022): ``34h + 5*a*s`` bytes per token per layer at 16-bit without
recomputation, and ``2h`` with full gradient checkpointing. It is an estimate
and is labelled as one; it is used because it is the published one, not
because it is exact.

``logits`` is separated out because it is frequently the largest single post
at long sequence lengths and large vocabularies, and a caller staring at an
OOM has no way to guess that. Trainers that chunk the loss shrink it; the
estimate does not assume they do.

There is no safety factor and no implicit ceiling anywhere in this module.
``cuda_context`` and the ledger's corridor are named posts with numbers
attached, which is the opposite of a hidden margin: a caller can see them,
argue with them, and configure them.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

MIB = 1024 * 1024
GIB = 1024 * MIB

#: One CUDA context plus the allocator's first arena, per training process.
#: A named post rather than a margin: it is real memory the driver takes
#: before the first tensor exists, and hiding it inside a fudge factor is how
#: a "fits" verdict turns into a runtime OOM.
DEFAULT_CUDA_CONTEXT_BYTES = 600 * MIB


class TrainingMethod(str, Enum):
    """The method ladder, cheapest storage class first.

    Ordering here is nominal. The report orders by *computed* cost on the
    actual request, because which method is cheaper depends on the model:
    freezing all but the last two layers of a 3B model beats LoRA over all of
    them, and the reverse holds at 70B.
    """

    QLORA = "qlora"
    LORA = "lora"
    FREEZE = "freeze"
    FULL_OFFLOAD = "full_offload"
    FULL = "full"

    @property
    def is_full_finetune(self) -> bool:
        return self in (TrainingMethod.FULL, TrainingMethod.FULL_OFFLOAD)


#: The ladder order DESIGN #341 D2 names, used when two methods tie on cost.
LADDER = (
    TrainingMethod.FREEZE,
    TrainingMethod.LORA,
    TrainingMethod.QLORA,
    TrainingMethod.FULL_OFFLOAD,
    TrainingMethod.FULL,
)


def parse_method(raw: str) -> TrainingMethod:
    try:
        return TrainingMethod(str(raw).strip().lower())
    except ValueError:
        known = ", ".join(m.value for m in LADDER)
        raise ValueError(
            f"unknown training method {raw!r}; known methods are {known}"
        ) from None


# ---------------------------------------------------------------------------
# The machine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CardResources:
    """One physical GPU as the gate sees it."""

    uuid: str
    index: int
    name: str
    total_bytes: int
    #: What the ledger says a new tenant could still claim, corridor removed.
    #: Equal to ``total_bytes`` minus the corridor when the card is empty.
    available_bytes: int

    def describe(self) -> str:
        return (
            f"card {self.index} {self.name} ({self.uuid}): "
            f"{self.total_bytes / MIB:.0f} MiB total, "
            f"{self.available_bytes / MIB:.0f} MiB claimable"
        )


@dataclass(frozen=True)
class MachineResources:
    """Everything the formula is evaluated against. All of it measured."""

    cards: tuple[CardResources, ...] = ()
    ram_total_bytes: int = 0
    ram_available_bytes: int = 0
    disk_free_bytes: int = 0
    disk_path: str = ""
    #: Filled in when NVML could not be reached, so a rejection can say why
    #: the card list is empty instead of claiming the machine has no GPUs.
    probe_error: Optional[str] = None

    @property
    def vram_total_bytes(self) -> int:
        return sum(c.total_bytes for c in self.cards)

    @property
    def vram_available_bytes(self) -> int:
        return sum(c.available_bytes for c in self.cards)

    def to_json(self) -> dict[str, Any]:
        return {
            "cards": [
                {
                    "uuid": c.uuid,
                    "index": c.index,
                    "name": c.name,
                    "total_mib": c.total_bytes // MIB,
                    "available_mib": c.available_bytes // MIB,
                }
                for c in self.cards
            ],
            "vram_total_mib": self.vram_total_bytes // MIB,
            "vram_available_mib": self.vram_available_bytes // MIB,
            "ram_total_mib": self.ram_total_bytes // MIB,
            "ram_available_mib": self.ram_available_bytes // MIB,
            "disk_free_mib": self.disk_free_bytes // MIB,
            "disk_path": self.disk_path,
            "probe_error": self.probe_error,
        }


def read_meminfo(path: str = "/proc/meminfo") -> tuple[int, int]:
    """``(MemTotal, MemAvailable)`` in bytes. ``(0, 0)`` off Linux.

    #871b: THE SECOND COPY OF THE SAME REIMPLEMENTATION, closed in the same
    pass as the first. This is only ADVISORY -- unlike
    ``turnkey/preflight.py``, nothing here refuses a boot -- but a defect class
    with a known second instance does not get left pending, because that is
    exactly how the first one survived long enough to reach a gate.

    The default path now goes through ``memtier.profile`` (#407 owner), which
    corrects for two things a raw ``/proc/meminfo`` read cannot: inside this
    container the file is synthesised by lxcfs (``MemAvailable`` can exceed
    ``MemTotal``, and with ``memory.max`` unlimited it reports the HOST's
    figures), and the owner additionally clamps by what this cgroup already
    holds.

    An EXPLICIT ``path`` still reads that file directly. Callers that pass one
    are asking about a specific file -- the tests do -- and silently ignoring
    an argument would be worse than the reading it corrects.
    """
    if path == "/proc/meminfo":
        try:
            from sglang.srt.memtier.profile import host_memory_bytes_for_pinning

            total, available = host_memory_bytes_for_pinning()
            if total is not None and available is not None:
                return int(total), int(available)
        except Exception:  # noqa: BLE001 - advisory probe, never raises
            pass
        # Fall through: the owner could not establish a pair, so the raw file
        # is a better answer than zeros -- this caller only reports.
    try:
        values: dict[str, int] = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    values[key.strip()] = int(parts[0]) * 1024
        return values.get("MemTotal", 0), values.get(
            "MemAvailable", values.get("MemFree", 0)
        )
    except (OSError, ValueError):
        return 0, 0


def probe_machine(
    *,
    artifact_root: Path | str = "/tmp",
    store: Any = None,
    card_filter: Optional[Sequence[str]] = None,
) -> MachineResources:
    """Read the actual machine. Never raises; an unreachable NVML is a fact.

    ``store`` is a :class:`~sglang.srt.registry.ledger.ReservationStore`. When
    given, ``available_bytes`` accounts for what other tenants already hold,
    so the gate answers "would this fit *now*" rather than "would this fit on
    an empty rig".
    """
    cards: list[CardResources] = []
    probe_error: Optional[str] = None
    try:
        from sglang.srt.registry import nvml  # noqa: PLC0415

        devices = nvml.list_devices()
    except Exception as exc:  # noqa: BLE001 - a missing NVML is reportable, not fatal
        devices = []
        probe_error = f"{type(exc).__name__}: {exc}"

    for device in devices:
        if card_filter and device.uuid not in card_filter:
            continue
        available = device.total_bytes
        if store is not None:
            try:
                from sglang.srt.registry.ledger import (  # noqa: PLC0415
                    available_bytes as ledger_available,
                )

                available = ledger_available(store, device.uuid, device.total_bytes)
            except Exception as exc:  # noqa: BLE001 - ledger optional at plan time
                logger.debug(
                    "training: ledger read for %s failed: %s", device.uuid, exc
                )
        cards.append(
            CardResources(
                uuid=device.uuid,
                index=device.index,
                name=device.name,
                total_bytes=device.total_bytes,
                available_bytes=int(available),
            )
        )

    ram_total, ram_available = read_meminfo()
    root = Path(artifact_root)
    probe_dir = root if root.is_dir() else root.parent
    try:
        usage = shutil.disk_usage(probe_dir)
        disk_free = usage.free
    except OSError:
        disk_free = 0
    return MachineResources(
        cards=tuple(cards),
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_available,
        disk_free_bytes=disk_free,
        disk_path=str(probe_dir),
        probe_error=probe_error,
    )


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

_DTYPE_BYTES = {
    "float32": 4.0,
    "float": 4.0,
    "fp32": 4.0,
    "bfloat16": 2.0,
    "bf16": 2.0,
    "float16": 2.0,
    "fp16": 2.0,
    "half": 2.0,
    "int8": 1.0,
    "float8_e4m3fn": 1.0,
    "fp8": 1.0,
    "uint8": 1.0,
    "int4": 0.5,
}


@dataclass(frozen=True)
class ModelProfile:
    """What the formula needs to know about the base model.

    Read from the model directory, never guessed from the name. ``source``
    records which of the two available derivations produced ``params`` so a
    caller can tell a counted number from an estimated one.
    """

    path: str
    params: int
    hidden_size: int
    num_layers: int
    num_heads: int
    vocab_size: int
    stored_dtype_bytes: float
    weight_bytes_on_disk: int
    source: str
    model_type: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "params": self.params,
            "params_billion": round(self.params / 1e9, 3),
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "vocab_size": self.vocab_size,
            "stored_dtype_bytes": self.stored_dtype_bytes,
            "weight_mib_on_disk": self.weight_bytes_on_disk // MIB,
            "param_count_source": self.source,
            "model_type": self.model_type,
        }


class ModelProfileError(ValueError):
    """The base model could not be profiled, so nothing can be decided."""


def _weight_bytes_on_disk(directory: Path) -> int:
    total = 0
    for pattern in ("*.safetensors", "*.bin", "*.gguf", "*.pt"):
        for candidate in directory.glob(pattern):
            with contextlib.suppress(OSError):
                total += candidate.stat().st_size
    return total


def _analytic_params(config: Mapping[str, Any]) -> int:
    """Parameter count from the architecture, for checkpoints we cannot weigh.

    Embeddings plus, per layer, the attention projections (GQA-aware) and the
    gated MLP. Accurate to a few percent on every dense decoder in this tree,
    which is the resolution the gate needs.
    """
    hidden = int(config.get("hidden_size") or 0)
    layers = int(config.get("num_hidden_layers") or 0)
    vocab = int(config.get("vocab_size") or 0)
    heads = int(config.get("num_attention_heads") or 1) or 1
    kv_heads = int(config.get("num_key_value_heads") or heads) or heads
    inter = int(config.get("intermediate_size") or 4 * hidden)
    head_dim = int(config.get("head_dim") or (hidden // heads if heads else 0))
    if not (hidden and layers and vocab):
        return 0
    tied = bool(config.get("tie_word_embeddings", False))
    embed = vocab * hidden * (1 if tied else 2)
    q = hidden * heads * head_dim
    kv = 2 * hidden * kv_heads * head_dim
    o = heads * head_dim * hidden
    mlp = 3 * hidden * inter
    return int(embed + layers * (q + kv + o + mlp))


def profile_model(path: str | Path) -> ModelProfile:
    """Profile a base model directory. Raises when it cannot be read."""
    directory = Path(path)
    if not directory.is_dir():
        raise ModelProfileError(
            f"base model path {str(directory)!r} is not a directory on this "
            "server. The fine-tuning API trains local weights, so the model "
            "must be a path this process can read."
        )
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise ModelProfileError(
            f"base model path {str(directory)!r} has no config.json; the "
            "resource formula cannot be evaluated without the architecture."
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelProfileError(f"could not read {config_path}: {exc}") from None
    if "text_config" in config and isinstance(config["text_config"], dict):
        # Multimodal checkpoints keep the decoder geometry one level down.
        merged = dict(config["text_config"])
        merged.setdefault("torch_dtype", config.get("torch_dtype"))
        merged.setdefault("model_type", config.get("model_type"))
        config = merged

    dtype_bytes = _DTYPE_BYTES.get(str(config.get("torch_dtype") or "").lower(), 2.0)
    disk_bytes = _weight_bytes_on_disk(directory)

    params = 0
    source = ""
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            index = json.loads(index_path.read_text(encoding="utf-8"))
            total = int((index.get("metadata") or {}).get("total_size") or 0)
            if total:
                params = int(total / dtype_bytes)
                source = "safetensors_index_total_size"
    if not params and disk_bytes:
        params = int(disk_bytes / dtype_bytes)
        source = "weight_files_on_disk"
    analytic = _analytic_params(config)
    if not params:
        params = analytic
        source = "config_architecture"
    if not params:
        raise ModelProfileError(
            f"could not determine a parameter count for {str(directory)!r}: no "
            "weight files, no safetensors index, and config.json lacks "
            "hidden_size / num_hidden_layers / vocab_size."
        )

    return ModelProfile(
        path=str(directory),
        params=params,
        hidden_size=int(config.get("hidden_size") or 0),
        num_layers=int(config.get("num_hidden_layers") or 0),
        num_heads=int(config.get("num_attention_heads") or 0),
        vocab_size=int(config.get("vocab_size") or 0),
        stored_dtype_bytes=dtype_bytes,
        weight_bytes_on_disk=disk_bytes,
        source=source,
        model_type=str(config.get("model_type") or ""),
    )


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingDemandSpec:
    """The knobs of the requested run that move the memory numbers."""

    sequence_length: int = 2048
    micro_batch_size: int = 1
    gradient_checkpointing: bool = True
    #: Fraction of parameters trained under ``freeze``. The default freezes
    #: everything but roughly the top eighth of the stack.
    freeze_trainable_fraction: float = 0.125
    #: LoRA adapter parameters as a fraction of the base. Rank 16 over the
    #: attention and MLP projections of a typical decoder lands near this.
    lora_trainable_fraction: float = 0.0025
    #: AdamW keeps two fp32 moments per trainable parameter.
    optimizer_state_bytes: float = 8.0
    gradient_dtype_bytes: float = 2.0
    train_dtype_bytes: float = 2.0
    #: How many cards the run may use, and whether they replicate (DDP) or
    #: shard (ZeRO-3 / FSDP) the weights, gradients and optimizer state.
    world_size: int = 1
    sharded: bool = False
    cuda_context_bytes: int = DEFAULT_CUDA_CONTEXT_BYTES
    #: Checkpoints kept on disk at once, for the disk post.
    checkpoints_retained: int = 2

    def with_world_size(
        self, world_size: int, *, sharded: bool
    ) -> "TrainingDemandSpec":
        return TrainingDemandSpec(
            **{
                **self.__dict__,
                "world_size": max(1, int(world_size)),
                "sharded": bool(sharded),
            }
        )


def _method_weight_bytes_per_param(method: TrainingMethod, train_dtype: float) -> float:
    if method is TrainingMethod.QLORA:
        # NF4 is 4 bits plus one fp16 absmax per block of 64 and a second
        # level of fp8 scales: 0.5 + 2/64 + 1/256 bytes per parameter.
        return 0.5 + 2.0 / 64.0 + 1.0 / 256.0
    return train_dtype


def _trainable_fraction(method: TrainingMethod, spec: TrainingDemandSpec) -> float:
    if method in (TrainingMethod.LORA, TrainingMethod.QLORA):
        return spec.lora_trainable_fraction
    if method is TrainingMethod.FREEZE:
        return spec.freeze_trainable_fraction
    return 1.0


@dataclass(frozen=True)
class Posts:
    """The named memory posts of one card, and the host-side ones."""

    weights: int = 0
    gradients: int = 0
    optimizer: int = 0
    activations: int = 0
    logits: int = 0
    cuda_context: int = 0
    host_offload: int = 0

    @property
    def device_total(self) -> int:
        return (
            self.weights
            + self.gradients
            + self.optimizer
            + self.activations
            + self.logits
            + self.cuda_context
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "weights_mib": self.weights // MIB,
            "gradients_mib": self.gradients // MIB,
            "optimizer_mib": self.optimizer // MIB,
            "activations_mib": self.activations // MIB,
            "logits_mib": self.logits // MIB,
            "cuda_context_mib": self.cuda_context // MIB,
            "host_offload_mib": self.host_offload // MIB,
            "device_total_mib": self.device_total // MIB,
        }

    def render(self) -> str:
        rows = [
            ("weights", self.weights),
            ("gradients", self.gradients),
            ("optimizer", self.optimizer),
            ("activations", self.activations),
            ("logits", self.logits),
            ("cuda_context", self.cuda_context),
        ]
        body = ", ".join(f"{name} {value / MIB:.0f} MiB" for name, value in rows)
        line = f"{body}; per-card total {self.device_total / MIB:.0f} MiB"
        if self.host_offload:
            line += f"; host RAM {self.host_offload / MIB:.0f} MiB"
        return line


def compute_posts(
    model: ModelProfile, method: TrainingMethod, spec: TrainingDemandSpec
) -> Posts:
    """The formula. Every term is named, and none of them is a fudge factor."""
    world = max(1, spec.world_size)
    shard = world if spec.sharded else 1

    params = model.params
    trainable = params * _trainable_fraction(method, spec)

    weights = params * _method_weight_bytes_per_param(method, spec.train_dtype_bytes)
    if method in (TrainingMethod.LORA, TrainingMethod.QLORA):
        # The adapters are trained in the compute dtype on top of the frozen
        # base, so they are a second, small weight post.
        weights += trainable * spec.train_dtype_bytes
    gradients = trainable * spec.gradient_dtype_bytes
    optimizer = trainable * spec.optimizer_state_bytes

    host_offload = 0
    if method is TrainingMethod.FULL_OFFLOAD:
        # ZeRO-3 with optimizer and gradient offload: the states live in host
        # RAM and stream in. They leave the card entirely, which is the whole
        # point of the rung, and land on the host post instead.
        host_offload = int(optimizer + gradients)
        optimizer = 0
        gradients = 0

    weights = weights / shard
    gradients = gradients / shard
    optimizer = optimizer / shard

    tokens = spec.micro_batch_size * spec.sequence_length
    per_token_per_layer = (
        2.0 * model.hidden_size
        if spec.gradient_checkpointing
        else 34.0 * model.hidden_size + 5.0 * model.num_heads * spec.sequence_length
    )
    activations = tokens * model.num_layers * per_token_per_layer
    # The loss upcasts logits to fp32. At 150k vocab and 4k sequence this is
    # gigabytes and routinely the post that decides the verdict.
    logits = tokens * model.vocab_size * 4.0

    return Posts(
        weights=int(weights),
        gradients=int(gradients),
        optimizer=int(optimizer),
        activations=int(activations),
        logits=int(logits),
        cuda_context=int(spec.cuda_context_bytes),
        host_offload=int(host_offload),
    )


def checkpoint_bytes(
    model: ModelProfile, method: TrainingMethod, spec: TrainingDemandSpec
) -> int:
    """Disk one saved checkpoint takes: trainable weights plus their states.

    Resumable checkpoints carry the optimizer moments, which is why a LoRA
    checkpoint is five times its adapter file and a full-finetune checkpoint
    is five times the model.
    """
    trainable = model.params * _trainable_fraction(method, spec)
    return int(trainable * (spec.train_dtype_bytes + spec.optimizer_state_bytes))


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodOption:
    """One rung of the ladder, priced against this machine."""

    method: TrainingMethod
    posts: Posts
    world_size: int
    fits: bool
    limiting_card: Optional[str] = None
    headroom_bytes: int = 0
    reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "fits": self.fits,
            "world_size": self.world_size,
            "per_card_mib": self.posts.device_total // MIB,
            "headroom_mib": self.headroom_bytes // MIB,
            "limiting_card": self.limiting_card,
            "posts": self.posts.to_json(),
            "reason": self.reason,
        }

    def render(self) -> str:
        verdict = (
            f"FITS, {self.headroom_bytes / MIB:.0f} MiB spare"
            if self.fits
            else f"does not fit, short {-self.headroom_bytes / MIB:.0f} MiB"
        )
        line = (
            f"  {self.method.value:<13} "
            f"{self.posts.device_total / MIB:>8.0f} MiB/card  {verdict}"
        )
        if self.reason:
            line += f"  [{self.reason}]"
        return line


@dataclass
class FeasibilityDecision:
    """The answer, with the arithmetic that produced it."""

    requested: TrainingMethod
    fits: bool
    machine: MachineResources
    model: ModelProfile
    spec: TrainingDemandSpec
    ladder: tuple[MethodOption, ...] = ()
    chosen_cards: tuple[str, ...] = ()
    per_card_bytes: int = 0
    message: str = ""
    remedies: tuple[str, ...] = field(default_factory=tuple)

    @property
    def option(self) -> Optional[MethodOption]:
        for candidate in self.ladder:
            if candidate.method is self.requested:
                return candidate
        return None

    def fitting_alternatives(self) -> tuple[MethodOption, ...]:
        return tuple(
            o for o in self.ladder if o.fits and o.method is not self.requested
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "requested_method": self.requested.value,
            "fits": self.fits,
            "per_card_mib": self.per_card_bytes // MIB,
            "cards": list(self.chosen_cards),
            "machine": self.machine.to_json(),
            "model": self.model.to_json(),
            "ladder": [o.to_json() for o in self.ladder],
            "message": self.message,
            "what_would_make_it_work": list(self.remedies),
        }

    def render(self) -> str:
        lines = [self.message, "", "Method ladder against this machine:"]
        lines.extend(o.render() for o in self.ladder)
        lines.append("")
        lines.append(
            "Machine: "
            + "; ".join(c.describe() for c in self.machine.cards)
            + f"; host RAM {self.machine.ram_available_bytes / MIB:.0f} MiB available "
            f"of {self.machine.ram_total_bytes / MIB:.0f} MiB; disk "
            f"{self.machine.disk_free_bytes / MIB:.0f} MiB free at "
            f"{self.machine.disk_path}"
        )
        if self.remedies:
            lines.append("")
            lines.append("What would make this work:")
            lines.extend(f"  - {r}" for r in self.remedies)
        return "\n".join(lines)


def _price(
    method: TrainingMethod,
    model: ModelProfile,
    spec: TrainingDemandSpec,
    machine: MachineResources,
) -> MethodOption:
    """Price one method, choosing the smallest card set that holds it."""
    cards = sorted(machine.cards, key=lambda c: c.available_bytes, reverse=True)
    if not cards:
        return MethodOption(
            method=method,
            posts=compute_posts(model, method, spec),
            world_size=0,
            fits=False,
            reason="no CUDA device is visible to this server",
        )

    sharded = method is TrainingMethod.FULL_OFFLOAD
    best: Optional[MethodOption] = None
    for world in range(1, len(cards) + 1):
        # Sharded methods split the state across the group, so more cards can
        # turn a "does not fit" into a "fits". Replicated ones cannot, but a
        # bigger group is still priced so the report is honest about it.
        if not sharded and world > 1:
            break
        candidate_cards = cards[:world]
        trial = spec.with_world_size(world, sharded=sharded)
        posts = compute_posts(model, method, trial)
        need = posts.device_total
        limiting = min(candidate_cards, key=lambda c: c.available_bytes)
        headroom = limiting.available_bytes - need
        reason = ""
        if posts.host_offload > machine.ram_available_bytes:
            reason = (
                f"host RAM short by "
                f"{(posts.host_offload - machine.ram_available_bytes) / MIB:.0f} MiB"
            )
        # Disk is priced per rung, not once for the request. A full finetune
        # checkpoints the whole model and its optimizer state while a LoRA
        # checkpoints a few hundred megabytes, so "does it fit" has a
        # different answer per rung and the ladder has to say so.
        disk_needed = (
            checkpoint_bytes(model, method, trial) * trial.checkpoints_retained
        )
        if not reason and disk_needed > machine.disk_free_bytes:
            reason = (
                f"disk short by "
                f"{(disk_needed - machine.disk_free_bytes) / MIB:.0f} MiB for "
                f"{trial.checkpoints_retained} checkpoint(s) at "
                f"{machine.disk_path}"
            )
        option = MethodOption(
            method=method,
            posts=posts,
            world_size=world,
            fits=headroom >= 0 and not reason,
            limiting_card=limiting.uuid,
            headroom_bytes=headroom,
            reason=reason,
        )
        if option.fits:
            return option
        if best is None or option.headroom_bytes > best.headroom_bytes:
            best = option
    assert best is not None
    return best


def evaluate(
    *,
    model: ModelProfile,
    method: TrainingMethod,
    spec: TrainingDemandSpec,
    machine: MachineResources,
) -> FeasibilityDecision:
    """Decide, and build the ladder either way.

    The ladder is computed on success too. A job that fits should still be
    able to tell its submitter that QLoRA would have left 9 GiB more headroom
    for the serving tenant, which is a real decision on a shared rig.
    """
    ladder = tuple(
        sorted(
            (_price(rung, model, spec, machine) for rung in LADDER),
            key=lambda o: (o.posts.device_total, LADDER.index(o.method)),
        )
    )
    chosen = next((o for o in ladder if o.method is method), None)
    assert chosen is not None

    cards = sorted(machine.cards, key=lambda c: c.available_bytes, reverse=True)
    chosen_cards = tuple(c.uuid for c in cards[: max(1, chosen.world_size)])

    if not machine.cards:
        message = (
            "No CUDA device is visible to this server, so no training job can "
            "be scheduled."
        )
        if machine.probe_error:
            message += f" NVML probe failed: {machine.probe_error}."
        return FeasibilityDecision(
            requested=method,
            fits=False,
            machine=machine,
            model=model,
            spec=spec,
            ladder=ladder,
            chosen_cards=(),
            per_card_bytes=chosen.posts.device_total,
            message=message,
            remedies=(
                "start this server on a host with a CUDA device",
                "unset CUDA_VISIBLE_DEVICES=99 if this is a CPU-only test process",
            ),
        )

    if chosen.fits:
        return FeasibilityDecision(
            requested=method,
            fits=True,
            machine=machine,
            model=model,
            spec=spec,
            ladder=ladder,
            chosen_cards=chosen_cards,
            per_card_bytes=chosen.posts.device_total,
            message=(
                f"{method.value} on {model.params / 1e9:.2f}B parameters needs "
                f"{chosen.posts.device_total / MIB:.0f} MiB per card across "
                f"{chosen.world_size} card(s); the tightest card has "
                f"{chosen.headroom_bytes / MIB:.0f} MiB to spare."
            ),
        )

    remedies: list[str] = []
    if chosen.reason.startswith("disk short"):
        one = checkpoint_bytes(model, method, spec)
        message = (
            f"{method.value} on {model.params / 1e9:.2f}B parameters would keep "
            f"{spec.checkpoints_retained} checkpoint(s) of {one / MIB:.0f} MiB "
            f"each, and {chosen.reason}."
        )
        remedies.append(
            f"free space at {machine.disk_path}, or point --training-artifact-root "
            "at a larger filesystem"
        )
        remedies.append(f"lower checkpoints_retained from {spec.checkpoints_retained}")
    else:
        limiting = next(
            (c for c in machine.cards if c.uuid == chosen.limiting_card), cards[0]
        )
        message = (
            f"{method.value} on {model.params / 1e9:.2f}B parameters needs "
            f"{chosen.posts.device_total / MIB:.0f} MiB per card "
            f"({chosen.posts.render()}), but the largest claimable budget is "
            f"{limiting.available_bytes / MIB:.0f} MiB on "
            f"{limiting.name} ({limiting.uuid}) -- short by "
            f"{-chosen.headroom_bytes / MIB:.0f} MiB."
        )
        if chosen.reason:
            message += f" {chosen.reason}."

    alternatives = tuple(o for o in ladder if o.fits and o.method is not method)
    if alternatives:
        cheapest = alternatives[0]
        remedies.append(
            f"submit the job with method {cheapest.method.value!r}: it needs "
            f"{cheapest.posts.device_total / MIB:.0f} MiB per card and fits with "
            f"{cheapest.headroom_bytes / MIB:.0f} MiB to spare"
        )
    else:
        remedies.append(
            "no rung of the ladder fits at this sequence length and batch size; "
            f"halving sequence_length ({spec.sequence_length}) removes roughly "
            f"{(chosen.posts.activations + chosen.posts.logits) / 2 / MIB:.0f} MiB"
        )
    if spec.micro_batch_size > 1:
        remedies.append(
            f"lower micro_batch_size from {spec.micro_batch_size} to 1 and raise "
            "gradient accumulation instead"
        )
    if not spec.gradient_checkpointing:
        remedies.append(
            "enable gradient_checkpointing: it trades compute for roughly "
            f"{(chosen.posts.activations * 0.9) / MIB:.0f} MiB of activations"
        )

    return FeasibilityDecision(
        requested=method,
        fits=False,
        machine=machine,
        model=model,
        spec=spec,
        ladder=ladder,
        chosen_cards=chosen_cards,
        per_card_bytes=chosen.posts.device_total,
        message=message,
        remedies=tuple(remedies),
    )


def default_artifact_root() -> Path:
    """Where uploads, configs and adapters go. Overridable by env."""
    raw = os.environ.get("HTSGLANG_TRAINING_ROOT")
    if raw:
        return Path(raw)
    return Path(os.environ.get("XDG_CACHE_HOME", "/var/tmp")) / "htsglang" / "training"


def round_up_mib(value: float) -> int:
    return int(math.ceil(value / MIB)) * MIB
