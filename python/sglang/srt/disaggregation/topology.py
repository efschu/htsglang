# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Free placement of the PD-disaggregation topology (#107).

Stock PD offers exactly one placement: whatever GPUs each of the two server
launches happens to see. This module makes the topology a first-class,
validated choice:

``disjoint``
    The prefill server runs on its own cards, named by
    ``--disaggregation-prefill-gpus``. An uneven prefill group (e.g. a 3:1
    layer split across two cards) is expressed with
    ``--disaggregation-prefill-layer-split`` and mapped onto the existing
    uneven-PP machinery (``SGLANG_PP_LAYER_PARTITION``,
    ``distributed/utils.py::get_pp_indices``).

``colocated-congruent``
    A card keeps its decode rank AND serves the prefill lane from the SAME
    process, computing with the decode rank's resident weight shard. The
    shard assignment of the lane is congruent with the decode assignment BY
    CONSTRUCTION, so both roles read the same tensor objects: co-location
    costs activations and scratch, not a second model copy. Because the two
    roles are lanes of ONE process there is one NCCL rank per card — the
    NCCL >= 2.30 co-location threshold and the MPS daemon requirement of
    two-process sharing do not apply.

``colocated-process``
    A separate prefill process (holding its own layer slice, hence
    ``--disaggregation-prefill-layer-split`` is mandatory) runs on a card
    that also hosts a decode rank. Two processes on one physical GPU need an
    NCCL runtime >= 2.30 and, for real concurrency, a CUDA MPS daemon; both
    are probed (``rigmon.capabilities``) and a missing prerequisite rejects
    the launch with the probe's reason instead of a late NCCL failure.

Every topology passes a boot-time VRAM feasibility check: per affected card,
decode claim + prefill weight slice + prefill budget must fit into the NVML
total. An infeasible plan is rejected naming the card, each item, the sum and
the capacity — not discovered as a runtime OOM. There is no additional safety
margin and no implicit context-overhead subtraction: the named items are the
whole sum, headroom is the operator's responsibility.

Without any of the ``--disaggregation-topology*`` flags this module is inert:
``apply_pd_topology`` returns before touching anything.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

TOPOLOGY_DISJOINT = "disjoint"
TOPOLOGY_COLOCATED_CONGRUENT = "colocated-congruent"
TOPOLOGY_COLOCATED_PROCESS = "colocated-process"
TOPOLOGY_CHOICES = (
    TOPOLOGY_DISJOINT,
    TOPOLOGY_COLOCATED_CONGRUENT,
    TOPOLOGY_COLOCATED_PROCESS,
)


# ---------------------------------------------------------------------------
# Plan data model
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CardPlan:
    """Everything the feasibility check needs to know about one card."""

    gpu_id: int
    total_mib: Optional[int]
    #: Decode-side claim on this card: the sum of the FULL per-rank budgets
    #: (weights + KV + scratch, i.e. --rank-gpu-memory-mib or
    #: mem_fraction_static * total) of every decode rank placed here.
    decode_claim_mib: Optional[int]
    decode_ranks: Tuple[int, ...]
    #: Prefill weight bytes RESIDENT on this card. 0 under congruent sharing
    #: (the lane reads the decode shard) and on cards without a prefill role.
    prefill_weight_mib: int
    prefill_layers: Optional[int]
    #: Activation/scratch budget of the prefill role on this card. This is
    #: the ENTIRE prefill-side budget item — no utilization ceiling or
    #: safety factor is applied on top of it.
    prefill_budget_mib: int
    #: True when prefill and decode read the same weight tensors here.
    weights_shared: bool

    def known(self) -> bool:
        return self.total_mib is not None and (
            self.decode_claim_mib is not None or not self.decode_ranks
        )

    def sum_mib(self) -> int:
        return (
            (self.decode_claim_mib or 0)
            + self.prefill_weight_mib
            + self.prefill_budget_mib
        )


@dataclasses.dataclass(frozen=True)
class PDTopologyPlan:
    topology: str
    cards: Tuple[CardPlan, ...]

    def card(self, gpu_id: int) -> CardPlan:
        for c in self.cards:
            if c.gpu_id == gpu_id:
                return c
        raise KeyError(gpu_id)


# ---------------------------------------------------------------------------
# Structural validation (no host access, pure server-args logic)
# ---------------------------------------------------------------------------


def _topology_flags(args: Any) -> Tuple[Any, Any, Any, Any]:
    return (
        getattr(args, "disaggregation_topology", None),
        getattr(args, "disaggregation_prefill_gpus", None),
        getattr(args, "disaggregation_prefill_layer_split", None),
        getattr(args, "disaggregation_prefill_budget_mib", None),
    )


def topology_requested(args: Any) -> bool:
    return getattr(args, "disaggregation_topology", None) is not None


def decode_gpu_of_ranks(args: Any) -> List[int]:
    """CUDA device index per decode-side TP rank, single node.

    --rank-gpu-id wins (explicit placement, may co-locate ranks); otherwise
    the classic base_gpu_id/gpu_id_step formula. Mirrors
    ServerArgs.gpu_id_for_rank for the pure-TP case.
    """
    rank_gpu_id = getattr(args, "rank_gpu_id", None)
    if rank_gpu_id is not None:
        return list(rank_gpu_id)
    base = getattr(args, "base_gpu_id", 0)
    step = getattr(args, "gpu_id_step", 1)
    return [base + r * step for r in range(args.tp_size)]


def validate_pd_topology_args(args: Any) -> None:
    """Reject invalid topology flag combinations early and with the reason.

    Operates only on server args — host state (NVML totals, NCCL runtime,
    MPS) is checked separately so this stays unit-testable.
    """
    topology, prefill_gpus, layer_split, budget_mib = _topology_flags(args)

    if topology is None:
        stray = [
            name
            for name, value in (
                ("--disaggregation-prefill-gpus", prefill_gpus),
                ("--disaggregation-prefill-layer-split", layer_split),
                ("--disaggregation-prefill-budget-mib", budget_mib),
            )
            if value is not None
        ]
        if stray:
            raise ValueError(
                f"{', '.join(stray)} require(s) --disaggregation-topology; "
                "without it the flags would be silently unused."
            )
        return

    if topology not in TOPOLOGY_CHOICES:
        raise ValueError(
            f"--disaggregation-topology={topology!r} is not one of "
            f"{TOPOLOGY_CHOICES}."
        )

    if prefill_gpus is None or len(prefill_gpus) == 0:
        raise ValueError(
            f"--disaggregation-topology={topology} requires "
            "--disaggregation-prefill-gpus (the cards carrying the prefill "
            "role)."
        )
    if any(g < 0 for g in prefill_gpus):
        raise ValueError(
            f"--disaggregation-prefill-gpus={prefill_gpus} contains a "
            "negative device index."
        )
    if len(set(prefill_gpus)) != len(prefill_gpus):
        raise ValueError(
            f"--disaggregation-prefill-gpus={prefill_gpus} contains "
            "duplicates; each card carries the prefill role at most once."
        )

    if budget_mib is None or budget_mib <= 0:
        raise ValueError(
            f"--disaggregation-topology={topology} requires "
            "--disaggregation-prefill-budget-mib > 0: the prefill side's "
            "activation/scratch budget per card is an explicit item of the "
            "boot-time VRAM check, not something the code guesses."
        )

    if layer_split is not None:
        if len(layer_split) != len(prefill_gpus):
            raise ValueError(
                f"--disaggregation-prefill-layer-split has "
                f"{len(layer_split)} entries but "
                f"--disaggregation-prefill-gpus names "
                f"{len(prefill_gpus)} cards; one layer count per card."
            )
        if any(n < 1 for n in layer_split):
            raise ValueError(
                f"--disaggregation-prefill-layer-split={layer_split} "
                "contains a count < 1."
            )

    if getattr(args, "dp_size", 1) > 1 or getattr(args, "ep_size", 1) > 1:
        raise ValueError(
            "--disaggregation-topology supports single-node TP/DCP servers "
            f"only (dp_size={getattr(args, 'dp_size', 1)}, "
            f"ep_size={getattr(args, 'ep_size', 1)})."
        )

    mode = getattr(args, "disaggregation_mode", "null")

    if topology == TOPOLOGY_COLOCATED_CONGRUENT:
        if mode != "null":
            raise ValueError(
                "--disaggregation-topology=colocated-congruent runs the "
                "prefill lane INSIDE the decode server's processes — there "
                "are no two servers, so --disaggregation-mode must stay "
                f"unset (got {mode!r})."
            )
        if layer_split is not None:
            raise ValueError(
                "--disaggregation-prefill-layer-split does not apply to "
                "colocated-congruent: the lane computes with the decode "
                "sharding (that congruence is what makes the weights "
                "shared). Layer placement belongs to the disjoint and "
                "colocated-process topologies."
            )
        decode_gpus = set(decode_gpu_of_ranks(args))
        missing = [g for g in prefill_gpus if g not in decode_gpus]
        if missing:
            raise ValueError(
                "colocated-congruent: every prefill-lane card must host a "
                f"decode rank whose shard the lane reuses; GPU(s) {missing} "
                f"host none (decode ranks sit on {sorted(decode_gpus)})."
            )
        return

    # disjoint / colocated-process are two-server topologies.
    if mode not in ("prefill", "decode"):
        raise ValueError(
            f"--disaggregation-topology={topology} describes a two-server "
            "PD deployment; it requires --disaggregation-mode prefill|decode "
            f"(got {mode!r})."
        )

    if topology == TOPOLOGY_DISJOINT and mode != "prefill":
        raise ValueError(
            "--disaggregation-topology=disjoint places the PREFILL server; "
            "pass the topology flags to the prefill launch. The decode "
            "server keeps its own placement flags (--rank-gpu-id / "
            "--base-gpu-id) and needs no topology flag."
        )

    if topology == TOPOLOGY_COLOCATED_PROCESS:
        if layer_split is None:
            raise ValueError(
                "colocated-process requires "
                "--disaggregation-prefill-layer-split: a full prefill model "
                "copy next to the resident decode shard is exactly the VRAM "
                "explosion this topology exists to avoid — only a layer "
                "slice per shared card is admissible."
            )
        if mode == "decode":
            decode_gpus = set(decode_gpu_of_ranks(args))
            if not decode_gpus.intersection(prefill_gpus):
                raise ValueError(
                    "colocated-process: no card is shared between the "
                    f"prefill placement {list(prefill_gpus)} and the decode "
                    f"ranks on {sorted(decode_gpus)} — that is the disjoint "
                    "topology; use --disaggregation-topology=disjoint."
                )

    if layer_split is not None:
        # Layer-sliced prefill is full-width per card (pipeline stages).
        # Partial TP inside a layer slice x uneven DCP is the hierarchical-
        # parallelism track, deliberately not built here.
        if mode == "prefill" and getattr(args, "tp_size", 1) != 1:
            raise ValueError(
                "--disaggregation-prefill-layer-split places FULL-WIDTH "
                "layer slices (one pipeline stage per card); tp_size must "
                f"be 1 on the prefill server (got {args.tp_size}). "
                "Combining a layer split with partial TP is the "
                "hierarchical-parallelism track, not this flag."
            )
        pp_size = getattr(args, "pp_size", 1)
        if mode == "prefill" and pp_size not in (1, len(layer_split)):
            raise ValueError(
                f"pp_size={pp_size} conflicts with "
                f"--disaggregation-prefill-layer-split ({len(layer_split)} "
                "stages); leave --pp-size unset (it is derived) or set it "
                "to the number of stages."
            )


# ---------------------------------------------------------------------------
# Host inputs (NVML totals, model size, layer count) — all injectable
# ---------------------------------------------------------------------------


def nvml_card_totals_mib() -> Optional[Dict[int, int]]:
    """Total VRAM per CUDA-visible device index via NVML (no CUDA context).

    Indexing follows the current CUDA_VISIBLE_DEVICES view so the ids match
    the device indices used by the placement flags. Returns None when NVML
    is unusable; the caller must then say so instead of silently skipping.
    """
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
    except Exception as exc:  # pragma: no cover - host dependent
        logger.warning("PD topology: NVML unavailable (%s)", exc)
        return None
    try:
        count = pynvml.nvmlDeviceGetCount()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible.strip():
            try:
                order = [int(x) for x in visible.split(",") if x.strip()]
            except ValueError:
                order = list(range(count))
            order = [i for i in order if 0 <= i < count]
        else:
            order = list(range(count))
        totals: Dict[int, int] = {}
        for cuda_idx, phys_idx in enumerate(order):
            handle = pynvml.nvmlDeviceGetHandleByIndex(phys_idx)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            totals[cuda_idx] = int(info.total // (1024 * 1024))
        return totals
    except Exception as exc:  # pragma: no cover - host dependent
        logger.warning("PD topology: NVML query failed (%s)", exc)
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def estimate_model_weight_mib(model_path: str) -> Optional[int]:
    """Checkpoint weight footprint in MiB from file sizes (no torch, no GPU).

    safetensors index metadata wins; otherwise the sizes of the weight files
    themselves. Quantized checkpoints are measured as stored — the loaded
    footprint may differ after repacking, which is why the result feeds an
    itemized feasibility ESTIMATE, not an allocation.
    """
    if not model_path or not os.path.isdir(model_path):
        return None
    index = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index):
        try:
            with open(index) as f:
                meta = json.load(f).get("metadata", {})
            size = meta.get("total_size")
            if size:
                return int(int(size) // (1024 * 1024))
        except Exception:
            pass
    total = 0
    for name in os.listdir(model_path):
        if name.endswith((".safetensors", ".gguf", ".bin", ".pt")):
            try:
                total += os.path.getsize(os.path.join(model_path, name))
            except OSError:
                pass
    return int(total // (1024 * 1024)) if total else None


def read_num_hidden_layers(model_path: str) -> Optional[int]:
    if not model_path:
        return None
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception:
        return None
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        value = cfg.get(key) or cfg.get("text_config", {}).get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


# ---------------------------------------------------------------------------
# Planning + feasibility
# ---------------------------------------------------------------------------


def _decode_rank_budgets_mib(
    args: Any, card_totals: Optional[Mapping[int, int]], decode_gpus: Sequence[int]
) -> Optional[List[int]]:
    """Full per-rank decode claim (weights + KV + scratch) in MiB, or None
    when the launch flags do not pin it down."""
    budget = getattr(args, "rank_gpu_memory_mib", None)
    if budget is not None:
        if isinstance(budget, int):
            return [budget] * len(decode_gpus)
        return list(budget)
    fraction = getattr(args, "mem_fraction_static", None)
    if fraction is not None and card_totals is not None:
        out = []
        for gpu in decode_gpus:
            total = card_totals.get(gpu)
            if total is None:
                return None
            out.append(int(total * float(fraction)))
        return out
    return None


def plan_pd_topology(
    *,
    topology: str,
    prefill_gpus: Sequence[int],
    prefill_layer_split: Optional[Sequence[int]],
    prefill_budget_mib: int,
    decode_rank_gpus: Sequence[int],
    decode_rank_budgets_mib: Optional[Sequence[int]],
    card_totals_mib: Optional[Mapping[int, int]],
    model_weight_mib: Optional[int],
    num_hidden_layers: Optional[int],
) -> PDTopologyPlan:
    """Pure per-card accounting; all host inputs are parameters.

    The prefill weight slice is the layer share of the checkpoint footprint
    (uniform per-layer approximation; embeddings/lm_head fold into it). With
    congruent sharing it is 0 by construction — the lane reads the decode
    shard that is resident anyway.
    """
    if prefill_layer_split is not None and num_hidden_layers is not None:
        if sum(prefill_layer_split) != num_hidden_layers:
            raise ValueError(
                f"--disaggregation-prefill-layer-split sums to "
                f"{sum(prefill_layer_split)} layers but the model has "
                f"{num_hidden_layers}."
            )

    cards: Dict[int, Dict[str, Any]] = {}

    def entry(gpu: int) -> Dict[str, Any]:
        return cards.setdefault(
            gpu,
            {
                "decode_ranks": [],
                "decode_claim": None,
                "prefill_weight": 0,
                "prefill_layers": None,
                "prefill_budget": 0,
                "shared": False,
            },
        )

    for rank, gpu in enumerate(decode_rank_gpus):
        e = entry(gpu)
        e["decode_ranks"].append(rank)
        if decode_rank_budgets_mib is not None:
            e["decode_claim"] = (e["decode_claim"] or 0) + int(
                decode_rank_budgets_mib[rank]
            )

    for i, gpu in enumerate(prefill_gpus):
        e = entry(gpu)
        e["prefill_budget"] = int(prefill_budget_mib)
        if topology == TOPOLOGY_COLOCATED_CONGRUENT:
            e["shared"] = True
            continue
        if prefill_layer_split is not None:
            layers = int(prefill_layer_split[i])
            e["prefill_layers"] = layers
            if model_weight_mib is not None and num_hidden_layers:
                e["prefill_weight"] = int(
                    model_weight_mib * layers / num_hidden_layers
                )
        elif model_weight_mib is not None:
            # Even TP prefill group: equal shard per card.
            e["prefill_weight"] = int(model_weight_mib / len(prefill_gpus))

    card_plans = tuple(
        CardPlan(
            gpu_id=gpu,
            total_mib=(card_totals_mib or {}).get(gpu),
            decode_claim_mib=e["decode_claim"],
            decode_ranks=tuple(e["decode_ranks"]),
            prefill_weight_mib=e["prefill_weight"],
            prefill_layers=e["prefill_layers"],
            prefill_budget_mib=e["prefill_budget"],
            weights_shared=e["shared"],
        )
        for gpu, e in sorted(cards.items())
    )
    return PDTopologyPlan(topology=topology, cards=card_plans)


def check_vram_feasibility(plan: PDTopologyPlan) -> List[str]:
    """Boot-time physical check: per card, the itemized sum must fit the
    NVML total. Infeasible -> ValueError naming card, items, sum, capacity.

    Cards whose inputs could not be determined are returned as warnings (the
    caller logs them) — skipped loudly, never silently. No safety margin is
    added and nothing is subtracted for context overhead: the items ARE the
    whole sum, headroom stays the operator's call.
    """
    problems: List[str] = []
    unknown: List[str] = []
    for card in plan.cards:
        items = []
        if card.decode_ranks:
            claim = (
                f"{card.decode_claim_mib} MiB"
                if card.decode_claim_mib is not None
                else "UNKNOWN"
            )
            ranks = ",".join(str(r) for r in card.decode_ranks)
            items.append(f"decode claim {claim} (ranks {ranks})")
        if card.weights_shared:
            items.append("prefill weights 0 MiB (congruent, shares the decode shard)")
        elif card.prefill_weight_mib or card.prefill_layers is not None:
            layers = (
                f" ({card.prefill_layers} layers)"
                if card.prefill_layers is not None
                else ""
            )
            items.append(f"prefill weight slice {card.prefill_weight_mib} MiB{layers}")
        if card.prefill_budget_mib:
            items.append(f"prefill budget {card.prefill_budget_mib} MiB")
        if not card.known():
            unknown.append(
                f"GPU {card.gpu_id}: feasibility not computable "
                f"(total={card.total_mib}, decode claim="
                f"{card.decode_claim_mib}); items: " + "; ".join(items)
            )
            continue
        total = card.total_mib
        assert total is not None
        s = card.sum_mib()
        if s > total:
            problems.append(
                f"GPU {card.gpu_id} (total {total} MiB): "
                + " + ".join(items)
                + f" = {s} MiB > {total} MiB"
            )
    if problems:
        raise ValueError(
            "PD topology infeasible — the per-card sums do not fit:\n  "
            + "\n  ".join(problems)
        )
    return unknown


# ---------------------------------------------------------------------------
# Host prerequisites for two processes on one physical GPU
# ---------------------------------------------------------------------------


def check_process_colocation_prerequisites(probe_env: Any = None) -> None:
    """colocated-process gate: NCCL runtime >= 2.30 and a reachable MPS
    daemon, decided by the rigmon probes. A missing prerequisite rejects
    with the probe's reason sentence instead of a late NCCL failure.

    colocated-congruent deliberately does NOT come through here: one
    process per card means one NCCL rank per card, so neither prerequisite
    applies.
    """
    from sglang.srt.rigmon.capabilities import (
        UNAVAILABLE,
        ProbeEnv,
        probe_mps,
        probe_nccl_colocation,
    )

    env = probe_env if probe_env is not None else ProbeEnv()
    failures = []
    for probe in (probe_nccl_colocation, probe_mps):
        cap = probe(env)
        if cap.state == UNAVAILABLE:
            failures.append(f"{cap.label}: {cap.reason}")
    if failures:
        raise ValueError(
            "--disaggregation-topology=colocated-process puts a prefill "
            "process and a decode process on the same physical GPU, and "
            "this host cannot do that:\n  " + "\n  ".join(failures)
        )


# ---------------------------------------------------------------------------
# Orchestration (called from the PD-disaggregation hook)
# ---------------------------------------------------------------------------


def apply_pd_topology(
    server_args: Any,
    *,
    probe_env: Any = None,
    card_totals_mib: Optional[Mapping[int, int]] = None,
    model_weight_mib: Optional[int] = None,
    num_hidden_layers: Optional[int] = None,
    setenv: Optional[Dict[str, str]] = None,
) -> Optional[PDTopologyPlan]:
    """Validate, gate and normalize the requested PD topology.

    Returns None untouched-fast when no topology is requested (the default
    path stays byte-identical). The keyword inputs exist for tests; the
    hook calls with defaults and the host is probed for real.
    """
    if not topology_requested(server_args):
        # Still reject stray companion flags.
        validate_pd_topology_args(server_args)
        return None

    validate_pd_topology_args(server_args)

    topology, prefill_gpus, layer_split, budget_mib = _topology_flags(server_args)
    env: Dict[str, str] = setenv if setenv is not None else os.environ  # type: ignore[assignment]

    if topology == TOPOLOGY_COLOCATED_PROCESS:
        check_process_colocation_prerequisites(probe_env)

    mode = getattr(server_args, "disaggregation_mode", "null")
    decode_gpus = (
        decode_gpu_of_ranks(server_args) if mode in ("null", "decode") else []
    )

    if card_totals_mib is None:
        card_totals_mib = nvml_card_totals_mib()
    if model_weight_mib is None:
        model_weight_mib = estimate_model_weight_mib(
            getattr(server_args, "model_path", "") or ""
        )
    if num_hidden_layers is None:
        num_hidden_layers = read_num_hidden_layers(
            getattr(server_args, "model_path", "") or ""
        )

    plan = plan_pd_topology(
        topology=topology,
        prefill_gpus=prefill_gpus,
        prefill_layer_split=layer_split,
        prefill_budget_mib=budget_mib,
        decode_rank_gpus=decode_gpus,
        decode_rank_budgets_mib=_decode_rank_budgets_mib(
            server_args, card_totals_mib, decode_gpus
        ),
        card_totals_mib=card_totals_mib,
        model_weight_mib=model_weight_mib,
        num_hidden_layers=num_hidden_layers,
    )
    for warning in check_vram_feasibility(plan):
        logger.warning("PD topology: %s", warning)

    # Process-level isolation: every scheduler process sees exactly its own
    # GPU via CUDA_VISIBLE_DEVICES (maybe_reindex_device_id) — no in-process
    # logical/physical mapping tables.
    env.setdefault("SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS", "1")

    if mode == "prefill":
        if layer_split is not None:
            # Uneven layer placement rides the existing uneven-PP machinery.
            server_args.pp_size = len(layer_split)
            env["SGLANG_PP_LAYER_PARTITION"] = ",".join(
                str(n) for n in layer_split
            )
        if "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in prefill_gpus)
            logger.info(
                "PD topology: prefill server restricted to CUDA devices %s",
                env["CUDA_VISIBLE_DEVICES"],
            )

    if topology == TOPOLOGY_COLOCATED_CONGRUENT:
        raise NotImplementedError(
            "--disaggregation-topology=colocated-congruent: the in-process "
            "prefill lane is not wired yet. The topology plan, congruence "
            "validation and VRAM feasibility check above are in place; the "
            "lane forward (scheduler slice) is the next stage and needs GPU "
            "validation. Until then use 'disjoint' or 'colocated-process'."
        )

    logger.info(
        "PD topology '%s': %s",
        topology,
        "; ".join(
            f"GPU {c.gpu_id}: decode ranks {list(c.decode_ranks)}, "
            f"prefill {'shared shard' if c.weights_shared else (str(c.prefill_layers) + ' layers' if c.prefill_layers is not None else ('slice' if c.prefill_weight_mib else 'none'))}, "
            f"sum {c.sum_mib()} MiB / total {c.total_mib} MiB"
            for c in plan.cards
        ),
    )
    return plan
