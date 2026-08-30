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
"""The serving engine's own ledger lines.

Every term below is produced by calling the EXISTING derivation and never by
restating it. ``mamba_pre_capture_reserve_mb``, ``speculative_capture_tokens``,
``gdn_prefill_scratch_mib``, ``dsv4_indexer_prefill_scratch_mib``,
``ladder_reserve_demand`` and ``mamba_pool_floor.mamba_hard_floor`` stay where
they are and stay the single source of truth for their quantity; this module
turns what they return into ledger rows with a provenance and a derivation
string. The reason is the #56/M21 lesson in a new place: a second copy of a
formula does not disagree loudly, it disagrees silently and only on the
configuration nobody tested.

TWO CORRECTIONS THIS MODULE MAKES to the demand model it inherits, both stated
here rather than buried:

1. THE ACTIVATION TERM IS PER RANK, NOT PER CARD. The #68 derivation scaled the
   graph-capture term by the co-located rank count and left the
   runtime/activation reserve shared per GPU ("as before"). Two ranks on one
   card are two processes; each runs its own prefill and each holds its own
   activation peak simultaneously with the other. Sharing that term
   under-charges exactly the co-located configuration it was introduced for.
   The ledger charges it per rank.

2. THE PREFILL SCRATCH TERMS ARE CHARGED, NOT ONLY MENTIONED. The GDN and
   DSV4-indexer prefill scratch were computed for a WARNING and then not added
   to anything -- which is why #493 could watch a card fall to 271 MiB free
   while every reserve looked healthy. They are ledger lines here, and each one
   carries the MECHANISM that caps it, because a transient is capped where it
   is allocated and never by a budget line (#493).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.mem_ledger.terms import (
    CardVramLedger,
    LedgerError,
    LedgerTerm,
    Provenance,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sglang.srt.mem_ledger.nccl_transport import CommunicatorGroup

logger = logging.getLogger(__name__)

__all__ = [
    "CardFacts",
    "DemandInputs",
    "build_card_ledgers",
    "TERM_WEIGHTS",
    "TERM_ACTIVATION",
    "TERM_GRAPH_CAPTURE",
    "TERM_LADDER",
    "TERM_GDN_SCRATCH",
    "TERM_INDEXER_SCRATCH",
    "TERM_MAMBA_POOL",
    "TERM_HARDWARE_RESIDUAL",
    "TERM_PARENT_CONTEXT",
    "TERM_ATTN_WORKSPACE",
    "BUDGET_FUNDED_TERMS",
    "TERM_NCCL_BUFFERS",
    "TERM_LOAD_TRANSIENT",
    "TERM_MOE_OFFLOAD_SCRATCH",
    "LOAD_TRANSIENT_REFERENCE_MIB",
    "LOAD_TRANSIENT_REFERENCE_TAG",
    "RUNTIME_COMMUNICATOR_GROUPS",
    "communicator_groups_from_server_args",
    "demand_outside_budget_mib",
]

TERM_WEIGHTS = "model weights (shards)"
TERM_ACTIVATION = "runtime activation + metadata"
TERM_GRAPH_CAPTURE = "CUDA graph capture"
TERM_LADDER = "adaptive draft ladder"
TERM_GDN_SCRATCH = "GDN prefill scratch"
TERM_INDEXER_SCRATCH = "DSV4 indexer prefill scratch"
TERM_MAMBA_POOL = "mamba/GDN state pool"
TERM_HARDWARE_RESIDUAL = "hardware residual (per process)"
TERM_PARENT_CONTEXT = "parent/tokenizer CUDA context"
TERM_NCCL_BUFFERS = "NCCL communicator buffers"
TERM_ATTN_WORKSPACE = "attention workspaces (capped)"
TERM_NVML_CARVE_OUT = "NVML driver carve-out (not allocatable)"
TERM_LOAD_TRANSIENT = "load transient (allocator peak over resident)"
TERM_MOE_OFFLOAD_SCRATCH = "MoE expert-offload spill scratch"

#: The load transient, MiB per rank process, as the 2026-08-06 corridor window
#: recorded it.
#:
#: PROVENANCE, STATED IN FULL BECAUSE THIS NUMBER IS INHERITED AND NOT MEASURED
#: BY ANYTHING IN THIS TREE. It comes from the #602 corridor acceptance runs
#: (docs/rig-runbook.md section 16.6, scripts/dev/602_corridor/README.md), which
#: sampled per-card NVML free at 10 Hz first at idle and then under serving
#: load, and found the load MINIMUM sitting up to ~70 MiB below the idle
#: reading on every card. That dip is an allocator peak above the resident set
#: that no ledger term charged: every other term is either resident (weights,
#: pools, workspaces) or already priced as its own transient (activation,
#: prefill scratch), and the corridor floor was still undershot by this much.
#:
#: WHAT IT IS NOT. It is not an attribution to one allocation site. The same
#: allocator family produces a peak during WEIGHT LOAD as well -- per-tensor
#: host-to-device staging plus the quantization repack in
#: ``model_loader/loader.py:867-899`` (``load_weights_and_postprocess``), where
#: each ``process_weights_after_loading`` holds a repacked copy beside the
#: original -- and that peak has never been measured separately on this rig.
#: The number below is therefore charged as ONE per-rank allocator transient
#: covering the phase where the peak is largest, not as a sum of sites.
#:
#: HOW IT STOPS BEING INHERITED. ``flight_recorder.mark`` now computes
#: ``allocator_transient_bytes`` (reserved peak minus reserved current) on every
#: mark, and ``reconcile`` compares this term against that field at the
#: ``weights_loaded`` mark. A boot with ``SGLANG_VRAM_FLIGHT_DIR`` set therefore
#: measures the quantity this constant stands in for; feeding the measurement
#: back through ``DemandInputs.load_transient_mib_per_rank`` replaces the
#: constant with a CALIBRATED per-rank number carrying the rig fingerprint.
#: #584: filed in measured.CALIBRATION_QUEUE -- known-unmeasured, awaiting a
#: recorder post. Do not recalibrate this from a desk.
LOAD_TRANSIENT_REFERENCE_MIB = 70


class _LoadTransientFallback:
    """Once-per-process latch for the inherited-constant warning below.

    A class attribute rather than a module global so a test can reset it
    without reaching into this module's namespace, and so the module-state
    ratchet stays green.
    """

    announced = False

#: The fingerprint the inherited number above carries. Deliberately a WINDOW
#: tag and not a rig fingerprint: it can never equal ``live_fingerprint()``, so
#: the ledger row renders as ``calibrated@window-2026-`` and a reader can see at
#: a glance that this line was not measured on the card in front of him. The
#: alternative -- borrowing the live fingerprint for a number the live rig never
#: produced -- would be an un-invalidatable literal wearing a measurement's
#: label, which is the exact defect this module exists to remove.
LOAD_TRANSIENT_REFERENCE_TAG = "window-2026-08-06"

#: Terms the RANK BUDGET already funds, i.e. terms that live INSIDE
#: ``--rank-gpu-memory-mib`` rather than outside it.
#:
#: This distinction is not cosmetic and getting it wrong is a double charge.
#: A card's memory splits into "the rank budget" and "everything else"; the
#: profiling step then subtracts the weight shards and the SSM pool FROM the
#: budget before it sizes the KV pool. So those two terms are real card memory
#: (they belong in the ledger, and the ledger prints them) but they must not
#: also be subtracted when the budget itself is being formed -- that would
#: reserve them twice and shrink the KV pool by their size for nothing.
#:
#: Everything not named here is genuinely outside the budget: the CUDA context
#: and the allocator residue exist before the allocator's first tensor, the
#: capture pool and the workspaces are allocated after the KV pool is sized,
#: and the prefill scratch lands on top of both at runtime.
BUDGET_FUNDED_TERMS = frozenset({TERM_WEIGHTS, TERM_MAMBA_POOL})

#: trtllm_mla's fixed private workspace. Quoted from
#: python/sglang/srt/layers/attention/trtllm_mla_backend.py, where
#: DEFAULT_WORKSPACE_SIZE_MB = 150 and workspace_size = that * 1024 * 1024.
#: Named here so the ledger row can cite the site instead of appearing to
#: invent a constant of its own.
TRTLLM_MLA_WORKSPACE_MIB = 150

#: The backends whose private workspace is a fixed constant. Anything not
#: here and not flashinfer becomes UNBOUNDED rather than zero.
_FIXED_BACKEND_WORKSPACE_MIB = {}

#: trtllm_mha's fallback workspace, from
#: python/sglang/srt/layers/attention/trtllm_mha_backend.py
#: (DEFAULT_WORKSPACE_SIZE_MB = 512, used unless
#: SGLANG_FLASHINFER_WORKSPACE_SIZE is set explicitly).
TRTLLM_MHA_WORKSPACE_MIB = 512

_FIXED_BACKEND_WORKSPACE_MIB.update(
    {"trtllm_mla": TRTLLM_MLA_WORKSPACE_MIB, "trtllm_mha": TRTLLM_MHA_WORKSPACE_MIB}
)

#: The stock per-captured-token graph coefficient, from
#: ``_handle_gpu_memory_settings``. Named rather than inlined so the ledger row
#: can say which existing constant it is quoting instead of appearing to invent
#: a factor of its own.
#: #584: filed in measured.CALIBRATION_QUEUE -- measured 3.3-3.8x LOW and
#: still shipped. Awaiting a per-rung post. Do not recalibrate from a desk.
GRAPH_MIB_PER_CAPTURED_TOKEN = 2


@dataclasses.dataclass(frozen=True)
class CardFacts:
    """Identity and size of one physical card.

    Separated from the derivation so the whole ledger is exercisable without
    NVML: a hermetic test constructs these, the boot path fills them from the
    #392 identity map (which resolves a ``--rank-gpu-id`` CUDA ordinal to the
    card it will actually bind, never positionally).
    """

    gpu_id: int
    uuid: str
    name: str
    total_mib: int
    #: MiB the driver holds back out of ``total_mib`` and never allocates, as
    #: NVML's v2 memory struct reports it. Booked as its own ledger term rather
    #: than quietly subtracted from ``total_mib``, so a card whose driver does
    #: not report it (0) is visibly different from one that reports none.
    reserved_mib: int = 0

    def describe(self) -> str:
        return f"GPU {self.gpu_id} ({self.name}, NVML total {self.total_mib} MiB)"


@dataclasses.dataclass(frozen=True)
class DemandInputs:
    """Everything the engine ledger needs, already derived.

    This dataclass is the SEAM. Production fills it from :meth:`from_server_args`
    (which calls the real formulas); tests fill it directly with numbers whose
    origin the test itself states. Both produce the identical ledger, so what a
    hermetic test proves about the ledger is true of the boot.

    A field that is ``None`` means "this quantity applies to this configuration
    but could not be derived here". It becomes an UNBOUNDED entry and refuses
    the boot -- deliberately not a zero, because a zero is indistinguishable
    from "does not apply" and that ambiguity is how a term goes missing.
    """

    #: Per-rank resident weight footprint, MiB, index = tp rank. Under uneven
    #: TP the entries differ; under even TP they are equal by construction.
    weight_mib_per_rank: Sequence[int]
    #: Per-rank runtime activation + metadata peak, MiB. An entry of ``None``
    #: means "this rank's activation peak is not calibrated for the live
    #: hardware and profile" and becomes an UNBOUNDED item, i.e. a refusal.
    #: It is emphatically NOT zero and never falls back to the inherited
    #: 512+tokens*1.5+tp*pp/8*1024 heuristic, which the 2026-08-05 window
    #: falsified (it books 3968 MiB where the binding card had 1766 MiB free
    #: and still completed a 70018-token prefill).
    activation_mib_per_rank: Sequence[Optional[float]]
    #: Per-rank captured tokens across all graph families. Used only when
    #: ``capture_mib_per_rank`` is absent -- the tokens x 2 MiB coefficient is
    #: itself an inherited estimate and the same window measured it 3.3-3.8x
    #: LOW (192 MiB booked against 633-730 MiB actually taken).
    capture_tokens_per_rank: Sequence[int]
    #: Per-rank measured graph-capture cost, MiB. Overrides the token estimate
    #: when present.
    capture_mib_per_rank: Optional[Sequence[float]] = None
    #: Per-rank provenance sentence for the activation and capture numbers.
    phase_footprint_source_per_rank: Sequence[str] = ()
    #: Hardware fingerprint the phase footprints were taken under. REQUIRED
    #: whenever an activation value is supplied: a calibrated number without
    #: the hardware it was measured on cannot be invalidated, and the ledger
    #: rejects such a term by construction. There is deliberately no "unknown"
    #: sentinel -- that would be the un-invalidatable literal wearing a label.
    phase_footprint_fingerprint: str = ""
    #: Per-rank mamba/GDN state pool, MiB. Empty tuple when the checkpoint has
    #: no such layers (which is "does not apply", not "unknown").
    mamba_pool_mib_per_rank: Sequence[float] = ()
    #: Per-rank allocator transient above the resident set, MiB. ``None`` means
    #: "no boot of this rig has measured it", and it does NOT mean zero: the
    #: term then charges :data:`LOAD_TRANSIENT_REFERENCE_MIB` per rank and says
    #: in its derivation that the number is inherited. A silent zero here is
    #: precisely the shape of the gap this term closes -- the #602 corridor
    #: floor was undershot by a transient that no row mentioned.
    load_transient_mib_per_rank: Optional[Sequence[float]] = None
    #: Per-rank GDN prefill scratch peak, MiB, or None when not applicable.
    gdn_scratch_mib_per_rank: Optional[Sequence[float]] = None
    #: Per-rank DSV4 C4-indexer prefill scratch peak, MiB, or None when the
    #: checkpoint has no indexer.
    indexer_scratch_mib_per_rank: Optional[Sequence[float]] = None
    #: Per-rank MoE expert-offload SPILL SCRATCH band, MiB, or None when this
    #: launch has no expert-offload lane (which is "does not apply", not
    #: "unknown"). It is the C extra GPU slots of the fixed-resident buffer,
    #: NOT the resident experts -- those are already inside
    #: ``weight_mib_per_rank``. See TERM_MOE_OFFLOAD_SCRATCH for why the
    #: reclaim is deliberately not credited back against it here.
    moe_scratch_mib_per_rank: Optional[Sequence[float]] = None
    #: ``{gpu_id: MiB}`` for the adaptive ladder, charged to the one GPU that
    #: hosts the solo draft rank.
    ladder_mib_per_gpu: Mapping[int, int] = dataclasses.field(default_factory=dict)
    #: Configuration values quoted in derivation strings. Names only, so a
    #: reader can look them up; the ledger never re-reads them.
    chunked_prefill_size: int = 0
    context_length: Optional[int] = None
    max_running_requests: Optional[int] = None
    mamba_floor_slots: Optional[int] = None
    mamba_floor_derivation: str = ""
    indexer_chunk_cap_mib: Optional[int] = None
    #: Per-rank attention workspace budgets that are CAPPED by a knob rather
    #: than derived from geometry: the flashinfer float workspace
    #: (SGLANG_FLASHINFER_WORKSPACE_SIZE) and the chunked-prefix attention
    #: scratch (--attn-scratch-budget-mib). Charged at the cap, because the cap
    #: is what the rank may take and the ledger must be able to fund the worst
    #: case the configuration permits.
    flashinfer_workspace_mib: Optional[int] = None
    attn_scratch_budget_mib: Optional[int] = None
    #: Why the flashinfer workspace is the size it is (which of the three rules
    #: applied). Carried into the ledger row so a reader can tell a 2048 MiB
    #: deterministic workspace from a 2048 MiB anything-else at a glance.
    flashinfer_workspace_note: str = ""
    #: Config inputs the workspace size actually depends on. Declared on the
    #: term so the "a MODELED term moves when its driver moves" test can reach
    #: them; this term charged a constant 384 MiB before they were wired in.
    enable_deterministic_inference: bool = False
    model_architectures: Tuple[str, ...] = ()
    #: True when a parent/tokenizer process binds a CUDA context on the cards
    #: (#237/#403). The SIZE of that context is a hardware residual.
    parent_binds_cuda_context: bool = False
    #: Active attention backend name, e.g. "flashinfer" or "trtllm_mla".
    #: Production always sets this from ServerArgs; "" means "not stated" and
    #: keeps the pre-#595 flashinfer-only accounting, so existing callers and
    #: the rig that runs flashinfer are byte-identical.
    attention_backend: str = ""
    #: ``{gpu id: measured NCCL communicator bytes, MiB}``. None until a boot
    #: measures it -- see TERM_NCCL_BUFFERS for why it cannot be derived.
    nccl_buffer_mib_per_gpu: Optional[Mapping[int, float]] = None
    #: What the measurement above is valid FOR: the communicator set this
    #: launch builds. A measured NCCL figure is only reusable while this is
    #: unchanged, which is why it is carried with the number instead of being
    #: keyed on the hardware fingerprint alone.
    nccl_signature: str = ""
    #: #598. The communicator groups this launch builds, as
    #: :class:`~sglang.srt.mem_ledger.nccl_transport.CommunicatorGroup`
    #: descriptions. ``None`` means "not stated" and keeps the pre-#598
    #: two-state term exactly (priced when measured, UNBOUNDED otherwise), so
    #: every existing caller is byte-identical. An EMPTY tuple is not the same
    #: thing: it is the positive statement that this launch builds no
    #: multi-rank group, and it resolves the term to NOT_APPLICABLE.
    communicator_groups: Optional[Sequence["CommunicatorGroup"]] = None

    def rank_count(self) -> int:
        return len(self.weight_mib_per_rank)

    def __post_init__(self) -> None:
        n = len(self.weight_mib_per_rank)
        for field_name in (
            "activation_mib_per_rank",
            "capture_tokens_per_rank",
        ):
            got = len(getattr(self, field_name))
            if got != n:
                raise LedgerError(
                    f"DemandInputs.{field_name} has {got} entries but "
                    f"{n} ranks are described; a per-rank term that does not "
                    "cover every rank leaves a card silently under-charged"
                )
        for field_name in (
            "mamba_pool_mib_per_rank",
            "gdn_scratch_mib_per_rank",
            "indexer_scratch_mib_per_rank",
            "load_transient_mib_per_rank",
        ):
            value = getattr(self, field_name)
            if value is not None and len(value) not in (0, n):
                raise LedgerError(
                    f"DemandInputs.{field_name} has {len(value)} entries but "
                    f"{n} ranks are described"
                )

    @classmethod
    def from_server_args(
        cls,
        server_args,
        *,
        rank_gpu_id: Sequence[int],
        gpu_total_mib: Mapping[int, int],
        weight_mib_per_rank: Sequence[int],
        head_share_per_rank: Optional[Sequence[float]] = None,
        mamba_pool_mib_per_rank: Optional[Sequence[float]] = None,
        card_uuid_by_gpu: Optional[Mapping[int, str]] = None,
    ) -> "DemandInputs":
        """Fill the inputs by calling the REAL ServerArgs derivations.

        ``weight_mib_per_rank`` and ``mamba_pool_mib_per_rank`` come from the
        cost model (``uneven_perf.PerfCostModel``) rather than from here: that
        model already owns the shard arithmetic and the SSM-pool arithmetic,
        and a second derivation of either is the duplication this module
        exists to avoid.
        """
        n = len(rank_gpu_id)
        shares = list(head_share_per_rank or [1.0 / max(n, 1)] * n)

        activation: List[Optional[float]] = []
        capture: List[int] = []
        gdn: List[float] = []
        indexer: List[float] = []
        gdn_applicable = True
        indexer_applicable = True
        # Phase footprints (activation peak, graph capture) come from the
        # calibration store, keyed by hardware fingerprint AND activation
        # profile. A miss yields None, which the builder turns into a refusal;
        # the inherited heuristics are never reached from here.
        from sglang.srt.mem_ledger.activation import (
            profile_from_server_args,
            resolve_phase_footprint,
        )
        from sglang.srt.mem_ledger.calibration import live_fingerprint

        live = live_fingerprint()
        hw_fp = live[0] if live else None
        capture_mib: List[float] = []
        sources: List[str] = []
        footprint_profile = None

        footprint_profile = profile_from_server_args(
            server_args, _model_architectures(server_args)
        )

        for rank, gpu_id in enumerate(rank_gpu_id):
            capture.append(int(server_args.speculative_capture_tokens()))
            uuid = (card_uuid_by_gpu or {}).get(gpu_id, "")
            fp = (
                resolve_phase_footprint(
                    uuid, hw_fingerprint=hw_fp, profile=footprint_profile
                )
                if uuid
                else None
            )
            if fp is None:
                activation.append(None)
                capture_mib.append(0.0)
                sources.append("")
            else:
                activation.append(float(fp.activation_mib))
                capture_mib.append(float(fp.capture_mib))
                sources.append(f"[{fp.provenance.value}] {fp.source}")
            scratch = server_args.gdn_prefill_scratch_mib(shares[rank])
            if scratch is None:
                gdn_applicable = False
            else:
                gdn.append(float(scratch))
            idx = server_args.dsv4_indexer_prefill_scratch_mib()
            if idx is None:
                indexer_applicable = False
            else:
                indexer.append(float(idx))

        ladder: Dict[int, int] = {}
        ladder_gpu = server_args.ladder_reserve_gpu_id()
        if ladder_gpu is not None:
            colocated = sum(1 for g in rank_gpu_id if g == ladder_gpu)
            demand = server_args.ladder_reserve_demand(colocated)
            if demand is not None and demand.total_mib:
                ladder[ladder_gpu] = int(demand.total_mib)

        floor_slots = None
        floor_note = ""
        try:
            from sglang.srt.mem_cache import mamba_pool_floor

            running = int(
                server_args.max_running_requests
                or server_args.cuda_graph_config.decode.max_bs
                or 1
            )
            floor_slots = mamba_pool_floor.mamba_hard_floor(server_args, running)
            floor_note = mamba_pool_floor.describe_mamba_floor(server_args, running)
        except Exception as e:  # pragma: no cover - config-source differences
            logger.debug("mamba pool floor unavailable for the ledger: %s", e)

        indexer_cap = None
        if indexer_applicable:
            try:
                from sglang.srt.environ import envs

                indexer_cap = int(envs.SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB.get())
            except Exception:  # pragma: no cover - env shape differences
                indexer_cap = None

        # The flashinfer workspace is NOT simply the env var: the backend
        # rewrites it during __init__ (512 MiB for listed architectures, then
        # 2048 MiB under deterministic inference, which overrides the first).
        # Reading the raw variable here charged 384 MiB for a deterministic
        # boot that allocates 2048 -- 1664 MiB the ledger never charged, and a
        # MODELED term that does not move with a config input it depends on.
        # resolve_flashinfer_workspace_mib is the SAME function the backend
        # uses, so the two cannot drift.
        flashinfer_mib = None
        deterministic = bool(
            getattr(server_args, "enable_deterministic_inference", False)
        )
        architectures: Tuple[str, ...] = ()
        try:
            from sglang.srt.layers.attention.flashinfer_workspace import (
                describe_flashinfer_workspace,
                resolve_flashinfer_workspace_mib,
            )

            architectures = tuple(_model_architectures(server_args))
            flashinfer_mib = resolve_flashinfer_workspace_mib(
                enable_deterministic_inference=deterministic,
                architectures=architectures,
            )
            flashinfer_note = describe_flashinfer_workspace(
                enable_deterministic_inference=deterministic,
                architectures=architectures,
            )
        except Exception as e:  # pragma: no cover - env/config shape differences
            logger.debug("flashinfer workspace size unavailable: %s", e)
            flashinfer_mib = None
            flashinfer_note = ""
        attn_scratch = getattr(server_args, "attn_scratch_budget_mib", None)

        # ------------------------------------------------------------------
        # #594. NCCL communicator buffers: state the communicator set, publish
        # its signature so the RANKS stamp their measurements with the same
        # key the LAUNCHER will look them up by (the #605 boot-id lesson), and
        # price the term from the cache when a measurement for exactly this
        # (rig, communicator set) exists.
        # ------------------------------------------------------------------
        _groups = communicator_groups_from_server_args(server_args, rank_gpu_id)
        _nccl_mib_per_gpu = None
        _nccl_sig = ""
        try:
            from sglang.srt.mem_ledger.nccl_probe import (
                load_nccl_buffers,
                publish_signature,
            )
            from sglang.srt.mem_ledger.nccl_transport import nccl_signature

            _nccl_sig = nccl_signature(_groups)
            publish_signature(_nccl_sig)
            if hw_fp:
                measured = load_nccl_buffers(hw_fp, _nccl_sig)
                if measured:
                    # Cache is keyed by card UUID; the ledger charges per
                    # gpu_id. Map through the identity map's uuid table rather
                    # than positionally -- a CUDA ordinal is not an NVML index
                    # on this rig and never was.
                    by_gpu = {}
                    for gpu_id, uuid in (card_uuid_by_gpu or {}).items():
                        if uuid in measured:
                            by_gpu[int(gpu_id)] = float(measured[uuid])
                    _nccl_mib_per_gpu = by_gpu or None
        except Exception as e:  # pragma: no cover - cache/env differences
            logger.debug("NCCL buffer measurement unavailable: %s", e)

        moe_scratch = _moe_offload_scratch_mib_per_rank(server_args, len(rank_gpu_id))

        return cls(
            weight_mib_per_rank=list(weight_mib_per_rank),
            activation_mib_per_rank=activation,
            capture_tokens_per_rank=capture,
            capture_mib_per_rank=(
                capture_mib if any(x > 0 for x in capture_mib) else None
            ),
            phase_footprint_source_per_rank=tuple(sources),
            phase_footprint_fingerprint=hw_fp or "",
            mamba_pool_mib_per_rank=list(mamba_pool_mib_per_rank or ()),
            gdn_scratch_mib_per_rank=gdn if gdn_applicable else None,
            indexer_scratch_mib_per_rank=indexer if indexer_applicable else None,
            moe_scratch_mib_per_rank=moe_scratch,
            ladder_mib_per_gpu=ladder,
            chunked_prefill_size=int(server_args.chunked_prefill_size or 0),
            context_length=getattr(server_args, "context_length", None),
            max_running_requests=getattr(server_args, "max_running_requests", None),
            mamba_floor_slots=floor_slots,
            mamba_floor_derivation=floor_note,
            indexer_chunk_cap_mib=indexer_cap,
            flashinfer_workspace_mib=flashinfer_mib,
            flashinfer_workspace_note=flashinfer_note,
            enable_deterministic_inference=deterministic,
            model_architectures=architectures,
            attn_scratch_budget_mib=(
                int(attn_scratch) if attn_scratch is not None else None
            ),
            parent_binds_cuda_context=False,
            # #598. Stating the groups is what lets the NCCL term reach its
            # third state. It is stated from the SAME ServerArgs the boot uses,
            # and the verdict then comes from the construction predicates
            # themselves, so a launch whose TP group barlink owns prices this
            # term at 0-with-a-reason instead of refusing the whole reserve
            # over a communicator that is never built.
            communicator_groups=_groups,
            # #594. The two fields that let TERM_NCCL_BUFFERS reach "priced".
            # They shipped declared and unassigned: #598 wired the groups so
            # the term could reach NOT_APPLICABLE, but nothing ever supplied a
            # measurement, so every TP>1 boot refused at parse time. The
            # measurement comes from mem_ledger/nccl_probe.py, keyed on the
            # rig fingerprint AND the communicator set it was taken under.
            nccl_buffer_mib_per_gpu=_nccl_mib_per_gpu,
            nccl_signature=_nccl_sig,
        )


def _moe_offload_scratch_mib_per_rank(
    server_args, n_ranks: int
) -> Optional[List[float]]:
    """Per-rank MoE expert-offload SPILL SCRATCH band, MiB, or ``None``.

    ``None`` means the launch has no expert-offload lane, i.e. the allocation
    does not exist here -- not that it is unknown.

    Every number below is READ from the existing derivation, never restated:
    the routed-expert mass per rank from ``PerfCostModel.
    per_rank_offloadable_weight_bytes`` (uneven_perf.py:4614-4637, the same
    call planner/capacity.py:77 and planner/placement.py:1614 make), the
    residency split from ``resident_slot_count`` and ``scratch_slot_count``
    (layers/moe/expert_offload.py:589-612), the buffer cap from
    ``ExpertResidencyPlanner.buffer_size`` (:496-498), and the per-rank
    fraction from the sanctioned accessor ``resident_fraction_vector``
    (layers/moe/resident_fraction.py:91), which takes an explicit
    ``server_args`` precisely so it can be read before distributed init.

    The cost model is built ONLY when a resident fraction is actually pinned.
    Constructing it reads the checkpoint config, and no launch that never
    offloads should pay for that -- the same stance planner/placement.py:1659
    takes about resolving the identity map only when there is something to
    resolve.
    """
    n = max(int(n_ranks), 1)
    try:
        from sglang.srt.layers.moe.expert_offload import (
            resident_slot_count,
            scratch_slot_count,
        )
        from sglang.srt.layers.moe.resident_fraction import resident_fraction_vector

        fracs = resident_fraction_vector(n, server_args)
        if not fracs or all(f >= 1.0 for f in fracs):
            return None  # no spill pool exists: every expert is resident

        from sglang.srt.uneven_perf import PerfCostModel, PlanInputs

        plan_inputs = PlanInputs.from_server_args(server_args)
        base_plan = list(plan_inputs.rank_tp_ratio or [1] * n)
        model = PerfCostModel(
            plan_inputs, base_plan, list(plan_inputs.effective_vram_mib or [1024] * n)
        )
        if int(model.num_experts) <= 0:
            return None  # dense checkpoint: there is nothing to offload
        mlp_vector = list(plan_inputs.rank_mlp_ratio or base_plan)
        routed_bytes = model.per_rank_offloadable_weight_bytes(mlp_vector)
        total = float(sum(routed_bytes))
        if total <= 0:
            return None

        out: List[float] = []
        for r in range(n):
            # Expert COUNT per rank from the same mass split that priced the
            # bytes, so the two can never disagree about where an expert is.
            e_local = max(1, int(round(model.num_experts * routed_bytes[r] / total)))
            per_expert_mib = routed_bytes[r] / e_local / (1 << 20)
            resident = resident_slot_count(e_local, float(fracs[r]))
            # buffer_size caps the band at E: a rank whose whole expert set
            # fits has no spill pool to stage into.
            scratch = min(scratch_slot_count(resident), e_local - resident)
            out.append(max(0.0, scratch * per_expert_mib))
        return out
    except Exception as e:  # pragma: no cover - config/env shape differences
        logger.debug("MoE offload scratch band unavailable for the ledger: %s", e)
        return None


def _model_architectures(server_args) -> Tuple[str, ...]:
    """The checkpoint's ``architectures`` list, or ``()`` when unreadable.

    Read straight from the HF config rather than through ``get_model_config()``:
    that one memoizes a full ModelConfig on the ServerArgs, and this runs during
    argument resolution, so caching one here would change what every later
    reader sees. Same stance (and the same reason) as
    ``_gdn_linear_attention_dims``.
    """
    try:
        from sglang.srt.utils.hf_transformers.config import get_config

        cfg = get_config(
            server_args.model_path,
            trust_remote_code=getattr(server_args, "trust_remote_code", False),
            revision=getattr(server_args, "revision", None),
            model_config_parser=getattr(server_args, "model_config_parser", None),
        )
        return tuple(str(a) for a in (getattr(cfg, "architectures", None) or ()))
    except Exception as e:  # pragma: no cover - config-source differences
        logger.debug("could not read the checkpoint architectures: %s", e)
        return ()


def _ranks_on(gpu_id: int, rank_gpu_id: Sequence[int]) -> Tuple[int, ...]:
    return tuple(r for r, g in enumerate(rank_gpu_id) if g == gpu_id)


def _classify_nccl_groups(inputs: "DemandInputs"):
    """Per-group transport verdicts, or ``None`` when the launch did not state
    its communicator groups.

    ``None`` is the pre-#598 world and must stay byte-identical there: a caller
    that says nothing about transports gets the old two-state term (priced when
    measured, UNBOUNDED otherwise). Only a caller that DESCRIBES its groups can
    reach the NOT_APPLICABLE state, because only then is there something to
    derive it from.
    """
    if inputs.communicator_groups is None:
        return None
    from sglang.srt.mem_ledger.nccl_transport import classify_communicator_groups

    return classify_communicator_groups(inputs.communicator_groups)


#: Every group name ``sglang.srt.distributed.parallel_state`` constructs a
#: GroupCoordinator for, mapped to the rule this module declares it by.
#:
#: ONE SOURCE OF TRUTH (#504-a pattern). A contract test AST-scans
#: ``parallel_state`` for its ``group_name=`` literals and refuses while that
#: set and these keys differ IN EITHER DIRECTION. Both directions are real
#: defects and they fail differently:
#:
#:   * a group built at runtime and missing here is an UNDER-declaration. Its
#:     communicator allocates buffers that no ledger row mentions, and it does
#:     not move ``nccl_signature``, so a measurement taken without it is reused
#:     for a launch that has it. That is the OOM direction.
#:   * a group declared here and never built is an OVER-declaration. It cannot
#:     under-charge, but it poisons ``nccl_signature`` with a phantom member,
#:     which invalidates measurements that are still exactly correct and can
#:     push an otherwise NOT_APPLICABLE launch to UNBOUNDED, i.e. a refusal
#:     nobody can satisfy by measuring.
#:
#: The value is the declaration rule, so a reader can check a row against the
#: construction site without reading this function.
RUNTIME_COMMUNICATOR_GROUPS: Mapping[str, str] = {
    "world": (
        "always built (init_world_group, parallel_state.py:2511-2525) and "
        "always with use_pynccl=False -- the world coordinator never "
        "constructs a PyNccl communicator. Declared with that flag, not with "
        "the default True"
    ),
    "tp": (
        "always built (parallel_state.py:3018) with the default transport; "
        "spans tp_size ranks"
    ),
    "pdmux_prefill_tp": (
        "built when duplicate_tp_group is passed, which model_runner takes "
        "from --enable-pdmux (parallel_state.py:3032, model_runner.py:1823). "
        "A SECOND communicator over the same tp ranks, so it allocates a "
        "second set of buffers"
    ),
    "dcp": "built when dcp_size > 1 (parallel_state.py:3054); spans dcp_size ranks",
    "dcp_spill": (
        "built when dcp_size > 1 AND SGLANG_KVSO_DECOUPLE=1 "
        "(parallel_state.py:3079): the kv-session-offload comm B, a second "
        "communicator over the same dcp ranks"
    ),
    "decoupled_kv": (
        "built by initialize_decoupled_kv_group over the planned pipeline "
        "ranks (parallel_state.py:3567-3600, #704b decoupled KV). Declared "
        "for INVENTORY completeness: this contract is checked against "
        "construction SITES, and the initializer has no production caller "
        "today, so no boot allocates it yet. Declaring it now is what keeps "
        "the day someone wires it from looking like an undeclared group"
    ),
    "attn_cp": (
        "built only when attn_cp_size != tp_size (parallel_state.py:3113-3129); "
        "equal sizes ALIAS _TP and construct nothing"
    ),
    "attention_tp": (
        "built only when attn_tp_size != tp_size, where attn_tp_size = "
        "tp_size // attn_cp_size // attn_dp_size (parallel_state.py:3145-3163); "
        "constructed with use_pynccl=(SYNC_TOKEN_IDS_ACROSS_TP or "
        "--enable-symm-mem)"
    ),
    "moe_dp": (
        "built only when moe_dp_size != tp_size AND attn_cp_size <= moe_dp_size "
        "(parallel_state.py:3168-3195); otherwise it aliases _ATTN_CP or _TP"
    ),
    "moe_ep": (
        "built only when ep_size != tp_size (parallel_state.py:3197-3223), and "
        "then with use_pynccl=False"
    ),
    "moe_tp": (
        "built only when moe_tp_size != tp_size, where moe_tp_size = tp_size // "
        "ep_size // moe_dp_size (parallel_state.py:3225-3251), and then with "
        "use_pynccl=False"
    ),
    "pp": (
        "always built (parallel_state.py:3267) with the default transport; "
        "spans pp_size ranks"
    ),
    # THE #631 SECONDARY (FLIP-TARGET) SET. Three more GroupCoordinators over
    # the SAME world, built by initialize_phase_flip_secondary_groups
    # (parallel_state.py:3421-3515) from model_runner.py:1972-1984 whenever
    # --enable-phase-flip is set. They are communicators like any other and
    # they are constructed INSIDE the nccl_init_begin/nccl_init_end window
    # (#605 placed the end mark after them deliberately, model_runner.py:1985),
    # so the first boot to measure TERM_NCCL_BUFFERS measures them. Leaving
    # them undeclared was the OOM direction twice over: their buffers were
    # charged to nobody, and they did not move nccl_signature, so a
    # measurement taken on a non-flip launch was reusable on a flip one.
    "flip_tp": (
        "built whenever --enable-phase-flip is set (parallel_state.py:3502, "
        "via model_runner.py:1980) with the default transport; spans "
        "len(--phase-flip-tp-vector) ranks, which server_args pins equal to "
        "pp_size (server_args.py:7708)"
    ),
    "flip_dcp": (
        "built with the secondary set when its dcp_size > 1 "
        "(parallel_state.py:3505-3507); model_runner passes "
        "dcp_size=len(--phase-flip-tp-vector), so it exists on every "
        "multi-rank flip launch -- a SECOND communicator over the same ranks "
        "flip_tp already covers"
    ),
    "flip_pp": (
        "built with the secondary set (parallel_state.py:3509-3512) and then "
        "with use_custom_allreduce=False but the DEFAULT use_pynccl; "
        "model_runner passes pp_size=1, so on Route A it spans a single rank"
    ),
}


def communicator_groups_from_server_args(
    server_args, rank_gpu_id: Sequence[int]
) -> Tuple["CommunicatorGroup", ...]:
    """The communicator groups this launch builds, one entry per group that
    ``initialize_model_parallel`` actually CONSTRUCTS.

    #612 rewrote this. The first cut declared ``world``/``tp``/``pp`` (+``dcp``)
    and argued from a composite that the rest could not matter. Both halves of
    that were wrong against the construction sites:

      * ``world`` was declared with the default ``use_pynccl=True`` while
        ``init_world_group`` passes ``use_pynccl=False`` unconditionally
        (parallel_state.py:2518). The world coordinator never builds a PyNccl
        communicator, so it was a phantom member of every ``nccl_signature``
        this function produced.
      * ``pdmux_prefill_tp``, ``dcp_spill``, ``attn_cp``, ``attention_tp``,
        ``moe_dp``, ``moe_ep`` and ``moe_tp`` were not declared at all. Two of
        them are DUPLICATE communicators over ranks another group already
        covers (the PDMUX prefill TP group and the kv-session-offload spill
        group), i.e. a second buffer set on the same ranks -- the direction
        that under-charges a card.

    The old argument ("the world group is multi-rank whenever anything else is,
    so the composite verdict cannot be wrong") survives its own premise being
    fixed, and that is why this rewrite does not change any verdict on a real
    launch: whenever ``world`` spans more than one rank so does ``tp`` or
    ``pp``, and both are declared with the default transport. What the rewrite
    changes is the SIGNATURE and the named group list -- i.e. what a
    measurement is filed as valid for.

    :data:`RUNTIME_COMMUNICATOR_GROUPS` is the checked inventory of names; this
    function is the per-launch instance of it.
    """
    from sglang.srt.mem_ledger.nccl_transport import CommunicatorGroup

    def _size(name: str, default: int = 1) -> int:
        return int(getattr(server_args, name, default) or default)

    tp = _size("tp_size")
    pp = _size("pp_size")
    dcp = _size("dcp_size")
    ep = _size("ep_size")
    moe_dp = _size("moe_dp_size")
    attn_cp = _size("attn_cp_size")
    # model_runner.py:561 -- attention DP collapses to 1 unless dp attention is
    # on, and the group geometry follows that value, not the raw --dp-size.
    attn_dp = (
        _size("dp_size") if getattr(server_args, "enable_dp_attention", False) else 1
    )

    # Conservative on purpose: whichever count is larger is the one that can
    # make the world group multi-rank.
    world = max(len(rank_gpu_id), tp * pp)
    groups = [
        # use_pynccl=False is not a policy choice here, it is a QUOTE of
        # init_world_group (parallel_state.py:2518). The world coordinator
        # exists on every launch and builds a torch process group, but never a
        # PyNccl communicator, so it allocates none of the buffers this term
        # prices.
        CommunicatorGroup(name="world", world_size=world, use_pynccl=False),
        CommunicatorGroup(name="tp", world_size=tp),
        CommunicatorGroup(name="pp", world_size=pp),
    ]
    if getattr(server_args, "enable_pdmux", False):
        groups.append(CommunicatorGroup(name="pdmux_prefill_tp", world_size=tp))
    if dcp > 1:
        groups.append(CommunicatorGroup(name="dcp", world_size=dcp))
        if os.environ.get("SGLANG_KVSO_DECOUPLE", "0") == "1":
            groups.append(CommunicatorGroup(name="dcp_spill", world_size=dcp))
    if attn_cp != tp:
        groups.append(CommunicatorGroup(name="attn_cp", world_size=attn_cp))
    # Clamped at 1: an invalid size combination (ep * moe_dp > tp, say)
    # floor-divides to 0, and a 0-rank group is not a thing the runtime
    # builds -- such a launch fails its own asserts long before this.
    attn_tp = max(1, tp // max(attn_cp, 1) // max(attn_dp, 1))
    if attn_tp != tp:
        groups.append(
            CommunicatorGroup(
                name="attention_tp",
                world_size=attn_tp,
                use_pynccl=_attention_tp_uses_pynccl(server_args),
            )
        )
    if attn_cp <= moe_dp and moe_dp != tp:
        groups.append(CommunicatorGroup(name="moe_dp", world_size=moe_dp))
    if ep != tp:
        groups.append(CommunicatorGroup(name="moe_ep", world_size=ep, use_pynccl=False))
    moe_tp = max(1, tp // max(ep, 1) // max(moe_dp, 1))
    if moe_tp != tp:
        groups.append(
            CommunicatorGroup(name="moe_tp", world_size=moe_tp, use_pynccl=False)
        )
    groups.extend(_phase_flip_secondary_groups(server_args, pp))
    return tuple(groups)


def _phase_flip_secondary_groups(server_args, pp: int):
    """The #631 flip-target set, as CommunicatorGroups. Empty without the flag.

    THE GEOMETRY IS QUOTED, NOT RESTATED. model_runner.py:1976-1984 calls
    ``initialize_phase_flip_secondary_groups`` with
    ``tp_size = dcp_size = len(--phase-flip-tp-vector)`` and ``pp_size=1``, and
    server_args refuses a vector whose length differs from ``pp_size``
    (server_args.py:7708) -- so the vector length IS the primary pp_size and
    this needs no second parse to agree with the launch that will happen. The
    vector is still read when present, because reading the value the call site
    reads is what keeps the two from drifting; ``pp`` is only the fallback.

    ``flip_dcp`` appears exactly on the construction site's condition
    (``dcp_size > 1``), not on "dcp is enabled": the secondary set's DCP width
    is its own, and the primary ``--dcp-size`` has nothing to do with it.
    """
    if not getattr(server_args, "enable_phase_flip", False):
        return ()
    from sglang.srt.mem_ledger.nccl_transport import CommunicatorGroup

    vec = getattr(server_args, "phase_flip_tp_vector", None)
    width = pp
    if vec:
        try:
            width = len([int(x) for x in str(vec).split(",")])
        except ValueError:
            # server_args refuses this launch outright; the ledger runs first
            # and must not raise on argv it is not the judge of.
            width = pp
    width = max(1, int(width))
    out = [CommunicatorGroup(name="flip_tp", world_size=width)]
    if width > 1:
        out.append(CommunicatorGroup(name="flip_dcp", world_size=width))
    # pp_size=1 at the call site. Declared anyway: the group is CONSTRUCTED,
    # and classify_communicator_groups is the thing that decides a single-rank
    # group allocates nothing. Deciding it here would hide the group from the
    # inventory the moment the call site passes a different pp_size.
    out.append(CommunicatorGroup(name="flip_pp", world_size=1))
    return tuple(out)


def _attention_tp_uses_pynccl(server_args) -> bool:
    """The flag ``attention_tp`` is constructed with
    (parallel_state.py:3159-3161).

    Read from the same module the construction site reads it from rather than
    restated. When that import is unavailable the answer is True, which
    OVER-declares: an over-declared PyNccl group can only push the term toward
    UNBOUNDED or priced, never toward NOT_APPLICABLE, and under-charging is the
    direction that OOMs a card.
    """
    if bool(getattr(server_args, "enable_symm_mem", False)):
        return True
    try:
        from sglang.srt.layers.sampler import SYNC_TOKEN_IDS_ACROSS_TP

        return bool(SYNC_TOKEN_IDS_ACROSS_TP)
    except Exception as e:  # pragma: no cover - import-environment differences
        logger.debug("could not read SYNC_TOKEN_IDS_ACROSS_TP (%s)", e)
        return True


def _history_band(history, uuid: str, post: str):
    """One band off a :class:`BootHistory`, or None, never raising.

    Guarded whole and typed loosely on purpose: the history is an OBSERVATION
    fed in from outside, and an instrument that can refuse a launch is worse
    than no instrument. A malformed or absent history leaves every term on the
    behaviour it had before this parameter existed.
    """
    if history is None:
        return None
    try:
        return history.band(uuid, post)
    except Exception:  # pragma: no cover - an observation must not raise
        return None


def build_card_ledgers(
    inputs: DemandInputs,
    *,
    cards: Sequence[CardFacts],
    rank_gpu_id: Sequence[int],
    user_reserve_mib: Mapping[int, int],
    calibration=None,
    tenant_terms: Optional[Mapping[int, Sequence[LedgerTerm]]] = None,
    history=None,
) -> List[CardVramLedger]:
    """One :class:`CardVramLedger` per card, itemized and provenance-tagged.

    ``calibration`` is a
    :class:`~sglang.srt.mem_ledger.calibration.CalibrationProfile` or None. None
    does NOT mean zero and does not mean a default: the hardware-residual term
    becomes UNBOUNDED and the boot refuses with the probe command, because a
    residual that the rig has never measured is exactly the number the old
    ``_PREDICT_OVERHEAD_MIB = 1280`` guessed.

    ``history`` is a
    :class:`~sglang.srt.mem_ledger.boot_history.BootHistory` or None, and
    ``None`` is the default precisely so every existing caller is byte-
    identical. When present it takes PRECEDENCE over the inherited constants
    for the two terms it can speak to, because it is a measurement of THIS rig
    in THIS window and they are not:

    * the hardware residual, where the window-2026-08-06 calibration measured
      25-34% low against 472 recorded boots;
    * the load transient, whose flat ``LOAD_TRANSIENT_REFERENCE_MIB`` the same
      history falsifies outright.

    A band the history REFUSES (too wide to summarise) does not silently fall
    through to the constant it was meant to replace. For the transient, which
    has no probe behind it, the term becomes UNBOUNDED and names the
    distribution. For the residual, which does have a probe, the measured
    calibration stands -- a wide history is a reason to distrust a summary of
    the history, not a reason to discard a direct measurement.
    """
    if len(rank_gpu_id) != inputs.rank_count():
        raise LedgerError(
            f"{len(rank_gpu_id)} ranks are placed but the demand inputs "
            f"describe {inputs.rank_count()}"
        )
    tenant_terms = tenant_terms or {}
    residual_by_uuid = calibration.by_uuid() if calibration is not None else {}
    # Once for the launch, not once per card: the communicator set is a
    # property of the launch, and the per-card part of the NCCL term is only
    # the co-located rank multiplier.
    nccl_verdicts = _classify_nccl_groups(inputs)
    nccl_unresolved = [v for v in nccl_verdicts or () if not v.resolved]
    nccl_building = [v for v in nccl_verdicts or () if v.builds_nccl]

    ledgers: List[CardVramLedger] = []
    for card in cards:
        ranks = _ranks_on(card.gpu_id, rank_gpu_id)
        if not ranks:
            continue
        terms: List[LedgerTerm] = []
        unbounded: List[str] = []
        n_co = len(ranks)

        # -- weights ---------------------------------------------------------
        weights = sum(int(inputs.weight_mib_per_rank[r]) for r in ranks)
        terms.append(
            LedgerTerm(
                name=TERM_WEIGHTS,
                mib=weights,
                provenance=Provenance.MODELED,
                derivation=(
                    "sum of the resident shard footprint of rank(s) "
                    + ", ".join(
                        f"{r}={int(inputs.weight_mib_per_rank[r])} MiB" for r in ranks
                    )
                    + "; the shard vector comes from the uneven-TP partition, so "
                    "an uneven split moves this line and nothing else"
                ),
                inputs=("rank_tp_ratio", "model_path", "quantization", "dtype"),
            )
        )

        # -- load transient, PER RANK ----------------------------------------
        # #612. An allocator peak above the resident set that no other term
        # covers. Charged per RANK for the same reason the activation term is
        # (correction 1 in the module docstring): co-located ranks are separate
        # processes, they load and serve concurrently, and each holds its own
        # peak. The 2026-08-06 window that produced the inherited number ran one
        # rank per card, so it cannot distinguish per-card from per-rank; per
        # rank is the reading that cannot under-charge a co-located card.
        transient_band = _history_band(history, card.uuid, "load_transient")
        measured_transient = inputs.load_transient_mib_per_rank
        if transient_band is not None and transient_band.refused:
            # The post is not a constant on this rig, so no constant is honest.
            # UNBOUNDED rather than the inherited 70 MiB: a falsified figure
            # that still prints is the failure mode this ledger exists to end.
            unbounded.append(
                f"{TERM_LOAD_TRANSIENT} on {card.name}: the boot history "
                f"refuses to summarise this post -- {transient_band.reason}. "
                "The inherited "
                f"{LOAD_TRANSIENT_REFERENCE_MIB} MiB constant is NOT used as a "
                "fallback: this rig's own boots falsify it, and a term that "
                "cannot be bounded has to say so"
            )
        elif transient_band is not None:
            terms.append(
                LedgerTerm(
                    name=TERM_LOAD_TRANSIENT,
                    mib=int(transient_band.charge_mib) * n_co,
                    provenance=Provenance.CALIBRATED,
                    derivation=(
                        f"boot-history band {transient_band.low_mib}-"
                        f"{transient_band.high_mib} MiB over "
                        f"{transient_band.n_boots} recorded boots of this "
                        f"card, charged at the band's HIGH x {n_co} rank "
                        "process(es) on this card"
                    ),
                    fingerprint=f"boot-history:{transient_band.n_boots}",
                    bounded_by=(
                        "the caching allocator: the peak is reservation above "
                        "the resident set within one phase, not a growing pool"
                    ),
                )
            )
        elif measured_transient and len(measured_transient) > max(ranks):
            transient = sum(float(measured_transient[r]) for r in ranks)
            terms.append(
                LedgerTerm(
                    name=TERM_LOAD_TRANSIENT,
                    mib=int(math.ceil(transient)),
                    provenance=Provenance.CALIBRATED,
                    derivation=(
                        "measured allocator peak above the resident set "
                        f"(reserved peak - reserved current), summed over "
                        f"{n_co} rank process(es) on this card"
                    ),
                    fingerprint=inputs.phase_footprint_fingerprint
                    or LOAD_TRANSIENT_REFERENCE_TAG,
                    bounded_by=(
                        "the caching allocator: the peak is reservation above "
                        "the resident set within one phase, not a growing pool"
                    ),
                )
            )
        else:
            # NAMED, LOGGED FALLBACK. The row has always said "INHERITED" and
            # carried a window tag that cannot match a live fingerprint (#612),
            # but a printout is only read by someone who goes looking. The
            # standing rule is that the ledger is the authority and a hand-set
            # number never decides silently, so the one term still standing on
            # an unmeasured literal announces itself once per process, by name.
            if not _LoadTransientFallback.announced:
                _LoadTransientFallback.announced = True
                logger.warning(
                    "mem_ledger: %s falls back to the INHERITED constant "
                    "%d MiB per rank on %s -- not measured on this rig. It "
                    "comes from the 2026-08-06 #602 corridor window and is "
                    "queued for calibration (measured.CALIBRATION_QUEUE). "
                    "Boot with SGLANG_VRAM_FLIGHT_DIR set to replace it with "
                    "this rig's own allocator_transient_bytes.",
                    TERM_LOAD_TRANSIENT,
                    LOAD_TRANSIENT_REFERENCE_MIB,
                    card.name,
                )
            terms.append(
                LedgerTerm(
                    name=TERM_LOAD_TRANSIENT,
                    mib=LOAD_TRANSIENT_REFERENCE_MIB * n_co,
                    provenance=Provenance.CALIBRATED,
                    derivation=(
                        f"INHERITED, not measured on this rig: "
                        f"{LOAD_TRANSIENT_REFERENCE_MIB} MiB per rank x {n_co} "
                        "rank process(es) on this card, from the 2026-08-06 "
                        "#602 corridor window, where the per-card NVML free "
                        "minimum under load sat up to that far below the idle "
                        "reading with every other term priced. Boot with "
                        "SGLANG_VRAM_FLIGHT_DIR set and reconcile this term "
                        "against the measured allocator_transient_bytes to "
                        "replace the inherited figure"
                    ),
                    fingerprint=LOAD_TRANSIENT_REFERENCE_TAG,
                    bounded_by=(
                        "the caching allocator: the peak is reservation above "
                        "the resident set within one phase, not a growing pool"
                    ),
                )
            )

        # -- activation, PER RANK, CALIBRATED (never the falsified heuristic) --
        missing_activation = [
            r for r in ranks if inputs.activation_mib_per_rank[r] is None
        ]
        if missing_activation:
            unbounded.append(
                f"{TERM_ACTIVATION} on {card.name} (rank(s) "
                f"{', '.join(str(r) for r in missing_activation)}): no phase "
                "footprint is calibrated for this hardware fingerprint and "
                "activation profile. Run "
                "`scripts/vram_ledger/probe_activation.py` once for this "
                "config. This term does NOT fall back to the inherited "
                "512+tokens*1.5+tp*pp/8*1024 heuristic: the 2026-08-05 window "
                "falsified it (3968 MiB booked against 1766 MiB free on the "
                "card that then completed a 70018-token prefill), and a "
                "falsified formula that still runs is worse than a refusal "
                "because it returns a number that looks like an answer"
            )
        else:
            activation = sum(float(inputs.activation_mib_per_rank[r]) for r in ranks)
            src = ""
            for r in ranks:
                if r < len(inputs.phase_footprint_source_per_rank):
                    src = inputs.phase_footprint_source_per_rank[r]
                    break
            terms.append(
                LedgerTerm(
                    name=TERM_ACTIVATION,
                    mib=int(math.ceil(activation)),
                    provenance=Provenance.CALIBRATED,
                    derivation=(
                        f"prefill activation peak, summed over {n_co} rank "
                        f"process(es) on this card at chunked_prefill_size="
                        f"{inputs.chunked_prefill_size}. Charged per RANK, not "
                        "per card: co-located ranks are separate processes and "
                        f"hold their prefill peaks simultaneously. {src}"
                    ),
                    fingerprint=inputs.phase_footprint_fingerprint,
                    bounded_by="chunked_prefill_size (the prefill is chunked, "
                    "so the peak is a function of the chunk, not of the "
                    "request)",
                )
            )

        # -- CUDA graph capture, measured where available ---------------------
        if inputs.capture_mib_per_rank is not None:
            cap_mib = sum(float(inputs.capture_mib_per_rank[r]) for r in ranks)
            src = ""
            for r in ranks:
                if r < len(inputs.phase_footprint_source_per_rank):
                    src = inputs.phase_footprint_source_per_rank[r]
                    break
            terms.append(
                LedgerTerm(
                    name=TERM_GRAPH_CAPTURE,
                    mib=int(math.ceil(cap_mib)),
                    provenance=Provenance.CALIBRATED,
                    derivation=(
                        f"measured graph-capture cost, summed over {n_co} rank "
                        f"process(es) on this card (each captures its own "
                        f"graphs). Replaces the captured-tokens x "
                        f"{GRAPH_MIB_PER_CAPTURED_TOKEN} MiB estimate, which "
                        "the 2026-08-05 window measured 3.3-3.8x LOW -- an "
                        f"under-charge, i.e. the dangerous direction. {src}"
                    ),
                    fingerprint=inputs.phase_footprint_fingerprint,
                )
            )
        else:
            # REFUSAL, not the token estimate. The 2026-08-05 window measured
            # that estimate 3.3-3.8x LOW (192 MiB booked against 633-730 MiB
            # actually taken per rank), and an under-charge is the direction
            # that OOMs a boot rather than merely costing KV. Keeping it as a
            # fallback would mean the ledger's most dangerous term is also its
            # least trustworthy one, silently, on any rig that has not been
            # probed.
            est = sum(int(inputs.capture_tokens_per_rank[r]) for r in ranks) * (
                GRAPH_MIB_PER_CAPTURED_TOKEN
            )
            unbounded.append(
                f"{TERM_GRAPH_CAPTURE} on {card.name} (rank(s) "
                f"{', '.join(str(r) for r in ranks)}): no phase footprint is "
                "calibrated for this hardware fingerprint and activation "
                "profile. Run `scripts/vram_ledger/probe_activation.py` for "
                "this configuration; it writes the per-rank "
                "`phase_footprint_rank<N>.json` this term reads. This term "
                "does NOT fall back to "
                f"the captured-tokens x {GRAPH_MIB_PER_CAPTURED_TOKEN} MiB "
                f"estimate (~{est} MiB here), which that window measured "
                "3.3-3.8x low -- an under-charge is the direction that OOMs"
            )

        # -- adaptive ladder, charged to exactly one GPU ---------------------
        ladder_mib = int(inputs.ladder_mib_per_gpu.get(card.gpu_id, 0))
        if ladder_mib:
            terms.append(
                LedgerTerm(
                    name=TERM_LADDER,
                    mib=ladder_mib,
                    provenance=Provenance.MODELED,
                    derivation=(
                        "ladder_reserve_demand(): the rungs the adaptive "
                        "controller builds beyond the boot rung plus the "
                        "serving margin its boot check enforces. Charged to "
                        "this GPU alone because it hosts the solo draft rank"
                    ),
                    inputs=(
                        "speculative_adaptive",
                        "speculative_draft_placement",
                        "speculative_num_draft_tokens",
                    ),
                )
            )

        # -- mamba / GDN state pool ------------------------------------------
        if inputs.mamba_pool_mib_per_rank:
            pool = sum(float(inputs.mamba_pool_mib_per_rank[r]) for r in ranks)
            floor_note = (
                f"; floor {inputs.mamba_floor_slots} slots "
                f"({inputs.mamba_floor_derivation})"
                if inputs.mamba_floor_slots
                else ""
            )
            terms.append(
                LedgerTerm(
                    name=TERM_MAMBA_POOL,
                    mib=int(math.ceil(pool)),
                    provenance=Provenance.MODELED,
                    derivation=(
                        "per-rank SSM state + conv buffers, sized by this "
                        "rank's GDN-unit share (the pool sticks to the units). "
                        "The slot count may not fall below the #581 hard "
                        "floor, which is the number of slots one running "
                        "request holds simultaneously" + floor_note
                    ),
                    inputs=(
                        "max_running_requests",
                        "speculative_num_draft_tokens",
                        "disable_radix_cache",
                        "disable_overlap_schedule",
                        "rank_tp_ratio",
                    ),
                )
            )

        # -- prefill scratch, transients bounded by MECHANISM (#493) ---------
        if inputs.gdn_scratch_mib_per_rank is None:
            pass  # no GDN/linear-attention layers: does not apply
        elif inputs.gdn_scratch_mib_per_rank:
            gdn = sum(float(inputs.gdn_scratch_mib_per_rank[r]) for r in ranks)
            terms.append(
                LedgerTerm(
                    name=TERM_GDN_SCRATCH,
                    mib=int(math.ceil(gdn)),
                    provenance=Provenance.MODELED,
                    derivation=(
                        "gdn_prefill_scratch_mib(): every intermediate of one "
                        "chunked gated-delta-rule layer is alive at once, "
                        f"summed over {n_co} rank process(es). Derived from the "
                        "allocation sites, not measured"
                    ),
                    inputs=("chunked_prefill_size", "rank_tp_ratio", "model_path"),
                    bounded_by="chunked_prefill_size (the chunk IS the cap; "
                    "no budget line can cap this transient -- #493)",
                )
            )

        if inputs.indexer_scratch_mib_per_rank is None:
            pass  # no DSV4 C4 indexer: does not apply
        elif inputs.indexer_scratch_mib_per_rank:
            idx = sum(float(inputs.indexer_scratch_mib_per_rank[r]) for r in ranks)
            if inputs.indexer_chunk_cap_mib is None:
                # The transient exists and nothing caps it. #493 proved padding
                # does not, so this is a refusal rather than a bigger number.
                unbounded.append(
                    f"{TERM_INDEXER_SCRATCH} (~{int(math.ceil(idx))} MiB): the "
                    "C4-indexer transient is present but "
                    "SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB could not be read, so "
                    "nothing caps it"
                )
            else:
                terms.append(
                    LedgerTerm(
                        name=TERM_INDEXER_SCRATCH,
                        mib=int(math.ceil(idx)),
                        provenance=Provenance.MODELED,
                        derivation=(
                            "dsv4_indexer_prefill_scratch_mib(): the paged-MQA "
                            "logits transient of one C4-indexer call, summed "
                            f"over {n_co} rank process(es), at "
                            f"chunked_prefill_size={inputs.chunked_prefill_size} "
                            f"and context_length={inputs.context_length}"
                        ),
                        inputs=(
                            "chunked_prefill_size",
                            "context_length",
                            "model_path",
                        ),
                        bounded_by=(
                            f"SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB="
                            f"{inputs.indexer_chunk_cap_mib} MiB (the query "
                            "chunk is staged, so the cap is real -- #493)"
                        ),
                    )
                )

        # -- MoE expert-offload spill scratch (#1036) ------------------------
        # The one device buffer the 13 inherited terms never named. It IS
        # accounted today, but by MEASUREMENT, not by itemization: KV sizing
        # reads live free VRAM after the offload has installed and released
        # (model_runner_kv_cache_mixin.py:350-353), and that file says so
        # outright at :371 -- "No new budget term is introduced". A measured
        # post has no name, cannot be planned before a card is loaded, fails
        # as an OOM instead of a parse-time refusal, and is lost entirely with
        # SGLANG_MOE_OFFLOAD_KV_REGAIN=0, which switches the ordering off
        # (environ.py:1507-1508). So it becomes a term.
        #
        # Charged for the SCRATCH slots only. The resident slots are already
        # inside TERM_WEIGHTS, which prices the FULL routed stack; the
        # (E - R) host-held experts are therefore over-charged there and this
        # term deliberately does NOT credit that reclaim back. A credit would
        # be a second bookkeeping of the release tally the boot log already
        # states at :1060-1064, and it would make the ledger optimistic in
        # exactly the direction that OOMs a 20 GiB card.
        if inputs.moe_scratch_mib_per_rank is None:
            pass  # no expert-offload lane: does not apply
        elif inputs.moe_scratch_mib_per_rank:
            moe_scratch = sum(
                float(inputs.moe_scratch_mib_per_rank[r]) for r in ranks
            )
            terms.append(
                LedgerTerm(
                    name=TERM_MOE_OFFLOAD_SCRATCH,
                    mib=int(math.ceil(moe_scratch)),
                    provenance=Provenance.MODELED,
                    derivation=(
                        "scratch_slot_count(resident_slot_count(E, fraction)) "
                        "slots of the fixed-resident GPU buffer, priced at the "
                        "per-expert mass PerfCostModel."
                        "per_rank_offloadable_weight_bytes() reports for this "
                        f"rank, summed over {n_co} rank process(es). The "
                        "resident slots are not charged here -- "
                        "TERM_WEIGHTS already prices the whole routed stack"
                    ),
                    inputs=(
                        "rank_moe_resident_fraction",
                        "rank_moe_ratio",
                        "rank_tp_ratio",
                        "model_path",
                    ),
                    bounded_by=(
                        "SGLANG_MOE_SCRATCH_SLOTS (the SLOT COUNT is the cap: "
                        "the buffer is (R + C) slots and is allocated once at "
                        "install, min(R + C, E) -- expert_offload.py:496-498, "
                        "589-612; a wave that needs more spill slots is "
                        "wave-split rather than allowed to grow the buffer)"
                    ),
                )
            )

        # -- attention workspaces, charged AT THEIR CAP ----------------------
        # These are the "allocated lazily after KV sizing" pools that the old
        # reserve text listed and no budget charged. Their size is not derived
        # from geometry; it is DECIDED by a knob, which is the strongest form
        # of the #493 rule -- a transient with a binding cap is fundable at the
        # cap, and only there.
        ws_parts: List[str] = []
        # #595 (b): price the ACTIVE backend's private workspace.
        #
        # Every attention backend allocates its own scratch at init, and until
        # now only flashinfer's was a term. That is not a small omission for a
        # rig on another backend: trtllm_mla alone takes a fixed 150 MiB
        # (DEFAULT_WORKSPACE_SIZE_MB, layers/attention/trtllm_mla_backend.py),
        # which on a 20 GiB card is most of a corridor.
        #
        # "" means the caller did not state a backend. That keeps the
        # pre-#595 accounting for existing callers and for this rig, which
        # runs flashinfer; production fills it from ServerArgs. A NAMED
        # backend nobody has priced becomes UNBOUNDED below rather than a
        # silent zero, because a zero there is indistinguishable from "this
        # backend allocates nothing" and no backend allocates nothing.
        backend = (inputs.attention_backend or "").strip().lower()
        ws_mib = 0
        if backend in ("", "flashinfer") and inputs.flashinfer_workspace_mib:
            ws_mib += inputs.flashinfer_workspace_mib * n_co
            detail = (
                inputs.flashinfer_workspace_note
                or f"{inputs.flashinfer_workspace_mib} MiB/rank "
                "(SGLANG_FLASHINFER_WORKSPACE_SIZE)"
            )
            ws_parts.append(f"flashinfer float workspace {detail}")
        elif backend in _FIXED_BACKEND_WORKSPACE_MIB:
            fixed = _FIXED_BACKEND_WORKSPACE_MIB[backend]
            ws_mib += fixed * n_co
            ws_parts.append(
                f"{backend} workspace {fixed} MiB/rank "
                "(DEFAULT_WORKSPACE_SIZE_MB in that backend's module)"
            )
        elif backend not in ("", "flashinfer"):
            unbounded.append(
                f"{TERM_ATTN_WORKSPACE} on {card.name}: attention backend "
                f"{backend!r} is active and its private workspace is not "
                "priced here. Every backend allocates scratch at init; "
                "charging zero would be a guess that this one does not. Add "
                "its size to the ledger's backend table (see "
                "TRTLLM_MLA_WORKSPACE_MIB) or run with flashinfer."
            )
        if inputs.attn_scratch_budget_mib:
            ws_mib += inputs.attn_scratch_budget_mib * n_co
            ws_parts.append(
                f"chunked-prefix attention scratch "
                f"{inputs.attn_scratch_budget_mib} MiB/rank "
                f"(--attn-scratch-budget-mib)"
            )
        if ws_mib:
            terms.append(
                LedgerTerm(
                    name=TERM_ATTN_WORKSPACE,
                    mib=ws_mib,
                    provenance=Provenance.MODELED,
                    derivation=(
                        " + ".join(ws_parts)
                        + f", x {n_co} rank process(es) on this card. Charged at "
                        "the cap, not at an observed size: the cap is what the "
                        "configuration permits the rank to take"
                    ),
                    inputs=(
                        "SGLANG_FLASHINFER_WORKSPACE_SIZE",
                        # The backend rewrites the env var from these two
                        # during __init__, so they -- not the raw variable --
                        # are what the size actually depends on.
                        "enable_deterministic_inference",
                        "hf_config.architectures",
                        "attn_scratch_budget_mib",
                        "attention_backend",
                    ),
                    bounded_by="the workspace knobs named in the derivation",
                )
            )

        # -- hardware residual, CALIBRATED (never a literal) ------------------
        # #595 (a): NCCL communicator buffers.
        #
        # WHY THIS CANNOT BE DERIVED. Nothing in the Python layer sizes them:
        # a grep for NCCL_BUFFSIZE finds only the Ascend/HCCL equivalents and
        # a window-register call. libnccl allocates its transport buffers
        # inside ncclCommInitRank, from NCCL_BUFFSIZE x channels x peers plus
        # algorithm-specific staging, none of which this process chooses. So
        # the only honest source is a measurement taken around communicator
        # init, and until one exists this term is UNBOUNDED -- which makes the
        # full-demand reserve refuse, correctly, rather than book a zero for
        # memory that is demonstrably allocated.
        #
        # WHY IT IS NOT KEYED ON THE HARDWARE FINGERPRINT ALONE, which is the
        # design question this term had to answer. The buffers are allocated
        # per COMMUNICATOR and scale with that communicator's peer count and
        # channel count, and one launch builds several (TP, DCP, world). Those
        # come from tp_size/dcp_size/pp_size -- configuration, not hardware --
        # while the calibration fingerprint covers card set, driver and wheel.
        # A value keyed on the fingerprint alone would therefore be served
        # across TP degrees that allocate different amounts, which is the
        # cross-attribution #589 already cost a window on. Hence the measured
        # value travels with nccl_signature and is usable only while that is
        # unchanged. Allocation happens at init and does not grow with
        # per-call message size, so a POINT value is sound within one
        # signature; if a measurement ever shows drift within a signature the
        # term must become a bound instead.
        #
        # #598: A THIRD STATE, AND THE MIS-INFERENCE THAT MADE IT NECESSARY.
        #
        # The paragraph that stood here claimed "THIS RIG ALWAYS PAYS IT",
        # reasoning from the window-8 boot line "custom all-reduce disabled ->
        # falling back to NCCL for TP collectives". That line is real and it
        # is about CUSTOM ALL-REDUCE being disabled; it says nothing about
        # what happens AFTERWARDS. On this rig barlink then takes ownership of
        # the TP group, and a group barlink owns never constructs a PyNccl
        # communicator at all -- GroupCoordinator logs exactly that: "barlink
        # is active for group 'tp:0': skipping PyNccl communicator
        # construction". Window 9 confirmed it from the other side:
        # NCCL_DEBUG=INFO with NCCL_DEBUG_SUBSYS=INIT,ALLOC produced ZERO
        # allocation lines, because there is nothing to allocate.
        #
        # So the evidence chain was wrong for this transport: "custom AR is
        # off" implies "NCCL would be the fallback", not "NCCL is what runs".
        # A term that reasons from a log line about a DIFFERENT decision is
        # exactly the adjacent-evidence failure this ledger keeps meeting; the
        # fix is to branch on the SAME predicate the construction site
        # branches on, which is what nccl_transport does (it imports
        # should_build_barlink / should_build_pynccl rather than restating
        # them, so the two cannot drift).
        #
        # The three states, and why NOT_APPLICABLE may not be folded into a
        # measured zero, are spelled out in mem_ledger/nccl_transport.py and
        # on LedgerTerm.not_applicable. In short: this term is UNBOUNDED while
        # an NCCL communicator is built and unmeasured, PRICED once measured
        # for a named communicator set, and NOT APPLICABLE when no NCCL
        # communicator is constructed at all. Only the last one is derivable
        # from configuration, which is why it is MODELED while the priced case
        # is CALIBRATED.
        nccl_mib = (inputs.nccl_buffer_mib_per_gpu or {}).get(card.gpu_id)
        if nccl_verdicts is not None and nccl_unresolved:
            # Conservative composite: one group nobody could place is enough
            # to refuse, and it is NAMED, because "could not tell" must not
            # read like "there is none".
            unbounded.append(
                f"{TERM_NCCL_BUFFERS} on {card.name}: the transport owning "
                f"{len(nccl_unresolved)} communicator group(s) could not be "
                "resolved, so whether NCCL buffers are allocated at all is "
                "unknown -- "
                + "; ".join(f"{v.name}: {v.reason}" for v in nccl_unresolved)
            )
        elif nccl_verdicts is not None and not nccl_building:
            skipped = ", ".join(f"{v.name} ({v.reason})" for v in nccl_verdicts) or (
                "this launch builds no multi-rank communicator group"
            )
            terms.append(
                LedgerTerm(
                    name=TERM_NCCL_BUFFERS,
                    mib=0,
                    provenance=Provenance.MODELED,
                    derivation=(
                        "NOT APPLICABLE: no group of this launch constructs an "
                        f"NCCL communicator, so there is nothing to allocate. {skipped}. "
                        "This is a configuration statement, not a measurement "
                        "of zero: it is void as soon as the transport changes, "
                        "and it is derived from the same predicates the "
                        "construction site branches on "
                        "(should_build_barlink / should_build_pynccl)"
                    ),
                    inputs=("communicator_groups", "SGLANG_BARLINK", "tp_size"),
                    not_applicable=True,
                )
            )
        elif nccl_mib is None:
            still_nccl = (
                " NCCL-owned group(s): "
                + ", ".join(f"{v.name} ({v.reason})" for v in nccl_building)
                + "."
                if nccl_building
                else ""
            )
            unbounded.append(
                f"{TERM_NCCL_BUFFERS} on {card.name}: allocated inside "
                "libnccl at communicator init and never sized by this "
                "process, so there is nothing to derive. Measure it once for "
                "this communicator set (VRAM delta around communicator init, "
                "or the NCCL_DEBUG=INFO alloc lines) and pass it as "
                "nccl_buffer_mib_per_gpu with its nccl_signature." + still_nccl
            )
        else:
            priced_for = (
                "; groups that build NCCL here: "
                + ", ".join(v.name for v in nccl_building)
                if nccl_building
                else ""
            )
            terms.append(
                LedgerTerm(
                    name=TERM_NCCL_BUFFERS,
                    mib=int(math.ceil(float(nccl_mib))) * n_co,
                    provenance=Provenance.CALIBRATED,
                    derivation=(
                        f"measured on {card.name}: {float(nccl_mib):.0f} MiB "
                        f"per rank process x {n_co} on this card, for the "
                        f"communicator set "
                        f"{inputs.nccl_signature or '<unstated>'}{priced_for}"
                    ),
                    inputs=("nccl_buffer_mib_per_gpu", "nccl_signature"),
                    bounded_by=(
                        "the measured communicator set; NCCL allocates at "
                        "init and does not grow with per-call message size"
                    ),
                    fingerprint=inputs.nccl_signature,
                )
            )

        residual = residual_by_uuid.get(card.uuid)
        residual_band = _history_band(history, card.uuid, "hardware_residual")
        processes = n_co + (1 if inputs.parent_binds_cuda_context else 0)
        if residual_band is not None and not residual_band.refused:
            # MEASURED beats CALIBRATED. The probe measures the residual of a
            # process that has just bound a context; the history measures the
            # residual of the processes this rig actually boots, on the wheel
            # and driver it actually runs. Where they disagree the boots win.
            terms.append(
                LedgerTerm(
                    name=TERM_HARDWARE_RESIDUAL,
                    mib=int(residual_band.charge_mib) * n_co,
                    provenance=Provenance.CALIBRATED,
                    derivation=(
                        f"boot-history band {residual_band.low_mib}-"
                        f"{residual_band.high_mib} MiB over "
                        f"{residual_band.n_boots} recorded boots of "
                        f"{card.name}, charged at the band's HIGH x {n_co} "
                        "rank process(es) on this card. Supersedes the probe "
                        "calibration"
                        + (
                            f" ({residual.total_mib} MiB, fingerprint "
                            f"{getattr(calibration, 'fingerprint', '')})"
                            if residual is not None
                            else ""
                        )
                    ),
                    fingerprint=f"boot-history:{residual_band.n_boots}",
                )
            )
            if inputs.parent_binds_cuda_context and residual is not None:
                terms.append(
                    LedgerTerm(
                        name=TERM_PARENT_CONTEXT,
                        mib=residual.cuda_context_bytes // (1 << 20),
                        provenance=Provenance.CALIBRATED,
                        derivation=(
                            "the parent/tokenizer process binds its own CUDA "
                            "primary context on this card (#237/#403)"
                        ),
                        fingerprint=getattr(calibration, "fingerprint", ""),
                    )
                )
        elif residual is None:
            unbounded.append(
                f"{TERM_HARDWARE_RESIDUAL} on {card.name}: no VRAM calibration "
                "matches this rig's fingerprint (card set / driver / wheel). "
                "Run `python -m sglang.srt.mem_ledger.probe` once; the result "
                "is cached until one of those three changes. This term is NOT "
                "defaulted to a constant -- a constant here is the "
                "_PREDICT_OVERHEAD_MIB guess this ledger replaces"
            )
        else:
            terms.append(
                LedgerTerm(
                    name=TERM_HARDWARE_RESIDUAL,
                    mib=residual.total_mib * n_co,
                    provenance=Provenance.CALIBRATED,
                    derivation=(
                        f"measured on {card.name}: CUDA primary context "
                        f"{residual.cuda_context_bytes // (1 << 20)} MiB + "
                        f"allocator granularity "
                        f"{residual.allocator_granularity_bytes // (1 << 20)} "
                        f"MiB + lazy kernel workspaces "
                        f"{residual.lazy_workspace_bytes // (1 << 20)} MiB, "
                        f"x {n_co} rank process(es) on this card"
                    ),
                    fingerprint=calibration.fingerprint,
                )
            )
            if inputs.parent_binds_cuda_context:
                terms.append(
                    LedgerTerm(
                        name=TERM_PARENT_CONTEXT,
                        mib=residual.cuda_context_bytes // (1 << 20),
                        provenance=Provenance.CALIBRATED,
                        derivation=(
                            "the parent/tokenizer process binds its own CUDA "
                            "primary context on this card (#237/#403); that "
                            "context costs the same measured bytes as a "
                            f"rank's. {processes} process(es) touch this card"
                        ),
                        fingerprint=calibration.fingerprint,
                    )
                )

        # -- the part of the card that is never allocatable -------------------
        # Charged ONCE per card, not once per rank: it is one reservation the
        # driver makes against the board, not a per-process cost. Without this
        # row the ledger spends MiB that no allocation can ever obtain, which
        # is #602: every card was budgeted against its nominal NVML total and
        # came up short by exactly this figure.
        if card.reserved_mib:
            terms.append(
                LedgerTerm(
                    name=TERM_NVML_CARVE_OUT,
                    mib=card.reserved_mib,
                    provenance=Provenance.REPORTED,
                    derivation=(
                        f"NVML v2 memory struct on {card.name} reports "
                        f"{card.reserved_mib} MiB reserved out of "
                        f"{card.total_mib} MiB total, leaving "
                        f"{card.total_mib - card.reserved_mib} MiB "
                        "allocatable. Read from the driver, not modelled: it "
                        "varies with card, driver and resizable-BAR state. "
                        "NVML also subtracts it from both 'used' and 'free', "
                        "so it cannot be recovered from total-used-free"
                    ),
                )
            )

        # -- co-resident tenants ---------------------------------------------
        for t in tenant_terms.get(card.gpu_id, ()):
            terms.append(t)

        ledgers.append(
            CardVramLedger(
                gpu_id=card.gpu_id,
                card=card.describe(),
                uuid=card.uuid,
                total_mib=card.total_mib,
                user_reserve_mib=int(user_reserve_mib.get(card.gpu_id, 0)),
                terms=tuple(terms),
                unbounded=tuple(unbounded),
                ranks=ranks,
            )
        )
    # #605: the MODELED side of the reconciliation, written wherever a boot
    # builds its ledger. Placed HERE, at the one function that constructs
    # them, and deliberately NOT at a caller: the first cut sat in
    # enforce_boot_contract, which only the --enable-vram-ledger path reaches,
    # so the production boot of 2026-08-05 21:11 wrote its marks and no
    # ledger. A dump that depends on which reserve path a flagset takes is a
    # dump that is missing exactly when someone finally looks.
    _dump_modeled_ledger(ledgers)
    return ledgers


def _dump_modeled_ledger(ledgers: Sequence[CardVramLedger]) -> None:
    """Hand the built ledger to the flight recorder. Never raises.

    Import is local and the whole body is guarded: constructing a ledger must
    not acquire a new way to fail, least of all one that only fires on the
    boots where the recorder is armed.
    """
    try:
        from sglang.srt.mem_ledger import flight_recorder

        flight_recorder.dump_ledger(ledgers)
    except Exception as e:  # pragma: no cover - never fail a boot over a dump
        logger.warning("could not hand the modeled ledger to the recorder: %s", e)


def demand_outside_budget_mib(ledger: CardVramLedger) -> int:
    """The card's demand MINUS the terms the rank budget itself funds.

    The quantity ``budget = (total - X) // colocated`` needs for X, together
    with the user reserve. See :data:`BUDGET_FUNDED_TERMS` for why the two
    numbers differ and why using ``ledger.demand_mib`` here would charge the
    weight shards and the SSM pool twice.
    """
    return sum(t.mib for t in ledger.terms if t.name not in BUDGET_FUNDED_TERMS)
