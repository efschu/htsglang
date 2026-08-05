"""CPU expert compute lane -- Slice 1: the executor and the slot-feed seam.

WHAT THIS IS

The MoE expert offload path keeps a hot subset of a layer's routed experts
resident on the GPU and streams the rest host->device on demand. Streaming an
expert costs its WEIGHTS over PCIe (a fixed ``3 * hidden * inter * dtype`` per
event, independent of how many tokens routed to it). This lane replaces that
fetch for a chosen subset of experts: the expert is computed ON THE CPU over a
pinned host-resident int8 shard, and only its OUTPUT ROWS cross the link.

The traffic ratio is the attraction -- for Qwen3.5-35B-A3B one expert is 6.29 MB
of weights against 8 KB of activations for a single token, roughly 800:1.

WHY INT8, AND WHY NOTHING IS EVER DEQUANTISED HERE

An earlier fp32 variant of this lane was probed and rejected: dequantising a
GPTQ-Int4 expert to fp32 costs 6.177 ms, roughly ten times the H2D fetch it was
meant to replace. Widening from int8 to fp32 is cheaper but still 2.831 ms --
also far above the entire fetch. The conclusion carried into this design is
absolute: **the lane computes directly on the int8 bytes and never widens a
weight.** Any future edit that introduces a per-event ``.float()`` on a weight
tensor re-opens a defect that has already been measured and rejected; the
regression test ``test_cpu_expert_lane`` pins it.

The int8 GEMM used is fbgemm's (AVX2 integer path), reached through torch's
dynamic-quantisation prepacked Linear. That choice is measured, not assumed:

    path                              M=1        M=64
    fbgemm dynamic int8            0.219 ms   0.560 ms   (719 GF/s)
    torch fp32 (oneDNN sgemm)      0.507 ms   1.215 ms
    aten::_weight_int8pack_mm      0.236 ms   23.4 ms    (GEMV only)
    torch._int_mm                  0.518 ms   46.4 ms

(Qwen3.5-35B-A3B shapes, 16 threads, rotating expert pool so weights come from
DRAM rather than L3.) Only the first is a real batched integer GEMM; the other
two int8 entry points collapse for M > 1 and must not be used.

THREADING

Cost is flat in M up to roughly M=8 -- the lane is DRAM-bandwidth-bound there,
not FLOP-bound, because reading the expert's 3.15 MB dominates. Two consequences
shape the executor:

* MTP/NEXTN verify batches are nearly free IN W8A8. Going from M=1 to M=4 costs
  7 % on the 35B shapes (0.183 -> 0.196 ms), so a verify batch of
  ``num_draft_tokens + 1`` rides along at almost the price of a single decode
  token. This does NOT hold for W8A32, which grows linearly (0.330 -> 1.271 ms).
* Wide intra-expert threading does not pay. At M=1 four threads BEAT sixteen
  (0.174 ms vs 0.219 ms) because thread fan-out overhead exceeds the work. The
  executor therefore keeps intra-op threads LOW and parallelises ACROSS experts,
  which is also what leaves cores for the serving process.

NUMERICS

int8 CPU compute is not bit-identical with the GPU's grouped GEMM. This lane is
lossy by arithmetic and is opt-in only; see ``CpuExpertLaneConfig``.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

#: Intra-op threads used for a single expert's GEMMs. Measured optimum is low
#: (see module docstring); the parallelism that matters is across experts.
DEFAULT_INTRA_OP_THREADS = 4

#: fbgemm is the AVX2 integer GEMM. "x86" resolves to the same kernels on this
#: class of host; onednn and qnnpack were measured slower and are not defaults.
DEFAULT_QUANT_ENGINE = "fbgemm"


#: Weight-only int8: weights are int8, activations stay fp32. Accurate
#: (~1.3e-2 relative, inside the accepted lossy-offload band) but the kernel is
#: GEMV-only, so cost grows linearly with M.
MODE_W8A32 = "w8a32"

#: Dynamic int8: both weights and activations are int8. Roughly 3x less
#: accurate (~3.5-5e-2) but a real batched GEMM, flat in M up to ~8.
MODE_W8A8 = "w8a8"

#: Above this many rows, W8A32 stops beating even the SLOWEST link (the x4
#: 3080 at 6.45 GB/s) and therefore stops being worth its accuracy advantage.
#: Measured: on the 35B shapes W8A32 costs 0.330 ms at M=1 and 0.662 ms at M=2
#: against a 0.975 ms fetch, but 1.271 ms at M=4 -- past the fetch it replaces.
#: See docs/dev/DESIGN_CPULANE.md for the full table.
W8A32_ROW_LIMIT = 2

#: Always use the fast batched GEMM. W8A8 measured faster than W8A32 at EVERY
#: row count, including M=1 (0.183 ms vs 0.330 ms on the 35B shapes with a
#: cold, DRAM-resident expert pool), so this is the throughput default.
PREFER_SPEED = "speed"

#: Use the accurate weight-only kernel while it still beats the link, and fall
#: back to W8A8 above W8A32_ROW_LIMIT rather than losing to the fetch.
PREFER_ACCURACY = "accuracy"


class CpuExpertLaneError(RuntimeError):
    """Raised for configuration or contract violations in the CPU lane."""


@dataclass
class CpuExpertLaneConfig:
    """Operator-facing configuration for the lane.

    The lane is OFF unless ``enabled`` is set. It changes numerics, so it never
    turns itself on as a side effect of any other feature.
    """

    enabled: bool = False
    #: Max experts computed concurrently on the CPU.
    max_workers: int = 4
    #: Intra-op threads per expert GEMM.
    intra_op_threads: int = DEFAULT_INTRA_OP_THREADS
    quant_engine: str = DEFAULT_QUANT_ENGINE
    #: PREFER_SPEED (default) or PREFER_ACCURACY. See Int8ExpertShard.select_mode.
    prefer: str = PREFER_SPEED

    def validate(self) -> None:
        if self.max_workers < 1:
            raise CpuExpertLaneError(
                f"cpu expert lane: max_workers must be >= 1, got {self.max_workers}"
            )
        if self.intra_op_threads < 1:
            raise CpuExpertLaneError(
                f"cpu expert lane: intra_op_threads must be >= 1, got "
                f"{self.intra_op_threads}"
            )
        if self.prefer not in (PREFER_SPEED, PREFER_ACCURACY):
            raise CpuExpertLaneError(
                f"cpu expert lane: prefer must be {PREFER_SPEED!r} or "
                f"{PREFER_ACCURACY!r}, got {self.prefer!r}"
            )
        supported = list(torch.backends.quantized.supported_engines)
        if self.quant_engine not in supported:
            raise CpuExpertLaneError(
                f"cpu expert lane: quant engine {self.quant_engine!r} is not "
                f"available in this torch build; supported: {supported}"
            )


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class Int8ExpertShard:
    """One expert's gate/up/down projections, int8-quantised and prepacked.

    Quantisation and prepacking happen ONCE, at construction. ``forward``
    performs no dequantisation, no widening, and no repacking -- see the module
    docstring for why that is a hard rule rather than an optimisation.

    Both modes are built, because they trade against each other:

    * ``MODE_W8A8`` is FASTER AT EVERY MEASURED ROW COUNT, M=1 included, so it
      is the throughput default. It quantises activations as well as weights,
      which costs accuracy (~3.4e-2 at M=1, degrading to ~6.7e-2 at M=33 because
      torch picks one activation scale per batch).
    * ``MODE_W8A32`` leaves activations in fp32, so its only error is weight
      quantisation (~1.1-1.6e-2, flat in M, the same band as the already-
      accepted marlin offload). It is the accuracy option and it is not free:
      about 1.8x slower at M=1, and past ``W8A32_ROW_LIMIT`` it is slower than
      the H2D fetch it replaces.

    An earlier reading of this comparison had W8A32 roughly free at M=1. That
    was an artefact of benchmarking one L3-resident expert; against a rotating
    DRAM-resident pool W8A8 wins at M=1 by 0.183 ms to 0.330 ms.

    Holding both costs one extra int8 copy of the expert -- two int8 copies is
    exactly the size of the one bf16 copy this tier displaces, so a dual-mode
    tier is RAM-neutral. ``select_mode`` encodes the switch.
    """

    __slots__ = ("gate_q", "up_q", "down_q", "gate_s", "up_s", "down_s",
                 "gate_d", "up_d", "down_d", "hidden", "inter", "modes")

    def __init__(
        self,
        gate_w: torch.Tensor,
        up_w: torch.Tensor,
        down_w: torch.Tensor,
        engine: str = DEFAULT_QUANT_ENGINE,
        modes: Tuple[str, ...] = (MODE_W8A32, MODE_W8A8),
    ) -> None:
        inter, hidden = gate_w.shape
        if up_w.shape != (inter, hidden):
            raise CpuExpertLaneError(
                f"cpu expert lane: up_proj shape {tuple(up_w.shape)} does not "
                f"match gate_proj {(inter, hidden)}"
            )
        if down_w.shape != (hidden, inter):
            raise CpuExpertLaneError(
                f"cpu expert lane: down_proj shape {tuple(down_w.shape)} does "
                f"not match the expected {(hidden, inter)}"
            )
        for m in modes:
            if m not in (MODE_W8A32, MODE_W8A8):
                raise CpuExpertLaneError(f"cpu expert lane: unknown mode {m!r}")
        self.hidden = hidden
        self.inter = inter
        self.modes = tuple(modes)
        torch.backends.quantized.engine = engine

        if MODE_W8A32 in modes:
            self.gate_q, self.gate_s = self._quant_weight_only(gate_w)
            self.up_q, self.up_s = self._quant_weight_only(up_w)
            self.down_q, self.down_s = self._quant_weight_only(down_w)
        if MODE_W8A8 in modes:
            self.gate_d = self._pack_dynamic(gate_w)
            self.up_d = self._pack_dynamic(up_w)
            self.down_d = self._pack_dynamic(down_w)

    @staticmethod
    def _quant_weight_only(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Symmetric per-output-channel int8. Returns (int8 weight, fp32 scale)."""
        w = w.to(torch.float32)
        scale = (w.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
        q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
        return q.contiguous(), scale.contiguous()

    @staticmethod
    def _pack_dynamic(w: torch.Tensor) -> torch.nn.Module:
        out_f, in_f = w.shape
        lin = torch.nn.Linear(in_f, out_f, bias=False)
        with torch.no_grad():
            lin.weight.copy_(w.to(torch.float32))
        return torch.ao.quantization.quantize_dynamic(
            torch.nn.Sequential(lin), {torch.nn.Linear}, dtype=torch.qint8
        )

    def select_mode(self, m_rows: int, prefer: str = PREFER_SPEED) -> str:
        """Pick the kernel for a batch of ``m_rows`` rows under ``prefer``.

        W8A8 is faster at every measured row count, so ``PREFER_SPEED`` never
        chooses W8A32. ``PREFER_ACCURACY`` buys the ~3x tighter error band at a
        real cost (about 1.8x at M=1) and only while that cost still undercuts
        the H2D fetch it replaces -- past ``W8A32_ROW_LIMIT`` it would be slower
        than simply streaming the weights, so it hands back to W8A8.
        """
        if prefer == PREFER_ACCURACY and m_rows <= W8A32_ROW_LIMIT and MODE_W8A32 in self.modes:
            return MODE_W8A32
        if MODE_W8A8 in self.modes:
            return MODE_W8A8
        return self.modes[0]

    def forward(
        self, x: torch.Tensor, mode: Optional[str] = None, prefer: str = PREFER_SPEED
    ) -> torch.Tensor:
        """SwiGLU expert FFN over ``x`` of shape [M, hidden] -> [M, hidden]."""
        mode = mode or self.select_mode(x.shape[0], prefer=prefer)
        if mode == MODE_W8A32:
            mm = torch.ops.aten._weight_int8pack_mm
            g = mm(x, self.gate_q, self.gate_s)
            u = mm(x, self.up_q, self.up_s)
            return mm((_silu(g) * u).contiguous(), self.down_q, self.down_s)
        return self.down_d(_silu(self.gate_d(x)) * self.up_d(x))


class CpuExpertPool:
    """The host-resident int8 expert tier for ONE MoE layer.

    Sized to be affordable: at int8 an expert is HALF the bytes of the existing
    bf16 pinned pool, so this tier does not compete with that pool for RAM the
    way an fp32 tier would. For Qwen3.5-35B-A3B all 256 x 40 experts occupy
    ~32 GB at int8 against ~64 GB at bf16 and ~129 GB at fp32 -- which is why
    the int8 route survives the RAM wall that killed the fp32 one.
    """

    def __init__(
        self,
        layer_id: int,
        hidden: int,
        inter: int,
        engine: str = DEFAULT_QUANT_ENGINE,
        modes: Tuple[str, ...] = (MODE_W8A32, MODE_W8A8),
    ):
        self.layer_id = layer_id
        self.hidden = hidden
        self.inter = inter
        self.engine = engine
        self.modes = tuple(modes)
        self._shards: Dict[int, Int8ExpertShard] = {}

    def add_expert(
        self, expert_id: int, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor
    ) -> None:
        self._shards[int(expert_id)] = Int8ExpertShard(
            gate_w, up_w, down_w, engine=self.engine, modes=self.modes
        )

    def has(self, expert_id: int) -> bool:
        return int(expert_id) in self._shards

    def get(self, expert_id: int) -> Int8ExpertShard:
        try:
            return self._shards[int(expert_id)]
        except KeyError:
            raise CpuExpertLaneError(
                f"cpu expert lane: layer {self.layer_id} has no CPU-resident "
                f"shard for expert {expert_id}; the placement decision routed an "
                f"expert to the CPU lane that was never admitted to the pool"
            ) from None

    @property
    def num_experts(self) -> int:
        return len(self._shards)

    def bytes_resident(self) -> int:
        """Host bytes held by this tier. One int8 copy of the expert per mode."""
        return self.num_experts * len(self.modes) * 3 * self.hidden * self.inter


@dataclass
class ExpertJob:
    """One expert's share of a step: which rows of the activation it consumes.

    ``rows`` indexes into the caller's activation tensor AND into the output
    buffer, so the executor never has to know the global token order.
    """

    expert_id: int
    rows: torch.Tensor  # int64 [M]
    #: Kernel override. None lets the shard pick from the row count.
    mode: Optional[str] = None


class CpuExpertExecutor:
    """Computes a set of experts on CPU threads and writes rows into a buffer.

    Parallelism is ACROSS experts. Each worker runs one expert's three GEMMs at
    a low intra-op thread count; fbgemm releases the GIL during compute, so the
    Python-level thread pool yields real concurrency.
    """

    def __init__(self, config: CpuExpertLaneConfig):
        config.validate()
        self.config = config
        self._pool = ThreadPoolExecutor(
            max_workers=config.max_workers, thread_name_prefix="cpu-expert-lane"
        )
        self._lock = threading.Lock()
        torch.backends.quantized.engine = config.quant_engine

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)

    def run(
        self,
        pool: CpuExpertPool,
        activations: torch.Tensor,
        jobs: Sequence[ExpertJob],
        out: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> int:
        """Compute ``jobs`` over ``activations`` and scatter results into ``out``.

        ``activations`` is [T, hidden] fp32 on CPU. ``out`` is [T, hidden] fp32
        on CPU -- typically the pinned staging mirror of a #462 bridge buffer.
        ``weights`` is an optional [T] fp32 vector of router gate values applied
        to each row before accumulation.

        Rows of ``out`` not covered by any job are left untouched, so the caller
        controls whether the buffer is zero-initialised or accumulated into.

        Returns the number of expert calls executed.
        """
        if activations.device.type != "cpu" or out.device.type != "cpu":
            raise CpuExpertLaneError(
                "cpu expert lane: activations and out must be CPU tensors "
                f"(got {activations.device} and {out.device})"
            )
        if activations.shape[1] != pool.hidden or out.shape[1] != pool.hidden:
            raise CpuExpertLaneError(
                f"cpu expert lane: hidden mismatch -- pool {pool.hidden}, "
                f"activations {tuple(activations.shape)}, out {tuple(out.shape)}"
            )
        if not jobs:
            return 0

        prev_threads = torch.get_num_threads()
        torch.set_num_threads(self.config.intra_op_threads)
        try:
            futures = [
                self._pool.submit(
                    self._run_one, pool, activations, job, out, weights, self.config.prefer
                )
                for job in jobs
            ]
            for f in futures:
                f.result()
        finally:
            torch.set_num_threads(prev_threads)
        return len(jobs)

    @staticmethod
    def _run_one(
        pool: CpuExpertPool,
        activations: torch.Tensor,
        job: ExpertJob,
        out: torch.Tensor,
        weights: Optional[torch.Tensor],
        prefer: str = PREFER_SPEED,
    ) -> None:
        shard = pool.get(job.expert_id)
        x = activations.index_select(0, job.rows)
        # Mode is chosen per JOB, not per step: a decode step and an MTP verify
        # step differ in rows-per-expert, and so does the right kernel.
        y = shard.forward(x, mode=job.mode, prefer=prefer)
        if weights is not None:
            y = y * weights.index_select(0, job.rows).unsqueeze(1)
        # Distinct jobs own distinct rows, so this scatter needs no lock. The
        # contract is asserted by the caller (build_jobs produces a partition).
        out.index_copy_(0, job.rows, y)


def build_jobs(topk_ids: torch.Tensor, cpu_expert_ids: Sequence[int]) -> List[ExpertJob]:
    """Turn a routing result into per-expert row lists for the CPU lane.

    ``topk_ids`` is [T, top_k] (or [T] for top_k=1). Only experts listed in
    ``cpu_expert_ids`` produce jobs; everything else stays on the GPU path.

    The returned jobs partition their rows: a given (row, expert) pair appears
    once. That is what allows :meth:`CpuExpertExecutor.run` to scatter without
    a lock.
    """
    wanted = set(int(e) for e in cpu_expert_ids)
    if not wanted:
        return []
    if topk_ids.dim() == 1:
        topk_ids = topk_ids.unsqueeze(1)
    flat_rows: Dict[int, List[int]] = {}
    ids = topk_ids.tolist()
    for row, experts in enumerate(ids):
        for e in experts:
            e = int(e)
            if e in wanted:
                flat_rows.setdefault(e, []).append(row)
    return [
        ExpertJob(expert_id=e, rows=torch.tensor(rows, dtype=torch.int64))
        for e, rows in sorted(flat_rows.items())
    ]


class CpuLaneSlotFeed:
    """The seam against the #462 breakable route.

    #462's discipline is that the captured graph addresses SLOTS at fixed device
    addresses, and the EAGER phase decides what occupies them, materialises the
    bytes, and publishes the mapping -- all before the replay that reads them.
    A CPU-computed expert is the same shape with a different producer: where the
    normal path fills a slot with an expert's WEIGHTS via H2D, this lane fills an
    output bridge with that expert's RESULT ROWS.

    The contract this class enforces, and which ``test_cpu_expert_lane`` pins:

    1. ``stage`` is the pinned host mirror; ``buf`` is the fixed-address device
       buffer the captured segment reads. They are allocated together and never
       reallocated, matching ``BreakableBridge``.
    2. All CPU compute completes BEFORE ``publish()`` issues the H2D copy. There
       is no path on which a replay reads a half-written bridge.
    3. ``publish()`` is the only writer of ``buf``, and it performs exactly ONE
       H2D copy per layer per step -- the payload is activations-out, not
       weights, which is the entire economic argument for the lane.

    Slice 1 holds this as a contract object testable without a GPU: ``buf`` may
    be any tensor-like with ``copy_``. Slice 2 binds it to the real arena.
    """

    def __init__(self, layer_id: int, stage: torch.Tensor, buf) -> None:
        if stage.device.type != "cpu":
            raise CpuExpertLaneError(
                f"cpu expert lane: stage must be a CPU (pinned) tensor, got "
                f"{stage.device}"
            )
        self.layer_id = layer_id
        self.stage = stage
        self.buf = buf
        self._published = False
        self._compute_done = False

    def begin_step(self) -> torch.Tensor:
        """Reset for a new step and hand out the buffer the executor writes."""
        self._published = False
        self._compute_done = False
        self.stage.zero_()
        return self.stage

    def mark_compute_done(self) -> None:
        self._compute_done = True

    def publish(self) -> None:
        """Copy the staged results into the graph-addressed device buffer."""
        if not self._compute_done:
            raise CpuExpertLaneError(
                f"cpu expert lane, layer {self.layer_id}: publish() called "
                "before mark_compute_done(). A captured replay would read a "
                "partially written bridge and return a wrong result. The eager "
                "phase must complete ALL CPU expert compute before publishing."
            )
        if self._published:
            raise CpuExpertLaneError(
                f"cpu expert lane, layer {self.layer_id}: publish() called "
                "twice in one step; the bridge takes exactly one H2D copy per "
                "layer per step."
            )
        self.buf.copy_(self.stage)
        self._published = True

    @property
    def published(self) -> bool:
        return self._published
