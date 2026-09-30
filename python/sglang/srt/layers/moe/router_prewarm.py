"""P-PREWARM (30.09.): the MoE router's first call made at boot, not in the
first prefill.

Measured reason (y4k 09301110 / y4l 09301150, P logs, 16384-token forwards,
FWD-TIMING-PREFILL, warm median n=216): the ``gate`` segment -- the span from
the shared expert's end to ``run_waves`` (router GEMM + top-k) -- costs on the
FIRST forward of every stage 112.4/110.2 ms (PP0), 102.7/107.3 (PP1),
91.9/89.7 (PP2) against warm medians 8 / 15 / 10 ms: ~+80-105 ms per stage,
once per process, independent of the stage's layer count (29 / 11 / 8). The
segment is fully exposed: the previous layer's ``run_waves`` ends in a host
rendezvous (the plan's D2H), so the device is idle while the host pays a cold
first call. The one cold loader the log shows in that span is the Triton
router ``_router_triton_kernel`` (``triton cold module load`` on every stage,
inside forward 1; the '[nan-49] PDL ...' line of the router's first
``is_arch_support_pdl`` is logged there too, 11:14:27 PP0) -- the first call of
``sglang.jit_kernel.moe_fused_gate`` in the process: module import, the
JITFunction's binder, the compile-from-cache and the module load. Its SECOND
specialization (M not a multiple of 16, forward 2: tok=65) costs only a few
ms (gate 13.3 / 6.4 / 4.2 ms), so the price is the first call, not the key.

What this does: per distinct router of this rank (the MoE blocks' own
``gate`` + ``topk``, deduplicated by geometry and top-k config) one call at
M=16 and one at M=17 through the serving entry -- ``block.gate(h)`` then
``block.topk(h, logits)`` -- i.e. the router's exact first call and both
Triton specializations the prefill uses (M % 16 == 0, and any other M), with
the serving dtype/strides. At the end of the scheduler init beside H101/H103
and P-COLD: after graph capture, before the #603b sampling barrier and the
first sleep. Rank-local, no collective, a few KiB of scratch; a prewarm
never kills a boot. Behind ``SGLANG_WEG2_ENABLE_TARGETED_PREWARM`` (off
until metal).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

#: the prewarm rows: one multiple of 16 and one not (Triton specialises an
#: int argument on divisibility by 16), so both keys a prefill meets are loaded
PREWARM_ROWS = (16, 17)


@dataclass(frozen=True)
class RouterPrewarmResult:
    routers: Tuple[str, ...]
    launched: int
    ms: float
    skipped: str = ""

    def line(self) -> str:
        if self.skipped:
            return f"P-PREWARM MOE-ROUTER skipped: {self.skipped}"
        return (
            f"P-PREWARM MOE-ROUTER routers=[{', '.join(self.routers)}] rows={list(PREWARM_ROWS)} "
            f"launches={self.launched} ms={self.ms:.1f} -- the router's first call "
            f"(import, binder, both Triton keys) is paid at boot, not in the first prefill"
        )


def _topk_key(cfg) -> tuple:
    return (
        getattr(cfg, "top_k", None),
        bool(getattr(cfg, "use_grouped_topk", False)),
        getattr(cfg, "topk_group", None),
        getattr(cfg, "num_expert_group", None),
        bool(getattr(cfg, "renormalize", True)),
        int(getattr(cfg, "num_fused_shared_experts", 0) or 0),
        getattr(cfg, "custom_routing_function", None) is not None,
        getattr(cfg, "correction_bias", None) is not None,
        str(getattr(cfg, "scoring_func", "softmax")),
        getattr(cfg, "routed_scaling_factor", None),
        bool(getattr(cfg, "apply_routed_scaling_factor_on_output", False)),
        str(getattr(cfg, "output_format", None)),
    )


def _hidden_size(gate) -> Optional[int]:
    size = getattr(gate, "input_size", None)
    if isinstance(size, int) and size > 0:
        return size
    w = getattr(gate, "weight", None)
    shape = getattr(w, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[1])
    return None


def router_blocks(modules: Iterable) -> list:
    """(label, block, hidden_size) of the first MoE block per distinct router
    in ``modules``: a module with a ``gate`` and a ``topk`` that carries a
    ``topk_config`` (``Qwen2MoeSparseMoeBlock`` and its kin). Empty on a rank
    without routed MoE layers."""
    out, seen = [], set()
    for m in modules:
        gate = getattr(m, "gate", None)
        topk = getattr(m, "topk", None)
        cfg = getattr(topk, "topk_config", None)
        if gate is None or cfg is None or not callable(gate) or not callable(topk):
            continue
        hidden = _hidden_size(gate)
        if hidden is None:
            continue
        key = (hidden, getattr(gate, "output_size", None), _topk_key(cfg))
        if key in seen:
            continue
        seen.add(key)
        label = (f"hidden={hidden} experts={getattr(gate, 'output_size', '?')} "
                 f"top_k={cfg.top_k} scoring={getattr(cfg, 'scoring_func', 'softmax')}")
        out.append((label, m, hidden))
    return out


def prewarm_routers(
    *,
    blocks,
    dtype,
    device,
    clock: Callable[[], float] = time.perf_counter,
) -> RouterPrewarmResult:
    """``block.gate(h)`` then ``block.topk(h, logits)`` per block and per
    :data:`PREWARM_ROWS` -- the serving entry, the serving dtype."""
    import torch

    t0 = clock()
    n = 0
    for _label, block, hidden in blocks:
        w = getattr(block.gate, "weight", None)
        wdt = getattr(w, "dtype", None)
        x_dtype = wdt if (wdt is not None and wdt.is_floating_point) else dtype
        for rows in PREWARM_ROWS:
            h = torch.zeros((rows, hidden), dtype=x_dtype, device=device)
            out = block.gate(h)
            logits = out[0] if isinstance(out, tuple) else out
            block.topk(h, logits)
            n += 1
    return RouterPrewarmResult(routers=tuple(b[0] for b in blocks), launched=n,
                               ms=(clock() - t0) * 1000.0)


def run_boot_prewarm(*, model, dtype, device) -> Optional[RouterPrewarmResult]:
    """The scheduler's delegate. Skips (named) when the switch is off, on a
    Form-A worker (routes nothing) and on a rank without routed MoE blocks.
    Never raises."""
    try:
        from sglang.srt.environ import envs

        if not envs.SGLANG_WEG2_ENABLE_TARGETED_PREWARM.get():
            res = RouterPrewarmResult((), 0, 0.0, skipped="SGLANG_WEG2_ENABLE_TARGETED_PREWARM off")
            logger.info("%s", res.line())
            return res
        from sglang.srt.rank_role import this_rank_is_form_a_worker

        if this_rank_is_form_a_worker():
            res = RouterPrewarmResult((), 0, 0.0, skipped="Form-A worker (routes nothing)")
            logger.info("%s", res.line())
            return res
        blocks = router_blocks(model.modules()) if model is not None else []
        if not blocks:
            res = RouterPrewarmResult((), 0, 0.0, skipped="no routed MoE block on this rank")
            logger.info("%s", res.line())
            return res
        import torch

        with torch.inference_mode():
            res = prewarm_routers(blocks=blocks, dtype=dtype, device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info("%s", res.line())
        return res
    except Exception as exc:  # noqa: BLE001 -- a prewarm never kills a boot
        logger.warning("P-PREWARM MOE-ROUTER failed (%s: %s); the router loads on the "
                       "first extend", type(exc).__name__, str(exc)[:200])
        return None
