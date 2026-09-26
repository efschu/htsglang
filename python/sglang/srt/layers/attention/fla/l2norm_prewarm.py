"""H103: load the run-time-T FLA l2norm kernel BEFORE serving starts.

Measured reason (rc9p, dkrnfbar1rc9p09261011, tree d1c7094ba6): the stock
``l2norm_fwd_kernel`` takes the row count ``T`` (tokens x heads) and
``NB = cdiv(T, 2048)`` as ``tl.constexpr``, so every new extend length is a new
Triton kernel, compiled or read from the disk cache and loaded on the
scheduler thread inside the launch. D log: 15 cold-load windows of that one
kernel after READY, all on TP0 in D extends, 16.7 s summed, one of them 12.7 s
(10:34:11, the X-direct extend weg2-22-25); P log: 78 more (26 per stage).
Every other FLA/GDN kernel already takes T at run time
(``do_not_specialize=["T"]``) and loads a bounded number of times.

``l2norm.l2norm_fwd_kernel_rt`` (27B strand, 6b24aa60da, taken verbatim) is the
same kernel body with T a run-time, unspecialised argument: ONE specialization
per (dtype, D, BD). That one would still be loaded by the first extend after
READY; this module loads it at boot instead, next to the other boot prewarms
(scheduler: after graph capture, before the #603b sampling barrier and the
first sleep). Rank-local, no collective; a prewarm never kills a boot.

The launch it builds: ``l2norm_fwd(q)`` on a [16, D] tensor of the model dtype
-- exactly what ``ChunkGatedDeltaRuleFunction.forward`` passes per head after
``x.view(-1, D)`` (D = ``linear_key_head_dim``, q and k both), so the key is the
serving key: dtype pointers (16-byte aligned, like every fresh allocation),
eps (float, never specialised), T (not specialised), D/BT/BD constexpr.
4 KiB of scratch, freed on return.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)

#: rows of the prewarm launch: one BT=16 tile, any value works (T is run-time)
PREWARM_ROWS = 16


@dataclass(frozen=True)
class L2NormPrewarmResult:
    head_dims: tuple
    dtype: str
    loaded: int
    ms: float
    skipped: str = ""

    def line(self) -> str:
        if self.skipped:
            return f"H103 FLA-L2NORM-PREWARM skipped: {self.skipped}"
        return (
            f"H103 FLA-L2NORM-PREWARM kernel=l2norm_fwd_kernel_rt head_dims="
            f"{list(self.head_dims)} dtype={self.dtype} launches={self.loaded} "
            f"ms={self.ms:.1f} -- T is a run-time bound, so no extend length "
            f"loads this kernel again after READY"
        )


def gdn_l2norm_head_dims(hf_text_config) -> tuple:
    """The D values the chunked GDN path normalises: q and k both carry
    ``linear_key_head_dim``. Empty for a model without GDN layers."""
    d = getattr(hf_text_config, "linear_key_head_dim", None)
    if not isinstance(d, int) or d <= 0:
        return ()
    return (d,)


def prewarm_l2norm(
    *,
    head_dims: Sequence[int],
    dtype,
    device,
    launch: Optional[Callable] = None,
    clock: Callable[[], float] = time.perf_counter,
) -> L2NormPrewarmResult:
    """Launch the run-time-T kernel once per D (only D <= 512 takes it; the
    D > 512 kernel1 has no T specialization at all)."""
    import torch

    from sglang.srt.layers.attention.fla import l2norm as l2

    launch = launch or l2.l2norm_fwd
    t0 = clock()
    n = 0
    dims = tuple(int(d) for d in head_dims if 0 < int(d) <= 512)
    for d in dims:
        x = torch.zeros((PREWARM_ROWS, d), dtype=dtype, device=device)
        launch(x)
        n += 1
    return L2NormPrewarmResult(
        head_dims=dims, dtype=str(dtype).replace("torch.", ""), loaded=n,
        ms=(clock() - t0) * 1000.0,
    )


def run_boot_prewarm(*, hf_text_config, dtype, device) -> Optional[L2NormPrewarmResult]:
    """The scheduler's delegate. Skips (named) when the run-time kernel is not
    selected (the stock kernel specialises per T, nothing to prewarm), on a
    Form-A worker (runs no GDN layer: rc9p D, TP1/TP2 loaded no chunk kernel)
    and on a model without GDN layers. Never raises."""
    try:
        from sglang.srt.layers.attention.fla import l2norm as l2

        if not l2.l2norm_runtime_t_on():
            res = L2NormPrewarmResult((), "", 0, 0.0,
                                      skipped=f"{l2.L2NORM_RUNTIME_T_ENV} not '1' (stock kernel)")
            logger.info("%s", res.line())
            return res
        from sglang.srt.rank_role import this_rank_is_form_a_worker

        if this_rank_is_form_a_worker():
            res = L2NormPrewarmResult((), "", 0, 0.0, skipped="Form-A worker (no GDN layer)")
            logger.info("%s", res.line())
            return res
        dims = gdn_l2norm_head_dims(hf_text_config)
        if not dims:
            res = L2NormPrewarmResult((), "", 0, 0.0, skipped="no linear_key_head_dim (no GDN layer)")
            logger.info("%s", res.line())
            return res
        import torch

        res = prewarm_l2norm(head_dims=dims, dtype=dtype, device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info("%s", res.line())
        return res
    except Exception as exc:  # noqa: BLE001 -- a prewarm never kills a boot
        logger.warning("H103 FLA-L2NORM-PREWARM failed (%s: %s); the kernel loads lazily",
                       type(exc).__name__, str(exc)[:200])
        return None
