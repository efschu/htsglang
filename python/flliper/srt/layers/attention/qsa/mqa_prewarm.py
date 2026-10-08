"""P-COLD: build the QSA indexer's TileLang prefill kernels BEFORE serving starts.

Measured reason (rc12z30c, boot ...09282039_f33274eac0_0928_203953, P log): the
first P forward after the boot took ~21 s for 30 new tokens (pdflip-0-6 on a
16896-token store hit). PP1 alone: '#1466 PASS-STALL pp_rank=1 ... fwd_ms=12783'
(20:46:00). Every Triton window of that forward closed after 0.0 s; the gap sits
between two of them -- '_fused_qk_rmsnorm_rope_gate_kernel' closed 20:45:49,
the next launch '_expand_qsa_block_indices_kernel' 20:45:58 -- and exactly that
gap is the TileLang JIT of the indexer (stderr, P log Z. 28557-28560):
'TileLang begins to compile kernel' 20:45:49 -> 'completes' 20:45:55 (the MQA
prefill kernel) and 20:45:55 -> 20:45:58 (the mask kernel). Same shape in the
boots of 20:16, 20:59 and 21:10 (PP1 fwd_ms 15622 / 16432 / 14214). PP2 (same
sm86) then hit PP1's disk cache; PP0 (sm120) was warm from D's own build.

Why nothing warmed it: ``@tilelang.jit`` compiles per process on the first call
of a (heads, head_dim, block_q) key, the P group runs no forward before its
first sleep, and the existing boot prewarms (H101 rows forms, H103 l2norm) are
Triton kernels. So the first real request of the boot paid ~9 s of compiler on
PP1 -- seen by the user as the stall after the first flip.

What this does: one tiny launch per indexer geometry on this rank, through the
serving entry ``tilelang_qsa_mqa_prefill`` (so both kernels -- the packed MQA
prefill and the mask -- are built with the serving key: heads/head_dim of the
``QSAIndexer`` modules this rank holds, ``block_q = 128 // heads``, bf16 q/k,
fp32 logits, int32 row bounds; rows and keys are ``T.dynamic``, so one build
serves every extend length). At the end of the scheduler init next to H101/H103:
after graph capture, before the #603b sampling barrier, the first sleep and
READY. The cost moves into the boot, never into a flip. Rank-local, no
collective; a few KiB of scratch; a prewarm never kills a boot.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: keys of the prewarm launch: one block_n=64 tile (keys are run-time)
PREWARM_KEYS = 64


@dataclass(frozen=True)
class MqaPrewarmResult:
    signatures: Tuple[Tuple[int, int], ...]
    launched: int
    ms: float
    skipped: str = ""

    def line(self) -> str:
        if self.skipped:
            return f"P-COLD QSA-MQA-TILELANG-PREWARM skipped: {self.skipped}"
        sigs = ", ".join(f"heads={h} head_dim={d}" for h, d in self.signatures)
        return (
            f"P-COLD QSA-MQA-TILELANG-PREWARM kernels=[mqa_prefill, mqa_mask] "
            f"sigs=[{sigs}] launches={self.launched} ms={self.ms:.1f} -- rows and "
            f"keys are run-time, so no extend after READY compiles them again"
        )


def indexer_prefill_signatures(modules: Iterable) -> Tuple[Tuple[int, int], ...]:
    """(heads, head_dim) of every ``QSAIndexer`` in ``modules`` -- the q shape
    ``select_prefill_tokens`` hands the prefill kernel ([rows, index_n_heads,
    index_head_dim], unpadded on extend). Empty on a rank without QSA layers."""
    from flliper.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    out = []
    for m in modules:
        if not isinstance(m, QSAIndexer):
            continue
        sig = (int(m.index_n_heads), int(m.index_head_dim))
        if sig not in out:
            out.append(sig)
    return tuple(out)


def prewarm_mqa(
    *,
    signatures: Sequence[Tuple[int, int]],
    device,
    launch: Optional[Callable] = None,
    clock: Callable[[], float] = time.perf_counter,
) -> MqaPrewarmResult:
    """One serving-keyed launch per signature: rows = one block_q tile, keys =
    one block_n tile, every row attends every key."""
    import torch

    from flliper.srt.layers.attention.qsa import mqa

    launch = launch or mqa.tilelang_qsa_mqa_prefill
    t0 = clock()
    n = 0
    for heads, head_dim in signatures:
        rows = max(1, 128 // int(heads))
        q = torch.zeros((rows, heads, head_dim), dtype=torch.bfloat16, device=device)
        k = torch.zeros((PREWARM_KEYS, 1, head_dim), dtype=torch.bfloat16, device=device)
        starts = torch.zeros((rows,), dtype=torch.int32, device=device)
        ends = torch.full((rows,), PREWARM_KEYS, dtype=torch.int32, device=device)
        launch(q, k, starts, ends)
        n += 1
    return MqaPrewarmResult(signatures=tuple(signatures), launched=n,
                            ms=(clock() - t0) * 1000.0)


def run_boot_prewarm(*, model, device) -> Optional[MqaPrewarmResult]:
    """The scheduler's delegate. Skips (named) without TileLang (the torch
    fallback compiles nothing), on a Form-A worker (attends nothing) and on a
    rank whose model holds no QSA indexer. Never raises."""
    try:
        from flliper.srt.layers.attention.qsa import mqa

        if not mqa.HAS_TILELANG:
            res = MqaPrewarmResult((), 0, 0.0, skipped="no TileLang (torch fallback)")
            logger.info("%s", res.line())
            return res
        from flliper.srt.rank_role import this_rank_is_form_a_worker

        if this_rank_is_form_a_worker():
            res = MqaPrewarmResult((), 0, 0.0, skipped="Form-A worker (attends nothing)")
            logger.info("%s", res.line())
            return res
        sigs = indexer_prefill_signatures(model.modules()) if model is not None else ()
        if not sigs:
            res = MqaPrewarmResult((), 0, 0.0, skipped="no QSAIndexer on this rank")
            logger.info("%s", res.line())
            return res
        import torch

        res = prewarm_mqa(signatures=sigs, device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info("%s", res.line())
        return res
    except Exception as exc:  # noqa: BLE001 -- a prewarm never kills a boot
        logger.warning("P-COLD QSA-MQA-TILELANG-PREWARM failed (%s: %s); the kernels "
                       "compile on the first extend", type(exc).__name__, str(exc)[:200])
        return None
