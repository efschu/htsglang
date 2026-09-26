"""H101: load every QSA rows launch form at boot, before the first sleep.

WHY. The rows kernel (sparse_attn._sparse_attn_rows_fwd) is built per launch
form (BLOCK_N / warps / stages from the rows table, USE_COUNTS) -- and every
form's FIRST launch is a cold module load (#1056 window) and, if the build
spills, a growth of the context's local memory inside cuLaunchKernel. rc9p
D-TP0 met its four forms over 12 minutes of serving (10:24:06 32/8/2, 10:26:33
64/8/2, 10:28:30 32/4/2, 10:34:24 16/1/2) and died at the fourth: an X-direct
extend of 3585 rows on a 23744-token prefix was the first request of the boot
with a prefix AND more than 512 new rows.

WHAT. One tiny launch per form and per USE_COUNTS value, at the end of the
scheduler init (after graph capture, before the sampling warmup's barrier and
the first sleep): q = zeros [T, heads, head_dim], rows = -1 (nothing attended,
the loop is masked), T = the smallest row count that selects the form. The
specialization (heads, head_dim, dtype, the rows width K and the pool tensors,
all constexpr or dtype keys) is the one the capture's own launch recorded
(sparse_attn.rows_prewarm_signature), so the prewarm builds exactly the
variants serving will launch. Each load passes the #1056 chokepoint, where
H101's census reads its LOCAL_SIZE and pre-grows the context stack
(utils/lmem_census.py) -- so the whole local-memory need of the rows kernel is
reached HERE, measured, printed per rank (``H101 QSA-ROWS-PREWARM``), carried
by the first WEG2-SLEEP-LMEM line and restored at every wake.

Fail-soft: no recorded launch (no attention on this rank, graphs disabled) is
a named skip; a launch that raises is logged and the next form still runs --
the lazy path stays what it was.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional, Sequence, Tuple

import msgspec

logger = logging.getLogger(__name__)

Form = Tuple[int, Tuple[int, int, int]]


class PrewarmResult(msgspec.Struct, frozen=True, kw_only=True):
    status: str
    forms: Tuple[str, ...] = ()
    stack_before: Optional[int] = None
    stack_after: Optional[int] = None
    census_max_bytes: int = 0
    census_kernel: str = ""
    ms: float = 0.0
    errors: Tuple[str, ...] = ()

    def line(self, threads: Optional[int] = None) -> str:
        mib = ""
        if threads and self.stack_before is not None and self.stack_after is not None:
            mib = " lmem %.0f->%.0f MiB" % (
                self.stack_before * threads / (1 << 20),
                self.stack_after * threads / (1 << 20),
            )
        return (
            f"H101 QSA-ROWS-PREWARM {self.status} forms=[{', '.join(self.forms)}] "
            f"stack {self.stack_before}->{self.stack_after} B{mib} "
            f"census_max={self.census_max_bytes}({self.census_kernel or '-'}) ms={self.ms:.0f}"
            + (f" errors=[{'; '.join(self.errors)}]" if self.errors else "")
        )


def prewarm_rows_forms(
    *,
    signatures: Sequence[dict],
    forms: Sequence[Form],
    launch: Callable,
    stack_bytes: Callable[[], Optional[int]],
    census: Callable[[], Tuple[int, str]],
    make_inputs: Callable,
) -> PrewarmResult:
    """Launch every (signature x form x USE_COUNTS) once. Pure orchestration,
    testable with fakes: ``launch(q, k_pool, v_pool, rows, scale, row_counts)``
    is sparse_attn_rows_triton, ``make_inputs(sig, total_q, use_counts)``
    builds (q, rows, counts)."""
    t0 = time.perf_counter()
    if not signatures:
        return PrewarmResult(status="skipped: no rows launch recorded on this rank "
                                    "(no QSA attention here, or no capture ran)")
    before = stack_bytes()
    done: List[str] = []
    errors: List[str] = []
    for i, sig in enumerate(signatures):
        pre = f"s{i}:" if len(signatures) > 1 else ""
        for total_q, (block_n, warps, stages) in forms:
            for use_counts in (False, True):
                tag = f"{pre}{block_n}/{warps}/{stages}{'+counts' if use_counts else ''}@{total_q}"
                try:
                    q, rows, counts = make_inputs(sig, int(total_q), use_counts)
                    launch(q, sig["k_pool"], sig["v_pool"], rows, 1.0, counts)
                    done.append(tag)
                except Exception as exc:  # noqa: BLE001 -- the lazy path stays
                    errors.append(f"{tag}: {type(exc).__name__}: {str(exc)[:120]}")
    after = stack_bytes()
    c_bytes, c_kernel = census()
    return PrewarmResult(
        status="ok" if not errors else "partial",
        forms=tuple(done),
        stack_before=before,
        stack_after=after,
        census_max_bytes=int(c_bytes),
        census_kernel=c_kernel,
        ms=(time.perf_counter() - t0) * 1000.0,
        errors=tuple(errors),
    )


def _make_inputs(sig: dict, total_q: int, use_counts: bool):
    import torch

    device = sig["k_pool"].device
    q = torch.zeros((total_q, sig["heads"], sig["head_dim"]), dtype=sig["dtype"], device=device)
    rows = torch.full((total_q, sig["k"]), -1, dtype=torch.int32, device=device)
    counts = torch.zeros((total_q,), dtype=torch.int32, device=device) if use_counts else None
    return q, rows, counts


def run_boot_prewarm() -> Optional[PrewarmResult]:
    """The scheduler's delegate: prewarm on a rank that attends (not a Form-A
    worker), log one line, synchronize so every load and pre-grow is done
    before the caller's barrier. Never raises."""
    try:
        from sglang.srt.rank_role import this_rank_is_form_a_worker

        if this_rank_is_form_a_worker():
            logger.info("H101 QSA-ROWS-PREWARM skipped: Form-A worker (attends nothing)")
            return None
        import torch

        from sglang.srt.layers.attention.qsa import sparse_attn as sa
        from sglang.srt.utils.lmem_census import census_max
        from sglang.srt.weg2.sleep_lmem import CudaDriverStackLimit

        drv = CudaDriverStackLimit()

        def _stack() -> Optional[int]:
            try:
                return drv.get_stack_bytes()
            except Exception:  # noqa: BLE001
                return None

        def _launch(q, k_pool, v_pool, rows, scale, counts):
            sa.sparse_attn_rows_triton(q, k_pool, v_pool, rows, scale, row_counts=counts)

        try:
            res = prewarm_rows_forms(
                signatures=sa.rows_prewarm_signatures(),
                forms=sa.rows_launch_forms(),
                launch=_launch,
                stack_bytes=_stack,
                census=census_max,
                make_inputs=_make_inputs,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        finally:
            sa.close_rows_prewarm_recording()
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        threads = int(props.multi_processor_count) * int(props.max_threads_per_multi_processor)
        (logger.warning if res.errors else logger.info)("%s", res.line(threads))
        return res
    except Exception as exc:  # noqa: BLE001 -- a prewarm never kills a boot
        logger.warning("H101 QSA-ROWS-PREWARM failed (%s: %s); the forms load lazily",
                       type(exc).__name__, str(exc)[:200])
        return None
