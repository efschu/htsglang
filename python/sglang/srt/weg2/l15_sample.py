"""L15-12c-E1C: the sample-check LOADER for the D wake
(plan L15-12-PART3-PLAN sec 4).

The wake compares a few sampled held/refilled DEVICE rows with their L2
source byte for byte (l15_wake_check.sample_rows_equal) and folds bad>0
into the group-wide F11 refusal. This module only LOADS the L2 source of
the sampled rows into a SCRATCH device pool -- never the live rows -- so
a bad load can only break the check, not the held tree:

* :func:`sample_plan` picks the sampled rows of a rid-tagged refill plan
  (l15_restore.sample_rows, deterministic, k even) and drops the rows
  without an L2 source (l2_slot < 0 -- the caller counts those missing);
* :func:`load_into_scratch` is the same ONE _load_pages_all_layers call
  as l15_refill.refill, but the device indices are scratch rows
  0..n-1 in sampled order;
* :func:`read_rows` reads the rows back as flat tensors (pure indexing).

P > 1 is refused exactly as refill refuses it: C2 records the arena
PAGE slot (l15_bind) with no lane, so the token's position inside its
page is unknown.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from sglang.srt.weg2 import l15_restore

#: one plan row: (rid, compact_row, l2_slot, l2_gen) -- the
#: l15_refill.refill / gen_checked shape.
PlanRow = Tuple[str, int, int, int]


class L15SampleError(RuntimeError):
    """A sample-check load that cannot succeed: the caller turns it into
    the check's bad/missing counts (the group then refuses via F11,
    "mismatch = stop"). Never swallowed, never a partial success."""


def sample_plan(plan: Sequence[PlanRow], k: int = 64) -> List[PlanRow]:
    """The sampled rows of ``plan`` as rid-tagged 4-tuples, in plan order.

    The subset is l15_restore.sample_rows over the compact rows
    (deterministic, evenly spaced, stable across ranks and runs). Rows
    whose l2_slot < 0 have no L2 source and are EXCLUDED here: the caller
    counts them missing (they can never be byte-compared)."""
    chosen = set(l15_restore.sample_rows([int(t[1]) for t in plan], k))
    out: List[PlanRow] = []
    for t in plan:
        if int(t[1]) in chosen and int(t[2]) >= 0:
            out.append((str(t[0]), int(t[1]), int(t[2]), int(t[3])))
    return out


def load_into_scratch(
    sampled: Sequence[PlanRow],
    host_pool,
    scratch_pool,
    page_tokens: int,
) -> List[int]:
    """Load the L2 source of ``sampled`` into scratch rows 0..n-1 of
    ``scratch_pool`` with ONE _load_pages_all_layers call (the l15_refill
    machinery); return the scratch row ids in sampled order.

    ``scratch_pool`` is the only load target -- the live rows are never
    an index here, so a wrong source can only make the check fail. Any
    load error raises L15SampleError (all-or-nothing, like refill)."""
    if page_tokens != 1:
        raise L15SampleError(
            "P>1 not supported yet: l2_slot is the arena page slot (C2); "
            "the lane is not recorded"
        )
    for t in sampled:
        if int(t[2]) < 0:
            raise L15SampleError(
                "sample: rid %s carries l2_slot -1 (sample_plan must drop it)"
                % (t[0],)
            )
    n = len(sampled)
    if n == 0:
        return []
    slots_t = torch.tensor([int(t[2]) for t in sampled], dtype=torch.int64)
    didx_t = torch.arange(n, dtype=torch.int64)
    try:
        host_pool._load_pages_all_layers(
            scratch_pool, slots_t, didx_t, lanes=None, mode=None
        )
    except Exception as exc:  # noqa: BLE001 -- fold into the check's bad count
        raise L15SampleError("sample load failed: %r" % (exc,)) from exc
    return list(range(n))


def read_rows(pool, rows: Sequence[int]) -> List[torch.Tensor]:
    """The K/V bytes of ``rows``, one flat 1-D tensor per row in order:
    all layers' k row first, then all layers' v row. Pure tensor
    indexing -- a hybrid pool is unwrapped via ``.full_kv_pool`` like
    _l15_flush_zero_kv_bounded; the caller hands the two lists (device,
    L2 source) to l15_wake_check.sample_rows_equal."""
    p = getattr(pool, "full_kv_pool", pool)
    out: List[torch.Tensor] = []
    for r in rows:
        r = int(r)
        parts = [b[r].reshape(-1) for b in p.k_buffer]
        parts += [b[r].reshape(-1) for b in p.v_buffer]
        out.append(torch.cat(parts))
    return out
