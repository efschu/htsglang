# SPDX-License-Identifier: Apache-2.0
# Ported 19.09.2026 from vLLM PR #56177 (01554/vllm, lab/upstream-pool-standalone-
# rebased, expert_pool/tables.py + copy.py, Apache-2.0) into this fork's
# per-layer MoE expert offload. Adapted, not copied: one bank per layer (the
# layer's existing [R+C] slot arena), a FIXED resident region [0, R) that is
# never a victim (no host copy of the resident experts is needed), a device-LRU
# region [R, R+C-S) and S staging rows; the host source is the layer's pinned
# SPILL pool addressed through a static ``host_row`` table.
"""Device-planned expert pool: the decode step and the row copies run entirely
on the device with fixed shapes and addresses, so a decode CUDA graph captures
them and every replay plans afresh -- no host readback between the router and
the GEMM (the breakable route paid 48 of them per token).

``step_reference`` (torch, synchronizing) defines the semantics; the desk
tests run it and the Triton program must match it exactly.

SPECULATIVE PREFETCH (``prefetch=True``, SGLANG_MOE_POOL_PREFETCH=1)
--------------------------------------------------------------------
Mixtral-Offloading (Eliseev/Mazur 2023) observed that the residual stream
changes slowly across layers, so layer L+1's router applied to layer L's MoE
input predicts L+1's experts well. We use that to hide the row copies: while
the main stream computes layer L's experts, a side stream runs THIS SAME
program over layer L+1's tables with the predicted ids and copies the rows it
is missing. Layer L+1 then waits on one event and plans for real -- a correct
prediction is now a plain hit, a wrong one is a synchronous miss exactly as
today.

The prefetch runs on the NEXT layer's tables, never on the running layer's, so
it can by construction not evict a row layer L is computing from: the pools
are per layer and disjoint. Inside the target layer the two accesses are kept
apart by ONE rule, which is the same expression the real step already uses --
``row_use >= clock`` is never a victim:

* the prefetch does NOT advance the clock, so the rows the target layer's LAST
  real step stamped (``row_use == clock``) are excluded from prefetch eviction.
  Those are the previous round's working set, i.e. exactly the rows a good
  prediction wants to keep;
* a row the prefetch just filled is stamped ``row_use = clock`` too, so two
  predicted misses in one prefetch cannot evict each other;
* the real step then advances the clock to ``clock + 1``, which un-protects the
  prefetched rows: a MISPREDICTED row is evictable again immediately and never
  outranks a row the real step used.

Prefetched rows go into the ordinary LRU region, not a separate prefetch
region. Numbers (rank 0 of the Qwen3.8-Flash-Next TP=3 target form, 19.09.):
LRU = 126 rows, staging = 30, top-k = 10 and 3-4 misses per layer per verify.
A reserved prefetch region would have to be >= the predicted-miss count (~10
rows) and would be dead capacity in every round the prediction is right --
while taken out of the very LRU pool whose depth is the measured recency lever
(fn5t: 18 more LRU rows took rank 0 from 5.4 to 3.3 misses per verify forward).
In the LRU region a correct prediction IS the cache line the next round wants,
and a wrong one costs a single LRU slot out of 126 for one round. Staging is
never used by the prefetch: staging rows are per-step scratch that the target
layer's own real step overwrites in the same round.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROW_USE_NEVER = 0x7FFFFFFFFFFFFFFF
PLAN_WIDTH = 64
COPY_PROGRAMS = 32
COPY_WORDS = 4096  # int32 words per program iteration (16 KiB)


@dataclass
class PoolTables:
    """One layer's device state; every tensor has a fixed address."""

    num_experts: int
    pool_rows: int  # rows [0, pool_rows) = resident + LRU; staging follows
    lru_start: int  # rows [0, lru_start) are fixed residents, never victims
    hot_phys: Any  # [E] int32 expert -> bank row / -1
    host_row: Any  # [E] int32 expert -> pinned spill row / -1 (static)
    row_key: Any  # [pool_rows + staging] int32 row -> expert / -1
    row_use: Any  # [pool_rows + staging] int64 clock of last selection
    clock: Any  # [1] int64
    gate: Any  # [1] int32 promotions allowed
    error: Any  # [1] int32 sticky device error
    promote_limit: Any  # [1] int32 (0 = unlimited)
    promote_interval: Any  # [1] int32
    forwards: Any  # [1] int32
    promote_min_misses: Any  # [1] int32
    protect_recent: Any  # [1] int32
    miss_count: Any  # [E] int32
    staging_rows: Any  # [S] int32
    misses_total: Any  # [1] int64 misses (promoted + staged) since the last report
    pf_row: Any  # [pool_rows + staging] int64 clock at which a prefetch filled the row
    pf_counts: Any  # [4] int64 predicted, fetched, hits, skipped (no victim)


@dataclass
class StepBuffers:
    gather_src: Any  # [W] int32 host rows
    gather_dst: Any  # [W] int32 bank rows
    gather_count: Any  # [1] int32
    routes: Any  # [W] int32 physical row per ids lane, -1 padding
    staged_expert: Any  # [W] int32
    staged_row: Any  # [W] int32
    staged_count: Any  # [1] int32
    promoted_count: Any  # [1] int32
    step_map: Any  # [E] int32


def allocate_pool_tables(
    device, num_experts: int, rows: int, lru_start: int, staging: int,
    hot_slot_of: Dict[int, int], host_row: Sequence[int],
) -> PoolTables:
    """``rows`` = R + C of the layer's arena; the last ``staging`` of them are
    staging rows, the ones in [lru_start, rows - staging) the LRU region.
    ``hot_slot_of`` = expert -> row for every initially resident expert."""
    import torch

    if staging < 1 or not 0 <= lru_start < rows - staging:
        raise ValueError("pool needs staging rows and a non-empty LRU region")
    if len(host_row) != num_experts:
        raise ValueError("host_row must have one entry per expert")
    pool_rows = rows - staging
    hot = torch.full((num_experts,), -1, dtype=torch.int32)
    key = torch.full((rows,), -1, dtype=torch.int32)
    use = torch.zeros(rows, dtype=torch.int64)
    for e, r in hot_slot_of.items():
        if not 0 <= r < pool_rows:
            raise ValueError(f"expert {e} placed outside the pool at row {r}")
        hot[e] = r
        key[r] = e
    use[:lru_start] = ROW_USE_NEVER
    use[pool_rows:] = ROW_USE_NEVER
    hr = torch.tensor(list(host_row), dtype=torch.int32)
    bad = [(e, r) for e, r in hot_slot_of.items() if int(hr[e]) >= 0]
    if bad:
        raise ValueError(f"resident experts must carry host_row -1: {bad[:3]}")

    def one(v, dtype=torch.int32):
        return torch.full((1,), v, dtype=dtype, device=device)

    return PoolTables(
        num_experts=num_experts, pool_rows=pool_rows, lru_start=lru_start,
        hot_phys=hot.to(device), host_row=hr.to(device), row_key=key.to(device),
        row_use=use.to(device), clock=one(0, torch.int64), gate=one(1), error=one(0),
        promote_limit=one(0), promote_interval=one(1), forwards=one(0),
        promote_min_misses=one(1), protect_recent=one(0),
        miss_count=torch.zeros(num_experts, dtype=torch.int32, device=device),
        staging_rows=torch.arange(pool_rows, rows, dtype=torch.int32, device=device),
        misses_total=one(0, torch.int64),
        pf_row=torch.full((rows,), -1, dtype=torch.int64, device=device),
        pf_counts=torch.zeros(4, dtype=torch.int64, device=device),
    )


def allocate_step_buffers(device, num_experts: int, width: int = PLAN_WIDTH) -> StepBuffers:
    import torch

    def ints(n):
        return torch.zeros(n, dtype=torch.int32, device=device)

    return StepBuffers(
        gather_src=ints(width), gather_dst=ints(width), gather_count=ints(1),
        routes=torch.full((width,), -1, dtype=torch.int32, device=device),
        staged_expert=ints(width), staged_row=ints(width), staged_count=ints(1),
        promoted_count=ints(1),
        step_map=torch.full((num_experts,), -1, dtype=torch.int32, device=device),
    )


def sync_tables(tables: PoolTables, lru_holds: Dict[int, int]) -> None:
    """After an EAGER forward rewrote the LRU rows (run_waves' fetches): the
    device tables take the host's truth. ``lru_holds`` = row -> expert for the
    rows the eager pass wrote; every other LRU row becomes free (-1), staging
    rows are never owned. Residents [0, lru_start) are untouched."""
    import torch

    E, lo, hi = tables.num_experts, tables.lru_start, tables.pool_rows
    hot = tables.hot_phys.cpu()
    key = tables.row_key.cpu()
    use = tables.row_use.cpu()
    clock = int(tables.clock[0])
    for r in range(lo, key.shape[0]):
        old = int(key[r])
        if old >= 0:
            hot[old] = -1
        key[r] = -1
    for r, e in lru_holds.items():
        if lo <= r < hi and 0 <= e < E and int(tables.host_row[e]) >= 0:
            hot[e] = r
            key[r] = e
            use[r] = clock
    dev = tables.hot_phys.device
    tables.hot_phys.copy_(hot.to(dev))
    tables.row_key.copy_(key.to(dev))
    tables.row_use.copy_(use.to(dev))
    # The eager pass rewrote these rows behind the prefetch's back: every
    # prefetch mark on them is stale and would count a later hit as a
    # prefetch hit it never was.
    tables.pf_row[lo:].fill_(-1)


def take_report(tables: PoolTables) -> Tuple[int, int]:
    """(forwards, misses) since the last report; both reset. One host read."""
    f, m = int(tables.forwards[0]), int(tables.misses_total[0])
    tables.forwards.fill_(0)
    tables.misses_total.fill_(0)
    return f, m


def take_prefetch_report(tables: PoolTables) -> Tuple[int, int, int, int]:
    """(predicted, fetched, hits, skipped) since the last report; all reset.

    ``predicted`` = distinct valid ids the next layer's router proposed,
    ``fetched`` = rows the prefetch actually copied in, ``hits`` = fetched rows
    the target layer's next real step genuinely needed, ``skipped`` = predicted
    misses that found no evictable LRU row. ``fetched - hits`` is the wasted
    prefetch. One host read."""
    v = [int(x) for x in tables.pf_counts.tolist()]
    tables.pf_counts.zero_()
    return v[0], v[1], v[2], v[3]


def step_reference(
    tables: PoolTables, ids, buffers: StepBuffers, prefetch: bool = False
) -> Tuple[List[Tuple[int, int]], Any]:
    """Plan and flip one step on the host (torch, synchronizing).

    Returns (gathers, step_map): gathers = (host row, bank row) in copy order,
    promotions first, then staged misses. Distinct valid ids in first
    occurrence order; hits in the LRU region are stamped with the clock; each
    miss takes the LRU row with the smallest (row_use, row) among rows not
    used this step, a FREE row (key -1, use -1) before any occupied one;
    without a victim it is staged for this step only. Residents never move.

    ``prefetch=True`` is the speculative pass described in the module
    docstring: same victim rule and same promotion writes, but it does not
    advance the clock, does not stamp hits (a prediction is not a use), does
    not count misses, never stages (a predicted miss with no victim is simply
    dropped -- the real step will stage it), and writes neither ``routes`` nor
    ``step_map``. It marks every row it fills in ``pf_row`` with the current
    clock; the next real step, which advances the clock by one, counts a hit on
    a row whose ``pf_row`` is ``clock - 1`` as a prefetch hit."""
    import torch

    E = tables.num_experts
    hot = tables.hot_phys.tolist()
    host_row = tables.host_row.tolist()
    row_key = tables.row_key.tolist()
    row_use = tables.row_use.tolist()
    staging_rows = tables.staging_rows.tolist()
    gate = bool(int(tables.gate[0]))
    raw = [int(v) for v in ids.reshape(-1).tolist()]
    if len(raw) > buffers.gather_src.shape[0] or len(raw) > len(staging_rows):
        raise ValueError("Step ids exceed the plan width or the staging rows")
    error = bool(int(tables.error[0]))
    selected: List[int] = []
    for v in raw:
        if v == -1:
            continue
        if not 0 <= v < E:
            error = True
            continue
        if v not in selected:
            selected.append(v)
    clock = int(tables.clock[0])
    forwards = int(tables.forwards[0])
    pf_row = tables.pf_row.tolist()
    pf_counts = [int(v) for v in tables.pf_counts.tolist()]
    if gate and not prefetch:
        clock += 1
        forwards += 1
        for e in selected:
            r = hot[e]
            if r >= tables.lru_start:
                if pf_row[r] == clock - 1:
                    pf_counts[2] += 1
                row_use[r] = clock
    if prefetch:
        pf_counts[0] += len(selected)
    limit = int(tables.promote_limit[0])
    interval = int(tables.promote_interval[0])
    min_misses = int(tables.promote_min_misses[0])
    protect = int(tables.protect_recent[0])
    miss_count = tables.miss_count.tolist()
    # The prefetch is not throttled by the promote interval / min-miss ramp:
    # those damp thrash between real forwards, and a speculative pass that
    # waited for them would never fetch anything.
    promote_ok = gate if prefetch else (gate and (forwards - 1) % interval == 0)
    gathers: List[Tuple[int, int]] = []
    staged: List[Tuple[int, int]] = []
    for e in selected:
        if hot[e] >= 0:
            continue
        if gate and not prefetch:
            miss_count[e] += 1
        victim = -1
        if promote_ok and (limit == 0 or len(gathers) < limit) and (
            prefetch or miss_count[e] >= min_misses
        ):
            best = None
            for r in range(tables.lru_start, tables.pool_rows):
                if row_key[r] < 0:
                    cand = (-1, r)
                elif row_use[r] < clock and (protect == 0 or row_use[r] < clock - protect):
                    cand = (row_use[r], r)
                else:
                    continue
                if best is None or cand < best:
                    best = cand
            if best is not None:
                victim = best[1]
        if victim < 0:
            if prefetch:
                # No evictable row: drop the prediction. The real step will
                # take this expert as an ordinary miss and stage it.
                pf_counts[3] += 1
                continue
            staged.append((e, staging_rows[len(staged)]))
            continue
        old = row_key[victim]
        if old >= 0:
            hot[old] = -1
        hot[e] = victim
        row_key[victim] = e
        row_use[victim] = clock
        miss_count[e] = 0
        if prefetch:
            pf_row[victim] = clock
            pf_counts[1] += 1
        gathers.append((host_row[e], victim))
    step_map = list(hot)
    for e, row in staged:
        step_map[e] = row
    routes = [-1] * buffers.routes.shape[0]
    for i, v in enumerate(raw):
        if 0 <= v < E:
            routes[i] = step_map[v]
    dev = tables.hot_phys.device

    def write(t, values, dtype):
        t.copy_(torch.tensor(values, dtype=dtype, device=dev))

    write(tables.hot_phys, hot, torch.int32)
    write(tables.row_key, row_key, torch.int32)
    write(tables.row_use, row_use, torch.int64)
    tables.clock.fill_(clock)
    tables.forwards.fill_(forwards)
    write(tables.miss_count, miss_count, torch.int32)
    write(tables.pf_row, pf_row, torch.int64)
    write(tables.pf_counts, pf_counts, torch.int64)
    tables.error.fill_(1 if error else 0)
    pairs = gathers + [(host_row[e], row) for e, row in staged]
    buffers.gather_count.fill_(len(pairs))
    if not prefetch:
        tables.misses_total.add_(len(pairs))
    buffers.promoted_count.fill_(len(gathers))
    buffers.staged_count.fill_(len(staged))
    for i, (s, d) in enumerate(pairs):
        buffers.gather_src[i], buffers.gather_dst[i] = s, d
    for i, (e, row) in enumerate(staged):
        buffers.staged_expert[i], buffers.staged_row[i] = e, row
    if prefetch:
        # The speculative pass routes nothing: it has no compute of its own,
        # and the target layer's real step publishes step_map/routes.
        return pairs, buffers.step_map
    write(buffers.step_map, step_map, torch.int32)
    write(buffers.routes, routes, torch.int32)
    return pairs, buffers.step_map


def step(tables: PoolTables, ids, buffers: StepBuffers, prefetch: bool = False) -> None:
    """Plan and flip one step: Triton on CUDA, the reference elsewhere.

    ``prefetch=True`` runs the speculative pass (module docstring); it needs
    its OWN ``buffers``, because its gather list must survive on the side
    stream until the copy is done while the target layer's real step writes
    the layer's normal buffers."""
    if tables.hot_phys.device.type != "cuda":
        step_reference(tables, ids, buffers, prefetch=prefetch)
        return
    flat = ids.reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    width = buffers.gather_src.shape[0]
    if flat.numel() > width or flat.numel() > tables.staging_rows.shape[0]:
        raise ValueError("Step ids exceed the plan width or the staging rows")
    _step_kernel()[(1,)](
        flat, flat.numel(),
        tables.hot_phys, tables.host_row, tables.row_key, tables.row_use,
        tables.clock, tables.gate, tables.error, tables.promote_limit,
        tables.promote_interval, tables.forwards, tables.promote_min_misses,
        tables.protect_recent, tables.miss_count, tables.staging_rows,
        tables.misses_total, tables.pf_row, tables.pf_counts,
        buffers.gather_src, buffers.gather_dst, buffers.gather_count,
        buffers.routes, buffers.staged_expert, buffers.staged_row,
        buffers.staged_count, buffers.promoted_count, buffers.step_map,
        tables.num_experts, tables.pool_rows, tables.lru_start,
        WIDTH=width, BLOCK_R=_next_power_of_two(tables.pool_rows),
        MAP_BLOCK=1024, PREFETCH=bool(prefetch), num_warps=8,
    )


def _next_power_of_two(value: int) -> int:
    return 1 << max(int(value) - 1, 0).bit_length()


_KERNELS: Dict[str, Any] = {}


def _step_kernel():
    if "step" in _KERNELS:
        return _KERNELS["step"]
    import triton
    import triton.language as tl

    @triton.jit
    def pool_step(
        ids_ptr, n,
        hot_phys_ptr, host_row_ptr, row_key_ptr, row_use_ptr,
        clock_ptr, gate_ptr, error_ptr, limit_ptr, interval_ptr, forwards_ptr,
        min_misses_ptr, protect_ptr, miss_count_ptr, staging_ptr, misses_total_ptr,
        pf_row_ptr, pf_counts_ptr,
        gather_src_ptr, gather_dst_ptr, gather_count_ptr, routes_ptr,
        staged_expert_ptr, staged_row_ptr, staged_count_ptr, promoted_count_ptr,
        step_map_ptr, num_experts, pool_rows, lru_start,
        WIDTH: tl.constexpr, BLOCK_R: tl.constexpr, MAP_BLOCK: tl.constexpr,
        PREFETCH: tl.constexpr,
    ):
        never = 0x7FFFFFFFFFFFFFFF
        lane = tl.arange(0, WIDTH)
        present = lane < n
        raw = tl.load(ids_ptr + lane, mask=present, other=-1).to(tl.int64)
        valid = present & (raw >= 0) & (raw < num_experts)
        bad = present & (raw != -1) & (~valid)
        if tl.sum(bad.to(tl.int32), 0) > 0:
            tl.store(error_ptr, 1)
        safe = tl.where(valid, raw, 0)
        same = safe[:, None] == safe[None, :]
        earlier = lane[None, :] < lane[:, None]
        duplicate = tl.sum((same & earlier & valid[None, :]).to(tl.int32), 1) > 0
        distinct = valid & (duplicate == 0)
        resident = tl.load(hot_phys_ptr + safe, mask=distinct, other=-1).to(tl.int64)
        hit = distinct & (resident >= 0)
        gate = tl.load(gate_ptr) != 0
        clock = tl.load(clock_ptr)
        forwards = tl.load(forwards_ptr)
        if PREFETCH:
            # The speculative pass leaves the clock where it is: that is what
            # protects the target layer's last real working set from being
            # evicted by a prediction, and what un-protects a mispredicted row
            # as soon as the real step advances the clock.
            tl.store(
                pf_counts_ptr,
                tl.load(pf_counts_ptr) + tl.sum(distinct.to(tl.int32), 0).to(tl.int64),
            )
        else:
            if gate:
                clock = clock + 1
                tl.store(clock_ptr, clock)
                forwards = forwards + 1
                tl.store(forwards_ptr, forwards)
                stamp = hit & (resident >= lru_start)
                marked = tl.load(
                    pf_row_ptr + tl.where(stamp, resident, 0), mask=stamp, other=-1
                )
                pf_hits = tl.sum((stamp & (marked == clock - 1)).to(tl.int32), 0)
                tl.store(
                    pf_counts_ptr + 2,
                    tl.load(pf_counts_ptr + 2) + pf_hits.to(tl.int64),
                )
                tl.store(row_use_ptr + tl.where(stamp, resident, 0), clock, mask=stamp)
        limit = tl.load(limit_ptr)
        interval = tl.load(interval_ptr)
        min_misses = tl.load(min_misses_ptr)
        protect = tl.load(protect_ptr).to(tl.int64)
        if PREFETCH:
            promote_ok = gate
        else:
            promote_ok = gate & (((forwards - 1) % interval) == 0)
        tl.debug_barrier()
        offs_r = tl.arange(0, BLOCK_R)
        in_lru = (offs_r >= lru_start) & (offs_r < pool_rows)
        misses = tl.sum((distinct & (~hit)).to(tl.int32), 0)
        scan = promote_ok & (misses > 0)
        use = tl.full((BLOCK_R,), never, tl.int64)
        if scan:
            use = tl.load(row_use_ptr + offs_r, mask=in_lru, other=never)
            key = tl.load(row_key_ptr + offs_r, mask=in_lru, other=0)
            occupied = in_lru & (key >= 0)
            # a used-this-step row (stamped with clock) is never a victim
            use = tl.where(occupied & (use >= clock), never, use)
            if protect > 0:
                use = tl.where(occupied & (use >= clock - protect), never, use)
            # a free row goes first
            use = tl.where(in_lru & (key < 0), -1, use)
        promoted = 0
        staged = 0
        for i in range(0, WIDTH):
            is_miss = tl.sum(tl.where(lane == i, (distinct & (~hit)).to(tl.int32), 0), 0)
            if is_miss > 0:
                expert = tl.load(ids_ptr + i).to(tl.int64)
                victim_row = tl.full((), -1, tl.int64)
                misses_so_far = tl.load(miss_count_ptr + expert)
                if not PREFETCH:
                    if gate:
                        misses_so_far = misses_so_far + 1
                        tl.store(miss_count_ptr + expert, misses_so_far)
                room = promote_ok & ((limit == 0) | (promoted < limit))
                if PREFETCH:
                    may_promote = room
                else:
                    may_promote = room & (misses_so_far >= min_misses)
                if may_promote:
                    best = tl.min(use, 0)
                    if best != never:
                        victim_row = tl.min(tl.where(use == best, offs_r.to(tl.int64), never), 0)
                src = tl.load(host_row_ptr + expert)
                if victim_row >= 0:
                    old = tl.load(row_key_ptr + victim_row).to(tl.int64)
                    if old >= 0:
                        tl.store(hot_phys_ptr + old, -1)
                    tl.store(hot_phys_ptr + expert, victim_row.to(tl.int32))
                    tl.store(row_key_ptr + victim_row, expert.to(tl.int32))
                    tl.store(row_use_ptr + victim_row, clock)
                    tl.store(miss_count_ptr + expert, 0)
                    if PREFETCH:
                        tl.store(pf_row_ptr + victim_row, clock)
                    tl.store(gather_src_ptr + promoted, src)
                    tl.store(gather_dst_ptr + promoted, victim_row.to(tl.int32))
                    use = tl.where(offs_r.to(tl.int64) == victim_row, never, use)
                    promoted += 1
                else:
                    if PREFETCH:
                        # no evictable row: drop it, the real step will stage it
                        tl.store(pf_counts_ptr + 3, tl.load(pf_counts_ptr + 3) + 1)
                    else:
                        tl.store(staged_expert_ptr + staged, expert.to(tl.int32))
                        tl.store(staged_row_ptr + staged, tl.load(staging_ptr + staged))
                        staged += 1
            tl.debug_barrier()
        tl.store(promoted_count_ptr, promoted)
        tl.store(staged_count_ptr, staged)
        tl.store(gather_count_ptr, promoted + staged)
        if PREFETCH:
            # The speculative pass has no compute to route and no miss to
            # report: it leaves step_map/routes/misses_total to the real step.
            tl.store(pf_counts_ptr + 1, tl.load(pf_counts_ptr + 1) + promoted.to(tl.int64))
        else:
            tl.store(misses_total_ptr, tl.load(misses_total_ptr) + (promoted + staged).to(tl.int64))
            tl.debug_barrier()
            for i in range(0, staged):
                e = tl.load(staged_expert_ptr + i)
                tl.store(gather_src_ptr + promoted + i, tl.load(host_row_ptr + e))
                tl.store(gather_dst_ptr + promoted + i, tl.load(staged_row_ptr + i))
            for start in range(0, num_experts, MAP_BLOCK):
                offs = start + tl.arange(0, MAP_BLOCK)
                in_range = offs < num_experts
                rows = tl.load(hot_phys_ptr + offs, mask=in_range, other=-1)
                tl.store(step_map_ptr + offs, rows, mask=in_range)
            tl.debug_barrier()
            for i in range(0, staged):
                e = tl.load(staged_expert_ptr + i)
                tl.store(step_map_ptr + e, tl.load(staged_row_ptr + i))
            tl.debug_barrier()
            route = tl.load(step_map_ptr + safe, mask=valid, other=-1)
            tl.store(routes_ptr + lane, tl.where(valid, route, -1), mask=lane < WIDTH)

    _KERNELS["step"] = pool_step
    return pool_step


# ---- row copies ------------------------------------------------------------
def copy_rows_reference(sources, destinations, pairs) -> None:
    for s, d in pairs:
        for src, dst in zip(sources, destinations):
            dst[d].copy_(src[s])


def _word_rows(tensor):
    import torch

    if not tensor.is_contiguous():
        raise ValueError("pool copies require contiguous rows")
    return tensor.view(torch.uint8).reshape(tensor.shape[0], -1).view(torch.int32)


def copy_rows(sources, destinations, src_rows, dst_rows, count) -> None:
    """Copy ``count`` (src, dst) row pairs of every tensor; ``count`` is a
    device scalar read by a fixed small grid (one launch per tensor), so an
    empty step costs a few programs and nothing synchronizes."""
    if destinations[0].device.type != "cuda":
        n = int(count.reshape(-1)[0].item())
        pairs = [(int(src_rows[i]), int(dst_rows[i])) for i in range(n)]
        copy_rows_reference(sources, destinations, pairs)
        return
    kern = _copy_kernel()
    for src, dst in zip(sources, destinations):
        s, d = _word_rows(src), _word_rows(dst)
        if s.shape[1] != d.shape[1]:
            raise ValueError("pool copy: row size differs between source and bank")
        kern[(COPY_PROGRAMS,)](
            s, d, src_rows, dst_rows, count, d.shape[1], s.stride(0), d.stride(0),
            PROGRAMS=COPY_PROGRAMS, BLOCK=COPY_WORDS, num_warps=4,
        )


def _copy_kernel():
    if "copy" in _KERNELS:
        return _KERNELS["copy"]
    import triton
    import triton.language as tl

    @triton.jit
    def pool_copy(src, dst, src_rows_ptr, dst_rows_ptr, count_ptr, words, sstride, dstride,
                  PROGRAMS: tl.constexpr, BLOCK: tl.constexpr):
        stripe = tl.program_id(0)
        count = tl.load(count_ptr)
        for lane in range(0, count):
            src_row = tl.load(src_rows_ptr + lane).to(tl.int64)
            dst_row = tl.load(dst_rows_ptr + lane).to(tl.int64)
            src_base = src + src_row * sstride
            dst_base = dst + dst_row * dstride
            for start in range(stripe * BLOCK, words, PROGRAMS * BLOCK):
                offsets = start + tl.arange(0, BLOCK)
                mask = offsets < words
                tl.store(dst_base + offsets, tl.load(src_base + offsets, mask=mask), mask=mask)

    _KERNELS["copy"] = pool_copy
    return pool_copy


__all__ = [
    "PLAN_WIDTH", "ROW_USE_NEVER", "PoolTables", "StepBuffers",
    "allocate_pool_tables", "allocate_step_buffers", "copy_rows",
    "copy_rows_reference", "step", "step_reference", "sync_tables", "take_report",
    "take_prefetch_report",
]
