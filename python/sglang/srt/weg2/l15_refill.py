"""L15-12c-D: TP0 (cap 0) refills its held rows from L2 at the wake.

Pure module: no scheduler, no CUDA, no arena imports. It turns the rank's
manifest-derived refill plan into ONE generation-checked, page-grouped
bulk H2D through ArenaMHAHostPool._load_pages_all_layers. The wiring into
the wake (weight_updater) is a later step.

Plan entries are 4-tuples ``(rid, compact_row, l2_slot, l2_gen)`` as
produced from the manifest (l15_restore.refill_plan yields
(compact_row, l2_slot, l2_gen); the wiring tags each row with the span's
rid -- the rid is what makes a generation mismatch drop a WHOLE request,
never a partial one).

OPEN (plan L15-12-PART3-PLAN section 8): the manifest (l15_manifest.
HoldSpan) carries no L2 identity for the GDN anchor -- ``anchor_slot`` is
the L1 slot the span is pinned to, and there is no anchor L2 slot/generation
field. Anchors are therefore NOT refilled here; inventing a manifest field
is not this module's call. When the identity exists, the GDN share goes
through ArenaMambaPoolHost._load_states_all_layers(device_pool, slots,
didx) (arena_mamba_pool.py:562) with the same all-or-nothing discipline.

OPEN (C2 review 01.10.): P>1 needs the lane per token (C2 to record
(row - S) % P) and DCP-owned lanes via the owner-page-prefix loader
(arena_pool.py:1562). Until then refill() refuses page_tokens != 1: C2's
l2_of records the arena PAGE slot (l15_bind.py: (row - S) // P, repeated
per token), never the lane, so a divmod reading of l2_slot would silently
copy whole foreign pages.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch


class L15RefillError(RuntimeError):
    """Refill could not be performed as one whole operation.

    The caller folds this into the wake's gather as a bad vote; a partial
    copy is never reported as success."""


def gen_check(
    plan: Iterable[Tuple[str, int, int, int]], host_pool
) -> Tuple[List[Tuple[str, int, int, int]], List[str]]:
    """F1 generation check BEFORE copying: one host_pool.slot_gens call.

    Returns (ok_rows, drop_rids). A row whose l2_slot is -1 (no L2 copy)
    or whose recorded l2_gen differs from the arena's current generation
    (incl. -1 = not COMPLETE) marks its WHOLE rid as dropped -- a token
    whose L2 row was re-claimed is foreign, and a half-refilled request is
    worse than a missing one. ok_rows keeps the plan's entry order."""
    rows = list(plan)
    # Unique real slots for the ONE census call, in first-appearance order.
    unique: List[int] = []
    seen: Set[int] = set()
    for _rid, _row, slot, _gen in rows:
        if slot >= 0 and slot not in seen:
            seen.add(slot)
            unique.append(slot)
    current = host_pool.slot_gens(unique)
    gens: Dict[int, int] = dict(zip(unique, (int(g) for g in current)))
    bad: Set[str] = set()
    for rid, _row, slot, gen in rows:
        cur = -1 if slot < 0 else gens.get(slot, -1)
        if cur != int(gen):
            bad.add(rid)
    ok = [r for r in rows if r[0] not in bad]
    return ok, sorted(bad)


def refill(
    plan_ok: Sequence[Tuple[str, int, int, int]],
    host_pool,
    device_pool,
    page_tokens: int,
) -> int:
    """Bulk H2D of the checked plan: ONE _load_pages_all_layers call,
    return the number of rows copied.

    Only page_tokens == 1 (the 27B form) is supported: C2's l2_of records
    the arena PAGE slot (l15_bind.py: (row - S) // P) with no lane, so for
    P > 1 the token's position inside its page is unknown and any mapping
    guess could copy foreign pages -- refill refuses with L15RefillError
    BEFORE touching the pool. With P == 1 the slot is the whole L2 address:
    slots = the unique page slots ascending (a slot run helps the "dma"
    mode's consecutive-slot copies), device_indices = the target compact
    rows in slot order -- the loader's own scatter order with lanes=None
    and page size 1. A duplicate slot or a -1 slot raises before any copy;
    any failure of the load itself likewise raises, never with a partial
    count reported as success."""
    if page_tokens != 1:
        raise L15RefillError(
            "P>1 not supported yet: l2_slot is the arena page slot (C2); "
            "the lane is not recorded"
        )
    rows_by_slot: Dict[int, int] = {}
    for rid, row, slot, _gen in plan_ok:
        if slot < 0:
            raise L15RefillError(
                "refill: rid %s carries l2_slot -1 (gen_check must drop it)" % (rid,)
            )
        if int(slot) in rows_by_slot:
            raise L15RefillError(
                "refill: duplicate page slot %d (rows %d, %d)"
                % (int(slot), rows_by_slot[int(slot)], int(row))
            )
        rows_by_slot[int(slot)] = int(row)
    slots_list = sorted(rows_by_slot)
    didx = [rows_by_slot[s] for s in slots_list]
    slots_t = torch.tensor(slots_list, dtype=torch.int64)
    didx_t = torch.tensor(didx, dtype=torch.int64)
    try:
        host_pool._load_pages_all_layers(
            device_pool, slots_t, didx_t, lanes=None, mode=None
        )
    except Exception as exc:  # noqa: BLE001 -- all-or-nothing: fold into a bad vote
        raise L15RefillError("refill: page load failed: %r" % (exc,)) from exc
    return len(didx)
