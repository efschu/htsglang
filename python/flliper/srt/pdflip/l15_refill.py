"""L15-12c-D: TP0 (cap 0) refills its held rows from L2 at the wake.

Pure module: no scheduler, no CUDA, no arena imports. It turns the rank's
manifest-derived refill plan into ONE generation-checked, page-grouped
bulk H2D through ArenaMHAHostPool._load_pages_all_layers. The wiring into
the wake (weight_updater) is a later step.

Plan entries are 4-tuples ``(rids, compact_row, l2_slot, l2_gen)`` (5 with
the lane, P > 1) as produced from the manifest by
l15_restore.rid_tagged_plan: one entry per distinct compact row -- shared
prefix rows are carried ONCE with the tuple of every rid sharing them
(L15-DEDUPE), so the copy loads each L2 row once and a generation
mismatch still drops every sharing request WHOLE, never a partial one.

OPEN (plan L15-12-PART3-PLAN section 8): the manifest (l15_manifest.
HoldSpan) carries no L2 identity for the GDN anchor -- ``anchor_slot`` is
the L1 slot the span is pinned to, and there is no anchor L2 slot/generation
field. Anchors are therefore NOT refilled here; inventing a manifest field
is not this module's call. When the identity exists, the GDN share goes
through ArenaMambaPoolHost._load_states_all_layers(device_pool, slots,
didx) (arena_mamba_pool.py:562) with the same all-or-nothing discipline.

CLOSED by L15-12c-P1 (was OPEN, C2 review 01.10.): P>1 (the NF form) now
has the lane per token -- l15_bind records lane = (row - S) % P next to the
page slot, retain carries it in HoldSpan.l2_lanes, and _refill_pgt loads
one page-grouped call with the common lane list (the loader's global-lanes
contract; per-page subsets are refused named, not guessed).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from flliper.srt.pdflip.l15_manifest import Manifest
from flliper.srt.pdflip.l15_restore import (  # noqa: F401 -- L15RefillError re-exported
    L15RefillError,
    owned_l2_rows,
    refill_plan,
)


def _rid_tag(tag) -> Tuple[str, ...]:
    """The rid set a plan entry carries: a plain str tag (single rid) or
    the tuple of EVERY rid sharing the row (l15_restore.owned_l2_rows).
    gen_check drops whole requests per rid, so a mismatch marks them all."""
    if isinstance(tag, str):
        return (tag,)
    return tuple(str(t) for t in tag)


def refill_plan_laned(
    m: Manifest, rank: int, prefix: Sequence[int],
    cap_rows_by_rank: Sequence[int],
) -> List[Tuple[int, int, int, int]]:
    """refill_plan's rows, tagged with the L2 page lane, as 4-tuples.

    ``(compact_row, l2_slot, l2_gen, lane)`` for exactly the rows and in the
    order l15_restore.refill_plan yields them -- both ride the SAME dedupe
    (``l15_restore.owned_l2_rows``: one entry per distinct compact row,
    shared-prefix visits collapsed, conflicting visits refused) and the
    same cap>0 shortcut -- plus the lane that refill_plan drops because it
    only carries 3-tuples. The lane is ``span.l2_lanes[i]`` at the token's
    own index i in the span (parallel to l2_slots/l2_gens, the P1 contract;
    the shared row's first visit provides it -- sharing spans record the
    same lane or the dedupe refuses), or -1 when the record carries no lane
    for that token: an old record (l2_lanes == ()), a P == 1 form, or a
    l2_lanes shorter than the span (the tail). -1 means "no lane recorded",
    which the P>1 loader (_refill_pgt) refuses named -- it never guesses."""
    if cap_rows_by_rank[rank] > 0:
        return []
    return [(row, slot, gen, lane)
            for row, slot, gen, lane, _rids in owned_l2_rows(m, rank, prefix)]


def gen_check(
    plan: Iterable[Tuple[str, int, int, int, ...]], host_pool
) -> Tuple[List[Tuple[str, int, int, int, ...]], List[str]]:
    """F1 generation check BEFORE copying: one host_pool.slot_gens call.

    Returns (ok_rows, drop_rids). A row whose l2_slot is -1 (no L2 copy)
    or whose recorded l2_gen differs from the arena's current generation
    (incl. -1 = not COMPLETE) marks its WHOLE rid as dropped -- a token
    whose L2 row was re-claimed is foreign, and a half-refilled request is
    worse than a missing one. The entry's rid tag may be a str or the
    tuple of every rid sharing the row (owned_l2_rows): a mismatch drops
    ALL of them whole, never just the first. ok_rows keeps the plan's
    entry order; an entry survives only when NONE of its rids was dropped."""
    rows = list(plan)
    # Unique real slots for the ONE census call, in first-appearance order.
    unique: List[int] = []
    seen: Set[int] = set()
    # Index-based (not structural) unpacking: 4-tuples (P == 1) and 5-tuples
    # (P > 1, lane appended) both read slot = [2], gen = [3].
    for entry in rows:
        slot = entry[2]
        if slot >= 0 and slot not in seen:
            seen.add(slot)
            unique.append(slot)
    current = host_pool.slot_gens(unique)
    gens: Dict[int, int] = dict(zip(unique, (int(g) for g in current)))
    bad: Set[str] = set()
    for entry in rows:
        slot, gen = entry[2], entry[3]
        cur = -1 if slot < 0 else gens.get(slot, -1)
        if cur != int(gen):
            bad.update(_rid_tag(entry[0]))
    ok = [r for r in rows if not (bad & set(_rid_tag(r[0])))]
    return ok, sorted(bad)


MODE_ENV = "FLLIPER_PDFLIP_L15_REFILL_MODE"


def refill_mode(env=None) -> Optional[str]:
    """L15-REFILL-DMA: the page-load mode of the cap-0 refill -- "dma" (one
    H2D copy per run of consecutive arena slots straight out of the
    cudaHostRegistered arena: no CPU gather, no JIT, no stage), "kernel"
    (the GPU gathers whole pages through the mapped arena), "cpu" (pinned
    stage); unset = the pool's own choice (27B's 32-KiB page: "kernel")."""
    import os as _os

    env = _os.environ if env is None else env
    m = str(env.get(MODE_ENV, "") or "").strip().lower()
    return m if m in ("dma", "kernel", "cpu") else None


def refill(
    plan_ok: Sequence[Tuple[str, int, int, int, ...]],
    host_pool,
    device_pool,
    page_tokens: int,
    mode: Optional[str] = None,
) -> int:
    """Bulk H2D of the checked plan: ONE _load_pages_all_layers call,
    return the number of rows copied.

    P == 1 (the 27B form): the slot is the whole L2 address -- slots = the
    unique page slots ascending (a slot run helps the "dma" mode's
    consecutive-slot copies), device_indices = the target compact rows in
    slot order -- the loader's own scatter order with lanes=None and page
    size 1. P > 1 (the NF form): plan rows are 5-tuples (rid, row, l2_slot,
    l2_gen, lane); see _refill_pgt for the page grouping and the loader's
    one-global-lane-list contract. A duplicate slot/lane or a -1 slot raises
    before any copy; any failure of the load itself likewise raises, never
    with a partial count reported as success."""
    if page_tokens != 1:
        return _refill_pgt(plan_ok, host_pool, device_pool, mode=mode)
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
            device_pool, slots_t, didx_t, lanes=None, mode=mode
        )
    except Exception as exc:  # noqa: BLE001 -- all-or-nothing: fold into a bad vote
        raise L15RefillError("refill: page load failed: %r" % (exc,)) from exc
    return len(didx)


def _refill_pgt(
    plan_ok: Sequence[Tuple[str, int, int, int, int]],
    host_pool,
    device_pool,
    mode: Optional[str] = None,
) -> int:
    """P>1 (the NF form) refill: rows grouped by page, ONE loader call.

    Plan rows are 5-tuples ``(rid, row, l2_slot, l2_gen, lane)``; the lane
    is the token's position inside its arena page, recorded at sleep
    (l15_bind: lane = (row - S) % P, HoldSpan.l2_lanes).
    ``_load_pages_all_layers`` takes a SINGLE global ``lanes`` tensor
    applied to EVERY page (arena_pool.py, proven by _owner_page_prefix_
    load at :1562: one lane list per call, len(lanes) rows per page in
    (page, lane) order), so arbitrary per-page lane SUBSETS are not
    expressible: every page loaded must own exactly the same lane set.
    A page with a different set raises L15RefillError naming why, and
    documenting the needed loader change (per-page lane sets in
    arena_pool.py) -- foreign lanes are never copied on a guess."""
    pages: Dict[int, Dict[int, int]] = {}
    for entry in plan_ok:
        if len(entry) != 5:
            raise L15RefillError(
                "refill(P>1): plan rows must be 5-tuples (rid, row, "
                "l2_slot, l2_gen, lane) -- the lane is HoldSpan.l2_lanes, "
                "recorded at sleep; got %r" % (entry,)
            )
        rid = entry[0]
        row, slot, _gen, lane = (int(x) for x in entry[1:])
        if slot < 0:
            raise L15RefillError(
                "refill(P>1): rid %s carries l2_slot -1 (gen_check must "
                "drop it)" % (rid,)
            )
        if lane < 0:
            raise L15RefillError(
                "refill(P>1): rid %s row %d is a staging row (lane -1); "
                "staging rows are not refilled from L2" % (rid, row)
            )
        page = pages.setdefault(slot, {})
        if lane in page:
            raise L15RefillError(
                "refill(P>1): duplicate (page %d, lane %d): rows %d and %d"
                % (slot, lane, page[lane], row)
            )
        page[lane] = row
    if not pages:
        return 0
    common: Set[int] = set()
    common_slot = None
    for slot in sorted(pages):
        lanes = set(pages[slot])
        if not common:
            common, common_slot = lanes, slot
        elif lanes != common:
            raise L15RefillError(
                "refill(P>1): page %d owns lanes %s, page %d owns %s; "
                "_load_pages_all_layers takes ONE global lanes tensor per "
                "call, per-page lane subsets are not expressible -- the "
                "loader would need per-page lane sets (arena_pool.py, "
                "outside this module)" % (
                    common_slot, sorted(common), slot, sorted(lanes)
                )
            )
    lane_list = sorted(common)
    slots_list = sorted(pages)
    didx = [pages[s][l] for s in slots_list for l in lane_list]
    try:
        host_pool._load_pages_all_layers(
            device_pool,
            torch.tensor(slots_list, dtype=torch.int64),
            torch.tensor(didx, dtype=torch.int64),
            lanes=torch.tensor(lane_list, dtype=torch.int64),
            mode=mode,
        )
    except Exception as exc:  # noqa: BLE001 -- all-or-nothing: fold into a bad vote
        raise L15RefillError("refill(P>1): page load failed: %r" % (exc,)) from exc
    return len(didx)
