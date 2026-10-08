"""L15-POOL stage S4: the END anchors (GDN/Mamba state) in the pool -- a WHOLE
request (KV + anchor) is held, or it is not held at all.

User order 04.10.2026 (deskq/L15-DESIGN-VERBINDLICH.md): the free VRAM of all
cards is ONE pool for whole requests, "fuer KV UND END-Anker von GDN/Mamba".
Design: docs/L15-POOL-ENTWURF-1004.md sec 1.2, 3.3, 4.5, 5.2 (N4, anchors in
N3/N5/N6), 7 (S4).

What an anchor is on this rig (read at the code, not derived from log bytes)
---------------------------------------------------------------------------
``hicache_migrate.MambaBlobSpec`` + ``qwen3_5_mamba_spec``: one anchor = the
GDN state of every linear layer, ``[temporal of all layers][conv of all
layers]``.  A D rank holds a HEAD SHARE of it: ``shard_for_rank(ratios,
rank)`` cuts heads and the three conv sub-blocks (q | k | v) with
``partition_sizes(.., ratios, units=gdn_tp_units)`` -- the TP head ratio
vector (``get_tp_partition_ratios``), NOT the DCP token vector.  For the 27B
(48 value heads, 16 key heads, head dims 128, conv width 3, 48 linear layers,
bf16 states) one anchor is 74.8125 MiB; the share per rank follows the vector:

    [2,1,1] / [32,16,16]   37.406 / 18.703 / 18.703 MiB   (the 2:1:1 of the
                                                           boot log is this)
    [1,1,1]                28.055 / 23.379 / 23.379 MiB   (16 units -> 6,5,5)
    [3725,2264,2259]       32.730 / 23.379 / 18.703 MiB

so the shares are NEVER equal, and "every rank = my own slot bytes"
(``own-assumed-equal`` of the S1 shadow) was wrong for every TP > 1 run.  The
anchor of a request has no DCP split: every rank holds a share of EVERY anchor
(``anchor_plan`` compacts the held anchors of all ranks to the slots
``[1, A_H)``, slot 0 = padding, so anchor ``i`` sits at slot ``i + 1`` on every
rank).

Where a guest anchor lies (a decision, with its reason)
-------------------------------------------------------
The Mamba hold region of a capped rank is ``anchor_cap + 1`` slots (default
9 slots = 168 MiB on a 18.7 MiB share), is ZEROED above ``A_H`` by every sleep
flush and wake restore (``reset_state(keep_rows)``) and its slots carry an
allocator ledger.  A foreign 37 MiB share x 5 anchors does not fit it and would
be wiped by the flush.  The KV hold region is the budgeted pool memory (7.6
GiB on the big post), is never zeroed on the sleep and is spared at the wake
down to ``hosted_end``.  So the anchor share of a rank WITHOUT a home segment
(cap 0 -- "a rank without a cap holds no anchor") lies as BYTES in the FREE KV
hold rows of a host rank, priced in rows (``ceil(bytes / row_bytes)``) against
the SAME free rows the KV guests use: the pool counts KV rows and anchor bytes
together against the sum of the segments (N4 -- "Anker zaehlt in die
Poolkapazitaet").  A capped rank keeps its own share at home (Mamba hold
region, unchanged).

S4b -- the DYNAMIC anchor count (``FLLIPER_PDFLIP_L15_POOL_S4B``, user idea 04.10.)
-------------------------------------------------------------------------------
The S4 rule above leaves a capped rank (WITH a home segment) holding its anchors
in the Mamba hold region with a FIXED count (``anchor_cap`` + 1 slots) -- a full
zone refuses a request although the KV hold rows have room.  S4b lets the anchors
of such a rank beyond ``anchor_cap`` lie as bytes in free KV hold rows too: first
the rank's OWN segment (a local copy, no collective), then foreign segments (the
same Q3 host key).  The pool is ONE row budget: admission
(:func:`select_hold_pool_s4b`) charges KV rows and the overflow anchor rows of ALL
ranks against the sum of the segments in the candidate order, the exact plan
(:func:`pool_park_plan_s4` with ``anchor_cap``) puts KV guests and anchors on the
same free rows and cursors (no row twice), ``kv_rows_left`` of the plan line is
what the anchors left for KV.  Everything rides the manifest v2 and the plan
digest (``anchor_cap`` is part of both); ``anchor_cap=None`` is S4 byte for byte.

Pure parts (stdlib): admission, plan, fingerprint, lines.  Torch parts (lazy):
gather/scatter of a slot stream, the source checksum, the transport (one uneven
all_to_all per piece and 16 MiB block, like ``l15_park.run_park``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from flliper.srt.pdflip import l15_pool
from flliper.srt.pdflip.l15_policy import Candidate, HoldSet, _order_key

POOL_S4_ENV = "FLLIPER_PDFLIP_L15_POOL_S4"
POOL_S4B_ENV = "FLLIPER_PDFLIP_L15_POOL_S4B"
_ON_VALUES = ("1", "true", "on", "yes")

REASON_POOL_FULL = l15_pool.REASON_POOL_FULL
REASON_ANCHOR_FULL = l15_pool.REASON_ANCHOR_FULL
REASON_ANCHORLESS = l15_pool.REASON_ANCHORLESS

#: one guest anchor piece, the manifest form:
#: ``(owner, host, a_lo, n_a, host_row_lo, host_rows, nbytes)``
AnchorGuest = Tuple[int, int, int, int, int, int, int]


def pool_s4_flag(env: Mapping[str, str]) -> bool:
    """True when ``FLLIPER_PDFLIP_L15_POOL_S4`` is set (1/true/on/yes), whatever
    the pool / S3 switches say (the launcher refuses the lonely flag by name)."""
    return str(env.get(POOL_S4_ENV, "") or "").strip().lower() in _ON_VALUES


def pool_s4_on(env: Mapping[str, str]) -> bool:
    """True when the anchors take part in the pool: the part switch AND the S3
    pool (S4 never runs without POOL and S3)."""
    return l15_pool.pool_s3_on(env) and pool_s4_flag(env)


def pool_s4b_flag(env: Mapping[str, str]) -> bool:
    """True when ``FLLIPER_PDFLIP_L15_POOL_S4B`` is set (1/true/on/yes), whatever the
    other switches say (the launcher refuses the lonely flag by name)."""
    return str(env.get(POOL_S4B_ENV, "") or "").strip().lower() in _ON_VALUES


def pool_s4b_on(env: Mapping[str, str]) -> bool:
    """True when the anchor count is dynamic: the S4b part switch AND S4 (which
    needs the S3 pool).  S4b never runs without them."""
    return pool_s4_on(env) and pool_s4b_flag(env)


S4B_MODE = ("POOL(S4b: dynamic anchor count -- the anchors of the ranks WITH a home "
            "segment beyond anchor_cap lie as byte pieces in free KV hold rows, own home "
            "segment first, then foreign segments; KV rows and all anchor bytes are "
            "planned together against the sum of the free rows)")

def s4b_boot_lines(env: Mapping[str, str]) -> List[str]:
    """The S4b boot line (``[]`` when the part switch is off): its mode text and
    the anchor cap the Mamba hold region is sized for."""
    if not pool_s4b_on(env):
        return []
    from flliper.srt.pdflip import l15_keep_split

    return ["L15-POOL-S4B mode=%s anchor_cap=%d" % (S4B_MODE, l15_keep_split.anchor_cap(env))]


def capped_own_ranks(manifest) -> Tuple[int, ...]:
    """Ranks WITH a home segment that own guest rows (S3 KV overflow) or anchor
    byte pieces (S4b overflow) in ``manifest``: no L2 refill path for them, so a
    park-back of the group that did not land must drop the hold."""
    if manifest is None:
        return ()
    caps = getattr(manifest, "caps", None)
    return tuple(sorted(set(
        l15_pool.capped_guest_ranks(getattr(manifest, "guests", None), caps))
        | set(capped_anchor_overflow_ranks(getattr(manifest, "anchor_guests", None), caps))))


S4_MODE = ("POOL(S4: whole requests -- KV and the END anchors of the ranks without a "
           "home segment lie as byte guests in the free hold rows of the hosts, "
           "admission counts both against the sum of the segments)")


@dataclass(frozen=True)
class AnchorCtx:
    """What the S4 plan is priced with (rank-uniform, replicated in the manifest)."""

    bytes_by_rank: Tuple[int, ...]  # head share of ONE anchor per rank, bytes
    row_bytes: int  # bytes of one KV hold row (all layers)
    src: str = ""  # where the bytes came from (env | spec(ratios=...))
    # S4b (None = S4): the anchors a rank WITH a home segment keeps in its Mamba
    # hold region (FLLIPER_PDFLIP_L15_ANCHOR_CAP); anchors beyond it are byte pieces
    # in the KV hold rows. Rank-uniform (env), part of the plan digest.
    anchor_cap: Optional[int] = None


def rows_for_bytes(nbytes: int, row_bytes: int) -> int:
    return -(-int(nbytes) // int(row_bytes))


# -- anchor share bytes: read from the MambaBlobSpec ---------------------------------


def anchor_bytes_from_spec(spec, ratio_candidates: Sequence[Sequence[int]], tp: int,
                           rank: int, own_bytes: int
                           ) -> Tuple[Optional[Tuple[int, ...]], str]:
    """``(bytes per rank, src)`` or ``(None, reason)``.

    ``shard_for_rank(ratios, r).total_bytes`` for every rank, for the first
    ratio candidate whose OWN-rank share equals ``own_bytes`` (what the live
    pool really holds per slot) -- the same verification ``cache_controller``
    makes for the window candidates.  A candidate the live pool does not
    confirm is never used: pricing the plan with a vector the pool was not cut
    by would be a silent corruption of the placement."""
    tried = []
    for ratios in ratio_candidates:
        ratios = [int(x) for x in ratios]
        if len(ratios) != int(tp) or any(x < 0 for x in ratios) or sum(ratios) <= 0:
            continue
        try:
            by_rank = tuple(int(spec.shard_for_rank(ratios, r).total_bytes)
                            for r in range(int(tp)))
        except Exception as exc:  # noqa: BLE001 -- a vector the spec cannot cut
            tried.append((ratios, type(exc).__name__))
            continue
        tried.append((ratios, by_rank[rank]))
        if int(own_bytes) > 0 and by_rank[rank] == int(own_bytes):
            return by_rank, "spec(ratios=%s)" % ",".join(str(x) for x in ratios)
    return None, "no head-ratio candidate gives this rank's slot bytes %d (tried %s)" % (
        int(own_bytes), tried)


def kv_row_bytes(kv_views) -> int:
    """Bytes of one KV hold row over all buffers (the cell size)."""
    return sum(int(v.shape[1]) for v in kv_views)


def resolve_anchor_ctx(sched, env: Mapping[str, str], tp: int, rank: int
                       ) -> Tuple[Optional[AnchorCtx], str]:
    """:func:`_resolve_anchor_ctx` plus the S4b anchor cap when the S4b switch is
    on (``ctx.anchor_cap`` = ``FLLIPER_PDFLIP_L15_ANCHOR_CAP``; None = S4)."""
    ctx, why = _resolve_anchor_ctx(sched, env, tp, rank)
    if ctx is not None and pool_s4b_on(env):
        from flliper.srt.pdflip import l15_keep_split

        ctx = dataclasses.replace(ctx, anchor_cap=int(l15_keep_split.anchor_cap(env)))
    return ctx, why


def _resolve_anchor_ctx(sched, env: Mapping[str, str], tp: int, rank: int
                        ) -> Tuple[Optional[AnchorCtx], str]:
    """The S4 pricing of THIS sleep, from the live scheduler: ``(ctx, "ok")`` or
    ``(None, reason)``.  Never raises.

    1. ``FLLIPER_PDFLIP_L15_POOL_ANCHOR_BYTES`` (``"b0,b1,b2"``, explicit operator
       input; malformed / wrong length = named refusal);
    2. the MambaBlobSpec of this checkpoint cut by the head ratio vector the
       live mamba pool was cut by (candidates in the order of the runtime:
       the installed TP plan, ``--rank-tp-ratio``, ``--phase-flip-tp-vector``,
       even), confirmed against this rank's own slot bytes.
    Rank-uniform by construction (config + vector are the same everywhere); the
    plan fingerprint carries the vector, so a rank that disagrees turns the
    round off for everyone."""
    try:
        kv, _pool = _kv_views(sched)
        rb = kv_row_bytes(kv)
        if rb <= 0:
            return None, "no KV rows to price guest bytes in"
        own = l15_pool.own_anchor_slot_bytes(sched)
        raw = env.get(l15_pool.POOL_ANCHOR_BYTES_ENV)
        if raw is not None and str(raw).strip():
            ab = l15_pool.parse_int_list(raw)
            if ab is None or len(ab) != int(tp) or any(x <= 0 for x in ab):
                return None, "%s malformed" % l15_pool.POOL_ANCHOR_BYTES_ENV
            if own > 0 and ab[rank] != own:
                return None, "%s says %d bytes for this rank, the pool holds %d" % (
                    l15_pool.POOL_ANCHOR_BYTES_ENV, ab[rank], own)
            return AnchorCtx(tuple(ab), rb, "env"), "ok"
        from flliper.srt.distributed.utils import get_tp_partition_ratios
        from flliper.srt.managers.cache_controller import parse_tp_ratio_vector
        from flliper.srt.mem_cache.canonical_page_store import derive_mamba_blob_spec

        mr = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
        rtp = getattr(sched, "req_to_token_pool", None)
        mc = getattr(getattr(rtp, "mamba_pool", None), "mamba_cache", None)
        n_layers = int(mc.temporal.shape[0])
        spec = derive_mamba_blob_spec(getattr(mr, "model_config", None),
                                      getattr(rtp, "mamba_pool", None),
                                      num_linear_layers=n_layers)
        sa = getattr(sched, "server_args", None)
        cands = []
        for v in (get_tp_partition_ratios(),
                  parse_tp_ratio_vector(getattr(sa, "rank_tp_ratio", None)),
                  [int(x) for x in str(getattr(sa, "phase_flip_tp_vector", "") or "").split(",")
                   if x.strip()] or None,
                  [1] * int(tp)):
            if v and list(v) not in cands:
                cands.append(list(v))
        by_rank, src = anchor_bytes_from_spec(spec, cands, int(tp), int(rank), own)
        if by_rank is None:
            return None, src
        return AnchorCtx(by_rank, rb, src), "ok"
    except Exception as exc:  # noqa: BLE001 -- no ctx = no S4 round, named
        return None, "%s: %s" % (type(exc).__name__, exc)


# -- admission: KV rows AND anchor bytes against the sum of the segments ------------


def anchor_guest_rows_per_anchor(caps: Sequence[int], anchor_bytes: Sequence[int],
                                 row_bytes: int) -> int:
    """Rows ONE anchor of a request costs the pool as guest: the shares of the
    ranks without a home segment, ``ceil(bytes / row_bytes)`` each (the
    admission's per-anchor estimate; the plan packs several anchors of one
    owner together, which is never more)."""
    return sum(rows_for_bytes(int(b), row_bytes)
               for r, b in enumerate(anchor_bytes)
               if r < len(caps) and int(caps[r]) <= 0 and int(b) > 0)


def select_hold_pool_s4(
    candidates: Sequence[Candidate],
    cap_rows_by_rank: Sequence[int],
    cap_anchor_slots: int,
    anchor_bytes_by_rank: Sequence[int],
    row_bytes: int,
) -> HoldSet:
    """``select_hold_pool_s3`` where the anchor is part of the WHOLE request:
    a candidate is held when its KV rows (all ranks) AND the guest rows of its
    anchor shares (ranks without a home segment) fit the pool together.

    Reasons: ``pool_full`` = the KV rows alone do not fit the free rows;
    ``anchor_full`` = the KV rows fit but not together with the anchor bytes,
    or the anchor count cap is full.  Everything else is today's policy word for
    word: anchorless candidates are excluded up front, seat > parked > served,
    later smaller candidates may still fit.  The sums are the pre-compaction
    estimate; the exact check on the COMPACTED rows is ``plan_round``
    (:func:`pool_park_plan_s4`), before anything moves.  No safety factor."""
    candidates = list(candidates)
    excluded: Dict[str, Tuple[str, str]] = {
        c.rid: (c.rid, REASON_ANCHORLESS) for c in candidates
        if c.anchor_depth != c.kv_depth
    }
    ordered = sorted((c for c in candidates if c.anchor_depth == c.kv_depth),
                     key=_order_key)
    caps = [max(0, int(c)) for c in cap_rows_by_rank]
    total_cap = sum(caps)
    a_rows = anchor_guest_rows_per_anchor(caps, anchor_bytes_by_rank, row_bytes)
    used = 0
    admitted = []
    for c in ordered:
        need = sum(int(v) for v in c.rows_by_rank)
        if used + need > total_cap:
            excluded[c.rid] = (c.rid, REASON_POOL_FULL)
            continue
        if used + need + a_rows > total_cap:
            excluded[c.rid] = (c.rid, REASON_ANCHOR_FULL)
            continue
        if len(admitted) >= cap_anchor_slots:
            excluded[c.rid] = (c.rid, REASON_ANCHOR_FULL)
            continue
        admitted.append(c)
        used += need + a_rows
    rows_by_rank = tuple(sum(c.rows_by_rank[r] for c in admitted) for r in range(len(caps)))
    return HoldSet(
        rids=tuple(c.rid for c in admitted),
        rows_by_rank=rows_by_rank,
        anchors=len(admitted),
        excluded=tuple(excluded.values()),
    )


def anchor_rows_of_index(i: int, caps: Sequence[int], anchor_bytes: Sequence[int],
                         row_bytes: int, anchor_cap: int) -> int:
    """Rows the ``i``-th held anchor (0-based, in admission order) costs the
    pool under S4b: the shares of the ranks WITHOUT a home segment always (they
    hold no anchor at home), the shares of the ranks WITH a home segment only
    from anchor ``anchor_cap`` on (below it they lie in the rank's Mamba hold
    region, which costs no KV row).  ``ceil(bytes / row_bytes)`` per share: the
    plan packs several anchors of one owner into one piece, which is never more."""
    over = int(i) >= int(anchor_cap)
    return sum(rows_for_bytes(int(b), row_bytes)
               for r, b in enumerate(anchor_bytes)
               if r < len(caps) and int(b) > 0 and (int(caps[r]) <= 0 or over))


def select_hold_pool_s4b(
    candidates: Sequence[Candidate],
    cap_rows_by_rank: Sequence[int],
    anchor_bytes_by_rank: Sequence[int],
    row_bytes: int,
    anchor_cap: int,
) -> HoldSet:
    """``select_hold_pool_s4`` with a DYNAMIC anchor count (user idea 04.10.
    ~17:50Z): the pool is one row budget (the sum of the segments), the anchor
    consumes rows, the KV gets the rest.  There is no anchor COUNT cap any more
    (the S4 ``cap_anchor_slots`` bounded the Mamba hold region); what bounds the
    anchors is the rows they take away from the KV.

    Candidates are walked in today's order (seat > parked > served).  Candidate
    ``k`` is held when its KV rows (all ranks) PLUS the anchor rows of anchor
    ``k`` (:func:`anchor_rows_of_index`, which charges a rank with a home
    segment only beyond ``anchor_cap``) fit in what the earlier ones left:
    ``used + kv + anchor <= sum(caps)`` -- ONE sum over KV and anchor together,
    so an anchor can never be counted on rows an already chosen request's KV
    holds, and the total is never exceeded.  Reasons: ``pool_full`` (KV alone
    does not fit), ``anchor_full`` (KV fits, not together with its anchor).  A
    later smaller candidate may still fit.  The sums are the pre-compaction
    estimate; the exact check is :func:`pool_park_plan_s4` on the compacted rows
    (``plan_round``), before anything moves.  No safety factor."""
    candidates = list(candidates)
    excluded: Dict[str, Tuple[str, str]] = {
        c.rid: (c.rid, REASON_ANCHORLESS) for c in candidates
        if c.anchor_depth != c.kv_depth
    }
    ordered = sorted((c for c in candidates if c.anchor_depth == c.kv_depth),
                     key=_order_key)
    caps = [max(0, int(c)) for c in cap_rows_by_rank]
    total_cap = sum(caps)
    used = 0
    admitted = []
    for c in ordered:
        need = sum(int(v) for v in c.rows_by_rank)
        if used + need > total_cap:
            excluded[c.rid] = (c.rid, REASON_POOL_FULL)
            continue
        a_rows = anchor_rows_of_index(len(admitted), caps, anchor_bytes_by_rank,
                                      row_bytes, anchor_cap)
        if need + a_rows + used > total_cap:
            excluded[c.rid] = (c.rid, REASON_ANCHOR_FULL)
            continue
        admitted.append(c)
        used += need + a_rows
    rows_by_rank = tuple(sum(c.rows_by_rank[r] for c in admitted) for r in range(len(caps)))
    return HoldSet(
        rids=tuple(c.rid for c in admitted),
        rows_by_rank=rows_by_rank,
        anchors=len(admitted),
        excluded=tuple(excluded.values()),
    )


# -- the plan: KV guests first (the S3 placement, unchanged), then the anchors -----


def pool_park_plan_s4(
    keep_rows: Sequence[int],
    caps: Sequence[int],
    anchor_bytes: Sequence[int],
    row_bytes: int,
    n_anchors: int,
    rates: Optional[Mapping[Tuple[int, int], float]] = None,
    default_rate: float = 1.0,
    anchor_cap: Optional[int] = None,
):
    """``(kv_pieces, anchor_guests, None, None)`` or ``([], [], reason, code)``
    with ``code`` in ``pool_full`` (a KV guest row has no room) /
    ``anchor_full`` (an anchor share has no room).

    The KV guests are EXACTLY :func:`l15_pool.pool_park_plan` (same loop, same
    Q3 host choice); the anchors then continue on the same free rows and
    cursors: for every rank WITHOUT a home segment (cap 0 -- it holds no
    anchor) with a share ``anchor_bytes[r] > 0``, its ``n_anchors`` shares lie
    as bytes in the free rows of the other segments, host chosen by the same
    key (not the slowest card, free area x directed rate, lower rank), whole
    anchors per piece, split over several hosts when one does not suffice.  A
    piece is ``(owner, host, a_lo, n_a, host_row_lo, host_rows, nbytes)``:
    anchors ``[a_lo, a_lo + n_a)`` (anchor ``i`` = slot ``i + 1``), their
    shares serialised back to back, in ``host_rows = ceil(nbytes / row_bytes)``
    rows from ``host_row_lo``.  Refused by name when they do not fit
    (all-or-nothing: no half hold).  Keyed by cap and rank, never by a card.

    ``anchor_cap`` (S4b; None = S4, the code below byte for byte): a rank WITH a
    home segment keeps its anchors ``[0, anchor_cap)`` in its Mamba hold region;
    the anchors ``[anchor_cap, n_anchors)`` of such a rank are byte pieces too,
    placed home first (the free rows of its OWN segment, after the KV guests),
    then as a guest in the foreign segments by the same Q3 key.  Every rank's
    home-first pass runs before any foreign placement, so an owner's own free
    rows are never taken by another owner's overflow first; all pieces continue
    on the SAME free rows and cursors as the KV guests, so no row is ever given
    twice (KV or anchor)."""
    R = len(keep_rows)
    if len(caps) != R:
        return [], [], "keep_rows for %d ranks, caps for %d" % (R, len(caps)), REASON_POOL_FULL
    if len(anchor_bytes) != R:
        return [], [], "anchor bytes for %d ranks, rows for %d" % (
            len(anchor_bytes), R), REASON_ANCHOR_FULL
    if int(row_bytes) <= 0:
        return [], [], "row bytes %d" % int(row_bytes), REASON_ANCHOR_FULL
    keep = [max(0, int(x)) for x in keep_rows]
    cap = [max(0, int(c)) for c in caps]
    home = [min(keep[r], cap[r]) for r in range(R)]
    free = {r: cap[r] - home[r] for r in range(R) if cap[r] - home[r] > 0}
    cursor = {r: home[r] for r in free}
    rt = l15_pool._Rates(rates, default_rate, [r for r in range(R) if cap[r] > 0])
    pieces = []
    for src in range(R):
        need, row = keep[src] - home[src], home[src]
        while need > 0:
            cands = [h for h, f in free.items() if f > 0 and h != src]
            if not cands:
                have = sum(max(0, v) for v in free.values())
                return [], [], ("rank %d needs %d more guest rows, the other segments "
                                "have %d free" % (src, need, have)), REASON_POOL_FULL
            h = min(cands, key=lambda x: l15_pool._host_key(rt, src, x, free[x]))
            n = min(need, free[h])
            pieces.append(l15_pool.ParkPiece(src, h, row, cursor[h], n))
            cursor[h] += n
            free[h] -= n
            need -= n
            row += n
    guests: List[AnchorGuest] = []
    n_anchors = max(0, int(n_anchors))
    if anchor_cap is not None:
        g4, why4, code4 = _anchor_guests_s4b(
            cap, anchor_bytes, int(row_bytes), n_anchors, int(anchor_cap), free, cursor, rt)
        if why4 is not None:
            return [], [], why4, code4
        return pieces, g4, None, None
    for owner in range(R):
        b = int(anchor_bytes[owner])
        if cap[owner] > 0 or b <= 0 or n_anchors == 0:
            continue
        left, a = n_anchors, 0
        while left > 0:
            cands = [h for h, f in free.items()
                     if h != owner and (f * int(row_bytes)) // b >= 1]
            if not cands:
                have = sum(max(0, v) for v in free.values())
                return [], [], ("anchor share of rank %d (%d x %d bytes) has no room, "
                                "the other segments have %d free rows"
                                % (owner, left, b, have)), REASON_ANCHOR_FULL
            h = min(cands, key=lambda x: l15_pool._host_key(rt, owner, x, free[x]))
            n = min(left, (free[h] * int(row_bytes)) // b)
            rows = rows_for_bytes(n * b, row_bytes)
            guests.append((owner, h, a, n, cursor[h], rows, n * b))
            cursor[h] += rows
            free[h] -= rows
            left -= n
            a += n
    return pieces, guests, None, None


def _anchor_guests_s4b(cap, anchor_bytes, rb, n_anchors, anchor_cap, free, cursor, rt):
    """``(guests, None, None)`` or ``([], reason, "anchor_full")``: the S4b anchor
    pieces on the free rows / cursors the KV guests left (mutated in place)."""
    R = len(cap)
    guests: List[AnchorGuest] = []
    nxt = {}
    # pass 1: HOME FIRST -- a rank with a home segment puts as many whole overflow
    # anchors as fit into the free rows of its own segment
    for owner in range(R):
        b = int(anchor_bytes[owner])
        if cap[owner] <= 0 or b <= 0 or n_anchors == 0:
            continue
        a = min(anchor_cap, n_anchors)
        left = n_anchors - a
        if left > 0 and owner in free:
            n = min(left, (free[owner] * rb) // b)
            if n >= 1:
                rows = rows_for_bytes(n * b, rb)
                guests.append((owner, owner, a, n, cursor[owner], rows, n * b))
                cursor[owner] += rows
                free[owner] -= rows
                a += n
        nxt[owner] = a
    # pass 2: the rest as guests in the FOREIGN segments (Q3 host key)
    for owner in range(R):
        b = int(anchor_bytes[owner])
        if b <= 0 or n_anchors == 0:
            continue
        a = nxt.get(owner, 0) if cap[owner] > 0 else 0
        if cap[owner] > 0 and owner not in nxt:
            continue
        left = n_anchors - a
        while left > 0:
            cands = [h for h, f in free.items()
                     if h != owner and (f * rb) // b >= 1]
            if not cands:
                have = sum(max(0, v) for v in free.values())
                return [], ("anchor share of rank %d (%d x %d bytes) has no room, "
                            "the other segments have %d free rows"
                            % (owner, left, b, have)), REASON_ANCHOR_FULL
            h = min(cands, key=lambda x: l15_pool._host_key(rt, owner, x, free[x]))
            n = min(left, (free[h] * rb) // b)
            rows = rows_for_bytes(n * b, rb)
            guests.append((owner, h, a, n, cursor[h], rows, n * b))
            cursor[h] += rows
            free[h] -= rows
            left -= n
            a += n
    return guests, None, None


def anchor_guest_tuples(guests: Sequence[Sequence[int]]) -> Tuple[AnchorGuest, ...]:
    return tuple(tuple(int(x) for x in g) for g in guests)  # type: ignore[misc]


def plan_fingerprint_s4(rids: Sequence[str], rows_by_rank: Sequence[int],
                        caps: Sequence[int], pieces: Sequence,
                        guests: Sequence[Sequence[int]], anchor_bytes: Sequence[int],
                        row_bytes: int, n_anchors: int,
                        anchor_cap: Optional[int] = None) -> str:
    """The digest of the WHOLE pool decision the group compares: the S3 digest
    (rids, compacted keep rows, caps, KV guest pieces) plus the anchor guests,
    the pricing (share bytes per rank, row bytes) and the anchor count.  Ranks
    that planned another placement or priced it with another vector turn the
    round off everywhere (``agree_pool``).  S4b (``anchor_cap`` not None): the
    anchor cap is part of the decision (it says which anchors lie at home and
    which are pieces); None = the S4 digest byte for byte."""
    import hashlib

    parts = (
        l15_pool.plan_fingerprint(rids, rows_by_rank, caps, pieces),
        anchor_guest_tuples(guests),
        tuple(int(x) for x in anchor_bytes),
        int(row_bytes),
        int(n_anchors),
    )
    if anchor_cap is not None:
        parts = parts + (("s4b", int(anchor_cap)),)
    blob = repr(parts)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def hosted_end_anchor(guests: Optional[Sequence[Sequence[int]]], rank: int) -> int:
    """First row after the last anchor guest row ``rank`` hosts (0 = none)."""
    return max([int(g[4]) + int(g[5]) for g in (guests or ()) if int(g[1]) == int(rank)]
               + [0])


def anchor_guest_totals(guests: Optional[Sequence[Sequence[int]]]) -> Tuple[int, int, int]:
    """``(pieces, bytes, rows)`` of an anchor guest list."""
    g = list(guests or ())
    return len(g), sum(int(x[6]) for x in g), sum(int(x[5]) for x in g)


def owner_covers_all_anchors(guests: Optional[Sequence[Sequence[int]]], rank: int,
                             n_anchors: int) -> bool:
    """True when the pieces ``rank`` OWNS cover anchors ``[0, n_anchors)`` once
    each -- i.e. the pool holds this rank's whole anchor share."""
    if n_anchors <= 0:
        return False
    idx = []
    for g in guests or ():
        if int(g[0]) == int(rank):
            idx.extend(range(int(g[2]), int(g[2]) + int(g[3])))
    return sorted(idx) == list(range(int(n_anchors)))


def plan_line_s4(base_line: str, guests: Sequence[Sequence[int]], n_anchors: int,
                 ctx: AnchorCtx, excluded_codes: Mapping[str, int]) -> str:
    """The S3 ``L15-POOL-PLAN`` line plus the S4 fields: ``anchors`` held,
    ``anchor_pieces`` / ``anchor_guest_bytes`` / ``anchor_guest_rows`` lying in
    foreign segments, who owns and who hosts them, the pricing and its source."""
    n_p, n_b, n_r = anchor_guest_totals(guests)
    pairs: Dict[Tuple[int, int], int] = {}
    for g in guests:
        pairs[(int(g[0]), int(g[1]))] = pairs.get((int(g[0]), int(g[1])), 0) + int(g[6])
    ex = ",".join("%s:%d" % (k, excluded_codes[k]) for k in sorted(excluded_codes)) or "-"
    return (base_line + " s4=1 anchors=%d anchor_pieces=%d anchor_guest_bytes=%d "
            "anchor_guest_rows=%d anchor_pairs=%s anchor_bytes=%s row_bytes=%d "
            "anchor_src=%s s4_excluded=%s" % (
                n_anchors, n_p, n_b, n_r,
                ",".join("%d>%d:%d" % (o, h, b) for (o, h), b in sorted(pairs.items())) or "-",
                ",".join(str(int(x)) for x in ctx.bytes_by_rank), int(ctx.row_bytes),
                ctx.src or "-", ex))


def anchor_overflow_totals(guests: Optional[Sequence[Sequence[int]]],
                           caps: Sequence[int]) -> Tuple[int, int, int, int]:
    """``(pieces, rows, bytes, home_rows)`` of the S4b OVERFLOW anchors: the
    pieces owned by a rank WITH a home segment (``caps[owner] > 0``); the cap-0
    ranks' pieces are S4's.  ``home_rows`` = the rows of those that lie in the
    owner's own segment (host == owner)."""
    g = [x for x in (guests or ())
         if 0 <= int(x[0]) < len(caps) and int(caps[int(x[0])]) > 0]
    return (len(g), sum(int(x[5]) for x in g), sum(int(x[6]) for x in g),
            sum(int(x[5]) for x in g if int(x[0]) == int(x[1])))


def capped_anchor_overflow_ranks(guests: Optional[Sequence[Sequence[int]]],
                                 caps: Optional[Sequence[int]]) -> Tuple[int, ...]:
    """Ranks WITH a home segment that own anchor byte pieces (S4b overflow).
    They have no L2 refill path for them, so a park-back of the group that did
    not land must drop the whole hold (same rule as ``capped_guest_ranks``)."""
    if not guests or caps is None:
        return ()
    return tuple(sorted({int(g[0]) for g in guests
                         if 0 <= int(g[0]) < len(caps) and int(caps[int(g[0])]) > 0}))


def pool_row_budget(caps: Sequence[int], keep_rows: Sequence[int],
                    guests: Optional[Sequence[Sequence[int]]]) -> Tuple[int, int, int, int]:
    """``(pool_rows, kv_rows, anchor_rows, rows_left)``: the pool as ONE row
    budget.  ``pool_rows`` = the sum of the segments, ``kv_rows`` = the held KV
    rows of all ranks (compacted keep rows), ``anchor_rows`` = every row the
    anchor byte pieces occupy (cap-0 owners and overflow alike), ``rows_left`` =
    what is left for more KV.  The plan places KV and anchors on the SAME free
    rows, so ``kv_rows + anchor_rows <= pool_rows`` always holds for a valid
    plan (``rows_left >= 0``)."""
    pool_rows = sum(max(0, int(c)) for c in caps)
    kv = sum(max(0, int(x)) for x in keep_rows)
    a = sum(int(g[5]) for g in (guests or ()))
    return pool_rows, kv, a, pool_rows - kv - a


def plan_line_s4b(base_line: str, guests: Sequence[Sequence[int]], caps: Sequence[int],
                  keep_rows: Sequence[int], anchor_cap: int) -> str:
    """The S4 ``L15-POOL-PLAN`` line plus the S4b fields: the overflow anchors of
    the ranks with a home segment (pieces / rows / bytes, how many rows lie in
    the owner's own segment) and ``kv_rows_left`` = the pool rows no held KV row
    and no anchor byte piece occupies."""
    n_p, n_r, n_b, n_h = anchor_overflow_totals(guests, caps)
    pool_rows, kv_rows, a_rows, left = pool_row_budget(caps, keep_rows, guests)
    return (base_line + " s4b=1 anchor_cap=%d anchor_overflow_pieces=%d "
            "anchor_overflow_rows=%d anchor_overflow_bytes=%d anchor_overflow_home_rows=%d "
            "pool_rows=%d kv_rows_held=%d anchor_rows=%d kv_rows_left=%d" % (
                int(anchor_cap), n_p, n_r, n_b, n_h, pool_rows, kv_rows, a_rows, left))


# -- torch parts ------------------------------------------------------------------


def _rows2d(buf):
    import torch

    if not buf.is_contiguous():
        raise ValueError("anchor park: buffer is not contiguous")
    return buf.view(-1).view(torch.uint8).view(int(buf.shape[0]), -1)


def _kv_views(sched):
    from flliper.srt.pdflip import l15_park

    bufs, pool = l15_park._kv_buffers(sched)
    return [_rows2d(b) for b in bufs], pool


def mamba_views(sched):
    """The per-layer mamba state views as ``(slots, bytes)`` uint8 matrices, in
    the order the retain hook and the keep split use: conv layers, then the
    temporal layers (dim 0 of every view = slots)."""
    mc = getattr(getattr(getattr(sched, "req_to_token_pool", None), "mamba_pool", None),
                 "mamba_cache", None)
    views = []
    for c in getattr(mc, "conv", None) or []:
        views.extend(_rows2d(c[i]) for i in range(int(c.shape[0])))
    temp = getattr(mc, "temporal", None)
    if temp is not None:
        views.extend(_rows2d(temp[i]) for i in range(int(temp.shape[0])))
    return views


def gather_stream(views, rows):
    """``(len(rows), W)`` uint8: row ``rows[i]`` of every view side by side
    (``W`` = sum of the view widths)."""
    import torch

    dev = views[0].device
    idx = torch.as_tensor(list(rows), dtype=torch.int64, device=dev)
    out = torch.empty((int(idx.numel()), sum(int(v.shape[1]) for v in views)),
                      dtype=torch.uint8, device=dev)
    col = 0
    for v in views:
        w = int(v.shape[1])
        out[:, col:col + w] = v.index_select(0, idx)
        col += w
    return out


def scatter_stream(views, rows, mat) -> None:
    """Inverse of :func:`gather_stream`: write ``mat`` (``(len(rows), W)``)
    into row ``rows[i]`` of every view."""
    import torch

    dev = views[0].device
    idx = torch.as_tensor(list(rows), dtype=torch.int64, device=dev)
    if int(mat.shape[0]) != int(idx.numel()) or int(mat.shape[1]) != sum(
            int(v.shape[1]) for v in views):
        raise ValueError("scatter_stream: matrix %s does not fit %d rows x %d bytes" % (
            tuple(mat.shape), int(idx.numel()), sum(int(v.shape[1]) for v in views)))
    col = 0
    for v in views:
        w = int(v.shape[1])
        v.index_copy_(0, idx, mat[:, col:col + w].contiguous())
        col += w


SUM_BLOCK = 1 << 22


def stream_sums(flat) -> List[int]:
    """Position-weighted checksum of a uint8 stream, one int per 4 MiB block
    (block-weighted too): any changed, moved or lost byte changes it."""
    import torch

    flat = flat.reshape(-1)
    out = []
    w = None
    for j, lo in enumerate(range(0, int(flat.numel()), SUM_BLOCK)):
        blk = flat[lo:lo + SUM_BLOCK].to(torch.int64)
        if w is None or int(w.numel()) < int(blk.numel()):
            w = (torch.arange(int(blk.numel()), dtype=torch.int64, device=blk.device)
                 % 251) + 1
        out.append(int((blk * w[:int(blk.numel())]).sum().item()) * (j + 1))
    return out


def _slots_of(g: Sequence[int]) -> List[int]:
    return [1 + int(g[2]) + i for i in range(int(g[3]))]


def anchor_bounds_refusal(guests: Sequence[Sequence[int]], rank: int, kv_views,
                          m_views, row_bytes: int, own_bytes: int) -> Optional[str]:
    """Everything this rank reads or writes lies inside its buffers and the
    widths are the plan's (checked BEFORE the agreement, so no collective starts
    on a plan one rank cannot run)."""
    if not kv_views or not m_views:
        return "no KV / mamba views for the anchor guests"
    if kv_row_bytes(kv_views) != int(row_bytes):
        return "KV row bytes %d != planned %d" % (kv_row_bytes(kv_views), int(row_bytes))
    kv_rows = min(int(v.shape[0]) for v in kv_views)
    slots = min(int(v.shape[0]) for v in m_views)
    w_own = sum(int(v.shape[1]) for v in m_views)
    for g in guests:
        owner, host, a_lo, n_a, h_lo, h_rows, nb = (int(x) for x in g)
        if owner == rank:
            if w_own != int(own_bytes):
                return "own slot bytes %d != planned %d" % (w_own, int(own_bytes))
            if a_lo + n_a + 1 > slots:
                return "anchor slots [%d,%d) beyond the %d-slot mamba views" % (
                    a_lo + 1, a_lo + n_a + 1, slots)
            if nb != n_a * w_own:
                return "piece bytes %d != %d anchors x %d" % (nb, n_a, w_own)
        if host == rank and h_lo + h_rows > kv_rows:
            return "host rows [%d,%d) beyond the %d-row KV buffers" % (
                h_lo, h_lo + h_rows, kv_rows)
    return None


def anchor_sums_of(guests: Sequence[Sequence[int]], rank: int, m_views) -> dict:
    """``{"<piece index>": [block sums]}`` of the anchor streams THIS rank owns,
    read from its mamba slots (taken at the source before the send, and again
    after the shares came back -- a share that arrived wrong or was clobbered in
    the foreign segment is caught independently of L2)."""
    out = {}
    for i, g in enumerate(guests):
        if int(g[0]) != int(rank):
            continue
        out[str(i)] = stream_sums(gather_stream(m_views, _slots_of(g)))
    return out


def run_anchor_park(direction: str, guests: Sequence[Sequence[int]], rank: int,
                    world: int, m_views, kv_views, a2a, env=None,
                    uniform: bool = False) -> int:
    """Move every anchor guest piece: ``direction`` "out" (sleep: the owner's
    mamba slots -> the host's KV hold rows) or "back" (wake: host rows -> the
    owner's mamba slots).  EVERY rank calls this with the same pieces in the
    same order (one collective per piece and block); ``a2a(output, input,
    out_splits, in_splits)`` is the group's uneven all_to_all, rows = host
    rows.  ``uniform`` (S2 pooled hold): each block is announced as
    ``a2a(..., rows=n)``, the rank-uniform row count (see
    ``l15_park.run_park``).  Returns the bytes this rank sent."""
    import torch

    from flliper.srt.pdflip import l15_park

    rb = kv_row_bytes(kv_views)
    sent = 0
    for g in guests:
        owner, host, a_lo, n_a, h_lo, h_rows, nbytes = (int(x) for x in g)
        if owner == host:
            # S4b: an overflow anchor in the owner's OWN segment -- a local copy
            # between its mamba slots and its KV hold rows, no collective (every
            # rank classifies the piece the same way, so none posts one for it;
            # not counted in the bytes SENT over the wire)
            if rank == owner:
                w_own = sum(int(v.shape[1]) for v in m_views)
                rows = range(h_lo, h_lo + h_rows)
                if direction == "out":
                    mat = gather_stream(m_views, _slots_of(g))
                    flat = torch.zeros(h_rows * rb, dtype=torch.uint8,
                                       device=kv_views[0].device)
                    flat[:nbytes] = mat.reshape(-1)
                    scatter_stream(kv_views, rows, flat.view(h_rows, rb))
                else:
                    back = gather_stream(kv_views, rows)
                    scatter_stream(m_views, _slots_of(g),
                                   back.reshape(-1)[:nbytes].view(n_a, w_own))
            continue
        frm, to = (owner, host) if direction == "out" else (host, owner)
        dev = kv_views[0].device
        empty = torch.empty((0, rb), dtype=torch.uint8, device=dev)
        send = None
        recv = None
        if rank == frm:
            if direction == "out":
                mat = gather_stream(m_views, _slots_of(g))
                flat = torch.zeros(h_rows * rb, dtype=torch.uint8, device=dev)
                flat[:nbytes] = mat.reshape(-1)
                send = flat.view(h_rows, rb)
            else:
                send = gather_stream(kv_views, range(h_lo, h_lo + h_rows))
        if rank == to:
            recv = torch.empty((h_rows, rb), dtype=torch.uint8, device=dev)
        step = l15_park.chunk_rows(rb, env)
        for c0 in range(0, h_rows, step):
            n = min(step, h_rows - c0)
            in_splits = [0] * world
            out_splits = [0] * world
            inp = empty
            out = empty
            if rank == frm:
                inp = send[c0:c0 + n]
                in_splits[to] = n
                sent += int(inp.numel())
            if rank == to:
                out = recv[c0:c0 + n]
                out_splits[frm] = n
            if uniform and rank != frm and rank != to:
                # same as l15_park.run_park: BAR1's a2a refuses in/out with one
                # data_ptr, and two empties are both 0 -- the bystander gets a
                # one-row ``out`` (all splits 0: nothing is read or written).
                out = torch.empty((1, rb), dtype=torch.uint8, device=dev)
            if uniform:
                a2a(out, inp, out_splits, in_splits, rows=n)
            else:
                a2a(out, inp, out_splits, in_splits)
        if rank == to:
            if direction == "out":
                scatter_stream(kv_views, range(h_lo, h_lo + h_rows), recv)
            else:
                w_own = sum(int(v.shape[1]) for v in m_views)
                scatter_stream(m_views, _slots_of(g),
                               recv.reshape(-1)[:nbytes].view(n_a, w_own))
    return sent
