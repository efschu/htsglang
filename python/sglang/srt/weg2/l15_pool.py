"""L15-POOL (stage S1): the pooled L1.5 hold model, pure and stdlib-only.

User order 04.10.2026 (deskq/L15-DESIGN-VERBINDLICH.md): the NOT FILLED VRAM
of all cards together is ONE L1.5 pool that holds WHOLE requests -- every KV
shard AND the END anchor (GDN/Mamba state, one HEAD share per D rank) -- and
admission runs against the SUM of the free areas, not per card/rank.  A
shard lies in its own card first ("home"); only the overflow (and the share
of a rank that has no segment at all) lies as a "guest" in a FOREIGN segment.
Design: docs/L15-POOL-ENTWURF-1004.md sec 3.1-3.3, 4, 5.2 (N1), 7 (S1).

STAGE S1 = INSTRUMENT + MODEL, NO BEHAVIOUR.  Nothing here moves a byte or
changes what the sleep holds.  The scheduler calls :func:`shadow_compare`
only behind ``SGLANG_WEG2_L15_POOL_SHADOW`` (default off) and only logs
``L15-POOL-SHADOW``; today's per-rank path (l15_policy.select_hold, the
retain) is untouched and stays the only decider.  No torch, no CUDA: the
whole module is hermetically testable.

Model
-----
* :class:`Segment` -- one per D rank: KV hold rows, anchor slots, and the
  byte sizes to price guests.  A rank whose cap is 0 has a segment with
  ``kv_rows == 0`` and ``anchor_slots == 0`` (or no segment at all): it
  holds nothing at home, its whole share is guest.
* :func:`pool_admit` -- greedy admission in the caller's order (the caller
  applies today's ordering, :func:`admit_candidates` does) against the pooled
  free space.  Per request ALL-OR-NOTHING: every KV shard AND every anchor
  share must find a place, else the request is not held (reason
  ``pool_full`` / ``anchor_full`` / ``anchorless``); a later, smaller request
  may still fit (same as today's policy).
* Placement: home first, cost 0; the overflow goes to a foreign segment
  chosen by (not the slowest card) > (free space x rate, larger first) >
  (lower rank), Q3 of the design; a KV shard may be split over several
  hosts, an anchor share goes to ONE host (conservative: a byte-piece split
  would admit more; S4 decides).
* Rates are an INPUT (``rates[(src_rank, dst_rank)]`` in any consistent unit,
  from the barlink measurement matrix); none given = uniform, no card is the
  slowest.  Nothing in here knows a card name, an ordinal or a UUID
  (HW-GENERIC order 07:05Z).

No additional safety factor or ceiling is applied on top of the caller's
capacities (same rule as l15_policy: the caps are the budget).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from sglang.srt.weg2.l15_park import ParkPiece
from sglang.srt.weg2.l15_policy import Candidate, HoldSet, _order_key, select_hold

POOL_SHADOW_ENV = "SGLANG_WEG2_L15_POOL_SHADOW"
POOL_ENV = "SGLANG_WEG2_L15_POOL"
POOL_RATES_ENV = "SGLANG_WEG2_L15_POOL_RATES"
POOL_ANCHOR_BYTES_ENV = "SGLANG_WEG2_L15_POOL_ANCHOR_BYTES"
_ON_VALUES = ("1", "true", "on", "yes")

REASON_POOL_FULL = "pool_full"
REASON_ANCHOR_FULL = "anchor_full"
REASON_ANCHORLESS = "anchorless"


def pool_shadow_on(env: Mapping[str, str]) -> bool:
    """True when the S1 pool SHADOW is switched on (default off)."""
    return str(env.get(POOL_SHADOW_ENV, "") or "").strip().lower() in _ON_VALUES


def pool_on(env: Mapping[str, str]) -> bool:
    """True when the S2 pooled hold takes effect (default off)."""
    return str(env.get(POOL_ENV, "") or "").strip().lower() in _ON_VALUES


# -- the model ---------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """One D rank's slice of the pool (its hold region on its card)."""

    rank: int
    kv_rows: int  # KV hold rows (capacity); 0 = no home rows
    anchor_slots: int  # anchor (mamba) slots usable for held anchors; 0 = none
    row_bytes: int = 0  # bytes of one KV row (all layers); prices guest bytes
    anchor_slot_bytes: int = 0  # bytes of ONE anchor slot on this rank
    card: int = -1  # informational (budget ordinal); never used to decide


@dataclass(frozen=True)
class RequestShards:
    """One WHOLE request: the KV rows each rank owns (DCP owner rule)."""

    rid: str
    rows_by_rank: Tuple[int, ...]


@dataclass(frozen=True)
class Piece:
    """One placed component: ``owner`` rank's share lying on ``host``."""

    kind: str  # "kv" | "anchor"
    owner: int
    host: int
    amount: int  # rows (kv) or slots on the host (anchor)
    nbytes: int  # bytes moved when this piece is a guest

    @property
    def home(self) -> bool:
        return self.owner == self.host


@dataclass(frozen=True)
class Placement:
    rid: str
    pieces: Tuple[Piece, ...]


@dataclass(frozen=True)
class PoolVerdict:
    """Result of one pooled admission."""

    n_ranks: int
    admitted: Tuple[str, ...]
    placements: Tuple[Placement, ...]
    excluded: Tuple[Tuple[str, str], ...]  # (rid, reason)
    rows_by_rank: Tuple[int, ...]  # admitted KV rows per owner rank (all of them)
    free_kv_by_rank: Tuple[int, ...]  # rows still free per HOST rank
    free_anchor_by_rank: Tuple[int, ...]  # anchor slots still free per host rank
    capacity_kv_rows: int
    capacity_anchor_slots: int

    # -- derived figures (all from the pieces; nothing is stored twice) --
    def _sum_pieces(self, kind: str, home: bool, attr: str, by: str) -> Tuple[int, ...]:
        out = [0] * self.n_ranks
        for pl in self.placements:
            for p in pl.pieces:
                if p.kind == kind and p.home == home:
                    out[getattr(p, by)] += getattr(p, attr)
        return tuple(out)

    @property
    def tokens(self) -> int:
        return sum(self.rows_by_rank)

    @property
    def kv_rows_home_by_rank(self) -> Tuple[int, ...]:
        """Home KV rows per OWNER rank."""
        return self._sum_pieces("kv", True, "amount", "owner")

    @property
    def kv_rows_guest_by_rank(self) -> Tuple[int, ...]:
        """Guest KV rows per OWNER rank (rows lying in a foreign segment)."""
        return self._sum_pieces("kv", False, "amount", "owner")

    @property
    def anchors_home(self) -> int:
        return sum(self._sum_pieces("anchor", True, "amount", "owner"))

    @property
    def anchors_guest(self) -> int:
        """Anchor SHARES (not slots) lying in a foreign segment."""
        return sum(
            1 for pl in self.placements for p in pl.pieces
            if p.kind == "anchor" and not p.home
        )

    @property
    def guest_pairs(self) -> Dict[Tuple[int, int], int]:
        """``{(owner, host): bytes}`` over KV and anchor guests together."""
        out: Dict[Tuple[int, int], int] = {}
        for pl in self.placements:
            for p in pl.pieces:
                if not p.home:
                    out[(p.owner, p.host)] = out.get((p.owner, p.host), 0) + p.nbytes
        return dict(sorted(out.items()))

    def guest_bytes_of(self, kind: str) -> int:
        return sum(
            p.nbytes for pl in self.placements for p in pl.pieces
            if p.kind == kind and not p.home
        )

    @property
    def guest_bytes(self) -> int:
        return sum(self.guest_pairs.values())

    def excluded_counts(self) -> Dict[str, int]:
        out = {REASON_POOL_FULL: 0, REASON_ANCHOR_FULL: 0, REASON_ANCHORLESS: 0}
        for _rid, why in self.excluded:
            out[why] = out.get(why, 0) + 1
        return out


# -- capacity ----------------------------------------------------------------


def pool_capacity(segments: Sequence[Segment]) -> Tuple[int, int]:
    """``(kv_rows, anchor_slots)``: the pool is the SUM of its segments."""
    return (
        sum(max(0, int(s.kv_rows)) for s in segments),
        sum(max(0, int(s.anchor_slots)) for s in segments),
    )


def segments_from_caps(
    caps_rows_by_rank: Sequence[int],
    anchor_cap: int,
    row_bytes: int = 0,
    anchor_slot_bytes_by_rank: Optional[Sequence[int]] = None,
    cards: Optional[Sequence[int]] = None,
) -> Tuple[Segment, ...]:
    """The segments today's caps already describe (same capacities, so the
    comparison with today's path is like for like): rank r holds
    ``caps[r]`` KV rows and, when it holds anything, ``anchor_cap`` anchor
    slots (``l15_keep_split.ensure_split_for_sched``: a cap-0 rank keeps no
    mamba region either)."""
    segs = []
    for r, cap in enumerate(caps_rows_by_rank):
        cap = max(0, int(cap))
        slot_b = (
            int(anchor_slot_bytes_by_rank[r])
            if anchor_slot_bytes_by_rank is not None and r < len(anchor_slot_bytes_by_rank)
            else 0
        )
        segs.append(
            Segment(
                rank=r,
                kv_rows=cap,
                anchor_slots=max(0, int(anchor_cap)) if cap > 0 else 0,
                row_bytes=int(row_bytes),
                anchor_slot_bytes=max(0, slot_b),
                card=int(cards[r]) if cards is not None and r < len(cards) else -1,
            )
        )
    return tuple(segs)


# -- the placement choice (Q3) -------------------------------------------------


class _Rates:
    """Directional rates with a uniform fallback; finds the slowest segment."""

    def __init__(self, rates: Optional[Mapping[Tuple[int, int], float]],
                 default: float, ranks: Sequence[int]) -> None:
        self._r = {(int(a), int(b)): float(v) for (a, b), v in (rates or {}).items()
                   if float(v) > 0.0}
        self._d = float(default) if float(default) > 0.0 else 1.0
        self.slow = self._slowest(list(ranks))

    def get(self, src: int, dst: int) -> float:
        v = self._r.get((src, dst))
        if v is None:
            v = self._r.get((dst, src))  # a link measured one way, same wire
        return self._d if v is None else v

    def _slowest(self, ranks: List[int]) -> frozenset:
        """The ranks whose mean INBOUND rate is strictly the lowest; empty
        when every rank is equal (including no rates at all)."""
        if len(ranks) < 2 or not self._r:
            return frozenset()
        speed = {}
        for h in ranks:
            srcs = [s for s in ranks if s != h]
            speed[h] = sum(self.get(s, h) for s in srcs) / len(srcs)
        lo, hi = min(speed.values()), max(speed.values())
        if hi - lo <= 1e-12 * max(1.0, hi):
            return frozenset()
        return frozenset(h for h, v in speed.items() if abs(v - lo) <= 1e-12 * max(1.0, hi))


def _host_key(rates: _Rates, owner: int, host: int, free: int):
    """Smaller sorts first: not the slowest card, then free x rate larger,
    then the lower rank (deterministic)."""
    return (1 if host in rates.slow else 0, -(free * rates.get(owner, host)), host)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _try_place(
    rows_by_rank: Sequence[int],
    anchor_bytes_by_rank: Optional[Sequence[int]],
    free_kv: Dict[int, int],
    free_a: Dict[int, int],
    seg: Dict[int, Segment],
    default_row_bytes: int,
    rates: _Rates,
    anchor_in_kv: bool = False,
) -> Tuple[Optional[List[Piece]], Optional[str]]:
    """Place ONE whole request on COPIES of the free state (the caller
    commits only on success).  ``(pieces, None)`` or ``(None, reason)``."""
    pieces: List[Piece] = []

    def row_bytes_of(owner: int) -> int:
        s = seg.get(owner)
        rb = int(s.row_bytes) if s is not None else 0
        return rb if rb > 0 else default_row_bytes

    # KV: home first, the overflow as guest rows
    for r, rows in enumerate(rows_by_rank):
        need = int(rows)
        if need <= 0:
            continue
        home = min(need, max(0, free_kv.get(r, 0)))
        if home > 0:
            pieces.append(Piece("kv", r, r, home, 0))
            free_kv[r] -= home
            need -= home
        while need > 0:
            cands = [h for h, f in free_kv.items() if h != r and f > 0]
            if not cands:
                return None, REASON_POOL_FULL
            h = min(cands, key=lambda x: _host_key(rates, r, x, free_kv[x]))
            n = min(need, free_kv[h])
            pieces.append(Piece("kv", r, h, n, n * row_bytes_of(r)))
            free_kv[h] -= n
            need -= n

    # anchors: one head share per rank; home slot first, else ONE guest host
    if anchor_bytes_by_rank is not None:
        for r, b in enumerate(anchor_bytes_by_rank):
            b = int(b)
            if b <= 0:
                continue
            if r in seg and free_a.get(r, 0) >= 1:
                pieces.append(Piece("anchor", r, r, 1, 0))
                free_a[r] -= 1
                continue
            if anchor_in_kv:
                # S4 (l15_pool_anchor): the guest share lies as bytes in the
                # FREE KV hold rows of a host, priced in rows of that host
                def _rows_h(h):
                    return _ceil_div(b, max(1, row_bytes_of(h)))

                cands = [h for h, f in free_kv.items()
                         if h != r and row_bytes_of(h) > 0 and f >= _rows_h(h)]
                if not cands:
                    return None, REASON_ANCHOR_FULL
                h = min(cands, key=lambda x: _host_key(rates, r, x, free_kv[x]))
                rows = _rows_h(h)
                pieces.append(Piece("anchor", r, h, rows, b))
                free_kv[h] -= rows
                continue
            cands = []
            for h, f in free_a.items():
                sb = int(seg[h].anchor_slot_bytes) if h in seg else 0
                if h != r and f > 0 and sb > 0 and f >= _ceil_div(b, sb):
                    cands.append(h)
            if not cands:
                return None, REASON_ANCHOR_FULL
            h = min(cands, key=lambda x: _host_key(rates, r, x, free_a[x]))
            slots = _ceil_div(b, int(seg[h].anchor_slot_bytes))
            pieces.append(Piece("anchor", r, h, slots, b))
            free_a[h] -= slots
    return pieces, None


def _norm_requests(request_shards) -> List[RequestShards]:
    out = []
    seen = set()
    for item in request_shards:
        rs = item if isinstance(item, RequestShards) else RequestShards(
            rid=item[0], rows_by_rank=tuple(item[1]))
        rows = tuple(int(x) for x in rs.rows_by_rank)
        if any(x < 0 for x in rows):
            raise ValueError(f"pool_admit: negative rows for {rs.rid!r}")
        if rs.rid in seen:
            raise ValueError(f"pool_admit: duplicate rid {rs.rid!r}")
        seen.add(rs.rid)
        out.append(RequestShards(rs.rid, rows))
    return out


def pool_admit(
    request_shards: Sequence,
    anchors: Mapping[str, Optional[Sequence[int]]],
    segments: Sequence[Segment],
    rates: Optional[Mapping[Tuple[int, int], float]] = None,
    default_rate: float = 1.0,
    anchor_in_kv: bool = False,
) -> PoolVerdict:
    """Admit whole requests against the SUM of the segments.

    ``request_shards``: ordered ``RequestShards`` (or ``(rid, rows_by_rank)``)
    -- the caller's admission order is kept (today's seat > parked > served,
    younger first; see :func:`admit_candidates`).
    ``anchors``: ``rid -> anchor bytes per rank`` (the HEAD share each rank
    holds of that request's end anchor); a rid that is missing or maps to
    ``None`` has no anchor and is excluded ``anchorless`` -- a request is
    only whole with its anchor.
    ``segments``: one :class:`Segment` per rank (ranks may be missing: a rank
    without a segment holds nothing at home).  ``rates``: optional
    ``{(src_rank, dst_rank): rate}`` for the guest choice.

    Never raises on a request that does not fit (it is excluded with a
    reason); raises ``ValueError`` only on malformed input (negative rows,
    duplicate rid, duplicate segment rank).
    """
    reqs = _norm_requests(request_shards)
    seg: Dict[int, Segment] = {}
    for s in segments:
        if int(s.rank) in seg:
            raise ValueError(f"pool_admit: duplicate segment for rank {s.rank}")
        seg[int(s.rank)] = s
    n_ranks = max([len(r.rows_by_rank) for r in reqs] + [max(seg) + 1 if seg else 0])
    free_kv = {r: max(0, int(s.kv_rows)) for r, s in seg.items()}
    free_a = {r: max(0, int(s.anchor_slots)) for r, s in seg.items()}
    default_rb = next((int(s.row_bytes) for s in segments if int(s.row_bytes) > 0), 0)
    rt = _Rates(rates, default_rate, sorted(r for r in seg if free_kv[r] or free_a[r]))

    admitted: List[str] = []
    placements: List[Placement] = []
    excluded: List[Tuple[str, str]] = []
    for req in reqs:
        ab = anchors.get(req.rid) if anchors is not None else None
        if ab is None:
            excluded.append((req.rid, REASON_ANCHORLESS))
            continue
        kv_try, a_try = dict(free_kv), dict(free_a)
        pieces, why = _try_place(req.rows_by_rank, ab, kv_try, a_try, seg, default_rb, rt,
                                 anchor_in_kv)
        if pieces is None:
            excluded.append((req.rid, why))
            continue
        free_kv, free_a = kv_try, a_try
        admitted.append(req.rid)
        placements.append(Placement(req.rid, tuple(pieces)))

    rows = [0] * n_ranks
    by_rid = {r.rid: r for r in reqs}
    for rid in admitted:
        for r, v in enumerate(by_rid[rid].rows_by_rank):
            rows[r] += v
    cap_kv, cap_a = pool_capacity(segments)
    return PoolVerdict(
        n_ranks=n_ranks,
        admitted=tuple(admitted),
        placements=tuple(placements),
        excluded=tuple(excluded),
        rows_by_rank=tuple(rows),
        free_kv_by_rank=tuple(free_kv.get(r, 0) for r in range(n_ranks)),
        free_anchor_by_rank=tuple(free_a.get(r, 0) for r in range(n_ranks)),
        capacity_kv_rows=cap_kv,
        capacity_anchor_slots=cap_a,
    )


def admit_candidates(
    candidates: Iterable[Candidate],
    segments: Sequence[Segment],
    anchor_bytes_by_rank: Optional[Sequence[int]] = None,
    rates: Optional[Mapping[Tuple[int, int], float]] = None,
    default_rate: float = 1.0,
    anchor_in_kv: bool = False,
) -> PoolVerdict:
    """Today's candidate set through the pooled admission.

    Same pre-selection and ordering as ``l15_policy.select_hold``: an
    anchorless candidate (anchor_depth != kv_depth) is excluded up front,
    the rest are ordered seat > parked > served, younger first, rid
    ascending.  Every candidate's anchor shares are the per-rank
    ``anchor_bytes_by_rank`` (default: one slot's bytes of each segment).
    """
    cands = list(candidates)
    anchorless = [(c.rid, REASON_ANCHORLESS) for c in cands if c.anchor_depth != c.kv_depth]
    ordered = sorted((c for c in cands if c.anchor_depth == c.kv_depth), key=_order_key)
    n = max([len(c.rows_by_rank) for c in cands] + [max((s.rank for s in segments), default=-1) + 1])
    if anchor_bytes_by_rank is None:
        by_rank = {int(s.rank): int(s.anchor_slot_bytes) for s in segments}
        shares = tuple(by_rank.get(r, 0) for r in range(n))
    else:
        shares = tuple(int(x) for x in anchor_bytes_by_rank)
    v = pool_admit(
        [RequestShards(c.rid, tuple(c.rows_by_rank)) for c in ordered],
        {c.rid: shares for c in ordered},
        segments,
        rates=rates,
        default_rate=default_rate,
        anchor_in_kv=anchor_in_kv,
    )
    if not anchorless:
        return v
    return PoolVerdict(
        **{**v.__dict__, "excluded": tuple(anchorless) + v.excluded}
    )


# -- the S1 instrument -----------------------------------------------------------


@dataclass(frozen=True)
class PoolShadow:
    """What the pool would hold at one sleep vs what today's path holds."""

    today: HoldSet
    pool: PoolVerdict
    n_candidates: int
    today_tokens: int
    anchor_cap: int

    @property
    def today_only(self) -> Tuple[str, ...]:
        s = set(self.pool.admitted)
        return tuple(r for r in self.today.rids if r not in s)

    @property
    def pool_only(self) -> Tuple[str, ...]:
        s = set(self.today.rids)
        return tuple(r for r in self.pool.admitted if r not in s)


def shadow_compare(
    candidates: Sequence[Candidate],
    caps_rows_by_rank: Sequence[int],
    anchor_cap: int,
    row_bytes: int = 0,
    anchor_slot_bytes_by_rank: Optional[Sequence[int]] = None,
    rates: Optional[Mapping[Tuple[int, int], float]] = None,
    cards: Optional[Sequence[int]] = None,
    anchor_in_kv: bool = False,
) -> PoolShadow:
    """Run today's ``select_hold`` and the pooled admission on the SAME
    candidates and the SAME capacities.  Pure; touches nothing.  Today's
    side uses the real retain cap (``anchor_cap`` anchors)."""
    cands = list(candidates)
    caps = [max(0, int(c)) for c in caps_rows_by_rank]
    today = select_hold(cands, caps, int(anchor_cap))
    segs = segments_from_caps(caps, anchor_cap, row_bytes, anchor_slot_bytes_by_rank, cards)
    pool = admit_candidates(cands, segs, rates=rates, anchor_in_kv=anchor_in_kv)
    return PoolShadow(
        today=today,
        pool=pool,
        n_candidates=len(cands),
        today_tokens=sum(today.rows_by_rank),
        anchor_cap=int(anchor_cap),
    )


def _fmt(values: Iterable) -> str:
    return ",".join(str(v) for v in values) or "-"


def shadow_line(epoch, ps: PoolShadow, caps_rows_by_rank: Sequence[int],
                anchor_src: str = "") -> str:
    """``L15-POOL-SHADOW at=sleep``: one line, log only.

    Reads: ``req`` candidates; ``today_n/today_tokens`` what the per-rank
    path would hold; ``pool_n/pool_tokens`` what the pool would hold whole
    (KV + anchor); ``today_only/pool_only`` requests only one side holds;
    ``kv_rows_home/guest`` per OWNER rank; ``anchors home/guest`` shares;
    ``guest_pairs`` owner>host:bytes; ``excluded`` the pool's reasons.
    """
    v = ps.pool
    pairs = ",".join(f"{o}>{h}:{b}" for (o, h), b in v.guest_pairs.items()) or "-"
    ex = v.excluded_counts()
    return (
        f"L15-POOL-SHADOW at=sleep epoch={epoch} req={ps.n_candidates} "
        f"today_n={len(ps.today.rids)} today_tokens={ps.today_tokens} "
        f"pool_n={len(v.admitted)} pool_tokens={v.tokens} "
        f"today_only={len(ps.today_only)} pool_only={len(ps.pool_only)} "
        f"kv_rows_home={_fmt(v.kv_rows_home_by_rank)} "
        f"kv_rows_guest={_fmt(v.kv_rows_guest_by_rank)} "
        f"anchors_home={v.anchors_home} anchors_guest={v.anchors_guest} "
        f"guest_bytes={v.guest_bytes} guest_kv_bytes={v.guest_bytes_of('kv')} "
        f"guest_anchor_bytes={v.guest_bytes_of('anchor')} guest_pairs={pairs} "
        f"excluded=pool_full:{ex[REASON_POOL_FULL]},"
        f"anchor_full:{ex[REASON_ANCHOR_FULL]},anchorless:{ex[REASON_ANCHORLESS]} "
        f"today_excluded={len(ps.today.excluded)} "
        f"cap_rows={_fmt(caps_rows_by_rank)} cap_sum_rows={v.capacity_kv_rows} "
        f"anchor_cap={ps.anchor_cap} cap_sum_anchor_slots={v.capacity_anchor_slots}"
        + (f" anchor_src={anchor_src}" if anchor_src else "")
    )


SHADOW_MODE = "SHADOW(log-only, no behaviour change)"
POOL_MODE = ("POOL(S2: KV hold admitted against the sum of the segments, a rank "
             "without a home segment parks its shards in foreign segments)")


def boot_line(mib_by_card: Sequence[int], src_by_card: Sequence[str],
              anchor_cap: int, mode: str = SHADOW_MODE) -> str:
    """The launcher's ``L15-POOL`` line: the pool as the SUM of the per-card
    posts (the planner still sizes per card -- physically it cannot be
    otherwise -- but admission is one pool).  S1: informational only."""
    mibs = [int(m) for m in mib_by_card]
    segs = sum(1 for m in mibs if m > 0)
    srcs = sorted(set(str(s) for s in src_by_card)) or ["none"]
    return (
        f"L15-POOL cards={len(mibs)} segments={segs} mib={_fmt(mibs)} "
        f"total_mib={sum(mibs)} anchor_cap={int(anchor_cap)} "
        f"src={','.join(srcs)} mode={mode}"
    )


# -- S2: admission against the SUM for the rank without a home segment ----------


REASON_NO_ROOM = "no_room"
REASON_KEEP_OVER_CAP = "keep_over_cap"


def select_hold_pool(
    candidates: Sequence[Candidate],
    cap_rows_by_rank: Sequence[int],
    cap_anchor_slots: int,
) -> HoldSet:
    """``l15_policy.select_hold`` with the pool rule for a rank that has NO home
    segment (cap 0): its KV shards are no longer waved through ("refilled from
    L2 at the wake") but must find GUEST rows in the capped ranks' segments --
    the admission runs against the SUM of the free rows, not rank by rank.

    Everything else is today's policy word for word: anchorless candidates are
    excluded up front, the order is seat > parked > served / younger / rid,
    a capped rank's own rows must fit its own remaining cap (``no_room``;
    capped ranks overflowing into foreign segments is S3), the anchor count
    cap gives ``anchor_full``, later smaller candidates may still fit.  New:
    ``pool_full`` = the home rows fit but the cap-0 ranks' rows do not fit the
    capped ranks' free rows together.  With no cap-0 rank (or no rows there)
    the result is identical to ``select_hold`` (pinned by a test).

    The row sums here are the pre-compaction estimate; the exact check on the
    compacted rows is ``l15_retain.plan_round`` (it drops the lowest-priority
    request until ``park_plan`` places every guest row), BEFORE anything moves.
    No safety factor, no extra ceiling: the caps are the budget.
    """
    candidates = list(candidates)
    excluded: Dict[str, Tuple[str, str]] = {
        c.rid: (c.rid, REASON_ANCHORLESS)
        for c in candidates
        if c.anchor_depth != c.kv_depth
    }
    ordered = sorted(
        (c for c in candidates if c.anchor_depth == c.kv_depth), key=_order_key
    )
    caps = [max(0, int(c)) for c in cap_rows_by_rank]
    rem = list(caps)
    capped = [r for r, c in enumerate(caps) if c > 0]
    guest_used = 0
    admitted = []
    for c in ordered:
        rows = list(c.rows_by_rank)
        # a capped rank's own rows must fit its own remaining cap -- decided
        # by the ORIGINAL cap (a full capped rank, rem == 0, blocks)
        if any(caps[r] > 0 and v > rem[r] for r, v in enumerate(rows)):
            excluded[c.rid] = (c.rid, REASON_NO_ROOM)
            continue
        need = sum(v for r, v in enumerate(rows) if caps[r] == 0)
        free_guest = sum(rem[h] - (rows[h] if h < len(rows) else 0) for h in capped) - guest_used
        if need > free_guest:
            excluded[c.rid] = (c.rid, REASON_POOL_FULL)
            continue
        if len(admitted) >= cap_anchor_slots:
            excluded[c.rid] = (c.rid, REASON_ANCHOR_FULL)
            continue
        admitted.append(c)
        for r, v in enumerate(rows):
            if caps[r] > 0:
                rem[r] -= v
        guest_used += need
    rows_by_rank = tuple(sum(c.rows_by_rank[r] for c in admitted) for r in range(len(caps)))
    return HoldSet(
        rids=tuple(c.rid for c in admitted),
        rows_by_rank=rows_by_rank,
        anchors=len(admitted),
        excluded=tuple(excluded.values()),
    )


def plan_fingerprint(rids: Sequence[str], rows_by_rank: Sequence[int],
                     caps: Sequence[int], pieces: Sequence) -> str:
    """A short digest of the pool decision: the held rids in order, the
    compacted keep rows per rank, the caps and the guest pieces
    (``src, dst, src_row, dst_row, rows``).  Every rank derives it from the
    SAME replicated lists; the group agreement compares it, so ranks that
    planned a different pool (a divergence that would start mismatched
    collectives) turn the round off instead of running it."""
    import hashlib

    blob = repr((
        tuple(str(r) for r in rids),
        tuple(int(x) for x in rows_by_rank),
        tuple(int(x) for x in caps),
        tuple((int(p.src), int(p.dst), int(p.src_row), int(p.dst_row), int(p.rows))
              for p in pieces),
    ))
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def agree_pool(why: Optional[str], fp: Optional[str], gather) -> Optional[str]:
    """The group's ONE vote of a pooled round: ``None`` when every rank votes
    ok AND every rank that planned a pool planned the SAME one (equal
    :func:`plan_fingerprint`); otherwise the first named refusal.  Exactly one
    gather, whatever the local state (like ``l15_sleep_agree.agree``)."""
    votes = gather((why, fp))
    bad = [v[0] for v in votes if v[0] is not None]
    if bad:
        return str(bad[0])
    fps = sorted({str(v[1]) for v in votes if v[1] is not None})
    nones = sum(1 for v in votes if v[1] is None)
    if len(fps) > 1 or (fps and nones):
        return "pool plan diverged across ranks (fingerprints %s, %d without a plan)" % (
            ",".join(fps), nones)
    return None


def plan_line(epoch, hs: HoldSet, caps: Sequence[int], keep_rows: Sequence[int],
              pieces: Sequence, fp: str, n_candidates: int) -> str:
    """``L15-POOL-PLAN at=sleep`` (S2, written by the planning, before anything
    moves): what the pooled hold keeps, where the guest rows go, why the rest
    stays out."""
    pairs = {}
    for p in pieces:
        pairs[(int(p.src), int(p.dst))] = pairs.get((int(p.src), int(p.dst)), 0) + int(p.rows)
    reasons: Dict[str, int] = {}
    for _rid, why in hs.excluded:
        reasons[why] = reasons.get(why, 0) + 1
    ex = ",".join("%s:%d" % (k, reasons[k]) for k in sorted(reasons)) or "-"
    return (
        f"L15-POOL-PLAN at=sleep epoch={epoch} req={n_candidates} held={len(hs.rids)} "
        f"keep_rows={_fmt(keep_rows)} caps={_fmt(caps)} "
        f"guest_rows={sum(pairs.values())} "
        f"guest_pairs={','.join(f'{o}>{h}:{n}' for (o, h), n in sorted(pairs.items())) or '-'} "
        f"excluded={ex} fp={fp}"
    )


# -- env parsing for the instrument (pure) ------------------------------------------


def parse_int_list(value: Optional[str]) -> Optional[Tuple[int, ...]]:
    """``"a,b,c"`` -> ints; None on empty or malformed (never raises)."""
    if value is None or not str(value).strip():
        return None
    try:
        out = tuple(int(x) for x in str(value).split(","))
    except ValueError:
        return None
    return out if all(v >= 0 for v in out) else None


def parse_rates(value: Optional[str]) -> Optional[Dict[Tuple[int, int], float]]:
    """``"0>1=13.4,1>0=14.4"`` (rank>rank=rate) -> ``{(0, 1): 13.4, ...}``;
    None on empty or malformed (never raises: the instrument must not break
    the path it observes)."""
    if value is None or not str(value).strip():
        return None
    out: Dict[Tuple[int, int], float] = {}
    try:
        for part in str(value).split(","):
            pair, rate = part.split("=")
            a, b = pair.split(">")
            r = float(rate)
            if r <= 0.0:
                return None
            out[(int(a), int(b))] = r
    except ValueError:
        return None
    return out


# -- the sleep hook (duck-typed scheduler access, no torch, never raises) -----------


def own_anchor_slot_bytes(sched) -> int:
    """Bytes of ONE anchor slot on this rank, all layers (0 = unknowable);
    the same sum as the wake's L15-HOSTBYTES ``anchor_bytes``."""
    try:
        mc = getattr(getattr(getattr(sched, "req_to_token_pool", None),
                             "mamba_pool", None), "mamba_cache", None)
        if mc is None:
            return 0
        ts = [mc.temporal] + list(getattr(mc, "conv", []) or [])
        return sum(int(t[:, 0].numel()) * int(t.element_size()) for t in ts)
    except Exception:  # noqa: BLE001 - estimate only
        return 0


def sleep_entries(sched, tp: int, ratios) -> List[dict]:
    """Candidate entries of this sleep, exactly the seat + parked requests the
    older ``L15-SHADOW`` block prices (span = seqlen - 1, rows follow the DCP
    token vector).  Tree-tip ("served") candidates need a group gather to
    agree on, so the shadow -- which must add no collective -- leaves them
    out (open gap for S2)."""
    from sglang.srt.weg2 import l15_shadow

    entries: List[dict] = []
    rb = getattr(sched, "running_batch", None)
    groups = (("seat", getattr(rb, "reqs", None) or []),
              ("parked", getattr(sched, "weg2_d_parked", None) or []))
    for kind, reqs in groups:
        for req in reqs:
            tok = len(getattr(req, "origin_input_ids", []) or []) + len(
                getattr(req, "output_ids", []) or [])
            span = max(tok - 1, 0)
            entries.append({
                "rid": getattr(req, "rid", None),
                "kind": kind,
                "last_active": 0.0,
                "rows_by_rank": list(l15_shadow.rows_split(span, tp, ratios)),
                "anchor_depth": span,
                "kv_depth": span,
            })
    return entries


def log_sleep_shadow(sched, env: Mapping[str, str], log) -> Optional[str]:
    """The ``L15-POOL-SHADOW`` line of this sleep (log only, no collective,
    changes nothing).  Returns the line (also passed to *log*) or None when
    it could not be built.  Never raises: an instrument must not break the
    path it observes."""
    try:
        from sglang.srt.weg2 import l15_keep_split, l15_shadow

        tp = int(getattr(sched, "tp_size", 0)
                 or getattr(getattr(sched, "server_args", None), "tp_size", 1) or 1)
        try:
            from sglang.srt.distributed.utils import get_cp_token_ratios

            ratios = get_cp_token_ratios()
        except Exception:  # noqa: BLE001 - even split is the fallback
            ratios = None
        mr = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
        cell = l15_shadow.cell_bytes_from(getattr(mr, "token_to_kv_pool", None))
        rgid = getattr(getattr(sched, "server_args", None), "rank_gpu_id", None)
        cards = (list(rgid) if isinstance(rgid, (list, tuple)) and len(rgid) == tp
                 else list(range(tp)))
        caps = l15_shadow.caps_from_env(env, tp, [cell] * tp, cards)
        a_cap = l15_keep_split.anchor_cap(env)
        cands = l15_shadow.candidates_from(sleep_entries(sched, tp, ratios))
        ab = parse_int_list(env.get(POOL_ANCHOR_BYTES_ENV))
        rank = int(getattr(getattr(sched, "ps", None), "tp_rank", 0) or 0)
        # S4 correction of the S1 assumption: the head shares of one anchor are
        # NOT equal over the ranks (27B at [2,1,1]: 37.4/18.7/18.7 MiB; even
        # [1,1,1]: 28.1/23.4/23.4 MiB -- MambaBlobSpec.shard_for_rank). They are
        # read from the spec cut by the head ratio vector the live pool was cut
        # by and confirmed against this rank's own slot bytes.
        from sglang.srt.weg2 import l15_pool_anchor

        ctx, ctx_why = (None, "")
        if ab is None or len(ab) != tp:
            ctx, ctx_why = l15_pool_anchor.resolve_anchor_ctx(sched, env, tp, rank)
        if ab is not None and len(ab) == tp:
            slot_b, src = list(ab), "env"
        elif ctx is not None:
            slot_b, src = list(ctx.bytes_by_rank), ctx.src
        else:
            own = own_anchor_slot_bytes(sched)
            if own > 0:
                slot_b, src = [own] * tp, "own-assumed-equal(UNVERIFIED,wrong-for-tp>1: %s)" % (
                    str(ctx_why)[:120].replace(" ", "_"),)
            else:
                slot_b, src = [1] * tp, "unit-slots(bytes unknown)"
        ps = shadow_compare(
            cands, caps, a_cap, row_bytes=cell,
            anchor_slot_bytes_by_rank=slot_b,
            rates=parse_rates(env.get(POOL_RATES_ENV)), cards=cards,
            anchor_in_kv=l15_pool_anchor.pool_s4_flag(env))
        try:
            from sglang.srt.weg2 import l15_bind  # imports torch: lazy, optional

            epoch = l15_bind.sleep_epoch(sched)
        except Exception:  # noqa: BLE001
            epoch = "?"
        line = shadow_line(epoch, ps, caps, anchor_src=src)
        log("%s", line)
        return line
    except Exception as exc:  # noqa: BLE001 - shadow must never block the flush
        try:
            log("L15-POOL-SHADOW skipped (%s: %s)", type(exc).__name__, exc)
        except Exception:  # noqa: BLE001
            pass
        return None


# -- S3: EVERY rank may overflow; guests by free area x measured rate ---------------
#
# Design sec 3.2-3.3 / 4.1 / 5.2 N1-N3,N5,N7 (KV only; the anchors stay the S4
# work).  S2 let only a rank WITHOUT a home segment park its shards in foreign
# segments; S3 lets any rank do it: what a rank's own hold region cannot take
# (rows beyond its cap) lies as guest rows in the FREE rows of the other
# segments.  The plan is pure and rank-uniform (same replicated lists, same
# rates -> same pieces), the group compares its digest (plan_fingerprint with
# the rates digest) and the manifest v2 carries the guest list in the group
# fingerprint.

POOL_S3_ENV = "SGLANG_WEG2_L15_POOL_S3"


def pool_s3_flag(env: Mapping[str, str]) -> bool:
    """True when ``SGLANG_WEG2_L15_POOL_S3`` is set (1/true/on/yes), whatever
    the pool switch says (the launcher refuses the lonely flag by name)."""
    return str(env.get(POOL_S3_ENV, "") or "").strip().lower() in _ON_VALUES


def pool_s3_on(env: Mapping[str, str]) -> bool:
    """True when the S3 overflow of every rank takes effect: the part switch
    AND the S2 pool switch (S3 never runs without POOL; the launcher refuses
    that combination by name, ``l15_plan.refuse_pool_s3_without_pool``)."""
    return pool_on(env) and pool_s3_flag(env)


S3_MODE = ("POOL(S3: every rank may overflow into foreign segments, guests by free "
           "area x measured link rate, manifest v2 with the guest list)")


def select_hold_pool_s3(
    candidates: Sequence[Candidate],
    cap_rows_by_rank: Sequence[int],
    cap_anchor_slots: int,
) -> HoldSet:
    """``select_hold`` against the SUM of the segments (S3): a request is held
    when the rows of ALL its shards together fit the pool, whichever rank they
    belong to -- a capped rank's overflow is no longer ``no_room`` but a guest.

    Per rank the guest need is ``max(0, rows_r - cap_r)`` and the guest room
    is ``max(0, cap_r - rows_r)``; summed over the ranks the need fits the
    room exactly when ``sum(rows) <= sum(caps)`` (``max(0,x) - max(0,-x) = x``),
    so the admission is that one inequality on the running totals.  Everything
    else is today's policy word for word: anchorless candidates are excluded up
    front, the order is seat > parked > served / younger / rid, the anchor
    count cap gives ``anchor_full``, later smaller candidates may still fit.
    The row sums are the pre-compaction estimate; the exact check on the
    COMPACTED rows is ``l15_retain.plan_round`` (it drops the lowest-priority
    request until :func:`pool_park_plan` places every guest row), BEFORE
    anything moves.  No safety factor, no extra ceiling: the caps are the
    budget."""
    candidates = list(candidates)
    excluded: Dict[str, Tuple[str, str]] = {
        c.rid: (c.rid, REASON_ANCHORLESS)
        for c in candidates
        if c.anchor_depth != c.kv_depth
    }
    ordered = sorted(
        (c for c in candidates if c.anchor_depth == c.kv_depth), key=_order_key
    )
    caps = [max(0, int(c)) for c in cap_rows_by_rank]
    total_cap = sum(caps)
    used = 0
    admitted = []
    for c in ordered:
        need = sum(int(v) for v in c.rows_by_rank)
        if used + need > total_cap:
            excluded[c.rid] = (c.rid, REASON_POOL_FULL)
            continue
        if len(admitted) >= cap_anchor_slots:
            excluded[c.rid] = (c.rid, REASON_ANCHOR_FULL)
            continue
        admitted.append(c)
        used += need
    rows_by_rank = tuple(sum(c.rows_by_rank[r] for c in admitted) for r in range(len(caps)))
    return HoldSet(
        rids=tuple(c.rid for c in admitted),
        rows_by_rank=rows_by_rank,
        anchors=len(admitted),
        excluded=tuple(excluded.values()),
    )


def pool_park_plan(
    keep_rows: Sequence[int],
    caps: Sequence[int],
    rates: Optional[Mapping[Tuple[int, int], float]] = None,
    default_rate: float = 1.0,
) -> Tuple[List[ParkPiece], Optional[str]]:
    """``(pieces, None)`` or ``([], reason)``: the S3 guest placement.

    ``keep_rows[r]``: rank r's COMPACTED keep rows of this hold (the
    manifest's rows_by_rank); ``caps[r]``: its hold region in rows (0 = no home
    segment).  Rank r keeps ``home_r = min(keep_r, cap_r)`` rows in place
    (rows ``[0, home_r)``, cost 0); the overflow rows ``[home_r, keep_r)`` --
    all of them for a cap-0 rank -- go to the FREE rows ``[home_h, cap_h)`` of
    the other segments.  A rank that overflows has no free rows itself, so a
    host never overflows and the pieces can never chain.

    Host choice per piece (Q3 of the design, user decision 04.10.): NOT the
    slowest card (strictly lowest mean inbound rate of the measured matrix --
    only as the last overflow), then the larger free area x directed rate
    (``rates[(src, host)]``), then the lower rank (deterministic).  Without
    rates every rank is equal: largest free area first, ties to the lower
    rank -- exactly :func:`l15_park.park_plan` for a hold in which only cap-0
    ranks overflow (pinned by a test).  A shard may be split over several
    hosts.  Refused by name when the free rows cannot take the whole
    overflow (a partial park would need an L2 refill the capped rank does
    not have).  Keyed by cap and rank, never by card name or ordinal."""
    R = len(keep_rows)
    if len(caps) != R:
        return [], "keep_rows for %d ranks, caps for %d" % (R, len(caps))
    keep = [max(0, int(x)) for x in keep_rows]
    cap = [max(0, int(c)) for c in caps]
    home = [min(keep[r], cap[r]) for r in range(R)]
    free = {r: cap[r] - home[r] for r in range(R) if cap[r] - home[r] > 0}
    cursor = {r: home[r] for r in free}
    rt = _Rates(rates, default_rate, [r for r in range(R) if cap[r] > 0])
    pieces: List[ParkPiece] = []
    for src in range(R):
        need, row = keep[src] - home[src], home[src]
        while need > 0:
            cands = [h for h, f in free.items() if f > 0 and h != src]
            if not cands:
                have = sum(max(0, v) for v in free.values())
                return [], ("rank %d needs %d more guest rows, the other segments "
                            "have %d free" % (src, need, have))
            h = min(cands, key=lambda x: _host_key(rt, src, x, free[x]))
            n = min(need, free[h])
            pieces.append(ParkPiece(src, h, row, cursor[h], n))
            cursor[h] += n
            free[h] -= n
            need -= n
            row += n
    return pieces, None


def guest_tuples(pieces: Sequence) -> Tuple[Tuple[int, int, int, int, int], ...]:
    """The manifest form of a placement: ``(src, dst, src_row, dst_row, rows)``."""
    return tuple((int(p.src), int(p.dst), int(p.src_row), int(p.dst_row), int(p.rows))
                 for p in pieces)


def pieces_of(guests: Sequence[Sequence[int]]) -> List[ParkPiece]:
    return [ParkPiece(*(int(x) for x in g)) for g in guests]


def quantize_rates(rates: Optional[Mapping[Tuple[int, int], float]]
                   ) -> Optional[Dict[Tuple[int, int], float]]:
    """Three digits like ``barlink_matrix._quant`` (about 1 MB/s, well below
    the measurement noise): a decision must not flip on the 12th digit, and
    the digest every rank compares must not either.  Non-finite and
    non-positive values are dropped; nothing left = None."""
    import math

    if not rates:
        return None
    out = {}
    for (a, b), v in rates.items():
        v = float(v)
        if math.isfinite(v) and v > 0.0 and round(v, 3) > 0.0:
            out[(int(a), int(b))] = round(v, 3)
    return dict(sorted(out.items())) or None


def rates_digest(rates: Optional[Mapping[Tuple[int, int], float]]) -> str:
    """Short digest of the rates a placement was scored with (``-`` = none)."""
    import hashlib

    q = quantize_rates(rates)
    if not q:
        return "-"
    blob = repr(tuple((a, b, v) for (a, b), v in sorted(q.items())))
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def rates_from_capacity(capacity, world: int, gi: int = -1
                        ) -> Optional[Dict[Tuple[int, int], float]]:
    """``{(src, dst): GB/s}`` for every directed pair of a measured matrix.
    ``capacity(src, dst, gi)`` is ``barlink_matrix.Measurement.capacity``
    (a measured edge wins, else min(outbound, inbound) -- its own upper-bound
    rule); ``gi`` = -1 is the LARGEST measured size (the saturated rate, the
    one a bulk guest transfer sees).  None when any pair is unavailable."""
    out = {}
    try:
        for a in range(world):
            for b in range(world):
                if a != b:
                    out[(a, b)] = float(capacity(a, b, gi))
    except Exception:  # noqa: BLE001 -- no rates is a named state, not a crash
        return None
    return quantize_rates(out)


def load_barlink_rates(env: Mapping[str, str], tp: int
                       ) -> Tuple[Optional[Dict[Tuple[int, int], float]], str]:
    """The measured barlink matrix of this rig as pair rates: the planner's
    cache file (``barlink_matrix.read`` path of ``load_config``: the config's
    ``measure.cache`` or the default) carries the last startup measurement.
    Accepted only when it is a measurement of ``tp`` ranks; the rate of a pair
    is read from the measurement, NEVER derived from a card name.  Returns
    ``(rates, source)``; ``(None, "none(<why>)")`` = uniform placement.  Never
    raises."""
    try:
        import json

        from sglang.srt.distributed.device_communicators import barlink_matrix as bm

        path = bm.load_config(env).collective.measure.cache or bm._default_cache()
        with open(path) as fh:
            d = json.load(fh)
        m = bm.Measurement.from_dict(d["measurement"])
    except Exception as exc:  # noqa: BLE001
        return None, "none(barlink matrix unreadable: %s)" % type(exc).__name__
    if int(m.world) != int(tp):
        return None, "none(barlink matrix is of %d ranks, the pool has %d)" % (
            int(m.world), int(tp))
    gi = len(m.sizes) - 1
    rates = rates_from_capacity(m.capacity, int(tp), gi)
    if rates is None:
        return None, "none(barlink matrix has no complete pair rates)"
    return rates, "barlink-matrix(ranks=%d,sensor=%s,size_kib=%d)" % (
        int(tp), m.sensor, int(m.sizes[gi]) >> 10)


def resolve_rates(env: Mapping[str, str], tp: int, loader=None
                  ) -> Tuple[Optional[Dict[Tuple[int, int], float]], str]:
    """The pair rates of the placement and where they came from:

    1. ``SGLANG_WEG2_L15_POOL_RATES`` (``"0>1=13.4,1>0=14.4"``, the explicit
       operator input; malformed = named, no silent fallback),
    2. the measured barlink matrix (:func:`load_barlink_rates`),
    3. none = uniform (no card is the slowest).

    Every rank resolves them itself from the same sources; the digest in the
    plan fingerprint turns a round off everywhere when two ranks disagree."""
    raw = env.get(POOL_RATES_ENV)
    if raw is not None and str(raw).strip():
        parsed = parse_rates(raw)
        if parsed is None:
            return None, "none(%s malformed)" % POOL_RATES_ENV
        return quantize_rates(parsed), "env"
    return (loader or load_barlink_rates)(env, tp)


def capped_guest_ranks(guests: Optional[Sequence[Sequence[int]]],
                       caps: Optional[Sequence[int]]) -> Tuple[int, ...]:
    """Ranks WITH a home segment that have guest pieces (S3 overflow of a
    capped rank).  Such a rank has no L2 refill path for them, so a failed
    park-back of the group must drop the whole hold."""
    if not guests or caps is None:
        return ()
    return tuple(sorted({int(g[0]) for g in guests
                         if 0 <= int(g[0]) < len(caps) and int(caps[int(g[0])]) > 0}))


def hosted_end(guests: Optional[Sequence[Sequence[int]]], rank: int) -> int:
    """First row after the last guest row ``rank`` hosts (0 = hosts none)."""
    return max([int(g[3]) + int(g[4]) for g in (guests or ()) if int(g[1]) == int(rank)]
               + [0])


def wake_keep_rows(rows_by_rank: Sequence[int], rank: int,
                   guests: Optional[Sequence[Sequence[int]]],
                   anchor_guests: Optional[Sequence[Sequence[int]]] = None) -> int:
    """The rows the wake's zero scrub must leave alone on ``rank``: its compact
    prefix and, for a v2 manifest, the guest rows it hosts (they come back to
    their owners only at the park-back, after the restore).  S4: the rows that
    host anchor byte guests count too (``anchor_guests``, None = no S4 round:
    exactly the S3 value)."""
    keep = int(rows_by_rank[rank]) if 0 <= rank < len(rows_by_rank) else 0
    if guests is None:
        return keep
    end = hosted_end(guests, rank)
    if anchor_guests:
        end = max(end, max([int(g[4]) + int(g[5]) for g in anchor_guests
                            if int(g[1]) == int(rank)] + [0]))
    return max(keep, end)


def guest_row_ranges(guests: Optional[Sequence[Sequence[int]]], rank: int
                     ) -> List[Tuple[int, int]]:
    """``[(lo, hi))`` compact-row ranges of the guest rows ``rank`` OWNS."""
    return sorted((int(g[2]), int(g[2]) + int(g[4]))
                  for g in (guests or ()) if int(g[0]) == int(rank))


def _even(items: Sequence, k: int) -> List:
    n = len(items)
    if k <= 0 or n == 0:
        return []
    if n <= k:
        return list(items)
    return [items[(j * n) // k] for j in range(k)]


def stratified_check_plan(plan: Sequence[tuple], ranges: Sequence[Tuple[int, int]],
                          k: int, guest_share: float = 0.5) -> List[tuple]:
    """The L15-CHECK sample of a rank that owns guest rows: ``plan`` entries
    are ``(rid, compact_row, l2_slot, l2_gen)``; up to ``k`` of them, at least
    ``guest_share`` of the sample taken from rows inside ``ranges`` (the guest
    rows, which travelled through a foreign segment) when there are any, the
    rest evenly over the home rows; each part evenly spaced by row.  The
    checker then reads exactly these rows (it samples evenly again and a
    sample of at most ``k`` entries comes back whole)."""
    entries = sorted(plan, key=lambda e: int(e[1]))
    # one pass: the old ``e not in in_guest`` was a list scan per entry --
    # O(n * guests), 36 s at 170k rows x 75k guest rows inside the wake RPC
    # (j4, 05.10. 02:06Z, L15-WAKE-TIMING check_decide_ms=36361).
    flags = [any(lo <= int(e[1]) < hi for lo, hi in ranges) for e in entries]
    in_guest = [e for e, f in zip(entries, flags) if f]
    if not in_guest:
        return _even(entries, k)
    home = [e for e, f in zip(entries, flags) if not f]
    kg = min(len(in_guest), max(1, int(k * guest_share)))
    kh = min(len(home), max(0, k - kg))
    kg = min(len(in_guest), k - kh)
    return sorted(_even(in_guest, kg) + _even(home, kh), key=lambda e: int(e[1]))


def plan_line_s3(epoch, hs: HoldSet, caps: Sequence[int], keep_rows: Sequence[int],
                 pieces: Sequence, fp: str, n_candidates: int, rates_src: str,
                 rates_fp: str) -> str:
    """The S3 ``L15-POOL-PLAN``: the S2 line plus who overflows, the guest
    pairs with their share of the pool's guests per host, and the rate source
    the placement was scored with."""
    base = plan_line(epoch, hs, caps, keep_rows, pieces, fp, n_candidates)
    over = ",".join(
        "%d:%d" % (r, max(0, int(keep_rows[r]) - max(0, int(caps[r]))))
        for r in range(len(keep_rows)) if int(keep_rows[r]) > max(0, int(caps[r])))
    hosts: Dict[int, int] = {}
    for p in pieces:
        hosts[int(p.dst)] = hosts.get(int(p.dst), 0) + int(p.rows)
    return (base + " s3=1 overflow_rows=%s guest_on_host=%s rates_src=%s rates_fp=%s"
            % (over or "-", ",".join("%d:%d" % kv for kv in sorted(hosts.items())) or "-",
               rates_src, rates_fp))
