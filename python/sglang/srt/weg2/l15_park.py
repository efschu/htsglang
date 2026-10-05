"""L15-16: the cap-0 rank parks its held KV rows on a capped rank's card.

User law 02.10.: with L1.5 on, the flip must be faster in BOTH directions and
move fewer host bytes. A rank with cap 0 (its card has no room in the P
layout) keeps nothing through the P phase; today it refills its owned held
rows from L2 at the wake (H2D, ``L15-HOSTBYTES h2d_refill``). Here it parks
them instead -- card to card over the BAR1 lanes -- into the FREE part of a
capped rank's hold region (compact rows ``[keep_rows_r, cap_r)``, kept
physically by the split hold extents), and takes them back at the wake.

This module is the pure plan (no torch, no transport):

* :func:`park_plan` -- which cap-0 rank's compact rows ``[0, keep_rows)`` go
  to which capped rank at which row, largest free region first, split over
  several capped ranks when one does not suffice; refused by name when the
  capped ranks' free rows cannot take them all (the L2 refill serves then).
* keyed by cap, never by card name or ordinal (user order 07:05Z).

The anchors (GDN head shares, a few MiB each) stay on the L2 refill; the
deposit region (L15-14) shares the same free rows and is off while a park
is planned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ParkPiece:
    src: int        # cap-0 rank whose compact rows are parked
    dst: int        # capped rank holding them through the P phase
    src_row: int    # first compact row on src
    dst_row: int    # first row on dst (inside its hold region, after its own)
    rows: int


def park_plan(keep_rows: Sequence[int], caps: Sequence[int]
              ) -> Tuple[List[ParkPiece], Optional[str]]:
    """``(pieces, None)`` or ``([], reason)``.

    ``keep_rows[r]``: rank r's compacted keep rows of this hold (the
    manifest's rows_by_rank); ``caps[r]``: its hold region in rows (0 = not
    held here). Every cap-0 rank with rows parks ALL of them, or the plan is
    refused (a partial park would need the L2 refill anyway)."""
    R = len(keep_rows)
    if len(caps) != R:
        return [], "keep_rows for %d ranks, caps for %d" % (R, len(caps))
    free = {r: int(caps[r]) - int(keep_rows[r]) for r in range(R) if int(caps[r]) > 0}
    if any(v < 0 for v in free.values()):
        bad = [r for r, v in free.items() if v < 0]
        return [], "capped rank(s) %s keep more rows than their cap" % bad
    cursor = {r: int(keep_rows[r]) for r in free}
    pieces: List[ParkPiece] = []
    for src in range(R):
        if int(caps[src]) > 0 or int(keep_rows[src]) <= 0:
            continue
        need, row = int(keep_rows[src]), 0
        while need > 0:
            # largest free region first; ties to the lower rank (deterministic)
            cands = sorted((r for r in free if free[r] > 0),
                           key=lambda r: (-free[r], r))
            if not cands:
                have = sum(max(0, v) for v in free.values())
                return [], ("cap-0 rank %d needs %d more rows to park, capped "
                            "ranks have %d free" % (src, need, have))
            dst = cands[0]
            n = min(need, free[dst])
            pieces.append(ParkPiece(src, dst, row, cursor[dst], n))
            cursor[dst] += n
            free[dst] -= n
            need -= n
            row += n
    return pieces, None


def parked_rows_on(pieces: Sequence[ParkPiece], dst: int) -> int:
    return sum(p.rows for p in pieces if p.dst == dst)


def park_bytes(pieces: Sequence[ParkPiece], row_bytes_all_layers: int) -> int:
    """Bytes the park moves per direction (all layers, K and V)."""
    return sum(p.rows for p in pieces) * int(row_bytes_all_layers)


# -- the transport (L15-16b): one uneven all_to_all per (piece, buffer) -----

PARK_ENV = "SGLANG_WEG2_L15_PARK"


def park_on(env) -> bool:
    """The park transport runs under its own opt-in OR under the S2 pooled
    hold (``SGLANG_WEG2_L15_POOL``: the pool's guest shards travel by it)."""
    if str(env.get(PARK_ENV, "0")).strip() == "1":
        return True
    from sglang.srt.weg2 import l15_pool

    return l15_pool.pool_on(env)


def _rows2d(buf):
    import torch

    t = buf if buf.is_contiguous() else None
    if t is None:
        raise ValueError("park: KV buffer is not contiguous")
    return t.view(-1).view(torch.uint8).view(int(buf.shape[0]), -1)


CHUNK_ENV = "SGLANG_WEG2_L15_PARK_CHUNK_MIB"


def chunk_rows(row_bytes: int, env=None) -> int:
    """Rows per collective: the barlink a2a takes its BAR1 path only for
    blocks its slot ring carries -- a larger block falls back to the pinned
    host + gloo layer, i.e. exactly the host bounce the park exists to avoid.
    Default 16 MiB per block (the same figure on every rank: env + geometry)."""
    import os as _os

    env = _os.environ if env is None else env
    try:
        mib = max(1, int(env.get(CHUNK_ENV, "16")))
    except (TypeError, ValueError):
        mib = 16
    return max(1, (mib << 20) // max(1, int(row_bytes)))


def run_park(direction: str, pieces: Sequence[ParkPiece], rank: int,
             world: int, buffers: Sequence, a2a, env=None,
             uniform: bool = False) -> int:
    """Move every piece's rows of every buffer: ``direction`` "out" (sleep:
    src -> dst) or "back" (wake: dst -> src). EVERY rank of the group calls
    this with the same pieces and buffers in the same order (one collective
    per (piece, buffer)); ``a2a(output, input, out_splits, in_splits)`` is the
    group's uneven all_to_all. Returns the bytes this rank sent.

    ``uniform`` (S2 pooled hold only): every chunk is announced to the a2a as
    ``a2a(..., rows=n)`` -- ``n`` is the one figure all ranks agree on (the
    source sends it, the destination receives it, a third rank sees nothing),
    so the transport choice and the slot check in barlink's a2a do not hinge on
    the rank-local send sum (0 on all but the source rank: 1780 wedge)."""
    sent = 0
    for p in pieces:
        frm, to = (p.src, p.dst) if direction == "out" else (p.dst, p.src)
        frm_row = p.src_row if direction == "out" else p.dst_row
        to_row = p.dst_row if direction == "out" else p.src_row
        for buf in buffers:
            b = _rows2d(buf)
            step = chunk_rows(int(b.shape[1]), env)
            for c0 in range(0, p.rows, step):
                n = min(step, p.rows - c0)
                in_splits = [0] * world
                out_splits = [0] * world
                inp = b[0:0]
                out = b[0:0]
                if rank == frm:
                    inp = b[frm_row + c0:frm_row + c0 + n]
                    in_splits[to] = n
                    sent += int(inp.numel())
                if rank == to:
                    out = b[to_row + c0:to_row + c0 + n]
                    out_splits[frm] = n
                if uniform and rank != frm and rank != to:
                    # barlink-BAR1's a2a refuses ``in`` and ``out`` with the
                    # same data_ptr ("in and out must not be the same"), and an
                    # EMPTY tensor's data_ptr is 0: the third rank of a piece
                    # (neither source nor destination) handed in two empties,
                    # raised in round 1 and left the other two waiting 60 s
                    # for it (j2, 05.10. 01:18Z). ``out`` may be larger than
                    # the receive counts (barlink checks only sum(out_rows) <=
                    # rows), ``in`` must match its splits exactly: give the
                    # bystander one row of ``out`` -- every split is 0, so no
                    # byte is read or written.
                    out = b[0:1]
                if uniform:
                    a2a(out, inp, out_splits, in_splits, rows=n)
                else:
                    a2a(out, inp, out_splits, in_splits)
    return sent


def bounds_refusal(pieces: Sequence[ParkPiece], rank: int, buffers) -> Optional[str]:
    """Every row this rank reads or writes lies inside its buffers, and the
    buffers share one row width (checked BEFORE the agreement, so the
    collectives never start on a plan one rank cannot run)."""
    if not buffers:
        return "no KV buffers"
    rows = min(int(b.shape[0]) for b in buffers)
    for p in pieces:
        for r, row in ((p.src, p.src_row), (p.dst, p.dst_row)):
            if r == rank and row + p.rows > rows:
                return "rows [%d,%d) beyond the %d-row buffers" % (row, row + p.rows, rows)
    return None


def sidecar_path(rank: int, env) -> str:
    import os

    base = str(env.get("SGLANG_WEG2_L15_PARK_DIR", "/tmp"))
    return os.path.join(base, "weg2_l15_park.D.%d.json" % int(rank))


def write_sidecar(rank: int, env, epoch: int, pieces: Sequence[ParkPiece],
                  sums: Optional[dict] = None,
                  anchor_guests: Optional[Sequence[Sequence[int]]] = None,
                  anchor_sums: Optional[dict] = None) -> None:
    """``sums`` (S2 pooled hold only): the source-side sample checksums of the
    guest rows, kept for the wake's round-trip check; absent otherwise, so the
    record of the per-card path is unchanged byte for byte."""
    import json
    import os

    path = sidecar_path(rank, env)
    tmp = path + ".tmp"
    rec = {"epoch": int(epoch),
           "pieces": [[p.src, p.dst, p.src_row, p.dst_row, p.rows]
                      for p in pieces]}
    if sums is not None:
        rec["sums"] = sums
    if anchor_guests is not None:
        # S4 only: the anchor byte guests and the source checksum of the shares
        # this rank owns; absent otherwise, the S3 record byte for byte
        rec["anchor_guests"] = [[int(x) for x in g] for g in anchor_guests]
        rec["anchor_sums"] = anchor_sums or {}
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, path)


def _read_sidecar(rank: int, env) -> Optional[dict]:
    """Read AND remove this rank's park record (one sleep-wake pair)."""
    import json
    import os

    path = sidecar_path(rank, env)
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (FileNotFoundError, ValueError):
        return None
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    return d


def take_sidecar(rank: int, env) -> Optional[Tuple[int, List[ParkPiece]]]:
    """Read AND remove this rank's park record (one sleep-wake pair)."""
    d = _read_sidecar(rank, env)
    if d is None:
        return None
    return int(d.get("epoch", -1)), [ParkPiece(*map(int, x)) for x in d.get("pieces", ())]


# -- S2 pooled hold: source checksum of the guest rows, plan agreement -------------

SAMPLE_ROWS = 16


def _sample_idx(row0: int, rows: int, k: int = SAMPLE_ROWS) -> List[int]:
    """Up to ``k`` rows of ``[row0, row0 + rows)``: first, last and evenly
    spaced in between (deterministic, the same on every call)."""
    if rows <= 0:
        return []
    if rows <= k:
        return [row0 + i for i in range(rows)]
    step = (rows - 1) / float(k - 1)
    return sorted({row0 + int(round(i * step)) for i in range(k)})


def guest_sums(pieces: Sequence[ParkPiece], rank: int, buffers: Sequence) -> dict:
    """Position-weighted checksum of sampled rows of every piece THIS rank
    parks (``p.src == rank``), per buffer: ``{"<piece index>": [sum, ...]}``.
    Taken at the source before the send at the sleep and again after the
    rows came back at the wake, so a guest row that arrived wrong (or was
    clobbered in the foreign segment) is caught independently of L2."""
    import torch

    out = {}
    for i, p in enumerate(pieces):
        if p.src != rank:
            continue
        idx = _sample_idx(p.src_row, p.rows)
        per = []
        for buf in buffers:
            b = _rows2d(buf)
            it = torch.tensor(idx, dtype=torch.int64, device=b.device)
            x = b.index_select(0, it).to(torch.int64)
            w_c = torch.arange(1, int(x.shape[1]) + 1, dtype=torch.int64, device=b.device)
            w_r = torch.arange(1, int(x.shape[0]) + 1, dtype=torch.int64, device=b.device)
            per.append(int((x * w_c[None, :] * w_r[:, None]).sum().item()))
        out[str(i)] = per
    return out


def compare_sums(before: dict, after: dict) -> Tuple[int, int]:
    """``(ok, bad)`` over (piece, buffer) samples."""
    ok = bad = 0
    for k, per in (before or {}).items():
        now = (after or {}).get(k)
        for j, v in enumerate(per):
            if now is not None and j < len(now) and int(now[j]) == int(v):
                ok += 1
            else:
                bad += 1
    return ok, bad


def _guest_key(pieces: Sequence[ParkPiece]):
    return tuple((int(p.src), int(p.dst), int(p.src_row), int(p.dst_row), int(p.rows))
                 for p in pieces)


def agree_plan(ok: bool, fp: Optional[str], gather) -> Optional[str]:
    """Pooled park: None when every rank is ok AND holds the SAME plan
    (equal digests); else the reason. One gather. A rank that would start the
    collectives on a different plan than its peers hangs the group, so the
    plan itself is part of the vote."""
    votes = gather((bool(ok), fp))
    if not all(bool(v[0]) for v in votes):
        return "a peer refused"
    if len({v[1] for v in votes}) > 1:
        return "park plan diverged across ranks"
    return None


def agree(ok: bool, gather) -> bool:
    """True iff every rank of the group says ok (one host gather)."""
    votes = gather(bool(ok))
    return all(bool(v) for v in votes)


def s3_pieces_of_manifest(m, caps: Sequence[int], env, world: int, rates=None):
    """``(pieces, None, fp)`` or ``([], reason, None)``: the S3 park pieces are
    the manifest v2 guest list the retain published -- never a re-plan -- and
    they must be exactly what the same pure plan gives from the manifest's own
    rows, this rank's caps and the rates it resolves (a rank whose caps or
    rates change the placement turns the round off here, before any collective
    starts); the digest the group compares is the retain's ``plan_fingerprint``."""
    from sglang.srt.weg2 import l15_pool

    if m.guests is None:
        return [], "manifest is v1 under SGLANG_WEG2_L15_POOL_S3 (no guest list)", None
    if m.caps is None or tuple(int(c) for c in m.caps) != tuple(int(c) for c in caps):
        return [], "manifest caps %s differ from this rank's caps %s" % (
            list(m.caps or ()), [int(c) for c in caps]), None
    if rates is None:
        rates, _src = l15_pool.resolve_rates(env, int(world))
    rates = l15_pool.quantize_rates(rates)
    want, why = l15_pool.pool_park_plan(list(m.rows_by_rank), list(caps), rates)
    if why is not None:
        return [], why, None
    have = l15_pool.pieces_of(m.guests)
    if l15_pool.guest_tuples(have) != l15_pool.guest_tuples(want):
        return [], "manifest guest list differs from the plan of this rank's rates", None
    fp = l15_pool.plan_fingerprint(
        [sp.rid for sp in m.spans], list(m.rows_by_rank), caps, have)
    return have, None, fp


def s4_plan_of_manifest(m, caps: Sequence[int], env, world: int, rank: int,
                        kv_views, m_views, rates=None):
    """``(anchor_guests, None, fp)`` or ``([], reason, None)``: the S4 anchor
    guests are the manifest's list the retain published -- never a re-plan --
    and must be exactly what the same pure plan gives from the manifest's own
    rows, caps, pricing and anchor count (plus the KV guests, which stay the S3
    plan); this rank's own slot bytes and the KV row bytes must be the priced
    ones.  The digest the group compares carries all of it
    (``l15_pool_anchor.plan_fingerprint_s4``)."""
    from sglang.srt.weg2 import l15_pool, l15_pool_anchor

    if m.anchor_guests is None or m.anchor_bytes is None:
        return [], "manifest has no anchor guests under SGLANG_WEG2_L15_POOL_S4", None
    if m.guests is None:
        return [], "S4 manifest without the S3 guest list", None
    if l15_pool_anchor.pool_s4b_on(env) != (m.anchor_cap is not None):
        # S4b is rank-uniform config: the switch and the manifest must agree on
        # whether the anchor count is dynamic (named, before any collective)
        return [], "S4b switch %s but the manifest anchor cap is %s" % (
            "on" if l15_pool_anchor.pool_s4b_on(env) else "off", m.anchor_cap), None
    ab = tuple(int(x) for x in m.anchor_bytes)
    rb = int(m.anchor_row_bytes)
    if len(ab) != int(world):
        return [], "anchor bytes for %d ranks, group of %d" % (len(ab), world), None
    own = sum(int(v.shape[1]) for v in m_views)
    if own != ab[rank]:
        return [], "this rank's slot is %d bytes, the plan priced %d" % (own, ab[rank]), None
    if l15_pool_anchor.kv_row_bytes(kv_views) != rb:
        return [], "KV row is %d bytes, the plan priced %d" % (
            l15_pool_anchor.kv_row_bytes(kv_views), rb), None
    if rates is None:
        rates, _src = l15_pool.resolve_rates(env, int(world))
    rates = l15_pool.quantize_rates(rates)
    n_anchors = int(m.anchor_slots) - 1
    kv_want, a_want, why, _code = l15_pool_anchor.pool_park_plan_s4(
        list(m.rows_by_rank), list(caps), ab, rb, n_anchors, rates,
        anchor_cap=m.anchor_cap)
    if why is not None:
        return [], why, None
    if l15_pool.guest_tuples(kv_want) != l15_pool.guest_tuples(
            l15_pool.pieces_of(m.guests)):
        return [], "manifest KV guest list differs from the plan of this rank's rates", None
    have = l15_pool_anchor.anchor_guest_tuples(m.anchor_guests)
    if have != l15_pool_anchor.anchor_guest_tuples(a_want):
        return [], "manifest anchor guest list differs from the plan of this rank's rates", None
    fp = l15_pool_anchor.plan_fingerprint_s4(
        [sp.rid for sp in m.spans], list(m.rows_by_rank), caps,
        l15_pool.pieces_of(m.guests), have, ab, rb, n_anchors, anchor_cap=m.anchor_cap)
    return list(have), None, fp


# -- scheduler entries --------------------------------------------------------

def _group_io(sched):
    """(rank, world, gather(obj)->list, a2a(out, inp, osp, isp, rows=None)) of
    D's TP group.  ``rows`` (pooled park, rank-uniform rows of this chunk) ->
    ``largest_block_rows=rows`` (no gloo group_max) and ``select_bytes=
    rows*row_bytes`` (every rank picks the transport from the same figure)."""
    import torch

    from sglang.srt.distributed import get_tp_group

    g = get_tp_group()
    world = int(g.world_size)
    rank = int(g.rank_in_group)

    def gather(obj):
        if world <= 1:
            return [obj]
        out = [None] * world
        torch.distributed.all_gather_object(out, obj, group=g.cpu_group)
        return out

    def a2a(out, inp, osp, isp, rows=None):
        if rows is None:
            g.all_to_all_single_v(out, inp, osp, isp)
            return
        row_bytes = int(inp.element_size())
        for d in inp.shape[1:]:
            row_bytes *= int(d)
        g.all_to_all_single_v(out, inp, osp, isp, largest_block_rows=int(rows),
                              select_bytes=int(rows) * row_bytes)

    return rank, world, gather, a2a


def _kv_buffers(sched):
    from sglang.srt.weg2 import l15_shadow

    mr = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
    pool = l15_shadow.kv_pool_of(getattr(mr, "token_to_kv_pool", None))
    return list(getattr(pool, "k_buffer", None) or []) + list(
        getattr(pool, "v_buffer", None) or []), pool


def _caps(sched, pool, tp, env):
    from sglang.srt.weg2 import l15_shadow

    rgid = getattr(getattr(sched, "server_args", None), "rank_gpu_id", None)
    cards = (list(rgid) if isinstance(rgid, (list, tuple)) and len(rgid) == tp
             else list(range(tp)))
    return l15_shadow.caps_from_env(env, tp, [l15_shadow.cell_bytes_from(pool)] * tp,
                                    cards)


def _pool_line(kind: str, epoch, rank, nbytes, ms, rounds, reason="-") -> str:
    """``L15-POOL-OUT`` / ``L15-POOL-BACK`` (S2 pooled hold, one per rank and
    direction; ``path=a2a`` = the D group's all_to_all -- barlink BAR1 or its
    host fallback, decided inside the collective; ``path=none`` = nothing
    moved and ``reason`` names why, the L2 refill serves)."""
    gbps = (nbytes / (ms * 1e6)) if ms > 0 and nbytes > 0 else 0.0
    return ("L15-POOL-%s epoch=%s rank=%s bytes=%d ms=%.0f GBps=%.2f rounds=%d "
            "path=%s reason=%s" % (kind, epoch, rank, nbytes, ms, gbps, rounds,
                                   "a2a" if nbytes > 0 or rounds > 0 else "none",
                                   str(reason).replace(" ", "_")))


def pool_check_line(epoch, rank: int, ok: int, bad: int, pieces: Sequence[ParkPiece],
                    s3: bool = False, anchor: Optional[Tuple[int, int, int]] = None) -> str:
    """``L15-POOL-CHECK``: the guest-row sample of one rank at the wake.  S3
    names what was sampled: pieces and rows this rank OWNS as guests and the
    hosts they lay on (``guest_rows=0`` = nothing of this rank travelled)."""
    base = ("L15-POOL-CHECK epoch=%s rank=%d ok=%d bad=%d guest_ok=%d guest_bad=%d"
            % (epoch, rank, ok, bad, ok, bad))
    if not s3:
        return base
    own = [p for p in pieces if int(p.src) == int(rank)]
    line = base + " s3=1 guest_pieces=%d guest_rows=%d hosts=%s" % (
        len(own), sum(int(p.rows) for p in own),
        ",".join(str(h) for h in sorted({int(p.dst) for p in own})) or "-")
    if anchor is not None:
        # S4: the source checksum blocks of the anchor shares this rank owns,
        # compared after they came back (anchor_pieces = pieces it owns)
        line += " s4=1 anchor_ok=%d anchor_bad=%d anchor_pieces=%d" % (
            int(anchor[0]), int(anchor[1]), int(anchor[2]))
    return line


def _counting(a2a):
    box = [0]

    def run(out, inp, osp, isp, rows=None):
        box[0] += 1
        if rows is None:
            return a2a(out, inp, osp, isp)
        return a2a(out, inp, osp, isp, rows=rows)

    return run, box


def park_at_release(sched, env, log) -> Optional[int]:
    """D sleep (release RPC, every D rank, before the kv pause): park the
    cap-0 ranks' held rows on capped ranks. Returns bytes sent by this rank,
    or None when no park ran (named in the line; the wake refills from L2).
    Never raises: a failure after the agreement is logged and leaves no
    sidecar, so the wake refills from L2 on every rank.

    S2 pooled hold (``SGLANG_WEG2_L15_POOL``): the same transport plus (i) the
    vote carries the digest of the plan, so ranks on different plans never
    start the collectives, (ii) the source checksum of sampled guest rows goes
    into the sidecar for the wake's round-trip check, (iii) the
    ``L15-POOL-OUT`` line. Off = the per-card behaviour, byte for byte."""
    import time

    import torch

    from sglang.srt.weg2 import l15_manifest, l15_pool

    t0 = time.perf_counter()
    pooled = l15_pool.pool_on(env)
    s3 = l15_pool.pool_s3_on(env)
    from sglang.srt.weg2 import l15_pool_anchor

    s4 = l15_pool_anchor.pool_s4_on(env)
    try:
        rank, world, gather, a2a = _group_io(sched)
    except Exception as exc:  # noqa: BLE001 -- no group, no park
        log("L15-PARK at=sleep skipped (no group: %s)" % (exc,))
        return None
    pieces, why, epoch, fp = [], None, -1, None
    a_guests, kv_v, m_v = [], [], []
    try:
        m = l15_manifest.read(l15_manifest.manifest_path("D", rank, env))
        bufs, pool = _kv_buffers(sched)
        if m is None:
            why = "no manifest on this rank"
        elif not bufs:
            why = "no KV buffers"
        else:
            epoch = int(m.epoch)
            caps = _caps(sched, pool, world, env)
            if s3:
                pieces, why, fp = s3_pieces_of_manifest(m, caps, env, world)
            else:
                pieces, why = park_plan(list(m.rows_by_rank), caps)
            if s4 and why is None:
                kv_v = [l15_pool_anchor._rows2d(b) for b in bufs]
                m_v = l15_pool_anchor.mamba_views(sched)
                a_guests, why, fp = s4_plan_of_manifest(
                    m, caps, env, world, rank, kv_v, m_v)
                if why is None:
                    why = l15_pool_anchor.anchor_bounds_refusal(
                        a_guests, rank, kv_v, m_v, int(m.anchor_row_bytes),
                        int(m.anchor_bytes[rank]))
            if why is None and not pieces and not a_guests:
                why = "nothing to park"
            if why is None and pieces:
                why = bounds_refusal(pieces, rank, bufs)
            if why is None and pooled and fp is None:
                fp = l15_pool.plan_fingerprint(
                    [sp.rid for sp in m.spans], list(m.rows_by_rank), caps, pieces)
    except Exception as exc:  # noqa: BLE001 -- votes no
        why = "%s: %s" % (type(exc).__name__, exc)
    ok = why is None
    if pooled:
        refusal = agree_plan(ok, fp, gather)
        agreed = refusal is None
        if not agreed and why is None:
            why = refusal
    else:
        agreed = agree(ok, gather)
    if not agreed:
        log("L15-PARK at=sleep rank=%d result=off reason=%s park_bytes=0 park_ms=%.0f"
            % (rank, why or "a peer refused", (time.perf_counter() - t0) * 1000.0))
        if pooled:
            log(_pool_line("OUT", epoch, rank, 0, (time.perf_counter() - t0) * 1000.0, 0,
                           why or "a peer refused"))
        return None
    err = None
    sent = 0
    a_sent = 0
    sums = None
    a_sums = None
    run_a2a, rounds = (_counting(a2a) if pooled else (a2a, [0]))
    t_x0 = time.perf_counter()
    try:
        if pooled:
            sums = guest_sums(pieces, rank, bufs)
        if a_guests:
            # the SOURCE checksum of the anchor shares this rank owns, before
            # the send (independent of L2: an anchor share has to arrive intact)
            a_sums = l15_pool_anchor.anchor_sums_of(a_guests, rank, m_v)
        sent = run_park("out", pieces, rank, world, bufs, run_a2a, env,
                        uniform=pooled)
        if a_guests:
            a_sent = l15_pool_anchor.run_anchor_park(
                "out", a_guests, rank, world, m_v, kv_v, run_a2a, env,
                uniform=pooled)
        torch.cuda.current_stream().synchronize()
    except Exception as exc:  # noqa: BLE001 -- the wake refills from L2
        err = "%s: %s" % (type(exc).__name__, exc)
    t_x1 = time.perf_counter()
    # the park counts only if it landed on EVERY rank (no sidecar anywhere
    # otherwise -> the wake refills from L2 on every rank)
    if not agree(err is None, gather):
        log("L15-PARK at=sleep rank=%d result=FAILED (%s) -- L2 refill at the wake"
            % (rank, err or "a peer failed"))
        if pooled:
            log(_pool_line("OUT", epoch, rank, 0, (t_x1 - t_x0) * 1000.0, rounds[0],
                           "transport failed: %s" % (err or "a peer failed")))
        return None
    write_sidecar(rank, env, epoch, pieces, sums=sums,
                  anchor_guests=(a_guests if s4 else None), anchor_sums=a_sums)
    log("L15-PARK at=sleep rank=%d result=parked epoch=%d pieces=%d park_bytes=%d "
        "park_ms=%.0f" % (rank, epoch, len(pieces), sent + a_sent,
                          (time.perf_counter() - t0) * 1000.0))
    if pooled:
        log(_pool_line("OUT", epoch, rank, sent + a_sent, (t_x1 - t_x0) * 1000.0,
                       rounds[0]))
    return sent + a_sent


def park_back_at_wake(sched, env, log, *, epoch: int, group_ok: bool,
                      manifest_guests=None, manifest_anchor_guests=None) -> bool:
    """D wake (resume RPC, every D rank, one list position, before the
    cap-0 refill): bring the parked rows back. True when they are back on
    this group (the cap-0 rank then refills only its anchors from L2).

    S2 pooled hold: the rows that came back are checked against the source
    checksum taken at the sleep (``L15-POOL-CHECK``); one bad sample on any
    rank = group fallback to the L2 refill (``L15-POOL-BACK ... reason=``).

    ``manifest_guests`` (S3, the v2 manifest's guest list; None = S2): the
    park record must list exactly these pieces, else the group falls back --
    sleep and wake agree on WHERE every guest row lies or nothing comes back."""
    import time

    import torch

    from sglang.srt.weg2 import l15_pool

    t0 = time.perf_counter()
    pooled = l15_pool.pool_on(env)
    try:
        rank, world, gather, a2a = _group_io(sched)
    except Exception as exc:  # noqa: BLE001
        log("L15-PARK at=wake skipped (no group: %s)" % (exc,))
        return False
    d = _read_sidecar(rank, env)
    rec = (None if d is None else
           (int(d.get("epoch", -1)),
            [ParkPiece(*map(int, x)) for x in d.get("pieces", ())]))
    sums_before = None if d is None else d.get("sums")
    # S4: the anchor guests and their source checksum of the sleep (absent in an
    # S3 / S2 record)
    a_rec = None
    a_sums_before = None
    if d is not None and d.get("anchor_guests") is not None:
        a_rec = [tuple(int(x) for x in g) for g in d["anchor_guests"]]
        a_sums_before = d.get("anchor_sums") or {}
    why = None
    if not group_ok:
        why = "kv resume refused in the group"
    elif rec is None:
        why = "no park record"
    elif int(rec[0]) != int(epoch):
        why = "park epoch %d != hold epoch %d" % (rec[0], epoch)
    elif manifest_guests is not None and _guest_key(rec[1]) != _guest_key(
            [ParkPiece(*(int(x) for x in g)) for g in manifest_guests]):
        why = "park record differs from the manifest guest list"
    elif manifest_anchor_guests is not None and (
            a_rec is None or [tuple(int(x) for x in g) for g in manifest_anchor_guests]
            != a_rec):
        why = "park record differs from the manifest anchor guest list"
    if not agree(why is None, gather):
        log("L15-PARK at=wake rank=%d result=off reason=%s park_bytes=0 park_ms=%.0f"
            % (rank, why or "a peer has no park", (time.perf_counter() - t0) * 1000.0))
        if pooled:
            log(_pool_line("BACK", epoch, rank, 0, (time.perf_counter() - t0) * 1000.0, 0,
                           why or "a peer has no park"))
        return False
    err = None
    sent = 0
    ck_ok = ck_bad = 0
    a_ok = a_bad = 0
    run_a2a, rounds = (_counting(a2a) if pooled else (a2a, [0]))
    t_x0 = time.perf_counter()
    try:
        bufs, _pool = _kv_buffers(sched)
        sent = run_park("back", rec[1], rank, world, bufs, run_a2a, env,
                        uniform=pooled)
        if a_rec:
            from sglang.srt.weg2 import l15_pool_anchor as _lpa

            _kv_v = [_lpa._rows2d(b) for b in bufs]
            _m_v = _lpa.mamba_views(sched)
            sent += _lpa.run_anchor_park("back", a_rec, rank, world, _m_v, _kv_v,
                                         run_a2a, env, uniform=pooled)
        torch.cuda.current_stream().synchronize()
        if pooled and sums_before is not None:
            ck_ok, ck_bad = compare_sums(sums_before, guest_sums(rec[1], rank, bufs))
            if ck_bad:
                err = "guest round-trip checksum: %d of %d samples differ" % (
                    ck_bad, ck_ok + ck_bad)
        if a_rec:
            a_ok, a_bad = compare_sums(
                a_sums_before, _lpa.anchor_sums_of(a_rec, rank, _m_v))
            if a_bad and err is None:
                err = "anchor round-trip checksum: %d of %d blocks differ" % (
                    a_bad, a_ok + a_bad)
    except Exception as exc:  # noqa: BLE001
        err = "%s: %s" % (type(exc).__name__, exc)
    t_x1 = time.perf_counter()
    if pooled:
        # every sampled row is a guest row in S2 (the home rows never leave
        # their segment; L15-CHECK samples them against L2)
        log(pool_check_line(
            epoch, rank, ck_ok, ck_bad, rec[1] if rec else (),
            s3=manifest_guests is not None,
            anchor=((a_ok, a_bad, sum(1 for g in a_rec if int(g[0]) == int(rank)))
                    if a_rec is not None else None)))
    if not agree(err is None, gather):
        log("L15-PARK at=wake rank=%d result=FAILED (%s) -- L2 refill"
            % (rank, err or "a peer failed"))
        if pooled:
            log(_pool_line("BACK", epoch, rank, 0, (t_x1 - t_x0) * 1000.0, rounds[0],
                           "failed: %s" % (err or "a peer failed")))
        return False
    log("L15-PARK at=wake rank=%d result=back epoch=%d park_bytes=%d park_ms=%.0f"
        % (rank, epoch, sent, (time.perf_counter() - t0) * 1000.0))
    if pooled:
        log(_pool_line("BACK", epoch, rank, sent, (t_x1 - t_x0) * 1000.0, rounds[0]))
    return True
