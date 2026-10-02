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
    return str(env.get(PARK_ENV, "0")).strip() == "1"


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
             world: int, buffers: Sequence, a2a, env=None) -> int:
    """Move every piece's rows of every buffer: ``direction`` "out" (sleep:
    src -> dst) or "back" (wake: dst -> src). EVERY rank of the group calls
    this with the same pieces and buffers in the same order (one collective
    per (piece, buffer)); ``a2a(output, input, out_splits, in_splits)`` is the
    group's uneven all_to_all. Returns the bytes this rank sent."""
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


def write_sidecar(rank: int, env, epoch: int, pieces: Sequence[ParkPiece]) -> None:
    import json
    import os

    path = sidecar_path(rank, env)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"epoch": int(epoch),
                   "pieces": [[p.src, p.dst, p.src_row, p.dst_row, p.rows]
                              for p in pieces]}, fh)
    os.replace(tmp, path)


def take_sidecar(rank: int, env) -> Optional[Tuple[int, List[ParkPiece]]]:
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
    return int(d.get("epoch", -1)), [ParkPiece(*map(int, x)) for x in d.get("pieces", ())]


def agree(ok: bool, gather) -> bool:
    """True iff every rank of the group says ok (one host gather)."""
    votes = gather(bool(ok))
    return all(bool(v) for v in votes)


# -- scheduler entries --------------------------------------------------------

def _group_io(sched):
    """(rank, world, gather(obj)->list, a2a(out, inp, osp, isp)) of D's TP group."""
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

    def a2a(out, inp, osp, isp):
        g.all_to_all_single_v(out, inp, osp, isp)

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


def park_at_release(sched, env, log) -> Optional[int]:
    """D sleep (release RPC, every D rank, before the kv pause): park the
    cap-0 ranks' held rows on capped ranks. Returns bytes sent by this rank,
    or None when no park ran (named in the line; the wake refills from L2).
    Never raises: a failure after the agreement is logged and leaves no
    sidecar, so the wake refills from L2 on every rank."""
    import time

    import torch

    from sglang.srt.weg2 import l15_manifest

    t0 = time.perf_counter()
    try:
        rank, world, gather, a2a = _group_io(sched)
    except Exception as exc:  # noqa: BLE001 -- no group, no park
        log("L15-PARK at=sleep skipped (no group: %s)" % (exc,))
        return None
    pieces, why, epoch = [], None, -1
    try:
        m = l15_manifest.read(l15_manifest.manifest_path("D", rank, env))
        bufs, pool = _kv_buffers(sched)
        if m is None:
            why = "no manifest on this rank"
        elif not bufs:
            why = "no KV buffers"
        else:
            epoch = int(m.epoch)
            pieces, why = park_plan(list(m.rows_by_rank), _caps(sched, pool, world, env))
            if why is None and not pieces:
                why = "nothing to park"
            if why is None:
                why = bounds_refusal(pieces, rank, bufs)
    except Exception as exc:  # noqa: BLE001 -- votes no
        why = "%s: %s" % (type(exc).__name__, exc)
    ok = why is None
    if not agree(ok, gather):
        log("L15-PARK at=sleep rank=%d result=off reason=%s park_bytes=0 park_ms=%.0f"
            % (rank, why or "a peer refused", (time.perf_counter() - t0) * 1000.0))
        return None
    err = None
    sent = 0
    try:
        sent = run_park("out", pieces, rank, world, bufs, a2a, env)
        torch.cuda.current_stream().synchronize()
    except Exception as exc:  # noqa: BLE001 -- the wake refills from L2
        err = "%s: %s" % (type(exc).__name__, exc)
    # the park counts only if it landed on EVERY rank (no sidecar anywhere
    # otherwise -> the wake refills from L2 on every rank)
    if not agree(err is None, gather):
        log("L15-PARK at=sleep rank=%d result=FAILED (%s) -- L2 refill at the wake"
            % (rank, err or "a peer failed"))
        return None
    write_sidecar(rank, env, epoch, pieces)
    log("L15-PARK at=sleep rank=%d result=parked epoch=%d pieces=%d park_bytes=%d "
        "park_ms=%.0f" % (rank, epoch, len(pieces), sent,
                          (time.perf_counter() - t0) * 1000.0))
    return sent


def park_back_at_wake(sched, env, log, *, epoch: int, group_ok: bool) -> bool:
    """D wake (resume RPC, every D rank, one list position, before the
    cap-0 refill): bring the parked rows back. True when they are back on
    this group (the cap-0 rank then refills only its anchors from L2)."""
    import time

    import torch

    t0 = time.perf_counter()
    try:
        rank, world, gather, a2a = _group_io(sched)
    except Exception as exc:  # noqa: BLE001
        log("L15-PARK at=wake skipped (no group: %s)" % (exc,))
        return False
    rec = take_sidecar(rank, env)
    why = None
    if not group_ok:
        why = "kv resume refused in the group"
    elif rec is None:
        why = "no park record"
    elif int(rec[0]) != int(epoch):
        why = "park epoch %d != hold epoch %d" % (rec[0], epoch)
    if not agree(why is None, gather):
        log("L15-PARK at=wake rank=%d result=off reason=%s park_bytes=0 park_ms=%.0f"
            % (rank, why or "a peer has no park", (time.perf_counter() - t0) * 1000.0))
        return False
    err = None
    sent = 0
    try:
        bufs, _pool = _kv_buffers(sched)
        sent = run_park("back", rec[1], rank, world, bufs, a2a, env)
        torch.cuda.current_stream().synchronize()
    except Exception as exc:  # noqa: BLE001
        err = "%s: %s" % (type(exc).__name__, exc)
    if not agree(err is None, gather):
        log("L15-PARK at=wake rank=%d result=FAILED (%s) -- L2 refill"
            % (rank, err or "a peer failed"))
        return False
    log("L15-PARK at=wake rank=%d result=back epoch=%d park_bytes=%d park_ms=%.0f"
        % (rank, epoch, sent, (time.perf_counter() - t0) * 1000.0))
    return True
