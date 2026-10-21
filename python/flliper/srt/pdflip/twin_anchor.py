# SPDX-License-Identifier: Apache-2.0
"""TWIN ANCHOR (NF y4a, 30.09.; P log ...0930_031042): the source of a fork
twin writes one extra Mamba anchor at the twin boundary.

THE MEASURED GAP. ``#TW TWIN-NO-GAIN`` twice in y4a:

* pdflip-16-28 (62856 tokens) shared 62439 with its sibling pdflip-16-27 (62779,
  start 59008). 16-27 ran ONE step [59008,62779) whose tracks were the turn
  anchor 62592 and the end anchor 62720 -- both past 62439 -- so nothing lay in
  (59008, 62439]: '#TW TWIN-DEFER' held 16-28 for 3 s, then NO-GAIN, and it
  re-prefilled 3847 tokens from 59008. An anchor at floor_page(62439) = 62400
  would have left it 456 (floor_page(shared - 1), bigram-safe).
* pdflip-12-19 (87844) shared 87678 with pdflip-12-21, which started at 87680 --
  past the boundary: no source anchor can serve it. It waited 3 s for nothing.

THE REPAIR -- two halves, one rule (``B = floor_page(shared - 1)``):

* the SOURCE (every P rank): a request whose prefill step holds ``B`` of a
  twin QUEUED behind it (``shared >= FLLIPER_PDFLIP_P_TWIN_MIN_TOKENS`` leading
  ids, same extra key) draws one more extend track at ``B`` -- the TURN
  ANCHOR's second-track machinery (pdflip/turn_anchor.py): one gather row per
  GDN/PLE layer in the same forward, one mamba slot from the pool (no
  eviction, no reserve; none free = no anchor, counted), inserted into the
  tree as its own node in position order before the step's insert, then
  published like every chunk anchor. The bounds are a pure function of the
  source's and the queued requests' token ids -- decided by PP0 ALONE
  (NF-TWIN-ROW 1021, below): the followers adopt PP0's bounds from the pass's
  proxy frame, they never derive them from their own waiting queue.
* the TWIN (PP0, pdflip/p_twin_defer.py): it waits only for a source that has
  PROMISED an anchor at or below ``shared`` -- its end anchor, a chunk end,
  or ``B`` while the source has not yet planned past it. A source that cannot
  promise (start past ``B``; ``B`` planned without a slot) holds nobody:
  the twin registers at once ('#TW TWIN-NO-COMMIT').

``FLLIPER_PDFLIP_TWIN_ANCHOR`` (default on) switches the source half; it runs
only where the turn anchor is armed (group P, extra-buffer tracks, no
overlap) and the twin deferral is. Host cost: the shared length of each
(source, queued twin) pair once (cached), per prefill batch O(batch x queue)
dict reads. Device: one gather row per boundary, inside a forward that runs
anyway -- no forward, no decode step, no flip is touched.

NF-TWIN-ROW 1021 (rc27 893d09fa3f, P log ..._1021_020616) -- THE BOUNDS ARE PP0'S,
THE FOLLOWERS ADOPT THEM. The rule above ("a follower plans PP0's pass-m batch with
PP0's pass-m requests, so every stage plans the same tracks without transport") held
for the batch MEMBERSHIP (the #631 row rides the frame) but not for the QUEUE the
bounds were drawn from: each rank read its OWN ``waiting_queue``. A follower plans
pass m when PP0's frame m lands -- seconds after PP0 planned it -- and by then it has
absorbed requests PP0 received only after its plan:

  02:09:27 PP0 P-FORK-CUT SKIP rid=pdflip-0-1 site=add_one_req ... (PP0 plans step [18432,34816))
  02:09:28 PP0 FORK-ANCHOR TRACK rid=pdflip-0-1 at=18688            (no twin queued on PP0)
  02:09:31 PP0 REQ-TRACE r9 ... rids=['pdflip-0-4', ...]           (the twin, after the plan)
  02:09:31 PP1 REQ-TRACE r10 ... rids=['pdflip-0-4', ...]          (the same chain hop)
  02:09:31 PP1 ROW-DELIVER d1 ... -> TWIN-ANCHOR TRACK at=18688 / at=18880
  02:09:34 PP2 TWIN-ANCHOR TRACK at=18688 / at=18880

PP1/PP2 wrote an anchor at 18880 PP0 never wrote: pdflip-0-4 resumed at PP0's 18688
(192 tokens again), and the split tree outlived the request -- pdflip-1-22 matched
``accepted=18688`` on PP0 against ``accepted=18880`` on PP1/PP2 and the TF verdict
fell to 18688 (120320 tokens, NF-TF-GAP in pdflip_told_fidelity).

THE RULE: PP0 computes the bounds of its pass (:func:`batch_bounds`, unchanged) and
puts them on that pass's proxy frame (:data:`ROW_KEY`, the ``__stamp__`` / admission
row / W2C precedent: a non-tensor entry popped by the receiver before anything maps
over the dict). A follower pops the row at its receive (:func:`note_frame`), the
#631 pre-plan block hands it to exactly the plan of that slot (:func:`plant_for_plan`)
and :func:`plan_bounds` returns PP0's bounds for the follower's batch -- never a
rank-local derivation. A middle rank relays what it adopted (its batch's bounds ARE
the row). A frame without the key means PP0 planned no twin track for the pass; a
follower plan without a planted row (no #631 row authority on the boot) plans none
and says so (RAENGE-NIE-UNEINS: no rank-local substitute).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: NF-TWIN-ROW: the proxy-frame key PP0's twin bounds ride under
ROW_KEY = "__pdflip_twin_bounds__"
#: scheduler attributes: the row popped at the last receive / planted for ONE plan
RECV_ATTR = "_pdflip_twin_row_recv"
PLAN_ATTR = "_pdflip_twin_row_plan"
_ROW_N: Dict[str, int] = {"sent": 0, "adopted": 0, "differs": 0, "absent": 0}

#: tracks per request and step (one mamba slot each)
MAX_BOUNDS_PER_REQ = 4
_CAP = 4096

#: (source rid, twin rid) -> shared leading ids
_SHARED: Dict[Tuple[str, str], int] = {}
#: source rid -> {B: "planned" | "written" | "declined"}
_STATUS: Dict[str, Dict[int, str]] = {}


_ARMED: Optional[bool] = None


def armed() -> bool:
    """The switch AND the twin deferral (read once per process: boot env)."""
    global _ARMED
    if _ARMED is None:
        try:
            from flliper.srt.environ import envs
            from flliper.srt.pdflip import p_twin_defer as _tw

            _ARMED = bool(envs.FLLIPER_PDFLIP_TWIN_ANCHOR.get()) and bool(_tw._env_on())
        except Exception:  # noqa: BLE001 - unreadable = off
            _ARMED = False
    return _ARMED


def boundary(shared: int, page: int) -> int:
    """``floor_page(shared - 1)``: a twin sharing ``ids[:shared]`` claims at
    most there -- bigram keys (MTP/EAGLE trees) take the next token too, the
    same bound the turn anchor sits on (turn_anchor.anchor_pos)."""
    page = max(1, int(page))
    return (max(0, int(shared) - 1) // page) * page


def _rid(r: Any) -> str:
    return str(getattr(r, "rid", ""))


def _shared(src: Any, twin: Any) -> int:
    from flliper.srt.pdflip import p_twin_defer as _tw

    key = (_rid(src), _rid(twin))
    v = _SHARED.get(key)
    if v is None:
        v = _tw.shared_prefix_len(_tw._ids(src), _tw._ids(twin))
        if len(_SHARED) >= _CAP:
            _SHARED.pop(next(iter(_SHARED)))
        _SHARED[key] = v
    return v


def bounds_for(src: Any, queued: Iterable[Any], *, page: int, min_tokens: int) -> List[int]:
    """The twin boundaries ``B`` of ``src``: one per queued request sharing at
    least ``min_tokens`` leading ids with it (same extra key), below the
    source's own length, at most MAX_BOUNDS_PER_REQ (the deepest kept)."""
    from flliper.srt.pdflip import p_twin_defer as _tw

    n_src = len(_tw._ids(src))
    out = set()
    for q in queued:
        if q is src or _rid(q) == _rid(src) or not _tw.is_twin(q, src, min_tokens):
            continue
        b = boundary(_shared(src, q), page)
        if 0 < b < n_src:
            out.add(b)
    return sorted(out)[-MAX_BOUNDS_PER_REQ:]


def batch_bounds(reqs: Sequence[Any], queued: Sequence[Any], *, page: int) -> Optional[Dict[str, List[int]]]:
    """Scheduler, before ``prepare_for_extend``: rid -> twin boundaries of every
    request of the new prefill batch with a twin still queued. None when off
    or nothing is queued (the batch then plans exactly as before)."""
    if not reqs or not queued or not armed():
        return None
    from flliper.srt.pdflip import p_twin_defer as _tw

    min_tokens = _tw._env_int(_tw.ENV_MIN_TOKENS, _tw.DEFAULT_MIN_TOKENS)
    long_q = [q for q in queued if len(_tw._ids(q)) >= min_tokens]
    if not long_q:
        return None
    out: Dict[str, List[int]] = {}
    for r in reqs:
        if len(_tw._ids(r)) < min_tokens:
            continue
        b = bounds_for(r, long_q, page=page, min_tokens=min_tokens)
        if b:
            out[_rid(r)] = b
    return out or None


# -- NF-TWIN-ROW: PP0 decides, the followers adopt (rank-uniform) -----------------
def _pp_rank_size(scheduler: Any) -> Tuple[int, int]:
    ps = getattr(scheduler, "ps", None)
    try:
        return int(getattr(ps, "pp_rank", 0) or 0), int(getattr(ps, "pp_size", 1) or 1)
    except (TypeError, ValueError):
        return 0, 1


def encode(bounds: Optional[Dict[str, Sequence[int]]]) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
    """Plain tuples on the wire (the admission-row form), sorted by rid."""
    return tuple(
        (str(r), tuple(sorted({int(b) for b in bs})))
        for r, bs in sorted((bounds or {}).items())
        if bs
    )


def decode(raw: Any) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for item in raw or ():
        try:
            rid, bs = item
            vals = sorted({int(b) for b in bs})
        except (TypeError, ValueError):
            continue
        if vals:
            out[str(rid)] = vals
    return out


def frame_extra(scheduler: Any, batch: Any, base: Optional[dict]) -> Optional[dict]:
    """Sender, at the proxy send of ``batch``'s slot: PP0 (its decision) and a middle
    rank (the row it adopted -- its batch's bounds ARE the row) put the batch's twin
    bounds into the frame extras. ``base`` unchanged when there are none (an absent key
    reads as "PP0 planned no twin track"). Never raises into the send."""
    try:
        rank, size = _pp_rank_size(scheduler)
        if size <= 1 or rank >= size - 1:
            return base
        row = encode(getattr(batch, "pdflip_twin_bounds", None))
        if not row:
            return base
        out = dict(base) if base else {}
        out[ROW_KEY] = row
        _ROW_N["sent"] += 1
        return out
    except Exception as exc:  # noqa: BLE001 - the frame must go
        logger.warning("PDFLIP TWIN-ROW frame_extra skipped (%s: %s)", type(exc).__name__, exc)
        return base


def note_frame(scheduler: Any, raw: Any) -> Dict[str, List[int]]:
    """Receiver: pop the row off a proxy frame (BEFORE anything maps over the dict)
    and keep it for the plan of this slot. Absent key = {} (PP0 planned none)."""
    val = raw.pop(ROW_KEY, None) if isinstance(raw, dict) else None
    row = decode(val)
    try:
        setattr(scheduler, RECV_ATTR, row)
    except Exception:  # noqa: BLE001
        pass
    return row


def plant_for_plan(scheduler: Any) -> None:
    """#631 pre-plan block, right after this slot's frame was received: the row of
    THAT frame is the one the next plan call adopts (moved, so a later receive on
    another path can never feed a plan it does not belong to)."""
    row = getattr(scheduler, RECV_ATTR, None)
    try:
        setattr(scheduler, PLAN_ATTR, {} if row is None else row)
        setattr(scheduler, RECV_ATTR, None)
    except Exception:  # noqa: BLE001
        pass


def clear_after_plan(scheduler: Any) -> None:
    try:
        setattr(scheduler, PLAN_ATTR, None)
    except Exception:  # noqa: BLE001
        pass


def plan_bounds(scheduler: Any, reqs: Sequence[Any], queued: Sequence[Any], *,
                page: int) -> Optional[Dict[str, List[int]]]:
    """Scheduler, before ``prepare_for_extend``: the twin bounds of the new prefill
    batch. PP0 and every non-PP group: :func:`batch_bounds` over its own queue (the
    decision). A PP follower: PP0's bounds from this pass's frame, restricted to the
    batch's rids -- never its own queue (NF-TWIN-ROW)."""
    rank, size = _pp_rank_size(scheduler)
    if size <= 1 or rank == 0:
        return batch_bounds(reqs, queued, page=page)
    if not reqs or not armed():
        return None
    row = getattr(scheduler, PLAN_ATTR, None)
    if row is None:
        _ROW_N["absent"] += 1
        n = _ROW_N["absent"]
        if n <= 3 or n % 1000 == 0:
            logger.warning(
                "PDFLIP TWIN-ROW ABSENT pp=%d n=%d: a follower plan without PP0's frame row "
                "(no #631 row authority on this path) -- no twin track, never a rank-local "
                "derivation (NF-TWIN-ROW, RAENGE-NIE-UNEINS)", rank, n)
        return None
    rids = {_rid(r) for r in reqs}
    out = {r: list(bs) for r, bs in row.items() if r in rids and bs}
    try:
        local = batch_bounds(reqs, queued, page=page) or {}
    except Exception:  # noqa: BLE001 - instrument only
        local = out
    _ROW_N["adopted"] += 1
    if {r: list(b) for r, b in local.items()} != out:
        _ROW_N["differs"] += 1
        n = _ROW_N["differs"]
        if n <= 24 or n % 256 == 0:
            logger.info(
                "PDFLIP TWIN-ROW ADOPT pp=%d n=%d adopted=%s own_queue_would=%s (NF-TWIN-ROW: "
                "PP0's bounds, not this rank's queue -- a twin PP0 received after its plan "
                "gets no track on any stage)", rank, n, out, local)
    return out or None


def row_stats() -> Dict[str, int]:
    return dict(_ROW_N)


# -- the promise ledger (per process; PP0's is the one the twin reads) -----------
def _set(rid: str, b: int, status: str) -> None:
    d = _STATUS.get(rid)
    if d is None:
        if len(_STATUS) >= _CAP:
            _STATUS.pop(next(iter(_STATUS)))
        d = _STATUS[rid] = {}
    if d.get(int(b)) == "written" and status != "written":
        return
    d[int(b)] = status


def note_planned(rid: str, b: int) -> None:
    _set(rid, b, "planned")


def note_written(rid: str, b: int) -> None:
    _set(rid, b, "written")


def note_declined(rid: str, b: int) -> None:
    _set(rid, b, "declined")


def status(rid: str, b: int) -> Optional[str]:
    return (_STATUS.get(str(rid)) or {}).get(int(b))


def note_track_planned(rid: str, pos: int) -> None:
    """S52 (NF rc14 19.10., P log ...1019_010742, pdflip-1-30): a TURN or FORK track the
    source planned for itself is an anchor it writes anyway -- a twin that arrives AFTER
    that step was planned (no ``twin_bounds`` then) resumes there all the same. Booked in
    the same ledger (planned -> written at the insert, declined at a skip), so
    :func:`deepest_track` sees it. Only where the twin anchor is armed (the switch-off
    path books nothing, byte for byte)."""
    if armed():
        _set(str(rid), int(pos), "planned")


def deepest_track(rid: str, lo: int, hi: int) -> Optional[int]:
    """S52: the deepest anchor position ``p`` of ``rid`` with ``lo < p <= hi`` whose track
    is planned or written (a declined one writes nothing), or None. ``hi`` is the twin
    boundary ``B = floor_page(shared - 1)``: an anchor there or below is on the twin's
    path; ``lo`` the source's start depth (an anchor at or below it is there already)."""
    d = _STATUS.get(str(rid)) or {}
    best = None
    for p, st in d.items():
        if st in ("planned", "written") and int(lo) < int(p) <= int(hi):
            if best is None or int(p) > best:
                best = int(p)
    return best


def forget(rid: str) -> None:
    _STATUS.pop(str(rid), None)


def _reset_for_test() -> None:
    global _ARMED
    _ARMED = None
    _SHARED.clear()
    _STATUS.clear()
    for k in _ROW_N:
        _ROW_N[k] = 0
