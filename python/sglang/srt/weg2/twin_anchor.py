# SPDX-License-Identifier: Apache-2.0
"""TWIN ANCHOR (NF y4a, 30.09.; P log ...0930_031042): the source of a fork
twin writes one extra Mamba anchor at the twin boundary.

THE MEASURED GAP. ``#TW TWIN-NO-GAIN`` twice in y4a:

* weg2-16-28 (62856 tokens) shared 62439 with its sibling weg2-16-27 (62779,
  start 59008). 16-27 ran ONE step [59008,62779) whose tracks were the turn
  anchor 62592 and the end anchor 62720 -- both past 62439 -- so nothing lay in
  (59008, 62439]: '#TW TWIN-DEFER' held 16-28 for 3 s, then NO-GAIN, and it
  re-prefilled 3847 tokens from 59008. An anchor at floor_page(62439) = 62400
  would have left it 456 (floor_page(shared - 1), bigram-safe).
* weg2-12-19 (87844) shared 87678 with weg2-12-21, which started at 87680 --
  past the boundary: no source anchor can serve it. It waited 3 s for nothing.

THE REPAIR -- two halves, one rule (``B = floor_page(shared - 1)``):

* the SOURCE (every P rank): a request whose prefill step holds ``B`` of a
  twin QUEUED behind it (``shared >= SGLANG_WEG2_P_TWIN_MIN_TOKENS`` leading
  ids, same extra key) draws one more extend track at ``B`` -- the TURN
  ANCHOR's second-track machinery (weg2/turn_anchor.py): one gather row per
  GDN/PLE layer in the same forward, one mamba slot from the pool (no
  eviction, no reserve; none free = no anchor, counted), inserted into the
  tree as its own node in position order before the step's insert, then
  published like every chunk anchor. The bounds are a pure function of the
  source's and the queued requests' token ids, and a follower plans PP0's
  pass-m batch with PP0's pass-m requests (the #1400 wire lag), so every
  stage plans the same tracks without transport.
* the TWIN (PP0, weg2/p_twin_defer.py): it waits only for a source that has
  PROMISED an anchor at or below ``shared`` -- its end anchor, a chunk end,
  or ``B`` while the source has not yet planned past it. A source that cannot
  promise (start past ``B``; ``B`` planned without a slot) holds nobody:
  the twin registers at once ('#TW TWIN-NO-COMMIT').

``SGLANG_WEG2_TWIN_ANCHOR`` (default on) switches the source half; it runs
only where the turn anchor is armed (group P, extra-buffer tracks, no
overlap) and the twin deferral is. Host cost: the shared length of each
(source, queued twin) pair once (cached), per prefill batch O(batch x queue)
dict reads. Device: one gather row per boundary, inside a forward that runs
anyway -- no forward, no decode step, no flip is touched.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

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
            from sglang.srt.environ import envs
            from sglang.srt.weg2 import p_twin_defer as _tw

            _ARMED = bool(envs.SGLANG_WEG2_TWIN_ANCHOR.get()) and bool(_tw._env_on())
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
    from sglang.srt.weg2 import p_twin_defer as _tw

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
    from sglang.srt.weg2 import p_twin_defer as _tw

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
    from sglang.srt.weg2 import p_twin_defer as _tw

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


def forget(rid: str) -> None:
    _STATUS.pop(str(rid), None)


def _reset_for_test() -> None:
    global _ARMED
    _ARMED = None
    _SHARED.clear()
    _STATUS.clear()
