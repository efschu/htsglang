# SPDX-License-Identifier: Apache-2.0
"""The D side of the transient vision stage in the flip layouts (user order 27.10.):
the PURE half -- what every TP rank of group D decides and exchanges around the
one rank (TP0, the 5090) that encodes.

Why this is not ``vision_rank_pass``: P's stage runs on PP0 alone, whose forward
embeds the tokens, so its rows need no transport. D is TP-lockstep: every rank
embeds the same tokens, so the rows the leader encoded must reach every rank
BEFORE the admission, and every rank must agree on what was encoded -- an
attach by position on a rank that saw another queue would put one image's rows
on another image's placeholders (fluent wrong text, not an error).

One pass (all ranks; the queue is the same on every rank by the lockstep):

1. :func:`plan` -- the candidate items, in queue order. None: return BEFORE any
   collective.
2. the leader decides (its prefix match prices the skip, ``skip_cached``): which
   requests to stage, which to skip (every image position already covered) and
   which to hold one pass. :class:`Selection` is broadcast; a follower
   :func:`resolve`s it against ITS queue (a rid or item it does not have is
   :data:`W_DISAGREE`, a named crash-stop of the group).
3. the leader runs the stage on the chosen requests, :func:`leader_packet` reads
   what it attached, :func:`exchange` broadcasts the verdict and one rows tensor
   per item; a follower attaches (:func:`attach_followers`).

Dual layout: not armed here (vision runs on P only; ``vision_d_guard`` and the
front stay as they are).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Protocol, Sequence, Tuple

import msgspec
import torch

W_DISAGREE = "W124 PdFlipVisionDRankDisagree"
W_NO_ROWS = "W125b PdFlipVisionDNoRows"

Entry = Tuple[Any, int, Any]  # (request, item index, item)


class Wanted(msgspec.Struct, frozen=True):
    """One image item: the request, the item's index in its request's unstaged
    list, and how many rows the leader produced (0 before the stage ran)."""

    rid: str
    idx: int
    rows: int = 0


class Selection(msgspec.Struct, frozen=True):
    """The leader's decision about the candidates, before the stage runs."""

    stage: Tuple[Wanted, ...] = ()
    skip: Tuple[str, ...] = ()
    wait: Tuple[str, ...] = ()


class Packet(msgspec.Struct, frozen=True):
    """The leader's verdict for the staged items."""

    ok: bool
    code: str = ""
    detail: str = ""
    wanted: Tuple[Wanted, ...] = ()
    #: width of every row the leader encoded (review M1: the followers size their buffers from THIS, not from a config
    #: field that omits the deepstack part)
    width: int = 0


class Comm(Protocol):
    """The collectives of the pass, over group D's TP ranks (the source is the leader)."""

    def broadcast_obj(self, obj: Any) -> Any: ...

    def broadcast_rows(self, rows: torch.Tensor) -> torch.Tensor: ...


class DisagreeError(RuntimeError):
    """The ranks planned different items (W124). Nothing is attached on the rank that raises; the group STOPS (named
    crash-stop): the other ranks sit in the next collective and end at its timeout. Review M3: this is no soft refusal."""


def plan(waiting: Sequence[Any], *, unstaged_items: Callable[[Any], Sequence[Any]],
         refused: Sequence[str] = ()) -> List[Entry]:
    """(request, item index, item) for every unstaged item of every waiting
    request that is not refused, in queue order -- the same list on every
    rank."""
    skip = set(refused)
    return [(r, i, it) for r in waiting if r.rid not in skip for i, it in enumerate(unstaged_items(r))]


def selection_of(entries: Sequence[Entry], *, skip: Sequence[str] = (), wait: Sequence[str] = ()) -> Selection:
    """The leader's selection: every candidate whose request is neither skipped
    nor held is staged."""
    out = set(skip) | set(wait)
    return Selection(stage=tuple(Wanted(r.rid, i) for r, i, _it in entries if r.rid not in out),
                     skip=tuple(skip), wait=tuple(wait))


def resolve(entries: Sequence[Entry], selection: Selection) -> List[Entry]:
    """The entries the selection names, in the leader's order; every rid and
    item the selection names must exist in THIS rank's plan (W124 otherwise)."""
    have = {(r.rid, i): (r, i, it) for r, i, it in entries}
    rids = {r.rid for r, _i, _it in entries}
    missing = [(w.rid, w.idx) for w in selection.stage if (w.rid, w.idx) not in have]
    missing += [(rid,) for rid in (*selection.skip, *selection.wait) if rid not in rids]
    if missing:
        raise DisagreeError(f"{W_DISAGREE}: the leader selected {missing} which this rank does not have; "
                            "nothing done")
    return [have[(w.rid, w.idx)] for w in selection.stage]


def leader_packet(entries: Sequence[Entry], *, ok: bool, code: str = "",
                  detail: str = "") -> Tuple[Packet, List[torch.Tensor]]:
    """The leader's packet after its stage: the rows the stage attached to its
    items (host tensors), or the refusal."""
    if not ok:
        return Packet(ok=False, code=code, detail=detail), []
    rows = []
    for r, i, it in entries:
        emb = it.precomputed_embeddings
        if emb is None:
            return Packet(ok=False, code=W_NO_ROWS,
                          detail=f"rid={r.rid} item {i}: the stage reported ok but attached no rows"), []
        rows.append(emb)
    widths = {int(emb.shape[1]) for emb in rows}
    if len(widths) != 1:
        return Packet(ok=False, code=W_NO_ROWS, detail=f"the stage attached rows of different widths {sorted(widths)}"), []
    wanted = tuple(Wanted(r.rid, i, int(emb.shape[0])) for (r, i, _it), emb in zip(entries, rows))
    return Packet(ok=True, wanted=wanted, width=widths.pop()), rows


def exchange(entries: Sequence[Entry], packet: Optional[Packet], rows: Sequence[torch.Tensor], *,
             is_leader: bool, comm: Comm,
             dtype: torch.dtype = torch.bfloat16) -> Tuple[Packet, List[torch.Tensor]]:
    """The collectives after the stage. ``entries`` = the chosen entries of
    this rank (:func:`resolve`). Returns (packet, rows on this rank); a
    follower's rows are NEW tensors, the leader's are the ones its stage
    attached. A refused pass moves no rows."""
    got = comm.broadcast_obj(packet if is_leader else None)
    if not got.ok:
        return got, []
    if not is_leader:
        mine = [(r.rid, i) for r, i, _it in entries]
        theirs = [(w.rid, w.idx) for w in got.wanted]
        if mine != theirs:
            raise DisagreeError(f"{W_DISAGREE}: this rank staged {mine}, the leader {theirs}; nothing attached")
    out = []
    for k, w in enumerate(got.wanted):
        t = rows[k] if is_leader else torch.empty((w.rows, int(got.width)), dtype=dtype)
        out.append(comm.broadcast_rows(t))
    return got, out


def attach_followers(entries: Sequence[Entry], rows: Sequence[torch.Tensor], *,
                     attach: Callable[..., int], width: int) -> int:
    """A follower attaches the received rows, item by item (the leader's items
    already carry theirs). ``attach`` is ``attach_precomputed_embeddings``."""
    if not entries:
        return 0
    return attach([it for _r, _i, it in entries], list(rows), expected_width=int(width))
