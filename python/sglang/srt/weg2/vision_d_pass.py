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

The flow of one pass (all ranks, same queue by construction of the lockstep):

1. :func:`plan` -- the candidate items, in queue order. No candidate: return
   BEFORE any collective (the ranks agree on that because they share the queue).
2. the leader runs ``stage_fn`` (the in-rank stage with its weights lease) and
   :func:`leader_packet` reads what it attached;
3. :func:`exchange` -- one object broadcast (verdict + the (rid, item, rows)
   list), then one rows broadcast per item;
4. a follower checks the packet against ITS plan (:func:`check_agreement`; a
   mismatch is :data:`W_DISAGREE`, nothing attached) and attaches.

Dual layout: not armed here (vision runs on P only; ``vision_d_guard`` and the
front stay as they are).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Protocol, Sequence, Tuple

import msgspec
import torch

W_DISAGREE = "W124 Weg2VisionDRankDisagree"
W_NO_ROWS = "W125b Weg2VisionDNoRows"


class Wanted(msgspec.Struct, frozen=True):
    """One image item the pass wants rows for: the request, the item's index in
    its request's unstaged list, and how many rows the leader produced."""

    rid: str
    idx: int
    rows: int = 0


class Packet(msgspec.Struct, frozen=True):
    """The leader's verdict for the whole pass."""

    ok: bool
    code: str = ""
    detail: str = ""
    wanted: Tuple[Wanted, ...] = ()


class Comm(Protocol):
    """The two collectives of the pass, over group D's TP ranks."""

    def broadcast_obj(self, obj: Optional[Packet]) -> Packet: ...

    def broadcast_rows(self, rows: torch.Tensor) -> torch.Tensor: ...


class DisagreeError(RuntimeError):
    """The ranks planned different items: nothing is attached (W124)."""


def plan(waiting: Sequence[Any], *, unstaged_items: Callable[[Any], Sequence[Any]],
         refused: Sequence[str] = ()) -> List[Tuple[Any, int, Any]]:
    """(request, item index, item) for every unstaged item of every waiting
    request that is not refused, in queue order -- the same list on every
    rank."""
    skip = set(refused)
    return [(r, i, it) for r in waiting if r.rid not in skip for i, it in enumerate(unstaged_items(r))]


def leader_packet(entries: Sequence[Tuple[Any, int, Any]], *, ok: bool, code: str = "",
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
    wanted = tuple(Wanted(r.rid, i, int(emb.shape[0])) for (r, i, _it), emb in zip(entries, rows))
    return Packet(ok=True, wanted=wanted), rows


def check_agreement(entries: Sequence[Tuple[Any, int, Any]], packet: Packet) -> None:
    """A follower plans the same (rid, item) sequence as the leader or attaches
    nothing."""
    mine = [(r.rid, i) for r, i, _it in entries]
    theirs = [(w.rid, w.idx) for w in packet.wanted]
    if mine != theirs:
        raise DisagreeError(f"{W_DISAGREE}: this rank planned {mine}, the leader {theirs}; nothing attached")


def exchange(entries: Sequence[Tuple[Any, int, Any]], packet: Optional[Packet],
             rows: Sequence[torch.Tensor], *, is_leader: bool, comm: Comm, width: int,
             dtype: torch.dtype = torch.bfloat16) -> Tuple[Packet, List[torch.Tensor]]:
    """The collectives of one pass. Returns (packet, rows on this rank); a
    follower's rows are NEW tensors, the leader's are the ones its stage
    attached. A refused pass moves no rows."""
    got = comm.broadcast_obj(packet if is_leader else None)
    if not got.ok:
        return got, []
    if not is_leader:
        check_agreement(entries, got)
    out = []
    for k, w in enumerate(got.wanted):
        t = rows[k] if is_leader else torch.empty((w.rows, int(width)), dtype=dtype)
        out.append(comm.broadcast_rows(t))
    return got, out


def attach_followers(entries: Sequence[Tuple[Any, int, Any]], rows: Sequence[torch.Tensor], *,
                     attach: Callable[..., int], width: int) -> int:
    """A follower attaches the received rows, item by item (the leader's items
    already carry theirs). ``attach`` is ``attach_precomputed_embeddings``."""
    if not entries:
        return 0
    return attach([it for _r, _i, it in entries], list(rows), expected_width=int(width))
