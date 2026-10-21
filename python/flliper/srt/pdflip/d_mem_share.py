# SPDX-License-Identifier: Apache-2.0
"""D-MEM-SHARE: the D-MEM-SCHED census counts a shared prefix ONCE (20.10.).

WHY (rc28 D ``..._297198745d_1020_174205``, 17:42-19:50Z). ``runtime_tick`` steps the KV stage on
``used = sum(len(request))`` over the running requests and ``incoming = sum(len(request))`` over the
admissible queue. Two requests of one agent conversation share their whole prompt prefix -- ONE set of
KV pages in the radix tree, locked by both -- but the census charged it twice:
``18:43:49 D-MEM-SCHED from=S4 to=S6 used=101466 incoming=101450 -- grow to hold 211598`` for
``pdflip-22-334`` whose extend was ``#969 EXTENT ('pdflip-22-334', 101248, 101450, 101248, 202)``: 202
new tokens, the other 101248 the running ``pdflip-22-333``'s tree pages. 106 of the boot's 116 grows
that the next ``#969 EXTENT`` names were held by the stage they left once the matched prefix is counted
once. Every such grow is a live move inside the admission iteration (KV cells mapped, the lent expert
rows given back / kept by S106 STAGE-KEEP) and its end-event shrink a second one: the decode pause
``D-DECODE-GAP cause=other`` (TP0: host p50 520 ms with a stage move, 409 ms without).

THE COUNT. Pages are shared only through the tree, and a tree path is unique per token sequence, so two
requests ``i`` and ``j`` hold the same pages for exactly their first

    credit(i, j) = page_floor(min(LCP(prompt_i, prompt_j), P_i, P_j))

tokens, ``P`` = ``req.cache_protected_len`` (the committed tree-owned prefix: after a request's insert
the tree owns the KV below it; for a queued request the arrival match's device prefix). The distinct
tokens of a set are ``sum(T_j) - sum_j max_{i<j} credit(i, j)`` (a trie built one path at a time).

RANK AGREEMENT (raenge-nie-uneins, SOLL R11). ``LCP`` is replicated (token ids), ``P`` is rank-local (the
arrival match of a queued request may differ per rank, RU in ``tp_match_floor``). A credit is therefore
taken ONLY from a group MIN: every pair with a QUEUED member that carries the front's price with a cached
page (``front_x`` header, replicated) and a non-zero page-floored LCP (token ids) is a candidate -- a
replicated list, so every rank enters with the same length. The credits ride in the reserve agreement's
existing CPU-group MIN (``d_seat_vram._agree_reserve``, entered only while the queue is non-empty): no
new collective, a decode round with nothing queued pays none. The agreed values go into a book keyed by the rid pair; a pair of two
running requests reads the book (written only with agreed values on ticks every rank runs) and is never
computed locally. A rid that leaves the set leaves the book. No device op, no host read of a tensor.

DANGER DIRECTION (credit too high = the stage too small = an admission NO_TOKEN that would hold forever):
* a pair without an agreed value has credit 0 (the old count);
* ``note_refused``: a request this rank's admission refused with NO_TOKEN contributes credit 0 from the
  next tick on (MIN -> the group's credit is 0), so the stage grows for its full length as before;
* the book never credits more than the smallest rank's tree prefix (MIN);
* a credit can only lower the NEED the machine steps on; shrinks still stop at the group's live floor.

Switch ``FLLIPER_PDFLIP_D_MEM_SHARE`` (default ON; off = the old sums byte for byte).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

MARK = "PDFLIP D-MEM-SHARE"
ENV = "FLLIPER_PDFLIP_D_MEM_SHARE"
ATTR = "_pdflip_d_mem_share"
#: block width of the LCP walk (list slices compare in C)
_BLOCK = 4096


def enabled() -> bool:
    try:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_D_MEM_SHARE.get())
    except Exception:  # noqa: BLE001 -- a desk double without environ: the env var itself
        return (os.environ.get(ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def lcp(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the common prefix of two token sequences (C-level slice compares, block by block)."""
    if not isinstance(a, (list, tuple)):
        a = list(a)
    if not isinstance(b, (list, tuple)):
        b = list(b)
    n = min(len(a), len(b))
    if n <= 0:
        return 0
    if a[:n] == b[:n]:
        return n
    lo = 0
    while lo < n:
        hi = min(n, lo + _BLOCK)
        if a[lo:hi] != b[lo:hi]:
            for k in range(lo, hi):
                if a[k] != b[k]:
                    return k
        lo = hi
    return n


def _rid(req) -> str:
    return str(getattr(req, "rid", id(req)))


def _tokens(req) -> int:
    return len(getattr(req, "origin_input_ids", None) or ()) + len(getattr(req, "output_ids", None) or ())


def _prompt(req) -> Sequence[int]:
    return getattr(req, "origin_input_ids", None) or ()


def _protected(req) -> int:
    try:
        return max(0, int(getattr(req, "cache_protected_len", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _key(a: str, b: str) -> FrozenSet[str]:
    return frozenset((a, b))


class ShareBook:
    """REPLICATED: the group-agreed credits of the current set (written only with agreed values)."""

    __slots__ = ("lcp", "agreed", "refused", "last")

    def __init__(self) -> None:
        self.lcp: Dict[Tuple[str, str], int] = {}  # rank-local cache of a replicated value (token ids)
        self.agreed: Dict[FrozenSet[str], int] = {}
        self.refused: Set[str] = set()  # rank-local: this rank's admission refused it (NO_TOKEN)
        self.last: Optional[tuple] = None  # the last logged (used, incoming, shared_used, shared_in)

    def lcp_of(self, ri: str, rj: str, a, b) -> int:
        k = (ri, rj) if ri <= rj else (rj, ri)
        v = self.lcp.get(k)
        if v is None:
            v = lcp(_prompt(a), _prompt(b))
            self.lcp[k] = v
        return v

    def keep_only(self, rids: Set[str]) -> None:
        self.lcp = {k: v for k, v in self.lcp.items() if k[0] in rids and k[1] in rids}
        self.agreed = {k: v for k, v in self.agreed.items() if k <= rids}
        self.refused &= rids


def book(sched) -> ShareBook:
    b = getattr(sched, ATTR, None)
    if b is None:
        b = ShareBook()
        try:
            setattr(sched, ATTR, b)
        except Exception:  # noqa: BLE001 -- a slotted double: a fresh book per call (credit 0)
            pass
    return b


def note_refused(sched, req) -> None:
    """The admission refused ``req`` with NO_TOKEN on THIS rank: from the next tick on its credit is 0 here,
    so the group MIN is 0 and the stage grows for its full length (the guard of the danger direction)."""
    try:
        if enabled():
            book(sched).refused.add(_rid(req))
    except Exception:  # noqa: BLE001 -- the guard never stops an admission
        pass


def _front_cached(req) -> Optional[int]:
    """REPLICATED: the tokens the front priced as cached (``len - front_priced``, the carried header of a
    first admission, the same on every D rank), or None without a header (a resumed request, no front)."""
    try:
        from flliper.srt.pdflip import front_x

        _x, priced = front_x.request_front_x(req)
    except Exception:  # noqa: BLE001 -- unreadable = no header
        return None
    if priced is None:
        return None
    return max(0, _tokens(req) - max(0, int(priced)))


def _local_credit(bk: ShareBook, a, b, page: int, cap: Optional[int] = None) -> int:
    ra, rb = _rid(a), _rid(b)
    if ra in bk.refused or rb in bk.refused:
        return 0
    n = min(bk.lcp_of(ra, rb, a, b), _protected(a), _protected(b))
    if cap is not None:
        n = min(n, int(cap))
    page = max(1, int(page))
    return max(0, (int(n) // page) * page)


class Plan:
    """One tick's census before the group call: the members, the candidate pairs and this rank's credits."""

    __slots__ = ("members", "rids", "n_run", "cands", "local", "used_raw", "incoming_raw", "on")

    def __init__(self) -> None:
        self.members: List = []
        self.rids: List[str] = []
        self.n_run = 0
        self.cands: List[Tuple[int, int]] = []
        self.local: List[int] = []
        self.used_raw = 0
        self.incoming_raw = 0
        self.on = False


def prepare(sched, running: List, admissible: List, page: int) -> Plan:
    """The replicated candidate list and this rank's local credits. A candidate is a pair with a QUEUED member
    that carries the front's price with at least one cached page and shares at least one page of prompt
    with the other member -- token ids and the carried header only, so every rank builds the same list and
    enters the group call with the same length. No candidate = no value in the call."""
    pl = Plan()
    pl.used_raw = sum(_tokens(r) for r in running)
    pl.incoming_raw = sum(_tokens(r) for r in admissible)
    if not enabled():
        return pl
    pl.on = True
    bk = book(sched)
    seen = set()
    # review R2 (nf-review-d-mem-share-1020): the credits ride an ELEMENT-WISE MIN, so the vector's slots must
    # be the same pairs on every rank. The queued part is taken in rid order (a set-determined order), never in
    # queue order -- a different queue order would put other pairs into the same slot or change the length.
    admissible = sorted(admissible, key=_rid)
    for k, r in enumerate(list(running) + list(admissible)):
        if id(r) not in seen:
            seen.add(id(r))
            pl.members.append(r)
            if k < len(running):
                pl.n_run += 1
    pl.rids = [_rid(r) for r in pl.members]
    bk.keep_only(set(pl.rids))
    page = max(1, int(page))
    for j in range(pl.n_run, len(pl.members)):
        cached = _front_cached(pl.members[j])
        if cached is None or cached < page:
            continue
        for i in range(j):
            # review R1: one tree path per (extra_key, token ids) -- other key, other pages (replicated field)
            if getattr(pl.members[i], "extra_key", None) != getattr(pl.members[j], "extra_key", None):
                continue
            if (bk.lcp_of(pl.rids[i], pl.rids[j], pl.members[i], pl.members[j]) // page) * page > 0:
                pl.cands.append((i, j))
                pl.local.append(_local_credit(bk, pl.members[i], pl.members[j], page, cap=cached))
    return pl


def finish(sched, pl: Plan, agreed: Optional[Sequence[int]]) -> Tuple[int, int, int, int]:
    """``(used, incoming, shared_used, shared_incoming)`` from the group's answer for the candidates (``agreed``,
    element-wise MIN; a wrong length credits nothing) and the book's earlier agreed values."""
    if not pl.on:
        return pl.used_raw, pl.incoming_raw, 0, 0
    bk = book(sched)
    now: Set[FrozenSet[str]] = set()
    if pl.cands:
        vals = list(agreed) if agreed is not None else []
        if len(vals) != len(pl.cands):
            vals = [0] * len(pl.cands)
        for (i, j), v in zip(pl.cands, vals):
            k = _key(pl.rids[i], pl.rids[j])
            bk.agreed[k] = max(0, int(v))
            now.add(k)

    def distinct(upto: int) -> Tuple[int, int]:
        total, shared = 0, 0
        for j in range(upto):
            c = 0
            for i in range(j):
                k = _key(pl.rids[i], pl.rids[j])
                # review R3: a QUEUED member is credited only by a pair agreed in THIS tick's call (so the
                # NO_TOKEN guard and its tree prefix are re-read); the book serves running pairs only. A
                # request parked / retracted back into the queue is no candidate -> it counts whole.
                if j >= pl.n_run and k not in now:
                    continue
                c = max(c, int(bk.agreed.get(k, 0)))
            t = _tokens(pl.members[j])
            c = min(c, t)
            total += t - c
            shared += c
        return total, shared

    used, shared_used = distinct(pl.n_run)
    every, shared_all = distinct(len(pl.members))
    incoming = every - used
    shared_in = shared_all - shared_used
    # the plain sums stay the bound: a credit only ever lowers the count
    used = min(used, pl.used_raw)
    incoming = max(0, min(incoming, pl.incoming_raw))
    key = (used, incoming, shared_used, shared_in)
    if (shared_used or shared_in) and key != bk.last:
        bk.last = key
        logger.info("%s used=%d incoming=%d shared_used=%d shared_incoming=%d raw_used=%d raw_incoming=%d "
                    "pairs=%d -- a prompt prefix two requests hold in ONE set of tree pages is counted once "
                    "(group-agreed credit)", MARK, used, incoming, shared_used, shared_in, pl.used_raw,
                    pl.incoming_raw, len(bk.agreed))
    return used, incoming, shared_used, shared_in


def census(sched, running: List, admissible: List, page: int, agree=None) -> Tuple[int, int, int, int]:
    """:func:`prepare` + ``agree(local credits)`` (the group's element-wise MIN; None = this rank alone) +
    :func:`finish`. The runtime path is ``d_seat_vram.runtime_tick``: it hands the credits to the reserve
    agreement's own call (``_agree_reserve``), so a queued tick still pays exactly ONE collective and this
    module reaches no collective itself."""
    pl = prepare(sched, running, admissible, page)
    agreed = None
    if pl.cands:
        agreed = list(agree(pl.local)) if callable(agree) else list(pl.local)
    return finish(sched, pl, agreed)
