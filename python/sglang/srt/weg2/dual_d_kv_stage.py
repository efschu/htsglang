"""DUAL-TP3PP3 unified KV per card (C): group D's KV actor.

D has priority on the card's ONE KV pool (user order 30.09. 07:25Z). D grows
from what is free and, when short, raises pressure on P through the card
ledger -- P pauses at its next chunk boundary and gives its context back
(front ``_dual_pause_inflight``, P actor ``dual_p_kv_stage``). D gives memory
back when its running work no longer needs it, so a waiting P prompt is not
starved by D's idle cache.

Why not ``d_seat_vram.apply_stage``: its cells are priced against D's BOOT
form ("the total never above the boot form") -- D could never grow into P's
share. The primitives are the same (born at a TOP range and trimmed,
``slot_spans`` on a fixed lattice, the allocator cap ``_engage_kv_cap``,
``max_live_page``); the decision is this module's (NF 30.09.: the stage
decision for the dual layout lives in the dual arbiter, not in MemSched).

GROUP-CONSISTENT: D's allocator is replicated over its TP ranks (uneven DCP:
each rank holds its owner share, ``KvTensorGeom.owner_block``). Every rank
computes the same wanted level from GLOBAL demand (``d_seat_vram.
global_demand``); the grant is local per card, so the ranks agree with one
MIN collective (``sched._weg2_group_min_ints``) -- entered only when the level
wants to move, which is decided from replicated inputs.
Armed only by ``SGLANG_WEG2_DUAL_LAYOUT=1`` + group D +
``SGLANG_WEG2_DUAL_D_KV_MAX_TOKENS`` > 0.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence, Tuple

from sglang.srt.weg2 import dual_p_kv_stage as _pk

logger = logging.getLogger(__name__)

MARK = "DUAL-TP3PP3 D-KV"
MAX_TOKENS_ENV = "SGLANG_WEG2_DUAL_D_KV_MAX_TOKENS"
#: rounds a shrink condition must hold before D gives memory back (no flap)
SHRINK_HOLD_ROUNDS = 64

_D_BORN: List[Tuple[int, object]] = []
#: D's boot level (GLOBAL tokens), set at _config_from_budget before the pool
#: is built; None = not the target pool (the draft's pool passes untouched)
_BOOT_TOKENS: Optional[int] = None


def armed(env=None) -> bool:
    env = os.environ if env is None else env
    if str(env.get("SGLANG_WEG2_DUAL_LAYOUT", "")).strip() != "1":
        return False
    if str(env.get("SGLANG_WEG2_GROUP", "")).strip().upper() != "D":
        return False
    try:
        return int(env.get(MAX_TOKENS_ENV, "0") or 0) > 0
    except ValueError:
        return False


def max_tokens(env=None) -> int:
    env = os.environ if env is None else env
    return int(env.get(MAX_TOKENS_ENV, "0") or 0)


def pool_tokens(boot_tokens: int) -> int:
    return max(int(boot_tokens), max_tokens()) if armed() else int(boot_tokens)


def _owner_block_for_rows(rows: int, top: int, page: int) -> Tuple[int, int]:
    from sglang.srt.weg2.d_seat_vram import kv_owner_block

    S, ratio = kv_owner_block()
    if S <= 0 or ratio <= 0 or ratio >= S:
        return (0, 0)
    compact = (int(top) // S + 1) * ratio
    return (S, ratio) if int(rows) in (compact, compact + int(page)) else (0, 0)


def born(pool, t, name: str, *, boot_tokens: Optional[int] = None, spans=None,
         granule: Optional[int] = None):
    """A D K/V buffer keeps its TOP range and D's BOOT level mapped. Only the
    target pool sized at the top (whole or compacted owner share) is touched."""
    if boot_tokens is None:
        boot_tokens = _BOOT_TOKENS
    if not armed() or t is None or not t.numel() or boot_tokens is None:
        return t
    top_ = max_tokens()
    size, page_ = int(getattr(pool, "size", 0) or 0), int(getattr(pool, "page_size", 1) or 1)
    if size < top_ and _owner_block_for_rows(size, top_, page_) == (0, 0):
        return t
    from sglang.srt.weg2.d_seat_vram import (
        KvTensorGeom,
        SlotTensorGeom,
        Weg2DSeatVramRefused,
        _sync_before_unmap,
        granule_for,
        slot_spans,
        tms,
    )

    spans = tms() if spans is None else spans
    info = spans.info(t.data_ptr()) if spans.available else None
    if info is None:
        raise Weg2DSeatVramRefused("%s: the KV tensor %s is not a saver allocation" % (MARK, name))
    g = int(granule or granule_for(t.device))
    rows = int(t.shape[0])
    slot_bytes = (t.numel() * t.element_size()) // max(1, rows)
    top = max_tokens()
    geom = KvTensorGeom(SlotTensorGeom(name, 1, rows, int(slot_bytes), int(info.size)),
                        token_ratio=1, token_pad=int(pool.page_size),
                        owner_block=_owner_block_for_rows(rows, top, int(pool.page_size)))
    step = _pk.step_tokens()
    boot = min(top, _pk.round_up(boot_tokens, step))
    cuts = [geom.slots_for(k) for k in _pk.lattice(top, step)]
    _sync_before_unmap(t)
    rc = spans.set_spans(t.data_ptr(), slot_spans(geom.geom, geom.slots_for(boot), g, cuts=cuts), now=True)
    if rc != 0:
        raise Weg2DSeatVramRefused("%s: trimming %s to %d tokens failed (rc=%d)" % (MARK, name, boot, rc))
    _D_BORN.append((int(t.data_ptr()), geom))
    pool.set_stage_backed_rows(geom.slots_for(boot))
    return t


class DKvStage(_pk.PKvStage):
    """D's share of its card. Same mapping mechanics as P's actor; the
    decision is group-wide (MIN over the D ranks) and D may press P."""

    def __init__(self, *a, gmin=None, **kw):
        super().__init__(*a, **kw)
        self.gmin = gmin or (lambda vals: list(vals))
        self._below = 0

    def group_grow(self, want: int) -> bool:
        """Every rank asks its card for the bytes up to ``want``; all map it or
        nobody does (MIN of the per-rank verdict). A short card's request left
        pressure on P there -- P pauses and releases, the next tick succeeds."""
        want = min(self.top, _pk.round_up(want, self.step))
        if want <= self.mapped_tokens:
            return True
        need = self.bytes_for(want) - self.bytes_for(self.mapped_tokens)
        granted, pressure = self.ledger.request(need)
        ok = 1 if granted >= need else 0
        if int(self.gmin([ok])[0]) != 1:
            if granted:
                self.ledger.release(granted)
            logger.info("%s GROUP-WAIT want=%d need=%d granted=%d pressure_on_P=%d", MARK, want, need,
                        granted, pressure)
            return False
        self._move(want)
        self._committed = int(getattr(self, "_committed", 0) or 0) + need
        logger.info("%s GROW %d -> %d tokens (+%d B)", MARK, self.mapped_tokens, want, need)
        self.mapped_tokens = want
        return True

    def group_shrink(self, target: int, live_floor_tokens: int) -> int:
        """Give back everything above ``target`` once no page above it is live
        (``live_floor_tokens``: the GROUP's highest live page, MAX over ranks).
        The allocator cap goes to ``target`` first, so new pages land below."""
        target = min(self.mapped_tokens, _pk.round_up(max(target, live_floor_tokens), self.step))
        if target >= self.mapped_tokens:
            return 0
        self._engage_cap(self.allocator, int(target), self.page)
        self._sync()
        n = self.bytes_for(self.mapped_tokens) - self.bytes_for(target)
        self._move(target)
        self.ledger.release(n)
        self._committed = max(0, int(getattr(self, "_committed", 0) or 0) - n)
        logger.info("%s SHRINK %d -> %d tokens (-%d B back to the card pool)", MARK, self.mapped_tokens,
                    target, n)
        self.mapped_tokens = target
        return n


def want_tokens(used: int, incoming: int, air: int, step: int) -> int:
    """The level D's running work needs: what the running requests hold, the
    queue head, and the decode/verify air -- on the lattice."""
    return _pk.round_up(max(0, int(used)) + max(0, int(incoming)) + max(0, int(air)), int(step))


def decide(mapped: int, want: int, p_waiting: bool, below_rounds: int, step: int,
           hold: int = SHRINK_HOLD_ROUNDS) -> Tuple[str, int]:
    """REPLICATED, pure: ('grow', want) / ('shrink', target) / ('hold', mapped),
    plus the new below-counter via the caller. Grow at once; shrink when a P
    prompt waits (D's cache must not starve it) or after ``hold`` rounds with
    at least two lattice steps of slack."""
    if want > mapped:
        return "grow", want
    slack = mapped - want
    if slack >= int(step) and (p_waiting or (slack >= 2 * int(step) and below_rounds >= int(hold))):
        return "shrink", want
    return "hold", mapped


# -- wiring (D process, dual only) -------------------------------------------

ACTOR_ATTR = "dual_d_kv"


def attach(runner) -> Optional[DKvStage]:
    if not armed() or getattr(runner, "is_draft_worker", False) or not _D_BORN:
        return None
    import torch

    from sglang.srt.weg2 import d_seat_vram as _sv
    from sglang.srt.weg2.card_kv_ledger import CardKvLedger, ledger_path

    dev = torch.device("cuda", int(runner.gpu_id))
    card = str(torch.cuda.get_device_properties(dev).uuid)
    tag = os.environ.get("SGLANG_WEG2_DUAL_KV_TAG", "") or os.environ.get("SGLANG_WEG2_TAG", "weg2")
    ledger = CardKvLedger(ledger_path(tag, card), "D")
    pools = _pk.stage_pools(runner.token_to_kv_pool)
    actor = DKvStage(list(_D_BORN), ledger, allocator=runner.token_to_kv_pool_allocator, pools=pools,
                     page_size=int(runner.page_size), granule=_sv.granule_for(dev), top_tokens=max_tokens(),
                     sync=lambda: torch.cuda.synchronize(dev))
    boot = min(actor.top, _pk.round_up(int(getattr(runner, "_dual_d_boot_tokens", 0) or 0), actor.step))
    boot_bytes = actor.bytes_for(boot) - actor.bytes_for(0)
    ledger.contribute(boot_bytes, committed=boot_bytes)
    actor.mapped_tokens = boot
    actor._committed = boot_bytes
    actor._engage_cap(actor.allocator, boot, actor.page)
    setattr(runner, ACTOR_ATTR, actor)
    logger.info("%s JOIN card=%s boot_tokens=%d contributed=%d B (kept mapped) top=%d", MARK, card[-12:],
                boot, boot_bytes, actor.top)
    return actor


def tick(sched) -> Optional[str]:
    """Once per scheduler iteration on every D rank (rank-symmetric inputs)."""
    runner = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
    actor = getattr(runner, ACTOR_ATTR, None)
    if actor is None:
        return None
    from sglang.srt.weg2 import d_seat_vram as _sv
    from sglang.srt.weg2.card_kv_ledger import peek

    actor.gmin = getattr(sched, "_weg2_group_min_ints", None) or actor.gmin
    used, incoming, _rids = _sv.global_demand(sched)
    want = want_tokens(used, incoming, _sv._air(sched), actor.step)
    st = peek(actor.ledger.path)
    # replicated enough: P's demand is card-local, so the shrink trigger takes
    # the group MAX of "a P prompt waits here" (one collective, every rank)
    p_wait_local = 1 if (st is not None and int(st.demand.get("P", 0)) > 0) else 0
    p_waiting = -int(actor.gmin([-p_wait_local])[0]) > 0
    below = actor._below + 1 if want < actor.mapped_tokens else 0
    verdict, level = decide(actor.mapped_tokens, want, p_waiting, below, actor.step)
    actor._below = below
    if verdict == "grow":
        actor.group_grow(level)
    elif verdict == "shrink":
        live = _sv.max_live_page(actor.allocator) * actor.page
        floor = -int(actor.gmin([-int(live)])[0])
        if actor.group_shrink(level, floor):
            actor._below = 0
    return verdict
