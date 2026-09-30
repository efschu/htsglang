"""DUAL-TP3PP3 unified KV per card (B): group P's KV actor.

User orders 30.09.: "der unified kv soll doch geshared werden" (07:10Z) and
"wird vram kv knapp, pausiert P und gibt den context frei und das erarbeitete
in den L2 zur späteren weiterverwendung wenn vram kv wieder frei wird"
(07:25Z). P holds KV pages only while it prefills; otherwise its share of the
card's ONE KV pool (``card_kv_ledger``) is free for D.

The mechanics are D's #251c stage form, reused piece by piece (NF-OK 30.09.:
additive, no shared actor with D):

* the P pool's K/V buffers are born at their TOP range (VA only -- the graphs
  keep one address range) and trimmed right away to 0 tokens
  (``d_seat_vram.slot_spans`` on torch_memory_saver span maps, cut on a fixed
  token lattice so a live move never remaps a kept extent);
* ``ensure(tokens)`` asks the ledger for the bytes, maps the spans live, moves
  the pool's backed rows and the allocator cap (``KvRowCap``) up;
* ``release_all()`` -- only when no request holds a page (the caller flushes
  the radix tree first) -- synchronizes, unmaps to 0 tokens and returns the
  bytes to the ledger.

P never presses D: a request whose bytes are not granted waits (the front only
dispatches leg 1 when the cards have room, ``front._dual_pump``).
Armed only by ``SGLANG_WEG2_DUAL_LAYOUT=1`` + group P +
``SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS`` > 0.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MARK = "DUAL-TP3PP3 P-KV"
MAX_TOKENS_ENV = "SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS"
#: the token lattice every P KV span plan is cut at (a live move keeps whole
#: cells) and the unit ``ensure`` rounds up to
STEP_TOKENS_ENV = "SGLANG_WEG2_DUAL_P_KV_STEP_TOKENS"
STEP_TOKENS_DEFAULT = 4096

#: (data_ptr, KvTensorGeom) of the P KV buffers born trimmed, process-local
_P_BORN: List[Tuple[int, object]] = []


def armed(env=None) -> bool:
    env = os.environ if env is None else env
    if str(env.get("SGLANG_WEG2_DUAL_LAYOUT", "")).strip() != "1":
        return False
    if str(env.get("SGLANG_WEG2_GROUP", "")).strip().upper() != "P":
        return False
    try:
        return int(env.get(MAX_TOKENS_ENV, "0") or 0) > 0
    except ValueError:
        return False


def max_tokens(env=None) -> int:
    env = os.environ if env is None else env
    return int(env.get(MAX_TOKENS_ENV, "0") or 0)


def step_tokens(env=None) -> int:
    env = os.environ if env is None else env
    try:
        return max(1, int(env.get(STEP_TOKENS_ENV, STEP_TOKENS_DEFAULT) or STEP_TOKENS_DEFAULT))
    except ValueError:
        return STEP_TOKENS_DEFAULT


def pool_tokens(boot_tokens: int) -> int:
    """The P pool's rows: the TOP (VA) when armed, else unchanged."""
    return max(int(boot_tokens), max_tokens()) if armed() else int(boot_tokens)


def lattice(top_tokens: int, step: int) -> List[int]:
    return list(range(0, int(top_tokens) + int(step), int(step)))


def round_up(tokens: int, step: int) -> int:
    return -(-max(0, int(tokens)) // int(step)) * int(step)


def _geom_for(t, pool_size: int, page_size: int, name: str, info_size: int):
    from sglang.srt.weg2.d_seat_vram import KvTensorGeom, SlotTensorGeom

    rows = int(t.shape[0])
    slot_bytes = (t.numel() * t.element_size()) // max(1, rows)
    return KvTensorGeom(SlotTensorGeom(name, 1, rows, int(slot_bytes), int(info_size)),
                        token_ratio=1, token_pad=int(page_size))


def born(pool, t, name: str, *, spans=None, granule: Optional[int] = None):
    """A K/V buffer of P's pool keeps its TOP range and maps 0 tokens."""
    if not armed() or t is None or not t.numel():
        return t
    from sglang.srt.weg2.d_seat_vram import (
        Weg2DSeatVramRefused,
        _sync_before_unmap,
        granule_for,
        slot_spans,
        tms,
    )

    spans = tms() if spans is None else spans
    info = spans.info(t.data_ptr()) if spans.available else None
    if info is None:
        raise Weg2DSeatVramRefused(
            "%s: the KV tensor %s is not a saver allocation; its pages could not be given to the "
            "card pool. Turn %s off for this boot." % (MARK, name, MAX_TOKENS_ENV))
    g = int(granule or granule_for(t.device))
    geom = _geom_for(t, int(pool.size), int(pool.page_size), name, int(info.size))
    step = step_tokens()
    cuts = [geom.slots_for(k) for k in lattice(int(pool.size), step)]
    _sync_before_unmap(t)
    plan = slot_spans(geom.geom, geom.slots_for(0), g, cuts=cuts)
    rc = spans.set_spans(t.data_ptr(), plan, now=True)
    if rc != 0:
        raise Weg2DSeatVramRefused("%s: trimming %s to 0 tokens failed (tms_set_spans rc=%d)"
                                   % (MARK, name, rc))
    _P_BORN.append((int(t.data_ptr()), geom))
    pool.set_stage_backed_rows(geom.slots_for(0))
    return t


class Weg2DualKvCapBreach(RuntimeError):
    """A free id above the mapped span: the next allocation would write into
    unmapped memory (metal e3hgpw: illegal memory access). Refused by name."""


def engage_cap(allocator, tokens: int, page_size: int) -> int:
    """The allocator cap at ``tokens`` -- ``d_seat_vram._engage_kv_cap`` with the
    page count read correctly for EVERY allocator: the page-size-1
    TokenToKVPoolAllocator has no ``num_pages``, and the d_seat_vram helper then
    took 'no attribute' as 'the cap covers everything' and released it (metal
    e3hgpw 30.09.: freed ids at the back of the free list were handed out above
    the mapped span)."""
    from sglang.srt.managers.kv_backing_relief import KvRowCap

    page = max(1, int(page_size))
    num_pages = getattr(allocator, "num_pages", None)
    if num_pages is None:
        num_pages = int(getattr(allocator, "size", 0) or 0) // page
    cap = getattr(allocator, "_weg2_kv_stage_cap", None)
    if cap is None:
        cap = KvRowCap(allocator)
        allocator._weg2_kv_stage_cap = cap
    pages = int(tokens) // page
    if pages >= int(num_pages):
        if cap.engaged:
            cap.release()
        return pages
    if cap.engaged and cap.cap is not None and pages > int(cap.cap):
        cap.release()
    cap.engage(pages)
    return pages


def max_live_id(allocator, page_size: int = 1) -> int:
    """The highest id a request (or the tree) holds: every id in no free list
    and not withheld by the cap -- ``d_seat_vram.max_live_page`` with the page
    count read for EVERY allocator (the token allocator has no ``num_pages``;
    there the helper answered 0 and a shrink floor would have been blind)."""
    import torch

    n = getattr(allocator, "num_pages", None)
    if n is None:
        n = int(getattr(allocator, "size", 0) or 0) // max(1, int(page_size))
    n = int(n or 0)
    if n <= 0:
        return 0
    live = torch.ones(n + 1, dtype=torch.bool)
    live[0] = False
    for name in ("free_pages", "release_pages"):
        ids = getattr(allocator, name, None)
        if ids is not None and hasattr(ids, "numel") and ids.numel():
            live[ids.detach().to("cpu", torch.int64)] = False
    cap = getattr(allocator, "_weg2_kv_stage_cap", None)
    held = getattr(cap, "_withheld", None) if cap is not None else None
    if held is not None and held.numel():
        live[held.to("cpu", torch.int64)] = False
    idx = torch.nonzero(live).flatten()
    return int(idx.max()) if idx.numel() else 0


def check_cap(allocator, tokens: int, page_size: int, where: str) -> None:
    """Every free id must lie inside the mapped span ``[1, tokens + page]``."""
    import torch

    bound = (int(tokens) + int(page_size)) // max(1, int(page_size))
    for name in ("free_pages", "release_pages"):
        ids = getattr(allocator, name, None)
        if ids is not None and hasattr(ids, "numel") and ids.numel():
            hi = int(torch.max(ids).item())
            if hi > bound:
                raise Weg2DualKvCapBreach(
                    "%s CAP-BREACH at %s: free id %d in %s lies above the mapped span (tokens=%d, page=%d, "
                    "bound=%d) -- the next allocation would write into unmapped KV" % (
                        MARK, where, hi, name, int(tokens), int(page_size), bound))


def stage_pools(pool) -> List[object]:
    """The pools whose stage rows follow the mapping: the MHATokenToKVPool
    itself, or the inner FA pool of a HybridLinearKVPool (the 27B's
    token_to_kv_pool; the wrapper has no ``set_stage_backed_rows`` -- metal
    dgkpwa 30.09. died on exactly that). Same resolution as
    ``d_seat_vram.bound_stage_tokens``."""
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MHATokenToKVPool

    subs = [pool]
    if isinstance(pool, HybridLinearKVPool):
        subs.append(getattr(pool, "full_kv_pool", None))
    return [s for s in subs if isinstance(s, MHATokenToKVPool)]


class PKvStage:
    """This P rank's share of the card pool. ``born``: the registered tensors;
    ``ledger``: its ``CardKvLedger`` (group P); ``pools`` get their backed rows,
    ``allocator`` its cap."""

    def __init__(self, born_tensors: Sequence[Tuple[int, object]], ledger, *, allocator, pools,
                 page_size: int, granule: int, top_tokens: int, spans=None, step: Optional[int] = None,
                 engage_cap=None, sync=None):
        from sglang.srt.weg2 import d_seat_vram as _sv

        self.born = list(born_tensors)
        self.ledger = ledger
        self.allocator = allocator
        self.pools = list(pools)
        self.page = int(page_size)
        self.granule = int(granule)
        self.top = int(top_tokens)
        self.spans = _sv.tms() if spans is None else spans
        self.step = int(step or step_tokens())
        self._engage_cap = engage_cap or globals()["engage_cap"]
        self._sync = sync or (lambda: None)
        self.mapped_tokens = 0
        self.cuts = None

    def bytes_for(self, tokens: int) -> int:
        from sglang.srt.weg2.d_seat_vram import kv_mapped_bytes

        return int(kv_mapped_bytes([g for _p, g in self.born], int(tokens), self.granule))

    def _plan(self, g, tokens: int):
        from sglang.srt.weg2.d_seat_vram import slot_spans

        cuts = [g.slots_for(k) for k in lattice(self.top, self.step)]
        return slot_spans(g.geom, g.slots_for(tokens), self.granule, cuts=cuts)

    def _move(self, tokens: int) -> None:
        for ptr, g in self.born:
            rc = self.spans.set_spans(ptr, self._plan(g, tokens), now=True)
            if rc != 0:
                raise RuntimeError("%s: tms_set_spans rc=%d moving %s to %d tokens"
                                   % (MARK, rc, getattr(g.geom, "name", "?"), tokens))
        rows = max((g.slots_for(tokens) for _p, g in self.born), default=0)
        for pool in self.pools:
            pool.set_stage_backed_rows(rows)
        self._engage_cap(self.allocator, int(tokens), self.page)
        check_cap(self.allocator, int(tokens), self.page, "move to %d" % int(tokens))

    def ensure(self, tokens: int) -> bool:
        """Map at least ``tokens`` (rounded up to the lattice). True when the
        pages are there; False = not granted, nothing changed (P waits)."""
        want = min(self.top, round_up(tokens, self.step))
        if want <= self.mapped_tokens:
            return True
        need = self.bytes_for(want) - self.bytes_for(self.mapped_tokens)
        granted, _pressure = self.ledger.request(need)
        if granted < need:
            if granted:
                self.ledger.release(granted)
            logger.info("%s WAIT tokens=%d need=%d granted=%d -- the card pool has no room; P waits "
                        "(it never presses D)", MARK, want, need, granted)
            return False
        self._move(want)
        logger.info("%s GROW %d -> %d tokens (+%d B from the card pool)", MARK, self.mapped_tokens, want, need)
        self._committed = int(getattr(self, "_committed", 0) or 0) + need
        self.mapped_tokens = want
        return True

    def table(self) -> List[int]:
        """Bytes this rank maps at every lattice level -- published so PP0 can
        price the WHOLE group's grant (``stage_file``)."""
        return [self.bytes_for(k) - self.bytes_for(0) for k in lattice(self.top, self.step)]

    def map_granted(self, tokens: int, charged: Optional[int] = None) -> None:
        """Adopt a grant PP0 already committed on this card's ledger for this
        rank (the atomic group grant): map, no ledger request. The previous
        commitment of this rank is dropped here -- a grant is always fresh."""
        want = min(self.top, round_up(tokens, self.step))
        # PP0 charged this card for the whole grant; adopt it, then keep only
        # what the mapping needs. The mapping is ONE high-water level: a
        # second grant on the same level is not a second span (metal dual13
        # 3q33cu: weg2-0-21 and -22 on one 65536 mapping counted 2952790016 B
        # on the 5090, the next grant starved on P's own count). A grant still
        # in flight to a follower stays charged in the ledger until adopted,
        # so the ledger never covers less than the mapping.
        # ``charged``: what PP0 actually took on THIS card for the grant (its own
        # card: only the difference to its mapping); None = the full unit
        # (a follower card, charged in full by PP0)
        add = (self.bytes_for(want) - self.bytes_for(0)) if charged is None else max(0, int(charged))
        self._committed = int(getattr(self, "_committed", 0) or 0) + add
        if want > self.mapped_tokens:
            self._move(want)
            self.mapped_tokens = want
        keep = self.bytes_for(self.mapped_tokens) - self.bytes_for(0)
        excess = self._committed - keep
        if excess > 0:
            self.ledger.release(excess)
            self._committed = keep
        logger.info("%s MAPPED-BY-GRANT tokens=%d (PP0's atomic group grant) committed=%d B returned=%d B",
                    MARK, want, self._committed, max(0, excess))

    def release_all(self) -> int:
        """Unmap everything; the caller guarantees no request holds a page."""
        if self.mapped_tokens <= 0:
            return 0
        try:
            live = int(max_live_id(self.allocator, self.page))
        except Exception:  # noqa: BLE001 -- an allocator without free lists: nothing to judge by
            live = 0
        if live > 0:
            logger.warning("%s RELEASE HELD: page %d is still live -- no unmap under a live id", MARK, live)
            return 0
        n = int(getattr(self, "_committed", 0) or 0) or (self.bytes_for(self.mapped_tokens) - self.bytes_for(0))
        self._engage_cap(self.allocator, 0, self.page)
        self._sync()
        self._move(0)
        self.ledger.release(n)
        logger.info("%s RELEASE %d tokens -> 0 (-%d B back to the card pool)", MARK, self.mapped_tokens, n)
        self.mapped_tokens = 0
        self._committed = 0
        return n


# -- wiring (P process, dual only) -------------------------------------------

ACTOR_ATTR = "dual_p_kv"


def attach(runner) -> Optional["PKvStage"]:
    """After P's KV pool and allocator exist: join the card pool with P's boot
    KV (the bytes P would have kept, sized exactly as before), keep 0 mapped
    and cap the allocator to 0. None when not armed or nothing was born."""
    if not armed() or getattr(runner, "is_draft_worker", False) or not _P_BORN:
        return None
    import torch

    from sglang.srt.weg2 import d_seat_vram as _sv
    from sglang.srt.weg2.card_kv_ledger import CardKvLedger, ledger_path

    dev = torch.device("cuda", int(runner.gpu_id))
    card = str(torch.cuda.get_device_properties(dev).uuid)
    tag = os.environ.get("SGLANG_WEG2_DUAL_KV_TAG", "") or os.environ.get("SGLANG_WEG2_TAG", "weg2")
    ledger = CardKvLedger(ledger_path(tag, card), "P")
    pools = stage_pools(runner.token_to_kv_pool)
    actor = PKvStage(list(_P_BORN), ledger, allocator=runner.token_to_kv_pool_allocator, pools=pools,
                     page_size=int(runner.page_size), granule=_sv.granule_for(dev),
                     top_tokens=max_tokens(), sync=lambda: torch.cuda.synchronize(dev))
    boot = int(getattr(runner, "_dual_p_boot_tokens", 0) or 0)
    boot_bytes = actor.bytes_for(boot) - actor.bytes_for(0)
    ledger.contribute(boot_bytes, committed=0)
    actor._engage_cap(actor.allocator, 0, actor.page)
    setattr(runner, ACTOR_ATTR, actor)
    publish_stage(actor, tag, int(getattr(runner, "pp_rank", 0) or 0))
    logger.info("%s JOIN card=%s boot_tokens=%d contributed=%d B top=%d tokens step=%d -- P keeps 0 "
                "mapped; its KV is the card pool's", MARK, card[-12:], boot, boot_bytes, actor.top,
                actor.step)
    return actor


def _actor(sched) -> Optional["PKvStage"]:
    runner = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
    return getattr(runner, ACTOR_ATTR, None)


#: the told's wire attribute: the token level PP0 granted on EVERY card
WIRE_DUAL_KV = "dual_kv"


def stage_file(tag: str, pp_rank: int, root: str = "/dev/shm") -> str:
    import hashlib

    return os.path.join(root, "wkvs-%s-pp%d.json" % (hashlib.sha1(str(tag).encode()).hexdigest()[:10],
                                                      int(pp_rank)))


def publish_stage(actor: "PKvStage", tag: str, pp_rank: int) -> str:
    """This P rank's ledger and byte table, for PP0's group grant."""
    import json

    path = stage_file(tag, pp_rank)
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "w") as f:
        json.dump({"ledger": actor.ledger.path, "step": actor.step, "top": actor.top,
                   "bytes": actor.table()}, f)
    os.replace(tmp, path)
    return path


def group_grant(stages: Sequence[dict], tokens: int, open_ledger, covered: Optional[dict] = None) -> int:
    """ATOMIC over all P stages (operator order 30.09.: all or none, fixed card
    order against deadlocks): commit each stage's bytes for ``tokens`` on its
    card's ledger; if any card is short, return every grant already taken and
    answer 0. Returns the granted token level.

    ``covered`` {stage index: bytes}: what that stage's ledger ALREADY covers
    for its mapping -- only the difference is asked (operator order after
    dual13: a level the mapping covers never waits). Only PP0's own card is
    passed: PP0 cannot see a follower's mapping, and a follower may
    idle-release between this grant and its adoption, so a follower card is
    charged the full unit and returns the excess on adoption."""
    covered = covered or {}
    if not stages:
        return 0
    step = int(stages[0]["step"])
    top = min(int(s["top"]) for s in stages)
    want = min(top, round_up(tokens, step))
    k = want // step
    order = sorted(range(len(stages)), key=lambda i: str(stages[i]["ledger"]))
    taken = []
    for i in order:
        need = max(0, int(stages[i]["bytes"][k]) - int(covered.get(i, 0)))
        led = open_ledger(stages[i]["ledger"])
        got, _ = led.request(need)
        taken.append((led, got))
        if got < need:
            for l2, g2 in taken:
                if g2:
                    l2.release(g2)
            return 0
    return want


def pp0_grant(sched, req) -> Optional[int]:
    """PP0 only: the atomic group grant for ``req``'s prompt. None = not armed
    here (no actor / not PP0); 0 = a card is short (hold the request); else the
    granted token level (PP0 maps its own now; the told carries it)."""
    actor = _actor(sched)
    if actor is None or int(getattr(getattr(sched, "ps", None), "pp_rank", 0) or 0) != 0:
        return None
    import json

    from sglang.srt.weg2.card_kv_ledger import CardKvLedger

    tag = os.environ.get("SGLANG_WEG2_DUAL_KV_TAG", "") or os.environ.get("SGLANG_WEG2_TAG", "weg2")
    pp = int(getattr(getattr(sched, "ps", None), "pp_size", 1) or 1)
    stages = []
    for r in range(pp):
        try:
            with open(stage_file(tag, r)) as f:
                stages.append(json.load(f))
        except OSError:
            logger.warning("%s PP0 GRANT waits: stage %d has not published its table yet", MARK, r)
            return 0
    tokens = len(getattr(req, "origin_input_ids", None) or ()) + int(actor.page)
    own = int(getattr(actor, "_committed", 0) or 0)    # PP0's card: the ledger covers its mapping exactly
    lvl = group_grant(stages, tokens, lambda pth: CardKvLedger(pth, "P"), covered={0: own})
    rid = str(getattr(req, "rid", "?"))[:16]
    if lvl:
        k = lvl // int(stages[0]["step"])
        actor.map_granted(lvl, charged=max(0, int(stages[0]["bytes"][k]) - own))
        req._dual_kv_tokens = lvl
        waited = _wait_granted(rid)
        logger.info("%s PP0 GRANT rid=%s tokens=%d on all %d cards%s", MARK, rid, lvl, pp,
                    (" after %d waits over %.1f s" % waited) if waited else "")
    else:
        _log_wait(rid, tokens)
    return lvl


# -- WAIT log rate (metal dual13: 497k WAIT lines kept a stalled boot "alive") --
#: rid -> [first_t, next_t, interval_s, waits, suppressed]
_WAITS: dict = {}
_CENSUS = {"next": 0.0, "iv": 1.0, "waits": 0}


def _now() -> float:
    import time

    return time.monotonic()


def _reset_wait_log() -> None:
    _WAITS.clear()
    _CENSUS.update(next=0.0, iv=1.0, waits=0)


def _log_wait(rid: str, tokens: int) -> None:
    """At most one line per rid per second, the gap doubling each line, plus
    one census line on the same backoff. A long wait therefore goes quiet, and
    the silence watchdog can see a stall instead of a busy log."""
    t = _now()
    e = _WAITS.get(rid)
    if e is None:
        e = _WAITS[rid] = [t, t, 1.0, 0, 0]
    e[3] += 1
    _CENSUS["waits"] += 1
    if t >= e[1]:
        logger.info("%s PP0 WAIT rid=%s tokens=%d: a card is short, nothing held (P never presses D) "
                    "waits=%d suppressed=%d waiting_s=%.1f", MARK, rid, int(tokens), e[3], e[4], t - e[0])
        e[4] = 0
        e[1] = t + e[2]
        e[2] *= 2.0
    else:
        e[4] += 1
    if t >= _CENSUS["next"]:
        logger.info("%s PP0 WAIT census: rids=%d waits=%d since the last census line (next in %.0f s)",
                    MARK, len(_WAITS), _CENSUS["waits"], _CENSUS["iv"])
        _CENSUS["waits"] = 0
        _CENSUS["next"] = t + _CENSUS["iv"]
        _CENSUS["iv"] *= 2.0


def _wait_granted(rid: str):
    """Forget ``rid``'s wait; returns (waits, seconds) when it had waited."""
    e = _WAITS.pop(rid, None)
    if not _WAITS:
        _CENSUS.update(next=0.0, iv=1.0, waits=0)
    return (e[3], _now() - e[0]) if e else None


def on_told(sched, item) -> None:
    """A follower adopts PP0's group grant carried by the told."""
    lvl = int(getattr(item, WIRE_DUAL_KV, 0) or 0)
    actor = _actor(sched)
    if lvl > 0 and actor is not None:
        actor.map_granted(lvl)


def with_dual_kv(told, req):
    lvl = int(getattr(req, "_dual_kv_tokens", 0) or 0)
    if lvl > 0:
        setattr(told, WIRE_DUAL_KV, lvl)
    return told


def flush_acks_when_idle(sched) -> bool:
    """DUAL-TP3PP3 (metal qu97hh): an idle P stage drains its write-through
    acks. The batch path flushes them once per iteration; an idle PP stage runs
    only on_idle, so a finished tail's ack sat unprocessed for 25 s and D's
    store read stayed short until it gave up (W50 re-route, 66-71 s instead of
    28-31 s per long prompt). In the flip form P sleeps and its seam flushes;
    in the dual layout P never sleeps. Dual + group P + hicache only."""
    if str(os.environ.get("SGLANG_WEG2_DUAL_LAYOUT", "")).strip() != "1":
        return False
    if str(os.environ.get("SGLANG_WEG2_GROUP", "")).strip().upper() != "P":
        return False
    if not getattr(sched, "enable_hierarchical_cache", False):
        return False
    sched.tree_cache.flush_write_through_acks()
    return True


def on_idle(sched) -> int:
    """A fully idle P rank gives its whole context back: device tree evicted
    (write-back keeps the pages in L2), then unmapped and released."""
    actor = _actor(sched)
    if actor is None or actor.mapped_tokens <= 0:
        return 0
    tree = getattr(sched, "tree_cache", None)
    if tree is not None:
        try:
            from sglang.srt.mem_cache.base_prefix_cache import EvictParams

            ev = int(tree.evictable_size() or 0)
            if ev > 0:
                tree.evict(EvictParams(num_tokens=ev))
            if int(tree.evictable_size() or 0) > 0 or int(getattr(tree, "protected_size", lambda: 0)() or 0) > 0:
                return 0  # still held (write-back in flight / locked): next idle pass
        except Exception as exc:  # noqa: BLE001 -- a failed flush keeps the pages, never frees under them
            logger.warning("%s idle flush skipped: %r", MARK, exc)
            return 0
    return actor.release_all()
