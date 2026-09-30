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
        self._engage_cap = engage_cap or _sv._engage_kv_cap
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

    def map_granted(self, tokens: int) -> None:
        """Adopt a grant PP0 already committed on this card's ledger for this
        rank (the atomic group grant): map, no ledger request. The previous
        commitment of this rank is dropped here -- a grant is always fresh."""
        want = min(self.top, round_up(tokens, self.step))
        # the commitments ACCUMULATE until the idle release: a previous
        # request may still hold pages above ``want`` on this stage, so the
        # ledger keeps covering the mapping (over-commit is the safe side of I1)
        self._committed = int(getattr(self, "_committed", 0) or 0) + (self.bytes_for(want) - self.bytes_for(0))
        if want > self.mapped_tokens:
            self._move(want)
            self.mapped_tokens = want
        logger.info("%s MAPPED-BY-GRANT tokens=%d (PP0's atomic group grant) committed=%d B",
                    MARK, want, self._committed)

    def release_all(self) -> int:
        """Unmap everything; the caller guarantees no request holds a page."""
        if self.mapped_tokens <= 0:
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
    pool = runner.token_to_kv_pool
    pools = [pool] + ([pool.full_kv_pool] if hasattr(pool, "full_kv_pool") else [])
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


def group_grant(stages: Sequence[dict], tokens: int, open_ledger) -> int:
    """ATOMIC over all P stages (operator order 30.09.: all or none, fixed card
    order against deadlocks): commit each stage's bytes for ``tokens`` on its
    card's ledger; if any card is short, return every grant already taken and
    answer 0. Returns the granted token level."""
    if not stages:
        return 0
    step = int(stages[0]["step"])
    top = min(int(s["top"]) for s in stages)
    want = min(top, round_up(tokens, step))
    k = want // step
    order = sorted(range(len(stages)), key=lambda i: str(stages[i]["ledger"]))
    taken = []
    for i in order:
        need = int(stages[i]["bytes"][k])
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
    lvl = group_grant(stages, tokens, lambda pth: CardKvLedger(pth, "P"))
    if lvl:
        actor.map_granted(lvl)
        req._dual_kv_tokens = lvl
        logger.info("%s PP0 GRANT rid=%s tokens=%d on all %d cards", MARK,
                    str(getattr(req, "rid", "?"))[:16], lvl, pp)
    else:
        logger.info("%s PP0 WAIT rid=%s tokens=%d: a card is short, nothing held (P never presses D)",
                    MARK, str(getattr(req, "rid", "?"))[:16], tokens)
    return lvl


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
