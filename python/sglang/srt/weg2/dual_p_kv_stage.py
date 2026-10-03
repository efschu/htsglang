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


class Weg2DualKvCoverBreach(RuntimeError):
    """The card ledger covers fewer bytes for this group than this rank maps
    (invariant: ledger committed >= mapped bytes, after every map/grant/release)."""


class Weg2DualKvMapShort(RuntimeError):
    """tms_set_spans rc=2 (cuMemCreate CUDA_ERROR_OUT_OF_MEMORY): the card is
    physically short -- D treats it as "card short", never as a crash."""

    def __init__(self, msg, rc=2):
        super().__init__(msg)
        self.rc = rc


class Weg2DualKvFollowerMapShort(Weg2DualKvMapShort):
    """DUAL-FOLLOWER-MAP-SHORT: a P follower (PP1/PP2) could not map the group
    grant PP0 already committed for it (its card physically short). PP0 cannot
    see a follower's card, so this rank cannot wait the grant away; it stops
    NAMED (residual risk of the MAP-SHORT WAIT, operator order 01.10.) -- the
    D-priority stages keep P's grants clear of a short card before it comes to
    this."""


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
            if rc == 2:
                raise Weg2DualKvMapShort("%s: tms_set_spans rc=2 (cuMemCreate out of memory) moving %s to %d "
                                         "tokens" % (MARK, getattr(g.geom, "name", "?"), tokens))
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
        check_cover(self, "ensure")
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
        if want > self.mapped_tokens:
            # MAP-SHORT WAIT (gmps7 dkr27bnvfp4dual1mbar1fs10011748, PP0 17:53:16):
            # the ledger granted 151552 tokens, the card was physically short
            # (cuMemCreate rc=2) and the raise killed PP0. Map FIRST: on a short
            # card this rank rolls back to its standing mapping and nothing is
            # adopted -- the caller (PP0's grant) returns every card's charge and
            # holds the request, as for a ledger-short card.
            try:
                self._move(want)
            except Weg2DualKvMapShort:
                self._move(self.mapped_tokens)        # unmap what this rank got; rows/cap back
                raise
            self.mapped_tokens = want
        self._committed = int(getattr(self, "_committed", 0) or 0) + add
        keep = self.bytes_for(self.mapped_tokens) - self.bytes_for(0)
        excess = self._committed - keep
        if excess > 0:
            self.ledger.release(excess)
            self._committed = keep
        logger.info("%s MAPPED-BY-GRANT tokens=%d (PP0's atomic group grant) committed=%d B returned=%d B",
                    MARK, want, self._committed, max(0, excess))
        check_cover(self, "map_granted")

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
        check_cover(self, "release_all")
        return n


def check_cover(actor, where: str) -> None:
    """Invariant after every map/grant/release (operator order after dual14):
    the card ledger's committed bytes for this group cover what this rank maps."""
    led = getattr(actor, "ledger", None)
    path = getattr(led, "path", None)
    if not path:
        return
    from sglang.srt.weg2.card_kv_ledger import peek

    st = peek(path)
    if st is None:
        return
    mine = actor.bytes_for(actor.mapped_tokens) - actor.bytes_for(0)
    have = int(st.committed.get(led.group, 0))
    if have < mine:
        raise Weg2DualKvCoverBreach(
            "%s COVER-BREACH at %s: the card ledger covers %d B for %s, this rank maps %d B (%d tokens) "
            "-- the pool would promise mapped bytes to the other group" % (MARK, where, have, led.group, mine,
                                                                            actor.mapped_tokens))


def phys_free_bytes() -> Optional[int]:
    """cuMemGetInfo's free bytes on this process's device (None off CUDA)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.mem_get_info()[0])
    except Exception:  # noqa: BLE001 -- an instrument never kills the rank
        return None


PHYS_CHECK_S = 10.0
#: consecutive OVER-PROMISE checks before the gap is booked into the budget
PHYS_BOOK_AFTER = 3


def phys_check(actor, group: str) -> None:
    """Throttled instrument (operator order after dual14): the ledger's free
    bytes against the card's physical free bytes. OVER-PROMISE = the ledger
    would grant bytes the card does not have (dual14: budget counted twice)."""
    t = _now()
    if t < getattr(actor, "_phys_next", 0.0):
        return
    actor._phys_next = t + PHYS_CHECK_S
    led = getattr(actor, "ledger", None)
    path = getattr(led, "path", None)
    phys = phys_free_bytes()
    if not path or phys is None:
        return
    from sglang.srt.weg2.card_kv_ledger import peek

    st = peek(path)
    if st is None:
        return
    over = int(st.free) - int(phys)
    win = actor.__dict__.setdefault("_phys_over", [])
    if over > 0:
        win.append(over)
        logger.warning("%s LEDGER-PHYS OVER-PROMISE by %d B group=%s ledger_free=%d phys_free=%d budget=%d "
                       "committed=%s", MARK, over, group, int(st.free), int(phys), int(st.budget),
                       dict(st.committed))
        if len(win) >= PHYS_BOOK_AFTER:
            # metal dual15: 628359168 B of card use outside the KV pools stood
            # for minutes. Persistent = real.
            # gmps7 (D 17:53:15): the window held 209/477/466 MB, the SMALLEST
            # (209 MB) was booked and the ledger still over-promised 466 MB a
            # second later -> PP0's cuMemCreate OOM. Persistence still decides
            # WHETHER to book; WHAT is booked is the gap measured now
            # (reconcile re-reads the ledger under its lock -- never twice).
            n = led.reconcile(phys, cap=over)
            win.clear()
            if n > 0:
                logger.warning("%s LEDGER-PHYS BOOKED %d B of unbooked card use into the budget (group=%s, "
                               "%d checks over %.0f s)", MARK, n, group, PHYS_BOOK_AFTER,
                               PHYS_BOOK_AFTER * PHYS_CHECK_S)
    else:
        win.clear()
        logger.info("%s LEDGER-PHYS ok group=%s ledger_free=%d phys_free=%d budget=%d committed=%s", MARK,
                    group, int(st.free), int(phys), int(st.budget), dict(st.committed))


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
    # D PRIORITY stage 2: the weights image this rank parks in host RAM at a sleep
    actor.weights_bytes = int(float(getattr(runner, "weight_load_mem_usage", 0) or 0) * (1 << 30))
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
                   "bytes": actor.table(),
                   # D PRIORITY stage 2: this rank's weights image (the host peak of a sleep)
                   "weights_bytes": int(getattr(actor, "weights_bytes", 0) or 0),
                   # D PRIORITY stage 2: what this rank lent its card pool while asleep (0
                   # awake) -- the front's wake check is per card: free >= lent + grant + air
                   "lent": int(getattr(actor, "_sleep_lent", 0) or 0)}, f)
    os.replace(tmp, path)
    return path


def group_grant(stages: Sequence[dict], tokens: int, open_ledger, covered: Optional[dict] = None,
                taken_out: Optional[list] = None) -> int:
    """ATOMIC over all P stages (operator order 30.09.: all or none, fixed card
    order against deadlocks): commit each stage's bytes for ``tokens`` on its
    card's ledger; if any card is short, return every grant already taken and
    answer 0. Returns the granted token level.

    ``covered`` {stage index: bytes}: what that stage's ledger ALREADY covers
    for its mapping -- only the difference is asked (operator order after
    dual13: a level the mapping covers never waits). Only PP0's own card is
    passed: PP0 cannot see a follower's mapping, and a follower may
    idle-release between this grant and its adoption, so a follower card is
    charged the full unit and returns the excess on adoption.

    ``taken_out``: on a grant, receives the (ledger, bytes) charges taken, so
    PP0 can return them when its own card is physically short (MAP-SHORT WAIT)."""
    covered = covered or {}
    if not stages:
        return 0
    step = int(stages[0]["step"])
    top = min(int(s["top"]) for s in stages)
    # NOTE (GRANT-SUM 1cd3c5ac00): ``tokens`` is the SUM of every held request's
    # tokens (live_grant_tokens), and min(top, ...) caps it at the stage top
    # (--dual-p-kv-max-tokens, 196608 on the 27B dual). Above top the level does
    # NOT cover the sum any more -- the normal admission (the SF load-back room,
    # the allocator) and ACK-ROOM (weg2_told_fallback._room_own) carry that case.
    # The sum is a sizing, never a guarantee.
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
    if taken_out is not None:
        taken_out.extend(taken)
    return want


def live_grant_tokens(sched, req, page: int = 1) -> int:
    """GRANT-SUM: tokens of every OTHER request on PP0 that holds a group grant
    and still occupies (or will occupy) rows -- the running batches, the chunked
    request, the queue and the store hold. Rank-uniform facts (the requests are
    the same on every stage), deduplicated by rid; a finished request is in
    none of these and its cached rows are evictable."""
    seen = {str(getattr(req, "rid", ""))}
    total = 0

    def _take(r) -> None:
        nonlocal total
        if r is None:
            return
        rid = str(getattr(r, "rid", ""))
        if not rid or rid in seen or not int(getattr(r, "_dual_kv_tokens", 0) or 0):
            return
        if getattr(r, "finished", None) and callable(r.finished) and r.finished():
            return
        seen.add(rid)
        ids = getattr(r, "origin_input_ids", None)
        total += (0 if ids is None else len(ids)) + int(page)

    batches = list(getattr(sched, "running_mbs", None) or ())
    batches.append(getattr(sched, "running_batch", None))
    batches.append(getattr(sched, "cur_batch", None))
    for b in batches:
        for r in list(getattr(b, "reqs", None) or ()):
            _take(r)
    _take(getattr(sched, "chunked_req", None))
    for r in list(getattr(sched, "waiting_queue", None) or ()):
        _take(r)
    for r in list((getattr(sched, "_weg2_store_held", None) or {}).values()):
        _take(r)
    return total


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
    _ids = getattr(req, "origin_input_ids", None)
    tokens = (0 if _ids is None else len(_ids)) + int(actor.page)   # never `x or ()` on a tensor
    # GRANT-SUM (dual1k 09:55:20Z, weg2-0-10): the mapping is ONE high-water
    # level shared by every request P holds -- a grant sized for this prompt
    # alone (20480) left the 61440 level of the concurrent weg2-0-9 prefill as
    # the whole pool, PP1 had to load back the twin head it held on the host
    # only and found avail=1717 (SF LOADBACK-ROOM PP-RESIDUAL) -> #968. The
    # level now covers this prompt PLUS every other request holding a grant.
    tokens += live_grant_tokens(sched, req, int(actor.page))
    own = int(getattr(actor, "_committed", 0) or 0)    # PP0's card: the ledger covers its mapping exactly
    taken: list = []
    lvl = group_grant(stages, tokens, lambda pth: CardKvLedger(pth, "P"), covered={0: own}, taken_out=taken)
    rid = str(getattr(req, "rid", "?"))[:16]
    if lvl:
        k = lvl // int(stages[0]["step"])
        try:
            actor.map_granted(lvl, charged=max(0, int(stages[0]["bytes"][k]) - own))
        except Weg2DualKvMapShort as exc:
            # MAP-SHORT WAIT: the card ledger promised bytes the card does not
            # have. Every card's charge goes back, PP0's ledger is reconciled
            # against cuMemGetInfo (the next grant is priced on what is really
            # there) and the request is HELD -- a wait, never a rank death.
            for led, got in taken:
                if got:
                    led.release(got)
            phys = phys_free_bytes()
            over = actor.ledger.reconcile(phys) if phys is not None else 0
            logger.warning("%s MAP-SHORT-WAIT rid=%s tokens=%d: %s -- every card's grant returned, PP0's "
                           "ledger reconciled by -%d B against phys_free=%s; the request is held",
                           MARK, rid, lvl, exc, int(over), phys)
            _log_wait(rid, tokens)
            return 0
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


class Weg2DualPWakeShort(RuntimeError):
    """W-DUAL-P-WAKE-SHORT: P was told to wake but the card pool cannot give
    back the bytes P lent it at its sleep (D committed into them). The front's
    hysteresis wakes P only when they are free; this is the named stop for the
    case it cannot see."""


def _sleep_armed(sched):
    if str(os.environ.get("SGLANG_WEG2_DUAL_LAYOUT", "")).strip() != "1":
        return None
    if str(os.environ.get("SGLANG_WEG2_GROUP", "")).strip().upper() != "P":
        return None
    return _actor(sched)


def sleep_phys_before(sched) -> Optional[int]:
    """D PRIORITY stage 2, P's sleep leg: cuMemGetInfo before the release (dual P
    only, else None -- the stock leg is untouched)."""
    return phys_free_bytes() if _sleep_armed(sched) is not None else None


def sleep_lend(sched, phys_before: Optional[int]) -> int:
    """After P's sleep leg released its weights: lend the freed bytes to the card
    pool (D's KV may grow into them while P sleeps)."""
    actor = _sleep_armed(sched)
    if actor is None or phys_before is None:
        return 0
    after = phys_free_bytes()
    if after is None:
        return 0
    freed = max(0, int(after) - int(phys_before))
    if freed:
        actor.ledger.lend(freed)
    actor._sleep_lent = int(getattr(actor, "_sleep_lent", 0) or 0) + freed
    _republish_stage(sched, actor)
    logger.warning("%s SLEEP-LEND freed=%d B -> the card pool while P sleeps (weights parked in host RAM, "
                   "freed at the wake)", MARK, freed)
    return freed


def _republish_stage(sched, actor) -> None:
    """The stage file again, with the current loan (the front reads it per card)."""
    try:
        runner = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
        pp_rank = int(getattr(runner, "pp_rank", 0) or 0)
        tag = os.environ.get("SGLANG_WEG2_DUAL_KV_TAG", "") or os.environ.get("SGLANG_WEG2_TAG", "weg2")
        publish_stage(actor, tag, pp_rank)
    except Exception as exc:  # noqa: BLE001 -- the front then waits on the old file: P stays asleep, never wrong
        logger.warning("%s stage file not republished after the loan changed: %r", MARK, exc)


def wake_reclaim(sched) -> int:
    """Before P's wake maps its weights: take the loan back, or stop named."""
    actor = _sleep_armed(sched)
    lent = int(getattr(actor, "_sleep_lent", 0) or 0) if actor is not None else 0
    if lent <= 0:
        return 0
    if not actor.ledger.reclaim(lent):
        raise Weg2DualPWakeShort(
            "W-DUAL-P-WAKE-SHORT: P's wake needs back the %d B it lent the card pool at its sleep, the pool "
            "has %d B free -- D committed into the loan" % (lent, int(actor.ledger.state().free)))
    actor._sleep_lent = 0
    _republish_stage(sched, actor)
    logger.warning("%s WAKE-RECLAIM %d B back from the card pool before P maps its weights", MARK, lent)
    return lent


def on_told(sched, item) -> None:
    """A follower adopts PP0's group grant carried by the told."""
    lvl = int(getattr(item, WIRE_DUAL_KV, 0) or 0)
    actor = _actor(sched)
    if lvl > 0 and actor is not None:
        try:
            actor.map_granted(lvl)
        except Weg2DualKvFollowerMapShort:
            raise
        except Weg2DualKvMapShort as exc:
            raise Weg2DualKvFollowerMapShort(
                "DUAL-FOLLOWER-MAP-SHORT rid=%s level=%d mapped=%d: %s -- PP0's committed group grant "
                "cannot be mapped on this follower's card (rolled back to the standing mapping)"
                % (str(getattr(item, "rid", "?"))[:16], lvl, actor.mapped_tokens, exc)) from exc


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


def _idle_marker(tag: str, root: str = "/dev/shm") -> str:
    import hashlib

    return os.path.join(root, "wkvs-%s-pp0-idle" % hashlib.sha1(str(tag).encode()).hexdigest()[:10])


def _dual_tag() -> str:
    return os.environ.get("SGLANG_WEG2_DUAL_KV_TAG", "") or os.environ.get("SGLANG_WEG2_TAG", "weg2")


def mark_pp0_idle(sched, now: Optional[float] = None) -> bool:
    """PP0, fully idle (dual P): stamp the time. A fully idle PP0 has applied its
    own aborts and every pass it launched completed the ring -- the same fact
    the #1268 lap's idle PP0 slot carries (the #791C liveness release), which
    the dual layout never runs (no flip, no quiesce). Throttled to 1/s."""
    import time as _t

    if int(getattr(getattr(sched, "ps", None), "pp_rank", 0) or 0) != 0:
        return False
    t = _t.time() if now is None else float(now)
    if t - float(getattr(sched, "_dual_pp0_idle_t", 0.0) or 0.0) < 1.0:
        return False
    sched._dual_pp0_idle_t = t
    path = _idle_marker(_dual_tag())
    try:
        tmp = "%s.%d.tmp" % (path, os.getpid())
        with open(tmp, "w") as f:
            # metal dual22: "idle" alone is not ordered with the ring -- PP0 was idle while
            # its last frame for the aborted rid (fwd 340) still sat in PP2's inbox. The
            # stamp carries PP0's forward count: a follower applies only once it has
            # executed every pass PP0 launched.
            f.write("%.6f %d" % (t, int(getattr(sched, "forward_ct", 0) or 0)))
        os.replace(tmp, path)
    except OSError:
        return False
    return True


def follower_release_aborted_chunk(sched, now: Optional[float] = None) -> bool:
    """A dual P FOLLOWER holding a recorded chunked abort applies it once PP0 is
    idle AFTER the abort reached this rank. Metal dual20: the front's pause
    aborted B on all stages; PP0 applied it and released, PP1/PP2 kept B as
    chunked_req ("applied when PP0's forwarded schedule stops naming it") --
    PP0 sent no further frame, so they never did: never idle, never released,
    855638016 B stayed committed on a 3080 and D's grow waited 45 s for it."""
    import time as _t

    if str(os.environ.get("SGLANG_WEG2_DUAL_LAYOUT", "")).strip() != "1":
        return False
    if str(os.environ.get("SGLANG_WEG2_GROUP", "")).strip().upper() != "P":
        return False
    if int(getattr(getattr(sched, "ps", None), "pp_rank", 0) or 0) == 0:
        return False
    req = getattr(sched, "_pending_chunked_abort_req", None)
    # metal dual1m-psleep (boot ...10020527, 05:36:22Z, weg2-0-73): the front's pause aborted a
    # request that PP1/PP2 still held in their WAITING queue; the #1180-W hold
    # (``_weg2_pending_waiting_aborts``) waits for PP0's forwarded schedule to decide it, PP0 sent
    # no further frame, and this release looked at the chunked abort only: 352321536 B / 528482304 B
    # stayed committed on two cards for 271 s, the front's RESUME-WAIT never saw all zeros, six
    # requests sat until the operator tore the boot down. A waiting hold is a held abort too.
    held = tuple(sorted(str(r) for r in (getattr(sched, "_weg2_pending_waiting_aborts", None) or ())))
    if req is None and not held:
        sched._dual_abort_seen = None
        return False
    t = _t.time() if now is None else float(now)
    seen = getattr(sched, "_dual_abort_seen", None)
    if seen is None or seen[0] is not req or not set(held) <= set(seen[2]):
        sched._dual_abort_seen = (req, t, held)
        return False
    try:
        with open(_idle_marker(_dual_tag())) as f:
            parts = f.read().split()
        idle_t = float(parts[0]) if parts else 0.0
        pp0_fwd = int(parts[1]) if len(parts) > 1 else None
    except (OSError, ValueError):
        return False
    if idle_t <= seen[1]:
        return False  # PP0 has not been idle since the abort reached this rank
    if pp0_fwd is None:
        return False  # an old stamp without PP0's pass count: not ordered with the ring
    if int(getattr(sched, "forward_ct", 0) or 0) < pp0_fwd:
        return False  # a pass PP0 launched (it may name the rid) has not run here yet
    drained = getattr(sched, "_pp_microbatches_drained", None)
    if callable(drained) and not drained():
        return False
    sched._791c_pp0_drained = True
    try:
        sched.process_pending_chunked_abort()
    finally:
        sched._791c_pp0_drained = False
    logger.info("%s FOLLOWER-ABORT-APPLIED rid=%s pp_rank=%s: PP0 idle since %.1f s after this rank "
                "saw the abort, every pass it launched ran here (fwd %d >= %d; the #791C liveness "
                "release the dual layout has no lap for)", MARK,
                str(getattr(req, "rid", None) or ",".join(held) or "?")[:48], getattr(sched.ps, "pp_rank", "?"),
                idle_t - seen[1],
                int(getattr(sched, "forward_ct", 0) or 0), pp0_fwd)
    return True


def on_idle(sched) -> int:
    """A fully idle P rank gives its whole context back: device tree evicted
    (write-back keeps the pages in L2), then unmapped and released."""
    if str(os.environ.get("SGLANG_WEG2_DUAL_LAYOUT", "")).strip() == "1":
        mark_pp0_idle(sched)
    actor = _actor(sched)
    if actor is not None:
        phys_check(actor, "P")
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
