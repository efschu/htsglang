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

from sglang.srt.weg2 import dual_parallel as _dpar

logger = logging.getLogger(__name__)

MARK = "DUAL-TP3PP3 P-KV"
MAX_TOKENS_ENV = "SGLANG_WEG2_DUAL_P_KV_MAX_TOKENS"
#: WEG2-ALLOC-CACHE-BOOK (item 170): MiB per card ordinal (P stage s on card s) of D's allocator
#: cache growth the card ledger books at P's join (launcher: record D_DUAL_ALLOC_CACHE_BOOK_MIB)
ALLOC_CACHE_BOOK_ENV = "SGLANG_WEG2_DUAL_ALLOC_CACHE_BOOK_MIB"
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
    book_alloc_cache(ledger, int(getattr(runner, "pp_rank", 0) or 0), card)
    actor._engage_cap(actor.allocator, 0, actor.page)
    setattr(runner, ACTOR_ATTR, actor)
    # #1962 P-LAYER-STREAM (PP0, default off): weights out, KV in, by the prompt's level
    from sglang.srt.weg2 import p_layer_stream as _pls

    if _pls.armed() and int(getattr(runner, "pp_rank", 0) or 0) == 0:
        try:
            actor.streamer = _pls.build_for_runner(runner)
        except Exception as exc:  # noqa: BLE001 -- no streamer = the grant path as before, never a death
            logger.warning("%s not built: %r -- PP0 grants as before", _pls.MARK, exc)
            actor.streamer = None
    publish_stage(actor, tag, int(getattr(runner, "pp_rank", 0) or 0))
    logger.info("%s JOIN card=%s boot_tokens=%d contributed=%d B top=%d tokens step=%d -- P keeps 0 "
                "mapped; its KV is the card pool's", MARK, card[-12:], boot, boot_bytes, actor.top,
                actor.step)
    return actor


def alloc_cache_book_mib(pp_rank: int, env=None) -> Optional[int]:
    """The MiB the launcher's record books on this stage's card (``None``: no
    record / this card unpriced -- nothing is booked, by name)."""
    env = os.environ if env is None else env
    raw = str(env.get(ALLOC_CACHE_BOOK_ENV, "") or "").strip()
    if not raw:
        return None
    parts = [x.strip() for x in raw.split(",")]
    if not (0 <= int(pp_rank) < len(parts)) or not parts[int(pp_rank)]:
        return None
    try:
        return max(0, int(parts[int(pp_rank)]))
    except ValueError:
        return None


def book_alloc_cache(ledger, pp_rank: int, card: str, env=None) -> int:
    """WEG2-ALLOC-CACHE-BOOK: after P's boot KV joined the card pool, book D's
    allocator cache growth (a measured record, same checkpoint and form) into
    the ledger -- priced at the join, not after three OVER-PROMISE checks and a
    rolled-back grow (dual mpsleep 10020527: 343 MiB over-promise, CORRIDOR LAW
    BREACHED min 56 MiB on the 5090). Returns the bytes booked."""
    mib = alloc_cache_book_mib(pp_rank, env)
    if mib is None:
        logger.info("%s ALLOC-CACHE-BOOK card=%s pp_rank=%d: UNMEASURED (no record prices this card), "
                    "nothing booked", MARK, str(card)[-12:], int(pp_rank))
        return 0
    want = mib << 20
    got = ledger.book_unpriced(want) if want > 0 else 0
    logger.info("%s ALLOC-CACHE-BOOK card=%s pp_rank=%d: booked %d of %d B (D's allocator-cache growth, "
                "record D_DUAL_ALLOC_CACHE_BOOK_MIB) into the card budget%s", MARK, str(card)[-12:],
                int(pp_rank), got, want, "" if got == want else " -- the rest does not fit under what is "
                "committed (I1) and stays unbooked")
    return got


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
                   # D PRIORITY: what this rank lent its card pool -- asleep (stage 2) plus
                   # awake (Q-660, stage 1) -- the front's return check is per card:
                   # free >= lent + grant + air
                   "lent": lent_bytes(actor)}, f)
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

    ``taken_out``: on a grant, receives the (stage index, ledger, bytes) charges
    taken, so PP0 can return them when its own card is physically short
    (MAP-SHORT WAIT) or when the told that carries them never leaves (Q-630)."""
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
        taken.append((i, led, got))
        if got < need:
            for _i2, l2, g2 in taken:
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


# -- #1530 GRANT-RETRY throttle (SGLANG_WEG2_DUAL_GRANT_RETRY_MS, default 0 = off) ----------
#: stage-file path -> (mtime_ns, size, parsed table): a stage table changes only at publish_stage
_STAGE_CACHE: dict = {}
#: rid -> monotonic time of its last attempt
_RETRY: dict = {}


def _retry_ms() -> int:
    """The throttle interval in ms; 0 (default) = every pass retries, exactly as before."""
    try:
        from sglang.srt.environ import envs

        return max(0, int(envs.SGLANG_WEG2_DUAL_GRANT_RETRY_MS.get()))
    except Exception:  # noqa: BLE001 -- a bad value never changes the grant path
        return 0


def _load_stage(path: str, cached: bool) -> dict:
    """One stage table. ``cached``: parsed once per (mtime, size) of the file
    (publish_stage writes tmp + os.replace, so a new table is a new inode/mtime)."""
    import json

    if not cached:
        with open(path) as f:
            return json.load(f)
    st = os.stat(path)
    key = (st.st_mtime_ns, st.st_size)
    hit = _STAGE_CACHE.get(path)
    if hit is not None and hit[0] == key:
        return hit[1]
    with open(path) as f:
        tab = json.load(f)
    _STAGE_CACHE[path] = (key, tab)
    return tab


def _retry_throttled(rid: str, ms: int, now: float) -> bool:
    """True = skip this attempt: the rid tried less than ``ms`` ms ago. A pure minimum interval:
    no ledger signal lifts it (b9h: D's ledger changes on every tick, a signature check held the
    throttle open at ~480 attempts/s). Never longer than ``ms`` (no starvation); the first attempt
    of a rid always runs."""
    e = _RETRY.get(rid)
    if e is None:
        return False
    return (now - e) * 1000.0 < ms


def _retry_note(rid: str, now: float) -> None:
    if len(_RETRY) > 4096:
        _RETRY.clear()
    _RETRY[rid] = now


#: #1640 rid -> True while its LAST evaluated grant was infeasible even with D at zero (PP0-local)
_INFEASIBLE: dict = {}


def _infeasible_skip_armed() -> bool:
    """SGLANG_WEG2_DUAL_GRANT_INFEASIBLE_SKIP (default off): ``_older_waits`` ignores infeasible heads."""
    try:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_WEG2_DUAL_GRANT_INFEASIBLE_SKIP.get())
    except Exception:  # noqa: BLE001 -- a bad value never changes the grant path
        return False


def infeasible_cards(stages, covered, level_tokens: int, stream_room: int = 0) -> list:
    """[(card, need, have, P, D)] of every card on which this grant stays short even if D gave back
    everything it holds there (need > ledger free + D committed). ``need`` is what ``group_grant``
    asks of that card (the level's bytes less what the ledger already covers for PP0's own mapping).
    Reads only (ledger peek); any failure answers [] ('feasible'), never an exception into the round."""
    try:
        from sglang.srt.weg2.card_kv_ledger import peek

        step = int(stages[0]["step"])
        top = min(int(s["top"]) for s in stages)
        k = min(top, round_up(int(level_tokens), step)) // step
        out = []
        for i, s in enumerate(stages):
            need = max(0, int(s["bytes"][k]) - int(covered.get(i, 0)))
            st = peek(s["ledger"])
            if st is None:
                continue
            have = int(st.free)
            d = int(st.committed.get("D", 0))
            # #1962: what PP0 could still free of its own weights counts on its own card (0 = off)
            if need > have + d + (int(stream_room) if i == 0 else 0):
                out.append((i, need, have, int(st.committed.get("P", 0)), d))
        return out
    except Exception:  # noqa: BLE001
        return []


#: #1920 head rid -> the level (tokens) of its LAST evaluated grant attempt (PP0-local; only kept while
#: SGLANG_WEG2_DUAL_HEAD_BYPASS_FLOOR is on)
_HEAD_LV: dict = {}
#: #1920 head rid -> how many younger grants passed it under the floor rule (per head, never reset by the rule)
_OVERTAKERS: dict = {}
#: #1920 [monotonic t of the last read, D's floor flag] -- one tiny /dev/shm read per FLOOR_CACHE_S at most
_FLOOR_CACHE: list = [None, False]
FLOOR_CACHE_S = 0.25
FLOOR_MARK = "#1920 HEAD-BYPASS-FLOOR"


def _stream_room(actor) -> int:
    """#1962: bytes PP0 could still free by pausing weight units (0 without a streamer)."""
    st = getattr(actor, "streamer", None)
    return int(st.room()) if st is not None else 0


def head_grantable_now(stages, covered, level_tokens: int) -> bool:
    """True when ``level_tokens``' group grant fits the ledgers' FREE bytes on every card right now (what the
    head's own next attempt would see). Anything unreadable answers True: the head is protected, never
    overtaken on a guess."""
    try:
        from sglang.srt.weg2.card_kv_ledger import peek

        step = int(stages[0]["step"])
        top = min(int(s["top"]) for s in stages)
        k = min(top, round_up(int(level_tokens), step)) // step
        for i, s in enumerate(stages):
            need = max(0, int(s["bytes"][k]) - int(covered.get(i, 0)))
            st = peek(s["ledger"])
            if st is None:
                return True
            if need > int(st.free):
                return False
        return True
    except Exception:  # noqa: BLE001
        return True


def d_floor_blocked(tag: str) -> bool:
    """D's published flag 'the live floor keeps the shrink from the waiting P' (dual_d_priority), fresh only.
    Cached for FLOOR_CACHE_S; a failure answers False (no bypass)."""
    try:
        t = _now()
        if _FLOOR_CACHE[0] is not None and t - float(_FLOOR_CACHE[0]) < FLOOR_CACHE_S:
            return bool(_FLOOR_CACHE[1])
        import time

        from sglang.srt.weg2 import dual_d_priority as _ddp

        v = _ddp.floor_signal_blocked(_ddp.read_floor_signal(_ddp.floor_signal_file(tag)), now=time.time())
        _FLOOR_CACHE[0], _FLOOR_CACHE[1] = t, bool(v)
        return bool(v)
    except Exception:  # noqa: BLE001
        return False


def floor_overtake_head(older, stages, own: int, tag: str) -> Optional[str]:
    """#1920: the head rid a younger grant may pass although the head has waited past the head age, or
    None (hold=older-head as before). ALL must hold: the switch is on; the head's overtaker budget is not
    spent; its last attempted level is known; D reports the live floor as what blocks the shrink; and the
    head's own grant is NOT satisfiable from the free bytes now (a head that can be served keeps
    its priority). PP0-local: the grant verdict rides the told, no rank decides anything else."""
    if not older or not _dpar.head_bypass_floor_armed():
        return None
    head, _since = min(older, key=lambda x: x[1])
    if int(_OVERTAKERS.get(head, 0)) >= _dpar.head_bypass_floor_max():
        return None
    lv = _HEAD_LV.get(head)
    if lv is None:
        return None
    if not d_floor_blocked(tag):
        return None
    if head_grantable_now(stages, {0: own}, lv):
        return None
    return head


def _infeasible_line(rid: str, tokens: int, level_tokens: int, cards: list) -> str:
    """'#1640 GRANT-INFEASIBLE' body: per short card the bytes asked and the pool (free + P + D).

    Text only (deskq 1977): ``tokens`` is the GRANT-SUM of the prompts of this request and every other request
    holding a stage (``pp0_grant`` / ``live_grant_tokens``), NOT this request's own prompt; ``level_tokens`` is
    min(sum, top). The verdict (``infeasible_cards``) compares need with have + D and leaves out ``P`` -- the
    follower pre-charge of grants just taken, which comes back when the follower adopts (about 1 s). When P
    closes the gap on every listed card the line says 'waits for the return', not 'only a falling level frees it'."""
    waits_p = bool(cards) and all(have + p + d >= need for _i, need, have, p, d in cards)
    why = ("the grant waits for the return of the followers' pre-charge of just-taken grants (P covers the gap; "
           "it frees on adoption, not by a falling level)" if waits_p else
           "the grant stays short even with D at zero and P returned (only a falling level or a lower grant sum frees it)")
    return "rid=%s tokens=%d(GRANT-SUM of all held prompts, not this prompt) level_tokens=%d cards=[%s]: %s" % (
        rid, int(tokens), int(level_tokens),
        " ".join("%d:need=%d,pool=%d,have=%d,P=%d,D=%d" % (i, need, have + p + d, have, p, d)
                 for i, need, have, p, d in cards), why)


def _grant_short_detail(stages, covered, level_tokens: int, sum_tokens: int, older: int, hold) -> str:
    """'#1530 GRANT-SHORT' body: per card the bytes this grant asks and what the ledger has free.
    A log line only: any failure returns a short note, never an exception into the round."""
    try:
        from sglang.srt.weg2.card_kv_ledger import peek

        step = int(stages[0]["step"])
        top = min(int(s["top"]) for s in stages)
        k = min(top, round_up(int(level_tokens), step)) // step
        cards = []
        for i, s in enumerate(stages):
            need = max(0, int(s["bytes"][k]) - int(covered.get(i, 0)))
            st = peek(s["ledger"])
            have = "?" if st is None else str(int(st.free))
            cards.append("%d:need=%d,have=%s%s" % (
                i, need, have, "" if st is None else ",P=%d,D=%d,demD=%d,presP=%d" % (
                    int(st.committed.get("P", 0)), int(st.committed.get("D", 0)),
                    int(st.demand.get("D", 0)), int(st.pressure.get("P", 0)))))
        return "level=%d sum=%d older=%d hold=%s cards=[%s]" % (
            int(level_tokens), int(sum_tokens), int(older), hold or "none", " ".join(cards))
    except Exception as exc:  # noqa: BLE001
        return "detail failed: %s: %s" % (type(exc).__name__, exc)


def pp0_grant(sched, req) -> Optional[int]:
    """PP0 only: the atomic group grant for ``req``'s prompt. None = not armed
    here (no actor / not PP0); 0 = a card is short (hold the request); else the
    granted token level (PP0 maps its own now; the told carries it)."""
    actor = _actor(sched)
    if actor is None or int(getattr(getattr(sched, "ps", None), "pp_rank", 0) or 0) != 0:
        return None
    from sglang.srt.weg2.card_kv_ledger import CardKvLedger

    tag = os.environ.get("SGLANG_WEG2_DUAL_KV_TAG", "") or os.environ.get("SGLANG_WEG2_TAG", "weg2")
    pp = int(getattr(getattr(sched, "ps", None), "pp_size", 1) or 1)
    _ms = _retry_ms() if _dual_layout_env() else 0       # #1530: dual layout only, 0 = off
    stages = []
    for r in range(pp):
        try:
            stages.append(_load_stage(stage_file(tag, r), _ms > 0))
        except OSError:
            logger.warning("%s PP0 GRANT waits: stage %d has not published its table yet", MARK, r)
            return 0
    if _ms > 0:
        _t_now = _now()
        _rid0 = str(getattr(req, "rid", "?"))[:16]
        if _retry_throttled(_rid0, _ms, _t_now):
            return 0                                      # still held; at most one attempt per N ms
        _retry_note(_rid0, _t_now)
    # Q-630: a re-intake of a request whose earlier grant never reached the
    # followers (intake_stall/abort before the told) -- that grant is returned
    # before a new one is taken, never charged twice on a follower card.
    return_untold_grant(sched, req, "regrant")
    _ids = getattr(req, "origin_input_ids", None)
    tokens = (0 if _ids is None else len(_ids)) + int(actor.page)   # never `x or ()` on a tensor
    # GRANT-SUM (dual1k 09:55:20Z, weg2-0-10): the mapping is ONE high-water
    # level shared by every request P holds -- a grant sized for this prompt
    # alone (20480) left the 61440 level of the concurrent weg2-0-9 prefill as
    # the whole pool, PP1 had to load back the twin head it held on the host
    # only and found avail=1717 (SF LOADBACK-ROOM PP-RESIDUAL) -> #968. The
    # level now covers this prompt PLUS every other request holding a grant.
    tokens += live_grant_tokens(sched, req, int(actor.page))
    rid = str(getattr(req, "rid", "?"))[:16]
    # Q-670 GRANT-BYPASS: an older request waiting for its card grant does not
    # hold this one back while it is young; past the age the head is the head.
    # pp0_grant runs only where _actor is armed (a dual-layout P rank); the gate is
    # restated here so the bypass can never reach a flip-form PP0.
    older = _older_waits(sched, rid) if _dual_layout_env() else []
    _ovt_head = None                      # #1920: the floor-blocked head this grant may pass
    if older and not _dpar.grant_may_bypass([t for _r, t in older], now=_now(), age_s=_dpar.head_age_s()):
        _ovt_head = floor_overtake_head(older, stages, int(getattr(actor, "_committed", 0) or 0), tag)
        if _ovt_head is None:
            _log_wait(rid, tokens, lambda: _grant_short_detail(stages, {}, tokens, tokens, len(older), "older-head"))
            return 0
    # Q-697b GRANT HOLD (dual P only): while a card shows D's unmet demand or a pressure
    # on P, no grant -- the bytes P released at idle-except-waiters are D's first. The
    # front's own gate for a PAUSED head is p_resume_ready; a waiter the front no longer
    # pauses needs the same one here (bounded by SGLANG_WEG2_DUAL_GRANT_HOLD_S).
    from sglang.srt.weg2 import dual_grant_wait as _dgw

    _hold = _dgw.grant_held_by_d(req, stages) if _dual_layout_env() else None
    if _hold is not None:
        _dgw.note_hold(rid, _hold, req)
        _log_wait(rid, tokens, lambda: _grant_short_detail(stages, {}, tokens, tokens, len(older), _hold))
        return 0
    own = int(getattr(actor, "_committed", 0) or 0)    # PP0's card: the ledger covers its mapping exactly
    taken: list = []
    lvl = group_grant(stages, tokens, lambda pth: CardKvLedger(pth, "P"), covered={0: own}, taken_out=taken)
    if not lvl and getattr(actor, "streamer", None) is not None:
        # #1962 P-LAYER-STREAM: short ONLY on PP0's card -> PP0 pauses weight units of its own (D untouched),
        # lends the bytes and asks once more. Off (no streamer): this branch does not exist.
        from sglang.srt.weg2 import p_layer_stream as _pls
        from sglang.srt.weg2.card_kv_ledger import peek as _peek

        if _pls.try_stream_for_grant(actor, stages, tokens, own, _peek) > 0:
            taken = []
            lvl = group_grant(stages, tokens, lambda pth: CardKvLedger(pth, "P"), covered={0: own},
                              taken_out=taken)
    if lvl and older:
        head, since = min(older, key=lambda x: x[1])
        logger.info("%s %s rid=%s tokens=%d past=%s head_wait_s=%.1f: the head waits for a card, this grant "
                    "fits now (Q-670, PP0 decides, the told carries it)", MARK, _dpar.GRANT_MARK, rid,
                    int(tokens), head, _now() - since)
    if lvl:
        k = lvl // int(stages[0]["step"])
        try:
            actor.map_granted(lvl, charged=max(0, int(stages[0]["bytes"][k]) - own))
        except Weg2DualKvMapShort as exc:
            # MAP-SHORT WAIT: the card ledger promised bytes the card does not
            # have. Every card's charge goes back, PP0's ledger is reconciled
            # against cuMemGetInfo (the next grant is priced on what is really
            # there) and the request is HELD -- a wait, never a rank death.
            for _i, led, got in taken:
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
        if _ovt_head is not None:
            # #1920: this grant went past a head that D's live floor keeps short (counted only once it holds)
            if len(_OVERTAKERS) > 4096:
                _OVERTAKERS.clear()
            _OVERTAKERS[_ovt_head] = int(_OVERTAKERS.get(_ovt_head, 0)) + 1
            logger.info("%s rid=%s head=%s overtakers=%d tokens=%d level=%d: the head's grant is short while D's "
                        "live floor holds the shrink, this grant fits now (max %d per head)", FLOOR_MARK, rid,
                        _ovt_head, _OVERTAKERS[_ovt_head], int(tokens), int(lvl), _dpar.head_bypass_floor_max())
        # Q-630: the followers' charges stand on their cards until a follower
        # adopts them from the told (map_granted) -- held on the request until
        # the told is on the wire (with_dual_kv), returned if it never leaves.
        req._dual_grant_untold = [(i, led, got) for i, led, got in taken if i != 0 and got] or None
        waited = _wait_granted(rid)
        logger.info("%s PP0 GRANT rid=%s tokens=%d on all %d cards%s", MARK, rid, lvl, pp,
                    (" after %d waits over %.1f s" % waited) if waited else "")
    else:
        _lv = min(tokens, int(stages[0]["top"])) if stages else tokens
        if _dpar.head_bypass_floor_armed():
            if len(_HEAD_LV) > 4096:
                _HEAD_LV.clear()
            _HEAD_LV[rid] = int(_lv)          # #1920: what this waiter asked at its last attempt
        _inf = None
        if _infeasible_skip_armed():
            # evaluated on every wait only while the switch is on (a ledger peek per card); otherwise the
            # marker is computed lazily on the log's backoff curve below
            _inf = infeasible_cards(stages, {0: own}, _lv, _stream_room(actor))
            _INFEASIBLE[rid] = bool(_inf)

        def _inf_line():
            cards = _inf if _inf is not None else infeasible_cards(stages, {0: own}, _lv, _stream_room(actor))
            return _infeasible_line(rid, tokens, _lv, cards) if cards else None

        _log_wait(rid, tokens, lambda: _grant_short_detail(
            stages, {0: own}, _lv, tokens, len(older), None), infeasible=_inf_line)
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
    _INFEASIBLE.clear()
    _HEAD_LV.clear()
    _OVERTAKERS.clear()
    _FLOOR_CACHE[0], _FLOOR_CACHE[1] = None, False
    _CENSUS.update(next=0.0, iv=1.0, waits=0)


def _log_wait(rid: str, tokens: int, detail=None, infeasible=None) -> None:
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
        if detail is not None:                    # #1530 GRANT-SHORT, on the same backoff curve
            try:
                logger.info("#1530 GRANT-SHORT rid=%s tokens=%d %s", rid, int(tokens), detail())
            except Exception:  # noqa: BLE001 -- a log line never reaches the round
                pass
        if infeasible is not None:                # #1640 GRANT-INFEASIBLE, same backoff curve (pure log)
            try:
                _line = infeasible()
                if _line:
                    logger.info("#1640 GRANT-INFEASIBLE %s", _line)
            except Exception:  # noqa: BLE001 -- a log line never reaches the round
                pass
    else:
        e[4] += 1
    if t >= _CENSUS["next"]:
        logger.info("%s PP0 WAIT census: rids=%d waits=%d since the last census line (next in %.0f s)",
                    MARK, len(_WAITS), _CENSUS["waits"], _CENSUS["iv"])
        _CENSUS["waits"] = 0
        _CENSUS["next"] = t + _CENSUS["iv"]
        _CENSUS["iv"] *= 2.0


def _dual_layout_env() -> bool:
    """Q-670: GRANT-BYPASS exists only in the dual layout (SGLANG_WEG2_DUAL_LAYOUT=1)."""
    return str(os.environ.get("SGLANG_WEG2_DUAL_LAYOUT", "")).strip() == "1"


def _older_waits(sched, rid: str) -> list:
    """(rid, wait start) of every request PP0 still holds for its card grant
    that started waiting BEFORE ``rid`` did -- only the held ones: a rid that
    left (abort, finish) keeps no say, and a younger waiter never blocks an
    older one (two aged waiters would otherwise hold each other for ever)."""
    held = getattr(sched, "_weg2_store_held", None) or {}
    own = _WAITS.get(rid)
    own_t = own[0] if own is not None else _now()
    skip_inf = _infeasible_skip_armed()    # #1640: an infeasible head does not hold the younger ones back
    out = []
    for r in held.values():
        if not getattr(r, "_dual_kv_wait", False):
            continue
        k = str(getattr(r, "rid", "?"))[:16]
        e = _WAITS.get(k)
        if k != rid and e is not None and e[0] < own_t:
            if skip_inf and _INFEASIBLE.get(k):
                continue
            out.append((k, e[0]))
    return out


def _wait_granted(rid: str):
    """Forget ``rid``'s wait; returns (waits, seconds) when it had waited."""
    e = _WAITS.pop(rid, None)
    _RETRY.pop(rid, None)                      # #1530: a granted rid starts a fresh throttle
    _INFEASIBLE.pop(rid, None)                 # #1640
    _HEAD_LV.pop(rid, None)                    # #1920
    _OVERTAKERS.pop(rid, None)
    if not _WAITS:
        _CENSUS.update(next=0.0, iv=1.0, waits=0)
    return (e[3], _now() - e[0]) if e else None


class Weg2DualPWakeShort(RuntimeError):
    """W-DUAL-P-WAKE-SHORT: P was told to wake but the card pool cannot give
    back the bytes P lent it at its sleep (D committed into them). The front's
    hysteresis wakes P only when they are free; this is the named stop for the
    case it cannot see."""


def _lend_armed(sched):
    """This rank's stage when it is a P rank of the dual layout (the only rank
    that lends its card pool -- awake at stage 1, asleep at stage 2), else None."""
    if str(os.environ.get("SGLANG_WEG2_DUAL_LAYOUT", "")).strip() != "1":
        return None
    if str(os.environ.get("SGLANG_WEG2_GROUP", "")).strip().upper() != "P":
        return None
    return _actor(sched)


_sleep_armed = _lend_armed


def lent_bytes(actor) -> int:
    """What this rank lent its card pool now: the sleep loan plus the awake loan."""
    return int(getattr(actor, "_sleep_lent", 0) or 0) + int(getattr(actor, "_awake_lent", 0) or 0)


LEND_GATE_MARK = "#1480 LEND-RESUME-GATE"
#: P ranks of the dual layout (PP0..PP2), the stage files the front reads
LEND_GATE_RANKS = 3


def lend_gate_reading(tags: Sequence[str], ranks: int = LEND_GATE_RANKS) -> Tuple[List[Optional[int]], str]:
    """#1480: what each P rank still lends its card pool (stage file ``lent``), read
    under the first of ``tags`` whose file exists. ``None`` = no readable file for
    that rank under any tag. Returns (per-rank list, tag that answered)."""
    import json

    per: List[Optional[int]] = []
    used = ""
    for r in range(int(ranks)):
        val: Optional[int] = None
        for tag in tags:
            try:
                with open(stage_file(tag, r)) as f:
                    val = max(0, int(json.load(f).get("lent") or 0))
            except (OSError, ValueError, AttributeError, TypeError):
                continue
            used = used or str(tag)
            break
        per.append(val)
    return per, used


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
    """Before P's wake maps its weights: take the loan back -- the sleep loan and
    an awake loan of stage 1 (Q-660) still standing -- or stop named."""
    actor = _sleep_armed(sched)
    lent = lent_bytes(actor) if actor is not None else 0
    if lent <= 0:
        return 0
    if not actor.ledger.reclaim(lent):
        raise Weg2DualPWakeShort(
            "W-DUAL-P-WAKE-SHORT: P's wake needs back the %d B it lent the card pool at its sleep, the pool "
            "has %d B free -- D committed into the loan" % (lent, int(actor.ledger.state().free)))
    actor._sleep_lent = 0
    actor._awake_lent = 0
    _republish_stage(sched, actor)
    logger.warning("%s WAKE-RECLAIM %d B back from the card pool before P maps its weights", MARK, lent)
    return lent


# -- Q-660 DUAL-AWAKE-LEND: stage 1 lends what an awake P does not need ----------
#
# User rule 03.10. (~15:35Z, verbatim): "wenn auf D kv knapp wird, gibt P seinen kv
# auf". Dual y8v (fs10031504, 15:20:53): D ran its pool to 0.96 and stalled while P
# kept its share -- the old ladder lent P's bytes only at stage 2 (the sleep). Stage
# 1 now: P stopped at its chunk boundary and released its KV (release_all, finished
# chunks in L2); then it returns its device allocator cache and LENDS the freed bytes
# to the card pool, awake -- D grows into them. The loan comes back (awake_reclaim)
# only once D's pressure is gone; the sleep (stage 2) stays the next step if not.

LEND_MARK = "Q-660 DUAL-P-LEND"
RECLAIM_MARK = "Q-660 DUAL-P-RECLAIM"


def _empty_device_cache() -> None:
    import torch

    torch.cuda.empty_cache()


def awake_lend(sched, why: str = "", *, phys=None, empty_cache=None) -> int:
    """Stage 1, awake: P has released its KV (no live page); give the device
    allocator's cached free blocks back and lend the freed bytes to the card
    pool. Returns the bytes lent (0 off a dual P rank, with a live page, or when
    nothing was freed). ``phys``/``empty_cache``: injectable for tests."""
    actor = _lend_armed(sched)
    if actor is None:
        return 0
    phys = phys or phys_free_bytes
    if int(getattr(actor, "mapped_tokens", 0) or 0) > 0:
        logger.warning("%s refused why=%s: P still maps %d KV tokens -- stage 1 lends only after P released "
                       "its KV", LEND_MARK, why or "-", int(actor.mapped_tokens))
        return 0
    before = phys()
    if before is None:
        return 0
    (empty_cache or _empty_device_cache)()
    after = phys()
    freed = max(0, int(after or 0) - int(before))
    if freed:
        actor.ledger.lend(freed)
        actor._awake_lent = int(getattr(actor, "_awake_lent", 0) or 0) + freed
        _republish_stage(sched, actor)
    logger.warning("%s bytes=%d lent_total=%d why=%s -- P awake (stage 1, KV released) lends its freed "
                   "device bytes to the card pool; D may grow into them until P reclaims them",
                   LEND_MARK, freed, int(getattr(actor, "_awake_lent", 0) or 0), why or "-")
    return freed


def awake_reclaim(sched, why: str = "") -> int:
    """P returns from stage 1: take the awake loan back. Only when the card pool
    has it free (D did not commit into it) -- else nothing changes and P stays
    stopped (the front asks again). Returns the bytes reclaimed."""
    actor = _lend_armed(sched)
    lent = int(getattr(actor, "_awake_lent", 0) or 0) if actor is not None else 0
    if lent <= 0:
        return 0
    if not actor.ledger.reclaim(lent):
        logger.warning("%s refused bytes=%d free=%d why=%s -- D still holds part of the awake loan; P stays "
                       "stopped", RECLAIM_MARK, lent, int(actor.ledger.state().free), why or "-")
        return 0
    actor._awake_lent = 0
    _republish_stage(sched, actor)
    logger.warning("%s bytes=%d why=%s -- the awake loan is back; P may prefill again", RECLAIM_MARK, lent,
                   why or "-")
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
        # Q-630: the grant is on the wire -- from here every follower adopts its
        # charge (on_told -> map_granted), nobody may return it a second time.
        req._dual_grant_untold = None
    return told


def return_untold_grant(sched, req, why: str) -> int:
    """Q-630 DUAL-GRANT-RETURN (PP0): give back the follower cards' share of
    ``req``'s group grant whose told never went on the wire. ``pp0_grant``
    charged every follower card at the intake; a follower takes the charge over
    only when the told reaches it (``on_told``). A request that leaves the
    queue before (abort, intake stall) left that charge on the follower cards
    with no owner: dual y8u 12:09:23 (boot ...fs10031206), weg2-0-1 + weg2-0-10
    aborted by the front's P-PAUSE before their told, Q-580 dropped the held
    told, PP1 stayed at 973078528 B / PP2 at 1459617792 B committed = (90112 +
    147456) tokens x 4096 / 6144 B -- the front's RESUME-WAIT never saw all
    zeros (210.8 s, up to 1441 s), long requests ran into the client timeout.
    Returns the bytes given back."""
    # getattr: the flip form reaches this through the Q-580 / publish-loop drops with
    # any Req-like; a request that never took a dual grant carries none
    untold = getattr(req, "_dual_grant_untold", None)
    if not untold:
        return 0
    req._dual_grant_untold = None
    n = 0
    for pp, led, got in untold:
        led.release(got)
        n += int(got)
        logger.warning("%s P-KV GRANT-RETURN rid=%s pp=%d bytes=%d why=%s: the told carrying PP0's group "
                       "grant never left -- this follower card's charge goes back to the card pool",
                       MARK, str(getattr(req, "rid", "?"))[:16], int(pp), int(got), why)
    return n


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
    tree = sched.tree_cache
    tree.flush_write_through_acks()
    drain_idle_load_acks(sched, tree)
    return True


def drain_idle_load_acks(sched, tree) -> int:
    """Q-680 IDLE-LOAD-ACK (dual P): an idle P stage drains its load-back acks
    too. The load-back lock (``ongoing_load_back``) is released only by
    ``loading_check``, which the batch path polls per microbatch; a follower
    whose request finished in the very pass that loaded its prefix goes idle and
    never polls again. Dual y8w fs10031623 16:45:43: PP1/PP2 loaded weg2-0-220's
    39099-row prefix, the request finished, and the lock stayed (lock census
    ongoing_lb=1 tracked_protected=39168 until 17:02) -- on_idle never released
    (protected > 0), PP1/PP2 kept 201/302 MB committed for 905 s, the front's
    RESUME-WAIT held every request (900-s timeouts). Rank-local (#737)."""
    ongoing = getattr(tree, "ongoing_load_back", None)
    check = getattr(tree, "loading_check", None)
    if not ongoing or check is None:
        return 0
    n0 = len(ongoing)
    check()
    done = n0 - len(getattr(tree, "ongoing_load_back", None) or ())
    if done > 0:
        logger.info("%s P-KV IDLE-LOAD-ACK pp_rank=%s drained=%d left=%d: an idle P stage released the "
                    "load-back lock(s) the batch path no longer polls (Q-680)", MARK,
                    getattr(getattr(sched, "ps", None), "pp_rank", "?"), done,
                    len(getattr(tree, "ongoing_load_back", None) or ()))
    return done


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
    # Q-697b (R2/R5): the guard sits HERE, after the stamp was read and right before the abort
    # is applied: a held abort whose rid PP0 holds again as a grant waiter (marker, which PP0
    # publishes before it stamps) stays held -- its verdict reads the rid and would take the new
    # instance with it. Only those rids; a chunked abort and the other holds proceed.
    from sglang.srt.weg2 import dual_grant_wait as _dgw

    conflict = _dgw.conflicting_rids(sched, held) if held else set()
    if conflict and req is None and set(held) <= conflict:
        return False
    pend = getattr(sched, "_weg2_pending_waiting_aborts", None)
    stash = {r: pend.pop(r) for r in list(pend) if str(r) in conflict} if (conflict and pend) else {}
    sched._791c_pp0_drained = True
    try:
        sched.process_pending_chunked_abort()
    finally:
        sched._791c_pp0_drained = False
        if stash:
            pend.update(stash)
    logger.info("%s FOLLOWER-ABORT-APPLIED rid=%s pp_rank=%s: PP0 idle since %.1f s after this rank "
                "saw the abort, every pass it launched ran here (fwd %d >= %d; the #791C liveness "
                "release the dual layout has no lap for)", MARK,
                str(getattr(req, "rid", None) or ",".join(held) or "?")[:48], getattr(sched.ps, "pp_rank", "?"),
                idle_t - seen[1],
                int(getattr(sched, "forward_ct", 0) or 0), pp0_fwd)
    return True


IDLE_HELD_LOG_S = 30.0


def _note_idle_held(sched, actor, tree) -> None:
    """Q-680: an idle P rank whose release is held names why, every 30 s -- a
    held mapping keeps 'P committed' on the card and the front's RESUME-WAIT
    reads it (dual y8w: 905 s without one line saying so)."""
    import time as _t

    now = _t.time()
    if now - float(getattr(sched, "_dual_idle_held_log_t", 0.0) or 0.0) < IDLE_HELD_LOG_S:
        return
    sched._dual_idle_held_log_t = now
    try:
        prot = int(getattr(tree, "protected_size", lambda: 0)() or 0)
        evict = int(tree.evictable_size() or 0)
    except Exception:  # noqa: BLE001 -- an instrument never raises
        prot = evict = -1
    logger.warning("%s P-KV IDLE-HELD pp_rank=%s mapped=%d committed=%d B protected=%d evictable=%d "
                   "ongoing_load_back=%d ongoing_write_through=%d: the idle release waits for these",
                   MARK, getattr(getattr(sched, "ps", None), "pp_rank", "?"), int(actor.mapped_tokens),
                   int(getattr(actor, "_committed", 0) or 0), prot, evict,
                   len(getattr(tree, "ongoing_load_back", None) or ()),
                   len(getattr(tree, "ongoing_write_through", None) or ()))


def on_idle(sched) -> int:
    """A fully idle P rank gives its whole context back: device tree evicted
    (write-back keeps the pages in L2), then unmapped and released."""
    if str(os.environ.get("SGLANG_WEG2_DUAL_LAYOUT", "")).strip() == "1":
        mark_pp0_idle(sched)
    actor = _actor(sched)
    if actor is not None:
        phys_check(actor, "P")
    if actor is not None and actor.mapped_tokens <= 0 and getattr(actor, "streamer", None) is not None:
        _stream_regain(sched, actor)          # #1962: P holds no KV -- weight units back if the card has them
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
                _note_idle_held(sched, actor, tree)
                return 0  # still held (write-back in flight / locked): next idle pass
        except Exception as exc:  # noqa: BLE001 -- a failed flush keeps the pages, never frees under them
            logger.warning("%s idle flush skipped: %r", MARK, exc)
            return 0
    n = actor.release_all()
    if n and getattr(actor, "streamer", None) is not None:
        _stream_regain(sched, actor)
    return n


def _stream_regain(sched, actor) -> int:
    """#1962 regain at idle; a refused resume keeps the unit paused and the loan standing (logged).
    Not while a held request still waits for its card grant: its next attempt would pause the units again
    (a stream-out/regain cycle per retry); the regain waits for the first idle pass without a waiter."""
    from sglang.srt.weg2 import p_layer_stream as _pls

    held = getattr(sched, "_weg2_store_held", None) or {}
    if any(getattr(r, "_dual_kv_wait", False) for r in held.values()):
        return 0
    try:
        return _pls.regain_at_idle(actor, phys_free=phys_free_bytes)
    except Exception as exc:  # noqa: BLE001 -- the unit stays streamed, P serves on
        logger.warning("%s REGAIN-REFUSED %r -- the unit stays paused, P keeps streaming", _pls.MARK, exc)
        return 0
