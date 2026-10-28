"""D-POOL-LEND (flip form, 27.10.): while ONE prefill runs, D's KV pool tail is card free memory.

User order 27.10.: "es gibt doch schon eine Funktion, die KV-VRAM freigibt vom Dual-Layout her, warum nutzt du die
nicht um die Chunkgroesse zuzulassen?" -- and the thought behind it: with a single prefill the pool is mostly empty
(27B INT8 flip boot dkr27bbar1drei10272126: 270000 global tokens = 2.58 GB on TP1/TP2, 3.1 GB on TP0; a 128k prompt
holds ~1.2 GB of it), what is released stays as free card memory, and the width vote (``extend_trim.width_vote``,
EXTEND-CAP-SOFT: cap = (free - hard line) / rate) then funds a wider chunk.

THE MECHANISM IS THE DUAL D-KV STAGE'S, NOT A NEW ONE. The K/V buffers are saver allocations (``tms_set_spans``,
``tms_csrc/PATCH.md`` patch 3). At the KV attach each buffer is re-cut into LATTICE CELLS (``d_seat_vram.slot_spans``
with ``cuts``, one extent per cell) with the FULL mapping -- boot = top, nothing is trimmed at the boot. A release
unmaps the cells above T (an extent wholly inside the new plan keeps its pages and bytes), a return maps them
again; ``dual_p_kv_stage.PKvStage._move`` is the one move (rows, allocator cap ``KvRowCap``, cap check). Nothing of
the card ledger: in the flip form D and P are never awake together on a card, there is no peer to book against.

WHEN. Every input is a replicated value or a group value (one MIN collective over the D ranks carries the live
floor, entered only on a cadence decided from replicated inputs), so every rank takes the same verdict. Only a MAP
the card refuses is voted (MIN) like the dual's GROW-SHORT.

* release: exactly ONE pending prefill (a chunked request or a waiting one D's admission can take, the lend's own
  ``d_transient_lend.takeable_waiting``), its whole token count >= MIN_ROWS, and T = live rows + the prefill's rows
  + air at least two cells under the top. T is the highest live id (``max_live_id``: decode seats AND the tree)
  plus what the prefill allocates. When live rows hold the floor although the USED ids would fit under it they are
  first MOVED down by ``dual_d_compact`` (the dual's compaction, same gate and switch), else nothing is released
  (named, backed off). D-LOW-FIRST keeps new KV low so that this is rare.
* grow / return: a second pending prefill (at once), the live need outgrowing T (read every RECHECK_ROUNDS),
  SETTLE decode rounds without a pending prefill (hysteresis, not per chunk), and before every D sleep.
  The return is ``empty_cache`` (when the card is short), the map, then the cap lifted. A card that cannot map it
  (rc=2) is voted: every rank rolls back, the tail stays unmapped, the retry backs off (GROW-SHORT, degraded, not
  dead). The cap keeps the allocator under T meanwhile.
* sleep: ``quiesce_for_sleep`` puts the plan back to the full lattice (``now=False``: nothing is mapped, the pause
  drops what is mapped, the wake maps every cell) and lifts the cap -- D never sleeps lent.

The release runs at the top of the scheduler's round (``get_next_batch_to_run``), BEFORE the width vote reads
``cuda.mem_get_info()``, after a full device sync (``_sync_before_unmap``: no kernel may still write an unmapped
page). The user's reserve is never touched: the lend only gives bytes back to the card and takes exactly them
again; the card's floor stays the near-OOM edge the vote prices (rule 24.10.: the 1 GB line is a plan target).

LINES (one per move and rank; ``card_free_mib`` read with cuMemGetInfo before and after)::

    PDFLIP D-POOL-LEND rank=1 release T=196608 (from 270000) bytes=... ms=... card_free_mib before=1062 after=1790 ...
    PDFLIP D-POOL-LEND rank=1 return T=270000 (from 196608) bytes=... ms=... card_free_mib before=1790 after=1062 ...

``return ... pre_lend_free_mib=`` is the card's free memory before the first release of the episode: a full
return must read ``after`` = ``pre_lend_free_mib`` (minus what the allocator cache took meanwhile).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, List, Optional, Tuple

import msgspec

from flliper.srt.pdflip import dual_p_kv_stage as _pk

logger = logging.getLogger(__name__)

MARK = "PDFLIP D-POOL-LEND"
MIB = 1 << 20
#: decode/verify air that never goes below this many tokens above the live rows + the prefill
AIR_FLOOR_TOKENS = 2048
#: after a refused return (GROW-SHORT) the next read waits this many rounds
GROW_SHORT_BACKOFF_ROUNDS = 64
#: a refused compaction backs off 1, 2, 4 ... x RECHECK_ROUNDS, up to this factor
COMPACT_BACKOFF_MAX = 64
#: a release must give at least this many lattice cells back
MIN_RELEASE_CELLS = 2
#: the auto lattice step: an eighth of the pool on this row alignment, within these bounds (the INT8 drei pool of
#: 270000 rows lands on 32768, the int4ar pool of 126178 rows on 12288: ~11 cells, ~330 extents per rank, the same
#: map count per D wake as the INT8 lattice)
STEP_ALIGN = 4096
STEP_MIN = 8192
STEP_MAX = 32768


class Move(msgspec.Struct, frozen=True):
    """One verdict of the round: ``release`` / ``grow`` (to ``target`` tokens) or ``hold`` (nothing moves)."""

    kind: str
    target: int = 0
    why: str = ""


HOLD = Move("hold")


class Intent(msgspec.Struct, frozen=True):
    """What the round's cheap context says before the live floor is read: ``back`` (return all), ``check``
    (read the floor and size the pool) or ``none``."""

    kind: str
    why: str = ""


class Knobs(msgspec.Struct, frozen=True):
    """The environment, read once at the attach (init-static)."""

    step: int
    settle: int
    min_rows: int
    recheck: int
    low_first: int


def auto_step(top: int) -> int:
    """The lattice step when ``FLLIPER_PDFLIP_D_POOL_LEND_STEP_TOKENS`` is 0 (auto): about an eighth of the pool, on
    ``STEP_ALIGN`` rows, within [``STEP_MIN``, ``STEP_MAX``]. A fixed 32768 leaves a small pool without a release:
    a release needs ``MIN_RELEASE_CELLS`` cells under the top, so the int4ar pool (126178 rows per rank, TP2 3080)
    could lend only while live + prefill + air stayed under 60642 rows, the INT8 drei pool (270000) under 204464.
    Pure."""
    eighth = (int(top) // 8 // STEP_ALIGN) * STEP_ALIGN
    return max(STEP_MIN, min(STEP_MAX, eighth))


def resolve_step(configured: int, top: int) -> int:
    """The knob's value (> 0 = as set) or the auto step of this pool (0). Pure."""
    return int(configured) if int(configured) > 0 else auto_step(top)


def knobs_from_env() -> Knobs:
    """``step`` 0 = auto: :func:`resolve_step` fixes it at the attach, where the pool size is known."""
    from flliper.srt.environ import envs

    return Knobs(step=max(0, int(envs.FLLIPER_PDFLIP_D_POOL_LEND_STEP_TOKENS.get())),
                 settle=max(1, int(envs.FLLIPER_PDFLIP_D_POOL_LEND_SETTLE_ROUNDS.get())),
                 min_rows=max(0, int(envs.FLLIPER_PDFLIP_D_POOL_LEND_MIN_ROWS.get())),
                 recheck=max(1, int(envs.FLLIPER_PDFLIP_D_POOL_LEND_RECHECK_ROUNDS.get())),
                 low_first=max(0, int(envs.FLLIPER_PDFLIP_D_POOL_LEND_LOW_FIRST.get())))


def armed(env=None) -> bool:
    """Flip form: group D, NOT the dual layout (its D-KV stage owns the pool there), NOT the seat/stage form
    (D-MEM-SCHED stages its KV), and the switch (default ON)."""
    from flliper.srt.environ import envs
    from flliper.srt.pdflip import d_seat_vram as _sv

    env = os.environ if env is None else env
    if str(env.get("FLLIPER_PDFLIP_GROUP", "")).strip().upper() != "D":
        return False
    if str(env.get("FLLIPER_PDFLIP_DUAL_LAYOUT", "")).strip() == "1":
        return False
    if _sv.armed(env):
        return False
    return bool(envs.FLLIPER_PDFLIP_D_POOL_LEND.get())


# -- pure verdicts --------------------------------------------------------------------------------------------

def intent(*, lent: bool, pend: int, incoming: int, busy: bool, streak: int, settle: int, min_rows: int) -> Intent:
    """REPLICATED, pure: what the round's context asks of the lend, before the live floor is read.

    * a second pending prefill: the tail goes back at once (the pool must hold both);
    * no pending prefill and ``settle`` decode rounds (a running batch) in a row: back (hysteresis);
    * exactly one pending prefill of at least ``min_rows`` (or any while lent): size the pool (release, or grow
      when it outgrew T);
    * lent, no pending prefill, decodes running: size the pool too (decode growth must not run into the cap)."""
    if pend >= 2:
        return Intent("back", "pending=%d" % pend) if lent else Intent("none")
    if pend == 0:
        if lent and busy and streak >= settle:
            return Intent("back", "settled streak=%d" % streak)
        return Intent("check") if (lent and busy) else Intent("none")
    if lent or incoming >= min_rows:
        return Intent("check")
    return Intent("none")


def next_streak(*, streak: int, pend: int, busy: bool) -> int:
    """Decode rounds in a row with a running batch and no pending prefill (a pending prefill resets it, an idle
    loop pass keeps it)."""
    if pend:
        return 0
    return streak + 1 if busy else streak


def need_tokens(*, live: int, incoming: int, air: int, step: int) -> int:
    """T: the highest live row, what the pending prefill allocates and the air, on the lattice."""
    return _pk.round_up(max(0, int(live)) + max(0, int(incoming)) + max(0, int(air)), int(step))


def size_pool(*, mapped: int, top: int, step: int, need: int) -> Move:
    """REPLICATED, pure: the move that fits the pool to ``need`` (already on the lattice). Full pool: release when
    at least MIN_RELEASE_CELLS cells go; lent pool: grow when the need outgrew T (never a second shrink)."""
    target = int(top) if int(need) >= int(top) else int(need)
    if int(mapped) >= int(top):
        if target <= int(top) - MIN_RELEASE_CELLS * int(step):
            return Move("release", target, "need=%d" % int(need))
        return Move("hold", 0, "floor need=%d top=%d" % (int(need), int(top)))
    if target > int(mapped):
        return Move("grow", target, "need=%d outgrew T=%d" % (int(need), int(mapped)))
    return HOLD


def compact_target(*, used_ids: int, step: int) -> int:
    """The level a compaction moves every live row under: the used ids plus one cell of room for an owner class
    that needs more free low ids than its proportion (``dual_d_compact`` refuses ``class_short`` otherwise)."""
    return _pk.round_up(int(used_ids), int(step)) + int(step)


# -- the actor ----------------------------------------------------------------------------------------------

class PoolLend(_pk.PKvStage):
    """This D rank's KV pool lattice. ``mapped_tokens`` = the level mapped (``top`` = full); every move is the
    dual stage's ``_move`` (spans, backed rows, allocator cap)."""

    def __init__(self, born_tensors, *, allocator, pools, page_size: int, granule: int, top_tokens: int,
                 knobs: Knobs, rank: int, spans=None, engage_cap=None, sync=None,
                 free_fn: Optional[Callable[[], int]] = None, trim_fn: Optional[Callable[[], None]] = None):
        super().__init__(born_tensors, None, allocator=allocator, pools=pools, page_size=page_size,
                         granule=granule, top_tokens=top_tokens, spans=spans, step=knobs.step,
                         engage_cap=engage_cap, sync=sync)
        self.knobs = knobs
        self.rank = int(rank)
        self.mapped_tokens = int(top_tokens)
        self.gmin: Callable = lambda vals: list(vals)
        self.free_fn = free_fn or _card_free
        self.trim_fn = trim_fn or _empty_cache
        self.rounds = 0
        self.streak = 0
        self.next_round = 0
        self.refusals = 0
        self.had_pending = False
        self.floor_key: Optional[Tuple[int, int]] = None
        self.pre_free: Optional[int] = None

    @property
    def lent(self) -> bool:
        return int(self.mapped_tokens) < int(self.top)

    def release_to(self, tokens: int, *, note: str) -> None:
        """Unmap the cells above ``tokens``: the cap first (new ids land below), a device sync, the span move.
        ``tokens`` is on the group's live floor, so no live id lies above it (a breach stops by name)."""
        before, t0 = self.free_fn(), time.perf_counter()
        nbytes = self.bytes_for(self.mapped_tokens) - self.bytes_for(tokens)
        self._engage_cap(self.allocator, int(tokens), self.page)
        live = int(_pk.max_live_id(self.allocator, self.page)) * self.page
        if live > int(tokens):
            raise _pk.PdFlipDualKvCapBreach(
                "%s release under a live row: id %d is live above T=%d although the group floor was below it"
                % (MARK, live, int(tokens)))
        t_sync = time.perf_counter()
        self._sync()
        t_move = time.perf_counter()
        self._move(int(tokens))
        t_done = time.perf_counter()
        was, self.mapped_tokens = int(self.mapped_tokens), int(tokens)
        if self.pre_free is None:
            self.pre_free = int(before)
        after = self.free_fn()
        logger.info("%s rank=%d release T=%d (from %d) bytes=%d ms=%.1f (sync_ms=%.1f move_ms=%.1f) "
                    "card_free_mib before=%d after=%d %s",
                    MARK, self.rank, int(tokens), was, nbytes, 1000.0 * (time.perf_counter() - t0),
                    1000.0 * (t_move - t_sync), 1000.0 * (t_done - t_move),
                    int(before) // MIB, int(after) // MIB, note)

    def grow_to(self, tokens: int, *, why: str) -> bool:
        """Map the cells up to ``tokens`` (``top`` = the whole pool), then lift the cap. Every rank maps or none
        does (MIN of the per-rank verdict); a short card rolls every rank back (GROW-SHORT)."""
        tokens = min(int(self.top), int(tokens))
        before, t0 = self.free_fn(), time.perf_counter()
        nbytes = self.bytes_for(tokens) - self.bytes_for(self.mapped_tokens)
        t_pre = time.perf_counter()
        if int(before) < nbytes:
            self._sync()
            self.trim_fn()                  # the extends' cache sits on the freed bytes: give it back first
        t_move = time.perf_counter()
        short = self._try_move(tokens)
        t_gmin = time.perf_counter()
        verdict = int(self.gmin([0 if short is not None else 1])[0])
        t_done = time.perf_counter()
        if verdict != 1:
            self._move(int(self.mapped_tokens))          # unmap what this rank got; rows and cap back
            self.next_round = self.rounds + GROW_SHORT_BACKOFF_ROUNDS
            logger.warning("%s rank=%d GROW-SHORT T=%d (from %d) need=%d card_free_mib=%d: %s -- rolled back on "
                           "every rank, the tail stays unmapped, the next read in %d rounds (%s)", MARK,
                           self.rank, tokens, int(self.mapped_tokens), nbytes, int(before) // MIB,
                           short if short is not None else "another rank's card is short",
                           GROW_SHORT_BACKOFF_ROUNDS, why)
            return False
        was, self.mapped_tokens = int(self.mapped_tokens), tokens
        if tokens >= int(self.top):
            self._unbound_rows()
        phases = "pre_ms=%.1f move_ms=%.1f gmin_ms=%.1f" % (
            1000.0 * (t_move - t_pre), 1000.0 * (t_gmin - t_move), 1000.0 * (t_done - t_gmin))
        self._log_return(tokens, was, nbytes, before, t0, why, phases=phases)
        return True

    def _try_move(self, tokens: int) -> Optional[Exception]:
        """The map, or the card-short error it raised (None = mapped)."""
        try:
            self._move(int(tokens))
        except _pk.PdFlipDualKvMapShort as exc:
            return exc
        return None

    def _unbound_rows(self) -> None:
        """Full again: no stage bound (``set_stage_backed_rows(None)``: the whole tensor is backed)."""
        for pool in self.pools:
            pool.set_stage_backed_rows(None)

    def _log_return(self, tokens: int, was: int, nbytes: int, before: int, t0: float, why: str,
                    phases: str = "") -> None:
        after = self.free_fn()
        full = tokens >= int(self.top)
        pre = self.pre_free if full else None
        logger.info("%s rank=%d return T=%d (from %d) bytes=%d ms=%.1f (%s) card_free_mib before=%d after=%d%s (%s)",
                    MARK, self.rank, tokens, was, nbytes, 1000.0 * (time.perf_counter() - t0), phases,
                    int(before) // MIB, int(after) // MIB,
                    "" if pre is None else " pre_lend_free_mib=%d delta_mib=%d" % (int(pre) // MIB,
                                                                                    (int(after) - int(pre)) // MIB),
                    why)
        if full:
            self.pre_free = None

    def quiesce(self) -> bool:
        """Before a D sleep: the plan back to the full lattice WITHOUT mapping (the pause drops what is mapped,
        the wake maps every cell), the cap lifted, the rows unbounded. True when something was lent."""
        if not self.lent:
            return False
        for ptr, g in self.born:
            rc = self.spans.set_spans(ptr, self._plan(g, int(self.top)), now=False)
            if rc != 0:
                raise RuntimeError("%s: tms_set_spans(full plan, now=False) rc=%d for %s" % (MARK, rc, g.geom.name))
        self._engage_cap(self.allocator, int(self.top), self.page)
        self._unbound_rows()
        was, self.mapped_tokens, self.pre_free = int(self.mapped_tokens), int(self.top), None
        logger.info("%s rank=%d return T=%d (from %d) at the D sleep: the plan is the full lattice again, the "
                    "pause drops the mapped cells, the wake maps all of them", MARK, self.rank, int(self.top), was)
        return True


def _card_free() -> int:
    import torch

    return int(torch.cuda.mem_get_info()[0])


def _empty_cache() -> None:
    import torch

    torch.cuda.empty_cache()


# -- birth: the lattice cells --------------------------------------------------------------------------------

def _tensor_geom(t, pool, top: int, alloc_bytes: int, name: str):
    """The tensor's token geometry, or None when the layout does not map ``top`` global tokens onto its rows
    (then nothing of this pool is cut). The check is on the UNCLIPPED slot count: ``slots_for`` clips to the
    rows, which would hide a tensor that is too small or too large for the lattice."""
    from flliper.srt.pdflip import dual_d_kv_stage as _ddk
    from flliper.srt.pdflip.d_seat_vram import KvTensorGeom, SlotTensorGeom

    rows, page = int(t.shape[0]), int(pool.page_size)
    ratio = max(1, (int(pool.size) + page) // max(1, rows))
    slot_bytes = (t.numel() * t.element_size()) // max(1, rows)
    owner = _ddk._owner_block_for_rows(rows, int(top), page)

    def geom(slots: int):
        return KvTensorGeom(SlotTensorGeom(name, 1, int(slots), int(slot_bytes), int(alloc_bytes)),
                            token_ratio=ratio, token_pad=page, owner_block=owner)

    unclipped = geom(1 << 60).slots_for(int(top))
    return geom(rows) if 0 <= unclipped - rows <= page else None


def _cut_tensor(t, geom, *, spans, granule: int, step: int, top: int) -> int:
    """Re-cut one tensor into lattice cells with the FULL mapping; returns the extent count. The old extent is
    released and the cells map fresh (a new tensor holds nothing, it is zeroed after)."""
    from flliper.srt.pdflip.d_seat_vram import PdFlipDSeatVramRefused, _sync_before_unmap, slot_spans

    cuts = [geom.slots_for(k) for k in _pk.lattice(int(top), int(step))]
    plan = slot_spans(geom.geom, geom.slots_for(int(top)), int(granule), cuts=cuts)
    _sync_before_unmap(t)
    rc = spans.set_spans(t.data_ptr(), plan, now=True)
    if rc != 0:
        raise PdFlipDSeatVramRefused("%s: cutting %s into lattice cells failed (tms_set_spans rc=%d)" % (
            MARK, geom.geom.name, rc))
    t.zero_()
    return len(plan)


def _kv_tensors(pool) -> List[Tuple[str, object]]:
    return ([("k%d" % i, b) for i, b in enumerate(pool.k_buffer)]
            + [("v%d" % i, b) for i, b in enumerate(pool.v_buffer)])


def _cut_pools(pools, *, top: int, spans, granule: int, step: int) -> Tuple[List[Tuple[int, object]], int, str]:
    """Cut every K/V tensor of ``pools``: ``(born, extents, "")``, or ``([], 0, reason)`` -- the reason names
    what kept the pool whole (the attach then arms nothing)."""
    from flliper.srt.pdflip.d_seat_vram import PdFlipDSeatVramRefused

    born: List[Tuple[int, object]] = []
    extents = 0
    for pool in pools:
        for name, t in _kv_tensors(pool):
            if not t.numel():
                continue
            info = spans.info(t.data_ptr())
            if info is None:
                return [], 0, "%s is not a saver allocation" % name
            geom = _tensor_geom(t, pool, top, int(info.size), name)
            if geom is None:
                return [], 0, "%s: its rows do not map %d global tokens" % (name, top)
            try:
                extents += _cut_tensor(t, geom, spans=spans, granule=granule, step=step, top=top)
            except PdFlipDSeatVramRefused as exc:
                return [], 0, str(exc)
            born.append((int(t.data_ptr()), geom))
    return born, extents, ""


_ACTOR: Optional[PoolLend] = None


def attach(runner) -> Optional[PoolLend]:
    """KV attach (after the pools and the allocator exist, before the graph capture): cut the target pool into
    lattice cells with the full mapping and arm the actor. Not armed: None, nothing touched."""
    global _ACTOR
    if not armed() or runner.is_draft_worker:
        return None
    import torch

    from flliper.srt.pdflip import d_seat_vram as _sv

    spans = _sv.tms()
    if not spans.available:
        logger.warning("%s NOT armed: the saver's span map (tms_set_spans) is not in the preloaded hook", MARK)
        return None
    dev = torch.device("cuda", int(runner.gpu_id))
    granule, allocator = _sv.granule_for(dev), runner.token_to_kv_pool_allocator
    pools, top = _pk.stage_pools(runner.token_to_kv_pool), int(allocator.size)
    knobs = knobs_from_env()
    knobs = msgspec.structs.replace(knobs, step=resolve_step(knobs.step, top))
    born, extents, reason = _cut_pools(pools, top=top, spans=spans, granule=granule, step=knobs.step)
    if reason:
        logger.warning("%s NOT armed: %s", MARK, reason)
        return None
    torch.cuda.synchronize(dev)
    actor = PoolLend(born, allocator=allocator, pools=pools, page_size=int(runner.page_size), granule=granule,
                     top_tokens=top, knobs=knobs, rank=int(runner.tp_rank), spans=spans,
                     sync=lambda: torch.cuda.synchronize(dev))
    _ACTOR = actor
    _arm_low_first(allocator, knobs.low_first)
    logger.info("%s armed rank=%d top=%d tokens step=%d cells<=%d tensors=%d extents=%d bytes=%d settle=%d "
                "min_rows=%d recheck=%d (flip form: the pool tail above T is card free memory while ONE prefill "
                "runs; every D wake maps %d extents)", MARK, actor.rank, top, knobs.step, top // knobs.step + 1,
                len(born), extents, actor.bytes_for(top), knobs.settle, knobs.min_rows, knobs.recheck, extents)
    return actor


def _arm_low_first(allocator, every: int) -> None:
    from flliper.srt.pdflip import dual_d_kv_stage as _ddk

    _ddk._arm_low_first(allocator, every)


# -- the round -----------------------------------------------------------------------------------------------

def _pending(sched) -> int:
    from flliper.srt.pdflip import d_transient_lend as _dtl

    return int(sched.chunked_req is not None) + len(_dtl.takeable_waiting(sched))


def _busy(sched) -> bool:
    return bool(sched.running_batch is not None and len(sched.running_batch.reqs))


def _air(sched, recheck: int) -> int:
    """Air above live + prefill: the decode/verify rounds of every seat until the next read (the chunk itself is
    in the prefill's rows)."""
    from flliper.srt.pdflip import d_seat_vram as _sv

    chunk = int(sched.chunked_prefill_size or 0)
    return AIR_FLOOR_TOKENS + max(0, int(_sv._air(sched)) - chunk) * int(recheck)


def _free_ids(alloc) -> int:
    n = 0
    for name in ("free_pages", "release_pages"):
        ids = getattr(alloc, name, None)
        if ids is not None and ids.numel():
            n += int(ids.numel())
    return n


def _group_floor(actor: PoolLend) -> Tuple[int, int]:
    """(live floor, used ids) of the D GROUP in tokens: the MAX over the ranks (one collective, entered on a
    cadence decided from replicated inputs)."""
    live = int(_pk.max_live_id(actor.allocator, actor.page)) * int(actor.page)
    used = max(0, int(actor.allocator.size) - _free_ids(actor.allocator))
    g = actor.gmin([-live, -used])
    return -int(g[0]), -int(g[1])


def _compact(sched, actor: PoolLend, *, live: int, used: int, incoming: int, air: int) -> Optional[int]:
    """The live floor blocks the release although the used ids would fit under it: move the live rows above the
    compaction level down (``dual_d_compact.run``, every D rank enters on group values). The group's new floor,
    or None (refused: backed off, nothing moved)."""
    from flliper.srt.pdflip import dual_d_compact as _dc

    if not _dc.armed():
        return None
    level = compact_target(used_ids=used, step=actor.step)
    ideal = need_tokens(live=level, incoming=incoming, air=air, step=actor.step)
    have = need_tokens(live=live, incoming=incoming, air=air, step=actor.step)
    if ideal > int(actor.top) - MIN_RELEASE_CELLS * int(actor.step) or have <= ideal:
        return None
    new = _dc.run(sched, actor, target=int(level), floor=int(live), gmin=actor.gmin)
    if new is None:
        actor.refusals = min(COMPACT_BACKOFF_MAX, max(1, actor.refusals * 2))
        actor.next_round = actor.rounds + actor.knobs.recheck * actor.refusals
        return None
    actor.refusals = 0
    return int(new)


def _note_floor(actor: PoolLend, *, live: int, incoming: int, why: str) -> None:
    """Why a sole prefill did not release (one line per change of the live floor / the prefill, in cells)."""
    key = (int(live) // int(actor.step), int(incoming) // int(actor.step))
    if key == actor.floor_key:
        return
    actor.floor_key = key
    logger.info("%s rank=%d skip live=%d incoming=%d mapped=%d (%s): the live rows and the prefill leave no cell "
                "to release", MARK, actor.rank, int(live), int(incoming), int(actor.mapped_tokens), why)


def _size_step(sched, actor: PoolLend, *, incoming: int) -> Optional[str]:
    """The floor read and the move it asks for (one per cadence)."""
    actor.next_round = actor.rounds + actor.knobs.recheck
    air = _air(sched, actor.knobs.recheck)
    live, used = _group_floor(actor)
    move = size_pool(mapped=actor.mapped_tokens, top=actor.top, step=actor.step,
                     need=need_tokens(live=live, incoming=incoming, air=air, step=actor.step))
    if move.kind == "hold" and not actor.lent:
        new = _compact(sched, actor, live=live, used=used, incoming=incoming, air=air)
        if new is not None:
            live = new
            move = size_pool(mapped=actor.mapped_tokens, top=actor.top, step=actor.step,
                             need=need_tokens(live=live, incoming=incoming, air=air, step=actor.step))
    if move.kind == "release":
        actor.release_to(move.target, note="live=%d incoming=%d air=%d %s" % (live, incoming, air, move.why))
        return "release"
    if move.kind == "grow":
        return "grow" if actor.grow_to(move.target, why=move.why) else None
    if move.kind == "hold":
        _note_floor(actor, live=live, incoming=incoming, why=move.why)
    return None


def round_start(sched) -> Optional[str]:
    """Top of ``get_next_batch_to_run``, before the width vote reads the card, on every D rank of the flip form.
    Returns the move made (``release`` / ``grow`` / ``back``), else None."""
    actor = _ACTOR
    if actor is None:
        return None
    from flliper.srt.pdflip.dual_d_kv_stage import d_incoming_tokens

    actor.gmin = sched._pdflip_group_min_ints
    actor.rounds += 1
    pend, busy = _pending(sched), _busy(sched)
    actor.streak = next_streak(streak=actor.streak, pend=pend, busy=busy)
    if pend and not actor.had_pending and actor.refusals == 0:
        actor.next_round = 0                    # a new prefill reads the floor now, not at the old cadence
    actor.had_pending = bool(pend)
    incoming = d_incoming_tokens(sched)
    it = intent(lent=actor.lent, pend=pend, incoming=incoming, busy=busy, streak=actor.streak,
                settle=actor.knobs.settle, min_rows=actor.knobs.min_rows)
    if it.kind == "back":
        actor.streak = 0
        actor.next_round = actor.rounds + actor.knobs.settle
        return "back" if actor.grow_to(int(actor.top), why=it.why) else None
    if it.kind == "check" and actor.rounds >= actor.next_round:
        return _size_step(sched, actor, incoming=incoming)
    return None


def quiesce_for_sleep(sched) -> bool:
    """Before D's KV pause (the sleep RPC, after the idle assert): D never sleeps lent."""
    actor = _ACTOR
    if actor is None:
        return False
    actor.streak, actor.next_round, actor.had_pending = 0, 0, False
    return actor.quiesce()
