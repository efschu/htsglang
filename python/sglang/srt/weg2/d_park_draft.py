"""H91d (Nutzer-Entscheid 25.09.: "Ausnahme nur fuers Parken"): the MTP
draft KV of a PARKED D request travels with it; nothing else does.

THE PATH WITHOUT THIS (measured at the code, not assumed):

* D's draft KV lives in the draft runner's own pool, indexed by the TARGET's
  slot ids -- the spec worker hands the draft the target's
  ``req_to_token_pool`` and allocator (``eagle_worker_v2.alloc_memory_pool``);
  under solo placement only the solo host (NF: TP0, the 5090) has that pool,
  the shadows (the 3080 expert workers) have none.
* ``SGLANG_WEG2_HICACHE_DRAFT_TIER`` resolves to ``off`` on NF (user order
  24.09.): no draft host pool is registered, ``draft_tier_armed`` is False
  for write/load/l3, so a HiCache write-back or load-back moves TARGET rows
  only.
* ``retract_all(retain=True)`` inserts the TARGET slots into the tree; the
  draft rows simply stay where they are in the draft pool.  The flip park's
  sleep flushes the tree, a pressure park's span is evicted by the older
  request's growth -- either way the rows are gone, and nothing wrote them
  anywhere.
* At the resume the prefix comes back from host/store into NEW slots.  The
  request was armed draft-cold at its first admission on D
  (``phase_flip_draft_bootstrap.COLD_ARMED_ATTR``, never cleared by
  ``reset_for_retract``), so ``arm_draft_cold_for_admission`` SKIPS it: no
  scrub, no cold mark.  The draft then speculates over whatever the new slots'
  previous occupants left in the draft pool.

THE FIX, for parked requests only:

* SAVE (``save_parked`` before the flip park's ``retract_all``;
  ``snapshot`` + ``save_retracted`` around the pressure park's
  ``retract_decode``): of the request's committed rows ``[0, kv_committed_len)``
  the NON-ZERO draft rows are copied into one pageable host buffer per request
  (P's prompt rows are the #993 zero fill; what D computed -- tail, decode --
  is non-zero).  No pin, bounded by ``SGLANG_WEG2_D_PARK_DRAFT_KV_HOST_MIB``.
* The buffer rides on the request object, which stays in this process through
  the sleep (``weg2_d_parked`` / the #1443 hold) -- so L2 is enough.  A flip
  park that would push the held bytes over the cap goes to L3: a file in the
  HiCacheFile directory (``SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR``), written
  in the background, the RAM freed once written.  A pressure park never sleeps
  and does not go to disk: over the cap it is not saved (named).
* RESTORE (``restore_admitted``, right after ``arm_draft_cold_for_admission``
  on the admission extend): the prefix's draft rows are put back EXACTLY --
  the saved rows at their positions, zeros at every other prefix position --
  at the NEW slots, on the stream the scrub uses (ordered before the forward).
  The buffer is freed there, at ``park_abort`` and (safety net) when the
  request object dies.

RANKS: nothing here is a collective and nothing feeds a scheduling decision
(order, gate, cold marks, votes are untouched).  A rank without a draft pool
skips deterministically; the copies are rank-local bytes on the one rank that
drafts.  Off (``SGLANG_WEG2_ENABLE_D_PARK_DRAFT_KV=0``, the tier not off, or
the park off) no entry is ever made, and every hook returns before touching
anything: byte-identical to H91b.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
import weakref
import zlib
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.weg2 import d_seats

logger = logging.getLogger(__name__)

ENTRY_ATTR = "_weg2_d_park_draft"
SAVE_LINE = "WEG2-D-PARK DRAFT-SAVE"
RESTORE_LINE = "WEG2-D-PARK DRAFT-RESTORE"
DROP_LINE = "WEG2-D-PARK DRAFT-DROP"
SKIP_LINE = "WEG2-D-PARK DRAFT-SKIP"
LEDGER_LINE = "WEG2-HOST-LEDGER D-PARK-DRAFT"
L3_SUBDIR = "weg2_d_park_draft"
#: rows per device chunk: the gather scratch stays a few MiB on a card with no
#: corridor reserve (a 262k request's rows are ~256 MiB in one piece).
CHUNK_ROWS = 8192
DIGEST_SAMPLES = 64

TIER_L2 = "L2"
TIER_L3 = "L3"


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------

def enabled() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_WEG2_ENABLE_D_PARK_DRAFT_KV.get())


def armed() -> bool:
    """On group D with the park active and the draft tier OFF (with the tier
    on, the draft rows already travel with the HiCache pages)."""
    if not enabled() or not d_seats.d_park_active():
        return False
    from sglang.srt.mem_cache.hicache_storage import hicache_draft_tier_off

    return hicache_draft_tier_off()


def host_cap_bytes() -> int:
    from sglang.srt.environ import envs

    return max(0, int(envs.SGLANG_WEG2_D_PARK_DRAFT_KV_HOST_MIB.get() or 0)) << 20


def l3_dir() -> Optional[str]:
    from sglang.srt.environ import envs

    root = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get()
    return os.path.join(str(root), L3_SUBDIR) if root else None


# ---------------------------------------------------------------------------
# the pool geometry
# ---------------------------------------------------------------------------

_SKIP_SAID: set = set()


def _skip_once(key: str, why: str) -> None:
    if key in _SKIP_SAID:
        return
    _SKIP_SAID.add(key)
    logger.info("%s %s", SKIP_LINE, why)


def draft_buffers(sched) -> Optional[List[torch.Tensor]]:
    """Byte views ``[slots, ...]`` of every draft KV buffer this rank holds,
    in the scrub's own geometry (``draft_kv_layer_ids``), or None when this
    rank has no draft pool it can address by target slot."""
    from sglang.srt.managers.phase_flip_draft_bootstrap import (
        draft_kv_layer_ids,
        draft_kv_pool,
    )

    pool = draft_kv_pool(getattr(sched, "draft_worker", None))
    if pool is None:
        _skip_once("nopool", "this rank holds no draft KV pool (solo shadow / expert "
                   "worker): the park carries no draft rows here, the solo host does")
        return None
    if getattr(pool, "weg2_slot_mapper", None) is not None:
        _skip_once("mapped", "the draft pool is slot-mapped (DFlash window), not indexed "
                   "by target slots: no park carry on this form")
        return None
    out: List[torch.Tensor] = []
    for layer_id in draft_kv_layer_ids(pool):
        k = pool.get_key_buffer(layer_id)
        out.append(k.view(torch.uint8))
        v = pool.get_value_buffer(layer_id)
        if v is not None and v.data_ptr() != k.data_ptr():
            out.append(v.view(torch.uint8))
    return out or None


def _row_bytes(buf: torch.Tensor) -> int:
    n = 1
    for d in buf.shape[1:]:
        n *= int(d)
    return n


def _order_after_forward(sched) -> None:
    """The rows a park reads were written by the forward stream; the copies run
    on the current stream, so it waits there (GPU-side, no host stall)."""
    fs = getattr(sched, "forward_stream", None)
    if fs is None or not torch.cuda.is_available():
        return
    try:
        cur = torch.cuda.current_stream()
        if fs != cur:
            cur.wait_stream(fs)
    except Exception:  # noqa: BLE001 - a stand-in stream: nothing to order
        pass


# ---------------------------------------------------------------------------
# the entry and its ledger post
# ---------------------------------------------------------------------------

class ParkDraftLedger:
    """The host post ``d_park_draft``: anonymous, unpinned, on demand."""

    def __init__(self) -> None:
        self.l2 = 0
        self.l3 = 0
        self.peak_l2 = 0
        self.live = 0
        self._lock = threading.Lock()

    def add(self, tier: str, nbytes: int) -> None:
        with self._lock:
            if tier == TIER_L2:
                self.l2 += nbytes
                self.peak_l2 = max(self.peak_l2, self.l2)
            else:
                self.l3 += nbytes
            self.live += 1

    def move_l2_to_l3(self, nbytes: int) -> None:
        with self._lock:
            self.l2 -= nbytes
            self.l3 += nbytes

    def remove(self, tier: str, nbytes: int) -> None:
        with self._lock:
            if tier == TIER_L2:
                self.l2 -= nbytes
            else:
                self.l3 -= nbytes
            self.live -= 1

    def line(self, event: str, rid: str, nbytes: int) -> str:
        gib = float(1 << 30)
        return (
            f"{LEDGER_LINE} event={event} rid={str(rid)[:16]} bytes={int(nbytes)} "
            f"held_l2={self.l2 / gib:.4f} GiB held_l3={self.l3 / gib:.4f} GiB "
            f"peak_l2={self.peak_l2 / gib:.4f} GiB live={self.live} "
            f"cap={host_cap_bytes() / gib:.3f} GiB -- anonym, ungepinnt, bedarfsweise "
            f"(der Posten 'd_park_draft'; nicht in der Arm-Summe)"
        )


def ledger_of(sched) -> ParkDraftLedger:
    led = getattr(sched, "_weg2_d_park_draft_ledger", None)
    if led is None:
        led = sched._weg2_d_park_draft_ledger = ParkDraftLedger()
    return led


_L3_POOL: Optional[ThreadPoolExecutor] = None


def _l3_writer() -> ThreadPoolExecutor:
    global _L3_POOL
    if _L3_POOL is None:
        _L3_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weg2-park-draft-l3")
    return _L3_POOL


def _release(ledger: ParkDraftLedger, state: dict) -> None:
    """Idempotent release of one entry (explicit or at the request's death)."""
    with state["lock"]:
        if state.get("released"):
            return
        state["released"] = True
        ledger.remove(state["tier"], state["nbytes"])
    path = state.get("path")
    fut = state.get("future")
    if path:
        def _unlink(_f=None, _p=path):
            try:
                os.unlink(_p)
            except OSError:
                pass
        if fut is not None and not fut.done():
            fut.add_done_callback(_unlink)
        else:
            _unlink()


class ParkDraftEntry:
    """One parked request's draft rows: ``positions`` (sorted row positions in
    the request's context) and one ``[n, row_bytes]`` uint8 block per draft
    buffer, in RAM (L2) or in a file (L3)."""

    def __init__(self, *, rid: str, site: str, length: int, positions: torch.Tensor,
                 blocks: List[torch.Tensor], ledger: ParkDraftLedger) -> None:
        self.rid = str(rid)
        self.site = site
        self.length = int(length)
        self.positions = positions
        self.row_bytes = [int(b.shape[1]) for b in blocks]
        self.nbytes = int(sum(b.numel() for b in blocks)) + int(positions.numel()) * 8
        self._blocks: Optional[List[torch.Tensor]] = blocks
        self._lock = threading.Lock()  # guards _blocks and the state (shared with _release)
        self.sample = _sample_rows(int(positions.numel()))
        self.digest = _digest_blocks(blocks, self.sample)
        self._state = {"tier": TIER_L2, "nbytes": self.nbytes, "released": False,
                       "path": None, "future": None, "lock": self._lock}
        ledger.add(TIER_L2, self.nbytes)
        self._ledger = ledger
        self._finalizer = weakref.finalize(self, _release, ledger, self._state)

    @property
    def rows(self) -> int:
        return int(self.positions.numel())

    @property
    def tier(self) -> str:
        return self._state["tier"]

    @property
    def path(self) -> Optional[str]:
        return self._state["path"]

    def spill(self, directory: str) -> Future:
        """L3: write the blocks to one file, free the RAM once written."""
        os.makedirs(directory, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", self.rid)[:96]
        path = os.path.join(directory, f"{os.getpid()}-{safe}-{id(self):x}.bin")
        self._state["path"] = path
        blocks = self._blocks

        def _write() -> None:
            try:
                with open(path, "wb") as f:
                    for b in blocks:
                        f.write(memoryview(b.numpy()).cast("B"))
            except OSError as e:  # the RAM copy stays: the entry remains L2
                logger.warning("%s rid=%s L3 write to %s failed (%s): kept in L2",
                               SAVE_LINE, self.rid[:16], path, e)
                return
            with self._lock:
                if not self._state["released"] and self._state["tier"] == TIER_L2:
                    self._blocks = None
                    self._state["tier"] = TIER_L3
                    self._ledger.move_l2_to_l3(self.nbytes)

        fut = _l3_writer().submit(_write)
        self._state["future"] = fut
        return fut

    def blocks(self) -> Tuple[List[torch.Tensor], str]:
        """The blocks and where they came from (RAM while still there)."""
        with self._lock:
            if self._blocks is not None:
                return self._blocks, TIER_L2
        fut = self._state.get("future")
        if fut is not None:
            fut.result()
        out = []
        with open(self._state["path"], "rb") as f:
            for rb in self.row_bytes:
                t = torch.empty((self.rows, rb), dtype=torch.uint8)
                f.readinto(memoryview(t.numpy()).cast("B"))
                out.append(t)
        return out, TIER_L3

    def release(self) -> None:
        self._finalizer()


def _sample_rows(n: int) -> torch.Tensor:
    if n <= 0:
        return torch.empty((0,), dtype=torch.long)
    k = min(n, DIGEST_SAMPLES)
    return torch.linspace(0, n - 1, k).round().long().unique()


def _digest_blocks(blocks: Sequence[torch.Tensor], sel: torch.Tensor) -> int:
    crc = 0
    for b in blocks:
        crc = zlib.crc32(b[sel].contiguous().numpy().tobytes(), crc)
    return crc & 0xFFFFFFFF


def entry_of(req) -> Optional[ParkDraftEntry]:
    return getattr(req, ENTRY_ATTR, None)


def drop(req, reason: str, ledger: Optional[ParkDraftLedger] = None) -> int:
    """Free a request's park buffer (abort, superseded).  Returns bytes."""
    entry = entry_of(req)
    if entry is None:
        return 0
    setattr(req, ENTRY_ATTR, None)
    nbytes = entry.nbytes
    entry.release()
    logger.info("%s rid=%s reason=%s bytes=%d", DROP_LINE, str(entry.rid)[:16], reason, nbytes)
    if ledger is not None:
        logger.info(ledger.line("drop", entry.rid, nbytes))
    return nbytes


# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------

def _gather_nonzero(bufs: Sequence[torch.Tensor], slots: torch.Tensor):
    """Positions (host, sorted) of the non-zero rows among ``slots`` and their
    bytes per buffer (host, pageable)."""
    n = int(slots.numel())
    masks = []
    for s in range(0, n, CHUNK_ROWS):
        idx = slots[s:s + CHUNK_ROWS]
        m = None
        for buf in bufs:
            nz = buf[idx].reshape(int(idx.numel()), -1).ne(0).any(dim=1)
            m = nz if m is None else (m | nz)
        masks.append(m)
    mask = torch.cat(masks).cpu() if masks else torch.zeros((0,), dtype=torch.bool)
    positions = mask.nonzero().flatten().long()
    k = int(positions.numel())
    blocks = [torch.empty((k, _row_bytes(b)), dtype=torch.uint8) for b in bufs]
    pos_dev = positions.to(slots.device)
    for s in range(0, k, CHUNK_ROWS):
        idx = slots[pos_dev[s:s + CHUNK_ROWS]]
        for buf, blk in zip(bufs, blocks):
            blk[s:s + int(idx.numel())].copy_(buf[idx].reshape(int(idx.numel()), -1))
    return positions, blocks


def _save_one(sched, bufs, req, *, pool_idx, length: int, site: str) -> Optional[ParkDraftEntry]:
    ledger = ledger_of(sched)
    if entry_of(req) is not None:
        drop(req, "superseded", ledger)
    if pool_idx is None or length <= 0:
        return None
    t0 = time.perf_counter()
    req_to_token = sched.req_to_token_pool.req_to_token
    slots = req_to_token[int(pool_idx), :int(length)].to(torch.long)
    positions, blocks = _gather_nonzero(bufs, slots)
    if int(positions.numel()) == 0:
        logger.info("%s rid=%s site=%s rows=0 of=%d bytes=0 tier=none (no draft row D "
                    "computed yet)", SAVE_LINE, str(req.rid)[:16], site, length)
        return None
    nbytes = int(sum(b.numel() for b in blocks)) + int(positions.numel()) * 8
    cap = host_cap_bytes()
    over = ledger.l2 + nbytes > cap
    directory = l3_dir()
    if over and (site != d_seats.SITE_FLIP or directory is None):
        # A pressure park never sleeps: no disk for it (L3 only across a
        # sleep); over the cap it is simply not carried -- named.
        why = "pressure park, L3 only across a sleep" if site != d_seats.SITE_FLIP else (
            "no HiCacheFile directory for L3")
        logger.info("%s rid=%s site=%s rows=%d of=%d bytes=%d tier=none (over the L2 cap "
                    "%d B, %s) -- resume drafts over the #993 cold rows", SAVE_LINE,
                    str(req.rid)[:16], site, int(positions.numel()), length, nbytes, cap, why)
        return None
    entry = ParkDraftEntry(rid=str(req.rid), site=site, length=length,
                           positions=positions, blocks=blocks, ledger=ledger)
    tier = TIER_L2
    if over:
        entry.spill(directory)
        tier = TIER_L3
    setattr(req, ENTRY_ATTR, entry)
    logger.info("%s rid=%s site=%s rows=%d of=%d bytes=%d tier=%s digest=%08x ms=%.1f",
                SAVE_LINE, str(req.rid)[:16], site, entry.rows, length, entry.nbytes, tier,
                entry.digest, (time.perf_counter() - t0) * 1000.0)
    logger.info(ledger.line("save", entry.rid, entry.nbytes))
    return entry


def save_parked(sched, reqs: Sequence, *, site: str) -> int:
    """The flip park: called BEFORE ``retract_all`` (the slots are still the
    request's).  Returns the number of entries made."""
    if not reqs or not armed():
        return 0
    bufs = draft_buffers(sched)
    if bufs is None:
        return 0
    _order_after_forward(sched)
    made = 0
    for req in reqs:
        if _save_one(sched, bufs, req, pool_idx=getattr(req, "req_pool_idx", None),
                     length=int(getattr(req, "kv_committed_len", 0) or 0), site=site):
            made += 1
    return made


def snapshot(sched, batch) -> Optional[Dict[int, Tuple[int, int]]]:
    """The pressure park, half one: BEFORE ``retract_decode`` releases them,
    every batch request's slot row and committed length.  None when off."""
    if not armed():
        return None
    snap = {}
    for req in list(getattr(batch, "reqs", None) or ()):
        pidx = getattr(req, "req_pool_idx", None)
        if pidx is not None:
            snap[id(req)] = (int(pidx), int(getattr(req, "kv_committed_len", 0) or 0))
    return snap


def save_retracted(sched, retracted: Sequence, snap) -> int:
    """The pressure park, half two: right after ``retract_decode`` -- nothing
    ran since, so the rows and the ``req_to_token`` rows are still the
    retracted requests' (the next forward is ordered after these copies)."""
    if not snap or not retracted:
        return 0
    bufs = draft_buffers(sched)
    if bufs is None:
        return 0
    _order_after_forward(sched)
    made = 0
    for req in retracted:
        got = snap.get(id(req))
        if got is None:
            continue
        if _save_one(sched, bufs, req, pool_idx=got[0], length=got[1],
                     site=d_seats.SITE_PRESSURE):
            made += 1
    return made


# ---------------------------------------------------------------------------
# RESTORE
# ---------------------------------------------------------------------------

def _prefix_len(req) -> int:
    from sglang.srt.managers.schedule_batch import prefix_len

    return int(prefix_len(req))


def _restore_one(sched, bufs, req, entry: ParkDraftEntry) -> None:
    t0 = time.perf_counter()
    n_prefix = _prefix_len(req)
    pool_idx = getattr(req, "req_pool_idx", None)
    if pool_idx is None or n_prefix <= 0:
        logger.info("%s rid=%s rows=0 prefix=%d tier=none (no cached prefix at the "
                    "resume: the extend recomputes every draft row)", RESTORE_LINE,
                    str(entry.rid)[:16], n_prefix)
        return
    blocks, tier = entry.blocks()
    if [int(b.shape[1]) for b in blocks] != [_row_bytes(b) for b in bufs] or len(blocks) != len(bufs):
        logger.warning("%s rid=%s rows=0 tier=%s REFUSED: draft geometry changed since the "
                       "park (%s vs %s)", RESTORE_LINE, str(entry.rid)[:16], tier,
                       [int(b.shape[1]) for b in blocks], [_row_bytes(b) for b in bufs])
        return
    host_digest = _digest_blocks(blocks, entry.sample)
    req_to_token = sched.req_to_token_pool.req_to_token
    slots = req_to_token[int(pool_idx), :n_prefix].to(torch.long)
    dev = slots.device
    m = int((entry.positions < n_prefix).sum())  # positions are sorted
    pos = entry.positions[:m]
    zero = torch.ones((n_prefix,), dtype=torch.bool)
    zero[pos] = False
    zpos = zero.nonzero().flatten()
    for s in range(0, int(zpos.numel()), CHUNK_ROWS):
        idx = slots[zpos[s:s + CHUNK_ROWS].to(dev)]
        for buf in bufs:
            buf[idx] = 0
    for s in range(0, m, CHUNK_ROWS):
        idx = slots[pos[s:s + CHUNK_ROWS].to(dev)]
        k = int(idx.numel())
        for buf, blk in zip(bufs, blocks):
            buf[idx] = blk[s:s + k].to(dev).view(k, *buf.shape[1:])
    sel = entry.sample[entry.sample < m]
    want = _digest_blocks([b[:m] for b in blocks], sel)
    got_crc = 0
    if int(sel.numel()) > 0:
        idx = slots[pos[sel].to(dev)]
        for buf in bufs:
            got_crc = zlib.crc32(
                buf[idx].reshape(int(idx.numel()), -1).cpu().contiguous().numpy().tobytes(),
                got_crc,
            )
    got_crc &= 0xFFFFFFFF
    ok = host_digest == entry.digest and got_crc == want
    verdict = "match" if ok else (
        f"MISMATCH(saved={entry.digest:08x} host={host_digest:08x} "
        f"want={want:08x} device={got_crc:08x})"
    )
    log = logger.info if ok else logger.warning
    log("%s rid=%s site=%s rows=%d of=%d zeroed=%d prefix=%d tier=%s digest=%s ms=%.1f",
        RESTORE_LINE, str(entry.rid)[:16], entry.site, m, entry.rows, int(zpos.numel()),
        n_prefix, tier, verdict, (time.perf_counter() - t0) * 1000.0)


def restore_admitted(sched, batch) -> int:
    """The resume: on the admission extend, after the #861 cold arming (whose
    scrub, where it runs, these writes follow) and before the forward.  The
    common path is one attribute read per request."""
    reqs = [r for r in list(getattr(batch, "reqs", None) or ()) if entry_of(r) is not None]
    if not reqs:
        return 0
    bufs = draft_buffers(sched)
    ledger = ledger_of(sched)
    done = 0
    for req in reqs:
        entry = entry_of(req)
        setattr(req, ENTRY_ATTR, None)
        try:
            if bufs is not None:
                _restore_one(sched, bufs, req, entry)
                done += 1
        finally:
            nbytes = entry.nbytes
            entry.release()
            logger.info(ledger.line("restore", entry.rid, nbytes))
    return done


def drop_all(sched, reqs: Sequence, reason: str) -> int:
    """Abort paths: free every listed request's park buffer."""
    mine = [r for r in (reqs or ()) if entry_of(r) is not None]
    if not mine:
        return 0
    ledger = ledger_of(sched)
    return sum(drop(r, reason, ledger) for r in mine)
