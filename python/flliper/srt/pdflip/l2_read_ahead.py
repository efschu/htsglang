"""S53 (19.10.) L2-READ-AHEAD: the next P request's prefix goes from the disk (L3) into the shared arena (L2) while
its predecessor still computes -- the H91c device plan stays as it is, only the disk -> host leg moves forward.

THE GAP (rc14 boot ..._92dd662831_1019_010742, P epoch 05:18:55-05:20:26Z, S52's census: PP0 busy 63 %). The front's
pool plan (H91 part C, ``phase_policy.p_overlap_admits``: est_prompt against P's 262144-token pool) lets one 150-200k
prompt onto P at a time; the next one leaves the front only when its predecessor's leg 1 has come back
(``P-PHASE epoch=1 ... pool_holds=290``). Then everything runs one after the other:
    pdflip-0-21 served 05:19:10.27 (front) -> pdflip-0-22 dispatched, its store read on PP0
    ``PDFLIP-READ-STAGES req=pdflip-0-22 pages=2087 l3fill_pages=2087 total_ms=2403 l3fill_ms=1676 extra_ms=718``
    -> ``PF TOLD-ACKED after=0.02s`` 05:19:13 -> admitted; PP0's card idle ``#PGAP gpu_gap_ms=5689.5``.
2087 of 2087 pages came from the disk although pdflip-0-22 had waited in the front since 05:18:53 -- 17 s in which
nothing read its prefix. A read that finds its pages in L2 is an order faster: ``pdflip-1-27 pages=2026
l3fill_pages=82 total_ms=216`` in the same epoch.

THE RULE. While the pool plan HOLDS the queue head (the one place the front knows "this request waits for P's
device room, not for P's compute"), the front hints that head to P (the H43 hint RPC, ``read_ahead`` set, ``ple``
off -- P is awake and its PLE ring belongs to the running request). PP0 alone (the arenas are ONE shared file per
width for every rank of both groups, ``HiCacheFile._arena_for``: a page one rank completes, every rank reads) fills
the head's stored prefix -- the KV pages and the QSA index pages of ``[0, start_hint)``, keyed exactly as the fetch
keys them (``pdflip_store_told._probe_key``: the tree's bigram RadixKey) -- from L3 into the arena with the
UNCHANGED ``HiCacheFile.arena_fill_from_disk``:
  * no reference is taken (``take_ref=False``): the pages land COMPLETE and evictable, nothing is pinned, no slot is
    withheld from the running request's publish -- the later real read references them like any L2 hit (or reads
    from L3 again if the clock took them meanwhile: only the gain is lost);
  * no room wait (``room_wait=False``): a FULL arena ends the read-ahead, no READ-ROOM post asks a tree to spill;
  * no kept page spent (S53r, ``_FILL_CTX.spare_kept``): its room comes from unkept pages only, never from the
    hand-off / park pages #248e stage (ii) gives a real fill -- D's hand-off and park reads keep their L2 copies;
  * pages already COMPLETE in L2 are not asked (LMCache #3540: no I/O for what the faster tier holds) and stay out
    of this fill's own room eviction (``fill_protect_scope``);
  * the prefix walk ends at the first page neither in L2 nor on disk (the real read ends there too).
WHEN: not at the hint but -- once no other rid's store read or admission is still ahead (S53r,
``predecessor_busy``) -- once the running prefill on PP0 has at most ``LEAD`` tokens left (default two chunks,
``2 * chunked_prefill_size`` ~ 6 s of PP0 compute against a 2.4-3.0 s cold read) -- early enough to finish before
the predecessor does, late enough that the predecessor's own chunk publish does not clock the pages out again.
ONE successor: one job runs, one waits; a newer hint replaces the waiting one. The job is dropped when P is dormant
(flipped away), when the real read of that rid has started (``tree.ongoing_prefetch``), or after ``MAX_AGE_S``.

HOST RAM. The arena is the existing fixed /dev/shm file (``FLLIPER_HICACHE_ARENA_GIB``); a read-ahead puts pages
into slots the clock frees, it allocates no host buffer (the read targets the slot itself). The QSA fill likewise.
The Mamba anchor blob is NOT read ahead: its arena has 112 slots and carries D's park anchors (EVICT-KEEP, W31).

R11 / rank agreement: no decision moves. The front decides (the held head); PP0 only warms the shared L2; every
rank's real read, PP0's told and the followers' acks run as before -- a read-ahead changes how fast a read is, never
what it answers. SOLL unchanged.

Switch ``FLLIPER_PDFLIP_L2_READ_AHEAD`` (front and P, default ON; 0 = no hint is sent and a hint is ignored, byte for
byte). ``FLLIPER_PDFLIP_L2_READ_AHEAD_LEAD_TOKENS`` (0 = two chunks). Instruments: front ``PDFLIP L2-READ-AHEAD
HINT``, PP0 ``PDFLIP L2-READ-AHEAD`` (queued / done with resident / asked / landed per arena / skipped verdicts);
the real read's ``PDFLIP-READ-STAGES ... l3fill_pages`` of that rid shows the hit.
"""
from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ATTR = "_pdflip_l2_read_ahead"
#: a prefix shorter than this is read in a few ms anyway (64 pages)
MIN_SPAN_TOKENS = 4096
#: a job older than this is a request the front no longer holds for P (flip, abort): dropped
MAX_AGE_S = 180.0
POLL_S = 0.2
#: S53r (review 19.10.): the front posts the read-ahead hint this long after the pool plan first held the head. The
#: hold begins in the same drain round that dispatched the predecessor's leg 1, so without the delay the hint's POST
#: and the predecessor's leg-1 POST reach P's HTTP loop together -- and the hint's synchronous tokenization (H43:
#: ``PLE-HINT ... ms=651`` for a 199k prompt on rc14) can run before the predecessor's own, in its critical path; it
#: can also reach PP0's scheduler before the predecessor and start the fill beside the predecessor's store read.
#: 1.5 s > P's tokenization of a 200k prompt (0.3-0.65 s) + the leg-1 pre-POST awaits; the read-ahead waits for the
#: predecessor's read and admission anyway (``predecessor_busy``), so the delay costs no gain.
HINT_DELAY_S = 1.5


def enabled() -> bool:
    try:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_L2_READ_AHEAD.get())
    except Exception:  # noqa: BLE001 -- no registry: the default
        return True


def lead_tokens(scheduler) -> int:
    """How many prefill tokens may still be left on PP0 when the read-ahead starts (0 = two chunks)."""
    try:
        from flliper.srt.environ import envs

        v = int(envs.FLLIPER_PDFLIP_L2_READ_AHEAD_LEAD_TOKENS.get() or 0)
    except Exception:  # noqa: BLE001
        v = 0
    if v > 0:
        return v
    return 2 * max(1, int(getattr(scheduler, "chunked_prefill_size", 16384) or 16384))


# ------------------------------------------------------------------------------------------------ front side


def hint_wanted(*, awake: Optional[str], skip_leg1: bool, span: Optional[int], sent: bool,
                client_gone: bool = False) -> bool:
    """The pool plan holds this queue head while P is awake: read its stored prefix ahead (once per request)."""
    if sent or client_gone or skip_leg1 or awake != "P":
        return False
    return span is not None and int(span) >= MIN_SPAN_TOKENS


def hint_body(path: str, payload: Any, span: int) -> Optional[dict]:
    """The H43 hint body (the leg-1 path and payload, P tokenizes it as the POST will be) with ``read_ahead`` set
    and the PLE admission off. None = not a leg-1 shape."""
    from flliper.srt.pdflip.ple_admit_hint import ple_hint_body

    body = ple_hint_body(path, payload, start_hint=int(span))
    if body is None:
        return None
    body["read_ahead"] = True
    body["ple"] = False
    return body


# ------------------------------------------------------------------------------------------------ P side


def armed(scheduler) -> bool:
    """PP0 of an awake group, attention-TP rank 0: the one rank that fills the shared arenas."""
    try:
        if not enabled():
            return False
        ps = getattr(scheduler, "ps", None)
        if int(getattr(ps, "pp_rank", 0) or 0) != 0:
            return False
        if int(getattr(ps, "tp_rank", getattr(scheduler, "tp_rank", 0)) or 0) != 0:
            return False
        if bool(getattr(scheduler, "pdflip_dormant", False)):
            return False
        return True
    except Exception:  # noqa: BLE001
        return False


def predecessor_busy(scheduler, rid: str) -> Optional[str]:
    """S53r (review 19.10.): why the read-ahead of ``rid`` must not start yet, None = it may.

    ``remaining_prefill_tokens`` answers 0 both when the predecessor's last chunk is out (the read-ahead's point)
    AND when the predecessor has not begun chunking yet -- its store read still running or its admission (TOLD)
    still pending, ``chunked_req`` None. That is exactly when the hint arrives (the pool plan holds the head in the
    drain round that dispatched the predecessor), so the old gate started the fill at once: beside the
    predecessor's cold read (one ``_l3_executor``, one disk, its stems off the predecessor fill's room eviction
    via ``_FILL_ACTIVE``) -- in the critical path -- and early enough that the predecessor's chunk publish clocks
    the pages out again. Wait instead while
      * another rid's real store read is registered on this rank (``tree.ongoing_prefetch``) -> ``read``;
      * no chunked request runs and another request sits in the waiting queue (read done, admission pending)
        -> ``admission``.
    Attribute reads from the worker thread (at worst one pass stale); a timing gate, never a verdict."""
    tree = getattr(scheduler, "tree_cache", None)
    try:
        ongoing = list(getattr(tree, "ongoing_prefetch", None) or ())
    except Exception:  # noqa: BLE001 -- unreadable: no wait on it
        ongoing = []
    if any(str(k) != rid for k in ongoing):
        return "read"
    if getattr(scheduler, "chunked_req", None) is None:
        try:
            waiting = list(getattr(scheduler, "waiting_queue", None) or ())
        except Exception:  # noqa: BLE001
            waiting = []
        if any(str(getattr(r, "rid", "")) != rid for r in waiting):
            return "admission"
    return None


def remaining_prefill_tokens(scheduler) -> int:
    """Prefill tokens PP0 still has to schedule for its chunked request (0 = none running / its last chunk is out).
    Read from the worker thread: an attribute read, at worst one pass stale -- a timing heuristic, never a verdict."""
    req = getattr(scheduler, "chunked_req", None)
    if req is None:
        return 0
    try:
        return max(0, len(req.origin_input_ids) - len(req.fill_ids))
    except Exception:  # noqa: BLE001
        return 0


def span_hashes(scheduler, ids, span: int) -> List[str]:
    """The page keys of ``ids[:span]`` as the fetch keys them (bigram RadixKey on a bigram tree, default
    namespace) -- the same derivation as ``pdflip_store_told._probe_key`` + ``get_hash_str``."""
    from array import array

    from flliper.srt.mem_cache.radix_cache import RadixKey
    from flliper.srt.mem_cache.utils import get_hash_str

    page = max(1, int(getattr(scheduler, "page_size", 1) or 1))
    tree = getattr(scheduler, "tree_cache", None)
    bigram = bool(getattr(tree, "is_eagle", False))
    usable = len(ids) - (1 if bigram else 0)
    span = (min(int(span), max(0, usable)) // page) * page
    if span <= 0:
        return []
    key = RadixKey(array("q", ids[: span + (1 if bigram else 0)]), extra_key=None, is_bigram=bigram)
    hv = get_hash_str(key, None, page_size=page)
    return list(hv)[: span // page]


def fill_arena(backend, arena, stems: List[str], total_bytes: int) -> Dict[str, int]:
    """L3 -> L2 for ``stems`` (a prefix walk): the pages not COMPLETE in ``arena`` are filled by the unchanged
    ``arena_fill_from_disk`` -- no reference, no room wait; the whole span stays off this fill's room eviction."""
    from flliper.srt.mem_cache import hicache_storage as _hs
    from flliper.srt.mem_cache.hicache_storage import fill_protect_scope

    n = len(stems)
    out = {"pages": n, "resident": 0, "asked": 0, "landed": 0}
    if n == 0 or arena is None:
        return out
    found = arena.find_slots(list(stems))
    resident = [int(sl) >= 0 and int(st) == 2 for sl, st in found]
    out["resident"] = sum(resident)
    missing = [s for s, r in zip(stems, resident) if not r]
    if not missing:
        return out
    out["asked"] = len(missing)
    op = SimpleNamespace(hash_value=(), _pdflip_protect_stems=list(stems))
    _prev_spare = getattr(_hs._FILL_CTX, "spare_kept", False)
    _hs._FILL_CTX.spare_kept = True   # S53r: a speculative fill never spends kept hand-off / park pages (#248e ii)
    try:
        with fill_protect_scope(backend, op):
            got = backend.arena_fill_from_disk(arena, missing, int(total_bytes), prefix=True, room_wait=False,
                                               take_ref=False)
    finally:
        _hs._FILL_CTX.spare_kept = _prev_spare
    out["landed"] = sum(1 for g in list(got) if isinstance(g, int) and not isinstance(g, bool) and g >= 0)
    return out


def fill_span(scheduler, ids, span: int) -> Dict[str, Any]:
    """The read-ahead of one request on PP0: its KV pages and its QSA index pages of ``[0, span)``."""
    from flliper.srt.managers.cache_controller import pdflip_suffixed_stems

    res: Dict[str, Any] = {}
    tree = getattr(scheduler, "tree_cache", None)
    ctrl = getattr(tree, "cache_controller", None)
    backend = getattr(ctrl, "storage_backend", None)
    pool = getattr(ctrl, "mem_pool_host", None)
    if backend is None or pool is None or not getattr(pool, "arena_read", False):
        res["verdict"] = "no_arena"
        return res
    hashes = span_hashes(scheduler, ids, span)
    res["pages"] = len(hashes)
    if not hashes:
        res["verdict"] = "short"
        return res
    if not pool.ensure_bound(backend, role="kv"):
        res["verdict"] = "unbound"
        return res
    res["kv"] = fill_arena(backend, pool.arena, pdflip_suffixed_stems(backend, hashes), int(pool._page_bytes))
    win = getattr(backend, "canonical_qsa_page", None)
    if win is not None:
        try:
            from flliper.srt.mem_cache.hicache_storage import PoolName

            qarena = backend._arena_for(int(win.total_bytes))
            if qarena is not None:
                qstems = [backend._get_suffixed_key(f"{h}.{PoolName.QSA_INDEXER}") for h in hashes]
                res["qsa"] = fill_arena(backend, qarena, qstems, int(win.total_bytes))
        except Exception as exc:  # noqa: BLE001 -- the QSA half is a gain, never a wall
            res["qsa_error"] = repr(exc)[:120]
    res["verdict"] = "done"
    return res


class ReadAhead:
    """PP0's read-ahead worker: one job runs, one waits (a newer hint replaces the waiting one)."""

    def __init__(self, scheduler, fill: Optional[Callable] = None, now: Callable[[], float] = time.monotonic,
                 poll_s: float = POLL_S, start: bool = True) -> None:
        self.scheduler = scheduler
        self._fill = fill or fill_span
        self._now = now
        self._poll_s = float(poll_s)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self.waiting: Optional[dict] = None
        self.running: Optional[str] = None
        self.done_rids: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
        self._stop = False
        self._thread = None
        if start:
            self._thread = threading.Thread(target=self._loop, name="pdflip-l2-read-ahead", daemon=True)
            self._thread.start()

    def _bump(self, k: str) -> int:
        self.counters[k] = self.counters.get(k, 0) + 1
        return self.counters[k]

    # -- scheduler thread ------------------------------------------------------------------------
    def submit(self, rid: str, ids, span: int) -> str:
        with self._lock:
            if rid == self.running or rid in self.done_rids or (self.waiting and self.waiting["rid"] == rid):
                verdict = "dup"
            else:
                verdict = "replaced" if self.waiting is not None else "queued"
                self.waiting = {"rid": rid, "ids": ids, "span": int(span), "t": self._now()}
        n = self._bump(verdict)
        if verdict != "dup" and (n <= 16 or n % 64 == 0):
            logger.info("PDFLIP L2-READ-AHEAD %s rid=%s span=%d lead_tokens=%d (n=%d): the front holds it for P's "
                        "device room -- its stored prefix goes L3 -> L2 before its leg 1",
                        verdict.upper(), rid, int(span), lead_tokens(self.scheduler), n)
        self._wake.set()
        return verdict

    # -- worker ----------------------------------------------------------------------------------
    def _take(self) -> Optional[dict]:
        with self._lock:
            job, self.waiting = self.waiting, None
            if job is not None:
                self.running = job["rid"]
            return job

    def _skip_reason(self, job: dict) -> Optional[str]:
        s = self.scheduler
        if bool(getattr(s, "pdflip_dormant", False)):
            return "dormant"
        if self._now() - job["t"] > MAX_AGE_S:
            return "stale"
        tree = getattr(s, "tree_cache", None)
        try:
            if job["rid"] in (getattr(tree, "ongoing_prefetch", None) or {}):
                return "read_started"
            # S53r: the real request is on this rank already (no read registered for it): its own path serves it
            if any(str(getattr(r, "rid", "")) == job["rid"] for r in list(getattr(s, "waiting_queue", None) or ())):
                return "arrived"
        except Exception:  # noqa: BLE001
            pass
        return None

    def step(self, job: dict) -> Optional[str]:
        """One look at a job: None = wait (lead not reached), else the verdict it ended with."""
        why = self._skip_reason(job)
        if why is not None:
            return "skip:" + why
        busy = predecessor_busy(self.scheduler, job["rid"])
        if busy is not None:
            if job.get("busy") != busy:
                job["busy"] = busy
                self._bump("wait_" + busy)
            return None
        rem = remaining_prefill_tokens(self.scheduler)
        if rem > lead_tokens(self.scheduler):
            return None
        t0 = self._now()
        res = self._fill(self.scheduler, job["ids"], job["span"])
        ms = (self._now() - t0) * 1000.0
        n = self._bump("ran")
        if n <= 32 or n % 64 == 0:
            kv = res.get("kv") or {}
            q = res.get("qsa") or {}
            logger.info("PDFLIP L2-READ-AHEAD DONE rid=%s span=%d pages=%s kv=resident:%s,asked:%s,landed:%s "
                        "qsa=resident:%s,asked:%s,landed:%s verdict=%s remaining_at_start=%d waited_ms=%.0f "
                        "waited_for=%s fill_ms=%.0f (n=%d; no reference taken -- the real read finds the pages "
                        "COMPLETE in L2)",
                        job["rid"], job["span"], res.get("pages", 0), kv.get("resident", "-"), kv.get("asked", "-"),
                        kv.get("landed", "-"), q.get("resident", "-"), q.get("asked", "-"), q.get("landed", "-"),
                        res.get("verdict"), rem, (t0 - job["t"]) * 1000.0, job.get("busy") or "-", ms, n)
        return str(res.get("verdict") or "done")

    def run_job(self, job: dict) -> str:
        """Hold the job until its lead (or a skip), then fill. Called on the worker thread."""
        try:
            while not self._stop:
                with self._lock:
                    if self.waiting is not None:
                        # a newer hint arrived while this one waited for its lead: the newer one wins (one successor)
                        self._bump("superseded")
                        return "superseded"
                verdict = self.step(job)
                if verdict is not None:
                    if verdict.startswith("skip:"):
                        n = self._bump(verdict)
                        if n <= 8 or n % 64 == 0:
                            logger.info("PDFLIP L2-READ-AHEAD SKIPPED rid=%s why=%s (n=%d)", job["rid"],
                                        verdict[5:], n)
                    return verdict
                time.sleep(self._poll_s)
            return "stopped"
        except Exception as exc:  # noqa: BLE001 -- a lost read-ahead costs the gain only
            n = self._bump("errors")
            if n <= 8 or n % 64 == 0:
                logger.warning("PDFLIP L2-READ-AHEAD raised (%d so far) rid=%s: %r -- the real read reads L3 as "
                               "before", n, job.get("rid"), exc, exc_info=n <= 2)
            return "error"
        finally:
            with self._lock:
                self.running = None
                self.done_rids[job["rid"]] = self._now()
                while len(self.done_rids) > 64:
                    self.done_rids.pop(next(iter(self.done_rids)))

    def _loop(self) -> None:
        while not self._stop:
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            job = self._take()
            if job is not None:
                self.run_job(job)
                if self.waiting is not None:
                    self._wake.set()

    def stop(self) -> None:
        self._stop = True
        self._wake.set()


def on_hint(scheduler, recv_req) -> Optional[str]:
    """``Scheduler.handle_ple_prefetch_hint``: a ``read_ahead`` hint. PP0 queues it; every other rank ignores it.
    Never raises (it runs in the scheduler's request dispatch)."""
    try:
        if not armed(scheduler):
            return None
        ids = list(getattr(recv_req, "input_ids", None) or [])
        span = int(getattr(recv_req, "start_hint", -1) or -1)
        if span < MIN_SPAN_TOKENS or not ids:
            return None
        ra = getattr(scheduler, ATTR, None)
        if ra is None:
            ra = ReadAhead(scheduler)
            setattr(scheduler, ATTR, ra)
        return ra.submit(str(recv_req.rid), ids, span)
    except Exception as exc:  # noqa: BLE001 -- a lost hint costs the gain only
        logger.warning("PDFLIP L2-READ-AHEAD hint dropped: %r", exc)
        return None
