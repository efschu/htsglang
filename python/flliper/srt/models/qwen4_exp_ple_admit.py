"""fnFL2 H43: the PLE gather of a request's FIRST chunk, started at its admission.

THE COST. H32 (``qwen4_exp_ple_prefetch``) reads the rows of chunk n+1 while
chunk n computes, so from chunk 1 on the PLE gather is off the critical path.
Chunk 0 is not: nothing ran before it. x146 (24.09., PP0 = 5090): the 97k
needle ``chunk=0 rows=262144 ready=none wait_ms=528.5`` (ARC-warm), the cold
12.6k code prompt ``chunk=0 rows=202688 ready=none wait_ms=4473.1`` -- 4.5 s of
a 17.2 s TTFT, with the GPU idle.

THE FIX. A request's token ids are complete when it enters the scheduler
(``origin_input_ids``), long before its first forward when it waits behind a
running request (the front keeps ``p_concurrency + P_QUEUE_AHEAD`` requests on
P, #1459) -- and, through the front's hint (:func:`admit_ple_hint`, route
``/pdflip/ple_prefetch_hint``), before P has even woken up for it. So:

* the PP0 scheduler hands every new request to :func:`admit_ple_request` on
  intake (``Scheduler._add_request_to_queue``, before the dormant hold, so a
  held request starts as well);
* :class:`PleAdmitPrefetchGather` hashes the request's first chunk with the
  layer's own hash (the H32 host mirror) and reads its rows on the SAME worker
  processes into a THIRD slot (:data:`PLE_ADMIT_SLOT`), but only while the
  H32 ring is idle -- the running request's next chunk always goes first; an
  admission that finds the ring busy waits in a short FIFO and starts at the
  end of the running request's last chunk gather (the ring's first idle
  moment) or at a later intake;
* when the request's first chunk runs, the admission is handed to H32's join
  as that chunk's prefetch (``_pending``): the same row-by-row comparison with
  the prediction, the same on-the-spot read of every row that differs, the
  same bytes. The chunk=0 line then reads ``ready=yes`` when the read had
  finished before the forward.

Correctness does not rest on the admission: it is used only when the request
is the FIRST request of the extend batch and its chunk starts where the
admission was made (#1416h: token 0, the registered head, or PP0's store
told -- ``pdflip_store_told``); everything else drops it (``start_moved``) and
the chunk is gathered as before. H32's join compares every row anyway.
An admission never reads while the H32 ring has a read in flight, and an
admission read in flight is joined before H32 submits anything
(``PlePreadProcs`` carries one read at a time).

THE HINT'S WINDOW (1001). The front's hint does not know where P's first
chunk will start (PP0's told does, ~150 ms before the forward), so it reads
the prompt's last ``chunk_size`` tokens (:func:`ple_hint_start`); a later
exact admission (told, intake) whose chunk lies inside that window RE-KEYS it
(:meth:`PleAdmitPrefetchGather._rekey`: the chunk's rows are used at their
offset in the slot, ``_Pending.off``) instead of dropping it, and exact
admissions wait ahead of hints.

``FLLIPER_QWEN4_PLE_PREFETCH_ADMIT=0`` keeps H32 exactly as it was (two slots,
no admission). Off on group D (its leg-2 requests arrive with the prefix
cached; a first-chunk read from token 0 would be read for nothing).
Host RAM: one more memfd slot of one chunk's rows (16384 x 16 x 320 B =
80 MiB, page-locked), allocated at the first admission; no further worker.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import List, Optional, Sequence

import torch

from flliper.srt.models import qwen4_exp_ple_prefetch as _pf

logger = logging.getLogger(__name__)

PLE_ADMIT_SLOT = _pf.PLE_PREFETCH_DEPTH  # the third slot, behind the H32 ring
PLE_ADMIT_QUEUE_MAX = 4
PLE_ADMIT_TTL_S = 300.0

#: the admission clock (tests replace it with a fake one)
_clock = time.monotonic

_SINKS: List["PleAdmitPrefetchGather"] = []
#: (rid, extend start) of the extend batch about to run, in batch order
_BATCH: tuple = ()


class _Admission:
    __slots__ = (
        "rid", "tokens", "t_admit", "source", "dormant", "ids", "vocab",
        "seq", "t_submit", "read_s", "joined", "orphan", "start", "lead", "row_off",
    )

    def __init__(self, rid: str, tokens: torch.Tensor, t_admit: float, source: str, dormant: bool,
                 start: int = 0, lead: int = 0):
        self.rid = rid
        self.tokens = tokens
        self.start = int(start)
        self.lead = int(lead)
        self.t_admit = t_admit
        self.source = source
        self.dormant = bool(dormant)
        self.ids: Optional[torch.Tensor] = None
        self.vocab: Optional[tuple] = None
        self.seq: Optional[int] = None
        self.t_submit = 0.0
        self.read_s = 0.0
        self.joined = False
        self.orphan = False
        #: the first row of ``ids`` in the admission slot (a re-keyed window)
        self.row_off = 0


# --------------------------------------------------------------------------
# Scheduler -> admission (all no-ops unless an admitting gather lives here).
# --------------------------------------------------------------------------


def ple_admission_armed() -> bool:
    return bool(_SINKS)


def note_ple_batch(reqs: Sequence) -> None:
    """Scheduler, before an extend batch's forward (beside H32's
    ``publish_ple_next_chunk``): the batch's request order and extend starts."""
    global _BATCH
    if not _SINKS:
        return
    batch = []
    for req in reqs:
        rng = getattr(req, "extend_range", None)
        batch.append((str(getattr(req, "rid", "")), int(rng.start) if rng is not None else -1))
    _BATCH = tuple(batch)
    # no pump here: a read started now would be joined by this very batch's
    # gather (the ring idles only after a gather, whose end pumps)
    for sink in list(_SINKS):
        sink.forget_served(_BATCH)


#: #1416h: intake verdicts that mean "a store read is registered (or held
#: behind a twin) and PP0's told will name the first chunk's start".
_STORE_READ_VERDICTS = ("issued", "declined:pdflip_twin_deferred")


def ple_first_chunk_start(req) -> int:
    """Where ``req``'s first extend chunk starts when nothing more than its
    registration is known: the device/host head its prefetch registration
    matched (``_prefetch_registered_prefix_len``), else 0."""
    try:
        return max(0, int(getattr(req, "_prefetch_registered_prefix_len", 0) or 0))
    except (TypeError, ValueError):
        return 0


def admit_ple_request(
    req, chunk_size: Optional[int], *, dormant: bool = False, start: Optional[int] = None,
    source: str = "queue",
) -> Optional[str]:
    """Scheduler intake (``start`` None) or PP0's store told (``start`` =
    the told offset, #1416h): start (or queue) the PLE gather of the chunk
    that begins at ``start``. Returns the verdict (``started``/``queued``/
    ``confirmed``/``skipped:<why>``), None when nothing in this process admits.

    #1416h: at intake a request whose store read is still running is NOT
    admitted from token 0 -- its first chunk starts where the read ends,
    which only PP0's told knows (z30e: 31 of 33 token-0 admissions dropped
    as ``cached_prefix``, 262144 rows read each for nothing); the told hook
    admits it at the told offset. Without a store read the first chunk
    starts at the registered head."""
    if not _SINKS:
        return None
    ids = getattr(req, "origin_input_ids", None)
    if not ids:
        return None
    rid = str(getattr(req, "rid", ""))
    if start is None:
        if getattr(req, "_969c_verdict", None) in _STORE_READ_VERDICTS:
            return "skipped:told_pending"
        start = ple_first_chunk_start(req)
    verdict = None
    for sink in list(_SINKS):
        verdict = sink.admit(rid, ids, chunk_size, dormant=dormant, source=source, start=int(start))
    return verdict


def admit_ple_hint(rid: str, input_ids, chunk_size: Optional[int], *, dormant: bool = False) -> Optional[str]:
    """The front's hint (``PlePrefetchHintReqInput``): the same as an intake,
    for a request that has not reached this scheduler yet -- but reading the
    prompt's TAIL window (:func:`ple_hint_start`), not its first chunk."""
    if not _SINKS or not input_ids:
        return None
    start = ple_hint_start(len(input_ids), chunk_size)
    verdict = None
    for sink in list(_SINKS):
        verdict = sink.admit(str(rid), input_ids, chunk_size, dormant=dormant, source="hint", start=start)
    return verdict


def ple_hint_start(n: int, chunk_size: Optional[int]) -> int:
    """Where the hint's read window starts: the last ``chunk_size`` tokens.

    1001 (NF bfpgwv ...dauer10011823 and three later boots): 0 of 69 hinted
    requests ever used their hint. Every one was read from token 0, and P's first
    chunk after a D->P flip starts behind the store prefix PP0's told names
    (``told ... start=84544`` for a 94234-token prompt) -- the hint was then
    dropped (logged as ``tokens_differ``), its 262144 rows read for nothing,
    and the told's own read started ~150 ms before the forward
    (``ready=no``). The hint cannot know the told; the store prefix of an
    agent turn reaches into the last chunk (52 of 65 hinted requests: the
    rest behind the told was shorter than one chunk), so the tail window
    holds the real first chunk and the told RE-KEYS the read
    (:meth:`PleAdmitPrefetchGather._rekey`). A prompt of at most one chunk
    reads it whole, exactly as before."""
    n = int(n)
    size = int(chunk_size) if chunk_size and int(chunk_size) > 0 else n
    return max(0, n - size)


def drop_ple_admission(rid: Optional[str], *, abort_all: bool = False, reason: str = "abort") -> int:
    """Scheduler abort: forget the admissions of ``rid`` (prefix match, like
    the abort itself) or of every request. Returns how many were dropped."""
    if not _SINKS:
        return 0
    return sum(sink.drop(rid, abort_all=abort_all, reason=reason) for sink in list(_SINKS))


# --------------------------------------------------------------------------
# The gather that takes admissions.
# --------------------------------------------------------------------------


class PleAdmitPrefetchGather(_pf.PlePrefetchGather):
    """:class:`PlePrefetchGather` plus the first-chunk admission (module
    docstring). The H32 ring and its join are the parent's, untouched."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._slots.append(_pf._Slot())
        self._adm: Optional[_Admission] = None  # the admission slot's holder
        self._adm_queue: "collections.OrderedDict[str, _Admission]" = collections.OrderedDict()
        self._last_vocab: Optional[tuple] = None
        self._hash_warm = False
        self._lock = threading.RLock()
        self.stats.update({"admits": 0, "admit_started": 0, "admit_used": 0,
                           "admit_dropped": 0, "admit_settle_s": 0.0, "admit_rekeyed": 0})
        _SINKS.append(self)

    # -- parent hooks ----------------------------------------------------------
    def _ensure_workers(self) -> _pf.PlePreadProcs:
        if self._workers is None:
            self._workers = _pf.PlePreadProcs(
                self._files,
                self._table.row_bytes,
                depth=PLE_ADMIT_SLOT + 1,
                procs=self._n_procs,
                threads=self._threads,
                delay_s=self._delay_s,
            )
            logger.info(
                "PLE-PREFETCH on: %d pread worker processes x %d threads (pids %s), "
                "%d shared slots (ring %d + admission 1), next chunk read during the "
                "current forward, first chunk read from the request's admission",
                self._workers.n_procs, self._threads, self._workers.pids(),
                PLE_ADMIT_SLOT + 1, _pf.PLE_PREFETCH_DEPTH,
            )
        return self._workers

    def _disable(self) -> None:
        with self._lock:
            self._adm = None
            self._adm_queue.clear()
            if self in _SINKS:
                _SINKS.remove(self)
            super()._disable()

    def _gather_into(self, flat_ids, out, vocab_start, vocab_end):
        global _BATCH
        with self._lock:
            end = self._table.total_rows if vocab_end is None else vocab_end
            self._last_vocab = (int(vocab_start), int(end))
            self._warm_hash()
            w = self._ensure_workers()
            self._before_gather(w)
            try:
                out = super()._gather_into(flat_ids, out, vocab_start, vocab_end)
            finally:
                _BATCH = ()
            # the ring's first idle moment (the running request's last chunk
            # queued no next chunk): the oldest waiting admission starts
            self._pump(w)
            return out

    # -- boot prewarm (P-PREWARM, 30.09.) ---------------------------------------
    def prewarm(self, *, vocab_start: int, vocab_end: Optional[int]) -> str:
        """At boot, with the weights on the device: exactly what the first
        prefill gather sets before its read -- the vocab range (the layer's
        own ``shard_indices``, the same pair every gather passes), the hash
        constants on the host (``_warm_hash``: copies of the same buffers) --
        plus the worker processes. From then on the first request's admission
        reads its first chunk instead of logging ``skipped: no prefill gather
        in this process yet``. No slot is mapped here: the admission maps its
        own at its first read, as before. Returns ``warm`` or why not."""
        with self._lock:
            if self._disabled:
                return "disabled"
            end = self._table.total_rows if vocab_end is None else vocab_end
            self._last_vocab = (int(vocab_start), int(end))
            self._warm_hash()
            try:
                self._ensure_workers()
            except (_pf.PleWorkerLost, OSError) as exc:
                # what the first gather would do on the same failure
                logger.error("PLE-PREFETCH disabled: %s -- this process gathers serially from now on", exc)
                self._disable()
                return "disabled"
            return "warm"

    # -- admission ---------------------------------------------------------------
    def admit(self, rid: str, ids, chunk_size: Optional[int], *, dormant: bool, source: str,
              start: int = 0) -> Optional[str]:
        with self._lock:
            if self._disabled:
                return None
            try:
                return self._admit(rid, ids, chunk_size, dormant, source, int(start))
            except (_pf.PleWorkerLost, OSError) as exc:
                logger.error("PLE-PREFETCH disabled at admission rid=%s: %s -- serial gather from now on", rid, exc)
                self._disable()
                return None

    def drop(self, rid: Optional[str], *, abort_all: bool, reason: str) -> int:
        with self._lock:
            def hit(r: str) -> bool:
                return abort_all or (bool(rid) and r.startswith(str(rid)))

            n = 0
            for r in [r for r in self._adm_queue if hit(r)]:
                self._adm_queue.pop(r)
                self._log_drop(r, reason)
                n += 1
            adm = self._adm
            if adm is not None and not adm.orphan and hit(adm.rid):
                self._log_drop(adm.rid, reason)
                n += 1
                if adm.seq is not None and not adm.joined:
                    adm.orphan = True  # joined at the next gather or pump, never waited on here
                else:
                    self._adm = None
            return n

    def forget_served(self, batch: tuple) -> None:
        """Waiting admissions of requests this batch serves are late: gone."""
        with self._lock:
            rids = {r for r, _ in batch}
            for rid in [r for r in self._adm_queue if r in rids]:
                self._adm_queue.pop(rid)
                self._log_drop(rid, "served_before_its_read_started")

    # -- internals ---------------------------------------------------------------
    def _hash_ready(self) -> bool:
        ready = getattr(self._hasher, "ple_hash_ready", None)
        return True if ready is None else bool(ready())

    def _warm_hash(self) -> None:
        # inside a forward (the model is awake): copy the hash constants to
        # the host now, so a later admission never reads the device -- which
        # may be asleep by then (weights paused during a flip)
        if not self._hash_warm:
            warm = getattr(self._hasher, "ple_hash_warm", None)
            if warm is not None:
                warm()
            self._hash_warm = True

    def _admit(self, rid, ids, chunk_size, dormant, source, start: int = 0) -> str:
        n = len(ids)
        size = int(chunk_size) if chunk_size and int(chunk_size) > 0 else n
        start = max(0, min(int(start), n))
        if start >= n:
            return "skipped:nothing_to_prefill"
        # #1416h: a chunk that starts mid-prompt hashes its n-grams over the
        # tokens before it -- the lead H32 uses for its next chunk
        lead = min(start, _pf.PLE_PREFETCH_LEAD_TOKENS)
        tokens = _pf._as_int64(ids[start - lead: min(n, start + size)])
        self.stats["admits"] += 1
        known = self._adm if (self._adm is not None and not self._adm.orphan and self._adm.rid == rid) else None
        if known is None:
            known = self._adm_queue.get(rid)
        if known is not None:
            if known.start == start and torch.equal(known.tokens, tokens):
                logger.info("PLE-PREFETCH admit rid=%s confirmed source=%s (admitted by %s)", rid, source, known.source)
                return "confirmed"
            if self._rekey(known, start, lead, tokens, source):
                return "rekeyed"
            # 1001: a different start was logged as tokens_differ -- the
            # hint's token-0 window against the told's offset, every time
            self.drop(rid, abort_all=False, reason="tokens_differ" if known.start == start else "start_moved")
        if self._last_vocab is None or not self._hash_ready():
            logger.info("PLE-PREFETCH admit rid=%s skipped: no prefill gather in this process yet "
                        "(hash constants and vocab range unknown)", rid)
            return "skipped:cold"
        w = self._ensure_workers()
        self._pump(w)  # stale admissions leave first
        if len(self._adm_queue) >= PLE_ADMIT_QUEUE_MAX and source != "hint":
            # an exact admission (its forward is next) displaces the newest
            # waiting hint (a request that is still to come)
            hints = [r for r, a in self._adm_queue.items() if a.source == "hint"]
            if hints:
                self._adm_queue.pop(hints[-1])
                self._log_drop(hints[-1], "displaced_by_exact")
        if len(self._adm_queue) >= PLE_ADMIT_QUEUE_MAX:
            logger.info("PLE-PREFETCH admit rid=%s skipped: %d admissions already waiting", rid, len(self._adm_queue))
            return "skipped:queue_full"
        adm = _Admission(rid, tokens, _clock(), source, dormant, start=start, lead=lead)
        self._adm_queue[rid] = adm
        self._exact_first()
        self._pump(w)
        if self._adm is adm:
            return "started"
        if rid in self._adm_queue:
            logger.info("PLE-PREFETCH admit rid=%s queued source=%s dormant=%d tokens=%d start=%d "
                        "(ring busy: the running request's read goes first)",
                        rid, source, int(adm.dormant), int(tokens.numel()) - lead, start)
            return "queued"
        return "skipped:small"

    def _rekey(self, known: _Admission, start: int, lead: int, tokens: torch.Tensor, source: str) -> bool:
        """1001: the chunk an exact admission names (told / intake) lies inside
        the window ``known`` already read (or is reading) -- the hint's tail
        window. Its rows stay where they are: ``ids`` becomes the chunk's
        slice of the window and ``row_off`` its first row in the slot, so the
        forward's join compares and copies exactly the chunk's rows. Only for
        a read that was started; the tokens of the chunk (with its lead, so
        every n-gram context) must equal the window's. No re-read, no copy."""
        if known.ids is None or known.seq is None or known.orphan:
            return False
        base = known.start - known.lead  # absolute index of known.tokens[0]
        s0 = start - lead
        end = s0 + int(tokens.numel())
        known_end = base + int(known.tokens.numel())
        if start < known.start or s0 < base or end > known_end or end <= start:
            return False
        if not torch.equal(known.tokens[s0 - base: end - base], tokens):
            return False
        span = known_end - known.start
        rows = int(known.ids.numel())
        if span <= 0 or rows % span:
            return False
        per = rows // span
        off, cnt = (start - known.start) * per, (end - start) * per
        if cnt < self.min_rows:
            return False  # what _start skips as small
        old = known.start
        known.ids = known.ids[off: off + cnt]
        known.row_off += off
        known.start, known.lead, known.tokens = start, lead, tokens
        self.stats["admit_rekeyed"] += 1
        logger.info(
            "PLE-PREFETCH admit rid=%s rekeyed source=%s (admitted by %s) start=%d->%d rows=%d row_off=%d "
            "ready=%s: the chunk lies inside the read window, its rows are used where they are",
            known.rid, source, known.source, old, start, cnt, known.row_off,
            "yes" if known.joined or (self._workers is not None and self._workers.ready(known.seq)) else "no",
        )
        return True

    def _exact_first(self) -> None:
        """Waiting admissions in reading order: exact ones (intake, told --
        their forward is next) before hints (requests still to come), each
        group in arrival order. bfpgwv 19:12:21: pdflip-42-307's told waited
        behind pdflip-42-308's hint, was served before its read started and
        gathered cold (ready=none wait_ms=817.3)."""
        q = self._adm_queue
        if any(a.source == "hint" for a in q.values()):
            items = list(q.items())
            q.clear()
            q.update([kv for kv in items if kv[1].source != "hint"])
            q.update([kv for kv in items if kv[1].source == "hint"])

    def _pump(self, w) -> None:
        """Start the oldest waiting admission if the admission slot is free
        and nothing is being read (never waits)."""
        now = _clock()
        for rid in [r for r, a in self._adm_queue.items() if now - a.t_admit > PLE_ADMIT_TTL_S]:
            self._adm_queue.pop(rid)
            self._log_drop(rid, "ttl")
        adm = self._adm
        if adm is not None:
            if adm.orphan or now - adm.t_admit > PLE_ADMIT_TTL_S:
                if adm.seq is not None and not adm.joined and not w.ready(adm.seq):
                    return
                if adm.seq is not None and not adm.joined:
                    w.join(adm.seq)
                if not adm.orphan:
                    self._log_drop(adm.rid, "ttl")
                self._adm = None
            else:
                return
        if getattr(w, "_inflight", None) is not None:
            return  # the H32 ring reads; it goes first
        while self._adm_queue:
            _, adm = self._adm_queue.popitem(last=False)
            if self._start(w, adm):
                return

    def _start(self, w, adm: _Admission) -> bool:
        vocab = self._last_vocab
        t0 = time.monotonic()
        nids = self._hasher(adm.tokens, adm.lead).to(torch.int64).reshape(-1)
        if int(nids.numel()) < self.min_rows:
            logger.info("PLE-PREFETCH admit rid=%s skipped: rows=%d below the gather's %d",
                        adm.rid, int(nids.numel()), self.min_rows)
            return False
        slot = self._slots[PLE_ADMIT_SLOT]
        slot.wait_copy()
        self._ensure_rows(PLE_ADMIT_SLOT, int(nids.numel()))
        n_in = (nids >= vocab[0]) & (nids < vocab[1]) & (nids < self._table.total_rows)
        keys = _pf.ple_row_keys(nids, n_in, self._table, self._file_index)
        adm.seq = w.submit(PLE_ADMIT_SLOT, torch.arange(nids.numel()), keys)
        adm.ids, adm.vocab, adm.t_submit = nids, vocab, _clock()
        self._adm = adm
        self.stats["admit_started"] += 1
        logger.info(
            "PLE-PREFETCH admit rid=%s rows=%d started source=%s dormant=%d start=%d waited_ms=%.1f host_ms=%.1f",
            adm.rid, int(nids.numel()), adm.source, int(adm.dormant), adm.start,
            (adm.t_submit - adm.t_admit) * 1000.0, (time.monotonic() - t0) * 1000.0,
        )
        return True

    def _settle(self, w, adm: _Admission) -> float:
        """Join an admission read that is in flight (H32 submits next)."""
        if adm.seq is not None and not adm.joined:
            t = time.monotonic()
            adm.read_s = w.join(adm.seq)
            adm.joined = True
            dt = time.monotonic() - t
            self.stats["admit_settle_s"] += dt
            return dt
        return 0.0

    def _before_gather(self, w) -> None:
        """Before H32 serves a chunk: hand this batch's admission to its join,
        drop what this batch makes stale, and leave nothing of an admission in
        flight that H32 does not join itself."""
        batch = _BATCH
        rids = {r for r, _ in batch}
        self.forget_served(batch)
        adm = self._adm
        if adm is None:
            return
        if adm.orphan:
            self._settle(w, adm)
            self._adm = None
            return
        first = batch[0] if batch else None
        if first is not None and adm.rid == first[0]:
            if first[1] == adm.start and self._pending is None and adm.vocab == self._last_vocab:
                self._adm = None
                ready = adm.joined or w.ready(adm.seq)
                self._pending = _pf._Pending(adm.seq, PLE_ADMIT_SLOT, adm.ids, adm.vocab, adm.row_off)
                self.stats["admit_used"] += 1
                now = _clock()
                logger.info(
                    "PLE-PREFETCH admit rid=%s rows=%d queued_ms_before_forward=%.1f "
                    "read_started_ms_before_forward=%.1f ready=%s source=%s dormant=%d row_off=%d",
                    adm.rid, int(adm.ids.numel()), (now - adm.t_admit) * 1000.0,
                    (now - adm.t_submit) * 1000.0, "yes" if ready else "no",
                    adm.source, int(adm.dormant), adm.row_off,
                )
                return
            reason = ("cached_prefix" if first[1] > adm.start else "start_moved") if first[1] != adm.start else (
                "ring_holds_a_prediction" if self._pending is not None else "vocab")
        elif adm.rid in rids:
            reason = "not_first_in_batch"
        else:
            dt = self._settle(w, adm)  # another request's chunk: keep the rows
            if dt > 0.0005:
                logger.info("PLE-PREFETCH admit rid=%s settled before another request's gather: wait_ms=%.1f",
                            adm.rid, dt * 1000.0)
            return
        self._settle(w, adm)
        self._adm = None
        self._log_drop(adm.rid, reason)

    def _log_drop(self, rid: str, reason: str) -> None:
        self.stats["admit_dropped"] += 1
        logger.info("PLE-PREFETCH admit rid=%s dropped reason=%s", rid, reason)


def ple_admission_wanted() -> bool:
    """``FLLIPER_QWEN4_PLE_PREFETCH_ADMIT`` and not group D (module docstring)."""
    from flliper.srt.environ import envs

    if not envs.FLLIPER_QWEN4_PLE_PREFETCH_ADMIT.get():
        return False
    try:
        from flliper.srt.managers.pdflip_memory_saver import pdflip_group_name

        group = pdflip_group_name()
    except Exception:  # noqa: BLE001 -- no Weg-2 machinery: a stock engine admits
        group = ""
    return str(group).strip().upper() != "D"


# --------------------------------------------------------------------------
# P-PREWARM (30.09.): the admission armed at boot.
# --------------------------------------------------------------------------
#
# Measured reason (y4k 09301110 / y4l 09301150, P logs, tree 3264fb08b5's
# parent line): the first request of every boot is admitted before any
# prefill gather ran in the process -- 'PLE-PREFETCH admit rid=pdflip-0-5
# skipped: no prefill gather in this process yet' (11:14:23 and 11:14:27) --
# so its chunk 0 is read inside its own forward: 'chunk=0 rows=262144
# ready=none wait_ms=195.1 ... host_ms=228.9', the worker processes spawned in
# that same gather ('PLE-PREFETCH on' 11:14:27), ple_ms=334.2 against a warm
# median of 39. Every later first chunk reads 'ready=yes ... host_ms=3-38'.
# The vocab range and the hash constants do not depend on any request; the
# boot sets them, the first admission reads.


class PleAdmitPrewarmResult(collections.namedtuple("PleAdmitPrewarmResult", "gathers verdicts ms skipped")):
    __slots__ = ()

    def line(self) -> str:
        if self.skipped:
            return f"P-PREWARM PLE-ADMIT skipped: {self.skipped}"
        return (
            f"P-PREWARM PLE-ADMIT gathers={self.gathers} verdicts={list(self.verdicts)} "
            f"ms={self.ms:.1f} -- vocab range and hash constants set, workers up: the "
            f"first request's admission reads its first chunk"
        )


def admitting_gathers(modules) -> list:
    """(gather, vocab_start, vocab_end) of every admitting PLE gather in
    ``modules`` -- the ``Qwen4ExpPinnedHostEmbedding`` that owns it carries
    the pair its gathers pass (``shard_indices.org_vocab_*_index``)."""
    out = []
    for m in modules:
        g = getattr(m, "_ckpt_pread", None)
        si = getattr(m, "shard_indices", None)
        if isinstance(g, PleAdmitPrefetchGather) and si is not None:
            if all(g is not x[0] for x in out):
                out.append((g, int(si.org_vocab_start_index), int(si.org_vocab_end_index)))
    return out


def run_boot_prewarm(*, model, clock=time.perf_counter) -> Optional[PleAdmitPrewarmResult]:
    """The scheduler's delegate (``FLLIPER_PDFLIP_ENABLE_TARGETED_PREWARM``).
    Skips, named, when the switch is off or the rank holds no admitting PLE
    gather (group D, PP1/PP2, admission off). Never raises."""
    try:
        from flliper.srt.environ import envs

        if not envs.FLLIPER_PDFLIP_ENABLE_TARGETED_PREWARM.get():
            res = PleAdmitPrewarmResult(0, (), 0.0, "FLLIPER_PDFLIP_ENABLE_TARGETED_PREWARM off")
            logger.info("%s", res.line())
            return res
        found = admitting_gathers(model.modules()) if model is not None else []
        if not found:
            res = PleAdmitPrewarmResult(0, (), 0.0, "no admitting PLE gather on this rank")
            logger.info("%s", res.line())
            return res
        t0 = clock()
        verdicts = tuple(g.prewarm(vocab_start=vs, vocab_end=ve) for g, vs, ve in found)
        res = PleAdmitPrewarmResult(len(found), verdicts, (clock() - t0) * 1000.0, "")
        logger.info("%s", res.line())
        return res
    except Exception as exc:  # noqa: BLE001 -- a prewarm never kills a boot
        logger.warning("P-PREWARM PLE-ADMIT failed (%s: %s); the first prefill gather "
                       "arms the admission", type(exc).__name__, str(exc)[:200])
        return None
