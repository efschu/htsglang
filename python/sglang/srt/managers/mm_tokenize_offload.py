"""Multimodal tokenization off the tokenizer event loop (D-HEALTH, 10.10.).

THE FINDING. The tokenizer process serves ``/health`` and ``/metrics`` on the
same asyncio loop that tokenizes requests. For an image request the loop ran
the whole CPU chain synchronously: the HF image processor
(``process_and_combine_mm_data``), the sha256 over ``pixel_values``
(``set_pad_value``), the shared-memory copy (``wrap_shm_features``). On the
dual 4096x4096 image (16384 image tokens) D's loop stood for ~41 s (boot
``dkr27bnvfp4dualvwweightsbar1fs10100032``: D-ADMIT 00:39:13.96, scheduler
intake 00:39:55), two ``/health`` probes timed out and the front stopped the
healthy group with ``W17 Weg2GroupDead``.

WHY A THREAD AND NOT A PROCESS (measured, CPU only, 4096x4096 JPEG, the 27B
preprocessor config, ``/root/.claude/jobs/1ab4cd30/tmp/gil_probe.py``): with
each step in a worker thread the loop's worst wake-up lag was 6.2 ms for the
PIL decode, 5.5 ms for the HF processor (326 ms of work), 0.4 ms for the
sha256 (177 ms), 0.4 ms for the shm fallocate+copy (95 ms) -- every one of
these releases the GIL. Only the INLINE transport (``pickle.dumps`` /
msgpack of the 400 MB ``pixel_values``) holds it (222 / 209 ms lag), and that
transport is not the one a single-node boot uses (``dist_init_addr=None`` ->
``cuda_ipc`` mode -> shm pointers, whose pickle is 0.3 ms).

THE RULES THIS MODULE KEEPS.

* Only for a single (non-batch) request with multimodal input, only when no
  vision tower service runs in this process
  (``vision_stage_service.installed() is None``; the tower runs inside
  ``process_and_combine_mm_data`` and stays where it was), and never with a
  device-resident frontend (CUDA IPC transport, ``--keep-mm-feature-on-device``,
  GPU preprocessing) -- no CUDA work moves to another thread.
* ONE worker thread: requests are processed one at a time, as on the loop
  before, so N large images never hold N feature tensors at once.
* The offloaded call runs on a PRIVATE copy of the HF processor
  (``BaseMultimodalProcessor.offload_twin``): the loop keeps tokenizing text
  with the shared Rust tokenizer, and two threads on one fast tokenizer raise
  ``RuntimeError: Already borrowed`` (reproduced, tokenizers 0.22.2).
* The ContextVars of the request (the transient vision stage's rid) are
  copied into the worker.
* DISPATCH ORDER IS KEPT. Before, nothing could reach the scheduler while an
  image request was being processed, and the image request went first. Now a
  request that wants to dispatch while an earlier image request is still open
  waits for that request's dispatch -- the same order, without the loop
  standing. The zmq send itself stays on the loop thread (zmq sockets are not
  thread-safe). The one exemption is the synthetic ``/health`` generate
  request (``HEALTH_CHECK_RID_PREFIX``): it carries ``input_ids=[0]``, is never
  handed off, and holding it behind the image is exactly the stop this fixes.
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import contextlib
import contextvars
import functools
import logging
import threading
import time
from typing import Any, AsyncIterator, Callable, Deque, Optional, Tuple

import msgspec

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.weg2 import host_mem_probe as _hmp

logger = logging.getLogger(__name__)

# True inside the tokenize section of a request whose multimodal CPU work may
# run on the worker thread. Read by the multimodal processor.
_OFFLOAD: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "mm_tokenize_offload", default=False
)

_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None
_EXECUTOR_LOCK = threading.Lock()

#: (I) a multimodal request slower than this prints its WEG2-MM-TOKENIZE line at INFO.
MM_TOKENIZE_REPORT_S = 1.0
#: (I) TOKENIZER-LOOP-LAG: the sampler's period and the lag that is reported.
LOOP_LAG_PERIOD_S = 0.5
LOOP_LAG_REPORT_S = 2.0


class MmTokenizeStats(msgspec.Struct):
    """(I) Phase times of one multimodal request in the tokenizer, for the WEG2-MM-TOKENIZE line.

    Shared by reference with the worker thread (the ContextVar copy carries the
    same object), so the phases noted there arrive here. -1 = phase not run.
    """

    rid: str
    offloaded: bool
    t0: float
    ru0: Tuple[float, float, int, int, int]
    vm0: dict
    img_tokens: int = -1
    pixels: int = -1
    load_ms: float = -1.0
    queue_ms: float = -1.0
    process_ms: float = -1.0
    hash_ms: float = -1.0
    mrope_ms: float = -1.0
    shm_wrap_ms: float = -1.0
    send_ms: float = -1.0

    @classmethod
    def start(cls, *, rid: Any, offloaded: bool) -> "MmTokenizeStats":
        return cls(rid=str(rid), offloaded=offloaded, t0=time.perf_counter(),
                   ru0=_hmp.rusage_self(), vm0=_hmp.read_vmstat())

    def report(self) -> None:
        total_s = time.perf_counter() - self.t0
        if total_s <= MM_TOKENIZE_REPORT_S:
            return
        ru1 = _hmp.rusage_self()
        vm = _hmp.vmstat_delta(self.vm0, _hmp.read_vmstat())
        logger.info(
            "WEG2-MM-TOKENIZE rid=%s img_tokens=%d pixels=%d load_ms=%.0f queue_ms=%.0f process_ms=%.0f "
            "hash_ms=%.0f mrope_ms=%.0f shm_wrap_ms=%.0f send_ms=%.0f total_ms=%.0f utime_ms=%.0f "
            "stime_ms=%.0f minflt=%d majflt=%d nivcsw=%d compact_stall=%d pswpin=%d pswpout=%d "
            "pgmajfault=%d offloaded=%d (process-wide rusage/vmstat deltas over the request)",
            self.rid, self.img_tokens, self.pixels, self.load_ms, self.queue_ms, self.process_ms,
            self.hash_ms, self.mrope_ms, self.shm_wrap_ms, self.send_ms, total_s * 1000,
            (ru1[0] - self.ru0[0]) * 1000, (ru1[1] - self.ru0[1]) * 1000, ru1[2] - self.ru0[2],
            ru1[3] - self.ru0[3], ru1[4] - self.ru0[4], vm["compact_stall"], vm["pswpin"],
            vm["pswpout"], vm["pgmajfault"], int(self.offloaded),
        )


_STATS: contextvars.ContextVar[Optional[MmTokenizeStats]] = contextvars.ContextVar(
    "mm_tokenize_stats", default=None
)


def note(**phases: Any) -> None:
    """(I) Record phase values on this request's stats; a no-op outside a multimodal request."""
    stats = _STATS.get()
    if stats is None:
        return
    for name, value in phases.items():
        setattr(stats, name, value)


def requested() -> bool:
    """May the multimodal processor run its CPU chain on the worker thread?"""
    return _OFFLOAD.get()


def offload_permitted() -> bool:
    """No tower service in this process and no device-resident frontend."""
    from sglang.srt.multimodal.processors.base_processor import (
        mm_frontend_gpu_enabled,
    )

    if mm_frontend_gpu_enabled():
        return False
    try:
        from sglang.srt.weg2 import vision_stage_service as _vss
    except ImportError:
        return True
    return _vss.installed() is None


def _executor() -> concurrent.futures.ThreadPoolExecutor:
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mm_tokenize"
            )
        return _EXECUTOR


async def run_offloaded(fn: Callable[..., Any], /, *args, **kwargs) -> Any:
    """Run ``fn`` on the single worker thread with a copy of this context."""
    ctx = contextvars.copy_context()
    t_submit = time.perf_counter()

    def call():
        stats = _STATS.get()
        if stats is not None:  # (I) summed over the request's offloaded steps
            stats.queue_ms = max(stats.queue_ms, 0.0) + (time.perf_counter() - t_submit) * 1000
        return fn(*args, **kwargs)

    return await asyncio.get_running_loop().run_in_executor(
        _executor(), functools.partial(ctx.run, call)
    )


async def loop_lag_sampler(
    *, period_s: float = LOOP_LAG_PERIOD_S, report_s: float = LOOP_LAG_REPORT_S
) -> None:
    """(I) TOKENIZER-LOOP-LAG: report every wake-up of this task later than ``report_s``.

    The loop that serves /health and /metrics; a late wake-up is the time it
    answered nothing. Log only, never raises.
    """
    loop = asyncio.get_running_loop()
    while True:
        t0 = loop.time()
        await asyncio.sleep(period_s)
        try:
            lag = loop.time() - t0 - period_s
            if lag > report_s:
                began = time.time() - lag - period_s
                logger.info(
                    "TOKENIZER-LOOP-LAG max_ms=%.0f at=%s.%03d (the tokenizer loop, which serves "
                    "/health and /metrics, answered nothing for this long)",
                    lag * 1000, time.strftime("%H:%M:%S", time.localtime(began)),
                    int((began % 1) * 1000),
                )
        except Exception:  # noqa: BLE001 -- an instrument never stops the loop
            pass


class _Turn:
    """One request's place in the dispatch order (see ``MmDispatchOrder``)."""

    def __init__(
        self,
        *,
        order: "MmDispatchOrder",
        own: Optional[asyncio.Event],
        before: Optional[Tuple[asyncio.Event, ...]],
    ):
        self._order = order
        self._own = own
        # None: wait for whatever is open at dispatch time; (): wait for nothing.
        self._before = before

    @property
    def offloaded(self) -> bool:
        return self._own is not None

    async def dispatch(self, send: Callable[[Any], None], tokenized_obj: Any) -> None:
        """Wait for the earlier image requests, wrap off the loop, send on it."""
        before = self._before
        if before is None:
            before = self._order.open_events()
        for ev in before:
            await ev.wait()
        if self._own is not None:
            from sglang.srt.managers.mm_utils import wrap_shm_features

            t0 = time.perf_counter()
            tokenized_obj = await run_offloaded(wrap_shm_features, tokenized_obj)
            note(shm_wrap_ms=(time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        send(tokenized_obj)
        note(send_ms=(time.perf_counter() - t0) * 1000)


class MmDispatchOrder:
    """FIFO dispatch across the offloaded multimodal section of the tokenizer.

    An offloaded image request opens an event when it enters and sets it when
    it leaves (dispatched, failed or cancelled -- always). It dispatches after
    the image requests that opened before it; any other request dispatches
    after every image request open at its own dispatch time.
    """

    def __init__(self):
        self._open: Deque[asyncio.Event] = collections.deque()

    def open_events(self) -> Tuple[asyncio.Event, ...]:
        return tuple(self._open)

    @contextlib.asynccontextmanager
    async def request(self, *, rid: Any, is_mm: bool) -> AsyncIterator[_Turn]:
        exempt = isinstance(rid, str) and rid.startswith(HEALTH_CHECK_RID_PREFIX)
        if exempt:
            yield _Turn(order=self, own=None, before=())
            return
        offload = bool(is_mm) and offload_permitted()
        own: Optional[asyncio.Event] = None
        before: Optional[Tuple[asyncio.Event, ...]] = None
        if offload:
            own = asyncio.Event()
            before = tuple(self._open)
            self._open.append(own)
        stats = MmTokenizeStats.start(rid=rid, offloaded=offload) if is_mm else None
        offload_token = _OFFLOAD.set(offload)
        stats_token = _STATS.set(stats)
        try:
            yield _Turn(order=self, own=own, before=before)
        finally:
            _STATS.reset(stats_token)
            _OFFLOAD.reset(offload_token)
            if own is not None:
                self._open.remove(own)
                own.set()
            if stats is not None:
                try:
                    stats.report()
                except Exception:  # noqa: BLE001 -- an instrument never fails a request
                    pass
