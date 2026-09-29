"""Ladezeit 2 (23.09., fnFL2x26): the expert-shard CONSUMER in threads.

The safetensors loader reads with eight file workers, but every expert shard
was consumed on ONE thread: the model's ``load_weights`` called
``FusedMoE.weight_loader`` per (expert, shard), and each call ends in a
strided host copy (``expert_data.copy_``) that was 50 % of a 107 s rank load
while the workers sat in ``threading.wait`` with a full buffer.

``ExpertLoadPool`` takes those calls off the loader thread:

* ``submit(fn, *args, **kwargs)`` queues one call; at most ``2 x threads``
  are in flight, so the loader thread is throttled to the consumers and the
  sliding-window file buffer remains the only holder of mmaps.
* ``drain()`` waits for everything and re-raises the FIRST exception. A
  half-loaded weight set loads silently wrong, so nothing is swallowed and
  after an error no further call is accepted.
* Every worker binds the CUDA device of the thread that built the pool
  (a fresh thread starts on device 0; the per-layer presplit that fires from
  the completing call launches kernels on the current device).

Why this is safe for FusedMoE: the destinations are disjoint rows of a
stacked ``[E, ...]`` parameter (one (expert, shard) per call), the per-layer
presplit trigger ``_ct_stream_note`` counts under its own lock and fires at
most once, and ``copy_`` releases the GIL. ``threads=0`` is the serial form:
``submit`` runs the call inline, which is exactly the pre-existing path.
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, List, Optional

from sglang.srt.environ import envs

#: The pool of the load in progress (one load at a time per process); read
#: by FusedMoE._ct_stream_note to hand the per-layer presplit back to the
#: loader thread. None outside a load.
_CURRENT: Optional["ExpertLoadPool"] = None


def current_pool() -> Optional["ExpertLoadPool"]:
    return _CURRENT


def consumer_threads() -> int:
    """The configured consumer count, never negative."""
    return max(0, int(envs.SGLANG_LOAD_CONSUMER_THREADS.get() or 0))


def presplit_thread_enabled() -> bool:
    return bool(envs.SGLANG_LOAD_PRESPLIT_THREAD.get())


class PresplitWorker:
    """BOOTZEIT 3 Stufe 1: the per-layer presplit on ONE serial thread.

    rc12z30o3: the presplit (H2D of the [E] stack, Marlin repack, host-store
    spill, pool release) summed to PP0 26.8 of 60 s and TP0 33.4 of 62.8 s,
    all on the loader thread -- reading and consuming stood still meanwhile.
    Here it runs behind the load instead:

    * serial: one layer at a time, in submission order, so the device
      working set and the pool release stay exactly as on the loader thread;
    * tagged: the thread mirrors the loader's TMS thread-local config
      (``mirrored_weights_region``) before its first call, so every
      allocation inside ``weight_chunk_scope`` carries the layer's band tag
      (fnFL2x31 was the same presplit on an UNTAGGED thread);
    * bounded: at most ``depth`` layers wait behind the running one (each
      keeps its host [E] stack alive); a full queue blocks the submitter;
    * loud: the first failure is re-raised at ``drain()`` and refuses every
      later submit -- a half-split layer set is worse than a failed boot.
    """

    def __init__(self, *, device_index: Optional[int], region: Any,
                 depth: int = 1):
        self._q: "queue.Queue" = queue.Queue(maxsize=max(1, int(depth)))
        self._device_index = device_index
        self._region = region
        self._lock = threading.Lock()
        self._first_error: Optional[BaseException] = None
        self.ran = 0
        self.busy_s = 0.0  # the thread running presplits
        self.submit_wait_s = 0.0  # submitters blocked on a full queue
        self._thread = threading.Thread(
            target=self._main, name="load-presplit", daemon=True)
        self._thread.start()

    def _main(self) -> None:
        import time as _t

        if self._device_index is not None:
            import torch

            if torch.cuda.is_available():
                torch.cuda.set_device(int(self._device_index))
        from sglang.srt.managers.weg2_memory_saver import mirrored_weights_region

        with mirrored_weights_region(self._region):
            while True:
                fn = self._q.get()
                try:
                    if fn is None:
                        return
                    with self._lock:
                        failed = self._first_error is not None
                    if failed:
                        continue  # drain the queue, run nothing after a failure
                    t0 = _t.perf_counter()
                    try:
                        fn()
                    except BaseException as e:  # noqa: BLE001 -- re-raised at drain()
                        with self._lock:
                            if self._first_error is None:
                                self._first_error = e
                    finally:
                        self.busy_s += _t.perf_counter() - t0
                        self.ran += 1
                finally:
                    self._q.task_done()

    def _raise_if_failed(self) -> None:
        with self._lock:
            err = self._first_error
        if err is not None:
            raise RuntimeError(
                f"BOOTZEIT3 PRESPLIT-THREAD failed ({type(err).__name__}: {err}); "
                f"refusing further loads -- a half-split layer set is worse "
                f"than a failed boot") from err

    def submit(self, fn: Callable[[], Any]) -> None:
        import time as _t

        self._raise_if_failed()
        t0 = _t.perf_counter()
        self._q.put(fn)
        self.submit_wait_s += _t.perf_counter() - t0

    def drain(self) -> None:
        self._q.join()
        self._raise_if_failed()

    def close(self) -> None:
        if self._thread.is_alive():
            self._q.put(None)
            self._thread.join()


class ExpertLoadPool:
    """A bounded pool of consumer threads for expert-shard loads."""

    def __init__(self, threads: int, *, device_index: Optional[int] = None,
                 in_flight: Optional[int] = None,
                 presplit_thread: Optional[bool] = None):
        self.threads = max(0, int(threads))
        self.device_index = device_index
        self._in_flight = int(in_flight if in_flight is not None else 2 * self.threads)
        self._slots = threading.BoundedSemaphore(max(1, self._in_flight))
        self._lock = threading.Lock()
        self._first_error: Optional[BaseException] = None
        self._pending: List[Future] = []
        self.submitted = 0
        self.completed = 0
        # fnFL2x31: WORK THE LOADER THREAD MUST DO ITSELF. The TMS tag that
        # makes a device allocation pausable is thread_local in the C++ hook
        # (tms_csrc/entrypoint.cpp:40), so the per-layer presplit -- which
        # fires from whichever thread lands a layer's last shard -- allocated
        # UNTAGGED from the consumers: PP1 slept with untagged_live=8408 MiB
        # (x30: 0), card 0 stayed at 10 GB, D TP1 died loading (CUDA OOM).
        # The consumers queue such work here; the loader thread runs it in
        # `submit()` and `drain()`, under its own tag, one at a time.
        self._deferred: List[Callable[[], Any]] = []
        self._loader_ident = threading.get_ident()
        self.deferred_run = 0
        # BOOTZEIT 3 (29.09.): where the loader thread's time goes. The
        # LOAD-PROFILE sampler saw ``threading.py wait`` 49-59 % on every
        # rank of rc12z30o3 but cannot say on WHAT -- consumer slots, reads,
        # or nothing. These three clocks split it (instrument only).
        self.wait_slots_s = 0.0  # loader blocked on a free consumer slot
        self.deferred_s = 0.0  # loader running deferred work (the presplit)
        self.drain_wait_s = 0.0  # loader waiting for the last consumers
        # BOOTZEIT 3 Stufe 1: the presplit off the loader thread. Only when
        # the loader's weights region can be mirrored (a TMS base region is
        # open HERE); otherwise the presplit stays on the loader thread.
        self._presplit: Optional[PresplitWorker] = None
        self.presplit_mode = "loader"
        if presplit_thread if presplit_thread is not None else presplit_thread_enabled():
            from sglang.srt.managers.weg2_memory_saver import (
                capture_weights_region_for_thread,
            )

            region = capture_weights_region_for_thread()
            if region is not None or not _tms_in_use():
                self._presplit = PresplitWorker(
                    device_index=device_index, region=region,
                    depth=int(envs.SGLANG_LOAD_PRESPLIT_DEPTH.get() or 1))
                self.presplit_mode = "thread"
            else:
                self.presplit_mode = "loader(region-not-mirrorable)"
        self._ex: Optional[ThreadPoolExecutor] = None
        if self.threads > 0:
            self._ex = ThreadPoolExecutor(
                max_workers=self.threads,
                thread_name_prefix="load-consumer",
                initializer=self._bind_device,
            )

    # -- device ---------------------------------------------------------
    def _bind_device(self) -> None:
        if self.device_index is None:
            return
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(int(self.device_index))

    # -- submission -----------------------------------------------------
    @property
    def parallel(self) -> bool:
        return self._ex is not None

    def on_loader_thread(self) -> bool:
        return threading.get_ident() == self._loader_ident

    def defer_to_loader(self, fn: Callable[[], Any]) -> None:
        """Run ``fn`` on the loader thread at its next submit() or drain().
        Called from a consumer thread; on the loader thread it runs at once."""
        if self.on_loader_thread():
            import time as _t

            t0 = _t.perf_counter()
            fn()
            self.deferred_s += _t.perf_counter() - t0
            self.deferred_run += 1
            return
        with self._lock:
            self._deferred.append(fn)

    def run_presplit(self, fn: Callable[[], Any]) -> None:
        """The per-layer presplit: on the presplit thread when it runs,
        else on the loader thread (at once there, deferred from a consumer)."""
        if self._presplit is not None:
            self._presplit.submit(fn)
            return
        self.defer_to_loader(fn)

    @property
    def presplit_busy_s(self) -> float:
        return self._presplit.busy_s if self._presplit is not None else 0.0

    @property
    def presplit_submit_wait_s(self) -> float:
        return self._presplit.submit_wait_s if self._presplit is not None else 0.0

    def run_deferred(self) -> None:
        """Loader thread only: run every queued deferred call, in order."""
        while True:
            with self._lock:
                if not self._deferred:
                    return
                fn = self._deferred.pop(0)
            import time as _t

            t0 = _t.perf_counter()
            fn()
            self.deferred_s += _t.perf_counter() - t0
            self.deferred_run += 1

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Queue one consumer call, or run it inline in the serial form."""
        self._raise_if_failed()
        self.run_deferred()
        if self._ex is None:
            fn(*args, **kwargs)
            self.submitted += 1
            self.completed += 1
            return
        # acquire BEFORE submitting: the loader thread blocks here when the
        # consumers are behind, which is the throttle that bounds the mmaps
        # (and the host RAM) the deferred calls keep alive
        import time as _t

        t0 = _t.perf_counter()
        self._slots.acquire()
        self.wait_slots_s += _t.perf_counter() - t0
        try:
            fut = self._ex.submit(self._run, fn, args, kwargs)
        except BaseException:
            self._slots.release()
            raise
        with self._lock:
            self._pending.append(fut)
            self.submitted += 1
            # keep the pending list short: drop what is already done
            if len(self._pending) > 4 * max(1, self._in_flight):
                self._pending = [f for f in self._pending if not f.done()]

    def _run(self, fn, args, kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001 -- re-raised at drain()
            with self._lock:
                if self._first_error is None:
                    self._first_error = e
            raise
        finally:
            with self._lock:
                self.completed += 1
            self._slots.release()

    def _raise_if_failed(self) -> None:
        with self._lock:
            err = self._first_error
        if err is not None:
            raise RuntimeError(
                f"expert-shard consumer failed earlier ({type(err).__name__}: {err}); "
                f"refusing further loads -- a half-loaded weight set is worse than "
                f"a failed boot") from err

    # -- completion -----------------------------------------------------
    def drain(self) -> None:
        """Wait for every queued call; re-raise the first failure."""
        if self._ex is None:
            self.run_deferred()
            self._raise_if_failed()
            if self._presplit is not None:
                self._presplit.drain()
            return
        while True:
            with self._lock:
                pending = list(self._pending)
                self._pending = []
            import time as _t

            t0 = _t.perf_counter()
            for f in pending:
                try:
                    f.result()
                except BaseException:  # noqa: BLE001 -- the first one is re-raised below
                    pass
            self.drain_wait_s += _t.perf_counter() - t0
            self._raise_if_failed()
            # a deferred call may have been queued by the last completions;
            # and nothing else can queue one once every future is done
            self.run_deferred()
            with self._lock:
                done = not self._pending and not self._deferred
            if done:
                # nothing can queue a presplit once every consumer is done
                if self._presplit is not None:
                    self._presplit.drain()
                return

    def close(self) -> None:
        global _CURRENT
        if _CURRENT is self:
            _CURRENT = None
        if self._presplit is not None:
            self._presplit.close()
        if self._ex is not None:
            self._ex.shutdown(wait=True)
            self._ex = None

    def __enter__(self) -> "ExpertLoadPool":
        global _CURRENT
        _CURRENT = self
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self.drain()
        finally:
            self.close()


def _tms_in_use() -> bool:
    """Whether torch_memory_saver's hook is loaded at all (None-safe)."""
    try:
        import torch_memory_saver as _tms  # noqa: F401

        return _tms.torch_memory_saver._impl is not None
    except Exception:  # noqa: BLE001 -- not installed
        return False


def current_device_index() -> Optional[int]:
    """The loader thread's CUDA device, to be inherited by the consumers."""
    import torch

    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.current_device())
