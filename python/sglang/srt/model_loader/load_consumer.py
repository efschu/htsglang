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

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, List, Optional

from sglang.srt.environ import envs


def consumer_threads() -> int:
    """The configured consumer count, never negative."""
    return max(0, int(envs.SGLANG_LOAD_CONSUMER_THREADS.get() or 0))


class ExpertLoadPool:
    """A bounded pool of consumer threads for expert-shard loads."""

    def __init__(self, threads: int, *, device_index: Optional[int] = None,
                 in_flight: Optional[int] = None):
        self.threads = max(0, int(threads))
        self.device_index = device_index
        self._in_flight = int(in_flight if in_flight is not None else 2 * self.threads)
        self._slots = threading.BoundedSemaphore(max(1, self._in_flight))
        self._lock = threading.Lock()
        self._first_error: Optional[BaseException] = None
        self._pending: List[Future] = []
        self.submitted = 0
        self.completed = 0
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

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Queue one consumer call, or run it inline in the serial form."""
        self._raise_if_failed()
        if self._ex is None:
            fn(*args, **kwargs)
            self.submitted += 1
            self.completed += 1
            return
        # acquire BEFORE submitting: the loader thread blocks here when the
        # consumers are behind, which is the throttle that bounds the mmaps
        # (and the host RAM) the deferred calls keep alive
        self._slots.acquire()
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
            self._raise_if_failed()
            return
        with self._lock:
            pending = list(self._pending)
            self._pending = []
        for f in pending:
            try:
                f.result()
            except BaseException:  # noqa: BLE001 -- the first one is re-raised below
                pass
        self._raise_if_failed()

    def close(self) -> None:
        if self._ex is not None:
            self._ex.shutdown(wait=True)
            self._ex = None

    def __enter__(self) -> "ExpertLoadPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self.drain()
        finally:
            self.close()


def current_device_index() -> Optional[int]:
    """The loader thread's CUDA device, to be inherited by the consumers."""
    import torch

    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.current_device())
