"""H25 (Nutzer-Order 24.09. 08:25Z): D's MTP draft sleeps in pinned system RAM.

*"die draft layerbytes aus D muessen dann waehrend P laeuft in den systemram
offgeloaded werden."*  Group P carries no draft any more, so D's
``weights_draft`` tag has no exchange partner and has left the weights family
(``weg2_memory_saver.draft_tag_in_family``).  Its region is opened WITHOUT a
cpu backup on the exchange arm (``weights_cpu_backup_armed`` is False under
exchange+authoritative, model_runner.py), so a plain ``pause`` would discard the
bytes -- and the quantized MTP checkpoint has no disk net (W106).  Hence this
park, proven at the code rather than assumed:

* SLEEP: every storage the draft runner reaches (parameters, buffers, tensor
  attributes -- the population ``weight_exchange.walk_live_tensors`` names),
  minus the storages it shares with its TARGET (H1b: embed_tokens/lm_head are
  the target's tensors, region ``weights``; copying them back later would
  write into a region the exchange owns), deduplicated by storage, is copied
  D2H into ONE pinned host image, then the tag is paused (VRAM released for
  P's experts on the 5090).
* WAKE: the tag is resumed -- the saver maps new physical pages under the SAME
  virtual addresses (cuMemMap at the recorded VA), which is why every CUDA
  graph the verifier captured stays valid without a re-capture, exactly like
  every other tag -- and the image is copied H2D on a side stream, overlapped
  with the rest of the wake; :meth:`DraftHostPark.join` makes the default
  stream wait before the rank reports its wake done.

The host image is allocated ONCE per process (first park) and reused: the
ledger books it as the permanent post ``d_draft_host``
(``host_ledger.charge_terms``), and re-allocating 1.5 GiB of pinned memory per
flip would cost ~0.3 s of cudaHostAlloc on every D sleep.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import msgspec
import torch

MIB = float(1 << 20)

PARK_LINE = "WEG2-DRAFT-PARK"
UNPARK_LINE = "WEG2-DRAFT-UNPARK"


class ParkEntry(msgspec.Struct, frozen=True):
    """One storage of the draft: its device base address and size, and where
    it lives in the host image."""

    key: Tuple[int, int]
    offset: int
    nbytes: int
    name: str


def _storage_key(t: torch.Tensor) -> Optional[Tuple[int, int]]:
    try:
        st = t.untyped_storage()
    except (RuntimeError, NotImplementedError):
        return None
    n = int(st.nbytes())
    if n <= 0:
        return None
    return int(st.data_ptr()), n


def _named_tensors(model: Any) -> Iterable[Tuple[str, torch.Tensor]]:
    for name, p in model.named_parameters():
        yield name, p
    for name, b in model.named_buffers():
        if b is not None:
            yield name, b
    for path, mod in model.named_modules():
        for attr, v in list(vars(mod).items()):
            if isinstance(v, torch.Tensor):
                yield (f"{path}.{attr}" if path else attr), v


def park_population(draft_model: Any, target_model: Any = None) -> List[Tuple[str, torch.Tensor]]:
    """``[(name, byte view of the whole storage)]`` the park must carry.

    Deduplicated by storage (a view is not a second tensor); META tensors hold
    no bytes (a solo-shadow draft on TP1/TP2); storages the TARGET also
    reaches are excluded -- they live in the target's region and are the
    exchange's to move.
    """
    shared = set()
    if target_model is not None:
        for _n, t in _named_tensors(target_model):
            if t.device.type == "meta":
                continue
            k = _storage_key(t)
            if k is not None:
                shared.add(k)
    seen = set()
    out: List[Tuple[str, torch.Tensor]] = []
    for name, t in _named_tensors(draft_model):
        if t.device.type == "meta":
            continue
        k = _storage_key(t)
        if k is None or k in shared or k in seen:
            continue
        seen.add(k)
        view = torch.empty(0, dtype=torch.uint8, device=t.device).set_(t.untyped_storage())
        out.append((str(name), view))
    return out


class ParkRecord(msgspec.Struct, frozen=True):
    tag: str
    storages: int
    nbytes: int
    alloc_ms: float
    copy_ms: float
    pause_ms: float

    def line(self) -> str:
        return (
            f"{PARK_LINE} tag={self.tag} bytes={self.nbytes} mib={self.nbytes / MIB:.1f} "
            f"storages={self.storages} ms={self.copy_ms + self.pause_ms:.0f} "
            f"(d2h {self.copy_ms:.0f} + pause {self.pause_ms:.0f}; host image "
            f"{'allocated ' + format(self.alloc_ms, '.0f') + ' ms' if self.alloc_ms >= 0 else 'reused'}; "
            f"ledger post d_draft_host)"
        )


class DraftHostPark:
    """The one pinned host image of this process's draft, and its two legs.

    ``pin`` / ``new_stream`` / ``resume`` / ``pause`` are injected so the
    order (copy -> pause at the sleep; resume -> copy -> join at the wake) is
    testable without a device.
    """

    def __init__(self, *, pin: bool = True,
                 new_stream: Optional[Callable[[], Any]] = None) -> None:
        self.pin = bool(pin)
        self.new_stream = new_stream
        self.host: Optional[torch.Tensor] = None
        self.entries: List[ParkEntry] = []
        self.views: List[torch.Tensor] = []
        self.parked = False
        self.stream: Any = None
        self.event: Any = None
        self.unpark_t0: Optional[float] = None
        self.unpark_issue_ms = 0.0

    @property
    def holds_image(self) -> bool:
        """A park has written the host image: the draft's wake source (H25d)."""
        return self.host is not None and bool(self.entries)

    @property
    def nbytes(self) -> int:
        return sum(e.nbytes for e in self.entries)

    def _layout(self, population: Sequence[Tuple[str, torch.Tensor]]) -> float:
        """Place every storage in the host image; allocate it ONCE. Returns the
        allocation ms (-1 = reused)."""
        entries = []
        off = 0
        for name, view in population:
            k = _storage_key(view)
            n = int(view.numel())
            entries.append(ParkEntry(key=k, offset=off, nbytes=n, name=name))
            off += n
        if self.host is not None and [e.key for e in entries] == [e.key for e in self.entries]:
            self.views = [v for _n, v in population]
            return -1.0
        if self.host is not None and int(self.host.numel()) >= off:
            self.entries, self.views = entries, [v for _n, v in population]
            return -1.0
        t0 = time.perf_counter()
        self.host = torch.empty(max(1, off), dtype=torch.uint8, pin_memory=self.pin)
        self.entries, self.views = entries, [v for _n, v in population]
        return (time.perf_counter() - t0) * 1000

    def park(self, population: Sequence[Tuple[str, torch.Tensor]], *, tag: str,
             pause: Callable[[str], None], sync: Callable[[], None]) -> ParkRecord:
        """D2H every storage into the host image, then ``pause(tag)``.

        The copy MUST complete before the pause: ``pause`` unmaps the pages
        the copy reads (the campaign (a) fault class), so ``sync`` sits
        strictly between them.
        """
        if self.parked:
            raise RuntimeError(f"{PARK_LINE} tag={tag}: already parked (a second park "
                               "would overwrite the image with unmapped pages)")
        alloc_ms = self._layout(population)
        t0 = time.perf_counter()
        for e, view in zip(self.entries, self.views):
            self.host[e.offset:e.offset + e.nbytes].copy_(view, non_blocking=True)
        sync()
        t1 = time.perf_counter()
        pause(tag)
        t2 = time.perf_counter()
        self.parked = True
        return ParkRecord(tag=str(tag), storages=len(self.entries), nbytes=self.nbytes,
                          alloc_ms=alloc_ms, copy_ms=(t1 - t0) * 1000,
                          pause_ms=(t2 - t1) * 1000)

    def unpark_start(self, *, tag: str, resume: Callable[[str], None]) -> float:
        """``resume(tag)`` (same VA, fresh pages), then issue the H2D copies on
        the side stream and record an event. Returns the resume ms."""
        if not self.parked:
            return 0.0
        t0 = time.perf_counter()
        resume(tag)
        t1 = time.perf_counter()
        self.unpark_t0 = t1
        stream = self.stream
        if stream is None and self.new_stream is not None:
            stream = self.stream = self.new_stream()
        ctx = torch.cuda.stream(stream) if stream is not None else _Null()
        with ctx:
            if stream is not None:
                # the side stream must not overtake work the default stream
                # already queued on these addresses (none after the pause, but
                # the ordering is the contract, not the hope)
                stream.wait_stream(torch.cuda.current_stream())
            for e, view in zip(self.entries, self.views):
                view.copy_(self.host[e.offset:e.offset + e.nbytes], non_blocking=True)
            if stream is not None:
                self.event = torch.cuda.Event()
                self.event.record(stream)
        if self.event is not None:
            # GPU-side ordering NOW, host-side wait only at join: every kernel
            # the wake queues after this point on the default stream (the
            # local-scratch zeroing and the expert rearm reach the draft's
            # own tensors) runs after the copy, while the host goes on.
            torch.cuda.current_stream().wait_event(self.event)
        self.unpark_issue_ms = (time.perf_counter() - t1) * 1000
        self.parked = False
        return (t1 - t0) * 1000

    def join(self, *, overlap: str) -> str:
        """Make the default stream wait for the H2D and block until it is
        done; returns the UNPARK line. ``overlap`` names the wake work that ran
        between :meth:`unpark_start` and here."""
        if self.unpark_t0 is None:
            return ""
        t0 = time.perf_counter()
        if self.event is not None:
            self.event.synchronize()
        t1 = time.perf_counter()
        total = (t1 - self.unpark_t0) * 1000
        line = (
            f"{UNPARK_LINE} tag=weights_draft bytes={self.nbytes} mib={self.nbytes / MIB:.1f} "
            f"ms={total:.0f} wait_ms={(t1 - t0) * 1000:.0f} issue_ms={self.unpark_issue_ms:.0f} "
            f"overlap={overlap or 'none'} (H2D on a side stream from resume to join; "
            f"wait_ms is what the wake paid on its own clock, 0 = fully hidden)"
        )
        self.unpark_t0 = None
        self.event = None
        return line


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
