# SPDX-License-Identifier: Apache-2.0
"""Movement half of the #286 offload register -- built ahead of the GPU
window (DESIGN_201 Nachtrag-13 Erg. 7b/7c).

The register (``offload_register.py``) owns the POLICY half: whether an item
may move. This module owns the MOVEMENT half: where it moves and how. It is
deliberately split into

* an injectable, thin DEVICE LAYER (``DeviceOps``) that holds every torch /
  CUDA touch -- ``CudaDeviceOps`` is the real thing (validated in the GPU
  phase), ``FakeDeviceOps`` keeps the whole logic hermetically CPU-testable
  (torch.cuda is never called in tests);
* a PARK-TARGET LADDER (Erg. 7c): ``own_vram`` (tier 0 = resident, not a
  park target) -> ``peer_vram`` (P2P, one hop, no host staging) ->
  ``host_ram`` -> ``remote`` (stub; the #224 RDMA lane is the named
  attachment point);
* a CAPACITY LEDGER per target. Peer-VRAM booking respects the lane/group
  budget of the TARGET card: without an explicitly granted budget for a
  peer device, peer parking is refused -- it must never silently poach the
  target card's KV / group budget. Feeding the granted figure from the
  dual-group per-card budget accounting (pool_configurator / group planner)
  and enforcing it jointly with the KV headroom is an open GPU-phase item
  (TODO below);
* a MOVEMENT STATE MACHINE per item (resident -> park-in-flight -> parked ->
  wave-in-flight -> resident) with explicit error paths: a failed park
  releases its booking and leaves the item resident; a failed wave-in leaves
  the item parked (retrieval is never refused, but a physical failure is
  reported honestly); double-park is a no-op; a wave-in during an in-flight
  park first joins the park, then reverses it.

P2P PATH MODEL (deliberately NOT "full peer addressability"): the upcoming
driver update enables a SPECIAL P2P on this rig -- the 5090 exposes a full
resizable BAR while the 3080s expose only their small BAR aperture. Hence:

* capability is per DIRECTED pair (asymmetry is the NORMAL case: parking
  5090 -> 3080 runs through the small window, 3080 -> 5090 through the full
  BAR), never per card;
* ``PeerPathCapability`` carries ``aperture_bytes`` (None = full
  addressability), reachable ``bandwidth_bytes_per_s`` and
  ``window_switch_cost_s`` -- ALL of these are PLACEHOLDERS to be filled
  from the probe run after the driver update. No aperture constant lives in
  this code; the numbers come from measurement, and an UNMEASURED path is
  never assumed usable;
* items larger than a path's aperture are handled by a SELECTABLE window
  policy: ``reject`` (degrade to host_ram) or ``chunk`` (move in
  aperture-sized chunks). Both exist as code; NEITHER is a settled design
  decision -- the default is the conservative ``reject`` until measured
  figures exist. Where/when allreduce or broadcast can run over this path
  is likewise still open; nothing here pins that down;
* without a usable P2P path, ``peer_vram`` degrades cleanly to ``host_ram``
  with a log line -- never an error.

Routes, by payload (the three generic ingredients from the #286 audit,
docs/dev/INTEGRATION_R3_VALIDATION.md):

* ``TensorPayload``  -- pinned host pool + async H2D behind compute, the
  ``MoEExpertOffloadCache.install/_fetch`` pattern (copy stream,
  ``wait_stream`` in BOTH directions: write-after-read before the copy,
  join before compute reads). For plain (non-graph-captured) buffers.
* ``TagPayload``     -- #93 tagged capture pools / VMM
  (``AdaptiveGraphMemoryManager`` + ``torch_memory_saver_adapter``
  pause/resume): the route for ``va_stable_required`` items -- captured
  graphs survive the park without re-capture.
* ``SuspendPayload`` -- the #89 suspend path (``weight_updater`` /
  ``hibernate`` memory-saver tags) for the ``cold_lane`` class.

Everything except ``CudaDeviceOps`` is pure Python; tests live in
test/registered/unit/model_executor/test_offload_movement.py.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from sglang.srt.model_executor.offload_register import OWN_VRAM_TIER as OWN_VRAM
from sglang.srt.model_executor.offload_register import (
    PARK_TARGETS,
    ClassPolicy,
    MovementBackend,
    OffloadItem,
    parse_park_target_order,
)

__all__ = [
    "OWN_VRAM",
    "PARK_TARGETS",
    "DEFAULT_PARK_TARGET_ORDER",
    "WINDOW_POLICIES",
    "DEFAULT_WINDOW_POLICY",
    "MovementError",
    "PeerPathCapability",
    "PeerProbe",
    "TensorPayload",
    "TagPayload",
    "SuspendPayload",
    "DeviceOps",
    "CudaDeviceOps",
    "FakeDeviceOps",
    "CapacityLedger",
    "RealMovementBackend",
    "MovementStats",
    "parse_park_target_order",
    "park_target_order_from_register",
]

logger = logging.getLogger(__name__)

# Conservative default: host RAM is the measured, proven path. The
# peer-first ladder ("peer_vram>host_ram") is opt-in via
# --lane-offload-park-targets until the P2P window is measured (house rule:
# Messpflicht before any faster default).
DEFAULT_PARK_TARGET_ORDER = ("host_ram",)


def park_target_order_from_register(
    default: Tuple[str, ...] = DEFAULT_PARK_TARGET_ORDER,
) -> Tuple[str, ...]:
    """The operator's ``--lane-offload-park-targets`` chain, or ``default``.

    #421 F2: the flag was parsed at argument time and thrown away. It now
    reaches the register at runner init
    (``configure_global_register_from_server_args``), and this is where the
    movement layer picks it up -- the register is the only object that
    exists in both phases.
    """
    from sglang.srt.model_executor.offload_register import get_global_register

    reg = get_global_register()
    if reg is None:
        return default
    order = reg.park_target_order
    return tuple(order) if order else default


# How a peer path whose BAR aperture is smaller than the item is handled.
# BOTH are selectable policy, NEITHER is a settled decision; the default is
# the conservative reject-and-degrade until the post-driver-update probe run
# supplies aperture / bandwidth / window-switch figures.
WINDOW_POLICIES = ("reject", "chunk")
DEFAULT_WINDOW_POLICY = "reject"

# Movement states.
STATE_RESIDENT = "resident"
STATE_PARK_IN_FLIGHT = "park_in_flight"
STATE_PARKED = "parked"
STATE_WAVE_IN_FLIGHT = "wave_in_flight"


class MovementError(RuntimeError):
    """A movement that physically failed (not a policy refusal -- those are
    ``OffloadRefused`` and never reach the backend)."""


# --- P2P path capability ----------------------------------------------------
@dataclass(frozen=True)
class PeerPathCapability:
    """Capability of ONE DIRECTED P2P path (src accesses dst's memory).

    Every field except ``p2p`` is a PLACEHOLDER to be filled from the probe
    run after the driver update -- no constant in this codebase claims an
    aperture size or a bandwidth; the numbers come from measurement.

    * ``p2p``: the path exists at all (driver-level peer access).
    * ``measured``: the path's parameters were actually probed. An
      unmeasured path is NEVER assumed usable for parking, however tempting
      -- nothing about the upcoming special-BAR P2P is decided yet.
    * ``aperture_bytes``: None = full addressability (e.g. into the 5090's
      full resizable BAR); a value = the window transfers must fit through
      (e.g. toward a 3080). This is the EMPIRICAL EFFECTIVE usable size from
      the probe run, NOT the nominal BAR size -- whether a nominal 256-MiB
      window is fully usable (addressability etc.) only the probe on the
      updated system shows; it may be less. The nominal figure is at most
      informational (``nominal_bar_bytes``) and never used for admission.
    * ``bandwidth_bytes_per_s``: reachable rate on this path, measured.
    * ``window_switch_cost_s``: cost of moving the BAR window, if the path
      is windowed; feeds the (later) chunking cost model.
    * ``nominal_bar_bytes``: optional info-only nominal BAR size for logs /
      telemetry; admission and chunking read ``aperture_bytes`` only.
    """

    p2p: bool
    measured: bool = False
    # Measured EFFECTIVE aperture (probe run), not the nominal BAR size.
    aperture_bytes: Optional[int] = None
    bandwidth_bytes_per_s: Optional[float] = None
    window_switch_cost_s: Optional[float] = None
    nominal_bar_bytes: Optional[int] = None


class PeerProbe:
    """Encapsulated, cached P2P capability probe -- per DIRECTED pair
    (asymmetric setups answer differently per direction; that is the normal
    case on this rig).

    The injected ``capabilities`` table wins; a pair without an entry falls
    back to the device layer's existence probe, which yields an UNMEASURED
    capability -- and unmeasured paths are refused for parking until the
    post-driver-update probe run feeds ``set_capability``.
    """

    def __init__(
        self,
        device_ops: "DeviceOps",
        capabilities: Optional[Mapping[Tuple[int, int], PeerPathCapability]] = None,
    ):
        self._ops = device_ops
        self._caps: Dict[Tuple[int, int], PeerPathCapability] = dict(capabilities or {})
        self._probed: Dict[Tuple[int, int], PeerPathCapability] = {}

    def set_capability(
        self, src_device: int, dst_device: int, cap: PeerPathCapability
    ) -> None:
        """Measurement feed-in (probe run after the driver update)."""
        self._caps[(int(src_device), int(dst_device))] = cap

    def capability(self, src_device: int, dst_device: int) -> PeerPathCapability:
        key = (int(src_device), int(dst_device))
        cap = self._caps.get(key)
        if cap is not None:
            return cap
        if key not in self._probed:
            self._probed[key] = PeerPathCapability(
                p2p=bool(self._ops.can_device_access_peer(*key)),
                measured=False,
            )
        return self._probed[key]

    def path_admits(
        self,
        src_device: int,
        dst_device: int,
        nbytes: int,
        window_policy: str = DEFAULT_WINDOW_POLICY,
    ) -> Tuple[bool, Optional[int], str]:
        """Whether this directed path can carry an ``nbytes`` park.

        Returns ``(ok, chunk_bytes, reason)``; ``chunk_bytes`` is None for a
        single-piece transfer, or the aperture size when the transfer must be
        chunked (window policy ``chunk``).
        """
        cap = self.capability(src_device, dst_device)
        if not cap.p2p:
            return False, None, "no P2P path"
        if not cap.measured:
            return (
                False,
                None,
                "P2P path exists but is unmeasured (aperture/bandwidth "
                "unknown until the post-driver-update probe run)",
            )
        if cap.aperture_bytes is None or nbytes <= cap.aperture_bytes:
            return True, None, "ok"
        if window_policy == "chunk":
            return True, int(cap.aperture_bytes), "chunked through BAR window"
        return (
            False,
            None,
            f"item ({nbytes} B) exceeds the {cap.aperture_bytes}-B BAR "
            f"aperture and the window policy is 'reject'",
        )


# --- payloads ---------------------------------------------------------------
@dataclass(frozen=True)
class TensorPayload:
    """Plain tensors moved via the pinned-pool D2H/H2D route. NOT valid for
    ``va_stable_required`` items: their storage pointers are baked into
    captured graphs, so they must take the #93 tag route instead."""

    tensors: Tuple[object, ...]


@dataclass(frozen=True)
class TagPayload:
    """A #93 memory-saver / tagged-capture-pool tag. Parking pauses the tag
    (VMM unmap, VA stays stable; ``cpu_backup`` keeps the content), waving in
    resumes it. GPU wiring note: tags owned by AdaptiveGraphMemoryManager
    must be paused THROUGH its ensure_active bookkeeping, not behind its
    back -- that arbitration is a named open GPU-phase item."""

    tag: str
    cpu_backup: bool = True


@dataclass(frozen=True)
class SuspendPayload:
    """The #89 suspend route for ``cold_lane``: pause a set of memory-saver
    tags (weights/kv/graphs family). Today those tags are process-wide;
    LANE-SCOPED tags are the named open GPU-phase item (audit table)."""

    tags: Tuple[str, ...]


# --- device layer -----------------------------------------------------------
class DeviceOps:
    """Injectable thin device layer -- the ONLY place that may touch torch /
    CUDA. Every method is a verb the backend needs, nothing more."""

    def can_device_access_peer(self, src_device: int, dst_device: int) -> bool:
        """DIRECTED P2P existence probe (src accesses dst's memory). Says
        nothing about aperture or rate -- those come from the measured
        ``PeerPathCapability``."""
        raise NotImplementedError

    def copy_out_tensors(
        self,
        payload: TensorPayload,
        target: str,
        peer_device: Optional[int],
        chunk_bytes: Optional[int] = None,
    ) -> object:
        """Start the async copy of the payload's tensors to ``target``
        (host_ram: pinned pool; peer_vram: ``peer_device``, chunked into
        ``chunk_bytes`` pieces when the path is BAR-windowed) and arrange for
        the source VRAM to be released once joined. Returns an opaque
        in-flight handle."""
        raise NotImplementedError

    def copy_in_tensors(self, handle: object) -> object:
        """Start the async copy back into the original device tensors.
        Takes the handle ``copy_out_tensors`` returned; returns an in-flight
        handle for ``wait``."""
        raise NotImplementedError

    def wait(self, handle: object) -> None:
        """Join an in-flight transfer (the compute stream must not read the
        destination before this returns)."""
        raise NotImplementedError

    def free_destination(self, handle: object) -> None:
        """Release the park destination (pinned rows / peer allocation)
        after a completed wave-in."""
        raise NotImplementedError

    def pause_tag(self, tag: str, cpu_backup: bool) -> None:
        raise NotImplementedError

    def resume_tag(self, tag: str) -> None:
        raise NotImplementedError

    def suspend_tags(self, tags: Tuple[str, ...]) -> None:
        raise NotImplementedError

    def resume_suspended_tags(self, tags: Tuple[str, ...]) -> None:
        raise NotImplementedError


class CudaDeviceOps(DeviceOps):  # pragma: no cover - requires CUDA (GPU phase)
    """The real device layer. Code complete ahead of the GPU window; its
    VALIDATION (and the measured rates) is the GPU phase -- nothing here runs
    in the hermetic CPU tests.

    The tensor route follows ``MoEExpertOffloadCache._fetch`` exactly:
    a dedicated copy stream, ``copy_stream.wait_stream(current)`` BEFORE the
    copies (write-after-read: the source may still be read by compute
    already enqueued), then ``current.wait_stream(copy_stream)`` at join
    time (compute must not read before the copy lands).

    PLACEHOLDER NOTE (peer path): how the special-BAR P2P window is actually
    driven (mapping granularity, window moves, what allreduce/broadcast may
    share the path) is UNKNOWN until the post-driver-update probe run. The
    chunk loop below is the neutral form -- per-chunk async copies -- and is
    expected to be reshaped by the probe results; it is not a design
    decision.
    """

    def __init__(self, memory_saver_adapter=None):
        import torch  # noqa: PLC0415

        self._torch = torch
        self._stream = torch.cuda.Stream()
        # The #93/#89 pause/resume adapter (TorchMemorySaverAdapter). Injected
        # by the runner-init wiring; without it the tag/suspend routes are
        # unavailable and the backend reports that honestly.
        self._saver = memory_saver_adapter

    def can_device_access_peer(self, src_device: int, dst_device: int) -> bool:
        if src_device == dst_device:
            return False
        return bool(self._torch.cuda.can_device_access_peer(src_device, dst_device))

    def copy_out_tensors(self, payload, target, peer_device, chunk_bytes=None):
        torch = self._torch
        stream = self._stream
        stream.wait_stream(torch.cuda.current_stream())
        stash = []
        with torch.cuda.stream(stream):
            for t in payload.tensors:
                if target == "peer_vram":
                    dst = torch.empty_like(t, device=f"cuda:{peer_device}")
                else:
                    dst = torch.empty_like(t, device="cpu").pin_memory()
                if chunk_bytes is None:
                    dst.copy_(t, non_blocking=True)
                else:
                    # BAR-windowed path: move in aperture-sized chunks.
                    # Neutral placeholder form until the probe run defines
                    # the real windowed transfer (see class docstring).
                    flat_src = t.reshape(-1).view(torch.uint8)
                    flat_dst = dst.reshape(-1).view(torch.uint8)
                    for off in range(0, flat_src.numel(), chunk_bytes):
                        end = min(off + chunk_bytes, flat_src.numel())
                        flat_dst[off:end].copy_(flat_src[off:end], non_blocking=True)
                stash.append((t, dst))
        event = torch.cuda.Event()
        event.record(stream)
        # Source VRAM is released only after the copy is joined (see wait()):
        # resizing the storage while the D2H copy is still reading it would
        # be a use-after-free on the copy stream.
        return {
            "direction": "out",
            "stash": stash,
            "event": event,
            "released": False,
            "chunk_bytes": chunk_bytes,
        }

    def copy_in_tensors(self, handle):
        torch = self._torch
        stream = self._stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for t, dst in handle["stash"]:
                if handle.get("released"):
                    t.untyped_storage().resize_(dst.untyped_storage().nbytes())
                t.copy_(dst, non_blocking=True)
        event = torch.cuda.Event()
        event.record(stream)
        return {"direction": "in", "stash": handle["stash"], "event": event}

    def wait(self, handle):
        handle["event"].synchronize()
        if handle.get("direction") == "out" and not handle.get("released"):
            # Park direction only: the copy has landed, now the source VRAM
            # can go (the wave-in direction must never re-release what it
            # just restored).
            for t, _dst in handle["stash"]:
                if t.device.type == "cuda":
                    t.untyped_storage().resize_(0)
            handle["released"] = True
        self._torch.cuda.current_stream().wait_stream(self._stream)

    def free_destination(self, handle):
        handle["stash"] = []

    def _require_saver(self, route: str):
        if self._saver is None:
            raise MovementError(
                f"the {route} route needs the memory-saver adapter "
                f"(TorchMemorySaverAdapter); none was injected at wiring time."
            )
        return self._saver

    def pause_tag(self, tag, cpu_backup):
        self._require_saver("#93 tag").pause(tag)

    def resume_tag(self, tag):
        self._require_saver("#93 tag").resume(tag)

    def suspend_tags(self, tags):
        saver = self._require_saver("#89 suspend")
        for tag in tags:
            saver.pause(tag)

    def resume_suspended_tags(self, tags):
        saver = self._require_saver("#89 suspend")
        for tag in reversed(tags):
            saver.resume(tag)


class FakeDeviceOps(DeviceOps):
    """Hermetic CPU stand-in: records every op, simulates the directed P2P
    existence matrix (asymmetric setups are testable) and injected failures.
    No torch, no CUDA."""

    def __init__(
        self,
        peer_matrix: Optional[Mapping[Tuple[int, int], bool]] = None,
        fail_ops: Optional[Mapping[str, Exception]] = None,
    ):
        self.peer_matrix = dict(peer_matrix or {})
        self.fail_ops = dict(fail_ops or {})
        self.calls: List[Tuple] = []
        self._next_handle = 0

    def _op(self, name: str, *args) -> None:
        self.calls.append((name, *args))
        exc = self.fail_ops.get(name)
        if exc is not None:
            raise exc

    def can_device_access_peer(self, src_device, dst_device):
        self._op("can_device_access_peer", src_device, dst_device)
        return bool(self.peer_matrix.get((src_device, dst_device), False))

    def copy_out_tensors(self, payload, target, peer_device, chunk_bytes=None):
        self._op("copy_out", target, peer_device, chunk_bytes)
        self._next_handle += 1
        return {
            "id": self._next_handle,
            "target": target,
            "chunk_bytes": chunk_bytes,
            "waited": False,
        }

    def copy_in_tensors(self, handle):
        self._op("copy_in", handle["id"])
        return {"id": handle["id"], "target": handle["target"], "waited": False}

    def wait(self, handle):
        self._op("wait", handle["id"])
        handle["waited"] = True

    def free_destination(self, handle):
        self._op("free_destination", handle["id"])

    def pause_tag(self, tag, cpu_backup):
        self._op("pause_tag", tag, cpu_backup)

    def resume_tag(self, tag):
        self._op("resume_tag", tag)

    def suspend_tags(self, tags):
        self._op("suspend_tags", tuple(tags))

    def resume_suspended_tags(self, tags):
        self._op("resume_suspended_tags", tuple(tags))


# --- capacity ledger --------------------------------------------------------
class CapacityLedger:
    """Byte bookkeeping per park target (and per device for peer_vram).

    Budget semantics:
    * ``host_ram``: no budget set = unlimited (host RAM sits outside the
      VRAM economy; its RATE is governed by the bus arbiter, not here).
    * ``peer_vram``: no budget set = NOT bookable. A peer park may only use
      bytes the target card's lane/group budget explicitly granted -- it
      must never silently poach the peer's KV / group budget.
      TODO (GPU phase): feed ``set_budget("peer_vram", dev, nbytes)`` from
      the dual-group per-card budget accounting (pool_configurator / group
      planner) and enforce it jointly with the KV headroom check there.
    * ``remote``: never bookable here -- stub tier; the #224 RDMA lane is
      the named attachment point.
    """

    def __init__(self):
        self._budget: Dict[Tuple[str, Optional[int]], int] = {}
        self._booked: Dict[Tuple[str, Optional[int]], int] = {}
        self._lock = threading.Lock()

    def set_budget(self, target: str, device: Optional[int], nbytes: int) -> None:
        if target not in PARK_TARGETS:
            raise ValueError(f"unknown park target {target!r}")
        with self._lock:
            self._budget[(target, device)] = int(nbytes)

    def budgeted_peer_devices(self) -> List[int]:
        with self._lock:
            return sorted(
                dev
                for (tgt, dev), n in self._budget.items()
                if tgt == "peer_vram" and dev is not None and n > 0
            )

    def headroom(self, target: str, device: Optional[int]) -> Optional[int]:
        """Remaining bookable bytes; None = unlimited (host_ram, no budget)."""
        with self._lock:
            budget = self._budget.get((target, device))
            if budget is None:
                return None if target == "host_ram" else 0
            return budget - self._booked.get((target, device), 0)

    def book(self, target: str, device: Optional[int], nbytes: int) -> bool:
        nbytes = int(nbytes)
        with self._lock:
            budget = self._budget.get((target, device))
            booked = self._booked.get((target, device), 0)
            if budget is None:
                if target != "host_ram":
                    return False
            elif booked + nbytes > budget:
                return False
            self._booked[(target, device)] = booked + nbytes
            return True

    def release(self, target: str, device: Optional[int], nbytes: int) -> None:
        with self._lock:
            key = (target, device)
            self._booked[key] = max(0, self._booked.get(key, 0) - int(nbytes))

    def booked(self, target: str, device: Optional[int] = None) -> int:
        with self._lock:
            if device is not None:
                return self._booked.get((target, device), 0)
            return sum(n for (tgt, _d), n in self._booked.items() if tgt == target)


# --- backend ----------------------------------------------------------------
@dataclass
class _ItemMovement:
    """Per-item movement bookkeeping."""

    state: str = STATE_RESIDENT
    target: Optional[str] = None
    peer_device: Optional[int] = None
    handle: Optional[object] = None
    booked_bytes: int = 0


@dataclass
class _Binding:
    payload: object
    source_device: int = 0


@dataclass
class MovementStats:
    parks: int = 0
    wave_ins: int = 0
    park_failures: int = 0
    wave_in_failures: int = 0
    #: Wave-ins whose RETRIEVAL succeeded and whose destination release then
    #: failed. Counted apart from ``wave_in_failures`` because the two mean
    #: opposite things about where the bytes are (#514/#505-B-01).
    destination_release_failures: int = 0
    #: Bytes still booked at a park target whose destination could not be
    #: released. They are deliberately never reclaimed automatically -- see
    #: ``wave_in``. A non-zero value here is an operator-visible leak.
    leaked_destination_bytes: int = 0
    peer_degradations: int = 0
    chunked_transfers: int = 0
    bytes_by_target: Dict[str, int] = field(default_factory=dict)


class RealMovementBackend(MovementBackend):
    """The GPU-phase movement backend, logic complete and CPU-testable via
    ``FakeDeviceOps``. What remains genuinely GPU-bound is the VALIDATION of
    ``CudaDeviceOps``, the probe run that fills the ``PeerPathCapability``
    placeholders, and the measured rates -- not this logic.

    Adapters ``bind()`` each item's payload (which route moves it) and source
    device. Route selection:

    * ``TagPayload``     -> #93 pause/resume; ``host_ram`` only (a VMM tag
      pool has no peer-VRAM remap route).
    * ``SuspendPayload`` -> #89 suspend; ``host_ram`` only.
    * ``TensorPayload``  -> pinned-pool D2H/H2D or peer-VRAM copy; REFUSED
      for ``va_stable_required`` items (their addresses are baked into
      captured graphs -- they need the #93 tag route; give them a
      ``TagPayload``).
    """

    def __init__(
        self,
        device_ops: DeviceOps,
        target_order: Optional[Tuple[str, ...]] = None,
        class_policies: Optional[Mapping[str, ClassPolicy]] = None,
        ledger: Optional[CapacityLedger] = None,
        probe: Optional[PeerProbe] = None,
        window_policy: str = DEFAULT_WINDOW_POLICY,
    ):
        # #421 F2: an omitted order takes the operator's
        # --lane-offload-park-targets chain from the configured register, so a
        # GPU-phase construction site honours the flag without having to know
        # it exists. Explicit beats configured; with no register (feature off)
        # this is DEFAULT_PARK_TARGET_ORDER, i.e. today's behaviour.
        if target_order is None:
            target_order = park_target_order_from_register()
        for target in target_order:
            if target not in PARK_TARGETS:
                raise ValueError(f"unknown park target {target!r}")
        if window_policy not in WINDOW_POLICIES:
            raise ValueError(
                f"unknown window policy {window_policy!r}; known: "
                f"{', '.join(WINDOW_POLICIES)}."
            )
        self._ops = device_ops
        self._order = tuple(target_order)
        self._window_policy = window_policy
        self._class_targets: Dict[str, Tuple[str, ...]] = {}
        for klass, policy in (class_policies or {}).items():
            if getattr(policy, "targets", None):
                self._class_targets[klass] = tuple(policy.targets)
        self.ledger = ledger if ledger is not None else CapacityLedger()
        self.probe = probe if probe is not None else PeerProbe(device_ops)
        self._bindings: Dict[str, _Binding] = {}
        self._movements: Dict[str, _ItemMovement] = {}
        self._lock = threading.RLock()
        self.stats = MovementStats()

    # -- wiring --------------------------------------------------------------
    def bind(self, item_id: str, payload: object, source_device: int = 0) -> None:
        """Attach the movement payload of one item (adapter side; CPU-safe
        bookkeeping only). Re-binding replaces (adapters may rebuild)."""
        if not isinstance(payload, (TensorPayload, TagPayload, SuspendPayload)):
            raise ValueError(
                f"unknown payload type {type(payload).__name__} for item {item_id!r}"
            )
        with self._lock:
            self._bindings[item_id] = _Binding(payload, int(source_device))

    def bound_payload(self, item_id: str) -> Optional[object]:
        with self._lock:
            binding = self._bindings.get(item_id)
            return binding.payload if binding is not None else None

    def state_of(self, item_id: str) -> str:
        with self._lock:
            mv = self._movements.get(item_id)
            return mv.state if mv is not None else STATE_RESIDENT

    def target_of(self, item_id: str) -> Optional[str]:
        with self._lock:
            mv = self._movements.get(item_id)
            return mv.target if mv is not None else None

    # -- target ladder -------------------------------------------------------
    def _order_for(self, item: OffloadItem, explicit: Optional[str]):
        if explicit is not None:
            if explicit == OWN_VRAM:
                raise ValueError(
                    f"{OWN_VRAM!r} is tier 0 (stay resident), not a park target"
                )
            if explicit not in PARK_TARGETS:
                raise ValueError(f"unknown park target {explicit!r}")
            # An EXPLICIT target is honoured exactly (fail loud, no silent
            # rerouting); degradation is a property of the POLICY ladder.
            return (explicit,)
        order = self._class_targets.get(item.offload_class, self._order)
        # A peer_vram entry in the ladder always degrades to host_ram when
        # unusable -- the ladder never dead-ends on a missing/unmeasured P2P
        # path (Erg. 7c).
        if "peer_vram" in order and "host_ram" not in order:
            order = order + ("host_ram",)
        return order

    def _select_target(self, item: OffloadItem, binding: _Binding, order):
        """Walk the ladder; returns (target, peer_device, chunk_bytes) or
        raises MovementError naming every rung's refusal."""
        payload = binding.payload
        reasons: List[str] = []
        for target in order:
            if target == "remote":
                # Stub tier: becomes real when the #224 RDMA lane attaches.
                reasons.append(
                    "remote: stub tier (#224 RDMA attachment point, not wired yet)"
                )
                continue
            if isinstance(payload, (TagPayload, SuspendPayload)):
                if target != "host_ram":
                    route = (
                        "#93 tag" if isinstance(payload, TagPayload) else "#89 suspend"
                    )
                    reasons.append(
                        f"{target}: the {route} route parks to host_ram only"
                    )
                    continue
                return target, None, None
            # TensorPayload
            if item.va_stable_required:
                reasons.append(
                    f"{target}: item requires VA stability; a plain tensor "
                    f"copy would break captured graphs -- bind a TagPayload "
                    f"(#93 route) instead"
                )
                continue
            if target == "peer_vram":
                dev, chunk_bytes, why = self._usable_peer_device(item, binding)
                if dev is None:
                    self.stats.peer_degradations += 1
                    logger.info(
                        "offload item %s: peer_vram unavailable from device "
                        "%d (%s); degrading to host_ram",
                        item.item_id,
                        binding.source_device,
                        why,
                    )
                    reasons.append(f"peer_vram: {why}")
                    continue
                return target, dev, chunk_bytes
            # host_ram
            if not self._has_headroom("host_ram", None, item.size_bytes):
                reasons.append("host_ram: capacity budget exhausted")
                continue
            return target, None, None
        raise MovementError(
            f"no park target available for item {item.item_id!r}: " + "; ".join(reasons)
        )

    def _usable_peer_device(self, item, binding):
        """First budgeted peer device whose DIRECTED path admits the item.
        Returns (device, chunk_bytes, reason-if-none)."""
        reasons: List[str] = []
        devices = self.ledger.budgeted_peer_devices()
        if not devices:
            return None, None, "no peer device with a granted budget"
        for dev in devices:
            if dev == binding.source_device:
                continue
            ok, chunk_bytes, why = self.probe.path_admits(
                binding.source_device, dev, item.size_bytes, self._window_policy
            )
            if not ok:
                reasons.append(f"dev{dev}: {why}")
                continue
            if not self._has_headroom("peer_vram", dev, item.size_bytes):
                reasons.append(f"dev{dev}: peer budget exhausted")
                continue
            return dev, chunk_bytes, "ok"
        return None, None, "; ".join(reasons)

    def _has_headroom(self, target, device, nbytes) -> bool:
        headroom = self.ledger.headroom(target, device)
        return headroom is None or headroom >= nbytes

    # -- movement ------------------------------------------------------------
    def park(self, item: OffloadItem, target: Optional[str] = None) -> None:
        with self._lock:
            mv = self._movements.setdefault(item.item_id, _ItemMovement())
            if mv.state in (STATE_PARKED, STATE_PARK_IN_FLIGHT):
                return  # double-park is a no-op
            if mv.state == STATE_WAVE_IN_FLIGHT:
                # Join the in-flight retrieval first, then park again.
                mv.state = STATE_RESIDENT
            binding = self._bindings.get(item.item_id)
            if binding is None:
                raise MovementError(
                    f"item {item.item_id!r} has no bound movement payload; "
                    f"the adapter must bind() a TensorPayload / TagPayload / "
                    f"SuspendPayload before the item can move."
                )
            order = self._order_for(item, target)
            chosen, peer_device, chunk_bytes = self._select_target(item, binding, order)
            if not self.ledger.book(chosen, peer_device, item.size_bytes):
                raise MovementError(
                    f"park target {chosen!r} refused the booking of "
                    f"{item.size_bytes} B for item {item.item_id!r}"
                )
            mv.state = STATE_PARK_IN_FLIGHT
            mv.target = chosen
            mv.peer_device = peer_device
            mv.booked_bytes = item.size_bytes
            try:
                payload = binding.payload
                if isinstance(payload, TagPayload):
                    self._ops.pause_tag(payload.tag, payload.cpu_backup)
                    mv.handle = None
                elif isinstance(payload, SuspendPayload):
                    self._ops.suspend_tags(payload.tags)
                    mv.handle = None
                else:
                    mv.handle = self._ops.copy_out_tensors(
                        payload, chosen, peer_device, chunk_bytes
                    )
                    if chunk_bytes is not None:
                        self.stats.chunked_transfers += 1
            except Exception as exc:
                # Error path: the park failed -- release the booking, the
                # item STAYS RESIDENT, and the caller learns why.
                self.ledger.release(chosen, peer_device, mv.booked_bytes)
                mv.state = STATE_RESIDENT
                mv.target = None
                mv.peer_device = None
                mv.handle = None
                mv.booked_bytes = 0
                self.stats.park_failures += 1
                raise MovementError(
                    f"park of item {item.item_id!r} to {chosen!r} failed: {exc}"
                ) from exc
            if mv.handle is None:
                # Tag/suspend routes settle synchronously.
                mv.state = STATE_PARKED
            # else: the tensor route's transfer is ENQUEUED behind compute
            # (#125 pattern); the item stays PARK_IN_FLIGHT until settle()
            # joins it at a boundary -- or a wave_in joins and reverses it.
            self.stats.parks += 1
            self.stats.bytes_by_target[chosen] = (
                self.stats.bytes_by_target.get(chosen, 0) + item.size_bytes
            )

    def settle(self, item_id: str) -> None:
        """Join an in-flight park (tensor route: the async copy-out was
        enqueued behind compute). Call at a phase/turn boundary once the
        transfer must have landed; tag/suspend routes settle synchronously
        and need no call. Idempotent."""
        with self._lock:
            mv = self._movements.get(item_id)
            if mv is None or mv.state != STATE_PARK_IN_FLIGHT:
                return
            if mv.handle is not None:
                self._ops.wait(mv.handle)
            mv.state = STATE_PARKED

    def wave_in(self, item: OffloadItem) -> None:
        with self._lock:
            mv = self._movements.get(item.item_id)
            if mv is None or mv.state == STATE_RESIDENT:
                return  # idempotent
            if mv.state == STATE_PARK_IN_FLIGHT:
                # A wave-in during an in-flight park: the park's transfer is
                # joined first (the destination must hold a complete copy
                # before it can be the source of the reverse move).
                if mv.handle is not None:
                    self._ops.wait(mv.handle)
                mv.state = STATE_PARKED
            binding = self._bindings.get(item.item_id)
            payload = binding.payload if binding is not None else None
            mv.state = STATE_WAVE_IN_FLIGHT
            # STAGE 1 -- RETRIEVAL. Everything here runs BEFORE the bytes are
            # known to be back, so a failure means they are still at the park
            # target and PARKED is the truthful state.
            try:
                if isinstance(payload, TagPayload):
                    self._ops.resume_tag(payload.tag)
                elif isinstance(payload, SuspendPayload):
                    self._ops.resume_suspended_tags(payload.tags)
                elif mv.handle is not None:
                    self._ops.wait(mv.handle)  # park copy must have landed
                    back = self._ops.copy_in_tensors(mv.handle)
                    self._ops.wait(back)
            except Exception as exc:
                # Retrieval failed physically. The item stays PARKED (its
                # bytes are still at the park target) and keeps its booking;
                # the failure is reported, never swallowed.
                mv.state = STATE_PARKED
                self.stats.wave_in_failures += 1
                raise MovementError(
                    f"wave-in of item {item.item_id!r} from {mv.target!r} failed: {exc}"
                ) from exc

            # STAGE 2 -- RELEASE OF THE PARK DESTINATION. Reached only once
            # the retrieval above returned, i.e. the bytes ARE back on the
            # device. A failure here is therefore NOT a failed wave-in and
            # must not be reported as one: the item is RESIDENT either way,
            # and only the destination allocation is in doubt (#514/#505-B-01
            # -- the old single `try` marked this case PARKED, the state whose
            # meaning is "the bytes are at the park target", and skipped the
            # ledger release below because it sat outside the try).
            #
            # The booking is deliberately RETAINED and counted as leaked
            # rather than released: we do not know how much of the destination
            # actually came back, and under-booking would let the next booker
            # allocate into bytes that may still be held. A leak that is named
            # and counted is recoverable by an operator; a silent one is not.
            if mv.handle is not None and not isinstance(
                payload, (TagPayload, SuspendPayload)
            ):
                try:
                    self._ops.free_destination(mv.handle)
                except Exception as exc:
                    leaked = int(mv.booked_bytes)
                    self.stats.destination_release_failures += 1
                    self.stats.leaked_destination_bytes += leaked
                    target, peer_device = mv.target, mv.peer_device
                    self._finish_wave_in(mv, release_booking=False)
                    logger.error(
                        "offload: item %r came back from %r but its destination "
                        "could not be released: %s -- %d bytes stay booked at "
                        "%r (device %r) and will NOT be reclaimed automatically",
                        item.item_id,
                        target,
                        exc,
                        leaked,
                        target,
                        peer_device,
                    )
                    raise MovementError(
                        f"item {item.item_id!r} was retrieved from {target!r} but its "
                        f"destination release failed: {exc}; {leaked} bytes remain "
                        f"booked at {target!r} and are leaked"
                    ) from exc
            self._finish_wave_in(mv, release_booking=True)
            self.stats.wave_ins += 1

    def _finish_wave_in(self, mv: _ItemMovement, *, release_booking: bool) -> None:
        """Put a retrieved item back into the RESIDENT state.

        Called on both wave-in exits, because in both of them the bytes are
        back on the device and the movement record must stop describing a
        park. ``release_booking=False`` is the leaked-destination exit: the
        record is cleared so no later cycle can release these bytes a second
        time, and the ledger keeps them booked on purpose (see ``wave_in``).
        The caller must have accounted the leak before calling.
        """
        if release_booking:
            self.ledger.release(mv.target, mv.peer_device, mv.booked_bytes)
        mv.state = STATE_RESIDENT
        mv.target = None
        mv.peer_device = None
        mv.handle = None
        mv.booked_bytes = 0
