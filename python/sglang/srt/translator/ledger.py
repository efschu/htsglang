# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The translator's audio modules, as ledger-visible assets.

User order, 2026-08-03: every asset lives under one runtime and one ledger.
A component whose VRAM the register cannot see makes every cross-asset
arbitration decision silently wrong -- the memory-hierarchy program only works
if spill, offload and eviction can compare a translator weight block against a
cold expert, an idle session and a parked graph family on ONE ladder.

So the four audio modules are registered in the #286 register as the
``audio_modules`` asset class, rank ``COLD_SECOND_MODEL``, and they really
park: :meth:`AudioAssetLedger.park` moves a module's tensors to host RAM and
:meth:`restore` brings them back, with the register's own bookkeeping updated
both ways. Registration alone would be decoration -- the whole point is that
the bytes actually move when something more important needs them.

Rung-B scope, stated so #488 does not inherit a wrong assumption: nothing here
captures CUDA graph addresses, which is why the descriptor sets
``va_stable_required=False`` and why the plain tensor route is valid. The
native-lane rung has to revisit that, because a captured graph pins addresses
and a free/re-allocate park would invalidate the capture.

The ledger degrades to a no-op recorder when the global register is not
configured (the ordinary case on the desk and in the hermetic suite), so the
translator runs identically with and without it.

#546 added three things to this module, all of them about a park being REAL
rather than nominal:

* **Park routes** (:class:`ParkRoute`). Not every audio asset is an
  ``nn.Module``. The recognizer is CTranslate2, whose weights never enter
  torch's caching allocator, so the tensor route above cannot see them at all.
  A route is the adapter seam that lets such an asset join the SAME
  ``audio_modules`` class through its own mover, instead of growing a second,
  private park mechanism next to this one. It is also the seam #488's native
  lane will register through.
* **Pinned host copies.** A park happens while nobody is waiting; a restore
  happens while somebody is. So the expensive side (page-locking) is paid at
  park time and the restore gets the DMA path.
* **Cache release.** Freeing a tensor returns its pages to torch's caching
  allocator, not to the driver: ``nvidia-smi`` does not move, and no other
  process can take the memory. A park that stops there is decorative in
  exactly the way this ledger's docstring warns about, one level lower down.
  :meth:`AudioAssetLedger.release_device_cache` is the call that finishes it.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "AudioAsset",
    "AudioAssetLedger",
    "OFFLOAD_CLASS",
    "ParkError",
    "ParkRoute",
]

#: The asset class this module registers under. Declared in
#: ``offload_register.OFFLOAD_CLASSES`` with a descriptor in
#: ``short_term_offload_register.ASSET_CLASSES``.
OFFLOAD_CLASS = "audio_modules"

#: NEED ORDER of the call pipeline. A wake restores rank by rank, lowest
#: first, and a request may be served as soon as the rank it needs is back --
#: an all-or-nothing barrier would make every wake cost the whole stack even
#: though a turn cannot reach the synthesizer before it has been recognized.
#:
#: The order is the pipeline's own: audio arrives and is RECOGNIZED (asr),
#: the speaker is identified and the translation is fetched (both of which
#: take real time and neither of which touches the talker), and only then is
#: anything SYNTHESIZED (talker, then the codec that decodes its output last).
#: So the codec -- the largest single module at 229 MB -- is the one with the
#: most slack to arrive late, which is the same reasoning that made it the
#: first park victim in rung B.
DEFAULT_WAKE_RANKS: Dict[str, int] = {
    "asr": 0,
    "talker_trunk": 1,
    "code_predictor": 1,
    "speaker_encoder": 1,
    "codec": 2,
}
#: Anything unnamed restores with the synthesizer: late enough to keep the
#: recognizer's head start, early enough that an unknown asset cannot be the
#: thing a turn silently waits for at the end.
FALLBACK_WAKE_RANK = 1
#: "Everything", for a caller that needs the whole stack.
WAKE_RANK_ALL = 1 << 30


class ParkError(RuntimeError):
    """A park or restore that could not be carried out."""


class ParkRoute(Protocol):
    """How an asset that is not an ``nn.Module`` moves itself.

    The contract is deliberately the same three questions the tensor route
    answers, so the ledger's bookkeeping, the register's accounting and the
    idle-park state machine are identical for both kinds of asset:

    * :meth:`park` moves the bytes off the device and returns how many were
      freed (0 = unknown, never 0 = none -- see :meth:`size_bytes`);
    * :meth:`restore` brings them back, synchronously;
    * :meth:`size_bytes` reports the current device footprint.

    ``device`` names the torch device the bytes live on, so a per-card
    residency event can be built without the ledger knowing what kind of
    asset this is.
    """

    @property
    def device(self) -> Optional[str]: ...

    def park(self) -> int: ...

    def restore(self) -> None: ...

    def size_bytes(self) -> int: ...


@dataclasses.dataclass
class AudioAsset:
    """One parkable audio asset.

    Either ``module`` is a torch ``nn.Module`` whose parameters and buffers
    move as a unit, or ``route`` is a :class:`ParkRoute` that owns the
    movement itself. Exactly one of the two carries an asset;
    ``restore_cost_ms`` is what the register uses to price bringing it back,
    and it is measured on the first real restore rather than guessed, because
    a guessed restore cost is what makes a planner pick the wrong victim.
    """

    name: str
    module: object
    #: Non-torch mover. When set, ``module`` is ignored.
    route: Optional[ParkRoute] = None
    #: Position in the wake's need order. See :data:`DEFAULT_WAKE_RANKS`.
    wake_rank: int = FALLBACK_WAKE_RANK
    #: Set once a real restore has been timed. Until then the registered
    #: estimate stands and is labelled as an estimate.
    measured_restore_ms: Optional[float] = None
    parked: bool = False
    #: Bytes the last park actually freed. Retained while parked, because a
    #: parked asset's live footprint is zero and the residency event has to
    #: report what LEFT the card, not what is on it now.
    parked_bytes: int = 0
    #: Host-resident copies while parked. Empty when resident.
    _cpu_state: Dict[str, object] = dataclasses.field(default_factory=dict)
    #: Host copies of the NON-PERSISTENT buffers, which ``state_dict()`` omits
    #: by definition and ``load_state_dict`` therefore cannot restore. Kept
    #: separately rather than merged into ``_cpu_state`` because the restore
    #: uses ``strict=True``, and a strict load handed a key the module does not
    #: expect in its state dict raises.
    _cpu_nonpersistent: Dict[str, object] = dataclasses.field(default_factory=dict)
    _device: Optional[str] = None

    @property
    def device(self) -> Optional[str]:
        """The torch device this asset's bytes live on (or last lived on)."""
        if self.route is not None:
            return self.route.device
        if self._device is not None:
            return self._device
        for tensor in self.tensors():
            device = getattr(tensor, "device", None)
            if device is not None:
                return str(device)
        return None

    def tensors(self) -> List[object]:
        """Parameters and buffers, in a stable order."""
        if self.route is not None:
            return []
        module = self.module
        if not hasattr(module, "state_dict"):
            return []
        return [t for _k, t in sorted(module.state_dict().items())]

    def size_bytes(self) -> int:
        if self.route is not None:
            return max(0, int(self.route.size_bytes()))
        total = 0
        for tensor in self.tensors():
            nbytes = getattr(tensor, "nbytes", None)
            if nbytes is None:
                numel = getattr(tensor, "numel", None)
                if numel is None:
                    continue
                element_size = getattr(tensor, "element_size", lambda: 4)()
                nbytes = numel() * element_size
            total += int(nbytes)
        return total


class AudioAssetLedger:
    """Registers the translator's audio modules and parks them on demand.

    Thread-safe: the park path can be driven by a memory-pressure planner on
    another thread while a turn is running, and a half-parked module is the
    one state that must never be observable.
    """

    def __init__(
        self,
        tenant_id: str = "translator",
        clock: Callable[[], float] = time.monotonic,
        pin_host_copies: bool = True,
    ) -> None:
        self.tenant_id = tenant_id
        self._clock = clock
        #: Page-lock the host copies. The asymmetry is the point: pinning is
        #: expensive and happens while nobody waits (park), the DMA it buys is
        #: cashed in while somebody does (restore). Falls back to pageable
        #: memory silently when the allocation fails -- a park that refuses to
        #: happen because the host is short of lockable pages would trade the
        #: whole feature for an optimisation.
        self.pin_host_copies = bool(pin_host_copies)
        self._assets: Dict[str, AudioAsset] = {}
        self._lock = threading.RLock()
        self._register = self._resolve_register()

    @staticmethod
    def _resolve_register():
        """The process-global #286 register, or None when it is not configured.

        Imported lazily and defensively: the translator must run on a desk with
        no register at all, and an import error here would make the whole
        audio path depend on runtime internals it otherwise does not touch.
        """
        try:
            from sglang.srt.model_executor.offload_register import (
                get_global_register,
            )

            return get_global_register()
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.debug("offload register unavailable: %s", exc)
            return None

    @property
    def registered_with_runtime(self) -> bool:
        return self._register is not None

    def __len__(self) -> int:
        return len(self._assets)

    def names(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._assets))

    def get(self, name: str) -> AudioAsset:
        with self._lock:
            try:
                return self._assets[name]
            except KeyError:
                raise ParkError(f"unknown audio asset {name!r}") from None

    # -- registration -------------------------------------------------------

    def register(
        self,
        name: str,
        module: object,
        estimated_restore_ms: float = 250.0,
        route: Optional[ParkRoute] = None,
        wake_rank: Optional[int] = None,
    ) -> AudioAsset:
        """Register one module as a ledgered, parkable asset."""
        with self._lock:
            if name in self._assets:
                raise ParkError(
                    f"audio asset {name!r} is already registered; registering "
                    "twice is a lifecycle bug, not a situation to paper over"
                )
            asset = AudioAsset(
                name=name,
                module=module,
                route=route,
                wake_rank=(
                    DEFAULT_WAKE_RANKS.get(name, FALLBACK_WAKE_RANK)
                    if wake_rank is None
                    else int(wake_rank)
                ),
            )
            self._assets[name] = asset

            if self._register is not None:
                item_id = self._item_id(name)
                self._register.register(
                    item_id=item_id,
                    offload_class=OFFLOAD_CLASS,
                    size_bytes=asset.size_bytes(),
                    restore_cost_ms=(
                        lambda a=asset, e=estimated_restore_ms: (
                            a.measured_restore_ms
                            if a.measured_restore_ms is not None
                            else e
                        )
                    ),
                    hot=(lambda a=asset: not a.parked),
                    va_stable_required=False,
                    time_constant_tier="turn",
                )
                self._bind_payload(item_id, asset)
            logger.info(
                "registered audio asset %s (%.1f MiB)%s",
                name,
                asset.size_bytes() / (1 << 20),
                "" if self._register is not None else " (no runtime register)",
            )
            return asset

    def register_route(
        self,
        name: str,
        route: ParkRoute,
        estimated_restore_ms: float = 1500.0,
        wake_rank: Optional[int] = None,
    ) -> AudioAsset:
        """Register an asset that owns its own mover (see :class:`ParkRoute`).

        The default estimate is deliberately larger than the tensor route's:
        the first route in the tree is a whole recognizer checkpoint coming
        back across PCIe, and an estimate that is too SMALL makes a planner
        think this asset is cheap to give up. Until a restore is measured,
        erring towards expensive is the safe direction.
        """
        return self.register(
            name,
            module=None,
            estimated_restore_ms=estimated_restore_ms,
            route=route,
            wake_rank=wake_rank,
        )

    def _item_id(self, name: str) -> str:
        return f"{self.tenant_id}:{name}"

    def _bind_payload(self, item_id: str, asset: AudioAsset) -> None:
        if asset.route is not None:
            # A route asset has no torch tensors to hand the movement layer.
            # Binding an empty TensorPayload would tell the register it can
            # move this item itself, which is exactly the false claim the
            # route exists to avoid.
            return
        try:
            from sglang.srt.model_executor.offload_movement import TensorPayload
            from sglang.srt.model_executor.offload_register import (
                maybe_bind_movement_payload,
            )

            maybe_bind_movement_payload(
                item_id, TensorPayload(tensors=tuple(asset.tensors()))
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.debug("could not bind movement payload for %s: %s", item_id, exc)

    def register_all(self, modules: Iterable[Tuple[str, object]]) -> None:
        for name, module in modules:
            self.register(name, module)

    # -- movement -----------------------------------------------------------

    def park(self, name: str) -> int:
        """Move one module's tensors to host RAM. Returns bytes freed.

        Idempotent: parking a parked asset is a no-op returning 0 rather than
        an error, because a planner that races another planner should not
        crash the tenant.
        """
        with self._lock:
            asset = self.get(name)
            if asset.parked:
                return 0

            if asset.route is not None:
                device = asset.route.device
                freed = max(0, int(asset.route.park()))
                asset._device = device
                asset.parked = True
                asset.parked_bytes = freed
                self._touch(name)
                self._mark_parked(name, True)
                logger.info(
                    "parked audio asset %s via its own route (%.1f MiB freed)",
                    name, freed / (1 << 20),
                )
                return freed

            tensors = asset.tensors()
            if not tensors:
                asset.parked = True
                asset.parked_bytes = 0
                self._mark_parked(name, True)
                return 0

            import torch

            state = {}
            device = None
            freed = 0
            module = asset.module
            for key, tensor in sorted(module.state_dict().items()):
                if not torch.is_tensor(tensor):
                    continue
                if device is None:
                    device = str(tensor.device)
                state[key] = self._to_host(tensor)
                freed += int(tensor.nbytes)
            # NON-PERSISTENT BUFFERS. `state_dict()` omits them BY DEFINITION,
            # so the restore's `load_state_dict` cannot bring them back, while
            # `to_empty()` DOES re-materialise them -- as uninitialised memory.
            # A park/restore round therefore replaced every such buffer with
            # garbage while reporting success. For the translator's rotary
            # `inv_freq` that means NaN cos/sin, NaN attention scores, NaN
            # logits, and the first visible symptom is "probability tensor
            # contains inf, nan or element < 0" from torch.multinomial, thirty
            # layers and one park away from the cause (the same failure
            # qwen3_tts_compat.refresh_rotary_buffers was written for on the
            # LOAD path -- the restore path never had an equivalent).
            #
            # Carried verbatim rather than recomputed: recomputation is a
            # per-module repair someone has to remember to write for each new
            # buffer, while copying the bytes is correct for every module the
            # ledger will ever be handed, and is byte-exact besides. Counted at
            # runtime -- how many there are is a property of the model, not a
            # constant worth asserting.
            persistent = set(state)
            nonpersistent = {}
            for key, tensor in sorted(module.named_buffers()):
                if key in persistent or not torch.is_tensor(tensor):
                    continue
                if tensor.device.type == "meta":
                    continue
                if device is None:
                    device = str(tensor.device)
                nonpersistent[key] = self._to_host(tensor)
                freed += int(tensor.nbytes)
            asset._cpu_state = state
            asset._cpu_nonpersistent = nonpersistent
            asset._device = device
            # Replace the live tensors with meta-device placeholders so the
            # VRAM is genuinely released rather than merely copied. Assigning
            # the CPU copies would keep the module usable and free nothing,
            # which is the difference between a real park and a decorative one.
            module.to("meta")
            asset.parked = True
            asset.parked_bytes = freed
            self._touch(name)
            self._mark_parked(name, True)
            logger.info("parked audio asset %s (%.1f MiB freed)",
                        name, freed / (1 << 20))
            return freed

    def _to_host(self, tensor):
        """One tensor's host copy, page-locked when that is possible.

        Pinned staging is worth roughly a factor of two on the restore leg,
        which is the leg a user waits through. It is attempted, never
        required: a host short of lockable pages must produce a slower park,
        not a failed one.
        """
        import torch

        detached = tensor.detach()
        if self.pin_host_copies and detached.device.type == "cuda":
            try:
                host = torch.empty(
                    detached.shape, dtype=detached.dtype, pin_memory=True
                )
                host.copy_(detached)
                return host
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                logger.warning(
                    "could not page-lock a %s host copy (%s); "
                    "falling back to pageable memory",
                    tuple(detached.shape), exc,
                )
        return detached.to("cpu", copy=True)

    def restore(self, name: str) -> float:
        """Bring one module back to its device. Returns the milliseconds taken.

        The measured figure replaces the registered estimate, so the register
        prices later victim choices on an observed number.
        """
        with self._lock:
            asset = self.get(name)
            if not asset.parked:
                return 0.0

            if asset.route is not None:
                started = self._clock()
                asset.route.restore()
                elapsed_ms = (self._clock() - started) * 1000.0
                asset.parked = False
                asset.parked_bytes = 0
                asset.measured_restore_ms = elapsed_ms
                self._touch(name)
                self._mark_parked(name, False)
                logger.info(
                    "restored audio asset %s via its own route in %.1f ms",
                    name, elapsed_ms,
                )
                return elapsed_ms

            if not asset._cpu_state:
                raise ParkError(
                    f"audio asset {name!r} is parked but holds no host copy; "
                    "the park was interrupted and the module is unrecoverable "
                    "in place -- reload it from the checkpoint"
                )

            import torch

            started = self._clock()
            target = asset._device or "cpu"
            module = asset.module
            # to_empty() materialises meta parameters on the target device;
            # load_state_dict then fills them. Doing it in one step is not
            # possible from meta, which is exactly why the park recorded the
            # device it came from.
            module.to_empty(device=target)
            module.load_state_dict(
                {
                    k: v.to(target, non_blocking=bool(getattr(v, "is_pinned", bool)()))
                    for k, v in asset._cpu_state.items()
                },
                strict=True,
            )
            # ...and the buffers no state dict can carry. `to_empty()` above
            # gave each of them a correctly shaped tensor full of whatever was
            # in that memory; without this they stay that way. Written through
            # `register_buffer(persistent=False)` so the flag survives the
            # round trip and a second park still finds them here.
            for key, host in asset._cpu_nonpersistent.items():
                owner, _, leaf = key.rpartition(".")
                target_module = module.get_submodule(owner) if owner else module
                target_module.register_buffer(
                    leaf,
                    host.to(
                        target,
                        non_blocking=bool(getattr(host, "is_pinned", bool)()),
                    ),
                    persistent=False,
                )
            if str(target).startswith("cuda"):
                # The pinned copies above are ASYNCHRONOUS. Without this the
                # clock would stop before the bytes have landed, so the
                # measured restore latency -- the number the whole wake-on-
                # demand claim rests on -- would be the launch cost of the
                # copies rather than their completion. It is also a
                # correctness barrier: the first kernel after a restore must
                # not read a half-copied weight.
                torch.cuda.synchronize()
            asset._cpu_state = {}
            asset._cpu_nonpersistent = {}
            asset.parked = False
            asset.parked_bytes = 0
            elapsed_ms = (self._clock() - started) * 1000.0
            asset.measured_restore_ms = elapsed_ms
            self._touch(name)
            self._mark_parked(name, False)
            logger.info("restored audio asset %s in %.1f ms", name, elapsed_ms)
            del torch
            return elapsed_ms

    def ensure_resident(self, *names: str) -> Dict[str, float]:
        """Restore whichever of ``names`` are parked. Returns per-name ms.

        Without an explicit list, restoration follows the NEED ORDER of the
        pipeline (:data:`DEFAULT_WAKE_RANKS`) rather than the alphabet, so
        even a caller that wants the whole stack gets the recognizer first
        and can be unblocked by a staged consumer part way through.
        """
        out: Dict[str, float] = {}
        for name in names or self.wake_order():
            asset = self.get(name)
            if asset.parked:
                out[name] = self.restore(name)
        return out

    def wake_order(self) -> Tuple[str, ...]:
        """Every asset, in need order; ties broken by name for determinism."""
        with self._lock:
            return tuple(
                a.name
                for a in sorted(
                    self._assets.values(), key=lambda x: (x.wake_rank, x.name)
                )
            )

    def parked_wake_ranks(self) -> Tuple[int, ...]:
        """The distinct ranks that currently have something parked, ascending."""
        with self._lock:
            return tuple(
                sorted({a.wake_rank for a in self._assets.values() if a.parked})
            )

    def restore_rank(self, rank: int) -> Dict[str, float]:
        """Restore only the parked assets at one need rank. Returns per-name ms.

        The unit of a CHUNKED wake. Restoring the whole stack under one
        barrier would make the first turn after an idle night wait for the
        codec, which that turn cannot reach until it has been recognized,
        translated and synthesized.
        """
        out: Dict[str, float] = {}
        with self._lock:
            names = [
                a.name
                for a in sorted(self._assets.values(), key=lambda x: x.name)
                if a.wake_rank == rank and a.parked
            ]
        for name in names:
            out[name] = self.restore(name)
        return out

    def park_all(self, release_cache: bool = True) -> int:
        """Park every asset and hand the pages back to the driver.

        ``release_cache`` defaults to True because that is what makes the park
        REAL. Freeing tensors returns their pages to torch's caching
        allocator, where they stay reserved by this process: ``nvidia-smi``
        does not move and no co-tenant can allocate them. Only
        :meth:`release_device_cache` completes the transaction.
        """
        freed = sum(self.park(name) for name in self.names())
        if release_cache:
            self.release_device_cache()
        return freed

    def release_device_cache(self) -> None:
        """Return torch's cached-but-unused pages to the CUDA driver.

        WHY THIS AND NOT THE #330 VMM DECOMMIT. #330's dial reaches the driver
        through ``KvVmmArena.decommit_range`` (``cuMemUnmap`` +
        ``cuMemRelease``) because the memory it dials is a VMM arena it
        reserved itself, and unmapping chunk-wise is the only way to shrink
        one without moving the virtual addresses a captured graph holds. The
        translator has no such arena: its weights are ordinary caching-
        allocator blocks, and nothing captures their addresses (the
        ``audio_modules`` descriptor says so with ``va_stable_required=False``).
        For that allocator the driver-level return is ``empty_cache``. Same
        destination -- pages back with the driver, visible in nvidia-smi,
        allocatable by another process -- reached through the allocator that
        owns the pages. Using the VMM path here would mean building a second
        arena under weights that never needed one.

        What does NOT come back: the CUDA context (a few hundred MiB) and
        cuDNN/cuBLAS workspaces. Those belong to the process, and only exiting
        it returns them. The runbook states the expected residual so nobody
        reads it as a failed park.
        """
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - a desk has no CUDA
            logger.debug("no device cache to release: %s", exc)

    def parked_bytes_by_device(self) -> Dict[str, int]:
        """``{torch device: bytes}`` for everything currently parked.

        The input to a per-card residency event. Keyed on the device the bytes
        LEFT, which a parked asset can no longer be asked for -- hence
        ``AudioAsset.parked_bytes`` and ``_device`` being retained across the
        park rather than recomputed.
        """
        with self._lock:
            out: Dict[str, int] = {}
            for asset in self._assets.values():
                if not asset.parked or asset.parked_bytes <= 0:
                    continue
                device = asset.device or "unknown"
                out[device] = out.get(device, 0) + asset.parked_bytes
            return out

    def resident_bytes_by_device(self) -> Dict[str, int]:
        """``{torch device: bytes}`` for everything currently resident."""
        with self._lock:
            out: Dict[str, int] = {}
            for asset in self._assets.values():
                if asset.parked:
                    continue
                size = asset.size_bytes()
                if size <= 0:
                    continue
                device = asset.device or "unknown"
                out[device] = out.get(device, 0) + size
            return out

    def _touch(self, name: str) -> None:
        try:
            from sglang.srt.model_executor.offload_register import maybe_touch_item

            maybe_touch_item(self._item_id(name))
        except Exception:  # pragma: no cover - environment dependent
            pass

    def _mark_parked(self, name: str, parked: bool) -> None:
        """Keep the #286 register's parked accounting in step with reality.

        Without this the register's ``_parked_bytes_locked`` -- what the class
        fraction cap and every planner pricing this class read -- reports an
        idle, fully host-resident translator as fully device-resident, which
        is wrong in exactly the state the park exists to reach.
        """
        try:
            from sglang.srt.model_executor.offload_register import maybe_mark_parked

            maybe_mark_parked(self._item_id(name), parked)
        except Exception:  # pragma: no cover - environment dependent
            pass

    # -- reporting ----------------------------------------------------------

    def to_json(self) -> Dict[str, object]:
        with self._lock:
            return {
                "tenant_id": self.tenant_id,
                "offload_class": OFFLOAD_CLASS,
                "registered_with_runtime": self.registered_with_runtime,
                "pin_host_copies": self.pin_host_copies,
                "assets": [
                    {
                        "name": a.name,
                        "mib": round(a.size_bytes() / (1 << 20), 2),
                        "parked": a.parked,
                        "parked_mib": round(a.parked_bytes / (1 << 20), 2),
                        "device": a.device,
                        "route": a.route is not None,
                        "wake_rank": a.wake_rank,
                        "restore_ms": (
                            round(a.measured_restore_ms, 1)
                            if a.measured_restore_ms is not None
                            else None
                        ),
                    }
                    for a in sorted(self._assets.values(), key=lambda x: x.name)
                ],
                "resident_mib": round(
                    sum(
                        a.size_bytes() for a in self._assets.values() if not a.parked
                    )
                    / (1 << 20),
                    2,
                ),
            }
