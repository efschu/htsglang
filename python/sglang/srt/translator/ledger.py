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
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "AudioAsset",
    "AudioAssetLedger",
    "OFFLOAD_CLASS",
    "ParkError",
]

#: The asset class this module registers under. Declared in
#: ``offload_register.OFFLOAD_CLASSES`` with a descriptor in
#: ``short_term_offload_register.ASSET_CLASSES``.
OFFLOAD_CLASS = "audio_modules"


class ParkError(RuntimeError):
    """A park or restore that could not be carried out."""


@dataclasses.dataclass
class AudioAsset:
    """One parkable audio module.

    ``module`` is a torch ``nn.Module`` whose parameters and buffers move as a
    unit. ``restore_cost_ms`` is what the register uses to price bringing it
    back; it is measured on the first real restore rather than guessed, because
    a guessed restore cost is what makes a planner pick the wrong victim.
    """

    name: str
    module: object
    #: Set once a real restore has been timed. Until then the registered
    #: estimate stands and is labelled as an estimate.
    measured_restore_ms: Optional[float] = None
    parked: bool = False
    #: Host-resident copies while parked. Empty when resident.
    _cpu_state: Dict[str, object] = dataclasses.field(default_factory=dict)
    _device: Optional[str] = None

    def tensors(self) -> List[object]:
        """Parameters and buffers, in a stable order."""
        module = self.module
        if not hasattr(module, "state_dict"):
            return []
        return [t for _k, t in sorted(module.state_dict().items())]

    def size_bytes(self) -> int:
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
    ) -> None:
        self.tenant_id = tenant_id
        self._clock = clock
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
    ) -> AudioAsset:
        """Register one module as a ledgered, parkable asset."""
        with self._lock:
            if name in self._assets:
                raise ParkError(
                    f"audio asset {name!r} is already registered; registering "
                    "twice is a lifecycle bug, not a situation to paper over"
                )
            asset = AudioAsset(name=name, module=module)
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

    def _item_id(self, name: str) -> str:
        return f"{self.tenant_id}:{name}"

    def _bind_payload(self, item_id: str, asset: AudioAsset) -> None:
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
            tensors = asset.tensors()
            if not tensors:
                asset.parked = True
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
                state[key] = tensor.detach().to("cpu", copy=True)
                freed += int(tensor.nbytes)
            asset._cpu_state = state
            asset._device = device
            # Replace the live tensors with meta-device placeholders so the
            # VRAM is genuinely released rather than merely copied. Assigning
            # the CPU copies would keep the module usable and free nothing,
            # which is the difference between a real park and a decorative one.
            module.to("meta")
            asset.parked = True
            self._touch(name)
            logger.info("parked audio asset %s (%.1f MiB freed)",
                        name, freed / (1 << 20))
            return freed

    def restore(self, name: str) -> float:
        """Bring one module back to its device. Returns the milliseconds taken.

        The measured figure replaces the registered estimate, so the register
        prices later victim choices on an observed number.
        """
        with self._lock:
            asset = self.get(name)
            if not asset.parked:
                return 0.0
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
                {k: v.to(target) for k, v in asset._cpu_state.items()},
                strict=True,
            )
            asset._cpu_state = {}
            asset.parked = False
            elapsed_ms = (self._clock() - started) * 1000.0
            asset.measured_restore_ms = elapsed_ms
            self._touch(name)
            logger.info("restored audio asset %s in %.1f ms", name, elapsed_ms)
            del torch
            return elapsed_ms

    def ensure_resident(self, *names: str) -> Dict[str, float]:
        """Restore whichever of ``names`` are parked. Returns per-name ms."""
        out: Dict[str, float] = {}
        for name in names or self.names():
            asset = self.get(name)
            if asset.parked:
                out[name] = self.restore(name)
        return out

    def park_all(self) -> int:
        return sum(self.park(name) for name in self.names())

    def _touch(self, name: str) -> None:
        try:
            from sglang.srt.model_executor.offload_register import maybe_touch_item

            maybe_touch_item(self._item_id(name))
        except Exception:  # pragma: no cover - environment dependent
            pass

    # -- reporting ----------------------------------------------------------

    def to_json(self) -> Dict[str, object]:
        with self._lock:
            return {
                "tenant_id": self.tenant_id,
                "offload_class": OFFLOAD_CLASS,
                "registered_with_runtime": self.registered_with_runtime,
                "assets": [
                    {
                        "name": a.name,
                        "mib": round(a.size_bytes() / (1 << 20), 2),
                        "parked": a.parked,
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
