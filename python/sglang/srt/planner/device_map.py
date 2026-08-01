# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DEPRECATED shell (#397): the planner's view of the ONE card-identity map.

Card identity has a single source of truth,
:class:`sglang.srt.registry.nvml.IdentityMap` (#331): cards are keyed on
their NVML UUID and the three enumerations that disagree on a mixed rig --
NVML/PCI index, CUDA ordinal, PCI BDF -- are derived views of it. New code
uses that class directly. This module remains only as the planner-shaped
adapter over it, so the dashboard payloads (``{nvml_index, cuda_index, name,
total_mib, uuid}``) keep their form.

Two DIFFERENT index spaces name the same physical cards, and the planner
speaks both:

  * CUDA / rank space -- the space the ENGINE flags live in. ``--rank-gpu-id``
    and ``--base-gpu-id`` values end up in ``CUDA_VISIBLE_DEVICES``
    (``utils/common.py maybe_reindex_device_id``), which is interpreted in
    CUDA enumeration order. With no ``CUDA_DEVICE_ORDER`` set (our launches)
    that is FASTEST_FIRST: the fastest card is ``cuda:0``.
  * NVML / physical space -- what pynvml / nvidia-smi enumerate, in PCI bus
    order. All telemetry (power, temperature, memory) is sampled here.

On a mixed rig the two DIVERGE (the reference box: cuda:0 = RTX 5090 =
nvml:1; the two RTX 3080s are cuda:1 = nvml:0 and cuda:2 = nvml:2).

What #397 removed: this module used to carry its own second resolver, whose
fallback was a documented FASTEST_FIRST EMULATION -- a stable sort of the
cards by their SEED_CARDS fp16 GEMM peak. It was labelled "heuristic" and
callers were asked to surface that label, but a label does not make an
ordering right: an unknown card ranked 0.0 and silently kept NVML order, so
the emulation answered confidently on exactly the rigs nobody had measured.
There is now no fallback. Either the identity map places every card, or
:class:`~sglang.srt.registry.nvml.DeviceOrderUnresolvedError` names the cards
it could not place and why, and the caller shows no CUDA index at all.

``build_device_map`` raises that error; :func:`device_map`, whose callers are
UI paths that must never crash, catches it, logs it once and caches an EMPTY
map (``source=None``). An empty map is the honest answer -- no bridge -- and
every caller already renders it as "unbridged".
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "DeviceMapEntry",
    "DeviceMap",
    "build_device_map",
    "device_map",
    "norm_uuid",
    "IDENTITY_MAP_SOURCE",
]

#: ``DeviceMap.source`` when the #331 identity map placed every card. The only
#: non-``None`` value there is since #397; "torch" and "heuristic" are gone.
IDENTITY_MAP_SOURCE = "identity-map"


def norm_uuid(u) -> str:
    """Normalize any GPU-UUID spelling ('GPU-xxxx-...', bytes, uuid.UUID) to
    plain lowercase hex, so torch / NVML / nvidia-smi spellings compare."""
    s = u.decode() if isinstance(u, bytes) else str(u)
    return "".join(ch for ch in s.lower() if ch in "0123456789abcdef")


@dataclasses.dataclass(frozen=True)
class DeviceMapEntry:
    """One physical card under BOTH index spaces."""

    nvml_index: int
    cuda_index: int
    name: str
    total_mib: int
    #: normalized lowercase hex (see :func:`norm_uuid`); "" when unreadable.
    uuid: str

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DeviceMap:
    entries: tuple
    #: :data:`IDENTITY_MAP_SOURCE` when every card was placed, else None (no
    #: GPU inventory, or an order that could not be resolved -- there is no
    #: third, guessed state since #397).
    source: Optional[str]

    def nvml_to_cuda(self) -> Dict[int, int]:
        return {e.nvml_index: e.cuda_index for e in self.entries}

    def cuda_to_nvml(self) -> Dict[int, int]:
        return {e.cuda_index: e.nvml_index for e in self.entries}

    def cuda_for_uuid(self, uuid) -> Optional[int]:
        u = norm_uuid(uuid) if uuid else ""
        if not u:
            return None
        for e in self.entries:
            if e.uuid == u:
                return e.cuda_index
        return None

    def cuda_for_nvml(self, nvml_index: int) -> Optional[int]:
        for e in self.entries:
            if e.nvml_index == nvml_index:
                return e.cuda_index
        return None

    def to_json(self) -> dict:
        return {
            "source": self.source,
            "entries": [e.to_json() for e in self.entries],
        }


def _device_infos(nvml=None):
    """The NVML card list as :class:`registry.nvml.DeviceInfo` records.

    ``nvml`` is an injected pynvml-alike (tests, and ``live_metrics``, which
    already holds an open handle and owns its init/shutdown). Without it the
    registry reads the driver itself, so the real path has exactly one NVML
    reader. Returns ``[]`` when nothing is readable -- a GPU-less host is not
    an error here, it is a rig with no cards, and the caller renders it as
    such.
    """
    from sglang.srt.registry.nvml import DeviceInfo

    if nvml is None:
        try:
            from sglang.srt.registry.nvml import list_devices

            return list_devices()
        except Exception as exc:  # noqa: BLE001 - GPU-less host, not an error
            logger.debug("device map: NVML inventory unavailable: %s", exc)
            return []

    def _text(v) -> str:
        return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)

    try:
        out = []
        for i in range(nvml.nvmlDeviceGetCount()):
            h = nvml.nvmlDeviceGetHandleByIndex(i)
            try:
                uuid = _text(nvml.nvmlDeviceGetUUID(h))
            except Exception:
                uuid = ""
            try:
                total = int(nvml.nvmlDeviceGetMemoryInfo(h).total)
            except Exception:
                total = 0
            try:
                bus = _text(nvml.nvmlDeviceGetPciInfo(h).busId)
            except Exception:
                # No BDF means no bridge to the CUDA side for this card. The
                # all-or-nothing check in the identity map turns that into a
                # named error rather than a plausible-looking index.
                bus = ""
            out.append(
                DeviceInfo(
                    index=i,
                    uuid=uuid,
                    name=_text(nvml.nvmlDeviceGetName(h)),
                    total_bytes=total,
                    pci_bus_id=bus,
                )
            )
        return out
    except Exception as exc:  # noqa: BLE001 - broken or absent NVML
        logger.debug("device map: injected NVML inventory unreadable: %s", exc)
        return []


def build_device_map(nvml=None, allow_torch: bool = True) -> DeviceMap:
    """Build a fresh (uncached) :class:`DeviceMap` from the #331 identity map.

    ``nvml`` is an injectable pynvml-alike for tests and for ``live_metrics``,
    which owns its own handle. ``allow_torch=False`` withholds the only source
    of the CUDA side, so no card can be placed and the result is the empty
    map -- where this used to hand back the emulated order.

    Raises :class:`~sglang.srt.registry.nvml.DeviceOrderUnresolvedError` when
    cards are present but their CUDA order cannot be resolved. Callers that
    must not crash use :func:`device_map`.
    """
    from sglang.srt.registry.nvml import identity_map

    infos = _device_infos(nvml)
    if not infos or not allow_torch:
        return DeviceMap(entries=(), source=None)

    imap = identity_map(infos, allow_cuda_init=True)
    imap.require_cuda_ordinals("the planner device map")
    entries = tuple(
        DeviceMapEntry(
            nvml_index=c.nvml_index,
            cuda_index=int(c.cuda_ordinal),
            name=c.name,
            total_mib=c.total_mib,
            uuid=norm_uuid(c.uuid) if c.uuid else "",
        )
        for c in imap
    )
    return DeviceMap(entries=entries, source=IDENTITY_MAP_SOURCE)


_CACHE: Optional[DeviceMap] = None
_LOGGED_UNRESOLVED = False


def device_map(refresh: bool = False) -> DeviceMap:
    """The cached live bridge for THIS host (built once; ``refresh=True``
    rebuilds, e.g. after a driver reload). Never raises.

    An unresolvable order is logged once, in full, and cached as the EMPTY
    map: the UI then shows cards without a CUDA index, which is what "we do
    not know" looks like. It is not cached as a guess.
    """
    global _CACHE, _LOGGED_UNRESOLVED
    if _CACHE is None or refresh:
        try:
            _CACHE = build_device_map()
        except Exception as exc:  # noqa: BLE001 - UI path, must not crash
            if not _LOGGED_UNRESOLVED or refresh:
                logger.warning("planner device map unavailable: %s", exc)
                _LOGGED_UNRESOLVED = True
            _CACHE = DeviceMap(entries=(), source=None)
    return _CACHE
