# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Re-export of the shared NVML device identity module.

The implementation moved to :mod:`sglang.srt.registry.nvml` in #333-M1.
Physical GPU identity is rig-wide infrastructure, not a Class-3 concern: the
registry, the ledger and every tenant have to agree on what "this card" means,
and they can only do that against one module.

The documented CLI keeps working from either name::

    python -m sglang.srt.registry.nvml --json
    python -m sglang.srt.video_enhance.nvml --json
"""

from __future__ import annotations

import sys

from sglang.srt.registry.nvml import (
    MIB,
    DeviceInfo,
    DeviceNotFoundError,
    DeviceOrderUnresolvedError,
    NvmlUnavailableError,
    _main,
    current_device_uuid,
    device_by_uuid,
    is_available,
    list_devices,
    memory_info_for_uuid,
    nvml_session,
    process_bytes_on_uuid,
    resolve_devices_by_name_fragment,
    resolve_index_by_name_fragment,
    total_bytes_for_uuid,
)

__all__ = [
    "MIB",
    "DeviceInfo",
    "DeviceNotFoundError",
    "DeviceOrderUnresolvedError",
    "NvmlUnavailableError",
    "current_device_uuid",
    "device_by_uuid",
    "is_available",
    "list_devices",
    "memory_info_for_uuid",
    "nvml_session",
    "process_bytes_on_uuid",
    "resolve_devices_by_name_fragment",
    "resolve_index_by_name_fragment",
    "total_bytes_for_uuid",
]


if __name__ == "__main__":  # pragma: no cover - CLI passthrough
    sys.exit(_main())
