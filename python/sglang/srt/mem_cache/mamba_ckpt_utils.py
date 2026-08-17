# Copyright 2025 SGLang Team
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
"""Boundary arithmetic for --mamba-checkpoint-interval.

With a fixed checkpoint interval G, every radix-cached mamba/GDN state must
sit at an absolute token position that is a multiple of G, so that the
checkpoint grid — and with it the prefix-resume point of identical requests —
is a pure function of the token history instead of the traffic-dependent
prefill split. These helpers are pure integer arithmetic so the invariants
can be unit-tested on CPU.
"""

from typing import Optional


def floor_to_interval(pos: int, interval: Optional[int]) -> int:
    """Round ``pos`` down to the checkpoint grid (identity when off)."""
    if interval is None:
        return pos
    return pos // interval * interval


def is_on_interval(pos: int, interval: Optional[int]) -> bool:
    """True when ``pos`` is a valid checkpoint position (always, when off)."""
    if interval is None:
        return True
    return pos % interval == 0


def protect_deepest_anchors(
    interval: Optional[int], host_tier_present: bool = False
) -> bool:
    """Should eviction spare the deepest checkpoint anchors of every path?

    ``MambaRadixCache`` protects them (``mamba_radix_cache.py:1092-1105``)
    because "losing the deepest one silently moves the resume point of
    identical requests and re-introduces run-to-run drift". The protection is
    not about capacity, it is about DETERMINISM.

    That premise holds only while an evicted anchor is a LOST anchor, which is
    true of a device-only pool. With a host tier the node stays a valid match
    after its device value is dropped and is loaded back on the next hit
    (``unified_cache_components/mamba_component.py:71-74`` and ``:139-144``),
    so the resume point does not move and there is nothing to protect against.

    Hence the branch, stated once here rather than diverging per lineage:

    * ``host_tier_present=False`` -> protect, exactly as today;
    * ``host_tier_present=True``  -> do not protect; spilling an anchor is
      allowed because it can be matched and reloaded.

    No interval means no grid, so there are no anchors either way -- the same
    ``interval is not None`` test the device-only path already used.
    """
    if interval is None:
        return False
    return not host_tier_present


def mamba_checkpoint_track_target(
    prefix_len: int,
    extend_len: int,
    interval: int,
    chunk_size: int,
) -> Optional[int]:
    """Deepest absolute multiple of ``interval`` whose GDN state can be
    snapshotted within this extend step, or None if no boundary is reachable.

    The prefill kernels expose intermediate states only at
    ``prefix_len + k * chunk_size`` (FLA chunk grid) plus the final position,
    so a target is reachable iff it lies inside ``(prefix_len, end]`` and its
    offset from ``prefix_len`` is chunk-aligned (the final position
    ``end == target`` is chunk-aligned by the same test, since callers
    validate ``interval % chunk_size == 0`` and keep prefill steps ending on
    the interval grid).
    """
    end = prefix_len + extend_len
    target = end // interval * interval
    if target <= prefix_len:
        return None
    if (target - prefix_len) % chunk_size != 0:
        return None
    return target
