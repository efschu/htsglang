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


def is_resume_candidate(
    depth: int,
    interval: Optional[int],
    has_device_value: bool,
    has_host_value: bool = False,
    device_only: bool = True,
) -> bool:
    """Is a node at absolute token position ``depth`` a valid mamba resume
    anchor for the prefix match?

    The compound rule both match walks apply (``mamba_radix_cache.py``'s
    ``_match_prefix_helper`` and the unified component's match validators):
    the node must CARRY a state, and with ``--mamba-checkpoint-interval`` set
    it must SIT on the grid -- off-grid checkpoints (legacy entries, storage
    written by a non-interval run, unaligned edge paths) would re-introduce
    traffic-dependent resume points.

    ``device_only=False`` additionally accepts a host-backed state
    (``has_host_value``): under HiCache an evicted-but-backuped anchor is a
    valid match that triggers ``load_back``. The grid applies to those
    equally -- surviving on the host tier does not legalise an off-grid
    position.

    ``interval=None`` degenerates to the pure presence test, byte-identical
    to the pre-#747 behaviour of both walks.
    """
    present = has_device_value or (not device_only and has_host_value)
    return present and is_on_interval(depth, interval)


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


def checkpoint_truncation_align(
    existing_align: Optional[int],
    interval: Optional[int],
    chunked_prefill_size: Optional[int],
) -> tuple:
    """``(truncation_align_size, folded)`` after the checkpoint grid weighs in.

    #750: the interval folds into the prefill truncation alignment ONLY
    while ``interval <= chunked_prefill_size`` (or chunked prefill is off).
    There the scheduler clips every step end onto the grid, which is what
    makes every cached snapshot position an absolute interval multiple.

    A SPARSE grid -- ``interval > chunked_prefill_size``, validated to be an
    exact multiple -- needs no clipping at all: full chunk ends land at
    ``n * chunked_prefill_size``, and every ``(interval // chunk)``-th one
    is ON the grid by divisibility, while the retention rule
    (``mamba_radix_cache.py``: an off-grid finish is not cached) drops the
    ends between. Folding 8192 into the alignment would inflate the
    truncation unit 16x past a 512 chunk budget, which is exactly the C30
    refusal that made the old coupling a hard limit.

    ``folded`` tells the caller whether the interval became part of the
    alignment (and so belongs in the C30 sources list).
    """
    if interval is None:
        return existing_align, False
    if (
        chunked_prefill_size is not None
        and chunked_prefill_size > 0
        and interval > chunked_prefill_size
    ):
        return existing_align, False
    if existing_align is None:
        return interval, True
    import math

    return math.lcm(existing_align, interval), True


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
    offset from ``prefix_len`` is chunk-aligned. Callers validate
    ``interval % chunk_size == 0``; prefill steps end on the interval grid
    while ``interval <= chunked_prefill_size`` (the fold arm of
    ``checkpoint_truncation_align``), and on a #750 sparse grid
    (``interval`` a multiple of the chunk budget) a boundary inside a full
    step can only be the step's own end, so the final-position routing
    serves every anchor.
    """
    end = prefix_len + extend_len
    target = end // interval * interval
    if target <= prefix_len:
        return None
    if (target - prefix_len) % chunk_size != 0:
        return None
    return target
