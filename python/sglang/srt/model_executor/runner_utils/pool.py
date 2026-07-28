# Copyright 2023-2026 SGLang Team
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
"""CUDA graph memory pool shared across the prefill and decode graph
backends. The two phases never replay concurrently, so sharing one pool
reserves only the larger phase's capture footprint.

PER LANE, not per process (#274 slice C). Graphs that share a memory pool
share the buffers their intermediate allocations live in: replaying two such
graphs AT THE SAME TIME lets one graph's activations overwrite the other's.
That is exactly what the multi-group runtime does once its lanes stop taking
turns, so the pool is keyed by the lane in scope -- ``None`` for the serving
group, 0..N for the lanes. Each key reserves its own capture footprint; that
extra VRAM is the price of concurrency and is a named post in the lane's
ledger, not a leak.
"""

from __future__ import annotations

from typing import Any, Optional

from sglang.srt.runtime_context import current_lane_id, get_resources


def get_global_graph_memory_pool() -> Optional[Any]:
    return get_resources().graph_memory_pool.get(current_lane_id())


def set_global_graph_memory_pool(val: Any) -> None:
    get_resources().graph_memory_pool[current_lane_id()] = val


def get_or_create_global_graph_memory_pool(device_module: Any) -> Any:
    """Return this lane's graph memory pool, creating it on first use so
    later backends of the SAME lane reuse the same handle."""
    pools = get_resources().graph_memory_pool
    key = current_lane_id()
    if pools.get(key) is None:
        pools[key] = device_module.graph_pool_handle()
    return pools[key]
