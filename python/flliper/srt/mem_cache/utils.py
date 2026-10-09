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
"""Common utilities."""

from typing import Any, Callable, List, Optional, Tuple

from flliper.kernels.ops.kvcache.mla_buffer import (
    get_mla_kv_buffer_kernel as get_mla_kv_buffer_kernel,
)
from flliper.kernels.ops.kvcache.mla_buffer import (
    get_mla_kv_buffer_triton as get_mla_kv_buffer_triton,
)
from flliper.kernels.ops.kvcache.mla_buffer import (
    set_mla_kv_buffer_fp8_quant_kernel as set_mla_kv_buffer_fp8_quant_kernel,
)
from flliper.kernels.ops.kvcache.mla_buffer import (
    set_mla_kv_buffer_kernel as set_mla_kv_buffer_kernel,
)
from flliper.kernels.ops.kvcache.mla_buffer import (
    set_mla_kv_buffer_triton as set_mla_kv_buffer_triton,
)
from flliper.kernels.ops.kvcache.mla_buffer import (
    set_mla_kv_buffer_triton_fp8_quant as set_mla_kv_buffer_triton_fp8_quant,
)
from flliper.kernels.ops.kvcache.mla_buffer import (
    set_mla_kv_scale_buffer_kernel as set_mla_kv_scale_buffer_kernel,
)
from flliper.kernels.ops.kvcache.mla_buffer import (
    set_mla_kv_scale_buffer_triton as set_mla_kv_scale_buffer_triton,
)
from flliper.srt.environ import envs
from flliper.srt.mem_cache.cpp_utils.native_hash import get_native_hash
from flliper.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
)

_EVICTION_POLICY_FACTORIES: dict[str, Callable[[], EvictionStrategy]] = {
    "lru": LRUStrategy,
    "lfu": LFUStrategy,
    "fifo": FIFOStrategy,
    "mru": MRUStrategy,
    "filo": FILOStrategy,
    "priority": PriorityStrategy,
    "slru": SLRUStrategy,
}


def get_eviction_strategy(eviction_policy: str) -> EvictionStrategy:
    policy = eviction_policy.lower()
    try:
        return _EVICTION_POLICY_FACTORIES[policy]()
    except KeyError:
        supported = "', '".join(_EVICTION_POLICY_FACTORIES)
        raise ValueError(
            f"Unknown eviction policy: {policy}. Supported policies: '{supported}'."
        ) from None


def maybe_init_custom_mem_pool(
    device: str,
) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Initialize custom memory pool based on environment variable.

    This function can be modified to support more features that require a custom memory pool.

    Args:
        device: The device to allocate memory on

    Returns:
        Tuple of (enable_custom_mem_pool, custom_mem_pool, custom_mem_pool_type)
    """
    enable_custom_mem_pool = (
        True if envs.FLLIPER_MOONCAKE_CUSTOM_MEM_POOL.get() is not None else False
    )

    if enable_custom_mem_pool:
        # Currently, only mooncake requires a custom mem pool for MNNVL/Barex PD disaggregation
        from flliper.srt.disaggregation.mooncake.utils import (
            init_mooncake_custom_mem_pool,
        )

        return init_mooncake_custom_mem_pool(device)
    else:
        return False, None, None


#: Q-460 SALT-ISOLATION (27B y8r 55c95a89c7, boot 1003_090656): the store page
#: key was a function of the TOKENS alone, so a page written under cache_salt A
#: was read by cache_salt B and by an unsalted request (probe 280/300: B1
#: cached=8187/8963, N1 cached=8192/8964 from the L2 arena, ``HiCache prefetch
#: ... loaded=8153``). The namespace (``RadixKey.extra_key``: cache_salt +
#: extra_key, LoRA id) now seeds the ROOT of every chain: the first page of a
#: namespaced key hashes over this seed instead of no prior, so every page of
#: the chain differs per namespace. ``None`` keeps the old key exactly -- the
#: unsalted store of earlier boots stays readable.
# F0-K (M2): PERSISTED identity, NOT a name -- it seeds the page key of every salted L2/L3 chain; the rename must leave the bytes alone
# (RENAME_PLAN 2 "persisted ids"; kit must-keep rule persisted-id).
_NAMESPACE_SEED_TAG = b"sglang-kv-namespace-v1\0"


def namespace_root_hash(extra_key: Optional[str]) -> Optional[str]:
    """The prior hash a chain of namespace ``extra_key`` starts from (hex),
    None for the default namespace (the unchanged unsalted chain)."""
    if extra_key is None:
        return None
    import hashlib

    return hashlib.sha256(_NAMESPACE_SEED_TAG + str(extra_key).encode()).hexdigest()


def get_hash_str(
    token_ids: List[int],
    prior_hash: Optional[str] = None,
    page_size: Optional[int] = None,
    extra_key: Optional[str] = None,
) -> str | List[str]:
    """Page hash chain of ``token_ids`` after ``prior_hash``.

    Q-460: a chain from the ROOT (no prior) is seeded by its namespace --
    ``extra_key`` when given, else the ``extra_key`` a ``RadixKey`` carries.
    A plain id list without ``extra_key`` is the default namespace."""
    if not prior_hash:
        ns = extra_key if extra_key is not None else getattr(token_ids, "extra_key", None)
        prior_hash = namespace_root_hash(ns)
    prior_digest = bytes.fromhex(prior_hash) if prior_hash else None
    return get_native_hash(token_ids, prior_digest, page_size)


def hash_str_to_int64(hash_str: str) -> int:
    """Convert SHA256 hex string to signed 64-bit integer for events.

    Takes first 16 hex characters (64 bits) and converts to signed int64 range.
    """
    uint64_val = int(hash_str[:16], 16)
    if uint64_val >= 2**63:
        return uint64_val - 2**64
    return uint64_val


def compute_node_hash_values(node: Any, page_size: int) -> List[str]:
    """Compute SHA256-based hash values for position-aware KV block IDs."""
    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]

    hash_values = get_hash_str(node.key, parent_hash, page_size=page_size)
    assert isinstance(hash_values, list)
    return hash_values


def split_node_hash_value(
    child_hash_value: Optional[List[str]], split_len: int, page_size: int
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    """Split hash_value between parent and child nodes during node splitting.

    Args:
        child_hash_value: The hash_value list from the child node being split
        split_len: The length at which to split (in tokens)
        page_size: The page size for calculating number of pages

    Returns:
        Tuple of (new_node_hash_value, updated_child_hash_value)
    """
    if child_hash_value is None:
        return None, None

    if page_size == 1:
        split_pages = split_len
    else:
        split_pages = split_len // page_size

    new_node_hash = child_hash_value[:split_pages]
    child_hash = child_hash_value[split_pages:]

    return new_node_hash, child_hash
