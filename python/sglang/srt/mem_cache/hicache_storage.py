from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional, Set

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.canonical_kv_page import CanonicalPageError

if TYPE_CHECKING:
    from sglang.srt.mem_cache.canonical_page_store import (
        CanonicalExtentWindow,
        CanonicalPageWindow,
    )
    from sglang.srt.mem_cache.pool_host import HostKVCache

logger = logging.getLogger(__name__)

# Max pages per batched storage IO call.
STORAGE_BATCH_SIZE = 128


def compute_model_identity_hash(
    server_args: Any, *, include_parallel_vectors: bool = True
) -> str:
    """Compute a short hash that uniquely identifies the model and KV layout.

    Storage page hashes cover token ids only, and the storage key suffix covers
    served_model_name plus parallel geometry. Neither includes the model
    identity (weights revision) or the KV byte format (dtype, quantization,
    kv_cache_dtype). Entries in a persistent storage tier outlive the server
    process, so a later run that shares the served_model_name and storage
    location but differs in e.g. --kv-cache-dtype would silently read pages
    written in another byte format. Incorporating this hash into the key
    suffix turns that silent wrong hit into a clean miss.

    Matches the recipe of upstream PR #24794 so keys converge with stock
    sglang once it lands; the two uneven-TP vectors below are fork-only and
    enter the string only when they are set, so an even-TP key is
    byte-identical to the upstream recipe.
    """
    # Every part is str()-coerced. ``dtype`` and ``kv_cache_dtype`` always
    # were; ``revision`` and ``quantization`` were not, which left the
    # function able to raise TypeError in "|".join for any server_args whose
    # fields are not already strings. That was latent while the only callers
    # ran late with a fully-resolved ServerArgs; #631a added a call on the PD
    # REGISTRATION path, where it became reachable. Coercion is byte-identical
    # for real inputs -- str() of a str is itself, and None still becomes ""
    # through the `or` -- so no persisted HiCache key moves.
    identity_parts = [
        os.path.normpath(str(server_args.model_path)) if server_args.model_path else "",
        str(server_args.revision or ""),
        str(server_args.dtype or "auto").lower(),
        str(server_args.quantization or ""),
        str(server_args.kv_cache_dtype or "auto").lower(),
    ]
    # #513 (audit #506, finding A3-2): under this fork's uneven TP,
    # tp_rank/tp_size in the key suffix do NOT determine a rank's kv-head
    # count -- `--rank-tp-ratio 13,6,6` and an even split are both
    # (tp_rank=0, tp_size=3) with different bytes per stored page, and page
    # hashes cover token ids only. The sibling fingerprint in
    # managers/kv_session_spill_destination.py already treats these vectors as
    # key-relevant for exactly this reason.
    #
    # APPENDED ONLY WHEN SET, so an even-TP deployment keeps the pages it has
    # already persisted: re-keying every rig to fix a case that cannot occur
    # there would be a cost with no benefit. Same convention
    # uneven_perf.measured_kv_budget_fingerprint_fields uses for pp_size.
    #
    # ``include_parallel_vectors=False`` drops this tail, and the distinction
    # is the whole reason the flag exists (#631a guard 1). These vectors
    # belong in a STORAGE KEY, where a page's bytes depend on the writing
    # rank's kv-head count, so two differently-split servers must not read
    # each other's pages. They do NOT belong in a PD HANDSHAKE, which asks a
    # different question -- "are these two servers serving the same model?" --
    # and where differing parallelism between the arms is EXPECTED and
    # supported: TransportIdentity.COMPARED deliberately omits tp_size and
    # pp_size, and the token-axis difference is handled by ``owned_ordinals``
    # (disaggregation/base/conn.py). Comparing the vectors there would refuse
    # a Route A pair (PP prefill + TP decode) that the engine transfers
    # correctly. Same recipe, one function, two honest questions.
    if include_parallel_vectors:
        for name in ("rank_tp_ratio", "rank_kv_ratio"):
            value = getattr(server_args, name, None)
            if value:
                identity_parts.append(f"{name}={value}")
    identity_str = "|".join(identity_parts)
    return hashlib.sha256(identity_str.encode()).hexdigest()[:16]


@dataclass
class HiCacheStorageConfig:
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    attn_cp_rank: int
    attn_cp_size: int
    is_mla_model: bool
    enable_storage_metrics: bool
    is_page_first_layout: bool
    model_name: Optional[str]
    # Hash over (model_path, revision, dtype, quantization, kv_cache_dtype),
    # see compute_model_identity_hash(). None keeps legacy key layout.
    model_identity_hash: Optional[str] = None
    tp_lcm_size: Optional[int] = None
    should_split_heads: bool = False
    extra_config: Optional[dict] = None
    # #810: `--hicache-host-role`. Carried here rather than re-derived because
    # a backend has to know whether it is the RETENTION tier: under 'staging'
    # the pinned host tier in front of it is deliberately small, so an
    # unbounded backend is no longer a cache that merely grows -- it is the
    # only copy, growing without a bound. Defaulted, so every other
    # construction site and the whole retention path are unchanged.
    host_role: str = "retention"
    # Weighted uneven-DCP owner mode (task #60): KV pages are token-sharded
    # with FULL replicated kv-heads, so a KV page's bytes are complete on its
    # owner rank and rank-independent. KV page keys drop the _{tp_rank}_{tp_size}
    # suffix (one shared file per page, written only by its backup-time owner,
    # readable by every rank), while component pools (mamba/SWA: genuinely
    # per-rank shards) keep the rank suffix.
    dcp_owner_mode: bool = False
    # #706 whole-page protocol: this rank's slot window in the canonical page
    # (mem_cache/canonical_page_store.py). Set only when the geometry-neutral
    # format is active; None keeps every key and every byte exactly as before.
    #
    # When set, a KV page is full width (every attention layer of one token) and
    # each stage deposits its own slots at their GLOBAL offset, so the bytes stop
    # depending on the PP cut and the key drops the _{pp_size}_{pp_rank} suffix
    # as well -- the same argument dcp_owner_mode above already made on the token
    # axis. The key then carries content alone: model identity and token hash.
    canonical_kv_page: Optional[CanonicalPageWindow] = None
    # #706 slice 2: this rank's window in the canonical {hash}.mamba blob, on
    # the same protocol. Required whenever the model HAS GDN/mamba layers and
    # the canonical page is active, because a KV-only prefix is worth nothing:
    # batch_exists_v2 takes the MINIMUM across pools and the mamba pool is
    # registered TRAILING_PAGES, so a missing blob truncates the whole KV
    # prefix to zero (test_mamba_gates_the_hit_706.py), and the device-side
    # MambaRadixCache match advances only at nodes that carry mamba state.
    canonical_mamba_blob: Optional[CanonicalExtentWindow] = None


@dataclass
class HiCacheStorageExtraInfo:
    prefix_keys: Optional[List[str]] = None
    extra_info: Optional[dict] = None


@dataclass(frozen=True)
class PrefetchTimeoutConfig:
    """Knobs for the linear prefetch-timeout policy used by HiCache."""

    base: float = 2.0  # seconds, fixed overhead unrelated to token count
    per_ki_token: float = 0.1  # seconds per 1024 tokens
    max: float = 30.0  # seconds, upper bound for the linear timeout


class PoolName(str, Enum):
    """Well-known pool names used as PoolTransfer/PoolEntry identifiers."""

    KV = "kv"
    MAMBA = "mamba"
    SWA = "swa"
    INDEXER = "indexer"
    # TODO(hzh0425): Current DeepSeek V4 pool naming is verbose; will be normalized to
    # 'COMPRESSED_KV / COMPRESSED_INDEXER / COMPRESSED_STATE' in the next PR.
    DEEPSEEK_V4_C4 = "deepseek_v4_c4"
    DEEPSEEK_V4_C4_INDEXER = "deepseek_v4_c4_indexer"
    DEEPSEEK_V4_C128 = "deepseek_v4_c128"
    DEEPSEEK_V4_C4_STATE = "deepseek_v4_c4_state"
    DEEPSEEK_V4_C4_INDEXER_STATE = "deepseek_v4_c4_indexer_state"
    DEEPSEEK_V4_C128_STATE = "deepseek_v4_c128_state"

    # Draft KV pool
    DRAFT = "draft"

    def __str__(self) -> str:
        return self.value


class PoolHitPolicy(str, Enum):
    """Hit policy for batch_exists_v2 per-pool prefix matching.

    ALL_PAGES      : every page in [0, kv_hit) must exist (e.g. DSA).
    TRAILING_PAGES : only the last N pages must exist (e.g. Mamba/SWA states).
    """

    ALL_PAGES = "all_pages"
    TRAILING_PAGES = "trailing_pages"


@dataclass
class PoolTransfer:
    """Unified per-pool transfer descriptor for batch v2 interface.

    device<->host path : host_indices + device_indices
    host<->storage path: host_indices + keys
    nodes_to_load      : evicted nodes this transfer covers
    """

    name: PoolName
    host_indices: Optional[torch.Tensor] = None
    device_indices: Optional[torch.Tensor] = None
    keys: Optional[List[str]] = None
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES
    nodes_to_load: Optional[List[Any]] = None
    indices_from_pool: Optional[PoolName] = None


@dataclass(frozen=True)
class SidecarPoolSpec:
    """Pool whose transfer indices are reused from one real source pool."""

    pool_name: PoolName
    indices_from_pool: PoolName
    hit_policy: PoolHitPolicy = PoolHitPolicy.ALL_PAGES


@dataclass
class PoolTransferResult:
    """Tracks how many pages were successfully processed per pool."""

    kv_hit_pages: int
    extra_pool_hit_pages: dict[str, int]

    @classmethod
    def empty(cls) -> PoolTransferResult:
        return cls(0, {})

    def update_kv_hit_pages(self, kv_hit_pages: int) -> None:
        """Accumulate kv_hit_pages across batches (max = last successful batch)."""
        self.kv_hit_pages = max(self.kv_hit_pages, kv_hit_pages)

    def update_extra_pool_hit_pages(self, results: dict[str, List[bool]]) -> None:
        """Record actual load/write success counts per extra pool."""
        self.extra_pool_hit_pages.update(
            {name: sum(rs) for name, rs in results.items()}
        )


class HiCacheStorage(ABC):
    """
    HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    """

    # todo, the page size of storage backend does not have to be the same as the same as host memory pool
    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        self.mem_pool_host = mem_pool_host

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        if not hasattr(self, "registered_pools"):
            self.registered_pools = {}
        self.registered_pools[host_pool_name] = host_pool

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        """Check which cache pages exist in storage, respecting per-pool hit policies.

        Longest-prefix semantics
        Extra-pool hit policies (``PoolTransfer.hit_policy``)
        ------------------------------------------------------
        Each ``PoolTransfer`` in ``pool_transfers`` describes a secondary
        cache pool (e.g. Mamba SSM states) that must be co-present with the
        KV pages.  The final ``final_pages`` is the minimum across all pools,
        so a missing auxiliary page shrinks the usable prefix.

        - ``"all_pages"`` (default):  every page in [0, kv_hit) must exist
          for this pool.  Used for pools that are required for every token
          in the prefix (e.g. DeepSeek DSA pool).

        - ``"trailing_pages"``:  only the *last* ``len(transfer.keys)`` pages
          of the KV prefix need to exist.  Used for pools whose data covers
          only the tail of a prefix (e.g. Mamba/SWA Pool).

        Returns
        -------
        PoolTransferResult
            ``kv_hit_pages`` = length of the usable KV prefix.
            ``extra_pool_hit_pages`` maps each pool name to the number of pages
            that were found.
        """
        raise NotImplementedError()

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        """Read data from storage into host memory for each PoolTransfer.

        Returns a dict mapping pool name to a per-entry success list.
        """
        raise NotImplementedError()

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        """Write data from host memory to storage for each PoolTransfer.

        Returns a dict mapping pool name to a per-entry success list.
        """
        raise NotImplementedError()

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Retrieve values for multiple keys.
        Returns a list of booleans indicating success for each key.
        """
        pass

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Store multiple key-value pairs.
        Returns a list of booleans indicating success for each key.
        """
        pass

    @abstractmethod
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        pass

    # TODO: Use a finer-grained return type (e.g., List[bool])
    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        for i in range(len(keys)):
            if not self.exists(keys[i]):
                return i
        return len(keys)

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None

    def capacity_stats(self) -> Optional[dict]:
        """Current capacity limits and usage, or None if this backend has none.

        Only backends that do their own on-disk capacity accounting (today:
        ``file``) report here; backends whose capacity lives in an external
        service return None.
        """
        return None

    def check_disk_space(self, force: bool = False) -> bool:
        """Periodic capacity watchdog; False means the backend stopped writing.

        Backends that own local storage override this (today: ``file``).
        Backends without a local capacity of their own are always writable.
        """
        return True

    def resize(
        self,
        *,
        max_size_bytes: Optional[int] = None,
        min_free_bytes: Optional[int] = None,
    ) -> Optional[dict]:
        """Change the capacity limits at runtime; ``None`` leaves one unchanged.

        Returns post-resize ``capacity_stats`` (plus ``freed_bytes``) on
        success, or None if this backend cannot be resized in place.
        """
        return None


class MetadataCache:
    def __init__(self, ttl_seconds: float):
        self.ttl_seconds = ttl_seconds
        # key -> monotonic timestamp
        self.cache: dict[str, float] = {}
        self.lock = threading.Lock()

    def add(self, key: str):
        with self.lock:
            if key not in self.cache:
                self.cache[key] = time.monotonic()

    def remove(self, key: str):
        with self.lock:
            self.cache.pop(key, None)

    def contains(self, key: str) -> bool:
        with self.lock:
            if key not in self.cache:
                return False
            if self.ttl_seconds == -1.0:
                return True
            if time.monotonic() - self.cache[key] > self.ttl_seconds:
                del self.cache[key]
                return False
            return True

    def clear(self):
        with self.lock:
            self.cache.clear()


# Shard directories the page files spread over: the first two hex characters
# of the key. Page keys are sha256 hex digests, so the first byte spreads
# uniformly over 256 subdirectories. Stems that do not start with two lowercase
# hex characters -- only ever synthetic keys -- share one shard.
_SHARD_HEX = "0123456789abcdef"
_SHARD_FALLBACK = "zz"


def page_shard(stem: str) -> str:
    """Shard subdirectory a page file with this key stem belongs in.

    Everything that touches the on-disk store must agree on this rule: the
    backend, and the offline geometry migration in ``hicache_migrate``. Before
    sharding, every page landed directly in the storage directory -- the
    incident of task #558 left 11.7 million entries in one flat directory,
    which cost ~114 s to scan at startup and turned every existence sweep into
    a full-directory walk.
    """
    prefix = stem[:2]
    if len(prefix) == 2 and all(c in _SHARD_HEX for c in prefix):
        return prefix
    return _SHARD_FALLBACK


class MixedLayoutError(RuntimeError):
    """The same page stem exists in BOTH the flat and the sharded layout."""


class MixedGenerationError(MixedLayoutError):
    """One store holds component blobs written by TWO key GENERATIONS.

    Subclasses ``MixedLayoutError`` on purpose (#558's mechanic, extended one
    axis) rather than introducing a parallel refusal: both say the same thing
    -- one content-addressed key resolves to more than one candidate file, and
    the read path would silently prefer one of them. #558 is the FILE-LAYOUT
    axis (flat vs sharded, same key). This is the KEY-FORMAT axis (the retired
    per-stage suffix vs the #706 geometry-neutral suffix, same pool).

    Anything already catching MixedLayoutError therefore keeps working.
    """


def audit_blob_generations(
    root_dir: str,
    *,
    stage_marker: str,
    canonical_marker: str,
    limit: int = 4,
    max_files: int = 200_000,
) -> tuple:
    """Sample the store for component blobs of BOTH key generations.

    THE HAZARD THIS NAMES (user order 2026-08-29, the retired second HiCache
    implementation): the per-stage component writer of that era keys a GDN blob
    ``{hash}.mamba{model}_{identity}_{tp_rank}_{tp_size}_{pp_size}_{pp_rank}``
    -- the ``_0_1_3_r`` tail -- while the #706 canonical writer keys the SAME
    pool ``{hash}.mamba{model}_{identity}``, geometry-neutral. Both writers ran
    against the same directory for weeks. Measured 2026-08-29 in the specimen
    store ``/tmp/hicache_783``: 328 canonical ``.mamba`` blobs beside 1091
    per-stage ones (``_0_1_3_0`` / ``_0_1_3_1`` / ``_0_1_3_2``). Those bytes
    describe different cuts of the state under one content-addressed hash.

    DRAFT BLOBS ARE NOT A SECOND GENERATION and must never be counted here:
    ``_is_shared_kv_key``'s docstring states the rule -- draft KV is
    head-SHARDED and token-COMPLETE, so it keeps the geometry suffix BY
    DESIGN, in every generation. Only the pool whose key MOVED between
    generations can be ambiguous, and that is the one the caller names.

    Returns ``(stage_samples, canonical_samples, files_seen, exhausted)``.
    This is a bounded DETECTOR, not a proof of coherence: it stops at
    ``max_files`` and ``exhausted`` reports whether the whole store was seen,
    so a caller never reads a clean result as more than it is (indicator law).
    """
    stage: list = []
    canonical: list = []
    seen = 0
    exhausted = True

    def _consider(name: str) -> bool:
        """False when the budget is spent."""
        nonlocal seen
        if not name.endswith(".bin"):
            return True
        seen += 1
        stem = name[:-4]
        # The stage marker EXTENDS the canonical one (same pool + model +
        # identity, plus the geometry tail), so it is the more specific of the
        # two and is tested first. On today's suffixes the two cannot both
        # match one stem; the order costs nothing and holds if a future suffix
        # rule ever makes them overlap.
        if stem.endswith(stage_marker):
            if len(stage) < limit:
                stage.append(stem)
        elif stem.endswith(canonical_marker):
            if len(canonical) < limit:
                canonical.append(stem)
        if stage and canonical:
            # Both generations found: the refusal is already decided.
            return False
        return seen < max_files

    try:
        with os.scandir(root_dir) as it:
            top = list(it)
    except (FileNotFoundError, NotADirectoryError):
        return ((), (), 0, True)
    for entry in top:
        if entry.is_dir():
            try:
                with os.scandir(entry.path) as shard_it:
                    for sub in shard_it:
                        if not _consider(sub.name):
                            # Early exit: either decided, or out of budget.
                            # Either way the whole store was NOT examined.
                            return (tuple(stage), tuple(canonical), seen, False)
            except OSError:
                continue
        elif not _consider(entry.name):
            return (tuple(stage), tuple(canonical), seen, False)
    return (tuple(stage), tuple(canonical), seen, exhausted)


def audit_layout(root_dir: str, *, limit: int = 8) -> list:
    """Stems present in BOTH layouts. Empty list when the store is coherent.

    Sharding (#558) is a READ-THROUGH migration: new writes are sharded, old
    flat files keep serving hits, and nothing moves. That is safe exactly while
    a stem lives in one place or the other. If a stem exists in both,
    ``_existing_path`` silently prefers the sharded one -- and the two files can
    differ, because the flat one was written under whatever geometry and format
    the store had at the time. A silent preference between two candidate pages
    for one content-addressed key is the failure this refuses.

    Only the top-level ``.bin`` files are enumerated (the legacy layout), and
    each is checked against its shard. A store that never had a flat layout
    costs one scandir of a directory holding only shard directories.
    """
    duplicates = []
    try:
        with os.scandir(root_dir) as it:
            entries = list(it)
    except FileNotFoundError:
        return duplicates
    for entry in entries:
        if entry.is_dir() or not entry.name.endswith(".bin"):
            continue
        stem = entry.name[:-4]
        sharded = os.path.join(root_dir, page_shard(stem), entry.name)
        if os.path.exists(sharded):
            duplicates.append(stem)
            if len(duplicates) >= limit:
                break
    return duplicates


class HiCacheFile(HiCacheStorage):

    def __init__(
        self, storage_config: HiCacheStorageConfig, file_path: str = "/tmp/hicache"
    ):
        self.file_path = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get() or file_path

        tp_rank, tp_size, pp_rank, pp_size, model_name, is_mla_model = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.pp_rank,
            storage_config.pp_size,
            storage_config.model_name,
            storage_config.is_mla_model,
        )
        attn_cp_rank = storage_config.attn_cp_rank
        attn_cp_size = storage_config.attn_cp_size
        model_name = "-".join(model_name.split("/")) if model_name else ""
        enable_pp = pp_size > 1
        self.dcp_owner_mode = bool(getattr(storage_config, "dcp_owner_mode", False))
        # #706: this rank's window in the canonical (full-width) page. None on
        # every default path, and the ONLY thing that moves a key.
        self.canonical_kv_page = getattr(storage_config, "canonical_kv_page", None)
        self.canonical_mamba_blob = getattr(
            storage_config, "canonical_mamba_blob", None
        )
        # Precomputed once: the KV page's generic (one-extent) form, so the hot
        # path does not rebuild and revalidate it per page.
        self._canonical_kv_extents = (
            self.canonical_kv_page.as_extents()
            if self.canonical_kv_page is not None
            else None
        )
        if self.canonical_mamba_blob is not None and self.canonical_kv_page is None:
            raise NotImplementedError(
                "The #706 canonical mamba blob was configured without the "
                "canonical KV page. The two travel together: a neutral GDN blob "
                "beside pp-suffixed KV pages still misses across the flip."
            )
        if self.canonical_kv_page is not None and attn_cp_size > 1:
            raise NotImplementedError(
                "The #706 canonical KV page and NSA context parallel both claim "
                "the page: each CP rank holds a disjoint slice of every page, "
                "which is a THIRD sharding axis the 16-slot layer format does "
                "not describe. Refusing rather than writing pages whose key no "
                "longer names their bytes."
            )
        # Model identity hash keeps runs that share a served_model_name but
        # differ in weights or KV byte format (dtype/quantization/
        # kv_cache_dtype) from hitting each other's persisted pages. Old-layout
        # keys (written without the hash) simply no longer match: clean miss
        # instead of a silent wrong-format hit.
        identity_hash = storage_config.model_identity_hash or ""
        # #969F: ONE DERIVATION MOMENT. These inputs are kept so the suffixes
        # can be RE-DERIVED when `install_canonical_windows` later changes the
        # one fact they depend on. See `_derive_key_suffixes`.
        self._key_geom = dict(
            model_name=model_name,
            identity_hash=identity_hash,
            is_mla_model=is_mla_model,
            tp_rank=tp_rank,
            tp_size=tp_size,
            enable_pp=enable_pp,
            pp_size=pp_size,
            pp_rank=pp_rank,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=attn_cp_size,
        )
        self._derive_key_suffixes()

        # Shard directories created so far (see page_shard): keeps the write path
        # to one makedirs per shard instead of one per page.
        self._known_shards: set[str] = set()

        # #410: the pin ledger, loaded before anything can evict or sweep. It
        # is durable because a checkpoint outlives the process, so a pin that
        # only lived in memory would silently stop protecting anything at the
        # next restart.
        from sglang.srt.mem_cache.pin_ledger import PinLedger

        self.pins = PinLedger(
            self.file_path,
            budget_bytes=int(envs.SGLANG_HICACHE_PIN_BUDGET_BYTES.get() or 0),
        )
        self.pins.load()

        # #558: a stem in BOTH layouts means two candidate pages for one
        # content-addressed key, and the read path would silently prefer one.
        # Checked once, at attach, and refused loudly rather than resolved.
        duplicates = audit_layout(self.file_path)
        if duplicates:
            raise MixedLayoutError(
                f"HiCacheFile store {self.file_path!r} holds the same page in "
                f"both the flat and the sharded layout: {duplicates}. The read "
                "path prefers the sharded copy, but the two files can differ -- "
                "the flat one predates sharding and may predate the current key "
                "format entirely. Resolve it deliberately (delete the legacy "
                "copies, or run the offline migration) rather than letting a "
                "content-addressed key resolve to whichever file the lookup "
                "order happens to find first."
            )

        # THE RETIRED SECOND HiCACHE IMPLEMENTATION MUST NOT SHARE A STORE WITH
        # THE ONE THAT REPLACED IT (user order 2026-08-29). Same law as #558
        # directly above, one axis over: there the same key lived in two
        # LAYOUTS, here the same pool is keyed by two FORMATS. The per-stage
        # component writer of the retired era appends this rank's geometry
        # (`_{tp_rank}_{tp_size}_{pp_size}_{pp_rank}`, the `_0_1_3_r` tail);
        # the #706 writer that replaced it keys the same pool geometry-neutral.
        # Bytes under those two keys are different CUTS of the same state, and
        # the store cannot tell a reader which cut it just handed over.
        #
        # Checked once, at attach, and only where it can actually be ambiguous:
        # the pool must have MOVED between generations, which is exactly the
        # condition `config_suffix != kv_config_suffix`. Draft blobs keep the
        # geometry suffix in BOTH generations by design (see
        # `_is_shared_kv_key`) and are therefore never a generation signal.
        if self.config_suffix != self.kv_config_suffix:
            stage_blobs, canonical_blobs, files_seen, exhausted = (
                audit_blob_generations(
                    self.file_path,
                    stage_marker=f".{PoolName.MAMBA}{self.config_suffix}",
                    canonical_marker=f".{PoolName.MAMBA}{self.kv_config_suffix}",
                )
            )
            if stage_blobs and canonical_blobs:
                raise MixedGenerationError(
                    f"HiCacheFile store {self.file_path!r} holds "
                    f"{PoolName.MAMBA} blobs written by TWO key generations: "
                    f"the retired second HiCache implementation's per-stage "
                    f"form (e.g. {stage_blobs[0]!r}) beside the #706 "
                    f"geometry-neutral form (e.g. {canonical_blobs[0]!r}). "
                    "The two name different cuts of the same state under one "
                    "content-addressed hash, so a hit can return bytes that do "
                    "not describe this rank's layers. The retired writer is "
                    "gone; its deposits are not. Migrate or drop them "
                    "deliberately -- see #975 for the offline cut "
                    "(hicache_migrate.MambaBlobSpec.for_layers / layer_extents) "
                    "-- rather than letting the read path pick whichever file "
                    "the lookup order happens to find first."
                )
            if stage_blobs:
                # Single-generation, but the RETIRED one, and this process
                # writes the new format. Not ambiguous yet, so not a refusal --
                # it becomes one the moment this boot deposits its first
                # canonical blob into the same directory.
                logger.warning(
                    "HiCacheFile store %s holds %d+ %s blob(s) in the retired "
                    "second HiCache implementation's per-stage key format "
                    "(e.g. %s). This process writes the #706 geometry-neutral "
                    "format, so the store becomes two-generation -- and this "
                    "attach a hard MixedGenerationError -- as soon as the "
                    "first canonical blob lands. Clear them or migrate them "
                    "now (#975).",
                    self.file_path,
                    len(stage_blobs),
                    PoolName.MAMBA,
                    stage_blobs[0],
                )
            elif not exhausted:
                logger.info(
                    "HiCacheFile generation audit stopped after %d files in "
                    "%s without seeing the whole store; a clean result here "
                    "bounds the check, it does not prove coherence.",
                    files_seen,
                    self.file_path,
                )

        # #706: orphaned partials are invisible to readers AND untracked by the
        # LRU evictor (it walks .bin only), so nothing else would ever reap a
        # page whose remaining writers never arrived. One sweep at attach, by
        # age, on the same principle as the .tmp. staging files: never reap
        # something a live writer might still be filling.
        self._partial_ttl_s = float(
            envs.SGLANG_HICACHE_CANONICAL_PARTIAL_TTL_S.get() or 3600.0
        )
        # #558: the free-space floor the canonical protocol refuses below. The
        # LRU evictor's watermark does not cover this: it is disabled entirely
        # unless a cap or a min-free is configured, which is the default.
        self._space_floor_bytes = int(
            envs.SGLANG_HICACHE_CANONICAL_MIN_FREE_BYTES.get() or 0
        )
        if self.canonical_kv_page is not None or self.canonical_mamba_blob is not None:
            from sglang.srt.mem_cache.canonical_page_store import sweep_partials

            # 2026-08-28 boot-3 store wipe: this sweep, keyed on age alone,
            # reaped ALL 16898 partial files the previous boot had deposited
            # into the store -- the restart gap (100 min) exceeded the TTL
            # (3600 s), and nothing had completed yet, so cross-boot retention
            # went to zero at attach. Age cannot tell abandonment from a
            # restart; the marker can. A pair whose marker decodes against the
            # geometry THIS attach writes is resumable deposited work and is
            # kept at any age (computed work is never thrown away); a genuine
            # format transition is reaped past the TTL but NAMED loudly, never
            # wiped silently.
            resumable_totals = []
            if self.canonical_kv_page is not None:
                resumable_totals.append(int(self.canonical_kv_page.spec.page_bytes))
            if self.canonical_mamba_blob is not None:
                resumable_totals.append(int(self.canonical_mamba_blob.total_bytes))
            try:
                sweep_partials(
                    self.file_path,
                    older_than_s=self._partial_ttl_s,
                    is_pinned=self.pins.is_pinned,
                    resumable_totals=tuple(resumable_totals),
                )
            except OSError as e:
                # A store that cannot be swept is still usable; orphans only
                # cost disk, and the free-space watchdog still sees them.
                logger.warning("Could not sweep canonical partial files: %s", e)

        # THE "ONLY RANK 0 CREATES IT" GUARD DOES NOT COVER PIPELINE PARALLELISM.
        #
        # Measured 2026-08-18 07:37Z: the ARM I harvest boot died before health with
        #
        #     FileExistsError: [Errno 17] File exists: '/tmp/hicache'
        #
        # on a pp_size=3, tp_size=1 deployment. The condition below elects a single
        # creator by TP and attention-CP rank, which is exactly right when the fan-out
        # is TP -- but with pure PP every stage has tp_rank == 0 and attn_cp_rank == 0,
        # so all three ranks elect THEMSELVES, and the losers of the race raise. The
        # directory found afterwards was empty and stamped with the boot's own minute:
        # not a stale leftover, the boot's own first rank.
        #
        # exist_ok also closes the TOCTOU that was always here regardless of rank: the
        # exists() check and the makedirs() are two steps, so even a correctly elected
        # single creator races against anything else on the box using the same path.
        # Creating a directory that already exists is precisely the no-op we want, so
        # the election is not worth defending -- the idempotent call is.
        if not os.path.exists(self.file_path) and tp_rank == 0 and attn_cp_rank == 0:
            os.makedirs(self.file_path, exist_ok=True)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")
        elif not os.path.exists(self.file_path):
            # A non-electing rank still needs the directory to exist before it writes
            # into it; under PP the elected rank may simply be a different process
            # that has not run yet.
            os.makedirs(self.file_path, exist_ok=True)

        # Metadata cache positive lookup toggle & TTL
        enable_cache_raw = None
        if storage_config.extra_config:
            enable_cache_raw = storage_config.extra_config.get("enable_metadata_cache")
        if enable_cache_raw is None:
            enable_cache_raw = (
                envs.SGLANG_HICACHE_FILE_BACKEND_ENABLE_METADATA_CACHE.get()
            )

        self.enable_metadata_cache = bool(enable_cache_raw)

        if self.enable_metadata_cache:
            ttl_raw = None
            if storage_config.extra_config:
                ttl_raw = storage_config.extra_config.get("metadata_ttl")
            if ttl_raw is None:
                ttl_raw = envs.SGLANG_HICACHE_FILE_BACKEND_METADATA_TTL.get()

            self.metadata_ttl = float(ttl_raw) if ttl_raw is not None else 5.0
            self.metadata_cache = MetadataCache(self.metadata_ttl)
            self._scan_existing_files_to_metadata_cache()
        else:
            self.metadata_cache = None

        # All LRU / size accounting and disk eviction lives in the evictor so
        # this backend stays a thin raw-bytes store. Imported lazily: the storage
        # package __init__ pulls in the backend factory, which imports this
        # module, so a top-level import here would be circular.
        from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

        # Every non-MLA rank writes its own files into this one directory, so
        # they share the configured byte budget; under MLA / dcp owner mode only
        # rank 0 writes, so it gets the whole budget.
        writer_count = 1 if is_mla_model else max(1, tp_size)
        # #410: the pin ledger, built BEFORE the evictor because the evictor
        # must never run a single pass without it -- a checkpoint's pages are
        # protected from the first eviction or the protection is a promise with
        # a hole in it. Durable, because a checkpoint outlives the process.
        from sglang.srt.mem_cache.pin_ledger import PinLedger

        self.pins = PinLedger(
            self.file_path,
            budget_bytes=int(envs.SGLANG_HICACHE_PIN_BUDGET_BYTES.get() or 0),
        )
        self.pins.load()
        self._evictor = LRUFileEvictor(
            self.file_path,
            self.config_suffix,
            tp_rank=tp_rank,
            is_mla_model=is_mla_model,
            extra_config=storage_config.extra_config,
            on_evict=(
                self.metadata_cache.remove if self.metadata_cache is not None else None
            ),
            writer_count=writer_count,
            path_for_stem=self._existing_path,
            iter_existing=self._iter_existing_files,
            pins=self.pins,
            # #810: under `--hicache-host-role staging` this store IS the
            # retention tier, so it may not run unbounded. Decided from the
            # config the evictor itself resolved, one line below, rather than
            # from a second reading of the same knobs at parse time -- the two
            # readings would be free to drift, and the one that refuses would
            # not be the one that evicts.
            require_watermark=(
                getattr(storage_config, "host_role", "retention") == "staging"
            ),
        )

    def _pin_path(self, stem: str) -> str:
        """Where ``stem`` lives in THIS lineage's flat layout.

        RE-PORTED for this lineage (#410 reconciliation onto the 0817 train).
        The reconciliation wrote this as a flat join because the branch it came
        from had no sharding. THIS store shards (#558), so a flat join names a
        path it never writes: `pin_checkpoint` would drop every stem as
        "missing" and pin nothing while reporting success, and a test that
        removes `_pin_path(stem)` gets FileNotFoundError.

        The property the reconciliation actually asked for is preserved, and it
        is the one that matters: the ledger must stat EXACTLY what the evictor
        unlinks. Both now resolve through ``_existing_path`` -- sharded, else
        legacy flat, else the path it would be written to -- so there is one
        join, whatever the layout.
        """
        return self._existing_path(stem)


    def pin_checkpoint(self, checkpoint_id: str, keys: List[str]):
        """Pin every store object a checkpoint references (#410 slice 2).

        Translates CONTENT keys -- what a manifest holds -- into the suffixed
        stems the evictor indexes, so the manifest never has to know the
        store's key layout and the ledger never has to guess a size.

        Reports what it could NOT pin. ``stems_with_sizes`` drops a stem whose
        file is gone, which is right for the budget and invisible to the
        caller; recording it here, where the CONTENT key is still known, is
        what lets a create refuse by name instead of leaving the shortfall to
        surface at the branch.
        """
        import dataclasses

        from sglang.srt.mem_cache.pin_ledger import stems_with_sizes

        pairs = []
        missing: List[str] = []
        for key in keys:
            stem = self._get_suffixed_key(key)
            path = self._pin_path(stem)
            pairs.append((stem, path))
            if not os.path.exists(path):
                missing.append(key)
        result = self.pins.pin(checkpoint_id, stems_with_sizes(pairs))
        return dataclasses.replace(result, unpinned=tuple(missing))

    def unpin_checkpoint(self, checkpoint_id: str) -> int:
        """Release a checkpoint's pins, returning the bytes actually freed."""
        return self.pins.unpin(checkpoint_id)

    def pin_stats(self) -> dict:
        return self.pins.ledger()

    # Longest filename most Linux filesystems accept, in bytes.
    _NAME_MAX = 255

    def _tmp_path_for(self, tensor_path: str) -> str:
        """Staging name for the atomic write, sized to fit NAME_MAX.

        The final name is already long (page hash + served model name +
        identity hash + geometry suffix; the served model name is a full
        checkpoint PATH when the user does not pass --served-model-name). A
        staging suffix carrying pid + thread id + a full uuid4 hex added ~50
        more bytes, which pushed component pages of a deeply-nested checkpoint
        over the limit: the write failed with "[Errno 36] File name too long"
        and the store silently stayed empty -- the page was never persisted
        even though the FINAL name would have fit. Keep the staging suffix
        short, and shrink it further rather than fail when the final name is
        near the limit; uniqueness comes from uuid4 alone.
        """
        base = os.path.basename(tensor_path)
        room = self._NAME_MAX - len(base.encode("utf-8")) - len(".tmp.")
        if room < 4:
            raise OSError(
                f"cannot stage an atomic write for '{base}': the final name "
                f"already uses {len(base.encode('utf-8'))} of "
                f"{self._NAME_MAX} bytes. Pass a short --served-model-name; "
                "the default is the full checkpoint path."
            )
        return f"{tensor_path}.tmp.{uuid.uuid4().hex[:min(16, room)]}"

    def _is_draft_key(self, key: str) -> bool:
        """Draft pages, excluded from every neutralisation rule BY NAME.

        Draft KV is the exact MIRROR of target KV: head-SHARDED and
        token-COMPLETE, written by every rank under its own suffix. No suffix
        rule can neutralise that, so the draft pool starts cold after a flip or
        a reboot -- the designed shape, and the reason a cross-phase hit is
        expected to be PARTIAL rather than total.
        """
        return key.endswith(f".{PoolName.DRAFT}")

    def _is_shared_kv_key(self, key: str) -> bool:
        """True for the keys whose bytes are geometry-independent.

        Plain KV page keys are bare page hashes (hex, no '.'); component and
        draft keys are '{hash}.{pool_name}'. Only the plain ones can lose a
        geometry suffix, under either rule that earns it:

        * ``dcp_owner_mode`` -- pages carry FULL replicated kv-heads and are
          token-sharded, so a page is complete on its owner rank (token axis).
        * ``canonical_kv_page`` (#706) -- pages carry every attention layer, so
          a page is complete across PP stages (layer axis).

        DRAFT PAGES ARE EXCLUDED BY NAME, and not as an oversight. Draft KV is
        the exact MIRROR of target KV: head-SHARDED and token-COMPLETE, written
        by every rank under its own suffix. No suffix rule can neutralise that,
        so the draft pool starts cold after a flip or a reboot -- the designed
        shape, which is why a cross-phase hit is expected to be PARTIAL.
        Component pools (mamba/SWA) are genuinely per-rank shards for the same
        kind of reason and keep their suffix too; their cross-geometry form is
        the offline cut in ``hicache_migrate`` (``MambaBlobSpec.for_layers`` /
        ``layer_extents`` for the layer axis), never a softened key.
        """
        return "." not in key

    def _is_shared_mamba_key(self, key: str) -> bool:
        """True when the GDN/mamba blob for this key is the canonical one.

        Gated on the window existing, because only then are the blob's bytes
        full-width in both axes (every layer, every head) instead of this
        rank's shard. Without it the blob stays per-rank and per-stage, exactly
        as it is today.
        """
        return self.canonical_mamba_blob is not None and key.endswith(
            f".{PoolName.MAMBA}"
        )

    def _derive_key_suffixes(self) -> None:
        """Build the two key suffixes from the geometry the BYTES still depend on.

        #969F: THIS USED TO RUN ONCE, IN `__init__`, AND THE FACT IT READS
        CHANGES AFTERWARDS. `install_canonical_windows` assigns
        `self.canonical_kv_page` (:1347 pre-fix) and re-derived nothing, so a
        store whose canonical page is installed AFTER construction -- which is
        the live path, `cache_controller.py:1278` -- kept keys carrying
        `_{tp_rank}_{tp_size}` and `_{pp_size}_{pp_rank}` even though the
        canonical format had made that geometry irrelevant to the bytes.

        The consequence is the whole read-side miss of this campaign: those
        terms CHANGE AT EVERY FLIP (PP phase tp_size=1/pp_size=3, TP phase
        tp_size=3/pp_size=1), so a page written in one phase is asked for under
        a key the other phase never wrote. Measured: 132 `#937 STALE PREFETCH
        INSERT REFUSED ... 0 token(s) fetched`, `#cached-token > 0` on 0 of 355
        prefill lines, 315 of 324 re-admissions matching an empty tree.

        The #706 rule the exemption states is unchanged and is the reason this
        is a re-derivation and not a new rule: "the key carries exactly the
        geometry the bytes still depend on". Under the canonical page the bytes
        stop depending on the cut -- whenever that becomes true, including
        later than construction.

        The constructor already refuses a canonical mamba blob configured
        WITHOUT the canonical KV page (":679", "a neutral GDN blob beside
        pp-suffixed KV pages still misses across the flip") -- the author saw
        this interaction; the guard simply could not fire for windows installed
        after construction.
        """
        g = self._key_geom
        self.config_suffix = f"_{g['model_name']}"
        self.kv_config_suffix = f"_{g['model_name']}"
        if g["identity_hash"]:
            self.config_suffix += f"_{g['identity_hash']}"
            self.kv_config_suffix += f"_{g['identity_hash']}"
        if not g["is_mla_model"]:
            self.config_suffix += f"_{g['tp_rank']}_{g['tp_size']}"
            if not self.dcp_owner_mode and self.canonical_kv_page is None:
                self.kv_config_suffix += f"_{g['tp_rank']}_{g['tp_size']}"
        if g["enable_pp"]:
            self.config_suffix += f"_{g['pp_size']}_{g['pp_rank']}"
            if self.canonical_kv_page is None:
                self.kv_config_suffix += f"_{g['pp_size']}_{g['pp_rank']}"
        if g["attn_cp_size"] > 1:
            self.config_suffix += f"_cp{g['attn_cp_rank']}_{g['attn_cp_size']}"
            self.kv_config_suffix += f"_cp{g['attn_cp_rank']}_{g['attn_cp_size']}"


    def _get_suffixed_key(self, key: str) -> str:
        if self._is_draft_key(key):
            return key + self.config_suffix
        if (
            self.dcp_owner_mode or self.canonical_kv_page is not None
        ) and self._is_shared_kv_key(key):
            return key + self.kv_config_suffix
        if self._is_shared_mamba_key(key):
            return key + self.kv_config_suffix
        return key + self.config_suffix

    def _get_component_key(self, key: str, component_name: Optional[str] = None) -> str:
        # #969G KEY INSTRUMENT (temporary). The ONE funnel every store key goes
        # through, read and write alike (6 call sites). The open question is
        # whether the re-admission asks for the SAME key the retention wrote:
        # equal keys with an empty answer is a store/lookup defect, different
        # keys is a derivation defect and the difference names the field.
        # The caller's name distinguishes read from write with no plumbing.
        # Grep: "#969G KEY".
        if component_name is None or component_name in ("__default__", PoolName.KV):
            out = self._get_suffixed_key(key)
        else:
            out = self._get_suffixed_key(f"{key}.{component_name}")
        try:
            import sys as _sys

            _n = getattr(HiCacheFile, "_969g_n", 0) + 1
            HiCacheFile._969g_n = _n
            if _n <= 60 or _n % 1024 == 0:
                logger.warning(
                    "#969G KEY n=%d caller=%s component=%s key=%s",
                    _n,
                    _sys._getframe(1).f_code.co_name,
                    component_name,
                    out,
                )
        except Exception:  # noqa: BLE001
            logger.warning("#969G KEY PROBE RAISED", exc_info=True)
        return out

    def _sharded_path(self, stem: str) -> str:
        """Path a NEW file for ``stem`` is written to."""
        return os.path.join(self.file_path, page_shard(stem), f"{stem}.bin")

    def _flat_path(self, stem: str) -> str:
        """Pre-sharding path for ``stem`` (read-only compatibility)."""
        return os.path.join(self.file_path, f"{stem}.bin")

    def _existing_path(self, stem: str) -> str:
        """Where ``stem`` currently lives: sharded if present, else the legacy
        flat path if present, else the sharded path it would be written to.

        Read-through migration: a directory written before sharding keeps
        serving hits and its files stay evictable, while every new write is
        sharded. Nothing rewrites or moves the old files.
        """
        sharded = self._sharded_path(stem)
        if os.path.exists(sharded):
            return sharded
        flat = self._flat_path(stem)
        if os.path.exists(flat):
            return flat
        return sharded

    def _stem_exists(self, stem: str) -> bool:
        """True when ``stem`` is on disk, sharded or in the legacy flat layout."""
        return os.path.exists(self._sharded_path(stem)) or os.path.exists(
            self._flat_path(stem)
        )

    def _ensure_shard_dir(self, path: str) -> None:
        """Create the shard directory of ``path`` once per shard."""
        shard = os.path.dirname(path)
        if shard in self._known_shards:
            return
        os.makedirs(shard, exist_ok=True)
        self._known_shards.add(shard)

    def _iter_existing_files(self):
        """(stem, stat) for every ``.bin`` under the storage directory.

        Walks the shard directories and the storage directory itself, so a
        directory that predates sharding is picked up unchanged.
        """
        try:
            with os.scandir(self.file_path) as it:
                top = list(it)
        except FileNotFoundError:
            return
        for entry in top:
            if entry.is_dir():
                try:
                    with os.scandir(entry.path) as shard_it:
                        shard_entries = list(shard_it)
                except OSError:
                    continue
                for sub in shard_entries:
                    if not sub.name.endswith(".bin"):
                        continue
                    try:
                        yield sub.name[:-4], sub.stat()
                    except OSError:
                        continue
            elif entry.name.endswith(".bin"):
                try:
                    yield entry.name[:-4], entry.stat()
                except OSError:
                    continue

    def _get_component_path(
        self, key: str, component_name: Optional[str] = None
    ) -> str:
        return self._existing_path(self._get_component_key(key, component_name))

    def _scan_existing_files_to_metadata_cache(self) -> None:
        for stem, _st in self._iter_existing_files():
            # Only files belonging to this rank/model. Shared KV files
            # (dcp_owner_mode on the token axis, #706 on the layer axis) carry
            # the geometry-free kv_config_suffix instead.
            if stem.endswith(self.config_suffix) or (
                (
                    self.dcp_owner_mode
                    or self.canonical_kv_page is not None
                    or self.canonical_mamba_blob is not None
                )
                and stem.endswith(self.kv_config_suffix)
            ):
                self.metadata_cache.add(stem)




    def _canonical_space_check(self, need_bytes: int) -> None:
        """Refuse a canonical write that would take the store below the floor."""
        from sglang.srt.mem_cache.canonical_page_store import ensure_space

        ensure_space(self.file_path, need_bytes, self._space_floor_bytes)

    def _canonical_window(self, key: str):
        """The canonical window serving this key, or None for the normal path.

        One dispatch for both pools: the KV page's window in its generic
        one-extent form, or the mamba blob's extent window. Draft keys never
        reach either (excluded by name), and any other component pool keeps its
        per-rank key because no canonical form is defined for it.
        """
        if self._is_draft_key(key):
            return None
        if self._canonical_kv_extents is not None and self._is_shared_kv_key(key):
            return self._canonical_kv_extents
        if self._is_shared_mamba_key(key):
            return self.canonical_mamba_blob
        return None

    def _get_canonical_slice(
        self, key: str, window, target_location: torch.Tensor
    ) -> torch.Tensor | None:
        """Cut this rank's extents out of the canonical blob (#706).

        Both phases arrive here with the SAME key and leave with different
        bytes: a PP stage takes the byte ranges of the layers it owns, a TP rank
        takes its head channels across every layer. That is the read-time cut
        the token axis already does for kv-heads, on the other two axes.
        """
        from sglang.srt.mem_cache.canonical_page_store import read_extents

        suffixed = self._get_suffixed_key(key)
        tensor_path = self._existing_path(suffixed)
        try:
            served = read_extents(tensor_path, window, target_location)
        except CanonicalPageError as e:
            # A geometry that cannot cut this blob is a configuration error, not
            # a cache miss. Loud and per-page rather than raised, because this
            # runs on the prefetch worker and a dead worker is a wedge.
            logger.error("Canonical read refused for %s: %s", key, e)
            return None
        if not served:
            if self.metadata_cache is not None:
                self.metadata_cache.remove(suffixed)
            return None
        self._evictor.touch(suffixed, tensor_path)
        if self.metadata_cache is not None:
            self.metadata_cache.add(suffixed)
        return target_location

    def _set_canonical_slice(self, key: str, window, value: torch.Tensor) -> bool:
        """Deposit this rank's extents into the canonical blob (#706).

        The blob becomes visible only once every byte is present, so a writer
        acting alone leaves nothing readable behind -- see
        ``canonical_page_store`` for the marker and the rename.
        """
        from sglang.srt.mem_cache.canonical_page_store import write_extents

        suffixed = self._get_suffixed_key(key)
        if self.exists(key):
            # A complete blob is content-addressed: it already holds exactly
            # these bytes. Refresh recency, write nothing.
            self._evictor.touch(suffixed, self._existing_path(suffixed))
            return True

        tensor_path = self._sharded_path(suffixed)
        reserved = False
        try:
            # Charged per SLICE: the writers together account for one blob, and
            # the one that completes it has ``commit`` correct the estimate to
            # the file's real allocation.
            if not self._evictor.reserve(suffixed, window.payload_bytes, key=key):
                return False
            reserved = True
            self._ensure_shard_dir(tensor_path)
            result = write_extents(
                tensor_path, window, value, space_check=self._canonical_space_check
            )
            self._evictor.commit(suffixed)
            if (
                result.completed or result.already_complete
            ) and self.metadata_cache is not None:
                self.metadata_cache.add(suffixed)
            return True
        except Exception as e:
            logger.error(f"Failed to save canonical slice for {key}: {e}")
            if reserved:
                self._evictor.abort(suffixed)
            return False

    def install_canonical_windows(self, kv_page, mamba_blob) -> None:
        """#706 x #719 (0828): swap this backend's read/write-time cut.

        Called by ``HiCacheController.rebind_canonical_windows`` at the flip
        cutover, AFTER the #719 pool rebind committed, with windows derived
        from the pools now bound. The store's bytes never move -- one
        canonical file per key, every phase cutting its own extents out of it
        -- only WHICH extents this rank reads and writes changes with the
        phase.

        Never a format transition: window presence decides the KEY shape
        (`_get_suffixed_key`), so flipping the format on or off here would
        silently re-key a live store and strand every page written under the
        other rule. Likewise the canonical TOTAL is a model constant: a
        different total is a different page format, not another phase's cut
        of the same one.
        """
        if (kv_page is None) != (self.canonical_kv_page is None) or (
            (mamba_blob is None) != (self.canonical_mamba_blob is None)
        ):
            raise CanonicalPageError(
                "refusing to switch the canonical format on or off at a "
                "cutover: window presence decides the key shape, and "
                "re-keying a live store strands every page written under "
                "the other rule."
            )
        if kv_page is None:
            return
        if int(kv_page.spec.page_bytes) != int(
            self.canonical_kv_page.spec.page_bytes
        ):
            raise CanonicalPageError(
                f"refusing to install a KV window of a "
                f"{kv_page.spec.page_bytes}-byte page over a store keyed for "
                f"{self.canonical_kv_page.spec.page_bytes}-byte pages: that "
                "is a different page format, not another phase's cut."
            )
        if mamba_blob is not None and int(mamba_blob.total_bytes) != int(
            self.canonical_mamba_blob.total_bytes
        ):
            raise CanonicalPageError(
                f"refusing to install a mamba window of a "
                f"{mamba_blob.total_bytes}-byte blob over a store keyed for "
                f"{self.canonical_mamba_blob.total_bytes}-byte blobs."
            )
        self.canonical_kv_page = kv_page
        self._canonical_kv_extents = kv_page.as_extents()
        self.canonical_mamba_blob = mamba_blob
        # #969F: THE FACT THE KEY SUFFIX DEPENDS ON JUST CHANGED. Re-derive it
        # here, at the one place that changes it, so there is a single
        # derivation function and no second moment. Without this the store
        # keeps the pre-canonical, geometry-bearing key and every page written
        # before a flip is unreachable after it.
        self._derive_key_suffixes()

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        window = self._canonical_window(key)
        if window is not None:
            return self._get_canonical_slice(key, window, target_location)
        suffixed = self._get_suffixed_key(key)
        tensor_path = self._existing_path(suffixed)
        try:
            expected = target_location.numel() * target_location.element_size()
            with open(tensor_path, "rb", buffering=0) as f:
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())
                # An unbuffered readinto is one syscall and may legitimately
                # return short on a large page -- loop until the page is whole
                # or the file really is truncated. KV pages (tens of KiB) never
                # hit this; component pages do (a Mamba/GDN state page is tens
                # of MiB), and a partial recurrent state is the worst possible
                # thing to hand back.
                got = 0
                while got < expected:
                    n = f.readinto(buf[got:])
                    if not n:
                        break
                    got += n
                if got != expected:
                    raise IOError(
                        f"Short read for {suffixed}: {got} of {expected} bytes"
                    )
            self._evictor.touch(suffixed, tensor_path)
            if self.metadata_cache is not None:
                self.metadata_cache.add(suffixed)
            return target_location
        except FileNotFoundError:
            if self.metadata_cache is not None:
                self.metadata_cache.remove(suffixed)
            logger.warning(f"Failed to fetch {key} from HiCacheFile storage.")
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        return [
            self.get(key, target_location)
            for key, target_location in zip(
                keys, target_locations or [None] * len(keys)
            )
        ]

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        window = self._canonical_window(key)
        if window is not None:
            return self._set_canonical_slice(key, window, value)
        suffixed = self._get_suffixed_key(key)

        # Fast path: same key already on disk. Refresh recency and skip rewrite.
        if self.exists(key):
            logger.debug(f"Key {key} already exists. Skipped.")
            self._evictor.touch(suffixed, self._existing_path(suffixed))
            return True

        # New pages are always sharded, whatever the directory looked like before.
        tensor_path = self._sharded_path(suffixed)

        tmp_path = None
        reserved = False
        try:
            value_bytes = value.numel() * value.element_size()
            # Ask the evictor to admit + reserve disk space (evicting if needed).
            if not self._evictor.reserve(suffixed, value_bytes, key=key):
                return False
            reserved = True

            self._ensure_shard_dir(tensor_path)
            tmp_path = self._tmp_path_for(tensor_path)
            value.contiguous().view(dtype=torch.uint8).numpy().tofile(tmp_path)
            os.replace(tmp_path, tensor_path)
            self._evictor.commit(suffixed)
            if self.metadata_cache is not None:
                self.metadata_cache.add(suffixed)
            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            # Roll back the reservation and clean up any half-written file.
            if reserved:
                self._evictor.abort(suffixed)
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if self.metadata_cache is not None:
                self.metadata_cache.remove(suffixed)
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        for key, value in zip(keys, values):
            if not self.set(key, value):
                return False
        return True

    def exists(self, key: str) -> bool:
        key = self._get_suffixed_key(key)
        if self.metadata_cache is not None and self.metadata_cache.contains(key):
            return True
        if self._stem_exists(key):
            if self.metadata_cache is not None:
                self.metadata_cache.add(key)
            return True
        return False

    def _collect_existing_component_keys(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
    ) -> Set[str]:
        target_files = {f"{self._get_component_key(key)}.bin" for key in keys}
        for transfer in pool_transfers or []:
            for key in keys:
                target_files.add(f"{self._get_component_key(key, transfer.name)}.bin")

        if self.metadata_cache is None:
            # One stat per candidate. This used to be a single full-directory
            # scandir, which was cheaper only while the directory was flat and
            # small; sharding makes the targeted lookups strictly better (the
            # incident directory held 11.7M entries, so every sweep walked
            # them all).
            existing_files = set()
            for filename in target_files:
                if self._stem_readable(filename[:-4]):
                    existing_files.add(filename)
            return existing_files

        existing_files = set()
        for filename in target_files:
            stem = filename[:-4]
            if self.metadata_cache.contains(stem):
                existing_files.add(filename)
            else:
                if self._stem_readable(stem):
                    self.metadata_cache.add(stem)
                    existing_files.add(filename)
        return existing_files

    def _canonical_total_for_stem(self, stem: str) -> Optional[int]:
        """The canonical width a stem's file must have to be readable, or None."""
        if self._canonical_kv_extents is None:
            return None
        suffix = self.kv_config_suffix
        if not suffix or not stem.endswith(suffix):
            return None
        bare = stem[: -len(suffix)]
        if self.canonical_mamba_blob is not None and bare.endswith(
            f".{PoolName.MAMBA}"
        ):
            return int(self.canonical_mamba_blob.total_bytes)
        if "." not in bare:
            return int(self._canonical_kv_extents.total_bytes)
        return None

    def _stem_readable(self, stem: str) -> bool:
        """Presence a reader can actually serve (#706 x #719, 0828).

        A canonical stem counts only when its file has the canonical width;
        a same-stem file of another width -- a leftover of a different format
        era in a long-lived store -- would pass an existence check and then
        refuse at `read_extents`, which is exactly the presence-vs-readability
        drift the 0828 specimen paid a full re-prefill for. Non-canonical
        stems keep the plain existence answer.
        """
        if not self._stem_exists(stem):
            return False
        total = self._canonical_total_for_stem(stem)
        if total is None:
            return True
        try:
            return os.path.getsize(self._existing_path(stem)) == int(total)
        except OSError:
            return False

    # #706 x #719 (0828): occurrences of a refused presence probe, class-wide
    # so the rate limit survives a backend re-attach.
    _probe_mismatch_count = 0

    def _canonical_probe_mismatch(self) -> Optional[str]:
        """Fix 2 of the 0828 specimen: presence must not outrun readability.

        `batch_exists_v2` promised pages by bare file existence while the
        reader's window refused every one of them ("read target holds 32768
        bytes but this KV page window is 8192 bytes"), so the issuance saw
        hits the fetch could never deliver, the match collapsed to 0, and the
        #928 anchor re-prefilled the WHOLE prefix. The compatibility between
        the window and the pool the read path actually fills therefore
        belongs in the probe itself: a store the current window cannot cut is
        an honest cold miss, never a promise.

        Returns the named mismatch, or None when the probe may answer. Pools
        this backend was never given (bare-backend unit tests, non-hybrid
        deployments) are not turned into a claim.
        """
        if self._canonical_kv_extents is None:
            return None
        pool = getattr(self, "mem_pool_host", None)
        if pool is not None:
            try:
                page = pool.get_dummy_flat_data_page()
                have = int(page.numel()) * int(page.element_size())
            except Exception:  # noqa: BLE001 - verification is best-effort
                have = None
            want = int(self._canonical_kv_extents.payload_bytes)
            if have is not None and have != want:
                return (
                    f"the KV window cuts {want} bytes but the bound host "
                    f"pool's page is {have} bytes"
                )
        blob = self.canonical_mamba_blob
        mamba_pool = (getattr(self, "registered_pools", None) or {}).get(
            PoolName.MAMBA
        )
        if blob is not None and mamba_pool is not None:
            try:
                page = mamba_pool.get_dummy_flat_data_page()
                have = int(page.numel()) * int(page.element_size())
            except Exception:  # noqa: BLE001 - verification is best-effort
                have = None
            if have is not None and have != int(blob.payload_bytes):
                return (
                    f"the mamba window cuts {blob.payload_bytes} bytes but "
                    f"the registered mamba pool's page is {have} bytes"
                )
        return None

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        mismatch = self._canonical_probe_mismatch()
        if mismatch is not None:
            HiCacheFile._probe_mismatch_count += 1
            n = HiCacheFile._probe_mismatch_count
            if n <= 3 or n % 100 == 0:
                logger.error(
                    "Canonical presence probe refused (occurrence %d): %s. "
                    "Answering 0 hits -- an honest cold miss -- instead of "
                    "promising pages the reader's window would refuse (the "
                    "0828 specimen: a #719 rebind without a window rebuild).",
                    n,
                    mismatch,
                )
            return PoolTransferResult(0, {})
        existing_files = self._collect_existing_component_keys(keys, pool_transfers)

        def has_component(page_idx: int, name: str) -> bool:
            return (
                f"{self._get_component_key(keys[page_idx], name)}.bin" in existing_files
            )

        # Longest contiguous KV prefix present in storage.
        kv_pages = next(
            (
                i
                for i in range(len(keys))
                if f"{self._get_component_key(keys[i])}.bin" not in existing_files
            ),
            len(keys),
        )

        hit_count: dict[str, int] = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        # #1035 R13: THE MISS THAT LOOKS LIKE SILENCE.
        # `#1028B` below only fires when a COMPONENT cap moved the number
        # (`final_pages != kv_pages`). When the KV prefix itself is 0 the claim
        # is already zero and nothing prints -- so "storage genuinely has
        # nothing for this key" and "everything is fine" are the SAME log, which
        # is how a dead read path can look healthy for a whole campaign. Say it
        # once per occurrence, rate-limited, and say which of the two it is.
        if kv_pages == 0 and keys:
            self._1035r13_n = getattr(self, "_1035r13_n", 0) + 1
            if self._1035r13_n <= 40 or self._1035r13_n % 256 == 0:
                logger.warning(
                    "#1035 R13 EMPTY KV PREFIX n=%d: storage holds NO leading "
                    "page for this key set (keys=%d) -- an honest cold miss, "
                    "not a component cap. Distinguishing this from a capped "
                    "claim is the whole point: #1028B stays silent here "
                    "because final==kv==0.",
                    self._1035r13_n,
                    len(keys),
                )

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            name = transfer.name
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = next(
                    (i for i in range(kv_pages) if not has_component(i, name)), kv_pages
                )
            else:  # trailing_pages
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                boundary = 0
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        has_component(i, name)
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break
            if boundary:
                hit_count[name] = boundary
            final_pages = min(final_pages, boundary)

        # #1028B THE CAP, NAMED. This `min` is the only place that decides how
        # much of an existing KV prefix a prefetch may actually claim, and it
        # printed NOTHING: measured 2026-08-30 on boot `boot_855_1028fence`,
        # the strings `final_pages`, `kv_pages`, `boundary=` and `hit_pages`
        # each occur 0 times in the whole 5.87 MB log. The consequence was
        # that a completion of 3072 against a 13179-token prompt could not be
        # attributed -- "the mamba anchors are too sparse" and "the KV prefix
        # in storage is short" produce the SAME number here and were not
        # separable from that boot at all.
        #
        # Both terms on one line, so the next boot answers it directly:
        # `kv` is the longest contiguous KV prefix present in storage and
        # `final` is what survives the component caps; when they differ, the
        # component named in `caps` is the binding constraint.
        if final_pages != kv_pages:
            self._1028b_n = getattr(self, "_1028b_n", 0) + 1
            if self._1028b_n <= 40 or self._1028b_n % 256 == 0:
                # `lost` is the REALISED loss of this claim: pages that exist
                # in storage and were given up. It is the quantity the
                # no-double-prefill law (#939) bounds at ONE chunk, so it is
                # printed as a number rather than left to be subtracted by
                # whoever reads the line.
                #
                # NOTE FOR THE READER OF `caps`: an empty dict does NOT mean
                # "no component cap applied". `hit_count[name]` is only
                # assigned `if boundary`, so a component whose boundary came
                # out 0 -- the total-miss case, which is exactly the one that
                # zeroes the claim -- leaves NO entry behind. Empty caps next
                # to final=0 therefore means "a component found no anchor in
                # this span at all", the opposite of "nothing capped it".
                # #1035b WHERE IS THE NEAREST ANCHOR? `claimed=0 caps={}` is
                # produced by two different worlds and the line above cannot
                # tell them apart:
                #   (i)  anchors ARE inside the queried range but not at a
                #        reachable trailing position -- a GRANULARITY problem,
                #        fixed by publishing anchors more often; or
                #   (ii) the queried range was cut SHORT of the nearest anchor
                #        -- a SPAN problem (the prefetch span is truncated, or
                #        the node boundary simply lies past the last queried
                #        key), where denser publication inside the span would
                #        change nothing.
                # Building the density step without separating these is exactly
                # the "fix an unverified root" move that has already cost this
                # strand two roots, so the discriminator is printed at the
                # failure point rather than argued: for each capped component,
                # how many of its blobs exist among the queried keys and the
                # DEEPEST index carrying one (-1 = none at all). Computed only
                # on the rate-limited logging path.
                _anchor_probe = {}
                for _t in pool_transfers or []:
                    if _t.name == PoolName.KV:
                        continue
                    _present = [
                        i for i in range(len(keys)) if has_component(i, _t.name)
                    ]
                    _anchor_probe[_t.name] = (
                        len(_present),
                        _present[-1] if _present else -1,
                    )
                logger.warning(
                    "#1028B FETCH CAP n=%d: kv=%d claimed=%d lost=%d caps=%s "
                    "keys=%d #1035b anchors_in_range(count,deepest_idx)=%s",
                    self._1028b_n,
                    kv_pages,
                    final_pages,
                    kv_pages - final_pages,
                    {k: v for k, v in hit_count.items() if k != PoolName.KV},
                    len(keys),
                    _anchor_probe,
                )

        return PoolTransferResult(final_pages, hit_count)

    def _log_key(self, pool_name: str, key: str) -> str:
        return key if pool_name == PoolName.KV else f"{key}.{pool_name}"

    def _read_buffer_pool(self, pool_name: str, host_pool):
        """#720: the reusable read target for this pool, or None (today's path).

        Built lazily, once per registered pool, because the page size is the
        pool's and is only known here. ``0`` -- the default -- keeps the
        per-read fresh allocation exactly as it was.
        """
        capacity = int(envs.SGLANG_HICACHE_READ_BUFFERS.get() or 0)
        if capacity <= 0:
            return None
        pools = getattr(self, "_read_buffers", None)
        if pools is None:
            pools = self._read_buffers = {}
        pool = pools.get(pool_name)
        if pool is None:
            from sglang.srt.mem_cache.read_buffer_pool import ReadBufferPool

            probe = host_pool.get_dummy_flat_data_page()
            pool = ReadBufferPool(
                name=f"HiCache read buffers [{pool_name}]",
                flag="SGLANG_HICACHE_READ_BUFFERS",
                capacity=capacity,
                page_bytes=int(probe.numel()) * int(probe.element_size()),
                factory=host_pool.get_dummy_flat_data_page,
            )
            pools[pool_name] = pool
        return pool

    def _read_page(self, pool_name: str, key: str, host_pool, page_offset: int) -> bool:
        """Read one page from storage into host_pool at page_offset."""
        from sglang.srt.mem_cache.read_buffer_pool import borrowed

        storage_key = self._log_key(pool_name, key)
        buffers = self._read_buffer_pool(pool_name, host_pool)
        with borrowed(buffers, host_pool.get_dummy_flat_data_page) as target:
            data_page = self.get(storage_key, target)
            if data_page is None:
                return False
            host_pool.set_from_flat_data_page(page_offset, data_page)
        return True

    def _write_page(
        self, pool_name: str, key: str, host_pool, page_offset: int
    ) -> bool:
        """Write one page from host_pool at page_offset to storage as raw bytes."""
        storage_key = self._log_key(pool_name, key)
        data_page = host_pool.get_data_page(page_offset, flat=True)
        return self.set(storage_key, data_page)

    def _batch_io_v2(self, transfers: List[PoolTransfer], op_fn):
        results: dict[str, List[bool]] = {}
        for transfer in transfers:
            host_pool = self.registered_pools[transfer.name]
            keys = transfer.keys or []
            page_size = getattr(host_pool, "page_size", 1) or 1
            expected = len(keys) * page_size
            host_indices = transfer.host_indices

            if host_indices is None or host_indices.numel() != expected:
                logger.error(
                    "%s indices length mismatch for %s: expected %s, got %s",
                    op_fn.__name__,
                    transfer.name,
                    expected,
                    host_indices.numel() if host_indices is not None else 0,
                )
                results[transfer.name] = [False] * len(keys)
                continue

            results[transfer.name] = [
                op_fn(transfer.name, key, host_pool, host_indices[i * page_size].item())
                for i, key in enumerate(keys)
            ]
        return results

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, self._read_page)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, self._write_page)

    def capacity_stats(self) -> Optional[dict]:
        stats = self._evictor.stats()
        stats["file_path"] = self.file_path
        return stats

    def resize(
        self,
        *,
        max_size_bytes: Optional[int] = None,
        min_free_bytes: Optional[int] = None,
    ) -> Optional[dict]:
        stats = self._evictor.set_limits(
            max_size_bytes=max_size_bytes, min_free_bytes=min_free_bytes
        )
        stats["file_path"] = self.file_path
        return stats

    def check_disk_space(self, force: bool = False) -> bool:
        """Run the free-space watchdog; False means writes are stopped.

        Called from the storage worker loop so a filesystem that fills up is
        noticed while the backend is idle, not only when the next page happens
        to be written.
        """
        return self._evictor.check_free_space(force=force)

    def clear(self) -> bool:
        try:
            for dirpath, _dirnames, filenames in os.walk(self.file_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            self._evictor.clear()
            if self.metadata_cache is not None:
                self.metadata_cache.clear()
            logger.info("Cleared all entries in HiCacheFile storage.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear HiCacheFile storage: {e}")
            return False
