# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""LRU/size-based file eviction for the HiCache file storage backend.

``HiCacheFile`` is a thin raw-bytes store: it suffixes keys, reads/writes
``.bin`` pages, and answers existence queries. Everything that bounds how much
disk those pages consume -- the LRU recency index, per-file size accounting,
free-space probing, scanning pre-existing files on startup, and unlinking
victims -- lives here so the backend stays a plain key/value store.

A backend constructs one evictor and drives it through a small lifecycle::

    touch(key, path)                 # read hit / already-on-disk: bump recency
    reserve(key, n_bytes) -> bool    # admit a new write, evicting if needed
        commit(key)                  #   write landed on disk
        abort(key)                   #   write failed; release the reservation
    clear()                          # backend wiped all files

When eviction is not configured the evictor is inert: ``reserve`` always admits
and the other calls are no-ops, so the backend behaves as unbounded storage.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Iterable, Optional, Set, Tuple

from sglang.srt.environ import envs
from sglang.srt.utils.common import human_readable_int

logger = logging.getLogger(__name__)

# How often the free-space watchdog may probe statvfs, in seconds.
_FREE_SPACE_PROBE_INTERVAL_S = 5.0
# Free space must climb this far above min_free before a latched write stop is
# released, so the backend cannot oscillate between stopped and writing.
_FREE_SPACE_RECOVERY_FACTOR = 1.05


def _parse_size_to_bytes(value: Any) -> int:
    """Parse a size to bytes via human_readable_int (e.g. '200G', '1Gi', '1048576').
    None / empty / '0' disables; an invalid value also disables (with a warning)."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value))
    s = str(value).strip()
    if not s or s == "0":
        return 0
    try:
        return max(0, human_readable_int(s))
    except (argparse.ArgumentTypeError, ValueError):
        logger.warning(f"Invalid size {value!r} for HiCacheFile; disabling.")
        return 0


class LRUFileEvictor:
    """Bounds the on-disk size of a HiCacheFile directory via LRU eviction.

    Tracks one ``.bin`` file per suffixed key (oldest at the front of the LRU),
    enforces an optional byte cap and an optional free-space watermark, and
    unlinks the least-recently-used files to stay within those bounds. Eviction
    config comes from ``extra_config`` (per-backend, takes precedence) falling
    back to the ``SGLANG_HICACHE_FILE_BACKEND_*`` env vars.
    """

    def __init__(
        self,
        file_path: str,
        config_suffix: str,
        *,
        tp_rank: int,
        is_mla_model: bool,
        extra_config: Optional[dict] = None,
        on_evict: Optional[Callable[[str], None]] = None,
        writer_count: int = 1,
        path_for_stem: Optional[Callable[[str], str]] = None,
        iter_existing: Optional[
            Callable[[], Iterable[Tuple[str, os.stat_result]]]
        ] = None,
        pins: Optional[Any] = None,
        require_watermark: bool = False,
    ) -> None:
        # #410: the pin ledger, or None. None is the default and every code
        # path guards on it, so a store without checkpoints behaves exactly as
        # before -- no extra lookup on the eviction hot path.
        self._pins = pins
        self.file_path = file_path
        self.config_suffix = config_suffix
        self._tp_rank = tp_rank
        self._on_evict = on_evict
        # Every rank that writes into this directory shares one filesystem, so
        # by default the configured cap is split between them (see _load_config).
        self._writer_count = max(1, int(writer_count))
        # The backend owns the on-disk layout (sharded subdirectories, legacy
        # flat files); it injects how a key stem maps to a path and how to
        # enumerate what is already there. The defaults below describe the
        # PRE-SHARDING flat layout only -- a caller whose store is sharded must
        # inject BOTH, or the scan will not see its sharded files (they stay
        # untracked, hence never evictable) and eviction will unlink paths that
        # do not exist. ``HiCacheFile`` injects both.
        self._path_for_stem = path_for_stem or (
            lambda stem: os.path.join(self.file_path, f"{stem}.bin")
        )
        self._iter_existing = iter_existing or self._iter_existing_flat
        # Free-space watchdog state: a latch, so a full disk produces one loud
        # error and cheap refusals instead of a per-page warning flood.
        self._write_stopped = False
        self._last_free_probe = 0.0
        self._refused_since_stop = 0
        # #410: the pin ledger, or None. None is the default and every code
        # path below is written so that a store without one behaves exactly as
        # it did before pins existed -- this is a protection added to the
        # evictor, not a change to how it evicts.
        self._pins = pins

        # MLA ranks share the same physical files, so centralize LRU bookkeeping
        # on rank 0; non-MLA ranks each own their own files via the suffix.
        self._is_storage_owner = (not is_mla_model) or (tp_rank == 0)

        # suffixed_key -> allocated disk bytes; oldest at front.
        self._lru: OrderedDict[str, int] = OrderedDict()
        self._pending_writes: Set[str] = set()
        self._total_bytes: int = 0
        self._lock = threading.Lock()

        self._load_config(extra_config or {})

        self._eviction_configured = self.max_size_bytes > 0 or self.min_free_bytes > 0
        if require_watermark and not self._eviction_configured:
            # #810: refused, not auto-armed. Arming needs a NUMBER, and there
            # is no honest source for one here. `max_size` would have to be
            # invented; `min_free` likewise; and the one derivation that does
            # exist -- `_clamp_max_size_to_fs` -- clamps a cap that was already
            # given and is not itself a bound, since the filesystem's own
            # capacity permits the store to consume all of it. An invented
            # default would read to the next operator as a considered budget.
            #
            # Refusing costs a launch. Not refusing costs the retention tier:
            # with a small staging host tier in front, this store holds the
            # only copy, and an unbounded store fills its filesystem until
            # `HiCacheFile.set` starts rolling back reservations on ENOSPC --
            # which the ack path reports as a partial backup, i.e. as a cache
            # miss. The capacity loss would be silent and permanent.
            raise ValueError(
                "--hicache-host-role staging requires a bounded file backend, "
                "and this one has no watermark: neither `max_size` nor "
                "`min_free_space` is set (checked in the per-backend "
                "extra_config first, then in "
                "SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE / "
                "SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE).\n"
                "Under 'staging' the pinned host tier is a small write-through "
                "buffer and THIS store is the retention tier, so an unbounded "
                "store is not a cache that grows -- it is the only copy of the "
                "cache, growing until the filesystem is full. At that point "
                "writes are refused page by page and the retention loss shows "
                "up as a hit-rate decay, not as an error.\n"
                'Set one, e.g. --hicache-storage-backend-extra-config \'{"max_size": '
                '"200G"}\' or \'{"min_free_space": "100G"}\'. No default is '
                "chosen for you: the right number depends on what else shares "
                "this filesystem, which this process cannot know."
            )
        self._eviction_enabled = self._eviction_configured and self._is_storage_owner
        if self._eviction_configured and not self._is_storage_owner:
            logger.info(
                f"HiCacheFile rank {self._tp_rank} (MLA): eviction handled by rank 0; "
                f"this rank skips LRU bookkeeping and will not create new files."
            )

        if not self._eviction_enabled:
            return

        self._clamp_max_size_to_fs()

        self._scan_existing_files()
        with self._lock:
            if self.max_size_bytes > 0 and self._total_bytes > self.max_size_bytes:
                self._evict_locked(0)
            if self.min_free_bytes > 0:
                self._enforce_free_space_locked(0)
        logger.info(
            f"HiCacheFile eviction enabled: cap={self.max_size_bytes} B "
            f"({self.max_size_scope} scope, {self._writer_count} writer ranks), "
            f"watermark={self.eviction_ratio:.2f}, min_free={self.min_free_bytes} B, "
            f"existing={self._total_bytes} B ({len(self._lru)} entries, "
            f"allocated bytes)"
        )

    def _load_config(self, extra: dict) -> None:
        # extra_config (per-backend) takes precedence over env vars.
        def _cfg(key, env):
            val = extra.get(key)
            return env.get() if val is None else val

        self.max_size_bytes = _parse_size_to_bytes(
            _cfg("max_size", envs.SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE)
        )
        self.min_free_bytes = _parse_size_to_bytes(
            _cfg("min_free_space", envs.SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE)
        )

        # max_size bounds the DIRECTORY, not one rank. Every TP rank builds its
        # own evictor over the same directory and only ever sees its own
        # suffixed files, so an undivided cap is spent once per rank: a TP=3 run
        # configured with 100Gi put ~294 GiB on disk (task #558). Split the cap
        # between the ranks that write here. "per_rank" restores the old
        # multiplying behaviour for anyone who wants a budget per rank.
        scope = extra.get("max_size_scope")
        scope = "shared" if scope is None else str(scope).strip().lower()
        if scope not in ("shared", "per_rank"):
            logger.warning(
                f"Unknown max_size_scope {scope!r} for HiCacheFile; using 'shared'."
            )
            scope = "shared"
        self.max_size_scope = scope
        if scope == "shared" and self._writer_count > 1 and self.max_size_bytes > 0:
            shared_total = self.max_size_bytes
            self.max_size_bytes //= self._writer_count
            logger.info(
                f"HiCacheFile max_size {shared_total} B is shared across "
                f"{self._writer_count} ranks writing to {self.file_path!r}: "
                f"{self.max_size_bytes} B for this rank. Pass "
                f'"max_size_scope": "per_rank" to budget each rank separately.'
            )

        ratio_raw = _cfg(
            "eviction_ratio", envs.SGLANG_HICACHE_FILE_BACKEND_EVICTION_RATIO
        )
        try:
            self.eviction_ratio = float(ratio_raw)
        except (TypeError, ValueError):
            self.eviction_ratio = 0.9
        if not (0.0 < self.eviction_ratio <= 1.0):
            self.eviction_ratio = 0.9

    def _clamp_max_size_to_fs(self) -> None:
        """Clamp max_size to the filesystem capacity so a too-large cap can't OOM tmpfs."""
        fs = self._fs_stats()
        if fs is not None and self.max_size_bytes > 0:
            safe_max = max(0, fs[0] - self.min_free_bytes)
            if self.max_size_bytes > safe_max:
                logger.warning(
                    f"HiCacheFile max_size exceeds filesystem capacity; "
                    f"clamping to {safe_max} B."
                )
                self.max_size_bytes = safe_max

    @property
    def write_stopped(self) -> bool:
        """True when the free-space watchdog has latched writes off."""
        return self._write_stopped

    @staticmethod
    def _allocated_size(st: os.stat_result) -> int:
        """Disk bytes a file occupies: the larger of its blocks and its length.

        The filesystem charges allocated blocks, not apparent length. On the
        incident filesystem (ZFS) the 512-byte ``.draft`` pages each occupied
        8704 bytes -- accounting them at apparent size undercounted real usage by
        17x, and there were 5.8 million of them. ``st_blocks`` is always in
        512-byte units, whatever the filesystem's own block size is.

        ``st_blocks`` alone is not enough: filesystems with delayed allocation
        (ZFS again) report a single block for a file that was just written and
        has not reached a transaction group yet. Never go below the payload
        length, so a fresh write is charged at least what its data must cost and
        the next scan of that file corrects it upward.
        """
        blocks = getattr(st, "st_blocks", None)
        if blocks is None:
            return st.st_size
        return max(int(blocks) * 512, st.st_size)

    def _stats_locked(self) -> dict:
        """``stats()`` for callers already holding ``_lock``."""
        return {
            "configured": self._eviction_configured,
            "enabled": self._eviction_enabled,
            "is_storage_owner": self._is_storage_owner,
            "max_size_bytes": self.max_size_bytes,
            "max_size_scope": self.max_size_scope,
            "writer_count": self._writer_count,
            "min_free_bytes": self.min_free_bytes,
            "eviction_ratio": self.eviction_ratio,
            "used_bytes": self._total_bytes,
            "num_entries": len(self._lru),
            "write_stopped": self._write_stopped,
            # #410 + the #715 lesson: never report as deliverable what the
            # actuator cannot deliver. Pinned bytes are held by conversation
            # checkpoints and eviction skips them, so a capacity decision must
            # read reclaimable_bytes rather than used_bytes.
            "pinned_entries": self._pins.pinned_entries() if self._pins else 0,
            "pinned_bytes": self._pins.pinned_bytes() if self._pins else 0,
            "reclaimable_bytes": max(
                0,
                self._total_bytes - (self._pins.pinned_bytes() if self._pins else 0),
            ),
            # #411 reconciliation: the overshoot the two accountings could
            # produce, reported rather than clamped away. It is 0 on this tree
            # because the evictor and the pin ledger now share ONE unit --
            # both charge `_allocated_size` -- but a non-zero value here means
            # they have diverged again, and a silent max(0, ...) would make an
            # incoherent ledger look like a full store.
            "accounting_overshoot_bytes": max(
                0,
                (self._pins.pinned_bytes() if self._pins else 0) - self._total_bytes,
            ),
        }

    def stats(self) -> dict:
        """Snapshot of the cap, the watermark, and current on-disk usage."""
        with self._lock:
            return self._stats_locked()

    def set_limits(
        self,
        *,
        max_size_bytes: Optional[int] = None,
        min_free_bytes: Optional[int] = None,
    ) -> dict:
        """Re-cap a live evictor. ``None`` leaves that limit unchanged.

        Growing only raises the ceiling: nothing is evicted and subsequent
        writes simply have more room. Shrinking evicts LRU victims inline --
        via the same ``_evict_locked`` path a write-time overflow uses, so the
        post-shrink target is ``max_size * eviction_ratio`` -- and returns once
        usage is back under the new cap. In-flight (reserved but uncommitted)
        writes are never evicted, so a shrink concurrent with a backup may
        legitimately land slightly above the target until those writes commit.

        Raising limits on an evictor that booted *unconfigured* switches
        eviction on. That evictor kept no LRU index (``reserve``/``touch`` were
        no-ops), so the index is seeded from disk here before the first
        eviction, exactly as ``__init__`` does.

        Non-owner MLA ranks record the new limits but stay inert: rank 0 owns
        the shared files and does all the evicting.
        """
        with self._lock:
            if max_size_bytes is not None:
                # Same directory-wide semantics as the boot-time cap.
                new_max = max(0, int(max_size_bytes))
                if self.max_size_scope == "shared" and self._writer_count > 1:
                    new_max //= self._writer_count
                self.max_size_bytes = new_max
            if min_free_bytes is not None:
                self.min_free_bytes = max(0, int(min_free_bytes))

            was_enabled = self._eviction_enabled
            self._eviction_configured = (
                self.max_size_bytes > 0 or self.min_free_bytes > 0
            )
            self._eviction_enabled = self._eviction_configured and (
                self._is_storage_owner
            )

            if not self._eviction_enabled:
                # Either the caller lifted the cap entirely (back to unbounded
                # storage) or this rank does not own the files. Nothing to evict.
                if was_enabled and not self._eviction_configured:
                    logger.info("HiCacheFile eviction disabled: storage is unbounded.")
                result = self._stats_locked()
                result["freed_bytes"] = 0
                return result

            self._clamp_max_size_to_fs()

            if not was_enabled and not self._lru:
                # Turning eviction on for the first time: adopt whatever this
                # rank already wrote while it was running unbounded.
                self._scan_existing_files()

            before = self._total_bytes
            if self.max_size_bytes > 0 and self._total_bytes > self.max_size_bytes:
                self._evict_locked(0)
            if self.min_free_bytes > 0:
                self._enforce_free_space_locked(0)
            else:
                # No watermark left to breach: drop any latched write stop so
                # stats() does not keep reporting a stop nothing can clear.
                self._write_stopped = False
            # Re-cap changed the watermark, so the watchdog's last verdict is
            # stale; let the next call probe instead of trusting its interval.
            self._last_free_probe = 0.0
            freed = before - self._total_bytes

            result = self._stats_locked()
            result["freed_bytes"] = freed

        logger.info(
            f"HiCacheFile eviction re-capped: cap={self.max_size_bytes} B, "
            f"min_free={self.min_free_bytes} B, freed={freed} B, "
            f"now {result['used_bytes']} B ({result['num_entries']} entries)"
        )
        return result

    @property
    def enabled(self) -> bool:
        """True when this rank actively evicts (configured AND storage owner)."""
        return self._eviction_enabled

    @property
    def configured(self) -> bool:
        """True when a cap or free-space watermark is set (on any rank)."""
        return self._eviction_configured

    @property
    def is_storage_owner(self) -> bool:
        """True when this rank owns (and may create/evict) the on-disk files."""
        return self._is_storage_owner

    def reserve(self, suffixed_key: str, value_bytes: int, key: str = "") -> bool:
        """Admit a new write of ``value_bytes``, evicting LRU victims as needed.

        On success the key is pre-reserved at MRU and flagged in-flight so a
        concurrent ``reserve`` won't evict it before the file is committed; the
        caller must then call ``commit`` (write landed) or ``abort`` (write
        failed). Returns ``False`` -- reserving nothing -- when the write is
        refused: this rank is not the storage owner, the value is larger than
        the cap, there is no evictable space, or the free-space watermark cannot
        be met. When eviction is not configured the write is always admitted.
        """
        if not self._eviction_configured:
            return True  # unbounded storage: nothing to enforce
        if not self._is_storage_owner:
            logger.warning(
                f"HiCacheFile rank {self._tp_rank} is not the MLA storage owner; "
                f"not caching new key {key} because file eviction is enabled."
            )
            return False
        if self.max_size_bytes > 0 and value_bytes > self.max_size_bytes:
            logger.warning(
                f"HiCacheFile: value {value_bytes} B exceeds cap "
                f"{self.max_size_bytes} B; not caching {key}"
            )
            return False
        # Latched by the watchdog: refuse cheaply and silently (it already said
        # so, once and loudly) rather than warning per page. Advisory only --
        # the authoritative watermark check is _enforce_free_space_locked below,
        # under the lock, so a page that races the latch is still refused there.
        if not self.check_free_space():
            with self._lock:
                self._refused_since_stop += 1
            return False

        with self._lock:
            # Cap-based eviction: evict, then bail if still over cap.
            if (
                self.max_size_bytes > 0
                and (self._total_bytes + value_bytes) > self.max_size_bytes
            ):
                self._evict_locked(value_bytes)
                if (self._total_bytes + value_bytes) > self.max_size_bytes:
                    logger.warning(
                        f"HiCacheFile: no evictable space for {value_bytes} B "
                        f"under cap {self.max_size_bytes} B; not caching {key}"
                    )
                    return False
            # Free-space watermark.
            if self.min_free_bytes > 0 and not self._enforce_free_space_locked(
                value_bytes
            ):
                logger.warning(
                    f"HiCacheFile: filesystem hosting {self.file_path!r} "
                    f"would fall below min_free={self.min_free_bytes} B "
                    f"after writing {value_bytes} B; refusing {key} "
                    f"to avoid OOM/ENOSPC."
                )
                return False
            # Pre-reserve at MRU so a concurrent evict won't grab this slot.
            prev = self._lru.pop(suffixed_key, None)
            if prev is not None:
                self._total_bytes -= prev
            self._lru[suffixed_key] = value_bytes
            self._pending_writes.add(suffixed_key)
            self._total_bytes += value_bytes
        return True

    def commit(self, suffixed_key: str) -> None:
        """Mark a reserved write as durably on disk (clears its in-flight flag).

        RECONCILE HERE, because ``reserve`` could only estimate. A reservation
        is taken before the file exists, so it charges the payload length --
        the only number available then. Once the write has landed the
        filesystem's own answer is available, and on a filesystem that
        allocates in blocks it is larger: a 64-byte page occupies 512 bytes on
        ZFS. Correcting at commit is what keeps ``_total_bytes`` in allocated
        units without making ``reserve`` stat a file that is not there yet.
        """
        if not self._eviction_enabled:
            return
        actual = None
        try:
            actual = self._allocated_size(os.stat(self._path_for_stem(suffixed_key)))
        except OSError:
            pass  # gone or unreadable; keep the reservation's estimate
        with self._lock:
            self._pending_writes.discard(suffixed_key)
            if actual is not None:
                prev = self._lru.get(suffixed_key)
                if prev is not None and prev != actual:
                    self._lru[suffixed_key] = actual
                    self._total_bytes += actual - prev

    def abort(self, suffixed_key: str) -> None:
        """Release a reservation whose write failed: drop it and refund the bytes."""
        if not self._eviction_enabled:
            return
        with self._lock:
            cur = self._lru.pop(suffixed_key, None)
            self._pending_writes.discard(suffixed_key)
            if cur is not None:
                self._total_bytes -= cur

    def touch(self, suffixed_key: str, tensor_path: str) -> None:
        """Mark key as MRU, adopting an untracked on-disk file if needed."""
        if not self._eviction_enabled:
            return
        with self._lock:
            if suffixed_key in self._lru:
                self._lru.move_to_end(suffixed_key, last=True)
                return
        # Untracked file: stat without holding the lock. Charged at its
        # ALLOCATED cost like every other entry -- adopting it at apparent
        # length would reintroduce the undercount one file at a time.
        try:
            size = self._allocated_size(os.stat(tensor_path))
        except OSError:
            return
        with self._lock:
            if suffixed_key in self._lru:
                self._lru.move_to_end(suffixed_key, last=True)
            else:
                self._lru[suffixed_key] = size
                self._total_bytes += size

    def clear(self) -> None:
        """Reset all bookkeeping after the backend has removed the files."""
        with self._lock:
            self._lru.clear()
            self._pending_writes.clear()
            self._total_bytes = 0

    def _fs_stats(self) -> Optional[tuple]:
        """(total, available) bytes for the filesystem; None if unavailable."""
        try:
            st = os.statvfs(self.file_path)
        except (OSError, AttributeError):
            return None
        frsize = st.f_frsize or st.f_bsize or 4096
        total = st.f_blocks * frsize
        free = st.f_bavail * frsize
        return total, free

    def _enforce_free_space_locked(self, value_bytes: int) -> bool:
        """Evict until writing value_bytes still leaves min_free_bytes free.
        Caller holds _lock. Returns False if the write can't be satisfied."""
        if self.min_free_bytes <= 0:
            return True
        fs = self._fs_stats()
        if fs is None:
            return True  # cannot probe -> permissive, fall back to OS errors
        # tmpfs frees space on unlink, so credit reclaimed bytes back to the
        # estimate rather than re-probing statvfs on every eviction.
        free = fs[1]
        self._evict_while(
            lambda reclaimed: (free + reclaimed) - value_bytes < self.min_free_bytes
        )
        # Re-probe: external writers may have changed free space meanwhile.
        fs = self._fs_stats()
        if fs is None:
            return True
        return fs[1] - value_bytes >= self.min_free_bytes

    def check_free_space(self, force: bool = False) -> bool:
        """Watchdog: police free space outside the write path. Returns writable.

        ``min_free_space`` used to be consulted only while admitting a write,
        so a filesystem that filled up (whether from this cache or from anything
        else sharing it) was never noticed while the backend sat idle, and the
        eventual notice was one ``warning`` per refused page. This probes
        ``statvfs`` at most every few seconds, tries to buy the space back by
        evicting this rank's own pages, and if that fails logs ONE error and
        latches writes off -- upstream then sees plain cache misses instead of a
        filling disk. The latch releases once free space recovers with margin.

        The latch is an advisory fast path, not the authority: ``reserve`` still
        runs ``_enforce_free_space_locked`` under the lock for every admitted
        page, so a page that races a latch being set is refused there anyway.
        The lock is dropped across the ``statvfs`` probe so a slow filesystem
        cannot block writers.
        """
        if self.min_free_bytes <= 0 or not self._eviction_enabled:
            return True
        with self._lock:
            now = time.monotonic()
            if (
                not force
                and (now - self._last_free_probe) < _FREE_SPACE_PROBE_INTERVAL_S
            ):
                return not self._write_stopped
            self._last_free_probe = now

        fs = self._fs_stats()
        if fs is None:
            # Cannot probe: leave the latch as it is.
            with self._lock:
                return not self._write_stopped
        free = fs[1]

        if free >= self.min_free_bytes:
            with self._lock:
                self._release_write_stop_locked(free)
                # Inside the hysteresis band (above min_free, below the recovery
                # margin) the latch is still set and writes stay refused.
                return not self._write_stopped

        # Below the watermark: first try to buy the space back from our own LRU.
        with self._lock:
            self._enforce_free_space_locked(0)
            fs = self._fs_stats()
            free = fs[1] if fs is not None else free
            if free >= self.min_free_bytes:
                self._release_write_stop_locked(free)
                return True
            if not self._write_stopped:
                self._write_stopped = True
                self._refused_since_stop = 0
                logger.error(
                    f"HiCacheFile STOPPING WRITES: the filesystem hosting "
                    f"{self.file_path!r} has {free} B free, below min_free="
                    f"{self.min_free_bytes} B, and evicting this rank's "
                    f"{len(self._lru)} cached pages ({self._total_bytes} B) did "
                    f"not recover it. New pages will be a cache miss instead of "
                    f"filling the disk. Free space on that filesystem, or lower "
                    f"max_size / min_free_space."
                )
        return False

    def _release_write_stop_locked(self, free: int) -> None:
        """Clear a latched write stop once free space recovered with margin."""
        if not self._write_stopped:
            return
        if free < self.min_free_bytes * _FREE_SPACE_RECOVERY_FACTOR:
            return
        refused = self._refused_since_stop
        self._write_stopped = False
        self._refused_since_stop = 0
        logger.warning(
            f"HiCacheFile resuming writes: {free} B free on the filesystem "
            f"hosting {self.file_path!r} is back above min_free="
            f"{self.min_free_bytes} B ({refused} pages were missed while stopped)."
        )

    def _iter_existing_flat(self) -> Iterable[Tuple[str, os.stat_result]]:
        """(stem, stat) for every ``.bin`` directly in the storage directory."""
        try:
            names = os.listdir(self.file_path)
        except FileNotFoundError:
            return
        for fn in names:
            if not fn.endswith(".bin"):
                continue
            try:
                st = os.stat(os.path.join(self.file_path, fn))
            except OSError:
                continue
            yield fn[:-4], st

    def _scan_existing_files(self) -> None:
        """Seed LRU index from disk on startup (oldest mtime first)."""
        entries = []
        for stem, st in self._iter_existing():
            # Only files belonging to this rank/model.
            if not stem.endswith(self.config_suffix):
                continue
            entries.append((st.st_mtime, stem, self._allocated_size(st)))
        entries.sort(key=lambda e: e[0])  # oldest first
        for _, stem, size in entries:
            self._lru[stem] = size
            self._total_bytes += size

    def _path_for_stem(self, stem: str) -> str:
        """The file this stem names. One join, so the evictor, the
        accounting and the pin ledger all stat the same path."""
        return os.path.join(self.file_path, f"{stem}.bin")


    def _evict_one_lru_locked(self) -> Tuple[str, int]:
        """Evict the single oldest evictable LRU entry. Caller holds _lock.

        The shared pop / skip-pending / unlink / ``_total_bytes`` step driven by
        `_evict_while`. Returns ``(outcome, freed_bytes)``:

        - ``("evicted", n)``: oldest entry dropped from the index; ``n`` disk
          bytes reclaimed (0 if the file was already gone).
        - ``("skipped", 0)``: oldest entry is an in-flight write; re-pinned at MRU
          so the writer is not evicted out from under itself.
        - ``("stop", 0)``: nothing evictable (empty index) or the unlink failed
          (entry re-pinned at LRU); the caller should stop its eviction loop.
        """
        if not self._lru:
            return "stop", 0
        evict_stem, evict_size = self._lru.popitem(last=False)  # oldest
        if evict_stem in self._pending_writes:
            # Keep in-flight reservations; their file isn't committed yet.
            self._lru[evict_stem] = evict_size
            return "skipped", 0
        if self._pins is not None and self._pins.is_pinned(evict_stem):
            # #410: a pinned page belongs to a conversation checkpoint that
            # paid to keep it. Same skip-and-repin an in-flight write gets, and
            # deliberately so: a store whose entries are ALL pinned cannot spin
            # here -- it exhausts the caller's attempts and the caller learns
            # the space is not there, rather than looping forever.
            self._lru[evict_stem] = evict_size
            return "skipped", 0
        # The INJECTED resolver, not a flat join: this backend hands the
        # evictor `path_for_stem=self._existing_path`, which knows the sharded
        # layout. A flat join here misses the file, `os.remove` fails, the
        # entry leaves the LRU and the bytes stay on disk -- the cap silently
        # stops holding. Line 466 already stats through the same resolver;
        # these two must not disagree.
        tensor_path = self._path_for_stem(evict_stem)
        try:
            os.remove(tensor_path)
            freed = evict_size
            if self._on_evict is not None:
                self._on_evict(evict_stem)
        except FileNotFoundError:
            freed = 0  # file already gone; still drop the stale index entry
            if self._on_evict is not None:
                self._on_evict(evict_stem)
        except OSError as e:
            logger.warning(f"HiCacheFile eviction failed for {evict_stem}: {e}")
            self._lru[evict_stem] = evict_size
            self._lru.move_to_end(evict_stem, last=False)
            return "stop", 0
        self._total_bytes -= evict_size
        return "evicted", freed

    def _evict_while(self, should_continue) -> int:
        """Evict oldest non-pending entries while ``should_continue(reclaimed)``.

        ``should_continue`` is passed the disk bytes reclaimed so far and returns
        whether to keep evicting. In-flight writes are skipped; the loop is bounded
        so it can't spin once every remaining entry is pending. Caller holds _lock.
        Returns the total disk bytes reclaimed.
        """
        reclaimed = 0
        attempts_left = len(self._lru)
        while self._lru and attempts_left > 0 and should_continue(reclaimed):
            outcome, freed = self._evict_one_lru_locked()
            if outcome == "stop":
                break
            if outcome == "skipped":
                attempts_left -= 1
                continue
            # An entry left the index; reset the skip budget and bank the bytes.
            reclaimed += freed
            attempts_left = len(self._lru)
        return reclaimed

    def _evict_locked(self, needed_bytes: int) -> None:
        """Evict LRU entries until total + needed <= cap*ratio. Caller holds _lock."""
        if self.max_size_bytes <= 0:
            return
        target = max(0, int(self.max_size_bytes * self.eviction_ratio) - needed_bytes)
        reclaimed = self._evict_while(lambda _: self._total_bytes > target)
        if reclaimed:
            logger.debug(
                f"HiCacheFile reclaimed {reclaimed} bytes; "
                f"now {self._total_bytes} bytes used"
            )
