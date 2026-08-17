"""#545: attach, resize and detach the HiCache disk tier AT RUNTIME.

Today an L3 directory and its size budget are boot-time facts: changing either
means a restart, which on this rig costs a full model load and a warm cache.
This module makes all three operations live, behind a flag.

**Three rules, and each exists because its violation is silent.**

1. **Shrink EVICTS DOWN to the new bound; it never truncates.** A smaller
   budget must remove whole pages through the evictor, not cut bytes off files
   that readers still hold. A truncated page is not a miss — it is a page that
   reads back short and wrong, which no cache-hit metric would show.
2. **Detach is REFUSED while any page is only-copy here.** Detaching a tier
   that holds the sole copy of a page silently destroys it; the request that
   later wants it takes a miss it cannot explain. Refusing is loud and
   recoverable, and `force=True` exists for an operator who has decided the
   loss is acceptable.
3. **A shrink that cannot be honoured without truncation is refused**, not
   partially applied. A bound the tier cannot reach is a configuration error;
   accepting it and quietly staying above it would make the reported capacity a
   lie.

**It coordinates with the existing file backend rather than replacing it.**
The controller owns the *budget and lifecycle*; the bytes stay with
`HiCacheStorage`/`HiCacheFile`. When the #706 canonical-page store lands it
shares that same backend and directory layout, so the two do not need to agree
about anything here — this file deliberately introduces no second layout.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)


class DiskTierError(RuntimeError):
    """A disk-tier operation that cannot be honoured."""


class DetachRefused(DiskTierError):
    """Detach would destroy the only copy of live pages."""


class ShrinkRefused(DiskTierError):
    """The requested bound cannot be reached by eviction alone."""


@dataclasses.dataclass(frozen=True)
class TierState:
    path: str
    capacity_bytes: int
    used_bytes: int
    page_count: int

    @property
    def free_bytes(self) -> int:
        return max(0, self.capacity_bytes - self.used_bytes)

    @property
    def over_bound_bytes(self) -> int:
        return max(0, self.used_bytes - self.capacity_bytes)


@dataclasses.dataclass(frozen=True)
class ResizeReport:
    old_capacity: int
    new_capacity: int
    evicted_keys: tuple[str, ...]
    freed_bytes: int
    used_after: int

    @property
    def shrank(self) -> bool:
        return self.new_capacity < self.old_capacity


class DiskTierController:
    """Runtime lifecycle for one L3 directory.

    Collaborators are injected so the whole object is exercisable on CPU,
    including the refusal paths a live tier reaches only under pressure:

    ``evict_fn(nbytes) -> Sequence[str]``  evict at least ``nbytes`` worth of
                                          WHOLE pages, returning their keys
    ``size_fn(key) -> int``               bytes a page occupies
    ``only_copy_fn(key) -> bool``         True when this tier holds the sole
                                          copy of that page
    """

    def __init__(
        self,
        enabled: bool = False,
        evict_fn: Callable[[int], Sequence[str]] | None = None,
        size_fn: Callable[[str], int] | None = None,
        only_copy_fn: Callable[[str], bool] | None = None,
    ):
        self.enabled = bool(enabled)
        self._evict_fn = evict_fn
        self._size_fn = size_fn or (lambda _k: 0)
        self._only_copy_fn = only_copy_fn or (lambda _k: False)
        self._path: str | None = None
        self._capacity: int = 0
        self._pages: dict[str, int] = {}

    # -- state ------------------------------------------------------------
    @property
    def attached(self) -> bool:
        return self._path is not None

    def state(self) -> TierState:
        if not self.attached:
            raise DiskTierError("no disk tier is attached.")
        used = sum(self._pages.values())
        return TierState(self._path, self._capacity, used, len(self._pages))

    def _require(self) -> None:
        if not self.enabled:
            raise DiskTierError(
                "runtime disk-tier control is disabled. It is flag-gated "
                "because attaching or resizing a live L3 changes what the "
                "evictor is allowed to delete."
            )
        if not self.attached:
            raise DiskTierError("no disk tier is attached.")

    # -- lifecycle --------------------------------------------------------
    def attach(self, path: str, capacity_bytes: int) -> TierState:
        """Bring a directory in as L3 while serving continues."""
        if not self.enabled:
            raise DiskTierError(
                "runtime disk-tier control is disabled; attach at boot instead."
            )
        if self.attached:
            raise DiskTierError(
                f"a tier is already attached at {self._path!r}. Detach it "
                "first: two directories under one controller would give the "
                "evictor two budgets and one bound."
            )
        if capacity_bytes <= 0:
            raise DiskTierError("capacity must be positive.")
        self._path = str(path)
        self._capacity = int(capacity_bytes)
        self._pages = {}
        logger.info(
            "hicache disk tier attached at %s with %d bytes", self._path, self._capacity
        )
        return self.state()

    def detach(self, force: bool = False) -> None:
        """Drop the tier. REFUSED while it holds the only copy of live pages."""
        self._require()
        sole = [k for k in self._pages if self._only_copy_fn(k)]
        if sole and not force:
            raise DetachRefused(
                f"refusing to detach {self._path!r}: {len(sole)} page(s) exist "
                f"ONLY here (e.g. {sole[0]!r}). Detaching would destroy them, "
                "and the requests that later want them would take misses with "
                "no cause visible anywhere. Replicate them first, or pass "
                "force=True to accept the loss deliberately."
            )
        if sole:
            logger.warning(
                "hicache disk tier %s force-detached with %d only-copy page(s); "
                "those pages are now gone",
                self._path,
                len(sole),
            )
        self._path = None
        self._capacity = 0
        self._pages = {}

    def resize(self, capacity_bytes: int) -> ResizeReport:
        """Grow or shrink the budget. Shrink evicts; it never truncates."""
        self._require()
        if capacity_bytes <= 0:
            raise DiskTierError("capacity must be positive.")
        old = self._capacity
        new = int(capacity_bytes)
        used = sum(self._pages.values())

        if new >= used:
            # Growing, or shrinking to a bound the content already respects.
            self._capacity = new
            return ResizeReport(old, new, (), 0, used)

        need = used - new
        if self._evict_fn is None:
            raise ShrinkRefused(
                f"cannot shrink {self._path!r} to {new} bytes: {need} bytes "
                "over the bound and no evictor is wired. Refusing rather than "
                "truncating -- a truncated page reads back short and wrong, "
                "which no cache-hit metric would reveal."
            )
        evicted = tuple(self._evict_fn(need))
        freed = 0
        for key in evicted:
            freed += self._pages.pop(key, 0)
        used_after = sum(self._pages.values())
        if used_after > new:
            raise ShrinkRefused(
                f"cannot shrink {self._path!r} to {new} bytes: the evictor "
                f"freed {freed} of the {need} needed and {used_after} bytes "
                "remain. Refusing to report a bound the tier is not under; "
                "the remaining pages would have to be truncated to reach it."
            )
        self._capacity = new
        return ResizeReport(old, new, evicted, freed, used_after)

    # -- accounting -------------------------------------------------------
    def note_write(self, key: str, nbytes: int | None = None) -> None:
        self._require()
        self._pages[str(key)] = int(
            nbytes if nbytes is not None else self._size_fn(key)
        )

    def note_evict(self, key: str) -> None:
        self._require()
        self._pages.pop(str(key), None)

    def snapshot(self) -> dict:
        """Flat shape for an admin/monitoring endpoint."""
        if not self.attached:
            return {"disk_tier_attached": 0, "disk_tier_enabled": int(self.enabled)}
        st = self.state()
        return {
            "disk_tier_enabled": int(self.enabled),
            "disk_tier_attached": 1,
            "disk_tier_path": st.path,
            "disk_tier_capacity_bytes": st.capacity_bytes,
            "disk_tier_used_bytes": st.used_bytes,
            "disk_tier_free_bytes": st.free_bytes,
            "disk_tier_pages": st.page_count,
        }
