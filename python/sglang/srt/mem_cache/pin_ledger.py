"""#410 slice 2: pins that keep a checkpoint branchable, and their honest cost.

Slice 1 verified a checkpoint's references before branching and refused when
any had been evicted. That is the correct failure, but it is still a failure:
the store is an LRU cache and a conversation checkpoint is not cache-shaped --
its value is precisely that it stays restorable long after its pages stopped
being hot. So a written checkpoint PINS what it references.

WHAT A PIN COSTS, and why the ledger is part of the feature rather than
instrumentation. Pinned bytes are bytes eviction can no longer reclaim. An
eviction path that counts them as reclaimable will promise space it cannot
deliver, then fail at the unlink instead of at the decision -- the #715 lesson.
So:

* eviction SKIPS pinned entries (the same skip-and-repin step in-flight writes
  already use, so the bounded skip budget applies unchanged and a store of
  nothing but pins cannot spin);
* the TTL sweeper skips a pinned page's partial too, because a checkpoint may
  reference a page another stage is still assembling;
* ``stats`` reports pinned entries, pinned bytes and RECLAIMABLE bytes
  separately, so the number a capacity decision reads is the one the actuator
  can actually deliver;
* creating a checkpoint whose pins would exceed the budget is REFUSED with the
  numbers, because a cache degraded into a pin museum serves nobody and does it
  silently.

Pins are REF-COUNTED by checkpoint: two checkpoints sharing a prefix (the
common case -- branching is why this exists) share its pages, and unpinning one
must not drop the other's. They are also DURABLE: a checkpoint outlives the
process, so a pin that only lived in memory would silently stop protecting
anything at the next restart.

ORPHANS. A pin whose manifest is gone protects nothing and blocks eviction
forever -- the same leak shape as an abandoned ``.part706``, and reaped the
same way: by age, loudly, against the authority that knows whether the
checkpoint still exists.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
import time
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "[#410 pins]"

PIN_DIR = "pins706"
PIN_SUFFIX = ".pin.json"


class PinBudgetExceeded(RuntimeError):
    """Refused: this checkpoint's pins would exceed the configured budget."""

    def __init__(
        self, *, checkpoint_id: str, want: int, held: int, budget: int
    ) -> None:
        self.want, self.held, self.budget = int(want), int(held), int(budget)
        super().__init__(
            f"{LOG_PREFIX} refusing checkpoint {checkpoint_id!r}: pinning it "
            f"needs {want} more bytes on top of {held} already pinned, which "
            f"exceeds the {budget}-byte pin budget by {held + want - budget}. "
            "Pinned bytes are bytes eviction can never reclaim, so accepting "
            "this would degrade the cache into a pin museum -- silently, which "
            "is the part worth refusing. Delete a checkpoint or raise the "
            "budget."
        )


@dataclasses.dataclass(frozen=True)
class PinResult:
    checkpoint_id: str
    stems: tuple[str, ...]
    bytes_added: int
    bytes_shared: int  # already pinned by another checkpoint
    #: References the CALLER asked to pin that were not pinned, because their
    #: file was not there when the sizes were taken (``stems_with_sizes``
    #: drops those deliberately -- see its docstring). Empty on the normal
    #: path. Carried here so a caller can compare what it asked for against
    #: what it got; without it a partial pin is indistinguishable from a whole
    #: one, and the shortfall only shows up at the branch that needed it.
    unpinned: tuple[str, ...] = ()


class PinLedger:
    """Which store stems are pinned, by which checkpoints, at what cost."""

    def __init__(self, root_dir: str, *, budget_bytes: int = 0) -> None:
        self.root_dir = root_dir
        self.budget_bytes = int(budget_bytes)
        self._lock = threading.RLock()
        # stem -> {checkpoint_id}; and stem -> size, so pinned_bytes never
        # re-stats a store that may hold millions of entries.
        self._by_stem: dict[str, set[str]] = {}
        self._sizes: dict[str, int] = {}
        self._by_checkpoint: dict[str, set[str]] = {}
        self._created: dict[str, float] = {}

    # -- durability ---------------------------------------------------------

    @property
    def pin_dir(self) -> str:
        return os.path.join(self.root_dir, PIN_DIR)

    def _pin_path(self, checkpoint_id: str) -> str:
        return os.path.join(self.pin_dir, f"{checkpoint_id}{PIN_SUFFIX}")

    def load(self) -> int:
        """Rebuild the index from disk. Returns the number of checkpoints."""
        with self._lock:
            self._by_stem.clear()
            self._sizes.clear()
            self._by_checkpoint.clear()
            self._created.clear()
            try:
                names = os.listdir(self.pin_dir)
            except FileNotFoundError:
                return 0
            for name in names:
                if not name.endswith(PIN_SUFFIX):
                    continue
                path = os.path.join(self.pin_dir, name)
                try:
                    with open(path, "r") as f:
                        raw = json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    # A pin file we cannot read protects nothing and we cannot
                    # tell what. Loud, and left on disk for inspection rather
                    # than deleted underneath an operator.
                    logger.warning("%s unreadable pin file %s: %s", LOG_PREFIX, path, e)
                    continue
                self._install(
                    raw.get("checkpoint_id") or name[: -len(PIN_SUFFIX)],
                    {k: int(v) for k, v in (raw.get("stems") or {}).items()},
                    float(raw.get("created_at") or 0.0),
                )
            return len(self._by_checkpoint)

    def _install(self, checkpoint_id: str, sizes: dict, created_at: float) -> None:
        self._by_checkpoint[checkpoint_id] = set(sizes)
        self._created[checkpoint_id] = created_at
        for stem, size in sizes.items():
            self._by_stem.setdefault(stem, set()).add(checkpoint_id)
            self._sizes[stem] = int(size)

    def _persist(self, checkpoint_id: str, sizes: dict) -> None:
        os.makedirs(self.pin_dir, exist_ok=True)
        path = self._pin_path(checkpoint_id)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(
                {
                    "checkpoint_id": checkpoint_id,
                    "stems": sizes,
                    "created_at": self._created.get(checkpoint_id, time.time()),
                },
                f,
                sort_keys=True,
            )
        os.replace(tmp, path)

    # -- the pin itself -----------------------------------------------------

    def pin(self, checkpoint_id: str, sizes_by_stem: dict) -> PinResult:
        """Pin every stem this checkpoint references. Refuses over budget.

        ``sizes_by_stem`` maps the STORE stem (the suffixed key the evictor
        indexes) to its size, so the ledger never has to guess either.
        """
        with self._lock:
            sizes = {str(k): int(v) for k, v in sizes_by_stem.items()}
            new_bytes = sum(
                size for stem, size in sizes.items() if stem not in self._by_stem
            )
            shared = sum(size for stem, size in sizes.items() if stem in self._by_stem)
            if self.budget_bytes > 0:
                held = self._pinned_bytes_locked()
                if held + new_bytes > self.budget_bytes:
                    raise PinBudgetExceeded(
                        checkpoint_id=checkpoint_id,
                        want=new_bytes,
                        held=held,
                        budget=self.budget_bytes,
                    )
            self._created.setdefault(checkpoint_id, time.time())
            self._install(checkpoint_id, sizes, self._created[checkpoint_id])
            self._persist(checkpoint_id, sizes)
            logger.info(
                "%s pinned %d stem(s) for checkpoint %s (+%d B new, %d B shared).",
                LOG_PREFIX,
                len(sizes),
                checkpoint_id,
                new_bytes,
                shared,
            )
            return PinResult(
                checkpoint_id=checkpoint_id,
                stems=tuple(sorted(sizes)),
                bytes_added=new_bytes,
                bytes_shared=shared,
            )

    def unpin(self, checkpoint_id: str) -> int:
        """Release a checkpoint's pins. Returns the bytes actually freed.

        Shared stems stay pinned for their other holders -- branching exists to
        share prefixes, so dropping a sibling's protection here would make the
        feature eat itself.
        """
        with self._lock:
            stems = self._by_checkpoint.pop(checkpoint_id, set())
            self._created.pop(checkpoint_id, None)
            freed = 0
            for stem in stems:
                holders = self._by_stem.get(stem)
                if holders is None:
                    continue
                holders.discard(checkpoint_id)
                if not holders:
                    del self._by_stem[stem]
                    freed += self._sizes.pop(stem, 0)
            try:
                os.remove(self._pin_path(checkpoint_id))
            except FileNotFoundError:
                pass
            return freed

    # -- what the eviction path asks ----------------------------------------

    def is_pinned(self, stem: str) -> bool:
        with self._lock:
            return stem in self._by_stem

    def _pinned_bytes_locked(self) -> int:
        return sum(self._sizes.values())

    def pinned_bytes(self) -> int:
        with self._lock:
            return self._pinned_bytes_locked()

    def pinned_entries(self) -> int:
        with self._lock:
            return len(self._by_stem)

    def ledger(self) -> dict:
        """Per-checkpoint visibility: count and bytes, plus the totals."""
        with self._lock:
            per = {}
            for checkpoint_id, stems in self._by_checkpoint.items():
                per[checkpoint_id] = {
                    "entries": len(stems),
                    "bytes": sum(self._sizes.get(s, 0) for s in stems),
                    "created_at": self._created.get(checkpoint_id, 0.0),
                }
            return {
                "checkpoints": per,
                "pinned_entries": len(self._by_stem),
                "pinned_bytes": self._pinned_bytes_locked(),
                "budget_bytes": self.budget_bytes,
            }

    # -- orphans ------------------------------------------------------------

    def reap_orphans(
        self,
        manifest_exists: Callable[[str], bool],
        *,
        older_than_s: float,
        now: Optional[Callable[[], float]] = None,
    ) -> tuple[str, ...]:
        """Drop pins whose checkpoint is gone. Returns the reaped ids.

        Age-gated for the same reason the ``.part706`` sweeper is: a checkpoint
        being written right now has pins before it has a manifest, and reaping
        those would delete the protection at the exact moment it is needed.
        """
        clock = now or time.time
        cutoff = clock() - float(older_than_s)
        with self._lock:
            candidates = [
                cid
                for cid, created in self._created.items()
                if created <= cutoff and not manifest_exists(cid)
            ]
        reaped = []
        for cid in candidates:
            freed = self.unpin(cid)
            reaped.append(cid)
            logger.warning(
                "%s reaped orphan pin set %s (%d B freed): it protected pages "
                "for a checkpoint whose manifest no longer exists, so it was "
                "blocking eviction for nothing.",
                LOG_PREFIX,
                cid,
                freed,
            )
        return tuple(reaped)

    def checkpoints(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._by_checkpoint))


def stems_with_sizes(paths: Iterable[tuple[str, str]]) -> dict:
    """``{stem: allocated size}`` for (stem, path) pairs that exist.

    ALLOCATED, not apparent, because the evictor accounts allocated bytes
    (``LRUFileEvictor._allocated_size``: the incident filesystem charged 8704
    bytes for a 512-byte page, a 17x undercount over 5.8M files). Charging the
    ledger in a different unit than the evictor would make
    ``reclaimable = used - pinned`` subtract centimetres from inches -- the
    exact shape of the #715 lesson this slice is supposed to honour.

    Missing files are dropped rather than pinned at size 0: pinning a page that
    is not there would charge the budget for protection nobody gets.
    """
    from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

    out = {}
    for stem, path in paths:
        try:
            out[stem] = int(LRUFileEvictor._allocated_size(os.stat(path)))
        except OSError:
            continue
    return out
