# SPDX-License-Identifier: Apache-2.0
"""Between-tick executor for the GDN resident-slot ladder (#364 slice 2).

Slice 1 shipped the three pieces a slot ladder needs and stopped one step
short of the only thing a user can observe: :func:`vacate_plan` DECIDES which
idle session should give up its state slot, ``MambaPool.export_state_blob`` /
``import_state_blob`` can carry that state out of VRAM and back bit-identical,
and ``--gdn-resident-state-slots`` shrinks the pool at boot. Nothing connected
them, so a capped server planned a vacate every round and still refused the
session the plan was meant to admit. This module is that connection.

WHY IT IS A SEPARATE, EXPLICITLY WINDOWED OBJECT
------------------------------------------------
The pool tensors are addressed by captured CUDA graphs. Freeing a slot and
overwriting it with another session's state while a graph could replay is the
corruption class of #52/#53 -- and it is silent: wrong recurrent state is
plausible output, not a crash. So the executor does not merely *happen* to be
called from a safe place, it REFUSES to run anywhere else:
:meth:`GdnSlotExecutor.run` raises :class:`GdnSlotWindowError` unless it is
inside :meth:`between_ticks`, which the scheduler arms at the same point the
#287 pressure ladder uses for its geometry flips (scheduler.py, after the
batch of the previous tick is retired and before the next one is selected).
A future caller that reaches for this from inside a forward gets a named
error at the first attempt instead of a corrupt state a week later.

ORDER OF OPERATIONS, AND WHY
----------------------------
Vacates run BEFORE restores. ``vacate_plan`` already sizes the vacate list to
cover the restores plus the overshoot, so running them in this order means a
restore never has to wait for an allocator that a later vacate would have
fed. The reverse order deadlocks a full pool: every restore needs a slot, and
the only slots are the ones the vacates have not released yet.

THE BLOB'S ROUTE (#212, #224)
-----------------------------
A vacated state travels as an OPAQUE session blob. It never takes the HiCache
radix route: a recurrent state is positional, not prefix-shareable, and a
radix truncation of it is a silent full re-prefill (#212). :class:`BlobStore`
is the seam. The default :class:`LocalGdnBlobStore` keeps the exported CPU
tensors in process memory -- which IS the ``local`` tier of #224 and the only
one the physics allows first, since every further hop stages through locally
pinned host RAM. :class:`TieredGdnBlobStore` puts the same blob on a #224
``DestinationTier`` by flattening it to one byte buffer plus a manifest, so a
capped server can push GDN state past host RAM without a second serializer.

Note for readers of #224: that module states GDN state "never travels" as its
own invariant. That was true of the KV-tail park path it describes and stays
true there -- it parks a KV tail and deliberately leaves the mamba pool
alone. #364 is the other axis: it relieves the mamba pool itself, and the
blob it moves is produced and consumed by the pool's own complete round-trip.

Nothing here imports torch at module level: the planner, the window guard and
the manifest arithmetic are testable on CPU with no CUDA present.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence

logger = logging.getLogger(__name__)

LOG_PREFIX = "GDN-SLOT-LADDER"


class GdnSlotWindowError(RuntimeError):
    """Raised when the executor is driven outside the between-tick window.

    Not a defensive nicety: the pool slots this executor rewrites are
    addressed by captured graphs, so "called from the wrong place" is the
    #52/#53 silent-corruption class. It fails loudly at the call instead.
    """


class GdnSlotError(RuntimeError):
    """A vacate or restore could not be completed. Always names the session
    and what was missing -- a shortfall that reports only a number is the
    thing slice 1's ``skipped`` map exists to prevent."""


class ExecResult(NamedTuple):
    """What one between-tick round actually moved (not what it planned).

    ``vacated`` / ``restored`` are session ids that completed; ``failed``
    maps a session id to why it did not, so the caller logs causes rather
    than a count.
    """

    vacated: List[str]
    restored: List[str]
    failed: Dict[str, str]

    @property
    def moved(self) -> int:
        return len(self.vacated) + len(self.restored)


# ---------------------------------------------------------------------------
# Blob stores
# ---------------------------------------------------------------------------


class BlobStore:
    """Where a vacated GDN state waits. Three methods, no lifecycle."""

    def put(self, session_id: str, blob: Any) -> None:
        raise NotImplementedError

    def pop(self, session_id: str) -> Any:
        raise NotImplementedError

    def has(self, session_id: str) -> bool:
        raise NotImplementedError

    def drop(self, session_id: str) -> None:
        """Forget a blob whose session died while parked. Never raises."""
        raise NotImplementedError


class LocalGdnBlobStore(BlobStore):
    """The default: the exported CPU tensors, held in process memory.

    ``export_state_blob`` already produces host tensors, so this tier costs
    one dict entry and no further copy. It is the ``local`` tier of #224 by
    construction -- the D2H landing zone every other tier stages through.
    """

    def __init__(self) -> None:
        self._blobs: Dict[str, Any] = {}

    def put(self, session_id: str, blob: Any) -> None:
        self._blobs[str(session_id)] = blob

    def pop(self, session_id: str) -> Any:
        return self._blobs.pop(str(session_id))

    def has(self, session_id: str) -> bool:
        return str(session_id) in self._blobs

    def drop(self, session_id: str) -> None:
        self._blobs.pop(str(session_id), None)

    def __len__(self) -> int:
        return len(self._blobs)


def blob_manifest(blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Describe a state blob so it can be rebuilt from flat bytes.

    One entry per leaf tensor, in a STABLE order (sorted by field name, then
    list position). The manifest is what makes the flat form self-checking:
    ``unflatten_blob`` refuses a buffer whose manifest does not match the
    pool's own field set, the same refusal ``import_state_blob`` makes on the
    structured form.
    """
    manifest: List[Dict[str, Any]] = []
    for name in sorted(blob):
        value = blob[name]
        items = value if isinstance(value, (list, tuple)) else [value]
        for pos, t in enumerate(items):
            manifest.append(
                {
                    "name": name,
                    "pos": pos if isinstance(value, (list, tuple)) else -1,
                    "shape": list(t.shape),
                    "dtype": str(t.dtype).replace("torch.", ""),
                    "numel": int(t.numel()),
                    "itemsize": int(t.element_size()),
                }
            )
    return manifest


def flatten_blob(blob: Dict[str, Any]):
    """(manifest, one contiguous uint8 CPU tensor) for a #224 tier put."""
    import torch

    manifest = blob_manifest(blob)
    parts = []
    for entry in manifest:
        value = blob[entry["name"]]
        t = value[entry["pos"]] if entry["pos"] >= 0 else value
        parts.append(t.contiguous().view(torch.uint8).reshape(-1))
    flat = torch.cat(parts) if parts else torch.zeros(0, dtype=torch.uint8)
    return manifest, flat


def unflatten_blob(manifest: Sequence[Dict[str, Any]], flat) -> Dict[str, Any]:
    """Inverse of :func:`flatten_blob`, by manifest, never by position alone."""
    import torch

    out: Dict[str, Any] = {}
    offset = 0
    for entry in manifest:
        nbytes = int(entry["numel"]) * int(entry["itemsize"])
        chunk = flat[offset : offset + nbytes]
        if chunk.numel() != nbytes:
            raise GdnSlotError(
                f"flat GDN blob is short at field {entry['name']!r}: needed "
                f"{nbytes} B, buffer has {chunk.numel()} B. A partially "
                "restored recurrent state is silently wrong output."
            )
        offset += nbytes
        dtype = getattr(torch, entry["dtype"])
        t = chunk.view(dtype).reshape(entry["shape"])
        if entry["pos"] < 0:
            out[entry["name"]] = t
        else:
            out.setdefault(entry["name"], []).append(t)
    return out


class TieredGdnBlobStore(BlobStore):
    """Park the blob on a #224 :class:`DestinationTier`.

    The tier moves one flat byte buffer per session; the manifest stays in
    process (it is a few hundred bytes and is what validates the buffer on
    the way back). A tier that refuses a put is not a silent loss: the blob
    stays local and the store says so, which is the same failover the
    destination controller does for KV tails.
    """

    def __init__(self, tier, key_fn: Callable[[str], str]):
        self._tier = tier
        self._key_fn = key_fn
        self._manifests: Dict[str, Any] = {}
        self._local = LocalGdnBlobStore()

    def put(self, session_id: str, blob: Any) -> None:
        sid = str(session_id)
        try:
            manifest, flat = flatten_blob(blob)
            if self._tier.put(self._key_fn(sid), flat):
                self._manifests[sid] = (manifest, flat.numel())
                return
        except Exception as exc:  # noqa: BLE001 -- tier failure = stay local
            logger.warning(
                "%s tier put failed for session %s (%r); the blob stays in "
                "the local tier",
                LOG_PREFIX,
                sid,
                exc,
            )
        self._local.put(sid, blob)

    def pop(self, session_id: str) -> Any:
        import torch

        sid = str(session_id)
        if self._local.has(sid):
            return self._local.pop(sid)
        manifest, nbytes = self._manifests.pop(sid)
        buf = torch.zeros(nbytes, dtype=torch.uint8)
        if not self._tier.get_into(self._key_fn(sid), buf):
            raise GdnSlotError(
                f"GDN state blob for session {sid} is not readable from tier "
                f"{getattr(self._tier, 'name', '?')!r}. The session's "
                "recurrent state is unrecoverable from here; failing loudly "
                "instead of resuming it on a zeroed slot."
            )
        return unflatten_blob(manifest, buf)

    def has(self, session_id: str) -> bool:
        sid = str(session_id)
        return self._local.has(sid) or sid in self._manifests

    def drop(self, session_id: str) -> None:
        sid = str(session_id)
        self._local.drop(sid)
        self._manifests.pop(sid, None)


# ---------------------------------------------------------------------------
# The executor
# ---------------------------------------------------------------------------


class GdnSlotExecutor:
    """Runs a :class:`VacatePlan`, once, inside an armed between-tick window.

    Every collaborator is injected, for the same reason ``KvPressureRuntime``
    injects its consensus channel: the whole object is then exercisable on
    CPU, including the failure paths that a GPU boot reaches only under
    pressure.

    ``slot_of(session_id) -> Optional[int]``   the session's resident slot
    ``bind(session_id, slot) -> None``         session now owns that slot
    ``unbind(session_id) -> None``             session owns no slot
    """

    def __init__(
        self,
        *,
        mamba_pool,
        mamba_allocator,
        slot_of: Callable[[str], Optional[int]],
        bind: Callable[[str, int], None],
        unbind: Callable[[str], None],
        blob_store: Optional[BlobStore] = None,
    ):
        self._pool = mamba_pool
        self._alloc = mamba_allocator
        self._slot_of = slot_of
        self._bind = bind
        self._unbind = unbind
        self._store = blob_store if blob_store is not None else LocalGdnBlobStore()
        self._in_window = False
        self.vacated_total = 0
        self.restored_total = 0
        self.rounds = 0

    # -- the window ----------------------------------------------------------
    @contextmanager
    def between_ticks(self):
        """Arm the executor for the duration of the between-tick boundary.

        Re-entry is refused rather than counted: a nested window would mean
        two callers each believe they own the boundary, and the second one's
        idea of "no graph can replay right now" is not checkable from here.
        """
        if self._in_window:
            raise GdnSlotWindowError(
                f"{LOG_PREFIX}: between_ticks() is already armed. Two callers "
                "cannot both own one boundary; the second would be running "
                "against the first one's half-applied plan."
            )
        self._in_window = True
        try:
            yield self
        finally:
            self._in_window = False

    @property
    def armed(self) -> bool:
        return self._in_window

    @property
    def parked_ids(self) -> List[str]:
        """Sessions whose state is currently a blob -- the ``parked_ids``
        argument of the next :func:`vacate_plan` call."""
        store = self._store
        if isinstance(store, LocalGdnBlobStore):
            return sorted(store._blobs)
        return sorted(getattr(store, "_manifests", {}))

    def forget(self, session_id: str) -> None:
        """A parked session finished or was aborted: drop its blob."""
        self._store.drop(session_id)

    # -- execution -----------------------------------------------------------
    def run(self, plan) -> ExecResult:
        if not self._in_window:
            raise GdnSlotWindowError(
                f"{LOG_PREFIX}: run() outside the between-tick window. The "
                "state pool is addressed by captured CUDA graphs, so freeing "
                "and rewriting a slot here is the #52/#53 corruption class -- "
                "and a wrong recurrent state is plausible output, not a "
                "crash. Wrap the call in executor.between_ticks() at the "
                "scheduler boundary that retires the previous batch."
            )
        self.rounds += 1
        vacated: List[str] = []
        restored: List[str] = []
        failed: Dict[str, str] = {}

        # Vacates first: they are what feeds the allocator the restores draw
        # from. See module docstring.
        for sid in plan.vacate:
            try:
                vacated.append(self._vacate_one(sid))
            except GdnSlotError as exc:
                failed[sid] = str(exc)
        for sid in plan.restore:
            try:
                restored.append(self._restore_one(sid))
            except GdnSlotError as exc:
                failed[sid] = str(exc)

        self.vacated_total += len(vacated)
        self.restored_total += len(restored)
        if vacated or restored or failed:
            logger.info(
                "%s round %d: vacated %s, restored %s%s (free slots now %s)",
                LOG_PREFIX,
                self.rounds,
                vacated or "-",
                restored or "-",
                f", FAILED {failed}" if failed else "",
                self._available(),
            )
        return ExecResult(vacated=vacated, restored=restored, failed=failed)

    # -- one session ---------------------------------------------------------
    def _vacate_one(self, session_id: str) -> str:
        slot = self._slot_of(session_id)
        if slot is None:
            raise GdnSlotError(
                f"session {session_id} was planned to vacate but holds no "
                "state slot; the plan was built from a stale session list."
            )
        blob = self._pool.export_state_blob(int(slot))
        self._store.put(session_id, blob)
        # Only now is the slot expendable: the bytes are out, and the session
        # is unbound BEFORE the free so no window exists in which the
        # allocator could hand this slot to a peer while a request still
        # points at it.
        self._unbind(session_id)
        self._free_slot(int(slot))
        return session_id

    def _restore_one(self, session_id: str) -> str:
        if not self._store.has(session_id):
            raise GdnSlotError(
                f"session {session_id} was planned to restore but has no "
                "parked blob; refusing to resume it on an uninitialised slot "
                "(that would be a silent re-prefill with wrong history)."
            )
        slot = self._alloc_slot()
        if slot is None:
            raise GdnSlotError(
                f"session {session_id} must resume this tick but no resident "
                f"state slot is free (pool holds {self._alloc.size} slots, "
                f"{self._available()} free). The plan promised room the "
                "vacates did not deliver -- check the same round's failures."
            )
        blob = self._store.pop(session_id)
        self._pool.import_state_blob(int(slot), blob)
        self._bind(session_id, int(slot))
        return session_id

    # -- allocator adapters --------------------------------------------------
    # MambaSlotAllocator speaks torch tensors; the tests drive plain ints.
    # Keeping the adaptation here means neither side has to know about the
    # other, and the CPU suite needs no CUDA.
    def _alloc_slot(self) -> Optional[int]:
        got = self._alloc.alloc(1)
        if got is None:
            return None
        if hasattr(got, "tolist"):
            got = got.tolist()
        return int(got[0]) if isinstance(got, (list, tuple)) else int(got)

    def _free_slot(self, slot: int) -> None:
        free = getattr(self._alloc, "free")
        try:
            import torch

            free(torch.tensor([slot], dtype=torch.int64, device=self._alloc.device))
        except Exception:  # noqa: BLE001 -- plain-int allocators in tests
            free([slot])

    def _available(self) -> int:
        try:
            return int(self._alloc.available_size())
        except Exception:  # noqa: BLE001
            return -1
