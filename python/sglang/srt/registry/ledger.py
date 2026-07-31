# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Cross-process VRAM reservation ledger (DESIGN #333 §3.3).

The measured per-boot registry stays where it is; it remains the truth about
what a tenant actually used. This module is the layer above it: a *declared,
pre-boot, cross-process* reservation, because a tenant that has not booted
cannot have measured itself and a tenant in another process cannot be
gathered from.

Shape, exactly as §3.3 specifies:

*   One JSON file per physical GPU, keyed by NVML UUID, under
    ``/run/htsglang/vram``. UUID rather than CUDA index because enumeration
    order shifts between boots and driver states (see ``nvml.py``).
*   Holder identity plus heartbeat plus an expiring lease, following the
    ``/spinning/gpu-arb`` cross-session convention, so a crashed tenant does
    not hold bytes forever.
*   The invariant, checked before every write::

        sum(reserved_bytes for HOT or WARM_GPU tenants on card C)
            + corridor_bytes(C) <= nvml_total_bytes(C)

    ``corridor_bytes`` is #330's 400 MiB absolutely-free rule. Under
    multi-tenancy it is a property of the card, not of any tenant, so no
    tenant may account for it and this store owns it.
*   Waste accounting, ``waste(C) = sum(reserved) - sum(measured)``, reported
    and never acted on: reclaiming a reservation a tenant has not used yet is
    how a runtime OOM appears three minutes later.

History and ownership. This store was written for #333-M2, the Class-3
video-enhance tenant, because M2 landed ahead of M1 and needed to declare a
static budget honestly before the registry existed. M1 makes it the registry's
own ledger and moves it here unchanged in behaviour;
``sglang.srt.video_enhance.reservation`` re-exports it so the first tenant
keeps working against the same names. There is exactly one implementation of
the invariant on this rig, and this is it.

What lives here is the *store*: per-card files, locking, leases, reaping,
single-card admission, and the per-card exclusive window that profiling and
TensorRT engine builds require (§6.4: an engine build is an exclusive window
on that card and it lasts minutes). What lives one layer up in
:mod:`sglang.srt.registry.arbiter` is policy: which tenant may hold which
bytes, who gets evicted, and how many engines fit at once.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)

MIB = 1024 * 1024

#: #330's corridor: at least this much absolutely free on every card before
#: any boot. It belongs to the card, so it is subtracted once here and never
#: folded into a tenant's own reservation.
DEFAULT_CORRIDOR_BYTES = 400 * MIB

#: Lease length used when a caller does not state one. Short enough that a
#: crashed tenant's bytes come back promptly, long enough that a tenant busy
#: inside a minutes-long engine build survives on heartbeats alone.
DEFAULT_LEASE_SECONDS = 120.0

_STORE_ROOT_ENV = "HTSGLANG_VRAM_LEDGER_ROOT"
_PRIMARY_ROOT = Path("/run/htsglang/vram")


class TenantState(str, Enum):
    HOT = "HOT"
    WARM_GPU = "WARM_GPU"
    WARM_HOST = "WARM_HOST"
    COLD = "COLD"


#: States in which the tenant is holding device memory right now. Only these
#: count against the invariant; a WARM_HOST or COLD tenant has given its bytes
#: back and must not block an admission.
GPU_RESIDENT_STATES: frozenset[TenantState] = frozenset(
    {TenantState.HOT, TenantState.WARM_GPU}
)


class LedgerError(RuntimeError):
    """Base class for ledger failures."""


class ReservationRejected(LedgerError):
    """The §3.3 invariant would be violated. Nothing was written."""

    def __init__(
        self,
        message: str,
        *,
        card_uuid: str,
        requested_bytes: int,
        shortfall_bytes: int,
        holders: Sequence[str],
    ) -> None:
        super().__init__(message)
        self.card_uuid = card_uuid
        self.requested_bytes = requested_bytes
        self.shortfall_bytes = shortfall_bytes
        self.holders = tuple(holders)


class UnknownTenantError(LedgerError):
    """The named tenant holds no entry on that card."""


class CardBusyError(LedgerError):
    """The card-exclusive lock is held elsewhere and the wait budget ran out."""


@dataclass
class ReservationEntry:
    """One tenant's declared claim on one physical GPU (§3.3 entry schema)."""

    tenant_id: str
    klass: int
    state: TenantState
    reserved_bytes: int
    measured_bytes: int = 0
    posts: dict[str, int] = field(default_factory=dict)
    pid: int = 0
    heartbeat_ts: float = 0.0
    lease_expiry_ts: float = 0.0
    #: #344: the client this tenant is serving has gone quiet and is a dead
    #: suspect, but has not been declared dead yet. The bytes are still
    #: reserved and the tenant is still alive -- what changed is that nobody
    #: is currently reading the work they are paying for, so a reclaimer may
    #: prefer these bytes over an actively used tenant's. Advisory only:
    #: nothing in this module acts on the flag.
    in_grace: bool = False
    #: When the flag was last set. Zero while ``in_grace`` is false.
    grace_since_ts: float = 0.0

    @property
    def is_gpu_resident(self) -> bool:
        return self.state in GPU_RESIDENT_STATES

    def lease_expired(self, now: float) -> bool:
        return now > self.lease_expiry_ts

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ReservationEntry":
        return cls(
            tenant_id=str(data["tenant_id"]),
            klass=int(data["klass"]),
            state=TenantState(str(data["state"])),
            reserved_bytes=int(data["reserved_bytes"]),
            measured_bytes=int(data.get("measured_bytes", 0)),
            posts={str(k): int(v) for k, v in dict(data.get("posts") or {}).items()},
            pid=int(data.get("pid", 0)),
            heartbeat_ts=float(data.get("heartbeat_ts", 0.0)),
            lease_expiry_ts=float(data.get("lease_expiry_ts", 0.0)),
            # Defaulted, so a ledger file written by a server from before
            # #344 reads back as an ordinary active entry.
            in_grace=bool(data.get("in_grace", False)),
            grace_since_ts=float(data.get("grace_since_ts", 0.0)),
        )

    def describe(self) -> str:
        grace = " [in grace, reclaimable]" if self.in_grace else ""
        return (
            f"{self.tenant_id} (class {self.klass}, {self.state.value}, pid {self.pid}) "
            f"reserved {self.reserved_bytes / MIB:.0f} MiB{grace}"
        )


@dataclass(frozen=True)
class CardLedger:
    """The full contents of one card's ledger file, as read."""

    card_uuid: str
    entries: tuple[ReservationEntry, ...]

    def tenant(self, tenant_id: str) -> ReservationEntry | None:
        for entry in self.entries:
            if entry.tenant_id == tenant_id:
                return entry
        return None

    @property
    def gpu_resident(self) -> tuple[ReservationEntry, ...]:
        return tuple(e for e in self.entries if e.is_gpu_resident)

    @property
    def reserved_bytes(self) -> int:
        return sum(e.reserved_bytes for e in self.gpu_resident)

    @property
    def measured_bytes(self) -> int:
        return sum(e.measured_bytes for e in self.gpu_resident)

    @property
    def waste_bytes(self) -> int:
        """``sum(reserved) - sum(measured)``. Negative means a tenant overran
        its declaration, which is a ledger finding, not something to clamp."""
        return self.reserved_bytes - self.measured_bytes

    @property
    def in_grace(self) -> tuple[ReservationEntry, ...]:
        return tuple(e for e in self.gpu_resident if e.in_grace)

    @property
    def grace_bytes(self) -> int:
        """Reserved bytes held by tenants whose client is a dead suspect (#344).

        Distinct from ``waste_bytes``: waste is memory a tenant declared and
        has not touched, which it may still need. This is memory a tenant is
        genuinely using on behalf of a consumer that appears to have left, so
        it is the more honest first target when a card is under pressure.
        Reporting only -- reclaiming is #287's staircase to decide.
        """
        return sum(e.reserved_bytes for e in self.in_grace)

    def render(self) -> str:
        if not self.entries:
            return f"{self.card_uuid}: no reservations"
        lines = [f"{self.card_uuid}:"]
        lines += [f"  {e.describe()}" for e in self.entries]
        summary = (
            f"  reserved {self.reserved_bytes / MIB:.0f} MiB, "
            f"measured {self.measured_bytes / MIB:.0f} MiB, "
            f"waste {self.waste_bytes / MIB:.0f} MiB"
        )
        if self.grace_bytes:
            summary += f", in grace {self.grace_bytes / MIB:.0f} MiB"
        lines.append(summary)
        return "\n".join(lines)


def default_store_root() -> Path:
    """``/run/htsglang/vram`` when writable, otherwise the runtime dir, otherwise temp.

    The fallbacks exist because the ledger has to work in a container without
    a writable ``/run`` and in a unit test with a ``tmp_path`` root. They are
    ordered so that co-tenant processes on the same host converge on the same
    directory whenever ``/run`` is available at all -- two tenants that pick
    different roots do not see each other's reservations, which is the one
    failure mode worse than a rejection.
    """
    override = os.environ.get(_STORE_ROOT_ENV)
    if override:
        return Path(override)
    for candidate in (
        _PRIMARY_ROOT,
        (
            Path(os.environ["XDG_RUNTIME_DIR"]) / "htsglang" / "vram"
            if os.environ.get("XDG_RUNTIME_DIR")
            else None
        ),
    ):
        if candidate is None:
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".writable"
            probe.touch()
            probe.unlink()
            return candidate
        except OSError:
            continue
    return Path(tempfile.gettempdir()) / "htsglang" / "vram"


def _default_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists and belongs to another user. Alive.
        return True
    except OSError as exc:  # pragma: no cover - platform oddity
        return exc.errno != errno.ESRCH
    return True


def _safe_name(card_uuid: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in card_uuid)


class ReservationStore:
    """The per-card reservation files, with locking and lease reaping.

    All mutating operations run under an exclusive ``flock`` on a per-card
    lock file and write via temp-file plus :func:`os.replace`, so a reader
    never observes a half-written ledger and two processes cannot both pass
    the invariant check against the same pre-state.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        corridor_bytes: int = DEFAULT_CORRIDOR_BYTES,
        total_bytes_resolver: Callable[[str], int] | None = None,
        pid_alive: Callable[[int], bool] = _default_pid_alive,
        clock: Callable[[], float] = time.time,
        critical_section_probe: Callable[[str], None] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else default_store_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.corridor_bytes = int(corridor_bytes)
        self._total_bytes_resolver = total_bytes_resolver
        self._pid_alive = pid_alive
        self._clock = clock
        # Called inside the per-card critical section with the operation name.
        # Its production use is lock-hold-time tracing; the concurrency test
        # uses it to widen the window deterministically instead of relying on
        # a race being observed by luck.
        self._probe = critical_section_probe

    # -- paths -------------------------------------------------------------

    def ledger_path(self, card_uuid: str) -> Path:
        return self.root / f"{_safe_name(card_uuid)}.json"

    def _ledger_lock_path(self, card_uuid: str) -> Path:
        return self.root / f"{_safe_name(card_uuid)}.lock"

    def _card_lock_path(self, card_uuid: str) -> Path:
        # Deliberately a different file from the ledger lock: an engine build
        # holds the card for minutes and must not block a co-tenant's
        # heartbeat, which needs the ledger lock for microseconds.
        return self.root / f"{_safe_name(card_uuid)}.card.lock"

    # -- locking -----------------------------------------------------------

    @contextlib.contextmanager
    def _ledger_lock(self, card_uuid: str, operation: str) -> Iterator[None]:
        path = self._ledger_lock_path(card_uuid)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if self._probe is not None:
                self._probe(operation)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @contextlib.contextmanager
    def card_exclusive_lock(
        self, card_uuid: str, *, timeout: float | None = None, purpose: str = ""
    ) -> Iterator[None]:
        """Serialise device-exclusive work on one physical GPU.

        Co-located tenants and co-located ranks must not run a TensorRT engine
        build or a memory probe at the same time on the same card: the build
        is an exclusive window (§6.4) and a probe measures free memory, which
        is meaningless while a neighbour is allocating. Hold this for the
        whole window.

        ``timeout=None`` blocks. ``timeout=0`` polls once. Anything else waits
        up to that many seconds before raising :class:`CardBusyError`.
        """
        path = self._card_lock_path(card_uuid)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        acquired = False
        try:
            if timeout is None:
                fcntl.flock(fd, fcntl.LOCK_EX)
                acquired = True
            else:
                deadline = self._clock() + timeout
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                        break
                    except OSError as exc:
                        if exc.errno not in (errno.EACCES, errno.EAGAIN):
                            raise
                        if self._clock() >= deadline:
                            raise CardBusyError(
                                f"card {card_uuid} is held exclusively by another "
                                f"process (waited {timeout:g}s"
                                f"{f', purpose {purpose}' if purpose else ''})"
                            ) from None
                        time.sleep(0.01)
            os.ftruncate(fd, 0)
            os.write(fd, f"pid={os.getpid()} purpose={purpose}\n".encode())
            yield
        finally:
            if acquired:
                with contextlib.suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    # -- raw IO ------------------------------------------------------------

    def _read_unlocked(self, card_uuid: str) -> list[ReservationEntry]:
        path = self.ledger_path(card_uuid)
        try:
            raw = path.read_text()
        except FileNotFoundError:
            return []
        if not raw.strip():
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.error(
                "VRAM ledger %s is not valid JSON; treating the card as unreserved. "
                "A tenant may be holding bytes this store cannot see.",
                path,
            )
            return []
        return [ReservationEntry.from_json(e) for e in data.get("entries", [])]

    def _write_unlocked(
        self, card_uuid: str, entries: Sequence[ReservationEntry]
    ) -> None:
        path = self.ledger_path(card_uuid)
        payload = {
            "card_uuid": card_uuid,
            "updated_ts": self._clock(),
            "entries": [e.to_json() for e in entries],
        }
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.root), prefix=path.name, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise

    # -- reading -----------------------------------------------------------

    def read(self, card_uuid: str) -> CardLedger:
        """Snapshot of the card's ledger. Does not reap and does not write."""
        with self._ledger_lock(card_uuid, "read"):
            return CardLedger(card_uuid, tuple(self._read_unlocked(card_uuid)))

    def waste(self, card_uuid: str) -> int:
        """``sum(reserved) - sum(measured)`` over GPU-resident tenants.

        Reported only. #330 makes net waste above 1.5 GiB a registered item;
        it does not make it reclaimable, because the tenant may simply not
        have reached its peak yet.
        """
        return self.read(card_uuid).waste_bytes

    # -- reaping -----------------------------------------------------------

    def _reap_unlocked(
        self, entries: list[ReservationEntry], now: float
    ) -> tuple[list[ReservationEntry], list[ReservationEntry]]:
        live: list[ReservationEntry] = []
        stale: list[ReservationEntry] = []
        for entry in entries:
            # Both conditions are required. A live pid keeps its bytes no
            # matter what the lease says: the tenant is running and holding
            # device memory, and a lease that lapsed only means its heartbeat
            # thread is starved. Reclaiming there would hand the same bytes to
            # a second tenant.
            if entry.lease_expired(now) and not self._pid_alive(entry.pid):
                stale.append(entry)
            else:
                live.append(entry)
        return live, stale

    def reap(self, card_uuid: str) -> list[ReservationEntry]:
        """Drop entries whose lease expired *and* whose pid is gone."""
        with self._ledger_lock(card_uuid, "reap"):
            entries = self._read_unlocked(card_uuid)
            live, stale = self._reap_unlocked(entries, self._clock())
            if stale:
                for entry in stale:
                    logger.warning(
                        "reclaiming %s from card %s: lease expired and pid %d is gone",
                        entry.describe(),
                        card_uuid,
                        entry.pid,
                    )
                self._write_unlocked(card_uuid, live)
            return stale

    # -- mutation ----------------------------------------------------------

    def _resolve_total(self, card_uuid: str, override: int | None) -> int:
        if override is not None:
            return int(override)
        if self._total_bytes_resolver is not None:
            return int(self._total_bytes_resolver(card_uuid))
        from sglang.srt.registry.nvml import total_bytes_for_uuid

        return total_bytes_for_uuid(card_uuid)

    def acquire(
        self,
        *,
        card_uuid: str,
        tenant_id: str,
        klass: int,
        reserved_bytes: int,
        nvml_total_bytes: int | None = None,
        state: TenantState = TenantState.HOT,
        posts: Mapping[str, int] | None = None,
        measured_bytes: int = 0,
        pid: int | None = None,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> ReservationEntry:
        """Declare ``reserved_bytes`` on ``card_uuid``, or reject.

        The §3.3 invariant is evaluated against the post-write state, under
        the card lock, before anything is written. An existing entry for the
        same ``tenant_id`` is replaced rather than added to, so re-acquiring
        with a larger budget is a legal, checked operation.
        """
        if reserved_bytes < 0:
            raise ValueError("reserved_bytes must not be negative")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        pid = os.getpid() if pid is None else pid
        total = self._resolve_total(card_uuid, nvml_total_bytes)

        with self._ledger_lock(card_uuid, "acquire"):
            now = self._clock()
            entries, _stale = self._reap_unlocked(self._read_unlocked(card_uuid), now)
            others = [e for e in entries if e.tenant_id != tenant_id]

            entry = ReservationEntry(
                tenant_id=tenant_id,
                klass=int(klass),
                state=state,
                reserved_bytes=int(reserved_bytes),
                measured_bytes=int(measured_bytes),
                posts=dict(posts or {}),
                pid=int(pid),
                heartbeat_ts=now,
                lease_expiry_ts=now + float(lease_seconds),
            )

            if entry.is_gpu_resident:
                held = sum(e.reserved_bytes for e in others if e.is_gpu_resident)
                required = held + entry.reserved_bytes + self.corridor_bytes
                if required > total:
                    raise self._rejection(card_uuid, others, entry, held, total)

            self._write_unlocked(card_uuid, [*others, entry])
            return entry

    def _rejection(
        self,
        card_uuid: str,
        others: Sequence[ReservationEntry],
        entry: ReservationEntry,
        held_bytes: int,
        total_bytes: int,
    ) -> ReservationRejected:
        shortfall = (
            held_bytes + entry.reserved_bytes + self.corridor_bytes - total_bytes
        )
        holders = [e.describe() for e in others if e.is_gpu_resident] or ["none"]
        message = (
            f"reservation rejected on card {card_uuid}: "
            f"{tenant_label(entry)} requested {entry.reserved_bytes / MIB:.0f} MiB, "
            f"existing holders already hold {held_bytes / MIB:.0f} MiB, "
            f"the card corridor is {self.corridor_bytes / MIB:.0f} MiB and NVML "
            f"reports {total_bytes / MIB:.0f} MiB total -- short by "
            f"{shortfall / MIB:.0f} MiB. Holders: " + "; ".join(holders)
        )
        return ReservationRejected(
            message,
            card_uuid=card_uuid,
            requested_bytes=entry.reserved_bytes,
            shortfall_bytes=shortfall,
            holders=[e.tenant_id for e in others if e.is_gpu_resident],
        )

    def heartbeat(
        self,
        card_uuid: str,
        tenant_id: str,
        *,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> ReservationEntry:
        """Touch the holder line and push the lease out. The gpu-arb convention."""
        return self._mutate(
            card_uuid,
            tenant_id,
            "heartbeat",
            lambda entry, now: replace(
                entry, heartbeat_ts=now, lease_expiry_ts=now + float(lease_seconds)
            ),
        )

    def set_grace(
        self, card_uuid: str, tenant_id: str, in_grace: bool
    ) -> ReservationEntry:
        """Mark the tenant's client as a dead suspect, or clear the mark (#344).

        The lease is untouched on purpose. A tenant in grace is still running
        and still holds its device memory -- shortening its lease would let
        the reaper hand the same bytes to somebody else while they are in
        use, which is exactly the failure ``_reap_unlocked`` refuses to make.
        What this changes is only what a reclaimer *sees*: bytes booked
        against a consumer that appears to have left.
        """
        return self._mutate(
            card_uuid,
            tenant_id,
            "set_grace",
            lambda entry, now: replace(
                entry,
                in_grace=bool(in_grace),
                grace_since_ts=(now if in_grace else 0.0),
            ),
        )

    def update_measured(
        self, card_uuid: str, tenant_id: str, measured_bytes: int
    ) -> ReservationEntry:
        """Record the last observed peak from the measured registry."""
        return self._mutate(
            card_uuid,
            tenant_id,
            "update_measured",
            lambda entry, now: replace(entry, measured_bytes=int(measured_bytes)),
        )

    def set_state(
        self, card_uuid: str, tenant_id: str, state: TenantState
    ) -> ReservationEntry:
        """Move a tenant along the residency ladder.

        A transition *into* a GPU-resident state is an admission and is
        checked against the invariant exactly like an acquire; a transition
        out of one only frees bytes and cannot fail.
        """
        if state in GPU_RESIDENT_STATES:
            with self._ledger_lock(card_uuid, "set_state"):
                entries = self._read_unlocked(card_uuid)
                current = next((e for e in entries if e.tenant_id == tenant_id), None)
                if current is None:
                    raise UnknownTenantError(
                        f"tenant {tenant_id!r} holds no entry on card {card_uuid}"
                    )
                if current.is_gpu_resident:
                    return self._replace_locked(
                        card_uuid, entries, tenant_id, replace(current, state=state)
                    )
                others = [e for e in entries if e.tenant_id != tenant_id]
                held = sum(e.reserved_bytes for e in others if e.is_gpu_resident)
                total = self._resolve_total(card_uuid, None)
                promoted = replace(current, state=state)
                if held + promoted.reserved_bytes + self.corridor_bytes > total:
                    raise self._rejection(card_uuid, others, promoted, held, total)
                return self._replace_locked(card_uuid, entries, tenant_id, promoted)
        return self._mutate(
            card_uuid, tenant_id, "set_state", lambda e, now: replace(e, state=state)
        )

    def release(self, card_uuid: str, tenant_id: str) -> bool:
        """Drop this tenant's entry. True when something was removed."""
        with self._ledger_lock(card_uuid, "release"):
            entries = self._read_unlocked(card_uuid)
            kept = [e for e in entries if e.tenant_id != tenant_id]
            if len(kept) == len(entries):
                return False
            self._write_unlocked(card_uuid, kept)
            return True

    def _mutate(
        self,
        card_uuid: str,
        tenant_id: str,
        operation: str,
        fn: Callable[[ReservationEntry, float], ReservationEntry],
    ) -> ReservationEntry:
        with self._ledger_lock(card_uuid, operation):
            entries = self._read_unlocked(card_uuid)
            current = next((e for e in entries if e.tenant_id == tenant_id), None)
            if current is None:
                raise UnknownTenantError(
                    f"tenant {tenant_id!r} holds no entry on card {card_uuid}"
                )
            return self._replace_locked(
                card_uuid, entries, tenant_id, fn(current, self._clock())
            )

    def _replace_locked(
        self,
        card_uuid: str,
        entries: Sequence[ReservationEntry],
        tenant_id: str,
        updated: ReservationEntry,
    ) -> ReservationEntry:
        merged = [updated if e.tenant_id == tenant_id else e for e in entries]
        self._write_unlocked(card_uuid, merged)
        return updated


def tenant_label(entry: ReservationEntry) -> str:
    return f"tenant {entry.tenant_id!r} (class {entry.klass}, pid {entry.pid})"


def available_bytes(
    store: ReservationStore,
    card_uuid: str,
    nvml_total_bytes: int,
    *,
    excluding_tenant: str | None = None,
) -> int:
    """Bytes a new tenant could still claim on this card, corridor removed.

    This is the plan-time question the shard planner asks before it places
    work on a card: what an admission would be checked against, computed
    without attempting one.
    """
    ledger = store.read(card_uuid)
    held = sum(
        e.reserved_bytes
        for e in ledger.gpu_resident
        if excluding_tenant is None or e.tenant_id != excluding_tenant
    )
    return max(0, nvml_total_bytes - held - store.corridor_bytes)


# ---------------------------------------------------------------------------
# Multi-card layer
#
# A single-card tenant (the Class-3 executor of M2) needs exactly one entry.
# A Class-1 tensor-parallel engine needs one entry per card it shards across,
# and it needs them all or none: a half-acquired engine is bytes held by a
# process that will never boot. Everything below is that all-or-nothing layer,
# plus the read-only projection the control plane answers /registry/plan with.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CardDemand:
    """What one tenant wants on one physical card."""

    card_uuid: str
    reserved_bytes: int
    posts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reserved_bytes < 0:
            raise ValueError("reserved_bytes must not be negative")


@dataclass(frozen=True)
class CardShortfall:
    """Why one card could not take a demand, in the terms the operator needs."""

    card_uuid: str
    requested_bytes: int
    held_bytes: int
    corridor_bytes: int
    total_bytes: int
    holders: tuple[str, ...]
    holder_lines: tuple[str, ...]

    @property
    def shortfall_bytes(self) -> int:
        return (
            self.held_bytes + self.requested_bytes + self.corridor_bytes
        ) - self.total_bytes

    def render(self) -> str:
        holders = "; ".join(self.holder_lines) if self.holder_lines else "none"
        return (
            f"card {self.card_uuid}: requested {self.requested_bytes / MIB:.0f} MiB, "
            f"held {self.held_bytes / MIB:.0f} MiB, corridor "
            f"{self.corridor_bytes / MIB:.0f} MiB, NVML total "
            f"{self.total_bytes / MIB:.0f} MiB -- short by "
            f"{self.shortfall_bytes / MIB:.0f} MiB. Holders: {holders}"
        )


@dataclass(frozen=True)
class FeasibilityReport:
    """Result of asking the ledger a question without changing it."""

    fits: bool
    shortfalls: tuple[CardShortfall, ...] = ()

    @property
    def shortfall_bytes(self) -> int:
        return sum(s.shortfall_bytes for s in self.shortfalls)

    def render(self) -> str:
        if self.fits:
            return "fits"
        return "does not fit: " + " | ".join(s.render() for s in self.shortfalls)


def plan_reservation(
    store: ReservationStore,
    demands: Sequence[CardDemand],
    totals: Mapping[str, int],
    *,
    excluding_tenant: str | None = None,
    ignoring_tenants: Sequence[str] = (),
    on_empty_rig: bool = False,
) -> FeasibilityReport:
    """Would these demands pass the §3.3 invariant right now? Reads only.

    ``excluding_tenant`` is the tenant being re-planned: its current bytes do
    not count against itself. ``ignoring_tenants`` are eviction candidates the
    caller is *considering* demoting, which is how the arbiter prices an
    eviction before performing one. ``on_empty_rig`` ignores every holder and
    asks the intrinsic question: could this ever fit, on a card with nothing
    else on it? That one is about the spec, not about the current occupancy,
    and it is the question registration must answer.
    """
    ignored = {*ignoring_tenants}
    if excluding_tenant is not None:
        ignored.add(excluding_tenant)
    shortfalls: list[CardShortfall] = []
    for demand in demands:
        ledger = store.read(demand.card_uuid)
        resident = (
            []
            if on_empty_rig
            else [e for e in ledger.gpu_resident if e.tenant_id not in ignored]
        )
        held = sum(e.reserved_bytes for e in resident)
        total = int(totals[demand.card_uuid])
        if held + demand.reserved_bytes + store.corridor_bytes > total:
            shortfalls.append(
                CardShortfall(
                    card_uuid=demand.card_uuid,
                    requested_bytes=demand.reserved_bytes,
                    held_bytes=held,
                    corridor_bytes=store.corridor_bytes,
                    total_bytes=total,
                    holders=tuple(e.tenant_id for e in resident),
                    holder_lines=tuple(e.describe() for e in resident),
                )
            )
    return FeasibilityReport(fits=not shortfalls, shortfalls=tuple(shortfalls))


class MultiCardReservation:
    """All-or-nothing reservation of one tenant across several cards.

    Cards are taken in sorted UUID order so two tenants contending for the
    same pair always contend in the same order, and a failure on the second
    card releases the first before raising. There is no cross-card atomicity
    beyond that: two tenants can both fail and both roll back. That is a
    livelock only under continuous contention, and it is preferred over a
    global lock, which would serialise every heartbeat on the rig behind a
    minutes-long engine build.
    """

    def __init__(
        self,
        store: ReservationStore,
        *,
        tenant_id: str,
        klass: int,
        totals: Mapping[str, int] | None = None,
    ) -> None:
        self.store = store
        self.tenant_id = tenant_id
        self.klass = int(klass)
        self._totals = dict(totals or {})
        self._held: list[str] = []

    @property
    def cards(self) -> tuple[str, ...]:
        return tuple(self._held)

    def _total_for(self, card_uuid: str) -> int | None:
        return self._totals.get(card_uuid)

    def acquire(
        self,
        demands: Sequence[CardDemand],
        *,
        state: TenantState = TenantState.HOT,
        measured_bytes: int = 0,
        pid: int | None = None,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> list[ReservationEntry]:
        ordered = sorted(demands, key=lambda d: d.card_uuid)
        taken: list[ReservationEntry] = []
        try:
            for demand in ordered:
                taken.append(
                    self.store.acquire(
                        card_uuid=demand.card_uuid,
                        tenant_id=self.tenant_id,
                        klass=self.klass,
                        reserved_bytes=demand.reserved_bytes,
                        nvml_total_bytes=self._total_for(demand.card_uuid),
                        state=state,
                        posts=demand.posts,
                        measured_bytes=measured_bytes,
                        pid=pid,
                        lease_seconds=lease_seconds,
                    )
                )
                self._held.append(demand.card_uuid)
        except LedgerError:
            self.release()
            raise
        return taken

    def set_state(self, state: TenantState) -> None:
        """Move every card's entry. Promotion is checked; demotion cannot fail.

        A promotion that fails part-way leaves the tenant split across states,
        which is why the caller must treat a raised exception as "still
        demoted" and roll the successful cards back itself -- done here.
        """
        if state not in GPU_RESIDENT_STATES:
            for card_uuid in list(self._held):
                self.store.set_state(card_uuid, self.tenant_id, state)
            return
        promoted: list[str] = []
        try:
            for card_uuid in sorted(self._held):
                self.store.set_state(card_uuid, self.tenant_id, state)
                promoted.append(card_uuid)
        except LedgerError:
            for card_uuid in promoted:
                self.store.set_state(card_uuid, self.tenant_id, TenantState.COLD)
            raise

    def heartbeat(self, *, lease_seconds: float = DEFAULT_LEASE_SECONDS) -> None:
        for card_uuid in list(self._held):
            with contextlib.suppress(UnknownTenantError):
                self.store.heartbeat(
                    card_uuid, self.tenant_id, lease_seconds=lease_seconds
                )

    def update_measured(self, measured: Mapping[str, int]) -> None:
        for card_uuid, value in measured.items():
            with contextlib.suppress(UnknownTenantError):
                self.store.update_measured(card_uuid, self.tenant_id, int(value))

    def release(self) -> None:
        for card_uuid in list(self._held):
            self.store.release(card_uuid, self.tenant_id)
        self._held.clear()


def adopt(
    store: ReservationStore,
    tenant_id: str,
    klass: int,
    cards: Sequence[str],
    *,
    totals: Mapping[str, int] | None = None,
) -> MultiCardReservation:
    """Rebuild a handle over entries this tenant already holds on ``cards``.

    Used after a control-plane restart: the files outlive the process, so the
    arbiter reattaches to its own reservations rather than re-acquiring them
    and racing itself.
    """
    handle = MultiCardReservation(
        store, tenant_id=tenant_id, klass=klass, totals=totals
    )
    for card_uuid in sorted(cards):
        if store.read(card_uuid).tenant(tenant_id) is not None:
            handle._held.append(card_uuid)
    return handle
