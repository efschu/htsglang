"""Weg-2 slice S1 -- the three things the upstream memory saver does not do.

Upstream owns the sleep/wake mechanism itself and Weg 2 uses it unchanged:
``TorchMemorySaverAdapter`` (``srt/utils/torch_memory_saver_adapter.py``), the
``release_memory_occupation`` / ``resume_memory_occupation`` RPCs, the tag
regions in ``memory_pool.py`` / ``model_runner.py``, and
``update_weights_from_disk`` as the backup-OFF wake source.  Nothing here
duplicates any of that -- a fork-owned twin of an upstream mechanism is a
defect, so this module adds exactly the three pieces upstream has no equivalent
for:

* **W12 ``Weg2MemorySaverInactive``** -- the refusal.  ``create(enable=False)``
  returns a no-op adapter whose ``pause()``/``resume()`` are literally ``pass``,
  so every sleep is a success value with no action.  Upstream's own
  ``check_validity()`` only warns and the release path never calls it.
* **The sleep-acceptance census** (design (S) 2.4 step 11) -- the runtime
  instrument that answers "the dormant group holds nothing" on the flip path.
  It reads the canonical device registry (``srt/registry/nvml.py``) and the
  arena's own counters (``arena_census()``); it keeps no counters of its own.
* **The per-physical-GPU PCIe serialisation lock** (design (S) 2.7) -- lifted
  from #89 hibernate's ``park_weights_to_disk`` (*adopt the lock, not the
  module*): a sleep-D2H and a wake-H2D must never overlap on one card's link,
  which halves both on the x4-linked 3080.  This is a DIFFERENT lock from the
  L2 ring's ``flock`` and is named separately on purpose.

Every wait here is bounded and every expiry is a refusal, never a longer wait.
"""

from __future__ import annotations

import errno
import fcntl
import logging
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

logger = logging.getLogger(__name__)

MIB = 1024 * 1024

#: Where the PCIe serialisation lock files live.  ``/dev/shm`` is a tmpfs that
#: every rank on this host shares, which is exactly the scope of the lock: one
#: physical GPU, all processes on this box.
PCIE_LOCK_DIR_ENV = "SGLANG_WEG2_PCIE_LOCK_DIR"
DEFAULT_PCIE_LOCK_DIR = "/dev/shm"

#: Deliberately NOT ``weg2-l2-<uuid>`` -- that name belongs to the L2 host ring
#: backing file and its own ``flock``.  Two locks, two names, so a reader never
#: has to guess which one a path refers to.
PCIE_LOCK_PREFIX = "weg2-pcie-serialize"

#: A sleep-D2H or wake-H2D of a 27 GiB shard runs ~2.1 s measured (campaign (a),
#: 2026-09-06, n=9).  The default deadline allows a full transfer of the sibling
#: plus slack; the caller may shorten it.
DEFAULT_PCIE_LOCK_TIMEOUT_S = 120.0


class Weg2MemorySaverInactive(RuntimeError):
    """W12: the memory-saver adapter is a no-op, so every sleep is a lie."""


class Weg2WakeRefused(RuntimeError):
    """W4: a wake leg failed.  VRAM has been mutated; there is no unwound state.

    Group-fatal STOP, never an automatic retry.  The recovery lane is an
    operator-driven full teardown and relaunch of that group (design (S) 2.7).
    """


class Weg2PcieLockTimeout(RuntimeError):
    """The PCIe serialisation lock was not acquired inside its deadline."""


# ---------------------------------------------------------------------------
# W12
# ---------------------------------------------------------------------------


def assert_memory_saver_active(adapter: Any, *, context: str) -> None:
    """Refuse unless ``adapter`` will actually release memory.

    ``adapter.enabled`` is upstream's own property and the single authority:
    ``_TorchMemorySaverAdapterNoop`` returns False, and the real adapter
    returns ``_memory_saver is not None and _memory_saver.enabled``, so a
    present-but-disabled library is caught too.  No second bookkeeping.

    Called from the launcher at boot (once per rank) and from the first sleep,
    which is the last moment before a no-op release would return success.
    """
    if adapter is not None and bool(getattr(adapter, "enabled", False)):
        return
    raise Weg2MemorySaverInactive(
        f"W12 Weg2MemorySaverInactive at {context}: the torch-memory-saver "
        f"adapter reports enabled=False, so pause()/resume() are no-ops that "
        f"return success and no VRAM is ever released. Launch this group with "
        f"--enable-memory-saver (and check that torch_memory_saver imported "
        f"and its hook mode is 'preload'). Refusing rather than sleeping."
    )


# ---------------------------------------------------------------------------
# The per-physical-GPU PCIe serialisation lock (design (S) 2.7, from #89)
# ---------------------------------------------------------------------------


def _sanitize(uuid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", uuid)


def pcie_lock_path(nvml_uuid: str, *, lock_dir: Optional[str] = None) -> str:
    """Path of the PCIe serialisation lock for one physical GPU."""
    directory = lock_dir or os.environ.get(PCIE_LOCK_DIR_ENV, DEFAULT_PCIE_LOCK_DIR)
    return os.path.join(directory, f".{PCIE_LOCK_PREFIX}-{_sanitize(nvml_uuid)}.lock")


def _resolve_uuid(nvml_uuid: Optional[str]) -> str:
    if nvml_uuid is not None:
        return nvml_uuid
    from sglang.srt.registry import nvml as nvml_registry

    return nvml_registry.current_device_uuid()


@contextmanager
def pcie_transfer_lock(
    *,
    nvml_uuid: Optional[str] = None,
    lock_dir: Optional[str] = None,
    timeout_s: float = DEFAULT_PCIE_LOCK_TIMEOUT_S,
    poll_s: float = 0.01,
    label: str = "transfer",
) -> Iterator[str]:
    """Serialise host<->device transfers per PHYSICAL GPU.

    Two co-located ranks (Weg 2 puts one P rank and one D rank on each card)
    would otherwise overlap a sleep-D2H with a wake-H2D and halve both on the
    x4-linked 3080.  Keyed on the NVML UUID, so co-located ranks agree on the
    key without any registry between them.

    ``flock`` is held on this open file description only, so two threads or two
    processes contend correctly and the lock dies with the holder.  Bounded:
    the deadline's expiry raises, it never waits longer.
    """
    uuid = _resolve_uuid(nvml_uuid)
    path = pcie_lock_path(uuid, lock_dir=lock_dir)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    deadline = time.monotonic() + float(timeout_s)
    waited_s = 0.0
    handle = open(path, "w")
    try:
        t_wait = time.perf_counter()
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    raise Weg2PcieLockTimeout(
                        f"PCIe serialisation lock {path} not acquired within "
                        f"{timeout_s:.1f}s while waiting to {label} on GPU "
                        f"{uuid}; the sibling rank on this card is still "
                        f"transferring. Refusing rather than overlapping."
                    ) from None
                time.sleep(poll_s)
        waited_s = time.perf_counter() - t_wait
        t_held = time.perf_counter()
        try:
            yield path
        finally:
            held_s = time.perf_counter() - t_held
            logger.debug(
                "[weg2 pcie] %s on %s: waited %.3fs, held %.3fs (denominator: "
                "one physical GPU, all processes on this host)",
                label,
                uuid,
                waited_s,
                held_s,
            )
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()


# ---------------------------------------------------------------------------
# Sleep-acceptance census (design (S) 2.4 step 11)
# ---------------------------------------------------------------------------


@dataclass
class SleepAcceptanceCensus:
    """What this process still holds on this card, right after a sleep.

    Every field names its instrument.  ``accepted`` is False whenever the
    instrument could not read -- a census that cannot measure must not report
    a pass, or it becomes the ``cutover_participants.py`` failure mode with a
    number attached.
    """

    pid: int
    nvml_uuid: Optional[str]
    #: NVML ``nvmlDeviceGetComputeRunningProcesses`` for THIS pid, in bytes.
    proc_used_bytes: Optional[int]
    nvml_free_bytes: Optional[int]
    nvml_total_bytes: Optional[int]
    #: Sums over ``kv_vmm_backing.arena_census()`` rows for this process.
    arena_reserved_bytes: int
    arena_backed_bytes: int
    arena_retained_bytes: int
    arena_rows: int
    #: False when any live arena still owns unmapped handles: NVML charges the
    #: process for that ADDRESS SPACE, so the freed-memory reading is not one.
    retain_handles_asserted: bool
    accepted: bool
    refusal_reason: Optional[str]
    denominator: str

    @property
    def proc_used_mib(self) -> Optional[float]:
        if self.proc_used_bytes is None:
            return None
        return self.proc_used_bytes / MIB

    def format_line(self) -> str:
        used = "n/a" if self.proc_used_bytes is None else f"{self.proc_used_mib:.1f}"
        free = (
            "n/a"
            if self.nvml_free_bytes is None
            else f"{self.nvml_free_bytes / MIB:.1f}"
        )
        return (
            f"[weg2 sleep-acceptance] uuid={self.nvml_uuid} pid={self.pid} "
            f"proc_used={used} MiB nvml_free={free} MiB "
            f"arena_backed={self.arena_backed_bytes / MIB:.1f} MiB "
            f"arena_retained={self.arena_retained_bytes / MIB:.1f} MiB "
            f"rows={self.arena_rows} retain_handles_asserted="
            f"{self.retain_handles_asserted} accepted={self.accepted} "
            f"reason={self.refusal_reason or '-'} denominator={self.denominator}"
        )


_ProcessBytes = Union[Dict[int, int], Callable[[str], Dict[int, int]], None]


def sleep_acceptance_census(
    *,
    nvml_uuid: Optional[str] = None,
    _process_bytes: _ProcessBytes = None,
    _memory_info: Optional[Tuple[int, int]] = None,
    _arena_census: Optional[Dict[int, Dict[str, int]]] = None,
    _pid: Optional[int] = None,
) -> SleepAcceptanceCensus:
    """Read what this rank still holds on its card after ``pause()``.

    Read-only and never raises: an instrument that can fail a boot is not an
    instrument (same contract as ``arena_census()``).  Every failure to read
    lands in ``refusal_reason`` and forces ``accepted=False``.

    The leading-underscore parameters are injection seams for the hermetic
    tests; production calls pass none of them.
    """
    pid = os.getpid() if _pid is None else int(_pid)
    reasons = []

    uuid = nvml_uuid
    if uuid is None:
        try:
            uuid = _resolve_uuid(None)
        except Exception as exc:  # pragma: no cover - depends on the rig
            uuid = None
            reasons.append(f"device uuid unresolved ({exc})")

    proc_used: Optional[int] = None
    proc_count = 0
    if uuid is not None:
        try:
            table = _process_bytes
            if table is None:
                from sglang.srt.registry import nvml as nvml_registry

                table = nvml_registry.process_bytes_on_uuid(uuid)
            elif callable(table):
                table = table(uuid)
            proc_count = len(table)
            if pid in table:
                proc_used = int(table[pid])
            else:
                reasons.append(f"pid {pid} not among NVML's {proc_count} process(es)")
        except Exception as exc:
            reasons.append(f"per-process read failed ({exc})")

    free_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    if uuid is not None:
        try:
            info = _memory_info
            if info is None:
                from sglang.srt.registry import nvml as nvml_registry

                mem = nvml_registry.memory_info_for_uuid(uuid)
                info = (mem.total_bytes, mem.free_bytes)
            total_bytes, free_bytes = int(info[0]), int(info[1])
        except Exception as exc:
            reasons.append(f"card memory read failed ({exc})")

    rows = _arena_census
    if rows is None:
        try:
            from sglang.srt.mem_cache.kv_vmm_backing import arena_census

            rows = arena_census()
        except Exception as exc:
            rows = {}
            reasons.append(f"arena census unavailable ({exc})")

    reserved = backed = retained = 0
    for row in (rows or {}).values():
        reserved += int(row.get("reserved", 0))
        backed += int(row.get("backed", 0))
        retained += int(row.get("retained", 0))

    retain_handles_asserted = retained == 0
    if not retain_handles_asserted:
        reasons.append(
            f"arena retain_handles is in force: {retained / MIB:.1f} MiB of "
            f"unmapped-but-owned ADDRESS SPACE stays charged to this process "
            f"by NVML, so this reading is address space, not free memory"
        )
    if backed:
        reasons.append(f"arena still backs {backed / MIB:.1f} MiB of device memory")

    accepted = proc_used is not None and not reasons
    denominator = (
        f"NVML per-process bytes for pid={pid} on uuid={uuid} over "
        f"{proc_count} compute process(es); arena rows={len(rows or {})}"
    )
    return SleepAcceptanceCensus(
        pid=pid,
        nvml_uuid=uuid,
        proc_used_bytes=proc_used,
        nvml_free_bytes=free_bytes,
        nvml_total_bytes=total_bytes,
        arena_reserved_bytes=reserved,
        arena_backed_bytes=backed,
        arena_retained_bytes=retained,
        arena_rows=len(rows or {}),
        retain_handles_asserted=retain_handles_asserted,
        accepted=accepted,
        refusal_reason="; ".join(reasons) if reasons else None,
        denominator=denominator,
    )
