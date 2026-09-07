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
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, Union

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

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

#: The S1 sleep-acceptance criterion in its DELTA form: how much of what this
#: process held before the pause must be gone after it.
#:
#: PROVENANCE, and it is a chosen number, not a measured one.  Campaign (a)
#: measured, on this rig, a genuine sleep going 30,154 -> 1,294 MiB per-process
#: (CAMPAIGN_a_0906.md §2 arm 1, n=9 steady cycles over two cold boots): the
#: sleep RELEASED 95.7 % and RETAINED 4.3 %.  A no-op sleep -- the exact
#: condition W12 and this census exist to catch -- releases 0 %.  Half is the
#: midpoint between those two, i.e. ~11x the measured retained margin, so the
#: criterion cannot be tripped by allocator noise and cannot be passed by a
#: no-op.  It is deliberately NOT a ceiling on residency: the dormant floor
#: D_c is UNMEASURED until S3 (register U2), and inventing a MiB ceiling here
#: would be a number the tree cannot defend.  When S3 measures D_c its launcher
#: passes ``expected_max_resident_bytes`` and the ceiling form takes over --
#: both forms may be supplied, and then both must hold.
WEG2_SLEEP_MIN_RELEASED_FRACTION = 0.5

#: The tag set a Weg-2 sleep releases, and therefore the only request shape the
#: DELTA form's denominator is the right one for.  The floor is a fraction of
#: this process's WHOLE device residency (NVML per-process bytes), so it may
#: only grade a release that actually targets the whole of it: the weights
#: image AND the KV pool (which carries the mamba/GDN anchors under the same
#: tag -- memory_pool.py:1017, there is no separate mamba tag).  Anything less
#: -- the #89 park's ``tags=["weights"]``, a kv-only release -- releases a
#: PROPER SUBSET and cannot be graded against the whole; see
#: :func:`sleep_acceptance_census`.
WEG2_SLEEP_TAGS = frozenset((GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS))


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


def checkpoint_quantization(model_config: Any, server_args: Any) -> Optional[str]:
    """The quantization actually in force for this process's checkpoint.

    ONE definition, two users (the launch arm and the wake), so a launcher edit
    cannot make the two disagree.  ``ModelConfig.quantization`` is the merged
    value -- the CLI flag AND the config.json ``quantization_config`` the loader
    auto-detects -- so a checkpoint that carries its own quant config without
    ``--quantization`` is covered; ``server_args`` is the fallback for the
    moments where no model config is reachable yet.
    """
    for holder in (model_config, server_args):
        if holder is None:
            continue
        value = getattr(holder, "quantization", None)
        if value:
            return str(value)
    return None


def assert_backup_off_wake_refill_is_defined(
    *, quantization: Optional[str], context: str
) -> None:
    """W4: refuse a backup-OFF wake whose refill would re-run a repacking pass.

    The backup-OFF arm (record 1b round-2 Q2, option (ii)) refills the weights
    with ``update_weights_from_disk``, which ends in
    ``loader.load_weights_and_postprocess`` -- ``model.load_weights(iter)``
    followed by ``quant_method.process_weights_after_loading(module)`` for every
    module (``model_loader/loader.py:921-931``).  Neither half is idempotent on
    a quantized checkpoint, and the FIRST half is the one that breaks:

    * the post-load pass REPLACES the parameter rather than writing into it --
      ``layer.weight = Parameter(weight.t(), requires_grad=False)``
      (``compressed_tensors_w8a8_int8.py:151,159``, and the same shape in the
      AWQ/FP8 schemes).  DESK-MEASURED on this tree with the reference
      checkpoint's own scheme: a ``ModelWeightParameter`` of shape ``(4, 8)``
      carrying a ``weight_loader`` comes back a plain ``Parameter`` of shape
      ``(8, 4)`` with no ``weight_loader``
      (``test_quantized_post_load_replaces_the_weight_parameter``).
    * so the wake's ``model.load_weights(iter)`` resolves
      ``getattr(param, "weight_loader", default_weight_loader)`` to the DEFAULT
      loader, which asserts ``param.size() == loaded_weight.size()``
      (``weight_utils.py:1709``) against a transposed parameter and raises.
      ``model_runner.py:2857-2865`` then re-runs the same failing load as its
      rollback, OUTSIDE any ``try``.

    There is no second lane in this tree: #89's ``HibernateModelLoader`` is a
    ``BaseModelLoader``, which ``update_weights_from_disk`` rejects outright
    (``model_runner.py:2827-2829``), and it supports GGUF only.

    So this arm is UNDEFINED for a quantized checkpoint, and the honest form is
    a named refusal at the earliest decidable moment rather than a wake that
    commits the VMM pages and then fails inside the loader.  The two lanes that
    ARE defined, both named in the message: launch the group with
    ``--enable-weights-cpu-backup`` (the TMS restore carries the post-transform
    bytes and no reload runs at all), or serve an unquantized checkpoint.
    """
    if not quantization:
        return
    raise Weg2WakeRefused(
        f"W4 Weg2WakeRefused at {context}: this group runs the backup-OFF wake "
        f"arm (--enable-memory-saver without --enable-weights-cpu-backup) on a "
        f"{quantization!r} checkpoint. That arm refills the weights with "
        f"update_weights_from_disk, whose load_weights_and_postprocess writes "
        f"the checkpoint tensors into parameters a previous "
        f"process_weights_after_loading has already REPLACED with transposed, "
        f"weight_loader-less ones (loader.py:921-931; "
        f"compressed_tensors_w8a8_int8.py:151,159), so the refill raises inside "
        f"the loader and the model_runner rollback re-runs the same failing "
        f"load. Weights would be left committed and undefined. Launch this "
        f"group with --enable-weights-cpu-backup (the TMS restore carries the "
        f"post-transform bytes and no reload runs), or serve an unquantized "
        f"checkpoint. Refusing rather than waking into undefined weights."
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
    """The card key, or a refusal -- never a CUDA context bought to answer.

    ``current_device_uuid()`` falls back to torch when the pin is not readable
    from the environment, and that fallback INITIALISES CUDA.  The canonical
    holder of that guard is the flight recorder
    (``srt/mem_ledger/flight_recorder.card_pin_unresolvable_without_cuda``);
    this call is a USE of it, not a copy.  On the sleep path the context always
    exists, so the guard never fires there -- it exists because
    :func:`sleep_acceptance_census` is public and the S3 launcher calls it
    PRE-LAUNCH per rank, which is exactly the moment where a context created by
    the instrument would corrupt the very number the instrument reports.
    """
    if nvml_uuid is not None:
        return nvml_uuid
    from sglang.srt.mem_ledger.flight_recorder import (
        card_pin_unresolvable_without_cuda,
    )

    unresolved = card_pin_unresolvable_without_cuda()
    if unresolved is not None:
        raise RuntimeError(unresolved)

    from sglang.srt.registry import nvml as nvml_registry

    return nvml_registry.current_device_uuid()


def resolve_pcie_lock_key() -> str:
    """This process's physical-GPU key for the PCIe lock.

    Public on purpose: a caller that wants to degrade to "unserialised" when
    the card key cannot be resolved must be able to resolve the key in a
    ``try`` of its own, and then take the lock OUTSIDE that handler -- so a
    :class:`Weg2PcieLockTimeout` propagates by construction rather than by the
    ordering of two ``except`` clauses.
    """
    return _resolve_uuid(None)


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
    #: The tag set the graded release declared, as the caller resolved it.
    #: ``None`` means the caller declared none, i.e. the population is this
    #: process's whole device residency.  Printed on the line: a verdict that
    #: cannot be attributed to the request that produced it is not one.
    tags: Optional[Tuple[str, ...]]
    #: NVML ``nvmlDeviceGetComputeRunningProcesses`` for THIS pid, in bytes.
    proc_used_bytes: Optional[int]
    nvml_free_bytes: Optional[int]
    nvml_total_bytes: Optional[int]
    #: The caller's pre-pause reading of the SAME instrument, and the two
    #: criteria it may grade against.  ``None`` everywhere means the caller
    #: supplied no criterion, and then ``accepted`` is False by construction:
    #: a reading without a criterion is a number, not a verdict.
    before_bytes: Optional[int]
    released_bytes: Optional[int]
    min_released_fraction: Optional[float]
    #: False when ``min_released_fraction`` was supplied but the declared tag
    #: set is a PROPER SUBSET of :data:`WEG2_SLEEP_TAGS`, so the whole-process
    #: floor is not the criterion in force.  The line then prints the supplied
    #: value together with the reason it does not grade, never a bare number
    #: that reads as a criterion that ran.
    delta_form_in_force: bool
    expected_max_resident_bytes: Optional[int]
    #: Sums over ``kv_vmm_backing.arena_census()`` rows for this process.
    arena_reserved_bytes: int
    arena_backed_bytes: int
    arena_retained_bytes: int
    arena_rows: int
    #: Tri-state.  True/False only when a live ``KvVmmArena`` was actually
    #: censused: False means some arena still owns unmapped handles, and NVML
    #: charges the process for that ADDRESS SPACE, so the freed-memory reading
    #: is not one.  ``None`` means there was NO live arena to read -- the
    #: permanent state under Weg 2, where ``--enable-vram-dial`` is refused
    #: (W13) and the phase flip is deleted (S0/S7).  Reporting that case as
    #: True would be an assertion the instrument never took.
    retain_handles_asserted: Optional[bool]
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
        if self.arena_rows == 0:
            # No live KvVmmArena in this process: print n/a, never a 0.0 MiB
            # that reads like a measurement, and never `retain_handles=True`,
            # which would be an assertion nothing was asserted against.
            arena = (
                "arena_backed=n/a arena_retained=n/a retain_handles=n/a "
                "(no live KvVmmArena in this process)"
            )
        else:
            arena = (
                f"arena_backed={self.arena_backed_bytes / MIB:.1f} MiB "
                f"arena_retained={self.arena_retained_bytes / MIB:.1f} MiB "
                f"retain_handles={self.retain_handles_asserted}"
            )
        before = (
            "n/a" if self.before_bytes is None else f"{self.before_bytes / MIB:.1f}"
        )
        released = (
            "n/a" if self.released_bytes is None else f"{self.released_bytes / MIB:.1f}"
        )
        ceiling = (
            "n/a"
            if self.expected_max_resident_bytes is None
            else f"{self.expected_max_resident_bytes / MIB:.1f}"
        )
        if self.min_released_fraction is None:
            floor = "n/a"
        elif self.delta_form_in_force:
            floor = f"{self.min_released_fraction:.2f}"
        else:
            floor = (
                f"n/a (partial tag set: {list(self.tags or ())}; the floor is a "
                f"fraction of this process's whole device residency)"
            )
        tags = (
            "n/a (whole-process residency)"
            if self.tags is None
            else f"{list(self.tags)}"
        )
        return (
            f"[weg2 sleep-acceptance] uuid={self.nvml_uuid} pid={self.pid} "
            f"tags={tags} "
            f"proc_used={used} MiB nvml_free={free} MiB "
            f"proc_used_before={before} MiB released={released} MiB "
            f"min_released_fraction={floor} ceiling={ceiling} MiB "
            f"rows={self.arena_rows} {arena} accepted={self.accepted} "
            f"reason={self.refusal_reason or '-'} denominator={self.denominator}"
        )


_ProcessBytes = Union[Dict[int, int], Callable[[str], Dict[int, int]], None]


def sleep_acceptance_census(
    *,
    nvml_uuid: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    before_bytes: Optional[int] = None,
    min_released_fraction: Optional[float] = None,
    expected_max_resident_bytes: Optional[int] = None,
    _process_bytes: _ProcessBytes = None,
    _memory_info: Optional[Tuple[int, int]] = None,
    _arena_census: Optional[Dict[int, Dict[str, int]]] = None,
    _pid: Optional[int] = None,
) -> SleepAcceptanceCensus:
    """Read what this rank still holds on its card after ``pause()``, and GRADE it.

    Read-only and never raises: an instrument that can fail a boot is not an
    instrument (same contract as ``arena_census()``).  Every failure to read
    lands in ``refusal_reason`` and forces ``accepted=False``.

    The NVML half carries the verdict, and therefore it needs a criterion.  Two
    are accepted and both, if supplied, must hold:

    * the DELTA form -- ``before_bytes`` (the caller's pre-pause reading of this
      same instrument) plus ``min_released_fraction``.  This is the S1 form:
      the dormant floor D_c is unmeasured until S3, so there is no honest
      ceiling yet, but "the sleep released nothing" is decidable without one.
    * the CEILING form -- ``expected_max_resident_bytes``, the declared dormant
      ceiling.  S3's launcher supplies it once D_c is measured.

    ``tags`` is the POPULATION the verdict is about, and it is printed.  The
    delta floor is a fraction of this process's WHOLE device residency, so it
    may only grade a release that targets the whole of it
    (:data:`WEG2_SLEEP_TAGS`: weights AND kv_cache, the latter carrying the
    mamba/GDN anchors under the same tag).  A declared PROPER SUBSET -- the #89
    park's ``tags=["weights"]``, or a kv-only release -- suppresses the delta
    form, because both errors are reachable on this one endpoint: a weights-only
    park need not clear half of an awake residency that also carries KV, graphs,
    activations and the CUDA context (false FAIL), and a kv-only release on a
    rank whose KV exceeds half its residency clears the floor with the entire
    weights shard still resident (false PASS).  With the delta form suppressed
    the verdict rests on the ceiling form, or -- with no ceiling either -- on an
    explicit refusal.  ``tags=None`` means the caller declared no restriction,
    and then the whole-process denominator is the right one.

    Supplying NEITHER is itself refused: with no criterion the function would
    report ``accepted=True`` for a rank still holding its entire shard, i.e.
    for the exact silent-no-op condition it exists to catch, and a boot
    postmortem would quote that pass.  A gate that cannot fail is not a gate.

    The payload it reads -- NVML per-process bytes, card free/total, and the KV
    arena's own counters -- has a canonical holder in this tree:
    ``srt/mem_ledger/flight_recorder`` (``_nvml_view`` / ``_kv_arena_view``).
    This function is a VERDICT WRAPPER over that same payload, not a second
    collector: it shares the recorder's card-pin guard (see
    :func:`_resolve_uuid`) and adds only ``accepted`` / ``refusal_reason`` /
    ``denominator``.  A boot postmortem grades residency off the recorder's
    field names; the ``[weg2 sleep-acceptance]`` line is the flip path's
    verdict, not a second set of numbers to reconcile.

    The leading-underscore parameters are injection seams for the hermetic
    tests; production calls pass none of them.
    """
    pid = os.getpid() if _pid is None else int(_pid)
    reasons = []
    declared_tags: Optional[Tuple[str, ...]] = (
        None if tags is None else tuple(str(tag) for tag in tags)
    )
    partial_tag_set = declared_tags is not None and not WEG2_SLEEP_TAGS.issubset(
        set(declared_tags)
    )

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

    arena_rows = len(rows or {})
    # Tri-state, never a fabricated pass: with zero live arenas there is
    # nothing to assert against, so `retain_handles` is n/a and `accepted` is
    # decided by the NVML half ALONE.  Weg 2 makes the zero-row case permanent
    # (W13 refuses --enable-vram-dial, S0/S7 delete the phase flip), so a
    # `retain_handles_asserted=True` printed over an empty _LIVE_ARENAS would
    # be quoted in every boot postmortem as a check that never ran.
    retain_handles_asserted: Optional[bool] = None if arena_rows == 0 else retained == 0
    if retain_handles_asserted is False:
        reasons.append(
            f"arena retain_handles is in force: {retained / MIB:.1f} MiB of "
            f"unmapped-but-owned ADDRESS SPACE stays charged to this process "
            f"by NVML, so this reading is address space, not free memory"
        )
    if backed:
        reasons.append(f"arena still backs {backed / MIB:.1f} MiB of device memory")

    # --- the NVML half's criterion.  Without one this whole function is a
    # printer: it reported accepted=True at 30,154 MiB and accepted=True at
    # 1,294 MiB -- the two ends of campaign (a)'s own 28.9 GiB swing -- and the
    # number in its test was decorative.
    delta_supplied = before_bytes is not None and min_released_fraction is not None
    # The floor's denominator is the WHOLE process residency, so a release that
    # declared a proper subset of the sleep tags is not gradeable by it -- in
    # either direction (see the docstring).  Suppressed, never silently applied.
    delta_form = delta_supplied and not partial_tag_set
    ceiling_form = expected_max_resident_bytes is not None
    released_bytes: Optional[int] = None
    if not delta_form and not ceiling_form:
        if delta_supplied and partial_tag_set:
            reasons.append(
                "no residency criterion in force: this release declared the "
                f"partial tag set {list(declared_tags or ())}, and the delta "
                "floor is a fraction of this process's WHOLE device residency, "
                "which a release of only those tags need neither clear nor be "
                "graded by -- this reading is a number, not a verdict"
            )
        else:
            reasons.append(
                "no residency criterion supplied (neither before_bytes + "
                "min_released_fraction nor expected_max_resident_bytes) -- this "
                "reading is a number, not a verdict"
            )
    elif proc_used is not None:
        if delta_form:
            released_bytes = int(before_bytes) - proc_used
            floor = int(float(min_released_fraction) * int(before_bytes))
            if released_bytes < floor:
                reasons.append(
                    f"the sleep released {released_bytes / MIB:.1f} MiB of the "
                    f"{int(before_bytes) / MIB:.1f} MiB this process held "
                    f"before the pause, below the required "
                    f"{float(min_released_fraction):.0%} "
                    f"({floor / MIB:.1f} MiB) -- a sleep that frees nothing "
                    f"returns success and holds the whole shard"
                )
        if ceiling_form and proc_used > int(expected_max_resident_bytes):
            reasons.append(
                f"this process still holds {proc_used / MIB:.1f} MiB, above "
                f"the declared dormant ceiling "
                f"{int(expected_max_resident_bytes) / MIB:.1f} MiB"
            )

    accepted = proc_used is not None and not reasons
    criterion_denominator = (
        (
            "delta form SUPPRESSED (partial tag set "
            f"{list(declared_tags or ())}), no other criterion"
            if delta_supplied and partial_tag_set
            else "no criterion"
        )
        if not delta_form and not ceiling_form
        else ", ".join(
            part
            for part in (
                (
                    f"delta form (>= {float(min_released_fraction):.0%} of "
                    f"{int(before_bytes) / MIB:.1f} MiB released)"
                    if delta_form
                    else ""
                ),
                (
                    f"ceiling form (<= "
                    f"{int(expected_max_resident_bytes) / MIB:.1f} MiB)"
                    if ceiling_form
                    else ""
                ),
            )
            if part
        )
    )
    arena_denominator = (
        "no live KvVmmArena, so the arena half is n/a and acceptance rests on "
        "the NVML half alone"
        if arena_rows == 0
        else f"{arena_rows} live arena row(s)"
    )
    tag_denominator = (
        "no tag set declared, so the population is this process's whole device "
        "residency"
        if declared_tags is None
        else f"declared tags {list(declared_tags)}"
    )
    denominator = (
        f"NVML per-process bytes for pid={pid} on uuid={uuid} over "
        f"{proc_count} compute process(es); {tag_denominator}; "
        f"arena_rows={arena_rows} "
        f"({arena_denominator}); criterion: {criterion_denominator}"
    )
    return SleepAcceptanceCensus(
        pid=pid,
        nvml_uuid=uuid,
        tags=declared_tags,
        proc_used_bytes=proc_used,
        nvml_free_bytes=free_bytes,
        nvml_total_bytes=total_bytes,
        before_bytes=None if before_bytes is None else int(before_bytes),
        released_bytes=released_bytes,
        min_released_fraction=(
            None if min_released_fraction is None else float(min_released_fraction)
        ),
        delta_form_in_force=delta_form,
        expected_max_resident_bytes=(
            None
            if expected_max_resident_bytes is None
            else int(expected_max_resident_bytes)
        ),
        arena_reserved_bytes=reserved,
        arena_backed_bytes=backed,
        arena_retained_bytes=retained,
        arena_rows=arena_rows,
        retain_handles_asserted=retain_handles_asserted,
        accepted=accepted,
        refusal_reason="; ".join(reasons) if reasons else None,
        denominator=denominator,
    )
