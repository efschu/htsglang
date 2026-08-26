# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Everything that must be true before a turnkey boot is allowed to start.

Each check answers one question, returns either ``None`` or a named
:class:`~sglang.srt.turnkey.refusal.Refusal`, and never repairs anything. The
split matters: a preflight that fixes what it finds is a preflight whose
result cannot be trusted as a description of the machine, and #539's whole
premise is that the boot is reproducible from a described state.

Every probe is injected through :class:`Probes` so that each failure mode is
reachable in a hermetic test without a GPU, without a driver and without
touching a port. A check that can only be exercised on live hardware is a
check that will first run in production.

**What is deliberately NOT here.** No safety margins invented by this module,
no capping, no rounding of the operator's numbers. The checks are
physical-impossibility and identity checks. Choosing values that leave
headroom is the operator's job; second-guessing them would make the config a
suggestion.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import socket
from typing import Callable, Dict, List, Optional, Sequence

from sglang.srt.turnkey.config import StackConfig, assert_repo_stable
from sglang.srt.turnkey.refusal import (
    REFUSE_CARD_BUSY,
    REFUSE_CARD_CENSUS,
    REFUSE_CARD_UNKNOWN_UUID,
    REFUSE_DISK_HEADROOM,
    REFUSE_HOST_HEADROOM,
    REFUSE_PATH_MISSING,
    REFUSE_PORT_BUSY,
    REFUSE_WHEEL_DIST_SHADOW,
    REFUSE_WHEEL_SHADOW,
    Refusal,
    RefusalError,
    refuse,
)

__all__ = [
    "Probes",
    "CardObs",
    "DistObs",
    "ImportObs",
    "run_all",
    "default_probes",
]

MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class CardObs:
    """What preflight needs to know about one card. A narrow view of NVML's
    ``DeviceInfo`` + ``MemoryInfo`` so tests can build one in a line."""

    uuid: str
    name: str
    total_bytes: int
    free_bytes: int
    #: The NVML driver carve-out, as ``MemoryInfo.reserved_bytes``. It is
    #: excluded from ``free`` and included in ``used``, so it must be
    #: discounted before ``total - free`` can be read as FOREIGN occupancy.
    #: Defaults to 0, which is also what ``registry/nvml.py`` returns when it
    #: cannot read the v2 struct -- an unknown carve-out degrades to the old,
    #: conservative answer rather than silently widening the threshold.
    reserved_bytes: int = 0

    def foreign_bytes(self) -> int:
        """Occupancy that belongs to somebody else.

        ``total - free`` is not that number: NVML keeps the driver's own
        carve-out out of ``free`` and inside ``used``, so the subtraction
        charges the driver's reservation to whatever tenant is being looked
        for. On this rig that is ~425 MiB on a 3080 and ~518 on a 5090
        (measured; see ``uneven_perf.py`` and ``TERM_NVML_CARVE_OUT``), which
        is enough on its own to trip a 512 MiB threshold on an EMPTY card.
        """
        return max(0, self.total_bytes - self.free_bytes - self.reserved_bytes)


@dataclasses.dataclass
class Probes:
    """Injected access to the outside world.

    Defaults reach the real machine; tests replace them wholesale.
    """

    cards: Callable[[], Sequence[CardObs]]
    procs_on: Callable[[str], Dict[int, int]]
    mem_available_bytes: Callable[[], int]
    disk_free_bytes: Callable[[str], int]
    port_busy: Callable[[int], bool]
    path_exists: Callable[[str], bool]
    #: Returns ``(module_file, has_arm)`` for the pinned import, or raises
    #: ImportError. Kept as one probe because the two facts come from the
    #: same import and splitting them would allow a half-checked state.
    probe_import: Callable[[str, str], "ImportObs"]
    #: Every installed distribution that ships files under ``<package>/``.
    #: Deliberately NOT folded into ``probe_import``: that probe answers "what
    #: does importing give me", which is a fact about the winner, and this one
    #: answers "how many candidates were there", which is a fact about the
    #: installation. The #384 state this catches is one where the import
    #: answers correctly and the installation is still wrong.
    dist_providers: Callable[[str], Sequence["DistObs"]]
    #: ``() -> (fingerprint, is_cached)`` for the VRAM ledger's hardware
    #: calibration. A probe rather than a direct call so the "never probed"
    #: state is reachable in a hermetic test -- which is the state that
    #: matters, since it is the one every fresh rig starts in.
    #:
    #: Defaulted, unlike its siblings: every existing Probes construction in
    #: the tree predates it, and the check it feeds is opt-in
    #: (``require_vram_calibration``), so the inert "nothing cached" answer
    #: cannot change any existing outcome.
    vram_calibration: Callable[[], "CalibrationObs"] = staticmethod(
        lambda: CalibrationObs(fingerprint="", cached=False)
    )


@dataclasses.dataclass(frozen=True)
class CalibrationObs:
    """Whether this rig has a cached VRAM calibration, and under which key."""

    fingerprint: str
    cached: bool


@dataclasses.dataclass(frozen=True)
class ImportObs:
    module_file: str
    version: str
    has_arm: bool


@dataclasses.dataclass(frozen=True)
class DistObs:
    """One installed distribution providing an import package."""

    dist_name: str
    version: str
    recorded_files: int


# --- real probes ----------------------------------------------------------


def _real_cards() -> Sequence[CardObs]:
    # Imported lazily: preflight is unit-tested on machines with no driver,
    # and the registry module opens an NVML session at import-adjacent time.
    from sglang.srt.registry import nvml

    out: List[CardObs] = []
    for d in nvml.list_devices():
        mem = nvml.memory_info_for_uuid(d.uuid)
        out.append(
            CardObs(
                uuid=d.uuid,
                name=d.name,
                total_bytes=d.total_bytes,
                # The NVML FREE column, never total-used: the
                # driver carve-out is excluded from free and
                # included in used, so the subtraction reads ~0
                # and hides the shortfall.
                free_bytes=mem.free_bytes,
                # ...and carry the carve-out itself, so the
                # occupancy check can subtract the term instead of
                # comparing against a constant guessed above it.
                # NVML already measures this; preflight used to be
                # the one consumer in the tree that dropped it.
                reserved_bytes=mem.reserved_bytes,
            )
        )
    return out


def _real_procs_on(uuid: str) -> Dict[int, int]:
    from sglang.srt.registry import nvml

    return nvml.process_bytes_on_uuid(uuid)


logger = logging.getLogger(__name__)

#: Returned by :func:`_real_mem_available` when host RAM cannot be established
#: honestly. NEGATIVE on purpose, so it can never be mistaken for a size: a
#: sentinel of 0 would read as "no RAM available" and refuse every boot on a
#: box whose cgroup files are absent, which is the opposite of the rule the
#: owner module states -- "a caller that cannot get a number must say so, not
#: invent one", and refusing a boot on a fabricated figure is worse than not
#: checking.
MEM_AVAILABLE_UNKNOWN: int = -1

#: The scope the figure below describes, named because it is not the only
#: possible answer. Measured on this box, three cgroup levels report three
#: different `memory.current` -- root 23.5 GiB, system.slice 23.3 GiB,
#: system.slice/claude.service 22.3 GiB -- and all three carry
#: `memory.max = max`. A headroom refusal that prints a bare GiB figure cannot
#: be checked against any of them, because the reader cannot tell which
#: question was asked.
MEM_SCOPE = "cgroup /sys/fs/cgroup (container aggregate), #407 owner"


def _real_mem_available() -> int:
    """Host bytes a boot may believe, from the #407 owner.

    THIS USED TO READ ``/proc/meminfo`` DIRECTLY, and it gates a HARD refusal
    (``check_host_headroom`` -> ``REFUSE_HOST_HEADROOM``), so it is the boot
    gate rather than a diagnostic. Reading that file here over-reported free
    RAM twice over, and both halves are already written down elsewhere in this
    tree:

    * LXCFS SCOPE -- ``memtier/profile.py:honest_host_memory_bytes``: inside
      this container ``/proc/meminfo`` is synthesised, ``MemAvailable`` can
      EXCEED ``MemTotal`` (observed on this rig), and with ``memory.max``
      unlimited it reports the HOST's figures on a box other containers are
      also spending.
    * CGROUP RESIDENT -- the owner additionally clamps by what this cgroup
      already holds. Measured here: MemAvailable 113.19 GiB -> honest 112.95.

    A gate that over-reports free RAM ADMITS a boot that should have been
    refused, and #721 is that outcome on this box: a real container OOM with
    ``oom_kill=17``.

    #534, AN OPEN QUESTION CARRIED RATHER THAN GUESSED. CUDA pinned host memory
    is accounted in the cgroup's ``file`` bucket, not ``anon`` -- measured in
    commit c043235272, "the offload ledger reported 49.66 GiB of pinned pool
    while ``anon`` sat steady at 14.6 GiB", and confirmed live on this box
    (anon 2.12 GiB, file 19.06 GiB, current 21.39 GiB). The owner deliberately
    never charges ``file``, correctly, because page cache is reclaimable --
    pinned bytes there are not. Whether that makes this gate over-report AGAIN,
    on top of what is corrected here, is decidable only by watching anon/file
    across a real pin allocation, i.e. a boot. It is NOT guessed at here;
    ``scripts/window_871a_verify.py`` prints the pair so the next boot with
    pins answers it.
    """
    from sglang.srt.memtier.profile import host_memory_bytes_for_pinning

    try:
        _total, available = host_memory_bytes_for_pinning()
    except Exception:  # noqa: BLE001 - a probe may never break the preflight
        return MEM_AVAILABLE_UNKNOWN
    if available is None:
        return MEM_AVAILABLE_UNKNOWN
    return int(available)


def _real_disk_free(path: str) -> int:
    return shutil.disk_usage(path).free


def _real_port_busy(port: int) -> bool:
    """True when something already holds the port.

    A CONNECT probe, not a bind: binding to test would race with the very
    process we are about to start, and on SO_REUSEADDR a successful bind
    proves less than a refused connect.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _real_probe_import(module: str, attr: str) -> ImportObs:
    import importlib

    mod = importlib.import_module(module)
    return ImportObs(
        module_file=getattr(mod, "__file__", "") or "",
        version=str(getattr(mod, "__version__", "")),
        has_arm=bool(attr) and hasattr(mod, attr),
    )


def _real_dist_providers(package: str) -> Sequence[DistObs]:
    """Enumerate providing distributions WITHOUT importing the package.

    Importing is both impossible in some callers and hazardous in others (see
    ``sglang.srt.utils.kernel_dist_guard``), and it would answer the wrong
    question anyway: the import tells you who won, not how many were running.
    """
    from sglang.srt.utils.kernel_dist_guard import list_providers

    return [
        DistObs(
            dist_name=p.dist_name,
            version=p.version,
            recorded_files=p.recorded_files,
        )
        for p in list_providers(package)
    ]


def _real_vram_calibration() -> "CalibrationObs":
    """Read-only: does a cached calibration match this rig's fingerprint?

    ``load_calibration`` never measures as a side effect, which is what makes
    it usable here -- this module checks and never repairs. Running the probe
    from preflight would both violate that and touch a card before the boot
    is allowed to.
    """
    try:
        from sglang.srt.mem_ledger.calibration import (
            calibration_fingerprint,
            load_calibration,
        )
    except ImportError:
        return CalibrationObs(fingerprint="", cached=False)
    try:
        fingerprint = calibration_fingerprint() or ""
    except Exception:
        fingerprint = ""
    try:
        cached = load_calibration() is not None
    except Exception:
        cached = False
    return CalibrationObs(fingerprint=fingerprint, cached=cached)


def default_probes() -> Probes:
    return Probes(
        cards=_real_cards,
        procs_on=_real_procs_on,
        mem_available_bytes=_real_mem_available,
        disk_free_bytes=_real_disk_free,
        port_busy=_real_port_busy,
        path_exists=os.path.exists,
        probe_import=_real_probe_import,
        dist_providers=_real_dist_providers,
        vram_calibration=_real_vram_calibration,
    )


# --- the checks -----------------------------------------------------------


def check_repo(cfg: StackConfig) -> Optional[Refusal]:
    try:
        assert_repo_stable(cfg.repo, cfg.allow_worktree)
    except RefusalError as e:
        return e.refusal
    return None


def check_paths(cfg: StackConfig, p: Probes) -> Optional[Refusal]:
    for subject, path in (("stack.repo", cfg.repo), ("stack.venv", cfg.venv)):
        if not p.path_exists(path):
            return refuse(REFUSE_PATH_MISSING, subject, "absent", path)
    for lane in cfg.enabled_lanes():
        d = os.path.dirname(lane.boot_log)
        if d and not p.path_exists(d):
            return refuse(
                REFUSE_PATH_MISSING,
                f"serving.{lane.name}.boot_log",
                f"parent dir {d} absent",
                "an existing directory",
                remedy="the unit's ExecStartPre creates log_dir",
            )
    return None


def check_wheel(cfg: StackConfig, p: Probes) -> Optional[Refusal]:
    """#384 wheel-shadow.

    Three ways this fails, all silent at install time:
    the import resolves to the wrong tree, the version is the shadowing
    dist's, or the arm the fork adds is simply absent. The arm check is the
    one that matters -- a version bump alone is not evidence, because both
    dists ship a plausible version.
    """
    w = cfg.wheel
    if not w.enabled:
        return None
    for module in w.must_import:
        try:
            obs = p.probe_import(module, "int8_scaled_mm")
        except ImportError as e:
            return refuse(
                REFUSE_WHEEL_SHADOW,
                module,
                f"ImportError: {e}",
                f"an importable {module}",
            )
        if w.version and obs.version != w.version:
            return refuse(
                REFUSE_WHEEL_SHADOW,
                f"{module}.__version__",
                obs.version or "<none>",
                w.version,
                remedy="a shadowing dist owns the files; see " "rig-runbook 2.1",
            )
        if not obs.has_arm:
            return refuse(
                REFUSE_WHEEL_SHADOW,
                f"{module}.int8_scaled_mm",
                "absent",
                "present",
                remedy="the INT8 arm was dropped by a plain pip install; "
                "reinstall the pinned wheel per rig-runbook 2.1",
            )
        if w.expect_prefix and not obs.module_file.startswith(w.expect_prefix):
            return refuse(
                REFUSE_WHEEL_SHADOW,
                f"{module}.__file__",
                obs.module_file,
                f"a path under {w.expect_prefix}",
            )
    return None


def check_wheel_dist_shadow(cfg: StackConfig, p: Probes) -> Optional[Refusal]:
    """#384 standing reinstall block: is the shadow STRUCTURALLY possible?

    :func:`check_wheel` asks whether the shadow has already chosen wrongly.
    This asks the earlier question -- whether there is a choice to make at all
    -- and refuses while everything still looks fine. That is the point.

    Two distributions with different names providing the same import package
    are not a conflict to pip, so no install fails, no warning is printed, and
    which one owns the files is decided by whichever ran last. A venv in that
    state serves correctly right up until an unrelated ``pip install -U``
    touches the other dist, and then it serves an armless kernel with no event
    marking the change. The rig has been through that twice (#357's
    roll-forward/roll-back pair flipped the same files both ways).

    So the refusal is deliberately raised on a HEALTHY-looking machine. An
    operator reading it will object that the arm is present and the server
    works; both are true, and neither is stable. The remedy is to uninstall
    the loser, which removes the choice rather than winning it again.
    """
    w = cfg.wheel
    if not w.enabled:
        return None
    for module in w.must_import:
        providers = list(p.dist_providers(module))
        if len(providers) <= 1:
            continue
        names = ", ".join(
            f"{d.dist_name} {d.version} ({d.recorded_files} files)"
            for d in sorted(providers, key=lambda d: d.dist_name)
        )
        keep = w.dist or "the fork distribution"
        losers = [d.dist_name for d in providers if d.dist_name != w.dist]
        return refuse(
            REFUSE_WHEEL_DIST_SHADOW,
            f"distributions providing {module}",
            names,
            f"exactly one ({keep})",
            remedy=(
                f"pip uninstall -y {' '.join(losers) or '<the other dist>'} "
                f"and reinstall the pinned wheel, so the import name has one "
                f"owner; see docs/rig-runbook.md 2.1. Do this only while the "
                f"venv is QUIET -- removing the files under a running server "
                f"breaks it mid-flight."
            ),
        )
    return None


def check_cards(cfg: StackConfig, p: Probes) -> Optional[Refusal]:
    """Identity first, then occupancy.

    Identity before occupancy on purpose: "card busy" is only meaningful once
    we know the UUID names the card we think it does.
    """
    try:
        observed = list(p.cards())
    except Exception as e:  # driver absent, NVML unavailable
        return refuse(REFUSE_CARD_CENSUS, "nvml", f"unavailable: {e}", "a working NVML")
    by_uuid = {c.uuid: c for c in observed}

    for want in cfg.cards:
        got = by_uuid.get(want.uuid)
        if got is None:
            return refuse(
                REFUSE_CARD_UNKNOWN_UUID,
                want.label or want.uuid,
                "absent",
                want.uuid,
                remedy="present UUIDs: " + ", ".join(sorted(by_uuid)),
            )
        if want.expect_name and want.expect_name not in got.name:
            return refuse(
                REFUSE_CARD_CENSUS,
                want.label or want.uuid,
                got.name,
                f"a name containing {want.expect_name}",
            )

    # Occupancy: a card the config claims must not already carry foreign
    # VRAM. The orphan-container trap -- a dead-but-not-reaped tenant holds
    # gigabytes, the new boot OOMs, and the new process's own log shows only
    # its own (correct) allocations.
    used_uuids = set()
    for lane in cfg.enabled_lanes():
        for i in lane.cards:
            used_uuids.add(cfg.cards[i].uuid)
    for uuid in sorted(used_uuids):
        card = by_uuid[uuid]
        # FOREIGN occupancy, i.e. with the driver's own carve-out discounted.
        # Reading `total - free` here charged the carve-out to the tenant
        # being hunted, and refused an IDLE machine: the 5090 reports 521 MiB
        # in use with zero compute pids, against a shipped threshold of 512,
        # and the refusal's remedy ("stop the named pids") named none because
        # there were none. `card_busy_mib` is an allowance for genuine
        # foreign bytes; it was never meant to have to clear a hardware
        # constant that differs per card.
        foreign_mib = card.foreign_bytes() // MIB
        if foreign_mib > cfg.preflight.card_busy_mib:
            try:
                procs = p.procs_on(uuid)
            except Exception:
                procs = {}
            who = (
                ", ".join(
                    f"pid {pid}={b // MIB}MiB" for pid, b in sorted(procs.items())
                )
                or "no compute pids"
            )
            spec = cfg.card_by_uuid(uuid)
            carve_mib = card.reserved_bytes // MIB
            return refuse(
                REFUSE_CARD_BUSY,
                (spec.label if spec else "") or uuid,
                f"{foreign_mib} MiB foreign ({who}); "
                f"NVML driver carve-out {carve_mib} MiB already discounted",
                f"<= {cfg.preflight.card_busy_mib} MiB foreign",
                remedy="stop the named pids BY PID; never pkill -f, which "
                "also matches the router",
            )
    return None


def check_host_headroom(cfg: StackConfig, p: Probes) -> Optional[Refusal]:
    want = cfg.preflight.host_headroom_gib
    if want <= 0:
        return None
    got = p.mem_available_bytes()
    if got < 0:
        # #871b: NO HONEST NUMBER -> NO GUARD, never a refusal. The sentinel is
        # negative precisely so this branch is reachable: read as a size it
        # would be "less than any want" and refuse every boot on a box whose
        # cgroup files are absent. The owner module states the rule -- refusing
        # a boot on a fabricated figure is worse than not checking -- and this
        # is the one place in the preflight where getting that backwards costs
        # an instance that would have run.
        logger.warning(
            "preflight: host RAM could not be established honestly from %s; "
            "the %d GiB headroom gate is STANDING DOWN for this boot. Size the "
            "host against the machine yourself.",
            MEM_SCOPE,
            want,
        )
        return None
    if got < want * GIB:
        return refuse(
            REFUSE_HOST_HEADROOM,
            f"host RAM available [{MEM_SCOPE}]",
            f"{got / GIB:.1f} GiB",
            f">= {want} GiB",
            remedy="a boot that starts short of host RAM dies in "
            "weight load, after paying for it",
        )
    return None


def check_disk(cfg: StackConfig, p: Probes) -> Optional[Refusal]:
    for path, want_gib in cfg.preflight.disk_paths:
        try:
            got = p.disk_free_bytes(path)
        except OSError as e:
            return refuse(REFUSE_PATH_MISSING, path, str(e), "an existing path")
        if got < want_gib * GIB:
            return refuse(
                REFUSE_DISK_HEADROOM, path, f"{got / GIB:.1f} GiB", f">= {want_gib} GiB"
            )
    return None


def check_ports(cfg: StackConfig, p: Probes) -> Optional[Refusal]:
    """Every lane port must be free, and protected ports are never probed.

    The protected list exists because this rig runs a router on 30099 whose
    liveness is a standing law. A preflight that expected it free would, at
    best, refuse every boot; at worst it would invite somebody to free it.
    """
    wanted: List[tuple] = [
        (lane.port, f"serving.{lane.name}.port") for lane in cfg.enabled_lanes()
    ]
    wanted += [(port, "preflight.check_ports") for port in cfg.preflight.check_ports]
    for port, subject in wanted:
        if port in cfg.preflight.protected_ports:
            continue
        if p.port_busy(port):
            return refuse(
                REFUSE_PORT_BUSY,
                subject,
                f"port {port} answers",
                "a free port",
                remedy="find the holder with `ss -ltnp` and stop it " "by pid",
            )
    return None


def check_vram_calibration(cfg: StackConfig, p: Probes) -> Optional[Refusal]:
    """The VRAM ledger is the sizing authority; say so before the boot, not
    after it has quietly fallen back.

    ``enable_vram_ledger`` is ON by default: the ledger prices every card and
    the inherited heuristic runs only where a term cannot be priced. That
    fallback is safe -- it boots, and it names the term in the log -- but it
    is also easy to never notice, and a rig sized by the falsified
    ``512 + tokens*1.5`` catch-all is exactly what the ledger exists to
    replace.

    So this states the fact at the one place an operator reads before a boot.
    It is a WARNING-shaped refusal only in the sense that it names a remedy;
    whether an uncalibrated rig may boot is the config's decision
    (``preflight.require_vram_calibration``), defaulting to permissive so a
    fresh rig is not bricked by a default it never chose.
    """
    if not getattr(cfg.preflight, "require_vram_calibration", False):
        return None
    obs = p.vram_calibration()
    if obs.cached:
        return None
    return Refusal(
        name="vram_calibration_missing",
        subject="VRAM ledger hardware calibration",
        observed=(
            f"no cached calibration for fingerprint "
            f"{obs.fingerprint or '<unresolved>'}"
        ),
        expected="a calibration cached for this card set, driver and torch build",
        remedy=(
            "run `python -m sglang.srt.mem_ledger.probe` once on this rig, or "
            "set preflight.require_vram_calibration = false to boot on the "
            "inherited heuristic (which the ledger will name in the log)"
        ),
    )


def run_all(cfg: StackConfig, p: Optional[Probes] = None) -> List[Refusal]:
    """Run every check and return ALL refusals, in a stable order.

    All of them rather than the first: an operator woken by a failed boot
    should learn everything wrong with the machine in one pass, not discover
    the next problem after fixing this one. The boot path still stops on any
    non-empty result.
    """
    p = p or default_probes()
    out: List[Refusal] = []
    for fn in (check_repo,):
        r = fn(cfg)
        if r:
            out.append(r)
    for fn in (
        check_paths,
        check_wheel,
        check_wheel_dist_shadow,
        check_cards,
        check_host_headroom,
        check_disk,
        check_ports,
        check_vram_calibration,
    ):
        try:
            r = fn(cfg, p)
        except RefusalError as e:
            r = e.refusal
        if r:
            out.append(r)
    return out
