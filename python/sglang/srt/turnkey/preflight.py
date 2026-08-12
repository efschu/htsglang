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
    REFUSE_WHEEL_SHADOW,
    Refusal,
    RefusalError,
    refuse,
)

__all__ = ["Probes", "CardObs", "Probes", "run_all", "default_probes"]

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


@dataclasses.dataclass(frozen=True)
class ImportObs:
    module_file: str
    version: str
    has_arm: bool


# --- real probes ----------------------------------------------------------


def _real_cards() -> Sequence[CardObs]:
    # Imported lazily: preflight is unit-tested on machines with no driver,
    # and the registry module opens an NVML session at import-adjacent time.
    from sglang.srt.registry import nvml

    out: List[CardObs] = []
    for d in nvml.list_devices():
        mem = nvml.memory_info_for_uuid(d.uuid)
        out.append(CardObs(uuid=d.uuid, name=d.name, total_bytes=d.total_bytes,
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
                           reserved_bytes=mem.reserved_bytes))
    return out


def _real_procs_on(uuid: str) -> Dict[int, int]:
    from sglang.srt.registry import nvml

    return nvml.process_bytes_on_uuid(uuid)


def _real_mem_available() -> int:
    with open("/proc/meminfo", "r") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


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


def default_probes() -> Probes:
    return Probes(
        cards=_real_cards,
        procs_on=_real_procs_on,
        mem_available_bytes=_real_mem_available,
        disk_free_bytes=_real_disk_free,
        port_busy=_real_port_busy,
        path_exists=os.path.exists,
        probe_import=_real_probe_import,
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
            return refuse(REFUSE_PATH_MISSING, f"serving.{lane.name}.boot_log",
                          f"parent dir {d} absent", "an existing directory",
                          remedy="the unit's ExecStartPre creates log_dir")
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
            return refuse(REFUSE_WHEEL_SHADOW, module, f"ImportError: {e}",
                          f"an importable {module}")
        if w.version and obs.version != w.version:
            return refuse(REFUSE_WHEEL_SHADOW, f"{module}.__version__",
                          obs.version or "<none>", w.version,
                          remedy="a shadowing dist owns the files; see "
                                 "rig-runbook 2.1")
        if not obs.has_arm:
            return refuse(
                REFUSE_WHEEL_SHADOW, f"{module}.int8_scaled_mm", "absent",
                "present",
                remedy="the INT8 arm was dropped by a plain pip install; "
                       "reinstall the pinned wheel per rig-runbook 2.1")
        if w.expect_prefix and not obs.module_file.startswith(w.expect_prefix):
            return refuse(REFUSE_WHEEL_SHADOW, f"{module}.__file__",
                          obs.module_file, f"a path under {w.expect_prefix}")
    return None


def check_cards(cfg: StackConfig, p: Probes) -> Optional[Refusal]:
    """Identity first, then occupancy.

    Identity before occupancy on purpose: "card busy" is only meaningful once
    we know the UUID names the card we think it does.
    """
    try:
        observed = list(p.cards())
    except Exception as e:  # driver absent, NVML unavailable
        return refuse(REFUSE_CARD_CENSUS, "nvml", f"unavailable: {e}",
                      "a working NVML")
    by_uuid = {c.uuid: c for c in observed}

    for want in cfg.cards:
        got = by_uuid.get(want.uuid)
        if got is None:
            return refuse(
                REFUSE_CARD_UNKNOWN_UUID, want.label or want.uuid, "absent",
                want.uuid,
                remedy="present UUIDs: " + ", ".join(sorted(by_uuid)))
        if want.expect_name and want.expect_name not in got.name:
            return refuse(REFUSE_CARD_CENSUS, want.label or want.uuid,
                          got.name, f"a name containing {want.expect_name}")

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
            who = ", ".join(f"pid {pid}={b // MIB}MiB"
                            for pid, b in sorted(procs.items())) or "no compute pids"
            spec = cfg.card_by_uuid(uuid)
            carve_mib = card.reserved_bytes // MIB
            return refuse(
                REFUSE_CARD_BUSY, (spec.label if spec else "") or uuid,
                f"{foreign_mib} MiB foreign ({who}); "
                f"NVML driver carve-out {carve_mib} MiB already discounted",
                f"<= {cfg.preflight.card_busy_mib} MiB foreign",
                remedy="stop the named pids BY PID; never pkill -f, which "
                       "also matches the router")
    return None


def check_host_headroom(cfg: StackConfig, p: Probes) -> Optional[Refusal]:
    want = cfg.preflight.host_headroom_gib
    if want <= 0:
        return None
    got = p.mem_available_bytes()
    if got < want * GIB:
        return refuse(REFUSE_HOST_HEADROOM, "MemAvailable",
                      f"{got / GIB:.1f} GiB", f">= {want} GiB",
                      remedy="a boot that starts short of host RAM dies in "
                             "weight load, after paying for it")
    return None


def check_disk(cfg: StackConfig, p: Probes) -> Optional[Refusal]:
    for path, want_gib in cfg.preflight.disk_paths:
        try:
            got = p.disk_free_bytes(path)
        except OSError as e:
            return refuse(REFUSE_PATH_MISSING, path, str(e), "an existing path")
        if got < want_gib * GIB:
            return refuse(REFUSE_DISK_HEADROOM, path, f"{got / GIB:.1f} GiB",
                          f">= {want_gib} GiB")
    return None


def check_ports(cfg: StackConfig, p: Probes) -> Optional[Refusal]:
    """Every lane port must be free, and protected ports are never probed.

    The protected list exists because this rig runs a router on 30099 whose
    liveness is a standing law. A preflight that expected it free would, at
    best, refuse every boot; at worst it would invite somebody to free it.
    """
    wanted: List[tuple] = [(lane.port, f"serving.{lane.name}.port")
                           for lane in cfg.enabled_lanes()]
    wanted += [(port, "preflight.check_ports")
               for port in cfg.preflight.check_ports]
    for port, subject in wanted:
        if port in cfg.preflight.protected_ports:
            continue
        if p.port_busy(port):
            return refuse(REFUSE_PORT_BUSY, subject, f"port {port} answers",
                          "a free port",
                          remedy="find the holder with `ss -ltnp` and stop it "
                                 "by pid")
    return None


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
    for fn in (check_paths, check_wheel, check_cards, check_host_headroom,
               check_disk, check_ports):
        try:
            r = fn(cfg, p)
        except RefusalError as e:
            r = e.refusal
        if r:
            out.append(r)
    return out
