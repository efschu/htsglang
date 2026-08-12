# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#695: host shmem as a priced axis, measured per rank at the at-rest point.

WHY THIS EXISTS
---------------
``CardVramLedger`` prices VRAM to the MiB and refuses a boot that cannot fit.
Host RAM had no equivalent, and one class of host allocation is both large and
invisible: page-locked ``MAP_SHARED`` memory. It is invisible three times over.

1.  ``/dev/shm`` does not show it. These mappings are ANONYMOUS shared
    (``MAP_SHARED|MAP_ANONYMOUS``); the kernel renders them in
    ``/proc/<pid>/maps`` as ``/dev/zero (deleted)`` and they belong to no
    visible tmpfs. On the boot below ``df /dev/shm`` reported 1.3 GB used
    while the ranks held 75 GiB.
2.  cgroup v2 files them under ``file``, not ``anon``, so every guard that
    reads ``memory.stat`` counted them as reclaimable page cache. They are not
    reclaimable: page-locked, and on a swapless box there is nowhere to
    reclaim them TO. Fixed in ``memtier/profile.py:honest_host_memory_bytes``.
3.  Nothing summed them. ``pinned_host_budget`` sums the pinned pools that
    REGISTER a post; the phase-flip weight images never did, so 72 GiB of
    pinned host RAM was outside the only ledger that adds host posts up.

THE BOOT IT COST, and the reason the output below is one grep-able line.
Measured 2026-08-12 on the PP=3 INT8 boot, three ``sglang::scheduler`` ranks::

    Pss_Shmem   34.62 + 17.85 + 26.23 GiB  = 75.0 GiB
    cgroup /.lxc  memory.stat shmem        = 75.07 GiB
                  memory.current           = 99.06 GiB
                  memory.events oom_kill   = 9
    /proc/meminfo SwapTotal                = 0

Nine cumulative cgroup OOM kills, one of which presented as a silent rank
death -- a rank that "just disappeared" while every GPU-side ledger said the
configuration fitted, because GPU-free was never the binding constraint. An
operator looking at that boot had no single line to read. That is what this
module produces, at the same at-rest instant the #485 residency census uses.

WHAT IT REPORTS
---------------
Per rank, by OWNER CLASS, in exclusively-owned bytes:

* ``anon-shared`` -- ``MAP_SHARED|MAP_ANONYMOUS``. The class that matters:
  pinned host images, ``cudaHostAlloc``/``cudaHostRegister`` regions.
* ``shm-nccl`` / ``shm-named`` -- ``/dev/shm`` segments, split because NCCL's
  are transport buffers of a size nobody chose and the named ones are ours.
* ``driver`` -- ``/dev/nvidia*``. Reported and then EXCLUDED from the host-RAM
  total: these are device/BAR mappings, not host pages, and charging them
  would inflate the very number this module exists to make trustworthy.
* ``file-shared`` -- everything else shared and file-backed.

EXCLUSIVELY OWNED means ``Pss``, not ``Rss``. A shmem page mapped by two
processes is charged ONCE to the cgroup, so summing ``Rss`` across ranks
double-counts and the total stops reconciling against ``memory.stat``. On the
measured boot the ``Pss`` sum (75.0 GiB) and the cgroup's ``shmem`` (75.07
GiB) agree to 0.1%, which is the check that the classification is honest.

Read-only: nothing here allocates, frees or advises anything.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "HOST-SHMEM"

_MIB = 1 << 20
_GIB = 1 << 30

#: Mappings at or above this size are named individually in the log line. The
#: point of the line is to make an OOM diagnosable in one grep, and a 16 GiB
#: image is the thing a reader needs to see; a 2 MiB workspace is not.
BIG_MAPPING_BYTES = 256 * _MIB

#: Residual (measured minus declared) above which the line is a WARNING. Set
#: at 1 GiB: below that the unattributed remainder is workspaces and driver
#: bookkeeping, above it something large is unpriced -- which is exactly the
#: state this module was written after.
RESIDUAL_WARN_BYTES = 1 * _GIB

CLASS_ANON_SHARED = "anon-shared"
CLASS_SHM_NCCL = "shm-nccl"
CLASS_SHM_NAMED = "shm-named"
CLASS_DRIVER = "driver"
CLASS_FILE_SHARED = "file-shared"

#: Classes that are genuinely host RAM. ``driver`` is measured but not summed:
#: see the module docstring.
HOST_RAM_CLASSES = (
    CLASS_ANON_SHARED,
    CLASS_SHM_NCCL,
    CLASS_SHM_NAMED,
    CLASS_FILE_SHARED,
)

_MAP_RE = re.compile(
    r"^([0-9a-f]+)-([0-9a-f]+) (\S{4}) (\S+) (\S+) (\S+)\s*(.*)$", re.IGNORECASE
)
_FIELD_RE = re.compile(r"^(\w+):\s+(\d+) kB")


@dataclass(frozen=True)
class SharedMapping:
    """One shared VMA, with the bytes this process exclusively owns."""

    start: int
    size_bytes: int
    pss_bytes: int
    rss_bytes: int
    path: str
    owner_class: str


@dataclass
class HostShmemCensus:
    """What one process holds in shared mappings, by class."""

    pid: int
    by_class_pss: Dict[str, int] = field(default_factory=dict)
    by_class_count: Dict[str, int] = field(default_factory=dict)
    big_mappings: List[SharedMapping] = field(default_factory=list)
    #: cgroup v2 facts, any of them None when unreadable.
    cgroup_shmem: Optional[int] = None
    cgroup_current: Optional[int] = None
    cgroup_max: Optional[int] = None
    oom_kills: Optional[int] = None
    swap_free: Optional[int] = None
    #: Sum of the pinned host posts registered in this process.
    declared_bytes: int = 0
    declared_posts: Tuple[Tuple[str, int], ...] = ()

    @property
    def host_ram_pss(self) -> int:
        """Exclusively-owned host bytes. Driver mappings deliberately absent."""
        return sum(self.by_class_pss.get(name, 0) for name in HOST_RAM_CLASSES)

    @property
    def residual_bytes(self) -> int:
        """Measured host shmem that no registered post accounts for."""
        return self.host_ram_pss - self.declared_bytes


def classify_mapping(path: str, perms: str) -> Optional[str]:
    """Owner class for a VMA, or None when it is not a shared mapping.

    Classification is on the SHARED bit, not on the path: an anonymous shared
    mapping has no path of its own and the kernel labels it ``/dev/zero
    (deleted)``, which is indistinguishable by name from a real ``/dev/zero``
    mapping and must not be matched by name alone.
    """
    if not perms.endswith("s"):
        return None
    base = path.strip()
    if base.startswith("/dev/nvidia"):
        return CLASS_DRIVER
    if not base or base.startswith("/dev/zero"):
        return CLASS_ANON_SHARED
    if base.startswith("/dev/shm/"):
        name = base[len("/dev/shm/") :]
        return CLASS_SHM_NCCL if name.startswith("nccl-") else CLASS_SHM_NAMED
    if base.startswith("/memfd:") or base.startswith("[") or base.startswith("anon_"):
        return CLASS_ANON_SHARED
    return CLASS_FILE_SHARED


def parse_shared_mappings(smaps_path: str) -> List[SharedMapping]:
    """Every shared VMA in a ``smaps`` file. Missing/unreadable -> empty.

    Takes a PATH rather than a pid so the parser is testable against a
    recorded fixture -- the shapes that matter (``/dev/zero (deleted)``, a
    deleted ``/dev/shm`` segment) are awkward to produce on demand and easy to
    record.
    """
    out: List[SharedMapping] = []
    cur_start = 0
    cur_size = 0
    cur_perms = ""
    cur_path = ""
    cur_pss = 0
    cur_rss = 0
    have = False

    def _flush() -> None:
        if not have:
            return
        owner = classify_mapping(cur_path, cur_perms)
        if owner is None:
            return
        out.append(
            SharedMapping(
                start=cur_start,
                size_bytes=cur_size,
                pss_bytes=cur_pss,
                rss_bytes=cur_rss,
                path=cur_path or "<anon-shared>",
                owner_class=owner,
            )
        )

    try:
        with open(smaps_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                header = _MAP_RE.match(line)
                if header:
                    _flush()
                    cur_start = int(header.group(1), 16)
                    cur_size = int(header.group(2), 16) - cur_start
                    cur_perms = header.group(3)
                    cur_path = header.group(7).strip()
                    cur_pss = 0
                    cur_rss = 0
                    have = True
                    continue
                field_match = _FIELD_RE.match(line)
                if field_match and have:
                    key = field_match.group(1)
                    if key == "Pss":
                        cur_pss = int(field_match.group(2)) * 1024
                    elif key == "Rss":
                        cur_rss = int(field_match.group(2)) * 1024
        _flush()
    except OSError:
        return []
    return out


def _read_cgroup_facts() -> (
    Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]
):
    """``(shmem, current, max, oom_kills, swap_free)`` for THIS process.

    Reads the process's OWN cgroup from ``/proc/self/cgroup`` rather than the
    namespace root. Inside this LXC container the two differ: the ranks live
    in ``/.lxc``, and on the measured boot that cgroup reported ``shmem``
    75.07 GiB and ``oom_kill`` 9 while the namespace root -- whose accounting
    the kernel does not fully maintain -- reported neither.
    """
    rel = ""
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split(":", 2)
                if len(parts) == 3 and parts[0] == "0":
                    rel = parts[2].lstrip("/")
                    break
    except OSError:
        rel = ""
    root = os.path.join("/sys/fs/cgroup", rel) if rel else "/sys/fs/cgroup"
    if not os.path.isdir(root):
        root = "/sys/fs/cgroup"

    shmem = current = limit = ooms = None
    try:
        with open(os.path.join(root, "memory.stat"), "r", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(" ")
                if key == "shmem":
                    shmem = int(rest.strip())
                    break
    except (OSError, ValueError):
        pass
    try:
        with open(os.path.join(root, "memory.current"), "r", encoding="utf-8") as h:
            current = int(h.read().strip())
    except (OSError, ValueError):
        pass
    try:
        with open(os.path.join(root, "memory.max"), "r", encoding="utf-8") as h:
            raw = h.read().strip()
        limit = None if raw == "max" else int(raw)
    except (OSError, ValueError):
        pass
    try:
        with open(os.path.join(root, "memory.events"), "r", encoding="utf-8") as h:
            for line in h:
                key, _, rest = line.partition(" ")
                if key == "oom_kill":
                    ooms = int(rest.strip())
                    break
    except (OSError, ValueError):
        pass

    swap_free = None
    try:
        from sglang.srt.memtier.profile import _swap_headroom_bytes

        swap_free = _swap_headroom_bytes(root)
    except Exception:  # noqa: BLE001 -- an instrument never breaks a boot
        swap_free = None
    return shmem, current, limit, ooms, swap_free


def collect_host_shmem_census(pid: Optional[int] = None) -> HostShmemCensus:
    """Measure this process's shared mappings. Never raises."""
    target = os.getpid() if pid is None else int(pid)
    who = "self" if pid is None else str(target)
    census = HostShmemCensus(pid=target)
    for mapping in parse_shared_mappings(f"/proc/{who}/smaps"):
        census.by_class_pss[mapping.owner_class] = (
            census.by_class_pss.get(mapping.owner_class, 0) + mapping.pss_bytes
        )
        census.by_class_count[mapping.owner_class] = (
            census.by_class_count.get(mapping.owner_class, 0) + 1
        )
        if mapping.size_bytes >= BIG_MAPPING_BYTES:
            census.big_mappings.append(mapping)
    census.big_mappings.sort(key=lambda m: m.size_bytes, reverse=True)
    (
        census.cgroup_shmem,
        census.cgroup_current,
        census.cgroup_max,
        census.oom_kills,
        census.swap_free,
    ) = _read_cgroup_facts()
    try:
        from sglang.srt.mem_cache.pinned_host_budget import registered_posts

        posts = registered_posts()
        census.declared_posts = tuple((p.name, int(p.nbytes)) for p in posts)
        census.declared_bytes = sum(int(p.nbytes) for p in posts)
    except Exception:  # noqa: BLE001 -- an instrument never breaks a boot
        census.declared_posts = ()
        census.declared_bytes = 0
    return census


def render_host_shmem_line(census: HostShmemCensus, rank: Optional[int] = None) -> str:
    """The one line. Fixed prefix, fixed key order, one grep."""
    who = f"rank{rank}" if rank is not None else f"pid{census.pid}"
    parts = [
        f"{LOG_PREFIX} {who}",
        f"host={census.host_ram_pss / _GIB:.2f}GiB",
        f"declared={census.declared_bytes / _GIB:.2f}GiB",
        f"residual={census.residual_bytes / _GIB:+.2f}GiB",
    ]
    for name in HOST_RAM_CLASSES + (CLASS_DRIVER,):
        by = census.by_class_pss.get(name)
        if not by:
            continue
        note = " (not host RAM)" if name == CLASS_DRIVER else ""
        parts.append(
            f"{name}={by / _GIB:.2f}GiB"
            f"/n={census.by_class_count.get(name, 0)}{note}"
        )
    if census.cgroup_shmem is not None:
        parts.append(f"cgroup_shmem={census.cgroup_shmem / _GIB:.2f}GiB")
    if census.cgroup_current is not None:
        parts.append(f"cgroup_current={census.cgroup_current / _GIB:.2f}GiB")
    parts.append(
        "cgroup_max="
        + (
            "unset"
            if census.cgroup_max is None
            else f"{census.cgroup_max / _GIB:.2f}GiB"
        )
    )
    if census.swap_free is not None:
        parts.append(f"swap_free={census.swap_free / _GIB:.2f}GiB")
    if census.oom_kills is not None:
        parts.append(f"oom_kills={census.oom_kills}")
    if census.big_mappings:
        big = " ".join(
            f"{m.size_bytes / _MIB:.0f}MiB:{m.owner_class}"
            for m in census.big_mappings[:8]
        )
        parts.append(f"big=[{big}]")
    return " ".join(parts)


def log_host_shmem_census(rank: Optional[int] = None) -> Optional[HostShmemCensus]:
    """Emit the boot line. Always on, and never able to break a boot.

    NOT env-gated, unlike the #485 residency census, and the difference is
    deliberate. That census is a calibration instrument an operator asks for.
    This one is the line somebody needs to have ALREADY been written when a
    rank dies at 03:00 with exit code -9 and no traceback -- an instrument
    that has to be requested in advance is no use to the boot that needed it.
    It costs one ``/proc/self/smaps`` walk, once, at the at-rest point.
    """
    try:
        census = collect_host_shmem_census()
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s census unavailable: %s", LOG_PREFIX, exc)
        return None
    line = render_host_shmem_line(census, rank=rank)
    if census.residual_bytes >= RESIDUAL_WARN_BYTES:
        logger.warning(
            "%s -- %.2f GiB of host shmem is held by no registered post. "
            "It is page-locked and, with swap at %s, unreclaimable; it is "
            "charged to the cgroup under `file`, so it will not appear in any "
            "`anon` figure. If this rank is later SIGKILLed with no "
            "traceback, this is the first line to read.",
            line,
            census.residual_bytes / _GIB,
            "0" if not census.swap_free else f"{census.swap_free / _GIB:.2f}GiB",
        )
    else:
        logger.info("%s", line)
    return census
