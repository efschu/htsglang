# Copyright 2026 SGLang Team
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
"""The comm-benchmark suite: one short run, one shareable artifact (#271).

WHAT IT IS
----------
One button. Under it, a fixed list of ARMS that each answer one question
about how this rig moves bytes -- inside a card, between two cards, between
two processes, over a wire -- and a rig profile that says what the rig is.
The whole thing is budgeted for **<= 90 s locally**, because a benchmark
nobody finishes running is a benchmark nobody runs.

Every arm ends in one of four states, and all four are DATA:

``ok``
    it ran and the numbers are here.
``warn``
    it ran, but something about the run limits what the numbers mean
    (clock throttling, a spread wider than the noise floor, a degraded
    transport). The reservation is named in ``notes``.
``error``
    it failed. The failure text is kept verbatim (scrubbed) -- a timeout, an
    OOM, a missing library and an exception are all findings about this rig,
    and swallowing them would make the shared artifact a survivorship
    sample.
``absent``
    it did not run, and ``absent_reason`` says why in one sentence. "Nobody
    measured this" and "this is zero" stay distinguishable (the three-word
    provenance rule of §8.1/§8.5, with ``error`` added because a failure is
    not the same as an absence).

THE NOISE FLOOR COMES FIRST
---------------------------
Harness rule 5: nothing is reported under the detection limit. The suite runs
ONE cell twice, back to back, as an A-vs-A pair before anything else, and
carries the A/B delta into the artifact. Every later comparison between arms
is read against that number; the UI prints it next to the arms rather than
in a footnote.

WHAT LEAVES THE MACHINE
-----------------------
Nothing from this module. The suite is one SOURCE of the shared rig artifact
and owns no part of the share path: :func:`to_sections` hands
:mod:`sglang.srt.planner.rig_artifact` the same
``Measurement`` / ``Capability`` / ``ErrorSignature`` rows the hardware-profile
source produces, and that module does the curation, the anonymization gate,
the preview and the #152 submit for both. A second source must not be a
second copy of the scrub.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sglang.srt.planner import rig_artifact

__all__ = [
    "ARMS",
    "SOURCE_NAME",
    "ArmResult",
    "ArmSpec",
    "CommSuiteJob",
    "CommSuiteJobStore",
    "JOBS",
    "arm_specs_json",
    "rig_profile",
    "to_sections",
]

#: How this source names itself in the shared artifact.
SOURCE_NAME = "comm_suite"

#: The decode-shaped size ladder, in KiB. Same three cells every arm sweeps,
#: so the arms are comparable without a conversion: 20 KiB = one hidden row
#: of a 5120-wide model in fp32 staging (bs=1 decode), 80 KiB = a 4-token MTP
#: verify, 256 KiB = a small prefill chunk.
SIZES_KIB = (20, 80, 256)

#: The direct-vs-staged size ladder for the ``gdr_crossover`` arm
#: (docs/EVAL_gdr_uebernahme.md §9 P2, window W1): the handover's original
#: four points (8 B, 4 KiB, 64 KiB, 1 MiB) plus 16/32/256 KiB to bracket this
#: fork's own decode (20 KiB) and verify (80 KiB) sizes. Kept as strings, not
#: an int tuple like SIZES_KIB, because the ladder mixes byte and KiB units.
GDR_CROSSOVER_SIZES: Tuple[str, ...] = (
    "8B", "4KiB", "16KiB", "32KiB", "64KiB", "256KiB", "1MiB")

#: Where a locally built copy of the handover's ping-pong binary would live.
#: Never vendored into this tree (EVAL_gdr_uebernahme.md §7: the reference C
#: files are MIT but version-locked to the installed open-kernel-module
#: driver's ioctl struct layout), so the arm only ever looks for it out of
#: tree, at a path this env var can override.
GDR_CROSSOVER_BIN_ENV = "SGLANG_GDR_CROSSOVER_BIN"
_GDR_CROSSOVER_DEFAULT_BIN = "/spinning/gdr-uebergabe/gpurdma_04_bench"

#: Iteration counts. Chosen against the 90 s box, not against precision: the
#: suite reports a spread per cell, so a reader can see when 60 iterations
#: were not enough rather than being told a tight number that isn't.
ITERS_CPU = 60
ITERS_GPU = 60
WARMUP = 10

#: Per-arm wall-clock ceilings. A hung collective is the failure mode this
#: whole family of code has (see the rank-local-before-collective note), so
#: every arm is killed rather than allowed to hold the button forever. A
#: timeout is recorded as an ``error`` arm with its budget named.
JOB_TTL_S = 3600.0

#: Card locks, runbook §7.1 (v2: one lock per physical card + a quiet flag).
LOCK_DIR_FMT = "/tmp/gpu-card-{}.lock"
LEGACY_LOCK_DIR = "/tmp/gpu-owner.lock"
QUIET_LOCK_DIR = "/tmp/gpu-quiet.lock"
#: A quiet flag older than this is presumed stale (§7.1).
QUIET_STALE_S = 15 * 60

PENDING, RUNNING, OK, ERROR, CANCELLED = (
    "pending", "running", "ok", "error", "cancelled")


# ===========================================================================
# Arm catalogue
# ===========================================================================
@dataclass(frozen=True)
class ArmSpec:
    """One measurement and the question it answers.

    ``kind`` decides what the arm needs before it may run:

    ``inventory``
        reads state, measures nothing, always runs.
    ``cpu``
        processes on this host, no CUDA context. Runs while a GPU window
        belongs to somebody else -- that is why the suite is useful on a busy
        rig at all.
    ``gpu``
        needs a card window (§7.1 locks + a card whose ``memory.used`` is
        near zero). Absent, with the holder named, when it cannot get one.
    ``network``
        needs a reachable peer. Absent with the §8 host-runner sentence when
        the container cannot reach the fast line.
    """

    id: str
    label: str
    kind: str
    question: str
    budget_s: float
    note: str = ""

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "question": self.question,
            "budget_s": self.budget_s,
            "note": self.note,
        }


ARMS: Tuple[ArmSpec, ...] = (
    ArmSpec(
        id="rig_profile",
        label="Rig profile",
        kind="inventory",
        question="What is this rig, in the terms a comparison needs?",
        budget_s=5.0,
        note="Card models, counts and VRAM; driver, CUDA, torch, NCCL, UCX. "
             "Models and counts only -- no serial, no UUID, no host.",
    ),
    ArmSpec(
        id="noise_floor",
        label="Noise floor (A vs A)",
        kind="cpu",
        question="How much does the SAME cell move between two runs?",
        budget_s=12.0,
        note="The detection limit for every other arm. Two identical gloo "
             "runs back to back; the delta between their medians is the "
             "floor under which no difference in this suite is reportable.",
    ),
    ArmSpec(
        id="collective_gloo",
        label="gloo (CPU reference)",
        kind="cpu",
        question="What do the stock CPU collectives cost at decode sizes?",
        budget_s=20.0,
        note="torch.distributed/gloo all_reduce + all_gather over loopback, "
             "world 2. The reference point that makes a transport number "
             "readable.",
    ),
    ArmSpec(
        id="collective_barlink_ucx",
        label="barlink / UCX",
        kind="cpu",
        question="What does the UCX transport cost per collective?",
        budget_s=35.0,
        note="scripts/r3val/link_collective_cost.py at the same three sizes, "
             "split into stage / post / wait / finish. Over loopback here; "
             "the same harness measures a real wire from the host.",
    ),
    ArmSpec(
        id="byte_gate",
        label="Exactness gate",
        kind="cpu",
        question="Are the transport's results bit-exact across its "
                 "thresholds?",
        budget_s=30.0,
        note="link_collective_cost.py --gate: every collective checked "
             "against the computed reference at atol 0, including the ragged "
             "sizes the ring pads. A correctness line, not a timing line.",
    ),
    ArmSpec(
        id="card_probe",
        label="Cards + pair matrix",
        kind="gpu",
        question="What does each card do, and what does each ORDERED pair "
                 "cost?",
        budget_s=60.0,
        note="rigmon.card_probe: membw, GEMM, fp8, H2D/D2H per card, plus "
             "the ordered pair matrix with the transport each pair actually "
             "took (cuda p2p vs host staging).",
    ),
    ArmSpec(
        id="collective_nccl",
        label="NCCL (intra-rig)",
        kind="gpu",
        question="What do the vendor collectives cost across the local "
                 "cards?",
        budget_s=90.0,
        note="One rank per card, same three sizes. The number a TP rank "
             "actually pays on this box.",
    ),
    ArmSpec(
        id="collective_barlink_shm",
        label="barlink / shm",
        kind="gpu",
        question="What does the single-node shared-memory transport cost?",
        budget_s=60.0,
        note="all_reduce only -- the transport implements no all_gather. "
             "Counted as a GPU arm because it pins its segment to a CUDA "
             "device for zero-copy H2D/D2H, so it cannot run device-free.",
    ),
    ArmSpec(
        id="gdr_crossover",
        label="dmabuf GPU-RDMA crossover (direct vs staged)",
        kind="gpu",
        question="At what payload size does a direct dmabuf RDMA write stop "
                 "beating a host-staged one, on THIS rig?",
        budget_s=40.0,
        note="Runs the handover's gpurdma_04_bench ping-pong "
             "(docs/EVAL_gdr_uebernahme.md) across "
             "8B/4/16/32/64/256 KiB/1 MiB and reports the crossover as a "
             "property of THIS rig, not of GPU RDMA in general (§1.4/§9 P2) "
             "-- a rig with Resizable BAR on every card may cross over "
             "somewhere else entirely. Needs the binary built out of tree "
             "(SGLANG_GDR_CROSSOVER_BIN) plus an exclusive card window; "
             "absent, naming both, when either is missing.",
    ),
    ArmSpec(
        id="cross_rig",
        label="Cross-rig link",
        kind="network",
        question="What does the wire to a second rig cost?",
        budget_s=30.0,
        note="Only from a runner that can reach the fast line. A dev "
             "container cannot, and the arm says so instead of substituting "
             "a loopback number for a wire.",
    ),
)

ARM_BY_ID: Dict[str, ArmSpec] = {a.id: a for a in ARMS}


def arm_specs_json() -> List[dict]:
    """The catalogue, for the page to draw before anything has run."""
    return [a.to_json() for a in ARMS]


@dataclass
class ArmResult:
    """One arm's outcome. ``status`` is the traffic light."""

    arm_id: str
    status: str = "absent"          # ok | warn | error | absent
    elapsed_s: Optional[float] = None
    cells: Dict[str, dict] = field(default_factory=dict)
    facts: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None
    absent_reason: Optional[str] = None

    def to_json(self) -> dict:
        spec = ARM_BY_ID.get(self.arm_id)
        return {
            "id": self.arm_id,
            "label": spec.label if spec else self.arm_id,
            "kind": spec.kind if spec else "cpu",
            "question": spec.question if spec else "",
            "note": spec.note if spec else "",
            "status": self.status,
            "elapsed_s": self.elapsed_s,
            "cells": self.cells,
            "facts": self.facts,
            "notes": list(self.notes),
            "error": self.error,
            "absent_reason": self.absent_reason,
        }


# ===========================================================================
# Rig profile
# ===========================================================================
def _run(cmd: List[str], timeout: float = 20.0, env: Optional[dict] = None
         ) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, check=False, env=env)
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout:g} s"
    except Exception as e:  # pragma: no cover - defensive
        return 1, "", f"{type(e).__name__}: {e}"


def _nvidia_smi_cards() -> List[dict]:
    rc, out, _ = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,driver_version,"
        "clocks.sm,clocks.max.sm",
        "--format=csv,noheader,nounits",
    ], timeout=15.0)
    if rc != 0:
        return []
    cards: List[dict] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue

        def _i(v):
            try:
                return int(float(v))
            except Exception:
                return None

        cards.append({
            "index": _i(parts[0]),
            "name": parts[1],
            "vram_mib": _i(parts[2]),
            "used_mib": _i(parts[3]),
            "driver": parts[4],
            "sm_clock_mhz": _i(parts[5]) if len(parts) > 5 else None,
            "sm_clock_max_mhz": _i(parts[6]) if len(parts) > 6 else None,
        })
    return cards


def _card_summary(cards: Sequence[dict]) -> str:
    """``2x RTX 3080 + 1x RTX 5090`` -- model and count, nothing else."""
    counts: Dict[str, int] = {}
    for c in cards:
        name = re.sub(r"^NVIDIA\s+GeForce\s+", "", c.get("name") or "?").strip()
        counts[name] = counts.get(name, 0) + 1
    return " + ".join(f"{n}x {name}" for name, n in sorted(counts.items()))


def _ucx_version() -> Optional[str]:
    rc, out, _ = _run(["ucx_info", "-v"], timeout=10.0)
    if rc == 0:
        m = re.search(r"(\d+\.\d+\.\d+)", out)
        if m:
            return m.group(1)
    try:
        import ctypes

        lib = ctypes.CDLL("libucp.so.0")
        major = ctypes.c_uint()
        minor = ctypes.c_uint()
        rel = ctypes.c_uint()
        lib.ucp_get_version(ctypes.byref(major), ctypes.byref(minor),
                            ctypes.byref(rel))
        return f"{major.value}.{minor.value}.{rel.value}"
    except Exception:
        return None


def _repo_commit() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    rc, out, _ = _run(["git", "-C", here, "rev-parse", "--short=10", "HEAD"],
                      timeout=10.0)
    return out.strip() if rc == 0 and out.strip() else None


def rig_profile() -> dict:
    """What this rig IS -- card models and counts, versions, nothing else.

    Deliberately NOT anonymized here: the whole artifact goes through
    :func:`scrub_tree` in one pass, so this function stays a plain reader and
    there is exactly one place that decides what is identifying.
    """
    cards = _nvidia_smi_cards()
    prof: dict = {
        "cards": [
            {k: c.get(k) for k in
             ("index", "name", "vram_mib", "sm_clock_mhz", "sm_clock_max_mhz")}
            for c in cards
        ],
        "card_summary": _card_summary(cards) or "no CUDA cards visible",
        "card_count": len(cards),
        "driver": cards[0]["driver"] if cards else None,
        "ucx": _ucx_version(),
        "commit": _repo_commit(),
        "python": "%d.%d" % sys.version_info[:2],
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        prof["torch"] = torch.__version__
        prof["cuda"] = torch.version.cuda
        try:
            prof["nccl"] = ".".join(str(x) for x in torch.cuda.nccl.version())
        except Exception:
            prof["nccl"] = None
        prof["arch_list"] = list(torch.cuda.get_arch_list())[-6:]
    except Exception as e:
        prof["torch"] = None
        prof["torch_error"] = f"{type(e).__name__}: {e}"
    return prof


# ===========================================================================
# Card window (runbook §7.1)
# ===========================================================================
def _live_identity_map():
    """The canonical UUID <-> index resolver, or ``None`` without NVML."""
    from sglang.srt.registry import nvml

    if not nvml.is_available():
        return None
    return nvml.identity_map()


class _CardWindow:
    """Per-card locks, taken in ascending NVML order, released together.

    Follows §7.1 v2 exactly, and the two rules that are easy to get wrong:

    * a card is free when its ``memory.used`` is near zero -- a PVE-host boot
      is invisible to the container's compute-apps query (PID namespace), so
      an empty process list alone proves nothing;
    * the quiet flag is honored: while it exists (and is not stale), no new
      GPU-heavy phase starts. The suite's GPU arms go ``absent`` with the
      flag named rather than pushing into somebody's measurement window.

    The lock *path* stays ``/tmp/gpu-card-<NVML index>.lock`` because five
    independent tools -- this module, ``scripts/gpu_battery``,
    ``scripts/p2p_readiness``, ``scripts/probe/*`` and the host-side battery
    -- arbitrate through exactly that name, and a scheme only this file
    understands would be no arbitration at all. What AUDIT #331 changes is the
    *content*: every ``info`` file now records the ``uuid`` and ``pci_bus_id``
    of the card the index meant when the lock was taken. A reader can then
    tell which physical card is held, and a lock directory that outlived a
    re-enumeration is visible as a uuid that no longer matches the card now
    sitting at that index, instead of silently reading as "card 1 is busy".
    """

    OWNER = "comm-suite"
    FREE_MIB = 512

    def __init__(self, indices: Sequence[int], identity=None):
        self.indices = sorted(indices)
        self.held: List[int] = []
        self.reason: Optional[str] = None
        self._identity = identity or _live_identity_map
        self._map = None

    def _read_info(self, path: str) -> str:
        try:
            with open(os.path.join(path, "info")) as f:
                return f.read()
        except Exception:
            return ""

    def _field_of(self, path: str, key: str) -> str:
        prefix = f"{key}="
        for line in self._read_info(path).splitlines():
            if line.startswith(prefix):
                return line.split("=", 1)[1].strip()
        return ""

    def _owner_of(self, path: str) -> str:
        return self._field_of(path, "owner") or "unknown"

    def _card_of(self, idx: int):
        """The physical card behind ``idx`` right now, or ``None``."""
        # ``None`` means not resolved yet, ``False`` means resolved to nothing
        # (no NVML on this host). One attempt per window, not per card.
        if self._map is None:
            try:
                self._map = self._identity()
            except Exception:  # noqa: BLE001 - a desk host has no NVML
                self._map = False
        if not self._map:
            return None
        return self._map.by_nvml_index(idx)

    def _held_card_note(self, path: str, idx: int) -> str:
        """How a foreign lock's recorded card compares with the live one."""
        recorded = self._field_of(path, "uuid")
        card = self._card_of(idx)
        if not recorded:
            return (" (the lock records no card uuid, so which physical card "
                    "it covers cannot be verified)")
        if card is None:
            return f" (lock names card {recorded})"
        if card.uuid == recorded:
            return f" (card {card.name}, {recorded})"
        return (
            f" (the lock names {recorded} but NVML index {idx} is now "
            f"{card.uuid} / {card.name}: the lock outlived a re-enumeration)"
        )

    def acquire(self) -> bool:
        if os.path.isdir(QUIET_LOCK_DIR):
            age = time.time() - os.path.getmtime(QUIET_LOCK_DIR)
            if age < QUIET_STALE_S:
                self.reason = (
                    f"a quiet window is open ({self._owner_of(QUIET_LOCK_DIR)}"
                    f", {age / 60:.0f} min old): §7.1 forbids starting a new "
                    f"GPU phase while it exists")
                return False
        if os.path.isdir(LEGACY_LOCK_DIR):
            self.reason = (
                f"the rig-wide legacy lock is held by "
                f"{self._owner_of(LEGACY_LOCK_DIR)}, which means all cards "
                f"are taken (§7.1 legacy compatibility)")
            return False
        for idx in self.indices:
            path = LOCK_DIR_FMT.format(idx)
            try:
                os.mkdir(path)
            except FileExistsError:
                self.reason = (
                    f"card {idx} is held by {self._owner_of(path)}"
                    f"{self._held_card_note(path, idx)}; the GPU "
                    f"arms need every local card at once")
                self.release()
                return False
            except Exception as e:
                self.reason = f"could not take the lock for card {idx}: {e}"
                self.release()
                return False
            card = self._card_of(idx)
            try:
                with open(os.path.join(path, "info"), "w") as f:
                    f.write(
                        f"owner={self.OWNER}\n"
                        f"purpose=comm-benchmark suite (#271), GPU arms\n"
                        f"nvml_index={idx}\n")
                    # The durable half of the identity. The index is in the
                    # path and in the line above; only these two survive a
                    # re-enumeration (AUDIT #331).
                    if card is not None:
                        f.write(
                            f"uuid={card.uuid}\n"
                            f"pci_bus_id={card.pci_bus_id}\n"
                            f"card_name={card.name}\n")
                    f.write(
                        f"acquired={time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime())}\n")
            except Exception:
                pass
            self.held.append(idx)
        busy = [c for c in _nvidia_smi_cards()
                if (c.get("used_mib") or 0) > self.FREE_MIB
                and c.get("index") in self.indices]
        if busy:
            names = ", ".join(
                f"card {c['index']} holds {c['used_mib']} MiB" for c in busy)
            self.reason = (
                f"the locks were free but the cards are not: {names}. A "
                f"host-side process is invisible here, so memory.used is the "
                f"check that counts (§7.1)")
            self.release()
            return False
        return True

    def release(self) -> None:
        for idx in list(self.held):
            path = LOCK_DIR_FMT.format(idx)
            try:
                try:
                    os.unlink(os.path.join(path, "info"))
                except FileNotFoundError:
                    pass
                os.rmdir(path)
            except Exception:
                pass
            self.held.remove(idx)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


# ===========================================================================
# Arm runners
# ===========================================================================
def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _repo_root() -> str:
    # .../python/sglang/srt/planner/comm_suite.py -> repo root
    here = os.path.abspath(__file__)
    return os.path.abspath(os.path.join(os.path.dirname(here), *([os.pardir] * 4)))


def _script(name: str) -> str:
    return os.path.join(_repo_root(), "scripts", "r3val", name)


def _comm_dir() -> str:
    return os.path.join(_repo_root(), "python", "sglang", "srt", "distributed",
                        "device_communicators")


def _worker_env(extra: Optional[dict] = None) -> dict:
    env = dict(os.environ)
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(_free_port())
    env["GLOO_SOCKET_IFNAME"] = env.get("GLOO_SOCKET_IFNAME", "lo")
    env["UCX_TLS"] = env.get("COMM_SUITE_UCX_TLS", "tcp,self,sm")
    # A hung rank must not outlive the arm's budget; the runner kills the
    # process group, and this keeps torch from waiting half an hour first.
    env.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    if extra:
        env.update(extra)
    return env


def _run_ranks(
    argv_for_rank: Callable[[int], List[str]],
    world: int,
    out_path: str,
    budget_s: float,
    env: dict,
    register: Optional[Callable[[subprocess.Popen], None]] = None,
) -> Tuple[dict, str]:
    """Launch ``world`` rank processes, wait inside ``budget_s``, read the JSON.

    Rank 0 writes the artifact; the others only have to finish. Every process
    is started in its own process group so a timeout kills the whole rank,
    not just the shell in front of it -- the hang family this repo keeps
    meeting is exactly "one rank waits in a collective forever".
    """
    procs: List[subprocess.Popen] = []
    logs: List[str] = []
    deadline = time.time() + budget_s
    try:
        for r in range(world):
            p = subprocess.Popen(
                argv_for_rank(r), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, env=env,
                start_new_session=True)
            procs.append(p)
            if register:
                register(p)
        for p in procs:
            remaining = max(deadline - time.time(), 0.1)
            try:
                out, _ = p.communicate(timeout=remaining)
                logs.append(out or "")
            except subprocess.TimeoutExpired:
                raise TimeoutError(
                    f"a rank did not finish inside the arm budget of "
                    f"{budget_s:g} s; killed. Last output: "
                    f"{(''.join(logs))[-400:]}")
    finally:
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), 9)
                except Exception:
                    pass
    log = "".join(logs)
    bad = [p for p in procs if p.returncode not in (0, None)]
    if not os.path.exists(out_path):
        rcs = ", ".join(str(p.returncode) for p in procs)
        raise RuntimeError(
            f"the arm produced no result file (rank exit codes: {rcs}). "
            f"Output: {log[-800:]}")
    with open(out_path) as f:
        data = json.load(f)
    if bad:
        data.setdefault("_rank_failures",
                        [p.returncode for p in bad])
    return data, log


def _tmp_out(tag: str) -> str:
    import tempfile

    fd, path = tempfile.mkstemp(prefix=f"commsuite_{tag}_", suffix=".json")
    os.close(fd)
    os.unlink(path)
    return path


def _worker_arm(backend: str, world: int, budget_s: float, iters: int,
                register=None, env_extra: Optional[dict] = None
                ) -> Tuple[dict, str]:
    out = _tmp_out(backend)
    comm_dir = _comm_dir()

    def argv(r: int) -> List[str]:
        a = [sys.executable, _script("comm_suite_worker.py"),
             "--backend", backend, "--rank", str(r), "--world", str(world),
             "--iters", str(iters), "--warmup", str(WARMUP),
             "--sizes", ",".join(str(s) for s in SIZES_KIB),
             "--comm-dir", comm_dir]
        if r == 0:
            a += ["--out", out]
        return a

    try:
        return _run_ranks(argv, world, out, budget_s,
                          _worker_env(env_extra), register)
    finally:
        try:
            os.unlink(out)
        except Exception:
            pass


def _ucx_arm(op: str, budget_s: float, register=None) -> Tuple[dict, str]:
    out = _tmp_out(f"ucx_{op}")
    comm_dir = _comm_dir()

    def argv(r: int) -> List[str]:
        a = [sys.executable, _script("link_collective_cost.py"),
             "--rank", str(r), "--world", "2", "--iters", str(ITERS_CPU),
             "--warmup", str(WARMUP), "--op", op,
             "--sizes", ",".join(str(s) for s in SIZES_KIB),
             "--comm-dir", comm_dir]
        if r == 0:
            a += ["--out", out]
        return a

    try:
        return _run_ranks(argv, 2, out, budget_s, _worker_env(), register)
    finally:
        try:
            os.unlink(out)
        except Exception:
            pass


def _spread_note(cells: Dict[str, dict], floor_pct: Optional[float]
                 ) -> List[str]:
    """Name every cell whose spread is wider than the measured noise floor.

    Not a failure -- a reservation. A wide cell means the median is a weaker
    statement than it looks, and the artifact says so next to the number
    instead of in a caveat nobody reads.
    """
    notes: List[str] = []
    if floor_pct is None:
        return notes
    wide = [k for k, v in cells.items()
            if (v.get("spread_pct") or 0) > max(3.0 * floor_pct, 25.0)]
    if wide:
        notes.append(
            "spread wider than 3x the noise floor in: " + ", ".join(sorted(wide))
            + " -- read those medians as an order of magnitude, not a figure")
    return notes


# --- individual arms -------------------------------------------------------
def _arm_rig_profile(ctx: "_RunCtx") -> ArmResult:
    prof = rig_profile()
    res = ArmResult(arm_id="rig_profile", status="ok", facts=prof)
    if not prof.get("cards"):
        res.status = "warn"
        res.notes.append(
            "no CUDA card is visible to this process; every GPU arm will be "
            "absent for the same reason")
    throttled = [c for c in prof.get("cards", [])
                 if c.get("sm_clock_mhz") and c.get("sm_clock_max_mhz")
                 and c["sm_clock_mhz"] < 0.6 * c["sm_clock_max_mhz"]]
    if throttled:
        res.status = "warn"
        res.notes.append(
            "cards below 60 % of their clock ceiling at profile time: "
            + ", ".join(str(c["index"]) for c in throttled)
            + " -- idle downclock or a thermal cap; timings taken now are "
              "not comparable with a warm rig")
    return res


def _arm_noise_floor(ctx: "_RunCtx") -> ArmResult:
    """A vs A: the same gloo cell twice. Harness rule 5, first, always."""
    res = ArmResult(arm_id="noise_floor")
    a, _ = _worker_arm("gloo", 2, 6.0, 40, ctx.register)
    b, _ = _worker_arm("gloo", 2, 6.0, 40, ctx.register)
    key = f"all_reduce/{SIZES_KIB[0]}KiB"
    am = (a.get("cells", {}).get(key) or {}).get("median_us")
    bm = (b.get("cells", {}).get(key) or {}).get("median_us")
    if am is None or bm is None:
        res.status = "error"
        res.error = f"the A/A cell {key} is missing from one of the two runs"
        return res
    delta = abs(am - bm) / max((am + bm) / 2.0, 1e-9) * 100.0
    res.cells = {"A": a["cells"][key], "B": b["cells"][key]}
    res.facts = {
        "cell": key,
        "a_median_us": am,
        "b_median_us": bm,
        "floor_pct": round(delta, 1),
    }
    ctx.floor_pct = round(delta, 1)
    res.status = "ok" if delta <= 15.0 else "warn"
    res.notes.append(
        f"two identical runs of {key} differ by {delta:.1f} %. Nothing in "
        f"this artifact is a difference below that.")
    if delta > 15.0:
        res.notes.append(
            "a floor this wide means the box was busy during the run; "
            "compare arms only when they differ by much more than it")
    return res


def _arm_collective_gloo(ctx: "_RunCtx") -> ArmResult:
    res = ArmResult(arm_id="collective_gloo")
    data, _ = _worker_arm("gloo", 2, ARM_BY_ID["collective_gloo"].budget_s,
                          ITERS_CPU, ctx.register)
    res.cells = data.get("cells", {})
    res.facts = {"world": data.get("world"), "iters": data.get("iters"),
                 "exact_mismatches": data.get("exact_mismatches")}
    res.status = "ok" if data.get("exact_mismatches") == 0 else "error"
    if res.status == "error":
        res.error = (f"gloo returned {data.get('exact_mismatches')} inexact "
                     f"all_reduce results -- a correctness finding, not a "
                     f"timing one")
    res.notes.extend(_spread_note(res.cells, ctx.floor_pct))
    if res.status == "ok" and res.notes:
        res.status = "warn"
    return res


def _arm_collective_barlink_ucx(ctx: "_RunCtx") -> ArmResult:
    res = ArmResult(arm_id="collective_barlink_ucx")
    budget = ARM_BY_ID["collective_barlink_ucx"].budget_s / 2.0
    cells: Dict[str, dict] = {}
    facts: Dict[str, Any] = {}
    for op in ("all_reduce", "all_gather"):
        data, _ = _ucx_arm(op, budget, ctx.register)
        cells.update(data.get("cells", {}))
        facts.setdefault("rndv_thresh", data.get("rndv_thresh"))
        facts.setdefault("fp32_reduce", data.get("fp32_reduce"))
        facts["world"] = data.get("world")
    res.cells = cells
    res.facts = facts
    res.status = "ok"
    res.notes.append(
        "loopback, world 2: this measures the transport's fixed cost per "
        "collective, not a wire. The stage/post/wait/finish split is the "
        "part that transfers to a real link.")
    res.notes.extend(_spread_note(cells, ctx.floor_pct))
    return res


def _arm_byte_gate(ctx: "_RunCtx") -> ArmResult:
    """Correctness, not timing: --gate checks every collective at atol 0."""
    res = ArmResult(arm_id="byte_gate")
    comm_dir = _comm_dir()

    def argv(r: int) -> List[str]:
        return [sys.executable, _script("link_collective_cost.py"),
                "--rank", str(r), "--world", "2", "--gate",
                "--comm-dir", comm_dir]

    procs: List[subprocess.Popen] = []
    budget = ARM_BY_ID["byte_gate"].budget_s
    deadline = time.time() + budget
    env = _worker_env()
    outs: List[str] = []
    try:
        for r in range(2):
            p = subprocess.Popen(argv(r), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 env=env, start_new_session=True)
            procs.append(p)
            ctx.register(p)
        for p in procs:
            out, _ = p.communicate(timeout=max(deadline - time.time(), 0.1))
            outs.append(out or "")
    except subprocess.TimeoutExpired:
        res.status = "error"
        res.error = f"the exactness gate did not finish inside {budget:g} s"
        return res
    finally:
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), 9)
                except Exception:
                    pass
    log = "".join(outs)
    m = re.search(r"gate:\s+(\d+)\s+mismatches", log)
    mismatches = int(m.group(1)) if m else None
    rcs = [p.returncode for p in procs]
    res.facts = {"mismatches": mismatches, "rank_returncodes": rcs}
    if mismatches is None:
        res.status = "error"
        res.error = ("the gate produced no verdict line; exit codes "
                     f"{rcs}. Output: {log[-500:]}")
    elif mismatches == 0:
        res.status = "ok"
        res.notes.append(
            "every all_reduce and all_gather across the ring threshold "
            "(including the ragged sizes the ring pads) matched the computed "
            "reference exactly")
    else:
        res.status = "error"
        res.error = (f"{mismatches} collectives did NOT match the reference "
                     f"at atol 0 -- the transport is wrong on this box, and "
                     f"no timing on it means anything until that is fixed")
    return res


def _arm_card_probe(ctx: "_RunCtx") -> ArmResult:
    res = ArmResult(arm_id="card_probe")
    from sglang.srt.rigmon import card_probe

    profile, _path = card_probe._run_probe_subprocess("local")
    prof = profile.to_json()
    cards = prof.get("cards") or prof.get("measurements") or []
    res.facts = {
        "cards": cards,
        "pairs": prof.get("pairs", []),
    }
    res.status = "ok"
    transports = {p.get("transport") for p in prof.get("pairs", [])
                  if isinstance(p, dict)}
    if transports:
        res.notes.append("pair transports seen: " + ", ".join(
            sorted(t for t in transports if t)))
    if transports and "cuda p2p" not in transports:
        res.notes.append(
            "no pair took a direct P2P route; every cross-card copy is host "
            "staging through pinned memory. That is this hardware's real "
            "transfer rate, not a degraded measurement")
    return res


def _arm_collective_nccl(ctx: "_RunCtx") -> ArmResult:
    res = ArmResult(arm_id="collective_nccl")
    world = ctx.card_count
    if world < 2:
        res.status = "absent"
        res.absent_reason = (
            f"NCCL across cards needs at least two cards; this rig shows "
            f"{world}")
        return res
    data, _ = _worker_arm("nccl", world,
                          ARM_BY_ID["collective_nccl"].budget_s, ITERS_GPU,
                          ctx.register)
    res.cells = data.get("cells", {})
    res.facts = {"world": data.get("world"),
                 "exact_mismatches": data.get("exact_mismatches")}
    res.status = "ok" if data.get("exact_mismatches") == 0 else "error"
    if res.status == "error":
        res.error = (f"NCCL returned {data.get('exact_mismatches')} inexact "
                     f"all_reduce results")
    res.notes.extend(_spread_note(res.cells, ctx.floor_pct))
    if res.status == "ok" and res.notes:
        res.status = "warn"
    return res


def _arm_collective_barlink_shm(ctx: "_RunCtx") -> ArmResult:
    res = ArmResult(arm_id="collective_barlink_shm")
    world = min(max(ctx.card_count, 2), 2)
    data, _ = _worker_arm("barlink_shm", world,
                          ARM_BY_ID["collective_barlink_shm"].budget_s,
                          ITERS_CPU, ctx.register)
    res.cells = data.get("cells", {})
    res.facts = {"world": data.get("world"),
                 "slot_bytes": data.get("slot_bytes"),
                 "exact_mismatches": data.get("exact_mismatches")}
    res.status = "ok" if data.get("exact_mismatches") == 0 else "error"
    if res.status == "error":
        res.error = (f"the shm transport returned "
                     f"{data.get('exact_mismatches')} inexact all_reduce "
                     f"results")
    res.notes.append(
        "all_reduce only: BarlinkShmTransport implements no all_gather, and "
        "the suite does not synthesize a cell the transport does not have")
    res.notes.extend(_spread_note(res.cells, ctx.floor_pct))
    return res


def _arm_cross_rig(ctx: "_RunCtx") -> ArmResult:
    """Reachability first, measurement second -- and honest when it cannot.

    The dev container has no route to the fast line (runbook §1.1/§1.2), so
    on the reference rig this arm is ``absent`` with the host-runner sentence
    rather than a loopback number wearing a wire's label.
    """
    res = ArmResult(arm_id="cross_rig")
    targets: List[str] = []
    try:
        from sglang.srt.rigmon import pairing

        for sess in pairing.STORE.list():
            t = getattr(sess, "target", None)
            if t:
                targets.append(str(t))
    except Exception as e:
        res.notes.append(f"the pairing store could not be read: {e}")
    env_target = os.environ.get("COMM_SUITE_PEER")
    if env_target:
        targets.append(env_target)
    targets = [t for t in dict.fromkeys(targets) if t]
    if not targets:
        res.status = "absent"
        res.absent_reason = (
            "no paired peer is known and COMM_SUITE_PEER is unset. Pair a "
            "rig on the Pair-rig tab, or run this arm from a host that can "
            "reach the fast line (runbook §8 host-runner pattern) -- a "
            "loopback figure would not describe a wire.")
        return res
    reachable: List[dict] = []
    for t in targets[:4]:
        host = t.replace("http://", "").replace("https://", "").split("/")[0]
        hostname, _, port = host.partition(":")
        t0 = time.perf_counter()
        try:
            with socket.create_connection((hostname, int(port or 80)),
                                          timeout=3.0):
                pass
            reachable.append({"rtt_ms": round(
                (time.perf_counter() - t0) * 1e3, 2)})
        except Exception as e:
            res.notes.append(
                f"a paired peer did not answer within 3 s: "
                f"{type(e).__name__}")
    if not reachable:
        res.status = "absent"
        res.absent_reason = (
            "every paired peer is unreachable from this process. A dev "
            "container has no route to the fast line; run the cross-rig arm "
            "from the host (runbook §8 host-runner pattern) -- needs host "
            "runner.")
        return res
    res.status = "warn"
    res.facts = {"peers_reachable": len(reachable),
                 "tcp_connect_ms": [r["rtt_ms"] for r in reachable]}
    res.notes.append(
        "TCP connect time only. A collective figure over this link needs the "
        "same harness started on BOTH sides (runbook §8 host-runner "
        "pattern); the suite does not start remote processes on its own.")
    return res


def _gdr_crossover_bin() -> Optional[str]:
    """The handover binary, if this box has one built. Env var wins."""
    path = os.environ.get(GDR_CROSSOVER_BIN_ENV) or _GDR_CROSSOVER_DEFAULT_BIN
    return path if os.path.isfile(path) and os.access(path, os.X_OK) else None


def _gdr_bench_run(binary: str, sizes: Sequence[str], budget_s: float) -> dict:
    """Shell out to the handover's ping-pong across the size ladder.

    Kept as one seam so the result-shaping in :func:`_arm_gdr_crossover` is
    testable without the binary -- every existing GPU arm mocks its own
    worker call (``_worker_arm``, ``card_probe._run_probe_subprocess``)
    rather than a raw subprocess, and this follows the same shape. Expected
    JSON shape: ``{"pair": ..., "sizes": {"<size>": {"direct_us": ...,
    "staged_us": ..., "n": ..., "spread_pct": ...}, ...}}``.
    """
    out = _tmp_out("gdr_crossover")
    argv = [binary, "--sizes", ",".join(sizes), "--out", out]
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=budget_s)
        if not os.path.exists(out):
            raise RuntimeError(
                f"{binary} produced no result file (rc={proc.returncode}). "
                f"Output: {((proc.stdout or '') + (proc.stderr or ''))[-800:]}")
        with open(out) as f:
            return json.load(f)
    finally:
        try:
            os.unlink(out)
        except Exception:
            pass


def _arm_gdr_crossover(ctx: "_RunCtx") -> ArmResult:
    """Direct-vs-staged crossover, as a fact about THIS rig, not about GDR.

    docs/EVAL_gdr_uebernahme.md is explicit that "GDR is bad at 1 MiB" is the
    wrong shape of claim (§1.4, the rig-is-a-lower-bound rule): the
    handover's own tables never targeted this rig's Resizable-BAR card as an
    RDMA destination, so the crossover point is a fact about a RIG, and this
    arm's whole job is to produce that fact for whichever rig runs it --
    without assuming in advance which side of the crossover it lands on.

    On a box with no card window at all (this desk session; the GPU belongs
    to another agent's exclusive slice), the driver already turns every
    ``kind="gpu"`` arm absent before this function is even called (see
    ``_gpu_phase``). What this function itself is responsible for is the
    other half: even WITH a window, the handover binary is a separate,
    out-of-tree build (never vendored, §7) that most boxes simply do not
    have -- and that absence must be named just as specifically.
    """
    res = ArmResult(arm_id="gdr_crossover")
    binary = _gdr_crossover_bin()
    if binary is None:
        looked_at = os.environ.get(GDR_CROSSOVER_BIN_ENV) or _GDR_CROSSOVER_DEFAULT_BIN
        res.status = "absent"
        res.absent_reason = (
            "the handover's gpurdma_04_bench is not built on this box "
            f"(looked at {looked_at}, override with ${GDR_CROSSOVER_BIN_ENV}; "
            "build per the handover's BUILD.md -- headers are fetched at "
            "build time, never vendored, docs/EVAL_gdr_uebernahme.md §7). "
            "This arm also needs an exclusive GPU card window (the #271 "
            "lock it is gated behind); declared here as a startable job, "
            "not run -- run it from a session that actually holds a card.")
        return res
    try:
        data = _gdr_bench_run(
            binary, GDR_CROSSOVER_SIZES, ARM_BY_ID["gdr_crossover"].budget_s)
    except Exception as e:
        res.status = "error"
        res.error = f"{type(e).__name__}: {e}"
        return res
    cells: Dict[str, dict] = {}
    crossover: Optional[str] = None
    for size in GDR_CROSSOVER_SIZES:
        row = (data.get("sizes") or {}).get(size)
        if not isinstance(row, dict):
            continue
        direct_us, staged_us = row.get("direct_us"), row.get("staged_us")
        if direct_us is None or staged_us is None:
            continue
        cells[f"direct/{size}"] = {"median_us": direct_us, "n": row.get("n"),
                                   "spread_pct": row.get("spread_pct")}
        cells[f"staged/{size}"] = {"median_us": staged_us, "n": row.get("n"),
                                   "spread_pct": row.get("spread_pct")}
        if crossover is None and direct_us > staged_us:
            crossover = size
    res.cells = cells
    res.facts = {
        "sizes": list(GDR_CROSSOVER_SIZES),
        "crossover_at": crossover,
        "pair": data.get("pair"),
    }
    res.status = "ok"
    res.notes.append(
        "crossover reported as a property of THIS rig "
        "(docs/EVAL_gdr_uebernahme.md §1.4, rig-is-a-lower-bound rule) -- do "
        "not carry it to another rig's Resizable-BAR card or NIC generation "
        "without re-measuring there.")
    res.notes.extend(_spread_note(cells, ctx.floor_pct))
    return res


ARM_RUNNERS: Dict[str, Callable[["_RunCtx"], ArmResult]] = {
    "rig_profile": _arm_rig_profile,
    "noise_floor": _arm_noise_floor,
    "collective_gloo": _arm_collective_gloo,
    "collective_barlink_ucx": _arm_collective_barlink_ucx,
    "byte_gate": _arm_byte_gate,
    "card_probe": _arm_card_probe,
    "collective_nccl": _arm_collective_nccl,
    "collective_barlink_shm": _arm_collective_barlink_shm,
    "gdr_crossover": _arm_gdr_crossover,
    "cross_rig": _arm_cross_rig,
}


# ===========================================================================
# The run
# ===========================================================================
@dataclass
class _RunCtx:
    job: "CommSuiteJob"
    floor_pct: Optional[float] = None
    card_count: int = 0

    def register(self, proc: subprocess.Popen) -> None:
        self.job._register(proc)


@dataclass
class CommSuiteJob:
    """One suite run, observed from outside the threads doing the work."""

    job_id: str
    state: str = PENDING
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    selected: List[str] = field(default_factory=list)
    current: Optional[str] = None
    results: Dict[str, ArmResult] = field(default_factory=dict)
    rig: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    _procs: List[subprocess.Popen] = field(default_factory=list, repr=False)
    _cancelled: set = field(default_factory=set, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- cancellation -------------------------------------------------------
    def _register(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._procs.append(proc)

    def cancel(self, arm_id: Optional[str] = None) -> List[str]:
        """Cancel one arm, or the whole run when ``arm_id`` is None.

        Cancelling kills the arm's live processes: a collective that is
        already hanging will not notice a polite flag, and the point of a
        per-arm cancel is that one stuck arm cannot cost the other eight.
        """
        with self._lock:
            if arm_id:
                self._cancelled.add(arm_id)
                targets = [arm_id]
            else:
                self._cancelled.update(a.id for a in ARMS)
                targets = [a.id for a in ARMS]
            live = [p for p in self._procs if p.poll() is None] \
                if (arm_id is None or arm_id == self.current) else []
        for p in live:
            try:
                os.killpg(os.getpgid(p.pid), 9)
            except Exception:
                pass
        return targets

    def is_cancelled(self, arm_id: str) -> bool:
        with self._lock:
            return arm_id in self._cancelled

    # -- reporting ----------------------------------------------------------
    def to_json(self, include_artifact: bool = False) -> dict:
        with self._lock:
            done = [r.to_json() for r in self.results.values()]
            state = self.state
            current = self.current
        out = {
            "job_id": self.job_id,
            "state": state,
            "current": current,
            "selected": list(self.selected),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": (
                round((self.finished_at or time.time()) - self.started_at, 1)
                if self.started_at else None),
            "progress": {
                "done": len(done),
                "total": len(self.selected) or len(ARMS),
            },
            "arms": done,
            "rig": self.rig,
            "error": self.error,
        }
        if include_artifact and state in (OK, ERROR, CANCELLED):
            try:
                out["digest"] = rig_artifact.build_digest([to_sections(self)])
            except Exception as e:
                out["digest"] = None
                out["digest_error"] = f"{type(e).__name__}: {e}"
        return out


class CommSuiteJobStore:
    """Starts suite runs and answers "is it done yet".

    Single-flight on purpose: two suites at once would measure each other,
    and the second one's numbers would be a report about the first.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, CommSuiteJob] = {}
        #: Overridable for tests: run inline instead of in a thread.
        self.synchronous = False
        #: Overridable for tests: what actually performs one arm.
        self.runners: Dict[str, Callable[[_RunCtx], ArmResult]] = dict(ARM_RUNNERS)

    def jobs(self) -> List[CommSuiteJob]:
        with self._lock:
            return list(self._jobs.values())

    def get(self, job_id: str) -> Optional[CommSuiteJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> Optional[CommSuiteJob]:
        with self._lock:
            for j in self._jobs.values():
                if j.state in (PENDING, RUNNING):
                    return j
        return None

    def latest(self) -> Optional[CommSuiteJob]:
        with self._lock:
            js = sorted(self._jobs.values(),
                        key=lambda j: j.started_at or 0.0)
            return js[-1] if js else None

    def start(self, arms: Optional[Sequence[str]] = None) -> CommSuiteJob:
        running = self.active()
        if running is not None:
            return running
        selected = [a for a in (arms or [s.id for s in ARMS])
                    if a in ARM_BY_ID]
        if "rig_profile" not in selected:
            selected.insert(0, "rig_profile")
        # noise_floor before everything it is the floor FOR.
        selected.sort(key=lambda a: [s.id for s in ARMS].index(a))
        job = CommSuiteJob(job_id=_uuid.uuid4().hex[:12], state=RUNNING,
                           started_at=time.time(), selected=selected)
        with self._lock:
            self._expire_locked(time.time())
            self._jobs[job.job_id] = job
        if self.synchronous:
            self._run(job)
        else:
            threading.Thread(target=self._run, args=(job,),
                             daemon=True).start()
        return job

    def _expire_locked(self, now: float) -> None:
        for jid, j in list(self._jobs.items()):
            if j.finished_at and now - j.finished_at > JOB_TTL_S:
                del self._jobs[jid]

    def _run(self, job: CommSuiteJob) -> None:
        ctx = _RunCtx(job=job)
        try:
            cpu_arms = [a for a in job.selected
                        if ARM_BY_ID[a].kind in ("inventory", "cpu")]
            gpu_arms = [a for a in job.selected if ARM_BY_ID[a].kind == "gpu"]
            net_arms = [a for a in job.selected
                        if ARM_BY_ID[a].kind == "network"]

            # CPU first, always: they need nothing from anybody, so a busy
            # rig still yields a usable artifact.
            for arm_id in cpu_arms:
                self._one(job, ctx, arm_id)
            job.rig = job.results.get(
                "rig_profile", ArmResult("rig_profile")).facts
            ctx.card_count = int(job.rig.get("card_count") or 0)

            if gpu_arms:
                self._gpu_phase(job, ctx, gpu_arms)
            for arm_id in net_arms:
                self._one(job, ctx, arm_id)

            with job._lock:
                job.state = CANCELLED if len(job._cancelled) >= len(
                    job.selected) else OK
                job.finished_at = time.time()
                job.current = None
        except Exception as e:  # pragma: no cover - defensive
            with job._lock:
                job.state = ERROR
                job.error = f"{type(e).__name__}: {e}"
                job.finished_at = time.time()
                job.current = None

    def _gpu_phase(self, job: CommSuiteJob, ctx: _RunCtx,
                   gpu_arms: List[str]) -> None:
        indices = [c["index"] for c in job.rig.get("cards", [])
                   if c.get("index") is not None]
        if not indices:
            for arm_id in gpu_arms:
                self._absent(job, arm_id,
                             "no CUDA card is visible to this process")
            return
        window = _CardWindow(indices)
        if not window.acquire():
            for arm_id in gpu_arms:
                self._absent(job, arm_id, window.reason or
                             "the card window could not be taken")
            return
        try:
            for arm_id in gpu_arms:
                self._one(job, ctx, arm_id)
        finally:
            window.release()

    def _absent(self, job: CommSuiteJob, arm_id: str, reason: str) -> None:
        with job._lock:
            job.results[arm_id] = ArmResult(
                arm_id=arm_id, status="absent", absent_reason=reason)

    def _one(self, job: CommSuiteJob, ctx: _RunCtx, arm_id: str) -> None:
        if job.is_cancelled(arm_id):
            self._absent(job, arm_id, "cancelled before it started")
            return
        with job._lock:
            job.current = arm_id
            job._procs = [p for p in job._procs if p.poll() is None]
        t0 = time.time()
        try:
            result = self.runners[arm_id](ctx)
        except Exception as e:
            # A failure IS the finding. It is recorded with its text, its
            # type and the time it took, and the suite goes on to the next
            # arm -- an artifact of only the arms that happened to work is a
            # survivorship sample.
            result = ArmResult(
                arm_id=arm_id, status="error",
                error=f"{type(e).__name__}: {e}")
        if job.is_cancelled(arm_id) and result.status == "error":
            result = ArmResult(arm_id=arm_id, status="absent",
                               absent_reason="cancelled while it ran")
        result.elapsed_s = round(time.time() - t0, 2)
        with job._lock:
            job.results[arm_id] = result


#: Module-level store, mirroring ``card_probe.JOBS`` / ``pairing.STORE``.
JOBS = CommSuiteJobStore()




# ===========================================================================
# Source adapter: suite run -> the shared artifact schema
# ===========================================================================
def _cell_context(arm_id: str, cell_name: str, facts: dict) -> Dict[str, Any]:
    """What a cell needs beside it to be comparable with anyone else's.

    Without ``op`` / ``size_kib`` / ``world`` a latency in microseconds is
    not a measurement, it is a number. The context travels with the row into
    the shared digest and survives every aggregation rung but the last.
    """
    op, _, size = cell_name.partition("/")
    ctx: Dict[str, Any] = {"op": op}
    m = re.match(r"(\d+)KiB", size or "")
    if m:
        ctx["size_kib"] = int(m.group(1))
    if facts.get("world") is not None:
        ctx["world"] = facts["world"]
    ctx["backend"] = arm_id.replace("collective_", "")
    return ctx


def to_sections(job: "CommSuiteJob") -> rig_artifact.SourceSections:
    """One finished suite run, as rows the shared artifact understands.

    Only aggregates leave: median, spread, n and the two tail quantiles per
    cell. The per-iteration samples never existed outside the worker, and the
    subprocess output is not carried at all -- an error becomes a SIGNATURE,
    not a log.
    """
    with job._lock:
        results = [job.results[a] for a in job.selected if a in job.results]
        rig = dict(job.rig)
    taken_at = time.strftime(
        "%Y-%m-%d", time.gmtime(job.finished_at or time.time()))

    measurements: List[rig_artifact.Measurement] = []
    capabilities: List[rig_artifact.Capability] = []
    errors: List[rig_artifact.ErrorSignature] = []
    notes: List[str] = []

    floor = None
    for res in results:
        if res.arm_id == "noise_floor":
            floor = res.facts.get("floor_pct")

    for res in results:
        spec = ARM_BY_ID.get(res.arm_id)
        label = spec.label if spec else res.arm_id
        # The capability table: what this rig CAN do, one line per arm, in
        # the dashboard's own provenance vocabulary. An arm that failed is
        # `absent` as a capability and an error SIGNATURE below -- the two
        # say different things and both are kept.
        capabilities.append(rig_artifact.Capability(
            name=f"comm/{res.arm_id}",
            value=res.status,
            provenance=("measured" if res.status in ("ok", "warn")
                        else "absent"),
            note=(res.absent_reason or (res.notes[0] if res.notes else ""))[:200],
        ))
        if res.error:
            errors.append(rig_artifact.error_signature(
                res.error, where=f"comm_suite/{res.arm_id}"))
        for n in res.notes:
            if res.status == "warn":
                notes.append(f"{label}: {n}")
        for cell_name, cell in (res.cells or {}).items():
            if not isinstance(cell, dict):
                continue
            measurements.append(rig_artifact.Measurement(
                id=f"comm/{res.arm_id}/{cell_name}",
                label=f"{label} {cell_name}",
                source=SOURCE_NAME,
                unit="us (median)",
                value=cell.get("median_us"),
                spread_pct=cell.get("spread_pct"),
                n=cell.get("n"),
                p5=cell.get("p5_us"),
                p95=cell.get("p95_us"),
                taken_at=taken_at,
                status=res.status,
                context=_cell_context(res.arm_id, cell_name, res.facts or {}),
                note=(f"noise floor {floor} %" if floor is not None else ""),
            ))
            if cell.get("gbit_s") is not None:
                measurements.append(rig_artifact.Measurement(
                    id=f"comm/{res.arm_id}/{cell_name}/rate",
                    label=f"{label} {cell_name} rate",
                    source=SOURCE_NAME,
                    unit="Gbit/s",
                    value=cell.get("gbit_s"),
                    spread_pct=cell.get("spread_pct"),
                    n=cell.get("n"),
                    taken_at=taken_at,
                    status=res.status,
                    context=_cell_context(res.arm_id, cell_name,
                                          res.facts or {}),
                ))
        # PER-CARD rates are deliberately NOT emitted here. The probe writes
        # them to the same cache the hardware-profile source reads, and that
        # source already renders them with ordinals, the fp8 capability line
        # and the throttle reservation. Emitting them twice would produce two
        # ids for one number and make the dedupe decide which is canonical.
        #
        # The pair matrix is keyed by GPU UUID on disk; UUIDs are identity and
        # never leave, so pairs are re-keyed to model#ordinal by the shared
        # helper the profile source uses -- one definition, both sources.
        if res.arm_id == "card_probe":
            from sglang.srt.planner import rig_profile_source

            keys = rig_profile_source._card_key_map(
                res.facts.get("cards") or [])
            for pair in res.facts.get("pairs") or []:
                if not isinstance(pair, dict) \
                        or pair.get("bandwidth_gbs") is None:
                    continue
                src = keys.get(str(pair.get("src_uuid")), "?")
                dst = keys.get(str(pair.get("dst_uuid")), "?")
                measurements.append(rig_artifact.Measurement(
                    id=f"pair/{src}->{dst}/bandwidth",
                    label=f"{src} -> {dst}",
                    source=SOURCE_NAME,
                    unit="GB/s",
                    value=pair.get("bandwidth_gbs"),
                    taken_at=taken_at,
                    status="ok",
                    context={"transport": pair.get("transport"),
                             "direction": "ordered",
                             "pair": f"{src}->{dst}"},
                ))

    if floor is not None:
        notes.append(
            f"Noise floor of this run: {floor} %. Nothing below that is a "
            f"difference.")
    return rig_artifact.SourceSections(
        source=SOURCE_NAME, rig=rig, measurements=measurements,
        capabilities=capabilities, errors=errors, notes=notes)
