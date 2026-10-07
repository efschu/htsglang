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
"""``flliper.hardware/1``: ONE hardware profile, as a VIEW over what is measured.

Order 950 (profile editor S2).  The rig already has three places that hold
measurements -- the card-probe cache (``rigmon.card_probe``: rates, host
transfers, the ordered pair matrix), the stage-0 profile
``hw_profile-<digest>.json`` (``uneven_perf``: GEMM lanes, memory rates, the
NCCL link table) and NVML (identity, memory, BAR1, PCIe).  This module is NOT a
fourth measurement file.  It reads those sources, keeps the newest value per
card and per ordered pair, and writes nothing: the document is assembled on
every call.  New measurements land where they always did (the card-probe
cache), through the arms of ``card_probe``.

**Every number says where it comes from.**  A numeric value is a node

    {"v": 62.7, "src": "gemessen", "at": 1790000000.0, "probe": "card_probe-ab12.json"}

with ``src`` one of ``gemessen`` (a kernel ran, ``at``/``probe`` name when and
in which file), ``NVML`` (read from the driver, no GPU work), ``Datenblatt``
(a nameplate figure computed from NVML facts), ``geschätzt`` (derived from
another measurement, never a measurement itself: a mirrored pair direction)
or ``nicht gemessen`` (``v`` is ``null`` and ``note`` says WHY).  A value
without a measurement is never shown as measured, and a lane that cannot run on
a card (no fp8 on sm_86) is "nicht gemessen" with the card's own reason -- not a
substitute number (HOCHRECHNUNG != MESSUNG).

**Persisted at the first start (AP-A, plan 06.10.).**  The view is still assembled on every call, but the
first call on a machine also WRITES it (``capture``): ``/var/lib/flliper/hardware.json`` (env
``FLLIPER_HARDWARE_PROFILE``), in the rig and in the release edition alike.  The file is the machine's identity
(NVML names, sizes, cc, SM, clocks, the measured rates known at that moment); a later call compares it with the live
cards and reports a difference instead of overwriting it, ``capture(force=True)`` ("Neu erfassen") replaces it.  Where
NVML says nothing (editor-only container without a GPU) the persisted file is the profile.  A card's SM count and
nominal bandwidth are merged in from the data sheets (``datasheet``: ``weg2/hw_sim.py`` for SM, the dashboard's card
catalog for the bandwidth), always labelled "Datenblatt"; a measured SM count wins over the data sheet.

**No rig constants.**  Card classes, order and count come from NVML and from
``weg2.card_identity`` (loaded by path, so this file also runs inside the
stdlib-only dashboard process); the file names the cards of the machine it runs
on and no other.

STDLIB ONLY at module level: ``pynvml``/``nvidia-smi`` are read lazily, ``torch``
is never imported.  The measurement itself runs in a child interpreter
(``run_measurement``) because it allocates a CUDA context on every card.

CLI::

    python hardware_profile.py                    # assemble and print the profile (no GPU work)
    python hardware_profile.py --measure --cards 0,1,2 [--python PY] [--timeout-s 540]
                                                  # run the card probe on exactly these NVML cards,
                                                  # then print the profile and the duration line
"""

from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "SCHEMA",
    "SRC_MEASURED",
    "SRC_NVML",
    "SRC_DATASHEET",
    "SRC_ESTIMATED",
    "SRC_NONE",
    "SOURCES",
    "BAR1_NOT_MEASURED",
    "COMPUTE_FORMATS",
    "build",
    "validate",
    "read_nvml",
    "run_measurement",
    "duration_line",
    "PERSIST_ENV",
    "DEFAULT_PERSIST_PATH",
    "persist_path",
    "load_profile",
    "save_profile",
    "compare",
    "capture",
    "hw_sim_datasheet",
]

SCHEMA = "flliper.hardware/1"

SRC_MEASURED = "gemessen"
SRC_NVML = "NVML"
SRC_DATASHEET = "Datenblatt"
SRC_ESTIMATED = "geschätzt"
SRC_NONE = "nicht gemessen"
SOURCES = (SRC_MEASURED, SRC_NVML, SRC_DATASHEET, SRC_ESTIMATED, SRC_NONE)

#: Where the profile is persisted (first start); the env names another file (container volume, test).
PERSIST_ENV = "FLLIPER_HARDWARE_PROFILE"
DEFAULT_PERSIST_PATH = "/var/lib/flliper/hardware.json"

#: Same horizon as ``card_probe.DEFAULT_MAX_AGE_S`` (one convention, not two).
MAX_AGE_S = 7 * 24 * 3600.0

#: The card-probe cache version this view understands (``card_probe.CARD_PROBE_VERSION``).
_PROBE_VERSION = 1

#: Why the BAR1 stretch is not in the pair matrix.  ``card_probe`` quotes the same text.
BAR1_NOT_MEASURED = (
    "BAR1 stretch per pair: NOT MEASURED. The BAR1 transport "
    "(barlink_bar1.build_bar1) needs a torch.distributed CPU group with one "
    "process per rank, the dmabuf_holder kernel module and a window the "
    "serving boot negotiates; there is no single-process entry that runs it "
    "without a server. The pair matrix is the cuda p2p / host staging "
    "path and is labelled as such."
)

#: The compute formats the card probe itself measures (the others come from the stage-0 profile only).
PROBE_FORMATS = ("bf16", "fp8_native", "int8", "nvfp4_w4a8", "nvfp4_marlin")

BAR1_SHORT ="BAR1-Strecke nicht gemessen: kein Einzelprozess-Pfad ohne Server (Begründung: bar1.note)"

#: The compute formats of the view, in display order: key, unit, label.  A format
#: a card cannot run keeps its row, as "nicht gemessen" with the reason.
COMPUTE_FORMATS: Tuple[Tuple[str, str, str], ...] = (
    ("bf16", "TFLOPS", "bf16"),
    ("fp8_native", "TFLOPS", "fp8 (native, _scaled_mm)"),
    ("fp8_marlin", "TFLOPS", "fp8 Marlin (nur Gewicht)"),
    ("fp8_w8a16", "TFLOPS", "fp8 W8A16 (Dequant)"),
    ("int8", "TOPS", "int8 W8A8"),
    ("nvfp4_w4a8", "TOPS", "NVFP4 W4A8 (int8-Kerne)"),
    ("nvfp4_marlin", "TFLOPS", "NVFP4 W4A16 (Marlin)"),
)


# ---------------------------------------------------------------------------
# value nodes
# ---------------------------------------------------------------------------


def node(
    v: Optional[float],
    src: str,
    *,
    at: Optional[float] = None,
    probe: Optional[str] = None,
    note: Optional[str] = None,
    unit: Optional[str] = None,
) -> Dict[str, Any]:
    """One value with its provenance.  ``v is None`` is only legal as "nicht gemessen"."""
    if v is None:
        src = SRC_NONE
    n: Dict[str, Any] = {"v": v, "src": src}
    if at is not None:
        n["at"] = at
    if probe:
        n["probe"] = probe
    if unit:
        n["unit"] = unit
    if note:
        n["note"] = note
    return n


def missing(note: str, unit: Optional[str] = None) -> Dict[str, Any]:
    return node(None, SRC_NONE, note=note, unit=unit)


def _num(x) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


# ---------------------------------------------------------------------------
# the identity module (card_identity, by path: no `import sglang`)
# ---------------------------------------------------------------------------


def _load_identity():
    """``weg2/card_identity.py`` of the tree this file sits in, or ``None``."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weg2", "card_identity.py")
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        return None
    name = "hwprofile_card_identity"
    if name in sys.modules:
        return sys.modules[name]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod  # @dataclass resolves types through sys.modules[__module__]
        spec.loader.exec_module(mod)
        return mod
    except Exception:  # pragma: no cover - a broken sibling must not break the view
        sys.modules.pop(name, None)
        return None


def _load_hw_sim():
    """``weg2/hw_sim.py`` of the tree this file sits in (stdlib at module level), or ``None``."""
    path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weg2", "hw_sim.py"))
    if not os.path.isfile(path):
        return None
    name = "hwprofile_hw_sim"
    if name in sys.modules:
        return sys.modules[name]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod  # @dataclass resolves types through sys.modules[__module__]
        spec.loader.exec_module(mod)
        return mod
    except Exception:  # pragma: no cover - a broken sibling must not break the view
        sys.modules.pop(name, None)
        return None


def hw_sim_datasheet(row: dict) -> Dict[str, Any]:
    """The data-sheet SM count of one NVML card row from ``weg2/hw_sim.py`` ``CATALOG`` (``SimCard.sm_count``).

    The match is the NVML name and the compute capability; among several entries of that name the one with the same
    total MiB wins, and when the total matches none the answer is given only if every candidate agrees (the 10 GB and
    the 20 GB RTX 3080 are both 68 SM).  No match, or candidates that disagree: ``{}`` -- never a guess."""
    sim = _load_hw_sim()
    cat = getattr(sim, "CATALOG", None) if sim is not None else None
    if not cat:
        return {}
    cc = tuple(row.get("cc") or ())
    cands = [c for c in cat.values() if c.name == row.get("name") and tuple(c.cc) == cc]
    exact = [c for c in cands if c.total_mib == row.get("total_mib")]
    pick = exact or cands
    counts = {c.sm_count for c in pick}
    if len(counts) != 1:
        return {}
    return {"sm_count": counts.pop(),
            "sm_note": "Datenblatt-Katalog weg2/hw_sim.py CATALOG[%s].sm_count (nicht gemessen)" % ",".join(c.key for c in pick)}


# ---------------------------------------------------------------------------
# NVML (no CUDA context; pynvml first, nvidia-smi as the fallback)
# ---------------------------------------------------------------------------


def _try(fn, *a):
    try:
        return fn(*a)
    except Exception:
        return None


def _dec(x):
    return x.decode() if isinstance(x, bytes) else x


def _read_pynvml() -> Tuple[List[dict], Optional[str]]:
    import pynvml

    pynvml.nvmlInit()
    try:
        driver = _dec(_try(pynvml.nvmlSystemGetDriverVersion))
        cards = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = _try(pynvml.nvmlDeviceGetMemoryInfo, h)
            cc = _try(pynvml.nvmlDeviceGetCudaComputeCapability, h)
            bar1 = _try(pynvml.nvmlDeviceGetBAR1MemoryInfo, h)
            pl = _try(pynvml.nvmlDeviceGetPowerManagementLimit, h)
            pd = _try(pynvml.nvmlDeviceGetPowerManagementDefaultLimit, h)
            cards.append(
                {
                    "nvml_index": i,
                    "uuid": _dec(pynvml.nvmlDeviceGetUUID(h)),
                    "name": _dec(pynvml.nvmlDeviceGetName(h)),
                    "total_mib": int(mem.total // (1024 * 1024)) if mem else 0,
                    "cc": [int(cc[0]), int(cc[1])] if cc else None,
                    "bar1_total_mib": int(bar1.bar1Total // (1024 * 1024)) if bar1 else None,
                    "pcie_max_gen": _try(pynvml.nvmlDeviceGetMaxPcieLinkGeneration, h),
                    "pcie_max_width": _try(pynvml.nvmlDeviceGetMaxPcieLinkWidth, h),
                    "pcie_cur_gen": _try(pynvml.nvmlDeviceGetCurrPcieLinkGeneration, h),
                    "pcie_cur_width": _try(pynvml.nvmlDeviceGetCurrPcieLinkWidth, h),
                    "mem_bus_width_bits": _try(pynvml.nvmlDeviceGetMemoryBusWidth, h),
                    "mem_clock_max_mhz": _try(pynvml.nvmlDeviceGetMaxClockInfo, h, pynvml.NVML_CLOCK_MEM),
                    "sm_clock_max_mhz": _try(pynvml.nvmlDeviceGetMaxClockInfo, h, pynvml.NVML_CLOCK_SM),
                    "power_limit_w": round(pl / 1000.0, 1) if pl else None,
                    "power_default_w": round(pd / 1000.0, 1) if pd else None,
                    "pci_bus_id": _dec(getattr(_try(pynvml.nvmlDeviceGetPciInfo, h), "busId", None)),
                }
            )
        return cards, driver
    finally:
        _try(pynvml.nvmlShutdown)


_SMI_FIELDS = (
    "index,uuid,name,memory.total,compute_cap,pcie.link.gen.max,pcie.link.width.max,"
    "pcie.link.gen.gpucurrent,pcie.link.width.current,power.limit,power.default_limit,"
    "clocks.max.sm,clocks.max.mem,driver_version,pci.bus_id"
)


def _read_smi() -> Tuple[List[dict], Optional[str]]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=" + _SMI_FIELDS, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    ).stdout
    cards, driver = [], None

    def f(x):
        x = x.strip()
        try:
            return float(x)
        except ValueError:
            return None

    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 15:
            continue
        cc = p[4].split(".")
        driver = p[13]
        cards.append(
            {
                "nvml_index": int(p[0]),
                "uuid": p[1],
                "name": p[2],
                "total_mib": int(f(p[3]) or 0),
                "cc": [int(cc[0]), int(cc[1])] if len(cc) == 2 and all(c.isdigit() for c in cc) else None,
                # nvidia-smi --query-gpu has no BAR1 total and no bus width
                "bar1_total_mib": None,
                "pcie_max_gen": int(f(p[5])) if f(p[5]) is not None else None,
                "pcie_max_width": int(f(p[6])) if f(p[6]) is not None else None,
                "pcie_cur_gen": int(f(p[7])) if f(p[7]) is not None else None,
                "pcie_cur_width": int(f(p[8])) if f(p[8]) is not None else None,
                "mem_bus_width_bits": None,
                "mem_clock_max_mhz": int(f(p[12])) if f(p[12]) is not None else None,
                "sm_clock_max_mhz": int(f(p[11])) if f(p[11]) is not None else None,
                "power_limit_w": f(p[9]),
                "power_default_w": f(p[10]),
                "pci_bus_id": p[14],
            }
        )
    return cards, driver


def read_nvml() -> Tuple[List[dict], Optional[str], List[str]]:
    """``(cards, driver, issues)``.  Never raises: an unreadable NVML is an issue string."""
    issues: List[str] = []
    try:
        cards, driver = _read_pynvml()
        if cards:
            return cards, driver, issues
        issues.append("pynvml: keine Karte gemeldet")
    except Exception as e:
        issues.append(f"pynvml nicht lesbar ({type(e).__name__}: {e})")
    try:
        cards, driver = _read_smi()
        if cards:
            issues.append("Fallback nvidia-smi: BAR1-Größe und Speicher-Busbreite fehlen")
            return cards, driver, issues
        issues.append("nvidia-smi: keine Karte gemeldet")
    except Exception as e:
        issues.append(f"nvidia-smi nicht lesbar ({type(e).__name__}: {e})")
    return [], None, issues


# ---------------------------------------------------------------------------
# reading the measurement caches
# ---------------------------------------------------------------------------


def default_cache_dir() -> str:
    return os.path.expanduser("~/.cache/sglang")


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def load_probes(cache_dir: str) -> List[dict]:
    """Every readable ``card_probe-*.json`` of the current version, oldest first.

    Each is returned as ``{"file", "created", "driver", "data"}``.  A file of
    another probe version is ignored, not reinterpreted (the card-probe rule)."""
    out = []
    for path in glob.glob(os.path.join(cache_dir, "card_probe-*.json")):
        d = _read_json(path)
        if not d or not d.get("cards") or int(d.get("version", -1)) != _PROBE_VERSION:
            continue
        created = _num(d.get("created")) or _mtime(path)
        out.append({"file": os.path.basename(path), "created": created, "driver": d.get("driver"), "data": d})
    out.sort(key=lambda p: p["created"])
    return out


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _parse_local_time(s: Any, fallback: float) -> float:
    try:
        return time.mktime(time.strptime(str(s), "%Y-%m-%d %H:%M:%S"))
    except (ValueError, OverflowError):
        return fallback


def load_stage0(cache_dir: str) -> List[dict]:
    """Every readable stage-0 ``hw_profile-*.json`` (``uneven_perf``), oldest first."""
    out = []
    for path in glob.glob(os.path.join(cache_dir, "hw_profile-*.json")):
        d = _read_json(path)
        if not d or not isinstance(d.get("gpus"), dict):
            continue
        out.append(
            {
                "file": os.path.basename(path),
                "created": _parse_local_time(d.get("created"), _mtime(path)),
                "driver": d.get("driver"),
                "data": d,
            }
        )
    out.sort(key=lambda p: p["created"])
    return out


# ---------------------------------------------------------------------------
# assembling
# ---------------------------------------------------------------------------


def _fallback_key(c: dict) -> str:
    name = " ".join(w for w in str(c.get("name", "")).split() if w.upper() not in ("NVIDIA", "GEFORCE"))
    cc = c.get("cc")
    return f"{name.replace(' ', '')}/{c.get('total_mib', 0)}MiB/sm{cc[0]}{cc[1]}" if cc else f"{name.replace(' ', '')}/{c.get('total_mib', 0)}MiB/sm?"


class _Pick:
    """The newest candidate wins; on a tie the later offer wins (probes are offered after stage 0)."""

    def __init__(self):
        self.best: Optional[Tuple[float, str, Any, str]] = None

    def offer(self, at: float, probe: str, value, note: str = ""):
        # a quantity a source did not measure never hides an older measured one
        if value is None:
            return
        if self.best is None or at >= self.best[0]:
            self.best = (at, probe, value, note)


def _nv(v, unit: Optional[str], what: str, note: Optional[str] = None) -> Dict[str, Any]:
    """An NVML value, or "nicht gemessen" naming what NVML did not report."""
    if v is None:
        return missing(f"NVML meldet {what} nicht", unit)
    return node(v, SRC_NVML, unit=unit, note=note)


def _m(v, at: float, probe: str, unit: Optional[str], why: str) -> Dict[str, Any]:
    if v is None:
        return missing(why, unit)
    return node(v, SRC_MEASURED, at=at, probe=probe, unit=unit)


def _lane_node(unit: str, pick: _Pick, notes: Dict[str, str], key: str) -> Dict[str, Any]:
    if pick.best is not None and pick.best[2] is not None:
        at, probe, v, _ = pick.best
        return node(v, SRC_MEASURED, at=at, probe=probe, unit=unit)
    return missing(notes.get(key) or "Messarm noch nicht gelaufen (Hardwareprofil messen)", unit)


def build(
    *,
    cache_dir: Optional[str] = None,
    nvml: Optional[Tuple[List[dict], Optional[str], List[str]]] = None,
    now: Optional[float] = None,
    identity: Any = "auto",
    datasheet: Any = "auto",
) -> Dict[str, Any]:
    """Assemble the ``flliper.hardware/1`` document.  Reads only; starts nothing.

    ``datasheet(row) -> dict`` is the data-sheet lookup of one NVML card row (keys ``sm_count`` + ``sm_note``,
    ``mem_bw_gbs`` + ``bw_note``, ``catalog``); ``"auto"`` = ``hw_sim_datasheet`` (SM count only), ``None`` = off (the
    document then has none of the data-sheet fields)."""
    now = time.time() if now is None else now
    ds_fn = hw_sim_datasheet if datasheet == "auto" else datasheet
    cache_dir = cache_dir or default_cache_dir()
    cards_nvml, driver, issues = nvml if nvml is not None else read_nvml()
    ci = _load_identity() if identity == "auto" else identity
    probes = load_probes(cache_dir)
    stage0 = load_stage0(cache_dir)

    # ---- order and class keys (planner order when the identity module is there)
    rows = [dict(c) for c in cards_nvml]
    for r in rows:
        r["cc_t"] = tuple(r["cc"]) if r.get("cc") else None
    keyed: List[Tuple[dict, str, str]] = []
    if ci is not None and rows:
        try:
            objs = [dict(r, cc=r["cc_t"]) for r in rows]
            ordered = ci.order_cards(objs, gate=False)
            by_uuid = {r["uuid"]: r for r in rows}
            for o in ordered:
                keyed.append((by_uuid[o["uuid"]], ci.class_label(o), ci.card_key(o)))
        except Exception as e:  # pragma: no cover - identity trouble must not hide the cards
            issues.append(f"card_identity unbrauchbar ({type(e).__name__}: {e}); NVML-Reihenfolge")
            keyed = []
    if not keyed:
        keyed = [(r, _fallback_key(r), _fallback_key(r)) for r in sorted(rows, key=lambda r: r["nvml_index"])]

    uuids = [r["uuid"] for r, _, _ in keyed]
    ord_of = {u: i for i, u in enumerate(uuids)}
    cards_out: List[Dict[str, Any]] = []
    unmeasured: Dict[str, List[str]] = {}

    for ordinal, (r, class_label, key) in enumerate(keyed):
        uuid = r["uuid"]
        # candidates per measured quantity: (created, file, value)
        picks: Dict[str, _Pick] = {}
        notes: Dict[str, str] = {}
        probe_card_seen: List[Tuple[float, str, dict, dict]] = []

        def pk(name: str) -> _Pick:
            return picks.setdefault(name, _Pick())

        for src in stage0:
            g = (src["data"].get("gpus") or {}).get(uuid)
            if not g:
                continue
            at, f = src["created"], src["file"]
            pk("bf16").offer(at, f, _num(g.get("gemm_tflops")))
            for lane, tf in (g.get("gemm_lanes") or {}).items():
                pk("int8" if lane == "int8_native" else lane).offer(at, f, _num(tf))
            for lane, why in (g.get("gemm_lane_notes") or {}).items():
                notes.setdefault("int8" if lane == "int8_native" else lane, str(why))
            pk("mem_read").offer(at, f, _num(g.get("membw_read_gbs")))
            pk("mem_copy").offer(at, f, _num(g.get("membw_copy_gbs")))
            pk("mem_gemv").offer(at, f, _num(g.get("membw_gemv_gbs")))
        for src in probes:  # oldest first; the later file overrides a tie by being offered later
            c = next((x for x in src["data"]["cards"] if x.get("uuid") == uuid), None)
            if not c:
                continue
            at, f = src["created"], src["file"]
            probe_card_seen.append((at, f, c, src["data"]))
            pk("bf16").offer(at, f, _num(c.get("gemm_bf16_tflops")))
            pk("fp8_native").offer(at, f, _num(c.get("gemm_fp8_tflops")))
            if c.get("fp8_note"):
                notes["fp8_native"] = str(c["fp8_note"])
            pk("int8").offer(at, f, _num(c.get("gemm_int8_tflops")))
            pk("nvfp4_w4a8").offer(at, f, _num(c.get("gemm_w4a8_int8_tflops")))
            pk("nvfp4_marlin").offer(at, f, _num(c.get("gemm_w4a16_tflops")))
            for lane, why in (c.get("lane_notes") or {}).items():
                notes["int8" if lane == "int8_native" else lane] = str(why)
            pk("mem_read").offer(at, f, _num(c.get("membw_read_gbs")))
            pk("mem_copy").offer(at, f, _num(c.get("membw_copy_gbs")))
            pk("mem_gemv").offer(at, f, _num(c.get("membw_gemv_gbs")))
            pk("h2d_gbs").offer(at, f, _num(c.get("h2d_gbs")))
            pk("d2h_gbs").offer(at, f, _num(c.get("d2h_gbs")))
            pk("h2d_lat").offer(at, f, _num(c.get("h2d_lat_us")))
            pk("d2h_lat").offer(at, f, _num(c.get("d2h_lat_us")))
            pk("sm_count").offer(at, f, _num(c.get("sm_count")))
            pk("l2_mib").offer(at, f, _num(c.get("l2_mib")))
        compute = {}
        for key_, unit, _label in COMPUTE_FORMATS:
            compute[key_] = _lane_node(unit, pk(key_), notes, key_)

        def val(name: str, unit: Optional[str], why: str) -> Dict[str, Any]:
            p = pk(name)
            if p.best is not None:
                at, f, v, _ = p.best
                return node(v, SRC_MEASURED, at=at, probe=f, unit=unit)
            return missing(why, unit)

        ni = "Messarm noch nicht gelaufen (Hardwareprofil messen)"
        latest_state = max(probe_card_seen, key=lambda t: t[0], default=None)
        state = None
        if latest_state:
            at, f, c, _data = latest_state
            why = "Zustand wurde bei der Messung nicht gelesen (kein NVML im Probe-Prozess)"
            state = {
                "sm_mhz": _m(_num(c.get("sm_clock_mhz")), at, f, "MHz", why),
                "sm_max_mhz": _m(_num(c.get("sm_clock_max_mhz")), at, f, "MHz", why),
                "temp_c": _m(_num(c.get("temp_c")), at, f, "°C", why),
                "throttle": list(c.get("throttle_reasons") or []),
                "throttled": bool(c.get("throttle_reasons")),
            }
        ds: Dict[str, Any] = {}
        if ds_fn is not None:
            try:
                ds = dict(ds_fn(r) or {})
            except Exception as e:  # a broken lookup must not hide the card
                issues.append(f"Datenblatt-Suche für {r.get('name')} fehlgeschlagen ({type(e).__name__}: {e})")
        bus, memclk = r.get("mem_bus_width_bits"), r.get("mem_clock_max_mhz")
        nameplate = (
            node(
                round(bus / 8.0 * memclk * 2.0 / 1000.0, 1),
                SRC_DATASHEET,
                unit="GB/s",
                note="Busbreite x max. Speichertakt x 2 aus NVML (Spitzenwert, kein Messwert)",
            )
            if bus and memclk
            else missing("NVML meldet Busbreite oder Speichertakt nicht", "GB/s")
        )
        copy_pick = pk("mem_copy")
        d2d = (
            node(copy_pick.best[2], SRC_MEASURED, at=copy_pick.best[0], probe=copy_pick.best[1], unit="GB/s",
                 note="Kopier-Kernel innerhalb der Karte (lesen+schreiben), dieselbe Messung wie mem_gbs.copy")
            if copy_pick.best
            else missing(ni, "GB/s")
        )
        probed_at = max((t[0] for t in probe_card_seen), default=None)
        # the SM count: a measurement wins; without one the data sheet fills it, labelled as such (never as measured)
        sm_measured = val("sm_count", None, "Geräteeigenschaft wird vom Messarm gelesen (Hardwareprofil messen)")
        sm_node = sm_measured
        if sm_measured["v"] is None and ds.get("sm_count"):
            sm_node = node(ds["sm_count"], SRC_DATASHEET, note=ds.get("sm_note") or "Datenblatt")
        entry = {
            "ord": ordinal,
            "nvml_index": r["nvml_index"],
            "uuid": uuid,
            "pci_bus_id": r.get("pci_bus_id"),
            "name": r["name"],
            "class_key": class_label,
            "card_key": key,
            "cc": r["cc"],
            "sm_count": sm_node,
            "l2_mib": val("l2_mib", "MiB", "Geräteeigenschaft (torch L2_cache_size) wird vom Messarm gelesen"),
            "vram_total_mib": _nv(r.get("total_mib") or None, "MiB", "Speichergröße"),
            "bar1_total_mib": _nv(r.get("bar1_total_mib"), "MiB", "BAR1-Größe (nvidia-smi-Fallback kennt sie nicht)"),
            "pcie": {
                "max_gen": _nv(r.get("pcie_max_gen"), None, "PCIe-Generation (Maximum)"),
                "max_width": _nv(r.get("pcie_max_width"), None, "PCIe-Breite (Maximum)"),
                "cur_gen": _nv(r.get("pcie_cur_gen"), None, "PCIe-Generation (aktuell)",
                               "Momentaufnahme, im Leerlauf meist abgesenkt (ASPM)"),
                "cur_width": _nv(r.get("pcie_cur_width"), None, "PCIe-Breite (aktuell)",
                                 "Momentaufnahme, im Leerlauf meist abgesenkt (ASPM)"),
            },
            "mem_gbs": {
                "read": val("mem_read", "GB/s", ni),
                "copy": val("mem_copy", "GB/s", ni),
                "gemv": val("mem_gemv", "GB/s", ni),
                "nameplate": nameplate,
            },
            "compute": compute,
            "d2d_intra_gbs": d2d,
            "h2d": {"gbs": val("h2d_gbs", "GB/s", ni), "lat_us": val("h2d_lat", "µs", ni)},
            "d2h": {"gbs": val("d2h_gbs", "GB/s", ni), "lat_us": val("d2h_lat", "µs", ni)},
            "power": {
                "limit_w": _nv(r.get("power_limit_w"), "W", "Leistungsgrenze"),
                "default_w": _nv(r.get("power_default_w"), "W", "Standard-Leistungsgrenze"),
            },
            "clocks": {
                "sm_max_mhz": _nv(r.get("sm_clock_max_mhz"), "MHz", "maximalen SM-Takt"),
                "mem_max_mhz": _nv(r.get("mem_clock_max_mhz"), "MHz", "maximalen Speichertakt"),
            },
            "mem_bus_bits": _nv(r.get("mem_bus_width_bits"), "bit", "Speicher-Busbreite (nvidia-smi-Fallback kennt sie nicht)"),
            "state": state,
            "probed_at": probed_at,
        }
        if ds_fn is not None:
            # the nominal bandwidth of the catalog card (a data-sheet figure, not NVML's bus x clock peak above)
            if ds.get("mem_bw_gbs"):
                entry["mem_gbs"]["nominal"] = node(ds["mem_bw_gbs"], SRC_DATASHEET, unit="GB/s",
                                                   note=ds.get("bw_note") or "Datenblatt-Nennbandbreite des Katalogs")
            else:
                entry["mem_gbs"]["nominal"] = missing("kein Katalogeintrag mit Nennbandbreite für diese Karte", "GB/s")
            entry["catalog"] = ds.get("catalog") or None
        if probed_at is not None:
            age = now - probed_at
            entry["age_s"] = round(age, 1)
            entry["stale"] = age > MAX_AGE_S
        # driver the newest probe saw vs the live one
        if latest_state and driver and latest_state[3].get("driver") not in (None, driver):
            entry["driver_mismatch"] = {"probe": latest_state[3].get("driver"), "live": driver}
        cards_out.append(entry)
        # An open gap is a measurement that has not run.  A format the card itself cannot run (fp8 on sm_86,
        # W4A8 on sm_12x: its reason is stored) is final, and the stage-0-only lanes (fp8 Marlin / W8A16) are
        # not the probe's to measure -- neither keeps "Hardwareprofil messen" lit forever.
        gaps = [k for k in PROBE_FORMATS if compute[k]["v"] is None and k not in notes]
        # a data-sheet SM count is no measurement: the gap stays open until the probe has read it
        gaps += ["sm_count"] if sm_measured["v"] is None else []
        gaps += [k for k in ("l2_mib", "d2d_intra_gbs") if entry[k]["v"] is None]
        gaps += [f"h2d.{k}" for k, n in entry["h2d"].items() if n["v"] is None]
        gaps += [f"d2h.{k}" for k, n in entry["d2h"].items() if n["v"] is None]
        unmeasured[str(ordinal)] = gaps

    # ---- links: the ordered pair matrix (newest per ordered pair) + NCCL + BAR1 stretch
    links: List[Dict[str, Any]] = []
    newest_pair: Dict[Tuple[str, str], Tuple[float, str, dict]] = {}
    for src in probes:
        for p in src["data"].get("pairs") or []:
            k = (p.get("src_uuid"), p.get("dst_uuid"))
            if k[0] in ord_of and k[1] in ord_of:
                if k not in newest_pair or src["created"] >= newest_pair[k][0]:
                    newest_pair[k] = (src["created"], src["file"], p)
    for (a, b), (at, f, p) in sorted(newest_pair.items(), key=lambda kv: (ord_of[kv[0][0]], ord_of[kv[0][1]])):
        t = str(p.get("transport") or "")
        links.append(
            {
                "src": ord_of[a],
                "dst": ord_of[b],
                "transport": "p2p" if "p2p" in t else "host_staging" if "staging" in t else t,
                "transport_label": t,
                "gbs": node(_num(p.get("bandwidth_gbs")), SRC_MEASURED, at=at, probe=f, unit="GB/s"),
                "lat_us": node(_num(p.get("latency_us")), SRC_MEASURED, at=at, probe=f, unit="µs"),
                "peer_access": bool(p.get("peer_access")),
                "note": p.get("note") or "",
            }
        )
    # the stage-0 NCCL table: one direction measured, the reverse mirrored
    nccl_seen: Dict[Tuple[int, int], Tuple[float, str, float]] = {}
    for src in stage0:
        for kname, ent in (src["data"].get("links") or {}).items():
            parts = str(kname).split("|")
            if len(parts) != 2 or parts[0] not in ord_of or parts[1] not in ord_of or not isinstance(ent, dict):
                continue
            gbs = _num(ent.get("p2p_gbs"))
            if gbs is None:
                continue
            kk = (ord_of[parts[0]], ord_of[parts[1]])
            if kk not in nccl_seen or src["created"] >= nccl_seen[kk][0]:
                nccl_seen[kk] = (src["created"], src["file"], gbs)
    for (a, b), (at, f, gbs) in sorted(nccl_seen.items()):
        links.append({"src": a, "dst": b, "transport": "nccl", "transport_label": "nccl p2p (Stufe 0)",
                      "gbs": node(gbs, SRC_MEASURED, at=at, probe=f, unit="GB/s"),
                      "lat_us": missing("Stufe-0-Probe misst keine Latenz je Paar", "µs"),
                      "peer_access": None, "note": ""})
        if (b, a) not in nccl_seen:
            links.append({"src": b, "dst": a, "transport": "nccl", "transport_label": "nccl p2p (Stufe 0)",
                          "gbs": node(gbs, SRC_ESTIMATED, at=at, probe=f, unit="GB/s",
                                      note="gespiegelt aus der Gegenrichtung, nicht gemessen"),
                          "lat_us": missing("Stufe-0-Probe misst keine Latenz je Paar", "µs"),
                          "peer_access": None, "note": ""})
    # the BAR1 stretch: explicitly not measured, per ordered pair
    for a in range(len(uuids)):
        for b in range(len(uuids)):
            if a != b:
                links.append({"src": a, "dst": b, "transport": "bar1", "transport_label": "BAR1 (barlink)",
                              "gbs": missing(BAR1_SHORT, "GB/s"), "lat_us": missing(BAR1_SHORT, "µs"),
                              "peer_access": None, "note": ""})

    # ---- provenance of the whole view
    used_probes = [
        {"file": p["file"], "created": p["created"], "driver": p["driver"], "cards": len(p["data"]["cards"])}
        for p in probes
        if any(c.get("uuid") in ord_of for c in p["data"]["cards"])
    ]
    used_stage0 = [
        {"file": p["file"], "created": p["created"], "driver": p["driver"], "cards": len(p["data"]["gpus"])}
        for p in stage0
        if any(u in ord_of for u in p["data"]["gpus"])
    ]
    torch_v = cuda_v = None
    if probes:
        last = probes[-1]["data"]
        torch_v, cuda_v = last.get("torch_version"), last.get("cuda_version")
    gaps_total = sum(len(v) for v in unmeasured.values())
    doc: Dict[str, Any] = {
        "schema": SCHEMA,
        "driver": driver,
        "cuda": cuda_v,
        "torch": torch_v,
        "inventory_sig": [c["class_key"] for c in cards_out],
        "cards": cards_out,
        "links": links,
        "sources": {"card_probe": used_probes, "stage0": used_stage0, "nvml": {"issues": issues, "cards": len(cards_nvml)}},
        "unmeasured": unmeasured,
        "measure_needed": (not cards_out) or any(c.get("probed_at") is None for c in cards_out) or gaps_total > 0,
        "bar1": {"measured": False, "note": BAR1_NOT_MEASURED},
        "formats": [{"key": k, "unit": u, "label": lbl} for k, u, lbl in COMPUTE_FORMATS],
        "src_vocab": list(SOURCES),
    }
    doc["id"] = "sha256:" + hashlib.sha256(_canonical({k: v for k, v in doc.items() if k != "id"}).encode()).hexdigest()
    doc["created"] = now
    return doc


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


# ---------------------------------------------------------------------------
# honesty check (also the test's gate)
# ---------------------------------------------------------------------------


def validate(doc: dict) -> List[str]:
    """Every value node has a known ``src``; "gemessen" names when/where; ``v`` null means "nicht gemessen" + why."""
    problems: List[str] = []

    def walk(path: str, x):
        if isinstance(x, dict):
            if "src" in x and "v" in x:
                src, v = x["src"], x["v"]
                if src not in SOURCES:
                    problems.append(f"{path}: unbekannte Quelle {src!r}")
                if src == SRC_MEASURED and (v is None or x.get("at") is None or not x.get("probe")):
                    problems.append(f"{path}: 'gemessen' ohne Wert/Zeit/Probe")
                if v is None and src != SRC_NONE:
                    problems.append(f"{path}: leerer Wert als {src!r} statt 'nicht gemessen'")
                if src == SRC_NONE and (v is not None or not x.get("note")):
                    problems.append(f"{path}: 'nicht gemessen' braucht v=null und einen Grund")
                if src == SRC_ESTIMATED and not x.get("note"):
                    problems.append(f"{path}: 'geschätzt' braucht eine Herleitung")
                return
            for k, v in x.items():
                walk(f"{path}.{k}" if path else k, v)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(f"{path}[{i}]", v)

    walk("", {"cards": doc.get("cards"), "links": doc.get("links")})
    return problems


# ---------------------------------------------------------------------------
# persistence: written at the first start, replaced only on request
# ---------------------------------------------------------------------------

#: ``capture`` states.  ``erst_erfasst`` = first start, written now; ``neu_erfasst`` = replaced on request;
#: ``vorhanden`` = file and live cards agree; ``abweichend`` = they differ (file kept); ``nur_gespeichert`` = NVML says
#: nothing, the file is the profile; ``keine_karten`` = nothing to persist; ``nicht_schreibbar`` = the write failed.
CAPTURE_STATES = ("erst_erfasst", "neu_erfasst", "vorhanden", "abweichend", "nur_gespeichert", "keine_karten", "nicht_schreibbar")


def persist_path(env: Optional[dict] = None) -> str:
    """The persisted profile's file: ``$FLLIPER_HARDWARE_PROFILE`` or ``/var/lib/flliper/hardware.json``."""
    e = os.environ if env is None else env
    return e.get(PERSIST_ENV) or DEFAULT_PERSIST_PATH


def load_profile(path: str) -> Tuple[Optional[dict], Optional[str]]:
    """``(document, problem)``.  Missing file: ``(None, None)``; unreadable or another schema: ``(None, why)``."""
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except (FileNotFoundError, NotADirectoryError):
        return None, None
    except (OSError, ValueError) as e:
        return None, f"nicht lesbar ({type(e).__name__}: {e})"
    if not isinstance(d, dict) or d.get("schema") != SCHEMA or not isinstance(d.get("cards"), list):
        return None, f"kein {SCHEMA}-Dokument"
    return d, None


def save_profile(doc: dict, path: str, *, reason: str, now: Optional[float] = None) -> Dict[str, Any]:
    """Write ``doc`` (plus a ``capture`` stamp) to ``path`` atomically.  Never raises: ``{"ok", "error"}``."""
    now = time.time() if now is None else now
    out = dict(doc)
    out["capture"] = {"at": now, "reason": reason}
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return {"ok": False, "error": f"{type(e).__name__}: {e.strerror or e}"}
    return {"ok": True, "error": None, "at": now}


def _identity(doc: dict) -> Tuple[Dict[str, tuple], Optional[str]]:
    cards = {}
    for c in doc.get("cards") or []:
        vram = (c.get("vram_total_mib") or {}).get("v")
        cards[str(c.get("uuid"))] = (c.get("name"), vram, tuple(c.get("cc") or ()), c.get("pci_bus_id"))
    return cards, doc.get("driver")


def compare(persisted: dict, live: dict) -> Dict[str, Any]:
    """Whether the persisted profile still describes the live cards: ``{"same", "changes": [text, ...]}``.

    Compared: the set of card UUIDs, each card's name / VRAM / cc / PCI bus, and the driver.  Measured values are not
    compared (a new measurement is not a different machine)."""
    a, da = _identity(persisted)
    b, db = _identity(live)
    changes: List[str] = []
    for u in sorted(set(a) - set(b)):
        changes.append(f"Karte {a[u][0]} ({u}) ist nicht mehr da")
    for u in sorted(set(b) - set(a)):
        changes.append(f"neue Karte {b[u][0]} ({u})")
    for u in sorted(set(a) & set(b)):
        for what, x, y in zip(("Name", "VRAM MiB", "cc", "PCI-Bus"), a[u], b[u]):
            if x != y:
                changes.append(f"Karte {u}: {what} war {x}, jetzt {y}")
    if da is not None and db is not None and da != db:
        changes.append(f"Treiber war {da}, jetzt {db}")
    return {"same": not changes, "changes": changes}


def capture(
    path: str,
    *,
    live: Optional[dict] = None,
    force: bool = False,
    reason: Optional[str] = None,
    now: Optional[float] = None,
    **build_kwargs,
) -> Dict[str, Any]:
    """First-start persistence (and "Neu erfassen" with ``force``).

    ``live`` is an assembled document (default: ``build(**build_kwargs)``).  Returns ``{"state", "show", "persisted",
    "live", "drift", "error", "path"}`` where ``show`` is the document to display: the live one when it has cards, else
    the persisted one.  Rules: no file -> write the live profile (``erst_erfasst``); file and live agree ->
    ``vorhanden``; they differ -> ``abweichend`` and the file stays; ``force`` replaces the file when the live profile
    has cards (an empty live view never overwrites a persisted one); no cards anywhere -> ``keine_karten``."""
    live = live if live is not None else build(**build_kwargs)
    has_cards = bool(live.get("cards"))
    persisted, problem = load_profile(path)
    res: Dict[str, Any] = {"state": None, "show": live, "persisted": persisted, "live": live, "drift": None,
                           "error": None, "path": path}
    if problem:
        res["error"] = f"gespeicherte Datei {problem}"
    if force or persisted is None:
        if not has_cards:
            res.update(state="keine_karten", show=persisted or live,
                       error=(res["error"] + "; " if res["error"] else "") + "NVML meldet keine Karte: nichts gespeichert, nichts überschrieben")
            return res
        why = reason or ("Neu erfassen" if force else "erster Start")
        w = save_profile(live, path, reason=why, now=now)
        if not w["ok"]:
            res.update(state="nicht_schreibbar", error=(res["error"] + "; " if res["error"] else "") + f"Speichern fehlgeschlagen: {w['error']}")
            return res
        res.update(state="neu_erfasst" if force else "erst_erfasst", persisted=load_profile(path)[0])
        return res
    if not has_cards:
        res.update(state="nur_gespeichert", show=persisted)
        return res
    drift = compare(persisted, live)
    res.update(state="vorhanden" if drift["same"] else "abweichend", drift=drift)
    return res


# ---------------------------------------------------------------------------
# measuring: one child interpreter, exactly the chosen cards
# ---------------------------------------------------------------------------


def _python_root() -> str:
    """``<tree>/python`` of this file (…/python/sglang/srt/rigmon/hardware_profile.py)."""
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def probe_command(python: str, prefix: Sequence[str] = ()) -> List[str]:
    return [*prefix, python, "-m", "sglang.srt.rigmon.card_probe", "--run", "--json"]


def probe_env(uuids: Sequence[str], base: Optional[dict] = None, pythonpath: Optional[str] = None) -> dict:
    """Environment of the probe child: only the chosen cards are visible, by UUID.

    UUIDs, not indices: CUDA's enumeration order and NVML's differ on this class of machine,
    and the pair cache is keyed by UUID anyway.  PCI bus order is pinned for the same reason."""
    env = dict(base if base is not None else os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    root = pythonpath or _python_root()
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_measurement(
    nvml_indexes: Sequence[int],
    *,
    python: Optional[str] = None,
    prefix: Sequence[str] = (),
    timeout_s: float = 540.0,
    cards: Optional[List[dict]] = None,
    runner: Optional[Callable[..., Any]] = None,
    pythonpath: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the card probe on exactly ``nvml_indexes`` (a child interpreter, CUDA context there only).

    Returns ``{"ok", "rc", "seconds", "uuids", "profile" (the probe's JSON), "stderr_tail", "warnings"}``.
    ``pythonpath`` is the ``<tree>/python`` the child imports ``sglang`` from (default: the tree this file sits in;
    the dashboard loads this file from a staged mini tree and points the child at the full one).
    ``runner(cmd, env, timeout)`` -> ``(rc, stdout, stderr)`` is injectable for tests."""
    cards = cards if cards is not None else read_nvml()[0]
    by_idx = {c["nvml_index"]: c for c in cards}
    unknown = [i for i in nvml_indexes if i not in by_idx]
    if unknown:
        raise ValueError(f"unbekannte Karte(n) {unknown}; NVML kennt {sorted(by_idx)}")
    uuids = [by_idx[i]["uuid"] for i in nvml_indexes]
    cmd = probe_command(python or sys.executable, prefix)
    env = probe_env(uuids, pythonpath=pythonpath)
    t0 = time.time()
    if runner is None:
        try:
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s, check=False)
            rc, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            rc, out, err = 124, "", f"Zeitüberschreitung nach {timeout_s:.0f} s"
    else:
        rc, out, err = runner(cmd, env, timeout_s)
    seconds = round(time.time() - t0, 1)
    profile = None
    for i, line in enumerate((out or "").splitlines()):
        if line.startswith("{"):
            try:
                profile = json.loads("\n".join((out or "").splitlines()[i:]))
            except ValueError:
                profile = None
            break
    warnings = [ln[len("WARNING "):] for ln in (err or "").splitlines() if ln.startswith("WARNING ")]
    return {
        "ok": rc == 0 and profile is not None,
        "rc": rc,
        "seconds": seconds,
        "uuids": uuids,
        "nvml_indexes": list(nvml_indexes),
        "profile": profile,
        "stderr_tail": (err or "")[-1200:],
        "warnings": warnings,
        "cmd": cmd,
    }


def duration_line(result: dict, cards: Sequence[dict]) -> str:
    """One line for the boot runner: wall time and the duration per card and per arm, from the data.

    Example: ``HWPROFIL-MESSUNG ok rc=0 wall=71.3s nvml0=24.1s[membw=4.2,bf16=0.9,...] nvml1=19.0s[...]``"""
    idx_of = {c["uuid"]: c["nvml_index"] for c in cards}
    parts = [f"HWPROFIL-MESSUNG {'ok' if result.get('ok') else 'FEHLER'} rc={result.get('rc')} wall={result.get('seconds')}s"]
    for c in ((result.get("profile") or {}).get("cards") or []):
        arms = ",".join(f"{k}={v}" for k, v in (c.get("arm_seconds") or {}).items())
        parts.append(f"nvml{idx_of.get(c.get('uuid'), '?')}={c.get('seconds')}s[{arms}]")
    pr = result.get("profile") or {}
    if pr.get("pairs"):
        parts.append(f"paare={len(pr['pairs'])}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="hardware_profile", description=__doc__.split("\n\n")[0])
    ap.add_argument("--measure", action="store_true", help="run the card probe first (needs a gpuq window!)")
    ap.add_argument("--cards", default="", help="NVML indexes, e.g. 0,1,2 (with --measure)")
    ap.add_argument("--python", default=None, help="interpreter with torch + sgl_kernel (default: this one)")
    ap.add_argument("--timeout-s", type=float, default=540.0)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--line", action="store_true", help="print only the duration line (with --measure)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    measured = None
    if args.measure:
        if not args.cards.strip():
            print("--measure braucht --cards (NVML-Indizes)", file=sys.stderr)
            return 2
        idx = [int(x) for x in args.cards.split(",") if x.strip()]
        cards, driver, issues = read_nvml()
        measured = run_measurement(idx, python=args.python, timeout_s=args.timeout_s, cards=cards)
        print(duration_line(measured, cards), file=sys.stderr if not args.line else sys.stdout)
        if args.line:
            return 0 if measured["ok"] else 1
        if not measured["ok"]:
            print(measured["stderr_tail"], file=sys.stderr)
    doc = build(cache_dir=args.cache_dir)
    if measured is not None:
        doc["messung"] = {k: measured[k] for k in ("ok", "rc", "seconds", "nvml_indexes", "warnings")}
    print(json.dumps(doc, indent=1, ensure_ascii=False))
    return 0 if measured is None or measured["ok"] else 1


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(_main())
