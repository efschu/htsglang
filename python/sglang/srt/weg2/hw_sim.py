"""HW-P1b 1003: the hardware SIMULATION harness -- a synthetic inventory
(1..8 cards, any mix of sm86 / sm89 / sm120, SM count and memory per card)
through the launcher's own pre-spawn hardware path, per release model.

User order 03.10. ~19:30Z (verbatim): "unsere software muss mit beliebiger
anzahl an karten und sm86 sm89 und sm120 laufen. 1,2,3,4,5,6 ... karten".
Test plan correction (~19:35Z): on the rig only N = 3 (NF, 27B), 27B INT8/FP8
on 5090 + 3080 and small GGUF/NVFP4 on 3080 + 3080 are testable; everything
else is SIMULATION. This module is that simulation and the progress meter of
the N-card work: the share of cells that RUN instead of being refused.

WHAT ONE CELL RUNS (in launcher.main's order, the REAL functions, no copy):

1. the inventory as an NVML recording through the replay seam
   (``SGLANG_NVML_REPLAY_JSON``) and the launch's ``--cards`` selection
   (``launcher.selected_nvml_cards``) -- the one card-list producer;
2. the arch gate (``card_identity.arch_gate`` via ``launcher.resolve_cards``,
   live ``SUPPORTED_ARCHS``): HW-ARCH;
3. the card order (``launcher.order_cards``);
4. the topology of the card count (``launcher.topology_check_line`` ->
   ``topology.plan_topology``): HW-TOPOLOGY / HW-COUNT with the concrete
   blockers of this model's launch argv/env;
5. the calibration inventory (``launcher.inventory_check_line``):
   HW-UNCALIBRATED;
6. a weight FIT bound (a NECESSARY condition only -- checkpoint bytes
   against the summed VRAM, one card for N = 1; NOT the planned fit check
   of the single mode, P4): FIT;
7. the profile resolution (``form.format_of`` + the row's per-arch kernel
   path) and the planner preset (``planner.flags._match_calibration``):
   notes, never a refusal;
8. the argv shape the topology gives (``launcher.argv_p`` / ``argv_d`` with
   one sentinel budget per card): ``--pp-size``, ``--tp-size``,
   ``--rank-gpu-id``.

The cell's RESULT is the first refusal in that order (what the launcher
would print), its BLOCKERS are all of them (what has to change). Every stage
is evaluated even after a refusal, so a refused row still names everything.

WHAT IT DOES NOT PROVE (HOCHRECHNUNG != MESSUNG): runtime -- allocator
residue, BAR1/smallbar driver chain, kernel correctness and speed on sm_89,
flip time, host RAM peaks. It does not run ``launcher.main`` itself (that
touches ports, /dev/shm and the host); it runs the hardware-dependent
decisions main makes before the first spawn.

CPU-only, GPU-free, NVML-free. ``python -m sglang.srt.weg2.hw_sim --help``
(or ``tools/hw_sim.py``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

MIB = 1 << 20

#: Result labels (table); "läuft" = every gate passes.
RUNS = "läuft"
REFUSED = "verweigert"


@dataclass(frozen=True)
class SimCard:
    """A synthetic card: what NVML would report (name, cc, total, BAR1, bus
    width, max memory clock) plus the SM count (catalog field; NVML-side
    identity does not use it, the table prints it)."""

    key: str
    name: str
    cc: Tuple[int, int]
    total_mib: int
    sm_count: int
    bar1_mib: int
    bus_bits: int
    mem_clock_mhz: int

    @property
    def sm(self) -> str:
        return f"sm{self.cc[0]}{self.cc[1]}"


#: The catalog (nameplate figures; totals as NVML reports them where the rig
#: or a recording has them: 3080-20G 20480, 5090 32607, A6000 49140, 4090
#: 24564, PRO 6000 97887). BAR1 256 MiB = GeForce without ReBAR (S).
CATALOG: Dict[str, SimCard] = {c.key: c for c in (
    SimCard("3080-10G", "NVIDIA GeForce RTX 3080", (8, 6), 10240, 68, 256, 320, 9501),
    SimCard("3080-20G", "NVIDIA GeForce RTX 3080", (8, 6), 20480, 68, 256, 320, 9501),
    SimCard("3090", "NVIDIA GeForce RTX 3090", (8, 6), 24576, 82, 256, 384, 9751),
    SimCard("A6000", "NVIDIA RTX A6000", (8, 6), 49140, 84, 256, 384, 8001),
    SimCard("4060Ti-16G", "NVIDIA GeForce RTX 4060 Ti", (8, 9), 16380, 34, 256, 128, 9001),
    SimCard("4080", "NVIDIA GeForce RTX 4080", (8, 9), 16376, 76, 256, 256, 11201),
    SimCard("4090", "NVIDIA GeForce RTX 4090", (8, 9), 24564, 128, 256, 384, 10501),
    SimCard("5080", "NVIDIA GeForce RTX 5080", (12, 0), 16303, 84, 16384, 256, 15001),
    SimCard("5090", "NVIDIA GeForce RTX 5090", (12, 0), 32607, 170, 32768, 512, 14001),
    SimCard("PRO6000", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", (12, 0), 97887, 188,
            131072, 512, 14001),
)}

#: The reference rig as NVML enumerates it (nvml0 3080, nvml1 5090, nvml2 3080).
REFERENCE_RIG: Tuple[str, ...] = ("3080-20G", "5090", "3080-20G")


@dataclass(frozen=True)
class SimModel:
    """A release model: launcher profile row + weight format + the launch
    argv/env of its release profile that the hardware path reads (the
    positional vectors, the weight source, dual, L1.5)."""

    key: str
    profile: str
    weight_format: str
    #: checkpoint MiB (du on the rig; INT8 from the plan, S); None = experts
    #: may live in the host store (NF), no VRAM bound
    ckpt_mib: Optional[int]
    argv: Tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    source: str = ""


#: 27B attention geometry (plan 3.1): 24 Q heads, 4 KV heads.
QWEN27B_KV_HEADS = 4

_27B_BASE = ("--weg2-weight-source", "exchange", "--user-reserve-mib", "1800,1400,1400")
_NF_FR_P = "0.324,0.637,0.39"
_NF_FR_D = "0.06,0.51,0.48"
#: The hardware-relevant part of each release profile (docker/profiles_release,
#: 03.10.): weight source, dual, L1.5 and every positional vector with its
#: value. The hardware path reads only their COUNT and the inventory they
#: were written for; test_hw_sim_p1b_1003 checks these against the real
#: PROFILE_ARGS where the docker tree is on the box (--profiles-dir sources
#: the real files instead).
MODELS: Dict[str, SimModel] = {m.key: m for m in (
    SimModel("27B-INT8", "qwen27b", "int8", 27648,
             _27B_BASE + ("--env-d", "SGLANG_WEG2_EXTEND_TRIM_MIB=1200,0,0"),
             {"SGLANG_WEG2_L15": "1", "SGLANG_WEG2_L15_MIB": "c1=7616,c2=1792"},
             "27b.env (27b-base.env, :50 extend trim, :117/:133 L15 c1=7616,c2=1792); ckpt 27 G (plan 3.1, S)"),
    SimModel("27B-FP8", "qwen27b", "fp8", 25422, _27B_BASE, {}, "27b-fp8.env; du 25422 MiB"),
    SimModel("27B-NVFP4", "qwen27b", "nvfp4", 18753,
             _27B_BASE + ("--pp-stage-ratio", "49,8,7", "--pp-attn-stage-ratio", "12,2,2"), {},
             "27b-nvfp4.env:79; du 18753 MiB"),
    SimModel("27B-GGUF", "qwen27b", "gguf", 13593,
             _27B_BASE + ("--pp-stage-ratio", "42,11,11", "--pp-attn-stage-ratio", "10,3,3"), {},
             "27b-gguf.env:49 (IQ4_XS); du 13593 MiB"),
    SimModel("27B-NVFP4-DUAL", "qwen27b", "nvfp4", 18753,
             _27B_BASE + ("--dual-share", "--pp-stage-ratio", "45,10,9", "--pp-attn-stage-ratio", "11,2,3",
                          "--extra-p=--max-running-requests=1 --rank-gpu-memory-mib 8740,3000,3500"),
             {}, "27b-nvfp4-dual.env:81/90; du 18753 MiB"),
    SimModel("NF", "nextflash", "int4-mixed", None,
             ("--weg2-weight-source", "exchange",
              "--pp-stage-ratio", "29,11,8", "--pp-attn-stage-ratio", "7,3,2",
              "--pp-cut-expert-device-fraction", _NF_FR_P, "--pp-cut-expert-lru-rows", "32,32,32",
              "--user-reserve-mib", "0,0,0", "--d-foreign-context-mib", "1446,896,894",
              "--d-nontorch-mib", "1981,528,524",
              "--env-p", f"SGLANG_MOE_SCRATCH_SLOTS=32;SGLANG_MOE_RESIDENT_EXPERT_FRACTION={_NF_FR_P}",
              "--env-d", f"SGLANG_MOE_SCRATCH_SLOTS=100,48,48;SGLANG_MOE_RESIDENT_EXPERT_FRACTION={_NF_FR_D}",
              f"--extra-p=--rank-moe-resident-fraction {_NF_FR_P} --rank-user-reserve-mib 0,0,0",
              "--extra-d=--rank-role host,worker,worker --rank-tp-ratio 1,0,0 --rank-moe-ratio 183,137,168 "
              f"--rank-moe-resident-fraction {_NF_FR_D} --rank-user-reserve-mib 0,0,0"),
             {}, "nf.env:90-92,105-107,128-139 (Form A); experts in the host store, no VRAM bound"),
)}

#: The release profile file of each model (docker/profiles_release).
PROFILE_FILES: Dict[str, str] = {
    "27B-INT8": "27b.env", "27B-FP8": "27b-fp8.env", "27B-NVFP4": "27b-nvfp4.env",
    "27B-GGUF": "27b-gguf.env", "27B-NVFP4-DUAL": "27b-nvfp4-dual.env", "NF": "nf.env",
}


def profile_args(path: str) -> List[str]:
    """``PROFILE_ARGS`` of a release profile file (sourced by bash, the way
    the entrypoint does; nothing else of the profile runs)."""
    import subprocess

    out = subprocess.run(
        ["bash", "-c", 'set +e; source "$1" >/dev/null 2>&1; printf "%s\\0" "${PROFILE_ARGS[@]}"', "x", path],
        capture_output=True, text=True, env={**os.environ, "HOME": os.environ.get("HOME", "/root")},
        check=False, timeout=60).stdout
    return [a for a in out.split("\0") if a]


def models_from_profiles(profiles_dir: str) -> Dict[str, SimModel]:
    """:data:`MODELS` with each model's argv replaced by its real profile
    file's PROFILE_ARGS (the L1.5 env stays the embedded one: the profile
    sets it through the entrypoint's _form, not PROFILE_ARGS). A model whose
    file is missing keeps its embedded argv."""
    out = dict(MODELS)
    for key, fname in PROFILE_FILES.items():
        path = os.path.join(profiles_dir, fname)
        if not os.path.isfile(path):
            continue
        m = MODELS[key]
        out[key] = SimModel(m.key, m.profile, m.weight_format, m.ckpt_mib,
                            tuple(profile_args(path)), m.env,
                            f"{path} (PROFILE_ARGS)")
    return out


def _round_robin(keys: Sequence[str], n: int) -> List[str]:
    return [keys[i % len(keys)] for i in range(n)]


def scenarios(n: int) -> List[Tuple[str, List[str]]]:
    """The standard inventories of ``n`` cards (label, catalog keys in NVML
    order): homogeneous per arch, the reference family, a three-arch mix."""
    out = [
        ("sm86 3080-20G", ["3080-20G"] * n),
        ("sm86 3090", ["3090"] * n),
        ("sm89 4090", ["4090"] * n),
        ("sm120 5090", ["5090"] * n),
        ("ref 5090+3080", (["3080-20G"] + ["5090"] + ["3080-20G"] * max(0, n - 2))[:n] if n >= 2 else ["5090"]),
    ]
    if n >= 2:
        out.append(("mix 5090/4090/3080", _round_robin(["5090", "4090", "3080-20G"], n)))
    return out


#: Subsets of the reference rig via --cards (plan 6, M3/M5): replay the rig,
#: select NVML indices.
RIG_SUBSETS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("rig --cards 1,0 (5090+3080)", (1, 0)),
    ("rig --cards 0,2 (3080+3080)", (0, 2)),
    ("rig --cards 1 (5090)", (1,)),
)


def replay_rows(keys: Sequence[str]) -> List[dict]:
    """The NVML recording of an inventory (registry.nvml replay format)."""
    rows = []
    for i, k in enumerate(keys):
        c = CATALOG[k]
        rows.append({
            "index": i, "uuid": f"GPU-sim{i:02d}-{k}", "name": c.name,
            "total_bytes": c.total_mib * MIB, "reserved_bytes": 0,
            "pci_bus_id": f"0000:{i + 1:02x}:00.0",
            "cc_major": c.cc[0], "cc_minor": c.cc[1],
            "bar1_total_bytes": c.bar1_mib * MIB,
            "mem_bus_width_bits": c.bus_bits, "mem_clock_max_mhz": c.mem_clock_mhz,
        })
    return rows


@contextlib.contextmanager
def replayed(keys: Sequence[str], selection: Optional[Sequence[int]] = None) -> Iterator[None]:
    """Arm the NVML replay seam with ``keys`` and the launcher's ``--cards``
    selection; both restored on exit."""
    from sglang.srt.registry import nvml as nvml_registry
    from sglang.srt.weg2 import launcher as L

    fd, path = tempfile.mkstemp(prefix="hw_sim_", suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(replay_rows(keys), fh)
    old_env = os.environ.get(nvml_registry.ENV_NVML_REPLAY)
    old_sel = L._CARD_SELECTION
    os.environ[nvml_registry.ENV_NVML_REPLAY] = path
    L._CARD_SELECTION = None if selection is None else tuple(int(i) for i in selection)
    try:
        yield
    finally:
        L._CARD_SELECTION = old_sel
        if old_env is None:
            os.environ.pop(nvml_registry.ENV_NVML_REPLAY, None)
        else:
            os.environ[nvml_registry.ENV_NVML_REPLAY] = old_env
        os.unlink(path)


@dataclass
class CellResult:
    inventory: str
    model: str
    n_cards: int
    result: str = RUNS
    code: str = ""
    blockers: List[str] = field(default_factory=list)
    details: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    order: List[str] = field(default_factory=list)
    argv: str = ""

    def refuse(self, code: str, detail: str) -> None:
        if self.result == RUNS:
            self.result, self.code = REFUSED, code
        self.details.append(f"{code}: {detail}")

    def row(self) -> Dict[str, object]:
        return {"inventory": self.inventory, "model": self.model, "n": self.n_cards,
                "result": self.result, "code": self.code, "blockers": list(self.blockers),
                "order": list(self.order), "argv": self.argv, "notes": list(self.notes),
                "details": list(self.details)}


_ARGV_CACHE: Dict[int, str] = {}


def argv_shape(n: int) -> str:
    """``--pp-size``/``--tp-size``/``--rank-gpu-id`` of the launcher's real
    argv builders for ``n`` cards (one sentinel budget per card)."""
    if n in _ARGV_CACHE:
        return _ARGV_CACHE[n]
    from sglang.srt.weg2 import launcher as L
    from sglang.srt.weg2 import topology as T

    if n < T.MIN_CARDS or n > T.MAX_CARDS_BAR1:
        shape = "-"
    else:
        try:
            p = L.argv_p("py", L.MODEL_DEFAULT, [1] * n, 1, 1, L.RING_FORM_SENTINEL_STORE_CFG, [], p_bs=1)
            d = L.argv_d("py", L.MODEL_DEFAULT, [1] * n, 1, 1, L.RING_FORM_SENTINEL_STORE_CFG, [], d_bs=1)

            def val(a, flag):
                return a[a.index(flag) + 1] if flag in a else "?"
            shape = (f"P tp{val(p, '--tp-size')}/pp{val(p, '--pp-size')} "
                     f"D tp{val(d, '--tp-size')}/pp{val(d, '--pp-size')} ranks {val(d, '--rank-gpu-id')}")
        except Exception as exc:  # noqa: BLE001 - named in the table, never a crash of the harness
            shape = f"ARGV-ERROR {type(exc).__name__}: {exc}"
    _ARGV_CACHE[n] = shape
    return shape


def _kernel_notes(model: SimModel, cards: Sequence) -> List[str]:
    from sglang.srt.weg2 import form as F

    row = F.profile_row(model.profile)
    wf = None if row is None else row.formats.get(model.weight_format)
    if wf is None:
        return [f"PROFILE: {model.profile}/{model.weight_format} not in the profile registry"]
    notes = []
    archs = sorted({tuple(c.cc) for c in cards if c.cc is not None})
    for cc in archs:
        path = (getattr(wf, "sm12x", "") if cc[0] == 12 else getattr(wf, "sm8x", "")) or "default"
        if cc == (8, 9):
            notes.append(f"sm89: {model.weight_format} via the sm8x path '{path}', unmeasured on Ada "
                         "(needs-hardware)" + ("; fp8_scaled_mm -> FP8-SM89-FALLBACK unless the wheel "
                                               "carries sm_89" if model.weight_format == "fp8" else ""))
        elif path not in ("default", "native"):
            notes.append(f"sm{cc[0]}{cc[1]}: {model.weight_format} kernel path '{path}'")
    n = len(cards)
    if model.profile == "qwen27b" and n > QWEN27B_KV_HEADS:
        notes.append(f"D = TP{n} > {QWEN27B_KV_HEADS} KV heads: replicated KV / uneven DCP "
                     "(KV capacity cost, plan 3.1, S)")
    return notes


def simulate(inventory: str, keys: Sequence[str], model: SimModel,
             selection: Optional[Sequence[int]] = None) -> CellResult:
    """One cell: ``keys`` (catalog keys, NVML order), the ``--cards``
    selection, one release model."""
    from sglang.srt.planner import flags as PF
    from sglang.srt.weg2 import card_identity as CI
    from sglang.srt.weg2 import form as F
    from sglang.srt.weg2 import launcher as L
    from sglang.srt.weg2 import topology as T

    n_sel = len(selection) if selection is not None else len(keys)
    cell = CellResult(inventory=inventory, model=model.key, n_cards=n_sel)
    ns = L.build_parser().parse_args(
        ["--tree", "/sim", "--tag", "sim", "--profile", model.profile,
         "--model", F.profile_row(model.profile).formats[model.weight_format].checkpoint,
         *model.argv])
    env = dict(model.env)
    with replayed(keys, selection):
        try:
            cards = L.selected_nvml_cards()
        except L.Weg2LaunchRefused as exc:  # --cards names no card
            cell.refuse(L.CODE_CARDS, str(exc))
            cell.blockers.append("CARDS")
            return cell
        # 2. arch gate (the real one, live SUPPORTED_ARCHS)
        try:
            L.resolve_cards()
        except L.Weg2LaunchRefused as exc:
            cell.refuse(CI.CODE_ARCH, str(exc))
            bad = sorted({CI._sm(tuple(c.cc)) for c in cards
                          if c.cc is None or tuple(c.cc) not in CI.SUPPORTED_ARCHS})
            cell.blockers.extend(f"ARCH-{b}" for b in bad)
    # 3. order
    order = L.order_cards(list(cards))
    cell.order = [CI.card_key(c) for c in order]
    # 4. topology of the count
    try:
        L.topology_check_line(ns, order, env)
    except L.Weg2LaunchRefused as exc:
        cause = exc.__cause__
        code = str(exc).split(":", 1)[0]
        cell.refuse(code, str(exc).split(" || ")[0])
        cell.blockers.extend(b.code for b in getattr(cause, "blockers", ()))
    # 5. calibration inventory
    try:
        L.inventory_check_line(ns, order)
    except L.Weg2LaunchRefused as exc:
        cell.refuse(CI.CODE_UNCALIBRATED, str(exc)[:400])
        cell.blockers.append("UNCALIBRATED")
    # 6. weight fit bound (necessary condition)
    if model.ckpt_mib is not None and order:
        room = order[0].total_mib if len(order) == 1 else sum(c.total_mib for c in order)
        if model.ckpt_mib >= room:
            cell.refuse("FIT", f"checkpoint {model.ckpt_mib} MiB >= VRAM {room} MiB "
                               f"({'one card' if len(order) == 1 else 'all cards'}), before any KV")
            cell.blockers.append("FIT")
    # 7. profile resolution + planner preset (notes)
    fmt = F.format_of(model.profile, ns.model)
    if fmt != model.weight_format:
        cell.notes.append(f"PROFILE: format_of -> {fmt!r}, expected {model.weight_format!r}")
    cell.notes.extend(_kernel_notes(model, order))
    gpus = [{"name": c.name, "total_mib": c.total_mib, "memory_mib": c.total_mib,
             "cc_major": c.cc[0] if c.cc else None, "cc_minor": c.cc[1] if c.cc else None}
            for c in order]
    quant = "fp8" if model.weight_format == "fp8" else None
    try:
        cal = PF._match_calibration(gpus, quant)
        cell.notes.append(f"planner calibration table (quant={quant}): "
                          + ("hit" if cal is not None else "no entry"))
    except Exception as exc:  # noqa: BLE001 - a planner error is a named note, never a crash
        cell.notes.append(f"planner preset: ERROR {type(exc).__name__}: {exc}")
    # 8. argv shape
    cell.argv = argv_shape(len(order))
    # de-dup blockers, keep order
    seen: List[str] = []
    for b in cell.blockers:
        if b not in seen:
            seen.append(b)
    cell.blockers = seen
    return cell


def grid(ns_cards: Sequence[int] = (1, 2, 3, 4, 5, 6), models: Optional[Sequence[str]] = None,
         with_subsets: bool = True,
         model_table: Optional[Mapping[str, SimModel]] = None) -> List[CellResult]:
    """The standard grid: :func:`scenarios` per N x every model, plus the
    reference rig's ``--cards`` subsets."""
    table = dict(model_table or MODELS)
    mkeys = list(models or table)
    out: List[CellResult] = []
    for n in ns_cards:
        for label, keys in scenarios(int(n)):
            for mk in mkeys:
                out.append(simulate(f"{n}x {label}", keys, table[mk]))
    if with_subsets:
        for label, sel in RIG_SUBSETS:
            for mk in mkeys:
                out.append(simulate(label, REFERENCE_RIG, table[mk], sel))
    return out


def golden_of(cells: Sequence[CellResult]) -> Dict[str, List[object]]:
    """The comparable core of a grid: cell id -> [result, code, blockers]."""
    return {f"{c.inventory} | {c.model}": [c.result, c.code, list(c.blockers)] for c in cells}


def table_md(cells: Sequence[CellResult], wide: bool = False) -> str:
    head = "| N | Inventar | Modell | Ergebnis | Code | Blocker |" + (" argv | Hinweise |" if wide else "")
    sep = "|---|---|---|---|---|---|" + ("---|---|" if wide else "")
    lines = [head, sep]
    for c in cells:
        row = (f"| {c.n_cards} | {c.inventory} | {c.model} | {c.result} | {c.code or '-'} | "
               f"{', '.join(c.blockers) or '-'} |")
        if wide:
            row += f" {c.argv} | {'; '.join(c.notes) or '-'} |"
        lines.append(row)
    return "\n".join(lines)


def summary(cells: Sequence[CellResult]) -> str:
    runs = sum(1 for c in cells if c.result == RUNS)
    by_code: Dict[str, int] = {}
    by_blocker: Dict[str, int] = {}
    for c in cells:
        if c.code:
            by_code[c.code] = by_code.get(c.code, 0) + 1
        for b in c.blockers:
            by_blocker[b] = by_blocker.get(b, 0) + 1
    return (f"HW-SIM cells={len(cells)} runs={runs} refused={len(cells) - runs} "
            f"by_code={json.dumps(by_code, sort_keys=True)} "
            f"by_blocker={json.dumps(by_blocker, sort_keys=True)}")


def _parse_inventory(text: str) -> List[str]:
    keys = [k.strip() for k in str(text).split(",") if k.strip()]
    bad = [k for k in keys if k not in CATALOG]
    if bad:
        raise SystemExit(f"hw_sim: unknown card(s) {bad}; catalog: {', '.join(CATALOG)}")
    return keys


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m sglang.srt.weg2.hw_sim",
        description="HW-P1b: synthetic inventories through the launcher's pre-spawn hardware "
                    "path (arch gate, card order, topology blockers, calibration inventory, "
                    "weight fit bound, profile resolution, argv shape). CPU only.")
    ap.add_argument("--inventory", default="",
                    help="one inventory, catalog keys in NVML order, e.g. '3080-20G,5090,3080-20G'; "
                         "default: the standard grid")
    ap.add_argument("--cards", default="", help="--cards selection (NVML indices) on --inventory")
    ap.add_argument("--n", default="1,2,3,4,5,6", help="card counts of the standard grid")
    ap.add_argument("--models", default=",".join(MODELS), help="models: " + ", ".join(MODELS))
    ap.add_argument("--no-subsets", action="store_true", help="skip the reference rig --cards subsets")
    ap.add_argument("--wide", action="store_true", help="add the argv shape and the notes columns")
    ap.add_argument("--details", action="store_true", help="print every refusal text under the table")
    ap.add_argument("--json", default="", help="write every cell (full rows) to this JSON file")
    ap.add_argument("--write-golden", default="", help="write golden_of(grid) to this file")
    ap.add_argument("--profiles-dir", default="",
                    help="source each model's argv from the real release profile file in this "
                         "directory (e.g. /spinning/gpu-arb/docker/profiles_release) instead of the "
                         "embedded copy")
    ap.add_argument("--catalog", action="store_true", help="print the card catalog and exit")
    ns = ap.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if ns.catalog:
        for c in CATALOG.values():
            print(f"{c.key:12s} {c.name:52s} {c.sm:6s} {c.total_mib:6d} MiB  {c.sm_count:3d} SMs  "
                  f"BAR1 {c.bar1_mib} MiB")
        return 0
    table = models_from_profiles(ns.profiles_dir) if ns.profiles_dir else dict(MODELS)
    models = [m.strip() for m in ns.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise SystemExit(f"hw_sim: unknown model(s) {unknown}; models: {', '.join(MODELS)}")
    if ns.inventory:
        keys = _parse_inventory(ns.inventory)
        sel = None
        if ns.cards:
            sel = [int(x) for x in ns.cards.split(",") if x.strip()]
        label = ",".join(keys) + (f" --cards {ns.cards}" if sel is not None else "")
        cells = [simulate(label, keys, table[m], sel) for m in models]
    else:
        cells = grid([int(x) for x in ns.n.split(",") if x.strip()], models, not ns.no_subsets, table)
    print(table_md(cells, ns.wide))
    print()
    print(summary(cells))
    if ns.details:
        for c in cells:
            for d in c.details:
                print(f"- {c.inventory} | {c.model}: {d}")
    if ns.json:
        with open(ns.json, "w") as fh:
            json.dump([c.row() for c in cells], fh, indent=1, ensure_ascii=False)
    if ns.write_golden:
        with open(ns.write_golden, "w") as fh:
            json.dump(golden_of(cells), fh, indent=1, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
