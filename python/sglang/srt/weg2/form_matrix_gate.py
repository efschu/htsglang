"""FORM-MATRIX-GATE: the desk gate before every 27B boot (no GPU, no boot).

Why (24.09., FORM_AXES_INVENTORY.md): the 27B form never booted on the
Next-Flash line, and each probe died on something a desk could have shown --
an NF assumption in shared code, an NF measurement priced for the 27B. This
gate runs, PER FORM, what the arm would launch -- as a dry run of THIS tree --
and holds it against the last boots of the same form:

  1. ``launcher --dry-run`` of the tree under test with the form's arm argv
     and the arm's exported environment (``--launcher-args`` hands the arm's
     exact argv over; without it the canonical argv below is used);
  2. its WEG2-FORM line must name the form's axes;
  3. group P/D argv and WEG2-GROUP-ENV diffed against the NEWEST boot of this
     form that reached front READY -- every differing flag/key classified
     (SOLVED from measured inputs / PER-BOOT identity / FORM change of this
     tree); an UNEXPLAINED one fails the gate unless ``--explain KEY=why``;
  4. the same argv diff against the last GOOD boot (27B: weg2xsn411), printed
     classified, informational;
  5. the post-READY launcher gates of THIS tree (W7/W10 launcher halves, W9,
     W45 carrier census, W10, W11/W11b) replayed against the reference boot's
     own P/D logs -- which is how a gate that only one producer feeds (W11b,
     xsn417) shows up at the desk instead of after READY.

Exit 0 PASS, 1 FAIL, 2 could not run. Launched as
``python <tree>/python/sglang/srt/weg2/form_matrix_gate.py [--form ...]`` --
BY PATH, so this parent imports no torch (``import sglang`` alone is 612 MiB
RSS; the dry run child needs its own ~1 GiB, and the agents' test cgroup is
3 GiB shared): the torch-free ``form.py`` is loaded by file, the launcher only
for the replay, after every dry run has exited. ``-m sglang.srt.weg2.
form_matrix_gate`` works too, at the higher peak.
It starts no rank, touches no card (``CUDA_VISIBLE_DEVICES=""``; the dry run
reads NVML only) and writes only below ``--work-dir``.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def _load_form():
    """``weg2/form.py`` -- from the package when it is already imported, else
    BY FILE (it is torch-free and imports nothing from sglang)."""
    if "sglang.srt.weg2" in sys.modules or __package__:
        from sglang.srt.weg2 import form as mod

        return mod
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "form.py")
    spec = importlib.util.spec_from_file_location("weg2_form_by_path", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


weg2_form = _load_form()

EVIDENCE_DIR = "/spinning/evidence-665-f1"
VENV_PY = "/spinning/htsglang-gpu/.venv/bin/python"
MODELS = "/spinning/llm_stuff/club-3090/models-cache"

#: Flags whose VALUE the launcher solves per boot from measured inputs (budgets,
#: residues, the cut): a changed value is the solve seeing other inputs, not a
#: changed form. Observed xsn411 -> xsn418: exactly the first three.
SOLVED_FLAGS = frozenset({
    "--max-total-tokens", "--pp-stage-ratio", "--rank-gpu-memory-mib",
    "--pp-attn-stage-ratio", "--pp-layer-set", "--rank-tp-ratio", "--hicache-size",
})
PER_BOOT_FLAGS = frozenset({"--admin-api-key"})
#: Environment keys that are one boot's identity (or exist only once a boot
#: really runs: the dry run builds no TMS preload and maps no xchg region).
PER_BOOT_ENV = frozenset({
    "SGLANG_WEG2_BOOT_TOKEN", "SGLANG_WEG2_XCHG_BOOT", "SGLANG_WEG2_XCHG_REGION",
    "SGLANG_WEG2_TMS_PRELOAD_SO", "SGLANG_MOE_COLD_TIER_INSTANCE",
})
P_PREFILL_TRANSIENT_ENV = "SGLANG_KV_BUDGET_PREFILL_TRANSIENT_MIB"


@dataclass(frozen=True)
class FormCase:
    name: str
    expect: Dict[str, str]
    tag: str
    args: Tuple[str, ...]
    env: Dict[str, str]
    good_ref: str = ""
    #: the NF dry run writes an expert map into --evidence-dir (symlink farm)
    evidence_farm: bool = False
    #: UNEXPLAINED differences against the newest boot of the form FAIL the
    #: gate. Off for NF: the canonical argv is the arm's DEFAULTS, and the NF
    #: seat boots with its own overrides (FR_P/FR_D, chunk, draft on P ...),
    #: so the reference diff there is information, not a verdict.
    strict_ref: bool = True


_Q27_MODEL = f"{MODELS}/Qwen3.8-27B-INT8-gdncov-vocabembed"
_NF_MODEL = f"{MODELS}/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
_NF_DRAFT = f"{MODELS}/Qwen3.8-Flash-Next-MTP-INT4-g32-albucino"
_DEVIATION = "1424e-ratchet-37gib-aus-xsn177-war-ganzarena-pinning-seit-1424e-je-slot"

#: 27B-DFLASH: arm_xsn418.sh step 6 'deviation' (BASE + EXTRA_D + deviation
#: pair), positional defaults CAP=262144 BDEPTH=1 LEGS=both, cpu backup off.
CASE_27B = FormCase(
    name="27B-DFLASH",
    expect={"arch": "dense", "experts": "none", "draft": "dflash", "p_draft": "cold",
            "kv": "paged_dcp", "flip": "family", "vision": "transient"},
    tag="weg2xsnFMG",
    args=(
        "--weg2-weight-source", "exchange",
        "--weg2-xchg-census", "/spinning/gpu-arb/weg2/census/xchg_census_weg2xsn246_27198a2711.json",
        "--weg2-xchg-oncard", "host", "--weg2-vision", "transient",
        "--user-reserve-mib", "1800,1400,1400", "--store-max-gb", "150",
        "--max-kv-per-request", "262144", "--xchg-bounce-depth", "1", "--pin-ledger-arm-m", "600",
        "--host-ledger-deviation", _DEVIATION, "--host-riegel-gib", "94",
        "--weg2-xchg-legs", "both", "--weg2-xchg-inject", "authoritative",
        "--p-bs", "1", "--extra-p=--max-running-requests=2", "--weg2-weights-cpu-backup", "off",
        "--spec-form", "DFLASH", "--d-tp-objective", "maxkv",
        "--host-ledger-deviation", _DEVIATION, "--host-riegel-gib", "93.0",
    ),
    env={
        "SGLANG_HICACHE_ARENA_PREPIN": "1", "SGLANG_HICACHE_ARENA_MAMBA_SLOTS": "112",
        "SGLANG_WEG2_SEQ_SYNC_BATCH_MIB": "256", "SGLANG_WEG2_SEQ_SYNC_BATCH_UNITS": "128",
        "SGLANG_ADMISSION_WEDGE_RECOVERY_SECONDS": "2", "SGLANG_HICACHE_ARENA_LOAD_BLOCK_QUOTA": "16",
        "NCCL_BUFFSIZE": "1048576", "NCCL_MAX_NCHANNELS": "8", "PARK_N": "4", "PARK_REPS": "2750",
    },
    good_ref="weg2xsn411",
)

_NF_EXTRA_P = (
    '--max-total-tokens 262144 --json-model-override-args "{\\"language_model_only\\":true}" '
    "--kv-cache-dtype fp8_e4m3 --page-size 64 --max-running-requests 1 --mamba-ssm-dtype bfloat16 "
    "--ple-offload-embedding --ple-offload-backend checkpoint --hicache-mamba-host-mib 300 "
    f"--hicache-size 4 --speculative-draft-model-path {_NF_DRAFT} "
    "--rank-moe-resident-fraction 0.35,0.6,0.6 --rank-user-reserve-mib 0,0,0 --disable-cuda-graph "
    "--context-length 262144 --max-kv-per-request 262144"
)
_NF_EXTRA_D = (
    '--max-total-tokens 262144 --json-model-override-args "{\\"language_model_only\\":true}" '
    "--kv-cache-dtype fp8_e4m3 --page-size 64 --max-running-requests 1 --mamba-ssm-dtype bfloat16 "
    "--ple-offload-embedding --ple-offload-backend checkpoint --hicache-mamba-host-mib 300 "
    "--rank-role host,worker,worker --rank-tp-ratio 1,0,0 --rank-moe-ratio 183,137,168 "
    "--rank-moe-resident-fraction 0.006,0.564,0.467 --rank-user-reserve-mib 0,0,0 "
    "--speculative-draft-placement solo --speculative-algorithm NEXTN --speculative-num-steps 3 "
    "--speculative-eagle-topk 1 --speculative-num-draft-tokens 4 "
    f"--speculative-draft-model-path {_NF_DRAFT} --cuda-graph-bs-decode 1 2 "
    "--cuda-graph-backend-decode=full --cuda-graph-backend-prefill=disabled --hicache-size 4 "
    "--hicache-mamba-host-mib 300 --context-length 262144 --max-kv-per-request 262144"
)
_NF_ENV_COMMON = ("SGLANG_UNEVEN_MOE_EXPERT_SHARD=1;SGLANG_WEG2_SEQ_BUFFER_DEPTH=1;"
                  "SGLANG_WEIGHT_LOADER_PREAD=0;SGLANG_LOAD_KEY_WORKERS=4;")

#: NF-MTP: arm_fnFL2.sh LARGS with every ${VAR:-default} at its default.
CASE_NF = FormCase(
    name="NF-MTP",
    expect={"arch": "moe", "experts": "offload", "draft": "mtp", "p_draft": "compute",
            "kv": "qsa_forma", "flip": "family", "vision": "off"},
    tag="fnFL2FMG",
    args=(
        "--corridor-budget-sample", "/spinning/gpu-arb/weg2/corridor_budget_sample_nextflash_0921.json",
        "--profile", "nextflash", "--model", _NF_MODEL, "--flip-weights", "family",
        "--weg2-weight-source", "exchange", "--weg2-xchg-inject", "authoritative",
        "--xchg-bounce-depth", "1", "--weg2-weights-cpu-backup", "off", "--weight-chunks", "16",
        "--weg2-xchg-census", "/spinning/gpu-arb/weg2/census/xchg_census_fnFL2_computed.json",
        "--env-p", "SGLANG_MOE_SCRATCH_SLOTS=32;SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.35,0.6,0.6;"
                   + _NF_ENV_COMMON + "SGLANG_MOE_OFFLOAD_WAVE_ORDER=expert;"
                   "SGLANG_QWEN4_PLE_CKPT_GATHER=pread;SGLANG_MOE_EXPERT_STORE_DIR=/mnt/nf-experts/fnFL2",
        "--env-d", "SGLANG_MOE_SCRATCH_SLOTS=44,48,48;SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.006,0.564,0.467;"
                   + _NF_ENV_COMMON + "SGLANG_MOE_OFFLOAD_GRAPH_MODE=pool;"
                   "SGLANG_MOE_EXPERT_STORE_DIR=/mnt/nf-experts/fnFL2",
        "--d-foreign-context-mib", "1446,896,894", "--d-nontorch-mib", "1981,528,524",
        "--d-tp-objective", "maxkv", "--draft-kv-on-p", "on", "--weg2-vision", "off",
        "--pp-solve-pool-floor", "0", "--pp-stage-ratio", "29,11,8", "--pp-attn-stage-ratio", "7,3,2",
        "--pp-cut-expert-device-fraction", "0.35,0.6,0.6", "--pp-cut-expert-lru-rows", "32,32,32",
        "--user-reserve-mib", "0,0,0", "--store-max-gb", "150", "--max-kv-per-request", "262144",
        "--p-bs", "1", "--d-bs", "1", "--extra-p", _NF_EXTRA_P, "--extra-d", _NF_EXTRA_D,
    ),
    env={
        "SGLANG_WEG2_WAKE_DEFER_BELOW_MIB": "0", "SGLANG_MOE_EXPERT_STORE_SLOT_FRACTION": "0.82",
        "SGLANG_MOE_COLD_TIER_SHM": "1", "SGLANG_MOE_COLD_TIER_INSTANCE": "fnFL2FMG",
        "MALLOC_ARENA_MAX": "2", "SGLANG_MOE_OFFLOAD_EXCLUDE_DRAFT": "1",
        "SGLANG_MOE_OFFLOAD_WAVE_SLICE": "40960", "SGLANG_NAN_GUARD": "1", "SGLANG_HC_MIXER_INT8": "1",
        "SGLANG_LOAD_PROFILE": "1", "SGLANG_MOE_MARLIN_ATOMIC_ADD": "0", "SGLANG_MOE_POOL_STAGING": "8",
        "SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK": "1",
    },
    # the NF dry run writes expert_map_<tag>.json into --evidence-dir: a
    # symlink farm keeps that write out of the live evidence directory.
    evidence_farm=True,
    strict_ref=False,
)

CASES = {c.name: c for c in (CASE_27B, CASE_NF)}


# --------------------------------------------------------------------------
# log parsing
# --------------------------------------------------------------------------

_ARGV_RE = re.compile(r"WEG2-LAUNCH group (?P<g>[PD]) argv: (?P<argv>.*)$")
_ENV_RE = re.compile(r"WEG2-LAUNCH WEG2-GROUP-ENV (?P<g>[PD]): (?P<env>.*)$")
_FRONT_LOG_TAG_RE = re.compile(r"^boot_weg2_(?P<tag>.+)_[0-9a-f]{7,40}_\d{4}_\d{6}\.front\.log$")


@dataclass
class BootLines:
    path: str
    argv: Dict[str, List[str]] = field(default_factory=dict)
    env: Dict[str, Dict[str, str]] = field(default_factory=dict)
    form: Optional[weg2_form.Weg2Form] = None
    tag: str = ""


def parse_boot_lines(path: str) -> BootLines:
    out = BootLines(path=path)
    m = _FRONT_LOG_TAG_RE.match(os.path.basename(path))
    out.tag = m.group("tag") if m else ""
    ident = weg2_form.log_identity(path)
    out.form = ident.form
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = _ARGV_RE.search(line)
            if m:
                out.argv[m.group("g")] = shlex.split(m.group("argv"))
                continue
            m = _ENV_RE.search(line)
            if m:
                body = m.group("env")
                env: Dict[str, str] = {}
                if body != "(leer)":
                    for item in body.split(";"):
                        if "=" in item:
                            k, v = item.split("=", 1)
                            env[k] = v
                out.env[m.group("g")] = env
    return out


def flag_map(argv: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
    """``--flag value`` / ``--flag=value`` / bare ``--flag`` -> flag: values."""
    out: Dict[str, List[str]] = {}
    i, argv = 0, list(argv)
    while i < len(argv):
        t = argv[i]
        if t.startswith("--"):
            if "=" in t:
                k, v = t.split("=", 1)
                out.setdefault(k, []).append(v)
                i += 1
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                vals = [argv[i + 1]]
                j = i + 2
                while j < len(argv) and not argv[j].startswith("--"):  # nargs='+'
                    vals.append(argv[j])
                    j += 1
                out.setdefault(t, []).append(" ".join(vals))
                i = j
            else:
                out.setdefault(t, []).append("")
                i += 1
        else:
            i += 1
    return {k: tuple(v) for k, v in out.items()}


@dataclass(frozen=True)
class DiffRow:
    where: str
    key: str
    ref: str
    new: str
    klass: str  # SOLVED | PER-BOOT | FORM | EXPLAINED | UNEXPLAINED
    why: str = ""

    def line(self) -> str:
        return (f"  [{self.klass}] {self.where} {self.key}: {self.ref!r} -> {self.new!r}"
                + (f" -- {self.why}" if self.why else ""))


def _same_modulo(a: str, b: str, aliases: Sequence[Tuple[str, str]]) -> bool:
    for x, y in aliases:
        if x:
            a = a.replace(x, y)
    return a == b


#: The calibration SOURCES a dry run names: when two runs solved different
#: numbers, these say whether a source moved (a CALIBRATION difference) or the
#: solve itself did (a code difference).
_CALIB_SOURCE_RES = (
    ("#1444 D residue record", re.compile(r"#1444 DC-RESIDUE group=D source=(\w+): (?:boot (\S+))?")),
    ("P-cut design prefix log", re.compile(r"design_prefix=\d+ tokens \((?:MEASURED[^/]*?(/\S+\.P\.log)|(\w+))")),
    ("host-ledger run sample", re.compile(r"MEASURED run-moment residual of boot (\S+)")),
)


def calibration_sources(log_path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(log_path, errors="replace") as f:
        for line in f:
            for name, rx in _CALIB_SOURCE_RES:
                if name in out:
                    continue
                m = rx.search(line)
                if m:
                    out[name] = " ".join(g for g in m.groups() if g)
    return out


def classify_argv(where: str, ref: Sequence[str], new: Sequence[str],
                  explain: Mapping[str, str]) -> List[DiffRow]:
    a, b = flag_map(ref), flag_map(new)
    rows: List[DiffRow] = []
    for k in sorted(set(a) | set(b)):
        if k == "--model-path" or a.get(k) == b.get(k):
            continue
        ra, rb = " ".join(a.get(k, ("<absent>",))), " ".join(b.get(k, ("<absent>",)))
        if k in explain:
            rows.append(DiffRow(where, k, ra, rb, "EXPLAINED", explain[k]))
        elif k in SOLVED_FLAGS:
            rows.append(DiffRow(where, k, ra, rb, "SOLVED", "solved per boot from measured inputs"))
        elif k in PER_BOOT_FLAGS:
            rows.append(DiffRow(where, k, ra, rb, "PER-BOOT", "one boot's identity"))
        else:
            rows.append(DiffRow(where, k, ra, rb, "UNEXPLAINED"))
    return rows


def classify_env(where: str, ref: Mapping[str, str], new: Mapping[str, str], *,
                 ref_tag: str, new_tag: str, form: Optional[weg2_form.Weg2Form],
                 explain: Mapping[str, str],
                 aliases: Sequence[Tuple[str, str]] = ()) -> List[DiffRow]:
    rows: List[DiffRow] = []
    for k in sorted(set(ref) | set(new)):
        ra, rb = ref.get(k), new.get(k)
        if ra == rb:
            continue
        sa, sb = ("<absent>" if ra is None else ra), ("<absent>" if rb is None else rb)
        if k in explain:
            rows.append(DiffRow(where, k, sa, sb, "EXPLAINED", explain[k]))
        elif k in PER_BOOT_ENV:
            rows.append(DiffRow(where, k, sa, sb, "PER-BOOT", "one boot's identity / real-boot only"))
        elif ra is not None and rb is not None and _same_modulo(
                ra, rb, ([(ref_tag, new_tag)] if ref_tag and new_tag else []) + list(aliases)):
            rows.append(DiffRow(where, k, sa, sb, "PER-BOOT", "differs only by the boot tag / work dir"))
        elif k == weg2_form.FORM_ENV and form is not None and rb == form.env_value():
            rows.append(DiffRow(where, k, sa, sb, "FORM", "the resolved WEG2-FORM of this tree"))
        elif k == P_PREFILL_TRANSIENT_ENV and rb is None and form is not None:
            rows.append(DiffRow(where, k, sa, sb, "FORM",
                                f"#114 slopes are not {form.model}'s (keyed on the checkpoint)"))
        else:
            rows.append(DiffRow(where, k, sa, sb, "UNEXPLAINED"))
    return rows


# --------------------------------------------------------------------------
# references
# --------------------------------------------------------------------------


def newest_reference(expect: Mapping[str, str], model: str, evidence_dir: str,
                     exclude_tags: Sequence[str] = ()) -> Optional[str]:
    """Newest front log of this form (checkpoint + draft; full axes when the
    boot printed a WEG2-FORM line) that reached front READY."""
    want = weg2_form.model_key(model)
    cands = []
    for n in os.listdir(evidence_dir):
        m = _FRONT_LOG_TAG_RE.match(n)
        if m and m.group("tag") not in exclude_tags:
            p = os.path.join(evidence_dir, n)
            cands.append((os.path.getmtime(p), p))
    for _, p in sorted(cands, reverse=True):
        ident = weg2_form.log_identity(p)
        if ident.model != want:
            continue
        if ident.form is not None and any(
                getattr(ident.form, a) != expect[a] for a in weg2_form.RESIDUE_AXES):
            continue
        if ident.form is None and ident.draft not in (None, expect["draft"]):
            continue
        with open(p, errors="replace") as f:
            if any("READY group=front" in line for line in f):
                return p
    return None


def tag_front_log(tag: str, evidence_dir: str) -> Optional[str]:
    hits = []
    for n in os.listdir(evidence_dir):
        m = _FRONT_LOG_TAG_RE.match(n)
        if m and m.group("tag") == tag:
            p = os.path.join(evidence_dir, n)
            hits.append((os.path.getmtime(p), p))
    return max(hits)[1] if hits else None


# --------------------------------------------------------------------------
# post-READY replay
# --------------------------------------------------------------------------

_FLOOR_RE = re.compile(r"CARRIER BOUND: .*?\bfloor=(\d+) \[")


@dataclass(frozen=True)
class GateVerdict:
    gate: str
    verdict: str  # PASS | REFUSED | SKIPPED | ERROR
    detail: str

    def line(self) -> str:
        return f"  [{self.verdict}] {self.gate}: {self.detail}"


def replay_post_ready(front_log: str) -> List[GateVerdict]:
    """THIS tree's post-READY launcher gates, in main()'s order and under
    main()'s conditions, against one boot's own P/D logs."""
    _tree_python = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    if _tree_python not in sys.path:
        sys.path.insert(0, _tree_python)
    from sglang.srt.weg2 import carrier_census as cc
    from sglang.srt.weg2 import launcher as L
    from sglang.srt.weg2 import ring_table

    boot = parse_boot_lines(front_log)
    p_log = front_log[: -len(".front.log")] + ".P.log"
    d_log = front_log[: -len(".front.log")] + ".D.log"
    argv_p, argv_d = boot.argv.get("P", []), boot.argv.get("D", [])
    out: List[GateVerdict] = []
    if not (os.path.exists(p_log) and os.path.exists(d_log) and argv_p and argv_d):
        return [GateVerdict("replay", "SKIPPED", f"{front_log}: P/D log or argv line missing")]
    hicache_disabled = "--enable-hierarchical-cache" not in argv_p

    def run(name, fn):
        lines: List[str] = []
        try:
            res = fn(lines.append)
            detail = "; ".join(lines[-2:]) if lines else str(res)
            if isinstance(res, str) and res.startswith("SKIPPED"):
                out.append(GateVerdict(name, "SKIPPED", res))
            else:
                out.append(GateVerdict(name, "PASS", detail[:600]))
        except L.REFUSALS as e:  # noqa: PERF203
            out.append(GateVerdict(name, "REFUSED", str(e)[:600]))
        except (Exception, SystemExit) as e:  # a gate that cannot run is not a pass
            out.append(GateVerdict(name, "ERROR", f"{type(e).__name__}: {e}"[:600]))

    def w7_p(log):
        if hicache_disabled:
            return "SKIPPED (--weg2-disable-hicache)"
        n_kv = L.count_marker(p_log, "#706 canonical KV page active")
        n_blob = L.count_marker(p_log, "canonical GDN blob active")
        log(f"P kv x{n_kv} blob x{n_blob}")
        if n_kv < 3 or n_blob < 3:
            raise L.Weg2LaunchRefused(f"W7/W10 (launcher half): P logged kv x{n_kv} blob x{n_blob}, need 3 each")

    def w7_d(log):
        if hicache_disabled:
            return "SKIPPED (--weg2-disable-hicache)"
        n_kv, n_blob, n_worker = L.canonical_marker_counts(d_log)
        log(f"D kv x{n_kv} blob x{n_blob} (Form A workers {n_worker})")
        if n_kv < 3 or n_blob < 3:
            raise L.Weg2LaunchRefused(f"W7/W10 (launcher half): D logged kv x{n_kv} blob x{n_blob}, need 3 each")

    def w9(log):
        if hicache_disabled:
            return "SKIPPED (--weg2-disable-hicache)"
        forced = "#1233 HICACHE BIGRAM KEYS FORCED"
        pb = ("--speculative-algorithm" in argv_p) or L.count_marker(p_log, forced) >= 1
        db = ("--speculative-algorithm" in argv_d) or L.count_marker(d_log, forced) >= 1
        log(f"P bigram={pb} D bigram={db}")
        if pb != db:
            raise L.Weg2LaunchRefused(f"W9 Weg2StoreIdentityMismatch: P bigram={pb} D bigram={db}")

    def w45(log):
        if hicache_disabled:
            return "SKIPPED (--weg2-disable-hicache)"
        floor = None
        with open(front_log, errors="replace") as f:
            for line in f:
                m = _FLOOR_RE.search(line)
                if m:
                    floor = int(m.group(1))
                    break
        if floor is None:
            return "SKIPPED (the reference front log names no CARRIER BOUND floor)"
        cen = cc.census(d_log, expected_ranks=cc.tp_size_of(argv_d), floor=floor,
                        abstain_ranks=cc.form_a_worker_ranks(d_log))
        dec = cc.decide_bound(cen, None, log_path=d_log, floor_why="replayed from the reference")
        log(f"verdict={cen.verdict} bound={dec.bound} ({dec.source})")
        if dec.refused:
            raise L.Weg2LaunchRefused(dec.detail)

    def spec_of_reference():
        fm = flag_map(argv_d)
        algo = (fm.get("--speculative-algorithm") or ("NEXTN",))[0]
        ns = argparse.Namespace(
            spec_form="DFLASH" if algo == "DFLASH" else "NEXTN",
            dflash_draft_path=(fm.get("--speculative-draft-model-path") or (L.DFLASH_DRAFT_PATH_DEFAULT,))[0],
            dflash_block=int((fm.get("--speculative-num-draft-tokens") or (L.DFLASH_BLOCK_DEFAULT,))[0]),
            dflash_window=int((fm.get("--speculative-draft-window-size") or (L.DFLASH_WINDOW_DEFAULT,))[0]),
            dflash_produce_on_p="on" if boot.env.get("P", {}).get(L.DFLASH_PRODUCE_ENV) == "1" else "off",
        )
        L.apply_spec_form(ns)
        return ns

    def w10(log):
        if hicache_disabled:
            return "SKIPPED (--weg2-disable-hicache)"
        if not ring_table.p_carries_drafter(argv_p):
            return "SKIPPED (--draft-kv-on-p off: P carries no drafter)"
        spec_of_reference()
        return L.gate_w10(p_log, d_log, log, p_produces_draft_pages=L.p_produces_draft_pages())

    def w11(log):
        if hicache_disabled:
            return "SKIPPED (--weg2-disable-hicache)"
        if not ring_table.p_carries_drafter(argv_p):
            return "SKIPPED (--draft-kv-on-p off: P carries no drafter)"
        spec_of_reference()
        return L.gate_w11(p_log, log)

    run("W7/W10 launcher half P", w7_p)
    run("W7/W10 launcher half D", w7_d)
    run("W9 key scheme", w9)
    run("W45 carrier census", w45)
    run("W10 drafter identity", w10)
    run("W11/W11b draft resident", w11)
    return out


# --------------------------------------------------------------------------
# the dry run
# --------------------------------------------------------------------------


def _evidence_farm(src: str, dst: str) -> str:
    os.makedirs(dst, exist_ok=True)
    for n in os.listdir(src):
        t = os.path.join(dst, n)
        if not os.path.lexists(t):
            os.symlink(os.path.join(src, n), t)
    return dst


def run_dry(case: FormCase, tree: str, work_dir: str, launcher_args: Optional[Sequence[str]],
            timeout_s: int = 900, reuse: bool = False) -> Tuple[int, str]:
    log_path = os.path.join(work_dir, f"dry_{case.name}.log")
    if reuse and os.path.exists(log_path):
        with open(log_path, errors="replace") as f:
            if "DRY-RUN complete" in f.read():
                return 0, log_path
    args = list(launcher_args) if launcher_args else list(case.args)
    argv = [VENV_PY, "-m", "sglang.srt.weg2.launcher", "--tree", tree, "--tag", case.tag] + args
    if case.evidence_farm and "--evidence-dir" not in args:
        argv += ["--evidence-dir", _evidence_farm(EVIDENCE_DIR, os.path.join(work_dir, "evidence_farm"))]
    argv.append("--dry-run")
    env = dict(os.environ)
    env.update(case.env)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.path.join(tree, "python")
    with open(log_path, "w") as fh:
        fh.write("# " + " ".join(shlex.quote(a) for a in argv) + "\n")
        fh.flush()
        try:
            rc = subprocess.run(argv, cwd=tree, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=timeout_s).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    return rc, log_path


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


def gate_form(case: FormCase, tree: str, work_dir: str, *, launcher_args=None,
              reference: Optional[str] = None, good_ref: Optional[str] = None,
              explain: Mapping[str, str] = {}, strict_ref: bool = True,
              emit=print, dry: Optional[Tuple[int, str, float]] = None,
              base_dry: Optional[Tuple[int, str, float]] = None, base_tree: str = "") -> bool:
    ok = True
    emit(f"== FORM {case.name} (tree {tree})")
    if dry is None:
        t0 = time.time()
        rc, dry_log = run_dry(case, tree, work_dir, launcher_args)
        dry = (rc, dry_log, time.time() - t0)
    rc, dry_log, secs = dry
    body = open(dry_log, errors="replace").read()
    emit(f"  dry run rc={rc} in {secs:.0f} s -> {dry_log}")
    if rc != 0 or "DRY-RUN complete" not in body:
        refused = re.findall(r"WEG2-LAUNCH REFUSED: .*", body)
        emit(f"  [FAIL] dry run did not complete: {refused[-1][:400] if refused else body[-400:]}")
        return False
    new = parse_boot_lines(dry_log)
    new.tag = case.tag
    if new.form is None or new.form.axes() != case.expect:
        emit(f"  [FAIL] WEG2-FORM {new.form.describe() if new.form else '<no line>'} "
             f"!= expected {' '.join(f'{k}={v}' for k, v in case.expect.items())}")
        ok = False
    else:
        emit(f"  [PASS] {new.form.line()[:220]}")
    for marker in ("#114 P-PREFILL-TRANSIENT", "#1444 DC-RESIDUE", "PP-CUT depth axis", "SKIPPED (form"):
        for line in body.splitlines():
            if marker in line:
                emit("  | " + line.split("WEG2-LAUNCH ", 1)[-1][:260])
    if base_dry is not None:
        brc, blog, _ = base_dry
        if brc != 0:
            emit(f"  [WARN] base tree dry run rc={brc} ({blog}) -- no byte-identity diff")
        else:
            bb = parse_boot_lines(blog)
            bb.tag = case.tag
            rows: List[DiffRow] = []
            alias = [(os.path.dirname(blog), os.path.dirname(dry_log))]
            for g in ("P", "D"):
                rows += classify_argv(f"argv {g}", bb.argv.get(g, []), new.argv.get(g, []), explain)
                rows += classify_env(f"env {g}", bb.env.get(g, {}), new.env.get(g, {}), ref_tag=case.tag,
                                     new_tag=case.tag, form=new.form, explain=explain, aliases=alias)
            src_b, src_n = calibration_sources(blog), calibration_sources(dry_log)
            moved = {k: (src_b.get(k, "-"), src_n.get(k, "-")) for k in set(src_b) | set(src_n)
                     if src_b.get(k) != src_n.get(k)}
            emit(f"  argv/env vs the SAME dry run on the BASE tree {base_tree}: {len(rows)} difference(s)")
            for k, (b_, n_) in sorted(moved.items()):
                emit(f"  [CALIBRATION] {k}: base reads {b_!r}, this tree {n_!r}")
            for r in rows:
                if r.klass == "SOLVED" and moved:
                    r = DiffRow(r.where, r.key, r.ref, r.new, "CALIBRATION",
                                "solved from a calibration source this tree reads differently (above)")
                emit(r.line())
            bad = [r for r in rows if r.klass == "UNEXPLAINED" or (r.klass == "SOLVED" and not moved)]
            if bad:
                emit(f"  [FAIL] {len(bad)} difference(s) against the base tree that are neither the form's "
                     f"own nor explained by a moved calibration source")
                ok = False
    model = (flag_map(new.argv.get("P", [])).get("--model-path") or ("",))[0]
    ref = reference or newest_reference(case.expect, model, EVIDENCE_DIR)
    if ref is None:
        emit("  [WARN] no boot of this form reached front READY in the evidence dir -- no reference diff")
    else:
        rb = parse_boot_lines(ref)
        rows: List[DiffRow] = []
        for g in ("P", "D"):
            rows += classify_argv(f"argv {g}", rb.argv.get(g, []), new.argv.get(g, []), explain)
            if g in rb.env:
                rows += classify_env(f"env {g}", rb.env[g], new.env.get(g, {}), ref_tag=rb.tag,
                                     new_tag=new.tag, form=new.form, explain=explain)
        emit(f"  argv/env vs NEWEST boot of this form {os.path.basename(ref)}: {len(rows)} difference(s)")
        for r in rows:
            emit(r.line())
        bad = [r for r in rows if r.klass == "UNEXPLAINED"]
        if bad and strict_ref:
            emit(f"  [FAIL] {len(bad)} UNEXPLAINED difference(s) (--explain KEY=why to accept)")
            ok = False
        emit(f"  post-READY gates of THIS tree replayed on {os.path.basename(ref)}'s P/D logs:")
        for v in replay_post_ready(ref):
            emit(v.line())
            if v.verdict in ("REFUSED", "ERROR"):
                ok = False
    good = good_ref if good_ref is not None else case.good_ref
    gpath = tag_front_log(good, EVIDENCE_DIR) if good else None
    if gpath:
        gb = parse_boot_lines(gpath)
        rows = []
        for g in ("P", "D"):
            rows += classify_argv(f"argv {g}", gb.argv.get(g, []), new.argv.get(g, []), explain)
        emit(f"  argv vs last GOOD boot {os.path.basename(gpath)} (informational): {len(rows)} difference(s)")
        for r in rows:
            emit(r.line())
    emit(f"  => {case.name}: {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--form", action="append", choices=sorted(CASES),
                    help="form(s) to gate (default: all)")
    ap.add_argument("--tree", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
    ap.add_argument("--work-dir", default="")
    ap.add_argument("--launcher-args", default="",
                    help="the arm's exact launcher argv (shell words, without --tree/--tag/--dry-run); "
                         "only with exactly one --form")
    ap.add_argument("--reference", default="", help="front log to diff/replay against (default: newest of the form)")
    ap.add_argument("--good-ref", default=None, help="tag of the last GOOD boot (27B default weg2xsn411)")
    ap.add_argument("--explain", action="append", default=[], metavar="KEY=why")
    ap.add_argument("--no-strict-ref", action="store_true",
                    help="UNEXPLAINED reference differences warn instead of failing")
    ap.add_argument("--reuse", action="store_true",
                    help="re-analyse completed dry-run logs already in --work-dir instead of re-running them")
    ap.add_argument("--base-tree", default="",
                    help="a second tree (e.g. the line before this change): the SAME dry run there, and "
                         "every argv/env difference that is not the form's own fails the gate -- the "
                         "byte-identity proof for a form this tree must not move")
    ns = ap.parse_args(argv)
    forms = ns.form or sorted(CASES)
    if ns.launcher_args and len(forms) != 1:
        ap.error("--launcher-args needs exactly one --form")
    work = ns.work_dir or tempfile.mkdtemp(prefix="form_matrix_gate_")
    os.makedirs(work, exist_ok=True)
    explain = dict(x.split("=", 1) for x in ns.explain if "=" in x)
    largs = shlex.split(ns.launcher_args) if ns.launcher_args else None
    # EVERY dry run first, while this parent is still torch-free; the replay
    # (which imports the launcher) only afterwards -- one memory peak, not two.
    drys, base_drys = {}, {}
    for name in forms:
        t0 = time.time()
        rc, path = run_dry(CASES[name], os.path.abspath(ns.tree), work, largs, reuse=ns.reuse)
        drys[name] = (rc, path, time.time() - t0)
        if ns.base_tree:
            bwork = os.path.join(work, "base")
            os.makedirs(bwork, exist_ok=True)
            t0 = time.time()
            rc, path = run_dry(CASES[name], os.path.abspath(ns.base_tree), bwork, largs, reuse=ns.reuse)
            base_drys[name] = (rc, path, time.time() - t0)
    results = {}
    for name in forms:
        results[name] = gate_form(
            CASES[name], os.path.abspath(ns.tree), work, launcher_args=largs,
            reference=ns.reference or None, good_ref=ns.good_ref, explain=explain,
            strict_ref=CASES[name].strict_ref and not ns.no_strict_ref, dry=drys[name],
            base_dry=base_drys.get(name), base_tree=os.path.abspath(ns.base_tree) if ns.base_tree else "")
    print("FORM-MATRIX-GATE " + " ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
