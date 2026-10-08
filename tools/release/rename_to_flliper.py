#!/usr/bin/env python3
"""rename_to_flliper.py -- deterministic, auditable rename "sglang -> fLLiper".

fLLiper (fast Large Language inference - Parallel Elastic Runtime) is derived
from SGLang (Apache-2.0).  This tool performs the *mechanical* part of the
rename of the fork's code tree and makes the result machine-checkable.

Sub-commands
------------
  inventory  Count every occurrence of the old names per category, read-only.
             Source is a git ref (--repo/--ref, read through `git cat-file`,
             no checkout needed) or a directory (--root).  Optional --scan
             directories (operator tools outside the tree) are classified too.
  apply      Rewrite a *throwaway worktree* in place: file contents and paths.
             --dry-run prints the diff statistics and writes nothing.
  verify     Independent check of an applied worktree against its base ref.
             It does NOT reuse the rewrite engine: it re-tokenises base and
             target and proves "byte-identical except for name tokens", checks
             the path mapping, and checks that attribution (URLs, copyright
             lines) is untouched.
  selftest   Unit checks of the rules on crafted strings.

Rules (profile "core", the phase-2a scope)
------------------------------------------
  sglang  -> flliper     package, module paths, process titles (sglang::x),
                         identifiers (bench_sglang -> bench_flliper), cache dirs
  SGLANG  -> FLLIPER     env variables SGLANG_* -> FLLIPER_* (the compat shim
                         that keeps reading SGLANG_* is authored separately)
  Sglang  -> Flliper     CamelCase identifiers
  SGLang  -> Flliper     when glued to an identifier character (SGLangError)
  SGLang  -> fLLiper     as a free word (prose, help texts, log messages)
  (same for the rare typos SGlang / sGLang)

  Never touched: anything preceded by "ht" (htsglang/HTSGLANG_* is the
  product/container layer, own phase), SGL_* (upstream legacy aliases and
  build macros), sgl_kernel / sgl-kernel / sglang-kernel (the kernel wheel,
  needs a rebuild -> own phase), sglang-router / sglang_router / sglang-grpc /
  sglang-jax (foreign packages), URLs, `org/sglang...` ids of upstream orgs,
  copyright / licence / attribution lines, C/C++/CUDA sources (unless --cxx),
  and whole directories listed in EXCLUDE (sgl-kernel, rust, proto, docs, ...).

Optional rule sets (same pass, same proof)
------------------------------------------
  --weg2        weg2 -> pdflip, WEG2 -> PDFLIP, Weg2 -> PdFlip: package
                srt/weg2 -> srt/pdflip, env SGLANG_WEG2_* -> FLLIPER_PDFLIP_*,
                WEG2_* -> PDFLIP_*, log markers WEG2-X -> PDFLIP-X, flags
                --weg2-x -> --pdflip-x, routes /weg2/x -> /pdflip/x.
                Kept: boot tags (weg2xsn25 ...: names of real evidence),
                host paths (/spinning/..., hicache-weg2, gpu-arb/weg2), image
                tags (cu130-weg2-...), operator doc refs (WEG2_*_SPEC_*).
  --ident-map   exact table {old: new} for Python NAME tokens (German ->
                English identifiers, built by english_audit.py
                ident-proposals, reviewed). Strings and comments are left to
                the checked translation pass.
  Application order per file: ident map, sglang rules, weg2 rules; the
  verifier accepts exactly the compositions of its own three tables.

Determinism: files are processed in sorted path order, no timestamps are
written, and the manifest ends with a digest over (new_path, sha256) pairs.
Two runs on the same base must print the same digest.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Tuple

TOOL_VERSION = "3"

# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------
INCLUDE = [
    "python/**", "test/**", "tests/**", "scripts/**", "benchmark/**",
    "examples/**", "bench/**", "bench216/**", "tools/**", "devtools/**",
    "weg2/**", "docker/**", "deploy/**", "clients/**", ".window/**",
    ".gitignore", ".coveragerc", ".isort.cfg", ".pre-commit-config.yaml",
    ".dockerignore", ".codespellrc",
]
# excluded for content AND path (whole foreign/separate trees)
EXCLUDE = [
    "3rdparty/**", "sgl-kernel/**", "sgl-model-gateway/**", "experimental/**",
    "rust/**", "proto/**", ".github/**", "docs/**", "docs_new/**", ".claude/**",
    ".deps/**",
]
# excluded for content only: the file keeps its bytes but moves with its directory
EXCLUDE_CONTENT = [
    "LICENSE*", "**/LICENSE*", "NOTICE*", "**/NOTICE*", "COPYING*", "**/COPYING*",
    "**/*_pb2.py", "**/*_pb2_grpc.py", "**/*.proto", "**/_vendor/**",
    # captured boot-log excerpts are EVIDENCE (RENAME_NF_INVENTORY 3.1/6-1a): the in-tree parsers read them and
    # the x162 reference logs at every boot; they keep their bytes (they move with their directory) so the
    # dual-name parsers are tested against unchanged old logs
    "test/registered/unit/weg2/fixtures/**",
    # ... and at their new place: a second pass over the renamed tree must be a no-op (FL4 26.09.: without this
    # line it rewrote 77 fixture files, 6,846 replacements, and the idempotency check caught it)
    "test/registered/unit/pdflip/fixtures/**",
    # F0-A (07.10.2026): the rename kit now lives IN the tree it renames (tools/release/**: engine, ident tables, translation
    # memory, profile conversions, scripts) and so does the dashboard rename entry (docker/weg2-release/rename_rigdash*). Their
    # content is tool input -- the ident table keys, the old-name regexes of the residue check, the translation memory --
    # and must stay byte-identical after the pass, or the kit in the renamed tree no longer reads the old tree's names.
    "tools/release/**", "docker/weg2-release/rename_rigdash*", "docker/pdflip-release/rename_rigdash*",
    # F0-A (07.10.2026): the dashboard's boot recordings carry a `vram_plan.plan_id` = sha256 over the plan body, and the body spells the
    # old names (nf-int4-abl: 15 hits, 27b-int8 / 27b-nvfp4-dual: 1 each). Renamed, `kartenplan.plan_id_ok` fails and every record is
    # shown as altered. They are evidence like the boot logs (RENAME_PLAN 2): bytes kept. rename_rigdash.py has locked them since 05.10.;
    # the whole-tree pass needs the same lock (tools/** is in scope).
    "tools/rig_dashboard/rigdash/kartenplan_data/**",
    # F0-A fix round 2 (08.10.2026): the container entrypoint spells old AND new names on purpose (state-path fallback, cache volume
    # detection, package-dir test: executable compat code). The mechanical pass would make both spellings one word (3 name-rule
    # collisions), so its content is locked here; F0-B writes it for both names, F0-G finishes it by hand on the renamed tree. The file
    # still moves with its directory (weg2-release -> pdflip-release).
    "docker/weg2-release/entrypoint.sh", "docker/pdflip-release/entrypoint.sh",
    # F0-D (08.10.2026, Planer-Sitz decision after the 27B probe 06:16Z, `ABORT: apply rc`, collision_auto REFUSED 8 files): since F0-B/F0-C these
    # files spell the OLD and the NEW name on purpose. The mechanical pass would turn both spellings into one word (name-rule collisions:
    # `sglang`|`flliper` -> `flliper`, `weg2`|`pdflip` -> `pdflip`). Their content is locked like the entrypoint above (precedent F0-A fix
    # round 2) where the file is a SHELL SCRIPT that nothing imports and whose old-name reading is a literal alternation (boot_deadman.sh:
    # `(WEG2|PDFLIP)` would become `(PDFLIP|PDFLIP)`): docker/htsglang-entrypoint.sh, scripts/weg2/devtools/boot_deadman.sh. A file that is
    # moved with its directory is listed under BOTH paths (the second pass over the renamed tree must be a no-op).
    # NOT locked, although the probe listed them (first F0-D kit run, dashboard suite 30 failed vs 3 on the old tree, tests/sources 08.10.):
    # test_f0b_tool_names_1007.py (imports `sglang.test.*` and walks `scripts/weg2/...`; locked it imports the site-packages sglang of another
    # checkout and misses the moved script), rigdash/{features,redact,sources,weg2line}.py (other renamed modules and tests use their CamelCase /
    # German identifiers: `Weg2Lines` -> `PdFlipLines`, `MODELL_VALUES` -> `MODEL_VALUES` ...; their double reading is done by names.py pair tokens, which survive the
    # rename) and rigdash/GLOSSARY_EN.md. Those six go to COLLISION_OK (data/collision_ok_1007_27b.json): the colliding words are comments,
    # docstrings and the split-token machinery, no executable old-name literal.
    "docker/htsglang-entrypoint.sh",
    "scripts/weg2/devtools/boot_deadman.sh", "scripts/pdflip/devtools/boot_deadman.sh",
]
# Files that spell BOTH names on purpose: the tests of the NF compat layer (name_compat 1a/1b, 00925f46bd) write the
# new package/subsystem word as a plain literal next to imports of the old one (`["sg" "lang", "flliper"]`,
# `OLD_L, NEW_L = "we" "g2", "pdflip"`). After the pass both are the same word, which the per-file collision check
# reports; there it is intended. Only exactly these (file, new word) pairs are exempt, and every exemption is listed
# in the apply summary (FL4 26.09.).
COLLISION_OK = {
    "test/registered/unit/weg2/test_weg2_name_compat_env_1b.py": frozenset({"flliper"}),
    "test/registered/unit/weg2/test_weg2_name_compat_readers_1a.py": frozenset({"flliper", "pdflip"}),
    # The 27B ENV-PAIR OPT test (839bf76021) loads name_compat as `["sg" "lang", "flliper"]` on purpose, like the
    # 1a/1b compat tests above -- the only collision of the unified release tree f7099c0cbd (02.10.).
    "test/registered/unit/weg2/test_weg2_env_opt_pdflip_pair_1001.py": frozenset({"flliper"}),
    # The deadman is in the tree since d54855493c (imported from gpu-arb/devtools, which FL5 made read BOTH
    # generations: `(WEG2|PDFLIP)-FLIP ...`, `(Weg2|PdFlip)WakeRefused`, `(sglang|flliper)::scheduler`). Renamed WITH
    # the tree on purpose (NOT content-locked): the image runs it against the renamed launcher, whose env
    # (WEG2_STATE_DIR -> PDFLIP_STATE_DIR, WEG2_DEADMAN_GROUP, WEG2_STATE_FILE_PY) and markers it must read; an
    # alternation of the two spellings becomes `(PDFLIP|PDFLIP)`, which matches the same lines (30.09., B1 on
    # ead403b7ab: the only collision of the whole tree).
    "scripts/weg2/devtools/boot_deadman.sh": frozenset(
        {"PDFLIP", "PdFlip", "PdFlipHostWatermarkBreached", "PdFlipWakeRefused", "flliper", "pdflip"}),
}
# Item 600 (03.10.): extra exemptions from a JSON file {path: [word, ...]} (env COLLISION_OK_FILE), for word-level clashes
# that a full-token check proved cosmetic (y8t: 22 files carry the ticket label `PDFLIP-A..X` in comments/test names next to
# `WEG2-*` markers; no mapped token equals a different existing token). Merged into COLLISION_OK.
if os.environ.get("COLLISION_OK_FILE"):
    import json as _json600
    for _p, _ws in _json600.load(open(os.environ["COLLISION_OK_FILE"])).items():
        COLLISION_OK[_p] = frozenset(COLLISION_OK.get(_p, frozenset())) | frozenset(_ws)
# F0-A (07.10.2026): reviewed ident-map collisions. ident_collisions() is file-wide: an ident-map target that is any NAME of the file
# aborts the pass, also where the two names live in DIFFERENT functions and never meet (launcher.py: `gefunden` -> `hit`, `hit` is a
# local of another function). IDENT_COLLISION_OK_FILE = JSON {path: [old word, ...]} exempts exactly those (path, old word) pairs;
# every use is listed in the apply summary ("ident_collisions_allowed"). Names that DO meet in one scope (`modell` -> `model` in
# one function) must not be listed: rename one side in the source first. verify is unaffected (the map itself is unchanged).
IDENT_COLLISION_OK: Dict[str, frozenset] = {}
if os.environ.get("IDENT_COLLISION_OK_FILE"):
    import json as _json_f0a
    for _p, _ws in _json_f0a.load(open(os.environ["IDENT_COLLISION_OK_FILE"])).items():
        if _p != "_comment":
            IDENT_COLLISION_OK[_p] = frozenset(_ws)
CXX_EXT = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp", ".inl", ".metal"}

# --------------------------------------------------------------------------
# Rewrite engine
# --------------------------------------------------------------------------
MAIN = re.compile(r"(?<![Hh][Tt])(sglang|SGLANG|SGLang|Sglang|SGlang|sGLang)")
_W = re.compile(r"\w")

# spans inside which nothing is replaced
DENY_SPAN = [
    ("url", re.compile(r"(?:https?|ssh|git|file|ftp)://[^\s'\"<>()\[\]{}`]+")),
    ("git-ssh", re.compile(r"git@[\w.-]+:[\w./-]+")),
    ("github-path", re.compile(r"github\.com[/:][\w./-]+")),
    ("upstream-org-id", re.compile(
        r"\b(?:lmsys|lmsysorg|sgl-project|sgl-workspace|ascend|intel|kernels-community|"
        r"mooncake|huggingface|nvcr\.io/nvidia|rocm)/[\w.:@+-]+")),
    ("foreign-package", re.compile(
        r"\bsglang[-_](?:router|kernel|grpc|jax|diffusion-benchmark)[\w.-]*")),
    # generated protobuf modules (foreign package smg_grpc_proto) and their attributes
    ("foreign-generated", re.compile(r"\b\w*_pb2(?:_grpc)?\b(?:\.\w+)*")),
    # foreign smg-* packages, their module paths and extras (smg-grpc-servicer[sglang])
    ("foreign-package", re.compile(r"\bsmg[\w-]*(?:\.\w+)*(?:\[\w+\])?")),
    # gRPC wire names defined by the foreign .proto (service full names, service keys)
    ("grpc-wire-name", re.compile(r"\bsglang\.grpc\.[\w.]+|\bSglang(?:Encoder|Scheduler)\w*")),
    # versioned payload schema ids written into JSON records (FL4 26.09.): a format id is data, not a name
    ("persisted-id", re.compile(r"\bsglang\.(?:expert_stats|forward_peak)/\d+")),
]
# whole lines that carry attribution / licence text
DENY_LINE = re.compile(
    r"Copyright|SGLang Team|Licensed under|SPDX-License|derived from SGLang|"
    r"[Aa]dapted from|[Mm]odified from|[Pp]orted from|[Bb]ased on .*(?:sglang|SGLang)|"
    r"upstream sglang|sglang upstream|upstream SGLang|SGLang upstream")


def _replacement(word: str, text: str, start: int, end: int) -> str:
    if word == "sglang":
        return "flliper"
    if word == "SGLANG":
        return "FLLIPER"
    if word == "Sglang":
        return "Flliper"
    # SGLang / SGlang / sGLang: glued to an identifier -> CamelCase, else brand
    glued = (start > 0 and _W.match(text[start - 1])) or (end < len(text) and _W.match(text[end]))
    return "Flliper" if glued else "fLLiper"


# ---- METRIC-NAME MUST-KEEP (F0-M, 08.10.2026, user decision "pdflip as the name, the metrics are must-keep") ---------------------
# The SERIES and LABEL names that leave the process (Prometheus exposition of the servers and of the front, the Influx points
# pushed to VictoriaMetrics, the sampler's /api/v1/import/prometheus lines, the Grafana panels) keep the spelling they had:
# weg2_* (P/D-flip subsystem) and sglang:* / sglang_* (engine).  Reason: the time series in VictoriaMetrics (192.168.0.88:8428) and
# the panels of Grafana (rig-verlauf) must not break.  The table is DATA (data/metric_names_1008.json, made by
# `metric_inventory.py keepfile` from the scan of the old trees); this block only turns it into deny spans:
#   sglang_colon / sglang_colon_patterns  `sglang:<name>` in ANY file (the colon form is unambiguous; docker tags and org ids are
#                                         caught first by DENY_SPAN, and only DEFINED metric names are in the table)
#   weg2_global / weg2_global_prefixes    exact `weg2_<name>` tokens and `weg2_front_`-style prefixes that occur only as metric names in the old trees
#   files                                 per OLD path: a prefix entry (ends in `_` or `:`: `weg2_`, `sglang_`) keeps every token starting with it in that
#                                         file (the writer / panel files), an exact entry keeps that token (`weg2_group` is also a
#                                         server_args attribute elsewhere, `weg2_d_parked` a request key)
# Entries are spelled OLD; the file keys are registered under the old AND the renamed path, so the second pass over the renamed tree
# is a no-op.  METRIC_KEEP_FILE="" switches the rule off (the pre-F0-M behaviour, used by the byte-identity proofs of earlier runs).
# a metric name may follow an ESCAPED newline / tab inside a source string ("...10\\nsglang:generation_tokens_total 20"): the `n` is no identifier char
_MK_B = r"(?:(?<![A-Za-z0-9_])|(?<=\\[nrt]))"
_MK_E = r"(?![A-Za-z0-9_])"
_MK_HIST = r"(?:_bucket|_sum|_count|_created|_total)?"
_MK_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "metric_names_1008.json")
METRIC_KEEP_FILE = os.environ.get("METRIC_KEEP_FILE", _MK_DEFAULT)


def _mk_exact(names: Iterable[str]) -> Optional["re.Pattern"]:
    names = sorted(set(names), key=lambda n: (-len(n), n))
    return re.compile(_MK_B + "(?:" + "|".join(re.escape(n) for n in names) + ")" + _MK_HIST + _MK_E) if names else None


def _load_metric_keep(path: str):
    """-> (global spans [(name, regex)], per-path spans {path: [regex]}); both empty without a table."""
    if not path or not os.path.isfile(path):
        return [], {}
    import json as _jmk
    with open(path) as f:
        d = _jmk.load(f)
    glob = []
    colon = sorted(set(d.get("sglang_colon", [])), key=lambda n: (-len(n), n))
    pats = [re.escape(n) for n in d.get("sglang_colon_patterns", [])]   # prose wildcards (`sglang:spill_tier_*_bytes`) are literal text
    alts = [re.escape(n) for n in colon] + pats
    if alts:
        glob.append(("metric-name", re.compile(_MK_B + "sglang:(?:" + "|".join(alts) + ")" + _MK_HIST + _MK_E)))
    rx = _mk_exact(d.get("weg2_global", []))
    if rx is not None:
        glob.append(("metric-name", rx))
    gp = d.get("weg2_global_prefixes", [])
    if gp:
        glob.append(("metric-name", re.compile(_MK_B + "(?:" + "|".join(re.escape(e) for e in gp) + ")")))
    files: Dict[str, list] = {}
    for fp, entries in d.get("files", {}).items():
        spans = []
        prefixes = [e for e in entries if e.endswith(("_", ":"))]
        if prefixes:
            spans.append(re.compile(_MK_B + "(?:" + "|".join(re.escape(e) for e in prefixes) + ")"))
        ex = _mk_exact([e for e in entries if not e.endswith(("_", ":"))])
        if ex is not None:
            spans.append(ex)
        for key in {fp, rewrite_path(fp, True)}:
            files.setdefault(key, []).extend(spans)
    return glob, files


_MK: Optional[tuple] = None   # loaded on first use (rewrite_path is defined further down)


def metric_keep_spans(text: str, path: Optional[str]) -> List[Tuple[int, int, str]]:
    global _MK
    if _MK is None:
        _MK = ([], {})   # the loader renames the table's own file keys (rewrite_path): no table yet while it does
        _MK = _load_metric_keep(METRIC_KEEP_FILE)
    glob, files = _MK
    iv = [(m.start(), m.end(), n) for n, rx in glob for m in rx.finditer(text)]
    for rx in files.get(path or "", ()):
        iv.extend((m.start(), m.end(), "metric-name") for m in rx.finditer(text))
    return iv


def deny_intervals(text: str, path: Optional[str] = None) -> List[Tuple[int, int, str]]:
    iv: List[Tuple[int, int, str]] = []
    for name, rx in DENY_SPAN:
        for m in rx.finditer(text):
            iv.append((m.start(), m.end(), name))
    iv.extend(metric_keep_spans(text, path))
    pos = 0
    for line in text.splitlines(keepends=True):
        if DENY_LINE.search(line):
            iv.append((pos, pos + len(line), "attribution-line"))
        pos += len(line)
    iv.sort()
    merged: List[Tuple[int, int, str]] = []
    for a, b, n in iv:  # union of overlapping spans -> disjoint, sorted
        if merged and a <= merged[-1][1]:
            pa, pb, pn = merged[-1]
            merged[-1] = (pa, max(pb, b), pn)
        else:
            merged.append((a, b, n))
    return merged


def _denied(iv: List[Tuple[int, int, str]], starts: List[int], s: int, e: int) -> Optional[str]:
    """iv is disjoint and sorted (deny_intervals); exact overlap test by bisection."""
    i = bisect.bisect_right(starts, s) - 1
    for k in (i, i + 1):
        if 0 <= k < len(iv):
            a, b, n = iv[k]
            if a < e and b > s:
                return n
    return None


# ---- rule set 2: weg2 -> pdflip (enabled with --weg2) --------------------------
# `Weg2Flip*` exception classes: `Weg2` -> `PdFlip` alone would stutter (`PdFlipFlipKvRelayInfeasible`, 9 classes,
# 274 occurrences, RENAME_NF_INVENTORY 4); the longest alternative wins, in names, strings and paths alike.
W_MAIN = re.compile(r"(Weg2Flip|weg2|WEG2|Weg2)")
W_DENY_SPAN = [
    ("url", DENY_SPAN[0][1]),
    # weg2xsn25, weg2rc2f: evidence names. Preceded by a non-ALNUM (an underscore counts as a separator), so the
    # tag inside `boot_weg2_weg2rg6_...log` / `BOOT_weg2pp2_0907.md` stays too (FL2 26.09.: `(?<![\w])` renamed it)
    # FL4 26.09.: ANY tag shape, not only with a digit -- a tag PREFIX (`startswith("weg2xsn")` against the real
    # `weg2xsn437`), synthetic test tags (`weg2seamtest`, `weg2shadowC`, `weg2ls`) and tmp prefixes (`weg2ring-`)
    # are tag names too; renamed, the prefix no longer matched the kept tags (test_weg2_dc_residue_capture_bs_rc1).
    # The subsystem word itself is never glued to lower-case letters (`weg2_x`, `weg2.x`, `Weg2X`, `WEG2-X`).
    ("boot-tag", re.compile(r"(?<![A-Za-z0-9])weg2[a-z]\w*")),
    # ... and the operator dir behind an INTERPOLATED root: f"{GPU_ARB}/weg2/boot_{tag}.json", CALIB_DIR,
    # admin_key, ${..:-/spinning/gpu-arb}/weg2/tms, os.makedirs(f"{GPU_ARB}/weg2") (7 sites; FL2 26.09.: renamed they point at a
    # /spinning/gpu-arb/pdflip/ that no operator tool reads)
    ("host-path", re.compile(r"/spinning/[\w./-]*|hicache-weg2[\w-]*|gpu-arb/weg2[\w./-]*|\}/weg2(?=[/'\"])")),
    ("image-tag", re.compile(r"cu1\d\d-weg2[\w.-]*")),
    # the boot-log file prefix is evidence naming like the tag: every existing log (x162 wake reference, the
    # evidence-665-f1 tree, fixtures) is boot_weg2_<tag>_...; renamed, tests that open real evidence SKIP
    # silently ("evidence tree absent", 9 in test_weg2_corridor_instrument_0908 alone, FL2 probe 26.09.)
    ("evidence-prefix", re.compile(r"(?<![A-Za-z0-9])boot_weg2(?=_)")),
    # the other evidence FILE names the tree writes and operator tools read (RENAME_PLAN 8.12-1, FL4 26.09.):
    # the launcher's per-boot `memts_weg2_<tag>.csv` / `preflight_weg2_<tag>.log` (docker/entrypoint.sh collects
    # both by name) and the append-only `weg2_measured_record.json` in the evidence dir (host_ledger
    # MEASURED_RECORD_NAME; launcher and ring_table read it, host_acceptance/prepare_context copy it by name).
    # Renamed, a new launcher reads an EMPTY measured record and the entrypoint collects nothing.
    ("evidence-prefix", re.compile(r"(?<![A-Za-z0-9])(?:memts|preflight)_weg2(?=_)")),
    ("evidence-file", re.compile(r"(?<![A-Za-z0-9])weg2_measured_record\b")),
    ("doc-ref", re.compile(r"\bWEG2_[A-Z0-9_]*(?:SPEC|DECISIONS|PLAN|DESIGN)[\w-]*")),
    # fixed-width wire constants: the exchange region's 8-byte header magic (`int.from_bytes(b"WEG2XCHG")`, packed
    # as one `Q`). Renamed to 10 bytes it overflows the header word: 27 tests of test_weg2_xchg_region_1273 died
    # with struct.error in the FL4 probe. A binary format keeps its bytes.
    ("wire-magic", re.compile(r"\bWEG2XCHG\b")),
    # identities that are HASHED or PERSISTED and compared by equality against what earlier boots wrote (FL4 probe,
    # launcher dry-runs nf/27b): the synthetic P-form flag `--weg2-xchg-region=armed` goes into ring_table.p_form_key
    # (renamed: every recorded P form key of the evidence tree stops matching -> the ring prices from nothing);
    # the calibration schema `weg2-pp-calib/1` (host_ledger CALIB_SCHEMA, 4 files under gpu-arb/weg2/calib, read
    # with `!=` -> refused) and `weg2-lane-coverage-1` (written into lane records)
    ("persisted-id", re.compile(r"--weg2-xchg-region\b|\bweg2-pp-calib/\d+|\bweg2-lane-coverage-\d+")),
    #
    # F0-D (08.10.2026, found by comparing the launcher dry-run of the old and the renamed tree, profile dual): three more identities
    # that are HASHED or checked by equality against persisted files.  (1) `weg2-footprint/1` (form.FOOTPRINT_SCHEMA) is part of the
    # JSON blob that is hashed into `footprint_key(model)`: renamed, every footprint key changes (dual dry-run: 1bf723bf1eeb ->
    # fc0c36830a6f for the NVFP4 checkpoint) and no longer equals `form.REFERENCE_FOOTPRINTS` (pinned constants of the INT8 reference
    # model, 28e1c5c3...) or `host_ledger.NF_FOOTPRINT`: the reference model would be judged FOREIGN and the host ledger would price its
    # ratchet differently.  (2) `weg2-x-curves/1` (x_curves.X_CURVES_FORMAT) is compared with `!=` against the `format` field of the curve
    # files (`profiles_release/27b.xcurves.json` carries it).  (3) `weg2.form_measures/2|3` (form_measures.SCHEMA_V2/V3) is checked with
    # `==` when a measure document is read.  The runtime has no reader for the renamed spelling and R1 forbids giving it one, so the
    # format id stays as written: "a format id is data, not a name".
    ("persisted-id", re.compile(r"\bweg2-footprint/\d+|\bweg2-x-curves/\d+|\bweg2\.form_measures/\d+")),
]


def _w_replacement(word: str, text: str, start: int, end: int) -> str:
    if word == "Weg2Flip":
        return "PdFlip"
    if word == "weg2":
        return "pdflip"
    if word == "WEG2":
        return "PDFLIP"
    return "PdFlip"  # always CapWords: a free-standing `Weg2` may be a Python name, never merge it with `pdflip`


def _intervals(text: str, spans, extra=()) -> List[Tuple[int, int, str]]:
    iv = sorted([(m.start(), m.end(), n) for n, rx in spans for m in rx.finditer(text)] + list(extra))
    merged: List[Tuple[int, int, str]] = []
    for a, b, n in iv:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b), merged[-1][2])
        else:
            merged.append((a, b, n))
    return merged


def rewrite_weg2(text: str, path: Optional[str] = None) -> Tuple[str, collections.Counter, collections.Counter]:
    iv = _intervals(text, W_DENY_SPAN, metric_keep_spans(text, path))
    starts = [a for a, _, _ in iv]
    out, last = [], 0
    rep: collections.Counter = collections.Counter()
    skip: collections.Counter = collections.Counter()
    for m in W_MAIN.finditer(text):
        s, e = m.span()
        why = _denied(iv, starts, s, e)
        if why:
            skip["weg2:" + why] += 1
            continue
        new = _w_replacement(m.group(1), text, s, e)
        out.append(text[last:s])
        out.append(new)
        last = e
        rep[(m.group(1), new)] += 1
    out.append(text[last:])
    return "".join(out), rep, skip


def rewrite_idents(text: str, imap: Dict[str, str]) -> Tuple[str, collections.Counter]:
    """Rename Python NAME tokens by an exact table (German -> English). Strings and
    comments are left alone: they belong to the checked translation pass."""
    import io as _io
    import tokenize as _tk
    rep: collections.Counter = collections.Counter()
    if not imap:
        return text, rep
    lines = text.splitlines(keepends=True)
    offs = [0]
    for l in lines:
        offs.append(offs[-1] + len(l))
    edits = []
    try:
        for t in _tk.generate_tokens(_io.StringIO(text).readline):
            if t.type == _tk.NAME and t.string in imap:
                s = offs[t.start[0] - 1] + t.start[1]
                edits.append((s, s + len(t.string), imap[t.string]))
    except (_tk.TokenError, IndentationError, SyntaxError):
        return text, rep
    for s, e, new in reversed(edits):
        rep[("ident", "")] += 1
        text = text[:s] + new + text[e:]
    return text, rep


def rewrite_all(text: str, is_py: bool, weg2: bool, imap: Dict[str, str], path: Optional[str] = None):
    rep: collections.Counter = collections.Counter()
    skip: collections.Counter = collections.Counter()
    if is_py and imap:
        text, r = rewrite_idents(text, imap)
        rep.update(r)
    text, r, k = rewrite_text(text, path)
    rep.update(r)
    skip.update(k)
    if weg2:
        text, r, k = rewrite_weg2(text, path)
        rep.update(r)
        skip.update(k)
    return text, rep, skip


def rewrite_text(text: str, path: Optional[str] = None) -> Tuple[str, collections.Counter, collections.Counter]:
    """Return (new_text, replaced_counter[(old,new)], skipped_counter[reason])."""
    iv = deny_intervals(text, path)
    starts = [a for a, _, _ in iv]
    out: List[str] = []
    last = 0
    rep: collections.Counter = collections.Counter()
    skip: collections.Counter = collections.Counter()
    for m in MAIN.finditer(text):
        s, e = m.span()
        why = _denied(iv, starts, s, e)
        if why:
            skip[why] += 1
            continue
        new = _replacement(m.group(1), text, s, e)
        out.append(text[last:s])
        out.append(new)
        last = e
        rep[(m.group(1), new)] += 1
    out.append(text[last:])
    return "".join(out), rep, skip


def file_collisions(old: str, new: str) -> Dict[str, List[str]]:
    """Two different old identifiers that end up as one identifier inside ONE file.

    Works on the aligned word tokens of old and new text (replacements never
    change the token structure), so denied spans are taken into account."""
    tok = re.compile(r"[A-Za-z0-9_]+|[^A-Za-z0-9_]+")
    a, b = tok.findall(old), tok.findall(new)
    if len(a) != len(b):
        return {"<token structure changed>": []}
    seen: Dict[str, set] = collections.defaultdict(set)
    for x, y in zip(a, b):
        if x[0].isalnum() or x[0] == "_":
            seen[y].add(x)
    # the free brand word (prose) is not an identifier namespace
    return {k: sorted(v) for k, v in seen.items() if len(v) > 1 and k != "fLLiper"}


def ident_collisions(text: str, imap: Dict[str, str]) -> List[Tuple[str, str]]:
    """An ident-map target that is already a NAME in this file would merge two bindings."""
    import io as _io
    import tokenize as _tk
    try:
        names = {t.string for t in _tk.generate_tokens(_io.StringIO(text).readline) if t.type == _tk.NAME}
    except (_tk.TokenError, IndentationError, SyntaxError):
        return []
    return sorted((o, n) for o, n in imap.items() if o in names and n in names)


def rewrite_path(path: str, weg2: bool = False) -> str:
    parts = path.split("/")
    new_parts = []
    for p in parts:
        p = MAIN.sub(lambda m: _replacement(m.group(1), p, m.start(), m.end()), p)
        if weg2:
            p = rewrite_weg2(p)[0]   # same deny spans as content: a path and its string references stay aligned
        new_parts.append(p)
    return "/".join(new_parts)


def _glob(path: str, pats: Iterable[str]) -> bool:
    for pat in pats:
        if fnmatch.fnmatchcase(path, pat):
            return True
        if pat.endswith("/**") and (path == pat[:-3] or path.startswith(pat[:-2])):
            return True
    return False


def in_path_scope(path: str) -> bool:
    return _glob(path, INCLUDE) and not _glob(path, EXCLUDE)


def in_scope(path: str) -> bool:
    """content scope"""
    return in_path_scope(path) and not _glob(path, EXCLUDE_CONTENT)


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def _git(repo: str, *args: str, inp: Optional[bytes] = None) -> bytes:
    return subprocess.run(["git", "-C", repo, *args], input=inp, check=True,
                          stdout=subprocess.PIPE).stdout


def iter_ref(repo: str, ref: str):
    """Yield (path, mode, bytes) for every blob of ref, sorted by path."""
    rows = []
    for rec in _git(repo, "ls-tree", "-r", "-z", "--full-tree", ref).split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, typ, sha = meta.split()
        if typ != b"blob":
            continue
        rows.append((path.decode("utf-8", "surrogateescape"), mode.decode(), sha.decode()))
    rows.sort()
    proc = subprocess.Popen(["git", "-C", repo, "cat-file", "--batch"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert proc.stdin and proc.stdout
    for path, mode, sha in rows:
        proc.stdin.write(sha.encode() + b"\n")
        proc.stdin.flush()
        header = proc.stdout.readline().split()
        size = int(header[2])
        data = proc.stdout.read(size)
        proc.stdout.read(1)
        yield path, mode, data
    proc.stdin.close()
    proc.wait()


def iter_root(root: str, tracked_only: bool = True):
    args = ["ls-files", "-z", "-c"] if tracked_only else ["ls-files", "-z", "-c", "-o", "--exclude-standard"]
    paths = sorted({p.decode("utf-8", "surrogateescape")
                    for p in _git(root, *args).split(b"\0") if p})
    for p in paths:
        full = os.path.join(root, p)
        if os.path.islink(full):
            yield p, "120000", os.readlink(full).encode("utf-8", "surrogateescape")
        elif os.path.isfile(full):
            with open(full, "rb") as f:
                yield p, "100644", f.read()


def as_text(data: bytes) -> Optional[str]:
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


# --------------------------------------------------------------------------
# Inventory (classification of every occurrence)
# --------------------------------------------------------------------------
PROC = re.compile(r"sglang::(sched\w*|detok\w*|tokenizer\w*|data_parallel\w*|router\b|server\b|"
                  r"detokenizer_router|sch\b)")
CATS = [
    "pkg-import", "pkg-dotted", "proc-title", "torch-op-namespace", "cxx-namespace",
    "env-SGLANG_", "env-SGL_(keep)", "product-HTSGLANG_", "product-htsglang",
    "host-path-htsglang(keep)", "kernel-wheel(separate)", "logger-explicit", "cli-flag",
    "identifier", "prose-brand", "attribution/foreign(keep)",
]


def classify_text(path: str, text: str, counts: collections.Counter,
                  env_names: collections.Counter, samples: Dict[str, List[str]]) -> None:
    ext = os.path.splitext(path)[1]
    iv = deny_intervals(text)
    starts = [a for a, _, _ in iv]

    def add(cat: str, snippet: str) -> None:
        counts[cat] += 1
        lst = samples.setdefault(cat, [])
        if len(lst) < 4:
            lst.append(f"{path}: {snippet.strip()[:120]}")

    # htsglang layer
    for m in re.finditer(r"(?i)[\w./:-]*htsglang[\w.-]*", text):
        tok = m.group(0)
        if re.search(r"/spinning/htsglang", tok):
            add("host-path-htsglang(keep)", tok)
        elif tok.startswith("HTSGLANG_") or "HTSGLANG_" in tok:
            add("product-HTSGLANG_", tok)
            env_names["HT:" + re.search(r"HTSGLANG_\w*", tok).group(0)] += 1
        else:
            add("product-htsglang", tok)
    for m in re.finditer(r"\bSGL_[A-Z0-9_]*", text):
        add("env-SGL_(keep)", m.group(0))
        env_names["SGL:" + m.group(0)] += 1
    for m in re.finditer(r"\bsgl[-_]kernel[\w-]*|\bsglang-kernel\b", text):
        add("kernel-wheel(separate)", m.group(0))
    for m in MAIN.finditer(text):
        s, e = m.span()
        ls = text.rfind("\n", 0, s) + 1
        le = text.find("\n", e)
        line = text[ls: le if le >= 0 else len(text)]
        if _denied(iv, starts, s, e):
            add("attribution/foreign(keep)", line)
            continue
        if ext in CXX_EXT:
            add("cxx-namespace", line)
            continue
        word = m.group(1)
        after = text[e:e + 60]
        before = text[max(0, s - 40):s]
        if word == "SGLANG":
            tokm = re.match(r"\w*", text[s:])
            add("env-SGLANG_", tokm.group(0))
            env_names["SG:" + tokm.group(0)] += 1
            continue
        if word == "sglang" and after.startswith("::"):
            if PROC.match(text[s:s + 60]):
                add("proc-title", line)
            else:
                add("torch-op-namespace", line)
            continue
        if word == "sglang" and re.search(r"Library\(\s*[\"']$", before):
            add("torch-op-namespace", line)
            continue
        if word == "sglang" and re.match(r"\s*(import|from)\s+sglang\b", line) and ext in (".py", ".pyi", ""):
            add("pkg-import", line)
            continue
        if word == "sglang" and re.search(r"getLogger\(\s*[\"']$", before):
            add("logger-explicit", line)
            continue
        if word == "sglang" and re.search(r"--$", before):
            add("cli-flag", line)
            continue
        if word == "sglang" and after.startswith(".") and re.match(r"\.[A-Za-z_]", after):
            add("pkg-dotted", line)
            continue
        glued = (s > 0 and _W.match(text[s - 1])) or (e < len(text) and _W.match(text[e]))
        if word in ("SGLang", "SGlang", "sGLang") and not glued:
            add("prose-brand", line)
            continue
        if word == "sglang" and not glued and ext in (".md", ".rst", ".txt"):
            add("prose-brand", line)
            continue
        add("identifier", line)


def cmd_inventory(a: argparse.Namespace) -> int:
    counts: collections.Counter = collections.Counter()
    oos: collections.Counter = collections.Counter()
    env_names: collections.Counter = collections.Counter()
    samples: Dict[str, List[str]] = {}
    files_with: collections.Counter = collections.Counter()
    dir_paths = 0
    path_moves = 0
    ident_map: Dict[str, set] = collections.defaultdict(set)
    src = iter_ref(a.repo, a.ref) if a.ref else iter_root(a.root)
    n_files = 0
    for path, mode, data in src:
        n_files += 1
        if in_path_scope(path) and rewrite_path(path) != path:
            path_moves += 1
        text = as_text(data) if mode != "120000" else None
        if text is None:
            continue
        before = sum(counts.values())
        c: collections.Counter = counts if in_scope(path) else oos
        classify_text(path, text, c, env_names if in_scope(path) else collections.Counter(),
                      samples if in_scope(path) else {})
        if in_scope(path) and sum(counts.values()) > before:
            files_with["in-scope"] += 1
        if in_scope(path) and (os.path.splitext(path)[1] not in CXX_EXT):
            new_text = rewrite_text(text)[0]
            for k, v in file_collisions(text, new_text).items():
                ident_map[f"{path}: {k}"] = set(v)
    collisions = {k: sorted(v) for k, v in ident_map.items()}
    ext_counts = {}
    for d in a.scan or []:
        ext_counts[d] = scan_external(d)
    res = {
        "tool_version": TOOL_VERSION,
        "source": a.ref and f"{a.repo}@{a.ref}" or a.root,
        "resolved": a.ref and _git(a.repo, "rev-parse", a.ref).decode().strip() or None,
        "files_total": n_files,
        "in_scope_files_with_hits": files_with["in-scope"],
        "in_scope_path_moves": path_moves,
        "in_scope": dict(sorted(counts.items())),
        "out_of_scope": dict(sorted(oos.items())),
        "distinct_env": {
            "SGLANG_": len([k for k in env_names if k.startswith("SG:")]),
            "SGL_": len([k for k in env_names if k.startswith("SGL:")]),
            "HTSGLANG_": len([k for k in env_names if k.startswith("HT:")]),
        },
        "identifier_collisions": collisions,
        "samples": samples,
        "external": ext_counts,
    }
    js = json.dumps(res, indent=1, ensure_ascii=False, sort_keys=False)
    if a.json:
        with open(a.json, "w") as f:
            f.write(js + "\n")
    else:
        print(js)
    return 0


EXT_PATTERNS = [
    ("proc-title sglang::", re.compile(r"sglang::")),
    ("module path sglang.srt / -m sglang", re.compile(r"\bsglang\.(?:srt|launch_server|bench|test|cli|jit)")),
    ("import sglang", re.compile(r"^\s*(?:import|from)\s+sglang\b", re.M)),
    ("env SGLANG_*", re.compile(r"\bSGLANG_[A-Z0-9_]+")),
    ("env HTSGLANG_*", re.compile(r"\bHTSGLANG_[A-Z0-9_]+")),
    ("env SGL_*", re.compile(r"\bSGL_[A-Z0-9_]+")),
    ("path /opt/htsglang|/var/lib/htsglang", re.compile(r"/(?:opt|var/lib|etc)/htsglang")),
    ("image/name htsglang (other)", re.compile(r"(?<![/\w])htsglang(?![-\w]*gpu)[\w:.-]*")),
    ("host path /spinning/htsglang*", re.compile(r"/spinning/htsglang")),
    ("sgl_kernel / sgl-kernel", re.compile(r"\bsgl[-_]kernel")),
    ("cache dir .cache/sglang", re.compile(r"\.cache/sglang")),
]
EXT_SKIP_DIRS = {"ctx", ".git", "__pycache__", "node_modules", "logs", ".venv", "venv", "site-packages", ".cache"}
EXT_OK = re.compile(r"(\.(sh|py|env|toml|yml|yaml|service|conf|cfg|awk)|Dockerfile[\w.-]*|entrypoint[\w.-]*)$")  # scripts/config only, no logs or records


def scan_external(d: str) -> dict:
    per_pat: collections.Counter = collections.Counter()
    per_file: Dict[str, collections.Counter] = {}
    for dp, dns, fns in os.walk(d):
        # prune source snapshots / worktrees / caches: they are copies of the tree, not tools
        dns[:] = sorted(x for x in dns if x not in EXT_SKIP_DIRS and not x.startswith("wt-")
                        and not x.startswith("base-") and x not in ("scratch", "hfhome")
                        and not os.path.exists(os.path.join(dp, x, ".git"))
                        and not os.path.isdir(os.path.join(dp, x, "python", "sglang"))
                        and not os.path.isdir(os.path.join(dp, x, "sglang")))
        for fn in sorted(fns):
            if not EXT_OK.search(fn) or ".bak" in fn:
                continue
            full = os.path.join(dp, fn)
            try:
                if os.path.getsize(full) > 2_000_000:
                    continue
                with open(full, "rb") as f:
                    t = as_text(f.read())
            except OSError:
                continue
            if t is None:
                continue
            c = collections.Counter()
            for name, rx in EXT_PATTERNS:
                n = len(rx.findall(t))
                if n:
                    c[name] += n
            if c:
                per_file[os.path.relpath(full, d)] = c
                per_pat.update(c)
    top = sorted(per_file.items(), key=lambda kv: -sum(kv[1].values()))[:25]
    return {"files_with_hits": len(per_file), "per_pattern": dict(per_pat),
            "top_files": {k: dict(v) for k, v in top}}


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------
def _load_imap(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    with open(path) as f:
        m = {k: v for k, v in json.load(f).items() if k != "_comment"}
    if len(set(m.values())) != len(m):
        sys.exit("ident map is not injective")
    return m


def cmd_apply(a: argparse.Namespace) -> int:
    root = os.path.abspath(a.root)
    if os.path.realpath(root) in ("/spinning/htsglang", "/spinning/htsglang-gpu"):
        sys.exit("refusing to rewrite a shared checkout; use a throwaway worktree")
    rules = collections.Counter()
    skipped = collections.Counter()
    per_file = []
    moves = []
    targets: Dict[str, str] = {}
    digest_rows = []
    cxx = a.cxx
    imap = _load_imap(a.ident_map)
    pending: List[Tuple[str, str, Optional[bytes]]] = []
    allowed_clash: Dict[str, List[str]] = {}
    allowed_ident: Dict[str, List[str]] = {}
    for path, mode, data in iter_root(root):
        new_path = rewrite_path(path, a.weg2) if in_path_scope(path) else path
        if new_path in targets and targets[new_path] != path:
            sys.exit(f"path collision: {targets[new_path]} and {path} -> {new_path}")
        targets[new_path] = path
        new_data = data
        nrep = 0
        text = as_text(data) if mode != "120000" else None
        ext = os.path.splitext(path)[1]
        if text is not None and in_scope(path) and (cxx or ext not in CXX_EXT):
            new_text, rep, skip = rewrite_all(text, path.endswith(".py"), a.weg2, imap, path)
            # name-rule collisions: token-aligned over the whole text, without the ident map
            clash = file_collisions(text, rewrite_all(text, path.endswith(".py"), a.weg2, {}, path)[0]) \
                if imap else file_collisions(text, new_text)
            ok = COLLISION_OK.get(path, frozenset())
            for k in sorted(set(clash) & ok):
                allowed_clash[f"{path}: {k}"] = clash.pop(k)
            if clash:
                sys.exit(f"identifier collision in {path}: {clash}")
            if imap and path.endswith(".py"):
                clash = ident_collisions(text, imap)
                okw = IDENT_COLLISION_OK.get(path, frozenset())
                for o, n in [c for c in clash if c[0] in okw]:
                    allowed_ident.setdefault(path, []).append(f"{o}->{n}")
                clash = [c for c in clash if c[0] not in okw]
                if clash:
                    sys.exit(f"ident-map collision in {path}: {clash}")
            rules.update(rep)
            skipped.update(skip)
            nrep = sum(rep.values())
            new_data = new_text.encode("utf-8")
        if new_path != path:
            moves.append((path, new_path))
        if nrep or new_path != path:
            per_file.append({"path": path, "new_path": new_path, "replacements": nrep,
                             "sha256_old": hashlib.sha256(data).hexdigest(),
                             "sha256_new": hashlib.sha256(new_data).hexdigest()})
        digest_rows.append(f"{new_path}\0{hashlib.sha256(new_data).hexdigest()}")
        if nrep or new_path != path:
            pending.append((path, new_path, new_data if nrep else None))
    # phase 2: write only after every file passed its checks (a refusal leaves the tree untouched)
    for path, new_path, new_data in ([] if a.dry_run else pending):
        src = os.path.join(root, path)
        dst = os.path.join(root, new_path)
        if new_path != path:
            if os.path.lexists(dst):
                sys.exit(f"target exists: {dst}")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(src, dst)
        if new_data is not None:
            with open(dst, "wb") as f:
                f.write(new_data)
    if False:
        pass
    if not a.dry_run:
        # remove directories emptied by the moves (deepest first)
        for old, _ in sorted(moves, key=lambda x: -x[0].count("/")):
            d = os.path.dirname(os.path.join(root, old))
            while d != root and os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
                d = os.path.dirname(d)
    if not a.dry_run and not a.no_stage:
        # Stage the result. `git add -A` alone silently drops moved files that a
        # .gitignore rule matches at their new path (tracked-but-ignored files,
        # e.g. .claude/ skills) -- force-add every new path of a tracked file.
        _git(root, "add", "-A")
        _git(root, "add", "-f", "--pathspec-from-file=-", "--pathspec-file-nul",
             inp=b"\0".join(p.encode("utf-8", "surrogateescape") for p in sorted(targets)))
    digest = hashlib.sha256("\n".join(sorted(digest_rows)).encode("utf-8", "surrogateescape")).hexdigest()
    summary = {
        "tool_version": TOOL_VERSION, "dry_run": a.dry_run, "cxx": cxx, "weg2": a.weg2,
        "ident_map": a.ident_map, "ident_map_entries": len(imap),
        "files_changed": len(per_file),
        "files_content_changed": sum(1 for r in per_file if r["replacements"]),
        "paths_moved": len(moves),
        "replacements_total": sum(rules.values()),
        "replacements_by_rule": {f"{o}->{n}": c for (o, n), c in sorted(rules.items())},
        "skipped_by_deny_reason": dict(sorted(skipped.items())),
        "collisions_allowed": allowed_clash,
        "ident_collisions_allowed": allowed_ident,
        "top_files": sorted(per_file, key=lambda r: -r["replacements"])[:15],
        "tree_digest": digest,
    }
    if a.manifest:
        with open(a.manifest, "w") as f:
            json.dump({**summary, "files": per_file, "moves": moves}, f, indent=1)
            f.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "top_files"}, indent=1))
    print("top files:")
    for r in summary["top_files"]:
        print(f"  {r['replacements']:6d}  {r['path']}")
    return 0


# --------------------------------------------------------------------------
# verify -- independent of the rewrite engine above
# --------------------------------------------------------------------------
_VTOK = re.compile(r"[A-Za-z0-9_]+|[^A-Za-z0-9_]+")
_VARIANTS = {"sglang": "flliper", "SGLANG": "FLLIPER", "Sglang": "Flliper"}
_BRAND = ("SGLang", "SGlang", "sGLang")


def _vmap_word(w: str) -> str:
    """Map every old-name occurrence inside one word token (verifier's own table)."""
    out = []
    i = 0
    n = len(w)
    while i < n:
        hit = None
        for old in ("sglang", "SGLANG", "Sglang") + _BRAND:
            if w.startswith(old, i) and not (i >= 2 and w[i - 2:i].lower() == "ht"):
                hit = old
                break
        if hit is None:
            out.append(w[i])
            i += 1
            continue
        if hit in _VARIANTS:
            out.append(_VARIANTS[hit])
        else:
            alone = (i == 0 and i + len(hit) == n)
            out.append("fLLiper" if alone else "Flliper")
        i += len(hit)
    return "".join(out)


def _vmap_path(p: str) -> str:
    return "/".join("".join(_vmap_word(t) if t[0].isalnum() or t[0] == "_" else t
                            for t in _VTOK.findall(c)) if c else c for c in p.split("/"))


def _vw_word(w: str) -> str:
    """verifier's own weg2 table on one word token"""
    out, i = [], 0
    while i < len(w):
        if w.startswith("weg2_", i) and any(
                i >= len(pre) and w[i - len(pre):i] == pre
                and (i == len(pre) or not w[i - len(pre) - 1].isalnum())
                for pre in _V_EVIDENCE_PREFIXES):   # evidence prefixes boot_/memts_/preflight_weg2_ (kept)
            out.append("weg2")
            i += 4
            continue
        if (w.startswith(_V_EVIDENCE_FILE, i) and (i == 0 or not w[i - 1].isalnum())
                and i + len(_V_EVIDENCE_FILE) == len(w)):   # weg2_measured_record(.json) (kept)
            out.append(_V_EVIDENCE_FILE)
            i += len(_V_EVIDENCE_FILE)
            continue
        if w.startswith("weg2", i) and (i == 0 or not w[i - 1].isalnum()) and _V_TAG_AT.match(w, i):
            m = _V_TAG_AT.match(w, i)   # boot tag (evidence name): kept, through the end of its word
            out.append(m.group(0))
            i = m.end()
            continue
        for old in ("Weg2Flip", "weg2", "WEG2", "Weg2"):
            if w.startswith(old, i):
                if old == "Weg2Flip":
                    out.append("PdFlip")
                    i += 8
                    break
                if old == "weg2":
                    out.append("pdflip")
                elif old == "WEG2":
                    out.append("PDFLIP")
                else:
                    out.append("PdFlip")
                i += 4
                break
        else:
            out.append(w[i])
            i += 1
    return "".join(out)


_V_EVIDENCE_PREFIXES = ("boot_", "memts_", "preflight_")
_V_EVIDENCE_FILE = "weg2_measured_record"
_V_BOOT_TAG = re.compile(r"^weg2[a-z]")   # evidence file names keep their boot tag (any tag shape, FL4)
_V_TAG_AT = re.compile(r"weg2[a-z][A-Za-z0-9_]*")


def _vw_path(p: str) -> str:
    return "/".join(c if (not c or _V_BOOT_TAG.match(c)) else
                    "".join(_vw_word(t) if t[0].isalnum() or t[0] == "_" else t for t in _VTOK.findall(c))
                    for c in p.split("/"))


def _v_candidates(x: str, imap: Dict[str, str], weg2: bool) -> set:
    firsts = {x, imap.get(x, x)}
    out = set()
    for f in firsts:
        for g in (f, _vmap_word(f)):
            out.add(g)
            if weg2:
                out.add(_vw_word(g))
    return out


_VURL = re.compile(r"(?:https?|ssh|git|file|ftp)://\S+")
_VINTERP = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")


def _vurls(t: str) -> List[str]:
    """URLs with shell/format interpolations blanked: a renamed ${SGLANG_TAG} inside a
    URL is a variable rename, not a changed address."""
    return [_VINTERP.sub("$V", u) for u in _VURL.findall(t)]


def cmd_verify(a: argparse.Namespace) -> int:
    base = {p: (m, d) for p, m, d in iter_ref(a.repo, a.base)}
    tgt = {p: (m, d) for p, m, d in iter_root(a.root, tracked_only=False)}
    index = {p.decode("utf-8", "surrogateescape") for p in _git(a.root, "ls-files", "-z", "-c").split(b"\0") if p}
    for p in base:  # present on disk but ignored -> still a candidate, flagged below
        for q in (p, _vmap_path(p)):
            full = os.path.join(a.root, q)
            if q not in tgt and os.path.isfile(full) and not os.path.islink(full):
                with open(full, "rb") as f:
                    tgt[q] = ("100644", f.read())
    fails: List[str] = []
    chosen = set()
    stats = collections.Counter()
    imap = _load_imap(a.ident_map)
    for p, (mode, bd) in sorted(base.items()):
        vp = _vmap_path(p)
        pcs = [p, vp] + ([_vw_path(vp), _vw_path(p)] if a.weg2 else [])
        cand = [x for x in dict.fromkeys(pcs) if x in tgt]
        if len(cand) != 1:
            fails.append(f"path: {p} -> candidates present {cand}")
            continue
        q = cand[0]
        chosen.add(q)
        if q not in index:
            fails.append(f"not staged (ignored at its new path?): {q}")
        if q != p:
            stats["paths_moved"] += 1
        td = tgt[q][1]
        if td == bd:
            stats["files_identical"] += 1
            continue
        bt, tt = as_text(bd), as_text(td)
        if bt is None or tt is None or mode == "120000":
            fails.append(f"binary/symlink changed: {p}")
            continue
        btok, ttok = _VTOK.findall(bt), _VTOK.findall(tt)
        if len(btok) != len(ttok):
            fails.append(f"token count differs: {p} ({len(btok)} vs {len(ttok)})")
            continue
        bad = 0
        for x, y in zip(btok, ttok):
            if x == y:
                continue
            if (x[0].isalnum() or x[0] == "_") and y in _v_candidates(x, imap, a.weg2):
                stats["tokens_renamed"] += 1
                stats["case:" + ("brand" if y == "fLLiper" else "ident" if x in imap and y.startswith(imap[x][:1]) and imap[x] in y else "mapped")] += 1
                continue
            bad += 1
            if bad <= 3:
                fails.append(f"non-name change in {p}: {x!r} -> {y!r}")
        if bad:
            continue
        # attribution must be untouched
        if collections.Counter(_vurls(bt)) != collections.Counter(_vurls(tt)):
            fails.append(f"URL changed in {p}")
        bl = [l for l in bt.splitlines() if "Copyright" in l or "Licensed under" in l]
        tl = [l for l in tt.splitlines() if "Copyright" in l or "Licensed under" in l]
        if bl != tl:
            fails.append(f"copyright/licence line changed in {p}")
        stats["files_renamed_only"] += 1
    extra = sorted(set(tgt) - chosen)
    for x in extra[:20]:
        fails.append(f"extra file in target: {x}")
    if len(extra) > 20:
        fails.append(f"... {len(extra) - 20} more extra files")
    # remaining old names in target (expected: denied ones only)
    residual = collections.Counter()
    for q in chosen:
        t = as_text(tgt[q][1])
        if t:
            for m in re.finditer(r"(?<![Hh][Tt])(sglang|SGLANG|SGLang|Sglang)", t):
                residual["in-scope" if in_scope(_unmap_hint(q, base)) else "out-of-scope"] += 1
    res = {"base": a.base, "base_sha": _git(a.repo, "rev-parse", a.base).decode().strip(),
           "files_base": len(base), "files_target": len(tgt), **stats,
           "residual_old_names": dict(residual), "failures": len(fails)}
    print(json.dumps(res, indent=1))
    for f in fails[:50]:
        print("FAIL", f)
    print("VERDICT:", "PASS" if not fails else "FAIL")
    return 0 if not fails else 1


def _unmap_hint(q: str, base: dict) -> str:
    return q if q in base else q.replace("flliper", "sglang")


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------
def cmd_selftest(_: argparse.Namespace) -> int:
    cases = [
        ("import sglang.srt.server_args as sa", "import flliper.srt.server_args as sa"),
        ("setproctitle('sglang::scheduler_TP0')", "setproctitle('flliper::scheduler_TP0')"),
        ("os.environ.get('SGLANG_WEG2_GROUP')", "os.environ.get('FLLIPER_WEG2_GROUP')"),
        ("HTSGLANG_PROFILE=27b /opt/htsglang/src", "HTSGLANG_PROFILE=27b /opt/htsglang/src"),
        ("from sgl_kernel import fused", "from sgl_kernel import fused"),
        ('"sglang-kernel==0.4.4",', '"sglang-kernel==0.4.4",'),
        ("see https://github.com/sgl-project/sglang/issues/1", "see https://github.com/sgl-project/sglang/issues/1"),
        ("# Copyright 2023-2024 SGLang Team", "# Copyright 2023-2024 SGLang Team"),
        ("SGLang serves models.", "fLLiper serves models."),
        ("class SGLangError(Exception):", "class FlliperError(Exception):"),
        ("class SglangSupervisor:", "class FlliperSupervisor:"),
        ("bench_sglang --sglang-url x", "bench_flliper --flliper-url x"),
        ('model="lmsys/sglang-EAGLE-llama2-chat-7B"', 'model="lmsys/sglang-EAGLE-llama2-chat-7B"'),
        ("import sglang_router", "import sglang_router"),
        ("SGL_ENABLE_JIT_DEEPGEMM", "SGL_ENABLE_JIT_DEEPGEMM"),
        ("~/.cache/sglang/x", "~/.cache/flliper/x"),
        ("from smg_grpc_proto import sglang_encoder_pb2", "from smg_grpc_proto import sglang_encoder_pb2"),
        ('ENCODER_SERVICE = "sglang.grpc.encoder.SglangEncoder"', 'ENCODER_SERVICE = "sglang.grpc.encoder.SglangEncoder"'),
        ("X = sglang_encoder_pb2_grpc.SglangEncoderServicer", "X = sglang_encoder_pb2_grpc.SglangEncoderServicer"),
        ("pip install smg-grpc-servicer[sglang]", "pip install smg-grpc-servicer[sglang]"),
        ('"schema": "sglang.expert_stats/1", sglang.srt.x', '"schema": "sglang.expert_stats/1", flliper.srt.x'),
        ("/spinning/htsglang-gpu/.venv/bin/python -m sglang.launch_server",
         "/spinning/htsglang-gpu/.venv/bin/python -m flliper.launch_server"),
    ]
    bad = 0
    for src, want in cases:
        got, _, _ = rewrite_text(src)
        ok = got == want
        bad += not ok
        print(("ok  " if ok else "FAIL"), repr(src), "->", repr(got))
    for p, want in [("python/sglang/srt/weg2/launcher.py", "python/flliper/srt/weg2/launcher.py"),
                    ("scripts/killall_sglang.sh", "scripts/killall_flliper.sh")]:
        got = rewrite_path(p)
        bad += got != want
        print(("ok  " if got == want else "FAIL"), p, "->", got)
        vgot = _vmap_path(p)
        bad += vgot != want
        print(("ok  " if vgot == want else "FAIL"), "(verifier)", p, "->", vgot)
    for w, want in [("SGLang", "fLLiper"), ("SGLangError", "FlliperError"), ("htsglang", "htsglang"),
                    ("SGLANG_X", "FLLIPER_X"), ("bench_sglang", "bench_flliper")]:
        got = _vmap_word(w)
        bad += got != want
        print(("ok  " if got == want else "FAIL"), "(verifier word)", w, "->", got)
    for src, want in [("raise Weg2FlipKvRelayInfeasible(x)", "raise PdFlipKvRelayInfeasible(x)"),
                      ('__all__ = ["Weg2FlipRankDisagree"]', '__all__ = ["PdFlipRankDisagree"]'),
                      ("WEG2-FLIP-TIMELINE Weg2StoreHandbackFailed", "PDFLIP-FLIP-TIMELINE PdFlipStoreHandbackFailed"),
                      ("boot_weg2_weg2xsn25_ab.log", "boot_weg2_weg2xsn25_ab.log"),
                      ('f"{EVID}/boot_weg2_{tag}_x" weg2_boot', 'f"{EVID}/boot_weg2_{tag}_x" pdflip_boot'),
                      ('f"{GPU_ARB}/weg2/boot_{tag}.json" "/weg2/state"', 'f"{GPU_ARB}/weg2/boot_{tag}.json" "/pdflip/state"'),
                      ('MEASURED_RECORD_NAME = "weg2_measured_record.json"', 'MEASURED_RECORD_NAME = "weg2_measured_record.json"'),
                      ('f"{GPU_ARB}/memts_weg2_{tag}.csv" f"{GPU_ARB}/preflight_weg2_{tag}.log" weg2_memts',
                       'f"{GPU_ARB}/memts_weg2_{tag}.csv" f"{GPU_ARB}/preflight_weg2_{tag}.log" pdflip_memts'),
                      ("xmemts_weg2_y weg2_measured_records", "xmemts_pdflip_y pdflip_measured_records"),
                      ('startswith("weg2xsn") prefix="weg2ring-" weg2_x weg2.front', 'startswith("weg2xsn") prefix="weg2ring-" pdflip_x pdflip.front'),
                      ('int.from_bytes(b"WEG2XCHG", "little") WEG2-XCHG', 'int.from_bytes(b"WEG2XCHG", "little") PDFLIP-XCHG'),
                      ('"--weg2-xchg-region=armed" "--weg2-xchg-inject" CALIB_SCHEMA = "weg2-pp-calib/1"',
                       '"--weg2-xchg-region=armed" "--pdflip-xchg-inject" CALIB_SCHEMA = "weg2-pp-calib/1"'),
                      ('FOOTPRINT_SCHEMA = "weg2-footprint/1" X_CURVES_FORMAT = "weg2-x-curves/1" SCHEMA_V3 = "weg2.form_measures/3" weg2.state/1',
                       'FOOTPRINT_SCHEMA = "weg2-footprint/1" X_CURVES_FORMAT = "weg2-x-curves/1" SCHEMA_V3 = "weg2.form_measures/3" pdflip.state/1')]:
        got, _, _ = rewrite_weg2(src)
        bad += got != want
        print(("ok  " if got == want else "FAIL"), "(weg2)", repr(src), "->", repr(got))
    for w, want in [("Weg2FlipTagUnparsable", "PdFlipTagUnparsable"), ("Weg2Front", "PdFlipFront"),
                    ("weg2_measured_record", "weg2_measured_record"), ("memts_weg2_", "memts_weg2_"),
                    ("preflight_weg2_", "preflight_weg2_"), ("weg2_measured_records", "pdflip_measured_records"),
                    ("xmemts_weg2_", "xmemts_pdflip_")]:
        got = _vw_word(w)
        bad += got != want
        print(("ok  " if got == want else "FAIL"), "(verifier weg2 word)", w, "->", got)
    # F0-M (08.10.2026): metric names are must-keep (table data/metric_names_1008.json); the same words as identifiers are renamed
    mk = [("name=\"sglang:num_running_reqs\"", "name=\"sglang:num_running_reqs\""),
          ("sglang:e2e_request_latency_seconds_bucket{le=\"1\"}", "sglang:e2e_request_latency_seconds_bucket{le=\"1\"}"),
          ("sglang:spill_tier_used_bytes sglang:spill_tier_*_bytes", "sglang:spill_tier_used_bytes sglang:spill_tier_*_bytes"),
          ("docker run sglang:dev local/sglang:latest", "docker run flliper:dev local/flliper:latest"),
          ("'sglang:prompt_tokens_total 10\\nsglang:generation_tokens_total 20\\n'", "'sglang:prompt_tokens_total 10\\nsglang:generation_tokens_total 20\\n'"),
          ("xsglang:num_running_reqs", "xflliper:num_running_reqs")]
    for src, want in mk:
        got, _, _ = rewrite_text(src)
        bad += got != want
        print(("ok  " if got == want else "FAIL"), "(metric keep)", repr(src), "->", repr(got))
    mkw = [("weg2_ttft_seconds weg2_prefill_s", None, "weg2_ttft_seconds pdflip_prefill_s"),
           ("weg2_front_up weg2_rank_prefill_new_tokens_total weg2_boot_decode_settled_ts", None,
            "weg2_front_up weg2_rank_prefill_new_tokens_total weg2_boot_decode_settled_ts"),
           ("weg2_served_total weg2_flips_total", None, "weg2_served_total weg2_flips_total"),
           ("getattr(sa, 'weg2_group') weg2_d_parked", None, "getattr(sa, 'pdflip_group') pdflip_d_parked"),
           ("weg2_group=\"P\" _weg2_rank()", "python/sglang/srt/weg2/front.py", "weg2_group=\"P\" _pdflip_rank()"),
           ("weg2_group=\"P\" _weg2_rank()", "python/flliper/srt/pdflip/front.py", "weg2_group=\"P\" _pdflip_rank()"),
           ("weg2_d_parked weg2_group_name", "python/sglang/srt/weg2/front_metrics.py", "weg2_d_parked weg2_group_name"),
           ("job_name: weg2-front", "tools/rig_dashboard/rigdash/deploy/vm/scrape.yml", "job_name: weg2-front"),
           ("job_name: weg2-front", "tools/rig_dashboard/rigdash/deploy/vm/other.yml", "job_name: pdflip-front"),
           ("sglang_{prefix}_{key} sglang_x", "python/sglang/srt/entrypoints/v1_loads.py", "sglang_{prefix}_{key} sglang_x")]
    for src, pth, want in mkw:
        got, _, _ = rewrite_all(src, False, True, {}, pth)
        bad += got != want
        print(("ok  " if got == want else "FAIL"), "(metric keep, weg2)", repr(src), pth, "->", repr(got))
    fx = "test/registered/unit/weg2/fixtures/wake_credit_h14/x.lines"
    got = (in_path_scope(fx), in_scope(fx))
    bad += got != (True, False)
    print(("ok  " if got == (True, False) else "FAIL"), "(fixture: moves, content kept)", fx, got)
    for kp in ("tools/release/data/merged_0928.json", "tools/release/rename_to_flliper.py", "docker/weg2-release/rename_rigdash.py",
               "tools/rig_dashboard/rigdash/kartenplan_data/nf-int4-abl.json"):
        got = (in_path_scope(kp), in_scope(kp))
        bad += got != (True, False)
        print(("ok  " if got == (True, False) else "FAIL"), "(kit: in the tree, content kept)", kp, got)
    print("SELFTEST", "PASS" if not bad else f"FAIL ({bad})")
    return 0 if not bad else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    i = sp.add_parser("inventory")
    i.add_argument("--repo", default="/spinning/htsglang")
    i.add_argument("--ref")
    i.add_argument("--root")
    i.add_argument("--scan", action="append", help="extra directory with operator tools")
    i.add_argument("--json")
    p = sp.add_parser("apply")
    p.add_argument("--root", required=True, help="throwaway worktree")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cxx", action="store_true", help="also rename C/C++/CUDA sources")
    p.add_argument("--manifest")
    p.add_argument("--no-stage", action="store_true", help="do not `git add` the result")
    p.add_argument("--weg2", action="store_true", help="also rename weg2 -> pdflip")
    p.add_argument("--ident-map", help="JSON {old_identifier: new_identifier} for Python NAME tokens")
    v = sp.add_parser("verify")
    v.add_argument("--repo", default="/spinning/htsglang")
    v.add_argument("--base", required=True)
    v.add_argument("--root", required=True)
    v.add_argument("--weg2", action="store_true")
    v.add_argument("--ident-map")
    sp.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "inventory" and not (a.ref or a.root):
        ap.error("inventory needs --ref or --root")
    return {"inventory": cmd_inventory, "apply": cmd_apply, "verify": cmd_verify,
            "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
