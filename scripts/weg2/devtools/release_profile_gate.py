#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Release-profile gate (item 380, desk, 03.10.2026): do the profiles in
``docker/profiles_release/`` match the user orders of 03.10.?

READ ONLY and desk only.  Each release profile is SOURCED in a throw-away
``env -i bash`` (the same way ``entrypoint.sh`` and ``profile_docker --check`` read it: ``source``
the file, then ``profile_form_env`` with ``_form`` exporting its value) and the *effective*
values are compared with the orders.  Nothing is launched: no GPU, no container, no
``weg2.launcher`` process, no port.  A red check names ``file:line`` (the last line that
assigns the value, following the ``source`` chain to ``27b-base.env``) or, when the line
that must exist is missing, the profile's identity line as the anchor.

Orders checked (one rule id each, printed with every finding):

  ALL    CARD-COUNT   PROFILE_CARD_COUNT=3
         INVENTORY    PROFILE_INVENTORY set (item 250)
         STATUS       PROFILE_STATUS is not ``experimentell``
  NF     NF-MODEL     PROFILE_MODEL = ...Minachist-abl-wxp and PROFILE_DRAFT = ...albucino-abl-wxp
                      (model of profiles/nf-int4-h6-abl.env), ``--model`` in PROFILE_ARGS equals it
         NF-NO-L15    no L1.5 master on NF (launcher refuses it: W-L15-27B-ONLY)
  27B    27B-MODEL    PROFILE_MODEL = Qwen3.8-27B-INT8-gdncov-vocabembed, no abl in model/draft
         27B-ARGV     ``--model <PROFILE_MODEL>`` explicit in PROFILE_ARGS (memory 27b-profil-ohne-model-flag-1003)
         27B-L15      SGLANG_WEG2_L15=1, _L15_REFILL=1, _L15_HOT_SHARE=1, SGLANG_WEG2_VMM_EXPORTABLE=1
  DUAL   DUAL-MODEL   PROFILE_MODEL = Qwen3.8-27B-NVFP4-RadixArk, no abl, ``--model`` explicit
         DUAL-NO-L15  no SGLANG_WEG2_L15* / HOT_HANDOVER in the effective env, no such code line (W-L15-DUAL)
         DUAL-MPS     SGLANG_WEG2_DUAL_MPS_OPT_IN=1 (set by profile_form_env, i.e. in the launcher's env)

Which file plays which role is the ROLES table; any other ``*.env`` in the directory (base,
aliases, drafts, experimental forms) is listed as SKIP, or checked with the generic ALL rules
under ``--all`` (informative, counted).

Exit: 0 all green, 1 at least one red, 2 usage / unreadable directory.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_DIR = "/spinning/gpu-arb/docker/profiles_release"
DEFAULT_REF = "/spinning/gpu-arb/docker/profiles/nf-int4-h6-abl.env"

#: profile file -> role (the shipped release profiles; make_flat_ctx takes them via PROFILE_OVERLAY)
ROLES: Dict[str, str] = {
    "nf-int4.env": "nf",
    "nf.env": "nf",
    "27b.env": "27b",
    "27b-nvfp4-dual.env": "dual",
}

NF_MODEL_BASE = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp"
NF_DRAFT_BASE = "Qwen3.8-Flash-Next-MTP-INT4-g32-albucino-abl-wxp"
B27_MODEL_BASE = "Qwen3.8-27B-INT8-gdncov-vocabembed"
DUAL_MODEL_BASE = "Qwen3.8-27B-NVFP4-RadixArk"
L15_27B_ENV = (
    "SGLANG_WEG2_L15",
    "SGLANG_WEG2_L15_REFILL",
    "SGLANG_WEG2_L15_HOT_SHARE",
    "SGLANG_WEG2_VMM_EXPORTABLE",
)
_ABL_RE = re.compile(r"(^|[-_/.])abl([-_/.]|$)", re.I)
_L15_CODE_RE = re.compile(r"SGLANG_WEG2_L15|SGLANG_WEG2_HOT_HANDOVER")
_SOURCE_RE = re.compile(r'^\s*(?:source|\.)\s+"\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/([^"]+)"')
_ON = ("1", "true", "on")

_VARS = (
    "PROFILE_NAME", "PROFILE_LINE", "PROFILE_STATUS", "PROFILE_MODEL", "PROFILE_DRAFT",
    "PROFILE_CARD_COUNT", "PROFILE_INVENTORY",
)

#: sourced under ``env -i``; _form/say/refuse are the entrypoint's helpers reduced to what a read needs
_DUMP = r"""
set +e
say() { :; }
refuse() { echo "refuse: $*" >&2; exit 97; }
_form() { export "$1=$2"; }
source "$1" || exit 98
if declare -F profile_form_env >/dev/null; then profile_form_env || exit 99; fi
printf '==VARS==\0'
for v in %s; do printf '%%s\0%%s\0' "$v" "${!v-}"; done
printf '==ARGS==\0'
if [ "${#PROFILE_ARGS[@]}" -gt 0 ]; then printf '%%s\0' "${PROFILE_ARGS[@]}"; fi
printf '==ENV==\0'
env -0
""" % " ".join(_VARS)


@dataclass
class Effective:
    path: str
    vars: Dict[str, str] = field(default_factory=dict)
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    error: str = ""


@dataclass
class Finding:
    profile: str
    rule: str
    ok: bool
    where: str
    detail: str

    def render(self) -> str:
        return "%-4s %-12s %s  %s" % ("ok" if self.ok else "ROT", self.rule, self.where, self.detail)


def evaluate(path: str, timeout: float = 60.0) -> Effective:
    """Source ``path`` in a clean bash and return its effective PROFILE_* values, args and exported env."""
    eff = Effective(path=path)
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LC_ALL": "C", "HTSGLANG_INSTRUMENTS": "0"}
    try:
        cp = subprocess.run(
            ["env", "-i"] + ["%s=%s" % kv for kv in env.items()]
            + ["bash", "--noprofile", "--norc", "-c", _DUMP, "gate", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="/tmp", timeout=timeout)
    except subprocess.TimeoutExpired:
        eff.error = "Sourcen dauerte laenger als %ds" % int(timeout)
        return eff
    if cp.returncode != 0:
        eff.error = "Sourcen bricht ab (rc=%d): %s" % (
            cp.returncode, cp.stderr.decode("utf-8", "replace").strip()[-200:])
        return eff
    parts = cp.stdout.split(b"\0")
    section = ""
    i = 0
    while i < len(parts):
        tok = parts[i].decode("utf-8", "replace")
        if tok in ("==VARS==", "==ARGS==", "==ENV=="):
            section = tok
            i += 1
            continue
        if section == "==VARS==":
            if i + 1 < len(parts):
                eff.vars[tok] = parts[i + 1].decode("utf-8", "replace")
            i += 2
            continue
        if section == "==ARGS==":
            if tok != "" or i + 1 < len(parts):
                eff.args.append(tok)
        elif section == "==ENV==" and "=" in tok:
            k, _, v = tok.partition("=")
            eff.env[k] = v
        i += 1
    return eff


# --------------------------------------------------------------------------- line lookup

def _strip_comment(text: str) -> str:
    s = text.lstrip()
    if s.startswith("#"):
        return ""
    return re.sub(r"\s#.*$", "", text)


def code_lines(path: str, _seen: Optional[set] = None) -> List[Tuple[str, int, str]]:
    """Non-comment lines in execution order, ``source "$(dirname ...)/X"`` expanded in place."""
    seen = _seen if _seen is not None else set()
    real = os.path.realpath(path)
    if real in seen:
        return []
    seen.add(real)
    out: List[Tuple[str, int, str]] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return out
    for n, raw in enumerate(lines, 1):
        code = _strip_comment(raw)
        if not code.strip():
            continue
        m = _SOURCE_RE.match(code)
        if m:
            out.extend(code_lines(os.path.join(os.path.dirname(path), m.group(1)), seen))
            continue
        out.append((path, n, raw))
    return out


def _here(path: str, lines: Sequence[Tuple[str, int, str]], entry: Optional[Tuple[str, int, str]], base: str) -> str:
    if entry is None:
        return "%s:?" % base
    f, n, _ = entry
    return "%s:%d" % (os.path.basename(f) if os.path.dirname(f) == os.path.dirname(path) else f, n)


def last_assign(lines: Sequence[Tuple[str, int, str]], var: str) -> Optional[Tuple[str, int, str]]:
    rx = re.compile(r"^\s*(?:export\s+)?%s=" % re.escape(var))
    hit = None
    for ent in lines:
        if rx.match(_strip_comment(ent[2])):
            hit = ent
    return hit


def last_match(lines: Sequence[Tuple[str, int, str]], pattern: str) -> Optional[Tuple[str, int, str]]:
    rx = re.compile(pattern)
    hit = None
    for ent in lines:
        if rx.search(_strip_comment(ent[2])):
            hit = ent
    return hit


def all_matches(lines: Sequence[Tuple[str, int, str]], pattern: str) -> List[Tuple[str, int, str]]:
    rx = re.compile(pattern)
    return [e for e in lines if rx.search(_strip_comment(e[2]))]


def _short(text: str, n: int = 150) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 3] + "..."


# --------------------------------------------------------------------------- rules

class _Ctx:
    def __init__(self, path: str, eff: Effective):
        self.path = path
        self.base = os.path.basename(path)
        self.eff = eff
        self.lines = code_lines(path)
        self.findings: List[Finding] = []

    def anchor(self, ent: Optional[Tuple[str, int, str]]) -> str:
        if ent is None:
            ent = last_assign(self.lines, "PROFILE_NAME")
        return _here(self.path, self.lines, ent, self.base)

    def add(self, rule: str, ok: bool, ent: Optional[Tuple[str, int, str]], detail: str,
            missing: bool = False) -> None:
        text = "" if ent is None else "  [%s]" % _short(ent[2])
        if missing and not ok:
            text = "  [Zeile fehlt -- Anker: %s]" % ("PROFILE_ARGS" if ent is not None else "PROFILE_NAME")
        self.findings.append(Finding(self.base, rule, ok, self.anchor(ent), detail + (text if not ok else "")))


def _arg_after(args: Sequence[str], flag: str) -> Optional[str]:
    val = None
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            val = args[i + 1]
    return val


def _basename(p: str) -> str:
    return os.path.basename(p.rstrip("/"))


def _on(env: Dict[str, str], key: str) -> bool:
    return str(env.get(key, "") or "").strip().lower() in _ON


def check_common(c: _Ctx) -> None:
    v = c.eff.vars
    cnt = v.get("PROFILE_CARD_COUNT", "")
    c.add("CARD-COUNT", cnt == "3", last_assign(c.lines, "PROFILE_CARD_COUNT"),
          "PROFILE_CARD_COUNT=%r, erwartet 3" % cnt)
    inv = v.get("PROFILE_INVENTORY", "")
    c.add("INVENTORY", bool(inv.strip()), last_assign(c.lines, "PROFILE_INVENTORY"),
          "PROFILE_INVENTORY=%r, erwartet gesetzt (Item 250)" % inv)
    st = v.get("PROFILE_STATUS", "")
    c.add("STATUS", st.strip().lower() != "experimentell", last_assign(c.lines, "PROFILE_STATUS"),
          "PROFILE_STATUS=%r, erwartet nicht experimentell" % st)


def _check_model_arg(c: _Ctx, rule: str) -> None:
    model = c.eff.vars.get("PROFILE_MODEL", "")
    arg = _arg_after(c.eff.args, "--model")
    ent = last_match(c.lines, r"--model\b") or last_assign(c.lines, "PROFILE_ARGS")
    if arg is None:
        c.add(rule, False, ent, "PROFILE_ARGS traegt kein '--model' (Launcher-Default greift, nicht PROFILE_MODEL)",
              missing=True)
    else:
        c.add(rule, arg == model, ent, "--model %s, PROFILE_MODEL %s" % (_basename(arg), _basename(model)))


def check_nf(c: _Ctx, ref_model: str, ref_draft: str) -> None:
    v = c.eff.vars
    model, draft = v.get("PROFILE_MODEL", ""), v.get("PROFILE_DRAFT", "")
    c.add("NF-MODEL", _basename(model) == ref_model, last_assign(c.lines, "PROFILE_MODEL"),
          "PROFILE_MODEL=%s, erwartet %s" % (_basename(model), ref_model))
    c.add("NF-MODEL", _basename(draft) == ref_draft, last_assign(c.lines, "PROFILE_DRAFT"),
          "PROFILE_DRAFT=%s, erwartet %s" % (_basename(draft), ref_draft))
    _check_model_arg(c, "NF-MODEL")
    c.add("NF-NO-L15", not _on(c.eff.env, "SGLANG_WEG2_L15"),
          last_match(c.lines, r"SGLANG_WEG2_L15\b"), "SGLANG_WEG2_L15 ist auf NF verboten (W-L15-27B-ONLY)")


def check_27b(c: _Ctx) -> None:
    v = c.eff.vars
    model, draft = v.get("PROFILE_MODEL", ""), v.get("PROFILE_DRAFT", "")
    ent = last_assign(c.lines, "PROFILE_MODEL")
    c.add("27B-MODEL", _basename(model) == B27_MODEL_BASE, ent,
          "PROFILE_MODEL=%s, erwartet %s" % (_basename(model), B27_MODEL_BASE))
    c.add("27B-MODEL", not _ABL_RE.search(model), ent, "PROFILE_MODEL=%s, erwartet ohne abl" % _basename(model))
    c.add("27B-MODEL", not _ABL_RE.search(draft), last_assign(c.lines, "PROFILE_DRAFT"),
          "PROFILE_DRAFT=%s, erwartet ohne abl" % _basename(draft))
    _check_model_arg(c, "27B-ARGV")
    for key in L15_27B_ENV:
        got = c.eff.env.get(key, "")
        ent = last_match(c.lines, r"\b%s\b" % re.escape(key) + r"(?![A-Z_])")
        c.add("27B-L15", got == "1", ent, "%s=%r nach profile_form_env, erwartet 1" % (key, got),
              missing=ent is None)


def check_dual(c: _Ctx) -> None:
    v = c.eff.vars
    model, draft = v.get("PROFILE_MODEL", ""), v.get("PROFILE_DRAFT", "")
    ent = last_assign(c.lines, "PROFILE_MODEL")
    c.add("DUAL-MODEL", _basename(model) == DUAL_MODEL_BASE, ent,
          "PROFILE_MODEL=%s, erwartet %s" % (_basename(model), DUAL_MODEL_BASE))
    c.add("DUAL-MODEL", not (_ABL_RE.search(model) or _ABL_RE.search(draft)), ent,
          "Dual erwartet ohne abl (Modell %s, Draft %s)" % (_basename(model), _basename(draft)))
    _check_model_arg(c, "DUAL-MODEL")
    l15_env = sorted(k for k in c.eff.env if k.startswith("SGLANG_WEG2_L15") or k == "SGLANG_WEG2_HOT_HANDOVER")
    hits = all_matches(c.lines, _L15_CODE_RE.pattern)
    if not hits and not l15_env:
        c.add("DUAL-NO-L15", True, None, "keine L15-Zeile, kein L15-Env")
    for ent in hits:
        c.add("DUAL-NO-L15", False, ent, "L15-Zeile im Dual-Profil (W-L15-DUAL)")
    if l15_env and not hits:
        c.add("DUAL-NO-L15", False, None, "L15-Env gesetzt: %s (W-L15-DUAL)" % ",".join(l15_env))
    mps = c.eff.env.get("SGLANG_WEG2_DUAL_MPS_OPT_IN", "")
    c.add("DUAL-MPS", mps == "1", last_match(c.lines, r"SGLANG_WEG2_DUAL_MPS_OPT_IN"),
          "SGLANG_WEG2_DUAL_MPS_OPT_IN=%r nach profile_form_env, erwartet 1" % mps)


def check_profile(path: str, role: Optional[str], ref_model: str = NF_MODEL_BASE,
                  ref_draft: str = NF_DRAFT_BASE) -> List[Finding]:
    eff = evaluate(path)
    c = _Ctx(path, eff)
    if eff.error:
        c.add("EVALUATE", False, None, eff.error)
        return c.findings
    check_common(c)
    if role == "nf":
        check_nf(c, ref_model, ref_draft)
    elif role == "27b":
        check_27b(c)
    elif role == "dual":
        check_dual(c)
    return c.findings


def reference_models(ref_path: str) -> Tuple[str, str]:
    """Model/draft basenames of the NF abl template (profiles/nf-int4-h6-abl.env); constants when unreadable."""
    if ref_path and os.path.isfile(ref_path):
        eff = evaluate(ref_path)
        if not eff.error:
            m, d = _basename(eff.vars.get("PROFILE_MODEL", "")), _basename(eff.vars.get("PROFILE_DRAFT", ""))
            if m and d and _ABL_RE.search(m) and _ABL_RE.search(d):
                return m, d
    return NF_MODEL_BASE, NF_DRAFT_BASE


def run(directory: str, ref: str = DEFAULT_REF, include_all: bool = False, out=sys.stdout) -> int:
    if not os.path.isdir(directory):
        print("release_profile_gate: Verzeichnis fehlt: %s" % directory, file=sys.stderr)
        return 2
    ref_model, ref_draft = reference_models(ref)
    names = sorted(n for n in os.listdir(directory) if n.endswith(".env"))
    red = 0
    checked = 0
    print("release_profile_gate: %s  (NF-Vorbild: %s / %s)" % (directory, ref_model, ref_draft), file=out)
    for name in names:
        role = ROLES.get(name)
        if role is None and not include_all:
            print("SKIP %-26s kein Standard-Release-Profil (Basis/Alias/experimentell); --all prueft die ALL-Regeln" % name, file=out)
            continue
        checked += 1
        fs = check_profile(os.path.join(directory, name), role, ref_model, ref_draft)
        bad = [f for f in fs if not f.ok]
        red += len(bad)
        print("== %s  (Rolle %s): %s" % (name, role or "generisch", "ROT, %d Befund(e)" % len(bad) if bad else "gruen, %d Pruefungen" % len(fs)), file=out)
        for f in fs:
            if not f.ok or os.environ.get("RPG_VERBOSE"):
                print("   " + f.render(), file=out)
    missing = [n for n in ROLES if n not in names]
    for n in missing:
        red += 1
        print("ROT  FILE         %s/%s:?  Release-Profil fehlt" % (directory, n), file=out)
    print("== Ergebnis: %d Profile geprueft, %d rote Befunde" % (checked, red), file=out)
    return 1 if red else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("directory", nargs="?", default=DEFAULT_DIR)
    ap.add_argument("--ref", default=DEFAULT_REF, help="NF-abl-Vorbild (default %(default)s)")
    ap.add_argument("--all", action="store_true", help="auch Nicht-Release-*.env mit den ALL-Regeln pruefen")
    ns = ap.parse_args(argv)
    return run(ns.directory, ns.ref, ns.all)


if __name__ == "__main__":
    sys.exit(main())
