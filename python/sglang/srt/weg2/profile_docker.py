"""docker/profiles/*.env FROM THE REGISTRY (UNIFY_PLAN Schritt 9, NF-Registry Punkt 15).

The profile table outside the tree (``/spinning/gpu-arb/docker/profiles``,
``<name>.env`` sourced by the entrypoint) states per model + format what the
registry states again: checkpoint, draft, format flags, draft form, start X
and X ceiling, the pinned P cut, the P chunk policy and the rank switches
whose default follows the profile. Two copies drift; this module makes the
registry the source:

* ``--out DIR`` writes one GENERATED fragment per (profile, format),
  ``<docker name>.registry.env``: the registry's facts as ``PROFILE_REG_*``
  variables, ``PROFILE_REG_ARGS`` and ``profile_registry_env`` (the switch
  defaults as ``_form`` lines). It never touches an existing ``*.env``.
* ``--check DIR`` sources every ``<name>.env`` in DIR (bash, ``_form``
  stubbed, nothing executed beyond the file's own top level) and compares
  what it sets with the registry: ``same`` / ``DIFF`` / ``env-only`` /
  ``registry-only``. Exit 1 when a DIFF exists (a gate), 0 otherwise.

    python -m sglang.srt.weg2.profile_docker --check /spinning/gpu-arb/docker/profiles
    python -m sglang.srt.weg2.profile_docker --out /tmp/profiles-gen

PURE apart from the bash subprocess of ``--check``; imports only weg2.form.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.weg2 import form as F

#: docker profile name -> (registry profile, registry format). The base
#: profiles; arm variants (27b-int8-*, *-chunkB ...) are matched by their
#: PROFILE_LINE + PROFILE_FORMAT instead of their name.
DOCKER_PROFILES: Dict[str, Tuple[str, str]] = {
    "27b": (F.PROFILE_QWEN27B, "int8"),
    "27b-fp8": (F.PROFILE_QWEN27B, "fp8"),
    "27b-nvfp4": (F.PROFILE_QWEN27B, "nvfp4"),
    "27b-gguf": (F.PROFILE_QWEN27B, "gguf"),
    "nf": (F.PROFILE_NEXTFLASH, "int4-mixed"),
    "nf-nvfp4": (F.PROFILE_NEXTFLASH, "nvfp4"),
}
#: PROFILE_LINE -> registry profile
LINE_PROFILE = {"27b": F.PROFILE_QWEN27B, "nf": F.PROFILE_NEXTFLASH}

#: the draft form flag per registry draft kind
_SPEC_FORM = {"dflash2": "DFLASH", "mtp": "NEXTN"}

#: flags whose VALUE the registry owns (compared); presence-only flags below
_VALUE_FLAGS = ("--profile", "--spec-form", "--tp-prefill-max-tokens", "--x-ceiling-tokens",
                "--pp-stage-ratio", "--pp-attn-stage-ratio", "--p-chunk-policy")
_BARE_FLAGS = ("--fp8-uniform-marlin", "--fp4-native-mixed")


@dataclass(frozen=True)
class Fact:
    key: str
    value: str


def registry_facts(profile: str, fmt: str) -> List[Fact]:
    """What the registry says about (profile, format), as comparable strings."""
    row = F.PROFILES[profile]
    wf = row.formats[fmt]
    out = [
        Fact("PROFILE_REG_ID", row.id),
        Fact("PROFILE_REG_FORMAT", wf.name),
        Fact("PROFILE_FORMAT", wf.profile_format or wf.name),
        Fact("PROFILE_MODEL", wf.checkpoint),
        Fact("PROFILE_DRAFT", wf.draft or row.draft.path),
        Fact("PROFILE_REG_CONTEXT", str(row.context_tokens)),
        Fact("PROFILE_REG_KERNELS", f"sm8x={wf.sm8x} sm12x={wf.sm12x}"),
    ]
    flags: Dict[str, str] = {}
    if profile != F.DEFAULT_PROFILE:
        flags["--profile"] = row.id
    spec = _SPEC_FORM.get(row.draft.kind)
    if spec and row.draft.kind == "dflash2":
        flags["--spec-form"] = spec
    flags["--tp-prefill-max-tokens"] = str(row.x_start_tokens)
    flags["--x-ceiling-tokens"] = str(row.x_ceiling_tokens)
    if wf.p_cut_pin:
        flags["--pp-stage-ratio"] = ",".join(map(str, wf.p_cut_pin[0]))
        flags["--pp-attn-stage-ratio"] = ",".join(map(str, wf.p_cut_pin[1]))
    if row.chunk.policy == "dynamic" and fmt in ("int8", "nvfp4"):
        flags["--p-chunk-policy"] = "dynamic"
    for k, v in flags.items():
        out.append(Fact(k, v))
    for b in _BARE_FLAGS:
        out.append(Fact(b, "set" if b in wf.args else "unset"))
    for name, val in sorted(row.switch_defaults().items()):
        out.append(Fact(f"_form {name}", _env_value(val)))
    return out


def _env_value(v: object) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


def render(docker_name: str, profile: str, fmt: str) -> str:
    """The generated fragment for one docker profile."""
    facts = registry_facts(profile, fmt)
    lines = [
        "# shellcheck shell=bash",
        f"# GENERATED -- python -m sglang.srt.weg2.profile_docker --out (registry {profile}, format {fmt}).",
        "# Source of truth: weg2/form.py PROFILES + weg2/profile_records_data. Do not edit; regenerate.",
        f"# Borrowed rows: {F.borrowed_constants_line(profile) or 'none'}",
    ]
    args: List[str] = []
    for f in facts:
        if f.key.startswith("_form "):
            continue
        if f.key.startswith("--"):
            if f.key in _BARE_FLAGS:
                if f.value == "set":
                    args.append(f.key)
            else:
                args += [f.key, f.value]
            continue
        lines.append(f"{f.key}={shlex.quote(f.value)}")
    lines.append("PROFILE_REG_ARGS=(" + " ".join(shlex.quote(a) for a in args) + ")")
    lines.append("profile_registry_env() {")
    for f in facts:
        if f.key.startswith("_form "):
            lines.append(f"  _form {f.key[6:]} {shlex.quote(f.value)}")
    lines.append("}")
    lines.append(f"PROFILE_REG_RECORDS={shlex.quote(records_summary(profile))}")
    return "\n".join(lines) + "\n"


def records_summary(profile: str) -> str:
    from sglang.srt.weg2 import profile_records as _pr

    own = _pr.own_records(profile)
    src, names, _ = _pr.borrow_of(profile)
    return (f"{len(own)} own record(s)" + (f", {len(names)} borrowed from {src}" if names else ""))


# ---------------------------------------------------------------------------
# reading an existing profile (bash)

_DUMP = r'''
_form() { printf 'FORM\t%s\t%s\n' "$1" "$2"; }
set +e
cd "$(dirname "$1")" || exit 2
source "$1" >/dev/null 2>&1 || true
for v in PROFILE_NAME PROFILE_LINE PROFILE_FORMAT PROFILE_MODEL PROFILE_DRAFT PROFILE_STATUS; do
  printf 'VAR\t%s\t%s\n' "$v" "${!v}"
done
for a in "${PROFILE_ARGS[@]}"; do printf 'ARG\t%s\n' "$a"; done
declare -F profile_form_env >/dev/null && profile_form_env 2>/dev/null
exit 0
'''


def read_env(path: str, runner=None) -> Dict[str, object]:
    """The facts an existing ``<name>.env`` sets: vars, argv, ``_form`` lines."""
    run = runner or (lambda p: subprocess.run(["bash", "-c", _DUMP, "dump", p], capture_output=True,
                                              text=True, timeout=60).stdout)
    out: Dict[str, object] = {"vars": {}, "args": [], "form": {}}
    for ln in run(os.path.abspath(path)).splitlines():
        parts = ln.split("\t")
        if parts[0] == "VAR" and len(parts) >= 3:
            out["vars"][parts[1]] = parts[2]
        elif parts[0] == "ARG" and len(parts) >= 2:
            out["args"].append(parts[1])
        elif parts[0] == "FORM" and len(parts) >= 3:
            out["form"][parts[1]] = parts[2]
    return out


def env_facts(env: Mapping[str, object]) -> Dict[str, str]:
    """The same keys as :func:`registry_facts`, read off one profile."""
    vars_ = env["vars"]
    args: List[str] = list(env["args"])
    out: Dict[str, str] = {}
    for k in ("PROFILE_FORMAT", "PROFILE_MODEL", "PROFILE_DRAFT"):
        if vars_.get(k):
            out[k] = str(vars_[k])
    # the last occurrence of a flag wins (argparse)
    for i, a in enumerate(args):
        if a in _VALUE_FLAGS and i + 1 < len(args):
            out[a] = args[i + 1]
        elif "=" in a and a.split("=", 1)[0] in _VALUE_FLAGS:
            k, v = a.split("=", 1)
            out[k] = v
    for b in _BARE_FLAGS:
        out[b] = "set" if b in args else "unset"
    for k, v in env["form"].items():
        out[f"_form {k}"] = str(v)
    return out


def match_profile(name: str, env: Mapping[str, object]) -> Optional[Tuple[str, str]]:
    """(registry profile, format) of a docker profile: by name, else by its
    PROFILE_LINE + PROFILE_FORMAT (arm variants)."""
    if name in DOCKER_PROFILES:
        return DOCKER_PROFILES[name]
    vars_ = env["vars"]
    prof = LINE_PROFILE.get(str(vars_.get("PROFILE_LINE", "")))
    if prof is None:
        return None
    pf = str(vars_.get("PROFILE_FORMAT", ""))
    for fmt, wf in F.PROFILES[prof].formats.items():
        if pf in (fmt, wf.profile_format):
            return prof, fmt
    return None


def compare(reg: Sequence[Fact], env: Mapping[str, str]) -> List[Tuple[str, str, str, str]]:
    """(key, registry, env, verdict) rows. ``_form`` switches the env does not
    set are 'registry-only' (the registry default applies); env ``_form``
    lines the registry does not own are not listed (arm form, not registry)."""
    rows = []
    for f in reg:
        if f.key.startswith("PROFILE_REG_"):
            continue
        got = env.get(f.key)
        if got is None:
            rows.append((f.key, f.value, "-", "registry-only"))
        elif got == f.value:
            rows.append((f.key, f.value, got, "same"))
        else:
            rows.append((f.key, f.value, got, "DIFF"))
    reg_keys = {f.key for f in reg}
    for k in _VALUE_FLAGS:
        if k in env and k not in reg_keys:
            rows.append((k, "-", env[k], "env-only"))
    return rows


def check_dir(d: str, names: Optional[Iterable[str]] = None, runner=None) -> Tuple[List[str], int]:
    lines: List[str] = []
    diffs = 0
    files = sorted(n[:-4] for n in os.listdir(d) if n.endswith(".env"))
    if names:
        files = [n for n in files if n in set(names)]
    for name in files:
        env = read_env(os.path.join(d, name + ".env"), runner)
        m = match_profile(name, env)
        if m is None:
            lines.append(f"== {name}: no registry row (PROFILE_LINE={env['vars'].get('PROFILE_LINE')!r} "
                         f"PROFILE_FORMAT={env['vars'].get('PROFILE_FORMAT')!r}) -- not checked")
            continue
        rows = compare(registry_facts(*m), env_facts(env))
        nd = sum(1 for r in rows if r[3] == "DIFF")
        diffs += nd
        lines.append(f"== {name} -> registry {m[0]}/{m[1]}: {nd} DIFF, "
                     f"{sum(1 for r in rows if r[3] == 'same')} same, "
                     f"{sum(1 for r in rows if r[3] == 'registry-only')} registry-only, "
                     f"{sum(1 for r in rows if r[3] == 'env-only')} env-only")
        for k, rv, ev, verdict in rows:
            if verdict != "same":
                lines.append(f"   {verdict:13s} {k}: registry={rv} env={ev}")
    return lines, diffs


def generate(out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, (prof, fmt) in sorted(DOCKER_PROFILES.items()):
        # always <name>.registry.env: an existing <name>.env is never written
        path = os.path.join(out_dir, f"{name}.registry.env")
        with open(path, "w") as fh:
            fh.write(render(name, prof, fmt))
        written.append(path)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="", help="write <name>.registry.env fragments here")
    ap.add_argument("--check", default="", help="compare the <name>.env files in this directory")
    ap.add_argument("--only", default="", help="comma list of profile names for --check")
    a = ap.parse_args(argv)
    if not a.out and not a.check:
        ap.error("--out and/or --check")
    rc = 0
    if a.out:
        for p in generate(a.out):
            print(f"wrote {p}")
    if a.check:
        lines, diffs = check_dir(a.check, [x for x in a.only.split(",") if x] or None)
        print("\n".join(lines))
        print(f"TOTAL DIFF {diffs}")
        rc = 1 if diffs else 0
    return rc


if __name__ == "__main__":
    sys.exit(main())
