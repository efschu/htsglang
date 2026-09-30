#!/usr/bin/env python3
"""Release profiles from the set that ACTUALLY reaches the ranks (fLLiper release, 30.09.).

A rig profile is a bash file the entrypoint sources. What reaches the launcher and the ranks is NOT its text but,
after sourcing: the final PROFILE_ARGS (with the --env-p/--env-d strings as they stand at the end -- an
`NF_ENV_D="$NF_ENV_D;..."` line after PROFILE_ARGS was built never arrived until the -ef fix, NF 30.09.), the
`_form` pairs of profile_form_env (the entrypoint exports them), the pairs of profile_instr_env (only with
HTSGLANG_INSTRUMENTS=1), and every variable the file itself EXPORTS. Plain assignments do not arrive.

  release_profile.py dump <profile.env> [--instr 0|1] [--transport bar1|nccl]      JSON of that effective set
  release_profile.py gen <target.env> <out.env> --name nf --status experimentell [--drop-arg FLAG ...]
                                                   [--owner TEXT] [--form-suffix TEXT]
      writes a FLAT release profile whose effective set equals the target's under all four combinations of
      HTSGLANG_INSTRUMENTS 0/1 x HTSGLANG_TRANSPORT bar1/nccl, except the named deliberate differences:
      the --drop-arg flags (with their value), PROFILE_NAME/STATUS/OWNER/FORM, and the release instrument
      default (HTSGLANG_INSTRUMENTS unset = 0, F8). The instrument and nccl parts of --env-p/--env-d are kept
      as conditional appends; a key they share with the base part refuses the generation (precedence would change).

Sourcing runs in a clean environment (env -i: PATH, HOME, the four knobs), with HTSGLANG_TAG and
SGLANG_WEG2_EVIDENCE_DIR set to markers that the generator turns back into the variable references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

TAG_MARK = "@@HTSGLANG_TAG@@"
EVD_MARK = "@@SGLANG_WEG2_EVIDENCE_DIR@@"
ENV_FLAGS = ("--env-p", "--env-d")

_DUMP_SH = r'''
set -u
_form(){ printf 'FORM\0%s\0%s\0' "$1" "$2"; }
say(){ :; }; refuse(){ :; }; warn(){ :; }
declare -A _pre=()
while IFS= read -r -d '' kv; do _pre[${kv%%=*}]=${kv#*=}; done < <(env -0)
cd "$(dirname "$PROF")" || exit 3
source "$PROF" >/dev/null 2>&1 || { echo "source failed" >&2; exit 4; }
for a in "${PROFILE_ARGS[@]}"; do printf 'ARG\0%s\0' "$a"; done
while IFS= read -r -d '' kv; do
  k=${kv%%=*}; v=${kv#*=}
  case "$k" in _|PWD|OLDPWD|SHLVL) continue ;; esac
  if [ -z "${_pre[$k]+x}" ] || [ "${_pre[$k]}" != "$v" ]; then printf 'EXPORT\0%s\0%s\0' "$k" "$v"; fi
done < <(env -0)
for v in $(compgen -v PROFILE_); do
  [ "$v" = PROFILE_ARGS ] && continue
  if declare -p "$v" 2>/dev/null | grep -q '^declare -a'; then
    eval "for x in \"\${$v[@]}\"; do printf 'ARR\0%s\0%s\0' \"$v\" \"\$x\"; done"
  else printf 'VAR\0%s\0%s\0' "$v" "${!v}"; fi
done
if declare -F profile_form_env >/dev/null; then ( profile_form_env ) 2>/dev/null; fi
printf 'SEP\0instr\0'
if declare -F profile_instr_env >/dev/null; then ( profile_instr_env ) 2>/dev/null; fi
'''


def dump(profile: str, instr: str = "0", transport: str = "bar1") -> dict:
    """The effective set of <profile> under the two knobs (see the module docstring)."""
    env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/root"), "PROF": os.path.abspath(profile),
           "HTSGLANG_INSTRUMENTS": instr, "HTSGLANG_TRANSPORT": transport, "HTSGLANG_TAG": TAG_MARK,
           "SGLANG_WEG2_EVIDENCE_DIR": EVD_MARK, "LC_ALL": "C"}
    r = subprocess.run(["bash", "-c", _DUMP_SH], env=env, capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"dump {profile}: bash rc {r.returncode}: {r.stderr.decode(errors='replace')[:300]}")
    tok = r.stdout.split(b"\0")
    out = {"args": [], "form": [], "instr": [], "exports": {}, "vars": {}, "arrays": {}}
    i, sect = 0, "form"
    while i < len(tok):
        t = tok[i].decode()
        if t == "ARG":
            out["args"].append(tok[i + 1].decode()); i += 2
        elif t == "FORM":
            out[sect].append([tok[i + 1].decode(), tok[i + 2].decode()]); i += 3
        elif t == "EXPORT":
            k = tok[i + 1].decode()
            if k not in ("PROF", "HTSGLANG_INSTRUMENTS", "HTSGLANG_TRANSPORT", "HTSGLANG_TAG", "SGLANG_WEG2_EVIDENCE_DIR"):
                out["exports"][k] = tok[i + 2].decode()
            i += 3
        elif t == "VAR":
            out["vars"][tok[i + 1].decode()] = tok[i + 2].decode(); i += 3
        elif t == "ARR":
            out["arrays"].setdefault(tok[i + 1].decode(), []).append(tok[i + 2].decode()); i += 3
        elif t == "SEP":
            sect = tok[i + 1].decode(); i += 2
        elif t == "":
            i += 1
        else:
            raise SystemExit(f"dump {profile}: unexpected record {t!r}")
    return out


def _env_tokens(args, flag):
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return [x for x in args[i + 1].split(";") if x]
    return None


def _without_env(args):
    out, skip = [], False
    for a in args:
        if skip:
            skip = False; out.append("<env>"); continue
        if a in ENV_FLAGS:
            skip = True
        out.append(a)
    return out


def drop_args(args, flags):
    out, i = [], 0
    while i < len(args):
        if args[i] in flags:
            i += 2 if i + 1 < len(args) and not args[i + 1].startswith("--") else 1
            continue
        out.append(args[i]); i += 1
    return out


def _q(v: str) -> str:
    """A double-quoted bash word; the markers become the variable references again."""
    s = v.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
    s = s.replace(TAG_MARK, "${HTSGLANG_TAG}").replace(EVD_MARK, "${SGLANG_WEG2_EVIDENCE_DIR:-/var/lib/htsglang/evidence}")
    return f'"{s}"'


def _extra(base, other, what):
    """Tokens of <other> not in <base>, in order. Appending them at the END is what the generated profile does; that is
    only equivalent when, in <other>, every extra token already comes AFTER every base token of the same key (the
    launcher's --env-p/--env-d: the last entry of a key wins). Otherwise refuse -- precedence would change."""
    bset = set(base)
    extra = [t for t in other if t not in bset]
    missing = [t for t in base if t not in set(other)]
    if missing:
        raise SystemExit(f"{what}: tokens of the base set missing under the variant: {missing[:5]}")
    key = lambda t: t.split("=", 1)[0]
    bad = []
    for j, t in enumerate(other):
        if t in bset:
            continue
        if any(key(x) == key(t) and x in bset for x in other[j + 1:]):
            bad.append(t)
    if bad:
        raise SystemExit(f"{what}: a base token of the same key follows the variant token (order would change): {bad[:5]}")
    return extra


def generate(target, out, name, status, drop, owner, form_suffix):
    d = {(i, t): dump(target, i, t) for i in ("0", "1") for t in ("bar1", "nccl")}
    base = d[("0", "bar1")]
    for k, v in d.items():
        if _without_env(v["args"]) != _without_env(base["args"]):
            raise SystemExit(f"non-env PROFILE_ARGS differ under {k}: the generator only knows env-string variants")
        if v["form"] != base["form"] or v["instr"] != base["instr"]:
            raise SystemExit(f"_form sets differ under {k}")
        if v["exports"] != base["exports"] or v["arrays"] != base["arrays"]:
            raise SystemExit(f"exports/arrays differ under {k}")
    envs = {}
    for flag in ENV_FLAGS:
        b = _env_tokens(base["args"], flag) or []
        ins = _extra(b, _env_tokens(d[("1", "bar1")]["args"], flag) or [], f"{flag} instruments")
        ncc = _extra(b, _env_tokens(d[("0", "nccl")]["args"], flag) or [], f"{flag} nccl")
        both = _env_tokens(d[("1", "nccl")]["args"], flag) or []
        if sorted(both) != sorted(b + ins + ncc):
            raise SystemExit(f"{flag}: instruments+nccl is not base + both extras")
        envs[flag] = (b, ins, ncc)
    tsha = hashlib.sha256(open(target, "rb").read()).hexdigest()
    V = dict(base["vars"])
    V["PROFILE_NAME"] = name
    V["PROFILE_STATUS"] = status
    if owner:
        V["PROFILE_OWNER"] = owner
    if form_suffix:
        V["PROFILE_FORM"] = V.get("PROFILE_FORM", "") + form_suffix
    args = drop_args(base["args"], set(drop))
    L = []
    w = L.append
    w("# shellcheck shell=bash")
    w(f"# RELEASE PROFILE '{name}' -- GENERATED by docker/flliper/release_profile.py gen ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})")
    w(f"# from {os.path.abspath(target)}")
    w(f"#   sha256 {tsha}")
    w("# = the set that ACTUALLY reaches the ranks after sourcing the target: final PROFILE_ARGS (incl. the final")
    w("#   --env-p/--env-d strings), the _form pairs, the instrument pairs, and the target's own exports.")
    w("# Deliberate differences to the target (checked by test_flliper_release_0930.py::TestReleaseProfileNF):")
    w(f"#   dropped launcher flags: {' '.join(drop) or 'none'}; PROFILE_NAME/STATUS/OWNER/FORM; instruments default OFF")
    w("#   (HTSGLANG_INSTRUMENTS unset = 0, F8; the target defaulted to 1).")
    w("# DRAFT until the owner seat releases it (then PROFILE_STATUS=abgenommen). Do not edit by hand -- regenerate.")
    w("")
    for k in sorted(V):
        w(f"{k}={_q(V[k])}")
    for k in sorted(base["arrays"]):
        w(f"{k}=(")
        for x in base["arrays"][k]:
            w(f"  {_q(x)}")
        w(")")
    w("")
    w("# group env, base part + the conditional instrument / nccl parts (as the target appends them)")
    for flag, var in (("--env-p", "REL_ENV_P"), ("--env-d", "REL_ENV_D")):
        b, ins, ncc = envs[flag]
        w(f"{var}={_q(';'.join(b))}")
        if ins:
            w(f'if [ "${{HTSGLANG_INSTRUMENTS:-0}}" = 1 ]; then {var}="${var};"{_q(";".join(ins))}; fi')
        if ncc:
            w(f'if [ "${{HTSGLANG_TRANSPORT:-bar1}}" = nccl ]; then {var}="${var};"{_q(";".join(ncc))}; fi')
    w("")
    w("PROFILE_ARGS=(")
    skip = None
    for a in args:
        if skip:
            w(f'  "${skip}"'); skip = None; continue
        if a in ENV_FLAGS:
            w(f"  {a}"); skip = "REL_ENV_P" if a == "--env-p" else "REL_ENV_D"; continue
        w(f"  {_q(a)}")
    w(")")
    w("")
    for k in sorted(base["exports"]):
        w(f"export {k}={_q(base['exports'][k])}")
    w("")
    w("profile_form_env() {")
    for k, v in base["form"]:
        w(f"  _form {k} {_q(v)}")
    w("  :")
    w("}")
    w("profile_instr_env() {")
    for k, v in base["instr"]:
        w(f"  _form {k} {_q(v)}")
    w("  :")
    w("}")
    open(out, "w").write("\n".join(L) + "\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("dump")
    a.add_argument("profile"); a.add_argument("--instr", default="0"); a.add_argument("--transport", default="bar1")
    g = sub.add_parser("gen")
    g.add_argument("target"); g.add_argument("out"); g.add_argument("--name", required=True)
    g.add_argument("--status", default="experimentell"); g.add_argument("--drop-arg", action="append", default=[])
    g.add_argument("--owner", default=""); g.add_argument("--form-suffix", default="")
    ns = ap.parse_args(argv)
    if ns.cmd == "dump":
        json.dump(dump(ns.profile, ns.instr, ns.transport), sys.stdout, indent=1)
        print()
    else:
        print(generate(ns.target, ns.out, ns.name, ns.status, ns.drop_arg, ns.owner, ns.form_suffix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
