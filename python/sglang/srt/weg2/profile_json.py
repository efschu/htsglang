"""PROFIL-EDITOR S1: the server profile as JSON (``flliper.server/1``), with a .env round trip.

A release profile today is a bash program (``profiles_release/<name>.env``: sourced by the entrypoint, with
``PROFILE_*`` variables, ``PROFILE_ARGS=(...)`` and the ``profile_form_env`` / ``profile_instr_env`` functions
that call ``_form NAME VALUE``).  The editor needs the same facts as DATA.  This module

* **evaluates** a ``.env`` the way the entrypoint consumes it (bash, ``_form`` stubbed, nothing started) and
  takes the RESOLVED facts -- ``PROFILE_*`` scalars and arrays, ``PROFILE_ARGS``, the variables the file exports
  at its top level, and the ``_form`` calls of both functions -- ``import_env``;
* **renders** those facts back as a ``.env`` in today's dialect -- ``render_env``;
* proves the round trip: ``dump(render(import(f))) == dump(f)`` for ``HTSGLANG_INSTRUMENTS`` 0 and 1 --
  ``roundtrip_check`` (the golden over all release profiles, ``test_profile_json_1003``).

What survives and what does not: comments and the file's own bash logic (loops, ``source``, function
redefinitions) are gone, because the RESULT of that logic is what the entrypoint reads.  The rendered file is
therefore flat.  Until the round trip is green for every release profile the ``.env`` stays the source
(Nutzer-Setzung E3); the JSON never replaces it earlier.

PURE: stdlib only (the dashboard loads this file by path, like ``card_identity``); the only side effect is the
``bash`` subprocess of :func:`dump_env`.  Nothing here starts a server, a container or touches a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "flliper.server/1"

#: bash that evaluates one profile.  $1 = file, $2 = HTSGLANG_INSTRUMENTS (the profiles read it while being
#: sourced).  Records are NUL separated triples ``kind, a, b`` so a value may hold any character but NUL.
_DUMP = r'''
set +e +u
export HTSGLANG_INSTRUMENTS="$2"
export HTSGLANG_TAG='@@HTSGLANG_TAG@@' SGLANG_WEG2_EVIDENCE_DIR='@@SGLANG_WEG2_EVIDENCE_DIR@@' SGLANG_WEG2_GPU_ARB='@@SGLANG_WEG2_GPU_ARB@@'
declare -A _B
while IFS= read -r -d '' _kv; do _B["${_kv%%=*}"]="${_kv#*=}"; done < <(env -0)
cd "$(dirname "$1")" || exit 2
source "$1" >/dev/null 2>&1
printf 'RC\0%s\0\0' "$?"
for _v in $(compgen -v | grep '^PROFILE_' | sort); do
  if [[ "$(declare -p "$_v" 2>/dev/null)" == "declare -a"* ]]; then
    declare -n _r="$_v"
    printf 'ARRN\0%s\0\0' "$_v"
    for _e in "${_r[@]}"; do printf 'ARR\0%s\0%s\0' "$_v" "$_e"; done
    unset -n _r
  else
    printf 'VAR\0%s\0%s\0' "$_v" "${!_v}"
  fi
done
while IFS= read -r -d '' _kv; do
  _k="${_kv%%=*}"; _x="${_kv#*=}"
  case "$_k" in _*|OLDPWD|PWD|SHLVL|BASH_FUNC_*|HTSGLANG_INSTRUMENTS|HTSGLANG_TAG|SGLANG_WEG2_EVIDENCE_DIR|SGLANG_WEG2_GPU_ARB) continue ;; esac
  if [ "${_B[$_k]+x}" != x ] || [ "${_B[$_k]}" != "$_x" ]; then printf 'EXP\0%s\0%s\0' "$_k" "$_x"; fi
done < <(env -0)
_FK=FORM
_form() { printf '%s\0%s\0%s\0' "$_FK" "$1" "$2"; }
declare -F profile_form_env >/dev/null && profile_form_env
_FK=INSTR
declare -F profile_instr_env >/dev/null && profile_instr_env
exit 0
'''

#: the keys that may differ between HTSGLANG_INSTRUMENTS=0 and 1 (nf*.env put NF_ENV_*_INSTR into --env-p/-d)
_VARIANT_KEYS = ("vars", "exports", "args", "form")


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def doc_id(doc: Mapping) -> str:
    """sha256 of the canonical document without its ``id`` (a saved profile names the exact content)."""
    return "sha256:" + hashlib.sha256(_canonical({k: v for k, v in doc.items() if k != "id"}).encode()).hexdigest()


# ---------------------------------------------------------------------------
# runtime placeholders
#
# A profile may use values the ENTRYPOINT sets before it sources the profile (the boot tag, the evidence and arb directories):
# ``_form SGLANG_MOE_COLD_TIER_INSTANCE "${HTSGLANG_TAG}"``.  Evaluated at import time they would be frozen to nothing.  The dump therefore
# runs with a sentinel for each of them; a value that carries a sentinel is stored with the placeholder ``${NAME}`` and written back as an
# expansion the entrypoint resolves at its own run time.  (Caller SWITCHES with a default -- ``${HTSGLANG_DRAFT:-x}`` -- cannot be told from
# a constant by evaluation; they are listed in ``meta.caller_switches`` and are baked with their default.)

PLACEHOLDERS = ("HTSGLANG_TAG", "SGLANG_WEG2_EVIDENCE_DIR", "SGLANG_WEG2_GPU_ARB")
_SENTINEL = {n: "@@%s@@" % n for n in PLACEHOLDERS}


def _to_placeholders(v: str) -> str:
    for n, sv in _SENTINEL.items():
        if sv in v:
            v = v.replace(sv, "${%s}" % n)
    return v


def _from_placeholders(v: str) -> str:
    for n, sv in _SENTINEL.items():
        if "${%s}" % n in v:
            v = v.replace("${%s}" % n, sv)
    return v


def _walk_strings(obj, fn):
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, list):
        return [_walk_strings(x, fn) for x in obj]
    if isinstance(obj, dict):
        return {k: _walk_strings(v, fn) for k, v in obj.items()}
    return obj


def caller_switches(path: str, _depth: int = 0, _seen=None) -> List[str]:
    """Names of ``HTSGLANG_*`` / ``FLLIPER_*`` variables the profile (and the profiles it sources, within its directory) READS and that the
    entrypoint does not set for it: the profile's caller switches, baked with their default by an import."""
    seen = _seen if _seen is not None else set()
    out: set = set()
    if path in seen or _depth > 3:
        return []
    seen.add(path)
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return []
    for ln in text.splitlines():
        if ln.lstrip().startswith("#"):
            continue
        i = 0
        while True:
            i = ln.find("$", i)
            if i < 0:
                break
            j = i + 1
            if j < len(ln) and ln[j] == "{":
                j += 1
            k = j
            while k < len(ln) and (ln[k].isalnum() or ln[k] == "_"):
                k += 1
            name = ln[j:k]
            if name.startswith(("HTSGLANG_", "FLLIPER_")) and name not in ("HTSGLANG_TAG", "HTSGLANG_INSTRUMENTS"):
                out.add(name)
            i = k
        if "source " in ln and ".env" in ln:
            base = ln.split("source ", 1)[1].strip().strip('"').strip("'")
            tail = base.rsplit("/", 1)[-1].strip('"').strip("'")
            if tail.endswith(".env"):
                out |= set(caller_switches(os.path.join(os.path.dirname(path), tail), _depth + 1, seen))
    return sorted(out)


# ---------------------------------------------------------------------------
# evaluation

def _run_dump(path: str, instruments: str) -> str:
    p = subprocess.run(["bash", "-c", _DUMP, "dump", path, instruments], capture_output=True, text=True,
                       errors="surrogateescape", timeout=60)
    return p.stdout


def dump_env(path: str, instruments: str = "0", runner=None) -> Dict[str, object]:
    """What the entrypoint would read from ``path``: the resolved facts, as plain data.

    ``{"rc", "vars": {name: str}, "arrays": {name: [str]}, "exports": [[name, value]], "form": [[n, v]],
    "instr": [[n, v]]}`` -- ``vars``/``arrays`` only ``PROFILE_*``."""
    raw = (runner or _run_dump)(os.path.abspath(path), instruments)
    parts = raw.split("\0")
    out: Dict[str, object] = {"rc": None, "vars": {}, "arrays": {}, "exports": [], "form": [], "instr": []}
    for i in range(0, len(parts) - 2, 3):
        k, a, b = parts[i], parts[i + 1], parts[i + 2]
        if k == "RC":
            out["rc"] = a
        elif k == "VAR":
            out["vars"][a] = b
        elif k == "ARRN":
            out["arrays"].setdefault(a, [])
        elif k == "ARR":
            out["arrays"].setdefault(a, []).append(b)
        elif k == "EXP":
            out["exports"].append([a, b])
        elif k == "FORM":
            out["form"].append([a, b])
        elif k == "INSTR":
            out["instr"].append([a, b])
    return out


# ---------------------------------------------------------------------------
# PROFILE_ARGS <-> entries

def _takes_value(flag: str, specs: Optional[Mapping[str, Mapping]]) -> Optional[int]:
    """How many value tokens ``flag`` takes (0 = bare, 1, ...), or None when unknown / variable."""
    if not specs or flag not in specs:
        return None
    s = specs[flag]
    if s.get("bare"):
        return 0
    n = s.get("nargs")
    if n is None:
        return 1
    if isinstance(n, int):
        return n
    return None


def parse_args(tokens: Sequence[str], specs: Optional[Mapping[str, Mapping]] = None) -> List[Dict[str, object]]:
    """PROFILE_ARGS tokens -> ordered entries ``{"flag", "values", "eq"?}`` (``{"token"}`` for a leading
    positional).  Lossless: :func:`render_args` returns exactly the same tokens.  With launcher ``specs`` a flag
    that takes a value takes the next token even when it starts with ``--`` (``--extra-p "--max-...=2"``)."""
    out: List[Dict[str, object]] = []
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t.startswith("--") and len(t) > 2:
            if "=" in t:
                flag, val = t.split("=", 1)
                out.append({"flag": flag, "values": [val], "eq": True})
                i += 1
                continue
            take = _takes_value(t, specs)
            vals: List[str] = []
            j = i + 1
            if take is None:
                while j < n and not (tokens[j].startswith("--") and len(tokens[j]) > 2):
                    vals.append(tokens[j])
                    j += 1
            else:
                while j < n and len(vals) < take:
                    vals.append(tokens[j])
                    j += 1
            out.append({"flag": t, "values": vals})
            i = j
            continue
        out.append({"token": t})
        i += 1
    return out


def render_args(entries: Iterable[Mapping]) -> List[str]:
    out: List[str] = []
    for e in entries:
        if "token" in e:
            out.append(str(e["token"]))
        elif e.get("eq") and len(e.get("values") or []) == 1:
            out.append("%s=%s" % (e["flag"], e["values"][0]))
        else:
            out.append(str(e["flag"]))
            out.extend(str(v) for v in (e.get("values") or []))
    return out


# ---------------------------------------------------------------------------
# document

def _vars_entries(vars_: Mapping[str, str], arrays: Mapping[str, Sequence[str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for name in sorted(set(vars_) | set(arrays)):
        if name == "PROFILE_ARGS":
            continue
        if name in arrays:
            out.append({"name": name, "values": list(arrays[name])})
        else:
            out.append({"name": name, "value": vars_[name]})
    return out


def _pairs(rows: Iterable[Sequence[str]]) -> List[Dict[str, str]]:
    return [{"name": r[0], "value": r[1]} for r in rows]


def _facts(raw: Mapping, specs) -> Dict[str, object]:
    return {"vars": _vars_entries(raw["vars"], raw["arrays"]),
            "exports": _pairs(raw["exports"]),
            "args": parse_args(raw["arrays"].get("PROFILE_ARGS", []), specs),
            "form": _pairs(raw["form"]),
            "instr": _pairs(raw["instr"])}


def import_env(path: str, specs: Optional[Mapping[str, Mapping]] = None, runner=None) -> Dict[str, object]:
    """``.env`` -> ``flliper.server/1``.  ``instruments1`` holds only what differs under
    ``HTSGLANG_INSTRUMENTS=1`` (the three nf*.env put instrument env into ``--env-p/-d``)."""
    r0 = dump_env(path, "0", runner)
    r1 = dump_env(path, "1", runner)
    f0, f1 = _walk_strings(_facts(r0, specs), _to_placeholders), _walk_strings(_facts(r1, specs), _to_placeholders)
    var0 = {v["name"]: v for v in f0["vars"]}
    doc: Dict[str, object] = {
        "schema": SCHEMA,
        "name": str((var0.get("PROFILE_NAME") or {}).get("value") or os.path.basename(path)[:-4]),
        "line": str((var0.get("PROFILE_LINE") or {}).get("value") or ""),
        "source": {"kind": "env", "file": os.path.basename(path), "sha256": _file_sha(path),
                   "rc": r0["rc"]},
        "vars": f0["vars"], "exports": f0["exports"], "args": f0["args"], "form": f0["form"], "instr": f0["instr"],
        "meta": {"origins": {}, "planner": {}, "notes": [], "caller_switches": caller_switches(os.path.abspath(path))},
    }
    delta = {k: f1[k] for k in _VARIANT_KEYS if f1[k] != f0[k]}
    if delta:
        doc["instruments1"] = delta
    if f1["instr"] != f0["instr"]:
        doc["instr"] = f0["instr"]          # profile_instr_env does not depend on the variable; keep the 0 reading
    doc["id"] = doc_id(doc)
    return doc


def _file_sha(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            return "sha256:" + hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# rendering

def _q(v: str) -> str:
    """Shell-quote ``v``; a ``${PLACEHOLDER}`` stays an expansion (double quotes), everything else is literal."""
    if "${" not in v:
        return shlex.quote(v)
    parts: List[str] = []
    rest = v
    while rest:
        hit = [(rest.find("${%s}" % n), n) for n in PLACEHOLDERS if rest.find("${%s}" % n) >= 0]
        if not hit:
            parts.append(shlex.quote(rest))
            break
        i, n = min(hit)
        if i:
            parts.append(shlex.quote(rest[:i]))
        parts.append('"${%s}"' % n)
        rest = rest[i + len(n) + 3:]
    return "".join(parts)


def _assign_block(f: Mapping[str, object], indent: str) -> List[str]:
    lines: List[str] = []
    for v in f["vars"]:
        if "values" in v:
            lines.append("%s%s=(%s)" % (indent, v["name"], " ".join(_q(x) for x in v["values"])))
        else:
            lines.append("%s%s=%s" % (indent, v["name"], _q(str(v["value"]))))
    for e in f["exports"]:
        lines.append("%sexport %s=%s" % (indent, e["name"], _q(str(e["value"]))))
    toks = render_args(f["args"])
    if toks:
        lines.append("%sPROFILE_ARGS=(" % indent)
        lines.extend("%s  %s" % (indent, _q(t)) for t in toks)
        lines.append("%s)" % indent)
    else:
        lines.append("%sPROFILE_ARGS=()" % indent)
    return lines


def render_env(doc: Mapping, header: bool = True) -> str:
    """``flliper.server/1`` -> a ``.env`` the entrypoint sources as it sources a hand-written one."""
    if doc.get("schema") != SCHEMA:
        raise ValueError("schema %r, expected %s" % (doc.get("schema"), SCHEMA))
    lines = ["# shellcheck shell=bash"]
    if header:
        lines += ["# GENERATED from %s %r by sglang.srt.weg2.profile_json (python -m sglang.srt.weg2.profile_json render)."
                  % (SCHEMA, doc.get("name")),
                  "# The JSON profile is the source; do not edit this file, edit the profile and render again.",
                  "# Flat form: the RESULT of the profile's logic (comments and bash control flow of a hand-written .env are not carried)."]
        src = doc.get("source") or {}
        if src.get("file"):
            lines.append("# Imported from %s (%s)." % (src.get("file"), src.get("sha256") or "no hash"))
    base = {k: doc.get(k) or [] for k in ("vars", "exports", "args")}
    lines += _assign_block(base, "")
    i1 = doc.get("instruments1") or {}
    if i1:
        alt = dict(base)
        alt.update({k: i1[k] for k in ("vars", "exports", "args") if k in i1})
        lines.append('if [ "${HTSGLANG_INSTRUMENTS:-0}" = "1" ]; then')
        lines += _assign_block(alt, "  ")
        lines.append("fi")
    for fn, key in (("profile_form_env", "form"), ("profile_instr_env", "instr")):
        rows = list(doc.get(key) or [])
        if key == "form" and "form" in i1:
            # the variant only matters for what the profile exports/sets while sourced; the function bodies are
            # identical in the three NF profiles (checked by the golden)
            pass
        lines.append("%s() {" % fn)
        if rows:
            lines.extend("  _form %s %s" % (r["name"], _q(str(r["value"]))) for r in rows)
        else:
            lines.append("  :")
        lines.append("}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# the golden

def effective(path: str, runner=None) -> Dict[str, object]:
    """Both evaluations of one file (HTSGLANG_INSTRUMENTS 0 and 1), without the return code of ``source``."""
    out = {}
    for ins in ("0", "1"):
        r = dump_env(path, ins, runner)
        r.pop("rc", None)
        out[ins] = r
    return out


def expected_effective(doc: Mapping) -> Dict[str, object]:
    """What :func:`effective` must return for the ``.env`` that :func:`render_env` writes for ``doc`` -- computed from the
    document alone (the check of an EDITED profile, where no original file exists)."""
    i1 = doc.get("instruments1") or {}
    out: Dict[str, object] = {}
    for ins in ("0", "1"):
        src = {k: (i1[k] if ins == "1" and k in i1 else doc.get(k) or []) for k in ("vars", "exports", "args", "form")}
        vars_: Dict[str, str] = {}
        arrays: Dict[str, List[str]] = {"PROFILE_ARGS": render_args(src["args"])}
        for v in src["vars"]:
            if "values" in v:
                arrays[v["name"]] = [str(x) for x in v["values"]]
            else:
                vars_[v["name"]] = str(v["value"])
        out[ins] = {"vars": vars_, "arrays": arrays, "exports": [[e["name"], str(e["value"])] for e in src["exports"]],
                    "form": [[r["name"], str(r["value"])] for r in src["form"]],
                    "instr": [[r["name"], str(r["value"])] for r in (doc.get("instr") or [])]}
        out[ins] = _walk_strings(out[ins], _from_placeholders)
    return out


def diff_effective(a: Mapping, b: Mapping) -> List[str]:
    """Human readable differences between two :func:`effective` results (empty = identical)."""
    rows: List[str] = []
    for ins in ("0", "1"):
        for key in ("vars", "arrays", "exports", "form", "instr"):
            x, y = a[ins][key], b[ins][key]
            if key == "exports":                       # the environment table has no order worth comparing
                x, y = sorted(x), sorted(y)
            if x == y:
                continue
            if isinstance(x, dict):
                for k in sorted(set(x) | set(y)):
                    if x.get(k) != y.get(k):
                        rows.append("INSTRUMENTS=%s %s %s: %r != %r" % (ins, key, k, _short(x.get(k)), _short(y.get(k))))
            else:
                rows.append("INSTRUMENTS=%s %s: %d rows != %d rows, first difference %r" % (
                    ins, key, len(x), len(y), _first_diff(x, y)))
    return rows


def _short(v, n: int = 160) -> str:
    s = repr(v)
    return s if len(s) <= n else s[:n] + "..."


def _first_diff(x: Sequence, y: Sequence):
    for i in range(max(len(x), len(y))):
        xi = x[i] if i < len(x) else None
        yi = y[i] if i < len(y) else None
        if xi != yi:
            return (i, xi, yi)
    return None


def roundtrip_check(path: str, specs: Optional[Mapping] = None, runner=None, tmpdir: Optional[str] = None) -> List[str]:
    """``effective(f) == effective(render(import(f)))``.  Returns the differences (empty = golden holds)."""
    import tempfile

    doc = import_env(path, specs, runner)
    text = render_env(doc)
    with tempfile.TemporaryDirectory(dir=tmpdir) as d:
        p = os.path.join(d, os.path.basename(path))
        with open(p, "w", encoding="utf-8", errors="surrogateescape") as fh:
            fh.write(text)
        return diff_effective(effective(path, runner), effective(p, runner))


# ---------------------------------------------------------------------------
# the editor's view: rows, origins, edits

ORIGIN_PROFIL, ORIGIN_NUTZER, ORIGIN_PLANER, ORIGIN_DEFAULT = "profil", "nutzer", "planer", "default"
ORIGIN_LABEL = {ORIGIN_PROFIL: "Profile", ORIGIN_NUTZER: "User", ORIGIN_PLANER: "Planner", ORIGIN_DEFAULT: "Default"}

_GROUP_SCOPE = {"--env-p": ("env", "P"), "--env-d": ("env", "D"), "--extra-p": ("extra", "P"), "--extra-d": ("extra", "D")}


def _dedup_key(base: str, seen: Dict[str, int]) -> str:
    seen[base] = seen.get(base, 0) + 1
    return base if seen[base] == 1 else "%s#%d" % (base, seen[base])


def _env_items(value: str) -> List[Tuple[str, str]]:
    out = []
    for item in [x for x in str(value).split(";") if x.strip()]:
        if "=" in item:
            k, v = item.split("=", 1)
            out.append((k.strip(), v.strip()))
        else:
            out.append((item.strip(), ""))
    return out


def _extra_entries(value: str, specs=None) -> List[Dict[str, object]]:
    try:
        toks = shlex.split(str(value))
    except ValueError:
        toks = str(value).split()
    return parse_args(toks, specs)


def _row(key: str, kind: str, scope: str, name: str, value: str, bare: bool = False, multi: bool = False) -> Dict[str, object]:
    return {"key": key, "kind": kind, "scope": scope, "name": name, "value": value, "bare": bare, "multi": multi}


def rows(doc: Mapping, specs: Optional[Mapping] = None) -> List[Dict[str, object]]:
    """The flat list of values of a profile, each with a STABLE key (``var:`` ``export:`` ``flag:`` ``env:P:`` ``extra:D:``
    ``form:`` ``instr:``; a repeated name gets ``#2``...).  Derived from the token-lossless document; editing goes through
    :func:`apply_edits`, which writes back into the owning token."""
    out: List[Dict[str, object]] = []
    seen: Dict[str, int] = {}
    for v in doc.get("vars") or []:
        multi = "values" in v
        val = " ".join(v["values"]) if multi else str(v.get("value", ""))
        out.append(_row(_dedup_key("var:%s" % v["name"], seen), "var", "profile", v["name"], val, multi=multi))
    for e in doc.get("exports") or []:
        out.append(_row(_dedup_key("export:%s" % e["name"], seen), "env", "all", e["name"], str(e["value"])))
    for e in doc.get("args") or []:
        if "token" in e:
            out.append(_row(_dedup_key("token:%s" % e["token"], seen), "flag", "launcher", str(e["token"]), "", bare=True))
            continue
        flag = str(e["flag"])
        vals = list(e.get("values") or [])
        if flag in _GROUP_SCOPE and vals:
            kind, grp = _GROUP_SCOPE[flag]
            text = " ".join(vals) if len(vals) > 1 else vals[0]
            if kind == "env":
                for k, v in _env_items(text):
                    out.append(_row(_dedup_key("env:%s:%s" % (grp, k), seen), "env", grp, k, v))
            else:
                for sub in _extra_entries(text, specs):
                    if "token" in sub:
                        out.append(_row(_dedup_key("extra:%s:%s" % (grp, sub["token"]), seen), "flag", grp, str(sub["token"]), "", bare=True))
                    else:
                        sv = list(sub.get("values") or [])
                        out.append(_row(_dedup_key("extra:%s:%s" % (grp, sub["flag"]), seen), "flag", grp, str(sub["flag"]),
                                        " ".join(sv), bare=not sv))
            continue
        out.append(_row(_dedup_key("flag:%s" % flag, seen), "flag", "launcher", flag, " ".join(vals), bare=not vals))
    for key in ("form", "instr"):
        for e in doc.get(key) or []:
            out.append(_row(_dedup_key("%s:%s" % (key, e["name"]), seen), "env", key, e["name"], str(e["value"])))
    return out


def _split_key(key: str) -> Tuple[str, str, str, int]:
    """``extra:P:--x#2`` -> (``extra``, ``P``, ``--x``, 2)."""
    nth = 1
    if "#" in key:
        key, n = key.rsplit("#", 1)
        nth = int(n) if n.isdigit() else 1
    parts = key.split(":", 2)
    if parts[0] in ("env", "extra"):
        return parts[0], parts[1], parts[2], nth
    return parts[0], "", ":".join(parts[1:]), nth


def _nth_arg(args: List[Dict], flag: str, nth: int) -> Optional[int]:
    seen = 0
    for i, e in enumerate(args):
        if e.get("flag") == flag or e.get("token") == flag:
            seen += 1
            if seen == nth:
                return i
    return None


def _rewrite_group(args: List[Dict], flag: str, items: List[Tuple[str, str]], is_env: bool, extra_entries=None) -> None:
    """Write the env items / extra entries of one group back into ``args`` (first token of that flag; others dropped)."""
    idxs = [i for i, e in enumerate(args) if e.get("flag") == flag]
    if is_env:
        text = ";".join("%s=%s" % (k, v) for k, v in items)
    else:
        text = shlex.join(render_args(extra_entries or []))
    if not text:
        for i in reversed(idxs):
            del args[i]
        return
    if idxs:
        args[idxs[0]] = {"flag": flag, "values": [text]}
        for i in reversed(idxs[1:]):
            del args[i]
    else:
        args.append({"flag": flag, "values": [text]})


def _set_in_doc(doc: Dict, key: str, value: Optional[str], specs=None) -> bool:
    """Set (``value`` string) or delete (None) the value at ``key``; add it when absent.  True if the document changed."""
    kind, scope, name, nth = _split_key(key)
    args: List[Dict] = doc.setdefault("args", [])
    if kind == "var":
        for v in doc.setdefault("vars", []):
            if v["name"] == name:
                if value is None:
                    doc["vars"].remove(v)
                elif "values" in v:
                    v["values"] = shlex.split(value)
                else:
                    v["value"] = value
                return True
        if value is not None:
            doc["vars"].append({"name": name, "value": value})
            doc["vars"].sort(key=lambda x: x["name"])
            return True
        return False
    if kind in ("export", "form", "instr"):
        lst = doc.setdefault("exports" if kind == "export" else kind, [])
        hits = [e for e in lst if e["name"] == name]
        if hits and len(hits) >= nth:
            if value is None:
                lst.remove(hits[nth - 1])
            else:
                hits[nth - 1]["value"] = value
            return True
        if value is not None:
            lst.append({"name": name, "value": value})
            return True
        return False
    if kind == "flag":
        i = _nth_arg(args, name, nth)
        spec_bare = bool((specs or {}).get(name, {}).get("bare"))
        if i is None:
            if value is None:
                return False
            if value == "" or (spec_bare and value in ("1", "on", "true")):
                args.append({"flag": name, "values": []})
            else:
                args.append({"flag": name, "values": [value]})
            return True
        if value is None:
            del args[i]
        elif value == "" or (spec_bare and value in ("1", "on", "true")):
            args[i] = {"flag": name, "values": []}
        elif spec_bare and value in ("0", "off", "false"):
            del args[i]
        else:
            args[i] = {"flag": name, "values": [value]}
        return True
    if kind in ("env", "extra"):
        gflag = "--env-%s" % scope.lower() if kind == "env" else "--extra-%s" % scope.lower()
        cur = [e for e in args if e.get("flag") == gflag]
        texts = [(e.get("values") or [""])[0] for e in cur]
        if kind == "env":
            items = _env_items(";".join(texts))
            idx = [i for i, (k, _v) in enumerate(items) if k == name]
            if idx and len(idx) >= nth:
                if value is None:
                    del items[idx[nth - 1]]
                else:
                    items[idx[nth - 1]] = (name, value)
            elif value is not None:
                items.append((name, value))
            else:
                return False
            _rewrite_group(args, gflag, items, True)
            return True
        entries = _extra_entries(" ".join(texts), specs)
        pos = [i for i, e in enumerate(entries) if e.get("flag") == name or e.get("token") == name]
        if pos and len(pos) >= nth:
            if value is None:
                del entries[pos[nth - 1]]
            elif value == "":
                entries[pos[nth - 1]] = {"flag": name, "values": []}
            else:
                entries[pos[nth - 1]] = {"flag": name, "values": [value]}
        elif value is not None:
            entries.append({"flag": name, "values": [value] if value != "" else []})
        else:
            return False
        _rewrite_group(args, gflag, [], False, entries)
        return True
    raise ValueError("unknown key %r" % key)


def apply_edits(doc: Mapping, edits: Sequence[Mapping], specs: Optional[Mapping] = None) -> Dict[str, object]:
    """New document with the edits applied; every touched key gets origin ``nutzer`` -- except ``reset`` ops, which
    restore ``planer`` / ``profil`` values.  Edit ops: ``{"key", "op": "set", "value"}`` | ``{"key", "op": "delete"}`` |
    ``{"key", "op": "reset", "to": "profil"|"planer"}``.  ``meta.profile_values`` / ``meta.planner`` hold the targets."""
    import copy

    new = copy.deepcopy(dict(doc))
    meta = new.setdefault("meta", {"origins": {}, "planner": {}, "notes": []})
    origins = meta.setdefault("origins", {})
    pv = meta.setdefault("profile_values", {})
    planner = meta.setdefault("planner", {})
    for ed in edits:
        key, op = str(ed["key"]), str(ed.get("op", "set"))
        if op == "set":
            _set_in_doc(new, key, str(ed.get("value", "")), specs)
            origins[key] = ORIGIN_NUTZER
        elif op == "delete":
            _set_in_doc(new, key, None, specs)
            origins.pop(key, None)
            if key in pv:
                origins[key] = ORIGIN_NUTZER           # removed from a profile that carries it: a user decision
        elif op == "reset":
            to = ed.get("to", "profil")
            src = planner if to == "planer" else pv
            if key in src:
                _set_in_doc(new, key, src[key], specs)
                origins[key] = ORIGIN_PLANER if to == "planer" else ORIGIN_PROFIL
            else:
                _set_in_doc(new, key, None, specs)
                origins.pop(key, None)
        else:
            raise ValueError("unknown edit op %r" % op)
    new["id"] = doc_id(new)
    return new


def freeze_profile_values(doc: Dict, specs: Optional[Mapping] = None) -> Dict[str, object]:
    """Remember the loaded values as the PROFILE values (the target of "zurueck auf Profil") -- once, at load."""
    meta = doc.setdefault("meta", {"origins": {}, "planner": {}, "notes": []})
    if "profile_values" not in meta:
        meta["profile_values"] = {r["key"]: r["value"] for r in rows(doc, specs)}
    return doc


def explain_row(r: Mapping, catalog: Optional[Mapping], comments: Optional[Mapping]) -> Dict[str, object]:
    """The explanation of one row from its sources (curated, code, profile comment), the dependencies and the verdict."""
    name = str(r["name"])
    ent = (catalog or {}).get(name)
    parts: List[Dict[str, str]] = []
    out: Dict[str, object] = {"status": "unerklaert", "parts": parts, "depends": [], "gain": "", "cost": "", "group": "", "level": "experte",
                              "planner_derived": False, "source": None, "default": None, "choices": None}
    if ent:
        if ent.get("text"):
            parts.append({"kind": "erklaert" if ent.get("status") == "erklaert" else "kuratiert", "text": str(ent["text"]),
                          "source": "profile_catalog_curated.py"})
        if ent.get("help"):
            src = ent.get("source") or {}
            parts.append({"kind": "code", "text": str(ent["help"])[:1400],
                          "source": "%s:%s" % (src.get("file", ""), src.get("line", ""))})
        out.update({"gain": ent.get("gain", ""), "cost": ent.get("cost", ""), "group": ent.get("group", ""),
                    "level": ent.get("level", "experte"), "planner_derived": bool(ent.get("planner_derived")),
                    "source": ent.get("source"), "default": ent.get("default"), "choices": ent.get("choices"),
                    "depends": [dict(d) for d in ent.get("depends", [])]})
    c = (comments or {}).get(name)
    if c:
        parts.append({"kind": "profil", "text": str(c["text"]), "source": str(c.get("source", ""))})
    if parts:
        if ent and ent.get("text"):
            out["status"] = "erklaert" if ent.get("status") == "erklaert" else "kuratiert"
        elif ent and ent.get("help"):
            out["status"] = "geerntet"
        else:
            out["status"] = "profil-kommentar"
    return out


def view(doc: Mapping, catalog: Optional[Mapping] = None, comments: Optional[Mapping] = None,
         specs: Optional[Mapping] = None) -> Dict[str, object]:
    """Everything the editor shows for a document: rows with value, origin, profile/planner value, explanation and
    dependencies (``present`` = the other value is set in this profile), plus the coverage."""
    meta = doc.get("meta") or {}
    origins, pv, planner = meta.get("origins") or {}, meta.get("profile_values") or {}, meta.get("planner") or {}
    rs = rows(doc, specs)
    present = {r["name"] for r in rs} | {r["key"] for r in rs}
    out = []
    for r in rs:
        key = r["key"]
        ex = explain_row(r, catalog, comments)
        for d in ex["depends"]:
            # a refusal code is no value of the profile: "set in this profile" does not apply (None), the chip points at the register
            d["present"] = None if d.get("to_kind") == "ablehnung" else d["to"] in present
        origin = origins.get(key) or (ORIGIN_PROFIL if (key in pv or not pv) else ORIGIN_NUTZER)
        row = dict(r)
        row.update({"origin": origin, "origin_label": ORIGIN_LABEL[origin],
                    "profile_value": pv.get(key), "planner_value": planner.get(key),
                    "changed": bool(pv) and (key not in pv or pv[key] != r["value"]),
                    "explain": ex})
        out.append(row)
    n = len(out)
    cov = {"rows": n, "kuratiert": sum(1 for x in out if x["explain"]["status"] == "kuratiert"),
           "maschinell": sum(1 for x in out if x["explain"]["status"] == "erklaert"),
           "geerntet": sum(1 for x in out if x["explain"]["status"] == "geerntet"),
           "profil_kommentar": sum(1 for x in out if x["explain"]["status"] == "profil-kommentar"),
           "unerklaert": sum(1 for x in out if x["explain"]["status"] == "unerklaert"),
           "geaendert": sum(1 for x in out if x["changed"])}
    cov["erklaert"] = n - cov["unerklaert"]
    keys = {x["key"] for x in out}
    planner_only = [{"key": k, "value": v} for k, v in planner.items() if k not in keys]
    removed = [{"key": k, "value": v} for k, v in pv.items() if k not in keys]
    return {"rows": out, "coverage": cov, "planner_only": planner_only, "removed": removed}


# ---------------------------------------------------------------------------
# lookups the editor uses

def args_dict(doc: Mapping) -> Dict[str, str]:
    """flag -> last value (argparse: the last occurrence wins; repeatable flags keep all in the entries)."""
    out: Dict[str, str] = {}
    for e in doc.get("args") or []:
        if "flag" in e:
            v = e.get("values") or []
            out[str(e["flag"])] = " ".join(v) if v else ""
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("import", help=".env -> JSON (stdout or -o)")
    a.add_argument("path")
    a.add_argument("-o", "--out", default="")
    r = sub.add_parser("render", help="JSON -> .env (stdout or -o)")
    r.add_argument("path")
    r.add_argument("-o", "--out", default="")
    c = sub.add_parser("check", help="round trip golden over .env files / directories (exit 1 on any difference)")
    c.add_argument("paths", nargs="+")
    ns = ap.parse_args(argv)
    if ns.cmd == "import":
        text = json.dumps(import_env(ns.path), indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    elif ns.cmd == "render":
        with open(ns.path, encoding="utf-8") as fh:
            text = render_env(json.load(fh))
    else:
        bad = 0
        files: List[str] = []
        for p in ns.paths:
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".env")) if os.path.isdir(p) else [p]
        for f in files:
            diffs = roundtrip_check(f)
            print("%-34s %s" % (os.path.basename(f), "golden" if not diffs else "DIFF %d" % len(diffs)))
            for d in diffs[:6]:
                print("    " + d)
            bad += bool(diffs)
        print("TOTAL DIFF-FILES %d of %d" % (bad, len(files)))
        return 1 if bad else 0
    if ns.out:
        with open(ns.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
