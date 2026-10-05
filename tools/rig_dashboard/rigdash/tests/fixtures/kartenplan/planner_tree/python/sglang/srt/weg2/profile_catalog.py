"""PROFIL-EDITOR S1: the value catalog -- what each flag / environment variable of a weg2 profile IS.

Every value in the editor is explained from a SOURCE, never from a guess.  The sources, in order:

1. **kuratiert** (``profile_catalog_curated.py``): the core values a person is meant to touch, with a plain
   sentence, the price (gain/cost) and the dependencies on other values (``depends``).
2. **argparse** -- the ``help=`` text, default, choices and arity of every ``add_argument`` of the launcher,
   harvested by AST from ``launcher.py`` (no import: the launcher pulls torch).
3. **environ** -- the comment lines above a field of ``environ.py``'s ``Envs`` plus its type and default.
4. **profil-kommentar** -- the comment above (or after) the line that sets the value in the profile itself
   (``harvest_profile_comments``; per profile, at load time).
5. Nothing found -> ``status: "unerklaert"``, shown as such with the place in the code to read.

``build_catalog`` merges 1-3 and the EDGE CATALOG (``kantenkatalog_1004.json``, see :func:`merge_edges`); ``coverage`` states how much is explained.  A curated dependency edge is checked
against the launcher by the test (``test_profile_catalog_1003``): the flags it names must exist.

PURE: stdlib only (AST over source files).  Nothing here imports sglang, starts anything or touches a GPU.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "flliper.catalog/1"

_BARE_ACTIONS = {"store_true", "store_false", "count", "store_const", "append_const"}


# ---------------------------------------------------------------------------
# AST helpers

def _const(node: ast.AST):
    """A literal's value, adjacent string parts joined; ``%``-formatted help keeps its template; else None."""
    try:
        return ast.literal_eval(node)
    except Exception:       # noqa: BLE001
        pass
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return _const(node.left)
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            parts.append(v.value if isinstance(v, ast.Constant) else "{…}")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _const(node.left), _const(node.right)
        if isinstance(a, str) and isinstance(b, str):
            return a + b
    return None


#: stored help text is cut here (the dashboard ships the catalog as a file)
HELP_MAX = 1600


def _clean(text: Optional[str]) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= HELP_MAX else t[:HELP_MAX] + " ..."


def launcher_flags(launcher_path: str) -> Dict[str, Dict[str, object]]:
    """``--flag`` -> ``{help, default, choices, nargs, bare, type, line}`` from the launcher's ``add_argument``
    calls (every spelling of the flag maps to the same record)."""
    with open(launcher_path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), launcher_path)
    out: Dict[str, Dict[str, object]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--")]
        if not names:
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        action = _const(kw["action"]) if "action" in kw else None
        rec: Dict[str, object] = {
            "help": _clean(_const(kw["help"])) if "help" in kw else "",
            "default": _const(kw["default"]) if "default" in kw else None,
            "choices": _const(kw["choices"]) if "choices" in kw else None,
            "nargs": _const(kw["nargs"]) if "nargs" in kw else None,
            "bare": action in _BARE_ACTIONS,
            "action": action,
            "type": (kw["type"].id if "type" in kw and isinstance(kw["type"], ast.Name) else None),
            "line": node.lineno,
        }
        for n in names:
            out[n] = dict(rec)
    return out


def server_args_flags(server_args_path: str) -> Dict[str, Dict[str, object]]:
    """``--flag`` -> ``{help, default, choices, line}`` from the ``ServerArgs`` dataclass: ``name: A[T, Arg(help=...)] = default``
    (the flag is ``--name-with-dashes`` unless ``Arg(cli_name=...)``).  These are the flags a profile hands to the
    groups through ``--extra-p`` / ``--extra-d`` (``--rank-moe-ratio``, ``--rank-gpu-memory-mib`` ...)."""
    with open(server_args_path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), server_args_path)
    out: Dict[str, Dict[str, object]] = {}
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "ServerArgs"]:
        for node in cls.body:
            if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
                continue
            ann = node.annotation
            arg_call = None
            if isinstance(ann, ast.Subscript):
                for sub in ast.walk(ann):
                    if isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "Arg":
                        arg_call = sub
                        break
            if arg_call is None:
                # the short form: name: A[Type, "help text"] = default
                doc = None
                if isinstance(ann, ast.Subscript) and isinstance(ann.slice, ast.Tuple):
                    for el in ann.slice.elts[1:]:
                        v = _const(el)
                        if isinstance(v, str):
                            doc = v
                if doc is None:
                    continue
                out["--" + node.target.id.replace("_", "-")] = {
                    "help": _clean(doc), "default": _const(node.value) if node.value is not None else None,
                    "choices": None, "bare": False, "nargs": None, "line": node.lineno}
                continue
            kw = {k.arg: k.value for k in arg_call.keywords if k.arg}
            if "no_cli" in kw and _const(kw["no_cli"]) is True:
                continue
            cli = _const(kw["cli_name"]) if "cli_name" in kw else "--" + node.target.id.replace("_", "-")
            default = None
            if node.value is not None:
                default = _const(node.value)
            action = _const(kw["action"]) if "action" in kw else None
            out[str(cli)] = {"help": _clean(_const(kw["help"])) if "help" in kw else "", "default": default,
                             "choices": _const(kw["choices"]) if "choices" in kw else None,
                             "bare": action in _BARE_ACTIONS, "nargs": _const(kw["nargs"]) if "nargs" in kw else None,
                             "line": node.lineno}
    return out


def arg_specs(flags: Mapping[str, Mapping]) -> Dict[str, Dict[str, object]]:
    """The slice :func:`profile_json.parse_args` needs: ``{flag: {bare, nargs}}``."""
    return {f: {"bare": bool(r.get("bare")), "nargs": r.get("nargs")} for f, r in flags.items()}


_RX_ENV = re.compile(r"^(\s*)(SGLANG_[A-Z0-9_]+|FLLIPER_[A-Z0-9_]+|HTSGLANG_[A-Z0-9_]+)\s*(?::[^=]+)?=\s*(Env\w+)\((.*)\)\s*(#.*)?$")


def environ_fields(environ_path: str) -> Dict[str, Dict[str, object]]:
    """``SGLANG_X`` -> ``{kind, default, comment, line}`` from ``environ.py`` (comment = the contiguous ``#`` lines
    directly above the field)."""
    with open(environ_path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    out: Dict[str, Dict[str, object]] = {}
    for i, ln in enumerate(lines):
        m = _RX_ENV.match(ln)
        if not m:
            continue
        j = i - 1
        com: List[str] = []
        while j >= 0 and lines[j].strip().startswith("#"):
            com.append(lines[j].strip().lstrip("#").strip())
            j -= 1
        com.reverse()
        out[m.group(2)] = {"kind": m.group(3), "default": m.group(4).strip(), "comment": _clean(" ".join(com)),
                           "trailing": _clean((m.group(5) or "").lstrip("#")), "line": i + 1}
    return out


# ---------------------------------------------------------------------------
# profile comments

_RX_SET = [
    re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)=(?!\()"),          # NAME=value
    re.compile(r"^\s*_form\s+([A-Z][A-Z0-9_]*)\b"),                      # _form NAME value
    re.compile(r"^\s*(--[a-z][a-z0-9-]*)\b"),                            # a flag on its own line inside PROFILE_ARGS=( ... )
]


def harvest_profile_comments(env_path: str) -> Dict[str, Dict[str, str]]:
    """name -> ``{text, source}``: the comment block directly above the line that sets ``name`` (or the trailing
    ``# ...`` of that line).  Flags inside ``PROFILE_ARGS=(...)`` are found per line and by their first word."""
    try:
        with open(env_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return {}
    out: Dict[str, Dict[str, str]] = {}
    base = os.path.basename(env_path)
    for i, ln in enumerate(lines):
        names: List[str] = []
        for rx in _RX_SET:
            m = rx.match(ln)
            if m:
                names.append(m.group(1))
        for m in re.finditer(r"(?<![\w-])(--[a-z][a-z0-9-]+)", ln):
            if ln.lstrip().startswith("#"):
                break
            names.append(m.group(1))
        if not names:
            continue
        com: List[str] = []
        j = i - 1
        while j >= 0 and lines[j].strip().startswith("#"):
            com.append(lines[j].strip().lstrip("#").strip())
            j -= 1
        com.reverse()
        trail = ""
        if "#" in ln:
            t = ln.split(" #", 1)
            if len(t) == 2:
                trail = t[1].strip()
        text = _clean(" ".join(com + ([trail] if trail else [])))
        if not text:
            continue
        for n in names:
            out.setdefault(n, {"text": text[:900], "source": "%s:%d" % (base, i + 1)})
    return out


# ---------------------------------------------------------------------------
# edge catalog (Auftrag 2002 C): evidence + one sentence per dependency, merged into ``depends``

EDGES_SCHEMA = "flliper.kanten/1"
EDGES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kantenkatalog_1004.json")
_RX_CODE = re.compile(r"^[A-Z]+(?:-[A-Z]+)+$")


def load_edges(path: str = "") -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """The edges of the edge catalog and a status dict.  A missing or unreadable file is NOT an error of the catalog: the
    dependencies stay as curated, the status says ``geladen: False`` and why (never a silent empty list)."""
    path = path or EDGES_FILE
    info: Dict[str, object] = {"datei": os.path.basename(path), "geladen": False, "grund": "", "schema": ""}
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        info["grund"] = "nicht lesbar: %s" % exc
        return [], info
    info["schema"] = str(doc.get("schema", ""))
    edges = doc.get("kanten")
    if info["schema"] != EDGES_SCHEMA or not isinstance(edges, list):
        info["grund"] = "Schema %r (erwartet %s) oder keine Kantenliste" % (info["schema"], EDGES_SCHEMA)
        return [], info
    ok = [e for e in edges if isinstance(e, dict) and e.get("von") and e.get("nach") and e.get("rel")]
    info.update({"geladen": True, "kanten_gesamt": len(ok), "verworfen": len(edges) - len(ok)})
    return ok, info


def _beleg(e: Mapping) -> Optional[Dict[str, object]]:
    b = e.get("beleg")
    if not isinstance(b, dict) or not b.get("datei"):
        return None
    return {"datei": str(b.get("datei")), "zeile": b.get("zeile"), "anker": str(b.get("anker") or "")}


def merge_edges(entries: Dict[str, Dict[str, object]], edges: Sequence[Mapping], info: Dict[str, object],
                refusal_codes: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """Merge the edge catalog into ``entries[von]["depends"]`` (key ``(von, nach)``), IN PLACE, and say what happened.

    * a curated edge with a catalog edge of the same pair gets ``kante`` (id), ``beleg`` (``datei``/``zeile``/``anker``), ``satz``
      (the tradeoff sentence), ``wert`` (the condition, shown as text only -- the editor does NOT evaluate it) and ``belegt: True``;
      when the catalog names another ``rel``, the curated one stays and ``rel_katalog`` carries the catalog's (no silent overwrite);
    * a catalog edge with no curated twin is appended (``quelle: "katalog"``);
    * a curated edge without catalog edge stays and is marked ``belegt: False`` -- "ohne Beleg", not refuted;
    * every edge gets ``to_kind``: ``flag`` / ``env`` / ``var`` (a row of the editor), ``ablehnung`` (a code of the refusal register,
      no row) or ``unbekannt`` -- so a chip never points silently at nothing.

    No rule is evaluated here (Nutzerentscheid 05.10.): the launcher and the dry run judge, the edges only explain."""
    codes = set(refusal_codes or ())
    by_pair: Dict[Tuple[str, str], Mapping] = {(str(e["von"]), str(e["nach"])): e for e in edges}
    used = set()
    skipped: List[str] = []
    rel_diff: List[str] = []
    for name, ent in entries.items():
        for d in ent.get("depends", []):
            e = by_pair.get((name, str(d.get("to"))))
            if e is None:
                d.update({"belegt": False, "quelle": "kuratiert", "beleg": None, "satz": "", "kante": "", "wert": None})
                continue
            used.add((name, str(d["to"])))
            d.update({"belegt": True, "quelle": "katalog+kuratiert", "kante": str(e.get("id", "")), "beleg": _beleg(e),
                      "satz": str(e.get("satz") or ""), "wert": e.get("wert")})
            if e["rel"] != d.get("rel"):
                d["rel_katalog"] = e["rel"]
                rel_diff.append(str(e.get("id", "")))
    new = 0
    for key, e in by_pair.items():
        if key in used:
            continue
        ent = entries.get(key[0])
        if ent is None:
            skipped.append(str(e.get("id", key)))
            continue
        ent.setdefault("depends", []).append({
            "to": key[1], "rel": e["rel"], "effect": str(e.get("satz") or ""), "calc": e.get("calc") or "text", "belegt": True,
            "quelle": "katalog", "kante": str(e.get("id", "")), "beleg": _beleg(e), "satz": str(e.get("satz") or ""),
            "wert": e.get("wert")})
        new += 1
    n_with = n_without = n_kind_unknown = 0
    for ent in entries.values():
        for d in ent.get("depends", []):
            to = str(d.get("to"))
            if to in entries:
                d["to_kind"] = str(entries[to].get("kind") or "unbekannt")
            elif to in codes or (not codes and _RX_CODE.match(to)):
                d["to_kind"] = "ablehnung"
            else:
                d["to_kind"] = "unbekannt"
                n_kind_unknown += 1
            if d.get("belegt"):
                n_with += 1
            else:
                n_without += 1
    info.update({"verschmolzen": len(used), "neu": new, "uebersprungen_ohne_von": skipped, "rel_abweichend": rel_diff,
                 "kanten_belegt": n_with, "kanten_ohne_beleg": n_without, "ziel_unbekannt": n_kind_unknown,
                 "wertbedingt": sum(1 for e in edges if e.get("wert"))})
    return info


def _refusals_module():
    """The register module next to this file (loaded by path, like the dashboard does); ``None`` when it cannot be loaded."""
    try:
        from importlib import util as _u
        spec = _u.spec_from_file_location("kp_refusals", os.path.join(os.path.dirname(os.path.abspath(__file__)), "refusals.py"))
        mod = _u.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:       # noqa: BLE001 -- optional data
        return None


# ---------------------------------------------------------------------------
# merge

def build_catalog(launcher_path: str, environ_path: str, curated: Mapping[str, Mapping], tree_rev: str = "",
                  server_args_path: str = "", edges_path: str = "") -> Dict[str, object]:
    flags = launcher_flags(launcher_path)
    envs = environ_fields(environ_path)
    entries: Dict[str, Dict[str, object]] = {}
    if server_args_path and os.path.isfile(server_args_path):
        for name, r in server_args_flags(server_args_path).items():
            if name in flags:
                continue
            entries[name] = {"id": name, "kind": "flag", "name": name, "type": "bool" if r["bare"] else "str",
                             "default": r["default"], "choices": r["choices"], "bare": r["bare"], "nargs": r["nargs"],
                             "text": "", "gain": "", "cost": "", "depends": [], "level": "experte", "group": "",
                             "planner_derived": False, "scope": "server",
                             "source": {"file": "server_args.py", "line": r["line"], "kind": "argparse"},
                             "help": r["help"], "status": "geerntet" if r["help"] else "unerklaert"}
    for name, r in flags.items():
        e: Dict[str, object] = {"id": name, "kind": "flag", "name": name, "type": r["type"] or ("bool" if r["bare"] else "str"),
                                "default": r["default"], "choices": r["choices"], "bare": r["bare"], "nargs": r["nargs"],
                                "text": "", "gain": "", "cost": "", "depends": [], "level": "experte", "group": "",
                                "planner_derived": False, "source": {"file": "weg2/launcher.py", "line": r["line"], "kind": "argparse"},
                                "help": r["help"], "status": "geerntet" if r["help"] else "unerklaert"}
        entries[name] = e
    for name, r in envs.items():
        txt = r["comment"] or r["trailing"]
        entries[name] = {"id": name, "kind": "env", "name": name, "type": r["kind"], "default": r["default"], "choices": None,
                         "bare": False, "nargs": None, "text": "", "gain": "", "cost": "", "depends": [], "level": "experte",
                         "group": "", "planner_derived": False,
                         "source": {"file": "environ.py", "line": r["line"], "kind": "environ"}, "help": txt,
                         "status": "geerntet" if txt else "unerklaert"}
    for name, c in curated.items():
        e = entries.setdefault(name, {"id": name, "kind": c.get("kind", "env"), "name": name, "type": "str", "default": None,
                                      "choices": None, "bare": False, "nargs": None, "help": "",
                                      "source": {"file": "profile", "line": 0, "kind": "kuratiert"}})
        for k in ("text", "gain", "cost", "group", "level", "planner_derived"):
            if k in c:
                e[k] = c[k]
        e["depends"] = [dict(d) for d in c.get("depends", [])]
        e["status"] = "kuratiert"
    ref = _refusals_module()
    edges, einfo = load_edges(edges_path)
    merge_edges(entries, edges, einfo, [r.code for r in ref.REGISTER] if ref is not None else None)
    cat = {"schema": SCHEMA, "tree_rev": tree_rev, "entries": entries, "stats": coverage(entries), "kanten": einfo}
    try:
        with open(launcher_path, encoding="utf-8") as fh:
            src = fh.read()
        mod = ref
        cat["register_wired"] = mod.wired_codes(src)
        cat["launcher_raise_sites"] = mod.launcher_raise_sites(src)
    except Exception:       # noqa: BLE001 -- the wiring statement is optional data
        cat["register_wired"] = []
        cat["launcher_raise_sites"] = 0
    return cat


def coverage(entries: Mapping[str, Mapping]) -> Dict[str, int]:
    n = len(entries)
    return {"entries": n, "kuratiert": sum(1 for e in entries.values() if e["status"] == "kuratiert"),
            "geerntet": sum(1 for e in entries.values() if e["status"] == "geerntet"),
            "unerklaert": sum(1 for e in entries.values() if e["status"] == "unerklaert"),
            "flags": sum(1 for e in entries.values() if e["kind"] == "flag"),
            "envs": sum(1 for e in entries.values() if e["kind"] == "env")}


def find_tree_files(python_dir: str) -> Tuple[str, str, str]:
    return (os.path.join(python_dir, "sglang", "srt", "weg2", "launcher.py"),
            os.path.join(python_dir, "sglang", "srt", "environ.py"),
            os.path.join(python_dir, "sglang", "srt", "server_args.py"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import importlib.util

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--python-dir", default="", help="<tree>/python (launcher.py and environ.py are read from it)")
    ap.add_argument("--rev", default="")
    ap.add_argument("-o", "--out", default="")
    ns = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    py = ns.python_dir or os.path.abspath(os.path.join(here, "..", "..", ".."))
    spec = importlib.util.spec_from_file_location("profile_catalog_curated", os.path.join(here, "profile_catalog_curated.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    launcher, environ, sargs = find_tree_files(py)
    cat = build_catalog(launcher, environ, mod.CURATED, ns.rev, sargs)
    text = json.dumps(cat, indent=1, sort_keys=True, ensure_ascii=False, default=str) + "\n"
    if ns.out:
        with open(ns.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    st = cat["stats"]
    print("catalog: %(entries)d entries (%(flags)d flags, %(envs)d envs): %(kuratiert)d kuratiert, %(geerntet)d geerntet, %(unerklaert)d unerklaert" % st, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
