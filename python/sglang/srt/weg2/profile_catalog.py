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

Honest completeness (Auftrag 990 W6): the sources above know only what is DECLARED (``Envs`` fields, ``add_argument``).  Two kinds of
names that a profile sets and the code reads were missing, and are added with the origin ``nur-leser`` (reader only -- no declaration
beside the reader):

6. **nur-leser (env)** -- every ``SGLANG_*`` / ``HTSGLANG_*`` name that code reads through ``os.environ`` / ``os.getenv`` /
   ``_env_int|float|bool|str(...)`` and that ``environ.py`` does not declare (``env_readers``, AST over ``python/sglang``).  Its text is the
   comment directly above the first reader; without one the explanation is ``ohne Doku`` (nothing is invented).
7. **nur-leser (flag)** -- flags another tree's launcher defines and this tree's does not (``--dual-*`` of the 27B line, snapshot
   ``profile_catalog_27b_flags.json`` harvested with ``--harvest-extra``); their ``help=`` is carried over, ``ohne Doku`` when empty.

``build_catalog`` merges 1-3; ``coverage`` states how much is explained.  A curated dependency edge is checked
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
    # the one-line pattern misses a field whose constructor call spans lines (16 of 958 at 5444cf8cde, among them
    # SGLANG_WEG2_VISION_FLIP_URGENT and SGLANG_WEG2_CTL_KICK_AFTER_FLIP, both set by release profiles): read those by AST
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return out
    rx_name = re.compile(r"^(SGLANG|FLLIPER|HTSGLANG)_[A-Z0-9_]+$")
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for node in cls.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name, val = node.target.id, node.value
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name, val = node.targets[0].id, node.value
            else:
                continue
            if name in out or not rx_name.match(name) or not isinstance(val, ast.Call):
                continue
            ctor = val.func.id if isinstance(val.func, ast.Name) else getattr(val.func, "attr", "")
            if not ctor.startswith("Env"):
                continue
            j = node.lineno - 2
            com: List[str] = []
            while j >= 0 and lines[j].strip().startswith("#"):
                com.append(lines[j].strip().lstrip("#").strip())
                j -= 1
            com.reverse()
            args = ", ".join(" ".join((ast.get_source_segment("\n".join(lines), a) or "").split()) for a in val.args)
            out[name] = {"kind": ctor, "default": args, "comment": _clean(" ".join(com)), "trailing": "", "line": node.lineno}
    return out


# ---------------------------------------------------------------------------
# nur-leser: names the code reads but no declaration beside them (Auftrag 990 W6)

_RX_READER_NAME = re.compile(r"^(?:SGLANG|HTSGLANG)_[A-Z0-9_]+$")
_RX_ENV_HELPER = re.compile(r"^_?env_(?:int|float|bool|str|flag|get)\w*$")
NO_DOC = "ohne Doku"


def _is_environ(node: ast.AST) -> bool:
    """``os.environ`` or a bare ``environ`` (``from os import environ``)."""
    if isinstance(node, ast.Attribute) and node.attr == "environ" and isinstance(node.value, ast.Name) and node.value.id == "os":
        return True
    return isinstance(node, ast.Name) and node.id == "environ"


def _module_str_consts(tree: ast.AST) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for node in getattr(tree, "body", []):
        tgt = val = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            tgt, val = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            tgt, val = node.target.id, node.value
        if tgt and isinstance(val, ast.Constant) and isinstance(val.value, str):
            out[tgt] = val.value
    return out


def _name_arg(node: Optional[ast.AST], consts: Mapping[str, str]) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in consts:
        return consts[node.id]
    return None


def _default_text(node: Optional[ast.AST]) -> Optional[str]:
    if isinstance(node, ast.Constant):
        return repr(node.value) if not isinstance(node.value, str) else node.value
    return None


#: a comment shorter than this many words is a section heading ("Global constants"), not an explanation of the name
MIN_DOC_WORDS = 4


def _comment_above(lines: Sequence[str], lineno: int) -> str:
    """The contiguous ``#`` block directly above (1-based) ``lineno`` -- or the trailing ``# ...`` of that line.  A block of fewer than
    :data:`MIN_DOC_WORDS` words is no explanation (``""`` -> the entry says ``ohne Doku``), nothing is made up."""
    j = lineno - 2
    com: List[str] = []
    while j >= 0 and lines[j].strip().startswith("#"):
        com.append(lines[j].strip().lstrip("#").strip().lstrip(":").strip())
        j -= 1
    com.reverse()
    text = " ".join(c for c in com if c)
    if not text and 0 <= lineno - 1 < len(lines) and " #" in lines[lineno - 1]:
        text = lines[lineno - 1].split(" #", 1)[1].strip()
    text = _clean(text)
    return text if len(text.split()) >= MIN_DOC_WORDS else ""


def env_readers(python_dir: str, skip: Sequence[str] = ("environ.py",)) -> Dict[str, Dict[str, object]]:
    """``SGLANG_X`` / ``HTSGLANG_X`` -> ``{sites, file, line, default, comment}`` for every name some code reads through
    ``os.environ.get|pop|setdefault(NAME ..)``, ``os.environ[NAME]``, ``NAME in os.environ``, ``os.getenv(NAME ..)`` or the
    ``_env_int|float|bool|str(NAME ..)`` helpers.  NAME is a string literal or a module-level string constant of the same file.
    ``sites`` counts the reading places; ``file``/``line`` is the first one (path order); ``comment`` is the block directly above
    it, ``""`` when none.  Test code (``test/``, ``tests/``, ``test_*.py``) is not scanned.  Declarations (``Envs`` fields in
    ``environ.py``) are not read here -- the caller removes them."""
    root = os.path.join(python_dir, "sglang")
    out: Dict[str, Dict[str, object]] = {}
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in ("__pycache__", "test", "tests"))
        for fn in sorted(fns):
            if not fn.endswith(".py") or fn in skip or fn.startswith("test_"):
                continue
            path = os.path.join(dp, fn)
            try:
                with open(path, encoding="utf-8") as fh:
                    src = fh.read()
                tree = ast.parse(src, path)
            except (OSError, SyntaxError, ValueError):
                continue
            lines = src.splitlines()
            consts = _module_str_consts(tree)
            rel = os.path.relpath(path, python_dir)
            stmt_of: Dict[int, int] = {}
            for st in ast.walk(tree):
                if isinstance(st, ast.stmt):
                    for sub in ast.walk(st):
                        stmt_of.setdefault(id(sub), st.lineno)
            for node in ast.walk(tree):
                name = default = None
                if isinstance(node, ast.Call):
                    f = node.func
                    if isinstance(f, ast.Attribute) and f.attr in ("get", "pop", "setdefault") and _is_environ(f.value) and node.args:
                        name = _name_arg(node.args[0], consts)
                        default = _default_text(node.args[1]) if len(node.args) > 1 else None
                    elif ((isinstance(f, ast.Attribute) and f.attr == "getenv" and isinstance(f.value, ast.Name) and f.value.id == "os")
                          or (isinstance(f, ast.Name) and f.id == "getenv")) and node.args:
                        name = _name_arg(node.args[0], consts)
                        default = _default_text(node.args[1]) if len(node.args) > 1 else None
                    elif isinstance(f, ast.Name) and _RX_ENV_HELPER.match(f.id) and node.args:
                        name = _name_arg(node.args[0], consts)
                        default = _default_text(node.args[1]) if len(node.args) > 1 else None
                elif isinstance(node, ast.Subscript) and _is_environ(node.value) and isinstance(node.ctx, ast.Load):
                    sl = node.slice
                    name = _name_arg(sl, consts)
                elif isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)) \
                        and len(node.comparators) == 1 and _is_environ(node.comparators[0]):
                    name = _name_arg(node.left, consts)
                if not name or not _RX_READER_NAME.match(name):
                    continue
                ln = getattr(node, "lineno", 0)
                rec = out.get(name)
                if rec is None:
                    top = stmt_of.get(id(node), ln)
                    out[name] = {"sites": 1, "file": rel, "line": ln, "default": default,
                                 "comment": _comment_above(lines, top)}
                else:
                    rec["sites"] = int(rec["sites"]) + 1
                    if rec.get("default") is None and default is not None:
                        rec["default"] = default
    return out


def extra_launcher_flags(launcher_path: str, prefix: str, rev: str, exclude: Sequence[str] = ()) -> Dict[str, Dict[str, object]]:
    """Flags ``prefix*`` another tree's launcher defines (the 27B line's ``--dual-*``): ``--flag`` -> record of
    :func:`launcher_flags` plus ``rev`` (the tree it was read from).  The snapshot file is what the committed catalog is built from."""
    out: Dict[str, Dict[str, object]] = {}
    for name, r in launcher_flags(launcher_path).items():
        if name.startswith(prefix) and name not in exclude:
            rec = dict(r)
            rec["rev"] = rev
            out[name] = rec
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
# merge

def build_catalog(launcher_path: str, environ_path: str, curated: Mapping[str, Mapping], tree_rev: str = "",
                  server_args_path: str = "", python_dir: str = "",
                  extra_flags: Optional[Mapping[str, Mapping]] = None) -> Dict[str, object]:
    """``python_dir`` (``<tree>/python``) switches on the ``nur-leser`` env names (:func:`env_readers`); ``extra_flags`` is a
    snapshot of :func:`extra_launcher_flags` (the 27B line's ``--dual-*``)."""
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
    # --- nur-leser: read by code, declared nowhere beside the reader (Auftrag 990 W6). Never invented: the explanation is the
    # comment above the first reader, else NO_DOC (status stays "unerklaert", so the editor still counts it as unexplained).
    if python_dir:
        for name, r in sorted(env_readers(python_dir).items()):
            if name in entries:
                continue
            txt = str(r["comment"] or "")
            entries[name] = {"id": name, "kind": "env", "name": name, "type": "str", "default": r["default"], "choices": None,
                             "bare": False, "nargs": None, "text": "", "gain": "", "cost": "", "depends": [], "level": "experte",
                             "group": "", "planner_derived": False,
                             "source": {"file": r["file"], "line": r["line"], "kind": "nur-leser", "sites": r["sites"]},
                             "help": txt, "doc_note": "" if txt else NO_DOC, "status": "geerntet" if txt else "unerklaert"}
    for name, r in sorted((extra_flags or {}).items()):
        if name in entries:
            continue
        txt = str(r.get("help") or "")
        entries[name] = {"id": name, "kind": "flag", "name": name, "type": r.get("type") or ("bool" if r.get("bare") else "str"),
                         "default": r.get("default"), "choices": r.get("choices"), "bare": bool(r.get("bare")), "nargs": r.get("nargs"),
                         "text": "", "gain": "", "cost": "", "depends": [], "level": "experte", "group": "", "planner_derived": False,
                         "source": {"file": "weg2/launcher.py@%s" % r.get("rev", "?"), "line": r.get("line", 0), "kind": "nur-leser"},
                         "help": txt, "doc_note": "" if txt else NO_DOC, "status": "geerntet" if txt else "unerklaert"}
    for name, c in curated.items():
        e = entries.setdefault(name, {"id": name, "kind": c.get("kind", "env"), "name": name, "type": "str", "default": None,
                                      "choices": None, "bare": False, "nargs": None, "help": "",
                                      "source": {"file": "profile", "line": 0, "kind": "kuratiert"}})
        for k in ("text", "gain", "cost", "group", "level", "planner_derived"):
            if k in c:
                e[k] = c[k]
        e["depends"] = [dict(d) for d in c.get("depends", [])]
        e["status"] = "kuratiert"
    cat = {"schema": SCHEMA, "tree_rev": tree_rev, "entries": entries, "stats": coverage(entries)}
    try:
        with open(launcher_path, encoding="utf-8") as fh:
            src = fh.read()
        from importlib import util as _u
        spec = _u.spec_from_file_location("kp_refusals", os.path.join(os.path.dirname(os.path.abspath(__file__)), "refusals.py"))
        mod = _u.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
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
            "envs": sum(1 for e in entries.values() if e["kind"] == "env"),
            "by_source": _by_source(entries)}


def _by_source(entries: Mapping[str, Mapping]) -> Dict[str, int]:
    """entries per origin: ``kuratiert`` / ``argparse`` / ``environ`` / ``nur-leser`` (flags and envs apart)."""
    out: Dict[str, int] = {}
    for e in entries.values():
        src = e.get("source") or {}
        key = "%s/%s" % (src.get("kind", "?"), e["kind"])
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def find_tree_files(python_dir: str) -> Tuple[str, str, str]:
    return (os.path.join(python_dir, "sglang", "srt", "weg2", "launcher.py"),
            os.path.join(python_dir, "sglang", "srt", "environ.py"),
            os.path.join(python_dir, "sglang", "srt", "server_args.py"))


#: the 27B line's ``--dual-*`` launcher flags (desk/27b-dual-integ-1003c @ 54a30199b1), harvested once with ``--harvest-extra``
EXTRA_FLAGS_SNAPSHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_catalog_27b_flags.json")


def load_extra_flags(path: str = EXTRA_FLAGS_SNAPSHOT) -> Dict[str, Dict[str, object]]:
    try:
        with open(path, encoding="utf-8") as fh:
            return dict(json.load(fh).get("flags", {}))
    except (OSError, ValueError):
        return {}


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import importlib.util

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--python-dir", default="", help="<tree>/python (launcher.py and environ.py are read from it)")
    ap.add_argument("--rev", default="")
    ap.add_argument("-o", "--out", default="")
    ap.add_argument("--no-readers", action="store_true", help="skip the nur-leser env names (env_readers)")
    ap.add_argument("--extra-flags", default=EXTRA_FLAGS_SNAPSHOT, help="snapshot of another tree's launcher flags (default: the 27B --dual-* snapshot)")
    ap.add_argument("--harvest-extra", default="", metavar="LAUNCHER.py", help="write the snapshot for LAUNCHER.py's flags starting with --prefix to -o and stop")
    ap.add_argument("--prefix", default="--dual-")
    ap.add_argument("--extra-rev", default="")
    ns = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    py = ns.python_dir or os.path.abspath(os.path.join(here, "..", "..", ".."))
    if ns.harvest_extra:
        snap = {"schema": "flliper.catalog.extra/1", "rev": ns.extra_rev, "prefix": ns.prefix,
                "source": "weg2/launcher.py @ " + ns.extra_rev,
                "flags": extra_launcher_flags(ns.harvest_extra, ns.prefix, ns.extra_rev, exclude=list(launcher_flags(find_tree_files(py)[0])))}
        with open(ns.out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(snap, indent=1, sort_keys=True, ensure_ascii=False, default=str) + "\n")
        print("snapshot: %d flags %s* from %s" % (len(snap["flags"]), ns.prefix, ns.harvest_extra), file=sys.stderr)
        return 0
    spec = importlib.util.spec_from_file_location("profile_catalog_curated", os.path.join(here, "profile_catalog_curated.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    launcher, environ, sargs = find_tree_files(py)
    cat = build_catalog(launcher, environ, mod.CURATED, ns.rev, sargs, python_dir="" if ns.no_readers else py,
                        extra_flags=load_extra_flags(ns.extra_flags) if ns.extra_flags else None)
    text = json.dumps(cat, indent=1, sort_keys=True, ensure_ascii=False, default=str) + "\n"
    if ns.out:
        with open(ns.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    st = cat["stats"]
    print("catalog: %(entries)d entries (%(flags)d flags, %(envs)d envs): %(kuratiert)d kuratiert, %(geerntet)d geerntet, %(unerklaert)d unerklaert" % st, file=sys.stderr)
    print("by source: " + ", ".join("%s=%d" % kv for kv in st["by_source"].items()), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
