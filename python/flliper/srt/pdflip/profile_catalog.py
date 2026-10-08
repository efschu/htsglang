"""PROFIL-EDITOR S1: the value catalog -- what each flag / environment variable of a pdflip profile IS.

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

PURE: stdlib only (AST over source files).  Nothing here imports flliper, starts anything or touches a GPU.
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


_RX_ENV = re.compile(r"^(\s*)(FLLIPER_[A-Z0-9_]+|FLLIPER_[A-Z0-9_]+|HTSGLANG_[A-Z0-9_]+)\s*(?::[^=]+)?=\s*(Env\w+)\((.*)\)\s*(#.*)?$")


def environ_fields(environ_path: str) -> Dict[str, Dict[str, object]]:
    """``FLLIPER_X`` -> ``{kind, default, comment, line}`` from ``environ.py`` (comment = the contiguous ``#`` lines
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
#: Display wording (English) for the vocabulary that stays a KEY in the data: the edge ``rel`` values and the entry ``status`` values.
#: The keys (``tauscht`` ... / ``kuratiert`` ...) are what code, tests and ``catalog.json`` entries carry; only the text a person reads is English.
#: ``main`` ships both tables in ``catalog.json`` under ``anzeige`` so the page does not need its own copy.
REL_ANZEIGE: Dict[str, str] = {"tauscht": "trades", "braucht": "requires", "schliesst_aus": "excludes", "abgeleitet_von": "derived_from",
                               "skaliert_mit": "scales_with"}
STATUS_ANZEIGE: Dict[str, str] = {"kuratiert": "curated", "erklaert": "explained", "geerntet": "harvested", "unerklaert": "unexplained"}
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
        info["grund"] = "not readable: %s" % exc
        return [], info
    info["schema"] = str(doc.get("schema", ""))
    edges = doc.get("kanten")
    if info["schema"] != EDGES_SCHEMA or not isinstance(edges, list):
        info["grund"] = "schema %r (expected %s) or no edge list" % (info["schema"], EDGES_SCHEMA)
        return [], info
    ok = [e for e in edges if isinstance(e, dict) and e.get("von") and e.get("nach") and e.get("rel")]
    info.update({"geladen": True, "kanten_gesamt": len(ok), "verworfen": len(edges) - len(ok)})
    return ok, info


# Evidence resolution (Auftrag 2013): the ANCHOR TEXT is the evidence, the stored line number is only a hint.  A line number in
# ``launcher.py``/``environ.py`` drifts with every edit above it; the anchor does not.  ``resolve_edge_belege`` finds each anchor in
# its file, so a pure line shift never breaks an edge -- only a missing or ambiguous anchor does.
ANKER_TOL = 10          # a multi-hit anchor must have a hit within this many lines of where the edge is expected
ANKER_OK = ("eindeutig", "nah")                      # resolved
ANKER_PROBLEM = ("veraltet", "mehrdeutig", "datei_fehlt")   # the evidence cannot be trusted: say so, never guess
ANKER_FREMD = ("andere_linie",)         # the edge's ``baeume`` does not name the tree being resolved (code of the other line): not checked here, no problem
_ANKER_REPO_FILE = "python/flliper/srt/pdflip/launcher.py"


def anchor_lines(text: str, anker: str) -> List[int]:
    """1-based START lines of every occurrence of ``anker`` in ``text`` (an anchor may span lines; one entry per line)."""
    out: List[int] = []
    if not anker:
        return out
    pos = text.find(anker)
    while pos >= 0:
        ln = text.count("\n", 0, pos) + 1
        if not out or out[-1] != ln:
            out.append(ln)
        pos = text.find(anker, pos + 1)
    return out


def resolve_anchor(hits: Sequence[int], hint: object, drift: int = 0) -> Tuple[Optional[int], str]:
    """``(line, status)`` for an anchor with the given hit lines.

    * no hit            -> ``(None, "veraltet")``: the anchor text is gone from the file (the evidence is stale);
    * exactly one hit   -> ``(hit, "eindeutig")`` wherever it is;
    * several hits      -> the one nearest to ``hint + drift`` (``drift`` = how far this region of the file moved, see
      :func:`resolve_edge_belege`) is ``"nah"``; when two hits are equally near, or the nearest is farther than ``ANKER_TOL`` from
      the expectation, the anchor is ``(None, "mehrdeutig")`` -- no silent pick."""
    if not hits:
        return None, "veraltet"
    if len(hits) == 1:
        return hits[0], "eindeutig"
    h = hint if isinstance(hint, int) and not isinstance(hint, bool) else 0
    want = h + drift
    ranked = sorted(hits, key=lambda x: (abs(x - want), x))
    d0 = abs(ranked[0] - want)
    if d0 > ANKER_TOL or abs(ranked[1] - want) == d0:
        return None, "mehrdeutig"
    return ranked[0], "nah"


def _repo_root_of(launcher_path: str) -> str:
    """The repo root implied by the launcher path (``<root>/python/flliper/srt/pdflip/launcher.py``); ``""`` for any other layout."""
    p = os.path.abspath(launcher_path).replace(os.sep, "/")
    return p[:-len(_ANKER_REPO_FILE) - 1] if p.endswith("/" + _ANKER_REPO_FILE) else ""


def resolve_edge_belege(edges: Sequence[Mapping], root: str, baum: str = "") -> Dict[str, Dict[str, object]]:
    """Resolve the evidence of every edge by its ANCHOR TEXT.  Returns ``{edge id: {zeile, hinweis, status, treffer, datei}}``.

    ``zeile`` is the resolved line (the stored line stays visible as ``hinweis`` when nothing resolves).  Relative files are
    read under ``root``, absolute ones as they are (``extern_fehlt`` when such a file is not on this machine: not a problem, not
    verifiable).  Drift: edges whose anchor is unique show how far their region moved (hit - stored); a multi-hit anchor expects
    its hit at ``stored + median drift of the nearest unique edges of the same file`` -- so a prepend of N lines moves every hint by N
    and an ambiguous anchor still lands on its own line instead of a stray namesake.

    ``baum`` (the label of the code tree under ``root``, ``"27b"`` or ``"nf"``; ``""`` = no filter, every edge is checked): an edge may
    carry ``baeume`` (the labels of the trees whose code it documents; absent = both lines).  An edge that does not name ``baum`` is
    ``andere_linie`` -- its evidence lives in the other line's code (the Dual form is a 27B-line feature), so it is NOT looked up in
    this tree (no ``veraltet`` for code that was never here), and it is not a problem either."""
    texts: Dict[str, Optional[str]] = {}
    rows: List[Dict[str, object]] = []
    for e in edges:
        b = e.get("beleg") if isinstance(e, Mapping) else None
        if not isinstance(b, Mapping) or not b.get("datei"):
            continue
        datei = str(b["datei"])
        path = datei if os.path.isabs(datei) else (os.path.join(root, datei) if root else "")
        if baum and isinstance(e.get("baeume"), list) and baum not in e["baeume"]:
            rows.append({"id": str(e.get("id", "")), "datei": datei, "hinweis": b.get("zeile"), "treffer": 0, "zeile": b.get("zeile"),
                         "status": "andere_linie", "_path": path})
            continue
        if path not in texts:
            try:
                with open(path, encoding="utf-8") as fh:
                    texts[path] = fh.read()
            except OSError:
                texts[path] = None
        text = texts[path]
        row: Dict[str, object] = {"id": str(e.get("id", "")), "datei": datei, "hinweis": b.get("zeile"), "treffer": 0,
                                  "zeile": b.get("zeile"), "status": "", "_path": path}
        if text is None:
            row["status"] = "extern_fehlt" if os.path.isabs(datei) else "datei_fehlt"
        else:
            row["_hits"] = anchor_lines(text, str(b.get("anker") or ""))
            row["treffer"] = len(row["_hits"])
        rows.append(row)
    unique: Dict[str, List[Tuple[int, int]]] = {}
    for r in rows:
        h = r.get("_hits")
        if h is not None and len(h) == 1 and isinstance(r["hinweis"], int):
            unique.setdefault(str(r["_path"]), []).append((r["hinweis"], h[0] - r["hinweis"]))
    out: Dict[str, Dict[str, object]] = {}
    for r in rows:
        if not r["status"]:
            near = sorted(unique.get(str(r["_path"]), []), key=lambda t: abs(t[0] - (r["hinweis"] if isinstance(r["hinweis"], int) else 0)))[:3]
            drifts = sorted(d for _s, d in near)
            drift = drifts[len(drifts) // 2] if drifts else 0
            line, r["status"] = resolve_anchor(r["_hits"], r["hinweis"], drift)
            if line is not None:
                r["zeile"] = line
        out[str(r["id"])] = {k: v for k, v in r.items() if not k.startswith("_")}
    return out


def _beleg(e: Mapping, res: Optional[Mapping] = None) -> Optional[Dict[str, object]]:
    b = e.get("beleg")
    if not isinstance(b, dict) or not b.get("datei"):
        return None
    out = {"datei": str(b.get("datei")), "zeile": b.get("zeile"), "anker": str(b.get("anker") or "")}
    if res is not None:
        # displayed line = the resolved one; the stored line is kept as a hint, the status says how it was found
        out.update({"zeile": res["zeile"], "zeile_hinweis": b.get("zeile"), "aufloesung": res["status"]})
    return out


def merge_edges(entries: Dict[str, Dict[str, object]], edges: Sequence[Mapping], info: Dict[str, object],
                refusal_codes: Optional[Sequence[str]] = None,
                belege: Optional[Mapping[str, Mapping]] = None) -> Dict[str, object]:
    """Merge the edge catalog into ``entries[von]["depends"]`` (key ``(von, nach)``), IN PLACE, and say what happened.

    * a curated edge with a catalog edge of the same pair gets ``kante`` (id), ``beleg`` (``datei``/``zeile``/``anker``), ``satz``
      (the tradeoff sentence), ``wert`` (the condition, shown as text only -- the editor does NOT evaluate it) and ``belegt: True``;
      when the catalog names another ``rel``, the curated one stays and ``rel_katalog`` carries the catalog's (no silent overwrite);
    * a catalog edge with no curated twin is appended (``quelle: "katalog"``);
    * a curated edge without catalog edge stays and is marked ``belegt: False`` -- "ohne Beleg", not refuted;
    * every edge gets ``to_kind``: ``flag`` / ``env`` / ``var`` (a row of the editor), ``ablehnung`` (a code of the refusal register,
      no row) or ``unbekannt`` -- so a chip never points silently at nothing.

    ``belege`` (from :func:`resolve_edge_belege`): the evidence line shown is the one resolved by its anchor text, the stored line
    becomes ``zeile_hinweis`` and ``aufloesung`` says how it was found (``eindeutig``/``nah``/``veraltet``/``mehrdeutig``/...).

    No rule is evaluated here (Nutzerentscheid 05.10.): the launcher and the dry run judge, the edges only explain."""
    codes = set(refusal_codes or ())
    belege = belege or {}
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
            d.update({"belegt": True, "quelle": "katalog+kuratiert", "kante": str(e.get("id", "")), "beleg": _beleg(e, belege.get(str(e.get("id", "")))),
                      "satz": str(e.get("satz") or ""), "wert": e.get("wert")})
            if isinstance(e.get("baeume"), list):
                d["baeume"] = list(e["baeume"])      # the edge documents the code of these trees only (absent = both lines)
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
            "quelle": "katalog", "kante": str(e.get("id", "")), "beleg": _beleg(e, belege.get(str(e.get("id", "")))), "satz": str(e.get("satz") or ""),
            "wert": e.get("wert")})
        if isinstance(e.get("baeume"), list):
            ent["depends"][-1]["baeume"] = list(e["baeume"])
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
    if belege:
        stat: Dict[str, int] = {}
        for r in belege.values():
            stat[str(r["status"])] = stat.get(str(r["status"]), 0) + 1
        info["beleg_aufloesung"] = {"status": stat, "problem": sorted(i for i, r in belege.items() if r["status"] in ANKER_PROBLEM)}
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

# ---------------------------------------------------------------------------
# os.environ consumers (names the code reads without an ``Envs`` field)

_RX_SGL_NAME = re.compile(r"^FLLIPER_[A-Z0-9_]+$")
#: how many reading places an entry lists
MAX_READS = 8


def _default_text(node: Optional[ast.AST]) -> Optional[str]:
    """The literal default of an ``os.environ.get(NAME, default)`` call as the catalog writes it (strings in double quotes), else None."""
    if node is None:
        return None
    try:
        v = ast.literal_eval(node)
    except Exception:       # noqa: BLE001
        return None
    return '"%s"' % v if isinstance(v, str) else str(v)


def _is_os_environ(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "environ" and isinstance(node.value, ast.Name) and node.value.id == "os"


def _file_consumers(tree: ast.AST) -> Tuple[Dict[str, Tuple[str, int]], List[Tuple[str, int, Optional[str]]]]:
    """(``{constant name: (FLLIPER_X, line)}``, ``[(FLLIPER_X, line, default)]``) of one parsed file: constants named ``*ENV*`` that hold an
    ``FLLIPER_`` name, and ``os.environ.get`` / ``os.getenv`` / ``os.environ[...]`` reads whose first argument is such a literal or constant."""
    consts: Dict[str, Tuple[str, int]] = {}
    for node in ast.walk(tree):
        tgt = val = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt, val = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            tgt, val = node.target, node.value
        if (isinstance(tgt, ast.Name) and "ENV" in tgt.id and isinstance(val, ast.Constant)
                and isinstance(val.value, str) and _RX_SGL_NAME.match(val.value)):
            consts[tgt.id] = (val.value, node.lineno)

    def resolve(arg: ast.AST) -> Optional[str]:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value if _RX_SGL_NAME.match(arg.value) else None
        if isinstance(arg, ast.Name) and arg.id in consts:
            return consts[arg.id][0]
        return None

    reads: List[Tuple[str, int, Optional[str]]] = []
    used: Dict[str, int] = {}        # a constant that the code USES (through a helper, a comparison ...) is a consumer too
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in consts:
            used.setdefault(node.id, node.lineno)
        if isinstance(node, ast.Call) and node.args:
            f = node.func
            # ``env.get(NAME_ENV, "1")`` through a local alias of os.environ is an env read only when the key is one of the
            # ``*ENV*`` constants; a bare literal key needs ``os.environ`` itself (any other dict has ``.get`` too)
            key_is_const = isinstance(node.args[0], ast.Name) and node.args[0].id in consts
            is_get = isinstance(f, ast.Attribute) and ((f.attr == "get" and (_is_os_environ(f.value) or key_is_const)) or
                                                       (f.attr == "getenv" and isinstance(f.value, ast.Name) and f.value.id == "os"))
            name = resolve(node.args[0]) if is_get else None
            if name:
                reads.append((name, node.lineno, _default_text(node.args[1] if len(node.args) > 1 else None)))
        elif isinstance(node, ast.Subscript) and _is_os_environ(node.value):
            name = resolve(node.slice)
            if name:
                reads.append((name, node.lineno, None))
    reads.extend((consts[c][0], ln, None) for c, ln in used.items())
    return consts, reads


def environ_constants(srt_dir: str) -> Dict[str, Dict[str, object]]:
    """``FLLIPER_X`` -> ``{default, comment, file, line, reads}`` for every name the code under ``srt_dir`` reads through ``os.environ`` or
    names in a ``*ENV*`` constant.  ``file``/``line`` = the first reading place (else the constant); ``default`` = the first literal default
    of a read; ``comment`` = the ``#`` lines directly above the constant (else above the first read).  Nothing is guessed: a name that no
    code consumes is not listed, and no text is made up from the name."""
    out: Dict[str, Dict[str, object]] = {}
    decl: Dict[str, str] = {}        # name -> the comment above its ``*ENV*`` constant
    for root, dirs, files in os.walk(srt_dir):
        dirs[:] = sorted(d for d in dirs if d not in ("tests", "__pycache__"))
        for fn in sorted(files):
            if not fn.endswith(".py") or fn.startswith("test_"):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, srt_dir).replace(os.sep, "/")
            try:
                with open(path, encoding="utf-8") as fh:
                    src = fh.read()
                tree = ast.parse(src, path)
            except (OSError, SyntaxError, ValueError):
                continue
            lines = src.splitlines()
            consts, reads = _file_consumers(tree)

            def comment_above(line: int) -> str:
                j, com = line - 2, []
                while j >= 0 and lines[j].strip().startswith("#"):
                    com.append(lines[j].strip().lstrip("#").lstrip(":").strip())     # ``#:`` = Sphinx doc comment
                    j -= 1
                return _clean(" ".join(reversed(com)))

            for name, line, dflt in reads:
                e = out.setdefault(name, {"default": None, "comment": "", "file": rel, "line": line, "reads": []})
                place = "%s:%d" % (rel, line)
                if len(e["reads"]) < MAX_READS and place not in e["reads"]:
                    e["reads"].append(place)
                if e["default"] is None and dflt is not None:
                    e["default"] = dflt
            for _cname, (name, line) in consts.items():
                decl.setdefault(name, comment_above(line))
    for name, e in out.items():
        e["comment"] = decl.get(name, "") or e["comment"]
    return out


# ---------------------------------------------------------------------------
# the catalog

def _harvest(launcher_path: str, environ_path: str, server_args_path: str = "", srt_dir: str = "") -> Dict[str, Dict[str, object]]:
    """The harvested entries of ONE tree (flags of the launcher and ServerArgs, ``Envs`` fields, and with ``srt_dir`` the names read through
    ``os.environ``), before any curated text or edge is applied."""
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
                                "planner_derived": False, "source": {"file": "pdflip/launcher.py", "line": r["line"], "kind": "argparse"},
                                "help": r["help"], "status": "geerntet" if r["help"] else "unerklaert"}
        entries[name] = e
    for name, r in envs.items():
        txt = r["comment"] or r["trailing"]
        entries[name] = {"id": name, "kind": "env", "name": name, "type": r["kind"], "default": r["default"], "choices": None,
                         "bare": False, "nargs": None, "text": "", "gain": "", "cost": "", "depends": [], "level": "experte",
                         "group": "", "planner_derived": False,
                         "source": {"file": "environ.py", "line": r["line"], "kind": "environ"}, "help": txt,
                         "status": "geerntet" if txt else "unerklaert"}
    if srt_dir and os.path.isdir(srt_dir):
        for name, r in environ_constants(srt_dir).items():
            if name in entries:
                continue        # declared in environ.py: that record stays
            entries[name] = {"id": name, "kind": "env", "name": name, "type": "os.environ", "default": r["default"], "choices": None,
                             "bare": False, "nargs": None, "text": "", "gain": "", "cost": "", "depends": [], "level": "experte",
                             "group": "", "planner_derived": False,
                             "source": {"file": r["file"], "line": r["line"], "kind": "os.environ"}, "lesestellen": r["reads"],
                             "help": r["comment"], "status": "geerntet" if r["comment"] else "unerklaert"}
    return entries


def _finish(entries: Dict[str, Dict[str, object]], curated: Mapping[str, Mapping], erklaert: Optional[Mapping[str, Mapping]],
            tree_rev: str, launcher_path: str, edges_path: str, edges_root: str, baum: str = "") -> Dict[str, object]:
    """Curated and explained texts, the edge catalog, statistics and the wiring statement on top of harvested ``entries`` (in place)."""
    # ``erklaert`` = one-sentence texts written from the code's consumers (``profile_catalog_curated.ERKLAERT``), status
    # "erklaert"; ``curated`` (the hand-curated core) is applied after it and wins on a name clash, status "kuratiert".
    for name, c, stat in [(n, c, "erklaert") for n, c in (erklaert or {}).items()] + [(n, c, "kuratiert") for n, c in curated.items()]:
        if "baeume_erwartet" in c and name not in entries:
            continue        # the name lives in another tree only: no entry in a catalog that does not cover that tree
        e = entries.setdefault(name, {"id": name, "kind": c.get("kind", "env"), "name": name, "type": "str", "default": None,
                                      "choices": None, "bare": False, "nargs": None, "help": "",
                                      "source": {"file": "profile", "line": 0, "kind": "kuratiert"}})
        for k in ("text", "gain", "cost", "group", "level", "planner_derived", "satz_quelle"):
            if k in c:
                e[k] = c[k]
        e["depends"] = [dict(d) for d in c.get("depends", [])]
        e["status"] = stat
    ref = _refusals_module()
    edges, einfo = load_edges(edges_path)
    root = edges_root or _repo_root_of(launcher_path)
    belege = resolve_edge_belege(edges, root, baum) if root else None
    merge_edges(entries, edges, einfo, [r.code for r in ref.REGISTER] if ref is not None else None, belege)
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


def build_catalog(launcher_path: str, environ_path: str, curated: Mapping[str, Mapping], tree_rev: str = "",
                  server_args_path: str = "", edges_path: str = "", edges_root: str = "",
                  erklaert: Optional[Mapping[str, Mapping]] = None, srt_dir: str = "", baum: str = "") -> Dict[str, object]:
    """The catalog of ONE tree.  ``srt_dir`` (optional) also harvests the names the code reads through ``os.environ``.  ``baum`` = the
    label of this tree (``"27b"`` / ``"nf"``, see :func:`resolve_edge_belege`); ``""`` checks every edge's evidence in this tree."""
    entries = _harvest(launcher_path, environ_path, server_args_path, srt_dir)
    return _finish(entries, curated, erklaert, tree_rev, launcher_path, edges_path, edges_root, baum)


def _differs(a: Mapping, b: Mapping) -> bool:
    return str(a.get("default")) != str(b.get("default")) or str(a.get("help") or "") != str(b.get("help") or "")


def build_union_catalog(trees: Sequence[Tuple[str, str]], curated: Mapping[str, Mapping], tree_revs: Optional[Mapping[str, str]] = None,
                        edges_path: str = "", edges_root: str = "", erklaert: Optional[Mapping[str, Mapping]] = None) -> Dict[str, object]:
    """ONE catalog over several code trees (the release image carries two: the 27B and the NF line).  ``trees`` = ``[(label, python_dir)]`` in
    priority order: the first tree that has a name gives its record.  Every entry gets ``baeume`` (the labels of the trees that have it); when
    the trees disagree on default or help text, ``abweichung`` carries each tree's own values.  Edges, refusal wiring and the curated texts
    come once, from the FIRST tree's launcher (``edges_root`` resolves the edges' evidence lines)."""
    if not trees:
        raise ValueError("build_union_catalog: no trees")
    union: Dict[str, Dict[str, object]] = {}
    per_tree: Dict[str, Dict[str, Dict[str, object]]] = {}
    first_launcher = ""
    for label, py in trees:
        launcher, environ, sargs = find_tree_files(py)
        first_launcher = first_launcher or launcher
        ents = _harvest(launcher, environ, sargs, os.path.join(py, "flliper", "srt"))
        per_tree[label] = ents
        for name, e in ents.items():
            if name not in union:
                # ``source.baum``: which tree the file:line belongs to (a name in both trees shows the first tree's place)
                union[name] = dict(e, baeume=[label], source=dict(e["source"], baum=label))
            else:
                union[name]["baeume"].append(label)
    for name, e in union.items():
        have = [(lb, per_tree[lb][name]) for lb in e["baeume"]]
        if len(have) > 1 and any(_differs(have[0][1], o) for _lb, o in have[1:]):
            e["abweichung"] = {lb: {"default": o.get("default"), "help": o.get("help"), "source": o.get("source")} for lb, o in have}
    revs = dict(tree_revs or {})
    cat = _finish(union, curated, erklaert, "+".join("%s=%s" % (lb, revs.get(lb, "")) for lb, _p in trees), first_launcher, edges_path,
                  edges_root, baum=trees[0][0])
    labels = [lb for lb, _p in trees]
    # The edges' evidence lines shown above are the FIRST tree's.  Every tree's own anchors are checked here (an edge that names only the
    # other line's tree in ``baeume`` is ``andere_linie``, not a problem), so a stale anchor in EITHER tree shows in the file.
    edges, _info = load_edges(edges_path)
    per = {}
    for i, (lb, py) in enumerate(trees):
        root = (edges_root if i == 0 and edges_root else "") or _repo_root_of(find_tree_files(py)[0])
        if root:
            res = resolve_edge_belege(edges, root, lb)
            stat: Dict[str, int] = {}
            for r in res.values():
                stat[str(r["status"])] = stat.get(str(r["status"]), 0) + 1
            per[lb] = {"status": stat, "problem": sorted(i2 for i2, r in res.items() if r["status"] in ANKER_PROBLEM)}
    if per:
        cat["kanten"]["beleg_aufloesung_baeume"] = per
    # no ``python_dir``: a build path in the output would make the file differ from machine to machine (reproducible build)
    cat["trees"] = {lb: {"rev": revs.get(lb, ""), "entries": len(per_tree[lb])} for lb, _py in trees}
    cat["stats"]["baeume"] = {"nur_" + lb: sum(1 for e in union.values() if e.get("baeume") == [lb]) for lb in labels}
    cat["stats"]["baeume"]["beide"] = sum(1 for e in union.values() if len(e.get("baeume", [])) == len(labels) > 1)
    cat["stats"]["baeume"]["abweichung"] = sum(1 for e in union.values() if "abweichung" in e)
    cat["warnungen"] = [
        "%s: the text expects the trees %s, the entry is in %s" % (n, c["baeume_erwartet"], union[n].get("baeume"))
        for n, c in sorted((erklaert or {}).items())
        if "baeume_erwartet" in c and n in union and union[n].get("baeume") != c["baeume_erwartet"]]
    cat["warnungen"] += ["%s: the text expects the trees %s, the name is in no tree" % (n, c["baeume_erwartet"])
                         for n, c in sorted((erklaert or {}).items()) if "baeume_erwartet" in c and n not in union]
    return cat


def coverage(entries: Mapping[str, Mapping]) -> Dict[str, int]:
    n = len(entries)
    return {"entries": n, "kuratiert": sum(1 for e in entries.values() if e["status"] == "kuratiert"),
            "erklaert": sum(1 for e in entries.values() if e["status"] == "erklaert"),
            "geerntet": sum(1 for e in entries.values() if e["status"] == "geerntet"),
            "unerklaert": sum(1 for e in entries.values() if e["status"] == "unerklaert"),
            "flags": sum(1 for e in entries.values() if e["kind"] == "flag"),
            "envs": sum(1 for e in entries.values() if e["kind"] == "env")}


def find_tree_files(python_dir: str) -> Tuple[str, str, str]:
    return (os.path.join(python_dir, "flliper", "srt", "pdflip", "launcher.py"),
            os.path.join(python_dir, "flliper", "srt", "environ.py"),
            os.path.join(python_dir, "flliper", "srt", "server_args.py"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import importlib.util

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--python-dir", default="", help="<tree>/python (launcher.py and environ.py are read from it); ONE tree")
    ap.add_argument("--rev", default="")
    ap.add_argument("--baum", default="", help="with --python-dir: the label of this tree (27b|nf); edges that name only the other tree are not checked")
    ap.add_argument("--tree-27b", default="", help="<tree>/python of the 27B line; with --tree-nf the catalog covers BOTH trees")
    ap.add_argument("--tree-nf", default="", help="<tree>/python of the NF line")
    ap.add_argument("--rev-27b", default="")
    ap.add_argument("--rev-nf", default="")
    ap.add_argument("-o", "--out", default="")
    ns = ap.parse_args(argv)
    if bool(ns.tree_27b) != bool(ns.tree_nf):
        ap.error("--tree-27b and --tree-nf come together (one tree: --python-dir)")
    if ns.tree_27b and ns.python_dir:
        ap.error("--python-dir and --tree-27b/--tree-nf exclude each other")
    here = os.path.dirname(os.path.abspath(__file__))
    py = ns.python_dir or os.path.abspath(os.path.join(here, "..", "..", ".."))
    spec = importlib.util.spec_from_file_location("profile_catalog_curated", os.path.join(here, "profile_catalog_curated.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if ns.tree_27b:
        cat = build_union_catalog([("27b", ns.tree_27b), ("nf", ns.tree_nf)], mod.CURATED, {"27b": ns.rev_27b, "nf": ns.rev_nf},
                                  erklaert=mod.ERKLAERT)
    else:
        launcher, environ, sargs = find_tree_files(py)
        cat = build_catalog(launcher, environ, mod.CURATED, ns.rev, sargs, erklaert=mod.ERKLAERT, srt_dir=os.path.join(py, "flliper", "srt"),
                            baum=ns.baum)
    cat["glossar"] = dict(mod.GLOSSAR)
    cat["anzeige"] = {"rel": dict(REL_ANZEIGE), "status": dict(STATUS_ANZEIGE)}
    text = json.dumps(cat, indent=1, sort_keys=True, ensure_ascii=False, default=str) + "\n"
    if ns.out:
        with open(ns.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    st = cat["stats"]
    print("catalog: %(entries)d entries (%(flags)d flags, %(envs)d envs): %(kuratiert)d kuratiert, %(erklaert)d erklaert, %(geerntet)d geerntet, %(unerklaert)d unerklaert" % st, file=sys.stderr)
    if "baeume" in st:
        print("trees: %s" % ", ".join("%s=%s" % kv for kv in sorted(st["baeume"].items())), file=sys.stderr)
    for w in cat.get("warnungen", []):
        print("WARNUNG: %s" % w, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
