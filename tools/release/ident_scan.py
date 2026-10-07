#!/usr/bin/env python3
"""ident_scan.py -- German identifiers and machine-readable KEYS/VALUES in the modules added since a date (F0-A, 07.10.2026).

The translation of the dashboard (desk/planer-english*) leaves machine names alone on purpose (JSON keys, enum values, flags).
This scan lists what is left for the identifier table: for a git ref it takes every file ADDED since --since (or the paths given
with --paths) and collects the words that are German by the kit's own lexicon (english_audit.Lexicon.german_ident):

  name        Python NAME token (def/class/variable/parameter/attribute)
  strkey      identifier-shaped Python string constant in a MACHINE position: dict key, subscript, first argument of
              get/pop/setdefault/getattr/hasattr/setattr, operand of ==/!=/in (enum value). Prose strings (log text, help) are not listed.
  json_key    key of a JSON data file (catalog.json, kantenkatalog, kartenplan_data ...)
  json_value  identifier-shaped lower-case string value of a JSON file (status values, relation words)
  js          occurrences of an already listed word in the static .js/.html files of the ref (the reader side; JS prose is not scanned
              for new words)

Read only. Output: JSON {word: {kinds, count, files, tree_files}} where tree_files = number of tracked in-scope files of the ref
that carry the word as a whole word (ident_fix.py rewrites every one of them, so a high number is a collision/side-effect risk).

usage: ident_scan.py --ref <git-ref> [--repo R] [--since 2026-10-03] [--paths regex] [--json out.json]
"""
import argparse
import ast
import io
import json
import os
import re
import subprocess
import sys
import tokenize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import english_audit as EA  # noqa: E402
import rename_to_flliper as R  # noqa: E402

SHAPE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{2,40}$")
EXTS = (".py", ".json")        # identifier sources; shell/text files carry prose and SGLANG_* env names, the JS side is only counted


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, check=True).stdout


def show(repo, ref, path):
    r = subprocess.run(["git", "-C", repo, "show", "%s:%s" % (ref, path)], capture_output=True)
    try:
        return r.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def walk_json(o, keys, vals):
    if isinstance(o, dict):
        for k, v in o.items():
            keys.add(k)
            walk_json(v, keys, vals)
    elif isinstance(o, list):
        for v in o:
            walk_json(v, keys, vals)
    elif isinstance(o, str):
        vals.add(o)


def machine_constants(tree):
    """String constants of a Python AST in machine positions (see the module doc)."""
    out = set()
    str_of = lambda n: n.value if isinstance(n, ast.Constant) and isinstance(n.value, str) else None
    for n in ast.walk(tree):
        if isinstance(n, ast.Dict):
            out.update(x for x in map(str_of, n.keys) if x)
        elif isinstance(n, ast.Subscript):
            sl = n.slice
            if str_of(sl):
                out.add(str_of(sl))
            elif isinstance(sl, ast.Tuple):
                out.update(x for x in map(str_of, sl.elts) if x)
        elif isinstance(n, ast.Call):
            f = n.func
            fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            if fname in ("get", "pop", "setdefault", "getattr", "hasattr", "setattr", "delattr") and n.args:
                for a in n.args[:2 if fname in ("getattr", "hasattr", "setattr", "delattr") else 1]:
                    if str_of(a):
                        out.add(str_of(a))
        elif isinstance(n, ast.Compare):
            for side in [n.left] + list(n.comparators):
                if str_of(side):
                    out.add(str_of(side))
                elif isinstance(side, (ast.Tuple, ast.List, ast.Set)):
                    out.update(x for x in map(str_of, side.elts) if x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--repo", default="/spinning/htsglang")
    ap.add_argument("--since", default="2026-10-03T00:00:00Z")
    ap.add_argument("--paths")
    ap.add_argument("--dict-dir", default=EA.DEFAULT_DICT_DIR)
    ap.add_argument("--json")
    a = ap.parse_args()
    lex = EA.Lexicon(a.dict_dir, a.repo, "upstream/main")
    if a.paths:
        rx = re.compile(a.paths)
        files = [p for p in git(a.repo, "ls-tree", "-r", "--name-only", a.ref).split("\n") if p and rx.search(p)]
    else:
        files = sorted({p for p in git(a.repo, "log", "--since=" + a.since, "--diff-filter=A", "--name-only", "--format=", a.ref).split("\n") if p})
        live = set(git(a.repo, "ls-tree", "-r", "--name-only", a.ref).split("\n"))
        files = [p for p in files if p in live]
    files = [p for p in files if p.endswith(EXTS) and R.in_scope(p) and "/fixtures/" not in p and "kartenplan_data/" not in p
             and not p.startswith(("tools/release/",))]
    found = {}

    def add(word, kind, path):
        if not lex.german_ident(word):
            return
        e = found.setdefault(word, {"kinds": {}, "count": 0, "files": {}})
        e["kinds"][kind] = e["kinds"].get(kind, 0) + 1
        e["count"] += 1
        e["files"][path] = e["files"].get(path, 0) + 1

    for p in files:
        src = show(a.repo, a.ref, p)
        if not src:
            continue
        if p.endswith(".py"):
            try:
                for t in tokenize.generate_tokens(io.StringIO(src).readline):
                    if t.type == tokenize.NAME:
                        add(t.string, "name", p)
            except (tokenize.TokenError, IndentationError, SyntaxError):
                pass
            try:
                tree = ast.parse(src)
            except (SyntaxError, ValueError):
                tree = None
            for c in machine_constants(tree) if tree else ():
                if SHAPE.match(c):
                    add(c, "strkey", p)
        elif p.endswith(".json"):
            try:
                k, v = set(), set()
                walk_json(json.loads(src), k, v)
            except ValueError:
                continue
            for w in k:
                if SHAPE.match(w):
                    add(w, "json_key", p)
            for w in v:
                if SHAPE.match(w) and w == w.lower():
                    add(w, "json_value", p)
    # tree-wide spread of each word (ident_fix.py rewrites whole words in every tracked in-scope .py/.sh/.json/... file)
    for w, e in found.items():
        out = subprocess.run(["git", "-C", a.repo, "grep", "-lw", "-I", w, a.ref, "--"], capture_output=True, text=True).stdout.split("\n")
        paths = [x.split(":", 1)[1] for x in out if ":" in x]
        e["tree_files"] = len([x for x in paths if R.in_scope(x)])
        e["js_files"] = len([x for x in paths if x.endswith((".js", ".html")) and R.in_scope(x)])
        e["files"] = dict(sorted(e["files"].items(), key=lambda kv: -kv[1])[:6])
    res = {"ref": a.ref, "sha": git(a.repo, "rev-parse", a.ref).strip(), "files_scanned": len(files),
           "words": dict(sorted(found.items(), key=lambda kv: -kv[1]["count"]))}
    js = json.dumps(res, indent=1, ensure_ascii=False)
    if a.json:
        open(a.json, "w").write(js + "\n")
    print("ident_scan: files=%d german_words=%d" % (len(files), len(found)))


if __name__ == "__main__":
    main()
