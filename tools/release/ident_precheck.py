#!/usr/bin/env python3
"""ident_precheck.py -- would ident_fix.py accept this identifier table on this tree? (F0-A, 07.10.2026; read only, no checkout)

ident_fix.py refuses a table when a target name already exists in a file it would touch, and it silently skips file types it
does not handle (.js, .html ...). Both findings surface only in the middle of the kit run (exit 3). This script answers them
up front for a git ref (or a directory), with the same rules as ident_fix.py:

  collision_refined  the same, but only where a merge is possible: both words as NAME (py) / as property or key (js, json) or both as an
              EXACT string constant (two keys of one dict) -- a word inside a longer string or in prose cannot merge anything
  collision   a file carries the old word (whole word) AND the new word (py: NAME token; md: inside backticks; other: any word)
  dup_target  two different old words map to the same new word, or a new word is itself an old word of the table
  unreached   a tracked in-scope file carries the old word but has an extension ident_fix.py does not rewrite (.js .html ...):
              the reader side of a renamed key would stay on the old name
  dir_component  a tracked DIRECTORY name carries the word whole (`profil_data/`): ident_fix.py moves file stems only, never directories,
              so the references would point at a directory that was not renamed
  file_stem   a tracked file stem carries the word (moved by ident_fix.py with `git mv`; imports/strings follow by the same table)
  conflicts   (--against) the key is also an entry of another kit table with a DIFFERENT target, or its target equals the target of another
              table's entry for a different key (two old words, one new word)
  per-word    number of files and occurrences, so that the table can be read as "what does this entry touch"

usage: ident_precheck.py <git-ref | directory> <map.json> [--repo R] [--json out.json] [--also map2.json ...] [--web]
--also: further tables applied BEFORE the first one (later entries of the first table win on the same key, as a merged table would).
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys
import tokenize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rename_to_flliper import in_scope  # noqa: E402

TEXT_EXT = (".sh", ".bash", ".env", ".md", ".txt", ".json", ".toml", ".cfg", ".ini", ".yaml", ".yml")


def names_py(src):
    try:
        return {t.string for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type == tokenize.NAME}
    except Exception:
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", src))


def refined_sets(path, src):
    """(names, exact_strings) of a file for the refined collision rule: two bindings or two KEYS can merge; a word inside a longer string or in prose cannot."""
    names, exact = set(), set()
    if path.endswith(".py"):
        import ast
        try:
            for t in tokenize.generate_tokens(io.StringIO(src).readline):
                if t.type == tokenize.NAME:
                    names.add(t.string)
                elif t.type == tokenize.STRING and not t.string.lower().startswith(("f", "rf", "fr", "b", "rb", "br")):
                    try:
                        v = ast.literal_eval(t.string)
                    except Exception:
                        continue
                    if isinstance(v, str) and len(v) < 80:
                        exact.add(v)
        except Exception:
            names |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", src))
    elif path.endswith(".json"):
        try:
            def walk(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        exact.add(k)
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)
                elif isinstance(o, str) and len(o) < 80:
                    exact.add(o)
            walk(json.loads(src))
        except ValueError:
            names |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", src))
    elif path.endswith((".js", ".html")):
        pat = r"\.([A-Za-z_][A-Za-z0-9_]*)|\b([A-Za-z_][A-Za-z0-9_]*)\s*:|\b(?:var|let|const|function)\s+([A-Za-z_][A-Za-z0-9_]*)"
        names |= {x for m in re.findall(pat, src) for x in m if x}
        exact |= set(re.findall(r"[\"']([A-Za-z_][A-Za-z0-9_]{1,40})[\"']", src))
    else:
        names |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", src))
    return names, exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    ap.add_argument("map")
    ap.add_argument("--repo", default="/spinning/htsglang")
    ap.add_argument("--also", action="append", default=[])
    ap.add_argument("--against", action="append", default=[], help="kit tables to compare keys/targets with (conflicts)")
    ap.add_argument("--web", action="store_true", help="count .js/.html as rewritten (the IDENT_FIX_WEB=1 mode of ident_fix.py)")
    ap.add_argument("--json")
    a = ap.parse_args()
    text_ext = TEXT_EXT + ((".js", ".html") if a.web else ())
    M = {}
    for f in a.also + [a.map]:
        M.update({k: v for k, v in json.load(open(f)).items() if k != "_comment"})
    is_dir = os.path.isdir(a.tree)

    def run(*cmd, **kw):
        return subprocess.run(cmd, capture_output=True, **kw)

    def grep_files(word):
        if is_dir:
            r = run("git", "-C", a.tree, "grep", "-lwI", word)
            return [x for x in r.stdout.decode("utf-8", "replace").split("\n") if x]
        r = run("git", "-C", a.repo, "grep", "-lwI", word, a.tree, "--")
        return [x.split(":", 1)[1] for x in r.stdout.decode("utf-8", "replace").split("\n") if ":" in x]

    def read(path):
        if is_dir:
            try:
                return open(os.path.join(a.tree, path), encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                return ""
        r = run("git", "-C", a.repo, "show", "%s:%s" % (a.tree, path))
        try:
            return r.stdout.decode("utf-8")
        except UnicodeDecodeError:
            return ""

    W = {k: re.compile(r"(?<![A-Za-z0-9_])" + re.escape(k) + r"(?![A-Za-z0-9_])") for k in M}
    cache = {}
    res = {"tree": a.tree, "entries": len(M), "collisions": [], "collisions_refined": [], "dup_target": [], "unreached": {}, "per_word": {}, "dir_component": {},
           "file_stem": {}, "conflicts": []}
    for f in a.against:
        other = {k: v for k, v in json.load(open(f)).items() if k != "_comment"}
        base = os.path.basename(f)
        for k, v in M.items():
            if k in other and other[k] != v:
                res["conflicts"].append({"table": base, "key": k, "this": v, "other": other[k]})
        rev = {}
        for k, v in other.items():
            rev.setdefault(v, []).append(k)
        for k, v in M.items():
            for ok_ in rev.get(v, []):
                if ok_ != k:
                    res["conflicts"].append({"table": base, "key": k, "this": v, "same_target_as": ok_})
    if is_dir:
        paths = [x for x in run("git", "-C", a.tree, "ls-files").stdout.decode("utf-8", "replace").split("\n") if x]
    else:
        paths = [x for x in run("git", "-C", a.repo, "ls-tree", "-r", "--name-only", a.tree).stdout.decode("utf-8", "replace").split("\n") if x]
    paths = [p for p in paths if in_scope(p)]
    dirs = {}
    stems = {}
    for p in paths:
        parts = p.split("/")
        for i, c in enumerate(parts[:-1]):
            dirs.setdefault(c, set()).add("/".join(parts[:i + 1]))
        stems.setdefault(os.path.splitext(parts[-1])[0], set()).add(p)
    tgt = {}
    for k, v in M.items():
        tgt.setdefault(v, []).append(k)
    for v, ks in tgt.items():
        if len(ks) > 1:
            res["dup_target"].append({"new": v, "old": sorted(ks)})
    for k, v in M.items():
        if v in M:
            res["dup_target"].append({"new": v, "old": [k], "note": "new word is itself an old word of the table"})
    for k, v in M.items():
        files = [f for f in grep_files(k) if in_scope(f)]
        occ = 0
        for f in files:
            if f not in cache:
                cache[f] = read(f)
            src = cache[f]
            occ += len(W[k].findall(src))
            if f.endswith(".py"):
                have = names_py(src)
            elif f.endswith(".md"):
                have = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", " ".join(re.findall(r"`[^`\n]+`", src))))
            elif f.endswith(text_ext):
                have = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", src))
            else:
                have = None
                res["unreached"].setdefault(k, []).append(f)
            if have is not None and v in have:
                res["collisions"].append({"file": f, "old": k, "new": v})
                names, exact = refined_sets(f, src)
                if (k in names and v in names) or (k in exact and v in exact):
                    res["collisions_refined"].append({"file": f, "old": k, "new": v,
                                                      "as": [w for w, ss in (("name", names), ("string", exact)) if k in ss and v in ss]})
        res["per_word"][k] = {"new": v, "files": len(files), "occurrences": occ}
        for c, ds in dirs.items():
            if W[k].search(c):
                res["dir_component"].setdefault(k, []).extend(sorted(ds)[:3])
        for c, ps in stems.items():
            if W[k].search(c):
                res["file_stem"].setdefault(k, []).extend(sorted(ps)[:3])
    res["summary"] = {"collisions": len(res["collisions"]), "collisions_refined": len(res["collisions_refined"]), "dup_target": len(res["dup_target"]), "conflicts": len(res["conflicts"]),
                      "words_with_dir_component": len(res["dir_component"]), "words_with_file_stem": len(res["file_stem"]),
                      "words_with_unreached_files": len(res["unreached"]),
                      "unreached_files": len({f for fs in res["unreached"].values() for f in fs})}
    if a.json:
        open(a.json, "w").write(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print("ident_precheck: %s" % json.dumps(res["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
