#!/usr/bin/env python3
"""FL7 identifier fix (RM 28.09.): the German identifiers the mechanical rename cannot take, WITH their pins.

rename_to_flliper.py --ident-map rewrites Python NAME tokens only. What is left after it (english_audit
ident-proposals on the renamed tree) is left for a reason: the name also lives in a string (source pins,
hasattr/getattr, monkeypatch targets, JSON keys) or in a non-Python file (shell env names), or its natural English
name collides in one of its files. This script renames exactly those, consistently:
  * .py   -- NAME tokens and whole-word occurrences INSIDE string tokens; comments untouched (the prose pass);
  * .sh / .env / .bash -- whole-word occurrences on code lines (a line whose first non-blank is '#' is prose);
  * .md   -- whole-word occurrences inside `backticks` only;
  * file stems -- `git mv` when a path component carries the old name (then its references follow via the rules above).
Before writing it refuses a target name that already exists as a NAME token (py) / word (other) in a file it would
touch -- a collision must be resolved by choosing another name, never silently merged.

usage: ident_fix.py <tree> <map.json> [--dry-run]

F0-A (07.10.2026): IDENT_FIX_WEB=1 additionally rewrites .js/.html (whole word on every line that is not a `//`-comment): the dashboard's
static pages READ the keys that the planner modules WRITE (`r.werte`, `e.zustand`); a key renamed on the Python/JSON side only would leave
the reader on the old name. Off by default (the 28.09. behaviour); data/ident_map_1007.json needs it (ident_precheck.py lists the files).
IDENT_FIX_COLLISION=refined: a target counts as taken only where a merge is possible (both words as NAME / property / key, or both as an
exact string constant -- ident_precheck.refined_sets); a common English word in prose or inside a longer string (`value`, `source`, `state`)
no longer aborts the run. Default stays the strict rule of 28.09. (any NAME of the file, any word of a non-Python file).
"""
import io, json, os, re, subprocess, sys, tokenize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rename_to_flliper import in_scope  # same content scope as the mechanical rename (docs/, rust/, 3rdparty/ ... stay)

ROOT, MAPF = sys.argv[1], sys.argv[2]
DRY = "--dry-run" in sys.argv
WEB = os.environ.get("IDENT_FIX_WEB") == "1"
REFINED = os.environ.get("IDENT_FIX_COLLISION") == "refined"
WEB_EXT = (".js", ".html")
# F0-A (07.10.2026): only `_comment` is metadata. The 28.09. filter `not k.startswith("_")` silently dropped every underscore-prefixed
# identifier of the table (identfix_map.json: `_karte`, `_rang_karte`, `_karten_residenz_lokal`, `_platzhalter`, `_form_a_extend_set_riegel`
# were never renamed; merged_0928.json keeps them, the engine filters `_comment` only).
M = {k: v for k, v in json.load(open(MAPF)).items() if k != "_comment"}
W = {k: re.compile(r"(?<![A-Za-z0-9_])" + re.escape(k) + r"(?![A-Za-z0-9_])") for k in M}


def git(*a):
    return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True, check=True).stdout


def files_with(name):
    r = subprocess.run(["git", "-C", ROOT, "grep", "-lw", name], capture_output=True, text=True)
    return [f for f in r.stdout.split() if f]


def fix_py(src):
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    lines = src.splitlines(keepends=True)
    off = [0]
    for ln in lines:
        off.append(off[-1] + len(ln))
    edits = []   # (start, end, new)
    for t in toks:
        s = off[t.start[0] - 1] + t.start[1]; e = off[t.end[0] - 1] + t.end[1]
        if t.type == tokenize.NAME and t.string in M:
            edits.append((s, e, M[t.string]))
        elif t.type == tokenize.STRING or (hasattr(tokenize, "FSTRING_MIDDLE") and t.type == tokenize.FSTRING_MIDDLE):
            new = t.string
            for k, rx in W.items():
                new = rx.sub(M[k], new)
            if new != t.string:
                edits.append((s, e, new))
    for s, e, new in sorted(edits, reverse=True):
        src = src[:s] + new + src[e:]
    return src, len(edits)


def names_py(src):
    try:
        return {t.string for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type == tokenize.NAME}
    except Exception:
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", src))


def fix_text(path, src):
    n = 0
    out = []
    if path.endswith(".md"):
        def bt(m):
            nonlocal n
            s = m.group(0)
            for k, rx in W.items():
                s2 = rx.sub(M[k], s); n += s2 != s; s = s2
            return s
        return re.sub(r"`[^`\n]+`", bt, src), n
    for ln in src.splitlines(keepends=True):
        if not ln.lstrip().startswith(("//",) if path.endswith(WEB_EXT) else ("#",)):
            for k, rx in W.items():
                ln2 = rx.sub(M[k], ln); n += ln2 != ln; ln = ln2
        out.append(ln)
    return "".join(out), n


touched = sorted({f for k in M for f in files_with(k) if in_scope(f)})
# collisions: a target that already exists in a file we touch
coll = []
for f in touched:
    p = os.path.join(ROOT, f)
    try:
        src = open(p, encoding="utf-8").read()
    except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
        continue
    if f.endswith(".py"):
        have = names_py(src)
    elif f.endswith(".md"):
        have = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", " ".join(re.findall(r"`[^`\n]+`", src))))
    else:
        have = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", src))
    if REFINED:
        from ident_precheck import refined_sets
        names_r, exact_r = refined_sets(f, src)
    for k, v in M.items():
        if W[k].search(src) and v in have:
            if REFINED and not ((k in names_r and v in names_r) or (k in exact_r and v in exact_r)):
                continue
            coll.append((f, k, v))
if coll:
    print(json.dumps({"REFUSED_collisions": coll[:40]}, indent=1)); sys.exit(3)

stats = {"files": 0, "edits": 0, "moves": []}
for f in touched:
    p = os.path.join(ROOT, f)
    try:
        src = open(p, encoding="utf-8").read()
    except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
        continue
    if f.endswith(".py"):
        new, n = fix_py(src)
    elif f.endswith((".sh", ".bash", ".env", ".md", ".txt", ".json", ".toml", ".cfg", ".ini", ".yaml", ".yml") + (WEB_EXT if WEB else ())):
        new, n = fix_text(f, src)
    else:
        continue
    if new != src:
        stats["files"] += 1; stats["edits"] += n
        if not DRY:
            open(p, "w", encoding="utf-8").write(new)
# file stems
for f in git("ls-files").split("\n"):
    if not f or not in_scope(f):
        continue
    parts = f.split("/")
    stem, ext = os.path.splitext(parts[-1])
    if stem in M or any(W[k].search(stem) for k in M):
        new_stem = stem
        for k, rx in W.items():
            new_stem = rx.sub(M[k], new_stem)
        dst = "/".join(parts[:-1] + [new_stem + ext])
        stats["moves"].append([f, dst])
        if not DRY:
            git("mv", f, dst)
print(json.dumps(stats))
