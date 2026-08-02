import ast
import os
import re
import json
from collections import defaultdict

ROOT = os.environ.get("AUDIT421_ROOT", os.getcwd())
mods = [line.strip() for line in open("forkonly_srt.txt") if line.strip()]
allpy = []
for base, dirs, files in os.walk(ROOT):
    if "/.git" in base or "egg-info" in base:
        continue
    for f in files:
        if f.endswith(".py"):
            allpy.append(os.path.join(base, f))


def is_test(p):
    r = os.path.relpath(p, ROOT)
    return (
        r.startswith("test")
        or "/test" in "/" + r
        or os.path.basename(r).startswith("test_")
        or "/bench" in "/" + r
        or r.startswith("benchmark")
    )


TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
idx = defaultdict(lambda: [set(), set()])
for p in allpy:
    try:
        c = open(p, errors="ignore").read()
    except Exception:
        continue
    rel = os.path.relpath(p, ROOT)
    t = is_test(p)
    for tok in set(TOK.findall(c)):
        idx[tok][1 if t else 0].add(rel)
# also: is the MODULE itself imported by production?
modimp = defaultdict(lambda: [set(), set()])
for p in allpy:
    try:
        c = open(p, errors="ignore").read()
    except Exception:
        continue
    rel = os.path.relpath(p, ROOT)
    t = is_test(p)
    for m in re.findall(r"(?:from|import)\s+([A-Za-z0-9_.]+)", c):
        modimp[m][1 if t else 0].add(rel)
out = []
for rel in mods:
    p = os.path.join(ROOT, rel)
    try:
        tree = ast.parse(open(p, errors="ignore").read())
    except Exception:
        continue
    pub = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not n.name.startswith("_")
    ]
    if not pub:
        continue
    unre = []
    for n in pub:
        prod = idx[n.name][0] - {rel}
        if not prod:
            unre.append(n.name)
    dotted = rel[len("python/") : -3].replace("/", ".")
    mi_prod = modimp[dotted][0] - {rel}
    out.append(
        (
            len(unre) / len(pub),
            len(pub),
            len(unre),
            rel,
            sorted(mi_prod)[:3],
            sorted(unre)[:6],
        )
    )
out.sort(key=lambda r: (-r[0], -r[1]))
print("MODULES WHERE 100% OF PUBLIC API HAS NO PRODUCTION REFERENCE (>=3 public defs)")
print(f"{'frac':>5} {'pub':>4} {'unref':>5}  module   | prod-importers")
n = 0
for frac, npub, nun, rel, mi, names in out:
    if frac < 1.0 or npub < 3:
        continue
    n += 1
    print(f"{frac:5.2f} {npub:4d} {nun:5d}  {rel}")
    print(f"                 prod-importers: {mi if mi else 'NONE'}")
print(f"\ntotal such modules: {n}")
json.dump(out, open("modrank.json", "w"), indent=0)
