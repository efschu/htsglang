import os
import re
import json
import sys

ROOT = os.environ.get("AUDIT421_ROOT", os.getcwd())
PY = []
for base, dirs, files in os.walk(ROOT):
    if any(s in base for s in ("/.git", "/node_modules", "/build/", ".egg-info")):
        continue
    for f in files:
        if f.endswith(
            (".py", ".sh", ".md", ".yaml", ".yml", ".json", ".cu", ".cpp", ".h")
        ):
            PY.append(os.path.join(base, f))


def is_test(p):
    r = os.path.relpath(p, ROOT)
    return (
        "/test" in "/" + r
        or r.startswith("test")
        or "/tests/" in "/" + r
        or os.path.basename(r).startswith("test_")
        or "/benchmark" in "/" + r
        or "/bench" in "/" + r
    )


def is_doc(p):
    r = os.path.relpath(p, ROOT)
    return r.endswith((".md", ".json", ".yaml", ".yml")) or r.startswith("docs/")


def is_decl(p):
    r = os.path.relpath(p, ROOT)
    return r in ("python/sglang/srt/server_args.py", "python/sglang/srt/environ.py")


CACHE = {}


def content(p):
    if p not in CACHE:
        try:
            CACHE[p] = open(p, errors="ignore").read()
        except Exception:
            CACHE[p] = ""
    return CACHE[p]


def scan(names, wordbound=True):
    res = {n: {"prod": [], "test": [], "doc": [], "decl": 0} for n in names}
    pat = {n: re.compile(r"\b" + re.escape(n) + r"\b") for n in names}
    for p in PY:
        c = content(p)
        for n in names:
            if n not in c:
                continue
            k = len(pat[n].findall(c))
            if not k:
                continue
            if is_decl(p):
                res[n]["decl"] += k
            elif is_doc(p):
                res[n]["doc"].append((os.path.relpath(p, ROOT), k))
            elif is_test(p):
                res[n]["test"].append((os.path.relpath(p, ROOT), k))
            else:
                res[n]["prod"].append((os.path.relpath(p, ROOT), k))
    return res


which = sys.argv[1]
names = [line.strip() for line in open(sys.argv[2]) if line.strip()]
r = scan(names)
json.dump(r, open(f"{which}_scan.json", "w"), indent=0)
rows = []
for n in names:
    d = r[n]
    prod = sum(k for _, k in d["prod"])
    test = sum(k for _, k in d["test"])
    doc = sum(k for _, k in d["doc"])
    if prod == 0 and test == 0:
        cls = "DEAD"
    elif prod == 0:
        cls = "TEST-ONLY"
    else:
        cls = "has-prod"
    rows.append((cls, n, prod, test, doc, len(d["prod"])))
for cls in ("DEAD", "TEST-ONLY", "has-prod"):
    sel = [x for x in rows if x[0] == cls]
    print(f"\n===== {cls} ({len(sel)}) =====")
    for c, n, p, t, dd, nf in sorted(sel, key=lambda x: x[2]):
        print(f"  {n:55s} prod={p:4d} files={nf:3d} test={t:4d} doc={dd:3d}")
