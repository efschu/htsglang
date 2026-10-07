#!/usr/bin/env python3
"""FL7: keep German every translated log/help unit whose text a test or an evaluator still matches.

A translated log line breaks every reader that matches its text: tests assert `'inkohaerent' in msg`, regexes in
assertRaisesRegex, evaluator scripts grep phrases. english_audit's must-keep list covers the operator's evaluators,
not these. Rule (exact, no heuristics about "German"): collect every string literal (>= 4 chars, with a letter) of the
tree's test and script files; a unit is PINNED when a literal matched the ORIGINAL text (substring, or as a regex) and
no longer matches the TRANSLATION. Pinned units get translation=None (they stay German before the release; unit and
pin are translated together afterwards).

usage: pinned_units.py <tree> <translated.jsonl> <out.jsonl>   (prints the pinned units to stderr as JSON lines)
"""
import ast, io, json, os, re, subprocess, sys, tokenize

TREE, SRC, OUT = sys.argv[1:4]
files = [f for f in subprocess.run(["git", "-C", TREE, "ls-files", "*.py"], capture_output=True, text=True).stdout.split()
         if re.match(r"(test|tests|scripts|benchmark)/", f) or "/test_" in f or "/checks/" in f]
import warnings
warnings.simplefilter("ignore")
_cache = {}


def literals(f):
    """string literals (>= 4 chars, a letter) of one reader file, and the ones that compile as regexes"""
    if f in _cache:
        return _cache[f]
    lits = set()
    try:
        src = open(os.path.join(TREE, f), encoding="utf-8").read()
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            if t.type == tokenize.STRING:
                try:
                    v = ast.literal_eval(t.string)
                except Exception:
                    continue
                if isinstance(v, str) and len(v) >= 4 and re.search(r"[A-Za-zÄÖÜäöüß]{4}", v):
                    lits.add(v)
    except Exception:
        src = ""
    rxs = []
    for v in lits:
        if re.search(r"[\\^$*+?()\[\]|]", v):
            try:
                rxs.append(re.compile(v))
            except re.error:
                pass
    _cache[f] = (src, lits, rxs)
    return _cache[f]


texts = {f: literals(f)[0] for f in files}


def readers(unit_file):
    """the reader files that name the unit's module (its stem) -- a test pins the logs of what it imports/runs"""
    stem = os.path.splitext(os.path.basename(unit_file))[0]
    if stem in ("__init__", "utils", "common"):
        stem = unit_file.replace("/", ".").rsplit(".", 1)[0]
    # the unit's own file is not a reader: its literals include the unit itself (it would always "pin" itself)
    return [f for f in files if f != unit_file and stem in texts[f]]


_Q = re.compile(r'^\s*[rRbBuUfF]{0,2}("""|\'\'\'|"|\')(.*)\1\s*$', re.S)


def body(tok):
    m = _Q.match(tok)
    return m.group(2) if m else tok


def spans(v, t, k=6):
    """v occurs in t, or (implicit string concatenation) t ends with a prefix / starts with a suffix of v of >= k chars"""
    b = body(t)
    if v in b:
        return True
    n = min(len(v), len(b))
    return any(b.endswith(v[:i]) or b.startswith(v[-i:]) for i in range(k, n + 1))


units = [json.loads(l) for l in open(SRC)]
pinned = 0
with open(OUT, "w") as o:
    for u in units:
        tr = u.get("translation")
        if tr and u["kind"] in ("log", "help"):
            t = u["text"]
            hit = None
            for f in readers(u["file"]):
                _, lits, rxs = literals(f)
                hit = next((v for v in lits if spans(v, t) and not spans(v, tr)), None) or \
                    next((r.pattern for r in rxs if r.search(t) and not r.search(tr)), None)
                if hit:
                    hit = f"{f}: {hit}"; break
            if hit is not None:
                pinned += 1
                print(json.dumps({"file": u["file"], "start": u["start"], "pin": hit[:80]}, ensure_ascii=False),
                      file=sys.stderr)
                u = dict(u, translation=None)
        o.write(json.dumps(u, ensure_ascii=False) + "\n")
print(json.dumps({"reader_files": len(files), "units": len(units), "pinned_kept_german": pinned}))
