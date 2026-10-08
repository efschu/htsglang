#!/usr/bin/env python3
"""F0-G fix round 1: protocols/SUMMARY.txt from the per-profile protocols written by drycmp.py.  usage: summary.py <protocol dir> [--check]"""
import collections
import glob
import os
import re
import sys


def cls(b):
    if b.startswith("complete"):
        return "complete"
    if "config.json" in b:
        return "config.json"
    m = re.search(r"W\d+", b)
    return m.group(0) if m else b


def build(pdir):
    rows = []
    for f in sorted(glob.glob(os.path.join(pdir, "*.txt"))):
        n = os.path.basename(f)[:-4]
        if n == "SUMMARY":
            continue
        t = open(f).read()
        bo = re.search(r"BOUNDARY old: (.*?)   \[", t).group(1)
        bn = re.search(r"BOUNDARY new: (.*?)   \[", t).group(1)
        v = re.search(r"VERDICT (\w+): ", t).group(1)
        tr = len(re.findall(r"^   pair: ", t, re.M))
        envg = re.findall(r"^ENV group (\w): keys old=(\d+) new=(\d+)", t, re.M)
        oth = re.search(r"OTHER lines: old=(\d+) new=(\d+)", t).groups()
        rows.append((n, bo, bn, v, tr, envg, oth))
    L = ["F0-G dry-run comparison, old (pre-rename tree + <profile>.env.alt) vs new (renamed tree + <profile>.env), %d profiles" % len(rows),
         "27B line: old tree dab894b021, new tree = desk/flliper-27b-f0g-1008 (11 profiles). NF line: old tree fd9bf48fe9, new tree = desk/flliper-nf-f0g-1008 (5 profiles).",
         "New tree = the F0-G commit plus the working tree of fix round 1 (the .env files and the launcher code are those of the commit; the .alt files are the fixed ones).",
         "Raw logs: /spinning/flliper/work/release/f0g-1008/{27b,nf}/old_<profile>.log, new_<profile>.log (outside the tree).", "",
         "%-22s %-11s %-8s %-6s %s" % ("profile", "boundary", "verdict", "prose", "env groups (keys old/new) | other lines old/new")]
    for n, bo, bn, v, tr, envg, oth in rows:
        b = cls(bo) if cls(bo) == cls(bn) else cls(bo) + "/" + cls(bn)
        L.append("%-22s %-11s %-8s %-6s %s | %s/%s" % (n, b, v, tr, " ".join("%s:%s/%s" % e for e in envg) or "none (stops before the env is built)", oth[0], oth[1]))
    c = collections.Counter(cls(r[1]) for r in rows)
    refused = c["config.json"] + c["W163"] + c["W128"]
    L += ["", "COUNT: %d profiles. Stop where the launcher refuses for model/draft files this box cannot see: %d (config.json %d, W163 %d, W128 %d)." %
          (len(rows), refused, c["config.json"], c["W163"], c["W128"]),
          "       Stop at the plan's own infeasibility (W64): %d. Plan complete (DRY-RUN complete, EXIT=0): %d.  %d + %d + %d = %d." %
          (c["W64"], c["complete"], refused, c["W64"], c["complete"], len(rows)),
          "VERDICT: %d OK, %d FAIL." % (sum(r[3] == "OK" for r in rows), sum(r[3] != "OK" for r in rows)),
          "LIMIT (stated, not a success): behind its boundary line nothing was compared. %d profiles are compared only up to the first refusal; %d reach a plan (%d complete, %d W64)." %
          (refused, c["W64"] + c["complete"], c["complete"], c["W64"]),
          "'prose' = residual line pairs that differ only by a German -> English translation (word level, listed in the protocol); they are the only lines that are not",
          "equal modulo names, digits (live-box readings) and name sort order."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    d = sys.argv[1]
    text = build(d)
    p = os.path.join(d, "SUMMARY.txt")
    if "--check" in sys.argv:
        sys.exit(0 if os.path.exists(p) and open(p).read() == text else 1)
    open(p, "w").write(text)
    print(text, end="")
