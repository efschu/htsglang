#!/usr/bin/env python3
"""F0-G (fix round 1): compare the dry-runs of one profile, OLD (pre-rename tree + <profile>.env.alt) vs NEW (renamed tree + <profile>.env).

usage: drycmp.py --dir <logdir from run_pair.sh> --out <protocol dir> [--old-tree-sha S --new-tree-sha S] <profile> ...

Per profile ONE protocol file <out>/<profile>.txt.  What "0 diff modulo names" means here, measured and not claimed:

  1. the PROFILE_ARGS line (the profile's flags after sourcing, host-path mapped) equals after both name families are folded to one token;
  2. every ENVDUMP group (the env dict the launcher builds per group): same key set after folding, same values after folding (run-specific
     values -- boot token, evidence dir -- are listed as 'run-specific', not hidden);
  3. every other log line: matched to a line of the other side after folding names, digits and word order inside the line
     (digits = live-box readings and timings, order = the sort order of folded names inside a list);
  4. whatever is left over is listed IN FULL as residual, split into 'translated prose' (the old line carries a German word the new one does not)
     and 'UNEXPLAINED' (anything else).  The verdict line says which.

The boundary is stated per profile: a dry-run ends where the launcher refuses (config.json / W61 / W128 / W163 are model- and draft-file
questions this box cannot answer) -- everything the launcher computed BEFORE that line is compared, nothing after it exists.
"""
import argparse
import collections
import difflib
import json
import os
import re
import sys

NAMES = re.compile(r"WEG2|PDFLIP|Weg2|PdFlip|SGLANG|FLLIPER|weg2|pdflip|sglang|flliper")
# German words that mark a line as translated prose (the F0-D pass translated user-facing text German -> English)
GERMAN = re.compile(r"\b(keine|kein|nicht|ENTFAELLT|aktiv|inaktiv|Experten|Karte|Allokator|Spitze|Allokator-Spitze|diese|dieser|Zeile|damit|ist|eine|einen|"
                    r"und|oder|ohne|gilt|gemessen|je|Rang|weil|wird|werden|nur|fuer|ueber|statt|Wert|fehlt|gefunden|erwartet|Quelle|zuerst)\b|[\xe4\xf6\xfc\xc4\xd6\xdc\xdf]")
RUN_SPECIFIC_KEYS = ("BOOT_TOKEN", "XCHG_MANIFEST_DIR")


def fold(s):
    return NAMES.sub("NAME", s)


def read(p):
    return re.sub(r" \(DIRTY: .*?\) stamp", " stamp", open(p, errors="ignore").read(), flags=re.S)


def volatile(l, old_tree, new_tree, work):
    """The run-specific noise of one line: paths of the two trees and the scratch dir, time stamps, SHA, the dirty marker, plan ids."""
    for t in (old_tree, new_tree):
        if t:
            l = l.replace(t, "<tree>")
    l = l.replace(work, "<work>") if work else l
    l = re.sub(r"<work>/(home|ev|store-probe)/?(old|new)_", r"<work>/\1/X_", l)
    l = re.sub(r"\[?\d{4}-\d{2}-\d{2}[T ][0-9:.]+Z?\]? ?", "", l)
    l = re.sub(r"stamp=[0-9_]+", "stamp=S", l)
    l = re.sub(r"_\d{4}_[0-9]{6}", "_STAMP", l)
    l = re.sub(r"W\d{4} [0-9:.]+ \d+ ", "W ", l)
    l = re.sub(r"@ [0-9a-f]{10}( \((clean|DIRTY[^)]*|dirty[^)]*)\))?", "@ SHA", l)
    l = re.sub(r"sha256:[0-9a-f]{12}", "sha256:H", l)
    l = re.sub(r"plan_id=\S+", "plan_id=ID", l)
    l = re.sub(r"\b\d{10}\b", "EPOCH", l)
    return l


def skeleton(l):
    """Order- and digit-insensitive form of a (volatility-stripped) line: names folded, digits -> N, words sorted."""
    l = fold(l)
    l = re.sub(r"\d+(?:\.\d+)?", "N", l)
    return " ".join(sorted(t for t in re.split(r"[\s;,:]+", l) if t))


def split(path, old_tree, new_tree, work, dropped):
    env, rest, first, skip = {}, [], None, 0
    for i, l in enumerate(read(path).splitlines()):
        if skip:        # the source line a compile warning echoes
            skip -= 1
            continue
        if re.search(r": (Syntax|Deprecation)Warning: ", l):      # byte-compile noise of a fresh checkout (no .pyc yet), not launcher output
            skip = 1
            dropped.append(l)
            continue
        if l.startswith("compat:"):
            continue
        m = re.match(r"ENVDUMP (\S+) (\[.*\])$", l)
        if m:
            d = {}
            for k, v in json.loads(m.group(2)):
                d[fold(k)] = fold(volatile(str(v), old_tree, new_tree, work))
            env[m.group(1)] = d
        elif l.startswith("PROFILE_ARGS("):
            first = fold(volatile(l, old_tree, new_tree, work))
        else:
            rest.append(volatile(l, old_tree, new_tree, work))
    return env, rest, first


def boundary(log):
    tail = open(log, errors="ignore").read().strip().splitlines()
    last = tail[-1] if tail else "?"
    body = "\n".join(tail[-40:])
    m = re.search(r"\b(W\d{2,3}) \w+", body)
    if "DRY-RUN complete" in body and last == "EXIT=0":
        return "complete (plan to the end, nothing started)", last
    if m:
        return "refusal %s" % m.group(1), last
    if "no config.json describes" in body or "cannot read the model config" in body:
        return "refusal: config.json of the model is invisible on this box", last
    return "other: see the log tail", last


def compare(prof, d, a, out):
    ot, nt, work = a.old_tree, a.new_tree, a.work
    lo, ln = os.path.join(d, "old_%s.log" % prof), os.path.join(d, "new_%s.log" % prof)
    drop_o, drop_n = [], []
    eo, ro, fo = split(lo, ot, nt, work, drop_o)
    en, rn, fn = split(ln, ot, nt, work, drop_n)
    bo, xo = boundary(lo)
    bn, xn = boundary(ln)
    L = []
    P = L.append
    P("PROFILE %s" % prof)
    P("OLD  tree %s (pre-rename)   profile file %s.env.alt (pure old-spelling chain)" % (a.old_tree_sha, prof))
    P("NEW  tree %s (renamed)      profile file %s.env" % (a.new_tree_sha, prof))
    P("BOUNDARY old: %s   [%s]" % (bo, xo))
    P("BOUNDARY new: %s   [%s]" % (bn, xn))
    bad = 0
    # 1. flags
    same = fo == fn
    P("PROFILE_ARGS folded: %s" % ("equal" if same else "DIFFERENT"))
    if not same:
        bad += 1
        P("   old: " + str(fo)[:600]); P("   new: " + str(fn)[:600])
    # 2. env groups
    if sorted(eo) != sorted(en):
        bad += 1
        P("ENV groups DIFFER: old=%s new=%s" % (sorted(eo), sorted(en)))
    for g in sorted(set(eo) | set(en)):
        x, y = eo.get(g, {}), en.get(g, {})
        only_o, only_n = sorted(set(x) - set(y)), sorted(set(y) - set(x))
        chg = [k for k in sorted(set(x) & set(y)) if x[k] != y[k] and not any(r in k for r in RUN_SPECIFIC_KEYS)]
        run = [k for k in sorted(set(x) & set(y)) if x[k] != y[k] and any(r in k for r in RUN_SPECIFIC_KEYS)]
        P("ENV group %s: keys old=%d new=%d  only_old=%s only_new=%s  changed=%d  run-specific=%s" % (g, len(x), len(y), only_o, only_n, len(chg), run))
        if only_o or only_n or chg:
            bad += 1
        for k in chg[:20]:
            P("   changed %s: old=%s new=%s" % (k, x[k][:120], y[k][:120]))
    # 3. the other lines
    so, sn = [skeleton(l) for l in ro], [skeleton(l) for l in rn]
    co, cn = collections.Counter(so), collections.Counter(sn)
    only_o_sk, only_n_sk = co - cn, cn - co
    resid_o = [l for l, s in zip(ro, so) if only_o_sk.get(s, 0) > 0 and not only_o_sk.__setitem__(s, only_o_sk[s] - 1)]
    resid_n = [l for l, s in zip(rn, sn) if only_n_sk.get(s, 0) > 0 and not only_n_sk.__setitem__(s, only_n_sk[s] - 1)]
    exact = sum(1 for x, y in zip(ro, rn) if fold(x) == fold(y)) if len(ro) == len(rn) else None
    digits = sum(1 for x, y in zip(ro, rn) if fold(x) != fold(y) and re.sub(r"\d+(?:\.\d+)?", "N", fold(x)) == re.sub(r"\d+(?:\.\d+)?", "N", fold(y))) if len(ro) == len(rn) else None
    P("compile warnings dropped (fresh checkout, not launcher output): old=%d new=%d" % (len(drop_o), len(drop_n)))
    P("OTHER lines: old=%d new=%d  equal after name folding=%s  equal except digits (live-box readings/timings)=%s  matched by skeleton=%d" %
      (len(ro), len(rn), exact, digits, sum((co & cn).values())))
    # pair every residual old line with its closest new line and walk the word diff: a replaced chunk is a TRANSLATION when the old chunk
    # carries a German word the new chunk no longer has; anything else is unexplained
    unexplained = abs(len(resid_o) - len(resid_n))
    prose_o, pairs = [], []
    free = list(resid_n)
    for lo_ in resid_o:
        if not free:
            unexplained += 1
            continue
        best = max(free, key=lambda c: difflib.SequenceMatcher(None, fold(lo_).split(" "), fold(c).split(" ")).ratio())
        free.remove(best)
        aw, bw = fold(lo_).split(" "), fold(best).split(" ")
        chunks, ok_pair = [], True
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, aw, bw).get_opcodes():
            if tag == "equal":
                continue
            oc, nc = " ".join(aw[i1:i2]), " ".join(bw[j1:j2])
            if re.sub(r"\d+(?:\.\d+)?", "N", oc) == re.sub(r"\d+(?:\.\d+)?", "N", nc):
                chunks.append(("reading", oc, nc))
            elif tag == "replace" and ({m.group(0) for m in GERMAN.finditer(oc)} - {m.group(0) for m in GERMAN.finditer(nc)}):
                chunks.append(("translation", oc, nc))
            else:
                chunks.append(("UNEXPLAINED", oc, nc)); ok_pair = False
        if not ok_pair:
            unexplained += 1
        if any(c[0] == "translation" for c in chunks):
            prose_o.append(lo_)
        pairs.append("   pair: " + " ; ".join("%s [%s -> %s]" % (k, o[:80], n[:80]) for k, o, n in chunks))
    P("RESIDUAL lines old-side %d  new-side %d (full lines first, then the word-level pairing)" % (len(resid_o), len(resid_n)))
    for l in resid_o:
        P("   - " + l)
    for l in resid_n:
        P("   + " + l)
    for x in pairs:
        P(x)
    if bad or unexplained:
        verdict = "DIFFERENT: %d structural difference(s), %d unexplained residual line(s)" % (bad, unexplained)
        ok = False
    elif resid_o or resid_n:
        verdict = "0 diff modulo names for flags and env; %d translated prose line(s) (German -> English, listed above), nothing unexplained" % len(prose_o)
        ok = True
    else:
        verdict = "0 diff modulo names"
        ok = True
    P("VERDICT %s: %s (up to the boundary above)" % ("OK" if ok else "FAIL", verdict))
    open(os.path.join(out, prof + ".txt"), "w").write("\n".join(L) + "\n")
    return ok, bo, bn, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--old-tree", default="")
    ap.add_argument("--new-tree", default="")
    ap.add_argument("--old-tree-sha", default="?")
    ap.add_argument("--new-tree-sha", default="?")
    ap.add_argument("--work", default="")
    ap.add_argument("profiles", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, rc = [], 0
    for p in a.profiles:
        ok, bo, bn, v = compare(p, a.dir, a, a.out)
        rows.append((p, ok, bo, bn, v))
        rc |= 0 if ok else 1
        print("%-24s %-4s old: %-44s new: %-44s %s" % (p, "OK" if ok else "FAIL", bo[:44], bn[:44], v[:100]))
    return rc


if __name__ == "__main__":
    sys.exit(main())
