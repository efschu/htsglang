#!/usr/bin/env python3
"""catalog_rev_check_1008.py [--root <worktree>] [--catalog <catalog.json>]   (F0-I fix round 1)

The shipped catalog.json cites source lines (``lesestellen``, ``source.line``) of the trees it was built from; its ``tree_rev`` names the
commits (``27b=<sha>+nf=<sha>``).  A line number is only true if the cited file is byte-identical between that commit and the tree that ships the
catalog.  This check compares, for every revision of ``tree_rev`` that is an ancestor of HEAD (= the revision of the line this tree belongs to), the
files cited by the catalog under ``python/flliper/srt/`` between the revision and HEAD.  Exit 0 = no cited file changed since the build, exit 1 =
drift (the catalog has to be rebuilt from a commit that contains the final text of the cited files), exit 2 = nothing could be checked.

It is a RELEASE gate (F0-I), not a unit test of every commit: any later edit of a cited file (launcher.py ...) makes it red on purpose.
"""
import argparse
import json
import os
import re
import subprocess
import sys

CATALOG = os.path.join("tools", "rig_dashboard", "rigdash", "profil_data", "catalog.json")
SRT = "python/flliper/srt/"


def _git(root, *args):
    return subprocess.run(["git", "-C", root] + list(args), capture_output=True, text=True)


def cited_files(cat):
    """every ``file`` the catalog cites (relative to python/flliper/srt/)"""
    files = set()
    for e in cat.get("entries", {}).values():
        f = (e.get("source") or {}).get("file")
        if f:
            files.add(f)
        for s in e.get("lesestellen") or []:
            m = re.match(r"(.+):\d+$", s)
            if m:
                files.add(m.group(1))
    return sorted(files)


def check(root, catalog_path=None):
    """-> (results, drift): results = [(label, rev, state, changed_files)]; state = checked | not-ancestor | unknown-rev"""
    cat = json.load(open(catalog_path or os.path.join(root, CATALOG), encoding="utf-8"))
    revs = dict(p.split("=", 1) for p in str(cat.get("tree_rev", "")).split("+") if "=" in p)
    paths = [SRT + f for f in cited_files(cat)]
    results = []
    drift = []
    for label, rev in sorted(revs.items()):
        if _git(root, "cat-file", "-e", rev + "^{commit}").returncode != 0:
            results.append((label, rev, "unknown-rev", []))
            continue
        if _git(root, "merge-base", "--is-ancestor", rev, "HEAD").returncode != 0:
            results.append((label, rev, "not-ancestor", []))
            continue
        out = _git(root, "diff", "--name-only", rev, "HEAD", "--", *paths).stdout.split()
        results.append((label, rev, "checked", out))
        drift += [(label, rev, f) for f in out]
    return results, drift


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--catalog", default="")
    ns = ap.parse_args(argv)
    results, drift = check(os.path.abspath(ns.root), ns.catalog or None)
    for label, rev, state, files in results:
        print("%s=%s: %s%s" % (label, rev, state, (" -- changed since: " + ", ".join(files)) if files else ""))
    if not any(s == "checked" for _l, _r, s, _f in results):
        print("nothing could be checked (no revision of tree_rev is an ancestor of HEAD)", file=sys.stderr)
        return 2
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
