#!/usr/bin/env python3
"""persisted_consts_1009.py -- F0-K: list the byte constants and the format ids of a tree (python/**.py), keyed by the RENAMED path.

Why.  The RC1 acceptance (09.10.2026, `done/flliper-rename-abnahme-1009.md` section 2) found three renames that changed a spelling whose other
side is R2 must-keep: the C++ namespace of the JIT kernels (B1), the header magic of the L3 index snapshot (M1) and the seed of the salted KV page
keys (M2).  The kit's AST comparison hides string constants and the rest inventory counts what is left over; neither can see "renamed wrongly".
This tool takes the missing look: every `bytes` constant (magics, seeds, wire names -- the things that end up in files and hashes) and every
format id (`<name>/<n>`, `<name>-v<n>`) of the FREEZE tree, so a test can hold the renamed tree against it.

    python3 tools/release/persisted_consts_1009.py --repo /spinning/htsglang --ref 86ff356d0d > fixtures/f0k_persisted_consts_1009.json

The paths in the output are the paths of the RENAMED tree (`rename_to_flliper.rewrite_path(p, weg2=True)`), so the test needs no mapping.
`--root <worktree>` reads a checked-out tree instead (the test uses it for the current side).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rename_to_flliper as R  # noqa: E402

#: a versioned payload / format id: ``flliper.state/1``, ``weg2-footprint/1``, ``weg2-lane-coverage-1``, ``xyz-kv-namespace-v1``
FORMAT_ID = re.compile(r"^[A-Za-z][\w-]*(\.[\w-]+)*[/-]v?\d+$|^[\w.-]*-v\d+$")
OLD_WORD = re.compile(r"(?i)weg2|sglang")


def consts(text: str):
    """-> (sorted [repr(bytes)], sorted [format id with an old-family word]) of one source text; ([], []) when it does not parse"""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return [], []
    b, f = [], []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant):
            if isinstance(n.value, bytes):
                b.append(repr(n.value))
            elif isinstance(n.value, str) and len(n.value) < 80 and FORMAT_ID.match(n.value) and OLD_WORD.search(n.value):
                f.append(n.value)
    return sorted(b), sorted(f)


def from_git(repo: str, ref: str) -> Dict[str, Dict[str, List[str]]]:
    names = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only", ref], capture_output=True, text=True, check=True).stdout.split("\n")
    out: Dict[str, Dict[str, List[str]]] = {"bytes": {}, "format_ids": {}}
    for p in names:
        if not (p.startswith("python/") and p.endswith(".py")):
            continue
        text = subprocess.run(["git", "-C", repo, "show", f"{ref}:{p}"], capture_output=True, check=True).stdout.decode("utf-8", "replace")
        b, f = consts(text)
        q = R.rewrite_path(p, True)
        if b:
            out["bytes"][q] = b
        if f:
            out["format_ids"][q] = f
    return out


def from_root(root: str) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {"bytes": {}, "format_ids": {}}
    base = os.path.join(root, "python")
    for d, _dirs, files in os.walk(base):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            fp = os.path.join(d, fn)
            with open(fp, "rb") as fh:
                text = fh.read().decode("utf-8", "replace")
            b, f = consts(text)
            rel = os.path.relpath(fp, root).replace(os.sep, "/")
            if b:
                out["bytes"][rel] = b
            if f:
                out["format_ids"][rel] = f
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ref")
    g.add_argument("--root")
    ap.add_argument("--repo", default=".")
    a = ap.parse_args(argv)
    data = from_git(a.repo, a.ref) if a.ref else from_root(a.root)
    doc = {"_comment": "F0-K: bytes constants and old-family format ids of the FREEZE tree, keyed by the renamed path "
                       "(tools/release/persisted_consts_1009.py); test_f0k_persisted_consts_1009.py holds the renamed tree against it",
           "ref": a.ref or "", **data}
    json.dump(doc, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
