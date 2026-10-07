#!/usr/bin/env python3
"""collision_proposal.py -- COLLISION_OK_FILE proposal for one line (F0-A, 07.10.2026; read only, no checkout).

usage: collision_proposal.py <worktree dir | ref:<git-ref>> <out.json> [--report out.txt]      (ref: REPO=<git-repo>)

The mechanical pass refuses a file in which two DIFFERENT old words become the same new word (`flliper` <- {flliper, sglang}:
the new modules write the schema id `flliper.server/1` next to imports of the old package). collision_survey.py lists them;
collision_auto.py (kit step 0) exempts only the ticket-label class PDFLIP/WEG2. This script classifies EVERY collision file of the
tree with one proof, the full-token proof of collision_auto.py generalised to all rule sets (token = maximal run of [A-Za-z0-9_] with inner `.`/`-`):

  exempt     no token of the file is mapped (by the engine's own rewrite, all rules) onto a DIFFERENT token that already exists
             in the same file. The colliding spellings only meet inside larger tokens (`flliper.server/1` next to `sglang.srt.x`), never
             as the same full word, so after the pass nothing in the file is merged: the exemption is cosmetic.
  unresolved a real full-token clash: `A` is mapped onto an existing different `B`. Needs a decision (rename one side by hand
             before the run, or pin the file); the kit would abort on it at step 1.

Files that COLLISION_OK (fixed in the engine) or $COLLISION_OK_FILE already cover are left out. Output JSON is a
COLLISION_OK_FILE ({path: [new words]}); the unresolved ones are only in the report. Same shape as collision_auto.py's output,
so `COLLISION_OK_FILE=<out.json>` is the wiring (not wired by this script).
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rename_to_flliper as R  # noqa: E402

# the same token shape as collision_fulltoken.py / collision_auto.py: a maximal run of [A-Za-z0-9_] that may contain `.` and `-` inside
TOK = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*[A-Za-z0-9_]|[A-Za-z0-9_]")


def iter_tree(root):
    if root.startswith("ref:"):
        return R.iter_ref(os.environ.get("REPO", "/spinning/htsglang"), root[4:])
    return R.iter_root(root)


def clashes(text):
    """[(token, mapped)] where mapped is a DIFFERENT token of the same file."""
    old = set(TOK.findall(text))
    out = []
    for t in old:
        new = R.rewrite_all(t, False, True, {})[0]
        if new != t and new in old:
            out.append((t, new))
    return sorted(out)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    root, out = argv[0], argv[1]
    report = argv[argv.index("--report") + 1] if "--report" in argv else None
    exempt, unresolved = {}, {}
    n = 0
    for path, mode, data in iter_tree(root):
        if mode == "120000":
            continue
        text = R.as_text(data)
        if text is None or not R.in_scope(path) or os.path.splitext(path)[1] in R.CXX_EXT:
            continue
        n += 1
        py = path.endswith(".py")
        clash = R.file_collisions(text, R.rewrite_all(text, py, True, {})[0])
        clash = {k: v for k, v in clash.items() if k not in R.COLLISION_OK.get(path, frozenset())}
        if not clash:
            continue
        real = clashes(text)
        if real:
            unresolved[path] = {"words": clash, "full_token": real[:8]}
        else:
            exempt[path] = sorted(clash)
    with open(out, "w") as f:
        json.dump(exempt, f, indent=1, sort_keys=True)
    lines = ["files scanned: %d" % n, "exempt by full-token proof: %d" % len(exempt), "unresolved (real full-token clash): %d" % len(unresolved)]
    for p, u in sorted(unresolved.items()):
        lines.append("UNRESOLVED %s %s" % (p, json.dumps(u, sort_keys=True)))
    if report:
        open(report, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:3]))
    for l in lines[3:]:
        print(l[:300])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
