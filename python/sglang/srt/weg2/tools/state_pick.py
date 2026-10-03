#!/usr/bin/env python3
"""ITEM 330: pick the state dir a DRY check may read, instead of trusting ``state/current``.

``state_file.init`` repoints ``<root>/current`` at the NEW boot id the moment a start enters
``preflight``.  A start the host guard refuses (lifecycle ``refused_preflight``, ``groups`` still
``{}``) therefore becomes ``current`` too, and the next DRY check that reads
``current/state.json`` (``#791C ABORT-MID-CHUNK`` plan: P url + chunk from ``groups.P.launch.argv``)
fails with ``p=None`` although the image is fine (03.10. 08:24Z).

Other readers (boot_quiet.sh, progress_watch.py) NEED ``current`` to follow a start from
``preflight`` on, so the pointer stays as it is and the check chooses:

  1. the ``current`` target, if it is a kind=boot state that carries the needed group(s);
  2. else the newest kind=boot state (by the UTC stamp in its boot id, then state.json mtime)
     that carries them;
  3. else nothing (exit 1) -- the caller keeps ``current`` and its check fails by name.

Stdlib only; no GPU, no server.  CLI (stdout line 1 = chosen state.json path, line 2 = note):

    python3 state_pick.py --root $ACC/state [--need P[,D]]
"""

import argparse
import json
import os
import re
import sys

_STAMP = re.compile(r"(\d{8}T\d{6}Z)")


def _load(d):
    try:
        with open(os.path.join(d, "state.json")) as fh:
            st = json.load(fh)
    except (OSError, ValueError):
        return None
    return st if isinstance(st, dict) else None


def has_groups(st, need=()):
    """True when ``groups`` is non-empty and every group in ``need`` has a non-empty record."""
    groups = st.get("groups")
    if not isinstance(groups, dict) or not groups:
        return False
    return all(isinstance(groups.get(g), dict) and groups.get(g) for g in need)


def _sort_key(root, name):
    m = _STAMP.search(name)
    try:
        mt = os.path.getmtime(os.path.join(root, name, "state.json"))
    except OSError:
        mt = 0.0
    return (m.group(1) if m else "", mt)


def _lc(st):
    return (st.get("lifecycle") or {}).get("state")


def pick(root, need=()):
    """Return (state_dir | None, note).  ``root`` is ``<ACC>/state``."""
    cur = os.path.join(root, "current")
    cur_name = os.path.basename(os.path.realpath(cur)) if os.path.lexists(cur) else None
    cur_st = _load(os.path.join(root, cur_name)) if cur_name else None
    if cur_st is not None and cur_st.get("kind") == "boot" and has_groups(cur_st, need):
        return os.path.join(root, cur_name), f"current={cur_name} ({_lc(cur_st)}) hat groups"
    why = "nicht lesbar" if cur_st is None else f"{_lc(cur_st)}, groups={sorted((cur_st.get('groups') or {}))}"
    names = [n for n in os.listdir(root)
             if n not in ("current",) and not os.path.islink(os.path.join(root, n))
             and os.path.isfile(os.path.join(root, n, "state.json"))]
    for n in sorted(names, key=lambda x: _sort_key(root, x), reverse=True):
        st = _load(os.path.join(root, n))
        if st is None or st.get("kind") != "boot" or not has_groups(st, need):
            continue
        return os.path.join(root, n), f"current={cur_name} uebersprungen ({why}); neuester State mit groups: {n} ({_lc(st)})"
    return None, f"current={cur_name} ({why}); kein State mit groups unter {root}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="<ACC>/state")
    ap.add_argument("--need", default="", help="comma list of groups that must be non-empty (e.g. P)")
    a = ap.parse_args(argv)
    need = tuple(g for g in a.need.split(",") if g)
    try:
        d, note = pick(a.root, need)
    except OSError as e:
        print("", flush=True)
        print(f"state_pick: {e}")
        return 1
    print(os.path.join(d, "state.json") if d else "")
    print(note)
    return 0 if d else 1


if __name__ == "__main__":
    sys.exit(main())
