"""Characterization of ``launcher.build_parser`` (refactor 28.10.): the launcher's parser, action by action, against a golden taken
BEFORE ``build_parser`` was split into ``_add_args_*`` functions.

What a split can silently break: the ORDER flags are registered in (``--help`` order, which flag wins an abbreviation), a lost or
duplicated flag, a changed default / type / choices / help text. The golden pins all of them per action (help as a hash) plus the
set of option strings, so a later edit of any flag shows up as a diff of exactly that flag. Regenerate it on purpose with
``python test/registered/unit/pdflip/test_launcher_parser_golden_1028.py --regen``.

Environment: defaults read the process environment (``FLLIPER_*`` / ``SGLANG_*`` switches); the test runs with those removed so the
golden does not depend on the machine that runs it.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

GOLDEN = pathlib.Path(__file__).with_name("golden_launcher_parser_1028.json")


def _tname(t):
    return None if t is None else getattr(t, "__name__", repr(t))


def _snapshot() -> dict:
    from flliper.srt.pdflip import launcher as L

    clean = {k: v for k, v in os.environ.items() if not k.startswith(("FLLIPER_", "SGLANG_"))}
    with mock.patch.dict(os.environ, clean, clear=True):
        ap = L.build_parser()
    rows = []
    for a in ap._actions:
        rows.append({
            "opts": list(a.option_strings), "dest": a.dest, "cls": type(a).__name__, "type": _tname(a.type), "nargs": repr(a.nargs),
            "required": a.required, "choices": None if a.choices is None else [repr(c) for c in a.choices], "default": repr(a.default),
            "metavar": repr(a.metavar), "help_sha1": hashlib.sha1((a.help or "").encode()).hexdigest()[:16],
        })
    return {"prog": ap.prog, "n_actions": len(rows), "option_strings": sorted(ap._option_string_actions), "actions": rows}


def test_the_parser_is_the_golden_action_by_action_and_in_order():
    got, want = _snapshot(), json.loads(GOLDEN.read_text())
    assert got["n_actions"] == want["n_actions"], (got["n_actions"], want["n_actions"])
    assert got["option_strings"] == want["option_strings"]
    diffs = [(i, g["opts"] or g["dest"]) for i, (g, w) in enumerate(zip(got["actions"], want["actions"])) if g != w]
    assert not diffs, f"flags that differ from the golden (position, flag): {diffs[:8]}"


if __name__ == "__main__":
    if "--regen" in sys.argv:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4] / "python"))
        GOLDEN.write_text(json.dumps(_snapshot(), indent=0, sort_keys=True))
        print("golden written:", GOLDEN)
