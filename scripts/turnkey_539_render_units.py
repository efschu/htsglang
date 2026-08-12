#!/usr/bin/env python
# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Render the #539 systemd units from the stack config that will run them.

WHY, measured on this rig 2026-08-12. All five units hardcoded
``/spinning/htsglang-gpu`` for PYTHONPATH and for the interpreter, and the
installer copied them byte for byte (cmp/install, no substitution). That
checkout predates the turnkey merge, so every unit died with ``No module named
sglang.srt.turnkey`` and the serving unit died on the failed dependency.
``[stack].repo`` reads as the single source of truth for where the stack lives
and was not one: nothing connected it to what the units actually executed, and
the divergence stayed silent right up to the import error.

So the units carry ``@@PLACEHOLDER@@`` tokens and the installer renders them.
A unit that still holds a literal path cannot follow the config, and a
renderer that leaves a placeholder in place would install a unit that fails at
start with a path nobody can grep for -- both are refused rather than written.

Rendering lives here, in Python, rather than in installer sed, so it can be
unit-tested against a config whose repo is somewhere else entirely. That test
(test/registered/unit/turnkey/test_unit_rendering_539.py) is the falsifier for
this whole change.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, Iterable, Mapping

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from sglang.srt.turnkey import config as C            # noqa: E402

#: Every placeholder a unit may use, with what it means. A unit using anything
#: else is a typo that would otherwise reach /etc.
PLACEHOLDERS: Dict[str, str] = {
    "@@REPO@@": "[stack].repo -- the checkout the stack runs from",
    "@@VENV@@": "[stack].venv -- the interpreter's virtualenv",
    "@@LOG_DIR@@": "[stack].log_dir -- where lane logs are written",
    "@@CONFIG@@": "the stack config the units read at runtime",
}

#: Where the units expect to find the config once installed. It is not the
#: file rendering READS -- the installer may render from the shipped template
#: before the config has been seeded.
DEFAULT_CONFIG_PATH = "/etc/htsglang/stack.toml"

UNITS = ("htsglang.target", "htsglang-preflight.service",
         "htsglang-planner.service", "htsglang-serving@.service",
         "htsglang-watchdog@.service")

_LEFTOVER = re.compile(r"@@[A-Za-z_]+@@")


class RenderError(Exception):
    """A unit could not be rendered into something safe to install."""


def substitutions_from_config(path: str,
                              config_path: str = DEFAULT_CONFIG_PATH,
                              ) -> Dict[str, str]:
    """Read [stack] and return the substitution map.

    Uses the turnkey config loader rather than a private TOML read, so a
    config this rejects is a config the units would have been built from
    incorrectly.
    """
    cfg = C.load(path)
    return {
        "REPO": cfg.repo.rstrip("/"),
        "VENV": cfg.venv.rstrip("/"),
        "LOG_DIR": cfg.log_dir.rstrip("/"),
        "CONFIG": config_path,
    }


def render_text(text: str, subs: Mapping[str, str],
                source: str = "<unit>") -> str:
    """Substitute every placeholder, and refuse to return text still holding
    one. A half-rendered unit installs cleanly and fails at start."""
    out = text
    for name, value in subs.items():
        out = out.replace(f"@@{name}@@", value)
    left = _LEFTOVER.findall(out)
    if left:
        raise RenderError(
            f"{source}: unresolved placeholder(s) {', '.join(sorted(set(left)))}"
            f"; known placeholders are {', '.join(sorted(PLACEHOLDERS))}")
    return out


def render_tree(src_dir: str, dst_dir: str, subs: Mapping[str, str],
                units: Iterable[str] = UNITS) -> Dict[str, str]:
    """Render every unit from ``src_dir`` into ``dst_dir``; return the paths."""
    os.makedirs(dst_dir, exist_ok=True)
    written: Dict[str, str] = {}
    for unit in units:
        src = os.path.join(src_dir, unit)
        try:
            with open(src, "r") as fh:
                text = fh.read()
        except OSError as e:
            raise RenderError(f"cannot read unit {src}: {e}") from e
        dst = os.path.join(dst_dir, unit)
        with open(dst, "w") as fh:
            fh.write(render_text(text, subs, source=unit))
        os.chmod(dst, 0o644)
        written[unit] = dst
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Render the #539 units from [stack].repo/.venv/.log_dir")
    ap.add_argument("--config", required=True,
                    help="stack config to read [stack] from")
    ap.add_argument("--src", required=True, help="directory of unit templates")
    ap.add_argument("--dst", required=True, help="directory to render into")
    ap.add_argument("--config-path", default=DEFAULT_CONFIG_PATH,
                    help="path the RENDERED units will read at runtime "
                         f"(default {DEFAULT_CONFIG_PATH})")
    a = ap.parse_args(argv)

    try:
        subs = substitutions_from_config(a.config, a.config_path)
        render_tree(a.src, a.dst, subs)
    except RenderError as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 1
    for name in sorted(subs):
        print(f"  @@{name}@@ -> {subs[name]}")

    # The units now follow [stack].venv. The LANE ARGV does not: argv[0] is
    # data carried from the capture, and nothing ties it to [stack].venv. That
    # is the same silent divergence one level down, so it is at least said out
    # loud rather than discovered as a second import error.
    cfg = C.load(a.config)
    for lane in cfg.serving:
        argv = list(getattr(lane, "argv", ()) or ())
        if argv and not argv[0].startswith(subs["VENV"] + "/"):
            print(f"  WARNING lane {lane.name}: argv[0] is {argv[0]}, which "
                  f"is not under [stack].venv ({subs['VENV']}). The units "
                  f"follow the config; this lane's interpreter does not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
