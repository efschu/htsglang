#!/usr/bin/env python
# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Make a captured ship environment usable, and checkable, as the only copy.

WHY THIS EXISTS, measured 2026-08-12. scripts/route_a_631_prod_boot.sh kept
its own hand-maintained copy of the ship environment. It had drifted from the
capture in seven keys -- five dropped, one added, and
SGLANG_UNEVEN_TOKEN_VECTOR set to 28,26,20 where the ship process carried
14,10,8 -- and the instance it booted came up, answered /model_info with 200
and never answered /generate. Nothing in the tree could turn a capture into
something a boot script could SOURCE, so every consumer wrote the values out
again by hand, and every hand-written copy is a copy that drifts.

Two modes, one file, because they must agree by construction:

  export   render the capture as sourceable shell, values shlex-quoted
  --check  compare a live environment against the capture and name every
           unsanctioned divergence, exit non-zero if there is one

QUOTING IS THE POINT OF THE EXPORT MODE. The replay path this replaces
(/spinning/evidence-631/val-r4/restore_ship.sh:17) does `export "$line"`
unquoted, which works only for as long as no captured value contains a space,
a quote or a `$`. The ship argv already carries {"preserve_thinking": true};
the env is one calibration away from carrying something similar. shlex.quote
removes the whole question.

WHAT IS AND IS NOT POLICED. The stack owns SGLANG_*, HTSGLANG_* and PYTORCH_*
plus PYTHONPATH, LD_LIBRARY_PATH and CUDA_VISIBLE_DEVICES. A gate that also
policed HOME, TERM and LS_COLORS would be unusable from an interactive shell
and would be bypassed within a week, and a bypassed gate is worse than none.

PER-BOOT KEYS. Four keys legitimately differ every boot; they are defined
ONCE, here, and scripts/turnkey_539_parity_proof.py imports this definition
rather than keeping its own -- two lists of sanctioned divergences is the same
defect one directory over.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

#: Env keys whose value is a per-boot identity, each with the reason it is
#: allowed to differ. THE SINGLE DEFINITION: turnkey_539_parity_proof.py
#: imports this as EXPECTED_DIVERGENCE.
PER_BOOT_KEYS: Dict[str, str] = {
    "PYTHONPATH": "capture ran from the wt-631-routea worktree; the unit "
                  "roots the stack in the canonical checkout",
    "CUDA_VISIBLE_DEVICES": "derived from the card UUIDs at boot (same value, "
                            "different provenance)",
    "SGLANG_PHASE_FLIP_INSTANCE": "per-boot identity; the capture embeds the "
                                  "dead pid 3940356",
    "SGLANG_BOOT_COMMIT": "provenance, measured from the repo at boot",
}

#: Prefixes the stack owns outright.
GOVERNED_PREFIXES = ("SGLANG_", "HTSGLANG_", "PYTORCH_")
#: Keys the stack owns that carry no prefix. Same list the parity proof uses.
GOVERNED_EXACT = frozenset({"PYTHONPATH", "LD_LIBRARY_PATH",
                            "CUDA_VISIBLE_DEVICES"})


class CaptureError(Exception):
    """The capture file cannot be read as one KEY=VALUE per line."""


def is_governed(key: str) -> bool:
    return key.startswith(GOVERNED_PREFIXES) or key in GOVERNED_EXACT


def parse_capture(text: str, source: str = "<capture>") -> Dict[str, str]:
    """Parse a capture: one KEY=VALUE per line, blanks and #comments skipped.

    A value may contain '=' (only the first splits), may be empty, and may
    contain anything else a shell value can hold. A line without '=' and a
    duplicate key are REFUSED by line number rather than guessed at: a capture
    is evidence, and evidence that needs interpreting is not evidence.
    """
    out: Dict[str, str] = {}
    seen_at: Dict[str, int] = {}
    for n, line in enumerate(text.split("\n"), 1):
        if line.strip() == "" or line.lstrip().startswith("#"):
            continue
        if "=" not in line:
            raise CaptureError(
                f"{source}:{n}: no '=' in {line.strip()!r}; a capture is one "
                f"KEY=VALUE per line")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise CaptureError(f"{source}:{n}: empty key in {line.strip()!r}")
        if key in out:
            raise CaptureError(
                f"{source}:{n}: duplicate key {key} (first seen at line "
                f"{seen_at[key]}); which one shipped is not decidable here")
        out[key] = value
        seen_at[key] = n
    return out


def load_capture(path: str) -> Dict[str, str]:
    try:
        with open(path, "r") as fh:
            text = fh.read()
    except OSError as e:
        raise CaptureError(f"cannot read capture {path}: {e}") from e
    return parse_capture(text, source=path)


def render_exports(capture: Mapping[str, str], governed_only: bool = False,
                   source: str = "") -> str:
    """Render the capture as shell a boot script can source.

    Per-boot keys are skipped: emitting the capture's SGLANG_BOOT_COMMIT would
    make every boot claim to be a commit it is not.
    """
    lines = [
        "# GENERATED by scripts/turnkey_539_export_env.py -- do not edit.",
        f"# capture: {source or '<stdin>'}",
        "# The per-boot keys (%s) are deliberately absent."
        % ", ".join(sorted(PER_BOOT_KEYS)),
    ]
    for key in sorted(capture):
        if key in PER_BOOT_KEYS:
            continue
        if governed_only and not is_governed(key):
            continue
        lines.append(f"export {key}={shlex.quote(capture[key])}")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class Divergence:
    #: CHANGED, MISSING (in the capture, absent from the environment) or
    #: EXTRA (in the environment, absent from the capture).
    kind: str
    key: str
    captured: str = ""
    actual: str = ""

    def line(self) -> str:
        if self.kind == "MISSING":
            return (f"MISSING  {self.key}: captured {self.captured!r}, "
                    f"the environment does not set it")
        if self.kind == "EXTRA":
            return (f"EXTRA    {self.key}={self.actual!r}: the capture does "
                    f"not carry this key")
        return (f"CHANGED  {self.key}: captured {self.captured!r}, "
                f"environment has {self.actual!r}")


def diff_env(capture: Mapping[str, str], env: Mapping[str, str],
             allow: Iterable[str] = (), all_keys: bool = False,
             ) -> List[Divergence]:
    """Every unsanctioned divergence of ``env`` from ``capture``.

    Scope is the governed key set unless ``all_keys``. ``allow`` is per key and
    never widens beyond the keys named in it.
    """
    allowed = set(allow) | set(PER_BOOT_KEYS)
    keys = set(capture) | set(env)
    out: List[Divergence] = []
    for key in sorted(keys):
        if key in allowed:
            continue
        if not all_keys and not is_governed(key):
            continue
        in_cap, in_env = key in capture, key in env
        if in_cap and in_env:
            if capture[key] != env[key]:
                out.append(Divergence("CHANGED", key, capture[key], env[key]))
        elif in_cap:
            out.append(Divergence("MISSING", key, captured=capture[key]))
        else:
            out.append(Divergence("EXTRA", key, actual=env[key]))
    return out


def _read_env_file(path: str) -> Dict[str, str]:
    return load_capture(path)


def _check(capture: Mapping[str, str], env: Mapping[str, str],
           allow: Sequence[str], all_keys: bool, capture_path: str,
           stream) -> int:
    # Announce every declared override BEFORE the verdict. An override that
    # nobody sees is a silent redefinition wearing a different hat.
    for key in allow:
        cap = capture.get(key, "<not in the capture>")
        got = env.get(key, "<unset>")
        if cap != got:
            print(f"OVERRIDE {key}: capture {cap!r} -> boot {got!r} "
                  f"(declared by the caller)", file=stream)
    divs = diff_env(capture, env, allow=allow, all_keys=all_keys)
    if not divs:
        print(f"env parity OK against {capture_path} "
              f"({len(capture)} captured keys, {len(allow)} declared "
              f"override(s))", file=stream)
        return 0
    print(f"ENV DIVERGES FROM THE CAPTURE {capture_path} -- "
          f"{len(divs)} unsanctioned key(s):", file=stream)
    for d in divs:
        print("  " + d.line(), file=stream)
    print("Each of these must either come from the capture or be declared as "
          "a named override. There is no blanket bypass on purpose: the "
          "2026-08-12 wedge was a boot whose env had drifted in seven keys.",
          file=stream)
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Render a captured ship env as sourceable shell, or "
                    "check a live environment against it.")
    ap.add_argument("capture", help="capture file, one KEY=VALUE per line")
    ap.add_argument("--check", action="store_true",
                    help="compare an environment against the capture and "
                         "exit non-zero on any unsanctioned divergence")
    ap.add_argument("--env-file", default=None,
                    help="with --check: read the environment from this file "
                         "instead of the current process environment")
    ap.add_argument("--allow", action="append", default=[], metavar="KEY",
                    help="with --check: this ONE key may diverge. Repeatable. "
                         "Per key by design; there is no --allow-all.")
    ap.add_argument("--governed-only", action="store_true",
                    help="export only keys the stack owns (SGLANG_*, "
                         "HTSGLANG_*, PYTORCH_*, PYTHONPATH, LD_LIBRARY_PATH, "
                         "CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--all-keys", action="store_true",
                    help="with --check: compare every captured key, not only "
                         "the governed ones")
    ap.add_argument("-o", "--output", default=None,
                    help="write the exports here instead of stdout")
    a = ap.parse_args(argv)

    try:
        capture = load_capture(a.capture)
    except CaptureError as e:
        print(f"REFUSE: {e}", file=sys.stderr)
        return 2

    if a.check:
        try:
            env = (_read_env_file(a.env_file) if a.env_file
                   else dict(os.environ))
        except CaptureError as e:
            print(f"REFUSE: {e}", file=sys.stderr)
            return 2
        return _check(capture, env, a.allow, a.all_keys, a.capture,
                      sys.stderr)

    text = render_exports(capture, governed_only=a.governed_only,
                          source=a.capture)
    if a.output:
        with open(a.output, "w") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
