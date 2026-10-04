#!/usr/bin/env python3
"""Derive the sm_89 (Ada) twin lines of a delta_prebuild kernel list (Auftrag 1301).

SM89-DURCHSPIEL-1002 / HW-P0 P1b: every JIT artefact the image pre-builds for the
reference rig is keyed by architecture (tvm-ffi: ``__cuda_arch_8.6``, FlashInfer:
the ``86`` directory, barlink: the group union in the extension name), so the rig
seed never hits on an Ada card and the first Ada boot compiles all of it
(FlashInfer ~15-20 min, tvm-ffi Marlin/W4A8 minutes). ``delta_prebuild.py``
already knows how to build any of these for another architecture WITHOUT a GPU:

  tvmffi <major.minor> <module>:<callable> [json] [ENV=V ...]   (override_jit_cuda_arch)
  fi <dirkey> <uri> [import|server]                             (prebuild_jit.FI_DIRS)
  barlink <archs> <module>:<callable>                           (arch union)
  module sglang.jit_kernel.prebuild_nvfp4_w4a8 --arch 8.6 ...

What was missing is the LIST. This tool reads the existing lists and writes the
sm_89 twins -- never the originals, so the output is an ADDITIONAL list for an
sm_89 image (``make_flat_ctx.sh`` unions lists), and a build with the existing
lists alone is byte-identical to today.

Rules (an sm_89 kernel selection equals the sm_86 one: same Triton/Marlin/FlashInfer
module choice by dtype and shape; only the cubin target differs):

  tvmffi 8.6 X          -> tvmffi 8.9 X
  fi 86 URI [ctx]       -> fi 89 URI [ctx]
  module ...prebuild_nvfp4_w4a8 ... --arch 8.6 ...
                        -> the same with --arch 8.9 (a --report path gets a ``-sm89`` suffix)
  barlink A MOD         -> barlink (A + 8.9) MOD   (mixed rig carrying an Ada card)
                           barlink 8.9 MOD         (pure Ada rig)
  everything else (12.0 lines, py/other module lines): not copied.

Usage: derive_sm89_kernel_list.py LIST [LIST ...] > delta_kernels_sm89.txt
"""

from __future__ import annotations

import re
import sys
from typing import Iterable, List


def _arch_key(a: str):
    major, minor = a.split(".")
    return (int(major), int(minor))


def _barlink_union(archs: str, extra: str = "8.9") -> str:
    s = {a.strip() for a in archs.split(",") if a.strip()}
    s.add(extra)
    return ",".join(sorted(s, key=_arch_key))


def derive_line(line: str) -> List[str]:
    """The sm_89 twin line(s) of ONE list line ([] if it has none)."""
    raw = line.rstrip("\n")
    st = raw.strip()
    if not st or st.startswith("#"):
        return []
    toks = st.split()
    kind = toks[0]
    if kind == "tvmffi" and len(toks) >= 3 and toks[1] == "8.6":
        return [re.sub(r"^(\s*tvmffi\s+)8\.6(\s)", r"\g<1>8.9\2", raw, count=1)]
    if kind == "fi" and len(toks) >= 3 and toks[1] == "86":
        return [re.sub(r"^(\s*fi\s+)86(\s)", r"\g<1>89\2", raw, count=1)]
    if kind == "module" and len(toks) >= 2 and toks[1].endswith(".prebuild_nvfp4_w4a8"):
        if "--arch" in toks and toks[toks.index("--arch") + 1] == "8.6":
            out = re.sub(r"(--arch\s+)8\.6\b", r"\g<1>8.9", raw, count=1)
            return [re.sub(r"(--report\s+\S+?)(\.json)?(\s|$)",
                           lambda m: f"{m.group(1)}-sm89{m.group(2) or ''}{m.group(3)}", out, count=1)]
        return []
    if kind == "barlink" and len(toks) >= 3:
        if "8.9" in toks[1].split(","):   # already an Ada-carrying union: no further twin
            return []
        mixed = _barlink_union(toks[1])
        pure = "8.9"
        rest = " ".join(toks[2:])
        outs = []
        if mixed != toks[1]:
            outs.append(f"barlink {mixed} {rest}")
        outs.append(f"barlink {pure} {rest}")
        return outs
    return []


def derive(lines: Iterable[str]) -> List[str]:
    """Twin lines of a whole list, original order, duplicates dropped."""
    seen, out = set(), []
    for line in lines:
        for twin in derive_line(line):
            if twin not in seen:
                seen.add(twin)
                out.append(twin)
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    lines: List[str] = []
    for path in argv:
        with open(path) as fh:
            lines.extend(fh.read().splitlines())
    twins = derive(lines)
    print("# sm_89 twins (derive_sm89_kernel_list.py, Auftrag 1301) of: " + " ".join(argv))
    print("# ADDITIONAL list for an sm_89 image -- the originals stay in their own lists; no GPU needed (delta_prebuild).")
    for t in twins:
        print(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
