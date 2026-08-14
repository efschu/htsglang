#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#485 seam decomposition -- read the seam census population and split the
measured seam transient into its named terms.

Input: any boot log carrying '[#631 seam-census]' lines. Both the old format
(free/step/slack) and the new one (with alloc=/res=) parse.
"""
import re
import sys
from collections import defaultdict

HEAD = re.compile(
    r"\[#631 seam-census\]\s+(?P<dir>\w+)\s+rank\s+(?P<rank>\d+):\s+"
    r"transient\s+(?P<tr>-?\d+)\s+MiB\s+\(baseline free\s+(?P<base>-?\d+)\s+MiB,\s+"
    r"trough\s+(?P<trough>-?\d+)\s+MiB at '(?P<stage>[^']+)'\)"
)
MARK = re.compile(
    r"(?P<stage>[a-z_0-9]+)\s+free=(?P<free>-?\d+)\s+step(?P<step>[+-]\d+)"
    r"(?:\s+slack=(?P<slack>-?\d+))?(?:\s+alloc=(?P<alloc>-?\d+))?"
    r"(?:\s+res=(?P<res>-?\d+))?"
)
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


class Flip:
    __slots__ = ("ts", "dir", "rank", "transient", "baseline", "trough",
                 "trough_stage", "marks", "tag")

    def steps_by_stage(self):
        out = defaultdict(float)
        for m in self.marks:
            out[m["stage"]] += m["step"]
        return out

    def trough_index(self):
        """Index of the mark whose free equals the reported trough (last one)."""
        best, bi = None, None
        for i, m in enumerate(self.marks):
            if best is None or m["free"] < best:
                best, bi = m["free"], i
        return bi


def parse(path):
    flips = []
    with open(path, errors="replace") as fh:
        for line in fh:
            if "[#631 seam-census]" not in line:
                continue
            h = HEAD.search(line)
            if not h:
                continue
            f = Flip()
            t = TS.match(line.strip())
            f.ts = t.group(1) if t else "?"
            f.dir = h.group("dir")
            f.rank = int(h.group("rank"))
            f.transient = int(h.group("tr"))
            f.baseline = int(h.group("base"))
            f.trough = int(h.group("trough"))
            f.trough_stage = h.group("stage")
            f.marks = []
            for seg in line.split("|")[1:]:
                m = MARK.search(seg.strip())
                if not m:
                    continue
                f.marks.append({
                    "stage": m.group("stage"),
                    "free": int(m.group("free")),
                    "step": int(m.group("step")),
                    "slack": int(m.group("slack")) if m.group("slack") else None,
                    "alloc": int(m.group("alloc")) if m.group("alloc") else None,
                    "res": int(m.group("res")) if m.group("res") else None,
                })
            flips.append(f)
    return flips


def main():
    flips = []
    for tag_path in sys.argv[1:]:
        tag, _, path = tag_path.partition("=")
        if not path:
            tag, path = path or tag, tag
        fs = parse(path)
        for f in fs:
            f.tag = tag
        flips += fs
        print(f"# {tag or path}: {len(fs)} seam-census lines")
    print(f"# TOTAL {len(flips)} flips\n")

    by = defaultdict(list)
    for f in flips:
        by[(f.dir, f.rank)].append(f)

    for key in sorted(by):
        fs = by[key]
        trs = sorted(f.transient for f in fs)
        stages = defaultdict(int)
        for f in fs:
            stages[f.trough_stage] += 1
        print(f"== {key[0]} rank {key[1]}: n={len(fs)}  "
              f"transient min={trs[0]} p50={trs[len(trs)//2]} max={trs[-1]}  "
              f"baseline modal={max(set(f.baseline for f in fs), key=[g.baseline for g in fs].count)}")
        print(f"   trough stage: {dict(stages)}")
        # per-stage step distribution
        agg = defaultdict(list)
        for f in fs:
            for st, v in f.steps_by_stage().items():
                agg[st].append(v)
        print(f"   {'stage':<28}{'n':>4}{'min':>9}{'p50':>9}{'max':>9}")
        for st in sorted(agg, key=lambda s: sum(agg[s]) / len(agg[s])):
            v = sorted(agg[st])
            print(f"   {st:<28}{len(v):>4}{v[0]:>9.0f}{v[len(v)//2]:>9.0f}{v[-1]:>9.0f}")
        print()


if __name__ == "__main__":
    main()
