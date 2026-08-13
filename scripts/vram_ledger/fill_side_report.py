#!/usr/bin/env python3
# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The fill-side attribution report: why a card is not full, in named posts.

WHAT QUESTION THIS ANSWERS. #602 fixed the break side and left the fill side
open: the three cards of the reference rig finish a boot at 2.0 / 5.7 / 3.7 GiB
free against a 1024 MiB corridor, and nobody could name what holds the
difference. This renders one line per post per card from the flight recorder's
marks, so the answer is a table of named bytes rather than a hypothesis.

THE IDENTITY IT ENFORCES, per card::

    nvml_total = carve_out + sum(resident of each pid) + other_pids + free

and, within one pid, the resident total is decomposed into named posts by
DIFFERENCING CONSECUTIVE MARKS. Every MiB lands in a named post or in an
explicitly named residual line; a report whose lines do not close is printed as
a FAILURE, not silently balanced.

TWO DECISIONS THAT MAKE THIS DIFFER FROM ``attribute_flight.py phases``
----------------------------------------------------------------------

**1. Grouping is by PID, never by the mark's rank field or its file.** Ranks 1
and 2 write their earliest marks -- process_start through the target runner's
kv_pool_sized -- carrying ``rank: 0``, because the rank is not yet assigned when
those marks are taken, and those marks therefore land in rank 0's file. Grouping
by rank consequently mixes three different cards into one table and attributes
another rank's CUDA context to rank 0. The pid is stamped on every mark by the
process that took it and is never wrong, so the pid is the grouping key. The
card uuid is carried along as a cross-check: one pid must see exactly one card.

**2. The decomposition differences NVML-resident bytes, not torch reserved.**
Torch's ``reserved_bytes`` is not a measure of occupancy on this build: after
the draft runner's pool is created, torch reports 7162 / 5514 / 4824 MiB MORE
reserved than NVML says the process physically holds. Reservation that carries
no physical backing is real in torch's books and absent from the card's, so a
post differenced from ``reserved`` would bill the card for bytes it never lost.
The divergence is not discarded -- it is reported on its own line, because a
sizing model that reads ``reserved`` is exactly how the card ends up
underfilled, and that line is the diagnosis.

Reads only. Touches no GPU, imports no torch, writes nothing back into the
ledger. Turning these measurements into ledger terms is a separate step.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

MIB = 1 << 20

#: How a transition between two consecutive marks is named as a post. The key
#: is ``(to_phase, draft_worker)``; the value is the post name. Differencing
#: consecutive marks means the post is what was allocated BETWEEN them, so the
#: post is named after the mark that CLOSES it.
#:
#: ``draft_worker`` is ``None`` on the marks taken outside a runner's init
#: (``process_start``, ``boot_complete``, ``first_forward``), which is why the
#: table keys on it rather than assuming every mark carries it.
POST_NAMES: Dict[Tuple[str, Optional[bool]], str] = {
    ("pre_weight_load", False): "cuda_context_and_comm",
    ("pre_weight_load", True): "inter_runner_gap",
    ("pre_weight_load", None): "cuda_context_and_comm",
    ("weights_loaded", False): "weights_target",
    ("weights_loaded", True): "weights_draft",
    ("kv_pool_sized", False): "kv_pool_target",
    ("kv_pool_sized", True): "kv_pool_draft",
    ("capture_begin", False): "attn_workspace_target",
    ("capture_begin", True): "attn_workspace_draft",
    ("capture_end", False): "graph_capture_target",
    ("capture_end", True): "graph_capture_draft",
    ("boot_complete", None): "boot_tail",
    ("first_forward", None): "first_forward_transient",
}

#: The order posts are printed in, so two reports of different boots line up
#: column by column and can be diffed by eye.
POST_ORDER: Tuple[str, ...] = (
    "cuda_context_and_comm",
    "weights_target",
    "kv_pool_target",
    "inter_runner_gap",
    "weights_draft",
    "kv_pool_draft",
    "attn_workspace_target",
    "attn_workspace_draft",
    "graph_capture_target",
    "graph_capture_draft",
    "boot_tail",
    "first_forward_transient",
    "unnamed_transition",
)


def _draft(mark: Mapping) -> Optional[bool]:
    extra = mark.get("extra") or {}
    value = extra.get("draft_worker")
    return None if value is None else bool(value)


def read_boot(directory: str, boot: Optional[str] = None) -> List[dict]:
    """Every mark of one boot, from every rank file, as one flat list.

    The rank files are read as a set rather than trusted to partition the ranks,
    for the misfiling reason in this module's docstring.
    """
    marks: List[dict] = []
    for name in sorted(os.listdir(directory)):
        if not name.startswith("flight_marks_rank") or not name.endswith(".jsonl"):
            continue
        with open(os.path.join(directory, name)) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                marks.append(record)
    if not marks:
        return []
    if boot is None:
        latest = max(marks, key=lambda m: m.get("wall") or 0.0)
        boot = latest.get("boot_id")
    return [m for m in marks if m.get("boot_id") == boot]


def card_names() -> Dict[str, str]:
    """``uuid -> product name``, straight from nvidia-smi. Best effort."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    names: Dict[str, str] = {}
    for line in out.splitlines():
        if "," not in line:
            continue
        uuid, _, name = line.partition(",")
        names[uuid.strip()] = name.strip()
    return names


class PidTimeline:
    """One process's marks on one card, and the posts differenced from them."""

    def __init__(self, pid: int, marks: Sequence[Mapping]):
        self.pid = pid
        self.marks = sorted(marks, key=lambda m: m.get("monotonic") or 0.0)
        uuids = {m.get("card_uuid") for m in self.marks if m.get("card_uuid")}
        #: A pid that saw two cards would invalidate every per-card sum below,
        #: so it is recorded rather than averaged away.
        self.card_uuid = sorted(uuids)[0] if uuids else "?"
        self.multi_card = len(uuids) > 1
        self.posts: Dict[str, int] = {}
        self.segments: List[dict] = []
        self.unnamed: List[str] = []
        self._difference()

    def _difference(self) -> None:
        """Difference consecutive marks into a SEQUENTIAL, signed segment list.

        A boot does not only allocate: between the two runners this process
        RELEASES 12.6 GiB on the reference rig and builds again. Summing
        equally-named phases into one bucket would net those two motions
        against each other and print a post that never existed. The sequence is
        therefore kept, each segment signed, and the roll-up is a convenience
        derived from it -- the segments are the record.
        """
        previous_resident = 0
        previous_phase = "(process entry)"
        for index, mark in enumerate(self.marks):
            resident = int(mark.get("nvml_self_bytes") or 0)
            delta = resident - previous_resident
            phase = str(mark.get("phase"))
            key = (phase, _draft(mark))
            name = POST_NAMES.get(key)
            if name is None:
                if phase == "process_start":
                    name = "baseline_no_cuda_yet"
                else:
                    name = "unnamed_transition"
                    self.unnamed.append(f"{phase}[draft={_draft(mark)}]")
            self.segments.append(
                {
                    "seq": index,
                    "from": previous_phase,
                    "to": phase,
                    "draft": _draft(mark),
                    "post": name,
                    "delta_bytes": delta,
                    "resident_bytes": resident,
                }
            )
            self.posts[name] = self.posts.get(name, 0) + delta
            previous_resident = resident
            previous_phase = phase

    @property
    def final(self) -> Mapping:
        return self.marks[-1] if self.marks else {}

    @property
    def resident_bytes(self) -> int:
        return int(self.final.get("nvml_self_bytes") or 0)

    @property
    def reserved_bytes(self) -> int:
        return int(self.final.get("reserved_bytes") or 0)

    @property
    def unbacked_bytes(self) -> int:
        """Torch reservation with no physical backing behind it.

        Positive means torch's books claim more of the card than the card says
        this process holds. Reported, never folded into the residency sum.
        """
        return self.reserved_bytes - self.resident_bytes


def report(marks: Sequence[Mapping], corridor_mib: int = 1024) -> Tuple[str, bool]:
    """Render the per-card table. Returns ``(text, ok)``."""
    if not marks:
        return ("No marks for this boot.\n", False)

    by_pid: Dict[int, List[Mapping]] = {}
    for mark in marks:
        by_pid.setdefault(int(mark.get("pid") or 0), []).append(mark)
    timelines = [PidTimeline(pid, group) for pid, group in sorted(by_pid.items())]

    by_card: Dict[str, List[PidTimeline]] = {}
    for timeline in timelines:
        by_card.setdefault(timeline.card_uuid, []).append(timeline)

    names = card_names()
    boot = marks[0].get("boot_id", "?")
    lines: List[str] = []
    ok = True
    lines.append(f"FILL-SIDE ATTRIBUTION -- boot {boot}")
    lines.append(f"corridor target: {corridor_mib} MiB free per card")
    lines.append("")

    for uuid in sorted(by_card):
        group = by_card[uuid]
        final = group[0].final
        total = int(final.get("nvml_total_bytes") or 0)
        free = int(final.get("nvml_free_bytes") or 0)
        carve = int(final.get("nvml_carve_out_bytes") or 0)
        label = names.get(uuid, "unknown card")
        lines.append(f"=== {label}  {uuid}")
        lines.append(f"    NVML total {total // MIB} MiB")
        lines.append("")
        lines.append(f"    {'post':<28} {'MiB':>9}   held by")
        lines.append(f"    {'-' * 28} {'-' * 9}   {'-' * 20}")
        lines.append(
            f"    {'nvml_carve_out':<28} {carve // MIB:>9}   driver (never allocatable)"
        )

        resident_total = 0
        for timeline in group:
            if timeline.multi_card:
                ok = False
                lines.append(
                    f"    !! pid {timeline.pid} saw more than one card; sums below are void"
                )
            resident_total += timeline.resident_bytes
            lines.append(f"    -- pid {timeline.pid}, in boot order (signed):")
            for segment in timeline.segments:
                if segment["seq"] == 0:
                    continue
                delta = segment["delta_bytes"] // MIB
                if delta == 0:
                    continue
                draft = segment["draft"]
                tag = "" if draft is None else ("  [draft]" if draft else "  [target]")
                lines.append(
                    f"    {segment['post']:<28} {delta:>+9}   "
                    f"{segment['from']} -> {segment['to']}{tag}"
                )
            lines.append(
                f"    {'= pid resident total':<28} {timeline.resident_bytes // MIB:>9}   pid {timeline.pid}"
            )
            if timeline.unnamed:
                lines.append(
                    f"      (unnamed transitions: {', '.join(sorted(set(timeline.unnamed)))})"
                )

        #: Anything on the card that is neither the carve-out, nor a pid this
        #: recorder followed, nor free. A foreign process, or a post this
        #: instrument cannot see. Named, bounded, never absorbed.
        used = int(final.get("nvml_used_bytes") or 0)
        foreign = used - carve - resident_total
        lines.append(
            f"    {'foreign_or_unattributed':<28} {foreign // MIB:>9}   not this boot's ranks"
        )
        lines.append(f"    {'free':<28} {free // MIB:>9}   unused")
        lines.append(f"    {'-' * 28} {'-' * 9}")

        closes = carve + resident_total + foreign + free
        drift = total - closes
        lines.append(
            f"    {'SUM':<28} {closes // MIB:>9}   vs NVML total {total // MIB}"
        )
        if abs(drift) > MIB:
            ok = False
            lines.append(f"    !! IDENTITY BROKEN by {drift // MIB} MiB")
        lines.append("")

        overshoot = free // MIB - corridor_mib
        lines.append(
            f"    corridor: {free // MIB} MiB free, target {corridor_mib} -> {overshoot:+d} MiB"
        )
        for timeline in group:
            unbacked = timeline.unbacked_bytes
            lines.append(
                f"    unbacked reservation (torch resv - resident), pid {timeline.pid}: "
                f"{unbacked // MIB:+d} MiB"
            )
        lines.append("")

    return ("\n".join(lines) + "\n", ok)


def boot_ids(directory: str) -> List[Tuple[str, float]]:
    """``[(boot_id, first wall), ...]`` oldest first, over every rank file."""
    first: Dict[str, float] = {}
    for name in sorted(os.listdir(directory)):
        if not name.startswith("flight_marks_rank") or not name.endswith(".jsonl"):
            continue
        with open(os.path.join(directory, name)) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                boot = record.get("boot_id")
                wall = record.get("wall") or 0.0
                if boot and (boot not in first or wall < first[boot]):
                    first[boot] = wall
    return sorted(first.items(), key=lambda pair: pair[1])


def variance(directory: str, count: int) -> str:
    """Per-card, per-post spread over the last ``count`` boots.

    WHY THIS IS THE POINT AND NOT A NICETY. A term calibrated from ONE boot is
    a guess with a decimal point: the JIT/autotune workspaces alone were
    observed to move ~690 MiB between boots of the same config. A term is only
    honest if it carries the spread it was measured with, so the calibration
    source is a DISTRIBUTION over boots of one fingerprint, and a post whose
    range is wide is visible as wide rather than averaged into false precision.
    """
    lines: List[str] = []
    boots = boot_ids(directory)[-count:]
    lines.append(f"POST SPREAD over {len(boots)} boots (oldest {boots[0][0]})")
    lines.append("")

    # card uuid -> post name -> [MiB per boot]
    series: Dict[str, Dict[str, List[int]]] = {}
    free_series: Dict[str, List[int]] = {}
    for boot, _ in boots:
        marks = read_boot(directory, boot)
        if not marks:
            continue
        by_pid: Dict[int, List[Mapping]] = {}
        for mark in marks:
            by_pid.setdefault(int(mark.get("pid") or 0), []).append(mark)
        for pid, group in by_pid.items():
            timeline = PidTimeline(pid, group)
            card = series.setdefault(timeline.card_uuid, {})
            for post, value in timeline.posts.items():
                card.setdefault(post, []).append(value // MIB)
            card.setdefault("= resident total", []).append(
                timeline.resident_bytes // MIB
            )
            card.setdefault("unbacked_reservation", []).append(
                timeline.unbacked_bytes // MIB
            )
            free_series.setdefault(timeline.card_uuid, []).append(
                int(timeline.final.get("nvml_free_bytes") or 0) // MIB
            )

    names = card_names()
    for uuid in sorted(series):
        lines.append(f"=== {names.get(uuid, 'unknown card')}  {uuid}")
        lines.append(
            f"    {'post':<28} {'n':>3} {'min':>8} {'max':>8} {'spread':>8} {'mean':>8}"
        )
        for post in sorted(series[uuid]):
            values = series[uuid][post]
            if not values:
                continue
            low, high = min(values), max(values)
            mean = sum(values) // len(values)
            lines.append(
                f"    {post:<28} {len(values):>3} {low:>8} {high:>8} "
                f"{high - low:>8} {mean:>8}"
            )
        free = free_series.get(uuid, [])
        if free:
            lines.append(
                f"    {'free at last mark':<28} {len(free):>3} {min(free):>8} "
                f"{max(free):>8} {max(free) - min(free):>8} {sum(free) // len(free):>8}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory")
    parser.add_argument("--boot", default=None, help="which boot (default: latest)")
    parser.add_argument("--corridor-mib", type=int, default=1024)
    parser.add_argument(
        "--across-boots",
        type=int,
        default=0,
        help="instead of one boot, print the per-post spread over the last N boots",
    )
    args = parser.parse_args(argv)

    if args.across_boots:
        sys.stdout.write(variance(args.directory, args.across_boots))
        return 0

    marks = read_boot(args.directory, args.boot)
    text, ok = report(marks, corridor_mib=args.corridor_mib)
    sys.stdout.write(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
