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
"""Boot history as a calibration source: measured posts, with their spread.

WHY A DISTRIBUTION AND NOT A NUMBER. A term calibrated from one boot is a guess
with a decimal point. Measured over the 14 ship boots of 2026-08-13, the posts
of one unchanged config fall into two populations that must not be treated
alike:

* ``cuda_context_and_comm`` -- spread **exactly 0** across all 14 boots (888
  MiB on the 5090, 482 on each 3080, every time). A modeled estimate for this
  is a guess standing where a constant was available.
* ``kv_pool_target`` -- spread **1364-2408 MiB** over the same boots. A median
  of that is a number with no predictive content, and substituting it for a
  modeled term would launder variance into false precision.

So this module never returns a bare number. It returns
:class:`MeasuredPost`, which carries ``n``, ``spread_bytes`` and a ``stable``
verdict, and :func:`residual_overrides` hands the ledger ONLY the posts whose
spread is inside a stated tolerance. A wide post is reported as wide and
declined -- that refusal is the feature, because the alternative is exactly the
guessing game this instrument exists to end.

MEASURED OVER ESTIMATED, BUT ONLY WHERE MEASURED MEANS SOMETHING. Precedence is
not "prefer any measurement". It is: prefer a measurement that is stable across
at least :data:`MIN_BOOTS` boots of the same fingerprint; otherwise keep the
modeled term and say why.

OFF BY DEFAULT. Nothing here runs unless :data:`MEASURED_ENV` is set. The
default boot path does not import this module's effects, does not read the mark
files and is byte-identical to a tree without it.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Dict, List, Mapping, Optional, Sequence

from sglang.srt.mem_ledger.flight_recorder import DIR_ENV, read_marks

logger = logging.getLogger(__name__)

MIB = 1 << 20

#: Opt-in switch. Absent, every entry point here returns "no data" and the
#: ledger keeps its modeled terms unchanged.
MEASURED_ENV = "SGLANG_VRAM_LEDGER_MEASURED"

#: Fewest boots that can support a calibration. Two boots agreeing is a
#: coincidence with a sample size; the ship history carries dozens.
MIN_BOOTS = 5

#: How far a post may range across boots and still be called stable. The CUDA
#: context measured 0 spread over 14 boots, so this is not a tolerance the
#: interesting term needs -- it is slack for allocator granularity on posts
#: that are otherwise constant.
DEFAULT_TOLERANCE_BYTES = 8 * MIB

#: Post name -> the :class:`CardResidual` field it calibrates. Only posts whose
#: measurement means the SAME THING as the field appear here; a post that
#: merely correlates with a field is not a calibration for it.
POST_TO_RESIDUAL: Dict[str, str] = {
    # Both are "what a process pays on this card before it loads anything":
    # the probe measures free-before minus free-after context creation, and
    # this post is the resident delta from process_start to pre_weight_load.
    "cuda_context_and_comm": "cuda_context_bytes",
}


@dataclasses.dataclass(frozen=True)
class MeasuredPost:
    """One post on one card, over several boots, with its spread kept."""

    card_uuid: str
    post: str
    values_bytes: Sequence[int]
    tolerance_bytes: int = DEFAULT_TOLERANCE_BYTES

    @property
    def n(self) -> int:
        return len(self.values_bytes)

    @property
    def min_bytes(self) -> int:
        return min(self.values_bytes)

    @property
    def max_bytes(self) -> int:
        return max(self.values_bytes)

    @property
    def spread_bytes(self) -> int:
        return self.max_bytes - self.min_bytes

    @property
    def median_bytes(self) -> int:
        ordered = sorted(self.values_bytes)
        return ordered[len(ordered) // 2]

    @property
    def stable(self) -> bool:
        """Enough boots, and a spread small enough to stand for a constant."""
        return self.n >= MIN_BOOTS and self.spread_bytes <= self.tolerance_bytes

    def why_not(self) -> str:
        if self.n < MIN_BOOTS:
            return f"only {self.n} boot(s), need {MIN_BOOTS}"
        if self.spread_bytes > self.tolerance_bytes:
            return (
                f"spread {self.spread_bytes // MIB} MiB over {self.n} boots "
                f"exceeds the {self.tolerance_bytes // MIB} MiB tolerance"
            )
        return ""

    def row(self) -> str:
        verdict = "stable" if self.stable else f"DECLINED ({self.why_not()})"
        return (
            f"{self.post:<28} n={self.n:<3} "
            f"min={self.min_bytes // MIB:>6} max={self.max_bytes // MIB:>6} "
            f"spread={self.spread_bytes // MIB:>5} MiB  {verdict}"
        )


def enabled() -> bool:
    return bool(os.environ.get(MEASURED_ENV))


def _boot_ids(directory: str) -> List[str]:
    """Boot ids present under ``directory``, oldest first."""
    first: Dict[str, float] = {}
    for name in sorted(os.listdir(directory)):
        if not (name.startswith("flight_marks_rank") and name.endswith(".jsonl")):
            continue
        import json

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
    return [b for b, _ in sorted(first.items(), key=lambda pair: pair[1])]


def _context_bytes_of_boot(
    marks_by_pid: Mapping[int, Sequence[Mapping]],
) -> Dict[str, int]:
    """``card uuid -> resident bytes at pre_weight_load`` for one boot.

    The CUDA context post is the resident delta from ``process_start`` (which
    is taken before any CUDA allocation, so it reads 0) to the FIRST
    ``pre_weight_load``. Later ``pre_weight_load`` marks belong to the second
    runner and sit on top of a loaded model, so only the first one is the
    context.
    """
    out: Dict[str, int] = {}
    for group in marks_by_pid.values():
        ordered = sorted(group, key=lambda m: m.get("monotonic") or 0.0)
        start = next((m for m in ordered if m.get("phase") == "process_start"), None)
        first_load = next(
            (m for m in ordered if m.get("phase") == "pre_weight_load"), None
        )
        if start is None or first_load is None:
            continue
        uuid = first_load.get("card_uuid") or start.get("card_uuid")
        if not uuid:
            continue
        delta = int(first_load.get("nvml_self_bytes") or 0) - int(
            start.get("nvml_self_bytes") or 0
        )
        if delta > 0:
            out[uuid] = delta
    return out


def measured_posts(
    directory: Optional[str] = None,
    *,
    boots: int = 20,
    tolerance_bytes: int = DEFAULT_TOLERANCE_BYTES,
) -> Dict[str, Dict[str, MeasuredPost]]:
    """``{card uuid: {post: MeasuredPost}}`` over the last ``boots`` boots.

    Returns an empty mapping when the recorder wrote nothing -- an absent
    history is not an error, it is a rig that has not measured yet.
    """
    directory = directory or os.environ.get(DIR_ENV)
    if not directory or not os.path.isdir(directory):
        return {}
    series: Dict[str, Dict[str, List[int]]] = {}
    for boot in _boot_ids(directory)[-boots:]:
        by_pid = read_marks(directory, boot=boot)
        if not by_pid:
            continue
        for uuid, value in _context_bytes_of_boot(by_pid).items():
            series.setdefault(uuid, {}).setdefault("cuda_context_and_comm", []).append(
                value
            )
    return {
        uuid: {
            post: MeasuredPost(
                card_uuid=uuid,
                post=post,
                values_bytes=tuple(values),
                tolerance_bytes=tolerance_bytes,
            )
            for post, values in posts.items()
        }
        for uuid, posts in series.items()
    }


def residual_overrides(
    directory: Optional[str] = None,
    *,
    boots: int = 20,
    tolerance_bytes: int = DEFAULT_TOLERANCE_BYTES,
    force: bool = False,
) -> Dict[str, Dict[str, int]]:
    """``{card uuid: {CardResidual field: bytes}}`` for the STABLE posts only.

    The precedence rule of this module lives here: a post that is not stable is
    logged and left out, so the ledger keeps its modeled term. ``force``
    bypasses only the env gate, never the stability rule -- there is no way to
    make this function return a number it does not stand behind.
    """
    if not (force or enabled()):
        return {}
    overrides: Dict[str, Dict[str, int]] = {}
    for uuid, posts in measured_posts(
        directory, boots=boots, tolerance_bytes=tolerance_bytes
    ).items():
        for post, measurement in posts.items():
            field = POST_TO_RESIDUAL.get(post)
            if field is None:
                continue
            if not measurement.stable:
                logger.info(
                    "mem_ledger: declining measured %s for %s -- %s",
                    post,
                    uuid,
                    measurement.why_not(),
                )
                continue
            overrides.setdefault(uuid, {})[field] = measurement.median_bytes
    return overrides


def describe(directory: Optional[str] = None, *, boots: int = 20) -> str:
    """Human-readable table of what the boot history supports, and what it does not."""
    posts = measured_posts(directory, boots=boots)
    if not posts:
        return "No flight-recorder history under this rig; nothing measured.\n"
    lines = ["MEASURED CALIBRATION SOURCE (flight-recorder boot history)", ""]
    for uuid in sorted(posts):
        lines.append(f"=== {uuid}")
        for post in sorted(posts[uuid]):
            lines.append(f"    {posts[uuid][post].row()}")
        lines.append("")
    return "\n".join(lines) + "\n"
