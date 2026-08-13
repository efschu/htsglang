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
"""Calibration from the boot MARK HISTORY, with no GPU and no probe.

WHAT THIS REPLACES. Two ledger terms shipped as constants inherited from a rig
window that no longer exists: the hardware residual (664 MiB on the 5090, 312
on each 3080, calibrated under fingerprint ``a191a0712717`` /
``window-2026-08-06``) and the load transient (a flat
``LOAD_TRANSIENT_REFERENCE_MIB = 70``). The first reconciliation of the
modelled ledger against a measured boot found the residual 25-34% LOW and the
transient falsified outright. Both are re-derived here from the flight
recorder's own accumulated marks, which is a measurement this rig has already
paid for: 486 boots were on disk before a single line of this module existed.

THE RULE: A WIDE POST IS REFUSED, NEVER AVERAGED. The residual repeats inside
a 100-136 MiB band across hundreds of boots, so a band is a fair summary of it.
The load transient does not: it ranges 0-18486 MiB on the same card over the
same history, because it tracks whatever the configuration asked that boot to
load. Averaging that produces ~9000 MiB -- a figure no boot ever exhibited,
carrying no warning that it is the midpoint of a bimodal spread. So a post
whose band is wide relative to its own high water mark yields a REFUSAL that
names the distribution, and the ledger reports the term UNBOUNDED rather than
priced. That is the #605 doctrine restated: a falsified formula that still
returns a number is worse than a refusal, because the number looks like an
answer.

WHY THE HIGH AND NOT THE MEDIAN. Where a band IS calibrated, the charge is its
HIGH. The ledger's job is to fund the worst boot the configuration can produce,
and every one of these posts has been observed at its high on this rig. A
median would under-charge the boots at the top of the band by construction, and
under-charge is the direction that OOMs.

WHY BOOTS AND NOT MARKS. The unit of observation is one (boot, process, card),
never one mark: a boot writes 42-54 marks and the rank0 file carries every
pid's marks as well as its own (a recorder defect documented in R1), so
counting marks would weight a long boot more heavily than a short one and
counting files would count most boots twice. :func:`read_marks` dedupes on
``(boot_id, pid, phase, monotonic)`` and every band reports ``n_boots``.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MIB = 1 << 20

__all__ = [
    "POST_HARDWARE_RESIDUAL",
    "POST_LOAD_TRANSIENT",
    "WIDE_SPREAD_FRACTION",
    "MIN_BOOTS_FOR_A_BAND",
    "HistoryBand",
    "BootHistory",
    "read_marks",
    "bands_from_marks",
    "load_boot_history",
]

#: The per-process driver-side overhead NVML charges that torch does not
#: account for: measured as ``non_torch_bytes`` at the target runner's
#: ``weights_loaded``, the one mark where the allocator is full of weights and
#: the KV arena has not yet opened, so nothing else is in the number.
POST_HARDWARE_RESIDUAL = "hardware_residual"

#: The allocator's reserved PEAK above what is still reserved, taken as the
#: MAXIMUM over the whole boot rather than at one phase. See
#: :data:`sglang.srt.mem_ledger.reconcile.TERM_TO_POST` for why reading it at a
#: single mark measured 0 on a boot whose true peak was 13392 MiB.
POST_LOAD_TRANSIENT = "load_transient"

#: A band is refused when ``(high - low) > WIDE_SPREAD_FRACTION * high``.
#:
#: Stated as a FRACTION OF THE HIGH rather than an absolute MiB figure so the
#: rule reads the same on a 480 MiB post and a 13000 MiB one. At 0.5 the rig's
#: measured history separates cleanly and not marginally: the residual bands
#: sit at 0.111 (5090) and 0.274 (3080), the load-transient bands at 1.0 on
#: every card. Nothing observed on this rig lands near the threshold, so the
#: verdicts do not hang on the exact value -- which is the property that makes
#: a chosen constant defensible rather than arbitrary.
WIDE_SPREAD_FRACTION = 0.5

#: One boot is an observation, not a distribution. A band drawn from a single
#: boot has spread 0 and would pass every width test by construction, which is
#: the tidiest possible way to launder one sample into a calibration.
MIN_BOOTS_FOR_A_BAND = 2

#: The phase whose ``non_torch_bytes`` is the residual. Deliberately the TARGET
#: runner's (``draft_worker`` false): by the draft runner's marks the KV arena
#: is open and ``non_torch_bytes`` clamps to 0, which would drag the band's low
#: to zero and refuse a post that is in fact stable.
_RESIDUAL_PHASE = "weights_loaded"


@dataclasses.dataclass(frozen=True)
class HistoryBand:
    """What the mark history says about one post on one card.

    ``charge_mib`` is ``None`` exactly when ``refused`` is true. The two are
    kept as separate fields rather than one optional number because a caller
    that forgets to check gets a ``None`` it cannot add to a total, instead of
    a zero it can.
    """

    uuid: str
    post: str
    charge_mib: Optional[int]
    low_mib: int
    high_mib: int
    n_boots: int
    refused: bool
    reason: str

    @property
    def spread_mib(self) -> int:
        return self.high_mib - self.low_mib

    def describe(self) -> str:
        if self.refused:
            return f"{self.post} on {self.uuid[:20]}: REFUSED -- {self.reason}"
        return (
            f"{self.post} on {self.uuid[:20]}: {self.charge_mib} MiB "
            f"(band {self.low_mib}-{self.high_mib} MiB over {self.n_boots} boots)"
        )


@dataclasses.dataclass(frozen=True)
class BootHistory:
    """Every band this rig's recorded boots support."""

    bands: Mapping[Tuple[str, str], HistoryBand]
    n_boots: int
    source: str = ""

    def band(self, uuid: str, post: str) -> Optional[HistoryBand]:
        """The band for one card and one post, or None when never observed.

        None means "this history has nothing to say", which is a different
        statement from a refusal (the history HAS something to say and it is
        that the post is not a constant). Callers must keep them apart.
        """
        return self.bands.get((str(uuid), str(post)))

    def describe(self) -> str:
        lines = [f"boot-history calibration over {self.n_boots} boots ({self.source})"]
        for key in sorted(self.bands):
            lines.append("  " + self.bands[key].describe())
        return "\n".join(lines)


def read_marks(directory: str) -> List[Dict[str, Any]]:
    """Every mark the recorder wrote into *directory*, deduplicated.

    Corrupt lines are skipped rather than fatal: this is an instrument reading
    an append-only log that a killed boot can truncate mid-line, and a
    calibration that refuses to load because one boot died is a calibration
    that is missing exactly when someone is investigating why it died.
    """
    seen = set()
    out: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(directory, "flight_marks_rank*.jsonl"))):
        try:
            handle = open(path)
        except OSError as e:  # pragma: no cover - filesystem differences
            logger.debug("boot history could not open %s: %s", path, e)
            continue
        with handle as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(m, dict):
                    continue
                key = (
                    m.get("boot_id"),
                    m.get("pid"),
                    m.get("phase"),
                    m.get("monotonic"),
                )
                if key in seen:
                    continue
                seen.add(key)
                out.append(m)
    return out


def _band(uuid: str, post: str, values: Sequence[int], *, what: str) -> HistoryBand:
    """Turn one card's observations of one post into a band or a refusal."""
    ordered = sorted(int(v) for v in values)
    n = len(ordered)
    low, high = ordered[0], ordered[-1]
    if n < MIN_BOOTS_FOR_A_BAND:
        return HistoryBand(
            uuid=uuid,
            post=post,
            charge_mib=None,
            low_mib=low,
            high_mib=high,
            n_boots=n,
            refused=True,
            reason=(
                f"{what} was observed on {n} boot of this card; "
                f"{MIN_BOOTS_FOR_A_BAND} are required before a single "
                "observation is called a band"
            ),
        )
    spread = high - low
    if high > 0 and spread > WIDE_SPREAD_FRACTION * high:
        return HistoryBand(
            uuid=uuid,
            post=post,
            charge_mib=None,
            low_mib=low,
            high_mib=high,
            n_boots=n,
            refused=True,
            reason=(
                f"{what} spans {low}-{high} MiB over {n} boots (spread "
                f"{spread} MiB, {spread / high:.0%} of the high, above the "
                f"{WIDE_SPREAD_FRACTION:.0%} refusal rule). This post is not a "
                "constant on this rig and no single figure summarises it; "
                "averaging it would publish a value no boot exhibited"
            ),
        )
    return HistoryBand(
        uuid=uuid,
        post=post,
        charge_mib=high,
        low_mib=low,
        high_mib=high,
        n_boots=n,
        refused=False,
        reason=(
            f"{what} over {n} boots of this card, band {low}-{high} MiB "
            f"(spread {spread} MiB); charged at the band's HIGH so the term "
            "funds the worst boot this rig has produced"
        ),
    )


def bands_from_marks(marks: Iterable[Mapping[str, Any]]) -> BootHistory:
    """Derive every band from a flat sequence of marks.

    The seam: production reads the marks off disk, tests hand them in. Both
    reach the identical arithmetic, so what a hermetic test proves about the
    refusal rule is true of the rig.
    """
    # (boot, pid, uuid) -> that process's marks. The process is the unit
    # because two ranks co-located on one card are two processes and each
    # holds its own residual.
    by_process: Dict[Tuple[Any, Any, str], List[Mapping[str, Any]]] = {}
    seen_marks = set()
    for m in marks:
        uuid = str(m.get("card_uuid") or "")
        if not uuid:
            continue
        dedupe = (m.get("boot_id"), m.get("pid"), m.get("phase"), m.get("monotonic"))
        if dedupe in seen_marks:
            continue
        seen_marks.add(dedupe)
        by_process.setdefault((m.get("boot_id"), m.get("pid"), uuid), []).append(m)

    residual: Dict[str, List[int]] = {}
    transient: Dict[str, List[int]] = {}
    boots = set()
    for (boot, _pid, uuid), process_marks in by_process.items():
        target_loaded = [
            m
            for m in process_marks
            if m.get("phase") == _RESIDUAL_PHASE
            and (m.get("extra") or {}).get("draft_worker") is False
        ]
        if not target_loaded:
            # No target weight load in this process's record: a boot that died
            # before loading, or a process that is not a model runner. It
            # contributes nothing rather than contributing a zero.
            continue
        boots.add(boot)
        residual.setdefault(uuid, []).append(
            int(target_loaded[0].get("non_torch_bytes", 0)) // MIB
        )
        transient.setdefault(uuid, []).append(
            max(int(m.get("allocator_transient_bytes", 0) or 0) for m in process_marks)
            // MIB
        )

    bands: Dict[Tuple[str, str], HistoryBand] = {}
    for uuid, values in residual.items():
        bands[(uuid, POST_HARDWARE_RESIDUAL)] = _band(
            uuid,
            POST_HARDWARE_RESIDUAL,
            values,
            what="the per-process driver residual (non_torch_bytes at the "
            "target runner's weights_loaded)",
        )
    for uuid, values in transient.items():
        bands[(uuid, POST_LOAD_TRANSIENT)] = _band(
            uuid,
            POST_LOAD_TRANSIENT,
            values,
            what="the allocator peak above the resident set (max "
            "allocator_transient_bytes over the boot)",
        )
    return BootHistory(bands=bands, n_boots=len(boots), source="")


def load_boot_history(directory: Optional[str] = None) -> Optional[BootHistory]:
    """Read the recorder's directory and derive the bands, or None.

    ``None`` when the recorder is not armed, the directory is absent, or no
    boot in it carries a target weight load -- three distinct ways of having
    nothing to say, and all three leave every caller on its previous
    behaviour rather than on a zero.
    """
    from sglang.srt.mem_ledger.flight_recorder import DIR_ENV

    directory = directory or os.environ.get(DIR_ENV) or ""
    if not directory or not os.path.isdir(directory):
        return None
    try:
        marks = read_marks(directory)
    except Exception as e:  # pragma: no cover - never fail a boot over a read
        logger.debug("boot history unavailable (%s)", e)
        return None
    if not marks:
        return None
    history = bands_from_marks(marks)
    if not history.bands:
        return None
    return dataclasses.replace(history, source=directory)
