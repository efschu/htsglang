# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""THE BOOT ARM for the corridor instrument: what the next boot must show.

FIX 2, finding 4.  The instrument correction shipped with a forward reference
to an acceptance that did not exist -- record ``[SECTION 1x]`` ends "which is
why the corrected in-tree line must now reproduce the sampler -- see the boot
arm", and there was no boot arm, no script and no acceptance list anywhere in
the weg2 record.  A corrected instrument with no executable acceptance is a
claim, and the whole point of the correction is that claims about instruments
are checked.

THE TWO CHECKS, and why the second one is the one that matters:

* :func:`arm_report` reads a boot's own front log.  It answers "did this boot
  sample at all, and in which unit" -- the ``instrument=`` token, the per-phase
  per-card minima, and their verdicts against the band IN FORCE.  It is cheap
  and it is not sufficient: a sampler that printed the right LABEL over the
  wrong NUMBER would pass it.  (That is precisely the defect class this
  module's commit exists to close, so the acceptance may not rest on it.)

  AND IT GRADES IN ONE UNIT (fix 3).  The band is allocatable free; a pre-fix
  log's minima are allocatable free PLUS the driver carve-out.  The report
  therefore converts before it grades -- through
  :func:`ring_table.corridor_allocatable`, the same converter the ring credit
  uses, not a second table here -- and prints the source figure, the
  correction and the converted figure on one line.  Where the conversion is
  impossible the card is UNGRADED with the reason, and under
  ``require_in_band`` that refusal is a failure: the predecessor of this
  paragraph graded rg6's ``859`` against the 819-1229 band and blessed the
  5090 that was 478 MiB below the floor.

* :func:`pair_verdict` is the proof.  It pairs the in-tree sampler's own
  per-card ``free`` against an INDEPENDENT reader -- ``nvidia-smi
  --query-gpu=memory.free`` -- taken at the same instant, and requires
  agreement.  This is the exact form that exposed the defect: on boot weg2rg6,
  07:31:30Z, the in-tree line said 1454 / 859 / 1276 while ``memory.free`` said
  1030 / 341 / 852, and the three deltas were 424 / 518 / 424 -- each card's
  driver carve-out, to the MiB.  A pass here is the corrected line reproducing
  the sampler that was right; a fail whose deltas are the carve-outs is the old
  subtraction back, named as such rather than left to be re-derived by hand.

Independent means a different PROCESS reading a different API, not a second
call into this tree: the point of the pairing is that a defect in the tree's
one reader cannot appear on both sides of it.

WHICH IS WHY THIS MODULE NEVER RUNS ``nvidia-smi``.  It takes the independent
reader's output as DATA (:func:`parse_smi_free`), and the invocation lives in
``scripts/weg2/corridor_arm_check.py``, outside ``srt/weg2/`` -- caught by this
package's own one-reader sweep on the first run, which is the sweep working:
a card-memory query added here would be exactly the second in-tree reader the
predecessor commit deleted, however good its intentions.  Data in, verdict
out; the shell-out is the caller's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from sglang.srt.managers import corridor_guard
from sglang.srt.weg2 import front, ring_table

#: The DELTAS a pre-fix pairing shows, MiB.  Used ONLY to recognise the pre-fix
#: failure shape in :func:`pair_verdict` and name it -- never to correct a
#: number and never as an input to a verdict.  (The correction table is
#: :data:`ring_table.RECORD_CARVE_OUT_MIB`, keyed by card UUID, and it is the
#: only one; this tuple is a different quantity in a different role.)
#:
#: WHY 424 SITS BESIDE 425.  These are DELTAS BETWEEN TWO TOOLS, not the driver
#: constant: pynvml and nvidia-smi floor different byte values, so the 3080's
#: 425 MiB carve-out showed up as a 424 MiB delta on the rg6 instant
#: (1454-1030) and as 425 idle.  The tuple therefore carries both, and
#: ``test_the_two_carve_out_tables_cannot_drift_apart`` pins every recorded
#: constant to within that 1 MiB of an entry here so the two cannot diverge in
#: silence.
KNOWN_CARVE_OUT_MIB = (424, 425, 518)

#: ``0, 1030`` from ``nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits``.
_SMI_ROW_RE = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*$")


def parse_smi_free(text: str) -> Dict[int, int]:
    """``{nvml index: free MiB}`` from the independent reader's CSV output.

    ``nvidia-smi``'s ``--format=csv,noheader,nounits`` only; a row this cannot
    parse is DROPPED, and the caller compares card sets, so a silently short
    read becomes a named "card missing from the independent reader" rather than
    a pairing over the subset that happened to parse.
    """
    out: Dict[int, int] = {}
    for line in text.splitlines():
        m = _SMI_ROW_RE.match(line)
        if m:
            out[int(m.group(1))] = int(m.group(2))
    return out


@dataclass
class PairResult:
    """One instant, two readers, per card."""

    #: ``{nvml index: (in-tree MiB, independent MiB, in-tree - independent)}``
    rows: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)
    tolerance_mib: int = 1
    #: Cards one reader saw and the other did not.
    only_in_tree: Tuple[int, ...] = ()
    only_independent: Tuple[int, ...] = ()
    #: Cards whose two readings differ by more than :attr:`tolerance_mib`.
    disagree: Tuple[int, ...] = ()
    #: True when every disagreement is (within tolerance) a known carve-out,
    #: i.e. the pre-fix ``total - used`` subtraction is back.  Diagnosis only;
    #: it never turns a fail into a pass.
    looks_like_the_carve_out_defect: bool = False

    @property
    def ok(self) -> bool:
        return not (self.disagree or self.only_in_tree or self.only_independent or not self.rows)

    def report(self) -> str:
        head = (
            f"WEG2-CORRIDOR-ARM pair in_tree=front._nvml_free "
            f"independent=nvidia-smi:memory.free tol={self.tolerance_mib}MiB "
            f"cards={len(self.rows)} verdict={'PASS' if self.ok else 'FAIL'}"
        )
        body = [
            f"  nvml{i}: in_tree={a}MiB independent={b}MiB delta={d}MiB "
            f"{'OK' if abs(d) <= self.tolerance_mib else 'DISAGREE'}"
            for i, (a, b, d) in sorted(self.rows.items())
        ]
        if self.only_in_tree:
            body.append(f"  cards only the in-tree reader saw: {list(self.only_in_tree)}")
        if self.only_independent:
            body.append(f"  cards only nvidia-smi saw: {list(self.only_independent)}")
        if self.looks_like_the_carve_out_defect:
            body.append(
                "  EVERY delta equals that card's driver carve-out: this is the "
                "pre-fix `total - used` subtraction (boot weg2rg6, 1454/859/1276 "
                "against 1030/341/852), not a new discrepancy"
            )
        return "\n".join([head] + body)


def pair_verdict(
    in_tree: Mapping[int, int],
    independent: Mapping[int, int],
    tolerance_mib: int = 1,
) -> PairResult:
    """Compare the two readers card by card. THE acceptance of this commit.

    ``tolerance_mib`` is 1 by default and is a ROUNDING allowance, not a
    margin: the two tools floor different byte values (measured 2026-09-08 on
    an idle RTX 3080, ``free`` 20054 via pynvml against 20055 via nvidia-smi),
    and the defect this guards against is 424-518 MiB wide.  Raising it past
    the smallest carve-out on the rig would make the check unable to fail on
    the thing it exists to catch.
    """
    res = PairResult(tolerance_mib=int(tolerance_mib))
    res.only_in_tree = tuple(sorted(set(in_tree) - set(independent)))
    res.only_independent = tuple(sorted(set(independent) - set(in_tree)))
    disagree: List[int] = []
    for idx in sorted(set(in_tree) & set(independent)):
        a, b = int(in_tree[idx]), int(independent[idx])
        res.rows[idx] = (a, b, a - b)
        if abs(a - b) > res.tolerance_mib:
            disagree.append(idx)
    res.disagree = tuple(disagree)
    res.looks_like_the_carve_out_defect = bool(disagree) and all(
        any(abs(res.rows[i][2] - c) <= res.tolerance_mib for c in KNOWN_CARVE_OUT_MIB)
        for i in disagree
    )
    return res


@dataclass
class ArmReport:
    """What one boot's front log says about its own corridor instrument."""

    path: str
    instrument: str = ""
    #: Lines carrying a real ``nvmlN:free=`` measurement.
    samples: int = 0
    #: Lines that mention ``WEG2-CORRIDOR`` and measure nothing -- the front's
    #: own prose about samples.  Printed because a zero in ``samples`` beside a
    #: nonzero here is a different finding from a log with neither (denominator
    #: law), and because reading one of these as a sample is exactly the defect
    #: :func:`ring_table._corridor_sample_phase` closes.
    prose_mentions: int = 0
    #: ``{phase: {nvml index: minimum free MiB}}``, IN THE SOURCE LOG'S OWN
    #: UNIT.  Never graded directly -- see :attr:`units`.
    minima: Dict[str, Dict[int, int]] = field(default_factory=dict)
    #: ``{phase: CorridorUnit}`` -- the same minima brought into ALLOCATABLE
    #: free, the unit :attr:`band_mib` is stated in, or a named refusal.  THE
    #: ONLY THING THIS REPORT GRADES.
    units: Dict[str, ring_table.CorridorUnit] = field(default_factory=dict)
    band_mib: Tuple[int, int] = (0, 0)
    #: #1257c: ``{nvml index: (floor MiB, source)}`` read back off the front's
    #: own CORRIDOR line. Empty on a pre-#1257c log.
    floors: Dict[int, Tuple[int, str]] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)
    #: #1257c: things worth saying that are NOT failures. The upper band edge
    #: lives here by user decision (2026-09-09, consequence 5).
    findings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def report(self) -> str:
        floor, ceil = self.band_mib
        head = (
            f"WEG2-CORRIDOR-ARM log={self.path} instrument={self.instrument} "
            f"samples={self.samples} prose_mentions={self.prose_mentions} "
            f"band={floor}-{ceil}MiB(allocatable free) "
            f"floors={{{', '.join(f'nvml{i}={m}({s})' for i, (m, s) in sorted(self.floors.items()))}}} "
            f"findings={len(self.findings)} "
            f"verdict={'PASS' if self.ok else 'FAIL'}"
        )
        body = []
        for phase in sorted(self.minima):
            unit = self.units.get(phase)
            body.append(f"  phase={phase}(awake) unit: "
                        + (unit.correction_line() if unit else "not converted"))
            for idx, mib in sorted(self.minima[phase].items()):
                # THE SOURCE NUMBER AND THE GRADED NUMBER, ALWAYS BOTH.  The
                # operator greps this line against the boot log, which prints
                # the source unit; printing only the converted figure would
                # make the two disagree with no way to see why.
                if unit is not None and unit.ok and idx in unit.allocatable:
                    got = unit.allocatable[idx]
                    correction = unit.correction_mib.get(idx, 0)
                    shown = f"min_free={got}MiB" + (
                        f"(allocatable; the log printed {mib}MiB, "
                        f"-{correction} MiB carve-out)"
                        if correction
                        else ""
                    )
                    body.append(
                        f"  phase={phase}(awake) nvml{idx}: {shown} "
                        f"verdict={front.corridor_verdict(got)}"
                    )
                else:
                    body.append(
                        f"  phase={phase}(awake) nvml{idx}: min_free={mib}MiB"
                        f"({self.instrument}) verdict=UNGRADED"
                    )
        body += [f"  PROBLEM: {p}" for p in self.problems]
        body += [f"  FINDING: {f}" for f in self.findings]
        return "\n".join([head] + body)


def arm_report(
    path: str,
    require_in_band: bool = False,
    carve_out_by_uuid: Optional[Mapping[str, int]] = None,
) -> ArmReport:
    """Read one front log and grade its corridor instrument.

    ``require_in_band`` is OFF by default and that is deliberate: BELOW the
    floor is a capacity finding for the strand, not evidence that the
    instrument is wrong.  This function's own subject is the INSTRUMENT.

    ONE UNIT (fix 3, round-2 blocker).  The band is stated in ALLOCATABLE free,
    so every number graded against it is brought into that unit first, by the
    SAME converter the ring credit uses (:func:`ring_table.corridor_allocatable`).
    The predecessor graded a pre-fix log's minima -- allocatable free PLUS the
    carve-out -- against the allocatable band, in the same function that had
    already printed "every free is over-stated by that card's driver carve-out"
    two lines earlier: on the real rg6 log it printed ``nvml1: min_free=859MiB
    verdict=IN`` for the 5090 that sat 478 MiB BELOW the floor, and listed the
    two 3080s that were genuinely IN band as ABOVE it.  A log whose samples
    cannot be converted (no card identity, an unknown instrument token, a card
    with no measured or recorded carve-out) is UNGRADED with the reason
    printed, and under ``require_in_band`` that refusal is itself a PROBLEM --
    a strict check must never pass by declining to look.

    ``carve_out_by_uuid`` is an optional measured carve-out per card; without
    it the converter falls back to the recorded per-card constants, which are
    keyed by UUID and so cannot be applied to a card they do not name.
    """
    rep = ArmReport(path=path, band_mib=front.corridor_band_mib())
    try:
        with open(path, errors="replace") as f:
            for line in f:
                if "WEG2-CORRIDOR" not in line:
                    continue
                if ring_table._corridor_sample_phase(line) is None:
                    rep.prose_mentions += 1
                else:
                    rep.samples += 1
    except OSError as e:
        rep.problems.append(f"front log unreadable: {e}")
        return rep

    rep.instrument = ring_table.front_corridor_instrument(path)
    rep.minima = ring_table.parse_front_corridor(path)
    _, by_nvml = ring_table.parse_card_identity(path)
    rep.units = {
        phase: ring_table.corridor_allocatable(
            mins, rep.instrument, by_nvml, carve_out_by_uuid
        )
        for phase, mins in rep.minima.items()
    }

    if rep.samples == 0:
        rep.problems.append(
            "no WEG2-CORRIDOR sample lines (a line carrying at least one "
            "nvmlN:free=<n>MiB field); the sampler did not run"
        )
    if rep.instrument == ring_table.CORRIDOR_INSTRUMENT_PRE_FIX:
        rep.problems.append(
            "samples carry NO instrument= token, i.e. this boot ran the pre-fix "
            "total-minus-used sampler: every free is over-stated by that card's "
            "driver carve-out"
        )
    elif rep.samples and rep.instrument != front.CORRIDOR_INSTRUMENT:
        rep.problems.append(
            f"instrument is {rep.instrument!r}, expected {front.CORRIDOR_INSTRUMENT!r} "
            f"(a boot whose NVML v2 struct was unreadable reports "
            f"{front.CORRIDOR_INSTRUMENT_NO_V2!r} instead -- carve-out-blind, "
            f"and the reserved= field on every line is 0)"
        )
    # #1257c: THE FLOOR COMES FROM THE FRONT'S OWN LINE, per card, and this
    # module grades against THAT. The front derives it once
    # (``corridor_guard.corridor_floor_mib``) and prints ``floor=``/``source=``
    # beside every ``nvmlN:free=``; reading it back is a one-directional
    # shared surface, so the boot's verdict and the sampler that produced it
    # cannot disagree about the number. A log with no ``floor=`` token is a
    # pre-#1257c boot and falls back to the rig-wide band -- named, never
    # silently equated.
    rep.floors = ring_table.parse_front_corridor_floors(path)
    if require_in_band:
        band_floor, band_ceil = rep.band_mib
        for phase in sorted(rep.minima):
            unit = rep.units.get(phase)
            if unit is None or not unit.ok:
                rep.problems.append(
                    f"phase={phase} cannot be graded against the "
                    f"{band_floor}-{band_ceil} MiB band: "
                    f"{unit.reason if unit else 'no unit conversion was attempted'}"
                )
                continue
            for idx, mib in sorted(unit.allocatable.items()):
                got = rep.floors.get(idx)
                floor = got[0] if got else band_floor
                source = got[1] if got else "PRE-1257C-BAND"
                if mib < floor:
                    rep.problems.append(
                        f"phase={phase} nvml{idx} minimum {mib} MiB (allocatable "
                        f"free) is BELOW its corridor floor of {floor} MiB "
                        f"(source={source})"
                    )
                elif got is not None and corridor_guard.unmobilised_free_mib(
                    mib, floor
                ):
                    # DECISION 5, 2026-09-09: the upper edge is a FINDING and
                    # never a FAIL on its own. It says MiB are sitting
                    # unmobilised, which is a capacity question for the
                    # planner, not a breach of the corridor law. The
                    # predecessor of this branch appended it to
                    # ``problems`` and failed acceptances on it.
                    rep.findings.append(
                        f"phase={phase} nvml{idx} unmobilised_free_mib="
                        f"{corridor_guard.unmobilised_free_mib(mib, floor)} "
                        f"(minimum {mib} MiB against a {floor} MiB floor, "
                        f"source={source}); a FINDING, not a failure"
                    )
    return rep


def live_pair(smi_text: str, tolerance_mib: int = 1) -> PairResult:
    """Take the in-tree read NOW and pair it with an already-taken ``smi_text``.

    ``smi_text`` is REQUIRED, not optional: this module does not invoke the
    independent reader (see the module docstring).  The caller runs
    ``nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits``
    and passes its stdout, which also keeps the two reads at one instant --
    the caller controls the ordering, and a pairing across a gap under load
    would fail on drift rather than on the defect.

    Reads only: it allocates nothing on any card and needs no GPU window.
    """
    in_tree = {c.nvml_index: c.free_mib for c in front._nvml_free()}
    return pair_verdict(in_tree, parse_smi_free(smi_text), tolerance_mib=tolerance_mib)
