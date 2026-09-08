"""Item `dormant`, the METAL half's arithmetic -- graded here, sampled by
``/spinning/gpu-arb/weg2/arm_dormant_boot.sh``.

Record section [1y] of ``/spinning/gpu-arb/weg2/WEG2_BUILD_DECISIONS_0906.md``.

WHY THIS IS A MODULE AND NOT A HEREDOC.  FIX 2 put the grading inside the arm
script as a ``python3 - <<'PYCORRIDOR'`` block, where the only way to test it
was to re-extract it with a regex.  It was wrong in a way a single executable
test would have caught (see FIX 3, finding 1, below), and nothing executed it
until a reviewer extracted it by hand.  Here it is importable, hermetic, and
graded by ``test_weg2_dormant_vram_1y.py`` on synthetic samples.

WHAT IT GRADES, and what it refuses to grade.

  A1  SAME-PHASE CONSERVATION.  ``free_after - free_before`` per card, with
      BOTH readings taken in the SAME front phase (P asleep / D awake) at
      opposite ends of one full flip cycle, each stamped with the front's own
      ``awake``/``epoch``.  A perfect run reads ~0.  It fails only in the LOSS
      direction, at one whole workspace: that is the hazard this item can
      actually introduce -- a pause whose resume did not hand the pages back.

      FIX 3, FINDING 1 -- WHY THIS IS NOT A DELIVERY TEST.  Commit 3 graded
      ``delta >= 384`` here and would have failed a PERFECT run by 384 MiB on
      every card, because it sampled both ends in the same phase.  The deeper
      reason it cannot be repaired by moving one sample is arithmetic, and it
      holds for ANY pair of steady phases::

          free            = TOTAL - resident
          resident(P awake) = P_full  + D_slept
          resident(D awake) = P_slept + D_full
          delta             = (P_full - P_slept) - (D_full - D_slept)
                            = P_release - D_release

      Both groups release the same tag, so the item's own contribution CANCELS
      in a cross-phase delta and what is left is the structural P-vs-D size
      difference.  The only in-boot pair that isolates ``P_release`` is
      "both groups awake" vs "P asleep", and the first of those exists only
      before ``launcher.py``'s startup ``sleep_group(PORT_P, ...)``, which runs
      before the front process exists -- i.e. before anything can attach.  So
      delivery is graded by A2 instead, on the instrument that can see it.

  A2  RELEASED-TAG ACCOUNTING, PER GROUP AND THEN IN TOTAL.  The
      ``WEG2-SLEEP released tags=['cuda_graph'] mib=`` lines the sleeping ranks
      emit INSIDE the cycle window (the window is the bytes appended to that
      group's log between the two instants, so it needs no timestamp parsing
      and cannot pick up a previous flip).  This is the item's own claim, per
      rank, at the moment of the release.

      FIX 4, THE FINDING -- SIX RANKS SLEEP, NOT THREE.  The boot is 3+3
      (``launcher.py`` refuses below three ranks per group) and ONE
      ``POST /weg2/flip`` is a ROUND TRIP, not a leg (``front.py``
      ``handle_manual_flip``: it flips awake->other and, whenever that leaves P
      awake, immediately flips back to D).  So inside ONE cycle BOTH groups
      sleep and BOTH release the tag.  FIX 3 read only ``<boot>.P.log``, had no
      D argument at all, and carried ``--ranks 3`` -- so a boot in which all
      three D ranks silently failed to release scored EXIT 5, "the item
      DELIVERED on every rank", off P's evidence alone, and the word D appeared
      nowhere in the report.  MEASURED against the parent commit 53c0ff0e62 on
      exactly that input (P: 3 lines, D: 0): ``EXIT 5``.

      So delivery is graded PER GROUP against ``ranks_per_group`` first, and a
      TOTAL (6 of 6) is printed only once BOTH groups delivered.  Every line is
      attributed to the log it was read from (that is the group -- the line's
      own text does not carry it) and to the rank token in its own prefix
      (``[... PP1]`` / ``[... TP1]``); a line is counted in exactly one group,
      never pooled and never twice.  Both group logs are REQUIRED: a P-only
      invocation is refused by argparse before any grading happens, and a group
      whose window could not be read is named in the refusal.

  B   WHERE THE CARD SITS.  ``free_after`` against the 819-1229 MiB corridor
      band.  A card that delivered and is still below the floor is a RESIDUAL,
      exit 5, not this wiring failing.

  The no-item column of table B is a CROSS-BOOT INDICATOR and never a grade:
  it pairs THIS boot's reading with ANOTHER boot's (weg2rg6, which ran without
  the item).  Different boot, possibly different rank->card layout; it is
  printed because it is the only baseline that can show delivery as a level,
  and labelled because a cross-boot number must never be an exit code.

EXIT CODES (the shell propagates them unchanged):
  0  the item delivered on every rank of BOTH groups and no card is below the
     floor
  2  a refusal, always named -- including "could not measure" and "which group
     could not be measured", never a "probably fine"
  4  the item did NOT deliver in at least one GROUP (A2, named), or the cycle
     lost a workspace (A1)
  5  it delivered in both groups and a card is still below the 819 MiB floor
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Dict, List, NamedTuple, Optional, Tuple

#: The flashinfer FLOAT workspace, MiB.  UNIFORM across ranks -- it is
#: ``SGLANG_FLASHINFER_WORKSPACE_SIZE``'s default, not a per-rank reading --
#: which is why it, and not the per-rank capture pool, is every edge below.
WORKSPACE_MIB = 384

#: The corridor band (memory `vram-korridor-regel`), NVML memory.free per card.
FLOOR_MIB, CEIL_MIB = 819, 1229

#: Emitted by ``weight_updater.py`` at the sleep, one line per sleeping rank.
#: ``mib=0.0`` is the saver's own "could not answer" sentinel and is NOT an
#: empty tag -- the line's own text says so, and A2 refuses it rather than
#: averaging it in.
RELEASED_RE = re.compile(
    r"WEG2-SLEEP released tags=\['cuda_graph'\]\s+mib=([0-9.]+)\s+ms=([0-9.]+)"
)

#: The rank token in the log line's OWN prefix -- ``[2026-09-08 07:06:31 PP1]``
#: on a P rank, ``[... TP1]`` on a D rank.  The released line's TEXT does not
#: name the rank and cannot name the group; this is where the rank comes from,
#: and the GROUP comes from which log the line was read out of.
RANK_RE = re.compile(r"\[(?:[^\]\n]*?\s)?([A-Z]{2,5}\d+)\]")

#: The two process groups of a Weg-2 boot.  Three ranks each (``launcher.py``
#: refuses below three per group), six sleeping ranks per flip cycle.
GROUPS = ("P", "D")


class ReleasedLine(NamedTuple):
    """One released-tag line, carrying WHERE it came from.

    ``group`` is the log it was read out of -- never inferred from the text,
    which carries no group token at all.  ``rank`` is the token in the line's
    own prefix, or ``"?"`` when the line has no prefix (a synthetic sample).
    """

    group: str
    rank: str
    mib: float
    ms: float

#: The named degrades, all of them.  FIX 3, finding 2: the group-gate refusal
#: is new in this fix; before it, a rank that lost ``SGLANG_WEG2_GROUP`` was
#: byte-identical in the log to a rank running the unchanged base tree.
DEGRADE_MARKERS = (
    "WEG2-SLEEP graph tag NOT armed",
    "WEG2-SLEEP graph scratch NOT tagged",
)


def read_free_csv(path: str) -> Dict[int, Tuple[str, int]]:
    """``nvidia-smi --query-gpu=index,name,memory.free`` rows -> {idx: (name, MiB)}.

    memory.free, never total-used: the RM carve-out (424 MiB on a 3080, 518 on
    the 5090) is invisible in total-used, which is the defect weg2rg6 found in
    the front's own ``_nvml_free()``.
    """
    out: Dict[int, Tuple[str, int]] = {}
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        idx, name, free = [x.strip() for x in ln.split(",")]
        out[int(idx)] = (name, int(free))
    return out


def parse_capture_map(arg: str) -> Dict[int, int]:
    """``0=102,1=92,2=133`` -> {nvml index: capture-pool MiB}."""
    out: Dict[int, int] = {}
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=")
        out[int(k)] = int(v)
    return out


def parse_released(window_text: str, group: str = "?") -> List[ReleasedLine]:
    """Every released-tag line in ONE group's window, attributed to that group.

    The group is a property of the LOG, not of the line: the emitter
    (``weight_updater.py``) writes no group token, so a pooled parse over two
    windows could not tell a P rank from a D rank afterwards.  It is therefore
    stamped here, at the only point that still knows which file the bytes came
    from, and a line parsed out of one window can never be counted in the other.
    """
    out: List[ReleasedLine] = []
    for ln in window_text.splitlines():
        m = RELEASED_RE.search(ln)
        if not m:
            continue
        rank = RANK_RE.search(ln)
        out.append(
            ReleasedLine(group, rank.group(1) if rank else "?", float(m.group(1)), float(m.group(2)))
        )
    return out


def find_degrades(text: str, group: str = "") -> List[str]:
    """The named degrade lines, deduplicated, in first-seen order.

    Searched over the WHOLE group log rather than the cycle window on purpose:
    the group-gate refusal is rate-limited to once per reason per process and
    fires at the first sleep, which is the launcher's startup sleep -- long
    before this arm attaches.  A window-only search would report "no degrade"
    on exactly the boot that degraded.

    ``group``, when given, is prefixed to every hit: the scan runs over BOTH
    group logs (FIX 4) and a degrade that names no group sends the reader to
    the wrong three ranks.
    """
    seen: List[str] = []
    for ln in text.splitlines():
        for marker in DEGRADE_MARKERS:
            hit = ("group %s: %s" % (group, ln.strip())) if group else ln.strip()
            if marker in ln and hit not in seen:
                seen.append(hit)
    return seen


def _tally(
    released_by_group: Dict[str, Optional[List[ReleasedLine]]], ranks_per_group: int
) -> str:
    """``P 3/3, D 3/3`` -- the per-group counts, always both, always named."""
    return ", ".join(
        "%s %s/%d"
        % (
            g,
            "n/a" if released_by_group.get(g) is None else len(released_by_group[g]),
            ranks_per_group,
        )
        for g in GROUPS
    )


def _group_shortfalls(
    released_by_group: Dict[str, Optional[List[ReleasedLine]]], ranks_per_group: int
) -> List[str]:
    """Why each group failed A2, in group order -- '' when a group delivered.

    Graded PER GROUP on purpose: pooling the two windows and testing the sum
    against six lets one group cover for the other, which is the same class of
    error as grading three of six and naming all six.  The order inside a group
    is count -> sentinel -> short, because "no line at all", "the saver could
    not answer" and "the rank released 383 of 384 MiB" are three different
    faults with three different next steps.
    """
    out: List[str] = []
    for group in GROUPS:
        rel = released_by_group.get(group)
        if rel is None:
            continue  # unmeasured is a refusal (exit 2), not a delivery failure
        if len(rel) < ranks_per_group:
            out.append(
                "group %s: %d released-tag line(s) in the cycle window, expected %d "
                "(one per sleeping rank of that group)" % (group, len(rel), ranks_per_group)
            )
        elif any(x.mib <= 0.0 for x in rel):
            out.append(
                "group %s: a released-tag line reads mib=0.0, which is the saver's "
                "'could not answer' sentinel and not an empty tag (rank(s) %s)"
                % (group, ", ".join(x.rank for x in rel if x.mib <= 0.0))
            )
        elif any(x.mib < WORKSPACE_MIB for x in rel):
            out.append(
                "group %s: a rank released less than the %d MiB workspace alone (%s)"
                % (
                    group,
                    WORKSPACE_MIB,
                    ", ".join("%s=%.1f" % (x.rank, x.mib) for x in rel if x.mib < WORKSPACE_MIB),
                )
            )
    return out


def _phase_of(instant: str) -> str:
    """``phase=D(awake) epoch=45`` -> ``D``.  '' when the string does not say."""
    m = re.search(r"phase=([A-Za-z]+)", instant)
    return m.group(1) if m else ""


def grade(
    before: Dict[int, Tuple[str, int]],
    after: Dict[int, Tuple[str, int]],
    instant_before: str,
    instant_after: str,
    capture: Dict[int, int],
    capture_prov: str,
    released_by_group: Dict[str, Optional[List[ReleasedLine]]],
    degrades_by_group: Dict[str, List[str]],
    ranks_per_group: int,
    noitem: Dict[int, int],
    noitem_prov: str,
    p_awake: Optional[Dict[int, Tuple[str, int]]] = None,
    instant_p_awake: str = "",
) -> Tuple[int, List[str]]:
    """Return ``(exit code, report lines)``.  No I/O, so it is testable."""
    lines: List[str] = []

    # ---- refusal 0: the two instants must be the same phase ---------------
    pb, pa = _phase_of(instant_before), _phase_of(instant_after)
    if not pb or not pa:
        return 2, [
            "REFUSED: an instant does not name its phase (before=%r after=%r). "
            "Both readings must carry the front's own awake/epoch or the delta "
            "below is a comparison of two unknown states." % (instant_before, instant_after)
        ]
    if pb != pa:
        return 2, [
            "REFUSED: the two NVML readings were taken in DIFFERENT front phases "
            "(before %s, after %s). A cross-phase delta measures "
            "P_release - D_release, in which this item's own contribution "
            "cancels; it is not a delivery measurement and is not graded as one."
            % (instant_before, instant_after)
        ]

    lines += [
        "",
        "INSTANTS (the front's own phase/epoch, printed beside both readings):",
        "  before: %s" % instant_before,
        "  after : %s" % instant_after,
        "  P-awake (level only, never graded): %s"
        % (instant_p_awake if instant_p_awake else "n/a (the P-awake window was not caught)"),
        "",
        "A1. SAME-PHASE CONSERVATION across one full flip cycle -- NOT a delivery",
        "    test (delivery is A2; the cancellation arithmetic is in the module",
        "    docstring). A perfect run reads ~0.",
        "",
        "| nvml | card | free before | free after | delta | verdict |",
        "|---|---|---:|---:|---:|---|",
    ]
    lost: List[Tuple[int, int]] = []
    for idx in sorted(before):
        name, fb = before[idx]
        fa = after.get(idx, (name, -1))[1]
        delta = fa - fb
        if delta <= -WORKSPACE_MIB:
            verdict = "LOST A WORKSPACE"
            lost.append((idx, delta))
        elif delta >= WORKSPACE_MIB:
            # Not graded here: a workspace that was released and never remapped
            # shows up as a WRONG ANSWER, and the answer probe is that
            # instrument.  Named so the reader does not read silence as zero.
            verdict = "gained >= one workspace (see the answer probe, not this table)"
        else:
            verdict = "conserved"
        lines.append(
            "| %d | %s | %d | %d | %+d | %s |" % (idx, name, fb, fa, delta, verdict)
        )
    lines.append("")
    lines.append(
        "A1 edge = one whole workspace (%d MiB) in the LOSS direction, because "
        "that is the quantity this item moves; a smaller swing is not "
        "attributable to it." % WORKSPACE_MIB
    )

    if p_awake:
        lines += [
            "",
            "P-AWAKE PHASE LEVEL (printed, never graded -- the delta against the",
            "other phase is P_release - D_release, in which this item cancels):",
            "",
            "| nvml | card | free (P awake) | free (P asleep, after) | phase delta |",
            "|---|---|---:|---:|---:|",
        ]
        for idx in sorted(p_awake):
            name, fp = p_awake[idx]
            fa = after.get(idx, (name, -1))[1]
            lines.append("| %d | %s | %d | %d | %+d |" % (idx, name, fp, fa, fa - fp))

    # ---- A2: the released-tag accounting, PER GROUP then in total ---------
    total_ranks = ranks_per_group * len(GROUPS)
    lines += [
        "",
        "A2. DID THE ITEM DELIVER -- the sleeping ranks' own released-tag lines",
        "    inside each group's OWN cycle window (instrument: tms_tag_bytes for",
        "    the ONE cuda_graph tag, read before the pause).",
        "",
        "    SIX ranks sleep in one cycle, not three: the boot is %d+%d and ONE"
        % (ranks_per_group, ranks_per_group),
        "    POST /weg2/flip is a ROUND TRIP, so BOTH groups sleep and both",
        "    release the tag. Each line below is attributed to the LOG it was",
        "    read from (the group -- the line's text carries no group token) and",
        "    to the rank in its own prefix; no line is pooled or counted twice.",
        "",
    ]
    unmeasured = [g for g in GROUPS if released_by_group.get(g) is None]
    could_not_measure = bool(unmeasured)

    lines.append("| group | rank | mib | against floor %d | verdict |" % WORKSPACE_MIB)
    lines.append("|---|---|---:|---:|---|")
    for group in GROUPS:
        rel = released_by_group.get(group)
        if rel is None:
            lines.append(
                "| %s | n/a | n/a | %d | GROUP LOG WINDOW COULD NOT BE READ |"
                % (group, WORKSPACE_MIB)
            )
            continue
        if not rel:
            lines.append(
                "| %s | n/a | n/a | %d | NO RELEASED LINE IN THIS GROUP'S WINDOW |"
                % (group, WORKSPACE_MIB)
            )
            continue
        for item in rel:
            if item.mib <= 0.0:
                verdict = "SAVER COULD NOT ANSWER (0.0 is the sentinel, not an empty tag)"
            elif item.mib < WORKSPACE_MIB:
                verdict = "SHORT of the workspace alone"
            else:
                verdict = "at or above the workspace"
            lines.append(
                "| %s | %s | %.1f | %d | %s |"
                % (item.group, item.rank, item.mib, WORKSPACE_MIB, verdict)
            )

    if could_not_measure:
        lines.append("")
        lines.append(
            "released tags: n/a for group(s) %s -- that group log window could not "
            "be read. A probe that could not measure prints n/a and exits 2; it "
            "does not pass, and it does not fall back to the group it CAN read."
            % ", ".join(unmeasured)
        )

    # The per-group denominator, stated before any total is claimed.
    shortfalls: List[str] = _group_shortfalls(released_by_group, ranks_per_group)
    lines += ["", "DELIVERY PER GROUP (denominator: %d ranks per group):" % ranks_per_group]
    for group in GROUPS:
        rel = released_by_group.get(group)
        if rel is None:
            lines.append("  %s: n/a -- window unreadable, not graded and not passed" % group)
        else:
            lines.append(
                "  %s: %d/%d released%s"
                % (
                    group,
                    len(rel),
                    ranks_per_group,
                    ", min %.1f MiB" % min(x.mib for x in rel) if rel else "",
                )
            )
    if not could_not_measure and not shortfalls:
        lines.append(
            "TOTAL: %d/%d sleeping ranks released the tag (%s)."
            % (total_ranks, total_ranks, _tally(released_by_group, ranks_per_group))
        )
    else:
        lines.append(
            "TOTAL: NOT CLAIMED -- %s. A six-rank claim carried by one group's "
            "evidence is exactly the defect FIX 4 closes; the total is printed "
            "only when both groups delivered."
            % ("; ".join(shortfalls) if shortfalls else "group(s) %s unmeasured" % ", ".join(unmeasured))
        )

    all_degrades = [d for group in GROUPS for d in degrades_by_group.get(group, [])]
    if all_degrades:
        lines.append("")
        lines.append(
            "NAMED DEGRADES FOUND IN THE GROUP LOGS (whole files, both groups, not "
            "just the windows):"
        )
        for d in all_degrades:
            lines.append("  " + d)
    elif not could_not_measure:
        lines.append("")
        lines.append(
            "named degrades: none in EITHER group log. The refusal instrument is "
            "weg2_memory_saver.weg2_graph_tag_armed, which logs "
            "'WEG2-SLEEP graph tag NOT armed' with the failing conjunct named; "
            "its absence is evidence only because that line exists (FIX 3, "
            "finding 2 -- before it, a rank that lost SGLANG_WEG2_GROUP was "
            "byte-identical to one running the base tree)."
        )

    if not could_not_measure:
        lines.append("")
        lines.append(
            "expected per rank: %d MiB workspace + this rank's own capture pool "
            "(%s). The pool is per rank and is REPORTED, never a pass/fail edge; "
            "the workspace is uniform and is the edge. Provenance: %s"
            % (WORKSPACE_MIB, ", ".join("nvml%d=%d" % (k, v) for k, v in sorted(capture.items())), capture_prov)
        )

    # ---- B: where the card sits ------------------------------------------
    lines += [
        "",
        "B. WHERE THE CARD SITS (corridor %d-%d MiB, NVML memory.free)" % (FLOOR_MIB, CEIL_MIB),
        "",
        "| nvml | card | free after | band | cross-boot no-item indicator | note |",
        "|---|---|---:|---|---:|---|",
    ]
    below: List[Tuple[int, int]] = []
    for idx in sorted(before):
        name, fb = before[idx]
        fa = after.get(idx, (name, -1))[1]
        pool = capture.get(idx)
        base = noitem.get(idx)
        if base is None or pool is None:
            shown = "n/a"
        else:
            shown = "%d+%d+%d=%d" % (base, WORKSPACE_MIB, pool, base + WORKSPACE_MIB + pool)
        band = "IN BAND" if FLOOR_MIB <= fa <= CEIL_MIB else ("BELOW" if fa < FLOOR_MIB else "ABOVE")
        if fa < FLOOR_MIB:
            below.append((idx, fa))
            note = "RESIDUAL: still short of the floor after a full release"
        elif band == "ABOVE":
            note = "reported, not failed (ABOVE grades the AWAKE group under load)"
        else:
            note = ""
        lines.append("| %d | %s | %d | %s | %s | %s |" % (idx, name, fa, band, shown, note))
    lines.append("")
    lines.append(
        "free = NVML memory.free (allocatable), never total-used: the RM "
        "carve-out (424 MiB / 3080, 518 MiB / 5090) is invisible in total-used."
    )
    lines.append(
        "CROSS-BOOT NO-ITEM INDICATOR = another boot's free + %d + this card's "
        "own capture pool. It is an INDICATOR, never an exit code: it pairs this "
        "boot's reading with %s. On that layout the 5090 predicts 340+%d+92=816 "
        "MiB, below the %d floor -- so this item cannot put that card in band on "
        "its own and a run that lands there is EXIT 5, not broken wiring."
        % (WORKSPACE_MIB, noitem_prov, WORKSPACE_MIB, FLOOR_MIB)
    )

    # ---- the verdict ------------------------------------------------------
    tally = _tally(released_by_group, ranks_per_group)
    if could_not_measure:
        rc = 2
        lines += [
            "",
            "VERDICT: EXIT 2 -- A2 could not be measured for group(s) %s (no cycle "
            "window for that group's log). Lines seen: %s. The groups that COULD "
            "be read are not promoted to a verdict for the boot."
            % (", ".join(unmeasured), tally),
        ]
    elif lost:
        rc = 4
        lines += [
            "",
            "VERDICT: EXIT 4 -- the cycle LOST at least one workspace on "
            + ", ".join("nvml%d (%+d MiB)" % (i, d) for i, d in lost)
            + ": the pause's pages did not come back at the resume.",
        ]
    elif shortfalls:
        rc = 4
        lines += [
            "",
            "VERDICT: EXIT 4 -- the item did NOT deliver (lines: %s). %s. Read the "
            "named degrades above before anything else."
            % (tally, "; ".join(shortfalls)),
        ]
    elif below:
        rc = 5
        lines += [
            "",
            "VERDICT: EXIT 5 -- the item DELIVERED on all %d sleeping ranks (%s), "
            "and " % (total_ranks, tally)
            + ", ".join("nvml%d is still at %d MiB" % (i, f) for i, f in below)
            + ". The corridor is not solved by this item alone; that is a "
              "residual for the next one, with a number.",
        ]
    else:
        rc = 0
        lines += [
            "",
            "VERDICT: EXIT 0 -- all %d sleeping ranks (%s) released at least the "
            "%d MiB workspace, the cycle conserved it, and no card is below the "
            "%d MiB floor." % (total_ranks, tally, WORKSPACE_MIB, FLOOR_MIB),
        ]
    return rc, lines


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--instant-before", required=True)
    ap.add_argument("--instant-after", required=True)
    ap.add_argument("--free-p-awake", default="")
    ap.add_argument("--instant-p-awake", default="")
    # BOTH group logs are REQUIRED (FIX 4).  Six ranks sleep in one cycle; a
    # P-only invocation cannot grade the boot, so it is refused HERE, by
    # argparse, naming the missing option -- not warned about downstream.
    ap.add_argument("--plog-window", required=True, help="bytes appended to the P group log during the cycle")
    ap.add_argument("--plog", required=True, help="the whole P group log, searched for named degrades")
    ap.add_argument("--dlog-window", required=True, help="bytes appended to the D group log during the cycle")
    ap.add_argument("--dlog", required=True, help="the whole D group log, searched for named degrades")
    ap.add_argument("--report", default="")
    ap.add_argument("--capture-mib", default="0=102,1=92,2=133")
    ap.add_argument(
        "--capture-prov",
        default="boot weg2rg6 P.log, 'Capture target decode CUDA graph end ... mem usage=' "
                "0.09/0.10/0.13 GB for PP0/PP1/PP2 = nvml1/nvml0/nvml2",
    )
    ap.add_argument("--noitem-mib", default="0=1029,1=340,2=851")
    ap.add_argument(
        "--noitem-prov",
        default="boot weg2rg6 (7f88b1c75d, no item), NVML memory.free while D served",
    )
    ap.add_argument(
        "--ranks-per-group",
        type=int,
        default=3,
        help="sleeping ranks per group (launcher.py refuses below 3); the total "
             "graded is this x2, because one flip is a round trip and both "
             "groups sleep inside it",
    )
    ns = ap.parse_args(argv)

    windows = {"P": ns.plog_window, "D": ns.dlog_window}
    whole = {"P": ns.plog, "D": ns.dlog}
    released_by_group: Dict[str, Optional[List[ReleasedLine]]] = {}
    degrades_by_group: Dict[str, List[str]] = {}
    for group in GROUPS:
        try:
            released_by_group[group] = parse_released(
                open(windows[group], errors="replace").read(), group
            )
        except OSError:
            # Named, never silent: an unreadable window is a refusal for THAT
            # group, and grade() will not pass the boot on the other one.
            released_by_group[group] = None
        try:
            degrades_by_group[group] = find_degrades(
                open(whole[group], errors="replace").read(), group
            )
        except OSError:
            degrades_by_group[group] = []

    p_awake = None
    if ns.free_p_awake:
        try:
            p_awake = read_free_csv(ns.free_p_awake)
        except OSError:
            p_awake = None

    rc, lines = grade(
        before=read_free_csv(ns.before),
        after=read_free_csv(ns.after),
        instant_before=ns.instant_before,
        instant_after=ns.instant_after,
        capture=parse_capture_map(ns.capture_mib),
        capture_prov=ns.capture_prov,
        released_by_group=released_by_group,
        degrades_by_group=degrades_by_group,
        ranks_per_group=ns.ranks_per_group,
        noitem=parse_capture_map(ns.noitem_mib),
        noitem_prov=ns.noitem_prov,
        p_awake=p_awake,
        instant_p_awake=ns.instant_p_awake,
    )
    txt = "\n".join(lines)
    print(txt)
    if ns.report:
        with open(ns.report, "a") as fh:
            fh.write(txt + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
