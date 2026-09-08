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

  A2  RELEASED-TAG ACCOUNTING.  The ``WEG2-SLEEP released tags=['cuda_graph']
      mib=`` lines the sleeping ranks emit INSIDE the cycle window (the window
      is the bytes appended to the group log between the two instants, so it
      needs no timestamp parsing and cannot pick up a previous flip).  This is
      the item's own claim, per rank, at the moment of the release.

  B   WHERE THE CARD SITS.  ``free_after`` against the 819-1229 MiB corridor
      band.  A card that delivered and is still below the floor is a RESIDUAL,
      exit 5, not this wiring failing.

  The no-item column of table B is a CROSS-BOOT INDICATOR and never a grade:
  it pairs THIS boot's reading with ANOTHER boot's (weg2rg6, which ran without
  the item).  Different boot, possibly different rank->card layout; it is
  printed because it is the only baseline that can show delivery as a level,
  and labelled because a cross-boot number must never be an exit code.

EXIT CODES (the shell propagates them unchanged):
  0  the item delivered on every card and no card is below the floor
  2  a refusal, always named -- including "could not measure", never a
     "probably fine"
  4  the item did NOT deliver (A2), or the cycle lost a workspace (A1)
  5  it delivered everywhere and a card is still below the 819 MiB floor
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Dict, List, Optional, Tuple

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


def parse_released(window_text: str) -> List[float]:
    """Every ``WEG2-SLEEP released tags=['cuda_graph'] mib=`` value in the window."""
    return [float(m.group(1)) for m in RELEASED_RE.finditer(window_text)]


def find_degrades(text: str) -> List[str]:
    """The named degrade lines, deduplicated, in first-seen order.

    Searched over the WHOLE group log rather than the cycle window on purpose:
    the group-gate refusal is rate-limited to once per reason per process and
    fires at the first sleep, which is the launcher's startup sleep -- long
    before this arm attaches.  A window-only search would report "no degrade"
    on exactly the boot that degraded.
    """
    seen: List[str] = []
    for ln in text.splitlines():
        for marker in DEGRADE_MARKERS:
            if marker in ln and ln.strip() not in seen:
                seen.append(ln.strip())
    return seen


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
    released: Optional[List[float]],
    degrades: List[str],
    ranks: int,
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

    # ---- A2: the released-tag accounting ---------------------------------
    lines += [
        "",
        "A2. DID THE ITEM DELIVER -- the sleeping ranks' own released-tag lines",
        "    inside the cycle window (instrument: tms_tag_bytes for the ONE",
        "    cuda_graph tag, read before the pause).",
        "",
    ]
    could_not_measure = released is None
    if could_not_measure:
        lines.append(
            "released tags: n/a -- the group log window could not be read. A probe "
            "that could not measure prints n/a and exits 2; it does not pass."
        )
    else:
        lines.append(
            "| line | mib | against floor %d | verdict |" % WORKSPACE_MIB
        )
        lines.append("|---:|---:|---:|---|")
        for i, mib in enumerate(released):
            if mib <= 0.0:
                verdict = "SAVER COULD NOT ANSWER (0.0 is the sentinel, not an empty tag)"
            elif mib < WORKSPACE_MIB:
                verdict = "SHORT of the workspace alone"
            else:
                verdict = "at or above the workspace"
            lines.append("| %d | %.1f | %d | %s |" % (i + 1, mib, WORKSPACE_MIB, verdict))
        lines.append("")
        lines.append(
            "expected per rank: %d MiB workspace + this rank's own capture pool "
            "(%s). The pool is per rank and is REPORTED, never a pass/fail edge; "
            "the workspace is uniform and is the edge. Provenance: %s"
            % (WORKSPACE_MIB, ", ".join("nvml%d=%d" % (k, v) for k, v in sorted(capture.items())), capture_prov)
        )

    if degrades:
        lines.append("")
        lines.append("NAMED DEGRADES FOUND IN THE GROUP LOG (whole file, not just the window):")
        for d in degrades:
            lines.append("  " + d)
    elif not could_not_measure:
        lines.append("")
        lines.append(
            "named degrades: none. The refusal instrument is "
            "weg2_memory_saver.weg2_graph_tag_armed, which logs "
            "'WEG2-SLEEP graph tag NOT armed' with the failing conjunct named; "
            "its absence is evidence only because that line exists (FIX 3, "
            "finding 2 -- before it, a rank that lost SGLANG_WEG2_GROUP was "
            "byte-identical to one running the base tree)."
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
    if could_not_measure:
        rc = 2
        lines += ["", "VERDICT: EXIT 2 -- A2 could not be measured (no group-log window)."]
    elif lost:
        rc = 4
        lines += [
            "",
            "VERDICT: EXIT 4 -- the cycle LOST at least one workspace on "
            + ", ".join("nvml%d (%+d MiB)" % (i, d) for i, d in lost)
            + ": the pause's pages did not come back at the resume.",
        ]
    elif len(released) < ranks:
        rc = 4
        lines += [
            "",
            "VERDICT: EXIT 4 -- the item did NOT deliver: %d released-tag line(s) "
            "in the cycle window, expected %d (one per sleeping rank). Read the "
            "named degrades above before anything else." % (len(released), ranks),
        ]
    elif any(m <= 0.0 for m in released):
        rc = 4
        lines += [
            "",
            "VERDICT: EXIT 4 -- the item did NOT deliver: a released-tag line "
            "reads mib=0.0, which is the saver's 'could not answer' sentinel and "
            "not an empty tag.",
        ]
    elif any(m < WORKSPACE_MIB for m in released):
        rc = 4
        lines += [
            "",
            "VERDICT: EXIT 4 -- the item did NOT deliver: a rank released less "
            "than the %d MiB workspace alone (%s)."
            % (WORKSPACE_MIB, ", ".join("%.1f" % m for m in released)),
        ]
    elif below:
        rc = 5
        lines += [
            "",
            "VERDICT: EXIT 5 -- the item DELIVERED on every rank, and "
            + ", ".join("nvml%d is still at %d MiB" % (i, f) for i, f in below)
            + ". The corridor is not solved by this item alone; that is a "
              "residual for the next one, with a number.",
        ]
    else:
        rc = 0
        lines += [
            "",
            "VERDICT: EXIT 0 -- every sleeping rank released at least the %d MiB "
            "workspace, the cycle conserved it, and no card is below the %d MiB "
            "floor." % (WORKSPACE_MIB, FLOOR_MIB),
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
    ap.add_argument("--plog-window", default="", help="bytes appended to the group log during the cycle")
    ap.add_argument("--plog", default="", help="the whole group log, searched for named degrades")
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
    ap.add_argument("--ranks", type=int, default=3)
    ns = ap.parse_args(argv)

    released: Optional[List[float]] = None
    if ns.plog_window:
        try:
            released = parse_released(open(ns.plog_window, errors="replace").read())
        except OSError:
            released = None
    degrades: List[str] = []
    if ns.plog:
        try:
            degrades = find_degrades(open(ns.plog, errors="replace").read())
        except OSError:
            degrades = []

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
        released=released,
        degrades=degrades,
        ranks=ns.ranks,
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
