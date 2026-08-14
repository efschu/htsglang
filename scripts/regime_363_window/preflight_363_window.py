#!/usr/bin/env python3
"""#363 stage-clock window -- CPU-only preflight. Run this BEFORE claiming cards.

WHY THIS EXISTS. The stage-clock window needs ``--regime-controller act``, and
``act`` is refused **at parse time** by a gate that is not obvious from the
launch line. A window that discovers this after claiming the cards has spent a
GPU window on an argument parser. Everything checked here is checkable with no
GPU, no serving process and no model, so it costs seconds and it is the
difference between a window that runs and a window that refuses.

WHAT IT CHECKS, and where each requirement comes from:

  C1 FLAGS PARSE. Every flag the runsheet uses is accepted by *this tree's*
     ``server_args``. Note that a literal grep for ``"--regime-stage-clock"``
     finds NOTHING: the flags are derived from annotated dataclass FIELD names
     (``regime_stage_clock: A[bool, Arg(...)]``, server_args.py:5327), so the
     only honest check is to parse them. This is the AUDIT-251 assembled-name
     trap in its natural habitat.

  C2 ENTRY GATE. ``EntryGate.open`` requires ALL FOUR items -- desyncs_zero,
     f2_live_replay, f3_bands_measured, f4_card_comparison -- each with
     ``passed: true`` AND a non-empty ``source``
     (regime_stages.py:465-511, GATE_ITEMS at :422-461).

     **THE CIRCULARITY, NAMED.** ``f4_card_comparison`` is produced by "three
     arms (off / observe / act) over one workload" (:452-459) -- so producing
     gate 4's evidence requires an ``act`` arm, and ``act`` requires gate 4
     already passed. RUNSHEET_363_card_gates.md section 7 says gate 4 "needs
     gates 1-3 in the evidence file first", which does NOT match the code:
     the code needs four. This tool reports that state explicitly instead of
     letting a window discover it, and section 3 of the window runsheet says
     what to write and how to attribute it honestly.

  C3 ACTUATOR WIRED. ``act`` additionally refuses unless at least one of
     ``--kv-reshard-vectors`` / ``--enable-vram-dial`` is given
     (server_args.py:7784-7793). The stage-clock ticket's own boot recipe
     omits both and would be refused; the window runsheet adds one.

  C4 STAGE TABLE MEASURED. The intra-phase axis never proposes an UNMEASURED
     stage -- "its placeholder zeros are not a measured gain of zero"
     (server_args.py, regime_stage_clock help). A planner-only table produces
     a clean run in which nothing happens and the window proves nothing
     (TICKET_363_STAGE_CLOCK.md P1).

  C5 TRANSIENT CENSUS. Without a census for the target stage a flip is
     REFUSED rather than priced at zero (same help text; ticket P3). The
     census is written under SGLANG_RESIDENCY_CENSUS_DIR (environ.py:659)
     with SGLANG_TRANSIENT_CENSUS=1 (:667).

EXIT CODES
  0  every check PASSed (and, with --strict, none were SKIPped)
  1  at least one check FAILed (or SKIPped under --strict)
  2  usage error

A check that cannot be shown failing is unvalidated, so ``--smoke`` drives
every check into its red state against fabricated fixtures.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

# The flags this window's boots actually use. Kept here so the preflight and
# the runsheet cannot drift: if a boot line grows a flag, it is added here and
# the preflight proves the tree still accepts it.
WINDOW_FLAGS = [
    "--regime-controller",
    "--regime-trace",
    "--regime-gate-evidence",
    "--regime-stage-clock",
    "--kv-reshard-vectors",
]

GATE_KEYS = (
    "desyncs_zero",
    "f2_live_replay",
    "f3_bands_measured",
    "f4_card_comparison",
)

FAILURES = 0
SKIPS = 0


def _p(status: str, name: str, msg: str) -> None:
    print(f"{status:<5} {name:<22} {msg}")


def ok(name: str, msg: str) -> None:
    _p("PASS", name, msg)


def bad(name: str, msg: str) -> None:
    global FAILURES
    FAILURES += 1
    _p("FAIL", name, msg)


def skip(name: str, msg: str) -> None:
    global SKIPS
    SKIPS += 1
    _p("SKIP", name, msg)


def info(name: str, msg: str) -> None:
    _p("INFO", name, msg)


# ---------------------------------------------------------------------------
# C1 -- the flags parse in THIS tree.
# ---------------------------------------------------------------------------
def check_flags_parse() -> None:
    name = "flags-parse"
    try:
        from sglang.srt.server_args import ServerArgs
    except Exception as exc:  # pragma: no cover - import failure is the finding
        bad(name, f"cannot import server_args ({type(exc).__name__}: {exc})")
        return

    parser = argparse.ArgumentParser()
    try:
        ServerArgs.add_cli_args(parser)
    except Exception as exc:
        bad(name, f"add_cli_args raised {type(exc).__name__}: {exc}")
        return

    known = set()
    for action in parser._actions:
        known.update(action.option_strings)

    missing = [f for f in WINDOW_FLAGS if f not in known]
    if missing:
        bad(name, f"this tree does not accept: {', '.join(missing)}")
    else:
        ok(name, f"all {len(WINDOW_FLAGS)} window flags accepted by server_args")


# ---------------------------------------------------------------------------
# C2 -- the entry gate, and the gate-4 circularity said out loud.
# ---------------------------------------------------------------------------
def check_entry_gate(path: str | None) -> None:
    name = "entry-gate"
    if not path:
        skip(name, "no --evidence given; act would be refused (gate closed, no file)")
        return
    try:
        from sglang.srt.managers.regime_stages import load_gate_evidence
    except Exception as exc:
        bad(name, f"cannot import regime_stages ({type(exc).__name__}: {exc})")
        return

    try:
        gate = load_gate_evidence(path)
    except Exception as exc:
        # A declared-but-unparseable file is an ERROR by design, not a closed
        # gate -- otherwise the refusal reads as "not measured yet".
        bad(name, f"evidence file declared but unreadable: {exc}")
        return

    summary = gate.summary()
    if gate.open:
        ok(name, f"gate OPEN; all four items passed with sources ({path})")
        return

    missing = summary.get("missing") or []
    malformed = summary.get("malformed") or []
    bad(
        name,
        "gate CLOSED -- act refused at parse time. "
        f"missing={missing or '[]'} malformed={malformed or '[]'}",
    )
    if "f4_card_comparison" in missing and len(missing) == 1:
        info(
            name,
            "ONLY f4_card_comparison is missing. This is the known circularity: "
            "gate 4 is produced by an act arm, and act needs gate 4. See the "
            "window runsheet section 3 for the honest bootstrap.",
        )


# ---------------------------------------------------------------------------
# C3 -- act needs an actuator wired, or it is an expensive observe.
# ---------------------------------------------------------------------------
def check_actuator(kv_vectors: str | None, vram_dial: bool) -> None:
    name = "actuator-wired"
    if kv_vectors or vram_dial:
        which = "--kv-reshard-vectors" if kv_vectors else "--enable-vram-dial"
        ok(name, f"{which} present; act has somewhere to actuate")
    else:
        bad(
            name,
            "neither --kv-reshard-vectors nor --enable-vram-dial. act is "
            "refused at parse time (server_args.py:7784-7793); without one, "
            "every proposal is refused for want of an actuator",
        )


# ---------------------------------------------------------------------------
# C4 -- a stage table with at least one MEASURED non-reference stage.
# ---------------------------------------------------------------------------
#
# WHY THIS CHECK GREW A SECOND INPUT (window shift 2026-08-14).
#
# ``--stage-table`` asks for "a JSON dump of the boot's stage table". NOTHING
# IN THE TREE PRODUCES THAT FILE: the only occurrence of the string outside
# this module is this module's own argparse, and ``server_args`` has no
# stage-table dump flag. So C4 -- the check that implements the runsheet's P4,
# the one STOP condition that would have ended this window before the cards
# were claimed -- could only ever SKIP.
#
# It duly skipped, the window was claimed, and the blocker was found on metal
# instead: the boot prints
#
#   REGIME-OBSERVE stage table: 1 stage(s), 1 reachable at runtime,
#                               0 flip target(s), booted on 'booted'
#
# and every round after it prints "would flip to nothing -- already on stage
# 'booted'". The runtime had been saying it in plain text all along; no
# desk-side check could read it.
#
# So C4 now also accepts the artefact that DOES exist -- the boot log -- and
# is red against the real one. A check whose only input cannot be produced is
# indistinguishable from no check at all, which is the same lesson as an
# instrument that cannot fail.
_STAGE_TABLE_RE = re.compile(
    r"stage table: (\d+) stage\(s\), (\d+) reachable at runtime, "
    r"(\d+) flip target\(s\)"
)
#: NOT anchored at line start: the runtime prefixes every line with
#: "[<ts> TP<n>] REGIME-OBSERVE   ". An anchored version matched nothing on
#: either a real log or the smoke fixture, and because the zero-flip-target
#: branch fires first it still LOOKED right -- the row parse was dead code
#: certifying nothing. Requiring "gain=" keeps it off the "stage table:" line.
_STAGE_ROW_RE = re.compile(
    r"stage (\S+) \[(\w+)\].*?gain=([+-][\d.]+)%/band=([\d.]+)%\s+flip=([\d.]+)s",
    re.MULTILINE,
)
_FEED_NOTE_RE = re.compile(r"stage feed: (.+)$", re.MULTILINE)
_NO_TABLE_RE = re.compile(r"could not build the boot stage table")


def check_stage_table_from_boot_log(path: str) -> None:
    """Answer C4 from a boot log, which is an artefact that actually exists."""
    name = "stage-table"
    p = Path(os.path.expanduser(path))
    if not p.is_file():
        bad(name, f"{p} does not exist")
        return
    text = p.read_text(errors="replace")

    if _NO_TABLE_RE.search(text):
        bad(
            name,
            f"{p} says the boot could NOT build a stage table at all "
            "('running without one ... nothing could be selected even in act "
            "mode'). The axis has nothing to propose",
        )
        return

    tables = _STAGE_TABLE_RE.findall(text)
    if not tables:
        bad(
            name,
            f"{p} carries no 'REGIME-OBSERVE stage table:' line. Either the "
            "controller was off, or this is not a boot log",
        )
        return

    # Every rank prints the same table; take the first and report the spread
    # only if the ranks disagree, which would itself be a finding.
    stages, reachable, targets = (int(x) for x in tables[0])
    if len({t for t in tables}) != 1:
        bad(name, f"{p}: ranks printed DIFFERENT stage tables: {sorted(set(tables))}")
        return

    notes = sorted({m.group(1).strip() for m in _FEED_NOTE_RE.finditer(text)})
    rows = _STAGE_ROW_RE.findall(text)
    measured = [
        r
        for r in rows
        if r[0] != "booted" and (float(r[2]) != 0.0 or float(r[4]) != 0.0)
    ]

    if targets > 0 and measured:
        ok(
            name,
            f"{stages} stage(s), {targets} flip target(s), "
            f"{len(measured)} carrying a measured gain; the axis has "
            "something it is allowed to propose",
        )
        return

    why = (
        f"{p}: the boot's OWN stage table reports {stages} stage(s), "
        f"{reachable} reachable, {targets} flip target(s). "
    )
    if targets == 0:
        why += (
            "With zero flip targets the stage clock can never propose a move, "
            "so 'proposals > 0 and actuations > 0' (window criterion A1) is "
            "unreachable BY CONSTRUCTION and an act arm is an expensive "
            "observe. "
        )
    if notes:
        why += "The boot named the cause: " + "; ".join(notes[:3])
    bad(name, why)


def check_stage_table(path: str | None) -> None:
    name = "stage-table"
    if not path:
        skip(name, "no --stage-table given; cannot confirm a measured stage exists")
        return
    p = Path(os.path.expanduser(path))
    if not p.is_file():
        bad(name, f"{p} does not exist")
        return
    try:
        data = json.loads(p.read_text())
    except ValueError as exc:
        bad(name, f"{p} is not readable JSON ({exc})")
        return

    stages = data.get("stages") if isinstance(data, dict) else data
    if not isinstance(stages, list) or not stages:
        bad(name, f"{p} carries no stage list")
        return

    measured = [
        s
        for s in stages
        if isinstance(s, dict)
        and not s.get("unmeasured", False)
        and not s.get("reference", False)
    ]
    if measured:
        ok(
            name,
            f"{len(measured)} measured non-reference stage(s) of {len(stages)}; "
            "the axis has something it is allowed to propose",
        )
    else:
        bad(
            name,
            f"all {len(stages)} stages are unmeasured or reference-only. The "
            "axis never proposes an UNMEASURED stage, so the window would run "
            "clean and prove nothing. This is #584's measurement pass, not "
            "this ticket (TICKET P1)",
        )


# ---------------------------------------------------------------------------
# C5 -- transient census present for the stages the window will visit.
# ---------------------------------------------------------------------------
def check_transient_census(census_dir: str | None, stages: list[str]) -> None:
    name = "transient-census"
    if not census_dir:
        skip(name, "no --census-dir given; flips would be REFUSED for want of a price")
        return
    d = Path(os.path.expanduser(census_dir))
    if not d.is_dir():
        bad(name, f"{d} is not a directory; no census was written")
        return
    files = list(d.glob("*.json"))
    if not files:
        bad(
            name,
            f"{d} holds no census JSON. An unpriced transient reads as free "
            "memory, so every flip is refused by name (TICKET P3)",
        )
        return
    if stages:
        blob = " ".join(f.name for f in files)
        absent = [s for s in stages if s not in blob]
        if absent:
            bad(
                name,
                f"{len(files)} census file(s) but nothing named for: "
                f"{', '.join(absent)}. A PARTIAL set is refused loudly by "
                "pp_cut_calibration, and that refusal is correct",
            )
            return
    ok(name, f"{len(files)} census file(s) under {d}")


# ---------------------------------------------------------------------------
# smoke: drive every check red against fabricated fixtures.
# ---------------------------------------------------------------------------
def smoke() -> int:
    global FAILURES, SKIPS
    red = 0
    total = 0
    print("== smoke: each case below MUST go red ==")

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)

        # 1. gate with only three items -> closed, and names the fourth
        total += 1
        ev = tmpd / "three.json"
        ev.write_text(
            json.dumps({k: {"passed": True, "source": "smoke"} for k in GATE_KEYS[:3]})
        )
        FAILURES = 0
        check_entry_gate(str(ev))
        if FAILURES:
            print("   red as required (gate closed on 3 of 4)")
            red += 1
        else:
            print("   GREEN -- check is broken")

        # 2. passed:true with no source -> malformed, refused
        total += 1
        ev2 = tmpd / "nosource.json"
        ev2.write_text(json.dumps({k: {"passed": True} for k in GATE_KEYS}))
        FAILURES = 0
        check_entry_gate(str(ev2))
        if FAILURES:
            print("   red as required (unattributed pass refused)")
            red += 1
        else:
            print("   GREEN -- an unattributed pass was accepted")

        # 3. no actuator wired
        total += 1
        FAILURES = 0
        check_actuator(None, False)
        if FAILURES:
            print("   red as required (no actuator)")
            red += 1
        else:
            print("   GREEN -- check is broken")

        # 4. stage table with every stage unmeasured
        total += 1
        st = tmpd / "stages.json"
        st.write_text(
            json.dumps(
                {
                    "stages": [
                        {"name": "a", "unmeasured": True},
                        {"name": "ref", "reference": True},
                    ]
                }
            )
        )
        FAILURES = 0
        check_stage_table(str(st))
        if FAILURES:
            print("   red as required (no measured stage)")
            red += 1
        else:
            print("   GREEN -- check is broken")

        # 4b. the SAME question asked of a boot log, which is the artefact
        #     that exists. Fixture is the real shape the runtime prints.
        total += 1
        bl = tmpd / "boot0.log"
        bl.write_text(
            "[TP0] REGIME-OBSERVE stage feed: prefill_heavy: the planner could "
            "not solve 'enc' (PlannerFeedUnavailable('no card probe on disk'))\n"
            "[TP0] REGIME-OBSERVE stage table: 1 stage(s), 1 reachable at "
            "runtime, 0 flip target(s), booted on 'booted'\n"
            "[TP0] REGIME-OBSERVE   stage booted [mixed] weights=auto "
            "kv=[30, 17, 17] pool=320640 gain=+0.0%/band=0.0% flip=0.00s -- "
            "the booted configuration\n"
        )
        FAILURES = 0
        check_stage_table_from_boot_log(str(bl))
        if FAILURES:
            print("   red as required (boot log with 0 flip targets)")
            red += 1
        else:
            print("   GREEN -- check is broken")

        # 4c. and it must go GREEN on a table that DOES carry a measured
        #     non-reference stage, or it is a wall rather than a check.
        total += 1
        bl2 = tmpd / "boot1.log"
        bl2.write_text(
            "[TP0] REGIME-OBSERVE stage table: 2 stage(s), 2 reachable at "
            "runtime, 1 flip target(s), booted on 'booted'\n"
            "[TP0] REGIME-OBSERVE   stage booted [mixed] gain=+0.0%/band=0.0% "
            "flip=0.00s -- the booted configuration\n"
            "[TP0] REGIME-OBSERVE   stage enc-heavy [prefill_heavy] "
            "gain=+6.4%/band=1.1% flip=1.80s -- measured\n"
        )
        FAILURES = 0
        check_stage_table_from_boot_log(str(bl2))
        if FAILURES == 0:
            print("   green as required (a measured flip target opens it)")
            red += 1
        else:
            print("   RED -- a usable stage table was refused")

        # 5. census dir empty
        total += 1
        cd = tmpd / "census"
        cd.mkdir()
        FAILURES = 0
        check_transient_census(str(cd), [])
        if FAILURES:
            print("   red as required (no census written)")
            red += 1
        else:
            print("   GREEN -- check is broken")

        # 6. a COMPLETE gate must go green, or the tool is just a wall
        total += 1
        ev3 = tmpd / "full.json"
        ev3.write_text(
            json.dumps(
                {k: {"passed": True, "source": "smoke fixture"} for k in GATE_KEYS}
            )
        )
        FAILURES = 0
        check_entry_gate(str(ev3))
        if FAILURES == 0:
            print("   green as required (complete gate opens)")
            red += 1
        else:
            print("   RED -- a complete evidence file was refused")

    print(f"== smoke: {red}/{total} cases behaved as required ==")
    FAILURES = 0
    SKIPS = 0
    return 0 if red == total else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="CPU-only preflight for the #363 stage-clock window."
    )
    ap.add_argument(
        "--evidence", help="path the boot will pass to --regime-gate-evidence"
    )
    ap.add_argument("--stage-table", help="JSON dump of the boot's stage table")
    ap.add_argument(
        "--boot-log",
        help=(
            "boot log of a previous boot, read for its own "
            "'REGIME-OBSERVE stage table:' line. Use this when there is no "
            "--stage-table JSON -- nothing in the tree writes one"
        ),
    )
    ap.add_argument(
        "--census-dir", help="SGLANG_RESIDENCY_CENSUS_DIR of the census run"
    )
    ap.add_argument(
        "--stage",
        action="append",
        default=[],
        help="stage name the window will visit; repeatable",
    )
    ap.add_argument("--kv-reshard-vectors", help="the value the boot will pass")
    ap.add_argument("--enable-vram-dial", action="store_true")
    ap.add_argument("--strict", action="store_true", help="treat SKIP as failure")
    ap.add_argument("--smoke", action="store_true", help="prove every check can fail")
    args = ap.parse_args(argv)

    if args.smoke:
        return smoke()

    check_flags_parse()
    check_entry_gate(args.evidence)
    check_actuator(args.kv_reshard_vectors, args.enable_vram_dial)
    if args.boot_log:
        check_stage_table_from_boot_log(args.boot_log)
    else:
        check_stage_table(args.stage_table)
    check_transient_census(args.census_dir, args.stage)

    if FAILURES:
        print(f"-- {FAILURES} check(s) FAILED; do not claim the window yet")
        return 1
    if args.strict and SKIPS:
        print(
            f"-- {SKIPS} check(s) SKIPPED under --strict; an unrunnable check is not a pass"
        )
        return 1
    print(f"-- preflight clear{f' ({SKIPS} skipped)' if SKIPS else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
