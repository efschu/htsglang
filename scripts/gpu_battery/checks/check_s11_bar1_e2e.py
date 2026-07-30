#!/usr/bin/env python3
"""s11 check -- did the standard run really go over BAR1, and did it answer?

Order matters here, because the first thing that is wrong is the thing worth
reporting:

  1. host reachable, BAR1 code in the worktree under test (STOP -- nothing was
     measured),
  2. the graph gate. bar1_graph_check.py must have run and every GATE case must
     have passed. SGLANG_HTCCL_GRAPH_FREIGABE=1 without that evidence produces
     numbers from an operating point nobody can defend,
  3. was a log harvested at ALL. An empty evidence list means one of two very
     different things -- nobody looked, or nothing was there -- and only the
     second is a measurement. `log_quellen` names the files that existed,
  4. THE BOLT. htccl._select raises instead of quietly dropping to the
     host-staged gloo level during a graph capture. If it fired, that is the
     coverage gap of the current integration -- reported with op and size, in
     its own wording, so nobody has to open a log to tell it from a crash,
  5. a fatal in the log. This sits HERE and not at the end, deliberately: a
     boot that died takes the smoke request, the setup lines and everything
     else down with it, and reporting one of those instead means reporting the
     consequence while the cause is two fields away. That is exactly how a run
     whose capture aborted got reported as "no ERREICHT line in the log",
  6. per-group attainment. EVERY communicator group must report ACHIEVED=bar1.
     The requested transport name says bar1 either way; it is not evidence. One
     group on bar1 and one on gloo is a MIXED run: correct-looking, and its
     numbers may not be reported as bar1 numbers,
  7. a BAR1 setup line per group. The ERREICHT line is written at group init;
     the setup line is written when the aperture actually handed over the
     space,
  8. the smoke request over /generate: coherent output (the continuation of
     "1 2 3 4", counted, not judged) and spec_accept_length present as a
     number. Three outcomes, not two -- "coherent text that never got to the
     numbers because the token budget went elsewhere" is its own verdict
     (UNTER-PROVISIONIERT) and says nothing about the transport.

NOT judged: the HEIGHT of spec_accept_length, the setup duration, and any
throughput. s11 answers "does the direct path carry a real run"; s12 answers
"what does it cost".
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_common import (  # noqa: E402
    CheckFail,
    CheckStop,
    classify_missing_result,
    is_number,
    load_json,
    require_envelope,
    run_check,
)

STEP = "s11_bar1_e2e"
KIND = "bar1_e2e"

# With SGLANG_UNEVEN_DCP=1 the standard run builds two communicator groups. The
# names are prefixes because the counter suffix (tp:0, dcp:0) is an
# implementation detail of _get_unique_name.
REQUIRED_GROUP_PREFIXES = ("tp", "dcp")
MIN_ORDERED_NUMBERS = 15


def _group_for(prefix: str, groups: list) -> list:
    return [g for g in groups if str(g.get("group", "")).startswith(prefix)]


def check(step_dir: str) -> None:
    path = os.path.join(step_dir, "bar1_e2e.json")
    classify_missing_result(step_dir, "bar1_e2e", path, "bar1_e2e.json")
    payload = load_json(path, "bar1_e2e.json")
    require_envelope(payload, KIND, "bar1_e2e.json", 5)

    if not payload.get("reachable"):
        raise CheckStop(
            f"host {payload.get('host') or '?'} not reachable -- the run never "
            "happened"
        )
    if not payload.get("integration_present"):
        raise CheckStop(
            "the BAR1 integration is not in the worktree under test "
            "(htccl_bar1.py / benchmark/bar1_graph_check.py) -- set BAR1_HOST_WT"
        )
    # "blockiert" stays: an out-of-scope unit test matches the verdict on it.
    if payload.get("blocked"):
        raise CheckStop(f"Schritt blockiert: {payload['blocked']}")

    gate = payload.get("graph_check") or {}
    if not gate.get("gate_cases"):
        raise CheckStop(
            "bar1_graph_check reported no gate case at all (rc="
            f"{gate.get('rc')!r}) -- the gate never ran, so "
            "SGLANG_HTCCL_GRAPH_FREIGABE=1 is unsupported"
        )
    if not gate.get("alle_bestanden"):
        raise CheckFail(
            f"graph gate fell: {','.join(gate.get('gefallen') or []) or '?'} "
            f"(rc={gate.get('rc')!r}) -- the direct path is not cleared under "
            "CUDA graphs"
        )

    # First the question of WHETHER anybody looked. An empty evidence list out
    # of a step directory with no log excerpt is not a finding about the run
    # but one about the harvest -- and telling those two apart is the
    # difference between "the run reported nothing" and "we did not look".
    if "log_quellen" not in payload:
        raise CheckStop(
            "bar1_e2e.json names no 'log_quellen' -- the artifact comes from an "
            "older producer that read the evidence out of htccl_lines.txt alone "
            "and came up empty on every abort path"
        )
    if not payload["log_quellen"]:
        raise CheckStop(
            "no log excerpt in the step directory (neither htccl_lines.txt nor "
            "server.log) -- niemand hat geschaut, so there is nothing to decide "
            "here"
        )

    bolt = payload.get("riegel")
    if bolt:
        raise CheckFail(
            f"RIEGEL: htccl._select aborted {bolt.get('op')!r} with "
            f"{bolt.get('bytes')} bytes during a CUDA graph capture, because "
            "bar1 does not carry the operation at that size -- a coverage gap "
            "in the BAR1 transport, not a crash. This is exactly the scenario "
            "the BAR1 integration is accepted on."
        )

    # And right after it the crash. A dead boot takes the smoke request, the
    # setup lines and the ERREICHT lines with it -- reporting one of those
    # reports the consequence and leaves the cause standing two fields away.
    fatal = payload.get("fatal")
    if fatal:
        raise CheckFail(f"Fatal im Serverlog -- {fatal}")

    groups = payload.get("gruppen") or []
    if not groups:
        raise CheckFail(
            "not a single 'ACHIEVED=' line in the log (sources: "
            f"{payload['log_quellen']}, {payload.get('log_zeilen')} lines) -- "
            "without it the arm of the measurement is unsupported (the "
            "requested transport name reads bar1 even on a fallback)"
        )
    for prefix in REQUIRED_GROUP_PREFIXES:
        found = _group_for(prefix, groups)
        if not found:
            raise CheckFail(
                f"no group {prefix!r} in the log (reported: "
                f"{[g.get('group') for g in groups]}) -- with "
                "SGLANG_UNEVEN_DCP=1 both tp and dcp have to show up"
            )
        not_bar1 = [g for g in found if g.get("achieved") != "bar1"]
        if not_bar1:
            g = not_bar1[0]
            raise CheckFail(
                f"group {g.get('group')!r} runs ACHIEVED={g.get('achieved')!r} "
                f"(requested {g.get('requested')!r}) while "
                f"{payload.get('gruppen_bar1')} run on bar1 -- ein gemischter "
                "Lauf, whose numbers do not count as bar1 numbers"
            )

    ledger_groups = payload.get("aufbau_gruppen") or []
    if not payload.get("aufbau_lines"):
        raise CheckFail(
            "no 'HTCCL-BAR1: setup in' line -- no rank actually built a BAR1 "
            "region"
        )
    for prefix in REQUIRED_GROUP_PREFIXES:
        if not any(str(name).startswith(prefix) for name in ledger_groups):
            raise CheckFail(
                f"no BAR1 setup evidenced for group {prefix!r} (built: "
                f"{ledger_groups}) -- the ACHIEVED line on its own is written "
                "before the aperture has committed"
            )

    smoke = payload.get("smoke") or {}
    if not smoke.get("vorhanden"):
        raise CheckFail("no smoke request was sent (smoke.json missing)")
    if smoke.get("error"):
        raise CheckFail(f"smoke request reports: {smoke['error']}")
    if smoke.get("endpunkt") == "chat":
        raise CheckStop(
            "the smoke came over /v1/chat/completions -- there the output "
            "hangs on the chat template (thinking preamble) and meta_info is "
            "opt-in, so neither the coherence nor spec_accept_length says "
            "anything. The artifact comes from a run before the switch to "
            "/generate; it is to be repeated, not judged"
        )
    # THE NAMED STATE. Coherent text, token budget spent to the stop, the
    # numbers never got their turn. On 2026-07-30 that was a thinking preamble
    # at temperature 0 -- an under-provisioned smoke, not a transport fault and
    # not corruption. Its own verdict, so it is never read as a bar1 finding.
    if smoke.get("unterprovisioniert"):
        raise CheckFail(
            f"smoke UNTER-PROVISIONIERT: the answer is coherent but spent the "
            f"token budget (finish_reason={smoke.get('finish_reason')!r}) "
            f"before the numbers got their turn -- "
            f"{smoke.get('zahlen_in_folge')} of {smoke.get('zahlen_erwartet')}. "
            f"This is a finding about the smoke, NICHT ueber den Transport: "
            f"the collectives are byte-proven, and a change of trajectory at "
            f"temperature 0 is in the expected numerics class. Start: "
            f"{str(smoke.get('content_prefix'))[:80]!r}"
        )
    if not smoke.get("kohaerent"):
        raise CheckFail(
            f"smoke output inkohaerent: only {smoke.get('zahlen_in_folge')} of "
            f"{smoke.get('zahlen_erwartet')} numbers in order (at least "
            f"{MIN_ORDERED_NUMBERS}); start: "
            f"{str(smoke.get('content_prefix'))[:80]!r}"
        )
    if not is_number(smoke.get("spec_accept_length")):
        raise CheckFail(
            f"spec_accept_length is {smoke.get('spec_accept_length')!r} -- the "
            "spec path is not running, or the answer carries no meta_info"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    args = ap.parse_args()
    return run_check(STEP, lambda: check(args.step_dir))


if __name__ == "__main__":
    sys.exit(main())
