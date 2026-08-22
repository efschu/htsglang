#!/usr/bin/env python3
"""#799 can-fail harness: every new edge must have a mutant that kills a test.

A green suite proves nothing on its own. Each entry below breaks exactly ONE
edge of the new signal path and names the test that must go red for it. An
entry whose test stays green is a test that does not test what it claims --
which is the specific defect this repository keeps finding in itself (a helper
covered, the call site that invokes it not).

Run: python scripts/mutants_799.py
"""

import subprocess
import sys

PY = "/spinning/htsglang-gpu/.venv/bin/python"
ROOT = "/spinning/wt-799-wedge-watchdog"
IC = "python/sglang/srt/managers/scheduler_components/invariant_checker.py"
RN = "python/sglang/srt/turnkey/runner.py"
WD = "python/sglang/srt/turnkey/watchdog.py"
WS = "python/sglang/srt/managers/wedge_status.py"
MN = "python/sglang/srt/turnkey/__main__.py"

T_WD = "test/registered/unit/turnkey/test_wedge_signal_799.py"
T_WS = "test/registered/unit/managers/test_wedge_status_799.py"

MUTANTS = [
    # (name, file, old, new, test target that MUST fail)
    (
        "M1 runner.tick stops carrying the verdict (the pre-#799 code)",
        RN,
        "        obs = W.Observation(port_open=d.port_probe(), api_ok=api.ok,\n"
        "                            generation=None, wedged=wedged)",
        "        obs = W.Observation(port_open=d.port_probe(), api_ok=api.ok,\n"
        "                            generation=None, wedged=None)",
        f"{T_WD}::TestRunnerCallEdge::"
        "test_tick_restarts_when_the_scheduler_reports_a_wedge",
    ),
    (
        "M2 the detector stops publishing its verdict",
        IC,
        "        _publish(alarm, detail)\n",
        "",
        f"{T_WS}::TestTheDetectorPublishes::"
        "test_the_production_poller_publishes_a_wedge",
    ),
    (
        "M3 the state machine ignores a published wedge",
        WD,
        "    if policy.wedge_signal_enabled and obs.wedged is not None:",
        "    if False and obs.wedged is not None:",
        f"{T_WD}::TestWedgeSignalConvicts::"
        "test_confirmations_convict_and_restart",
    ),
    (
        "M4 FALSE-ALARM DIRECTION: convict on a CLEAN verdict",
        WD,
        "        if obs.wedged:",
        "        if not obs.wedged:",
        f"{T_WD}::TestRunnerCallEdge::"
        "test_tick_never_restarts_a_lane_that_reports_no_wedge",
    ),
    (
        "M5 the operator stop marker is ignored",
        RN,
        "        stop = d.operator_stop()",
        "        stop = None",
        f"{T_WD}::TestOperatorStop::test_the_marker_suspends_every_restart",
    ),
    (
        "M6 the CLI drops the config key on the floor",
        MN,
        "                    wedge_signal_enabled=w.wedge_signal_enabled,\n",
        "",
        f"{T_WD}::TestConfigReachesTheRunner::"
        "test_cmd_watch_hands_both_settings_to_the_runner",
    ),
    (
        "M7 a stale verdict is believed",
        WS,
        "        if age > float(stale_after_s):",
        "        if False:",
        f"{T_WS}::TestPublishAndRead::test_a_stale_verdict_is_not_a_verdict",
    ),
    (
        "M8 the conviction threshold collapses to one report",
        WD,
        "            if hits < policy.wedge_confirmations:",
        "            if hits < 1:",
        f"{T_WD}::TestWedgeSignalConvicts::"
        "test_one_wedge_report_is_not_a_conviction",
    ),
    (
        "M9 the drift veto is never consulted",
        RN,
        "            drift = self.deps.restart_drift()",
        "            drift = None",
        f"{T_WD}::TestRestartTargetDrift::"
        "test_a_drifted_lane_alarms_but_never_restarts",
    ),
    (
        "M10 a refused restart still spends the budget",
        RN,
        "                decision = W.Decision(state=before, action=W.ACT_ALARM,",
        "                decision = W.Decision(state=decision.state, "
        "action=W.ACT_ALARM,",
        f"{T_WD}::TestRestartTargetDrift::"
        "test_a_refused_restart_does_not_spend_the_restart_budget",
    ),
    (
        "M11 drift is decided on missing evidence",
        RN,
        "    if not configured or not booted:\n        return None",
        "    if configured is None and booted is None:\n        return None",
        f"{T_WD}::TestRestartTargetDrift::"
        "test_an_unknowable_comparison_is_not_drift",
    ),
    (
        "M12 the detector thread becomes unstoppable again",
        IC,
        "            if stop is not None and stop.is_set():",
        "            if False:",
        f"{T_WS}::TestTheThreadRunsThePoll::"
        "test_the_watchdog_thread_actually_calls_the_poller",
    ),
]


def run(target):
    r = subprocess.run(
        [PY, "-m", "pytest", target, "-q", "--no-header", "-x"],
        cwd=ROOT, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "",
             "PYTHONPATH": f"{ROOT}/python", "HOME": "/root"},
        timeout=600)
    return r.returncode == 0


def main():
    failures = []
    for name, path, old, new, target in MUTANTS:
        full = f"{ROOT}/{path}"
        with open(full) as fh:
            src = fh.read()
        if src.count(old) != 1:
            print(f"SKIP-BROKEN {name}: anchor appears {src.count(old)}x")
            failures.append(name)
            continue
        with open(full, "w") as fh:
            fh.write(src.replace(old, new, 1))
        try:
            still_green = run(target)
        finally:
            with open(full, "w") as fh:
                fh.write(src)
        verdict = "SURVIVED (test is decorative)" if still_green else "KILLED"
        print(f"{verdict:30s} {name}")
        if still_green:
            failures.append(name)
    print()
    if failures:
        print(f"CAN-FAIL PROOF INCOMPLETE: {len(failures)} mutant(s) survived")
        return 1
    print(f"CAN-FAIL PROOF COMPLETE: all {len(MUTANTS)} mutants killed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
