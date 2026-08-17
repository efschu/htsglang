"""The D2 injector must be able to FAIL. These tests are the proof.

The #713(a) acceptance passed on every criterion and then the control build
passed them too, because no criterion had ever been shown capable of failing.
This harness is the replacement, so the thing that has to be pinned is not
that it detects the defect -- it is that it reports NOT AS NAMED when the
defect is absent, and when the arm is the other one.

Every fixture assertion below is anchored on a RECORDED log excerpt
(scripts/fixtures/d2_injector_pingpong_excerpt.txt, taken verbatim from
evidence-665-f1/boot_bundle.log.20260817T070900Z, 2026-08-17 06:53-06:54Z),
so a change to the parser that stops seeing the real defect breaks the suite.
"""

import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "d2_phase_locked_injector.py"
FIXTURE = REPO / "scripts" / "fixtures" / "d2_injector_pingpong_excerpt.txt"


def _load():
    spec = importlib.util.spec_from_file_location("d2_injector", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # REGISTER BEFORE EXEC. The script uses `from __future__ import
    # annotations`, so its dataclass field annotations stay strings, and
    # dataclasses resolves them through sys.modules[cls.__module__]. Without
    # this line that lookup returns None and every test errors in the loader
    # rather than in the code under test.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def inj():
    return _load()


@pytest.fixture(scope="module")
def markers(inj):
    with open(FIXTURE, "r", errors="replace") as fh:
        return inj.parse_markers(fh)


def test_import_smoke():
    """The script must import with no server, no GPU and no side effects."""
    mod = _load()
    assert hasattr(mod, "parse_markers")
    assert hasattr(mod, "verdict")


def test_parser_reads_every_marker_class(markers):
    """A silent parser regression is the failure mode this file exists for."""
    for key in ("arms", "cutovers", "dones", "holds", "extents", "carries", "prefills"):
        assert markers[key], f"parsed nothing for {key}"


def test_only_origin_rank_arms_are_counted(inj):
    """Counting all three ranks would triple every run length.

    WRITTEN AGAINST SYNTHETIC LINES ON PURPOSE. The first version asserted
    this over the recorded fixture, where every arming line happens to be
    PP0 already -- so deleting the origin-rank guard entirely left the whole
    suite green. A test that cannot fail proves nothing, which is the same
    defect this harness was built to stop shipping. Here the non-origin arm
    is present in the input, so dropping the guard changes the count.
    """
    lines = [
        "[2026-08-17 06:00:00 PP0] PHASE-POLICY arming tp_to_pp: IDLE-LOCKED: x "
        "(0 req resident, 22 tok prefill pending) and the target layout can run one",
        "[2026-08-17 06:00:00 PP1] PHASE-POLICY arming tp_to_pp: IDLE-LOCKED: x "
        "(0 req resident, 22 tok prefill pending) and the target layout can run one",
        "[2026-08-17 06:00:00 PP2] PHASE-POLICY arming tp_to_pp: IDLE-LOCKED: x "
        "(0 req resident, 22 tok prefill pending) and the target layout can run one",
    ]
    arms = inj.parse_markers(lines)["arms"]
    assert len(arms) == 1, "only the origin rank may contribute an arm"
    assert arms[0].rank == inj.ORIGIN_RANK


def test_recorded_pingpong_is_detected(inj, markers):
    """The known 12-arm / 31 s run at 06:53:56 must be found."""
    runs = inj.alternating_runs(markers["arms"])
    assert len(runs) == 1
    run = runs[0]
    assert run.length == 12
    assert 25.0 <= run.seconds <= 40.0
    assert run.arms[0].ts == "2026-08-17 06:53:56"


def test_signature_requires_both_mirrored_halves(inj, markers):
    """Alternation alone is not the defect; the state disagreement is.

    Layouts may legitimately alternate under changing load. What identifies
    THIS defect is that the tp side reports the request as not resident with
    tokens pending while the pp side reports it resident with none.
    """
    run = inj.alternating_runs(markers["arms"])[0]
    assert run.signature
    assert any(a.mirrored_tp for a in run.arms)
    assert any(a.mirrored_pp for a in run.arms)


def test_signature_false_when_only_one_half_present(inj):
    """Can-fail for the signature rule itself, on synthetic arms."""
    arms = [
        inj.Arm("2026-08-17 06:00:00", "PP0", "tp_to_pp", 0, 22),
        inj.Arm("2026-08-17 06:00:03", "PP0", "pp_to_tp", 2, 55),  # not mirrored
        inj.Arm("2026-08-17 06:00:06", "PP0", "tp_to_pp", 0, 22),
    ]
    run = inj.alternating_runs(arms)[0]
    assert run.length == 3
    assert not run.signature, "signature must need BOTH halves, not just alternation"


def test_stuck_pending_is_constant_in_this_run(inj, markers):
    """The recorded run holds one request, so pending must not grow.

    Recorded elsewhere (06:12 rotation) a run DOES grow 81 -> 639 -> 1078; the
    two shapes are different and the report prints the sequence rather than a
    single number so they stay distinguishable.
    """
    run = inj.alternating_runs(markers["arms"])[0]
    seq = [a.pending for a in run.arms if a.direction == "tp_to_pp"]
    assert seq and len(set(seq)) == 1 and seq[0] == 22


def test_half_cycle_matches_the_ttft_quantisation_step(inj, markers):
    """The point of the whole finding: the step IS the half-cycle.

    The #713 tables quantised at ~3.1 s and ~5.9 s. If the measured half-cycle
    stopped matching that, the explanation would need revisiting, so it is
    pinned here rather than left in prose.
    """
    runs = inj.alternating_runs(markers["arms"])
    hc = inj.half_cycles(runs)
    assert hc
    assert 2.0 <= (sum(hc) / len(hc)) <= 4.5


def test_control_arm_passes_on_the_recorded_defect(inj, markers):
    res = inj.report(markers, "test")
    ok, why = inj.verdict(res, "control")
    assert ok, why


def test_fixed_arm_FAILS_on_the_recorded_defect(inj, markers):
    """Can-fail #1: the same evidence must condemn a build claiming the fix."""
    res = inj.report(markers, "test")
    ok, why = inj.verdict(res, "fixed")
    assert not ok
    assert "signature run" in why


def test_control_arm_FAILS_when_nothing_ping_pongs(inj):
    """Can-fail #2: a quiet instance must not be read as a reproduction.

    This is the exact trap the #713 acceptance fell into -- a criterion that
    passes when the defect is absent. Here absence must read as NOT AS NAMED,
    and the message must say the trigger model may be wrong rather than
    inviting a retune.
    """
    quiet = inj.report({"arms": [], "dones": []}, "quiet")
    ok, why = inj.verdict(quiet, "control")
    assert not ok
    assert "NO SIGNATURE RUN" in why
    assert "do not retune" in why


def test_runs_need_opposite_directions_and_proximity(inj):
    """Two same-direction arms, or distant arms, are not a run."""
    same = [
        inj.Arm("2026-08-17 06:00:00", "PP0", "tp_to_pp", 0, 22),
        inj.Arm("2026-08-17 06:00:03", "PP0", "tp_to_pp", 0, 22),
    ]
    assert inj.alternating_runs(same) == []
    distant = [
        inj.Arm("2026-08-17 06:00:00", "PP0", "tp_to_pp", 0, 22),
        inj.Arm("2026-08-17 06:05:00", "PP0", "pp_to_tp", 1, 0),
    ]
    assert inj.alternating_runs(distant) == []


def test_shot_cadence_below_the_run_window_is_refused(inj):
    """The harness must not be allowed to manufacture its own result.

    Runs are opposite-direction arms within RUN_MAX_GAP_S. Since the recorded
    evidence has each small prompt costing roughly one layout round trip,
    shots fired faster than that window would chain into a single long "run"
    that is purely the injector's cadence. The first draft defaulted to 6 s
    against an 8 s window and would have done exactly that.
    """
    assert inj._check_gap(inj.RUN_MAX_GAP_S - 0.1), "must refuse a sub-window gap"
    assert inj._check_gap(inj.RUN_MAX_GAP_S), "must refuse a gap equal to the window"
    assert inj._check_gap(inj.RUN_MAX_GAP_S + 0.1) == "", "must accept a gap above it"


def test_default_gap_is_above_the_run_window(inj):
    """A safe default matters more than a guard nobody trips.

    Reads the REAL parser default rather than restating it, so lowering the
    default back under the run window fails here instead of silently
    reintroducing the self-manufactured run.
    """
    defaults = {a.dest: a.default for a in inj.build_parser()._actions}
    assert defaults["gap"] > inj.RUN_MAX_GAP_S
    assert inj._check_gap(defaults["gap"]) == ""


def test_per_shot_attribution_windows_arms(inj):
    arms = [
        inj.Arm("2026-08-17 06:00:05", "PP0", "tp_to_pp", 0, 22),
        inj.Arm("2026-08-17 06:00:08", "PP0", "pp_to_tp", 1, 0),
        inj.Arm("2026-08-17 06:01:00", "PP0", "tp_to_pp", 0, 22),
    ]
    lo = inj._epoch("2026-08-17 06:00:04")
    hi = inj._epoch("2026-08-17 06:00:09")
    assert len(inj.arms_between(arms, lo, hi)) == 2
    assert len(inj.arms_between(arms, lo, lo)) == 0


def test_live_mode_refuses_without_a_window_claim(inj, tmp_path, monkeypatch):
    """The live run is a window ticket; import and dry-run must not reach it."""
    monkeypatch.setattr(inj, "ARB_HOLDER", str(tmp_path / "nope"))
    assert inj._guard_live(), "must refuse when no holder file exists"
    holder = tmp_path / "holder"
    holder.write_text("")
    monkeypatch.setattr(inj, "ARB_HOLDER", str(holder))
    assert inj._guard_live(), "must refuse on an empty holder file"
    holder.write_text("662-F4-r4")
    assert inj._guard_live() == "", "must accept a real claim"
