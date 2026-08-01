"""#363 phase 3 (desk half) -- boot stage table, the flag, and the act path.

Hermetic: no GPU, no server, stubbed actuators. What is pinned here is the
machinery behind an off flag; ENABLING it stays gated on the card evidence of
DESIGN_363 section 11.7, and the gate itself is one of the things under test.

The falsifier duty of this slice is symmetric and both halves are here:

* the gate must CLOSE -- ``act`` is refused at parse time until the evidence
  exists, and the refusal names every missing item;
* the gate must OPEN -- with a complete evidence file the act path fires and
  reaches the stubbed actuators. A gate that cannot open is as wrong as one
  that cannot close: it hides a broken acting path behind a permanent refusal.

And the phase-2 property has to survive the arrival of the action path:
observe still cannot reach #297 or #330, asserted on the import graph rather
than promised.
"""

import json
import os
import tempfile
import types
import unittest

from sglang.srt.managers.regime_act import (
    ARM_SOURCE,
    RegimeActuator,
)
from sglang.srt.managers.regime_classifier import (
    REGIME_DECODE_HEAVY,
    REGIME_KV_PRESSURE,
    REGIME_PREFILL_HEAVY,
    DwellGate,
    RegimeError,
    Stage,
)
from sglang.srt.managers.regime_runtime import (
    MODE_ACT,
    MODE_OBSERVE,
    MODE_OFF,
    MODES,
    RegimeObserver,
    resolve_mode,
)
from sglang.srt.managers.regime_stages import (
    GATE_ITEMS,
    REACH_BOOTED,
    REACH_NO_WEIGHT_MOVER,
    REACH_RESHARD,
    REACH_UNDECLARED_VECTOR,
    REACH_VRAM_DIAL,
    EntryGate,
    build_stage_table,
    load_gate_evidence,
    planner_candidates,
    reachability,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


_FLOOR = 4.2
_BOOT_VRAM = (29607, 17780, 17780)


def _stage(
    name,
    regime,
    *,
    weights=None,
    kv=(7, 3, 3),
    vram=_BOOT_VRAM,
    pool=453_632,
    gain=0.0,
    band=_FLOOR,
    flip=0.3,
):
    return Stage(
        name=name,
        regime=regime,
        weight_vector=weights,
        kv_token_vector=tuple(kv),
        vram_budget_mib=tuple(vram),
        max_total_num_tokens=pool,
        measured_gain_pct=gain,
        measured_band_pct=band,
        flip_cost_s=flip,
    )


BOOTED = _stage("fp8-decode", REGIME_DECODE_HEAVY)
#: Reachable: same weights, a different DECLARED KV vector.
KV_STAGE = _stage(
    "fp8-prefill-kv",
    REGIME_PREFILL_HEAVY,
    kv=(2, 11, 10),
    pool=380_000,
    gain=12.0,
    flip=0.4,
)
#: Not reachable: the weight cut has no runtime actuator (#354/#357).
WEIGHT_STAGE = _stage(
    "fp8-prefill-weights",
    REGIME_PREFILL_HEAVY,
    weights=(16, 1, 1),
    kv=(2, 11, 10),
    pool=96_256,
    gain=22.6,
    flip=4.5,
)
DECLARED = ((2, 11, 10),)


# ---------------------------------------------------------------------------
# 1. Reachability and the boot table contract
# ---------------------------------------------------------------------------


class TestReachability(CustomTestCase):
    def test_the_booted_stage_is_where_we_are(self):
        code, _why = reachability(BOOTED, BOOTED, DECLARED)
        self.assertEqual(code, REACH_BOOTED)

    def test_a_declared_kv_vector_is_reachable_through_297(self):
        code, why = reachability(BOOTED, KV_STAGE, DECLARED)
        self.assertEqual(code, REACH_RESHARD)
        self.assertIn("#297", why)

    def test_an_undeclared_kv_vector_is_not_reachable(self):
        code, why = reachability(BOOTED, KV_STAGE, declared_vectors=())
        self.assertEqual(code, REACH_UNDECLARED_VECTOR)
        self.assertIn("--kv-reshard-vectors", why)

    def test_a_different_weight_vector_is_never_reachable(self):
        """The measured fact behind the whole reachability concept: the
        MLP/GEMM cut has no runtime actuator, so this stage is real, solved,
        and unreachable from this boot."""
        code, why = reachability(BOOTED, WEIGHT_STAGE, DECLARED)
        self.assertEqual(code, REACH_NO_WEIGHT_MOVER)
        self.assertIn("restart", why)

    def test_the_weight_check_wins_over_the_vector_check(self):
        """A stage differing in BOTH must report the reason a flag cannot
        fix, or the operator adds a vector and tries again for nothing."""
        code, _why = reachability(BOOTED, WEIGHT_STAGE, declared_vectors=())
        self.assertEqual(code, REACH_NO_WEIGHT_MOVER)

    def test_a_budget_only_difference_is_the_dial(self):
        grown = _stage(
            "fp8-decode-wide",
            REGIME_KV_PRESSURE,
            vram=(29607, 18780, 18780),
            gain=6.0,
        )
        code, why = reachability(BOOTED, grown, DECLARED)
        self.assertEqual(code, REACH_VRAM_DIAL)
        self.assertIn("grow", why)


class TestBootStageTable(CustomTestCase):
    def test_the_table_reports_present_with_its_stages_described(self):
        """The phase-2 'table absent' report becomes 'table present, N
        stages' -- and every stage says what this server can do about it."""
        plan = build_stage_table(
            booted=BOOTED, candidates=[KV_STAGE], declared_vectors=DECLARED
        )
        self.assertEqual(len(plan), 2)
        self.assertEqual(len(plan.flip_targets), 1)
        # The booted stage is reachable too -- returning to it is the normal
        # end of a regime episode, not an unreachable configuration.
        self.assertEqual(len(plan.reachable), 2)
        self.assertEqual(plan.booted, "fp8-decode")
        described = " ".join(plan.describe())
        self.assertIn("fp8-prefill-kv", described)
        self.assertIn("#297", described)

    def test_an_unreachable_stage_stays_visible_and_unselectable(self):
        plan = build_stage_table(
            booted=BOOTED, candidates=[WEIGHT_STAGE], declared_vectors=DECLARED
        )
        self.assertEqual(len(plan), 2)
        self.assertEqual(len(plan.flip_targets), 0)
        ok, why = plan.is_selectable("fp8-prefill-weights")
        self.assertFalse(ok)
        self.assertIn("restart", why)
        self.assertIn("no_weight_mover", plan.summary()["unreachable"].values())

    def test_two_stages_for_one_regime_are_refused_at_boot(self):
        """A tie broken by list order is not a decision."""
        other = _stage("second-prefill", REGIME_PREFILL_HEAVY, kv=(3, 3, 7), gain=9.0)
        with self.assertRaises(RegimeError) as cm:
            build_stage_table(
                booted=BOOTED,
                candidates=[KV_STAGE, other],
                declared_vectors=DECLARED,
            )
        self.assertIn("One regime selects one stage", str(cm.exception))

    def test_a_stage_inside_its_own_band_is_refused_at_boot(self):
        """The #360 rule, enforced where the table is built rather than at
        the first select."""
        noisy = _stage(
            "noisy-prefill", REGIME_PREFILL_HEAVY, kv=(2, 11, 10), gain=3.0, band=4.2
        )
        with self.assertRaises(RegimeError) as cm:
            build_stage_table(
                booted=BOOTED, candidates=[noisy], declared_vectors=DECLARED
            )
        self.assertIn("band", str(cm.exception))

    def test_a_candidate_reusing_the_booted_name_is_refused(self):
        clash = _stage("fp8-decode", REGIME_PREFILL_HEAVY, kv=(2, 11, 10), gain=9.0)
        with self.assertRaises(RegimeError):
            build_stage_table(
                booted=BOOTED, candidates=[clash], declared_vectors=DECLARED
            )

    def test_a_table_of_one_is_legal_and_selects_nothing(self):
        """The honest state on a rig with no planner feed."""
        plan = build_stage_table(booted=BOOTED)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan.flip_targets, ())
        self.assertEqual(len(plan.reachable), 1)

    def test_the_admissibility_ceiling_is_noted_per_selectable_stage(self):
        plan = build_stage_table(
            booted=BOOTED, candidates=[KV_STAGE], declared_vectors=DECLARED
        )
        notes = " ".join(plan.notes)
        self.assertIn("selectable up to", notes)
        self.assertIn(str(int(0.85 * KV_STAGE.max_total_num_tokens)), notes)


class TestPlannerFeed(CustomTestCase):
    def test_no_feed_means_the_booted_stage_only_and_says_so(self):
        stages, notes = planner_candidates(types.SimpleNamespace())
        self.assertEqual(stages, [])
        self.assertIn("no planner feed bound", " ".join(notes))

    def test_a_planner_failure_is_a_note_not_a_boot_failure(self):
        def solve(goal):
            raise RuntimeError("no card probe")

        stages, notes = planner_candidates(types.SimpleNamespace(), solve_fn=solve)
        self.assertEqual(stages, [])
        self.assertIn("no card probe", " ".join(notes))

    def test_the_feed_asks_one_goal_per_regime(self):
        asked = []

        def solve(goal):
            asked.append(goal)
            return KV_STAGE if goal == "enc" else None

        stages, _notes = planner_candidates(types.SimpleNamespace(), solve_fn=solve)
        self.assertIn("enc", asked)
        self.assertIn("dec", asked)
        self.assertIn("maxkv", asked)
        self.assertEqual([s.name for s in stages], ["fp8-prefill-kv"])


# ---------------------------------------------------------------------------
# 2. The entry gate -- it must close, and it must open
# ---------------------------------------------------------------------------


def _evidence(**overrides):
    data = {
        item.key: {"passed": True, "source": f"run-{item.key}"} for item in GATE_ITEMS
    }
    data.update(overrides)
    return data


class TestEntryGateCloses(CustomTestCase):
    def test_an_empty_gate_is_closed_and_names_every_item(self):
        gate = EntryGate({})
        self.assertFalse(gate.open)
        text = gate.refusal()
        for item in GATE_ITEMS:
            self.assertIn(item.key, text)
            self.assertIn(item.produced_by, text)

    def test_a_partial_gate_names_only_what_is_missing(self):
        gate = EntryGate(
            {"desyncs_zero": {"passed": True, "source": "observe run 2026-08-01"}}
        )
        self.assertFalse(gate.open)
        text = gate.refusal()
        self.assertIn("[ok]      desyncs_zero", text)
        self.assertIn("[MISSING] f2_live_replay", text)

    def test_a_pass_without_a_source_is_a_claim_not_evidence(self):
        gate = EntryGate(_evidence(f3_bands_measured={"passed": True}))
        self.assertFalse(gate.open)
        self.assertIn("unattributed pass", gate.refusal())

    def test_a_false_pass_is_missing_not_malformed(self):
        gate = EntryGate(_evidence(f4_card_comparison={"passed": False, "source": "x"}))
        self.assertFalse(gate.open)
        self.assertIn("f4_card_comparison", gate.missing)

    def test_a_non_mapping_entry_is_refused_by_shape(self):
        gate = EntryGate(_evidence(desyncs_zero=True))
        self.assertFalse(gate.open)
        self.assertIn("expected a mapping", gate.refusal())

    def test_a_missing_file_is_a_closed_gate_not_an_error(self):
        gate = load_gate_evidence("/nonexistent/regime-gate.json")
        self.assertFalse(gate.open)
        self.assertIn("not found", gate.origin)

    def test_a_declared_but_unparsable_file_is_an_error(self):
        """A typo must not read as 'not measured yet'."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gate.json")
            with open(path, "w") as f:
                f.write("{not json")
            with self.assertRaises(RegimeError) as cm:
                load_gate_evidence(path)
            self.assertIn("measured and unreadable", str(cm.exception))

    def test_a_json_array_is_refused_by_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gate.json")
            with open(path, "w") as f:
                json.dump([1, 2], f)
            with self.assertRaises(RegimeError):
                load_gate_evidence(path)


class TestEntryGateOpens(CustomTestCase):
    """FALSIFIER: a gate that cannot open is as wrong as one that cannot
    close -- it hides a broken acting path behind a permanent refusal."""

    def test_a_complete_evidence_set_opens_the_gate(self):
        gate = EntryGate(_evidence())
        self.assertTrue(gate.open, gate.refusal())
        self.assertEqual(sorted(gate.passed), sorted(i.key for i in GATE_ITEMS))
        self.assertEqual(gate.missing, [])

    def test_a_complete_evidence_file_opens_the_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gate.json")
            with open(path, "w") as f:
                json.dump(_evidence(), f)
            gate = load_gate_evidence(path)
            self.assertTrue(gate.open, gate.refusal())
            self.assertEqual(gate.origin, path)


# ---------------------------------------------------------------------------
# 3. The flag
# ---------------------------------------------------------------------------


def _args(**kw):
    base = dict(
        regime_controller="off",
        regime_gate_evidence=None,
        kv_reshard_vectors=None,
        enable_vram_dial=False,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _validate(args):
    from sglang.srt.server_args import ServerArgs

    ServerArgs._handle_regime_controller(args)


class TestFlagModes(CustomTestCase):
    def test_off_is_the_default_and_validates(self):
        args = _args()
        _validate(args)
        self.assertEqual(args.regime_controller, "off")

    def test_observe_needs_no_gate_and_no_actuator(self):
        args = _args(regime_controller="observe")
        _validate(args)
        self.assertEqual(args.regime_controller, "observe")

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            _validate(_args(regime_controller="on"))
        self.assertIn("not a known mode", str(cm.exception))
        self.assertEqual(MODES, ("off", "observe", "act"))

    def test_act_without_evidence_is_refused_and_names_the_gate(self):
        with self.assertRaises(ValueError) as cm:
            _validate(_args(regime_controller="act"))
        text = str(cm.exception)
        self.assertIn("entry gate", text)
        for item in GATE_ITEMS:
            self.assertIn(item.key, text)
        self.assertIn("--regime-controller observe", text)

    def test_act_with_evidence_but_no_actuator_is_refused(self):
        """The gate authorizes acting; it does not conjure an actuator."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gate.json")
            with open(path, "w") as f:
                json.dump(_evidence(), f)
            with self.assertRaises(ValueError) as cm:
                _validate(_args(regime_controller="act", regime_gate_evidence=path))
            self.assertIn("at least one runtime actuator", str(cm.exception))

    def test_act_with_evidence_and_an_actuator_validates(self):
        """The gate OPENS. Both halves of the falsifier duty are exercised."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gate.json")
            with open(path, "w") as f:
                json.dump(_evidence(), f)
            args = _args(
                regime_controller="act",
                regime_gate_evidence=path,
                kv_reshard_vectors="2,11,10",
            )
            _validate(args)
            self.assertEqual(args.regime_controller, "act")

    def test_a_malformed_evidence_file_is_caught_even_in_observe(self):
        """Declaring the evidence before switching the mode is the natural
        order, so it is parsed then -- not on the boot that finally acts."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gate.json")
            with open(path, "w") as f:
                f.write("{oops")
            with self.assertRaises(RegimeError):
                _validate(_args(regime_controller="observe", regime_gate_evidence=path))


class TestModeResolution(CustomTestCase):
    def test_the_flag_decides(self):
        from unittest import mock

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_mode(_args()), MODE_OFF)
            self.assertEqual(
                resolve_mode(_args(regime_controller="observe")), MODE_OBSERVE
            )
            self.assertEqual(resolve_mode(_args(regime_controller="act")), MODE_ACT)

    def test_the_env_override_can_only_turn_observation_on(self):
        from unittest import mock

        from sglang.srt.managers.regime_runtime import ENV_MODE

        with mock.patch.dict(os.environ, {ENV_MODE: "1"}):
            self.assertEqual(resolve_mode(_args()), MODE_OBSERVE)
            # It cannot downgrade an acting server either way.
            self.assertEqual(resolve_mode(_args(regime_controller="act")), MODE_ACT)

    def test_the_env_override_cannot_ask_for_act(self):
        from unittest import mock

        from sglang.srt.managers.regime_runtime import ENV_MODE

        with mock.patch.dict(os.environ, {ENV_MODE: "act"}):
            with self.assertRaises(ValueError) as cm:
                resolve_mode(_args())
            self.assertIn("cannot carry evidence", str(cm.exception))


# ---------------------------------------------------------------------------
# 4. The actuator bridge
# ---------------------------------------------------------------------------


class _Spy:
    def __init__(self, ok=True, msg="ok"):
        self.ok, self.msg = ok, msg
        self.calls = []

    def arm(self, vector, source):
        self.calls.append(("arm", tuple(vector), source))
        return self.ok, self.msg

    def vram(self, budgets):
        self.calls.append(("vram", tuple(budgets)))
        return self.ok, self.msg


class TestActuatorBridge(CustomTestCase):
    def test_a_kv_difference_arms_297_with_the_controller_named(self):
        spy = _Spy()
        act = RegimeActuator(reshard_arm=spy.arm)
        result = act.apply(KV_STAGE, BOOTED)
        self.assertTrue(result.armed)
        self.assertEqual(spy.calls, [("arm", (2, 11, 10), ARM_SOURCE)])
        self.assertEqual(act.arms, 1)

    def test_a_grow_applies_the_dial_before_the_reshard(self):
        """A bigger pool cannot make a reshard fail; a reshard into a pool
        about to grow would be sized against the smaller one."""
        spy = _Spy()
        grown = _stage(
            "wide", REGIME_PREFILL_HEAVY, kv=(2, 11, 10), vram=(29607, 18780, 18780)
        )
        act = RegimeActuator(reshard_arm=spy.arm, vram_apply=spy.vram)
        act.apply(grown, BOOTED)
        self.assertEqual([c[0] for c in spy.calls], ["vram", "arm"])

    def test_a_shrink_is_refused_before_anything_is_armed(self):
        spy = _Spy()
        shrunk = _stage(
            "narrow", REGIME_PREFILL_HEAVY, kv=(2, 11, 10), vram=(29607, 16780, 16780)
        )
        act = RegimeActuator(reshard_arm=spy.arm, vram_apply=spy.vram)
        result = act.apply(shrunk, BOOTED)
        self.assertFalse(result.armed)
        self.assertIn("SHRINK", result.reason)
        self.assertIn("radix cache", result.reason)
        self.assertEqual(spy.calls, [], "a refused move armed something anyway")

    def test_a_missing_actuator_names_the_flag(self):
        act = RegimeActuator()
        result = act.apply(KV_STAGE, BOOTED)
        self.assertFalse(result.armed)
        self.assertIn("--kv-reshard-vectors", result.reason)

    def test_an_unselectable_stage_never_reaches_an_actuator(self):
        spy = _Spy()
        act = RegimeActuator(
            reshard_arm=spy.arm,
            selectable_fn=lambda name: (False, "weights differ; restart needed"),
        )
        result = act.apply(WEIGHT_STAGE, BOOTED)
        self.assertFalse(result.armed)
        self.assertEqual(spy.calls, [])
        self.assertIn("restart", result.reason)

    def test_an_actuator_refusal_is_passed_through_verbatim(self):
        spy = _Spy(ok=False, msg="target not in the declared ceiling set")
        act = RegimeActuator(reshard_arm=spy.arm)
        result = act.apply(KV_STAGE, BOOTED)
        self.assertFalse(result.armed)
        self.assertIn("declared ceiling set", result.reason)
        self.assertEqual(act.refusals, 1)

    def test_a_no_op_move_is_refused_rather_than_armed(self):
        act = RegimeActuator(reshard_arm=_Spy().arm)
        result = act.apply(BOOTED, BOOTED)
        self.assertFalse(result.armed)
        self.assertIn("no reachable axis", result.reason)

    def test_it_never_raises(self):
        """A controller that raises inside the scheduler loop turns a bad
        proposal into a dead server."""

        def boom(vector, source):
            raise RuntimeError("nope")

        act = RegimeActuator(reshard_arm=boom)
        with self.assertRaises(RuntimeError):
            # The actuator itself is allowed to propagate a genuine actuator
            # fault; what it must not do is invent one. Documented here so the
            # boundary is explicit rather than assumed.
            act.apply(KV_STAGE, BOOTED)


# ---------------------------------------------------------------------------
# 5. The act path end to end, and the interlocks that gate it
# ---------------------------------------------------------------------------


def _plan():
    return build_stage_table(
        booted=BOOTED, candidates=[KV_STAGE], declared_vectors=DECLARED
    )


def _observer(mode, *, commit_fn=None, dwell=None, tp_size=1, collective=None):
    plan = _plan()
    return (
        RegimeObserver(
            consensus_interval=2,
            tp_size=tp_size,
            collective_min=collective,
            mode=mode,
            table=plan.table,
            table_plan=plan,
            current_stage=plan.booted,
            commit_fn=commit_fn,
            dwell=dwell,
        ),
        plan,
    )


def _drive(obs, rounds, *, prefill=True, held=40_000, ms=10.0):
    out = []
    for _ in range(rounds):
        rec = obs.on_round(
            phase="prefill" if prefill else "decode",
            held_tokens=held,
            capacity_tokens=453_632,
            running_bs=1,
            rank_forward_ms=ms,
        )
        if rec is not None:
            out.append(rec)
    return out


class TestActPathFires(CustomTestCase):
    """FALSIFIER, the OPEN half: under a gate-passed condition the act path
    must actually reach the actuator."""

    def test_a_proposal_becomes_an_arm(self):
        spy = _Spy()
        actuator = RegimeActuator(reshard_arm=spy.arm)
        obs, _plan_ = _observer(MODE_ACT, commit_fn=actuator.apply)
        records = _drive(obs, 16)
        armed = [r for r in records if r["actuated"]]
        self.assertTrue(armed, [r["reason"] for r in records])
        self.assertEqual(spy.calls[0], ("arm", (2, 11, 10), ARM_SOURCE))
        self.assertEqual(obs.summary()["actuations"], len(armed))
        self.assertEqual(obs.summary()["mode"], MODE_ACT)

    def test_the_current_stage_advances_so_the_move_is_not_repeated(self):
        spy = _Spy()
        actuator = RegimeActuator(reshard_arm=spy.arm)
        obs, _plan_ = _observer(MODE_ACT, commit_fn=actuator.apply)
        _drive(obs, 40)
        arms = [c for c in spy.calls if c[0] == "arm"]
        self.assertEqual(len(arms), 1, "the same flip was armed repeatedly")

    def test_the_action_detail_travels_in_the_record(self):
        actuator = RegimeActuator(reshard_arm=_Spy().arm)
        obs, _plan_ = _observer(MODE_ACT, commit_fn=actuator.apply)
        records = _drive(obs, 16)
        acted = [r for r in records if r["action"] is not None][0]
        self.assertIn("kv", acted["action"]["detail"])


class TestActInterlocks(CustomTestCase):
    def test_the_dwell_gate_vetoes_a_second_flip(self):
        spy = _Spy()
        actuator = RegimeActuator(reshard_arm=spy.arm)
        obs, _plan_ = _observer(
            MODE_ACT, commit_fn=actuator.apply, dwell=DwellGate(min_rounds=10_000)
        )
        # First flip lands (no flip has happened yet, so the gate permits it),
        # then the regime flaps back and forth.
        _drive(obs, 20, prefill=True)
        _drive(obs, 20, prefill=False)
        _drive(obs, 20, prefill=True)
        arms = [c for c in spy.calls if c[0] == "arm"]
        self.assertLessEqual(len(arms), 1)
        self.assertTrue(any("dwell" in k for k in obs.vetoes), obs.vetoes)

    def test_an_unselectable_stage_is_vetoed_before_the_actuator(self):
        spy = _Spy()
        plan = build_stage_table(
            booted=BOOTED, candidates=[WEIGHT_STAGE], declared_vectors=DECLARED
        )
        actuator = RegimeActuator(reshard_arm=spy.arm)
        obs = RegimeObserver(
            consensus_interval=2,
            tp_size=1,
            mode=MODE_ACT,
            table=plan.table,
            table_plan=plan,
            current_stage=plan.booted,
            commit_fn=actuator.apply,
        )
        records = _drive(obs, 16)
        self.assertEqual(spy.calls, [])
        self.assertTrue(any("ACT VETO" in r["reason"] for r in records))
        self.assertEqual(obs.summary()["actuations"], 0)

    def test_the_admissibility_veto_still_binds_in_act_mode(self):
        """The #350 lesson carried over: a stage the guards reject is not
        selectable, whatever the regime says. 340 000 held tokens exceed
        0.85 x the 380 000 pool of the target."""
        spy = _Spy()
        actuator = RegimeActuator(reshard_arm=spy.arm)
        obs, _plan_ = _observer(MODE_ACT, commit_fn=actuator.apply)
        records = _drive(obs, 16, held=340_000)
        self.assertEqual(spy.calls, [])
        self.assertTrue(any("380000" in r["reason"] for r in records))

    def test_a_disputed_verdict_vetoes_the_flip(self):
        from sglang.srt.managers.regime_classifier import REGIME_CODES

        def channel(payload):
            peer = list(payload)
            code = REGIME_CODES[REGIME_DECODE_HEAVY]
            peer[0], peer[1] = code, -code
            return [min(a, b) for a, b in zip(payload, peer)]

        spy = _Spy()
        actuator = RegimeActuator(reshard_arm=spy.arm)
        obs, _plan_ = _observer(
            MODE_ACT, commit_fn=actuator.apply, tp_size=2, collective=channel
        )
        records = _drive(obs, 16)
        self.assertEqual(spy.calls, [])
        self.assertTrue(any("disputed verdict" in r["reason"] for r in records))

    def test_the_one_boundary_stale_veto_binds_when_no_group_timing_exists(self):
        """Phase 2 built the lag; act makes it bite. Every rank blind means
        the planner's split has never been checked against this rig."""

        def channel(payload):
            return list(payload)

        spy = _Spy()
        actuator = RegimeActuator(reshard_arm=spy.arm)
        obs, _plan_ = _observer(
            MODE_ACT, commit_fn=actuator.apply, tp_size=2, collective=channel
        )
        records = _drive(obs, 16, ms=None)
        self.assertEqual(spy.calls, [])
        self.assertTrue(any("one-boundary-" in r["reason"] for r in records))

    def test_a_multi_rank_group_with_no_channel_may_not_move_anything(self):
        spy = _Spy()
        actuator = RegimeActuator(reshard_arm=spy.arm)
        obs, _plan_ = _observer(
            MODE_ACT, commit_fn=actuator.apply, tp_size=3, collective=None
        )
        records = _drive(obs, 16)
        self.assertEqual(spy.calls, [])
        self.assertTrue(any("unchecked" in r["reason"] for r in records))


# ---------------------------------------------------------------------------
# 6. The phase-2 property must survive phase 3
# ---------------------------------------------------------------------------


class TestObserveStillCannotAct(CustomTestCase):
    """The F4 property, re-asserted now that an action path exists."""

    @staticmethod
    def _imports(module):
        import ast
        import pathlib

        tree = ast.parse(pathlib.Path(module.__file__).read_text())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module)
        return mods, tree

    def test_the_observer_module_imports_no_actuator_at_module_scope(self):
        import ast

        import sglang.srt.managers.regime_runtime as mod

        mods, tree = self._imports(mod)
        # Module-scope imports only: the act branch of build_regime_observer
        # imports regime_act deliberately, inside the function.
        top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top.add(node.module)
        for forbidden in ("kv_reshard", "vram_dial", "regime_act"):
            self.assertFalse(
                any(forbidden in m for m in top),
                f"{forbidden} is imported at module scope: {sorted(top)}",
            )
        for forbidden in ("kv_reshard", "vram_dial"):
            self.assertFalse(
                any(forbidden in m for m in mods),
                f"{forbidden} is reachable from the observer: {sorted(mods)}",
            )

    def test_the_observer_calls_no_actuator_method(self):
        import ast
        import pathlib

        import sglang.srt.managers.regime_runtime as mod

        tree = ast.parse(pathlib.Path(mod.__file__).read_text())
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        for forbidden in ("arm", "apply_budget_request", "throttle", "try_spill"):
            self.assertNotIn(forbidden, called)

    def test_observe_mode_refuses_to_hold_a_commit_path(self):
        """Not merely unused: observe may not even be handed one, so a future
        edit cannot quietly start calling it."""
        plan = _plan()
        with self.assertRaises(ValueError) as cm:
            RegimeObserver(
                mode=MODE_OBSERVE,
                table=plan.table,
                table_plan=plan,
                current_stage=plan.booted,
                commit_fn=lambda *a: None,
            )
        self.assertIn("must not hold a path to an actuator", str(cm.exception))

    def test_act_mode_refuses_to_run_without_one(self):
        with self.assertRaises(ValueError) as cm:
            RegimeObserver(mode=MODE_ACT)
        self.assertIn("expensive observe", str(cm.exception))

    def test_observe_actuates_nothing_even_with_a_full_table(self):
        obs, _plan_ = _observer(MODE_OBSERVE)
        records = _drive(obs, 32)
        self.assertTrue(any(r["would_flip_to"] == "fp8-prefill-kv" for r in records))
        self.assertTrue(all(r["actuated"] is False for r in records))
        self.assertTrue(all(r["action"] is None for r in records))
        self.assertEqual(obs.summary()["actuations"], 0)

    def test_the_act_module_touches_only_the_two_documented_entry_points(self):
        import ast
        import pathlib

        import sglang.srt.managers.regime_act as mod

        tree = ast.parse(pathlib.Path(mod.__file__).read_text())
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        # It may call arm and apply_budget_request. It may not reach for the
        # admission limiter or the spill manager -- those belong to #287.
        for forbidden in ("throttle", "try_spill", "set_limit"):
            self.assertNotIn(forbidden, called)


if __name__ == "__main__":
    unittest.main()
