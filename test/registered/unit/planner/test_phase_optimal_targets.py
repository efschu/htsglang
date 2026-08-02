"""The phase-optimal arms of ``--rank-perf-tune`` (task #357).

#354 measured the phase-optimal recipe on real boots: one weight vector for
the PREFILL phase, the plain VRAM-auto split for the DECODE phase, per
checkpoint format. The vectors it booted came out of the planner's own
REFUSAL line -- the ladder solved them and then rejected them, so the recipe
could not be asked for by name.

The refusal had two different reasons at two different operating points, and
only one of them was wrong:

``--rank-auto-reserve-mib 3000,2700,2700`` (the runbook reserve)
    UNBOOTABLE: ``16,1,1`` leaves rank 0 with 3000 MiB residual against a
    derived reserve demand of 4160 MiB. That verdict is CORRECT and is kept
    binding here. #264 measured the OOM (6,1,1 at reserve 3000, GPU 0 down to
    41.69 MiB free), and #354 confirmed the demand model from the other side:
    its prefill boots ran at reserve 4500 and ended with 87 MiB free on the
    5090, i.e. the non-budget posts really consumed 4413 MiB.

``--rank-auto-reserve-mib 4500,2700,2700`` (the reserve #354 actually booted)
    REJECTED by the decode-knee guard, predicted decode step +24.7 %. The
    guard's NUMBER is right -- the #354 decode arm measured +20.2 % (bs=8) to
    +25.0 % (bs=1) step time when that vector is left in place for decode.
    The verdict is not: in the phase recipe this vector serves prefill only,
    and the boot that ran it measured +22.6 % prefill at s=1 against the
    +22.9 % the same ladder predicted. A decode price the recipe never pays
    cannot be the gate that refuses a prefill vector.

So ``phase-prefill`` makes the decode-knee guard ADVISORY and keeps
fundability and the context floor binding, and the tests below pin exactly
that split: a concentrated vector is accepted at a reserve that funds it, and
the unbootable ones stay rejected at the reserve that OOMs.

What #435 and #437 changed about the reserve in that sentence. #354's boots
ran the VRAM-BUDGET KV split, under which a rank that is not the tightest
keeps its unused capacity as free VRAM -- #433 measured 7045 / 7109 MiB idle
on ranks 1 and 2 of exactly this shape. #435 makes the phase arm boot the
solve's own MATCHED token vector, which spends that slack, and #437 makes the
fundability gate price what the boot will run: under a matched vector every
rank is checked ABSOLUTELY against the derived reserve demand, not relative to
a base plan whose residual is the same number. ``4500,2700,2700`` therefore no
longer funds 16,1,1 -- not because the gate got stricter about rank 0, but
because ranks 1 and 2 no longer have the slack the old pricing credited them
with. The phase-arm tests run at ``_RESERVE_MATCHED``; the #264 negative
control keeps its own runbook reserve unchanged.

Fixture note: this is the #264/#265 fixture shape with the v3 profile that
carries the measured GEMM LANES of the #354 window (fp8_native 568.48 /
fp8_marlin 58.44 / 59.15, and the int8_native lane 676.69 / 183.78 / 164.77
probed during it). The lane ratio is what makes the ladder solve 16,1,1 for
FP8 and the flatter 10,1,1 for INT8; on the bf16 ``gemm_tflops`` numbers the
ladder solves neither.
"""

import os
import types
import unittest
from unittest import mock

from sglang.srt import uneven_perf
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

_CACHE = os.environ.get("HTSGLANG_TEST_MODEL_DIR", "")
_MODELS = {
    "fp8": os.path.join(_CACHE, "Qwen3.6-27B-FP8") if _CACHE else "",
    "int8": os.path.join(_CACHE, "Qwen3.6-27B-INT8-W8A8") if _CACHE else "",
}
#: The vector #354 booted for each format, and its measured prefill delta.
_PHASE_VECTOR = {"fp8": "16,1,1", "int8": "10,1,1"}
#: What the phase arm solves at ``_RESERVE_MATCHED`` below. FP8 keeps #354's
#: own vector; INT8 moves one rung down the ladder because the raised 3080
#: reserve shrinks their budgets -- the SAME re-solve the pre-#437 optimizer
#: performs at that reserve (verified against the base tree), and the vector
#: NOTE_433 section 4 already recorded as INT8's optimum once the reserve
#: goes up. Nothing about #437 picks it.
_PHASE_VECTOR_MATCHED = {"fp8": "16,1,1", "int8": "8,1,1"}

#: NVML totals of the reference rig in rank order (5090, 3080, 3080).
_TOTALS = [32607, 20480, 20480]
#: ``derived_rank_auto_reserve_mib`` for this boot shape, from the #264 logs.
_DEMAND = 4160
#: The reserve #354's prefill boots ran at, and the runbook reserve that the
#: same vector cannot boot on.
_RESERVE_BOOTED = (4500, 2700, 2700)
_RESERVE_RUNBOOK = (3000, 2700, 2700)
#: The reserve the phase arm needs AFTER #435, and why it is not
#: ``_RESERVE_BOOTED``. #354's boots ran the VRAM-budget KV split, under which
#: the non-binding ranks keep their unused capacity as free VRAM -- #433
#: measured 7045 / 7109 MiB idle on ranks 1 and 2. #435 makes the boot run the
#: solve's own MATCHED token vector, which spends exactly that, so at
#: 2700 MiB the two 3080s no longer cover the 4160 MiB derived reserve demand
#: and #437's gate refuses every candidate on them. NOTE_433's confirmation-arm
#: spec names the same remedy from the other side ("the retry raises
#: --rank-auto-reserve-mib on those two cards"). 4600 clears 4160 + #330's
#: 400 MiB corridor on all three.
_RESERVE_MATCHED = (4600, 4600, 4600)

#: Measured GEMM lanes of the #354 window (lanes.txt of the result set).
_LANES = {
    0: {
        "fp8_native": 568.48,
        "fp8_marlin": 216.34,
        "fp8_w8a16": 181.43,
        "int8_native": 676.69,
    },
    1: {"fp8_marlin": 58.44, "fp8_w8a16": 53.43, "int8_native": 183.78},
    2: {"fp8_marlin": 59.15, "fp8_w8a16": 53.78, "int8_native": 164.77},
}
_CARDS = [
    ("U0", 0, "NVIDIA GeForce RTX 5090", 32607, 232.97, 1664.2, 1529.7),
    ("U1", 1, "NVIDIA GeForce RTX 3080", 20480, 62.72, 717.8, 717.8),
    ("U2", 2, "NVIDIA GeForce RTX 3080", 20480, 62.98, 717.4, 717.8),
]


def _profile():
    gpus = {}
    for uuid, idx, name, total, tflops, bw, gemv in _CARDS:
        gpus[uuid] = {
            "name": name,
            "cuda_index": idx,
            "total_mib": total,
            "gemm_tflops": tflops,
            "membw_gbs": bw,
            "membw_gemv_gbs": gemv,
            "gemm_lanes": dict(_LANES[idx]),
            "gemm_lane_notes": {},
        }
    profile = {
        "version": 3,
        "driver": "595.58.03",
        "gpus": gpus,
        "links": {
            "U0|U1": {"p2p_gbs": 5.1},
            "U0|U2": {"p2p_gbs": 9.06},
            "U1|U2": {"p2p_gbs": 5.83},
            "__group__": {"ar_10kb_us": 32.4, "ar_1mb_us": 361.3},
        },
    }
    inventory = [
        {
            "uuid": uuid,
            "cuda_index": g["cuda_index"],
            "name": g["name"],
            "total_mib": g["total_mib"],
        }
        for uuid, g in gpus.items()
    ]
    return profile, inventory


#: The #354 boots ran with the bf16 SSM state; the model reads it from the
#: environment.
_ENV = {"SGLANG_MAMBA_SSM_DTYPE": "bfloat16"}


def _args(
    model,
    reserve=_RESERVE_MATCHED,
    tune="phase-prefill",
    loose=0.0,
    kv_ratio="coupled",
):
    """Duck-typed ServerArgs with exactly the fields the optimizer reads."""
    budgets = [t - r for t, r in zip(_TOTALS, reserve)]
    sa = types.SimpleNamespace(
        model_path=model,
        tp_size=3,
        rank_gpu_id=[0, 1, 2],
        rank_gpu_memory_mib=list(budgets),
        rank_tp_ratio=list(budgets),
        rank_mlp_ratio=None,
        rank_vocab_ratio=None,
        rank_moe_ratio=None,
        rank_kv_ratio=kv_ratio,
        rank_kv_capacity_seed=None,
        rank_auto_reserve_mib=",".join(map(str, reserve)),
        rank_perf_tune=tune,
        rank_perf_loose_ctx_percent=loose,
        kv_cache_dtype="fp8_e4m3",
        context_length=32768,
        page_size=1,
        quantization=None,
        max_running_requests=16,
        chunked_prefill_size=2048,
        mem_fraction_static=0.7435115625,
        speculative_algorithm="EAGLE",
        speculative_draft_model_path=None,
        speculative_num_draft_tokens=4,
        speculative_adaptive=False,
        speculative_adaptive_config=None,
        speculative_cross_algorithm=False,
        speculative_draft_placement="split",
        disable_cuda_graph=False,
        dcp_size=3,
        _derived_rank_auto_reserve_per_gpu={0: _DEMAND, 1: _DEMAND, 2: _DEMAND},
        _measured_kv_budget_registry_path="/nonexistent/registry.json",
        cuda_graph_config=types.SimpleNamespace(
            decode=types.SimpleNamespace(max_bs=24)
        ),
    )
    sa.uneven_kv_flag_active = lambda: sa.rank_kv_ratio != "coupled"
    sa.uneven_kv_capacity_mode = lambda: sa.rank_kv_ratio == "capacity"
    sa.uneven_kv_speed_mode = lambda: sa.rank_kv_ratio == "speed"
    sa.uneven_kv_derived_mode = lambda: (
        sa.uneven_kv_capacity_mode() or sa.uneven_kv_speed_mode()
    )
    return sa


def _plan(**kw):
    """Run the optimizer on the fixture; return ``(server_args, log text)``."""
    sa = _args(**kw)
    profile, inventory = _profile()
    captured = []
    with mock.patch.dict(os.environ, _ENV), mock.patch.object(
        uneven_perf,
        "get_hardware_profile",
        return_value=(profile, "test fixture", inventory),
    ), mock.patch.object(
        uneven_perf.logger,
        "info",
        lambda *a, **k: captured.append(a[0] if a else ""),
    ):
        uneven_perf.apply_auto_performance(sa)
    return sa, "\n".join(captured)


def _line_for(log, vector):
    for line in log.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"candidate MLP vector {vector}:"):
            return stripped
    return None


def _have(fmt):
    return bool(_MODELS[fmt]) and os.path.isdir(_MODELS[fmt])


_SKIP = unittest.skipUnless(
    _have("fp8") and _have("int8"),
    "HTSGLANG_TEST_MODEL_DIR/Qwen3.6-27B-{FP8,INT8-W8A8} not present",
)


@_SKIP
class TestFixtureReproducesThe354Ladder(CustomTestCase):
    """The fixture is the #354 plan log, not a plausible-looking stand-in."""

    def test_the_solved_vectors_are_the_ones_354_booted(self):
        """The refusal line of ``plan_fp8_auto.txt`` / ``plan_int8_auto.txt``
        named 16,1,1 (+22.2 %) and 10,1,1 (+9.1 %) as the best forfeited
        candidates. Same fixture, same ladder, same two vectors."""
        for fmt, gain in (("fp8", "+22.2%"), ("int8", "+9.1%")):
            with self.subTest(fmt=fmt):
                _sa, log = _plan(
                    model=_MODELS[fmt], reserve=_RESERVE_RUNBOOK, tune="enc"
                )
                line = _line_for(log, _PHASE_VECTOR[fmt])
                self.assertIsNotNone(line, "the #354 vector is not on the ladder")
                self.assertIn(f"predicted prefill gain {gain}", line)

    def test_the_lane_ratio_is_what_separates_the_two_formats(self):
        """FP8 concentrates harder than INT8 because the 3080s run INT8
        natively instead of through Marlin (9.73:1 vs 3.68:1). If both formats
        solved the same vector the fixture would be scoring one lane twice."""
        self.assertNotEqual(_PHASE_VECTOR["fp8"], _PHASE_VECTOR["int8"])
        vectors = []
        for fmt in ("fp8", "int8"):
            sa, _log = _plan(model=_MODELS[fmt], tune="phase-prefill")
            vectors.append(sa.rank_mlp_ratio)
        self.assertNotEqual(vectors[0], vectors[1])


@_SKIP
class TestPhasePrefillAcceptsWhatBooted(CustomTestCase):
    """The regression: the planner must propose a concentrated prefill vector,
    the shape #354 measured.

    FALSIFIER: on the pre-#357 ladder every one of these runs returns None and
    the log says "rejected by: knee" -- at a reserve whose boot served
    1540.3 tok/s prefill.

    FP8 still solves #354's own 16,1,1 here. INT8 solves 8,1,1 rather than
    #354's 10,1,1 because ``_RESERVE_MATCHED`` shrinks the 3080 budgets; the
    pre-#437 optimizer re-solves it the same way at the same reserve, and
    NOTE_433 section 4 recorded 8,1,1 as INT8's optimum once the reserve goes
    up. The property under test is that a real lopsided vector is proposed,
    not that one frozen number survives every operating point.
    """

    def test_the_354_prefill_vector_is_installed_for_both_formats(self):
        for fmt in ("fp8", "int8"):
            with self.subTest(fmt=fmt):
                sa, log = _plan(model=_MODELS[fmt], tune="phase-prefill")
                self.assertEqual(
                    ",".join(map(str, sa.rank_mlp_ratio or [])),
                    _PHASE_VECTOR_MATCHED[fmt],
                    "the planner does not propose the phase-prefill vector",
                )
                self.assertIn("CHOSEN MLP vector", log)

    def test_enc_still_rejects_it_on_the_decode_knee(self):
        """The single-vector targets are untouched: enc/both serve BOTH
        phases from one vector, so the decode cost is theirs to pay and the
        guard keeps rejecting."""
        for tune in ("enc", "both"):
            with self.subTest(tune=tune):
                sa, log = _plan(model=_MODELS["fp8"], tune=tune)
                self.assertIsNone(sa.rank_mlp_ratio)
                self.assertIn("REJECTED by decode-knee guard", log)

    def test_the_advisory_verdict_still_prints_the_decode_number(self):
        """Advisory is not silent: the number that would have rejected the
        candidate stays in the log, so a reader can disagree with the mode.

        The number is +23.8 % at ``_RESERVE_MATCHED`` and +24.7 % at
        ``_RESERVE_BOOTED``: the reserve sets the budgets, the budgets set the
        MLP unit grid, and the grid sets which streamed-byte share 16,1,1
        materializes. Both are the same prediction of the same cost, and #354
        measured +20.2 % (bs=8) to +25.0 % (bs=1) for it."""
        _sa, log = _plan(model=_MODELS["fp8"], tune="phase-prefill")
        line = _line_for(log, "16,1,1")
        self.assertIn("decode-knee ADVISORY", line)
        self.assertIn("predicted decode step +23.8%", line)
        # And the number is printed even where the verdict is something else
        # entirely -- at _RESERVE_BOOTED the same candidate is UNBOOTABLE
        # post-#435, and the decode cost is still stated next to it.
        _sa, log = _plan(
            model=_MODELS["fp8"], reserve=_RESERVE_BOOTED, tune="phase-prefill"
        )
        self.assertIn(
            "predicted decode step +24.7%", _line_for(log, "16,1,1")
        )

    def test_the_installed_arm_names_its_companion_and_the_restart(self):
        _sa, log = _plan(model=_MODELS["fp8"], tune="phase-prefill")
        self.assertIn("phase-optimal PREFILL arm installed", log)
        self.assertIn("--rank-perf-tune phase-decode", log)
        self.assertIn("RESTART", log)


@_SKIP
class TestPhaseDecodeIsTheOtherArm(CustomTestCase):
    """The decode arm is the plain VRAM-auto split -- the #265 finding, which
    #354 reproduced (FP8 decode 122.2 tok/s at bs=1 on auto against 97.8 on
    the concentrated vector). It still has to NAME the prefill companion, or
    the recipe is only half askable."""

    def test_phase_decode_keeps_the_auto_split(self):
        for fmt in ("fp8", "int8"):
            with self.subTest(fmt=fmt):
                sa, _log = _plan(model=_MODELS[fmt], tune="phase-decode")
                self.assertIsNone(sa.rank_mlp_ratio)

    def test_phase_decode_names_the_prefill_companion(self):
        for fmt in ("fp8", "int8"):
            with self.subTest(fmt=fmt):
                _sa, log = _plan(model=_MODELS[fmt], tune="phase-decode")
                self.assertIn("phase-optimal DECODE arm", log)
                self.assertIn(
                    f"companion PREFILL vector for this rig and checkpoint is "
                    f"{_PHASE_VECTOR_MATCHED[fmt]}",
                    log,
                )


@_SKIP
class TestTheGateStillRejectsWhatDoesNotBoot(CustomTestCase):
    """The negative control. A gate that accepts everything is not a gate.

    FALSIFIER for a future loosening: make fundability advisory too, or bump
    the phase arm past the reserve check, and this class goes red -- at the
    reserve where #264 measured the OOM.
    """

    def test_the_354_vector_stays_unbootable_at_the_runbook_reserve(self):
        for fmt in ("fp8", "int8"):
            with self.subTest(fmt=fmt):
                sa, log = _plan(
                    model=_MODELS[fmt],
                    reserve=_RESERVE_RUNBOOK,
                    tune="phase-prefill",
                )
                line = _line_for(log, _PHASE_VECTOR[fmt])
                self.assertIn("UNBOOTABLE", line)
                self.assertIn("rank 0 residual free 3000 MiB", line)
                self.assertIn("derived reserve demand 4160 MiB", line)
                self.assertNotEqual(
                    ",".join(map(str, sa.rank_mlp_ratio or [])),
                    _PHASE_VECTOR[fmt],
                    "an unbootable vector was installed by the phase arm",
                )

    def test_whatever_the_phase_arm_installs_there_clears_the_demand(self):
        """It may fall back to a flatter vector -- but never to one that the
        fundability model says cannot fund its own boot."""
        for fmt in ("fp8", "int8"):
            with self.subTest(fmt=fmt):
                sa, log = _plan(
                    model=_MODELS[fmt],
                    reserve=_RESERVE_RUNBOOK,
                    tune="phase-prefill",
                )
                if sa.rank_mlp_ratio is None:
                    continue
                line = _line_for(log, ",".join(map(str, sa.rank_mlp_ratio)))
                self.assertNotIn("UNBOOTABLE", line or "")

    def test_the_context_floor_still_binds_in_the_phase_arm(self):
        """Forced the same way #265 forces it: shave 0.1 % off every
        concentrated candidate's predicted context, three orders of magnitude
        above the summation noise the tolerance absorbs."""
        real = uneven_perf.PerfCostModel.predict_capacity

        def shaved(self, mlp_vector):
            out = dict(real(self, mlp_vector))
            if list(mlp_vector) != list(self.base_plan):
                out["ctx"] = out["ctx"] * 0.999
            return out

        with mock.patch.object(
            uneven_perf.PerfCostModel, "predict_capacity", shaved
        ):
            sa, log = _plan(model=_MODELS["fp8"], tune="phase-prefill")
        self.assertIn("REJECTED by floor", log)
        self.assertIsNone(sa.rank_mlp_ratio)


@_SKIP
class TestTheCapacityModeGateIsNotBlind(CustomTestCase):
    """#437: the same negative control, in the mode that used to be exempt.

    ``--rank-kv-ratio capacity`` derives a token vector PER CANDIDATE, so the
    residual's slack term -- capacity the vector does not use -- is ~0 for
    every candidate including the base. The shipped gate compared the two and
    could never fire: it accepted ``16,1,1`` at reserve ``3000,2700,2700``,
    the configuration #264 booted into an OOM at 41.69 MiB free on GPU 0,
    while the identical command on the default coupled ratio called the same
    candidate UNBOOTABLE.

    FALSIFIER: revert ``_fund_matched`` in ``uneven_perf`` and
    ``test_the_264_config_is_refused_in_capacity_mode`` goes red -- the
    planner installs 16,1,1 at the OOM reserve again.
    """

    def test_the_264_config_is_refused_in_capacity_mode(self):
        for mode in ("capacity", "speed"):
            with self.subTest(mode=mode):
                sa, log = _plan(
                    model=_MODELS["fp8"],
                    reserve=_RESERVE_RUNBOOK,
                    tune="phase-prefill",
                    kv_ratio=mode,
                )
                line = _line_for(log, "16,1,1")
                self.assertIn("UNBOOTABLE", line)
                self.assertIn("derived reserve demand 4160 MiB", line)
                self.assertIsNone(
                    sa.rank_mlp_ratio,
                    "the OOM configuration of #264 was installed anyway",
                )

    def test_the_matched_basis_is_named_and_the_base_is_warned_about(self):
        """When no candidate can be funded, the reader has to be told that the
        split it falls back to carries the same risk -- otherwise 'no vector
        installed' reads as 'safe'."""
        _sa, log = _plan(
            model=_MODELS["fp8"],
            reserve=_RESERVE_RUNBOOK,
            tune="phase-prefill",
            kv_ratio="capacity",
        )
        self.assertIn("fundability basis: MATCHED KV token vector", log)
        self.assertIn("fundability WARNING", log)
        self.assertIn("--rank-auto-reserve-mib", log)

    def test_a_matched_vector_at_an_adequate_reserve_still_passes(self):
        """The anti-over-tightening control. The absolute basis is not a
        blanket refusal: at a reserve that covers the derived demand on all
        three cards the same mode installs the same concentrated vector the
        coupled arm installs."""
        for mode in ("capacity", "speed"):
            with self.subTest(mode=mode):
                sa, log = _plan(
                    model=_MODELS["fp8"], tune="phase-prefill", kv_ratio=mode
                )
                self.assertEqual(
                    ",".join(map(str, sa.rank_mlp_ratio or [])),
                    _PHASE_VECTOR_MATCHED["fp8"],
                )
                self.assertNotIn("fundability WARNING", log)
                self.assertNotIn("UNBOOTABLE", _line_for(log, "16,1,1"))

    def test_the_corridor_is_reported_and_is_not_binding(self):
        """#330's 400 MiB absolutely-free corridor is priced explicitly and
        stated per card, but it does not reject: #354's 16,1,1 booted and
        served at 87 MiB free, so a candidate between the demand and the
        corridor is a decision, not an error. Forced here by setting the
        corridor so wide that the accepted vector must fall inside it."""
        with mock.patch.dict(
            os.environ, {"SGLANG_PLANNER_CORRIDOR_MIB": "100000"}
        ):
            sa, log = _plan(model=_MODELS["fp8"], tune="phase-prefill")
        self.assertIn("CORRIDOR-TIGHT", log)
        self.assertIn("100000 MiB absolutely-free corridor", log)
        self.assertEqual(
            ",".join(map(str, sa.rank_mlp_ratio or [])),
            _PHASE_VECTOR_MATCHED["fp8"],
            "the corridor rejected a candidate it may only report on",
        )


class TestTheFlagAcceptsTheArmsByName(CustomTestCase):
    """The mode has to be reachable from the command line, and the arms that
    are NOT phase arms must keep their side effects."""

    def test_both_arms_are_valid_perf_tune_targets(self):
        from sglang.srt.server_args import _RANK_PERF_TUNE_CHOICES

        for tune in ("phase-prefill", "phase-decode"):
            self.assertIn(tune, _RANK_PERF_TUNE_CHOICES)

    def test_phase_decode_does_not_silently_switch_the_kv_mode(self):
        """``dec`` selects --rank-kv-ratio speed; ``phase-decode`` must not --
        the decode numbers #354 quotes are from a coupled-KV boot."""
        from sglang.srt.server_args import ServerArgs

        for tune, expected in (
            ("dec", "speed"),
            ("phase-decode", "coupled"),
            ("phase-prefill", "coupled"),
        ):
            with self.subTest(tune=tune):
                view = types.SimpleNamespace(
                    rank_perf_loose_ctx_percent=0.0,
                    rank_perf_tune=tune,
                    rank_kv_ratio="coupled",
                )
                ServerArgs._check_perf_flags(view, True)
                self.assertEqual(view.rank_kv_ratio, expected)

    def test_an_unknown_target_is_still_refused(self):
        from sglang.srt.server_args import ServerArgs

        view = types.SimpleNamespace(
            rank_perf_loose_ctx_percent=0.0,
            rank_perf_tune="phase",
            rank_kv_ratio="coupled",
        )
        with self.assertRaises(ValueError):
            ServerArgs._check_perf_flags(view, True)


if __name__ == "__main__":
    unittest.main()
