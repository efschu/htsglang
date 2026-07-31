"""Hermetic tests for the #307 ceiling fit: a --max-running-requests-ceiling
(#287) that the cards cannot hold is fitted to the budget instead of killing
the boot in the per-rank budget ledger.

The numbers are the Fenster-3 constellation (2026-07-30, Qwen3.6-27B-FP8,
TP=3 uneven, --rank-kv-ratio 7,3,3, NEXTN with 4 draft tokens): mamba ratio 5,
per-request state 32.73 / 23.38 / 18.70 MiB on ranks 0/1/2, and a post-weights
budget of roughly 7.4 GiB on the 20 GB cards. Ceiling 16 booted; ceiling 32
died 407 MiB over the budget and ceiling 64 died 559 MiB over it, both before
the first KV token. Card sizes are mocked -- nothing here touches a GPU.

The post-weights budget is RECONSTRUCTED from the recorded ledger (budget
minus the weights post); the posts that run after the mamba one in that boot
are not in the record, so the failure reproduces in SHAPE (the shortfall is
the activation reserve, which is why 32 and 64 failed by such similar
amounts) and not to the MiB.
"""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    MAMBA_AUTO_ACTIVATION_RESERVE_MIB,
    MAMBA_BUDGET_POST,
    MAMBA_CEILING_FIT_MIN_KV_MIB,
    ModelRunnerKVCacheMixin,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MIB = 1 << 20
GIB = 1 << 30

#: Fenster-3 per-request GDN state, per rank, in MiB (read off the boot logs).
PER_REQ_MIB = (32.73, 23.38, 18.70)
#: Post-weights budget the two 20 GB cards had left in that run.
REST_GB_3080 = 7.43
#: ...and the 5090's, which is the rank that is never the binding one.
REST_GB_5090 = 12.0
RATIO = 5
DRAFT_TOKENS = 4


def _args(**kw) -> ServerArgs:
    return ServerArgs(model_path="dummy", **kw)


def _stub(sa: ServerArgs, per_req_mib: float, *, spec: bool = True, sync=None):
    """A ModelRunner stand-in with exactly the surface handle_max_mamba_cache
    touches on the demand-driven (uneven-DCP) branch."""
    per_req = int(per_req_mib * MIB)
    stub = SimpleNamespace(
        server_args=sa,
        dp_size=1,
        spec_algorithm=SimpleNamespace(is_none=lambda: not spec),
        mambaish_config=SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(mamba_cache_per_req=per_req)
        ),
    )
    stub._calculate_mamba_ratio = lambda: RATIO
    stub._auto_mamba_demand_active = lambda: True
    stub._sync_uneven_mamba_cache_size = sync or (lambda: None)
    stub._auto_mamba_target_concurrency = (
        lambda: ModelRunnerKVCacheMixin._auto_mamba_target_concurrency(stub)
    )
    stub._auto_mamba_demand_size = lambda ratio: (
        ModelRunnerKVCacheMixin._auto_mamba_demand_size(stub, ratio)
    )
    stub._mamba_pool_budget_cost_gb = (
        lambda size,
        per_req_b,
        ratio,
        d: ModelRunnerKVCacheMixin._mamba_pool_budget_cost_gb(
            stub, size, per_req_b, ratio, d
        )
    )
    stub._fit_mamba_pool_to_budget = (
        lambda *a: ModelRunnerKVCacheMixin._fit_mamba_pool_to_budget(stub, *a)
    )
    return stub


def _size_and_rest(sa: ServerArgs, per_req_mib: float, rest_gb: float, **kw):
    """Run the real sizing routine; return (pool slots, KV budget left)."""
    stub = _stub(sa, per_req_mib, **kw)
    rest = ModelRunnerKVCacheMixin.handle_max_mamba_cache(stub, rest_gb)
    return sa.max_mamba_cache_size, rest


def _resolve_reqs(sa: ServerArgs, token_capacity: int = 400_000) -> int:
    stub = SimpleNamespace(
        server_args=sa,
        dp_size=1,
        model_config=SimpleNamespace(context_len=32768),
        mambaish_config=object(),
    )
    stub._calculate_mamba_ratio = lambda: RATIO
    return ModelRunnerKVCacheMixin._resolve_max_num_reqs(stub, token_capacity)


class TestFenster3Constellation(unittest.TestCase):
    """The two boots that died now fit -- and the one that carried is
    untouched."""

    def test_ceiling_64_fits_on_the_binding_3080(self):
        sa = _args(
            max_running_requests=8,
            max_running_requests_ceiling=64,
            speculative_num_draft_tokens=DRAFT_TOKENS,
        )
        size, rest = _size_and_rest(sa, PER_REQ_MIB[2], REST_GB_3080)
        self.assertGreater(size, 0)
        self.assertGreater(
            rest,
            MAMBA_CEILING_FIT_MIN_KV_MIB / 1024.0,
            "the fitted pool must leave a KV pool, not a rounding error",
        )
        # The float now runs below the fitted capacity, not below 64.
        self.assertLess(size // RATIO, 64)
        self.assertGreaterEqual(
            size // RATIO, 8, "the START value must remain servable"
        )

    def test_ceiling_32_fits_on_the_binding_3080(self):
        sa = _args(
            max_running_requests=8,
            max_running_requests_ceiling=32,
            speculative_num_draft_tokens=DRAFT_TOKENS,
        )
        _size, rest = _size_and_rest(sa, PER_REQ_MIB[2], REST_GB_3080)
        self.assertGreater(rest, MAMBA_CEILING_FIT_MIN_KV_MIB / 1024.0)

    def test_ceiling_64_was_over_budget_before_the_fit(self):
        """The falsifier: the pre-fit arithmetic really does go negative, so
        the tests above are measuring the fix and not a benign path."""
        sa = _args(
            max_running_requests=8,
            max_running_requests_ceiling=64,
            speculative_num_draft_tokens=DRAFT_TOKENS,
        )
        stub = _stub(sa, PER_REQ_MIB[2])
        per_req = int(PER_REQ_MIB[2] * MIB)
        # Exactly the old expression: min(demand, fit_cap) with no fit.
        demand = ModelRunnerKVCacheMixin._auto_mamba_demand_size(stub, RATIO)
        fit_cap = int(REST_GB_3080 * GIB // (per_req * (1 + DRAFT_TOKENS / RATIO)))
        unfitted = min(demand, max(fit_cap, 0))
        cost = ModelRunnerKVCacheMixin._mamba_pool_budget_cost_gb(
            stub, unfitted, per_req, RATIO, DRAFT_TOKENS
        )
        reserve_gb = MAMBA_AUTO_ACTIVATION_RESERVE_MIB / 1024.0
        left = REST_GB_3080 - cost - reserve_gb
        self.assertLess(left, 0.0)
        # Structural shape of the failure: once the pool is clamped by the
        # old fit_cap, the main state plus the intermediate state consume the
        # whole budget, and the shortfall is the activation reserve (minus
        # whatever the integer slot count left over). That is why raising the
        # ceiling changed the shortfall so little between 32 and 64.
        self.assertLessEqual(-left, reserve_gb + per_req * (1 + RATIO) / GIB)


class TestByteIdenticalPaths(unittest.TestCase):
    """Everything that boots today keeps its pool, slot for slot."""

    def _unfitted_size(self, stub, per_req: int, rest_gb: float) -> int:
        demand = ModelRunnerKVCacheMixin._auto_mamba_demand_size(stub, RATIO)
        fit_cap = int(rest_gb * GIB // (per_req * (1 + DRAFT_TOKENS / RATIO)))
        return min(demand, max(fit_cap, 0))

    def test_the_4_16_boot_is_unchanged_on_every_rank(self):
        for rank, per_req_mib in enumerate(PER_REQ_MIB):
            rest_gb = REST_GB_5090 if rank == 0 else REST_GB_3080
            sa = _args(
                max_running_requests=4,
                max_running_requests_ceiling=16,
                speculative_num_draft_tokens=DRAFT_TOKENS,
            )
            stub = _stub(sa, per_req_mib)
            want = self._unfitted_size(stub, int(per_req_mib * MIB), rest_gb)
            size, rest = _size_and_rest(sa, per_req_mib, rest_gb)
            self.assertEqual(size, want, f"rank {rank} pool changed")
            self.assertEqual(size, 100, f"rank {rank}: the boot log says 100 slots")
            self.assertGreater(rest, 0)

    def test_no_ceiling_is_unchanged(self):
        sa = _args(max_running_requests=16, speculative_num_draft_tokens=DRAFT_TOKENS)
        stub = _stub(sa, PER_REQ_MIB[2])
        want = self._unfitted_size(stub, int(PER_REQ_MIB[2] * MIB), REST_GB_3080)
        size, _rest = _size_and_rest(sa, PER_REQ_MIB[2], REST_GB_3080)
        self.assertIsNone(sa.max_running_requests_ceiling)
        self.assertEqual(size, want)

    def test_a_generous_budget_never_engages_the_fit(self):
        sa = _args(
            max_running_requests=8,
            max_running_requests_ceiling=64,
            speculative_num_draft_tokens=DRAFT_TOKENS,
        )
        stub = _stub(sa, PER_REQ_MIB[2])
        want = self._unfitted_size(stub, int(PER_REQ_MIB[2] * MIB), 40.0)
        size, rest = _size_and_rest(sa, PER_REQ_MIB[2], 40.0)
        self.assertEqual(size, want)
        self.assertEqual(size, 400, "64 * ratio 5 * safety 1.25")
        self.assertGreater(rest, 0)

    def test_fit_is_a_no_op_without_spec_decoding(self):
        sa = _args(max_running_requests=16, max_running_requests_ceiling=16)
        stub = _stub(sa, PER_REQ_MIB[2], spec=False)
        demand = ModelRunnerKVCacheMixin._auto_mamba_demand_size(stub, RATIO)
        size, _rest = _size_and_rest(sa, PER_REQ_MIB[2], REST_GB_3080, spec=False)
        self.assertEqual(size, demand)


class TestFitProperties(unittest.TestCase):
    def test_fit_never_exceeds_the_request(self):
        for rest_gb in (2.0, 4.0, 7.43, 12.0, 40.0):
            sa = _args(
                max_running_requests=8,
                max_running_requests_ceiling=64,
                speculative_num_draft_tokens=DRAFT_TOKENS,
            )
            size, _rest = _size_and_rest(sa, PER_REQ_MIB[2], rest_gb)
            self.assertLessEqual(size, 400, f"rest={rest_gb}")

    def test_fit_is_monotone_in_the_budget(self):
        sizes = []
        for rest_gb in (4.0, 6.0, 7.43, 9.0, 12.0):
            sa = _args(
                max_running_requests=8,
                max_running_requests_ceiling=64,
                speculative_num_draft_tokens=DRAFT_TOKENS,
            )
            size, _rest = _size_and_rest(sa, PER_REQ_MIB[2], rest_gb)
            sizes.append(size)
        self.assertEqual(sizes, sorted(sizes))

    def test_a_budget_that_holds_nothing_still_fails_loudly(self):
        sa = _args(
            max_running_requests=8,
            max_running_requests_ceiling=64,
            speculative_num_draft_tokens=DRAFT_TOKENS,
        )
        with self.assertRaises(RuntimeError) as ctx:
            _size_and_rest(sa, PER_REQ_MIB[2], 0.05)
        self.assertIn("max_mamba_cache_size", str(ctx.exception))


class TestRankUniformity(unittest.TestCase):
    """The fit is a rank-LOCAL decision; the value the scheduler acts on must
    still be one number for the whole group -- the known collective trap."""

    def _fitted_per_rank(self, ceiling: int):
        sizes = []
        for rank, per_req_mib in enumerate(PER_REQ_MIB):
            rest_gb = REST_GB_5090 if rank == 0 else REST_GB_3080
            sa = _args(
                max_running_requests=8,
                max_running_requests_ceiling=ceiling,
                speculative_num_draft_tokens=DRAFT_TOKENS,
            )
            size, _rest = _size_and_rest(sa, per_req_mib, rest_gb)
            sizes.append(size)
        return sizes

    def test_ranks_disagree_locally_and_the_min_reduce_unifies_them(self):
        sizes = self._fitted_per_rank(64)
        self.assertGreater(
            len(set(sizes)), 1, "the constellation must be genuinely uneven"
        )
        agreed = min(sizes)
        # Every rank ends on the agreed size, so every rank derives the same
        # admission ceiling -- no desync in the throttle/retract controller.
        resolved = []
        for _rank in range(len(sizes)):
            sa = _args(
                max_running_requests=8,
                max_running_requests_ceiling=64,
                speculative_num_draft_tokens=DRAFT_TOKENS,
            )
            sa.override("test.min_reduce", max_mamba_cache_size=agreed)
            resolved.append(_resolve_reqs(sa))
        self.assertEqual(len(set(resolved)), 1)
        self.assertEqual(resolved[0], agreed // RATIO)

    def test_the_admission_limiter_floats_below_the_fitted_ceiling(self):
        from sglang.srt.managers.admission_limiter import (
            AdmissionLimiter,
            resolve_admission_start,
        )

        sa = _args(
            max_running_requests=8,
            max_running_requests_ceiling=64,
            speculative_num_draft_tokens=DRAFT_TOKENS,
        )
        size, _rest = _size_and_rest(sa, PER_REQ_MIB[2], REST_GB_3080)
        sa.override("test.min_reduce", max_mamba_cache_size=size)
        fitted = _resolve_reqs(sa)
        self.assertLess(fitted, 64)
        start = resolve_admission_start(
            fitted, sa.max_running_requests_start, dp_size=1, floor=sa.admission_floor
        )
        limiter = AdmissionLimiter(
            fitted, start, floor=min(sa.admission_floor, fitted), auto=True
        )
        self.assertEqual(limiter.ceiling, fitted)
        self.assertEqual(limiter.current, 8)
        self.assertLessEqual(limiter.current, limiter.ceiling)

    def test_a_start_above_the_fitted_ceiling_is_clamped_not_desynced(self):
        from sglang.srt.managers.admission_limiter import resolve_admission_start

        self.assertEqual(resolve_admission_start(18, 64, dp_size=1, floor=1), 18)


class TestThrottleBeforeRetractOrder(unittest.TestCase):
    """#287's ordering is untouched by the fit: the limiter still throttles
    before the retraction fallback, and still on replicated inputs."""

    def test_throttle_then_release_hysteresis_unchanged(self):
        from sglang.srt.managers.admission_limiter import AdmissionLimiter

        limiter = AdmissionLimiter(
            18,
            18,
            floor=1,
            throttle_high=0.30,
            release_low=0.10,
            release_hysteresis=8,
            auto=True,
        )
        limiter.observe(0.9, 18)
        self.assertLess(limiter.current, 18, "pressure must throttle first")
        throttled = limiter.current
        for _ in range(7):
            limiter.observe(0.0, throttled)
        self.assertEqual(limiter.current, throttled, "release waits out hysteresis")
        limiter.observe(0.0, throttled)
        self.assertGreater(limiter.current, throttled)

    def test_pressure_input_stays_replicated(self):
        from sglang.srt.managers.admission_limiter import replicated_pool_usage

        # Same replicated inputs on every rank -> same verdict on every rank.
        verdicts = {replicated_pool_usage(1000, 4000) for _ in range(3)}
        self.assertEqual(len(verdicts), 1)


class TestHonestRefusalMessage(unittest.TestCase):
    """The paths that are not auto-fitted (pinned --max-mamba-cache-size, the
    fixed-fraction split) still refuse -- but they now name the ceiling that
    the budget would carry."""

    @staticmethod
    def _message(ceiling, mamba_gb=6.0, short_gb=-0.55):
        return ModelRunnerKVCacheMixin.budget_exhausted_message(
            tp_rank=2,
            budget_mib=16280,
            budget_gb=15.90,
            posts=[("weights + runtime state", 8.47), (MAMBA_BUDGET_POST, mamba_gb)],
            rest_memory_gb=short_gb,
            device_free_gb=15.90,
            occupancy=(20.0, 0.0),
            ceiling=ceiling,
        )

    def test_the_message_names_a_carriable_ceiling(self):
        msg = self._message(64)
        self.assertIn("--max-running-requests-ceiling=64", msg)
        self.assertIn("carries a ceiling of about", msg)
        # 64 * (6.00 - 0.55) / 6.00 = 58
        self.assertIn("about 58", msg)

    def test_the_message_is_unchanged_without_a_ceiling(self):
        self.assertNotIn("carries a ceiling", self._message(None))

    def test_a_hopeless_budget_says_so(self):
        self.assertIn(
            "even at one request", self._message(64, mamba_gb=0.5, short_gb=-9.0)
        )


class TestCardProofMarkerCoupling(unittest.TestCase):
    """The card proof (scripts/gpu_battery/s307_ceiling_fit.py) scores the
    boot log with literal markers. A rename in an emitter that is not carried
    into the consumer scores 0 hits and reads as "the feature did not fire",
    so the coupling is measured here rather than assumed."""

    @staticmethod
    def _verdict_module():
        path = (
            Path(__file__).resolve().parents[4]
            / "scripts"
            / "gpu_battery"
            / "s307_ceiling_fit.py"
        )
        spec = importlib.util.spec_from_file_location("s307_verdict", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _source(*parts):
        return (Path(__file__).resolve().parents[4] / Path(*parts)).read_text()

    def test_every_marker_exists_in_its_emitter(self):
        mod = self._verdict_module()
        mixin = self._source(
            "python",
            "sglang",
            "srt",
            "model_executor",
            "model_runner_kv_cache_mixin.py",
        )
        sched = self._source("python", "sglang", "srt", "managers", "scheduler.py")
        for marker, source, name in (
            (mod.M_FIT, mixin, "mixin"),
            (mod.M_POOL, mixin, "mixin"),
            (mod.M_LEDGER, mixin, "mixin"),
            (mod.M_SCHED, sched, "scheduler"),
            (mod.M_ADMIT, sched, "scheduler"),
        ):
            self.assertIn(marker, source, f"{marker!r} is not emitted by {name}")

    def test_the_fitted_ceiling_regex_matches_a_real_emission(self):
        mod = self._verdict_module()
        rendered = (
            "Dynamic admission limit: the requested ceiling 64 (per worker) "
            "does not fit the memory budget; the state pools and the float "
            "were fitted to 14. Raise the per-rank budget"
        )
        self.assertEqual(mod._FITTED_RE.findall(rendered), ["14"])
        self.assertEqual(mod._REQUESTED_RE.findall(rendered), ["64"])
        self.assertIn(mod.M_SCHED, rendered)

    def test_the_slot_regex_matches_the_pool_line(self):
        mod = self._verdict_module()
        rendered = (
            "[auto-mamba] demand-driven mamba pool: target_concurrency=16 "
            "ratio=5 safety=1.25 -> max_mamba_cache_size=100 slots (1.83 GB "
            "@ per_req=18.70 MiB; fit_cap=204) -> admits ~20 reqs;"
        )
        self.assertEqual(mod._SLOTS_RE.findall(rendered), ["100"])


if __name__ == "__main__":
    unittest.main()
