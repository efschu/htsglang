"""fnFL2 v11 (20.09.): a Form A worker carries no GDN/mamba state
(per_req == 0). When its weights overrun the budget the ceiling fit divided
by a zero per-slot cost -> ZeroDivisionError killed TP1/TP2."""

import types

from sglang.srt.model_executor import model_runner_kv_cache_mixin as m


def _runner():
    self = types.SimpleNamespace(server_args=types.SimpleNamespace(mamba_full_memory_ratio=0.5, max_running_requests_ceiling=6, max_running_requests=6, max_running_requests_user_set=False))
    self._mamba_pool_budget_cost_gb = lambda wanted, per_req, ratio, D: wanted * per_req * (1.0 + D / max(ratio, 1)) / 2**30
    return self


def test_worker_without_state_keeps_the_ceiling_even_when_over_budget():
    self = _runner()
    fit = m.ModelRunnerKVCacheMixin._fit_mamba_pool_to_budget
    assert fit(self, 6, total_rest_memory=-0.3, reserve_gb=0.0, per_req=0, ratio=1, D=4) == 6
    assert fit(self, 6, total_rest_memory=0.1, reserve_gb=0.0, per_req=0, ratio=1, D=0) == 6


def test_host_with_state_still_fits_to_the_budget():
    self = _runner()
    fit = m.ModelRunnerKVCacheMixin._fit_mamba_pool_to_budget
    per_req = 50 * 2**20  # 50 MiB of state per request
    # 6 slots would cost 300 MiB, the mamba share is 0.5 GiB / 3 -> ~3 slots
    got = fit(self, 6, total_rest_memory=0.5, reserve_gb=0.0, per_req=per_req, ratio=1, D=0)
    assert 0 < got < 6
