"""fnFA6 (20.09. 12:44Z): a Form A worker sizes its pools with ZERO heads.
The replicated-KV geometry (kv < tp) handed every worker all kv heads and a
full-context KV pool; the mamba sizing asserted per-request state bytes > 0
although the worker's uneven share is 0."""

import inspect

import pytest

from sglang.srt import rank_role
from sglang.srt.configs import model_config as mc
from sglang.srt.distributed import utils as du
from sglang.srt.rank_role import HOST, WORKER, RankRolePlan, set_form_a_role_plan

FORM_A = RankRolePlan((HOST, WORKER, WORKER))


@pytest.fixture
def replicated_plan(monkeypatch):
    monkeypatch.setattr(du, "tp_plan_active", lambda tp: True)
    monkeypatch.setattr(du, "attn_kv_replicated", lambda tp, kv: True)
    yield
    set_form_a_role_plan(None)


def test_worker_reports_zero_kv_heads_host_keeps_them_all(replicated_plan):
    set_form_a_role_plan(FORM_A, rank=1)
    assert mc.ModelConfig._uneven_tp_num_kv_heads(2, 3, None) == 0
    assert mc.ModelConfig._uneven_tp_num_kv_heads(2, 3, 2) == 0
    assert mc.ModelConfig._uneven_tp_num_kv_heads(2, 3, 0) == 2
    set_form_a_role_plan(FORM_A, rank=0)
    assert mc.ModelConfig._uneven_tp_num_kv_heads(2, 3, None) == 2


def test_classic_boot_keeps_the_replicated_count(replicated_plan):
    set_form_a_role_plan(None)
    assert mc.ModelConfig._uneven_tp_num_kv_heads(2, 3, None) == 2
    assert mc.ModelConfig._uneven_tp_num_kv_heads(2, 3, 1) == 2


def test_mamba_sizing_admits_zero_state_bytes_on_a_worker_only():
    from sglang.srt.model_executor import model_runner_kv_cache_mixin as mix

    src = inspect.getsource(mix.ModelRunnerKVCacheMixin.handle_max_mamba_cache)
    assert "assert per_req > 0 or this_rank_is_form_a_worker()" in src
    set_form_a_role_plan(FORM_A, rank=1)
    try:
        assert rank_role.this_rank_is_form_a_worker()
    finally:
        set_form_a_role_plan(None)
    assert not rank_role.this_rank_is_form_a_worker()
