"""The flashinfer backend's head geometry per rank on the weightless lane and Form B (29.09., boot 09291353).

Two deaths in the same run, both in the attention backend, both past a green parse:

  * a3_formb (TP 3: W = {0, 1} with 77/23, rank 2 KV-only): SIGFPE in
    flashinfer ``plan()`` on the untuned dummy forward of the warmup autotune.
    The prefill updater planned the RAGGED wrapper with the rank's "local" head
    counts from the global plan over attn_tp=3 -- (0, 1) on the KV-only rank
    (an integer division by zero in plan()), (18, 2) on the lead whose layers
    carry (18, 3).
  * a3_lane (TP 3: head 0, KV workers 1, 2; DFLASH solo on 0): "32 is not
    divisible by 3" -- the solo draft host's model is built under a weight-TP=1
    override, its attention backend was not, so the draft's 32 heads were split
    over the serving TP=3.

What this pins: every (q, kv) pair a rank's backend hands to plan() is > 0 and
is what the rank's layers carry; a rank with no local head does not plan the
ragged wrapper; the solo draft host's backend reads the TP=1 build geometry.
"""

import os
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_CACHE = "/spinning/llm_stuff/club-3090/models-cache"
_DENSE_27B = f"{_CACHE}/Qwen3.8-27B-NVFP4-RadixArk"
_DRAFT = f"{_CACHE}/Qwen3.8-27B-DFlash2-NVFP4-RTNcal"
_have = pytest.mark.skipif(
    not (os.path.isfile(f"{_DENSE_27B}/config.json") and os.path.isfile(f"{_DRAFT}/config.json")),
    reason="27B checkpoint / DFlash2 draft not on this box",
)


@pytest.fixture(autouse=True)
def _clean_process_state():
    from sglang.srt.distributed import utils as du

    yield
    du.set_weightless_kv_head_rank(None)
    du.set_tp_partition_ratios(None)
    du.set_cp_token_ratios(None)


def _configs():
    from sglang.srt.configs.model_config import ModelConfig

    target = ModelConfig(_DENSE_27B, trust_remote_code=True,
                         model_override_args='{"language_model_only": true}')
    draft = ModelConfig(_DRAFT, trust_remote_code=True, is_draft_model=True)
    return target, draft


def _install(form_b: bool):
    """What configure_scheduler_process installs for the two arms."""
    from sglang.srt.distributed import utils as du

    if form_b:
        du.set_tp_partition_ratios([77, 23, 0], allow_zero=True)
        du.set_weightless_kv_weight_ranks((0, 1), (77, 23))
    else:
        du.set_tp_partition_ratios(None)
        du.set_weightless_kv_head_rank(0)


def _plans(runner, r, uneven=True):
    from sglang.srt.layers.attention import flashinfer_backend as fb
    from sglang.srt.runtime_context import get_parallel

    with get_parallel().override(tp_size=3, tp_rank=r, attn_tp_size=3, attn_tp_rank=r,
                                 attn_dcp_size=3, attn_dcp_rank=r):
        return fb.flashinfer_plan_head_counts(runner, uneven)


def _assert_plannable(plans):
    for w, (q, kv) in plans.items():
        if w == "local":
            continue
        assert q > 0 and kv > 0 and q % kv == 0, (w, q, kv)


def test_ragged_plan_is_skipped_only_for_a_headless_uneven_dcp_rank():
    from sglang.srt.layers.attention.flashinfer_backend import ragged_plan_needed

    assert ragged_plan_needed(True, 0) is False     # the lane / Form B KV worker
    assert ragged_plan_needed(True, 6) is True      # a Form B weight rank
    assert ragged_plan_needed(False, 0) is True     # not DCP: the stock path, unchanged
    assert ragged_plan_needed(False, 8) is True


@_have
def test_form_b_every_plan_is_positive_and_matches_the_layers():
    target, _ = _configs()
    _install(form_b=True)
    runner = SimpleNamespace(model_config=target, is_draft_worker=False)
    p = {r: _plans(runner, r) for r in range(3)}
    for r in range(3):
        _assert_plannable(p[r])
    # the layers: W builds 18/3 and 6/1 (test_form_b_headset_lane_0929), K nothing
    assert p[0]["local"] == (18, 3) and p[0]["prefill_ragged"] == (18, 3)
    assert p[1]["local"] == (6, 1) and p[1]["prefill_ragged"] == (6, 1)
    assert p[2]["local"] == (0, 0) and "prefill_ragged" not in p[2]   # 09291353: was plan(0, 1) -> SIGFPE
    for r in range(3):   # the paged/decode wrappers read the gathered q + the full kv heads
        assert p[r]["decode"] == (24, 4) and p[r]["prefill_paged"] == (24, 4)


@_have
def test_lane_head_plans_its_real_heads_and_workers_plan_no_ragged():
    target, _ = _configs()
    _install(form_b=False)
    runner = SimpleNamespace(model_config=target, is_draft_worker=False)
    p = {r: _plans(runner, r) for r in range(3)}
    for r in range(3):
        _assert_plannable(p[r])
    assert p[0]["local"] == (24, 4) and p[0]["prefill_ragged"] == (24, 4)   # was (8, 1): the global even split
    for r in (1, 2):
        assert p[r]["local"] == (0, 0) and "prefill_ragged" not in p[r]


@_have
@pytest.mark.parametrize("form_b", [False, True])
def test_solo_draft_host_backend_reads_the_tp1_build_geometry(form_b):
    from sglang.srt.layers.attention import flashinfer_backend as fb
    from sglang.srt.model_executor.model_runner import draft_solo_host_geometry_ctx
    from sglang.srt.runtime_context import get_parallel

    _, draft = _configs()
    _install(form_b=form_b)
    host = SimpleNamespace(model_config=draft, is_draft_worker=True, is_draft_solo_host=True)
    with get_parallel().override(tp_size=3, tp_rank=0, attn_tp_size=3, attn_tp_rank=0,
                                 attn_dcp_size=3, attn_dcp_rank=0):
        # the global context the backend saw before: lane -> 32 % 3 raises,
        # Form B -> a 24/6 plan for a 32/8 draft
        if form_b:
            assert fb._local_attn_head_counts(host) == (24, 6)
        else:
            with pytest.raises(AssertionError, match="32 is not divisible by 3"):
                fb._local_attn_head_counts(host)
        with draft_solo_host_geometry_ctx(host):
            assert fb._local_attn_head_counts(host) == (32, 8)
            plans = fb.flashinfer_plan_head_counts(host, False)
    _assert_plannable(plans)
    assert plans["prefill_ragged"] == (32, 8) and plans["decode"] == (32, 8)


def test_runner_builds_its_attention_backend_in_the_draft_host_geometry(monkeypatch):
    from sglang.srt.model_executor import model_runner as mr
    from sglang.srt.runtime_context import get_parallel

    seen = {}

    def _record(self):
        seen[self.is_draft_solo_host] = (get_parallel().attn_tp_size, get_parallel().tp_size)

    monkeypatch.setattr(mr.ModelRunner, "_init_attention_backend_in_geometry", _record)
    with get_parallel().override(tp_size=3, tp_rank=0, attn_tp_size=3, attn_tp_rank=0):
        for is_host in (True, False):
            runner = object.__new__(mr.ModelRunner)
            runner.is_draft_solo_host = is_host
            mr.ModelRunner.init_attention_backend(runner)
    assert seen[True] == (1, 1)     # the solo draft host: the TP=1 build geometry
    assert seen[False] == (3, 3)    # every other runner: untouched
