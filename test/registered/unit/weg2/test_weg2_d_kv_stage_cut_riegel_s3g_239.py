"""#239 S3g, the must-point found on 28.09. (M1 prep): under the token cut the
#251c stage form maps a KV-holding worker's pool at the TOP stage, untrimmed.

``kv_stage_pool_tokens`` sizes EVERY rank's KV pool at the top stage (the
virtual range the graphs keep) and only the attention host trims its tensors
to S0 (``kv_stage_trims_here``, rc12z11: a worker trims nothing). A Form A
worker used to hold byteless KV, so its top-stage pool cost nothing. Under the
cut the worker HOLDS full-attention KV: its compacted pool
(``_dcp_token_sharded_pool_rows(max_total)``) follows the top stage and is
mapped whole -- on M1a (cut 0/48/16) TP1 would map 262144 x 2 x 9984 B instead
of 262144 x 9984 B, +2.5 GiB on a 3080 the plan booked to its edge.

Until the stages run per KV rank (the rest of S3g), two named stops:
(1) the launcher writes no stage form under such a cut (its line says why);
(2) a KV-holding worker that still meets a stage form (an operator's env)
    refuses by name at the pool sizing, before anything is mapped.
Form A (no cut, byteless workers) is unchanged.
"""
from __future__ import annotations

import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt import rank_role  # noqa: E402
from sglang.srt.weg2 import d_seat_vram as dsv  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_weg2_d_kv_stage_launcher_251c as T251  # noqa: E402

STAGES = "64,128,192"


class _Plan:
    def __init__(self, workers):
        self._w = set(workers)

    def role_of(self, rank):
        return "worker" if rank in self._w else "host"

    def is_worker(self, rank):
        return rank in self._w


def teardown_function(_fn):
    rank_role.set_form_a_role_plan(None, 0)


def _armed(rank, bounds):
    from sglang.srt.environ import envs

    rank_role.set_form_a_role_plan(_Plan(workers={1, 2}), rank=rank)
    return [mock.patch.dict(os.environ, {dsv.GROUP_ENV: "D"}),
            envs.SGLANG_OPT_WEG2_D_SEAT_VRAM.override(True),
            envs.SGLANG_WEG2_D_KV_STAGE_TOKENS.override(STAGES),
            envs.SGLANG_WEG2_D_KV_STAGE_ROWS.override(32),
            mock.patch("sglang.srt.distributed.utils.uneven_dcp_owner_bounds",
                       lambda: bounds)]


def _enter(ctx):
    for c in ctx:
        c.__enter__()


def _exit(ctx):
    for c in reversed(ctx):
        c.__exit__(None, None, None)


def test_a_kv_holding_worker_refuses_the_top_stage_pool():
    """RED before: the worker's pool rows were the top stage's (192), mapped
    whole, with nothing trimming them."""
    ctx = _armed(rank=1, bounds=(64, 0, 48))  # TP1 owns 48 of every 64 tokens
    _enter(ctx)
    try:
        assert rank_role.form_a_worker_holds_kv()
        with pytest.raises(dsv.Weg2DSeatVramRefused, match="holds full-attention KV"):
            dsv.kv_stage_pool_tokens(10)
    finally:
        _exit(ctx)


def test_a_byteless_worker_keeps_the_hosts_rows():
    """Form A (no cut): the worker holds no KV -- unchanged, the top stage's
    rows in step with TP0's allocator (rc12z11)."""
    ctx = _armed(rank=1, bounds=(64, 0, 0))
    _enter(ctx)
    try:
        assert not rank_role.form_a_worker_holds_kv()
        assert dsv.kv_stage_pool_tokens(10) == 192
    finally:
        _exit(ctx)


def test_the_launcher_writes_no_stage_form_under_a_cut_with_worker_kv():
    """RED before: the form (tokens, rows, max by seats) went into --env-d
    under the cut exactly as in Form A."""
    ns = types.SimpleNamespace(env_d="SGLANG_MOE_SCRATCH_SLOTS=100,48,48")
    lines = L.apply_d_kv_stage_form(ns, T251.er, T251._rows(), T251.FORM, T251._plan(), "D",
                                    verify_tokens=4, top_k=10, kv_token_shares=(0, 48, 16))
    assert len(lines) == 1 and "entfaellt unter dem Token-Schnitt" in lines[0]
    assert "Rang 1,2" in lines[0] and "S3g" in lines[0]
    for key in L.D_KV_STAGE_KEYS:
        assert key not in ns.env_d
    assert ns.env_d == "SGLANG_MOE_SCRATCH_SLOTS=100,48,48"


def test_an_unsolved_cut_counts_as_worker_kv():
    assert L.kv_stage_cut_workers("owned") == [-1]
    assert L.kv_stage_cut_workers("owned:0,64,0") == [1]
    assert L.kv_stage_cut_workers((64, 0, 0)) == []
    assert L.kv_stage_cut_workers(None) == []


def test_form_a_still_gets_its_stage_form():
    ns = types.SimpleNamespace(env_d="SGLANG_MOE_SCRATCH_SLOTS=100,48,48")
    lines = L.apply_d_kv_stage_form(ns, T251.er, T251._rows(), T251.FORM, T251._plan(), "D",
                                    verify_tokens=4, top_k=10, kv_token_shares=None)
    assert L.D_KV_STAGE_KEYS[0] in ns.env_d and "entfaellt" not in lines[0]


def test_the_seat_table_hands_the_cut_to_the_stage_form():
    import inspect

    src = inspect.getsource(L.d_seat_table_lines)
    assert 'kv_token_shares=plan_kwargs.get("kv_token_shares")' in src
