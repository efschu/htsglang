# SPDX-License-Identifier: Apache-2.0
"""#239 S3a -- the planner's token cut reaches the runtime under Form A.

Before S3a the planner could price a token cut of the full-attention KV
(form ``kv=qsa_forma_dcp``, S1/S2/S2b), but nothing carried it to D:
``rank_role.resolve_dcp_under_host_kv`` knew only "dcp 1" and had no
caller, the uneven-TP auto-engage would have turned an inherited
SGLANG_UNEVEN_DCP=1 into a token-sharded pool nobody planned, and every
sizing step divided by the rank's share -- which the optimum sets to 0 on
the attention host at x1/x2 ([0,46,18] / [0,37,27], S2b dry run).

Pinned here, stage by stage of the coupling:
  resolve   (3, vector) for a cut that gives a worker tokens; dcp 1 otherwise
  server    Form A decides its DCP before placement, the auto-engage stays off
  install   the env vector may carry a 0 under Form A (and only there)
  pools     rows = share x T (whole owner blocks), 0 rows for share 0
  cell      a share-0 rank is priced per GLOBAL token (index + draft)
  sizing    a share-0 rank bounds C by its global capacity, or not at all
  launcher  the cut goes to D as --uneven-token-vector ... role pin
"""

import inspect
import os
import types
from unittest import mock

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt import rank_role  # noqa: E402
from sglang.srt.distributed import utils as dutils  # noqa: E402
from sglang.srt.layers.dcp.owner import dcp_compact_pool_rows  # noqa: E402

PLAN = rank_role.RankRolePlan(("host", "worker", "worker"))


@pytest.fixture
def clean():
    saved = dutils.get_cp_token_ratios()
    yield
    dutils.set_cp_token_ratios(saved)
    rank_role.set_form_a_role_plan(None, 0)


# ---- resolve -------------------------------------------------------------------


@pytest.mark.parametrize(
    "vec,reduced",
    [([0, 46, 18], (0, 23, 9)), ([0, 37, 27], (0, 37, 27)), ([4, 31, 29], (4, 31, 29)),
     ([21, 21, 22], (21, 21, 22))],
)
def test_a_cut_that_gives_a_worker_tokens_is_dcp_over_the_group(vec, reduced):
    res = rank_role.resolve_dcp_under_host_kv(PLAN, None, None, token_vector=vec)
    assert res.dcp_size == 3
    assert res.uneven_dcp_kv_replicated is True
    assert res.token_vector == reduced


def test_a_cut_that_leaves_the_workers_empty_is_classic_form_a():
    res = rank_role.resolve_dcp_under_host_kv(PLAN, None, None, token_vector=[64, 0, 0])
    assert (res.dcp_size, res.token_vector) == (1, None)
    assert rank_role.resolve_dcp_under_host_kv(PLAN, None, None).dcp_size == 1


@pytest.mark.parametrize("vec", [[1, 2], [0, 0, 0], [-1, 40, 25], ["a", 1, 1]])
def test_a_malformed_cut_is_refused(vec):
    with pytest.raises(rank_role.RankRoleError):
        rank_role.resolve_dcp_under_host_kv(PLAN, None, None, token_vector=vec)


def test_an_explicit_dcp_size_that_contradicts_the_cut_is_refused():
    with pytest.raises(rank_role.RankRoleError):
        rank_role.resolve_dcp_under_host_kv(PLAN, 2, None, token_vector=[0, 46, 18])
    assert rank_role.resolve_dcp_under_host_kv(
        PLAN, 3, None, token_vector=[0, 46, 18]).dcp_size == 3


# ---- server_args ---------------------------------------------------------------


def _sa(**kw):
    from sglang.srt.server_args import ServerArgs

    class _S:
        _resolve_form_a_dcp = ServerArgs._resolve_form_a_dcp
        form_a_dcp_vector = ServerArgs.form_a_dcp_vector
        uneven_weighted_dcp_enabled = ServerArgs.uneven_weighted_dcp_enabled
        uneven_kv_flag_active = ServerArgs.uneven_kv_flag_active

    s = _S()
    s.rank_role = ["host", "worker", "worker"]
    s.uneven_token_vector = None
    s.dcp_size = 1
    s.rank_kv_ratio = "coupled"
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_server_args_resolves_the_cut_to_dcp_3(monkeypatch):
    monkeypatch.delenv("SGLANG_UNEVEN_TOKEN_VECTOR", raising=False)
    monkeypatch.setenv("SGLANG_UNEVEN_DCP_WEIGHTED", "0")
    s = _sa(uneven_token_vector="0,46,18")
    s._resolve_form_a_dcp()
    assert s.dcp_size == 3
    assert s.form_a_dcp_vector() == [0, 23, 9]
    # the weighted owner rule installs the vector (scheduler gate)
    assert s.uneven_weighted_dcp_enabled() is True


def test_server_args_without_a_cut_keeps_dcp_1(monkeypatch):
    monkeypatch.delenv("SGLANG_UNEVEN_TOKEN_VECTOR", raising=False)
    monkeypatch.setenv("SGLANG_UNEVEN_DCP", "1")  # inherited, not planned
    s = _sa()
    s._resolve_form_a_dcp()
    assert s.dcp_size == 1 and s.form_a_dcp_vector() is None


def test_form_a_decides_dcp_before_placement_and_the_auto_engage_stays_off():
    from sglang.srt import server_args as sa_mod

    src = inspect.getsource(sa_mod.ServerArgs._handle_uneven_tp)
    call = src.index("self._resolve_form_a_dcp()")
    early = src.index("if self.rank_gpu_id is None:\n            self._handle_uneven_mlp_ratio()")
    assert call < early
    engage = src.index('"Uneven DCP: auto-set dcp_size')
    assert "and not self.rank_role" in src[engage - 600:engage]


# ---- install -------------------------------------------------------------------


def _resolve_args(**kw):
    a = types.SimpleNamespace(rank_tp_ratio=[1, 0, 0], dcp_size=3,
                              rank_role=["host", "worker", "worker"],
                              uneven_token_vector_role="pin",
                              uneven_token_vector_provenance="planner-239-token-cut")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_the_env_vector_may_carry_a_zero_under_form_a(monkeypatch):
    monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", "0,46,18")
    monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR_ROLE", "pin")
    assert dutils.resolve_cp_token_ratios(_resolve_args()) == [0, 23, 9]


def test_outside_form_a_a_zero_is_still_refused(monkeypatch):
    monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", "0,46,18")
    with pytest.raises(ValueError):
        dutils.resolve_cp_token_ratios(_resolve_args(rank_role=None, rank_tp_ratio=[2, 1, 1]))


# ---- pools ---------------------------------------------------------------------


def test_pool_rows_are_the_share_of_the_context(clean):
    rank_role.set_form_a_role_plan(PLAN, 0)
    T, S = 262144, 32
    for share in (23, 9):
        rows = dcp_compact_pool_rows(T, S, share)
        assert T * share // S <= rows < T * share // S + share + 1
        assert rows % share == 0  # whole owner blocks, page-aligned per block
    assert dcp_compact_pool_rows(T, S, 0) == 0


def test_outside_form_a_a_zero_share_pool_is_still_refused(clean):
    with pytest.raises(ValueError):
        dcp_compact_pool_rows(262144, 32, 0)


# ---- cell ----------------------------------------------------------------------


def _mr(rank):
    return types.SimpleNamespace(dcp_size=3, tp_rank=rank)


@pytest.mark.parametrize("rank,factor,qsa,want", [(0, 1.0, 768, 768 + 1087), (1, 0.0, 0, 0)])
def test_a_zero_share_rank_is_priced_per_global_token(clean, rank, factor, qsa, want):
    from sglang.srt.model_executor import pool_configurator as pc

    dutils.set_cp_token_ratios([0, 23, 9] if rank == 0 else [23, 0, 9])
    with mock.patch.object(pc, "get_parallel",
                           return_value=types.SimpleNamespace(attn_dcp_rank=rank)), \
            mock.patch.object(pc, "solo_draft_kv_cell_factor", return_value=factor):
        cell = pc.zero_token_share_cell(_mr(rank), 12288 + qsa, 12288 + qsa + 1087, qsa)
    assert cell == want


def test_a_rank_with_a_share_keeps_its_cell(clean):
    from sglang.srt.model_executor import pool_configurator as pc

    dutils.set_cp_token_ratios([0, 23, 9])
    with mock.patch.object(pc, "get_parallel",
                           return_value=types.SimpleNamespace(attn_dcp_rank=1)):
        assert pc.zero_token_share_cell(_mr(1), 12288, 12288, 0) is None


def test_the_draft_host_with_share_0_keeps_its_draft(clean):
    from sglang.srt.model_executor import pool_configurator as pc

    dutils.set_cp_token_ratios([0, 23, 9])
    sa = types.SimpleNamespace(speculative_draft_solo_active=lambda: True,
                               speculative_draft_solo_rank=lambda: 0,
                               speculative_cross_algorithm=False)
    mr = types.SimpleNamespace(server_args=sa, is_draft_worker=False, tp_rank=0, dcp_size=3)
    with mock.patch.object(pc, "get_parallel",
                           return_value=types.SimpleNamespace(attn_dcp_rank=0)):
        assert pc.solo_draft_kv_cell_factor(mr) == 1.0


def test_the_configurator_applies_the_zero_share_cell():
    from sglang.srt.model_executor import pool_configurator as pc

    src = inspect.getsource(pc.DefaultPoolConfigurator.__init__)
    assert "zero_token_share_cell(" in src
    assert src.index("apply_window_pool_draft_charge(") < src.index("zero_token_share_cell(")


# ---- sizing --------------------------------------------------------------------


def _constraints(ratio_vec, rank, capacity, other_unit, cell):
    from sglang.srt.model_executor import model_runner_kv_cache_mixin as mixin

    dutils.set_cp_token_ratios(ratio_vec)
    stub = types.SimpleNamespace(
        server_args=types.SimpleNamespace(max_total_tokens=None,
                                          uneven_memory_budgets_active=lambda: False),
        pp_size=1, dcp_size=3, tp_rank=rank,
        _hybrid_kv_token_cap=lambda: None, _swa_hybrid_kv_token_cap=lambda: None,
        _apply_hybrid_kv_token_cap=lambda tc, cap, kind="mamba": tc,
    )

    def fake_all_reduce(t, op=None, group=None):
        t.copy_(torch.minimum(t, torch.tensor(other_unit, dtype=t.dtype)))

    grp = types.SimpleNamespace(world_size=3, cpu_group=object())
    with mock.patch.object(mixin, "get_world_group", return_value=grp), \
            mock.patch.object(mixin, "get_parallel",
                              return_value=types.SimpleNamespace(attn_dcp_rank=rank)), \
            mock.patch("torch.distributed.all_reduce", side_effect=fake_all_reduce):
        return mixin.ModelRunnerKVCacheMixin._apply_token_constraints(
            stub, capacity, kv_cell_bytes=cell)


def test_the_host_at_share_0_bounds_c_by_its_global_capacity(clean):
    # S = 32; the host holds index + draft for 300000 global tokens -> unit 9375
    assert _constraints([0, 23, 9], 0, 300_000, 20_000, 1855) == 9375 * 32


def test_a_worker_at_share_0_bounds_nothing(clean):
    # [23,0,9]: rank 1 has no cell at all -> the other ranks' unit decides
    assert _constraints([23, 0, 9], 1, 1 << 20, 20_000, 0) == 20_000 * 32


def test_a_rank_with_a_share_keeps_the_classic_unit(clean):
    assert _constraints([0, 23, 9], 1, 23 * 12_000, 20_000, 12288) == 12_000 * 32


def test_the_calibration_leaves_the_planned_cut_alone():
    from sglang.srt.model_executor import model_runner_kv_cache_mixin as mixin

    src = inspect.getsource(mixin.ModelRunnerKVCacheMixin._maybe_suggest_dcp_token_vector)
    i = src.index('getattr(self.server_args, "rank_role", None)')
    assert i < src.index("active = get_cp_token_ratios()")


# ---- launcher ------------------------------------------------------------------


def test_the_launcher_turns_the_cut_into_a_vector():
    from sglang.srt.weg2 import launcher as L

    assert L.d_token_cut_vector((0, 46, 18), "joint") == (0, 46, 18)
    assert L.d_token_cut_vector((), (1.0, 2.0, 1.0)) == (16, 32, 16)
    assert L.d_token_cut_vector((), (1.0, 0.0, 1.0)) == (32, 0, 32)
    assert L.d_token_cut_vector((), None) is None


def test_the_launcher_ships_the_cut_to_d_once():
    from sglang.srt.weg2 import launcher as L

    ns = types.SimpleNamespace(d_uneven_token_vector=None, d_kv_token_cut="joint",
                               extra_d="--foo 1 --uneven-token-vector 9,9,9 "
                                       "--uneven-token-vector-role=seed")
    line = L.publish_d_token_cut(ns, (0, 46, 18), "D")
    toks = ns.extra_d.split()
    assert toks.count("--uneven-token-vector") == 1
    assert toks[toks.index("--uneven-token-vector") + 1] == "0,46,18"
    assert toks[toks.index("--uneven-token-vector-role") + 1] == "pin"
    assert "--uneven-token-vector-role=seed" not in toks and toks[:2] == ["--foo", "1"]
    assert "(#239 S3a)" in line
    assert L.publish_d_token_cut(ns, None, "D") is None


def test_an_operator_vector_beside_the_cut_is_refused():
    from sglang.srt.weg2 import launcher as L

    ns = types.SimpleNamespace(d_uneven_token_vector="3,1,1", d_kv_token_cut="joint",
                               extra_d="")
    with pytest.raises(L.Weg2LaunchRefused):
        L.publish_d_token_cut(ns, (0, 46, 18), "D")


def test_the_d_solve_publishes_the_cut():
    from sglang.srt.weg2 import launcher as L

    src = inspect.getsource(L.log_d_rank_vram_solve)
    assert "publish_d_token_cut(" in src and "d_token_cut_vector(" in src
