"""Form B (27B dense, 29.09.): the weightless-KV lane with a head SET W.

Form B is W = {5090, 3080 x8} as TP2 with a weight share per flag (77/23) and the
3080 x4 as a pure KV donor (weight share 0), the KV over all three ranks by
uneven DCP. It is the SAME mechanism as Form A + KV-only (the lane): the lane's
one head becomes the head set W named by ``--rank-tp-ratio``. What this pins,
starting from the flags a launch line carries:

  1. the parser admits ``--weightless-kv-fastlane --rank-tp-ratio 77,23,0`` as
     Form B (W = {0, 1}, lead 0, shares 77/23) and refuses the wrong spellings
     by name;
  2. the weight cut per rank: W builds as one of |W| with the W-restricted
     plan (27B: q heads 18/6, kv heads 3/1, MLP and GDN by 77/23), K builds
     nothing (refused under the W build context, it is the lane worker);
  3. the KV window per rank (uneven DCP, --uneven-token-vector): the owner rows
     every rank publishes are the rows its lane backend writes;
  4. the lane wire with two heads: the ONE installer puts W (not the lead
     alone) into the process, on every rank, and the lane's sources (token,
     accept, draft solo) all name the lead.
"""

import os
import pickle
import socket
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

RATIO = [77, 23, 0]
TOKENS = "1,2,2"
# Qwen3.8-27B (NVFP4-RadixArk config.json)
H, KV, HIDDEN, INTER, GDN_V, GDN_K = 24, 4, 5120, 17408, 48, 16


def _args(**kw):
    from sglang.srt.server_args import ServerArgs

    kw.setdefault("enable_vram_ledger", False)
    kw.setdefault("dcp_size", 3)
    kw.setdefault("weightless_kv_fastlane", True)
    kw.setdefault("rank_tp_ratio", list(RATIO))
    kw.setdefault("uneven_token_vector", TOKENS)
    a = ServerArgs(model_path="dummy", tp_size=3, **kw)
    a._handle_uneven_tp()
    return a


@pytest.fixture(autouse=True)
def _clean_process_state():
    from sglang.srt.distributed import utils as du

    yield
    du.set_weightless_kv_head_rank(None)
    du.set_tp_partition_ratios(None)
    du.set_cp_token_ratios(None)


# ---------------------------------------------------------------- 1. parser --
def test_parser_admits_form_b_on_the_lane_with_its_head_set():
    a = _args()
    assert a.form_b_active()
    assert a.form_b_model_tp_partition() == [[0, 1], [2]]
    assert a.rank_tp_ratio == RATIO                       # the base plan stays (W shares + K's 0)
    assert a.weightless_kv_fastlane and a.weightless_kv_head_rank == 0
    assert getattr(a, "_lane_rank_tp_ratio", None) is None  # not the one-head lane admission
    assert not a.form_a_active()
    from sglang.srt import rank_form as rf

    form = rf.resolve_rank_form(RATIO, [1, 2, 2])
    assert (form.kind, form.backend, form.head_rank) == (rf.FORM_B, rf.BACKEND_LANE, 0)
    assert form.serve_argv() == ["--tp-size", "3", "--dcp-size", "3", "--weightless-kv-fastlane",
                                 "--weightless-kv-head-rank", "0", "--rank-tp-ratio", "77,23,0",
                                 "--uneven-token-vector", "1,2,2"]


def test_parser_keeps_the_one_head_lane_and_refuses_wrong_spellings_by_name():
    lane = _args(rank_tp_ratio=[1, 0, 0])                 # Form A + KV-only: unchanged
    assert not lane.form_b_active() and lane._lane_rank_tp_ratio == [1, 0, 0]
    assert lane.rank_tp_ratio is None
    with pytest.raises(ValueError, match="head set W.*--weightless-kv-fastlane"):
        _args(weightless_kv_fastlane=False)                 # B rides the lane, named
    with pytest.raises(ValueError, match="--weightless-kv-head-rank 2.*lead 0"):
        _args(weightless_kv_head_rank=2)                     # the head rank IS the lead
    b = _args(rank_tp_ratio=[0, 3, 1], weightless_kv_head_rank=1)   # W = {1, 2}, lead 1
    assert b.form_b_model_tp_partition() == [[1, 2], [0]]
    with pytest.raises(ValueError, match="token share 0"):
        _args(uneven_token_vector="1,2,0")                  # a KV donor holding nothing
    with pytest.raises(ValueError, match="dcp-size 1 != --tp-size 3"):
        _args(dcp_size=1)


def test_f15_is_wired_in_the_registry():
    from sglang.srt import rank_role

    assert rank_role.SEAMS["F15"].wired is True
    assert rank_role.UNWIRED_ORDER == ()


# ------------------------------------------------------- 2. weight cut / rank --
def test_weight_cut_per_rank_follows_the_share_flag():
    from sglang.srt import rank_form as rf
    from sglang.srt.distributed import utils as du

    a = _args()
    part = a.form_b_model_tp_partition()
    du.set_tp_partition_ratios(a.rank_tp_ratio, allow_zero=True)   # what the scheduler installs
    cuts = {}
    for r in (0, 1):
        override, base, fams = rf.form_b_build_override(part, r, du.get_tp_partition_ratios(), {})
        assert override == dict(tp_size=2, tp_rank=r, attn_tp_size=2, attn_tp_rank=r)
        assert base == [77, 23]
        with du.scoped_tp_partition_ratios(base):
            cuts[r] = dict(
                q=du.tp_partition_sizes(H, 2, units=du.attn_q_partition_units(H, KV, 2),
                                        groups=du.attn_q_partition_groups(KV, 2))[r],
                kv=du.tp_partition_sizes(KV, 2, units=KV)[r],
                gdn_v=du.tp_partition_sizes(GDN_V, 2, units=GDN_K)[r],
            )
    assert (cuts[0]["q"], cuts[1]["q"]) == (18, 6)
    assert (cuts[0]["kv"], cuts[1]["kv"]) == (3, 1)
    assert cuts[0]["gdn_v"] + cuts[1]["gdn_v"] == GDN_V and cuts[0]["gdn_v"] > cuts[1]["gdn_v"]
    # K holds no weight: under the W build context it is refused by name -- it
    # builds the lane worker's meta model instead (model_runner)
    with pytest.raises(rf.RankFormKvRankBuild, match="lane worker"):
        rf.form_b_build_override(part, 2, du.get_tp_partition_ratios(), {})


# ------------------------------------------------------ 3. KV window / rank --
@contextmanager
def _rank(r):
    par = SimpleNamespace(tp_rank=r, attn_dcp_rank=r, attn_dcp_size=3)
    with mock.patch("sglang.srt.runtime_context.get_parallel", lambda: par):
        yield


def test_kv_window_per_rank_is_the_uneven_dcp_cut(monkeypatch):
    from sglang.srt import rank_form as rf
    from sglang.srt.distributed import utils as du
    from sglang.srt.managers.cache_controller import canonical_kv_owner_rows_for

    a = _args()
    assert a.uneven_weighted_dcp_enabled()
    monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", TOKENS)   # the flag's export (server_args)
    vec = du.resolve_cp_token_ratios(a)
    assert vec == [1, 2, 2]
    du.set_cp_token_ratios(vec)
    du.set_tp_partition_ratios(a.rank_tp_ratio, allow_zero=True)
    rf.install_weightless_heads(a)
    rows = []
    for r in range(3):
        with _rank(r):
            S, lo, hi = du.uneven_dcp_owner_bounds()
            p = du.cp_token_prefix(3)
            assert (S, lo, hi) == (p[-1], p[r], p[r + 1])          # == the lane backend's cp_lo/cp_hi
            page, S2, lo2, hi2 = canonical_kv_owner_rows_for((S, lo, hi), 64, canonical_kv_page=object())
            assert (page, S2, lo2, hi2) == (64, 5, lo, hi)
            rows.append((lo, hi))
    assert rows == [(0, 1), (1, 3), (3, 5)]   # 5090 1/5, 3080 x8 2/5, 3080 x4 2/5 of every page


def test_kv_token_vector_on_b_needs_every_rank_positive(monkeypatch):
    from sglang.srt.distributed import utils as du

    a = _args()
    monkeypatch.setenv("SGLANG_UNEVEN_TOKEN_VECTOR", "0,1,1")
    with pytest.raises(ValueError, match="POSITIVE"):
        du.resolve_cp_token_ratios(a)


# ------------------------------------------------ 4. the lane wire, 2 heads --
def test_one_installer_puts_the_head_set_not_the_lead_alone():
    """The scheduler used to install W and THEN the lane's single head, which
    overwrote W with (lead,) -- W1 would have come up as a weightless worker."""
    import inspect

    from sglang.srt import rank_form as rf
    from sglang.srt.distributed import utils as du
    from sglang.srt.managers import scheduler as sch

    assert rf.install_weightless_heads(_args()) == ((0, 1), (77, 23))
    assert du.get_weightless_kv_weight_ranks() == (0, 1)
    assert du.get_weightless_kv_head_rank() == 0
    assert du.weightless_head_counts(H, 3, units=du.attn_q_partition_units(H, KV, 2),
                                     groups=du.attn_q_partition_groups(KV, 2)) == [18, 6, 0]
    assert du.weightless_dcp_head_counts(H, KV, 3) == ([18, 6, 0], [3, 1, 0])
    assert rf.install_weightless_heads(_args(rank_tp_ratio=[1, 0, 0])) == ((0,), None)
    assert du.get_weightless_kv_weight_ranks() == (0,)
    assert rf.install_weightless_heads(SimpleNamespace(weightless_kv_fastlane=False)) is None
    src = inspect.getsource(sch.configure_scheduler_process)
    assert "install_weightless_heads(server_args)" in src
    assert "set_weightless_kv_head_rank(" not in src and "set_weightless_kv_weight_ranks(" not in src


def _run(rank, port, out, args_blob):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      RANK=str(rank), WORLD_SIZE="3", LOCAL_RANK=str(rank))
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.distributed import utils as du

    a = pickle.loads(args_blob)
    _orig = ps.init_model_parallel_group
    ps.init_model_parallel_group = lambda *x, **k: _orig(*x, **{
        **k, "use_pynccl": False, "use_custom_allreduce": False,
        "use_mscclpp_allreduce": False, "use_torch_symm_mem_allreduce": False})
    ps.init_distributed_environment(world_size=3, rank=rank, local_rank=rank,
                                    distributed_init_method=f"tcp://127.0.0.1:{port}",
                                    backend="gloo")
    # the order of model_runner / scheduler: partition, groups, plan, heads
    ps.set_model_tp_partition(a.form_b_model_tp_partition())
    ps.initialize_model_parallel(tensor_model_parallel_size=3, backend="gloo")
    du.set_tp_partition_ratios(a.rank_tp_ratio, allow_zero=True)
    from sglang.srt import rank_form as rf

    rf.install_weightless_heads(a)
    from sglang.srt.managers import tp_worker as tw
    from sglang.srt.speculative import form_b_spec as fbs
    from sglang.srt.utils import broadcast_pyobj

    res = {"head": du.is_weightless_head_rank(rank), "worker": du.weightless_worker_rank(rank),
           "set": du.get_weightless_kv_weight_ranks(), "token_src": tw._wl_token_src(a),
           "model_tp": list(ps.get_model_tp_group().ranks),
           "spec_active": fbs.form_b_spec_active(), "spec_lead": fbs.form_b_lead()}
    # the token takeover on the world cpu group: the lead sends, both others adopt
    mine = [100 + rank, 200 + rank]
    got = broadcast_pyobj(mine if rank == res["token_src"] else [], rank,
                          ps.get_world_group().cpu_group, src=res["token_src"])
    res["adopted"] = list(got)
    with open(os.path.join(out, f"{rank}.pkl"), "wb") as f:
        pickle.dump(res, f)
    du.set_weightless_kv_head_rank(None)
    du.set_tp_partition_ratios(None)
    ps.destroy_model_parallel()
    ps.set_model_tp_partition(None)
    ps.destroy_distributed_environment()


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_lane_wire_with_two_heads_on_three_ranks():
    a = _args()
    d = tempfile.mkdtemp()
    mp.spawn(_run, args=(_port(), d, pickle.dumps(a)), nprocs=3, join=True)
    res = [pickle.load(open(os.path.join(d, f"{r}.pkl"), "rb")) for r in range(3)]
    assert [(r["head"], r["worker"]) for r in res] == [(True, False), (True, False), (False, True)]
    for r in res:
        assert r["set"] == (0, 1) and r["token_src"] == 0 and r["spec_lead"] == 0
        assert r["spec_active"] and r["adopted"] == [100, 200]
    assert res[0]["model_tp"] == res[1]["model_tp"] == [0, 1] and res[2]["model_tp"] == [2]
