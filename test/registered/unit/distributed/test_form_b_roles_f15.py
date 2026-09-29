"""F15 (5b): Form B (dense 27B) runs the weightless lane over the head SET W.

Three real processes over gloo with the Form B partition W={0,1}, K={2}, the
head set installed the way the scheduler does (rank_form.form_b_head_set ->
set_weightless_kv_weight_ranks). Checked per rank: the lane roles (W heads, K
worker), the head counts every rank plans its attention with (identical on all
ranks, 0 on K, = the W build), the token/accept source (the lead), the sampler's
sync group (model_tp: W, K alone), the tp_worker token takeover (every head but
the lead adopts the lead's tokens), and that a MoE model is refused by name.
"""

import os
import pickle
import socket
import tempfile
import types

import pytest
import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

RATIO = [77, 23, 0]


def _run(rank, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      RANK=str(rank), WORLD_SIZE="3", LOCAL_RANK=str(rank))
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.distributed import utils as du

    _orig = ps.init_model_parallel_group
    ps.init_model_parallel_group = lambda *a, **k: _orig(*a, **{
        **k, "use_pynccl": False, "use_custom_allreduce": False,
        "use_mscclpp_allreduce": False, "use_torch_symm_mem_allreduce": False})
    ps.init_distributed_environment(world_size=3, rank=rank, local_rank=rank,
                                    distributed_init_method=f"tcp://127.0.0.1:{port}",
                                    backend="gloo")
    part = [[0, 1], [2]]
    ps.set_model_tp_partition(part)
    ps.initialize_model_parallel(tensor_model_parallel_size=3, backend="gloo")
    from sglang.srt import rank_form as rf

    w, wr = rf.form_b_head_set(part, RATIO)                      # as the scheduler installs it
    du.set_weightless_kv_weight_ranks(w, wr)
    res = {"head": du.is_weightless_head_rank(rank), "worker": du.weightless_worker_rank(rank),
           "lead": du.get_weightless_kv_head_rank()}
    res["q_counts"] = du.weightless_head_counts(24, 3, units=du.attn_q_partition_units(24, 4, 2),
                                                groups=du.attn_q_partition_groups(4, 2))
    res["kv_counts"] = du.weightless_head_counts(4, 3, units=4)
    # F15 (5d): the GDN state pool is sized under the geometry the layers were
    # built with -- W: |W| ranks, W-restricted plan; K: the lane worker's 1 elem
    from sglang.srt.configs.qwen3_next import Qwen3NextConfig
    from sglang.srt.distributed.utils import set_tp_partition_ratios, tp_partition_sizes

    set_tp_partition_ratios(RATIO, allow_zero=True)
    cfg = Qwen3NextConfig(num_hidden_layers=4)
    cfg.full_attention_interval = 4
    mp_ = cfg.mamba2_cache_params
    res["gdn_temporal"] = tuple(mp_.shape.temporal)
    with du.scoped_tp_partition_ratios([77, 23]):
        res["gdn_want_v_heads"] = tp_partition_sizes(cfg.linear_num_value_heads, 2,
                                                     units=cfg.linear_num_key_heads)
    res["gdn_v_heads_total"] = cfg.linear_num_value_heads
    set_tp_partition_ratios(None)
    # the sampler's sync group: the ranks that sample
    from sglang.srt.layers.sampler import Sampler
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", enable_vram_ledger=False))
    try:
        smp = Sampler()
        res["sampler_group"] = list(smp._tp_sync_coordinator.ranks)
    except Exception as e:  # noqa: BLE001 -- a server-args-less construction, named
        res["sampler_group"] = f"{type(e).__name__}: {e}"
    # the tp_worker token source and takeover
    from sglang.srt.managers import tp_worker as tw

    res["token_src"] = tw._wl_token_src(types.SimpleNamespace(weightless_kv_head_rank=9))
    from sglang.srt.utils import broadcast_pyobj

    mine = [100 + rank, 200 + rank]
    got = broadcast_pyobj(mine if rank == res["token_src"] else [], rank,
                          ps.get_world_group().cpu_group, src=res["token_src"])
    res["adopted"] = list(got)
    with open(os.path.join(out, f"{rank}.pkl"), "wb") as f:
        pickle.dump(res, f)
    du.set_weightless_kv_head_rank(None)
    ps.destroy_model_parallel()
    ps.set_model_tp_partition(None)
    ps.destroy_distributed_environment()


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_form_b_roles_counts_and_sources_on_every_rank():
    d = tempfile.mkdtemp()
    mp.spawn(_run, args=(_port(), d), nprocs=3, join=True)
    res = [pickle.load(open(os.path.join(d, f"{r}.pkl"), "rb")) for r in range(3)]
    assert [(r["head"], r["worker"]) for r in res] == [(True, False), (True, False), (False, True)]
    for r in res:
        assert r["lead"] == 0 and r["token_src"] == 0
        assert r["q_counts"] == [18, 6, 0] and r["kv_counts"] == [3, 1, 0]   # rank-uniform
        assert r["adopted"] == [100, 200]                                    # the lead's tokens
    assert res[0]["sampler_group"] == res[1]["sampler_group"] == [0, 1], res[0]["sampler_group"]
    # GDN state: W rank r holds its W-share of the value heads; K the 1-element pool
    want = res[0]["gdn_want_v_heads"]
    assert res[0]["gdn_temporal"][0] == want[0] and res[1]["gdn_temporal"][0] == want[1], (
        res[0]["gdn_temporal"], res[1]["gdn_temporal"], want)
    assert sum(want) == res[0]["gdn_v_heads_total"]
    assert res[2]["gdn_temporal"] == (1, 1, 1)
    assert res[2]["sampler_group"] == [2]


def test_moe_under_form_b_is_refused_by_name():
    from sglang.srt import rank_form as rf

    dense = types.SimpleNamespace(text_config=types.SimpleNamespace(num_experts_per_tok=0))
    moe = types.SimpleNamespace(text_config=types.SimpleNamespace(num_experts_per_tok=8))
    assert rf.form_b_kv_path(dense) == rf.FORM_B_KV_PATH_LANE
    with pytest.raises(rf.RankFormBMoeNotBuilt, match="W189.*NF-K unter Form B braucht"):
        rf.form_b_kv_path(moe)
    assert rf.form_b_head_set([[2], [0, 1]], [3, 1, 0]) == ((0, 1), (3, 1))
    assert rf.form_b_head_set([[0, 1], [2]], None) == ((0, 1), None)


def test_the_role_and_install_sites_read_the_head_set():
    """Source pins: model_runner sets the roles from the head set (and builds W
    under form_b_build_context, not the lane's TP=1); the scheduler installs the
    set before the lane; the accept source follows the lead."""
    import inspect

    from sglang.srt.managers import scheduler as sch
    from sglang.srt.model_executor import model_runner as mr
    from sglang.srt.speculative import eagle_utils as eu

    src = inspect.getsource(mr)
    assert "self.is_weightless_head = is_weightless_head_rank(self.tp_rank)" in src
    assert "(self.is_weightless_head and not self.is_form_b_head)" in src
    assert "set_weightless_kv_weight_ranks(_w, _wr)" in inspect.getsource(sch)
    assert "get_weightless_kv_head_rank()" in inspect.getsource(eu.spec_accept_broadcast_src)
