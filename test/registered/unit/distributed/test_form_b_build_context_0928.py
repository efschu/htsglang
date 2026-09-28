"""F6 step 2c: a Form B weight rank BUILDS as one of |W| ranks (draft v2 §6).

Linears cache tp_size / tp_rank from the parallel context at construction
(layers/linear.py), attention projections attn_tp_size / attn_tp_rank, and
RowParallelLinear gates its all-reduce on the cached value. Three real
processes over gloo with the model_tp partition W={0,1}, K={2}: inside
rank_form.form_b_build_context a weight rank reads |W| and its index in W, the
shard plan is the weight vector restricted to W, a real RowParallelLinear gets
its W-share of the input dimension, and the attention layers' collectives
(attention_tensor_model_parallel_all_reduce, dp_attention.attn_tp_*) run on
model_tp. The KV-only rank is refused by name (W188, seam F15).
"""

import os
import pickle
import socket
import tempfile

import pytest
import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

RATIO = [3, 1, 0]            # weight vector over ALL ranks; K has 0


def _run(rank, port, out, partition):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      RANK=str(rank), WORLD_SIZE="3", LOCAL_RANK=str(rank))
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.distributed.utils import set_tp_partition_ratios, tp_partition_sizes

    _orig = ps.init_model_parallel_group
    ps.init_model_parallel_group = lambda *a, **k: _orig(*a, **{
        **k, "use_pynccl": False, "use_custom_allreduce": False,
        "use_mscclpp_allreduce": False, "use_torch_symm_mem_allreduce": False})
    ps.init_distributed_environment(world_size=3, rank=rank, local_rank=rank,
                                    distributed_init_method=f"tcp://127.0.0.1:{port}",
                                    backend="gloo")
    ps.set_model_tp_partition(partition)
    ps.initialize_model_parallel(tensor_model_parallel_size=3, backend="gloo")
    set_tp_partition_ratios(RATIO if partition else None, allow_zero=bool(partition))
    from sglang.srt import rank_form as rf
    from sglang.srt.distributed import communication_op as co
    from sglang.srt.layers import dp_attention as dpa
    from sglang.srt.runtime_context import get_parallel

    res = {}
    x = torch.full((2,), float(rank + 1))
    res["attn_ar"] = co.attention_tensor_model_parallel_all_reduce(x.clone()).tolist()
    res["dpa_ar"] = dpa.attn_tp_all_reduce(x.clone()).tolist()
    res["attn_tp_group_ranks"] = list(ps.get_attn_tp_group().ranks)    # control: ALL
    try:
        ctx = rf.form_b_build_context(rank)
    except rf.RankFormKvRankBuild as e:
        res["refused"] = str(e)
        ctx = None
    if ctx is not None:
        from sglang.srt.layers.linear import RowParallelLinear

        with ctx:
            p = get_parallel()
            res["ctx"] = (p.tp_size, p.tp_rank, p.attn_tp_size, p.attn_tp_rank)
            res["sizes"] = tp_partition_sizes(64, p.tp_size)
            row = RowParallelLinear(64, 8, bias=False, params_dtype=torch.float32)
            res["row"] = (row.tp_size, row.tp_rank, row.input_size_per_partition)
        res["after"] = (get_parallel().tp_size, get_parallel().tp_rank)
    elif partition is None:
        res["ctx"] = None
    with open(os.path.join(out, f"{rank}.pkl"), "wb") as f:
        pickle.dump(res, f)
    set_tp_partition_ratios(None)
    ps.destroy_model_parallel()
    ps.set_model_tp_partition(None)
    ps.destroy_distributed_environment()


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn(partition):
    d = tempfile.mkdtemp()
    mp.spawn(_run, args=(_port(), d, partition), nprocs=3, join=True)
    return [pickle.load(open(os.path.join(d, f"{r}.pkl"), "rb")) for r in range(3)]


def test_weight_ranks_build_as_w_and_attention_collectives_run_on_model_tp():
    res = _spawn([[0, 1], [2]])
    assert res[0]["ctx"] == (2, 0, 2, 0) and res[1]["ctx"] == (2, 1, 2, 1)
    assert res[0]["sizes"] == res[1]["sizes"] == [48, 16]          # 3:1 over W, not over 3
    assert res[0]["row"] == (2, 0, 48) and res[1]["row"] == (2, 1, 16)
    assert res[0]["after"] == (3, 0) and res[1]["after"] == (3, 1)  # restored
    assert "W188" in res[2]["refused"] and "F15" in res[2]["refused"]
    for r in range(3):
        assert res[r]["attn_tp_group_ranks"] == [0, 1, 2]          # scheduler side: ALL
    assert res[0]["attn_ar"] == res[1]["attn_ar"] == [3.0, 3.0]    # 1 + 2 over W
    assert res[0]["dpa_ar"] == res[1]["dpa_ar"] == [3.0, 3.0]
    assert res[2]["attn_ar"] == res[2]["dpa_ar"] == [3.0, 3.0]     # alone: its own


def test_without_a_partition_nothing_changes():
    res = _spawn(None)
    for r in range(3):
        assert res[r]["ctx"] is None
        assert res[r]["attn_ar"] == res[r]["dpa_ar"] == [6.0, 6.0]  # the attn-tp group


def test_build_override_arithmetic():
    from sglang.srt import rank_form as rf

    part = [[0, 1], [2]]
    ov, base, fams = rf.form_b_build_override(part, 1, [77, 23, 0], {"mlp": [3, 1, 0]})
    assert ov == dict(tp_size=2, tp_rank=1, attn_tp_size=2, attn_tp_rank=1)
    assert base == [77, 23] and fams == {"mlp": [3, 1]}
    ov, base, fams = rf.form_b_build_override([[2], [0, 3], [1]], 3)
    assert ov["tp_size"] == 2 and ov["tp_rank"] == 1 and base is None and fams == {}
    with pytest.raises(rf.RankFormKvRankBuild, match="W188"):
        rf.form_b_build_override(part, 2, [77, 23, 0])
    with pytest.raises(rf.RankFormShapeMismatch, match="disagrees with the weight ranks"):
        rf.form_b_build_override(part, 0, [77, 0, 5])
    with pytest.raises(rf.RankFormShapeMismatch, match="names 2 ranks"):
        rf.form_b_build_override(part, 0, [77, 23])
