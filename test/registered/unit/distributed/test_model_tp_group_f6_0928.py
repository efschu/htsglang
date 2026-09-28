"""F6 step 2: Form B's model_tp group in parallel_state (draft v2 §2).

Three real processes over gloo build the REAL GroupCoordinators: the
scheduler group ``tp`` over ALL ranks (NF objection 1) and ``model_tp`` over
the weight ranks W={0,1} with the KV-only rank 2 alone -- its own device and
cpu group. A model_tp all_reduce sums over W only and never touches rank 2;
without a partition get_model_tp_group() IS the TP group (byte-identical).
"""

import os
import pickle
import socket
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _run(rank, port, out, partition):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      RANK=str(rank), WORLD_SIZE="3", LOCAL_RANK=str(rank))
    from sglang.srt.distributed import parallel_state as ps

    # CPU: no device communicators (pynccl / custom AR need CUDA); the
    # GroupCoordinators and their gloo groups are the real ones.
    _orig = ps.init_model_parallel_group
    ps.init_model_parallel_group = lambda *a, **k: _orig(*a, **{
        **k, "use_pynccl": False, "use_custom_allreduce": False,
        "use_mscclpp_allreduce": False, "use_torch_symm_mem_allreduce": False})
    ps.init_distributed_environment(world_size=3, rank=rank, local_rank=rank,
                                    distributed_init_method=f"tcp://127.0.0.1:{port}",
                                    backend="gloo")
    ps.set_model_tp_partition(partition)
    ps.initialize_model_parallel(tensor_model_parallel_size=3, backend="gloo")
    mtp = ps.get_model_tp_group()
    tp = ps.get_tp_group()
    res = {"tp_ranks": list(tp.ranks), "mtp_ranks": list(mtp.ranks),
           "mtp_name": mtp.unique_name, "mtp_is_tp": mtp is tp,
           "own_cpu_group": mtp.cpu_group is not tp.cpu_group}
    x = torch.full((2,), float(rank + 1))
    res["mtp_sum"] = mtp.all_reduce(x.clone()).tolist()
    res["tp_sum"] = tp.all_reduce(x.clone()).tolist()
    # F6 2b: the model's linear/vocab ops run on model_tp, control on tp
    from sglang.srt.distributed import communication_op as co

    res["op_ar"] = co.tensor_model_parallel_all_reduce(x.clone()).tolist()
    res["op_ag"] = co.tensor_model_parallel_all_gather(x.clone(), dim=-1).tolist()
    # F6 step 3: the reversed guard on the REAL groups (coordinator and
    # ProcessGroup forms): control on model_tp / layer on tp refused.
    from sglang.srt.rank_role import (
        COLLECTIVE_CONTROL as C,
        COLLECTIVE_LAYER as L,
        FormBCollectiveWrongGroup,
        guard_collective_subgroup,
    )

    def _verdict(kind, g):
        try:
            guard_collective_subgroup(kind, g, "t")
            return "ok"
        except FormBCollectiveWrongGroup:
            return "refused"

    res["guard"] = {
        "control_tp": _verdict(C, tp), "control_tp_cpu": _verdict(C, tp.cpu_group),
        "control_mtp": _verdict(C, mtp), "control_mtp_cpu": _verdict(C, mtp.cpu_group),
        "layer_mtp": _verdict(L, mtp), "layer_tp": _verdict(L, tp),
        "layer_tp_dev": _verdict(L, tp.device_group),
    }
    with open(os.path.join(out, f"{rank}.pkl"), "wb") as f:
        pickle.dump(res, f)
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


def test_model_tp_is_the_weight_ranks_and_tp_stays_everyone():
    res = _spawn([[0, 1], [2]])
    for r in range(3):
        assert res[r]["tp_ranks"] == [0, 1, 2]              # scheduler group: ALL
        assert res[r]["tp_sum"] == [6.0, 6.0]
        assert res[r]["mtp_is_tp"] is False and res[r]["own_cpu_group"]
        assert "model_tp" in res[r]["mtp_name"]
    assert res[0]["mtp_ranks"] == res[1]["mtp_ranks"] == [0, 1]
    assert res[0]["mtp_sum"] == res[1]["mtp_sum"] == [3.0, 3.0]   # 1 + 2, rank 2 not in it
    assert res[2]["mtp_ranks"] == [2] and res[2]["mtp_sum"] == [3.0, 3.0]  # alone: its own 3
    # 2b: tensor_model_parallel_* reduce/gather over W only
    assert res[0]["op_ar"] == res[1]["op_ar"] == [3.0, 3.0]
    assert res[0]["op_ag"] == res[1]["op_ag"] == [1.0, 1.0, 2.0, 2.0]
    assert res[2]["op_ag"] == [3.0, 3.0]
    # 3: reversed guard -- control only on tp, layer only on model_tp
    for r in range(3):
        g = res[r]["guard"]
        assert g["control_tp"] == g["control_tp_cpu"] == "ok", (r, g)
        assert g["control_mtp"] == g["control_mtp_cpu"] == "refused", (r, g)
        assert g["layer_mtp"] == "ok" and g["layer_tp"] == g["layer_tp_dev"] == "refused", (r, g)


def test_without_a_partition_model_tp_is_the_tp_group():
    res = _spawn(None)
    for r in range(3):
        assert res[r]["mtp_is_tp"] is True and res[r]["mtp_sum"] == [6.0, 6.0]
        assert res[r]["op_ar"] == [6.0, 6.0]                   # byte-identical: the TP group
        assert set(res[r]["guard"].values()) == {"ok"}         # 3: no-op without Form B


def test_a_partition_must_cover_every_rank_once():
    from sglang.srt.distributed import parallel_state as ps

    for bad in ([[0, 1], [1, 2]], [[0], [], [1, 2]], [[1, 2]]):
        with pytest.raises(ValueError, match="W181"):
            ps.set_model_tp_partition(bad)
    # a partition that is well-formed but misses ranks of the WORLD is refused
    # when the group is built (the world size is known only there)
    ps.set_model_tp_partition([[0, 1]])
    with pytest.raises(ValueError, match="does not cover the world"):
        ps.init_model_tp_group(0, "gloo", 3)
    ps.set_model_tp_partition(None)


def test_reversed_guard_arithmetic():
    """F6 step 3 on rank sets: W={0,1}, K={2}, all=(0,1,2)."""
    from sglang.srt.rank_role import (
        COLLECTIVE_CONTROL as C,
        COLLECTIVE_LAYER as L,
        FormBCollectiveWrongGroup,
        FormASeamNotWired,
        RankRoleError,
        check_collective_group,
    )

    allr = (0, 1, 2)
    for mt in ((0, 1), (2,)):                        # a W rank's and the K rank's view
        check_collective_group(C, allr, mt, allr, "sched")
        check_collective_group(L, mt, mt, allr, "o_proj")
        with pytest.raises(FormBCollectiveWrongGroup, match="model_tp group"):
            check_collective_group(C, mt, mt, allr, "sched")
        with pytest.raises(FormBCollectiveWrongGroup, match="ALL-ranks scheduler group"):
            check_collective_group(L, allr, mt, allr, "o_proj")
    # a third group (e.g. a stray subgroup) is wrong for both kinds
    with pytest.raises(FormBCollectiveWrongGroup):
        check_collective_group(C, (0, 2), (0, 1), allr)
    with pytest.raises(FormBCollectiveWrongGroup):
        check_collective_group(L, (0, 2), (0, 1), allr)
    # classic: no partition, or model_tp == tp -> nothing checked
    check_collective_group(L, allr, None, allr)
    check_collective_group(C, (0, 1), allr, allr)
    check_collective_group(L, allr, allr, allr)
    with pytest.raises(RankRoleError, match="collective kind"):
        check_collective_group("sched", allr, None, allr)
    # its own class: survives F6 being wired
    assert not issubclass(FormBCollectiveWrongGroup, FormASeamNotWired)
