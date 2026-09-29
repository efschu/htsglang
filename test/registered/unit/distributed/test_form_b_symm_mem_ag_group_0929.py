"""Form B x the logits multimem all-gather: the symmetric-memory rendezvous
runs over model_tp (the weight ranks W), never over the all-ranks tp group.

Metal, 29.09. (formb-hang-09292056 / -09292128, 2 of 2 boots): TP0 and TP1
stood in ``symm_mem.rendezvous`` <- create_state <- MultimemAllGatherer._build
<- LogitsProcessor._get_logits during the decode-graph warmup, TP2 (KV-only,
no lm_head) in the next tp barrier of run_capture_warmups. _build rendezvoused
over get_tp_group() = {0,1,2}; rank 2 never enters it -> deadlock.

Three real gloo processes build the REAL groups (tp over all, model_tp
[[0,1],[2]]). create_state is replaced by a recorder (no CUDA on CPU) that
notes the ranks of the group it was handed and then fails, so the gatherer
takes its NCCL fallback -- which must stay on W too. Rank 2 does what it does
on metal: it never calls the gatherer and goes straight to the tp barrier.
"""

import os
import pickle
import socket
import tempfile
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _run(rank, port, out, partition):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      RANK=str(rank), WORLD_SIZE="3", LOCAL_RANK=str(rank))
    from sglang.srt import runtime_context
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.distributed.device_communicators import triton_symm_mem_ag as ag

    _orig = ps.init_model_parallel_group
    ps.init_model_parallel_group = lambda *a, **k: _orig(*a, **{
        **k, "use_pynccl": False, "use_custom_allreduce": False,
        "use_mscclpp_allreduce": False, "use_torch_symm_mem_allreduce": False})
    ps.init_distributed_environment(world_size=3, rank=rank, local_rank=rank,
                                    distributed_init_method=f"tcp://127.0.0.1:{port}",
                                    backend="gloo")
    ps.set_model_tp_partition(partition)
    ps.initialize_model_parallel(tensor_model_parallel_size=3, backend="gloo")
    runtime_context.get_server_args = lambda: types.SimpleNamespace(nnodes=1)

    seen = []

    def _recording_create_state(group, rank_in_group, max_tokens, hidden_size, device=None):
        # The rendezvous set, instead of a rendezvous that would wait for it.
        seen.append({"ranks": list(dist.get_process_group_ranks(group)),
                     "rank_in_group": rank_in_group, "hidden_size": hidden_size})
        raise RuntimeError("recorder: no symmetric memory on CPU")

    ag.create_state = _recording_create_state
    res = {"rank": rank}
    kv_only = partition is not None and rank in partition[-1] and len(partition[-1]) == 1
    if not kv_only:
        g = ag.MultimemAllGatherer(max_tokens=16, enabled=True, skip_entry_sync=True)
        x = torch.full((2, 8), float(rank + 1), dtype=torch.bfloat16)
        res["gathered"] = g(x).float()[0].tolist()
        res["state_after"] = "none" if g._state is None else "other"
    res["seen"] = seen
    # run_capture_warmups: every rank meets in the tp barrier after the forward
    ps.get_tp_group().barrier()
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


def test_form_b_rendezvous_set_is_the_weight_ranks():
    res = _spawn([[0, 1], [2]])
    for r in (0, 1):
        assert len(res[r]["seen"]) == 1, res[r]
        seen = res[r]["seen"][0]
        # the deadlock: {0,1,2} -- rank 2 never enters it
        assert seen["ranks"] == [0, 1], seen
        assert seen["rank_in_group"] == r
        assert seen["hidden_size"] == 8 * 2              # shard width x |W|
        # NCCL fallback on W: rank 0's 1s next to rank 1's 2s, no rank 2
        assert res[r]["gathered"] == [1.0] * 8 + [2.0] * 8
        assert res[r]["state_after"] == "none"
    assert res[2]["seen"] == [] and "gathered" not in res[2]


def test_classic_rendezvous_set_is_the_tp_group():
    """No partition: model_tp IS tp -- byte-identical to before."""
    res = _spawn(None)
    for r in range(3):
        seen = res[r]["seen"][0]
        assert seen["ranks"] == [0, 1, 2] and seen["rank_in_group"] == r
        assert seen["hidden_size"] == 8 * 3
        assert res[r]["gathered"] == [1.0] * 8 + [2.0] * 8 + [3.0] * 8
