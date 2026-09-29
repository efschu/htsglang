"""F6 step 4: Form B's spec channel (form_b_spec; draft v2 §3, NF objection 3).

Three real processes over gloo build the REAL GroupCoordinators (tp, dcp over
all ranks, model_tp over W={0,1}); the lead is rank 0. Checked:

* spec_k: the lead's k reaches every rank BEFORE the draft; an out-of-range k
  is a named stop on EVERY rank (after the collective, so nobody hangs).
* draft_block / accept: the collective's element count is the k_max form
  whatever this round's k -- measured at the transport, not declared.
* without a partition the channel is off.
And, without processes, the EAGLE worker's round under Form B: only the lead
decides k, every rank adopts the published one.
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

K_MAX = 3


def _run(rank, port, out, partition):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      RANK=str(rank), WORLD_SIZE="3", LOCAL_RANK=str(rank))
    from sglang.srt.distributed import parallel_state as ps

    _orig = ps.init_model_parallel_group
    ps.init_model_parallel_group = lambda *a, **k: _orig(*a, **{
        **k, "use_pynccl": False, "use_custom_allreduce": False,
        "use_mscclpp_allreduce": False, "use_torch_symm_mem_allreduce": False,
        "use_message_queue_broadcaster": False})
    ps.init_distributed_environment(world_size=3, rank=rank, local_rank=rank,
                                    distributed_init_method=f"tcp://127.0.0.1:{port}",
                                    backend="gloo")
    ps.set_model_tp_partition(partition)
    # DCP's platform gate (CUDA/HIP only) -- the groups themselves are gloo.
    ps.is_cuda = lambda: True
    ps.initialize_model_parallel(tensor_model_parallel_size=3,
                                 decode_context_parallel_size=3, backend="gloo")
    from sglang.srt.speculative import form_b_spec as fbs

    res = {"active": fbs.form_b_spec_active()}
    if res["active"]:
        fbs.set_spec_k_max(K_MAX)
        sizes = []
        real = fbs._broadcast
        fbs._broadcast = lambda g, buf, src: (sizes.append((g.unique_name, buf.numel(), src)),
                                             real(g, buf, src))
        lead = rank == fbs.form_b_lead()
        res["lead"] = fbs.form_b_lead()
        res["ks"], res["blocks"], res["accepts"] = [], [], []
        bs = 2
        for k_lead in (1, 3, 2):
            k = fbs.spec_k(k_lead if lead else None)
            res["ks"].append(k)
            blk = torch.arange(bs * k, dtype=torch.int64).view(bs, k) + 100 if lead else None
            got = fbs.broadcast_padded(blk, (bs, k), fbs.draft_block_numel(bs),
                                       dtype=torch.int64, device="cpu")
            res["blocks"].append(got.tolist())
            packed = torch.full((bs * (2 * (k + 1) + 1),), 7 if lead else 0, dtype=torch.int32)
            fbs.broadcast_padded_inplace(packed, fbs.accept_numel(bs))
            res["accepts"].append(sorted(set(packed.tolist())))
        # an out-of-range k: named on every rank, after the collective
        try:
            fbs.spec_k(K_MAX + 2 if lead else None)
            res["bad_k"] = "adopted"
        except fbs.FormBSpecError as e:
            res["bad_k"] = "W186" in str(e)
        res["sizes"] = sizes
        fbs._broadcast = real
        # F15 (5e): the DFLASH lane accept under Form B -- a head that is not
        # the lead passes its OWN accept and gets the lead's back (and must adopt)
        import types as _t

        from sglang.srt.speculative import dflash_worker_v2 as dw

        stub = _t.SimpleNamespace(device="cpu", tp_rank=rank)
        mine = (torch.tensor([rank, rank]), torch.tensor([10 + rank, 10 + rank]))
        acc, bon = dw.DFlashWorkerV2._lane_accept_broadcast(
            stub, 2, *(mine if rank in (0, 1) else (None, None)))
        res["lane_accept"] = (acc.tolist(), bon.tolist())
        res["follower"] = dw._form_b_accept_follower(stub)
        fbs.set_spec_k_max(None)
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


def test_spec_channel_publishes_k_first_and_keeps_a_fixed_form():
    res = _spawn([[0, 1], [2]])
    for r in range(3):
        assert res[r]["active"] and res[r]["lead"] == 0
        assert res[r]["ks"] == [1, 3, 2]                     # the lead's k, everywhere
        for k, blk in zip((1, 3, 2), res[r]["blocks"]):
            assert blk == (torch.arange(2 * k).view(2, k) + 100).tolist()
        assert all(a == [7] for a in res[r]["accepts"])      # the lead's accept
        assert res[r]["bad_k"] is True                       # named on every rank
        # the transport saw the k_max form every round: spec_k 1, block 2*3,
        # accept 2*(2*4+1) -- never a k-dependent size; all over dcp from 0
        sizes = res[r]["sizes"]
        assert {s[0].split(":")[0] for s in sizes} == {"dcp"} and {s[2] for s in sizes} == {0}
        assert [s[1] for s in sizes] == [1, 6, 18] * 3 + [1]
    assert res[0]["sizes"] == res[1]["sizes"] == res[2]["sizes"]
    for r in range(3):
        assert res[r]["lane_accept"] == ([0, 0], [10, 10])        # the lead's, everywhere
    assert [res[r]["follower"] for r in range(3)] == [False, True, True]


def test_without_a_partition_the_channel_is_off():
    res = _spawn(None)
    assert [r["active"] for r in res] == [False] * 3


def test_lead_and_numel_arithmetic():
    from sglang.srt.speculative import form_b_spec as fbs

    assert fbs.lead_rank_of([[1, 2], [0]]) == 1
    assert fbs.lead_rank_of([[0, 2], [1], [3]]) == 0
    for bad in ([[0], [1], [2]], [[0, 1], [2, 3]]):
        with pytest.raises(fbs.FormBSpecError, match="W186"):
            fbs.lead_rank_of(bad)
    assert fbs.draft_block_numel(4, k_max=3) == 12
    assert fbs.accept_numel(4, k_max=3) == 4 * 9
    with pytest.raises(fbs.FormBSpecError, match="k_max is not installed"):
        fbs.spec_k_max()
    with pytest.raises(fbs.FormBSpecError, match="k_max must be"):
        fbs.set_spec_k_max(0)


class _Ctl:
    def __init__(self, k):
        self.k, self.built_steps, self.calls = k, [1, 2, 3], []

    def activate_step_by_batch(self, bs):
        self.calls.append(("ema", bs))

    def activate_steps(self, k):
        self.calls.append(("set", k))
        return k in self.built_steps


@pytest.mark.parametrize("rank", [0, 2])
def test_eagle_round_under_form_b_adopts_the_leads_k(monkeypatch, rank):
    """_form_b_activate_round: only the lead (0) runs its policy and publishes;
    rank 2 (KV-only) adopts k without consulting anything of its own."""
    from sglang.srt.speculative import eagle_worker_v2 as ew
    from sglang.srt.speculative import form_b_spec as fbs

    published = []
    monkeypatch.setattr(fbs, "form_b_spec_active", lambda: True)
    monkeypatch.setattr(fbs, "form_b_lead", lambda: 0)
    monkeypatch.setattr(fbs, "spec_k", lambda k, device=None: published.append(k) or 3)
    ctl = _Ctl(2)
    me = types.SimpleNamespace(tp_rank=rank, chain_policy=None, adaptive_controller=ctl,
                               speculative_num_steps=2, device="cpu")
    me._apply_chain_policy = lambda: False
    ew.EAGLEWorkerV2._form_b_activate_round(me, 4)
    if rank == 0:
        assert published == [2] and ctl.calls == [("ema", 4), ("set", 3)]
    else:
        assert published == [None] and ctl.calls == [("set", 3)]
    # a k with no runtime state on this rank: named, not a silent keep
    monkeypatch.setattr(fbs, "spec_k", lambda k, device=None: 5)
    with pytest.raises(fbs.FormBSpecError, match="no runtime state"):
        ew.EAGLEWorkerV2._form_b_activate_round(me, 4)


def test_consensus_is_identity_under_form_b(monkeypatch):
    from sglang.srt.speculative import eagle_worker_v2 as ew
    from sglang.srt.speculative import form_b_spec as fbs

    monkeypatch.setattr(fbs, "form_b_spec_active", lambda: True)
    assert ew.EAGLEWorkerV2._agree_chain_length(types.SimpleNamespace(), 2) == 2
