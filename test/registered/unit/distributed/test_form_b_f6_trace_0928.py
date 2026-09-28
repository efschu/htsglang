"""F6 trace test, CPU half (Form-B-F6 draft v2 §4; the GPU half is
/spinning/gpu-arb/devtools/form_b_f6_trace_gpu.py).

Three real processes over gloo. Every collective goes through the REAL
primitives (``cp_all_gather_heads_uneven``, ``cp_lse_ag_out_ar_mha_uneven``,
``capture_safe_tp_broadcast``) and the REAL Python call sites that exist on
CPU (``form_a_dcp.worker_attention_step`` / ``share_kv`` / ``share_topk`` /
``gather_q`` / ``merge``, ``form_a_attends``, the DFLASH lane accept
``DFlashWorkerV2._lane_accept_broadcast``). A shim ``GroupCoordinator`` over a
gloo group records ``(op, shape, dtype)`` per communicator. The check is the
F6 claim itself: **per communicator, every member issues the identical
sequence** -- a divergence is a hang (or a silent wrong answer) on metal.

Communicators, as in the draft: ``tp`` (the scheduler group, ALL ranks),
``dcp`` (all ranks; A, T, Q, M and the spec channel), ``model_tp`` (Form B:
the weight ranks W={0,1}; rank 2 alone in its own group).

Cases: decode, target-verify, extend chunk 1 (no prefix) and chunk 2 (WITH
prefix -- the M1s root c5b6efa309 class), idle, the DFLASH lane round, the
Form B dense layer schedule with spec_k-steered draft steps k = 1..3.
``spec_k`` and the padded broadcasts are the product functions of
speculative/form_b_spec.py (F6 step 4); the dcp primitives are the real ones.
"""

import os
import pickle
import socket
import tempfile
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

WORLD = 3
W = (0, 1)          # Form B weight ranks
K = (2,)            # KV-only
T, D = 4, 8


class ShimGroup:
    """The GroupCoordinator surface the primitives touch, over gloo, recording."""

    def __init__(self, name, ranks, pg, rec):
        self.unique_name = name
        self.ranks = list(ranks)
        self.world_size = len(ranks)
        self.rank_in_group = self.ranks.index(dist.get_rank())
        self.device_group = self.cpu_group = pg
        self.pynccl_comm = None
        self._rec = rec

    def _log(self, op, t):
        self._rec.append((self.unique_name, op, tuple(t.shape), str(t.dtype)))

    def all_gather(self, x, dim=-1):
        self._log("all_gather", x)
        parts = [torch.empty_like(x) for _ in range(self.world_size)]
        dist.all_gather(parts, x.contiguous(), group=self.device_group)
        return torch.cat(parts, dim=dim)

    def all_reduce(self, x):
        self._log("all_reduce", x)
        y = x.clone()
        dist.all_reduce(y, group=self.device_group)
        return y

    def broadcast(self, x, src=0):
        self._log(f"broadcast<{src}", x)
        dist.broadcast(x, src=self.ranks[src], group=self.device_group)
        return x


def _groups(rec):
    everyone = dist.new_group(list(range(WORLD)), backend="gloo")
    everyone_dcp = dist.new_group(list(range(WORLD)), backend="gloo")
    mt = {}
    for part in ([0, 1], [2]):                     # torch rule: every rank builds every group
        mt[tuple(part)] = dist.new_group(part, backend="gloo")
    me = dist.get_rank()
    my_part = next(p for p in mt if me in p)
    return (ShimGroup("tp", range(WORLD), everyone, rec),
            ShimGroup("dcp", range(WORLD), everyone_dcp, rec),
            ShimGroup("model_tp", my_part, mt[my_part], rec))


def _mode(kind):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return {"decode": ForwardMode.DECODE, "verify": ForwardMode.TARGET_VERIFY,
            "extend": ForwardMode.EXTEND, "idle": ForwardMode.IDLE}[kind]


def _form_a_layer(rank, dcp, kind, prefix):
    """One full-attention layer under Form A (#239 QSA host 0), real call sites."""
    from sglang.srt.layers.attention.qsa import form_a_dcp as fa

    geo = fa.FormADcpGeometry(world=WORLD, host_rank=0, rank=rank, q_heads=4, kv_heads=2,
                              head_dim=D, topk_width=4, scaling=D ** -0.5)
    if kind == "idle":
        return                                      # the host issues nothing: neither does a worker
    fb = types.SimpleNamespace(forward_mode=_mode(kind), seq_lens_cpu=[prefix + T],
                               extend_seq_lens_cpu=[T])
    attends = fa.form_a_attends(fb)
    if rank == 0:
        k = torch.randn(T, 2, D)
        fa.share_kv(k, torch.randn(T, 2, D), dcp, geo)
        if attends:
            fa.share_topk(torch.zeros(T, 4, dtype=torch.int32), T, 4, dcp, geo, "cpu")
            fa.gather_q(torch.randn(T, 4, D), dcp, geo)
            fa.merge(torch.randn(T, 4, D), torch.randn(T, 4), dcp, geo)
    else:
        fa.worker_attention_step(
            rows=T, head_dim=D, topk_width=4, dtype=torch.float32, device="cpu", group=dcp,
            geo=geo, write=lambda kf, vf: None, attends=attends,
            resolve_rows=lambda topk: (torch.zeros(0, dtype=torch.long), None),
            attend=lambda q, rows, counts: (torch.zeros(T, 4, D), torch.full((T, 4), -1e30)))


def _form_b_full_layer(rank, dcp, model_tp):
    """Form B dense full-attention layer: dcp A, Q, M with counts [h0,h1,0], then
    model_tp o_proj + MLP all-reduce on W only."""
    from sglang.srt.layers.dcp.comm import cp_all_gather_heads_uneven, cp_lse_ag_out_ar_mha_uneven

    kv_counts, q_counts = [1, 1, 0], [3, 1, 0]
    cp_all_gather_heads_uneven(torch.randn(2 * T, kv_counts[rank], D), dcp, kv_counts)
    cp_all_gather_heads_uneven(torch.randn(T, q_counts[rank], D), dcp, q_counts)
    cp_lse_ag_out_ar_mha_uneven(torch.randn(T, 4, D), torch.randn(T, 4), dcp, q_counts)
    if rank in W:
        model_tp.all_reduce(torch.randn(T, 16))     # o_proj
        model_tp.all_reduce(torch.randn(T, 16))     # dense MLP


def _form_b_gdn_layer(rank, model_tp):
    if rank in W:
        model_tp.all_reduce(torch.randn(T, 16))


def _dflash_lane_round(rank, tp, bs=2, block=4):
    """G-A1: draft block (lead -> all) + ONE accept broadcast, real call sites."""
    from sglang.srt.speculative import dflash_worker_v2 as dw
    from sglang.srt.speculative import eagle_utils
    from sglang.srt.speculative.spec_utils import capture_safe_tp_broadcast

    dw.get_tp_group = lambda: tp
    eagle_utils.spec_accept_broadcast_src = lambda: 0
    blk = torch.arange(bs * block, dtype=torch.int64).view(bs, block) if rank == 0 else \
        torch.empty(bs, block, dtype=torch.int64)
    capture_safe_tp_broadcast(tp, (blk,), src=0)
    stub = types.SimpleNamespace(device="cpu")
    if rank == 0:
        dw.DFlashWorkerV2._lane_accept_broadcast(
            stub, bs, torch.tensor([2, 0], dtype=torch.int32), torch.tensor([7, 9]))
    else:
        dw.DFlashWorkerV2._lane_accept_broadcast(stub, bs, None, None)


def _form_b_spec_round(rank, dcp, model_tp, k_by_round, k_max=3, bs=2):
    """F6 step 4, the PRODUCT spec channel (speculative/form_b_spec.py):
    spec_k (lead -> all over dcp) BEFORE the draft; k draft vocab gathers over
    model_tp on W; the padded draft block and the packed accept over dcp in the
    k_max form (the real broadcast_padded / broadcast_padded_inplace)."""
    from sglang.srt.speculative import form_b_spec as fbs

    lead = rank == 0
    for k_lead in k_by_round:
        k = fbs.spec_k(k_lead if lead else None, group=dcp, lead=0, k_max=k_max)
        for _ in range(k):
            if rank in W:
                model_tp.all_gather(torch.randn(bs, 2), dim=-1)  # vocab-parallel top-1
        blk = torch.zeros(bs, k, dtype=torch.int64) if lead else None
        fbs.broadcast_padded(blk, (bs, k), fbs.draft_block_numel(bs, k_max),
                             dtype=torch.int64, device="cpu", group=dcp, lead=0)
        packed = torch.zeros(bs * (2 * (k + 1) + 1), dtype=torch.int32)
        fbs.broadcast_padded_inplace(packed, fbs.accept_numel(bs, k_max), group=dcp, lead=0)


def _run(rank, port, out_dir, case):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    rec = []
    tp, dcp, model_tp = _groups(rec)
    torch.manual_seed(rank)
    if case == "form_a":
        for kind, prefix in (("decode", 97), ("verify", 97), ("extend", 0), ("extend", 512),
                             ("idle", 0)):
            _form_a_layer(rank, dcp, kind, prefix)
    elif case == "dflash_lane":
        for _ in range(3):
            _dflash_lane_round(rank, tp)
    elif case == "form_b":
        for layer in range(8):                       # full attention every 4th, as the 27B
            if layer % 4 == 3:
                _form_b_full_layer(rank, dcp, model_tp)
            else:
                _form_b_gdn_layer(rank, model_tp)
        # k is the LEAD's choice (adaptive draft); every rank follows spec_k
        _form_b_spec_round(rank, dcp, model_tp, k_by_round=[1, 3, 2])
    with open(os.path.join(out_dir, f"{case}.{rank}.pkl"), "wb") as f:
        pickle.dump((rank, [tuple(g.ranks) for g in (tp, dcp, model_tp)], rec), f)
    dist.barrier()
    dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _traces(case):
    d = tempfile.mkdtemp()
    mp.spawn(_run, args=(_free_port(), d, case), nprocs=WORLD, join=True)
    out = {}
    for r in range(WORLD):
        with open(os.path.join(d, f"{case}.{r}.pkl"), "rb") as f:
            rank, parts, rec = pickle.load(f)
        out[rank] = (dict(zip(("tp", "dcp", "model_tp"), parts)), rec)
    return out


def _assert_lockstep(traces):
    """Per communicator instance: identical (op, shape, dtype) sequence on every member."""
    for name in ("tp", "dcp", "model_tp"):
        instances = {}
        for rank, (parts, rec) in traces.items():
            seq = [e[1:] for e in rec if e[0] == name]
            instances.setdefault(parts[name], {})[rank] = seq
        for members, seqs in instances.items():
            ref = seqs[members[0]]
            for r in members:
                assert seqs[r] == ref, f"{name}{members}: rank {r} diverges from {members[0]}"
    return traces


@pytest.mark.parametrize("case", ["form_a", "dflash_lane", "form_b"])
def test_every_communicator_runs_one_sequence(case):
    traces = _assert_lockstep(_traces(case))
    counts = {r: len(rec) for r, (_, rec) in traces.items()}
    assert all(c > 0 for c in counts.values()), counts
    if case == "form_b":
        # rank 2 (KV-only) never enters model_tp; W do, identically
        assert not [e for e in traces[2][1] if e[0] == "model_tp"]
        assert [e for e in traces[0][1] if e[0] == "model_tp"]
        # step 4: every dcp broadcast of the spec rounds has a k-independent
        # shape (spec_k [1], block [bs*k_max], accept [bs*(2k_max+3)])
        bshapes = {e[2] for e in traces[0][1] if e[0] == "dcp" and e[1].startswith("broadcast")}
        assert bshapes == {(1,), (6,), (18,)}, bshapes


def test_the_checker_catches_a_divergence():
    fake = {0: ({"tp": (0, 1, 2), "dcp": (0, 1, 2), "model_tp": (0, 1)},
                [("dcp", "all_gather", (4,), "f")]),
            1: ({"tp": (0, 1, 2), "dcp": (0, 1, 2), "model_tp": (0, 1)},
                [("dcp", "all_gather", (4,), "f")]),
            2: ({"tp": (0, 1, 2), "dcp": (0, 1, 2), "model_tp": (2,)}, [])}
    with pytest.raises(AssertionError, match="dcp"):
        _assert_lockstep(fake)
