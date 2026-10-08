# SPDX-License-Identifier: Apache-2.0
"""#239 S3c -- the Form A worker forward joins every full-attention layer.

S3b gave the host its half (the QSA backend shares k/v, top-k and q, and
merges); S3c is the worker's half and its wiring:

* the WORKER runs the same backend paths (``_set_kv_buffer``,
  ``_rows_and_counts``, ``_attend_rows``) through
  ``QwenSparseAttnBackend.form_a_worker_attention`` -- REAL gloo collectives on
  three ranks, the host driving the backend exactly as its decode / extend
  paths do: the merged output equals the host-only reference, every owner
  holds its new rows, every rank issues the same sequence, for decode, extend
  with prefix and extend without prefix (A only), both merge modes;
* ``run_form_a_worker_layers`` puts each full-attention layer's step BEFORE
  that layer's MoE-input carrier (the host's order), refuses an attention
  layer without a MoE block and attention ids without a step;
* ``form_a_attends`` restates the host's branching, rank-uniformly;
* the boot gate declares A/T/Q/M on every rank's attention layers, and a rank
  whose backend did not take the cut disagrees there;
* the graph guard knows the new worker body and refuses it on the host;
* the host shares a top-k only at the width every worker receives.
"""

import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

WORLD = 3
H, KV, D = 4, 2, 8
C = 256
K = 12
SCALE = 0.5


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _GlooGroup:
    def __init__(self, rank, world):
        import torch.distributed as dist

        self._dist = dist
        self.rank_in_group = rank
        self.world_size = world
        self.log = []

    def all_gather(self, t, dim=0):
        t = t.contiguous()
        self.log.append(("all_gather", tuple(t.shape[:1]) + tuple(t.shape[2:]), str(t.dtype)))
        parts = [torch.empty_like(t) for _ in range(self.world_size)]
        self._dist.all_gather(parts, t)
        return torch.cat(parts, dim=dim)

    def all_reduce(self, t):
        self.log.append(("all_reduce", tuple(t.shape), str(t.dtype)))
        out = t.clone()
        self._dist.all_reduce(out)
        return out

    def all_to_all_single_v(self, output, input, output_split_sizes=None, input_split_sizes=None):
        self.log.append(("all_to_all", str(input.dtype)))
        self._dist.all_to_all_single(
            output, input.contiguous(), output_split_sizes, input_split_sizes
        )
        return output


def _ref_attend(q, k_pool, v_pool, rows, scaling):
    T, h, d = q.shape
    group = h // k_pool.shape[1]
    out = torch.zeros(T, h, d, dtype=torch.float32)
    lse = torch.full((T, h), float("-inf"), dtype=torch.float32)
    for t in range(T):
        idx = rows[t][rows[t] >= 0].long()
        if idx.numel() == 0:
            continue
        for hh in range(h):
            k = k_pool[idx, hh // group].float()
            v = v_pool[idx, hh // group].float()
            s = (k @ q[t, hh].float()) * scaling
            lse[t, hh] = torch.logsumexp(s, dim=0)
            out[t, hh] = torch.softmax(s, dim=0) @ v
    return out, lse


def _case_inputs(seed, rows, prefix):
    g = torch.Generator().manual_seed(seed)
    k_all = torch.randn(C, KV, D, generator=g)
    v_all = torch.randn(C, KV, D, generator=g)
    new_loc = torch.arange(C - rows, C) if prefix else torch.arange(0, rows)
    q = torch.randn(rows, H, D, generator=g)
    topk = torch.randint(0, C, (rows, K), generator=g).to(torch.int32)
    topk[:, -2:] = -1
    return k_all, v_all, new_loc, q, topk


class _Pool:
    """The pool surface the backend's write and read touch."""

    def __init__(self, n_rows):
        self.k = torch.zeros(n_rows, KV, D)
        self.v = torch.zeros(n_rows, KV, D)

    def set_kv_buffer(self, layer, loc, k, v, dcp_kv_mask=None):
        m = dcp_kv_mask
        self.k[loc[m]] = k[m]
        self.v[loc[m]] = v[m]

    def get_key_buffer(self, layer_id):
        return self.k

    def get_value_buffer(self, layer_id):
        return self.v


def _worker(rank, port, cases, q_out):
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        from flliper.srt import rank_role, runtime_context
        from flliper.srt.distributed.utils import set_cp_token_ratios
        from flliper.srt.layers.attention import qwen_sparse_attn_backend as be
        from flliper.srt.layers.attention.qsa import form_a_dcp as fa
        from flliper.srt.layers.dcp import comm as c
        from flliper.srt.layers.dcp.owner import (
            dcp_compact_pool_rows,
            dcp_weighted_owner_bounds,
            dcp_weighted_write_slots,
        )
        from flliper.srt.model_executor.forward_batch_info import ForwardMode

        be._QSA_ROWS_COMPACT["on"] = False
        be.sparse_attn_rows_triton = (
            lambda q, kp, vp, rows, scaling, row_counts=None: _ref_attend(q, kp, vp, rows, scaling)
        )
        rank_role.set_form_a_role_plan(rank_role.RankRolePlan(("host", "worker", "worker")), rank)
        results = []
        for idx, (vec, kind, mode) in enumerate(cases):
            rows, prefix = {"decode": (4, True), "extend": (48, True), "fresh": (16, False)}[kind]
            set_cp_token_ratios(vec)
            c._LSE_MERGE["mode"] = mode
            c._LSE_MERGE["dtype"] = "fp32"
            c._LSE_MERGE["fused"] = False
            S, lo, hi, ratio = dcp_weighted_owner_bounds(WORLD, rank)
            geo = fa.form_a_dcp_geometry(
                H, KV, WORLD, rank, head_dim=D, topk_width=K, scaling=SCALE
            )
            k_all, v_all, new_loc, q, topk = _case_inputs(300 + idx, rows, prefix)
            pool = _Pool(dcp_compact_pool_rows(C, S, ratio) + 1)
            old = torch.tensor([s for s in range(C) if s not in set(new_loc.tolist())])
            loc, mask = dcp_weighted_write_slots(old, S, lo, hi, ratio)
            pool.k[loc[mask]] = k_all[old[mask]]
            pool.v[loc[mask]] = v_all[old[mask]]
            grp = _GlooGroup(rank, WORLD)
            runtime_context.get_parallel = lambda: SimpleNamespace(dcp_group=grp)

            obj = object.__new__(be.QwenSparseAttnBackend)
            obj.dcp_size, obj.dcp_rank = WORLD, rank
            obj.uneven_dcp = obj.uneven_dcp_weighted = True
            obj.cp_S, obj.cp_lo, obj.cp_hi, obj.cp_ratio = S, lo, hi, ratio
            obj.form_a_dcp = geo
            obj.form_a_dtype = torch.float32
            obj.token_to_kv_pool = pool
            obj.req_to_token = None
            metadata = SimpleNamespace(
                is_cuda_graph=False,
                token_to_batch_idx=torch.zeros(rows, dtype=torch.int32),
                sequence_lengths=torch.tensor([C], dtype=torch.int32),
                token_slot_table=torch.arange(C, dtype=torch.int32).view(1, C),
                row_req_pool_indices=None,
            )
            obj._resolve_metadata = lambda fb: metadata
            fb = SimpleNamespace(
                out_cache_loc=new_loc,
                forward_mode=ForwardMode.DECODE if kind == "decode" else ForwardMode.EXTEND,
                seq_lens_cpu=[C if prefix else rows],
                extend_seq_lens_cpu=[1 if kind == "decode" else rows],
            )
            merged = None
            if geo.is_host:
                layer = SimpleNamespace(layer_id=3, scaling=SCALE)
                obj._set_kv_buffer(fb, layer, k_all[new_loc].reshape(rows, -1),
                                   v_all[new_loc].reshape(rows, -1))
                if prefix:
                    owned, counts = obj._rows_and_counts(topk, metadata)
                    merged = obj._attend_rows(q, layer, owned, counts)
            else:
                obj.form_a_worker_attention(fb, 3)
            wl, wm = dcp_weighted_write_slots(new_loc, S, lo, hi, ratio)
            new_ok = bool(torch.equal(pool.k[wl[wm]], k_all[new_loc][wm])) and bool(
                torch.equal(pool.v[wl[wm]], v_all[new_loc][wm]))
            results.append((None if merged is None else merged.numpy().copy(), grp.log,
                            new_ok, int(wm.sum())))
        q_out.put((rank, results))
    except Exception as exc:  # pragma: no cover - surfaced by the parent
        import traceback

        q_out.put((rank, "ERR " + traceback.format_exc() + repr(exc)))
    finally:
        dist.destroy_process_group()


CASES = [
    (vec, kind, mode)
    for vec in ([64, 0, 0], [0, 46, 18], [4, 31, 29], [32, 0, 32])
    for kind in ("decode", "extend", "fresh")
    for mode in ("ar", "a2a")
]


@pytest.fixture(scope="module")
def run():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, port, CASES, q)) for r in range(WORLD)]
    for p in procs:
        p.start()
    got = {}
    for _ in range(WORLD):
        rank, res = q.get(timeout=240)
        got[rank] = res
    for p in procs:
        p.join(timeout=60)
    for r, res in got.items():
        assert not isinstance(res, str), f"rank {r}: {res}"
        got[r] = [(None if m is None else torch.from_numpy(m), *rest) for m, *rest in res]
    return got


@pytest.mark.parametrize("i", range(len(CASES)))
def test_the_worker_step_merges_to_the_host_only_reference(run, i):
    vec, kind, mode = CASES[i]
    merged = run[0][i][0]
    if kind == "fresh":
        assert merged is None
        return
    rows = 4 if kind == "decode" else 48
    k_all, v_all, _new_loc, q, topk = _case_inputs(300 + i, rows, True)
    ref, _ = _ref_attend(q, k_all, v_all, topk, SCALE)
    torch.testing.assert_close(merged.float(), ref, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("i", range(len(CASES)))
def test_every_owner_holds_its_new_rows(run, i):
    total = 0
    for r in range(WORLD):
        _, _, new_ok, n = run[r][i]
        assert new_ok, (CASES[i], r)
        total += n
    assert total == {"decode": 4, "extend": 48, "fresh": 16}[CASES[i][1]]


@pytest.mark.parametrize("i", range(len(CASES)))
def test_host_and_worker_issue_the_same_sequence(run, i):
    logs = [run[r][i][1] for r in range(WORLD)]
    assert logs[0] == logs[1] == logs[2]
    kinds = [e[0] for e in logs[0]]
    kind, mode = CASES[i][1], CASES[i][2]
    if kind == "fresh":
        assert kinds == ["all_gather"]
    else:
        merge = "all_reduce" if mode == "ar" else "all_to_all"
        assert kinds[:3] == ["all_gather"] * 3 and kinds[-1] == merge


# ---- the worker forward's per-layer order ------------------------------------------

def _plan(rank):
    from flliper.srt import rank_role

    rank_role.set_form_a_role_plan(rank_role.RankRolePlan(("host", "worker", "worker")), rank)


def teardown_function(_fn):
    from flliper.srt import rank_role

    rank_role.set_form_a_role_plan(None, 0)


def test_the_attention_step_sits_before_its_layers_moe_carrier():
    from flliper.srt.form_a_worker_forward import run_form_a_worker_layers

    _plan(1)
    seq = []
    blocks = [(i, lambda x, fb, i=i: seq.append(("mlp", i))) for i in range(8)]

    def recv(n, h, **kw):
        seq.append(("recv",))
        return None

    served = run_form_a_worker_layers(
        blocks, num_tokens=2, hidden_size=4, dtype=torch.float32, device="cpu",
        receive=recv, attention=lambda lid: seq.append(("attn", lid)),
        attention_layer_ids=(3, 7),
    )
    assert served == 8
    at3 = seq.index(("attn", 3))
    assert seq[at3 + 1] == ("recv",) and seq[at3 + 2] == ("mlp", 3)
    assert seq[at3 - 1] == ("mlp", 2)
    assert [e for e in seq if e[0] == "attn"] == [("attn", 3), ("attn", 7)]


def test_an_attention_layer_without_a_moe_block_or_without_a_step_is_refused():
    from flliper.srt.form_a_worker_forward import (
        FormAWorkerForwardError,
        FormAWorkerLayerMismatch,
        run_form_a_worker_layers,
    )

    _plan(2)
    blocks = [(0, lambda x, fb: None), (1, lambda x, fb: None)]
    kw = dict(num_tokens=2, hidden_size=4, dtype=torch.float32, device="cpu",
              receive=lambda *a, **k: None)
    with pytest.raises(FormAWorkerLayerMismatch, match=r"\[3\]"):
        run_form_a_worker_layers(blocks, attention=lambda lid: None,
                                 attention_layer_ids=(1, 3), **kw)
    with pytest.raises(FormAWorkerForwardError, match="no attention step"):
        run_form_a_worker_layers(blocks, attention_layer_ids=(1,), **kw)
    # without the cut: no ids, no step -- the S6a route unchanged
    assert run_form_a_worker_layers(blocks, **kw) == 2


def test_form_a_attends_restates_the_host_branching():
    from flliper.srt.layers.attention.qsa.form_a_dcp import form_a_attends
    from flliper.srt.model_executor.forward_batch_info import ForwardMode

    def fb(mode, seq, ext):
        return SimpleNamespace(forward_mode=mode, seq_lens_cpu=seq, extend_seq_lens_cpu=ext)

    assert form_a_attends(fb(ForwardMode.DECODE, [9], [1]))
    assert form_a_attends(fb(ForwardMode.TARGET_VERIFY, [9], [4]))
    assert form_a_attends(fb(ForwardMode.EXTEND, [100, 30], [100, 10]))
    assert not form_a_attends(fb(ForwardMode.EXTEND, [100, 30], [100, 30]))
    with pytest.raises(ValueError, match="draft-extend"):
        form_a_attends(fb(ForwardMode.DRAFT_EXTEND_V2, [9], [4]))


def test_the_host_shares_a_topk_only_at_the_workers_width():
    from flliper.srt.layers.attention.qsa.form_a_dcp import form_a_dcp_geometry, share_topk

    _plan(0)
    geo = form_a_dcp_geometry(H, KV, 3, 0, head_dim=D, topk_width=K, scaling=SCALE)
    with pytest.raises(ValueError, match="13 wide"):
        share_topk(torch.zeros(2, 13, dtype=torch.int32), 2, 13, None, geo, "cpu")


def test_the_qsa_topk_width_is_the_config_fact():
    from flliper.srt.layers.attention.qsa.form_a_dcp import qsa_topk_width

    assert qsa_topk_width(SimpleNamespace(indexer_budget=2048, indexer_compress_ratio=4)) == 2051


# ---- boot gate and graph guard ---------------------------------------------------------

def test_the_boot_gate_declares_a_t_q_m_on_every_rank():
    from flliper.srt import rank_role
    from flliper.srt.form_a_boot_gate import declare_layer_collectives

    plan = rank_role.RankRolePlan(("host", "worker", "worker"))
    kw = dict(worker_skips_dense=True, host_dense_is_unsharded=True,
              host_uses_moe_exchange=False, moe_input_carrier="all_reduce_zero")
    for merge, last in (("ar", "all_reduce"), ("a2a", "all_to_all")):
        decl = [
            [op.key() for op in declare_layer_collectives(
                plan, r, is_attention_layer=True, form_a_dcp_merge=merge, **kw)]
            for r in range(3)
        ]
        assert decl[0] == decl[1] == decl[2]
        sites = [k[1] for k in decl[0]]
        assert sites[:4] == ["layer0.dcp_kv", "layer0.dcp_topk", "layer0.dcp_q", "layer0.dcp_lse"]
        assert decl[0][3][0] == last and sites[4] == "layer0.moe_in"
        # a linear-attention layer is untouched
        lin = declare_layer_collectives(plan, 1, is_attention_layer=False,
                                        form_a_dcp_merge=merge, **kw)
        assert not any("dcp" in op.site for op in lin)
    # a rank whose backend did not take the cut declares a different sequence
    host = declare_layer_collectives(plan, 0, is_attention_layer=True,
                                     form_a_dcp_merge="ar", **kw)
    worker = declare_layer_collectives(plan, 1, is_attention_layer=True, **kw)
    assert [o.key() for o in host] != [o.key() for o in worker]


def test_the_graph_guard_knows_the_dcp_route():
    from flliper.srt import rank_role

    plan = rank_role.RankRolePlan(("host", "worker", "worker"))
    rank_role.guard_graph_mode(plan, 1, "full", body=rank_role.GRAPH_BODY_MOE_ROUTE_DCP)
    with pytest.raises(rank_role.FormAWorkerDenseGraph):
        rank_role.guard_graph_mode(plan, 0, "full", body=rank_role.GRAPH_BODY_MOE_ROUTE_DCP)
    with pytest.raises(rank_role.FormAWorkerDenseGraph):
        rank_role.guard_graph_mode(plan, 2, "full", body=rank_role.GRAPH_BODY_MODEL_FORWARD)


def test_form_a_dcp_of_reads_the_backend_by_its_type():
    from flliper.srt.form_a_construction import FormAWorkerAttnBackend
    from flliper.srt.layers.attention import qwen_sparse_attn_backend as be
    from flliper.srt.layers.attention.qsa.form_a_dcp import form_a_dcp_of

    qsa = object.__new__(be.QwenSparseAttnBackend)
    qsa.form_a_dcp = "geo"
    wrapper = SimpleNamespace(full_attn_backend=qsa)
    assert form_a_dcp_of(wrapper) == "geo" and form_a_dcp_of(qsa) == "geo"
    assert form_a_dcp_of(object.__new__(FormAWorkerAttnBackend)) is None
    assert form_a_dcp_of(SimpleNamespace()) is None


def test_the_runner_wires_the_worker_step_and_the_gate():
    import inspect

    from flliper.srt.model_executor import model_runner as mr
    from flliper.srt.model_executor.runner import decode_cuda_graph_runner as gr

    # model_runner.py is frozen (large-class-style): it only delegates to
    # form_a_dcp_wiring, whose three decisions are tested below by behaviour.
    src = inspect.getsource(mr.ModelRunner)
    route = src[src.index("def run_form_a_worker_route"):]
    route = route[: route.index("def _forward_raw")]
    assert "form_a_worker_attention_step(" in route
    assert "attention_layer_ids=attention_layer_ids" in route
    gate = src[src.index("def _run_form_a_boot_gate"):]
    assert "form_a_dcp_merge=form_a_dcp_merge_of(self.attn_backend)" in gate[:3000]
    init = src[src.index("def init_attention_backend(self)"):]
    assert "form_a_worker_attn_backend(self)" in init[:2000]
    assert "QwenSparseAttnBackend" not in init[:2000]
    assert "GRAPH_BODY_MOE_ROUTE_DCP" in inspect.getsource(gr)


def test_the_wiring_chooses_the_worker_backend_by_the_cut():
    from unittest import mock

    from flliper.srt import form_a_dcp_wiring as w
    from flliper.srt.form_a_construction import FormAWorkerAttnBackend
    from flliper.srt.layers.attention import qwen_sparse_attn_backend as be

    def runner(cut, draft=False, qsa=True):
        return SimpleNamespace(
            is_draft_worker=draft,
            server_args=SimpleNamespace(form_a_dcp_vector=lambda: cut),
            model_config=SimpleNamespace(hf_config=SimpleNamespace(qsa=qsa)),
        )

    sentinel = object()
    with mock.patch(
        "flliper.srt.form_a_construction.FormAWorkerAttnBackend",
        lambda r: sentinel,
    ):
        for r in (runner(None), runner([64, 0, 0], draft=True)):
            got = w.form_a_worker_attn_backend(r)
            assert got == w.FormAWorkerAttn(w.FORM_A_WORKER, sentinel)

    class _Qsa:
        def __init__(self, r, geo="geo"):
            self.form_a_dcp = geo

    with mock.patch(
        "flliper.srt.layers.attention.qsa.config.is_qwen_qsa",
        lambda hf: hf.qsa,
    ), mock.patch.object(be, "QwenSparseAttnBackend", _Qsa):
        got = w.form_a_worker_attn_backend(runner([32, 0, 32]))
        assert got.name == w.FORM_A_WORKER_QSA_DCP and got.backend.form_a_dcp == "geo"
        with pytest.raises(RuntimeError, match="QSA full attention only"):
            w.form_a_worker_attn_backend(runner([32, 0, 32], qsa=False))
        with mock.patch.object(
            be, "QwenSparseAttnBackend", lambda r: _Qsa(r, geo=None)
        ):
            with pytest.raises(RuntimeError, match="did not take the Form A DCP"):
                w.form_a_worker_attn_backend(runner([32, 0, 32]))
    assert FormAWorkerAttnBackend.form_a_dcp is None


def test_the_wiring_declares_the_merge_and_builds_the_step():
    from unittest import mock

    from flliper.srt import form_a_dcp_wiring as w
    from flliper.srt.form_a_construction import FormAWorkerAttnBackend
    from flliper.srt.layers.attention import qwen_sparse_attn_backend as be

    qsa = object.__new__(be.QwenSparseAttnBackend)
    qsa.form_a_dcp = "geo"
    calls = []
    qsa.form_a_worker_attention = lambda fb, lid: calls.append((fb, lid))
    with mock.patch("flliper.srt.layers.dcp.comm.lse_merge_mode", lambda: "a2a"):
        assert w.form_a_dcp_merge_of(qsa) == "a2a"
        assert w.form_a_dcp_merge_of(SimpleNamespace(full_attn_backend=qsa)) == "a2a"
        assert w.form_a_dcp_merge_of(object.__new__(FormAWorkerAttnBackend)) is None
    pool = SimpleNamespace(full_attention_layer_id_mapping={3: 0, 7: 1})
    step, ids = w.form_a_worker_attention_step(qsa, "fb", pool)
    assert ids == (3, 7)
    step(7)
    assert calls == [("fb", 7)]
    worker = object.__new__(FormAWorkerAttnBackend)
    assert w.form_a_worker_attention_step(worker, "fb", pool) == (None, ())
