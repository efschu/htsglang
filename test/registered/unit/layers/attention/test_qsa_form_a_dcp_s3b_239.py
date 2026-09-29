# SPDX-License-Identifier: Apache-2.0
"""#239 S3b -- the QSA sparse attention under Form A x the token cut.

REAL gloo collectives on three ranks, no GPU. The host (rank 0) holds every
q/kv head and the top-k; every rank owns a token range of the full-attention
KV by the weighted owner rule. The host runs collectives A, T, Q, M through
the functions the QSA backend calls; the workers run ``worker_attention_step``.
Pinned:

* the merged output on the host equals the host-only reference (all rows, all
  heads, one rank) up to fp reassociation -- for the cuts [64,0,0] (host owns
  all), [0,46,18] and [0,37,27] (host owns nothing, the S2b optimum at x1/x2),
  [4,31,29] (x4), [21,21,22] (near even) and [32,0,32] (a worker owns nothing),
  each for a decode-like forward (4 rows = bs 1 x verify 4) and an extend
  with prefix (48 rows), in both merge modes (ar and a2a);
* the new k/v of the forward land in exactly the owner's rows (collective A),
  also for an extend without prefix that stops after A;
* every rank issues the SAME collective sequence (kinds and shapes) -- the
  lockstep a hang would break;
* the backend routes its write, its top-k and its q-head counts through the
  Form A step when the geometry is installed, and the seam F5 is wired.
"""

import inspect
import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

WORLD = 3
H, KV, D = 4, 2, 8
C = 256          # global context slots
K = 12           # top-k width


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _GlooGroup:
    """The GroupCoordinator surface the uneven gathers and merges use."""

    def __init__(self, rank, world):
        import torch.distributed as dist

        self._dist = dist
        self.rank_in_group = rank
        self.world_size = world
        self.log = []

    def all_gather(self, t, dim=0):
        t = t.contiguous()
        self.log.append(("all_gather", tuple(t.shape), str(t.dtype)))
        parts = [torch.empty_like(t) for _ in range(self.world_size)]
        self._dist.all_gather(parts, t)
        return torch.cat(parts, dim=dim)

    def all_reduce(self, t):
        self.log.append(("all_reduce", tuple(t.shape), str(t.dtype)))
        out = t.clone()
        self._dist.all_reduce(out)
        return out

    def all_to_all_single_v(self, output, input, output_split_sizes=None, input_split_sizes=None):
        self.log.append(("all_to_all", tuple(input.shape), str(input.dtype)))
        self._dist.all_to_all_single(
            output, input.contiguous(), output_split_sizes, input_split_sizes
        )
        return output


def _ref_attend(q, k_pool, v_pool, rows, scaling):
    """Sparse attention over ``rows`` (-1 = none): (out fp32 [T,H,D], lse [T,H])."""
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


def _global(seed):
    g = torch.Generator().manual_seed(seed)
    k_all = torch.randn(C, KV, D, generator=g)
    v_all = torch.randn(C, KV, D, generator=g)
    return k_all, v_all, g


def _case_inputs(seed, rows, prefix):
    k_all, v_all, g = _global(seed)
    new_loc = torch.arange(C - rows, C) if prefix else torch.arange(0, rows)
    q = torch.randn(rows, H, D, generator=g)
    topk = torch.randint(0, C, (rows, K), generator=g).to(torch.int32)
    topk[:, -2:] = -1  # padded lanes, as the indexer emits
    return k_all, v_all, new_loc, q, topk


def _worker(rank, port, cases, q_out):
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        from flliper.srt import rank_role
        from flliper.srt.distributed.utils import set_cp_token_ratios
        from flliper.srt.layers.attention.qsa import form_a_dcp as fa
        from flliper.srt.layers.dcp import comm as c
        from flliper.srt.layers.dcp.owner import (
            dcp_compact_pool_rows,
            dcp_weighted_owner_bounds,
            dcp_weighted_read_slots,
            dcp_weighted_write_slots,
        )

        plan = rank_role.RankRolePlan(("host", "worker", "worker"))
        rank_role.set_form_a_role_plan(plan, rank)
        results = []
        for idx, (vec, rows, prefix, mode) in enumerate(cases):
            set_cp_token_ratios(vec)
            c._LSE_MERGE["mode"] = mode
            c._LSE_MERGE["dtype"] = "fp32"
            c._LSE_MERGE["fused"] = False
            S, lo, hi, ratio = dcp_weighted_owner_bounds(WORLD, rank)
            geo = fa.form_a_dcp_geometry(H, KV, WORLD, rank)
            k_all, v_all, new_loc, q, topk = _case_inputs(100 + idx, rows, prefix)
            n_rows = dcp_compact_pool_rows(C, S, ratio) + 1  # + the dump row
            k_pool = torch.zeros(n_rows, KV, D)
            v_pool = torch.zeros(n_rows, KV, D)
            # the prefix: every slot that is not this forward's, owner rows only
            old = torch.tensor([s for s in range(C) if s not in set(new_loc.tolist())])
            loc, mask = dcp_weighted_write_slots(old, S, lo, hi, ratio)
            k_pool[loc[mask]] = k_all[old[mask]]
            v_pool[loc[mask]] = v_all[old[mask]]
            grp = _GlooGroup(rank, WORLD)

            def write(k_full, v_full):
                wl, wm = dcp_weighted_write_slots(new_loc, S, lo, hi, ratio)
                k_pool[wl[wm]] = k_full[wm]
                v_pool[wl[wm]] = v_full[wm]

            def resolve(tk):
                compact, owned = dcp_weighted_read_slots(tk.clamp(min=0), S, lo, hi, ratio)
                return torch.where((tk >= 0) & owned, compact, torch.full_like(compact, -1)), None

            def attend(q_full, owned_rows, _counts):
                return _ref_attend(q_full, k_pool, v_pool, owned_rows, 0.5)

            merged = None
            if geo.is_host:
                kf, vf = fa.share_kv(k_all[new_loc], v_all[new_loc], grp, geo)
                write(kf, vf)
                if prefix:
                    tk = fa.share_topk(topk, rows, K, grp, geo, "cpu")
                    owned_rows, _ = resolve(tk)
                    q_full = fa.gather_q(q, grp, geo)
                    out, lse = attend(q_full, owned_rows, None)
                    merged = fa.merge(out, lse, grp, geo)
            else:
                fa.worker_attention_step(
                    rows=rows, head_dim=D, topk_width=K, dtype=torch.float32,
                    device="cpu", group=grp, geo=geo, write=write, attends=prefix,
                    resolve_rows=resolve, attend=attend,
                )
            # what this rank's pool now holds for the new slots it owns
            wl, wm = dcp_weighted_write_slots(new_loc, S, lo, hi, ratio)
            new_ok = bool(torch.equal(k_pool[wl[wm]], k_all[new_loc][wm])) and bool(
                torch.equal(v_pool[wl[wm]], v_all[new_loc][wm]))
            # by value: a tensor on the queue is an fd the parent fetches from
            # this process, which may already be gone (FileNotFoundError)
            results.append((None if merged is None else merged.numpy().copy(), grp.log,
                            new_ok, int(wm.sum())))
        q_out.put((rank, results))
    except Exception as exc:  # pragma: no cover - surfaced by the parent
        import traceback

        q_out.put((rank, "ERR " + traceback.format_exc() + repr(exc)))
    finally:
        dist.destroy_process_group()


CASES = [
    (vec, rows, prefix, mode)
    for vec in ([64, 0, 0], [0, 46, 18], [0, 37, 27], [4, 31, 29], [21, 21, 22], [32, 0, 32])
    for rows, prefix in ((4, True), (48, True), (16, False))
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
def test_the_host_merge_equals_the_host_only_reference(run, i):
    vec, rows, prefix, mode = CASES[i]
    merged = run[0][i][0]
    if not prefix:
        assert merged is None
        return
    k_all, v_all, new_loc, q, topk = _case_inputs(100 + i, rows, prefix)
    ref, _ = _ref_attend(q, k_all, v_all, topk, 0.5)
    assert merged.shape == ref.shape
    torch.testing.assert_close(merged.float(), ref, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("i", range(len(CASES)))
def test_every_owner_holds_the_new_rows_it_owns(run, i):
    vec = CASES[i][0]
    owned_total = 0
    for r in range(WORLD):
        _, _, new_ok, n_owned = run[r][i]
        assert new_ok, (vec, r)
        owned_total += n_owned
        if vec[r] == 0:
            assert n_owned == 0
    assert owned_total == CASES[i][1]  # every new slot has exactly one owner


@pytest.mark.parametrize("i", range(len(CASES)))
def test_every_rank_issues_the_same_collectives(run, i):
    logs = [run[r][i][1] for r in range(WORLD)]
    assert logs[0] == logs[1] == logs[2]
    kinds = [e[0] for e in logs[0]]
    prefix, mode = CASES[i][2], CASES[i][3]
    if not prefix:
        assert kinds == ["all_gather"]  # A only
    elif mode == "ar":
        assert kinds == ["all_gather", "all_gather", "all_gather", "all_gather", "all_reduce"]
    else:
        assert kinds == ["all_gather", "all_gather", "all_gather", "all_gather", "all_to_all"]


def test_the_backend_routes_through_the_form_a_step():
    from flliper.srt.layers.attention import qwen_sparse_attn_backend as be

    src = inspect.getsource(be)
    assert "self._init_form_a_dcp(model_config, dcp_size, dcp_rank)" in src
    w = src.index("def _set_kv_buffer(")
    assert "share_kv(" in src[w : src.index("def _local_rows(")]
    r = src.index("def _rows_and_counts(")
    assert "share_topk(" in src[r : src.index("def _logical_to_physical_graph(")]
    a = src.index("def _attend_rows(")
    assert "form_a.q_counts" in src[a : a + 1700]


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_the_backend_takes_form_a_dcp_instead_of_refusing_it(rank):
    """Before S3b a Form A rank under DCP hit the replicated-kv refusal
    ('QSA sparse attention under uneven DCP needs replicated kv heads')."""
    from types import SimpleNamespace
    from unittest import mock

    from flliper.srt import rank_role
    from flliper.srt.distributed.utils import (
        get_tp_partition_ratios,
        set_cp_token_ratios,
        set_tp_partition_ratios,
    )
    from flliper.srt.layers.attention import qwen_sparse_attn_backend as be

    saved_tp = get_tp_partition_ratios()
    set_tp_partition_ratios([1, 0, 0], allow_zero=True)
    set_cp_token_ratios([0, 23, 9])
    rank_role.set_form_a_role_plan(rank_role.RankRolePlan(("host", "worker", "worker")), rank)
    try:
        cfg = SimpleNamespace(
            hf_text_config=SimpleNamespace(
                num_attention_heads=24, indexer_budget=2048, indexer_compress_ratio=4
            ),
            get_total_num_kv_heads=lambda: 2,
            head_dim=256,
        )
        par = SimpleNamespace(attn_dcp_size=3, attn_dcp_rank=rank, attn_tp_size=3)
        obj = object.__new__(be.QwenSparseAttnBackend)
        with mock.patch("flliper.srt.runtime_context.get_parallel", return_value=par):
            obj._init_dcp(SimpleNamespace(is_draft_worker=False, dtype=torch.bfloat16), cfg)
        assert obj.form_a_dcp is not None and obj.dcp_size == 3
        assert obj.uneven_dcp_weighted is True
        assert obj.form_a_dcp.q_counts == [24, 0, 0]
        assert obj.cp_ratio == [0, 23, 9][rank]
        # #239 S3c: what a worker needs without a layer of its own
        assert (obj.form_a_dcp.head_dim, obj.form_a_dcp.topk_width) == (256, 2051)
        assert obj.form_a_dcp.scaling == 256**-0.5 and obj.form_a_dtype == torch.bfloat16
    finally:
        rank_role.set_form_a_role_plan(None, 0)
        set_cp_token_ratios(None)
        set_tp_partition_ratios(saved_tp)


def test_the_geometry_is_host_only_heads():
    from flliper.srt.layers.attention.qsa.form_a_dcp import FormADcpGeometry

    g = FormADcpGeometry(world=3, host_rank=0, rank=2, q_heads=24, kv_heads=2)
    assert g.q_counts == [24, 0, 0] and g.kv_counts == [2, 0, 0]
    assert g.topk_counts == [1, 0, 0] and not g.is_host


def test_without_a_role_plan_there_is_no_form_a_geometry():
    from flliper.srt import rank_role
    from flliper.srt.layers.attention.qsa.form_a_dcp import form_a_dcp_geometry

    rank_role.set_form_a_role_plan(None, 0)
    assert form_a_dcp_geometry(24, 2, 3, 0) is None


def test_the_seam_f5_is_wired():
    from flliper.srt import rank_role

    assert rank_role.SEAMS["F5"].wired is True
    assert "F5" not in rank_role.UNWIRED_ORDER
