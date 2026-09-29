"""G-A1 (rank form 28.09.): the DFLASH chain on the weightless-KV lane.

The lane runs DFLASH as a CHAIN: the draft solo and TP=1-local on the head, the
workers join exactly (a) the draft-block broadcast, (b) the verify's per-layer
DCP dispatch and (c) ONE accept broadcast from the head. CPU only.

1. ORACLE. A verify round (bs * block rows, causal inside the block) over an
   owner-sharded committed prefix -- weighted token vector, the head's own rows
   plus the block attended locally on the head, every worker its owned prefix
   rows -- merged by LSE equals ONE TP=1 causal attention over prefix + block.
2. THE ACCEPT PROTOCOL. The head's (accept_len, bonus) crosses in one packed
   broadcast; the worker's committed block equals the head's.
3. TP=1-BUILT HEAD IS LOCAL. The candidate top-k over a TP=1-built lm_head
   issues no gather and equals the tp=1 group's result.
4. ADMISSION. DFLASH is admitted on the lane (solo on the head), a tree verify
   and DSPARK are refused by name.
"""

import inspect
import types

import pytest
import torch

from sglang.srt.distributed import utils as du
from sglang.srt.layers.dcp.test_weightless_kv_math import (
    _merge_lse,
    _partial_attention_with_lse,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _causal_attention(q, k, v, scale, prefix):
    """TP=1 oracle: q rows are the block [prefix, prefix+T); row i sees every
    prefix token and block tokens 0..i."""
    t, n = q.shape[0], k.shape[0]
    scores = torch.einsum("thd,nhd->thn", q, k) * scale
    mask = torch.ones(t, n, dtype=torch.bool)
    for i in range(t):
        mask[i, prefix + i + 1:] = False
    scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
    return torch.einsum("thn,nhd->thd", torch.softmax(scores, -1), v)


def test_verify_round_over_the_owner_sharded_prefix_equals_tp1():
    torch.manual_seed(0)
    H, D, prefix, block = 4, 32, 197, 8
    scale = D ** -0.5
    q = torch.randn(block, H, D, dtype=torch.float64)
    k = torch.randn(prefix + block, H, D, dtype=torch.float64)
    v = torch.randn(prefix + block, H, D, dtype=torch.float64)
    ref = _causal_attention(q, k, v, scale, prefix)

    # the lane's owner rule, installed exactly as the scheduler installs it
    saved = du.get_cp_token_ratios()
    try:
        du.set_cp_token_ratios(du.reduce_token_vector([2, 64, 32]))
        pre = du.cp_token_prefix(3)
        S = pre[-1]
        owners = [[p for p in range(prefix) if pre[r] <= p % S < pre[r + 1]] for r in range(3)]
    finally:
        du.set_cp_token_ratios(saved)
    assert sorted(sum(owners, [])) == list(range(prefix))
    partials = []
    for r in range(3):
        rows = owners[r]
        if r == 0:
            # the head: its owned prefix rows + the block, causal inside it
            idx = torch.tensor(rows + list(range(prefix, prefix + block)))
            o = torch.empty_like(q)
            lse = torch.empty(block, H, dtype=torch.float64)
            for i in range(block):
                sel = idx[: len(rows) + i + 1]
                oi, li = _partial_attention_with_lse(q[i:i + 1], k[sel], v[sel], scale)
                o[i], lse[i] = oi[0], li[0]
        else:
            idx = torch.tensor(rows)
            o, lse = _partial_attention_with_lse(q, k[idx], v[idx], scale)
        partials.append((o, lse))
    merged, by_rank = _merge_lse(partials, [H, 0, 0], 0)
    assert torch.allclose(merged, ref, atol=1e-10)
    assert by_rank[0].shape == ref.shape and by_rank[1].numel() == 0 == by_rank[2].numel()


def _worker_stub(device="cpu"):
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    w = types.SimpleNamespace(device=device)
    w._lane_accept_broadcast = types.MethodType(DFlashWorkerV2._lane_accept_broadcast, w)
    return w


def test_one_accept_broadcast_carries_the_heads_decision(monkeypatch):
    from sglang.srt.speculative import dflash_worker_v2 as dw
    from sglang.srt.speculative import eagle_utils

    sent = {}

    class _G:
        world_size = 3

    def fake_bcast(group, tensors, src):
        (buf,) = tensors
        if "head" in sent:
            buf.copy_(sent["head"])
        else:
            sent["head"] = buf.clone()
        sent["src"] = src
        sent["n"] = sent.get("n", 0) + 1

    monkeypatch.setattr(dw, "get_tp_group", lambda: _G())
    monkeypatch.setattr(dw, "capture_safe_tp_broadcast", fake_bcast)
    monkeypatch.setattr(eagle_utils, "spec_accept_broadcast_src", lambda: 0)
    cand = torch.tensor([[5, 11, 12, 13], [7, 21, 22, 23]])
    acc, bon = torch.tensor([2, 0], dtype=torch.int32), torch.tensor([99, 98])
    head_out, head_commit = dw._commit_accept(cand, acc, bon)
    _worker_stub()._lane_accept_broadcast(2, acc, bon)      # the head publishes
    r_acc, r_bon = _worker_stub()._lane_accept_broadcast(2, None, None)  # a worker
    w_out, w_commit = dw._commit_accept(cand, r_acc, r_bon)
    assert torch.equal(w_out, head_out) and torch.equal(w_commit, head_commit)
    assert sent["n"] == 2 and sent["src"] == 0


def test_lane_worker_verify_returns_before_any_logits_use():
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    src = inspect.getsource(DFlashWorkerV2.forward_batch_generation)
    w = src.index('if _lane == "worker":')
    assert src.index("logits_output = target_out.logits_output") < w
    ret = src.index("return GenerationBatchResult(", w)
    worker_body = src[w:ret]
    assert "_lane_accept_broadcast(bs, None, None)" in worker_body
    assert "next_token_logits" not in worker_body
    assert "_tp_sync" not in worker_body
    # the head publishes exactly once, after its decision
    assert src.count("self._lane_accept_broadcast(bs, accept_len, bonus)") == 1
    # the lane head issues no shadow-directed hidden/selector broadcasts
    # (29.09.: via _solo_vocab_parallel -- False on the one-head lane, True on
    # Form B's head set W, whose lead's lm_head is sharded over W)
    assert src.count("self._spec_solo_active and self._solo_vocab_parallel()") == 3


def test_tp1_built_head_selects_candidates_locally(monkeypatch):
    from sglang.srt.models import dflash as dm

    logits = torch.tensor([[0.1, 3.0, -1.0, 2.0, 0.5], [1.0, 0.0, 4.0, 2.5, -2.0]])
    monkeypatch.setattr(dm, "_project_candidate_logits", lambda h, lm, **kw: logits)
    calls = []

    def ident_gather(t, dim=-1):
        calls.append(1)
        return t

    monkeypatch.setattr(dm, "tensor_model_parallel_all_gather", ident_gather)
    monkeypatch.setattr(dm, "_flashinfer_top_k", None)   # CPU: the torch.topk path
    shard = types.SimpleNamespace(num_org_elements=5, org_vocab_start_index=0)
    hs = torch.zeros(2, 3)
    ids_1, vals_1 = dm.gather_candidate_topk(
        types.SimpleNamespace(shard_indices=shard, tp_size=1), hs, 3, use_quant_head=False)
    assert calls == []                                   # no gather on a TP=1-built head
    ids_g, vals_g = dm.gather_candidate_topk(
        types.SimpleNamespace(shard_indices=shard, tp_size=3), hs, 3, use_quant_head=False)
    assert len(calls) == 2
    assert torch.equal(ids_1, ids_g) and torch.equal(vals_1, vals_g)


def _lane_spec_args(**kw):
    from sglang.srt.server_args import ServerArgs

    kw.setdefault("enable_vram_ledger", False)
    return ServerArgs(model_path="dummy", tp_size=3, dcp_size=3, weightless_kv_fastlane=True,
                      speculative_draft_placement="solo", **kw)


def test_dflash_chain_is_admitted_on_the_lane_and_the_tree_is_not():
    _lane_spec_args(speculative_algorithm="DFLASH")._reject_unsupported_weightless_spec()
    tree = _lane_spec_args(speculative_algorithm="DFLASH")
    tree.speculative_dflash_tree_verify = True
    with pytest.raises(ValueError, match="W184"):
        tree._reject_unsupported_weightless_spec()
    with pytest.raises(ValueError, match="EAGLE-family"):
        _lane_spec_args(speculative_algorithm="DSPARK")._reject_unsupported_weightless_spec()
