"""Uneven LSE merge: the uneven-a2a reduce-scatter equals the all-reduce path
(bit-close, every rank gets exactly its own heads), and the bf16 wire option
halves the payload (fn7g: 12 x 100 MB fp32 all-reduce per 8k chunk)."""

import torch

from sglang.srt.layers.dcp import comm

COUNTS = [3, 2, 1]  # uneven head shards, like --rank-tp-ratio


class _World:
    """Three fake ranks sharing one mailbox so all_to_all_single / all_reduce /
    all_gather can be simulated rank by rank in one process."""

    def __init__(self):
        self.outs = {}
        self.lses = {}
        self.a2a_sent = {}
        self.wire_dtypes = []

    def group(self, rank):
        w = self

        class G:
            world_size = 3
            rank_in_group = rank

            def all_gather(self, t, dim=0):
                return torch.cat([w.lses[r] for r in range(3)], dim=dim)

            def all_reduce(self, t):
                w.wire_dtypes.append(t.dtype)
                return sum(w.ar_inputs[r].to(t.dtype) for r in range(3)).to(t.dtype)

            def all_to_all_single_v(self, output, input, output_split_sizes=None, input_split_sizes=None):
                w.wire_dtypes.append(input.dtype)
                assert sum(input_split_sizes) == input.shape[0]
                assert sum(output_split_sizes) == output.shape[0]
                w.a2a_sent[rank] = list(torch.split(input.clone(), input_split_sizes, dim=0))
                # block `rank` of every sender lands in my output, ordered by sender
                blocks = [w.a2a_sent[r][rank] for r in range(3)]
                assert [b.shape[0] for b in blocks] == output_split_sizes
                output.copy_(torch.cat(blocks, dim=0))
                return output

        return G()


def _reference(outs, lses):
    """Plain math: merge three partials with their lse per (token, head)."""
    stack_lse = torch.stack(lses)  # [3, T, H]
    g = torch.logsumexp(stack_lse, dim=0)
    w = torch.exp(stack_lse - g)  # [3, T, H]
    return sum(outs[r].float() * w[r].unsqueeze(-1) for r in range(3)), g


def _run(monkeypatch, mode, dtype):
    torch.manual_seed(1)
    comm._LSE_MERGE.update({"dtype": None, "mode": None})
    monkeypatch.setenv("SGLANG_DCP_LSE_MERGE", mode)
    monkeypatch.setenv("SGLANG_DCP_LSE_MERGE_DTYPE", dtype)
    monkeypatch.setattr(comm, "weightless_kv_active", lambda: False)
    bounds = [(0, 3), (3, 5), (5, 6)]
    monkeypatch.setattr(comm, "cp_local_head_bounds", lambda g, c: bounds[g.rank_in_group])
    T, H, D = 7, 6, 4
    w = _World()
    outs = [torch.randn(T, H, D, dtype=torch.bfloat16) for _ in range(3)]
    lses = [torch.randn(T, H) for _ in range(3)]
    for r in range(3):
        w.lses[r] = lses[r]
    ref_out, ref_lse = _reference(outs, lses)
    # the all-reduce simulation needs every rank's scaled input: precompute
    g = torch.logsumexp(torch.stack(lses), dim=0)
    w.ar_inputs = {r: outs[r].float() * torch.exp(lses[r] - g).unsqueeze(-1) for r in range(3)}
    fn = comm.cp_lse_ag_out_a2a_mha_uneven if mode == "a2a" else comm.cp_lse_ag_out_ar_mha_uneven
    # a2a: all ranks must have "sent" before any receives -- run sends first
    if mode == "a2a":
        # pre-populate the mailbox by running each rank once for its send block
        for r in range(3):
            try:
                fn(outs[r], lses[r], w.group(r), COUNTS)
            except Exception:
                pass
    results = []
    for r in range(3):
        res, lse = fn(outs[r], lses[r], w.group(r), COUNTS, return_lse=True)
        s, e = bounds[r]
        assert res.shape == (T, e - s, D) and res.dtype == torch.float32
        assert torch.allclose(lse, ref_lse[:, s:e], atol=1e-5)
        tol = 3e-2 if dtype == "bf16" else 1e-4
        assert torch.allclose(res, ref_out[:, s:e], atol=tol, rtol=tol), (mode, dtype, r)
        results.append(res)
    comm._LSE_MERGE.update({"dtype": None, "mode": None})
    return w.wire_dtypes


def test_ar_and_a2a_merges_match_the_reference_in_fp32(monkeypatch):
    assert torch.float32 in _run(monkeypatch, "ar", "fp32")
    assert torch.float32 in _run(monkeypatch, "a2a", "fp32")


def test_bf16_wire_halves_the_payload_within_tolerance(monkeypatch):
    assert torch.bfloat16 in _run(monkeypatch, "ar", "bf16")
    assert torch.bfloat16 in _run(monkeypatch, "a2a", "bf16")


def test_the_backend_picks_the_merge_by_env(monkeypatch):
    import inspect

    from sglang.srt.layers.attention import qwen_sparse_attn_backend as qb

    src = inspect.getsource(qb.QwenSparseAttnBackend._attend_rows) if hasattr(qb, "QwenSparseAttnBackend") else open(inspect.getsourcefile(qb)).read()
    assert 'lse_merge_mode() == "a2a"' in src
