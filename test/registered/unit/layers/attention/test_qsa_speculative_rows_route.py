"""fn4o 19.09.: the speculative-paged forward (target verify / draft extend)
routed the DRAFT backend (dcp_size 1) to the FA4-cute varlen fallback, whose
MLIR refuses sm86 ('Operation creation failed', pack_gqa). The decode path had
armed the rows kernel on every rank since fn5e; the speculative path now uses
the same rule. Also: the gather-width bound reads num_tokens_per_req on a
draft-extend spec_info (no draft_token_num there, upstream form)."""

from types import SimpleNamespace

import torch


def test_rows_route_is_armed_on_a_cuda_rank_without_dcp():
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        _speculative_rows_route,
    )

    assert _speculative_rows_route(1, True, True, True)      # fn4o case
    assert _speculative_rows_route(3, False, True, True)     # DCP always
    assert _speculative_rows_route(1, False, True, False)    # no FA at all
    assert not _speculative_rows_route(1, False, True, True) # opt-out + FA present
    assert not _speculative_rows_route(1, True, False, True) # CPU q: reference path


def test_speculative_row_bound_reads_num_tokens_per_req_for_draft_extend():
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        QwenSparseAttnBackend,
    )

    seq = torch.tensor([10, 7], dtype=torch.int32)
    fb_verify = SimpleNamespace(seq_lens_cpu=seq, spec_info=SimpleNamespace(draft_token_num=3))
    fb_extend = SimpleNamespace(seq_lens_cpu=seq, spec_info=SimpleNamespace(num_tokens_per_req=3))
    fb_none = SimpleNamespace(seq_lens_cpu=seq, spec_info=None)
    f = QwenSparseAttnBackend._speculative_max_row_length
    assert f(fb_verify, seq) == 13
    assert f(fb_extend, seq) == 13
    assert f(fb_none, seq) == 10
