"""The PLE batch's processed-token count on this line's ForwardBatch, which
has no DP-global non-padded count (fn1p boot 2026-09-16: first forward after
'ready' raised AttributeError on global_num_token_non_padded_cpu)."""

from types import SimpleNamespace

import pytest

from sglang.srt.models.qwen4_exp import _get_processed_token_count


def _fb(**kw):
    base = dict(extend_seq_lens_cpu=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_falls_through_global_then_local_then_extend_then_physical():
    assert _get_processed_token_count(_fb(global_num_token_non_padded_cpu=5), 8) == 5
    assert _get_processed_token_count(_fb(num_token_non_padded_cpu=6), 8) == 6
    assert _get_processed_token_count(_fb(extend_seq_lens_cpu=[3, 4]), 8) == 7
    assert _get_processed_token_count(_fb(), 8) == 8  # decode, nothing set


def test_a_count_beyond_the_physical_tokens_is_refused():
    with pytest.raises(RuntimeError, match="invalid PLE token counts"):
        _get_processed_token_count(_fb(num_token_non_padded_cpu=9), 8)
