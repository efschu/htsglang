"""stage_skip_verdict / the skip re-check must take len() of a tensor prefix_indices, never `tensor or ()`.

y7m 13:17Z: the second, cached image request killed P PP0/1/2 with
"Boolean value of Tensor with no values is ambiguous" at vision_rank_runner stage_skip_verdict.
"""
import inspect

import torch

from sglang.srt.weg2 import vision_rank_runner as vrr


def test_no_tensor_or_on_prefix_indices():
    src = inspect.getsource(vrr)
    assert 'prefix_indices", None) or ()' not in src


def test_empty_and_full_tensor_prefix_len():
    for t, want in ((torch.empty(0, dtype=torch.int64), 0), (torch.arange(5), 5)):
        pi = t
        assert (0 if pi is None else len(pi)) == want
