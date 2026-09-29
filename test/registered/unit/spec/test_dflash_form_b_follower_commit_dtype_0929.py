"""Form B follower head x the Triton accept buffers: _commit_accept takes a
bonus in the buffer's int32.

Metal, 29.09. 22:47:55Z (nvfp4lane/09292246, e923ba2468, first request): TP1,
the second Form B weight rank (a head, not the lead), died in
_commit_accept -> ``out_tokens.scatter_(..., bonus_tokens[:, None])``:
"scatter(): Expected self.dtype to be equal to src.dtype". The follower adopts
the lead's decision with ``bonus = _bon.to(bonus.dtype)`` -- and on the default
Triton accept path ``bonus`` is a view of ``_bonus_id_bufs`` (int32), while
``out_tokens`` is int64 by construction. The eager path hands int64 (dflash_utils
compute_dflash_correct_drafts_and_bonus), so only the Triton x follower pair
reached it.
"""

import torch

from sglang.srt.speculative.dflash_worker_v2 import _commit_accept
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _case():
    # bs=2, block 4: slot 0 = anchor, slots 1..3 = drafts
    candidates = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int64)
    accept_len = torch.tensor([1, 3], dtype=torch.int32)
    bonus = torch.tensor([99, 77], dtype=torch.int64)
    return candidates, accept_len, bonus


def test_follower_adoption_with_triton_buffer_dtypes():
    """The follower branch verbatim, with the Triton path's buffer dtypes."""
    candidates, accept_len, bonus64 = _case()
    # what the Triton accept path leaves bound to the names before adoption
    accept_len_buf = torch.zeros(2, dtype=torch.int32)
    bonus_buf = torch.zeros(2, dtype=torch.int32)
    # the lead's decision as _lane_accept_broadcast returns it: int64 rows
    _acc, _bon = accept_len.to(torch.int64), bonus64
    accept_len_f = _acc.to(accept_len_buf.dtype)
    bonus_f = _bon.to(bonus_buf.dtype)
    out_tokens, commit_lens = _commit_accept(candidates, accept_len_f, bonus_f)
    assert out_tokens.dtype == torch.int64
    assert out_tokens.tolist() == [[11, 99, 13, 0], [21, 22, 23, 77]]
    assert commit_lens.dtype == torch.int32 and commit_lens.tolist() == [2, 4]
    # byte-identical to the eager (int64 bonus) result the lead would derive
    ref, ref_lens = _commit_accept(candidates, accept_len, bonus64)
    assert torch.equal(out_tokens, ref) and torch.equal(commit_lens, ref_lens)


def test_eager_int64_path_unchanged():
    candidates, accept_len, bonus64 = _case()
    out_tokens, commit_lens = _commit_accept(candidates, accept_len, bonus64)
    assert out_tokens.tolist() == [[11, 99, 13, 0], [21, 22, 23, 77]]
    assert commit_lens.tolist() == [2, 4]
