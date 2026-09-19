"""fn4n 19.09.: QwenSparseAttnBackend._capture_mtp_sparse_indices_from_extend_lens
reads spec_info.num_front_tokens; our EagleDraftExtendInput lacked the upstream
field and the NEXTN warmup died with AttributeError."""

import torch


def test_draft_extend_input_carries_num_front_tokens_default_zero():
    from sglang.srt.speculative.eagle_info import EagleDraftExtendInput

    si = EagleDraftExtendInput(
        num_correct_drafts=torch.tensor([1, 0]),
        num_accept_tokens=torch.tensor([2, 1]),
        num_tokens_per_req=3,
    )
    assert si.num_front_tokens == 0
    # anchor arithmetic of the capture path: block_end - (extend - front - accept)
    extend = torch.tensor([3, 3]); block_ends = extend.cumsum(0) - 1
    anchor = block_ends - (extend - si.num_front_tokens - si.num_accept_tokens)
    assert anchor.tolist() == [1, 3]
