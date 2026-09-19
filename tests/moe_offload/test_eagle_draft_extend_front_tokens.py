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


def test_draft_extend_input_matches_the_upstream_field_set():
    """Every field upstream's EagleDraftExtendInput carries exists here (the
    #37500 MTP port reads them: num_front_tokens, select_index, ...)."""
    import dataclasses

    from sglang.srt.speculative.eagle_info import EagleDraftExtendInput

    ours = {f.name for f in dataclasses.fields(EagleDraftExtendInput)}
    upstream = {
        "hidden_states", "num_correct_drafts", "num_accept_tokens", "num_front_tokens",
        "num_accept_tokens_cpu", "input_ids", "seq_lens", "seq_lens_cpu", "req_pool_indices",
        "positions", "bonus_tokens", "capture_hidden_mode", "num_tokens_per_req",
        "num_tokens_for_logprob_per_req", "dsa_seed_topk_capture", "dsa_seed_topk_select",
        "select_index", "kv_indptr",
    }
    assert upstream - ours == set()
    assert EagleDraftExtendInput().select_index is None
