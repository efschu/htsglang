"""#631: the mamba commit's step arithmetic uses the width that RAN.

THE ANSWER-CORRUPTING MEMBER of the configured-vs-ran width family, and
the one that bites a hybrid GDN model.

`commit_mamba_states_after_verify` recovers each request's accepted TREE
STEP -- the step whose recurrent state to commit -- by subtracting a
per-request offset from a GLOBAL node id:

    last_correct_step = accept_index[req, accept_lens - 1]
                        - arange(0, bs * W, W)[req]

`accept_index` holds node ids minted in the width THIS verify ran. Given
the CONFIGURED width instead, the subtraction becomes `i - 4i` on a
1-wide bootstrap round: negative step ids for every request except row 0,
whose offset is 0 in either coordinate system.

WHY IT SURFACES LATE, and why that made it hard. The damage is a
RECURRENT state written from the wrong step. Nothing raises, the KV and
append clocks stay in perfect agreement (`kv == seen - 1` on every row),
and the row decodes on from a subtly wrong linear-attention state and
DRIFTS -- measured as a wrong token roughly 28 tokens after the cutover,
always sparing batch row 0. Speculation off is clean because no verify
runs; the no-flip control is clean because the widths never differ
without a bootstrap round.

These pins cover the arithmetic itself, at the two widths, because that
is the part that can be checked without a GPU.
"""

import torch


def last_correct_step_indices(accept_index, accept_lens, draft_token_num):
    """The expression under test, lifted verbatim from
    spec_utils.commit_mamba_states_after_verify (the `bs`-strided offset
    subtracted from a global node id)."""
    bs = accept_lens.shape[0]
    offset = torch.arange(
        0,
        bs * draft_token_num,
        step=draft_token_num,
        dtype=accept_lens.dtype,
        device=accept_lens.device,
    )
    req_idx = torch.arange(bs, dtype=torch.int64)
    return accept_index[req_idx, (accept_lens - 1).to(torch.int64)] - offset


def _bootstrap_round(bs=3):
    """A 1-node trivial verify: one row per request, node id == row id,
    every request accepts its single root."""
    accept_index = torch.arange(bs, dtype=torch.int64).unsqueeze(1)
    accept_lens = torch.ones(bs, dtype=torch.int64)
    return accept_index, accept_lens


def test_narrowed_round_commits_step_zero_for_every_request():
    """THE FIX. Width 1: every request's single accepted node is step 0."""
    accept_index, accept_lens = _bootstrap_round()
    steps = last_correct_step_indices(accept_index, accept_lens, 1)
    assert steps.tolist() == [0, 0, 0]


def test_can_fail_the_configured_width_yields_negative_steps():
    """THE DEFECT, reproduced. Same 1-wide round, offsets built from the
    configured width of 4: row 0 is accidentally right and rows 1 and 2
    address steps that do not exist. This is the proof the pin above can
    fail."""
    accept_index, accept_lens = _bootstrap_round()
    steps = last_correct_step_indices(accept_index, accept_lens, 4)
    assert steps.tolist() == [0, -3, -6]
    assert steps[0].item() == 0, "row 0 is spared: offset 0 in both widths"
    assert (steps[1:] < 0).all(), "every later row addresses a negative step"


def test_ordinary_full_width_round_is_unchanged():
    """The two widths agree on every ordinary round, which is why the
    static value stood in. Pinned so the fix cannot regress it."""
    w = 4
    # Request i's nodes are ids [i*w, i*w+w); each accepts a different depth.
    accept_index = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    accept_lens = torch.tensor([4, 2, 1])
    steps = last_correct_step_indices(accept_index, accept_lens, w)
    assert steps.tolist() == [3, 1, 0]


def test_unequal_accept_lengths_do_not_by_themselves_break_it():
    """Unequal accept lengths are NORMAL and must stay correct: the
    defect is the width, not the split. Same widths, ragged accepts."""
    w = 4
    accept_index = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    for lens in ([4, 1, 1], [1, 4, 2], [2, 2, 4]):
        steps = last_correct_step_indices(accept_index, torch.tensor(lens), w)
        assert steps.tolist() == [x - 1 for x in lens]
