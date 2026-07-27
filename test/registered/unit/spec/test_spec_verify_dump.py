# SPDX-License-Identifier: Apache-2.0
"""Pure tests for the speculative verify dump (#143 correctness oracle).

The dump exists because the #124 determinism dump cannot see the spec path
at all: ``ModelRunner._determinism_dump_logits`` hangs off
``ModelRunner.sample`` and early-returns unless ``logits.shape[0] == 1``,
while a target verify never reaches that call site and carries
``bs * (k + 1)`` rows. Without this, candidate (b) of the oracle -- compare
the verified distribution rather than the drawn token -- has no tap.

The load-bearing pure logic is :func:`accepted_row_indices`: which verify
row produced which emitted token. Everything downstream (the harness
projection, the near-tie margin, the accept-length floor) is indexed by it.
"""

import pytest
import torch

from sglang.srt.speculative.spec_verify_dump import (
    accepted_row_indices,
    build_verify_record,
)


def test_accepted_row_indices_chain_all_accepted():
    """Chain layout, bs=1, k=3: accept_index is the flat verify-row index of
    each accepted node, so all four rows in order."""
    accept_index = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    accept_lens = torch.tensor([4], dtype=torch.int32)
    assert accepted_row_indices(accept_index, accept_lens) == [[0, 1, 2, 3]]


def test_accepted_row_indices_partial_accept_truncates():
    accept_index = torch.tensor([[0, 1, -1, -1]], dtype=torch.int32)
    accept_lens = torch.tensor([2], dtype=torch.int32)
    assert accepted_row_indices(accept_index, accept_lens) == [[0, 1]]


def test_accepted_row_indices_multi_request_is_offset_by_draft_block():
    """accept_index carries GLOBAL flat row indices, so request 1's rows are
    already offset by draft_token_num -- the reader must not add its own."""
    accept_index = torch.tensor([[0, 1, -1, -1], [4, 5, 6, -1]], dtype=torch.int32)
    accept_lens = torch.tensor([2, 3], dtype=torch.int32)
    assert accepted_row_indices(accept_index, accept_lens) == [[0, 1], [4, 5, 6]]


def test_accepted_row_indices_rejects_a_negative_inside_the_accepted_span():
    """A -1 inside the accepted span means accept_lens and accept_index
    disagree -- a plumbing bug, and silently indexing row -1 would produce a
    plausible-looking wrong trajectory."""
    accept_index = torch.tensor([[0, -1, 2, 3]], dtype=torch.int32)
    accept_lens = torch.tensor([3], dtype=torch.int32)
    with pytest.raises(ValueError, match="accept_index"):
        accepted_row_indices(accept_index, accept_lens)


def test_accepted_row_indices_rejects_a_zero_accept_len():
    accept_index = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    accept_lens = torch.tensor([0], dtype=torch.int32)
    with pytest.raises(ValueError, match="at least"):
        accepted_row_indices(accept_index, accept_lens)


def test_accepted_row_indices_rejects_out_of_range():
    accept_index = torch.tensor([[0, 9]], dtype=torch.int32)
    accept_lens = torch.tensor([2], dtype=torch.int32)
    with pytest.raises(ValueError, match="out of range"):
        accepted_row_indices(accept_index, accept_lens, num_rows=4)


def _fake_step(vocab=8, bs=1, d=4):
    logits = torch.zeros(bs * d, vocab, dtype=torch.float32)
    for r in range(bs * d):
        logits[r, (r + 1) % vocab] = 5.0
    predict = torch.tensor(
        [(r + 1) % vocab for r in range(bs * d)], dtype=torch.int32
    )
    candidates = torch.tensor([[0] + [(r + 1) % vocab for r in range(d - 1)]] * bs)
    accept_index = torch.tensor([list(range(d))] * bs, dtype=torch.int32)
    accept_lens = torch.tensor([d] * bs, dtype=torch.int32)
    return logits, predict, candidates, accept_index, accept_lens


def test_build_verify_record_shape_and_content():
    logits, predict, candidates, accept_index, accept_lens = _fake_step()
    rec = build_verify_record(
        step=7,
        tp_rank=0,
        logits=logits,
        candidates=candidates,
        predict=predict,
        accept_index=accept_index,
        accept_lens=accept_lens,
        draft_token_num=4,
    )
    assert rec["mode"] == "target_verify"
    assert rec["step"] == 7
    assert rec["bs"] == 1
    assert rec["draft_token_num"] == 4
    assert rec["accepted_rows"] == [[0, 1, 2, 3]]
    assert rec["emitted"] == [[1, 2, 3, 4]]
    # The FULL verify matrix is kept, not only the accepted rows: the
    # rejected slots are what explain an accept-length difference between
    # two arms.
    assert rec["logits"].shape == (4, 8)
    assert rec["logits"].device.type == "cpu"


def test_build_verify_record_emitted_tokens_are_their_rows_argmax():
    """The record must satisfy the invariant the oracle then asserts, or the
    dump itself is the bug rather than the run."""
    logits, predict, candidates, accept_index, accept_lens = _fake_step()
    rec = build_verify_record(
        step=0,
        tp_rank=0,
        logits=logits,
        candidates=candidates,
        predict=predict,
        accept_index=accept_index,
        accept_lens=accept_lens,
        draft_token_num=4,
    )
    for rows, toks in zip(rec["accepted_rows"], rec["emitted"]):
        for row, tok in zip(rows, toks):
            assert int(rec["logits"][row].argmax()) == tok


def test_build_verify_record_detaches_and_copies():
    logits, predict, candidates, accept_index, accept_lens = _fake_step()
    logits.requires_grad_(False)
    rec = build_verify_record(
        step=0,
        tp_rank=0,
        logits=logits,
        candidates=candidates,
        predict=predict,
        accept_index=accept_index,
        accept_lens=accept_lens,
        draft_token_num=4,
    )
    logits[0, 0] = 99.0
    assert float(rec["logits"][0, 0]) != 99.0


def test_build_verify_record_preserves_dtype():
    """The byte-identity classes are dtype-strict; the dump must not upcast."""
    logits, predict, candidates, accept_index, accept_lens = _fake_step()
    rec = build_verify_record(
        step=0,
        tp_rank=0,
        logits=logits.to(torch.bfloat16),
        candidates=candidates,
        predict=predict,
        accept_index=accept_index,
        accept_lens=accept_lens,
        draft_token_num=4,
    )
    assert rec["logits"].dtype == torch.bfloat16


def test_build_verify_record_writes_atomically(tmp_path):
    from sglang.srt.speculative.spec_verify_dump import write_verify_record

    logits, predict, candidates, accept_index, accept_lens = _fake_step()
    rec = build_verify_record(
        step=3,
        tp_rank=1,
        logits=logits,
        candidates=candidates,
        predict=predict,
        accept_index=accept_index,
        accept_lens=accept_lens,
        draft_token_num=4,
    )
    path = write_verify_record(str(tmp_path), rec)
    assert path.endswith("rank1_verify0000003.pt")
    assert not list(tmp_path.glob("*.tmp"))
    back = torch.load(path, weights_only=False)
    assert back["emitted"] == rec["emitted"]
