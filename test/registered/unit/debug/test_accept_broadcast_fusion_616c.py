"""#616c: the accept broadcast must not be silently substitutable.

Background (docs/dev/NOTE_616c_index_values.md). A GPU coredump taken with
global memory retained showed the index tensor of `predict[accept_index]` and
the tensor it gathers from holding element-wise IDENTICAL values, with the
non-accepted slots carrying 0 -- `predict`'s zero-init, not `accept_index`'s
-1 fill. So `accept_index` was carrying a whole, well-formed `predict`. All
three observed faults landed on RECEIVING ranks; rank 0 is the broadcast
source and only reads its own buffers.

The enabling condition is that `predict` and `accept_index` are both int32
with `bs * draft_token_num` elements: as separate collectives their payloads
are the same SIZE, so a one-collective pairing shift substitutes one for the
other with nothing for NCCL to reject.

These tests pin the fused payload as the fix, and -- the part that matters --
demonstrate on the OLD shape that the substitution really is silent, so the
new shape's loudness is a measured difference rather than an assertion.

Hermetic: CPU tensors only, no CUDA, no distributed backend.
"""

import pytest
import torch

from sglang.srt.speculative.eagle_utils import (
    accept_payload_lengths,
    pack_accept_payload,
    unpack_accept_payload,
)

BS = 3
DRAFT_TOKEN_NUM = 4
MAX_TREE_DEPTH = 4


def _make_triplet():
    """The three accept-result tensors, shaped exactly as eagle_sample builds
    them, and filled with the signature from the coredump."""
    # predict: zero-init, written only at accepted slots, values are token IDs.
    predict = torch.zeros(BS * DRAFT_TOKEN_NUM, dtype=torch.int32)
    predict[0] = 21966
    predict[4], predict[5], predict[6] = 2286, 1098, 1510
    predict[8], predict[9], predict[10], predict[11] = 1204, 280, 3173, 6326
    # accept_index: -1 fill, chain offsets at the accepted slots.
    accept_index = torch.full((BS, MAX_TREE_DEPTH), -1, dtype=torch.int32)
    accept_index[0, 0] = 0
    accept_index[1, 0], accept_index[1, 1], accept_index[1, 2] = 4, 5, 6
    accept_index[2, 0], accept_index[2, 1] = 8, 9
    accept_index[2, 2], accept_index[2, 3] = 10, 11
    num_correct_drafts = torch.tensor([0, 2, 3], dtype=torch.int32)
    return predict, accept_index, num_correct_drafts


class _Wire:
    """A broadcast transport that can be put one collective out of phase.

    ``shift=0`` is a healthy wire. ``shift=1`` models the receiver consuming
    the payload of the PREVIOUS collective -- the condition the coredump is
    consistent with. Size disagreement raises, exactly as NCCL would.
    """

    def __init__(self, shift: int = 0):
        self.sent = []
        self.shift = shift

    def send(self, t: torch.Tensor) -> None:
        self.sent.append(t.detach().clone())

    def recv_into(self, dst: torch.Tensor, seq: int) -> None:
        src = self.sent[seq - self.shift]
        if src.numel() != dst.numel():
            raise RuntimeError(
                f"size mismatch: peer sent {src.numel()}, receiver expected "
                f"{dst.numel()}"
            )
        dst.reshape(-1).copy_(src.reshape(-1))


def test_pack_unpack_roundtrip_is_exact_and_in_place():
    predict, accept_index, num_correct_drafts = _make_triplet()
    want = (predict.clone(), accept_index.clone(), num_correct_drafts.clone())

    packed = pack_accept_payload(predict, accept_index, num_correct_drafts)
    assert packed.dtype == torch.int32
    assert packed.numel() == sum(
        accept_payload_lengths(predict, accept_index, num_correct_drafts)
    )

    # Scribble over the destinations so a no-op unpack cannot pass.
    predict.zero_()
    accept_index.fill_(-99)
    num_correct_drafts.zero_()

    ids = (id(predict), id(accept_index), id(num_correct_drafts))
    unpack_accept_payload(packed, predict, accept_index, num_correct_drafts)

    assert torch.equal(predict, want[0])
    assert torch.equal(accept_index, want[1])
    assert torch.equal(num_correct_drafts, want[2])
    # In place: downstream aliases must keep seeing the same objects.
    assert (id(predict), id(accept_index), id(num_correct_drafts)) == ids


def test_old_three_collective_shape_substitutes_predict_SILENTLY():
    """CAN-FAIL PROOF, on the pre-fix shape.

    Three same-sized collectives + a one-collective shift == accept_index
    arrives holding predict, with no error raised anywhere. This is the fault
    the coredump recorded; if this test ever stops reproducing it, the
    premise of the fix is wrong and the fix should be re-argued.
    """
    predict, accept_index, num_correct_drafts = _make_triplet()
    predict_sent = predict.clone()

    wire = _Wire(shift=1)
    # Source rank issues the three collectives in order.
    for t in (predict, accept_index, num_correct_drafts):
        wire.send(t)

    # Receiver consumes them one collective out of phase. Collective #1
    # (accept_index) therefore takes the payload of collective #0 (predict).
    recv_predict = torch.empty_like(predict)
    recv_accept = torch.empty_like(accept_index)
    wire.send(predict)  # keep the wire long enough to index seq-1 for seq 0
    wire.recv_into(recv_accept, seq=1)

    # It is silent: identical sizes, no exception, and accept_index now holds
    # predict -- including predict's ZERO fill where accept_index had -1.
    assert torch.equal(recv_accept.reshape(-1), predict_sent.reshape(-1))
    assert (recv_accept == 0).any(), "predict's zero-init should be visible"
    assert not (recv_accept == -1).any(), "accept_index's -1 fill is gone"

    # And that is precisely what makes the downstream gather illegal.
    bound = predict.numel()
    assert (recv_accept.reshape(-1) >= bound).any(), (
        "token-ID magnitudes must exceed predict.numel() -- this is the "
        "out-of-bounds condition IndexKernel.cu:111 asserts on"
    )
    del recv_predict


def test_fused_payload_makes_the_same_shift_LOUD():
    """The fix: one fused collective, whose length matches none of the three
    individually, so the same shift can no longer be absorbed silently."""
    predict, accept_index, num_correct_drafts = _make_triplet()
    n_p, n_a, n_c = accept_payload_lengths(predict, accept_index, num_correct_drafts)
    packed = pack_accept_payload(predict, accept_index, num_correct_drafts)

    # The enabling coincidence is gone at the type level.
    assert n_p == n_a, "precondition: this is why the old shape was silent"
    assert packed.numel() not in (n_p, n_a, n_c)

    # A neighbouring collective of any of the old sizes now fails to pair.
    wire = _Wire(shift=1)
    wire.send(torch.zeros(n_p, dtype=torch.int32))  # some earlier collective
    wire.send(packed)
    with pytest.raises(RuntimeError, match="size mismatch"):
        wire.recv_into(torch.empty_like(packed), seq=1)


def test_unpack_rejects_a_wrong_length_payload():
    """Second line of defence: even if a same-length-but-wrong payload were
    delivered, a shape disagreement between ranks is refused loudly."""
    predict, accept_index, num_correct_drafts = _make_triplet()
    packed = pack_accept_payload(predict, accept_index, num_correct_drafts)

    with pytest.raises(ValueError, match="payload length mismatch"):
        unpack_accept_payload(packed[:-1], predict, accept_index, num_correct_drafts)
