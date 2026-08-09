"""#631: causal_conv1d_update's batch-vs-indices bound is not optional.

THE SPECIMEN, /spinning/evidence-631/oom_and_abandon_20260809T0521Z,
2026-08-09 05:21:16Z. A pp_to_tp flip abandoned at its 30 s park deadline
under load, and PP2 died. The first fault in the log is a ValueError from
fused_recurrent_gated_delta_rule_packed_decode:

    `ssm_state_indices` must have shape [B] (got (1,); expected (2048,))

-- a DECODE batch of one request carrying 2048 rows of hidden state, 2048
being exactly that boot's chunked_prefill_size. A prefill chunk's hidden
states had been paired with a decode batch's cache indices.

THE POINT OF THIS FILE IS THE KERNEL THAT RAN FIRST. One call earlier,
gdn_backend.forward_decode hands the SAME mismatched pair to
causal_conv1d_update. It launches one program per row of x, and each
program reads conv_state_indices[row]; with 2048 rows and a 1-element
index tensor, 2047 of those reads are out of bounds and the garbage they
return is used as a conv-state line number -- so the write goes to an
unowned line. That is the "illegal memory access" the run was blamed for.
It surfaced a second later inside barlink's BAR1 status poll, because a
sticky CUDA fault reports at the next synchronising call, and that sent
the first investigation into the wrong subsystem entirely.

The assert for exactly this existed already -- behind ``validate_data``,
which defaults to False, so it was compiled out on every real call. A
bounds check that is off in production is not a bounds check.

CPU-only: the guard runs before any device work, which is the whole point.
"""

import pytest
import torch

from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_update,
)


def _args(batch, n_indices, dim=8, width=4, lines=16):
    x = torch.zeros(batch, dim, dtype=torch.float32)
    conv_state = torch.zeros(lines, dim, width - 1, dtype=torch.float32)
    weight = torch.zeros(dim, width, dtype=torch.float32)
    idx = torch.zeros(n_indices, dtype=torch.int32)
    return x, conv_state, weight, idx


def test_a_short_index_tensor_is_refused_before_the_launch():
    """THE CAN-FAIL. Exactly the specimen's shapes: 2048 rows, 1 index."""
    x, conv_state, weight, idx = _args(batch=2048, n_indices=1)

    with pytest.raises(ValueError) as exc:
        causal_conv1d_update(x, conv_state, weight, conv_state_indices=idx)

    msg = str(exc.value)
    assert "1 entr" in msg and "2048" in msg, msg
    assert "conv_state" in msg, "the message must name what would be corrupted"


def test_the_guard_fires_without_validate_data():
    """It must protect the DEFAULT call, which is the one production makes.

    This is the entire defect: the pre-existing assert was correct and
    unreachable.
    """
    x, conv_state, weight, idx = _args(batch=4, n_indices=2)
    with pytest.raises(ValueError):
        causal_conv1d_update(
            x, conv_state, weight, conv_state_indices=idx, validate_data=False
        )


def test_a_longer_index_tensor_is_NOT_refused():
    """Only the out-of-bounds direction is an error.

    Surplus entries are never addressed by the kernel, so refusing them
    would break working callers for no safety gain -- and a guard that can
    break a correct call is one nobody dares leave on. Proven by the shape
    of the failure: it must get PAST the guard and die in the kernel
    launch instead (no triton on this CPU box).
    """
    x, conv_state, weight, idx = _args(batch=2, n_indices=8)

    with pytest.raises(Exception) as exc:
        causal_conv1d_update(x, conv_state, weight, conv_state_indices=idx)

    assert "entr(ies) for a batch of" not in str(exc.value), (
        "the guard refused a HARMLESS longer index tensor; it must only "
        "reject the short direction"
    )


def test_no_indices_at_all_is_untouched():
    """The guard may not disturb the conv_state_indices=None path."""
    x, conv_state, weight, _ = _args(batch=2, n_indices=1)

    with pytest.raises(Exception) as exc:
        causal_conv1d_update(x, conv_state, weight, conv_state_indices=None)

    assert "entr(ies) for a batch of" not in str(exc.value)
