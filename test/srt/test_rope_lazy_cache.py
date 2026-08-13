# SPDX-License-Identifier: Apache-2.0
"""#656 T1: the lazy RoPE reserve must cost nothing and lie about nothing.

A lazy cos/sin cache reserves the context ceiling and writes rows only as
positions reach them. That buys back the 440 MiB per rank a 1048576 ceiling
costs today (register 69), and it introduces exactly one new way to be wrong:
a row that is READ before it is WRITTEN. Unwritten reservation is not an
error condition -- it reads as whatever was in the pages -- so every test here
is about the boundary between filled and reserved.

The rows themselves are register 47's problem restated: the lazy fill and the
growth path must produce the rows the constructor would have produced, from
the SAME frequencies and the SAME amplitude. Those tests live in
test_yarn_rope_cache_growth.py and are parametrized over the lazy path there.
"""

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.rotary_embedding import lazy_cos_sin_cache
from sglang.srt.layers.rotary_embedding.yarn import YaRNScalingRotaryEmbedding
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs

get_context().set_server_args(ServerArgs(model_path="dummy"))

HEAD_SIZE = 64
ROTARY_DIM = 64
MAX_POS = 512
BASE = 10000
SCALING_FACTOR = 4.0
EAGER_ROWS = int(MAX_POS * SCALING_FACTOR)  # 2048
CHUNK = 256
MIN_ROWS = 1024


@pytest.fixture(autouse=True)
def _lazy_env():
    """The knobs are read where they are used -- the chunk size at every
    growth, not once at construction -- so the overrides have to span the
    whole test, not just the constructor."""
    with (
        envs.SGLANG_ROPE_LAZY_CHUNK_ROWS.override(CHUNK),
        envs.SGLANG_ROPE_LAZY_MIN_ROWS.override(MIN_ROWS),
    ):
        yield
    for module in list(lazy_cos_sin_cache._LAZY_MODULES):
        lazy_cos_sin_cache.drop(module)


def _build(lazy: bool, min_rows: int = MIN_ROWS):
    with (
        envs.SGLANG_ROPE_LAZY_CACHE.override(lazy),
        envs.SGLANG_ROPE_LAZY_MIN_ROWS.override(min_rows),
    ):
        return YaRNScalingRotaryEmbedding(
            head_size=HEAD_SIZE,
            rotary_dim=ROTARY_DIM,
            max_position_embeddings=MAX_POS,
            base=BASE,
            is_neox_style=True,
            scaling_factor=SCALING_FACTOR,
            dtype=torch.float32,
        )


def _reference_rows(rope, start, end):
    """What the constructor would have built for rows [start, end)."""
    inv_freq = rope._compute_inv_freq(SCALING_FACTOR)
    t = torch.arange(start, end, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos() * rope.mscale, freqs.sin() * rope.mscale), dim=-1)


def test_default_is_eager_and_unchanged():
    """The shipped default must not move. This is the backward-compat law."""
    rope = _build(lazy=False)
    assert rope._lazy_cos_sin is None
    assert int(rope.cos_sin_cache.shape[0]) == EAGER_ROWS
    assert not lazy_cos_sin_cache.any_installed()


def test_lazy_reserves_the_ceiling_and_fills_one_chunk():
    rope = _build(lazy=True)
    state = rope._lazy_cos_sin
    assert state is not None, "lazy cache did not install"
    # The reservation covers what the eager cache would have been ...
    assert state.capacity == EAGER_ROWS
    assert int(rope.cos_sin_cache.shape[0]) == EAGER_ROWS
    # ... and only the first chunk is written.
    assert state.filled == CHUNK


def test_lazy_prefix_is_the_eager_prefix():
    """Filled rows must equal the rows the eager constructor would produce.

    If this fails, the lazy path is a different model, not a cheaper one.
    """
    eager = _build(lazy=False)
    lazy = _build(lazy=True)
    filled = lazy._lazy_cos_sin.filled
    torch.testing.assert_close(
        lazy.cos_sin_cache[:filled], eager.cos_sin_cache[:filled], rtol=0, atol=0
    )


def test_growth_fills_in_place_and_the_address_never_moves():
    """The CUDA-graph safety property, stated as a test.

    A captured decode graph holds the ADDRESS of cos_sin_cache. Growth by
    reallocation (torch.cat) leaves every replay reading a freed buffer, which
    is why the lazy path fills a reservation instead. If this assertion ever
    fails, graph replay after growth is unsound.
    """
    rope = _build(lazy=True)
    before = rope.cos_sin_cache.data_ptr()
    rope._ensure_cos_sin_cache_length(700)
    assert rope.cos_sin_cache.data_ptr() == before
    assert rope._lazy_cos_sin.filled >= 701
    assert int(rope.cos_sin_cache.shape[0]) == EAGER_ROWS


def test_grown_rows_are_the_rows_the_constructor_would_have_built():
    rope = _build(lazy=True)
    start = rope._lazy_cos_sin.filled
    rope._ensure_cos_sin_cache_length(700)
    end = rope._lazy_cos_sin.filled
    torch.testing.assert_close(
        rope.cos_sin_cache[start:end],
        _reference_rows(rope, start, end),
        rtol=1e-5,
        atol=1e-5,
    )


def test_growth_is_chunked_not_row_by_row():
    """A fill per position would put a kernel launch on the decode path."""
    rope = _build(lazy=True)
    rope._ensure_cos_sin_cache_length(CHUNK + 1)
    assert rope._lazy_cos_sin.filled == CHUNK * 2


def test_past_the_reservation_relinquishes_and_materializes_the_tail():
    """Overrunning the reservation must not leave unwritten rows readable.

    The reservation is uninitialized memory past `filled`. When the layer
    stops being lazy, those rows have to be WRITTEN, not merely declared
    present -- otherwise the fallback hands out garbage with a correct shape.
    """
    rope = _build(lazy=True)
    rope._ensure_cos_sin_cache_length(EAGER_ROWS + 10)
    assert rope._lazy_cos_sin is None, "should have relinquished"
    assert int(rope.cos_sin_cache.shape[0]) > EAGER_ROWS
    # every row, including the ones laziness had deferred
    torch.testing.assert_close(
        rope.cos_sin_cache[:EAGER_ROWS],
        _reference_rows(rope, 0, EAGER_ROWS),
        rtol=1e-5,
        atol=1e-5,
    )


def test_dtype_conversion_materializes_the_tail_first():
    """A .to(dtype) copies the buffer -- including rows never written."""
    rope = _build(lazy=True)
    query = torch.empty(1, dtype=torch.float16)
    rope._match_cos_sin_cache_dtype(query)
    assert rope._lazy_cos_sin is None
    assert rope.cos_sin_cache.dtype == torch.float16
    torch.testing.assert_close(
        rope.cos_sin_cache.float(),
        _reference_rows(rope, 0, EAGER_ROWS).float(),
        rtol=1e-2,
        atol=1e-2,
    )


def test_capacity_request_reserves_without_filling():
    """A reserve is a ceiling statement; prepaying it is the cost we removed."""
    rope = _build(lazy=True)
    rope.ensure_cos_sin_cache_capacity(EAGER_ROWS * 4)
    state = rope._lazy_cos_sin
    assert state is not None
    assert state.capacity == EAGER_ROWS * 4
    assert state.filled == CHUNK, "the reserve must not have written rows"
    assert int(rope.cos_sin_cache.shape[0]) == EAGER_ROWS * 4
    # the rows that WERE filled survived the re-reservation
    torch.testing.assert_close(
        rope.cos_sin_cache[:CHUNK],
        _reference_rows(rope, 0, CHUNK),
        rtol=1e-5,
        atol=1e-5,
    )


def test_verify_guard_can_fail_and_then_stops_failing():
    """The guard has to be able to fail, or it proves nothing."""
    rope = _build(lazy=True)
    filled = rope._lazy_cos_sin.filled
    with pytest.raises(AssertionError, match="read past the fill"):
        lazy_cos_sin_cache.verify_positions_are_filled(filled, where="test")
    lazy_cos_sin_cache.verify_positions_are_filled(filled - 1, where="test")
    rope._ensure_cos_sin_cache_length(filled + 10)
    lazy_cos_sin_cache.verify_positions_are_filled(filled, where="test")


def test_a_cache_below_the_threshold_stays_eager():
    """Small caches are not worth a reservation, and must not get one."""
    rope = _build(lazy=True, min_rows=EAGER_ROWS + 1)
    assert rope._lazy_cos_sin is None
    assert int(rope.cos_sin_cache.shape[0]) == EAGER_ROWS


def test_an_unverified_subclass_declines_the_lazy_path():
    """Unknown growth hooks must not be trusted: register 47's failure mode
    is silent, so the allowlist is the safety property."""

    class UnverifiedYaRN(YaRNScalingRotaryEmbedding):
        pass

    with envs.SGLANG_ROPE_LAZY_CACHE.override(True):
        rope = UnverifiedYaRN(
            head_size=HEAD_SIZE,
            rotary_dim=ROTARY_DIM,
            max_position_embeddings=MAX_POS,
            base=BASE,
            is_neox_style=True,
            scaling_factor=SCALING_FACTOR,
            dtype=torch.float32,
        )
    assert rope._lazy_cos_sin is None
    assert int(rope.cos_sin_cache.shape[0]) == EAGER_ROWS


def test_the_batch_hook_is_free_when_nothing_is_lazy():
    """The default path must not pay for a feature it did not enable."""
    _build(lazy=False)
    assert not lazy_cos_sin_cache.any_installed()
    lazy_cos_sin_cache.ensure_capacity_for_position(10**9)  # must not raise


def test_the_batch_hook_grows_every_installed_cache():
    """One hook, every stack: the phase-flip instance holds two model stacks
    and a draft model, and they share RoPE instances through _ROPE_DICT."""
    a = _build(lazy=True)
    b = _build(lazy=True)
    lazy_cos_sin_cache.ensure_capacity_for_position(900)
    assert a._lazy_cos_sin.filled >= 901
    assert b._lazy_cos_sin.filled >= 901


def test_written_rows_reports_the_fill_not_the_reservation():
    """The distinction every caller that caches a length has to respect.

    Under a lazy reserve the TENSOR is the whole reservation, so a caller that
    remembers cos_sin_cache.shape[0] as "how much do I have" concludes it
    never needs to grow again -- which is how the fused KV materialization
    path would have stopped growing after its first call.
    """
    rope = _build(lazy=True)
    assert int(rope.cos_sin_cache.shape[0]) == EAGER_ROWS
    assert lazy_cos_sin_cache.written_rows(rope) == CHUNK
    rope._ensure_cos_sin_cache_length(700)
    assert lazy_cos_sin_cache.written_rows(rope) >= 701
    eager = _build(lazy=False)
    assert lazy_cos_sin_cache.written_rows(eager) == EAGER_ROWS
