# SPDX-License-Identifier: Apache-2.0
"""#656 T3: the YaRN cos/sin cache must extend with the SAME parameters it was
built with.

The scaled RoPE variants build their cache from
``_compute_inv_freq(self.scaling_factor)`` and multiply it by ``self.mscale``.
``RotaryEmbedding._ensure_cos_sin_cache_length`` -- the path that appends rows
when a position beyond the boot-time cache is requested -- used to call
``_compute_inv_freq(self.base)`` and apply no mscale at all. That passes the
RoPE theta where a scaling factor is expected, so every appended row carried
different frequencies and the wrong amplitude from the rows before it.

Nothing raises when that happens. The failure is silent wrong attention past
the boot cache length, which is exactly the region a raised context ceiling
makes reachable.
"""

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.rotary_embedding import lazy_cos_sin_cache
from sglang.srt.layers.rotary_embedding.yarn import YaRNScalingRotaryEmbedding
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs

# RotaryEmbedding.__init__ reads the process-wide ServerArgs. These tests
# construct the layer directly, outside a server, so install a minimal one.
get_context().set_server_args(ServerArgs(model_path="dummy"))

HEAD_SIZE = 64
ROTARY_DIM = 64
MAX_POS = 512
BASE = 10000
SCALING_FACTOR = 4.0


# #656 T1: every test here runs twice. The lazy reserve writes its rows
# through the SAME growth path this file was written to guard, so if the two
# ever diverge, the lazy arm is the one that says so. In the lazy arm the
# cache TENSOR is the full reservation and only `filled` rows are written, so
# the tests measure the seam at `filled`, not at shape[0].
@pytest.fixture(params=["eager", "lazy"], autouse=True)
def rope_mode(request):
    with (
        envs.SGLANG_ROPE_LAZY_CACHE.override(request.param == "lazy"),
        envs.SGLANG_ROPE_LAZY_CHUNK_ROWS.override(128),
        envs.SGLANG_ROPE_LAZY_MIN_ROWS.override(64),
    ):
        yield request.param
    for module in list(lazy_cos_sin_cache._LAZY_MODULES):
        lazy_cos_sin_cache.drop(module)


def _written_rows(rope) -> int:
    """Rows that carry values: the reservation is longer than the fill."""
    state = getattr(rope, "_lazy_cos_sin", None)
    return state.filled if state is not None else int(rope.cos_sin_cache.shape[0])


def _build():
    return YaRNScalingRotaryEmbedding(
        head_size=HEAD_SIZE,
        rotary_dim=ROTARY_DIM,
        max_position_embeddings=MAX_POS,
        base=BASE,
        is_neox_style=True,
        scaling_factor=SCALING_FACTOR,
        dtype=torch.float32,
    )


def test_appended_rows_match_the_constructed_rows():
    """Rows added by growth must obey the same formula as rows built at init.

    This is the regression guard. Before the fix the appended block was
    computed with inv_freq(self.base) and without mscale, so it failed here by
    a wide margin rather than by a rounding tolerance.
    """
    rope = _build()
    grown_from = _written_rows(rope)

    rope._ensure_cos_sin_cache_length(grown_from + 128)
    grown_to = _written_rows(rope)
    cache = rope.cos_sin_cache[:grown_to]
    assert grown_to > grown_from, "growth path did not extend the cache"

    # Reference: the identity _compute_cos_sin_cache itself uses.
    inv_freq = rope._compute_inv_freq(SCALING_FACTOR)
    t = torch.arange(grown_from, cache.shape[0], dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    want = torch.cat((freqs.cos() * rope.mscale, freqs.sin() * rope.mscale), dim=-1).to(
        dtype=cache.dtype
    )

    got = cache[grown_from:]
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_growth_is_continuous_across_the_seam():
    """No discontinuity at the join.

    Independent of the reference above: cos of adjacent positions must stay
    close for the low-frequency components. A cache extended with the wrong
    inv_freq jumps at the seam even though both halves are individually
    smooth, so this catches the bug without restating the formula.
    """
    rope = _build()
    seam = _written_rows(rope)
    rope._ensure_cos_sin_cache_length(seam + 128)
    cache = rope.cos_sin_cache[: _written_rows(rope)]

    # Slowest-rotating component: adjacent positions must barely differ.
    slowest = cache[:, ROTARY_DIM // 2 - 1]
    step_before = (slowest[seam - 1] - slowest[seam - 2]).abs()
    step_across = (slowest[seam] - slowest[seam - 1]).abs()
    assert step_across <= step_before * 5 + 1e-6, (
        f"discontinuity at the growth seam: step across {step_across:.3e} vs "
        f"step before {step_before:.3e}"
    )


def test_mscale_is_actually_applied():
    """Amplitude guard.

    yarn_get_mscale_simple(4.0) > 1, so appended rows built without mscale are
    uniformly too small. Asserting on the max magnitude catches the missing
    multiply even if the frequencies happened to match.
    """
    rope = _build()
    seam = _written_rows(rope)
    assert rope.mscale > 1.0, "test needs a scaling factor whose mscale != 1"

    rope._ensure_cos_sin_cache_length(seam + 128)
    appended = rope.cos_sin_cache[seam : _written_rows(rope)]
    assert appended.abs().max() > 1.0 + 1e-3, (
        "appended rows never exceed 1.0, so mscale was not applied "
        f"(mscale={rope.mscale})"
    )
