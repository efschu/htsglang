# SPDX-License-Identifier: Apache-2.0
"""#656: the derived KV pool must be one whose SEAM every rank can fund.

HISTORY, because the shape of this gate changed and the reason matters.

With the TP pool sized from the PP id space, a boot with no
--max-total-tokens sizes the pool to whatever the VRAM backs. On the 3-rank
rig that reached 683150 tokens and produced a server that booted, held the
1024 MiB corridor with zero breaches, answered /health with 200 -- and emitted
no tokens at all (#656 flip livelock). The pool filled to the corridor floor
and left NOTHING for the flip seam, so every cutover was abandoned and, under
strict purity, a prefill could not be built in the TP phase at all.

The first response was a quarantine constant: default the pool to 620000, the
largest number the rig had been proven to SERVE. That constant always carried
its own deletion instruction -- "it should be deleted, not raised, when the
livelock is fixed" -- because a token count is not a physics number and,
worse, capping re-masked the per-rank capacity imbalance (register entry 43).

It is now deleted. The sizer reserves the seam (phase_flip_seam_reserve),
stands a margin back from its own measured position, and the policy counts
refusals instead of re-arming at the dwell interval. Measured 2026-08-13:
the derived pool is 651498 -- above the constant it replaced -- with 24
completed cutovers both directions, 0 abandons, 0 refusals, and a corridor
minimum of 1426/3305/1902 MiB under load.

So this file no longer gates a NUMBER. It gates the MECHANISM that makes any
number safe, which is the thing that must never regress.
"""
import os

import pytest

from sglang.srt.managers import phase_flip_seam_reserve as sr
from sglang.srt.server_args import ServerArgs

MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8-yarn1.5"
OLD_QUARANTINE = 620000
ENV = "SGLANG_PHASE_FLIP_UNPROVEN_POOL"


def _flip_args(**extra):
    return dict(
        model_path=MODEL,
        enable_phase_flip=True,
        phase_flip_tp_vector="32,16,16",
        pp_size=3,
        pp_stage_ratio=[14, 10, 8],
        trust_remote_code=True,
        **extra,
    )


@pytest.fixture(autouse=True)
def _clear_env():
    """These read process env, so neighbouring tests must not leak into them."""
    saved = {k: os.environ.pop(k, None) for k in (ENV, sr.ENV_ENABLE, sr.ENV_MARGIN_MIB)}
    yield
    for k, v in saved.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v


def test_the_quarantine_constant_is_gone():
    """No silent clamp: the pool is left to the seam-aware sizer.

    This is the assertion that would have failed before the fix, and it is
    the one that must not be quietly reverted to a magic number.
    """
    assert ServerArgs(**_flip_args()).max_total_tokens is None


def test_the_seam_reserve_is_the_net_and_it_is_on_by_default():
    """The mechanism that replaced the constant must be default-ON.

    A pool sized as "VRAM minus corridor", with nothing left to stage the
    seam, is precisely the livelock. It must not be constructible by
    accident -- only by an explicit opt-out.
    """
    assert sr.seam_reserve_enabled() is True


def test_the_seam_reserve_can_be_switched_off_only_explicitly():
    for off in ("0", "false", "no", "off", "OFF"):
        os.environ[sr.ENV_ENABLE] = off
        assert sr.seam_reserve_enabled() is False, off


def test_the_solver_never_ships_a_zero_margin_pool():
    """Boot K3 sized to its floor exactly and re-measured 25 MiB short.

    The margin is the second half of the net: without it the sizer lands ON
    the floor by construction and has nothing to absorb allocator drift.
    """
    assert sr.seam_margin_bytes() > 0


def test_explicit_operator_value_still_wins():
    """Unchanged: naming a pool has always been the operator's right."""
    args = ServerArgs(**_flip_args(max_total_tokens=700000))
    assert args.max_total_tokens == 700000


def test_a_pool_above_the_old_quarantine_is_no_longer_refused():
    """651498 was derived and served on metal; 620000 must not still bind."""
    args = ServerArgs(**_flip_args(max_total_tokens=OLD_QUARANTINE + 31498))
    assert args.max_total_tokens == 651498


def test_the_unproven_pool_env_is_no_longer_needed_to_derive():
    """The opt-out existed only to escape the clamp; with no clamp, the
    derived path is the default and the env changes nothing about it."""
    os.environ[ENV] = "1"
    assert ServerArgs(**_flip_args()).max_total_tokens is None


if __name__ == "__main__":
    pytest.main([__file__])
