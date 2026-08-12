# SPDX-License-Identifier: Apache-2.0
"""#656: the derived KV pool must default to a pool that has been proven to SERVE.

With the TP pool sized from the PP id space, a boot with no --max-total-tokens
sizes the pool to whatever the VRAM backs. On the 3-rank rig that reached
683150 tokens and produced a server that booted, held the 1024 MiB corridor
with zero breaches, answered /health with 200 -- and emitted no tokens at all
(#656 flip livelock). The same build serves at 620000.

So the default is the largest pool proven to serve, the operator can still
name any pool explicitly, and the env var exists to size to the hardware once
someone is ready to prove a larger pool with a real generation.
"""
import os

import pytest

from sglang.srt.server_args import ServerArgs

MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8-yarn1.5"
SERVING_PROVEN = 620000
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
    """The gate reads process env, so neighbouring tests must not leak into it."""
    before = os.environ.pop(ENV, None)
    yield
    os.environ.pop(ENV, None)
    if before is not None:
        os.environ[ENV] = before


def test_default_is_the_serving_proven_pool():
    assert ServerArgs(**_flip_args()).max_total_tokens == SERVING_PROVEN


def test_env_opt_out_sizes_to_the_hardware():
    """Left None, the sizer derives the pool from VRAM via the PP id space."""
    os.environ[ENV] = "1"
    assert ServerArgs(**_flip_args()).max_total_tokens is None


def test_explicit_operator_value_still_wins():
    """The gate is a DEFAULT, not a clamp: it must never lower a chosen value."""
    args = ServerArgs(**_flip_args(max_total_tokens=700000))
    assert args.max_total_tokens == 700000


def test_gate_does_not_touch_non_flip_boots():
    """A plain PP boot has no flip controller, so it has no livelock to dodge."""
    args = ServerArgs(
        model_path=MODEL,
        pp_size=3,
        pp_stage_ratio=[14, 10, 8],
        trust_remote_code=True,
    )
    assert args.max_total_tokens is None
