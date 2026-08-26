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
import torch

from sglang.srt.managers import phase_flip_seam_reserve as sr
from sglang.srt.server_args import ServerArgs

#: THIS FILE NEEDS A VISIBLE DEVICE, and says so instead of failing.
#:
#: Every case here builds a real ``ServerArgs``, whose ``__post_init__``
#: resolves a device; without one it raises ``RuntimeError: No accelerator
#: (CUDA, XPU, HPU, NPU, MUSA, MPS) or platform plugin is available`` before
#: any assertion in this file is reached. Measured under the canonical
#: CPU-only desk protocol (``CUDA_VISIBLE_DEVICES=99``): 4 failed, 3 passed;
#: with a device visible: 7 passed. MERGE-R7 §2 proved the four reds are the
#: file's own device requirement rather than a merge regression, by running
#: the identical arm against the untouched source worktree.
#:
#: WHY SKIP RATHER THAN LEAVE IT RED. This file is named by
#: ``scripts/run_631_flip_family.sh``, whose contract is that a CPU-only run
#: is green -- that is what makes a NEW red in it mean something. A permanent
#: red entry trains the reader to discount the runner, which is the failure
#: mode the explicit family list exists to prevent.
#:
#: WHAT THIS COSTS, stated plainly: three cases that DID pass without a device
#: now skip too. A module-level marker makes the device requirement one
#: declared fact about the file instead of four scattered ones, and the file's
#: subject -- that the quarantine constant is gone and the mechanism replaced
#: it -- is not meaningfully gated by three cases in isolation. The three are
#: recoverable by converting this to per-test markers if a shift wants them.
#:
#: THE GATE IS THEREFORE NOT DISCHARGED BY A CPU RUN. Closing it needs a
#: device-visible arm in a GPU window; a skip is an honest "not measured",
#: never a pass.
MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8-yarn1.5"

#: #910 (the second of #862's two restposten): this file's OTHER requirement,
#: and it is not the device. ``MODEL`` is the Qwen3.6 checkpoint the
#: 2026-08-13 boot ran on, and it is no longer on this box -- so on a machine
#: that DOES have a card, every ``ServerArgs(**_flip_args())`` below now dies
#: in huggingface_hub with ``HFValidationError: Repo id must be in the form
#: 'repo_name' or 'namespace/repo_name'``, which names neither the file nor
#: the missing directory. The device skip above hid it at the desk.
#:
#: NOT REPOINTED, for the reason #862 gave for
#: test_gdn_resident_cap_floor_656.py: the assertions quote that boot
#: literally. ``651498`` in
#: ``test_a_pool_above_the_old_quarantine_is_no_longer_refused`` is the pool
#: the seam-aware sizer DERIVED for this checkpoint on this rig, and the
#: ``phase_flip_tp_vector="32,16,16"`` / ``pp_stage_ratio=[14,10,8]`` geometry
#: below is that boot's. Aiming the module at a surviving Qwen3.8 build would
#: keep the numbers and change what they are numbers ABOUT, which is the one
#: failure a proof gate must not have. Name the dependency and let a box that
#: carries the checkpoint run it.
#:
#: Both markers stay, and neither subsumes the other: with a card but no
#: checkpoint this skips on the checkpoint, with the checkpoint but no card it
#: skips on the device.
pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason=(
            "needs a visible device: every case builds a real ServerArgs, whose "
            "__post_init__ resolves one. 7/7 pass with a device; without one, 4 "
            "of 7 fail in device resolution before reaching an assertion."
        ),
    ),
    pytest.mark.skipif(
        not os.path.isdir(MODEL),
        reason=(
            f"requires the #656 phase-flip acceptance checkpoint {MODEL}, which "
            "is gone from this box: the assertions quote that boot literally "
            "(derived pool 651498, tp vector 32,16,16, stage ratio 14/10/8), so "
            "the module cannot be repointed at another checkpoint without "
            "changing the specimen it pins."
        ),
    ),
]
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
