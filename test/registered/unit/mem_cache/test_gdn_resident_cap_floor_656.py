# SPDX-License-Identifier: Apache-2.0
"""#656: `--gdn-resident-state-slots` must clear the mamba hard demand floor.

WHAT THIS PINS, and why it is not a duplicate of test_mamba_pool_floor.py.
That file pins the floor arithmetic and its application to
`--max-mamba-cache-size`. This one pins the gap that arithmetic fell through:
`--gdn-resident-state-slots` boot-sizes the state pool to its OWN value,
overriding the validated `--max-mamba-cache-size`, and nothing re-checked the
number that actually wins.

THE SPECIMEN is the #656 formal acceptance boot of 2026-08-13:
`--max-mamba-cache-size 12 --max-running-requests 4
--gdn-resident-state-slots 4`. 12 clears the floor of 12 exactly, so the
sibling validator passed it; the cap then sized the pool to 4, which is the
floor for ONE running request. The instance served for about four minutes of
mixed load and then died with SIGQUIT out of `alloc_req_slots`
(`mamba_available=0, mamba_schedulable=0`) when a deep prefill held slots
across an admission. A parse-time refusal is what that boot was owed.
"""

import os

import pytest

from sglang.srt.mem_cache.mamba_pool_floor import mamba_hard_floor
from sglang.srt.server_args import ServerArgs

MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8-yarn1.5"

# #862: this module was dark hermetically -- the mem_cache conftest turned
# get_device() into a skip, so nobody saw that its checkpoint dependency had
# rotted. With the conftest handing out "cpu" instead, ServerArgs() reaches
# huggingface_hub and the absent directory surfaces as an opaque
# `HFValidationError: Repo id must be in the form ...`.
#
# The dependency is REAL and is not repointed here. The assertions pin the
# 2026-08-13 acceptance specimen literally ("--gdn-resident-state-slots to at
# least 12"); 12 is that checkpoint's mamba floor, so aiming the module at a
# surviving Qwen3.8 build would silently change the specimen the test claims to
# pin. Name the dependency instead, and let a box that carries the checkpoint
# run it. The same retired path is still referenced by
# test/srt/test_phase_flip_serving_proof_gate.py,
# test/registered/unit/planner/test_pp_family_cut_485.py and
# scripts/route_a_631_unmanned_acceptance.sh -- one drift, four sites.
pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL),
    reason=(
        f"requires the #656 acceptance checkpoint {MODEL}, which is not on this "
        "box: the assertions quote that boot's floor of 12 literally, so the "
        "module cannot be repointed at another checkpoint without changing the "
        "specimen it pins."
    ),
)


def _args(**extra):
    return dict(
        model_path=MODEL,
        trust_remote_code=True,
        disable_overlap_schedule=True,
        **extra,
    )


def _floor(bs):
    probe = ServerArgs(**_args(max_running_requests=bs, max_mamba_cache_size=64))
    return mamba_hard_floor(probe, bs)


def test_the_acceptance_specimen_is_refused():
    """The exact 2026-08-13 configuration must not boot."""
    with pytest.raises(ValueError) as exc:
        ServerArgs(
            **_args(
                max_running_requests=4,
                max_mamba_cache_size=12,
                gdn_resident_state_slots=4,
            )
        )
    msg = str(exc.value)
    # The message must name the flag that actually wins, the floor it missed,
    # and BOTH ways out -- an error that names only the symptom sends the
    # reader back to the same guess the crash already cost.
    assert "--gdn-resident-state-slots 4" in msg
    assert f"floor of {_floor(4)} slots" in msg
    assert "--max-running-requests to at most 1" in msg
    assert "--gdn-resident-state-slots to at least 12" in msg


def test_cap_at_the_floor_is_accepted():
    """At the floor the configuration is serviceable and must be left alone."""
    sa = ServerArgs(
        **_args(
            max_running_requests=4,
            max_mamba_cache_size=12,
            gdn_resident_state_slots=_floor(4),
        )
    )
    assert sa.gdn_resident_state_slots == _floor(4)


def test_cap_above_the_floor_is_accepted():
    """Everything above the floor is cache; sizing it is the operator's call."""
    sa = ServerArgs(
        **_args(
            max_running_requests=4,
            max_mamba_cache_size=32,
            gdn_resident_state_slots=24,
        )
    )
    assert sa.gdn_resident_state_slots == 24


def test_the_same_cap_is_fine_at_the_concurrency_it_can_serve():
    """4 slots is not wrong in itself -- it is wrong for FOUR requests.

    This is the direction check: the refusal must follow the running set, not
    the cap alone, or it would forbid the bs=1 MAX-KV configuration the cap
    exists for.
    """
    sa = ServerArgs(
        **_args(
            max_running_requests=1,
            max_mamba_cache_size=12,
            gdn_resident_state_slots=4,
        )
    )
    assert sa.gdn_resident_state_slots == 4
    assert sa.max_running_requests == 1


def test_unset_flag_leaves_the_baseline_path_untouched():
    """With the flag unset nothing new can fire -- the default path is byte-identical."""
    sa = ServerArgs(**_args(max_running_requests=4, max_mamba_cache_size=12))
    assert sa.gdn_resident_state_slots is None


def test_unpinned_concurrency_is_not_second_guessed():
    """`max_running_requests=None` means the auto path floors itself (sibling guard)."""
    sa = ServerArgs(**_args(gdn_resident_state_slots=4))
    assert sa.gdn_resident_state_slots == 4


def test_gate_can_fail_the_can_fail_proof():
    """A gate nobody has seen refuse is not evidence.

    Drive the predicate one slot BELOW the floor and one slot AT it, and
    require the two to disagree. If the validator were removed, this asserts
    a raise that never comes and turns red -- which is the property that
    makes the four tests above worth reading.
    """
    floor = _floor(4)
    with pytest.raises(ValueError):
        ServerArgs(
            **_args(
                max_running_requests=4,
                max_mamba_cache_size=floor,
                gdn_resident_state_slots=floor - 1,
            )
        )
    ok = ServerArgs(
        **_args(
            max_running_requests=4,
            max_mamba_cache_size=floor,
            gdn_resident_state_slots=floor,
        )
    )
    assert ok.gdn_resident_state_slots == floor
