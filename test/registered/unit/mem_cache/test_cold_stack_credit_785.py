# SPDX-License-Identifier: Apache-2.0
"""#785 rung 4: what deferring the PHASE-COLD stack posts is worth, and its price.

MEASURED STATE, boot 735-bal785 (commit 31c655d64b, cut 32,18,14 / attn 8,4,4,
anchor 8192, no pool pin, healthy):

    pool          525462 / 584355 / 705859  -> global 525462
    stack post      3735.1 / 2540.3 / 1977.3 MiB
    of which residual 2294 / 1188 / 625 MiB  (backends + decode graphs)
    cell           16384 / 8192 / 8192 B

Rank 0 binds. The 669k reference needs 10452 MiB of KV-available on it and it
has 8210 -- short 2242 MiB against a residual of 2294 MiB on the same rank.
Those are the same number, and that identity is why rung 4's recorded refusal
is being revisited rather than a new mechanism invented.
"""

import pytest

from sglang.srt.managers.arena_tail_probe import (
    STACK_RESIDUAL_MIB,
    post_sizing_stack_bytes,
)

MIB = 1048576
CELL = (16384, 8192, 8192)
BUDGETED_AVAILABLE_MIB = (8210.0, 4565.0, 5514.0)  # after the rung-3 post
REFERENCE_TOKENS = 669000
MEASURED_POOL = 525462

# rank 0 on cut 32,18,14: layout_pp exceeds layout_tp, so the arena adds nothing
RANK0 = dict(layout_pp_bytes=int(16362.72 * MIB), layout_tp_bytes=int(15925.80 * MIB))
RANK0_DRAFT = int(1441.14 * MIB)


def _tokens(available_mib, rank):
    return int(available_mib * MIB / CELL[rank])


def test_the_shortfall_and_the_residual_are_the_same_number():
    """THE WHOLE CASE FOR RUNG 4, as arithmetic rather than intuition."""
    needed_mib = REFERENCE_TOKENS * CELL[0] / MIB
    shortfall = needed_mib - BUDGETED_AVAILABLE_MIB[0]
    assert 2200 < shortfall < 2300
    assert abs(shortfall - STACK_RESIDUAL_MIB[0]) < 60


def test_todays_post_reproduces_the_boot_that_measured_it():
    charged = post_sizing_stack_bytes(0, draft_bytes=RANK0_DRAFT, **RANK0)
    assert abs(charged / MIB - 3735.1) < 1.0
    assert _tokens(BUDGETED_AVAILABLE_MIB[0], 0) == pytest.approx(
        MEASURED_POOL, rel=1e-3
    )


def test_deferring_the_cold_posts_clears_the_reference_on_the_binding_rank():
    charged = post_sizing_stack_bytes(
        0, draft_bytes=RANK0_DRAFT, cold_stack_deferred=True, **RANK0
    )
    credited = post_sizing_stack_bytes(0, draft_bytes=RANK0_DRAFT, **RANK0) - charged
    assert credited == STACK_RESIDUAL_MIB[0] * MIB
    assert _tokens(BUDGETED_AVAILABLE_MIB[0] + credited / MIB, 0) >= REFERENCE_TOKENS


def test_the_draft_weights_are_never_credited_only_the_cold_posts():
    """Weights stay resident: the draft runs in TP and its weights are needed
    there. Crediting them would be the #678 OOM again, one rung deeper."""
    charged = post_sizing_stack_bytes(
        0, draft_bytes=RANK0_DRAFT, cold_stack_deferred=True, **RANK0
    )
    assert charged >= RANK0_DRAFT


def test_the_credit_is_bounded_by_what_was_measured_not_by_what_is_needed():
    """CAN-FAIL GUARD against sizing to the target instead of to the rig.

    The credit must be the measured residual even when that is not enough --
    a term that grows to close whatever gap it is shown is not a measurement.
    """
    for rank in (1, 2):
        pp = int(9000 * MIB)
        tp = int(8573.78 * MIB)
        full = post_sizing_stack_bytes(rank, pp, tp, draft_bytes=int(1352.35 * MIB))
        deferred = post_sizing_stack_bytes(
            rank, pp, tp, draft_bytes=int(1352.35 * MIB), cold_stack_deferred=True
        )
        assert full - deferred == STACK_RESIDUAL_MIB[rank] * MIB


def test_an_unknown_rank_is_charged_no_residual_rather_than_a_guess():
    pp, tp = int(9000 * MIB), int(8573.78 * MIB)
    assert post_sizing_stack_bytes(7, pp, tp, draft_bytes=0) == 0
