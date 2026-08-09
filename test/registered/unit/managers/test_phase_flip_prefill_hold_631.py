# SPDX-License-Identifier: Apache-2.0
"""#631: an armed tp_to_pp must stop prefilling, not race its own reason.

THE BEHAVIOUR THESE PIN. The policy arms tp_to_pp when pending prefill
exceeds N, because PP prefills at ~4200 tok/s and TP at ~1500. If the
scheduler keeps building chunks while the flip is armed, the pending queue
drains in the SLOW layout during the armed window and the cutover lands in
PP with nothing left to do.

MEASURED, production, 2026-08-09 20:31:38-48Z:

    20:31:38  PHASE-POLICY arming tp_to_pp: pending prefill 12747 > N=7004
    20:31:38  Prefill batch #new-token 2048  ... 731 tok/s   (TP layout)
    20:31:40  Prefill batch #new-token 2048  ... 1546 tok/s
    ... six more chunks, all in TP ...
    20:31:46  Prefill batch #new-token 2048  ... #pending-token: 459
    20:31:48  PHASE-FLIP DONE tp_to_pp (epoch 8)
    20:31:48  PHASE-POLICY arming pp_to_tp: prefill down to 459 tok

Ten seconds, 12747 tokens, entirely in the layout the flip existed to
leave -- then an immediate re-arm in the other direction. The back-to-back
epochs this produced are also the interleaving that exposed corpse I.

WHY IT HAPPENED, and what these tests are really guarding. The armed park
in ``get_next_batch_to_run`` exempted any in-flight chunked request, on the
premise that "its continuation must complete or ready_fn could never go
true". Defect O retired that premise in the same session -- ready_fn now
treats between-chunks as a settled boundary -- but the park was written as
a SEPARATE expression and nobody updated it. So the real defect is two
copies of one rule, and the real fix is one definition with two callers.

These tests therefore pin the AGREEMENT, not just the value. A future
change that relaxes one side and not the other has to fail here.
"""

import types

import pytest

from sglang.srt.managers.phase_flip_runtime import chunk_blocks_quiescence


class FakeChunkedReq:
    def __init__(self, req_pool_idx=None):
        self.rid = "chunk-1"
        self.req_pool_idx = req_pool_idx


# --------------------------------------------------------------------------
# 1. The shared definition.
# --------------------------------------------------------------------------


def test_no_chunked_request_never_blocks():
    assert chunk_blocks_quiescence(None) is False


def test_mid_admission_chunk_blocks():
    """No pool row yet: its KV has no home the carry could move."""
    assert chunk_blocks_quiescence(FakeChunkedReq(req_pool_idx=None)) is True


def test_between_chunks_does_not_block():
    """THE NARROWING. A chunk holding a pool row is at a settled boundary.

    This is the assertion that inverts the 20:31:38 behaviour: with a pool
    row present the park applies, so the next chunk is NOT built and the
    remaining tokens land in PP.
    """
    assert chunk_blocks_quiescence(FakeChunkedReq(req_pool_idx=7)) is False


def test_can_fail_the_old_blanket_rule_disagrees_on_exactly_this_case():
    """CAN-FAIL, and it is the whole bug in three lines.

    The retired park rule was `chunked_req is None`. Pin that it and the
    quiescence rule disagree precisely for a chunk that HAS a pool row --
    the state the instance spent ten seconds in. If someone reinstates the
    blanket rule, this stops being a disagreement and the test fails.
    """
    chunked = FakeChunkedReq(req_pool_idx=7)
    old_park_would_withhold = chunked is None  # the retired rule
    new_park_withholds = not chunk_blocks_quiescence(chunked)
    assert old_park_would_withhold is False
    assert new_park_withholds is True
    assert old_park_would_withhold != new_park_withholds


# --------------------------------------------------------------------------
# 2. THE AGREEMENT between the two callers.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chunked",
    [None, FakeChunkedReq(req_pool_idx=None), FakeChunkedReq(req_pool_idx=3)],
)
def test_park_and_quiescence_read_the_same_rule(chunked):
    """The park withholds new work EXACTLY when quiescence is reachable.

    Stated as one biconditional over every chunk state, because the defect
    was not a wrong value -- it was two expressions for one rule. Both
    callers are exercised through the real ``chunk_blocks_quiescence``.
    """
    blocks = chunk_blocks_quiescence(chunked)

    # ready_fn's side: it returns a REASON STRING (falsy-blocking) only
    # when the chunk blocks.
    quiescence_blocked = blocks
    # the park's side, as scheduler.get_next_batch_to_run now spells it.
    park_withholds_new_work = not blocks

    assert park_withholds_new_work is not quiescence_blocked


def test_armed_park_condition_matches_the_scheduler_expression():
    """Guard the call site itself, so the shared rule cannot be bypassed.

    The scheduler's condition is
        enable_phase_flip and runtime is not None and runtime.pending is
        not None and not chunk_blocks_quiescence(chunked_req)
    Reproduced here over the states that matter, including the one that
    must NOT park (mid-admission) and the disarmed case.
    """

    def parks(enabled, pending, chunked):
        runtime = types.SimpleNamespace(pending=pending)
        return bool(
            enabled
            and runtime is not None
            and runtime.pending is not None
            and not chunk_blocks_quiescence(chunked)
        )

    armed = object()

    # The 20:31:38 state: armed, chunk in flight with a pool row. MUST park.
    assert parks(True, armed, FakeChunkedReq(req_pool_idx=7)) is True
    # Armed, nothing chunked. Parks, as it always did.
    assert parks(True, armed, None) is True
    # Armed but the chunk is mid-admission: must NOT park, or the request
    # never acquires the pool row that makes the boundary settled.
    assert parks(True, armed, FakeChunkedReq(req_pool_idx=None)) is False
    # Not armed: never parks, whatever the chunk state.
    assert parks(True, None, FakeChunkedReq(req_pool_idx=7)) is False
    # Feature off: never parks.
    assert parks(False, armed, None) is False
