"""#701: the chunked-prefill self-deadlock falsifier.

Specimen: ONE chunked request (new-seq 1, new-token 512, cached 0) drove pool
usage 0.95 -> 1.00 with no retract, abort or finish. Root cause
(DESIGN_701_chunked_admission.md): chunked prefill bounds the COMPUTE per step,
not the KV COMMITMENT, and `schedule_policy.py:1389-1407` charged the budget
`trunc_len` -- one 512-token chunk -- while admitting a request whose real
commitment is its entire remaining length. A 327,680-token request was admitted
on a 512-token affordability check, a 640x under-charge, and its own locked
prefix then made the pool unfreeable.

The falsifier the ticket asked for is `test_required_exceeding_fundable_is_never_admitted`.

Per the binding generality clause the rule is pool ARITHMETIC, never a rig
threshold: `test_no_rig_threshold_only_arithmetic` pins that a request at 99 %
of a large pool is admissible while one at 101 % of a small pool is refused,
under the same function with no constant between them.

Hermetic: pure arithmetic, no CUDA, no scheduler import.
"""

import pytest

from sglang.srt.planner.chunked_admission import (
    PoolState,
    decide_chunked_admission,
    effective_running_bs,
)


def _pool(free, unlocked, locked=0.0, capacity=437_000.0, spillable=0.0):
    return PoolState(
        free_tokens=free,
        evictable_unlocked_tokens=unlocked,
        locked_tokens=locked,
        total_capacity_tokens=capacity,
        spillable_tokens=spillable,
    )


def test_required_exceeding_fundable_is_never_admitted():
    """THE FALSIFIER. Must not admit into the deadlock."""
    pool = _pool(free=20_000.0, unlocked=5_000.0)
    d = decide_chunked_admission(remaining_tokens=100_000, pool=pool)
    assert d.verdict in ("defer", "refuse")
    assert not d.admitted


def test_the_specimen_is_not_admitted():
    """327,680 tokens against a pool with 437k capacity but ~20k free."""
    pool = _pool(free=20_000.0, unlocked=1_000.0, locked=416_000.0)
    d = decide_chunked_admission(remaining_tokens=327_680, pool=pool)
    assert not d.admitted
    assert d.required_tokens == 327_680


def test_verdict_does_not_depend_on_chunk_size():
    """The precise substitution the bug makes.

    The old code decided on `trunc_len`. The rule must be a function of the
    REMAINING LENGTH and the pool only -- there is no chunk parameter to pass,
    and that is the point.
    """
    pool = _pool(free=20_000.0, unlocked=1_000.0)
    a = decide_chunked_admission(remaining_tokens=327_680, pool=pool)
    # A caller who "affords" a 512-token chunk gets the same answer, because the
    # chunk never enters the arithmetic.
    b = decide_chunked_admission(remaining_tokens=327_680, pool=pool)
    assert a.verdict == b.verdict == "defer"
    with pytest.raises(TypeError):
        decide_chunked_admission(remaining_tokens=327_680, pool=pool, chunk_tokens=512)  # noqa


def test_larger_than_total_capacity_is_refused_loudly_not_deferred():
    """Refuse and defer are different verdicts: one can never succeed."""
    pool = _pool(free=400_000.0, unlocked=30_000.0, capacity=437_000.0)
    d = decide_chunked_admission(remaining_tokens=500_000, pool=pool)
    assert d.verdict == "refuse"
    assert "capacity" in d.reason.lower()


def test_a_request_that_fits_is_admitted():
    pool = _pool(free=200_000.0, unlocked=150_000.0)
    d = decide_chunked_admission(remaining_tokens=327_680, pool=pool)
    assert d.verdict == "admit"
    assert d.admitted


def test_locked_tokens_are_not_fundable():
    """The whole deadlock in one assertion: a locked chain cannot pay for growth."""
    generous_but_locked = _pool(free=1_000.0, unlocked=0.0, locked=430_000.0)
    d = decide_chunked_admission(remaining_tokens=100_000, pool=generous_but_locked)
    assert not d.admitted
    assert d.fundable_tokens == 1_000.0


def test_no_rig_threshold_only_arithmetic():
    """GENERALITY: same rule, opposite verdicts, no constant between them."""
    big = _pool(free=990_000.0, unlocked=0.0, capacity=1_000_000.0)
    assert decide_chunked_admission(990_000, big).verdict == "admit"
    small = _pool(free=100.0, unlocked=0.0, capacity=1_000.0)
    assert decide_chunked_admission(1_010, small).verdict == "refuse"


def test_a_spill_capability_raises_fundable_without_changing_the_rule():
    """Future-proofing (a): spill plugs in, the rule is not re-derived."""
    without = _pool(free=20_000.0, unlocked=1_000.0)
    with_spill = _pool(free=20_000.0, unlocked=1_000.0, spillable=400_000.0)
    assert decide_chunked_admission(327_680, without).verdict == "defer"
    assert decide_chunked_admission(327_680, with_spill).verdict == "admit"


def test_effective_running_bs_counts_resident_chunked_requests():
    """#631 defect O as a counting truth: resident-but-batchless is RUNNING."""
    assert effective_running_bs(running_bs=0, resident_chunked=1) == 1
    assert effective_running_bs(running_bs=3, resident_chunked=2) == 5
    # And the idle conclusion every consumer draws must now be false.
    assert effective_running_bs(running_bs=0, resident_chunked=1) > 0
