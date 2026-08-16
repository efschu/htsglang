"""#701: the chunked-prefill admission rule (arithmetic half).

CORRECTED after the review gate. The original version of this file carried a
mechanism story that the gate refuted and that I verified false in-code:

* it claimed a request was "admitted on a 512-token affordability check", a
  640x under-charge. In fact ``schedule_policy.py:1464`` already gates the FULL
  lifetime (``total_tokens >= self.rem_total_tokens -> NO_TOKEN``). There is no
  missing full-length gate at first admission.
* it cited ``:1389-1407`` as the defect site. That is the ``ignore_eos`` branch,
  reachable only with ``ignore_eos AND tree_cache.disable``. The serving line
  runs radix ON, so the specimen went through the MAIN chunked branch at
  ``:1569-1610``.

The two real holes, and where each is tested:

1. **Cross-pass commitment invisibility** -- a resident chunked request's
   remaining PREFILL is represented in no later pass. Tested against the REAL
   ``PrefillAdder`` in
   ``test/registered/unit/managers/test_chunked_commitment_701.py``, because
   arithmetic on this module alone cannot fail against scheduler behaviour.
2. **Paper-evictable overcount on hybrid-SSM** -- ``rem_total_tokens`` counts
   ``full_evictable_size()`` while the allocator recovers only mamba-recoverable
   bytes (``:734-737``). That is the likely specimen mechanism, and it is why
   ``PoolState`` now takes ``recoverable_evictable_tokens`` and refuses the old
   field name outright.

This file keeps the threshold-free arithmetic half.

Hermetic: pure arithmetic, no CUDA, no scheduler import.
"""

import pytest
from sglang.srt.planner.chunked_admission import (
    ChunkedCommitmentLedger,
    PoolState,
    decide_chunked_admission,
    effective_running_bs,
)


def _pool(free, recoverable, locked=0.0, capacity=437_000.0, permanent=0.0, spill=0.0):
    return PoolState(
        free_tokens=free,
        recoverable_evictable_tokens=recoverable,
        locked_tokens=locked,
        total_capacity_tokens=capacity,
        permanent_reserve_tokens=permanent,
        spillable_tokens=spill,
    )


def test_required_exceeding_fundable_is_never_admitted():
    """THE FALSIFIER. Must not admit into the deadlock."""
    d = decide_chunked_admission(100_000, _pool(free=20_000.0, recoverable=5_000.0))
    assert d.verdict in ("defer", "refuse")
    assert not d.admitted


def test_the_specimen_is_not_admitted():
    d = decide_chunked_admission(
        327_680, _pool(free=20_000.0, recoverable=1_000.0, locked=416_000.0)
    )
    assert not d.admitted
    assert d.required_tokens == 327_680


def test_the_chunk_is_unrepresentable():
    """The substitution the original code made cannot even be expressed."""
    pool = _pool(free=20_000.0, recoverable=1_000.0)
    assert decide_chunked_admission(327_680, pool).verdict == "defer"
    with pytest.raises(TypeError):
        decide_chunked_admission(327_680, pool, chunk_tokens=512)


def test_above_the_achievable_ceiling_is_refused_loudly():
    """Refuse and defer differ: one can never succeed at any future time."""
    d = decide_chunked_admission(500_000, _pool(free=400_000.0, recoverable=30_000.0))
    assert d.verdict == "refuse"
    assert "achievable" in d.reason.lower()


def test_a_request_that_fits_is_admitted():
    d = decide_chunked_admission(327_680, _pool(free=200_000.0, recoverable=150_000.0))
    assert d.verdict == "admit"


def test_locked_tokens_are_not_fundable():
    """The deadlock in one assertion: a locked chain cannot pay for growth."""
    d = decide_chunked_admission(
        100_000, _pool(free=1_000.0, recoverable=0.0, locked=430_000.0)
    )
    assert not d.admitted
    assert d.fundable_tokens == 1_000.0


def test_paper_evictable_is_not_representable_as_fundable():
    """Defect 2: the old field name funded the specimen through the new rule."""
    with pytest.raises(TypeError):
        PoolState(
            free_tokens=1.0,
            evictable_unlocked_tokens=2.0,
            locked_tokens=0.0,
            total_capacity_tokens=3.0,
        )


def test_no_rig_threshold_only_arithmetic():
    """GENERALITY: same rule, opposite verdicts, no constant between them."""
    big = _pool(free=990_000.0, recoverable=0.0, capacity=1_000_000.0)
    assert decide_chunked_admission(990_000, big).verdict == "admit"
    small = _pool(free=100.0, recoverable=0.0, capacity=1_000.0)
    assert decide_chunked_admission(1_010, small).verdict == "refuse"


def test_permanent_reserves_lower_the_refusal_bound():
    """Defect 3: refusing at RAW capacity leaves a forever-defer band."""
    pool = _pool(free=10_000.0, recoverable=0.0, capacity=437_000.0, permanent=50_000.0)
    assert decide_chunked_admission(400_000, pool).verdict == "refuse"
    assert decide_chunked_admission(300_000, pool).verdict == "defer"


def test_a_spill_capability_raises_fundable_without_changing_the_rule():
    without = _pool(free=20_000.0, recoverable=1_000.0)
    with_spill = _pool(free=20_000.0, recoverable=1_000.0, spill=400_000.0)
    assert decide_chunked_admission(327_680, without).verdict == "defer"
    assert decide_chunked_admission(327_680, with_spill).verdict == "admit"


def test_effective_running_bs_counts_resident_chunked_requests():
    """#631 defect O: resident-but-batchless is RUNNING."""
    assert effective_running_bs(running_bs=0, resident_chunked=1) == 1
    assert effective_running_bs(running_bs=3, resident_chunked=2) == 5


# #699 ride-along: separate a retry loop from real progress.
#
# forward_ct counts batch ATTEMPTS (scheduler.py:6933, `+= 1` at the top of
# run_batch), so a batch that re-runs without committing advances it while
# nothing progresses. The liveness detector cannot tell that from health
# without a COMMIT counter, and spend() is already the single commit path.


def test_committed_chunks_counts_only_actual_commits():
    led = ChunkedCommitmentLedger()
    led.commit("a", 1000)
    assert led.committed_chunks == 0, "admission is not a commit"
    led.spend("a", 512)
    led.spend("a", 488)
    assert led.committed_chunks == 2


def test_committed_chunks_is_not_rewound_by_release():
    """A progress counter that goes backwards reads as a restart to a watcher."""
    led = ChunkedCommitmentLedger()
    led.commit("a", 100)
    led.spend("a", 100)
    led.release("a")
    assert led.committed_chunks == 1


def test_a_refused_spend_does_not_count_as_a_commit():
    led = ChunkedCommitmentLedger()
    led.commit("a", 100)
    with pytest.raises(ValueError):
        led.spend("a", 200)
    assert led.committed_chunks == 0
