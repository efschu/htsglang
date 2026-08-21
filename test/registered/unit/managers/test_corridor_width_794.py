# SPDX-License-Identifier: Apache-2.0
"""#794: the corridor guard stops advising and starts ACTUATING.

WHAT THESE PIN, AND WHY THE FIRST ONE IS A SIMULATION.

On 2026-08-21 the prefill admission guard measured a corridor shortfall of
981 MiB for rank 0, logged that "the corridor cannot be restored ahead of this
chunk", and admitted the chunk. It died in ``in_proj_qkvz`` asking for 256.00
MiB with 131.69 MiB free. Every number in that sequence was correct. The
defect was that none of them changed what ran.

A test that asserted "the gate reports a shortfall" would therefore have
PASSED on the crashing tree -- the gate did report it. So the first test here
MODELS THE CRASH instead: the same chunk width, the same price, the same free
column, run once with the actuator absent (the pre-#794 tree) and once with it.
The run without the actuator must OOM for this file to be worth anything.

The rest pin the four decisions in the actuator that are easy to "simplify"
back into defects: that an unpriced chunk is never cut, that a fitting chunk is
never cut, that the cut saturates at a liveness floor instead of reaching zero,
and that a stepped price is inverted exactly rather than approximated.
"""

import math

from sglang.srt.managers.corridor_width import (
    MIN_CHUNK_TOKENS,
    fundable_chunk_tokens,
    width_was_cut,
)

MIB = 1024 * 1024

# The rig's measured figures, from boot_735_bal785.log and the corrected
# estimator (server_args.py gdn_prefill_scratch_mib, commit 56e3ccd9f4):
# 1270 MiB of transient for an 8192-token GDN prefill chunk on rank 0.
CRASH_TOKENS = 8192
CRASH_TRANSIENT_MIB = 1270.0
BYTES_PER_TOKEN = CRASH_TRANSIENT_MIB * MIB / CRASH_TOKENS


def linear_price(tokens: int) -> float:
    """The transient of a ``tokens``-wide chunk, linear as the formula is."""
    return tokens * BYTES_PER_TOKEN


# --------------------------------------------------------------------------
# 1. THE CRASH, MODELLED. Without the actuator this OOMs; with it, it fits.
# --------------------------------------------------------------------------


def test_the_crash_chunk_is_narrowed_until_it_fits_the_free_column():
    # The card at the moment the guard gave up: the ladder is exhausted and
    # what remains spendable before the arming floor is 300 MiB.
    budget = 300 * MIB

    # PRE-#794: the width is whatever was configured, because the verdict was
    # evidence and not a decision. This is the run that must fail.
    uncut = CRASH_TOKENS
    assert linear_price(uncut) > budget, (
        "the simulation is worthless unless the pre-fix width really does "
        "exceed what the card can fund"
    )

    granted = fundable_chunk_tokens(
        requested_tokens=CRASH_TOKENS,
        budget_bytes=budget,
        price_bytes=linear_price,
        granularity_tokens=256,
    )

    assert width_was_cut(CRASH_TOKENS, granted), "the actuator must actuate"
    assert (
        linear_price(granted) <= budget
    ), "the granted width must fit the budget that the full width did not"
    # It must also stay useful: cutting to the floor when 1900 tokens fit
    # would trade an OOM for a stall.
    assert granted >= 1536, f"cut far more than the budget required: {granted}"


def test_the_transient_scales_linearly_so_a_prefix_always_exists():
    """The property the whole actuator rests on, pinned as arithmetic.

    Every term of the GDN prefill estimate is a positive multiple of T or of
    ceil(T / 64), so halving the width at least halves the transient. If this
    were ever false -- a term with a floor, a fixed workspace -- narrowing
    would stop being a remedy and this module would have to refuse instead.
    """
    for tokens in (512, 1024, 2048, 4096, 8192):
        assert linear_price(tokens // 2) <= linear_price(tokens) / 2 + 1e-9


# --------------------------------------------------------------------------
# 2. NOT PRICED MEANS NOT CUT.
# --------------------------------------------------------------------------


def test_an_unpriced_chunk_is_admitted_at_full_width():
    granted = fundable_chunk_tokens(
        requested_tokens=CRASH_TOKENS,
        budget_bytes=0,
        price_bytes=None,
        granularity_tokens=256,
    )
    assert granted == CRASH_TOKENS
    assert not width_was_cut(CRASH_TOKENS, granted)


def test_an_estimator_that_returns_none_does_not_cut():
    granted = fundable_chunk_tokens(
        requested_tokens=CRASH_TOKENS,
        budget_bytes=0,
        price_bytes=lambda _tokens: None,
    )
    assert granted == CRASH_TOKENS


def test_a_raising_estimator_does_not_take_prefill_down():
    def explode(_tokens: int) -> float:
        raise RuntimeError("no config here")

    granted = fundable_chunk_tokens(
        requested_tokens=CRASH_TOKENS,
        budget_bytes=0,
        price_bytes=explode,
    )
    assert granted == CRASH_TOKENS


def test_an_estimator_that_answers_only_the_full_width_does_not_cut():
    """A half-answer must not become a cut.

    If the price is readable at the requested width but not at a narrower one,
    the bisection has no ground to stand on; cutting anyway would be a width
    chosen by the failure mode rather than by the card.
    """

    def only_full(tokens: int):
        return linear_price(tokens) if tokens == CRASH_TOKENS else None

    granted = fundable_chunk_tokens(
        requested_tokens=CRASH_TOKENS,
        budget_bytes=1,
        price_bytes=only_full,
    )
    assert granted == CRASH_TOKENS


# --------------------------------------------------------------------------
# 3. A FITTING CHUNK IS NEVER CUT.
# --------------------------------------------------------------------------


def test_a_chunk_the_corridor_funds_is_not_narrowed():
    budget = 4096 * MIB
    granted = fundable_chunk_tokens(
        requested_tokens=CRASH_TOKENS,
        budget_bytes=budget,
        price_bytes=linear_price,
        granularity_tokens=256,
    )
    assert granted == CRASH_TOKENS
    assert not width_was_cut(CRASH_TOKENS, granted)


def test_the_common_path_prices_once():
    calls = []

    def counting(tokens: int) -> float:
        calls.append(tokens)
        return linear_price(tokens)

    fundable_chunk_tokens(
        requested_tokens=CRASH_TOKENS,
        budget_bytes=4096 * MIB,
        price_bytes=counting,
    )
    assert calls == [CRASH_TOKENS], (
        "a chunk that fits must cost exactly one price evaluation; this runs "
        "on every prefill admission"
    )


# --------------------------------------------------------------------------
# 4. THE CUT SATURATES. A width of zero is a deadlock, not a safety measure.
# --------------------------------------------------------------------------


def test_a_hopeless_budget_saturates_at_the_liveness_floor():
    granted = fundable_chunk_tokens(
        requested_tokens=CRASH_TOKENS,
        budget_bytes=0,
        price_bytes=linear_price,
        granularity_tokens=256,
    )
    assert granted == MIN_CHUNK_TOKENS
    assert granted > 0, "a zero width admits nothing and never recovers"


def test_a_negative_budget_still_admits_the_floor():
    granted = fundable_chunk_tokens(
        requested_tokens=CRASH_TOKENS,
        budget_bytes=-(512 * MIB),
        price_bytes=linear_price,
    )
    assert granted == MIN_CHUNK_TOKENS


def test_a_request_already_at_the_floor_is_left_alone():
    granted = fundable_chunk_tokens(
        requested_tokens=MIN_CHUNK_TOKENS,
        budget_bytes=0,
        price_bytes=linear_price,
    )
    assert granted == MIN_CHUNK_TOKENS


def test_zero_and_negative_widths_pass_through_untouched():
    assert (
        fundable_chunk_tokens(
            requested_tokens=0, budget_bytes=0, price_bytes=linear_price
        )
        == 0
    )


# --------------------------------------------------------------------------
# 5. THE INVERSION IS EXACT, INCLUDING THE STEP.
# --------------------------------------------------------------------------


def stepped_price(tokens: int) -> float:
    """The GDN shape: linear terms plus one ``ceil(T / 64)`` workspace term."""
    return tokens * BYTES_PER_TOKEN + math.ceil(tokens / 64) * (4 * MIB)


def test_a_stepped_price_is_inverted_exactly_not_approximated():
    budget = 500 * MIB
    granted = fundable_chunk_tokens(
        requested_tokens=CRASH_TOKENS,
        budget_bytes=budget,
        price_bytes=stepped_price,
    )
    assert stepped_price(granted) <= budget, "the granted width must fit"
    assert stepped_price(granted + 1) > budget, (
        "and it must be the WIDEST that fits -- a linear division would have "
        "left tokens on the table on a stepped price"
    )


def test_the_granted_width_is_aligned_down_never_up():
    budget = 500 * MIB
    granted = fundable_chunk_tokens(
        requested_tokens=CRASH_TOKENS,
        budget_bytes=budget,
        price_bytes=linear_price,
        granularity_tokens=256,
    )
    assert granted % 256 == 0
    assert linear_price(granted) <= budget


def test_alignment_never_pushes_the_width_below_the_floor():
    """A coarse granularity must not align a fundable width down to zero."""
    granted = fundable_chunk_tokens(
        requested_tokens=CRASH_TOKENS,
        budget_bytes=20 * MIB,
        price_bytes=linear_price,
        granularity_tokens=4096,
    )
    assert granted >= MIN_CHUNK_TOKENS


def test_the_width_is_monotone_in_the_budget():
    """More headroom never grants less width -- an actuator that flapped here
    would oscillate the chunk size pass to pass."""
    widths = [
        fundable_chunk_tokens(
            requested_tokens=CRASH_TOKENS,
            budget_bytes=mib * MIB,
            price_bytes=linear_price,
        )
        for mib in (0, 100, 200, 400, 800, 1600)
    ]
    assert widths == sorted(widths)
    assert widths[-1] == CRASH_TOKENS


# --------------------------------------------------------------------------
# 6. THE GATE'S ACTUATOR. The arithmetic above is only useful if the gate
#    feeds it a real budget and a real price, and if the scheduler applies
#    the answer -- the two joints where a measuring mechanism usually stops.
# --------------------------------------------------------------------------

import types

from sglang.srt.managers.corridor_admission import PrefillAdmissionGate


class FakeGuard:
    def __init__(self, free_mib: float, delta_mib: int = 256):
        self._free = free_mib * MIB
        self.delta_mib = delta_mib
        self.floor_mib = 1331
        self.law_floor_mib = 1024
        self.device_index = 0
        self.providers = ()

    def free_bytes(self) -> int:
        return int(self._free)


class FakeServerArgs:
    """A config that prices the GDN transient linearly, as the rig's does."""

    enable_phase_flip = True
    tp_size = 1

    def gdn_prefill_scratch_mib(self, head_share, tokens=None):
        width = CRASH_TOKENS if tokens is None else int(tokens)
        return width * BYTES_PER_TOKEN / MIB


def build_actuator(free_mib: float, cache_mib: float = 0.0):
    scheduler = types.SimpleNamespace(server_args=FakeServerArgs())
    gate = PrefillAdmissionGate(scheduler)
    guard = FakeGuard(free_mib)
    gate._guard = lambda: guard  # noqa: SLF001
    gate._allocator_cache_bytes = lambda: int(cache_mib * MIB)  # noqa: SLF001
    return gate


def test_the_gate_narrows_a_chunk_a_full_card_cannot_fund():
    # The 18:01 specimen: the card is full, the ladder returned nothing.
    gate = build_actuator(free_mib=6.0)
    granted = gate.granted_width(4096)
    assert granted < 4096, "a card with 6 MiB free must not run a 4096 chunk"
    assert granted == MIN_CHUNK_TOKENS
    assert gate.cuts == 1
    assert gate.tokens_withheld == 4096 - granted


def test_the_gate_leaves_a_funded_chunk_alone():
    gate = build_actuator(free_mib=4096.0)
    assert gate.granted_width(4096) == 4096
    assert gate.cuts == 0


def test_the_allocator_cache_counts_towards_the_budget():
    """Bytes torch already holds are bytes the forward can take.

    Without this the actuator would narrow every chunk on a rig whose NVML
    free column sits near 350 MiB while the transient is served out of the
    allocator's own reservation -- a stall built out of a correct measurement
    of the wrong column.
    """
    lean = build_actuator(free_mib=700.0, cache_mib=0.0)
    fat = build_actuator(free_mib=700.0, cache_mib=2048.0)
    assert fat.granted_width(4096) > lean.granted_width(4096)
    assert fat.granted_width(4096) == 4096


def test_the_margin_is_the_guards_own_delta_not_a_new_constant():
    gate = build_actuator(free_mib=1000.0)
    guard = gate._guard()  # noqa: SLF001
    spendable = gate.spendable_bytes(guard)
    assert spendable == (1000 - guard.delta_mib) * MIB


def test_a_config_without_gdn_layers_is_never_narrowed():
    class NoGdn(FakeServerArgs):
        def gdn_prefill_scratch_mib(self, head_share, tokens=None):
            return None

    scheduler = types.SimpleNamespace(server_args=NoGdn())
    gate = PrefillAdmissionGate(scheduler)
    gate._guard = lambda: FakeGuard(1.0)  # noqa: SLF001
    gate._allocator_cache_bytes = lambda: 0  # noqa: SLF001
    assert gate.granted_width(4096) == 4096
    assert gate.cuts == 0


def test_a_measured_probe_only_ever_lowers_the_price():
    """Calibration may correct the bound downward, never upward.

    A measurement above the config bound means the bound is not one; letting
    it widen the chunk would be a probe artefact deciding the width.
    """
    gate = build_actuator(free_mib=1000.0)
    gate._measured_calibration = lambda tokens: 1.0  # noqa: SLF001
    uncalibrated = gate.granted_width(4096)
    cheap = build_actuator(free_mib=1000.0)
    cheap._measured_calibration = lambda tokens: 0.25  # noqa: SLF001
    assert cheap.granted_width(4096) >= uncalibrated


# --------------------------------------------------------------------------
# 7. WHO IS ENTITLED TO NARROW. A downstream PP rank that narrows produces a
#    batch that does not match PP0's forwarded schedule, which is a refused
#    pass and then a voided one -- a livelock wearing a safety jacket.
# --------------------------------------------------------------------------

from sglang.srt.managers.scheduler import Scheduler

WIDTH = Scheduler._corridor_granted_prefill_width


class CuttingGate:
    def __init__(self):
        self.calls = 0

    def granted_width(self, requested):
        self.calls += 1
        return MIN_CHUNK_TOKENS


def scheduler_stub(pp_size, pp_rank, tp_world=1):
    return types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_size=pp_size, pp_rank=pp_rank),
        tp_cpu_group=None if tp_world <= 1 else object(),
    )


def test_pp0_narrows_because_the_ring_carries_its_decision(monkeypatch):
    gate = CuttingGate()
    monkeypatch.setattr(
        "sglang.srt.managers.scheduler.get_prefill_admission_gate",
        lambda _s: gate,
    )
    assert WIDTH(scheduler_stub(pp_size=3, pp_rank=0), 4096) == MIN_CHUNK_TOKENS
    assert gate.calls == 1


def test_a_downstream_pp_rank_never_narrows(monkeypatch):
    gate = CuttingGate()
    monkeypatch.setattr(
        "sglang.srt.managers.scheduler.get_prefill_admission_gate",
        lambda _s: gate,
    )
    for rank in (1, 2):
        assert WIDTH(scheduler_stub(pp_size=3, pp_rank=rank), 4096) == 4096
    assert gate.calls == 0, (
        "a downstream rank must not even price the cut: its narrower batch "
        "would be refused against PP0's forwarded schedule"
    )


def test_a_broken_actuator_leaves_the_width_alone(monkeypatch):
    class Exploding:
        def granted_width(self, requested):
            raise RuntimeError("no guard")

    monkeypatch.setattr(
        "sglang.srt.managers.scheduler.get_prefill_admission_gate",
        lambda _s: Exploding(),
    )
    assert WIDTH(scheduler_stub(pp_size=3, pp_rank=0), 4096) == 4096


def test_a_widened_answer_is_refused(monkeypatch):
    """The actuator may only ever narrow. A width above the requested one
    would be an unfunded promise dressed as a measurement."""

    class Widening:
        def granted_width(self, requested):
            return requested * 2

    monkeypatch.setattr(
        "sglang.srt.managers.scheduler.get_prefill_admission_gate",
        lambda _s: Widening(),
    )
    assert WIDTH(scheduler_stub(pp_size=3, pp_rank=0), 4096) == 4096
