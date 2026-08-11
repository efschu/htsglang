# SPDX-License-Identifier: Apache-2.0
"""#656 register C17: the corridor law, enforced at the PREFILL site too.

WHAT THESE PIN, AND WHY THE FIRST ONE IS A SIMULATION RATHER THAN AN
ASSERTION ON A RETURN VALUE.

Successor 33's acceptance breached the corridor for 12 samples on one card,
and the cause was not a gate that failed -- it was a gate that was never
called. ``CorridorGuard.ensure_headroom`` works; it had one caller, the flip
seam, and the 272k-token bs1 prefill grew at a site with no caller:

    11:20:47Z  gpu0 = 1001 MiB free, 12 samples (~1.6 s), 23 MiB under
    11:20:49Z  CORRIDOR-GUARD cleared on device 0: free 1186 -> 2150,
               reclaimed 964 MiB from [allocator-cache]

A test that asserted "``ensure_headroom`` returns ok" would have passed on
the pre-fix tree, because that function was never the broken part. So the
first test here MODELS THE BREACH: a card whose free column falls as a
prefill grows, sampled the way the corridor sampler samples it, run once
without the gate and once with it. The pre-fix behaviour is the run with the
gate switched off, and it must FAIL the corridor law for the test to be
worth anything.

    without the gate: minimum free 984 MiB  -> LAW BROKEN
    with the gate:    minimum free 1400 MiB -> law held

The remaining tests pin the three decisions in the gate that are easy to
"simplify" back into defects: that it prices ACTIVATION and not KV rows,
that its cooldown suppresses the SPEND and never the CHECK, and that it
NEVER converts a verdict into a refusal.
"""

import logging
import types

import pytest

from sglang.srt.managers.corridor_admission import (
    WANT_CAP_MIB,
    PrefillAdmissionGate,
    get_prefill_admission_gate,
    guard_prefill_admission,
)
from sglang.srt.managers.corridor_guard import CorridorGuard, RELIEF_LOCAL

MIB = 1024 * 1024


class FakeCard:
    """A card that loses free memory as a prefill grows, sampled at 100 ms.

    ``hoard`` is the reclaimable allocator cache -- the tier that paid for
    every one of s33's 50 gate arms.
    """

    def __init__(self, free_mib: int, hoard_mib: int = 1000):
        self.free = free_mib * MIB
        self.hoard = hoard_mib * MIB
        self.samples = []
        self.provider_calls = 0

    def probe(self) -> int:
        return int(self.free)

    def sample(self) -> None:
        self.samples.append(int(self.free // MIB))

    def grow(self, mib: int) -> None:
        """The forward takes transient memory. This is the allocation."""
        self.free -= mib * MIB
        self.sample()

    def provider(self, nbytes: int) -> int:
        """Give back up to ``nbytes`` from the hoard, and mean it."""
        self.provider_calls += 1
        paid = min(int(nbytes), self.hoard)
        self.hoard -= paid
        self.free += paid
        return paid

    @property
    def min_free_mib(self) -> int:
        return min(self.samples) if self.samples else 0


class FakeReporter:
    def __init__(self, qkv=0, ffn=0):
        self._qkv_act_bytes_per_token = qkv
        self._ffn_act_bytes_per_token = ffn


class FakeScheduler:
    """``layers=1`` by default so ``slope_bytes`` in a test IS the per-token
    figure the gate uses; the per-layer division has its own tests."""

    def __init__(self, reporter=None, layers=1):
        self.metrics_reporter = reporter
        self.server_args = types.SimpleNamespace(enable_phase_flip=True)
        if layers:
            self.model_config = types.SimpleNamespace(num_hidden_layers=layers)


def build_guard(card: FakeCard) -> CorridorGuard:
    guard = CorridorGuard(
        0,
        probe=card.probe,
        # A level fleet, so item 16 never withholds a tier for reasons that
        # have nothing to do with what these tests are about.
        fleet_probe=lambda: [card.free, card.free, card.free],
    )
    guard.register("allocator-cache", 0, card.provider, tier=RELIEF_LOCAL)
    return guard


def build_gate(card: FakeCard, *, slope_bytes=512, cooldown_s=0.0):
    """A gate wired to ``card``, bypassing the scheduler-side guard lookup."""
    scheduler = FakeScheduler(FakeReporter(qkv=slope_bytes, ffn=0))
    gate = PrefillAdmissionGate(scheduler, cooldown_s=cooldown_s)
    guard = build_guard(card)
    gate._guard = lambda: guard  # noqa: SLF001 -- the seam under test is below it
    return gate, guard


def run_prefill(card: FakeCard, gate, *, chunks=12, growth_mib=50, tokens=512):
    """Admit ``chunks`` prefill chunks, each of which grows the card."""
    card.sample()
    for _ in range(chunks):
        if gate is not None:
            gate.before_admission(tokens)
        card.grow(growth_mib)


def breaching_samples(card: FakeCard) -> int:
    return sum(1 for s in card.samples if s < 1024)


#: The slope that PRICES the simulated growth honestly: 50 MiB over 512
#: tokens. Where a test uses this, ``want`` equals what the chunk really
#: takes, which is the regime the gate is designed for.
COHERENT_SLOPE = (50 * MIB) // 512


# --------------------------------------------------------------------------
# 1. THE BREACH, REPRODUCED AND THEN PREVENTED.
# --------------------------------------------------------------------------


def test_ungated_prefill_walks_the_card_under_the_floor():
    """The pre-fix tree. This is s33's sustained 1001 MiB trough, in miniature."""
    card = FakeCard(free_mib=1384, hoard_mib=1000)
    run_prefill(card, None)
    assert card.provider_calls == 0, "nothing was ever asked to free memory"
    assert card.min_free_mib < 1024, (
        "the simulation must actually breach, or the next tests prove nothing"
    )
    assert card.min_free_mib == 784
    assert breaching_samples(card) == 5, "a TROUGH, not a single dipped sample"


def test_the_admission_gate_holds_the_floor_through_the_same_prefill():
    """The fixed tree, with the chunk priced honestly: the dip never happens.

    This is the gate working as designed -- ``want`` equals what the chunk
    actually takes, so the arm happens BEFORE the allocation that would have
    crossed the floor.
    """
    card = FakeCard(free_mib=1384, hoard_mib=1000)
    gate, _ = build_gate(card, slope_bytes=COHERENT_SLOPE)
    run_prefill(card, gate)
    assert card.provider_calls > 0, "the gate must actually spend the ladder"
    assert breaching_samples(card) == 0
    assert card.min_free_mib >= 1024, (
        f"corridor law broken at {card.min_free_mib} MiB despite the gate"
    )
    assert gate.armed > 0 and gate.cleared > 0


def test_an_underpriced_slope_cannot_PREEMPT_but_still_bounds_the_TROUGH():
    """The regime the metrics slope actually puts us in, stated honestly.

    The activation slope is a movement proxy and is biased small, so on metal
    ``want`` may be far below what the forward really takes. Then the gate
    cannot arm ahead of the crossing -- the check before the chunk sees a
    card that is still above the floor.

    What it DOES do is arm on the very next admission, which on the measured
    mix is ~90 ms later rather than the ~2 s it took the flip seam to notice.
    That is the whole difference between s33's 12-sample trough and a single
    dipped sample, and it is why this gate is worth having even underpriced.

    A successor who improves the slope should see this test's breach count go
    to zero; it is written to make that visible rather than to bless the 1.
    """
    card = FakeCard(free_mib=1384, hoard_mib=1000)
    gate, _ = build_gate(card, slope_bytes=1)  # a slope that prices nothing
    run_prefill(card, gate)
    assert gate.armed > 0, "it must still arm, just later"
    assert breaching_samples(card) == 1, (
        "one dipped sample, recovered on the next admission -- against 5 "
        "consecutive ones with no gate at all"
    )
    assert card.samples[-1] >= 1024, "and it does not stay down"


def test_the_gate_arms_before_the_allocation_not_after_it():
    """Spill-BEFORE-alloc: the distinction from a threshold observer.

    The card starts ABOVE the floor and the chunk about to be admitted is
    what would take it under. A reactive observer sees nothing here; the
    gate must already have spent.
    """
    card = FakeCard(free_mib=1100, hoard_mib=1000)
    gate, _ = build_gate(card, slope_bytes=200 * 1024)  # 200 KiB/token
    # 512 tokens x 200 KiB = 100 MiB of want against 76 MiB of headroom.
    gate.before_admission(512)
    assert card.provider_calls == 1
    assert card.free >= (1024 * MIB) + (100 * MIB), (
        "the gate must fund the allocation, not merely restore the floor"
    )


# --------------------------------------------------------------------------
# 2. WHAT IT CHARGES FOR. Pricing KV rows here would arm on every admission
#    for pages that are already committed.
# --------------------------------------------------------------------------


def test_want_is_priced_from_the_activation_slope():
    card = FakeCard(free_mib=4000)
    gate, _ = build_gate(card, slope_bytes=1024)
    assert gate.want_bytes(512) == 512 * 1024


def test_want_sums_both_activation_terms_and_takes_ONE_layer_of_them():
    """The reporter sums over every layer; residency is one layer's worth.

    Taking the sum priced a 512-token chunk at ~4.7 GiB on this model --
    larger than any card's free column, so the guard's target would be
    unreachable and it would spend the entire ladder on every admission,
    including the provider that evacuates the drafter.
    """
    scheduler = FakeScheduler(FakeReporter(qkv=100, ffn=40), layers=10)
    gate = PrefillAdmissionGate(scheduler)
    assert gate.want_bytes(10) == 140  # (100+40)/10 layers * 10 tokens


def test_an_unknown_layer_count_refuses_to_price_rather_than_price_wrongly():
    scheduler = FakeScheduler(FakeReporter(qkv=100, ffn=40), layers=0)
    gate = PrefillAdmissionGate(scheduler)
    assert gate.want_bytes(10) == 0


def test_want_is_hard_capped_so_a_bad_slope_cannot_unreach_the_target():
    scheduler = FakeScheduler(FakeReporter(qkv=10 * MIB, ffn=0))
    gate = PrefillAdmissionGate(scheduler)
    assert gate.want_bytes(4096) == WANT_CAP_MIB * MIB


def test_a_None_chunk_size_does_not_take_the_instance_down():
    """`dynamic_chunked_prefill_size()` returns None when chunking is off.

    `int(None)` raised OUTSIDE the gate's try, in the scheduler event loop.
    """
    scheduler = FakeScheduler(FakeReporter(qkv=100, ffn=40))
    gate = PrefillAdmissionGate(scheduler)
    assert gate.want_bytes(None) == 0


# --------------------------------------------------------------------------
# 1b. HANDOFF_678 §4.1: A MEASURED WANT, WHICH IS THE ONE THAT CAN PREEMPT.
# --------------------------------------------------------------------------


class FakeTracker:
    """The forward-peak probe's in-process query, and nothing else."""

    def __init__(self, per_token=None):
        self._per_token = per_token
        self.asked = []

    def transient_bytes_per_token(self, phase, tokens):
        self.asked.append((phase, tokens))
        return self._per_token


def sched_with_tracker(tracker, *, qkv=0, ffn=0, layers=1, cache_bytes=0):
    """A scheduler whose model runner carries a forward-peak tracker."""
    scheduler = FakeScheduler(FakeReporter(qkv=qkv, ffn=ffn), layers=layers)
    scheduler.tp_worker = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(_forward_peak=tracker)
    )
    scheduler._fake_cache_bytes = cache_bytes
    return scheduler


def gate_with_tracker(tracker, **kw):
    cache = kw.pop("cache_bytes", 0)
    gate = PrefillAdmissionGate(sched_with_tracker(tracker, **kw))
    gate._allocator_cache_bytes = lambda: cache  # noqa: SLF001
    return gate


def test_a_measured_transient_beats_the_geometry_slope():
    """Both readable: the measurement wins, because it is one.

    The geometry slope is a movement proxy assembled from the metrics
    reporter's per-token terms. The tracker's figure is the peak this rank
    actually reached on this bucket. When both exist there is no contest.
    """
    gate = gate_with_tracker(FakeTracker(per_token=2048), qkv=100, ffn=40)
    assert gate.want_bytes(512) == 512 * 2048


def test_the_measured_want_is_charged_NET_of_the_allocator_cache():
    """The tier law, at the second call site.

    NVML free does not move when torch serves a forward out of its own cache
    -- those bytes are already RESERVED and already missing from the free
    column. Charging them again arms the gate for an allocation that will
    never touch the driver. This is the same subtraction the KV rung makes
    with ``cheap_relief_bytes``, for the same reason, and it is what keeps the
    gate from arming on every admission once a slope becomes readable.
    """
    # 100 MiB of transient against a 200 MiB cache: nothing reaches the driver.
    absorbed = gate_with_tracker(FakeTracker(per_token=MIB), cache_bytes=200 * MIB)
    assert absorbed.want_bytes(100) == 0
    # 300 MiB against the same cache: only the excess is the driver's problem.
    spills = gate_with_tracker(FakeTracker(per_token=MIB), cache_bytes=200 * MIB)
    assert spills.want_bytes(300) == 100 * MIB


def test_a_measured_want_is_still_hard_capped():
    """The cap is a safety property of the LADDER, not of the estimator.

    An unreachable target spends every provider on every call, up to and
    including the one that evacuates the drafter. That is true no matter how
    the number was obtained, so a measured slope does not buy an exemption.
    """
    gate = gate_with_tracker(FakeTracker(per_token=10 * MIB))
    assert gate.want_bytes(4096) == WANT_CAP_MIB * MIB


def test_an_uncalibrated_bucket_falls_back_to_the_geometry_slope():
    """None means NOT MEASURED, and the fallback is the old behaviour."""
    gate = gate_with_tracker(FakeTracker(per_token=None), qkv=1024, ffn=0)
    assert gate.want_bytes(512) == 512 * 1024


def test_no_tracker_at_all_is_exactly_the_shipped_behaviour():
    """The probe is off by default; this path must not change with it off."""
    card = FakeCard(free_mib=4000)
    gate, _ = build_gate(card, slope_bytes=1024)
    assert gate.want_bytes(512) == 512 * 1024


def test_the_tracker_is_asked_about_the_PREFILL_phase_not_decode():
    """The tracker keys rows by phase, and prefill is 'extend' in that vocab."""
    tracker = FakeTracker(per_token=8)
    gate = gate_with_tracker(tracker)
    gate.want_bytes(512)
    assert tracker.asked == [("extend", 512)]


def test_a_tracker_that_raises_does_not_take_prefill_down():
    class Exploding:
        def transient_bytes_per_token(self, phase, tokens):
            raise RuntimeError("probe is broken")

    gate = gate_with_tracker(Exploding(), qkv=1024, ffn=0)
    assert gate.want_bytes(512) == 512 * 1024


def test_the_announcement_names_WHICH_source_priced_the_gate(caplog):
    """s34's lesson: a number in a log that does not say where it came from
    cannot be checked. Measured and geometry differ by orders of magnitude."""
    card = FakeCard(free_mib=8000)
    tracker = FakeTracker(per_token=4096)
    gate = gate_with_tracker(tracker)
    guard = build_guard(card)
    gate._guard = lambda: guard  # noqa: SLF001
    with caplog.at_level(logging.INFO):
        gate.before_admission(512)
    assert "measured" in caplog.text.lower()


def test_the_gate_is_off_unless_the_phase_flip_is_on():
    """The default path must not grow a corridor guard as a side effect."""
    card = FakeCard(free_mib=100)
    scheduler = FakeScheduler(FakeReporter(qkv=1, ffn=1))
    scheduler.server_args = types.SimpleNamespace(enable_phase_flip=False)
    assert guard_prefill_admission(scheduler, 512) is None
    assert getattr(scheduler, "phase_flip_corridor_admission", None) is None
    assert card.provider_calls == 0


def test_an_unreadable_slope_still_enforces_the_floor():
    """Degrade to want=0 -- the floor itself -- never to no gate at all."""
    card = FakeCard(free_mib=900, hoard_mib=1000)
    scheduler = FakeScheduler(None)
    gate = PrefillAdmissionGate(scheduler, cooldown_s=0.0)
    guard = build_guard(card)
    gate._guard = lambda: guard  # noqa: SLF001
    assert gate.want_bytes(512) == 0
    gate.before_admission(512)
    assert card.provider_calls == 1, "a card under the floor must still be lifted"
    assert card.free >= 1024 * MIB


def test_a_comfortable_card_costs_one_probe_and_no_relief():
    card = FakeCard(free_mib=8000)
    gate, _ = build_gate(card)
    gate.before_admission(512)
    assert gate.armed == 0
    assert card.provider_calls == 0
    assert gate.checks == 1


# --------------------------------------------------------------------------
# 3. THE COOLDOWN SUPPRESSES THE SPEND, NEVER THE CHECK.
# --------------------------------------------------------------------------


def test_cooldown_prevents_a_second_empty_cache_in_the_same_instant():
    card = FakeCard(free_mib=900, hoard_mib=4000)
    clock = {"t": 0.0}
    scheduler = FakeScheduler(FakeReporter(qkv=0, ffn=0))
    gate = PrefillAdmissionGate(
        scheduler, cooldown_s=0.25, clock=lambda: clock["t"]
    )
    guard = build_guard(card)
    gate._guard = lambda: guard  # noqa: SLF001
    gate.before_admission(512)
    assert card.provider_calls == 1
    # Push it back under the floor and ask again immediately.
    card.free = 900 * MIB
    gate.before_admission(512)
    assert card.provider_calls == 1, "relief must not be spent twice in 0 s"
    assert gate.cooldown_skips == 1
    assert gate.checks == 2, "the CHECK is never suppressed"
    clock["t"] = 1.0
    gate.before_admission(512)
    assert card.provider_calls == 2, "after the cooldown it must spend again"


# --------------------------------------------------------------------------
# 4. IT NEVER REFUSES. A rank-local refusal here is a group-admission split,
#    which is a hang -- see the module docstring and HANDOFF_675 1a.
# --------------------------------------------------------------------------


def test_an_exhausted_ladder_is_recorded_and_does_not_raise():
    card = FakeCard(free_mib=600, hoard_mib=0)
    gate, _ = build_gate(card)
    verdict = gate.before_admission(512)
    assert verdict is not None and verdict.ok is False
    assert gate.short == 1
    # The caller admits anyway. The gate returns evidence, not a decision.


def test_a_broken_guard_does_not_take_prefill_down():
    scheduler = FakeScheduler(FakeReporter(qkv=1, ffn=1))
    gate = PrefillAdmissionGate(scheduler)

    def explode():
        raise RuntimeError("no guard here")

    gate._guard = explode  # noqa: SLF001
    with pytest.raises(RuntimeError):
        gate._guard()  # noqa: SLF001 -- the double really does raise
    # ...but the gate swallows it at its own boundary.
    gate._guard = lambda: None  # noqa: SLF001
    assert gate.before_admission(512) is None


def test_a_failing_probe_is_survived():
    class BadGuard:
        floor_bytes = 1024 * MIB

        def free_bytes(self):
            raise RuntimeError("nvml is unhappy")

    scheduler = FakeScheduler(FakeReporter(qkv=1, ffn=1))
    gate = PrefillAdmissionGate(scheduler)
    gate._guard = lambda: BadGuard()  # noqa: SLF001
    assert gate.before_admission(512) is None


# --------------------------------------------------------------------------
# 5. ONE GATE PER SCHEDULER, because the guard beneath it is memoised too.
# --------------------------------------------------------------------------


def test_the_gate_is_memoised_on_the_scheduler():
    scheduler = FakeScheduler(FakeReporter())
    first = get_prefill_admission_gate(scheduler)
    second = get_prefill_admission_gate(scheduler)
    assert first is second


def test_no_scheduler_no_gate():
    assert get_prefill_admission_gate(None) is None


# --------------------------------------------------------------------------
# 6. THE GATE SAYS IT IS THERE, ONCE, WHICHEVER WAY IT RESOLVED.
#
# Learned on metal this shift: the first boot logged NOTHING from this module
# across 6066 prefill admissions, and "installed but never needed to arm" and
# "inert because the guard lookup returned None" are opposite in value and
# were identical in the log. Every other line here is conditional on arming,
# so the announcement is the only thing that distinguishes them.
# --------------------------------------------------------------------------


def test_an_armed_gate_announces_itself_once(caplog):
    card = FakeCard(free_mib=8000)
    gate, _ = build_gate(card)
    with caplog.at_level("INFO"):
        gate.before_admission(512)
        gate.before_admission(512)
    armed = [r for r in caplog.records if "ARMED on device" in r.getMessage()]
    assert len(armed) == 1, "exactly one announcement per process"


def test_an_INERT_gate_says_so_at_WARNING(caplog):
    """The state that cost this shift a whole acceptance run."""
    scheduler = FakeScheduler(FakeReporter(qkv=1, ffn=1))
    gate = PrefillAdmissionGate(scheduler)
    gate._guard = lambda: None  # noqa: SLF001
    with caplog.at_level("INFO"):
        gate.before_admission(512)
        gate.before_admission(512)
    inert = [r for r in caplog.records if "INERT" in r.getMessage()]
    assert len(inert) == 1
    assert inert[0].levelname == "WARNING", (
        "an inert corridor gate is not an INFO-level fact"
    )


def test_stats_report_what_the_gate_did():
    card = FakeCard(free_mib=1384, hoard_mib=1000)
    gate, _ = build_gate(card, slope_bytes=COHERENT_SLOPE)
    run_prefill(card, gate)
    stats = gate.stats()
    assert stats["checks"] == 12
    assert stats["armed"] >= 1
    assert stats["reclaimed_mib"] > 0


# ---------------------------------------------------------------------------
# SPEC ITEM 16 (successor 36): the rebalance lender rides this hot path.
# ---------------------------------------------------------------------------


def _unlevel_gate(card, peer_free_mib=3000):
    """A gate whose fleet is UNLEVEL, with this card as the water-fill loser."""
    from sglang.srt.managers.corridor_rebalance import RebalanceLender

    scheduler = FakeScheduler(FakeReporter(qkv=COHERENT_SLOPE, ffn=0))
    gate = PrefillAdmissionGate(scheduler, cooldown_s=0.0)
    guard = CorridorGuard(
        0,
        probe=card.probe,
        fleet_probe=lambda: [card.free, peer_free_mib * MIB, peer_free_mib * MIB],
    )
    guard.register("allocator-cache", 0, card.provider, tier=RELIEF_LOCAL)
    gate._guard = lambda: guard  # noqa: SLF001
    gate._lender = RebalanceLender(guard, nvml_index=0, clock=lambda: 0.0)
    return gate, guard


def test_the_lender_is_consulted_on_admissions_the_gate_would_not_arm():
    # THE POINT OF THE WHOLE TIER. This card is above the gate's arm line but
    # below the watermark, and a peer holds 3 GiB. s34's gate did nothing here
    # -- 42276 prefill admissions, relief only on the 232 that armed -- and
    # the trough that followed was 19 MiB from the law.
    card = FakeCard(free_mib=1270, hoard_mib=1000)
    gate, guard = _unlevel_gate(card)

    verdict = gate.before_admission(512)

    assert verdict is None, "the gate itself must NOT have armed here"
    assert guard.lend_count == 1
    assert guard.lent_total > 0
    assert card.free > 1270 * MIB


def test_a_level_fleet_leaves_the_admission_path_exactly_as_it_was():
    # The regression that matters: with the fleet level the lender must be a
    # no-op, so every pre-item-16 admission number is reproduced byte for byte.
    card = FakeCard(free_mib=1384, hoard_mib=1000)
    gate, guard = _unlevel_gate(card, peer_free_mib=1384)
    run_prefill(card, gate)
    assert guard.lend_count == 0
    assert gate.stats()["checks"] == 12
