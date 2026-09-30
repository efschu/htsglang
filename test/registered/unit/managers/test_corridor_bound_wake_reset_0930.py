"""y3r Klasse E/A2: the #1028c window must not carry the dormant phase into the wake.

Metal (y3r, ...dauer09292330, D TP1 = 3080, wake 18 at 23:45:28-30):

    23:45:25  #1028c BOUND driver_free bound=81.8 instant=81.8     (D dormant, P owns the card)
    23:45:29  #1028c BOUND driver_free bound=81.8 instant=2443.8   (after the resume)
    23:45:30  #1028c BOUND driver_free bound=81.8 instant=2187.8
    23:45:32  #1028c BOUND driver_free bound=81.8 instant=2099.8
    23:45:34  #1028c BOUND driver_free bound=2087.8                (the 81.8 aged out)

and in between '[#656 CORRIDOR-ADMISSION] NARROWED ... from 4096 to 64 tokens:
the card can fund -124 MiB ... the full width prices at 198 MiB' plus the
group's '#794 GROUP-NARROWED ... from 4096 to 64'. The 220-token extend of
pdflip-35-47 ran as 64 + 156 (2490 + 1922 ms) while the card had ~2100 MiB free
and the 158-row piece measured a 67 MiB transient.

The tests model that sequence with the gate's own arithmetic (price 198 MiB at
4096 tokens, delta 256 MiB) and a fake clock. On the base (c1c012dd5e /
b2b6b51323) the post-wake grant is 64; with the wake reset it is 4096.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.managers.corridor_admission import PrefillAdmissionGate  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MIB = 1024 * 1024

#: y3r TP1: 4096 tokens price at 198 MiB ('the full width prices at 198 MiB').
PRICE_4096_MIB = 198.0
DELTA_MIB = 256.0


class _Clock:
    def __init__(self, t0: float = 1000.0):
        self.t = t0

    def __call__(self) -> float:
        return self.t


class _Guard:
    """The corridor guard as the gate reads it: a free column and a delta."""

    def __init__(self):
        self.free_mib = 81.8
        self.delta_mib = DELTA_MIB

    def free_bytes(self) -> float:
        return self.free_mib * MIB


def _gate(monkeypatch, wake_t=500.0):
    sched = SimpleNamespace(
        server_args=SimpleNamespace(
            tp_size=1,
            gdn_prefill_scratch_mib=lambda share, tokens: PRICE_4096_MIB * tokens / 4096.0,
        ),
        _pdflip_last_wake_t=wake_t,
    )
    gate = PrefillAdmissionGate(sched)
    guard = _Guard()
    monkeypatch.setattr(gate, "_guard", lambda: guard)
    # the cache term is 0 off metal (no CUDA); the trapped probe is injected so
    # the snapshot path is never reached
    monkeypatch.setattr(gate, "_allocator_cache_bytes", lambda: 0)
    gate._graph_pool_free_probe = lambda: 0
    clock = _Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    return sched, gate, guard, clock


def _dormant_then_wake(sched, gate, guard, clock):
    """D sleeps 5 s at 81.8 MiB (P owns the card), then wakes to 2443.8."""
    for _ in range(5):
        assert gate.granted_width(4096) == 64  # dormant: the card is P's
        clock.t += 1.0
    guard.free_mib = 2443.8
    sched._pdflip_last_wake_t = 900.0  # weight_updater: '...admission seams admit'
    clock.t += 0.4


def test_post_wake_grant_is_the_full_width(monkeypatch):
    sched, gate, guard, clock = _gate(monkeypatch)
    _dormant_then_wake(sched, gate, guard, clock)
    # base: min(81.8, 2443.8) - 256 < 0 -> 64 ('#794 GROUP-NARROWED 4096 -> 64')
    assert gate.granted_width(4096) == 4096
    # a real D extend of 220 tokens (ep36) is one forward, not 64 + 156
    assert gate.granted_width(220) == 220
    assert gate.bound_wake_resets == 1


def test_the_bound_is_still_a_minimum_within_the_phase(monkeypatch):
    """The reset drops the OTHER phase only: a dip after the wake still binds."""
    sched, gate, guard, clock = _gate(monkeypatch)
    _dormant_then_wake(sched, gate, guard, clock)
    assert gate.granted_width(4096) == 4096
    clock.t += 1.0
    guard.free_mib = 300.0  # 300 - 256 = 44 MiB spendable
    got = gate.granted_width(4096)
    assert 64 <= got < 4096
    assert got * PRICE_4096_MIB / 4096 <= 44.0 + 1e-6
    clock.t += 1.0
    guard.free_mib = 2400.0  # recovered, but the dip is still in the window
    assert gate.granted_width(4096) == got
    assert gate.bound_wake_resets == 1


def test_no_second_reset_without_a_new_wake(monkeypatch):
    sched, gate, guard, clock = _gate(monkeypatch)
    _dormant_then_wake(sched, gate, guard, clock)
    for _ in range(8):
        gate.granted_width(4096)
        clock.t += 1.0
    assert gate.bound_wake_resets == 1


def test_switch_off_is_the_old_window(monkeypatch):
    with envs.FLLIPER_PDFLIP_CORRIDOR_BOUND_WAKE_RESET.override(False):
        sched, gate, guard, clock = _gate(monkeypatch)
        _dormant_then_wake(sched, gate, guard, clock)
        assert gate.granted_width(4096) == 64
        assert gate.bound_wake_resets == 0


def test_no_wake_attribute_is_byte_identical(monkeypatch):
    """Non-pdflip boots never write _pdflip_last_wake_t: the window never resets."""
    sched, gate, guard, clock = _gate(monkeypatch, wake_t=None)
    del sched._pdflip_last_wake_t
    for _ in range(3):
        gate.granted_width(4096)
        clock.t += 1.0
    guard.free_mib = 2443.8
    assert gate.granted_width(4096) == 64
    assert gate.bound_wake_resets == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
