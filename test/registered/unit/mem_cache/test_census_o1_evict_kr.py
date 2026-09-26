"""KR 26.09.: the producer-phase ledgers' FIFO eviction at the cap.

Boot n4h (dkr27bnvfp4bar1mwh09261131): one 217,614-token prefetch completion on
group D noted 217,614 arrivals while ``_arrived`` stood at 467,649 of its
524,288 cap; the 161k evictions each walked the growing run of deleted slots
at the front of the dict (``next(iter(d))``) and held the scheduler thread for
~6 s inside ``check_prefetch_progress`` -- the kv resume sat unread in TP0's
socket for that long. ``SGLANG_WEG2_CENSUS_O1_EVICT=1`` evicts with
``OrderedDict.popitem(last=False)``.

What is pinned here:
  * switch OFF: the plain dict stays a plain dict, contents and FIFO order as
    before (the default path is unchanged);
  * switch ON: identical contents, identical order, identical counters to the
    OFF path over the same key stream -- only the container and the cost move;
  * switch ON: the eviction cost does not grow with the number of evictions
    (the quadratic is gone), measured as a ratio so a slow CI box cannot flake;
  * the batched ``note_prefetch_adopted`` under the switch equals per-key
    ``note_arrival`` (late arrivals included).
Hermetic: no GPU, no sglang runtime.
"""

import time
from collections import OrderedDict

import pytest

import sglang.srt.mem_cache.producer_phase_census as m


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(m.ENV_O1_EVICT, raising=False)
    m.reset_for_test()
    yield
    m.reset_for_test()


def _arm(monkeypatch, on: bool):
    monkeypatch.setenv(m.ENV_O1_EVICT, "1" if on else "0")
    m.reset_for_test()


def _drive(keys):
    for k in keys:
        m.note_arrival(k)
        m.note_store_write(k, 3)
    return (list(m._arrived.items()), list(m._ledger.items()),
            m.arrival_stats(), m.ledger_stats())


def test_default_is_off_and_keeps_a_plain_dict(monkeypatch):
    monkeypatch.setattr(m, "_LEDGER_MAX", 4)
    assert m.census_o1_evict_armed() is False
    _drive([f"k{i}" for i in range(10)])
    assert type(m._arrived) is dict and type(m._ledger) is dict
    assert list(m._arrived) == ["k6", "k7", "k8", "k9"]
    assert list(m._ledger) == ["k6", "k7", "k8", "k9"]


@pytest.mark.parametrize("cap", [1, 3, 7])
def test_on_equals_off_over_the_same_stream(monkeypatch, cap):
    monkeypatch.setattr(m, "_LEDGER_MAX", cap)
    # repeats and re-arrivals of evicted keys on purpose: a key already in the
    # map must not move, an evicted one re-enters at the back.
    stream = [f"k{i % 11}" for i in range(40)] + ["k0", "k1", "k0"]
    m.note_consult("k5", accepted=False)  # one late arrival in the stream
    off = _drive(stream)
    _arm(monkeypatch, True)
    m.note_consult("k5", accepted=False)
    on = _drive(stream)
    assert on == off
    assert isinstance(m._arrived, OrderedDict) and isinstance(m._ledger, OrderedDict)


def test_on_does_not_convert_below_the_cap(monkeypatch):
    _arm(monkeypatch, True)
    monkeypatch.setattr(m, "_LEDGER_MAX", 100)
    _drive([f"k{i}" for i in range(10)])
    assert type(m._arrived) is dict  # nothing evicted, nothing converted


def test_batched_adoption_equals_per_key(monkeypatch):
    monkeypatch.setattr(m, "census_armed", lambda: 1)
    monkeypatch.setattr(m, "_LEDGER_MAX", 5)
    keys = [f"p{i}" for i in range(12)] + [None, "p3"]
    m.note_consult("p2", accepted=False)
    for k in keys:
        m.note_arrival(k)
    ref = (list(m._arrived.items()), m.arrival_stats())
    _arm(monkeypatch, True)
    m.note_consult("p2", accepted=False)
    m.note_prefetch_adopted(keys)
    assert (list(m._arrived.items()), m.arrival_stats()) == ref


def _evict_cost(monkeypatch, on: bool, fill: int, n: int) -> float:
    _arm(monkeypatch, on)
    monkeypatch.setattr(m, "_LEDGER_MAX", fill)
    for i in range(fill):
        m.note_arrival(f"f{i}")
    t0 = time.perf_counter()
    for i in range(n):
        m.note_arrival(f"e{i}")
    return time.perf_counter() - t0


def test_on_eviction_cost_is_linear_not_quadratic(monkeypatch):
    # 4x the evictions must cost ~4x (not ~16x). Generous bound (8x) so a noisy
    # box cannot flake; the OFF path at this size is measurably super-linear.
    small = _evict_cost(monkeypatch, True, 40000, 10000)
    large = _evict_cost(monkeypatch, True, 40000, 40000)
    assert large < 8.0 * max(small, 1e-4), (small, large)
