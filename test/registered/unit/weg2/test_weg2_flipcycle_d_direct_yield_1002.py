"""FLIPCYCLE H5b (02.10.): no D-direct prefill while a D->P flip is foreseeable.

y6z: the park RPC of a D->P flip waited 1.87 s (ep 8) and 1.96 s (ep 22) behind
a D-direct SHORT pass (expert-major, pool.host_fetch at the link floor). When a
P-bound request is already queued, the SHORT rides P's batch instead.
"""

import collections
import inspect
import re
import types
import unittest

from sglang.srt.environ import envs
from sglang.srt.weg2 import front
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _pbound_flip_now_off(monkeypatch):
    """These tests pin the seat/KV verdict and the dwell holds for requests that
    need P -- the rule PBOUND-FLIP-NOW (default on, 02.10.) replaces; they keep
    covering it as the switch-off path."""
    monkeypatch.setenv("SGLANG_WEG2_PBOUND_FLIP_NOW", "0")



def _f(cands, fits=None):
    f = types.SimpleNamespace(counters=collections.Counter(), epoch=7)
    f._asr_live_p_cands = lambda: cands
    st = {"fits": {c.rid: True for c in cands} if fits is None else fits}
    f._asr_state = st
    return f


def _f_fetch():
    """Fake with the FETCH-COST-YIELD helper bound -- the pre-existing fakes
    (Default AUS path) never see it; only these tests wire it."""
    f = _f([])
    f._d_direct_yield_fetch_cost = lambda rid, uncached: (
        front.Front._d_direct_yield_fetch_cost(f, rid, uncached))
    return f


class AForeseeableFlipYields(unittest.TestCase):
    def test_p_bound_queued_yields(self):
        f = _f([types.SimpleNamespace(rid="weg2-8-19")])
        self.assertTrue(front.Front._d_direct_yields(f, "weg2-8-18"))
        self.assertEqual(f.counters["d_direct_yield"], 1)

    def test_nothing_queued_runs_on_d(self):
        f = _f([])
        self.assertFalse(front.Front._d_direct_yields(f, "weg2-8-18"))

    def test_itself_is_not_a_reason(self):
        f = _f([types.SimpleNamespace(rid="weg2-8-18")])
        self.assertFalse(front.Front._d_direct_yields(f, "weg2-8-18"))

    def test_a_blocked_p_request_does_not_flip_so_no_yield(self):
        # rule 29.09.: no seat for the P-bound one -> no flip; the SHORT waits on D
        f = _f([types.SimpleNamespace(rid="big")], fits={"big": False})
        self.assertFalse(front.Front._d_direct_yields(f, "short"))

    def test_switch_off(self):
        f = _f([types.SimpleNamespace(rid="weg2-8-19")])
        with envs.SGLANG_WEG2_ENABLE_D_DIRECT_YIELD.override(False):
            self.assertFalse(front.Front._d_direct_yields(f, "weg2-8-18"))

    def test_default_on_and_wired_into_the_short_seat(self):
        self.assertTrue(envs.SGLANG_WEG2_ENABLE_D_DIRECT_YIELD.get())
        src = inspect.getsource(front.Front._acquire_short_seat)
        self.assertEqual(src.count("self._d_direct_yields(rid, uncached=uncached)"), 2)


class BFetchCostYield(unittest.TestCase):
    """FETCH-COST-YIELD (1369 draft, built 1400): with NO foreseeable flip, a
    SHORT whose own D prefill is fetch-dominated rides P's batch instead.
    Measured floor (1362): one expert-fetch wave per offloaded MoE layer
    (48 fetches at 25..138 new tokens, 96 = two waves at 191..3072) at a
    per-fetch median of 12.7-17.2 ms -- ~624 ms of link stall per pass."""

    def test_fetch_cost_switch_off_keeps_h5b(self):
        # Default path unchanged: with the fetch switch off, an empty candidate
        # list still answers False straight away and the new code is never read.
        f = _f_fetch()
        with envs.SGLANG_WEG2_ENABLE_D_DIRECT_YIELD_FETCH.override(False):
            self.assertFalse(
                front.Front._d_direct_yields(f, "weg2-8-18", uncached=300))
        self.assertEqual(f.counters["d_direct_yield_fetch_cost"], 0)

    def test_fetch_cost_yields_at_default_budget(self):
        self.assertFalse(envs.SGLANG_WEG2_ENABLE_D_DIRECT_YIELD_FETCH.get())
        self.assertEqual(envs.SGLANG_WEG2_D_DIRECT_YIELD_FETCH_MS.get(), 600)
        f = _f_fetch()
        with envs.SGLANG_WEG2_ENABLE_D_DIRECT_YIELD_FETCH.override(True):
            self.assertTrue(front.Front._d_direct_yields(f, "r", uncached=90))
            self.assertEqual(f.counters["d_direct_yield_fetch_cost"], 1)
            # 8-token floor ("a D pass with >= 8 new tokens streams the routed
            # experts of all 48 layers"): below it the case does not apply.
            self.assertFalse(front.Front._d_direct_yields(f, "r", uncached=5))
            # uncached None (no front estimate) stays on D too.
            self.assertFalse(front.Front._d_direct_yields(f, "r"))
        self.assertEqual(f.counters["d_direct_yield_fetch_cost"], 1)

    def test_fetch_cost_budget_dials_the_class(self):
        # Budget 1300 exposes only the two-wave class (uncached >= 190,
        # estimate 2 x 48 x 13 = 1248); budget 600 sits at the one-wave floor.
        f = _f_fetch()
        with envs.SGLANG_WEG2_ENABLE_D_DIRECT_YIELD_FETCH.override(True):
            with envs.SGLANG_WEG2_D_DIRECT_YIELD_FETCH_MS.override(1300):
                self.assertFalse(
                    front.Front._d_direct_yields(f, "r", uncached=300))
            with envs.SGLANG_WEG2_D_DIRECT_YIELD_FETCH_MS.override(600):
                self.assertTrue(
                    front.Front._d_direct_yields(f, "r", uncached=300))

    def test_uncached_reaches_both_seat_branches_of_the_route(self):
        # Wiring (1369 review point): the route passes the priced uncached
        # remainder to BOTH seat paths -- the ARRIVAL-SEAT branch and the
        # plain short-seat branch. Losing the second one would silently
        # make the fetch rule unreachable off ASR.
        src = inspect.getsource(front.Front.handle_generate)
        # the lookbehind keeps the X-route's ``est_uncached=remainder`` out
        self.assertEqual(
            len(re.findall(r"(?<![A-Za-z_])uncached=remainder", src)), 2)


if __name__ == "__main__":
    unittest.main()
