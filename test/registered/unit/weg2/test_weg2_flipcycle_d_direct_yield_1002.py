"""FLIPCYCLE H5b (02.10.): no D-direct prefill while a D->P flip is foreseeable.

y6z: the park RPC of a D->P flip waited 1.87 s (ep 8) and 1.96 s (ep 22) behind
a D-direct SHORT pass (expert-major, pool.host_fetch at the link floor). When a
P-bound request is already queued, the SHORT rides P's batch instead.
"""

import collections
import inspect
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
        self.assertEqual(src.count("self._d_direct_yields(rid)"), 2)


if __name__ == "__main__":
    unittest.main()
