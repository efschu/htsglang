"""FLIPCYCLE H6 (02.10.): the waker's kv_cache resume rides its weights leg.

y6z P->D: `wake-kv` p50 158 / p90 185 ms on the front against rpc_ms 41 on D --
the separate kv RPC waited for D's post-wake scheduler pass (CTRL-RECV of the kv
call 81 ms after the weights leg ended). The weights leg now carries kv_cache
with ``kv_late`` (the late site after the legs, z30y7 bounded fit wait; never the
early resume before the legs that cost barlink its peer mapping, xsn315-318) and
the front issues no second call. A refused kv resume makes the weights leg
non-200 (z30y7), which the front already stops on.
"""

import inspect
import unittest

from sglang.srt.environ import envs
from sglang.srt.managers import io_struct
from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.weg2 import front


class TheFusedWake(unittest.TestCase):
    def test_switch_on_by_default(self):
        self.assertTrue(envs.SGLANG_WEG2_ENABLE_WAKE_KV_FUSED.get())

    def test_request_carries_kv_late(self):
        r = io_struct.ResumeMemoryOccupationReqInput(tags=["weights_0", "kv_cache"], kv_late=True)
        self.assertTrue(r.kv_late)
        self.assertIsNone(io_struct.ResumeMemoryOccupationReqInput(tags=["kv_cache"]).kv_late)

    def test_kv_late_never_takes_the_early_resume(self):
        src = inspect.getsource(wu)
        self.assertIn('if _kv_in and not getattr(recv_req, "kv_late", None) else False', src)

    def test_front_fuses_and_skips_the_second_call(self):
        src = inspect.getsource(front.Front.flip)
        self.assertIn('"tags": list(pause_order) + ([KV_TAG] if _kv_fused else [])', src)
        self.assertIn('_w_payload["kv_late"] = True', src)
        self.assertIn("if w_code == 200 and not _kv_fused:", src)
        self.assertIn("code, body = w_code, w_body", src)
        # the early send keeps its own path (fusion is off under it)
        self.assertIn("and not bool(_kv_early_on())", src)
        # defined on every path (weights-resident form included)
        k = src.index("_kv_task = None   # fnFL2x83")
        self.assertLess(src.index("_kv_fused = False", k), src.index("if self.weights_resident:", k))

    def test_plan_with_kv_and_weights_and_not_fundable_is_late(self):
        from sglang.srt.weg2.wake_kv import wake_kv_plan

        self.assertEqual(wake_kv_plan(kv_in_tags=True, weights_in_tags=True, fundable=False,
                                      deferred=False, epoch="e1", epoch_done=None), "late")


class TheFusedFlipOnTheFakeFront(unittest.TestCase):
    """The GatheredLegsTest harness (test_weg2_flip_cost_1235) with the switch on."""

    def _harness(self):
        import importlib.util
        import pathlib

        p = pathlib.Path(__file__).parent / "test_weg2_flip_cost_1235.py"
        spec = importlib.util.spec_from_file_location("_flip_cost_1235", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.GatheredLegsTest("test_one_rpc_per_group_per_leg_not_one_per_tag")

    def test_kv_rides_the_wakers_weights_leg_and_no_second_kv_call(self):
        h = self._harness()
        f = h._front()
        h._flip(f)
        occ = [c for c in f.calls if c.path.endswith("memory_occupation")]
        wake_legs = [c for c in occ if c.group == "P" and c.path == "/resume_memory_occupation"]
        self.assertEqual(len(wake_legs), 1, [(c.group, c.path, c.tags) for c in occ])
        self.assertIn("kv_cache", wake_legs[0].tags)
        self.assertTrue(set(f.weights_tags) <= set(wake_legs[0].tags))
        self.assertEqual(f.stops, [])


if __name__ == "__main__":
    unittest.main()
