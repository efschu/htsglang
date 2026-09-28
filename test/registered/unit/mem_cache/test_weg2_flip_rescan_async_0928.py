"""Flip regression 28.09.: the wake's L3 store rescan ran INSIDE the resume RPC.

27B flip timeline wake-kv -> dc: 0.3 s at 41,869 store files (13:00) -> 4.8 s
at 700,338 (18:51); WEG2-WAKE-TAIL store_rescan= carried all of it, linear in
the persistent store. The walk now runs on a thread; its W8b verdict is voted
at the release fence.
"""

import ast
import os
import threading
import types
import unittest

from sglang.srt.managers.scheduler_components import weight_updater as WU
from sglang.srt.mem_cache.weg2_store_gates import Weg2StoreIndexBlind


def _fake(backend):
    controller = types.SimpleNamespace(storage_backend=backend)
    scheduler = types.SimpleNamespace(
        tree_cache=types.SimpleNamespace(cache_controller=controller),
        enable_hierarchical_cache=True)
    M = WU.SchedulerWeightUpdaterManager
    fake = types.SimpleNamespace(scheduler=scheduler, weg2_store_rescan_failure="",
                                 weg2_store_rescan_thread=None)
    fake._weg2_rescan_store_index_sync = lambda: M._weg2_rescan_store_index_sync(fake)
    fake._weg2_join_store_rescan = lambda where: M._weg2_join_store_rescan(fake, where)
    return M, fake


class _Gated:
    """A walk that finishes only when the test says so."""

    def __init__(self, blind=False):
        self.go = threading.Event()
        self.started = threading.Event()
        self.blind = blind

    def rescan_eviction_index(self):
        self.started.set()
        self.go.wait(5)
        if self.blind:
            raise Weg2StoreIndexBlind("W8b: 0.0 % of the store is indexed")
        return {"indexed_entries": 1, "seen_entries": 1, "indexed_bytes": 1,
                "seen_bytes": 1, "fraction": 1.0}


class TestRescanLeavesTheResumeRpc(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.pop("SGLANG_WEG2_STORE_RESCAN_ASYNC", None)

    def tearDown(self):
        if self._env is not None:
            os.environ["SGLANG_WEG2_STORE_RESCAN_ASYNC"] = self._env

    def test_the_wake_returns_before_the_walk_ends(self):
        b = _Gated()
        M, fake = _fake(b)
        M._weg2_rescan_store_index(fake)          # must not wait for the walk
        self.assertTrue(b.started.wait(5))
        self.assertIsNotNone(fake.weg2_store_rescan_thread)
        self.assertTrue(fake.weg2_store_rescan_thread.is_alive())
        b.go.set()
        M._weg2_join_store_rescan(fake, "test")
        self.assertIsNone(fake.weg2_store_rescan_thread)

    def test_a_blind_verdict_is_there_after_the_join(self):
        b = _Gated(blind=True)
        M, fake = _fake(b)
        M._weg2_rescan_store_index(fake)
        self.assertEqual(fake.weg2_store_rescan_failure, "")   # not yet
        b.go.set()
        M._weg2_join_store_rescan(fake, "release")
        self.assertIn("Weg2WakeRefused", fake.weg2_store_rescan_failure)

    def test_switch_off_keeps_the_in_rpc_walk(self):
        os.environ["SGLANG_WEG2_STORE_RESCAN_ASYNC"] = "0"
        b = _Gated()
        b.go.set()
        M, fake = _fake(b)
        M._weg2_rescan_store_index(fake)
        self.assertIsNone(fake.weg2_store_rescan_thread)
        self.assertTrue(b.started.is_set())


class TestTheReleaseFenceVotesTheAsyncVerdict(unittest.TestCase):
    def test_release_joins_then_votes(self):
        src = open(WU.__file__).read()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "release_memory_occupation")
        body = ast.get_source_segment(src, fn)
        j = body.find('_weg2_join_store_rescan("release")')
        f = body.find('self._weg2_group_fence(')
        self.assertGreater(j, 0, "the release leg never joins the async rescan")
        self.assertLess(j, f, "the join must precede the release fence")
        voted = any(kw.arg == "ok" and "store_failure" in (ast.get_source_segment(src, kw.value) or "")
                    for n in ast.walk(fn) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "_weg2_group_fence"
                    for kw in n.keywords)
        self.assertTrue(voted, "the release fence's ok-bit does not carry the store verdict")


if __name__ == "__main__":
    unittest.main()
