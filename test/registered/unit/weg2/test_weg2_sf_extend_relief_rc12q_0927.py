# SPDX-License-Identifier: Apache-2.0
"""SF-X (rc12q PP2 15:38:49Z, weg2-4-37): the same-pass load-back spent the local-PP floor and the
SAME pass's extend (775) was funded by nobody.

  SF LOADBACK-ROOM SAME-PASS kv_tokens=5932 floor=5336 avail=5336 evictable=245003 evicted=810
  #988 LOADBACK prefix moved to 20479; WEG2-ARENA-LOAD rows=5932
  #969 EXTENT (20479, 21254) 775 -> "NO relief provider" -> Out of memory

Root: alloc_token_slots' trigger reads uniform_avail_for_evict = floor - admitted ledger = 5336 >= 775
and skips the eviction, while the live pool held 5336 + 810 - 5932 = 214. Fix: the load charges the
ledger on the local-PP form (the trigger then evicts), and a rank-local relief provider is the net.
"""

import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache import common as C  # noqa: E402
from sglang.srt.weg2 import pp_slot_fidelity as SF  # noqa: E402

FLOOR, EVICTABLE, LOAD, EXTEND = 5336, 245003, 5932, 775


class _Alloc:
    def __init__(self, avail):
        self.avail = avail

    def available_size(self):
        return self.avail

    def alloc(self, n):
        if self.avail < n:
            return None
        self.avail -= n
        return list(range(n))


class _Tree:
    def __init__(self, local_pp=True):
        self.token_to_kv_pool_allocator = _Alloc(FLOOR)
        self._ev = EVICTABLE
        self.uniform_avail_floor = FLOOR
        self.uniform_admitted_since_floor = 0
        setattr(self, SF.FLOOR_LOCAL_PP_ATTR, local_pp)

    def is_chunk_cache(self):
        return False

    def evictable_size(self):
        return self._ev

    def evict(self, params):
        n = min(int(params.num_tokens), self._ev)
        self._ev -= n
        self.token_to_kv_pool_allocator.avail += n
        return types.SimpleNamespace(num_tokens_evicted=n)

    def available_and_evictable_str(self):
        return f"avail={self.token_to_kv_pool_allocator.avail} evictable={self._ev}\n"

    def pretty_print(self):
        pass


def _load(tree, charge=True):
    """The b23/rc12q same-pass room + load: evict the shortfall, take LOAD rows."""
    assert SF.local_pp_room(tree, LOAD, FLOOR, "weg2-4-37") is True
    rows = tree.token_to_kv_pool_allocator.alloc(LOAD)
    assert rows is not None
    if charge:
        SF.note_loaded(tree, len(rows))


def _env(on=True):
    return mock.patch.dict(os.environ, {SF.ENV: "1" if on else "0"})


class TheExtendAfterTheLoadIsFunded(unittest.TestCase):
    def setUp(self):
        C.clear_extend_relief_providers()

    def tearDown(self):
        C.clear_extend_relief_providers()

    def test_rc12q_the_ledger_charge_makes_the_extend_trigger_evict(self):
        with _env(True):
            t = _Tree()
            _load(t)
            self.assertEqual(t.uniform_admitted_since_floor, LOAD)
            out = C.alloc_token_slots(t, EXTEND)
        self.assertEqual(len(out), EXTEND)

    def test_the_metal_behaviour_without_charge_or_net_is_the_oom(self):
        with _env(True):
            t = _Tree()
            _load(t, charge=False)
            with self.assertRaisesRegex(RuntimeError, "Out of memory"):
                C.alloc_token_slots(t, EXTEND)

    def test_the_relief_provider_is_the_net(self):
        with _env(True):
            t = _Tree()
            sched = types.SimpleNamespace(tree_cache=t)
            SF.ensure_relief_provider(sched)
            SF.ensure_relief_provider(sched)  # once per process
            self.assertEqual(len(C._extend_relief_providers), 1)
            _load(t, charge=False)
            with self.assertLogs(SF.logger, level="WARNING") as cap:
                out = C.alloc_token_slots(t, EXTEND)
        self.assertEqual(len(out), EXTEND)
        self.assertTrue(any("EXTEND-RELIEF evicted=" in m and "asked=775" in m for m in cap.output))

    def test_the_net_does_nothing_on_a_group_floor(self):
        with _env(True):
            t = _Tree(local_pp=False)
            SF.ensure_relief_provider(types.SimpleNamespace(tree_cache=t))
            ev0 = t._ev
            self.assertEqual(C._extend_relief_providers[0](EXTEND), 0)
            self.assertEqual(t._ev, ev0, "a TP group's tree is never touched rank-locally")

    def test_charge_only_on_the_local_pp_form_and_switch(self):
        t = _Tree(local_pp=False)
        with _env(True):
            SF.note_loaded(t, LOAD)
        self.assertEqual(t.uniform_admitted_since_floor, 0)
        t2 = _Tree()
        with _env(False):
            SF.note_loaded(t2, LOAD)
            SF.ensure_relief_provider(types.SimpleNamespace(tree_cache=t2))
        self.assertEqual(t2.uniform_admitted_since_floor, 0)
        self.assertEqual(C._extend_relief_providers, [])

    def test_wiring(self):
        from sglang.srt.managers import scheduler as S
        from sglang.srt.mem_cache import unified_radix_cache as U

        src = open(U.__file__).read()
        i = src.index("if device_indices is None:\n            self.dec_host_lock_ref(best_match_node, host_anchor_params)")
        self.assertIn("_sf.note_loaded(self, len(device_indices))", src[i:i + 500])
        s2 = open(S.__file__).read()
        self.assertIn("_sf.ensure_relief_provider(self)", s2)


if __name__ == "__main__":
    unittest.main()
