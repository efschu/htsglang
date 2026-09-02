"""#1060 / #1068 slice 3 (T14, G7): the store-presence cache on a request is
keyed by the HiCache binding generation.

THE DEFECT: `_prefetch_kvcache` caches the content-key presence probe on the
request as `(matched_len, span_len) -> present`. The key carries no binding
generation, so a verdict taken under the PP binding (before a cutover)
governs the TP binding (after it): a store that was absent for PP's pool
answers 'absent' for TP without ever being asked (#1060).

RED on 846c6797b9: the key is `(_matched_len, len(_new_input_tokens))`; with
the generation advanced the probe is NOT re-run (call count stays 1).

The real `Scheduler._prefetch_kvcache` is bound to a stand-in that supplies
exactly what the method reads; the probe counts its calls.
"""

import types
import unittest
from unittest import mock

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache import hicache_phase_binding as hpb
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Node:
    def __init__(self, backuped=False):
        self.backuped = backuped
        self.parent = None
        self.key = None

    def get_last_hash_value(self):
        return "hash"

    def get_prefix_hash_values(self, parent):
        return []


class _Probe:
    def __init__(self, answer=True):
        self.calls = 0
        self.answer = answer

    def __call__(self, tokens, last_hash, prefix_keys):
        self.calls += 1
        return self.answer


class _Tree:
    def __init__(self, probe):
        self.root_node = _Node(backuped=True)
        self.cache_controller = types.SimpleNamespace(store_presence_pages=probe)
        self.hicache_storage_pass_prefix_keys = False
        self.ongoing_prefetch = {}
        self.prefetch_calls = 0

    def prefetch_from_storage(self, *a, **k):
        self.prefetch_calls += 1


class _Req:
    def __init__(self, rid="r", n=600):
        self.rid = rid
        self.prefix_indices = []
        self.host_hit_length = 0
        self.full_untruncated_fill_ids = list(range(n))
        self.last_host_node = _Node(backuped=False)

    def init_next_round_input(self, tree_cache, cow_mamba=False):
        pass

    def _compute_max_prefix_len(self, n):
        return n - 1


class _Sched:
    _prefetch_kvcache = Scheduler._prefetch_kvcache

    def __init__(self, probe):
        self.enable_hicache_storage = True
        self.tree_cache = _Tree(probe)


class TestTheKeyCarriesTheGeneration(CustomTestCase):
    def test_a_new_generation_reprobes_the_store(self):
        probe = _Probe(answer=False)
        s = _Sched(probe)
        req = _Req()
        with mock.patch.object(hpb, "current_generation", return_value=4):
            s._prefetch_kvcache(req)
            s._prefetch_kvcache(req)
        self.assertEqual(probe.calls, 1, "same generation, same span: cached")
        with mock.patch.object(hpb, "current_generation", return_value=5):
            s._prefetch_kvcache(req)
        self.assertEqual(
            probe.calls,
            2,
            "the binding generation advanced (cutover): the PP verdict must "
            "not govern TP -- the store has to be asked again",
        )
        self.assertEqual(req._pp_store_presence_cache[0][0], 5)

    def test_the_same_generation_keeps_the_cache(self):
        probe = _Probe(answer=True)
        s = _Sched(probe)
        req = _Req()
        with mock.patch.object(hpb, "current_generation", return_value=9):
            for _ in range(3):
                s._prefetch_kvcache(req)
        self.assertEqual(probe.calls, 1)

    def test_the_prefetch_span_is_stamped_on_the_request(self):
        # slice 3 A12.2: the UNDEFERRABLE check needs the request's own span.
        probe = _Probe(answer=True)
        s = _Sched(probe)
        req = _Req(n=600)
        with mock.patch.object(hpb, "current_generation", return_value=1):
            s._prefetch_kvcache(req)
        self.assertEqual(req._prefetch_span_tokens, 599)


if __name__ == "__main__":
    unittest.main()
