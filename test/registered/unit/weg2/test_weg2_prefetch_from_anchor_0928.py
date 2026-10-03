"""FA (28.09.): the store read starts at the END of the match's host chain.

NF rc12z22 D (dkrnfh91dprsavisnoadoptstbar1dauer09281447, 7d507357b9) TP0: the
parked weg2-10-16 came back from its wake read complete (``HiCache prefetch
success ... matched=18240 loaded=50944``, 69184 = KV + Mamba state), then its
admission registration read ``#823 HEAD-VOTE ANCHOR matched=50944 (device 0 +
host 50944) anchor_depth=69184`` -- the 18240-token device node carries no
recurrent state, so the mamba validator cut the device match to 0 and the host
sum counted only 50944. The span started at 50944 with the LAST HASH OF 69184
(``last_host_node`` is ``best_match_node``): ``#1472 READ-TRACE asked=286
readable=0 why=no-file`` -- keys that exist nowhere; weg2-4-11 asked the same
first stem at 14:56:01 and again at 14:56:26 (asked=41/43). Nothing was
missing: no Mamba blob, no KV page. A store-short read and 5 s of X-DEFER
followed until the X gate priced the true remainder (uncached=42).

Pinned here with the real ``_prefetch_kvcache`` on a recording tree cache:
the span starts at ``state_anchor_depth`` (the registration asks for the tail
only; the HP1 span base is the anchor); a match without a deeper anchor, the PP
group, a tree without a controller and the switch off keep the old start.
"""

import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import scheduler as sched_mod  # noqa: E402

DEVICE, HOST, ANCHOR, PROMPT = 18240, 50944, 69184, 69226   # weg2-10-16


class _Tree:
    def __init__(self, controller=True):
        self.ongoing_prefetch = {}
        self.calls = []
        self.hicache_storage_pass_prefix_keys = False
        self.root_node = object()
        self.cache_controller = types.SimpleNamespace() if controller else None

    def prefetch_from_storage(self, req_id, last_host_node, new_input_tokens, last_hash, prefix_keys, **kw):
        self.calls.append((req_id, list(new_input_tokens), last_hash, kw.get("span_base")))
        self.ongoing_prefetch[req_id] = object()


def _sched(tree, pp_size=1):
    s = types.SimpleNamespace(enable_hicache_storage=True, tree_cache=tree, page_size=64,
                              ps=types.SimpleNamespace(pp_size=pp_size, pp_rank=0, tp_rank=0))
    s._prefetch_kvcache = types.MethodType(sched_mod.Scheduler._prefetch_kvcache, s)
    s._note_prefetch_unregistered = types.MethodType(sched_mod.Scheduler._note_prefetch_unregistered, s)
    return s


def _req(anchor=ANCHOR):
    r = types.SimpleNamespace(rid="probe-10-16")   # not "weg2-": no hand-off file read
    r.prefix_indices = []                          # the device node was refused (no state)
    r.host_hit_length = HOST
    r.state_anchor_depth = anchor
    r.full_untruncated_fill_ids = list(range(PROMPT))
    r.init_next_round_input = lambda *a, **k: None
    r.last_host_node = types.SimpleNamespace(backuped=True, get_last_hash_value=lambda: "h@69184",
                                             get_prefix_hash_values=lambda parent: None, parent=None)
    r._compute_max_prefix_len = lambda n: n - 1
    return r


def _span(tree):
    (_rid, toks, last_hash, base), = tree.calls
    return toks, last_hash, base


class TheReadStartsAtTheHostChainsEnd(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop("SGLANG_WEG2_PREFETCH_FROM_ANCHOR", None)

    def tearDown(self):
        self._env.stop()

    def test_weg2_10_16_asks_only_the_tail(self):
        tree = _Tree()
        _sched(tree)._prefetch_kvcache(_req())
        toks, last_hash, _base = _span(tree)
        self.assertEqual(toks, list(range(ANCHOR, PROMPT - 1)))   # 41 tokens, not 18281
        self.assertEqual(last_hash, "h@69184")

    def test_before_it_asked_the_device_length_at_the_wrong_offset(self):
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_PREFETCH_FROM_ANCHOR": "0"}):
            tree = _Tree()
            _sched(tree)._prefetch_kvcache(_req())
        toks, last_hash, _base = _span(tree)
        self.assertEqual(toks[0], HOST)                  # [50944, ...) under the hash of 69184
        self.assertEqual(len(toks), PROMPT - 1 - HOST)   # 18281 tokens = 286 pages: READ-TRACE asked=286

    def test_the_hp1_span_base_is_the_anchor(self):
        tree = _Tree()
        s = _sched(tree)
        s.tree_cache.prefetch_participation_is_collective = lambda: True
        with mock.patch.object(sched_mod.tp_match_floor, "adopt_host_prefetch_span", lambda *a, **k: None):
            s._prefetch_kvcache(_req())
        self.assertEqual(_span(tree)[2], ANCHOR)

    def test_no_deeper_anchor_keeps_the_old_start(self):
        for anchor in (None, HOST, 0):
            tree = _Tree()
            _sched(tree)._prefetch_kvcache(_req(anchor=anchor))
            self.assertEqual(_span(tree)[0][0], HOST, anchor)

    def test_pp_group_and_no_controller_keep_the_old_start(self):
        for tree, pp in ((_Tree(), 3), (_Tree(controller=False), 1)):
            _sched(tree, pp_size=pp)._prefetch_kvcache(_req())
            self.assertEqual(_span(tree)[0][0], HOST)

    def test_the_helper_names_the_move(self):
        s = _sched(_Tree())
        with self.assertLogs(sched_mod.logger.name, level="INFO") as cm:
            self.assertEqual(sched_mod._weg2_prefetch_span_start(s, _req(), HOST), ANCHOR)
        self.assertTrue(any("FA PREFETCH-FROM-ANCHOR" in m and "anchor=69184" in m for m in cm.output))


if __name__ == "__main__":
    unittest.main()
