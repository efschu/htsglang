"""NF-RC31 B (rc30 review, Folgefix B): the SPILL-MEMO state carries a host-lock term and the write-throughs in flight.

The W3 room spill keeps a node that is ``host_locked`` or ``write_pending``. When that lock is let go
(``UnifiedRadixCache.dec_host_lock_ref``) or the write-through retires, the next spill could release it -- but no
arena or tree term of the old memo state moved, so the memo answered "released nothing" again without a walk
(review: Auflage 6, ``SPILL-MEMO SKIP`` in series while ``#1421 BACKUP-REFUSED`` / ``UD-LAST DROP-AFTER`` run).

Danger direction "a stale memo hides a release": mutant = the two terms dropped from ``state`` -> the release after
the lock is skipped.
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flliper.srt.pdflip import spill_memo as SM  # noqa: E402

import test_evict_fit_spill_memo_1021 as _base  # noqa: E402  (module import: its TestCases are not collected twice)

_env, _follower = _base._env, _base._follower


def _cache():
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    c, pool = _follower()
    c.disable = False
    c._components_tuple = ()
    c._update_evictable_leaf_sets = lambda node: None
    c.ongoing_write_through = {}
    c.dec_host_lock_ref = types.MethodType(UnifiedRadixCache.dec_host_lock_ref, c)
    return c, pool


class HostLockTerm(unittest.TestCase):
    _run = _base.SpillMemo._run

    def test_a_released_host_lock_walks_again(self):
        with _env(**{SM.ENV: 1}):
            c, pool = _cache()
            self.assertEqual(len(self._run(c, pool, 3)), 1)          # memoised after the first empty walk
            c.dec_host_lock_ref(types.SimpleNamespace(id=7))
            self.assertEqual(c.__dict__[SM.HOST_LOCK_GEN], 1)      # the literal in unified_radix_cache matches
            self.assertEqual(len(self._run(c, pool, 3)), 1, "the lock release must forget the memo")

    def test_a_retired_write_through_walks_again(self):
        with _env(**{SM.ENV: 1}):
            c, pool = _cache()
            c.ongoing_write_through = {11: object()}
            self.assertEqual(len(self._run(c, pool, 2)), 1)
            c.ongoing_write_through = {}
            self.assertEqual(len(self._run(c, pool, 2)), 1)

    def test_no_change_still_skips(self):
        with _env(**{SM.ENV: 1}):
            c, pool = _cache()
            self.assertEqual(len(self._run(c, pool, 10)), 1)
            self.assertEqual(SM.counters(c)[1], 9)

    def test_mutant_without_the_terms_skips_the_release(self):
        real = SM.state

        def old_state(cache, pool):
            st = real(cache, pool)
            return None if st is None else st[:-2]

        with _env(**{SM.ENV: 1}), mock.patch.object(SM, "state", old_state):
            c, pool = _cache()
            self.assertEqual(len(self._run(c, pool, 1)), 1)
            c.dec_host_lock_ref(types.SimpleNamespace(id=7))
            self.assertEqual(len(self._run(c, pool, 1)), 0, "mutant must reproduce the stale skip")


if __name__ == "__main__":
    unittest.main()
