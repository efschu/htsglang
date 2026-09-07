# SPDX-License-Identifier: Apache-2.0
"""Weg 2 draft KV across the flip (#1233, FIX 1 of boot weg2dk1's review):
the draft L3 WRITE instrument counts what the backend did.

``_draft_page_set`` (C19) folds the backend's answer into
``_draft_l3_write_issued`` / ``_draft_l3_write_refused`` -- the counters the
PUBLISH-SWEEP line and the acceptance letter read (L3 ``complete>0
refused=0``). Both implementations behind it therefore carry a bool
contract: a page the backend accepted is ISSUED, a page it refused is
REFUSED, never the other way round (instrument-text-lies law).
"""

import unittest

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.test.test_utils import CustomTestCase

DRAFTER = "a30db4b7c362c786"


class _Backend:
    def __init__(self, answer):
        self.answer = answer
        self.keys = []
        self.v2 = []

    def batch_set(self, keys, values):
        self.keys.extend(keys)
        assert len(keys) == len(values)
        return self.answer

    def batch_set_v2(self, transfers):
        self.v2.extend(transfers)
        return {t.name: [self.answer] * len(t.keys) for t in transfers}


class _DraftPool:
    def get_size_per_token(self):
        return 2048

    def get_data_page(self, index):
        return torch.full((2048,), int(index) + 1, dtype=torch.uint8)


def _controller(backend):
    ctrl = HiCacheController.__new__(HiCacheController)
    ctrl.storage_backend = backend
    ctrl.mem_pool_host_draft = _DraftPool()
    ctrl.page_size = 1
    ctrl.draft_identity = DRAFTER
    ctrl.canonical_draft_page_window = None
    ctrl._draft_l3_write_issued = 0
    ctrl._draft_l3_write_refused = 0
    ctrl._draft_set_warned_at = 0
    ctrl.draft_page_set_func = ctrl._draft_page_set_generic
    return ctrl


class TestDraftL3WriteContract(CustomTestCase):
    def test_accepted_pages_are_issued_not_refused(self):
        be = _Backend(True)
        ctrl = _controller(be)
        self.assertIs(ctrl._draft_page_set_generic(["h1", "h2"], [0, 1]), True)
        self.assertTrue(ctrl._draft_page_set(["h3", "h4", "h5"], [2, 3, 4]))
        self.assertEqual(
            (ctrl._draft_l3_write_issued, ctrl._draft_l3_write_refused), (3, 0)
        )
        self.assertEqual(
            be.keys,
            [f"h{i}.draft-{DRAFTER}" for i in range(1, 6)],
        )

    def test_refused_pages_are_refused_not_issued(self):
        be = _Backend(False)
        ctrl = _controller(be)
        self.assertIs(ctrl._draft_page_set_generic(["h1"], [0]), False)
        self.assertFalse(ctrl._draft_page_set(["h2", "h3"], [1, 2]))
        self.assertEqual(
            (ctrl._draft_l3_write_issued, ctrl._draft_l3_write_refused), (0, 2)
        )

    def test_v2_route_carries_the_same_contract(self):
        ctrl = _controller(_Backend(True))
        self.assertIs(
            ctrl._draft_page_set_v2(["h1", "h2"], torch.tensor([0, 1])), True
        )
        self.assertEqual(ctrl.storage_backend.v2[0].name, PoolName.DRAFT)
        ctrl = _controller(_Backend(False))
        self.assertIs(ctrl._draft_page_set_v2(["h1"], torch.tensor([0])), False)

    def test_an_exception_is_a_named_refusal(self):
        def boom(keys, values):
            raise OSError("no space left on device")

        be = _Backend(True)
        be.batch_set = boom
        ctrl = _controller(be)
        self.assertFalse(ctrl._draft_page_set(["h1"], [0]))
        self.assertEqual(
            (ctrl._draft_l3_write_issued, ctrl._draft_l3_write_refused), (0, 1)
        )


if __name__ == "__main__":
    unittest.main()
