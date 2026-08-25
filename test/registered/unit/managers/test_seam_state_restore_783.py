"""#783 half 2: the re-admitted cutover population restores instead of recomputing.

WHERE THIS HAD TO GO, and why not where it looks like it should.
`readmit_seam_residents` (scheduler.py:4749) only RE-QUEUES. By the time it
runs, `reset_for_retract()` has already cleared `req_pool_idx` and the rows are
gone, so there is nothing to load into. The restore belongs where the request
gets rows BACK, which in the non-disaggregation path is after
`alloc_for_extend` (schedule_batch.py:2519-2520): from there every req has a
`req_pool_idx` and addressable rows. That mirrors the proven disagg shape --
`_pre_alloc(req)` then `req.load_kv_cache(...)`
(disaggregation/decode.py:730-736).

THE GUARD IS THE COPY ITSELF, NOT `is_retracted`.
`is_retracted` is True for decode-PRESSURE retractions too, and those never
copied anything (half 1 fires at exactly one site -- see
test_seam_state_copy_783). Guarding on `is_retracted` would therefore ask for a
restore that does not exist and would have to fail soft, which is the shape
that hides defects. Guarding on the PRESENCE OF THE COPY is self-limiting: only
the population that copied can restore, by construction, and no epoch check or
capability query is needed. It is also why this needs no flag.

FIXTURE SHAPE, prior-art gate applied to the harness this time (the lesson from
half 1): `prepare_for_extend` needs real pools, exactly like `release_req` --
`test_uniform_retract_count_583.py:105` settled that question for this repo
already. So the restore is a NAMED HELPER that is tested directly on real code,
plus a structural check that `prepare_for_extend` calls it at the right point.
The same split half 1 uses.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest

import torch

from sglang.test.test_utils import CustomTestCase

MAMBA_SLOT = torch.tensor([3], dtype=torch.int64)


class _Alloc:
    def __init__(self):
        self.kv = {"tokens": "LIVE"}
        self.loaded = False

    def supports_mamba_cpu_copy(self):
        return False

    def get_cpu_copy(self, indices, mamba_indices=None):
        return {"full": dict(self.kv), "swa": None}

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        self.kv = dict(kv_cache_cpu["full"])
        self.loaded = True


class _MambaPool:
    def __init__(self):
        self.state = {3: "conv+temporal@3"}

    def get_cpu_copy(self, indices):
        return [self.state[int(i)] for i in indices.tolist()]

    def load_cpu_copy(self, cpu, indices):
        for i, val in zip(indices.tolist(), cpu):
            self.state[int(i)] = val


def _fixture():
    from sglang.srt.managers.schedule_batch import Req

    mamba = _MambaPool()
    rtp = types.SimpleNamespace(
        req_to_token=torch.zeros((1, 8), dtype=torch.int64),
        mamba_pool=mamba,
        translate_mamba_indices=lambda ids: ids,
    )
    req = types.SimpleNamespace(
        rid="r0",
        req_pool_idx=0,
        seqlen=4,
        mamba_pool_idx=MAMBA_SLOT,
        mamba_state_cpu=None,
        kv_cache_cpu=None,
        is_retracted=True,
    )
    req.offload_kv_cache = types.MethodType(Req.offload_kv_cache, req)
    req.load_kv_cache = types.MethodType(Req.load_kv_cache, req)
    req._mamba_cpu_copy_is_mine = types.MethodType(Req._mamba_cpu_copy_is_mine, req)
    return req, rtp, _Alloc(), mamba


class TestTheRestoreHelper(CustomTestCase):
    def test_a_saved_copy_is_restored(self):
        """RED until half 2 lands. Copy out (half 1), clobber the device
        state, restore (half 2) -- both halves through real code."""
        from sglang.srt.managers.schedule_batch import restore_seam_state

        req, rtp, alloc, mamba = _fixture()
        req.offload_kv_cache(rtp, alloc)

        alloc.kv = {"tokens": "CLOBBERED"}
        mamba.state[3] = "CLOBBERED"

        restored = restore_seam_state(req, rtp, alloc)
        self.assertTrue(restored)
        self.assertEqual(alloc.kv["tokens"], "LIVE")
        self.assertEqual(mamba.state[3], "conv+temporal@3")

    def test_restoring_twice_is_not_possible(self):
        """RED until half 2 lands. The copy is consumed, so a second pass
        cannot re-apply stale bytes over newer state."""
        from sglang.srt.managers.schedule_batch import restore_seam_state

        req, rtp, alloc, mamba = _fixture()
        req.offload_kv_cache(rtp, alloc)
        self.assertTrue(restore_seam_state(req, rtp, alloc))

        alloc.kv = {"tokens": "NEWER"}
        self.assertFalse(restore_seam_state(req, rtp, alloc))
        self.assertEqual(alloc.kv["tokens"], "NEWER")


class TestTheGuardIsTheCopyNotTheFlag(CustomTestCase):
    """Controls. Both pass TODAY once the helper exists, and they are what
    keeps the restore off the decode-pressure population."""

    def test_a_pressure_retraction_has_nothing_to_restore(self):
        """`is_retracted` is True here and NO copy was taken -- half 1 fires at
        one site only. The restore must be a no-op, not a soft failure."""
        from sglang.srt.managers.schedule_batch import restore_seam_state

        req, rtp, alloc, mamba = _fixture()
        self.assertTrue(req.is_retracted)
        self.assertFalse(restore_seam_state(req, rtp, alloc))
        self.assertEqual(alloc.kv["tokens"], "LIVE")
        self.assertFalse(alloc.loaded)

    def test_a_fresh_request_is_untouched(self):
        from sglang.srt.managers.schedule_batch import restore_seam_state

        req, rtp, alloc, _ = _fixture()
        req.is_retracted = False
        self.assertFalse(restore_seam_state(req, rtp, alloc))
        self.assertFalse(alloc.loaded)


class TestItIsWiredIntoTheAllocationPath(CustomTestCase):
    def test_prepare_for_extend_restores_before_clearing_is_retracted(self):
        """STRUCTURAL, because `prepare_for_extend` needs real pools.

        Two properties, both load-bearing: the call exists, and it happens
        AFTER allocation (rows must exist) and BEFORE `is_retracted` is
        cleared, so the ordering cannot silently invert."""
        import inspect

        from sglang.srt.managers.schedule_batch import ScheduleBatch

        src = inspect.getsource(ScheduleBatch.prepare_for_extend)
        self.assertIn("restore_seam_state", src)

        alloc_at = src.index("alloc_for_extend(self)")
        restore_at = src.index("restore_seam_state")
        clear_at = src.index("req.is_retracted = False")
        self.assertLess(
            alloc_at, restore_at, "the restore must run AFTER rows are allocated"
        )
        self.assertLess(
            restore_at,
            clear_at,
            "the restore must run BEFORE is_retracted is cleared",
        )


class TestTheFixtureItselfIsSound(CustomTestCase):
    """MUST PASS TODAY, and it uses only code that already exists.

    Every other test here imports `restore_seam_state`, which does not exist
    yet, so they are all red by ImportError -- and an all-red file cannot tell
    a real finding from a broken fixture. That is the trap that bit this
    strand three times today. This case touches nothing new: if the stand-in
    req, the bound methods or the stub pools were wrong, it would fail here
    and the reds above would be meaningless.
    """

    def test_offload_still_works_on_this_fixture(self):
        req, rtp, alloc, mamba = _fixture()
        req.offload_kv_cache(rtp, alloc)
        self.assertIsNotNone(req.kv_cache_cpu)
        self.assertIsNotNone(req.mamba_state_cpu)
        self.assertEqual(req.mamba_state_cpu, ["conv+temporal@3"])


if __name__ == "__main__":
    unittest.main()
