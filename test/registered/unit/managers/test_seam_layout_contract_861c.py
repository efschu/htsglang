"""#861c: the seam copy carries its LAYOUT identity, and a restore into a different one REFUSES.

This is the caller half of #861c. The pool half is
test/registered/unit/mem_cache/test_seam_layer_layout_861c.py, which pins that
`load_cpu_copy` stops trusting its own `layer_num` for a list it did not build.
That guard REFUSES -- which, on its own, turns the W40 IndexError into a
ValueError and still takes the scheduler down. Something above it has to decline
the restore before it is attempted. That is this file.

THE SHAPE IS THE ONE #783 ALREADY ESTABLISHED FOR A DIFFERENT AXIS.
`restore_seam_state` (schedule_batch.py) compares `kv_cache_cpu_extent` -- how
many ROWS the copy covers -- against what the request now needs, and refuses on
drift rather than indexing (25e7849844). W40 is the same defect one axis over:
the copy also has a LAYER geometry, it was equally unrecorded, and the flip
changes it. So the extent field gains a sibling rather than the tree gaining a
second mechanism.

WHY THE COUNT GUARD BELOW IS NOT SUFFICIENT AND THIS IS NOT REDUNDANT WITH IT.
`check_cpu_copy_layers` compares two integers. It is structurally blind to a
layout change that keeps the count: two PP stages of the same size, or a stage
ratio permuted (32,18,14 -> 18,32,14 gives rank 0 a copy of 32 layers and a
destination of 18, caught; but 16,16,32 -> 16,16,32 with the ranks re-ordered
gives 16 into 16, NOT caught). The recorded identity carries `start_layer` as
well, so those cases are refused by name instead of restoring the wrong global
layers into a matching-sized hole.

REFUSAL, NOT REMAP -- the reasoning is in the pool-side file's header and is not
repeated here. Short form: rank-locally a PP stage's copy can never cover a TP
pool's layers, so any remap leaves most of the destination stale, and the KV
head sharding differs between the two phases as well.

Hermetic, no CUDA, no pool: the allocator is a stub and the assertion is on the
REFUSAL, i.e. on what `restore_seam_state` decides.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest

import torch

from sglang.test.test_utils import CustomTestCase

MAMBA_SLOT = torch.tensor([3], dtype=torch.int64)


class _Layout(tuple):
    """Stands in for whatever the pool declares. `restore_seam_state` must only
    ever compare these for equality -- it must not parse them, exactly as `Req`
    does not parse the copy payload."""


PP1 = _Layout(("kv", 18, 32))
TP = _Layout(("kv", 64, 0))


class _Alloc:
    """A pool that answers what layout it is, and whose `load_cpu_copy` is
    faithful about the consequence of being asked to restore into another one."""

    def __init__(self, layout):
        self.layout = layout
        self.loaded = False

    def cpu_copy_layout(self):
        return self.layout

    def supports_mamba_cpu_copy(self):
        return True  # the rig's pool (HybridLinearKVPool) owns the mamba copy

    def get_cpu_copy(self, indices, mamba_indices=None):
        return {"n": int(indices.numel()), "layout": self.layout}

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        if kv_cache_cpu["layout"] != self.layout:
            # What the real pool does: walks its own layer_num over a list built
            # for a different one. IndexError one way, wrong-layer writes the
            # other. Either way, reaching here at all is the failure.
            raise IndexError("list index out of range")
        self.loaded = True


def _req(alloc, *, allocated=20, logical_len=21):
    from sglang.srt.managers.schedule_batch import Req

    req = types.SimpleNamespace(
        rid="rid-861c",
        req_pool_idx=0,
        seqlen=logical_len,
        kv_allocated_len=allocated,
        mamba_pool_idx=MAMBA_SLOT,
        mamba_state_cpu=None,
        kv_cache_cpu=None,
        kv_cache_cpu_extent=None,
        kv_cache_cpu_layout=None,
    )
    req.offload_kv_cache = types.MethodType(Req.offload_kv_cache, req)
    req.load_kv_cache = types.MethodType(Req.load_kv_cache, req)
    req._mamba_cpu_copy_is_mine = types.MethodType(Req._mamba_cpu_copy_is_mine, req)
    rtp = types.SimpleNamespace(
        req_to_token=torch.zeros((1, 64), dtype=torch.int64),
        mamba_pool=None,
        translate_mamba_indices=lambda ids: ids,
    )
    return req, rtp


class TestTheCopyRecordsItsLayout(CustomTestCase):
    def test_offload_records_the_layout_it_was_taken_from(self):
        """RED. Without this nothing downstream can tell a flip from a no-op --
        which is precisely the state W40 crashed in."""
        alloc = _Alloc(PP1)
        req, rtp = _req(alloc)
        req.offload_kv_cache(rtp, alloc)
        self.assertEqual(req.kv_cache_cpu_layout, PP1)


class TestTheRestoreRefusesOnLayoutDrift(CustomTestCase):
    def test_a_pp_copy_is_refused_by_a_tp_pool(self):
        """THE SPECIMEN, at the level that keeps the scheduler alive.

        Copy taken in PP, flip, restore attempted in TP. Before the fix
        `restore_seam_state` saw the extents agree (they do -- the row count did
        not change) and called straight into the pool."""
        from sglang.srt.managers.schedule_batch import restore_seam_state

        pp = _Alloc(PP1)
        req, rtp = _req(pp)
        req.offload_kv_cache(rtp, pp)

        tp = _Alloc(TP)
        self.assertFalse(restore_seam_state(req, rtp, tp))
        self.assertFalse(tp.loaded)

    def test_a_tp_copy_is_refused_by_a_pp_pool(self):
        """THE MIRROR. Loud here, silent in the pool before the fix."""
        from sglang.srt.managers.schedule_batch import restore_seam_state

        tp = _Alloc(TP)
        req, rtp = _req(tp)
        req.offload_kv_cache(rtp, tp)

        pp = _Alloc(PP1)
        self.assertFalse(restore_seam_state(req, rtp, pp))
        self.assertFalse(pp.loaded)

    def test_equal_layer_counts_at_different_offsets_are_refused(self):
        """The case the pool-level COUNT guard is structurally blind to: same
        number of layers, different global layers. Nothing below this refuses
        it, so if this line goes green by accident the wrong KV lands."""
        from sglang.srt.managers.schedule_batch import restore_seam_state

        a = _Alloc(_Layout(("kv", 16, 0)))
        req, rtp = _req(a)
        req.offload_kv_cache(rtp, a)

        b = _Alloc(_Layout(("kv", 16, 16)))
        self.assertFalse(restore_seam_state(req, rtp, b))
        self.assertFalse(b.loaded)

    def test_a_refused_copy_is_DROPPED_not_kept(self):
        """#783's rule, inherited: a copy held past its refusal can be applied
        later by a coincidentally-matching layout, and by then it is ancient."""
        from sglang.srt.managers.schedule_batch import restore_seam_state

        pp = _Alloc(PP1)
        req, rtp = _req(pp)
        req.offload_kv_cache(rtp, pp)
        restore_seam_state(req, rtp, _Alloc(TP))
        self.assertIsNone(req.kv_cache_cpu)
        self.assertIsNone(req.kv_cache_cpu_layout)
        self.assertIsNone(req.mamba_state_cpu)

    def test_the_refusal_is_counted_and_names_the_request(self):
        """A refusal nobody can see reads as a seam that is carrying state. The
        operator must be able to tell 'the flip loses its prefixes' from 'the
        flip carries them'."""
        from sglang.srt.managers import schedule_batch as sb

        before = sb._SEAM_STATE_COUNTS.get("refused_layout", 0)
        pp = _Alloc(PP1)
        req, rtp = _req(pp)
        req.offload_kv_cache(rtp, pp)
        with self.assertLogs("sglang.srt.managers.schedule_batch", "WARNING") as logs:
            sb.restore_seam_state(req, rtp, _Alloc(TP))
        self.assertEqual(sb._SEAM_STATE_COUNTS.get("refused_layout", 0), before + 1)
        blob = "\n".join(logs.output)
        self.assertIn("rid-861c", blob)


MAMBA_PP1 = _Layout(("mamba", 18, 32))
MAMBA_TP = _Layout(("mamba", 64, 0))


class _MambaPool:
    def __init__(self, layout):
        self.layout = layout
        self.loaded = False

    def cpu_copy_layout(self):
        return self.layout

    def get_cpu_copy(self, indices):
        return {"layout": self.layout}

    def load_cpu_copy(self, cpu, indices):
        if cpu["layout"] != self.layout:
            raise IndexError("list index out of range")
        self.loaded = True


class _AllocWithoutMamba(_Alloc):
    def supports_mamba_cpu_copy(self):
        return False  # so `Req` owns the mamba copy, schedule_batch.py:1769


class TestTheMambaHalfCarriesItsOwnLayout(CustomTestCase):
    """The briefing's second named sibling. `Req` takes a SECOND copy from a
    SECOND pool when the KV pool declares it does not move mamba, and the flip
    splits the mamba layers by the same `--pp-stage-ratio`. One stamp for both
    copies would let a mamba-layer change ride in under a matching KV layout."""

    def _req_with_mamba(self, alloc, mamba_pool):
        req, rtp = _req(alloc)
        req.mamba_state_cpu_layout = None
        rtp.mamba_pool = mamba_pool
        return req, rtp

    def test_offload_records_the_mamba_layout_separately(self):
        alloc = _AllocWithoutMamba(TP)
        req, rtp = self._req_with_mamba(alloc, _MambaPool(MAMBA_PP1))
        req.offload_kv_cache(rtp, alloc)
        self.assertEqual(req.kv_cache_cpu_layout, TP)
        self.assertEqual(req.mamba_state_cpu_layout, MAMBA_PP1)

    def test_a_mamba_layout_change_under_a_matching_kv_layout_is_refused(self):
        """THE CASE A SINGLE STAMP WOULD MISS. The KV layout agrees; only the
        mamba split moved. Nothing else in the path looks at it."""
        from sglang.srt.managers.schedule_batch import restore_seam_state

        alloc = _AllocWithoutMamba(TP)
        req, rtp = self._req_with_mamba(alloc, _MambaPool(MAMBA_PP1))
        req.offload_kv_cache(rtp, alloc)

        rtp.mamba_pool = _MambaPool(MAMBA_TP)
        self.assertFalse(restore_seam_state(req, rtp, alloc))
        self.assertFalse(alloc.loaded)
        self.assertFalse(rtp.mamba_pool.loaded)

    def test_a_matching_mamba_layout_still_restores(self):
        from sglang.srt.managers.schedule_batch import restore_seam_state

        alloc = _AllocWithoutMamba(TP)
        req, rtp = self._req_with_mamba(alloc, _MambaPool(MAMBA_TP))
        req.offload_kv_cache(rtp, alloc)
        self.assertTrue(restore_seam_state(req, rtp, alloc))
        self.assertTrue(alloc.loaded)
        self.assertTrue(rtp.mamba_pool.loaded)


class TestTheMatCHINGCaseStillRestores(CustomTestCase):
    """THE PATH THAT MUST KEEP WORKING. A guard that refuses everything is not
    a fix, it is the feature switched off."""

    def test_a_same_layout_restore_is_still_performed(self):
        from sglang.srt.managers.schedule_batch import restore_seam_state

        pool = _Alloc(TP)
        req, rtp = _req(pool)
        req.offload_kv_cache(rtp, pool)
        self.assertTrue(restore_seam_state(req, rtp, pool))
        self.assertTrue(pool.loaded)

    def test_a_pool_that_cannot_state_a_layout_is_not_refused_on_that_ground(self):
        """CHECKED, NOT ASSUMED. `UnifiedKVPool` and `DeepSeekV4UnifiedKVPool`
        declare no base class (allocator/base.py:347 says so for the sibling
        predicate), so they cannot answer. Refusing them would switch the seam
        off for pools that never had this defect; the pool-level count guard
        still covers them. The tolerance is deliberate and is stated here so it
        cannot be mistaken for an oversight."""
        from sglang.srt.managers.schedule_batch import restore_seam_state

        class _Mute(_Alloc):
            def cpu_copy_layout(self):
                return None

        pool = _Mute(TP)
        req, rtp = _req(pool)
        req.offload_kv_cache(rtp, pool)
        self.assertIsNone(req.kv_cache_cpu_layout)
        self.assertTrue(restore_seam_state(req, rtp, pool))


if __name__ == "__main__":
    unittest.main()
