"""#783: a saved KV copy carries the extent it covers, and a restore REFUSES on drift.

THE CLASS THIS CLOSES: a LOGICAL length used as a PHYSICAL extent.
`Req.seqlen` is `len(origin_input_ids) + len(output_ids)`
(schedule_batch.py:1124-1127) -- the request's FINAL logical length, which says
nothing about how much of `req_to_token` has actually been written. During
chunked prefill the row is filled only to `extend_range.end ==
req.kv_allocated_len`, strictly less. Both `offload_kv_cache` and
`load_kv_cache` index by `self.seqlen - 1`, so both are correct ONLY once
prefill is complete.

HOW IT FAILED (W38-A, 2026-08-25 16:44:42, all three ranks): a restore whose
index extent exceeded the copy's walked off the end of the saved chunk list --
`kv_cache_cpu[layer_id][i // chunk_size]`, memory_pool.py:3295, IndexError.
The only length check below the caller is per-chunk
(`assert k_cpu.shape[0] == ... == len(chunk_indices)`, :3298); nothing checks
the NUMBER of chunks, which is the axis that overflowed. Every pool-level
`load_cpu_copy` merely forwards the indices it is handed, so THERE IS NO
BACKSTOP BELOW THE CALLER -- the guarantee has to be established here.

WHY A REFUSAL AND NOT A CLAMP: the two extents describe different things. A
`min()` would write a prefix's KV into the wrong rows -- a wrong ANSWER instead
of a crash. Refusing costs a recompute, which is merely slow.

THE TREE ALREADY SETTLED THE UNDERLYING QUESTION and this file matches its
answer. `kv_session_offload.py` refuses chunked admission outright rather than
restoring into it -- ":445 return False  # would be CHUNKED -> needs PS3" and
the assert at :4496 naming "PS3 (host-prefix extend read)". PS3 has four
mentions in the tree, all comments: it is unimplemented. So mid-chunk restore
is NOT supported here either; it is refused loudly and counted, and the counter
measures how often the case actually arises instead of guessing it.

The in-tree proof that extent bookkeeping is the right shape is
`DecodeKVCacheOffloadManager` (decode_kvcache_offload_manager.py:130), which
tracks its own `prefill_len`/`inc_len` rather than assuming a full row.
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
        self.last_indices = None

    def supports_mamba_cpu_copy(self):
        return True  # the rig's pool (HybridLinearKVPool) owns the mamba copy

    def cpu_copy_layout(self):
        # #861c: the layout stamp lives beside the extent stamp. Constant here,
        # because this file's subject is the ROW axis; the LAYER axis has its
        # own file (test_seam_layout_contract_861c.py).
        return ("kv", 4, 0)

    def get_cpu_copy(self, indices, mamba_indices=None):
        self.last_indices = int(indices.numel())
        return {"full": dict(self.kv), "n": int(indices.numel())}

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        # Faithful to the real pool: it walks the saved chunks positionally and
        # cannot tolerate a longer index set.
        if int(indices.numel()) > int(kv_cache_cpu["n"]):
            raise IndexError("list index out of range")
        self.kv = dict(kv_cache_cpu["full"])
        self.loaded = True


def _req(*, allocated, logical_len):
    """A request whose PHYSICAL extent (`kv_allocated_len`) and LOGICAL length
    (`seqlen`) are set independently -- which is the whole point."""
    from sglang.srt.managers.schedule_batch import Req

    req = types.SimpleNamespace(
        rid="r0",
        req_pool_idx=0,
        seqlen=logical_len,
        kv_allocated_len=allocated,
        mamba_pool_idx=MAMBA_SLOT,
        mamba_state_cpu=None,
        kv_cache_cpu=None,
        kv_cache_cpu_extent=None,
    )
    req.offload_kv_cache = types.MethodType(Req.offload_kv_cache, req)
    req.load_kv_cache = types.MethodType(Req.load_kv_cache, req)
    req._mamba_cpu_copy_is_mine = types.MethodType(Req._mamba_cpu_copy_is_mine, req)
    rtp = types.SimpleNamespace(
        req_to_token=torch.zeros((1, 64), dtype=torch.int64),
        mamba_pool=None,
        translate_mamba_indices=lambda ids: ids,
    )
    return req, rtp, _Alloc()


class TestTheCopyRecordsWhatItCovers(CustomTestCase):
    def test_offload_records_the_extent(self):
        """RED. The copy must say how much it covers, or nothing downstream
        can tell drift from agreement."""
        req, rtp, alloc = _req(allocated=20, logical_len=21)
        req.offload_kv_cache(rtp, alloc)
        self.assertEqual(req.kv_cache_cpu_extent, 20)


class TestTheRestoreRefusesOnDrift(CustomTestCase):
    def test_a_grown_extent_is_refused_not_indexed(self):
        """RED, and this is the W38-A crash turned into a refusal.

        The request comes back with MORE rows than were copied. The old code
        indexed and raised IndexError three layers down in the pool; the
        contract must decline here instead."""
        from sglang.srt.managers.schedule_batch import restore_seam_state

        req, rtp, alloc = _req(allocated=20, logical_len=21)
        req.offload_kv_cache(rtp, alloc)

        # Re-admitted with a longer row: exactly the W38-A shape.
        req.seqlen = 41
        req.kv_allocated_len = 40

        restored = restore_seam_state(req, rtp, alloc)
        self.assertFalse(restored, "a drifted extent must be refused")
        self.assertFalse(alloc.loaded, "nothing may be written on a refusal")

    def test_a_matching_extent_restores(self):
        """RED. The refusal must not be a blanket no."""
        from sglang.srt.managers.schedule_batch import restore_seam_state

        req, rtp, alloc = _req(allocated=20, logical_len=21)
        req.offload_kv_cache(rtp, alloc)
        alloc.kv = {"tokens": "CLOBBERED"}

        self.assertTrue(restore_seam_state(req, rtp, alloc))
        self.assertEqual(alloc.kv["tokens"], "LIVE")

    def test_a_refusal_drops_the_copy(self):
        """RED. A refused copy must not linger: it would be stale against a
        request the model has since advanced, and a later matching extent
        would then restore ancient bytes."""
        from sglang.srt.managers.schedule_batch import restore_seam_state

        req, rtp, alloc = _req(allocated=20, logical_len=21)
        req.offload_kv_cache(rtp, alloc)
        req.seqlen = 41
        req.kv_allocated_len = 40
        restore_seam_state(req, rtp, alloc)

        self.assertIsNone(getattr(req, "kv_cache_cpu", None))
        self.assertIsNone(req.kv_cache_cpu_extent)


class TestMidChunkIsNotCopiedAtAll(CustomTestCase):
    def test_an_incomplete_prefill_is_not_copied(self):
        """RED. A request still being chunk-prefilled has no well-defined full
        extent, so the seam must not copy it -- and must SAY it declined, so
        the PS3 question is measured rather than guessed."""
        from sglang.srt.managers.schedule_batch import seam_copy_state

        # allocated 12 of a 40-token prompt: mid-chunk
        req, rtp, alloc = _req(allocated=12, logical_len=41)
        copied = seam_copy_state(req, rtp, alloc)
        self.assertFalse(copied)
        self.assertIsNone(getattr(req, "kv_cache_cpu", None))

    def test_a_complete_prefill_is_copied(self):
        """RED. The decode-phase resident -- the population the cutover
        actually retracts -- must be covered."""
        from sglang.srt.managers.schedule_batch import seam_copy_state

        req, rtp, alloc = _req(allocated=40, logical_len=41)
        self.assertTrue(seam_copy_state(req, rtp, alloc))
        self.assertEqual(req.kv_cache_cpu_extent, 40)


class TestTheFixtureItselfIsSound(CustomTestCase):
    """MUST PASS TODAY -- touches only code that already exists. Every other
    case here is red by ImportError or by a missing field, and an all-red file
    cannot distinguish a finding from a broken fixture. That trap has bitten
    this strand three times."""

    def test_offload_and_load_round_trip_on_this_fixture(self):
        req, rtp, alloc = _req(allocated=20, logical_len=21)
        req.offload_kv_cache(rtp, alloc)
        self.assertIsNotNone(req.kv_cache_cpu)
        self.assertEqual(alloc.last_indices, 20)

        alloc.kv = {"tokens": "CLOBBERED"}
        req.load_kv_cache(rtp, alloc)
        self.assertEqual(alloc.kv["tokens"], "LIVE")

    def test_the_stub_pool_reproduces_the_w38a_overflow(self):
        """The fixture must be ABLE to show the bug, or the reds above prove
        nothing about the real pool."""
        req, rtp, alloc = _req(allocated=20, logical_len=21)
        req.offload_kv_cache(rtp, alloc)
        req.seqlen = 41
        with self.assertRaises(IndexError):
            req.load_kv_cache(rtp, alloc)


if __name__ == "__main__":
    unittest.main()


class TestTheWiring(CustomTestCase):
    """The two halves are wired at exactly one site each, and in the right
    order. `prepare_for_extend` and `release_req` both need real pools
    (test_uniform_retract_count_583.py:105 settled that for this repo), so the
    wiring is checked structurally and the behaviour is checked above."""

    def test_the_cutover_is_the_only_caller_that_copies(self):
        import inspect

        import sglang.srt.managers.schedule_batch as sb
        from sglang.srt.managers import phase_flip_runtime

        self.assertIn(
            "copy_state=True",
            inspect.getsource(phase_flip_runtime.build_cutover_release),
        )
        self.assertNotIn(
            "copy_state=True", inspect.getsource(sb.ScheduleBatch.retract_all)
        )

    def test_release_req_routes_through_the_guard_not_raw_offload(self):
        """The copy must go through `seam_copy_state`, or a mid-chunk request
        would be copied at an extent no restore can accept -- which is the
        W38-A shape re-entering by the back door."""
        import inspect

        import sglang.srt.managers.schedule_batch as sb

        src = inspect.getsource(sb.release_req)
        self.assertIn("seam_copy_state(", src)

    def test_restore_runs_after_alloc_and_before_the_flag_is_cleared(self):
        import inspect

        from sglang.srt.managers.schedule_batch import ScheduleBatch

        src = inspect.getsource(ScheduleBatch.prepare_for_extend)
        self.assertLess(
            src.index("alloc_for_extend(self)"), src.index("restore_seam_state")
        )
        self.assertLess(
            src.index("restore_seam_state"), src.index("req.is_retracted = False")
        )
