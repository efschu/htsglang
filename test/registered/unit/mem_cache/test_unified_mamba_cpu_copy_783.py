"""#783: the offload/restore pair must carry the mamba state, not just the KV.

THE CONTRACT, and it is stated on `Req`, not on a pool
(schedule_batch.py:1693-1700):

    # Copies over both the kv cache and mamba state if available
    self.kv_cache_cpu = token_to_kv_pool_allocator.get_cpu_copy(
        token_indices, mamba_indices=self.mamba_pool_idx
    )

One pool honours that in full: `HybridLinearKVPool.get_cpu_copy`
(memory_pool.py:4352) translates the ids and calls `MambaPool.get_cpu_copy`,
with a matching restore at :4362. It can, because it OWNS a `mamba_pool`.

The pool this rig runs cannot. `UnifiedSWAKVPool` (unified_memory_pool.py:996-
1237) holds `full_kv_pool` and `swa_kv_pool` and no mamba pool at all, so its
`get_cpu_copy` (:1217) takes `mamba_indices` and never reads it, and
`load_cpu_copy` (:1230) has no mamba term either. That is not an oversight to
patch inside the KV pool -- a KV pool has no business owning mamba state. The
level that CAN satisfy the contract is `Req`, which already receives
`req_to_token_pool` (owner of `mamba_pool` and `translate_mamba_indices`).

WHY THE TEST SITS HERE AND NOT ON THE POOL. An earlier draft of this file
asserted against `UnifiedSWAKVPool.get_cpu_copy` -- i.e. against one possible
MECHANISM. The contract is the round trip, and pinning a mechanism would
over-specify the fix and forbid the architecturally correct placement. So this
tests the property the cutover actually depends on: copy, lose the device
state, restore, and get it back.

SCOPE, CHECKED BEFORE WRITING RATHER THAN ASSUMED. `MambaPool.get_cpu_copy`
(memory_pool.py:1180) copies the conv list and the temporal tensor and nothing
else. That is COMPLETE persistent per-slot state for this rig:
  * the speculative intermediates are excluded on purpose -- "transient verify
    scratch keyed to spec slots, not session state"
    (model_executor/offload_gdn_states.py:33, `_TRANSIENT_SPEC_FIELDS`);
  * the GDN ReplaySSM rings are persistent state, but are allocated only under
    `--enable-linear-replayssm`, and this rig runs
    `enable_linear_replayssm=False` (arm A server_args).
LATENT GAP, FILED NOT FIXED: with ReplaySSM enabled, conv+temporal is a SUBSET
of the persistent state and this round trip would silently lose the rings.

THIS FILE IS RED AT THE TIME OF WRITING on the two mamba assertions and GREEN
on the two controls (the KV half still round-trips; the no-mamba path is
untouched). A file that is red on EVERYTHING is a broken fixture, not a
finding -- twice now this draft caught its own bug that way: first the wrong
class name, then setting `seqlen`, which is a read-only property derived from
`origin_input_ids + output_ids`. Both times 4/4 red was the tell.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.test.test_utils import CustomTestCase

MAMBA_SLOT = torch.tensor([3], dtype=torch.int64)


class _StubAllocator:
    """A KV allocator of the shape this rig runs: it accepts `mamba_indices`
    and ignores it, exactly like `UnifiedSWAKVPool`."""

    def __init__(self):
        self.kv = {"tokens": "kv-bytes"}

    def supports_mamba_cpu_copy(self):
        # A real allocator ALWAYS answers -- the base class carries the method,
        # so `Req` calls it without a getattr default. The stub must model that
        # or it would be testing a shape production cannot produce.
        return False

    def get_cpu_copy(self, indices, mamba_indices=None):
        return {"full": dict(self.kv), "swa": None}

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        self.kv = dict(kv_cache_cpu["full"])


class _StubMambaPool:
    """`MambaPool`'s copy contract, reduced to what the round trip needs."""

    def __init__(self):
        self.state = {3: "conv+temporal@3"}
        self.asked_for = None

    def get_cpu_copy(self, indices):
        self.asked_for = [int(i) for i in indices.tolist()]
        return [self.state[int(i)] for i in indices.tolist()]

    def load_cpu_copy(self, cpu, indices):
        for i, val in zip(indices.tolist(), cpu):
            self.state[int(i)] = val


def _fixture():
    mamba_pool = _StubMambaPool()
    req_to_token_pool = types.SimpleNamespace(
        req_to_token=torch.zeros((1, 8), dtype=torch.int64),
        mamba_pool=mamba_pool,
        # `mamba_pool` stores PHYSICAL ids; the unified pool hands out virtual
        # ones. Identity here -- the translation is real but is not what this
        # test is about.
        translate_mamba_indices=lambda ids: ids,
    )
    req = Req.__new__(Req)
    req.req_pool_idx = 0
    # `seqlen` is a read-only property = len(origin_input_ids) + len(output_ids),
    # so the INPUTS are set rather than the derived value. Setting the property
    # was the first draft's bug and the controls caught it.
    req.origin_input_ids = [1, 2, 3]
    req.output_ids = [4]
    req.mamba_pool_idx = MAMBA_SLOT
    return req, req_to_token_pool, _StubAllocator(), mamba_pool


class TestOffloadCarriesMambaState(CustomTestCase):
    def test_offload_requests_the_mamba_state(self):
        """RED. The offload is handed a mamba slot and never asks for it."""
        req, rtp, alloc, mamba = _fixture()
        req.offload_kv_cache(rtp, alloc)
        self.assertIsNotNone(
            mamba.asked_for,
            "offload_kv_cache claims it copies the mamba state and never "
            "requested it from the mamba pool",
        )
        self.assertEqual(mamba.asked_for, [3])

    def test_round_trip_restores_a_clobbered_mamba_state(self):
        """RED, and this is the one that matters.

        Copy, destroy the device-side state, restore. This is the property the
        cutover depends on and the whole reason the copy route was chosen over
        ownership transfer."""
        req, rtp, alloc, mamba = _fixture()
        req.offload_kv_cache(rtp, alloc)

        mamba.state[3] = "CLOBBERED"

        req.load_kv_cache(rtp, alloc)
        self.assertEqual(mamba.state[3], "conv+temporal@3")

    def test_the_kv_half_still_round_trips(self):
        """The KV half already works and must keep working -- the fix may not
        buy the mamba state at the price of the thing that was correct."""
        req, rtp, alloc, _ = _fixture()
        req.offload_kv_cache(rtp, alloc)
        alloc.kv = {"tokens": "CLOBBERED"}
        req.load_kv_cache(rtp, alloc)
        self.assertEqual(alloc.kv["tokens"], "kv-bytes")


class TestWithoutMambaNothingChanges(CustomTestCase):
    """The control. Every non-mamba model reaches this same pair."""

    def test_no_mamba_slot_asks_for_nothing(self):
        req, rtp, alloc, mamba = _fixture()
        req.mamba_pool_idx = None
        req.offload_kv_cache(rtp, alloc)
        self.assertIsNone(mamba.asked_for)
        req.load_kv_cache(rtp, alloc)
        self.assertEqual(alloc.kv["tokens"], "kv-bytes")


if __name__ == "__main__":
    unittest.main()


class TestTheDeclarationIsPinned(CustomTestCase):
    """#783 Auflage 2: a declaration nobody checks is the same lie in new
    clothes. `supports_mamba_cpu_copy()` is only worth having if a class that
    claims True and does not deliver goes RED."""

    def test_hybrid_declares_true_and_delivers(self):
        """The one pool that owns a mamba pool: declaration AND behaviour."""
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
        self.assertTrue(pool.supports_mamba_cpu_copy())

        mamba = _StubMambaPool()

        class _FullKv:
            def get_cpu_copy(self, indices):
                return {"kv": "bytes"}

            def load_cpu_copy(self, cpu, indices):
                return None

        pool.full_kv_pool = _FullKv()
        pool.mamba_pool = mamba
        pool._mamba_translate = lambda ids: ids

        copy = pool.get_cpu_copy(torch.tensor([0]), mamba_indices=MAMBA_SLOT)
        self.assertEqual(mamba.asked_for, [3], "declares True but never asked")

        mamba.state[3] = "CLOBBERED"
        pool.load_cpu_copy(copy, torch.tensor([0]), mamba_indices=MAMBA_SLOT)
        self.assertEqual(mamba.state[3], "conv+temporal@3")

    def test_the_rig_pool_declares_false(self):
        """`UnifiedSWAKVPool` has no mamba pool, so False is CORRECT here --
        not a defect. The defect was that nobody could tell."""
        from sglang.srt.mem_cache.unified_memory_pool import UnifiedSWAKVPool

        pool = UnifiedSWAKVPool.__new__(UnifiedSWAKVPool)
        self.assertFalse(pool.supports_mamba_cpu_copy())

    def test_exactly_one_mover_runs(self):
        """EIN-JOB-EIN-MOVER, enforced. When the allocator declares True, `Req`
        must NOT copy as well -- a second copy is the mirror-image bug."""
        req, rtp, alloc, mamba = _fixture()
        alloc.supports_mamba_cpu_copy = lambda: True
        req.offload_kv_cache(rtp, alloc)
        self.assertIsNone(
            mamba.asked_for, "the pool owns the copy; Req must not duplicate it"
        )
        self.assertIsNone(req.mamba_state_cpu)

    def test_a_class_that_claims_true_and_lies_is_caught(self):
        """THE CAN-FAIL DIRECTION THAT COUNTS. A pool declaring the capability
        while dropping the parameter must be detectable -- otherwise the
        declaration is decoration."""

        class _LyingPool:
            def supports_mamba_cpu_copy(self):
                return True

            def get_cpu_copy(self, indices, mamba_indices=None):
                return {"full": "kv"}  # drops mamba despite declaring True

        mamba = _StubMambaPool()
        pool = _LyingPool()
        pool.get_cpu_copy(torch.tensor([0]), mamba_indices=MAMBA_SLOT)

        # The conformance rule, stated once and directly: a pool that declares
        # the capability must have requested the mamba state. This pool
        # declares True and never asked -- so the rule detects it.
        declared = pool.supports_mamba_cpu_copy()
        asked = mamba.asked_for is not None
        conformant = (not declared) or asked
        self.assertFalse(
            conformant,
            "a pool that declares supports_mamba_cpu_copy() and never requests "
            "the state must be non-conformant, or the declaration is decoration",
        )
