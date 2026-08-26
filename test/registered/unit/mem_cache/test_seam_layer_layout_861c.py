"""#861c: a per-layer copy carried across a cutover, indexed with the DESTINATION's geometry.

SPECIMEN (W40, boot_w40_857strict_0826_0516.log, 05:20:50, all three ranks,
24 s after health 200):

    prepare_for_extend -> restore_seam_state (schedule_batch.py:2107)
      -> Req.load_kv_cache (:1778) -> allocator/paged.py:326 load_cpu_copy
      -> memory_pool.py:4476 -> memory_pool.py:3370
    IndexError: list index out of range

    05:20:41  rid=4b15fe... admitted in PP, prefilled there
    05:20:44  FLIP EXTENT PROBE takes the seam copy, extent=13 rows, IN PP
    05:20:50  PHASE-FLIP DONE pp_to_tp (epoch 7)   -- now in TP
    05:20:50  SEAM RESTORE ATTEMPT ... entering load_kv_cache -> IndexError

THE MECHANISM, AND IT IS AN AXIS THE TREE HAS NOT GUARDED YET.
`get_cpu_copy` (memory_pool.py:3341) builds a list with `self.layer_num`
entries -- the layer count of the layout AT COPY TIME. `load_cpu_copy` (:3366)
walks `range(self.layer_num)` -- the layer count of the layout AT RESTORE TIME.
Across the phase flip these are different objects: the PP stack's pool is
`scheduler.tp_worker.model_runner.token_to_kv_pool` and the TP stack's is
`stacks.tp_worker.model_runner.token_to_kv_pool` (phase_flip_runtime.py:3374).
With `--pp-stage-ratio 32,18,14` the PP pool on a rank holds only that stage's
layers; the TP pool holds all of them.

  PP copy -> TP restore : the loop runs past the end of the shorter saved list.
                          IndexError, dead scheduler. This is the specimen.
  TP copy -> PP restore : the loop runs FEWER iterations than the copy has
                          entries. No error at all. Copy entry i is GLOBAL
                          layer `copy.start_layer + i`; it is written into
                          destination LOCAL slot i, which is GLOBAL layer
                          `dest.start_layer + i`. On PP1 (start_layer=32) that
                          restores global layers 0..17 into global layers
                          32..49 -- WRONG-LAYER KV under a prefix the tree
                          reports as restored. No crash, no log.

The silent half is the dangerous half, and it is the reason the fix cannot be
`for layer_id in range(len(kv_cache_cpu))`: that removes the IndexError and
leaves the wrong-layer write, i.e. converts the loud direction into the quiet
one. Both directions are asserted below.

WHY REFUSE RATHER THAN REMAP. A remap needs the copy to identify its entries as
global layers -- which the layout identity added here does provide. It is still
refused, because a remap would be correct only if the copy COVERED every layer
the destination needs, and rank-locally under PP it structurally cannot: PP1
holds 18 of 64 layers, so 46 layers of the destination would keep whatever bytes
those rows carried. A prefix that is right in 18 layers and stale in 46 is a
wrong answer wearing the shape of a restore. (The shard geometry differs too --
PP holds all KV heads of its stage, TP holds a head shard of every layer -- so
even the overlapping entries are not interchangeable.) A refusal costs one
recompute.

WHAT THIS FILE PINS, two independent guards because they answer different
questions:

  1. POOL LEVEL (`check_cpu_copy_layers`): a `load_cpu_copy` must not trust its
     own `layer_num` for a list it did not build. This catches BOTH directions
     of the count mismatch and is a backstop for every caller, present and
     future. It refuses; it does not keep the process alive by itself.
  2. CALLER LEVEL (`restore_seam_state`): the copy records the LAYOUT IDENTITY
     it was taken from, and a restore into a different layout is refused,
     counted and named before `load_kv_cache` is entered. This is what keeps the
     scheduler alive, and it is the only one of the two that also catches an
     equal-COUNT layout change (two PP stages of the same size, or a stage
     ratio permutation) -- where the count guard is structurally inert.

Hermetic: real pool methods bound to stubs with CPU buffers, no CUDA. Follows
the idiom of test_seam_row_bounds_783b.py deliberately -- running the tree's own
methods rather than a reimplementation, so the test cannot encode the defect's
own assumption.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.memory_pool import MambaPool, MHATokenToKVPool
from sglang.test.test_utils import CustomTestCase

ROWS = 8
DIM = 2

# The rig's shape, reduced: a 6-layer model cut 4 / 2 across two PP stages.
TP_LAYERS = 6
PP0_LAYERS = 4
PP1_LAYERS = 2
PP1_START = 4


class _StubKVPool:
    """The REAL `MHATokenToKVPool` copy methods over CPU buffers.

    `layer_num` and `start_layer` are the two numbers under test, so they are
    constructor arguments rather than class constants: one instance stands for
    the PP-stage pool and another for the TP pool, and they are DIFFERENT
    OBJECTS, exactly as the two phase stacks are.
    """

    use_hnd = False
    cpu_offloading_chunk_size = 4

    def __init__(self, layer_num, start_layer=0, fill=0.0):
        self.size = ROWS
        self.layer_num = layer_num
        self.start_layer = start_layer
        self.end_layer = start_layer + layer_num - 1
        # Each layer is filled with a value that names the GLOBAL layer it
        # holds, so a wrong-layer write is visible as a number and not only as
        # an absence.
        self.k_buffer = [
            torch.full((ROWS, DIM), float(start_layer + i) + fill)
            for i in range(layer_num)
        ]
        self.v_buffer = [
            torch.full((ROWS, DIM), float(start_layer + i) + fill)
            for i in range(layer_num)
        ]
        self.get_cpu_copy = types.MethodType(MHATokenToKVPool.get_cpu_copy, self)
        self.load_cpu_copy = types.MethodType(MHATokenToKVPool.load_cpu_copy, self)
        _bind_layout(self, MHATokenToKVPool)

    def _committed_row_bound(self):
        """#913: an eager stub pool has no VMM arena, so it cannot state a
        backing -- None, never 0. Production pools answer this from
        ``KVCache._committed_row_bound`` or the ``MHATokenToKVPool`` override;
        modelling it here keeps the double from diverging from the real pool,
        which is the W29 failure ("the suite's own double had the attribute and
        not the method, exactly backwards from production").
        """
        return None


def _bind_layout(stub, cls):
    """Bind `cpu_copy_layout` only if the tree has it.

    DELIBERATE. Binding it unconditionally would make the CONTROL cases below
    fail at construction before the fix exists, and a control that is red for
    the same reason as the reds it is meant to calibrate proves nothing."""
    fn = getattr(cls, "cpu_copy_layout", None)
    if fn is not None:
        stub.cpu_copy_layout = types.MethodType(fn, stub)


def _rows():
    return torch.tensor([0, 1, 2], dtype=torch.int64)


def _copy_from(pool):
    with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
        return pool.get_cpu_copy(_rows())


def _restore_into(pool, saved):
    with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
        pool.load_cpu_copy(saved, _rows())


class TestTheStubReproducesTheRealShapes(CustomTestCase):
    """CONTROL. Passes with and without the fix. Without these facts the reds
    below say nothing about the real pool."""

    def test_a_copy_has_exactly_the_source_layer_count(self):
        saved = _copy_from(_StubKVPool(PP1_LAYERS, PP1_START))
        self.assertEqual(len(saved), PP1_LAYERS)

    def test_a_same_layout_restore_still_round_trips(self):
        """THE PATH THAT MUST KEEP WORKING. The guard is worth nothing if it
        also refuses the ordinary same-layout retract/resume."""
        src = _StubKVPool(TP_LAYERS, 0, fill=0.0)
        dst = _StubKVPool(TP_LAYERS, 0, fill=100.0)
        saved = _copy_from(src)
        _restore_into(dst, saved)
        for i in range(TP_LAYERS):
            self.assertTrue(
                torch.equal(dst.k_buffer[i][_rows()], src.k_buffer[i][_rows()]),
                f"same-layout restore lost layer {i}",
            )


class TestThePoolRefusesACrossLayoutRestore(CustomTestCase):
    """RED before the fix, both directions."""

    def test_pp_copy_into_tp_pool_is_refused_not_indexed(self):
        """THE SPECIMEN. 2 saved entries, a destination that walks 6.

        Before the fix this raised `IndexError: list index out of range` at
        memory_pool.py:3370 and took the scheduler with it -- and it had
        already written two wrong-layer entries before it got there.
        """
        saved = _copy_from(_StubKVPool(PP1_LAYERS, PP1_START))
        tp = _StubKVPool(TP_LAYERS, 0, fill=100.0)
        with self.assertRaises(ValueError) as caught:
            _restore_into(tp, saved)
        msg = str(caught.exception)
        self.assertIn(str(PP1_LAYERS), msg)
        self.assertIn(str(TP_LAYERS), msg)

    def test_the_refusal_happens_before_the_first_store(self):
        """A guard that fires halfway through is a louder corruption -- the rule
        `check_cpu_copy_rows` already states and this one inherits."""
        saved = _copy_from(_StubKVPool(PP1_LAYERS, PP1_START))
        tp = _StubKVPool(TP_LAYERS, 0, fill=100.0)
        before = [b.clone() for b in tp.k_buffer]
        with self.assertRaises(ValueError):
            _restore_into(tp, saved)
        for i, b in enumerate(before):
            self.assertTrue(
                torch.equal(tp.k_buffer[i], b),
                f"layer {i} was written before the refusal",
            )

    def test_tp_copy_into_pp_pool_is_refused_and_NOT_silently_applied(self):
        """THE MIRROR, and the half that never crashed.

        6 saved entries, a destination that walks 2. Before the fix this ran to
        completion and wrote global layers 0 and 1 into global layers 4 and 5.
        The assertion is on the BUFFER, not only on the exception: a guard that
        raised after writing would pass an exception-only test.
        """
        saved = _copy_from(_StubKVPool(TP_LAYERS, 0))
        pp1 = _StubKVPool(PP1_LAYERS, PP1_START, fill=0.0)
        untouched = [b.clone() for b in pp1.k_buffer]
        with self.assertRaises(ValueError):
            _restore_into(pp1, saved)
        for i, b in enumerate(untouched):
            self.assertTrue(
                torch.equal(pp1.k_buffer[i], b),
                f"PP local slot {i} (global {PP1_START + i}) took foreign KV",
            )

    def test_the_unguarded_mirror_really_was_silent(self):
        """CONTROL, and the reason `range(len(kv_cache_cpu))` is not the fix.

        Pinned as a fact about the OLD loop rather than about the tree, so it
        keeps saying what it says after the fix: walking the destination's
        layer count over a longer copy raises nothing and writes the wrong
        global layers.
        """
        saved = _copy_from(_StubKVPool(TP_LAYERS, 0))
        pp1 = _StubKVPool(PP1_LAYERS, PP1_START, fill=0.0)
        rows = _rows()
        for layer_id in range(pp1.layer_num):  # the pre-fix loop, verbatim
            pp1.k_buffer[layer_id][rows] = saved[layer_id][0][0]
        # local slot 0 is GLOBAL layer 4, and it now holds GLOBAL layer 0.
        self.assertTrue(
            torch.all(pp1.k_buffer[0][rows] == 0.0),
            "the mirror direction was expected to be silent and wrong",
        )


class TestTheLayoutIdentityIsCarried(CustomTestCase):
    """The count guard is inert when the counts happen to agree. Two PP stages
    of equal size, or a permuted stage ratio, are exactly that case."""

    def test_a_pool_states_its_layer_geometry(self):
        layout = _StubKVPool(PP1_LAYERS, PP1_START).cpu_copy_layout()
        self.assertEqual(layout.layer_num, PP1_LAYERS)
        self.assertEqual(layout.start_layer, PP1_START)

    def test_equal_counts_at_different_offsets_are_not_the_same_layout(self):
        a = _StubKVPool(3, 0).cpu_copy_layout()
        b = _StubKVPool(3, 3).cpu_copy_layout()
        self.assertNotEqual(
            a,
            b,
            "two 3-layer stages at different offsets compared equal -- the "
            "identity is then blind to exactly the case the count guard "
            "cannot see",
        )

    def test_the_same_layout_compares_equal(self):
        self.assertEqual(
            _StubKVPool(3, 3).cpu_copy_layout(),
            _StubKVPool(3, 3).cpu_copy_layout(),
        )


SLOTS = 6


class _StubMambaPool:
    """The REAL `MambaPool` copy methods. Same class of defect, different axis:
    `conv` is a LIST whose length is the rank's mamba-layer count, and
    `load_cpu_copy` enumerates the DESTINATION list while indexing the SAVED
    one (`conv_cpu[i]`)."""

    def __init__(self, layers):
        self.mamba_cache = types.SimpleNamespace(
            conv=[torch.zeros((1, SLOTS, 3)) for _ in range(layers)],
            temporal=torch.zeros((layers, SLOTS, 3)),
        )
        self.get_cpu_copy = types.MethodType(MambaPool.get_cpu_copy, self)
        self.load_cpu_copy = types.MethodType(MambaPool.load_cpu_copy, self)
        _bind_layout(self, MambaPool)


class TestTheMambaHalfCarriesTheSameClass(CustomTestCase):
    def test_a_shorter_mamba_copy_is_refused_not_indexed(self):
        with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
            saved = _StubMambaPool(2).get_cpu_copy(torch.tensor([0], dtype=torch.int64))
            dst = _StubMambaPool(5)
            with self.assertRaises(ValueError):
                dst.load_cpu_copy(saved, torch.tensor([0], dtype=torch.int64))

    def test_a_longer_mamba_copy_is_refused_not_truncated(self):
        """The silent direction on the mamba axis: 5 saved entries, 2 walked."""
        with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
            saved = _StubMambaPool(5).get_cpu_copy(torch.tensor([0], dtype=torch.int64))
            dst = _StubMambaPool(2)
            with self.assertRaises(ValueError):
                dst.load_cpu_copy(saved, torch.tensor([0], dtype=torch.int64))

    def test_a_same_layout_mamba_restore_still_round_trips(self):
        with patch("sglang.srt.mem_cache.memory_pool.current_platform"):
            src = _StubMambaPool(3)
            for c in src.mamba_cache.conv:
                c.fill_(7.0)
            src.mamba_cache.temporal.fill_(7.0)
            saved = src.get_cpu_copy(torch.tensor([0], dtype=torch.int64))
            dst = _StubMambaPool(3)
            dst.load_cpu_copy(saved, torch.tensor([0], dtype=torch.int64))
            for c in dst.mamba_cache.conv:
                self.assertTrue(torch.all(c[:, 0] == 7.0))


if __name__ == "__main__":
    unittest.main()
