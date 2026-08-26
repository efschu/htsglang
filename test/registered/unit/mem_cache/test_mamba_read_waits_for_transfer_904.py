"""#904 (a): the mamba state read must wait for the transfer that fills it.

USER HYPOTHESIS, VERBATIM: "either the kv is read before it is loaded, or
something invalidates the kv after it was loaded and before it is read.
same for mamba." This file is the READ-BEFORE-LOAD half, mamba side.

THE ORDERING, TRACED RATHER THAN ASSUMED
----------------------------------------
Under ``--enable-hierarchical-cache`` on a hybrid SSM model the live tree
cache is ``UnifiedRadixCache`` (``registry.py:107-110`` states outright that
``HiMambaRadixCache`` has no construction site), and its stack is built by
``hybrid_pool_assembler.build_hybrid_mamba_stack``:

  * ``full_layer_mapping = hybrid_kv.full_attention_layer_id_mapping`` and
    ``mamba_layer_mapping = req_to_token_pool.mamba_map`` -- both keyed by
    GLOBAL layer id (``hybrid_pool_assembler.py:1556-1558``).
  * ``transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping)``
    (``:858``), i.e. the layer count of the WHOLE model.
  * ``HybridCacheController.start_loading`` then runs
    ``for i in range(self.layer_num)`` over exactly that frame and calls
    ``producer_event.complete(i)`` after each step
    (``hybrid_cache_controller.py:636-660``). ``_make_layer_mapper``
    (``hybrid_pool_assembler.py:43-52``) makes step ``i`` the step that
    moves GLOBAL layer ``i`` -- so THE TRANSFER INDEX IS THE GLOBAL LAYER ID,
    for the mamba entry exactly as for the KV entry.

The copies ride ``load_stream``. The only join back to the compute stream is
``LayerLoadingEvent.wait`` -> ``current_stream().wait_event(load_events[idx])``
(``cache_controller.py:61-62``), reached solely through
``layer_transfer_counter.wait_until``. The KV pools call it; the mamba pool
called NOTHING, so a GDN layer whose global id precedes the first attention
layer read in this forward had no join at all before
``gdn_backend.py:492`` read its state.

WHY THE OLD JUSTIFICATION DOES NOT HOLD
---------------------------------------
The #752 note claimed mamba states "move as WHOLE blobs ... and are complete
before the batch launches". They do move as whole blobs -- and they move on
the SAME asynchronous per-layer loop as the KV, enqueued at their own global
layer index, with nothing between ``ready_to_load_host_cache()``
(``scheduler.py:8712``) and the forward that waits for them.
``check_hicache_events`` (``scheduler.py:8024``) drains the PREVIOUS batch's
acks and runs BEFORE the enqueue in the same function, so it is not that
guarantee either.

#752's actual finding stands and is re-pinned below: the mamba read must
never wait on ``local_slot(layer_id)``, the KV frame. That consult was the
AttributeError that killed the first GDN forward. The fix is not "no wait",
it is "wait in the MAMBA transfer frame".

Same shape as ``09c4e49bb7`` (moe-offload write-after-read on the scratch
region): a side stream whose join to compute exists in one direction only.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _code_lines(fn):
    """Source of ``fn`` with comment lines removed.

    The guard below is about what the code CALLS, not about what the comment
    is allowed to name -- and the comment must be free to name `local_slot`,
    because explaining why it is the wrong frame is the whole point of it.
    """
    import inspect

    return "\n".join(
        line
        for line in inspect.getsource(fn).splitlines()
        if not line.lstrip().startswith("#")
    )


class _CounterSpy:
    """Stands in for LayerDoneCounter; records the thresholds asked for."""

    def __init__(self, num_layers: int = 48):
        self.num_layers = num_layers
        self.waits = []

    def wait_until(self, threshold):
        self.waits.append(threshold)


class _MambaPoolStub:
    def __init__(self):
        self.asked = []

    def mamba2_layer_cache(self, slot):
        self.asked.append(slot)
        return f"state-for-slot-{slot}"


def _hybrid_pool(counter, frame=None):
    """A HybridReqToTokenPool with the mamba half wired and nothing else.

    Layer layout mirrors a GDN checkpoint: global layers 0..3, attention at
    3, linear (mamba) at 0, 1, 2 -- so the FIRST layer read in a forward is a
    mamba layer, which is the case that has no incidental FIFO cover from a
    later attention wait.
    """
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    pool = object.__new__(HybridReqToTokenPool)
    pool.mamba_map = {0: 0, 1: 1, 2: 2}
    pool.mamba_pool = _MambaPoolStub()
    pool.layer_transfer_counter = None
    pool.start_layer = 0
    pool._mamba_transfer_frame = None
    if counter is not None:
        pool.register_layer_transfer_counter(counter, mamba_transfer_frame=frame)
    return pool


class TestMambaReadJoinsItsTransfer(CustomTestCase):
    def test_read_waits_on_the_global_transfer_index(self):
        """RED-FIRST for #904(a).

        With the hybrid controller's frame registered, reading global mamba
        layer 2 must join the load stream at transfer step 2 -- the step at
        which THAT layer's blob was copied.
        """
        spy = _CounterSpy(num_layers=4)
        pool = _hybrid_pool(spy, frame=4)
        out = pool.mamba2_layer_cache(2)
        self.assertEqual(out, "state-for-slot-2")
        self.assertEqual(
            spy.waits,
            [2],
            "the mamba read did not join the transfer that fills it; the "
            "recurrent step can read state the H2D copy has not landed yet "
            "(#904 read-before-load)",
        )

    def test_first_layer_of_the_model_is_covered(self):
        """Global layer 0 is a mamba layer on a GDN checkpoint. It has no
        earlier attention wait to hide behind, so it is the specimen the
        incidental-FIFO argument cannot cover."""
        spy = _CounterSpy(num_layers=4)
        pool = _hybrid_pool(spy, frame=4)
        pool.mamba2_layer_cache(0)
        self.assertEqual(spy.waits, [0])

    def test_the_wait_is_not_the_kv_local_slot(self):
        """#752 REGRESSION GUARD, kept live.

        The crash was ``wait_until(local_slot(layer_id))`` -- the KV indexing
        half, which a req-to-token pool does not have, and which on a lineage
        where it does not raise waits on a slot mamba transfers never
        advance. The threshold must be the GLOBAL layer id, and the pool must
        not grow a ``local_slot`` consult.
        """
        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

        spy = _CounterSpy(num_layers=48)
        pool = _hybrid_pool(spy, frame=48)
        pool.mamba_map = {30: 0, 31: 1}
        pool.start_layer = 24
        pool.mamba2_layer_cache(31)
        self.assertEqual(
            spy.waits,
            [31],
            "the mamba wait must be keyed by the GLOBAL layer id (the "
            "transfer frame), never by a mamba slot or a KV local slot",
        )
        self.assertNotIn(
            "local_slot",
            _code_lines(HybridReqToTokenPool.mamba2_layer_cache),
            "#752: the KV frame is not ours",
        )


class TestTheFrameGatesTheWait(CustomTestCase):
    """The can-fail direction. Turning the wait into an unconditional
    ``wait_until(layer_id)`` reds these: a KV-only controller's counter is
    sized to the ATTENTION layer count, so a global mamba id can be out of
    range, and a pool with no hicache stack must stay byte-identical."""

    def test_no_counter_no_wait(self):
        pool = _hybrid_pool(None)
        self.assertEqual(pool.mamba2_layer_cache(1), "state-for-slot-1")

    def test_counter_without_a_frame_does_not_wait(self):
        """A counter registered by a stack that does NOT carry the mamba pool
        (KV-only controller) has no step that moves mamba state. Waiting on
        it would block on another pool's progress, or index past its end."""
        spy = _CounterSpy(num_layers=12)
        pool = _hybrid_pool(spy, frame=None)
        pool.mamba2_layer_cache(1)
        self.assertEqual(spy.waits, [])

    def test_layer_outside_the_frame_does_not_wait(self):
        spy = _CounterSpy(num_layers=4)
        pool = _hybrid_pool(spy, frame=4)
        pool.mamba_map = {9: 0}
        pool.mamba2_layer_cache(9)
        self.assertEqual(
            spy.waits,
            [],
            "a layer outside the registered transfer frame has no step in "
            "this counter; waiting would index past its events",
        )


class TestTheTwoIndexSpacesAreOneRegistration(CustomTestCase):
    """The desk half of the discriminator: the frame the pool waits in and
    the frame the controller counts in must come from the SAME number.

    They desynced silently before (#753/#756 gapped ownership), and a
    desynced frame is indistinguishable at runtime from a correct one until
    the wrong bytes are read. So the wiring site is pinned, not the value.
    """

    def test_assembler_passes_the_controller_frame_to_the_pool(self):
        import inspect

        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler

        src = inspect.getsource(hybrid_pool_assembler)
        self.assertIn(
            "mamba_transfer_frame=",
            src,
            "the assembler must hand the pool the controller's transfer "
            "frame; without it the mamba wait is silently inert (#904)",
        )

    def test_live_attach_path_is_the_one_that_carries_it(self):
        """``registry.py`` routes hybrid-SSM + hicache to UnifiedRadixCache,
        so ``_apply_stack_result`` -- not the HiMambaRadixCache entrypoint --
        is the live wiring. A fix installed only on the dead path is the
        W31/W32/W33 shape."""
        import inspect

        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler

        src = inspect.getsource(hybrid_pool_assembler._apply_stack_result)
        self.assertIn("mamba_transfer_frame=", src)


if __name__ == "__main__":
    unittest.main()
