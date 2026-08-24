"""#856/W28: a seam whose plan moves nothing needs no waves.

W27-retry measured the residue the no-KV design leaves behind: the plan is
rebuilt EMPTY before the wave loop, and the loop then walks 16 waves that pack
nothing, exchange nothing and write nothing -- ~314 ms of pure backing churn.

SKIPPING THE LOOP IS NOT SIMPLY DELETING IT, which is why this was deferred
rather than done inline. Each wave still released a slice of the source pool's
backing and restored the matching slice of the destination's, and
``finalize_wave`` is the call that marks the destination RESIDENT again. A skip
that omits it leaves the destination answering NO to ``backing_is_resident``,
and every caller of that property is asking "may a kernel touch this pool".

The replacement is the whole-pool swap that already exists on the same object
(``WavedBackingSwap.__call__``): release the source, reclaim, restore the
destination. Here it is pinned to be state-equivalent to the wave loop, and the
source semantics it depends on are pinned against the real
``memory_pool.py`` so this file cannot quietly drift away from them.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import inspect
import unittest

import torch

from sglang.srt.layers.dcp.phase_flip_plan import PhaseFlipTransition
from sglang.test.test_utils import CustomTestCase

PP_TO_TP = "pp_to_tp"
TP_TO_PP = "tp_to_pp"


def _empty(n=0):
    return torch.zeros(n, dtype=torch.int64)


def _plan(
    *,
    send_layers=None,
    send_rows=None,
    recv_layers=None,
    recv_rows=None,
    local_pp=0,
    local_tp=0,
):
    return PhaseFlipTransition(
        rank=0,
        direction=PP_TO_TP,
        n_layers=4,
        layer_map=((0, 1), (2, 3)),
        tp_vector=(1, 1),
        local_layers=(0, 1),
        local_pp_rows=_empty(local_pp),
        local_tp_rows=_empty(local_tp),
        send_layers=send_layers or {},
        send_rows=send_rows or {},
        recv_layers=recv_layers or {},
        recv_rows=recv_rows or {},
        total_slots=0,
    )


class TestMovesNothingSeesEveryLeg(CustomTestCase):
    """A plan has THREE legs. A predicate that checked only the peer exchange
    would skip the loop on a plan that still has a local move, and that drops
    KV silently -- the one outcome worse than the 314 ms."""

    def test_a_fully_empty_plan_moves_nothing(self):
        self.assertTrue(_plan().moves_nothing)

    def test_a_plan_with_a_SEND_does_not(self):
        self.assertFalse(
            _plan(send_layers={1: (0, 1)}, send_rows={1: _empty(3)}).moves_nothing
        )

    def test_a_plan_with_a_RECV_does_not(self):
        self.assertFalse(
            _plan(recv_layers={1: (2, 3)}, recv_rows={1: _empty(5)}).moves_nothing
        )

    def test_a_plan_with_only_a_LOCAL_move_does_not(self):
        # THE LEG A PEER-ONLY PREDICATE MISSES. No send, no recv, and still
        # rows to move: my layers crossed with my own dcp-owned slots.
        self.assertFalse(_plan(local_pp=7, local_tp=7).moves_nothing)

    def test_a_peer_entry_with_zero_ROWS_still_moves_nothing(self):
        # The empty plan is rebuilt with the peer structure intact but no
        # rows; a predicate keyed on dict membership rather than on row counts
        # would refuse to skip and the 314 ms would stay.
        self.assertTrue(
            _plan(
                send_layers={1: (0, 1)},
                send_rows={1: _empty(0)},
                recv_layers={1: (2, 3)},
                recv_rows={1: _empty(0)},
            ).moves_nothing
        )


class _FakePool:
    """A pool with the REAL residency semantics of ``memory_pool.py``.

    Pinned to the source below, so the day those semantics change this file
    fails rather than silently modelling something the tree no longer does.
    """

    def __init__(self, layer_num=4):
        self.layer_num = layer_num
        self._released_layers = set()
        self.calls = []

    def release_backing(self, layers=None):
        self.calls.append(("release", None if layers is None else tuple(layers)))
        if layers is None:
            self._released_layers = set(range(self.layer_num))
        else:
            self._released_layers.update(int(i) for i in layers)
        return 0

    def restore_backing(self, layers=None):
        self.calls.append(("restore", None if layers is None else tuple(layers)))
        if layers is None:
            self._released_layers = set()
        else:
            self._released_layers.difference_update(int(i) for i in layers)

    @property
    def backing_is_resident(self):
        return not self._released_layers


def _swap(pp_pool, tp_pool):
    from sglang.srt.managers.phase_flip_runtime import WavedBackingSwap

    s = object.__new__(WavedBackingSwap)
    s._pp_pool = pp_pool
    s._tp_pool = tp_pool
    s._my_layers = (0, 1, 2, 3)
    s._release_allocator_cache = lambda *a, **k: 0
    s._spill_depth = 0
    s._spill_device = None
    return s


class TestTheFakePoolIsFaithful(CustomTestCase):
    """The model above is only evidence if it models the real thing."""

    def test_residency_is_the_absence_of_released_layers(self):
        from sglang.srt.mem_cache import memory_pool

        src = inspect.getsource(memory_pool.MHATokenToKVPool.backing_is_resident.fget)
        self.assertIn("not self._released_layers", src)

    def test_a_full_restore_clears_every_released_layer(self):
        from sglang.srt.mem_cache import memory_pool

        src = inspect.getsource(memory_pool.MHATokenToKVPool.restore_backing)
        self.assertIn("if layers is None:", src)
        self.assertIn("self._released_layers = set()", src)


class TestTheWholePoolSwapPreservesTheInvariant(CustomTestCase):
    """What the skip must reproduce: destination resident, source released."""

    def test_the_destination_ends_RESIDENT(self):
        for direction in (PP_TO_TP, TP_TO_PP):
            with self.subTest(direction=direction):
                pp, tp = _FakePool(), _FakePool()
                _swap(pp, tp)(direction)
                dst = tp if direction == PP_TO_TP else pp
                src = pp if direction == PP_TO_TP else tp
                self.assertTrue(dst.backing_is_resident)
                self.assertFalse(src.backing_is_resident)

    def test_it_releases_the_source_BEFORE_restoring_the_destination(self):
        # The order is the residency property: restoring first would hold both
        # layouts' pages for the width of the swap, and the corridor floor is a
        # CONTINUOUS minimum, so a peak lasting milliseconds still counts.
        pp, tp = _FakePool(), _FakePool()
        _swap(pp, tp)(PP_TO_TP)
        self.assertEqual(pp.calls[0][0], "release")
        self.assertEqual(tp.calls[0][0], "restore")

    def test_it_reaches_the_same_state_as_the_wave_loop_it_replaces(self):
        waves = ((0,), (1,), (2,), (3,))
        looped_pp, looped_tp = _FakePool(), _FakePool()
        loop_swap = _swap(looped_pp, looped_tp)
        for wave in waves:
            loop_swap.release_wave(PP_TO_TP, wave)
            loop_swap.restore_wave(PP_TO_TP, wave)
            loop_swap.finalize_wave(PP_TO_TP, wave)

        once_pp, once_tp = _FakePool(), _FakePool()
        _swap(once_pp, once_tp)(PP_TO_TP)

        self.assertEqual(
            looped_tp.backing_is_resident, once_tp.backing_is_resident, "destination"
        )
        self.assertTrue(once_tp.backing_is_resident)
        # ...and it got there in ONE restore instead of one per wave.
        self.assertEqual(len([c for c in once_tp.calls if c[0] == "restore"]), 1)
        self.assertGreater(
            len([c for c in looped_tp.calls if c[0] == "restore"]),
            1,
        )

    def test_a_PARTIAL_restore_leaves_the_pool_NON_resident(self):
        # THE CAN-FAIL PARTNER for the whole invariant. If restoring a subset
        # still reported "resident", the assertions above would pass for a
        # skip that never restored anything and the test would be inert.
        pool = _FakePool(layer_num=4)
        pool.release_backing()
        self.assertFalse(pool.backing_is_resident)
        pool.restore_backing([0, 1])
        self.assertFalse(pool.backing_is_resident)
        pool.restore_backing([2, 3])
        self.assertTrue(pool.backing_is_resident)


if __name__ == "__main__":
    unittest.main()
