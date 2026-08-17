"""#719: moving the HiCache pool bindings at the cutover, coherently.

#718 disarmed device-tier I/O in the phase that did not build the binding,
which leaves the tier usable in ONE phase. This is the other half. The whole
risk lives in one property: the pool identity is captured in THREE places, and
a rebind that moves some of them is worse than one that moves none -- every
call still succeeds, against different memory, with no error anywhere.

So the tests below are built around what CANNOT be checked by inspection:

* ``test_a_reader_left_behind_is_caught`` -- the failure this design exists
  for, planted deliberately: one reader keeps the old generation and the
  coherence check refuses. Without generations this state is invisible.
* ``test_shape_mismatch_is_refused`` -- the trap a naive pointer swap walks
  into. A host pool is ALLOCATED FROM a device pool, so pairing the TP device
  pool with the PP host pool gives matching row ids at mismatched widths, and
  the copy would RUN. Refused instead.
* ``test_disarm_lifts_only_after_a_coherent_rebind`` -- the end-to-end claim,
  expressed where it is decidable hermetically: device-tier I/O is refused in
  the TP phase (today's state), and permitted there once the binding has
  coherently moved. That is the "post-flip device-tier hit" requirement at the
  seam that decides it.
* ``test_unarmed_is_byte_identical`` -- with the flag off nothing moves, no
  generation advances, and the #718 predicate answers exactly as before.

Not covered here, and named rather than mocked away: the boot-time allocation
of the second phase's host pool. It cannot be derived from the device pool at
cutover (it is allocated from it, and host RAM is the binding constraint on
this box), so ``phase_pools_for`` refuses without it and this suite supplies
one explicitly -- which is exactly what a boot that built one would do.
"""

import unittest

from sglang.srt.mem_cache import hicache_phase_binding as binding
from sglang.srt.mem_cache import hicache_phase_guard as guard
from sglang.test.test_utils import CustomTestCase


class _Pool:
    def __init__(self, layer_num, tag):
        self.layer_num = layer_num
        self.tag = tag


class _HostPool:
    def __init__(self, layer_num, tag):
        self.layer_num = layer_num
        self.tag = tag


class _Controller:
    def __init__(self, device_pool, host_pool, allocator):
        self.mem_pool_device = device_pool
        self.mem_pool_device_hybrid = device_pool
        self.mem_pool_device_allocator = allocator
        self.mem_pool_host = host_pool


class _Tree:
    def __init__(self, controller, device_pool, host_pool):
        self.cache_controller = controller
        self.hybrid_kv_cache = device_pool
        self.kvcache = device_pool
        self.token_to_kv_pool_host = host_pool


class _Sched:
    def __init__(self, tree, allocator, server_args):
        self.tree_cache = tree
        self.token_to_kv_pool_allocator = allocator
        self.server_args = server_args


class _Args:
    def __init__(self, armed):
        self.phase_flip_rebind_hicache = armed


def _world(armed=True):
    pp_dev, pp_host = _Pool(7, "pp"), _HostPool(7, "pp")
    alloc = object()
    controller = _Controller(pp_dev, pp_host, alloc)
    tree = _Tree(controller, pp_dev, pp_host)
    sched = _Sched(tree, alloc, _Args(armed))
    return sched, binding.readers_of(sched)


def _tp_pools():
    return binding.PhasePools(
        phase="tp",
        device_pool=_Pool(16, "tp"),
        host_pool=_HostPool(16, "tp"),
        allocator=object(),
    )


class TestRebindCoherence(CustomTestCase):
    def setUp(self):
        binding.binding_state().reset()
        guard.reset_warnings()
        self._routing = False
        self._orig = guard.flip_routing_active
        guard.flip_routing_active = lambda: self._routing
        self.addCleanup(setattr, guard, "flip_routing_active", self._orig)
        self.addCleanup(binding.binding_state().reset)
        self.addCleanup(guard.reset_warnings)

    # -- the default ---------------------------------------------------------

    def test_unarmed_is_byte_identical(self):
        sched, readers = _world(armed=False)
        self.assertIsNone(binding.rebind_for_cutover(sched, "tp"))
        self.assertEqual(binding.bound_phase(), "pp")
        self.assertEqual(binding.binding_state().generation, 0)
        self._routing = True
        self.assertTrue(guard.device_tier_disarmed("write"))

    # -- the three readers ---------------------------------------------------

    def test_all_three_readers_move_together(self):
        sched, readers = _world()
        self.assertEqual(
            sorted(readers), ["cache_controller", "scheduler", "tree_cache"]
        )
        generation = binding.rebind(readers, _tp_pools())
        self.assertEqual(binding.coherence_check(readers), generation)
        self.assertEqual(sched.tree_cache.cache_controller.mem_pool_device.tag, "tp")
        self.assertEqual(sched.tree_cache.hybrid_kv_cache.tag, "tp")
        self.assertEqual(sched.tree_cache.cache_controller.mem_pool_host.tag, "tp")
        self.assertIs(
            sched.token_to_kv_pool_allocator,
            sched.tree_cache.cache_controller.mem_pool_device_allocator,
        )

    def test_a_reader_left_behind_is_caught(self):
        """The invisible failure, planted. Without generations, a controller
        still naming the PP pool while its peers name the TP one produces no
        error at all -- every call succeeds, against different memory."""
        sched, readers = _world()
        binding.rebind(readers, _tp_pools())
        sched.tree_cache.cache_controller.hicache_binding_generation = 0
        with self.assertRaises(binding.RebindIncoherent) as cm:
            binding.coherence_check(readers)
        self.assertIn("cache_controller", str(cm.exception))

    def test_an_absent_reader_refuses_the_whole_rebind(self):
        sched, _ = _world()
        sched.tree_cache = None
        with self.assertRaises(binding.RebindRefused):
            binding.rebind(binding.readers_of(sched), _tp_pools())

    def test_a_failing_reader_leaves_the_set_unusable_not_half_moved(self):
        """If one stamp fails the set is split; the state must not look
        coherent afterwards, or the tier would re-arm onto a torn binding."""

        class _Hostile:
            hicache_binding_generation = 0

            def __setattr__(self, name, value):
                raise RuntimeError("cannot stamp")

        sched, readers = _world()
        readers["tree_cache"] = _Hostile()
        with self.assertRaises(binding.RebindIncoherent):
            binding.rebind(readers, _tp_pools())
        with self.assertRaises(binding.RebindIncoherent):
            binding.coherence_check(readers)

    # -- the trap a pointer swap walks into ----------------------------------

    def test_shape_mismatch_is_refused(self):
        mixed = binding.PhasePools(
            phase="tp",
            device_pool=_Pool(16, "tp"),
            host_pool=_HostPool(7, "pp"),  # the other phase's host pool
            allocator=object(),
        )
        with self.assertRaises(binding.RebindRefused) as cm:
            binding.check_shapes(mixed)
        self.assertIn("mismatched widths", str(cm.exception))

    def test_unmeasurable_shapes_are_refused(self):
        opaque = binding.PhasePools(
            phase="tp", device_pool=object(), host_pool=object(), allocator=object()
        )
        with self.assertRaises(binding.RebindRefused):
            binding.check_shapes(opaque)

    def test_a_phase_without_a_host_pool_is_refused_with_the_reason(self):
        """Today's real state: no second host pool exists, so the rebind cannot
        arm -- and says why, instead of proceeding onto the wrong one."""
        sched, _ = _world()
        with self.assertRaises(binding.RebindRefused) as cm:
            binding.phase_pools_for(sched, "tp")
        self.assertIn("host pool", str(cm.exception))

    # -- the payoff ----------------------------------------------------------

    def test_disarm_lifts_only_after_a_coherent_rebind(self):
        sched, readers = _world()
        self._routing = True
        # Today: TP phase, binding still on PP -> device tier refused (#718).
        self.assertTrue(guard.device_tier_disarmed("write"))
        binding.rebind(readers, _tp_pools())
        binding.coherence_check(readers)
        # Rebound coherently: the active phase IS the bound phase.
        self.assertFalse(guard.device_tier_disarmed("write"))
        # And flipping back re-disarms until the return rebind runs.
        self._routing = False
        self.assertTrue(guard.device_tier_disarmed("load"))

    def test_the_return_leg_rebinds_back(self):
        sched, readers = _world()
        binding.rebind(readers, _tp_pools())
        pp = binding.PhasePools(
            phase="pp",
            device_pool=_Pool(7, "pp"),
            host_pool=_HostPool(7, "pp"),
            allocator=object(),
        )
        gen = binding.rebind(readers, pp)
        self.assertEqual(binding.coherence_check(readers), gen)
        self.assertEqual(binding.bound_phase(), "pp")
        self.assertFalse(guard.device_tier_disarmed("write"))

    def test_armed_cutover_rebinds_when_the_boot_supplied_a_host_pool(self):
        sched, readers = _world()
        sched.phase_flip_stacks = type(
            "_S",
            (),
            {
                "tp_worker": type(
                    "_W",
                    (),
                    {
                        "model_runner": type(
                            "_R",
                            (),
                            {
                                "token_to_kv_pool": _Pool(16, "tp"),
                                "token_to_kv_pool_allocator": object(),
                            },
                        )()
                    },
                )()
            },
        )()
        sched.phase_flip_host_pools = {"tp": _HostPool(16, "tp")}
        generation = binding.rebind_for_cutover(sched, "tp")
        self.assertEqual(generation, 1)
        self.assertEqual(binding.bound_phase(), "tp")
        self.assertEqual(binding.coherence_check(readers), 1)


if __name__ == "__main__":
    unittest.main()
