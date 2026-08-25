"""#847/W33: the WRITER for `scheduler.phase_flip_host_pools`.

MECHANISMUS VORHANDEN, AKTUATOR FEHLT -- the whole #718 rebind chain already
existed and was already wired: `rebind_for_cutover` is called at the cutover,
the #719 generation stamp and `coherence_check` are built, and
`phase_pools_for` knows exactly what it wants. It wanted
`scheduler.phase_flip_host_pools[phase]`, and across the entire tree that name
appeared ONLY in its own docstring and its own refusal message. Nothing ever
wrote it.

W32 measured the consequence end to end (SPECIMEN_w32_policy_purity_copy_
pulls_back_to_pp.log): no host pool -> RebindRefused -> the rebind never arms
-> `bound_phase()` stays "pp" -> `device_tier_disarmed("load")` is True for the
whole TP phase -> `HiCacheController.load()` returns None -> zero tokens reach
the device. The one transport prefill logged `#cached-token: 0` on what should
have been a perfect disk hit, and the specimen carries 6 `#718
hicache-phase-guard` warnings beside `phase_flip_rebind_hicache=False`.

REFUSAL CONVERSION, NOT GUARD DELETION (#847). The guard still refuses for a
genuinely absent or mis-shaped pool -- that is the can-fail direction below.
What changes is that the pool can now EXIST and is PRICED: a small staging pin
(#810) whose bytes are a named HOST-LEDGER post (#721), where the post shrinks
and the floor never does.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import types
import unittest

from sglang.srt.managers.phase_flip_boot import (
    PHASE_FLIP_STAGING_CHUNKS,
    _staging_pin_gib,
    build_phase_flip_host_pools,
)
from sglang.test.test_utils import CustomTestCase


class _HostPool:
    """Records the arguments the real host-pool constructors take."""

    last = None

    def __init__(
        self, device_pool, ratio, size_gb, page_size, layout, allocator_type=None
    ):
        self.device_pool = device_pool
        self.ratio = ratio
        self.size_gb = size_gb
        self.page_size = page_size
        self.layout = layout
        self.allocator_type = allocator_type
        _HostPool.last = self


class _DevicePool:
    layer_num = 16

    def __init__(self, cell=8192):
        self._cell = cell

    def get_kv_size_per_token(self):
        return self._cell


def _sched(*, rebind=True, host=True, tp_pool=True, cell=8192):
    sa = types.SimpleNamespace(
        phase_flip_rebind_hicache=rebind,
        chunked_prefill_size=4096,
        max_running_requests=8,
        page_size=1,
        hicache_mem_layout="layer_first",
        hicache_storage_backend="file",
    )
    tree = types.SimpleNamespace(
        token_to_kv_pool_host=_HostPool(None, 0, 1, 1, "layer_first") if host else None
    )
    stacks = types.SimpleNamespace(
        tp_worker=types.SimpleNamespace(
            model_runner=types.SimpleNamespace(
                token_to_kv_pool=_DevicePool(cell) if tp_pool else None
            )
        )
    )
    return types.SimpleNamespace(
        server_args=sa, tree_cache=tree, phase_flip_stacks=stacks
    )


class TestTheDefaultBootIsUntouched(CustomTestCase):
    def test_the_flag_off_allocates_nothing(self):
        # Every boot that does not ask for the rebind must be byte-identical.
        self.assertEqual(build_phase_flip_host_pools(_sched(rebind=False)), {})


class TestTheWriterBuildsBothPhases(CustomTestCase):
    def _patched(self, sched):
        """Assembly needs real pools; patch the NAMED primitives instead.

        Patching the three builders the writer is required to use is itself
        the assertion that it uses them -- a writer that went back to cloning
        `type(pp_host)` would ignore these patches and fail here.
        """
        import unittest.mock as mock

        m = mock.patch.multiple(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler",
            build_kv_host_pool=mock.DEFAULT,
            build_pool_entry=mock.DEFAULT,
        )
        with (
            m,
            mock.patch("sglang.srt.mem_cache.memory_pool_host.HostPoolGroup") as grp,
        ):
            grp.side_effect = lambda entries: types.SimpleNamespace(
                entries=entries,
                device_pool=sched.phase_flip_stacks.tp_worker.model_runner.token_to_kv_pool,
            )
            return build_phase_flip_host_pools(sched)

    def test_both_phases_are_registered(self):
        pools = self._patched(_sched())
        self.assertEqual(sorted(pools), ["pp", "tp"])

    def test_the_pp_entry_is_the_tier_the_boot_already_built(self):
        # The rebind needs a HANDLE per phase, not a second pp pool.
        s = _sched()
        pools = build_phase_flip_host_pools(s)
        self.assertIs(pools["pp"], s.tree_cache.token_to_kv_pool_host)

    def test_the_tp_pin_is_allocated_from_the_tp_device_pool(self):
        # DESIGN_706 C1: a host pool is allocated FROM its device pool, which
        # is why this cannot be derived after the fact and must run at boot.
        s = _sched()
        pools = self._patched(s)
        self.assertIs(
            pools["tp"].device_pool,
            s.phase_flip_stacks.tp_worker.model_runner.token_to_kv_pool,
        )


class TestItIsAStagingPinNotAMirror(CustomTestCase):
    """#810: a `retention` tier is sized as a RATIO of the device pool; a
    `staging` tier holds only what is in flight. The rebind needs the second,
    and a ratio-sized second pool would duplicate retention the pp tier already
    provides while charging the pinned host budget for capacity nothing reads.
    """

    def test_the_ratio_is_not_used(self):
        build_phase_flip_host_pools(_sched())
        self.assertEqual(_HostPool.last.ratio, 0, "a RATIO is the mirror-shaped answer")
        self.assertGreater(_HostPool.last.size_gb, 0, "an explicit size instead")

    def test_the_size_is_derived_from_measured_cells_not_guessed(self):
        s = _sched(cell=8192)
        gib = _staging_pin_gib(s, _DevicePool(8192))
        expected = (4096 * 8 * PHASE_FLIP_STAGING_CHUNKS * 8192) / float(1 << 30)
        self.assertAlmostEqual(gib, expected, places=6)

    def test_it_scales_with_the_work_not_with_the_pool(self):
        # The defining property of a staging pin: doubling the device pool
        # changes nothing; doubling the in-flight work doubles the pin.
        s = _sched()
        base = _staging_pin_gib(s, _DevicePool(8192))
        s.server_args.max_running_requests = 16
        self.assertAlmostEqual(_staging_pin_gib(s, _DevicePool(8192)), 2 * base, 6)


class TestTheRefusalIsCONVERTEDNotDeleted(CustomTestCase):
    """CAN-FAIL. A genuinely absent or mis-shaped pool must STILL refuse."""

    def test_no_host_tier_at_all_yields_nothing_to_bind(self):
        self.assertEqual(build_phase_flip_host_pools(_sched(host=False)), {})

    def test_no_tp_device_pool_leaves_the_tp_phase_unbound(self):
        pools = build_phase_flip_host_pools(_sched(tp_pool=False))
        self.assertEqual(sorted(pools), ["pp"])
        self.assertNotIn("tp", pools, "and the cutover must then refuse")

    def test_the_guard_still_raises_when_the_phase_is_unbound(self):
        # THE REAL GUARD, not a restatement. This is the W32 failure exactly,
        # and it must remain reachable -- the point of #847 is that the pool
        # can now exist, never that the check was removed.
        from sglang.srt.mem_cache.hicache_phase_binding import (
            RebindRefused,
            phase_pools_for,
        )

        s = _sched(tp_pool=False)
        s.phase_flip_host_pools = build_phase_flip_host_pools(s)
        with self.assertRaises(RebindRefused) as caught:
            phase_pools_for(s, "tp")
        self.assertIn("host pool", str(caught.exception))

    def test_a_constructor_that_refuses_does_not_take_down_the_boot(self):
        class _Boom(_HostPool):
            def __init__(self, *a, **k):
                raise RuntimeError("mis-shaped pool")

        s = _sched()
        s.tree_cache.token_to_kv_pool_host = _Boom.__new__(_Boom)
        s.tree_cache.token_to_kv_pool_host.__class__ = _Boom
        pools = build_phase_flip_host_pools(s)
        self.assertEqual(sorted(pools), ["pp"], "refuse loudly, boot anyway")


class TestTheBootWiresIt(CustomTestCase):
    def test_the_scheduler_calls_the_writer(self):
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        # W33 arm 1: the wiring must sit AFTER `self.tree_cache` is assigned,
        # not inside `init_model_worker`. It needs THREE inputs -- both device
        # pools AND the host tier -- and the host tier hangs off `tree_cache`,
        # which is assigned after that method returns. Placed too early the
        # writer runs, finds no host tier, and refuses; measured on metal.
        src = inspect.getsource(Scheduler.__init__)
        self.assertNotIn(
            "build_phase_flip_host_pools",
            inspect.getsource(Scheduler.init_model_worker),
            "too early: the host tier does not exist yet there",
        )
        self.assertIn("build_phase_flip_host_pools", src)
        self.assertIn("phase_flip_host_pools", src)

    def test_the_ledger_post_is_named(self):
        import inspect

        from sglang.srt.managers import phase_flip_boot

        src = inspect.getsource(phase_flip_boot.build_phase_flip_host_pools)
        self.assertIn("HOST-LEDGER POST", src)
        self.assertIn("floor", src.lower())


if __name__ == "__main__":
    unittest.main()


class TestTheHostTierAccessorKnowsTheLiveTree(CustomTestCase):
    """W33 arm 2 -- the W29 defect, repeated one strand later by me.

    The writer read `getattr(tree, "token_to_kv_pool_host", None)`. That
    attribute belongs to `HiRadixCache`; the tree this box runs is
    `UnifiedRadixCache`, which does not have it and reaches the host tier
    through `cache_controller.mem_pool_host`. The read returned None on the
    live tree, the writer logged its own "no HiCache host tier" refusal on
    every rank, and the rebind could not arm -- 6 `#718 hicache-phase-guard`
    warnings, the W32 read-through miss reproduced with its fix installed.

    Identical in shape to W29's `full_evictable_size_`: an attribute three
    tree types have and `UnifiedRadixCache` does not, with `getattr(..., 0)`
    turning the absence into a value that silently disabled the mechanism.
    Same tree class, same silent default. Hence a NAMED accessor plus this
    drift-detector, which tests the REAL classes rather than a double.
    """

    def test_the_direct_attribute_route(self):
        from sglang.srt.managers.phase_flip_boot import host_tier_of

        pool = object()
        self.assertIs(
            host_tier_of(types.SimpleNamespace(token_to_kv_pool_host=pool)), pool
        )

    def test_the_cache_controller_route_the_live_tree_uses(self):
        from sglang.srt.managers.phase_flip_boot import host_tier_of

        pool = object()
        tree = types.SimpleNamespace(
            token_to_kv_pool_host=None,
            cache_controller=types.SimpleNamespace(mem_pool_host=pool),
        )
        self.assertIs(host_tier_of(tree), pool)

    def test_absent_is_absent_not_a_route_i_forgot_to_look_at(self):
        from sglang.srt.managers.phase_flip_boot import host_tier_of

        self.assertIsNone(host_tier_of(types.SimpleNamespace()))
        self.assertIsNone(host_tier_of(None))

    def test_the_live_tree_class_really_lacks_the_direct_attribute(self):
        # THE DRIFT-DETECTOR, against the REAL class. If UnifiedRadixCache ever
        # grows `token_to_kv_pool_host`, the first route starts working and
        # this test says so -- rather than the accessor quietly depending on a
        # route that only some trees have, which is the whole defect.
        import inspect

        from sglang.srt.mem_cache import unified_radix_cache

        src = inspect.getsource(unified_radix_cache)
        self.assertNotIn(
            "self.token_to_kv_pool_host",
            src,
            "UnifiedRadixCache reaches the host tier via cache_controller; if "
            "that changed, revisit host_tier_of",
        )

    def test_the_writer_uses_the_named_accessor(self):
        import inspect

        from sglang.srt.managers import phase_flip_boot

        src = inspect.getsource(phase_flip_boot.build_phase_flip_host_pools)
        self.assertIn("host_tier_of(tree)", src)
        self.assertNotIn('getattr(tree, "token_to_kv_pool_host"', src)
