"""#811: release the admission anchor pin at the write-through ack.

#773's determination named the remaining capacity axis: a running request
holds its matched/donated checkpoint's mamba pin from admission until the
request finishes, long after the checkpoint's write-through backup has been
ACKNOWLEDGED. Releasing the pin at the ack drops the per-request floor term
(2 -> 1 on the reference boot) and re-sizes the pool (20 -> ~12 slots,
~299 MiB of state on PP0: 8 slots x 37.41 MiB per the boot log's
[auto-mamba] line).

THE EDGE THIS FILE EXISTS TO GUARD. An early release is memory-safety work:
a pin released before the persistent copy exists does not crash, it
CORRUPTS -- #767 measured degenerate output in 9/10 salted probes when
releasing on ``host_value`` alone, because write-through publishes
``host_value`` when the transfer is HANDED OVER, not when it lands. Every
release edge in the mechanism must therefore run through
``MambaComponent.anchor_release_admissible`` (host copy present AND the node
no longer in ``ongoing_write_through``), and the tests below pin each edge:

* an in-flight backup is never released (the #767 mutant dies here);
* a device-only anchor is never released by the sweep;
* a released pin is never decremented a second time (#583 pairing);
* a ref this request never took is never given back (theft guard);
* the admission-site pin is never settled early -- it protects the
  deferred-COW source until the first extend forward has run.

CPU-only: no GPU, no DMA controller. The multi-turn load behavior (ack
latency under pressure, PP rank skew) is explicitly METAL scope, queued in
/spinning/gpu-arb/WINDOW-QUEUE.md.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
from sglang.srt.mem_cache.mamba_pool_floor import (
    mamba_anchor_ack_release_active,
    mamba_slots_per_running_req,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from test_mamba_pin_budget_live_773 import _build, _checkpoint_nodes, _mamba_value

register_cpu_ci(est_time=15)


def _mamba_lock_ref(node):
    return node.component_data[ComponentType.MAMBA].lock_ref


def _armed_args(**overrides) -> ServerArgs:
    """ServerArgs on which mamba_anchor_ack_release_active answers True."""
    sa = ServerArgs(model_path="dummy", page_size=1)
    sa.enable_hierarchical_cache = True
    sa.hicache_write_policy = "write_through"
    sa.mamba_anchor_ack_release = True
    for k, v in overrides.items():
        setattr(sa, k, v)
    return sa


def _arm(server_args) -> None:
    server_args.enable_hierarchical_cache = True
    server_args.hicache_write_policy = "write_through"
    server_args.mamba_anchor_ack_release = True


def _req(node) -> SimpleNamespace:
    return SimpleNamespace(
        last_node=node,
        mamba_anchor_pin_held=False,
        mamba_anchor_pin_released=False,
    )


class _ReorderEnv(CustomTestCase):
    """The #811 predicate requires the #755/#773 reorder to be armed."""

    def setUp(self):
        self._env = mock.patch.dict(
            os.environ, {"SGLANG_MAMBA_SLOT_REORDER": "1"}, clear=False
        )
        self._env.start()
        self.addCleanup(self._env.stop)


class TestThePredicate(_ReorderEnv):
    def test_armed_when_all_conditions_hold(self):
        self.assertTrue(mamba_anchor_ack_release_active(_armed_args()))

    def test_off_by_default(self):
        """No flag -> inert, floor byte-identical to the #755 state."""
        sa = _armed_args(mamba_anchor_ack_release=None)
        self.assertFalse(mamba_anchor_ack_release_active(sa))
        self.assertEqual(mamba_slots_per_running_req(sa), 2)

    def test_requires_the_reorder(self):
        with mock.patch.dict(os.environ, {"SGLANG_MAMBA_SLOT_REORDER": "0"}):
            self.assertFalse(mamba_anchor_ack_release_active(_armed_args()))

    def test_excluded_dec_site_families_refuse_arming(self):
        """kv-session-offload, streaming sessions, session radix cache and PD
        disaggregation release the admission lock without DecLockRefParams
        (dec-site audit #811); arming alongside them would double-release."""
        for field, value in (
            ("enable_kv_session_offload", True),
            ("enable_streaming_session", True),
            ("enable_session_radix_cache", True),
            ("disaggregation_mode", "decode"),
        ):
            with self.subTest(field=field):
                self.assertFalse(
                    mamba_anchor_ack_release_active(_armed_args(**{field: value}))
                )

    def test_armed_floor_drops_the_pinned_checkpoint_term(self):
        """per-req 2 -> 1: the pin is retention-budget funded when armed."""
        self.assertEqual(mamba_slots_per_running_req(_armed_args()), 1)

    def test_describe_names_the_mechanism(self):
        from sglang.srt.mem_cache.mamba_pool_floor import describe_mamba_floor

        self.assertIn("#811", describe_mamba_floor(_armed_args(), 8))


class _ArmedCache(_ReorderEnv):
    """A live UnifiedRadixCache with the #811 mechanism armed."""

    def _armed_one_node(self):
        cache, allocator, pool, server_args = _build()
        _arm(server_args)
        nodes = _checkpoint_nodes(cache, allocator, pool, 1)
        self.assertGreater(len(nodes), 0)
        node = nodes[0]
        self.assertIsNotNone(_mamba_value(node))
        self.assertTrue(cache._anchor_ack_release_armed)
        return cache, node

    def _pin(self, cache, node, settle: bool = False):
        req = _req(node)
        result = cache.inc_lock_ref(node)
        cache.note_anchor_pin(req, result, settle=settle)
        return req


class TestReleaseAtTheAck(_ArmedCache):
    """The red-first core: pin held across the in-flight window, released
    exactly when `_finish_write_through_ack` has retired the backup."""

    def test_the_full_ack_lifecycle(self):
        cache, node = self._armed_one_node()
        req = self._pin(cache, node)
        self.assertTrue(req.mamba_anchor_pin_held)
        self.assertEqual(_mamba_lock_ref(node), 1)

        # Backup handed over: host_value published, ack outstanding.
        node.component_data[ComponentType.MAMBA].host_value = object()
        cache._track_write_through_node(node, None)

        # CAN-FAIL EDGE (#767): before the ack, the sweep must refuse.
        self.assertFalse(cache.release_acked_anchor_pin(req))
        self.assertTrue(req.mamba_anchor_pin_held)
        self.assertEqual(_mamba_lock_ref(node), 1)

        # The ack lands; the next sweep releases the pin.
        cache._finish_write_through_ack(node.id)
        self.assertTrue(cache.release_acked_anchor_pin(req))
        self.assertFalse(req.mamba_anchor_pin_held)
        self.assertTrue(req.mamba_anchor_pin_released)
        self.assertEqual(_mamba_lock_ref(node), 0)
        self.assertGreater(
            cache.component_evictable_size_[ComponentType.MAMBA],
            0,
            "the released checkpoint must be evictable, that is the capacity",
        )

    def test_a_device_only_anchor_is_never_released_by_the_sweep(self):
        """No host copy anywhere: releasing would strand a dead anchor."""
        cache, node = self._armed_one_node()
        req = self._pin(cache, node)
        self.assertIsNone(node.component_data[ComponentType.MAMBA].host_value)
        self.assertFalse(cache.release_acked_anchor_pin(req))
        self.assertTrue(req.mamba_anchor_pin_held)
        self.assertEqual(_mamba_lock_ref(node), 1)

    def test_the_sweep_reissues_the_missing_backup(self):
        """A held pin on an unbacked anchor re-triggers write_backup (budget
        still applies inside), so the ack that permits the release comes."""
        cache, node = self._armed_one_node()
        req = self._pin(cache, node)
        with mock.patch.object(cache, "write_backup", return_value=0) as wb:
            self.assertFalse(cache.release_acked_anchor_pin(req))
        wb.assert_called_once_with(node)

    def test_a_ref_this_request_never_took_is_never_given_back(self):
        """Theft guard: pin-less request + someone else's ref on an acked
        node -- the sweep must not touch it."""
        cache, node = self._armed_one_node()
        node.component_data[ComponentType.MAMBA].host_value = object()
        other = cache.inc_lock_ref(node)  # another holder's ref
        del other
        req = _req(node)  # this request never pinned the node
        self.assertFalse(cache.release_acked_anchor_pin(req))
        self.assertEqual(_mamba_lock_ref(node), 1)

    def test_a_released_pin_is_never_decremented_twice(self):
        """#583 pairing: after the ack release, the request-end dec must
        skip the mamba component, preserving a second holder's ref."""
        cache, node = self._armed_one_node()
        node.component_data[ComponentType.MAMBA].host_value = object()
        second_holder = cache.inc_lock_ref(node)
        del second_holder
        req = self._pin(cache, node)
        self.assertEqual(_mamba_lock_ref(node), 2)
        self.assertTrue(cache.release_acked_anchor_pin(req))
        self.assertEqual(_mamba_lock_ref(node), 1)

        # The request finishes: its dec must skip mamba on this node.
        dec_params = DecLockRefParams()
        cache._anchor_dec_skip(req, dec_params)
        cache.dec_lock_ref(node, dec_params)
        self.assertEqual(
            _mamba_lock_ref(node),
            1,
            "the second holder's mamba ref was stolen by an unpaired dec",
        )
        self.assertFalse(req.mamba_anchor_pin_released, "marker must be consumed")


class TestTheSettlePolicy(_ArmedCache):
    """note_anchor_pin: admission never settles; the boundary settles."""

    def test_admission_site_keeps_even_an_acked_pin(self):
        """The matched anchor is the deferred-COW source until the first
        extend forward runs; settling at admission would let eviction reuse
        the source slot before the copy executes -- silent corruption."""
        cache, node = self._armed_one_node()
        node.component_data[ComponentType.MAMBA].host_value = object()
        req = self._pin(cache, node, settle=False)
        self.assertTrue(req.mamba_anchor_pin_held)
        self.assertEqual(_mamba_lock_ref(node), 1)

    def test_boundary_releases_an_acked_pin(self):
        cache, node = self._armed_one_node()
        node.component_data[ComponentType.MAMBA].host_value = object()
        req = self._pin(cache, node, settle=True)
        self.assertFalse(req.mamba_anchor_pin_held)
        self.assertTrue(req.mamba_anchor_pin_released)
        self.assertEqual(_mamba_lock_ref(node), 0)

    def test_boundary_keeps_an_in_flight_pin(self):
        """CAN-FAIL EDGE (#767): a backup that is handed over but not acked
        must keep its pin."""
        cache, node = self._armed_one_node()
        node.component_data[ComponentType.MAMBA].host_value = object()
        cache._track_write_through_node(node, None)
        req = self._pin(cache, node, settle=True)
        self.assertTrue(req.mamba_anchor_pin_held)
        self.assertFalse(req.mamba_anchor_pin_released)
        self.assertEqual(_mamba_lock_ref(node), 1)

    def test_boundary_refuses_to_persist_an_unbackable_pin(self):
        """No backup, none in flight: the pin is given back in the same
        step. This is the #581 half of the armed floor: a persistent pin may
        only exist while the retention budget bounds it. The state stays
        cached and evictable; there is no host copy a resume could
        half-read."""
        cache, node = self._armed_one_node()
        self.assertIsNone(node.component_data[ComponentType.MAMBA].host_value)
        req = self._pin(cache, node, settle=True)
        self.assertFalse(req.mamba_anchor_pin_held)
        self.assertTrue(req.mamba_anchor_pin_released)
        self.assertEqual(_mamba_lock_ref(node), 0)


class TestUnarmedIsInert(CustomTestCase):
    """Without the flag the mechanism must not exist at runtime."""

    def test_note_and_sweep_are_no_ops(self):
        cache, allocator, pool, server_args = _build()
        # Reorder off, flag off: the plain #773 state.
        nodes = _checkpoint_nodes(cache, allocator, pool, 1)
        node = nodes[0]
        self.assertFalse(cache._anchor_ack_release_armed)
        req = _req(node)
        result = cache.inc_lock_ref(node)
        cache.note_anchor_pin(req, result, settle=True)
        self.assertFalse(req.mamba_anchor_pin_held)
        self.assertFalse(req.mamba_anchor_pin_released)
        self.assertEqual(_mamba_lock_ref(node), 1, "the pin must stay")
        node.component_data[ComponentType.MAMBA].host_value = object()
        self.assertFalse(cache.release_acked_anchor_pin(req))
        self.assertEqual(_mamba_lock_ref(node), 1)
        dec_params = DecLockRefParams()
        cache._anchor_dec_skip(req, dec_params)
        self.assertNotIn(
            ComponentType.MAMBA,
            dec_params.skip_lock_node_ids,
            "unarmed dec params must be byte-identical",
        )


if __name__ == "__main__":
    unittest.main()
