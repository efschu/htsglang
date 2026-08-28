"""#545: UnifiedRadixCache gains runtime attach/detach.

SIBLING to ``test_hicache_runtime_resize_545.py``, not an extension of it:
that file owns the RESIZE authority (21 pins) and this one owns attach/detach.
Splitting by operation keeps one authority per behaviour; merging would have
put two unrelated failure stories in one file.

WHY THIS EXISTS. ``UnifiedRadixCache.attach_storage_backend`` and
``detach_storage_backend`` were hard stubs that always returned failure --
"does not support runtime HiCache storage attach yet" -- while
``resize_storage_backend`` worked. ``UnifiedRadixCache`` is what
``mem_cache/registry.py:191`` constructs on the path that appends the MAMBA
component for ``is_hybrid_ssm``, so the hybrid-GDN family had resize-only: the
ticket's headline capability was exactly the half that was stubbed.

WHY IT IS SAFE TO IMPLEMENT, which was the open question. Attach must
re-derive the prefetch capacity through ``_symmetrize_prefetch_capacity``,
which enters an **all_reduce** across DCP/TP ranks and whose own guard says a
rank-local early return "would leave the other ranks in the all_reduce with no
partner". A single-rank attach would hang. It cannot happen: attach fans out
through ``FanOutCommunicator`` (``tokenizer_control_mixin.py:125,:360``) to
every rank and merges the results, so the group runs it together, and the
scheduler refuses a non-idle scheduler by name first.

These pins drive the real methods against a controller double. The double is
deliberately thin -- the methods under test are orchestration, and what must
be pinned is the ORDER and the REFUSALS, not the controller's internals.
"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


class _Controller:
    def __init__(self, backend_type=None):
        self.storage_backend_type = backend_type
        self.attached = []
        self.detached = 0
        self.mem_pool_host = MagicMock()
        self.mem_pool_host.entries = {"full": object(), "mamba": object()}

    def attach_storage_backend(self, **kwargs):
        self.attached.append(kwargs)
        self.storage_backend_type = kwargs.get("storage_backend")

    def detach_storage_backend(self):
        self.detached += 1


def _cache(*, controller=_Controller, enable_storage=False, backend_type=None):
    """A UnifiedRadixCache shell carrying only what these two methods read."""
    c = UnifiedRadixCache.__new__(UnifiedRadixCache)
    c.cache_controller = controller(backend_type) if controller else None
    c.enable_storage = enable_storage
    c.enable_storage_metrics = False
    c.prefetch_threshold = 256
    c.prefetch_stop_policy = "best_effort"
    c.write_through_threshold = 2
    # #966: detach now also releases the retired prefetch records, whose reap
    # its own clearing of `enable_hicache_storage` makes unreachable. A real
    # instance always carries this list (`_reset_full`, which `__init__` runs);
    # the shell has to carry it too, or it pins a detach that cannot run.
    c._retired_prefetch = []
    c.calls = []
    c._apply_storage_runtime_config = lambda **kw: c.calls.append(("config", kw))
    c._symmetrize_prefetch_capacity = lambda: c.calls.append(("symmetrize", None))
    c._drain_storage_control_queues_impl = lambda **kw: c.calls.append(("drain", kw))
    return c


class TestAttachValidatesBeforeActing(unittest.TestCase):
    """Validation must have NO side effects -- a rejected policy must not
    leave a half-attached controller behind."""

    def test_a_bad_prefetch_policy_is_refused_by_name(self):
        c = _cache()
        ok, msg = c.attach_storage_backend("file", hicache_storage_prefetch_policy="x")
        self.assertFalse(ok)
        self.assertIn("hicache_storage_prefetch_policy", msg)
        self.assertEqual(c.cache_controller.attached, [], "no side effect on refusal")

    def test_a_bad_write_policy_is_refused_by_name(self):
        c = _cache()
        ok, msg = c.attach_storage_backend("file", hicache_write_policy="nope")
        self.assertFalse(ok)
        self.assertIn("hicache_write_policy", msg)
        self.assertEqual(c.cache_controller.attached, [])

    def test_unparsable_extra_config_is_refused_not_raised(self):
        c = _cache()
        ok, msg = c.attach_storage_backend(
            "file", storage_backend_extra_config_json="{not json"
        )
        self.assertFalse(ok)
        self.assertIn("Failed to parse", msg)
        self.assertEqual(c.cache_controller.attached, [])


class TestAttachHappyPath(unittest.TestCase):
    def test_it_attaches_and_enables_storage(self):
        c = _cache()
        ok, msg = c.attach_storage_backend("file")
        self.assertTrue(ok, msg)
        self.assertEqual(len(c.cache_controller.attached), 1)
        self.assertEqual(c.cache_controller.attached[0]["storage_backend"], "file")

    def test_the_state_component_is_covered_not_refused(self):
        """No silent partial capability (#268): the controller's host pools
        span every component it owns, MAMBA included, and that whole set is
        what gets passed -- so this is full coverage rather than a KV-only
        attach wearing a success message."""
        c = _cache()
        c.attach_storage_backend("file")
        pools = c.cache_controller.attached[0]["host_pools"]
        self.assertIn("mamba", pools)
        self.assertIn("full", pools)

    def test_capacity_is_resymmetrized_AFTER_the_config_is_applied(self):
        """Order matters: the limit is the MIN host-pool size across ranks and
        can only be derived once each rank knows storage is on."""
        c = _cache()
        c.attach_storage_backend("file")
        names = [n for n, _ in c.calls]
        self.assertIn("symmetrize", names)
        self.assertLess(names.index("config"), names.index("symmetrize"))


class TestAttachIdempotenceAndBackendSwap(unittest.TestCase):
    def test_reattaching_the_same_backend_is_success(self):
        c = _cache(enable_storage=True, backend_type="file")
        ok, msg = c.attach_storage_backend("file")
        self.assertTrue(ok)
        self.assertIn("already", msg)

    def test_a_different_backend_is_refused_rather_than_swapped(self):
        """A silent swap would strand every page written under the old
        backend."""
        c = _cache(enable_storage=True, backend_type="file")
        ok, msg = c.attach_storage_backend("mooncake")
        self.assertFalse(ok)
        self.assertIn("detach", msg)
        self.assertEqual(c.cache_controller.attached, [])


class TestNoControllerIsRefusedByName(unittest.TestCase):
    def test_attach_without_a_controller(self):
        c = _cache(controller=None)
        ok, msg = c.attach_storage_backend("file")
        self.assertFalse(ok)
        self.assertIn("no cache controller", msg)

    def test_detach_without_a_controller(self):
        c = _cache(controller=None)
        ok, msg = c.detach_storage_backend()
        self.assertFalse(ok)
        self.assertIn("no cache controller", msg)


class TestDetachOrderIsTheContract(unittest.TestCase):
    """Drain BEFORE the controller teardown, or acks and releases can no
    longer be matched to their nodes and host pages and locks leak."""

    def test_it_drains_before_and_after_the_teardown(self):
        c = _cache(enable_storage=True, backend_type="file")
        ok, _ = c.detach_storage_backend()
        self.assertTrue(ok)
        names = [n for n, _ in c.calls]
        self.assertEqual(names.count("drain"), 2, "drain before AND after")
        self.assertEqual(c.cache_controller.detached, 1)

    def test_the_drain_is_local_not_collective(self):
        """None limits mean 'everything on this rank'. A detach may not wait
        on an all_reduce whose peers may already have left it."""
        c = _cache(enable_storage=True, backend_type="file")
        c.detach_storage_backend()
        drains = [kw for n, kw in c.calls if n == "drain"]
        for kw in drains:
            self.assertIsNone(kw["n_revoke"])
            self.assertIsNone(kw["n_backup"])
            self.assertIsNone(kw["n_release"])

    def test_storage_is_disabled_after_detach(self):
        c = _cache(enable_storage=True, backend_type="file")
        c.detach_storage_backend()
        self.assertFalse(c.enable_storage)
        self.assertFalse(c.enable_storage_metrics)

    def test_detach_is_idempotent_on_already_disabled_storage(self):
        """Leftover state from a partial detach must still be cleaned up, so
        the controller is asked regardless of the flag."""
        c = _cache(enable_storage=False, backend_type="file")
        ok, _ = c.detach_storage_backend()
        self.assertTrue(ok)
        self.assertEqual(c.cache_controller.detached, 1)


class TestAdminOperationsNeverKillTheServer(unittest.TestCase):
    """A failing admin call returns a message; it does not raise into the
    request path."""

    def test_a_raising_controller_attach_is_reported(self):
        c = _cache()
        c.cache_controller.attach_storage_backend = MagicMock(
            side_effect=RuntimeError("backend exploded")
        )
        ok, msg = c.attach_storage_backend("file")
        self.assertFalse(ok)
        self.assertIn("backend exploded", msg)

    def test_a_raising_controller_detach_is_reported(self):
        c = _cache(enable_storage=True, backend_type="file")
        c.cache_controller.detach_storage_backend = MagicMock(
            side_effect=RuntimeError("teardown exploded")
        )
        ok, msg = c.detach_storage_backend()
        self.assertFalse(ok)
        self.assertIn("teardown exploded", msg)


if __name__ == "__main__":
    unittest.main()
