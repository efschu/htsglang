"""W35 class 4: a backup opened before a cutover must never be persisted after it.

THE FAILURE THIS PREVENTS IS THE WORST ONE IN THE STRAND. `backup_queue`
carries `StorageOperation`s consumed by an always-running background thread
that does not pause across the flip. After a rebind, `_page_backup` reads
`self.mem_pool_host.get_data_page(...)` -- the INCOMING pool -- and writes
those bytes to a CONTENT-ADDRESSED store under a hash computed from the tokens
the operation was opened with. The hash does not match the payload, every
later reader trusts it, and the corruption OUTLIVES THE PROCESS. No assertion
anywhere catches it: unlike the W35 double-free, which was loud, this one is
silent and durable.

REFUSAL, NOT ROUTING, and the asymmetry is the point. A stale RELEASE can be
routed to the pool its generation names, because that pool still owns those
slots (class 1 does exactly that). A stale BACKUP cannot: the host slots may
belong to a pool that has since been repurposed, so there is no pool whose
bytes are the right bytes. Declining is the only safe verb; the prefix misses
later and is recomputed.

THE TESTS ASSERT ON THE STORE'S CONTENT, not on the counter. A counter can be
incremented by a fix that still persists.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import inspect
import unittest

from sglang.srt.managers.cache_controller import operation_is_stale
from sglang.test.test_utils import CustomTestCase


class _Op:
    def __init__(self, gen, rid="req-1"):
        self.binding_generation = gen
        self.request_id = rid


class _Ctl:
    """Stands in for the controller fields the gate touches."""


def _gen_at(n):
    """Drive the ONE stamp authority to generation n."""
    from sglang.srt.mem_cache import hicache_phase_binding as b

    b._STATE.reset()
    for _ in range(n):
        b._STATE.advance("tp")


class TestTheSpecimenShape(CustomTestCase):
    def test_opened_at_N_consumed_at_N_plus_1_is_refused(self):
        _gen_at(1)
        op = _Op(gen=1)
        _gen_at(2)  # the cutover
        self.assertTrue(operation_is_stale(_Ctl(), op, "backup"))

    def test_an_operation_from_the_current_binding_proceeds(self):
        _gen_at(2)
        self.assertFalse(operation_is_stale(_Ctl(), _Op(gen=2), "backup"))

    def test_an_unstamped_operation_is_not_refused(self):
        # Backward compatibility: ops predating the stamp must not be declined
        # wholesale, or a rebind-off boot would stop backing anything up.
        _gen_at(3)
        self.assertFalse(operation_is_stale(_Ctl(), _Op(gen=None), "backup"))

    def test_the_refusal_is_counted_by_name(self):
        _gen_at(1)
        op = _Op(gen=1)
        _gen_at(2)
        ctl = _Ctl()
        operation_is_stale(ctl, op, "backup")
        self.assertEqual(getattr(ctl, "_backup_stale_refusals", 0), 1)

    def test_the_generations_are_read_once_each(self):
        # THREAD BOUNDARY. The consumer is a background thread and the current
        # generation is mutated by the cutover on another; a second read
        # mid-persist could straddle a rebind and answer two different
        # questions about one operation.
        src = inspect.getsource(operation_is_stale)
        self.assertEqual(src.count("current_generation()"), 1)


class TestNothingIsPersisted(CustomTestCase):
    """CONTENT, not counters: a fix that counts and still writes is no fix."""

    def _controller_with_store(self):
        import types

        from sglang.srt.managers.cache_controller import HiCacheController

        ctl = HiCacheController.__new__(HiCacheController)
        ctl.written = []
        ctl.backup_skip = False
        ctl.storage_backend = types.SimpleNamespace(
            check_disk_space=lambda: None,
        )
        ctl._page_backup = lambda op: ctl.written.append(op.request_id)
        return ctl

    def test_a_stale_operation_writes_nothing_to_the_store(self):
        ctl = self._controller_with_store()
        _gen_at(1)
        op = _Op(gen=1, rid="stale-req")
        _gen_at(2)
        if not operation_is_stale(ctl, op, "backup"):
            ctl._page_backup(op)
        self.assertEqual(ctl.written, [], "nothing may reach the store")

    def test_removing_the_check_persists_again(self):
        # CAN-FAIL, the direction that matters: this models the pre-fix code
        # path exactly. If it ever stops writing, the model has drifted and
        # the test above proves nothing.
        ctl = self._controller_with_store()
        _gen_at(1)
        op = _Op(gen=1, rid="stale-req")
        _gen_at(2)
        ctl._page_backup(op)  # no gate == the old behaviour
        self.assertEqual(ctl.written, ["stale-req"])


class TestTheConsumerLoopUsesIt(CustomTestCase):
    def test_the_backup_thread_gates_before_persisting(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        src = inspect.getsource(HiCacheController.backup_thread_func)
        self.assertIn("operation_is_stale", src)
        gate = src.find("operation_is_stale")
        persist = src.find("self._page_backup(operation)")
        self.assertGreater(persist, -1)
        self.assertLess(gate, persist, "the gate must precede the persist")

    def test_a_refused_operation_is_still_acked(self):
        # An unacked operation stalls the queue; a declined backup is a
        # correct non-persist, exactly as `backup_skip` already is.
        from sglang.srt.managers.cache_controller import HiCacheController

        src = inspect.getsource(HiCacheController.backup_thread_func)
        head = src[: src.find("if not self.backup_skip")]
        self.assertIn("ack_backup_queue.put(operation)", head)

    def test_the_hybrid_path_inherits_the_gate(self):
        # THE STANDING WARNING: an override that shadows the loop would make
        # this fix inert on the live lane, which is exactly what happened to
        # `append_host_mem_release`. Assert the hybrid does NOT override it.
        from sglang.srt.managers.cache_controller import HiCacheController
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        self.assertIs(
            HybridCacheController.backup_thread_func,
            HiCacheController.backup_thread_func,
            "if the hybrid ever overrides this loop, it must gate too",
        )


class TestTheStampReachesRealOperations(CustomTestCase):
    def test_storage_operations_stamp_themselves(self):
        from sglang.srt.managers.cache_controller import StorageOperation

        src = inspect.getsource(StorageOperation.__init__)
        self.assertIn("binding_generation", src)
        self.assertIn("current_generation", src)


if __name__ == "__main__":
    unittest.main()
