"""#631 Route A: the DCP owner rule is bypassed silently when the DCP
process group is not built before the attention backend is constructed.

The first class here is the FALSIFIER. It does not test the guard; it
demonstrates the defect the guard exists for, so that the guard is not a
statement about code nobody has seen fail. Every assertion in it describes
the BROKEN state, and every one of them passes today:

* ``attn_dcp_size`` reports 1 and ``attn_dcp_rank`` reports 0 -- these are
  the two values the attention backend constructor caches for the lifetime
  of the process.
* ``uneven_dcp_owner_bounds()`` returns ``None`` on every rank even though
  the uneven-TP token ratios ARE installed, i.e. the operator did ask for
  token sharding. With the group built, the same three ranks get three
  disjoint owner ranges.
* Nothing raises and nothing warns.

The comparison in ``test_owner_rule_is_bypassed_without_the_group`` is the
whole point: the same process state, the same ratios, and the only
difference is whether the group exists.

The remaining classes are the guard's can-fail proof and its no-op proof
on the paths that must stay byte-identical.
"""

import logging
import unittest

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.dcp_group_guard import (
    assert_dcp_group_formed,
    assert_pd_decode_dcp_supported,
)
from sglang.srt.distributed.utils import (
    set_cp_token_ratios,
    set_tp_partition_ratios,
    uneven_dcp_owner_bounds,
)
from sglang.srt.runtime_context import get_parallel

# Uneven TP over three ranks: 5/5/6 token shares, block size S = 16.
RATIOS = [5, 5, 6]
BLOCK = sum(RATIOS)
EXPECTED_BOUNDS = [(BLOCK, 0, 5), (BLOCK, 5, 10), (BLOCK, 10, BLOCK)]


class _Args:
    """The few ServerArgs fields the guard reads."""

    def __init__(
        self,
        dcp_size=1,
        disaggregation_mode="null",
        disaggregation_transfer_backend="mooncake",
    ):
        self.dcp_size = dcp_size
        self.disaggregation_mode = disaggregation_mode
        self.disaggregation_transfer_backend = disaggregation_transfer_backend


class _RatiosInstalled(unittest.TestCase):
    """Install the uneven-TP plan and tear it back down.

    Both setters write module globals in distributed/utils.py, so the
    cleanup matters for every other test in the process.
    """

    def setUp(self):
        set_tp_partition_ratios(RATIOS)
        set_cp_token_ratios(RATIOS)
        self.addCleanup(set_cp_token_ratios, None)
        self.addCleanup(set_tp_partition_ratios, None)
        # No DCP group is built in a unit-test process; assert that rather
        # than assume it, because these tests mean nothing otherwise.
        self.assertIsNone(parallel_state.get_dcp_group_no_assert())

    def _as_rank(self, rank, size=3):
        """Stand in for a built DCP group of `size` at `rank`.

        Overriding the ParallelContext fields is exactly equivalent for
        every reader here: ``_v()`` consults the override map before the
        parallel_state getter, and the getter is the only thing a real
        group would change.
        """
        return get_parallel().override(
            dcp_enabled=True,
            dcp_size=size,
            dcp_rank=rank,
            attn_dcp_size=size,
            attn_dcp_rank=rank,
        )


class FalsifierSilentOwnerRuleBypass(_RatiosInstalled):
    """The defect, demonstrated. These assertions describe broken state."""

    def test_backend_caches_dcp_size_1_without_the_group(self):
        parallel = get_parallel()
        # The two reads layers/attention/triton_backend.py makes in its
        # constructor and caches on the instance.
        self.assertEqual(parallel.attn_dcp_size, 1)
        self.assertEqual(parallel.attn_dcp_rank, 0)
        self.assertFalse(parallel.dcp_enabled)

    def test_the_swallowing_read_is_attn_dcp_size_not_dcp_size(self):
        # dcp_size asserts, which is why a guard written against it would
        # report the ordering bug as a bare AssertionError from
        # parallel_state. attn_dcp_size is the one that quietly says 1,
        # and it is the one the backend reads.
        with self.assertRaises(AssertionError):
            _ = get_parallel().dcp_size
        self.assertEqual(get_parallel().attn_dcp_size, 1)

    def test_owner_rule_is_bypassed_without_the_group(self):
        # With the group: three disjoint ranges covering the block exactly.
        with_group = []
        for rank in range(3):
            with self._as_rank(rank):
                with_group.append(uneven_dcp_owner_bounds())
        self.assertEqual(with_group, EXPECTED_BOUNDS)
        covered = sum(hi - lo for _, lo, hi in with_group)
        self.assertEqual(covered, BLOCK)

        # Without the group: None on every rank. Not an error, not a
        # different split -- no split at all. Every rank now considers
        # every global slot its own local row.
        self.assertIsNone(uneven_dcp_owner_bounds())

    def test_bypass_is_silent(self):
        # No exception (covered above) and no log record either. There is
        # nothing in a boot log to notice this by.
        with self.assertNoLogs(level=logging.WARNING):
            self.assertIsNone(uneven_dcp_owner_bounds())
            self.assertEqual(get_parallel().attn_dcp_size, 1)


class GuardCanFail(_RatiosInstalled):
    """The guard fires on precisely the state the falsifier demonstrated."""

    def test_raises_when_group_missing(self):
        with self.assertRaises(RuntimeError) as ctx:
            assert_dcp_group_formed(_Args(dcp_size=3), where="unit-test")
        msg = str(ctx.exception)
        self.assertIn("unit-test", msg)
        self.assertIn("dcp_size=3", msg)
        self.assertIn("attn_dcp_size=1", msg)
        self.assertIn("built=False", msg)

    def test_raises_when_group_has_the_wrong_size(self):
        # A built group of the wrong width is the same class of defect and
        # is equally silent, so it is equally refused.
        with self._as_rank(0, size=2):
            with self.assertRaises(RuntimeError) as ctx:
                assert_dcp_group_formed(_Args(dcp_size=3), where="unit-test")
        self.assertIn("attn_dcp_size=2", str(ctx.exception))

    def test_passes_when_group_matches(self):
        for rank in range(3):
            with self._as_rank(rank):
                assert_dcp_group_formed(_Args(dcp_size=3), where="unit-test")


class GuardIsNoOpOnUnaffectedPaths(unittest.TestCase):
    """Byte-identity: the guard cannot fire where Route A is not in play."""

    def test_default_single_group_server(self):
        # dcp_size=1, no group: 1 == 1.
        assert_dcp_group_formed(_Args(dcp_size=1), where="unit-test")

    def test_pp_prefill_group(self):
        # The Route A prefill side is dcp_size=1 with no DCP group, i.e.
        # indistinguishable from a default server as far as this guard is
        # concerned. It must not need an exemption.
        assert_dcp_group_formed(
            _Args(dcp_size=1, disaggregation_mode="prefill"), where="unit-test"
        )
        assert_pd_decode_dcp_supported(_Args(dcp_size=1, disaggregation_mode="prefill"))

    def test_pd_decode_without_dcp(self):
        assert_pd_decode_dcp_supported(_Args(dcp_size=1, disaggregation_mode="decode"))

    def test_missing_fields_do_not_crash_the_guard(self):
        class Bare:
            pass

        assert_dcp_group_formed(Bare(), where="unit-test")
        assert_pd_decode_dcp_supported(Bare())


class GuardIsWiredIntoTheScheduler(_RatiosInstalled):
    """Binding proof: the guard runs, and runs BEFORE any backend is built.

    A guard module that nothing calls is dead code, and a guard called
    after the value has already been cached is worse than none. Both are
    checked here by driving the real
    ``Scheduler.init_all_attention_backends`` against a stub self, which
    needs no GPU and no distributed init.
    """

    class _StubScheduler:
        """Stub carrying ONLY what exists at this point in a real boot.

        Deliberately has no ``token_to_kv_pool_allocator``: the real
        Scheduler assigns that attribute after ``init_model_worker()``
        returns, i.e. strictly after this method runs. An earlier draft of
        the guard read it here and was therefore dead code -- the page_size
        check could never fire. Keeping the stub faithful to boot-time
        state is what catches that class of mistake, so do not add the
        attribute to make a test pass.
        """

        def __init__(self, server_args, page_size=1):
            self.server_args = server_args
            self.draft_worker = None
            self.order = []
            outer = self

            class _Allocator:
                pass

            allocator = _Allocator()
            allocator.page_size = page_size

            class _Worker:
                def init_attention_backends(self):
                    outer.order.append("build_backends")

                def get_memory_pool(self):
                    # Real signature: (req_to_token_pool, allocator).
                    return (None, allocator)

            self.tp_worker = _Worker()

    def test_page_size_is_read_from_the_worker_not_the_scheduler(self):
        # The regression guard for the dead-code bug described above: a
        # paged allocator must be REACHED and refused through the real
        # call site, not merely through a direct call to the function.
        from sglang.srt.managers.scheduler import Scheduler

        stub = self._StubScheduler(
            _Args(dcp_size=3, disaggregation_mode="decode"), page_size=64
        )
        self.assertFalse(hasattr(stub, "token_to_kv_pool_allocator"))
        with self._as_rank(0):
            with self.assertRaises(RuntimeError) as ctx:
                Scheduler.init_all_attention_backends(stub)
        self.assertIn("page_size == 1, got 64", str(ctx.exception))
        self.assertEqual(stub.order, [])

    def test_page_size_1_from_the_worker_is_accepted(self):
        from sglang.srt.managers.scheduler import Scheduler

        stub = self._StubScheduler(
            _Args(dcp_size=3, disaggregation_mode="decode"), page_size=1
        )
        with self._as_rank(0):
            Scheduler.init_all_attention_backends(stub)
        self.assertEqual(stub.order, ["build_backends"])

    def test_guard_refuses_before_backends_are_built(self):
        from sglang.srt.managers.scheduler import Scheduler

        stub = self._StubScheduler(_Args(dcp_size=3))
        with self.assertRaises(RuntimeError) as ctx:
            Scheduler.init_all_attention_backends(stub)
        self.assertIn("attn_dcp_size=1", str(ctx.exception))
        # The decisive assertion: the backend was never constructed, so
        # nothing cached the wrong value before the guard spoke.
        self.assertEqual(stub.order, [])

    def test_backends_are_built_when_the_group_matches(self):
        from sglang.srt.managers.scheduler import Scheduler

        stub = self._StubScheduler(_Args(dcp_size=3))
        with self._as_rank(0):
            Scheduler.init_all_attention_backends(stub)
        self.assertEqual(stub.order, ["build_backends"])

    def test_default_server_reaches_the_backends_unchanged(self):
        from sglang.srt.managers.scheduler import Scheduler

        stub = self._StubScheduler(_Args(dcp_size=1))
        Scheduler.init_all_attention_backends(stub)
        self.assertEqual(stub.order, ["build_backends"])


class PdDecodePreconditions(unittest.TestCase):
    """Boot-time versions of two refusals that today arrive per-request."""

    def test_stock_head_sharded_dcp_refused_at_boot(self):
        # No --rank-tp-ratio installed: decode.py would refuse on the
        # first transferred request instead.
        self.assertIsNone(parallel_state.get_dcp_group_no_assert())
        with self.assertRaises(RuntimeError) as ctx:
            assert_pd_decode_dcp_supported(
                _Args(dcp_size=3, disaggregation_mode="decode")
            )
        self.assertIn("uneven-TP replicated-KV", str(ctx.exception))

    def test_non_mooncake_transport_refused_at_boot(self):
        set_tp_partition_ratios(RATIOS)
        set_cp_token_ratios(RATIOS)
        self.addCleanup(set_cp_token_ratios, None)
        self.addCleanup(set_tp_partition_ratios, None)
        for backend in ("nixl", "mori"):
            with self.subTest(backend=backend):
                with self.assertRaises(RuntimeError) as ctx:
                    assert_pd_decode_dcp_supported(
                        _Args(
                            dcp_size=3,
                            disaggregation_mode="decode",
                            disaggregation_transfer_backend=backend,
                        )
                    )
                self.assertIn("mooncake", str(ctx.exception))

    def test_paged_allocator_refused_at_boot(self):
        # --page-size resolves late, so this cannot be caught by reading
        # the command line; the resolved allocator value is passed in.
        set_tp_partition_ratios(RATIOS)
        set_cp_token_ratios(RATIOS)
        self.addCleanup(set_cp_token_ratios, None)
        self.addCleanup(set_tp_partition_ratios, None)
        with self.assertRaises(RuntimeError) as ctx:
            assert_pd_decode_dcp_supported(
                _Args(dcp_size=3, disaggregation_mode="decode"), page_size=64
            )
        self.assertIn("page_size == 1, got 64", str(ctx.exception))

    def test_page_size_1_and_unknown_accepted(self):
        set_tp_partition_ratios(RATIOS)
        set_cp_token_ratios(RATIOS)
        self.addCleanup(set_cp_token_ratios, None)
        self.addCleanup(set_tp_partition_ratios, None)
        args = _Args(dcp_size=3, disaggregation_mode="decode")
        assert_pd_decode_dcp_supported(args, page_size=1)
        # None means "no allocator to ask yet" and must not invent a
        # refusal -- the request-path check still stands behind it.
        assert_pd_decode_dcp_supported(args, page_size=None)

    def test_mooncake_accepted(self):
        set_tp_partition_ratios(RATIOS)
        set_cp_token_ratios(RATIOS)
        self.addCleanup(set_cp_token_ratios, None)
        self.addCleanup(set_tp_partition_ratios, None)
        assert_pd_decode_dcp_supported(_Args(dcp_size=3, disaggregation_mode="decode"))


if __name__ == "__main__":
    unittest.main()
