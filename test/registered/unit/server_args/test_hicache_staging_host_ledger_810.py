"""#810: a staging host tier is PRICED at parse time, so the boot preflight
sees the RAM it does and does not take.

WHAT THE PREFLIGHT COULD NOT SEE. The launcher-side joint pinned-host check
(#550/#729) is real, but it is reached only inside the
`--enable-kv-session-offload` x hierarchical-cache branch of ServerArgs
validation, and only when the spill pool is non-zero. A boot without
kv-session-offload prices no host tier at all. And the DEFAULT sizing is
`--hicache-ratio`, which `hicache_configured_host_bytes` deliberately declines
to price before the device pool exists. So on the standing boot the largest
pinned host consumer on the rig -- 22.01 GB of MHATokenToKVPoolHost across
three PP ranks -- reached the preflight as nothing at all.

Under `--hicache-host-role staging` both obstacles are gone: an absolute
`--hicache-size` is mandatory, so the number is exact at parse time. Declaring
it is what turns a smaller tier into headroom somebody has summed, rather than
into a number that only appears in a boot log after the RAM is already gone.

THE RANK PRODUCT IS PART OF THE CLAIM. `--hicache-size` sizes ONE rank's tier;
every scheduler process allocates its own. A per-rank figure checked against
machine-wide RAM under-counts a multi-rank boot by the rank count -- a factor
of three on this rig -- which is an error in the operator's favour, the shape
this feature refuses everywhere else.

Both directions of every gate, plus the default: with the role left at
'retention' none of this runs.
"""

import unittest
from unittest import mock

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GB = 1_000_000_000
#: Comfortable machine: everything below fits unless a test makes it not fit.
ROOMY = (128 * GB, 100 * GB)


def _run(*, host_memory=ROOMY, **kwargs):
    """Drive the REAL handler with a stated host-memory reading.

    The reading is injected rather than measured: a guard whose verdict
    depends on how much RAM the CI box happens to have free is not a test of
    the guard.
    """
    base = dict(
        hicache_host_role="staging",
        enable_hierarchical_cache=True,
        hicache_size=2,
        hicache_storage_backend="file",
    )
    base.update(kwargs)
    args = ServerArgs(model_path="dummy", **base)
    with mock.patch(
        "sglang.srt.mem_cache.pinned_host_budget.pinned_host_memory_bytes",
        lambda: host_memory,
    ):
        args._handle_hicache_host_role()
    return args


class StagingTierIsPricedTest(CustomTestCase):
    def test_a_tier_that_fits_is_accepted(self):
        args = _run(hicache_size=2)
        self.assertEqual(args.hicache_size, 2)

    def test_a_tier_that_cannot_fit_is_refused(self):
        """THE FALSIFIER. Before this, no parse-time check priced the host tier
        on a boot without kv-session-offload -- any size at all was accepted
        and the failure surfaced as an OOM kill in a worker."""
        with self.assertRaises(ValueError) as ctx:
            _run(hicache_size=200)
        message = str(ctx.exception)
        self.assertIn("--hicache-size", message)
        self.assertIn("PINNED", message)

    def test_the_tier_is_counted_once_per_rank(self):
        """The rank product. 30 GB fits on its own and does not fit three
        times; a per-rank figure checked against machine-wide RAM would accept
        the three-rank boot that cannot fit."""
        self.assertIsNotNone(_run(hicache_size=30, tp_size=1, pp_size=1))
        with self.assertRaises(ValueError) as ctx:
            _run(hicache_size=30, tp_size=1, pp_size=3)
        self.assertIn("3 rank(s)", str(ctx.exception))

    def test_tensor_parallel_ranks_count_too(self):
        """Each TP rank holds its own host-pool shard, exactly as each PP stage
        does. Counting only the PP stages would under-count a TP boot."""
        with self.assertRaises(ValueError):
            _run(hicache_size=30, tp_size=3, pp_size=1)

    def test_the_spill_pool_is_summed_with_it(self):
        """The check is JOINT or it is not a check: two pools that each fit and
        together do not are the whole reason the launcher-side guard exists."""
        self.assertIsNotNone(_run(hicache_size=40))
        with self.assertRaises(ValueError) as ctx:
            _run(hicache_size=40, kv_session_offload_host_ram_gib=60)
        self.assertIn("--kv-session-offload-host-ram-gib", str(ctx.exception))

    def test_no_honest_host_number_means_no_guard(self):
        """`pinned_host_memory_bytes` returns (None, None) when it cannot get an
        honest figure. Refusing a boot on a fabricated one is worse than not
        checking -- the same rule the rest of this module follows."""
        self.assertIsNotNone(_run(hicache_size=500, host_memory=(None, None)))


class RetentionIsUnpricedTest(CustomTestCase):
    """The backward-compatibility half. The default role must reach none of it."""

    def test_the_default_role_prices_nothing(self):
        args = ServerArgs(
            model_path="dummy",
            enable_hierarchical_cache=True,
            hicache_ratio=1.5,
            hicache_size=500,
            hicache_storage_backend="file",
        )
        with mock.patch(
            "sglang.srt.mem_cache.pinned_host_budget.pinned_host_memory_bytes",
            lambda: (1 * GB, 1 * GB),
        ):
            args._handle_hicache_host_role()
        self.assertEqual(args.hicache_host_role, "retention")

    def test_a_ratio_sized_tier_has_no_parse_time_number(self):
        """Not an omission: a ratio multiplies the DEVICE pool, which does not
        exist until the model is loaded and profiled. `staging` refuses the
        ratio precisely so that this case cannot arise under the role."""
        from sglang.srt.mem_cache.pinned_host_budget import (
            hicache_configured_host_bytes,
        )

        self.assertIsNone(hicache_configured_host_bytes(0, 1.5))


if __name__ == "__main__":
    unittest.main()
