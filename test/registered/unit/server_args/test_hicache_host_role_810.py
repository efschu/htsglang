"""Argument-time fail-fast for #810's host-tier role.

WHY THIS GATE IS A REFUSAL AND NOT A PRECEDENCE RULE. `--hicache-size`
already wins over `--hicache-ratio` in `pool_host/base.py` (the
`host_size > 0` branch), so a staging boot carrying both would BOOT and
quietly ignore the ratio. The operator reads the ratio back out of their own
launch line and believes the pinned host tier is device-pool-scaled when it
is not. The two flags express opposite intents under this role, so the
combination is a stated contradiction rather than a question of which wins.

Both directions of every gate, and the DEFAULT: with the role left at
'retention' none of this validation runs and no other argument changes
meaning. That default case is the one that protects the standing boot, which
passes --hicache-ratio 1.5 and must keep booting byte-identically.

    python -m pytest test/registered/unit/server_args/test_hicache_host_role_810.py -v
"""

import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RATIO_DEFAULT = ServerArgs.__dataclass_fields__["hicache_ratio"].default


def _args(**kwargs):
    """ServerArgs with model_path='dummy' short-circuits __post_init__ (the
    repo-wide convention for argument tests), so the #810 handler is invoked
    explicitly -- which also pins it as a handler of its own rather than a
    block buried in an unrelated one."""
    args = ServerArgs(model_path="dummy", **kwargs)
    args._handle_hicache_host_role()
    return args


def _staging(**kwargs):
    base = dict(
        hicache_host_role="staging",
        enable_hierarchical_cache=True,
        hicache_size=2,
        hicache_storage_backend="file",
    )
    base.update(kwargs)
    return _args(**base)


class TestHicacheHostRoleDefault(CustomTestCase):
    """The default path. These are the tests that keep the standing boot."""

    def test_retention_is_the_default(self):
        self.assertEqual(
            ServerArgs.__dataclass_fields__["hicache_host_role"].default,
            "retention",
        )

    def test_default_role_runs_no_validation(self):
        # The standing Arm-2 boot's shape: a ratio, no absolute size, and
        # hierarchical cache on. Under 'retention' this must stay legal --
        # it is the configuration currently serving.
        args = _args(
            enable_hierarchical_cache=True,
            hicache_ratio=1.5,
            hicache_storage_backend="file",
        )
        self.assertEqual(args.hicache_host_role, "retention")
        self.assertEqual(args.hicache_ratio, 1.5)
        self.assertEqual(args.hicache_size, 0)

    def test_retention_tolerates_what_staging_refuses(self):
        # Every one of the three staging refusals below is legal under the
        # default role. This is the both-directions half of each gate: the
        # refusals must come from the ROLE, not from the other flags.
        self.assertIsNotNone(_args(hicache_ratio=1.5))
        self.assertIsNotNone(_args(enable_hierarchical_cache=False))
        self.assertIsNotNone(
            _args(enable_hierarchical_cache=True, hicache_storage_backend=None)
        )


class TestHicacheHostRoleStaging(CustomTestCase):
    def test_staging_accepts_a_complete_configuration(self):
        args = _staging()
        self.assertEqual(args.hicache_host_role, "staging")
        self.assertEqual(args.hicache_size, 2)
        # Untouched: the role declares intent, it does not compute a size.
        # The size is the planner's to emit (#584/#785); a role that also
        # picked a number would be a second budget authority.
        self.assertEqual(args.hicache_ratio, RATIO_DEFAULT)

    def test_staging_refuses_a_ratio(self):
        # THE DANGER DIRECTION. Accepting this is the silent failure: the
        # boot succeeds, --hicache-size wins, and the launch line still says
        # 1.5 -- so the operator believes a device-pool-scaled tier is in
        # place. A capacity claim that is wrong in the operator's favour is
        # worse than a refusal.
        with self.assertRaises(ValueError) as cm:
            _staging(hicache_ratio=1.5)
        msg = str(cm.exception)
        self.assertIn("--hicache-ratio", msg)
        self.assertIn("--hicache-size", msg)
        # The message must say WHY the shapes differ, not merely that they
        # conflict, or the next reader re-adds the ratio.
        self.assertIn("bandwidth", msg)

    def test_staging_refuses_a_missing_size(self):
        # Without this gate the fallback is the default ratio -- i.e. exactly
        # the device-pool-scaled pinned pool this role exists to remove,
        # restored silently under a flag that claims to have removed it.
        with self.assertRaises(ValueError) as cm:
            _staging(hicache_size=0)
        self.assertIn("--hicache-size", str(cm.exception))

    def test_staging_refuses_without_a_storage_backend(self):
        # Staging is a buffer in FRONT of a retention tier. With nothing to
        # stage into, shrinking the host tier does not RELOCATE the cache, it
        # DISCARDS it: a capacity change silently becomes a hit-rate
        # collapse, which no capacity metric would show.
        with self.assertRaises(ValueError) as cm:
            _staging(hicache_storage_backend=None)
        self.assertIn("--hicache-storage-backend", str(cm.exception))

    def test_staging_requires_hierarchical_cache(self):
        with self.assertRaises(ValueError) as cm:
            _staging(enable_hierarchical_cache=False)
        self.assertIn("--enable-hierarchical-cache", str(cm.exception))

    def test_ratio_equal_to_the_default_is_not_a_contradiction(self):
        # The gate compares against the field default rather than a literal,
        # so a staging boot that happens to carry the default ratio is not
        # refused -- there is nothing to contradict. Pinning this keeps the
        # comparison honest if the default ever moves.
        args = _staging(hicache_ratio=RATIO_DEFAULT)
        self.assertEqual(args.hicache_ratio, RATIO_DEFAULT)


if __name__ == "__main__":
    unittest.main()
