"""#421 F2 falsifier: the ``--lane-offload-*`` flags must reach the register.

``server_args._handle_lane_offload_register`` validated
``--lane-offload-profile`` / ``--lane-offload-class-policy`` /
``--lane-offload-park-targets`` and discarded the resolved values
("recomputed at configure time"). Nothing in production reached configure
time, and ``get_global_register()``'s fallback built a bare latency-profile
register instead -- so the three flags were accepted, validated and silently
ignored. A silent default is worse than a crash: the operator gets a server
that looks configured.

These tests exercise the real entry point
(``configure_global_register_from_server_args``), not a hand-built register.
Can-fail proof: delete the call at ``model_runner.py`` and
``test_the_runner_init_site_calls_the_configure_entry`` goes red; delete the
body of the configure entry and every observability test below goes red.

Hermetic: no GPU, no model, no torch.distributed.
"""

import ast
import pathlib
import unittest

from sglang.srt.model_executor.offload_movement import (
    DEFAULT_PARK_TARGET_ORDER,
    park_target_order_from_register,
)
from sglang.srt.model_executor.offload_register import (
    configure_global_register_from_server_args,
    get_global_register,
    reset_global_register,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]


class _ServerArgs:
    def __init__(self, profile="latency", class_policy=None, park_targets=None):
        self.lane_offload_profile = profile
        self.lane_offload_class_policy = class_policy
        self.lane_offload_park_targets = park_targets


class _RegisterOn:
    """Turn SGLANG_OFFLOAD_REGISTER on for the duration of a test and leave
    the process-global register clean afterwards."""

    def __enter__(self):
        from sglang.srt.environ import envs

        reset_global_register()
        self._ctx = envs.SGLANG_OFFLOAD_REGISTER.override(True)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        reset_global_register()
        return False


class TestFlagsAreObservableInTheRegister(CustomTestCase):
    def test_capacity_profile_reaches_the_policies(self):
        """The defect verbatim: --lane-offload-profile capacity produced a
        latency register."""
        with _RegisterOn():
            configure_global_register_from_server_args(_ServerArgs("capacity"))
            reg = get_global_register()
            modes = {k: p.mode for k, p in reg.policies.items()}
        self.assertTrue(modes, "register has no policies")
        self.assertEqual(set(modes.values()), {"ram"}, modes)

    def test_default_profile_is_still_latency(self):
        with _RegisterOn():
            configure_global_register_from_server_args(_ServerArgs("latency"))
            modes = {k: p.mode for k, p in get_global_register().policies.items()}
        self.assertEqual(set(modes.values()), {"resident"}, modes)

    def test_per_class_override_beats_the_preset(self):
        with _RegisterOn():
            configure_global_register_from_server_args(
                _ServerArgs("latency", class_policy="graph_rungs=ram")
            )
            policies = get_global_register().policies
        self.assertEqual(policies["graph_rungs"].mode, "ram")
        self.assertEqual(policies["lane_workspaces"].mode, "resident")

    def test_park_target_order_reaches_the_movement_layer(self):
        """``parse_park_target_order``'s result had exactly one production
        caller -- the argument-time syntax check -- and no runtime consumer."""
        with _RegisterOn():
            configure_global_register_from_server_args(
                _ServerArgs("latency", park_targets="peer_vram>host_ram")
            )
            self.assertEqual(
                get_global_register().park_target_order,
                ("peer_vram", "host_ram"),
            )
            self.assertEqual(
                park_target_order_from_register(),
                ("peer_vram", "host_ram"),
            )

    def test_movement_backend_default_takes_the_configured_order(self):
        from sglang.srt.model_executor.offload_movement import (
            FakeDeviceOps,
            RealMovementBackend,
        )

        with _RegisterOn():
            configure_global_register_from_server_args(
                _ServerArgs("latency", park_targets="peer_vram>host_ram")
            )
            backend = RealMovementBackend(FakeDeviceOps())
        self.assertEqual(backend._order, ("peer_vram", "host_ram"))


class TestRefusalsAreLoud(CustomTestCase):
    def test_planted_profile_typo_refuses(self):
        """#240-style hard reject: a typo must not degrade to the default."""
        with _RegisterOn():
            with self.assertRaises(ValueError) as ctx:
                configure_global_register_from_server_args(_ServerArgs("capacty"))
        self.assertIn("capacty", str(ctx.exception))

    def test_planted_park_target_typo_refuses(self):
        with _RegisterOn():
            with self.assertRaises(ValueError) as ctx:
                configure_global_register_from_server_args(
                    _ServerArgs("latency", park_targets="peer_vam>host_ram")
                )
        self.assertIn("peer_vam", str(ctx.exception))

    def test_own_vram_as_a_park_target_refuses(self):
        with _RegisterOn():
            with self.assertRaises(ValueError):
                configure_global_register_from_server_args(
                    _ServerArgs("latency", park_targets="own_vram>host_ram")
                )

    def test_planted_class_policy_typo_refuses(self):
        with _RegisterOn():
            with self.assertRaises(ValueError) as ctx:
                configure_global_register_from_server_args(
                    _ServerArgs("latency", class_policy="graph_rung=ram")
                )
        self.assertIn("graph_rung", str(ctx.exception))


class TestDefaultPathAndIdempotence(CustomTestCase):
    def test_disabled_feature_is_a_no_op(self):
        """SGLANG_OFFLOAD_REGISTER off => no register, no side effect."""
        reset_global_register()
        self.assertIsNone(
            configure_global_register_from_server_args(_ServerArgs("capacity"))
        )
        self.assertIsNone(get_global_register())
        self.assertEqual(park_target_order_from_register(), DEFAULT_PARK_TARGET_ORDER)

    def test_second_runner_does_not_rebuild_the_register(self):
        """A draft runner / #274 lane runner constructs a second ModelRunner
        in the same process. A rebuild there would drop every item the first
        runner's adapters already booked."""
        with _RegisterOn():
            configure_global_register_from_server_args(_ServerArgs("capacity"))
            reg = get_global_register()
            reg.register("item/one", "graph_rungs", 1024, 0.0)
            configure_global_register_from_server_args(_ServerArgs("latency"))
            again = get_global_register()
        self.assertIs(again, reg)
        self.assertIsNotNone(again.get("item/one"))
        self.assertEqual(
            {p.mode for p in again.policies.values()},
            {"ram"},
            "the second call rebuilt the register",
        )


class TestTheWiringIsAtRunnerInit(CustomTestCase):
    """A structural pin: the register must be configured BEFORE the first
    adapter read, and the adapters run inside ModelRunner. Asserting the call
    site (not just the function) is what keeps a future refactor from moving
    the call after the pools are built, which would silently restore the
    fallback register."""

    RUNNER = "python/sglang/srt/model_executor/model_runner.py"

    def test_the_runner_init_site_calls_the_configure_entry(self):
        tree = ast.parse((_REPO_ROOT / self.RUNNER).read_text())
        init = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ModelRunner":
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name == "__init__":
                        init = sub
        self.assertIsNotNone(init, "ModelRunner.__init__ not found")
        names = [
            n.func.id
            for n in ast.walk(init)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
        self.assertIn(
            "configure_global_register_from_server_args",
            names,
            "#421 F2 regressed: the --lane-offload-* flags no longer reach "
            "the offload register at runner init.",
        )


if __name__ == "__main__":
    unittest.main()
