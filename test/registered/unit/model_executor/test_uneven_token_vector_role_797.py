"""#797 -- the token-vector calibration stops being advisory-only.

THE BUG CLASS, stated once. After profiling, ``_maybe_suggest_dcp_token_
vector`` computes the OPTIMAL token-ownership vector from each rank's MEASURED
capacity, compares it against the active vector, and -- when an explicit vector
is present -- throws the result away as a log line asking a human to restart
with it. The advisory is correct, is printed on every boot, and nothing
consumes it. On this fork's reference rig that gap left roughly 10 % of the KV
pool unreachable while every boot named the better vector.

The cause is that ``SGLANG_UNEVEN_TOKEN_VECTOR`` is read as a BOOLEAN: "a
vector is present" was conflated with "the operator asserts THIS vector". #797
splits the two with ``--uneven-token-vector-role``:

  pin  (default) -- the vector is an assertion. Unchanged behaviour: it
                    suppresses the install and the runtime may only hint.
  seed           -- the vector is a sizing estimate. The measured optimum
                    supersedes it IN-PROCESS, with no restart.

Covered here:
  1. pin is byte-identical to the pre-#797 behaviour (the regression guard).
  2. seed installs the measured optimum, in coupled mode, without needing a
     derived --rank-kv-ratio (which _handle_uneven_tp would silently downgrade).
  3. seed still refuses to install a vector that does not improve the budget,
     and still refuses for the draft worker.
  4. THE CUTOVER SURVIVAL: a seed install rewrites the env the phase flip
     rebuilds its vector from, so the flip cannot revert the boot to the seed.
  5. The provenance warning fires on a pinned vector that is being beaten,
     names the size of the loss, and goes silent once the loop is closed.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.distributed.utils import get_cp_token_ratios, set_cp_token_ratios
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_VEC_ENV = "SGLANG_UNEVEN_TOKEN_VECTOR"
_ROLE_ENV = "SGLANG_UNEVEN_TOKEN_VECTOR_ROLE"


class _StubConfig:
    def __init__(self, tokens: int):
        self.max_total_num_tokens = tokens


class _StubConfigurator:
    def __init__(self, tokens: int):
        self._tokens = tokens

    def calculate_pool_sizes(self, budget_bytes, page_size):
        return _StubConfig(self._tokens)


def _server_args(derived: bool = False):
    """The gate reads the ROLE off the environment, never off this object --
    that is deliberate (the flip's second stack build does not consult this
    ServerArgs), and it is why this stub needs no role attribute at all."""
    return SimpleNamespace(
        uneven_kv_capacity_mode=lambda: derived,
        uneven_kv_speed_mode=lambda: False,
        uneven_kv_corridor_mode=lambda: False,
        uneven_kv_derived_mode=lambda: derived,
        rank_kv_ratio="capacity" if derived else "coupled",
        speculative_draft_solo_active=lambda: False,
        speculative_draft_solo_rank=lambda: None,
    )


def _run(
    per_rank_tokens,
    active,
    *,
    role=None,
    env_vector=None,
    derived=False,
    allow_install=True,
    draft_worker=False,
):
    """Drive the real method once per simulated DCP rank.

    Returns ``(installed_vector, warnings, env_after)``. ``env_after`` is a
    snapshot of the two environment variables taken BEFORE this helper restores
    them -- the cutover-survival assertions are precisely about what the method
    left in the environment, so they cannot read it after the restore."""
    dcp_size = len(per_rank_tokens)

    def _configurator_for(rank):
        return _StubConfigurator(per_rank_tokens[rank])

    def fake_all_gather_object(gathered, payload, group=None):
        gathered[:dcp_size] = [
            (r, max(per_rank_tokens[r], 0), None, None) for r in range(dcp_size)
        ]

    warnings: list = []
    prev = {k: os.environ.get(k) for k in (_VEC_ENV, _ROLE_ENV)}
    set_cp_token_ratios(list(active))
    try:
        if env_vector is None:
            os.environ.pop(_VEC_ENV, None)
        else:
            os.environ[_VEC_ENV] = env_vector
        if role is None:
            os.environ.pop(_ROLE_ENV, None)
        else:
            os.environ[_ROLE_ENV] = role

        for rank in range(dcp_size):
            stub = SimpleNamespace(
                dcp_size=dcp_size,
                page_size=1,
                tp_rank=rank,
                server_args=_server_args(derived),
                is_draft_worker=draft_worker,
            )
            stub._is_solo_draft_kv_host = lambda s=stub: (
                ModelRunnerKVCacheMixin._is_solo_draft_kv_host(s)
            )
            stub._solo_host_capacity_curve = lambda *a, s=stub: (
                ModelRunnerKVCacheMixin._solo_host_capacity_curve(s, *a)
            )
            stub._solo_fixed_point_capacity = (
                ModelRunnerKVCacheMixin._solo_fixed_point_capacity
            )
            stub._hybrid_kv_token_cap = lambda: None
            stub._corridor_local_capacity = lambda cfg: None

            world_group = mock.Mock(world_size=dcp_size, cpu_group=None)
            parallel = mock.Mock(attn_dcp_rank=rank)
            base = "sglang.srt.model_executor.model_runner_kv_cache_mixin"
            with (
                mock.patch(f"{base}.get_world_group", return_value=world_group),
                mock.patch(f"{base}.get_parallel", return_value=parallel),
                mock.patch(
                    "sglang.srt.model_executor.pool_configurator"
                    ".create_memory_pool_configurator",
                    side_effect=lambda mr: _configurator_for(mr.tp_rank),
                ),
                mock.patch(
                    "torch.distributed.all_gather_object",
                    side_effect=fake_all_gather_object,
                ),
                mock.patch(f"{base}.logger") as log,
            ):
                ModelRunnerKVCacheMixin._maybe_suggest_dcp_token_vector(
                    stub, 1 << 30, allow_install=allow_install
                )
                if rank == 0:
                    warnings.extend(c.args[0] for c in log.warning.call_args_list)
        env_after = {k: os.environ.get(k) for k in (_VEC_ENV, _ROLE_ENV)}
        return list(get_cp_token_ratios()), warnings, env_after
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# Capacities deliberately unbalanced against the active vector, so the measured
# optimum differs from it and the budget strictly improves. Mirrors the shape
# of the reference rig (one large rank, two smaller) without its exact numbers.
_CAPS = [600000, 360000, 372000]
_ACTIVE = [29, 19, 16]


class PinIsUnchangedBehaviour797(CustomTestCase):
    def test_an_explicit_vector_defaults_to_pin_and_suppresses_the_install(self):
        """The regression guard: no role set at all is the pre-#797 world."""
        installed, _, _env = _run(_CAPS, _ACTIVE, env_vector="29,19,16", role=None)
        self.assertEqual(installed, _ACTIVE)

    def test_role_pin_is_explicitly_the_same_as_no_role(self):
        installed, _, _env = _run(_CAPS, _ACTIVE, env_vector="29,19,16", role="pin")
        self.assertEqual(installed, _ACTIVE)

    def test_a_pin_suppresses_the_install_even_in_a_derived_mode(self):
        """capacity mode + an explicit pin stayed hint-only before #797 and
        must keep doing so: the pin is the stronger statement."""
        installed, _, _env = _run(
            _CAPS, _ACTIVE, env_vector="29,19,16", role="pin", derived=True
        )
        self.assertEqual(installed, _ACTIVE)


class SeedInstallsTheMeasuredVector797(CustomTestCase):
    def test_seed_installs_the_measured_optimum(self):
        installed, _, _env = _run(_CAPS, _ACTIVE, env_vector="29,19,16", role="seed")
        self.assertNotEqual(installed, _ACTIVE)

    def test_the_installed_vector_raises_the_context_budget(self):
        from sglang.srt.distributed.utils import cp_token_context_budget

        installed, _, _env = _run(_CAPS, _ACTIVE, env_vector="29,19,16", role="seed")
        self.assertGreater(
            cp_token_context_budget(installed, _CAPS),
            cp_token_context_budget(_ACTIVE, _CAPS),
        )

    def test_seed_does_not_need_a_derived_rank_kv_ratio(self):
        """The seed IS the request. Requiring --rank-kv-ratio as well would
        route this through _handle_uneven_tp's silent downgrade to 'coupled'
        on any boot without a --rank-tp-ratio plan -- i.e. exactly the
        PP-prefill boots that most need the measured vector."""
        installed, _, _env = _run(
            _CAPS, _ACTIVE, env_vector="29,19,16", role="seed", derived=False
        )
        self.assertNotEqual(installed, _ACTIVE)

    def test_seed_still_refuses_a_vector_that_does_not_improve(self):
        """Balanced capacities: the active vector is already optimal, so even
        a seed must leave it alone rather than churn the pool."""
        balanced = [320000, 320000, 320000]
        installed, _, _env = _run(balanced, [1, 1, 1], env_vector="1,1,1", role="seed")
        self.assertEqual(installed, [1, 1, 1])

    def test_the_draft_worker_stays_hint_only_under_seed(self):
        installed, _, _env = _run(
            _CAPS, _ACTIVE, env_vector="29,19,16", role="seed", draft_worker=True
        )
        self.assertEqual(installed, _ACTIVE)

    def test_seed_does_not_install_when_install_is_not_allowed(self):
        """The post-capture resize pass passes allow_install=False: the vector
        is frozen into pools and graphs by then."""
        installed, _, _env = _run(
            _CAPS, _ACTIVE, env_vector="29,19,16", role="seed", allow_install=False
        )
        self.assertEqual(installed, _ACTIVE)


class TheInstallSurvivesTheCutover797(CustomTestCase):
    """parse_flip_token_vector rebuilds the decode vector from the ENV, and
    phase_flip_runtime re-installs it at EVERY cutover. An install left only
    in set_cp_token_ratios is therefore reverted at the first flip, silently
    returning the boot to the seed it was supposed to supersede."""

    def test_a_seed_install_rewrites_the_env_the_flip_reads(self):
        installed, _, env = _run(_CAPS, _ACTIVE, env_vector="29,19,16", role="seed")
        # Assert the install HAPPENED first. Without this the equality below
        # also holds when nothing was installed at all (env still the seed,
        # installed still the seed) -- i.e. the test would pass against a
        # completely disabled gate.
        self.assertNotEqual(installed, _ACTIVE)
        self.assertEqual(env[_VEC_ENV], ",".join(str(v) for v in installed))

    def test_the_flip_resolver_then_returns_the_measured_vector(self):
        """End-to-end on the REAL resolver, not a re-implementation of it: the
        env this run left behind is fed to parse_flip_token_vector exactly as
        the flip's second stack build would read it."""
        from sglang.srt.managers.phase_flip_boot import parse_flip_token_vector

        installed, _, env = _run(_CAPS, _ACTIVE, env_vector="29,19,16", role="seed")
        self.assertNotEqual(installed, _ACTIVE)
        prev = os.environ.get(_VEC_ENV)
        try:
            os.environ[_VEC_ENV] = env[_VEC_ENV]
            sa = SimpleNamespace(phase_flip_tp_vector="32,16,16", tp_size=1, pp_size=3)
            self.assertEqual(parse_flip_token_vector(sa), installed)
        finally:
            if prev is None:
                os.environ.pop(_VEC_ENV, None)
            else:
                os.environ[_VEC_ENV] = prev

    def test_a_pin_leaves_the_env_untouched(self):
        installed, _, env = _run(_CAPS, _ACTIVE, env_vector="29,19,16", role="pin")
        self.assertEqual(env[_VEC_ENV], "29,19,16")
        self.assertEqual(installed, _ACTIVE)


class TheProvenanceWarning797(CustomTestCase):
    """An advisory that does not say how to stop needing it is how a 10 % pool
    gap survives for months."""

    def test_a_beaten_pin_is_named_with_the_size_of_the_loss(self):
        _, warns, _env = _run(_CAPS, _ACTIVE, env_vector="29,19,16", role="pin")
        self.assertTrue(any("#797 PINNED VECTOR IS COSTING" in w for w in warns), warns)

    def test_the_warning_names_the_flag_that_closes_the_loop(self):
        _, warns, _env = _run(_CAPS, _ACTIVE, env_vector="29,19,16", role="pin")
        joined = " ".join(warns)
        self.assertIn("--uneven-token-vector-role seed", joined)

    def test_the_warning_is_silent_once_the_loop_is_closed(self):
        _, warns, _env = _run(_CAPS, _ACTIVE, env_vector="29,19,16", role="seed")
        self.assertFalse(
            any("#797 PINNED VECTOR IS COSTING" in w for w in warns), warns
        )

    def test_no_warning_when_there_is_no_pin_to_blame(self):
        """Without an explicit vector the remedy is --rank-kv-ratio capacity,
        which the existing restart hint already names; blaming a pin that does
        not exist would be noise."""
        _, warns, _env = _run(_CAPS, _ACTIVE, env_vector=None, role=None)
        self.assertFalse(
            any("#797 PINNED VECTOR IS COSTING" in w for w in warns), warns
        )


class TheFlagItself797(CustomTestCase):
    def test_the_default_role_is_pin(self):
        from sglang.srt.server_args import ServerArgs

        self.assertEqual(
            ServerArgs.__dataclass_fields__["uneven_token_vector_role"].default,
            "pin",
        )

    def test_only_pin_and_seed_are_accepted(self):
        import argparse

        from sglang.srt.server_args import ServerArgs

        p = argparse.ArgumentParser()
        ServerArgs.add_cli_args(p)
        ns = p.parse_args(["--model-path", "m", "--uneven-token-vector-role", "seed"])
        self.assertEqual(ns.uneven_token_vector_role, "seed")
        with self.assertRaises(SystemExit):
            p.parse_args(
                ["--model-path", "m", "--uneven-token-vector-role", "authoritative"]
            )


class TheCallSiteAdmitsTheSeed797(CustomTestCase):
    """THE GATE INSIDE THE HELPER IS NOT ENOUGH, and a real boot proved it.

    `_maybe_suggest_dcp_token_vector` is called twice: once advisory
    (allow_install defaults to False) and once from `_resolve_memory_pool_
    config` with allow_install=True. That second call is itself gated -- on
    `uneven_kv_derived_mode()` when post-capture sizing is planned. A seeded
    vector in 'coupled' mode fails that outer gate, so the install branch is
    never reached and the helper's own seed handling never runs.

    Observed exactly so on boot_seed796.log: role 'seed' present in
    server_args, the pin warning correctly silent, and the vector still not
    installed. Structural rather than behavioural because the call site sits
    mid-way through a method that needs a fully profiled ModelRunner; the
    point is to fail if a future edit drops the seed clause from the branch."""

    def _source(self):
        import inspect

        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            ModelRunnerKVCacheMixin,
        )

        return inspect.getsource(ModelRunnerKVCacheMixin._resolve_memory_pool_config)

    def test_the_install_call_site_consults_the_seed_predicate(self):
        self.assertIn(
            "uneven_token_vector_is_seed",
            self._source(),
            "_resolve_memory_pool_config no longer admits a seeded vector to "
            "the allow_install=True call: a seed in 'coupled' mode with "
            "post-capture sizing will print the better vector and keep the "
            "smaller pool, which is the #797 defect returning",
        )

    def test_the_derived_mode_route_is_still_admitted(self):
        self.assertIn("uneven_kv_derived_mode", self._source())


class CanFail797(CustomTestCase):
    """Guards that must FAIL if the gate stops distinguishing the two roles --
    the failure mode being defended against is a future edit collapsing the
    role back into a boolean 'is the env set' test."""

    def test_pin_and_seed_do_not_produce_the_same_vector(self):
        pinned, _, _env = _run(_CAPS, _ACTIVE, env_vector="29,19,16", role="pin")
        seeded, _, _env = _run(_CAPS, _ACTIVE, env_vector="29,19,16", role="seed")
        self.assertNotEqual(
            pinned,
            seeded,
            "pin and seed produced the same vector: the role is being ignored "
            "and #797 is not wired",
        )

    def test_an_unknown_role_string_is_treated_as_a_pin(self):
        """Fail closed. A typo must not silently license an install."""
        installed, _, _env = _run(
            _CAPS, _ACTIVE, env_vector="29,19,16", role="SEEDLING"
        )
        self.assertEqual(installed, _ACTIVE)

    def test_the_role_is_read_case_insensitively(self):
        installed, _, _env = _run(_CAPS, _ACTIVE, env_vector="29,19,16", role="Seed")
        self.assertNotEqual(installed, _ACTIVE)


if __name__ == "__main__":
    unittest.main()
