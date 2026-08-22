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
from contextlib import ExitStack
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


def _skip_infos(
    *,
    dcp_size: int,
    ratios: list,
    world_size: int = 3,
    caps: list = None,
    allow_install: bool = True,
    ratios_override=None,
):
    """Drive ONE rank and return the INFO lines the method emitted.

    Deliberately not folded into ``_run``: that helper collects warnings from
    rank 0 of a full simulated group, and every gate under test here returns
    before the collective, so a group is the wrong shape for it entirely.
    """
    caps = caps or [600000, 360000, 372000]
    prev = {k: os.environ.get(k) for k in (_VEC_ENV, _ROLE_ENV)}
    set_cp_token_ratios(list(ratios) if ratios else [])

    def fake_all_gather_object(gathered, payload, group=None):
        gathered[:world_size] = [
            (r, max(caps[r], 0), None, None) for r in range(world_size)
        ]

    try:
        stub = SimpleNamespace(
            dcp_size=dcp_size,
            page_size=1,
            tp_rank=0,
            server_args=_server_args(False),
            is_draft_worker=False,
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

        world_group = mock.Mock(world_size=world_size, cpu_group=None)
        parallel = mock.Mock(attn_dcp_rank=0)
        base = "sglang.srt.model_executor.model_runner_kv_cache_mixin"
        with ExitStack() as stack:
            log = stack.enter_context(mock.patch(f"{base}.logger"))
            stack.enter_context(
                mock.patch(f"{base}.get_world_group", return_value=world_group)
            )
            stack.enter_context(
                mock.patch(f"{base}.get_parallel", return_value=parallel)
            )
            stack.enter_context(
                mock.patch(
                    "sglang.srt.model_executor.pool_configurator"
                    ".create_memory_pool_configurator",
                    side_effect=lambda mr: _StubConfigurator(caps[0]),
                )
            )
            stack.enter_context(
                mock.patch(
                    "torch.distributed.all_gather_object",
                    side_effect=fake_all_gather_object,
                )
            )
            if ratios_override is not None:
                # The method imports get_cp_token_ratios INSIDE its body, so
                # the patch has to land on the source module, not on a name
                # bound in the mixin's namespace.
                stack.enter_context(
                    mock.patch(
                        "sglang.srt.distributed.utils.get_cp_token_ratios",
                        return_value=ratios_override,
                    )
                )
            ModelRunnerKVCacheMixin._maybe_suggest_dcp_token_vector(
                stub, 1 << 30, allow_install=allow_install
            )
        return [c.args[0] % c.args[1:] for c in log.info.call_args_list]
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TheSilentGatesNowNameThemselves797(CustomTestCase):
    """#797: the four uniform refusals stop being invisible.

    WHY THIS EXISTS AT ALL. A boot on which the token vector never calibrates
    and a boot on which it calibrates and agrees are, in the log, the same
    boot: both print nothing. That ambiguity is the reason #797 was argued
    from inference ("the vector must be refused somewhere upstream") instead
    of read off a line. Every early return in
    ``_maybe_suggest_dcp_token_vector`` now names itself and carries the three
    numbers that decide it -- dcp_size, world_size, allow_install.

    These are assertions about DIAGNOSABILITY, which is a real property: the
    fix for #797 is chosen by which gate fires, so a gate that fires silently
    makes the fix a guess.
    """

    def test_the_length_mismatch_gate_names_itself(self):
        """THE #797 SUSPECT ITSELF.

        A vector of three entries against dcp_size=1 -- which is the shape at
        PP-phase sizing -- is refused by ``uneven_dcp_active`` at
        distributed/utils.py:329. Before this, that refusal was the silent
        ``return`` that made the whole ticket an inference.
        """
        infos = _skip_infos(dcp_size=1, ratios=[29, 19, 16])
        self.assertEqual(len(infos), 1, f"expected exactly one skip line: {infos}")
        line = infos[0]
        self.assertIn("#797 token-vector calibration SKIPPED", line)
        self.assertIn("uneven DCP is not active", line)
        # The numbers are the point. A reason without dcp_size does not let a
        # reader tell this gate from the world-size gate.
        self.assertIn("dcp_size=1", line)
        self.assertIn("world_size=3", line)
        # And the vector itself, so this cause is separable from the two below.
        self.assertIn("3 entries against dcp_size 1", line)

    def test_the_three_causes_of_the_first_gate_are_told_apart(self):
        """ONE GATE, THREE CAUSES, THREE FIXES.

        ``uneven_dcp_active`` refuses for no-vector-installed, for a uniform
        vector, and for a length mismatch (utils.py:326-330). They are not
        interchangeable: on a phase-flip boot the real one is the FIRST --
        the only install-capable call site runs from init_memory_pools
        (scheduler.py:1429), before build_phase_flip_tp_stack
        (scheduler.py:1446) performs the first set_cp_token_ratios, so there
        is no vector installed yet for it to improve on. A line that said only
        "not active" would collapse the three and leave #797 an inference,
        which is exactly the state this whole class exists to end.
        """
        absent = _skip_infos(dcp_size=3, ratios=[])[0]
        uniform = _skip_infos(dcp_size=3, ratios=[8, 8, 8])[0]
        mismatch = _skip_infos(dcp_size=1, ratios=[29, 19, 16])[0]

        self.assertIn("no token vector is installed", absent)
        self.assertIn("is uniform", uniform)
        self.assertIn("entries against dcp_size", mismatch)

        # Pairwise distinct, asserted rather than eyeballed: three reasons that
        # happened to share a substring would pass the three checks above.
        self.assertEqual(
            len({absent, uniform, mismatch}),
            3,
            "two of the three causes produced the same line",
        )

    def test_the_world_size_gate_names_itself_and_is_distinguishable(self):
        """Two gates, two different readings -- not one generic line twice."""
        infos = _skip_infos(dcp_size=3, ratios=[29, 19, 16], world_size=1)
        self.assertEqual(len(infos), 1, f"expected exactly one skip line: {infos}")
        self.assertIn("world size is 1", infos[0])
        self.assertIn("dcp_size=3", infos[0])
        # If both gates printed the same text, a log could not tell an
        # unsplittable group from a stale vector, and the #797 fix would still
        # be a guess. Assert they are actually different sentences.
        other = _skip_infos(dcp_size=1, ratios=[29, 19, 16])[0]
        self.assertNotEqual(
            infos[0].split(":")[1], other.split(":")[1], "both gates read alike"
        )

    def test_the_post_gather_gate_names_the_capacities(self):
        """The one refusal that happens AFTER the collective.

        It must be distinguishable from never having reached the collective at
        all, and it must print the capacities: a zero here is a budget defect
        on a named rank, not a calibration opt-out.
        """
        infos = _skip_infos(
            dcp_size=3, ratios=[29, 19, 16], caps=[600000, 0, 372000], world_size=3
        )
        self.assertEqual(len(infos), 1, f"expected exactly one skip line: {infos}")
        self.assertIn("no usable capacity", infos[0])
        self.assertIn("600000", infos[0])
        self.assertIn("0", infos[0])

    def test_the_install_flag_is_reported_because_it_selects_the_call_site(self):
        """``allow_install`` is what separates the two sizing call sites from
        the hint-only post-capture pass. A skip line that omits it cannot say
        WHICH of the three calls refused, which is exactly the question #797
        resolution (a) has to answer."""
        install = _skip_infos(dcp_size=1, ratios=[29, 19, 16], allow_install=True)[0]
        hint = _skip_infos(dcp_size=1, ratios=[29, 19, 16], allow_install=False)[0]
        self.assertIn("allow_install=True", install)
        self.assertIn("(install)", install)
        self.assertIn("allow_install=False", hint)
        self.assertIn("(hint-only)", hint)

    def test_a_calibrating_boot_prints_no_skip_line(self):
        """THE CONTROL, and the one that makes the others mean anything.

        An instrumentation that printed on every boot would satisfy every
        assertion above while telling a reader nothing. A run that passes all
        four gates must be silent on this channel.
        """
        infos = _skip_infos(dcp_size=3, ratios=[29, 19, 16], world_size=3)
        skips = [i for i in infos if "SKIPPED" in i]
        self.assertEqual(skips, [], f"a calibrating run still printed a skip: {skips}")

    def test_the_third_gate_is_documented_as_unreachable_in_production(self):
        """AN HONEST NEGATIVE RESULT, recorded rather than papered over.

        The third gate reads ``active = get_cp_token_ratios()`` and refuses on
        ``not active or len(active) != dcp_size``. Both that call and
        ``uneven_dcp_active`` read the SAME module-level ``_CP_TOKEN_RATIOS``
        (utils.py:244 and :326), so by the time control reaches it, gate one
        has already established that the vector is truthy and that its length
        equals dcp_size. It cannot fire in production.

        It is kept because it is a cheap defence against those two functions
        drifting apart, and it is tested here through an explicit patch so
        that the branch is known to be wired -- while this test states plainly
        that a passing assertion here is NOT evidence of a reachable path.
        Deleting the gate would also be defensible; silently counting it as
        covered would not be.
        """
        infos = _skip_infos(dcp_size=3, ratios=[29, 19, 16], ratios_override=[29, 19])
        self.assertEqual(len(infos), 1, f"expected exactly one skip line: {infos}")
        self.assertIn("no usable active vector", infos[0])


if __name__ == "__main__":
    unittest.main()
