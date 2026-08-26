"""#897 -- ``SGLANG_UNEVEN_TOKEN_VECTOR`` beats ``--rank-kv-ratio``, silently.

THE DEFECT, at base commit 65a4b8dbd2 (= pin 0cd27d957d + #889 + #894)
----------------------------------------------------------------------
``resolve_cp_token_ratios`` (``distributed/utils.py:816-838`` at the base)
reads the env vector FIRST and returns on its PRESENCE::

    env_vec = envs.SGLANG_UNEVEN_TOKEN_VECTOR.get()
    if env_vec:
        ...
        return reduced

The explicit pin ``--rank-kv-ratio a,b,c`` is read twenty lines further down
and is therefore never compared to the value that beat it. ``grep -n
'logger\\.'`` over the whole function returns zero at the base, so a stale
vector -- from an old A/B run, or written back into the environment by this
process's own KV calibration -- sizes every rank's KV pool while the operator
sees his flag in ``ps`` and in the ServerArgs repr. That is #894 S5's shape
one module down, and #894 pinned it as the single ``KNOWN_SILENT`` entry of
its supersession ratchet rather than fixing it there.

WHAT CHANGES, AND WHAT DOES NOT
-------------------------------
The PRECEDENCE does not move. It is documented in the flag's own help text
("The environment variable SGLANG_UNEVEN_TOKEN_VECTOR (explicit vector) takes
precedence over this flag") and the env is how the post-profiling calibration
feeds its measured optimum back in. Flipping it would change which vector
serves; refusing the combination would kill a boot on every process carrying
the variable, the in-process writeback path included. The silence ends; the
rule stays.

WHERE THE LINE LIVES
--------------------
NOT inside ``resolve_cp_token_ratios``. That function's own docstring calls it
a deterministic pure function of the args -- every rank must derive the same
vector for the pool pinning and the owner rule to agree -- and it has several
callers, direct unit calls among them. ``announce_superseded_rank_kv_ratio``
sits beside it in the same module, so the rule and the announcement cannot
drift apart, and is CALLED once per process from the boot-time site that
installs the vector (``scheduler.configure_scheduler_process``). Every
behavioural guard below therefore drives the REAL boot function, not the
announcer in isolation: a line that only fires in its own unit test is the
failure this ticket is about.

THE REMEDY IT NAMES
-------------------
Remove the variable, never blank it. ``server_args.py:5607`` records what an
empty override already cost: SGLANG_UNEVEN_TOKEN_VECTOR set, then silently
cleared by a later empty append, uneven token sharding off for a day with
nobody aware. A message that suggested ``SGLANG_UNEVEN_TOKEN_VECTOR=`` would
be advising the next instance of that.

CPU only: the boot function is driven with a stub server_args and stopped at
``setproctitle``, the first statement after the block under test. No device,
no process group, no model.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

# THE REAL BOOT FUNCTION -- not a re-implementation of its gate.
import sglang.srt.distributed.utils as du  # noqa: E402
from sglang.srt.managers import scheduler as sched_mod  # noqa: E402

_TOKVEC = "SGLANG_UNEVEN_TOKEN_VECTOR"
_ROLE = "SGLANG_UNEVEN_TOKEN_VECTOR_ROLE"
_WEIGHTED = "SGLANG_UNEVEN_DCP_WEIGHTED"
LOGGER = "sglang.srt.distributed.utils"


class _ReachedProcTitle(Exception):
    """Raised in place of setproctitle: the boot got past the block."""


def _server_args(*, kv_ratio, plan=(2, 1), dcp_size=2, tp_size=2, role="pin"):
    sa = SimpleNamespace(
        rank_tp_ratio=list(plan) if plan is not None else None,
        rank_mlp_ratio=None,
        rank_moe_ratio=None,
        rank_vocab_ratio=None,
        rank_kv_ratio=kv_ratio,
        rank_kv_capacity_seed=None,
        rank_gpu_memory_mib=None,
        rank_gpu_id=None,
        uneven_token_vector_role=role,
        uneven_token_vector_provenance=None,
        model_path=None,
        dcp_size=dcp_size,
        tp_size=tp_size,
        pp_size=1,
        attn_cp_size=1,
        moe_dp_size=1,
        ep_size=1,
        nnodes=1,
        weightless_kv_fastlane=False,
    )
    # A non-'coupled' --rank-kv-ratio implies the weighted-DCP owner rule
    # without the env pair; that is what ServerArgs.uneven_weighted_dcp_enabled
    # answers on a real boot.
    sa.uneven_weighted_dcp_enabled = lambda: True
    sa.world_rank = lambda pp_rank, tp_rank: pp_rank * tp_size + tp_rank
    sa.apply_rank_memory_budget = lambda rank: None
    return sa


def _boot(sa):
    """Run the real ``configure_scheduler_process`` up to ``setproctitle``."""

    def _boom(*a, **kw):
        raise _ReachedProcTitle

    with mock.patch.object(sched_mod, "kill_itself_when_parent_died", lambda: None):
        with mock.patch.object(sched_mod.setproctitle, "setproctitle", _boom):
            try:
                sched_mod.configure_scheduler_process(
                    sa,
                    gpu_id=0,
                    tp_rank=0,
                    attn_cp_rank=0,
                    moe_dp_rank=0,
                    moe_ep_rank=0,
                    pp_rank=0,
                    dp_rank=0,
                )
            except _ReachedProcTitle:
                pass
    return du.get_cp_token_ratios()


class _Base(CustomTestCase):
    def setUp(self):
        self._saved_vec = du.get_cp_token_ratios()
        self._saved_plan = du.get_tp_partition_ratios()
        du.set_cp_token_ratios(None)
        du.reset_kv_ratio_supersession_announcement()
        self.addCleanup(du.reset_kv_ratio_supersession_announcement)

    def tearDown(self):
        du.set_tp_partition_ratios(self._saved_plan)
        du.set_cp_token_ratios(self._saved_vec)

    def _env(self, **values):
        """Set exactly the uneven-token env, clearing what is not named."""
        ctx = mock.patch.dict(os.environ, {}, clear=False)
        ctx.start()
        self.addCleanup(ctx.stop)
        for name in (_TOKVEC, _ROLE, _WEIGHTED):
            os.environ.pop(name, None)
        for name, value in values.items():
            if value is not None:
                os.environ[name] = value

    def _boot_lines(self, sa):
        with self.assertLogs(LOGGER, level="WARNING") as cap:
            vector = _boot(sa)
        return vector, [r.getMessage() for r in cap.records]

    def _announcement(self, sa):
        _, lines = self._boot_lines(sa)
        hits = [line for line in lines if "#897" in line]
        self.assertEqual(len(hits), 1, f"expected exactly one #897 line, got {lines}")
        return hits[0]


class TestTheLossIsAnnounced(_Base):
    """RED AT BASE: nothing on the boot path says the flag lost."""

    def test_an_explicit_pin_beaten_by_the_env_is_announced(self):
        self._env(**{_TOKVEC: "7,3", _WEIGHTED: "1"})
        msg = self._announcement(_server_args(kv_ratio=[3, 7]))
        # Both values, so the reader can see WHICH vector he lost.
        self.assertIn("'7,3'", msg)
        self.assertIn("--rank-kv-ratio 3,7", msg)
        # Who won.
        self.assertIn(_TOKVEC, msg)
        # Why: the presence rule, not a comparison.
        self.assertIn("PRESENCE", msg)
        self.assertIn("INERT", msg)

    def test_the_line_names_the_installed_vector_and_the_role(self):
        self._env(**{_TOKVEC: "14,6", _WEIGHTED: "1"})
        sa = _server_args(kv_ratio=[3, 7])
        msg = self._announcement(sa)
        # The gcd-reduced form is what actually gets installed; naming the raw
        # env string alone would leave the reader comparing different shapes.
        self.assertIn("7,3", msg)
        self.assertIn("'pin'", msg)
        self.assertEqual(du.get_cp_token_ratios(), [7, 3])

    def test_the_boot_still_installs_the_env_vector(self):
        """The announcement changes nothing about who wins."""
        self._env(**{_TOKVEC: "7,3", _WEIGHTED: "1"})
        vector, _ = self._boot_lines(_server_args(kv_ratio=[3, 7]))
        self.assertEqual(vector, [7, 3])


class TestADerivedModeSaysWhichPhasesItLost(_Base):
    def test_a_pinned_env_vector_makes_the_mode_inert_in_both_phases(self):
        self._env(**{_TOKVEC: "7,3", _ROLE: "pin", _WEIGHTED: "1"})
        msg = self._announcement(_server_args(kv_ratio="capacity"))
        self.assertIn("--rank-kv-ratio capacity", msg)
        self.assertIn("both phases", msg)
        # The second half of the loss lives in another module; name it, or the
        # reader concludes the measured install will still save him.
        self.assertIn("pinned_vector", msg)

    def test_a_seeded_env_vector_is_reported_as_late_not_lost(self):
        """role='seed' still supersedes in-process (#797), so do not claim the
        mode never arrives -- an over-stated warning is its own defect."""
        self._env(**{_TOKVEC: "7,3", _ROLE: "seed", _WEIGHTED: "1"})
        msg = self._announcement(_server_args(kv_ratio="capacity", role="seed"))
        self.assertIn("supersedes", msg)
        self.assertNotIn("both phases", msg)


class TestItStaysQuietWhenNothingWasLost(_Base):
    """A line that also fires when nothing was lost is a line readers skip."""

    def test_an_env_vector_equal_to_the_pin_is_not_reported(self):
        """Equal after the gcd reduction both sides go through."""
        self._env(**{_TOKVEC: "6,2", _WEIGHTED: "1"})
        with self.assertNoLogs(LOGGER, level="WARNING"):
            self.assertEqual(_boot(_server_args(kv_ratio=[3, 1])), [3, 1])

    def test_the_default_coupled_ratio_is_not_reported(self):
        """'coupled' asks for exactly the env-gated behaviour."""
        self._env(**{_TOKVEC: "7,3", _WEIGHTED: "1"})
        with self.assertNoLogs(LOGGER, level="WARNING"):
            self.assertEqual(_boot(_server_args(kv_ratio="coupled")), [7, 3])

    def test_no_env_vector_says_nothing(self):
        self._env(**{_WEIGHTED: "1"})
        with self.assertNoLogs(LOGGER, level="WARNING"):
            self.assertEqual(_boot(_server_args(kv_ratio=[3, 1])), [3, 1])

    def test_it_is_said_once_per_process(self):
        self._env(**{_TOKVEC: "7,3", _WEIGHTED: "1"})
        _, first = self._boot_lines(_server_args(kv_ratio=[3, 7]))
        self.assertEqual(len([m for m in first if "#897" in m]), 1)
        with self.assertNoLogs(LOGGER, level="WARNING"):
            _boot(_server_args(kv_ratio=[3, 7]))


class TestTheRemedyAndTheRefusalBoundary(_Base):
    def test_the_remedy_never_advises_an_empty_override(self):
        """The uneven-distribution law forbids empty env overrides, and
        server_args.py:5607 records the day one cost."""
        self._env(**{_TOKVEC: "7,3", _WEIGHTED: "1"})
        msg = self._announcement(_server_args(kv_ratio=[3, 7]))
        self.assertIn("REMOVE SGLANG_UNEVEN_TOKEN_VECTOR", msg)
        self.assertIn("not by setting it to an empty string", msg)

    def test_a_malformed_env_vector_is_left_to_the_existing_refusal(self):
        """Wrong length: the resolver raises, naming the variable and the
        shape. A second quieter line here would compete with the loud one."""
        self._env(**{_TOKVEC: "7,3,1", _WEIGHTED: "1"})
        with self.assertNoLogs(LOGGER, level="WARNING"):
            with self.assertRaises(ValueError) as ei:
                _boot(_server_args(kv_ratio=[3, 7]))
        self.assertIn(_TOKVEC, str(ei.exception))
        self.assertNotIn("#897", str(ei.exception))

    def test_a_vector_without_a_plan_still_raises_unchanged(self):
        """#182's honesty guard is untouched; the announcer must not swallow
        it by returning early on a shape it cannot describe."""
        self._env(**{_TOKVEC: "7,3", _WEIGHTED: "1"})
        with self.assertRaises(ValueError) as ei:
            _boot(_server_args(kv_ratio="coupled", plan=None))
        self.assertIn("silently ignored", str(ei.exception))


class TestTheResolverStaysPure(_Base):
    """The announcement belongs at the boot site, not in the resolver.

    ``resolve_cp_token_ratios`` is documented as a deterministic pure function
    called from several sites; a logger inside it would fire per call and per
    rank. This is the guard that keeps a later author from "simplifying" the
    fix by moving the line one function down.
    """

    def _args(self, **kw):
        base = dict(kv_ratio=[3, 7])
        base.update(kw)
        return _server_args(**base)

    def test_the_resolver_logs_nothing_on_the_superseding_path(self):
        self._env(**{_TOKVEC: "7,3", _WEIGHTED: "1"})
        with self.assertNoLogs(LOGGER, level="WARNING"):
            self.assertEqual(du.resolve_cp_token_ratios(self._args()), [7, 3])

    def test_precedence_is_unchanged(self):
        """env > explicit pin > seed, pinned over the shapes this fix touches."""
        cases = (
            # (env, rank_kv_ratio, seed, expected)
            ("7,3", [3, 7], None, [7, 3]),
            ("14,6", [3, 7], None, [7, 3]),
            ("6,2", [3, 1], None, [3, 1]),
            ("5,5", [3, 7], None, None),
            (None, [3, 7], None, [3, 7]),
            (None, [6, 14], None, [3, 7]),
            (None, [4, 4], None, None),
            (None, "coupled", [9, 3], [3, 1]),
            ("7,3", "coupled", [9, 3], [7, 3]),
        )
        for env, kv_ratio, seed, expected in cases:
            with self.subTest(env=env, kv_ratio=kv_ratio, seed=seed):
                self._env(**{_TOKVEC: env, _WEIGHTED: "1"})
                sa = self._args(kv_ratio=kv_ratio)
                sa.rank_kv_capacity_seed = seed
                self.assertEqual(
                    du.resolve_cp_token_ratios(sa, checkpoint_size_mib=0), expected
                )


if __name__ == "__main__":
    unittest.main()
