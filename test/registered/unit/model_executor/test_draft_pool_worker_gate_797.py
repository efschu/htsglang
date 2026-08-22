"""#797, the last gate: the install exclusion must ask about the POOL, not
about the worker's label.

``ModelRunner`` defines the pair at model_runner.py:512::

    self.is_draft_pool_worker = is_draft_worker and not is_phase_flip_tp_stack

so the two flags COINCIDE everywhere except on the phase-flip TP stack, where
a draft runner is a draft worker whose POOLS take the target-model treatment.
Every case in the sibling suite lives in the coinciding region, which is why
this defect could not be seen there: the divergence is the whole bug.

Measured, boot_798_0822_0629.log. At the TP-stack sizing site all three ranks
logged ``allow_install=True role='seed' dcp_size=3
active_vector=[29, 19, 16]`` and no SKIP -- the calibration ran to completion,
computed the better vector, and still printed "restart with
SGLANG_UNEVEN_TOKEN_VECTOR=30,17,17". The PINNED-VECTOR warning did not fire,
so ``pinned_vector`` was False and ``seed_role`` True, leaving the draft
conjunct as the only one that could have been False.
"""

from test_uneven_token_vector_role_797 import _run

from sglang.test.test_utils import CustomTestCase

# Deliberately unequal, so the capacity-proportional optimum differs from the
# active vector and an install has something to do.
CAPS = [620560, 360392, 375560]
ACTIVE = [29, 19, 16]


class FlipTpDraftRunnerInstalls797(CustomTestCase):
    """THE falsifier. Before the fix this asserted-on install did not happen;
    the method printed a restart hint instead.

    CanFail: change the gate back to ``not self.is_draft_worker`` and
    test_the_flip_tp_draft_runner_installs goes red. Verified."""

    def test_the_flip_tp_draft_runner_installs(self):
        # is_draft_worker True, is_draft_pool_worker False -- exactly the flip
        # TP stack, and exactly the runner that owns the real KV pool.
        installed, _warnings, _env = _run(
            CAPS,
            ACTIVE,
            role="seed",
            env_vector="29,19,16",
            draft_worker=True,
            draft_pool_worker=False,
        )
        self.assertNotEqual(
            list(installed),
            ACTIVE,
            "the runner that resolved the pool config was refused the install",
        )

    def test_it_installs_the_capacity_proportional_vector(self):
        installed, _w, _e = _run(
            CAPS,
            ACTIVE,
            role="seed",
            env_vector="29,19,16",
            draft_worker=True,
            draft_pool_worker=False,
        )
        # Proportional to measured capacity, gcd-reduced, summing to 64 units.
        self.assertEqual(sum(installed), 64)
        # The binding rank under the active vector must stop binding: the
        # whole point is that no rank is left idle.
        world_before = min(c // r for c, r in zip(CAPS, ACTIVE)) * sum(ACTIVE)
        world_after = min(c // r for c, r in zip(CAPS, installed)) * sum(installed)
        self.assertGreater(world_after, world_before)


class ARealDraftPoolWorkerStillDoesNotInstall797(CustomTestCase):
    """The exclusion is narrowed, not removed. A runner with its OWN draft
    pool must still never install the target's vector -- its capacity is not
    the target's capacity.

    CanFail: drop the conjunct entirely and this goes red."""

    def test_a_draft_pool_worker_is_refused(self):
        installed, _w, _e = _run(
            CAPS,
            ACTIVE,
            role="seed",
            env_vector="29,19,16",
            draft_worker=True,
            draft_pool_worker=True,
        )
        # "No install" leaves the ACTIVE vector standing; the helper reports
        # the installed vector, which is then simply the one it started with.
        self.assertEqual(list(installed), ACTIVE)

    def test_the_non_flip_case_is_unchanged(self):
        # Where the two flags coincide, behaviour must be byte-identical to
        # before: a plain draft worker is still excluded.
        installed, _w, _e = _run(
            CAPS,
            ACTIVE,
            role="seed",
            env_vector="29,19,16",
            draft_worker=True,
        )
        # "No install" leaves the ACTIVE vector standing; the helper reports
        # the installed vector, which is then simply the one it started with.
        self.assertEqual(list(installed), ACTIVE)


class TheTargetRunnerIsUnaffected797(CustomTestCase):
    def test_a_plain_target_runner_still_installs(self):
        installed, _w, _e = _run(
            CAPS,
            ACTIVE,
            role="seed",
            env_vector="29,19,16",
            draft_worker=False,
            draft_pool_worker=False,
        )
        self.assertNotEqual(list(installed), ACTIVE)

    def test_a_pin_is_still_never_installed(self):
        # The role split from 95fdc54009 must survive this change untouched.
        installed, _w, _e = _run(
            CAPS,
            ACTIVE,
            role="pin",
            env_vector="29,19,16",
            draft_worker=True,
            draft_pool_worker=False,
        )
        # "No install" leaves the ACTIVE vector standing; the helper reports
        # the installed vector, which is then simply the one it started with.
        self.assertEqual(list(installed), ACTIVE)
