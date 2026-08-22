"""The seed-liveness gate against the ACTUAL historical defect (#797).

The latch's own semantics are unit-tested under
test/registered/unit/distributed/test_seed_liveness_797.py. What this suite
proves is the part that matters operationally: driving the REAL sizing method,
the gate stays silent on a boot whose seed was superseded and refuses one whose
seed was not.

THE EXECUTED CAN-FAIL PROOF, both directions, on the boot-0629 configuration
(the flip TP stack: ``is_draft_worker=True``, ``is_draft_pool_worker=False``):

  predicate ``not self.is_draft_worker``      (pre-fix)
      -> installed vector [29, 19, 16]  -> gate FIRED
  predicate ``not self.is_draft_pool_worker`` (shipped)
      -> installed vector [29, 17, 18]  -> gate silent

So the gate would have refused the boot that shipped, and admits the boot that
is fixed. That pair is the whole claim of this file; it cannot be expressed as
a single test, because reproducing the broken boot requires the broken
predicate, so it is recorded here and was run by hand.

Neutering ``assert_seed_superseded`` to return unconditionally turns
test_a_lone_declining_site_is_refused red. Verified.
"""

from test_uneven_token_vector_role_797 import _run

from sglang.srt.distributed.utils import (
    assert_seed_superseded,
    note_seed_awaiting_supersession,
    reset_seed_liveness,
)
from sglang.srt.planner.retracted import SeedNotSupersededError
from sglang.test.test_utils import CustomTestCase

# The capacities boot 0646 actually profiled, and the vector that rode.
CAPS = [620560, 360392, 375560]
SEED = [29, 19, 16]
MEASURED = [29, 17, 18]


class SeedLivenessEndToEnd797(CustomTestCase):
    def setUp(self):
        reset_seed_liveness()
        self.addCleanup(reset_seed_liveness)

    def test_the_flip_tp_runner_supersedes_the_seed(self):
        """Boot 0629's configuration, under the shipped predicate.

        A draft runner on the flip TP stack owns the real KV pool, so it
        installs, and the seed's claim comes true. The gate must be silent --
        if it were not, every flip boot would be refused.
        """
        note_seed_awaiting_supersession(SEED)
        installed, _, _ = _run(
            CAPS,
            SEED,
            role="seed",
            env_vector="29,19,16",
            draft_worker=True,
            draft_pool_worker=False,
        )
        self.assertEqual(list(installed), MEASURED)
        assert_seed_superseded()

    def test_a_lone_declining_site_is_refused(self):
        """THE FALSIFIER: the only install-capable site in the process declined
        and nothing else superseded the seed, so the pools were built from an
        estimate that still called itself provisional."""
        note_seed_awaiting_supersession(SEED)
        installed, _, _ = _run(
            CAPS, SEED, role="seed", env_vector="29,19,16", draft_pool_worker=True
        )
        self.assertEqual(list(installed), SEED)
        with self.assertRaises(SeedNotSupersededError) as ctx:
            assert_seed_superseded()
        # Name the value that rode, or the operator cannot tell which of
        # several vectors in the boot was the broken promise.
        self.assertIn("29, 19, 16", str(ctx.exception))

    def test_a_genuine_draft_pool_worker_does_not_trip_the_gate(self):
        """The exclusion that SURVIVED the fix must not become a false alarm.

        A real draft-pool worker shares the target's pool geometry and rightly
        installs nothing. Its target runs in the SAME process and does install,
        which is why the latch is process-global: the target's verdict disarms
        the claim for both. Order is deliberate -- the decline comes first, so
        a latch that could not be disarmed after a decline would fail here.
        """
        note_seed_awaiting_supersession(SEED)
        _run(CAPS, SEED, role="seed", env_vector="29,19,16", draft_pool_worker=True)
        _run(CAPS, SEED, role="seed", env_vector="29,19,16", draft_pool_worker=False)
        assert_seed_superseded()

    def test_a_pinned_boot_is_never_held_to_a_claim_it_did_not_make(self):
        """No arming, so no refusal, even though the pin suppresses the install
        exactly the way the broken seed did. Role is the whole difference."""
        _run(CAPS, SEED, role="pin", env_vector="29,19,16")
        assert_seed_superseded()
