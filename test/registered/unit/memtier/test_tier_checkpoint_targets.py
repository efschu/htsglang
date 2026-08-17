"""The #407 registry's WRITE-PATH selection helper, and its first consumer.

``checkpoint_tier_targets`` is what cut 3 owes and what #410 calls. Cut 1's
shim was read-only and exercised only by tests; this one is reached from
``managers/session_checkpoint.py`` on a real control request, which is why the
#421 "zero consumers" pin is retired in the same merge.

What is proved here, each with the can-fail half:

*   the VRAM -> RAM -> Disk ladder is driven by AGE and narrows the kinds
    considered -- the same rig answers "the card" for a fresh checkpoint and
    "not the card" for an old one;
*   ``durable`` promotes the payload class to ``PERSISTENCE_REQUIRED``, which
    the volatility table admits only on a ``PERSISTENT`` tier. Host RAM is
    therefore refused BY NAME for a durable checkpoint and admitted for a
    non-durable one -- the same tier, the same rig, one flag apart;
*   a rig where nothing is admissible produces the itemised sentence, not an
    empty list;
*   no ``object_class`` is passed, so a tier does not have to declare a
    checkpoint-shaped ``OFFLOAD_CLASSES`` member to hold one. A tier that
    admits nothing at all still admits checkpoints.

    python -m pytest test/registered/unit/memtier/test_tier_checkpoint_targets.py -v
"""

import unittest

from sglang.srt.memtier.consumers import CheckpointTierPolicy, checkpoint_tier_targets
from sglang.srt.memtier.registry import TierRegistry
from sglang.srt.memtier.tiers import (
    TierCapacity,
    TierCaps,
    TierDescriptor,
    TierHealth,
    TierKind,
    TierTransport,
    Volatility,
)
from sglang.srt.planner.cost_model import Rate
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GIB = 1024**3
CARD = "GPU-abcdabcd-0000-0000-0000-000000000001"


def caps(bandwidth=None, ledger_key="k"):
    return TierCaps(
        latency_us=Rate.absent("not measured", unit="us"),
        bandwidth_gbs=(
            Rate.measured(bandwidth, "unit fixture", unit="GB/s")
            if bandwidth is not None
            else Rate.absent("not measured", unit="GB/s")
        ),
        aperture_bytes=Rate.absent("not an aperture tier", unit="bytes"),
        ledger_key=ledger_key,
    )


def tier(
    tier_id,
    kind,
    *,
    volatility,
    admits=(),
    total=100 * GIB,
    floor=0,
    bandwidth=None,
    verdict="ok",
    reason="",
    properties=None,
):
    return TierDescriptor(
        id=tier_id,
        kind=kind,
        host="unit",
        capacity=TierCapacity(
            total=Rate.measured(float(total), "unit fixture", unit="bytes"),
            floor=Rate.measured(float(floor), "unit fixture", unit="bytes"),
        ),
        volatility=volatility,
        admits=frozenset(admits),
        caps=caps(bandwidth, ledger_key=tier_id.split(":")[0]),
        health=TierHealth(reachable=verdict != "block", verdict=verdict, reason=reason),
        transport=TierTransport(name="unit"),
        properties=dict(properties or {}),
        profile_id="unit",
    )


VRAM = tier(
    f"vram:{CARD}",
    TierKind.DEVICE,
    volatility=Volatility.DEVICE_BOUND_ONLY,
    total=24 * GIB,
    bandwidth=900.0,
)
HOST = tier(
    "host:unit",
    TierKind.HOST,
    volatility=Volatility.EXPENSIVE_OK,
    total=128 * GIB,
    bandwidth=40.0,
)
NVME = tier(
    "fs:unit:/fast",
    TierKind.FILESYSTEM,
    volatility=Volatility.PERSISTENT,
    total=2000 * GIB,
    bandwidth=2.0,
    properties={"medium": "nvme"},
)


def registry(*tiers):
    return TierRegistry(list(tiers), profile_id="unit", local_host="unit")


POLICY = CheckpointTierPolicy(vram_max_age_s=60.0, host_max_age_s=900.0)


class TestCheckpointLadder(unittest.TestCase):
    def test_a_fresh_checkpoint_may_rest_on_the_card_it_came_off(self):
        answer = checkpoint_tier_targets(
            registry(VRAM, HOST, NVME),
            bytes_needed=1 * GIB,
            age_s=0.0,
            origin=VRAM.id,
            policy=POLICY,
        )
        self.assertTrue(answer.ok, answer.refusal)
        self.assertEqual(answer.tier_id, VRAM.id)
        self.assertEqual(answer.alternatives, ("host:unit", "fs:unit:/fast"))

    def test_an_aged_checkpoint_stops_competing_for_vram(self):
        # can-fail proof: the SAME rig and the SAME bytes, one dial apart.
        aged = checkpoint_tier_targets(
            registry(VRAM, HOST, NVME),
            bytes_needed=1 * GIB,
            age_s=120.0,
            origin=VRAM.id,
            policy=POLICY,
        )
        self.assertTrue(aged.ok, aged.refusal)
        self.assertEqual(aged.tier_id, "host:unit")
        self.assertNotIn(VRAM.id, (aged.tier_id, *aged.alternatives))
        self.assertEqual(aged.selection.refusal_for(VRAM.id).rule.value, "kind")

    def test_an_old_checkpoint_must_be_on_something_that_survives_exit(self):
        old = checkpoint_tier_targets(
            registry(VRAM, HOST, NVME),
            bytes_needed=1 * GIB,
            age_s=5000.0,
            policy=POLICY,
        )
        self.assertTrue(old.ok, old.refusal)
        self.assertEqual(old.tier_id, "fs:unit:/fast")
        self.assertNotIn("host:unit", (old.tier_id, *old.alternatives))

    def test_the_ladder_is_ordered_vram_ram_disk_by_the_registrys_own_key(self):
        answer = checkpoint_tier_targets(
            registry(NVME, HOST, VRAM), bytes_needed=1 * GIB, policy=POLICY
        )
        self.assertEqual(
            [answer.tier_id, *answer.alternatives],
            [VRAM.id, "host:unit", "fs:unit:/fast"],
        )


class TestDurability(unittest.TestCase):
    def test_durable_refuses_host_ram_by_name(self):
        durable = checkpoint_tier_targets(
            registry(HOST), bytes_needed=1 * GIB, durable=True, policy=POLICY
        )
        self.assertFalse(durable.ok)
        refusal = durable.selection.refusal_for("host:unit")
        self.assertEqual(refusal.rule.value, "volatility")
        self.assertIn("persistence_required", durable.refusal)
        # can-fail proof: the identical query without durable is admitted.
        loose = checkpoint_tier_targets(
            registry(HOST), bytes_needed=1 * GIB, durable=False, policy=POLICY
        )
        self.assertTrue(loose.ok)
        self.assertEqual(loose.tier_id, "host:unit")

    def test_durable_lands_on_the_persistent_tier(self):
        answer = checkpoint_tier_targets(
            registry(VRAM, HOST, NVME),
            bytes_needed=1 * GIB,
            durable=True,
            policy=POLICY,
        )
        self.assertTrue(answer.ok, answer.refusal)
        self.assertEqual(answer.tier_id, "fs:unit:/fast")

    def test_age_past_the_host_threshold_implies_durable(self):
        self.assertEqual(
            POLICY.payload_for(5000.0, durable=False).value, "persistence_required"
        )
        self.assertEqual(
            POLICY.payload_for(5.0, durable=False).value, "expensive_reconstructable"
        )


class TestRefusals(unittest.TestCase):
    def test_an_empty_rig_refuses_with_a_sentence_not_an_empty_list(self):
        answer = checkpoint_tier_targets(registry(), bytes_needed=1, policy=POLICY)
        self.assertFalse(answer.ok)
        self.assertTrue(answer.refusal)
        self.assertEqual(answer.alternatives, ())
        self.assertIn("session checkpoint", answer.refusal)

    def test_a_checkpoint_that_does_not_fit_is_refused_with_the_arithmetic(self):
        answer = checkpoint_tier_targets(
            registry(HOST), bytes_needed=10_000 * GIB, policy=POLICY
        )
        self.assertFalse(answer.ok)
        self.assertIn("bytes were asked for", answer.refusal)
        self.assertEqual(
            answer.selection.refusal_for("host:unit").rule.value, "capacity"
        )

    def test_a_blocked_tier_never_becomes_a_target(self):
        blocked = tier(
            "host:blocked",
            TierKind.HOST,
            volatility=Volatility.EXPENSIVE_OK,
            bandwidth=40.0,
            verdict="block",
            reason="the mount went read-only",
        )
        answer = checkpoint_tier_targets(
            registry(blocked), bytes_needed=1 * GIB, policy=POLICY
        )
        self.assertFalse(answer.ok)
        self.assertIn("read-only", answer.refusal)

    def test_an_unmeasured_link_is_usable_but_reported_as_unmeasured(self):
        unmeasured = tier(
            "host:unmeasured", TierKind.HOST, volatility=Volatility.EXPENSIVE_OK
        )
        answer = checkpoint_tier_targets(
            registry(unmeasured), bytes_needed=1 * GIB, policy=POLICY
        )
        self.assertTrue(answer.ok, answer.refusal)
        self.assertEqual(
            answer.selection.candidates[0].bandwidth_gbs.provenance.value, "absent"
        )
        self.assertTrue(
            any(
                "bandwidth is absent" in n for n in answer.selection.candidates[0].notes
            )
        )


class TestNoObjectClassIsRequired(unittest.TestCase):
    """``OFFLOAD_CLASSES`` has no member for KV pages, and ``TierQuery``
    documents that case: hibernate images and HiCache pages are gated by
    volatility alone. A tier that declares no ``admits`` set at all must
    therefore still be able to hold a checkpoint."""

    def test_a_tier_admitting_no_offload_class_still_holds_a_checkpoint(self):
        bare = tier(
            "host:bare",
            TierKind.HOST,
            volatility=Volatility.EXPENSIVE_OK,
            bandwidth=10.0,
        )
        self.assertEqual(bare.admits, frozenset())
        answer = checkpoint_tier_targets(
            registry(bare), bytes_needed=1 * GIB, policy=POLICY
        )
        self.assertTrue(answer.ok, answer.refusal)
        self.assertEqual(answer.tier_id, "host:bare")

    def test_the_answer_is_serialisable_for_an_error_message(self):
        answer = checkpoint_tier_targets(
            registry(HOST), bytes_needed=1 * GIB, policy=POLICY
        )
        payload = answer.to_json()
        self.assertEqual(payload["tier_id"], "host:unit")
        self.assertIn("candidates", payload["selection"])


if __name__ == "__main__":
    unittest.main()
