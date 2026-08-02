"""Consumability, demonstrated rather than asserted (#407 slice 1).

#421 F6 found this package with a complete API, 82 passing tests and zero
callers, and the catalog claiming "all consumers pick targets from it" as if it
were true. The lesson is not "wire a consumer now" -- the cut plan says
otherwise -- it is that a package nobody has ever *called from outside* has an
unmeasured gap between what it offers and what a caller needs.

Two hooks are exercised here, both read-only, both from outside the package's
own vocabulary:

1. :func:`expert_offload_host_targets` -- the #77/#123 question, asked the way
   expert offload can ask it: bytes, origin card, nothing else. Proves a caller
   needs no tier id, no host name, no profile and no probe to get an answer,
   and that the failure path is a sentence rather than an empty list.

2. :func:`link_disjointness` -- the #423 striping gate. Striping only pays when
   two moves do not queue behind one wire, and this rig already carries the
   counterexample: three RDMA pairs in parallel cost 2.34x the latency for
   1.28x the aggregate. The asymmetry between the three verdicts is the whole
   point and is falsified below in all three directions.

    python -m pytest test/registered/unit/memtier/test_tier_consumer_shim.py -v
"""

import unittest

from sglang.srt.memtier.consumers import expert_offload_host_targets
from sglang.srt.memtier.registry import TierRegistry
from sglang.srt.memtier.tiers import (
    LinkVerdict,
    TierCapacity,
    TierCaps,
    TierDescriptor,
    TierHealth,
    TierKind,
    TierTransport,
    Volatility,
    link_disjointness,
)
from sglang.srt.planner.cost_model import Rate
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GIB = 1024**3
CARD = "GPU-abcdabcd-0000-0000-0000-000000000001"


def caps(bandwidth=None, ledger_key="k", link=(), complete=False):
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
    link=(),
    complete=False,
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
        transport=TierTransport(
            name="unit", link_path=tuple(link), link_path_complete=complete
        ),
        properties=dict(properties or {}),
        profile_id="unit",
    )


HOST = tier(
    "host:unit",
    TierKind.HOST,
    volatility=Volatility.EXPENSIVE_OK,
    admits=("experts", "cold_lane"),
    total=128 * GIB,
    bandwidth=40.0,
)
NVME = tier(
    "fs:unit:/fast",
    TierKind.FILESYSTEM,
    volatility=Volatility.PERSISTENT,
    admits=("experts",),
    total=2000 * GIB,
    bandwidth=2.0,
    properties={"medium": "nvme"},
)
VRAM = tier(
    f"vram:{CARD}",
    TierKind.DEVICE,
    volatility=Volatility.DEVICE_BOUND_ONLY,
    admits=("experts",),
    total=24 * GIB,
    bandwidth=900.0,
)


def registry(*tiers):
    return TierRegistry(list(tiers), profile_id="unit", local_host="unit")


class TestExpertOffloadShim(unittest.TestCase):
    def test_a_caller_with_bytes_and_an_origin_gets_an_ordered_answer(self):
        answer = expert_offload_host_targets(
            registry(HOST, NVME, VRAM), bytes_needed=8 * GIB, origin=VRAM.id
        )
        self.assertTrue(answer.ok)
        self.assertEqual(answer.tier_id, "host:unit")
        self.assertEqual(answer.alternatives, ("fs:unit:/fast",))

    def test_device_tiers_are_out_of_scope_for_this_question(self):
        answer = expert_offload_host_targets(
            registry(HOST, VRAM), bytes_needed=1 * GIB, origin=VRAM.id
        )
        self.assertNotIn(VRAM.id, (answer.tier_id, *answer.alternatives))
        refusal = answer.selection.refusal_for(VRAM.id)
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal.rule.value, "kind")

    def test_a_tier_that_does_not_admit_experts_is_refused_by_name(self):
        no_experts = tier(
            "host:other",
            TierKind.HOST,
            volatility=Volatility.EXPENSIVE_OK,
            admits=("kv_shadow",),
            bandwidth=40.0,
        )
        answer = expert_offload_host_targets(registry(no_experts), bytes_needed=1 * GIB)
        self.assertFalse(answer.ok)
        self.assertEqual(
            answer.selection.refusal_for("host:other").rule.value, "object_class"
        )
        self.assertIn("kv_shadow", answer.refusal)

    def test_a_request_that_does_not_fit_is_refused_with_the_arithmetic(self):
        answer = expert_offload_host_targets(registry(HOST), bytes_needed=1000 * GIB)
        self.assertFalse(answer.ok)
        self.assertIn("bytes were asked for", answer.refusal)
        self.assertIn("total", answer.refusal)

    def test_an_empty_rig_refuses_with_a_sentence_not_an_empty_list(self):
        answer = expert_offload_host_targets(registry(), bytes_needed=1)
        self.assertFalse(answer.ok)
        self.assertTrue(answer.refusal)
        self.assertEqual(answer.alternatives, ())

    def test_requiring_a_measurement_refuses_an_unmeasured_tier(self):
        unmeasured = tier(
            "host:unmeasured",
            TierKind.HOST,
            volatility=Volatility.EXPENSIVE_OK,
            admits=("experts",),
        )
        loose = expert_offload_host_targets(registry(unmeasured), bytes_needed=1 * GIB)
        strict = expert_offload_host_targets(
            registry(unmeasured), bytes_needed=1 * GIB, require_measured_bandwidth=True
        )
        # can-fail proof: the same rig answers one way and refuses the other.
        self.assertTrue(loose.ok)
        self.assertFalse(strict.ok)
        self.assertEqual(
            strict.selection.refusal_for("host:unmeasured").rule.value,
            "bandwidth_absent",
        )
        self.assertIn("bandwidth is absent", strict.refusal)

    def test_a_blocked_tier_never_becomes_a_target(self):
        blocked = tier(
            "host:blocked",
            TierKind.HOST,
            volatility=Volatility.EXPENSIVE_OK,
            admits=("experts",),
            bandwidth=40.0,
            verdict="block",
            reason="the wire has never been demonstrated",
        )
        answer = expert_offload_host_targets(registry(blocked), bytes_needed=1 * GIB)
        self.assertFalse(answer.ok)
        self.assertIn("never been demonstrated", answer.refusal)

    def test_the_answer_is_serialisable_for_an_error_message(self):
        answer = expert_offload_host_targets(registry(HOST), bytes_needed=1 * GIB)
        payload = answer.to_json()
        self.assertEqual(payload["tier_id"], "host:unit")
        self.assertIn("candidates", payload["selection"])


class TestLinkDisjointness(unittest.TestCase):
    """#423's gate: three verdicts, and the asymmetry between them."""

    def test_a_shared_segment_is_shared_even_from_partial_paths(self):
        a = tier(
            "host:a",
            TierKind.HOST,
            volatility=Volatility.EXPENSIVE_OK,
            link=("nic:wire0",),
            complete=False,
        )
        b = tier(
            "fs:unit:/b",
            TierKind.FILESYSTEM,
            volatility=Volatility.PERSISTENT,
            link=("nic:wire0", "blk:disk0"),
            complete=False,
        )
        answer = link_disjointness(a, b)
        self.assertIs(answer.verdict, LinkVerdict.SHARED)
        self.assertEqual(answer.shared, ("nic:wire0",))

    def test_disjointness_requires_both_paths_to_be_complete(self):
        partial_a = tier(
            "host:a",
            TierKind.HOST,
            volatility=Volatility.EXPENSIVE_OK,
            link=("pcie:0000:07:00.0",),
            complete=False,
        )
        partial_b = tier(
            "host:b",
            TierKind.HOST,
            volatility=Volatility.EXPENSIVE_OK,
            link=("pcie:0000:0b:00.0",),
            complete=False,
        )
        self.assertIs(
            link_disjointness(partial_a, partial_b).verdict, LinkVerdict.UNKNOWN
        )
        # can-fail proof: mark both complete and the SAME pair is disjoint.
        full_a = partial_a.evolve(
            transport=TierTransport(
                name="unit",
                link_path=("pcie:0000:07:00.0", "root:0000:00:01.1"),
                link_path_complete=True,
            )
        )
        full_b = partial_b.evolve(
            transport=TierTransport(
                name="unit",
                link_path=("pcie:0000:0b:00.0", "root:0000:00:03.1"),
                link_path_complete=True,
            )
        )
        self.assertIs(link_disjointness(full_a, full_b).verdict, LinkVerdict.DISJOINT)

    def test_an_unrecorded_path_is_unknown_and_names_the_tier(self):
        recorded = tier(
            "host:a",
            TierKind.HOST,
            volatility=Volatility.EXPENSIVE_OK,
            link=("nic:wire0",),
            complete=True,
        )
        blank = tier("host:b", TierKind.HOST, volatility=Volatility.EXPENSIVE_OK)
        answer = link_disjointness(recorded, blank)
        self.assertIs(answer.verdict, LinkVerdict.UNKNOWN)
        self.assertIn("host:b", answer.reason)
        self.assertIn("not a 'no'", answer.reason)

    def test_a_complete_path_that_converges_upstream_is_shared(self):
        a = tier(
            "host:a",
            TierKind.HOST,
            volatility=Volatility.EXPENSIVE_OK,
            link=("pcie:0000:07:00.0", "root:0000:00:01.1"),
            complete=True,
        )
        b = tier(
            "host:b",
            TierKind.HOST,
            volatility=Volatility.EXPENSIVE_OK,
            link=("pcie:0000:0b:00.0", "root:0000:00:01.1"),
            complete=True,
        )
        answer = link_disjointness(a, b)
        self.assertIs(answer.verdict, LinkVerdict.SHARED)
        self.assertEqual(answer.shared, ("root:0000:00:01.1",))

    def test_the_verdict_survives_the_json_round_trip(self):
        a = tier(
            "host:a",
            TierKind.HOST,
            volatility=Volatility.EXPENSIVE_OK,
            link=("nic:wire0",),
            complete=True,
        )
        payload = link_disjointness(a, a).to_json()
        self.assertEqual(payload["verdict"], "shared")
        self.assertEqual(payload["shared"], ["nic:wire0"])

    def test_the_transport_record_carries_the_path_through_json(self):
        a = tier(
            "host:a",
            TierKind.HOST,
            volatility=Volatility.EXPENSIVE_OK,
            link=("nic:wire0",),
            complete=True,
        )
        payload = a.to_json()["transport"]
        self.assertEqual(payload["link_path"], ["nic:wire0"])
        self.assertTrue(payload["link_path_complete"])


if __name__ == "__main__":
    unittest.main()
