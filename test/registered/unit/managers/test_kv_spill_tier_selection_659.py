# SPDX-License-Identifier: Apache-2.0
"""The KV spill ladder asked of the #407 registry instead of asserted (#659).

Two properties carry this cut, and they pull in opposite directions on
purpose:

1.  **The default must not move.** One local host tier with headroom selects
    the local host tier -- which is what the rung does today, so the spill
    path is byte-identical.
2.  **Local-first must be DERIVABLE, not just declared.** #224 hardcodes
    "local is first" and cites a measurement in a comment. Here the order
    falls out of capacity and cost, and ``local_first_disagreement`` is the
    falsifier that says so when the two stop agreeing.

The third thing tested is the one the operator was bitten by twice: a tier
that is FULL must produce a NAMED refusal that degrades to the next tier or
declines cleanly -- never an unbudgeted pin.
"""

import unittest

from sglang.srt.managers.kv_spill_tier_selection import (
    KV_SPILL_PAYLOAD,
    choose_kv_spill_tier,
    kv_spill_registry,
    ladder_rows,
    local_first_disagreement,
    local_host_kv_tier,
    occupied_and_other_pinned,
    park_tier,
    refusal_report,
)
from sglang.srt.memtier.tiers import (
    PayloadClass,
    TierCapacity,
    TierKind,
    Volatility,
)
from sglang.srt.planner.cost_model import Rate
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GB = 10**9
LOCAL = "rig-1"


def _local(pool_gb=40, occupied_gb=0, other_gb=0, bandwidth=None):
    return local_host_kv_tier(
        host=LOCAL,
        pool_bytes=int(pool_gb * GB),
        occupied_bytes=int(occupied_gb * GB),
        other_pinned_bytes=int(other_gb * GB),
        bandwidth_gbs=bandwidth,
        profile_id="test",
    )


def _remote(total_gb=8, bandwidth_gbs=0.075, reachable=True, verdict="ok", reason=""):
    """The measured rig-2 RAM candidate, as this rig actually measures it."""
    return park_tier(
        tier_id="host:rig-2",
        kind=TierKind.HOST,
        host="rig-2",
        volatility=Volatility.EXPENSIVE_OK,
        capacity=TierCapacity(
            total=Rate.measured(int(total_gb * GB), "free -m on rig-2", unit="bytes"),
            floor=Rate.measured(0, "no pinned floor recorded", unit="bytes"),
        ),
        bandwidth_gbs=Rate.measured(
            bandwidth_gbs, "dd 1500 MiB through nc, s41", unit="GB/s"
        ),
        latency_us=Rate.measured(265.0, "ping -c 20, s41", unit="us"),
        transport_name="tcp",
        stages_through=f"host:{LOCAL}",
        reachable=reachable,
        verdict=verdict,
        reason=reason,
        profile_id="test",
    )


class TheDefaultIsByteIdentical(CustomTestCase):
    def test_one_local_tier_with_headroom_selects_the_local_tier(self):
        reg = kv_spill_registry([_local()], local_host=LOCAL)
        chosen, selection = choose_kv_spill_tier(reg, 1 * GB)
        self.assertEqual(chosen, f"host:{LOCAL}")
        self.assertEqual(selection.refusals, ())

    def test_a_ladder_without_a_local_tier_is_refused_at_construction(self):
        """Not a preference: a device copy has nowhere else to land."""
        with self.assertRaises(ValueError) as ctx:
            kv_spill_registry([_remote()], local_host=LOCAL)
        self.assertIn("only possible FIRST rung", str(ctx.exception))

    def test_the_payload_class_is_never_droppable(self):
        """Spilled KV costs a re-prefill to reconstruct -- #224's whole point."""
        self.assertIs(KV_SPILL_PAYLOAD, PayloadClass.EXPENSIVE_RECONSTRUCTABLE)

    def test_a_reconstructable_only_tier_may_not_hold_a_spilled_session(self):
        reg = kv_spill_registry(
            [
                _local(),
                park_tier(
                    tier_id="host:scratch",
                    kind=TierKind.HOST,
                    host="scratch",
                    volatility=Volatility.RECONSTRUCTABLE_OK,
                    capacity=TierCapacity(
                        total=Rate.measured(999 * GB, "huge", unit="bytes"),
                        floor=Rate.measured(0, "none", unit="bytes"),
                    ),
                    bandwidth_gbs=Rate.measured(99.0, "fake", unit="GB/s"),
                    latency_us=Rate.measured(1.0, "fake", unit="us"),
                    transport_name="tcp",
                    stages_through=f"host:{LOCAL}",
                ),
            ],
            local_host=LOCAL,
        )
        _, selection = choose_kv_spill_tier(reg, 1 * GB)
        refused = {r.tier_id: r for r in selection.refusals}
        self.assertIn("host:scratch", refused)
        self.assertEqual(refused["host:scratch"].rule.value, "volatility")


class ExhaustionIsANamedRefusal(CustomTestCase):
    """The operator was OOM-killed twice. A full tier must SAY it is full."""

    def test_a_full_local_tier_refuses_by_capacity_and_names_itself(self):
        reg = kv_spill_registry([_local(pool_gb=10, occupied_gb=10)], local_host=LOCAL)
        chosen, selection = choose_kv_spill_tier(reg, 2 * GB)
        self.assertIsNone(chosen)
        self.assertEqual(len(selection.refusals), 1)
        self.assertEqual(selection.refusals[0].rule.value, "capacity")
        report = refusal_report(selection, 2 * GB)
        self.assertIn(f"host:{LOCAL}", report)
        self.assertIn("capacity", report)

    def test_exhaustion_degrades_to_the_next_tier_rather_than_declining(self):
        """The whole reason a second tier exists."""
        reg = kv_spill_registry(
            [_local(pool_gb=10, occupied_gb=10), _remote(total_gb=8)],
            local_host=LOCAL,
        )
        chosen, selection = choose_kv_spill_tier(reg, 2 * GB)
        self.assertEqual(chosen, "host:rig-2")
        self.assertEqual([r.tier_id for r in selection.refusals], [f"host:{LOCAL}"])

    def test_other_pinned_posts_reduce_this_tier_s_headroom(self):
        """#407's `reserved` field, populated from pinned_host_budget at last.

        Two independently plausible pinned budgets can be jointly impossible;
        that missing sum is the defect pinned_host_budget was built to end,
        and a headroom that ignores it is the headroom of an empty machine.
        """
        roomy = kv_spill_registry([_local(pool_gb=10, other_gb=0)], local_host=LOCAL)
        crowded = kv_spill_registry([_local(pool_gb=10, other_gb=9)], local_host=LOCAL)
        self.assertEqual(choose_kv_spill_tier(roomy, 5 * GB)[0], f"host:{LOCAL}")
        self.assertIsNone(choose_kv_spill_tier(crowded, 5 * GB)[0])

    def test_the_pools_own_post_is_not_charged_against_itself(self):
        posts = [
            _Post("kv-session-offload spill pool", 40 * GB),
            _Post("hicache host pool", 7 * GB),
        ]
        occupied, others = occupied_and_other_pinned(
            occupied_bytes=3 * GB,
            posts=posts,
            own_post_name="kv-session-offload spill pool",
        )
        self.assertEqual(occupied, 3 * GB)
        self.assertEqual(others, 7 * GB)

    def test_an_unreachable_tier_is_enumerated_not_omitted(self):
        reg = kv_spill_registry(
            [
                _local(pool_gb=10, occupied_gb=10),
                _remote(reachable=False, verdict="block", reason="ssh refused"),
            ],
            local_host=LOCAL,
        )
        chosen, selection = choose_kv_spill_tier(reg, 2 * GB)
        self.assertIsNone(chosen)
        rules = {r.tier_id: r.rule.value for r in selection.refusals}
        self.assertEqual(rules["host:rig-2"], "health")
        self.assertIn("ssh refused", refusal_report(selection, 2 * GB))


class LocalFirstIsDerivedNotAsserted(CustomTestCase):
    def test_the_measured_ladder_agrees_with_224s_hardcoded_law(self):
        """The load-bearing one: the rule and the measurement must agree.

        Local host RAM at a measured 12 GB/s against rig-2 at the 0.075 GB/s
        this rig actually measured -- the ladder must put local first without
        being told to.
        """
        reg = kv_spill_registry(
            [
                _local(bandwidth=Rate.measured(12.0, "D2H probe", unit="GB/s")),
                _remote(),
            ],
            local_host=LOCAL,
        )
        self.assertIsNone(
            local_first_disagreement(reg, 1 * GB, local_host=LOCAL),
            "the measured ladder must rank the local host tier first",
        )
        self.assertEqual(choose_kv_spill_tier(reg, 1 * GB)[0], f"host:{LOCAL}")

    def test_disagreement_is_reported_rather_than_obeyed(self):
        """CAN-FAIL proof for the falsifier itself.

        Give the remote tier an absurd bandwidth and starve the local one of
        headroom, and the derived ladder puts the park tier first. The law
        still stands -- a device copy cannot reach it -- so the function must
        REPORT, and the report must say which of the two records to distrust.
        """
        reg = kv_spill_registry(
            [
                _local(pool_gb=1, occupied_gb=1),
                _remote(total_gb=500, bandwidth_gbs=999.0),
            ],
            local_host=LOCAL,
        )
        msg = local_first_disagreement(reg, 2 * GB, local_host=LOCAL)
        self.assertIsNotNone(msg)
        self.assertIn("law stands", msg)
        self.assertIn("report, not a reorder", msg)


class TheLadderRendersItsAbsences(CustomTestCase):
    def test_an_absent_bandwidth_prints_why_rather_than_a_dash(self):
        """'unknown' and 'zero' lead to opposite decisions."""
        reg = kv_spill_registry([_local()], local_host=LOCAL)
        rows = ladder_rows(reg)
        self.assertEqual(len(rows), 1)
        tier_id, role, headroom, bandwidth = rows[0]
        self.assertEqual(tier_id, f"host:{LOCAL}")
        self.assertEqual(role, "ram-local")
        self.assertIn("40.00 GB", headroom)
        self.assertIn("absent", bandwidth)
        self.assertIn("no D2H bandwidth probe", bandwidth)

    def test_a_remote_tier_reads_as_remote(self):
        reg = kv_spill_registry([_local(), _remote()], local_host=LOCAL)
        roles = {row[0]: row[1] for row in ladder_rows(reg)}
        self.assertEqual(roles[f"host:{LOCAL}"], "ram-local")
        self.assertEqual(roles["host:rig-2"], "ram-remote")


class _Post:
    def __init__(self, name, nbytes):
        self.name = name
        self.nbytes = nbytes


if __name__ == "__main__":
    unittest.main()
