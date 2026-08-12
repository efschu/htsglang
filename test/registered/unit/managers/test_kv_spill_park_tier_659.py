# SPDX-License-Identifier: Apache-2.0
"""#659 cut 2: the file park tier, measured, and the selection policy it feeds.

Cut 1 put the #407 registry under the KV spill rung with one tier on it and
called it an observer. These tests cover the second rung and the moment the
registry stops observing:

*   the tier's metrics are MEASURED at registration, and an unmeasurable path
    says ``absent`` rather than borrowing a number (register C24);
*   the budget honours df headroom BEFORE the operator's ceiling;
*   the selection policy returns the same verdict as #224's hardcoded law
    wherever budgets allow -- the falsifier that keeps this cut honest;
*   the #224 park counters are READ: a tier that eats parks is refused by name.

Every assertion below has a sibling arm producing the opposite verdict from the
same function, so a test that passes because the mechanism is inert is visible
as the missing arm rather than as a green run (register law 12).
"""

from __future__ import annotations

import unittest

from sglang.srt.managers.kv_spill_park_tier import (
    BYTES_QUANTUM,
    PARK_FAULT_BLOCK,
    choose_park_tier,
    file_park_tier,
    park_counter_row,
    park_fault_key,
    park_filesystem_capacity,
    park_health,
    park_refusal_lines,
    probe_park_filesystem,
    quantize_bandwidth,
)
from sglang.srt.managers.kv_spill_tier_selection import (
    kv_spill_registry,
    local_host_kv_tier,
)
from sglang.srt.memtier.tiers import TierKind, Volatility
from sglang.srt.planner.cost_model import Provenance, Rate
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

GB = 10**9
LOCAL = "rig-1"


def _probe_stub(bandwidth_gbs=3.0, latency_us=120.0):
    return (
        Rate.measured(bandwidth_gbs, "test probe", unit="GB/s", label="bandwidth_gbs"),
        Rate.measured(latency_us, "test probe", unit="us", label="latency_us"),
        {"write_gbs": bandwidth_gbs, "readback_gbs": 9.0, "latency_us_p90": 200.0},
    )


def _park(tmpdir, *, budget=100 * GB, headroom=0, faults=0, bandwidth_gbs=3.0):
    return file_park_tier(
        host=LOCAL,
        directory=tmpdir,
        budget_bytes=budget,
        df_headroom_bytes=headroom,
        faults=faults,
        local_host_tier_id="host:" + LOCAL,
        probe=_probe_stub(bandwidth_gbs=bandwidth_gbs),
    )


class TheMetricsAreMeasuredHere(unittest.TestCase):
    """C24's lesson as a test: a number comes from the path, or it is absent."""

    def test_a_real_probe_of_a_real_directory_is_measured(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            bandwidth, latency, raw = probe_park_filesystem(
                d, probe_bytes=1 << 20, latency_samples=4
            )
        self.assertIs(bandwidth.provenance, Provenance.MEASURED)
        self.assertIs(latency.provenance, Provenance.MEASURED)
        self.assertGreater(bandwidth.require("bandwidth_gbs"), 0.0)
        self.assertIn("incompressible", bandwidth.source)
        # The read-back is recorded but is NOT the ranked number: it runs with
        # the cache warm and would credit the tier with a speed no park pays.
        self.assertIn("readback_gbs", raw)

    def test_an_unprobeable_path_is_absent_not_borrowed(self):
        """The sibling arm. An absent rate is refusable; a borrowed one is not."""
        bandwidth, latency, _ = probe_park_filesystem(
            "/proc/nonexistent-park-dir/nope", probe_bytes=1 << 20
        )
        self.assertIs(bandwidth.provenance, Provenance.ABSENT)
        self.assertIs(latency.provenance, Provenance.ABSENT)
        self.assertIn("park probe failed", bandwidth.source)

    def test_the_entry_declares_persistent_and_stages_through_local(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            tier = _park(d)
        self.assertIs(tier.kind, TierKind.FILESYSTEM)
        self.assertIs(tier.volatility, Volatility.PERSISTENT)
        # A device copy cannot land in a file: the edge is what makes "below
        # local" a statement about physics rather than about a name.
        self.assertEqual(tier.transport.stages_through, "host:" + LOCAL)


class TheBudgetHonoursDfHeadroomFirst(unittest.TestCase):
    def test_the_operator_ceiling_binds_when_the_volume_is_roomy(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            cap = park_filesystem_capacity(
                d, budget_bytes=1 * GB, df_headroom_bytes=0
            )
        self.assertEqual(cap.total.require("total"), 1 * GB)
        self.assertIn("configured park budget", cap.total.source)

    def test_df_headroom_binds_first_and_can_shrink_the_tier_to_zero(self):
        """The sibling arm: headroom comes off BEFORE the budget is honoured.

        A generous budget on a volume with no room must yield a SMALL tier, not
        the budget. This is the #558-family discipline and the arm that would
        go red if the min() were ever reordered.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            cap = park_filesystem_capacity(
                d, budget_bytes=1000 * GB, df_headroom_bytes=1 << 62
            )
        self.assertEqual(cap.total.require("total"), 0)

    def test_an_unstatable_directory_reports_absent_capacity(self):
        cap = park_filesystem_capacity(
            "/proc/nonexistent-park-dir/nope", budget_bytes=1 * GB
        )
        self.assertTrue(cap.total.is_absent)
        self.assertTrue(cap.headroom().is_absent)

    def test_parked_bytes_reduce_this_tier_s_headroom(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            empty = park_filesystem_capacity(
                d, budget_bytes=10 * GB, df_headroom_bytes=0, parked_bytes=0
            )
            filled = park_filesystem_capacity(
                d, budget_bytes=10 * GB, df_headroom_bytes=0, parked_bytes=9 * GB
            )
        self.assertEqual(empty.headroom().require("h"), 10 * GB)
        self.assertEqual(filled.headroom().require("h"), 1 * GB)


class TheSelectionPolicyAgreesWithTheHardcodedLaw(unittest.TestCase):
    """The falsifier. Where budgets allow, the derived verdict IS #224's."""

    def _registry(self, park_tiers, *, local_pool=40 * GB, occupied=0):
        local = local_host_kv_tier(
            host=LOCAL, pool_bytes=local_pool, occupied_bytes=occupied
        )
        return kv_spill_registry([local, *park_tiers], local_host=LOCAL)

    def test_one_healthy_park_tier_is_chosen_exactly_as_index_zero_was(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            tier = _park(d)
            reg = self._registry([tier])
            index, selection = choose_park_tier(reg, 1 * GB, park_tier_ids=[tier.id])
        self.assertEqual(index, 0)
        self.assertEqual(park_refusal_lines(selection, [tier.id], 1 * GB), [])

    def test_the_faster_measured_tier_outranks_the_listed_first_one(self):
        """The sibling arm: the ladder can DISAGREE with the configured order.

        Without this arm the test above would pass on a mechanism that always
        answered 0 -- which is precisely the inert-mechanism failure mode.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as slow_dir:
            with tempfile.TemporaryDirectory() as fast_dir:
                slow = _park(slow_dir, bandwidth_gbs=0.5)
                fast = _park(fast_dir, bandwidth_gbs=9.0)
                reg = self._registry([slow, fast])
                index, _ = choose_park_tier(
                    reg, 1 * GB, park_tier_ids=[slow.id, fast.id]
                )
        self.assertEqual(index, 1)

    def test_a_tier_too_small_for_the_ask_is_refused_by_name(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            tier = _park(d, budget=1 * GB)
            reg = self._registry([tier])
            index, selection = choose_park_tier(reg, 5 * GB, park_tier_ids=[tier.id])
        self.assertIsNone(index)
        lines = park_refusal_lines(selection, [tier.id], 5 * GB)
        self.assertEqual(len(lines), 1)
        self.assertIn("capacity", lines[0])
        self.assertIn(tier.id, lines[0])

    def test_quantization_makes_near_identical_measurements_rank_alike(self):
        """Group uniformity: measurement noise may not reorder a ladder.

        Two ranks probing one filesystem measure different floats. If those
        sorted directly, two ranks could park one session to two tiers and the
        divergence would surface as a hang (register law 14).
        """
        self.assertEqual(quantize_bandwidth(3.01), quantize_bandwidth(3.19))
        # ... and a genuinely different medium still separates.
        self.assertNotEqual(quantize_bandwidth(3.01), quantize_bandwidth(9.0))


class TheParkCountersAreRead(unittest.TestCase):
    """#659 (d): the tally stops being write-only and starts deciding."""

    def test_a_clean_tier_is_healthy_and_a_faulted_one_is_blocked(self):
        self.assertEqual(park_health(0)[1], "ok")
        self.assertEqual(park_health(1)[1], "warn")
        self.assertTrue(park_health(1)[0])
        reachable, verdict, reason = park_health(PARK_FAULT_BLOCK)
        self.assertFalse(reachable)
        self.assertEqual(verdict, "block")
        self.assertIn(str(PARK_FAULT_BLOCK), reason)

    def test_a_blocked_tier_is_refused_by_the_registry_on_health(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            tier = _park(d, faults=PARK_FAULT_BLOCK)
            local = local_host_kv_tier(host=LOCAL, pool_bytes=40 * GB)
            reg = kv_spill_registry([local, tier], local_host=LOCAL)
            index, selection = choose_park_tier(reg, 1 * GB, park_tier_ids=[tier.id])
        self.assertIsNone(index)
        lines = park_refusal_lines(selection, [tier.id], 1 * GB)
        self.assertIn("health", lines[0])
        # Refused, but still ENUMERATED: a spill target that silently becomes a
        # different spill target is the failure #224's docstring names.
        self.assertIn(tier.id, [t.id for t in reg.tiers()])

    def test_the_ledger_line_reports_traffic_not_occupancy(self):
        row = park_counter_row(
            {"parks_committed": 4, "parks_failed": 1, "park_bytes_out": 77}
        )
        self.assertIn("4", row)
        self.assertIn("moved out/in=77/0 B", row)

    def test_the_fault_key_is_per_tier(self):
        self.assertNotEqual(park_fault_key("file"), park_fault_key("mooncake"))
        self.assertIn("file", park_fault_key("file"))


class TheAskIsABootConstant(unittest.TestCase):
    def test_the_ask_is_quantized_up_so_ranks_cannot_straddle_a_threshold(self):
        from sglang.srt.managers.kv_session_spill_destination import _park_ask_bytes

        class _Pool:
            def get_size_per_token(self):
                return 1000

        class _Mgr:
            host_pool = _Pool()
            region_tokens = 1500

        ask = _park_ask_bytes(_Mgr(), None)
        self.assertEqual(ask % BYTES_QUANTUM, 0)
        self.assertGreaterEqual(ask, 1500 * 1000)

    def test_no_pool_means_no_ask_rather_than_a_guess(self):
        class _Mgr:
            host_pool = None
            region_tokens = 0

        self.assertEqual(
            __import__(
                "sglang.srt.managers.kv_session_spill_destination",
                fromlist=["_park_ask_bytes"],
            )._park_ask_bytes(_Mgr(), None),
            0,
        )


if __name__ == "__main__":
    unittest.main()
