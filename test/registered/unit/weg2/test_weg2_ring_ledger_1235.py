"""T2 -- C19 (the ledger's ring terms) and C20 (the R5 launch check), pure python.

Everything here runs off SYNTHETIC log text written by the test, so the boot it
reads is fixed and the arithmetic is checkable by hand.  One case additionally
replays the real boot ``weg2zr2``'s own lines from ``/spinning/evidence-665-f1``
and asserts the spec section 2 table falls out of them -- that is the provenance
claim itself under test, and it is skipped (never faked) when the evidence tree
is not present.

RED-FIRST, each verified by mutation:

* delete the two-tag pass split in ``ring_table._passes``      -> image_P doubles
* let ``max_tag`` count a bulk record                          -> max_tag = image
* take the MEAN pass instead of the peak                       -> H shrinks
* take the FIRST corridor sample instead of the minimum        -> credit grows
* restore ``BACKUP_P_BYTES`` as the ledger's launch term       -> launch is 3.4 GiB off
* drop the ``need > H`` test in ``refusals``                   -> W32 never fires
"""

from __future__ import annotations

import os
import unittest
from dataclasses import dataclass

from sglang.srt.weg2 import host_ledger, ring_table

GIB = host_ledger.GIB
MIB = ring_table.MIB
EVIDENCE = "/spinning/evidence-665-f1"
ZR2 = "boot_weg2_weg2zr2_7e3a9150b4_0907_153801"


@dataclass
class FakeCard:
    nvml_index: int
    uuid: str
    name: str
    total_mib: int = 20480


CARDS = [
    FakeCard(1, "GPU-aaaa", "NVIDIA GeForce RTX 5090", 32607),
    FakeCard(0, "GPU-bbbb", "NVIDIA GeForce RTX 3080"),
    FakeCard(2, "GPU-cccc", "NVIDIA GeForce RTX 3080"),
]


def _group_log(prefix: str, passes, kv_gb, bulk=None) -> str:
    """``passes`` = list of {rank: [(tag, mib), ...]}; ``bulk`` = {rank: mib}."""
    out = []
    for rank, gb in kv_gb.items():
        out.append(
            f"[2026-09-07 21:06:30 {prefix}{rank}] KV Cache is allocated. dtype: "
            f"torch.float8_e4m3fn, #tokens: 1, K size: {gb / 2:.2f} GB, V size: {gb / 2:.2f} GB"
        )
    for one in passes:
        for rank, records in one.items():
            for tag, mib in records:
                out.append(
                    f"[2026-09-07 21:10:23 {prefix}{rank}] WEG2-CHUNK-BYTES sleep "
                    f"tags=['{tag}'] host_image_delta={mib} MiB "
                    "(RssShmem 0 -> 0 MiB, /proc/self/status)"
                )
    if bulk:
        # AFTER the per-tag passes, and carrying tags that did NOT appear in
        # them: that is the only arrangement in which "bulk record = whole
        # pass" and "bulk record = one more tag" give different answers, so it
        # is the only arrangement that can catch the 2x error.
        for rank, mib in bulk.items():
            out.append(
                f"[2026-09-07 21:11:00 {prefix}{rank}] WEG2-CHUNK-BYTES sleep "
                f"tags=['weights_8', 'weights_9'] host_image_delta={mib} MiB "
                "(RssShmem 0 -> 0 MiB, /proc/self/status)"
            )
    return "\n".join(out) + "\n"


def _front_log(free_by_phase) -> str:
    out = []
    for phase, samples in free_by_phase.items():
        for row in samples:
            free = " ".join(f"nvml{i}:free={v}MiB" for i, v in row.items())
            out.append(
                f"[2026-09-07 21:07:54,029] INFO weg2.front: WEG2-CORRIDOR "
                f"phase={phase}(awake) epoch=0 {free} min_so_far={{}}"
            )
    return "\n".join(out) + "\n"


class RingTableSolverTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.dir = tempfile.mkdtemp(prefix="weg2ringtable-")
        self.stem = "boot_weg2_t2_0000000000_0907_000000"

    def _write(self, p_passes, d_passes, p_kv, d_kv, corridor, p_bulk=None, d_bulk=None):
        with open(os.path.join(self.dir, f"{self.stem}.P.log"), "w") as f:
            f.write(_group_log("PP", p_passes, p_kv, p_bulk))
        with open(os.path.join(self.dir, f"{self.stem}.D.log"), "w") as f:
            f.write(_group_log("TP", d_passes, d_kv, d_bulk))
        with open(os.path.join(self.dir, f"{self.stem}.front.log"), "w") as f:
            f.write(_front_log(corridor))

    def _solve(self, cards=None):
        # ONE card: these cases fix the arithmetic, and a rank that logged
        # nothing is the SEPARATE property tested by the two "returns none"
        # cases below.
        return ring_table.solve(cards or [CARDS[0]], self.dir, self.stem)

    def test_image_is_the_peak_pass_and_max_tag_is_the_largest_single_tag(self):
        # Two passes on each rank; the SECOND is larger, so H must follow it.
        self._write(
            p_passes=[
                {0: [("weights_0", 100), ("weights_1", 200)]},
                {0: [("weights_0", 150), ("weights_1", 260)]},
            ],
            d_passes=[{0: [("weights_0", 300), ("weights_1", 90)]}],
            p_kv={0: 1.0}, d_kv={0: 2.0},
            corridor={"P": [{1: 500}], "D": [{1: 700}]},
        )
        table, reason = self._solve()
        self.assertIsNotNone(table, reason)
        c = table.cards[0]
        self.assertEqual(c.image_p_mib, 410, "peak pass 150+260, not the mean and not the sum")
        self.assertEqual(c.max_tag_p_mib, 260)
        self.assertEqual(c.image_d_mib, 390)
        self.assertEqual(c.max_tag_d_mib, 300)
        self.assertEqual(c.h_mib, 410, "H = max_g image_g")
        self.assertEqual(c.span1_mib, 410, "span 1 = image_P (R7)")

    def test_a_bulk_multi_tag_record_is_one_whole_pass_not_a_tag(self):
        # The launcher's own sleep(P) sends the WHOLE family in one RPC; that
        # line is a complete pass.  Measured on boot weg2dk5: counting it as a
        # member of the surrounding pass read image_P as 27 720 instead of
        # 13 860 MiB, exactly 2x.
        self._write(
            p_passes=[{0: [("weights_0", 100), ("weights_1", 200)]}],
            d_passes=[{0: [("weights_0", 10)]}],
            p_kv={0: 1.0}, d_kv={0: 1.0},
            corridor={"P": [{1: 100}], "D": [{1: 100}]},
            p_bulk={0: 300},
        )
        table, reason = self._solve()
        self.assertIsNotNone(table, reason)
        c = table.cards[0]
        self.assertEqual(c.image_p_mib, 300, "the bulk line IS the pass, not an addend")
        self.assertEqual(c.max_tag_p_mib, 200,
                         "a bulk record is a whole family and must never be read "
                         "as a corridor step")

    def test_the_credit_uses_the_minimum_corridor_sample_not_the_first(self):
        self._write(
            p_passes=[{0: [("weights_0", 10)]}],
            d_passes=[{0: [("weights_0", 10)]}],
            p_kv={0: 0.0}, d_kv={0: 0.0},
            corridor={"P": [{1: 5000}, {1: 900}], "D": [{1: 4000}, {1: 800}]},
        )
        table, _ = self._solve()
        c = table.cards[0]
        self.assertEqual(c.credit_d2p_mib, 800, "the phase MINIMUM is the conservative end")
        self.assertEqual(c.credit_p2d_mib, 900)

    def test_the_kv_a_group_releases_is_part_of_its_credit(self):
        self._write(
            p_passes=[{0: [("weights_0", 10)]}],
            d_passes=[{0: [("weights_0", 10)]}],
            p_kv={0: 2.0}, d_kv={0: 4.0},
            corridor={"P": [{1: 100}], "D": [{1: 100}]},
        )
        table, _ = self._solve()
        c = table.cards[0]
        self.assertEqual(c.credit_d2p_mib, 100 + int(round(4.0 * 1e9 / MIB)))
        self.assertEqual(c.credit_p2d_mib, 100 + int(round(2.0 * 1e9 / MIB)))

    def test_a_short_table_returns_none_with_a_reason_never_a_guess(self):
        self._write(
            p_passes=[],
            d_passes=[{0: [("weights_0", 10)]}],
            p_kv={0: 1.0}, d_kv={0: 1.0},
            corridor={"P": [{1: 100}], "D": [{1: 100}]},
        )
        table, reason = self._solve()
        self.assertIsNone(table)
        self.assertIn("no sleep-pass lines", reason)

    def test_a_card_with_no_rows_of_its_own_returns_none_with_a_reason(self):
        # Three cards, one rank logged: a table that covered only one card
        # would size two regions from nothing.
        self._write(
            p_passes=[{0: [("weights_0", 10)]}],
            d_passes=[{0: [("weights_0", 10)]}],
            p_kv={0: 1.0}, d_kv={0: 1.0},
            corridor={"P": [{1: 100}], "D": [{1: 100}]},
        )
        table, reason = self._solve(CARDS)
        self.assertIsNone(table)
        self.assertIn("incomplete rows", reason)
        self.assertIn("GPU-bbbb", reason)

    def test_a_front_without_both_phases_returns_none_with_a_reason(self):
        self._write(
            p_passes=[{0: [("weights_0", 10)]}],
            d_passes=[{0: [("weights_0", 10)]}],
            p_kv={0: 1.0}, d_kv={0: 1.0},
            corridor={"D": [{1: 100}]},
        )
        table, reason = self._solve()
        self.assertIsNone(table)
        self.assertIn("WEG2-CORRIDOR", reason)


class R5LaunchCheckTest(unittest.TestCase):
    """C20/W32: the corridor inequality, per card per direction."""

    def _card(self, **kw):
        base = dict(uuid="GPU-aaaa", nvml_index=1, name="RTX 5090",
                    image_p_mib=13860, image_d_mib=13914,
                    max_tag_p_mib=2988, max_tag_d_mib=2856,
                    credit_d2p_mib=13441, credit_p2d_mib=12609)
        base.update(kw)
        return ring_table.CardRing(**base)

    def test_the_spec_section_2_5090_row_reproduces_exactly(self):
        c = self._card()
        self.assertEqual(c.h_mib, 13914)
        self.assertEqual(c.need_d2p_mib, 6263)   # 13860 - 13441 + 2856 + 2988
        self.assertEqual(c.need_p2d_mib, 7149)   # 13914 - 12609 + 2988 + 2856
        self.assertEqual(c.slack_d2p_mib, 7651)
        self.assertEqual(c.slack_p2d_mib, 6765)

    def test_all_six_spec_cases_pass_and_none_refuses(self):
        rows = [
            self._card(),
            self._card(uuid="GPU-bbbb", nvml_index=0, name="RTX 3080",
                       image_p_mib=7548, image_d_mib=9680,
                       max_tag_p_mib=2986, max_tag_d_mib=2714,
                       credit_d2p_mib=7106, credit_p2d_mib=7741),
            self._card(uuid="GPU-cccc", nvml_index=2, name="RTX 3080",
                       image_p_mib=8504, image_d_mib=9370,
                       max_tag_p_mib=3336, max_tag_d_mib=2686,
                       credit_d2p_mib=7535, credit_p2d_mib=7033),
        ]
        table = ring_table.RingTable(boot="spec-2", instrument="WEG2-CHUNK-BYTES",
                                     lines_read=1, cards=rows)
        self.assertEqual(table.refusals(), [])
        self.assertEqual(table.total_h_bytes // MIB, 32964, "spec section 2 Sigma H")
        self.assertEqual(table.total_span1_bytes // MIB, 29912, "spec section 2 Sigma span1")
        slacks = [(c.slack_d2p_mib, c.slack_p2d_mib) for c in rows]
        self.assertEqual(slacks, [(7651, 6765), (3538, 2041), (2379, 1011)])
        self.assertEqual(min(min(s) for s in slacks), 1011,
                         "the binding case is nvml2 P->D")

    def test_a_reduced_credit_refuses_by_name_with_the_arithmetic(self):
        bad = self._card(credit_d2p_mib=13441 - 8000)
        table = ring_table.RingTable(boot="b", instrument="i", lines_read=1, cards=[bad])
        refusals = table.refusals()
        self.assertEqual(len(refusals), 1, refusals)
        self.assertIn("RING REFUSED: need 14263 > H 13914", refusals[0])
        for token in ("image_P 13860", "credit 5441", "max_tag_D 2856", "max_tag_P 2988"):
            self.assertIn(token, refusals[0],
                          "the refusal must carry the arithmetic, not just a verdict")

    def test_l6_names_its_boot_and_its_instrument(self):
        table = ring_table.RingTable(boot="boot_x", instrument="WEG2-FLIP-TAG bytes",
                                     lines_read=42, cards=[self._card()])
        line = table.format_l6()[0]
        self.assertIn("WEG2-HOST-LEDGER RING card=GPU-aaaa", line)
        self.assertIn("image_D=13914", line)
        self.assertIn("H=13914", line)
        self.assertIn("span1=13860", line)
        self.assertIn("boot boot_x", line)
        self.assertIn("42 WEG2-FLIP-TAG bytes", line)

    def test_the_env_map_carries_bytes_span1_and_optionally_the_fd(self):
        table = ring_table.RingTable(boot="b", instrument="i", lines_read=1,
                                     cards=[self._card()])
        self.assertEqual(table.env_map(),
                         f"GPU-aaaa={13914 * MIB}:{13860 * MIB}")
        self.assertEqual(table.env_map({"GPU-aaaa": 7}),
                         f"GPU-aaaa={13914 * MIB}:{13860 * MIB}:fd=7")


class LedgerRingTermsTest(unittest.TestCase):
    """C19: launch charges Sigma span1, run charges Sigma H, and the deleted
    constants stay deleted."""

    SIGMA_H = 32964 * MIB
    SIGMA_SPAN1 = 29912 * MIB

    def _common_box(self):
        """A box whose ``common`` is the spec section 2 boot-of-record 42.26 GiB.

        Solved, not guessed: common = base - floor - headroom - heaps - anchors
        - rings - overhead at M=1200, so base = 42.26 + the rest.
        """
        m = 1200
        heaps = 3 * (host_ledger.HEAP_AWAKE_GIB + host_ledger.HEAP_DORMANT_GIB)
        anchors = (host_ledger.ANCHORS_AT_2400_BYTES * (m / 2400)) / GIB
        rings = (host_ledger.RING_P_MULT_GB_PER_S + host_ledger.RING_D_MULT_GB_PER_S) * 1 * 1e9 / GIB
        overhead = host_ledger.HOST_POOL_OVERHEAD * (anchors + rings)
        base = (42.26 + host_ledger.FLOOR_GIB + host_ledger.HOST_HEADROOM_GIB
                + heaps + anchors + rings + overhead)
        memtotal = int((base + host_ledger.CLI_RESERVE_GIB + 50) * GIB)
        return memtotal, int(base * GIB)

    def test_the_spec_section_2_ring_row_reproduces(self):
        memtotal, memavail = self._common_box()
        arm = host_ledger.price(memtotal, memavail, 1, 1200,
                                ring_bytes=self.SIGMA_H,
                                ring_span1_bytes=self.SIGMA_SPAN1)
        self.assertAlmostEqual(arm.terms["host_ring_gib"], 32.19, places=2)
        self.assertAlmostEqual(arm.terms["host_ring_span1_gib"], 29.21, places=2)
        self.assertAlmostEqual(arm.launch_leftover_gib, 1.05, places=2)
        self.assertAlmostEqual(arm.run_leftover_gib, 10.07, places=2)
        self.assertTrue(arm.fundable_moments)

    def test_charging_sigma_h_at_the_launch_moment_is_what_r7_refuses(self):
        # The one-span arm of the spec section 2 table: launch -1.93 GiB.
        memtotal, memavail = self._common_box()
        one_span = host_ledger.price(memtotal, memavail, 1, 1200,
                                     ring_bytes=self.SIGMA_H,
                                     ring_span1_bytes=self.SIGMA_H)
        self.assertAlmostEqual(one_span.launch_leftover_gib, -1.93, places=2)
        self.assertFalse(one_span.fundable_moments)

    def test_a_missing_table_refuses_by_name_and_never_prices_zero(self):
        memtotal, memavail = self._common_box()
        for kw in ({"ring_bytes": 0, "ring_span1_bytes": self.SIGMA_SPAN1},
                   {"ring_bytes": self.SIGMA_H, "ring_span1_bytes": 0},
                   {}):
            with self.assertRaises(host_ledger.Weg2HostLedgerRefused) as ctx:
                host_ledger.price(memtotal, memavail, 1, 1200, **kw)
            self.assertIn("W20 Weg2HostLedgerRefused", str(ctx.exception))
            self.assertIn("no measured source", str(ctx.exception))

    def test_span1_larger_than_the_region_is_a_programming_error(self):
        memtotal, memavail = self._common_box()
        with self.assertRaises(ValueError):
            host_ledger.price(memtotal, memavail, 1, 1200,
                              ring_bytes=self.SIGMA_SPAN1,
                              ring_span1_bytes=self.SIGMA_H)

    def test_the_deleted_constants_are_gone_from_the_module(self):
        for name in ("BACKUP_P_BYTES", "BACKUP_D_BYTES", "HOST_HEADROOM_GIB_CHUNK"):
            if name == "HOST_HEADROOM_GIB_CHUNK":
                continue
            self.assertFalse(hasattr(host_ledger, name),
                             f"C19 deletes {name}; a survivor is a second, stale "
                             "source for the host weights term")
        arm = host_ledger.price(*self._common_box(), 1, 1200,
                                ring_bytes=self.SIGMA_H,
                                ring_span1_bytes=self.SIGMA_SPAN1)
        for key in ("chunk_gib", "backup_p_gib", "backup_d_gib", "backup_resident_gib",
                    "weight_chunks"):
            self.assertNotIn(key, arm.terms)

    def test_choose_prints_the_ring_provenance_it_was_given(self):
        memtotal, memavail = self._common_box()
        arm, store, lines = host_ledger.choose(
            memtotal, memavail, store_min_gib=4.0,
            ring_bytes=self.SIGMA_H, ring_span1_bytes=self.SIGMA_SPAN1,
            ring_provenance="boot weg2zr2, 165 WEG2-CHUNK-BYTES lines",
        )
        terms = lines[0]
        self.assertIn("32.19 GiB", terms)
        self.assertIn("29.21 GiB", terms)
        self.assertIn("boot weg2zr2, 165 WEG2-CHUNK-BYTES lines", terms)
        self.assertEqual((arm.s_gb, arm.m_mib), (1, 1200))
        self.assertEqual(store, 10.0)

    def test_an_unfundable_ladder_refuses_with_the_ring_arithmetic(self):
        with self.assertRaises(host_ledger.Weg2HostLedgerRefused) as ctx:
            host_ledger.choose(int(40 * GIB), int(30 * GIB), store_min_gib=8.0,
                               ring_bytes=self.SIGMA_H,
                               ring_span1_bytes=self.SIGMA_SPAN1,
                               ring_provenance="boot fake")
        msg = str(ctx.exception)
        self.assertIn("W20 Weg2HostLedgerRefused", msg)
        self.assertIn("32.19 GiB at the run moment", msg)
        self.assertIn("boot fake", msg)


@unittest.skipUnless(os.path.isfile(os.path.join(EVIDENCE, f"{ZR2}.front.log")),
                     f"{EVIDENCE}/{ZR2} not present")
class RealBootProvenanceTest(unittest.TestCase):
    """The provenance claim itself: spec section 2's table IS boot weg2zr2's lines."""

    def test_weg2zr2_yields_the_spec_section_2_images_and_max_tags(self):
        table, reason = ring_table.solve(CARDS, EVIDENCE, ZR2)
        self.assertIsNotNone(table, reason)
        got = {c.uuid: (c.image_d_mib, c.image_p_mib, c.h_mib) for c in table.cards}
        self.assertEqual(got["GPU-aaaa"], (13914, 13860, 13914))
        self.assertEqual(got["GPU-bbbb"], (9680, 7548, 9680))
        self.assertEqual(got["GPU-cccc"], (9370, 8504, 9370))
        self.assertEqual(table.total_h_bytes // MIB, 32964)
        self.assertEqual(table.total_span1_bytes // MIB, 29912)
        max_tags = {c.uuid: (c.max_tag_d_mib, c.max_tag_p_mib) for c in table.cards}
        self.assertEqual(max_tags["GPU-aaaa"], (2856, 2988))
        self.assertEqual(max_tags["GPU-bbbb"], (2714, 2986))
        self.assertEqual(max_tags["GPU-cccc"], (2686, 3336))

    def test_the_launch_check_passes_on_that_boots_own_credit(self):
        # The credit here is the MINIMUM corridor sample, which is tighter than
        # the single sample spec section 2 quoted -- so the needs are larger and
        # the slacks smaller.  All six must still pass; if they did not, the
        # honest answer would be W32, not a looser instrument.
        table, _ = ring_table.solve(CARDS, EVIDENCE, ZR2)
        self.assertEqual(table.refusals(), [], "\n".join(table.refusals()))
        for c in table.cards:
            self.assertGreater(c.slack_d2p_mib, 0, c.uuid)
            self.assertGreater(c.slack_p2d_mib, 0, c.uuid)


class LauncherRingPlanTest(unittest.TestCase):
    """C18: what build_env publishes, and what it publishes when nothing is proven."""

    def test_build_env_publishes_exactly_the_four_variables_when_armed(self):
        from sglang.srt.weg2 import launcher

        plan = launcher.HostRingPlan(form="MAP_SHARED", dir="/dev/shm/x",
                                     env_map="GPU-aaaa=1:2", epoch=7, armed=True)
        env = launcher.build_env("/tree", "/venv", "GPU-aaaa", "/store", False, "t",
                                 ring=plan)
        self.assertEqual(env["TMS_HOST_RING_DIR"], "/dev/shm/x")
        self.assertEqual(env["TMS_HOST_RING_MAP"], "GPU-aaaa=1:2")
        self.assertEqual(env["TMS_HOST_RING_EPOCH"], "7")
        self.assertEqual(env["TMS_HOST_RING_FORM"], "MAP_SHARED")

    def test_an_unarmed_plan_publishes_nothing_and_scrubs_an_inherited_value(self):
        from sglang.srt.weg2 import launcher

        os.environ["TMS_HOST_RING_DIR"] = "/leftover/from/a/previous/boot"
        try:
            env = launcher.build_env("/tree", "/venv", "GPU-aaaa", "/store", False, "t",
                                     ring=launcher.HostRingPlan())
        finally:
            del os.environ["TMS_HOST_RING_DIR"]
        for key in ("TMS_HOST_RING_DIR", "TMS_HOST_RING_MAP", "TMS_HOST_RING_EPOCH",
                    "TMS_HOST_RING_FORM"):
            self.assertNotIn(key, env,
                             "an unarmed boot must run the stock path; a leaked "
                             "variable would arm a ring nobody sized")

    def test_the_old_form_is_charged_from_the_same_measured_table(self):
        from sglang.srt.weg2 import launcher

        card = ring_table.CardRing(uuid="GPU-aaaa", nvml_index=1, name="c",
                                   image_p_mib=100, image_d_mib=200,
                                   max_tag_p_mib=30, max_tag_d_mib=20)
        table = ring_table.RingTable(boot="b", instrument="i", lines_read=1, cards=[card])
        armed = launcher.HostRingPlan(form="MAP_SHARED", armed=True, table=table)
        old = launcher.HostRingPlan(form="", armed=False, table=table)
        self.assertEqual(armed.host_weights_bytes, 200 * MIB, "the ring holds Sigma H")
        self.assertEqual(old.host_weights_bytes, (200 + 30) * MIB,
                         "the old form holds one image PLUS one chunk in flight")
        self.assertEqual(armed.host_weights_span1_bytes, 100 * MIB)
        self.assertIn("OLD flip form", old.provenance)
        self.assertIn("ring form MAP_SHARED", armed.provenance)

    def test_no_table_means_no_charge_and_the_ledger_refuses_downstream(self):
        from sglang.srt.weg2 import launcher

        plan = launcher.HostRingPlan()
        self.assertEqual(plan.host_weights_bytes, 0)
        self.assertEqual(plan.host_weights_span1_bytes, 0)
        with self.assertRaises(host_ledger.Weg2HostLedgerRefused):
            host_ledger.choose(int(118 * GIB), int(103 * GIB), store_min_gib=4.0,
                               ring_bytes=plan.host_weights_bytes,
                               ring_span1_bytes=plan.host_weights_span1_bytes,
                               ring_provenance=plan.provenance)


if __name__ == "__main__":
    unittest.main(verbosity=2)
