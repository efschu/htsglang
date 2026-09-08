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

#: FIX 2: the evidence-backed cases must name the cards the SOURCE BOOT names.
#: The table is keyed by UUID end to end now, so a fake uuid against a real log
#: no longer "works by position" -- it is refused, which is the point.
REAL_5090 = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
REAL_3080_A = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"   # nvml 0 on that boot
REAL_3080_B = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"   # nvml 2 on that boot
REAL_CARDS = [
    FakeCard(1, REAL_5090, "NVIDIA GeForce RTX 5090", 32607),
    FakeCard(0, REAL_3080_A, "NVIDIA GeForce RTX 3080"),
    FakeCard(2, REAL_3080_B, "NVIDIA GeForce RTX 3080"),
]


def _group_log(prefix: str, passes, kv_gb, bulk=None, rss=None) -> str:
    """``passes`` = list of {rank: [(tag, mib), ...]}; ``bulk`` = {rank: mib}.

    ``rss`` = {rank: (lowest, highest)} drives the RESIDENT-IMAGE instrument
    independently of the deltas, so a case can make the two instruments
    disagree on purpose.  Default (0, 0) = "this boot logged no absolute".
    """
    rss = rss or {}
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
                    f"(RssShmem {rss.get(rank, (0, 0))[0]} -> "
                    f"{rss.get(rank, (0, 0))[1]} MiB, /proc/self/status)"
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
                f"(RssShmem {rss.get(rank, (0, 0))[0]} -> "
                f"{rss.get(rank, (0, 0))[1]} MiB, /proc/self/status)"
            )
    return "\n".join(out) + "\n"


def _front_log(free_by_phase, cards=None, identity=True) -> str:
    out = []
    if identity:
        # Every real launcher prints this; a fixture without it is a boot whose
        # card identity was never recorded, which is a REFUSAL (see
        # test_a_source_boot_without_an_identity_line_is_refused_by_name).
        out.append(
            "[2026-09-07T21:05:00Z] WEG2-LAUNCH NVML -> CUDA ordinal map: "
            + ", ".join(
                f"ordinal {i} = nvml {c.nvml_index} {c.name} {c.uuid} "
                f"total {c.total_mib} MiB"
                for i, c in enumerate(cards or CARDS)
            )
        )
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

    def _write(self, p_passes, d_passes, p_kv, d_kv, corridor, p_bulk=None,
               d_bulk=None, identity=True, p_rss=None, d_rss=None):
        with open(os.path.join(self.dir, f"{self.stem}.P.log"), "w") as f:
            f.write(_group_log("PP", p_passes, p_kv, p_bulk, p_rss))
        with open(os.path.join(self.dir, f"{self.stem}.D.log"), "w") as f:
            f.write(_group_log("TP", d_passes, d_kv, d_bulk, d_rss))
        with open(os.path.join(self.dir, f"{self.stem}.front.log"), "w") as f:
            f.write(_front_log(corridor, identity=identity))

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

    def test_the_env_map_carries_bytes_and_span1_and_never_an_fd(self):
        table = ring_table.RingTable(boot="b", instrument="i", lines_read=1,
                                     cards=[self._card()])
        self.assertEqual(table.env_map(),
                         f"GPU-aaaa={13914 * MIB}:{13860 * MIB}")
        # FIX 1: the map is sizes only.  An fd number published here would name
        # a different object in a spawn-started rank, so the field is gone and
        # env_map takes no argument that could reintroduce it.
        with self.assertRaises(TypeError):
            table.env_map({"GPU-aaaa": 7})


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
        table, reason = ring_table.solve(REAL_CARDS, EVIDENCE, ZR2)
        self.assertIsNotNone(table, reason)
        got = {c.uuid: (c.image_d_mib, c.image_p_mib, c.h_mib) for c in table.cards}
        # FIX 2: image is now max(weight-tag census, measured resident image).
        # On this boot the two agree exactly on P and the resident instrument
        # reads 20 MiB MORE on every D rank -- a backed-up tag the weights
        # family does not name.  The census numbers are still asserted below.
        self.assertEqual(got[REAL_5090], (13934, 13860, 13934))
        self.assertEqual(got[REAL_3080_A], (9700, 7548, 9700))
        self.assertEqual(got[REAL_3080_B], (9390, 8504, 9390))
        census = {c.uuid: (c.tags_d_mib, c.tags_p_mib) for c in table.cards}
        self.assertEqual(census[REAL_5090], (13914, 13860))
        self.assertEqual(census[REAL_3080_A], (9680, 7548))
        self.assertEqual(census[REAL_3080_B], (9370, 8504))
        self.assertEqual(table.total_h_bytes // MIB, 33024)
        self.assertEqual(table.total_span1_bytes // MIB, 29912)
        self.assertEqual(table.total_tags_mib, 32964)
        self.assertEqual(table.total_dormant_mib, 33024)
        # Both instruments in the provenance line, neither quotable alone.
        self.assertIn("weight-tag census 32964 MiB", table.provenance())
        self.assertIn("measured resident image 33024 MiB", table.provenance())
        # FIX 1: max_tag is the step of the front's per-tag interleave, so the
        # unchunked family tag 'weights' -- the launcher's own bulk sleep, which
        # names the family that 'weights_<k>' partitions -- no longer feeds it.
        # D's step on the 5090 is weights_0 at 1 608 MiB, which is the figure the
        # D log carries; 2 856 was the bulk record read as a step.
        max_tags = {c.uuid: (c.max_tag_d_mib, c.max_tag_p_mib) for c in table.cards}
        self.assertEqual(max_tags[REAL_5090], (1608, 2988))
        self.assertEqual(max_tags[REAL_3080_A], (1010, 2986))
        self.assertEqual(max_tags[REAL_3080_B], (982, 2916))

    def test_the_launch_check_passes_on_that_boots_own_credit(self):
        # The credit here is the MINIMUM corridor sample, which is tighter than
        # the single sample spec section 2 quoted -- so the needs are larger and
        # the slacks smaller.  All six must still pass; if they did not, the
        # honest answer would be W32, not a looser instrument.
        table, _ = ring_table.solve(REAL_CARDS, EVIDENCE, ZR2)
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
        table = ring_table.RingTable(boot="b", instrument="i", lines_read=1, cards=[card],
                                     max_step_total_mib=30)
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


# ---------------------------------------------------------------------------
# FIX 1 (round 1) -- the defects the reviewer found in ec595b9aa9.
# ---------------------------------------------------------------------------


def _zr2_table():
    """Boot weg2zr2's own numbers, as a table, without touching the evidence tree."""
    rows = [
        ring_table.CardRing(uuid="GPU-aaaa", nvml_index=1, name="RTX 5090",
                            image_p_mib=13860, image_d_mib=13914,
                            max_tag_p_mib=2988, max_tag_d_mib=2856,
                            credit_d2p_mib=13441, credit_p2d_mib=12609),
        ring_table.CardRing(uuid="GPU-bbbb", nvml_index=0, name="RTX 3080",
                            image_p_mib=7548, image_d_mib=9680,
                            max_tag_p_mib=2986, max_tag_d_mib=2714,
                            credit_d2p_mib=7106, credit_p2d_mib=7741),
        ring_table.CardRing(uuid="GPU-cccc", nvml_index=2, name="RTX 3080",
                            image_p_mib=8504, image_d_mib=9370,
                            max_tag_p_mib=3336, max_tag_d_mib=2686,
                            credit_d2p_mib=7535, credit_p2d_mib=7033),
    ]
    return ring_table.RingTable(boot="weg2zr2", instrument="i", lines_read=1,
                                cards=rows, max_step_total_mib=3600)


class SerialFormLaunchCheckTest(unittest.TestCase):
    """W34: R5's corridor is not the requirement the SERIAL flip form has.

    RED before FIX 1: ``serial_refusals`` did not exist and ``prepare_host_ring``
    armed the ring on this very table -- which wedges the FIRST flip on every
    card, because S must acquire while W still holds its whole parked image.
    """

    def test_the_serial_requirement_is_the_parked_image_plus_the_first_step(self):
        c = _zr2_table().cards[0]
        self.assertEqual(c.need_serial_d2p_mib, 13860 + 2856)
        self.assertEqual(c.need_serial_p2d_mib, 13914 + 2988)
        # free at flip start = H - the parked image.  54 MiB against a 2 856 MiB
        # first acquire: the wedge, in one subtraction.
        self.assertEqual(c.h_mib - c.image_p_mib, 54)
        self.assertEqual(c.slack_serial_d2p_mib, 54 - 2856)

    def test_all_six_r5_cases_pass_while_all_six_serial_cases_refuse(self):
        table = _zr2_table()
        self.assertEqual(table.refusals(), [],
                         "R5 passes on this table -- which is exactly why W32 alone "
                         "let the ring arm into a deadlock")
        serial = table.serial_refusals()
        self.assertEqual(len(serial), 6, "\n".join(serial))
        self.assertIn("RING NEEDS INTERLEAVE", serial[0])
        for token in ("image_P 13860", "max_tag_D 2856", "H is 13914",
                      "only 54 MiB is free"):
            self.assertIn(token, serial[0])

    def test_the_p2d_direction_wedges_structurally_whenever_image_d_is_the_peak(self):
        # H = max(image_P, image_D).  When image_D is the peak, the host has
        # exactly ZERO free bytes at the start of P->D, so no step size however
        # small is funded.  A property of the form, not of a number.
        for c in _zr2_table().cards:
            self.assertEqual(c.image_d_mib, c.h_mib)
            self.assertEqual(c.slack_serial_p2d_mib, -c.max_tag_p_mib)

    def test_the_front_declares_the_leg_form_and_the_launcher_reads_it(self):
        from sglang.srt.weg2 import front, launcher

        self.assertEqual(front.FLIP_LEG_FORM, "serial")
        self.assertEqual(launcher._front_leg_form(), "serial",
                         "the launcher must READ the front's constant, not restate it")

    def _prepare(self, launcher, lines, form, leg_form="", directional=None,
                 table=None):
        table = _zr2_table() if table is None else table
        real = ring_table.solve
        ring_table.solve = lambda *a, **k: (table, "stubbed")
        try:
            return launcher.prepare_host_ring(
                [], lines.append, "t2", form, "/nonexistent", "", True,
                leg_form=leg_form, pcie_directional=directional)
        finally:
            ring_table.solve = real

    def test_auto_prints_w34_and_runs_the_old_form_instead_of_arming(self):
        from sglang.srt.weg2 import launcher

        lines = []
        plan = self._prepare(launcher, lines, form="auto")
        self.assertFalse(plan.armed, "arming here wedges the first flip")
        joined = "\n".join(lines)
        self.assertIn("W34 Weg2RingNeedsInterleave", joined)
        self.assertIn("OLD flip form runs", joined)
        self.assertNotIn("WEG2-HOST-RING ARMED", joined,
                         "the ARMED line must never certify a corridor for a flip "
                         "form this tree does not contain")

    def test_an_explicitly_named_form_raises_rather_than_downgrading(self):
        from sglang.srt.weg2 import launcher

        lines = []
        with self.assertRaises(ring_table.Weg2RingNeedsInterleave) as cm:
            self._prepare(launcher, lines, form="MAP_SHARED")
        self.assertIn("W34 Weg2RingNeedsInterleave", str(cm.exception))
        self.assertIn("RING NEEDS INTERLEAVE", str(cm.exception))

    def test_the_check_lifts_when_the_front_gathers_its_legs(self):
        from sglang.srt.weg2 import launcher

        lines = []
        plan = self._prepare(launcher, lines, form="auto", leg_form="interleave",
                             directional=True)
        self.assertTrue(plan.armed, "\n".join(lines))
        self.assertEqual(plan.form, "MAP_SHARED")
        self.assertIn("leg_form=interleave", "\n".join(lines))
        self.assertIn("pcie_lock_directional=True", "\n".join(lines))

    # -- FIX 2: the gate is keyed to BOTH halves of the hazard ---------------

    def test_the_gate_reads_the_lock_fact_from_the_module_that_owns_it(self):
        from sglang.srt.managers import weg2_memory_saver
        from sglang.srt.weg2 import launcher

        # The direction split is C12/C13, a LATER slice: today the per-card
        # PCIe lock is keyed on the uuid alone and held around a whole leg.
        self.assertFalse(weg2_memory_saver.PCIE_LOCK_SEPARATES_DIRECTIONS)
        self.assertFalse(launcher._pcie_lock_directional(),
                         "the launcher must READ the saver's constant, not restate it")

    def test_a_gathered_front_alone_does_not_open_the_gate(self):
        # The moment C9 flips front.FLIP_LEG_FORM to 'interleave' this is the
        # ONLY thing standing between the ring and the deadlock W34 exists to
        # prevent: S blocks in a bounded acquire while holding the per-card lock,
        # W's release RPC cannot run on that card, and the budget expires into
        # W31 -> group-fatal W4.
        from sglang.srt.weg2 import launcher

        lines = []
        plan = self._prepare(launcher, lines, form="auto", leg_form="interleave",
                             directional=False)
        self.assertFalse(plan.armed, "\n".join(lines))
        joined = "\n".join(lines)
        self.assertIn("W34 Weg2RingNeedsInterleave", joined)
        self.assertIn("PCIE_LOCK_SEPARATES_DIRECTIONS is False", joined)
        self.assertNotIn("WEG2-HOST-RING ARMED", joined)

    def test_l6_prints_the_serial_row_beside_the_r5_row(self):
        line = _zr2_table().format_l6()[0]
        self.assertIn("SERIAL FORM need_d2p=16716", line)
        self.assertIn("need_d2p=6263", line, "R5's row must still be there")


class RingTableBootPinTest(unittest.TestCase):
    """FIX 1: ``--ring-table-boot`` is a substring pin, and it never raises."""

    def setUp(self):
        import tempfile

        self.dir = tempfile.mkdtemp(prefix="weg2ringpin-")
        for stem in ("boot_weg2_zzz1_aaaaaaaaaa_0907_010101",
                     "boot_weg2_zzz2_bbbbbbbbbb_0907_020202"):
            for suffix in (".P.log", ".D.log", ".front.log"):
                open(os.path.join(self.dir, stem + suffix), "w").close()

    def test_a_missing_pin_returns_a_reason_and_never_an_oserror(self):
        table, reason = ring_table.solve([], self.dir, "weg2zr2")
        self.assertIsNone(table)
        self.assertIn("matches 0 of the 2 complete boots", reason)

    def test_an_ambiguous_pin_returns_a_reason_naming_the_candidates(self):
        table, reason = ring_table.solve([], self.dir, "zzz")
        self.assertIsNone(table)
        self.assertIn("matches 2 of the 2 complete boots", reason)
        self.assertIn("zzz1", reason)

    def test_the_boot_tag_alone_resolves_the_stem(self):
        # The recorded evidence line quotes '--ring-table-boot weg2zr2'; before
        # FIX 1 that opened '<dir>/weg2zr2.P.log' and raised FileNotFoundError.
        table, reason = ring_table.solve([], self.dir, "zzz1")
        self.assertIsNone(table, "empty logs cannot solve")
        self.assertIn("boot_weg2_zzz1_aaaaaaaaaa_0907_010101", reason)
        self.assertNotIn("matches", reason, "the pin itself resolved")


class OldFormPriceTest(unittest.TestCase):
    """FIX 1: the DEFAULT (un-armed) path must still fund an arm.

    RED before FIX 1: the un-armed charge was ``Sigma H + Sigma_c max_k``, which
    prices a step the flip never takes -- each card's largest tag is a different
    tag -- and it made every arm of the ladder refuse with W20.
    """

    def test_one_tag_is_one_rpc_so_the_sum_is_over_cards_and_the_max_over_tags(self):
        # weights_0 is 100+100+20 = 220; weights_1 is 40+40+280 = 360.  Each
        # card's own largest tag summed is 100+100+280 = 480, but no single RPC
        # ever holds 480 -- the maxima are different tags.  The largest step is
        # 360, and 120 MiB of the difference is a charge nothing ever incurs.
        rows = [ring_table.CardRing(uuid=f"GPU-{i}", nvml_index=i, name="c",
                                    image_p_mib=1000, image_d_mib=1000,
                                    max_tag_p_mib=100, max_tag_d_mib=100)
                for i in range(3)]
        rows[2].max_tag_d_mib = 280
        table = ring_table.RingTable(boot="b", instrument="i", lines_read=1,
                                     cards=rows, max_step_total_mib=360)
        from sglang.srt.weg2 import launcher

        old = launcher.HostRingPlan(form="", armed=False, table=table)
        self.assertEqual(old.host_weights_bytes, (3000 + 360) * MIB)
        self.assertNotEqual(old.host_weights_bytes,
                            (3000 + 100 + 100 + 280) * MIB)

    def test_a_family_root_tag_is_a_bulk_record_not_a_step(self):
        import tempfile

        d = tempfile.mkdtemp(prefix="weg2bulkroot-")
        path = os.path.join(d, "g.log")
        with open(path, "w") as f:
            for rank, records in enumerate((
                (("weights_0", 1200), ("weights_1", 1100), ("weights", 2856)),
                (("weights_0", 700), ("weights_1", 900), ("weights", 1600)),
            )):
                for tag, mib in records:
                    f.write(f"[2026-09-07 21:10:23 TP{rank}] WEG2-CHUNK-BYTES sleep "
                            f"tags=['{tag}'] host_image_delta={mib} MiB\n")
        g = ring_table.parse_group_log(path)
        max_tag, steps = g.max_tag, g.tag_totals
        self.assertEqual(max_tag[0], 1200,
                         "'weights' names the family that 'weights_<k>' partitions, "
                         "so its delta is a whole pass and not a corridor step")
        self.assertEqual(max_tag[1], 900)
        self.assertNotIn("weights", steps)
        # The step total is summed OVER RANKS -- one tag is one RPC whose shards
        # land on every card at once, so a per-rank maximum is not the step.
        self.assertEqual(steps["weights_0"], 1200 + 700)
        self.assertEqual(steps["weights_1"], 1100 + 900)

    @unittest.skipUnless(os.path.isdir(EVIDENCE), "no evidence tree")
    def test_the_default_path_still_selects_an_arm_at_the_record_box(self):
        from sglang.srt.weg2 import launcher

        table, reason = ring_table.solve(REAL_CARDS, EVIDENCE, ZR2)
        self.assertIsNotNone(table, reason)
        old = launcher.HostRingPlan(form="", armed=False, table=table)
        arm, store, _ = host_ledger.choose(
            int(118.05 * GIB), int(103.95 * GIB), store_min_gib=8.0,
            ring_bytes=old.host_weights_bytes,
            ring_span1_bytes=old.host_weights_span1_bytes,
            ring_provenance=old.provenance)
        self.assertGreaterEqual(arm.launch_leftover_gib, 0.0)
        self.assertGreaterEqual(store, 8.0)


class MemfdArmIsDeletedTest(unittest.TestCase):
    """FIX 1: the step-0 probe retired the memfd form 25 min before the commit.

    It was also broken by construction: the launcher published an fd NUMBER and
    the ranks are ``spawn``-started scheduler processes, so that number named a
    different object -- or nothing -- in the process that mmapped it.
    """

    def test_the_launcher_offers_only_auto_none_and_the_proven_form(self):
        import argparse

        from sglang.srt.weg2 import launcher

        seen = {}
        real = argparse.ArgumentParser.add_argument

        def spy(self, *a, **k):
            if a and a[0] == "--ring-form":
                seen["choices"] = k.get("choices")
            return real(self, *a, **k)

        argparse.ArgumentParser.add_argument = spy
        try:
            with self.assertRaises(SystemExit):
                launcher.main(["--help"])
        finally:
            argparse.ArgumentParser.add_argument = real
        self.assertEqual(seen.get("choices"), ["auto", "none", "MAP_SHARED"])

    def test_no_source_of_this_slice_still_creates_or_passes_a_ring_fd(self):
        from sglang.srt.weg2 import launcher

        src = open(launcher.__file__).read()
        for token in ("memfd_create", "pass_fds", '"MEMFD"', "'MEMFD'"):
            self.assertNotIn(token, src, f"{token} survives in the launcher")

    def test_the_c_side_takes_a_path_and_validates_the_object_before_mmap(self):
        from sglang.srt.weg2 import launcher

        cpp = os.path.join(os.path.dirname(launcher.__file__),
                           "tms_csrc", "host_ring.cpp")
        src = open(cpp).read()
        self.assertNotIn("entry.fd", src)
        # The exact guard, not just the call: a short-circuited or negated
        # condition leaves the call in place while the refusal stops firing.
        guard = ("if (fstat(ring->fd_, &st) != 0 ||\n"
                 "        static_cast<uint64_t>(st.st_size) < "
                 "static_cast<uint64_t>(ring->map_bytes_)) {")
        self.assertIn(guard, src)
        self.assertLess(src.index(guard), src.index("ring->base_ = mmap("),
                        "the size check must precede the mapping, not follow it")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class NoMeasuredTableRefusesTheBootTest(unittest.TestCase):
    """FIX 2 finding 1: the R22 arm may not promise a fallback it cannot run.

    RED before the fix: the printed line said "the OLD flip form runs" while
    ``host_weights_bytes`` returned 0 and the ledger killed the boot two steps
    later with W20 -- the operator was told a fallback had run that never did.
    """

    def _plan(self, lines):
        from sglang.srt.weg2 import launcher

        real = ring_table.solve
        ring_table.solve = lambda *a, **k: (None, "no boot in /x carries all three logs")
        try:
            return launcher.prepare_host_ring(
                [], lines.append, "t2", "auto", "/x", "", True, leg_form="serial")
        finally:
            ring_table.solve = real

    def test_the_r22_line_says_the_boot_refuses_and_names_what_follows(self):
        lines = []
        plan = self._plan(lines)
        joined = "\n".join(lines)
        self.assertIsNone(plan.table)
        self.assertFalse(plan.armed)
        self.assertIn("WEG2-HOST-RING R22", joined)
        self.assertIn("THIS BOOT REFUSES", joined)
        self.assertIn("W20 Weg2HostLedgerRefused", joined)
        self.assertNotIn("the OLD flip form runs", joined,
                         "the OLD form is priced from the SAME table; with no "
                         "table it cannot run either")

    def test_the_line_and_the_refusal_that_follows_it_agree(self):
        lines = []
        plan = self._plan(lines)
        # What the line promises IS what happens: no host weights term, and the
        # ledger stops the launch by name rather than pricing a guess.
        self.assertEqual(plan.host_weights_bytes, 0)
        self.assertEqual(plan.host_weights_span1_bytes, 0)
        with self.assertRaises(host_ledger.Weg2HostLedgerRefused) as cm:
            host_ledger.choose(int(118 * GIB), int(104 * GIB), store_min_gib=8.0,
                               ring_bytes=plan.host_weights_bytes,
                               ring_span1_bytes=plan.host_weights_span1_bytes,
                               ring_provenance=plan.provenance)
        msg = str(cm.exception)
        self.assertIn("THIS BOOT REFUSES", msg)
        self.assertNotIn("and the OLD flip form is what would run.", msg)


class RefusalsPropagateAsRefusalsTest(unittest.TestCase):
    """FIX 2 finding 2: a named refusal may not leave the launcher as a traceback."""

    def test_every_ring_refusal_shares_one_base_class(self):
        for exc in (ring_table.Weg2RingCreditRefused, ring_table.Weg2RingNeedsInterleave):
            self.assertTrue(issubclass(exc, ring_table.Weg2RingRefused), exc)

    def test_each_named_refusal_becomes_the_one_line_and_exit_2(self):
        import io
        from contextlib import redirect_stdout

        from sglang.srt.weg2 import launcher

        cases = [
            launcher.Weg2LaunchRefused("W1 test"),
            ring_table.Weg2RingCreditRefused("W32 test"),
            # RED before the fix: this one was not in the except list, so it
            # left as a traceback with exit 1 -- read as a crash by anything
            # keying on the code.
            ring_table.Weg2RingNeedsInterleave("W34 test"),
            host_ledger.Weg2HostLedgerRefused("W20 test"),
        ]
        real = launcher.main
        try:
            for exc in cases:
                def boom(argv=None, _e=exc):
                    raise _e

                launcher.main = boom
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = launcher.cli([])
                self.assertEqual(rc, 2, type(exc).__name__)
                self.assertIn("WEG2-LAUNCH REFUSED", buf.getvalue())
                self.assertIn(str(exc), buf.getvalue())
        finally:
            launcher.main = real


class CardIdentityIsByUuidTest(unittest.TestCase):
    """FIX 2 finding 4: rank/nvml positions never name a card across boots."""

    def setUp(self):
        import tempfile

        self.dir = tempfile.mkdtemp(prefix="weg2ringident-")
        self.stem = "boot_weg2_t2_0000000000_0907_000000"

    def _write(self, identity=True, flip_tag=False):
        one = [{0: [("weights_0", 10)]}]
        for prefix, name in (("PP", "P"), ("TP", "D")):
            text = _group_log(prefix, one, {0: 1.0})
            if flip_tag:
                text += (
                    f"[2026-09-07 21:10:24 {prefix}0] WEG2-FLIP-TAG group={name} "
                    f"rank=0 card={CARDS[0].uuid} dir=d2h tag=weights_0 bytes=42 MiB\n"
                )
            with open(os.path.join(self.dir, f"{self.stem}.{name}.log"), "w") as f:
                f.write(text)
        with open(os.path.join(self.dir, f"{self.stem}.front.log"), "w") as f:
            f.write(_front_log({"P": [{1: 100}], "D": [{1: 100}]}, identity=identity))

    def test_a_source_boot_without_an_identity_line_is_refused_by_name(self):
        self._write(identity=False)
        table, reason = ring_table.solve([CARDS[0]], self.dir, self.stem)
        self.assertIsNone(table, "positional identity is never assumed")
        self.assertIn("NVML -> CUDA ordinal map", reason)
        self.assertIn("never assumes card identity", reason)

    def test_the_flip_tag_line_names_its_own_card(self):
        self._write(flip_tag=True)
        table, reason = ring_table.solve([CARDS[0]], self.dir, self.stem)
        self.assertIsNotNone(table, reason)
        self.assertEqual(table.cards[0].uuid, CARDS[0].uuid)
        self.assertIn("WEG2-FLIP-TAG", table.instrument)

    @unittest.skipUnless(os.path.isfile(os.path.join(EVIDENCE, f"{ZR2}.front.log")),
                         f"{EVIDENCE}/{ZR2} not present")
    def test_a_re_enumeration_between_two_boots_does_not_swap_two_rows(self):
        # The rig rules state NVML enumeration can shift between boots.  Same
        # cards, this boot's indices permuted: every row must follow its UUID.
        # RED before the fix: the two 3080 rows swapped, H moved by 2.8 GiB, and
        # the L6 line printed a uuid beside the other card's numbers.
        straight, _ = ring_table.solve(REAL_CARDS, EVIDENCE, ZR2)
        # BOTH positional keys moved: the list ORDER (which fed rank -> card)
        # and the nvml indices (which fed the corridor lookup).
        shuffled = [
            FakeCard(0, REAL_3080_B, "NVIDIA GeForce RTX 3080"),
            FakeCard(2, REAL_5090, "NVIDIA GeForce RTX 5090", 32607),
            FakeCard(1, REAL_3080_A, "NVIDIA GeForce RTX 3080"),
        ]
        moved, _ = ring_table.solve(shuffled, EVIDENCE, ZR2)

        def rows(t):
            return {c.uuid: (c.image_p_mib, c.image_d_mib, c.max_tag_p_mib,
                             c.max_tag_d_mib, c.credit_d2p_mib, c.credit_p2d_mib)
                    for c in t.cards}

        self.assertEqual(rows(straight), rows(moved),
                         "a row followed a position instead of its card")
        self.assertEqual(straight.total_h_bytes, moved.total_h_bytes)


class ResidentImageInstrumentTest(unittest.TestCase):
    """FIX 2 finding 5: the census is a lower bound, and the absolute is not the answer."""

    def setUp(self):
        import tempfile

        self.dir = tempfile.mkdtemp(prefix="weg2ringresident-")
        self.stem = "boot_weg2_t2_0000000000_0907_000000"

    def _solve(self, p_rss, d_rss):
        one = [{0: [("weights_0", 300)]}]
        with open(os.path.join(self.dir, f"{self.stem}.P.log"), "w") as f:
            f.write(_group_log("PP", one, {0: 1.0}, None, p_rss))
        with open(os.path.join(self.dir, f"{self.stem}.D.log"), "w") as f:
            f.write(_group_log("TP", one, {0: 1.0}, None, d_rss))
        with open(os.path.join(self.dir, f"{self.stem}.front.log"), "w") as f:
            f.write(_front_log({"P": [{1: 100}], "D": [{1: 100}]}))
        table, reason = ring_table.solve([CARDS[0]], self.dir, self.stem)
        self.assertIsNotNone(table, reason)
        return table.cards[0]

    def test_the_resident_image_wins_when_it_exceeds_the_weight_tag_census(self):
        # The census sums the tags the sleep loop NAMES; the resident image
        # holds every backed-up tag.  Record 1p STEP-0 ADDENDUM: size H from the
        # measured image, never from the weight tags alone.
        c = self._solve({0: (4000, 5000)}, {0: (4000, 4400)})
        self.assertEqual(c.tags_p_mib, 300)
        self.assertEqual(c.dormant_p_mib, 1000)
        self.assertEqual(c.image_p_mib, 1000, "the larger instrument is charged")
        self.assertEqual(c.image_d_mib, 400)
        self.assertEqual(c.h_mib, 1000)

    def test_the_absolute_is_not_charged_because_the_baseline_is_posted_elsewhere(self):
        # The absolute RssShmem of a sleeping rank also contains that group's
        # HiCache ring and mamba anchors, which host_ledger posts by name
        # (rings_gib / anchors_gib).  Charging the absolute here would post the
        # same bytes twice.  MEASURED on boot weg2dk6 group P: absolute peak
        # 15 831 MiB, baseline 1 971, delta 13 860 = the census exactly.
        c = self._solve({0: (4000, 5000)}, {0: (4000, 4400)})
        self.assertNotEqual(c.image_p_mib, 5000)
        self.assertEqual(c.image_p_mib, 5000 - 4000)

    def test_a_boot_that_logged_no_absolute_still_solves_from_the_census(self):
        c = self._solve(None, None)
        self.assertEqual(c.dormant_p_mib, 0)
        self.assertEqual(c.image_p_mib, 300, "an absent instrument is not a zero image")
