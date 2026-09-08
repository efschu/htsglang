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

from sglang.srt.weg2 import front, host_ledger, ring_table

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

    ``rss`` = {rank: (lowest, highest)}, still written onto the line because
    the real emitter writes it -- but it is NO LONGER an instrument here.  The
    predecessor derived a "resident image" from peak-minus-baseline of exactly
    these two readings; the emitter takes both immediately around the weights
    pause loop, so that number was the weight-tag census a second time (carried
    review finding 1).  A1-2's cross-check is the front's WEG2 DORMANT-IMAGE
    line instead, written by :func:`_front_log`.
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


def _dormant_line(group: str, rss_gib: float, tags_gib: float) -> str:
    """The front's own WEG2 DORMANT-IMAGE line, in the emitter's exact shape.

    Copied from ``host_ledger.format_dormant_image`` on the branch that emits
    it (draft-KV fix 8) rather than re-invented: this module READS that line, so
    a fixture in any other shape would test a parser against itself.
    """
    return (
        f"[2026-09-07T21:12:00Z] INFO weg2.front: WEG2 DORMANT-IMAGE group={group} "
        f"shmem_delta_gib=1.23 rss_shmem_gib={rss_gib:.2f} "
        f"weight_tags_gib={tags_gib:.2f} extra_gib={rss_gib - tags_gib:.2f} "
        "(pids [1, 2, 3] of [1, 2, 3]; INTERLEAVED: the shmem delta is confounded"
        "; run_residual_gib=none (x); boot t2 @ deadbeef at 2026-09-07T21:12:00Z)"
    )


def _front_log(free_by_phase, cards=None, identity=True, dormant=(), arm=None,
               instrument=front.CORRIDOR_INSTRUMENT) -> str:
    """A fixture front log.

    ``instrument`` is the ``instrument=`` token the corridor samples carry, and
    it DEFAULTS TO WHAT THE TREE EMITS TODAY (allocatable free).  Fix 3 made
    the unit load-bearing: the ring credit converts a pre-fix log's samples
    before crediting them, so a fixture that silently omitted the token would
    be asserting ring arithmetic against a converted number.  Pass
    ``instrument=None`` for the pre-fix shape deliberately.
    """
    out = []
    if arm is not None:
        # The SOURCE boot's own chosen ledger arm.  Without it the RssShmem
        # cross-check has no subtrahend and may not raise a census (FIX 1
        # finding 3) -- so a fixture that wants the netted route must write it.
        out.append(
            f"[2026-09-07T21:05:01Z] WEG2-HOST-LEDGER CHOSEN S={arm[0]} GB "
            f"(--hicache-size, both groups) M={arm[1]} MiB "
            "(--hicache-mamba-host-mib, both groups) store=4 GiB tmpfs"
        )
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
    token = f"instrument={instrument} " if instrument else ""
    for phase, samples in free_by_phase.items():
        for row in samples:
            free = " ".join(f"nvml{i}:free={v}MiB" for i, v in row.items())
            out.append(
                f"[2026-09-07 21:07:54,029] INFO weg2.front: WEG2-CORRIDOR "
                f"phase={phase}(awake) epoch=0 {token}{free} min_so_far={{}}"
            )
    for group, rss_gib, tags_gib in dormant:
        out.append(_dormant_line(group, rss_gib, tags_gib))
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
        self.assertIn("42 lines, instrument: WEG2-FLIP-TAG bytes", line)

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
        # A1-2: the charged image is the PER-CARD tag census; weg2zr2 predates
        # both C16 and the front's WEG2 DORMANT-IMAGE line, so there is no
        # cross-check and the census stands -- marked BOUND, because a
        # weights-only census is a lower bound on the dormant image.  The
        # predecessor charged 20 MiB more per D card from a "resident image"
        # that was the same census read a second time (carried finding 1).
        self.assertEqual(got[REAL_5090], (13914, 13860, 13914))
        self.assertEqual(got[REAL_3080_A], (9680, 7548, 9680))
        self.assertEqual(got[REAL_3080_B], (9370, 8504, 9370))
        self.assertTrue(all(c.dormant_p_bound and c.dormant_d_bound for c in table.cards))
        census = {c.uuid: (c.tags_d_mib, c.tags_p_mib) for c in table.cards}
        self.assertEqual(census[REAL_5090], (13914, 13860))
        self.assertEqual(census[REAL_3080_A], (9680, 7548))
        self.assertEqual(census[REAL_3080_B], (9370, 8504))
        self.assertEqual(table.total_h_bytes // MIB, 32964)
        self.assertEqual(table.total_span1_bytes // MIB, 29912)
        self.assertEqual(table.total_tags_mib, 32964)
        self.assertEqual(table.total_dormant_mib, 0)
        # Both instruments in the provenance line, neither quotable alone, and
        # the absent one printed as ABSENT rather than as a zero.
        self.assertIn("per-card tag census 32964 MiB", table.provenance())
        self.assertIn("measured dormant image unmeasured MiB", table.provenance())
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


class RingCreditIsAllocatableFreeTest(unittest.TestCase):
    """FIX 3: ``credit(S->W)`` reads its source boot's UNIT, not just its number.

    Disclosed by fix 1 and left open there: a pre-fix boot's WEG2-CORRIDOR
    samples are allocatable free PLUS that card's driver carve-out, and
    :func:`ring_table.solve` credited them raw.  Every card was over-credited
    by 425/518/425 MiB, which UNDERSTATES ``need = image_W - credit + max_tag_S
    + max_tag_W`` by the same amount -- so an R5 case that should have been
    refused at launch arms instead, and the flip it funds is short by the
    carve-out.  The unsafe direction, and the same defect the boot arm had one
    level up: two readers grading a number against a rule stated in another
    unit.

    Every assertion here fails on the parent commit 4302724e8b.
    """

    def setUp(self):
        import tempfile

        self.dir = tempfile.mkdtemp(prefix="weg2ringunit-")
        self.stem = "boot_weg2_unit_0000000000_0908_000000"

    def _write(self, corridor, instrument, cards=REAL_CARDS):
        one = {0: [("weights_0", 4000)]}
        with open(os.path.join(self.dir, f"{self.stem}.P.log"), "w") as f:
            f.write(_group_log("PP", [one], {0: 0.0}))
        with open(os.path.join(self.dir, f"{self.stem}.D.log"), "w") as f:
            f.write(_group_log("TP", [one], {0: 0.0}))
        with open(os.path.join(self.dir, f"{self.stem}.front.log"), "w") as f:
            f.write(_front_log(corridor, cards=cards, instrument=instrument))

    def _solve(self, cards=None):
        return ring_table.solve(cards or [REAL_CARDS[0]], self.dir, self.stem)

    #: The 5090 is ordinal 0 = nvml 1 in REAL_CARDS, carve-out 518 MiB.
    CORRIDOR = {"P": [{1: 1441}], "D": [{1: 993}]}

    def test_a_pre_fix_source_is_credited_minus_the_carve_out(self):
        self._write(self.CORRIDOR, instrument=None)
        table, reason = self._solve()
        self.assertIsNotNone(table, reason)
        c = table.cards[0]
        self.assertEqual(c.credit_d2p_mib, 993 - 518)
        self.assertEqual(c.credit_p2d_mib, 1441 - 518)

    def test_a_post_fix_source_is_credited_as_it_stands(self):
        self._write(self.CORRIDOR, instrument=front.CORRIDOR_INSTRUMENT)
        table, reason = self._solve()
        self.assertIsNotNone(table, reason)
        c = table.cards[0]
        self.assertEqual((c.credit_d2p_mib, c.credit_p2d_mib), (993, 1441))

    def test_the_carve_out_blind_v2less_source_is_still_allocatable_free(self):
        """``nvml_v1_free,allocatable(carve-out-unknown)``: the FREE is fine.

        Both NVML structs report the same allocatable ``free``; only the
        carve-out beside it is missing in v1.  Subtracting anything here would
        double-count a carve-out that was never added.
        """
        self._write(self.CORRIDOR, instrument=front.CORRIDOR_INSTRUMENT_NO_V2)
        table, _ = self._solve()
        self.assertEqual(table.cards[0].credit_d2p_mib, 993)

    def test_the_correction_is_printed_in_the_ring_provenance_line(self):
        self._write(self.CORRIDOR, instrument=None)
        table, _ = self._solve()
        prov = table.provenance()
        self.assertIn("CREDIT unit: ALLOCATABLE free", prov)
        self.assertIn(f"source instrument {ring_table.CORRIDOR_INSTRUMENT_PRE_FIX}", prov)
        self.assertIn("nvml1-518", prov)
        self.assertIn("keyed by card UUID", prov)
        # And on the operator-facing L6 line, which carries the provenance.
        self.assertIn("nvml1-518", table.format_l6()[0])

    def test_a_card_with_no_measured_or_recorded_carve_out_refuses_the_boot(self):
        """A missing carve-out may never become a zero: that IS the defect."""
        self._write({"P": [{1: 1441}], "D": [{1: 993}]}, instrument=None, cards=CARDS)
        table, reason = ring_table.solve([CARDS[0]], self.dir, self.stem)
        self.assertIsNone(table)
        self.assertIn("no measured or recorded driver carve-out", reason)
        self.assertIn("refused rather than graded", reason)

    def test_the_callers_own_measured_carve_out_wins_over_the_record(self):
        """``Card.reserved_mib`` is this rig's live registry snapshot."""
        self._write(self.CORRIDOR, instrument=None)
        measured = FakeCard(1, REAL_5090, "NVIDIA GeForce RTX 5090", 32607)
        measured.reserved_mib = 500
        table, reason = self._solve([measured])
        self.assertIsNotNone(table, reason)
        self.assertEqual(table.cards[0].credit_d2p_mib, 993 - 500)
        self.assertIn("matched by UUID", table.provenance())

    def test_an_unknown_instrument_token_refuses_rather_than_credits(self):
        self._write(self.CORRIDOR, instrument="free_by_some_future_reader")
        table, reason = self._solve()
        self.assertIsNone(table)
        self.assertIn("names neither the allocatable unit", reason)

    def test_the_understatement_is_exactly_the_carve_out(self):
        """The consequence, stated as the arithmetic R5 actually runs."""
        self._write(self.CORRIDOR, instrument=None)
        pre_fix_table, _ = self._solve()
        self._write(self.CORRIDOR, instrument=front.CORRIDOR_INSTRUMENT)
        as_if_allocatable, _ = self._solve()
        self.assertEqual(
            pre_fix_table.cards[0].need_d2p_mib
            - as_if_allocatable.cards[0].need_d2p_mib,
            518,
            "crediting front units understates need by exactly the carve-out",
        )


@unittest.skipUnless(os.path.isfile(os.path.join(EVIDENCE, f"{ZR2}.front.log")),
                     f"{EVIDENCE}/{ZR2} not present")
class RealBootCreditsAreReDerivedTest(unittest.TestCase):
    """The numbers the next boot's ARMED line is checked against.

    All four candidate source boots are PRE-FIX, so every credit the launcher
    would compute today moves by that card's carve-out.  These are the
    re-derived figures, recorded here and in
    ``/spinning/gpu-arb/weg2/WEG2_BUILD_DECISIONS_0906.md`` so a boot's own
    RING line can be checked against them rather than re-derived by hand.
    """

    RG3 = "boot_weg2_weg2rg3_5b015ad139_0908_041053"

    #: ``{uuid: (credit_d2p, credit_p2d, need_d2p, need_p2d, slack_d2p, slack_p2d)}``
    #: -- allocatable free, i.e. each card's corridor minimum minus its
    #: carve-out (425/518/425), plus the kv that group releases.
    RG3_EXPECTED = {
        REAL_5090: (11756, 11250, 6700, 7260, 7214, 6654),
        REAL_3080_A: (5779, 7018, 5765, 6658, 3915, 3022),
        REAL_3080_B: (6157, 6218, 6245, 7050, 3125, 2320),
    }
    ZR2_EXPECTED = {
        REAL_5090: (11652, 11318, 6804, 7192, 7110, 6722),
        REAL_3080_A: (5803, 6940, 5741, 6736, 3939, 2944),
        REAL_3080_B: (6179, 6232, 6223, 7036, 3147, 2334),
    }

    def _check(self, stem, expected):
        if not os.path.isfile(os.path.join(EVIDENCE, f"{stem}.front.log")):
            self.skipTest(f"{stem} not in the evidence tree")
        table, reason = ring_table.solve(REAL_CARDS, EVIDENCE, stem)
        self.assertIsNotNone(table, reason)
        got = {
            c.uuid: (c.credit_d2p_mib, c.credit_p2d_mib, c.need_d2p_mib,
                     c.need_p2d_mib, c.slack_d2p_mib, c.slack_p2d_mib)
            for c in table.cards
        }
        self.assertEqual(got, expected)
        self.assertEqual(table.refusals(), [], "\n".join(table.refusals()))
        self.assertIn("nvml1-518", table.provenance())
        return table

    def test_rg3_credits_are_the_re_derived_allocatable_numbers(self):
        table = self._check(self.RG3, self.RG3_EXPECTED)
        # H is an IMAGE quantity and does not move with the credit: the credit
        # enters `need`, and an over-credit hid a need that was 425-518 MiB
        # larger per card.  Stated because "the ring was sized too small" is
        # the tempting shorthand and it names the wrong term.
        self.assertEqual(table.total_h_bytes // MIB, 32964)
        self.assertEqual(table.total_span1_bytes // MIB, 29912)

    def test_zr2_credits_are_the_re_derived_allocatable_numbers(self):
        self._check(ZR2, self.ZR2_EXPECTED)


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
        # FIX 3 round 3: the un-armed charge is still computed and printed --
        # the arithmetic is what a refusal has to show -- but its provenance no
        # longer NAMES a form as running, because A1-3 leaves none.
        self.assertIn("NO ARMED FORM", old.provenance)
        self.assertIn("refuses", old.provenance)
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

        self.assertEqual(front.FLIP_LEG_FORM, "interleave",
                         "C9: the gathered pair is the only leg form this front has")
        self.assertEqual(launcher._front_leg_form(), "interleave",
                         "the launcher must READ the front's constant, not restate it")

    def _prepare(self, launcher, lines, form, leg_form="", directional=None,
                 table=None, duplex_probe=""):
        table = _zr2_table() if table is None else table
        real = ring_table.solve
        ring_table.solve = lambda *a, **k: (table, "stubbed")
        try:
            return launcher.prepare_host_ring(
                [], lines.append, "t2", form, "/nonexistent", "", True,
                leg_form=leg_form, pcie_directional=directional,
                duplex_probe=duplex_probe)
        finally:
            ring_table.solve = real

    def test_a_launcher_without_gathered_legs_REFUSES_by_name_and_never_arms(self):
        # W34 is UNREACHABLE BY CONSTRUCTION on this tip -- no state of this tree
        # sets FLIP_LEG_FORM to anything but 'interleave' -- so it is driven here
        # through the injection point that a stale worktree on PYTHONPATH would
        # produce.  What is asserted is that such a launcher REFUSES: there is no
        # OLD flip form to fall back to on this host budget (A1-3).
        from sglang.srt.weg2 import launcher

        lines = []
        with self.assertRaises(ring_table.Weg2RingNeedsInterleave) as cm:
            self._prepare(launcher, lines, form="auto", leg_form="serial")
        self.assertIn("W34 Weg2RingNeedsInterleave", str(cm.exception))
        self.assertIn("RING NEEDS INTERLEAVE", str(cm.exception))
        self.assertNotIn("WEG2-HOST-RING ARMED", "\n".join(lines),
                         "the ARMED line must never certify a corridor for a flip "
                         "form this tree does not contain")

    def test_no_arm_of_this_launcher_claims_the_old_flip_form_runs(self):
        # CARRIED REVIEW FINDING 2.  The sentence "the host ring is NOT armed and
        # the OLD flip form runs (spec section 10.4)" stood in the W32 / W33 /
        # W34 arms while the metal had already refuted it: A1-3 rules the OLD
        # form INFEASIBLE on this host budget, and boot weg2rg1 died of W20 two
        # steps after printing it.  A refusal that names a fallback must have one.
        from sglang.srt.weg2 import launcher

        for form, leg_form, exc in (
            ("auto", "serial", ring_table.Weg2RingNeedsInterleave),
            ("MAP_SHARED", "serial", ring_table.Weg2RingNeedsInterleave),
            ("none", "interleave", launcher.Weg2RingFormUnproven),
        ):
            lines = []
            with self.assertRaises(exc) as cm:
                self._prepare(launcher, lines, form=form, leg_form=leg_form)
            joined = "\n".join(lines) + str(cm.exception)
            self.assertNotIn("the OLD flip form runs", joined)
            self.assertIn("REFUSE", joined.upper())

    def test_every_refusal_of_this_module_leaves_cli_as_the_named_line_and_exit_2(self):
        # Boot weg2rg1 saw W34 as a RAW TRACEBACK with exit 1, which any wrapper
        # keying on the exit code reads as a crash rather than as the refusal it
        # is.  Every arm must inherit ``Weg2RingRefused`` and therefore cli()'s
        # handler -- asserted on the CLASS, so a future arm cannot opt out.
        from sglang.srt.weg2 import launcher

        for exc in (ring_table.Weg2RingCreditRefused,
                    ring_table.Weg2RingNeedsInterleave,
                    launcher.Weg2RingFormUnproven):
            self.assertTrue(issubclass(exc, ring_table.Weg2RingRefused))
            self.assertTrue(issubclass(exc, launcher.REFUSALS))

    def test_the_check_lifts_when_the_front_gathers_its_legs(self):
        from sglang.srt.weg2 import launcher

        lines = []
        plan = self._prepare(launcher, lines, form="auto", leg_form="interleave")
        self.assertTrue(plan.armed, "\n".join(lines))
        self.assertEqual(plan.form, "MAP_SHARED")
        self.assertIn("leg_form=interleave", "\n".join(lines))

    # -- A1-4: the direction split is PER CARD and is not part of this gate --

    def test_the_gate_reads_the_lock_fact_from_the_module_that_owns_it(self):
        from sglang.srt.managers import weg2_memory_saver
        from sglang.srt.weg2 import launcher

        # The fact now lives WITH the key (review nb1): a function that takes a
        # card, not a hand-typed module boolean beside a keyer that had no
        # direction parameter at all.
        ratios = {"GPU-aaaa": 1.759, "GPU-bbbb": 1.316}
        self.assertTrue(weg2_memory_saver.pcie_lock_separates_directions(
            "GPU-aaaa", ratios=ratios))
        self.assertFalse(weg2_memory_saver.pcie_lock_separates_directions(
            "GPU-bbbb", ratios=ratios))
        self.assertFalse(weg2_memory_saver.pcie_lock_separates_directions(
            "GPU-unmeasured", ratios=ratios),
            "an unmeasured card never splits")

        # FIX 1 round 1: whether the key SPLITS is a decision, not a reading of
        # the ratio, and the launcher makes it once.  Under gathered legs every
        # card splits -- one key is a DEADLOCK there, not a slowdown -- so no
        # card is serialised and R5 is every card's requirement.
        cards = ["GPU-aaaa", "GPU-bbbb"]
        gathered = launcher._split_decisions(cards, ratios, "interleave")
        self.assertEqual(gathered, {"GPU-aaaa": True, "GPU-bbbb": True})
        published_gathered = "format=v2,GPU-aaaa=1.759:split,GPU-bbbb=1.316:split"
        self.assertEqual(
            launcher._serialised_cards(cards, published_gathered, "interleave"), [])
        # A front that does NOT gather has no pair in flight to deadlock, so
        # there the ratio gate stands and the card below it keeps one key.
        serial = launcher._split_decisions(cards, ratios, "serial")
        self.assertEqual(serial, {"GPU-aaaa": True, "GPU-bbbb": False})
        self.assertEqual(
            launcher._serialised_cards(cards, "format=v2,GPU-aaaa=1.759:split,"
                                       "GPU-bbbb=1.316:single", "serial"), cards,
            "a front that does not gather serialises every card, whatever its key")
        # And the RANK reads the same decision the launcher published, through
        # the module that owns the key -- not the ratio it was derived from.
        published = "format=v2,GPU-aaaa=1.759:split,GPU-bbbb=1.316:split"
        self.assertEqual(weg2_memory_saver.duplex_splits(published),
                         {"GPU-aaaa": True, "GPU-bbbb": True})
        self.assertTrue(weg2_memory_saver.pcie_lock_separates_directions(
            "GPU-bbbb", splits=weg2_memory_saver.duplex_splits(published)),
            "the published decision outranks the ratio it came from")
        self.assertTrue(weg2_memory_saver.pcie_lock_path(
            "GPU-bbbb", lock_dir="/dev/shm", direction="d2h",
            splits=weg2_memory_saver.duplex_splits(published)).endswith(".d2h.lock"),
            "and the KEY the rank builds is the one the decision names")

    @staticmethod
    def _v1_reader(published, uuids):
        """The PARENT tree's resolution (5b015ad139), verbatim, as a key count.

        This is not a model of an old reader -- it is what
        ``git show 5b015ad139:...weg2_memory_saver.py`` does: split rows on
        ``,``, split each on the FIRST ``=``, take ``<ratio>[:decision]``, and
        build ``<uuid>.<direction>`` when the row resolved.  Every other weg2
        worktree on this box is that shape, which is why the launcher's wire
        format has to survive it.
        """
        ratios, splits = {}, {}
        for item in published.split(","):
            uuid, _, value = item.strip().partition("=")
            uuid = uuid.strip()
            ratio_text, _, decision = value.partition(":")
            if decision.strip() in ("split", "single"):
                splits[uuid] = decision.strip() == "split"
            try:
                ratios[uuid] = float(ratio_text)
            except ValueError:
                continue
        out = {}
        for u in uuids:
            if u in splits:
                separates = splits[u]
            else:
                separates = ratios.get(u) is not None and ratios[u] >= 1.5
            out[u] = 2 if separates else 1
        return out

    def test_the_version_is_a_ROW_and_an_older_reader_still_resolves_EVERY_card(self):
        # FIX 3 round 3, BLOCKING FINDING 1, and the mutant fix 2 left alive.
        #
        # A version PREFIX ("v2|<uuid>=...") does not make an unknown-format
        # reader drop everything -- it mangles exactly ONE uuid, the FIRST, and
        # resolves the rest correctly.  On this rig the first card is the 5090:
        # the card carrying the co-located P/D pair and the flip's critical
        # path.  So the prefix gave that one card a SINGLE key while the
        # launcher printed SPLIT for it and armed -- boot weg2rg2's precondition
        # verbatim (S blocks in acquire holding the key W's release needs ->
        # W31 at the 120 s budget -> W29 on every rank -> group-fatal W4), on
        # the worst card, 120 s into a flip that has already mutated VRAM.  It
        # was a REGRESSION, not an incomplete fix: the UN-versioned string it
        # replaced resolved all three cards correctly in that same reader.
        #
        # As a ROW the version costs the old reader one dropped line
        # (float("v2") raises, exactly as an unparsable row already did) and
        # costs the real cards nothing.
        from sglang.srt.managers import weg2_memory_saver
        from sglang.srt.weg2 import launcher

        lines = []
        plan = self._prepare(launcher, lines, form="auto", leg_form="interleave")
        published = plan.duplex_env
        uuids = [c.uuid for c in _zr2_table().cards]
        self.assertTrue(uuids, "the fixture must publish at least one card")

        # PRODUCER (this is the assertion mutant M4 of the fix-2 review survived:
        # replacing the version token with "" left every suite green).
        self.assertTrue(
            published.startswith(
                f"{weg2_memory_saver.PCIE_DUPLEX_VERSION_KEY}="
                f"{weg2_memory_saver.PCIE_DUPLEX_FORMAT},"),
            f"the published string must lead with the version ROW: {published!r}")
        self.assertNotIn("|", published,
                         "a prefix is what mangled the first card; there is no "
                         f"bar in this format: {published!r}")

        # CONSUMER, this tree: the row is consumed and never mistaken for a card.
        self.assertEqual(
            sorted(weg2_memory_saver.duplex_splits(published)), sorted(uuids),
            "the version row must not appear as a card, and no card may go missing")
        self.assertNotIn(weg2_memory_saver.PCIE_DUPLEX_VERSION_KEY,
                         weg2_memory_saver.duplex_ratios(published))

        # CONSUMER, the PARENT tree -- the compatibility claim, as a test that
        # can fail rather than a docstring.  Every real card resolves to TWO
        # keys there; nothing is mangled.
        self.assertEqual(
            self._v1_reader(published, uuids), {u: 2 for u in uuids},
            "an older reader must resolve EVERY card from this string")

        # ... and the shape that must never come back: the prefix form, in the
        # same reader, single-keys exactly the first card.
        prefixed = (f"{weg2_memory_saver.PCIE_DUPLEX_FORMAT}|"
                    + published.partition(",")[2])
        self.assertEqual(
            self._v1_reader(prefixed, uuids)[uuids[0]], 1,
            "the regression this test exists for: the prefix costs the FIRST "
            "card its split, which is the deadlock, not a slowdown")

    def test_a_published_string_with_no_version_ROW_REFUSES(self):
        # The other direction of the same seam: fix 2's own wire format, and
        # the un-versioned one before it, reaching THIS reader.  Neither is
        # guessed at -- a dialect this tree cannot name is a named stop, since
        # reading it anyway is what mangles a card.
        from sglang.srt.managers import weg2_memory_saver

        for published in ("GPU-aaaa=1.759:split,GPU-bbbb=1.316:split",
                          "v2|GPU-aaaa=1.759:split,GPU-bbbb=1.316:split"):
            with self.assertRaises(
                    weg2_memory_saver.Weg2DuplexDecisionRefused,
                    msg=f"{published!r} must not resolve") as cm:
                weg2_memory_saver.duplex_splits(published)
            self.assertIn("W36 Weg2DuplexDecisionRefused", str(cm.exception))
            self.assertIn("No fallback", str(cm.exception))
            # AND IT MUST DIAGNOSE THE ABSENT ROW, not report a version of
            # None.  Deleting this arm still refuses -- the unknown-version arm
            # catches it, since None != "v2" -- so a test that only asserts
            # "something was raised" cannot tell the two apart.  They have
            # different fixes: an absent row means an older launcher published
            # this string, an unknown one means a newer launcher did.
            self.assertIn(
                f"carries no {weg2_memory_saver.PCIE_DUPLEX_VERSION_KEY}="
                f"{weg2_memory_saver.PCIE_DUPLEX_FORMAT} row", str(cm.exception))
        # The EMPTY string stays the honest "nothing was published", not a
        # refusal -- the launch check reads it as SERIALISED and refuses there.
        self.assertEqual(weg2_memory_saver.duplex_splits(""), {})

    def test_an_unknown_wire_format_version_REFUSES_instead_of_one_key(self):
        # FIX 2 round 2, finding 2.  The predecessor's format
        # (<uuid>=<ratio>:split) was a SILENT superset of the one before it
        # (<uuid>=<ratio>): the older parser did float("1.759:split"), dropped
        # every row, resolved an empty table, and therefore took ONE key on ALL
        # THREE cards -- strictly worse than boot weg2rg2, which had one key on
        # nvml0 only, and with no line saying so anywhere.  A format that
        # degrades to the unsafe answer is not a format.
        from sglang.srt.managers import weg2_memory_saver

        with self.assertRaises(weg2_memory_saver.Weg2DuplexDecisionRefused) as cm:
            weg2_memory_saver.duplex_splits("format=v3,GPU-aaaa=1.759:split")
        self.assertIn("W36 Weg2DuplexDecisionRefused", str(cm.exception))
        self.assertIn("'v3'", str(cm.exception))
        # ... and no fallback is claimed, because there is none.
        self.assertIn("No fallback", str(cm.exception))

    def test_decisions_published_but_none_parsed_REFUSES(self):
        # The other half of the same mismatch: a non-empty published string that
        # names no split/single row at all.  Resolving that to R17's ratio gate
        # is the silent degrade -- the ratio grades whether the split is WORTH
        # having, never whether the single key is SAFE, and under gathered legs
        # it is not (boot weg2rg2: W31 -> W29 -> group-fatal W4).
        from sglang.srt.managers import weg2_memory_saver
        from sglang.srt.weg2 import launcher

        with self.assertRaises(weg2_memory_saver.Weg2DuplexDecisionRefused) as cm:
            weg2_memory_saver.duplex_splits("GPU-aaaa=1.759,GPU-bbbb=1.316")
        self.assertIn("W36 Weg2DuplexDecisionRefused", str(cm.exception))
        # The EMPTY string is not a refusal: nothing was published, which is an
        # honest state, and it resolves to no split -- which the launch check
        # then reads as SERIALISED and refuses at launch (W34).
        self.assertEqual(weg2_memory_saver.duplex_splits(""), {})
        self.assertEqual(
            launcher._serialised_cards(["GPU-aaaa", "GPU-bbbb"], "", "interleave"),
            ["GPU-aaaa", "GPU-bbbb"])

    def test_the_launch_check_resolves_the_KEY_THE_RANK_WILL_BUILD(self):
        # FIX 2 round 2, finding 2 -- THE TAUTOLOGY.  The predecessor passed
        # _split_decisions' own answer back into
        # pcie_lock_separates_directions(splits=...), so the saver returned it
        # verbatim and the check could not see a rank that would resolve the
        # published string differently -- which is the ONE thing it exists to
        # see.  The launcher's W34 comment names that scenario ("a stale
        # worktree on PYTHONPATH, a partial rebase") as the one the arm exists
        # for; the leg-form half was caught, the KEY half was not.
        from sglang.srt.managers import weg2_memory_saver
        from sglang.srt.weg2 import launcher

        cards = ["GPU-aaaa", "GPU-bbbb"]
        ratios = {"GPU-aaaa": 1.759, "GPU-bbbb": 1.316}
        splits = launcher._split_decisions(cards, ratios, "interleave")
        published = ",".join(
            [f"{weg2_memory_saver.PCIE_DUPLEX_VERSION_KEY}="
             f"{weg2_memory_saver.PCIE_DUPLEX_FORMAT}"]
            + [f"{u}={'%.3f' % ratios[u]}:{'split' if splits[u] else 'single'}"
               for u in cards])
        # The check the launch gate was missing: feed the launcher's OWN
        # published string to the saver's parser and compare the resolved key
        # per card against the launcher's decision per card.
        for uuid in cards:
            keys = {weg2_memory_saver.pcie_lock_path(
                uuid, lock_dir="/dev/shm", direction=d,
                splits=weg2_memory_saver.duplex_splits(published))
                for d in weg2_memory_saver.PCIE_DIRECTIONS}
            self.assertEqual(
                len(keys) == len(weg2_memory_saver.PCIE_DIRECTIONS), splits[uuid],
                f"{uuid}: the launcher decided {splits[uuid]} and the rank builds "
                f"{len(keys)} key(s)")

    # -- FIX 3 round 3, BLOCKING FINDING 2: the check must ask the RANK's TREE --

    @staticmethod
    def _stub_tree(root, body):
        """A REAL second tree: ``<root>/python/sglang/srt/managers/``, importable.

        The point of the shape is that nothing here is monkeypatched.  The
        launch check has to reach this module the way a rank reaches it -- by
        ``PYTHONPATH=<tree>/python`` in a process of its own -- which is exactly
        the divergence an in-process import cannot produce.
        """
        pkg = os.path.join(root, "python", "sglang", "srt", "managers")
        os.makedirs(pkg, exist_ok=True)
        for d in (os.path.join(root, "python", "sglang"),
                  os.path.join(root, "python", "sglang", "srt"), pkg):
            open(os.path.join(d, "__init__.py"), "w").close()
        with open(os.path.join(pkg, "weg2_memory_saver.py"), "w") as fh:
            fh.write(body)
        return root

    #: The PARENT tree's saver (5b015ad139) reduced to the two names the probe
    #: touches, with its resolution verbatim: ``float()`` over the whole
    #: right-hand side, so every ``<ratio>:<decision>`` row drops, the table is
    #: empty, and ONE path comes back for both directions.
    _STALE_SAVER = (
        "import os\n"
        "PCIE_DIRECTIONS = ('d2h', 'h2d')\n"
        "PCIE_DUPLEX_ENV = 'SGLANG_WEG2_PCIE_DUPLEX'\n"
        "def pcie_lock_path(u, *, lock_dir=None, direction=None,\n"
        "                   ratios=None, splits=None):\n"
        "    table = {}\n"
        "    for item in os.environ.get(PCIE_DUPLEX_ENV, '').split(','):\n"
        "        k, _, v = item.partition('=')\n"
        "        try:\n"
        "            table[k.strip()] = float(v)\n"
        "        except ValueError:\n"
        "            continue\n"
        "    key = u\n"
        "    r = table.get(u)\n"
        "    if direction and r is not None and r >= 1.5:\n"
        "        key = '%s.%s' % (key, direction)\n"
        "    return '/dev/shm/.weg2-pcie-serialize-%s.lock' % key\n"
    )

    def test_the_launch_check_asks_THE_TREE_THE_RANKS_IMPORT(self):
        # FIX 3 round 3, BLOCKING FINDING 2.  Fix 2's check resolved the key
        # through the LAUNCHER's own weg2_memory_saver, never through the saver
        # at --tree -- the module the ranks import via PYTHONPATH=<tree>/python
        # (build_env).  So a stale-tree deployment is exactly what it could not
        # see, which is the ONE thing the arm exists for, and its pinning test
        # monkeypatched inside the launcher's own process: not the deployment
        # shape.  Here the second tree is REAL and the answer comes from a
        # process that imported it.
        import sys
        import tempfile

        from sglang.srt.weg2 import launcher

        cards = ["GPU-aaaa", "GPU-bbbb"]
        published = "format=v2,GPU-aaaa=1.759:split,GPU-bbbb=1.316:split"
        with tempfile.TemporaryDirectory() as root:
            tree = self._stub_tree(root, self._STALE_SAVER)
            notes = []
            self.assertEqual(
                launcher._serialised_cards(cards, published, "interleave",
                                           py=sys.executable, tree=tree,
                                           notes=notes),
                cards,
                "a tree whose saver cannot read this wire format single-keys "
                "every card, and the launch check must report that from the "
                "tree's own answer -- not from its own import")
            self.assertTrue(any(tree in n for n in notes), notes)

    def test_a_rank_tree_that_cannot_answer_makes_EVERY_card_serialised(self):
        # The probe dying is itself the named refusal, one step earlier: no
        # module at --tree (a partial rebase, a tree with no weg2 slice at all)
        # must not read as "all cards split".  It reads as SERIALISED, which is
        # W34 and exit 2 before either group starts.
        import sys
        import tempfile

        from sglang.srt.weg2 import launcher

        cards = ["GPU-aaaa", "GPU-bbbb"]
        published = "format=v2,GPU-aaaa=1.759:split,GPU-bbbb=1.316:split"
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "python"), exist_ok=True)
            notes = []
            self.assertEqual(
                launcher._serialised_cards(cards, published, "interleave",
                                           py=sys.executable, tree=root,
                                           notes=notes),
                cards)
            self.assertTrue(any("KEY-PROBE FAILED" in n for n in notes), notes)

    def test_a_rank_tree_that_CAN_read_the_format_resolves_SPLIT(self):
        # The can-fail half: the check must not be "everything is serialised".
        # Pointed at THIS tree -- the one a real boot names with --tree -- the
        # subprocess resolves two keys per card and no card is serialised.
        import sys

        from sglang.srt.weg2 import launcher

        root = launcher.__file__
        for _ in range(5):  # launcher.py -> weg2 -> srt -> sglang -> python -> tree
            root = os.path.dirname(root)
        cards = ["GPU-aaaa", "GPU-bbbb"]
        published = "format=v2,GPU-aaaa=1.759:split,GPU-bbbb=1.316:split"
        notes = []
        self.assertEqual(
            launcher._serialised_cards(cards, published, "interleave",
                                       py=sys.executable, tree=root,
                                       notes=notes),
            [], "\n".join(notes))
        self.assertTrue(any("RANK TREE" in n for n in notes), notes)

    def test_a_saver_that_cannot_read_the_format_is_reported_SERIALISED(self):
        # The DESK path of the same property (no --tree given): an older saver
        # answers the question wrongly and silently.  Here it is, verbatim --
        # the parent's duplex_ratios did float() on the whole right-hand side,
        # so every row dropped, the table was empty, and pcie_lock_path returned
        # ONE path for both directions.  The launch check must see that and hand
        # the card to W34, not arm.  The cross-process shape of this is
        # test_the_launch_check_asks_THE_TREE_THE_RANKS_IMPORT above; this one
        # keeps the in-process branch honest.
        from sglang.srt.managers import weg2_memory_saver
        from sglang.srt.weg2 import launcher

        cards = ["GPU-aaaa", "GPU-bbbb"]
        published = "format=v2,GPU-aaaa=1.759:split,GPU-bbbb=1.316:split"
        real = weg2_memory_saver.pcie_lock_path

        def old_pcie_lock_path(nvml_uuid, *, lock_dir=None, direction=None,
                               ratios=None, splits=None):
            # the parent's resolution: float("1.759:split") raises, row dropped
            table = {}
            for item in os.environ.get(
                    weg2_memory_saver.PCIE_DUPLEX_ENV, "").split(","):
                uuid, _, value = item.partition("=")
                try:
                    table[uuid.strip()] = float(value)
                except ValueError:
                    continue
            key = nvml_uuid
            ratio = table.get(nvml_uuid)
            if direction and ratio is not None and ratio >= 1.5:
                key = f"{key}.{direction}"
            return f"/dev/shm/.weg2-pcie-serialize-{key}.lock"

        weg2_memory_saver.pcie_lock_path = old_pcie_lock_path
        try:
            self.assertEqual(
                launcher._serialised_cards(cards, published, "interleave"), cards,
                "an old saver takes one key on EVERY card, and the launch check "
                "must see the key it will actually build, not the launcher's dict")
        finally:
            weg2_memory_saver.pcie_lock_path = real

    def test_no_module_outside_the_decision_claims_a_card_keeps_ONE_key(self):
        # FIX 2 round 2, finding 4, as a structural guard.  FIX 1 made every key
        # split under gathered legs and closed the false sentence in the
        # launcher's printed lines -- and left three callers asserting the
        # opposite, one of them (front.py's L5 comment) instructing the operator
        # to read ZERO OVERLAP on the flip's critical-path card as expected
        # behaviour.  Under this tree zero overlap there means the key did NOT
        # split, i.e. the blocking finding has occurred, and the comment told
        # the reader to dismiss it.
        from sglang.srt.weg2 import launcher

        cards = ["GPU-aaaa", "GPU-bbbb"]
        self.assertEqual(
            set(launcher._split_decisions(
                cards, {"GPU-aaaa": 1.759, "GPU-bbbb": 1.316}, "interleave").values()),
            {True},
            "the premise: under gathered legs every card splits, whatever its ratio")
        root = os.path.dirname(os.path.dirname(os.path.abspath(launcher.__file__)))
        # The dead sentences, verbatim.  They are NOT re-quoted anywhere in the
        # tree -- a retraction that reprints the claim gives the next grep a
        # false positive and the next reader a true one -- so any occurrence is
        # an assertion and an offender.
        banned = ("keeps the single key", "keep the single key",
                  "serialises the legs by design", "still serialise by design",
                  "serialise by design")
        offenders = []
        for rel in ("weg2/front.py", "weg2/ring_table.py", "weg2/host_ledger.py",
                    "weg2/launcher.py", "managers/weg2_memory_saver.py"):
            path = os.path.join(root, rel)
            if not os.path.exists(path):
                continue
            for n, line in enumerate(open(path, errors="replace"), 1):
                if any(b in line for b in banned):
                    offenders.append(f"{rel}:{n}: {line.strip()}")
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_no_module_tells_the_reader_the_OLD_FLIP_FORM_RUNS(self):
        # FIX 3 round 3, the SAME CLASS one seam over, and the survivors the
        # fix-2 guard could not reach because it banned only key phrases.
        # Amendment A1-3 rules the OLD flip form INFEASIBLE on this host budget
        # and every refusal arm now says so -- but four sentences still told the
        # reader it RUNS when no ring arms: the state file's "none (OLD flip
        # form)", build_env's popped-family comment, HostRingPlan.provenance,
        # ring_table's module docstring and its R22 comment.  Verified on the
        # rig in the fix-2 review: a --ring-table-boot pin matching no boot
        # gives W20 and rc=2, i.e. the R22 path REFUSES.  A sentence promising a
        # fallback that does not exist is the class the metal refuted, and it is
        # worse here than in a docstring: it is what an operator reads while
        # deciding whether the boot that just exited did something.
        from sglang.srt.weg2 import launcher

        root = os.path.dirname(os.path.dirname(os.path.abspath(launcher.__file__)))
        # The dead sentences, by their RUN claim -- the bare name "OLD flip
        # form" stays legal, because every refusal arm has to be able to name
        # the form it is refusing to run.
        banned = ("(OLD flip form)", "OLD flip form (no ring)",
                  "OLD serial flip form run", "runs the OLD flip form",
                  "runs the OLD form")
        offenders = []
        for rel in ("weg2/front.py", "weg2/ring_table.py", "weg2/host_ledger.py",
                    "weg2/launcher.py", "managers/weg2_memory_saver.py"):
            path = os.path.join(root, rel)
            if not os.path.exists(path):
                continue
            for n, line in enumerate(open(path, errors="replace"), 1):
                if any(b in line for b in banned):
                    offenders.append(f"{rel}:{n}: {line.strip()}")
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_a_card_below_the_gate_is_checked_against_the_SERIAL_requirement(self):
        # BOOT weg2rg2, THE KILLER, AS A TEST.  The predecessor computed the
        # duplex answer, printed it on the ARMED line, and gated NOTHING: "a card
        # that keeps one key costs time on that card, and nothing else".  The
        # metal refuted that 20 minutes later.  nvml0 (measured 1.316,
        # DUPLEX-NULL) passed R5 on both directions, the SAME L6 line said
        # SERIAL FORM slack=1122/-2986 MiB, the launcher armed, and the FIRST
        # flip wedged: W31 Weg2HostRingExhausted need=24 free=12 MiB after
        # 120.189 s in acquire, then W29 on all three P ranks, then W4.
        from sglang.srt.weg2 import launcher

        table = _zr2_table()
        self.assertEqual(table.refusals(), [],
                         "R5 passes on this table -- the check that armed weg2rg2")
        lines = []
        with self.assertRaises(ring_table.Weg2RingNeedsInterleave) as cm:
            self._prepare(launcher, lines, form="auto", leg_form="interleave",
                          directional=False)
        msg = str(cm.exception)
        self.assertIn("W34 Weg2RingNeedsInterleave", msg)
        self.assertIn("the measured duplex ratio does not reach R17's gate", msg)
        self.assertIn("needs image_D 9680 + max_tag_P 2986 = 12666 MiB", msg,
                      "the arithmetic that would have caught weg2rg2 must be in "
                      "the refusal, per card and per direction")
        self.assertNotIn("WEG2-HOST-RING ARMED", "\n".join(lines))

    def test_gathered_legs_split_every_key_so_no_card_is_held_to_the_serial_rule(self):
        # THE FIX, GREEN.  weg2rg2's own table arms once the key stops
        # serialising the pair -- R5 passes 6/6 on it, and R5 is the right
        # requirement precisely because both legs can then be in flight on every
        # card.  The predecessor armed the same table with nvml0 on ONE key,
        # which is the state that wedged 120.189 s into the first flip.
        from sglang.srt.weg2 import launcher

        lines = []
        plan = self._prepare(launcher, lines, form="auto", leg_form="interleave")
        joined = "\n".join(lines)
        self.assertTrue(plan.armed, joined)
        self.assertIn("serialised_cards=none", joined)
        self.assertIn("key=SPLIT per direction", joined)
        self.assertIn("the R5 corridor (W32) alone", joined)
        for card in _zr2_table().cards:
            self.assertIn(f"{card.uuid}=:split", plan.duplex_env,
                          "the decision is PUBLISHED to the ranks, or the key "
                          "the metal takes is not the one that was checked")

    def test_the_serial_requirement_can_be_restricted_to_named_cards(self):
        # ``only`` is what makes the check per card at all.  H = max(image_P,
        # image_D), so on ANY card one of the two serial directions exceeds H by
        # a whole max_tag -- the requirement is a property of the FORM.  What
        # ``only`` decides is which cards have to meet it.
        table = _zr2_table()
        self.assertEqual(len(table.serial_refusals()), 6)
        rows = table.serial_refusals(only=["GPU-bbbb"])
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertIn("GPU-bbbb", row)
        self.assertEqual(table.serial_refusals(only=[]), [])

    def test_the_armed_line_says_which_cards_were_held_to_which_inequality(self):
        from sglang.srt.weg2 import launcher

        rows = [ring_table.CardRing(uuid="GPU-aaaa", nvml_index=1, name="c",
                                    image_p_mib=1000, image_d_mib=1000,
                                    max_tag_p_mib=100, max_tag_d_mib=100,
                                    credit_d2p_mib=200, credit_p2d_mib=200)]
        table = ring_table.RingTable(boot="b", instrument="i", lines_read=1,
                                     cards=rows, max_step_total_mib=100)
        lines = []
        plan = self._prepare(launcher, lines, form="auto", leg_form="interleave",
                             table=table)
        joined = "\n".join(lines)
        self.assertTrue(plan.armed, joined)
        self.assertIn("serialised_cards=none", joined)
        self.assertIn("WEG2-HOST-RING CHECK card=GPU-aaaa", joined)
        self.assertIn("the R5 corridor (W32) alone", joined)
        self.assertNotIn("pcie_lock_directional_on_every_card", joined,
                         "the printed-but-ungating boolean is gone: what the "
                         "ARMED line certifies must be what was checked")

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


class DormantImageInstrumentTest(unittest.TestCase):
    """A1-2 + carried review finding 1: what SIZES H(c), and what only checks it.

    RED-FIRST, verified by mutation:

    * charge the census when the measurement exceeds it   -> H is short by the
                                                             whole excess
    * apportion the excess EQUALLY instead of by share    -> a card is charged
                                                             bytes another
                                                             card's line
                                                             accounts for
    * drop the BOUND marking for group D                  -> a bound prints as
                                                             a measurement
    * restore the peak-minus-baseline RssShmem span       -> the "second"
                                                             instrument agrees
                                                             with the first by
                                                             construction
    """

    def setUp(self):
        import tempfile

        self.dir = tempfile.mkdtemp(prefix="weg2ringdormant-")
        self.stem = "boot_weg2_t2_0000000000_0907_000000"

    def _solve(self, dormant=(), cards=None, rss=None, arm=None, scale=1):
        cards = cards or [CARDS[0]]
        merged = {i: [("weights_0", 1024 * scale * (i + 1))] for i in range(len(cards))}
        kv = {i: 1.0 for i in range(len(cards))}
        with open(os.path.join(self.dir, f"{self.stem}.P.log"), "w") as f:
            f.write(_group_log("PP", [merged], kv, None, rss))
        with open(os.path.join(self.dir, f"{self.stem}.D.log"), "w") as f:
            f.write(_group_log("TP", [merged], kv, None, rss))
        with open(os.path.join(self.dir, f"{self.stem}.front.log"), "w") as f:
            f.write(_front_log(
                {"P": [{c.nvml_index: 100 for c in cards}],
                 "D": [{c.nvml_index: 100 for c in cards}]},
                cards=cards, dormant=dormant, arm=arm,
            ))
        table, reason = ring_table.solve(cards, self.dir, self.stem)
        self.assertIsNotNone(table, reason)
        return table

    def test_the_rss_span_of_the_chunk_line_is_no_longer_an_instrument(self):
        # THE CARRIED FINDING.  The emitter takes both absolutes immediately
        # around the weights pause loop, so peak-minus-baseline IS the census a
        # second time.  A boot whose absolutes are wildly larger than its deltas
        # must therefore change NOTHING about the charged image.
        loud = self._solve(rss={0: (4000, 90000)})
        quiet = self._solve(rss={0: (0, 0)})
        self.assertEqual(loud.cards[0].image_p_mib, quiet.cards[0].image_p_mib)
        self.assertEqual(loud.cards[0].image_p_mib, 1024)

    def test_the_cross_check_may_not_raise_a_census_before_the_terms_are_netted(self):
        # FIX 1 round 1, finding 3.  RssShmem is the sleeping group's WHOLE
        # shared-memory residency, and the ledger already posts the mamba
        # anchors and the HiCache rings against the same host budget by name.
        # Scaling a census up to an un-netted RssShmem charges those bytes a
        # SECOND time -- Sigma H walks from ~32 toward ~42 GiB and the ledger
        # W20-refuses at every rung, which is A1-3's state for the old form.
        rows, source, bound = ring_table.apportion_dormant(
            "P", ring_table.DormantImage("P", 38630, 28830, 9800),
            {"a": 20000, "b": 8830}, True,
        )
        self.assertEqual(rows, {"a": 20000, "b": 8830},
                         "with no subtrahend the census stands unraised")
        self.assertIn("REPORTED but NOT CHARGED", source)
        self.assertIn("charge those bytes a second time", source)

    def test_a_netted_measurement_above_the_census_is_charged_and_split_by_share(self):
        # Subtract, THEN raise.  Census 900, RssShmem 2000, of which 200 is the
        # group's own anchors + rings: 1800 is chargeable and splits 600 / 1200,
        # each card's SHARE of the census it is being corrected against.
        rows, source, bound = ring_table.apportion_dormant(
            "P", ring_table.DormantImage("P", 2000, 900, 1100),
            {"a": 300, "b": 600}, True, non_backup_mib=200,
        )
        self.assertEqual(rows, {"a": 600, "b": 1200})
        self.assertEqual(sum(rows.values()), 1800,
                         "the apportioned rows must sum to the NETTED "
                         "measurement, or bytes are lost or double-charged")
        self.assertFalse(bound)
        self.assertIn("route = netted RssShmem", source)
        self.assertIn("minus the ledger's own posted non-backup host terms", source)

    def test_the_subtrahend_is_the_source_boot_s_own_arm_not_a_number_typed_here(self):
        import tempfile

        path = os.path.join(tempfile.mkdtemp(prefix="weg2arm-"), "f.log")
        with open(path, "w") as f:
            f.write(_front_log({"P": [{1: 10}], "D": [{1: 10}]}, arm=(1, 2400)))
        self.assertEqual(ring_table.parse_chosen_arm(path), (1, 2400))
        # and the terms are host_ledger's OWN, split the way b0 measured them
        self.assertAlmostEqual(
            (host_ledger.ANCHORS_P_AT_2400_BYTES + host_ledger.ANCHORS_D_AT_2400_BYTES)
            / host_ledger.GB, 10.19, places=6,
            msg="the per-group halves must sum to the measured total, or the "
                "subtrahend and the ledger's own anchors term are two books")
        self.assertGreater(host_ledger.non_backup_host_bytes("D", 1, 2400),
                           host_ledger.non_backup_host_bytes("P", 1, 2400),
                           "record 1e: group D owns the 6xS ring half, P the 2xS")
        with self.assertRaises(ValueError):
            host_ledger.non_backup_host_bytes("X", 1, 2400)

    def test_an_unreadable_arm_leaves_the_measurement_reported_and_uncharged(self):
        # An absence is not a licence: without the source boot's arm there is no
        # subtrahend, so the census stands and the line says why.
        table = self._solve(dormant=[("P", 6.0, 3.0)])
        self.assertEqual(table.cards[0].image_p_mib, 1024)
        self.assertIn("REPORTED but NOT CHARGED", table.provenance())

    def test_a_measurement_below_the_netted_line_leaves_the_census_standing(self):
        table = self._solve(dormant=[("P", 0.5, 0.5)], arm=(1, 600))
        self.assertEqual(table.cards[0].image_p_mib, 1024)
        self.assertIn("does NOT exceed the per-card census", table.provenance())
        self.assertIn("anchors + rings", table.provenance())

    def test_group_D_borrows_only_the_NETTED_excess_and_stays_a_BOUND(self):
        # A1-2: D's first sleep is a flip and is interleaved, so no un-confounded
        # D sample exists.  D is charged its own census plus the excess P showed
        # AFTER P's own anchors and rings came out -- an un-netted excess would
        # double-book on D as well.
        p_non_backup = host_ledger.non_backup_host_bytes("P", 1, 600) // MIB
        table = self._solve(dormant=[("P", 40.0, 20.0)], arm=(1, 600), scale=8)
        c = table.cards[0]
        self.assertEqual(c.tags_d_mib, 8192)
        excess = int(round(20.0 * GIB / MIB)) - p_non_backup
        self.assertEqual(c.dormant_d_mib, 8192 + excess)
        self.assertTrue(c.dormant_d_bound)
        self.assertIn("BOUND, NOT A MEASUREMENT, for group(s) D", table.provenance())
        self.assertIn("came out", table.provenance())

    def test_a_ring_era_census_is_itself_the_measurement_and_is_not_a_bound(self):
        # C16's WEG2-FLIP-TAG names EVERY backed-up tag, so a boot that carries
        # it needs no uplift and its census is not a bound.
        for prefix, group, mib in (("PP", "P", 300), ("TP", "D", 400)):
            with open(os.path.join(self.dir, f"{self.stem}.{group}.log"), "w") as f:
                f.write(
                    f"[2026-09-07 21:10:23 {prefix}0] WEG2-FLIP-TAG group={group} rank=0 "
                    f"card={CARDS[0].uuid} dir=d2h tag=weights_0 bytes={mib} MiB "
                    f"population={ring_table.TAG_POPULATION_ALL} "
                    "(source: tms_tag_bytes, NOT RssShmem) ms=100 GB/s=3.00 granules=150\n"
                    f"[2026-09-07 21:06:30 {prefix}0] KV Cache is allocated. dtype: "
                    "torch.float8_e4m3fn, #tokens: 1, K size: 0.50 GB, V size: 0.50 GB\n"
                )
        with open(os.path.join(self.dir, f"{self.stem}.front.log"), "w") as f:
            f.write(_front_log({"P": [{1: 100}], "D": [{1: 100}]}))
        table, reason = ring_table.solve([CARDS[0]], self.dir, self.stem)
        self.assertIsNotNone(table, reason)
        self.assertFalse(table.cards[0].dormant_p_bound)
        self.assertFalse(table.cards[0].dormant_d_bound)
        self.assertIn("it IS A1-2's measured dormant image", table.provenance())

    def test_a_flip_tag_line_without_a_population_claim_stays_a_LOWER_BOUND(self):
        # FIX 1 round 1, finding 2.  The predecessor set covers_all on the FIRST
        # WEG2-FLIP-TAG line it saw, while the emitter looped over the weights
        # family alone -- so the first ring-era boot would have called a
        # weights-only census "the measured dormant image" and sized H from a
        # lower bound (dk7: 38.63 GiB measured against 28.83 of weight tags).
        # The claim now has to be ON the line.
        for prefix, group in (("PP", "P"), ("TP", "D")):
            with open(os.path.join(self.dir, f"{self.stem}.{group}.log"), "w") as f:
                f.write(
                    f"[2026-09-07 21:10:23 {prefix}0] WEG2-FLIP-TAG group={group} rank=0 "
                    f"card={CARDS[0].uuid} dir=d2h tag=weights_0 bytes=300 MiB "
                    f"population={ring_table.TAG_POPULATION_WEIGHTS} "
                    "(source: tms_tag_bytes, NOT RssShmem) ms=100 GB/s=3.00 granules=150\n"
                    f"[2026-09-07 21:06:30 {prefix}0] KV Cache is allocated. dtype: "
                    "torch.float8_e4m3fn, #tokens: 1, K size: 0.50 GB, V size: 0.50 GB\n"
                )
        with open(os.path.join(self.dir, f"{self.stem}.front.log"), "w") as f:
            f.write(_front_log({"P": [{1: 100}], "D": [{1: 100}]}))
        table, reason = ring_table.solve([CARDS[0]], self.dir, self.stem)
        self.assertIsNotNone(table, reason)
        self.assertTrue(table.cards[0].dormant_p_bound)
        self.assertIn("a LOWER BOUND", table.provenance())

    def test_a_flip_tag_line_with_NO_population_FIELD_stays_a_LOWER_BOUND(self):
        # FIX 2 round 2, finding 3.  The test above writes population=
        # weights-family EXPLICITLY, so it exercises the explicit-weak-claim
        # branch and never the ABSENT-claim branch its name promises.  The
        # absent branch is the live one: the emitter at 8c7f5d8d00 -- the commit
        # boots weg2rg1 and weg2rg2 actually ran -- writes WEG2-FLIP-TAG lines
        # with no population token at all, so the first table solved from a
        # weg2rg2-family log takes exactly this path.  ring_table's
        # ``m.group(5) or TAG_POPULATION_WEIGHTS`` IS the guard, and mutating
        # that default to TAG_POPULATION_ALL -- i.e. reinstating round 1's
        # finding 2 -- left the whole suite green.  It does not any more.
        #
        # The h2d line is here for a second reason: _TAG_RE matches dir=\S+, so
        # ANY direction's line can flip covers_all.  A silent claim must not
        # reach the census through the direction nobody was looking at.
        for prefix, group in (("PP", "P"), ("TP", "D")):
            with open(os.path.join(self.dir, f"{self.stem}.{group}.log"), "w") as f:
                f.write(
                    f"[2026-09-07 21:10:23 {prefix}0] WEG2-FLIP-TAG group={group} rank=0 "
                    f"card={CARDS[0].uuid} dir=d2h tag=weights_0 bytes=300 MiB "
                    "(source: tms_tag_bytes, NOT RssShmem) ms=100 GB/s=3.00 granules=150\n"
                    f"[2026-09-07 21:10:24 {prefix}0] WEG2-FLIP-TAG group={group} rank=0 "
                    f"card={CARDS[0].uuid} dir=h2d tag=weights_0 bytes=300 MiB "
                    "(source: tms_tag_bytes, NOT RssShmem) ms=100 GB/s=3.00 granules=150\n"
                    f"[2026-09-07 21:06:30 {prefix}0] KV Cache is allocated. dtype: "
                    "torch.float8_e4m3fn, #tokens: 1, K size: 0.50 GB, V size: 0.50 GB\n"
                )
        with open(os.path.join(self.dir, f"{self.stem}.front.log"), "w") as f:
            f.write(_front_log({"P": [{1: 100}], "D": [{1: 100}]}))
        table, reason = ring_table.solve([CARDS[0]], self.dir, self.stem)
        self.assertIsNotNone(table, reason)
        self.assertTrue(
            table.cards[0].dormant_p_bound,
            "a line that does not state its population states the WEAKER claim")
        self.assertTrue(table.cards[0].dormant_d_bound)
        self.assertIn("a LOWER BOUND", table.provenance())
        self.assertNotIn("it IS A1-2's measured dormant image", table.provenance())


class DuplexTableTest(unittest.TestCase):
    """C12/C13 + A1-4: the per-card duplex ratio, solved from the probe's lines."""

    PROBE = "/spinning/gpu-arb/weg2/PROBE_RING_0907.md"

    def test_an_unreadable_probe_is_a_named_reason_and_no_split(self):
        table, why = ring_table.solve_duplex("/nonexistent/probe.md")
        self.assertIsNone(table)
        self.assertIn("unreadable", why)

    def test_a_probe_without_rows_is_a_named_reason(self):
        import tempfile

        path = os.path.join(tempfile.mkdtemp(prefix="weg2probe-"), "p.md")
        with open(path, "w") as f:
            f.write("nothing measured here\n")
        table, why = ring_table.solve_duplex(path)
        self.assertIsNone(table)
        self.assertIn("carries no", why)

    def test_the_granule_row_is_read_not_the_one_block_row(self):
        # C3/C4 issue 2 MiB granules, so the granule row is the measurement that
        # describes the flip.  On the 5090 the two differ (1.821 block vs 1.759
        # granule) and reading the wrong one over-states the split's benefit.
        import tempfile

        path = os.path.join(tempfile.mkdtemp(prefix="weg2probe-"), "p.md")
        with open(path, "w") as f:
            f.write(
                "PROBE start mode=card carrier=map_shared uuid=GPU-aaaa ring_bytes=1\n"
                "PROBE duplex tag=block bytes_per_dir=1 ratio=1.821 verdict=DUPLEX-OK\n"
                "PROBE granule tag=block_2MiB granule=2097152 ratio=1.759 verdict=DUPLEX-OK\n"
            )
        table, why = ring_table.solve_duplex(path)
        self.assertIsNotNone(table, why)
        self.assertAlmostEqual(table.ratios["GPU-aaaa"], 1.759)

    @unittest.skipUnless(os.path.isfile(PROBE), f"{PROBE} not present")
    def test_the_real_probe_reproduces_the_record_1p_table(self):
        table, why = ring_table.solve_duplex(self.PROBE)
        self.assertIsNotNone(table, why)
        self.assertAlmostEqual(table.ratios[REAL_5090], 1.759)
        self.assertAlmostEqual(table.ratios[REAL_3080_A], 1.316)
        self.assertAlmostEqual(table.ratios[REAL_3080_B], 1.687)
        self.assertEqual(table.verdicts[REAL_3080_A], "DUPLEX-NULL")

    @unittest.skipUnless(os.path.isfile(PROBE), f"{PROBE} not present")
    def test_the_printed_table_names_the_x4_card_as_the_critical_path(self):
        table, _ = ring_table.solve_duplex(self.PROBE)
        lines = table.format_lines(REAL_CARDS, 1.5)
        by_card = {ln.split("card=")[1].split()[0]: ln for ln in lines}
        self.assertIn("reaches_gate=YES", by_card[REAL_5090])
        self.assertIn("reaches_gate=NO", by_card[REAL_3080_A])
        self.assertIn("gets no benefit", by_card[REAL_3080_A])
        for ln in lines:
            self.assertIn(self.PROBE, ln, "no ratio without its provenance")


#: One ``WEG2-FLIP-TAG`` line in the GATHERED-LEG shape, copied verbatim from
#: boot weg2rg5's own P log rather than re-invented -- ``group=?`` and
#: ``rank=-1`` are what ``weight_updater`` writes once C9 gathers the legs and
#: ``_weg2_rank()`` has no per-rank tag edge to report.  A fixture in any other
#: shape would test the parser against itself, which is exactly how the defect
#: below survived: every fixture wrote a rank the regex could already match.
RG5_TAG = (
    "[2026-09-08 05:06:49 PP2] WEG2-FLIP-TAG group=? rank=-1 card={card} "
    "dir=d2h tag={tag} bytes={mib} MiB population=all-backed-up-tags "
    "(source: tms_tag_bytes, NOT RssShmem) ms=837 GB/s=4.18 granules=1668"
)
RG5_BOOT = "boot_weg2_weg2rg5_15a46a611a_0908_050519"


class GatheredLegTagParsingTest(unittest.TestCase):
    """FIX 3 round 3, boot weg2rg5's finding 2: ``rank=-1`` never parsed."""

    def setUp(self):
        import tempfile

        self.dir = tempfile.mkdtemp(prefix="weg2tagparse-")

    def _log(self, rows):
        path = os.path.join(self.dir, "g.log")
        with open(path, "w") as f:
            for card, tag, mib in rows:
                f.write(RG5_TAG.format(card=card, tag=tag, mib=mib) + "\n")
        return path

    def test_a_gathered_leg_tag_line_is_READ_and_the_card_is_the_identity(self):
        # THE DEFECT: _TAG_RE required rank=(\d+), which cannot match a minus
        # sign, while the ring-era emitter has written rank=-1 since C9.  So the
        # instrument built to make MEASURED possible was the instrument that made
        # every ring-era boot ineligible -- max_tag parsed to {0,0,0},
        # CardRing.complete was false, the boot was skipped, and weg2rg5 sized
        # its ring from the PRE-RING weg2sc3 while printing BOUND.
        #
        # The rank is gone because C9 removed the per-rank tag edge, so the CARD
        # is the identity -- which is the stronger one anyway (#589).
        path = self._log([
            ("GPU-aaaa", "weights_0", 100), ("GPU-aaaa", "weights_1", 300),
            ("GPU-bbbb", "weights_0", 50), ("GPU-bbbb", "weights_1", 70),
            # second pass on aaaa: repeating a tag opens the next pass
            ("GPU-aaaa", "weights_0", 110), ("GPU-aaaa", "weights_1", 320),
        ])
        gl = ring_table.parse_group_log(path)
        self.assertEqual(gl.lines_read, 6)
        self.assertTrue(gl.covers_all_backed_up_tags,
                        "population=all-backed-up-tags must still be read")
        by_card = {gl.uuid_by_rank[r]: r for r in gl.image}
        self.assertEqual(sorted(by_card), ["GPU-aaaa", "GPU-bbbb"],
                         "each card must get its own row, not one shared bucket")
        self.assertEqual(gl.image[by_card["GPU-aaaa"]], 430, "the PEAK pass")
        self.assertEqual(gl.max_tag[by_card["GPU-aaaa"]], 320,
                         "max_tag must be non-zero -- zero is what made every "
                         "ring-era boot fail CardRing.complete")
        self.assertEqual(gl.max_tag[by_card["GPU-bbbb"]], 70)

    def test_tag_lines_that_do_NOT_parse_REFUSE_by_name_instead_of_reading_zero(self):
        # The half that makes the regex fix a fix rather than a patch, and the
        # instrument-text law applied to a PARSER: a reader whose text claims to
        # read an instrument must fail loudly when the emitter has moved, never
        # degrade to the numbers it can still compute.  Silence is what let 402
        # lines of weg2rg5's own instrument go unread while the launcher printed
        # a well-formed table solved from a boot five generations back.
        path = os.path.join(self.dir, "future.log")
        with open(path, "w") as f:
            f.write("[2026-09-09 00:00:00 PP0] WEG2-FLIP-TAG group=P rank=leader "
                    "card=GPU-aaaa dir=d2h tag=weights_0 bytes=100 MiB\n")
        with self.assertRaises(ring_table.Weg2FlipTagUnparsable) as cm:
            ring_table.parse_group_log(path)
        self.assertIn("W37 Weg2FlipTagUnparsable", str(cm.exception))
        self.assertIn("rank=leader", str(cm.exception),
                      "the refusal must quote the line it could not read")
        # It inherits the launcher's named-refusal handler, so it can never
        # reach the operator as a bare traceback.
        self.assertIsInstance(cm.exception, ring_table.Weg2RingRefused)
        # ZERO OUT OF ZERO IS NOT A REFUSAL: a log with no tag lines at all is a
        # pre-ring boot, and those are still legal sources.
        quiet = os.path.join(self.dir, "quiet.log")
        with open(quiet, "w") as f:
            f.write("[2026-09-07 21:10:23 PP0] WEG2-CHUNK-BYTES sleep "
                    "tags=['weights_0'] host_image_delta=100 MiB "
                    "(RssShmem 0 -> 0 MiB, /proc/self/status)\n")
        self.assertEqual(ring_table.parse_group_log(quiet).lines_read, 1)

    @unittest.skipUnless(
        os.path.isfile(os.path.join(EVIDENCE, f"{RG5_BOOT}.P.log")),
        "boot weg2rg5's logs are not on this box")
    def test_the_REAL_rg5_logs_now_carry_a_non_zero_max_tag(self):
        # The postmortem's own determination, as a test: at the parent this read
        # max_tag={0,0,0} and covers_all=False for both groups.
        for group, cards in (("P", 3), ("D", 3)):
            gl = ring_table.parse_group_log(
                os.path.join(EVIDENCE, f"{RG5_BOOT}.{group}.log"))
            tags = [v for r, v in gl.max_tag.items() if r >= ring_table._CARD_RANK_BASE]
            self.assertEqual(len(tags), cards, f"{group}: one row per card")
            self.assertTrue(all(v > 0 for v in tags), f"{group}: {gl.max_tag}")
            self.assertTrue(gl.covers_all_backed_up_tags, group)


class SkippedBootsAreNamedOnSuccessTest(RingTableSolverTest):
    """FIX 3 round 3, boot weg2rg5's finding 3: the solver's silent skips."""

    def test_a_newer_boot_that_was_skipped_is_NAMED_when_an_older_one_solves(self):
        # solve() accumulated rejection reasons and returned them ONLY when no
        # boot worked.  That is backwards: when nothing works the operator gets a
        # refusal to read, and when something works they get a table whose choice
        # they cannot check.  weg2rg5 solved from a boot five generations back
        # and its own log could not say why.
        import time

        newer = "boot_weg2_t3_0000000000_0907_235959"
        self._write(
            p_passes=[{0: [("weights_0", 100)]}], d_passes=[{0: [("weights_0", 90)]}],
            p_kv={0: 2.0}, d_kv={0: 2.0},
            corridor={"P": [{CARDS[0].nvml_index: 900}],
                      "D": [{CARDS[0].nvml_index: 900}]})
        # The newer boot has group logs but NO corridor samples in its front log
        # -- weg2rg5's own real reason for being skipped.
        for suffix in ("P", "D"):
            with open(os.path.join(self.dir, f"{newer}.{suffix}.log"), "w") as f:
                f.write(open(os.path.join(self.dir, f"{self.stem}.{suffix}.log")).read())
        with open(os.path.join(self.dir, f"{newer}.front.log"), "w") as f:
            f.write(_front_log({}, identity=True))
        now = time.time()
        os.utime(os.path.join(self.dir, f"{newer}.front.log"), (now, now))
        os.utime(os.path.join(self.dir, f"{self.stem}.front.log"), (now - 600, now - 600))

        table, reason = ring_table.solve([CARDS[0]], self.dir, None)
        self.assertIsNotNone(table, reason)
        self.assertEqual(table.boot, self.stem)
        skipped = [ln for ln in reason.split("\n") if "SKIPPED" in ln]
        self.assertEqual(len(skipped), 1, reason)
        self.assertIn(newer, skipped[0])
        self.assertIn("WEG2-CORRIDOR", skipped[0],
                      "the line must carry the REASON, not just the name")

    def test_the_launcher_prints_one_line_per_skipped_boot(self):
        # ... and the reason string is rendered, not merely returned: the whole
        # point is a log an operator can check the table choice against.
        from sglang.srt.weg2 import launcher

        lines = []
        table = _zr2_table()
        real = ring_table.solve
        ring_table.solve = lambda *a, **k: (
            table, "solved from b\nWEG2-HOST-RING SKIPPED (newer than the chosen "
                   "table) boot_x: front carries no WEG2-CORRIDOR samples")
        try:
            launcher.prepare_host_ring([], lines.append, "t3", "auto",
                                       "/nonexistent", "", True,
                                       leg_form="interleave")
        finally:
            ring_table.solve = real
        self.assertTrue(
            any("SKIPPED" in ln and "boot_x" in ln for ln in lines),
            "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
