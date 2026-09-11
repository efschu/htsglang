# SPDX-License-Identifier: Apache-2.0
"""B4c item 1: the S6 exchange census has a PRODUCER (#1273).

``--weg2-weight-source exchange`` refused at launch by name -- W71
Weg2XchgResidencyUnarmable, "needs a per-card census (--weg2-xchg-census
<path>) and none was given" -- because nothing in the tree wrote that file.
``xchg_residency.load_census`` refuses to invent one and says so; this module
builds it, and every number in it comes from an instrument a named boot wrote.

WHAT IS ASSERTED HERE, and each is a direction a mutant goes:

* the SELECTION is not this module's: it is delegated to ``ring_table.solve``,
  the same call the launcher makes, and the stem that comes back is the stem
  used.  A producer that picked the newest boot itself would price a census of
  a boot the launcher never reads.
* an xchg-shadow source is REFUSED, because #1305 item 4's ruling excludes it
  from the launcher's own selection -- a census built on one is provably not
  the launcher's table.
* the per-card bytes are keyed by the log's OWN ``card=`` UUID, never by rank
  position, and a rank that names no card refuses.
* a card's tag enumeration must be PROVEN COMPLETE before an absent (card,
  tag) pair may be written as a zero: the card's tag bytes must sum to that
  rank's own image.  ``check_partition`` says "an absent tag is not a
  zero-byte tag"; the sum identity is what turns one into the other, and
  without it the zero-fill would be exactly the under-pricing that gate is
  for.
* BOTH groups, always: ``load_census`` refuses a one-group census because it
  cannot state a co-residency peak.
* a BOUND is never printed as a reading: a weights-only census says LOWER
  BOUND, and a dormant residue taken from the boot-named constant says which
  constant and which boot rather than reading like a measurement of this one.
* no source for a card's dormant residue REFUSES; there is no default.

Hermetic: temp files only, no GPU, no evidence tree, no NVML.
"""

import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import ring_table, xchg_census, xchg_residency
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BIG = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
SM1 = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
SM2 = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"
FAMILY = ("weights_0", "weights_1", "weights")


class FakeCard:
    """The four attributes ``xchg_residency.solve`` reads off a live card."""

    def __init__(self, nvml_index, uuid, name, total_mib):
        self.nvml_index = nvml_index
        self.uuid = uuid
        self.name = name
        self.total_mib = total_mib
        self.reserved_mib = 0


CARDS = (
    FakeCard(1, BIG, "NVIDIA GeForce RTX 5090", 32607),
    FakeCard(0, SM1, "NVIDIA GeForce RTX 3080", 20480),
    FakeCard(2, SM2, "NVIDIA GeForce RTX 3080", 20480),
)

ORDINAL_MAP = (
    "[2026-09-09T15:11:45Z] WEG2-LAUNCH NVML -> CUDA ordinal map: "
    f"ordinal 0 = nvml 1 NVIDIA GeForce RTX 5090 {BIG} total 32607 MiB, "
    f"ordinal 1 = nvml 0 NVIDIA GeForce RTX 3080 {SM1} total 20480 MiB, "
    f"ordinal 2 = nvml 2 NVIDIA GeForce RTX 3080 {SM2} total 20480 MiB\n"
)


def _dc(group, uuid, mib, reserve=1986):
    return (
        f"[2026-09-09T15:12:00Z] INFO weg2.front: WEG2-DC group={group} "
        f"uuid={uuid} measured={mib} MiB reserve={reserve}\n"
    )


def _tag(group, rank, card, tag, mib, population=ring_table.TAG_POPULATION_ALL):
    return (
        f"[2026-09-09T15:11:45Z] INFO weg2.front: WEG2-FLIP-TAG group={group} "
        f"rank={rank} card={card} dir=d2h tag={tag} bytes={mib} MiB "
        f"population={population} (source: tms_tag_bytes) ms=10 GB/s=1.0\n"
    )


#: P is the PP group, so a chunk tag names only the stage whose layers it
#: covers: ``weights_1`` has NO bytes on the big card, which is the genuine
#: zero the completeness proof licenses.  D is the TP group and holds every
#: tag on every card.  Every per-card sum is a distinct number.
P_TAGS = {BIG: {"weights_0": 300, "weights": 50}, SM1: {"weights_1": 200, "weights": 40},
          SM2: {"weights_1": 100, "weights": 30}}
D_TAGS = {BIG: {"weights_0": 90, "weights_1": 91, "weights": 20},
          SM1: {"weights_0": 60, "weights_1": 61, "weights": 11},
          SM2: {"weights_0": 60, "weights_1": 61, "weights": 12}}

#: EVERY CARD IS WRITTEN TWICE, i.e. as TWO FLIP LEGS, and that is what makes
#: the completeness identity able to fail at all (seat 4's fixture trap: two
#: quantities that coincide cannot see a mutant that swaps them).  ``image`` is
#: the MAX over passes and a pass ends at the second sighting of a tag, so with
#: one leg the image IS the sum of the per-tag maxima by construction and the
#: identity is tautological.  With two legs it says something: the per-tag
#: maxima all have to come from a single coherent pass.  ``skew`` moves bytes
#: BETWEEN tags in leg 2 while keeping the leg's own sum identical, so the pass
#: totals agree, the maxima do not, and only the identity notices.
SKEW = 50


def _write(d, name, lines):
    p = os.path.join(d, name)
    with open(p, "w") as fh:
        fh.writelines(lines)
    return p


def _rig(p_tags=None, d_tags=None, front_extra=(), population=ring_table.TAG_POPULATION_ALL,
         complete=True, stem="boot_weg2_weg2fix_0000000000_0909_000000"):
    """A three-log source boot, whose numbers are the two dicts above."""
    p_tags = P_TAGS if p_tags is None else p_tags
    d_tags = D_TAGS if d_tags is None else d_tags
    d = tempfile.mkdtemp()
    front = [ORDINAL_MAP]
    front += [_dc("P", BIG, 1326), _dc("P", SM1, 864), _dc("P", SM2, 926)]
    front += [_dc("D", BIG, 1668), _dc("D", SM1, 1334), _dc("D", SM2, 1334)]
    front += list(front_extra)
    _write(d, f"{stem}.front.log", front)
    for group, per_card in (("P", p_tags), ("D", d_tags)):
        lines = []
        for leg in (1, 2):
            for card, tags in sorted(per_card.items()):
                names = sorted(tags)
                for i, tag in enumerate(names):
                    mib = tags[tag]
                    if leg == 2 and not complete and len(names) > 1:
                        # +SKEW on the first tag, -SKEW on the last: the leg's
                        # own sum is unchanged, so the image is unchanged, and
                        # the per-tag maxima now exceed it.
                        mib += SKEW if i == 0 else (-SKEW if i == len(names) - 1 else 0)
                    lines.append(_tag(group, -1, card, tag, mib, population))
        _write(d, f"{stem}.{group}.log", lines)
    return d, stem


def _build(**kw):
    d, stem = _rig(**{k: v for k, v in kw.items() if k in
                      ("p_tags", "d_tags", "front_extra", "population", "complete")})
    return xchg_census.census_from_logs(
        CARDS, d, stem, family=FAMILY, n_cards=3,
        selection="fixture: pinned", tool_sha="deadbeef",
    )


class TheCensusIsBuiltFromTheRingTablesOwnBoot(CustomTestCase):
    def test_the_blob_is_what_load_census_accepts(self):
        build = _build()
        p = os.path.join(tempfile.mkdtemp(), "census.json")
        xchg_census.write_census(build, p)
        census = xchg_residency.load_census(p)
        self.assertEqual(sorted(census.cards), sorted(c.uuid for c in CARDS))
        self.assertEqual(census.waves, (FAMILY,))
        self.assertIn("deadbeef", census.provenance)

    def test_both_groups_per_card_keyed_by_the_logs_own_uuid(self):
        build = _build()
        cards = build.blob["cards"]
        self.assertEqual(sorted(cards[BIG]["tags"]), ["D", "P"])
        self.assertEqual(cards[BIG]["tags"]["P"]["weights_0"], 300)
        self.assertEqual(cards[SM1]["tags"]["P"]["weights_1"], 200)
        self.assertEqual(cards[SM2]["tags"]["P"]["weights_1"], 100)
        self.assertEqual(cards[BIG]["tags"]["D"]["weights_1"], 91)

    def test_a_genuine_per_stage_zero_is_written_as_a_zero(self):
        """P's ``weights_1`` is not on the big card, and the census says 0.

        ``check_partition`` refuses a wave tag a card has no bytes for, so the
        zero must be PRESENT -- and it is only writable because the card's tag
        bytes sum to its image.
        """
        build = _build()
        self.assertEqual(build.blob["cards"][BIG]["tags"]["P"]["weights_1"], 0)
        self.assertEqual(
            xchg_residency.check_partition(
                xchg_residency.XchgCensus(
                    cards={u: xchg_residency.CardCensus(
                        uuid=u,
                        tags=e["tags"],
                        dormant_proc_used_mib=e["dormant_proc_used_mib"],
                    ) for u, e in build.blob["cards"].items()},
                    waves=(FAMILY,),
                )
            ),
            [],
        )

    def test_an_incomplete_enumeration_refuses_instead_of_filling_zeros(self):
        """The one thing that turns an absence into a zero is the sum identity."""
        with self.assertRaises(xchg_residency.Weg2XchgResidencyUnarmable) as caught:
            _build(complete=False)
        msg = str(caught.exception)
        self.assertIn("W71", msg)
        self.assertIn("does not account for", msg)

    def test_a_tag_the_log_measured_but_the_family_omits_refuses(self):
        """Dropping measured bytes would break the identity in silence."""
        extra = {k: dict(v) for k, v in P_TAGS.items()}
        extra[BIG]["weights_9"] = 7
        with self.assertRaises(xchg_residency.Weg2XchgResidencyUnarmable) as caught:
            _build(p_tags=extra)
        self.assertIn("weights_9", str(caught.exception))


class TheDormantResidueNamesItsSource(CustomTestCase):
    def test_the_reading_wins_and_is_the_larger_of_the_two_groups(self):
        build = _build()
        cards = build.blob["cards"]
        self.assertEqual(cards[BIG]["dormant_proc_used_mib"], 1668)
        self.assertEqual(cards[SM1]["dormant_proc_used_mib"], 1334)
        for uuid, other in ((BIG, 1326), (SM1, 864), (SM2, 926)):
            src = cards[uuid]["dormant_source"]
            self.assertIn("WEG2-DC", src)
            self.assertIn(str(other), src, src)

    def test_the_peak_reading_is_taken_not_the_last_one(self):
        d, stem = _rig(front_extra=(_dc("D", BIG, 1669), _dc("D", BIG, 7)))
        build = xchg_census.census_from_logs(
            CARDS, d, stem, family=FAMILY, n_cards=3, selection="x", tool_sha="s")
        self.assertEqual(build.blob["cards"][BIG]["dormant_proc_used_mib"], 1669)

    def test_no_reading_falls_back_to_the_boot_named_constant_and_says_so(self):
        from sglang.srt.weg2 import launcher

        d, stem = _rig()
        front = os.path.join(d, f"{stem}.front.log")
        with open(front) as fh:
            kept = [ln for ln in fh if "WEG2-DC" not in ln]
        with open(front, "w") as fh:
            fh.writelines(kept)
        build = xchg_census.census_from_logs(
            CARDS, d, stem, family=FAMILY, n_cards=3, selection="x", tool_sha="s")
        big = build.blob["cards"][BIG]
        self.assertEqual(big["dormant_proc_used_mib"], launcher.DC_MEASURED_D_5090_MIB)
        self.assertIn("weg2ls1b2", big["dormant_source"])
        self.assertIn("DC_MEASURED_D_5090_MIB", big["dormant_source"])
        self.assertEqual(
            build.blob["cards"][SM1]["dormant_proc_used_mib"],
            launcher.DC_MEASURED_D_3080_MIB,
        )

    def test_the_expectations_row_is_never_a_source(self):
        from sglang.srt.weg2 import launcher

        build = _build()
        for entry in build.blob["cards"].values():
            self.assertNotIn(str(launcher.DC_EXPECT_5090_MIB), entry["dormant_source"])
            self.assertNotIn(str(launcher.DC_EXPECT_3080_MIB), entry["dormant_source"])
            self.assertNotEqual(entry["dormant_proc_used_mib"], launcher.DC_EXPECT_5090_MIB)

    def test_a_card_with_neither_source_refuses(self):
        d, stem = _rig()
        front = os.path.join(d, f"{stem}.front.log")
        with open(front) as fh:
            kept = [ln for ln in fh if "WEG2-DC" not in ln]
        with open(front, "w") as fh:
            fh.writelines(kept)
        odd = CARDS + (FakeCard(3, "GPU-cccccccc-0000-0000-0000-000000000003",
                                "NVIDIA GeForce GTX 780", 3072),)
        with self.assertRaises(xchg_residency.Weg2XchgResidencyUnarmable) as caught:
            xchg_census.census_from_logs(
                odd, d, stem, family=FAMILY, n_cards=4, selection="x", tool_sha="s")
        self.assertIn("GTX 780", str(caught.exception))


class TheBoundIsNeverPrintedAsAReading(CustomTestCase):
    def test_a_weights_only_census_says_lower_bound(self):
        build = _build(population=ring_table.TAG_POPULATION_WEIGHTS)
        self.assertIn("LOWER BOUND", build.provenance)
        self.assertIn("weights-family", build.provenance)

    def test_an_all_tags_census_does_not(self):
        self.assertNotIn("LOWER BOUND", _build().provenance)


class TheSelectionIsTheLaunchersNotThisModules(CustomTestCase):
    def _solver(self, stem, reason=""):
        class Table:
            boot = stem
            form_same = True
            source_form_key = "abc"
            image_source = "P: x | D: y"

        return lambda *a, **k: ((Table() if stem else None), reason)

    def test_the_stem_comes_back_from_the_solver(self):
        d, stem = _rig()
        build = xchg_census.build_census(
            CARDS, d, family=FAMILY, n_cards=3, tool_sha="s",
            solver=self._solver(stem), wave_map_arm="ranks")
        self.assertEqual(build.stem, stem)
        self.assertIn(stem, build.provenance)

    def test_a_solver_with_no_table_refuses_with_its_reason(self):
        d, stem = _rig()
        with self.assertRaises(xchg_residency.Weg2XchgResidencyUnarmable) as caught:
            xchg_census.build_census(
                CARDS, d, family=FAMILY, n_cards=3, tool_sha="s",
                solver=self._solver("", "no boot carries all three logs"))
        self.assertIn("no boot carries all three logs", str(caught.exception))

    def test_an_xchg_shadow_source_is_refused(self):
        marker = (
            f"[2026-09-09T15:11:49Z] WEG2-LAUNCH {ring_table.XCHG_FORM_MARKER} "
            "epoch=1 path=/dev/shm/x/xchg.bin slots=6x2x32MiB\n"
        )
        d, stem = _rig(front_extra=(marker,))
        with self.assertRaises(xchg_residency.Weg2XchgResidencyUnarmable) as caught:
            xchg_census.build_census(
                CARDS, d, family=FAMILY, n_cards=3, tool_sha="s",
                solver=self._solver(stem))
        self.assertIn(ring_table.XCHG_FORM_MARKER, str(caught.exception))

    def test_the_selection_oracle_refuses_a_disagreeing_stem(self):
        """The reference boot's own log says which stem its launcher chose."""
        d, stem = _rig()
        oracle = _write(d, "ref.front.log", [
            "[2026-09-11T02:50:08Z] WEG2-LAUNCH WEG2-HOST-RING SOURCE solved from "
            "boot_weg2_somethingelse_0000000000_0909_000000 (form DIFFERENT)\n",
        ])
        with self.assertRaises(xchg_residency.Weg2XchgResidencyUnarmable) as caught:
            xchg_census.build_census(
                CARDS, d, family=FAMILY, n_cards=3, tool_sha="s",
                solver=self._solver(stem), selection_oracle=oracle)
        self.assertIn("somethingelse", str(caught.exception))

    def test_the_selection_oracle_accepts_an_agreeing_stem(self):
        d, stem = _rig()
        oracle = _write(d, "ref.front.log", [
            "[2026-09-11T02:50:08Z] WEG2-LAUNCH WEG2-HOST-RING SOURCE solved from "
            f"{stem} (form DIFFERENT, source key e3fe683c5d40)\n",
        ])
        build = xchg_census.build_census(
            CARDS, d, family=FAMILY, n_cards=3, tool_sha="s",
            solver=self._solver(stem), selection_oracle=oracle,
            wave_map_arm="ranks")
        self.assertIn("oracle", build.provenance)


#: The reference boot's own record of the PP form: the split, the chunk
#: geometry, the NVML stage order, and the map it published.  Values are the
#: serving form's, so the arithmetic below is the one the rig runs.
ORDER_MAP_LINE = (
    "[2026-09-11T02:50:12Z] WEG2-LAUNCH WEG2-FLIP-ORDER MAP group=P (SOLVED cut "
    "39,13,12/9,4,3 -> REALIZED layer split [39, 13, 12] over 64 layers, 8 layers "
    "per chunk, nvml [1, 0, 2] in stage order): {'weights_0': [1], 'weights_1': [1], "
    "'weights_2': [1], 'weights_3': [1], 'weights_4': [0, 1], 'weights_5': [0], "
    "'weights_6': [0, 2], 'weights_7': [2]}; group=D TP -> no map\n"
)
BIG_FAMILY = tuple(f"weights_{k}" for k in range(8)) + ("weights",)


class TheWaveMapArmsAreBothNamed(CustomTestCase):
    def test_the_ranks_arm_is_the_empty_map_and_yields_one_wave(self):
        mapping, why = xchg_census.resolve_wave_map("ranks", "")
        self.assertEqual(mapping, {})
        self.assertIs(mapping, xchg_census.RANK_WAVE_MAP)
        self.assertIn("UNIFORM", why)
        self.assertIn("unarmable", why)
        build = _build()
        self.assertEqual(build.blob["waves"], [list(FAMILY)])

    def test_the_launcher_arm_rebuilds_the_pp_map_in_ordinal_space(self):
        """Card 0 is the FIRST STAGE, not nvml 0: the translation is the point."""
        d = tempfile.mkdtemp()
        front = _write(d, "ref.front.log", [ORDINAL_MAP, ORDER_MAP_LINE])
        mapping, why = xchg_census.resolve_wave_map("launcher", front)
        self.assertEqual(mapping["weights_0"], (0,))
        self.assertEqual(mapping["weights_4"], (0, 1))
        self.assertEqual(mapping["weights_6"], (1, 2))
        self.assertEqual(mapping["weights_7"], (2,))
        self.assertIn("CHECKED against the map that boot published", why)
        self.assertIn("PRECONDITION", why)

    def test_the_pp_map_yields_the_three_waves_the_spec_expects(self):
        from sglang.srt.weg2 import weight_exchange as wx

        d = tempfile.mkdtemp()
        front = _write(d, "ref.front.log", [ORDINAL_MAP, ORDER_MAP_LINE])
        mapping, _ = xchg_census.resolve_wave_map("launcher", front)
        waves = wx.derive_waves(list(BIG_FAMILY), mapping, (0, 1, 2))
        self.assertEqual(len(waves), 3, waves)
        self.assertEqual(
            sorted(t for w in waves for t in w), sorted(BIG_FAMILY))
        self.assertEqual(waves[-1][-1], "weights")

    def test_a_published_map_that_disagrees_with_the_rebuild_refuses(self):
        doctored = ORDER_MAP_LINE.replace("'weights_7': [2]", "'weights_7': [0]")
        d = tempfile.mkdtemp()
        front = _write(d, "ref.front.log", [ORDINAL_MAP, doctored])
        with self.assertRaises(xchg_residency.Weg2XchgResidencyUnarmable) as caught:
            xchg_census.resolve_wave_map("launcher", front)
        self.assertIn("does not match the map", str(caught.exception))

    def test_a_split_that_contradicts_its_own_layer_count_refuses(self):
        bad = ORDER_MAP_LINE.replace("[39, 13, 12] over 64 layers", "[39, 13, 12] over 60 layers")
        d = tempfile.mkdtemp()
        front = _write(d, "ref.front.log", [ORDINAL_MAP, bad])
        with self.assertRaises(xchg_residency.Weg2XchgResidencyUnarmable) as caught:
            xchg_census.resolve_wave_map("launcher", front)
        self.assertIn("sums to 64, not the 60", str(caught.exception))

    def test_a_missing_map_line_refuses_and_never_falls_back_to_uniform(self):
        d = tempfile.mkdtemp()
        front = _write(d, "ref.front.log", [ORDINAL_MAP])
        with self.assertRaises(xchg_residency.Weg2XchgResidencyUnarmable) as caught:
            xchg_census.resolve_wave_map("launcher", front)
        msg = str(caught.exception)
        self.assertIn("no fallback to the uniform map", msg)

    def test_an_unknown_arm_is_not_silently_one_of_the_two(self):
        with self.assertRaises(xchg_residency.Weg2XchgResidencyUnarmable):
            xchg_census.resolve_wave_map("whatever", "")

    def test_build_census_defaults_to_the_launcher_arm(self):
        """The DEFAULT is the arm that fits, and the default is asserted.

        Without this the arms could be swapped at the signature and every
        other test here would still pass, because they all name their arm.
        """
        d, stem = _rig(front_extra=(ORDER_MAP_LINE,))
        front = os.path.join(d, f"{stem}.front.log")

        class Table:
            boot = stem

        build = xchg_census.build_census(
            CARDS, d, family=FAMILY, n_cards=3, tool_sha="s",
            solver=lambda *a, **k: (Table(), ""), wave_map_from=front)
        self.assertIn("per-card chunk_tag_cards", build.provenance)
        self.assertIn("PRECONDITION", build.provenance)
        self.assertTrue(any("map=per-card" in ln for ln in build.lines), build.lines)

    def test_a_partition_that_is_not_the_family_refuses(self):
        """``build_plan``'s stale-wave-map ratchet, one layer earlier.

        PROVEN REACHABLE by substituting the producer, not asserted to exist:
        ``derive_waves`` partitions the family by construction, so with it in
        place this branch cannot fire and would be a guard with no can-fail
        proof.  A producer that drops the base tag -- the exact shape a future
        wave derivation could take -- fires it.
        """
        d, stem = _rig()
        with self.assertRaises(xchg_residency.Weg2XchgResidencyUnarmable) as caught:
            xchg_census.census_from_logs(
                CARDS, d, stem, family=FAMILY, n_cards=3, selection="x",
                tool_sha="s", waves_of=lambda fam, m, c: [[t for t in fam if t != "weights"]])
        self.assertIn("not a partition of the family", str(caught.exception))

    def test_the_default_producer_is_the_owners(self):
        from sglang.srt.weg2 import weight_exchange as wx

        d, stem = _rig()
        seen = []

        def spy(fam, m, c):
            seen.append((tuple(fam), dict(m), tuple(c)))
            return wx.derive_waves(fam, m, c)

        xchg_census.census_from_logs(
            CARDS, d, stem, family=FAMILY, n_cards=3, selection="x",
            tool_sha="s", waves_of=spy)
        self.assertEqual(seen, [(tuple(FAMILY), {}, (0, 1, 2))])

    def test_the_family_must_not_be_empty(self):
        d, stem = _rig()
        with self.assertRaises(xchg_residency.Weg2XchgResidencyUnarmable):
            xchg_census.census_from_logs(
                CARDS, d, stem, family=(), n_cards=3, selection="x", tool_sha="s")


if __name__ == "__main__":
    unittest.main()
