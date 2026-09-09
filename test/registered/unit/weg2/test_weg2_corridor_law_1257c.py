# SPDX-License-Identifier: Apache-2.0
"""#1257c -- the corridor floor is DERIVED, and an unmeasured one may not cut.

USER DECISION 2026-09-09, verbatim, which is the law these tests pin::

    "die 1024er grenze von mir existiert ja nur weil du den wahren vram
     verbrauch nicht bepreisen konntest UND weil ich manchmal noch vram fuer
     andere prozesse brauche. wenn du jetzt korrekt bepreisen kannst, dann kann
     die default 1024er grenze auch weg (das feature muss aber erhalten
     bleiben, eben weil ich noch andere prozesse manchmal nebenher habe die
     vram brauchen)"

and, on whether the cut becomes a switch::

    "den cut als schalter? solange es nicht wirklich oom geht brauchen wir ja
     nichts zu aendern"

Five consequences follow, and each has a test below:

1. the per-card user-reserve knob STAYS, its default moves 1024 -> 0;
2. ``corridor_floor(card) = measured transient peak of the group awake on that
   card + that card's user reserve``;
3. unmeasured => ``floor=1024 source=UNMEASURED-FALLBACK``, a VERDICT ONLY --
   it never actuates a budget cut;
4. a cut is installed only where a measured floor or an explicit reserve
   requires it, and it prints its provenance;
5. the upper band edge is a FINDING (``unmobilised_free_mib=``), never a FAIL.

THE REAL CASE ON THIS RIG IS MIXED, and it is tested as such: group P's
transient IS measured (S3 ingest 2026-09-09,
``/root/.cache/sglang/phase_footprint-a191a0712717-055c2e4b0867.json``, 1055 /
1097 / 858 MiB per card) and group D's is NOT (no boot has yet written
``phase_footprint_D_rank*.json``). A test that only exercised "all measured"
or "none measured" would miss exactly the boot this tree is about to take.

HERMETIC: no NVML, no torch, no GPU, no boot. Every footprint is written into
a tmp cache dir and read back through the shipped store.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from dataclasses import dataclass
from unittest import mock

from sglang.srt.managers import corridor_guard as cg
from sglang.srt.mem_ledger import activation as act
from sglang.srt.weg2 import corridor_budget as cb

HW = "a191a0712717"
#: The three cards of this rig, and the P peaks the S3 ingest measured for
#: them. NOT typed by hand into an expectation: they are written into a tmp
#: store below and read back, so the test exercises the reader, not a literal.
C5090 = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
C3080A = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
C3080B = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"
P_PEAKS = {C5090: 1055, C3080A: 1097, C3080B: 858}


@dataclass
class _Card:
    """The launcher's ``Card`` surface the corridor pass actually reads."""

    uuid: str
    name: str
    nvml_index: int
    total_mib: int
    reserved_mib: int = 0


def _write_footprints(cache_dir: str, digest: str, peaks) -> str:
    """Write a phase-footprint cache file through the SHIPPED writer."""
    fps = {
        uuid: act.PhaseFootprint(
            activation_mib=int(mib),
            capture_mib=0,
            provenance=act.FootprintProvenance.MEASURED_PEAK,
            source="test: scripts/vram_ledger/probe_activation.py ingest",
            card_uuid=uuid,
            profile_digest=digest,
        )
        for uuid, mib in peaks.items()
    }
    path = act.footprint_cache_path(HW, digest, cache_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "version": act.PHASE_FOOTPRINT_VERSION,
        "hw_fingerprint": HW,
        "profile_digest": digest,
        "profile": {},
        "cards": {k: v.to_json() for k, v in fps.items()},
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def _publish(cache_dir: str, groups) -> str:
    path = os.path.join(cache_dir, cg.FLOOR_DIGEST_FILE_DEFAULT)
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"hw_fingerprint": HW, "groups": dict(groups)}, f)
    return path


class CorridorFloorDerivation(unittest.TestCase):
    """Consequence 2 and 3: what the floor IS, and what it says it is."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        # LAW_ENV must not leak in from the ambient environment: it would
        # stamp ENV-OVERRIDE on every floor and make every assertion below
        # pass for the wrong reason.
        self._env = mock.patch.dict(
            os.environ, {cg.FLOOR_DIGEST_FILE_ENV: os.path.join(self.cache, "d.json")}
        )
        self._env.start()
        self.addCleanup(self._env.stop)
        os.environ.pop(cg.LAW_ENV, None)

    def _floor(self, uuid, group, reserve=0, digest="pdigest"):
        return cg.corridor_floor_mib(
            uuid,
            group=group,
            hw_fingerprint=HW,
            profile_digest=digest,
            user_reserve_mib=reserve,
            cache_dir=self.cache,
        )

    def test_p_measured_floor_is_the_peak_plus_the_reserve(self):
        _write_footprints(self.cache, "pdigest", P_PEAKS)
        f = self._floor(C5090, "P")
        self.assertEqual(f.source, "MEASURED-P")
        self.assertEqual(f.transient_mib, 1055)
        self.assertEqual(f.reserve_mib, 0)
        self.assertEqual(f.mib, 1055)
        # A MEASURED peak carries NO -20 % tolerance: it is a physical
        # requirement, not a stated target with slack.
        self.assertEqual(f.verdict_floor_mib, 1055)
        self.assertTrue(f.actuates)
        self.assertEqual(f.reason, "measured-peak")

    def test_the_floor_is_per_card_and_not_rig_uniform(self):
        _write_footprints(self.cache, "pdigest", P_PEAKS)
        got = {u: self._floor(u, "P").mib for u in P_PEAKS}
        self.assertEqual(got, {C5090: 1055, C3080A: 1097, C3080B: 858})
        # And two of the three are TIGHTER than the 1024 they replace, which is
        # the finding that makes "dropping the reserve is a relaxation" false.
        self.assertGreater(got[C5090], cg.CORRIDOR_LAW_MIB)
        self.assertLess(got[C3080B], cg.CORRIDOR_LAW_MIB)

    def test_unmeasured_is_the_named_fallback_and_keeps_the_band(self):
        f = self._floor(C5090, "D")
        self.assertEqual(f.source, cg.FLOOR_SOURCE_FALLBACK)
        self.assertEqual(f.mib, 1024)
        # Byte-identical to the shipped tree's band floor at reserve 0.
        self.assertEqual(f.verdict_floor_mib, cg.corridor_band_floor_mib())
        self.assertEqual(f.verdict_floor_mib, 819)
        self.assertFalse(f.actuates)

    def test_the_reserve_raises_the_floor_and_makes_it_actuate(self):
        f = self._floor(C5090, "D", reserve=512)
        self.assertEqual(f.source, cg.FLOOR_SOURCE_FALLBACK)
        self.assertEqual(f.mib, 1024 + 512)
        self.assertTrue(f.actuates, "an explicit reserve IS a priced floor")
        self.assertEqual(f.reason, "user-reserve")

    def test_the_real_rig_case_p_measured_d_unmeasured(self):
        """The mixed case, which is what the next boot on this rig looks like."""
        _write_footprints(self.cache, "pdigest", P_PEAKS)
        _publish(self.cache, {"P": "pdigest"})
        p = cg.corridor_floor_mib(
            C5090, group="P", user_reserve_mib=0, cache_dir=self.cache
        )
        d = cg.corridor_floor_mib(
            C5090, group="D", user_reserve_mib=0, cache_dir=self.cache
        )
        self.assertEqual((p.source, p.mib), ("MEASURED-P", 1055))
        self.assertEqual((d.source, d.mib), (cg.FLOOR_SOURCE_FALLBACK, 1024))
        self.assertTrue(p.actuates)
        self.assertFalse(d.actuates)

    def test_d_measured_is_used_and_not_silently_ignored(self):
        """MUTANT DIRECTION: a reader that only ever looks up group P.

        The moment a boot writes ``phase_footprint_D_rank*.json`` and it is
        ingested, D's floor must MOVE. A derivation that hard-codes the P
        digest, or that keys on a filename instead of the (fingerprint,
        profile digest, card) triple, passes every other test in this file and
        fails this one.
        """
        _write_footprints(self.cache, "pdigest", P_PEAKS)
        _write_footprints(self.cache, "ddigest", {C5090: 622})
        _publish(self.cache, {"P": "pdigest", "D": "ddigest"})
        d = cg.corridor_floor_mib(
            C5090, group="D", user_reserve_mib=0, cache_dir=self.cache
        )
        self.assertEqual(d.source, "MEASURED-D")
        self.assertEqual(d.mib, 622)
        self.assertTrue(d.actuates)
        self.assertEqual(d.verdict_floor_mib, 622)

    def test_the_provenance_line_names_the_file_and_the_number(self):
        path = _write_footprints(self.cache, "pdigest", P_PEAKS)
        f = self._floor(C3080A, "P")
        self.assertIn(path, f.provenance)
        self.assertIn("1097", f.provenance)
        line = f.line
        for token in (
            "CORRIDOR-FLOOR",
            f"card={C3080A}",
            "group=P",
            "floor=1097",
            "source=MEASURED-P",
            "reserve=0",
            "actuates=yes",
        ):
            self.assertIn(token, line, line)

    def test_the_fallback_line_says_1024_and_says_it_is_a_fallback(self):
        line = self._floor(C5090, "D").line
        self.assertIn("floor=1024", line)
        self.assertIn(f"source={cg.FLOOR_SOURCE_FALLBACK}", line)
        self.assertIn("actuates=no", line)

    def test_one_reader_group_wide_no_per_rank_entry_point(self):
        """``raenge-nie-uneins``: co-located ranks cannot disagree.

        The transient is stored per CARD and the reserve is already collapsed
        per card, so there is no argument by which two ranks on one card could
        derive different floors. Pinned as a shape assertion: the batch
        entry point takes CARD uuids and nothing rank-shaped.
        """
        _write_footprints(self.cache, "pdigest", P_PEAKS)
        _publish(self.cache, {"P": "pdigest"})
        got = cg.corridor_floors_for_cards(
            [C5090, C5090, C3080A], group="P", cache_dir=self.cache
        )
        self.assertEqual(sorted(got), sorted({C5090, C3080A}))
        self.assertEqual(got[C5090].mib, 1055)


class TheActuationGate(unittest.TestCase):
    """Consequence 3 and 4: a fallback floor may never spend KV pool."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        os.environ.pop(cg.LAW_ENV, None)
        self.cards = [
            _Card(C3080A, "RTX 3080", 0, 20480, 425),
            _Card(C5090, "RTX 5090", 1, 32607, 518),
            _Card(C3080B, "RTX 3080", 2, 20480, 425),
        ]
        self.budgets = [17000, 29352, 17000]
        self.dormant = {C3080A: 900, C5090: 1334, C3080B: 900}

    def _sample(self):
        """A sample whose 5090 predicts 474 MiB free -- the sb5f measurement."""
        return cb.CorridorSample(
            provenance="test (weg2sb5f shape)",
            cell_size=32768,
            token_vector=(17, 7, 8),
            budget_return_ratio=None,
            budget_return_provenance="",
            cards=(
                cb.SampleCard(C3080A, "RTX 3080", 0, 20480, 17000, 900, 425, 1101, 1095, 300000),
                cb.SampleCard(C5090, "RTX 5090", 1, 32607, 29352, 1334, 518, 578, 474, 500000),
                cb.SampleCard(C3080B, "RTX 3080", 2, 20480, 17000, 900, 425, 1111, 1103, 300000),
            ),
        )

    def _floors(self, **kw):
        return {
            c.uuid: cg.corridor_floor_mib(
                c.uuid, group="D", hw_fingerprint=HW, cache_dir=self.cache, **kw
            )
            for c in self.cards
        }

    def test_a_fallback_floor_never_cuts_a_budget(self):
        """THE DANGER DIRECTION, pinned.

        The 5090 predicts 474 MiB free, 345 below the 819 MiB band floor the
        fallback grades against. The shipped pass would have cut. Under the
        user decision it must NOT: nothing measured that floor, so it is a
        verdict.
        """
        floors = self._floors(profile_digest="absent")
        self.assertTrue(all(not f.actuates for f in floors.values()))
        solve = cb.solve_corridor_budgets(
            self.cards, self.budgets, self.dormant, self._sample(), floors=floors
        )
        self.assertEqual(list(solve.budgets), self.budgets)
        self.assertFalse(solve.changed)
        text = "\n".join(solve.lines)
        self.assertIn(cb.UNMEASURED_FLOOR_NAME, text)
        self.assertIn("REFUSED-UNMEASURED-FLOOR", text)
        self.assertNotIn("INSTALLED", text)

    def test_an_explicit_reserve_makes_the_same_card_actuate(self):
        """The knob the user kept is exactly what turns the cut back on."""
        floors = self._floors(profile_digest="absent", user_reserve_mib=256)
        self.assertTrue(all(f.actuates for f in floors.values()))
        solve = cb.solve_corridor_budgets(
            self.cards, self.budgets, self.dormant, self._sample(), floors=floors
        )
        text = "\n".join(solve.lines)
        self.assertNotIn("REFUSED-UNMEASURED-FLOOR", text)

    def test_an_installed_cut_prints_its_provenance(self):
        """Consequence 4: never a cut without the line that prices it."""
        _write_footprints(self.cache, "ddigest", {C5090: 900, C3080A: 300, C3080B: 300})
        floors = self._floors(profile_digest="ddigest")
        self.assertEqual(floors[C5090].source, "MEASURED-D")
        solve = cb.solve_corridor_budgets(
            self.cards, self.budgets, self.dormant, self._sample(), floors=floors
        )
        text = "\n".join(solve.lines)
        installed = [l for l in solve.lines if " INSTALLED " in l]
        if installed:
            line = installed[0]
            for token in (
                "cut_mib=",
                "reason=measured-peak",
                "pool_before=",
                "pool_after=",
                "vector=",
                "binder=",
                "measured_peak=",
                "provenance=",
                "source=MEASURED-D",
            ):
                self.assertIn(token, line, line)
        else:
            # No cut happened -- then the refusal must name itself, and it must
            # NOT be the unmeasured one, because this floor IS measured.
            self.assertNotIn(cb.UNMEASURED_FLOOR_NAME, text)
            self.assertRegex(text, r"verdict=(SATISFIED|REFUSED-WOULD-BIND|REFUSED-)")

    def test_the_per_card_line_carries_floor_source_and_reserve(self):
        floors = self._floors(profile_digest="absent", user_reserve_mib=128)
        solve = cb.solve_corridor_budgets(
            self.cards, self.budgets, self.dormant, self._sample(), floors=floors
        )
        card_lines = [l for l in solve.lines if f"card={C5090} " in l]
        self.assertTrue(card_lines)
        self.assertIn("floor=1152", card_lines[0])
        self.assertIn(f"source={cg.FLOOR_SOURCE_FALLBACK}", card_lines[0])
        self.assertIn("reserve=128", card_lines[0])


class TheReserveDefault(unittest.TestCase):
    """Consequence 1: the knob stays, the default is 0, and 0 is passable."""

    def test_the_declared_default_is_zero(self):
        from sglang.srt.mem_ledger.terms import DEFAULT_USER_RESERVE_MIB

        self.assertEqual(DEFAULT_USER_RESERVE_MIB, 0)

    def test_the_field_default_is_a_sentinel_not_a_number(self):
        """MUTANT DIRECTION: 'default silently 1024', and its twin.

        A VALUE default cannot say whether it was passed. At default 0 an
        explicit ``--rank-user-reserve-mib 0`` would read as unset and would
        silently disarm the ledger refusal -- the flag doing nothing without a
        word, which is the exact failure that refusal exists to prevent.
        """
        from sglang.srt.mem_ledger.terms import USER_RESERVE_UNSET
        from sglang.srt.server_args import ServerArgs

        field = ServerArgs.__dataclass_fields__["rank_user_reserve_mib"]
        self.assertEqual(field.default, USER_RESERVE_UNSET)
        self.assertNotEqual(str(field.default), "1024")

        class _SA:
            rank_user_reserve_mib = USER_RESERVE_UNSET
            _user_reserve_was_passed = ServerArgs._user_reserve_was_passed
            user_reserve_mib_scalar = ServerArgs.user_reserve_mib_scalar
            user_reserve_mib_per_gpu = ServerArgs.user_reserve_mib_per_gpu

        sa = _SA()
        self.assertFalse(sa._user_reserve_was_passed())
        self.assertEqual(sa.user_reserve_mib_scalar(), 0)
        self.assertEqual(sa.user_reserve_mib_per_gpu([0, 1]), {0: 0, 1: 0})

        sa.rank_user_reserve_mib = 0
        self.assertTrue(
            sa._user_reserve_was_passed(),
            "an EXPLICIT 0 is a real answer and must read as passed",
        )
        sa.rank_user_reserve_mib = "1024,512"
        self.assertTrue(sa._user_reserve_was_passed())
        self.assertEqual(sa.user_reserve_mib_per_gpu([0, 1]), {0: 1024, 1: 512})
        self.assertEqual(sa.user_reserve_mib_scalar(), 1024)

    def test_the_help_text_says_what_the_knob_is_for(self):
        """The knob is KEPT and its purpose is stated, per the user decision."""
        import inspect

        from sglang.srt import server_args as sa_mod

        src = inspect.getsource(sa_mod)
        i = src.index("rank_user_reserve_mib: A[")
        help_text = src[i : i + 2600]
        self.assertIn("Default 0", help_text)
        self.assertIn("OUTSIDE this engine", help_text)
        self.assertNotIn("Default 1024", help_text)

    def test_no_second_1024_default_in_the_model_runner(self):
        """MUTANT DIRECTION: a private copy that keeps charging the gibibyte.

        ``_gapped_corridor_holdback`` held ``reserve_mib = 1024 if configured
        is None else ...`` -- a second hard default for a number
        ``mem_ledger.terms`` declares. Left in place it would have kept
        charging 1 GiB on the gapped path after the operator moved the default
        to 0, invisibly, on the one path with a demonstrated OOM.
        """
        import inspect

        from sglang.srt.model_executor import model_runner_kv_cache_mixin as m

        src = inspect.getsource(m.KvCacheMixin._gapped_corridor_holdback)
        self.assertNotIn("reserve_mib = 1024", src)
        self.assertIn("DEFAULT_USER_RESERVE_MIB", src)
        self.assertIn("USER_RESERVE_UNSET", src)


class TheUpperEdgeIsAFinding(unittest.TestCase):
    """Consequence 5: above the ceiling is never a failure on its own."""

    def test_corridor_verdict_still_says_above(self):
        from sglang.srt.weg2 import front

        f = cg.CorridorFloor(C5090, "P", 1055, 0, "MEASURED-P")
        self.assertEqual(front.corridor_verdict(500, f), "BELOW")
        self.assertEqual(front.corridor_verdict(1055, f), "IN")
        self.assertEqual(front.corridor_verdict(1266, f), "IN")
        self.assertEqual(front.corridor_verdict(9000, f), "ABOVE")

    def test_arm_report_puts_above_in_findings_not_problems(self):
        """MUTANT DIRECTION: an acceptance that fails on unmobilised free.

        The predecessor appended the ABOVE case to ``problems``, so a boot with
        gibibytes idle FAILED its corridor arm. Under the user decision that is
        a capacity finding for the planner, not a breach of the law.
        """
        from sglang.srt.weg2 import corridor_arm

        rep = corridor_arm.ArmReport(path="x")
        self.assertEqual(rep.problems, [])
        self.assertEqual(rep.findings, [])
        rep.findings.append("phase=D nvml1 unmobilised_free_mib=4000")
        self.assertTrue(rep.ok, "a FINDING must not fail the arm")
        rep.problems.append("phase=D nvml1 minimum 100 MiB is BELOW its floor")
        self.assertFalse(rep.ok, "a BREACH must still fail the arm")


class TheSharedSurfaces(unittest.TestCase):
    """Deliverables 5 and 6: everyone reads the SAME derived floor."""

    def test_the_front_line_carries_the_floor_and_the_arm_reads_it_back(self):
        """Deliverable 6, answered: VIA THE FRONT LINE.

        The front derives the floor once and prints ``floor=``/``source=``
        beside every ``nvmlN:free=``; ``corridor_arm`` parses that back and
        grades against it. One producer, one direction, so the sampler and the
        verdict cannot disagree about the number.
        """
        from sglang.srt.weg2 import ring_table

        line = (
            "WEG2-CORRIDOR phase=D(awake) epoch=3 "
            "instrument=nvml_v2_free,allocatable band=819-1266MiB "
            "nvml0:free=1095MiB reserved=425MiB floor=1097MiB source=MEASURED-P "
            "reserve=0MiB verdict=BELOW "
            "nvml1:free=474MiB reserved=518MiB floor=1055MiB source=MEASURED-P "
            "reserve=0MiB verdict=BELOW\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
            fh.write(line)
            path = fh.name
        self.addCleanup(os.unlink, path)
        got = ring_table.parse_front_corridor_floors(path)
        self.assertEqual(got, {0: (1097, "MEASURED-P"), 1: (1055, "MEASURED-P")})

    def test_a_pre_1257c_log_yields_no_floor_and_says_nothing_false(self):
        from sglang.srt.weg2 import ring_table

        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
            fh.write(
                "WEG2-CORRIDOR phase=D(awake) epoch=1 "
                "instrument=nvml_v2_free,allocatable band=819-1229MiB "
                "nvml0:free=1095MiB reserved=425MiB verdict=IN\n"
            )
            path = fh.name
        self.addCleanup(os.unlink, path)
        self.assertEqual(ring_table.parse_front_corridor_floors(path), {})

    def test_corridor_budget_has_no_private_law_constant(self):
        """UPSTREAM-MINIMAL: the fourth private copy must not come back."""
        import inspect

        src = inspect.getsource(cb)
        self.assertNotIn("\nCORRIDOR_LAW_MIB = 1024", src)
        self.assertIs(cb.CORRIDOR_LAW_MIB, cg.CORRIDOR_LAW_MIB)

    def test_the_launcher_corridor_constant_split_keeps_the_measured_404(self):
        from sglang.srt.weg2 import launcher

        self.assertEqual(launcher.D_AWAKE_OVERSHOOT_MIB, 404)
        self.assertEqual(
            launcher.CORRIDOR_MIB,
            cg.CORRIDOR_LAW_MIB + launcher.D_AWAKE_OVERSHOOT_MIB,
            "the fallback must still subtract exactly the shipped 1428",
        )

    def test_the_p_holdback_default_follows_the_reserve(self):
        """MUTANT DIRECTION: a pool model that keeps charging 1 GiB per rank.

        ``gapped corridor holdback`` IS the user reserve at the runtime end. At
        reserve 0 the runtime charges 0.000, so a launcher still pinning
        1024.0 over-charges group P by a gibibyte PER RANK.
        """
        from sglang.srt.weg2 import launcher

        self.assertEqual(launcher.p_corridor_holdback_default_mib(0), 0.0)
        self.assertEqual(launcher.p_corridor_holdback_default_mib(1024), 1024.0)
        self.assertEqual(
            launcher.p_corridor_holdback_default_mib(),
            0.0,
            "with no reserve given it must follow the DECLARED default, 0",
        )

    def test_the_reserve_env_name_is_declared_once(self):
        from sglang.srt.weg2 import front

        self.assertIs(front.RESERVE_ENV, cg.USER_RESERVE_ENV)

    def test_the_launcher_reserve_parser_refuses_rather_than_guesses(self):
        from sglang.srt.weg2 import launcher

        cards = [
            _Card(C3080A, "a", 0, 20480),
            _Card(C5090, "b", 1, 32607),
            _Card(C3080B, "c", 2, 20480),
        ]
        self.assertEqual(
            launcher.parse_user_reserve("0", cards),
            {C3080A: 0, C5090: 0, C3080B: 0},
        )
        self.assertEqual(
            launcher.parse_user_reserve("0,1024,0", cards),
            {C3080A: 0, C5090: 1024, C3080B: 0},
        )
        with self.assertRaises(SystemExit):
            launcher.parse_user_reserve("0,1024", cards)
        with self.assertRaises(SystemExit):
            launcher.parse_user_reserve("-1", cards)
        with self.assertRaises(SystemExit):
            launcher.parse_user_reserve("nope", cards)


class TheEnvOverrideStillWins(unittest.TestCase):
    def test_env_override_stamps_its_own_source_and_actuates(self):
        with mock.patch.dict(os.environ, {cg.LAW_ENV: "1536"}):
            f = cg.corridor_floor_mib(C5090, group="P", user_reserve_mib=64)
        self.assertEqual(f.source, cg.FLOOR_SOURCE_ENV)
        self.assertEqual(f.transient_mib, 1536)
        self.assertEqual(f.mib, 1600)
        self.assertTrue(f.actuates, "a hand-set law is a hand-priced floor")
        self.assertIn(cg.LAW_ENV, f.provenance)


class TheDocumentedNumbersAreGrepable(unittest.TestCase):
    """Every new line an operator will grep for, pinned as a shape."""

    def test_the_floor_line_shape(self):
        f = cg.CorridorFloor(C5090, "P", 1055, 0, "MEASURED-P", "some/path#x")
        self.assertRegex(
            f.line,
            r"^CORRIDOR-FLOOR card=\S+ group=\S+ floor=\d+ verdict_floor=\d+ "
            r"ceiling=\d+ transient=\d+ source=\S+ reserve=\d+ actuates=(yes|no) "
            r"reason=\S+ provenance=",
        )

    def test_every_new_line_is_english_and_has_no_emoji(self):
        f = cg.CorridorFloor(C5090, "P", 1055, 0, "MEASURED-P")
        self.assertTrue(all(ord(ch) < 128 for ch in f.line), f.line)
        self.assertIsNone(re.search(r"[^\x00-\x7f]", f.line))


if __name__ == "__main__":
    unittest.main()
