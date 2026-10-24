# SPDX-License-Identifier: Apache-2.0
"""#201 TRIM-LAW (Unify-RISK (c), 24.10.): the flip arm's trim threshold follows the line of the D rank's OWN card.

ROOT (base c07b574410): ``launcher._d_extend_flip_trim_env`` priced every D rank with the STATED law
``corridor_guard.corridor_law_mib()`` (1024): ``ceil(1024 + 256 x 0.3091) = 1104`` -- the int4ar / duo metal value
(boot ..._c07b574410_1024_125438 front: ``EXTEND-TRIM-FLIP FLLIPER_PDFLIP_EXTEND_TRIM_MIB=1104,1104``), while the rank's
vote keeps its card's DERIVED line (``extend_trim.corridor_line_mib``: MEASURED-D 767 / 701 / 700 on the drei rig, boot
..._1ecda02a58_1024_110644 D: ``EXTEND-CORRIDOR-LINE rank=0 line_mib=767 source=MEASURED-D:767``). Under a measured
line the stated law only trims earlier; over it (a measured card law above 1024) the threshold sits UNDER the card's
line + floor chunk -- the cache stays, the vote cannot fund the floor chunk above the line, the Q-710 floor raises the
width to what the whole free card funds, and the forward ends under the card's law. That direction is pinned here
(``MeasuredOverTheStatedLaw`` and ``RankLift``): it must be RED when the threshold is priced under the card's law.
"""
import json
import os
import types
import unittest
from unittest import mock

try:
    from flliper.test.ci.ci_register import register_cpu_ci
except ImportError:  # pragma: no cover

    def register_cpu_ci(*args, **kwargs):
        return None


import pytest  # noqa: E402

from flliper.srt.managers import corridor_guard as CG
from flliper.srt.pdflip import extend_trim as ET
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

MIB = 1 << 20
RATE = 0.3091
RATES3 = f"{RATE},{RATE},{RATE}"
RATE_ENV = "FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB"
TRIM_ENV = "FLLIPER_PDFLIP_EXTEND_TRIM_MIB"
FIXED_ENV = "FLLIPER_PDFLIP_EXTEND_VOTE_FIXED_MIB"
LAW_ENV = "FLLIPER_CORRIDOR_LAW_FLOOR_MIB"
UUIDS = ("GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d",  # D TP0, RTX 5090 (drei rig)
         "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4",  # D TP1, RTX 3080
         "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7")  # D TP2, RTX 3080


@pytest.fixture(autouse=True)
def _stated_law_default(monkeypatch):
    """No law override: the stated law is the shipped 1024 (the metal boots set none)."""
    monkeypatch.delenv(LAW_ENV, raising=False)
    ET.reset_for_tests()
    yield
    ET.reset_for_tests()


def _cards(n=3):
    from flliper.srt.pdflip import launcher as L

    names = ("NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 3080", "NVIDIA GeForce RTX 3080")
    totals = (32607, 20480, 20480)
    return [L.Card(nvml_index=i, uuid=UUIDS[i], name=names[i], total_mib=totals[i]) for i in range(n)]


def _store(tmp_path, uuids, tp=3, pp=1):
    """A phase-footprint store of THIS form (cards = the D cards, tp x pp = their count)."""
    p = tmp_path / "phase_footprint-a191a0712717-cc0ac5caf91e.json"
    p.write_text(json.dumps({"cards": {u: {"activation_mib": 1} for u in uuids},
                             "profile": [4096, "x", tp, pp, "fp8_e4m3", 3, 24]}))
    return str(p)


def _floors(transients, source, store_path=""):
    def fake(cards, group, user_reserve_mib=None):
        return {c.uuid: CG.CorridorFloor(card_uuid=c.uuid, group="D", transient_mib=int(t), reserve_mib=0,
                                         source=source,
                                         provenance=f"{store_path}#cards.{c.uuid}.activation_mib={int(t)}MiB")
                for c, t in zip(cards, transients)}
    return fake


def _ns(env_d=""):
    return types.SimpleNamespace(profile="qwen27b", model="/nonexistent/Qwen3.8-27B-INT8-gdncov", extra_d="",
                                 env_d=env_d, dual_layout=False, dual_share=False)


def _write(ns, cards, transients, source, tmp_path, rates=RATES3, store_uuids=None):
    from flliper.srt.pdflip import launcher as L

    path = _store(tmp_path, store_uuids if store_uuids is not None else [c.uuid for c in cards])
    lines = []
    with mock.patch.object(L.corridor_budget, "floors_for_cards", _floors(transients, source, path)):
        now = L._d_extend_flip_trim_env(ns, lines.append, "D", rates, cards)
    return now, lines


class Derivation(CustomTestCase):
    def test_per_rank_lines(self):
        # drei D measured 767 / 701 / 700: ceil(line + 256 x 0.3091 = 79.13)
        self.assertEqual(ET.flip_trim_thresholds(RATES3, 4096, 1, [767, 701, 700]), "847,781,780")

    def test_scalar_is_the_old_form(self):
        self.assertEqual(ET.flip_trim_thresholds(f"{RATE},{RATE}", 4096, 1, 1024), "1104,1104")

    def test_single_line(self):
        # VOTE-LINE-SINGLE: near-OOM 400 + fixed 238 = 638
        self.assertEqual(ET.flip_trim_thresholds(RATES3, 4096, 1, [638, 638, 638]), "718,718,718")

    def test_length_mismatch_takes_the_highest_line(self):
        """Never zipped short and never the lowest line: a threshold under a card's line is the OOM direction."""
        self.assertEqual(ET.flip_trim_thresholds(RATES3, 4096, 1, [700, 1300]), "1380,1380,1380")

    def test_rank_without_rate_still_never_trims(self):
        self.assertEqual(ET.flip_trim_thresholds(f"{RATE},0,{RATE}", 4096, 1, [767, 701, 700]), "847,0,780")


class LauncherCardLaw(CustomTestCase):
    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = tmp_path

    def test_measured_drei_takes_each_cards_line(self):
        ns = _ns()
        now, lines = _write(ns, _cards(), (767, 701, 700), "MEASURED-D", self.tmp)
        self.assertEqual(now, "847,781,780")
        from flliper.srt.pdflip import launcher as L

        self.assertEqual(L.parse_group_env(ns.env_d)[TRIM_ENV], "847,781,780")
        ln = [x for x in lines if L.EXTEND_TRIM_FLIP_MARKER in x]
        self.assertEqual(len(ln), 1, lines)
        self.assertIn(L.TRIM_LAW_MARKER, ln[0])
        self.assertIn("rang0 nvml0: Kartengesetz 767 (MEASURED-D:767) -> 847", ln[0])
        self.assertIn("rang2 nvml2: Kartengesetz 700 (MEASURED-D:700) -> 780", ln[0])

    def test_unmeasured_int4ar_is_the_metal_value_named(self):
        """int4ar 2x3080 (boot _1024_125438): UNMEASURED-FALLBACK 1024 on both D cards -> 1104,1104 as before,
        with the fallback named."""
        ns = _ns()
        now, lines = _write(ns, _cards(2)[:2], (1024, 1024), CG.FLOOR_SOURCE_FALLBACK, self.tmp,
                            rates=f"{RATE},{RATE}")
        self.assertEqual(now, "1104,1104")
        self.assertTrue(any("Rueckfall Korridor-Gesetz 1024" in x and "nicht gemessen" in x for x in lines), lines)

    def test_foreign_form_store_falls_back_named(self):
        """A measured floor whose footprint is of another form (a duo store read by a drei pass) is not this
        form's line: the stated law, with the form reason."""
        ns = _ns()
        now, lines = _write(ns, _cards(), (767, 701, 700), "MEASURED-D", self.tmp, store_uuids=list(UUIDS[:2]))
        self.assertEqual(now, "1104,1104,1104")
        self.assertTrue(any("Rueckfall Korridor-Gesetz 1024 (footprint cards" in x for x in lines), lines)

    def test_no_cards_is_the_stated_law_named(self):
        from flliper.srt.pdflip import launcher as L

        ns, lines = _ns(), []
        now = L._d_extend_flip_trim_env(ns, lines.append, "D", RATES3)
        self.assertEqual(now, "1104,1104,1104")
        self.assertTrue(any("keine D-Karten an dieser Stelle" in x for x in lines), lines)

    def test_vote_line_single_prices_the_ranks_line(self):
        """VOTE-LINE-SINGLE armed (the env the ranks read carries the fixed part): the line is 400 + 238."""
        ns = _ns(env_d=f"{FIXED_ENV}=238")
        now, lines = _write(ns, _cards(), (767, 701, 700), "MEASURED-D", self.tmp)
        self.assertEqual(now, "718,718,718")
        self.assertTrue(any("VOTE-LINE-SINGLE near-OOM + fixed" in x for x in lines), lines)

    def test_hand_value_wins_and_the_computed_value_is_named(self):
        """INT8 27b.env ``1200,1100,800`` stays byte for byte; the log names what the card law would give."""
        from flliper.srt.pdflip import launcher as L

        given = f"{TRIM_ENV}=1200,1100,800"
        ns = _ns(env_d=given)
        now, lines = _write(ns, _cards(), (767, 701, 700), "MEASURED-D", self.tmp)
        self.assertEqual(now, "1200,1100,800")
        self.assertTrue(ns.env_d.startswith(given))
        self.assertTrue(any("aus --env-d gesetzt, unveraendert; berechnet 847,781,780" in x for x in lines), lines)

    def _arm(self, fixed_record):
        from flliper.srt.pdflip import launcher as L

        ns = _ns()
        cards = _cards()
        path = _store(self.tmp, [c.uuid for c in cards])
        with mock.patch.object(L, "_d_extend_cap_rate_env", lambda *a, **k: RATES3), \
                mock.patch.object(L, "d_extend_vote_fixed_record", lambda *a, **k: fixed_record), \
                mock.patch.object(L.corridor_budget, "floors_for_cards", _floors((767, 701, 700), "MEASURED-D", path)):
            L._d_extend_flip_rate_env(ns, [].append, "D", cards)
        return L.parse_group_env(ns.env_d)

    def test_flip_arm_hands_cards_through(self):
        """The armed branch passes the D cards: the per-card line reaches the env (integration)."""
        env = self._arm((None, "test: record off"))
        self.assertEqual(env[TRIM_ENV], "847,781,780")
        self.assertNotIn(FIXED_ENV, env)

    def test_flip_arm_prices_the_fixed_part_it_just_wrote(self):
        """VOTE-LINE-SINGLE armed by the same pass: the fixed part is written BEFORE the threshold is priced, so the
        threshold follows the line the ranks will keep (400 + 238), not the floor line."""
        env = self._arm(([238.0, 238.0, 238.0], "test record"))
        self.assertEqual(env[FIXED_ENV], "238,238,238")
        self.assertEqual(env[TRIM_ENV], "718,718,718")


class MeasuredOverTheStatedLaw(CustomTestCase):
    """THE DANGER DIRECTION at the launcher: a measured D card law above 1024. Pre-#201 the threshold was 1104 for
    it -- 276 MiB under ``line 1300 + floor chunk 79``. Must be RED when the threshold is priced under the card law."""

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = tmp_path

    def test_threshold_never_under_the_cards_line_plus_floor_chunk(self):
        ns = _ns()
        now, _ = _write(ns, _cards(), (1300, 701, 700), "MEASURED-D", self.tmp)
        thr = [int(x) for x in now.split(",")]
        for line, t in zip((1300, 701, 700), thr):
            self.assertGreaterEqual(t, line + ET.floor_rows(4096, 1) * RATE)
        self.assertEqual(thr[0], 1380)


class _Cuda:
    """A D card: ``free`` driver MiB, ``releasable`` MiB of wholly free cache segments."""

    def __init__(self, free, releasable):
        self.free = float(free)
        self.reserved = 18476.0
        self.allocated = 17184.0
        self.releasable = float(releasable)
        self.emptied = 0

    def is_current_stream_capturing(self):
        return False

    def mem_get_info(self, *a):
        return int(self.free * MIB), 20055 * MIB

    def memory_reserved(self, *a):
        return int(self.reserved * MIB)

    def memory_allocated(self, *a):
        return int(self.allocated * MIB)

    def synchronize(self):
        pass

    def empty_cache(self):
        self.emptied += 1
        self.reserved -= self.releasable
        self.free += self.releasable
        self.releasable = 0.0


class RankLift(CustomTestCase):
    """THE DANGER DIRECTION at the rank: the launcher priced a line under the rank's own (its read named another
    floor than the rank's). The rank lifts its static threshold to its own line + floor chunk at its first vote."""

    def _vote(self, trim, line, free, releasable, src="MEASURED-D"):
        cuda = _Cuda(free, releasable)
        env = {RATE_ENV: f"{RATE},{RATE}", "FLLIPER_PDFLIP_EXTEND_CAP_FLOOR": "1"}
        if trim is not None:
            env[TRIM_ENV] = trim
        with mock.patch.dict(os.environ, env, clear=False), \
                mock.patch.object(ET, "_read_corridor_line", lambda: (float(line), f"{src}:{int(line)}")):
            os.environ.pop("FLLIPER_PDFLIP_TORCH_CACHE_CAP", None)
            os.environ.pop(FIXED_ENV, None)
            if trim is None:
                os.environ.pop(TRIM_ENV, None)
            vote = ET.width_vote(cuda, 1, 4096, 1, True)
        return vote, cuda

    def test_card_does_not_run_under_its_law(self):
        """Card law 1300, launcher threshold 1104, 1200 MiB driver free with 300 MiB wholly free cache: the lifted
        threshold 1380 trims, the vote funds from 1500 above the line, and the chunk ends at or over the line.
        Without the lift: no trim, the vote is 1 row, the Q-710 floor raises it to 256 and the chunk ends at
        1200 - 79 = 1121 MiB -- 179 under the card's law. RED there."""
        vote, cuda = self._vote("1104,1104", 1300, 1200, 300)
        self.assertEqual(cuda.emptied, 1)
        self.assertGreaterEqual(cuda.free - vote * RATE, 1300 - 1e-6)
        self.assertEqual(ET.threshold_for(1, ET.thresholds()), 1380.0)

    def test_lift_is_logged_once_and_read_by_every_trim_site(self):
        with self.assertLogs(ET.logger, level="INFO") as cm:
            self._vote("1104,1104", 1300, 1200, 300)
            self._vote("1104,1104", 1300, 1200, 300)
        lifts = [m for m in cm.output if ET.LAW_LIFT_MARKER in m]
        self.assertEqual(len(lifts), 1, cm.output)
        self.assertIn("threshold=1104 -> 1380", lifts[0])
        # the forward trim (``_before_extend_trim``) reads the same parsed list
        self.assertEqual(ET.thresholds(), [1104.0, 1380.0])

    def test_single_value_env_covers_only_this_process(self):
        self._vote("1104", 1300, 1200, 300)
        self.assertEqual(ET.thresholds(), [1380.0])

    def test_never_lowered(self):
        """INT8 hand value 1200 over line 767 + 79: byte for byte; and 1104 over a 700 line stays 1104."""
        self._vote("1200,1200", 767, 2000, 0)
        self.assertEqual(ET.threshold_for(1, ET.thresholds()), 1200.0)
        ET.reset_for_tests()
        self._vote("1104,1104", 700, 2000, 0)
        self.assertEqual(ET.threshold_for(1, ET.thresholds()), 1104.0)

    def test_metal_int4ar_no_lift(self):
        """int4ar metal: line 1024 UNMEASURED-FALLBACK, threshold 1104 = ceil(1024 + 79.13): no lift, no log."""
        with self.assertLogs(ET.logger, level="INFO") as cm:
            self._vote("1104,1104", 1024, 2000, 0, src="UNMEASURED-FALLBACK")
            ET.logger.info("sentinel")
        self.assertFalse(any(ET.LAW_LIFT_MARKER in m for m in cm.output), cm.output)
        self.assertEqual(ET.threshold_for(1, ET.thresholds()), 1104.0)

    def test_nothing_armed_where_nothing_was(self):
        """No env threshold: no trim, no lift (other profiles byte for byte); ``0`` stays never-trims."""
        vote, cuda = self._vote(None, 1300, 1200, 300)
        self.assertEqual(cuda.emptied, 0)
        self.assertIsNone(ET.thresholds())
        ET.reset_for_tests()
        vote, cuda = self._vote("0,0", 1300, 1200, 300)
        self.assertEqual(cuda.emptied, 0)
        self.assertEqual(ET.thresholds(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
