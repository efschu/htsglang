# SPDX-License-Identifier: Apache-2.0
"""27B int4ar 23.10.: the flip arm arms the rc12e trim wherever it arms the width vote.

METAL (boot dkr27bint4arbar1int4ar10231400, a45112428e, D.log _1023_140032, D = TP2 on 2x RTX 3080, W4A16 Marlin):
the profile (derived from the duo profile) carried no ``FLLIPER_PDFLIP_EXTEND_TRIM_MIB`` -> 0 trims all boot. D TP1
sat at 40-170 MiB driver free from 14:11Z on, reserved 18376-18478 MiB against ~17150-17250 MiB allocated. The vote
(line 1024 UNMEASURED-FALLBACK, rate 0.3091) was 1 row, the Q-710 floor raised it to 256
(``EXTEND-CAP-FLOOR rank=1 cap=1 width=256 floor=256 post=918``) and to 135 / 128 at 42 / 40 MiB
(``EXTEND-CAP-FLOOR rank=1 cap=1 width=135 floor=256 post=42``). The flip window 14:18-14:23Z: 84 of 162 TP0 extends
were 128-191 rows wide at ~274 ms each.
"""
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

from flliper.srt.pdflip import extend_trim as ET
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MIB = 1 << 20
RATE = 0.3091
RATE_ENV = "FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB"
TRIM_ENV = "FLLIPER_PDFLIP_EXTEND_TRIM_MIB"
LAW_ENV = "FLLIPER_CORRIDOR_LAW_FLOOR_MIB"
# ceil(1024 + 256 x 0.3091) = ceil(1103.13)
THRESHOLD = 1104


@pytest.fixture(autouse=True)
def _law_1024(monkeypatch):
    """The metal boot's line: the stated law 1024 (``EXTEND-CORRIDOR-LINE ... source=UNMEASURED-FALLBACK:1024``).

    EXTEND-CAP-SOFT (27.10.): this module pins the pre-soft vote the 23.10. boot ran (plan line, start rate -- the
    FLLIPER_PDFLIP_EXTEND_SOFT_LINE=0 path); the soft rule is pinned in test_pdflip_extend_soft_line_1027."""
    monkeypatch.setenv(LAW_ENV, "1024")
    monkeypatch.setenv("FLLIPER_PDFLIP_EXTEND_SOFT_LINE", "0")
    ET.reset_for_tests()
    yield


def _ns(profile="qwen27b", env_d="", **kw):
    base = dict(profile=profile, model="/nonexistent/Qwen3.8-27B-INT8-gdncov", extra_d="", env_d=env_d,
                dual_layout=False, dual_share=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


class Derivation(CustomTestCase):
    def test_threshold_is_line_plus_floor_chunk_growth(self):
        self.assertEqual(ET.floor_rows(4096, 1), 256)
        self.assertEqual(ET.flip_trim_thresholds(f"{RATE},{RATE}", 4096, 1, 1024), f"{THRESHOLD},{THRESHOLD}")

    def test_rank_without_rate_never_trims(self):
        self.assertEqual(ET.flip_trim_thresholds(f"{RATE},0", 4096, 1, 1024), f"{THRESHOLD},0")
        self.assertEqual(ET.flip_trim_thresholds("", 4096, 1, 1024), "")

    def test_measured_law_moves_the_threshold(self):
        # INT8 3080 measured line 700 (round 3c): 700 + 79.1
        self.assertEqual(ET.flip_trim_thresholds(f"{RATE}", 4096, 1, 700), "780")


class FlipArmWritesTheTrim(CustomTestCase):
    def test_flip_arm_writes_trim_beside_rate_and_floor(self):
        """RED on 3318be15cf: env_d carried rate + floor, no trim (the int4ar boot's group env)."""
        from flliper.srt.pdflip import launcher as L

        ns, lines = _ns(), []
        rtext = L._d_extend_flip_rate_env(ns, lines.append, "D")
        self.assertTrue(rtext)
        env = L.parse_group_env(ns.env_d)
        n = len(rtext.split(","))
        self.assertEqual(env.get(TRIM_ENV), ",".join([str(THRESHOLD)] * n))
        self.assertEqual(env.get("FLLIPER_PDFLIP_EXTEND_CAP_FLOOR"), "1")
        self.assertIn(RATE_ENV, env)
        self.assertTrue(any(L.EXTEND_TRIM_FLIP_MARKER in ln and "aus --env-d" not in ln for ln in lines), lines)

    def test_second_pass_is_idempotent(self):
        from flliper.srt.pdflip import launcher as L

        ns = _ns()
        L._d_extend_flip_rate_env(ns, [].append, "D")
        first = ns.env_d
        lines = []
        L._d_extend_flip_rate_env(ns, lines.append, "D")
        self.assertEqual(ns.env_d, first)
        self.assertFalse(any("aus --env-d" in ln for ln in lines if L.EXTEND_TRIM_FLIP_MARKER in ln))

    def test_profile_value_wins(self):
        """The INT8 27b.env hand value stays byte-identical."""
        from flliper.srt.pdflip import launcher as L

        given = f"{TRIM_ENV}=1200,1100,800"
        ns, lines = _ns(env_d=given), []
        L._d_extend_flip_rate_env(ns, lines.append, "D")
        self.assertTrue(ns.env_d.startswith(given + ";"))
        self.assertEqual(L.parse_group_env(ns.env_d)[TRIM_ENV], "1200,1100,800")
        self.assertTrue(any(L.EXTEND_TRIM_FLIP_MARKER in ln and "aus --env-d gesetzt" in ln for ln in lines), lines)

    def test_dual_and_nextflash_untouched(self):
        from flliper.srt.pdflip import launcher as L

        for ns in (_ns(dual_layout=True), _ns(dual_share=True), _ns(profile="nextflash")):
            lines = []
            self.assertIsNone(L._d_extend_flip_rate_env(ns, lines.append, "D"))
            self.assertEqual((ns.env_d, lines), ("", []))


class _Cuda:
    """D TP1 at 14:20:05Z: 42 MiB driver free, reserved 18476, allocated ~17184 (1292 MiB in the cache)."""

    def __init__(self, free, releasable):
        self.free = free
        self.reserved = 18476
        self.allocated = 17184
        self.releasable = releasable
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
        self.releasable = 0


class RankVoteOnTheMetalState(CustomTestCase):
    def setUp(self):
        ET.reset_for_tests()

    def tearDown(self):
        ET.reset_for_tests()

    def _vote(self, env, free, releasable):
        cuda = _Cuda(free, releasable)
        base = {RATE_ENV: f"{RATE},{RATE}", "FLLIPER_PDFLIP_EXTEND_CAP_FLOOR": "1", LAW_ENV: "1024"}
        base.update(env)
        with mock.patch.dict(os.environ, base, clear=False):
            os.environ.pop("FLLIPER_PDFLIP_TORCH_CACHE_CAP", None)
            if TRIM_ENV not in env:
                os.environ.pop(TRIM_ENV, None)
            return ET.width_vote(cuda, 1, 4096, 1, True), cuda

    def test_without_trim_the_metal_widths(self):
        """The boot's lines: post 918 -> 256, post 42 -> 135 (no trim armed, the cache never returns)."""
        vote, cuda = self._vote({}, 918, 300)
        self.assertEqual((vote, cuda.emptied), (256, 0))
        ET.reset_for_tests()
        vote, cuda = self._vote({}, 42, 300)
        self.assertEqual((vote, cuda.emptied), (135, 0))

    def test_with_trim_the_floor_chunk_holds(self):
        """With the flip arm's threshold the vote trims first; any release >= 38 MiB restores the 256-row floor
        (the release itself is NOT measured on this boot -- INT8 3080 trims released 276-346 MiB)."""
        vote, cuda = self._vote({TRIM_ENV: f"{THRESHOLD},{THRESHOLD}"}, 42, 38)
        self.assertEqual(cuda.emptied, 1)
        self.assertEqual(vote, 256)

    def test_trim_does_not_fire_above_threshold(self):
        vote, cuda = self._vote({TRIM_ENV: f"{THRESHOLD},{THRESHOLD}"}, 1400, 300)
        self.assertEqual(cuda.emptied, 0)
        # floor((1400 - 1024) / 0.3091) = 1216; the 14:05:22 TP0 line printed cap=1215 at post=1400 (a rounded
        # 1399.x MiB reading)
        self.assertEqual(vote, 1216)


if __name__ == "__main__":
    unittest.main()
