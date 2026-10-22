"""PLANER A2 1022: ``propose_oracle.d_decode_figures`` -- group D's decode figures as the launcher printed them (the input of the planner page's
"startable, slow (estimate)" hint, rigdash ``planer_ziel.slow_hint``).

Fixtures ``fixtures/planer_a2_1022/*.lines.txt``: the D-decode lines of four real launcher dry runs (22.10., this tree, ``propose_oracle.run_profile`` with
``--force`` on catalog cards / the rig NVML recording ``nvml_devices_1378.json``), each line verbatim:

  nf_2x3090  profiles_release/nf-2x3090.env, 2 x RTX 3090   -> waves 10, round bs1 73.84/73.84 ms, the nfblock wave fallback (Env 2 -> Form A 6)
  nf_3x3090  profiles_release/nf-3x3090.env, 3 x RTX 3090   -> waves 3, round bs1 27.26/24.66/23.96 ms, no fallback
  nf_4x3090  profiles_release/nf-4x3090.env, 4 x RTX 3090   -> waves 3, Form A kept (no cut solve, no round)
  nf_rig     profiles_release/nf.env, rig NVML recording    -> waves 2, round bs1 22.93/24.27/24.77 ms, no fallback
"""

import os
import unittest

from flliper.srt.pdflip import propose_oracle as O

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "planer_a2_1022")


def lines(name):
    with open(os.path.join(FIX, name + ".lines.txt"), encoding="utf-8") as fh:
        return fh.read().splitlines()


class DecodeFigures(unittest.TestCase):
    def test_two_rtx3090_the_review_case(self):
        d = O.d_decode_figures(lines("nf_2x3090"))
        self.assertEqual(d["waves"], 10)                                         # what group D is STARTED with (the last PDFLIP-GROUP-ENV D), not a pass's 14
        self.assertEqual(d["waves_default"], 2)
        self.assertEqual(d["round_bs1_ms"], [73.84, 73.84])                      # the LAST owned solve (the pinned form), not the first pass's 75.64/70.23
        self.assertEqual(d["form_a_round_bs1_ms"], [57.35, 106.15])
        self.assertEqual(d["miss_cost"], "0.1/0.2")
        self.assertTrue(d["miss_cost_source"].startswith("Saat UNMEASURED"))
        self.assertEqual(d["wave_fallback"], {"env_waves": 2, "form_a_waves": 6, "result": "12 tragbar"})

    def test_three_rtx3090_and_the_rig_have_no_wave_fallback(self):
        d3 = O.d_decode_figures(lines("nf_3x3090"))
        self.assertEqual((d3["waves"], d3["waves_default"], d3["round_bs1_ms"], d3["wave_fallback"]), (3, 2, [27.26, 24.66, 23.96], None))
        rig = O.d_decode_figures(lines("nf_rig"))
        self.assertEqual((rig["waves"], rig["waves_default"], rig["round_bs1_ms"], rig["wave_fallback"]), (2, 2, [22.93, 24.27, 24.77], None))

    def test_form_a_kept_prints_no_round(self):
        d = O.d_decode_figures(lines("nf_4x3090"))
        self.assertEqual((d["waves"], d["round_bs1_ms"], d["form_a_round_bs1_ms"], d["wave_fallback"]), (3, None, None, None))

    def test_the_p_group_env_is_not_read_as_d(self):
        d = O.d_decode_figures(["[TS] PDFLIP-LAUNCH PDFLIP-GROUP-ENV P: A=1;FLLIPER_OPT_MOE_POOL_OVERFLOW_WAVES=9;B=2",
                                "[TS] PDFLIP-LAUNCH PDFLIP-GROUP-ENV D: A=1;FLLIPER_OPT_MOE_POOL_OVERFLOW_WAVES_X=7;B=2"])
        self.assertIsNone(d["waves"])                                            # neither P's value nor a longer name's

    def test_a_plan_without_these_lines_has_no_figures(self):
        d = O.d_decode_figures(["[TS] PDFLIP-LAUNCH PP-CUT SHIPPED: layers=40,13,11"])
        self.assertEqual(set(d.values()), {None})

    def test_parse_plan_dump_and_plan_summary_carry_them(self):
        from flliper.srt.pdflip import propose_verdict as V

        text = "# argv_n=3 rc=0 exc=: \n" + "\n".join(lines("nf_2x3090")) + "\n"
        self.assertEqual(O.parse_plan_dump(text)["d_decode"], O.d_decode_figures(lines("nf_2x3090")))

        class Res:
            def dump(self):
                return text

        self.assertEqual(V.plan_summary(Res())["d_decode"]["waves"], 10)


if __name__ == "__main__":
    unittest.main()
