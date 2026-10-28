"""fnFL2 v39 (21.09.): group P's speculative flags must be GROUP D's, not the
module constants.

The constants are the 27B/DFlash2 depth (2 steps, 3 draft tokens).  The Next
Flash arm runs the MTP head at 3/4 through ``--extra-d``.  P shipped the
constants, so the two groups registered different drafters and D would have
asked the carrier for ``{hash}.draft-<other id>`` pages P never wrote.
"""

import unittest

from flliper.srt.pdflip import launcher as L


def _launcher_text() -> str:
    """The launcher source as one text (launcher.py + launcher_*.py) with the split functions expanded back to the original statement
    order: the launcher was split behaviour-neutrally (refactor 28.10.), see ``_launcher_source.expanded_launcher_text``."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location("_launcher_source", pathlib.Path(__file__).with_name("_launcher_source.py"))
    ls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ls)
    return ls.expanded_launcher_text()


class PDraftDepthFollowsD(unittest.TestCase):
    def test_extra_d_depth_wins_over_the_constants(self):
        flags = L.p_draft_kv_flags(
            "--speculative-draft-placement solo --speculative-algorithm NEXTN "
            "--speculative-num-steps 3 --speculative-eagle-topk 1 "
            "--speculative-num-draft-tokens 4".split()
        )
        self.assertEqual(flags[flags.index("--speculative-num-steps") + 1], "3")
        self.assertEqual(
            flags[flags.index("--speculative-num-draft-tokens") + 1], "4"
        )
        self.assertEqual(flags[flags.index("--speculative-algorithm") + 1], "NEXTN")

    def test_an_arm_that_names_nothing_keeps_the_constants(self):
        self.assertEqual(
            L.p_draft_kv_flags([]),
            tuple(L.P_DRAFT_KV_FLAGS),
        )

    def test_kv_only_is_p_s_alone_and_always_last(self):
        for extra in ([], "--speculative-num-steps 5".split()):
            self.assertEqual(
                L.p_draft_kv_flags(extra)[-1], "--speculative-draft-kv-only"
            )
        self.assertNotIn(
            "--speculative-draft-placement", L.p_draft_kv_flags(
                "--speculative-draft-placement solo".split()
            )
        )

    def test_argv_p_ships_the_handed_flags(self):
        argv = L.argv_p(
            "py", "/m", [1, 1, 1], 1, 1, L.RING_FORM_SENTINEL_STORE_CFG, [],
            p_bs=1, draft_kv_on_p=True,
            spec_flags=L.p_draft_kv_flags(
                "--speculative-num-steps 3 --speculative-num-draft-tokens 4".split()
            ),
        )
        self.assertEqual(argv[argv.index("--speculative-num-steps") + 1], "3")
        self.assertEqual(argv[argv.index("--speculative-num-draft-tokens") + 1], "4")

    def test_both_live_call_sites_hand_it_in(self):
        import inspect

        src = _launcher_text()
        self.assertEqual(
            src.count("spec_flags=p_draft_kv_flags(shlex.split(ns.extra_d))"), 2
        )


if __name__ == "__main__":
    unittest.main()
