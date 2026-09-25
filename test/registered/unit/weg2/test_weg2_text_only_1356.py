# SPDX-License-Identifier: Apache-2.0
"""#1356 -- P and D boot TEXT-ONLY, and the tower does not load.

MEASURED on weg2xsn27: the vision tower was loaded on all six ranks and
replicated threefold on P -- 879 MiB per P card in the manifest, 2.63 GiB
across the host ring. No Weg-2 form serves images.

THE MECHANISM IS NOT `--language-only`. That flag is the encoder-disagg
RECEIVER and refuses without `--encoder-urls` (server_args.py:17996), so
passing it would kill the launch rather than drop the tower.

AND IT WAS NOT SPELLABLE. `enable_multimodal` is declared `Optional[bool]` and
its `False` branch is live in model_config.py:436-447, read at
server_args.py:14112 -- but the CLI generated a bare `store_true`, so `False`
existed in the type and in the config and NO CALLER COULD SAY IT. This commit
makes it a `BooleanOptionalAction`: the tri-state is unchanged, one spelling
was missing.

NOT a model-name list: adding the architecture to `mm_disabled_models` would
be the #1362 fossil class, a constant with no model reference.
"""

import argparse
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher as lc
from sglang.srt.weg2 import ring_table as rt
from sglang.test.test_utils import CustomTestCase


class TheOffSwitchIsSpellable(CustomTestCase):
    def test_the_tri_state_has_all_three_spellings(self):
        from sglang.srt.server_args import ServerArgs

        ap = argparse.ArgumentParser()
        ServerArgs.add_cli_args(ap)
        opts = [o for a in ap._actions for o in a.option_strings]
        self.assertIn("--enable-multimodal", opts)
        self.assertIn("--no-enable-multimodal", opts)
        base = ["--model-path", "/m"]
        self.assertIsNone(ap.parse_args(base).enable_multimodal)
        self.assertTrue(ap.parse_args(base + ["--enable-multimodal"]).enable_multimodal)
        self.assertFalse(
            ap.parse_args(base + ["--no-enable-multimodal"]).enable_multimodal)

    def test_language_only_is_NOT_the_mechanism(self):
        """The refusal that sent this slice back once -- pinned so it stays."""
        import inspect

        from sglang.srt import server_args as sa

        src = inspect.getsource(sa)
        self.assertIn("--language-only is set without --encoder-urls", src)
        self.assertNotIn("language_only", inspect.getsource(lc))


class BothGroupsBootTextOnly(CustomTestCase):
    def test_off_is_the_default_and_emits_the_flag(self):
        off = lc.common_flags("/m", 1, 150, "x", 262144)
        self.assertIn("--no-enable-multimodal", off)

    def test_resident_keeps_the_tower_and_is_byte_identical_to_before(self):
        res = lc.common_flags("/m", 1, 150, "x", 262144, vision=lc.VISION_RESIDENT)
        self.assertNotIn("--no-enable-multimodal", res)
        self.assertNotIn("--enable-multimodal", res)

    def test_the_knob_exists_with_off_as_default(self):
        ns = lc.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
        self.assertEqual(ns.weg2_vision, lc.VISION_OFF)
        ns2 = lc.build_parser().parse_args(
            ["--tree", "/t", "--tag", "t", "--weg2-vision", "resident"])
        self.assertEqual(ns2.weg2_vision, lc.VISION_RESIDENT)


class TheFormKeyConsequenceIsNamed(CustomTestCase):
    """W48 EXPECTED ONCE, and W20 NOT.

    A tower that is absent IS a different weight statement, so the flag is in
    the argv and deliberately NOT in FORM_KEY_EXCLUDED_FLAGS. The first boot of
    the text-only form therefore has no same-form predecessor and solves its
    ring from ITS OWN stem. Inheriting the multimodal form's Sigma H would
    price 2.63 GiB of tower this form does not hold.
    """

    def _k(self, **kw):
        argv = ["py", "-m", "sglang.launch_server"] + lc.common_flags(
            "/m", 1, 150, "x", 262144, **kw)
        return rt.p_form_key(argv)[0]

    def test_text_only_and_resident_are_different_forms(self):
        self.assertNotEqual(self._k(), self._k(vision=lc.VISION_RESIDENT))

    def test_the_flag_is_not_excluded_from_the_form_key(self):
        self.assertNotIn("--no-enable-multimodal", rt.FORM_KEY_EXCLUDED_FLAGS)
        self.assertNotIn("--weg2-vision", rt.FORM_KEY_EXCLUDED_FLAGS)


class ImagesAreRefusedByName(CustomTestCase):
    def test_an_image_part_is_counted_structurally(self):
        from sglang.srt.weg2 import front as fr

        img = {"messages": [{"content": [{"type": "image_url"}, {"type": "text"}]}]}
        self.assertEqual(fr._image_parts(img), 1)

    def test_a_prompt_that_merely_says_image_is_text(self):
        """The #995 prose trap one layer up: the word is not the thing."""
        from sglang.srt.weg2 import front as fr

        txt = {"messages": [{"content": [{"type": "text",
                                          "text": "describe this image_url"}]}]}
        self.assertEqual(fr._image_parts(txt), 0)
        self.assertEqual(fr._image_parts({"prompt": "an image"}), 0)
        self.assertEqual(fr._image_parts(None), 0)

    def test_the_refusal_names_the_code_and_the_way_out(self):
        import inspect

        from sglang.srt.weg2 import front as fr

        src = inspect.getsource(fr.Front.handle_generate)
        self.assertIn("W101 Weg2VisionRefused", src)
        self.assertIn("status=501", src)
        # #58 (0aeb52e070): the way out is written by vision_verdict, whose
        # text handle_generate returns in the 501 body -- asserted there.
        verdict, why = fr.vision_verdict(1, 0, fr.VISION_MODE_OFF)
        self.assertEqual(verdict, fr.VERDICT_REFUSE_IMAGE)
        self.assertIn("--weg2-vision resident", why)


if __name__ == "__main__":
    unittest.main()


class TheProductionCallerBuildsTheArgv(CustomTestCase):
    """#1356 [fix] -- a boot blocker my desk tests were blind to BY SHAPE.

    2cc618b819 inserted `vision` mid-signature while BOTH production callers of
    `argv_p` pass positionally (:10189 the form-key build, :10620 the argv the
    ranks RUN). SEVEN parameters shifted one place: vision<-depth,
    depth<-window_mib, window_mib<-random_seed, random_seed<-cap_cycles,
    census_interval<-draft_kv_on_p, and so on.

    WHAT SAVED THIS BOOT WAS A TYPE ACCIDENT, NOT A GUARD. `window_mib` is the
    string "24,PP_0=96", so `str(int(depth))` raised and every boot died loudly
    at launcher.py:2775. Had that value been purely numeric the launch would
    have SUCCEEDED with the seed, the BAR1 window and the census interval
    silently swapped, and the form key hashed over an argv nobody ever meant --
    a wrong ring table inherited on a form that never existed. The crash was
    the good case, and it is the only reason this was found before a boot.

    MY TESTS COULD NOT SEE IT: they called `argv_p(...)` with KEYWORDS, which is
    immune to a positional shift by construction. A test that builds its call
    differently from the caller it guards is testing a different function.

    THE REPAIR IS THE CLASS, NOT THE INSTANCE: `argv_p` is keyword-only from
    its first OPTIONAL parameter, so a positional caller can no longer reach any
    optional term at all, and both call sites name every argument. The AST pin
    below keeps it that way for whoever adds the next parameter.
    """

    def test_vision_is_keyword_only_in_both_signatures(self):
        import inspect

        for fn in (lc.common_flags, lc.argv_p):
            with self.subTest(fn=fn.__name__):
                kind = inspect.signature(fn).parameters["vision"].kind
                self.assertEqual(
                    kind, inspect.Parameter.KEYWORD_ONLY,
                    f"{fn.__name__}'s `vision` must be keyword-only, or the "
                    f"next inserted parameter shifts a positional caller again")

    def test_the_positional_call_shape_of_every_caller_still_works(self):
        """EXECUTION, in the caller's own shape -- not a keyword paraphrase."""
        # argv_p:2772 and argv_d:2888 shape
        out = lc.common_flags("/m", 1, 150, "store", 262144,
                              "write_through", "P", 785500001, 300000000000, 50)
        self.assertIn("--model-path", out)
        # the form-key build at :10060 shape (one fewer positional)
        out2 = lc.common_flags("/m", lc.RING_FORM_SENTINEL_S_GB,
                               lc.RING_FORM_SENTINEL_M_MIB,
                               lc.RING_FORM_SENTINEL_STORE_CFG, 262144,
                               "write_through", "P", 785500001, 300000000000, 50)
        self.assertIn("--no-enable-multimodal", out2,
                      "text-only must still be the default on this path")

    def test_the_form_key_build_runs_end_to_end(self):
        """THE EXACT PATH THAT DIED: argv_p positionally, then the form key.

        This is the ratchet the train seat asked for -- it builds the argv the
        way launcher.py:10164 does, so a future middle-insert fails HERE rather
        than at the next boot.
        """
        # THE PRODUCTION SHAPE as it is now: seven positionals, everything
        # optional by keyword. `window_mib` carries the real per-rank string
        # "24,PP_0=96" -- the value whose type raised the ValueError when it
        # landed in `depth`, kept here so this test fails if it ever lands
        # there again.
        argv = lc.argv_p(
            "py", "/m", [1, 2, 3], lc.RING_FORM_SENTINEL_S_GB,
            lc.RING_FORM_SENTINEL_M_MIB, lc.RING_FORM_SENTINEL_STORE_CFG, [],
            p_bs=2, max_kv_per_request=262144, stage_ratio="39,13,12",
            attn_stage_ratio="10,3,3", write_policy="write_through",
            depth=lc.RING_FORM_SENTINEL_DEPTH, window_mib="24,PP_0=96",
            random_seed=785500001,
        )
        self.assertIn("--model-path", argv)
        key, form = rt.p_form_key(["py", "-m", "sglang.launch_server"] + argv)
        self.assertEqual(len(key), 12, f"form key not built: {key!r}")
        self.assertIn("--no-enable-multimodal", form,
                      "the text-only default must reach the form this path hashes")

    def test_no_production_callsite_passes_optionals_positionally(self):
        """THE AST PIN. Source text, parsed -- not a grep over it.

        Any `argv_p(...)` in the launcher with more positional arguments than
        the function has POSITIONAL parameters is re-binding an optional by
        position, which is the defect. Reported with the line number so the
        next one is found at the line that caused it.
        """
        import ast
        import inspect

        src = inspect.getsource(lc)
        n_pos = len([
            p for p in inspect.signature(lc.argv_p).parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD)])
        offenders = []
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "argv_p"
                    and len(node.args) > n_pos):
                offenders.append((node.lineno, len(node.args)))
        self.assertEqual(
            offenders, [],
            f"argv_p called with more than {n_pos} positional args at "
            f"(line, count) {offenders} -- an inserted parameter would "
            f"re-bind them silently")

    def test_both_production_callsites_are_keyword_form(self):
        import inspect

        src = inspect.getsource(lc)
        for marker in ("form_argv_p = xchg_form_argv(argv_p(",
                       "shipped_argv_p = argv_p("):
            with self.subTest(callsite=marker):
                i = src.index(marker)
                window = src[i:i + 900]
                for kw in ("window_mib=", "random_seed=", "census_interval=",
                           "depth="):
                    self.assertIn(kw, window,
                                  f"{marker} does not name {kw}")
