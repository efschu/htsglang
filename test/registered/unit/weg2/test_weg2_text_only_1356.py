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
        self.assertIn("--weg2-vision resident", src)
        self.assertIn("status=501", src)


if __name__ == "__main__":
    unittest.main()
