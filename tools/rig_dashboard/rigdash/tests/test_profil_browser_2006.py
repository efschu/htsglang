"""Browser-Gegenprobe 2006 (05.10.): Text-Kuerzung der Force-Liste und die CSS-Regeln, die der Browser (Playwright, 390 px) gebraucht hat."""

import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import profil as P  # noqa: E402

STATIC = os.path.join(os.path.dirname(HERE), "static")


class ClipText(unittest.TestCase):
    def test_leading_code_is_dropped(self):
        self.assertEqual(P._clip_text("HW-COUNT", "HW-COUNT: 1 card(s) visible"), "1 card(s) visible")
        self.assertEqual(P._clip_text("HW-COUNT", "andere Form: HW-COUNT"), "andere Form: HW-COUNT")

    def test_long_text_ends_on_a_word_with_ellipsis(self):
        t = "HW-X: " + " ".join("wort%03d" % i for i in range(200))
        c = P._clip_text("HW-X", t)
        self.assertLessEqual(len(c), P.MAX_CODE_TEXT)
        self.assertTrue(c.endswith("…"))
        self.assertRegex(c[:-1], r"wort\d{3}$")           # nicht mitten im Wort

    def test_short_and_empty(self):
        self.assertEqual(P._clip_text("A", "kurz"), "kurz")
        self.assertEqual(P._clip_text("A", None), "")

    def test_hint_uses_it(self):
        reg = [{"code": "HW-COUNT", "forcebar": True, "wired": True, "title": "t", "force_scope": "s"}]
        h = P.force_hint({"rejections": [{"code": "HW-COUNT", "text": "HW-COUNT: eine Karte"}]}, reg)
        self.assertEqual(h["force_codes"][0]["text"], "eine Karte")


class Css(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
        cls.js = open(os.path.join(STATIC, "profil.js"), encoding="utf-8").read()

    def test_phone_stacks_the_rows(self):
        self.assertRegex(self.html, r"@media \(max-width: 640px\)[^}]*\.pf-t thead \{ display: none; \}")

    def test_profile_select_cannot_widen_the_page(self):
        self.assertIn("#pf-pick { max-width: calc(100vw - 90px); }", self.html)

    def test_run_block_wraps_despite_kp_pre(self):
        self.assertIn("#pf-root .pf-run { white-space: pre-wrap;", self.html)

    def test_badges_are_not_dimmed_by_opacity(self):
        self.assertNotIn("opacity: .85", self.html.split(".pf-dep i {")[1].split("}")[0])
        self.assertNotIn("opacity: .55", self.html.split(".pf-dep-off {")[1].split("}")[0])

    def test_chip_message_sits_in_the_row(self):
        self.assertIn("pf-cmsg", self.html)
        self.assertIn("pf-cmsg", self.js)

    def test_chip_parts_are_separated_by_spaces(self):
        self.assertIn("<b>${esc(d.to)}</b> ${wert} ${badge}</span>", self.js)


if __name__ == "__main__":
    unittest.main()
