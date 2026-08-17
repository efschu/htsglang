"""#447 candidate F: the SWA window is checked against the checkpoint.

``SWA_WINDOW = 128`` was a bare constant. Every DSV4 checkpoint on this box
declares ``"sliding_window": 128`` -- 0731-GGUF/UD-Q3_K_XL, the DSpark head and
its filtered sibling were all read before this was written -- so the constant
was right by LUCK. ANALYSE_447 §4 candidate F called the gain "zero today,
correctness-by-luck removed", and that is exactly what this is: no computed
value changes, a divergent checkpoint stops being silent.

It is a CHECK, not a plumb-through, and that is the design decision worth
stating: the window is not a free parameter. The C128 compressor pools a page
with ``tl.static_range(COMPRESS_RATIO)`` and the backend asserts
``swa_page_size == SWA_WINDOW``, so threading a different number through would
compress against the wrong span rather than honour it.
"""

import inspect
import unittest

from sglang.srt.layers.attention.deepseek_v4_backend import (
    SWA_WINDOW,
    verify_swa_window,
)
from sglang.test.test_utils import CustomTestCase


class _Config:
    def __init__(self, declared):
        self.sliding_window_size = declared


class TestTheDeclaredWindowIsChecked(CustomTestCase):
    def test_the_matching_checkpoint_is_accepted(self):
        """The real case: every DSV4 config here declares 128."""
        self.assertEqual(verify_swa_window(_Config(128)), SWA_WINDOW)

    def test_an_undeclared_window_keeps_todays_behaviour(self):
        """Most configs in this family carry no key at all, and refusing them
        would be a regression in the name of tidiness."""
        self.assertEqual(verify_swa_window(_Config(None)), SWA_WINDOW)

    def test_a_config_object_without_the_attribute_at_all_is_accepted(self):
        self.assertEqual(verify_swa_window(object()), SWA_WINDOW)

    def test_a_divergent_checkpoint_is_refused_with_both_numbers(self):
        with self.assertRaises(ValueError) as cm:
            verify_swa_window(_Config(256))
        message = str(cm.exception)
        self.assertIn("256", message)
        self.assertIn(str(SWA_WINDOW), message)
        self.assertIn("compress", message.lower())

    def test_a_string_window_is_compared_as_a_number(self):
        """Config values arrive from JSON; a str 128 is still 128."""
        self.assertEqual(verify_swa_window(_Config("128")), SWA_WINDOW)


class TestTheCheckIsActuallyWired(CustomTestCase):
    """The anti-orphan pin. Constructing the real backend needs a ModelRunner,
    a device and pools, so the call site is pinned by NAME in the source --
    the house solution from #698, which survives either spelling and still
    fails if the call is deleted."""

    def test_the_backend_init_calls_the_check(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        source = inspect.getsource(DeepseekV4AttnBackend.__init__)
        self.assertIn(
            "verify_swa_window",
            source,
            "DeepseekV4AttnBackend.__init__ no longer checks the SWA window "
            "against the checkpoint; the constant is back to being right by luck",
        )

    def test_the_backend_no_longer_hardcodes_the_page_size(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        source = inspect.getsource(DeepseekV4AttnBackend.__init__)
        self.assertNotIn(
            "self.swa_page_size = 128",
            source,
            "swa_page_size is hardcoded again, so the check cannot bind it",
        )


if __name__ == "__main__":
    unittest.main()
