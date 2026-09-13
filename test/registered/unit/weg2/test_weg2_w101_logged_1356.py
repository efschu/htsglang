# SPDX-License-Identifier: Apache-2.0
"""#1356 -- W101 is LOGGED, not only returned to the caller.

The refusal existed solely in the 501 response body: `grep W101 front.log` read
0 even when it had fired cleanly, so nobody reading the boot afterwards could
tell a refusal that HAPPENED from one that never came up. Measured by the train
seat on d8424a6de1: logger calls in front.py = 83, logger calls around
front.py:2575-2587 = 0.

It nearly cost the xsn29 vision probe a FALSE RED -- the planned acceptance
grepped for the log line and would have found nothing.

THE EMITTER RULE, general rather than local to this line: **every acceptance
marker is a logger emitter.** A W-code only a client sees is not a marker,
because the record is what the next reader has. Every other W-code in this file
is visible; W101 was the exception, and it sat on the PRODUCT rather than on a
test -- the same absence class as reading a null from a bad grep and calling it
"the instrument does not exist".

The comment at that site already named the hazard it was silent about: "WRONG
OUTPUT, not an exception, the one shape that must never be reached silently."
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.test_utils import CustomTestCase


class TheRefusalReachesTheRecord(CustomTestCase):
    def test_the_site_emits_a_logger_call(self):
        import inspect

        from sglang.srt.weg2 import front as fr

        src = inspect.getsource(fr.Front.handle_generate)
        i = src.index("_img = _image_parts(payload)")
        j = src.index("status=501", i)
        window = src[i:j]
        self.assertIn("logger.warning", window,
                      "W101 must be logged, not only returned -- a refusal only "
                      "the caller sees is invisible to every later reader")

    def test_the_line_carries_the_code_the_rid_and_the_count(self):
        import inspect

        from sglang.srt.weg2 import front as fr

        src = inspect.getsource(fr.Front.handle_generate)
        i = src.index("logger.warning")
        window = src[i:i + 400]
        for token in ("W101 Weg2VisionRefused", "rid=", "image_parts="):
            self.assertIn(token, window, f"the log line does not carry {token}")

    def test_it_is_greppable_as_a_marker(self):
        """The acceptance reads the LOG, so the code must be a literal there."""
        import inspect

        from sglang.srt.weg2 import front as fr

        src = inspect.getsource(fr.Front.handle_generate)
        i = src.index("logger.warning")
        self.assertIn('"W101 Weg2VisionRefused', src[i:i + 120],
                      "the marker must be a leading literal in the format "
                      "string, not assembled at runtime")

    def test_every_w_code_in_this_file_is_logged_somewhere(self):
        """The general rule, asserted rather than asserted-about-W101-only.

        If another W-code in front.py has no logger emitter, it has the same
        defect and this test says so by name instead of leaving it to the next
        boot's acceptance.
        """
        import re

        from sglang.srt.weg2 import front as fr

        src = open(fr.__file__).read()
        code_lines = [ln for ln in src.splitlines()
                      if not ln.strip().startswith("#")]
        body = "\n".join(code_lines)
        codes = set(re.findall(r"W(\d{2,3}) Weg2[A-Za-z]+", body))
        # A WINDOW AFTER THE CALL, not the same line: `logger.warning(` and
        # its format string sit on different lines, so a per-line filter reads
        # a correctly logged code as unlogged. This test found that on its own
        # first run -- the scan has to match how the code is WRITTEN, not how
        # it would be convenient to grep.
        logged = set()
        for m in re.finditer(r"(logger\.\w+\(|do_stop\()", body):
            for c in re.findall(r"W(\d{2,3}) Weg2[A-Za-z]+",
                                body[m.start():m.start() + 600]):
                logged.add(c)
        self.assertIn("101", logged, "W101 is not on a logging path")
        missing = sorted(codes - logged)
        self.assertEqual(
            missing, [],
            f"W-codes raised in front.py with no logger/do_stop emitter: "
            f"{missing} -- each is invisible in the boot record")


if __name__ == "__main__":
    unittest.main()
