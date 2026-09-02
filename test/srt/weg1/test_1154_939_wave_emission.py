"""#1154: the '[#939 double-prefill]' line must reach the log once per cutover
wave, not once per 64 recording calls.

RED ON 8bf12cfd44: the emission gate was `_dpc_emitted % every == 0` with a
process-lifetime counter and `every` = the shared #904 knob (default 64), so
the FIRST line of a boot needed the 64th call. boot_855_weg1b2 (2026-09-02)
had two readmit waves, a handful of recording calls, and ZERO #939 lines --
and the acceptance read that absence as "half B is not wired".

Hermetic: no GPU, no scheduler, no server. The census module's recording API
is driven directly and the emitted lines are captured from a stub logger.
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"
    ),
)

from sglang.srt.mem_cache import producer_phase_census as ppc  # noqa: E402


class _Chunked:
    """Minimal stand-in for the scheduler `bind_chunk` reads."""

    chunked_prefill_size = 4096
    dynamic_chunked_prefill_size = 4096


class _Logger:
    def __init__(self):
        self.lines = []

    def warning(self, fmt, *args):
        self.lines.append(fmt % args if args else fmt)

    def error(self, fmt, *args):
        self.lines.append(fmt % args if args else fmt)


class WaveEmission(unittest.TestCase):
    def setUp(self):
        # Arm the shared #904 knob at its DEFAULT 64 -- the value the boot
        # ran with. Arming it at 1 would hide the very defect under test.
        self._saved = os.environ.get("SGLANG_MATCH_REFUSAL_CENSUS_EVERY")
        os.environ["SGLANG_MATCH_REFUSAL_CENSUS_EVERY"] = "64"
        if hasattr(ppc.census_armed, "cache_clear"):
            ppc.census_armed.cache_clear()
        ppc.reset_double_prefill_census()
        ppc._dpc_emitted = 0
        ppc._dpc_suppressed = 0

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("SGLANG_MATCH_REFUSAL_CENSUS_EVERY", None)
        else:
            os.environ["SGLANG_MATCH_REFUSAL_CENSUS_EVERY"] = self._saved
        ppc.reset_double_prefill_census()

    def _wave(self, log, n, tag):
        """One cutover wave: n re-admitted requests, then the cutover reset."""
        for i in range(n):
            ppc.note_double_prefill(f"{tag}{i:04x}", 68, 68, scheduler=_Chunked())
            ppc.emit_double_prefill(log)
        ppc.reset_double_prefill_census()

    def test_small_wave_emits_at_least_one_line(self):
        """THE weg1b2 SHAPE: two waves of one request each, knob at 64."""
        log = _Logger()
        self._wave(log, 1, "aa")
        self._wave(log, 1, "bb")
        lines = [x for x in log.lines if ppc.DOUBLE_PREFILL_LINE_PREFIX in x]
        self.assertGreaterEqual(
            len(lines),
            2,
            "one '[#939 double-prefill]' line per cutover wave is the floor; "
            f"got {len(lines)} from 2 waves. Lines: {log.lines}",
        )

    def test_rate_limit_still_holds_inside_one_wave(self):
        """The first-of-wave rule must not become 'emit every call'."""
        log = _Logger()
        self._wave(log, 40, "cc")
        lines = [x for x in log.lines if ppc.DOUBLE_PREFILL_LINE_PREFIX in x]
        self.assertEqual(
            len(lines),
            1,
            "40 recording calls in ONE wave with no breach must emit exactly "
            f"the first-of-wave line, got {len(lines)}",
        )
        self.assertIn("suppressed=", lines[0])


if __name__ == "__main__":
    unittest.main()
