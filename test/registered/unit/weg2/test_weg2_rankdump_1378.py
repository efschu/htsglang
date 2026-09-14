# SPDX-License-Identifier: Apache-2.0
"""#1378 xsn43 (TEIL 1): the W68/W29 family's refusals carry their own
diagnosis -- all threads + the rank-local leg state, per rank, named.

THE MEASURED GAP: xsn42's deposit hung at GPU 0 % and the boot died without
anyone knowing what the deposit rank was doing (the operator saw no load,
the logs showed only silence). The sglang watchdog's own thread dump
(utils/watchdog.py:176, seen on weg2xsn38's P log) named the LanePermit
deadlock only because it fired LATER than the leg's refusal. The dump at
the refusal closes that gap: every future death of this family carries its
diagnosis with it.

PINNED:
* the helper writes a named file with the reason, the tag, the rank and
  faulthandler's all-thread traceback;
* the W68 raise site calls it BEFORE the raise (the mutant: the dump call
  removed from the site -> the pin dies);
* the launcher publishes SGLANG_WEG2_RANKDUMP_DIR to both groups.
"""

from __future__ import annotations

import inspect
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange_bounce as bx  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402


class TheDumpHelperWritesNamedFiles(unittest.TestCase):
    def test_the_helper_writes_reason_tag_rank_and_frames(self):
        with tempfile.TemporaryDirectory(prefix="weg2-rankdump-") as d:
            old = os.environ.get("SGLANG_WEG2_RANKDUMP_DIR")
            os.environ["SGLANG_WEG2_RANKDUMP_DIR"] = d
            try:
                path = bx.dump_rank_stacks(
                    "W68-not-posted", tag="xsn43/authoritative",
                    rank=0, extra="slot=0 seq=0 unit=('weights_0', 0)")
            finally:
                if old is None:
                    os.environ.pop("SGLANG_WEG2_RANKDUMP_DIR", None)
                else:
                    os.environ["SGLANG_WEG2_RANKDUMP_DIR"] = old
            self.assertTrue(path and os.path.isfile(path), path)
            text = open(path).read()
            self.assertIn("reason=W68-not-posted", text)
            self.assertIn("tag=xsn43/authoritative", text)
            self.assertIn("rank=0", text)
            self.assertIn("extra=slot=0 seq=0", text)
            self.assertIn("Current thread", text,
                          "faulthandler's traceback must name the frames")

    def test_no_env_dir_is_a_silent_skip(self):
        """A rank without an evidence dir has no reader for the dump -- the
        helper skips instead of failing the leg."""
        old = os.environ.get("SGLANG_WEG2_RANKDUMP_DIR")
        os.environ.pop("SGLANG_WEG2_RANKDUMP_DIR", None)
        try:
            self.assertEqual(bx.dump_rank_stacks("r", "t", 0), "")
        finally:
            if old is not None:
                os.environ["SGLANG_WEG2_RANKDUMP_DIR"] = old


class TheW68SiteDumpsBeforeTheRaise(unittest.TestCase):
    def test_the_W68_site_calls_the_dump_first(self):
        """The mutant (danger direction): the dump call removed from the W68
        site -- the diagnosis is lost with the boot again. The source pin
        fails on that removal."""
        src = inspect.getsource(bx.run_bounce_leg)
        raise_at = src.index('raise Weg2XchgBouncePhaseUnordered(\n'
                             '                                f"W68 '
                             'Weg2XchgPlanDisagree: slot="')
        region = src[max(0, raise_at - 600):raise_at]
        self.assertIn("dump_rank_stacks(", region,
                      "the W68 refusal must dump the rank stacks BEFORE the "
                      "raise -- the diagnosis dies with the boot otherwise")


class TheLauncherPublishesTheDumpDir(unittest.TestCase):
    def test_build_env_publishes_the_rankdump_dir(self):
        env = L.build_env(tree="/tmp/t", venv="/tmp/v", cvd="",
                          store_dir="/tmp/store", debug_hold=False,
                          tag="weg2xsn43")
        self.assertEqual(env["SGLANG_WEG2_RANKDUMP_DIR"], L.EVIDENCE_DIR,
                         "the ranks must write their dumps into the boot's "
                         "evidence dir")


if __name__ == "__main__":
    unittest.main()
