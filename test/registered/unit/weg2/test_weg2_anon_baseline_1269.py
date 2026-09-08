# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#1269 fix 4 follow-up: the pre-boot anon baseline reaches the front.

W22's `sglang=` / `foreign=` split subtracts a PRE-BOOT cgroup anon reading --
subtractable precisely because it is the same quantity at an earlier time,
before any group process exists. `host_ledger` replaced the old
`sum(RssAnon)` split with it (that one printed `sglang=61.48 foreign=-30.91`,
because cgroup anon counts a shared page ONCE while RssAnon counts it once per
mapper over 111 pids).

THE MEASUREMENT WAS ALREADY BEING TAKEN AND THROWN AWAY. `choose_host_ledger`
returns the preflight cgroup snapshot as `cg`, `main` already binds it, and
`cg["anon"]` went nowhere -- so boot weg2sb5c's split had to be reconstructed
by hand afterwards. This pins the wire.

ONE READ, CARRIED. The baseline must be the PREFLIGHT's own reading, never a
second `read_cgroup()` at front-launch time: by then the groups exist and the
number is no longer a baseline. Two reads are two quantities.
"""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace

from sglang.srt.weg2 import launcher as L
from sglang.test.test_utils import CustomTestCase

NS = SimpleNamespace(tag="t", fairness_w_s=45.0, drain_deadline_s=90.0,
                     min_dwell_ms=None, d_admit_max_tokens=None)


def argv(**kw):
    return L.front_argv_for("py", "/s", 1, 2, {}, [], NS, 0, 0, 8, 8, 1, 1, "D", **kw)


class TheFrontReceivesTheMeasuredValue(CustomTestCase):
    """RED-FIRST: before the wire, no argv carried the flag at all."""

    def test_the_flag_carries_the_measured_bytes(self):
        a = argv(anon_preboot_bytes=10_790_000_000)
        self.assertIn("--anon-preboot-bytes", a)
        self.assertEqual(a[a.index("--anon-preboot-bytes") + 1], "10790000000")

    def test_an_unreadable_baseline_ships_nothing(self):
        """0 means unset: the front then prints NO split rather than an
        invented one. `host_ledger` is explicit that -1/absent must never be
        read as zero."""
        self.assertNotIn("--anon-preboot-bytes", argv(anon_preboot_bytes=0))
        self.assertNotIn("--anon-preboot-bytes", argv())

    def test_the_front_parses_it_into_the_field_the_ledger_reads(self):
        from sglang.srt.weg2 import front as F

        src = inspect.getsource(F)
        self.assertIn('"--anon-preboot-bytes"', src)
        self.assertIn("front._anon_preboot_bytes = int(args.anon_preboot_bytes)", src)
        # and the guard's own consumer reads that field
        self.assertIn("anon_preboot_bytes=self._anon_preboot_bytes", src)


class TheBaselineIsThePreflightsOwnReading(CustomTestCase):
    def test_main_takes_it_from_the_ledger_snapshot_not_a_second_read(self):
        src = inspect.getsource(L.main)
        self.assertIn('anon_preboot_bytes = int(cg.get("anon") or 0)', src)
        # exactly one read_cgroup in main -- the preflight's, via the ledger
        tree = ast.parse(textwrap.dedent(src))
        reads = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "read_cgroup"]
        self.assertEqual(reads, [], "main must not re-read the cgroup itself")

    def test_the_capture_precedes_the_front_launch(self):
        src = inspect.getsource(L.main)
        self.assertLess(src.index("anon_preboot_bytes = int("),
                        src.index("front_argv = front_argv_for("),
                        "the baseline must be captured before the front is built")

    def test_the_launcher_names_the_baseline_in_one_line(self):
        src = inspect.getsource(L.main)
        self.assertIn("WEG2-LAUNCH ANON-BASELINE", src)
        self.assertIn("preboot_anon=", src)

    def test_both_front_argv_call_sites_pass_it(self):
        """The dry-run print and the real launch must agree, or --dry-run stops
        showing what a real boot runs."""
        src = inspect.getsource(L.main)
        self.assertEqual(src.count("anon_preboot_bytes=anon_preboot_bytes"), 2)


if __name__ == "__main__":
    unittest.main()
