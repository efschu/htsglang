# SPDX-License-Identifier: Apache-2.0
"""BOOTZEIT 4 (29.09.): the per-layer host reclaim of the expert presplit.

Metal (z30w-park, NF D, LOAD-PROFILE of the loader thread): expert_offload.py
``gc.collect()`` 7.8-9.6 % and ``malloc_trim(0)`` 1.3-4.0 %, under the GIL,
48 times per D rank. The reclaim is now one clocked function
(``presplit_host_reclaim``) behind ``SGLANG_OPT_LOAD_PRESPLIT_GC``; the
``[ct-stream-presplit]`` line prints ``reclaim gc= found= trim=``.

The call-edge tests read the real modules (AST), not a stand-in: the class of
bug they guard is a helper whose own test is green while the caller never
reaches it, or reaches it with arguments the log line cannot format.
"""

import ast
import inspect
import re
import textwrap
import unittest
from unittest import mock

from sglang.srt.environ import PresplitGcMode, envs
from sglang.srt.layers.moe import expert_offload as eo
from sglang.srt.layers.moe.fused_moe_triton import layer as fl


class TestPresplitHostReclaim(unittest.TestCase):
    def setUp(self):
        self._saved = eo.expert_store_clock()

    def tearDown(self):
        eo._STORE_CLOCK.update(self._saved)

    def test_default_is_full(self):
        self.assertEqual(envs.SGLANG_OPT_LOAD_PRESPLIT_GC.get(), PresplitGcMode.FULL)

    def test_full_collects_and_counts_what_it_found(self):
        before = eo.expert_store_clock()
        with envs.SGLANG_OPT_LOAD_PRESPLIT_GC.override(PresplitGcMode.FULL), \
                mock.patch("gc.collect", return_value=7) as collect:
            eo.presplit_host_reclaim()
        collect.assert_called_once_with()
        after = eo.expert_store_clock()
        self.assertEqual(after["gc_found"] - before["gc_found"], 7)
        self.assertGreaterEqual(after["gc_s"], before["gc_s"])
        self.assertGreaterEqual(after["trim_s"], before["trim_s"])

    def test_trim_skips_the_collect_but_still_trims(self):
        before = eo.expert_store_clock()
        with envs.SGLANG_OPT_LOAD_PRESPLIT_GC.override(PresplitGcMode.TRIM), \
                mock.patch("gc.collect") as collect, \
                mock.patch("ctypes.CDLL") as cdll:
            eo.presplit_host_reclaim()
        collect.assert_not_called()
        cdll.return_value.malloc_trim.assert_called_once_with(0)
        after = eo.expert_store_clock()
        self.assertEqual(after["gc_found"], before["gc_found"])
        self.assertEqual(after["gc_s"], before["gc_s"])


class TestPresplitCallEdges(unittest.TestCase):
    def test_presplit_reaches_the_reclaim_and_collects_nowhere_else(self):
        src = textwrap.dedent(inspect.getsource(eo.presplit_expert_offload_after_repack))
        fn = ast.parse(src).body[0]
        calls = [
            n.func for n in ast.walk(fn) if isinstance(n, ast.Call)
        ]
        names = {c.id for c in calls if isinstance(c, ast.Name)}
        attrs = {c.attr for c in calls if isinstance(c, ast.Attribute)}
        self.assertIn("presplit_host_reclaim", names)
        self.assertNotIn("collect", attrs)
        self.assertNotIn("malloc_trim", attrs)

    def test_presplit_log_line_formats_every_argument(self):
        # The [ct-stream-presplit] logger.info: placeholders == arguments.
        src = textwrap.dedent(inspect.getsource(fl.FusedMoE._ct_stream_presplit_now))
        fn = ast.parse(src).body[0]
        hits = []
        for n in ast.walk(fn):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "info" and n.args):
                continue
            fmt = n.args[0]
            if isinstance(fmt, ast.Constant) and isinstance(fmt.value, str) \
                    and fmt.value.startswith("[ct-stream-presplit] layer") \
                    and "split h2d=" in fmt.value:
                hits.append((fmt.value, len(n.args) - 1))
        self.assertEqual(len(hits), 1)
        fmt, nargs = hits[0]
        self.assertIn("reclaim gc=", fmt)
        self.assertEqual(len(re.findall(r"%[-+0-9.]*[sdf]", fmt)), nargs)


if __name__ == "__main__":
    unittest.main()
