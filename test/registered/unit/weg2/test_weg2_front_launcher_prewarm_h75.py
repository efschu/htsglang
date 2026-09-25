"""H75 (x174): the launcher import that ``resolve_x_live`` does lazily costs 5-7 s.

On x172/x173/x174 it ran on the front's event loop inside the first completed
flip (``WEG2-HOST RATE-GAP`` 7.7/7.7/7.9 s), and on x174 the P leg dispatched
right after it went out on a pooled keep-alive connection that P had closed
during the freeze: ``Server disconnected``, the 97k needle lost. The front now
imports the module once at startup, in a worker thread. Pinned here:

1. the prewarm runs the import OFF the event loop: a ticker on the loop keeps
   ticking while a (stand-in) slow import runs, and the import runs in a thread
   other than the loop's;
2. a failing prewarm is not fatal and names the fallback;
3. ``Front.startup`` schedules the prewarm (the wiring, not just the method);
4. the real launcher module imports from a non-main thread in a fresh
   interpreter (nothing main-thread-only, e.g. ``signal.signal``, at import).
"""

import asyncio
import inspect
import subprocess
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.weg2 import front as F


class TestLauncherPrewarm(unittest.TestCase):
    def test_the_loop_keeps_ticking_while_the_import_runs(self):
        seen = {}

        def slow_import(name):
            seen["name"] = name
            seen["thread"] = threading.get_ident()
            time.sleep(0.6)
            return SimpleNamespace()

        async def run():
            ticks = []

            async def ticker():
                while True:
                    ticks.append(time.monotonic())
                    await asyncio.sleep(0.02)

            t = asyncio.create_task(ticker())
            with mock.patch.object(F.importlib, "import_module", side_effect=slow_import):
                await F.Front._prewarm_launcher_import(SimpleNamespace())
            t.cancel()
            return threading.get_ident(), ticks

        loop_thread, ticks = asyncio.run(run())
        self.assertEqual(seen["name"], "sglang.srt.weg2.launcher")
        self.assertNotEqual(seen["thread"], loop_thread)
        gaps = [b - a for a, b in zip(ticks, ticks[1:])]
        # 0.6 s of import at a 20 ms tick is ~30 ticks; on the loop it was one gap of 0.6 s.
        self.assertGreater(len(ticks), 10)
        self.assertLess(max(gaps), 0.3)

    def test_a_failing_prewarm_is_not_fatal_and_names_the_fallback(self):
        def boom(name):
            raise ValueError("signal only works in main thread")

        with mock.patch.object(F.importlib, "import_module", side_effect=boom), self.assertLogs(
            F.logger, level="WARNING"
        ) as cm:
            asyncio.run(F.Front._prewarm_launcher_import(SimpleNamespace()))
        self.assertTrue(any("launcher prewarm failed" in m and "resolve_x_live" in m for m in cm.output))

    def test_startup_schedules_the_prewarm(self):
        src = inspect.getsource(F.Front.startup)
        self.assertIn('app["launcher_prewarm"] = asyncio.create_task(self._prewarm_launcher_import())', src)

    def test_the_launcher_imports_from_a_worker_thread(self):
        code = (
            "import importlib, threading\n"
            "err = []\n"
            "def go():\n"
            "    try:\n"
            "        importlib.import_module('sglang.srt.weg2.launcher')\n"
            "    except BaseException as e:\n"
            "        err.append(repr(e))\n"
            "t = threading.Thread(target=go)\n"
            "t.start()\n"
            "t.join()\n"
            "print('THREAD-IMPORT', 'ERR' if err else 'OK', err)\n"
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        lines = [l for l in r.stdout.splitlines() if l.startswith("THREAD-IMPORT")]
        self.assertEqual(len(lines), 1, r.stderr[-2000:])
        self.assertTrue(lines[0].startswith("THREAD-IMPORT OK"), lines[0])


if __name__ == "__main__":
    unittest.main()
