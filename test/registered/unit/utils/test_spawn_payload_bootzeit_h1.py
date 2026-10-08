"""NF-Bootzeit H1: a rank start must not wait for the previous child's imports.

MEASURED (rc12z10, boot_weg2_dkrnfh91dprsavisbar1dauer09280831_*.P.log lines
32/35/51): the parent started PP0, PP1, PP2 8 s apart (D: 5 s apart); the 27B
boot of the same tree starts its three ranks in the same second. ``spawn``
writes the whole pickled process object into one pipe; a payload above the
pipe size holds ``proc.start()`` until the child has imported the parent's
main module and reads on.

The behaviour test reproduces exactly that: the payload's first element makes
the child import a module that sleeps 3 s (the stand-in for
``flliper.launch_server``), followed by 2 MB (a ServerArgs above the pipe
size). RED on fa548e7c46: ``proc.start()`` takes >= 3 s. GREEN with the fix:
the payload travels by file reference, ``start()`` returns at once, and the
child still receives the identical object.
"""

from __future__ import annotations

import inspect
import multiprocessing as mp
import os
import sys
import tempfile
import textwrap
import time
import unittest

try:  # the fix; absent on the base -> the old inline path (the RED case)
    from flliper.srt.utils.spawn_payload import FILE_PREFIX, by_reference
except ImportError:  # pragma: no cover - base
    FILE_PREFIX = "flliper-spawn-payload-"

    def by_reference(obj, **_kw):
        return obj


SLOW_IMPORT_S = 3.0


def subprocess_dead_pid() -> int:
    """The pid of a process that has exited (and been reaped)."""
    import subprocess

    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _child(payload, q):
    marker, blob = payload
    q.put((type(marker).__name__, len(blob), blob[:4], blob[-4:]))


class TestSpawnPayloadByReference(unittest.TestCase):
    def setUp(self):
        self.mod_dir = tempfile.mkdtemp(prefix="bootzeit_h1_")
        with open(os.path.join(self.mod_dir, "bootzeit_h1_slowmod.py"), "w") as fh:
            fh.write(textwrap.dedent(f"""
                import time
                time.sleep({SLOW_IMPORT_S})
                class SlowMarker:
                    pass
            """))
        sys.path.insert(0, self.mod_dir)
        self._old_pp = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = self.mod_dir + (
            os.pathsep + self._old_pp if self._old_pp else "")

    def tearDown(self):
        sys.path.remove(self.mod_dir)
        if self._old_pp is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = self._old_pp
        sys.modules.pop("bootzeit_h1_slowmod", None)

    def _payload(self):
        t0 = time.perf_counter()
        import bootzeit_h1_slowmod  # parent pays the import once, up front

        self.assertGreaterEqual(time.perf_counter() - t0, SLOW_IMPORT_S * 0.9)
        blob = b"B" + b"x" * (2 * 1024 * 1024) + b"E"  # far above a 64 KiB pipe
        return (bootzeit_h1_slowmod.SlowMarker(), blob)

    def test_start_does_not_wait_for_child_imports(self):
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        payload = self._payload()
        proc = ctx.Process(target=_child, args=(by_reference(payload), q))
        t0 = time.perf_counter()
        proc.start()
        start_s = time.perf_counter() - t0
        try:
            got = q.get(timeout=60)
        finally:
            proc.join(timeout=60)
        # the child received the identical object
        self.assertEqual(got, ("SlowMarker", len(payload[1]), payload[1][:4], payload[1][-4:]))
        self.assertEqual(proc.exitcode, 0)
        # ... and the parent did not sit in start() through the child's import
        self.assertLess(
            start_s, SLOW_IMPORT_S * 0.5,
            f"proc.start() took {start_s:.2f} s: the parent waited for the "
            f"child's {SLOW_IMPORT_S:.0f} s import (payload through the pipe)")

    def test_payload_file_is_consumed_by_the_child(self):
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        before = {f for f in os.listdir(tempfile.gettempdir()) if f.startswith(FILE_PREFIX)}
        proc = ctx.Process(target=_child, args=(by_reference(self._payload()), q))
        proc.start()
        q.get(timeout=60)
        proc.join(timeout=60)
        after = {f for f in os.listdir(tempfile.gettempdir()) if f.startswith(FILE_PREFIX)}
        self.assertEqual(after - before, set(), "the child must unlink its payload file")

    def test_stale_file_of_dead_writer_is_swept(self):
        """A parent SIGKILLed before its atexit leaves its unread payload
        behind; the next writer removes files of DEAD writers only."""
        from flliper.srt.utils import spawn_payload as sp

        d = tempfile.mkdtemp(prefix="bootzeit_h1_sweep_")
        dead = subprocess_dead_pid()
        stale = os.path.join(d, f"{FILE_PREFIX}{dead}-abc.pkl")
        live = os.path.join(d, f"{FILE_PREFIX}{os.getpid()}-def.pkl")
        for p in (stale, live):
            with open(p, "wb") as fh:
                fh.write(b"x")
        self.assertEqual(sp.sweep_stale(d), 1)
        self.assertFalse(os.path.exists(stale))
        self.assertTrue(os.path.exists(live))
        # a new payload file names its writer, so the sweep can judge it
        ref = sp.by_reference({"a": 1}, directory=d)
        self.assertTrue(os.path.basename(ref.path).startswith(f"{FILE_PREFIX}{os.getpid()}-"))

    def test_scheduler_launch_passes_server_args_by_reference(self):
        from flliper.srt.entrypoints import engine

        src = inspect.getsource(engine.Engine._launch_scheduler_processes)
        self.assertIn("spawn_payload.by_reference(server_args)", src)


if __name__ == "__main__":
    unittest.main()
