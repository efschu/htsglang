# SPDX-License-Identifier: Apache-2.0
"""SWAP READINESS 0926 -- the GGUF trim's memory.reclaim stays file-only.

``memory.reclaim can only take clean page cache`` holds only while the cgroup
cannot swap. When swap is possible the request carries ``swappiness=0``; a
kernel rejecting that (EINVAL) falls back to the plain request ONLY when swap
is impossible, otherwise the OSError propagates (maybe_trim disables the trim).
"""
from __future__ import annotations

import errno
import os
import tempfile
import unittest
from unittest import mock

from sglang.srt.model_loader import gguf_shards as gs


def _setup(tmp, swap_max=None, swap_total_kb=0):
    open(os.path.join(tmp, "memory.reclaim"), "w").close()
    if swap_max is not None:
        with open(os.path.join(tmp, "memory.swap.max"), "w") as f:
            f.write(swap_max + "\n")
    mi = os.path.join(tmp, "meminfo")
    with open(mi, "w") as f:
        f.write(f"MemTotal: 1 kB\nSwapTotal: {swap_total_kb} kB\n")
    return mi


def _read(tmp):
    with open(os.path.join(tmp, "memory.reclaim")) as f:
        return f.read()


class _EinvalOnArg:
    """open() stand-in: a write carrying ``swappiness=`` raises EINVAL."""

    def __init__(self, real_open):
        self.real_open = real_open

    def __call__(self, path, mode="r", *a, **k):
        fh = self.real_open(path, mode, *a, **k)
        if str(path).endswith("memory.reclaim") and "w" in mode:
            real_write = fh.write

            def write(s):
                if "swappiness=" in s:
                    raise OSError(errno.EINVAL, "Invalid argument")
                return real_write(s)

            fh.write = write
        return fh


class ReclaimSwapGuard(unittest.TestCase):
    def setUp(self):
        env = {k: v for k, v in os.environ.items() if k != gs._SWAP_AWARE_ENV}
        self._p = mock.patch.dict(os.environ, env, clear=True)
        self._p.start()
        self.addCleanup(self._p.stop)

    def test_swapless_is_the_plain_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            mi = _setup(tmp, swap_max="max", swap_total_kb=0)  # CT999 shape
            self.assertEqual(gs._write_reclaim(4096, tmp, mi), "4096")
            self.assertEqual(_read(tmp), "4096")

    def test_docker_swap_max_zero_is_swapless_even_with_host_swaptotal(self):
        with tempfile.TemporaryDirectory() as tmp:
            mi = _setup(tmp, swap_max="0", swap_total_kb=32 << 20)
            self.assertFalse(gs._swap_possible(tmp, mi))
            self.assertEqual(gs._write_reclaim(4096, tmp, mi), "4096")

    def test_swap_possible_carries_swappiness_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            mi = _setup(tmp, swap_max="max", swap_total_kb=32 << 20)
            self.assertTrue(gs._swap_possible(tmp, mi))
            self.assertEqual(gs._write_reclaim(4096, tmp, mi), "4096 swappiness=0")
            self.assertEqual(_read(tmp), "4096 swappiness=0")

    def test_switch_forces_the_argument_when_swapless(self):
        with tempfile.TemporaryDirectory() as tmp:
            mi = _setup(tmp, swap_max="max", swap_total_kb=0)
            with mock.patch.dict(os.environ, {gs._SWAP_AWARE_ENV: "1"}):
                self.assertEqual(gs._write_reclaim(7, tmp, mi), "7 swappiness=0")

    def test_einval_falls_back_only_when_swap_impossible(self):
        import builtins

        with tempfile.TemporaryDirectory() as tmp:
            mi = _setup(tmp, swap_max="max", swap_total_kb=0)
            with mock.patch.dict(os.environ, {gs._SWAP_AWARE_ENV: "1"}), \
                    mock.patch.object(builtins, "open", _EinvalOnArg(builtins.open)):
                self.assertEqual(gs._write_reclaim(7, tmp, mi), "7")
            self.assertEqual(_read(tmp), "7")
        with tempfile.TemporaryDirectory() as tmp:
            mi = _setup(tmp, swap_max="max", swap_total_kb=32 << 20)
            with mock.patch.object(builtins, "open", _EinvalOnArg(builtins.open)):
                with self.assertRaises(OSError) as cm:
                    gs._write_reclaim(7, tmp, mi)
            self.assertEqual(cm.exception.errno, errno.EINVAL)
            self.assertEqual(_read(tmp), "")  # nothing was reclaimed


if __name__ == "__main__":
    unittest.main()
