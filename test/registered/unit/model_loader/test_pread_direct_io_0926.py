# SPDX-License-Identifier: Apache-2.0
"""27B-ODIRECT 0926 -- --weight-loader-direct-io on the pread path and on the
multi-thread iterator's whole-file branch.

Before: ``buffered_multi_thread_safetensors_weights_iterator`` (the DEFAULT
loader, enable_multithread_load=True) and ``pread_safetensors_file`` accepted
``direct_io`` and ignored it; only the single-thread iterator honoured it.
Pinned here: bytes identical to the buffered read for every dtype/offset shape
(odd offsets, a tail not a multiple of 4096, tensors larger than the bounce),
O_DIRECT really requested, EINVAL on open falls back to buffered with a
warning, and the default (direct_io=False) never opens O_DIRECT.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import safetensors.torch
import torch

from sglang.srt.model_loader import weight_utils as wu

O_DIRECT = getattr(os, "O_DIRECT", 0o40000)


def _tensors():
    g = torch.Generator().manual_seed(926)
    return {
        "a.u8": torch.randint(0, 255, (4097,), dtype=torch.uint8, generator=g),
        "b.bf16": torch.randn(333, 7, generator=g).to(torch.bfloat16),
        "c.f32": torch.randn(5, 3, generator=g),
        "d.i8": torch.randint(-128, 127, (1,), dtype=torch.int8, generator=g),
        # larger than the 16 MiB bounce -> several aligned reads
        "e.f16": torch.randn(9 << 20, generator=g).to(torch.float16),
        "f.i32": torch.randint(0, 1 << 30, (4099,), dtype=torch.int32, generator=g),
    }


class _Base(unittest.TestCase):
    def setUp(self):
        # on the worktree's filesystem (ZFS here, XFS on the host) -- tmpfs
        # would refuse O_DIRECT and only exercise the fallback
        self.dir = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(__file__)))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir, True))
        self.ref = _tensors()
        self.path = os.path.join(self.dir, "m.safetensors")
        safetensors.torch.save_file(self.ref, self.path)
        self.assertNotEqual(os.path.getsize(self.path) % 4096, 0)
        wu._DIRECT_FALLBACK_WARNED.clear()
        try:
            os.close(os.open(self.path, os.O_RDONLY | O_DIRECT))
            self.direct_ok = True
        except OSError:
            self.direct_ok = False  # e.g. tmpfs: only the fallback is exercised

    def _same(self, got):
        self.assertEqual(sorted(got), sorted(self.ref))
        for k, v in self.ref.items():
            self.assertEqual(got[k].dtype, v.dtype, k)
            self.assertEqual(tuple(got[k].shape), tuple(v.shape), k)
            self.assertTrue(torch.equal(got[k], v), k)


class PreadDirect(_Base):
    def _flags_seen(self, **kw):
        seen = []
        real = os.open

        def spy(path, flags, *a):
            seen.append(flags)
            return real(path, flags, *a)

        with mock.patch.object(wu.os, "open", spy):
            got = wu.pread_safetensors_file(self.path, **kw)
        return got, seen

    def test_serial_direct_is_byte_identical_and_uses_o_direct(self):
        with mock.patch.dict(os.environ, {"SGLANG_LOAD_KEY_WORKERS": "1"}):
            got, seen = self._flags_seen(direct_io=True)
        self._same(got)
        self.assertTrue(any(f & O_DIRECT for f in seen))
        if self.direct_ok:  # the read really went O_DIRECT, not the fallback
            self.assertFalse(wu._DIRECT_FALLBACK_WARNED)

    def test_key_parallel_direct_is_byte_identical(self):
        with mock.patch.dict(os.environ, {"SGLANG_LOAD_KEY_WORKERS": "3"}):
            got, seen = self._flags_seen(direct_io=True)
        self._same(got)
        self.assertEqual(sum(1 for f in seen if f & O_DIRECT), 3)

    def test_default_never_opens_o_direct(self):
        for kw in ("1", "3"):
            with mock.patch.dict(os.environ, {"SGLANG_LOAD_KEY_WORKERS": kw}):
                got, seen = self._flags_seen()
            self._same(got)
            self.assertFalse(any(f & O_DIRECT for f in seen))

    def test_post_load_and_should_load_still_apply(self):
        with mock.patch.dict(os.environ, {"SGLANG_LOAD_KEY_WORKERS": "1"}):
            got = wu.pread_safetensors_file(
                self.path,
                should_load=lambda n: "meta" if n == "c.f32" else n != "d.i8",
                post_load=lambda n, t: t.clone(),
                direct_io=True,
            )
        self.assertNotIn("d.i8", got)
        self.assertEqual(got["c.f32"].device.type, "meta")
        self.assertTrue(torch.equal(got["e.f16"], self.ref["e.f16"]))

    def test_einval_on_open_falls_back_buffered_and_warns_once(self):
        real = os.open

        def refuse(path, flags, *a):
            if flags & O_DIRECT:
                raise OSError(22, "Invalid argument")
            return real(path, flags, *a)

        with mock.patch.object(wu.os, "open", refuse), \
                mock.patch.dict(os.environ, {"SGLANG_LOAD_KEY_WORKERS": "3"}), \
                self.assertLogs(wu.logger, "WARNING") as cm:
            got = wu.pread_safetensors_file(self.path, direct_io=True)
        self._same(got)
        self.assertEqual(sum("O_DIRECT refused" in m for m in cm.output), 1)


class MultiThreadIterator(_Base):
    def test_pread_branch_forwards_direct_io(self):
        # 27B-ODIRECT-STREAM 0927: pread + direct_io streams by default (the
        # per-file form held (max_workers + 1) whole shards -> b23 hit its cap);
        # the stream carries direct_io. SGLANG_WEIGHT_LOADER_PREAD_STREAM=0 keeps
        # the per-file form, which forwards it as before.
        with mock.patch.object(wu, "pread_safetensors_stream",
                               wraps=wu.pread_safetensors_stream) as spy:
            os.environ.pop(wu.STREAM_ENV, None)
            got = dict(wu.buffered_multi_thread_safetensors_weights_iterator(
                [self.path], max_workers=2, pread=True, direct_io=True))
        self._same(got)
        self.assertTrue(spy.call_args.kwargs.get("direct_io"))
        with mock.patch.dict(os.environ, {wu.STREAM_ENV: "0"}), \
                mock.patch.object(wu, "pread_safetensors_file",
                                  wraps=wu.pread_safetensors_file) as spy2:
            got = dict(wu.buffered_multi_thread_safetensors_weights_iterator(
                [self.path], max_workers=2, pread=True, direct_io=True))
        self._same(got)
        self.assertTrue(spy2.call_args.kwargs.get("direct_io"))

    def test_disable_mmap_branch_honours_direct_io(self):
        with mock.patch.object(wu, "read_file_direct",
                               wraps=wu.read_file_direct) as spy:
            got = dict(wu.buffered_multi_thread_safetensors_weights_iterator(
                [self.path], max_workers=2, disable_mmap=True, direct_io=True))
        self._same(got)
        self.assertEqual(spy.call_count, 1)

    def test_disable_mmap_without_direct_io_is_unchanged(self):
        with mock.patch.object(wu, "read_file_direct") as spy:
            got = dict(wu.buffered_multi_thread_safetensors_weights_iterator(
                [self.path], max_workers=2, disable_mmap=True))
        self._same(got)
        spy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
