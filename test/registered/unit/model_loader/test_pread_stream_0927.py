# SPDX-License-Identifier: Apache-2.0
"""27B-ODIRECT-STREAM 0927: the O_DIRECT pread path streams TENSORS with a bounded
in-flight window instead of whole shards.

Boot b23 (27b-park-odirect-draft, rc12g-flat, 27.09.): --weight-loader-disable-mmap
--weight-loader-direct-io + FLLIPER_WEIGHT_LOADER_PREAD=1 hit the 76g container cap
(oom_kill 3, anon+shmem 72,5 GiB) before serving -- the multi-thread iterator held
(max_workers + 1) whole shards as anon (pread returned one dict per file). These tests use
mock safetensors files: same bytes, same order as the per-file path, and a window that never
exceeds max(budget, largest tensor).
"""

import os
import tempfile
import threading
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from flliper.srt.model_loader import weight_utils as W

MIB = 1 << 20


def _files(root, n_files=3, per_file=12, elems=(MIB // 4)):
    """n_files shards of per_file float32 tensors (1 MiB each by default), plus
    one odd-sized and one bf16 tensor per file; names deliberately NOT in offset
    order so name order and file order differ."""
    paths = []
    g = torch.Generator().manual_seed(7)
    for f in range(n_files):
        d = {}
        for i in reversed(range(per_file)):
            d[f"model.layers.{f}.w{i:02d}"] = torch.randn(elems, generator=g)
        d[f"model.layers.{f}.odd"] = torch.randn(1234, generator=g)
        d[f"model.layers.{f}.bf16"] = torch.randn(333, generator=g).to(torch.bfloat16)
        p = os.path.join(root, f"model-{f:05d}-of-{n_files:05d}.safetensors")
        save_file(d, p)
        paths.append(p)
    return paths


def _per_file_reference(paths, should_load=None):
    out = []
    for p in paths:
        r = W.pread_safetensors_file(p, should_load)
        for k in sorted(r):
            out.append((k, r[k]))
    return out


class _FakeDirect:
    """Stand-in for _DirectReader on a tmpfs test dir (O_DIRECT refused there):
    counts the reads that went through the 'direct' route."""

    opened = 0
    reads = 0
    closed = 0
    lock = threading.Lock()

    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY)

    @classmethod
    def open(cls, path):
        with cls.lock:
            cls.opened += 1
        return cls(path)

    def read_into(self, dst, off):
        with self.lock:
            _FakeDirect.reads += 1
        n = os.preadv(self.fd, [dst], off)
        assert n == len(dst)

    def close(self):
        with self.lock:
            _FakeDirect.closed += 1
        os.close(self.fd)

    @classmethod
    def reset(cls):
        cls.opened = cls.reads = cls.closed = 0


class TheStreamYieldsTheSameTensorsInTheSameOrder(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="pread-stream-")
        self.paths = _files(self.root)

    def test_bytes_and_order_identical_to_the_per_file_path(self):
        ref = _per_file_reference(self.paths)
        for workers in (1, 4):
            got = list(W.pread_safetensors_stream(self.paths, workers=workers, budget_bytes=3 * MIB, log=False))
            self.assertEqual([k for k, _ in got], [k for k, _ in ref], workers)
            for (k, a), (_, b) in zip(got, ref):
                self.assertEqual(a.dtype, b.dtype, k)
                self.assertEqual(tuple(a.shape), tuple(b.shape), k)
                self.assertTrue(torch.equal(a, b), k)

    def test_should_load_skip_and_meta_asked_once_per_name(self):
        asked = []

        def sl(name):
            asked.append(name)
            if name.endswith("w00"):
                return False
            if name.endswith("odd"):
                return "meta"
            return True

        got = dict(W.pread_safetensors_stream(self.paths, should_load=sl, workers=3, budget_bytes=2 * MIB, log=False))
        self.assertFalse(any(k.endswith("w00") for k in got))
        self.assertTrue(all(got[k].device.type == "meta" for k in got if k.endswith("odd")))
        self.assertEqual(len(asked), len(set(asked)), "should_load asked twice for a name")

    def test_post_load_runs_in_a_reader_thread(self):
        main = threading.get_ident()
        where = set()

        def pl(name, t):
            where.add(threading.get_ident())
            return t * 2

        ref = dict(_per_file_reference(self.paths))
        got = dict(W.pread_safetensors_stream(self.paths, post_load=pl, workers=2, budget_bytes=4 * MIB, log=False))
        self.assertNotIn(main, where)
        k = sorted(got)[0]
        self.assertTrue(torch.equal(got[k], ref[k] * 2))

    def test_an_exception_in_a_reader_surfaces(self):
        def pl(name, t):
            if name.endswith("w05"):
                raise RuntimeError("transpose failed")
            return t

        with self.assertRaises(RuntimeError):
            list(W.pread_safetensors_stream(self.paths, post_load=pl, workers=3, budget_bytes=2 * MIB, log=False))


class TheWindowIsBounded(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="pread-stream-")
        self.paths = _files(self.root, n_files=3, per_file=16)  # 16 MiB+ per file

    def test_in_flight_never_exceeds_the_budget(self):
        st = W.StreamStats(0, 0)
        held_max = 0
        for _k, _t in W.pread_safetensors_stream(self.paths, workers=4, budget_bytes=3 * MIB, stats=st, log=False):
            held_max = max(held_max, st.inflight)
        self.assertLessEqual(st.inflight_peak, 3 * MIB)
        self.assertLessEqual(held_max, 3 * MIB)
        self.assertEqual(st.tensors, 3 * 18)
        # the per-file iterator held whole files: >= 16 MiB each, x (workers + 1)
        file_bytes = min(os.path.getsize(p) for p in self.paths)
        self.assertLess(st.inflight_peak, file_bytes)

    def test_a_tensor_larger_than_the_budget_still_loads_alone(self):
        st = W.StreamStats(0, 0)
        got = list(W.pread_safetensors_stream(self.paths, workers=4, budget_bytes=MIB // 2, stats=st, log=False))
        self.assertEqual(len(got), 3 * 18)
        self.assertLessEqual(st.inflight_peak, st.max_tensor)  # one at a time, never two

    def test_the_line_names_the_numbers(self):
        with self.assertLogs(W.logger, level="INFO") as cm:
            list(W.pread_safetensors_stream(self.paths, workers=2, budget_bytes=2 * MIB))
        line = [l for l in cm.output if W.STREAM_MARKER in l][-1]
        for key in ("files=3", "tensors=54", "budget_mib=2", "inflight_peak_mib=", "anon_peak_delta_mib=",
                    "workers=2", "GB/s="):
            self.assertIn(key, line)


class DirectIoGoesThroughTheDirectReaderAndClosesIt(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="pread-stream-")
        self.paths = _files(self.root)
        _FakeDirect.reset()

    def test_every_tensor_is_read_direct_and_every_reader_closed(self):
        with mock.patch.object(W, "_DirectReader", _FakeDirect):
            st = W.StreamStats(0, 0)
            got = list(W.pread_safetensors_stream(self.paths, direct_io=True, workers=3, budget_bytes=2 * MIB,
                                                  stats=st, log=False))
        self.assertEqual(_FakeDirect.reads, len(got))
        self.assertEqual(_FakeDirect.opened, _FakeDirect.closed)
        self.assertEqual(st.buffered_readers, 0)

    def test_an_early_stop_closes_every_reader(self):
        fds = len(os.listdir("/proc/self/fd"))
        with mock.patch.object(W, "_DirectReader", _FakeDirect):
            it = W.pread_safetensors_stream(self.paths, direct_io=True, workers=3, budget_bytes=2 * MIB, log=False)
            for i, _ in enumerate(it):
                if i == 5:
                    break
            it.close()
        self.assertEqual(_FakeDirect.opened, _FakeDirect.closed)
        self.assertLessEqual(len(os.listdir("/proc/self/fd")), fds)

    def test_real_direct_reader_on_this_filesystem(self):
        # tmpfs refuses O_DIRECT -> the buffered fallback, logged; a real disk reads direct.
        ref = _per_file_reference(self.paths)
        got = list(W.pread_safetensors_stream(self.paths, direct_io=True, workers=2, budget_bytes=2 * MIB,
                                              log=False))
        self.assertEqual([k for k, _ in got], [k for k, _ in ref])
        self.assertTrue(all(torch.equal(a, b) for (_, a), (_, b) in zip(got, ref)))


class TheIteratorsUseTheStreamOnlyWhenAsked(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="pread-stream-")
        self.paths = _files(self.root, n_files=2)

    def test_default_on_with_direct_io_off_without(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(W.STREAM_ENV, None)
            self.assertTrue(W.pread_stream_enabled(True))
            self.assertFalse(W.pread_stream_enabled(False))
        with mock.patch.dict(os.environ, {W.STREAM_ENV: "0"}):
            self.assertFalse(W.pread_stream_enabled(True))
        with mock.patch.dict(os.environ, {W.STREAM_ENV: "1"}):
            self.assertTrue(W.pread_stream_enabled(False))

    def test_multi_thread_iterator_streams_under_direct_io_with_identical_output(self):
        ref = _per_file_reference(self.paths)
        with mock.patch.object(W, "_DirectReader", _FakeDirect), \
                mock.patch.object(W, "pread_safetensors_stream", wraps=W.pread_safetensors_stream) as spy:
            os.environ.pop(W.STREAM_ENV, None)
            got = list(W.buffered_multi_thread_safetensors_weights_iterator(
                self.paths, max_workers=4, disable_mmap=True, direct_io=True, pread=True))
        spy.assert_called_once()
        self.assertEqual(spy.call_args.kwargs["workers"], 4)
        self.assertEqual([k for k, _ in got], [k for k, _ in ref])
        self.assertTrue(all(torch.equal(a, b) for (_, a), (_, b) in zip(got, ref)))

    def test_multi_thread_iterator_keeps_the_per_file_path_without_direct_io(self):
        with mock.patch.object(W, "pread_safetensors_stream") as spy:
            os.environ.pop(W.STREAM_ENV, None)
            list(W.buffered_multi_thread_safetensors_weights_iterator(
                self.paths, max_workers=2, disable_mmap=True, direct_io=False, pread=True))
        spy.assert_not_called()

    def test_single_thread_iterator_streams_under_direct_io(self):
        ref = _per_file_reference(self.paths)
        with mock.patch.object(W, "_DirectReader", _FakeDirect):
            got = list(W.safetensors_weights_iterator(self.paths, disable_mmap=True, direct_io=True, pread=True))
        self.assertEqual([k for k, _ in got], [k for k, _ in ref])


if __name__ == "__main__":
    unittest.main()
