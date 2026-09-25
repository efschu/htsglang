"""The GGUF weight stream runs with numpy's MADV_HUGEPAGE hint OFF.

WHY.  weg2rc7gg (RC7b bb086e1120, 2026-09-25): group D's GGUF producer took
700-730 s per rank against ~50 s on group P for the same code and file; the
loading thread's rusage put it in the kernel (sys 670.7 s against 20.4 s, user
equal).  All three D ranks sat in gguf-py's ``_apply_over_grouped_rows`` at
``np.concatenate(..., out=out)`` -- the first touch of the out_proj dequant
buffer, which numpy had madvise(MADV_HUGEPAGE)d.  Under the host's THP
``defrag=madvise`` every 2 MiB fault on it compacted synchronously, behind group
P's pre-pinned arena, and failed.  See ``model_loader/gguf_numpy_hugepage.py``.

Five proofs, all hermetic (no GPU, no GGUF file):

a) the scope switches numpy's hint off and restores the previous value -- on a
   normal exit, when the load raises, and when it was already off; the opt-out
   env leaves it alone; a numpy without the switch is tolerated.
b) the kernel sees it: a fresh 64 MiB numpy array's mapping carries the ``hg``
   (VM_HUGEPAGE) flag in /proc/self/smaps with the hint on and not inside the
   scope.
c) byte identity: the REAL ``Qwen35GGUFAdapter.transform_stream`` dequantizes an
   IQ4_XS out_proj (block 256 against head_v_dim 128: the path weg2rc7gg's
   Q4_K/Q5_K out_proj took) to the same bytes with the hint on and off.
d) the loader wiring: ``GGUFModelLoader.load_model`` runs ``model.load_weights``
   -- i.e. the whole weight stream -- with the hint off, and numpy's setting is
   back afterwards.  RED on bb086e1120: the stream ran with numpy's default.
e) in the rank process, without an env: a process SPAWNED the way sglang
   starts its schedulers, NUMPY_MADVISE_HUGEPAGE removed from its environment,
   starts with numpy's hint on, streams its GGUF with it off (numpy flag and
   kernel VmFlags) and has it back afterwards.
"""

import logging
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch

from sglang.srt.model_loader import gguf_numpy_hugepage as H
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_SWITCH = H.numpy_madvise_hugepage_switch()


def _get_hint():
    try:
        from numpy._core import multiarray as ma
    except ImportError:  # numpy 1.x
        from numpy.core import multiarray as ma
    return bool(ma._get_madvise_hugepage())


@unittest.skipIf(_SWITCH is None, "this numpy has no _set_madvise_hugepage")
class _HintCase(CustomTestCase):
    """Every test starts with numpy's hint ON and leaves the process as found."""

    def setUp(self):
        self._orig = bool(_SWITCH(True))
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("SGLANG_GGUF_NUMPY_HUGEPAGE", None)

    def tearDown(self):
        _SWITCH(self._orig)


class TheScopeSwitchesTheHintOffAndBack(_HintCase):
    def test_off_inside_restored_after(self):
        with H.numpy_hugepage_off_for_gguf_load(rank=0) as switched:
            self.assertTrue(switched)
            self.assertFalse(_get_hint())
        self.assertTrue(_get_hint())

    def test_restored_when_the_load_raises(self):
        with self.assertRaises(RuntimeError):
            with H.numpy_hugepage_off_for_gguf_load(rank=0):
                self.assertFalse(_get_hint())
                raise RuntimeError("load failed")
        self.assertTrue(_get_hint())

    def test_a_process_that_had_it_off_keeps_it_off(self):
        _SWITCH(False)
        with H.numpy_hugepage_off_for_gguf_load(rank=0):
            self.assertFalse(_get_hint())
        self.assertFalse(_get_hint(), "the scope must restore, not force on")

    def test_opt_out_env_keeps_numpys_setting(self):
        os.environ["SGLANG_GGUF_NUMPY_HUGEPAGE"] = "1"
        with H.numpy_hugepage_off_for_gguf_load(rank=0) as switched:
            self.assertFalse(switched)
            self.assertTrue(_get_hint())
        self.assertTrue(_get_hint())

    def test_a_numpy_without_the_switch_is_tolerated(self):
        with mock.patch.object(H, "numpy_madvise_hugepage_switch", return_value=None):
            with H.numpy_hugepage_off_for_gguf_load(rank=0) as switched:
                self.assertFalse(switched)
        self.assertTrue(_get_hint())

    def test_the_named_line_says_off_and_what_it_was(self):
        with self.assertLogs(H.logger, level=logging.INFO) as cm:
            with H.numpy_hugepage_off_for_gguf_load(rank=2):
                pass
        line = "\n".join(cm.output)
        self.assertIn("[GGUF-NUMPY-THP] rank 2: numpy MADV_HUGEPAGE OFF", line)
        self.assertIn("(was on;", line)
        self.assertIn("host THP enabled=", line)

    def test_thp_mode_reads_the_bracketed_value(self):
        m = mock.mock_open(read_data="always defer defer+madvise [madvise] never\n")
        with mock.patch("builtins.open", m):
            self.assertEqual(H.thp_mode("defrag"), "madvise")
        with mock.patch("builtins.open", side_effect=OSError):
            self.assertEqual(H.thp_mode("defrag"), "?")


def _vmflags_of(addr: int):
    """VmFlags of the /proc/self/smaps mapping that contains ``addr``."""
    with open("/proc/self/smaps") as f:
        inside = False
        for line in f:
            head = line.split()[0] if line.strip() else ""
            if "-" in head and ":" not in head:
                lo, hi = (int(x, 16) for x in head.split("-"))
                inside = lo <= addr < hi
            elif inside and line.startswith("VmFlags:"):
                return line.split()[1:]
    return None


_THP_PRESENT = os.path.isdir("/sys/kernel/mm/transparent_hugepage")


@unittest.skipUnless(sys.platform == "linux" and _THP_PRESENT, "needs Linux THP")
class TheKernelSeesNoHugepageAdvice(_HintCase):
    """64 MiB > glibc's largest mmap threshold (32 MiB): always a fresh mapping,
    so the flag read is this array's own and not a reused heap range's."""

    SIZE = 64 << 20

    def _flags(self):
        a = np.empty(self.SIZE, dtype=np.uint8)
        flags = _vmflags_of(a.ctypes.data + self.SIZE // 2)
        del a
        self.assertIsNotNone(flags)
        return flags

    def test_numpys_default_advises_hugepages(self):
        self.assertIn("hg", self._flags())

    def test_inside_the_scope_it_does_not(self):
        with H.numpy_hugepage_off_for_gguf_load(rank=0):
            self.assertNotIn("hg", self._flags())
        self.assertIn("hg", self._flags())


def _iq4_xs_out_proj(rows: int, cols: int, seed: int = 0) -> torch.Tensor:
    """Random IQ4_XS payload [rows, cols/256*136] with finite fp16 scales."""
    import gguf

    block, type_size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.IQ4_XS]
    n_blocks = rows * cols // block
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(n_blocks, type_size), dtype=np.uint8)
    d = rng.uniform(1e-3, 2e-2, size=n_blocks).astype(np.float16)
    raw[:, 0:2] = d.view(np.uint8).reshape(n_blocks, 2)
    return torch.from_numpy(raw.reshape(rows, cols // block * type_size))


class TheTransformIsBitIdentical(_HintCase):
    """The weg2rc7gg tensor class (there Q4_K/Q5_K, here IQ4_XS: same block 256,
    same path): out_proj of a GDN layer with num_k 16 != num_v 48 and
    head_v_dim 128 (not a multiple of block 256) is dequantized in
    transform_stream, then v-head un-tiled.  256 rows x 6144 columns -> a 6 MiB
    float32 ``out`` buffer, above numpy's 4 MiB hint floor."""

    ROWS, COLS = 256, 48 * 128

    def _adapter(self):
        from sglang.srt.model_loader.gguf_qwen35 import Qwen35GGUFAdapter

        a = Qwen35GGUFAdapter.__new__(Qwen35GGUFAdapter)
        a.is_draft = False
        a.num_layers = 64
        a.num_k, a.num_v, a.head_k_dim, a.head_v_dim = 16, 48, 128, 128
        a.config = types.SimpleNamespace(torch_dtype=torch.bfloat16)
        return a

    def _run(self):
        import gguf

        name = "model.layers.0.linear_attn.out_proj.qweight"
        stream = [
            (name + "_type", torch.tensor(int(gguf.GGMLQuantizationType.IQ4_XS))),
            (name, _iq4_xs_out_proj(self.ROWS, self.COLS)),
        ]
        return list(self._adapter().transform_stream(stream))

    def test_hint_on_and_off_give_the_same_bytes(self):
        self.assertTrue(_get_hint())
        on = self._run()
        with H.numpy_hugepage_off_for_gguf_load(rank=0):
            off = self._run()
        self.assertEqual([n for n, _ in on], [n for n, _ in off])
        self.assertEqual(
            [n for n, _ in on], ["model.layers.0.linear_attn.out_proj.weight"]
        )
        (_, w_on), (_, w_off) = on[0], off[0]
        self.assertEqual(w_on.dtype, torch.bfloat16)
        self.assertEqual(tuple(w_on.shape), (self.ROWS, self.COLS))
        self.assertTrue(torch.isfinite(w_on.float()).all())
        self.assertEqual(
            w_on.view(torch.int16).numpy().tobytes(),
            w_off.view(torch.int16).numpy().tobytes(),
        )


def _stubbed_gguf_load(probe):
    """Run the REAL ``GGUFModelLoader.load_model`` with everything around
    ``model.load_weights`` stubbed; ``probe()`` runs while load_weights consumes
    the weight stream, and its result is returned."""
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.model_loader import loader as L

    seen = {}

    class _Model:
        def load_weights(self, weights):
            seen["n"] = sum(1 for _ in weights)
            seen["probe"] = probe()
            return set()

    model_config = types.SimpleNamespace(
        model_path="/nonexistent/x.gguf",
        hf_config=types.SimpleNamespace(tie_word_embeddings=False),
        dtype=torch.float32,
    )
    ldr = L.GGUFModelLoader(LoadConfig(load_format="gguf", tp_rank=1))
    with mock.patch.object(
        L.GGUFModelLoader, "_prepare_weights", return_value="/nonexistent/x.gguf"
    ), mock.patch.object(
        L.GGUFModelLoader, "_get_gguf_weights_map", return_value={}
    ), mock.patch.object(
        L.GGUFModelLoader,
        "_get_weights_iterator",
        return_value=iter([("w", torch.zeros(1))]),
    ), mock.patch(
        "sglang.srt.model_loader.gguf_registry.create_gguf_adapter",
        return_value=None,
    ), mock.patch(
        "sglang.srt.model_loader.gguf_dflash.is_dflash_gguf_config",
        return_value=False,
    ), mock.patch.object(
        L, "get_gguf_extra_tensor_names", return_value=[]
    ), mock.patch.object(
        L, "_get_quantization_config", return_value=None
    ), mock.patch.object(
        L, "_initialize_model", return_value=_Model()
    ), mock.patch.object(
        L, "_process_weights_after_loading_by_layer_chunk"
    ):
        ldr.load_model(
            model_config=model_config,
            device_config=types.SimpleNamespace(device="cpu"),
        )
    assert seen.get("n") == 1, seen
    return seen["probe"]


def _hint_and_kernel_flag():
    """numpy's hint, and whether a fresh 64 MiB array's mapping carries ``hg``
    (None where /proc/self/smaps or THP is not there)."""
    flag = None
    if sys.platform == "linux" and _THP_PRESENT:
        a = np.empty(64 << 20, dtype=np.uint8)
        flags = _vmflags_of(a.ctypes.data + (32 << 20))
        del a
        flag = None if flags is None else ("hg" in flags)
    return _get_hint(), flag


def _rank_process_main(conn):
    """Body of a freshly SPAWNED process -- the way sglang starts a rank:
    numpy imported with its own default, no NUMPY_MADVISE_HUGEPAGE in the
    environment, then the real GGUF loader."""
    try:
        before = _hint_and_kernel_flag()
        during = _stubbed_gguf_load(_hint_and_kernel_flag)
        after = _hint_and_kernel_flag()
        conn.send(
            {
                "env": os.environ.get("NUMPY_MADVISE_HUGEPAGE"),
                "before": before,
                "during": during,
                "after": after,
            }
        )
    except BaseException as e:  # report, never hang the parent
        conn.send({"error": repr(e)})
    finally:
        conn.close()


class TheLoaderStreamsWithTheHintOff(_HintCase):
    """In-process: the fake model records numpy's hint while it consumes the
    weight stream.  RED on bb086e1120 (the stream ran with numpy's default)."""

    def test_load_weights_runs_inside_the_scope(self):
        hint, flag = _stubbed_gguf_load(_hint_and_kernel_flag)
        self.assertFalse(hint, "the GGUF weight stream ran with MADV_HUGEPAGE")
        if flag is not None:
            self.assertFalse(flag, "a 64 MiB array in the stream got VM_HUGEPAGE")
        self.assertTrue(_get_hint(), "numpy's setting was not restored")


@unittest.skipIf(_SWITCH is None, "this numpy has no _set_madvise_hugepage")
class TheRankProcessNeedsNoEnv(CustomTestCase):
    """The fix lives in the process that loads -- a rank spawned like sglang
    spawns its schedulers, with NUMPY_MADVISE_HUGEPAGE REMOVED from its
    environment (memory rule: an env in the launcher is not an env in the rank;
    /proc/<rank>/environ cannot even show it, setproctitle zeroes it).  Inside
    that process numpy starts with the hint on, the loader turns it off for the
    stream -- at numpy's flag AND at the kernel's VmFlags -- and gives it back."""

    def test_spawned_rank_streams_without_hugepage_advice(self):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe(duplex=False)
        with mock.patch.dict(os.environ):
            os.environ.pop("NUMPY_MADVISE_HUGEPAGE", None)
            os.environ.pop("SGLANG_GGUF_NUMPY_HUGEPAGE", None)
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            proc = ctx.Process(target=_rank_process_main, args=(child,))
            proc.start()
        child.close()
        self.assertTrue(parent.poll(240), "spawned rank process did not report")
        got = parent.recv()
        proc.join(30)
        self.assertNotIn("error", got, got)
        self.assertIsNone(got["env"])
        hint_before, hg_before = got["before"]
        hint_during, hg_during = got["during"]
        hint_after, hg_after = got["after"]
        self.assertTrue(hint_before, "numpy's Linux default is the hint ON")
        self.assertFalse(hint_during, "the rank streamed its GGUF with the hint on")
        self.assertTrue(hint_after, "the rank did not get numpy's setting back")
        if hg_before is not None:
            self.assertTrue(hg_before)
            self.assertFalse(hg_during, "the kernel saw MADV_HUGEPAGE in the stream")
            self.assertTrue(hg_after)


if __name__ == "__main__":
    unittest.main()
