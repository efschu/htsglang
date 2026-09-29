# SPDX-License-Identifier: Apache-2.0
"""BOOTZEIT 5d (29.09.): the presplit's copy of a layer's [E] host stack to the
card moves only the rows this rank read.

Metal (z30w-park, [ct-stream-presplit] summed): h2d 3.76 s on D TP0 over 48
layers while the rank read 29 of 201 expert rows per layer (H2 veto; the
repack already skips the rest, SGLANG_MOE_REPACK_SKIP_VETOED). The unread rows
crossed the bus for nothing.

What must hold: every read row arrives byte-identical; the unread rows arrive
as zeros (what the untouched host pages held), never as foreign bytes; only
expert-major [E, ...] parameters are cut; without rows everything is copied
as before. CPU/meta only -- no card.
"""

import ast
import inspect
import textwrap
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.fused_moe_triton import layer as fl
from sglang.srt.model_loader import loader as loader_mod


class TestToDeviceRows(unittest.TestCase):
    def test_read_rows_identical_unread_rows_zero(self):
        # Derived property of the row-cut copy (target cpu stands in for the
        # card: the function is device-agnostic).
        data = torch.arange(7 * 3 * 2, dtype=torch.int32).reshape(7, 3, 2) + 1
        rows = [0, 2, 5]
        out = loader_mod._to_device_rows(data, torch.device("cpu"), rows)
        self.assertEqual(out.shape, data.shape)
        self.assertEqual(out.dtype, data.dtype)
        for r in range(7):
            if r in rows:
                self.assertTrue(torch.equal(out[r], data[r]), r)
            else:
                self.assertTrue(bool((out[r] == 0).all()), r)

    def test_bf16_scales_keep_their_bits(self):
        data = torch.randn(5, 4).to(torch.bfloat16)
        out = loader_mod._to_device_rows(data, torch.device("cpu"), [1, 3])
        self.assertTrue(torch.equal(out[1], data[1]) and torch.equal(out[3], data[3]))


class _P:
    """A parameter stand-in whose .data may change device (a real CPU
    Parameter refuses a meta tensor; the context only needs these fields)."""

    def __init__(self, t):
        self.data = t

    @property
    def device(self):
        return self.data.device

    @property
    def shape(self):
        return self.data.shape

    def dim(self):
        return self.data.dim()


class _Moe:
    def __init__(self, e=6):
        self.num_local_experts = e
        self._p = {
            "w13_weight_packed": _P(torch.ones(e, 2, 3)),
            "w2_weight_scale": _P(torch.ones(e, 4)),
            "other": _P(torch.ones(3, 3)),
        }

    def named_parameters(self):
        return list(self._p.items())


class TestDeviceLoadingContextRows(unittest.TestCase):
    def _run(self, rows):
        m = _Moe()
        seen = []
        real = loader_mod._to_device_rows

        def spy(data, dev, r):
            seen.append(tuple(data.shape))
            return real(data, dev, r)

        with mock.patch.object(loader_mod, "_to_device_rows", side_effect=spy), \
                mock.patch.object(loader_mod, "is_pin_memory_available", return_value=False):
            with loader_mod.device_loading_context(m, torch.device("meta"), rows=rows):
                for n, p in m.named_parameters():
                    self.assertEqual(p.device.type, "meta", n)
                    # the presplit replaces what it consumed with a host
                    # placeholder, so the exit copies nothing back
                    p.data = torch.empty((0,) + tuple(p.shape[1:]))
        return seen

    def test_only_expert_major_params_are_cut(self):
        self.assertEqual(sorted(self._run([0, 4])), [(6, 2, 3), (6, 4)])

    def test_without_rows_everything_is_copied_whole(self):
        self.assertEqual(self._run(None), [])


class TestPresplitCallEdge(unittest.TestCase):
    def test_presplit_copies_with_the_read_rows(self):
        # Call edge: the presplit asks ct_h2d_rows and hands its answer to
        # device_loading_context as rows= (no rows -> the old call, unchanged).
        src = textwrap.dedent(inspect.getsource(fl.FusedMoE._ct_stream_presplit_now))
        fn = ast.parse(src).body[0]
        asked = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", "") == "ct_h2d_rows"]
        with_rows = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                     and getattr(n.func, "id", "") == "device_loading_context"
                     and any(kw.arg == "rows" for kw in n.keywords)]
        self.assertEqual(len(asked), 1)
        self.assertEqual(len(with_rows), 1)

    def test_rows_are_the_repack_rows_and_off_is_all(self):
        layer = fl.FusedMoE.__new__(fl.FusedMoE)
        torch.nn.Module.__init__(layer)
        layer.num_local_experts = 201
        with mock.patch("sglang.srt.layers.moe.store_adopt.repack_rows",
                        return_value=[0, 3, 7]) as rr:
            with envs.SGLANG_OPT_LOAD_H2D_READ_ROWS.override(True):
                self.assertEqual(fl.ct_h2d_rows(layer), [0, 3, 7])
            rr.assert_called_once_with(layer, 201)
            with envs.SGLANG_OPT_LOAD_H2D_READ_ROWS.override(False):
                self.assertIsNone(fl.ct_h2d_rows(layer))
        self.assertTrue(envs.SGLANG_OPT_LOAD_H2D_READ_ROWS.get())


if __name__ == "__main__":
    unittest.main()
