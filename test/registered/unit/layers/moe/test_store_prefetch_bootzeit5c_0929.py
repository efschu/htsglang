# SPDX-License-Identifier: Apache-2.0
"""BOOTZEIT 5c (29.09.): the next layer's expert-store files open in the
background while this layer's shards are consumed.

Metal (z30w-park, [ct-stream-presplit] summed): store_open 5.17 s on PP0
(29 layers x 4 tmpfs files, 23.78 GiB registered) and 2.90 s on D TP0, all on
the loader thread inside the presplit. Every MoE layer of a rank has the same
store geometry, so layer N's opens name layer N+1's files exactly.

What must hold: the presplit gets the same file at the same size as without
the prefetch; a prefetch of the wrong geometry never survives into the real
open; only layers this process presplits are touched; and the prefetch starts
only after the layer's host stack is gone (the host peak argument).
CPU only: registration is patched off (no CUDA context).
"""

import ast
import inspect
import os
import tempfile
import textwrap
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import expert_offload as eo
from sglang.srt.layers.moe import expert_store as es
from sglang.srt.layers.moe import store_prefetch as sp
from sglang.srt.layers.moe.fused_moe_triton import layer as fl
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_wNa16_moe as ctm,
)
from sglang.srt.models import qwen4_exp

GEOM = [
    ("w13_weight_packed", (4, 8), torch.int32),
    ("w13_weight_scale", (2, 8), torch.bfloat16),
]
SLOTS = 6


class _Case(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = self._dir.name
        self._cuda = mock.patch("torch.cuda.is_available", return_value=False)
        self._cuda.start()
        sp.drain("setup")

    def tearDown(self):
        sp.drain("teardown")
        self._cuda.stop()
        self._dir.cleanup()

    def _open(self, layer_id, attr, row, dtype, slots=SLOTS):
        return es.open_store(self.dir, f"L{layer_id}", attr, slots, row, dtype, num_slots=slots)


class TestPrefetchRoundTrip(_Case):
    def test_next_layer_gets_the_prefetched_file_same_bytes_same_size(self):
        with envs.SGLANG_OPT_LOAD_STORE_PREFETCH.override(True):
            sp.note_armed(3)
            sp.note_armed(4)
            for attr, row, dt in GEOM:
                self._open(3, attr, row, dt)
            n = sp.prefetch_next(3, directory=self.dir, slots=SLOTS,
                                 geometry=GEOM, device_index=None)
            self.assertEqual(n, len(GEOM))
            t, created = self._open(4, "w13_weight_packed", (4, 8), torch.int32)
        self.assertEqual(sp.STATS["taken"], 1)
        self.assertTrue(created)
        self.assertEqual(tuple(t.shape), (SLOTS, 4, 8))
        path = es.store_path(self.dir, "L4", "w13_weight_packed")
        self.assertEqual(os.path.getsize(path), SLOTS * 4 * 8 * 4)
        # the same shared file: a write through the prefetched tensor is what
        # a second opener (group D) reads
        t[2].fill_(7)
        again, created2 = es.open_store_uncached(
            self.dir, "L4", "w13_weight_packed", SLOTS, (4, 8), torch.int32,
            num_slots=SLOTS)
        self.assertFalse(created2)
        self.assertTrue(bool((again[2] == 7).all()))

    def test_only_layers_this_process_presplits_are_touched(self):
        # PP0 owns 0..28: its last layer must not create PP1's layer-29 file.
        with envs.SGLANG_OPT_LOAD_STORE_PREFETCH.override(True):
            sp.note_armed(28)
            n = sp.prefetch_next(28, directory=self.dir, slots=SLOTS,
                                 geometry=GEOM, device_index=None)
        self.assertEqual(n, 0)
        self.assertFalse(os.path.exists(es.store_path(self.dir, "L29", "w13_weight_packed")))

    def test_switch_off_prefetches_nothing(self):
        self.assertFalse(envs.SGLANG_OPT_LOAD_STORE_PREFETCH.get())
        sp.note_armed(1)
        self.assertEqual(sp.prefetch_next(0, directory=self.dir, slots=SLOTS,
                                          geometry=GEOM, device_index=None), 0)


class TestWrongGeometryNeverSurvives(_Case):
    def test_a_created_file_of_the_wrong_size_is_replaced(self):
        # Derived: the prefetch guessed SLOTS, the layer wants SLOTS + 2. The
        # file the prefetch created must go, or the real open refuses it
        # ("shared store ... has N bytes, this layout wants M").
        with envs.SGLANG_OPT_LOAD_STORE_PREFETCH.override(True):
            sp.note_armed(5)
            sp.prefetch_next(4, directory=self.dir, slots=SLOTS,
                             geometry=GEOM[:1], device_index=None)
            t, created = self._open(5, "w13_weight_packed", (4, 8), torch.int32,
                                    slots=SLOTS + 2)
        self.assertEqual(sp.STATS["dropped"], 1)
        self.assertTrue(created)
        self.assertEqual(tuple(t.shape), (SLOTS + 2, 4, 8))
        self.assertEqual(
            os.path.getsize(es.store_path(self.dir, "L5", "w13_weight_packed")),
            (SLOTS + 2) * 4 * 8 * 4)

    def test_an_existing_file_is_never_resized_by_the_prefetch(self):
        # Group D opens files P created: a mismatching prefetch fails and the
        # real open behaves exactly as without the prefetch (it refuses).
        es.open_store_uncached(self.dir, "L7", "w13_weight_packed", SLOTS + 2,
                               (4, 8), torch.int32, num_slots=SLOTS + 2)
        with envs.SGLANG_OPT_LOAD_STORE_PREFETCH.override(True):
            sp.note_armed(7)
            sp.prefetch_next(6, directory=self.dir, slots=SLOTS,
                             geometry=GEOM[:1], device_index=None)
            with self.assertRaises(ValueError):
                self._open(7, "w13_weight_packed", (4, 8), torch.int32, slots=SLOTS)
        self.assertEqual(sp.STATS["failed"], 1)
        self.assertEqual(
            os.path.getsize(es.store_path(self.dir, "L7", "w13_weight_packed")),
            (SLOTS + 2) * 4 * 8 * 4)


class TestCallEdges(unittest.TestCase):
    def test_prefetch_starts_after_the_host_stack_is_gone(self):
        # The host-peak argument: layer N+1's tmpfs pages may exist only once
        # layer N's host stack is dropped, i.e. after the device_loading_context
        # of the presplit has exited -- never inside it.
        src = textwrap.dedent(inspect.getsource(fl.FusedMoE._ct_stream_presplit_now))
        fn = ast.parse(src).body[0]
        inside_ctx = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.With) and any(
                isinstance(i.context_expr, ast.Call)
                and getattr(i.context_expr.func, "id", "") == "device_loading_context"
                for i in node.items
            ):
                inside_ctx |= {id(n) for n in ast.walk(node)}
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "prefetch_next"]
        self.assertEqual(len(calls), 1)
        self.assertNotIn(id(calls[0]), inside_ctx)

    def test_presplit_hands_its_geometry_and_the_scheme_arms_the_layer(self):
        self.assertIn("_store_prefetch_next",
                      inspect.getsource(eo.presplit_expert_offload_after_repack))
        self.assertIn("note_armed", inspect.getsource(ctm.CompressedTensorsWNA16MoE.create_weights))
        src = inspect.getsource(qwen4_exp.Qwen4ExpForConditionalGeneration.load_weights)
        self.assertIn("store_prefetch.drain", src)

    def test_open_store_asks_the_prefetch_first(self):
        src = textwrap.dedent(inspect.getsource(es.open_store))
        fn = ast.parse(src).body[0]
        names = [n.func.attr for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertIn("take", names)


if __name__ == "__main__":
    unittest.main()
