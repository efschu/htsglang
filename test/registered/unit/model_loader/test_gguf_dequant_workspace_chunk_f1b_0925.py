"""F1b (boot weg2rc6gg, 2026-09-25): the GGUF dequant workspace is born in the
chunk tag of the LAYER that sized it, not in the base tag.

WHAT THE METAL SHOWED.  weg2rc6gg (RC6 f1c3c25ff5, F1 in) flipped D->P cleanly
(every P own-tag census unmapped=0, every COVER row OVERHANG) and died on the
first flip back, at P's release_memory_occupation:

    W106 Weg2XchgWakeSourceGapRefused: group=P rank=1 tag=weights
    expected_bytes=178257920: ... this rank's own plan carries zero
    descriptors for it on the source side

178257920 B = 17408 x 5120 x 2 = the bf16 dequant target of ONE MLP shard at
TP=1 = the shared dequant workspace (_DEQUANT_WS).  F1 kept it in the base tag
(dequant_workspace_deferred, "operator order").  PP1 (layers 42..52) owns no
embed/norm/head, so its base tag held NOTHING but that scratch: bytes resident,
zero source descriptors -> W106.  PP0 and PP2 carry the same ~170 MiB as base
overhang (+173.0 / +172.4 MiB at the boot's COVER) but also real base-tag
parameters, so the check passes there; the D ranks likewise (+94.2 / +44.2 /
+44.2 MiB = their TP shares of the same target).  In RC4 (INT8, NVFP4) P rank 1
deposited `tag=weights pieces=0 resident_bytes=0` on all 25 flips: without GGUF
the base tag of a layers-only stage is empty.

THE RULE (#1233, F1's own): a post-load allocation lands in the chunk tag of
its layer.  The workspace has no single owner, so it takes the chunk of the
layer whose request set its size -- the first such layer in the pass.  A size
set by a module outside every layer keeps the base tag, as before.
"""

import contextlib
import os
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.linear import MergedColumnParallelLinear
from sglang.srt.layers.quantization import gguf as G
from sglang.srt.managers import weg2_memory_saver as ms
from sglang.srt.model_loader import loader as loader_mod
from sglang.test.test_utils import CustomTestCase

from test_gguf_postload_chunk_scope_0925 import (  # same directory, no package
    K_IN,
    MERGED,
    Q8_0,
    _allocations,
    _chunked_weights_region,
    _gguf_layer,
    _q8,
    _Saver,
)


def _stage(prefixes, *, tp_size=1, tp_rank=0):
    """A model holding merged GGUF qweights under their real names, loaded like
    a column-parallel linear on rank ``tp_rank`` of ``tp_size``."""
    root = torch.nn.Module()
    for prefix in prefixes:
        layer = _gguf_layer(prefix, MERGED)
        fake = SimpleNamespace(output_sizes=MERGED, tp_size=tp_size,
                               tp_rank=tp_rank, tp_units=None, tp_family=None)
        for sid, rows in enumerate(MERGED):
            MergedColumnParallelLinear.weight_loader(
                fake, layer.qweight_type, torch.tensor(Q8_0, dtype=torch.uint8), sid)
            MergedColumnParallelLinear.weight_loader(
                fake, layer.qweight, _q8(rows, sid + 1), sid)
        cur = root
        parts = prefix.split(".")
        for part in parts[:-1]:
            if not hasattr(cur, part):
                cur.add_module(part, torch.nn.Module())
            cur = getattr(cur, part)
        cur.add_module(parts[-1], layer)
    return root


def _post_load(root):
    """Run the GGUF post-load pass; return (log, current tag after the pass)."""
    saver = _Saver(ms.GPU_MEMORY_TYPE_WEIGHTS)
    log: list = []
    with _chunked_weights_region(saver), _allocations(saver, log):
        loader_mod._process_weights_after_loading_by_layer_chunk(
            root, torch.device("cpu"))
    return log, saver.current


class _Isolated(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict(G._DEQUANT_WS, clear=True).start()
        mock.patch.dict(G._DEQUANT_PEAK_TARGET, clear=True).start()
        # the in-place ggml_dequantize(out=) schema is a CUDA wheel property;
        # the workspace path is what is under test
        mock.patch.object(G, "_dequant_supports_out", lambda: True).start()


class TheWorkspaceIsBornInALayersChunk(_Isolated):
    def test_a_layers_only_stage_allocates_nothing_under_the_base_tag(self):
        """The W106 shape: P rank 1 of the 42,11,11 cut holds layers 42..52
        only.  RED on f1c3c25ff5: the workspace was the one base-tag
        allocation of the whole pass."""
        root = _stage([f"model.layers.{i}.mlp.gate_up_proj" for i in (42, 47, 48, 52)])
        log, current = _post_load(root)
        base = [row for row in log if row[3] == ms.GPU_MEMORY_TYPE_WEIGHTS]
        self.assertEqual(base, [], f"base-tag allocations on a layers-only stage: {base}")
        work = [row for row in log if row[1] == torch.bfloat16]
        self.assertEqual(work, [("empty", torch.bfloat16, max(MERGED) * K_IN, "weights_5")])
        self.assertEqual(current, ms.GPU_MEMORY_TYPE_WEIGHTS)

    def test_the_first_layer_that_set_the_size_owns_the_chunk(self):
        """A later layer asking for the SAME size does not move it; a later
        layer asking for MORE does (its request set the size)."""
        small = _gguf_layer("model.layers.3.mlp.gate_up_proj", [MERGED[1]])
        root = _stage(["model.layers.40.mlp.gate_up_proj",
                       "model.layers.63.mlp.gate_up_proj"])
        root.model.layers.add_module("3", torch.nn.Module())
        root.model.layers._modules["3"].add_module("mlp", torch.nn.Module())
        fake = SimpleNamespace(output_sizes=[MERGED[1]], tp_size=1, tp_rank=0,
                               tp_units=None, tp_family=None)
        MergedColumnParallelLinear.weight_loader(
            fake, small.qweight_type, torch.tensor(Q8_0, dtype=torch.uint8), 0)
        MergedColumnParallelLinear.weight_loader(fake, small.qweight, _q8(MERGED[1], 9), 0)
        root.model.layers._modules["3"].mlp.add_module("gate_up_proj", small)
        log, _current = _post_load(root)
        work = [row for row in log if row[1] == torch.bfloat16]
        # pass order: layer 40 (sets the size), 63 (same size, no move), 3
        # (smaller, no move) -> the chunk of layer 40
        self.assertEqual(work, [("empty", torch.bfloat16, max(MERGED) * K_IN, "weights_5")])

    def test_every_d_tp_rank_puts_its_workspace_in_a_chunk(self):
        """Group D: TP3, every rank holds a shard of every layer; the workspace
        (its rank's shard size) lands in the chunk of the first layer that
        sized it, never in the base tag."""
        for rank in range(3):
            with self.subTest(rank=rank):
                G._DEQUANT_WS.clear()
                G._DEQUANT_PEAK_TARGET.clear()
                root = _stage([f"model.layers.{i}.mlp.gate_up_proj" for i in (0, 40, 63)],
                              tp_size=3, tp_rank=rank)
                log, current = _post_load(root)
                work = [row for row in log if row[1] == torch.bfloat16]
                self.assertEqual(len(work), 1, work)
                self.assertEqual(work[0][3], "weights_0")
                self.assertEqual(current, ms.GPU_MEMORY_TYPE_WEIGHTS)

    def test_a_size_set_outside_every_layer_keeps_the_base_tag(self):
        """A module with no layer id (the vocab family) sizing the workspace:
        weight_chunk_scope(None) is a no-op, exactly as before F1b."""
        root = _stage(["lm_head"])
        log, _current = _post_load(root)
        work = [row for row in log if row[1] == torch.bfloat16]
        self.assertEqual(work, [("empty", torch.bfloat16, max(MERGED) * K_IN,
                                 ms.GPU_MEMORY_TYPE_WEIGHTS)])


class NoChunkingNoChange(_Isolated):
    def test_without_the_flip_the_workspace_is_allocated_once_as_before(self):
        root = _stage(["model.layers.40.mlp.gate_up_proj"])
        saver = _Saver(ms.GPU_MEMORY_TYPE_WEIGHTS)
        log: list = []
        # no chunk envs: weight_chunk_scope is a no-op
        with mock.patch.dict(os.environ, {}, clear=False), _allocations(saver, log):
            for key in (ms.WEIGHT_CHUNK_ENV_LAYERS, ms.WEIGHT_CHUNK_ENV_COUNT):
                os.environ.pop(key, None)
            loader_mod._process_weights_after_loading_by_layer_chunk(
                root, torch.device("cpu"))
        work = [row for row in log if row[1] == torch.bfloat16]
        self.assertEqual([(f, d, n) for f, d, n, _t in work],
                         [("empty", torch.bfloat16, max(MERGED) * K_IN)])
        self.assertEqual(len(G._DEQUANT_WS), 1)

    def test_immediate_growth_outside_the_deferral_is_unchanged(self):
        G._reserve_dequant_workspace(128, torch.bfloat16, torch.device("cpu"))
        self.assertEqual([b.numel() for b in G._DEQUANT_WS.values()], [128])
