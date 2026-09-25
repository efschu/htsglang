"""F1 covers BOTH flip groups: group D (TP shards of EVERY layer) and a group-P stage (a layer SUBSET).

Companion of ``test_gguf_postload_chunk_scope_0925.py`` (S's F1 test, RC6 d294b3e362), which pins the
GGUF post-load pass and ``GGUFUninitializedParameter.materialize`` at ``tp_size=1`` over layers
0/40/41/63. Boot weg2rc5gg (RC5, 2026-09-25) showed the defect on BOTH groups: P died at its first
wake, and group D stood on 24 of 30 tags at SAVER-BOOKS-LESS-THAN-WALK -- it would have died the same
way at the P->D flip. What this file adds, and why it is the same code path for D:

* group D loads the ``.gguf`` through the same GGUFModelLoader (weg2rc5gg: ``[#89 STEP-0]`` and
  ``process_weights_after_loading (flat-assembly`` on TP0-2 as on PP0-2);
* the launcher publishes ONE chunk geometry to both groups (``build_env(..., chunk_layers,
  chunk_count, ..., group="P"|"D")``), and the chunk tag is ``weights_{layer_id // L}`` of the GLOBAL
  layer id, so a TP rank (all 64 layers, sharded) and a PP stage (a layer range) map a layer to the
  same tag;
* F1 keys the scope on the layer, never on the shard: the post-load pass by module name, the
  materialize by ``weg2_layer_id`` from ``layer.prefix``.

The GGUF weight path runs for real on CPU tensors; only the memory saver's current-tag entry and the
per-tag CUDA pool are stand-ins (no saver on a desk), exactly as in S's test.
"""

from __future__ import annotations

import contextlib
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.managers.weg2_memory_saver as ms
import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.layers.linear import MergedColumnParallelLinear
from sglang.srt.layers.quantization import gguf as G
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# the fixture shape of S's F1 test (and test_weg2_gguf_flat_declaration_g1)
K_IN = 64
Q8_0 = 8
Q8_ROW = K_IN // 32 * 34
MERGED = [12, 6]
FULL_FLAT_BYTES = sum(MERGED) * Q8_ROW  # one rank holding the whole merged qweight


class _Saver:
    """The saver's C entry point: which tag is current for the next allocation."""

    def __init__(self, base: str):
        self.current = base

    def tms_set_current_tag(self, tag: bytes) -> None:
        self.current = tag.decode("utf-8")


@contextlib.contextmanager
def _chunked_weights_region(saver: _Saver):
    """The launcher's chunk geometry for the 64-layer 27B (8 layers per chunk, 8 chunks) -- the ONE
    geometry build_env publishes to both groups -- and model_runner's open BASE weights region."""
    env = {ms.WEIGHT_CHUNK_ENV_LAYERS: "8", ms.WEIGHT_CHUNK_ENV_COUNT: "8"}
    with mock.patch.dict(os.environ, env), ms.weights_region_tag(
        ms.GPU_MEMORY_TYPE_WEIGHTS
    ), mock.patch.object(ms, "_tms_cdll_in_region", lambda: saver), mock.patch.object(
        ms, "tag_pool_scope", lambda tag: contextlib.nullcontext(tag)
    ):
        yield


@contextlib.contextmanager
def _allocations(saver, log):
    """Record (factory, dtype, numel, current tag) of every torch.empty/zeros."""
    real_empty, real_zeros = torch.empty, torch.zeros

    def _numel(args):
        shape = (
            args[0]
            if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size))
            else args
        )
        n = 1
        for d in shape:
            n *= int(d)
        return n

    def empty(*args, **kw):
        out = real_empty(*args, **kw)
        log.append(("empty", out.dtype, _numel(args), saver.current))
        return out

    def zeros(*args, **kw):
        out = real_zeros(*args, **kw)
        log.append(("zeros", out.dtype, _numel(args), saver.current))
        return out

    with mock.patch.object(torch, "empty", empty), mock.patch.object(torch, "zeros", zeros):
        yield


def _q8(rows, salt):
    return torch.tensor(
        [[(salt * 31 + r * 7 + b * 3) % 251 + 1 for b in range(Q8_ROW)] for r in range(rows)],
        dtype=torch.uint8,
    )


def _gguf_layer(prefix, sizes):
    layer = torch.nn.Module()
    layer.prefix = prefix  # LinearBase sets it before create_weights
    method = G.GGUFLinearMethod(G.GGUFConfig([]))
    method.create_weights(layer, K_IN, list(sizes), K_IN, sum(sizes), torch.bfloat16)
    layer.quant_method = method
    return layer


def _post_load_pass(prefix: str, tp_size: int, tp_rank: int):
    """The merged qweight of ``prefix`` loaded like a column-parallel linear on rank ``tp_rank`` of
    ``tp_size``, then the GGUF post-load pass over a model that holds it under its real name.
    Returns (flat-container tags, flat bytes, workspace tags, current tag after the pass)."""
    layer = _gguf_layer(prefix, MERGED)
    fake = SimpleNamespace(
        output_sizes=MERGED, tp_size=tp_size, tp_rank=tp_rank, tp_units=None, tp_family=None
    )
    for sid, rows in enumerate(MERGED):
        MergedColumnParallelLinear.weight_loader(
            fake, layer.qweight_type, torch.tensor(Q8_0, dtype=torch.uint8), sid
        )
        MergedColumnParallelLinear.weight_loader(fake, layer.qweight, _q8(rows, sid + 1), sid)
    root = torch.nn.Module()
    cur = root
    parts = prefix.split(".")
    for part in parts[:-1]:
        child = torch.nn.Module()
        cur.add_module(part, child)
        cur = child
    cur.add_module(parts[-1], layer)
    saver = _Saver(ms.GPU_MEMORY_TYPE_WEIGHTS)
    log: list = []
    # every rank is its own process with its own workspace table: start each pass empty
    with mock.patch.dict(G._DEQUANT_WS, clear=True), mock.patch.dict(
        G._DEQUANT_PEAK_TARGET, clear=True
    ), _chunked_weights_region(saver), _allocations(saver, log):
        loader_mod._process_weights_after_loading_by_layer_chunk(root, torch.device("cpu"))
    flat = [row for row in log if row[0] == "zeros" and row[1] == torch.uint8]
    work = [row for row in log if row[1] == torch.bfloat16]
    return (
        {tag for *_x, tag in flat},
        sum(n for _f, _d, n, _t in flat),
        {tag for *_x, tag in work},
        saver.current,
    )


class _Isolated(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict(G._DEQUANT_WS, clear=True).start()
        mock.patch.dict(G._DEQUANT_PEAK_TARGET, clear=True).start()
        # the in-place ggml_dequantize(out=) schema is a CUDA wheel property;
        # the workspace path is what is under test
        mock.patch.object(G, "_dequant_supports_out", lambda: True).start()


class GroupDTensorParallelShards(_Isolated):
    """Group D: TP3, every rank holds a SHARD of EVERY layer (the 27B D group's weights_0..7)."""

    def test_every_tp_rank_puts_its_shard_in_the_layers_chunk(self):
        for layer_id, want in ((0, "weights_0"), (40, "weights_5"), (63, "weights_7")):
            for rank in range(3):
                with self.subTest(layer=layer_id, rank=rank):
                    flat, nbytes, work, current = _post_load_pass(
                        f"model.layers.{layer_id}.mlp.gate_up_proj", 3, rank
                    )
                    self.assertEqual(flat, {want})
                    # the shard is really a shard: a third of the one-rank container
                    self.assertEqual(nbytes * 3, FULL_FLAT_BYTES)
                    # the shared dequant workspace stays in the base tag (operator order)
                    self.assertEqual(work, {"weights"})
                    self.assertEqual(current, "weights")

    def test_a_tp_shard_materialized_in_load_weights_lands_in_its_layers_chunk(self):
        """A single-shard qweight (down_proj, out_proj, o_proj) gets its bytes in load_weights via
        ``param.materialize`` -- on D with the rank's SHARD shape."""
        for rows in (MERGED[0], MERGED[0] // 3):
            with self.subTest(rows=rows):
                layer = _gguf_layer("model.layers.41.mlp.down_proj", [MERGED[0]])
                saver = _Saver(ms.GPU_MEMORY_TYPE_WEIGHTS)
                log: list = []
                with _chunked_weights_region(saver), _allocations(saver, log):
                    layer.qweight.materialize((rows, Q8_ROW), dtype=torch.uint8)
                self.assertEqual([tag for *_x, tag in log], ["weights_5"])
                self.assertEqual(saver.current, "weights")


class GroupPStageSubset(_Isolated):
    """Group P: a PP stage holds a layer RANGE (PP1 of 42,11,11 = layers 42..52); the global layer
    id decides the tag, so the stage's chunks are weights_5 and weights_6."""

    def test_a_stage_maps_its_layers_by_the_global_layer_id(self):
        for layer_id, want in ((42, "weights_5"), (47, "weights_5"), (48, "weights_6"), (52, "weights_6")):
            with self.subTest(layer=layer_id):
                flat, _n, work, current = _post_load_pass(
                    f"model.layers.{layer_id}.linear_attn.in_proj_qkvz", 1, 0
                )
                self.assertEqual(flat, {want})
                self.assertEqual(work, {"weights"})
                self.assertEqual(current, "weights")


class BothGroupsGetTheSameGeometry(CustomTestCase):
    """The launcher publishes ONE chunk geometry to both groups -- the premise of the two classes
    above. ``build_env`` is the one place the chunk envs are written; P and D are built from the
    same ``chunk_layers`` / ``chunk_count``."""

    def test_build_env_publishes_the_same_chunk_geometry_to_p_and_d(self):
        from sglang.srt.weg2 import launcher as L

        envs = {
            g: L.build_env("/tree", "/venv", "0,1,2", "/store", False, "tag", 8, 8, group=g)
            for g in ("P", "D")
        }
        for g, env in envs.items():
            with self.subTest(group=g):
                self.assertEqual(env.get(ms.WEIGHT_CHUNK_ENV_LAYERS), "8")
                self.assertEqual(env.get(ms.WEIGHT_CHUNK_ENV_COUNT), "8")

    def test_main_builds_p_and_d_from_the_same_chunk_arguments(self):
        import ast
        import inspect

        from sglang.srt.weg2 import launcher as L

        tree = ast.parse(inspect.getsource(L))
        seen = {}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "build_env"):
                continue
            group = next((k.value.value for k in node.keywords
                          if k.arg == "group" and isinstance(k.value, ast.Constant)), None)
            if group in ("P", "D"):
                # positional 7 and 8 are chunk_layers and chunk_count (build_env's signature)
                seen.setdefault(group, set()).add(
                    tuple(ast.unparse(a) for a in node.args[6:8])
                )
        self.assertEqual(set(seen), {"P", "D"}, seen)
        self.assertEqual(seen["P"], seen["D"])
        self.assertEqual(seen["P"], {("chunk_layers", "chunk_count")})


if __name__ == "__main__":
    unittest.main()
