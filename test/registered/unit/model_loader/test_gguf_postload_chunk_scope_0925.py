"""GGUF's post-load pass allocates under its LAYER's weight chunk tag (#1233 rule).

Boot weg2rc5gg (27B line, RC5 5f13f1aad9, 2026-09-25), first flip D->P:

    [05:41:05 PP0] WEG2-RESUME-PTRATTR tag=weights_5 ... own_tag=weights_5 mapped=22
        unmapped=8 first_unmapped=model.layers.40.linear_attn.in_proj_qkvz.qweight
        (44564480B,type=0)
    [05:41:08 PP1] WEG2-BAR1 mapped lane=p0 phase=collect seq=0-weights_5 ...
    Fatal Python error: Segmentation fault
      weight_exchange_transport.py:852 memcpy_async <- bar1_lanes.py:1112
      _run_bar1_tag_streamed (cuMemcpyAsync)
    [05:41:12] Subprocess scheduler_1 (pid=527843) is gone: killed by SIGSEGV

Mechanism. ``GGUFLinearMethod.process_weights_after_loading`` builds the flat
qweight container (``_create_flat_weight_param``) -- the bulk of every layer's
bytes -- and grows the dequant workspace. ``DefaultModelLoader`` runs that pass
inside ``weight_chunk_scope(layer_id_from_module_name(name))``, so a post-load
allocation lands in the chunk tag its layer is paused and resumed under.
``GGUFModelLoader`` ran it bare: everything was born under the BASE ``weights``
tag (WEG2-XCHG-COVER on PP0: ``weights`` tms 7766 MiB against a walk of 521;
``weights_0..5`` tms 122-522 against 477-1804). The wake resumes the chunks first
and the base tag last, so the collect of ``weights_5`` wrote into pages no tag
had remapped yet, and the driver took the unknown destination for pageable host
memory.

The pass runs for real here: the real ``weight_chunk_scope`` and chunk geometry
from the real env variables; only the memory saver's C entry point (which tag is
current) and the per-tag CUDA pool are stand-ins, since there is no saver on a
desk.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import os
import textwrap
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


class _Saver:
    """The saver's C entry point: which tag is current for the next allocation."""

    def __init__(self, base: str):
        self.current = base

    def tms_set_current_tag(self, tag: bytes) -> None:
        self.current = tag.decode("utf-8")


class _Recording:
    """A quant method whose post-load pass records the tag it allocates under."""

    def __init__(self, saver: _Saver, seen: dict, name: str):
        self.saver, self.seen, self.name = saver, seen, name

    def process_weights_after_loading(self, module) -> None:
        self.seen[self.name] = self.saver.current


def _gguf_like_model(saver: _Saver, seen: dict) -> torch.nn.Module:
    names = [
        "model.embed_tokens",
        "model.layers.0.linear_attn.in_proj_qkvz",
        "model.layers.40.linear_attn.in_proj_qkvz",
        "model.layers.63.mlp.gate_up_proj",
        "lm_head",
    ]
    root = torch.nn.Module()
    for name in names:
        parent = root
        parts = name.split(".")
        for part in parts[:-1]:
            if not hasattr(parent, part):
                parent.add_module(part, torch.nn.Module())
            parent = getattr(parent, part)
        leaf = torch.nn.Module()
        leaf.quant_method = _Recording(saver, seen, name)
        parent.add_module(parts[-1], leaf)
    return root


@contextlib.contextmanager
def _chunked_weights_region(saver: _Saver):
    """The launcher's chunk geometry (8 layers per chunk, 8 chunks -- the 64-layer
    27B) and model_runner's open BASE weights region, as on the boot."""
    env = {ms.WEIGHT_CHUNK_ENV_LAYERS: "8", ms.WEIGHT_CHUNK_ENV_COUNT: "8"}
    with mock.patch.dict(os.environ, env), ms.weights_region_tag(
        ms.GPU_MEMORY_TYPE_WEIGHTS
    ), mock.patch.object(ms, "_tms_cdll_in_region", lambda: saver), mock.patch.object(
        ms, "tag_pool_scope", lambda tag: contextlib.nullcontext(tag)
    ):
        yield


class GgufPostLoadChunkScope(CustomTestCase):
    def test_every_layer_module_allocates_under_its_chunk_tag(self):
        helper = getattr(
            loader_mod, "_process_weights_after_loading_by_layer_chunk", None
        )
        self.assertIsNotNone(
            helper,
            "model_loader.loader has no chunk-scoped post-load pass for GGUF -- the "
            "flat qweight containers are born under the base weights tag",
        )
        saver = _Saver(ms.GPU_MEMORY_TYPE_WEIGHTS)
        seen: dict = {}
        model = _gguf_like_model(saver, seen)
        with _chunked_weights_region(saver):
            helper(model, torch.device("cpu"))
        self.assertEqual(
            seen,
            {
                "model.embed_tokens": "weights",
                "model.layers.0.linear_attn.in_proj_qkvz": "weights_0",
                "model.layers.40.linear_attn.in_proj_qkvz": "weights_5",
                "model.layers.63.mlp.gate_up_proj": "weights_7",
                "lm_head": "weights",
            },
        )
        # the base tag is back once the pass is done
        self.assertEqual(saver.current, "weights")

    def test_gguf_load_model_runs_its_post_load_pass_through_the_chunked_helper(self):
        """The GGUF loader's own call site (the pass is inline in load_model): no
        bare ``quant_method.process_weights_after_loading(module)`` left in it."""
        src = textwrap.dedent(inspect.getsource(loader_mod.GGUFModelLoader.load_model))
        calls = [
            node for node in ast.walk(ast.parse(src)) if isinstance(node, ast.Call)
        ]
        names = {
            getattr(c.func, "id", None) or getattr(c.func, "attr", None) for c in calls
        }
        self.assertIn("_process_weights_after_loading_by_layer_chunk", names)
        self.assertNotIn("process_weights_after_loading", names)

    def test_no_chunking_no_change(self):
        """Chunk envs unset (every boot without the flip's chunking): the pass
        runs exactly as before, nothing is retagged."""
        helper = getattr(
            loader_mod, "_process_weights_after_loading_by_layer_chunk", None
        )
        self.assertIsNotNone(helper)
        saver = _Saver(ms.GPU_MEMORY_TYPE_WEIGHTS)
        seen: dict = {}
        model = _gguf_like_model(saver, seen)
        env = {ms.WEIGHT_CHUNK_ENV_LAYERS: "", ms.WEIGHT_CHUNK_ENV_COUNT: ""}
        with mock.patch.dict(os.environ, env), mock.patch.object(
            ms, "_tms_cdll_in_region", lambda: saver
        ):
            helper(model, torch.device("cpu"))
        self.assertEqual(set(seen.values()), {"weights"})
        self.assertEqual(len(seen), 5)


# ---------------------------------------------------------------------------
# The real GGUF weight path on CPU tensors (the fixture shape of
# test_weg2_gguf_flat_declaration_g1): which tag is current at each
# ALLOCATION of a qweight -- the flat container, a single-shard qweight -- and of
# the shared dequant workspace.
# ---------------------------------------------------------------------------
K_IN = 64
Q8_0 = 8
Q8_ROW = K_IN // 32 * 34
MERGED = [12, 6]


def _q8(rows, salt):
    return torch.tensor(
        [
            [(salt * 31 + r * 7 + b * 3) % 251 + 1 for b in range(Q8_ROW)]
            for r in range(rows)
        ],
        dtype=torch.uint8,
    )


def _gguf_layer(prefix, sizes):
    layer = torch.nn.Module()
    layer.prefix = prefix  # LinearBase sets it before create_weights
    method = G.GGUFLinearMethod(G.GGUFConfig([]))
    method.create_weights(layer, K_IN, list(sizes), K_IN, sum(sizes), torch.bfloat16)
    layer.quant_method = method
    return layer


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

    with mock.patch.object(torch, "empty", empty), mock.patch.object(
        torch, "zeros", zeros
    ):
        yield


class RealGgufAllocations(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict(G._DEQUANT_WS, clear=True).start()
        mock.patch.dict(G._DEQUANT_PEAK_TARGET, clear=True).start()
        # the in-place ggml_dequantize(out=) schema is a CUDA wheel property;
        # the workspace path is what is under test
        mock.patch.object(G, "_dequant_supports_out", lambda: True).start()

    def test_single_shard_qweight_is_materialized_in_its_layer_chunk(self):
        """A non-merged qweight (down_proj, out_proj, o_proj) gets its bytes in
        load_weights via param.materialize -- after construction's chunk scope."""
        layer = _gguf_layer("model.layers.41.mlp.down_proj", [MERGED[0]])
        saver = _Saver(ms.GPU_MEMORY_TYPE_WEIGHTS)
        log: list = []
        with _chunked_weights_region(saver), _allocations(saver, log):
            layer.qweight.materialize((MERGED[0], Q8_ROW), dtype=torch.uint8)
        self.assertEqual(
            [(d, n, tag) for _f, d, n, tag in log],
            [(torch.uint8, MERGED[0] * Q8_ROW, "weights_5")],
        )
        self.assertEqual(saver.current, "weights")

    def test_flat_container_in_the_chunk_workspace_in_the_base_tag(self):
        """The merged qweight's flat container is allocated in the post-load
        pass under weights_5; the dequant workspace ONCE, afterwards, under the
        base tag (Operator order: _DEQUANT_WS stays in the base scope)."""
        layer = _gguf_layer("model.layers.40.mlp.gate_up_proj", MERGED)
        fake = SimpleNamespace(
            output_sizes=MERGED, tp_size=1, tp_rank=0, tp_units=None, tp_family=None
        )
        for sid, rows in enumerate(MERGED):
            MergedColumnParallelLinear.weight_loader(
                fake, layer.qweight_type, torch.tensor(Q8_0, dtype=torch.uint8), sid
            )
            MergedColumnParallelLinear.weight_loader(
                fake, layer.qweight, _q8(rows, sid + 1), sid
            )
        root = torch.nn.Module()
        model = torch.nn.Module()
        layers = torch.nn.Module()
        l40 = torch.nn.Module()
        mlp = torch.nn.Module()
        root.add_module("model", model)
        model.add_module("layers", layers)
        layers.add_module("40", l40)
        l40.add_module("mlp", mlp)
        mlp.add_module("gate_up_proj", layer)
        saver = _Saver(ms.GPU_MEMORY_TYPE_WEIGHTS)
        log: list = []
        with _chunked_weights_region(saver), _allocations(saver, log):
            loader_mod._process_weights_after_loading_by_layer_chunk(
                root, torch.device("cpu")
            )
        flat = [row for row in log if row[0] == "zeros" and row[1] == torch.uint8]
        work = [row for row in log if row[1] == torch.bfloat16]
        self.assertTrue(flat, log)
        self.assertEqual({tag for *_x, tag in flat}, {"weights_5"})
        self.assertEqual(
            work, [("empty", torch.bfloat16, max(MERGED) * K_IN, "weights")]
        )
        self.assertEqual(len(G._DEQUANT_WS), 1)
        self.assertEqual(saver.current, "weights")


class NonGgufUnchanged(CustomTestCase):
    def test_a_plain_uninitialized_parameter_is_not_retagged(self):
        saver = _Saver(ms.GPU_MEMORY_TYPE_WEIGHTS)
        log: list = []
        param = torch.nn.parameter.UninitializedParameter(requires_grad=False)
        with _chunked_weights_region(saver), _allocations(saver, log):
            param.materialize((4, 4), dtype=torch.float32)
        self.assertEqual([tag for *_x, tag in log], ["weights"])

    def test_the_workspace_is_allocated_at_once_outside_the_deferral(self):
        with mock.patch.dict(G._DEQUANT_WS, clear=True), mock.patch.dict(
            G._DEQUANT_PEAK_TARGET, clear=True
        ), mock.patch.object(G, "_dequant_supports_out", lambda: True):
            G._reserve_dequant_workspace(128, torch.bfloat16, torch.device("cpu"))
            self.assertEqual([b.numel() for b in G._DEQUANT_WS.values()], [128])

    def test_the_default_loader_does_not_take_the_gguf_pass(self):
        src = inspect.getsource(loader_mod.DefaultModelLoader)
        self.assertNotIn("_process_weights_after_loading_by_layer_chunk", src)
        self.assertNotIn("dequant_workspace_deferred", src)


if __name__ == "__main__":
    unittest.main()
