# SPDX-License-Identifier: Apache-2.0
"""``DeepseekV4ForCausalLM.load_weights`` must stay a generator on GGUF (#391).

With ``SGLANG_OPT_FP8_WO_A_GEMM`` off -- which is the only reachable setting
for a GGUF checkpoint, since forcing it on raises ``NotImplementedError`` in
``DeepseekV4AttentionMHC`` -- ``load_weights`` used to open with::

    weights = list(weights)
    exists_wo_a_scale = any(n.endswith(".wo_a.scale") for n, t in weights)

i.e. it drained the whole weight stream to answer one boolean. On a large GGUF
that materializes every post-repack tensor in host RAM before the first weight
loader runs (boot 6: linear 0.7 GiB/s to 74.8 GiB with no ``weight_loader``
frame in any py-spy stack), so every downstream streaming or per-rank sharding
step is fed by an already-fully-materialized list.

The answer is also a constant on that route: a packed GGUF ``wo_a`` carries no
separate block-scale tensor by construction. So the GGUF branch skips the
lookahead and instead checks the claim per tensor as the stream flows
(``_reject_wo_a_scale_on_gguf``) -- if a ``.wo_a.scale`` ever does appear it is
refused by name rather than silently dropped.

The non-GGUF route is untouched: it holds mmapped tensors, the list is cheap,
and the flag can genuinely be true. The last class pins that.

No GPU and no checkpoint: the weight tensors are meta tensors and the model is
a stub carrying only the attributes ``load_weights`` reads.
"""

import types
import unittest
import weakref
from typing import List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.models import deepseek_v4
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


#: Enough tensors that "one at a time" and "all at once" are unmistakably apart.
_N_TENSORS = 64


class _FakeQuantConfig:
    """Only ``get_name()`` is read (``is_gguf_quant_config``)."""

    def __init__(self, name: str):
        self._name = name

    def get_name(self) -> str:
        return self._name


class _RecordingParam:
    """Stands in for an ``nn.Parameter`` with a ``weight_loader`` attribute."""

    def __init__(self, calls: List[int], probe: "_StreamProbe"):
        self._calls = calls
        self._probe = probe

    def weight_loader(self, param, loaded_weight, *args, **kwargs) -> None:
        # How many tensors the source stream had produced by the time this
        # tensor reached its loader. Streaming => i+1. Drained => the whole
        # stream, on every single call.
        self._calls.append(self._probe.yielded)


class _StreamProbe:
    """A lazy (name, tensor) source that records production and liveness.

    ``peak_live`` is the high-water mark of tensors produced by this probe that
    are still reachable -- the direct measurement of materialization, not a
    proxy for it. A drained stream pins every tensor in a list, so ``peak_live``
    reaches the stream length; a streamed one keeps a couple of loop locals
    alive at a time.
    """

    def __init__(self, names: List[str]):
        self._names = names
        self.yielded = 0
        self.peak_live = 0
        self._refs: List[weakref.ref] = []

    def _note_live(self) -> None:
        live = sum(1 for ref in self._refs if ref() is not None)
        self.peak_live = max(self.peak_live, live)

    def stream(self):
        for name in self._names:
            tensor = torch.empty(4, device="meta")
            self._refs.append(weakref.ref(tensor))
            self.yielded += 1
            self._note_live()
            yield name, tensor
            del tensor
            self._note_live()


def _probe_names(n: int = _N_TENSORS) -> List[str]:
    # Names that land in the plain else-branch of the loader loop: no stacked
    # mapping (gate/up/down_proj), no expert mapping, no compressor, and not
    # one of the wq_a/wkv suffixes the wqkv-a fusion buffers.
    return [f"model.layers.{i}.self_attn.wo_b.weight" for i in range(n)]


def _make_stub(
    quant_name: Optional[str],
    params: dict,
) -> types.SimpleNamespace:
    """A stub carrying exactly the attributes ``load_weights`` reads."""
    stub = types.SimpleNamespace()
    stub.config = types.SimpleNamespace(
        num_hidden_layers=_N_TENSORS, n_routed_experts=2
    )
    stub.quant_config = None if quant_name is None else _FakeQuantConfig(quant_name)
    stub.num_fused_shared_experts = 0
    stub.model = types.SimpleNamespace()
    stub.pp_group = types.SimpleNamespace(is_first_rank=True, is_last_rank=True)
    stub.named_parameters = lambda: params.items()
    stub.remap_weight_name_to_dpsk_hf_format = (
        DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format
    )
    stub.post_load_weights = lambda **kwargs: None
    stub._prewarm_mhc_pre_kernels = lambda: None
    return stub


def _run_load(
    quant_name: Optional[str],
    names: List[str],
) -> Tuple[_StreamProbe, List[int]]:
    """Drive the real ``load_weights`` over a probed stream on the given route."""
    probe = _StreamProbe(names)
    calls: List[int] = []
    params = {
        name: _RecordingParam(calls, probe)
        for name in names
        if not name.endswith(".wo_a.scale")
    }
    stub = _make_stub(quant_name, params)
    with envs.SGLANG_OPT_FP8_WO_A_GEMM.override(False):
        DeepseekV4ForCausalLM.load_weights(stub, probe.stream())
    return probe, calls


class TestGgufStreamIsNotMaterialized(CustomTestCase):
    """(b) generator preservation, with the unfixed behaviour as the arm."""

    def test_gguf_stream_is_consumed_one_tensor_at_a_time(self):
        names = _probe_names()
        probe, calls = _run_load("gguf", names)

        self.assertEqual(len(calls), len(names), "every probe tensor must load")
        # Streaming: tensor i reaches its loader when exactly i+1 have been
        # produced. This is the whole claim, stated per tensor.
        self.assertEqual(calls, list(range(1, len(names) + 1)))
        # And nothing accumulates behind it.
        self.assertLessEqual(
            probe.peak_live,
            4,
            f"GGUF stream held {probe.peak_live} tensors alive at once; "
            "expected a couple of loop locals",
        )

    def test_can_fail_arm_the_lookahead_route_does_materialize(self):
        """The same probe on the route that still takes the lookahead.

        This is the unfixed behaviour, kept executable: a non-GGUF quant config
        with the same stream takes ``weights = list(weights)``, so the first
        loader call already sees the entire stream and every tensor is pinned.
        If this ever starts looking like the test above, the assertions above
        have stopped being able to fail.
        """
        names = _probe_names()
        probe, calls = _run_load("fp8", names)

        self.assertEqual(len(calls), len(names))
        self.assertEqual(
            calls,
            [len(names)] * len(names),
            "the lookahead route is expected to drain before loading",
        )
        self.assertEqual(probe.peak_live, len(names))


class TestGgufWoAScaleIsRefused(CustomTestCase):
    """(a) the skip must not silently swallow a real dequant input."""

    def test_planted_wo_a_scale_is_refused_by_name(self):
        names = _probe_names(8)
        planted = "model.layers.3.self_attn.wo_a.scale"
        names.insert(4, planted)

        with self.assertRaises(ValueError) as ctx:
            _run_load("gguf", names)

        message = str(ctx.exception)
        self.assertIn("unexpected .wo_a.scale on a GGUF checkpoint", message)
        self.assertIn(planted, message)

    def test_refusal_fires_mid_stream_not_after_a_drain(self):
        """The check is per tensor, so it must not wait for the stream to end."""
        names = _probe_names(64)
        names.insert(4, "model.layers.3.self_attn.wo_a.scale")

        probe = _StreamProbe(names)
        calls: List[int] = []
        params = {
            name: _RecordingParam(calls, probe)
            for name in names
            if not name.endswith(".wo_a.scale")
        }
        stub = _make_stub("gguf", params)
        with envs.SGLANG_OPT_FP8_WO_A_GEMM.override(False):
            with self.assertRaises(ValueError):
                DeepseekV4ForCausalLM.load_weights(stub, probe.stream())

        self.assertEqual(probe.yielded, 5, "stream ran past the offending tensor")
        self.assertEqual(calls, [1, 2, 3, 4])

    def test_clean_gguf_stream_loads(self):
        """No ``.wo_a.scale`` in a real export, so the guard is transparent."""
        names = _probe_names(8)
        _, calls = _run_load("gguf", names)
        self.assertEqual(calls, list(range(1, 9)))


class TestNonGgufRouteUnchanged(CustomTestCase):
    """(c) the safetensors route keeps its lookahead and its dequant."""

    def test_fp8_stream_with_wo_a_scale_still_triggers_the_dequant(self):
        names = _probe_names(8)
        scale_name = "model.layers.3.self_attn.wo_a.scale"
        names.insert(4, scale_name)

        seen: List[List[str]] = []

        def _spy(weights):
            materialized = list(weights)
            seen.append([n for n, _ in materialized])
            # Stand in for the real dequant: it consumes the scale and emits
            # the dense weight, so the scale never reaches the loader loop.
            for name, tensor in materialized:
                if not name.endswith(".wo_a.scale"):
                    yield name, tensor

        original = deepseek_v4._dequant_fp8_wo_a
        deepseek_v4._dequant_fp8_wo_a = _spy
        try:
            _, calls = _run_load("fp8", names)
        finally:
            deepseek_v4._dequant_fp8_wo_a = original

        self.assertEqual(len(seen), 1, "the dequant must be reached exactly once")
        self.assertIn(scale_name, seen[0])
        self.assertEqual(len(seen[0]), len(names), "it still receives the full list")
        self.assertEqual(len(calls), len(names) - 1)

    def test_unquantized_route_also_keeps_the_lookahead(self):
        """``quant_config is None`` is not GGUF; behaviour must not change."""
        names = _probe_names(8)
        probe, calls = _run_load(None, names)
        self.assertEqual(calls, [len(names)] * len(names))
        self.assertEqual(probe.peak_live, len(names))


if __name__ == "__main__":
    unittest.main()
