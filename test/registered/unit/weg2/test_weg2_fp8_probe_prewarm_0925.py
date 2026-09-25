"""weg2xsn442 R7: on the RTX 5090 both groups KEPT their load transient pool
(``WEG2-TAG-POOL transient pool KEPT reason=after-load pool=load live=9.1 MiB``,
PP0 and TP0; the 3080 ranks released theirs). Desk, no GPU (CPU stand-ins).

The live 9.1 MiB is the cuBLAS + cuBLASLt workspace (8.125 + 1 MiB): the FP8
capability probe ``fp8_native_gemm_available`` runs ``torch._scaled_mm``, and its
FIRST call happens inside the weights region -- the loader's capability check
``quant_config.needs_device_kernel()`` -> ``fp8_needs_dequant_fallback()``. Since
b434831067 the probe steps out of the tag pool (W106 on PP1), i.e. into the load's
transient pool, which the workspace then pins for the life of the process: the
pool is KEPT and its cached segments stay reserved on the 5090 through every flip.
Only the 5090 has a native fp8 GEMM, so only there does ``_scaled_mm`` get far
enough to create a workspace.

Fix: ``ModelRunner.load_model`` answers the probe ONCE right before it opens the
weights region (``prewarm_fp8_native_gemm_probe``), for checkpoints whose config is
an ``Fp8Config`` -- so the workspace lands in the default pool like in any process,
and the in-region call is a cache hit that allocates nothing. Every other checkpoint
(INT8 compressed-tensors, ModelOpt NVFP4, unquantized) is untouched.
"""

from __future__ import annotations

import inspect
import os

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.layers.quantization import fp8_utils as FU  # noqa: E402
from sglang.srt.layers.quantization.fp8 import Fp8Config  # noqa: E402
from sglang.srt.model_executor import model_runner as MR  # noqa: E402


@pytest.fixture
def probe_recorder(monkeypatch):
    """The probe body, observed: every tensor it allocates and the GEMM it runs
    (where cuBLAS creates its workspace), each with the region state at that moment.
    The 5090 case: a native fp8 GEMM exists, so ``_scaled_mm`` runs."""
    state = {"region_open": False, "events": []}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    real = {n: getattr(torch, n) for n in ("zeros", "ones")}

    def rec(name):
        def alloc(*a, **k):
            state["events"].append((name, state["region_open"]))
            k.pop("device", None)
            return real[name](*a, **k)

        return alloc

    for n in real:
        monkeypatch.setattr(torch, n, rec(n))

    def scaled_mm(*a, **k):
        state["events"].append(("_scaled_mm(workspace)", state["region_open"]))
        return real["zeros"](16, 16)

    monkeypatch.setattr(torch, "_scaled_mm", scaled_mm)
    FU.fp8_native_gemm_available.cache_clear()
    FU.fp8_needs_dequant_fallback.cache_clear()
    yield state
    FU.fp8_native_gemm_available.cache_clear()
    FU.fp8_needs_dequant_fallback.cache_clear()


def _load_order(state, quantization):
    """What ModelRunner.load_model does, in its order: the pre-region step (if this
    tree has one), then the weights region, inside it the loader's capability check."""
    pre = getattr(FU, "prewarm_fp8_native_gemm_probe", None)
    if pre is not None:
        pre(quantization)
    state["region_open"] = True
    try:
        return Fp8Config(is_checkpoint_fp8_serialized=True).needs_device_kernel()
    finally:
        state["region_open"] = False


def test_an_fp8_load_allocates_nothing_inside_the_weights_region(probe_recorder):
    """RED on b434831067: the probe's tensors and its GEMM (the cuBLAS workspace)
    are all created while the weights region is open."""
    assert (
        _load_order(probe_recorder, "fp8") is True
    )  # native fp8 -> a device kernel is needed
    ev = probe_recorder["events"]
    assert ("_scaled_mm(workspace)", False) in ev, ev
    inside = [name for name, region_open in ev if region_open]
    assert inside == [], "allocated inside the weights region: %s" % inside


def test_the_probe_body_runs_once_per_device(probe_recorder):
    _load_order(probe_recorder, "fp8")
    _load_order(probe_recorder, "fp8")
    assert [n for n, _ in probe_recorder["events"]].count("_scaled_mm(workspace)") == 1


@pytest.mark.parametrize(
    "quantization",
    [
        None,
        "",
        "compressed-tensors",
        "w8a8_int8",
        "modelopt",
        "modelopt_fp4",
        "gguf",
        "not-a-method",
    ],
)
def test_every_other_checkpoint_is_untouched(probe_recorder, quantization):
    """INT8 (the serving line), ModelOpt NVFP4, GGUF, unquantized: no prewarm, no
    probe, no allocation -- their load order is byte-identical."""
    pre = getattr(FU, "prewarm_fp8_native_gemm_probe")
    assert pre(quantization) is None
    assert probe_recorder["events"] == []


def test_the_prewarm_answer_is_the_answer_the_loader_reads(probe_recorder):
    """The prewarm asks the loader's own question (Fp8Config.needs_device_kernel()
    = not fp8_needs_dequant_fallback()), so the in-region check reads a cached
    value identical to it -- nothing about the load's decision moves."""
    assert FU.prewarm_fp8_native_gemm_probe("fp8") is True
    probe_recorder["region_open"] = True
    assert Fp8Config(is_checkpoint_fp8_serialized=True).needs_device_kernel() is True
    assert Fp8Config(is_checkpoint_fp8_serialized=False).needs_device_kernel() is True
    assert [n for n, r in probe_recorder["events"] if r] == []


def test_mxfp8_never_probes_so_nothing_is_prewarmed(probe_recorder):
    """needs_device_kernel() answers True for mxfp8 without asking the device, and
    Fp8LinearMethod never reaches the probe for it: a prewarm would ADD a probe
    (and a misleading 'No native fp8 GEMM' line on sm86) that the load never ran."""
    assert (
        Fp8Config(
            is_checkpoint_fp8_serialized=True, use_mxfp8=True
        ).needs_device_kernel()
        is True
    )
    assert probe_recorder["events"] == []
    assert FU.prewarm_fp8_native_gemm_probe("mxfp8") is None
    assert probe_recorder["events"] == []


@pytest.mark.parametrize(
    "env", ["SGLANG_FORCE_FP8_DEQUANT", "SGLANG_DETERMINISTIC_FP8_GEMM"]
)
def test_an_env_that_answers_first_keeps_the_probe_unrun(
    probe_recorder, monkeypatch, env
):
    """Where the loader's check is answered by an env switch before the probe, the
    prewarm answers the same way and runs no probe either. (The determinism switch
    only answers on sm80..88: the card is an sm86 stand-in for that case.)"""
    from sglang.srt.utils.common import clear_per_device_gate_caches

    monkeypatch.setenv(env, "1")
    monkeypatch.setattr(FU, "is_cuda", lambda: True)
    monkeypatch.setattr(FU, "get_device_capability", lambda device_id=None: (8, 6))
    clear_per_device_gate_caches()
    try:
        assert FU.prewarm_fp8_native_gemm_probe("fp8") is False
        assert _load_order(probe_recorder, "fp8") is False
        assert probe_recorder["events"] == []
    finally:
        clear_per_device_gate_caches()


def test_no_cuda_means_no_prewarm(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert FU.prewarm_fp8_native_gemm_probe("fp8") is None


def test_load_model_prewarms_right_before_it_opens_the_weights_region():
    src = inspect.getsource(MR.ModelRunner.load_model)
    assert "prewarm_fp8_native_gemm_probe(self.model_config.quantization)" in src
    pre = src.index("prewarm_fp8_native_gemm_probe(self.model_config.quantization)")
    region = src.index("with weights_region(")
    assert pre < region
    # nothing else opens a region between the two
    assert "memory_saver_adapter.region(" not in src[pre:region]


def test_the_in_region_probe_keeps_its_step_out_as_the_belt():
    """The b434831067 step-out stays: a probe that is ever reached uncached inside
    the region (a config this prewarm does not cover) still lands in the load pool,
    never in the weights tag (W106)."""
    body = inspect.getsource(FU.fp8_native_gemm_available)
    assert '_outside_weg2_tag_pool("fp8-native-gemm-probe")' in body
