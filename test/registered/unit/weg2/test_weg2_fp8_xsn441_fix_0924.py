"""weg2xsn441 (FP8 on the flip, tree fdade8572f) died at its first flip pair.
Desk, no GPU. Each test here is RED on fdade8572f and GREEN with the fix.

Root B (the fatal one): ``W106 Weg2XchgWakeSourceGapRefused group=P rank=1
tag=weights expected_bytes=2097152`` -> W29 on the whole group. The loader's
capability check (``quant_config.needs_device_kernel()`` ->
``fp8_needs_dequant_fallback`` -> ``fp8_native_gemm_available``) probes by
ALLOCATING three small fp8 tensors, inside the base ``weights`` region (log
order on PP1: ``WEG2-TAG-POOL tag=weights pool=new`` -> ``Detected fp8
checkpoint.`` -> ``No native fp8 GEMM on this device`` -> ``WEG2-TAG-POOL
tag=weights_4``). A private tag pool never hands a freed block back, so a 2 MiB
small-block segment stayed under tag ``weights`` (``WEG2-XCHG-RESIDENT
tag=weights mib=2.0``; INT8 weg2rc1: ``mib=0.0``). On P's middle stage that tag
holds nothing else, so the plan has no descriptor for it.

Root A (the VRAM pressure): the FP8 Marlin repack left the checkpoint-format
fp8 tensors and its working set as dead blocks of the tag pools, which the
flip pauses and resumes with every tag: tms minus walk per rank P
4083/2182/1725 MiB, D 3759/1533/1664 MiB (INT8 weg2rc1: 56-292 MiB) -- so the
on-card staging of lane c0 (1552 MiB) found no unpromised VRAM ("staging
refused by the card's VRAM credit") and P's next staging hit ``cudaMalloc
rc=2``. The wNa16 drafter already runs its repack outside the tag pool (H39,
SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL); the FP8 linear now does the same.

Class check before the metal: the dry run refuses a census measured on another
checkpoint (W161) -- xsn441 ran on the INT8 census of weg2xsn246.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import json
import os
import types

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.layers.quantization import fp8 as F  # noqa: E402
from sglang.srt.layers.quantization import fp8_utils as FU  # noqa: E402
from sglang.srt.layers.quantization import marlin_utils_fp8 as MU  # noqa: E402
from sglang.srt.managers import weg2_memory_saver as MS  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402


def _recording_outside(state):
    @contextlib.contextmanager
    def fake(reason: str = "", *, into: str = "load"):
        state.setdefault("calls", []).append((reason, into))
        state["inside"] = state.get("inside", 0) + 1
        try:
            yield True
        finally:
            state["inside"] -= 1

    return fake


# -- root B: the capability probe is not born in the weights tag pool -------------


def test_the_fp8_native_gemm_probe_allocates_outside_the_weg2_tag_pool(monkeypatch):
    state = {}
    monkeypatch.setattr(MS, "outside_tag_pool", _recording_outside(state))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    seen = []
    real = {n: getattr(torch, n) for n in ("zeros", "ones")}

    def rec(name):
        def alloc(*a, **k):
            seen.append((name, state.get("inside", 0) > 0))
            k.pop("device", None)
            return real[name](*a, **k)
        return alloc

    for n in real:
        monkeypatch.setattr(torch, n, rec(n))

    def no_fp8_gemm(*a, **k):  # sm86: torch refuses, exactly as on PP1
        raise RuntimeError("torch._scaled_mm is only supported on CUDA devices with "
                           "compute capability >= 9.0 or 8.9")

    monkeypatch.setattr(torch, "_scaled_mm", no_fp8_gemm)
    assert FU.fp8_native_gemm_available.__wrapped__(0) is False
    assert [n for n, _ in seen] == ["zeros", "zeros", "ones"]
    assert all(inside for _, inside in seen), seen
    assert state["calls"] == [("fp8-native-gemm-probe", "load")]


def test_no_per_device_capability_probe_allocates_inside_a_tag_pool():
    """The CLASS, statically: every ``@per_device_gate`` probe under sglang/srt
    that allocates on a device does it stepped out of the weg2 tag pool (these
    run from the loader, inside the weights region, before any chunk scope)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(F.__file__)))
    root = os.path.dirname(root)  # .../sglang/srt
    alloc = {"zeros", "empty", "ones", "full", "randn", "rand", "arange", "tensor",
             "zeros_like", "empty_like", "ones_like"}
    offenders, probes = [], []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path) as fh:
                    tree = ast.parse(fh.read())
            except (OSError, SyntaxError, ValueError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                if not any("per_device_gate" in ast.unparse(d) for d in node.decorator_list):
                    continue
                dev = [c for c in ast.walk(node)
                       if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                       and c.func.attr in alloc and any(k.arg == "device" for k in c.keywords)]
                if not dev:
                    continue
                probes.append(node.name)
                body = ast.unparse(node)
                if "_outside_weg2_tag_pool(" not in body and "outside_tag_pool(" not in body:
                    offenders.append("%s:%s" % (os.path.relpath(path, root), node.name))
    assert "fp8_native_gemm_available" in probes
    assert offenders == [], offenders


# -- root A: the FP8 Marlin repack runs outside the tag pool (H39) ---------------


def _method(use_marlin=True, block=True):
    m = object.__new__(F.Fp8LinearMethod)
    m.use_marlin = use_marlin
    m.use_mxfp8 = False
    m.block_quant = block
    m.is_checkpoint_fp8_serialized = True
    m.quant_config = types.SimpleNamespace(weight_block_size=[128, 128],
                                           activation_scheme="dynamic")
    m.validate_block_quant_shapes = lambda *a, **k: None
    return m


def _publish_qwen27b(monkeypatch):
    """UNIFY S2/S7: the switch's default is the published profile's (qwen27b
    off, the 27B line's; nextflash on); the 27B assertions below run under it."""
    from sglang.srt.weg2 import form as _F

    monkeypatch.setenv(_F.FORM_ENV, _F.Weg2Form(
        arch="dense", experts="none", draft="dflash", p_draft="none", kv="paged_dcp",
        flip="family", vision="off", profile="qwen27b", model="m").env_value())



def test_h39_arms_only_for_a_marlin_rank_under_the_switch(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL", "1")
    assert _method(use_marlin=True)._weg2_marlin_outside_pool() is True
    assert _method(use_marlin=False)._weg2_marlin_outside_pool() is False
    mx = _method()
    mx.use_mxfp8 = True
    assert mx._weg2_marlin_outside_pool() is False
    monkeypatch.delenv("SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL")
    _publish_qwen27b(monkeypatch)
    assert _method(use_marlin=True)._weg2_marlin_outside_pool() is False


def _ckpt_layer(N=128, K=256):
    layer = torch.nn.Module()
    layer.output_size_per_partition = N
    layer.input_size_per_partition = K
    layer.orig_dtype = torch.bfloat16
    layer.weight_block_size = [128, 128]
    layer.weight = torch.nn.Parameter(torch.zeros(N, K, dtype=torch.float8_e4m3fn), requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(torch.ones(N // 128, K // 128), requires_grad=False)
    return layer


def test_the_checkpoint_weight_dies_before_its_survivor_is_born(monkeypatch):
    """No separate checkpoint pool (it would hold every layer's fp8 weight next
    to all survivors until the end of the load -- twice the weights at the
    peak): the checkpoint weight is released right before the repacked weight
    is allocated in the same tag pool, which then takes its block (N*K bytes
    both). Without the hook the order is the old one."""
    seen = {}

    def fake_repack(b_q_weight, perm, size_k, size_n, num_bits):
        seen["ckpt_weight_alive"] = seen["layer"].weight is not None
        return torch.zeros(size_k // 16, size_n * 4, dtype=torch.int32)

    monkeypatch.setattr(MU, "marlin_make_workspace", lambda device, m=1: torch.zeros(68, dtype=torch.int))
    monkeypatch.setattr(MU, "gptq_marlin_repack", fake_repack, raising=False)
    for hook, alive in ((contextlib.nullcontext, False), (None, True)):
        layer = _ckpt_layer()
        seen["layer"] = layer
        MU.prepare_fp8_layer_for_marlin(layer, size_k_first=False, born_in=hook)
        assert seen["ckpt_weight_alive"] is alive, hook
        assert layer.weight.dtype == torch.int32 and tuple(layer.weight.shape) == (256 // 16, 128 * 4)
        # N*K fp8 bytes == [K/16, 4N] int32 bytes: the freed block fits exactly
        assert layer.weight.numel() * 4 == 128 * 256


def test_the_post_load_pass_steps_out_and_hands_the_survivor_hook_down(monkeypatch):
    # the hook reaches the repack (read before the method is patched below)
    src = inspect.getsource(F.Fp8LinearMethod._process_weights_after_loading)
    assert "prepare_fp8_layer_for_marlin(layer, not self.block_quant, born_in=born_in)" in src
    monkeypatch.setenv("SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL", "1")
    state = {}
    monkeypatch.setattr(MS, "outside_tag_pool", _recording_outside(state))
    got = {}

    def inner(self, layer, born_in=None):
        got["inside"] = state.get("inside", 0) > 0
        got["born_in"] = born_in

    monkeypatch.setattr(F.Fp8LinearMethod, "_process_weights_after_loading", inner)
    _method().process_weights_after_loading(torch.nn.Module())
    assert got == {"inside": True, "born_in": MS.back_into_tag_pool}
    assert state["calls"] == [("fp8-dense-marlin", "load")]
    # off: no step, no hook
    monkeypatch.delenv("SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL")
    _publish_qwen27b(monkeypatch)
    state.clear()
    _method().process_weights_after_loading(torch.nn.Module())
    assert got == {"inside": False, "born_in": None} and "calls" not in state


def test_the_marlin_survivors_are_born_under_the_hook_and_nothing_else(monkeypatch):
    """prepare_fp8_layer_for_marlin(born_in=...): the workspace, the repacked
    weight and the final scales are allocated under the hook; the fp8->int32
    packing, its transpose and the scale intermediates are not."""
    depth = {"n": 0}
    births = []

    @contextlib.contextmanager
    def born_in():
        depth["n"] += 1
        try:
            yield True
        finally:
            depth["n"] -= 1

    def fake_workspace(device, max_blocks_per_sm=1):
        births.append(("workspace", depth["n"] > 0))
        return torch.zeros(68, dtype=torch.int)

    def fake_repack(b_q_weight, perm, size_k, size_n, num_bits):
        births.append(("repack", depth["n"] > 0))
        return torch.zeros(size_k // 16, size_n * 4, dtype=torch.int32)  # Marlin 8-bit: [K/16, 4N]

    real_empty = torch.empty

    def rec_empty(*a, **k):
        births.append(("empty", depth["n"] > 0))
        return real_empty(*a, **k)

    monkeypatch.setattr(MU, "marlin_make_workspace", fake_workspace)
    monkeypatch.setattr(MU, "gptq_marlin_repack", fake_repack, raising=False)
    K, N = 256, 128
    layer = _ckpt_layer(N, K)
    monkeypatch.setattr(torch, "empty", rec_empty)
    MU.prepare_fp8_layer_for_marlin(layer, size_k_first=False, born_in=born_in)
    monkeypatch.setattr(torch, "empty", real_empty)
    assert ("workspace", True) in births and ("repack", True) in births
    # the scales' survivor copy is the one torch.empty under the hook
    assert ("empty", True) in births
    assert all(inside for what, inside in births if what in ("workspace", "repack"))
    assert tuple(layer.weight_scale.shape) == (K // 128, N)
    assert not hasattr(layer, "weight_scale_inv")


def test_without_the_hook_nothing_is_copied(monkeypatch):
    monkeypatch.setattr(MU, "marlin_make_workspace", lambda device, m=1: torch.zeros(68, dtype=torch.int))
    monkeypatch.setattr(MU, "gptq_marlin_repack",
                        lambda b_q_weight, perm, size_k, size_n, num_bits: torch.zeros(size_k // 16, size_n * 4, dtype=torch.int32),
                        raising=False)
    calls = []
    real = MU._as_survivor

    def spy(t, born_in):
        calls.append(born_in)
        return real(t, born_in)

    monkeypatch.setattr(MU, "_as_survivor", spy)
    layer = torch.nn.Module()
    layer.output_size_per_partition, layer.input_size_per_partition = 128, 256
    layer.orig_dtype = torch.bfloat16
    layer.weight_block_size = [128, 128]
    layer.weight = torch.nn.Parameter(torch.zeros(128, 256, dtype=torch.float8_e4m3fn), requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(torch.ones(1, 2), requires_grad=False)
    MU.prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    assert calls == [None]


# -- the class check in the dry run: a census of THIS checkpoint (W161) ------------


def _census(tmp_path, provenance="", checkpoint=None):
    blob = {"cards": {}, "waves": [], "provenance": provenance}
    if checkpoint:
        blob["checkpoint"] = checkpoint
    p = tmp_path / "census.json"
    p.write_text(json.dumps(blob))
    return str(p)


def _p_log(tmp_path, stem, model):
    (tmp_path / (stem + ".P.log")).write_text(
        "[2026-09-17 15:37:40] server_args=ServerArgs(model_path='%s', tokenizer_path='x')\n" % model)


def test_a_census_measured_on_another_checkpoint_is_refused_in_the_dry_run(tmp_path):
    stem = "boot_weg2_weg2xsn246_27198a2711_0917_153729"
    _p_log(tmp_path, stem, "/models/Qwen3.8-27B-INT8-gdncov-vocabembed")
    c = _census(tmp_path, provenance="ring-table boot %s; selection: pinned" % stem)
    with pytest.raises(L.Weg2LaunchRefused) as e:
        L.census_checkpoint_decision(c, "/models/Qwen3.8-27B-FP8", [str(tmp_path)], False)
    msg = str(e.value)
    assert msg.startswith("W161 Weg2XchgCensusForeign")
    assert "Qwen3.8-27B-INT8-gdncov-vocabembed" in msg and "Qwen3.8-27B-FP8" in msg
    line = L.census_checkpoint_decision(c, "/models/Qwen3.8-27B-FP8", [str(tmp_path)], True)
    assert line.startswith("WEG2 XCHG-CENSUS FOREIGN (allowed by --weg2-xchg-census-foreign)")
    # the census of the booted checkpoint prints nothing new
    assert L.census_checkpoint_decision(
        c, "/models/Qwen3.8-27B-INT8-gdncov-vocabembed", [str(tmp_path)], False) is None


def test_the_census_own_checkpoint_field_wins_and_an_unknown_one_is_named(tmp_path):
    c = _census(tmp_path, checkpoint="/models/A")
    assert L.census_checkpoint(c, []) == ("/models/A", "the census's own checkpoint field")
    with pytest.raises(L.Weg2LaunchRefused):
        L.census_checkpoint_decision(c, "/models/B", [], False)
    u = _census(tmp_path, provenance="hand-made")
    line = L.census_checkpoint_decision(u, "/models/B", [str(tmp_path)], False)
    assert line.startswith("WEG2 XCHG-CENSUS checkpoint UNVERIFIED")


def test_the_flag_parses_and_defaults_off():
    ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
    assert ns.weg2_xchg_census_foreign is False
    ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--weg2-xchg-census-foreign"])
    assert ns.weg2_xchg_census_foreign is True


def test_the_census_writer_records_its_checkpoint():
    from sglang.srt.weg2 import xchg_census as XC

    src = inspect.getsource(XC.census_from_logs)
    assert 'blob["checkpoint"] = checkpoint' in src
    assert "p_log_model_path" in src


XSN441_CENSUS = "/spinning/gpu-arb/weg2/census/xchg_census_weg2xsn246_27198a2711.json"


@pytest.mark.skipif(not os.path.isfile(XSN441_CENSUS)
                    or not os.path.isdir(L.EVIDENCE_DIR), reason="rig evidence absent")
def test_xsn441s_own_census_names_the_int8_checkpoint():
    src, how = L.census_checkpoint(XSN441_CENSUS, [L.EVIDENCE_DIR])
    if not src:
        pytest.skip("source boot log absent: " + how)
    assert src.endswith("Qwen3.8-27B-INT8-gdncov-vocabembed"), (src, how)
    with pytest.raises(L.Weg2LaunchRefused):
        L.census_checkpoint_decision(
            XSN441_CENSUS, "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-FP8",
            [L.EVIDENCE_DIR], False)
