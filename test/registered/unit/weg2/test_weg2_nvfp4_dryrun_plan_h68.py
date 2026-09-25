"""H68: the metadata-only MIXED_PRECISION dry run
(``weg2/nvfp4_ckpt_plan.plan_from_checkpoint``) over a hand-written mini
checkpoint in the ModelOpt NVFP4 layout: real ``.safetensors`` files (header +
bytes), an index, ``config.json`` + ``hf_quant_config.json`` -- NVFP4 routed
experts (weight U8 / weight_scale E4M3 / weight_scale_2 / input_scale), a
block-FP8 draft head (``weight_scale_inv``), a sharded per-tensor FP8 PLE
table, BF16 everything else.

What is pinned:
* the plan resolves every module, finds NO anomaly on a correct export, and
  sizes the routed-expert row (checkpoint bytes and the Marlin slot);
* every class of defect is named: a missing companion scale, a wrong scale
  shape, a module whose tensors carry another algorithm than declared;
* a truncated shard is REFUSED, never measured (a download in flight);
* ``read_scalars`` checks gate/up ``weight_scale_2`` equality (Marlin keeps
  ONE w13 global scale);
* LOADER DRY RUN: the real ``ModelOptMixedPrecisionConfig`` resolves the same
  algorithm for every sglang module prefix as the dry run (drift guard).

No torch tensor is created, no GPU; the only file writes are in a temp dir.
"""

import json
import os
import struct
import tempfile

import pytest

try:
    from sglang.srt.weg2 import nvfp4_ckpt_plan as P
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

H, I, E, G = 64, 32, 4, 16  # hidden, moe intermediate, experts, NVFP4 group
BLK = 32  # draft block-FP8 edge in the mini model (the real export: 128)
NB = {"U8": 1, "F8_E4M3": 1, "BF16": 2, "F32": 4, "I64": 8}


def _t(dtype, shape, payload=None):
    n = NB[dtype]
    for d in shape:
        n *= d
    return (dtype, tuple(shape), payload if payload is not None else bytes(n))


def _f32(v):
    return _t("F32", (), struct.pack("<f", v))


def write_st(path, tensors):
    header, off, blobs = {}, 0, []
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(data)]}
        off += len(data)
        blobs.append(data)
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        fh.write(b"".join(blobs))


def nvfp4_expert(prefix, s2=1e-4, up_s2=None):
    out = {}
    for proj, (n, k) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
        v = up_s2 if (proj == "up_proj" and up_s2 is not None) else s2
        out[f"{prefix}.{proj}.weight"] = _t("U8", (n, k // 2))
        out[f"{prefix}.{proj}.weight_scale"] = _t("F8_E4M3", (n, k // G))
        out[f"{prefix}.{proj}.weight_scale_2"] = _f32(v)
        out[f"{prefix}.{proj}.input_scale"] = _f32(2e-3)
    return out


def build_ckpt(tmp, *, mutate=None, up_s2=None):
    lm = "model.language_model"
    s1 = {f"{lm}.embed_tokens.weight": _t("BF16", (100, H))}
    s1[f"{lm}.layers.0.linear_attn.in_proj_qkv.weight"] = _t("BF16", (96, H))
    s1[f"{lm}.layers.0.linear_attn.A_log"] = _t("BF16", (4,))
    s1[f"{lm}.layers.0.mlp.gate.weight"] = _t("BF16", (E, H))
    s1[f"{lm}.layers.0.mlp.shared_expert.gate_proj.weight"] = _t("BF16", (I, H))
    s1[f"{lm}.layers.0.mlp.shared_expert_gate.weight"] = _t("BF16", (1, H))
    for e in range(E):
        s1.update(nvfp4_expert(f"{lm}.layers.0.mlp.experts.{e}", up_s2=up_s2 if e == 0 else None))
    s2 = {"lm_head.weight": _t("BF16", (100, H))}
    for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
        s2[f"{lm}.layers.1.self_attn.{p}.weight"] = _t("BF16", (H, H))
    s2[f"{lm}.layers.1.ple.key_proj.weight"] = _t("BF16", (H, H))
    s2[f"{lm}.layers.1.mlp.gate.weight"] = _t("BF16", (E, H))
    for e in range(E):
        s2.update(nvfp4_expert(f"{lm}.layers.1.mlp.experts.{e}"))
    s3 = {}
    tbl = f"{lm}.layers.1.ple.ple_embedding.ngram_embedding"
    s3[f"{tbl}.shard_0.weight"] = _t("F8_E4M3", (10, 8))
    s3[f"{tbl}.shard_1.weight"] = _t("F8_E4M3", (10, 8))
    s3[f"{tbl}.weight_scale"] = _t("BF16", (1,))
    s3[f"{lm}.layers.1.ple.ple_embedding.ngram_heads_offsets"] = _t("I64", (4,))
    for e in range(2):
        for proj, (n, k) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            s3[f"mtp.layers.0.mlp.experts.{e}.{proj}.weight"] = _t("F8_E4M3", (n, k))
            s3[f"mtp.layers.0.mlp.experts.{e}.{proj}.weight_scale_inv"] = _t(
                "BF16", (-(-n // BLK), -(-k // BLK)))
    s3["mtp.layers.0.self_attn.q_proj.weight"] = _t("BF16", (H, H))
    files = {
        "model-00001-of-00002.safetensors": s1,
        "model-00002-of-00002.safetensors": s2,
        "model-fp8-mtp-ple.safetensors": s3,
    }
    if mutate:
        mutate(files)
    weight_map = {}
    for fname, tensors in files.items():
        write_st(os.path.join(tmp, fname), tensors)
        weight_map.update({n: fname for n in tensors})
    with open(os.path.join(tmp, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {}, "weight_map": weight_map}, fh)
    ql = {f"{lm}.layers.{i}.mlp.experts": {"quant_algo": "NVFP4", "group_size": G} for i in (0, 1)}
    ql[tbl] = {"quant_algo": "FP8"}
    ignore = ["lm_head", f"{lm}.embed_tokens", f"{lm}.layers.0.linear_attn*",
              f"{lm}.layers.1.self_attn*"]
    for i in (0, 1):
        ignore += [f"{lm}.layers.{i}.mlp.gate", f"{lm}.layers.{i}.mlp.shared_expert*",
                   f"{lm}.layers.{i}.mlp.shared_expert_gate"]
    qc = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
          "producer": {"name": "modelopt", "version": "0.46.0"}, "ignore": ignore,
          "quantized_layers": dict(ql, **{"mtp.layers.0.mlp.experts": {"quant_algo": "FP8_PB_WO", "group_size": BLK}})}
    with open(os.path.join(tmp, "config.json"), "w") as fh:
        json.dump({"architectures": ["Qwen4ExpForConditionalGeneration"], "quantization_config": qc}, fh)
    hf = {"producer": qc["producer"], "quantization": {
        "quant_algo": "MIXED_PRECISION", "group_size": G, "exclude_modules": ignore,
        "quantized_layers": dict(ql, **{"mtp.layers.0.mlp.experts": {"quant_algo": "FP8_BLOCK_SCALES", "group_size": BLK}})}}
    with open(os.path.join(tmp, "hf_quant_config.json"), "w") as fh:
        json.dump(hf, fh)
    return tmp


EXPERT_CKPT = 2 * (I * H // 2 + I * H // G + 8) + (H * I // 2 + H * I // G + 8)
EXPERT_MARLIN = 2 * (I * H // 2 + I * H // G) + (H * I // 2 + H * I // G) + 4


def test_correct_export_plans_clean():
    with tempfile.TemporaryDirectory() as tmp:
        plan = P.plan_from_checkpoint(build_ckpt(tmp), read_scalars=True)
    assert plan.anomalies == []
    assert plan.files == 3
    assert plan.fmt.key == (
        "modelopt:MIXED_PRECISION|draft.moe.routed=fp8blk32;moe.routed=nvfp4a4g16;ple.table=fp8pt;*=bf16"
    )
    assert len(plan.fmt.inconsistencies) == 1  # FP8_PB_WO vs FP8_BLOCK_SCALES
    row = plan.expert_row
    assert (row.algo, row.layers, row.experts_per_layer, row.uniform) == ("NVFP4", 2, E, True)
    assert row.ckpt_bytes == EXPERT_CKPT
    assert row.marlin_slot_bytes == EXPERT_MARLIN
    assert row.gate_up_scale2_equal == (2 * E, 2 * E)
    assert plan.draft_expert_row.algo == "FP8_BLOCK"
    assert plan.draft_expert_row.marlin_slot_bytes is None
    assert plan.census[("moe.routed", "NVFP4")] == [2 * E * 3, 2 * E * EXPERT_CKPT]
    assert plan.census[("ple.table", "FP8")][1] == 2 * 10 * 8 + 2  # shards + bf16 scale
    assert plan.bytes_for("draft.moe.routed") > 0
    lines = plan.lines()
    assert lines[0].startswith("NVFP4-FORMAT key=")
    assert any(ln.startswith("NVFP4-EXPERT-ROW algo=NVFP4") for ln in lines)


def test_gate_up_global_scale_mismatch_is_counted():
    with tempfile.TemporaryDirectory() as tmp:
        plan = P.plan_from_checkpoint(build_ckpt(tmp, up_s2=3e-4), read_scalars=True)
    assert plan.expert_row.gate_up_scale2_equal == (2 * E - 1, 2 * E)


def _drop(name):
    def mutate(files):
        for tensors in files.values():
            tensors.pop(name, None)
    return mutate


def test_missing_weight_scale_2_is_named():
    victim = "model.language_model.layers.1.mlp.experts.2.down_proj.weight_scale_2"
    with tempfile.TemporaryDirectory() as tmp:
        plan = P.plan_from_checkpoint(build_ckpt(tmp, mutate=_drop(victim)))
    assert any("experts.2.down_proj: declared NVFP4" in a for a in plan.anomalies), plan.anomalies


def test_wrong_scale_shape_is_named():
    def mutate(files):
        files["model-00002-of-00002.safetensors"][
            "model.language_model.layers.1.mlp.experts.0.gate_proj.weight_scale"] = _t("F8_E4M3", (I, H // 32))
    with tempfile.TemporaryDirectory() as tmp:
        plan = P.plan_from_checkpoint(build_ckpt(tmp, mutate=mutate))
    assert any("experts.0.gate_proj: weight_scale" in a for a in plan.anomalies), plan.anomalies


def test_undeclared_quantized_module_is_named():
    """A module the config leaves BF16 but whose tensors are FP8."""
    def mutate(files):
        s = files["model-00002-of-00002.safetensors"]
        s["model.language_model.layers.1.ple.key_proj.weight"] = _t("F8_E4M3", (H, H))
        s["model.language_model.layers.1.ple.key_proj.weight_scale"] = _t("F32", (1,))
    with tempfile.TemporaryDirectory() as tmp:
        plan = P.plan_from_checkpoint(build_ckpt(tmp, mutate=mutate))
    assert any("ple.key_proj: declared BF16" in a and "carry FP8" in a for a in plan.anomalies)


def test_truncated_shard_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        build_ckpt(tmp)
        path = os.path.join(tmp, "model-00002-of-00002.safetensors")
        with open(path, "r+b") as fh:
            fh.truncate(os.path.getsize(path) - 3)
        with pytest.raises(P.IncompleteShard):
            P.plan_from_checkpoint(tmp)


def test_sglang_prefixes():
    assert P.sglang_module_prefix("model.language_model.layers.3.mlp.experts.9.up_proj") == "model.layers.3.mlp.experts"
    assert P.sglang_module_prefix("model.language_model.layers.3.self_attn.k_proj") == "model.layers.3.self_attn.qkv_proj"
    assert P.sglang_module_prefix("model.language_model.layers.0.linear_attn.in_proj_z") == "model.layers.0.linear_attn.in_proj_qkvz"
    assert P.sglang_module_prefix("model.language_model.layers.0.mlp.shared_expert.up_proj") == "model.layers.0.mlp.shared_expert.gate_up_proj"
    assert P.sglang_module_prefix("mtp.layers.0.mlp.experts.1.down_proj") == "mtp.layers.0.mlp.experts"
    assert P.sglang_module_prefix("lm_head") == "lm_head"


def test_loader_dry_run_matches_the_real_quant_config():
    """The dry run and ``ModelOptMixedPrecisionConfig`` resolve the same
    algorithm for every sglang module prefix of the mini checkpoint."""
    try:
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptMixedPrecisionConfig,
        )
    except Exception as ex:  # pragma: no cover - import-chain dependent
        pytest.skip(f"modelopt_quant not importable on this host: {ex}")
    with tempfile.TemporaryDirectory() as tmp:
        build_ckpt(tmp)
        qc = json.load(open(os.path.join(tmp, "config.json")))["quantization_config"]
        cfg = ModelOptMixedPrecisionConfig.from_config(qc)
        plan = P.plan_from_checkpoint(tmp, resolver=cfg.resolve_quant_algo)
    assert [a for a in plan.anomalies if a.startswith("loader drift")] == []
    assert plan.loader_view["model.layers.0.mlp.experts"] == "NVFP4"
    assert plan.loader_view["mtp.layers.0.mlp.experts"] == "FP8_BLOCK"
    assert plan.loader_view["model.layers.1.ple.ple_embedding.ngram_embedding"] == "FP8"
    assert plan.loader_view["model.layers.1.self_attn.qkv_proj"] == "BF16"


def test_resolver_drift_is_reported():
    with tempfile.TemporaryDirectory() as tmp:
        plan = P.plan_from_checkpoint(
            build_ckpt(tmp),
            resolver=lambda prefix: "NVFP4" if prefix.endswith("experts") else None,
        )
    assert any("loader drift: mtp.layers.0.mlp.experts" in a for a in plan.anomalies)


def test_cli_format_only(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        rc = P.main([build_ckpt(tmp), "--format-only"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "NVFP4-FORMAT key=modelopt:MIXED_PRECISION|" in out
    assert "inconsistency mtp.layers.0.mlp.experts" in out
