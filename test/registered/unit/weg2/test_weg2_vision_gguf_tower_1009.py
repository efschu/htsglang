"""VISION-GGUF (09.10.): ``--weg2-vision transient`` on a GGUF ``--model``
(Qwen3.8-27B-UD-IQ4_XS.gguf). A GGUF carries no tower (llama.cpp keeps it in
an mmproj; the GGUF loader maps text tensors only), so the transient stage
reads the tower from ``--tokenizer-path`` -- the safetensors checkpoint of the
same model -- and the GGUF branch of model_config keeps multimodal
TOKENIZATION on in exactly the transient form (no tower is built there).

RED on a8e569fd83 (the bugs these guard, black-box):
  * the rank read the tower's index at ``<x>.gguf/model.safetensors.index.json``
    and refused every image (W111b / W112);
  * model_config switched multimodal off for every GGUF without an mmproj, so
    the P tokenizer built no processor and arming refused ("NO multimodal
    processor");
  * the launcher had no word for a GGUF transient boot whose --tokenizer-path
    cannot serve a tower (the first image refused, after the cards were taken).
Unchanged (negative branches): a safetensors ``--model`` is its own source and
never reads ``tokenizer_path``; a GGUF without ``--weg2-vision transient``
keeps the text-only fallback and the launcher says nothing.

The victim side (plan §2/§4) is storage-level: on GGUF the ``.mlp.`` victims
are the flat uint8 ``qweight`` storages (+ the tiny ``qweight_type``, whose
DATA no forward reads -- gguf.py reads the python attributes). A whole stage
on such an attrappe borrows, encodes the checkpoint's numbers and returns
every byte (checksum ok). CPU stands in for the card.
"""

from __future__ import annotations

import json
import os
import types

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.planner import vision_stage_load as VSL
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import vision_d_guard as VDG
from sglang.srt.weg2 import vision_rank_runner as vrr
from sglang.srt.weg2 import vision_rank_stage as vrs
from sglang.srt.weg2 import vision_victim as vv
from sglang.srt.weg2 import vision_victim_27b as v27
from test_weg2_vision_rank_runner import _Alloc, _build, _Item, _kv, _req, _write_model  # noqa: E402

CPU = ("cpu",)
VISION_CONFIG = {"model_type": "qwen3_5", "depth": 27, "hidden_size": 1152, "out_hidden_size": 5120}


def _gguf_dir(tmp_path, vision_config=VISION_CONFIG, name="ggufdir"):
    """A GGUF file with its sibling config.json (what unsloth ships)."""
    d = tmp_path / name
    d.mkdir()
    cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"}
    if vision_config is not None:
        cfg["vision_config"] = dict(vision_config, model_type="qwen3_5_vision")
    (d / "config.json").write_text(json.dumps(cfg))
    path = d / "Qwen3.8-27B-UD-IQ4_XS.gguf"
    path.write_bytes(b"GGUF" + bytes(60))
    return str(path)


def _tower_dir(tmp_path, vision_config=VISION_CONFIG):
    """The safetensors checkpoint of the same model: index + the tower shard."""
    d = tmp_path / "int8dir"
    d.mkdir()
    ck = _write_model(d)
    (d / "config.json").write_text(json.dumps({"vision_config": vision_config}))
    return str(d), ck


# ------------------------------------------------------ the source choice --


def test_the_tower_source_of_a_gguf_is_the_tokenizer_dir_and_nothing_else_moves(tmp_path):
    gguf = _gguf_dir(tmp_path)
    tok, _ = _tower_dir(tmp_path)
    assert VSL.tower_source_dir(model_path=gguf, tokenizer_path=tok) == tok
    # safetensors --model: its own source, whatever the tokenizer path says
    assert VSL.tower_source_dir(model_path=tok, tokenizer_path="/elsewhere") == tok
    # the magic alone makes a GGUF (check_gguf_file's rule), a directory never does
    magic = tmp_path / "blob"
    magic.write_bytes(b"GGUF....")
    assert VSL.is_gguf_file(str(magic)) and not VSL.is_gguf_file(tok)
    # the server's default --tokenizer-path IS the .gguf, and a dir without an index: both named
    for bad in (gguf, str(tmp_path / "ggufdir"), ""):
        with pytest.raises(VSL.VisionStageLoadRefused, match="GGUF file and carries no vision tower"):
            VSL.tower_source_dir(model_path=gguf, tokenizer_path=bad)


def test_the_rank_reads_the_tokenizer_path_only_for_a_gguf(tmp_path):
    """The rank's helper: a safetensors server_args without a tokenizer_path
    attribute (every existing fake) resolves exactly as before."""
    tok, _ = _tower_dir(tmp_path)
    assert vrr.tower_dir(types.SimpleNamespace(model_path=tok)) == tok
    gguf = _gguf_dir(tmp_path)
    assert vrr.tower_dir(types.SimpleNamespace(model_path=gguf, tokenizer_path=tok)) == tok


def test_a_weights_arm_on_a_gguf_plans_from_the_tokenizer_checkpoint(tmp_path, monkeypatch):
    """RED on a8e569fd83: W111b (no model.safetensors.index.json under the .gguf)."""
    gguf = _gguf_dir(tmp_path)
    tok, _ = _tower_dir(tmp_path)
    model = _gguf_model()
    monkeypatch.setattr(vv, "resolve_source", lambda s: v27.DenseMlpVictims(model, device_types=CPU))

    def sched(tokenizer_path):
        return types.SimpleNamespace(server_args=types.SimpleNamespace(model_path=gguf, tokenizer_path=tokenizer_path))

    source, why = vrr.arm_victims(sched(tok), env={})
    assert why == "" and source.kind == v27.KIND_DENSE
    source, why = vrr.arm_victims(sched(gguf), env={})
    assert source is None and vv.W_VICTIM_PLAN_REFUSED in why and "carries no vision tower" in why


# ------------------------------------------------------- the model config --


@pytest.mark.parametrize("lmo,env,expect", [
    (True, {"SGLANG_WEG2_VISION": "transient", "SGLANG_WEG2_GROUP": "P"}, True),
    (True, {"SGLANG_WEG2_GROUP": "D"}, True),                       # D of the transient form (d_guard_armed)
    (True, {"SGLANG_WEG2_GROUP": "P"}, False),                      # P without the transient stage
    (False, {"SGLANG_WEG2_VISION": "transient", "SGLANG_WEG2_GROUP": "P"}, False),  # a tower would be built
    (False, {"SGLANG_WEG2_GROUP": "D"}, False),
    (True, {}, False),                                              # not a weg2 group at all
])
def test_the_gguf_multimodal_gate_is_the_transient_form_only(lmo, env, expect):
    hf = types.SimpleNamespace(language_model_only=True) if lmo else types.SimpleNamespace()
    assert VDG.transient_group_without_tower(hf, env=env) is expect


def test_a_real_gguf_model_config_stays_multimodal_only_in_the_transient_form(tmp_path, monkeypatch):
    """RED on a8e569fd83: is_multimodal False for the transient P form too (the
    GGUF branch switched every GGUF without an mmproj text-only), so P's
    tokenizer built no processor. A qwen35 GGUF (no mmproj) through the real
    ModelConfig: on only for language_model_only + the transient P env."""
    from test_weg2_gguf_launcher_g2_0925 import _write_gguf

    from sglang.srt.configs.model_config import ModelConfig

    n = 64
    text = {"model_type": "qwen3_5_text", "num_hidden_layers": n, "vocab_size": 8,
            "layer_types": ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(n)]}
    d = tmp_path / "g"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"],
                                               "model_type": "qwen3_5", "text_config": text, "vision_config": {}}))
    gguf = _write_gguf(str(d / "m.gguf"), block_count=n + 1)
    p_transient = {"SGLANG_WEG2_VISION": "transient", "SGLANG_WEG2_GROUP": "P"}
    for env, lmo, expect in ((p_transient, True, True), (p_transient, False, False), ({}, True, False)):
        for k in ("SGLANG_WEG2_VISION", "SGLANG_WEG2_GROUP"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        mc = ModelConfig(gguf, model_override_args=json.dumps({"language_model_only": True} if lmo else {}),
                         quantization="gguf")
        assert bool(mc.is_multimodal) is expect, (env, lmo)


# ------------------------------------------------------------ the launcher --


def _ns(model, tokenizer_path="", vision="transient"):
    return types.SimpleNamespace(model=model, tokenizer_path=tokenizer_path, weg2_vision=vision,
                                 weg2_vision_place="weights", weg2_boot_form=types.SimpleNamespace(arch="dense"),
                                 dual_layout=False)


def test_the_launcher_refusal_matrix(tmp_path):
    gguf = _gguf_dir(tmp_path)
    tok, ck = _tower_dir(tmp_path)
    assert L.vision_gguf_tower_refusal(_ns(gguf, tok)) is None
    assert L.vision_gguf_tower_refusal(_ns(gguf, "", vision="off")) is None      # GGUF text-only: unchanged
    assert L.vision_gguf_tower_refusal(_ns(tok, "")) is None                     # safetensors: unchanged
    why = L.vision_gguf_tower_refusal(_ns(gguf, str(tmp_path / "ggufdir")))
    assert why.startswith("W111 Weg2VisionArmRefused") and "carries no vision tower" in why
    # the host post prices the tower of the tokenizer checkpoint, not UNMEASURED
    mib, prov = L.vision_victim_host_term(_ns(gguf, tok))
    tower = sum(t.numel() * t.element_size() for n, t in ck.items() if "visual" in n)
    assert mib == pytest.approx(tower / (1 << 20)) and "UNMEASURED" not in prov


def test_a_tower_whose_geometry_differs_from_the_ggufs_is_refused(tmp_path):
    gguf = _gguf_dir(tmp_path)
    tok, _ = _tower_dir(tmp_path, vision_config=dict(VISION_CONFIG, depth=24))
    why = L.vision_gguf_tower_refusal(_ns(gguf, tok))
    assert why and "differs in depth" in why
    no_vision = _gguf_dir(tmp_path, vision_config=None, name="textonly")
    assert "is missing" in L.vision_gguf_tower_refusal(_ns(no_vision, tok))


# ---------------------------------------------- the victims of a GGUF rank --


class _GgufLinear(torch.nn.Module):
    """gguf.py's GGUFLinearMethod after loading: a flat uint8 ``qweight``
    (merged shards back to back, plus their plain-attribute views) and a
    ``qweight_type`` of one byte per shard."""

    def __init__(self, nbytes, shards=1, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.qweight = torch.nn.Parameter(torch.randint(0, 256, (nbytes,), generator=g, dtype=torch.uint8),
                                          requires_grad=False)
        self.qweight_type = torch.nn.Parameter(torch.full((shards,), 23, dtype=torch.uint8), requires_grad=False)
        step = nbytes // shards
        self._gguf_shard_views = {str(i): self.qweight.data.narrow(0, i * step, step) for i in range(shards)}


def _gguf_model(layers=2):
    m = torch.nn.Module()
    m.model = torch.nn.Module()
    m.model.embed_tokens = _GgufLinear(512, seed=1)
    m.model.layers = torch.nn.ModuleList()
    for i in range(layers):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.qkv_proj = _GgufLinear(2048, shards=3, seed=10 + i)
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_up_proj = _GgufLinear(4096, shards=2, seed=20 + i)
        layer.mlp.down_proj = _GgufLinear(2048, seed=30 + i)
        m.model.layers.append(layer)
    m.lm_head = _GgufLinear(512, seed=2)
    return m


def test_a_gguf_rank_lends_only_its_mlp_qweights_and_gets_every_byte_back(tmp_path):
    """Storage-level, quantization-neutral: the stage on a GGUF rank borrows
    the flat uint8 MLP qweights (never attention, embed, head), encodes the
    tokenizer checkpoint's numbers, and returns every byte of the model."""
    tok, ck = _tower_dir(tmp_path)
    model = _gguf_model()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    victims = v27.DenseMlpVictims(model, device_types=CPU)
    names = [c.name for c in victims.inventory()]
    assert names and all(".mlp." in n for n in names)
    assert {n for n in names if n.endswith("qweight")} == {
        f"model.layers.{i}.mlp.{p}.qweight" for i in range(2) for p in ("gate_up_proj", "down_proj")}
    it = _Item()
    px = it.feature.clone()
    out = vrr.run_rank_stage(types.SimpleNamespace(token_to_kv_pool_allocator=_Alloc(_kv())),
                             [_req("r", [it])], model_dir=tok, hf_config=None, device=torch.device("cpu"),
                             build=_build(), place=vrs.PLACE_WEIGHTS, victims=victims)
    assert out.ok, out.detail
    assert out.checksum == "ok" and "victim=dense" in out.victim_fields and "split_tensors=0" in out.victim_fields
    w1, b1 = ck["model.visual.blocks.0.attn.qkv.weight"], ck["model.visual.blocks.0.attn.qkv.bias"]
    w2, b2 = ck["model.visual.merger.linear_fc1.weight"], ck["model.visual.merger.linear_fc1.bias"]
    assert torch.equal(it.precomputed_embeddings, F.linear(F.linear(px.to(torch.bfloat16), w1, b1), w2, b2))
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())
    assert victims.host_bytes == 0
