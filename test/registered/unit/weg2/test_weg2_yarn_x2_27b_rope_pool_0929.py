"""YaRN x2 for the 27B (F14, 29.09.): the P pool floor prices the RoPE cache.

``reserve_rope_cache_for_long_sequences`` sizes every cos/sin cache EAGERLY to
the context (fp32, rows x rotary_dim) inside ``load_model`` -- before the KV
pool is sized -- so a longer context comes straight out of the pool. MEASURED
at 262144 on the 27B P group (dkr27browauthoritybar1fs09290956, P.log):

    PP0/PP1/PP2  'MRotaryEmbedding 262528x64 float32 64.1 MiB ... EAGER'
    PP2          'RotaryEmbedding 262528x128 float32 128.2 MiB ... EAGER'  (DFlash2 draft, cold on the last stage)

The pool model's per-stage posts (P_PP_STAGE_FIXED_MIB) were read at that
context. Launcher dry run of 27b-row-authority-yarn2 on 895559fed2: the
cap+chunk floor 526336 shipped 34,17,13 at a PRICED 526648 -- while the rope
at 524288 takes PP0 +64 MiB (16 KiB/token: -4096) and PP2 +192 MiB (8 KiB/token:
-24576), so the realised pool lands near 516.6k, under the per-request cap it
promised to hold. Now the delta is priced (``p_rope_context_delta_mib``) and the
solver says the truth: W40 at 524288, 516574 servable."""
from __future__ import annotations

import inspect
import json
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.weg2 import launcher as L  # noqa: E402

TARGET_27B = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
              "text_config": {"head_dim": 256, "hidden_size": 5120, "num_attention_heads": 24,
                              "max_position_embeddings": 262144,
                              "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                                                  "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                                                  "rope_type": "default"}}}
DRAFT_27B = {"architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "head_dim": 128,
             "hidden_size": 5120, "num_attention_heads": 32,
             "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"}}
YARN = ("'{\"language_model_only\":true,\"text_config\":{\"rope_parameters\":{\"rope_type\":\"yarn\","
        "\"factor\":2.0,\"original_max_position_embeddings\":262144},\"max_position_embeddings\":524288}}'")


def _dirs(tmp_path):
    m, d = tmp_path / "target", tmp_path / "draft"
    m.mkdir()
    d.mkdir()
    (m / "config.json").write_text(json.dumps(TARGET_27B))
    (d / "config.json").write_text(json.dumps(DRAFT_27B))
    return str(m), str(d)


def _ns(model, draft, extra_p="", spec_form="DFLASH"):
    return types.SimpleNamespace(model=model, dflash_draft_path=draft, spec_form=spec_form,
                                 extra_p=extra_p, extra_d="")


def test_x1_prices_nothing(tmp_path):
    """Byte-equality guard: every 262144 form keeps its posts exactly."""
    m, d = _dirs(tmp_path)
    assert L.p_rope_context_delta_mib(_ns(m, d), 3) == ((0.0, 0.0, 0.0), None)
    assert L.p_rope_context_delta_mib(
        _ns(m, d, extra_p="--max-running-requests=2 --context-length 262144"), 3) == ((0.0, 0.0, 0.0), None)


def test_x2_27b_prices_target_on_every_stage_and_the_draft_on_the_last(tmp_path):
    m, d = _dirs(tmp_path)
    delta, line = L.p_rope_context_delta_mib(
        _ns(m, d, extra_p=f"--max-running-requests=2 --context-length 524288 --json-model-override-args {YARN}"), 3)
    assert delta == (64.0, 64.0, 192.0)
    assert "PP-CUT ROPE-KONTEXT" in line and "524288" in line and "draft" in line


def test_context_from_the_yarn_override_alone(tmp_path):
    """No --context-length in the extra: the YaRN factor stretches it (the
    same group_context_tokens the D solve reads)."""
    m, d = _dirs(tmp_path)
    delta, _ = L.p_rope_context_delta_mib(_ns(m, d, extra_p=f"--json-model-override-args {YARN}"), 3)
    assert delta == (64.0, 64.0, 192.0)


def test_nextn_form_has_no_separate_draft_cache(tmp_path):
    """NF (NEXTN/MTP): the draft shares the target's rope instance -- only the
    target term, NF's argv stays byte-identical (dry run 29.09.: P argv md5
    equal on 895559fed2 and with the fix; priced pool 847902 -> 838540)."""
    m, d = _dirs(tmp_path)
    delta, line = L.p_rope_context_delta_mib(
        _ns(m, d, extra_p="--context-length 524288", spec_form="NEXTN"), 3)
    assert delta == (64.0, 64.0, 64.0) and "draft" not in line


def test_cols_match_the_measured_x1_caches():
    """The measured x1 caches (262528 rows, fp32) from the formula's columns."""
    assert L._rope_cache_cols(TARGET_27B) == 64
    assert L._rope_cache_cols(DRAFT_27B) == 128
    assert round(262528 * 64 * 4 / 2**20, 1) == 64.1
    assert round(262528 * 128 * 4 / 2**20, 1) == 128.2


def test_the_pool_model_takes_the_priced_posts():
    """Wiring: solve_p_cut hands the posts PLUS the rope delta to the pool
    model (one construction site), not the bare flag vector."""
    src = inspect.getsource(L.solve_p_cut)
    assert "p_rope_context_delta_mib(ns, len(_stage_fixed))" in src
    assert "stage_fixed_mib=_stage_fixed," in src
    assert "stage_fixed_mib=tuple(_csv_floats(ns.pp_cut_stage_fixed_mib))" not in src


def test_unreadable_config_prices_zero_columns_not_a_crash(tmp_path):
    ns = _ns(str(tmp_path / "missing"), str(tmp_path / "missing2"), extra_p="--context-length 524288")
    delta, line = L.p_rope_context_delta_mib(ns, 3)
    assert delta == (0.0, 0.0, 0.0) and "0 cols" in line
