"""--spec-form DFLASH (PLAN_DFLASH2_P_0917 part E): the launcher's speculative
form, and the external draft's pricing families.

Hermetic. The checkpoint-header reader runs against the DFlash2 draft on this
rig when present (the numbers pinned are the measured 2026-09-17 headers)
and is skipped elsewhere.
"""

import os
from types import SimpleNamespace

import pytest

import sglang.srt.weg2.launcher as L
from sglang.srt.speculative import dflash_pricing as P


@pytest.fixture
def restore_form():
    saved = dict(L._SPEC_FORM)
    try:
        yield
    finally:
        L._SPEC_FORM.clear()
        L._SPEC_FORM.update(saved)


def test_nextn_form_is_the_shipping_family(restore_form):
    L.apply_spec_form(SimpleNamespace(spec_form="NEXTN"))
    assert L.spec_flags(producer=True) == list(L.P_DRAFT_KV_FLAGS)
    assert L.spec_flags(producer=False) == [
        "--speculative-algorithm", L.SPEC_ALGORITHM,
        "--speculative-num-steps", str(L.SPEC_NUM_STEPS),
        "--speculative-eagle-topk", str(L.SPEC_EAGLE_TOPK),
        "--speculative-num-draft-tokens", str(L.SPEC_NUM_DRAFT_TOKENS),
    ]
    assert L.spec_form_env("D") == {} and L.spec_form_env("P") == {}
    assert L.spec_plan_fields() == {
        "speculative_algorithm": L.SPEC_ALGORITHM,
        "speculative_num_draft_tokens": L.SPEC_NUM_DRAFT_TOKENS,
    }


def test_dflash_form_is_byte_identical_on_both_groups_but_for_the_role(restore_form, tmp_path):
    L.apply_spec_form(
        SimpleNamespace(spec_form="dflash", dflash_draft_path=str(tmp_path), dflash_block=8, dflash_window=2048)
    )
    p, d = L.spec_flags(producer=True), L.spec_flags(producer=False)
    ident = ["--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", str(tmp_path),
             "--speculative-num-draft-tokens", "8"]
    assert p == ident + ["--speculative-draft-kv-only"]
    assert d == ident + ["--speculative-draft-window-size", "2048"]
    assert L.spec_form_env("D") == {"SGLANG_DFLASH_WINDOW_POOL": "1"}
    assert L.spec_form_env("P") == {}
    assert L.spec_plan_fields() == {
        "speculative_algorithm": "DFLASH",
        "speculative_num_draft_tokens": 8,
        "speculative_draft_model_path": str(tmp_path),
    }


def test_dflash_form_refuses_a_missing_draft(restore_form, tmp_path):
    with pytest.raises(SystemExit, match="not a directory"):
        L.apply_spec_form(SimpleNamespace(spec_form="DFLASH", dflash_draft_path=str(tmp_path / "nope")))


DRAFT = L.DFLASH_DRAFT_PATH_DEFAULT


@pytest.mark.skipif(not os.path.isdir(DRAFT), reason="DFlash2 draft checkpoint not on this host")
def test_draft_families_read_off_the_checkpoint_headers():
    fam = P.dflash_draft_family_bytes(DRAFT)
    mib = {k: v / 2**20 for k, v in fam.items()}
    # measured 2026-09-17 on Qwen3.8-27B-DFlash2-W8-lued (W8 pack-quantized)
    assert 250 < mib["attn"] < 265 and 1300 < mib["mlp"] < 1330 and 490 < mib["repl"] < 510
    assert P.dflash_draft_kv_heads(DRAFT) == 8


def test_dflash_attn_family_shards_on_the_drafts_head_grid():
    from sglang.srt.uneven_perf import PerfCostModel

    m = PerfCostModel.__new__(PerfCostModel)
    m.tp_size = 3
    m.base_plan = [3991, 1000, 1000]
    m.dflash_kv_heads = 8
    m.solo_rank = 0
    fr = m._shard_fractions("dflash_attn", [3991, 1000, 1000])
    assert [round(f * 8) for f in fr] == [5, 2, 1]
    m.dflash_kv_heads = 2  # fewer heads than ranks: replicated-KV regime
    assert m._shard_fractions("dflash_attn", [3991, 1000, 1000]) == [1.0, 1.0, 1.0]
