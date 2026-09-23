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
    # 19.09. (user order): the draft is SHARDED on D ('split' = the server
    # default, no placement flag); solo is the explicit A/B opt-in below.
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


@pytest.mark.skipif(not os.path.isdir(DRAFT), reason="DFlash2 draft checkpoint not on this host")
def test_external_drafter_bytes_come_off_the_dflash_headers():
    from sglang.srt.weg2.ring_table import (
        checkpoint_stage_weights,
        external_drafter_mib_from_argv,
    )

    argv_p = ["--pp-size", "3", "--speculative-algorithm", "DFLASH",
              "--speculative-draft-model-path", DRAFT, "--speculative-num-draft-tokens", "8",
              "--speculative-draft-kv-only"]
    mib = external_drafter_mib_from_argv(argv_p)
    assert 2040 < mib < 2100  # 258 + 1315 + 499 MiB of headers (measured 2026-09-17)
    assert external_drafter_mib_from_argv(["--speculative-algorithm", "NEXTN"]) == 0.0
    assert external_drafter_mib_from_argv([]) == 0.0
    # the stage term: the whole draft on the last stage, nothing on the others
    stages = checkpoint_stage_weights(
        L.MODEL_DEFAULT, [43, 11, 10], [10, 3, 3], True,
        drafter_head_from_target=True, external_drafter_mib=mib,
    )
    assert [round(s.drafter_mib) for s in stages] == [0, 0, round(mib)]
    assert "external drafter checkpoint" in stages[-1].drafter_terms


def test_solo_placement_is_an_explicit_ab_opt_in(restore_form, tmp_path, monkeypatch):
    L.apply_spec_form(
        SimpleNamespace(spec_form="dflash", dflash_draft_path=str(tmp_path), dflash_block=8, dflash_window=2048)
    )
    monkeypatch.setenv(L.ENV_DFLASH_PLACEMENT, "solo")
    assert L.spec_flags(producer=False)[-2:] == ["--speculative-draft-placement", "solo"]
    assert L.spec_form_env("D") == {"SGLANG_DFLASH_WINDOW_POOL": "1",
                                    "SGLANG_DFLASH_SOLO_COMPACT": "1"}
    monkeypatch.setenv(L.ENV_DFLASH_PLACEMENT, "nonsense")
    assert "--speculative-draft-placement" not in L.spec_flags(producer=False)
    assert L.spec_form_env("D") == {"SGLANG_DFLASH_WINDOW_POOL": "1"}
