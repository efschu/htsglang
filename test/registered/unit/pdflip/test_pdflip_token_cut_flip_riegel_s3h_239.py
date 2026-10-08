"""#239 S3h: the FLIP boot under the token cut.

S3a-e and S4a/S4b wired the cut for group D (F4/F5/F12/F14). The P->D
hand-off under it rides the one arena L2 and the canonical page store: P
writes whole geometry-neutral pages, a KV-holding D worker reads its owner
rows of them (S4b parts 2/3), tail adopt is per owner (part 5), the prefetch
claim is the group MIN (the worker votes in the min arm only), the wake credit
is taken per rank against its own card. The riegel lets exactly that form
through -- F14 wired, a host tier, and either ONE operator-named KV stage or
the launcher's S3g stage form with a trim cell for EVERY KV worker -- and
refuses every other flip boot under the cut by name. ``--d-only`` (H87) keeps
booting; every other form is untouched.

29.09. (S3h lifted where S3g holds): the riegel runs before the plan, so an
unnamed stage form passes it provisionally and the proof comes after the
FRACTION-SOLVE (``refuse_flip_stage_form_without_trim_cells``): every rank the
cut gives FA KV must carry its own stage rows in the written form, else the
flip boot stops by name there. Operator-named stages (more than one) stay
refused -- the launcher writes no per-rank rows for them.
"""

from __future__ import annotations

import inspect
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt import rank_role  # noqa: E402
from flliper.srt.planner import expert_residency as er  # noqa: E402
from flliper.srt.pdflip import launcher as L  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pdflip_d_kv_stage_launcher_251c as T251  # noqa: E402
import test_pdflip_d_kv_stage_per_rank_s3g_239 as S3G  # noqa: E402

CUT = types.SimpleNamespace(kv="qsa_forma_dcp")

#: main 28.09. ~19:10Z: the serving profile nf-h91-dpr-sa-vis-noadopt-st-cut.env
#: = -st + --d-kv-token-cut owned + one KV stage in NF_ENV_D, host tier on.
SERVING_ENV_D = ("FLLIPER_QWEN4_PLE_CKPT_GATHER=pread;FLLIPER_MOE_SCRATCH_SLOTS=100,48,48;"
                 "FLLIPER_PDFLIP_D_KV_STAGE_TOKENS=262144")


def _ns(**kw):
    ns = types.SimpleNamespace(dry_run=False, d_only=False, d_kv_token_cut="owned",
                               pdflip_disable_hicache=False, env_d=SERVING_ENV_D)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_the_serving_form_is_let_through_and_named(capsys):
    line = L.refuse_flip_under_token_cut(_ns(), CUT)
    assert line and line.startswith("#239 S3h FLIP UNTER KV-TOKEN-SCHNITT ERLAUBT")
    assert "FLLIPER_PDFLIP_D_KV_STAGE_TOKENS=262144" in line
    assert "#239 F14 KV-WORKER-WINDOW" in line  # the metal marker it names
    assert line in capsys.readouterr().out


def test_without_a_host_tier_the_flip_is_refused_by_name():
    with pytest.raises(L.PdFlipTokenCutFlipNotWired) as ei:
        L.refuse_flip_under_token_cut(_ns(pdflip_disable_hicache=True), CUT)
    assert "#239 S3h FLIP UNTER KV-TOKEN-SCHNITT" in str(ei.value)
    assert "--pdflip-disable-hicache" in str(ei.value)
    assert "--d-only" in str(ei.value)
    # a launch refusal like every other: the arm's handler catches it
    assert isinstance(ei.value, L.PdFlipLaunchRefused)


def test_operator_named_stages_stay_refused():
    with pytest.raises(L.PdFlipTokenCutFlipNotWired) as ei:
        L.refuse_flip_under_token_cut(
            _ns(env_d="FLLIPER_PDFLIP_D_KV_STAGE_TOKENS=262144,393216,524288"), CUT)
    assert "3 stages" in str(ei.value)
    assert "S3g" in str(ei.value)


def test_unnamed_stages_pass_to_the_solve(capsys):
    """RED on 6d418b49c2: 'the D KV stage form is not named ... (S3g open)' --
    the riegel refused the very form S3g writes per KV rank."""
    line = L.refuse_flip_under_token_cut(_ns(env_d="FLLIPER_MOE_SCRATCH_SLOTS=100,48,48"), CUT)
    assert line and line.startswith(L.KV_TOKEN_CUT_FLIP_MARKER + " ERLAUBT")
    assert "Trim-Zelle" in line and "S3g" in line
    assert line in capsys.readouterr().out


# ---- after the FRACTION-SOLVE: every KV worker has its trim cell ---------------------

def _solved(plan, shares=(0, 48, 16), env_d="FLLIPER_MOE_SCRATCH_SLOTS=100,48,48", **kw):
    ns = _ns(env_d=env_d, **kw)
    lines = L.apply_d_kv_stage_form(ns, er, T251._rows(), T251.FORM, plan, "D",
                                    verify_tokens=4, top_k=10, kv_token_shares=shares)
    return ns, lines


def test_the_s3g_form_with_every_worker_trim_cell_is_let_through():
    """RED on 6d418b49c2: no such check -- the flip boot never got this far."""
    ns, _ = _solved(S3G._cut_plan())
    out = []
    line = L.refuse_flip_stage_form_without_trim_cells(ns, CUT, out.append)
    assert line == out[-1]
    assert line.startswith(L.KV_TOKEN_CUT_FLIP_MARKER + " STUFEN ERLAUBT")
    assert "rang1: 22" in line and "rang2: 8" in line
    assert "'#251c KV-STAGE form=" in line  # the metal marker it names


def test_a_dropped_stage_form_is_refused_by_name():
    """A plan without the workers' trim cells writes no form ('entfaellt unter
    dem Token-Schnitt'); the flip boot does not fall back to fixed tokens."""
    ns, lines = _solved(T251._plan())
    assert "entfaellt unter dem Token-Schnitt" in lines[0]
    with pytest.raises(L.PdFlipTokenCutFlipNotWired) as ei:
        L.refuse_flip_stage_form_without_trim_cells(ns, CUT, print)
    assert L.KV_TOKEN_CUT_FLIP_MARKER in str(ei.value)
    assert "keine Stufenform" in str(ei.value) and "--d-only" in str(ei.value)


def test_a_worker_without_its_stage_rows_is_refused_by_name():
    ns, _ = _solved(S3G._cut_plan())
    del ns._d_kv_stage_written["worker_rows"][2]
    with pytest.raises(L.PdFlipTokenCutFlipNotWired) as ei:
        L.refuse_flip_stage_form_without_trim_cells(ns, CUT, print)
    assert "Rang 2" in str(ei.value) and "Trim-Zelle" in str(ei.value)


def test_the_dry_run_names_the_missing_trim_cell():
    ns, _ = _solved(T251._plan(), dry_run=True)
    out = []
    line = L.refuse_flip_stage_form_without_trim_cells(ns, CUT, out.append)
    assert line and "(dry run: would refuse)" in out[-1]


def test_one_operator_named_stage_has_nothing_to_prove():
    ns, lines = _solved(S3G._cut_plan(), env_d=SERVING_ENV_D)
    assert "--env-d nennt" in lines[0]
    assert L.refuse_flip_stage_form_without_trim_cells(ns, CUT, print) is None


@pytest.mark.parametrize("form, kw", [
    (types.SimpleNamespace(kv="qsa_forma"), {}), (None, {}), (CUT, {"d_only": True})])
def test_the_post_check_leaves_every_other_boot_alone(form, kw):
    ns, _ = _solved(T251._plan(), **kw)
    assert L.refuse_flip_stage_form_without_trim_cells(ns, form, print) is None


def test_main_checks_before_every_flip_d_start():
    src = inspect.getsource(L.main)
    starts = [i for i in range(len(src))
              if src.startswith("refuse_d_form_off_the_map(ns, log)", i)]
    checks = [i for i in range(len(src))
              if src.startswith("refuse_flip_stage_form_without_trim_cells(", i)]
    assert len(checks) == 2  # dry run and real flip boot; d-only never flips
    for c in checks:
        assert any(0 < s - c < 200 for s in starts)


def test_an_unwired_seam_refuses_the_flip(monkeypatch):
    monkeypatch.setattr(rank_role, "unwired_token_cut_seams", lambda: ("F14",))
    with pytest.raises(L.PdFlipTokenCutFlipNotWired) as ei:
        L.refuse_flip_under_token_cut(_ns(), CUT)
    assert "F14" in str(ei.value)


def test_d_only_boots_the_cut():
    assert L.refuse_flip_under_token_cut(_ns(d_only=True, env_d=""), CUT) is None


def test_the_dry_run_names_a_refusal_and_does_not_raise(capsys):
    line = L.refuse_flip_under_token_cut(_ns(dry_run=True, pdflip_disable_hicache=True), CUT)
    assert line and line.startswith("#239 S3h FLIP UNTER KV-TOKEN-SCHNITT")
    assert "(dry run: would refuse)" in capsys.readouterr().out


@pytest.mark.parametrize("form", [None, types.SimpleNamespace(kv="qsa_forma"),
                                  types.SimpleNamespace(kv="qsa")])
def test_every_other_form_is_untouched(form, capsys):
    assert L.refuse_flip_under_token_cut(_ns(env_d=""), form) is None
    assert capsys.readouterr().out == ""


def test_stage_tokens_are_read_from_the_d_env():
    assert L.d_kv_stage_tokens_named(_ns()) == (262144,)
    assert L.d_kv_stage_tokens_named(_ns(env_d="A=1")) is None


def test_main_calls_it_right_after_the_seam_riegel():
    src = inspect.getsource(L.main)
    i = src.index("refuse_unwired_token_cut(ns, boot_form)")
    assert src.index("refuse_flip_under_token_cut(ns, boot_form)", i) - i < 200
