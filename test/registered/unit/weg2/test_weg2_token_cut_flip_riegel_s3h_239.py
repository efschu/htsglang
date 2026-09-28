"""#239 S3h: the FLIP boot under the token cut.

S3a-e and S4a/S4b wired the cut for group D (F4/F5/F12/F14). The P->D
hand-off under it rides the one arena L2 and the canonical page store: P
writes whole geometry-neutral pages, a KV-holding D worker reads its owner
rows of them (S4b parts 2/3), tail adopt is per owner (part 5), the prefetch
claim is the group MIN (the worker votes in the min arm only), the wake credit
is taken per rank against its own card. The riegel lets exactly that form
through -- F14 wired, a host tier, ONE KV stage (the #251c stage form would
build a KV worker's pool for the top stage, S3g) -- and refuses every other
flip boot under the cut by name. ``--d-only`` (H87) keeps booting; every other
form is untouched.
"""

from __future__ import annotations

import inspect
import types

import pytest

from sglang.srt import rank_role
from sglang.srt.weg2 import launcher as L

CUT = types.SimpleNamespace(kv="qsa_forma_dcp")

#: main 28.09. ~19:10Z: the serving profile nf-h91-dpr-sa-vis-noadopt-st-cut.env
#: = -st + --d-kv-token-cut owned + one KV stage in NF_ENV_D, host tier on.
SERVING_ENV_D = ("SGLANG_QWEN4_PLE_CKPT_GATHER=pread;SGLANG_MOE_SCRATCH_SLOTS=100,48,48;"
                 "SGLANG_WEG2_D_KV_STAGE_TOKENS=262144")


def _ns(**kw):
    ns = types.SimpleNamespace(dry_run=False, d_only=False, d_kv_token_cut="owned",
                               weg2_disable_hicache=False, env_d=SERVING_ENV_D)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_the_serving_form_is_let_through_and_named(capsys):
    line = L.refuse_flip_under_token_cut(_ns(), CUT)
    assert line and line.startswith("#239 S3h FLIP UNTER KV-TOKEN-SCHNITT ERLAUBT")
    assert "SGLANG_WEG2_D_KV_STAGE_TOKENS=262144" in line
    assert "#239 F14 KV-WORKER-WINDOW" in line  # the metal marker it names
    assert line in capsys.readouterr().out


def test_without_a_host_tier_the_flip_is_refused_by_name():
    with pytest.raises(L.Weg2TokenCutFlipNotWired) as ei:
        L.refuse_flip_under_token_cut(_ns(weg2_disable_hicache=True), CUT)
    assert "#239 S3h FLIP UNTER KV-TOKEN-SCHNITT" in str(ei.value)
    assert "--weg2-disable-hicache" in str(ei.value)
    assert "--d-only" in str(ei.value)
    # a launch refusal like every other: the arm's handler catches it
    assert isinstance(ei.value, L.Weg2LaunchRefused)


@pytest.mark.parametrize("env_d, why", [
    ("SGLANG_MOE_SCRATCH_SLOTS=100,48,48", "not named"),
    ("SGLANG_WEG2_D_KV_STAGE_TOKENS=262144,393216,524288", "3 stages"),
])
def test_without_exactly_one_kv_stage_the_flip_is_refused(env_d, why):
    with pytest.raises(L.Weg2TokenCutFlipNotWired) as ei:
        L.refuse_flip_under_token_cut(_ns(env_d=env_d), CUT)
    assert why in str(ei.value)
    assert "S3g" in str(ei.value)


def test_an_unwired_seam_refuses_the_flip(monkeypatch):
    monkeypatch.setattr(rank_role, "unwired_token_cut_seams", lambda: ("F14",))
    with pytest.raises(L.Weg2TokenCutFlipNotWired) as ei:
        L.refuse_flip_under_token_cut(_ns(), CUT)
    assert "F14" in str(ei.value)


def test_d_only_boots_the_cut():
    assert L.refuse_flip_under_token_cut(_ns(d_only=True, env_d=""), CUT) is None


def test_the_dry_run_names_a_refusal_and_does_not_raise(capsys):
    line = L.refuse_flip_under_token_cut(_ns(dry_run=True, weg2_disable_hicache=True), CUT)
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
