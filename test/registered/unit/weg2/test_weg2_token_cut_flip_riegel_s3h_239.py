"""#239 S3h: a FLIP boot under the token cut stops by name.

S3a-e and S4a/S4b wired the cut for group D (F4/F5/F12/F14, the launch riegel
``refuse_unwired_token_cut`` lets ``kv=qsa_forma_dcp`` through), but the P->D
hand-off under the cut is not built: P writes whole canonical pages, the D
workers must take their owner rows in every wake path, and the wake credit
per card moves with the KV bytes. Before this riegel nothing refused a flip
boot with the cut -- it would have booted unproven. ``--d-only`` (H87) keeps
booting; every other form is untouched.
"""

from __future__ import annotations

import inspect
import types

import pytest

from sglang.srt.weg2 import launcher as L

CUT = types.SimpleNamespace(kv="qsa_forma_dcp")


def _ns(**kw):
    ns = types.SimpleNamespace(dry_run=False, d_only=False, d_kv_token_cut="joint")
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_a_flip_boot_under_the_cut_is_refused_by_name():
    with pytest.raises(L.Weg2TokenCutFlipNotWired) as ei:
        L.refuse_flip_under_token_cut(_ns(), CUT)
    assert "#239 S3h FLIP UNTER KV-TOKEN-SCHNITT" in str(ei.value)
    assert "--d-only" in str(ei.value)
    # a launch refusal like every other: the arm's handler catches it
    assert isinstance(ei.value, L.Weg2LaunchRefused)


def test_d_only_boots_the_cut():
    assert L.refuse_flip_under_token_cut(_ns(d_only=True), CUT) is None


def test_the_dry_run_names_it_and_does_not_raise(capsys):
    line = L.refuse_flip_under_token_cut(_ns(dry_run=True), CUT)
    assert line and line.startswith("#239 S3h FLIP UNTER KV-TOKEN-SCHNITT")
    assert "(dry run: would refuse)" in capsys.readouterr().out


@pytest.mark.parametrize("form", [None, types.SimpleNamespace(kv="qsa_forma"),
                                  types.SimpleNamespace(kv="qsa")])
def test_every_other_form_is_untouched(form, capsys):
    assert L.refuse_flip_under_token_cut(_ns(), form) is None
    assert capsys.readouterr().out == ""


def test_main_calls_it_right_after_the_seam_riegel():
    src = inspect.getsource(L.main)
    i = src.index("refuse_unwired_token_cut(ns, boot_form)")
    assert src.index("refuse_flip_under_token_cut(ns, boot_form)", i) - i < 200
