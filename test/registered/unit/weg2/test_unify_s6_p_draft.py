"""UNIFY S6: the ``p_draft`` axis is the ONE writer of "does group P carry a
draft producer, and does it compute".

Gabel L2. Both lines decided "no draft COMPUTING on P, the space goes to KV /
experts", with two mechanics because the draft kinds differ:

* 27B (c60a5d387f, user 2026-09-24 "P ohne draft rechnen ... layer kalt dort
  liegen"): DFLASH producer BUILT on P's last stage, never asked --
  ``--dflash-produce-on-p off`` -> ``p_draft=cold``, draft.park=card;
* NF (H25, user 2026-09-24 08:25Z "Draft auf P streichen"): no MTP head on P
  at all, D parks its draft in pinned host RAM -- SGLANG_WEG2_DRAFT_ON_P=0 ->
  ``p_draft=none``, draft.park=host.

On this tree weg2/form.resolve_form answers it ONCE (the H25 rule injected,
the profile row's ``p_draft`` as the default, the env and both flags as
aliases) and the launcher reads ``ns.weg2_draft_on_p`` instead of resolving a
second time. Hermetic, CPU.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import shlex
import tempfile

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_weg2_form_axes import _nf_words  # noqa: E402

from sglang.srt.weg2 import form as F  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="stage-a-weg2-unit")


@pytest.fixture
def models():
    with tempfile.TemporaryDirectory() as root:
        out = {}
        for name, moe in (("Qwen3.8-27B-INT8-gdncov-vocabembed", False),
                          ("Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist", True)):
            path = os.path.join(root, name)
            os.makedirs(path)
            cfg = {"architectures": ["X"], "model_type": "x", "text_config": {"model_type": "x_text"}}
            if moe:
                cfg["text_config"].update({"num_experts": 512, "num_experts_per_tok": 10})
            with open(os.path.join(path, "config.json"), "w") as f:
                json.dump(cfg, f)
            out["moe" if moe else "dense"] = path
        yield out


def _q27(model, *extra):
    return ["--tree", "/tmp", "--tag", "t", "--model", model, "--spec-form", "DFLASH",
            "--weg2-weight-source", "exchange", *extra]


def _nf(model, *extra):
    """test_weg2_form_axes' NF arm without its --draft-kv-on-p pair."""
    words = _nf_words(model)
    i = words.index("--draft-kv-on-p")
    return words[:i] + words[i + 2:] + list(extra)


def _resolve(words, env=(False, False)):
    ns = L.build_parser().parse_args(words)
    form = F.resolve_form(ns, words, parse_group_env=L.parse_group_env, shlex_split=shlex.split,
                          draft_on_p_rule=L.resolve_draft_on_p, draft_on_p_env=env)
    return ns, form


# ------------------------------------------------------------- the profiles
def test_profile_rows_name_the_two_forms_and_their_park():
    q, n = F.PROFILES["qwen27b"], F.PROFILES["nextflash"]
    assert (q.p_draft, q.draft.park) == ("cold", "card")
    assert (n.p_draft, n.draft.park) == ("none", "host")
    for row in F.PROFILES.values():
        # a parked draft (host) is exactly the profile without a producer on P
        assert (row.draft.park == "host") == (row.p_draft == "none"), row.id


# ------------------------------------------------------------- 27B profile
def test_27b_default_is_cold_producer_built(models):
    ns, form = _resolve(_q27(models["dense"]))
    assert form.p_draft == "cold"
    assert ns.draft_kv_on_p == "on" and ns.dflash_produce_on_p == "off"
    assert ns.weg2_draft_on_p[0] is True


def test_27b_produce_on_is_compute_and_env_zero_is_none(models):
    ns, form = _resolve(_q27(models["dense"], "--dflash-produce-on-p", "on"))
    assert form.p_draft == "compute"
    ns, form = _resolve(_q27(models["dense"]), env=(True, False))
    assert form.p_draft == "none" and ns.draft_kv_on_p == "off"
    ns, form = _resolve(_q27(models["dense"], "--draft-kv-on-p", "off"))
    assert form.p_draft == "none" and ns.weg2_draft_on_p[0] is False


def test_27b_explicit_on_is_honoured_not_overruled(models):
    ns, form = _resolve(_q27(models["dense"], "--draft-kv-on-p", "on"))
    assert form.p_draft == "cold" and "OVERRULES" not in ns.weg2_draft_on_p[1]


# ------------------------------------------------------------- NF profile
def test_nf_default_is_no_producer_h25(models):
    ns, form = _resolve(_nf(models["moe"]))
    assert form.p_draft == "none" and ns.draft_kv_on_p == "off"


def test_nf_explicit_on_is_overruled_by_h25_and_the_form_says_so(models):
    """The two-writer defect: the form used to publish p_draft=compute from
    the CLI while H25 later built no producer. Now the form IS the H25 answer."""
    ns, form = _resolve(_nf(models["moe"], "--draft-kv-on-p", "on"))
    assert form.p_draft == "none"
    assert "OVERRULES" in ns.weg2_draft_on_p[1]
    ns, form = _resolve(_nf(models["moe"], "--draft-kv-on-p", "on"), env=(True, True))
    assert form.p_draft == "compute"


def test_nf_release_arm_is_unchanged(models):
    """nf.env passes --draft-kv-on-p off: none, the value H25 always wrote."""
    ns, form = _resolve(_nf(models["moe"], "--draft-kv-on-p", "off"))
    assert form.p_draft == "none" and ns.draft_kv_on_p == "off"
    on, why = L.resolve_draft_on_p("off", True, False, False)
    assert ns.weg2_draft_on_p == (on, why)


def test_env_against_explicit_flag_is_still_w127(models):
    with pytest.raises(L.Weg2LaunchRefused, match="W127"):
        _resolve(_nf(models["moe"], "--draft-kv-on-p", "on"), env=(True, False))


def test_stated_form_outranks_the_default_and_refuses_a_contradicting_env(models):
    ns, form = _resolve(_nf(models["moe"], "--form-p-draft", "compute"))
    assert form.p_draft == "compute"
    with pytest.raises(F.Weg2FormContradiction, match="W140"):
        _resolve(_nf(models["moe"], "--form-p-draft", "none"), env=(True, True))


# ------------------------------------------------------------- one writer
def test_the_launcher_reads_the_axis_and_does_not_resolve_twice():
    src = inspect.getsource(L.main)
    assert "draft_on_p_rule=resolve_draft_on_p" in src
    assert 'getattr(ns, "weg2_draft_on_p", None)' in src
    assert src.index("draft_on_p_rule=resolve_draft_on_p") < src.index(
        'getattr(ns, "weg2_draft_on_p", None)')


def test_one_producer_detection_and_one_produce_switch():
    tree = ast.parse(inspect.getsource(L))
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    for name in ("p_group_has_draft_producer", "dflash_produce_on_p", "resolve_draft_on_p",
                 "gate_w10", "spec_form_env"):
        assert names.count(name) == 1, name


def test_cold_producer_is_no_draft_page_producer(tmp_path):
    saved = dict(L._SPEC_FORM)
    try:
        L.apply_spec_form(type("NS", (), dict(
            spec_form="DFLASH", dflash_draft_path=str(tmp_path), dflash_block=8,
            dflash_window=2048, dflash_produce_on_p="off"))())
        L._SPEC_FORM["draft_kv_on_p"] = True
        assert L.p_group_has_draft_producer() is False
        assert L.p_produces_draft_pages() is False
        assert L.spec_form_env("P") == {"SGLANG_WEG2_DFLASH_PRODUCE": "0"}
    finally:
        L._SPEC_FORM.clear()
        L._SPEC_FORM.update(saved)
