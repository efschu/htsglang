"""F15 (4): the weg2 launcher calls rank_form.check_form_b_windows for a Form B
D group (NF answer 4 posts), before any rank loads."""

import json
import types

import pytest

from sglang.srt.weg2 import launcher as L
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CARDS = [L.Card(1, "GPU-5090", "NVIDIA GeForce RTX 5090", 32607),
         L.Card(0, "GPU-3080x8", "NVIDIA GeForce RTX 3080", 20480),
         L.Card(2, "GPU-3080x4", "NVIDIA GeForce RTX 3080", 20480)]


def _ns(tmp_path, extra_d, moe=False):
    (tmp_path / "config.json").write_text(json.dumps(
        {"text_config": {"num_experts_per_tok": 8 if moe else 0}}))
    return types.SimpleNamespace(extra_d=extra_d, model=str(tmp_path))


def test_not_form_b_is_silent(tmp_path):
    for extra in ("", "--rank-tp-ratio 3,1,1", "--rank-tp-ratio 1,0,0 --rank-role host,worker,worker"):
        assert L.refuse_form_b_windows(_ns(tmp_path, extra), CARDS) is None


def test_dense_form_b_today_spec_fits_exactly_and_the_form_b_spec_leaves_8(tmp_path):
    line = L.refuse_form_b_windows(_ns(tmp_path, "--rank-tp-ratio 3,1,0 --uneven-token-vector 1,2,2"),
                                   CARDS)
    assert "dense" in line and "RTX 3080 224/224 MiB (headroom 0)" in line
    line = L.refuse_form_b_windows(_ns(
        tmp_path, "--rank-tp-ratio 3,1,0 --barlink-bar1-window-mib 16,TP_0=8,MODEL_TP_0=32,DCP_0=40"),
        CARDS)
    assert "RTX 3080 216/224 MiB (headroom 8)" in line


def test_a_card_that_does_not_fit_is_refused_by_name(tmp_path):
    with pytest.raises(L.Weg2FormBWindowRefused, match=r"W187.*GPU-3080x8.*--barlink-bar1-window-mib "
                       r"16,TP_0=8,MODEL_TP_0=32,DCP_0=40"):
        L.refuse_form_b_windows(_ns(
            tmp_path, "--rank-tp-ratio 3,1,0 --barlink-bar1-window-mib 16,TP_0=32,MODEL_TP_0=32,DCP_0=40"),
            CARDS)
    with pytest.raises(L.Weg2FormBWindowRefused, match="TP_0 8 MiB < 32 under MoE"):
        L.refuse_form_b_windows(_ns(
            tmp_path, "--rank-tp-ratio 3,1,0 --barlink-bar1-window-mib 16,TP_0=8,MODEL_TP_0=32,DCP_0=40",
            moe=True), CARDS)
    with pytest.raises(L.Weg2FormBWindowRefused, match="one rank per card"):
        L.refuse_form_b_windows(_ns(tmp_path, "--rank-tp-ratio 3,1,0,0"), CARDS)


def test_the_launcher_calls_it_where_the_cards_are_known():
    import inspect

    src = inspect.getsource(L)
    i = src.index("    state.cvd = cvd\n")
    assert "refuse_form_b_windows(ns, cards)" in src[i:i + 400]
    assert src.index("host_preflight(log, ns.tag, dry)") < i        # the boot's preflight, before ranks
