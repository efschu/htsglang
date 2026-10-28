# SPDX-License-Identifier: Apache-2.0
"""27B-Port von FORMHP-DRAFT-PLAN-SKIP (NF-Commit 9e0b8d8cd): der DFLASH-Draft plant den von keiner Schicht gelesenen
FlashInfer-Wrapper nicht mehr.

CPU only. Der 27B-Draft (Qwen3.8-27B-DFlash2: 5 x sliding_attention, sliding_window 2048) wird ueber den echten
Layer-Parameter-Pfad (models.dflash._get_dflash_layer_attention_params) auf die Fensterwerte abgebildet, die die statische Pruefung
liest. Das Gate des Workers (_draft_plan_skip_probe_ids) wird auf Standins geprueft: NF band es an FLLIPER_FORMHP, im 27B-Baum gilt
stattdessen 'Solo-Host oder TP=1'. Rot ohne den Port: kein Modul dflash_plan_skip, keine skip/probe-Ids am Updater."""

import types

import pytest
import torch

from flliper.srt.models.dflash import _get_dflash_layer_attention_params
from flliper.srt.speculative import dflash_plan_skip as ps
from flliper.srt.speculative.dflash_worker_v2 import DFlashWorkerV2


class _Attn(torch.nn.Module):
    def __init__(self, layer_id, sws):
        super().__init__()
        self.layer_id = layer_id
        self.sliding_window_size = sws


class _Model(torch.nn.Module):
    def __init__(self, windows):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Attn(i, w) for i, w in enumerate(windows)])


def _windows_from_config(layer_types):
    cfg = types.SimpleNamespace(layer_types=layer_types, sliding_window=2048, is_causal=False)
    return [_get_dflash_layer_attention_params(cfg, i)[0] for i in range(len(layer_types))]


def test_27b_draft_config_leaves_full_wrapper_unread():
    """Der 27B-Draft (5 x sliding_attention) liest nur Wrapper 0; ein Draft mit einer full_attention-Schicht muss beide planen."""
    w27 = _windows_from_config(["sliding_attention"] * 5)
    assert w27 == [2047] * 5
    assert ps.unread_wrapper_ids(_Model(w27), 2, True) == (1,)
    mixed = _windows_from_config(["sliding_attention"] * 4 + ["full_attention"])
    assert ps.unread_wrapper_ids(_Model(mixed), 2, True) == ()


def _worker(*, solo_host=False, solo_shadow=False, tp_size=3, greedy=True, verdict=None):
    w = object.__new__(DFlashWorkerV2)
    w.__dict__["_spec_solo_active"] = solo_host or solo_shadow
    w.__dict__["_spec_solo_is_host"] = solo_host
    w.__dict__["draft_model_runner"] = types.SimpleNamespace(tp_size=tp_size, model=_Model([2047] * 5))
    state = ps.DraftPlanSkip()
    state.verdict = verdict
    w.__dict__["_draft_plan_skip"] = state
    updater = types.SimpleNamespace(
        skip_wrapper_ids=frozenset(), probe_wrapper_ids=frozenset(), attn_backend=types.SimpleNamespace(num_wrappers=2)
    )
    updater.update_sliding_window = lambda: None
    updater.update = updater.update_sliding_window
    w.__dict__["_draft_plan_skip_updater"] = lambda: updater
    sampling = types.SimpleNamespace(greedy=greedy)
    return w, sampling


@pytest.fixture(autouse=True)
def _greedy_stub(monkeypatch):
    from flliper.srt.speculative import dflash_worker_v2 as mod

    monkeypatch.setattr(mod, "_is_all_greedy", lambda si: si.greedy)
    monkeypatch.delenv(ps.ENV, raising=False)


def _ids(w, sampling, *, host_lens_exact=True, hole_buf=None):
    return w._draft_plan_skip_probe_ids(types.SimpleNamespace(sampling_info=sampling), host_lens_exact, hole_buf)


def test_gate_tp1_and_solo_host_probe_without_formhp_env():
    """Im 27B-Baum ist FLLIPER_FORMHP nicht gesetzt: TP=1-Draft und Solo-Host muessen trotzdem probieren."""
    w, s = _worker(tp_size=1)
    assert _ids(w, s) == (1,)
    w, s = _worker(solo_host=True, tp_size=3)
    assert _ids(w, s) == (1,)


@pytest.mark.parametrize(
    "kw, call",
    [
        (dict(tp_size=3), {}),  # TP-Draft: Kollektiv im Forward moeglich
        (dict(solo_shadow=True, tp_size=1), {}),  # Shadow faehrt keinen Draft
        (dict(tp_size=1, greedy=False), {}),  # Sampler wuerde doppelt ziehen
        (dict(tp_size=1), dict(host_lens_exact=False)),  # Geraeteplan: Probe-Layout ist Hostplan
        (dict(tp_size=1), dict(hole_buf=object())),  # Lochmaske wird nach dem ersten Forward genullt
        (dict(tp_size=1, verdict=True), {}),  # schon entschieden
    ],
)
def test_gate_refuses(kw, call):
    w, s = _worker(**kw)
    assert _ids(w, s, **call) == ()


def test_gate_switch_off(monkeypatch):
    monkeypatch.setenv(ps.ENV, "0")
    w, s = _worker(tp_size=1)
    assert _ids(w, s) == ()
