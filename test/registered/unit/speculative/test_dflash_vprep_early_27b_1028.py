# SPDX-License-Identifier: Apache-2.0
"""27B-Port von FORMHP-VPREP-EARLY (NF-Commit 162200e79): Verify-Vorbau einer Early-Draft-Runde vor dem Laengen-Wait.

CPU only. Die Ablauflogik (alte Reihenfolge, Selbsttest, scharf, Eager-Wiederholung) pinnen die NF-Tests
(test_formhp_devdraft_vprep_1028.py, laufen im 27B-Baum unveraendert). Hier: die Rundenzaehler (am Metall ist die Wandzeitwirkung offen --
die Zaehler sagen, wie viele Runden welchen Weg nahmen) und die Verdrahtung im 27B-Worker (Wait + Vorbau laufen ueber staged_prep, der
Schalter-Default ist AN, die Eignung haengt am Early-Draft der Runde). Rot ohne den Port: kein Modul dflash_vprep_early."""

import inspect
import logging

import torch

from flliper.srt.speculative import dflash_vprep_early as ve
from flliper.srt.speculative.dflash_worker_v2 import DFlashWorkerV2


class _FB:
    def __init__(self, bound):
        self.seq_lens_cpu = bound.clone()
        self.seq_lens_sum = int(bound.sum())


class _Runner:
    buffer_registry = None
    attn_backend = None


def _drive(state, *, eligible=True, graph=True):
    exact = {}

    def prep():
        b = exact["v"] + 8 if "v" in exact else torch.tensor([30, 40])
        return ("vi", _FB(b), None, graph)

    def wait():
        exact["v"] = torch.tensor([20, 25])

    return ve.staged_prep(state=state, eligible=eligible, prep=prep, marks=lambda: None, wait=wait,
                          exact_lens=lambda: exact.get("v"), runner=_Runner(), block_size=8, bs=2)


def test_counters_follow_the_path_of_each_round(monkeypatch):
    """old: switch off / not eligible; selftest: the first SELFTEST_ROUNDS graph rounds; early: armed; redo: armed round whose prep chose eager."""
    monkeypatch.delenv(ve.ENV, raising=False)
    st = ve.VprepEarly()
    _drive(st, eligible=False)
    for _ in range(ve.SELFTEST_ROUNDS):
        _drive(st)
    assert st.verdict is True
    _drive(st)
    _drive(st)
    _drive(st, graph=False)
    assert st.counts == {"old": 1, "selftest": ve.SELFTEST_ROUNDS, "early": 2, "redo": 1}, st.counts
    monkeypatch.setenv(ve.ENV, "0")
    st2 = ve.VprepEarly()
    _drive(st2)
    assert st2.counts["old"] == 1 and sum(st2.counts.values()) == 1


def test_counter_line_is_logged_every_LOG_EVERY_rounds(monkeypatch, caplog):
    monkeypatch.setattr(ve, "LOG_EVERY", 4)
    monkeypatch.delenv(ve.ENV, raising=False)
    st = ve.VprepEarly()
    with caplog.at_level(logging.INFO, logger=ve.logger.name):
        for _ in range(9):
            _drive(st, eligible=False)
    lines = [r.getMessage() for r in caplog.records if "rounds=" in r.getMessage()]
    assert len(lines) == 2 and "rounds=4" in lines[0] and "rounds=8" in lines[1], lines


def test_switch_default_is_on(monkeypatch):
    monkeypatch.delenv(ve.ENV, raising=False)
    assert ve.switch_on() is True
    assert ve.VprepEarly().mode() == "selftest"
    monkeypatch.setenv(ve.ENV, "off")
    assert ve.VprepEarly().mode() is None


def test_worker_runs_the_verify_prep_through_staged_prep_for_early_draft_rounds():
    """Wiring pin: the length wait and the verify prep of forward_batch_generation go through staged_prep; the eligibility needs the
    round's early draft, the deferred lengths (batch.seq_lens_cpu still None) and no grammar; the TP sync runs once, before the prep."""
    src = inspect.getsource(DFlashWorkerV2.forward_batch_generation)
    assert "_vpe_mod.staged_prep(" in src
    elig = src[src.index("eligible=bool(") : src.index("prep=_verify_prep")]
    for needle in ("seq_lens_cpu_ready is not None", "_early_draft", "batch.seq_lens_cpu is None", "has_grammar"):
        assert needle in elig, needle
    section = src[src.index("# --- 2) Target verify.") : src.index("_vpe_mod.staged_prep(")]
    # a self-test round prepares twice; the collective must not run twice: one sync in the verify section, outside _verify_prep
    assert section.count("SpecTpSyncSite.DFLASH_SELECTOR") == 1
    assert "SpecTpSyncSite.DFLASH_SELECTOR" not in section[section.index("def _verify_prep():") : section.index("self._tp_sync.sync(")]
    # the prep itself no longer sits inline behind the wait
    assert src.index("def _verify_prep():") < src.index("_vpe_mod.staged_prep(")


def test_worker_state_is_per_worker():
    a = object.__new__(DFlashWorkerV2)
    b = object.__new__(DFlashWorkerV2)
    assert a._vprep_early_state() is a._vprep_early_state()
    assert a._vprep_early_state() is not b._vprep_early_state()
