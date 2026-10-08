"""HANDBACK on NF (01.10., user via 27B: "den anchor fix koennen alle brauchen").

NF's P->D contract is E2 (P's END state after N + its token, D computes 0);
measured on 12 NF boots, 492 of 565 hand-offs ran it. What this pins:
* a hand-off store read on group D is read whatever its length (min_tokens 1):
  NF metal had 33x '#915 PREFETCH REFUSED vote_negative need=64..255
  keys=handoff' -- P's own pages refused as too short, prefilled again on D;
* the PDFLIP-HANDBACK line names d_prefix / d_compute per path (skip = N / 0,
  e1 = cut, an extend from the page anchor otherwise -- metal pdflip-4-15:
  N=41464, page anchor 41408, adopt=skipped:end_only:batch_not_empty, 56);
* what NF deliberately does NOT take from 27B fe5c55041b: D's claim stays the
  upstream N-1 raw tokens (P's CLAIM ANCHOR floor_page(N-2) is the same
  geometry), and the L3 store identity stays byte for byte (no anchor_keying).
"""

import inspect
import logging
from types import SimpleNamespace

import pytest

from flliper.srt.pdflip import handback_claim as hc
from flliper.srt.pdflip import tail_adopt as ta

N, PAGE_PREFIX, CUT = 41464, 41408, 41460


@pytest.mark.parametrize(
    "env,has_handoff,want",
    [
        ({"FLLIPER_PDFLIP_GROUP": "D"}, True, 1),
        ({"FLLIPER_PDFLIP_GROUP": "d"}, True, 1),
        ({"FLLIPER_PDFLIP_GROUP": "D"}, False, None),  # D's own reads keep #915
        ({"FLLIPER_PDFLIP_GROUP": "P"}, True, None),
        ({}, True, None),
        ({"FLLIPER_PDFLIP_GROUP": "D", "FLLIPER_PDFLIP_DUAL_LAYOUT": "1"}, False, 1),
    ],
)
def test_handback_min_tokens(env, has_handoff, want):
    assert hc.handback_min_tokens(has_handoff=has_handoff, env=env) == want


def test_prefetch_wiring_reads_handoff_below_threshold():
    from flliper.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler._prefetch_kvcache)
    assert "_pdflip_hb_handoff = bool(_hd)" in src
    assert "handback_min_tokens(has_handoff=_pdflip_hb_handoff)" in src
    # the hand-off minimum lowers, never raises, a caller's own minimum
    assert "min(int(_tail_min), int(_hb_min))" in src
    # and it is computed before the kwargs go to the tree
    assert src.index("handback_min_tokens(") < src.index('_tail_kw = {"min_tokens"')


def _entry(skip=True, agreed=True, e1=False):
    spec = SimpleNamespace(n_tokens=N, cut=CUT, page_prefix=PAGE_PREFIX, rows=N - PAGE_PREFIX, extend=N - CUT)
    staged = SimpleNamespace(spec=spec, e1=e1, drop_end=lambda: None)
    return SimpleNamespace(staged=staged, agreed=agreed, skip=skip, skip_note="", waited_ms=0.0)


@pytest.fixture
def lines(monkeypatch):
    got = []
    monkeypatch.setattr(ta, "_log_ready", lambda *a, **k: None)
    monkeypatch.setattr(ta, "_is_park", lambda staged: False)
    monkeypatch.setattr(ta, "adopt_ids", lambda req, park: [0] * N)
    monkeypatch.setattr(ta, "uniform_refusal", lambda *a, **k: "")
    monkeypatch.setattr(hc, "handback_line", lambda *a: got.append(a) or "")
    return got


def _req():
    return SimpleNamespace(rid="pdflip-4-15", full_untruncated_fill_ids=[0] * N, extra_key=None)


def _run(entry, batch_empty, monkeypatch, skip_why=""):
    monkeypatch.setattr(ta, "skip_refusal", lambda e, r, b: skip_why)
    out, path = ta._plan_adopt_entry(entry, _req(), PAGE_PREFIX, batch_empty)
    ta._handback(_req(), entry, PAGE_PREFIX, out, path)
    return out, path


def test_e2_skip_holds_n_computes_zero(lines, monkeypatch):
    out, path = _run(_entry(skip=True), True, monkeypatch)
    assert out is not None and path == "skip"
    assert lines == [("pdflip-4-15", N, N, 0, "skip")]


def test_end_only_batch_not_empty_is_the_page_anchor_extend(lines, monkeypatch):
    # metal pdflip-4-15 (f20011a9fd 15:37:17): X-GATE uncached=56,
    # TAIL-READY adopt=skipped:end_only:batch_not_empty
    out, path = _run(_entry(skip=True, e1=False), False, monkeypatch, skip_why="batch_not_empty")
    assert out is None and path == "extend:end_only:batch_not_empty"
    assert lines == [("pdflip-4-15", N, PAGE_PREFIX, N - PAGE_PREFIX, path)]
    assert N - PAGE_PREFIX == 56


def test_e1_holds_cut(lines, monkeypatch):
    out, path = _run(_entry(skip=True, e1=True), False, monkeypatch, skip_why="batch_not_empty")
    assert out is not None and not out.skip and path == "e1:batch_not_empty"
    assert lines == [("pdflip-4-15", N, CUT, N - CUT, path)]


def test_group_vote_refusal_named(lines, monkeypatch):
    out, path = _run(_entry(agreed=False), True, monkeypatch)
    assert out is None and path == "extend:group_vote"
    assert lines[0][2:] == (PAGE_PREFIX, N - PAGE_PREFIX, "extend:group_vote")


def test_handback_line_text(caplog):
    with caplog.at_level(logging.INFO, logger=hc.__name__):
        line = hc.handback_line("pdflip-0-3", 65042, 65042, 0, "skip")
    assert line.startswith("PDFLIP-HANDBACK rid=pdflip-0-3 N=65042 d_prefix=65042 d_compute=0 path=skip")
    assert "PDFLIP-HANDBACK" in caplog.text


def test_instrument_never_gates(monkeypatch):
    # a broken line must not change the admission's answer
    monkeypatch.setattr(hc, "handback_line", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    ta._handback(_req(), _entry(), PAGE_PREFIX, None, "extend:x")


def test_nf_d_claim_stays_upstream(monkeypatch):
    # 27B fe5c55041b widens an exact-bigram D to N raw tokens; NF keeps N-1
    # (P's CLAIM ANCHOR floor_page(N-2) is filed for exactly this claim)
    from flliper.srt.managers.schedule_batch import Req
    from flliper.srt.pdflip import form

    # RELEASE-HEAD 1002: the 27B claim is in this tree too, gated by the
    # profile (handback_claim_n) -- the NF form names its profile
    monkeypatch.setattr(form, "current_form", lambda environ=None: SimpleNamespace(profile="nextflash"))
    monkeypatch.delenv("FLLIPER_PDFLIP_HANDBACK_CLAIM_N", raising=False)
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_PDFLIP_BIGRAM_ANCHOR_EXACT", "1")
    me = SimpleNamespace(return_logprob=False, logprob_start_len=-1)
    assert Req._compute_max_prefix_len(me, N) == N - 1


def test_nf_l3_identity_unchanged():
    from flliper.srt.pdflip.launcher import l3_persist_identity

    ident = l3_persist_identity("", profile="nextflash")
    assert "anchor_keying" not in ident
    assert set(ident) == {
        "model_path", "model_config_sha1", "weights_fp", "profile", "form_kv",
        "kv_cache_dtype", "override_p", "override_d", "vision", "generation",
    }


# RELEASE-HEAD 1002: the 27B N-claim and NF's upstream claim live in one tree.
# Both profiles key exact bigrams, so the tree note (BIGRAM_EXACT_TREE) cannot
# tell them apart; the profile switch FLLIPER_PDFLIP_HANDBACK_CLAIM_N does.


def test_profile_rows_split_the_claim():
    from flliper.srt.pdflip.form import PROFILE_SWITCH_DEFAULTS

    assert PROFILE_SWITCH_DEFAULTS["nextflash"]["FLLIPER_PDFLIP_HANDBACK_CLAIM_N"] is False
    assert PROFILE_SWITCH_DEFAULTS["qwen27b"]["FLLIPER_PDFLIP_HANDBACK_CLAIM_N"] is True


@pytest.mark.parametrize("profile,want", [("nextflash", False), ("qwen27b", True)])
def test_nf_d_on_an_exact_tree_keeps_the_upstream_claim(monkeypatch, profile, want):
    from flliper.srt.pdflip import form

    monkeypatch.setattr(hc, "BIGRAM_EXACT_TREE", [True])  # the NF tree IS exact
    monkeypatch.setattr(form, "current_form", lambda environ=None: SimpleNamespace(profile=profile))
    env = {"FLLIPER_PDFLIP_GROUP": "D"}
    assert hc.handback_bigram_claim(env=env) is want
    # an explicit value wins over the row
    assert hc.handback_bigram_claim(env={**env, "FLLIPER_PDFLIP_HANDBACK_CLAIM_N": "0"}) is False
    assert hc.handback_bigram_claim(env={**env, "FLLIPER_PDFLIP_HANDBACK_CLAIM_N": "1"}) is True


def test_nf_req_claim_on_an_exact_tree(monkeypatch):
    from flliper.srt.managers.schedule_batch import Req
    from flliper.srt.pdflip import form

    monkeypatch.setattr(hc, "BIGRAM_EXACT_TREE", [True])
    monkeypatch.setattr(form, "current_form", lambda environ=None: SimpleNamespace(profile="nextflash"))
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.delenv("FLLIPER_PDFLIP_HANDBACK_CLAIM_N", raising=False)
    me = SimpleNamespace(return_logprob=False, logprob_start_len=-1)
    assert Req._compute_max_prefix_len(me, N) == N - 1
