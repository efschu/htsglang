# SPDX-License-Identifier: Apache-2.0
"""S48 D-DECODE-GAP: the decode pauses of group D with their cause (pdflip/d_decode_gap.py).

Metal shapes (rc13 D.log ``..._6767e25d89_1018_200305``, TP0):

* 20:22:20.491 ``Decode rank batch #round 6305 bs 4`` -> extend pdflip-14-153 (2224 new, 90688 cached, ~3.1 s) ->
  extend pdflip-14-155 (449 new, MIXED-D with 5 decode rows, 1.36 s) -> 20:22:24.946 round 6308 bs 6: one pause of
  4.455 s that no ``Decode batch`` line (every ~5 s) could resolve.
* 207 pauses around a single extend of 0-256 tokens, median 1.22 s each (the D extend's expert stream).
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from flliper.srt.pdflip import d_decode_gap as G


class _Mode:
    def __init__(self, kind):
        self.kind = kind

    def is_decode(self):
        return self.kind == "decode"

    def is_extend(self):
        return self.kind == "extend"


def _decode(bs):
    return SimpleNamespace(forward_mode=_Mode("decode"), batch_size=lambda: bs)


def _extend(tokens, decoding=()):
    return SimpleNamespace(forward_mode=_Mode("extend"), extend_num_tokens=tokens, decoding_reqs=list(decoding))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    for e in (G.ENV, G.ENV_MS, G.ENV_CENSUS_S):
        monkeypatch.delenv(e, raising=False)


def _sched(rank=0, wake=3):
    return SimpleNamespace(tp_rank=rank, _pdflip_wake_seq=wake)


def test_metal_two_extends_one_pause(caplog):
    s = _sched()
    with caplog.at_level(logging.INFO, logger=G.__name__):
        G.on_batch(s, _decode(4), now=940.491)
        G.on_batch(s, _extend(2224), now=940.60)                       # 14-153, pure extend
        G.on_batch(s, _extend(449, decoding=[1, 2, 3, 4, 5]), now=943.58)  # 14-155 MIXED-D: decode rows ride
    rec = [r.getMessage() for r in caplog.records if r.getMessage().startswith(G.MARK + " gap_ms")]
    assert len(rec) == 1
    line = rec[0]
    assert "cause=d_prefill" in line and "extends=1" in line and "prefill_tokens=2224" in line and "rows=4" in line
    assert "gap_ms=3089" in line and "d_prefill_ms=2980" in line


def test_short_extend_pause_is_named():
    tr = G.Tracker(400, 0, 0.0)
    assert tr.on_batch(0.0, 6, 0, 1) is None
    assert tr.on_batch(0.11, 6, 0, 1) is None                         # a plain round: no pause
    tr.on_batch(0.22, 0, 188, 1)                                       # one R2 short, 188 tokens
    rec = tr.on_batch(1.44, 6, 0, 1)
    assert rec["cause"] == "d_prefill" and rec["prefill_tokens"] == 188 and rec["extends"] == 1
    assert round(rec["gap_ms"]) == 1330 and round(rec["d_prefill_ms"]) == 1220 and round(rec["other_ms"]) == 110


def test_flip_between_rounds():
    tr = G.Tracker(400, 0, 0.0)
    tr.on_batch(0.0, 5, 0, 7)
    rec = tr.on_batch(30.0, 5, 0, 8)                                   # D slept, P epoch, woke
    assert rec["cause"] == "flip" and rec["flip"] is True


def test_noted_causes_without_extend():
    tr = G.Tracker(400, 0, 0.0)
    tr.on_batch(0.0, 3, 0, 1)
    tr.note("skipfirst_hold")
    tr.note("seat")
    rec = tr.on_batch(0.9, 3, 0, 1)
    assert rec["cause"] == "skipfirst_hold" and rec["notes"] == {"skipfirst_hold": 1, "seat": 1}
    tr.note("wake_read")
    rec = tr.on_batch(1.8, 3, 0, 1)
    assert rec["cause"] == "wake_read"
    rec = tr.on_batch(2.7, 3, 0, 1)
    assert rec["cause"] == "other" and rec["notes"] == {}


def test_idle_group_is_not_a_pause():
    tr = G.Tracker(400, 0, 0.0)
    tr.on_batch(0.0, 0, 500, 1)                                        # an extend with nothing decoding
    assert tr.on_batch(5.0, 1, 0, 1) is None                           # first round of the new request
    tr.last_rows = 0
    assert tr.on_batch(9.0, 1, 0, 1) is None


def test_census_window():
    tr = G.Tracker(400, 60, 0.0)
    tr.on_batch(0.0, 2, 0, 1)
    tr.on_batch(0.5, 0, 100, 1)
    tr.on_batch(2.0, 2, 0, 1)
    assert tr.census_due(30.0) is None
    cen = tr.census_due(61.0)
    assert cen["gaps"] == 1 and cen["rounds"] == 2 and cen["by_cause"]["d_prefill"][0] == 1
    assert tr.census_due(62.0) is None and tr.gaps == 0


def test_only_group_d_first_rank_and_switch(monkeypatch):
    assert G.tracker(_sched(rank=1)) is None
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    assert G.tracker(_sched()) is None
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv(G.ENV, "0")
    assert G.tracker(_sched()) is None
    monkeypatch.delenv(G.ENV)
    assert isinstance(G.tracker(_sched()), G.Tracker)
    assert G.enabled() is True


def test_never_breaks_the_pass():
    s = _sched()
    G.on_batch(s, SimpleNamespace(), now=1.0)                          # no forward_mode
    G.on_batch(s, SimpleNamespace(forward_mode=_Mode("decode"), batch_size=None), now=2.0)
    G.note(SimpleNamespace(__slots__=()), "seat")                      # unsettable scheduler


def test_bound_env(monkeypatch):
    monkeypatch.setenv(G.ENV_MS, "1000")
    tr = G.tracker(_sched())
    assert tr.bound_ms == 1000.0
    tr.on_batch(0.0, 2, 0, 1)
    assert tr.on_batch(0.9, 2, 0, 1) is None
