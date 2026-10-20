# SPDX-License-Identifier: Apache-2.0
"""S78 D-DECODE-GAP: a gap opened by a MIXED-D round is named ``mixed_prefill`` with its terms.

Metal (rc17 D ``boot_weg2_..._7d93470b5e_1019_171636.D.log`` 17:23:26-17:23:30, TP0): decode round 398 starts
at 606.471; MIXED-D n=9 (1533 prefill tokens, 4 decode rows, ``D-LEAD-MS extend_ms=1792430606849``) runs its
forward until ``LOOP-ITER-SLOW iter=3064 run_ms=2408.0 t=...609.258``; MIXED-D n=10 (307 tokens, 5 rows)
starts at 609.357 -> the line ``D-DECODE-GAP gap_ms=2508 cause=other ... other_ms=2508`` blamed nothing, and
it stood next to the n=10 round although the 2408 ms were n=9's forward. n=10 then ran to the decode round 401
at 610.462 (gap 1104/1105, also ``cause=other``). 388 of rc17's 703 ``other`` gaps (473.9 of 648.9 s) were
opened by a MIXED-D round.
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


def _mixed(prefill, rows):
    return SimpleNamespace(forward_mode=_Mode("extend"), extend_num_tokens=prefill + rows,
                           decoding_reqs=list(range(rows)))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    for e in (G.ENV, G.ENV_MS, G.ENV_CENSUS_S):
        monkeypatch.delenv(e, raising=False)


def _lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(G.MARK + " gap_ms")]


def test_metal_rc17_172329_the_mixed_round_that_opened_the_gap_is_named(caplog):
    s = SimpleNamespace(tp_rank=0, _pdflip_wake_seq=4)
    with caplog.at_level(logging.INFO, logger=G.__name__):
        G.on_batch(s, _decode(4), now=606.471)
        G.on_run_end(s, _decode(4), now=606.475)
        G.on_batch(s, _mixed(1533, 4), now=606.849)          # n=9
        G.on_run_end(s, _mixed(1533, 4), now=609.257)        # its forward blocks the host to its end
        G.on_batch(s, _mixed(307, 5), now=609.357)           # n=10
        G.on_run_end(s, _mixed(307, 5), now=610.362)
        G.on_batch(s, _decode(6), now=610.462)               # round 401
    first, second = _lines(caplog)
    assert "gap_ms=2508 cause=mixed_prefill rows=4" in first
    assert "mixed_tokens=1533 run_ms=2408 host_ms=100" in first
    assert "gap_ms=1105 cause=mixed_prefill rows=5" in second and "mixed_tokens=307 run_ms=1005" in second


def test_a_decode_opened_gap_with_a_lend_is_named_lend(caplog):
    s = SimpleNamespace(tp_rank=0, _pdflip_wake_seq=4)
    with caplog.at_level(logging.INFO, logger=G.__name__):
        G.on_batch(s, _decode(4), now=100.000)
        G.on_run_end(s, _decode(4), now=100.004)
        G.note(s, "lend")                                    # D-TRANSIENT-LEND: 720 cells mapped
        G.on_batch(s, _decode(4), now=100.412)
    (line,) = _lines(caplog)
    assert "cause=lend " in line and "mixed_tokens=0 run_ms=4 host_ms=408" in line and "notes=lend:1" in line


def test_flip_still_wins_and_pure_extends_keep_their_name(caplog):
    s = SimpleNamespace(tp_rank=0, _pdflip_wake_seq=1)
    with caplog.at_level(logging.INFO, logger=G.__name__):
        G.on_batch(s, _mixed(300, 4), now=10.0)
        G.on_run_end(s, _mixed(300, 4), now=10.8)
        s._pdflip_wake_seq = 2
        G.on_batch(s, _decode(4), now=40.0)
        G.on_run_end(s, _decode(4), now=40.01)
        G.on_batch(s, SimpleNamespace(forward_mode=_Mode("extend"), extend_num_tokens=2224, decoding_reqs=[]),
                   now=40.1)
        G.on_batch(s, _decode(4), now=43.2)
    flip, d_pre = _lines(caplog)
    assert "cause=flip" in flip
    assert "cause=d_prefill" in d_pre and "mixed_tokens=0" in d_pre and "host_ms=" in d_pre


def test_the_census_counts_mixed_prefill(caplog):
    s = SimpleNamespace(tp_rank=0, _pdflip_wake_seq=4)
    tr = G.tracker(s)
    tr.census_s = 1.0
    with caplog.at_level(logging.INFO, logger=G.__name__):
        G.on_batch(s, _decode(4), now=tr.window_t0 + 0.1)
        G.on_batch(s, _mixed(200, 4), now=tr.window_t0 + 0.2)
        G.on_run_end(s, _mixed(200, 4), now=tr.window_t0 + 0.9)
        G.on_batch(s, _decode(4), now=tr.window_t0 + 1.2)
    cen = [r.getMessage() for r in caplog.records if "CENSUS" in r.getMessage()]
    assert cen and "mixed_prefill:1/" in cen[-1]


def test_run_end_never_breaks_the_pass():
    G.on_run_end(SimpleNamespace(tp_rank=0), object())
    G.on_run_end(None, None)
