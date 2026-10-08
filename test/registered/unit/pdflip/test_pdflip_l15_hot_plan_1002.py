# SPDX-License-Identifier: Apache-2.0
"""L15-10 S2: the hot-handover decision with REAL candidates (front side).

At a D->P flip the front knows, per queued follow-up, its session's previous
rid and the common token prefix (_sess_prev, the SESSION-PREFIX line). The
follow-up is HOT in D when that previous rid is still live on D (running or
parked: D holds its KV). front_candidates builds hot_handover.decide's input
from that, and plan_line names what a real handover would move -- the N1
measurement (how often hot, how many tokens) before any copy exists.
"""

from __future__ import annotations

from types import SimpleNamespace

from flliper.srt.pdflip import hot_handover as hh


def _q(rid):
    return SimpleNamespace(rid=rid)


def test_hot_when_the_session_predecessor_is_live_on_d():
    sess_prev = {"r2": ("r1", 4000), "r4": ("r3", 900), "r6": ("r5", 50)}
    cands = hh.front_candidates([_q("r2"), _q("r4"), _q("r6"), _q("r9")],
                                sess_prev, d_live={"r1", "r5"})
    assert [c["rid"] for c in cands] == ["r2", "r4", "r6", "r9"]
    assert [c["hot_in_d"] for c in cands] == [True, False, True, False]
    assert cands[0]["prefix_tokens"] == 4000 == cands[0]["anchor_depth"]
    assert cands[3]["prefix_tokens"] == 0


def test_plan_line_names_counts_and_the_chosen_handover():
    sess_prev = {"r2": ("r1", 4000), "r6": ("r5", 50)}
    cands = hh.front_candidates([_q("r2"), _q("r6")], sess_prev, d_live={"r1", "r5"})
    line = hh.plan_line(7, cands, p_free_rows=1 << 30)
    assert line.startswith("HOT-HANDOVER-PLAN epoch=7 waiting=2 hot=2 hot_tokens=4050")
    assert "chosen=r2 n=4000" in line


def test_no_hot_candidate_says_none():
    line = hh.plan_line(3, hh.front_candidates([_q("x")], {}, d_live=set()),
                        p_free_rows=1 << 30)
    assert "hot=0" in line and "chosen=none" in line
