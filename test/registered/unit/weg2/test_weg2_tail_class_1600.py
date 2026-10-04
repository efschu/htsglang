# SPDX-License-Identifier: Apache-2.0
"""#1600 instrument: the refused D-HANDBACK-DEFER line and the W50-REROUTE x_refusal line carry a
class label (tail_class / route_class). LOG ONLY: no decision reads the label.

Boot data (b9h d31609a726, b9i 6b54506928, b9j 8e41596ed4; DEFER_REARM=4 on): of 6+1+0 x_refusal
reroutes in ~430 routes, 5 were whole_short/short_credit (tail=70..72, front_price=1 from an l3_index
credit, 2.1 s bound, nothing for a re-read to land) and 1 was anchor_tail (tail=4061, rearm 4/4 used in
862 ms); the old TAIL class (16 of 109 in boot B9) is gone with the re-arm.

DANGER DIRECTIONS (mutants run by hand, red): label used as a gate (begin returns by class) -> the
'label never changes the verdict' test fails; thresholds swapped -> the boundary tests fail.
"""
from __future__ import annotations

import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.weg2 import dual_handback_defer as HB
from sglang.srt.weg2 import front as F
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

DUAL_D = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D"}
REARM = "SGLANG_WEG2_DUAL_HANDBACK_DEFER_REARM"


def _req(rid="weg2-0-5"):
    return types.SimpleNamespace(rid=rid, output_ids=[], kv_arrival_seq=1, _pp_store_presence_cache="neg",
                                 _weg2_store_match_cache="neg")


@pytest.fixture()
def dual_d(monkeypatch):
    for k, v in DUAL_D.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv(REARM, raising=False)
    HB._REARM_LAST_LOG[0] = None


@pytest.mark.parametrize("tail,cls", [(0, "whole_short"), (70, "whole_short"), (255, "whole_short"),
                                      (256, "anchor_tail"), (3169, "anchor_tail"), (4061, "anchor_tail"),
                                      (4095, "anchor_tail"), (4096, "long"), (31841, "long")])
def test_tail_class_boundaries(tail, cls):
    assert HB.tail_class(tail) == cls


@pytest.mark.parametrize("fp,de,cls", [(1, 70, "short_credit"), (1, 72, "short_credit"), (0, 5, "short_credit"),
                                       (405, 405, "anchor_tail"), (31841, 31841, "long"), (31841, 3169, "anchor_tail"),
                                       (31841, 4096, "long"), (31841, 0, "long"),
                                       (None, 70, "?"), ("?", 70, "?"), (5, None, "?")])
def test_reroute_class(fp, de, cls):
    assert F._reroute_class(fp, de) == cls


def test_refused_line_carries_rearm_and_class(dual_d, monkeypatch, caplog):
    monkeypatch.setenv(REARM, "1")
    t = [0.0]
    req = _req()
    with caplog.at_level(logging.WARNING):
        assert HB.begin(req, 70, now=lambda: t[0]) is True           # mark
        req._weg2_hb_defer["issued"] = True                          # a read was issued
        assert HB.begin(req, 70, now=lambda: t[0]) is True           # re-arm 1/1
        req._weg2_hb_defer["issued"] = True
        assert HB.begin(req, 70, now=lambda: t[0]) is False          # limit reached: spent
    line = [r.getMessage() for r in caplog.records if "state=refused" in r.getMessage()]
    assert len(line) == 1
    assert "rearm=1/1" in line[0] and "tail_class=whole_short" in line[0] and "tail=70" in line[0]


def test_label_never_changes_the_verdict(dual_d, monkeypatch):
    """The verdict sequence is identical for a short and a long tail (the label is not a gate)."""
    monkeypatch.setenv(REARM, "2")
    seqs = []
    for tail in (70, 3169, 31841):
        req = _req("weg2-0-%d" % tail)
        out = [HB.begin(req, tail, now=lambda: 0.0)]
        for _ in range(3):
            req._weg2_hb_defer["issued"] = True
            out.append(HB.begin(req, tail, now=lambda: 0.0))
        seqs.append(out)
    assert seqs[0] == seqs[1] == seqs[2] == [True, True, True, False]


def test_default_off_is_single_shot_with_label(dual_d, caplog):
    req = _req()
    with caplog.at_level(logging.WARNING):
        assert HB.begin(req, 4061, now=lambda: 0.0) is True
        req._weg2_hb_defer["issued"] = True
        assert HB.begin(req, 4061, now=lambda: 0.0) is False
    line = [r.getMessage() for r in caplog.records if "state=refused" in r.getMessage()]
    assert line and "rearm=0/0" in line[0] and "tail_class=anchor_tail" in line[0]
