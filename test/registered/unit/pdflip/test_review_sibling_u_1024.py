"""REVIEW SIBLING-U-1024 (#229): danger probes at the front edge -- what happens to a sibling RIDER on a D verdict.

Each probe asserts the behaviour it OBSERVES on the head (green = the danger is real), so the run documents it.
P1 (SOLL R1/R2/R7, Nutzergesetz "D prefillt nie ueber X"): a rider that is a halb SHORT (own u halb, rest kurz) is released to D
   in the SAME tick as its anchor at D_AFTER_TIMER, while S counted it with 0 -- on D, before its anchor is computed, its D
   figure is its own u (the author's own reasoning in phase_policy.sibling_riders).
P2 (Risk 2 under R8): a queue-entry rider (own u > X) left behind by its halb anchor's D verdict is classed on its own figure
   the next tick and, with no decode on D, flips ALONE at once -- no T_halb, no S63 chance to see the anchor finished."""
from __future__ import annotations

import logging
import os
import time

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_set_scheduler_1017 import _fx, _on, _verdicts  # noqa: E402
from test_pdflip_arrival_seat_rule_0929 import _pending  # noqa: E402


def _session(f, rid, sess, ids, depth):
    f.__dict__.setdefault("_sess_by_rid", {})[rid] = sess
    f.__dict__.setdefault("_set_ids", {})[rid] = np.asarray(ids, dtype=np.int32)
    f.__dict__.setdefault("_set_depth", {})[rid] = depth


def test_probe_p1_halb_short_rider_goes_to_d_with_its_anchor_uncounted(monkeypatch, caplog):
    """15:36 pair (X 6185, halbX 3092): 912 halb 5396 + 913 own 5534 halb / rest 1502 kurz. Base c07b574410: S=6898 >= X ->
    R3 FLIP_P at t0+1.5. Head: S=5396 -> COLLECT, at T_halb D_AFTER_TIMER: BOTH released to D in one tick. D's load is the
    union 6898 at best (own 10930 without in-batch dedupe) > X, while the verdict's S was 5396."""
    _on(monkeypatch)
    f = _fx()
    f.tp_prefill_max_tokens = 6185
    _session(f, "pdflip-80-912", "a4c8b097ac", list(range(25428)), 20032)
    _session(f, "pdflip-80-913", "a4c8b097ac", list(range(24064)) + [-i - 1 for i in range(25566 - 24064)], 20032)
    t0 = time.time()
    st = f._set_st()
    st["shorts"]["pdflip-80-912"] = (t0, 5396, 6185)
    st["shorts"]["pdflip-80-913"] = (t0 + 1.5, 5534, 6185)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 1.5) == "hold"
        route = f._decode_collect(t0 + 4.7)
    lines = _verdicts(caplog)
    assert route == "D", (route, lines)
    assert st["released"].get("pdflip-80-912") == "D" and st["released"].get("pdflip-80-913") == "D", (st["released"], lines)
    last = lines[-1]
    assert "verdict=D_AFTER_TIMER" in last and "S=5396 " in last and "sibling-rider=pdflip-80-913:1502" in last, last
    assert 5396 + 1502 > 6185   # what D must compute at the very least (R1u union) is over X


def test_probe_p2_queue_rider_left_by_a_d_verdict_flips_alone_under_r8(monkeypatch, caplog):
    """10:33:42 shape (X 4096): anchor halb SHORT 3950, rider = queue entry own 4108 / rest 162, nothing decodes on D (R8).
    Tick 1: S=3950 < X -> D_AFTER_TIMER, the anchor goes to D. Tick 2: the rider is alone, classed on its own 4108 (gross) ->
    R8 FLIP_P at once (no T_halb). Base: the pair flipped together at tick 1 (S=4112 >= X)."""
    _on(monkeypatch)
    f = _fx(running=())
    f.tp_prefill_max_tokens = 4096
    ids_a = list(range(60000 + 3950))
    ids_r = list(range(60000 + 3950 - 4)) + [-i - 1 for i in range(166)]
    _session(f, "pdflip-40-179", "s1033", ids_a, 60000)
    _session(f, "pdflip-40-180", "s1033", ids_r, 60000)
    t0 = time.time()
    st = f._set_st()
    st["shorts"]["pdflip-40-179"] = (t0 - 0.6, 3950, 4096)
    f.queue = [_pending("pdflip-40-180", 4108, t0 - 0.1, est_prompt=60000 + 4108)]
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        r1 = f._decode_collect(t0)
        lines1 = list(_verdicts(caplog))
        r2 = f._decode_collect(t0 + 0.05)
    lines = _verdicts(caplog)
    assert r1 == "D", (r1, lines1)
    assert "sibling-rider=pdflip-40-180" in lines1[-1] and "S=3950 " in lines1[-1], lines1
    assert st["released"].get("pdflip-40-179") == "D"
    assert r2 == "P", (r2, lines)
    assert "pdflip-40-180" in st["go_p"] and "pdflip-40-179" not in st["go_p"], (st["go_p"], lines)
    assert "classes=gross" in lines[-1] and "R8" in lines[-1], lines[-1]
