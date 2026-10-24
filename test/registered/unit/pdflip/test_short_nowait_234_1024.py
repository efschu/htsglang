"""SHORT-NOWAIT-1024 (#234): a SHORT kurz on its own figure goes to D at once while a set collects (SOLL R2; R1a, R6k, R8, R11).

SOLL v2 R2 verbatim: "u < halbX: sofort auf D prefillen, keine Wartezeit, auch während ein Set sammelt. Nie ein Flip für einen
kurzen Request." Before #234 a SHORT that sat in ``st["shorts"]`` with a KURZ own figure -- a halb arrival the S63 re-probe moved
under halbX, or a sibling rider whose re-probed rest is kurz -- was left out of the set (``set_keep``) but released only with the
set's verdict: it waited on its seat path up to T_halb while REAL halb/gross members collected (named open in
done/sibling-u-1024.md section 10, "Offen 2"; fixed there only for the riders-only case).

Exception (planner decision 24.10., variant (a)): a RIDER -- kurz only on its R1u rest behind an anchor not computed yet, its own
figure not kurz -- is no R2 case; it keeps waiting behind its anchor (a kurz SHORT anchor released here: the MUSS-1 hold).

Danger directions and their mutants (each must turn at least one test here red):
  (1) "the kurz SHORT waits again" -> R2 broken (repro);
  (2) "the rider is released at once too" -> its own figure goes to D uncounted behind an anchor not computed (R1/R7)."""
from __future__ import annotations

import asyncio
import logging
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_set_scheduler_1017 import _fx, _on, _verdicts  # noqa: E402
from test_pdflip_arrival_seat_rule_0929 import _pending  # noqa: E402
from test_review_sibling_u_1024 import A10, B10, _anchor_done, _pair_s10, _reprobe_stub, _waits  # noqa: E402

from flliper.srt.pdflip import front as front_mod  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402

H, K = "pdflip-234-h", "pdflip-234-k"
X6 = 6185   # halbX 3092 at F_halb 0.5


def _kurz_now(caplog):
    return [r.getMessage() for r in caplog.records if "PDFLIP SET-SHORT-KURZ-NOW" in r.getMessage()]


def _halb_and_kurz(monkeypatch, t0, running=("dec",)):
    """A halb SHORT H (4000, collects: S < X) and a SHORT K whose exact u is kurz (1500, e.g. after its S63 re-probe)."""
    _on(monkeypatch)
    f = _fx(running=running)
    f.tp_prefill_max_tokens = X6
    st = f._set_st()
    st["shorts"][H] = (t0, 4000, X6)
    st["shorts"][K] = (t0 + 0.1, 1500, X6)
    return f, st


def test_r2_repro_kurz_short_goes_to_d_while_a_halb_member_collects(monkeypatch, caplog):
    """The repro (D decodes): H opens a set (R3, S=4000 < X, T_halb runs). K is kurz on its own figure: R2 -> D in THIS tick,
    not with H's verdict (before #234: K stayed in ``shorts`` until T_halb -- red here)."""
    t0 = time.time()
    f, st = _halb_and_kurz(monkeypatch, t0)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 0.3) == "hold"                                     # H collects (R3)
    assert st["released"].get(K) == "D", st["released"]                                  # R2: no wait
    assert K not in st["shorts"] and H in st["shorts"] and H not in st["released"]
    (line,) = _kurz_now(caplog)
    assert "rid=%s u=1500 X=6185 halfX=3092 site=collect" % K in line, line
    assert f.counters["set_short_kurz_now"] == 1
    assert f.__dict__["_r6k_class"][K][0] == pp.CLASS_KURZ                                # R6k: its D prefill stays on D
    assert f._d_pf_book().is_open(rid=K)                                                 # X-SUM-PRICE: booked
    col = [v for v in _verdicts(caplog) if "verdict=COLLECT" in v][-1]
    assert "S=4000 " in col and "kurz-anchor" not in col, col
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 0.5) == "hold"                                     # no second release, H still collects
    assert f.counters["set_short_kurz_now"] == 1
    st["released"].pop(K, None)                                                          # its seat path took the release
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 4.7) == "D"                                        # H: T_halb over, S < X (R3)
    assert st["released"].get(H) == "D" and "verdict=D_AFTER_TIMER" in _verdicts(caplog)[-1]


def test_r2_repro_seat_path_after_the_reprobe_takes_d_at_once_and_logs_d_now_on_the_kurz_number(monkeypatch, caplog):
    """End to end on the seat path (``_acquire_short_seat``): K arrives halb (3000 at F_halb 0.4, halbX 2474) and joins H's set
    (S=5500 < X: COLLECT). The S63 re-probe moves K to 1000 (kurz): K gets its D seat at once (before #234: only at T_halb =
    4.5 s -- the 2 s bound here is red), and its line is ``D_NOW`` on the kurz number."""
    _on(monkeypatch, frac="0.4")
    f = _fx()
    f.tp_prefill_max_tokens = X6
    st = f._set_st()
    st["shorts"][H] = (time.time(), 2500, X6)

    async def run():
        task = asyncio.ensure_future(f._acquire_short_seat(K, 3000, uncached=3000, x_set=X6))
        for _ in range(50):
            await asyncio.sleep(0.02)
            if K in st["shorts"]:
                break
        assert K in st["shorts"] and K not in st["released"]
        tup = st["shorts"][K]
        st["shorts"][K] = (tup[0], 1000) + tuple(tup[2:])                                # what _set_reprobe_apply writes
        t1 = time.time()
        seat = await asyncio.wait_for(task, 2.0)
        return seat, time.time() - t1

    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        seat, dt = asyncio.run(run())
    assert seat is not None and dt < 1.0, dt
    assert H in st["shorts"] and H not in st["released"]                                 # the set still collects
    (now_line,) = [m for m in _verdicts(caplog) if "classes=kurz" in m and "verdict=D_NOW" in m]
    assert "members=%s:1000" % K in now_line and "why=R2" in now_line, now_line
    assert any("rid=%s u=1000" % K in m and "site=collect" in m for m in _kurz_now(caplog)), _kurz_now(caplog)


def test_variant_a_rider_behind_the_kurz_anchor_keeps_waiting_while_a_halb_member_collects(monkeypatch, caplog):
    """Planner (a): H (halb 4000, own session) collects; A10 (kurz 1500) is a kurz SHORT and B10's anchor; B10 (own 4000, rest
    2500 kurz behind A10) is a RIDER, not an R2 case. A10 goes to D at once (R2); B10 is NOT released -- it waits behind its D
    anchor (MUSS-1 hold), no member, no S share. Once A10 is computed, the re-probe reads B10's rest (2500, kurz on its own
    figure now) and B10 goes to D at once too, while H still collects."""
    t0 = time.time()
    f, st = _pair_s10(monkeypatch, t0)
    st["shorts"][H] = (t0 - 0.1, 4000, X6)
    calls = _reprobe_stub(monkeypatch, apply=lambda r: st["shorts"].__setitem__(r, (t0 + 0.2, 2500, X6)))
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 0.3) == "hold"
        assert st["released"].get(A10) == "D" and A10 not in st["shorts"]                # R2: the kurz anchor at once
        assert B10 not in st["released"], st["released"]                                 # (a): not in the anchor's tick
        assert st["rider_hold"][B10]["anchor"] == A10 and B10 in st["shorts"]
        assert H in st["shorts"] and H not in st["released"]
        (begin,) = _waits(caplog)
        assert "begin rid=%s anchor=%s own=4000 rest=2500 where=short" % (B10, A10) in begin, begin
        assert f._decode_collect(t0 + 0.5) == "hold"                                     # anchor still on D: B10 waits
        assert B10 not in st["released"] and B10 in st["rider_hold"] and not calls
        col = [v for v in _verdicts(caplog) if "verdict=COLLECT" in v][-1]
        assert "members=%s:4000" % H in col and B10 not in col and "S=4000 " in col, col
        _anchor_done(f, st, A10)
        assert f._decode_collect(t0 + 0.9) == "hold"                                     # H still collects
    assert calls == [((B10,), "sibling-rider", "RIDER_RELEASE")], calls
    assert st["released"].get(B10) == "D" and B10 not in st.get("rider_hold", {}) and B10 not in st["shorts"]
    assert [m for m in _kurz_now(caplog) if "rid=%s u=2500" % B10 in m], _kurz_now(caplog)
    assert f.counters["set_short_kurz_now"] == 2 and f.counters["sibling_rider_wait_release_rest"] == 1


def test_variant_a_rider_behind_a_collecting_member_anchor_stays_a_rider_of_the_set(monkeypatch, caplog):
    """A rider behind a MEMBER anchor (the 15:36 pair: 912 halb, 913 own 5534 / rest 1502) is no R2 case either: nothing is
    released while the set collects -- 913's own figure is not kurz."""
    t0 = time.time()
    from test_review_sibling_u_1024 import A912, R913, _pair_1536
    f, st = _pair_1536(monkeypatch, t0)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 1.6) == "hold"
    assert not st["released"] and A912 in st["shorts"] and R913 in st["shorts"] and not _kurz_now(caplog)
    assert "sibling-rider=%s:1502" % R913 in _verdicts(caplog)[-1]


def test_r8_no_decode_decides_at_once_both_to_d(monkeypatch, caplog):
    """R8 (nothing decodes on D): the set is decided at once -- S=4000 < X -> D; H and K both go to D in the same tick by the D
    verdict (no collect, nothing waits; the #234 release is not needed)."""
    t0 = time.time()
    f, st = _halb_and_kurz(monkeypatch, t0, running=())
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 0.3) == "D"
    assert st["released"].get(H) == "D" and st["released"].get(K) == "D" and not st["shorts"]
    assert "verdict=D_AFTER_TIMER" in _verdicts(caplog)[-1] and "R8" in _verdicts(caplog)[-1]


def test_r8_r1a_no_decode_big_flips_and_the_kurz_short_rides_it(monkeypatch, caplog):
    """R8 + R1a: nothing decodes, a big request is queued -> FLIP_P at once; the waiting kurz SHORT rides the decided flip."""
    t0 = time.time()
    f, st = _halb_and_kurz(monkeypatch, t0, running=())
    del st["shorts"][H]
    f.queue = [_pending("pdflip-234-g", 9000, t0 + 0.05, est_prompt=50000)]
    _reprobe_stub(monkeypatch)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 0.3) == "P"
    assert st["released"].get(K) == "P" and "pdflip-234-g" in st["go_p"] and not _kurz_now(caplog)


def test_r1a_a_collect_that_holds_a_decided_flip_is_ridden_not_left(monkeypatch, caplog):
    """R4OVER-1024 + R1a: a COLLECT that is the R2 hold of a DECIDED flip is no collecting set -- the kurz SHORT rides that flip,
    it is not released to D (a fresh kurz arrival yields to it the same way, ``_set_flip_foreseeable``)."""
    t0 = time.time()
    f, st = _halb_and_kurz(monkeypatch, t0)
    monkeypatch.setattr(front_mod.Front, "_r2_flip_held", lambda self: True)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 0.3) == "hold"
    assert K in st["shorts"] and K not in st["released"] and not _kurz_now(caplog)


def test_r2_a_flip_waiting_for_its_reprobe_is_not_decided_the_kurz_short_goes(monkeypatch, caplog):
    """S63 F1 + R2: a flip verdict (big + halb company, R4) waiting for D's current numbers is not decided -- the kurz SHORT goes
    to D now (site=reprobe-hold), the members wait for the re-probe."""
    t0 = time.time()
    f, st = _halb_and_kurz(monkeypatch, t0)
    f.queue = [_pending("pdflip-234-g", 9000, t0 + 0.05, est_prompt=50000)]
    monkeypatch.setattr(front_mod.Front, "_set_reprobe_gate", lambda self, rids, site, verdict: False)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 0.3) == "hold"
    assert st["released"].get(K) == "D" and H in st["shorts"] and "pdflip-234-g" not in st["go_p"]
    assert any("site=reprobe-hold" in m for m in _kurz_now(caplog)), _kurz_now(caplog)
