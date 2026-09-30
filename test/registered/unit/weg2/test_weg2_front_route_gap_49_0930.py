"""#49 measurement tool (weg2/tools/front_route_gap.py): one synthetic front log, one LONG turn per class.

Pins the classes the #49 plan counts on the real agent trace (w109290020: 96 over P, 34 needed):
OK_LONG, K1_FIXES, INFLIGHT(after_p), DDIRECT_GAP_served, OTHER -- and that SHORT turns are not counted.
"""

from sglang.srt.weg2.tools import front_route_gap as G

T = "[2026-09-29 00:%02d:%02d,000] INFO weg2.front: "


def _rv(m, s, rid, verdict, unc, span=0, src="none", X=4096):
    return (T % (m, s) + "WEG2 ROUTE-VERDICT rid=%s verdict=%s uncached=%d (base for X=%d, what D must "
            "PREFILL) presence_span=%d presence_src=%s" % (rid, verdict, unc, X, span, src))


def _xp(m, s, rid, tokens, credit, reused):
    return (T % (m, s) + "WEG2 X-EXACT-PRICE rid=%s pending=1 tokens=%d credit=%d src=d_leg2_cached known=1 "
            "(x) reused=%d encoded=1" % (rid, tokens, credit, reused))


def _p1(m, s, rid, pt, ct):
    return T % (m, s) + "WEG2-SERVED group=P leg=1 rid=%s prompt_tokens=%d cached_tokens=%d wall=1s" % (rid, pt, ct)


def _d2(m, s, rid, pt, ct, epoch=3):
    return (T % (m, s) + "WEG2-SERVED group=D leg=2 rid=%s stream=1 status=200 prompt_tokens=%d cached_tokens=%d "
            "completion_tokens=5 uncached=0 verdict=serve priced=True wall=1s epoch=%d" % (rid, pt, ct, epoch))


def _fc(m, s, rid, via, epoch=3):
    return T % (m, s) + "WEG2 LEG2-FIRST-CONTENT rid=%s epoch=%d via=%s leg2_ms=1" % (rid, epoch, via)


LOG = [
    # a source P prefilled (after_p), first content at 01:00
    _xp(0, 50, "r-src", 30000, 0, 0), _rv(0, 50, "r-src", "long", 30000),
    _p1(0, 58, "r-src", 30000, 0), _fc(1, 0, "r-src", "after_p"), _d2(1, 5, "r-src", 30000, 29998),
    # K1: continuation priced AFTER the source's first content, P read all but 300
    _xp(1, 10, "r-k1", 30300, 20000, 30000), _rv(1, 10, "r-k1", "long", 10300, 20000, "d_leg2_cached"),
    _p1(1, 12, "r-k1", 30300, 30000),
    # a source still in flight: priced before its first content
    _xp(2, 0, "r-src2", 40000, 0, 0), _rv(2, 0, "r-src2", "long", 40000), _p1(2, 5, "r-src2", 40000, 0),
    _xp(2, 3, "r-in", 40100, 0, 40000), _rv(2, 3, "r-in", "long", 40100), _p1(2, 9, "r-in", 40100, 39999),
    _fc(2, 7, "r-src2", "after_p"),
    # D-direct source served (finished) before the next turn, credited only its arrival reading
    _xp(3, 0, "r-dd", 22000, 20000, 0), _rv(3, 0, "r-dd", "short", 2000, 20000, "d_leg2_cached"),
    _fc(3, 1, "r-dd", "d_direct"), _d2(3, 4, "r-dd", 22000, 20000),
    _xp(3, 30, "r-ddn", 22500, 20000, 22000), _rv(3, 30, "r-ddn", "long", 4500, 20000, "d_leg2_cached"),
    _p1(3, 33, "r-ddn", 22500, 21999),
    # OK: real rest over X
    _xp(4, 0, "r-ok", 50000, 0, 0), _rv(4, 0, "r-ok", "long", 50000), _p1(4, 5, "r-ok", 50000, 30000),
    # OTHER: store held it, no earlier whole prompt as prefix (repeated document diverging inside)
    _xp(5, 0, "r-oth", 40767, 0, 47), _rv(5, 0, "r-oth", "long", 40767), _p1(5, 5, "r-oth", 40767, 36863),
]


def test_each_class_once_and_short_not_counted():
    routed, long_, cls, ex = G.classify(LOG)
    assert routed == 8 and long_ == 7
    assert cls["K1_FIXES"] == 1 and ex["K1_FIXES"][0][:2] == ("r-k1", "r-src")
    assert cls["INFLIGHT(after_p)"] == 1
    assert cls["DDIRECT_GAP_served"] == 1 and ex["DDIRECT_GAP_served"][0][:2] == ("r-ddn", "r-dd")
    # r-src and r-src2 (no store hit) and r-ok (rest 20000) needed P
    assert cls["OK_LONG"] == 3
    assert cls["OTHER"] == 1 and ex["OTHER"] == ["r-oth"]


def test_main_prints_the_three_headline_numbers(tmp_path, capsys):
    p = tmp_path / "f.front.log"
    p.write_text("\n".join(LOG) + "\n")
    G.main([str(p)])
    out = capsys.readouterr().out
    assert "routed turns (long+short): 8  over P (long): 7  needed P (OK_LONG): 3" in out
