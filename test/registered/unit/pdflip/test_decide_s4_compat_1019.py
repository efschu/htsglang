"""EZ-S2a section-4 compatibility (plan S2a (b), SOLL section 4): ``set_verdict`` with the section-4 arguments only and
``SetMember(rid, uncached, t_arrive, p_only)`` answers BYTE-IDENTICALLY before and after S2a (the golden was made by
``gen_decide_s4_golden_1019.py`` on the S1 tree 6bba3b5901); ``classify_arrival`` too; ``d_granted_unstarted`` = 0 is the
default and changes nothing; ``decide`` on section-4 members equals ``set_verdict`` when no member is kurz."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flliper.srt.pdflip import phase_policy as pp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(os.path.join(HERE, "decide_s4_compat_golden_1019.json")))


def _sv(row, **extra):
    members = [pp.SetMember(r, u, t, p) for r, u, t, p in row["members"]]
    return members, pp.set_verdict(members, **row["kw"], **extra)


def test_set_verdict_byte_identical_to_the_s1_golden():
    bad = []
    for i, row in enumerate(G["set_verdict"]):
        _m, sv = _sv(row)
        if [sv.verdict, sv.why, round(sv.left_s, 6), sv.s_tokens, sv.half_x] != row["out"]:
            bad.append(i)
    assert not bad and len(G["set_verdict"]) == 600, bad[:10]


def test_d_granted_unstarted_zero_is_the_default_and_changes_nothing():
    for row in G["set_verdict"]:
        _m, a = _sv(row)
        _m, b = _sv(row, d_granted_unstarted=0)
        assert a == b


def test_classify_arrival_byte_identical():
    for u, x, po, want in G["classify"]:
        assert pp.classify_arrival(u, x, 0.5, po) == want


def test_decide_equals_set_verdict_when_no_member_is_kurz():
    n = 0
    for row in G["set_verdict"]:
        members, sv = _sv(row)
        xc = row["kw"]["x_tokens"]
        if any(pp.classify_arrival(m.uncached, xc, 0.5, m.p_only) == "kurz" for m in members):
            continue
        d = pp.decide("set", members=members, **row["kw"])
        assert d.verdict == sv
        n += 1
    assert n > 40, "the grid must contain enough all-halb/big sets"
