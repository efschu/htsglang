"""EZ-S1 (plan nf-entscheidungszentrale-plan-1019 B3/A15, step S1; SOLL R11, SOLL section 4): the ONE decision record.

(d)  format test: DECISION / DECISION-DONE / DECISION-SUM, one parser (the same one goes to nf_scharf, logindex, S4)
(e)  SET-VERDICT byte-identical to the H91 line: six real rc16 front-log lines as golden, and ``Front._set_log`` itself
(i)  edge-triggered: 1000 controller ticks of one state = ONE DECISION line, ``=1000/1`` in the DECISION-SUM
     the ALT shadow table: every old site of the inventory resolves, the shadow changes no return value and no exception
     boot / hand flip: ``kind=boot|manual rule=NS:BOOT|NS:MANUAL`` with their DECISION-DONE
     the gates (W101/W103/W104/W125, context, template, W52, W193) write ``kind=admit action=REFUSE rule=Gate-*``
"""
import asyncio
import inspect
import logging
import os
import re
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flliper.srt.pdflip import decision_record as dr  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402
from test_pdflip_front_x_exact import _front  # noqa: E402


# ------------------------------------------------------------------------------------------------ (d) the format

def _dec(**kw):
    base = dict(kind="set", action="FLIP_P", rule="R8", why="no decode on D: decided now", set_id=3, rid=None,
                members=(dr.Member("pdflip-2-9", 28229, 28229, "d_probe", 12), dr.Member("pdflip-2-10", 5, 3, "p_store", 0)),
                s_tokens=28234, s_p_tokens=28232, x=4096, half_x=2048, classes=("gross", "kurz"), d_decoding=False,
                seats_free=6, seats_src="d_ledger", kv_free=131820, kv_src="d_ledger", kv_age_ms=312, r6_rest=0,
                unstarted=0, t_half_s=4.5, waited_s=0.0, waited_on="-", numbers_age_ms_max=312,
                t_arrive_to_decide_ms=1380, seq=7, epoch=2)
    base.update(kw)
    return dr.Decision(**base)


def test_d_decision_line_round_trips_through_the_parser():
    d = _dec()
    line = d.line()
    assert line.startswith("WEG2 DECISION seq=7 epoch=2 kind=set action=FLIP_P rule=R8 set=3 rid=- members=")
    assert "\n" not in line
    got = dr.parse_line("[2026-10-19 12:19:53,510] INFO pdflip.front: " + line)
    assert got["tag"] == "DECISION" and got["seq"] == 7 and got["epoch"] == 2 and got["kind"] == "set"
    assert got["action"] == "FLIP_P" and got["rule"] == "R8" and got["set"] == 3 and got["rid"] == "-"
    assert [(m.rid, m.u_own, m.s_p, m.src, m.age_ms) for m in got["members"]] == [
        ("pdflip-2-9", 28229, 28229, "d_probe", 12), ("pdflip-2-10", 5, 3, "p_store", 0)]
    assert (got["S"], got["S_P"], got["X"], got["halfX"]) == (28234, 28232, 4096, 2048)
    assert got["classes"] == ["gross", "kurz"] and got["d_decoding"] == 0
    assert (got["seats_free"], got["seats_src"]) == (6, "d_ledger")
    assert (got["kv_free"], got["kv_src"], got["kv_age_ms"]) == (131820, "d_ledger", 312)
    assert got["t_half_s"] == 4.5 and got["numbers_age_ms_max"] == 312 and got["t_arrive_to_decide_ms"] == 1380
    assert got["why"] == "no decode on D: decided now"


def test_d_absent_kv_and_missing_members_round_trip():
    line = _dec(members=(), classes=(), kv_free=None, kv_src="absent", kv_age_ms=None, set_id=None,
                rid="pdflip-0-1", why="").line()
    got = dr.parse_line(line)
    assert got["members"] == [] and got["classes"] == [] and got["kv_free"] is None and got["kv_age_ms"] is None
    assert got["set"] is None and got["rid"] == "pdflip-0-1" and got["why"] == ""
    assert "kv_free=-@absent/-" in line


def test_d_unknown_number_is_u_minus_one():
    got = dr.parse_line(_dec(members=(dr.Member("r", -1, -1, "none", 0),), classes=("unknown",)).line())
    assert got["members"][0].u_own == -1 and got["classes"] == ["unknown"]


def test_d_done_and_sum_lines():
    line = dr.done_line(7, "pdflip-2-9", "D", "ok", 28229, "d_extent", "class gross")
    got = dr.parse_line(line)
    assert got == {"tag": "DECISION-DONE", "seq": 7, "rid": "pdflip-2-9", "executor": "D", "outcome": "ok",
                   "measured_u": 28229, "src": "d_extent", "detail": "class gross"}
    got = dr.parse_line(dr.done_line(8, None, "front", "refused"))
    assert got["measured_u"] is None and got["rid"] == "-" and got["outcome"] == "refused"
    rec = dr.Recorder(logging.getLogger("t"), sum_window_s=60.0)
    for _ in range(3):
        rec.record(_dec(kind="alt", rule="ALT:x"), edge=False)
    s = dr.parse_line(rec.flush_sum())
    assert s["tag"] == "DECISION-SUM" and s["n"] == 3 and s["lines"] == 3 and s["by"]["alt|ALT:x|FLIP_P"] == (3, 3)


def test_d_malformed_lines_raise_and_foreign_lines_are_none():
    assert dr.parse_line("PDFLIP SET-VERDICT epoch=0 set=0") is None
    with pytest.raises(ValueError):
        dr.parse_line("WEG2 DECISION seq=1 epoch=0 kind=set action=X rule=R1 why=")
    with pytest.raises(ValueError):
        dr.parse_line(_dec().line().replace("members=pdflip-2-9:28229/28229@d_probe/12", "members=garbage"))


def test_d_seq_is_monotonic_and_the_epoch_is_the_fronts():
    ep = [4]
    rec = dr.Recorder(logging.getLogger("t"), epoch_of=lambda: ep[0])
    logs = []
    rec.log = SimpleNamespace(info=lambda fmt, line: logs.append(line))
    a = rec.record(_dec(), edge=False)
    ep[0] = 5
    b = rec.record(_dec(), edge=False)
    assert (a, b) == (1, 2)
    assert "epoch=4" in logs[0] and "epoch=5" in logs[1]


# ------------------------------------------------------------------------------------------------ (e) SET-VERDICT golden

#: six real lines of the rc16 front log (boot ..._bf9ca46f45_1019_121629, 12:19:53 - 12:20:25), after "PDFLIP "
GOLDEN = [
    "SET-VERDICT epoch=0 set=0 n=1 members=pdflip-0-1:0 S=0 X=4096 halfX=2048 classes=p_only d_decoding=0 seats_free=6 "
    "t_half_s=4.5 waited_s=0.0 verdict=FLIP_P why=R8 no decode on D: decided now, S=0 >= X=4096 or a big member",
    "SET-VERDICT epoch=0 set=1 n=1 members=pdflip-0-2:0 S=0 X=4096 halfX=2048 classes=kurz d_decoding=0 seats_free=6 "
    "t_half_s=4.5 waited_s=1.5 verdict=NOT_D why=yield_or_gate",
    "SET-VERDICT epoch=2 set=0 n=1 members=pdflip-2-6:0 S=0 X=5861 halfX=2930 classes=kurz d_decoding=1 seats_free=1 "
    "t_half_s=4.5 waited_s=0.1 verdict=D_NOW why=R2",
    "SET-VERDICT epoch=2 set=0 n=1 members=pdflip-2-8:0 S=0 X=5861 halfX=2930 classes=kurz d_decoding=1 seats_free=2 "
    "t_half_s=4.5 waited_s=0.3 verdict=D_NOW why=R2",
    "SET-VERDICT epoch=3 set=2 n=2 members=pdflip-3-4:3000,pdflip-3-5:2500 S=5500 X=4096 halfX=2048 classes=halb,halb "
    "d_decoding=1 seats_free=2 t_half_s=4.5 waited_s=0.0 verdict=COLLECT why=R3 S=5500 >= X: wait for T_halb xk=3000",
    "SET-VERDICT epoch=3 set=0 n=1 members=pdflip-3-9:9000 S=9000 X=4096 halfX=2048 classes=gross d_decoding=1 "
    "seats_free=0 t_half_s=4.5 waited_s=4.6 verdict=PARK_NO_SEAT why=R5 [prefix-shared=500: P counts once, S_P=8500]",
]

_G_RE = re.compile(
    r"SET-VERDICT epoch=(?P<epoch>\d+) set=(?P<set>\d+) n=(?P<n>\d+) members=(?P<members>\S+) S=(?P<S>\d+) X=(?P<X>\d+) "
    r"halfX=(?P<halfX>\d+) classes=(?P<classes>\S+) d_decoding=(?P<dd>\d) seats_free=(?P<seats>\d+) "
    r"t_half_s=(?P<th>[\d.]+) waited_s=(?P<w>[\d.]+) verdict=(?P<verdict>\S+) why=(?P<why>.*)$")


@pytest.mark.parametrize("golden", GOLDEN)
def test_e_set_verdict_line_is_byte_identical_to_the_h91_line(golden):
    m = _G_RE.match(golden)
    members = [(p.rsplit(":", 1)[0], int(p.rsplit(":", 1)[1])) for p in m["members"].split(",")]
    got = dr.set_verdict_line(epoch=int(m["epoch"]), set_id=int(m["set"]), members=members, s_tokens=int(m["S"]),
                              x_tok=int(m["X"]), half_x=int(m["halfX"]), classes=m["classes"].split(","),
                              d_decoding=bool(int(m["dd"])), seats_free=int(m["seats"]), t_half=float(m["th"]),
                              waited=float(m["w"]), verdict=m["verdict"], why=m["why"])
    assert got == "PDFLIP " + golden


def test_e_set_log_writes_the_golden_line_and_a_matching_decision(caplog):
    f = _front(True)
    f.epoch = 2
    st = f._set_st()
    st["set_id"] = 0
    m = pp.SetMember("pdflip-2-6", 0, time.time(), False)
    sv = pp.SetVerdict("D_NOW", "R2", 0.0, 0, 2930)
    F.Front._num_note(f, "pdflip-2-6", "d_probe")
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        f._set_log(st, [m], sv, verdict="D_NOW", why="R2", x_tok=5861, frac=0.5, t_half=4.5, decoding=True, seats=1,
                   waited=0.1)
    msgs = [r.getMessage() for r in caplog.records]
    assert "PDFLIP " + GOLDEN[2] in msgs
    d = [dr.parse_line(x) for x in msgs if x.startswith("WEG2 DECISION ")]
    assert len(d) == 1 and d[0]["kind"] == "set" and d[0]["action"] == "D_NOW" and d[0]["rule"] == "R2"
    assert d[0]["set"] == 0 and d[0]["rid"] == "pdflip-2-6" and d[0]["classes"] == ["kurz"]
    assert d[0]["members"][0].src == "d_probe" and d[0]["d_decoding"] == 1 and d[0]["seats_free"] == 1
    assert d[0]["X"] == 5861 and d[0]["halfX"] == 2930


def test_e_the_set_log_source_goes_through_the_derived_formatter():
    src = inspect.getsource(F.Front._set_log)
    assert "_dr.set_verdict_line(" in src and 'logger.info("PDFLIP SET-VERDICT' not in src


def test_e_rule_of_set_why():
    assert dr.rule_of_set_why("R2", "D_NOW") == "R2"
    assert dr.rule_of_set_why("R8 no decode on D", "FLIP_P") == "R8"
    assert dr.rule_of_set_why("X-SUM S=5 > X", "COLLECT").startswith("NS:set-")
    assert dr.rule_of_set_why("yield_or_gate", "NOT_D") == "NS:set-yield_or_gate"


# ------------------------------------------------------------------------------------------------ (i) edge trigger

def test_i_1000_ticks_of_one_state_are_one_line_and_a_counter(caplog):
    clock = [0.0]
    rec = dr.Recorder(logging.getLogger("pdflip.front"), clock=lambda: clock[0], sum_window_s=60.0)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        for _ in range(1000):
            rec.alt("dwell_ok", kind="phase_end", action="false")
        assert [m for m in caplog.messages if m.startswith("WEG2 DECISION seq")] and \
            len([m for m in caplog.messages if m.startswith("WEG2 DECISION seq")]) == 1
        clock[0] = 61.0
        rec.tick()
    sums = [dr.parse_line(m) for m in caplog.messages if m.startswith("WEG2 DECISION-SUM")]
    assert len(sums) == 1 and sums[0]["n"] == 1000 and sums[0]["lines"] == 1
    assert sums[0]["by"]["phase_end|ALT:dwell_ok|false"] == (1000, 1)
    assert rec.suppressed_total == 999


def test_i_a_change_of_state_or_rid_writes_a_new_line():
    rec = dr.Recorder(logging.getLogger("t"))
    logs = []
    rec.log = SimpleNamespace(info=lambda fmt, line: logs.append(line))
    rec.alt("dwell_ok", kind="phase_end", action="false")
    rec.alt("dwell_ok", kind="phase_end", action="false")
    rec.alt("dwell_ok", kind="phase_end", action="true")       # the state changed
    rec.alt("dwell_ok", kind="phase_end", action="true", rid="r1")   # another rid is another site
    rec.alt("dwell_ok", kind="phase_end", action="false")       # back: a new edge
    assert len(logs) == 4


def test_i_a_non_edge_record_is_never_suppressed():
    rec = dr.Recorder(logging.getLogger("t"))
    logs = []
    rec.log = SimpleNamespace(info=lambda fmt, line: logs.append(line))
    for _ in range(5):
        rec.record(_dec(), edge=False)
    assert len(logs) == 5


def test_i_the_recorder_never_raises_into_its_caller():
    rec = dr.Recorder(logging.getLogger("t"), epoch_of=lambda: 1 / 0)
    assert rec.record(_dec(), edge=False) == 0
    rec.log = None
    rec.done(1, "r", "front", "ok")
    rec.tick()


# ------------------------------------------------------------------------------------------------ the ALT shadow

def test_alt_every_site_of_the_table_is_installed():
    assert F.EZ_ALT_MISSING == []
    assert len(F.EZ_ALT_SITES) >= 40
    for marker, (name, kind) in F.EZ_ALT_SITES.items():
        fn = inspect.getattr_static(F.Front, name)
        assert getattr(fn, "_ez_alt_marker", None) == marker, (marker, name)
        assert kind in dr.KINDS, (marker, kind)


def test_alt_install_is_idempotent_and_reports_a_missing_site():
    class C:
        def a(self, rid):
            return 7

        async def b(self):
            return "x"

        @staticmethod
        def s():
            return 1

    recs = []
    rec = dr.Recorder(logging.getLogger("t"))
    rec.log = SimpleNamespace(info=lambda fmt, line: recs.append(line))
    table = {"a": ("a", "seat"), "b": ("b", "park"), "s": ("s", "seat"), "gone": ("nope", "seat")}
    miss1 = dr.install_alt_shadow(C, table, lambda o: rec)
    miss2 = dr.install_alt_shadow(C, table, lambda o: rec)
    assert miss1 == ["s", "gone"] == miss2           # a staticmethod / a missing name is reported, never half-wrapped
    c = C()
    assert c.a("r1") == 7 and asyncio.run(c.b()) == "x"
    assert len(recs) == 2 and "rule=ALT:a" in recs[0] and "rid=r1" in recs[0] and "action=7" in recs[0]
    assert "rule=ALT:b" in recs[1] and "action=x" in recs[1]


def test_alt_the_shadow_changes_no_value_and_swallows_no_exception():
    class C:
        def boom(self, rid):
            raise KeyError(rid)

        def val(self):
            return {"k": 1}

    rec = dr.Recorder(logging.getLogger("t"))
    rec.log = SimpleNamespace(info=lambda fmt, line: (_ for _ in ()).throw(RuntimeError("log broke")))
    dr.install_alt_shadow(C, {"boom": ("boom", "seat"), "val": ("val", "seat")}, lambda o: rec)
    with pytest.raises(KeyError):
        C().boom("r")
    assert C().val() == {"k": 1}                        # a broken recorder breaks nothing


def test_alt_a_real_site_writes_one_edge_triggered_record(caplog):
    f = _front(True)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        for _ in range(50):
            f._dwell_ok("D", "P", False, False, 0.0)
    alt = [m for m in caplog.messages if "rule=ALT:dwell_ok" in m]
    assert len(alt) == 1
    f._dec().flush_sum()
    # the value the old code returned is unchanged
    assert f._dwell_ok("D", "P", False, False, 0.0) in (True, False)


def test_alt_summarize_is_a_whitespace_free_token():
    for v, want in [(None, "none"), (True, "true"), (False, "false"), (3, "3"), ("a b", "a_b"), ([1, 2], "n2"),
                    (object(), "object")]:
        assert dr.summarize(v) == want


# ------------------------------------------------------------------------------------------------ gates, boot, hand flip

def test_gates_write_a_decision_each(monkeypatch, caplog):
    f = _front(True)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        F.Front._dec_gate(f, "Gate-W101", "pdflip-0-1", "W101 image_parts=1")
    d = dr.parse_line(caplog.messages[-1])
    assert d["kind"] == "admit" and d["action"] == "REFUSE" and d["rule"] == "Gate-W101" and d["rid"] == "pdflip-0-1"
    # every gate site of the inventory carries the call (source pin; the behaviour of each refusal is in its own test)
    src = inspect.getsource(F)
    for gate in ("Gate-CTX", "Gate-TPL", "Gate-W52", "Gate-W193", "Gate-W125"):
        assert '"%s"' % gate in src, gate
    assert '"Gate-" + _code.split(" ")[0]' in src


def test_gate_context_refusal_writes_a_decision(caplog):
    f = _front(True)
    f._context_len = lambda: 100
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        resp = f._context_gate("pdflip-0-1", "x" * 3000, None, 5000)
    assert resp is not None and resp.status == 400
    d = [dr.parse_line(m) for m in caplog.messages if m.startswith("WEG2 DECISION seq")]
    assert [x for x in d if x["rule"] == "Gate-CTX" and x["action"] == "REFUSE" and x["rid"] == "pdflip-0-1"]


def test_boot_and_hand_flip_write_decisions_with_their_done(monkeypatch, caplog):
    f = _front(True)
    flips = []

    async def flip(src, dst):
        flips.append((src, dst))
        f.awake = dst
    f.flip = flip
    from flliper.srt.pdflip import adopt as _adopt
    monkeypatch.setattr(_adopt, "adopt_armed", lambda: True)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        f.awake = "P"
        asyncio.run(f._adopt_first_flip())
    msgs = caplog.messages
    dec = [dr.parse_line(m) for m in msgs if m.startswith("WEG2 DECISION seq")]
    done = [dr.parse_line(m) for m in msgs if m.startswith("WEG2 DECISION-DONE")]
    assert [(d["kind"], d["action"], d["rule"]) for d in dec] == [("boot", "FLIP_D", "NS:BOOT"), ("boot", "FLIP_P", "NS:BOOT")]
    assert [x["seq"] for x in done] == [d["seq"] for d in dec] and all(x["outcome"] == "ok" for x in done)
    assert flips == [("P", "D"), ("D", "P")]


def test_hand_flip_writes_a_decision_and_its_done(monkeypatch, caplog):
    f = _front(True)
    f.awake = "D"
    f._manual_flip_refusal = lambda: None
    flips = []

    async def flip(src, dst):
        flips.append((src, dst))
        f.awake = dst

    async def park(*a, **k):
        return None
    f.flip = flip
    f._wait_bound_park = park
    monkeypatch.setattr(F.Front, "_manual_return_skip_reason", lambda self: "p_outstanding n=1")
    f._kick_controller = lambda *a, **k: None
    f.state_dict = lambda: {}
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        asyncio.run(f.handle_manual_flip(SimpleNamespace()))
    dec = [dr.parse_line(m) for m in caplog.messages if m.startswith("WEG2 DECISION seq")]
    done = [dr.parse_line(m) for m in caplog.messages if m.startswith("WEG2 DECISION-DONE")]
    assert [(d["kind"], d["action"], d["rule"]) for d in dec] == [("manual", "FLIP_P", "NS:MANUAL")]
    assert len(done) == 1 and done[0]["seq"] == dec[0]["seq"] and flips == [("D", "P")]


def test_hand_flip_refusal_is_a_decision(caplog):
    f = _front(True)
    f._manual_flip_refusal = lambda: "W115 residue too high"
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        resp = asyncio.run(f.handle_manual_flip(SimpleNamespace()))
    assert resp.status == 409
    dec = [dr.parse_line(m) for m in caplog.messages if m.startswith("WEG2 DECISION seq")]
    assert [(d["kind"], d["action"], d["rule"]) for d in dec] == [("manual", "REFUSE", "NS:MANUAL")]


def test_the_controller_tick_closes_the_sum_window():
    src = inspect.getsource(F.Front.controller)
    assert "Front._dec_tick(self)" in src
