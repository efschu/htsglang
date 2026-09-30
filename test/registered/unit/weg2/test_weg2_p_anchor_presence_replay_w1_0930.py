"""#49 L1 (30.09.): replay of the REAL agent trace w109290020 through the front's token spans, with the
P-anchor presence witness (SGLANG_WEG2_ENABLE_P_ANCHOR_PRESENCE, PREFILL-EINBRUCH-0929 K1) off and on.

Trace: boot_weg2_dkr27browauthoritybar1w109290020_bb82fbcb68 (29.09. 00:20-00:54Z, 30 min agent load,
one agent tree), 993 events extracted from its front log into fixtures/front_span_49/w109290020_events.json
(flips with epoch/awake, X-EXACT prices with tokens + reused prefix, route verdicts with X, P leg-1 and
D leg-2 serves with cached/prompt/epoch/#59 depth, leg-2 first content with its via).

The replay drives the SAME objects the front drives, in the log's order:
  * pricing        TokenSpans.pending(ids, epoch)  -- epoch only while D is awake (the router's rule);
  * D leg-2 finish TokenSpans.record_presence(ids, ct, pt, held_epoch if 200+priced, depth);
  * first content  d_direct -> TokenSpans.record_inflight (FRONT_SPAN_INFLIGHT, on on the 27B row);
                   after_p  -> TokenSpans.record_store_anchor(ids, P prompt) ONLY with the switch on.
Token ids are synthesised from the log's facts: a request whose X-EXACT ``reused`` prefix equals an
earlier request's whole prompt shares exactly that prefix with it (fresh tokens after it); otherwise it
shares ``reused`` tokens with the latest earlier request that long.

Measured (tools/front_route_gap on the same log): 96 of 185 routes went over P, 34 needed P. That log
predates K2 (a59c95ae36, a D-park resume clamped to its own text); replayed through TODAY's spans:
  * OFF: 58 of the 182 priced+routed turns LONG (the log: 94) -- K2 already moved 36;
  * ON (the P-anchor witness): 43 LONG -- 15 more turns D-direct, none back to LONG.
Pinned here:
  * OFF never routes SHORT what P then prefilled over X;
  * ON moves >= 12 turns; a moved turn whose P side held less than the credit (P leg-1 rest > X:
    weg2-20-21, 36-39, 42-42, 118-130) is safe only because its source was RUNNING on D at the price
    time (first content given, leg 2 not finished -- D's radix held it); any other such move would be a
    W50 over-credit and fails the test;
  * a credit only appears after a first-content witness, never below 0.
Replayed with today's reprice semantics (a re-priced queued LONG reaches D only through the short drain,
i.e. while D is awake and serving): OFF 59, L1 38 (21 moved, 7 of them by the reprice at the witness).
L2 (#49, D-epoch publish presence): alone 56, on top of L1 still 38 (see its test). The
router's held-credit rule includes "not while flipping" (flip begin .. done) and D's #59 depths come
from the D log's '#59 RESUMABLE' lines.
"""

import json
import pathlib

import numpy as np
import pytest

from sglang.srt.weg2.front_tokens import TokenSpans

FIX = pathlib.Path(__file__).with_name("fixtures") / "front_span_49" / "w109290020_events.json"


def replay(anchor_on: bool, publish_on: bool = False):
    ev = json.loads(FIX.read_text())["events"]
    ts = TokenSpans(agent_span=True)
    ids, order, fresh = {}, [], [10_000_000]
    epoch, awake, flipping = 0, "D", False
    p1, out, anchor_t = {}, {}, {}
    src_of, fc_t, d2_t = {}, {}, {}
    # when each LONG route leaves the queue for P: the first D->P flip done after its price
    dp_done = [e["t"] for e in ev if e["k"] == "flip" and e["awake"] == "P"]

    def dispatch_t(t):
        for d in dp_done:
            if d > t:
                return d
        return 1e18

    def new_ids(tokens, reused):
        src = None
        if reused > 0:
            for r in reversed(order):
                if ids[r].size == reused:
                    src = r
                    break
            if src is None:
                for r in reversed(order):
                    if ids[r].size >= reused:
                        src = r
                        break
        new_ids.src = src
        head = ids[src][:reused] if src is not None else np.arange(0, 0, dtype=np.int64)
        n = tokens - head.size
        tail = np.arange(fresh[0], fresh[0] + n, dtype=np.int64)
        fresh[0] += n
        return np.concatenate([head, tail])

    def price_epoch():
        # the router's rule: a held (#49) credit only while D is awake AND serving
        return epoch if (awake == "D" and not flipping) else None

    def reprice(t, why):
        # _x_exact_reprice_queue re-prices a queued LONG (routing flags unchanged); it reaches D only
        # through the short drain (_d_short_drain), i.e. while D is awake and serving -- a reprice at
        # a D->P flip done moves nothing (P is awake and takes the queue)
        if not (awake == "D" and not flipping):
            return
        for rid, o in out.items():
            if o.get("long_now") and o["t"] < t < o["dispatch"]:
                pend, credit, _k, _s = ts.pending(ids[rid], epoch=price_epoch())
                if pend <= o["X"]:
                    o.update(pending=pend, credit=credit, long_now=False, moved_by=why)

    for e in ev:
        k = e["k"]
        if k == "flip_begin":
            flipping = True
        elif k == "flip":
            epoch, awake, flipping = e["epoch"], e["awake"], False
            if publish_on and awake == "P":
                if ts.promote_published(epoch)[0]:
                    reprice(e["t"] - 1e-3, "d_epoch_publish")   # before the controller dispatches to P
        elif k == "price":
            rid = e["rid"]
            if rid not in ids:
                ids[rid] = new_ids(e["tokens"], e["reused"])
                src_of[rid] = new_ids.src
                order.append(rid)
            pend, credit, _known, _src = ts.pending(ids[rid], epoch=price_epoch())
            out[rid] = {"pending": pend, "credit": credit, "t": e["t"], "log_credit": e["credit"],
                        "dispatch": dispatch_t(e["t"])}
        elif k == "route":
            o = out.get(e["rid"])
            if o is not None:
                o.update(X=e["X"], log_verdict=e["verdict"], long_now=o["pending"] > e["X"])
        elif k == "p1":
            p1[e["rid"]] = (e["pt"], e["ct"])
        elif k == "fc" and e["rid"] in ids:
            fc_t[e["rid"]] = e["t"]
            if e["via"] == "d_direct":
                ts.record_inflight(ids[e["rid"]], e["epoch"])
            elif e["via"] == "after_p" and anchor_on and e["rid"] in p1:
                if ts.record_store_anchor(ids[e["rid"]], p1[e["rid"]][0]) > 0:
                    anchor_t[e["rid"]] = e["t"]
                    reprice(e["t"], "p_anchor")
        elif k == "d2" and e["rid"] in ids:
            d2_t[e["rid"]] = e["t"]
            held = e["status"] == 200 and e["priced"] and e["verdict"] != "reroute"
            ts.record_presence(ids[e["rid"]], e["ct"], prompt_tokens=e["pt"],
                               held_epoch=e["epoch"] if held else None, resumable_depth=e["depth"])
    for rid, o in out.items():
        o["long"] = bool(o.get("long_now"))
        o["p1"] = p1.get(rid)
        s = src_of.get(rid)
        # the source was on D (first content given, leg 2 not yet finished) when this text was priced:
        # D's radix held it whatever P's side held
        o["src_on_d"] = bool(s and fc_t.get(s, 1e18) < o["t"] < d2_t.get(s, 1e18))
    return out, anchor_t


@pytest.fixture(scope="module")
def off():
    return replay(False)[0]


@pytest.fixture(scope="module")
def on():
    return replay(True)


def _routed(o):
    return {r: v for r, v in o.items() if v.get("log_verdict") in ("long", "short")}


def test_off_is_todays_code_and_already_below_the_log(off):
    """The log (bb82fbcb68, 00:20Z 29.09.) predates K2 (a59c95ae36, 07:55Z: a D-park resume's reading
    clamped to its own text), so today's spans credit more than the log did: 94 LONG in the log (of 182
    priced + routed), fewer in the replay. Pinned so a regression of today's OFF path shows."""
    r = _routed(off)
    log_long = sum(1 for v in r.values() if v["log_verdict"] == "long")
    model_long = sum(1 for v in r.values() if v["long"])
    assert log_long == 94 and len(r) == 182
    assert model_long <= 62, model_long
    # OFF never routes SHORT what P then had to prefill over X (the measured side of #1324)
    assert [x for x, v in r.items() if not v["long"] and v["p1"] and v["p1"][0] - v["p1"][1] > v["X"]] == []


def test_on_moves_turns_to_d_and_never_over_credits(off, on):
    o_on, _ = on
    r_off, r_on = _routed(off), _routed(o_on)
    moved = [rid for rid in r_on if r_off[rid]["long"] and not r_on[rid]["long"]]
    back = [rid for rid in r_on if not r_off[rid]["long"] and r_on[rid]["long"]]
    assert len(moved) >= 18, len(moved)
    assert back == []
    for rid in moved:
        v = r_on[rid]
        assert v["p1"] is not None, rid          # it went over P in reality: P's side is known
        pt, ct = v["p1"]
        if pt - ct > v["X"]:
            # P's own side held less than the credit -- safe ONLY if D held the source itself (its
            # leg 2 was running on D at the price time); anything else is a W50 over-credit
            assert v["src_on_d"], (rid, pt, ct, v["credit"])


def test_credit_only_after_the_witness(on):
    """Every credit ON adds rests on an anchor recorded at an after_p FIRST CONTENT before the price."""
    o_on, anchor_t = on
    assert anchor_t, "no P-anchor presence recorded"
    first = min(anchor_t.values())
    # a turn priced before the first witness may carry a credit only if a later witness repriced it
    early = [r for r, v in o_on.items() if v["t"] < first and v["credit"] > 0 and v["log_credit"] == 0
             and v.get("moved_by") is None]
    assert early == []


def test_never_past_the_own_prompt(on):
    o_on, _ = on
    for rid, v in o_on.items():
        assert v["pending"] >= 0 and v["credit"] >= 0


@pytest.fixture(scope="module")
def on_l2():
    return replay(True, True)[0]


@pytest.fixture(scope="module")
def l2_only():
    return replay(False, True)[0]


def test_l2_publish_presence_on_this_trace(off, on, on_l2, l2_only):
    """#49 L2 (D-epoch publish presence): the texts D served in an ended epoch credit their #59 depth
    once D's sleep leg published them. On w109290020 it moves 3 turns alone (weg2-15-18, 27-27, 90-99:
    priced after the promotion) and NOTHING on top of L1 -- K1 already covers them, and the 7
    DDIRECT_GAP turns of front_route_gap were priced DURING the D->P flip that was already under way
    (state flipping: no held credit, no witness yet); their LONG route costs no extra flip, since the
    flip happened anyway. None moves back in either arm."""
    o_l1, _ = on
    r0, r1, r12, r2 = _routed(off), _routed(o_l1), _routed(on_l2), _routed(l2_only)
    alone = [rid for rid in r2 if r0[rid]["long"] and not r2[rid]["long"]]
    assert len(alone) >= 3, alone
    assert [rid for rid in r2 if not r0[rid]["long"] and r2[rid]["long"]] == []
    assert [rid for rid in r12 if not r1[rid]["long"] and r12[rid]["long"]] == []
    assert sum(v["long"] for v in r12.values()) <= sum(v["long"] for v in r1.values())
