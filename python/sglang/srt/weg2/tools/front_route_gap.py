"""#49 FRONT-SPAN GAP -- how many agent turns went over P (+2 flips) that the store would have let D serve.

    python3 -m sglang.srt.weg2.tools.front_route_gap <boot>.front.log

Reads ONE front log (no GPU, no tokenizer). Per LONG route (P leg 1 -> D leg 2):
  real_rest = P leg-1 prompt_tokens - P leg-1 cached_tokens (what P really prefilled; the store/radix held the rest)
  the SOURCE = the earlier request whose exact prompt_tokens equals this one's X-EXACT ``reused`` prefix
  (the front tokenizer's longest cached tokenisation = an earlier full prompt), and WHEN that source became
  a store presence (its after_p leg-2 FIRST CONTENT = D resumed P's END anchor; a d_direct source only at
  its D leg-2 finish).
Classes of a LONG turn:
  OK_LONG               real_rest > X: P was right
  K1_FIXES              real_rest <= X, after_p source, first content BEFORE the route (P_ANCHOR_PRESENCE witness)
  INFLIGHT(after_p)     real_rest <= X, after_p source not yet at first content (reprice-queue territory)
  DDIRECT_GAP_served    real_rest <= X, source D-direct served before the route; the front credited only its
                        arrival cached_tokens after the epoch (#49: a D serve teaches the span only in-epoch)
  OTHER                 real_rest <= X, no source the front saw (e.g. a shorter LCP than any whole earlier prompt:
                        repeated documents diverging inside -- dual16 P2 --, or L3 content of an earlier boot)
"""

import collections
import datetime as dt
import re


def classify(lines):
    def ts(l):
        m = re.match(r"\[(\S+ \S+?)\]", l); return dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp() if m else None
    rv, xp, p1, d2, fc, route_t = {}, {}, {}, {}, {}, {}
    prompt_of = {}
    for l in lines:
        m = re.search(r"ROUTE-VERDICT rid=(\S+) verdict=(\w+) uncached=(\d+) \(base for X=(\d+).*?presence_span=(\d+) presence_src=(\w+)", l)
        if m:
            rv[m.group(1)] = (m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)), m.group(6)); route_t[m.group(1)] = ts(l); continue
        m = re.search(r"X-EXACT-PRICE rid=(\S+) pending=(\d+) tokens=(\d+) credit=(\d+) src=(\w+).*?reused=(\d+) encoded=(\d+)", l)
        if m:
            xp[m.group(1)] = dict(tokens=int(m.group(3)), credit=int(m.group(4)), reused=int(m.group(6))); prompt_of[m.group(1)] = int(m.group(3)); continue
        m = re.search(r"WEG2-SERVED group=P leg=1 rid=(\S+) prompt_tokens=(\d+) cached_tokens=(\d+)", l)
        if m:
            p1[m.group(1)] = (int(m.group(2)), int(m.group(3)), ts(l)); continue
        m = re.search(r"WEG2-SERVED group=D leg=2 rid=(\S+) .*?prompt_tokens=(\d+) cached_tokens=(\d+).*?verdict=(\w+).*?epoch=(\d+)", l)
        if m:
            d2[m.group(1)] = (int(m.group(2)), int(m.group(3)), m.group(4), ts(l), int(m.group(5))); continue
        m = re.search(r"LEG2-FIRST-CONTENT rid=(\S+) epoch=(\d+) via=(\w+)", l)
        if m:
            fc[m.group(1)] = (m.group(3), ts(l), int(m.group(2)))
    cls = collections.Counter(); ex = collections.defaultdict(list)
    order = sorted(route_t, key=lambda r: route_t[r])
    for rid in order:
        v = rv[rid]
        if v[0] != "long":
            continue
        X = v[2]
        if rid not in p1:
            cls["LONG_no_P_leg"] += 1; continue
        pt, pc, _ = p1[rid]
        rest = pt - pc
        if rest > X:
            cls["OK_LONG"] += 1; continue
        x = xp.get(rid)
        src = None
        if x and x["reused"] > 0:
            cands = [r for r in order if route_t[r] < route_t[rid] and prompt_of.get(r) == x["reused"]]
            src = cands[-1] if cands else None
        if src is None:
            cls["OTHER"] += 1; ex["OTHER"].append(rid); continue
        via, t_fc, ep = fc.get(src, (None, None, None))
        t = route_t[rid]
        if via == "after_p" and t_fc is not None and t_fc < t:
            cls["K1_FIXES"] += 1; ex["K1_FIXES"].append((rid, src, round(t - t_fc, 1)))
        elif via in ("d_direct", "d_single") and src in d2 and d2[src][3] < t:
            cls["DDIRECT_GAP_served"] += 1; ex["DDIRECT_GAP_served"].append((rid, src, x["credit"], x["reused"]))
        elif via in ("d_direct", "d_single"):
            cls["DDIRECT_GAP_inflight"] += 1; ex["DDIRECT_GAP_inflight"].append((rid, src, x["credit"], x["reused"]))
        else:
            cls["INFLIGHT(" + str(via) + ")"] += 1; ex["INFLIGHT"].append((rid, src))
    routed = sum(1 for r in rv.values() if r[0] in ("long", "short"))
    long_ = sum(1 for r in rv.values() if r[0] == "long")
    return routed, long_, cls, ex


def main(argv=None):
    import sys as _s
    argv = _s.argv[1:] if argv is None else argv
    with open(argv[0], errors="replace") as fh:
        routed, long_, cls, ex = classify(fh)
    print("routed turns (long+short): %d  over P (long): %d  needed P (OK_LONG): %d"
          % (routed, long_, cls.get("OK_LONG", 0)))
    for k, n in sorted(cls.items()):
        print("  %-26s %d" % (k, n))
    for k, v in ex.items():
        print(k, v[:8])


if __name__ == "__main__":
    main()
