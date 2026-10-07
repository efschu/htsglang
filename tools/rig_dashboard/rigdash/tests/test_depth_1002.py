"""Token x–y (n neu) je Prefill und je Decode-Anfrage beim Hover (Nutzer 02.10. ~12:04Z / ~12:15Z).

27B N5a als Vorbild: ein Prefill von 136713 Token ab Tiefe 77824 (Prompt 214537) lief am Anfang schneller
als am Ende; das Dashboard zeigte nur "wie viele Token"."""

import os
import shutil
import subprocess
import tempfile

import pytest

from rigdash import activity, ipcboot, ipcstate, server

KEY = "P.tp0pp0"


def _p_ring(chunks, ext=None, t0=1000.0):
    """One-stage P group: chunks = [(new, cached, gpu_ms)], back to back from t0; one sample per chunk end.
    ext = per chunk [[rid, start, end]] (prefill.last.ext) or None."""
    ring, t, cn, cc, cg = [{"t": t0 - 1.0, "r": {KEY: {"ts": t0 - 1.0, "pnew": 0.0, "pcached": 0.0, "pchunks": 0.0,
                                                    "pcomp": 0.0, "pgpu": 0.0}}, "front": {}}], t0, 0.0, 0.0, 0.0
    for i, (n, c, ms) in enumerate(chunks):
        t += ms / 1000.0
        cn, cc, cg = cn + n, cc + c, cg + ms
        r = {"ts": t + 0.05, "pnew": cn, "pcached": cc, "pchunks": float(i + 1), "pcomp": cg, "pgpu": cg,
             "plast_t": t, "plast_gpu": ms, "plast_own": ms, "plast_new": n,
             "plast_ext": (ext[i] if ext else None)}
        ring.append({"t": t + 0.1, "r": {KEY: r}, "front": {}})
    return ring


# 136713 new tokens from depth 77824: 8 chunks, the later ones slower (deeper attention)
N5A = [(17089, 77824, 1600.0)] + [(17089, 0, 1600.0 + 400.0 * i) for i in range(1, 8)]
N5A_N = sum(c[0] for c in N5A)


def _burst(ring):
    cs = activity.chunks(ring, {KEY}, "P")
    bs = activity.bursts(cs)
    assert len(bs) == 1
    return bs[0]


def test_compact_reads_the_new_fields_and_falls_back_to_none():
    rec = {"ts": 1.0, "prefill": {"last": {"t": 1.0, "ext": [["r1", 4096, 8192], ["bad"]]}},
           "decode": {"reqs": [["r1", 25000, 120], ["r2", 30000, 7]]}}
    c = ipcboot.compact(rec)
    assert c["plast_ext"] == [["r1", 4096, 8192]]
    assert c["dreqs"] == [["r1", 25000, 120], ["r2", 30000, 7]]
    old = ipcboot.compact({"ts": 1.0, "prefill": {"last": {"t": 1.0}}, "decode": {}})
    assert old["plast_ext"] is None and old["dreqs"] is None


def test_prefill_depth_falls_back_to_the_prefix_of_the_first_chunk_and_shows_the_slope():
    b = _burst(_p_ring(N5A))
    d = activity.prefill_depth(b, None, "P")
    assert d["n"] == N5A_N
    assert d["x"] == 77824 and d["y"] == 77824 + N5A_N
    assert d["exact"] is False and "cached_tokens" in d["src"]
    # the rate falls with the depth: start (first chunks) faster than end (last chunks)
    assert d["tps_start"] > d["tps_end"] > 0
    # 15 % of the tokens = the first / last two chunks: (1,6 + 2,0) s at the start, (4,0 + 4,4) s at the end
    assert round(d["tps_start"]) == round(2 * 17089 / 3.6)
    assert round(d["tps_end"]) == round(2 * 17089 / 8.4)


def test_prefill_depth_from_request_done_names_the_request():
    ring = _p_ring(N5A)
    b = _burst(ring)
    end = b["e"] + 0.4
    done = [{"rid": "weg2-22-59", "arrival_ts": end - 70.0, "queue_ms": 3153, "p_prefill_ms": int((70.0 - 3.153) * 1000),
             "prefill": {"P": {"cached": 77824, "prompt": 77824 + N5A_N, "tokens": N5A_N, "ms": 60618}}},
            {"rid": "other", "arrival_ts": end - 900.0, "queue_ms": 10, "p_prefill_ms": 1000,
             "prefill": {"P": {"cached": 0, "prompt": 500, "tokens": 500}}}]
    d = activity.prefill_depth(b, done, "P")
    assert d["exact"] is True and d["src"].startswith("events request_done")
    assert [r["rid"] for r in d["reqs"]] == ["weg2-22-59"]
    assert (d["x"], d["y"], d["n"]) == (77824, 77824 + N5A_N, N5A_N)
    # the port seat's explicit leg-1 end wins over arrival + queue + p_prefill
    done[0]["p_leg1_end_ts"] = b["s"] - 50.0
    assert activity.prefill_depth(b, done, "P")["src"].startswith("rankstats prefill.cached")


def test_prefill_depth_from_the_chunk_extent_reaches_back_to_the_start():
    # a sample only shows the NEWEST chunk's extent; the burst's token count gives the true start
    ext, pos = [], 77824
    for n, _c, _ms in N5A:
        ext.append([["weg2-22-59", pos, pos + n]])
        pos += n
    ext[0] = None                       # the first chunk's sample was missed
    d = activity.prefill_depth(_burst(_p_ring(N5A, ext)), None, "P")
    assert d["exact"] is True and "ext" in d["src"]
    assert (d["x"], d["y"], d["n"]) == (77824, 77824 + N5A_N, N5A_N)
    assert d["reqs"] == [{"rid": "weg2-22-59", "x": 77824, "y": 77824 + N5A_N, "n": N5A_N}]


def test_timeline_prefill_segment_and_last_burst_carry_the_depth():
    ring = _p_ring(N5A)
    m = activity.Model(ring, [], [])
    now = ring[-1]["t"] + 30.0
    tl = ipcboot.timeline_view(m, False, None, now, None, [])
    ps = [x for x in tl["segs"] if x["k"] == "P"]
    assert ps and ps[0]["depth"]["x"] == 77824 and ps[0]["depth"]["n"] == N5A_N
    assert isinstance(ps[0]["depth"]["tps_start"], float)
    pv = ipcboot.prefill_view(m, "P", now, [])
    assert pv["last_burst"]["depth"]["y"] == 77824 + N5A_N
    # the sampler's history paths skip the hover detail
    tl2 = ipcboot.timeline_view(m, False, None, now, None, detail=False)
    assert all("depth" not in x for x in tl2["segs"])


def _iv(s, e, tok, bs, busy=None):
    busy = (e - s) if busy is None else busy
    return {"s": s, "e": e, "parts": [(s, e)], "dur": e - s, "tok": tok, "busy": busy,
            "bs_min": bs, "bs_max": bs, "bs_mean": bs, "seat_s": busy * bs if bs else None}


def test_decode_by_bs_splits_pure_intervals_and_names_mixed_ones():
    iv = [_iv(0, 1, 50, 1), _iv(1, 2, 50, 1), _iv(2, 3, 90, 2), _iv(3, 4, 90, 2),
          dict(_iv(4, 5, 70, None), bs_min=1, bs_max=2, bs_mean=1.5, seat_s=1.5)]
    rows = activity.decode_by_bs(iv, 0.0, 5.0)
    assert [r["bs"] for r in rows] == [1, 2, "mix"]
    assert rows[0]["tps"] == 50 and rows[0]["per_slot"] == 50
    assert rows[1]["tps"] == 90 and rows[1]["per_slot"] == 45
    assert rows[2]["bs_mean"] == 1.5
    # only the share inside the window counts
    half = activity.decode_by_bs(iv, 0.5, 1.0)
    assert half == [{"bs": 1, "tps": 50.0, "per_slot": 50.0, "bs_mean": 1.0, "busy_s": 0.5, "tok": 25.0}]


def test_decode_reqs_from_rankstats_decode_reqs_per_request():
    ring = []
    for i in range(11):
        t = 100.0 + i
        ring.append({"t": t, "r": {"D.tp0pp0": {"ts": t, "dreqs": [["a", 25000, 10 + 50 * i], ["b", 4000, 20 * i]]}}})
    rows, src = activity.decode_reqs(ring, "D.tp0pp0", 102.0, 108.0, None)
    assert src.startswith("rankstats decode.reqs")
    a = next(r for r in rows if r["rid"] == "a")
    assert (a["x"], a["y"], a["n"]) == (25000 + 110, 25000 + 410, 300) and a["tps"] == 50.0
    b = next(r for r in rows if r["rid"] == "b")
    assert b["tps"] == 20.0 and b["est"] is False


def test_decode_reqs_fall_back_to_request_done_linear_estimate():
    done = [{"rid": "r1", "first_token_ts": 100.0, "end_ts": 200.0, "decode_tokens": 1000, "context_tokens": 26000},
            {"rid": "late", "first_token_ts": 300.0, "end_ts": 310.0, "decode_tokens": 10, "context_tokens": 20}]
    rows, src = activity.decode_reqs([], None, 150.0, 160.0, done)
    assert src.startswith("events request_done")
    assert rows == [{"rid": "r1", "x": 25500, "y": 25600, "n": 100, "tps": 10.0, "est": True}]
    assert activity.decode_reqs([], None, 150.0, 160.0, []) == ([], None)


def test_events_keep_request_done_and_the_view_does_not_ship_it():
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "events.jsonl")
        with open(p, "w") as fh:
            fh.write('{"schema": "weg2.event/1", "type": "request_done", "ts": 5.0, "data": {"rid": "x", "end_ts": 5.0}}\n')
        ev = ipcstate._Events(p)
        ev.poll()
        bv = ipcstate.boot_view(d, {"boot_id": "b"}, ev, 10.0)
        assert bv["request_done"] == [{"rid": "x", "end_ts": 5.0}]
        v = ipcboot.build_view(bv, [], {"rankstats": {}, "rankstate": {}}, None, 10.0)
        assert "request_done" not in v["ipc"]
    finally:
        shutil.rmtree(d)


def test_flip_view_names_the_prefill_before_a_pd_flip():
    segs = [{"s": 90.0, "e": 99.5, "k": "P", "depth": {"x": 77824, "y": 214537, "n": 136713, "exact": True,
                                                       "src": "s", "tps_start": 4300.0, "tps_end": 2400.0}},
            {"s": 100.0, "e": 102.0, "k": "flip_pd"}, {"s": 102.5, "e": 150.0, "k": "dec"}]
    ipc = {"ipc_events": [{"type": "flip_begin", "ts": 100.0, "data": {"flip_begin_ts": 100.0, "sleep": "P", "wake": "D"}},
                          {"type": "flip_done", "ts": 102.0, "data": {"flip_begin_ts": 100.0, "t": 102.0, "flip_ms": 2000,
                                                                      "epoch": 5, "sleep": "P", "wake": "D"}}]}
    row = ipcboot.flip_views(segs, ipc, 200.0)[0]
    assert row["p_depth"]["x"] == 77824 and row["p_depth"]["n"] == 136713


NODE = next((p for p in (shutil.which("node"), "/opt/node-v22.14.0-linux-x64/bin/node",
                         "/opt/node-v22.11.0-linux-x64/bin/node") if p and os.path.exists(p)), None)


def _js_block():
    html = open(os.path.join(server.STATIC, "index.html"), encoding="utf-8").read()
    a, b = html.index("/* DEPTH-BEGIN"), html.index("/* DEPTH-END */")
    return html[a:b]


def _run_js(expr, lang="de"):
    prelude = ('const window = {i18nLang: () => "%s"};\n'
               'const fmt = (v, d = 0) => (v == null || !isFinite(v)) ? "—" : Number(v).toLocaleString("en-US", '
               '{maximumFractionDigits: d, minimumFractionDigits: d});\n'
               'const esc = (s) => String(s);\n' % lang)
    out = subprocess.run([NODE, "-e", prelude + _js_block() + "\nprocess.stdout.write(" + expr + ");"],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.mark.skipif(NODE is None, reason="kein node")
def test_hover_text_prefill_and_decode_de_and_en():
    d = ('{x: 77824, y: 214537, n: 136713, tps_start: 4312.4, tps_end: 2391.0, exact: true, '
         'src: "events request_done (prefill.P cached/prompt/tokens)", reqs: [{rid: "weg2-22-59", x: 77824, y: 214537, n: 136713}]}')
    de = _run_js("depthTip(%s)" % d)
    assert "Token 77,824&ndash;214,537 (136,713 neu)" in de
    assert "4,312 &rarr; 2,391</b> tok/s" in de and "weg2-22-59" in de
    en = _run_js("depthTip(%s)" % d, "en")
    assert "Tokens 77,824&ndash;214,537 (136,713 new)" in en and "(start &rarr; end)" in en
    approx = _run_js('depthTip({x: 4096, y: 77349, n: 73253, exact: false, src: "rankstats prefill.cached_tokens (x)", reqs: []})')
    assert "Summe der Präfixe" in approx
    x = ('{by_bs: [{bs: 1, tps: 52.3, per_slot: 52.3, busy_s: 40}, {bs: 2, tps: 90.2, per_slot: 45.1, busy_s: 20}], '
         'reqs: [{rid: "a", x: 25110, y: 25410, n: 300, tps: 50, est: false}], reqs_n: 1, reqs_src: "rankstats decode.reqs (je Probe)"}')
    dec = _run_js("decodeDetailTip(%s)" % x)
    assert "bs 1: <b>52.3</b> tok/s &middot; 52.3 je Platz" in dec and "bs 2: <b>90.2</b> tok/s &middot; 45.1 je Platz" in dec
    assert "<span class=\"mono\">a</span> Token 25,110&ndash;25,410 (300 neu) &middot; <b>50.0</b> tok/s" in dec
    none = _run_js('decodeDetailTip({by_bs: [], reqs: [], reqs_n: 0, reqs_src: null})')
    assert "fehlt in IPC (rankstats decode.reqs)" in none
