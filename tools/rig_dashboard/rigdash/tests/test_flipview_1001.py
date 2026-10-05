"""Flipzeit in Nutzersicht und Phase jetzt (Nutzer 01.10. ~08:20Z)."""

from rigdash import ipcboot


def _ipc(front_first_ts):
    ev = [{"type": "flip_begin", "ts": 100.0, "data": {"flip_begin_ts": 100.0, "sleep": "P", "wake": "D"}},
          {"type": "flip_done", "ts": 102.0, "data": {"flip_begin_ts": 100.0, "t": 102.0, "flip_ms": 2000, "epoch": 5, "sleep": "P", "wake": "D"}},
          {"type": "flip_begin", "ts": 200.0, "data": {"flip_begin_ts": 200.0, "sleep": "D", "wake": "P"}},
          {"type": "flip_done", "ts": 201.5, "data": {"flip_begin_ts": 200.0, "t": 201.5, "flip_ms": 1500, "epoch": 6, "sleep": "D", "wake": "P"}},
          {"type": "flip_begin", "ts": 300.0, "data": {"flip_begin_ts": 300.0, "sleep": "P", "wake": "D"}},
          {"type": "flip_done", "ts": 302.0, "data": {"flip_begin_ts": 300.0, "t": 302.0, "flip_ms": 2000, "epoch": 7, "sleep": "P", "wake": "D"}}]
    return {"ipc_events": ev,
            "flip_first_work": [{"dir": "P>D", "flip_begin_ts": 100.0, "first_work_ts": front_first_ts, "flip_time_ms": (front_first_ts - 100) * 1000, "what": "decode_token"}],
            "flip_user_time": [{"epoch": 6, "flip_user_ms": 2300, "start_ts": 199.6, "prefill_start_ts": 201.9, "idle_flip": False,
                                "parts": {"pre_begin_ms": 400, "legs_ms": 1500, "first_chunk_ms": 400, "park_rpc_ms": 400}}]}


SEGS = [{"s": 90.0, "e": 99.5, "k": "P"}, {"s": 100.0, "e": 102.0, "k": "flip_pd"}, {"s": 102.0, "e": 105.0, "k": "D"},
        {"s": 105.0, "e": 199.0, "k": "dec"}, {"s": 200.0, "e": 201.5, "k": "flip_dp"}, {"s": 202.0, "e": 299.0, "k": "P"},
        {"s": 300.0, "e": 302.0, "k": "flip_pd"}, {"s": 302.0, "e": 400.0, "k": "idle", "awake": "D"}]


def _ring(p_rise_ts=None, keys=("P.tp0pp0", "P.tp0pp1", "D.tp0pp0"), t0=95.0, t1=420.0):
    """1-s rank samples; P.tp0pp0 (first P stage) forward_ct rises at ``p_rise_ts``, P.tp0pp1 (last) 3 s later."""
    out, t = [], t0
    while t <= t1:
        r = {}
        for k in keys:
            rise = None if p_rise_ts is None else {"P.tp0pp0": p_rise_ts, "P.tp0pp1": p_rise_ts + 3.0}.get(k)
            r[k] = {"ts": t, "fwd": 5.0 + (rise is not None and t >= rise), "dtok": 1.0}
        out.append({"t": t + 0.5, "r": r})
        t += 1.0
    return out


def test_pd_early_front_event_is_never_a_small_flipzeit():
    """Nutzer 02.10.: the front's decode_token 0,2 s after flip_begin (D's layers not back, NF y6d class) is
    rejected; without a rank counter for D the flip is "fehlt (Feld ...)", never 0,2 s."""
    pd = ipcboot.flip_views(SEGS, _ipc(100.2), 450.0)[0]
    assert pd["dir"] == "P>D" and pd["kind"] == "fehlt" and pd["total_ms"] is None
    assert "decode_token" in pd["missing"]
    ok = ipcboot.flip_views(SEGS, _ipc(102.5), 450.0)[0]
    assert ok["kind"] == "ok" and round(ok["total_ms"]) == 3000            # P end 99,5 (segments) -> 102,5
    assert (round(ok["vorlauf_ms"]), round(ok["layer_ms"]), round(ok["wake_kv_dc_ms"]), round(ok["nachlauf_ms"]),
            round(ok["rest_ms"])) == (500, 2000, 0, 500, 0)
    assert round(ok["nachlauf_d_extend_ms"]) == 500


def test_dp_ends_at_the_last_stage_forward_and_idle_pd_flip():
    # D's last TP0 round of the D phase ends at 199,6 (open 199,55 + 50 gpu-ms)
    v = ipcboot.flip_views(SEGS, _ipc(102.5), 450.0, _ring(p_rise_ts=204.0), d_rounds=[(150.0, 150.05), (199.55, 199.6)])
    dp = v[1]
    assert dp["dir"] == "D>P" and dp["kind"] == "ok"
    assert round(dp["total_ms"]) == 4400              # 199,6 -> 204,0 (first forward on PP0), not the front's 2300
    assert (round(dp["vorlauf_ms"]), round(dp["layer_ms"]), round(dp["wake_kv_dc_ms"]), round(dp["nachlauf_ms"]),
            round(dp["rest_ms"])) == (400, 1500, 0, 1500, 1000)
    idle = v[2]
    assert idle["dir"] == "P>D" and idle["kind"] == "leerlauf" and idle["total_ms"] is None
    last = ipcboot.flip_last(v)
    assert last["P>D"]["n"] == 1 and last["P>D"]["idle_n"] == 1 and last["D>P"]["n"] == 1
    assert last["D>P"]["max"] == dp["total_ms"]
    # without the last stage's counter in the ring: missing, not the leg-1 dispatch
    miss = ipcboot.flip_views(SEGS, _ipc(102.5), 450.0)[1]
    assert miss["kind"] == "fehlt" and "forward_ct" in miss["missing"] and miss["total_ms"] is None


def test_phase_now_flip_running_and_decode():
    ipc = _ipc(102.5)
    ipc["ipc_events"] = ipc["ipc_events"][:4] + [{"type": "flip_begin", "ts": 410.0, "data": {"flip_begin_ts": 410.0, "sleep": "D", "wake": "P"}}]
    v = ipcboot.flip_views(SEGS, ipc, 411.0)
    p = ipcboot.phase_now(SEGS, ipc, {"state": "flipping"}, v, True, 411.0)
    assert p["k"] == "flip" and p["sub"] == "Layer-Tausch" and p["since"] == 410.0
    segs = SEGS[:4]
    p2 = ipcboot.phase_now(segs, {}, {}, [], True, 199.5)
    assert p2["label"] == "D aktiv: Decode" and p2["since"] == 105.0


def test_phase_now_flip_has_a_direction_for_the_active_frame():
    # Nutzer 02.10.: der Aktivrahmen wandert mit der Phase -- Vorlauf und Nachlauf brauchen die Richtung
    p = ipcboot.phase_now([{"k": "dec", "s": 0.0, "e": 10.0}], {}, {"state": "flipping", "awake": "D", "ts": 9.0}, [], True, 10.0)
    assert p["k"] == "flip" and p["dir"] == "D>P"
    views = [{"dir": "P>D", "kind": "fertig", "begin": 1.0, "done": 3.0}]
    p2 = ipcboot.phase_now([{"k": "flip_tail", "s": 0.0, "e": 10.0}], {}, {}, views, True, 10.0)
    assert p2["k"] == "flip" and p2["dir"] == "P>D"


def test_vorlauf_never_reaches_past_the_previous_flip_probe_artifact():
    """N3u 07:33:29/:39 (11,3/21,0 s) and N4p 09:58:14 (17,2 s): the acceptance probe's manual D->P,
    P does no work and flips back at once -- the P>D vorlauf was measured from the P chunk of the
    phase before the probe. Bounded by the previous flip's done: no P chunk in this P phase = no vorlauf."""
    ev = [{"type": "flip_begin", "ts": 100.0, "data": {"flip_begin_ts": 100.0, "sleep": "P", "wake": "D"}},
          {"type": "flip_done", "ts": 102.0, "data": {"flip_begin_ts": 100.0, "t": 102.0, "flip_ms": 2000, "epoch": 2, "sleep": "P", "wake": "D"}},
          {"type": "flip_begin", "ts": 110.0, "data": {"flip_begin_ts": 110.0, "sleep": "D", "wake": "P"}},
          {"type": "flip_done", "ts": 112.0, "data": {"flip_begin_ts": 110.0, "t": 112.0, "flip_ms": 2000, "epoch": 3, "sleep": "D", "wake": "P"}},
          {"type": "flip_begin", "ts": 112.003, "data": {"flip_begin_ts": 112.003, "sleep": "P", "wake": "D"}},
          {"type": "flip_done", "ts": 114.5, "data": {"flip_begin_ts": 112.003, "t": 114.5, "flip_ms": 2497, "epoch": 4, "sleep": "P", "wake": "D"}}]
    segs = [{"s": 90.0, "e": 99.5, "k": "P"}, {"s": 100.0, "e": 102.0, "k": "flip_pd"}, {"s": 102.5, "e": 110.0, "k": "dec"},
            {"s": 110.0, "e": 112.0, "k": "flip_dp"}, {"s": 112.003, "e": 114.5, "k": "flip_pd"}, {"s": 115.0, "e": 130.0, "k": "dec"}]
    fw = [{"dir": "P>D", "flip_begin_ts": 100.0, "first_work_ts": 102.5, "what": "decode_token"},
          {"dir": "P>D", "flip_begin_ts": 112.003, "first_work_ts": 115.0, "what": "decode_token"}]
    v = ipcboot.flip_views(segs, {"ipc_events": ev, "flip_first_work": fw}, 140.0, _ring(t0=80.0, t1=135.0))
    first, probe_back = v[0], v[2]
    assert round(first["vorlauf_ms"]) == 500                       # a real P phase keeps its vorlauf
    # Nutzer 02.10.: P computed no forward in its phase -- no last token to measure from: a Leerlauf-Flip,
    # never "flip_begin -> first decode" as a Flipzeit
    assert probe_back["dir"] == "P>D" and probe_back["kind"] == "leerlauf" and probe_back["total_ms"] is None


def test_pd_first_decode_comes_from_the_front_event_not_the_rank_raster():
    """N4p 10:00:28: front first decode 0,45 s after flip_done; the 2-s rank raster put "dec" 1,88 s
    after it, behind a D-direct extend of a new arrival (1,43 s "d_extend")."""
    ev = [{"type": "flip_begin", "ts": 100.0, "data": {"flip_begin_ts": 100.0, "sleep": "P", "wake": "D"}},
          {"type": "flip_done", "ts": 102.66, "data": {"flip_begin_ts": 100.0, "t": 102.66, "flip_ms": 2660, "epoch": 10, "sleep": "P", "wake": "D"}}]
    segs = [{"s": 90.0, "e": 99.9, "k": "P"}, {"s": 100.0, "e": 102.66, "k": "flip_pd"},
            {"s": 102.66, "e": 104.1, "k": "D"}, {"s": 104.54, "e": 130.0, "k": "dec"}]
    ipc = {"ipc_events": ev,
           "flip_first_work": [{"dir": "P>D", "flip_begin_ts": 100.0, "first_work_ts": 103.11, "what": "decode_token",
                                "flip_time_ms": 3110, "p_end_ts": 99.95, "p_end_source": "p_leg1_end"}]}
    pd = ipcboot.flip_views(segs, ipc, 140.0)[0]
    # Nutzer 02.10. ~17:50Z: the flip starts at P's last chunk end (99,9; no ring here: the P segment), the
    # front's leg-1 end (99,95) is only named -- the 50 ms between are flip time
    assert pd["end_src"].startswith("front flip_first_work") and pd["start_src"].startswith("P-Segment-Ende")
    assert pd["p_end_front"] == 99.95
    assert round(pd["nachlauf_ms"]) == 450 and round(pd["total_ms"]) == 3210
    assert round(pd["nachlauf_d_extend_ms"]) == 450                 # only the extend before the real first token
    # without the front event and without D's rank counters: missing (Nutzer 02.10.), not the segment raster
    pd2 = ipcboot.flip_views(segs, {"ipc_events": ev}, 140.0)[0]
    assert pd2["kind"] == "fehlt" and pd2["total_ms"] is None
