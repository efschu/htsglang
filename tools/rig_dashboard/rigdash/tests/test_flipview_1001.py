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


def test_pd_user_view_has_vorlauf_layer_nachlauf_with_d_extend_and_flags_the_early_front_event():
    v = ipcboot.flip_views(SEGS, _ipc(100.2), 450.0)
    pd = v[0]
    assert pd["dir"] == "P>D" and pd["kind"] == "ok"
    assert round(pd["vorlauf_ms"]) == 500 and pd["layer_ms"] == 2000 and round(pd["nachlauf_ms"]) == 3000
    assert round(pd["nachlauf_d_extend_ms"]) == 3000 and round(pd["total_ms"]) == 5500
    assert pd["front_early"] is True and round(pd["front_ms"]) == 200       # the front said 0,2 s


def test_dp_from_flip_user_time_parts_and_idle_pd_flip():
    v = ipcboot.flip_views(SEGS, _ipc(102.5), 450.0)
    dp = v[1]
    assert dp["dir"] == "D>P" and dp["total_ms"] == 2300 and dp["vorlauf_ms"] == 400 and dp["nachlauf_ms"] == 400
    idle = v[2]
    assert idle["dir"] == "P>D" and idle["kind"] == "leerlauf" and idle["total_ms"] is None
    last = ipcboot.flip_last(v)
    assert last["P>D"]["n"] == 1 and last["P>D"]["idle_n"] == 1 and last["P>D"]["front_early_n"] == 0


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
    v = ipcboot.flip_views(segs, {"ipc_events": ev}, 140.0)
    first, probe_back = v[0], v[2]
    assert round(first["vorlauf_ms"]) == 500                       # a real P phase keeps its vorlauf
    assert probe_back["dir"] == "P>D" and probe_back["vorlauf_ms"] is None
    assert probe_back["total_ms"] is not None and round(probe_back["total_ms"]) == 2997   # flip_begin -> first decode
    assert probe_back["start_src"].startswith("flip_begin")
