"""TSDB (user order 01.10. ~07:40Z, docs/TSDB-DELTA-27B-1001.md) for the 27B
line: the front's /metrics (own pdflip_* + P/D relabelled), the optional Influx
push in the IPC writer thread, the NF arrival stamp, the rank gauges and the
decode-round histogram.

Danger directions pinned:
  * rid as a label (cardinality) -- never; the push carries it as a field;
  * a metric/push failure raised into the front -- counted instead;
  * the push running in the caller (event loop / flip) -- it is handed to the
    writer's ``submit``; with no env it does not exist at all;
  * a failed group scrape breaking /metrics -- it reads pdflip_group_scrape_ok 0;
  * duplicate HELP/TYPE families in the joint exposition."""

import asyncio
import inspect
import os
import socket
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.pdflip import front_metrics as fm  # noqa: E402
from flliper.srt.pdflip import front_state_ipc  # noqa: E402


def _metrics(**env):
    calls = []
    m = fm.FrontMetrics(submit=lambda fn, *a: calls.append((fn, a)), environ=env)
    return m, calls


# --- Influx line protocol / pusher --------------------------------------------------------


def test_influx_line_escapes_and_types():
    line = fm.influx_line("pdflip_req", {"model": "27B", "via": "after p", "x": None},
                          {"rid": 'pdflip-1-2"', "ttft_ms": 12.5, "prompt": 7, "ok": True, "none": None},
                          ts_ns=1)
    assert line == ('pdflip_req,model=27B,via=after\\ p '
                    'rid="pdflip-1-2\\"",ttft_ms=12.5,prompt=7i,ok=true 1')
    assert fm.influx_line("m", {}, {"a": None}) is None


def test_pusher_bundles_and_hands_the_write_to_the_writer():
    clock = [0.0]
    posted, submitted = [], []
    p = fm.InfluxPusher("http://x/write", submit=lambda fn, *a: submitted.append((fn, a)),
                        every_s=2.0, post=lambda u, b, t: posted.append(b), clock=lambda: clock[0])
    p.add("a 1i 1")
    assert not p.maybe_flush()          # not due yet
    clock[0] = 2.5
    p.add("b 2i 2")
    assert p.maybe_flush()
    assert posted == []                 # nothing written in the caller
    fn, args = submitted[0]
    fn(*args)                           # the writer thread runs it
    assert posted == [b"a 1i 1\nb 2i 2\n"] and p.writes == 1


def test_pusher_counts_errors_and_drops_the_oldest():
    def _boom(u, b, t):
        raise OSError("unreachable")
    p = fm.InfluxPusher("http://x", submit=lambda fn, *a: fn(*a), every_s=0.0, maxlen=2, post=_boom)
    for i in range(3):
        p.add(f"m v={i}i {i}")
    assert p.dropped == 1
    assert p.maybe_flush(force=True)
    assert p.errors == 1 and p.writes == 0 and not p._inflight


def test_no_env_no_push():
    m, calls = _metrics()
    assert m.pusher is None
    m.served_leg("D", "pdflip-1-1", 1.0, 10, 5, 3)
    assert calls == []


def test_push_rides_the_writer_with_rid_as_field():
    """The pdflip_req point comes from request_done (DASHBOARD-IPC) through the
    ONE pusher of the TSDB front metrics; rid is a field, never a tag."""
    from flliper.srt.pdflip import front as F

    m, calls = _metrics(**{fm.PUSH_URL_ENV: "http://vm:8428/write", fm.MODEL_ENV: "27B"})
    m.pusher.every_s = 0.0
    stub = types.SimpleNamespace(_front_metrics=m)
    stub._metrics = types.MethodType(F.Front._metrics, stub)
    assert F.Front._ipc_pusher(stub) is m.pusher
    m.served_leg("D", "pdflip-3-9", 2.0, 1000, 900, 50)
    assert calls == []                       # nothing pushed per leg
    F.Front._ipc_req_push(stub, {"rid": "pdflip-3-9", "via": "after_p", "status": 200,
                                 "ttft_ms": 1250.0, "end_ts": 1.0})
    assert calls, "the write is handed to the writer"
    fn, (lines,) = calls[-1]
    assert lines[-1].startswith("pdflip_req,")
    assert 'rid="pdflip-3-9"' in lines[-1] and "ttft_ms=1250.0" in lines[-1]
    head = lines[-1].split(" ")[0]
    assert "rid" not in head and "via=after_p" in head


# --- front metrics --------------------------------------------------------------------------


def test_request_metrics_and_no_rid_label():
    m, _ = _metrics()
    m.leg2_first_content("pdflip-1-7", "d_direct", 0.2, arrival_ts=10.0, now=10.7)
    m.served_leg("D", "pdflip-1-7", 3.0, 500, 400, 20)
    m.served_leg("P", "pdflip-1-8", 4.0, 800, 0, 0)
    out = m.render()
    assert 'pdflip_ttft_seconds_count{via="d_direct"} 1.0' in out
    assert 'pdflip_leg2_first_content_seconds_count{via="d_direct"} 1.0' in out
    assert 'pdflip_served_total{group="D"} 1.0' in out and 'pdflip_served_total{group="P"} 1.0' in out
    assert 'pdflip_tokens_total{group="D",kind="cached"} 400.0' in out
    assert 'pdflip_request_seconds_count{group="P"} 1.0' in out
    assert "rid" not in out.replace("pdflip_", "")  # never a rid label


def test_flip_events_and_park_rpc():
    m, _ = _metrics()
    m.on_event("flip_done", {"epoch": 4, "sleep": "D", "wake": "P", "flip_ms": 2100})
    m.on_event("flip_first_work", {"epoch": 4, "dir": "D>P", "flip_time_ms": 2600})
    m.on_event("flip_user_time", {"epoch": 4, "dir": "D>P", "flip_user_ms": 3100,
                                  "parts": {"park_rpc_ms": 400}})
    m.park_rpc_done(0.4)
    out = m.render()
    assert 'pdflip_flips_total{dir="D>P"} 1.0' in out
    for kind in ("layer", "first_work", "user"):
        assert f'pdflip_flip_seconds_count{{dir="D>P",kind="{kind}"}} 1.0' in out
    assert "pdflip_park_rpc_seconds_count 1.0" in out


def test_a_metric_failure_is_counted_never_raised():
    m, _ = _metrics()
    m.on_event("flip_done", {"sleep": "D", "wake": "P", "flip_ms": "not-a-number"})
    m.leg2_first_content("r", "d_direct", "x", arrival_ts=None)
    assert m.errors["on_event"] == 1 and m.errors["leg2_first_content"] == 1
    assert 'pdflip_metrics_errors_total{where="on_event"} 1.0' in m.render()


def test_relabel_adds_the_group_and_names_each_family_once():
    own = "# HELP flliper_x x\n# TYPE flliper_x gauge\nflliper_x 1.0\n"
    seen = fm.meta_of(own)
    p = fm.relabel_group(own + 'flliper_y{a="b"} 2.0\n', "P", seen)
    d = fm.relabel_group("# TYPE flliper_y gauge\nflliper_y 3.0\n", "D", seen)
    assert "# TYPE flliper_x" not in p           # already named by the front's text
    assert 'flliper_x{pdflip_group="P"} 1.0' in p
    assert 'flliper_y{pdflip_group="P",a="b"} 2.0' in p
    assert 'flliper_y{pdflip_group="D"} 3.0' in d and "# TYPE flliper_y gauge" in d


def test_aggregate_names_a_failed_group():
    m, _ = _metrics()
    out = m.aggregate([("P", "flliper_up 1.0\n"), ("D", None)])
    assert 'flliper_up{pdflip_group="P"} 1.0' in out
    assert 'pdflip_group_scrape_ok{pdflip_group="D"} 0.0' in out
    assert 'pdflip_group_scrape_ok{pdflip_group="P"} 1.0' in out


# --- the front's route, arrival stamp and rid end ----------------------------------------


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_handle_metrics_aggregates_p_and_d_and_survives_a_dead_group():
    from aiohttp import ClientSession, web

    from flliper.srt.pdflip import front as F

    async def _run():
        app = web.Application()
        app.router.add_get("/metrics", lambda r: web.Response(text="flliper_num_running_reqs 2.0\n"))
        runner = web.AppRunner(app)
        await runner.setup()
        port = _free_port()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()
        try:
            async with ClientSession() as session:
                stub = types.SimpleNamespace(
                    groups={"P": types.SimpleNamespace(url=f"http://127.0.0.1:{port}", outstanding={}),
                            "D": types.SimpleNamespace(url=f"http://127.0.0.1:{_free_port()}",
                                                       outstanding={"pdflip-1-1": 0.0})},
                    session=session, queue=[1, 2], awake="D", d_bs=6, _d_parked={})
                stub._metrics = types.MethodType(F.Front._metrics, stub)
                stub._ipc_submit = lambda fn, *a: None
                resp = await F.Front.handle_metrics(stub, None)
                return resp.status, resp.body.decode()
        finally:
            await runner.cleanup()

    status, body = asyncio.run(_run())
    assert status == 200
    assert 'flliper_num_running_reqs{pdflip_group="P"} 2.0' in body
    assert 'pdflip_group_scrape_ok{pdflip_group="D"} 0.0' in body
    assert "pdflip_queue_len 2.0" in body and "pdflip_outstanding 1.0" in body
    assert 'pdflip_awake{group="D"} 1.0' in body and "pdflip_d_seats 6.0" in body


def test_metrics_is_the_fronts_route_not_a_passthrough():
    from flliper.srt.pdflip import front as F

    assert "/metrics" not in F.PASSTHROUGH_GET and F.METRICS_PATH == "/metrics"
    src = inspect.getsource(F.main)
    assert "app.router.add_get(METRICS_PATH, front.handle_metrics)" in src
    assert "front.ipc_out_wrap(front.handle_generate)" in src


def test_arrival_stamp_and_rid_end_on_every_exit():
    from flliper.srt.pdflip import front as F
    from flliper.srt.pdflip import handoff_seam as hs

    stub = types.SimpleNamespace()
    stub._ipc_out_book = types.MethodType(F.Front._ipc_out_book, stub)
    stub._ipc_out_book().arrive("pdflip-2-5", 12.0)
    assert stub._ipc_out_book().arrival["pdflip-2-5"] == 12.0

    async def _handler(request):
        raise RuntimeError("leg failed")

    wrapped = F.Front.ipc_out_wrap(stub, _handler)
    with pytest.raises(RuntimeError):
        asyncio.run(wrapped({hs.RID_KEY: "pdflip-2-5"}))
    assert "pdflip-2-5" not in stub._ipc_out_book().arrival


def test_outstanding_book_is_the_nf_class():
    b = front_state_ipc.OutstandingBook()
    b.arrive("r1", 1.0)
    b.arrive("r1", 5.0)          # the first stamp stands
    assert b.arrival["r1"] == 1.0
    blk = b.block(10.0, queued=[], p_out={}, d_out={"r1": 2.0}, parked=(), flipping=False)
    assert blk["outstanding_n"] == 1 and blk["oldest_outstanding_where"] == "D"


# --- ranks --------------------------------------------------------------------------------


def test_rank_metrics_are_off_without_server_metrics(monkeypatch):
    from flliper.srt.pdflip import rank_metrics as rm

    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    rm._reset_for_tests()
    rm.observe_decode_round(2, 0.03)
    rm.on_rankstats({"sched": {"running": 1}}, 0, 1)
    assert rm._M is None and rm.ERRORS == {}


def test_rank_gauges_round_histogram_and_token_delta(monkeypatch, tmp_path):
    import prometheus_client as pc

    from flliper.srt.pdflip import rank_metrics as rm

    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    rm._reset_for_tests()
    try:
        rm.observe_decode_round(2, 0.031)
        rec = {"sched": {"running": 3, "full_token_usage": 0.42}, "tokens": {"decode_total": 100}}
        rm.on_rankstats(rec, 0, 2, vram_bytes=123)
        rm.on_rankstats(dict(rec, tokens={"decode_total": 160}), 0, 2, vram_bytes=124)
        out = pc.generate_latest(pc.REGISTRY).decode()
        assert 'pdflip_decode_round_seconds_count{bs="2"} 1.0' in out
        assert 'pdflip_rank_running{pp_rank="2",tp_rank="0"} 3.0' in out
        assert 'pdflip_rank_kv_usage_ratio{pp_rank="2",tp_rank="0"} 0.42' in out
        assert 'pdflip_rank_vram_used_bytes{pp_rank="2",tp_rank="0"} 124.0' in out
        assert "pdflip_decode_tokens_total 60.0" in out
        assert rm.ERRORS == {}
    finally:
        for c in list((rm._M or {}).values()):
            try:
                pc.REGISTRY.unregister(c)
            except Exception:  # noqa: BLE001
                pass
        rm._reset_for_tests()


def test_rankstats_timer_feeds_the_rank_metrics(monkeypatch):
    from flliper.srt.pdflip import rank_metrics as rm
    from flliper.srt.pdflip import rankstats

    seen = []
    monkeypatch.setattr(rm, "on_rankstats", lambda rec, tp, pp: seen.append((rec["sched"], tp, pp)))
    rs = rankstats.RankStats(state_dir="/tmp/pdflip-tsdb-test", group="P", tp_rank=0, pp_rank=1,
                             read_counters=lambda: {"sched": {"running": 2}}, period=60.0)
    rs.record()
    rs.sync_metrics()
    assert seen == [({"running": 2}, 0, 1)]
