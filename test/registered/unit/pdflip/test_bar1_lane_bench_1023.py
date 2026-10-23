"""Desk tests of scripts/pdflip/bar1_lane_bench.py (barlink-switch-1023).

CPU only: the script's pure parts -- argument parsing, the lane and raw cell
matrices, percentile/summary math, the CONTENDED rule, the artifact schema, the
synthetic descriptor plan + byte pattern (checked against the real
``tp.batch_descs`` cut), result assembly -- imported WITHOUT CUDA, a gpuq window
or NVML. What this cannot show: the metal numbers; that is the point of the
bench, which the planner runs inside a gpuq window.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.pdflip import weight_exchange_transport as tp  # noqa: E402
from flliper.srt.pdflip.weight_exchange import XchgDesc  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

BENCH = (Path(__file__).resolve().parents[4] / "scripts" / "pdflip"
         / "bar1_lane_bench.py")


def _load():
    spec = importlib.util.spec_from_file_location("bar1_lane_bench_under_test", BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load()

LANE_TIME = (
    "PDFLIP-BAR1 lane-time lane=p0 phase=bench seq=c0-fwd-p0-t1 units=40 "
    "batches=160 bytes=671088640 total_ms=3352 wait_ms=12 copy_sync_ms=3300 "
    "rate_GBs=0.2 via=bar1 credits=sock credit_ms=12 issue_ms=40 sync_ms=3300 "
    "sync_cpu_ms=3290 max_sync_ms=50.5 send_ms=3 mode=serial "
    "write_order=readbackx160 overlap_ms=0 lag=1 prep_ms=0 issue_gil=held "
    "sync_query=held sync_skipped=0/160")


# ---------------------------------------------------------------- arguments

def test_defaults():
    a = B.parse_args([])
    assert (a.src_card, a.dst_card) == (0, 1)
    assert a.directions == ["fwd", "rev", "both"]
    assert a.slots == ["prod", "7", "19"]
    assert a.write_orders == ["on", "off"]
    assert a.ring is None and a.tag_mib == 640 and a.readback_probe == 64
    assert a.cards_list == [0, 1] and not a.raw and not a.skip_lanes
    assert a.connect_probe == "off"


def test_parse_lists_and_env():
    a = B.parse_args(["--directions", "both,fwd", "--slot-mib", "7,7,prod",
                      "--write-order", "off", "--ring", "5",
                      "--env", "A=1", "--env", "B=x=y"])
    assert a.directions == ["fwd", "both"]          # canonical order, deduped
    assert a.slots == ["7", "prod"]
    assert a.write_orders == ["off"] and a.ring == 5
    assert a.env_extra == {"A": "1", "B": "x=y"}


@pytest.mark.parametrize("argv", [
    ["--directions", "sideways"],
    ["--slot-mib", "big"],
    ["--slot-mib", "0"],
    ["--write-order", "maybe"],
    ["--ring", "2"],
    ["--ring", "17"],
    ["--src-card", "1", "--dst-card", "1"],
    ["--tags", "0"],
    ["--env", "NOEQUALS"],
    ["--skip-lanes"],
    ["--cards", "0,1,2,3"],
    ["--cards", "0,0"],
    ["--raw", "--raw-window-mib", "400", "--raw-src-mib", "256"],
    ["--raw", "--raw-window-mib", "128", "--raw-src-mib", "64"],
    ["--raw", "--raw-chunk-mib", "256"],
    ["--raw", "--raw-engines", "dma"],
])
def test_parse_refusals(argv):
    with pytest.raises(SystemExit):
        B.parse_args(argv)


# ------------------------------------------------------------- cell matrix

def test_cell_matrix_and_configs():
    a = B.parse_args([])
    cells = B.expand_cells(a.directions, a.slots, a.write_orders)
    assert len(cells) == 3 * 3 * 2
    assert cells[0] == ("fwd", "prod", "on") and cells[0].direction == "fwd"
    # on/off arms of one slot are adjacent in time
    assert [c.write_order for c in cells[:6]] == ["on"] * 3 + ["off"] * 3
    cfgs = B.group_configs(cells)
    assert len(cfgs) == 6 and all(c["directions"] == ["fwd", "rev", "both"] for c in cfgs)
    assert (cfgs[0]["slot"], cfgs[0]["write_order"]) == ("prod", "on")
    assert B.lanes_for(["fwd"]) == ["p0"]
    assert B.lanes_for(["both", "rev"]) == ["p0", "p1"]


def test_receivers_for():
    a = B.parse_args(["--src-card", "0", "--dst-card", "2"])
    assert B.receivers_for(a, ["fwd"]) == [2]
    assert B.receivers_for(a, ["rev"]) == [0]
    assert sorted(B.receivers_for(a, ["both"])) == [0, 2]


# ------------------------------------------------------------------- stats

def test_percentile_and_dist():
    xs = [1, 2, 3, 4, 100]
    assert B.percentile(xs, 50) == 3
    assert B.percentile(xs, 0) == 1 and B.percentile(xs, 100) == 100
    assert B.percentile(xs, 90) == pytest.approx(61.6)
    assert B.percentile([], 50) is None and B.percentile([7], 90) == 7
    d = B.dist([1, 2, 6, 7, 9], gt=5)
    assert (d["n"], d["min"], d["max"], d["n_gt"]) == (5, 1, 9, 3)
    assert B.dist([])["p50"] is None


def test_periods_ms():
    assert B.periods_ms(10.0, [10.001, 10.034, 10.0345]) == pytest.approx([1.0, 33.0, 0.5])


def test_recorder_one_event_per_finish():
    r = B.Recorder()
    r.sync_done(1.0, 2.0)          # outside begin(): ignored
    r.begin(t0=0.0)
    r.skip(0.5)
    r.sync_done(0.6, 0.634)
    got = r.end()
    assert got["batch_ms"] == pytest.approx([500.0, 134.0])
    assert got["sync_ms"] == pytest.approx([0.0, 34.0])
    assert r.end() == {"batch_ms": [], "sync_ms": []}


def test_parse_lane_time():
    d = B.parse_lane_time(LANE_TIME)
    assert d["lane"] == "p0" and d["batches"] == 160.0 and d["max_sync_ms"] == 50.5
    assert d["write_order"] == "readbackx160" and d["sync_skipped"] == "0/160"
    assert B.parse_lane_time("PDFLIP-BAR1 mapped lane=p0 bytes=1") == {}


# --------------------------------------------------------------- CONTENDED

def test_contended_rule():
    v = B.contention_verdict({0: [0, 1, 4], 1: [3, 5.0], 2: [2, 37, None]}, 5.0)
    assert not v[0]["contended"] and not v[1]["contended"]     # 5.0 is not > 5.0
    assert v[2]["contended"] and v[2]["max_util"] == 37 and v[2]["n"] == 2
    assert B.any_contended(v)
    assert not B.any_contended(B.contention_verdict({0: [None]}))
    assert B.contention_verdict({0: []})[0]["max_util"] is None


def test_window_stats():
    s = [{"t": 1, 0: (10, 100.0, 500)}, {"t": 2, 0: (50, 300.0, 900)},
         {"t": 3, 0: (None, None, None)}]
    w = B.window_stats(s, [0], 1.5, 3.5)
    assert w[0]["util_max"] == 50 and w[0]["power_max"] == 300.0 and w[0]["n"] == 2


# ----------------------------------------------- descs, pattern, real cut

def test_plan_descs_default_tag_shape():
    specs = B.plan_descs(640 << 20)
    sf, df, pay = B.footprints(specs)
    assert pay == 640 << 20
    kinds = {s["kind"] for s in specs}
    assert kinds == {"FLAT", "STRIDED2D"}
    two = [s for s in specs if s["kind"] == "STRIDED2D"]
    assert two and all(s["dpitch"] != s["spitch"] and s["dpitch"] >= s["run_bytes"] for s in two)
    assert all(s["src_off"] % 4096 == 0 and s["dst_off"] % 4096 == 0 for s in specs)
    assert sf > pay and df > pay          # pitch padding is part of the footprint
    assert B.plan_descs(640 << 20) == specs      # deterministic
    # logical offsets tile the payload stream exactly
    pos = 0
    for s in specs:
        assert s["logical_off"] == pos
        pos += s["nbytes"]


SMALL = (("FLAT", (3 << 20) + 8), ("2D", 4096, 64), ("FLAT", 1 << 20))


def _descs(specs, src, dst):
    return [XchgDesc(tag=B.TAG_NAME, src_rank=0, dst_rank=1,
                     param_name=d["param_name"], kind=d["kind"], nbytes=d["nbytes"],
                     rows=d["rows"], run_bytes=d["run_bytes"], spitch=d["spitch"],
                     dpitch=d["dpitch"], src_off=d["src_off"], dst_off=d["dst_off"],
                     pieces=1, src_ptr=0 if src else None, dst_ptr=0 if dst else None)
            for d in specs]


def _through_slots(specs, src_arena, dst_arena, slot_bytes):
    """What the lane does to the bytes, on the host: depositor writes each
    batch's pieces into a slot, collector scatters them out -- the real cut."""
    descs = _descs(specs, True, True)
    slot = torch.zeros(slot_bytes, dtype=torch.uint8)
    for batch in tp.batch_descs(descs, slot_bytes=slot_bytes, first_seq=0):
        for pc in batch.pieces:
            if pc.kind == tp.FLAT:
                slot[pc.slot_off:pc.slot_off + pc.nbytes] = \
                    src_arena[pc.src_off:pc.src_off + pc.nbytes]
            else:
                for r in range(pc.rows):
                    o = pc.src_off + r * pc.spitch
                    slot[pc.slot_off + r * pc.run_bytes:
                         pc.slot_off + (r + 1) * pc.run_bytes] = src_arena[o:o + pc.run_bytes]
        for pc in batch.pieces:
            if pc.kind == tp.FLAT:
                dst_arena[pc.dst_off:pc.dst_off + pc.nbytes] = \
                    slot[pc.slot_off:pc.slot_off + pc.nbytes]
            else:
                for r in range(pc.rows):
                    o = pc.dst_off + r * pc.dpitch
                    dst_arena[o:o + pc.run_bytes] = \
                        slot[pc.slot_off + r * pc.run_bytes:
                             pc.slot_off + (r + 1) * pc.run_bytes]


@pytest.mark.parametrize("slot_mib", [2, 3, 7])
def test_pattern_roundtrip_through_the_real_batch_cut(monkeypatch, slot_mib):
    monkeypatch.setattr(B, "CHUNK", 1 << 20)
    specs = B.plan_descs(11 << 20, SMALL)
    assert {s["kind"] for s in specs} == {"FLAT", "STRIDED2D"}
    sf, df, _ = B.footprints(specs)
    src = torch.empty(sf, dtype=torch.uint8)
    dst = torch.zeros(df, dtype=torch.uint8)
    B.fill_src(src, specs, seed=42)
    _through_slots(specs, src, dst, slot_mib << 20)
    got = B.check_dst(dst, specs, seed=42)
    assert got == {"bad_payload": 0, "bad_padding": 0, "first_bad": ""}
    # a stale tag (other seed), a flipped byte and a scribbled pad are all caught
    assert B.check_dst(dst, specs, seed=43)["bad_payload"] > 0
    dst[specs[0]["dst_off"] + 5] ^= 0x55
    bad = B.check_dst(dst, specs, seed=42)
    assert bad["bad_payload"] == 1 and bad["first_bad"] == "w0"
    dst[specs[0]["dst_off"] + 5] ^= 0x55
    two = next(s for s in specs if s["kind"] == "STRIDED2D")
    dst[two["dst_off"] + two["run_bytes"] + 1] = 9          # inside the pitch gap
    assert B.check_dst(dst, specs, seed=42)["bad_padding"] == 1


def test_pattern_is_never_zero_and_position_dependent():
    p = B.pattern(0, 1 << 16, 7, "cpu")
    assert int(p.min()) >= 1 and p.dtype == torch.uint8
    assert not torch.equal(B.pattern(0, 4096, 7, "cpu"), B.pattern(4096, 4096, 7, "cpu"))
    assert not torch.equal(B.pattern(0, 4096, 7, "cpu"), B.pattern(0, 4096, 8, "cpu"))


# ----------------------------------------------------------- cell assembly

def _run(direction, lane, role, tag, wall, *, warm=False, batch=(1.0,), err="",
         verify=None, nbytes=640 << 20, lt=True):
    return {"direction": direction, "lane": lane, "role": role, "tag": tag,
            "warm": warm, "seq": f"s{tag}", "wall_ms": wall, "bytes": nbytes,
            "err": err, "lane_time_raw": LANE_TIME if lt else "",
            "lane_time": B.parse_lane_time(LANE_TIME) if lt else {},
            "batch_ms": list(batch), "sync_ms": [x / 2 for x in batch],
            "verify": verify}


OK = {"bad_payload": 0, "bad_padding": 0, "first_bad": ""}


def _results(direction="fwd", bad=False):
    runs0, runs1 = [], []
    lane = "p0"
    for t in range(3):
        warm = t == 0
        runs0.append(_run(direction, lane, "src", t, 3000.0, warm=warm,
                          batch=(30.0, 33.0, 34.0)))
        runs1.append(_run(direction, lane, "dst", t, 3001.0, warm=warm,
                          verify={"bad_payload": 3, "bad_padding": 0, "first_bad": "w1"}
                          if (bad and t == 2) else OK, lt=False))
    probe = {"read_idle": B._lat_stats([33.0, 34.0, 4.0]), "slot_mib": 7, "n": 3}
    return {0: {"runs": runs0, "window": {lane: {"slot_mib": 7, "ring": 4, "window_mib": 28}},
                "probes": {"fwd": probe}},
            1: {"runs": runs1, "window": {lane: {"slot_mib": 7, "ring": 4, "window_mib": 28}}}}


def test_assemble_cell_ok():
    c = B.assemble_cell("fwd", "7", "on", 4, 640 << 20, _results())
    assert c["status"] == "ok" and c["correct"]
    st = c["lanes"]["fwd"]
    assert st["tags"] == 2                              # the warm-up tag is not counted
    assert st["GBs"] == pytest.approx(2 * (640 << 20) / 6.0 / 1e9)
    assert st["batch_ms"]["max"] == 34.0 and st["batch_ms"]["p50"] == 33.0
    assert st["lane_time"]["max_sync_ms"]["max"] == 50.5
    assert c["slot_mib"] == {"fwd": 7}
    assert c["readback"]["fwd"]["read_idle"]["n_gt"] == 2
    assert len(c["lane_lines"]) == 2


def test_assemble_cell_wrong_bytes_is_loud():
    c = B.assemble_cell("fwd", "7", "on", 4, 640 << 20, _results(bad=True))
    assert c["status"] == "WRONG" and not c["correct"] and c["wrong"]
    rows = B.summary_rows([c])
    assert rows[0]["status"] == "WRONG"
    assert "WRONG" in B.format_summary(rows)


def test_assemble_cell_error_and_missing_verification():
    res = _results()
    res[0]["runs"][2]["err"] = "bar1 deposit lane=p0: no 'free'"
    c = B.assemble_cell("fwd", "7", "on", 4, 640 << 20, res)
    assert c["status"] == "ERROR" and any("free" in n for n in c["notes"])
    res = _results()
    res[1]["runs"] = []
    c = B.assemble_cell("fwd", "7", "on", 4, 640 << 20, res)
    assert c["status"] == "ERROR"


def test_both_rows_and_refused_cell():
    ok = B.assemble_cell("fwd", "prod", "off", 4, 640 << 20, _results())
    both = dict(ok, direction="both",
                lanes={"fwd": ok["lanes"]["fwd"], "rev": ok["lanes"]["fwd"]},
                agg_GBs=1.5)
    refused = B.refused_cell("rev", "19", "on", 4, "REFUSED", "BAR1 too small")
    rows = B.summary_rows([ok, both, refused])
    names = [r["direction"] for r in rows]
    assert names == ["fwd", "both:fwd", "both:rev", "both:sum", "rev"]
    assert rows[0]["rb_p50"] is not None and rows[-1]["status"] == "REFUSED"
    text = B.format_summary(rows)
    assert "REFUSED: BAR1 too small" in text and text.count("\n") == len(rows) + 1


# ----------------------------------------------------------------- raw cells

def test_raw_patterns_two_and_three_cards():
    p2 = dict(B.expand_raw_patterns(2, "auto", ["fwd", "both"]))
    assert p2 == {"fwd": [(0, 1)], "both": [(0, 1), (1, 0)]}
    p3 = B.expand_raw_patterns(3, "auto")
    names = [n for n, _ in p3]
    assert len(p3) == 6 + 3 + 2
    assert {"0>1", "2>1", "0<>2", "fan", "ring"} <= set(names)
    d = dict(p3)
    assert d["fan"] == [(0, 1), (0, 2)] and d["ring"] == [(0, 1), (1, 2), (2, 0)]
    assert [n for n, _ in B.expand_raw_patterns(3, "ring,fan")] == ["fan", "ring"]
    a = B.parse_args(["--raw", "--cards", "0,1,2", "--skip-lanes"])
    assert len(a.raw_pattern_list) == 11


def test_raw_cell_matrix_counts():
    a = B.parse_args(["--raw", "--cards", "0,1,2", "--skip-lanes"])
    cells = B.expand_raw_cells(a.raw_pattern_list, a.raw_engine_list, a.raw_chunks,
                               a.raw_stream_counts, a.raw_blocks)
    assert len(cells) == 11 * 3 * (3 + 3)
    assert [c["id"] for c in cells] == list(range(len(cells)))
    assert all(c["streams"] == 1 for c in cells if c["engine"] == "sm")
    assert all(c["blocks"] == 0 for c in cells if c["engine"] == "ce")


def test_raw_plan_total_and_agg():
    plan = B.raw_plan(2048 << 20, 64 << 20, 64 << 20, 256 << 20)
    assert len(plan) == 32 and all(o == 0 for o, _ in plan)          # one window slot
    assert max(s for _, s in plan) == (256 - 64) << 20 and plan[4][1] == 0   # source wraps
    p = B.raw_plan(1 << 30, 16 << 20, 64 << 20, 256 << 20)
    assert [o >> 20 for o, _ in p[:5]] == [0, 16, 32, 48, 0]
    assert B.raw_total_for(0.2, 2048 << 20, 128 << 20, 6.0) == int(0.2e9 * 6.0)
    assert B.raw_total_for(20.0, 2048 << 20, 128 << 20, 6.0) == 2048 << 20
    assert B.raw_total_for(0.0, 2048 << 20, 128 << 20, 6.0) == 128 << 20
    assert B.raw_total_for(0.001, 2048 << 20, 128 << 20, 6.0) == 128 << 20   # never below the pilot
    agg = B.raw_agg([[100.0, 200.0], [200.0, 200.0]], [1e9, 1e9])
    assert agg == pytest.approx(2e9 / 0.2 / 1e9)       # 10 GB/s in both reps


def _cell(pattern="fan", edges=((0, 1), (0, 2))):
    return {"id": 3, "pattern": pattern, "edges": [list(e) for e in edges],
            "engine": "ce", "streams": 2, "blocks": 0, "chunk_mib": 16}


def _w(e, gbs, bytes_=2 << 30, err=""):
    return {"edge": list(e), "GBs": gbs, "GBs_best": gbs, "bytes": bytes_,
            "reps_ms": [bytes_ / gbs / 1e6] * 2, "pilot_GBs": gbs, "err": err}


def test_assemble_raw_cell_ok_wrong_error():
    cell = _cell()
    ok = B.assemble_raw_cell(cell, [_w((0, 1), 10.0), _w((0, 2), 5.0)],
                             [{"edge": [0, 1], "bad": 0, "checked": 64 << 20},
                              {"edge": [0, 2], "bad": 0, "checked": 64 << 20}])
    assert ok["status"] == "ok" and ok["correct"]
    # both edges moved 2 GiB; the slow one (5 GB/s) sets the wall -> 2 x 5 = 10 GB/s
    assert ok["agg_GBs"] == pytest.approx(10.0, rel=1e-6)
    wrong = B.assemble_raw_cell(cell, [_w((0, 1), 10.0), _w((0, 2), 5.0)],
                                [{"edge": [0, 1], "bad": 0, "checked": 64 << 20},
                                 {"edge": [0, 2], "bad": 12, "checked": 64 << 20}])
    assert wrong["status"] == "WRONG" and not wrong["correct"]
    err = B.assemble_raw_cell(cell, [_w((0, 1), 10.0), _w((0, 2), 5.0, err="SM-REFUSED: x")],
                              [{"edge": [0, 1], "bad": 0, "checked": 1},
                               {"edge": [0, 2], "bad": 0, "checked": 1}])
    assert err["status"] == "ERROR"
    verr = B.assemble_raw_cell(cell, [_w((0, 1), 10.0), _w((0, 2), 5.0)],
                               [{"edge": [0, 1], "bad": 0, "checked": 1},
                                {"edge": [0, 2], "bad": 1, "checked": 0, "error": "boom"}])
    assert verr["status"] == "ERROR" and verr["edges"][1]["bad"] is None
    rows = B.raw_summary_rows([ok, wrong, err])
    text = B.format_raw_summary(rows)
    assert "0>1 10.00, 0>2 5.00" in text and "WRONG" in text and "ERROR" in text


def test_bar1_fit_named_refusal():
    assert B.bar1_fit_reason(2, 200, 128) == ""
    assert B.bar1_fit_reason(2, None, 128) == ""
    why = B.bar1_fit_reason(2, 100, 128)
    assert "card 2" in why and "128 MiB" in why and "100 MiB BAR1 free" in why
    pats = B.expand_raw_patterns(3, "auto")
    need = B.raw_bar1_need(3, pats, 64)
    assert need == {0: 128, 1: 128, 2: 128}              # two incoming edges each
    assert B.raw_bar1_need(2, B.expand_raw_patterns(2, "fwd"), 64) == {0: 0, 1: 64}


# ----------------------------------------------------- estimates / artifact

def test_estimates():
    a = B.parse_args([])
    typ, up = B.estimate_seconds(a, 4)
    assert 0 < typ < up
    v = B.lane_vram_mib(a, 4)
    assert v["total"] >= v["arena_src"] + v["arena_dst"] + 19 * 4
    a = B.parse_args(["--raw", "--cards", "0,1,2", "--skip-lanes"])
    rtyp, rup, n = B.estimate_raw_seconds(a)
    assert n == 198 and 0 < rtyp < rup
    assert B.raw_vram_mib(a, 2)["total"] < 1024


def test_artifact_schema_is_json_and_carries_contended(capsys):
    a = B.parse_args(["--raw", "--cards", "0,1,2"])
    art = B.new_artifact(a, ["--raw"])
    for key in ("schema", "created_utc", "argv", "args", "cards", "gpuq", "baseline",
                "after", "contended", "contention", "nvml_during", "worker_logs",
                "cells", "raw_cells", "summary", "raw_summary", "summary_text",
                "raw_summary_text", "notes"):
        assert key in art, key
    assert art["schema"] == B.SCHEMA and art["contended"] is False
    art["contention"] = {str(c): v for c, v in
                         B.contention_verdict({2: [3, 41]}).items()}
    art["contended"] = B.any_contended(B.contention_verdict({2: [3, 41]}))
    assert art["contention"]["2"]["contended"] and art["contended"]
    json.dumps(art, default=str)
    assert B.CONTENDED == "CONTENDED"


def test_plan_mode_touches_nothing(capsys):
    assert B.main(["--plan"]) == 0
    out = capsys.readouterr().out
    assert "LANE cells: 18 in 6 worker-pair config(s)" in out
    assert "lane VRAM per card" in out and "FLLIPER_PDFLIP_BAR1_WRITE_ORDER" in out
    assert B.main(["--plan", "--raw", "--skip-lanes", "--cards", "0,1,2"]) == 0
    out = capsys.readouterr().out
    assert "RAW cells: 198" in out and "raw VRAM per card" in out
    assert "LANE cells" not in out


def test_worker_env_sets_write_order_before_import():
    a = B.parse_args(["--ring", "6", "--env", "FLLIPER_PDFLIP_LANE_PARALLEL_COPY=0"])
    on = B._worker_env(a, {"uuid": "GPU-x"}, "on", None, a.ring)
    off = B._worker_env(a, {"uuid": "GPU-x"}, "off", None, a.ring)
    assert on["FLLIPER_PDFLIP_BAR1_WRITE_ORDER"] == "1"
    assert off["FLLIPER_PDFLIP_BAR1_WRITE_ORDER"] == "0"
    assert on["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" and on["CUDA_VISIBLE_DEVICES"] == "GPU-x"
    assert on["FLLIPER_PDFLIP_BAR1_RING_SLOTS"] == "6"
    assert on["FLLIPER_PDFLIP_BAR1_READBACK_PROBE_N"] == "0"     # --connect-probe off
    assert on["FLLIPER_PDFLIP_LANE_PARALLEL_COPY"] == "0"
    a2 = B.parse_args(["--connect-probe", "prod"])
    assert "FLLIPER_PDFLIP_BAR1_READBACK_PROBE_N" not in B._worker_env(
        a2, {"uuid": "g"}, "on", None, None)


def test_the_script_never_edits_production_lane_code():
    """The one bench-side seam is a module-constant assignment in the worker
    (bar1_lanes.BIG_BAR_MIN), and the depositor path is production's."""
    src = BENCH.read_text()
    assert "bl.BIG_BAR_MIN = 1" in src
    assert "bl.run_bar1_units(" in src and "lanes.setup(" in src
    assert "def connect_peer" not in src and "def open_window" not in src


# -- barlink-lanes2-1023: the collector-issue arms ------------------------------

def test_layer_piece_mix_is_production_shaped():
    specs = B.plan_descs(640 << 20, B.PIECE_MIXES["layer"])
    assert sum(s["nbytes"] for s in specs) == 640 << 20
    assert 150 <= len(specs) <= 260
    assert sum(1 for s in specs if s["kind"] == "STRIDED2D") >= 20
    assert B.PIECE_MIXES["bench"] is B.TEMPLATES


def test_piece_mix_and_gil_threads_parse_and_plan(capsys):
    assert B.main(["--plan", "--piece-mix", "layer", "--gil-threads", "2",
                   "--env", "FLLIPER_PDFLIP_BAR1_COLLECT_BATCH=0"]) == 0
    out = capsys.readouterr().out
    assert "205 descs" in out and "FLLIPER_PDFLIP_BAR1_COLLECT_BATCH" in out
    with pytest.raises(SystemExit):
        B.main(["--plan", "--gil-threads", "9"])


def test_gil_burner_stops():
    import threading as _th
    stop = _th.Event()
    t = _th.Thread(target=B.gil_burner, args=(stop,), daemon=True)
    t.start()
    stop.set()
    t.join(2)
    assert not t.is_alive()


def test_collector_columns_for_the_batch_ab():
    lt_d = B.parse_lane_time("PDFLIP-BAR1 lane-time lane=p0 phase=bench seq=x units=9 total_ms=40 "
                             "issue_ms=1 rate_GBs=16.0 write_order=OFF why=FLLIPER_PDFLIP_BAR1_WRITE_ORDER=0 ffwait=0")
    cols = []
    for iss, cb in ((30.0, "on"), (10.0, "on")):
        cols.append({"lane_time": B.parse_lane_time(
            f"PDFLIP-BAR1 lane-time lane=p0 phase=bench seq=x units=205 total_ms=41 issue_ms={iss:.0f} "
            f"write_order=OFF why=FLLIPER_PDFLIP_BAR1_WRITE_ORDER=0 prep_ms=3 collect_batch={cb} calls=68"),
            "verify": {"bad_payload": 0, "bad_padding": 0}})
    deps = [{"wall_ms": 40.0, "bytes": 640 << 20, "lane_time": lt_d, "batch_ms": [1.0]}] * 2
    st = B.lane_stats(deps, cols, 640 << 20)
    assert st["collect_lane_time"]["issue_ms"]["max"] == 30.0
    assert st["collect_batch"] == "on" and st["collect_batch_why"] is None
    off = [{"lane_time": B.parse_lane_time(
        "PDFLIP-BAR1 lane-time lane=p0 issue_ms=55 write_order=readbackx3 "
        "collect_batch=OFF calls=0 why=cudart_batch_copier_refused:_canary"), "verify": None}]
    st2 = B.lane_stats(deps[:1], off, 640 << 20)
    assert st2["collect_batch"] == "OFF" and st2["collect_batch_why"].startswith("cudart_batch")
