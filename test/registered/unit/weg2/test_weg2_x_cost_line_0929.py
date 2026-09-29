"""X-COST-LINE (29.09., third part of the ski-rental decision): r_D under load.

THE DEFECT, MEASURED. The live X re-solve needs an r_D sample, and the only
r_D probe was the SOLO leg (nothing else admitted to D over the whole leg,
D holding only that rid, >= 2048 uncached). Under agent load that never
happens: NF z30w-park 09290827 printed 3 ``WEG2 X R_D`` lines in 32 min, all
``verdict=short``, and ``X NO-SOLVE: no r_d`` kept X at the carried-in 4096;
27B row-authority 09290020 the same. X was glued to its floor.

THE FIX. D publishes its per-forward prefill cost (the ``Prefill rank batch``
gpu-ms, mixed chunk off -> a prefill forward is prefill work only, decodes
are their own forwards) as a ring on ``/get_server_info``; the front fits
``ms = a + b*n + c*n*prefix`` over it under any load and solves
``X* = (price/k - a) / (b + c*prefix - 1000/r_P)`` with the ski instrument's
warm round trip as the price and the mean requests per P phase as k (27B
review 29.09.: amortised over the backlog, depth-aware, IPC not logs).

Fixtures: the TP0 prefill forwards and P-DRAIN windows of the two boots above
(``fixtures/park_collect_0929/x_cost_*.json``, extracted from the logs named
inside them).
"""

from __future__ import annotations

import collections
import json
import logging
import pathlib
import statistics
import time
import types

import pytest

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import phase_policy as pp  # noqa: E402
from sglang.srt.weg2 import prefill_clock as pc  # noqa: E402

FIX = pathlib.Path(__file__).parent / "fixtures" / "park_collect_0929"


def _load(name):
    doc = json.loads((FIX / f"x_cost_{name}.json").read_text())
    rows = [tuple(r) for r in doc["d_prefill_forwards_n_cached_ms"]]
    drains = doc["p_drains_prefilled_s"]
    r_p = doc["p_new_tokens_pp0"] / doc["p_drain_s_sum"]
    return rows, drains, r_p


@pytest.fixture(autouse=True)
def _clean_ring():
    pc._reset_for_tests()
    yield
    pc._reset_for_tests()


# ---------------------------------------------------------------- D side (IPC)
def test_rank_fills_the_ring_from_its_own_measurement_with_the_logger_muted():
    """27B review (3): the value in the ring comes from the rank's measured
    numbers, never parsed back out of the log line -- a muted logger fills it
    all the same."""
    from sglang.srt.managers.scheduler_components.metrics_reporter import RankPrefillLog

    class _Timer:
        def __init__(self, log):
            self.log, self.completed = log, []

        def _report(self):
            while self.completed:
                self.log._on_duration(self.completed.pop(0))

    log = RankPrefillLog()
    log.timer = _Timer(log)
    mr = logging.getLogger("sglang.srt.managers.scheduler_components.metrics_reporter")
    was = mr.disabled
    mr.disabled = True
    try:
        log.record(1574, 55744, timed=True)
        log.timer.completed.append(3.8833)       # z30w 08:46:52 TP0: 3883.3 gpu-ms
        log.flush()
    finally:
        mr.disabled = was
    snap = pc.cost_snapshot()
    assert snap["seq"] == 1
    assert snap["recent"] == [{"seq": 1, "n": 1574, "cached": 55744, "ms": 3883.3, "chunks": 1}]


def test_an_untimed_line_is_no_cost_record():
    from sglang.srt.managers.scheduler_components.metrics_reporter import RankPrefillLog

    log = RankPrefillLog()                         # no timer: the untimed line
    log.record(624, 54592, timed=True)
    assert pc.cost_snapshot()["seq"] == 0          # never a zero-cost row
    pc.note_batch_cost(25, 0, 0.0)
    assert pc.cost_snapshot()["seq"] == 0


def test_the_front_reader_takes_each_forward_once_and_names_a_wrapped_ring():
    for i in range(5):
        pc.note_batch_cost(100 + i, 0, 1000.0 + i)
    new, mark, lost = pc.cost_since(pc.cost_snapshot(), None)
    assert [r["n"] for r in new] == [100, 101, 102, 103, 104] and lost == 0
    new2, mark2, lost2 = pc.cost_since(pc.cost_snapshot(), mark)
    assert new2 == [] and mark2 == mark and lost2 == 0
    for i in range(pc.RING_MAX + 3):                # more than the ring holds
        pc.note_batch_cost(200 + i, 0, 900.0)
    new3, mark3, lost3 = pc.cost_since(pc.cost_snapshot(), mark2)
    assert len(new3) == pc.RING_MAX and lost3 == 3 and mark3[1] == 5 + pc.RING_MAX + 3
    # a D restart: a new boot id restarts the mark, the whole ring is this process's
    other = {"boot": "other-boot", "seq": 2, "recent": [{"seq": 1, "n": 9, "ms": 5.0},
                                                        {"seq": 2, "n": 8, "ms": 4.0}]}
    new4, mark4, _ = pc.cost_since(other, mark3)
    assert [r["n"] for r in new4] == [9, 8] and mark4 == ("other-boot", 2)


# ---------------------------------------------------------------- the fit
@pytest.mark.parametrize("name,a_ms,b_ms", [
    ("nextflash_z30w_park", 1900.0, 1.352),       # offline fit of the same rows
    ("qwen27b_row_authority", 227.0, 0.611),
])
def test_fit_recovers_each_models_line_from_its_own_boot(name, a_ms, b_ms):
    rows, _, _ = _load(name)
    line, why = pp.fit_cost_line(rows, min_tokens=64, min_samples=8)
    assert why == "3p"
    assert line["a_ms"] == pytest.approx(a_ms, rel=0.02)
    assert line["b_ms"] == pytest.approx(b_ms, rel=0.02)
    assert line["c_ms"] > 0.0                      # deeper prefix, dearer token
    assert line["n_lo"] >= 64


def test_fit_refuses_by_name_and_drops_a_negative_depth_term():
    assert pp.fit_cost_line([(100, 0, 2000.0)] * 3, min_tokens=64, min_samples=8)[0] is None
    line, why = pp.fit_cost_line([(1000, 0, 3000.0)] * 10, min_tokens=64, min_samples=8)
    assert line is None and why.startswith("no-spread")
    # all at one depth: the depth term cannot be separated -> 2 parameters
    rows = [(n, 50000, 1700.0 + 1.4 * n) for n in (100, 400, 900, 1600, 2500, 3600, 4000, 800)]
    line, why = pp.fit_cost_line(rows, min_tokens=64, min_samples=8)
    assert why == "2p(no-depth-spread)" and line["c_ms"] == 0.0
    assert line["b_ms"] == pytest.approx(1.4) and line["a_ms"] == pytest.approx(1700.0)
    # tail extends below min_tokens are not the line (46-106 ms on z30w)
    line2, _ = pp.fit_cost_line(rows + [(4, 80000, 46.0)] * 20, min_tokens=64, min_samples=8)
    assert line2["n"] == len(rows)


# ---------------------------------------------------------------- the solve
def test_solve_amortises_the_round_trip_over_the_backlog():
    rows, drains, r_p = _load("nextflash_z30w_park")
    line, _ = pp.fit_cost_line(rows, min_tokens=64, min_samples=8)
    k = statistics.mean(d[0] for d in drains if d[0] > 0)
    x1, _ = pp.solve_x_cost_line(price_s=7.866, k=1.0, line=line, r_p=r_p, prefix_tokens=55000)
    xk, _ = pp.solve_x_cost_line(price_s=7.866, k=k, line=line, r_p=r_p, prefix_tokens=55000)
    assert xk < x1                                   # shared round trip -> lower X
    # hand arithmetic of the model: (1000*price/k - a) / (b + c*p - 1000/r_P)
    marg = line["b_ms"] + line["c_ms"] * 55.0 - 1000.0 / r_p
    assert xk == pytest.approx((1000.0 * 7.866 / k - line["a_ms"]) / marg)
    # NF under the measured inputs lands BELOW the old hard floor of 4096
    assert 2000 < xk < 4096
    # deeper prefix: dearer D token -> the break-even comes sooner
    xdeep, _ = pp.solve_x_cost_line(price_s=7.866, k=k, line=line, r_p=r_p, prefix_tokens=240000)
    assert xdeep < xk
    # D's marginal token not dearer than P's: no crossing. D is the cheaper
    # route for every size while its fixed cost stays under the price share
    # (inf -> the ceiling), P for every size otherwise (0 -> the floor)
    x, why = pp.solve_x_cost_line(price_s=7.866, k=1.0, line=line, r_p=500.0, prefix_tokens=0)
    assert x == float("inf") and why.startswith("d-never-dearer")
    x, why = pp.solve_x_cost_line(price_s=7.866, k=6.0, line=line, r_p=500.0, prefix_tokens=0)
    assert x == 0.0 and why.startswith("p-always-cheaper")
    # a price share below D's fixed cost: 0 -- the caller's floor decides
    x, why = pp.solve_x_cost_line(price_s=1.0, k=6.0, line=line, r_p=r_p, prefix_tokens=0)
    assert x == 0.0 and why == "ok"


def test_record_round_trip_per_form(tmp_path):
    line = {"a_ms": 1900.0, "b_ms": 1.35, "c_ms": 0.00088, "n": 121, "n_lo": 64,
            "n_hi": 4096, "prefix_med": 55000}
    path = tmp_path / "rec.json"
    recs = [pp.x_cost_line_record(form_key="NF|a", line=line, k=1.6, boot_tag="z30w",
                                  commit="c", at="2026-09-29 09:00:00,000"),
            pp.x_cost_line_record(form_key="27B|d", line=dict(line, b_ms=0.61), k=1.5,
                                  boot_tag="row", commit="c", at="2026-09-29 00:40:00,000")]
    path.write_text(json.dumps({"samples": recs}))
    assert pp.read_x_cost_line(str(path), "NF|a")["b_ms"] == 1.35
    assert pp.read_x_cost_line(str(path), "27B|d")["b_ms"] == 0.61
    assert pp.read_x_cost_line(str(path), "other|x") is None
    assert pp.read_x_cost_line(str(tmp_path / "missing.json"), "NF|a") is None


# ---------------------------------------------------------------- the front
def _flip_log(dp_ms, pd_ms, pairs=3):
    log = [{"sleep": "D", "wake": "P", "flip_ms": 24600.0}]
    for _ in range(pairs):
        log += [{"sleep": "D", "wake": "P", "flip_ms": dp_ms}, {"sleep": "P", "wake": "D", "flip_ms": pd_ms}]
    return log


def _front(name, **attrs):
    from sglang.srt.weg2 import front as F

    rows, drains, r_p = _load(name)
    ns = types.SimpleNamespace(
        tp_prefill_max_tokens=4096, flip_min_work_tokens=4096, _x_min_work_follows=True,
        x_ceiling_tokens=12288, x_floor_tokens=4096, counters=collections.Counter(),
        _x_samples={"r_d": collections.deque(maxlen=32), "r_p": collections.deque([r_p], maxlen=32),
                    "flip_s": collections.deque([2.5], maxlen=32)},
        _x_last_missing=[], _x_cost_last_missing=[], _x_r_d_src="none yet",
        _x_seed_note="launcher solve X=4096", X_SAMPLE_WINDOW=32,
        _d_cost_rows=collections.deque(maxlen=64), _d_cost_mark=None,
        _p_cost_rows=collections.deque(maxlen=64), _p_cost_mark=None, _x_cost_seed_p=None,
        _p_phase_k=collections.deque([d[0] for d in drains if d[0] > 0], maxlen=32),
        _x_cost_seed=None, _x_cost_written={"D": 0, "P": 0}, _park_rt_seed=None,
        _park_form_key="", measured_record="", p_phase_max_requests=6,
        flip_log=_flip_log(2525.0, 2218.0), _resume_ms_log=[11800.0, 3100.0, 3123.0, 3150.0],
    )
    ns._park_warm_legs_ms = lambda: F.Front._park_warm_legs_ms(ns)
    ns._x_cost_inputs = lambda: F.Front._x_cost_inputs(ns)
    ns._resolve_x_cost_line = lambda: F.Front._resolve_x_cost_line(ns)
    ns._note_x_cost_line = lambda *a: F.Front._note_x_cost_line(ns, *a)
    ns._x_cost_line_of = lambda *a: F.Front._x_cost_line_of(ns, *a)
    ns._note_cost_rows = lambda b, side: F.Front._note_cost_rows(ns, b, side)
    ns.x_flip_s_provenance = lambda: F.Front.x_flip_s_provenance(ns)
    for k, v in attrs.items():
        setattr(ns, k, v)
    return F.Front, ns, rows


def _feed(ns, rows, every=20, side="D"):
    """One group's forwards in log order, the front reading the ring every
    ``every`` forwards (D: its leg-2 reads under load; P: one read per drain)."""
    for n, c, ms in rows:
        pc.note_batch_cost(n, c, ms)
        if pc.cost_snapshot()["seq"] % every == 0:
            ns._note_cost_rows(pc.cost_snapshot(), side)
    ns._note_cost_rows(pc.cost_snapshot(), side)


def test_red_state_solo_probe_under_load_leaves_x_at_4096(caplog):
    """The z30w/09290020 state: no solo r_D sample ever arrives, the old
    re-solve names it and X stays 4096 -- whatever D's forwards measured."""
    F, ns, rows = _front("nextflash_z30w_park")
    for n, c, ms in rows:
        pc.note_batch_cost(n, c, ms)
    ns._note_cost_rows(pc.cost_snapshot(), "D")     # the ring IS read ...
    with envs.SGLANG_WEG2_ENABLE_X_COST_LINE.override(False), caplog.at_level("WARNING"):
        assert F.resolve_x_live(ns) is None          # ... but the old solve ignores it
    assert ns.tp_prefill_max_tokens == 4096
    assert any("X NO-SOLVE" in m and "no r_d" in m for m in caplog.messages)


@pytest.mark.parametrize("name", ["nextflash_z30w_park", "qwen27b_row_authority"])
def test_green_x_moves_from_ds_forwards_under_load(name, caplog):
    F, ns, rows = _front(name)
    _feed(ns, rows)
    assert ns.counters["x_cost_rows"] == len(rows) and ns.counters["x_cost_rows_lost"] == 0
    with envs.SGLANG_WEG2_ENABLE_X_COST_LINE.override(True), caplog.at_level("INFO"):
        x = F.resolve_x_live(ns)
    assert x is not None and x != 4096
    assert ns.tp_prefill_max_tokens == x and ns.flip_min_work_tokens == x
    msg = next(m for m in caplog.messages if "X COST-LINE RE-SOLVE" in m)
    assert "[live:" in msg and "ski-live" in msg and "k=" in msg and "X* by depth" in msg
    if name == "nextflash_z30w_park":
        assert 2000 < x < 4096                       # the old floor would have held it at 4096
    else:
        assert x == 12288                            # 27B without a P line: drain r_P -> ceiling


def test_front_seeds_from_the_record_and_names_missing_inputs(caplog):
    F, ns, _ = _front("nextflash_z30w_park", flip_log=[], _resume_ms_log=[])
    with envs.SGLANG_WEG2_ENABLE_X_COST_LINE.override(True), caplog.at_level("WARNING"):
        assert F.resolve_x_live(ns) is None
    assert any("COST-LINE NO-SOLVE" in m and "d_cost_line and round_trip" in m for m in caplog.messages)
    ns._x_cost_seed = {"a_ms": 1900.0, "b_ms": 1.35, "c_ms": 0.00088, "n": 121, "n_lo": 64,
                       "n_hi": 4096, "prefix_med": 55000, "k": 1.6, "boot_tag": "z30w",
                       "at": "2026-09-29 09:00:00,000"}
    ns._park_rt_seed = {"round_trip_s": 7.87, "boot_tag": "z30w", "at": "2026-09-29 09:01:00,000"}
    caplog.clear()
    with envs.SGLANG_WEG2_ENABLE_X_COST_LINE.override(True), caplog.at_level("INFO"):
        x = F.resolve_x_live(ns)
    assert x is not None and 64 <= x < 4096
    msg = next(m for m in caplog.messages if "X COST-LINE RE-SOLVE" in m)
    assert "[record:z30w@" in msg and "[ski-record:z30w@" in msg
    assert ns._x_cost_written == {"D": 0, "P": 0}    # a seed is never written back as new


def test_front_writes_its_first_live_lines_per_form(tmp_path):
    F, ns, rows = _front("qwen27b_row_authority")
    path = tmp_path / "weg2_measured_record.json"
    submitted = []
    ns.measured_record, ns._park_form_key = str(path), "Qwen3.8-27B|arch=dense"
    ns.tag, ns.commit = "dkr27brow", "bb82fbcb68"
    ns._sidecar_submit = lambda fn, *a: submitted.append(fn(*a))
    _feed(ns, rows)
    with envs.SGLANG_WEG2_ENABLE_X_COST_LINE.override(True):
        for _ in range(pp.X_COST_LINE_RECORDS + 2):
            F.resolve_x_live(ns)
    assert len(submitted) == pp.X_COST_LINE_RECORDS
    got = pp.read_x_cost_line(str(path), "Qwen3.8-27B|arch=dense")
    assert got["b_ms"] == pytest.approx(0.611, rel=0.02) and got["boot_tag"] == "dkr27brow"
    assert pp.read_x_cost_line(str(path), "Qwen3.8-Flash-Next|arch=moe") is None


# ---------------------------------------------------------------- P's side (r_P)
def _load_p(name):
    doc = json.loads((FIX / f"x_cost_{name}.json").read_text())
    return [tuple(r) for r in doc["p_pp0_prefill_forwards_n_cached_ms"]]


def test_the_cost_ring_is_published_by_both_weg2_groups_only():
    """P answers /get_server_info from PP0 and has no law-4 riegel, so the
    D-only publish gate kept P's ring dark; a stock engine stays dark."""
    assert pc.cost_published(armed=True, group="D")
    assert pc.cost_published(armed=False, group="P")      # P: no riegel, group name
    assert not pc.cost_published(armed=False, group="")
    assert not pc.cost_published(armed=False, group="  ")


@pytest.mark.parametrize("name,a_ms,b_ms", [
    ("nextflash_z30w_park", 1498.0, 0.117),   # offline fit of the same PP0 rows
    ("qwen27b_row_authority", 93.0, 0.039),
])
def test_p_line_from_pp0_is_compute_honest(name, a_ms, b_ms):
    line, why = pp.fit_cost_line(_load_p(name), min_tokens=64, min_samples=8)
    assert line is not None and line["c_ms"] >= 0.0
    assert line["a_ms"] == pytest.approx(a_ms, rel=0.05)
    assert line["b_ms"] == pytest.approx(b_ms, rel=0.05)
    # PP0's marginal token is far cheaper than the drain rate says (27B:
    # drain 1437 tok/s = 0.696 ms/tok; NF: 1769 = 0.565 ms/tok)
    _, _, r_p = _load(name)
    assert line["b_ms"] < 0.5 * (1000.0 / r_p)


def test_solve_with_p_line_uses_its_fixed_and_marginal_cost():
    line = {"a_ms": 227.0, "b_ms": 0.611, "c_ms": 0.00093}
    line_p = {"a_ms": 93.0, "b_ms": 0.039, "c_ms": 0.00088}
    x, why = pp.solve_x_cost_line(price_s=6.94, k=1.53, line=line, r_p=1437.0,
                                  prefix_tokens=92000, line_p=line_p)
    per_req = 1000.0 * 6.94 / 1.53
    denom = (0.611 + 0.00093 * 92) - (0.039 + 0.00088 * 92)
    assert why == "ok" and x == pytest.approx((per_req + 93.0 - 227.0) / denom)
    # the drain rate alone (the old answer): the crossing lies far beyond the ceiling
    x_drain, why_drain = pp.solve_x_cost_line(price_s=6.94, k=1.53, line=line, r_p=1437.0,
                                              prefix_tokens=92000)
    assert x_drain > 12288
    assert x < 12288 < x_drain


def test_record_sides_never_mix(tmp_path):
    line = {"a_ms": 93.0, "b_ms": 0.039, "c_ms": 0.0009, "n": 40, "n_lo": 64, "n_hi": 16384,
            "prefix_med": 0}
    path = tmp_path / "rec.json"
    rec_p = pp.x_cost_line_record(form_key="27B|d", line=line, k=1.5, boot_tag="row",
                                  commit="c", at="2026-09-29 00:40:00,000", side="P")
    legacy_d = pp.x_cost_line_record(form_key="27B|d", line=dict(line, b_ms=0.61), k=1.5,
                                     boot_tag="row", commit="c", at="2026-09-29 00:30:00,000")
    legacy_d.pop("side")                     # an entry written before sides: D's
    path.write_text(json.dumps({"samples": [rec_p, legacy_d]}))
    assert pp.read_x_cost_line(str(path), "27B|d", "P")["b_ms"] == 0.039
    assert pp.read_x_cost_line(str(path), "27B|d")["b_ms"] == 0.61
    assert pp.read_x_cost_line(str(path), "NF|a", "P") is None


def test_green_27b_x_below_the_ceiling_with_ps_own_line(caplog):
    """27B 09290020: with the drain r_P the solve had no crossing and X sat at
    the ceiling 12288; with P's PP0 line it is a real break-even."""
    F, ns, rows = _front("qwen27b_row_authority")
    _feed(ns, rows)
    _feed(ns, _load_p("qwen27b_row_authority"), every=10, side="P")
    assert ns.counters["x_cost_p_rows"] > 0
    with envs.SGLANG_WEG2_ENABLE_X_COST_LINE.override(True), caplog.at_level("INFO"):
        x = F.resolve_x_live(ns)
    msg = next(m for m in caplog.messages if "X COST-LINE RE-SOLVE" in m)
    assert "P line a=" in msg and "[live:" in msg.split("P line")[1]
    assert x is not None and 4096 < x < 12288


def test_green_nf_x_with_ps_own_line(caplog):
    F, ns, rows = _front("nextflash_z30w_park")
    _feed(ns, rows)
    _feed(ns, _load_p("nextflash_z30w_park"), every=10, side="P")
    with envs.SGLANG_WEG2_ENABLE_X_COST_LINE.override(True), caplog.at_level("INFO"):
        x = F.resolve_x_live(ns)
    assert x is not None and 1000 < x < 12288
    assert any("P line a=" in m for m in caplog.messages)


def test_p_ring_read_off_the_flip_path_counts_a_failure_never_a_row():
    import asyncio

    from sglang.srt.weg2 import front as F

    class _Resp:
        def __init__(self, status, body):
            self.status, self._body = status, body

        async def json(self):
            return self._body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        def __init__(self, resp=None, exc=None):
            self.resp, self.exc, self.urls = resp, exc, []

        def get(self, url, **kw):
            self.urls.append(url)
            if self.exc is not None:
                raise self.exc
            return self.resp

    _, ns, _ = _front("qwen27b_row_authority")
    ns.groups = {"P": types.SimpleNamespace(url="http://p:1")}
    pc.note_batch_cost(1440, 0, 153.0)
    pc.note_batch_cost(3120, 0, 215.0)
    ns.session = _Session(_Resp(200, {"internal_states": [{pc.COST_STATE_KEY: pc.cost_snapshot()}]}))
    asyncio.run(F.Front._read_p_cost(ns))
    assert list(ns._p_cost_rows) == [(1440, 0, 153.0), (3120, 0, 215.0)]
    assert ns.session.urls == ["http://p:1/get_server_info"] and not ns._d_cost_rows
    ns.session = _Session(exc=OSError("P asleep"))
    asyncio.run(F.Front._read_p_cost(ns))
    assert ns.counters["x_cost_p_read_errors"] == 1 and len(ns._p_cost_rows) == 2
