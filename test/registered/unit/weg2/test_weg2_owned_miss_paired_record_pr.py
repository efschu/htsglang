"""PR: the owned solve's miss cost from PAIRED prefill forwards, gated by count.

Metal (NF y3u 00:27:27, y3v 00:58:20, y3w 01:30:34, y3y 01:54:17, y3z
02:33:18): every ``D-EIGENTUM (#239 S3f)`` line priced the D ranks' expert
misses with ``Fehlgriff-Kosten Saat UNMEASURED``. Two reasons stacked:
``SGLANG_WEG2_OWNED_MISS_RECORD`` was never set for D (0x OWNED-MISS-COST),
and the record's numerator came from split decode rounds only -- graphed
production rounds have no split -- while its denominator summed the misses of
EVERY pool sync, graphed decode steps included.

Now: a window around each timed prefill forward (the per-rank prefill
timer's bracket) collects that forward's pool-sync misses; the prefill line
adds the forward's own pool.fetch ms only for a PAIRED, split-known duration
whose window is clean (every layer reported exactly one forward since its
last sync). The planner reads RECORD from K paired forwards per rank; fewer
keep the seed and the line names the count. The launcher turns the record on.
RED on 06c61a7988 (no windows, no count gate, no default), GREEN with PR.
"""
from __future__ import annotations

import contextlib
import os
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.moe import pool_miss_cost as pmc  # noqa: E402
from sglang.srt.managers.scheduler_components import metrics_reporter as mr  # noqa: E402
from sglang.srt.planner import expert_residency as er  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

LAYERS = 4


@pytest.fixture
def armed(tmp_path):
    with envs.SGLANG_WEG2_OWNED_MISS_RECORD.override(str(tmp_path)):
        pmc._reset_for_test()
        yield tmp_path
    pmc._reset_for_test()


class _Clock:
    armed = True

    @contextlib.contextmanager
    def span(self, label=None):
        yield


def _forward(rows_per_layer):
    """One timed prefill forward (PR2): the window around its host-plan
    fetches, one per layer."""
    holder: dict = {}
    pmc.open_window(holder)
    for _ in range(LAYERS):
        with pmc.host_fetch_span(rows_per_layer, _Clock()):
            pass
    pmc.close_window()
    return holder


def _fams(ms, count):
    return {pmc.HOST_FETCH_FAMILY: types.SimpleNamespace(total_ms=ms, count=count),
            "tp.all_reduce": types.SimpleNamespace(total_ms=5.0, count=96)}


def _log():
    log = mr.RankPrefillLog()
    log.timer = types.SimpleNamespace(_report=lambda: None)
    return log


def test_a_paired_prefill_forward_writes_its_own_cost(armed):
    """RED: no window exists (AttributeError) -- the record paired decode fetch
    with every sync's misses. GREEN: 12 ms over this forward's 4 x 10 rows."""
    w = _forward(10)
    log = _log()
    log.record(2144, 0, timed=True)
    log._on_duration(4.0, None)  # the clock slot: none in this hermetic run
    log._durations[-1] = (4.0, 0.4, _fams(12.0, LAYERS), w)
    log.flush()
    rec = pmc.rank_record(rank=1, group="D", reason="test", model="m")
    assert rec is not None and rec["pairing"] == pmc.PAIRING
    assert rec["miss_rows"] == 4 * 10 and rec["rounds"] == 1
    assert abs(rec["ms_per_row"] - 12.0 / 40) < 1e-9


def test_a_foreign_forward_is_dropped(armed):
    """A duration whose window covered no pool sync (a forward the window did
    not bracket, e.g. a draft extend on the same timer) pairs nothing; and a
    refused pairing (#691 drift) drops every queued duration untimed."""
    assert pmc.note_paired(fetch_ms=12.0, fetch_count=48, window={}) is False
    log = _log()
    for _ in range(mr.RankPrefillLog.MAX_PAIR_SKEW + 2):
        log.record(8, 0, timed=True)
    log._durations.append((0.1, 0.01, _fams(3.0, LAYERS), _forward(5)))
    log.flush()  # skew > limit: refused, drained untimed
    assert log.pairing_refused
    assert pmc.rank_record(rank=1, group="D", reason="t", model="m") is None


def _rec(rank, n, fetch=40.0, rows=100, pairing="prefill_hostfetch_v3"):
    return {"kind": er.OWNED_MISS_RANK_KIND, "rank": rank, "time_unix": time.time(),
            "fetch_ms": fetch, "miss_rows": rows, "rounds": n, "model": None,
            "pairing": pairing}


def test_fewer_than_k_paired_forwards_keep_the_seed_and_name_the_count():
    k = er.OWNED_MISS_MIN_PAIRED_FORWARDS
    recs = [_rec(0, k), _rec(1, k - 1), _rec(2, k)]
    ms, tier, src = er.resolve_owned_miss_ms(rank_records=recs, host=0)
    assert tier == er.OWNED_MISS_UNMEASURED and ms == er.OWNED_MISS_MS_PER_ROW_SEED
    assert "RECORD zu jung" in src and "1: %d" % (k - 1) in src


def test_k_paired_forwards_on_every_rank_are_the_record():
    k = er.OWNED_MISS_MIN_PAIRED_FORWARDS
    recs = [_rec(0, k, 20.0, 100), _rec(1, k, 60.0, 100), _rec(2, k, 40.0, 100)]
    ms, tier, _src = er.resolve_owned_miss_ms(rank_records=recs, host=0)
    assert tier == er.OWNED_MISS_RECORD
    assert ms == (0.2, 0.5)


def test_an_unpaired_old_record_never_counts():
    k = er.OWNED_MISS_MIN_PAIRED_FORWARDS
    recs = [_rec(0, k, pairing=None), _rec(1, k, pairing=None)]
    _ms, tier, _src = er.resolve_owned_miss_ms(rank_records=recs, host=0)
    assert tier == er.OWNED_MISS_UNMEASURED


def test_the_launcher_turns_the_record_on_unless_env_d_names_it():
    from sglang.srt.weg2 import launcher as L

    ns = types.SimpleNamespace(env_d="")
    assert L.apply_owned_miss_record_default(ns) is not None
    assert L.parse_group_env(ns.env_d)[L.OWNED_MISS_RECORD_ENV] == L.owned_miss_record_root()
    ns2 = types.SimpleNamespace(env_d="%s=" % L.OWNED_MISS_RECORD_ENV)
    assert L.apply_owned_miss_record_default(ns2) is None  # stated (here: off) wins


def test_the_miss_window_timer_brackets_the_same_interval(armed):
    """The installed timer carries the window in the interval's metadata and
    closes it before the interval ends (MissWindowTimer)."""
    seen = []

    class _Base:
        @contextlib.contextmanager
        def wrap(self, metadata):
            yield
            seen.append(metadata)

    orig = mr.SplitDeviceTimer.wrap
    mr.SplitDeviceTimer.wrap = _Base.wrap
    try:
        t = object.__new__(mr.MissWindowTimer)
        with t.wrap({"category": "extend"}):
            with pmc.host_fetch_span(7, _Clock()):
                pass
    finally:
        mr.SplitDeviceTimer.wrap = orig
    assert seen and seen[0]["miss_window"]["rows"] == 7 and pmc._WIN is None
