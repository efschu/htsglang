"""PR2: the big D prefills (host plan) pair THEIR fetch ms with THEIR rows.

The paired miss record of 363c173670 opened a window around each timed
prefill forward and filled it from the pool syncs inside it, against the
forward's ``pool.fetch`` spans. Neither half belonged to the forward:

* the host-planned eager forward (``_run_eager_host_plan`` -> run_waves, the
  big D prefills; y3w: 2144-token prefills) fetches through ``_fetch``, which
  had no clock span -- no numerator, the pair was refused, n stayed 0;
* its per-layer sync reports the DEVICE counters since the previous sync,
  i.e. the graphed decode steps before it, never its own rows;
* a device-planned eager layer (D-Mini-Extend) has a ``pool.fetch`` span but
  never syncs, so a forward mixing both paths paired one layer set's fetch ms
  with another's earlier decode misses.

PR2: each ``_fetch`` inside the window adds the rows of its ``fetch_plan`` --
the rows the waves really reload, known on the host -- and runs its copies
under one ``pool.host_fetch`` span (device events, no host read). The line
sums that family only; the pair holds when the harvested span count equals
the window's fetch count. Mixed windows are counted consistently: the host-
plan layers in both halves, the device-step layers in neither (their rows
are device counters, readable only by a host sync).
Hermetic: a CPU cache (H12 fixture), a stand-in clock. RED on 363c173670.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import contextlib  # noqa: E402
import types  # noqa: E402
from collections import namedtuple  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.layers.moe import expert_offload as eo  # noqa: E402
from flliper.srt.layers.moe import expert_pool_device as ep  # noqa: E402
from flliper.srt.layers.moe import pool_miss_cost as pmc  # noqa: E402
from flliper.srt.layers.moe.topk import StandardTopKOutput  # noqa: E402
from flliper.srt.managers.scheduler_components import metrics_reporter as mr  # noqa: E402
from flliper.srt.utils import collective_clock as cc  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(__file__)

E, R, C, S, W = 12, 2, 4, 1, 3
DispatchOutput = namedtuple("DispatchOutput", "hidden_states hidden_states_scale topk_output")
CombineOutput = namedtuple("CombineOutput", "hidden_states")
# 8 distinct spill experts (2..9), expert-major: each fetched once
ROUTES = [[2, 3, 0], [4, 5, 1], [2, 6, 0], [3, 7, 1], [8, 9, 2]]
SPAN_MS = 0.5
#: the family the host plan's fetch spans carry (pool_miss_cost.HOST_FETCH_FAMILY)
HOST_FETCH = "pool.host_fetch"


class _Clock:
    """Stand-in collective clock: armed, records each span's family."""

    def __init__(self):
        self.armed = True
        self.spans = []

    @contextlib.contextmanager
    def span(self, label=None):
        self.spans.append(label)
        yield

    def families(self, extra=None):
        fams = dict(extra or {})
        n = sum(1 for s in self.spans if s == HOST_FETCH)
        if n:
            fams[HOST_FETCH] = types.SimpleNamespace(total_ms=SPAN_MS * n, count=n)
        return fams


@pytest.fixture
def armed(tmp_path, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(cc, "collective_clock", lambda: clock)
    with envs.FLLIPER_PDFLIP_OWNED_MISS_RECORD.override(str(tmp_path)):
        pmc._reset_for_test()
        yield clock
    pmc._reset_for_test()


def _pool_cache(monkeypatch):
    monkeypatch.setenv("FLLIPER_MOE_SCRATCH_SLOTS", str(C))
    monkeypatch.setitem(eo._PARTIALS_MODE, "mode", "stream")
    layer = types.SimpleNamespace(
        num_local_experts=E, layer_id=23,
        moe_runner_config=types.SimpleNamespace(routed_scaling_factor=1.0),
    )
    with envs.FLLIPER_OPT_MOE_POOL_EAGER_DEVICE_STEP.override(False):
        cache = eo.MoEExpertOffloadCache(layer, R / E)
    spill = torch.zeros((E - R, W), dtype=torch.float32)
    for row in range(E - R):
        spill[row].fill_(R + row)
    bank = torch.full((R + C, W), -1.0, dtype=torch.float32)
    for e in range(R):
        bank[e].fill_(e)
    cache._pinned = {"w13": spill}
    cache._resident = {"w13": bank}
    cache._installed = True
    hot_slot_of, host_row = cache._pool_layout()
    cache._pool_tables = ep.allocate_pool_tables("cpu", E, R + C, R, S, hot_slot_of, host_row)
    cache._pool_buffers = ep.allocate_step_buffers("cpu", E, 8)
    cache._pool_ready = True
    # graphed decode steps before this prefill: their counters sit in the
    # tables and surface at this forward's own sync
    cache._pool_tables.forwards.fill_(1)
    cache._pool_tables.misses_total.fill_(99)
    return cache


def _host_plan_forward(cache, routes=ROUTES):
    ids = torch.tensor(routes, dtype=torch.int32)
    bank = cache._resident["w13"]
    topk = StandardTopKOutput(topk_weights=torch.ones(ids.shape), topk_ids=ids, router_logits=None)

    def _apply(sub):
        rows = sub.topk_output.topk_ids.long()
        per_pair = bank[rows.clamp(min=0)][..., 0] * sub.topk_output.topk_weights
        return CombineOutput(hidden_states=per_pair.sum(dim=-1, keepdim=True).expand(-1, W).contiguous())

    return cache._run_eager_host_plan(DispatchOutput(torch.zeros(ids.shape[0], W), None, topk), _apply)


def _timed(cache, routes=ROUTES):
    """One timed prefill forward: the window the prefill timer opens."""
    w: dict = {}
    pmc.open_window(w)
    try:
        _host_plan_forward(cache, routes)
    finally:
        pmc.close_window()
    return w


def _line(families, window):
    log = mr.RankPrefillLog()
    log.timer = types.SimpleNamespace(_report=lambda: None)
    log.record(2144, 0, timed=True)
    log._durations.append((4.0, 0.4, families, window))
    log.flush()


def test_a_host_plan_forward_pairs_its_fetch_ms_with_the_rows_it_reloaded(armed, monkeypatch):
    """RED on 363c173670: the host plan's fetches carried no span (0 spans),
    its window held the 99 foreign decode misses of the sync."""
    cache = _pool_cache(monkeypatch)
    w = _timed(cache)
    n = armed.spans.count(HOST_FETCH)
    assert n >= 1 and w["fetches"] == n
    assert w["rows"] == 8  # the 8 spill experts the waves reloaded, not the sync's 99
    _line(armed.families(), w)
    rec = pmc.rank_record(rank=1, group="D", reason="t", model="m")
    assert rec is not None and rec["pairing"] == pmc.PAIRING and rec["rounds"] == 1
    assert rec["miss_rows"] == 8 and rec["ms_per_row"] == pytest.approx(SPAN_MS * n / 8)
    # a harvested span count that is not the window's (another forward's
    # spans on the same line) is refused, never mixed in
    w2 = _timed(cache)
    _line(armed.families({HOST_FETCH: types.SimpleNamespace(total_ms=9.0, count=99)}), w2)
    rec = pmc.rank_record(rank=1, group="D", reason="t", model="m")
    assert rec["rounds"] == 1 and rec["paired_refused"] == 1


def test_a_mixed_window_counts_the_host_plan_layers_only(armed, monkeypatch):
    """Device-step layers of the same forward (``pool.fetch`` 8 ms over 48
    spans) stay out of BOTH halves: their rows are device counters that only
    a host sync could read. RED on 363c173670: it paired those 8 ms with the
    sync's 99 foreign decode misses."""
    cache = _pool_cache(monkeypatch)
    w = _timed(cache)
    device_step = {"pool.fetch": types.SimpleNamespace(total_ms=8.0, count=48)}
    fams = armed.families(device_step)
    _line(fams, w)
    rec = pmc.rank_record(rank=1, group="D", reason="t", model="m")
    n = armed.spans.count(HOST_FETCH)
    assert rec is not None and rec["miss_rows"] == 8
    assert rec["fetch_ms"] == pytest.approx(SPAN_MS * n)


def test_a_host_plan_forward_without_misses_pairs_nothing(armed, monkeypatch):
    """Every routed expert resident: the waves reload no row. With device-step
    layers beside it the forward still yields no pair. RED on 363c173670: the
    sync's foreign 99 misses and the device-step ms made one."""
    cache = _pool_cache(monkeypatch)
    w = _timed(cache, routes=[[0, 1, 0], [1, 0, 1]])
    assert armed.spans.count(HOST_FETCH) == 0 and w["rows"] == 0
    _line(armed.families({"pool.fetch": types.SimpleNamespace(total_ms=8.0, count=48)}), w)
    assert pmc.rank_record(rank=1, group="D", reason="t", model="m") is None
