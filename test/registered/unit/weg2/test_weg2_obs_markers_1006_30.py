"""nf-next-1006-30: two observation-only log markers (no behaviour change).

(1) ``#59 RESUMABLE`` (mode form-a-dcp-min / tp-min) names this rank's OWN
    pre-reduce depth: TP0's line gains ``local_depth=`` (appended after
    ``mode=``), every other rank logs ``#59 RESUMABLE-LOCAL ... local_depth=
    ... min=``. nf-next-1006-27 could not tell which rank dragged the MIN to a
    short depth (class B, 5 of 780). No collective added, return values equal.
(2) the FLOOR-CHECK census line of ``d_seat_vram.runtime_tick`` appends this
    rank's ``used=`` and ``chunk_live=`` (nf-next-1006-28 Fix A tripwire).

RED on 369f31e2e2: neither field exists.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import contextlib  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import types  # noqa: E402
from array import array  # noqa: E402

import torch  # noqa: E402
from unittest import mock  # noqa: E402

from sglang.srt import rank_role  # noqa: E402
from sglang.srt.mem_cache.base_prefix_cache import MatchResult  # noqa: E402
from test_weg2_d_mem_sched_0929 import _req as _dreq, floor_env, tick_env  # noqa: E402,F401

GRID = 25600
#: gcd-reduced token cut (test_weg2_form_a_dcp_s3d_239)
BOUNDS = {0: (64, 0, 0), 1: (64, 0, 46), 2: (64, 46, 64)}
RES_LOGGER = "sglang.srt.managers.weg2_resumable_depth"


def _node(name, anchor=True):
    host = torch.tensor([3]) if anchor else None
    return types.SimpleNamespace(
        name=name,
        component_data=[None, None, types.SimpleNamespace(value=None, host_value=host)],
    )


class _Tree:
    def __init__(self, reach, anchor):
        self.reach, self.anchor = reach, anchor
        self.root_node = _node("root", anchor=False)
        self.cache_controller = None
        self.is_eagle = False
        self.supports_mamba = lambda: True
        self.swa_reprefill_tail_tokens = lambda: 0

    def match_prefix(self, params):
        d = min(self.reach, len(params.key))
        node = _node(f"n{d}", anchor=self.anchor)
        return MatchResult(device_indices=torch.arange(d, dtype=torch.int64),
                           last_device_node=node, last_host_node=node, best_match_node=node)


def _req(rid="weg2-1006-30", branching=None, last_track=None):
    from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats

    return types.SimpleNamespace(
        rid=rid, origin_input_ids=array("q", range(GRID + 11)), output_ids=list(range(300)),
        extra_key=None, positional_embed_overrides=None, cached_tokens=GRID,
        time_stats=SchedulerReqTimeStats(), finished=lambda: True, finished_output=False,
        mamba_branching_seqlen=branching, mamba_last_track_seqlen=last_track,
        _compute_max_prefix_len=lambda k: max(k - 1, 0),
    )


def _ps(rank):
    return types.SimpleNamespace(tp_size=3, attn_dp_size=3, pp_size=1, attn_tp_rank=rank)


@contextlib.contextmanager
def _as_rank(rank):
    prev = (rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK)
    rank_role.set_form_a_role_plan(rank_role.RankRolePlan(("host", "worker", "worker")), rank)
    try:
        with mock.patch("sglang.srt.distributed.utils.uneven_dcp_owner_bounds",
                        lambda: BOUNDS[rank]):
            yield
    finally:
        rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK = prev


def _run_group(caplog):
    """Three ranks of Form A x token cut stamp one finished request; the
    group MIN is 16384 (rank 1 lost rows of the prefix)."""
    from sglang.srt.managers import weg2_resumable_depth as m

    trees = {0: _Tree(GRID, True), 1: _Tree(16384, False), 2: _Tree(GRID, False)}
    own = {0: GRID, 1: 16384, 2: GRID}
    out, reduce_calls = {}, []
    caplog.set_level(logging.INFO, logger=RES_LOGGER)
    # nf-next-1006-29: the host keeps the branching point, a follow worker lost it
    tracks = {0: (23232, 23232), 1: (None, 24128), 2: (None, 24128)}
    for rank in (0, 1, 2):
        reqs = [_req(branching=tracks[rank][0], last_track=tracks[rank][1])]

        def reduce(v, rank=rank):
            reduce_calls.append((rank, list(v)))
            return [min(own.values())]

        with _as_rank(rank):
            mode = m.stamp_finished(trees[rank], reqs, _ps(rank), reduce_min=reduce)
        out[rank] = (mode, reqs[0].time_stats.weg2_resumable_depth)
    return out, reduce_calls


def test_resumable_line_names_each_ranks_own_depth_before_the_min(caplog):
    _run_group(caplog)
    lines = [r.getMessage() for r in caplog.records if "#59 RESUMABLE" in r.getMessage()]
    tp0 = [x for x in lines if x.startswith("#59 RESUMABLE rid=")]
    local = [x for x in lines if x.startswith("#59 RESUMABLE-LOCAL")]
    assert len(tp0) == 1 and len(local) == 2, lines
    assert re.search(
        r"depth=16384 seq=\d+ cached_tokens=\d+ mode=form-a-dcp-min local_depth=%d "
        r"branching=23232 last_track=23232$" % GRID,
        tp0[0]), tp0[0]
    got = sorted(int(re.search(r"local_depth=(\d+) min=(\d+)", x).group(1)) for x in local)
    assert got == [16384, GRID]
    assert all(re.search(r"min=16384 ", x) for x in local)
    assert all(x.endswith("branching=- last_track=24128") for x in local), local


def test_resumable_old_readers_still_match_and_values_unchanged(caplog):
    out, reduce_calls = _run_group(caplog)
    # the readers of nf-next-1006-25/27 (prefix up to cached_tokens)
    rx = re.compile(r"#59 RESUMABLE rid=(\S+) depth=(\d+) seq=(\d+) cached_tokens=(\d+)")
    tp0 = [r.getMessage() for r in caplog.records if r.getMessage().startswith("#59 RESUMABLE rid=")]
    m = rx.search(tp0[0])
    assert m and int(m.group(2)) == 16384
    # return values / stamp unchanged: every rank stamps the group MIN, one reduce per rank
    assert all(v == ("form-a-dcp-min", 16384) for v in out.values())
    assert [c[0] for c in reduce_calls] == [0, 1, 2]
    assert all(len(c[1]) == 1 for c in reduce_calls)


def test_group_depths_signature_and_result_unchanged():
    from sglang.srt.managers import weg2_resumable_depth as m

    with _as_rank(1):
        mode, depths = m.group_depths(_Tree(16384, False), [_req()], _ps(1),
                                      reduce_min=lambda v: list(v))
    assert (mode, depths) == (m.MODE_DCP_MIN, [16384])


def _floor_lines(caplog):
    return [r.getMessage() for r in caplog.records if "FLOOR-CHECK" in r.getMessage()]


def test_floor_check_line_carries_used_and_chunk_live(floor_env, caplog):
    dsv, sched, caps, floor = floor_env
    caplog.set_level(logging.INFO)
    sched.running_batch.reqs = [_dreq("seat", 1000, 10)]
    for _ in range(dsv.FLOOR_CHECK_EVERY + 1):
        dsv.runtime_tick(sched)
    lines = _floor_lines(caplog)
    assert lines, "no FLOOR-CHECK line in %d ticks" % (dsv.FLOOR_CHECK_EVERY + 1)
    line = lines[0]
    # old fields in their old order, new ones appended after recheck_max
    assert re.search(
        r"ticks=\d+ floor_reads=\d+ floor_cached=\d+ room_reads=\d+ room_cached=\d+ "
        r"collectives=\d+ pending=S\S+ lifted=(yes|no) recheck_max=\d+ used=1010 chunk_live=0$",
        line), line
    # the reader of probe_1005e_caplift_chunk.py
    assert re.search(r"FLOOR-CHECK .*? pending=S(\d+|-) lifted=(yes|no)", line)


def test_floor_check_chunk_live_follows_a_live_chunked_req(floor_env, caplog):
    dsv, sched, caps, floor = floor_env
    caplog.set_level(logging.INFO)
    sched.running_batch.reqs = [_dreq("seat", 1000, 10)]
    sched.chunked_req = _dreq("chunk", 3006)
    for _ in range(dsv.FLOOR_CHECK_EVERY + 1):
        dsv.runtime_tick(sched)
    assert re.search(r"chunk_live=1$", _floor_lines(caplog)[0])
