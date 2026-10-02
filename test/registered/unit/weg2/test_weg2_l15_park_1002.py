# SPDX-License-Identifier: Apache-2.0
"""L15-16: the cap-0 rank's park plan (pure)."""

from __future__ import annotations

from sglang.srt.weg2.l15_park import ParkPiece, park_bytes, park_plan, parked_rows_on


def test_one_capped_rank_takes_everything_after_its_own_rows():
    pieces, why = park_plan(keep_rows=[700, 400, 500], caps=[0, 2000, 600])
    assert why is None
    assert pieces == [ParkPiece(src=0, dst=1, src_row=0, dst_row=400, rows=700)]
    assert parked_rows_on(pieces, 1) == 700 and park_bytes(pieces, 10) == 7000


def test_split_over_two_capped_ranks_largest_free_first():
    pieces, why = park_plan(keep_rows=[700, 400, 500], caps=[0, 900, 800])
    assert why is None
    assert pieces == [ParkPiece(0, 1, 0, 400, 500), ParkPiece(0, 2, 500, 500, 200)]


def test_refused_by_name_when_the_free_rows_do_not_suffice():
    pieces, why = park_plan(keep_rows=[700, 400, 500], caps=[0, 600, 600])
    assert pieces == [] and "needs" in why and "free" in why


def test_rank_agnostic_any_cap0_position_and_nothing_to_park():
    pieces, why = park_plan(keep_rows=[300, 50, 0], caps=[1000, 0, 400])
    assert why is None and pieces == [ParkPiece(1, 0, 0, 300, 50)]
    assert park_plan([10, 20], [100, 100]) == ([], None)


def test_capped_rank_over_its_cap_and_length_mismatch_refuse():
    assert park_plan([10, 200], [0, 100])[1] is not None
    assert park_plan([10, 20], [0, 100, 5])[1] is not None


# -- L15-16b transport over a simulated group (threads + a fake uneven a2a) --

import threading

import torch

from sglang.srt.weg2 import l15_park as P


class _World:
    """Collective semantics of all_to_all_single_v for N threads: every rank
    posts (out, inp, osp, isp); when all posted, rank i's inp block for j
    lands in rank j's out block for i."""

    def __init__(self, n):
        self.n = n
        self.barrier = threading.Barrier(n)
        self.posts = [None] * n
        self.calls = 0

    def a2a_for(self, rank):
        def a2a(out, inp, osp, isp):
            self.posts[rank] = (out, inp, list(osp), list(isp))
            self.barrier.wait()
            if rank == 0:
                for i in range(self.n):
                    o_i, in_i, osp_i, isp_i = self.posts[i]
                    off = 0
                    for j in range(self.n):
                        k = isp_i[j]
                        if k:
                            o_j, _, osp_j, _ = self.posts[j]
                            assert osp_j[i] == k
                            o_j.copy_(in_i[off:off + k])
                        off += k
                self.calls += 1
            self.barrier.wait()
        return a2a

    def gather_for(self, rank, box):
        def gather(obj):
            box[rank] = obj
            self.barrier.wait()
            out = list(box)
            self.barrier.wait()
            return out
        return gather


def _bufs(rank, rows=40, fill=None):
    out = []
    for layer in range(2):
        b = torch.zeros(rows, 4, dtype=torch.uint8)
        if fill is not None:
            for r in range(rows):
                b[r] = (fill + layer * 50 + r) % 251
        out.append(b)
    return out


def test_park_out_and_back_restores_the_cap0_rows_byte_exact():
    pieces, why = P.park_plan(keep_rows=[12, 5, 6], caps=[0, 15, 30])
    assert why is None
    w = _World(3)
    bufs = {0: _bufs(0, fill=7), 1: _bufs(1), 2: _bufs(2)}
    orig0 = [b.clone() for b in bufs[0]]

    def run(direction):
        th = [threading.Thread(target=P.run_park,
                               args=(direction, pieces, r, 3, bufs[r], w.a2a_for(r)))
              for r in range(3)]
        [t.start() for t in th]
        [t.join() for t in th]

    run("out")
    for p in pieces:
        for L in range(2):
            assert torch.equal(bufs[p.dst][L][p.dst_row:p.dst_row + p.rows],
                               orig0[L][p.src_row:p.src_row + p.rows])
    for b in bufs[0]:
        b.zero_()                       # the kv pause: TP0's pool is gone
    run("back")
    for L in range(2):
        assert torch.equal(bufs[0][L][:12], orig0[L][:12])
    assert w.calls == 2 * len(pieces) * 2


def test_bounds_refusal_before_any_collective():
    pieces = [P.ParkPiece(0, 1, 0, 38, 5)]
    assert P.bounds_refusal(pieces, 1, _bufs(1)) is not None
    assert P.bounds_refusal(pieces, 2, _bufs(2)) is None
    assert P.bounds_refusal(pieces, 0, []) == "no KV buffers"


def test_sidecar_is_taken_once_and_agree_needs_every_rank(tmp_path):
    env = {"SGLANG_WEG2_L15_PARK_DIR": str(tmp_path)}
    P.write_sidecar(1, env, 9, [P.ParkPiece(0, 1, 0, 5, 7)])
    assert P.take_sidecar(1, env) == (9, [P.ParkPiece(0, 1, 0, 5, 7)])
    assert P.take_sidecar(1, env) is None
    assert P.agree(True, lambda v: [True, True, v]) is True
    assert P.agree(True, lambda v: [True, False, v]) is False


def test_wiring_release_before_pause_and_wake_before_the_refill():
    import inspect

    from sglang.srt.managers.scheduler_components import weight_updater as wu

    rel = inspect.getsource(wu)
    i = rel.index("_l15_pk.park_at_release(self.scheduler, os.environ, logger.info)")
    assert '"kv_cache" in tags' in rel[i - 600:i]
    j = rel.index("_l15_pk.park_back_at_wake(")
    assert j < rel.index("_l15_opt_failed = self._l15_optimistic_refill()")
    assert "group_ok=not _weg2_kv_refusal" in rel[j:j + 300]
    k = rel.index('parked = bool(getattr(self, "_l15_park_back_ok", False))')
    assert "plan = []" in rel[k:k + 120]


def _run_ranks(fn):
    out = [None] * 3
    th = [threading.Thread(target=lambda r=r: out.__setitem__(r, fn(r))) for r in range(3)]
    [t.start() for t in th]
    [t.join() for t in th]
    return out


def _entries(monkeypatch, tmp_path, manifests, bufs, caps):
    from types import SimpleNamespace

    from sglang.srt.weg2 import l15_manifest

    w = _World(3)
    box = [None] * 3
    tl = threading.local()
    monkeypatch.setattr(P, "_group_io", lambda sched: (
        sched.rank, 3, w.gather_for(sched.rank, box), w.a2a_for(sched.rank)))
    monkeypatch.setattr(P, "_kv_buffers", lambda sched: (bufs[sched.rank], None))
    monkeypatch.setattr(P, "_caps", lambda sched, pool, tp, env: list(caps))
    monkeypatch.setattr(l15_manifest, "manifest_path",
                        lambda g, r, env: str(tmp_path / ("m%d" % r)))
    monkeypatch.setattr(l15_manifest, "read",
                        lambda path: manifests[int(path[-1])])
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda *a: SimpleNamespace(synchronize=lambda: None))
    env = {"SGLANG_WEG2_L15_PARK_DIR": str(tmp_path)}
    scheds = [SimpleNamespace(rank=r) for r in range(3)]
    return w, env, scheds


def test_release_and_wake_entries_end_to_end(monkeypatch, tmp_path):
    from sglang.srt.weg2.l15_manifest import Manifest

    m = Manifest(epoch=5, pid=1, spans=(), rows_by_rank=(12, 5, 6), anchor_slots=1)
    bufs = {0: _bufs(0, fill=3), 1: _bufs(1), 2: _bufs(2)}
    orig0 = [b.clone() for b in bufs[0]]
    w, env, scheds = _entries(monkeypatch, tmp_path, [m, m, m], bufs, [0, 15, 30])
    logs = []
    sent = _run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append))
    assert sent[0] > 0 and sent[1] == 0 and sent[2] == 0
    assert sum("result=parked" in x for x in logs) == 3
    for b in bufs[0]:
        b.zero_()
    back = _run_ranks(lambda r: P.park_back_at_wake(scheds[r], env, logs.append,
                                                    epoch=5, group_ok=True))
    assert back == [True, True, True]
    assert all(torch.equal(bufs[0][L][:12], orig0[L][:12]) for L in range(2))
    assert not list(tmp_path.glob("weg2_l15_park.*"))       # sidecars consumed


def test_one_rank_without_manifest_turns_the_park_off_everywhere(monkeypatch, tmp_path):
    from sglang.srt.weg2.l15_manifest import Manifest

    m = Manifest(epoch=5, pid=1, spans=(), rows_by_rank=(12, 5, 6), anchor_slots=1)
    bufs = {r: _bufs(r) for r in range(3)}
    w, env, scheds = _entries(monkeypatch, tmp_path, [m, None, m], bufs, [0, 15, 30])
    logs = []
    assert _run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append)) == [None] * 3
    assert sum("result=off" in x for x in logs) == 3 and w.calls == 0
    assert not list(tmp_path.glob("weg2_l15_park.*"))
    back = _run_ranks(lambda r: P.park_back_at_wake(scheds[r], env, logs.append,
                                                    epoch=5, group_ok=True))
    assert back == [False] * 3


def test_refused_kv_resume_never_touches_the_pools(monkeypatch, tmp_path):
    from sglang.srt.weg2.l15_manifest import Manifest

    m = Manifest(epoch=5, pid=1, spans=(), rows_by_rank=(12, 5, 6), anchor_slots=1)
    bufs = {r: _bufs(r, fill=r + 1) for r in range(3)}
    w, env, scheds = _entries(monkeypatch, tmp_path, [m, m, m], bufs, [0, 15, 30])
    _run_ranks(lambda r: P.park_at_release(scheds[r], env, lambda s: None))
    calls = w.calls
    back = _run_ranks(lambda r: P.park_back_at_wake(scheds[r], env, lambda s: None,
                                                    epoch=5, group_ok=False))
    assert back == [False] * 3 and w.calls == calls


def test_blocks_are_cut_to_the_chunk_size_and_still_byte_exact():
    pieces, _ = P.park_plan(keep_rows=[12, 5, 6], caps=[0, 15, 30])
    w = _World(3)
    bufs = {0: _bufs(0, fill=11), 1: _bufs(1), 2: _bufs(2)}
    orig0 = [b.clone() for b in bufs[0]]
    env = {"SGLANG_WEG2_L15_PARK_CHUNK_MIB": "1"}
    assert P.chunk_rows(4, env) == (1 << 20) // 4
    # 1 MiB per block with 4-byte rows would be one block; force 5-row blocks
    import unittest.mock as um
    with um.patch.object(P, "chunk_rows", lambda rb, e=None: 5):
        th = [threading.Thread(target=P.run_park,
                               args=("out", pieces, r, 3, bufs[r], w.a2a_for(r), env))
              for r in range(3)]
        [t.start() for t in th]
        [t.join() for t in th]
    for p in pieces:
        for L in range(2):
            assert torch.equal(bufs[p.dst][L][p.dst_row:p.dst_row + p.rows],
                               orig0[L][p.src_row:p.src_row + p.rows])
    expect = sum(-(-p.rows // 5) for p in pieces) * 2
    assert w.calls == expect
