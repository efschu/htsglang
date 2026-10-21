"""Review EM-PREFETCH (21.10., independent reviewer; code a0056194c2 on desk/nf-rc30-dprefill-1021).

What the author's file (test_em_prefetch_1021.py) does not pin, on the same S51 CPU pool harness:

* R-1 rank-local skip / defect-off: a rank that skips the prefetch (resume-warm guard) or switched it off after a
  defect computes the SAME output bits with the SAME structure (waves, applies, forward counter) as a rank that
  prefetches -- the property the "only this rank skips" path rests on (Rang-Einigkeit), over several forwards,
  with right, wrong and partial predictions, one- and two-wave targets;
* R-2 the band verdict does not depend on residency / clock / prefetch marks (the rank-local state);
* R-3 a prefetch never fills an OFF seat row, and a lend shrink / return after a landed prefetch keeps every
  output exact (D-TRANSIENT-LEND rows_on at the table level);
* R-4 fire/join invariant over a 3-layer chain on fake CUDA streams: every fired target is joined exactly once,
  before its own path, and nothing is left armed or pending at the end of the forward (what the un-waited
  ``_pool_pf_armed = False`` sites in rearm_after_wake / _promote_deferred rely on);
* R-5 memory: the prefetch buffers are allocated once per layer, at the EM step's width (so the only new kernel
  specialisation is PREFETCH=True), and stay the same object;
* R-6 the prefetch side stream is a registered ROW stream (F3 fence covers it);
* B-1 (FINDING, fixed by review A1 -- revoke_prefetched_rows): a defect AFTER the prefetch step ran but BEFORE its row copy (here: the copy raises)
  leaves the target's tables naming rows whose bytes never arrived -- the next real step takes them as hits and
  computes with the wrong bytes. Was xfail strict on a0056194c2; plain test since A1.
"""

import importlib.util
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.layers.moe import expert_offload as eo  # noqa: E402
from flliper.srt.layers.moe import expert_pool_device as ep  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(__file__)

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "_t_em_prefetch_review_s51", os.path.join(_HERE, "test_pool_eager_device_em_s51.py"))
S51 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S51)

E, R = S51.E, S51.R


@pytest.fixture(autouse=True)
def _switch():
    eo.pool_em_prefetch_reset_for_tests()
    yield
    eo.pool_em_prefetch_reset_for_tests()


def _cold(routes):
    return sorted({e for row in routes for e in row if e >= R})


def _pair(monkeypatch, warm=()):
    a = S51._pool_cache(monkeypatch, warm=warm)
    b = S51._pool_cache(monkeypatch, warm=warm)
    b.layer.layer_id = 8
    return a, b


# a's routes, b's routes, the prediction a hands b (right / wrong / partial / two-wave target)
_RIGHT = S51.ROUTES
_WRONG = [[e for e in range(R, E) if e not in _cold(S51.ROUTES)]]
_PARTIAL = [[5, 6, 7, 20, 23, -1]]
_SEQ = [
    (S51.ROUTES, S51.ROUTES, _RIGHT),
    (S51.WIDE, S51.ROUTES, _WRONG),
    (S51.ROUTES, S51.WIDE, _PARTIAL),
    (S51.WIDE, S51.WIDE, S51.WIDE),
    (S51.ROUTES, S51.ROUTES, _PARTIAL),
]


def _run_seq(monkeypatch, mode):
    a, b = _pair(monkeypatch)
    if mode == "skip":
        b._rw_plan = object()  # the resume-warm guard: only this rank skips
    rec = []
    for i, (ra, rb, pred) in enumerate(_SEQ):
        if mode == "defect":
            eo._POOL_EM_PREFETCH["defect"] = True
        calls_a, calls_b = [], []
        a.set_em_prefetch(b, torch.tensor(pred, dtype=torch.int32))
        out_a = S51._extend(a, ra, seed=10 + i, calls=calls_a)
        out_b = S51._extend(b, rb, seed=20 + i, calls=calls_b)
        assert torch.equal(out_a[:, 0], S51._truth(ra, seed=10 + i)), (mode, i)
        assert torch.equal(out_b[:, 0], S51._truth(rb, seed=20 + i)), (mode, i)
        rec.append((out_a.clone(), out_b.clone(), tuple(calls_a), tuple(calls_b),
                    int(a._pool_tables.forwards[0]), int(b._pool_tables.forwards[0]),
                    a.eager_device_em_waves(len(ra) * 3)[0], b.eager_device_em_waves(len(rb) * 3)[0]))
        for t in (a._pool_tables, b._pool_tables):
            assert int(t.error[0]) == 0 and ep.bijection_breaks(t) == 0, (mode, i)
    return rec, b


def test_r1_skip_and_defect_ranks_match_the_prefetching_rank_bit_for_bit_and_in_structure(monkeypatch):
    on, b_on = _run_seq(monkeypatch, "on")
    assert int(b_on._pool_tables.pf_counts[2]) > 0, "the prefetching rank really hit prefetched rows"
    skip, b_skip = _run_seq(monkeypatch, "skip")
    assert b_skip._pool_pf_buffers is None, "the skipping rank never prefetched"
    eo.pool_em_prefetch_reset_for_tests()
    off, b_off = _run_seq(monkeypatch, "defect")
    assert b_off._pool_pf_buffers is None
    for i, (x, y, z) in enumerate(zip(on, skip, off)):
        for k in range(2):
            assert torch.equal(x[k], y[k]) and torch.equal(x[k], z[k]), ("output", i, k)
        # waves, apply calls (the per-wave GEMM chain), forward counter: no structural difference
        assert x[2:] == y[2:] == z[2:], ("structure", i, x[2:], y[2:], z[2:])


def test_r2_the_band_verdict_does_not_read_residency_clock_or_prefetch_marks(monkeypatch):
    plain = S51._pool_cache(monkeypatch)
    busy = S51._pool_cache(monkeypatch, warm=list(range(4, 13)))
    t = busy._pool_tables
    t.clock[0] = 41
    t.pf_row[t.lru_start:t.pool_rows] = 40
    t.pf_counts[:] = 7
    busy.prefetch_pool_em(torch.tensor([[20, 21, 22]], dtype=torch.int32))
    for n_ids in (1, 5, 6, 7, 18, 24, 64, 4096):
        assert plain.em_prefetch_wanted(n_ids) == busy.em_prefetch_wanted(n_ids), n_ids


# ---------------------------------------------------------------------------------------------- seat rows / lend

_X = 4


def _seat_cache(monkeypatch, k_on):
    cache = S51._pool_cache(monkeypatch)
    C, S = cache.scratch, int(cache._pool_tables.staging_rows.shape[0])
    spill = cache._pinned["w13"]
    bank = torch.full((R + C + _X, S51.W), -1.0, dtype=torch.float32)
    for e in range(R):
        bank[e].fill_(e)
    hot_slot_of, host_row = cache._pool_layout()
    t = ep.allocate_pool_tables("cpu", E, R + C + _X, R, S, hot_slot_of, host_row, seat_rows=_X)
    ep.set_seat_rows_on(t, k_on)
    cache._resident = {"w13": bank}
    cache._pool_tables = t
    cache._pool_dsts = [bank]
    cache._pool_srcs = [spill]
    return cache, t, bank


def _off_rows(t):
    lo, hi = ep.seat_off_range(t) or (0, 0)
    return set(range(lo, hi))


def _move_rows(bank):
    def _mv(src, dst):
        bank[dst.long()] = bank[src.long()].clone()

    return _mv


def test_r3_a_prefetch_never_fills_an_off_seat_row_and_a_lend_after_it_stays_exact(monkeypatch):
    cache, t, bank = _seat_cache(monkeypatch, k_on=2)
    off = _off_rows(t)
    assert off, "two seat rows stay OFF"
    keys_off = t.row_key[list(off)].clone()
    # every cold expert predicted: far more misses than free rows
    cache.prefetch_pool_em(torch.tensor([list(range(R, E))], dtype=torch.int32))
    assert all(int(t.hot_phys[e]) not in off for e in range(E))
    assert torch.equal(t.row_key[list(off)], keys_off)
    assert int(t.pf_counts[1]) > 0
    # lend: two more seat rows go OFF after the prefetch landed (coldest-first moves, device form)
    ep.seat_rows_off_device(t, 0, move_rows=_move_rows(bank))
    got = S51._extend(cache, S51.ROUTES, seed=31)
    assert torch.equal(got[:, 0], S51._truth(S51.ROUTES, seed=31))
    # return: all seat rows ON again, a second prefetch, then a two-wave extend
    ep.set_seat_rows_on(t, _X)
    cache.prefetch_pool_em(torch.tensor(S51.WIDE, dtype=torch.int32))
    got = S51._extend(cache, S51.WIDE, seed=32)
    assert torch.equal(got[:, 0], S51._truth(S51.WIDE, seed=32))
    assert int(t.error[0]) == 0 and ep.bijection_breaks(t) == 0
    for e in range(E):
        r = int(t.hot_phys[e])
        if r >= t.lru_start:
            assert float(bank[r, 0]) == float(e), (e, r)


# ---------------------------------------------------------------------------------------------- fire / join

class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_event(self, ev):
        self.log.append(("wait", self.name, ev.name))


class _Event:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def record(self, stream=None):
        self.log.append(("record", self.name, getattr(stream, "name", None)))


def _fake_cuda(monkeypatch, log):
    main = _Stream("main", log)
    side = _Stream("side", log)
    cur = {"s": main}

    class _Ctx:
        def __init__(self, s):
            self.s = s

        def __enter__(self):
            self.prev, cur["s"] = cur["s"], self.s

        def __exit__(self, *a):
            cur["s"] = self.prev

    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: cur["s"])
    monkeypatch.setattr(torch.cuda, "stream", _Ctx)
    monkeypatch.setattr(eo, "pool_prefetch_stream", lambda: side)


def test_r4_every_fired_target_is_joined_once_before_its_path_and_nothing_is_left_armed(monkeypatch):
    log = []
    _fake_cuda(monkeypatch, log)
    caches = [S51._pool_cache(monkeypatch) for _ in range(3)]
    for i, c in enumerate(caches):
        c.layer.layer_id = 7 + i
        c._pool_pf_begin, c._pool_pf_done = _Event("begin%d" % i, log), _Event("done%d" % i, log)
        monkeypatch.setattr(c, "_em_prefetch_streams", lambda: True)
        real_path = c._run_eager_pool_path

        def _path(d, f, n, _i=i, _real=real_path):
            log.append(("path", _i))
            return _real(d, f, n)

        monkeypatch.setattr(c, "_run_eager_pool_path", _path)
    routes = [S51.ROUTES, S51.WIDE, S51.ROUTES]
    for fwd in range(2):
        log.clear()
        for i, c in enumerate(caches):
            if i + 1 < len(caches):
                c.set_em_prefetch(caches[i + 1], torch.tensor(routes[i + 1], dtype=torch.int32))
            got = S51._extend(c, routes[i], seed=40 + i + fwd)
            assert torch.equal(got[:, 0], S51._truth(routes[i], seed=40 + i + fwd))
        for i in (1, 2):
            waits = [k for k, x in enumerate(log) if x == ("wait", "main", "done%d" % i)]
            records = [k for k, x in enumerate(log) if x == ("record", "done%d" % i, "side")]
            path = log.index(("path", i))
            assert len(waits) == 1 and len(records) == 1, (fwd, i, log)
            assert records[0] < waits[0] < path, (fwd, i, log)
        assert ("record", "done0", "side") not in log  # layer 0 is nobody's target
        assert not any(c._pool_pf_armed for c in caches), "a join left owed at the end of the forward"
        assert all(c._em_pf_pending is None for c in caches)


# ---------------------------------------------------------------------------------------------- memory / JIT

def test_r5_prefetch_buffers_once_per_layer_at_the_em_step_width(monkeypatch):
    a, b = _pair(monkeypatch)
    a.set_em_prefetch(b, torch.tensor(S51.ROUTES, dtype=torch.int32))
    S51._extend(a, S51.ROUTES, seed=1)
    pf = b._pool_pf_buffers
    assert pf is not None and a._pool_pf_buffers is None  # only a TARGET allocates, and only once
    width = int(b._pool_buffers.gather_src.shape[0])
    assert int(pf.gather_src.shape[0]) == width  # same WIDTH constexpr as the EM step: one new variant only
    total = sum(int(getattr(pf, f).numel()) for f in pf.__dataclass_fields__)
    assert total <= 7 * width + E + 3
    S51._extend(b, S51.ROUTES, seed=2)
    a.set_em_prefetch(b, torch.tensor(S51.WIDE, dtype=torch.int32))
    S51._extend(a, S51.ROUTES, seed=3)
    assert b._pool_pf_buffers is pf


def test_r6_the_prefetch_side_stream_is_a_registered_row_stream(monkeypatch):
    from flliper.srt.pdflip import d_row_fence

    monkeypatch.setattr(d_row_fence, "_ROW_STREAMS", {})
    monkeypatch.setattr(eo, "_POOL_PREFETCH_STREAM", None)
    fake = SimpleNamespace(stream_id=77, device_index=0, device_type=1)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **k: fake)
    s = eo.pool_prefetch_stream()
    assert s is fake
    assert any(x is fake for x in d_row_fence.row_streams())


# ---------------------------------------------------------------------------------------------- finding B-1

def test_b1_a_defect_after_the_prefetch_step_never_serves_a_row_whose_copy_failed(monkeypatch):
    a, b = _pair(monkeypatch)
    real_copy = ep.copy_rows

    def _copy(srcs, dsts, src_rows, dst_rows, count):
        pf = b._pool_pf_buffers
        if pf is not None and src_rows is pf.gather_src:
            raise RuntimeError("prefetch copy failed after the step")
        return real_copy(srcs, dsts, src_rows, dst_rows, count)

    monkeypatch.setattr(ep, "copy_rows", _copy)
    a.set_em_prefetch(b, torch.tensor(S51.ROUTES, dtype=torch.int32))
    S51._extend(a, S51.ROUTES, seed=50)
    assert eo.pool_em_prefetch_enabled() is False  # the DEFECT path ran and switched it off
    got = S51._extend(b, S51.ROUTES, seed=51)
    assert torch.equal(got[:, 0], S51._truth(S51.ROUTES, seed=51))
