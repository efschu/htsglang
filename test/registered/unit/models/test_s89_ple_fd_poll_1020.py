"""S89: NF D TP0 died on ``ValueError: filedescriptor out of range in select()``.

THE METAL (NF rc19 10e9b2b492, boot 1020_072658, D.log line 62710, 07:43:23Z): ``_gather_into -> w.ready(pend.seq)
-> select.select`` (PLE-PREFETCH, an extend on D, a vision request). The PLE-PREFETCH workers started ONCE, at
07:30:50Z (``PLE-PREFETCH on: 4 pread worker processes``, disabled 0) -- a pipe keeps its number for life, so their
pipes were numbered >= 1024 from the start; the first ``ready()`` with a gather in flight came 13 min later. The
1024 fds before them are no leak: S36 KV-DURCHLAUF holds one L3 file per live disk page by design (TP0 07:30:42Z
live 892 -> 4070, peak 5465, back to 382 at 07:38:18Z; live == alloc - released every line).

WHAT MUST HOLD.
(1) The PLE pread workers wait on their pipes with poll(): a worker set made behind 1100 open fds gathers bytes
    identical to the serial gather, ``ready()`` and ``collect()`` included (RED on rc20 b40f47c91c: ValueError).
(2) N prefetch rounds keep the process' fd count constant; create/close cycles give every fd back.
(3) A dead worker is still a named PleWorkerLost (poll reports HUP as select reported EOF as readable).
(4) KV-DURCHLAUF names its held fds: one ``PDFLIP FD-BUDGET`` census line per crossed power of two from 1024,
    WARNING once the process reaches ``FD_BUDGET_FRAC`` of its soft RLIMIT_NOFILE; the count is exact
    (disk pages + QSA sidecars held), and release gives both back.
"""
from __future__ import annotations

import logging
import os
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.models import qwen4_exp_ple_prefetch as pf  # noqa: E402
from flliper.srt.pdflip import kv_durchlauf as kd  # noqa: E402

from test_qwen4_exp_ple_prefetch_h32 import _Run, _same, _serial, _table  # noqa: E402

LOGGER = "flliper.srt.models.qwen4_exp_ple_prefetch"


def _nfds() -> int:
    return len(os.listdir("/proc/self/fd"))


@pytest.fixture
def table():
    with tempfile.TemporaryDirectory() as d:
        yield _table(d)


@pytest.fixture
def ids():
    g = torch.Generator().manual_seed(89)
    return torch.randint(0, 256, (1200,), generator=g, dtype=torch.int64)


@pytest.fixture
def many_fds():
    """The D rank of the metal: > 1024 fds open before the PLE workers start."""
    held = [os.open(os.devnull, os.O_RDONLY) for _ in range(1100)]
    assert max(held) >= 1024
    yield held
    for fd in held:
        os.close(fd)


# ---- (1) the metal --------------------------------------------------------------------------------------------

def test_rc19_nf_ple_prefetch_behind_1100_open_fds_gathers_the_same_bytes(table, ids, many_fds):
    """RED on b40f47c91c: ``ValueError: filedescriptor out of range in select()`` at the first ``ready()``."""
    run = _Run(None, table, ids, 200)
    try:
        for c in range(4):
            got_ids, out = run.step(c)
            assert _same(out, _serial(table, got_ids)), f"chunk {c}"
        w = run.g._workers
        assert w is not None and not run.g._disabled, "the prefetcher fell back to the serial gather"
        assert min(p.stdout.fileno() for p in w._procs) >= 1024, "the metal shape: worker pipes above 1023"
    finally:
        run.close()


def test_collect_and_ready_with_pipes_above_1023(table, many_fds):
    w = pf.PlePreadProcs(list(dict.fromkeys(table.shard_files)), table.row_bytes, depth=1, procs=2)
    try:
        assert min(p.stdout.fileno() for p in w._procs) >= 1024
        w.map_slot(0, 64 * table.row_bytes)
        keys = torch.arange(32, dtype=torch.int64) * table.row_bytes
        seq = w.submit(0, torch.arange(32, dtype=torch.int64), keys)
        done, _s = w.collect(seq, 5.0)
        assert sorted(done) == [0, 1] and w.ready(seq)
    finally:
        w.close()


# ---- (2) no fd grows ------------------------------------------------------------------------------------------

def test_n_prefetch_rounds_keep_the_fd_count_constant(table, ids):
    big = torch.cat([ids] * 4)
    run = _Run(None, table, big, 100)
    try:
        run.step(0)
        base = _nfds()
        for c in range(1, 40):
            run.step(c)
        assert _nfds() == base, "a prefetch round left an fd open"
    finally:
        run.close()


def test_create_close_cycles_give_every_fd_back(table, ids):
    base = _nfds()
    for _ in range(4):
        run = _Run(None, table, ids, 200)
        run.step(0)
        run.step(1)
        procs = list(run.g._workers._procs)      # held, as a leaking owner would hold them
        run.close()
        for p in procs:
            assert p.stdin.closed and p.stdout.closed, "close() left a worker pipe open"
            assert p.returncode is not None, "close() left a worker running"
        del procs
    assert _nfds() == base, "a closed prefetcher kept pipes or slot memfds"


# ---- (3) a dead worker stays named ----------------------------------------------------------------------------

def test_a_dead_worker_is_still_lost_and_the_gather_falls_back_with_the_same_bytes(table, ids, caplog):
    run = _Run(None, table, ids, 200)
    try:
        run.step(0)
        w = run.g._workers
        pend = run.g._pending
        assert pend is not None
        for p in w._procs:
            p.kill()
            p.wait()
        with pytest.raises(pf.PleWorkerLost):
            w.collect(pend.seq, 2.0) if not w.ready(pend.seq) else w.join(pend.seq)
        caplog.set_level(logging.ERROR, LOGGER)
        got_ids, out = run.step(1)
        assert _same(out, _serial(table, got_ids))
        assert run.g._disabled
    finally:
        run.close()


def test_poll_readable_does_not_care_about_the_fd_number(many_fds):
    r, wfd = os.pipe()
    try:
        assert r >= 1024
        assert pf._poll_readable([r], 0.0) == []
        os.write(wfd, b"x")
        assert pf._poll_readable([r], 0.0) == [r]
        os.close(wfd)
        wfd = -1
        os.read(r, 1)
        assert pf._poll_readable([r], 0.0) == [r], "EOF (HUP) must count as readable, as select() did"
    finally:
        os.close(r)
        if wfd >= 0:
            os.close(wfd)


# ---- (4) the named fd budget -----------------------------------------------------------------------------------

def _files(d, n):
    out = []
    for i in range(n):
        p = os.path.join(d, f"page-{i}")
        with open(p, "wb") as fh:
            fh.write(b"\0" * 4096)
        out.append(p)
    return out


def test_fd_budget_census_per_power_of_two_and_warning_at_the_fraction(monkeypatch, caplog):
    caplog.set_level(logging.INFO, "flliper.srt.pdflip.kv_durchlauf")
    reg = kd.DiskPages()
    with tempfile.TemporaryDirectory() as d:
        paths = _files(d, 1100)
        ds = reg.alloc([f"k{i}" for i in range(1000)], [f"s{i}" for i in range(1000)], paths[:1000], 4096)
        assert not [m for m in caplog.messages if kd.FD_BUDGET_MARK in m]
        assert reg.held_fds() == 1000
        ds += reg.alloc(["k1000", "k1001"] + [f"k{i}" for i in range(1002, 1030)],
                        [f"s{i}" for i in range(1000, 1030)], paths[1000:1030], 4096)
        lines = [r for r in caplog.records if kd.FD_BUDGET_MARK in r.getMessage()]
        assert len(lines) == 1 and lines[0].levelno == logging.INFO
        assert "held=1030 (disk pages 1030 + qsa 0)" in lines[0].getMessage()
        assert reg.attach_qsa(ds[0], paths[1050], False) and reg.held_fds() == 1031
        # the same power of two: no second line
        reg.alloc(["x"], ["y"], [paths[1060]], 4096)
        assert len([m for m in caplog.messages if kd.FD_BUDGET_MARK in m]) == 1
        # release gives disk-page AND QSA fds back, counted exactly
        before = kd.open_fds()
        assert reg.release(ds) == len(ds)
        assert reg.held_fds() == 1 and kd.open_fds() == before - len(ds) - 1
        # a soft limit the process is near: the next crossing is a WARNING naming it
        monkeypatch.setattr(kd, "nofile_soft", lambda: 1000)
        reg2 = kd.DiskPages()
        reg2.alloc([f"a{i}" for i in range(1024)], [f"b{i}" for i in range(1024)], paths[:1024], 4096)
        warn = [r for r in caplog.records if kd.FD_BUDGET_MARK in r.getMessage() and r.levelno == logging.WARNING]
        assert warn and "OVER the budget" in warn[-1].getMessage()
        reg2.release(list(range(1024)))
        reg.release(list(reg._e))
