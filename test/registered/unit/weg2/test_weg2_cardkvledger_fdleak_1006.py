# SPDX-License-Identifier: Apache-2.0
"""CardKvLedger fd leak in the dual P grant path (27B dual rc12z30y9nf21, 06.10. 12:18:13Z).

METAL: P sat 19 min in the P-KV wait (GRANT-SHORT, D kept the card pool), PP0 retried
``pp0_grant`` every scheduler round (~523 attempts/s, SGLANG_WEG2_DUAL_GRANT_RETRY_MS=0)
and every attempt opened a fresh ``CardKvLedger`` per card through ``group_grant`` --
its ``_lock_fd`` (an int from ``os.open``, no finalizer) was never closed. ~172k attempts
x 3 cards = the 524288 fd limit: ``OSError: [Errno 24] Too many open files:
'/dev/shm/wkv-559f09d88d0b'`` in ``CardKvLedger.__init__``, PP0 dead, ADMISSION-WEDGE.

DANGER DIRECTIONS guarded here (each measured as the process fd count in /proc/self/fd):
* the short path of ``group_grant`` (return 0) closes every ledger it opened;
* an exception while opening/charging closes them too AND returns the charges taken;
* ``pp0_grant``'s MAP-SHORT wait closes them;
* a granted ledger the caller keeps (the followers' untold charge) is closed when the
  told goes on the wire (``with_dual_kv``) or the charge goes back (``return_untold_grant``);
* an explicit ``close()`` is idempotent (a double close must never hit a reused fd number);
* the grant itself (levels, charges, all-or-none) is unchanged.
"""
from __future__ import annotations

import inspect
import json
import os
import textwrap
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.weg2 import card_kv_ledger as K  # noqa: E402
from sglang.srt.weg2 import dual_p_kv_stage as S  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MIB = 1 << 20
STEP = 4096
PER_TOKEN = (2048, 4096, 6144)
N = 200
#: an open CardKvLedger holds TWO descriptors: its lock fd and the dup() that ``mmap.mmap`` keeps
#: (the mmap one is given back when the object is collected, the lock fd never -- the leak)
FD = 2


def _nfd() -> int:
    return len(os.listdir("/proc/self/fd"))


def _cards(tmp_path, monkeypatch, map_granted=None):
    """3 cards, D contributed 4096 MiB each; returns (paths, the P ledgers kept open, actor)."""
    paths, held = [], []
    for r, per in enumerate(PER_TOKEN):
        path = str(tmp_path / ("card%d" % r))
        d = K.CardKvLedger(path, "D")
        d.contribute(4096 * MIB, committed=0)
        p = K.CardKvLedger(path, "P")
        p.contribute(0)
        held += [d, p]
        stage = {"ledger": path, "step": STEP, "top": 196608,
                 "bytes": [k * STEP * per for k in range(196608 // STEP + 1)]}
        with open(str(tmp_path / ("stage%d" % r)), "w") as f:
            json.dump(stage, f)
        paths.append(path)
    monkeypatch.setattr(S, "stage_file", lambda tag, r, root="/dev/shm": str(tmp_path / ("stage%d" % r)))
    actor = types.SimpleNamespace(page=64, _committed=0, ledger=held[1],
                                  map_granted=map_granted or (lambda lvl, charged=None: None))
    monkeypatch.setattr(S, "_actor", lambda sched: actor)
    return paths, held, actor


def _sched():
    return types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0, pp_size=3), waiting_queue=[],
                                 _weg2_store_told_armed=True, _weg2_store_held={})


def _req(rid="weg2-0-1", tokens=90000):
    return types.SimpleNamespace(rid=rid, origin_input_ids=list(range(tokens)), _dual_grant_untold=None)


def _p(path):
    return K.peek(path).committed["P"]


def _stages(paths):
    return [{"ledger": p, "step": STEP, "top": 16384, "bytes": [k * (10 << 20) for k in range(5)]}
            for p in paths]


def _open_p(pth):
    return K.CardKvLedger(pth, "P")


def _starve_last_card(held):
    """D takes the whole pool of card 2: the sorted order reaches it last, so a short
    grant has opened (and taken from) all three cards before it fails."""
    held[4].request(1 << 40)


# -- group_grant ------------------------------------------------------------------


def test_group_grant_short_path_closes_every_ledger_it_opened(tmp_path, monkeypatch):
    paths, held, _a = _cards(tmp_path, monkeypatch)
    _starve_last_card(held)
    before = _nfd()
    for _ in range(N):
        assert S.group_grant(_stages(paths), 4000, _open_p) == 0
    assert _nfd() - before <= 0, "fds grew by %d over %d short grants" % (_nfd() - before, N)
    for p in paths:
        assert _p(p) == 0, "all or none: the charges taken on cards 0/1 went back"


def test_group_grant_success_without_taken_out_closes_the_ledgers(tmp_path, monkeypatch):
    paths, _held, _a = _cards(tmp_path, monkeypatch)
    before = _nfd()
    for _ in range(N):
        assert S.group_grant(_stages(paths), 4000, _open_p) == 4096
    assert _nfd() - before <= 0, "nobody can own the ledgers of a grant without taken_out"


def test_group_grant_with_taken_out_hands_the_open_ledgers_to_the_caller(tmp_path, monkeypatch):
    paths, _held, _a = _cards(tmp_path, monkeypatch)
    taken: list = []
    before = _nfd()
    assert S.group_grant(_stages(paths), 4000, _open_p, taken_out=taken) == 4096
    assert len(taken) == 3 and _nfd() - before == 3 * FD, "the caller owns three live ledgers"
    for _i, led, got in taken:                                 # they are usable (not closed under the caller)
        assert got == 10 << 20
        led.release(got)
    for _i, led, _g in taken:
        led.close()
    assert _nfd() - before == 0
    for p in paths:
        assert _p(p) == 0


def test_group_grant_open_failure_returns_the_charges_and_closes(tmp_path, monkeypatch):
    paths, _held, _a = _cards(tmp_path, monkeypatch)
    before = _nfd()
    for _ in range(50):
        calls = []

        def flaky(pth):
            calls.append(pth)
            if len(calls) == 3:
                raise OSError(24, "Too many open files", pth)
            return K.CardKvLedger(pth, "P")

        with pytest.raises(OSError):
            S.group_grant(_stages(paths), 4000, flaky)
    assert _nfd() - before <= 0
    for p in paths:
        assert _p(p) == 0, "the two charges taken before the failing open went back"


def test_close_is_idempotent_and_never_touches_a_reused_fd(tmp_path):
    led = K.CardKvLedger(str(tmp_path / "c"), "P")
    led.close()
    other = os.open(str(tmp_path / "other"), os.O_RDWR | os.O_CREAT)   # likely reuses the closed fd number
    try:
        led.close()                                                    # must not close ``other``
        os.fstat(other)
    finally:
        os.close(other)


# -- pp0_grant ------------------------------------------------------------------------


def test_pp0_grant_short_wait_leaks_nothing_over_many_rounds(tmp_path, monkeypatch):
    paths, held, _a = _cards(tmp_path, monkeypatch)
    _starve_last_card(held)
    sched, req = _sched(), _req()
    before = _nfd()
    for _ in range(N):
        assert S.pp0_grant(sched, req) == 0
    assert _nfd() - before <= 0, "fds grew by %d over %d pp0_grant waits (3 per round = the metal)" % (
        _nfd() - before, N)


def test_pp0_grant_map_short_wait_closes_the_ledgers_and_returns_the_charges(tmp_path, monkeypatch):
    def short(lvl, charged=None):
        raise S.Weg2DualKvMapShort("cuMemCreate OOM")

    paths, _held, _a = _cards(tmp_path, monkeypatch, map_granted=short)
    monkeypatch.setattr(S, "phys_free_bytes", lambda: None)
    sched, req = _sched(), _req()
    before = _nfd()
    for _ in range(50):
        assert S.pp0_grant(sched, req) == 0
    assert _nfd() - before <= 0
    for p in paths:
        assert _p(p) == 0


def test_pp0_grant_unexpected_error_after_the_grant_closes_the_ledgers(tmp_path, monkeypatch):
    def boom(lvl, charged=None):
        raise ZeroDivisionError("anything but MapShort")

    _paths, _held, _a = _cards(tmp_path, monkeypatch, map_granted=boom)
    sched, req = _sched(), _req(tokens=4000)               # small: the charges of an unexpected error stand (a rank death)
    before = _nfd()
    for _ in range(20):
        with pytest.raises(ZeroDivisionError):
            S.pp0_grant(sched, req)
    assert _nfd() - before <= 0


def test_pp0_grant_success_keeps_only_the_followers_until_the_told_leaves(tmp_path, monkeypatch):
    paths, _held, _a = _cards(tmp_path, monkeypatch)
    sched = _sched()
    before = _nfd()
    for k in range(N):
        req = _req(rid="weg2-0-%d" % k, tokens=4000)
        lvl = S.pp0_grant(sched, req)
        assert lvl == 4096
        assert len(req._dual_grant_untold) == 2 and _nfd() - before == 2 * FD, "PP0's own + zero-charge ledgers closed"
        told = S.with_dual_kv(types.SimpleNamespace(rid=req.rid), req)   # the told is on the wire
        assert getattr(told, S.WIRE_DUAL_KV) == lvl and req._dual_grant_untold is None
        assert _nfd() - before == 0, "the followers' ledgers closed when the told left"
        for p, per in zip(paths, PER_TOKEN):                  # a follower adopts+returns its charge (stand-in)
            led = K.CardKvLedger(p, "P")
            led.release(lvl * per)
            led.close()
    assert _nfd() - before == 0


def test_return_untold_grant_releases_then_closes(tmp_path, monkeypatch):
    paths, _held, _a = _cards(tmp_path, monkeypatch)
    sched, req = _sched(), _req(tokens=4000)
    before = _nfd()
    lvl = S.pp0_grant(sched, req)
    assert lvl == 4096 and _nfd() - before == 2 * FD
    assert S.return_untold_grant(sched, req, "abort") == lvl * (PER_TOKEN[1] + PER_TOKEN[2])
    assert _nfd() - before == 0 and req._dual_grant_untold is None
    assert _p(paths[1]) == 0 and _p(paths[2]) == 0


def test_re_intake_regrant_does_not_stack_open_ledgers(tmp_path, monkeypatch):
    _paths, _held, _a = _cards(tmp_path, monkeypatch)
    sched, req = _sched(), _req(tokens=4000)
    before = _nfd()
    for _ in range(20):
        assert S.pp0_grant(sched, req) == 4096                # intake again, no told in between
    assert _nfd() - before == 2 * FD, "one untold pair, the earlier pairs were returned and closed"


# -- mutants: the guards are what keeps the count flat ---------------------------------------


def _mutant(fn, fixed, back, count=1):
    src = textwrap.dedent(inspect.getsource(inspect.unwrap(fn)))
    assert src.count(fixed) == count, "the guarded line moved -- re-aim the mutant (%d found)" % src.count(fixed)
    ns = dict(vars(S))
    exec(compile(src.replace(fixed, back), S.__file__, "exec"), ns)
    return ns[fn.__name__]


def test_mutant_no_close_on_the_short_path_turns_the_short_grant_test_red(tmp_path, monkeypatch):
    m = _mutant(S.group_grant, "    if short:\n        try:\n            _give_back(taken)\n        finally:\n"
                "            _close_all(taken)\n        return 0", "    if short:\n        _give_back(taken)\n        return 0")
    paths, held, _a = _cards(tmp_path, monkeypatch)
    _starve_last_card(held)
    before = _nfd()
    for _ in range(20):
        m(_stages(paths), 4000, _open_p)
    assert _nfd() - before > 0, "the mutant must leak (else the test guards nothing)"


def test_mutant_no_close_in_with_dual_kv_leaks_the_followers(tmp_path, monkeypatch):
    m = _mutant(S.with_dual_kv, "_close_untold(req)", "req._dual_grant_untold = None")
    _paths, _held, _a = _cards(tmp_path, monkeypatch)
    sched, req = _sched(), _req(tokens=4000)
    before = _nfd()
    S.pp0_grant(sched, req)
    m(types.SimpleNamespace(rid=req.rid), req)
    assert _nfd() - before > 0


def test_mutant_no_close_on_map_short_leaks(tmp_path, monkeypatch):
    def short(lvl, charged=None):
        raise S.Weg2DualKvMapShort("cuMemCreate OOM")

    _paths, _held, _a = _cards(tmp_path, monkeypatch, map_granted=short)
    monkeypatch.setattr(S, "phys_free_bytes", lambda: None)
    m = _mutant(S.pp0_grant, "            try:\n                _give_back(taken)\n            finally:\n"
                "                _close_all(taken)\n            phys = phys_free_bytes()",
                "            _give_back(taken)\n            phys = phys_free_bytes()")
    sched, req = _sched(), _req()
    before = _nfd()
    for _ in range(10):
        assert m(sched, req) == 0
    assert _nfd() - before > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
