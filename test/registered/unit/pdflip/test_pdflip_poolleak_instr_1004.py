"""POOLLEAK-INSTR (NF y9nf6 boot 3, 191228abc5, D log ...10040650_191228abc5_1004_065037,
07:03:50Z): log-only instruments that name the rows of the post-park pool leak.

Specimen on TP1/TP2, right after ``PDFLIP-D-PARK park_running epoch=24`` (3 running retracted):
``[full] total=524288, available=0, evictable=416640, withheld=92672 ... deficit of 128 row(s)``
(TREE CENSUS protected=14848) and ``[mamba] total=38, available=18, evictable=16``
(protected 1, ``leaked_mamba_pages={33, 34, 31}``). The enumerated ``leaked_full_pages`` held
92800 ids: the 92672 KvRowCap-withheld ids plus the 128 nobody owns.

Pinned: (1)/(2) the park's ledger names DEFICIT 128 / 3 from exactly those terms;
(3) the leak report subtracts the withheld ids and prints only the unowned ones;
off (the default) nothing is read and nothing is logged; the park and the leak report
call the instruments at the named points.
"""

import inspect
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from flliper.srt.managers.kv_backing_relief import KvRowCap
from flliper.srt.managers.scheduler_components import invariant_checker as ic
from flliper.srt.pdflip import d_park_runtime
from flliper.srt.pdflip import poolleak_instr as pi

SWITCH = "FLLIPER_PDFLIP_POOLLEAK_INSTR"


class _Tree:
    def __init__(self, values=()):
        self._values = torch.tensor(list(values), dtype=torch.int64)

    def supports_mamba(self):
        return True

    def full_evictable_size(self):
        return 416640

    def full_protected_size(self):
        return 14848

    def mamba_evictable_size(self):
        return 16

    def mamba_protected_size(self):
        return 1

    def all_values_flatten(self):
        return self._values


class _MambaAlloc:
    def __init__(self):
        self.slot_used = torch.zeros(39, dtype=torch.bool)
        self.slot_used[[31, 33, 34]] = True

    def available_size(self):
        return 18

    def phase_withheld_slots(self):
        return 0


def _sched():
    alloc = SimpleNamespace(size=524288, residency_withheld_slots=92672,
                            available_size=lambda: 0)
    return SimpleNamespace(
        token_to_kv_pool_allocator=alloc,
        tree_cache=_Tree(),
        req_to_token_pool=SimpleNamespace(mamba_allocator=_MambaAlloc(),
                                          mamba_pool=SimpleNamespace(size=38)),
        tp_group=SimpleNamespace(rank=1),
    )


class _Capture(logging.Handler):
    """The instrument's own logger only (no root-level change: the front
    tests after this file measure wall time and the root level is theirs)."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def cap():
    h = _Capture()
    lg = pi.logger
    old = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    try:
        yield h
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(SWITCH, "1")


def test_switch_is_declared_default_off(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)
    assert pi.enabled() is False


def test_off_reads_nothing_and_logs_nothing(monkeypatch, cap):
    monkeypatch.delenv(SWITCH, raising=False)

    class _Boom:
        def __getattr__(self, name):
            raise AssertionError(f"read {name} while off")

    assert pi.park_snapshot(_Boom(), [_Boom()], phase="before-retract", epoch=24) is None
    assert pi.ledger_line(_Boom(), epoch=24) is None
    assert pi.unwithheld_leak_ids(_Boom(), _Boom()) is None
    assert not [r for r in cap.records if pi.TAG in r.getMessage()]


def test_y9nf6_b3_park_ledger_names_the_deficit(on, cap):
    sched = _sched()
    line = pi.ledger_line(sched, epoch=24)
    assert "full=DEFICIT(128)" in line, line
    assert "mamba=DEFICIT(3)" in line, line
    assert "mamba_slot_used=3" in line
    led = pi.ledger(sched)
    assert led["full_unowned"] == 128 and led["mamba_unowned"] == 3


def test_park_snapshot_per_request_and_delta(on, cap):
    sched = _sched()
    req = SimpleNamespace(rid="pdflip-12-34", req_pool_idx=5, mamba_pool_idx=torch.tensor(33),
                          prefix_indices=torch.arange(209152), kv_committed_len=209362,
                          kv_allocated_len=209408, fill_ids=list(range(209362)))
    before = pi.park_snapshot(sched, [req], phase="before-retract", epoch=24)
    sched.token_to_kv_pool_allocator.available_size = lambda: 64
    line = pi.ledger_line(sched, epoch=24, before=before)
    msgs = [r.getMessage() for r in cap.records]
    reqline = [m for m in msgs if "PARK-REQ" in m][0]
    for part in ("rid=pdflip-12-34", "req_pool_idx=5", "mamba_pool_idx=33", "prefix=209152",
                 "committed=209362", "allocated=209408"):
        assert part in reqline, reqline
    assert "delta[full_available=+64 full_unowned=-64]" in line, line


def test_leak_ids_subtract_the_withheld(on):
    # ids 1..32; tree holds 1..10; free 11..14; KvRowCap withholds 17..32;
    # 15 and 16 have no owner -- the "128 rows" of the specimen in small
    alloc = SimpleNamespace(size=32, residency_withheld_slots=16,
                            free_pages=torch.tensor([11, 12, 13]),
                            release_pages=torch.tensor([14]))
    cap = KvRowCap(alloc)
    cap._withheld = torch.arange(17, 33)
    alloc._pdflip_kv_stage_cap = cap
    line = pi.unwithheld_leak_ids(alloc, _Tree(range(1, 11)))
    assert "leaked=18 withheld_ids=16" in line, line
    assert "unexplained=2 ids=[15-16(2)]" in line, line


def test_report_leak_carries_the_ids_when_on(on):
    alloc = SimpleNamespace(size=4, residency_withheld_slots=0,
                            free_pages=torch.tensor([1]), release_pages=torch.tensor([], dtype=torch.int64))
    class _Chk(ic.SchedulerInvariantChecker):
        def _allocator(self):
            return alloc

    chk = object.__new__(_Chk)
    object.__setattr__(chk, "tree_cache", _Tree([2]))
    seen = {}
    with patch.object(ic, "raise_error_or_warn", lambda _s, _f, _c, msg: seen.setdefault("msg", msg)):
        chk._report_leak("pool", "[full] total=4 ...")
    assert "POOLLEAK-INSTR LEAK-IDS" in seen["msg"] and "ids=[3-4(2)]" in seen["msg"], seen


def test_report_leak_unchanged_when_off(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)
    chk = object.__new__(ic.SchedulerInvariantChecker)
    seen = {}
    with patch.object(ic, "raise_error_or_warn", lambda _s, _f, _c, msg: seen.setdefault("msg", msg)):
        chk._report_leak("pool", "[full] total=4 ...")
    assert seen["msg"] == "pool memory leak detected! [full] total=4 ..."


def test_the_park_measures_around_its_retraction():
    src = inspect.getsource(d_park_runtime.park_running)
    i_before = src.index('phase="before-retract"')
    i_retract = src.index("retract_all(")
    i_after = src.index('phase="after-retract"')
    i_ledger = src.index("_poolleak.ledger_line(")
    assert i_before < i_retract < i_after < i_ledger
