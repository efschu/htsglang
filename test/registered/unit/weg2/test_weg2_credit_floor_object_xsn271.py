"""weg2xsn271 (18.09.2026): group D died in flip 6 -- the memory saver's
cu_mem_create ran out of memory (exit(1)) when TP0 resumed weights_4 with
free 2510 MiB against a 2502 MiB request: the credit wait had graded free
against the request alone, `corridor_floor_mib=0`. The boot HAD measured a
floor for group D on that card (MEASURED-D 3567 = 767 transient + 2800
reserve): `corridor_floor_mib` returns a CorridorFloor object, the updater
did `int(floor)`, the TypeError fell into the bare except and became None.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import corridor_guard as cg  # noqa: E402
from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402

MIB = 1 << 20
UUID = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"


@pytest.fixture(autouse=True)
def _card_and_group(monkeypatch):
    M = wu.SchedulerWeightUpdaterManager           # a slots dataclass: patch the CLASS
    monkeypatch.setattr(M, "_weg2_card_uuid", lambda self: UUID)
    monkeypatch.setattr(M, "_weg2_group_name", lambda self: "D")


def _mgr():
    M = wu.SchedulerWeightUpdaterManager
    m = M.__new__(M)
    m._weg2_floor_noted = False
    return m


def test_the_measured_floor_object_reaches_the_credit_wait(monkeypatch):
    seen = {}

    def fake_floor(uuid, *, group=None, user_reserve_mib=0, **kw):
        seen.update(uuid=uuid, group=group, reserve=user_reserve_mib)
        return cg.CorridorFloor(card_uuid=uuid, group=group, transient_mib=767,
                                reserve_mib=user_reserve_mib, source="MEASURED-D")

    monkeypatch.setattr(cg, "corridor_floor_mib", fake_floor)
    monkeypatch.setattr(cg, "user_reserve_by_card", lambda: {UUID: 2800})
    m = _mgr()
    assert wu.SchedulerWeightUpdaterManager._weg2_corridor_floor_bytes(m) == 3567 * MIB
    assert seen == {"uuid": UUID, "group": "D", "reserve": 2800}


def test_the_real_corridor_floor_object_is_not_a_number():
    """The defect's shape: int() on the authority's return value raises."""
    f = cg.CorridorFloor(card_uuid=UUID, group="D", transient_mib=1, reserve_mib=2, source="MEASURED-D")
    with pytest.raises(TypeError):
        int(f)
    assert f.mib == 3


def test_an_absent_reserve_grades_with_the_transient_alone(monkeypatch):
    monkeypatch.setattr(cg, "user_reserve_by_card", lambda: {})
    monkeypatch.setattr(cg, "corridor_floor_mib",
                        lambda uuid, *, group=None, user_reserve_mib=0, **kw: cg.CorridorFloor(
                            card_uuid=uuid, group=group, transient_mib=500,
                            reserve_mib=user_reserve_mib, source="UNMEASURED-FALLBACK"))
    assert wu.SchedulerWeightUpdaterManager._weg2_corridor_floor_bytes(_mgr()) == 500 * MIB


def test_an_unreadable_authority_is_still_none_and_the_field_is_declared(monkeypatch):
    import dataclasses
    assert "_weg2_floor_noted" in {f.name for f in dataclasses.fields(wu.SchedulerWeightUpdaterManager)}

    def boom(*a, **k):
        raise RuntimeError("no store")

    monkeypatch.setattr(cg, "corridor_floor_mib", boom)
    monkeypatch.setattr(cg, "user_reserve_by_card", lambda: {})
    assert wu.SchedulerWeightUpdaterManager._weg2_corridor_floor_bytes(_mgr()) is None
