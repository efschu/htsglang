"""weg2xsn264 (17.09.2026): the first P->D wake of the DFLASH form refused
W4 on all three D ranks -- "the draft worker loads from <DFlash2 path>, not
from <target path>, and the backup-OFF wake has exactly one model_path to
give". That refusal predates the exchange: under `--weg2-weight-source
exchange --weg2-xchg-inject authoritative` with `weights_draft` in the
weights family, the draft's bytes come from the peer group through the same
legs as every other tag (disk only as the draft's own fallback), so a
separate draft checkpoint is legitimate. Outside that arm the refusal stays.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import weg2_memory_saver as ms  # noqa: E402
from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from sglang.srt.managers.weg2_memory_saver import Weg2WakeRefused  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402

Manager = wu.SchedulerWeightUpdaterManager


class _FakeRunner:
    model = None
    model_config = None


class _FakeWorker:
    def __init__(self):
        self.model_runner = _FakeRunner()


def _manager(monkeypatch, *, armed: bool, in_family: bool):
    args = SimpleNamespace(
        enable_memory_saver=True,
        enable_weights_cpu_backup=False,
        enable_draft_weights_cpu_backup=False,
        model_path="/models/target-lued",
        speculative_draft_model_path="/models/dflash2-draft-lued",
    )
    monkeypatch.setattr(Manager, "_weg2_server_args", lambda self: args, raising=True)
    monkeypatch.setattr(wx, "weights_cpu_backup_armed", lambda: False, raising=True)
    monkeypatch.setattr(wx, "exchange_armed", lambda: armed, raising=True)
    monkeypatch.setattr(wx, "inject_authoritative", lambda: armed, raising=True)
    monkeypatch.setattr(ms, "draft_tag_in_family", lambda: in_family, raising=True)
    return Manager(tp_worker=_FakeWorker(), draft_worker=_FakeWorker(), tp_cpu_group=None,
                   memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
                   is_fully_idle=lambda *a, **k: True)


def test_a_separate_draft_checkpoint_is_carried_by_the_exchange(monkeypatch):
    m = _manager(monkeypatch, armed=True, in_family=True)
    assert m._weg2_wake_weight_carrier() == Manager.CARRIER_EXCHANGE


def test_without_the_exchange_the_separate_draft_checkpoint_still_refuses_w4(monkeypatch):
    m = _manager(monkeypatch, armed=False, in_family=False)
    with pytest.raises(Weg2WakeRefused) as exc:
        m._weg2_wake_weight_carrier()
    assert "W4" in str(exc.value) and "draft worker loads from" in str(exc.value)


def test_an_exchange_that_does_not_carry_the_draft_still_refuses_w4(monkeypatch):
    """MUTANT direction: the exemption is keyed on the draft being IN the
    family, not on the exchange being armed at all."""
    m = _manager(monkeypatch, armed=True, in_family=False)
    with pytest.raises(Weg2WakeRefused):
        m._weg2_wake_weight_carrier()
