"""H25d (fnFL2x142, 24.09.): the parked D draft wakes from its host image.

x142 (302651eb22) booted P without a draft, D's TP0 parked its draft at the
first sleep (``WEG2-DRAFT-PARK tag=weights_draft bytes=1596748576 ...
credited=yes``) -- and all three D ranks died at the first P->D wake in
``_weg2_wake_weight_carrier``: ``W4 Weg2WakeRefused: the draft worker loads
from ...MTP-INT4-g32-albucino, not from ...Minachist, and the backup-OFF wake
has exactly one model_path to give``.  The W4 exemption knew only one
non-disk source for a separate draft checkpoint (the exchange, i.e.
``weights_draft`` in the family); with the draft OFF the family, the refusal
fired BEFORE ``_weg2_unpark_draft_start`` could copy the image back.

Pinned in the x142 form (exchange armed + authoritative, draft NOT in the
family, a separate draft checkpoint): park -> wake carrier is the exchange ->
unpark restores the bytes from the image; without a park W4 still refuses.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import weg2_memory_saver as ms  # noqa: E402
from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from sglang.srt.managers.weg2_memory_saver import Weg2WakeRefused  # noqa: E402
from sglang.srt.weg2 import draft_park as dpk  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402

Manager = wu.SchedulerWeightUpdaterManager

#: conftest: this module pins its own draft form (the family is patched).
H25_OWN_DRAFT_FORM = True


class _FakeRunner:
    model = None
    model_config = None


class _FakeWorker:
    def __init__(self):
        self.model_runner = _FakeRunner()


def _x142_manager(monkeypatch):
    """x142's D rank: exchange+authoritative, draft OFF the family, the NF MTP
    checkpoint beside the target."""
    args = SimpleNamespace(
        enable_memory_saver=True,
        enable_weights_cpu_backup=False,
        enable_draft_weights_cpu_backup=False,
        model_path="/models/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist",
        speculative_draft_model_path="/models/Qwen3.8-Flash-Next-MTP-INT4-g32-albucino",
    )
    monkeypatch.setattr(Manager, "_weg2_server_args", lambda self: args, raising=True)
    monkeypatch.setattr(wx, "weights_cpu_backup_armed", lambda: False, raising=True)
    monkeypatch.setattr(wx, "exchange_armed", lambda: True, raising=True)
    monkeypatch.setattr(wx, "inject_authoritative", lambda: True, raising=True)
    monkeypatch.setattr(ms, "draft_tag_in_family", lambda: False, raising=True)
    return Manager(tp_worker=_FakeWorker(), draft_worker=_FakeWorker(), tp_cpu_group=None,
                   memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
                   is_fully_idle=lambda *a, **k: True)


class _Draft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.arange(8, dtype=torch.float32), requires_grad=False)


def test_x142_without_a_park_still_refuses_w4(monkeypatch):
    """No image on this rank = no host source: the refusal stays (and this is
    exactly what x142 hit, because the exemption did not know the park)."""
    m = _x142_manager(monkeypatch)
    assert m._weg2_draft_park is None
    with pytest.raises(Weg2WakeRefused) as exc:
        m._weg2_wake_weight_carrier()
    assert "W4" in str(exc.value) and "draft worker loads from" in str(exc.value)


def test_x142_after_a_park_the_wake_carrier_is_the_exchange_and_the_image_restores(monkeypatch):
    m = _x142_manager(monkeypatch)
    draft = _Draft()
    park = dpk.DraftHostPark(pin=False)
    m._weg2_draft_park = park

    def pause(tag):
        draft.w.data.zero_()          # the saver's unmap: the device bytes are gone

    park.park(dpk.park_population(draft, None), tag="weights_draft", pause=pause,
              sync=lambda: None)
    assert park.holds_image and park.parked
    # the first P->D wake asks the carrier BEFORE the family loop and again per
    # tag; neither may refuse now -- the draft's source is the image
    assert m._weg2_wake_weight_carrier() == Manager.CARRIER_EXCHANGE
    park.unpark_start(tag="weights_draft", resume=lambda t: None)
    # after the unpark started (the reload phase asks the carrier again)
    assert m._weg2_wake_weight_carrier() == Manager.CARRIER_EXCHANGE
    line = park.join(overlap="reload+cg_resume+kv_resume+admit")
    assert torch.equal(draft.w, torch.arange(8, dtype=torch.float32))
    assert line.startswith("WEG2-DRAFT-UNPARK tag=weights_draft bytes=32 ")
    assert "wait_ms=" in line and "overlap=reload+cg_resume+kv_resume+admit" in line
    # and every later wake of the process keeps its source
    assert m._weg2_wake_weight_carrier() == Manager.CARRIER_EXCHANGE


def test_an_empty_park_object_is_not_a_source(monkeypatch):
    """MUTANT direction: the exemption is keyed on the IMAGE, not on the park
    object merely existing (``nothing to park`` ranks never write one)."""
    m = _x142_manager(monkeypatch)
    m._weg2_draft_park = dpk.DraftHostPark(pin=False)
    assert not m._weg2_draft_park.holds_image
    with pytest.raises(Weg2WakeRefused):
        m._weg2_wake_weight_carrier()
