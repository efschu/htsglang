"""H25e (fnFL2x143, 24.09.): a rank whose draft tag carries nothing needs no source.

x143 (bccb555f9a = H25d, SGLANG_WEG2_DRAFT_ON_P=0) died at the first P->D
wake on the Form-A WORKERS: 10:14:17 TP1/TP2 ``WEG2-DRAFT-PARK
tag=weights_draft nothing to park on this rank (storages=0 tms_bytes=0)``,
then 10:15:04 TP1 and TP2 ``W4 Weg2WakeRefused: the draft worker loads from
...MTP-INT4-g32-albucino, not from ...Minachist`` -- while TP0, which parked,
passed (H25d).  The W4 exemption knew two sources (exchange, park image); a
shadow draft whose tag holds no byte needs none.  The verdict is the park's
own (recorded by ``_weg2_park_draft_at_sleep``), not a second guess.

x143 form: exchange armed + authoritative, draft NOT in the family, the NF MTP
checkpoint beside the target.
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


class _Target(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(4), requires_grad=False)


class _ShadowDraft(torch.nn.Module):
    """A Form-A worker's draft: the module tree exists, its tensors are META."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty(8, device="meta"), requires_grad=False)


class _RealDraft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.arange(8, dtype=torch.float32), requires_grad=False)


def _x143_rank(monkeypatch, *, draft, tms_bytes):
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
    monkeypatch.setattr(Manager, "_weg2_tag_bytes", lambda self, tag: tms_bytes, raising=True)
    drafter = SimpleNamespace(model=draft)
    monkeypatch.setattr(wu, "_weg2_drafter_of", lambda sched: drafter, raising=True)
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=_Target(), model_config=None))
    return Manager(tp_worker=worker, draft_worker=SimpleNamespace(model_runner=drafter),
                   tp_cpu_group=None, memory_saver_adapter=None,
                   flush_cache=lambda *a, **k: True, is_fully_idle=lambda *a, **k: True)


def test_x143_worker_with_a_shadow_draft_wakes_without_w4(monkeypatch):
    m = _x143_rank(monkeypatch, draft=_ShadowDraft(), tms_bytes=0)
    assert m._weg2_draft_park_armed()
    m._weg2_park_draft_at_sleep(credit=None)      # the real sleep-side call
    assert m._weg2_draft_not_carried == "storages=0 tms_bytes=0"
    assert m._weg2_draft_park is None             # nothing was parked
    assert m._weg2_wake_weight_carrier() == Manager.CARRIER_EXCHANGE


def test_x143_tp0_with_a_park_wakes_from_the_image_as_in_h25d(monkeypatch):
    draft = _RealDraft()
    m = _x143_rank(monkeypatch, draft=draft, tms_bytes=1631584256)
    park = dpk.DraftHostPark(pin=False)
    m._weg2_draft_park = park
    park.park(dpk.park_population(draft, None), tag="weights_draft",
              pause=lambda t: draft.w.data.zero_(), sync=lambda: None)
    assert m._weg2_draft_not_carried is None
    assert m._weg2_wake_weight_carrier() == Manager.CARRIER_EXCHANGE
    park.unpark_start(tag="weights_draft", resume=lambda t: None)
    park.join(overlap="admit")
    assert torch.equal(draft.w, torch.arange(8, dtype=torch.float32))


def test_mutant_a_rank_with_draft_bytes_and_no_park_still_refuses_w4(monkeypatch):
    """Draft bytes on the card, no image, no exchange: W4 still guards."""
    m = _x143_rank(monkeypatch, draft=_RealDraft(), tms_bytes=1631584256)
    with pytest.raises(Weg2WakeRefused) as exc:
        m._weg2_wake_weight_carrier()
    assert "W4" in str(exc.value) and "draft worker loads from" in str(exc.value)


def test_the_verdict_is_the_parks_own_and_is_cleared_by_a_real_park(monkeypatch):
    """A tag that DOES carry bytes never records 'not carried' (the sleep
    parks it instead); only the park's nothing-branch writes the verdict."""
    import inspect

    src = inspect.getsource(Manager._weg2_park_draft_at_sleep)
    nothing = src.index("nothing to park on this rank")
    assert src.index("self._weg2_draft_not_carried = (") < nothing
    assert src.index("self._weg2_draft_not_carried = None") > nothing
    carrier = inspect.getsource(Manager._weg2_wake_weight_carrier)
    assert "self._weg2_draft_not_carried is not None" in carrier
