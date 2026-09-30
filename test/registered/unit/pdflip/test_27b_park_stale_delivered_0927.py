"""STALE-DELIVERED (27B rc12k27 b23, 10:23:15, pdflip-24-100): a flip park opens a
new read cycle, so the #1324 delivered stamp of the previous cycle's read is
cleared -- otherwise the #1471 wake settle released a request whose own read
had answered zero ("SETTLE-TAIL delivered=79103 remainder=1405"), its
admission found nothing and the X gate refused it mid-stream (W50)."""
from __future__ import annotations

import os
import types
from collections import deque

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import d_park_runtime as rt  # noqa: E402
from flliper.srt.pdflip import d_seats as ds  # noqa: E402


@pytest.fixture(autouse=True)
def _group_d(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.delenv(ds.PARK_ENV, raising=False)


class _Batch:
    def __init__(self, reqs):
        self.reqs, self.batch_is_full = list(reqs), True

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, **_kw):
        pass

    def retract_all(self, server_args, offload_kv=True, retain=False):
        out, self.reqs = self.reqs, []
        return out


class _Sched:
    def __init__(self, running):
        self.running_batch = _Batch(running)
        self.waiting_queue, self.last_batch, self.enable_overlap = [], None, False
        self.result_queue, self.chunked_req, self.anchor_tails = deque(), None, []
        self.server_args, self.pdflip_dormant, self.noted = types.SimpleNamespace(), False, []

    def _969ad_note_retract(self, req, site):
        self.noted.append(req.rid)


def _req(rid, delivered):
    r = types.SimpleNamespace(rid=rid, kv_arrival_seq=1, origin_input_ids=[1] * 10, output_ids=[2] * 3,
                              is_fast_lane=False, spill_class=None)
    if delivered is not None:
        r._pdflip_store_delivered = delivered
    return r


def test_a_flip_park_clears_the_previous_cycles_delivered_stamp():
    from flliper.srt.managers.io_struct import PdFlipParkRunningReqInput
    from flliper.srt.managers.scheduler import _pdflip_store_short_remainder, _pdflip_store_tail_min_tokens

    a, b = _req("pdflip-24-100", 79103), _req("pdflip-46-165", None)
    s = _Sched([a, b])
    out = rt.park_running(s, PdFlipParkRunningReqInput(epoch=60, reason="immediate-over-x"))
    assert out.success and set(out.parked) == {"pdflip-24-100", "pdflip-46-165"}
    assert a._pdflip_store_delivered is None and not hasattr(b, "_pdflip_store_delivered")
    a.full_untruncated_fill_ids = [0] * 80508
    assert _pdflip_store_short_remainder(a) is None, "no stamp: the settle cannot release on it"
    assert _pdflip_store_tail_min_tokens(a) is None
