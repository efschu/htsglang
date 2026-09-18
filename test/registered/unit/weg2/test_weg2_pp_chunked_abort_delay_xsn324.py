"""xsn324: an abort of the in-flight chunked request on PP stopped PP0 at
receipt while PP1 (abort one pass behind on the wire) had scheduled the next
chunk and waited for proxy tensors -- the ring stood. Stage r applies the
chunked abort pp_size-1-r passes after receipt."""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import pp_abort  # noqa: E402


def test_delay_per_stage():
    assert [pp_abort.chunked_abort_delay(3, r) for r in (0, 1, 2)] == [2, 1, 0]
    assert pp_abort.chunked_abort_delay(1, 0) == 0
    assert pp_abort.chunked_abort_delay(None, None) == 0
    assert pp_abort.chunked_abort_delay(2, 5) == 0


def test_countdown():
    assert pp_abort.countdown_step(0) == (True, 0)
    assert pp_abort.countdown_step(2) == (False, 1)
    assert pp_abort.countdown_step(1) == (False, 0)


def test_process_pending_chunked_abort_waits_the_delay_then_applies(monkeypatch):
    from sglang.srt.managers import scheduler as sched_mod

    sent = []
    monkeypatch.setattr(sched_mod, "prepare_abort", lambda req, why: setattr(req, "aborted_why", why))
    monkeypatch.setattr(sched_mod, "release_kv_cache", lambda *a, **k: None)
    req = types.SimpleNamespace(
        rid="weg2-6-3", req_pool_idx=None, kv_committed_freed=True, to_finish=1,
        finished=lambda: False,
        time_stats=types.SimpleNamespace(trace_ctx=types.SimpleNamespace(abort=lambda **k: None)),
    )
    fake = types.SimpleNamespace(
        _pending_chunked_abort_req=req, chunked_req=req, _pending_chunked_abort_delay=2,
        disaggregation_mode=None, enable_hicache_storage=False,
        tree_cache=types.SimpleNamespace(supports_mamba=lambda: False),
        ipc_channels=types.SimpleNamespace(send_to_tokenizer=types.SimpleNamespace(
            send_output=lambda obj, r: sent.append(obj.rid))),
    )
    f = sched_mod.Scheduler.process_pending_chunked_abort
    f(fake)
    assert fake.chunked_req is req and fake._pending_chunked_abort_delay == 1 and not sent
    f(fake)
    assert fake.chunked_req is req and fake._pending_chunked_abort_delay == 0 and not sent
    f(fake)
    assert fake.chunked_req is None and fake._pending_chunked_abort_req is None
    assert sent == ["weg2-6-3"] and req.aborted_why == "Aborted"


def test_abort_request_now_records_the_stage_delay():
    from sglang.srt.managers import scheduler as sched_mod
    src = open(sched_mod.__file__).read()
    i = src.index("self._pending_chunked_abort_req = chunked_req")
    assert "chunked_abort_delay(" in src[i:i + 600]
