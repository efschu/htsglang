# SPDX-License-Identifier: Apache-2.0
"""#1490: a kv_cache resume that did not happen must be refused BY NAME, not
believed.

THE SPECIMEN (boot weg2xsn408, 2026-09-20 17:57:21Z; boot weg2xsn406 68
minutes earlier is the same failure). The wake's own arithmetic printed the
shortfall and then ignored it:

    TP0  WEG2-WAKE-KV-FIRST LATE free=13355 MiB floor=767 MiB need=13024 MiB
    TP1  WEG2-WAKE-KV-FIRST LATE free=5974  MiB floor=700 MiB need=6904  MiB
                             (... free - floor - margin < kv)

`_weg2_wake_kv_first_ok` uses that answer ONLY to pick EARLY vs LATE, then
resumes LATE anyway. The hook rolled the tag back and could not say so:

    [core.cpp] WEG2-TMS-RESUME REFUSED tag=kv_cache rc=2 (out of memory)
               failed_alloc=30/41 rolled_back_bytes=5442109440
               tag_bytes=7239368704 -- every allocation of the tag is PAUSED again
    [torch_memory_saver.cpp] tms_resume failed rc=2 tag=kv_cache (void ABI: exiting)

Python was handed a normal return and zeroed the pools it believed it had
remapped. TP0 and TP1 died with NO Python traceback at all; TP2, whose card
did fund its pool, lived on to report them gone.

Hermetic: the decisions are pure functions and the adapter is driven with a
stub saver. No CUDA, no scheduler.
"""

from __future__ import annotations

import pytest

from sglang.srt.weg2.wake_kv import (
    RESUME_LANDED_MIN_BYTES,
    kv_resume_fit_refusal,
    resume_landed,
)

MiB = 1 << 20


# --- the fit test that was computed and thrown away -------------------------


def test_tp1_of_xsn408_is_refused_by_the_numbers_it_printed():
    why = kv_resume_fit_refusal(5974 * MiB, 6904 * MiB, 700 * MiB)
    assert why is not None
    assert "free=5974 MiB" in why and "need=6904 MiB" in why
    assert "short by 930 MiB" in why


def test_the_corridor_floor_is_reported_and_never_subtracted():
    """"Keine Korridor-Reserve, nie" -- a reserve may shape a plan, it may
    never be the reason a wake is refused. TP0 of xsn408 fits by 331 MiB
    only because the 767 MiB floor is NOT taken off the budget."""
    assert kv_resume_fit_refusal(13355 * MiB, 13024 * MiB, 767 * MiB) is None
    why = kv_resume_fit_refusal(5974 * MiB, 6904 * MiB, 700 * MiB)
    assert "floor=700 MiB is reported, not subtracted" in why


def test_an_exact_fit_is_not_a_refusal():
    assert kv_resume_fit_refusal(4096 * MiB, 4096 * MiB, 0) is None
    assert kv_resume_fit_refusal(4095 * MiB, 4096 * MiB, 0) is not None


@pytest.mark.parametrize(
    "free,need",
    [(None, 4096 * MiB), (4096 * MiB, None), (4096 * MiB, 0), (None, None)],
)
def test_an_absent_probe_is_never_a_refusal(free, need):
    assert kv_resume_fit_refusal(free, need, 0) is None


# --- the resume that returned success and had not happened ------------------


def test_a_rolled_back_resume_did_not_land():
    """TP0 of xsn408: 13 GiB claimed, device free unmoved."""
    assert resume_landed(13355 * MiB, 13355 * MiB, 13024 * MiB) is False


def test_a_real_resume_landed():
    assert resume_landed(13355 * MiB, 331 * MiB, 13024 * MiB) is True


def test_a_partial_map_is_still_not_landed():
    """failed_alloc=30/41 rolled everything back; a resume that mapped a
    fraction is a refusal, not a success."""
    assert resume_landed(8000 * MiB, 7800 * MiB, 6904 * MiB) is False


def test_concurrent_slack_does_not_turn_a_landed_resume_into_a_refusal():
    """One-sided by construction: another thread taking or giving a few
    hundred MiB during the call must not flip the verdict."""
    assert resume_landed(13355 * MiB, 500 * MiB, 13024 * MiB) is True
    assert resume_landed(13355 * MiB, 100 * MiB, 13024 * MiB) is True


@pytest.mark.parametrize(
    "before,after,need",
    [
        (None, 1 * MiB, 4096 * MiB),
        (1 * MiB, None, 4096 * MiB),
        (4096 * MiB, 4096 * MiB, None),
        (4096 * MiB, 4096 * MiB, RESUME_LANDED_MIN_BYTES - 1),
    ],
)
def test_undecidable_is_none_and_never_false(before, after, need):
    """An absence must be reported as an absence. A None here is what keeps
    the adapter from raising on a rank that cannot measure."""
    assert resume_landed(before, after, need) is None


# --- the adapter that now verifies ------------------------------------------


class _StubSaver:
    def __init__(self):
        self.resumed = []

    def resume(self, tag):
        self.resumed.append(tag)
        return "saver-return"

    def pause(self, tag):
        return None


def _adapter(monkeypatch, free_before, free_after, tag_bytes):
    from sglang.srt.utils import torch_memory_saver_adapter as tms

    saver = _StubSaver()
    monkeypatch.setattr(tms, "_memory_saver", saver, raising=False)
    reads = iter([free_before, free_after])
    monkeypatch.setattr(tms, "_device_free_bytes", lambda: next(reads))
    a = tms._TorchMemorySaverAdapterReal()
    monkeypatch.setattr(type(a), "tag_bytes", lambda self, tag: tag_bytes)
    return tms, a, saver


def test_adapter_raises_by_name_when_the_resume_did_not_land(monkeypatch):
    tms, a, saver = _adapter(monkeypatch, 5974 * MiB, 5974 * MiB, 6904 * MiB)
    with pytest.raises(tms.Weg2TmsResumeRefused) as e:
        a.resume("kv_cache")
    msg = str(e.value)
    assert "W119 Weg2TmsResumeRefused" in msg  # renumbered on this line: W114 is Weg2FlipHostPoolDoubled here
    assert "tag=kv_cache" in msg
    assert "tag_bytes=6904 MiB" in msg
    assert "delta=0 MiB" in msg
    assert saver.resumed == ["kv_cache"], "the saver is still called -- we verify, not predict"


def test_adapter_returns_the_savers_value_on_a_landed_resume(monkeypatch):
    tms, a, saver = _adapter(monkeypatch, 13355 * MiB, 331 * MiB, 13024 * MiB)
    assert a.resume("kv_cache") == "saver-return"


def test_adapter_stays_silent_when_it_cannot_measure(monkeypatch):
    """No probe -> no verdict. A guard that can break bring-up is worse than
    the gap it closes."""
    tms, a, saver = _adapter(monkeypatch, None, None, 13024 * MiB)
    assert a.resume("kv_cache") == "saver-return"


def test_adapter_stays_silent_when_the_tag_size_is_unknown(monkeypatch):
    tms, a, saver = _adapter(monkeypatch, 4096 * MiB, 4096 * MiB, None)
    assert a.resume("kv_cache") == "saver-return"


def test_adapter_survives_a_tag_bytes_probe_that_raises(monkeypatch):
    from sglang.srt.utils import torch_memory_saver_adapter as tms

    saver = _StubSaver()
    monkeypatch.setattr(tms, "_memory_saver", saver, raising=False)
    monkeypatch.setattr(tms, "_device_free_bytes", lambda: 4096 * MiB)
    a = tms._TorchMemorySaverAdapterReal()

    def _boom(self, tag):
        raise OSError("no such symbol")

    monkeypatch.setattr(type(a), "tag_bytes", _boom)
    assert a.resume("kv_cache") == "saver-return"


def test_the_refusal_is_a_runtime_error_subclass(monkeypatch):
    """Callers that only catch the base class still stop; callers that name it
    can keep the rank alive and dormant."""
    from sglang.srt.utils import torch_memory_saver_adapter as tms

    assert issubclass(tms.Weg2TmsResumeRefused, RuntimeError)


def test_device_free_bytes_never_raises(monkeypatch):
    from sglang.srt.utils import torch_memory_saver_adapter as tms

    import torch

    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: (_ for _ in ()).throw(RuntimeError("gone"))
    )
    assert tms._device_free_bytes() is None
