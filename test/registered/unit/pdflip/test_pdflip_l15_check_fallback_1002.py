# SPDX-License-Identifier: Apache-2.0
"""L15-FIX-CHECK-FALLBACK (N3r 02.10. 06:31:42Z): a failed wake sample check
falls back -- it never kills D.

N3r: "L15-CHECK rank=1 ok=0 bad=64", same on rank 2 -> decide() raised
L15CheckRefused on every rank -> resume_memory_occupation FAILED -> W29
PdFlipRankDisagree -> scheduler exception on TP0/1/2, W17 group dead.
decide() raises the refusal from the SAME gathered vote list on every rank,
so catching it at the one call site is group-uniform: the verdict becomes
"fallback" (the act drops the hold, plain restore) and a named line counts it.
"""

from __future__ import annotations

from types import SimpleNamespace

from flliper.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager as W,
)
from flliper.srt.pdflip.l15_wake_check import L15CheckRefused


def _self(raise_refused: bool, check=(0, 64, 0)):
    calls = []

    def _decide(wake_on, fp, chk, *, epoch):
        calls.append((wake_on, fp, chk, epoch))
        if raise_refused:
            raise L15CheckRefused("L15-CHECK REFUSED epoch=%d bad_ranks=1,2"
                                  % epoch)
        return "hold"

    s = SimpleNamespace(_l15_decide_wake_verdict=_decide,
                        _l15_wake_sample_check=lambda: check,
                        _l15_check_refusals=0)
    return s, calls


def test_refused_check_becomes_fallback_not_raise(caplog):
    s, calls = _self(True)
    v = W._l15_wake_check_and_decide(s, True, 123, epoch=2)
    assert v == "fallback"
    assert s._l15_check_refusals == 1
    assert len(calls) == 1
    assert any("L15-CHECK FALLBACK" in r.getMessage() for r in caplog.records)


def test_clean_check_keeps_the_verdict():
    s, _ = _self(False, check=(64, 0, 0))
    assert W._l15_wake_check_and_decide(s, True, 123, epoch=2) == "hold"
    assert s._l15_check_refusals == 0


def test_master_off_no_call():
    s, calls = _self(True)
    assert W._l15_wake_check_and_decide(s, False, 123, epoch=2) is None
    assert calls == []
