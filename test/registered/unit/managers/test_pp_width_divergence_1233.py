"""#1233: a PP width divergence is a named refusal on every form, and PP0's
rank-local corridor narrowing is disarmed where no row carrier exists.

THE SPECIMEN (boots weg2ls2b1 08:40:55Z and weg2ls2b2 08:58:37Z, 2026-09-07,
no-flip PP=3 group P of the Weg-2 form, P logs under /spinning/evidence-665-f1/
boot_weg2_weg2ls2b{1,2}_*.P.log): PP0 alone logged '[#656 CORRIDOR-ADMISSION]
NARROWED this prefill chunk from 4096 to 448 tokens' (785 MiB free on the
5090), planned '#969 EXTENT n=3 (8192, 8640, 448)' while PP1/PP2 planned
'(8192, 12288, 4096)'; PP1 logged '#1004 #631 WIDTH GUARD BYPASSED (flip off):
448 row(s) for 4096 token(s)' and died in chunk_gated_delta_rule with 'CUDA
error: an illegal memory access was encountered'.

WHY A PREDICATE-LEVEL TEST (speed mode, one targeted check with a stated
reason): the divergence is a pure function of (pp_size, pp_rank, row-carrier
presence) at the width decision, and the refusal is a pure function of
(received_rows, wanted_rows) at the receive; the CUDA fault is only where it
surfaced 30 layers later. The ring-level proof is the boot.

RED on 4de85fdb73: `refuse_pp_width_divergence` does not exist, and
`_corridor_granted_prefill_width` consults the gate on PP0 whenever
pp_size > 1. GREEN after the fix.
"""

import types

import pytest

from sglang.srt.managers.pp_admission_congruence import (
    PPWidthDivergenceRefused,
    pp_row_carrier_present,
    refuse_pp_width_divergence,
)


def test_width_divergence_is_refused_by_name():
    with pytest.raises(PPWidthDivergenceRefused) as ei:
        refuse_pp_width_divergence(448, 4096, "sender stamp rows=448")
    msg = str(ei.value)
    assert "W27" in msg and "448" in msg and "4096" in msg


def test_equal_width_passes():
    assert refuse_pp_width_divergence(4096, 4096, "x") is None


class _Gate:
    """Stand-in for PrefillAdmissionGate: would narrow every request."""

    def __init__(self, grant: int):
        self.grant = grant
        self.calls = 0

    def granted_width(self, requested: int) -> int:
        self.calls += 1
        return self.grant


def _stand_in(pp_size: int, pp_rank: int, counters=None, gate=None):
    sched = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_size=pp_size, pp_rank=pp_rank)
    )
    if counters is not None:
        sched.pp_flip_counters = counters
    if gate is not None:
        # ADMISSION_GATE_ATTR of corridor_admission.py
        sched.phase_flip_corridor_admission = gate
    return sched


def _width_fn():
    from sglang.srt.managers.scheduler import Scheduler

    return Scheduler._corridor_granted_prefill_width


def test_pp0_narrowing_disarmed_without_carrier(monkeypatch):
    """The specimen form: PP=3, PP0, no pp_flip_counters -> width unchanged,
    the gate is never consulted, the pass is counted."""
    monkeypatch.delenv("SGLANG_PP_ROW_AUTHORITY", raising=False)
    gate = _Gate(448)
    pp0 = _stand_in(3, 0, gate=gate)
    assert pp_row_carrier_present(pp0) is False
    assert _width_fn()(pp0, 4096) == 4096
    assert _width_fn()(pp0, 4096) == 4096
    assert gate.calls == 0
    assert pp0._corridor_width_disarmed_calls == 2


def test_pp0_narrowing_armed_with_carrier(monkeypatch):
    """The flip form (carrier present): PP0 narrows as before, byte-identical."""
    monkeypatch.delenv("SGLANG_PP_ROW_AUTHORITY", raising=False)
    gate = _Gate(448)
    pp0 = _stand_in(3, 0, counters=object(), gate=gate)
    assert pp_row_carrier_present(pp0) is True
    assert _width_fn()(pp0, 4096) == 448
    assert gate.calls == 1


def test_followers_never_narrow(monkeypatch):
    monkeypatch.delenv("SGLANG_PP_ROW_AUTHORITY", raising=False)
    gate = _Gate(448)
    pp1 = _stand_in(3, 1, gate=gate)
    assert _width_fn()(pp1, 4096) == 4096
    assert gate.calls == 0
