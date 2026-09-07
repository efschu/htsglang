"""#973: on a PP group without the #631 row carrier, PP0 must not withhold a
pass for its own pending prefetch while the followers take without waiting.

THE SPECIMEN (boot weg2ls1b3proof, 2026-09-07 07:21:30Z, no-flip PP=3 group,
P log /spinning/evidence-665-f1/boot_weg2_weg2ls1b3proof_0907_072050.P.log):
PP1/PP2 logged '#631 ROW AUTHORITY DISABLED: no pp_flip_counters side channel'
and '#969Z PREFETCH VERDICT NOT TAKEN HERE ... TAKE WITHOUT WAITING', received
the 14.6k-token request at 07:21:30, admitted it at once ('#969 EXTENT',
'#924D station=alloc') and blocked in the proxy receive
(`pp:0/recv_object[src=0] awaiting_size`). PP0 logged
'#1028 HICACHE-ROUND n=125 ... ongoing_prefetch=1', skipped the pass
(`prefetch_pending_pp0`, the #1066 wait), never admitted ('1 queued, 0
running, no prefill chunk'), and 120 s later its deferred chain-send join
raised '#973 RING COMMIT TIMEOUT'. Same wall on weg2s0 (4c0bbe407d) and
weg2s2 run 1. The #1066 wait was priced against the row carrier ("followers
execute PP0's decision without forming an opinion"); without the carrier the
followers DO form one, and PP0 must reach the same one.

WHY A PREDICATE-LEVEL TEST AND NOT A THREE-PROCESS RING (speed mode, one
targeted check with a stated reason): the divergence is a pure function of
(pp_size, pp_rank, row-carrier presence) evaluated inside
`get_new_batch_prefill`; the ring is only where it becomes visible 120 s
later. Pinning the predicate and the gate that consumes it is the check that
can fail on exactly this edit. The ring-level reproduction is the boot.

RED on d49e432cdf: `pp_row_carrier_present` does not exist and the gate keys
the PP0 wait on `pp_rank == 0` alone. GREEN after the fix.
"""

import importlib.util
import pathlib
import re
import types

import pytest

from sglang.srt.managers.pp_admission_congruence import (
    pp_row_authority_enabled,
    pp_row_carrier_present,
)


def _stand_in(pp_size: int, pp_rank: int, counters=None):
    sched = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_size=pp_size, pp_rank=pp_rank)
    )
    if counters is not None:
        sched.pp_flip_counters = counters
    return sched


def test_no_flip_form_has_no_carrier(monkeypatch):
    """The specimen form: PP=3, no pp_flip_counters -> PP0 may not withhold."""
    monkeypatch.delenv("SGLANG_PP_ROW_AUTHORITY", raising=False)
    pp0 = _stand_in(3, 0)
    assert pp_row_authority_enabled(pp0)  # the LAW is on ...
    assert pp_row_carrier_present(pp0) is False  # ... but cannot be executed


def test_carrier_present_keeps_the_flip_form_terms(monkeypatch):
    monkeypatch.delenv("SGLANG_PP_ROW_AUTHORITY", raising=False)
    assert pp_row_carrier_present(_stand_in(3, 0, counters=object())) is True


@pytest.mark.parametrize("env", ["0", "false"])
def test_row_authority_kill_switch_removes_the_carrier(monkeypatch, env):
    monkeypatch.setenv("SGLANG_PP_ROW_AUTHORITY", env)
    assert pp_row_carrier_present(_stand_in(3, 0, counters=object())) is False


def test_single_stage_has_no_carrier(monkeypatch):
    monkeypatch.delenv("SGLANG_PP_ROW_AUTHORITY", raising=False)
    assert pp_row_carrier_present(_stand_in(1, 0, counters=object())) is False


def test_admission_gate_keys_the_pp0_wait_on_the_carrier():
    """The consumer half: the `prefetch_pending_pp0` skip must sit under a
    branch guarded by the carrier, never under `pp_rank == 0` alone."""
    origin = importlib.util.find_spec("sglang.srt.managers.scheduler").origin
    src = pathlib.Path(origin).read_text()
    assert "pp_row_carrier_present(self)" in src
    # the licensed wait is the `elif` that follows the disarmed branch
    m = re.search(
        r"if self\.ps\.pp_rank == 0 and not _pp0_may_withhold:.*?"
        r"elif self\.ps\.pp_rank == 0:.*?_note_skip\(\"prefetch_pending_pp0\"",
        src,
        re.S,
    )
    assert m is not None, "PP0's prefetch wait is not gated on the row carrier"
    # and no other path reaches the skip
    assert src.count('_note_skip("prefetch_pending_pp0"') == 1
