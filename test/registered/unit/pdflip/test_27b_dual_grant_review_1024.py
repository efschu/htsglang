# SPDX-License-Identifier: Apache-2.0
"""Review DUAL-GRANT-1024 (desk/27b-dual-grant-1024 @ 8997446810): two danger-direction pins.

Metal: boot_weg2_dkr27bnvfp4dualbar1drei10241158_1ecda02a58_1024_115913.P.log, card 0 (5090, e773fd938f6d).
Verbatim lines (every number below is parsed from them):
"""

from __future__ import annotations

import os
import re
import tempfile
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.managers import corridor_guard as CG
from flliper.srt.pdflip import card_kv_ledger as K
from flliper.srt.pdflip import dual_p_kv_stage as S
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MIB = 1 << 20

#: verbatim, boot ..._115913.P.log
L_END = ("[2026-10-24 12:41:55 PP0] DUAL-TP3PP3 P-KV LEDGER-PHYS ok group=P ledger_free=2885812224 "
         "phys_free=2885812224 budget=4764860416 committed={'P': 0, 'D': 1879048192}")
L_IDLE = ("[2026-10-24 12:02:03 PP0] DUAL-TP3PP3 P-KV LEDGER-PHYS ok group=P ledger_free=4831838208 "
          "phys_free=8439070720 budget=4898947072 committed={'P': 0, 'D': 67108864}")
L_JOIN = ("[2026-10-24 12:01:43 PP0] DUAL-TP3PP3 P-KV JOIN card=e773fd938f6d boot_tokens=97920 "
          "contributed=1379926016 B top=266240 tokens step=4096 -- P keeps 0 mapped; its KV is the card pool's")
L_SHORT = ("[2026-10-24 12:08:38 PP0] #1530 GRANT-SHORT rid=pdflip-0-20 tokens=154149 level=154149 sum=154149 "
           "older=0 hold=none cards=[0:need=2231369728,have=2214592512,P=0,D=2684354560,demD=0,presP=0")
#: the CORRIDOR-FLOOR laws of card 0 (front lines of the same boot): P 1055 MEASURED-P, D 767 MEASURED-D
LAW_P, LAW_D = 1055, 767


def _i(pat, line):
    return int(re.search(pat, line).group(1))


BUDGET = _i(r"budget=(\d+)", L_END)
PHYS_END = _i(r"phys_free=(\d+)", L_END)
D_IDLE = _i(r"'D': (\d+)\}", L_IDLE)                     # D's idle commit on card 0 (64 MiB)
TOP = _i(r"top=(\d+)", L_JOIN)
STEP = _i(r"step=(\d+)", L_JOIN)
NEED_0820 = _i(r"0:need=(\d+)", L_SHORT)
LEVEL_0820 = -(-_i(r"level=(\d+)", L_SHORT) // STEP) * STEP
PER_STEP = NEED_0820 // (LEVEL_0820 // STEP)             # 58720256 B (= 14336 B/token) on card 0
#: --max-kv-per-request 262144 (27b-base.env; user order 06.10.: P carries the full context)
FULL = 262144


def _floor(mib, group):
    return CG.CorridorFloor(card_uuid="GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d", group=group,
                            transient_mib=int(mib), reserve_mib=0, source="MEASURED-" + group)


def _end_card(d_committed):
    """Card 0's ledger at 12:41:55 (budget of the line, gap 0), with D at ``d_committed``."""
    path = os.path.join(tempfile.mkdtemp(prefix="wkvrv"), "card")
    d = K.CardKvLedger(path, "D")
    p = K.CardKvLedger(path, "P")
    d.contribute(BUDGET, committed=int(d_committed))
    p.contribute(0, committed=0)
    assert K.peek(path).budget == BUDGET
    return path, d, p


def _checks(monkeypatch, led, phys, floor, n=S.PHYS_BOOK_AFTER):
    actor = types.SimpleNamespace(ledger=led)
    clock = [0.0]
    monkeypatch.setattr(S, "_now", lambda: clock[0])
    monkeypatch.setattr(S, "phys_free_bytes", lambda: int(phys))
    monkeypatch.setattr(CG, "corridor_floor_for_current_device", lambda group=None, **_k: floor)
    for _ in range(n):
        S.phys_check(actor, led.group)
        clock[0] += S.PHYS_CHECK_S


def _stages(path):
    return [{"step": STEP, "top": TOP, "bytes": [k * PER_STEP for k in range(TOP // STEP + 1)], "ledger": path}]


def test_numbers_of_the_lines():
    assert PER_STEP == 58720256 and PER_STEP * (LEVEL_0820 // STEP) == NEED_0820
    assert PHYS_END == _i(r"ledger_free=(\d+)", L_END), "gap 0 at 12:41:55"


def test_haenger_full_context_p_grant_stays_feasible_with_d_idle(monkeypatch):
    """HAENGER pin. The 12:41:55 card-0 pool (gap 0) with D at its idle commit: a full-context P grant
    (262144 tokens, --max-kv-per-request) must stay FEASIBLE -- otherwise it waits for ever (P never presses D,
    and even D at zero would not free it), holding every younger P request behind it
    (FLLIPER_PDFLIP_DUAL_GRANT_INFEASIBLE_SKIP and HEAD_BYPASS_FLOOR default off)."""
    path, d, _p = _end_card(D_IDLE)
    # gap 0 at the end of the boot: D's tick prices its law, P's idle tick prices P's (one-way ratchet)
    phys = PHYS_END + (1879048192 - D_IDLE)          # same card, D at idle: its decommit is free card memory
    _checks(monkeypatch, d, phys, _floor(LAW_D, "D"))
    _checks(monkeypatch, _p, phys, _floor(LAW_P, "P"))
    assert S.infeasible_cards(_stages(path), {}, FULL) == [], (
        "full-context P grant infeasible even with D at zero: pool %d B, need %d B"
        % (K.peek(path).budget, FULL // STEP * PER_STEP))


def test_card_law_does_not_depend_on_which_group_ticked_last(monkeypatch):
    """R11/card-uniform pin: one card, one law. D's tick (767) and P's idle tick (1055) on the SAME card must
    leave the same ledger -- today the price depends on whether P idled since the last cache growth."""
    out = []
    for groups in (("D",), ("D", "P")):
        path, d, p = _end_card(D_IDLE)
        phys = PHYS_END + (1879048192 - D_IDLE)
        for g in groups:
            _checks(monkeypatch, d if g == "D" else p, phys, _floor(LAW_D if g == "D" else LAW_P, g))
        out.append(K.peek(path).budget)
    assert out[0] == out[1], "budget after D only %d != after D then P %d" % tuple(out)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
