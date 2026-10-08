# SPDX-License-Identifier: Apache-2.0
"""N3t 07:02:44-47Z: a client-gone abort of a PARKED request must leave the
front's D ledger too.

PDFLIP-CLIENT-GONE rid=pdflip-0-1 state=parked action=abort-d-park popped the rid
from ``_d_parked`` but NOT from ``D.outstanding``: the flip ledger
(outstanding minus parked) then counted it as RUNNING while D was idle ->
"W3 PdFlipDrainWitnessDisagreement -- rank idle, front still holds requests",
STOP, 6 x 503. handle_abort already pops both; the parked client-gone branch
now does the same.
"""

from __future__ import annotations

import pathlib

_SRC = (pathlib.Path(__file__).resolve().parents[4] / "python" / "flliper" /
        "srt" / "pdflip" / "front.py").read_text()


def test_parked_client_gone_pops_the_d_ledger():
    i = _SRC.find('elif state == "parked":')
    j = _SRC.find('action = f"abort-d-park status={code}"', i)
    block = _SRC[i:j]
    assert 'getattr(self, "_d_parked", {}).pop(rid, None)' in block
    assert 'self.groups["D"].outstanding.pop(rid, None)' in block
