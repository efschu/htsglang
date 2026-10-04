"""Q-699b W3 PARK-LAPSE RACE (NF y9n abl 76163d3aef, boot ...10032307, front 23:21:43.873Z).

D parked six (PARK-RUNNING 23:21:15.725; the front stamps ``t_park`` before the 2.59-s RPC,
~23:21:13.1). The quiesce waited 28 s (729 polls) for weg2-10-42, which D's #248h capacity
requeue ran to its end, and returned idle 30.7 s after the front's stamp: the front's part-B
clock (PARK_REQUEUE_S 30) had lapsed the five other parks back into its flip ledger, D's own
clock (stamped at the park itself) had not re-queued them -- D idle, holding them parked ->
``WEG2 STOP W3 Weg2DrainWitnessDisagreement -- rank idle, front still holds requests: front
ledger ['weg2-10-43', 'weg2-11-44', 'weg2-11-45', 'weg2-14-46', 'weg2-14-47']``. No request of
that ledger was aborted at the stop (the CLIENT-GONE lines follow it).

Pinned: idle D + a ledger made only of parks D confirmed = agreement (named line, the flip
carries them as parks); anything else keeps W3.
"""

import inspect

from sglang.srt.weg2 import front as F

LEDGER = ["weg2-10-43", "weg2-11-44", "weg2-11-45", "weg2-14-46", "weg2-14-47"]
T_PARK = 1791069673.1


def test_the_10032307_specimen_is_a_lapse_race():
    parked = {r: T_PARK for r in LEDGER + ["weg2-10-42"]}
    assert F.witness_verdict(len(LEDGER), True) == "rank idle, front still holds requests"
    assert F.park_lapse_race("D", LEDGER, parked, True) == LEDGER


def test_a_rid_never_parked_keeps_w3():
    parked = {r: T_PARK for r in LEDGER[:-1]}
    assert F.park_lapse_race("D", LEDGER, parked, True) == []


def test_not_idle_or_p_or_nothing_parked_keeps_the_old_verdict():
    parked = {r: T_PARK for r in LEDGER}
    assert F.park_lapse_race("D", LEDGER, parked, False) == []
    assert F.park_lapse_race("P", LEDGER, parked, True) == []
    assert F.park_lapse_race("D", LEDGER, {}, True) == []
    assert F.park_lapse_race("D", [], parked, True) == []


def test_the_flip_path_asks_it_before_stopping():
    src = inspect.getsource(F.Front)
    i_race = src.index("park_lapse_race(getattr(S, \"name\", \"\")")
    i_stop = src.index('self.do_stop("W3 Weg2DrainWitnessDisagreement"')
    assert i_race < i_stop
    assert "Q-699b W3-PARK-LAPSE-RACE" in src
