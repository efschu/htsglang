"""#81 (21.09.): das Lane-Budget muss VOR dem Front-Bound greifen.

Beide standen auf 120,0 -- `weight_exchange_bounce` und
`front.DRAIN_DEADLINE_DEFAULT_S` -- aus zwei Specs, ohne voneinander zu
wissen. Bei Gleichstand meldet die Front ihren STALL, waehrend der Lane-Wait
noch laeuft, und der Boot stirbt an der unspezifischen Meldung statt an der
Lane-Verweigerung, die Tensor und Lane NENNT.

GEMESSEN, dreimal hintereinander:
    w3: WEG2-FLIP STALL epoch=0 elapsed=129,2 s bound=120,0 s
    w5: ... elapsed=125,x s
    w7: ... elapsed=125,2 s bound=120,0 s
und die Lane-Verweigerung ("c1/weights_9: budget expired at unit 0
'model.layers.29.attn_hyper_connection.block_inject_weight.weight'") war nur
im Traceback zu finden.
"""

import inspect


def test_the_lane_gives_up_before_the_front_calls_a_stall():
    from sglang.srt.weg2 import front
    from sglang.srt.weg2 import weight_exchange_bounce as wxb

    assert wxb.SEQ_LANE_BUDGET_S < front.DRAIN_DEADLINE_DEFAULT_S, (
        f"Lane {wxb.SEQ_LANE_BUDGET_S}s gegen Front "
        f"{front.DRAIN_DEADLINE_DEFAULT_S}s -- bei Gleichstand oder mehr kann "
        f"die Lane nie zuerst greifen (w3/w5/w7 starben genau daran)")


def test_the_lead_is_big_enough_to_be_read():
    """Nicht knapp: die Meldung muss ankommen, bevor die Front abraeumt."""
    from sglang.srt.weg2 import front
    from sglang.srt.weg2 import weight_exchange_bounce as wxb

    lead = front.DRAIN_DEADLINE_DEFAULT_S - wxb.SEQ_LANE_BUDGET_S
    assert lead >= 15.0, f"nur {lead}s Vorlauf"


def test_both_entry_points_use_the_constant():
    """Zwei Signaturen trugen die 120,0 einzeln; eine Konstante haelt sie
    zusammen, sonst driftet die naechste Aenderung wieder auseinander."""
    from sglang.srt.weg2 import weight_exchange_bounce as wxb

    assert (inspect.signature(wxb.run_sequential_units)
            .parameters["budget_s"].default == wxb.SEQ_LANE_BUDGET_S)
    assert (inspect.signature(wxb.CrossSlotRendezvous.__init__)
            .parameters["budget_s"].default == wxb.SEQ_LANE_BUDGET_S)


def test_the_budget_stays_far_above_a_healthy_deposit():
    """Kein Toleranzband fuer langsame Deposits: die Quelle deponiert einen
    Chunk-Tag in 643-1412 ms (fnFL2w7), also zwei Groessenordnungen
    darunter. Wer 90 s wartet, wartet auf etwas, das nicht kommt."""
    from sglang.srt.weg2 import weight_exchange_bounce as wxb

    assert wxb.SEQ_LANE_BUDGET_S > 50 * 1.412
