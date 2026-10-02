"""LEAST-DEFICIT flip order (NF y6u, 01.10. 23:47:31.299Z and 23:47:47.729Z,
D->P epochs 2->3 and 4->5): when no order is funded at the measured free, the
front no longer keeps the given order that runs into its credit cycle.

Front 23:47:31.299 `WEG2-FLIP-ORDER epoch=2 src=D driver_free={0: 1281, 1: 2212,
2: 1269}` with the 17-tag order below and the verdict "W126 ... NO order funds
the wake (stuck after 1 of 17 tags) -- given order kept". On the metal that
order cycled exactly as the model's uplifted runs predict (D log 23:47:33
`CYCLE-SPILL ... sleeper1 blocked depositing weights_15 to waker2 -> sleeper2
blocked depositing weights_11 to waker1`, deposit stall 1.6-1.9 s, gathered legs
3396 ms against 1314-1447 ms uncycled). Epochs 6/8 (free {0: 2591, 1: 4236,
2: 2413} / {0: 3003, 1: 4770, 2: 2573}) got the cycle-breaking reorder and
did not stall.

The model is pessimistic there for a reason this commit does NOT fix (see
`test_the_pessimism_is_the_stale_release_table`): the planner's release table
holds D's tag sizes from the boot start (906/927/1208-1250 MiB per tag on
card0/2/1), while D's tags had grown to 1064/1066/1450 MiB (D log 23:47:31
WEG2-DC-BREAKDOWN stage=release) -- the growth is exactly what lowered
driver_free, and the model books the loss without the matching release."""
from __future__ import annotations

import os
from dataclasses import replace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import wake_credit as wc  # noqa: E402

#: the front's --wake-credit-plan of y6u, "D->P" (front.log WEG2-LAUNCH argv)
Y6U_TABLE = (
    [{'card': 1,
      'demand': {'weights': 630.0,
                 'weights_0': 1753.982177734375,
                 'weights_1': 1761.982177734375,
                 'weights_2': 1701.982177734375,
                 'weights_3': 1701.982177734375,
                 'weights_4': 1707.982177734375,
                 'weights_5': 1701.982177734375,
                 'weights_6': 1701.982177734375,
                 'weights_7': 1701.982177734375,
                 'weights_8': 1707.982177734375,
                 'weights_9': 1136.65478515625},
      'oncard': {'weights': 615.7226715087891,
                 'weights_0': 350.81531362213707,
                 'weights_1': 311.45124957480846,
                 'weights_2': 311.7557399490749,
                 'weights_3': 311.73686545924147,
                 'weights_4': 318.5239069391448,
                 'weights_5': 311.7557399490749,
                 'weights_6': 311.7557399490749,
                 'weights_7': 311.73686545924147,
                 'weights_8': 318.5239069391448,
                 'weights_9': 205.56511230414995},
      'release': {'weights': 1258.0,
                  'weights_0': 1249.7533416748047,
                  'weights_1': 1275.7533416748047,
                  'weights_10': 1207.7533416748047,
                  'weights_11': 1211.7533416748047,
                  'weights_12': 1217.7533416748047,
                  'weights_13': 1207.7533416748047,
                  'weights_14': 1207.7533416748047,
                  'weights_15': 1211.7533416748047,
                  'weights_2': 1207.7533416748047,
                  'weights_3': 1211.7533416748047,
                  'weights_4': 1217.7533416748047,
                  'weights_5': 1207.7533416748047,
                  'weights_6': 1207.7533416748047,
                  'weights_7': 1211.7533416748047,
                  'weights_8': 1217.7533416748047,
                  'weights_9': 1207.7533416748047,
                  'weights_draft': 1556.0},
      'sleeper': 'D TP0',
      'waker': 'P PP0'},
     {'card': 0,
      'demand': {'weights_10': 3148.7845306396484,
                 'weights_11': 3084.7845306396484,
                 'weights_12': 3090.7845306396484,
                 'weights_13': 1030.9281768798828,
                 'weights_9': 1036.9281768798828},
      'oncard': {'weights_10': 524.2376070374077,
                 'weights_11': 524.2376070374077,
                 'weights_12': 524.2376070374077,
                 'weights_13': 174.74586901246923,
                 'weights_9': 174.74586901246923},
      'release': {'weights': 2.0,
                  'weights_0': 906.2466583251953,
                  'weights_1': 1006.2466583251953,
                  'weights_10': 906.2466583251953,
                  'weights_11': 906.2466583251953,
                  'weights_12': 906.2466583251953,
                  'weights_13': 906.2466583251953,
                  'weights_14': 906.2466583251953,
                  'weights_15': 906.2466583251953,
                  'weights_2': 906.2466583251953,
                  'weights_3': 906.2466583251953,
                  'weights_4': 906.2466583251953,
                  'weights_5': 906.2466583251953,
                  'weights_6': 906.2466583251953,
                  'weights_7': 906.2466583251953,
                  'weights_8': 906.2466583251953,
                  'weights_9': 906.2466583251953},
      'sleeper': 'D TP1',
      'waker': 'P PP1'},
     {'card': 2,
      'demand': {'weights': 650.0,
                 'weights_13': 2654.743896484375,
                 'weights_14': 4016.1158447265625,
                 'weights_15': 3956.1158447265625},
      'oncard': {'weights': 0.6262161254882812,
                 'weights_13': 381.2486439526081,
                 'weights_14': 571.8729659289122,
                 'weights_15': 571.8729659289122},
      'release': {'weights': 2.0,
                  'weights_0': 927.4888610839844,
                  'weights_1': 1027.4888610839844,
                  'weights_10': 927.4888610839844,
                  'weights_11': 927.4888610839844,
                  'weights_12': 927.4888610839844,
                  'weights_13': 927.4888610839844,
                  'weights_14': 927.4888610839844,
                  'weights_15': 927.4888610839844,
                  'weights_2': 927.4888610839844,
                  'weights_3': 927.4888610839844,
                  'weights_4': 927.4888610839844,
                  'weights_5': 927.4888610839844,
                  'weights_6': 927.4888610839844,
                  'weights_7': 927.4888610839844,
                  'weights_8': 927.4888610839844,
                  'weights_9': 927.4888610839844},
      'sleeper': 'D TP2',
      'waker': 'P PP2'}]
)

#: front 23:47:31.299 pause_order (= 23:47:47.729), 17 tags
Y6U_ORDER = ["weights_0", "weights_9", "weights_14", "weights_1", "weights_10", "weights_15",
             "weights_2", "weights_11", "weights_3", "weights_12", "weights_4", "weights_13",
             "weights_5", "weights_6", "weights_7", "weights_8", "weights"]
#: the waker's corridor floors as the front's own card lines print them
Y6U_FLOOR = {1: 1055.0, 0: 1095.0, 2: 858.0}
FREE_E2 = {0: 1281.0, 1: 2212.0, 2: 1269.0}
FREE_E4 = {0: 1315.0, 1: 2444.0, 2: 1133.0}
#: D log 23:47:31 WEG2-DC-BREAKDOWN stage=release, tms_resident per weights_N tag
LIVE_RELEASE_E2 = {1: 1450.0, 0: 1064.0, 2: 1066.0}
#: the pair that cycled on the metal
CYCLED = ("weights_15", "weights_11")


def _cards(free, live=None):
    out = []
    for pc in Y6U_TABLE:
        c = int(pc["card"])
        rel = dict(pc["release"])
        if live is not None:
            rel = {k: (live[c] if k.startswith("weights_") and k != "weights_draft" else v)
                   for k, v in rel.items()}
        out.append(wc.WakeCard(card=c, free_mib=float(free[c]), floor_mib=Y6U_FLOOR[c],
                               release_mib=rel, demand_mib=dict(pc["demand"]),
                               oncard_mib=dict(pc.get("oncard") or {})))
    return out


def _card_free(nvml, free):
    from sglang.srt.weg2.front import CardFree

    return CardFree(nvml_index=nvml, uuid="u%d" % nvml, free_mib=free, reserved_mib=0)


def _front(free):
    from sglang.srt.weg2 import front

    return front.credit_pause_order(
        list(Y6U_ORDER), "rr", {"D->P": Y6U_TABLE}, "D", "P",
        [_card_free(c, f) for c, f in sorted(free.items())],
        floor_of=lambda uuid: Y6U_FLOOR[int(uuid[1:])])


def test_y6u_front_no_longer_keeps_the_cycling_order():
    for free in (FREE_E2, FREE_E4):
        new, why = _front(free)
        # base 103233b7e0: the given order, "given order kept"
        assert new != Y6U_ORDER, why
        assert sorted(new) == sorted(Y6U_ORDER) and new[-1] == "weights"
        assert "LEAST-DEFICIT" in why and "NO order funds the wake" in why
        # the model still names the measured state honestly (W126 text stays)
        assert wc.REFUSAL_CODE in why


def test_the_chosen_order_is_funded_where_the_metal_was_and_the_given_one_is_not():
    """Against y6u's LIVE tag sizes (the state the metal actually ran in) the
    least-deficit order funds every step; the given order does not -- it ends
    in a cycle, which on the metal was the W109b spill."""
    new, _why = _front(FREE_E2)
    live = _cards(FREE_E2, LIVE_RELEASE_E2)
    assert wc.simulate(new, live).complete
    given = wc.simulate(Y6U_ORDER, live)
    assert not given.complete and given.chain


def test_least_deficit_needs_less_uplift_than_the_given_order():
    cards = _cards(FREE_E2)
    new, up = wc.least_deficit_order(Y6U_ORDER, cards)
    assert up <= 512
    lifted = [replace(c, free_mib=c.free_mib + up) for c in cards]
    assert wc.simulate(new, lifted).complete
    assert not wc.simulate(Y6U_ORDER, lifted).complete
    # the given order is funded only from +3072 on
    assert not wc.simulate(Y6U_ORDER, [replace(c, free_mib=c.free_mib + 2048)
                                       for c in cards]).complete
    # and at y6u's measured state it cycles on the metal pair once lifted past the
    # first shortage (+1536: sleeper@card2 weights_11 -> waker@card0, card0 weights_15)
    run = wc.simulate(Y6U_ORDER, [replace(c, free_mib=c.free_mib + 1536) for c in cards])
    assert {t for _s, _d, t in run.chain} == set(CYCLED)
    # the chosen order puts weights_15 behind weights_11's partner tags, never first
    assert new.index("weights_15") > new.index("weights_2")


def test_a_funded_or_greedy_order_is_untouched_by_the_fallback():
    for free in ({0: 2591.0, 1: 4236.0, 2: 2413.0}, {0: 4655.0, 1: 5588.0, 2: 4261.0}):
        cards = _cards(free)
        assert (wc.credit_order(Y6U_ORDER, cards, least_deficit=True)
                == wc.credit_order(Y6U_ORDER, cards))


def test_the_planner_riegel_and_the_switch_keep_the_given_order():
    cards = _cards(FREE_E2)
    order, run, why = wc.credit_order(Y6U_ORDER, cards)
    assert order == Y6U_ORDER and not run.complete and "given order kept" in why
    with envs.SGLANG_WEG2_FLIP_ORDER_LEAST_DEFICIT.override(False):
        new, why = _front(FREE_E2)
    assert new == Y6U_ORDER and "given order kept" in why
    assert envs.SGLANG_WEG2_FLIP_ORDER_LEAST_DEFICIT.get() is True


def test_the_pessimism_is_the_stale_release_table():
    """Separate finding, not fixed here: with D's live tag sizes the plain
    credit order already finds a funded reorder at the measured free."""
    order, run, why = wc.credit_order(Y6U_ORDER, _cards(FREE_E2, LIVE_RELEASE_E2))
    assert run.complete and order != Y6U_ORDER and "reordered" in why
    plan_rel = {int(pc["card"]): pc["release"]["weights_9"] for pc in Y6U_TABLE}
    assert all(LIVE_RELEASE_E2[c] > plan_rel[c] * 1.1 for c in plan_rel)
