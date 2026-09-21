"""#77 (21.09.): die Bounce-Terme gehoeren an den AUSTAUSCH-Arm, nicht an
den Host-Pin.

fnFL2w3 starb daran. Die Kette, jede Stufe gemessen:

    WEG2-XCHG-DEPOSIT-SKIPPED group=D rank=1: the arm published no bounce
    terms (SGLANG_WEG2_XCHG_BOUNCE_TERMS), so this rank cannot size a deposit

  -> von 16 Chunk-Tags deponierte D GENAU EINEN (weights_0, 643/682/693 ms je
     Rang); fuer die anderen 15 steht `deposit_ms=0` neben `pause_ms=1`.
  -> P wartete sein volles Lane-Budget auf `unit 0` von `weights_9`:
     "W68: 3 lane(s) of this leg refused: p0/weights_9: budget expired at
     unit 0". Das Budget ist 120,0 s (weight_exchange_bounce.py) und der
     Front-Bound ebenfalls 120,0 s -- der Lane-Wait kann strukturell nie vor
     dem Front-STALL aufgeben.
  -> WEG2-FLIP STALL epoch=0 elapsed=129,2 s, dann W29 Weg2FlipRankDisagree
     auf resume_memory_occupation, 0/6 Raenge.

Die Wurzel ist EIN Praedikat fuer ZWEI Fragen. `xchg_bounce_arm_pins_host`
beantwortet "pinnt dieser Arm Host-Bytes?" -- bei `oncard=ipc` korrekt
FALSCH. Die Publikation der Terme hing daran, obwohl der Rang sie in JEDEM
Austausch-Arm braucht.
"""

import inspect

from sglang.srt.weg2 import launcher
from sglang.srt.weg2 import weight_exchange_transport as wxt


def test_the_host_pin_predicate_still_answers_only_its_own_question():
    """Es wird NICHT geweitet -- die Ledger-Frage bleibt eng, sonst zahlt ein
    ipc-Boot Host-Bytes, die er nie allokiert."""
    assert launcher.xchg_bounce_arm_pins_host("exchange",
                                              wxt.ONCARD_MODE_HOST) is True
    assert launcher.xchg_bounce_arm_pins_host("exchange", "ipc") is False
    # `ring` erzeugt gar keine Region -- unabhaengig vom oncard-Modus.
    assert launcher.xchg_bounce_arm_pins_host(launcher.WEIGHT_SOURCE_DEFAULT,
                                              wxt.ONCARD_MODE_HOST) is False


def test_the_terms_now_hang_on_the_exchange_arm_not_on_the_host_pin():
    """Der Fix selbst: die Publikation steht hinter `_xchg_armed`, und das
    ist wahr fuer JEDEN Arm ausser `ring`."""
    src = inspect.getsource(launcher.main)
    assert "_xchg_armed = str(ns.weg2_weight_source) != WEIGHT_SOURCE_DEFAULT" in src
    i_armed = src.index("_xchg_armed = ")
    i_pub = src.index("bounce_terms_for_ranks, _widest_line")
    i_pin = src.find("xchg_bounce_arm_pins_host(ns.weg2_weight_source", i_armed)
    assert i_armed < i_pub, "die Terme muessen hinter dem Austausch-Arm stehen"
    assert i_pin == -1 or i_pin > i_pub, (
        "der Host-Pin darf die Publikation nicht mehr torwaechtern")


def test_a_ring_boot_still_publishes_nothing():
    """Die Gegenrichtung, die bleiben muss: ohne Austausch gibt es nichts zu
    deponieren, also auch keinen Term -- `read_published_terms` antwortet
    None und die Naht verweigert BENANNT statt zu defaulten."""
    from sglang.srt.weg2 import xchg_bounce

    assert xchg_bounce.read_published_terms("") is None
    assert launcher.xchg_bounce_arm_pins_host(launcher.WEIGHT_SOURCE_DEFAULT,
                                              "ipc") is False


def test_the_lane_budget_is_not_the_front_bound():
    """Der zweite Teil desselben Todes: ein Lane-Budget, das genauso lang ist
    wie der Front-Bound, kann nie zuerst aufgeben -- der Boot stirbt am Stall
    statt an einer benannten Lane-Verweigerung. Dieser Test haelt die Zahl
    fest, die fnFL2w3 gemessen hat, damit der Gleichstand auffaellt."""
    from sglang.srt.weg2 import weight_exchange_bounce as wxb

    sig = inspect.signature(wxb.run_sequential_units)
    budget = sig.parameters["budget_s"].default
    assert budget == 120.0, (
        f"Lane-Budget {budget}s -- fnFL2w3 mass 120,0 s gegen einen "
        f"Front-Bound von ebenfalls 120,0 s (STALL bei elapsed=129,2 s). "
        f"Aendert sich eine der beiden Zahlen, gehoert die andere geprueft.")
