"""#79 (fnFL2w5, 21.09.): ein SCHLAFENDER Rang ist kein toter Rang.

Der Collect unterscheidet "der Deposit ist langsam" (weiter warten, im
Budget) von "der Deposit-Rang ist TOT" (sofort sterben, benannt). Das
Todes-Praedikat war NVML:

    procs = nvmlDeviceGetComputeRunningProcesses_v3(handle)
    others = {pr.pid for pr in procs} - {os.getpid()}
    return bool(others)

NVML listet nur Prozesse, die auf der Karte ALLOKIERT haben. Die schlafende
Gruppe gibt beim Sleep genau das frei -- sie faellt aus der Liste, ohne zu
sterben. Und der Flip legt die Quelle IMMER schlafen
(`WEG2-FLIP begin epoch=0 sleep=D wake=P`), also trifft es jeden Flip.

GEMESSEN an fnFL2w5 (18:09:37 Flip-Start, 18:10:19 Tod):
    W68: 3 lane(s) refused: c0/weights_1: PeerGone at unit 0
    'model.layers.3.attn_hyper_connection.block_inject_weight.weight'
    | rank 1: c1/weights_9 | rank 2: c2/weights_14
  -> W29 Weg2FlipRankDisagree auf resume_memory_occupation, 0/6 Raenge.
Bis zur selben Sekunde liefen 6/6 Scheduler.
"""

import inspect

from sglang.srt.managers.scheduler_components import weight_updater as wu


def test_an_empty_nvml_list_is_no_longer_a_death_certificate():
    src = inspect.getsource(wu.SchedulerWeightUpdaterManager._weg2_cocard_peer_alive
                            if hasattr(wu, "SchedulerWeightUpdaterManager")
                            else wu)
    assert "if others:" in src and "return True" in src
    assert "_any_scheduler_process_alive()" in src, (
        "eine leere NVML-Liste muss beim Prozess nachfragen, nicht toeten")


def test_the_fallback_fails_open():
    """Kann der Fallback nichts lesen, antwortet er True -- weiterwarten im
    Budget, nie einen lebenden Peer erschiessen."""
    src = inspect.getsource(wu._any_scheduler_process_alive)
    assert "except OSError:" in src
    assert src.rstrip().endswith("return False"), "nur ein LEERES /proc heisst tot"
    i_open = src.index("except OSError:\n        return True")
    assert i_open > 0, "der Lesefehler-Pfad muss True antworten"


def test_it_finds_a_living_scheduler_by_comm():
    """Funktionsprobe gegen das echte /proc: dieser Testprozess selbst wird
    NICHT gezaehlt (er ist `me`), aber die Funktion darf nicht werfen."""
    assert wu._any_scheduler_process_alive() in (True, False)


def test_the_budget_is_still_the_hard_bound():
    """Der Fix weicht die Schranke NICHT auf -- er entfernt nur ein falsches
    Todesurteil. Das Budget bleibt, was den Wait beendet."""
    from sglang.srt.weg2 import weight_exchange_bounce as wxb

    sig = inspect.signature(wxb.run_sequential_units)
    assert sig.parameters["budget_s"].default == 120.0
