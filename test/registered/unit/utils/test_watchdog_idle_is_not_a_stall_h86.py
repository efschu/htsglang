"""H86: Leerlauf ist kein Stillstand -- der Scheduler-Watchdog darf eine Schlafphase nicht als Haenger zaehlen.

fnNV4f4 (25.09.2026): P schlief (WEG2-DORMANT) von 09:36:06 bis 09:40:59, 293 s. Beim Wake setzten PP1/PP2 ihren
cur_batch und warteten Sekunden auf die Proxy-Tensoren von PP0, bevor forward_ct (erst in _run_batch_forward) weiterzog.
Die naechste Pruefung sah den Zaehler unveraendert und die Uhr der letzten Bewegung von VOR dem Schlaf: "Scheduler
watchdog timeout (self.watchdog_timeout=300, self.soft=False) tripped_by=forward-counter-frozen(cur_batch set)" um
09:41:04, SIGQUIT an den Elternprozess, 60 s Coredump-Wartezeit, Abbau der ganzen P-Gruppe 7 s in die Nadel.

Der Wachhund misst nur, solange is_active() gilt; waehrend is_active() falsch ist, blieb watchdog_last_time stehen und
die Leerlaufzeit landete im ersten aktiven Vergleich. Der Test baut den Wachhund ohne Thread (Konstruktor umgangen),
soft=True (loggt statt SIGQUIT) und faehrt beide Faelle:
  - Leerlauf 4x Timeout, dann aktiv mit kurz eingefrorenem Zaehler (< Timeout), dann laufender Zaehler: KEIN Ausloesen.
  - Aktiv mit dauerhaft eingefrorenem Zaehler: Ausloesen (der echte Haenger bleibt erkannt).
"""
import threading
import time
import unittest
from unittest import mock

from sglang.srt.utils import watchdog as W

TIMEOUT = 0.4   # Pruefung alle TIMEOUT/2 = 0,2 s


class _Stop(Exception):
    pass


class _Scenario:
    """Zeitplan aus Abschnitten (Dauer, aktiv, Zaehler 'fest'|'laeuft'); nach dem letzten beendet _Stop den Wachhund.

    Der erste Abschnitt ist aktiv mit festem Zaehler: so kennt der Wachhund den Zaehlerstand VOR dem Schlaf, wie der
    echte Scheduler, dessen letzter Forward vor WEG2-DORMANT lag."""

    def __init__(self, segments):
        self.t0 = time.perf_counter()
        self.segments = segments

    def _where(self):
        t, start = time.perf_counter() - self.t0, 0.0
        for dur, active, mode in self.segments:
            if t < start + dur:
                return active, mode, t - start
            start += dur
        raise _Stop()

    def is_active(self):
        return self._where()[0]

    def get_counter(self):
        _, mode, dt = self._where()
        return 7 if mode == "fest" else 7 + 1 + int(dt / 0.05)


def _run(scenario):
    wd = W.WatchdogRaw.__new__(W.WatchdogRaw)
    wd.debug_name, wd.watchdog_timeout, wd.soft = "Scheduler", TIMEOUT, True
    wd.get_counter, wd.is_active = scenario.get_counter, scenario.is_active
    wd.dump_info, wd.describe_arm, wd.parent_process = None, None, mock.Mock()
    trips = []
    with mock.patch.object(W, "pyspy_dump_schedulers", lambda *a, **k: None), \
            mock.patch.object(W, "logger") as log:
        log.error.side_effect = lambda msg, *a, **k: trips.append(str(msg))

        def body():
            try:
                while True:
                    wd._watchdog_once()
            except _Stop:
                pass

        th = threading.Thread(target=body, daemon=True)
        th.start()
        th.join(timeout=10)
    return [m for m in trips if "watchdog timeout" in m]


class IdleIsNotAStall(unittest.TestCase):
    def test_a_long_sleep_then_a_slow_first_forward_does_not_trip(self):
        trips = _run(_Scenario([(0.3, True, "fest"), (4 * TIMEOUT, False, "fest"), (0.15, True, "fest"),
                                (1.0, True, "laeuft")]))
        self.assertEqual(trips, [], "Leerlauf wurde als Stillstand gezaehlt (fnNV4f4 09:41:04)")

    def test_a_real_stall_still_trips(self):
        trips = _run(_Scenario([(1.5, True, "fest")]))
        self.assertTrue(trips, "ein aktiver Scheduler ohne Fortschritt muss den Wachhund ausloesen")


if __name__ == "__main__":
    unittest.main()
