"""H86b: ein spaet einschaltender Arm darf durch den H86-Reset nicht langsamer werden.

H86 (b191dc8bb0) setzt watchdog_last_time zurueck, solange is_active() falsch ist. Arm #821
(invariant_checker.pp_receive_is_overdue) schaltet is_active aber erst ein, wenn ein PP-Empfang schon laenger als
der Timeout T blockiert. Die Blockade begann also T VOR dem Einschalten, der Wachhund zaehlte sie erst ab dem
Einschalten: Ausloesen nach 2T-2,5T (bei T=300: 600-750 s) statt nach T-1,5T.

Fix: WatchdogRaw nimmt einen optionalen ``stall_age``-Callback (Sekunden, die der Arm schon haengt, als DAUER) und
setzt im Zweig "aktiv, Zaehler steht" watchdog_last_time = min(watchdog_last_time, current - age).

Die Tests laufen auf einer SIMULIERTEN Uhr (sleep rueckt sie vor), deterministisch und ohne Wartezeit. Die zwei
Uhren sind absichtlich gegeneinander versetzt (perf_counter = sim + 1e6, monotonic = sim + 7): reicht irgendwo ein
Zeitstempel statt einer Dauer ueber die Naht, faellt das Ausloesen weit daneben.

  1. #821 ueber den ECHTEN create_scheduler_watchdog: is_active wird erst nach T Blockade wahr, stall_age > T,
     Ausloesen innerhalb 1,5T nach Blockadebeginn. Mutant ohne die min()-Zeile: rot (Ausloesen bei ~2,5T).
  2. H86-Leerlauf (Schlaf > T, dann langsamer erster Forward) bleibt ohne Ausloesen -- auch MIT stall_age.
  3. Ohne stall_age ist der Ablauf gleich H86: identische Proben- und Ausloesezeiten gegen eine woertliche Kopie
     der H86-Schleife, ueber mehrere Szenarien.
"""
import types
import unittest
from unittest import mock

from sglang.srt.utils import watchdog as W

T = 300.0
PERF_OFFSET = 1.0e6
MONO_OFFSET = 7.0


class _Stop(Exception):
    pass


class _SimClock:
    """Gemeinsame Simulationszeit; sleep rueckt vor und ruft den Szenario-Takt."""

    def __init__(self, limit, on_tick=None):
        self.now = 0.0
        self.limit = limit
        self.on_tick = on_tick
        self.samples = []

    def perf_counter(self):
        return self.now + PERF_OFFSET

    def monotonic(self):
        return self.now + MONO_OFFSET

    def sleep(self, dt):
        self.samples.append(self.now)
        self.now += dt
        if self.now > self.limit:
            raise _Stop()
        if self.on_tick is not None:
            self.on_tick(self.now)


def _bare_watchdog(get_counter, is_active, stall_age=None, describe_arm=None):
    wd = W.WatchdogRaw.__new__(W.WatchdogRaw)
    wd.debug_name, wd.watchdog_timeout, wd.soft = "Scheduler", T, True
    wd.get_counter, wd.is_active = get_counter, is_active
    wd.dump_info, wd.describe_arm, wd.parent_process = None, describe_arm, mock.Mock()
    if stall_age is not None:
        wd.stall_age = stall_age
    return wd


def _run_once(wd, clock, extra_patches=()):
    """Eine _watchdog_once-Runde auf der Simulationsuhr. Liefert (Ausloesezeit|None, Ausloesezeilen)."""
    fake_time = types.SimpleNamespace(perf_counter=clock.perf_counter, sleep=clock.sleep,
                                      monotonic=clock.monotonic)
    lines = []
    patches = [mock.patch.object(W, "time", fake_time),
               mock.patch.object(W, "pyspy_dump_schedulers", lambda *a, **k: None),
               mock.patch.object(W, "logger")] + list(extra_patches)
    for p in patches:
        p.start()
    try:
        W.logger.error.side_effect = lambda msg, *a, **k: lines.append(str(msg))
        try:
            wd._watchdog_once()
        except _Stop:
            return None, lines
        return clock.now, [m for m in lines if "watchdog timeout" in m]
    finally:
        for p in reversed(patches):
            p.stop()


# --------------------------------------------------------------------------------------------------------------
# 1. Arm #821 ueber den echten create_scheduler_watchdog
# --------------------------------------------------------------------------------------------------------------

def _build_real(scheduler):
    from sglang.srt.managers.scheduler_components import invariant_checker as ic

    captured = {}

    class _Fake:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with mock.patch.object(ic, "WatchdogRaw", _Fake):
        ic.create_scheduler_watchdog(scheduler, watchdog_timeout=T, soft=True)
    return ic, captured


def _pp_wedge_scenario(serve_until=440.0, blocked_from=460.0):
    """Server bedient bis ``serve_until`` (cur_batch gesetzt, Zaehler laeuft, nach der letzten Probe ungesehen weiter),
    ist kurz leer, und ab ``blocked_from`` haengt der Rang in einem PP-Empfang, der nie zurueckkehrt."""
    sched = types.SimpleNamespace(is_initializing=False, cur_batch_for_debug=object(), forward_ct=1,
                                  _pp_blocked_recv_since=None, _pp_blocked_recv_arm=None)

    def tick(now):
        if now < serve_until:
            sched.cur_batch_for_debug = object()
            sched.forward_ct = 1 + int(now / 10.0)
        else:
            sched.cur_batch_for_debug = None
            sched.forward_ct = 1 + int(serve_until / 10.0)
            if now >= blocked_from and sched._pp_blocked_recv_since is None:
                sched._pp_blocked_recv_since = blocked_from + MONO_OFFSET   # monotonic-Stempel wie im Code
                sched._pp_blocked_recv_arm = "typed-dict/proxy"

    return sched, tick


class LateArmTripsWithinOnePointFiveT(unittest.TestCase):
    def _trip(self, *, with_stall_age=True, blocked_from=460.0):
        sched, tick = _pp_wedge_scenario(blocked_from=blocked_from)
        clock = _SimClock(limit=blocked_from + 4 * T, on_tick=tick)
        ic, kw = _build_real(sched)
        self.assertIn("stall_age", kw, "create_scheduler_watchdog muss stall_age verdrahten")
        wd = _bare_watchdog(kw["get_counter"], kw["is_active"],
                            kw["stall_age"] if with_stall_age else None, kw["describe_arm"])
        fake_time = types.SimpleNamespace(monotonic=clock.monotonic, perf_counter=clock.perf_counter,
                                          sleep=clock.sleep)
        t_trip, lines = _run_once(wd, clock, [mock.patch.object(ic, "time", fake_time)])
        return t_trip, lines, clock

    def test_arm_is_dark_for_the_first_T(self):
        sched, _ = _pp_wedge_scenario()
        ic, kw = _build_real(sched)
        clock = _SimClock(limit=1e9)
        with mock.patch.object(ic, "time", types.SimpleNamespace(monotonic=clock.monotonic)):
            sched.cur_batch_for_debug = None
            sched._pp_blocked_recv_since = 100.0 + MONO_OFFSET
            clock.now = 100.0 + T - 1.0
            self.assertFalse(kw["is_active"](), "#821 darf erst nach T Blockade einschalten")
            clock.now = 100.0 + T + 1.0
            self.assertTrue(kw["is_active"]())
            self.assertAlmostEqual(kw["stall_age"](), T + 1.0, places=6)
            sched._pp_blocked_recv_since = None
            self.assertEqual(kw["stall_age"](), 0.0)

    def test_overdue_pp_receive_trips_within_1_5_T(self):
        for blocked_from in (460.0, 455.0, 520.0, 599.0):
            with self.subTest(blocked_from=blocked_from):
                t_trip, lines, _ = self._trip(blocked_from=blocked_from)
                self.assertIsNotNone(t_trip, "der haengende PP-Empfang wurde nie erkannt")
                self.assertGreater(t_trip - blocked_from, T, "vor T ausgeloest")
                self.assertLessEqual(t_trip - blocked_from, 1.5 * T,
                                     f"erst nach {(t_trip - blocked_from) / T:.2f}T erkannt (H86-Verlangsamung)")
                self.assertTrue(lines and "blocked-recv[typed-dict/proxy]" in lines[-1], lines)

    def test_without_stall_age_it_is_the_slow_h86_behaviour(self):
        # Beleg, dass Test 1 die Verlangsamung wirklich misst: ohne stall_age (= H86) > 1,5T.
        t_trip, _, _ = self._trip(with_stall_age=False)
        self.assertIsNotNone(t_trip)
        self.assertGreater(t_trip - 460.0, 1.5 * T)

    def test_a_raising_stall_age_never_kills_the_watchdog(self):
        clock = _SimClock(limit=10 * T)

        def boom():
            raise RuntimeError("kaputt")

        wd = _bare_watchdog(lambda: 7, lambda: True, boom)
        t_trip, lines = _run_once(wd, clock)
        self.assertIsNotNone(t_trip, "ein werfender stall_age hat den Wachhund getoetet")
        self.assertLessEqual(t_trip, 1.5 * T + 1e-6)


# --------------------------------------------------------------------------------------------------------------
# 2. H86-Leerlauf bleibt gruen, auch mit stall_age verdrahtet
# --------------------------------------------------------------------------------------------------------------

class H86IdleStaysQuietWithStallAge(unittest.TestCase):
    def test_long_sleep_then_slow_first_forward_with_short_recv(self):
        # fnNV4f4 in Sekunden: aktiv mit festem Zaehler, 293 s dormant... hier 4T, dann cur_batch gesetzt und 4 s
        # Warten auf die Proxy-Tensoren (blockierter Empfang mit kurzem Alter), dann laufender Zaehler.
        sleep_from, wake = 50.0, 50.0 + 4 * T

        def is_active(now_fn):
            t = now_fn()
            return t < sleep_from or t >= wake

        clock = _SimClock(limit=wake + 3 * T)

        def counter():
            t = clock.now
            return 7 if t < wake + 4.0 else 8 + int((t - wake) / 0.5)

        def stall_age():
            t = clock.now
            return (t - wake) if wake <= t < wake + 4.0 else 0.0

        wd = _bare_watchdog(counter, lambda: is_active(lambda: clock.now), stall_age)
        t_trip, lines = _run_once(wd, clock)
        self.assertIsNone(t_trip, f"Leerlauf als Stillstand gezaehlt: {lines}")


# --------------------------------------------------------------------------------------------------------------
# 3. Ohne stall_age: gleich H86
# --------------------------------------------------------------------------------------------------------------

def _h86_reference(self_, clock):
    """Woertliche H86-Schleife (b191dc8bb0), nur die Uhr injiziert."""
    watchdog_last_counter = 0
    watchdog_last_time = clock.perf_counter()
    while True:
        current = clock.perf_counter()
        if self_.is_active():
            current_counter = self_.get_counter()
            if watchdog_last_counter == current_counter:
                if current > watchdog_last_time + self_.watchdog_timeout:
                    break
            else:
                watchdog_last_counter = current_counter
                watchdog_last_time = current
        else:
            watchdog_last_time = current
        clock.sleep(self_.watchdog_timeout / 2)


def _scenarios():
    out = []
    # a) echter Stillstand ab Start
    out.append(("frozen", lambda c: 7, lambda c: True))
    # b) Leerlauf 4T, dann festes Stueck < T, dann laufend
    out.append(("idle-then-run", lambda c: 7 if c.now < 4.5 * T else 8 + int(c.now),
                lambda c: not (0.1 * T <= c.now < 4.1 * T)))
    # c) Leerlauf 4T, dann Stillstand
    out.append(("idle-then-stall", lambda c: 7, lambda c: not (0.1 * T <= c.now < 4.1 * T)))
    # d) wechselnd aktiv/inaktiv mit festem Zaehler
    out.append(("flapping", lambda c: 7, lambda c: int(c.now / (0.7 * T)) % 2 == 0))
    # e) laufender Zaehler, der nach 3T einfriert
    out.append(("run-then-freeze", lambda c: 1 + int(min(c.now, 3 * T) / 7.0), lambda c: True))
    return out


class WithoutStallAgeItIsH86(unittest.TestCase):
    def test_same_samples_and_same_trip_as_h86(self):
        for name, counter, active in _scenarios():
            with self.subTest(scenario=name):
                results = []
                for impl in ("new", "h86"):
                    clock = _SimClock(limit=12 * T)
                    wd = _bare_watchdog(lambda: counter(clock), lambda: active(clock))
                    if impl == "new":
                        self.assertIsNone(wd.stall_age)
                        t_trip, _ = _run_once(wd, clock)
                    else:
                        try:
                            _h86_reference(wd, clock)
                            t_trip = clock.now
                        except _Stop:
                            t_trip = None
                    results.append((t_trip, list(clock.samples)))
                self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
