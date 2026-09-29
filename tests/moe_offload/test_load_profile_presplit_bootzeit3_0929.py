"""BOOTZEIT 3 (29.09., z30r3): der Ladeprofiler tastet auch den
Presplit-Thread ab.

z30r3 PP0: presplit_busy_s=51.15 auf "load-presplit" gegen 26,8 s seriell
auf dem Ladethread -- und weder die Ladethread- noch die Konsumenten-Zeile
konnte sagen, wo der Thread die Zeit laesst. Eigene Zeile, eigener Zaehler.
"""

import logging
import threading
import time

from sglang.srt.model_loader.loader import _LoadSampler


class _Sink(logging.Logger):
    def __init__(self):
        super().__init__("sink")
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(msg % args if args else msg)


def _spin(stop):
    while not stop.is_set():
        pass


def test_the_presplit_thread_gets_its_own_line():
    stop = threading.Event()
    th = threading.Thread(target=_spin, args=(stop,), name="load-presplit", daemon=True)
    th.start()
    s = _LoadSampler(hz=200.0)
    s.start()
    time.sleep(0.2)
    sink = _Sink()
    s.report(sink, 0.2)
    stop.set()
    th.join(1.0)
    pre = [l for l in sink.lines if "WEG2 LOAD-PROFILE presplit" in l]
    assert len(pre) == 1, sink.lines
    assert "_spin" in pre[0]
    # the consumer line stays the consumers' own
    assert not any("LOAD-PROFILE consumers" in l for l in sink.lines)


def test_no_presplit_thread_no_presplit_line():
    s = _LoadSampler(hz=200.0)
    s.start()
    time.sleep(0.05)
    sink = _Sink()
    s.report(sink, 0.05)
    assert not any("LOAD-PROFILE presplit" in l for l in sink.lines)
