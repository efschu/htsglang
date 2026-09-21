"""#66 (21.09.): der Ladeprofiler laeuft WAEHREND des Ladens und ist aus,
solange ihn niemand einschaltet.

Der Befund, der ihn ausgeloest hat: fnFL2v75 laedt mit 86 % CPU idle und
0,5 % iowait -- also weder Platte noch Maschine, sondern eine serielle
Schleife. py-spy von aussen traf sie nur zufaellig; dieser Sampler zaehlt
die Aufenthaltsorte des Ladethreads selbst.
"""

import inspect
import logging
import threading
import time

from sglang.srt.model_loader.loader import DefaultModelLoader, _LoadSampler


class _Sink(logging.Logger):
    def __init__(self):
        super().__init__("sink")
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(msg % args if args else msg)


def test_the_sampler_counts_the_thread_that_built_it():
    s = _LoadSampler(hz=200.0)
    s.start()
    t_end = time.perf_counter() + 0.2
    while time.perf_counter() < t_end:
        pass
    sink = _Sink()
    s.report(sink, 0.2)
    assert len(sink.lines) == 1
    line = sink.lines[0]
    assert "WEG2 LOAD-PROFILE" in line and "samples over 0.2 s" in line
    # the busy loop above lives in THIS file, so it must be the top site
    assert "test_load_profile_sampler_0921.py" in line


def test_a_sampler_with_no_samples_says_nothing():
    s = _LoadSampler(hz=1.0)
    sink = _Sink()
    s.report(sink, 1.0)   # never started
    assert sink.lines == []


def test_report_stops_the_thread():
    s = _LoadSampler(hz=100.0)
    s.start()
    time.sleep(0.05)
    s.report(_Sink(), 0.05)
    assert not any(t.name == "load-sampler" and t.is_alive() for t in threading.enumerate())


def test_it_is_off_unless_the_env_says_on_and_never_blocks_the_load():
    src = inspect.getsource(DefaultModelLoader.load_model)
    assert 'os.environ.get("SGLANG_LOAD_PROFILE") == "1"' in src
    # both the start and the report are wrapped: a broken sampler must never
    # keep a model from loading
    assert src.count("except Exception:") == 2
    assert "_prof, _prof_t0 = None, time.perf_counter()" in src
