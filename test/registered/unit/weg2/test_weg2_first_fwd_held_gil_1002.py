"""L15-FWD-INST (N6k: the first forward after a held wake +300 ms in the
non-attention layers, other mean 10-12 ms vs ~4 ms unheld): FIRST-FWD-TIMING
carries the HOST gap beside the device gap per layer class and marks held
wakes; a GIL sampler runs during the first forward of a held wake."""
import time

from sglang.srt.weg2 import first_fwd_timing as F


class _Ev:
    _t = 0.0

    def __init__(self):
        self.t = None

    def record(self, stream=None):
        _Ev._t += 1.0
        self.t = _Ev._t

    def elapsed_time(self, other):
        return other.t - self.t

    def query(self):
        return True


class _Load:
    def wait(self, threshold):
        time.sleep(0.003)


def _reset(monkeypatch):
    monkeypatch.setattr(F, "S", F._State())
    F.S.event_factory = _Ev
    F.S.available = lambda: True


def _one_forward(layers=8):
    F.on_set_consumer(0)
    for lid in range(layers):
        F.timed_wait(_Load(), lid)
    F.on_set_consumer(1)   # closes the record (and the sampler)


def test_held_wake_gets_host_split_and_gil(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.delenv(F.GIL_ENV, raising=False)
    out = []
    monkeypatch.setattr(F.logger, "info", lambda fmt, *a: out.append(fmt % a))
    F.mark_held()            # restore before arm: carried to the next wake
    F.arm("kv_resume")
    _one_forward()
    F.on_set_consumer(2)     # harvest
    line = "\n".join(out)
    assert "held=1" in line and "host[full_attn sum=" in line and "gil[samples=" in line


def test_unheld_wake_has_no_sampler_by_default(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.delenv(F.GIL_ENV, raising=False)
    F.arm("kv_resume")
    F.on_set_consumer(0)
    assert F.S.sampler is None and F.S.cur["held"] == 0
    F.on_set_consumer(1)


def test_mark_after_arm_applies_to_this_wake(monkeypatch):
    _reset(monkeypatch)
    F.arm("kv_resume")
    F.mark_held()
    assert F.S.held_wake == F.S.wake and F.S.held_next is False


def test_gil_mode_switch(monkeypatch):
    monkeypatch.setenv(F.GIL_ENV, "off")
    assert F.gil_mode() == "off"
    monkeypatch.setenv(F.GIL_ENV, "all")
    assert F.gil_mode() == "all"
