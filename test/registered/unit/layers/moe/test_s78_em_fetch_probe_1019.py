"""S78 EM-FETCH: rows, bytes and device time of the expert row copies of an EAGER-DEVICE-EM forward.

rc17 D (``..._1019_171636``): 388 MIXED-D-opened gaps, forward 444.8 s, fit 639 ms + 1.07 ms/token; all three
ranks leave the forward within 2 ms of each other, so no host stamp names the pacer or the H2D share. The
probe counts on the device what the pool steps copied and times it with events; the line comes a forward
later, from an event query (never a wait).

What must hold, on the S51 CPU pool harness (fake timing events):
* the rows equal the forward's misses (``misses_total``), the bytes rows x one expert row's bytes over all
  pool tensors, one pair of events per wave, one layer;
* nothing is logged before the device work completed; the scheduler funnel closes and drains;
* off under graph capture, with the switch off, and on a device type it does not time.
"""

import importlib.util
import logging
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.layers.moe import em_fetch_probe as P  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(__file__)

_spec = importlib.util.spec_from_file_location(
    "_t_em_s51_for_fetch", os.path.join(os.path.dirname(__file__), "test_pool_eager_device_em_s51.py"))
S51 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S51)


class _Clock:
    t = 0.0


class _Ev:
    """A timing event on a fake device clock: each record advances it 1.5 ms."""

    done = True

    def __init__(self, enable_timing=False):
        self.ts = None

    def record(self):
        _Clock.t += 1.5
        self.ts = _Clock.t

    def elapsed_time(self, other):
        return float(other.ts - self.ts)

    def query(self):
        return _Ev.done


class _FakeCuda:
    Event = _Ev
    capturing = False

    @staticmethod
    def is_current_stream_capturing():
        return _FakeCuda.capturing


@pytest.fixture(autouse=True)
def _probe(monkeypatch):
    P.reset_for_tests()
    _Ev.done = True
    _FakeCuda.capturing = False
    monkeypatch.setattr(P, "DEVICE_TYPES", ("cuda", "cpu"))
    monkeypatch.setattr(P, "_cuda", lambda: _FakeCuda)
    yield
    P.reset_for_tests()


def test_rows_and_bytes_are_the_forwards_misses(monkeypatch, caplog):
    dev = S51._pool_cache(monkeypatch, warm=[8, 9])
    t = dev._pool_tables
    t.misses_total.zero_()
    P.on_forward(41)
    with caplog.at_level(logging.INFO, logger=P.__name__):
        got = S51._extend(dev, S51.WIDE, seed=5)
        P.on_forward(42)  # the next forward's funnel closes 41 and drains it
    assert torch.equal(got[:, 0], S51._truth(S51.WIDE, seed=5))  # the probe changes no output
    misses = int(t.misses_total[0])
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(P.MARK)]
    assert len(lines) == 1, lines
    row_bytes = S51.W * 4  # one float32 row of W words in the one pool tensor
    line = lines[0]
    assert "fwd=41 T=8 layers=1 waves=2 rows=%d " % misses in line
    assert "gib=%.3f" % (misses * row_bytes / float(1 << 30)) in line
    assert misses == 16 - 2


def test_nothing_is_logged_before_the_device_finished(monkeypatch, caplog):
    dev = S51._pool_cache(monkeypatch)
    P.on_forward(1)
    with caplog.at_level(logging.INFO, logger=P.__name__):
        S51._extend(dev)
        _Ev.done = False
        P.on_forward(2)
        assert not [r for r in caplog.records if r.getMessage().startswith(P.MARK)]
        _Ev.done = True
        P.on_forward(3)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(P.MARK)]
    assert len(lines) == 1 and "fwd=1 " in lines[0]


def test_capture_switch_and_device_type_keep_it_silent(monkeypatch, caplog):
    _FakeCuda.capturing = True
    assert P.begin_layer(torch.device("cpu")) is None
    _FakeCuda.capturing = False
    monkeypatch.setattr(P, "DEVICE_TYPES", ("cuda",))
    assert P.begin_layer(torch.device("cpu")) is None
    monkeypatch.setattr(P, "DEVICE_TYPES", ("cuda", "cpu"))
    P.reset_for_tests()
    from flliper.srt.environ import envs

    with envs.FLLIPER_PDFLIP_EM_FETCH.override(False):
        assert P.begin_layer(torch.device("cpu")) is None
        P.on_forward(5)
    P.reset_for_tests()
    assert P.begin_layer(torch.device("cpu")) is not None


def test_the_funnel_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("no driver")

    monkeypatch.setattr(P, "_cuda", boom)
    P.on_forward(7)
    assert P.begin_layer(torch.device("cpu")) is None
