"""F3 EM-FETCH TIME PER WAVE: the empty wave's apply is measured, not guessed.

S106 (20.10.): ``run_pool_em`` runs every wave's apply over ALL T x K lanes; on today's D prefills wave 1's copy
fits LRU + staging (rows <= ~63 per layer of 105), so wave 2 serves no lane -- and still runs its apply. Whether
that is a second full expert GEMM per layer, ``em_ms - copy_ms`` cannot say. The line now names, per wave index:
copy time, apply time (copy end -> the wave's last apply slice) and the lanes the wave served (device counter).

On the S51 CPU pool harness with the EM-FETCH fake timing events:
* a wide extend (two waves really needed): both waves serve lanes, served total = the valid lanes;
* a narrow extend (one wave holds it): wave 2 serves 0 lanes and its apply is still timed;
* the probe changes no output; no host read is added to the forward (the counter rides the existing D2H).
"""

import importlib.util
import logging
import os
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.layers.moe import em_fetch_probe as P  # noqa: E402

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "_t_f3_em_probe", os.path.join(_HERE, "test_s78_em_fetch_probe_1019.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
S51 = T.S51
_probe = T._probe  # noqa: F401 -- the autouse fixture (fake events, CPU allowed)


def _line(caplog):
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(P.MARK)]
    assert len(lines) == 1, lines
    return lines[0]


def _terms(line):
    m = re.search(r"wave_copy_ms=(\S+) wave_apply_ms=(\S+) wave_served=(\S+)", line)
    assert m, line
    return [[float(x) for x in g.split("/")] for g in m.groups()]


def test_a_wide_extend_serves_every_lane_over_two_waves(monkeypatch, caplog):
    dev = S51._pool_cache(monkeypatch, warm=[8, 9])
    P.on_forward(21)
    with caplog.at_level(logging.INFO, logger=P.__name__):
        got = S51._extend(dev, S51.WIDE, seed=5)
        P.on_forward(22)
    assert torch.equal(got[:, 0], S51._truth(S51.WIDE, seed=5))
    copy, apply_, served = _terms(_line(caplog))
    assert len(copy) == len(apply_) == len(served) == 2
    assert all(a > 0 for a in apply_)
    assert served[0] > 0 and served[1] > 0


def test_the_empty_second_wave_is_measured(monkeypatch, caplog):
    dev = S51._pool_cache(monkeypatch)
    P.on_forward(31)
    with caplog.at_level(logging.INFO, logger=P.__name__):
        S51._extend(dev)
        P.on_forward(32)
    copy, apply_, served = _terms(_line(caplog))
    assert len(served) == 2
    assert served[0] > 0 and served[1] == 0           # wave 2 served nothing ...
    assert apply_[1] > 0                              # ... and its apply ran anyway: now on the line


def test_the_wave_counter_rides_the_one_non_blocking_d2h(monkeypatch, caplog):
    seen = []
    real = P._d2h
    monkeypatch.setattr(P, "_d2h", lambda host, counter: (seen.append(tuple(counter.shape)), real(host, counter)))
    dev = S51._pool_cache(monkeypatch)
    P.on_forward(41)
    with caplog.at_level(logging.INFO, logger=P.__name__):
        S51._extend(dev)
        P.on_forward(42)
    assert seen == [(2 + 2 * P.MAX_WAVES,)]
