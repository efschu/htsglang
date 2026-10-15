"""COLLECT-WAIT MONITOR 15.10. (Auftrag nf-core-collect-1015, task #119): the 'D war IDLE' verdict of 13:13:02Z was a false alarm.

NF h16p (226aad7105) boot 12:42:45Z, 15.10.: the front held weg2-42-129 (213 uncached, +5089 of weg2-40-128 already booked on D = 5302
of X=6057) 13:12:55.053 -> 13:13:02.558 (release n=1 tokens=5302 X=6057 waited_s=7.5 route=D stage=dcheck). The monitor
(nf_collect_wait_monitor.py) reported "D war IDLE", but D TP0 decoded bs=5-6 (Decode rank batch 9034-9061, 13:12:53-56) and then
prefilled weg2-40-128 in two chunks (#969 EXTENT 13:12:56 and 13:13:00, chunk 13:13:00.255-13:13:04.228). The monitor read only
'Decode batch, #running-req' (one line per ~64 rounds, absent while a chunk halts the rounds) and 'Prefill batch, #new-seq' (stamped
at the END of the chunk, 13:13:04 = outside the window). Over all 15.10. boots: 1320 releases, 0 with D idle; old monitor 2 alarms.

Every fixture line below is a verbatim TP0 line of that boot (or of the 14.10. b6e5685092 boot for the true-idle counter case).
"""
from __future__ import annotations

import importlib.util
import os
import types

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_PATH = os.environ.get("NF_COLLECT_MONITOR_PATH") or os.path.join(_REPO, "devtools", "nf_collect_wait_monitor.py")

# ---- the real 15.10. 13:13:02Z case ----------------------------------------------------------------------------
HOLD = ("[2026-10-15 13:12:55,053] INFO weg2.front: WEG2 DECODE-COLLECT hold n=1 tokens=5302 X=6057 waited_s=0.0 window_s=15.0 "
        "d_check_s=7.5 epoch=42 -- D decodes on, the arriving prefills collect")
RELEASE = ("[2026-10-15 13:13:02,558] INFO weg2.front: WEG2 DECODE-COLLECT release n=1 tokens=5302 X=6057 waited_s=7.5 route=D "
           "stage=dcheck p_bound=False epoch=42")
ROUNDS = [  # the window is [13:12:55.058, 13:13:02.558] (+-1 s tolerance of the point witnesses)
    "[2026-10-15 13:12:55 TP0] Decode rank batch, rank: 0, #round: 9055, t: 1792069974.933, bs: 5, #rows: 20, #fwd: 1, gpu-ms: 77.4 (split unavailable: graph-replay-reader-off, graphed-fwd 1/1)",
    "[2026-10-15 13:12:56 TP0] Decode rank batch, rank: 0, #round: 9061, t: 1792069975.971, bs: 5, #rows: 20, #fwd: 1, gpu-ms: 68.2 (split unavailable: graph-replay-reader-off, graphed-fwd 1/1)",
]
EXTENDS = [
    "[2026-10-15 13:12:56 TP0] #969 EXTENT n=150 fwd=1 reqs=[('weg2-40-128', 45056, 47744, 45056, 2688)]",
    "[2026-10-15 13:13:00 TP0] #969 EXTENT n=151 fwd=1 reqs=[('weg2-40-128', 47744, 50145, 47744, 2401)]",
]
CHUNK = ("[2026-10-15 13:13:04 TP0] WEG2-VRAM-PEAK rank=0 phase=chunk rows=2401 n=1 t0_unix_ms=1792069980255 t_unix_ms=1792069984228 "
         "window_ms=3973 peak_allocated_mib=30972 peak_reserved_mib=32290 start_allocated_mib=30426 transient_mib=546 allocated_mib=30469 "
         "reserved_mib=32252 card_free_start_mib=833 card_free_mib=869 card_total_mib=320")
# what the OLD monitor saw: the end-stamped prefill batch lines (13:13:04, after hi=13:13:03.558) and the first round after the chunk
LATE = [
    "[2026-10-15 13:13:04 TP0] Prefill batch, #new-seq: 1, #new-token: 2688, #cached-token: 45056, full token usage: 0.57, mamba usage: 0.63, #running-req: 5, #queue-req: 0, #pending-token: 2401, cuda graph: False, input throughput (token/s): 180.26",
    "[2026-10-15 13:13:04 TP0] Prefill batch, #new-seq: 1, #new-token: 2401, #cached-token: 0, full token usage: 0.57, mamba usage: 0.63, #running-req: 5, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 53619.91",
    "[2026-10-15 13:13:05 TP0] Decode batch, #running-req: 6, #full token: 297600, full token usage: 0.57, mamba num: 24, mamba usage: 0.63, accept len: 2.11, accept rate: 0.37, cuda graph: True, gen throughput (token/s): 38.79, #queue-req: 0",
]

# ---- the true-idle counter case: 14.10. 13:33:29Z (b6e5685092): D parked all seven decodes, idle 16 s (phase=idle n=0) -------------
HOLD_IDLE = ("[2026-10-14 13:33:14,350] INFO weg2.front: WEG2 DECODE-COLLECT hold n=1 tokens=13575 X=7367 waited_s=0.0 window_s=15.0 "
             "d_check_s=7.5 epoch=20 -- D decodes on, the arriving prefills collect")
RELEASE_IDLE = ("[2026-10-14 13:33:29,397] INFO weg2.front: WEG2 DECODE-COLLECT release n=1 tokens=13575 X=7367 waited_s=15.0 route=P "
                "stage=window p_bound=False epoch=20")
D_BEFORE_IDLE = [  # the last D work before the idle stretch: a chunk ending 13:33:13.537 (window starts 13:33:14.397)
    "[2026-10-14 13:33:13 TP0] WEG2-VRAM-PEAK rank=0 phase=chunk rows=2240 n=1 t0_unix_ms=1791984789493 t_unix_ms=1791984793537 window_ms=4044 peak_allocated_mib=30934 peak_reserved_mib=32084 start_allocated_mib=30463 transient_mib=471 allocated_mib=30463 re",
    "[2026-10-14 13:33:13 TP0] Prefill batch, #new-seq: 1, #new-token: 2240, #cached-token: 0, full token usage: 0.84, mamba usage: 0.63, #running-req: 5, #queue-req: 1, #pending-token: 63029, cuda graph: False, input throughput (token/s): 340274.79",
    "[2026-10-14 13:33:29 TP0] WEG2-VRAM-PEAK rank=0 phase=idle n=0 t0_unix_ms=1791984793537 t_unix_ms=1791984809582 window_ms=16045 peak_allocated_mib=30853 peak_reserved_mib=32088 start_allocated_mib=30463 transient_mib=390 allocated_mib=30408 reserved_mib=32088",
]


def _load():
    spec = importlib.util.spec_from_file_location("nf_collect_wait_monitor_under_test", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mon(monkeypatch):
    """A dry monitor whose clock stands just after the logs (settle() prunes witnesses older than 2 h of ITS clock)."""
    mod = _load()
    import time as _t

    now = {"t": mod.ts("2026-10-15 13:13:30")}
    shim = types.SimpleNamespace(time=lambda: now["t"], strftime=_t.strftime, gmtime=_t.gmtime, strptime=_t.strptime, sleep=_t.sleep)
    monkeypatch.setattr(mod, "time", shim)
    m = mod.Mon(dry=True)
    m._mod, m._now = mod, now
    return m


def _run(m, d_lines, hold, release, capsys, t_now=None):
    if t_now is not None:
        m._now["t"] = m._mod.ts(t_now)
    capsys.readouterr()
    for l in d_lines:
        m.feed_d(l)
    m.feed_front(hold, "x")
    m.feed_front(release, "x")
    m.settle(force=True)
    return capsys.readouterr().out


def test_the_real_1313_release_is_not_idle(mon, capsys):
    """RED on the base monitor ('D war IDLE'), GREEN with the D-busy witnesses."""
    out = _run(mon, ROUNDS + EXTENDS + [CHUNK] + LATE, HOLD, RELEASE, capsys)
    assert "D war IDLE" not in out, out
    assert mon.stats["idle_waits"] == 0 and mon.stats["justified"] == 1


def test_decode_rank_rounds_alone_justify_the_wait(mon, capsys):
    out = _run(mon, ROUNDS, HOLD, RELEASE, capsys)
    assert "D war IDLE" not in out and mon.stats["justified"] == 1


def test_an_extend_start_alone_justifies_the_wait(mon, capsys):
    """D prefills a chunk (no decode round runs while it does): the extend line stamps its START."""
    out = _run(mon, EXTENDS, HOLD, RELEASE, capsys)
    assert "D war IDLE" not in out and mon.stats["justified"] == 1


def test_a_chunk_interval_alone_justifies_the_wait(mon, capsys):
    """The chunk 13:13:00.255-13:13:04.228 overlaps the window by 2.3 s; its line comes after the release."""
    out = _run(mon, [CHUNK], HOLD, RELEASE, capsys)
    assert "D war IDLE" not in out and mon.stats["justified"] == 1


def test_the_old_witnesses_alone_still_justify_the_wait(mon, capsys):
    """Nothing the base monitor counted is dropped: a window-time 'Decode batch, #running-req' line."""
    line = ("[2026-10-15 13:12:58 TP0] Decode batch, #running-req: 6, #full token: 297600, full token usage: 0.57, mamba num: 24, "
            "mamba usage: 0.63, accept len: 2.11, accept rate: 0.37, cuda graph: True, gen throughput (token/s): 38.79, #queue-req: 0")
    out = _run(mon, [line], HOLD, RELEASE, capsys)
    assert "D war IDLE" not in out and mon.stats["justified"] == 1


def test_a_d_that_really_idled_is_still_flagged(mon, capsys):
    """The 14.10. 13:33:29Z case (D parked its decodes, 16 s phase=idle): the chunk before it ENDED 0.86 s before the window."""
    out = _run(mon, D_BEFORE_IDLE, HOLD_IDLE, RELEASE_IDLE, capsys, t_now="2026-10-14 13:34:00")
    assert "D war IDLE" in out and mon.stats["idle_waits"] == 1 and mon.stats["justified"] == 0


def test_an_empty_d_log_is_idle(mon, capsys):
    out = _run(mon, [], HOLD, RELEASE, capsys)
    assert "D war IDLE" in out and mon.stats["idle_waits"] == 1


def test_a_chunk_that_only_grazes_the_window_is_no_evidence(mon, capsys):
    """Window [13:12:55.058, 13:13:02.558]; a chunk ending 0.3 s into it (< CHUNK_MIN_OVERLAP_S) does not make D busy."""
    graze = ("[2026-10-15 13:12:56 TP0] WEG2-VRAM-PEAK rank=0 phase=chunk rows=100 n=1 t0_unix_ms=1792069970000 t_unix_ms=1792069975358 "
             "window_ms=5358 peak_allocated_mib=1 peak_reserved_mib=1")
    out = _run(mon, [graze], HOLD, RELEASE, capsys)
    assert "D war IDLE" in out and mon.stats["idle_waits"] == 1
