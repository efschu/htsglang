"""COLLECT-WAIT MONITOR 15.10. (Auftrag nf-core-collect-1015, task #119): the 'D war IDLE' verdict of 13:13:02Z was a false alarm.

NF h16p (226aad7105) boot 12:42:45Z, 15.10.: the front held pdflip-42-129 (213 uncached, +5089 of pdflip-40-128 already booked on D = 5302
of X=6057) 13:12:55.053 -> 13:13:02.558 (release n=1 tokens=5302 X=6057 waited_s=7.5 route=D stage=dcheck). The monitor
(nf_collect_wait_monitor.py) reported "D war IDLE", but D TP0 decoded bs=5-6 (Decode rank batch 9034-9061, 13:12:53-56) and then
prefilled pdflip-40-128 in two chunks (#969 EXTENT 13:12:56 and 13:13:00, chunk 13:13:00.255-13:13:04.228). The monitor read only
'Decode batch, #running-req' (one line per ~64 rounds, absent while a chunk halts the rounds) and 'Prefill batch, #new-seq' (stamped
at the END of the chunk, 13:13:04 = outside the window). Over all 15.10. boots: 1320 releases, 0 with D idle; old monitor 2 alarms.

Every fixture line below is a verbatim TP0 line of that boot (or of the 14.10. b6e5685092 boot for the true-idle counter case).

ROUND 2 (review NO, deskq/done/nf-core-collect-1015-review.md): the round-1 monitor was BLIND for the real class (route=D, D empty): it counted the
EXTENT of the RELEASED set itself (+1 s after the release) and read ``t0_unix_ms`` of the VRAM instrument as the chunk start. The three real
14.10. cases (D idle, release route=D stage=dcheck; boots bc35462240_1014_094512 10:29:45Z + 10:30:14Z and 95914428c2_1014_105023 10:54:41Z)
must stay ALARM; 15.10. 13:13:02Z must not alarm. Law: D work counts only if it BEGAN before the release fell and is not the released set.
The ``synthetic`` fixtures are built from the real numbers of 13:13 and marked so; every other fixture line is verbatim.
"""
from __future__ import annotations

import importlib.util
import os
import types

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_PATH = os.environ.get("NF_COLLECT_MONITOR_PATH") or os.path.join(_REPO, "devtools", "nf_collect_wait_monitor.py")

# ---- the real 15.10. 13:13:02Z case ----------------------------------------------------------------------------
HOLD = ("[2026-10-15 13:12:55,053] INFO pdflip.front: PDFLIP DECODE-COLLECT hold n=1 tokens=5302 X=6057 waited_s=0.0 window_s=15.0 "
        "d_check_s=7.5 epoch=42 -- D decodes on, the arriving prefills collect")
RELEASE = ("[2026-10-15 13:13:02,558] INFO pdflip.front: PDFLIP DECODE-COLLECT release n=1 tokens=5302 X=6057 waited_s=7.5 route=D "
           "stage=dcheck p_bound=False epoch=42")
ROUNDS = [  # the window is [13:12:55.058, 13:13:02.558] (+-1 s tolerance of the point witnesses)
    "[2026-10-15 13:12:55 TP0] Decode rank batch, rank: 0, #round: 9055, t: 1792069974.933, bs: 5, #rows: 20, #fwd: 1, gpu-ms: 77.4 (split unavailable: graph-replay-reader-off, graphed-fwd 1/1)",
    "[2026-10-15 13:12:56 TP0] Decode rank batch, rank: 0, #round: 9061, t: 1792069975.971, bs: 5, #rows: 20, #fwd: 1, gpu-ms: 68.2 (split unavailable: graph-replay-reader-off, graphed-fwd 1/1)",
]
EXTENDS = [
    "[2026-10-15 13:12:56 TP0] #969 EXTENT n=150 fwd=1 reqs=[('pdflip-40-128', 45056, 47744, 45056, 2688)]",
    "[2026-10-15 13:13:00 TP0] #969 EXTENT n=151 fwd=1 reqs=[('pdflip-40-128', 47744, 50145, 47744, 2401)]",
]
CHUNK = ("[2026-10-15 13:13:04 TP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=2401 n=1 t0_unix_ms=1792069980255 t_unix_ms=1792069984228 "
         "window_ms=3973 peak_allocated_mib=30972 peak_reserved_mib=32290 start_allocated_mib=30426 transient_mib=546 allocated_mib=30469 "
         "reserved_mib=32252 card_free_start_mib=833 card_free_mib=869 card_total_mib=320")
# what the OLD monitor saw: the end-stamped prefill batch lines (13:13:04, after hi=13:13:03.558) and the first round after the chunk
LATE = [
    "[2026-10-15 13:13:04 TP0] Prefill batch, #new-seq: 1, #new-token: 2688, #cached-token: 45056, full token usage: 0.57, mamba usage: 0.63, #running-req: 5, #queue-req: 0, #pending-token: 2401, cuda graph: False, input throughput (token/s): 180.26",
    "[2026-10-15 13:13:04 TP0] Prefill batch, #new-seq: 1, #new-token: 2401, #cached-token: 0, full token usage: 0.57, mamba usage: 0.63, #running-req: 5, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 53619.91",
    "[2026-10-15 13:13:05 TP0] Decode batch, #running-req: 6, #full token: 297600, full token usage: 0.57, mamba num: 24, mamba usage: 0.63, accept len: 2.11, accept rate: 0.37, cuda graph: True, gen throughput (token/s): 38.79, #queue-req: 0",
]

# ---- the true-idle counter case: 14.10. 13:33:29Z (b6e5685092): D parked all seven decodes, idle 16 s (phase=idle n=0) -------------
HOLD_IDLE = ("[2026-10-14 13:33:14,350] INFO pdflip.front: PDFLIP DECODE-COLLECT hold n=1 tokens=13575 X=7367 waited_s=0.0 window_s=15.0 "
             "d_check_s=7.5 epoch=20 -- D decodes on, the arriving prefills collect")
RELEASE_IDLE = ("[2026-10-14 13:33:29,397] INFO pdflip.front: PDFLIP DECODE-COLLECT release n=1 tokens=13575 X=7367 waited_s=15.0 route=P "
                "stage=window p_bound=False epoch=20")
D_BEFORE_IDLE = [  # the last D work before the idle stretch: a chunk ending 13:33:13.537 (window starts 13:33:14.397)
    "[2026-10-14 13:33:13 TP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=2240 n=1 t0_unix_ms=1791984789493 t_unix_ms=1791984793537 window_ms=4044 peak_allocated_mib=30934 peak_reserved_mib=32084 start_allocated_mib=30463 transient_mib=471 allocated_mib=30463 re",
    "[2026-10-14 13:33:13 TP0] Prefill batch, #new-seq: 1, #new-token: 2240, #cached-token: 0, full token usage: 0.84, mamba usage: 0.63, #running-req: 5, #queue-req: 1, #pending-token: 63029, cuda graph: False, input throughput (token/s): 340274.79",
    "[2026-10-14 13:33:29 TP0] PDFLIP-VRAM-PEAK rank=0 phase=idle n=0 t0_unix_ms=1791984793537 t_unix_ms=1791984809582 window_ms=16045 peak_allocated_mib=30853 peak_reserved_mib=32088 start_allocated_mib=30463 transient_mib=390 allocated_mib=30408 reserved_mib=32088",
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


def _run(m, d_lines, hold, release, capsys, t_now=None, front_after=()):
    if t_now is not None:
        m._now["t"] = m._mod.ts(t_now)
    capsys.readouterr()
    for l in d_lines:
        m.feed_d(l)
    m.feed_front(hold, "x")
    m.feed_front(release, "x")
    for l in front_after:  # front lines that follow the release line (D-ADMIT of the released rids)
        m.feed_front(l, "x")
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


# synthetic (derived from the real 13:13 numbers): a chunk that STARTED before the window and runs past the release
EXT_STRADDLE = "[2026-10-15 13:12:50 TP0] #969 EXTENT n=149 fwd=1 reqs=[('pdflip-40-128', 45056, 47457, 45056, 2401)]"


def test_a_chunk_interval_alone_justifies_the_wait(mon, capsys):
    """A chunk whose own EXTENT is stamped 13:12:50 (before the window) and which ends 13:13:04.228 (after the release) ran through the
    whole window: no round, no point witness inside it -- only the interval from ITS extent says D worked."""
    out = _run(mon, [EXT_STRADDLE, CHUNK], HOLD, RELEASE, capsys)
    assert "D war IDLE" not in out and mon.stats["justified"] == 1


def test_a_chunk_end_alone_justifies_the_wait(mon, capsys):
    """synthetic: a chunk line (no EXTENT line exists -- the #969 probe is throttled after n>400, schedule_batch.py) ending 13:12:58.000,
    3 s into the window: D worked until then. The end is an exact epoch."""
    end = ("[2026-10-15 13:12:58 TP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=100 n=1 t0_unix_ms=1792069900000 t_unix_ms=1792069978000 "
           "window_ms=78000")
    out = _run(mon, [end], HOLD, RELEASE, capsys)
    assert "D war IDLE" not in out and mon.stats["justified"] == 1


def test_a_chunk_start_from_the_rank_batch_duration_justifies_the_wait(mon, capsys):
    """synthetic: no EXTENT line; the chunk (rows=2401) ends 13:13:04.228 (after the release) and its 'Prefill rank batch' (#new-token 2401,
    gpu-ms 5000) dates its start 13:12:59.2 -- inside the window. A rank batch of ANOTHER row count dates nothing."""
    rank = ("[2026-10-15 13:13:04 TP0] Prefill rank batch, #new-token: {n}, #cached-token: 47744, #chunks: 1, gpu-ms: 5000.0 (compute 900.0, wait 4100.0)")
    out = _run(mon, [CHUNK, rank.format(n=2401)], HOLD, RELEASE, capsys)
    assert "D war IDLE" not in out and mon.stats["justified"] == 1


def test_a_rank_batch_of_other_rows_dates_no_chunk(mon, capsys):
    rank = ("[2026-10-15 13:13:04 TP0] Prefill rank batch, #new-token: 2688, #cached-token: 47744, #chunks: 1, gpu-ms: 5000.0 (compute 900.0, wait 4100.0)")
    out = _run(mon, [CHUNK, rank], HOLD, RELEASE, capsys)
    assert "D war IDLE" in out and mon.stats["idle_waits"] == 1


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
    graze = ("[2026-10-15 13:12:56 TP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=100 n=1 t0_unix_ms=1792069970000 t_unix_ms=1792069975358 "
             "window_ms=5358 peak_allocated_mib=1 peak_reserved_mib=1")
    out = _run(mon, [graze], HOLD, RELEASE, capsys)
    assert "D war IDLE" in out and mon.stats["idle_waits"] == 1


# ================================================================================================================================
# ROUND 2: the REAL route=D / D-idle cases of 14.10. -- the monitor must keep alarming (verbatim lines of the evidence logs)
# ================================================================================================================================

# --- case A: boot bc35462240_1014_094512, 10:29:45Z (front :10760 / :10782, D :231571-232045). D parked all decodes at 10:29:36 and sat idle
#     10:29:38-46; the EXTENT/chunk lines at 10:29:46/48 are THE RELEASED MENGE ITSELF (pdflip-54-154, 647 rows), 0.07 s after the release.
A_HOLD = ("[2026-10-14 10:29:38,412] INFO pdflip.front: PDFLIP DECODE-COLLECT hold n=1 tokens=956 X=6283 waited_s=0.0 window_s=15.0 "
          "d_check_s=7.5 epoch=54 -- D decodes on, the arriving prefills collect")
A_RELEASE = ("[2026-10-14 10:29:45,928] INFO pdflip.front: PDFLIP DECODE-COLLECT release n=1 tokens=915 X=6283 waited_s=7.5 route=D "
             "stage=dcheck p_bound=False epoch=54")
A_ADMIT = ["[2026-10-14 10:29:45,937] INFO pdflip.front: PDFLIP D-ADMIT rid=pdflip-54-154 seat=1/6 rank=58 oldest_wait_s=0.0 source=short seat_wait_s=0.0"]
A_D = [
    "[2026-10-14 10:29:36 TP0] Decode rank batch, rank: 0, #round: 31349, t: 1791973775.716, bs: 1, #rows: 4, #fwd: 1, gpu-ms: 43.4",
    "[2026-10-14 10:29:36 TP0] #969 EXTENT n=221 fwd=1 reqs=[('pdflip-40-83', 96832, 96843, 96832, 11), ('pdflip-42-90', 38784, 38813, 38784, 29)]",
    "[2026-10-14 10:29:37 TP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=40 n=1 t0_unix_ms=1791973775720 t_unix_ms=1791973777954 window_ms=2235",
    "[2026-10-14 10:29:37 TP0] Prefill rank batch, #new-token: 40, #cached-token: 135616, #chunks: 1, gpu-ms: 1488.0 (compute 736.8, wait 751.2)",
    "[2026-10-14 10:29:37 TP0] Prefill batch, #new-seq: 2, #new-token: 40, #cached-token: 135616, full token usage: 0.25, mamba usage: 0.62, #running-req: 0",
    "[2026-10-14 10:29:46 TP0] #969 EXTENT n=222 fwd=1 reqs=[('pdflip-54-154', 140544, 141191, 140544, 647)]",
    "[2026-10-14 10:29:48 TP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=647 n=1 t0_unix_ms=1791973777954 t_unix_ms=1791973788811 window_ms=10857",
    "[2026-10-14 10:29:48 TP0] Prefill batch, #new-seq: 1, #new-token: 647, #cached-token: 140544, full token usage: 0.25, mamba usage: 0.62, #running-req: 0",
    "[2026-10-14 10:29:48 TP0] Decode rank batch, rank: 0, #round: 31352, t: 1791973788.829, bs: 1, #rows: 4, #fwd: 1, gpu-ms: 77.2",
]

# --- case B: same boot, 10:30:14Z (front :10824 / :10834): the one decode parked 0.34 s BEFORE the hold (last round 10:30:07.153, hold 10:30:07.495)
B_HOLD = ("[2026-10-14 10:30:07,495] INFO pdflip.front: PDFLIP DECODE-COLLECT hold n=1 tokens=676 X=6283 waited_s=0.0 window_s=15.0 "
          "d_check_s=7.5 epoch=54 -- D decodes on, the arriving prefills collect")
B_RELEASE = ("[2026-10-14 10:30:14,995] INFO pdflip.front: PDFLIP DECODE-COLLECT release n=1 tokens=676 X=6283 waited_s=7.5 route=D "
             "stage=dcheck p_bound=False epoch=54")
B_ADMIT = ["[2026-10-14 10:30:15,010] INFO pdflip.front: PDFLIP D-ADMIT rid=pdflip-54-156 seat=1/6 rank=59 oldest_wait_s=0.0 source=short seat_wait_s=0.0"]
B_D = [
    "[2026-10-14 10:30:05 TP0] Decode batch, #running-req: 1, #full token: 142400, full token usage: 0.27, mamba num: 4",
    "[2026-10-14 10:30:06 TP0] Decode batch, #running-req: 1, #full token: 142528, full token usage: 0.27, mamba num: 4",
    "[2026-10-14 10:30:07 TP0] Decode rank batch, rank: 0, #round: 31964, t: 1791973807.153, bs: 1",
    "[2026-10-14 10:30:15 TP0] #969 EXTENT n=223 fwd=1 reqs=[('pdflip-54-156', 142336, 142744, 142336, 408)]",
    "[2026-10-14 10:30:18 TP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=408 n=1 t0_unix_ms=1791973818032 t_unix_ms=1791973818032 window_ms=0",
    "[2026-10-14 10:30:18 TP0] Prefill batch, #new-seq: 1, #new-token: 408, #cached-token: 142336, full token usage: 0.27, mamba usage: 0.62, #running-req: 0",
    "[2026-10-14 10:30:18 TP0] Decode rank batch, rank: 0, #round: 31966, t: 1791973818.062, bs: 1",
]

# --- case C: boot 95914428c2_1014_105023, 10:54:41Z (front :717 / :739, named 'LANES FIX 1014' in front.py): last round 10:54:32.202, window 10:54:34.0-41.6
C_HOLD = ("[2026-10-14 10:54:34,033] INFO pdflip.front: PDFLIP DECODE-COLLECT hold n=1 tokens=70 X=4096 waited_s=0.0 window_s=15.0 "
          "d_check_s=7.5 epoch=0 -- D decodes on, the arriving prefills collect")
C_RELEASE = ("[2026-10-14 10:54:41,557] INFO pdflip.front: PDFLIP DECODE-COLLECT release n=9 tokens=3243 X=4096 waited_s=7.5 route=D "
             "stage=dcheck p_bound=False epoch=0")
C_ADMIT = ["[2026-10-14 10:54:41,585] INFO pdflip.front: PDFLIP D-ADMIT rid=pdflip-0-11 seat=1/6 rank=1 oldest_wait_s=0.0 source=short seat_wait_s=0.0"]
C_D = [
    "[2026-10-14 10:54:29 TP0] Decode batch, #running-req: 1, #full token: 486208, full token usage: 0.93, mamba num: 4",
    "[2026-10-14 10:54:31 TP0] Decode batch, #running-req: 1, #full token: 486400, full token usage: 0.93, mamba num: 4",
    "[2026-10-14 10:54:32 TP0] Decode rank batch, rank: 0, #round: 215, t: 1791975272.202, bs: 1, #rows: 4, #fwd: 1, gpu-ms: 40.1",
    "[2026-10-14 10:54:42 TP0] #969 EXTENT n=2 fwd=1 reqs=[('pdflip-0-11', 191488, 191728, 191488, 240)]",
    "[2026-10-14 10:54:43 TP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=240 n=1 t0_unix_ms=1791975283869 t_unix_ms=1791975283869 window_ms=0",
    "[2026-10-14 10:54:43 TP0] Prefill batch, #new-seq: 1, #new-token: 240, #cached-token: 191488, full token usage: 0.93, mamba usage: 0.11, #running-req: 1",
    "[2026-10-14 10:54:43 TP0] Decode rank batch, rank: 0, #round: 217, t: 1791975283.883, bs: 1, #rows: 4, #fwd: 1, gpu-ms: 40.1",
]


def test_real_case_a_route_d_on_an_empty_d_is_still_alarmed(mon, capsys):
    """RED on the round-1 monitor (it saw 'justified': the released menge's own EXTENT +0.07 s and the chunk interval from t0_unix_ms
    10:29:37.954, which is the END of the previous chunk, not the start of this one)."""
    out = _run(mon, A_D, A_HOLD, A_RELEASE, capsys, t_now="2026-10-14 10:30:30", front_after=A_ADMIT)
    assert "D war IDLE" in out and mon.stats["idle_waits"] == 1 and mon.stats["justified"] == 0, out


def test_real_case_b_is_still_alarmed(mon, capsys):
    out = _run(mon, B_D, B_HOLD, B_RELEASE, capsys, t_now="2026-10-14 10:30:40", front_after=B_ADMIT)
    assert "D war IDLE" in out and mon.stats["idle_waits"] == 1 and mon.stats["justified"] == 0, out


def test_real_case_c_is_still_alarmed(mon, capsys):
    out = _run(mon, C_D, C_HOLD, C_RELEASE, capsys, t_now="2026-10-14 10:55:10", front_after=C_ADMIT)
    assert "D war IDLE" in out and mon.stats["idle_waits"] == 1 and mon.stats["justified"] == 0, out


def test_the_three_real_alarms_and_the_false_alarm_in_one_monitor(mon, capsys):
    """One monitor, the four releases in log order of their own boots' D lines: 3 alarms (A, B, C), and the 15.10. 13:13 one justified."""
    out = capsys.readouterr().out
    for d, h, r, adm in ((A_D, A_HOLD, A_RELEASE, A_ADMIT), (B_D, B_HOLD, B_RELEASE, B_ADMIT), (C_D, C_HOLD, C_RELEASE, C_ADMIT),
                         (ROUNDS + EXTENDS + [CHUNK] + LATE, HOLD, RELEASE, ())):
        for l in d:
            mon.feed_d(l)
        mon.feed_front(h, "x")
        mon.feed_front(r, "x")
        for l in adm:
            mon.feed_front(l, "x")
    mon.settle(force=True)
    out = capsys.readouterr().out
    assert out.count("D war IDLE") == 3 and mon.stats["idle_waits"] == 3 and mon.stats["justified"] == 1, out


# ---- the mutant-killing boundaries (each pins one rule of the round-2 law) -------------------------------------------------------

def test_a_decode_round_after_the_release_is_no_evidence(mon, capsys):
    """Exact epoch 13:13:02.600 is 42 ms AFTER the release (02.558): idle. 02.500 (58 ms before) justifies."""
    after = "[2026-10-15 13:13:02 TP0] Decode rank batch, rank: 0, #round: 9099, t: 1792069982.600, bs: 5, #rows: 20, #fwd: 1, gpu-ms: 70.0"
    before = after.replace("1792069982.600", "1792069982.500")
    assert "D war IDLE" in _run(mon, [after], HOLD, RELEASE, capsys)
    assert "D war IDLE" not in _run(_fresh(mon), [before], HOLD, RELEASE, capsys)


def test_a_decode_round_before_the_window_is_no_evidence(mon, capsys):
    """Window starts 13:12:55.058; a round at 13:12:54.900 belongs to the time before the hold."""
    early = "[2026-10-15 13:12:54 TP0] Decode rank batch, rank: 0, #round: 9050, t: 1792069974.900, bs: 5, #rows: 20, #fwd: 1, gpu-ms: 70.0"
    assert "D war IDLE" in _run(mon, [early], HOLD, RELEASE, capsys)


def test_an_extent_stamped_in_the_release_second_is_no_evidence(mon, capsys):
    """Stamps are floored: an EXTENT stamped 13:13:02 happened in [02.000, 03.000) -- it may be after the release (02.558). Stamped 13:13:01
    is certainly before (<= 02.000 + 1)."""
    ext = "[2026-10-15 13:13:0{s} TP0] #969 EXTENT n=152 fwd=1 reqs=[('pdflip-99-1', 100, 110, 100, 10)]"
    assert "D war IDLE" in _run(mon, [ext.format(s=2)], HOLD, RELEASE, capsys)
    assert "D war IDLE" in _run(_fresh(mon), [ext.format(s=3)], HOLD, RELEASE, capsys)
    assert "D war IDLE" not in _run(_fresh(mon), [ext.format(s=1)], HOLD, RELEASE, capsys)


def test_an_old_witness_stamped_in_the_release_second_is_no_evidence(mon, capsys):
    """The legacy 'Decode batch' / 'Prefill batch' lines keep their witness role, but not past the release (round 1 allowed +1 s)."""
    dec = ("[2026-10-15 13:13:02 TP0] Decode batch, #running-req: 6, #full token: 297600, full token usage: 0.57, mamba num: 24, mamba usage: 0.63, "
           "accept len: 2.11, accept rate: 0.37, cuda graph: True, gen throughput (token/s): 38.79, #queue-req: 0")
    pre = ("[2026-10-15 13:13:03 TP0] Prefill batch, #new-seq: 1, #new-token: 2688, #cached-token: 45056, full token usage: 0.57, mamba usage: 0.63, "
           "#running-req: 5, #queue-req: 0, #pending-token: 2401, cuda graph: False, input throughput (token/s): 180.26")
    assert "D war IDLE" in _run(mon, [dec, pre], HOLD, RELEASE, capsys)


def test_the_chunk_interval_does_not_start_at_t0_unix_ms(mon, capsys):
    """synthetic: the instrument window t0 = 13:12:20 (30 s before the hold: the idle stretch before the chunk), the chunk's own extent is
    stamped 13:13:03 (after the release) and it ends 13:13:05. With t0 as the start the interval would cover the whole window."""
    ext = "[2026-10-15 13:13:03 TP0] #969 EXTENT n=152 fwd=1 reqs=[('pdflip-99-1', 0, 500, 0, 500)]"
    chunk = ("[2026-10-15 13:13:05 TP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=500 n=1 t0_unix_ms=1792069940000 t_unix_ms=1792069985000 "
             "window_ms=45000")
    assert "D war IDLE" in _run(mon, [ext, chunk], HOLD, RELEASE, capsys)


def test_an_unpaired_chunk_gets_no_interval_from_t0(mon, capsys):
    """synthetic: chunk line without any EXTENT / rank batch (probe throttled), t0 30 s before the hold, end after the release: idle."""
    chunk = ("[2026-10-15 13:13:05 TP0] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=500 n=1 t0_unix_ms=1792069940000 t_unix_ms=1792069985000 "
             "window_ms=45000")
    assert "D war IDLE" in _run(mon, [chunk], HOLD, RELEASE, capsys)
    na = chunk.replace("t0_unix_ms=1792069940000", "t0_unix_ms=na").replace("window_ms=45000", "window_ms=na")
    assert "D war IDLE" in _run(_fresh(mon), [na], HOLD, RELEASE, capsys)


def test_an_extent_with_other_rows_than_the_chunk_dates_nothing(mon, capsys):
    ext = "[2026-10-15 13:12:50 TP0] #969 EXTENT n=149 fwd=1 reqs=[('pdflip-40-128', 45056, 47457, 45056, 1111)]"
    out = _run(mon, [ext, CHUNK], HOLD, RELEASE, capsys)  # CHUNK rows=2401 != 1111
    assert "D war IDLE" in out


def test_work_of_the_released_rids_never_counts(mon, capsys):
    """synthetic: the released rid pdflip-99-1 is D-ADMITted right after the release; an EXTENT of that rid stamped 13:13:00 (a clock/stamp
    anomaly that would pass the time rule) must not make D 'busy'. An EXTENT of another rid at the same stamp does."""
    adm = ["[2026-10-15 13:13:02,570] INFO pdflip.front: PDFLIP D-ADMIT rid=pdflip-99-1 seat=1/6 rank=1 oldest_wait_s=0.0 source=short seat_wait_s=0.0"]
    ext = "[2026-10-15 13:13:00 TP0] #969 EXTENT n=152 fwd=1 reqs=[('{rid}', 100, 110, 100, 10)]"
    assert "D war IDLE" in _run(mon, [ext.format(rid="pdflip-99-1")], HOLD, RELEASE, capsys, front_after=adm)
    assert "D war IDLE" not in _run(_fresh(mon), [ext.format(rid="pdflip-77-7")], HOLD, RELEASE, capsys, front_after=adm)


def test_an_admit_before_the_release_is_a_carried_rid_and_counts(mon, capsys):
    """The 13:13 carried rid (pdflip-40-128) was D-ADMITted at 13:12:55.053, BEFORE the hold: its extents ARE D work in the window."""
    adm = ["[2026-10-15 13:12:55,053] INFO pdflip.front: PDFLIP D-ADMIT rid=pdflip-40-128 seat=6/6 rank=9 oldest_wait_s=0.0 source=batch seat_wait_s=0.0"]
    out = _run(mon, EXTENDS, HOLD, RELEASE, capsys, front_after=adm)
    assert "D war IDLE" not in out


def test_replay_keeps_the_whole_d_log(tmp_path, monkeypatch, capsys):
    """round 1 pruned witnesses older than 2 h before the END of the D log (the replay reads D first): a 3 h boot reported every release
    of its first hour idle (e.g. 15.10. 04:21 boot). Replay must keep all."""
    mod = _load()
    front = tmp_path / "boot_x.front.log"
    dlog = tmp_path / "boot_x.D.log"
    front.write_text(HOLD.replace("2026-10-15 13:12:55", "2026-10-15 09:00:00") + "\n" + RELEASE.replace("2026-10-15 13:13:02", "2026-10-15 09:00:07") + "\n")
    rnd = "[2026-10-15 09:00:03 TP0] Decode rank batch, rank: 0, #round: 1, t: 1792054803.000, bs: 5, #rows: 20, #fwd: 1, gpu-ms: 70.0"
    tail = "[2026-10-15 12:30:00 TP0] Decode rank batch, rank: 0, #round: 2, t: 1792067400.000, bs: 5, #rows: 20, #fwd: 1, gpu-ms: 70.0"
    dlog.write_text(rnd + "\n" + tail + "\n")
    mod.run_replay(str(front))
    out = capsys.readouterr().out
    assert "D war IDLE" not in out and "begruendet (D rechnete)=1" in out, out


def _fresh(mon):
    """A second monitor on the same module/clock (a test may check several variants of one case)."""
    m = mon._mod.Mon(dry=True)
    m._mod, m._now = mon._mod, mon._now
    return m
