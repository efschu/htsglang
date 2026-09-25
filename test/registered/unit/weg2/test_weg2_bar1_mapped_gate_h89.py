"""H89 (dkrnfbar1agent09251908, cu130 image, 25.09.2026): the first P->D flip
died with W68 ``p0/weights_0: unit identity mismatch at unit 0: the deposit
record names ... tag=weights_9, this collect expects ... tag=weights_0``.

P's boot connect to D's windows p0/p2/p4 ran out (240 s; D served them 4 min
late on a cold JIT volume), so P deposited those lanes on the HOST ring. D
served the windows and its tag-order gate read "window here" as "BAR1 lane",
skipped the gate, and two collects in flight read the same slot counter
(both logged ``seq=0``). The collector must count a lane as BAR1 only when the
depositor mapped it; a depositor whose boot connect ran out tries once more at
its first flip."""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import bar1_lanes as b1  # noqa: E402

WU = os.path.join(os.path.dirname(__file__), *[".."] * 4, "python", "sglang", "srt", "managers",
                  "scheduler_components", "weight_updater.py")

PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
TIMED_OUT = "connect: no window served at /dev/shm/weg2-bar1-n1/p0.D/window.sock within 240 s"


def _lanes(tmp_path, group, rank):
    return b1.Bar1Lanes("n1", group, rank, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))


def _window():
    return SimpleNamespace(dptr=1, slot_bytes=8, ring=2)


def test_a_served_window_without_a_mapped_depositor_is_a_host_lane(tmp_path):
    dep = _lanes(tmp_path, "P", 0)      # PP0 deposits p0 into D TP1
    col = _lanes(tmp_path, "D", 1)
    col.recv["p0"] = _window()
    assert not col.peer_mapped("p0") and not col.collect_rides_bar1("p0")
    dep._mark_mapped("p0")
    assert col.peer_mapped("p0") and col.collect_rides_bar1("p0")
    # the marker is keyed by the RECEIVING group: P's own p0 window (D->P) is untouched
    back = _lanes(tmp_path, "P", 1)
    back.recv["p0"] = _window()
    assert not back.collect_rides_bar1("p0")
    # no window on this side: never BAR1, whatever the marker says
    assert not _lanes(tmp_path, "D", 1).collect_rides_bar1("p0")


def test_the_collect_gate_follows_the_depositor_not_the_window(tmp_path):
    col = _lanes(tmp_path, "D", 1)
    col.recv["p0"] = _window()
    assert b1.collect_gate_is_bar1(col, "p0", 0) is False      # the cu130 death: gate kept
    _lanes(tmp_path, "P", 0)._mark_mapped("p0")
    assert b1.collect_gate_is_bar1(col, "p0", 0) is True
    assert b1.collect_gate_is_bar1(col, "p0", None) is False   # diagonal lanes are host/IPC
    assert b1.collect_gate_is_bar1(None, "p0", 0) is False
    # a registry without the H89 query keeps the window-only rule
    old = SimpleNamespace(role=lambda lk: "dst", window_for=lambda lk, r: (1, 8, 2))
    assert b1.collect_gate_is_bar1(old, "p0", 0) is True


def test_the_updater_gate_asks_the_registry_query():
    # weight_updater imports torch/triton device probes; read its source instead
    src = open(WU).read()
    gate = src[src.index("# 18.09. (xsn369/370): the tag-order gate"):]
    gate = gate[:gate.index("_turns = self._weg2_lane_turns")]
    assert "_is_bar1_lane = self._weg2_collect_lane_is_bar1(_lane_key, pair)" in gate
    assert "window_for" not in gate
    body = src[src.index("def _weg2_collect_lane_is_bar1"):]
    assert "b1.collect_gate_is_bar1(getattr(self, \"_weg2_bar1\", None), lane_key, pair)" in body[:600]


def _fake_connect(dep, calls, succeed=True):
    def connect_peer(lane_key, timeout_s=None):
        calls.append((lane_key, timeout_s))
        if not succeed:
            dep.refusals[lane_key] = TIMED_OUT
            return None
        p = SimpleNamespace(dev_ptr=1, slot_bytes=8, ring=2)
        dep.peers[lane_key] = p
        dep._mark_mapped(lane_key)
        return p
    return connect_peer


def test_a_timed_out_boot_connect_is_retried_once_at_the_first_flip(tmp_path, monkeypatch):
    monkeypatch.setenv(b1.ENV_RECONNECT_S, "3")
    dep = _lanes(tmp_path, "P", 0)
    col = _lanes(tmp_path, "D", 1)
    col.recv["p0"] = _window()
    dep.refusals["p0"] = TIMED_OUT
    calls = []
    dep.connect_peer = _fake_connect(dep, calls)
    assert dep.lane_mode("p0", "src", seq="1-weights_9") == b1.MODE_BAR1
    assert calls == [("p0", 3.0)]
    assert col.lane_mode("p0", "dst", seq="1-weights_9", timeout_s=1.0) == b1.MODE_BAR1
    assert col.collect_rides_bar1("p0")
    assert dep.lane_mode("p0", "src", seq="1-weights_0") == b1.MODE_BAR1
    assert calls == [("p0", 3.0)]                   # mapped: no second connect


def test_a_failed_reconnect_stays_host_and_is_not_repeated_per_tag(tmp_path):
    dep = _lanes(tmp_path, "P", 0)
    col = _lanes(tmp_path, "D", 1)
    col.recv["p0"] = _window()
    dep.refusals["p0"] = TIMED_OUT
    calls = []
    dep.connect_peer = _fake_connect(dep, calls, succeed=False)
    for seq in ("1-weights_9", "1-weights_0", "1-weights_1"):
        assert dep.lane_mode("p0", "src", seq=seq) == b1.MODE_HOST
        assert col.lane_mode("p0", "dst", seq=seq, timeout_s=1.0) == b1.MODE_HOST
    assert len(calls) == 1
    assert not col.collect_rides_bar1("p0")        # the collector keeps the host gate


def test_a_named_refusal_is_not_retried(tmp_path):
    for why in ("peer serves no window: too small", "map: RuntimeError: none of 3 sg entries",
                "connect: no fd in the message"):
        dep = _lanes(tmp_path, "P", 0)
        dep.refusals["p0"] = why
        calls = []
        dep.connect_peer = _fake_connect(dep, calls)
        assert dep.lane_mode("p0", "src", seq=0) == b1.MODE_HOST
        assert calls == []
