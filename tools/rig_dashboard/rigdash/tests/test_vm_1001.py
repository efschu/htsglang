"""VictoriaMetrics-Anbindung und geteilter /api/live-Schnappschuss (Nutzer-Orders 01.10.)."""

import threading
import time

from rigdash import server, vmpush


def _ipc():
    return {"boot_id": "nfh91xyz-boot-20261001T064831Z-2025", "terminal": False,
            "front": {"served": {"P": 3, "D": 9}, "awake": "D", "queue": [1, 2],
                      "served_tokens": {"D": {"prompt": 100, "cached": 80, "completion": 7, "n": 9}},
                      "arrival_seat": {"ttft_n": 4, "ttft_ms_sum": 8000, "ttft_ms_max": 3100}},
            "flip_first_work": [{"dir": "P>D", "flip_begin_ts": 100.0, "flip_time_ms": 2100.0},
                                {"dir": "P>D", "flip_begin_ts": 200.0, "flip_time_ms": 900.0, "what": "none"},
                                {"dir": "D>P", "flip_begin_ts": 300.0, "flip_time_ms": 1500.0}],
            "flip_user_time": [{"start_ts": 301.0, "flip_user_ms": 3700.0}]}


def test_short_boot_is_one_label_per_boot():
    assert vmpush.short_boot("nfh91abc-boot-20261001T064831Z-2025") == "boot-20261001T064831Z-2025"


def test_lines_carry_ttft_counters_and_no_rid():
    rank = {"D.tp0pp0": {"decode": {"tokens": 50, "running": 2}, "sched": {"full_token_usage": 0.4}}}
    ls = vmpush.lines_for_boot(_ipc(), rank, "NF", 1_000)
    txt = "\n".join(ls)
    assert 'weg2_front_ttft_count{boot="boot-20261001T064831Z-2025",model="NF"} 4.0 1000' in ls
    assert 'weg2_front_ttft_ms_sum{boot="boot-20261001T064831Z-2025",model="NF"} 8000.0 1000' in ls
    assert 'weg2_front_queue{boot="boot-20261001T064831Z-2025",model="NF"} 2.0 1000' in ls
    assert 'weg2_rank_decode_tokens_total{boot="boot-20261001T064831Z-2025",group="D",model="NF",rank="tp0pp0"} 50.0 1000' in ls
    assert "rid" not in txt and "weg2-" not in txt


def test_flip_points_at_flip_time_without_what_none_and_only_new():
    pts, newest = vmpush.flip_points(_ipc(), "NF", 0.0)
    assert pts == []                           # Nutzer 02.10.: the front's own numbers are no Flipzeit, not pushed
    assert newest == 301.0
    pts2, _ = vmpush.flip_points(_ipc(), "NF", newest)
    assert pts2 == []


def test_live_cache_computes_once_for_concurrent_readers():
    n = {"c": 0}

    def compute():
        n["c"] += 1
        time.sleep(0.05)
        return {"t": time.time()}
    c = server.LiveCache(compute, ttl=5.0)
    out = []
    th = [threading.Thread(target=lambda: out.append(c.get("v", lambda s: b"x"))) for _ in range(8)]
    [t.start() for t in th]
    [t.join() for t in th]
    assert n["c"] == 1 and out == [b"x"] * 8
    import gzip
    assert gzip.decompress(c.get("v", lambda s: b"x", gz=True)) == b"x"


def test_lean_snapshot_keeps_shown_boot_whole_and_strips_rows():
    live = {"stem": "a", "live": True, "series": {"t": [1]}, "timeline": {}, "fields": {"A1": 1}}
    old = {"stem": "b", "live": False, "primary": False, "series": {"t": [1]}, "timeline": {"segs": []},
           "fields": {"A1": 1}, "prefill": {"P": {"last_burst": {"tps": 5.0}, "now": {}}}, "totals": {"p_new": 3},
           "ipc": {"lifecycle": "stopped_clean", "launch": {"P": {}}}}
    snap = {"boots": [live, old], "features": {"x": 1}, "image_changes": {}}
    lean = server.lean_snapshot(snap, with_dev=False)
    assert lean["boots"][0] is live
    row = lean["boots"][1]
    assert "series" not in row and "fields" not in row and "launch" not in row["ipc"]
    assert row["prefill"]["P"]["last_burst"]["tps"] == 5.0 and row["totals"] == {"p_new": 3}
    assert "features" not in lean and "features" in server.lean_snapshot(snap, with_dev=True)


def test_sampler_restart_keeps_the_ring(tmp_path):
    """01.10.: a deploy restarts the sampler; its first append must not wipe the 16-min ring of the store."""
    from rigdash import ipcboot, sampler
    st = sampler.RingStore(str(tmp_path / "ring.sqlite"))
    now = time.time()
    st.append([("boot-a", now - 300 + i, {"t": now - 300 + i, "r": {}, "front": {}}) for i in range(5)],
              {"boot-a": now - 300})
    b = ipcboot.IpcBoots(roots={}, store=st, role="sampler")
    assert b.warm_from_store(now) == 5
    assert [s["t"] for s in b.rings["boot-a"]] == [now - 300 + i for i in range(5)]


def test_ttft_series_mean_per_bucket_and_gap():
    class Fake:
        def query_range(self, q, start, end, step):
            assert start == 105 and end == 125 and step == 5
            if "ttft_ms_sum" in q:
                return {105: 12000.0, 110: 0.0, 120: 3000.0}
            return {105: 3.0, 110: 0.0, 120: 1.0}
    out = vmpush.ttft_series(Fake(), "NF", [100, 105, 110, 115, 120], 5)
    assert out["mean_ms"] == [4000.0, None, None, 3000.0, None]     # bucket [t, t+5) read at t+5
    assert out["n"][0] == 3.0 and out["n"][1] is None


def test_pcie_series_gb_per_bucket_and_pcie_points_from_samples(monkeypatch):
    class Fake:
        def query_range_by(self, q, start, end, step, label):
            assert "weg2_gpu_pcie_bytes_per_second" in q and start == 15 and step == 5
            return {"0": {15: 1.5, 20: 0.25}, "2": {20: 3.0}}
    out = vmpush.pcie_series(Fake(), [10, 15], 5)
    assert out["series"]["g0.rx"] == [1.5, 0.25] and out["series"]["g2.tx"] == [None, 3.0]

    class Ipc:
        lock = threading.Lock()
        _st, _ev = {}, {}

    class Boots:
        ipc = Ipc()
        lock = threading.Lock()
        rank = {}
    br = vmpush.Bridge(Boots(), "http://127.0.0.1:1")
    br.pcie_source = lambda: [(100.0, [(2000.0, 1000.0)])]
    sent = []
    monkeypatch.setattr(vmpush, "push", lambda lines, url: sent.extend(lines) or len(sent))
    br.tick(101.0)
    assert 'weg2_gpu_pcie_bytes_per_second{dir="rx",gpu="0"} 2000000.0 100000' in sent
    assert br.pcie_t == 100.0
