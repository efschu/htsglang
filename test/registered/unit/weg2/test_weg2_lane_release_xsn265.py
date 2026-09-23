"""weg2xsn265 (17.09.2026): persistent lane buffers grew over both flip
directions to ~25 GB of tmpfs (priced 15.75 GiB); under the DFLASH form the
host ledger latched W98 18 s after the first flip. Registering is cheap now
(tmpfs populate before cudaHostRegister: 22-146 ms per lane), so the buffers
are released at every leg end: the depositor truncates after the drain
waits, the collector unmaps. These tests drive `_persistent_host_buffer` and
`release_host_lane_buffers` with a fake device-ops on a temp shm root.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange_bounce as bx  # noqa: E402


class _Ops:
    def __init__(self):
        self.registered = []
        self.unregistered = []

    def host_register(self, addr, nbytes, flags):
        self.registered.append((int(addr), int(nbytes)))

    def host_unregister(self, addr):
        self.unregistered.append(int(addr))


@pytest.fixture(autouse=True)
def _clean_cache():
    with bx._SEQ_CACHE_LOCK:
        bx._SEQ_HOST_BUF.clear()
    yield
    with bx._SEQ_CACHE_LOCK:
        for ent in list(bx._SEQ_HOST_BUF.values()):
            try:
                ent["mm"].close(); ent["fh"].close()
            except Exception:  # noqa: BLE001
                pass
        bx._SEQ_HOST_BUF.clear()


def test_the_truncating_release_unregisters_unmaps_and_truncates(tmp_path, monkeypatch):
    monkeypatch.setenv(bx.SEQ_HOST_REGISTER_ENV, "1")     # the pinned A/B form
    ops = _Ops()
    lines = []
    path = str(tmp_path / "weg2-seq-t" / "p0_unit_buffer.bin")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mm, addr, reg, refusal = bx._persistent_host_buffer(path, 1 << 20, ops, "p0", lines.append)
    assert refusal == "" and reg == "yes" and os.path.getsize(path) == 1 << 20
    assert any("persist lane=p0 new" in l for l in lines)
    n, total = bx.release_host_lane_buffers(truncate=True, log=lines.append)
    assert (n, total) == (1, 1 << 20)
    assert ops.unregistered == [addr]
    assert os.path.getsize(path) == 0
    assert not bx._SEQ_HOST_BUF
    # the next leg starts fresh: a NEW buffer, registered again
    lines.clear()
    mm2, addr2, reg2, _ = bx._persistent_host_buffer(path, 1 << 20, ops, "p0", lines.append)
    assert reg2 == "yes" and any("persist lane=p0 new" in l for l in lines)
    assert os.path.getsize(path) == 1 << 20


def test_the_unmapping_release_keeps_the_file_size(tmp_path, monkeypatch):
    monkeypatch.setenv(bx.SEQ_HOST_REGISTER_ENV, "1")
    ops = _Ops()
    path = str(tmp_path / "weg2-seq-t" / "c1_unit_buffer.bin")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bx._persistent_host_buffer(path, 4096, ops, "c1", lambda *_a: None)
    n, total = bx.release_host_lane_buffers(truncate=False, log=lambda *_a: None)
    assert (n, total) == (1, 4096)
    assert len(ops.unregistered) == 1
    assert os.path.getsize(path) == 4096      # the collector truncates, not the depositor


def test_xsn284_lanes_persist_by_default_and_an_env_one_restores_the_release(monkeypatch):
    """xsn284: registered, persistent lanes at depth 1 ran the flips at
    1.8-3.3 s per direction (5-12 s before); the per-leg release is the
    opt-in A/B form now."""
    monkeypatch.delenv(bx.SEQ_RELEASE_LANES_ENV, raising=False)
    assert bx.seq_release_lanes() is False
    monkeypatch.setenv(bx.SEQ_RELEASE_LANES_ENV, "1")
    assert bx.seq_release_lanes() is True
    monkeypatch.delenv(bx.SEQ_BUFFER_DEPTH_ENV, raising=False)
    assert bx.seq_buffer_depth() == 1
    monkeypatch.setenv(bx.SEQ_BUFFER_DEPTH_ENV, "4")
    assert bx.seq_buffer_depth() == 4


def test_both_leg_ends_call_the_release():
    from sglang.srt.managers.scheduler_components import weight_updater as wu
    src = open(wu.__file__).read()
    i = src.index("def _weg2_xchg_drain_outstanding")
    j = src.index("def _weg2_wake_collect_one", i)
    # xsn266: the DEPOSITOR only unmaps (its leg-end drain wait consumes
    # primed credits too, so it is no proof the collector is done); the
    # COLLECTOR, the last reader, truncates after its wake worker joined.
    assert "release_host_lane_buffers(truncate=False" in src[i:j]
    k = src.index("WEG2-WAKE-OVERLAP collects=")
    assert "release_host_lane_buffers(truncate=True" in src[k:k + 1500]


def test_xsn268_host_lanes_are_not_registered_by_default(tmp_path, monkeypatch):
    """py-spy --native at the DRAIN-STALL of xsn268: PP0 and TP0 of one card
    both inside cudaHostUnregister (ioctl) for the whole ~12.5 s stall. The
    pageable form (env 0): no register, no unregister, the release is unmap +
    truncate only. xsn284: the DEFAULT is registered again (once per lane,
    never unregistered), because the stall was the unregister, not the
    register -- 5090 lanes 12-13 GB/s vs 1.6-3.9 pageable."""
    monkeypatch.delenv(bx.SEQ_HOST_REGISTER_ENV, raising=False)
    assert bx.seq_host_register() is True
    monkeypatch.setenv(bx.SEQ_HOST_REGISTER_ENV, "0")
    assert bx.seq_host_register() is False
    ops = _Ops()
    lines = []
    path = str(tmp_path / "weg2-seq-t" / "p1_unit_buffer.bin")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mm, addr, reg, refusal = bx._persistent_host_buffer(path, 1 << 20, ops, "p1", lines.append)
    assert refusal == "" and reg == "no(off)"
    assert ops.registered == []
    n, total = bx.release_host_lane_buffers(truncate=True, log=lines.append)
    assert (n, total) == (1, 1 << 20)
    assert ops.unregistered == []
    assert os.path.getsize(path) == 0
