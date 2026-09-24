"""fnFL2 H22: the BAR1 deposit lanes copy with an SM kernel under
SGLANG_WEG2_LANE_PARALLEL_COPY instead of the card's one D2H copy engine.

Proven here on host memory with an ASYNC fake device (a stream's copies land
only at its ``synchronize``): the switch off is the copy-engine path call for
call; the switch on changes the engine and nothing else (same streams, same
slots, same order, same credits); a slot is refilled only after the
collector's free; a refused or failing kernel is a named mode or refusal;
overlap_ms is the union of the other lanes' spans; the kernel itself builds
with NVRTC (no GPU) for sm_86 and sm_120 without local memory."""
from __future__ import annotations

import ctypes
import os
import socket
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import bar1_lanes as b1  # noqa: E402
from sglang.srt.weg2 import lane_sm_copy as sm  # noqa: E402
from sglang.srt.weg2 import weight_exchange_transport as tp  # noqa: E402

PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))


def _lanes(tmp_path, group, rank):
    return b1.Bar1Lanes("n1", group, rank, 0, PAIRS, log=lambda *_a: None, root=str(tmp_path))


def _addr(buf):
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


class _AsyncOps:
    """A device whose copies are ASYNC: queued per stream, performed at that
    stream's synchronize. A full credit sent before its slot's sync would
    hand the collector an unwritten slot -- the byte check catches it."""

    name = "fake-async"

    def __init__(self, trace, sm_result=None):
        self.trace = trace
        self.q = {}
        self.n = 0
        self.sm_result = sm_result
        self.sm_asked = 0

    def create_stream(self, device):
        self.n += 1
        self.q[self.n] = []
        return self.n

    def destroy_stream(self, stream):
        pass

    def synchronize(self, stream):
        for fn in self.q.get(stream, []):
            fn()
        self.q[stream] = []
        self.trace.append(("sync", stream))

    def memcpy_async(self, dst, src, n, stream):
        self.trace.append(("copy", "ce", dst, src, n, stream))
        self.q[stream].append(lambda: ctypes.memmove(dst, src, n))

    def memcpy2d_async(self, dst, dpitch, src, spitch, width, height, stream):
        self.trace.append(("copy2d", "ce", dst, dpitch, src, spitch, width, height, stream))

        def _do():
            for r in range(height):
                ctypes.memmove(dst + r * dpitch, src + r * spitch, width)
        self.q[stream].append(_do)

    def sm_copier(self, device):
        self.sm_asked += 1
        if self.sm_result is None:
            raise AssertionError("sm_copier must not be asked with the switch off")
        return self.sm_result(self)


class _FakeSm:
    """The SM copier on the same async device: enqueues on the lane's stream."""

    name = "sm"

    def __init__(self, ops, fail_at=None):
        self.ops = ops
        self.calls = 0
        self.fail_at = fail_at

    def copy_async(self, dst, src, n, stream, *, blocks=0):
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise RuntimeError("cuLaunchKernel -> 719 (unspecified launch failure)")
        self.ops.trace.append(("copy", "sm", dst, src, n, stream))
        self.ops.q[stream].append(lambda: ctypes.memmove(dst, src, n))
        return 0

    def copy2d_async(self, dst, dpitch, src, spitch, width, height, stream, *, blocks=0):
        self.calls += 1
        self.ops.trace.append(("copy2d", "sm", dst, dpitch, src, spitch, width, height, stream))

        def _do():
            for r in range(height):
                ctypes.memmove(dst + r * dpitch, src + r * spitch, width)
        self.ops.q[stream].append(_do)
        return 0


class _HostOps:
    """The collector: copies out of its own window at once."""

    def __init__(self, delay_first=0.0):
        self.delay_first = delay_first
        self.copies = 0

    def create_stream(self, device):
        return 0

    def synchronize(self, stream):
        pass

    def memcpy_async(self, dst, src, n, stream):
        self.copies += 1
        if self.copies == 1 and self.delay_first:
            time.sleep(self.delay_first)
        ctypes.memmove(dst, src, n)

    def memcpy2d_async(self, dst, dpitch, src, spitch, width, height, stream):
        self.copies += 1
        for r in range(height):
            ctypes.memmove(dst + r * dpitch, src + r * spitch, width)


@pytest.fixture
def credit_trace(monkeypatch):
    """Every credit the DEPOSITOR sends/receives, in the depositor's trace."""
    holder = {}

    class _Traced(b1.SocketCredits):
        def send(self, kind, g, payload=None):
            t = holder.get(threading.get_ident())
            if t is not None:
                t.append(("send", kind, int(g)))
            return super().send(kind, g, payload)

        def recv(self, kind, g, timeout_s):
            got = super().recv(kind, g, timeout_s)
            t = holder.get(threading.get_ident())
            if t is not None:
                t.append(("recv", kind, int(g), got is not None))
            return got

    monkeypatch.setattr(b1, "SocketCredits", _Traced)
    return holder


def _transfer(tmp_path, credit_trace, *, ring, sm_result=None, parallel=False,
              collector_delay=0.0, lane="p0", budget=5.0):
    """One tag P->D over a host-memory window with socket credits; returns
    (depositor result, collector result, trace, lines, bytes_ok, ops)."""
    slot = 4096
    window = bytearray(slot * ring)
    dep = _lanes(tmp_path, "P", 0)
    col = _lanes(tmp_path, "D", 1 if lane == "p0" else 2)
    a, b = socket.socketpair()
    col.recv[lane] = SimpleNamespace(dptr=_addr(window), slot_bytes=slot, ring=ring, conn=b)
    dep.peers[lane] = SimpleNamespace(dev_ptr=_addr(window), slot_bytes=slot, ring=ring, sock=a)
    flat_src = bytearray(os.urandom(5 * slot + 1234))
    flat_dst = bytearray(len(flat_src))
    rows, run, spitch, dpitch = 20, 1000, 1536, 2048
    s2_src = bytearray(os.urandom(rows * spitch))
    s2_dst = bytearray(rows * dpitch)
    descs = [
        SimpleNamespace(kind=tp.FLAT, nbytes=len(flat_src), src_off=0, dst_off=0, param_name="w.flat",
                        tag="t0", src_ptr=_addr(flat_src), dst_ptr=_addr(flat_dst)),
        SimpleNamespace(kind=tp.STRIDED2D, nbytes=rows * run, rows=rows, run_bytes=run,
                        spitch=spitch, dpitch=dpitch, src_off=0, dst_off=0, param_name="w.s2",
                        tag="t0", src_ptr=_addr(s2_src), dst_ptr=_addr(s2_dst)),
    ]
    trace, lines = [], []
    ops = _AsyncOps(trace, sm_result=sm_result)
    out = {}

    def _collect():
        out["c"] = b1.run_bar1_units(descs, _HostOps(collector_delay), lanes=col, lane_key=lane,
                                     role="dst", seq="1-t0", phase="collect", budget_s=budget,
                                     log=lambda *_a: None)
    th = threading.Thread(target=_collect)
    th.start()
    credit_trace[threading.get_ident()] = trace
    with envs.SGLANG_WEG2_LANE_PARALLEL_COPY.override(parallel):
        out["d"] = b1.run_bar1_units(descs, ops, lanes=dep, lane_key=lane, role="src", seq="1-t0",
                                     phase="deposit", budget_s=5.0, log=lines.append)
    th.join(10)
    a.close()
    b.close()
    ok = bytes(flat_dst) == bytes(flat_src) and all(
        s2_dst[r * dpitch:r * dpitch + run] == s2_src[r * spitch:r * spitch + run] for r in range(rows))
    nb = len(tp.batch_descs(descs, slot_bytes=slot))
    return out.get("d"), out.get("c"), trace, lines, ok, ops, nb


def _check_protocol(trace, nb, ring):
    """The ring contract, read off the depositor's trace:
    * batches are issued in order, each on stream slot g % ring;
    * full(g) goes out only after the sync of g's stream that follows g's issue;
    * batch g >= ring is issued only after free(g - ring) was received."""
    issue_at, full_at, free_at, stream_of = {}, {}, {}, {}
    g = -1
    prev_copy_stream = None
    for i, ev in enumerate(trace):
        if ev[0] in ("copy", "copy2d"):
            # a batch = consecutive copies on ONE stream (batch g+1 is on the
            # next ring stream, so a change of stream is a new batch)
            if prev_copy_stream != ev[-1]:
                g += 1
                issue_at[g] = i
                stream_of[g] = ev[-1]
            prev_copy_stream = ev[-1]
            continue
        prev_copy_stream = None
        if ev[0] == "send" and ev[1] == "full":
            full_at[ev[2]] = i
        if ev[0] == "recv" and ev[1] == "free":
            free_at[ev[2]] = i
    assert sorted(issue_at) == list(range(nb))
    assert sorted(full_at) == list(range(nb))
    assert [full_at[k] for k in range(nb)] == sorted(full_at.values())      # in order
    streams = sorted({stream_of[k] for k in range(nb)})
    for k in range(nb):
        assert stream_of[k] == streams[k % ring]                           # slot g % ring
        syncs = [i for i, ev in enumerate(trace)
                 if ev[0] == "sync" and ev[1] == stream_of[k] and issue_at[k] < i < full_at[k]]
        assert syncs, f"full({k}) without a sync of its stream after its issue"
        if k >= ring:
            assert free_at[k - ring] < issue_at[k], f"slot of batch {k} refilled before free({k - ring})"


# -- the switch --------------------------------------------------------------

def test_switch_is_registered_and_off_by_default():
    assert envs.SGLANG_WEG2_LANE_PARALLEL_COPY.get() is False
    assert envs.SGLANG_WEG2_LANE_SM_COPY_BLOCKS.get() == 64
    ops = _AsyncOps([])
    assert b1.deposit_copy_mode(ops, 0) == (b1.COPY_SERIAL, None, "")
    assert ops.sm_asked == 0                   # off: the copier is never even asked


@pytest.mark.parametrize("ring", [3, 4])
def test_off_is_the_copy_engine_path_call_for_call(tmp_path, credit_trace, ring):
    d, c, trace, lines, ok, ops, nb = _transfer(tmp_path, credit_trace, ring=ring)
    assert d == "" and c == "" and ok
    assert ops.sm_asked == 0
    kinds = {ev[1] for ev in trace if ev[0] in ("copy", "copy2d")}
    assert kinds == {"ce"}
    _check_protocol(trace, nb, ring)
    line = [x for x in lines if "lane-time" in x][0]
    # the old fields keep their place; the new ones are appended
    assert line.index("rate_GBs=") < line.index("credit_ms=") < line.index("send_ms=") < line.index("mode=serial")
    assert "mode_why" not in line and "overlap_ms=" in line


@pytest.mark.parametrize("ring", [3, 4])
def test_sm_mode_changes_the_engine_and_nothing_else(tmp_path, credit_trace, ring):
    d0, c0, t_off, _l, ok0, _o, nb = _transfer(tmp_path / "off", credit_trace, ring=ring)
    d1, c1, t_on, lines, ok1, ops, nb1 = _transfer(
        tmp_path / "on", credit_trace, ring=ring, parallel=True,
        sm_result=lambda o: (_FakeSm(o), ""))
    assert (d0, c0, d1, c1) == ("", "", "", "") and ok0 and ok1 and nb == nb1
    assert {ev[1] for ev in t_on if ev[0] in ("copy", "copy2d")} == {"sm"}

    def norm(t):
        # addresses differ between the two runs' buffers; the SHAPE may not
        out = []
        for ev in t:
            if ev[0] == "copy":
                out.append(("copy", ev[4], ev[5]))
            elif ev[0] == "copy2d":
                out.append(("copy2d", ev[3], ev[5], ev[6], ev[7], ev[8]))
            else:
                out.append(ev)
        return out
    assert norm(t_on) == norm(t_off)          # same streams, sizes, syncs, credits, order
    _check_protocol(t_on, nb, ring)
    line = [x for x in lines if "lane-time" in x][0]
    assert "mode=sm sm_blocks=64 sm_slow_bytes=0" in line


@pytest.mark.parametrize("parallel", [False, True])
def test_a_slot_is_refilled_only_after_the_collectors_free(tmp_path, credit_trace, parallel):
    """The collector holds batch 0 for 300 ms: the depositor may fill the other
    ring-1 slots, never slot 0 again before free(0)."""
    ring = 3
    t0 = time.perf_counter()
    d, c, trace, _l, ok, _o, nb = _transfer(
        tmp_path, credit_trace, ring=ring, parallel=parallel, collector_delay=0.3,
        sm_result=(lambda o: (_FakeSm(o), "")) if parallel else None)
    assert d == "" and c == "" and ok and nb > ring
    assert time.perf_counter() - t0 >= 0.3
    _check_protocol(trace, nb, ring)


def test_a_refused_kernel_stays_on_the_copy_engine_with_its_reason(tmp_path, credit_trace):
    d, c, trace, lines, ok, ops, nb = _transfer(
        tmp_path, credit_trace, ring=4, parallel=True,
        sm_result=lambda o: (None, "RuntimeError: no libnvrtc could be loaded"))
    assert d == "" and c == "" and ok and ops.sm_asked == 1
    assert {ev[1] for ev in trace if ev[0] in ("copy", "copy2d")} == {"ce"}
    _check_protocol(trace, nb, 4)
    line = [x for x in lines if "lane-time" in x][0]
    assert "mode=serial mode_why=RuntimeError:_no_libnvrtc_could_be_loaded" in line
    # ops without the seam at all (a desk fake, the host ops): named as well
    assert b1.deposit_copy_mode(_HostOps(), 0)[0] == b1.COPY_SERIAL
    with envs.SGLANG_WEG2_LANE_PARALLEL_COPY.override(True):
        mode, cp, why = b1.deposit_copy_mode(_HostOps(), 0)
    assert (mode, cp) == (b1.COPY_SERIAL, None) and "no sm copier" in why


def test_a_failing_launch_is_a_named_refusal_and_no_full_follows(tmp_path, credit_trace):
    d, c, trace, lines, ok, ops, nb = _transfer(
        tmp_path, credit_trace, ring=4, parallel=True, budget=1.0,
        sm_result=lambda o: (_FakeSm(o, fail_at=3), ""))
    assert "unspecified launch failure" in d
    fulls = [ev for ev in trace if ev[:2] == ("send", "full")]
    assert all(ev[2] < 2 for ev in fulls)    # batches 0..1 went out, nothing after the failure
    assert not ok and c != ""                 # the collector names the missing batch


# -- overlap_ms ------------------------------------------------------------------

def test_overlap_is_the_union_of_the_other_lanes_spans():
    b1._SPANS.clear()
    a = b1.span_open("pa", now=100.0)
    b = b1.span_open("pb", now=105.0)
    c = b1.span_open("pc", now=106.0)
    assert b1.span_close("pb", b, now=112.0) == pytest.approx(7.0)    # a and c running: 105..112
    assert b1.span_close("pc", c, now=108.0) == pytest.approx(2.0)    # a (running) and b: union 106..108
    assert b1.span_close("pa", a, now=110.0) == pytest.approx(5.0)    # b 105..110, c 106..108 inside
    d = b1.span_open("pa", now=200.0)                                  # alone later
    assert b1.span_close("pa", d, now=201.0) == 0.0
    b1._SPANS.clear()


def test_two_lanes_in_threads_report_their_overlap(tmp_path, credit_trace):
    """p0 and p1 of one depositor at the same time (the flip's PP0 form): both
    lines carry overlap_ms, the later one close to the shorter lane's time."""
    b1._SPANS.clear()
    res = {}

    def run(lane, sub):
        res[lane] = _transfer(tmp_path / sub, credit_trace, ring=4, lane=lane, collector_delay=0.15)
    ths = [threading.Thread(target=run, args=("p0", "a")), threading.Thread(target=run, args=("p1", "b"))]
    for th in ths:
        th.start()
    for th in ths:
        th.join(20)
    ov = []
    for lane in ("p0", "p1"):
        d, c, _t, lines, ok, _o, _nb = res[lane]
        assert d == "" and c == "" and ok
        line = [x for x in lines if "lane-time" in x][0]
        ov.append(int(line.split("overlap_ms=")[1].split()[0]))
    assert max(ov) >= 100
    b1._SPANS.clear()


# -- the kernel's host half ---------------------------------------------------------

def test_vector_choice_split_and_grid():
    assert sm.pick_vec(0x1000, 0x2000, 4096) == 16
    assert sm.pick_vec(0x1004, 0x2000, 4096) == 4
    assert sm.pick_vec(0x1001, 0x2000, 4096) == 1
    # a 2-D copy is only as aligned as its pitches
    assert sm.pick_vec(0x1000, 0x2000, 1024, 8, 1024, 1536) == 16
    assert sm.pick_vec(0x1000, 0x2000, 1024, 8, 1024, 1540) == 4
    assert sm.pick_vec(0x1000, 0x2000, 1024, 1, 1024, 1541) == 16    # FLAT ignores pitches
    # a FLAT piece keeps a 16-B body and a byte tail
    assert sm.split_flat(0x1000, 0x2000, 3 * (1 << 20) + 7) == [
        (0x1000, 0x2000, 3 * (1 << 20), 16),
        (0x1000 + 3 * (1 << 20), 0x2000 + 3 * (1 << 20), 7, 1)]
    assert sm.split_flat(0x1000, 0x2000, 64) == [(0x1000, 0x2000, 64, 16)]
    assert sm.split_flat(0x1001, 0x2000, 64) == [(0x1001, 0x2000, 64, 1)]
    assert sm.split_flat(0x1000, 0x2000, 0) == []
    # 3 MiB at 16 B, 4 items per thread, 256 threads: 192 blocks needed, capped
    assert sm.grid_for(3 << 20, 1, 16, 64) == 64
    assert sm.grid_for(3 << 20, 1, 16, 1000) == 192
    assert sm.grid_for(16, 1, 16, 64) == 1
    # a 2-D piece: a block per row, capped
    assert sm.grid_for(4096, 1024, 16, 64) == 64
    assert sm.grid_for(4096, 8, 16, 64) == 8


def test_the_kernel_builds_for_both_cards_without_local_memory():
    try:
        lib = sm.find_libnvrtc()
    except RuntimeError as exc:
        pytest.skip(f"no NVRTC on this desk: {exc}")
    import shutil
    import subprocess
    cuobjdump = shutil.which("cuobjdump") or (
        "/usr/local/cuda/bin/cuobjdump" if os.path.exists("/usr/local/cuda/bin/cuobjdump") else None)
    for arch in ((8, 6), (12, 0)):
        image, kind = sm.compile_kernel(arch, lib)
        assert kind == "cubin" and len(image) > 1000
        if cuobjdump:
            # STACK:0 -- no helper call (a 64-bit division costs sm_86 a 16-B
            # frame), so the launch needs no local memory under H15's limit
            p = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"h22_lane_copy_{os.getpid()}_{arch[0]}{arch[1]}.cubin")
            with open(p, "wb") as fh:
                fh.write(image)
            try:
                res = subprocess.run([cuobjdump, "-res-usage", p], capture_output=True, text=True, timeout=60)
            finally:
                os.unlink(p)
            if res.returncode == 0 and "Function weg2_lane_copy" in res.stdout:
                assert "STACK:0 " in res.stdout and "LOCAL:0 " in res.stdout, res.stdout
    sm._preload_builtins(lib)
    nv = ctypes.CDLL(lib)
    prog = ctypes.c_void_p()
    assert nv.nvrtcCreateProgram(ctypes.byref(prog), sm.KERNEL_SRC.encode(), b"k.cu", 0, None, None) == 0
    opts = (ctypes.c_char_p * 1)(b"--gpu-architecture=compute_120")
    assert nv.nvrtcCompileProgram(prog, 1, opts) == 0
    size = ctypes.c_size_t(0)
    nv.nvrtcGetPTXSize(prog, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    nv.nvrtcGetPTX(prog, buf)
    nv.nvrtcDestroyProgram(ctypes.byref(prog))
    ptx = buf.value.decode()
    assert ".local" not in ptx and "__local_depot" not in ptx     # launches under H15's stack 0
    assert "st.global.wt.v4.u32" in ptx                            # the barlink store form
    assert ".entry weg2_lane_copy" in ptx


# -- the build lives at the lane setup, not in a flip ------------------------------

def test_setup_builds_the_copier_only_under_the_switch(tmp_path, monkeypatch):
    built = []

    def fake_copier_for(device, **_kw):
        built.append(device)
        return SimpleNamespace(arch=(12, 0), image_kind="cubin", build_ms=12.0), ""
    monkeypatch.setattr(sm, "copier_for", fake_copier_for)
    lines = []
    reg = b1.Bar1Lanes("n2", "P", 0, 0, PAIRS, log=lines.append, root=str(tmp_path))
    monkeypatch.setattr(reg, "open_window", lambda lk: None)
    monkeypatch.setattr(reg, "connect_peer",
                        lambda lk: reg.peers.__setitem__(lk, SimpleNamespace(dev_ptr=1, slot_bytes=8, ring=4)))
    reg.setup(["p0", "p1"])
    assert built == [] and not any("sm-copy" in x for x in lines)
    with envs.SGLANG_WEG2_LANE_PARALLEL_COPY.override(True):
        reg.setup(["p0", "p1"])
    assert built == [0]
    assert any("WEG2-BAR1 sm-copy ready device=0 arch=sm_120 image=cubin" in x for x in lines)
    # a refused build is named at the setup and leaves the copy engine
    monkeypatch.setattr(sm, "copier_for", lambda device, **_kw: (None, "RuntimeError: nvrtc"))
    with envs.SGLANG_WEG2_LANE_PARALLEL_COPY.override(True):
        reg.setup(["p0", "p1"])
    assert any("sm-copy REFUSED device=0 RuntimeError: nvrtc -> mode=serial" in x for x in lines)
