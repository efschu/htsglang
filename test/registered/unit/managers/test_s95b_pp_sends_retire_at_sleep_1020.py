"""S95b: the device frames a P rank sent and still references go at the group's sleep.

Metal (rc22 NF P, boot ..._9c40fe40a2_1020_094747, P.log): PDFLIP-SLEEP-RESIDUE sleep=2 untagged_live PP0 754 /
PP1 721 / PP2 423 MiB after full 16384-token chunks, 106-160 MiB at the other sleeps; the MEMHIST boot
(..._1020_110748) names the holder at a baseline sleep: ``P2PWork.payload <- list <- deque`` 3.8 MiB per frame after
~194-token chunks (194 x 5120 x 2 B x 2 tensors) -- the same frames at 16384 tokens are ~320 MiB each. The group-P
dormant record samples sleep 2; its MAX over 8 boots prices D's expert residency (rc22: TP2 18 -> 7 residents).

What must hold:
(1) the device halves of the lap-late join book and of the P-overlap lead deque are waited (device-side) ONCE and
    released; host halves stay in the book for the lap-late join; empty laps go; a second call is a no-op;
(2) the lap-late join afterwards joins exactly the host halves (never a released frame twice);
(3) THREE RANKS (gloo, real isend/irecv, real P2PWork, the real mixin method): every rank retires at the same
    "sleep", the payloads are freed (weakref dead), and the ring + a collective after it run on every rank alike
    (no hang, no mismatch) -- the retire is rank-local and enters no collective;
(4) the sleep calls it before any pause (release_memory_occupation order), and a raising retire never breaks it.
"""
from __future__ import annotations

import collections
import gc
import inspect
import os
import types
import weakref

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from flliper.srt.distributed.parallel_state import P2PWork  # noqa: E402
from flliper.srt.managers import scheduler_pp_mixin as M  # noqa: E402
from flliper.srt.managers.scheduler_components import weight_updater as WU  # noqa: E402

RETIRE = M.SchedulerPPMixin._pp_retire_device_sends_at_sleep if hasattr(M, "SchedulerPPMixin") else None


def _mixin_cls():
    for name, obj in vars(M).items():
        if isinstance(obj, type) and hasattr(obj, "_pp_retire_device_sends_at_sleep"):
            return obj
    raise AssertionError("no class carries _pp_retire_device_sends_at_sleep")


class _Work:
    def __init__(self):
        self.waits = 0

    def wait(self):
        self.waits += 1
        return True


class _Payload:
    def __init__(self, cuda, n=1024):
        self.is_cuda = cuda
        self._n = n

    def numel(self):
        return self._n

    def element_size(self):
        return 2


def _w(cuda, n=1024):
    return P2PWork(_Work(), _Payload(cuda, n))


def _holder():
    h = types.SimpleNamespace()
    cls = _mixin_cls()
    h.retire = types.MethodType(cls._pp_retire_device_sends_at_sleep, h)
    h.drain = types.MethodType(cls._pp_drain_due_sends, h)
    h.ps = types.SimpleNamespace(pp_size=3)
    h._pp_send_join_lag = types.MethodType(cls._pp_send_join_lag, h)
    h.joined = []
    h._pp_join_comm_work = lambda work: h.joined.append(list(work))
    return h


# (1) ---------------------------------------------------------------------------------------------------------
def test_device_halves_go_host_halves_stay_and_a_second_call_is_a_noop():
    h = _holder()
    d1, d2, d3, g1, g2 = _w(True, 1000), _w(True, 2000), _w(True, 4000), _w(False), _w(False)
    h._pp_pending_sends = {7: [[g1, d1]], 8: [[g2, d2], [d3]]}
    h._pdflip_proxy_lead = collections.deque([[d2], [d3]])      # the same frames, referenced twice
    frames, nbytes, kept = h.retire()
    assert (frames, nbytes, kept) == (3, (1000 + 2000 + 4000) * 2, 2)
    assert [w.work.waits for w in (d1, d2, d3)] == [1, 1, 1], "a frame waited twice or not at all"
    assert g1.work.waits == g2.work.waits == 0, "a host half was waited/dropped"
    assert h._pp_pending_sends == {7: [[g1]], 8: [[g2]]}
    assert not h._pdflip_proxy_lead
    assert h.retire() == (0, 0, 2)                               # idempotent: nothing left to release


def test_a_book_of_device_halves_only_leaves_no_empty_lap():
    h = _holder()
    h._pp_pending_sends = {3: [[_w(True)], [_w(True)]]}
    h._pdflip_proxy_lead = None
    assert h.retire()[0] == 2
    assert h._pp_pending_sends == {}


# (2) ---------------------------------------------------------------------------------------------------------
def test_the_lap_late_join_afterwards_joins_only_the_host_halves():
    h = _holder()
    d, g = _w(True), _w(False)
    h._pp_pending_sends = {1: [[g, d]]}
    h._pp_send_lap = 10                                          # lap 1 is due (10 - 3 >= 1)
    h.retire()
    h.drain()
    assert h.joined == [[g]], h.joined
    assert d.work.waits == 1


# (3) ---------------------------------------------------------------------------------------------------------
def _rank_main(rank, world, port, q):
    try:
        os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
        dist.init_process_group("gloo", rank=rank, world_size=world)
        cls = _mixin_cls()
        h = types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=world), _pp_send_lap=0)
        retire = types.MethodType(cls._pp_retire_device_sends_at_sleep, h)
        nxt, prv = (rank + 1) % world, (rank - 1) % world
        refs = []
        h._pp_pending_sends, h._pdflip_proxy_lead = {}, collections.deque()
        for lap in range(2):                                     # two laps of ring traffic (the last passes)
            frame = torch.full((4096,), float(rank * 10 + lap))
            meta = torch.tensor([lap, rank], dtype=torch.int64)
            w_meta = P2PWork(dist.isend(meta, nxt), meta)
            w_frame = P2PWork(dist.isend(frame, nxt), frame)
            refs.append(weakref.ref(frame))
            h._pp_pending_sends.setdefault(lap, []).append([w_meta, w_frame])
            h._pdflip_proxy_lead.append([w_frame])
            got_m, got_f = torch.empty(2, dtype=torch.int64), torch.empty(4096)
            dist.recv(got_m, prv)
            dist.recv(got_f, prv)                                # the drain: every frame received downstream
            assert int(got_f[0]) == prv * 10 + lap
            del frame
        del w_meta, w_frame                                      # the pass's locals end with the pass
        # THE SLEEP: every rank retires its frames (the host-side "frame" here plays the device half)
        frames, nbytes, kept = retire(releasable=lambda w: w.payload is not None and w.payload.numel() == 4096)
        gc.collect()
        freed = [r() is None for r in refs]
        # after the sleep: the ring and a collective run alike on every rank
        t = torch.tensor([float(rank)])
        w2 = dist.isend(t, nxt)
        r2 = torch.empty(1)
        dist.recv(r2, prv)
        w2.wait()
        s = torch.tensor([1.0])
        dist.all_reduce(s)
        # the lap-late join of the kept host halves still works (no released frame joined)
        for lap in sorted(h._pp_pending_sends):
            for lst in h._pp_pending_sends.pop(lap):
                for w in lst:
                    w.work.wait()
        q.put((rank, frames, nbytes, kept, freed, int(r2[0]), float(s[0])))
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001
        q.put((rank, "ERR", repr(exc)))


def test_three_ranks_retire_alike_free_the_frames_and_keep_the_ring_and_collectives_in_step():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29500 + (os.getpid() % 2000)
    procs = [ctx.Process(target=_rank_main, args=(r, 3, port, q)) for r in range(3)]
    for p in procs:
        p.start()
    out = [q.get(timeout=120) for _ in range(3)]
    for p in procs:
        p.join(timeout=60)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    out.sort()
    for rank, *rest in out:
        assert rest[0] != "ERR", (rank, rest)
        frames, nbytes, kept, freed, r2, s = rest
        assert frames == 2 and nbytes == 2 * 4096 * 4 and kept == 2, (rank, rest)
        assert all(freed), (rank, "a retired frame is still referenced")
        assert r2 == (rank - 1) % 3 and s == 3.0


# (4) ---------------------------------------------------------------------------------------------------------
def test_the_sleep_retires_before_any_pause():
    src = inspect.getsource(WU.WeightUpdater.release_memory_occupation if hasattr(WU, "WeightUpdater")
                            else next(o for o in vars(WU).values() if isinstance(o, type)
                                      and hasattr(o, "release_memory_occupation")).release_memory_occupation)
    i_retire = src.index("self._pdflip_retire_pp_sends_before_sleep()")
    i_drain = src.index("self._pdflip_drain_hicache_before_sleep()")
    i_pause = src.index(".pause(")
    assert i_drain < i_retire < i_pause


def test_a_raising_retire_never_breaks_the_sleep(caplog):
    cls = next(o for o in vars(WU).values() if isinstance(o, type) and hasattr(o, "_pdflip_retire_pp_sends_before_sleep"))
    obj = types.SimpleNamespace()

    def boom():
        raise RuntimeError("ring")
    obj.scheduler = types.SimpleNamespace(_pp_retire_device_sends_at_sleep=boom)
    cls._pdflip_retire_pp_sends_before_sleep(obj)                # no raise
    assert any("PP-SENDS-RETIRE skipped" in m for m in caplog.messages)
    ok = types.SimpleNamespace(scheduler=types.SimpleNamespace(_pp_retire_device_sends_at_sleep=lambda: (2, 4 << 20, 1)))
    import logging
    caplog.set_level(logging.INFO)
    cls._pdflip_retire_pp_sends_before_sleep(ok)
    assert any("PP-SENDS-RETIRED frames=2 mib=4.0 host_kept=1" in m for m in caplog.messages)
