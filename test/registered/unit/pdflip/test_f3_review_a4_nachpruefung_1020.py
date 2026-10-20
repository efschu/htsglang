"""F3 review, second pass (20.10.): A4 -- the deferred unmap on the map thread (2a9be99aa3).

N1 GREEN: an unmap is handed to the map thread only after its fence's E1 completed -- never before.
N2 GREEN: FIFO with the lend's map -- a lend started after a return runs its map behind the unmap (settle joins).
N3 RED on 2a9be99aa3 (Auflage B1): ``_bytes_now`` (the extend / width vote needs the returned bytes NOW) returns
   early when ``pending_releases`` is empty -- but after a ``poll`` the unmap sits in ``release_jobs`` on the map
   thread, not yet done; the width vote and the extend then read the card before the cells went back.
Serial, CPU only, MemoryMax=2G.
"""
from __future__ import annotations

import importlib.util
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402,F401

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("_t_f3_builder_a4", os.path.join(_HERE, "test_f3_lend_no_sync_1020.py"))
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

F, dsv, dtl, rb, h = B.F, B.dsv, B.dtl, B.rb, B.h
ops = B.ops


def test_n1_unmap_reaches_the_map_thread_only_after_e1(ops):
    tms = h.CoreTms()
    ex = B.ManualExecutor()
    with h._armed(tms), B._floor():
        r, ctl = B._lent(tms)
        ctl.release_executor = ex
        mapped = B._bank_mapped(tms, r)
        ctl.unlend()
        for _ in range(3):
            ctl.drain_releases(block=False)          # E1 not done
        assert not ex.jobs and ctl.pending_releases and B._bank_mapped(tms, r) == mapped
        ops.done.add(ctl.pending_releases[0].fence.e1)
        ctl.drain_releases(block=False)
        assert len(ex.jobs) == 1 and B._bank_mapped(tms, r) == mapped   # handed over, not yet run
        ex.run_all()
        ctl.drain_releases(block=False)
        assert not ctl.release_jobs and B._bank_mapped(tms, r) < mapped


def test_n2_a_lend_after_a_return_maps_behind_the_unmap(ops):
    tms = h.CoreTms()
    ex = B.ManualExecutor()
    with h._armed(tms), B._floor():
        r, ctl = B._lent(tms)
        ctl.release_executor = ex
        ctl.unlend()
        ops.complete_all()
        ctl.drain_releases(block=False)
        order = []
        real_join = dsv.SeatVram._join_releases

        def _join(self=ctl):
            ex.run_all()
            order.append("unmap-joined")
            return real_join(self)

        ctl._join_releases = _join
        ctl.start_lend(10 * B.STEP, executor=ex)
        order.append("map-submitted")
        assert order == ["unmap-joined", "map-submitted"]


def test_n3_bytes_now_joins_an_unmap_already_on_the_map_thread(ops):
    """RED (B1)."""
    tms = h.CoreTms()
    ex = B.ManualExecutor()
    with h._armed(tms), B._floor():
        r, ctl = B._lent(tms)
        ctl.release_executor = ex
        ctl.unlend()                                   # return; E1 pending
        ops.complete_all()
        ctl.drain_releases(block=False)                # the pass's poll: handed to the map thread
        assert not ctl.pending_releases and ctl.release_jobs
        joined = []
        real_join = dsv.SeatVram._join_releases

        def _join(self=ctl):
            ex.run_all()
            joined.append(True)
            return real_join(self)

        ctl._join_releases = _join
        dtl._bytes_now(r.sched, ctl, "the pending extend's width vote")
        assert joined and not ctl.release_jobs, \
            "_bytes_now returned with the unmap still on the map thread -- the extend reads the card before the cells went"
