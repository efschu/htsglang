"""NF1c 09291739 (rc12z30y3e @ 410bb48e32): D-TP0 died in the sleep flush.

THE METAL. The 32k cell woke D at n=1, S1: the kv_cache tag was still paused,
so ``apply_stage`` gave the GDN temporal pool the plan of one seat (slots
1..7, ``_pdflip_seat_keep=8``) and the saver mapped only that. After the
request ended, the ack flush ran (fine), then the D-MEM-SCHED runtime tick
shrank S1 -> S0 LIVE (17:44:50 ``LIVE-SPANS n=1 stage=S0 rows_on 22->10``).
Its live branch ("the Mamba pool is mapped (live): it keeps its cap form")
set ``keep = pool_size + 1`` and wrote ``_pdflip_seat_keep = None`` -- but a
live pool is never re-planned, its pages stayed those of 1 seat. The sleep's
``flush_cache -> MambaPool.reset_state`` then zeroed the WHOLE temporal
tensor: GPU coredump ``FillFunctor<BFloat16>`` grid 539136x128 (x16 elems =
36 x 39 x 786432 = the full temporal), device exception, 58 GB dump, abort.

WHAT MUST HOLD. ``_pdflip_seat_keep`` names the slots that HAVE pages: it moves
only with a plan that is actually set (a paused pool); a live apply leaves it.
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402

from test_pdflip_d_seat_live_wipe_s1_0928 import (  # noqa: E402
    G, CoreTms, _armed, _kv_leg, _pause_all, _rank, _seats)


def _covered(tms, temporal, keep):
    """Every layer's slots [0, keep) lie inside the saver's extents."""
    ext = tms.allocs[temporal.data_ptr()]["ext"]
    sb = temporal[0, 0].numel() * temporal.element_size()
    S = temporal.shape[1]

    def inside(lo, hi):
        cur = lo
        while cur < hi:
            e = [x for x in ext if x[0] <= cur < x[1]]
            if not e:
                return False
            cur = e[0][1]
        return True

    return all(inside(l * S * sb, l * S * sb + keep * sb) for l in range(temporal.shape[0]))


def _wake_one_seat(tms, r):
    _pause_all(tms)
    dsv.on_wake(r.sched, types.SimpleNamespace(epoch="e1"), None)
    st = dsv.on_wake(r.sched, _kv_leg("e1", 1, 100), _seats(1))
    assert st.stage == 1
    for p in list(tms.allocs):
        tms.resume(p)  # the phase runs: the pool is mapped in its 1-seat plan
    pool = r.sched.tp_worker.model_runner.req_to_token_pool.mamba_pool
    keep = pool._pdflip_seat_keep
    assert keep is not None and keep < pool.size + 1, "the wake did not trim the pool"
    assert _covered(tms, r.temporal, keep) and not _covered(tms, r.temporal, keep + 1)
    return pool, keep


def test_a_live_stage_shrink_keeps_the_pools_seat_keep():
    tms = CoreTms()
    with _armed(tms):
        r = _rank(tms)
        pool, keep = _wake_one_seat(tms, r)
        ctl = r.sched._pdflip_d_seat_vram
        # the D-MEM-SCHED runtime tick: S1 -> S0 while everything is mapped
        ctl.apply_stage(1, 0)
        assert pool._pdflip_seat_keep == keep, (
            "a live apply wrote _pdflip_seat_keep=%r but the pool's pages are still the "
            "1-seat plan (keep=%d): the sleep flush zeroes unmapped slots" % (
                pool._pdflip_seat_keep, keep))
        # and the reset's range is covered by pages
        assert _covered(tms, r.temporal, pool._pdflip_seat_keep or r.temporal.shape[1])


def test_a_live_apply_without_stage_keeps_the_pools_seat_keep():
    tms = CoreTms()
    with _armed(tms):
        r = _rank(tms)
        pool, keep = _wake_one_seat(tms, r)
        r.sched._pdflip_d_seat_vram.apply(1)
        assert pool._pdflip_seat_keep == keep


def test_the_next_paused_plan_still_moves_the_seat_keep():
    tms = CoreTms()
    with _armed(tms):
        r = _rank(tms)
        pool, keep = _wake_one_seat(tms, r)
        _pause_all(tms)  # the next sleep
        dsv.on_wake(r.sched, types.SimpleNamespace(epoch="e2"), None)  # cap form
        assert pool._pdflip_seat_keep is None


def test_the_reset_zeroes_only_what_the_seat_keep_names():
    from flliper.srt.mem_cache.memory_pool import MambaPool

    temporal = torch.ones(2, 6, 3)
    fake = types.SimpleNamespace(
        mamba_cache=MambaPool.State(conv=[torch.ones(2, 6, 4)], temporal=temporal),
        SpeculativeState=MambaPool.SpeculativeState, device="cpu",
        replayssm_write_pos=None, _sync_device=lambda: None, _pdflip_seat_keep=4)
    MambaPool.reset_state(fake)
    assert temporal[:, :4].abs().sum() == 0 and temporal[:, 4:].eq(1).all()


def test_the_live_line_names_the_reset_range(caplog):
    """The metal marker: every LIVE-SPANS line carries ``mamba_keep=`` -- on
    the next boot the tick's line after a 1-seat wake reads the wake's keep,
    never ``all``."""
    import logging

    tms = CoreTms()
    with _armed(tms):
        r = _rank(tms)
        pool, keep = _wake_one_seat(tms, r)
        caplog.set_level(logging.INFO)
        r.sched._pdflip_d_seat_vram.apply_stage(1, 0)
        lines = [m for m in caplog.messages if dsv.LIVE_MARK in m]
        assert lines and ("mamba_keep=%d " % keep) in lines[-1]
