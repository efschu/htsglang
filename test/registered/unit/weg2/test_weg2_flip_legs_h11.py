"""H11 (24.09.): the non-physics time of the P->D flip legs (x83-x105).

Measured on the first P->D flip of x83/x87/x104/x105 (P PP3, D Form A, 10 %
residence): the legs run 1.9-2.1 s against ~0.9 s of link physics (card 1's
x4 ingress: p0 4.56 GB + p5 1.30 GB at 6.5 GB/s). Three waits that are not
physics, each pinned here by the mechanism that produced it:

1. FIRST TAG: the per-tag plan key (``agreed=None``,
   ``require_agreement=False``) that every deposit (hook=source) and collect
   (hook=authoritative) reads was not among the boot warm-ups; PP0's no-op
   first tag derived it for 106/121/517/134 ms and D's first two collect
   workers derived it side by side (first collect 0.2-0.33 s after the first
   resume).
2. LANE GATE: a host/IPC lane waited for EVERY earlier tag of the wake order,
   i.e. for the other source cards' tags the round-robin order puts between
   two of its own (D TP0 WEG2-TAG-GATE 790-889 ms per flip); and the wake
   pool held exactly the run-ahead bound while bound+1 collects are
   submitted, so the just-resumed tag queued (PP0 weights_1/2 lane p0 waited
   115/139 ms on x105).
3. CREDIT: the waker subtracted the peer's IPC staging from a free reading
   that already lacks it once allocated (x105 D TP2: 943 MiB, credit waits
   311 + 293 ms at weights_5/weights_8 while the card held the bytes).

Hermetic: no device, no process group, the manifest module stubbed.
"""
from __future__ import annotations

import os
import threading
import time

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers import weg2_memory_saver as ms  # noqa: E402
from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from sglang.srt.weg2 import weight_exchange_bounce as bx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402
from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402
from sglang.srt.weg2.lane_turns import LaneTurns  # noqa: E402

MIB = 1 << 20


# ---------------------------------------------------------------- 1. first tag

class _Rank(wu.SchedulerWeightUpdaterManager):
    """A manager with only the fields the warm-up and the cached plan read."""

    def __init__(self, group, rank):
        self._g, self._r = group, rank
        self._weg2_xchg_leg_cache = None

    def _weg2_group_name(self):
        return self._g

    def _weg2_rank(self):
        return self._r

    def _weg2_shadow_manifest(self, group, peer, rank, *, leg, epoch):
        return None, "peer-unready"      # the hook half is not what this pins


def _stub_derivation(monkeypatch, calls, delay_s=0.0, tag="v1"):
    monkeypatch.setattr(xm, "manifests_for_boot", lambda **kw: (["m"], ""))
    monkeypatch.setattr(xm, "join_manifests", lambda mans, **kw: "JOIN")
    monkeypatch.setattr(xm, "plan_from_join", lambda join, *, direction: f"LANES:{direction}")

    def _uncached(self, hook, group, rank, *, agreed=None, require_agreement):
        calls.append((hook, group, rank, agreed, require_agreement, threading.current_thread().name))
        if delay_s:
            time.sleep(delay_s)
        return (f"PLAN:{tag}:{hook}", "")

    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager, "_weg2_shadow_plan_uncached", _uncached)


def test_the_boot_warm_up_fills_the_key_the_deposit_and_the_collect_read(monkeypatch):
    """Bug regression (x83-x105 first tag): after the boot warm-up, the
    per-tag call shape of ``_weg2_xchg_deposit_before_sleep`` (hook=source)
    and ``_weg2_xchg_inject_from_peer`` (hook=authoritative) -- agreed=None,
    require_agreement=False -- is a cache HIT: no derivation at the flip."""
    calls = []
    _stub_derivation(monkeypatch, calls)
    m = _Rank("P", 0)
    out = m._weg2_warm_leg_cache(agree_budget_s=0.0, agree_poll_s=0.0)
    assert out["tag:source"] >= 0.0 and out["tag:authoritative"] >= 0.0
    warm = len(calls)
    assert warm == 2
    for hook in (sh.HOOK_SOURCE, "authoritative"):
        plan, _why = m._weg2_shadow_plan(hook, "P", 0, agreed=None, require_agreement=False)
        assert plan == f"PLAN:v1:{hook}"
    assert len(calls) == warm, "the flip's per-tag call derived again"


def test_a_later_warm_up_round_overwrites_the_tag_plan(monkeypatch):
    """A second round runs because a manifest changed (the drafter rewrites
    its own after the load); the flip must find the NEWER manifests' plan,
    not the first round's."""
    calls = []
    _stub_derivation(monkeypatch, calls, tag="v1")
    m = _Rank("D", 0)
    m._weg2_warm_leg_cache(agree_budget_s=0.0, agree_poll_s=0.0)
    _stub_derivation(monkeypatch, calls, tag="v2")
    m._weg2_warm_leg_cache(agree_budget_s=0.0, agree_poll_s=0.0)
    plan, _ = m._weg2_shadow_plan("authoritative", "D", 0, agreed=None, require_agreement=False)
    assert plan == "PLAN:v2:authoritative"


def test_the_tag_plan_warm_up_is_switchable(monkeypatch):
    calls = []
    _stub_derivation(monkeypatch, calls)
    with envs.SGLANG_WEG2_TAG_PLAN_PREWARM.override(False):
        out = _Rank("P", 1)._weg2_warm_leg_cache(agree_budget_s=0.0, agree_poll_s=0.0)
    assert "tag:source" not in out and calls == []


def test_two_collect_workers_missing_the_same_key_derive_it_once(monkeypatch):
    """Bug regression (x105 D TP0): the collect workers of the first two tags
    both missed the key and derived the same plan side by side. Single
    flight: the second waits for the first and finds the key."""
    calls = []
    _stub_derivation(monkeypatch, calls, delay_s=0.2)
    m = _Rank("D", 0)
    m._weg2_xchg_leg_cache = {}
    got = []

    def _collect():
        got.append(m._weg2_shadow_plan("authoritative", "D", 0, agreed=None,
                                       require_agreement=False))

    ts = [threading.Thread(target=_collect) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(calls) == 1
    assert got[0] == got[1] == ("PLAN:v1:authoritative", "")


# ---------------------------------------------------------------- 2. lane gate

def test_a_lane_waits_only_for_the_earlier_tags_that_ride_it():
    """Derived property: the x105 wake order on D TP0 is 13,9,0,14,10,1 --
    9/0/1 ride c0 (PP0's diagonal), 13/14/10 come from PP2/PP1 over p4/p2.
    Tag 1's c0 run may start as soon as 0 left c0, whatever 14 and 10 do."""
    order = ["weights_13", "weights_9", "weights_0", "weights_14", "weights_10", "weights_1"]
    uses = {0: ["p2", "p4"], 1: ["c0", "p2"], 2: ["c0"], 3: ["p4"], 4: ["p2"], 5: ["c0"]}
    t = LaneTurns(["c0", "p2", "p4"])
    for i in range(len(order)):
        t.register(i)
    for i, lanes in uses.items():      # every collect has started and said its lanes
        t.release(i, used=lanes)
    ok, waited = t.take("c0", 5, timeout_s=0.05)
    assert not ok and waited == (1, 2), "tag 1 must wait for 9 and 0 on c0"
    t.leave("c0", 1)
    t.leave("c0", 2)
    ok, waited = t.take("c0", 5, timeout_s=0.05)
    assert ok, "14 and 10 are still collecting on p4/p2 -- they must not hold c0"


def test_a_tag_that_has_not_said_its_lanes_still_holds_every_lane():
    """The completeness half: until an earlier tag's collect starts (and
    releases what it does not use) it may still ride this lane, so the lane
    waits -- the stream's order is never guessed."""
    t = LaneTurns(["c1"])
    t.register(0)
    t.register(1)
    done = []

    def _later():
        done.append(t.take("c1", 1, timeout_s=5.0))

    th = threading.Thread(target=_later)
    th.start()
    time.sleep(0.1)
    assert not done
    t.release(0, used=["p0"])          # tag 0 turned out to ride p0 only
    th.join(timeout=2.0)
    assert done and done[0][0] is True and done[0][1] == (0,)


def test_the_manager_routes_register_release_and_leave_to_the_turns():
    """Bookkeeping: the wake loop registers in tag order, the collect start
    releases the unused lanes, the lane's end leaves it, the collect's end
    releases everything -- a missed release is a 600 s lane-turn refusal."""
    m = _Rank("D", 0)
    m._weg2_leg_tag_order = ["weights_13", "weights_9", "weights_0"]
    m._weg2_lane_turns = LaneTurns(wu._WEG2_TURN_LANES)
    for tag in m._weg2_leg_tag_order:
        m._weg2_turns_register(tag)
    m._weg2_turns_release("weights_13", used=["p4"])
    m._weg2_turns_release("weights_9", used=["c0"])
    assert m._weg2_lane_turns.take("c0", 2, timeout_s=0.01)[0] is False
    m._weg2_turns_leave("weights_9", "c0")
    assert m._weg2_lane_turns.take("c0", 2, timeout_s=0.01) == (True, ())
    m._weg2_turns_release("weights_13")
    assert m._weg2_lane_turns.take("p4", 2, timeout_s=0.01)[0] is True


def test_the_wake_pool_holds_every_collect_the_run_ahead_bound_submits():
    """Bookkeeping (x105 PP0 weights_1/2, lane p0 115/139 ms): the loop waits
    for tag t-n only AFTER submitting t, so n+1 collects are submitted; the
    pool must carry the spare or tag t queues behind two tags of other
    source cards. The run-ahead rule itself (the VRAM guard of xsn110) is
    unchanged."""
    src = open(wu.__file__).read()
    i = src.index('thread_name_prefix="weg2-wake-collect"')
    blk = src[i - 400:i]
    assert "_n_wake_workers + max(0, _envs_h11.SGLANG_WEG2_WAKE_COLLECT_SPARE.get())" in blk
    assert "_wake_futs[-(_n_wake_workers + 1)][1].result()" in src
    assert envs.SGLANG_WEG2_WAKE_COLLECT_SPARE.get() == 1


# ---------------------------------------------------------------- 3. credit

def _x105_card2(tmp_path, name):
    """x105 D TP2 at weights_5: the counter is short (balance 0), PP2 has two
    c2 stagings of 471.5 MiB booked (943 MiB) -- both allocated long ago."""
    c = ms.VramCredit(name, credit_dir=str(tmp_path))
    c.begin_leg("flip-2")
    c.publish("weights_13", 943 * MIB)
    b1 = ms.StageBooking(c, "ipc-stage", 943 * MIB // 2)
    b2 = ms.StageBooking(c, "ipc-stage", 943 * MIB - 943 * MIB // 2)
    assert c.debit("ipc-stage", 943 * MIB // 2) and c.debit("ipc-stage", 943 * MIB - 943 * MIB // 2)
    return c, (b1, b2)


def test_an_allocated_staging_is_not_subtracted_twice(tmp_path):
    """Bug regression (x105 D TP2, 311 ms at weights_5): free 2003 MiB,
    floor 701, need 868 -- the card holds 1302 MiB above its floor, the peer's
    943 MiB of staging are ALREADY missing from that reading. Pre-H11 the
    waker subtracted them again (359 < 868) and waited for PP2's draft pause."""
    c, bookings = _x105_card2(tmp_path, "GPU-h11-a")
    for b in bookings:
        b.live()
    t0 = time.perf_counter()
    rec = c.wait_for(868 * MIB, budget_s=0.6, tag="weights_5", free_bytes_now=2003 * MIB,
                     free_reader=lambda: 2003 * MIB, floor_bytes=701 * MIB,
                     epoch="flip-2", poll_s=0.05)
    assert time.perf_counter() - t0 < 0.3
    assert rec["claimed_bytes"] == 868 * MIB and rec["waited_s"] == 0.0


def test_a_booked_staging_whose_malloc_has_not_returned_is_still_subtracted(tmp_path):
    """The safety half (xsn269): booked but not yet allocated, the staging
    may still hide in the free reading -- the waker must not take it."""
    c, _bookings = _x105_card2(tmp_path, "GPU-h11-b")   # never marked live
    with pytest.raises(ms.Weg2VramCreditRefused):
        c.wait_for(868 * MIB, budget_s=0.4, tag="weights_5", free_bytes_now=2003 * MIB,
                   free_reader=lambda: 2003 * MIB, floor_bytes=701 * MIB,
                   epoch="flip-2", poll_s=0.05)


def test_a_staging_that_moved_during_the_reading_is_subtracted_whole():
    """Derived property: a debit/refund/mark_live between the two ledger
    records means the free reading may straddle a malloc or a free -- the
    whole booking (the larger record) is subtracted, never less than the
    pre-H11 rule."""
    before = {"staged_bytes": 943 * MIB, "live_bytes": 943 * MIB, "stage_gen": 7}
    same = {"staged_bytes": 943 * MIB, "live_bytes": 943 * MIB, "stage_gen": 7}
    moved = {"staged_bytes": 471 * MIB, "live_bytes": 0, "stage_gen": 8}
    assert ms.staging_to_subtract(before, same) == 0
    assert ms.staging_to_subtract(before, moved) == 943 * MIB
    half = {"staged_bytes": 943 * MIB, "live_bytes": 471 * MIB, "stage_gen": 7}
    assert ms.staging_to_subtract(half, half) == 472 * MIB


def test_the_live_rule_is_switchable(tmp_path):
    c, bookings = _x105_card2(tmp_path, "GPU-h11-c")
    for b in bookings:
        b.live()
    with envs.SGLANG_WEG2_CREDIT_LIVE_STAGING.override(False):
        with pytest.raises(ms.Weg2VramCreditRefused):
            c.wait_for(868 * MIB, budget_s=0.3, tag="weights_5", free_bytes_now=2003 * MIB,
                       free_reader=lambda: 2003 * MIB, floor_bytes=701 * MIB,
                       epoch="flip-2", poll_s=0.05)


class _Ops:
    def __init__(self, fail=False):
        self.fail, self.n = fail, 0

    def raw_malloc(self, device, nbytes):
        if self.fail:
            raise RuntimeError("out of memory")
        self.n += 1
        return 0x1000 * self.n

    def raw_free(self, ptr):
        pass


def test_the_stage_allocator_marks_the_booking_live_and_the_free_refunds_it(tmp_path):
    """Bookkeeping: live exactly while the cudaMalloc'd staging exists --
    marked after the malloc returned, gone with the free (re-stage of the
    same slot included), never live when the malloc failed."""
    c = ms.VramCredit("GPU-h11-d", credit_dir=str(tmp_path))
    c.begin_leg("flip-2")
    c.publish("weights_0", 2000 * MIB)

    def charge(n):
        return ms.StageBooking(c, "ipc-stage", n) if c.debit("ipc-stage", n) else None

    key = ("boot-h11", "c2_s0")
    bx._stage_alloc(_Ops(), 0, key, 400 * MIB, lambda *_a: None, "c2", charge=charge)
    st = c.read()
    assert st["staged_bytes"] == 400 * MIB and st["live_bytes"] == 400 * MIB
    bx._stage_alloc(_Ops(), 0, key, 300 * MIB, lambda *_a: None, "c2", charge=charge)
    st = c.read()
    assert st["staged_bytes"] == 300 * MIB and st["live_bytes"] == 300 * MIB
    with pytest.raises(RuntimeError):
        bx._stage_alloc(_Ops(fail=True), 0, ("boot-h11", "c2_s1"), 200 * MIB,
                        lambda *_a: None, "c2", charge=charge)
    st = c.read()
    assert st["staged_bytes"] == 300 * MIB and st["live_bytes"] == 300 * MIB
    bx.release_stage_buffers("boot-h11")
    st = c.read()
    assert st["staged_bytes"] == 0 and st["live_bytes"] == 0
