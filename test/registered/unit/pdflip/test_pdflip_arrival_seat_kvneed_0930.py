"""NF-STAU-KV (Klasse J, 30.09.): the head of the ARRIVAL-SEAT wait stalls on
its KV need.

Metal (front logs, boots y3u ...0930_002717 and y3v ...0930_005810):
* y3u 00:42:24 pdflip-38-56 ``verdict=wait_seat why=kv kv need=141854 >
  free=131820``: 141854 = 77854 prompt + 64000 (Claude Code's max_tokens);
  76992 of the prompt already on D (presence). It waited 13.6 s and decoded
  47 tokens. D's own admission reserves ``min(max_new_tokens, 4096)`` per
  request and grows past it elastically (pressure park, span retained).
* y3v 01:09:35 pdflip-28-58 ``why=kv need=150598 > free=146048`` (86598 +
  64000) -- then D refused the kept SHORT (x_refusal), it was re-routed
  through P with the stale ``leg1_done=True`` of its skip_leg1 drain, so
  ``needs_p`` said no and the rule's step never saw it: 66.2 s hold
  (``DP-WAIT hold_by=d-work``) while D decoded with free seats, ended only by
  an unrelated arrival's flip.

Pinned here: the KV need is prompt - shared-with-a-running-turn + D's decode
clip (never the client's worst case); a head whose KV still does not fit
parks the youngest running decode that ARRIVED after it (#246) at once,
never an older one, never a park that cannot cover the deficit; a re-routed
pending runs leg 1 again (``leg1_done`` cleared) so the step flips for it."""
from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_arrival_seat_rule_0929 import X, _front, _on, _pending  # noqa: E402

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import arrival_seat_rule as asr  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import phase_policy  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _pbound_flip_now_off(monkeypatch):
    """These tests pin the seat/KV verdict and the dwell holds for requests that
    need P -- the rule PBOUND-FLIP-NOW (default on, 02.10.) replaces; they keep
    covering it as the switch-off path."""
    monkeypatch.setenv("FLLIPER_PDFLIP_PBOUND_FLIP_NOW", "0")


# y3u 00:42:24: ladder 524288-392468=131820, stage free 131648, D clip 4096
Y3U_READING = {"available": 100000, "evictable": 31648, "capacity": 524288,
               "ladder_ceiling": 524288, "ladder_stage": 393216, "ladder_used": 392468,
               "decode_clip": 4096}


# ---------------------------------------------------------------- pure

def test_the_decode_term_is_ds_own_clip_not_the_clients_max_tokens():
    assert asr.decode_part(64000, 2048, 4096) == 4096          # y3u/y3v: Claude Code's 64000
    assert asr.decode_part(256, 2048, 4096) == 256             # a small budget stays itself
    assert asr.decode_part(None, 2048, 4096) == 2048           # no max_tokens: the reserve
    assert asr.decode_part(64000, 2048, None) == 64000         # no clip known: unchanged
    assert asr.kv_need(77854, asr.decode_part(64000, 2048, 4096)) == 81950
    assert asr.kv_fits(81950, Y3U_READING)[0] is True          # the metal verdict was False
    assert asr.kv_fits(141854, Y3U_READING)[0] is False


def test_the_shared_prefix_counts_only_while_the_previous_turn_runs():
    # y3u: prompt 77854, presence 76992 (uncached 862), common with the previous turn 77854
    assert asr.shared_prefix_credit(77854, 862, 77854, True) == 76992
    assert asr.shared_prefix_credit(77854, 862, 70000, True) == 70000
    assert asr.shared_prefix_credit(77854, 862, 77854, False) == 0   # parked/finished: evictable
    assert asr.shared_prefix_credit(77854, None, 77854, True) == 0
    assert asr.kv_free(Y3U_READING) == 131820 and asr.kv_free(None) is None


def test_the_displacement_victim_is_younger_and_can_cover_the_deficit():
    arr = {"old": 10.0, "y1": 30.0, "y2": 40.0}
    tok = {"old": 90000, "y1": 3000, "y2": 5000}
    run = ["old", "y1", "y2", "unstamped"]
    assert asr.kv_displace_victim(20.0, run, arr, tok, 6000) == "y2"   # youngest by arrival
    assert asr.kv_displace_victim(20.0, run, arr, tok, 9000) is None   # 8000 cannot cover it
    assert asr.kv_displace_victim(50.0, run, arr, tok, 100) is None    # only elders: never
    assert asr.kv_displace_victim(20.0, run, arr, tok, 0) is None
    assert asr.kv_displace_victim(None, run, arr, tok, 100) is None


# ---------------------------------------------------------------- front, SHORT path

def test_y3u_the_head_is_granted_at_once_with_the_honest_decode_term(monkeypatch):
    """Red on the base: need=141854 > free=131820 -> it waits (13.6 s on metal)."""
    _on(monkeypatch)
    f = _front(running=["a", "b", "c"], n=5, kv=dict(Y3U_READING))

    async def body():
        return await asyncio.wait_for(
            f._acquire_short_seat("pdflip-38-56", 77854, [], max_tokens=64000), 1.0)

    seat = asyncio.run(body())
    assert seat is not None and seat.rid == "pdflip-38-56"
    assert f.counters["arrival_seat_d_prefill"] == 1 and f.counters["arrival_seat_wait_kv"] == 0
    assert f.counters["arrival_seat_kv_decode_clipped"] == 1


def test_the_prefix_a_running_turn_holds_is_not_counted_twice(monkeypatch):
    _on(monkeypatch)
    small = {"available": 5000, "evictable": 0, "decode_clip": 4096}
    f = _front(running=["pdflip-38-55"], n=5, kv=small)
    f._sess_prev = {"pdflip-38-56": ("pdflip-38-55", 77854)}

    async def body():
        return await asyncio.wait_for(
            f._acquire_short_seat("pdflip-38-56", 77854, [], max_tokens=256, uncached=862), 1.0)

    seat = asyncio.run(body())            # need = 77854 - 76992 + 256 = 1118 <= 5000
    assert seat is not None
    assert f.counters["arrival_seat_kv_shared_tokens"] == 76992
    # the same turn parked (pages evictable, in D's free already): no credit, it waits
    g = _front(running=["pdflip-38-55"], n=5, kv=small)
    g._sess_prev = {"pdflip-38-56": ("pdflip-38-55", 77854)}
    g._asr_st()["parked"]["pdflip-38-55"] = time.time()
    fits, _why, deficit = asyncio.run(g._arrival_seat_kv(77854, 256, "pdflip-38-56", 862))
    assert fits is False and deficit == 77854 + 256 - 5000


def _stamp(f, rid, t):
    f.__dict__.setdefault("_asr_arrive", {})[rid] = t


def test_a_head_whose_kv_does_not_fit_parks_the_youngest_younger_decode(monkeypatch):
    """Red on the base: nothing parks before the 60 s bound (c)."""
    _on(monkeypatch)
    now = time.time()
    kv = {"available": 20000, "evictable": 0, "decode_clip": 4096}
    f = _front(running=["old", "young1", "young2"], n=5, kv=kv,
               admit_t={"old": 1.0, "young1": 2.0, "young2": 3.0})
    _stamp(f, "old", now - 100.0)
    _stamp(f, "head", now - 10.0)
    _stamp(f, "young1", now - 5.0)
    _stamp(f, "young2", now - 4.0)
    seats = {r: F.Seat(f, r, "short", tokens=t) for r, t in
             (("old", 90000), ("young1", 30000), ("young2", 25000))}

    async def body():
        t = asyncio.ensure_future(f._acquire_short_seat("head", 40000, [], max_tokens=64000))
        await asyncio.sleep(0.2)
        return t

    async def run():
        t = await body()
        assert f.rpc_calls, "the head displaced at once, not after the bound"
        path, rb = f.rpc_calls[0]
        assert rb["youngest"] == "young2" and rb["reason"] == asr.REASON_KV
        assert seats["young2"].held is False and seats["old"].held is True
        assert f.counters["arrival_seat_kv_displace"] == 1
        assert len(f.rpc_calls) == 1, "one park per KV tick (cooldown)"
        t.cancel()

    asyncio.run(run())


def test_an_older_running_decode_is_never_displaced(monkeypatch):
    _on(monkeypatch)
    now = time.time()
    f = _front(running=["old"], n=5, kv={"available": 20000, "evictable": 0, "decode_clip": 4096})
    _stamp(f, "old", now - 100.0)
    _stamp(f, "head", now - 10.0)
    F.Seat(f, "old", "short", tokens=90000)

    async def run():
        t = asyncio.ensure_future(f._acquire_short_seat("head", 40000, [], max_tokens=64000))
        await asyncio.sleep(0.2)
        assert not f.rpc_calls and f.counters["arrival_seat_kv_displace"] == 0
        t.cancel()

    asyncio.run(run())


# ---------------------------------------------------------------- front, the step (needs P)

def test_the_step_displaces_before_it_backfills(monkeypatch):
    _on(monkeypatch)
    now = time.time()
    f = _front(running=["young"], n=5, kv={"available": 30000, "evictable": 0, "decode_clip": 4096})
    head = _pending("head", X + 1000, now - 3.0, est_prompt=60000, payload={"max_tokens": 64000})
    tail = _pending("tail", X + 10, now - 2.0, est_prompt=5000, payload={"max_tokens": 256})
    _stamp(f, "head", now - 3.0)
    _stamp(f, "young", now - 1.0)
    F.Seat(f, "young", "short", tokens=40000)
    f.queue = [head, tail]
    res = asyncio.run(f._arrival_seat_step(f.groups["D"], now))
    assert f.rpc_calls and f.rpc_calls[0][1]["youngest"] == "young"
    assert f.counters["arrival_seat_backfill"] == 0, "strict arrival order: no backfill past the head"
    assert res[2] is None


# ---------------------------------------------------------------- y3v: the re-routed kept SHORT

N = 86598
W50 = (b'{"error": {"message": "W50 PdFlipTpPrefillExceeded: this group may prefill at '
       b'most 3246 uncached tokens itself (--tp-prefill-max-tokens); this request\'s '
       b'extent after prefix matching is 86598. Refused by name so the caller re-routes '
       b'it through the prefill group -- never prefilled here silently."}}')


def test_y3v_a_rerouted_kept_short_needs_p_again():
    """Red on the base: leg1_done stayed True -> needs_p False -> no verdict, 66 s hold."""
    from flliper.srt.pdflip.front_tokens import TokenSpans

    async def run():
        f = F.Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
                    carrier_max_tokens=524288, tp_prefill_max_tokens=3246)
        f.epoch = 30
        ids = np.arange(N, dtype=np.int32)
        f.x_exact = True
        f.ftok = SimpleNamespace(ids_for=lambda text: ids)
        f.tspans = TokenSpans(agent_span=False)
        loop = asyncio.get_event_loop()
        p = F.Pending("pdflip-28-58", "/v1/messages", {}, "t" * 10, 0.0, loop.create_future(),
                      est_prompt=N, est_uncached=454, span_known=True, store_span_est=86144,
                      skip_leg1=True, short_kept=True, price_epoch=29)
        p.leg1_done = True    # the P drain of epoch 29: skip_leg1 -> leg1_done, no leg 1 ran
        task = asyncio.create_task(f._requeue_after_x_refusal(
            SimpleNamespace(path="/v1/messages"), p.rid, {}, "t" * 10, False, p, None, W50))
        for _ in range(200):
            if task.done() or f.queue:
                break
            await asyncio.sleep(0.005)
        assert list(f.queue) == [p]
        assert p.leg1_done is False and p.skip_leg1 is False
        assert phase_policy.immediate_park_trigger([p], 3246) is p
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass

    asyncio.run(run())
