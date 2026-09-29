# SPDX-License-Identifier: Apache-2.0
"""SK-X: W35 wrong class from NF rc12t (boot 09271756, pdflip-6-30; pdflip-7-31 identical).

The sequence on the metal:
  18:16:36 D-ADMIT SHORT-KEPT (presence 54388, src=d_leg2_cached)
  18:17:06 D refuses before the first byte -> X-REQUEUE n=1, W50-REROUTE path=fresh
  18:17:44 SHORT-KEPT-REPRICE est_uncached 54388 -> 0 src=d_leg2_cached "-> D" -- P never prefilled
  18:17:45 D-ADMIT, HOLD-REFETCH zero-answer ... 18:19:30 X-GATE 54388 > X -> W31
  -> the front counted a second W31 "after a full P prefill" -> W35, though P never ran.

(1) the reroute voids the kept presence: no longer kept, leg 1 on P; the re-price credits only D
    evidence recorded after the refusal. (2) W35 only when P's leg 1 actually ran for the rid; a
    rid P never ran gets the regular P reroute (once -- a third refusal is W35 whatever ran).
Hermetic, CPU: the real Front._requeue_after_x_refusal / _sk_admission_reprice, the real TokenSpans.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip.front_tokens import TokenSpans  # noqa: E402

X = 4096
N = 54388
RID = "pdflip-6-30"
TEXT = "t" * 10
W50 = (b'{"error": {"message": "W50 PdFlipTpPrefillExceeded: this group may prefill at '
       b'most 4096 uncached tokens itself (--tp-prefill-max-tokens); this request\'s '
       b'extent after prefix matching is 54388. Refused by name so the caller re-routes '
       b'it through the prefill group -- never prefilled here silently."}}')
IDS = np.arange(N, dtype=np.int32)


def _front(epoch=30) -> F.Front:
    f = F.Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
                carrier_max_tokens=262144, tp_prefill_max_tokens=X)
    f.epoch = epoch
    f.x_exact = True
    f.ftok = SimpleNamespace(ids_for=lambda text: IDS)
    f.tspans = TokenSpans(agent_span=False)
    # 18:16:36: D's measured leg-2 cached prefix -> the kept SHORT's presence
    f.tspans.record_presence(IDS, N, prompt_tokens=N)
    return f


def _kept(loop, epoch=30) -> F.Pending:
    p = F.Pending(RID, "/v1/messages", {}, TEXT, 0.0, loop.create_future(),
                  est_prompt=N, est_uncached=0, span_known=True, store_span_est=N,
                  skip_leg1=True, short_kept=True, price_epoch=epoch)
    return p


async def _refuse(f, p, body=W50):
    """D's refusal before the first byte, then give the reroute a moment."""
    req = SimpleNamespace(path="/v1/messages")
    task = asyncio.create_task(
        f._requeue_after_x_refusal(req, RID, {}, TEXT, False, p, None, body))
    for _ in range(200):
        if task.done() or f.queue:
            break
        await asyncio.sleep(0.005)
    return task


async def _drop(task):
    if not task.done():
        task.cancel()
    try:
        return await task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        return None


def test_the_metal_sequence_reroute_voids_the_kept_presence():
    async def run():
        f = _front()
        p = _kept(asyncio.get_event_loop())
        task = await _refuse(f, p)
        assert list(f.queue) == [p], "rerouted through P"
        assert p.short_kept is False and p.skip_leg1 is False, "leg 1 on P, not skipped"
        assert p.sk_void_seq == f.tspans.seq
        assert f.counters["short_kept_void_reroute"] == 1
        # the P drain now prices it as a real P prefill (no CARRIER-EXCEEDS skip)
        assert F.phase_policy.p_request_cost(p.est_prompt, p.leg1_prompt_tokens, p.skip_leg1) > 0
        await _drop(task)

    asyncio.run(run())


def test_reprice_after_the_reroute_does_not_credit_the_stale_presence():
    # 18:17:44 red on the base: 54388 -> 0 src=d_leg2_cached "-> D". Even if the rid were still
    # kept, the re-price after a refusal credits only D evidence recorded after it.
    async def run():
        f = _front()
        p = _kept(asyncio.get_event_loop())
        task = await _refuse(f, p)
        await _drop(task)
        f.queue.clear()
        f.epoch = 31  # the flip between the reroute and the D admission
        p.short_kept = True  # defense: whatever kept it again, the void holds
        assert f._sk_admission_reprice(p) is True, "no fresh D confirmation: the P reroute stands"
        assert p.est_uncached == N
        assert list(f.queue) == [p] and p.skip_leg1 is False

    asyncio.run(run())


def test_a_fresh_d_confirmation_after_the_reroute_may_price_it_back_to_d():
    async def run():
        f = _front()
        p = _kept(asyncio.get_event_loop())
        task = await _refuse(f, p)
        await _drop(task)
        f.queue.clear()
        f.tspans.record_presence(IDS, N, prompt_tokens=N)  # D measured it again, after the refusal
        p.short_kept = True
        assert f._sk_admission_reprice(p) is False, "fresh D presence: D"
        assert p.est_uncached == 0 and not f.queue

    asyncio.run(run())


def test_same_epoch_reprice_does_not_skip_the_void():
    async def run():
        f = _front()
        p = _kept(asyncio.get_event_loop())
        task = await _refuse(f, p)
        await _drop(task)
        f.queue.clear()
        p.short_kept, p.price_epoch = True, f.epoch
        assert f._sk_admission_reprice(p) is True

    asyncio.run(run())


def test_second_refusal_without_a_p_leg_is_not_w35_but_the_p_reroute():
    async def run():
        f = _front()
        p = _kept(asyncio.get_event_loop())
        await _drop(await _refuse(f, p))
        f.queue.clear()
        # 18:19:30: D refuses again; P never ran for this rid (leg1_ran False)
        task = await _refuse(f, p)
        assert not task.done(), "not answered W35: re-queued"
        assert list(f.queue) == [p]
        assert f.counters["W35_PdFlipXReQueueLoop"] == 0
        assert f.counters["x_requeue_p_never_ran"] == 1
        await _drop(task)

    asyncio.run(run())


def test_second_refusal_after_a_real_p_leg_is_w35():
    async def run():
        f = _front()
        p = _kept(asyncio.get_event_loop())
        await _drop(await _refuse(f, p))
        f.queue.clear()
        p.leg1_ran, p.leg1_done = True, True
        task = await _refuse(f, p)
        resp = await task
        assert resp.status == 503 and "W35" in json.loads(resp.body)["error"]
        assert f.counters["W35_PdFlipXReQueueLoop"] == 1

    asyncio.run(run())


def test_third_refusal_is_w35_whatever_ran():
    async def run():
        f = _front()
        p = _kept(asyncio.get_event_loop())
        f._x_requeues[RID] = 2
        task = await _refuse(f, p)
        resp = await task
        assert resp.status == 503
        assert f.counters["W35_PdFlipXReQueueLoop"] == 1

    asyncio.run(run())


def test_leg1_marks_that_p_ran():
    import inspect

    src = inspect.getsource(F.Front.leg1)
    i = src.index('raise RuntimeError(f"leg1 on P returned {status}')
    assert src.index("p.leg1_ran = True") > i


def test_since_seq_filters_older_records_only():
    ts = TokenSpans(agent_span=False)
    ts.record_presence(IDS, 100)
    s = ts.seq
    assert ts.pending(IDS, since_seq=s)[1] == 0
    assert ts.pending(IDS)[1] == 100
    ts.record_presence(IDS, 200)
    assert ts.pending(IDS, since_seq=s)[1] == 200
    ts.record_presence(IDS, 0)  # a measured zero retracts
    assert IDS.tobytes() and ts.pending(IDS)[1] == 0 and not ts.entry_seq
