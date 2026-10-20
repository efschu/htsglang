"""NF-R2-DPOOL (20.10.): a kurz request never waits for D's `/server_info` pool read (SOLL R2).

SPECIMENS (rc26, front logs under /spinning/docker-acceptance/nf/evidence/):

* ``..._1020_143107.front.log`` 14:36:00,245 ``ARRIVAL-SEAT rid=pdflip-0-4 verdict=d_prefill`` -> 14:36:01,246
  ``PDFLIP D-POOL UNREADABLE (cause=TimeoutError: ): ... within 1.0 s`` -> same millisecond ``SET-VERDICT ...
  classes=kurz d_decoding=0 seats_free=2 ... waited_s=1.1 verdict=D_NOW why=R2``;
* ``..._1020_142108.front.log`` 14:25:06,177 -> 14:25:07,178 (pdflip-0-4, same UNREADABLE line) and
  14:25:09,025 -> 14:25:10,026 (pdflip-0-5, the second failed read of the boot, its line suppressed);
* rc27 ``..._893d09fa3f_1020_145331.front.log`` 15:10:02,106 ARRIVAL-SEAT pdflip-4-53 (u=2949, X=8671, epoch 4, D
  prefilling pdflip-4-52) -> 15:10:03,107 the same UNREADABLE line -> SET-VERDICT waited_s=1.1 D_NOW R2. The 1.1 s is
  0.05 s (arrival KV read budget) + 1.0 s (this read's timeout) -- mid-operation, not a boot warm-up.

Each wait is exactly ``D_POOL_READ_TIMEOUT_S`` (1.0 s): the seats-vs-pool gate awaited the read inline while D was
inside a prefill pass (D answers `/server_info` only at its pass boundary), and a timed-out read ends in "no reading,
gate off" -- the same verdict, one second later. The read now runs as one shared task; the decision waits at most the
arrival KV read budget (``FLLIPER_PDFLIP_ARRIVAL_KV_READ_BUDGET_S``, the bound the sibling `/server_info` read of the
arrival seat already uses).

Hermetic: no GPU, no model, no network.
"""

import asyncio
import logging
import time

from flliper.srt.pdflip import arrival_seat_rule as _asr
from flliper.srt.pdflip import front as front_mod
from flliper.srt.pdflip.front import Front, Seat


def _reading(available: int, occupied: int, limit: int):
    return {"available": available, "occupied": occupied, "limit": limit, "size": limit, "threshold": 256,
            "pool_id": 1, "phase": "TP", "generation": 1}


class _SlowResp:
    """D inside a pass: the reply comes after ``delay_s`` (or the request times out after it)."""

    def __init__(self, sess, delay_s: float, body, fail: bool):
        self.sess, self.delay_s, self._body, self.fail = sess, delay_s, body, fail
        self.status = 200

    async def json(self):
        return self._body

    async def __aenter__(self):
        await asyncio.sleep(self.delay_s)
        self.sess.answered += 1
        if self.fail:
            raise asyncio.TimeoutError()
        return self

    async def __aexit__(self, *a):
        return False


class _SlowSession:
    def __init__(self, delay_s: float, reading=None, fail: bool = False):
        self.calls = 0
        self.answered = 0
        self.delay_s = delay_s
        self.fail = fail
        self._body = {"internal_states": [{"hicache_prefetch": reading}]}

    def get(self, *a, **k):
        self.calls += 1
        return _SlowResp(self, self.delay_s, self._body, self.fail)


def _front(session):
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=6, carrier_max_tokens=27466)
    f.session = session
    Seat(f, "running", "batch", tokens=1)   # one seat in use: the gate is armed (FIX 5 never-starve exit)
    assert f._d_gate_armed()
    return f


def test_kurz_decision_does_not_wait_for_a_d_inside_its_pass(caplog):
    """The specimen: D answers only after its pass (here 0.6 s, below the 1.0 s timeout). The decision returns
    after the read budget with no reading (gate off for it, never an old number); the read lands for the next."""
    budget = _asr.kv_read_budget_s()
    assert 0.0 < budget < 0.5

    async def body():
        sess = _SlowSession(0.6, _reading(27466, 0, 27466))
        f = _front(sess)
        with caplog.at_level(logging.INFO, logger=front_mod.logger.name):
            t0 = time.monotonic()
            got = await f._d_reading_if_armed()
            waited = time.monotonic() - t0
        assert got is None, "no reading inside the budget = no reading (EZ-S1 A10)"
        assert waited < budget + 0.25, (
            f"the kurz decision waited {waited:.3f} s for D's pool read; SOLL R2 allows no wait "
            f"(rc26 specimens: 1.0-1.1 s = D_POOL_READ_TIMEOUT_S)")
        assert f.counters["d_pool_read_pending"] == 1
        assert "PDFLIP D-POOL READ-PENDING" in caplog.text
        assert "PDFLIP D-POOL UNREADABLE" not in caplog.text, "a pending read is not a failed read"
        assert f._d_token_budget_blocks("r", 10 ** 9, got) is False
        # the read kept running and lands for the NEXT decision
        await f._d_pool_task
        assert sess.calls == 1 and sess.answered == 1
        again = await f._d_reading_if_armed()
        assert again is not None and again["available"] == 27466, "the landed reading arms the gate again"
        assert sess.calls == 1, "a fresh landed reading costs no second round trip"

    asyncio.run(body())


def test_a_prompt_d_still_gates_in_the_same_decision():
    """A D that answers inside the budget (between passes) still decides THIS admission: the gate is not weakened
    where it can be fed."""

    async def body():
        sess = _SlowSession(0.0, _reading(5000, 0, 27466))
        f = _front(sess)
        got = await f._d_reading_if_armed()
        assert got is not None and got["available"] == 5000
        assert f.counters["d_pool_read_pending"] == 0
        assert f._d_token_budget_blocks("r", 9000, got) is True, "the pool cannot take 9000 rows: the seat waits"

    asyncio.run(body())


def test_concurrent_decisions_share_one_read():
    """Several admissions while D is inside its pass: ONE round trip, every decision bounded by the budget."""

    async def body():
        sess = _SlowSession(0.5, _reading(27466, 0, 27466))
        f = _front(sess)
        t0 = time.monotonic()
        outs = await asyncio.gather(*[f._d_reading_if_armed() for _ in range(4)])
        waited = time.monotonic() - t0
        assert outs == [None] * 4
        assert sess.calls == 1, "one in-flight read, not one per decision"
        assert waited < _asr.kv_read_budget_s() + 0.25
        await f._d_pool_task

    asyncio.run(body())


def test_a_read_that_times_out_keeps_its_named_refusal_and_negative_cache(caplog):
    """FIX 5 bookkeeping is unchanged, only off the decision's path: the timed-out read is named once
    (UNREADABLE), remembered for one age window, and suppressed reads are counted apart."""

    async def body():
        sess = _SlowSession(0.2, None, fail=True)
        f = _front(sess)
        with caplog.at_level(logging.WARNING, logger=front_mod.logger.name):
            assert await f._d_reading_if_armed() is None
            await f._d_pool_task
            t_fail_bound = time.time()
            assert f.counters["d_pool_read_failed"] == 1
            assert caplog.text.count("PDFLIP D-POOL UNREADABLE") == 1
            assert "TimeoutError" in caplog.text
            assert f._d_pool_retry_after <= t_fail_bound + front_mod.D_POOL_MAX_AGE_S
            assert f._d_pool_retry_after > t_fail_bound + front_mod.D_POOL_MAX_AGE_S - 0.5
            assert await f._d_reading_if_armed() is None
            assert sess.calls == 1 and f.counters["d_pool_read_suppressed"] == 1

    asyncio.run(body())


def test_the_decision_wait_is_the_arrival_read_budget_not_the_read_timeout(monkeypatch):
    """The bound is the existing arrival KV read budget (one knob for both `/server_info` reads), not a new clock:
    with the budget at 0 the decision does not wait at all; the read timeout stays D_POOL_READ_TIMEOUT_S."""
    monkeypatch.setenv("FLLIPER_PDFLIP_ARRIVAL_KV_READ_BUDGET_S", "0")

    async def body():
        sess = _SlowSession(0.3, _reading(27466, 0, 27466))
        f = _front(sess)
        t0 = time.monotonic()
        assert await f._d_reading_if_armed() is None
        assert time.monotonic() - t0 < 0.15
        await f._d_pool_task

    asyncio.run(body())
