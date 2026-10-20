"""S55 (19.10.) P-POOL-FROM-P: the front's H91c overlap budget is P's REALISED KV pool, not the constant 262144.

rc14 boot ..._92dd662831_1019_010742, P epoch 1 (05:18:55-05:20:26Z): PP0 busy 52.8 s of 91 s (58 %), 24.2 s of
``output_commit`` -- at every request switch PP0 waited for PP1/PP2 to finish the predecessor's last chunk, because
the front held the successor (``P-PHASE epoch=1 ... pool_tokens=262144 pool_holds=290``): 199014 + 152764 do not fit
262144. P's own KV sizing on the binding rank was 530944 tokens (``PP0 KV pool sizing: available_bytes=4044095488
(3.766 GiB), cell_size=7616 -> max_total_num_tokens=530944``); the argv pin ``--max-total-tokens 262144`` cut it, and
the front's constant repeated the pin. The front now reads the pool P actually allocated (``/get_server_info``
``max_total_num_tokens``, min-reduced across P's ranks) -- ONE reading, made by the front (SOLL R11) -- and plans
against it; a written ``--p-pool-tokens`` is kept but never above it.

Driven for real: ``phase_policy.p_pool_realised`` / ``resolve_p_pool_tokens``, ``Front._apply_p_pool_info``,
``Front._p_pool_boot`` against a fake HTTP session, and the real ``_p_drain_pool`` with epoch 1's prompt sizes.
"""
from __future__ import annotations

import asyncio
import collections
import inspect
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import front  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402

# rc14 epoch 1, est_prompt per request (front PDFLIP-SERVED prompt_tokens)
EPOCH1 = {"0-21": 199014, "0-22": 152764, "0-23": 147200, "1-24": 166100}


def _info(realised=None, pin=262144):
    """A P /get_server_info body: server_args (the pin) + scheduler_info (the realised pool)."""
    body = {"max_total_tokens": pin, "context_length": 262144, "internal_states": [{}]}
    if realised is not None:
        body["max_total_num_tokens"] = realised
    return body


# ============================================================================ the pure half


def test_realised_is_the_scheduler_value_never_the_argv_pin():
    assert pp.p_pool_realised(_info(524288, pin=524288)) == 524288
    # the pin says 524288, P's own sizing came out smaller: the realised value wins
    assert pp.p_pool_realised(_info(480000, pin=524288)) == 480000
    # only the pin, no scheduler value: no reading (a pin above P's sizing would over-plan)
    assert pp.p_pool_realised(_info(None, pin=524288)) is None


def test_realised_reads_a_dp_list_and_refuses_garbage():
    assert pp.p_pool_realised([_info(393216)]) == 393216
    for bad in (None, [], "x", 7, {}, _info(0), _info(-64), _info(True), _info("lots"), _info(None)):
        assert pp.p_pool_realised(bad) is None, bad


def test_resolve_matrix():
    # the H91 standard-form default (not written): P's realised pool, up AND down
    assert pp.resolve_p_pool_tokens(262144, False, 524288) == (524288, pp.P_POOL_SRC_REALISED)
    assert pp.resolve_p_pool_tokens(262144, False, 200000) == (200000, pp.P_POOL_SRC_REALISED)
    # an operator value: kept below P's pool, clamped above it
    assert pp.resolve_p_pool_tokens(300000, True, 524288) == (300000, pp.P_POOL_SRC_WRITTEN)
    assert pp.resolve_p_pool_tokens(600000, True, 524288) == (524288, pp.P_POOL_SRC_CLAMPED)
    # P not read yet: the configured value (rc14)
    assert pp.resolve_p_pool_tokens(262144, False, None) == (262144, pp.P_POOL_SRC_CONFIGURED)
    assert pp.resolve_p_pool_tokens(262144, True, 0) == (262144, pp.P_POOL_SRC_CONFIGURED)
    # the rule off (qwen27b, --p-pool-tokens 0) stays off whatever P reports
    assert pp.resolve_p_pool_tokens(0, False, 524288) == (0, pp.P_POOL_SRC_OFF)
    assert pp.resolve_p_pool_tokens(0, True, 524288) == (0, pp.P_POOL_SRC_OFF)


# ============================================================================ the front


def _front(configured=262144, written=False):
    f = object.__new__(front.Front)
    f.p_pool_tokens = configured
    f.p_pool_tokens_configured = configured
    f.p_pool_tokens_written = written
    f.p_pool_realised = None
    f.p_pool_src = pp.P_POOL_SRC_CONFIGURED if configured > 0 else pp.P_POOL_SRC_OFF
    f.counters = collections.Counter()
    f.MODE_PARITY_RETRY_S = 0.0
    return f


def test_metal_shape_p_answers_its_pool_and_the_budget_follows():
    """rc14 front: pool_tokens=262144 (standard-form default). P with a two-leg pool answers 524288."""
    f = _front()
    assert front.Front._apply_p_pool_info(f, _info(524288, pin=524288)) is True
    assert (f.p_pool_tokens, f.p_pool_src, f.p_pool_realised) == (524288, pp.P_POOL_SRC_REALISED, 524288)
    assert f.counters["p_pool_from_p_read"] == 1 and f.counters["p_pool_from_p_changed"] == 1


def test_rc14_p_answers_262144_and_nothing_changes():
    f = _front()
    assert front.Front._apply_p_pool_info(f, _info(262144)) is True
    assert f.p_pool_tokens == 262144 and f.counters["p_pool_from_p_changed"] == 0


def test_an_unusable_body_leaves_the_budget_alone():
    f = _front()
    assert front.Front._apply_p_pool_info(f, _info(None, pin=524288)) is False
    assert front.Front._apply_p_pool_info(f, None) is False
    assert f.p_pool_tokens == 262144 and f.p_pool_src == pp.P_POOL_SRC_CONFIGURED


class _Resp:
    def __init__(self, status, body):
        self.status, self._body = status, body

    async def json(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Session:
    """Answers each GET from a script: an Exception instance is raised, a tuple is (status, body)."""

    def __init__(self, script):
        self.script = list(script)
        self.urls = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        step = self.script.pop(0) if self.script else (200, _info(524288))
        if isinstance(step, BaseException):
            raise step
        return _Resp(*step)


def _boot(f, script):
    f.groups = {"P": types.SimpleNamespace(url="http://p:30001"), "D": types.SimpleNamespace(url="http://d:30002")}
    f.session = _Session(script)
    asyncio.run(asyncio.wait_for(front.Front._p_pool_boot(f), 5))
    return f.session


def test_boot_read_retries_until_p_answers_and_reads_only_p():
    f = _front()
    sess = _boot(f, [ConnectionError("P not up"), (503, None), (200, {"internal_states": []}),
                     (200, _info(524288, pin=524288))])
    assert f.p_pool_tokens == 524288 and f.p_pool_src == pp.P_POOL_SRC_REALISED
    assert sess.urls == ["http://p:30001/get_server_info"] * 4


def test_before_p_answered_the_configured_plan_stands():
    """The read never gates the drain: while P is silent the budget is the configured 262144."""
    f = _front()
    f.groups = {"P": types.SimpleNamespace(url="http://p:30001")}
    f.session = _Session([ConnectionError("P not up")] * 1000)
    f.MODE_PARITY_RETRY_S = 0.001

    async def main():
        t = asyncio.ensure_future(front.Front._p_pool_boot(f))
        await asyncio.sleep(0.05)
        seen = f.p_pool_tokens
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        return seen

    assert asyncio.run(main()) == 262144
    assert f.p_pool_src == pp.P_POOL_SRC_CONFIGURED


def test_switch_off_reads_nothing(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_P_POOL_FROM_P", "0")
    f = _front()
    sess = _boot(f, [(200, _info(524288))])
    assert sess.urls == [] and f.p_pool_tokens == 262144


def test_rule_off_reads_nothing():
    f = _front(configured=0)
    sess = _boot(f, [(200, _info(524288))])
    assert sess.urls == [] and f.p_pool_tokens == 0


def test_switch_defaults_on():
    assert envs.FLLIPER_PDFLIP_P_POOL_FROM_P.get() is True


def test_written_value_is_clamped_by_the_boot_read():
    f = _front(configured=600000, written=True)
    _boot(f, [(200, _info(524288))])
    assert (f.p_pool_tokens, f.p_pool_src) == (524288, pp.P_POOL_SRC_CLAMPED)


# ============================================================================ wiring pins


def test_main_marks_written_before_the_defaults_fill_it():
    src = inspect.getsource(front.main)
    i_mark = src.index('_p_pool_written = getattr(args, "p_pool_tokens", None) is not None')
    i_fill = src.index("phase_policy.resolve_front_defaults(args")
    assert i_mark < i_fill
    assert "p_pool_tokens_written=_p_pool_written" in src


def test_the_boot_read_is_a_task_never_awaited_on_the_startup_path():
    src = inspect.getsource(front.Front)
    assert 'app["p_pool_from_p"] = asyncio.create_task(self._p_pool_boot())' in src
    assert "await self._p_pool_boot()" not in src


# ============================================================================ the drain with the read budget


def _drain_order(budget, finish_after=0.05):
    """The real ``_p_drain_pool``; returns the dispatch rounds as lists of rids."""
    rounds = []

    async def main():
        q = collections.deque(EPOCH1)
        cur = []
        loop = asyncio.get_running_loop()
        t0 = loop.time()

        async def one(it):
            cur.append((round((loop.time() - t0) / finish_after), it))
            await asyncio.sleep(finish_after)
            return it

        await front._p_drain_pool(q, 6, one, lambda it: None, lambda: True, max_dispatch=6,
                                  cost=lambda it: EPOCH1[it], budget=budget, stats={})
        by = collections.defaultdict(list)
        for k, it in cur:
            by[k].append(it)
        rounds.extend(by[k] for k in sorted(by))

    asyncio.run(main())
    return rounds


def test_metal_shape_the_read_pool_puts_the_successors_beside_the_predecessor():
    """262144 (rc14): one leg at a time, PP0 drains at every switch. 524288 (P's two-leg pool, read from P):
    0-21, 0-22 and 0-23 go out together (498978), 1-24 as soon as 0-21 returns."""
    assert _drain_order(262144) == [["0-21"], ["0-22"], ["0-23"], ["1-24"]]
    f = _front()
    front.Front._apply_p_pool_info(f, _info(524288, pin=524288))
    assert _drain_order(f.p_pool_tokens) == [["0-21", "0-22", "0-23"], ["1-24"]]


# ============================================================================ S55r: one budget per P phase


def test_one_overlap_budget_per_p_phase():
    """S55r: the reading may land while a P phase runs (it is a free task). The phase's plan, its drain and its
    P-PHASE line take ONE number at the phase start, so the line never prints a budget the drain did not plan
    with; the new value acts from the next P phase on."""
    src = inspect.getsource(front.Front)
    i_cap = src.index("_p_pool_budget = self.p_pool_tokens")
    i_plan = src.index("_overlap_plan = phase_policy.plan_p_overlap(")
    i_drain = src.index("budget=_p_pool_budget, stats=_phase_stats,")
    i_line = src.index("_p_pool_budget, _phase_stats[\"pool_holds\"], len(self.queue))")
    assert i_cap < i_plan < i_drain < i_line
    assert "budget=self.p_pool_tokens" not in src
    assert "self.p_pool_tokens, _phase_stats[\"pool_holds\"]" not in src
