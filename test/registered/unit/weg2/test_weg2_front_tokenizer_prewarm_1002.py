"""FRONT-PREWARM (NF y7y 17:41:47, user 02.10. flip-time rule).

Every NF boot of 02.10. held its first arrivals for the front's tokenizer stack
(weg2-0-1 ``X-EXACT-HOLD waited_ms=4594`` on y7y, 3669-7573 ms on y7w..y7x, 27B
N6i 8468 ms): the front answered 3 s after group D, the stack ~8 s later, and
the host announced ``serving`` in between. Now /weg2/state reports
``warming`` until the X-EXACT load ended -- tokenizer stack ready AND the L3
presence probe opened -- so a host waiting for ``serving`` sends its first
request into a warm front; the load ends with one dummy render + encode per
chat path; the front's own state and its route decisions are unchanged.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time

from sglang.srt.environ import envs
from sglang.srt.weg2 import front as F
from sglang.srt.weg2 import front_tokens as FT

X = 4096


def _boot_front():
    with envs.SGLANG_WEG2_FRONT_EXACT_TOKENS.override(True):
        f = F.Front(prefill="http://p", decode="http://d", awake="D", tag="prewarm",
                    store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
                    weight_chunks=2, tp_prefill_max_tokens=X, flip_min_work_tokens=X)
    f.state = "serving"
    return f


def test_state_is_warming_until_the_load_and_the_probe_are_done():
    """Red on the parent: /weg2/state said serving while the tokenizer loaded."""
    async def go():
        f = _boot_front()
        assert f._x_exact_ready_event() is not None
        before = f.state_dict()["state"]
        f._x_exact_ready_event().set()
        return before, f.state_dict()["state"], f.state

    before, after, own = asyncio.run(go())
    assert before == "warming"
    assert after == "serving"
    assert own == "serving", "the front's own state never changes"


def test_warming_is_bounded_by_the_hold_bound(monkeypatch):
    monkeypatch.setattr(F.Front, "X_EXACT_HOLD_MAX_S", 0.0, raising=False)

    async def go():
        f = _boot_front()
        f._x_exact_ready_event()  # created, never set: the load never ends
        return f.state_dict()["state"]

    assert asyncio.run(go()) == "serving"


def test_switch_off_reports_serving_at_once():
    async def go():
        with envs.SGLANG_WEG2_FRONT_TOKENIZER_PREWARM.override(False):
            f = _boot_front()
            f._x_exact_ready_event()
            return f.state_dict()["state"]

    assert asyncio.run(go()) == "serving"


def test_a_flip_state_is_reported_as_is():
    async def go():
        f = _boot_front()
        f._x_exact_ready_event()
        f.state = "flipping"
        return f.state_dict()["state"]

    assert asyncio.run(go()) == "flipping"


def test_x_exact_off_never_warms():
    async def go():
        with envs.SGLANG_WEG2_FRONT_EXACT_TOKENS.override(False):
            f = F.Front(prefill="http://p", decode="http://d", awake="D", tag="noexact",
                        store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
                        weight_chunks=2, tp_prefill_max_tokens=X, flip_min_work_tokens=X)
        f.state = "serving"
        return f.state_dict()["state"]

    assert asyncio.run(go()) == "serving"


def test_the_hold_is_released_only_after_the_presence_probe_opened(caplog):
    """27B N6i weg2-0-2 (extension 02.10.): a held first request must be priced
    only after the L3 presence probe can run. The boot task sets the release
    event after the whole load -- the probe's open included -- and logs the
    FRONT-PREWARM line naming the probe."""
    caplog.set_level(logging.INFO, logger="weg2.front")
    seen = []

    async def go():
        f = _boot_front()
        evt = f._x_exact_ready_event()

        async def load():
            f.ftok.state = "ready"
            f.store_probe = object()  # the probe opened inside the load
            seen.append(("probe_open", evt.is_set()))

        f._x_exact_boot_load = load
        await f._x_exact_boot()
        seen.append(("after", evt.is_set()))

    asyncio.run(go())
    assert seen == [("probe_open", False), ("after", True)]
    msgs = [r.getMessage() for r in caplog.records]
    line = [m for m in msgs if m.startswith("WEG2 FRONT-PREWARM done after ")]
    assert line and "state=ready" in line[0] and "probe=open" in line[0]


def test_warm_renders_each_chat_path_once_and_names_a_failure():
    ft = FT.FrontTokens()
    calls = []

    def fake(path, payload):
        calls.append(path)
        if path == "/v1/messages":
            raise ValueError("no adapter")
        return None

    ft._count = fake
    ft.warm()
    assert calls == ["/v1/chat/completions", "/v1/messages"]
    assert ft.warm_ms >= 0.0
    assert "/v1/messages ValueError: no adapter" in ft.warm_why
    assert not ft.ids_by_text, "the warm render remembers nothing"


def test_the_load_ends_with_the_warm_render_behind_the_switch():
    src = inspect.getsource(FT.FrontTokens.load)
    i = src.index('self.state = "ready"')
    j = src.index("if envs.SGLANG_WEG2_FRONT_TOKENIZER_PREWARM.get():")
    assert i < j < src.index("self.warm()")


def test_the_switch_defaults_on():
    assert envs.SGLANG_WEG2_FRONT_TOKENIZER_PREWARM.get() is True


def test_wiring_state_dict_reports_the_warming_state():
    assert '"state": self._reported_state()' in inspect.getsource(F.Front.state_dict)
    assert time.monotonic() >= 0
