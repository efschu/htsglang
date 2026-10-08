# SPDX-License-Identifier: Apache-2.0
"""#1730 FRONT-REJECT-OVERLONG (deskq/done/1740): in the dual layout the front answers a request whose exact
token count is over the groups' --max-kv-per-request with 400 invalid_request_error before any route, seat or P
leg 1. ENV FLLIPER_PDFLIP_FRONT_REJECT_OVERLONG (default off); inert in the flip form.

The front's exact count (X-EXACT) was 1 above P's in b9o (front prompt=149780 / 133970 vs P 149779 / 133969), and
the scheduler refuses len >= cap -- so the front refuses count - 1 >= cap and quotes P's number.
"""
from __future__ import annotations

import collections
import inspect
import json
import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ENV = "FLLIPER_PDFLIP_FRONT_REJECT_OVERLONG"
CAP = 131072


class _NoTouch:
    """P/D groups, session and queue: any use of them in the gate is a failure ('P untouched')."""

    def __getattr__(self, n):
        raise AssertionError(f"the gate touched {n!r}")


def _front(dual=True, cap=CAP):
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f.dual_layout = dual
    f._store_probe_info = {"context_length": 262144, **({"max_kv_per_request": cap} if cap is not None else {})}
    f.groups = _NoTouch()
    f.session = _NoTouch()
    return f


def _xx(n, mm=False):
    return types.SimpleNamespace(n=n, mm=mm)


def _body(resp):
    return resp.status, json.loads(resp.body)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


def test_switch_off_is_unchanged_even_far_over_the_cap():
    f = _front()
    assert f._overlong_gate("pdflip-0-5", "/v1/messages", _xx(149780)) is None
    assert not f.counters


def test_on_dual_over_cap_is_400_without_touching_p_openai(monkeypatch, caplog):
    monkeypatch.setenv(ENV, "1")
    caplog.set_level(logging.INFO)
    f = _front()
    status, body = _body(f._overlong_gate("pdflip-0-5", "/v1/chat/completions", _xx(149780)))
    assert status == 400
    assert body["error"]["type"] == "invalid_request_error" and body["error"]["code"] == 400
    assert body["error"]["message"] == ("Input length (149779 tokens) exceeds the maximum allowed length "
                                        "(131072 tokens). Use a shorter input or enable --allow-auto-truncate.")
    assert any(m.startswith("#1730 FRONT-REJECT-OVERLONG rid=pdflip-0-5 tokens=149779 cap=131072") for m in caplog.messages)
    assert f.counters["front_reject_overlong"] == 1


def test_on_dual_over_cap_anthropic_wire(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    status, body = _body(_front()._overlong_gate("pdflip-0-4", "/v1/messages", _xx(133970)))
    assert status == 400
    assert body["type"] == "error" and body["error"]["type"] == "invalid_request_error"
    assert "Input length (133969 tokens) exceeds the maximum allowed length (131072 tokens)" in body["error"]["message"]


def test_on_dual_under_cap_is_unchanged(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    f = _front()
    assert f._overlong_gate("r", "/v1/messages", _xx(1000)) is None
    # P's own count would be cap-1 = accepted; the front's is 1 higher
    assert f._overlong_gate("r", "/v1/messages", _xx(CAP)) is None
    assert f._overlong_gate("r", "/v1/messages", _xx(CAP + 1)) is not None   # P len == cap -> P refuses (>=)
    assert not f.counters.get("front_reject_overlong") or f.counters["front_reject_overlong"] == 1


def test_on_flip_form_is_unchanged(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    f = _front(dual=False)
    assert f._overlong_gate("r", "/v1/messages", _xx(10 ** 6)) is None
    assert not f.counters


def test_no_exact_count_no_cap_or_image_request_passes(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert _front()._overlong_gate("r", "/v1/messages", None) is None      # chars/3 estimate: never refuse on it
    assert _front(cap=None)._overlong_gate("r", "/v1/messages", _xx(10 ** 6)) is None
    assert _front()._overlong_gate("r", "/v1/messages", _xx(10 ** 6, mm=True)) is None


def test_wiring_gate_runs_before_any_verdict_seat_or_pending():
    src = inspect.getsource(F)
    g = src.index("_ol_refusal = self._overlong_gate(rid, request.path, _xx)")
    assert src.index("_ctx_refusal = self._context_gate(rid, text, _xx, est_prompt)") < g
    assert g < src.index("route = serviceable_route(remainder, carrier_est,")
    assert g < src.index("p = Pending(rid, request.path, payload, text, time.time(), fut,")


# ---------------------------------------------------------------------------------- #1958 FRONT-CAP-LEVEL-TOP
# Profile top112 (--dual-p-kv-max-tokens 114688): P's pool is 114688 rows -> max_req_len 114687 ->
# max_req_input_len 114682 (tp_worker.get_worker_info: pool - 1 - 5); the scheduler refuses len >= min(114682,
# max_kv_per_request). The front must take the same minimum, from P's own /get_server_info.

P_CAP = 114682


def _front_p(p_cap=P_CAP, max_kv=CAP):
    f = _front(cap=max_kv)
    if p_cap is not None:
        f._overlong_p_input_cap = p_cap
    return f


def test_1958_p_cap_below_max_kv_prompt_between_is_400_with_p_message(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    # b: the 117635-token prompt of the release record: front count 117636 -> P's count 117635
    status, body = _body(_front_p()._overlong_gate("r", "/v1/chat/completions", _xx(117636)))
    assert status == 400
    assert body["error"]["message"] == ("Input length (117635 tokens) exceeds the maximum allowed length "
                                        "(114682 tokens). Use a shorter input or enable --allow-auto-truncate.")


def test_1958_boundary_is_the_schedulers_len_ge_cap(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    f = _front_p()
    assert f._overlong_gate("r", "/v1/messages", _xx(P_CAP)) is None             # P count 114681 -> accepted
    assert f._overlong_gate("r", "/v1/messages", _xx(P_CAP + 1)) is not None     # P count 114682 -> P refuses


def test_1958_effective_cap_is_the_minimum_of_both(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert _front_p(p_cap=200000)._overlong_cap() == CAP        # P roomier: max-kv still rules
    assert _front_p(p_cap=P_CAP)._overlong_cap() == P_CAP
    assert _front_p(p_cap=None)._overlong_cap() == CAP          # P cap not known: unchanged (#1730)
    assert _front_p(p_cap=P_CAP, max_kv=None)._overlong_cap() == P_CAP   # no max-kv flag: scheduler uses ctx, min is P's


def test_1958_p_cap_known_switch_off_or_flip_is_unchanged(monkeypatch):
    f = _front_p()
    assert f._overlong_gate("r", "/v1/messages", _xx(117636)) is None            # switch off
    monkeypatch.setenv(ENV, "1")
    g = _front_p()
    g.dual_layout = False
    assert g._overlong_gate("r", "/v1/messages", _xx(10 ** 6)) is None           # flip form
    assert _front_p()._overlong_gate("r", "/v1/messages", _xx(10 ** 6, mm=True)) is None   # image request
    assert _front_p()._overlong_gate("r", "/v1/messages", None) is None          # no exact count


def test_1958_p_cap_from_info_takes_only_a_positive_int():
    f = F.Front._overlong_p_cap_from_info
    assert f({"max_req_input_len": 114682}) == 114682
    for bad in ({}, {"max_req_input_len": 0}, {"max_req_input_len": -1}, {"max_req_input_len": "x"},
                {"max_req_input_len": True}, {"max_req_input_len": None}, None):
        assert f(bad) == 0


class _Resp:
    def __init__(self, status, body):
        self.status, self._b = status, body

    async def json(self):
        return self._b

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Sess:
    def __init__(self, answers):
        self.answers, self.urls = list(answers), []

    def get(self, url, **kw):
        self.urls.append(url)
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return _Resp(*a)


def _probe_front(answers):
    f = _front()
    f.groups = {"P": types.SimpleNamespace(url="http://p:1"), "D": types.SimpleNamespace(url="http://d:2")}
    f.session = _Sess(answers)
    f.OVERLONG_P_PROBE_SLEEP_S = 0.0
    return f


def test_1958_probe_reads_p_group_not_d_and_retries_until_it_answers(monkeypatch):
    import asyncio

    monkeypatch.setenv(ENV, "1")
    f = _probe_front([RuntimeError("P not up"), (503, {}), (200, [{"max_req_input_len": P_CAP}])])
    asyncio.run(f._overlong_probe_p())
    assert f._overlong_p_input_cap == P_CAP
    assert f.session.urls == ["http://p:1/get_server_info"] * 3     # only P is asked, never D


def test_1958_probe_gives_up_named_and_leaves_the_1730_cap(monkeypatch, caplog):
    import asyncio

    monkeypatch.setenv(ENV, "1")
    caplog.set_level(logging.WARNING)
    f = _probe_front([(200, {})] * 3)
    f.OVERLONG_P_PROBE_TRIES = 3
    asyncio.run(f._overlong_probe_p())
    assert f.__dict__.get("_overlong_p_input_cap", 0) == 0
    assert f._overlong_cap() == CAP
    assert any("#1958" in m for m in caplog.messages)


def test_1958_probe_is_not_started_without_switch_or_in_flip_form(monkeypatch):
    import asyncio

    f = _probe_front([(200, {"max_req_input_len": P_CAP})])
    assert asyncio.run(f._overlong_probe_p()) is None                # switch off
    monkeypatch.setenv(ENV, "1")
    f.dual_layout = False
    asyncio.run(f._overlong_probe_p())                               # flip form
    assert f.session.urls == [] and not f.__dict__.get("_overlong_p_input_cap")


def test_1958_wiring_boot_load_starts_the_probe():
    src = inspect.getsource(F.Front._x_exact_boot_load)
    assert "_overlong_probe_p" in src
    assert src.index("self._store_probe_info = info") < src.index("_overlong_probe_p")
