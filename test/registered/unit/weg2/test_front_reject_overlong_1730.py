# SPDX-License-Identifier: Apache-2.0
"""#1730 FRONT-REJECT-OVERLONG (deskq/done/1740): in the dual layout the front answers a request whose exact
token count is over the groups' --max-kv-per-request with 400 invalid_request_error before any route, seat or P
leg 1. ENV SGLANG_WEG2_FRONT_REJECT_OVERLONG (default off); inert in the flip form.

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

from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ENV = "SGLANG_WEG2_FRONT_REJECT_OVERLONG"
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
    assert f._overlong_gate("weg2-0-5", "/v1/messages", _xx(149780)) is None
    assert not f.counters


def test_on_dual_over_cap_is_400_without_touching_p_openai(monkeypatch, caplog):
    monkeypatch.setenv(ENV, "1")
    caplog.set_level(logging.INFO)
    f = _front()
    status, body = _body(f._overlong_gate("weg2-0-5", "/v1/chat/completions", _xx(149780)))
    assert status == 400
    assert body["error"]["type"] == "invalid_request_error" and body["error"]["code"] == 400
    assert body["error"]["message"] == ("Input length (149779 tokens) exceeds the maximum allowed length "
                                        "(131072 tokens). Use a shorter input or enable --allow-auto-truncate.")
    assert any(m.startswith("#1730 FRONT-REJECT-OVERLONG rid=weg2-0-5 tokens=149779 cap=131072") for m in caplog.messages)
    assert f.counters["front_reject_overlong"] == 1


def test_on_dual_over_cap_anthropic_wire(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    status, body = _body(_front()._overlong_gate("weg2-0-4", "/v1/messages", _xx(133970)))
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
