# SPDX-License-Identifier: Apache-2.0
"""CONTEXT-GATE-KV-CAP (27b-int4ar 23.10., boot ..._216204f8f7_1023_144422, rid pdflip-4-82): a 131360-token prompt
under the context (262144) but over --max-kv-per-request (131072) was routed and died on D (REQ RE-CONSTRUCTED,
#801 aborts) -- Claude Code compacts only on a 400. The CONTEXT-GATE (every form, flip included) now refuses against
min(context_length, max_kv_per_request). ENV FLLIPER_PDFLIP_CONTEXT_GATE_KV_CAP, default on.

Off-by-one: the front's exact count is 1 above the group's (#1730 note), the scheduler refuses group_len >= cap, so
front n > cap is the refusal and n == cap still passes."""
from __future__ import annotations

import collections
import json
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.pdflip import front as F  # noqa: E402

ENV = "FLLIPER_PDFLIP_CONTEXT_GATE_KV_CAP"


def _front(ctx=262144, cap=131072, dual=False):
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f.dual_layout = dual
    info = {"context_length": ctx}
    if cap is not None:
        info["max_kv_per_request"] = cap
    f._store_probe_info = info
    f._dec_gate = lambda *a, **k: None
    return f


def _xx(n):
    return types.SimpleNamespace(n=n, mm=False, unpriced=False, n_src="front_ids")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


def test_flip_form_refuses_the_incident_prompt_with_400():
    f = _front()
    resp = F.Front._context_gate(f, "pdflip-4-82", "x", _xx(131360), 131360)
    assert resp is not None and resp.status == 400
    body = json.loads(resp.body)
    assert body["error"]["code"] == 400 and "131072" in body["error"]["message"]
    assert f.counters["context_gate_refused"] == 1


def test_at_the_cap_still_passes_and_one_above_refuses():
    f = _front()
    assert F.Front._context_gate(f, "r", "x", _xx(131072), 131072) is None   # group sees 131071 < cap
    assert F.Front._context_gate(f, "r", "x", _xx(131073), 131073) is not None


def test_switch_off_keeps_the_context_alone(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    f = _front()
    assert F.Front._context_len(f) == 262144
    assert F.Front._context_gate(f, "r", "x", _xx(131360), 131360) is None


def test_no_cap_or_cap_above_context_changes_nothing():
    assert F.Front._context_len(_front(cap=None)) == 262144
    assert F.Front._context_len(_front(ctx=131072, cap=262144)) == 131072
    assert F.Front._context_len(_front(cap=0)) == 262144


def test_estimate_margin_still_applies_to_an_unpriced_request():
    f = _front()
    # chars/3 estimate: refuses only past cap x 1.6
    assert F.Front._context_gate(f, "r", "x" * 10, None, 150000) is None
    assert F.Front._context_gate(f, "r", "x" * 10, None, int(131072 * 1.6) + 1) is not None
