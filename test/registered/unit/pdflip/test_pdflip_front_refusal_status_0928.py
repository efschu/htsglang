# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ==============================================================================
"""413 only for a MEASURED size fault; every state refusal is 503 + Retry-After.

MEASURED, NF boot rc12z30d (``...09282104_37c76c65e9_0928_210437.front.log``):

    Z.1194  PDFLIP X-REQUEUE rid=pdflip-5-10 n=2 verdict=W53
    Z.1196  W53 PdFlipStoreHandbackFailed rid=pdflip-5-10 ...
    Z.1198  "POST /v1/messages?beta=true HTTP/1.1" 413 1733   (request 21:11:35)

and the same shape for pdflip-1-9 (Z.1311-1319): ``leg1_prompt_tokens=17053
d_extent=17053 carrier_max=314553``. D did not find P's KV under the cut -- a
STATE fault -- and the front answered 413 for a 17,053-token prompt on a form
that carries 314,553. Claude Code reads 413 as "Request too large (max 32MB)"
and abandons the whole agent; a 503 it retries.

The rule (``front.py::refusal_status``): 413 only when a MEASURED length
(exact prompt tokens, P's measured count -- never the char estimate) exceeds
the form's STATIC capacity (``carrier_max_tokens``). Anything else is 503 with
Retry-After and the W-code plus the full detail in the body, on both wires.
The decision reads the numbers, never the name of the refusal.

Three sites carried a bare 413 at ca2a9706ec, each pinned here:

  1. ``handle_generate`` ``route == "none"``   -> W52, admission
  2. ``_requeue_after_x_refusal`` exact > carrier -> W52, first refusal
  3. ``_requeue_after_x_refusal`` second refusal  -> W53 (the 21:11:35 path)
"""

import asyncio
import collections
import hashlib
import json
from unittest import mock

from aiohttp import web

from flliper.srt.pdflip import front as front_mod
from flliper.srt.pdflip.front import (HANDBACK_NAME, NO_ROUTE_NAME, Front,
                                   STATE_REFUSAL_RETRY_AFTER_S,
                                   refusal_response, refusal_status)
from flliper.test.ci.ci_register import register_cpu_ci

try:  # package import under pytest, bare name under a flat invocation
    from .test_pdflip_handback_measured_1296 import (BODY_CHARS, SALAD_CPT,
                                                   MeasuredHarness)
except ImportError:  # pragma: no cover - flat invocation
    from test_pdflip_handback_measured_1296 import (BODY_CHARS, SALAD_CPT,
                                                  MeasuredHarness)

#: the rc12z30d form and the pdflip-1-9 prompt, both read off the front log
RC12Z30D_CARRIER_MAX = 314553
RC12Z30D_PROMPT = 17053


# ------------------------------------------------------------ the decision
def test_the_status_is_decided_by_measured_numbers_not_by_the_name():
    # the 21:11:35 case: measured, far inside the form -> state, retry
    assert refusal_status(RC12Z30D_PROMPT, RC12Z30D_CARRIER_MAX) == 503
    # a measured prompt that really exceeds the form -> 413
    assert refusal_status(RC12Z30D_CARRIER_MAX + 1, RC12Z30D_CARRIER_MAX) == 413
    # the boundary belongs to the form: equal still fits
    assert refusal_status(RC12Z30D_CARRIER_MAX, RC12Z30D_CARRIER_MAX) == 503
    # no measurement (an estimate) never yields 413
    assert refusal_status(None, RC12Z30D_CARRIER_MAX) == 503
    # no static capacity configured -> nothing to exceed
    assert refusal_status(RC12Z30D_PROMPT, 0) == 503


def _body(resp: web.Response) -> dict:
    return json.loads(resp.text)


def test_the_503_carries_retry_after_code_and_the_whole_detail_on_both_wires():
    detail = "W53 PdFlipStoreHandbackFailed rid=r: " + "x" * 3000  # no cut
    a = refusal_response("/v1/messages", HANDBACK_NAME, detail, 503,
                         {"leg1_prompt_tokens": RC12Z30D_PROMPT})
    assert a.status == 503
    assert a.headers["Retry-After"] == str(STATE_REFUSAL_RETRY_AFTER_S)
    b = _body(a)
    assert b["type"] == "error"
    # W88/W50 529 (29.09.): overloaded_error is the 529 on this wire
    assert b["error"]["type"] == front_mod.STATE_REFUSAL_ANTHROPIC_TYPE
    assert b["error"]["message"] == detail
    assert b["code"] == HANDBACK_NAME
    assert b["leg1_prompt_tokens"] == RC12Z30D_PROMPT

    o = refusal_response("/v1/chat/completions", HANDBACK_NAME, detail, 503)
    assert o.status == 503
    assert o.headers["Retry-After"] == str(STATE_REFUSAL_RETRY_AFTER_S)
    ob = _body(o)
    assert ob["error"]["message"] == detail
    assert ob["error"]["type"] == "overloaded_error"
    assert ob["error"]["code"] == HANDBACK_NAME


def test_the_413_is_named_and_carries_no_retry_after():
    a = refusal_response("/v1/messages", NO_ROUTE_NAME, "W52 ...", 413)
    assert a.status == 413 and "Retry-After" not in a.headers
    assert _body(a)["error"]["type"] == "request_too_large"
    assert _body(a)["code"] == NO_ROUTE_NAME
    o = refusal_response("/generate", NO_ROUTE_NAME, "W52 ...", 413)
    assert o.status == 413 and "Retry-After" not in o.headers
    assert _body(o)["error"]["code"] == NO_ROUTE_NAME


# ------------------------------------------- site 3: W53, the 21:11:35 path
async def _post_raw(h: MeasuredHarness, mark: str):
    """Like ``Harness.post`` but keeps the headers (Retry-After)."""
    body = {"prompt": f"MARK{mark} " + f"{mark}" * BODY_CHARS}
    async with h.client.post(str(h.server.make_url("/generate")), json=body) as r:
        return r.status, dict(r.headers), await r.text()


def _prompt_sha(mark: str) -> str:
    text = f"MARK{mark} " + f"{mark}" * BODY_CHARS
    return hashlib.sha1(text.encode(errors="replace")).hexdigest()


def test_w53_inside_the_form_is_a_retryable_503_not_a_413():
    """rc12z30d pdflip-1-9 in desk form: P prefilled twice, D refused twice with
    the whole prompt as its extent (the store handed nothing back), and the
    prompt is far inside ``carrier_max``. RED at ca2a9706ec (413)."""

    async def body():
        async with MeasuredHarness(p_cpt=SALAD_CPT, d_cpt=SALAD_CPT,
                                   awake="D", p_concurrency=2, d_bs=4,
                                   tp_prefill_max_tokens=500,
                                   carrier_max_tokens=RC12Z30D_CARRIER_MAX,
                                   flip_min_work_tokens=1,
                                   idle_layout="D") as h:
            h.d.refuse_real_for["hb"] = 9
            status, headers, text = await asyncio.wait_for(_post_raw(h, "hb"), 30.0)
            assert h.front.counters["W53_PdFlipStoreHandbackFailed"] == 1, text[:400]
            assert status == 503, (status, text[:400])
            assert headers.get("Retry-After") == str(STATE_REFUSAL_RETRY_AFTER_S)
            b = json.loads(text)
            assert b["error"]["code"] == HANDBACK_NAME
            assert HANDBACK_NAME in b["error"]["message"]
            assert "requeue_n=2" in b["error"]["message"]
            assert 0 < b["leg1_prompt_tokens"] <= RC12Z30D_CARRIER_MAX
            assert b["carrier_max"] == RC12Z30D_CARRIER_MAX

    asyncio.run(body())


# ---------------------------------- site 2: W52 at the first refusal (requeue)
def test_a_prompt_over_the_form_stays_413_via_the_requeue_w52():
    """The genuine size fault keeps its 413. P's leg 1 records the MEASURED
    prompt (``_note_exact``), D refuses, and the requeue's W52 -- entered only
    with an exact count above ``carrier_max`` -- answers before a second P
    prefill. This is also why W53 never meets an over-the-form prompt in the
    flow: W52 takes it at n=1, and ``refusal_status`` covers W53 by numbers."""

    async def body():
        async with MeasuredHarness(p_cpt=SALAD_CPT, d_cpt=SALAD_CPT,
                                   awake="D", p_concurrency=2, d_bs=4,
                                   tp_prefill_max_tokens=500,
                                   carrier_max_tokens=600,
                                   flip_min_work_tokens=1,
                                   idle_layout="D") as h:
            h.d.refuse_real_for["ex"] = 9
            status, headers, text = await asyncio.wait_for(_post_raw(h, "ex"), 30.0)
            assert h.front.counters["W52_PdFlipNoServiceableRoute"] == 1, text[:400]
            assert h.front.counters.get("W53_PdFlipStoreHandbackFailed", 0) == 0
            assert status == 413, (status, text[:400])
            assert "Retry-After" not in headers
            b = json.loads(text)
            assert b["error"]["code"] == NO_ROUTE_NAME
            assert b["error"]["type"] == "invalid_request_error"
            assert b["carrier_est"] > b["carrier_max"] == 600, b
            assert h.front.exact_tokens.get(_prompt_sha("ex")) == b["carrier_est"]

    asyncio.run(body())


# ------------------------------------------ site 1: W52 at admission ("none")
class _Req:
    path = "/generate"
    path_qs = "/generate"

    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


def _bare_front(carrier_max: int) -> Front:
    """The #1290 end-to-end fixture: just enough Front to reach the verdict."""
    f = Front.__new__(Front)
    f.state = "serving"
    f.awake = "D"
    f.admit_d = True
    f.epoch = 1
    f._rid = 0
    f.counters = collections.Counter()
    f.tp_prefill_max_tokens = 500
    f.carrier_max_tokens = carrier_max
    f.exact_tokens = {}
    f.spans = None
    f.queue = []
    f.vision = front_mod.VISION_MODE_OFF
    f.x_start_tokens = f.tp_prefill_max_tokens
    f.x_exact = False
    return f


def _admit_none(f: Front, text: str):
    async def body():
        with mock.patch.object(front_mod, "serviceable_route",
                               lambda *a, **k: "none"), \
                mock.patch.object(front_mod, "price_remainder",
                                  lambda t, s, epoch=None: (9000, 9000, True)):
            return await asyncio.wait_for(
                f.handle_generate(_Req({"text": text, "stream": False})), 2.0)
    return asyncio.run(body())


def test_w52_at_admission_on_an_estimate_is_503():
    """The front's char estimate is not a measurement: RED at ca2a9706ec."""
    f = _bare_front(carrier_max=600)
    resp = _admit_none(f, "w" * 6000)
    assert f.counters["W52_PdFlipNoServiceableRoute"] == 1
    assert resp.status == 503, resp.status
    assert resp.headers["Retry-After"] == str(STATE_REFUSAL_RETRY_AFTER_S)
    b = json.loads(resp.text)
    assert b["error"]["code"] == NO_ROUTE_NAME
    assert NO_ROUTE_NAME in b["error"]["message"]


def test_w52_at_admission_on_an_exact_count_over_the_form_is_413():
    f = _bare_front(carrier_max=600)
    text = "w" * 6000
    f.exact_tokens[hashlib.sha1(text.encode(errors="replace")).hexdigest()] = 5000
    resp = _admit_none(f, text)
    assert f.counters["W52_PdFlipNoServiceableRoute"] == 1
    assert resp.status == 413, resp.status
    assert "Retry-After" not in resp.headers
    b = json.loads(resp.text)
    assert b["error"]["type"] == "invalid_request_error"
    assert b["carrier_est"] == 5000


register_cpu_ci(__file__)
