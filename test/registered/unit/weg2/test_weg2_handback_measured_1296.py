# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ==============================================================================
"""#1296: the W53 terminal must compare a MEASUREMENT, not a char estimate.

WHAT BOOT weg2sb5h MEASURED (record
`/spinning/gpu-arb/weg2/BOOT_weg2sb5h_0909.md`, front/P/D logs).  One 62k-char
body was offered in two prose styles, 13 requests each:

    class    n   chars    est_uncached  measured tokens  chars/token  outcome
    salad   13   62,019   20,645        22,164-22,578    2.747-2.798  W53 -> 413
    natural 11   62,058   20,670        15,984-18,495    3.355-3.883  W35 -> 503

`est_uncached` is `len(text) / CHARS_PER_TOKEN` with `CHARS_PER_TOKEN = 3.0`
(`front.py`), and 3.0 sits exactly in the gap between the two classes -- max
salad 2.798 < 3.0 < min natural 3.355.  That is the whole discriminator; the
split is 13/11 and not noisy because a constant separated them.

THE STORE HANDED BACK NOTHING FOR EITHER CLASS.  All 24 rids have
`d_extent == P's leg-1 prompt_tokens` exactly and `cached_tokens=0` on that
leg 1 -- D priced the whole prompt as if P had never run, in both classes,
unanimously across its three ranks (D log 77297/77300/77301 natural,
77902/77904/77905 salad: same number from TP0/TP1/TP2, no rank disagreement).
So the W53 CONDITION was true 24 times and the PREDICATE detected it 13 times.
The 11 natural rids looked like a partial handback that never happened, were
re-offered, and spent a second full P leg 1 to reach the same refusal: 14
repeat leg-1 prefills over the arm, 156.1 s of P wall, 237,983 tokens
re-prefilled, `cached_tokens=0` on every one.

THE FIX is one operand (`front.py::Front._requeue_after_x_refusal`): compare
D's measured extent against P's measured leg-1 count
(`Pending.leg1_prompt_tokens`, already written by `Front.leg1`), never against
the front's estimate.  No new bookkeeping, no second store, no new counter.

WHY THIS FILE EXISTS AND THE SLICE-A SUITE DID NOT CATCH IT -- two harness
gaps, both of them load-bearing:

  1. `test_weg2_scheduling_slice_a_0907.py:76-78` -- the D stub's refusal body
     is `{"error": "W50 Weg2TpPrefillExceeded rid=? uncached=99999"}`, which
     does NOT match the producer's wording
     (`scheduler.py::_weg2_answer_x_refusals`: "...extent after prefix
     matching is {n}").  `_d_refusal_extent` therefore returns None on every
     slice-A refusal and the W53 branch has never been EXECUTED by a
     behavioural test -- only asserted on as source text.
  2. `test_weg2_scheduling_slice_a_0907.py:110` -- the stub answers a
     hard-coded `prompt_tokens: 100` whatever the prompt is, so
     estimate-vs-measurement divergence cannot be expressed at all.

Both are closed here by `MeasuredGroup`: D speaks the producer's sentence, and
both groups derive `prompt_tokens` from the prompt's length and a per-group
`chars_per_token`.  The default slice-A stubs are left alone on purpose -- t9c
/ f2a / f2b keep exercising the UNKNOWN-body path they were written for.

DANGER DIRECTION: over-refusing.  A wrong W53 turns a request the base would
serve into a 413, so arms C and D below (a genuine partial handback, and a
CARRIER-EXCEEDS arrival that never had a leg 1) are the mutants that matter.
Arm D is red at the parent for the OPPOSITE reason to arm B: it never ran a
leg 1, and the estimate comparison refused it anyway.
"""

import asyncio
import importlib.util
import inspect
import re
from typing import Dict

from aiohttp import web

from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2.front import (CHARS_PER_TOKEN, HANDBACK_NAME, Front,
                                   _d_refusal_extent)

try:  # this directory is a package (`weg2/__init__.py`), so pytest imports
    # it as `weg2.test_...`; a flat invocation gets the bare name instead.
    from .test_weg2_scheduling_slice_a_0907 import FakeGroup, Harness
except ImportError:  # pragma: no cover - flat invocation
    from test_weg2_scheduling_slice_a_0907 import FakeGroup, Harness


# ----------------------------------------------------------- the producer
def d_refusal_body(x_tokens: int, uncached: int) -> str:
    """D's refusal, in D's OWN words (`scheduler.py::_weg2_answer_x_refusals`).

    Pinned against the producer by
    :func:`test_the_fixture_speaks_the_producers_sentence` so the fixture
    cannot drift into agreeing with the front while the rig does not.
    """
    return (
        f"W50 Weg2TpPrefillExceeded: this group may prefill at most {x_tokens} "
        f"uncached tokens itself (--tp-prefill-max-tokens); this request's "
        f"extent after prefix matching is {uncached}. Refused by name so the "
        f"caller re-routes it through the prefill group -- never prefilled "
        f"here silently."
    )


def test_the_fixture_speaks_the_producers_sentence():
    """The fixture's wording is READ OFF the producer, not remembered.

    A stub that words the refusal its own way is exactly gap 1 above: the
    front's parser answered UNKNOWN for every slice-A refusal, so the branch
    under test was unreachable and its 13/11 split shipped green.
    """
    origin = importlib.util.find_spec("sglang.srt.managers.scheduler").origin
    with open(origin, encoding="utf-8") as fh:
        producer = fh.read()
    # the producer writes the sentence as adjacent f-string literals across
    # four source lines, so the pin has to join implicit concatenations first
    # -- reading it raw is how a fixture "verifies" a sentence that is there.
    joined = re.sub(r'"\s*\n\s*f?"', "", producer)
    assert "extent after prefix matching is {uncached}" in joined, (
        "the producer's sentence moved; this fixture is now fiction")
    assert _d_refusal_extent(d_refusal_body(8742, 18495).encode()) == 18495
    # ... and the slice-A stub's wording is the one that does NOT parse.
    assert _d_refusal_extent(b"W50 Weg2TpPrefillExceeded rid=? uncached=99999") is None


# ---------------------------------------------------------------- fakes
class MeasuredGroup(FakeGroup):
    """A group whose token counts are a FUNCTION OF THE PROMPT.

    Two properties the slice-A stub cannot express, and the sb5h arm turns
    on both:

    * ``chars_per_token`` -- the realised tokenisation of this text.  The
      front prices every prompt at ``CHARS_PER_TOKEN = 3.0`` with no
      tokenizer, so a group above 3.0 makes the front OVER-estimate (natural
      prose) and one below it makes the front UNDER-estimate (token salad).
    * ``handback`` -- the fraction of the prompt D's ``match_prefix`` finds in
      the store.  ``0.0`` is the sb5h reality for both classes; a positive
      value is the partial handback f2a/f2b/t9c protect.
    """

    def __init__(self, name: str, chars_per_token: float = 3.0, **kw):
        super().__init__(name, **kw)
        self.chars_per_token = float(chars_per_token)
        self.handback = 0.0
        self.x_tokens = 0
        #: mark -> how many times to answer the REAL W50 sentence
        self.refuse_real_for: Dict[str, int] = {}

    def measured(self, text: str) -> int:
        """What a tokenizer would return for this text."""
        return int(len(text) / self.chars_per_token)

    async def _generate(self, request: web.Request) -> web.Response:
        payload = await request.json()
        text = front_mod.request_text(payload)
        mark = self._mark(payload)
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        self.timeline.append(f"gen:{mark}")
        self.gen_marks.append(mark)
        try:
            left = self.refuse_real_for.get(mark, 0)
            if left > 0:
                self.refuse_real_for[mark] = left - 1
                self.x_refusals.append(mark)
                uncached = int(self.measured(text) * (1.0 - self.handback))
                return web.json_response(
                    {"error": d_refusal_body(self.x_tokens, uncached)},
                    status=503)
            await asyncio.sleep(self.delay)
            pt = self.measured(text)
            ct = int(pt * self.handback)
            return web.json_response(
                {"choices": [{"text": "x"}],
                 "usage": {"prompt_tokens": pt, "completion_tokens": 1,
                           "prompt_tokens_details": {"cached_tokens": ct}}})
        finally:
            self.in_flight -= 1
            self.timeline.append(f"done:{mark}")


class MeasuredHarness(Harness):
    """The slice-A harness with both groups replaced by :class:`MeasuredGroup`."""

    def __init__(self, p_cpt: float = 3.0, d_cpt: float = 3.0,
                 handback: float = 0.0, **front_kwargs):
        super().__init__(**front_kwargs)
        self.p = MeasuredGroup("P", chars_per_token=p_cpt)
        self.d = MeasuredGroup("D", chars_per_token=d_cpt)
        self.d.handback = handback
        x = int(front_kwargs.get("tp_prefill_max_tokens", 0) or 0)
        self.p.x_tokens = self.d.x_tokens = x


#: One body, two prose styles.  ~3,008 chars: the front prices it at 1,003
#: tokens, salad measures 1,082 and natural 895 -- the sb5h separation
#: (2.75-2.80 vs 3.36-3.88 chars/token) at a size a desk test can run.
BODY_CHARS = 1500
SALAD_CPT = 2.78
NATURAL_CPT = 3.36


def _terms(h: MeasuredHarness, mark: str) -> Dict[str, int]:
    """The three numbers this fault is about, for the posted text."""
    text = f"MARK{mark} " + f"{mark}" * BODY_CHARS
    return {"chars": len(text),
            "est": int(len(text) / CHARS_PER_TOKEN) + 1,
            "measured": h.d.measured(text)}


# ----------------------------------------------- arm A: salad (green today)
def test_arm_a_salad_an_empty_handback_is_terminal_on_the_first_refusal():
    """The class the shipped predicate already caught, kept as the control.

    chars/token 2.78 < 3.0, so the front UNDER-estimates and
    ``d_extent >= est_uncached`` happens to be true.  One P prefill, W53, 413.
    """

    async def body():
        async with MeasuredHarness(p_cpt=SALAD_CPT, d_cpt=SALAD_CPT,
                                   awake="D", p_concurrency=2, d_bs=4,
                                   tp_prefill_max_tokens=500,
                                   flip_min_work_tokens=1,
                                   idle_layout="D") as h:
            t = _terms(h, "sa")
            assert t["measured"] > t["est"], t  # the front under-estimates
            h.d.refuse_real_for["sa"] = 9  # the store never hands anything back
            status, text = await asyncio.wait_for(h.post("sa", chars=BODY_CHARS), 30.0)
            assert status == 413, (status, text[:400])
            assert HANDBACK_NAME in text
            assert h.front.counters["W53_Weg2StoreHandbackFailed"] == 1
            assert h.front.counters.get("W35_Weg2XReQueueLoop", 0) == 0
            assert h.p.gen_marks.count("sa") == 1, h.p.gen_marks

    asyncio.run(body())


# --------------------------------------- arm B: natural (RED at the parent)
def test_arm_b_natural_prose_is_the_same_fault_and_must_reach_the_same_verdict():
    """RED-FIRST, the #1296 falsifier.

    Identical body, identical empty handback, identical D behaviour -- only
    the prose style differs (3.36 chars/token > 3.0, so the front
    OVER-estimates).  At the parent this returns 503 with W35 after a SECOND
    full P prefill, because the estimate made an empty handback look partial.
    """

    async def body():
        async with MeasuredHarness(p_cpt=NATURAL_CPT, d_cpt=NATURAL_CPT,
                                   awake="D", p_concurrency=2, d_bs=4,
                                   tp_prefill_max_tokens=500,
                                   flip_min_work_tokens=1,
                                   idle_layout="D") as h:
            t = _terms(h, "na")
            assert t["measured"] < t["est"], t  # the front OVER-estimates
            h.d.refuse_real_for["na"] = 9
            status, text = await asyncio.wait_for(h.post("na", chars=BODY_CHARS), 30.0)
            assert status == 413, (status, text[:400])
            assert HANDBACK_NAME in text
            assert h.front.counters["W53_Weg2StoreHandbackFailed"] == 1
            assert h.front.counters.get("W35_Weg2XReQueueLoop", 0) == 0
            # THE COST THE ESTIMATE WAS BUYING: sb5h spent 14 of these.
            assert h.p.gen_marks.count("na") == 1, h.p.gen_marks
            assert h.d.x_refusals.count("na") == 1, h.d.x_refusals
            # the refusal carries BOTH numbers, so the estimate's error is
            # visible to the next reader instead of being re-derived.
            m = re.search(r"leg1_prompt_tokens=(\d+)", text)
            assert m and int(m.group(1)) == t["measured"], text[:400]
            assert f"est_uncached={t['est']}" in text, text[:400]

    asyncio.run(body())


def test_the_two_classes_differ_only_in_chars_per_token():
    """The discriminator, stated as an arithmetic fact rather than prose.

    Same chars, same measured-vs-front relationship as the sb5h arm; 3.0 is
    the separator, and it is a constant in the front, not a property of the
    store.
    """
    text = "MARKx " + "x" * BODY_CHARS
    est = int(len(text) / CHARS_PER_TOKEN) + 1
    salad = int(len(text) / SALAD_CPT)
    natural = int(len(text) / NATURAL_CPT)
    assert salad > est > natural, (salad, est, natural)
    assert SALAD_CPT < CHARS_PER_TOKEN < NATURAL_CPT


# ------------------------------------------------- arm C: MUTANT, over-refusing
def test_mutant_arm_c_a_genuine_partial_handback_still_gets_its_re_offer():
    """THE danger direction.  D priced a SMALLER extent than P measured, so
    the store DID hand pages back and the re-offer genuinely serves the
    client (f2a/f2b/t9c).  W53 must stay silent.
    """

    async def body():
        async with MeasuredHarness(p_cpt=SALAD_CPT, d_cpt=SALAD_CPT,
                                   handback=0.6,
                                   awake="D", p_concurrency=2, d_bs=4,
                                   tp_prefill_max_tokens=500,
                                   flip_min_work_tokens=1,
                                   idle_layout="D") as h:
            h.d.refuse_real_for["pa"] = 1  # refuse once, then serve
            status, text = await asyncio.wait_for(h.post("pa", chars=BODY_CHARS), 30.0)
            assert status == 200, (status, text[:400])
            assert h.front.counters.get("W53_Weg2StoreHandbackFailed", 0) == 0
            assert h.front.counters.get("W35_Weg2XReQueueLoop", 0) == 0
            assert h.p.gen_marks.count("pa") == 2, h.p.gen_marks

    asyncio.run(body())


# ------------------------------------------------- arm D: MUTANT, over-refusing
def test_mutant_arm_d_a_carrier_exceeds_arrival_never_ran_a_leg_one():
    """Route CARRIER-EXCEEDS sets ``leg1_done = True`` WITHOUT a leg 1
    (`front.py`, the drain's ``if p.skip_leg1``), so ``leg1_done`` alone does
    not say P ran.  ``leg1_prompt_tokens == 0`` is UNKNOWN and the re-offer
    must survive -- at the parent the estimate comparison refuses it 413 on a
    prefill that never happened.
    """

    async def body():
        async with MeasuredHarness(p_cpt=SALAD_CPT, d_cpt=SALAD_CPT,
                                   awake="P", p_concurrency=2, d_bs=4,
                                   tp_prefill_max_tokens=5000,
                                   carrier_max_tokens=1000,
                                   flip_min_work_tokens=1,
                                   idle_layout="D") as h:
            h.d.refuse_real_for["ce"] = 1
            status, text = await asyncio.wait_for(h.post("ce", chars=BODY_CHARS), 30.0)
            assert status == 200, (status, text[:400])
            assert h.front.counters["route_carrier_exceeds"] == 1
            assert h.front.counters.get("W53_Weg2StoreHandbackFailed", 0) == 0
            assert h.p.gen_marks.count("ce") == 0, (
                "CARRIER-EXCEEDS runs no leg 1 at all", h.p.gen_marks)

    asyncio.run(body())


# ------------------------------------------------- MUTANT: UNKNOWN stays UNKNOWN
def test_mutant_an_unparsable_refusal_body_still_re_offers():
    """The slice-A wording, kept alive on purpose: a body whose extent cannot
    be read is UNKNOWN, and UNKNOWN re-offers.  #1290 round 2's law is
    untouched by #1296 -- only the comparand moved.
    """

    async def body():
        async with Harness(awake="D", p_concurrency=2, d_bs=4,
                           tp_prefill_max_tokens=10,
                           flip_min_work_tokens=1, idle_layout="D") as h:
            h.d.refuse_x_for["un"] = 1  # the `uncached=99999` shape: unparsable
            status, text = await asyncio.wait_for(h.post("un", chars=200), 30.0)
            assert status == 200, (status, text[:400])
            assert h.front.counters.get("W53_Weg2StoreHandbackFailed", 0) == 0
            assert "un" in h.p.gen_marks

    asyncio.run(body())


# ------------------------------------------------------------ source mutants
def test_mutant_the_estimate_is_not_the_comparand_anywhere_in_the_branch():
    src = inspect.getsource(Front._requeue_after_x_refusal)
    assert "d_extent >= measured_whole" in src
    assert "d_extent >= pending.est_uncached" not in src, (
        "#1296: the char estimate may never gate the terminal verdict again")


def test_mutant_zero_measured_tokens_is_unknown_not_empty():
    """The `> 0` guard is the whole of arm D, in source terms."""
    src = inspect.getsource(Front._requeue_after_x_refusal)
    assert "measured_whole > 0" in src
    i = src.find("measured_whole = ")
    assert i > -1
    assert "leg1_prompt_tokens" in src[i:i + 200], src[i:i + 200]


def test_mutant_the_measured_field_is_the_one_leg_one_writes():
    """No second bookkeeping: the comparand is the field `leg1` already sets."""
    leg1 = inspect.getsource(Front.leg1)
    assert "p.leg1_prompt_tokens = pt" in leg1
    assert "self.counters[\"W53" not in leg1, "the counter keeps ONE writer"
