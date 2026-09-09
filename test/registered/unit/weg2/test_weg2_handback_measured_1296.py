# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ==============================================================================
"""#1296: WHERE the W53 terminal fires, and WHAT it is allowed to compare.

TWO FAULTS, ONE SITE (`front.py::Front._requeue_after_x_refusal`).

FAULT 1 -- THE COMPARAND (round 1, still fixed here).  Boot weg2sb5h offered
one 62k-char body in two prose styles, 13 requests each:

    class    n   chars    est_uncached  measured tokens  chars/token
    salad   13   62,019   20,645        22,164-22,578    2.747-2.798
    natural 13   62,058   20,670        15,984-18,495    3.355-3.883

`est_uncached` is `len(text) / CHARS_PER_TOKEN` with `CHARS_PER_TOKEN = 3.0`,
and 3.0 sits exactly in the gap (max salad 2.798 < 3.0 < min natural 3.355).
The store handed back NOTHING for either class -- all 24 rids that reached a
refusal have `d_extent == P's leg-1 prompt_tokens` exactly, `cached_tokens=0`,
unanimous across D's three ranks.  So the CONDITION held 24 times and the
estimate-based PREDICATE detected it 13 times.  The comparand must be P's
measured count (`Pending.leg1_prompt_tokens`, already written by `Front.leg1`).

FAULT 2 -- THE PLACE (round 2, the refutation of round 1).  Round 1 kept
#1291's first-refusal terminal and only swapped the operand.  The metal says
that terminal is wrong at n=1 for BOTH classes.  Of the 13 sb5h rids that were
re-offered instead of refused, TWO paid:

    weg2-28-259  leg 1 pt=18559 ct=0 | offer 1 uncached=18559 REFUSED
                 -> re-offer -> offer 2 cached_tokens=18557 uncached=2, 200
    weg2-12-235  leg 1 pt=16522 ct=0 | offer 1 uncached=16522 REFUSED
                 -> re-offer -> offer 2 cached_tokens=8190 (PARTIAL)

`weg2-28-259` is the ONLY 200 that population produced.  A terminal at the
first refusal deletes it.  The store read lands BETWEEN the two offers, so at
the first refusal an empty handback is indistinguishable BY EXTENT from the
not-yet-landed read f2a/f2b/t9c already protect -- and D's witness census for
the boot (`state=unprobed` 186 / `state=cold` 9, `X-DEFER` 0) offers no
read-state signal to gate on instead.  #1291's own justification carries the
same refutation: the weg2sb5g figures it quotes are "3 served out of 53
(natural 1/27, salad 2/26)", three requests that exist only because the
re-offer ran.

THE FIX: W53 moves to the SECOND refusal, where W35 already stands, and W35's
bare 503 becomes W53's named, measured 413.  W52 (`carrier_est > carrier_max`)
stays terminal at n=1 -- that one is STRUCTURAL, no pass can move it, which is
exactly the distinction round 1 lost.  No new bookkeeping, no new counter, no
second store read.  W53 becomes a SUBSET of W35 (its population), so both
denominators are readable off the census.

WHY THE SLICE-A SUITE DID NOT CATCH FAULT 1 -- two harness gaps, both
load-bearing:

  1. `test_weg2_scheduling_slice_a_0907.py:76-78` -- the D stub's refusal body
     is `{"error": "W50 Weg2TpPrefillExceeded rid=? uncached=99999"}`, which
     does NOT match the producer's wording
     (`scheduler.py::_weg2_answer_x_refusals`: "...extent after prefix
     matching is {n}").  `_d_refusal_extent` therefore returns None on every
     slice-A refusal and the W53 branch had never been EXECUTED behaviourally.
  2. `test_weg2_scheduling_slice_a_0907.py:110` -- the stub answers a
     hard-coded `prompt_tokens: 100` whatever the prompt is, so
     estimate-vs-measurement divergence cannot be expressed at all.

Both are closed by `MeasuredGroup`.  AND THE GAP THAT HID FAULT 2: no fixture
could express a store read that LANDS BETWEEN TWO OFFERS -- `handback` was a
constant.  `MeasuredGroup.handback_after` closes that; arms E and E2 are the
two sb5h rids above, and they are the regression pin for the deleted 200.
The default slice-A stubs are left alone on purpose -- t9c / f2a / f2b keep
exercising the UNKNOWN-body path they were written for.

DANGER DIRECTION: over-refusing.  A wrong W53 turns a request the base would
serve into a 413.  Arms C, D, E and E2 are the mutants that matter, and
`test_mutant_w53_can_never_fire_on_the_first_refusal` is the direct pin
against restoring the round-1 defect.
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
        #: mark -> the handback fraction that applies ONCE this mark has been
        #: refused at least once.  This is the sb5h shape the old fixture
        #: could not express: at offer 1 the store has nothing, at offer 2 the
        #: leg-1 write has landed and `match_prefix` finds it (weg2-28-259
        #: 0/18559 then 18557/18559; weg2-12-235 0/16522 then 8190/16522).
        self.handback_after: Dict[str, float] = {}
        self.refused_once: set = set()

    def _handback_for(self, mark: str) -> float:
        if mark in self.refused_once and mark in self.handback_after:
            return self.handback_after[mark]
        return self.handback

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
                uncached = int(self.measured(text) * (1.0 - self._handback_for(mark)))
                self.refused_once.add(mark)
                return web.json_response(
                    {"error": d_refusal_body(self.x_tokens, uncached)},
                    status=503)
            await asyncio.sleep(self.delay)
            pt = self.measured(text)
            ct = int(pt * self._handback_for(mark))
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


# ------------------------------------------------ arm A: salad, never lands
def test_arm_a_salad_an_empty_handback_is_terminal_only_after_the_re_offer():
    """The class the shipped predicate caught -- now via the re-offer.

    chars/token 2.78 < 3.0, so the front UNDER-estimates.  The store never
    hands anything back, so the one re-offer is spent and the SECOND refusal
    is the named, measured 413.  RED at the parent for the placement: the
    parent refuses at n=1 after a single P prefill, which is exactly the lap
    that would have deleted weg2-28-259 had the estimate fallen the other way.
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
            # THE PLACEMENT: the terminal is the SECOND refusal, and W53 is a
            # subset of W35's population, never a replacement for it.
            assert h.front.counters["W35_Weg2XReQueueLoop"] == 1
            assert "requeue_n=2" in text, text[:400]
            assert h.p.gen_marks.count("sa") == 2, h.p.gen_marks
            assert h.d.x_refusals.count("sa") == 2, h.d.x_refusals

    asyncio.run(body())


# ---------------------------------------- arm B: natural prose, never lands
def test_arm_b_natural_prose_is_the_same_fault_and_must_reach_the_same_verdict():
    """FAULT 1's red arm: identical body, identical empty handback, only the
    prose style differs (3.36 chars/token > 3.0, so the front OVER-estimates).

    At the base this ends 503/W35 after the same two P prefills, because the
    estimate made an empty handback look partial.  The two classes must now be
    indistinguishable -- same route, same verdict, same counters as arm A.
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
            assert h.front.counters["W35_Weg2XReQueueLoop"] == 1
            assert h.p.gen_marks.count("na") == 2, h.p.gen_marks
            # the refusal carries BOTH numbers, so the estimate's error is
            # visible to the next reader instead of being re-derived.
            m = re.search(r"leg1_prompt_tokens=(\d+)", text)
            assert m and int(m.group(1)) == t["measured"], text[:400]
            assert f"est_uncached={t['est']}" in text, text[:400]
            # ... and the front's X is labelled as the FRONT's, because D
            # enforced a different one for all of sb5h (SECTION 1ax-b 7).
            assert "front_X=" in text, text[:400]

    asyncio.run(body())


# ================= arm E: THE FALSIFIER -- the read lands between the offers
def test_arm_e_a_read_that_lands_between_the_offers_must_still_be_served():
    """weg2-28-259, the request a first-refusal terminal deletes.

    Offer 1: the store has nothing, D prices the WHOLE prompt, refuses.  That
    is byte-identical, in every quantity the front can see, to arm A's
    permanently-empty handback.  Offer 2: the leg-1 write has landed and D
    serves with `cached_tokens` ~= the whole prompt.

    On the metal this is 1 of 13 re-offers and the ONLY 200 that population
    produced.  RED at round 1 (`c4a4e5f4`): 413 at the first refusal, request
    destroyed.
    """

    async def body():
        async with MeasuredHarness(p_cpt=NATURAL_CPT, d_cpt=NATURAL_CPT,
                                   awake="D", p_concurrency=2, d_bs=4,
                                   tp_prefill_max_tokens=500,
                                   flip_min_work_tokens=1,
                                   idle_layout="D") as h:
            h.d.refuse_real_for["la"] = 1        # refused once ...
            h.d.handback_after["la"] = 0.999     # ... then the read lands
            status, text = await asyncio.wait_for(h.post("la", chars=BODY_CHARS), 30.0)
            assert status == 200, (status, text[:400])
            assert h.front.counters.get("W53_Weg2StoreHandbackFailed", 0) == 0
            assert h.front.counters.get("W35_Weg2XReQueueLoop", 0) == 0
            # the first refusal really did look empty: D priced the whole
            # prompt, so `d_extent >= measured_whole` held at n=1 and only the
            # PLACEMENT of the terminal saved this request.
            t = _terms(h, "la")
            assert h.d.x_refusals.count("la") == 1, h.d.x_refusals
            assert h.p.gen_marks.count("la") == 2, h.p.gen_marks
            assert t["measured"] > 0

    asyncio.run(body())


def test_arm_e2_a_partial_landing_between_the_offers_is_not_a_handback_failure():
    """weg2-12-235: the read lands PARTIALLY between the offers (8190/16522).

    D's second answer prices a smaller extent, so this is the f2a/f2b/t9c
    shape arriving one offer late.  It must not be refused, and W53 must stay
    silent on it.  RED at round 1 for the same reason as arm E.
    """

    async def body():
        async with MeasuredHarness(p_cpt=NATURAL_CPT, d_cpt=NATURAL_CPT,
                                   awake="D", p_concurrency=2, d_bs=4,
                                   tp_prefill_max_tokens=500,
                                   flip_min_work_tokens=1,
                                   idle_layout="D") as h:
            h.d.refuse_real_for["pl"] = 1
            h.d.handback_after["pl"] = 0.5
            status, text = await asyncio.wait_for(h.post("pl", chars=BODY_CHARS), 30.0)
            assert status == 200, (status, text[:400])
            assert h.front.counters.get("W53_Weg2StoreHandbackFailed", 0) == 0
            assert h.p.gen_marks.count("pl") == 2, h.p.gen_marks

    asyncio.run(body())


def test_arm_f_a_partial_handback_refused_twice_is_named_w35_not_a_false_w53():
    """INSTRUMENT-LIES GUARD, and the behavioural kill for the comparand.

    Moving the terminal to n>1 makes both branches refuse, so a wrong
    predicate no longer destroys a request -- it publishes a FALSE SENTENCE
    instead.  W53's text asserts "the store handed back NOTHING"; here the
    store hands back 60 % on both offers, so that claim would be a lie logged
    as evidence, and the next reader would go hunting a store fault that does
    not exist (`d_extent=433 leg1_prompt_tokens=1082`).

    Without this arm the comparand is pinned only by a source assertion.
    """

    async def body():
        async with MeasuredHarness(p_cpt=SALAD_CPT, d_cpt=SALAD_CPT,
                                   handback=0.6,
                                   awake="D", p_concurrency=2, d_bs=4,
                                   tp_prefill_max_tokens=500,
                                   flip_min_work_tokens=1,
                                   idle_layout="D") as h:
            h.d.refuse_real_for["pt"] = 9   # refused twice, but 60 % came back
            status, text = await asyncio.wait_for(h.post("pt", chars=BODY_CHARS), 30.0)
            assert status == 503, (status, text[:400])
            assert "W35 Weg2XReQueueLoop" in text, text[:400]
            assert h.front.counters["W35_Weg2XReQueueLoop"] == 1
            assert h.front.counters.get("W53_Weg2StoreHandbackFailed", 0) == 0, (
                "the store handed back 60 %; claiming NOTHING came back is a "
                "false sentence in the log, not a harmless mis-name")

    asyncio.run(body())


def test_the_two_classes_differ_only_in_chars_per_token():
    """FAULT 1's discriminator, as arithmetic rather than prose.

    3.0 is a constant in the front, not a property of the store, and it is the
    only thing that separated the two classes on sb5h.
    """
    text = "MARKx " + "x" * BODY_CHARS
    est = int(len(text) / CHARS_PER_TOKEN) + 1
    salad = int(len(text) / SALAD_CPT)
    natural = int(len(text) / NATURAL_CPT)
    assert salad > est > natural, (salad, est, natural)
    assert SALAD_CPT < CHARS_PER_TOKEN < NATURAL_CPT


# ------------------------------------------------- arm C: MUTANT, over-refusing
def test_mutant_arm_c_a_genuine_partial_handback_still_gets_its_re_offer():
    """THE danger direction.  D priced a SMALLER extent than P measured from
    the first offer on, so the store DID hand pages back and the re-offer
    genuinely serves the client (f2a/f2b/t9c).  W53 must stay silent.
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
    not say P ran.  ``leg1_prompt_tokens == 0`` is UNKNOWN.

    D is made to refuse TWICE here, which is the shape round 1 never
    exercised: the request reaches the n>1 terminal with no measurement of its
    own, so it must fall through to the plain W35 503 and W53 must NOT claim
    "the store handed nothing back" about a prefill that never happened.
    """

    async def body():
        async with MeasuredHarness(p_cpt=SALAD_CPT, d_cpt=SALAD_CPT,
                                   awake="P", p_concurrency=2, d_bs=4,
                                   tp_prefill_max_tokens=5000,
                                   carrier_max_tokens=1000,
                                   flip_min_work_tokens=1,
                                   idle_layout="D") as h:
            h.d.refuse_real_for["ce"] = 9
            status, text = await asyncio.wait_for(h.post("ce", chars=BODY_CHARS), 30.0)
            assert status == 503, (status, text[:400])
            assert "W35 Weg2XReQueueLoop" in text, text[:400]
            assert h.front.counters["route_carrier_exceeds"] == 1
            assert h.front.counters.get("W53_Weg2StoreHandbackFailed", 0) == 0
            assert h.front.counters["W35_Weg2XReQueueLoop"] == 1
            assert h.p.gen_marks.count("ce") == 0, (
                "CARRIER-EXCEEDS runs no leg 1 at all", h.p.gen_marks)

    asyncio.run(body())


def test_mutant_arm_d_served_on_the_re_offer_is_still_served():
    """The same arrival, refused ONCE: the re-offer must survive.  This is the
    arm round 1 shipped, kept so the n>1 change cannot break the n=1 path."""

    async def body():
        async with MeasuredHarness(p_cpt=SALAD_CPT, d_cpt=SALAD_CPT,
                                   awake="P", p_concurrency=2, d_bs=4,
                                   tp_prefill_max_tokens=5000,
                                   carrier_max_tokens=1000,
                                   flip_min_work_tokens=1,
                                   idle_layout="D") as h:
            h.d.refuse_real_for["cf"] = 1
            status, text = await asyncio.wait_for(h.post("cf", chars=BODY_CHARS), 30.0)
            assert status == 200, (status, text[:400])
            assert h.front.counters.get("W53_Weg2StoreHandbackFailed", 0) == 0

    asyncio.run(body())


# ------------------------------------------------- MUTANT: UNKNOWN stays UNKNOWN
def test_mutant_an_unparsable_refusal_body_still_re_offers():
    """The slice-A wording, kept alive on purpose: a body whose extent cannot
    be read is UNKNOWN, and UNKNOWN re-offers.  #1290 round 2's law is
    untouched -- only the comparand and the placement moved.
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


def test_mutant_an_unparsable_body_at_the_second_refusal_is_a_plain_w35():
    """UNKNOWN at the terminal too: no extent means no W53, only W35's 503.
    The 413 must never be reachable on a guess."""

    async def body():
        async with Harness(awake="D", p_concurrency=2, d_bs=4,
                           tp_prefill_max_tokens=10,
                           flip_min_work_tokens=1, idle_layout="D") as h:
            h.d.refuse_x_for["uu"] = 9
            status, text = await asyncio.wait_for(h.post("uu", chars=200), 30.0)
            assert status == 503, (status, text[:400])
            assert "W35 Weg2XReQueueLoop" in text, text[:400]
            assert h.front.counters.get("W53_Weg2StoreHandbackFailed", 0) == 0
            assert h.front.counters["W35_Weg2XReQueueLoop"] == 1

    asyncio.run(body())


# ------------------------------------------------------------ source mutants
def test_mutant_the_estimate_is_not_the_comparand_anywhere_in_the_branch():
    src = inspect.getsource(Front._requeue_after_x_refusal)
    assert "d_extent >= measured_whole" in src
    assert "d_extent >= pending.est_uncached" not in src, (
        "#1296: the char estimate may never gate the terminal verdict again")


def test_mutant_w53_can_never_fire_on_the_first_refusal():
    """THE ROUND-2 PIN, in control-flow terms rather than prose.

    Restoring #1291's placement deletes served requests silently -- arm E is
    the behavioural proof, this is the structural one, so a refactor cannot
    reintroduce it by moving the block rather than changing the predicate.
    W52 must stay OUTSIDE the guard: `carrier_est > carrier_max` is
    structural, and that is the distinction round 1 lost.
    """
    import ast
    tree = ast.parse(inspect.getsource(Front._requeue_after_x_refusal).lstrip())
    fn = tree.body[0]

    def incs(node):
        return [ast.literal_eval(n.target.slice)
                for n in ast.walk(node)
                if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Subscript)]

    guards = [s for s in fn.body
              if isinstance(s, ast.If) and isinstance(s.test, ast.Compare)
              and isinstance(s.test.left, ast.Name) and s.test.left.id == "n"
              and isinstance(s.test.ops[0], ast.Gt)]
    assert len(guards) == 1, "the `if n > 1:` guard is the only terminal gate"
    guard = guards[0]
    outside = [c for s in fn.body if s is not guard for c in incs(s)]

    assert "W53_Weg2StoreHandbackFailed" in incs(guard)
    assert "W53_Weg2StoreHandbackFailed" not in outside, (
        "#1296 round 2: W53 fired at requeue_n=1 and deleted weg2-28-259")
    assert "W35_Weg2XReQueueLoop" in incs(guard)
    assert "W52_Weg2NoServiceableRoute" in outside, (
        "W52 is STRUCTURAL and stays terminal on the first refusal")
    assert "W52_Weg2NoServiceableRoute" not in incs(guard)


def test_mutant_w53_is_a_subset_of_w35_its_population():
    """DENOMINATOR LAW: W53 counts a subset of the second refusals, so the
    census must always satisfy W53 <= W35 and the line must say so."""
    src = inspect.getsource(Front._requeue_after_x_refusal)
    i, j = src.find('W35_Weg2XReQueueLoop"] += 1'), src.find('W53_Weg2StoreHandbackFailed"] += 1')
    assert -1 < i < j, "W35 must be counted before the W53 subset branches off"
    assert "SUBSET of W35_Weg2XReQueueLoop" in src, (
        "the emitted line must name W53's population")


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
