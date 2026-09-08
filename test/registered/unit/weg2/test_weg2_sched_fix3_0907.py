# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Slice A, FIX 3: the four review blockers, the seats-vs-pool gate, the W27 root.

WHAT EACH GROUP OF TESTS IS FOR, and which of them are RED at the parent
``dc69d0e2cf`` versus which exist to kill a MUTANT:

* ``b1_*`` -- law 4's D-side ENFORCEMENT POINT.  Every X test in the slice
  called the predicate directly; nothing asserted that
  ``_get_new_batch_prefill_raw`` REACHES it, so ``if False and
  self._weg2_x_refuses(...)`` survived the whole suite.  A source
  CONTAINMENT check does not kill that mutant (the call is still in the
  text), so the assertion is over the AST: the call must BE the ``if``'s
  test, not a term in it.  Not red at the parent -- red under the mutant,
  which is the defect these close.
* ``b2_*`` -- the L-lines.  A1-1's fire line and the L1..L14 format strings
  had no caplog assertion anywhere, so deleting the WEG2-FAIRNESS warning
  left the suite byte-identical.  Also mutant-killers rather than parent-red.
* ``b3_*`` -- LAW 2 at D.  RED at the parent: ``_apply_uniform_head_order``
  re-sorts the head by group prefix length on every prefill pass, so arrival
  order 1,2,3 was admitted as 2,3,1.
* ``b4_*`` -- W31 as a TOTAL group verdict.  RED at the parent: ``min(local,
  group)`` picks the rank-local value whenever ``local <= group``, and two
  ranks whose host halves differ priced 10,000 and 5,000 for the same
  request.
* ``a_*`` -- the aggregate seats-vs-host-pool gate and its named L-line.
* ``bb_*`` -- the W27 root: on a PP group with no #631 row carrier the store
  READ is refused, because a completion landing on one rank alone lengthens
  that rank's ``prefix_indices`` and nothing on the PP axis reduces it.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no model, no GPU.  The gloo test at
the end runs three REAL processes, because the property it asserts ("the
three ranks build the same geometry") is not observable in one.
"""

import ast
import asyncio
import builtins
import inspect
import json
import logging
import os
import sys
import tempfile
import time
from types import MethodType, SimpleNamespace

import pytest

from sglang.srt.managers import tp_head_congruence as thc
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2.front import Front, Pending, Seat


# --------------------------------------------------------------- helpers
def _bare_pending(rid: str) -> Pending:
    p = Pending.__new__(Pending)
    p.rid = rid
    p.payload = {}
    p.text = ""
    p.fut = asyncio.get_event_loop().create_future()
    p.t_arrive = 0.0
    p.est_prompt = 0
    p.seat = None
    p.posted_evt = None
    p.posted = False
    return p


async def _until(pred, timeout: float = 5.0, tick: float = 0.02) -> bool:
    loop = asyncio.get_event_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if pred():
            return True
        await asyncio.sleep(tick)
    return pred()


def _fn_ast(func):
    src = inspect.getsource(func)
    return ast.parse(inspect.cleandoc(src.replace("\n    ", "\n")) if False else _dedent(src))


def _dedent(src: str) -> str:
    import textwrap

    return textwrap.dedent(src)


def _gate_stub(x: int, host_carry: int = 10 ** 9, tp_size: int = 3):
    """A Scheduler stand-in for the X gate with the REAL bodies bound.

    Local rather than imported from the slice-A file: a test module is not
    importable as a package here (`test.registered` is not a package), and a
    shared fixture between two test files is one more thing to keep in step.
    """
    stub = SimpleNamespace(
        server_args=SimpleNamespace(tp_prefill_max_tokens=x),
        ps=SimpleNamespace(tp_size=tp_size),
        tree_cache=SimpleNamespace(
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(size=host_carry)
            )
        ),
    )
    stub.weg2_uncached_extent = lambda req, head=None: Scheduler.weg2_uncached_extent(
        stub, req, head
    )
    stub._weg2_host_carry_tokens = lambda: Scheduler._weg2_host_carry_tokens(stub)
    return stub


def _stub_req(prompt_tokens: int, prefix: int = 0, host_hit: int = 0, rid: str = "r"):
    return SimpleNamespace(
        rid=rid,
        full_untruncated_fill_ids=list(range(prompt_tokens)),
        prefix_indices=list(range(prefix)),
        host_hit_length=host_hit,
    )


def _calls_named(tree, name):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == name:
                out.append(node)
    return out


# ================================================= b1: the enforcement point
def test_b1a_the_x_gate_is_the_if_itself_at_the_prefill_call_site():
    """BLOCKING 1.  Law 4 says X is ENFORCED on D at the prefill batch.  The
    suite proved the predicate and never the placement, and the measured
    mutant ``if False and self._weg2_x_refuses(req, _head_inputs):`` left the
    whole slice suite byte-identical -- 136 passed / 1 pre-existing failed,
    twice.

    A ``"_weg2_x_refuses" in getsource`` assertion cannot kill that mutant:
    the call is still in the text.  The property is structural -- the call
    must be the ``if``'s ENTIRE test, its body must skip the request, and the
    refused list must be answered after the loop -- so it is asserted over
    the AST."""
    tree = _fn_ast(Scheduler._get_new_batch_prefill_raw)
    gates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Attribute)
        and node.test.func.attr == "_weg2_x_refuses"
    ]
    assert len(gates) == 1, (
        "law 4's D-side gate must be exactly one `if self._weg2_x_refuses(...)` "
        "whose test is the CALL ITSELF -- a call buried in a BoolOp (`if False "
        "and ...`) or behind another condition is the mutant the suite missed"
    )
    body = ast.dump(ast.Module(body=gates[0].body, type_ignores=[]))
    assert "Continue" in body, "the refused request must be skipped this pass"
    assert "_x_refused" in body, "and collected for the named answer"


def test_b1b_the_refusals_are_answered_after_the_loop():
    """BLOCKING 1, the other half of the call site: the collected refusals
    must reach ``_weg2_answer_x_refusals``.  Deleting that call turns W31
    into a silent skip -- a livelock wearing the costume of a policy."""
    tree = _fn_ast(Scheduler._get_new_batch_prefill_raw)
    answers = _calls_named(tree, "_weg2_answer_x_refusals")
    assert len(answers) == 1, "exactly one answer site"
    args = [ast.dump(a) for a in answers[0].args]
    assert any("_x_refused" in a for a in args), "it must answer THIS pass's list"
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "_x_refused"
    ]
    assert guards, "guarded by `if _x_refused:` so an empty pass sends nothing"


def _consults_state(test) -> bool:
    """Can this ``if`` test possibly depend on anything the loop varies?

    True only when the test reaches for STATE -- a call or an attribute.  A
    test built solely from bare names, constants and operators between them
    (``True``, ``req is not None``, ``1 == 1``) consults nothing the scheduler
    can change, so an ``if`` of that shape whose body jumps is an
    unconditional skip wearing a condition.
    """
    return any(isinstance(n, (ast.Call, ast.Attribute)) for n in ast.walk(test))


def _always_jumps(stmt) -> bool:
    """Does control ALWAYS leave the enclosing loop body at this statement?

    A jump statement itself, or an ``if`` that cannot fall through: both of
    its branches jump, or its body jumps and its test consults no state.

    ROUND 5 -- WHY THE ROUND-4 VERSION WAS ONE LITERAL FROM VOID.  It required
    ``isinstance(test, ast.Constant)``, so it saw ``if True: ... continue``
    and missed ``if req is not None: ... continue`` -- a guard that is always
    true IN FACT rather than literally.  The measured mutant, inserted
    immediately above the gate inside the same loop body, left the suite at
    the baseline 1 failed / 90 passed.  :func:`_consults_state` closes that
    literal, but ONLY that literal: ``if req.rid is not None`` would walk
    around it too.  NO AST PREDICATE CAN SEPARATE "the statement is present"
    FROM "the statement runs" -- that separation needs execution, and it is
    ``test_b1e``/``test_b1f`` below, not this function, that provides it.
    """
    if isinstance(stmt, (ast.Continue, ast.Break, ast.Return, ast.Raise)):
        return True
    if isinstance(stmt, ast.If):
        body = any(_always_jumps(x) for x in stmt.body)
        orelse = any(_always_jumps(x) for x in stmt.orelse)
        return (body and not _consults_state(stmt.test)) or (body and orelse)
    return False


def test_b1d_no_stateless_guard_skips_the_request_before_the_x_gate():
    """BLOCKING 1, the CHEAP STRUCTURAL COMPANION -- explicitly NOT the reach
    proof (that is ``test_b1e``/``test_b1f``, which execute the loop).

    What it still buys, in one call and without a stub: the gate must be a
    DIRECT statement of the ``for req in self.waiting_queue`` body, so no
    outer condition wraps it; and no statement above it in that body jumps
    unconditionally in the shapes an AST can actually recognise (a bare
    ``continue``/``break``/``return``, an if/else where both arms jump, or a
    jump under a test that consults no state at all).

    Its LIMIT is the round-5 finding and is stated here so no future reader
    mistakes it for reach again: a guard whose test consults state and is
    nonetheless always true passes this check untouched."""
    tree = _fn_ast(Scheduler._get_new_batch_prefill_raw)
    gates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Attribute)
        and node.test.func.attr == "_weg2_x_refuses"
    ]
    assert len(gates) == 1
    gate = gates[0]
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and gate in node.body
    ]
    assert len(loops) == 1, (
        "law 4's gate must be a DIRECT statement of the waiting-queue loop -- "
        "nested one level deeper it is reachable only under whatever condition "
        "wraps it, and the AST placement check at the gate cannot see that"
    )
    before = loops[0].body[: loops[0].body.index(gate)]
    dominating = [st for st in before if _always_jumps(st)]
    assert not dominating, (
        "an unconditional continue/break/return above the gate makes law 4 "
        "unreachable with the suite green: "
        + ", ".join(f"{type(st).__name__}@line {st.lineno}" for st in dominating)
    )


# ------------- law 4's reach, PROVEN BY EXECUTION rather than by AST -------
class _No:
    """Answers every question with "no", and every call with another ``_No``.

    The stand-in the reach slice runs on.  Every guard above law 4's gate asks
    the scheduler for a reason to SKIP this request; this object gives all of
    them the answer that does NOT skip, so the only thing that can still keep
    control from arriving at the gate is a statement that skips no matter what
    it is told -- which is exactly the property under test.  Overrides passed
    to the constructor win, and are how the slice gets its waiting queue and
    its spy.
    """

    def __init__(self, **over):
        self.__dict__["_over"] = over

    def __getattr__(self, name):
        return self.__dict__["_over"].get(name, _NO)

    def __call__(self, *a, **k):
        return _NO

    def __bool__(self):
        return False

    def __len__(self):
        return 0

    def __iter__(self):
        return iter(())

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True

    def __lt__(self, other):
        return False

    __le__ = __gt__ = __ge__ = __lt__

    def __hash__(self):
        return 0


_NO = _No()


def _law4_reach_slice():
    """Compile the REAL waiting-queue loop, truncated at law 4's gate, into a
    function that can be RUN -- the separation an AST proxy cannot make.

    Built from ``inspect.getsource`` at call time, so it is the shipping
    source that executes, not a transcription of it.  Three deliberate
    surgeries, each of which only ever makes the test STRICTER:

    * the body is cut after the gate, because everything below it is the PP
      admission machinery and none of it bears on whether the gate was
      reached;
    * one synthetic statement is appended after the gate,
      ``_reach_past_gate.append(req)``, so "the refusal actually skipped the
      request" is observable rather than inferred from the AST;
    * every FREE NAME of the slice becomes a parameter, computed from the
      source instead of listed.  A future edit that reaches for one more local
      therefore fails loudly with an unexpected keyword rather than silently
      binding a module global.
    """
    tree = _fn_ast(Scheduler._get_new_batch_prefill_raw)
    gates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Attribute)
        and node.test.func.attr == "_weg2_x_refuses"
    ]
    assert len(gates) == 1, "law 4's gate must be exactly one `if` (see test_b1a)"
    gate = gates[0]
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For) and gate in n.body]
    assert len(loops) == 1, "and a direct statement of the waiting-queue loop"
    loop = loops[0]
    idx = loop.body.index(gate)
    sentinel = ast.parse("_reach_past_gate.append(req)").body[0]
    trunc = ast.For(
        target=loop.target,
        iter=loop.iter,
        body=loop.body[: idx + 1] + [sentinel],
        orelse=[],
        type_comment=None,
    )
    bound = {
        n.id
        for n in ast.walk(trunc)
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
    }
    free = sorted(
        {
            n.id
            for n in ast.walk(trunc)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        - bound
        - set(dir(builtins))
    )
    fn = ast.FunctionDef(
        name="_law4_reach",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=n) for n in free],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[trunc],
        decorator_list=[],
        returns=None,
        type_params=[],
    )
    mod = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {}
    exec(compile(mod, inspect.getsourcefile(Scheduler), "exec"), ns)
    return ns["_law4_reach"], free


def _run_law4_reach(refuses: bool, drive_pp: bool = False):
    """Run that slice for ONE request and report what the loop actually did.

    ``drive_pp`` (ROUND 6, item 4) turns the permissive stub into a Weg-2
    GROUP P: HiCache storage on and ``pp_size > 1``, so the conditional PP
    admission block at ``scheduler.py:11419`` -- which the permissive stub
    steps OVER, because ``_No`` answers ``enable_hicache_storage`` with
    "no" -- is EXECUTED on the way to the gate.  That block was FIX 5's own
    named coverage residue.
    """
    fn, free = _law4_reach_slice()
    skips, refused, past, seen = [], [], [], []
    req = SimpleNamespace(
        rid="weg2-1-7",
        init_next_round_input=lambda *a, **k: None,
    )

    def _spy(r, head=None):
        seen.append(r)
        return refuses

    over = dict(waiting_queue=[req], _weg2_x_refuses=_spy)
    kw = {name: _NO for name in free}
    if drive_pp:
        # Group P's OWN shape, and every override is one of its facts:
        # PP=3 with no #631 row carrier (so `_pp0_may_withhold` is False and
        # the #973 disarmed-take arm runs), storage on, this rank's prefetch
        # complete (the only answer that does not `continue`), and the three
        # running counters those arms increment -- `_No` would hand them a
        # sentinel and `sentinel + 1` is a TypeError, not a skip.
        over.update(
            enable_hicache_storage=True,
            ps=SimpleNamespace(pp_size=3, pp_rank=0),
            _prefetch_done_for=lambda *a, **k: True,
            _admit_under_group_completion=lambda *a, **k: True,
            tree_cache=SimpleNamespace(pop_prefetch_loaded_tokens=lambda rid: 0),
            _973_pp0_take_n=0,
            _973_pp0_take_pending_n=0,
            _969z_followed=0,
        )
        kw["pp_row_carrier_present"] = lambda *a, **k: False
        kw["observe_store_witness"] = lambda *a, **k: None
        kw["prefetch_verdicts"] = {}
        kw["logger"] = logging.getLogger("weg2.reach.pp")
    kw["self"] = _No(**over)
    kw["_note_skip"] = lambda kind, rid: skips.append((kind, rid))
    kw["_x_refused"] = refused
    kw["_reach_past_gate"] = past
    fn(**kw)
    return SimpleNamespace(req=req, entered=seen, skips=skips,
                           refused=refused, past=past, sched=kw["self"])


def test_b1e_the_x_gate_is_ENTERED_when_the_real_loop_runs():
    """BLOCKING 1 (round 3), CLOSED BY EXECUTION.

    The round-4 answer was an AST reach proxy, and it was one literal from
    void: ``_always_jumps`` only recognised a truthy ``ast.Constant`` test, so
    the measured round-5 mutant -- ``if req is not None: _note_skip("mutant",
    req.rid); continue``, inserted immediately above the gate inside the same
    loop body -- left the suite at its baseline 1 failed / 90 passed while
    law 4's D-side enforcement was dead.  ``req`` is the loop variable and is
    never None.

    THE ONLY THING THAT SEPARATES "the statement is present" FROM "the
    statement runs" IS RUNNING IT.  This test executes the real loop source
    with a spy in place of ``_weg2_x_refuses`` and asserts the spy was
    ENTERED -- a counter, not an AST shape.  Any statement above the gate that
    skips the request, under any test and with any literal, makes this red."""
    got = _run_law4_reach(refuses=False)
    assert len(got.entered) == 1 and got.entered[0] is got.req, (
        "law 4's D-side gate was never entered for a request that walked the "
        "whole waiting-queue loop body -- something above it skipped the "
        "request, so the X bound is not enforced on D at all"
    )
    assert got.past == [got.req], (
        "and a request the gate does NOT refuse must fall through to the rest "
        "of the pass rather than be skipped"
    )
    assert got.skips == [] and got.refused == []


def test_b1f_a_refused_request_is_skipped_and_collected_by_the_real_loop():
    """The behavioural half of the same execution: when the gate refuses, the
    request must leave THIS pass and be collected for the named answer.

    ``test_b1c`` proves what that answer is on the wire (an ``AbortReq``
    naming W31 reaching ``send_to_tokenizer``, which is what lets C12 re-route
    it through P exactly once).  This proves the loop actually hands it over
    -- the ``continue`` is observed by a statement that did NOT run, not read
    off the AST."""
    got = _run_law4_reach(refuses=True)
    assert len(got.entered) == 1
    assert got.refused == [got.req], "collected for _weg2_answer_x_refusals"
    assert got.skips == [("weg2_x_refused", "weg2-1-7")], "and named in the trace"
    assert got.past == [], (
        "a refused request must not continue into the rest of the pass -- the "
        "gate's `continue` is what keeps it out of this batch"
    )


def test_b1g_the_gate_is_reached_THROUGH_the_conditional_pp_block():
    """ROUND 6, item 4: FIX 5's own named coverage residue, closed.

    FIX 5 recorded it rather than leaving it to be discovered: ``_No`` answers
    every question with "no", so ``if self.enable_hicache_storage and
    _pp_group:`` at ``scheduler.py:11419`` is FALSE in ``b1e``/``b1f`` and the
    whole 150-line PP admission block is stepped over.  A mutant planted
    INSIDE it (a ``continue`` on the #973 disarmed-take arm, say) was reached
    by neither the execution slice nor the AST companion.

    This arm drives that block with group P's own configuration and asserts
    the SAME property: the gate is still entered, the request still falls
    through.  Two honest limits, stated so nobody reads more into it:

    * the exposure it closes is P-SIDE ONLY.  Group D is TP-only
      (``--tp-size 3 --pp-size 1``), so on the group law 4 is actually
      enforced on, ``_pp_group`` is False and this block cannot execute at
      all -- a skip inside it could never have disarmed X on D.  What it CAN
      do is drop a request out of a P prefill pass, which is why it is worth
      a test rather than a shrug.
    * it drives ONE path through the block (carrier-less, prefetch complete),
      which is Weg 2's, not all of them.  The block's other exits are
      `continue`s by design (a pending prefetch), and a test that made them
      fire would be asserting the stub.
    """
    got = _run_law4_reach(refuses=False, drive_pp=True)
    # FIRST: prove the block RAN, or this test is the permissive arm again
    # under a longer name.  Both counters are incremented inside it and
    # nowhere else in the slice.
    assert got.sched._973_pp0_take_n == 1 and got.sched._969z_followed == 1, (
        "the PP admission block did not execute -- the drive is not driving, "
        "and the arm proves nothing the permissive stub did not already"
    )
    assert len(got.entered) == 1 and got.entered[0] is got.req, (
        "with HiCache storage on and pp_size > 1 the loop runs the PP "
        "admission block before law 4's gate -- and something in there kept "
        "control from arriving at the gate"
    )
    assert got.past == [got.req]
    assert got.skips == [] and got.refused == []
    # and the refusing half still refuses through the same path
    got = _run_law4_reach(refuses=True, drive_pp=True)
    assert got.refused == [got.req] and got.past == []


def test_b1c_the_answer_removes_the_request_and_names_w31_on_the_wire():
    """BLOCKING 1, behavioural.  The named 503 is what lets C12 re-route the
    request through P exactly once; a refusal that only skipped would leave
    it in the queue for ever."""
    sent = []
    req = SimpleNamespace(
        rid="weg2-1-7",
        full_untruncated_fill_ids=list(range(30000)),
        prefix_indices=[],
        host_hit_length=0,
        time_stats=SimpleNamespace(trace_ctx=SimpleNamespace(abort=lambda abort_info: None)),
    )
    other = SimpleNamespace(rid="weg2-1-8")
    stub = SimpleNamespace(
        server_args=SimpleNamespace(tp_prefill_max_tokens=8192),
        waiting_queue=[req, other],
        enable_hicache_storage=False,
        enable_hierarchical_cache=False,
        ipc_channels=SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(
                send_output=lambda abort, r: sent.append((abort, r))
            )
        ),
    )
    stub.weg2_uncached_extent = lambda r, h=None: Scheduler.weg2_uncached_extent(stub, r, h)
    Scheduler._weg2_answer_x_refusals(stub, [req], None)
    assert [r.rid for r in stub.waiting_queue] == ["weg2-1-8"], "removed from THIS queue"
    assert len(sent) == 1
    abort, r = sent[0]
    assert r is req
    assert abort.finished_reason["status_code"] == 503
    assert "W31 Weg2TpPrefillExceeded" in abort.finished_reason["message"]


# ============================================================ b2: the L-lines
def test_b2a_the_fairness_switch_prints_its_line_on_every_fire(caplog):
    """BLOCKING 2 / Amendment A1-1: "every fire prints the switch name, the
    oldest wait and the queue it pre-empted".  No caplog assertion existed
    anywhere in the slice, so a silent fire was byte-identical to a loud
    one."""
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=6,
              carrier_max_tokens=27466)
    with caplog.at_level(logging.WARNING, logger=front_mod.logger.name):
        assert f._fairness_switch(oldest_arrival=0.0, queue_name="batch") is True
    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "WEG2-FAIRNESS" in line
    assert "switch=--fairness-w-s" in line, "the SWITCH is named"
    assert "value=45" in line, "with its value"
    assert "queue=batch" in line, "and the queue it pre-empted"
    assert "waited" in line, "and the oldest wait"
    caplog.clear()
    off = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 0.0, d_bs=6,
                carrier_max_tokens=27466)
    with caplog.at_level(logging.WARNING, logger=front_mod.logger.name):
        assert off._fairness_switch(oldest_arrival=0.0, queue_name="batch") is False
    assert "WEG2-FAIRNESS" not in caplog.text, "0 disables it, silently"


def test_b2b_the_x_gate_prints_l9_on_both_verdicts(caplog):
    """BLOCKING 2: L9 ``WEG2 X-GATE ... replicated_term=`` is what boot
    acceptance clause 2 reads back, on BOTH outcomes."""
    import sglang.srt.managers.scheduler as sched_mod

    head = _head("r", 0)
    stub = _gate_stub(x=10000)
    with caplog.at_level(logging.INFO, logger=sched_mod.logger.name):
        assert Scheduler._weg2_x_refuses(stub, _stub_req(9999), head) is False
        assert Scheduler._weg2_x_refuses(stub, _stub_req(10001), head) is True
    text = caplog.text
    assert "WEG2 X-GATE" in text
    assert "verdict=admit" in text and "verdict=W31" in text
    assert "replicated_term=group" in text
    assert "uncached=9999" in text and "X=10000" in text


def test_b2c_every_l_line_the_spec_names_is_present_with_its_denominators():
    """BLOCKING 2, the rest of §4.  Each format string is asserted with the
    FIELDS the spec names, so a line cannot be renamed or stripped of its
    denominator without a red test.  Slice-C lines (L6/L7/L8) are named as
    NOT SHIPPED here rather than silently omitted."""
    front_src = inspect.getsource(front_mod)
    sched_src = inspect.getsource(sys.modules["sglang.srt.managers.scheduler"])
    required = {
        "L1": (front_src, "WEG2 P-DRAIN", ["epoch=", "prefilled=", "arrived_during=",
                                           "p_concurrency=", "passes=", "queue_at_exit="]),
        "L2": (front_src, "WEG2 D-ADMIT rid=", ["seat=", "rank=", "oldest_wait_s=", "source="]),
        "L3": (front_src, "WEG2 D-REFILL", ["freed_by=", "seats_free=", "queued_d="]),
        "L4": (front_src, "WEG2 SHORT-BEHIND-P", ["epoch=", "n=", "oldest_wait_s="]),
        "L5": (front_src, "WEG2 LATE-BATCH", ["deferred_to_epoch="]),
        "L9": (sched_src, "WEG2 X-GATE", ["uncached=", "X=", "replicated_term=", "verdict="]),
        "L10": (front_src, "WEG2 X-ROUTE", ["est_uncached=", "X=", "ESTIMATE"]),
        # L11 gains `configured=` and `reason=` in FIX 6 (MF-1): a rest line
        # that names only the AWAKE layout cannot say whether the front is
        # resting where it was told to or merely where it happens to be.
        "L11": (front_src, "WEG2 IDLE-REST", ["layout=", "configured=", "reason=",
                                              "ready_for_d=", "d_outstanding=", "held_s="]),
        # L15 (MF-3): the interim cost of W38 on group P, per drain epoch.
        "L15": (front_src, "WEG2 P-PREFIX-REUSE",
                ["epoch=", "requests=", "prefix_tokens_available_in_store=",
                 "prefix_tokens_reused=", "forgone_tokens="]),
        "L12": (front_src, "WEG2 MIN-DWELL", ["src=", "dst=", "awake_ms=",
                                              "derived_from_flip_ms=", "overridden_by="]),
        "L13": (front_src, "WEG2 FLIP-ECONOMICS", ["queued_tokens=", "threshold=",
                                                   "fairness=", "verdict="]),
        "L14": (front_src, "WEG2 X-REQUEUE", ["n=", "verdict="]),
        "A1-1": (front_src, "WEG2-FAIRNESS", ["switch=--fairness-w-s", "value=", "queue="]),
        "A-seat": (front_src, "WEG2 D-SEAT-WAIT", ["need=", "available=", "limit=",
                                                   "held_passes="]),
    }
    missing = []
    for name, (src, marker, fields) in required.items():
        if marker not in src:
            missing.append(f"{name}: the whole line {marker!r} is gone")
            continue
        # EVERY occurrence, not the first: a line named in a docstring above
        # its own emitter would otherwise answer for it.
        windows = []
        start = 0
        while True:
            i = src.find(marker, start)
            if i < 0:
                break
            windows.append(src[i: i + 800])
            start = i + 1
        if not any(all(f in w for f in fields) for w in windows):
            for field in fields:
                if not any(field in w for w in windows):
                    missing.append(f"{name}: {marker!r} lost its {field!r} denominator")
    assert not missing, missing
    for slice_c in ("WEG2 SUSPEND", "WEG2 RESUME", "WEG2 SUSPEND-CENSUS"):
        assert slice_c not in front_src, (
            f"{slice_c} is slice C (law 3) and must not appear before the ring")


# ================================================== b3: LAW 2, order at D
def test_b3a_the_uniform_head_keeps_arrival_order():
    """BLOCKING 3, the measured break.  ``uniform_head_order`` sorted the
    head by ``(-group_match_len, rid)``; with arrival order 1,2,3 and
    matches 0/9000/100 it returned ``['weg2-1-2','weg2-1-3','weg2-1-1']`` --
    the OLDEST admitted LAST, i.e. law 2 broken at D by the module that was
    making D uniform."""
    canonical = thc.canonical_head_rids(["weg2-1-1", "weg2-1-2", "weg2-1-3"])
    payload = thc.build_head_order_payload(
        canonical, {"weg2-1-1": 0, "weg2-1-2": 9000, "weg2-1-3": 100}
    )
    arrival = {"weg2-1-1": 1, "weg2-1-2": 2, "weg2-1-3": 3}
    assert thc.uniform_head_order(canonical, payload, arrival_seqs=arrival) == [
        "weg2-1-1", "weg2-1-2", "weg2-1-3",
    ]
    # the ABSENT filter still comes from the MIN payload: a rid some rank does
    # not hold is dropped whatever its arrival rank (delay, never force).
    payload2 = thc.build_head_order_payload(canonical, {"weg2-1-2": 9000, "weg2-1-3": 100})
    assert thc.uniform_head_order(canonical, payload2, arrival_seqs=arrival) == [
        "weg2-1-2", "weg2-1-3",
    ]
    # and the key is TOTAL: a rid with no arrival seq sorts behind, by rid.
    assert thc.uniform_head_order(
        canonical, payload, arrival_seqs={"weg2-1-3": 1}
    ) == ["weg2-1-3", "weg2-1-1", "weg2-1-2"]


def test_b3b_the_decision_carries_the_arrival_key_through():
    """The order arm is reached through ``head_decision``; a key that stops
    at ``uniform_head_order``'s signature changes nothing in production."""
    canonical = thc.canonical_head_rids(["a", "b"])
    payload = thc.build_head_order_payload(canonical, {"a": 0, "b": 9000})
    order, source = thc.head_decision(
        canonical, payload, ["a", "b"], {}, digest_agreed=True,
        enforcer_enabled=True, arrival_seqs={"a": 1, "b": 2},
    )
    assert order == ["a", "b"] and source == thc.SOURCE_GROUP
    order, _ = thc.head_decision(
        canonical, payload, ["a", "b"], {}, digest_agreed=True, enforcer_enabled=True
    )
    assert order == ["b", "a"], "without the key the cache heuristic still rules"


def test_b3c_the_scheduler_supplies_the_arrival_key_under_fcfs_and_under_weg2():
    """And the scheduler must actually PASS it.  Two independent triggers:
    a cache-agnostic policy (where ``_sort_by_longest_prefix`` never ran, so
    there is nothing for this arm to replace) and the Weg-2 marker
    ``--tp-prefill-max-tokens`` (where law 2 binds whatever the policy)."""
    from sglang.srt.managers.schedule_policy import CacheAgnosticPolicy, CacheAwarePolicy

    reqs = [SimpleNamespace(rid="r1", kv_arrival_seq=7),
            SimpleNamespace(rid="r2", kv_arrival_seq=8)]
    fcfs = SimpleNamespace(
        server_args=SimpleNamespace(tp_prefill_max_tokens=0),
        policy=SimpleNamespace(policy=CacheAgnosticPolicy.FCFS),
        waiting_queue=reqs,
    )
    assert Scheduler._head_order_arrival_seqs(fcfs) == {"r1": 7, "r2": 8}
    weg2 = SimpleNamespace(
        server_args=SimpleNamespace(tp_prefill_max_tokens=8192),
        policy=SimpleNamespace(policy=CacheAwarePolicy.LPM),
        waiting_queue=reqs,
    )
    assert Scheduler._head_order_arrival_seqs(weg2) == {"r1": 7, "r2": 8}, (
        "law 2 binds on the Weg-2 group whatever the policy")
    lpm = SimpleNamespace(
        server_args=SimpleNamespace(tp_prefill_max_tokens=0),
        policy=SimpleNamespace(policy=CacheAwarePolicy.LPM),
        waiting_queue=reqs,
    )
    assert Scheduler._head_order_arrival_seqs(lpm) is None, (
        "outside Weg 2 a cache-aware policy DID sort, and this arm replaces it")
    vacuous = SimpleNamespace(
        server_args=SimpleNamespace(tp_prefill_max_tokens=8192),
        policy=SimpleNamespace(policy=CacheAgnosticPolicy.FCFS),
        waiting_queue=[SimpleNamespace(rid="r1"), SimpleNamespace(rid="r2")],
    )
    assert Scheduler._head_order_arrival_seqs(vacuous) is None, (
        "an EMPTY mapping is not a key: it would order the head by rid "
        "string, a third order nobody asked for")
    # and the applier hands it to the decision
    tree = _fn_ast(Scheduler._apply_uniform_head_order)
    calls = _calls_named(tree, "head_decision")
    assert calls, "the applier calls head_decision"
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert "arrival_seqs" in kw, "with the arrival key, or law 2 is decorative"
    # THE VALUE, not merely the keyword: `arrival_seqs=None` is a keyword and
    # a mutant (measured: it survived the first census).  The argument must be
    # the derivation itself.
    value = kw["arrival_seqs"]
    assert (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "_head_order_arrival_seqs"
    ), ast.dump(value)


# ======================================= b4: W31 as a TOTAL group verdict
def _drift_req(host_hit: int, rid: str = "r", total: int = 30000, prefix: int = 1000):
    return SimpleNamespace(
        rid=rid,
        full_untruncated_fill_ids=list(range(total)),
        prefix_indices=list(range(prefix)),
        host_hit_length=host_hit,
    )


def _head(rid: str, group_match: int):
    canonical = thc.canonical_head_rids([rid])
    return thc.build_uniform_head_inputs(
        canonical, thc.build_head_order_payload(canonical, {rid: group_match}), None, True
    )


def test_b4a_the_extent_is_the_group_term_alone():
    """BLOCKING 4.  ``uncached + max(0, local - group)`` IS
    ``len(fill_ids) - min(local, group)``, and ``min`` picks the RANK-LOCAL
    value whenever ``local <= group`` -- the ordinary case, since the group
    value is a MIN.  Measured at the parent: the same rid, the same group
    value, host halves 19,000 and 29,000 -> 10,000 and 5,000, so with
    X=8,000 one rank raises W31 and the other admits, and the refusal
    deletes the request from that rank's queue alone (``send_to_tokenizer``
    is a ``SenderWrapper(None)`` off rank 0: silent and permanent)."""
    head = _head("r", 25000)
    stub = SimpleNamespace()
    a = Scheduler.weg2_uncached_extent(stub, _drift_req(19000), head)
    b = Scheduler.weg2_uncached_extent(stub, _drift_req(29000), head)
    assert a == b == 5000, (a, b)
    # no rank-local quantity is in the compared term: change the DEVICE half
    # too and the answer still does not move.
    c = Scheduler.weg2_uncached_extent(
        stub, _drift_req(0, total=30000, prefix=27000), head)
    assert c == 5000, c


def test_b4b_an_impossible_group_match_is_a_named_stop():
    """MUST NOT 7's detector.  A group match longer than the request's own
    replicated token ids can only mean the ranks reduced over different
    requests under one rid -- a batch-formation split.  The group stops by
    name rather than pricing through it."""
    head = _head("r", 40000)
    with pytest.raises(RuntimeError) as exc:
        Scheduler.weg2_uncached_extent(SimpleNamespace(), _drift_req(0), head)
    assert "W37 X-TERM SPLIT STOP" in str(exc.value)
    assert "rid=r" in str(exc.value)


def test_b4c_the_drift_reading_is_counted_and_decides_nothing(caplog):
    """The other direction -- this rank's own tree BELOW the group MIN it
    voted, which the HiCache controller threads can produce between the vote
    and the gate -- is a READING with its denominator, not a verdict and not
    a boot killer: the price does not read the local value at all."""
    import sglang.srt.managers.scheduler as sched_mod

    stub = SimpleNamespace()
    head = _head("r", 25000)
    with caplog.at_level(logging.INFO, logger=sched_mod.logger.name):
        priced = Scheduler.weg2_uncached_extent(stub, _drift_req(1000), head)
    assert priced == 5000, "the verdict is unaffected by the drift"
    assert "WEG2 X-TERM-DRIFT" in caplog.text
    assert "local=2000" in caplog.text and "group=25000" in caplog.text
    assert stub._weg2_x_term_drift == 1
    assert "priced=" in caplog.text, "the denominator rides on the line"


def test_b4d_the_gate_still_abstains_without_a_group_value():
    """Abstain-never-refuse survives the rewrite: below ``tp_size > 1`` with
    no group match published, no verdict is taken."""
    stub = _gate_stub(x=1000)
    assert Scheduler._weg2_x_refuses(stub, _drift_req(0), None) is False
    assert stub._weg2_x_abstained == 1


# ======================================= a: the aggregate seats-vs-pool gate
def _reading(available: int, occupied: int, limit: int, t: float):
    """A group-D host-pool reading in the shape `prefetch_residency` publishes."""
    return {"available": available, "occupied": occupied, "limit": limit,
            "size": limit, "threshold": 256, "pool_id": 1, "phase": "TP",
            "generation": 1, "t": t}


def _front_with_reading(reading, **kw):
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, **kw)

    async def _stub():
        return reading

    f._d_pool_reading = _stub
    return f


def test_a1_a_seat_is_taken_only_when_the_store_read_can_be_issued(caplog):
    """(A) The coupling the postmortem named unaddressed: six seats x ~9k
    tokens against a 27,466-token pool bound.  The head WAITS in arrival
    order -- it is not skipped over, and it takes no seat -- and it is served
    the moment a running request frees its tokens.  The wait prints
    ``WEG2 D-SEAT-WAIT`` with need/available/limit.

    ROUND 4, the coverage half.  The round-3 version of this test asserted
    "it is not skipped over" at a moment when the deque held exactly ONE
    entry, so no mutation that lets a younger request overtake the oldest
    could be seen: ``self._ready_for_d.rotate(-1)`` inside the blocked branch
    left the suite 18/18 green.  There are now TWO waiters at the block, the
    head does not fit and the younger one does, and the assertion is that the
    younger one is STILL not admitted."""

    async def body():
        t0 = time.time()
        f = _front_with_reading(_reading(27466, 0, 27466, t0), d_bs=6,
                                carrier_max_tokens=27466)
        ps = []
        for i, est in enumerate([8400, 8400, 8400, 8400, 100]):
            p = _bare_pending(f"r{i}")
            p.est_prompt = est
            ps.append(p)
        f._ready_for_d.extend(ps)
        f._sync_batch_gate()
        task = asyncio.create_task(f.d_admitter())
        for p in ps[:3]:
            assert await _until(lambda p=p: p.fut.done()), f"{p.rid} not admitted"
            p.posted_evt.set()
        with caplog.at_level(logging.INFO, logger=front_mod.logger.name):
            await asyncio.sleep(0.3)
            assert not ps[3].fut.done(), "3 x 8400 = 25200 of 27466; the 4th does not fit"
            # LAW 2 WITH A REAL ALTERNATIVE PRESENT: r4 needs 100 tokens and
            # 2266 are free, so only the arrival order keeps it waiting.
            assert not ps[4].fut.done(), (
                "law 2: a younger request that FITS may not overtake the blocked "
                "head -- the admitter peeks and continues, it does not rotate")
            assert list(f._ready_for_d) == [ps[3], ps[4]], (
                "law 2: the head of the arrival queue is not skipped over and the "
                "order behind it is untouched")
            assert f.seats_free() == 3, "and the COUNT alone would have admitted it"
        text = caplog.text
        assert "WEG2 D-SEAT-WAIT" in text, text[-400:]
        assert "rid=r3" in text and "need=8400" in text
        assert "available=2266" in text and "limit=27466" in text
        assert "held_passes=" in text, "the suppressed-pass denominator"
        assert "reading_age_s=" in text, "and the age of the numbers it decided on"
        ps[0].seat.release("leg2_finished")
        assert await _until(lambda: ps[3].fut.done()), (
            "a freed seat's tokens must come back with it, or the gate wedges")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(body())


def test_a2_the_gate_never_starves_and_is_a_flag():
    """It only ever DELAYS: a request with no seat in use is admitted whatever
    it costs (the oversized case is CARRIER-EXCEEDS's), and 0 disables it.

    The no-seat term is load-bearing and not a courtesy: with nothing running
    on D nothing will ever free a host row, so a gate that refused there would
    wedge the queue for good."""
    t0 = time.time()
    r = _reading(available=10, occupied=0, limit=27466, t=t0)
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=6,
              carrier_max_tokens=1000)
    assert f._d_token_budget_blocks("solo", 900000, r) is False, "no seat in use"
    Seat(f, "running", "batch", tokens=100)
    assert f._d_token_budget_blocks("second", 900000, r) is True, "now it bounds"
    off = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=6,
                carrier_max_tokens=27466, d_admit_max_tokens=0)
    Seat(off, "running", "batch", tokens=10 ** 9)
    assert off._d_token_budget_blocks("r", 10 ** 9, r) is False


def test_a3_the_gate_prices_the_pool_by_the_pool_not_by_its_own_tally():
    """ROUND 4, THE ROOT.  The round-1 gate compared ``_d_inflight_tokens``
    (a counter written only by ``Seat``, therefore 0 at every D-epoch start)
    against ``carrier_max_tokens``.  That indicator cannot see the STANDING
    RESIDENCY that actually refuses the store read, and boot weg2sc1 refutes
    it in the same second on both logs: the front's proxy said 27,466 rows
    were free while group D printed ``#915 PREFETCH REFUSED
    reason=vote_negative need=8629 available=5418 occupied=25100
    limit=27466``.

    The scenario below is that boot's shape: 25,100 rows of residency the
    FRONT DID NOT CREATE, one small request running, and a 8,629-token head.
    The round-1 arithmetic (100 + 8629 <= 27466) admits it and D refuses it;
    the reading (8629 > 5418) holds it."""
    t0 = time.time()
    r = _reading(available=5418, occupied=25100, limit=27466, t=t0)
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=6,
              carrier_max_tokens=27466)
    Seat(f, "small", "batch", tokens=100)
    assert f._d_inflight_tokens() == 100, (
        "the front's own tally is 100 -- the number round 1 decided on")
    assert f._d_token_budget_blocks("weg2-0-4", 8629, r) is True, (
        "the ALLOC term is D's available_size(), not budget-minus-my-tally")


def test_a4_the_rate_term_is_d_s_own_brake():
    """The second of D's two terms: ``prefetch_rate_limited`` refuses once the
    REGISTERED prefetches hold the capacity, even with rows still free."""
    t0 = time.time()
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=6,
              carrier_max_tokens=27466)
    Seat(f, "small", "batch", tokens=100)
    roomy = _reading(available=100000, occupied=27466, limit=27466, t=t0)
    assert f._d_token_budget_blocks("r", 10, roomy) is True, "occupied >= limit"
    assert f._d_token_budget_blocks("r", 10, _reading(100000, 0, 27466, t0)) is False


def test_a5_grants_made_after_the_reading_are_charged_on_top_of_it():
    """The window a reading cannot contain: a seat granted AFTER D sampled is
    a store read D has not yet registered.  It is charged; a seat granted
    before the reading is already in D's own numbers and is not charged
    twice."""
    t0 = time.time()
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=6,
              carrier_max_tokens=27466)
    Seat(f, "before", "batch", tokens=9000)  # t_taken < t_reading
    time.sleep(0.002)
    t_read = time.time()
    time.sleep(0.002)
    r = _reading(available=10000, occupied=0, limit=27466, t=t_read)
    assert f._d_charged_since(t_read) == 0, "already inside D's reading"
    assert f._d_token_budget_blocks("r", 9000, r) is False
    later = Seat(f, "after", "batch", tokens=9000)
    assert later.t_taken >= t_read
    assert f._d_charged_since(t_read) == 9000
    assert f._d_token_budget_blocks("r", 9000, r) is True, (
        "9000 of the 10000 rows are already promised to a grant D has not seen")


def test_a6_no_reading_means_the_gate_is_off_and_says_so(caplog):
    """A NAMED refusal, never a silent fallback.  With no reading the front
    must NOT substitute its own admission tally -- that is the exact quantity
    round 3 found wrong -- so the gate goes off and prints why, once."""

    class _DeadSession:
        def get(self, *a, **k):
            raise RuntimeError("group D is not answering")

    async def body():
        f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=6,
                  carrier_max_tokens=27466)
        f.session = _DeadSession()
        Seat(f, "running", "batch", tokens=27466)
        with caplog.at_level(logging.WARNING, logger=front_mod.logger.name):
            assert await f._d_pool_reading() is None
            assert await f._d_pool_reading() is None
        assert f._d_token_budget_blocks("r", 10 ** 9, None) is False, (
            "gate OFF, not a fallback to the front-local tally")
        assert caplog.text.count("WEG2 D-POOL UNREADABLE") == 1, "once, not per pass"
        # FIX 5: the SECOND decision inside the age window does not re-issue
        # the read at all (test_a10) -- it is answered from the negative
        # cache, which is a different event from a failure and is counted as
        # one.  Round 4 asserted `failed == 2` here, i.e. it asserted the
        # round trip that a silent group D charged to every admission.
        assert f.counters["d_pool_read_failed"] == 1
        assert f.counters["d_pool_read_suppressed"] == 1

    asyncio.run(body())


class _Resp:
    def __init__(self, body, status=200):
        self.status = status
        self._body = body

    async def json(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _CountingSession:
    """Answers the first ``ok`` reads and fails afterwards, counting GETs."""

    def __init__(self, reading, ok=1):
        self.calls = 0
        self.ok = ok
        self._body = {"internal_states": [{"hicache_prefetch": reading}]}

    def get(self, *a, **k):
        self.calls += 1
        if self.calls > self.ok:
            raise RuntimeError("group D went silent")
        return _Resp(self._body)


def _pool_front(session, **kw):
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0,
              d_bs=6, carrier_max_tokens=27466, **kw)
    f.session = session
    return f


def test_a9_a_reading_that_has_aged_out_is_no_reading(caplog):
    """FIX 5, finding 3.  The safety property both the docstring and the round-4
    commit message ASSERTED and nothing ASSERTED ON.

    Deleting the age term from the freshness check leaves the front deciding
    for ever on an arbitrarily old residency -- a stale pool number that
    cannot see what actually refuses the store read, the same indicator class
    round 3 killed -- and the named ``WEG2 D-POOL UNREADABLE`` refusal never
    fires again.  Round 4's test_a6 only ever exercised the FIRST-read-fails
    case (no prior reading at all), so that mutant survived: the whole suite
    stayed at the baseline 1 failed / 90 passed.

    Both arms, because one without the other is half a property:

    * WITHIN the bound the reading is reused and costs NO round trip;
    * PAST the bound it is not a reading at all -- ``None``, the named
      refusal, and a gate that is consequently off.
    """

    async def body():
        t0 = time.time()
        sess = _CountingSession(_reading(27466, 0, 27466, t0), ok=1)
        f = _pool_front(sess)
        Seat(f, "running", "batch", tokens=1)
        with caplog.at_level(logging.WARNING, logger=front_mod.logger.name):
            first = await f._d_pool_reading()
            assert first is not None and first["available"] == 27466
            assert sess.calls == 1

            # ARM ONE: inside the age bound, reused without asking D again.
            again = await f._d_pool_reading()
            assert again is first, "a fresh reading is reused, not re-read"
            assert sess.calls == 1, "and costs no round trip"
            assert "WEG2 D-POOL UNREADABLE" not in caplog.text

            # ARM TWO: the same reading, now older than its own bound.
            f._d_pool["t"] = time.time() - (front_mod.D_POOL_MAX_AGE_S + 5.0)
            f._d_pool_retry_after = 0.0
            aged = await f._d_pool_reading()
            assert aged is None, (
                "a reading older than D_POOL_MAX_AGE_S must not be handed to "
                "the gate -- that is the whole age term"
            )
            assert sess.calls == 2, "it tried to refresh first"
            assert f._d_pool is None
        assert caplog.text.count("WEG2 D-POOL UNREADABLE") == 1
        assert f._d_token_budget_blocks("r", 10 ** 9, None) is False, (
            "and with no reading the gate is OFF, never a fallback tally")

    asyncio.run(body())


def test_a10_a_failed_read_is_remembered_for_one_age_window(caplog):
    """FIX 5, finding 2, third consequence: NEGATIVE CACHING.

    Round 4 left ``_d_pool = None`` on a failure, so the next admission
    decision re-issued the request immediately.  A silent or slow group D then
    charged its request timeout to EVERY D admission decision, on both the
    BATCH admitter and the SHORT path.  A failure is now remembered exactly as
    long as a success would have been."""

    async def body():
        sess = _CountingSession(None, ok=0)
        f = _pool_front(sess)
        Seat(f, "running", "batch", tokens=1)
        with caplog.at_level(logging.WARNING, logger=front_mod.logger.name):
            t_fail = time.time()
            assert await f._d_pool_reading() is None
            assert sess.calls == 1 and f.counters["d_pool_read_failed"] == 1
            for _ in range(9):
                assert await f._d_pool_reading() is None
            assert sess.calls == 1, (
                "nine further decisions inside the age window must not each "
                "pay a round trip to a group that is not answering")
            assert f.counters["d_pool_read_suppressed"] == 9
            assert f.counters["d_pool_read_failed"] == 1, (
                "a suppressed read is counted apart from a failed one -- the "
                "UNREADABLE line's denominators name what they count")

            # ROUND 6, FINDING 1: THE BOUND ITSELF, READ OFF THE STAMP THE
            # CODE CHOSE.  Round 5 asserted the CONSEQUENCE (suppressed reads)
            # and then HAND-SET `_d_pool_retry_after` to prove the retry, so
            # the length of the window was never checked against anything: a
            # stamp of `t0 + 10 * D_POOL_MAX_AGE_S` -- a group D silent for
            # ten age windows after ONE slow answer -- left this file green,
            # and so did `t0 + 0` for the second half.  The window is now
            # asserted where the code writes it, with a tolerance for the
            # wall clock between the two readings and nothing more.
            window = f._d_pool_retry_after - t_fail
            assert front_mod.D_POOL_MAX_AGE_S <= window <= front_mod.D_POOL_MAX_AGE_S + 0.5, (
                f"a failed read must be remembered for exactly one age window "
                f"({front_mod.D_POOL_MAX_AGE_S:.1f} s), not {window:.3f} s: shorter and the "
                f"negative cache does not bound anything, longer and a single slow answer "
                f"blinds the gate for as long as the stamp says")

            # And the window is BOUNDED: it retries, it does not give up.
            f._d_pool_retry_after = time.time() - 0.001
            assert await f._d_pool_reading() is None
            assert sess.calls == 2

    asyncio.run(body())


def test_a11_the_reading_is_not_fetched_when_the_gate_cannot_refuse():
    """FIX 5, finding 2, THE ROOT: the read was a call ARGUMENT.

    ``_d_token_budget_blocks(rid, est, await self._d_pool_reading())`` makes
    Python evaluate the round trip BEFORE the callee's own two cheap guards
    can decline to use it.  So ``--d-admit-max-tokens 0`` -- documented in
    ``front.py`` and ``launcher.py`` as "0 disables the gate" -- still paid a
    full ``/server_info`` GET per admission decision, and so did the gate's
    never-starve exit, which is the path taken at every epoch's FIRST
    admission and at every unblock.  ``/server_info`` is not cheap
    (``dataclasses.asdict(server_args)`` plus a scheduler RPC), and its RTT
    lands inside the 0.05 s window the admitter races the controller's D->P
    arm in.

    The arming pre-check is SYNCHRONOUS by construction: a guard that may
    await is a guard that can cost what it exists to avoid."""

    async def body():
        # (a) the flag that says "disabled" disables the READ, not just the verdict.
        sess = _CountingSession(_reading(27466, 0, 27466, time.time()))
        off = _pool_front(sess, d_admit_max_tokens=0)
        Seat(off, "running", "batch", tokens=1)
        assert off._d_gate_armed() is False
        for _ in range(5):
            assert await off._d_reading_if_armed() is None
        assert sess.calls == 0, "a disabled gate must not poll group D at all"

        # (b) the never-starve exit: no seat in use, no reading needed.
        sess2 = _CountingSession(_reading(27466, 0, 27466, time.time()))
        idle = _pool_front(sess2)
        assert idle._d_gate_armed() is False, "no seat in use"
        assert await idle._d_reading_if_armed() is None
        assert sess2.calls == 0, (
            "the epoch's first admission and every unblock take this path")

        # (c) armed: the reading is fetched, and the gate can refuse on it.
        seat = Seat(idle, "running", "batch", tokens=1)
        assert idle._d_gate_armed() is True
        assert await idle._d_reading_if_armed() is not None
        assert sess2.calls == 1
        seat.release("test")

    asyncio.run(body())


def test_a12_neither_admission_path_awaits_the_read_as_an_argument():
    """The anti-regression for a11, at the two call sites themselves.

    A test that only exercises ``_d_reading_if_armed`` cannot see a call site
    that goes back to awaiting ``_d_pool_reading`` inline -- and that inline
    await is the entire defect, because argument evaluation happens first."""
    for meth in (Front._acquire_short_seat, Front.d_admitter):
        tree = _fn_ast(meth)
        assert not _calls_named(tree, "_d_pool_reading"), (
            f"{meth.__name__} must obtain the reading through "
            "_d_reading_if_armed, so the arming pre-check runs BEFORE the "
            "round trip rather than inside the callee"
        )
        assert _calls_named(tree, "_d_reading_if_armed"), (
            f"{meth.__name__} still has to price the pool")


def test_a13_the_read_timeout_is_derived_from_its_own_freshness_bound():
    """No hand numbers on this path.  Round 4 carried a bare
    ``ClientTimeout(total=5)``: with the gate off and a silent group D that is
    up to 5 s added to every D admission decision.  A reply that arrives later
    than ``D_POOL_MAX_AGE_S`` describes a pool state the next freshness check
    would discard anyway, so the timeout IS that bound."""
    assert front_mod.D_POOL_READ_TIMEOUT_S == front_mod.D_POOL_MAX_AGE_S
    tree = _fn_ast(Front._d_pool_reading)
    timeouts = [
        kw.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "timeout"
    ]
    assert timeouts, "the read must carry a timeout at all"
    for value in timeouts:
        names = {n.id for n in ast.walk(value) if isinstance(n, ast.Name)}
        assert "D_POOL_READ_TIMEOUT_S" in names, (
            "the timeout must be the derived constant, not a literal: "
            + ast.dump(value)
        )


def test_a14_the_forgone_prefix_reuse_is_priced_by_the_routing_probe(caplog):
    """MF-3 (a): the interim cost of W38 on group P, MEASURED per drain epoch.

    W38 refuses every storage read on group P -- correctly, because without
    the #631 row carrier a prefetch that completes on one PP rank and not
    another splits the geometry (the W27 divergence that killed weg2sc1).
    The consequence is that a multi-turn follow-up whose prefix has left P's
    device tier is prefilled WHOLE again: the user's soft no-double-prefill
    law paying for a hard correctness refusal.  MF-3 orders that cost priced
    rather than left as a sentence in a postmortem.

    THE DENOMINATOR IS THE FRONT'S OWN ROUTING PROBE, not a new one.
    ``price_remainder`` already asks the span LRU how much of an arriving
    prompt is a prefix this front saw realised before -- a prefix P prefilled
    and wrote through -- and the CARRIER-EXCEEDS / SHORT routing decision is
    taken on the difference.  ``est_prompt - remainder`` is therefore the
    store-resident estimate at no extra cost, and it is captured at ARRIVAL
    because leg 1 records this very text into the same LRU moments later.

    The reused term is MEASURED (P's own leg-1 ``cached_tokens``), never the
    literal 0 the finding predicts, so the day the carrier arrives and the
    read re-arms this line moves on its own instead of lying.
    """
    f = _front_with_reading(None)
    text = "a shared system preamble that two turns of one conversation carry"
    f.spans.record(text, 120)

    follow_up = text + " ... and the second turn's own question"
    remainder, est_prompt, known = front_mod.price_remainder(follow_up, f.spans)
    span = max(0, est_prompt - remainder)
    assert known and span > 0, "the routing probe itself must see this prefix"

    before = (f.counters.get("p_prefill_requests", 0),
              f.counters.get("p_prefix_tokens_in_store", 0),
              f.counters.get("p_prefix_tokens_reused", 0))
    p = Pending("weg2-3-1", "/generate", {}, follow_up, 0.0, None, store_span_est=span)
    f._note_p_prefix_reuse(p, 7)          # P's device tier covered 7 of them
    cold = Pending("weg2-3-2", "/generate", {}, "cold", 0.0, None)
    f._note_p_prefix_reuse(cold, 0)       # nothing of this one is in the store
    with caplog.at_level(logging.INFO, logger=front_mod.logger.name):
        f._log_p_prefix_reuse(before)

    line = [ln for ln in caplog.text.splitlines() if "WEG2 P-PREFIX-REUSE" in ln]
    assert len(line) == 1, "one line per drain epoch, printed with requests=0 too"
    assert f"requests={2}" in line[0]
    assert f"prefix_tokens_available_in_store={span}" in line[0]
    assert "prefix_tokens_reused=7" in line[0], (
        "the reused term is P's own leg-1 cached_tokens, a MEASUREMENT -- a "
        "hardcoded 0 would keep printing 0 after the carrier lands")
    assert f"forgone_tokens={span - 7}" in line[0], (
        "forgone = what the store held minus what P's device tier saved; that "
        "difference IS the double prefill W38 buys the geometry with")
    assert "W38" in line[0] and "#968" in line[0], (
        "the line must name the refusal it prices and the remedy that ends it")


def test_a15_a_w31_requeue_does_not_inflate_the_forgone_figure():
    """The one place the estimate is deliberately floored at 0, with its
    reason: a request D refused with W31 had an uncached extent LARGER than X
    after ``match_prefix``, i.e. the prefix the span LRU would price as
    store-resident demonstrably did not come back on D.  Counting it would
    inflate the cost of W38 with tokens no store read was going to save.  The
    line is a LOWER bound, and the code says so where it makes it one."""
    src = inspect.getsource(Front._requeue_after_x_refusal)
    assert "store_span_est" in src and "LOWER" in src, (
        "the W31 re-queue path must name why it prices no store span")
    assert "store_span_est=" not in src.split("Pending(")[1].split(")")[0], (
        "and must not pass one")
    assert Pending("weg2-4-1", "/generate", {}, "", 0.0, None).store_span_est == 0, (
        "the dataclass default is the floor")


def test_a7_the_reading_is_group_d_s_own_915_terms():
    """ANTI-DRIFT.  What the front reads must be what D's own #915 line
    prints, term for term -- otherwise the front decides on one arithmetic and
    D refuses on another, which is the round-3 finding one layer down."""
    from sglang.srt.mem_cache.prefetch_budget import prefetch_residency
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    pool = SimpleNamespace(size=30518, available_size=lambda: 5418, anchor_entry=None)
    cc = SimpleNamespace(mem_pool_host=pool, prefetch_tokens_occupied=25100,
                         prefetch_capacity_limit=27466)
    cache = SimpleNamespace(cache_controller=cc, prefetch_threshold=256)
    got = prefetch_residency(cache)
    line = UnifiedRadixCache._prefetch_line_terms(cache, 8629)
    for term in ("available", "occupied", "limit"):
        assert got[term] == line[term], f"{term} drifted from the #915 line"
    assert (got["available"], got["occupied"], got["limit"]) == (5418, 25100, 27466)
    assert prefetch_residency(SimpleNamespace()) is None, "no controller, no reading"


def test_a8_the_reading_is_published_on_the_endpoint_the_front_polls():
    """The channel itself.  Without this assignment the front has no number
    and the gate is permanently off -- green suite, dead coupling."""
    tree = _fn_ast(Scheduler.get_internal_state)
    calls = _calls_named(tree, "prefetch_residency") + [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "prefetch_residency"
    ]
    assert calls, "get_internal_state must publish the host-pool reading"
    keys = [
        n.slice.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
    ]
    assert "hicache_prefetch" in keys, "under the key the front reads"


# ================================= bb: the W27 root, carrier-less PP admission
def _pp_stub(pp_rank: int, carrier: bool):
    return SimpleNamespace(
        enable_hicache_storage=True,
        ps=SimpleNamespace(pp_size=3, pp_rank=pp_rank, tp_size=1),
        pp_flip_counters=SimpleNamespace() if carrier else None,
    )


def test_bb1_a_carrierless_pp_group_refuses_the_store_read():
    """(B) THE W27 ROOT.  Boot weg2sc1: a W31-refused request was re-queued to
    P; PP2's prefetch had completed and PP0/PP1's had not, so PP2 admitted
    ``(8747, 8748)`` and its peers ``(0, 4096)`` -- '#1233 W27 PP WIDTH
    DIVERGENCE REFUSED'.  A completed prefetch is not a credit but a
    GEOMETRY: it inserts into this rank's radix tree and lengthens
    ``prefix_indices``, which is what sizes the cross-stage tensor.  The TP
    axis MIN-reduces that completion; the PP axis has nothing, and #631's
    wire was reverted twice on metal.  So on a PP form without the row
    carrier the READ is refused by name -- the third member of the class
    that already disarmed PP0's #1066 wait (#973) and PP0's width cut
    (#1233)."""
    for rank in (0, 1, 2):
        assert Scheduler._carrierless_pp_store_read_refused(_pp_stub(rank, False)) is True, (
            "every rank of the group, or the disarm is itself an asymmetry")
    assert Scheduler._carrierless_pp_store_read_refused(_pp_stub(0, True)) is False, (
        "with the carrier the followers execute PP0's row; the read returns")
    tp_only = SimpleNamespace(ps=SimpleNamespace(pp_size=1, pp_rank=0), pp_flip_counters=None)
    assert Scheduler._carrierless_pp_store_read_refused(tp_only) is False, (
        "group D is TP-only: its store read is what Weg 2 depends on")


def test_bb2_the_refusal_is_the_first_exit_of_the_prefetch_and_is_counted():
    """It sits at ``_prefetch_kvcache``'s own gate, so all three issue sites
    (intake, the A12.2 retry, the #946 escape) are covered by one exit, and
    it is a member of the #915 intake partition -- an exit outside it would
    break ``intake == sum(partition)``."""
    from sglang.srt.mem_cache import match_refusal_census as census

    stub = _pp_stub(2, carrier=False)
    stub.tree_cache = None  # any use beyond the gate would raise
    stub._carrierless_pp_store_read_refused = MethodType(
        Scheduler._carrierless_pp_store_read_refused, stub
    )
    verdict = Scheduler._prefetch_kvcache(stub, SimpleNamespace(rid="requeued-1"))
    assert verdict == "declined:carrierless_pp"
    assert "carrierless_pp" in census.PREFETCH_INTAKE_PARTITION
    tree = _fn_ast(Scheduler._prefetch_kvcache)
    names = [n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert names.index("_carrierless_pp_store_read_refused") < 3, (
        "before anything that touches the tree")


def _geometry_worker(rank, init_file, out_dir, carrier):
    """One PP rank: build the admission geometry the way production does and
    all_gather it.  Rank 2's store read has completed; ranks 0 and 1's has
    not -- the order the metal produced."""
    import torch  # noqa: F401
    import torch.distributed as dist

    res = {"rank": rank, "error": None, "geom": None, "refused": None}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=3
        )
        from sglang.srt.managers.pp_admission_congruence import (
            PPWidthDivergenceRefused,
            refuse_pp_width_divergence,
        )
        from sglang.srt.managers.scheduler import Scheduler as S

        stub = SimpleNamespace(
            enable_hicache_storage=True,
            ps=SimpleNamespace(pp_size=3, pp_rank=rank, tp_size=1),
            pp_flip_counters=SimpleNamespace() if carrier else None,
        )
        read_refused = S._carrierless_pp_store_read_refused(stub)
        # The prompt: 8748 tokens, of which 8747 are in the store. The store
        # read has landed on rank 2 only (the metal's arrival order).
        store_span = 8747 if rank == 2 else 0
        prefix = 0 if read_refused else store_span
        extent = min(4096, 8748 - prefix)
        res["geom"] = [prefix, extent]
        res["refused"] = bool(read_refused)
        gathered = [None, None, None]
        dist.all_gather_object(gathered, [prefix, extent])
        res["all"] = gathered
        # the production width guard, on the group's own numbers
        try:
            refuse_pp_width_divergence(gathered[0][1], gathered[rank][1], "fix3 gloo")
            res["w27"] = False
        except PPWidthDivergenceRefused:
            res["w27"] = True
        dist.barrier()
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"
    with open(os.path.join(out_dir, f"r{rank}.json"), "w") as fh:
        json.dump(res, fh)


def _run_geometry(carrier: bool):
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "store")
        mp.spawn(_geometry_worker, args=(init_file, tmp, carrier), nprocs=3, join=True)
        return [json.load(open(os.path.join(tmp, f"r{r}.json"))) for r in range(3)]


def test_bb3_three_real_ranks_build_the_same_geometry_without_a_carrier():
    """(B), on three REAL processes, because "the ranks agree" is not
    observable in one.  Rank 2's store read completes first, as it did on
    metal.  WITHOUT the carrier every rank refuses the read and all three
    build ``(0, 4096)`` -- the width guard passes.  WITH a carrier present
    the read is taken and the same inputs produce ``(8747, 1)`` on rank 2
    against ``(0, 4096)`` on its peers, which is the W27 the boot died on --
    that arm is the can-fail proof for this test, and it is also why the
    disarm keys on the carrier rather than on the form."""
    out = _run_geometry(carrier=False)
    assert all(r["error"] is None for r in out), [r["error"] for r in out]
    geoms = [tuple(r["geom"]) for r in out]
    assert len(set(geoms)) == 1, geoms
    assert geoms[0] == (0, 4096), geoms
    assert all(r["refused"] for r in out)
    assert not any(r["w27"] for r in out), "no width divergence to refuse"

    div = _run_geometry(carrier=True)
    assert all(r["error"] is None for r in div), [r["error"] for r in div]
    dgeoms = [tuple(r["geom"]) for r in div]
    assert len(set(dgeoms)) == 3 - 1, dgeoms  # rank 2 apart from its two peers
    assert dgeoms[2] == (8747, 1), dgeoms
    assert div[2]["w27"] is True, (
        "the control arm must reproduce the boot killer, or the green arm "
        "proves nothing")
