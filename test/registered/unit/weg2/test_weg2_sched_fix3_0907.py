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
import inspect
import json
import logging
import os
import sys
import tempfile
from types import MethodType, SimpleNamespace

import pytest

from sglang.srt.managers import tp_head_congruence as thc
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2.front import Front, Pending


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
        "L11": (front_src, "WEG2 IDLE-REST", ["layout=", "ready_for_d=", "d_outstanding=", "held_s="]),
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
def test_a1_a_seat_is_taken_only_when_the_store_read_can_be_issued(caplog):
    """(A) The coupling the postmortem named unaddressed: six seats x ~9k
    tokens against a 27,466-token pool bound.  The head WAITS in arrival
    order -- it is not skipped over, and it takes no seat -- and it is served
    the moment a running request frees its tokens.  The wait prints
    ``WEG2 D-SEAT-WAIT`` with need/available/limit."""

    async def body():
        f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=6,
                  carrier_max_tokens=27466)
        ps = []
        for i in range(4):
            p = _bare_pending(f"r{i}")
            p.est_prompt = 8400
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
            assert f.seats_free() == 3, "and the COUNT alone would have admitted it"
            assert f._ready_for_d and f._ready_for_d[0] is ps[3], (
                "law 2: it keeps the head of the arrival queue, it is not skipped")
        text = caplog.text
        assert "WEG2 D-SEAT-WAIT" in text, text[-400:]
        assert "rid=r3" in text and "need=8400" in text
        assert "available=2266" in text and "limit=27466" in text
        assert "held_passes=" in text, "the suppressed-pass denominator"
        ps[0].seat.release("leg2_finished")
        assert await _until(lambda: ps[3].fut.done()), (
            "a freed seat's tokens must come back with it, or the gate wedges")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(body())


def test_a2_the_gate_never_starves_and_is_a_flag():
    """It only ever DELAYS: a request alone in flight is admitted whatever it
    costs (the oversized case is CARRIER-EXCEEDS's), and 0 disables it."""
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=6,
              carrier_max_tokens=1000)
    assert f._d_token_budget_blocks("solo", 900000) is False
    off = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=6,
                carrier_max_tokens=27466, d_admit_max_tokens=0)
    off._d_inflight_tokens = 10 ** 9
    assert off._d_token_budget_blocks("r", 10 ** 9) is False


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
