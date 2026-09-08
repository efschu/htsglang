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
"""Slice A, FIX 7: the X gate waits for the store read to LAND, and ``need``
is priced from the realised count.

THE MEASUREMENT THIS FILE IS BUILT FROM -- boot weg2sc3, 2026-09-08.  The
seat gate FIX 4/5 built passed its own acceptance criterion: ``WEG2
D-SEAT-WAIT`` carried a non-zero ``occupied=`` (23282 / 15047 / 8863) read
off group D's own ``#915`` terms.  And W31 still fired 25 times on servable
8.6-11.2k prompts, W35 10 times.  The refused requests were priced at their
WHOLE extent::

    WEG2 X-GATE rid=b8b55831aa954c5d uncached=8866 X=8192
      replicated_term=group verdict=W31

while the six that got through -- same prompts, same route, same gate --
priced at ``uncached = 2``.  The only difference was whether D's store read
for that rid had LANDED when the gate ran.  The gate holds on a CAPACITY
predicate (``available >= need``, ``occupied < limit``); law 4 needs a
COMPLETION predicate before it prices.

WHAT EACH GROUP CLOSES, and which arm is RED at the parent ``e314042062``:

* ``c1_*`` -- THE COMPLETION PREDICATE.  RED at the parent:
  ``_weg2_x_defers`` does not exist, so the gate prices whatever is in the
  radix tree at that instant.  The local predicate is deliberately NARROWER
  than ``check_prefetch_progress``, which answers True at its first line for
  a rid that is merely not in ``ongoing_prefetch`` (hiradix_cache.py:1914) --
  conflating "never registered", "just reaped" and "landed".
* ``c2_*`` -- THE FACT IS THE GROUP'S.  Group D is ``--tp-size 3``, and the
  downstream decision DELETES the request from a rank's queue, so a
  rank-local completion term makes the deletion rank-local and permanent.
  ``c2b`` runs three REAL gloo processes, because "the ranks agree" is not
  observable in one.
* ``c3_*`` -- NEED PRICING.  RED at the parent: ``need`` is the front's
  ``len(text)/3`` arrival estimate for ever (15,047 for a realised 8,865 =
  1.71x), so two requests charged 30,094 against a 27,466 limit and
  effective concurrency was 2 of 6; and ``available`` went to ``-1663``,
  which is not a physical quantity.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no model, no GPU.
"""

import ast
import inspect
import json
import logging
import os
import tempfile
import textwrap
import time
from types import MethodType, SimpleNamespace

import pytest

import sglang.srt.managers.scheduler as sched_mod
import sglang.srt.managers.tp_head_congruence as thc
import sglang.srt.weg2.front as front_mod
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.weg2.front import Front

# EVERY FIX-7 SYMBOL IS REACHED THROUGH ITS MODULE, never imported by name.
# A ``from ... import`` of something the parent commit does not have turns
# the whole file into ONE collection error, and eighteen tests that each name
# a different defect would arrive as a single unreadable line -- the
# extraction-count trap.  Reached this way, each test is red on its own
# assertion (or on an ``AttributeError`` naming exactly the missing member).
NEED_REALISED = "realised"
NEED_ESTIMATE = "estimate"


def d_seat_need(*a, **k):
    return front_mod.d_seat_need(*a, **k)


# --------------------------------------------------------------- helpers
def _fn_ast(fn):
    return ast.parse(inspect.cleandoc(inspect.getsource(fn)))


def _pending_head(rid: str, group_ms, match: int = 0):
    """A ``UniformHeadInputs`` carrying one rid's group pending age.

    ``group_ms=None`` means every rank voted neutral, which is what the
    payload's MIN produces when nothing is pending anywhere.
    """
    canonical = thc.canonical_head_rids([rid])
    ages = {} if group_ms is None else {rid: int(group_ms)}
    return thc.build_uniform_head_inputs(
        canonical,
        thc.build_head_order_payload(canonical, {rid: match}),
        None,
        True,
        thc.build_x_pending_payload(canonical, ages),
    )


def _req(rid="r", prompt_tokens=8866, prefix=0, host_hit=0,
         ongoing=False, deferred=False):
    return SimpleNamespace(
        rid=rid,
        full_untruncated_fill_ids=list(range(prompt_tokens)),
        origin_input_ids=list(range(prompt_tokens)),
        prefix_indices=list(range(prefix)),
        host_hit_length=host_hit,
        prefetch_deferred="rate_limited" if deferred else None,
        _ongoing=ongoing,
    )


def _defer_stub(reqs, *, base=1.0, per_page=0.01, page_size=64,
                priceable=True, x=8192, host_carry=10 ** 9):
    """A Scheduler stand-in with the REAL FIX-7 bodies bound.

    ``priceable=False`` drops the tree's three timeout terms, which is the
    tree ``_deferred_prefetch_bound_s`` refuses to answer for -- the arm that
    proves an unpriceable bound becomes "price now" and never an unbounded
    wait.
    """
    ongoing = {r.rid for r in reqs if getattr(r, "_ongoing", False)}
    tree = SimpleNamespace(
        ongoing_prefetch={r: object() for r in ongoing},
        cache_controller=SimpleNamespace(
            mem_pool_host=SimpleNamespace(size=host_carry)
        ),
    )
    if priceable:
        tree.prefetch_timeout_base = base
        tree.prefetch_timeout_per_page = per_page
        tree.page_size = page_size
    stub = SimpleNamespace(
        waiting_queue=list(reqs),
        tree_cache=tree,
        ps=SimpleNamespace(tp_size=3),
        server_args=SimpleNamespace(tp_prefill_max_tokens=x),
    )
    for name in (
        "_weg2_store_read_is_pending",
        "_weg2_local_store_read_pending_ages",
        "_weg2_local_store_read_pending_ms",
        "_weg2_x_store_read_bound_s",
        "_weg2_x_defers",
        "_deferred_prefetch_bound_s",
        "_weg2_x_refuses",
        "_weg2_host_carry_tokens",
        "weg2_uncached_extent",
    ):
        setattr(stub, name, MethodType(getattr(Scheduler, name), stub))
    return stub


def _stamp(stub, rid, age_s):
    stub._weg2_x_defer_since = dict(
        getattr(stub, "_weg2_x_defer_since", None) or {}
    )
    stub._weg2_x_defer_since[rid] = time.monotonic() - age_s


# ============================== c1: the completion predicate at the X gate
def test_c1a_the_gate_defers_while_the_group_store_read_is_in_flight(caplog):
    """THE FIX, in one assertion.  A request whose store read is still in
    flight is NOT priced: it stays in the queue, nothing is deleted and
    nothing is answered, and the line says why with its age and its bound.

    At the parent this request is priced at ``uncached=8866`` against
    ``X=8192`` and answered W31 -- the boot's own line, verbatim."""
    req = _req(prompt_tokens=8866, ongoing=True)
    stub = _defer_stub([req])
    _stamp(stub, "r", 0.5)
    head = _pending_head("r", 500)
    with caplog.at_level(logging.INFO, logger=sched_mod.logger.name):
        assert stub._weg2_x_defers(req, head) is True
    text = caplog.text
    assert "WEG2 X-DEFER" in text, "the defer must speak, or it is a silent skip"
    assert "reason=prefetch_pending" in text
    assert "age_s=0.50" in text, "the GROUP's age, not this rank's"
    assert "bound_s=" in text and "verdict=defer" in text
    assert "replicated_term=group" in text
    assert "held_passes=" in text, "the suppressed-count denominator"


def test_c1b_the_gate_prices_a_landed_read_and_admits_the_small_remainder():
    """The other half: once the read has landed the group holds no pending
    fact, the gate prices, and the extent is the ``uncached = 2`` the seven
    served requests of boot weg2sc3 actually showed -- so the defer costs the
    servable case nothing."""
    req = _req(prompt_tokens=8865, prefix=8863)
    stub = _defer_stub([req], x=8192)
    head = _pending_head("r", None, match=8863)
    assert stub._weg2_x_defers(req, head) is False, "nothing pending -> price"
    assert stub.weg2_uncached_extent(req, head) == 2
    assert stub._weg2_x_refuses(req, head) is False, "and 2 <= X admits"


def test_c1c_past_the_bound_the_request_is_priced_and_w31_fires_honestly(caplog):
    """THE DEFER IS BOUNDED, so a read that never lands cannot wedge the
    request -- that would be the same livelock in the other costume.  Past
    the span's own length-priced store-read timeout the request IS priced and
    W31 may fire, on an extent nothing further was going to improve."""
    req = _req(prompt_tokens=8866, ongoing=True)
    stub = _defer_stub([req], base=1.0, per_page=0.01, page_size=64, x=8192)
    bound = stub._weg2_x_store_read_bound_s(req)
    _stamp(stub, "r", bound + 5.0)
    head = _pending_head("r", int((bound + 5.0) * 1000))
    with caplog.at_level(logging.INFO, logger=sched_mod.logger.name):
        assert stub._weg2_x_defers(req, head) is False, "the bound expired"
        assert stub._weg2_x_refuses(req, head) is True, "and W31 fires honestly"
    assert "verdict=bound_expired" in caplog.text
    assert "verdict=W31" in caplog.text


def test_c1d_the_completion_gate_is_the_statement_directly_above_the_price_gate():
    """PLACEMENT, over the AST, for the same reason ``test_b1a`` asserts law
    4's own: a completion check that any statement can be slipped between is
    not a precondition of the pricing, it is a suggestion.

    Three structural facts: the defer is a DIRECT statement of the
    ``for req in self.waiting_queue`` body, it is IMMEDIATELY before the
    ``if self._weg2_x_refuses(...)`` gate, and its body skips WITHOUT
    collecting the request for the W31 answer (a deferred request is not a
    refused one and must not be answered by name)."""
    tree = _fn_ast(Scheduler._get_new_batch_prefill_raw)

    def _gates(attr):
        return [
            n for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Call)
            and isinstance(n.test.func, ast.Attribute)
            and n.test.func.attr == attr
        ]

    defers, prices = _gates("_weg2_x_defers"), _gates("_weg2_x_refuses")
    assert len(defers) == 1, "exactly one `if self._weg2_x_defers(...)`"
    assert len(prices) == 1
    loops = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.For) and defers[0] in n.body and prices[0] in n.body
    ]
    assert len(loops) == 1, (
        "both gates must be DIRECT statements of the waiting-queue loop -- "
        "nested deeper, the completion check runs under whatever wraps it"
    )
    body = loops[0].body
    assert body.index(prices[0]) == body.index(defers[0]) + 1, (
        "the completion check must sit IMMEDIATELY above the pricing gate: a "
        "statement in between can price, skip or mutate the request before "
        "law 4's own gate ever sees it"
    )
    dumped = ast.dump(ast.Module(body=defers[0].body, type_ignores=[]))
    assert "Continue" in dumped, "a deferred request leaves this pass"
    assert "_x_refused" not in dumped, (
        "and is NOT collected for the W31 answer -- deferring is not refusing, "
        "and answering it by name would delete it from the queue"
    )


def test_c1e_the_local_predicate_is_narrower_than_check_prefetch_progress():
    """THE DEFECT, isolated.  ``check_prefetch_progress`` returns True at its
    first line for any rid not in ``ongoing_prefetch``, so three different
    states share one answer.  The completion vote must separate them: a read
    in flight and a rate-limited read waiting to be re-issued are PENDING; a
    read that terminated having loaded nothing is not, and prices at once."""
    stub = _defer_stub([])
    in_flight = _req(rid="a", ongoing=True)
    deferred = _req(rid="b", deferred=True)
    terminated = _req(rid="c")
    stub.tree_cache.ongoing_prefetch = {"a": object()}
    assert stub._weg2_store_read_is_pending(in_flight) is True
    assert stub._weg2_store_read_is_pending(deferred) is True, (
        "the #1068 A12.2 mark is a read that WILL be re-issued -- pricing "
        "through it is exactly what the deferral exists to avoid"
    )
    assert stub._weg2_store_read_is_pending(terminated) is False, (
        "a read that is not coming back must price immediately, or the defer "
        "becomes the wedge it was built to prevent"
    )


def test_c1f_the_bound_is_derived_from_the_trees_own_prefetch_timeout():
    """DERIVED, NOT PICKED: the same ``base + pages x per_page`` the #1068
    deferral machinery already prices a store read with.  One derivation, two
    consumers -- there is no second number to keep in step.

    And an UNPRICEABLE bound is 0.0, which the verdict turns into "price
    now": a missing bound may never mean an unbounded wait."""
    req = _req(prompt_tokens=640)
    stub = _defer_stub([req], base=2.0, per_page=0.5, page_size=64)
    assert stub._weg2_x_store_read_bound_s(req) == pytest.approx(2.0 + 10 * 0.5)
    assert stub._weg2_x_store_read_bound_s(req) == pytest.approx(
        stub._deferred_prefetch_bound_s(640)
    ), "the X gate and the deferral must price the same span the same way"
    blind = _defer_stub([req], priceable=False)
    assert blind._weg2_x_store_read_bound_s(req) == 0.0
    assert thc.x_completion_verdict(10, 0.0) == thc.X_BOUND_EXPIRED


def test_c1g_the_defer_stamps_have_a_writer_a_reader_and_a_deleter():
    """LIFECYCLE TABLE for ``_weg2_x_defer_since``, asserted rather than
    written in a comment (the rule every new state field carries).  The age
    must measure THE WAIT, not the request's life, and a rid that leaves the
    head must take its stamp with it -- otherwise the dict outgrows the head
    and a re-queued rid inherits an ancient clock that expires its bound
    instantly."""
    a, b = _req(rid="a", ongoing=True), _req(rid="b", ongoing=True)
    stub = _defer_stub([a, b])
    ages = stub._weg2_local_store_read_pending_ages(["a", "b"])
    assert set(ages) == {"a", "b"} and all(v >= 0 for v in ages.values())
    assert set(stub._weg2_x_defer_since) == {"a", "b"}
    # b's read lands; a's does not.
    stub.tree_cache.ongoing_prefetch.pop("b")
    b._ongoing = False
    again = stub._weg2_local_store_read_pending_ages(["a", "b"])
    assert set(again) == {"a"}
    assert set(stub._weg2_x_defer_since) == {"a"}, "the landed rid's stamp is reaped"
    # a leaves the queue entirely.
    stub.waiting_queue = []
    assert stub._weg2_local_store_read_pending_ages(["a"]) == {}
    assert stub._weg2_x_defer_since == {}, "and so is a departed rid's"


def test_c1h_the_single_rid_read_does_not_reap_its_neighbours():
    """The reader may not be the writer under another name.  ``_weg2_x_defers``
    asks for ONE rid's age on every gate call; if that read reaped, every other
    rid's clock would restart on every pass and no bound could ever expire."""
    a, b = _req(rid="a", ongoing=True), _req(rid="b", ongoing=True)
    stub = _defer_stub([a, b])
    stub._weg2_local_store_read_pending_ages(["a", "b"])
    before = dict(stub._weg2_x_defer_since)
    assert stub._weg2_local_store_read_pending_ms(a) is not None
    assert stub._weg2_x_defer_since == before, "a read with side effects"


class _No:
    """Answers every question "no" and every call with another ``_No``.

    The same permissive stand-in ``test_b1e`` runs law 4's reach slice on, and
    for the same reason: every guard above the gate asks for a reason to SKIP
    this request, so answering all of them "no" leaves exactly one thing that
    can still keep control from arriving -- a statement that skips whatever it
    is told.
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


def _run_gate_slice(defers: bool):
    """Execute the REAL waiting-queue loop, cut after law 4's pricing gate,
    with a spy on ``_weg2_x_refuses`` and the DEFER answering ``defers``.

    Built from ``inspect.getsource`` at call time, so it is the shipping
    source that runs.  Same three surgeries as ``test_b1e``'s slice: cut after
    the gate, append an observable sentinel, and turn every free name into a
    parameter so an edit that reaches for one more local fails loudly.
    """
    import builtins

    tree = _fn_ast(Scheduler._get_new_batch_prefill_raw)
    gates = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.If) and isinstance(n.test, ast.Call)
        and isinstance(n.test.func, ast.Attribute)
        and n.test.func.attr == "_weg2_x_refuses"
    ]
    assert len(gates) == 1
    loop = [n for n in ast.walk(tree)
            if isinstance(n, ast.For) and gates[0] in n.body][0]
    idx = loop.body.index(gates[0])
    trunc = ast.For(
        target=loop.target, iter=loop.iter,
        body=loop.body[: idx + 1] + [ast.parse("_past.append(req)").body[0]],
        orelse=[], type_comment=None,
    )
    bound = {n.id for n in ast.walk(trunc)
             if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
    free = sorted({n.id for n in ast.walk(trunc)
                   if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                  - bound - set(dir(builtins)))
    fn_def = ast.FunctionDef(
        name="_slice",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in free],
                           vararg=None, kwonlyargs=[], kw_defaults=[],
                           kwarg=None, defaults=[]),
        body=[trunc], decorator_list=[], returns=None, type_params=[],
    )
    mod = ast.Module(body=[fn_def], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {}
    exec(compile(mod, inspect.getsourcefile(Scheduler), "exec"), ns)

    skips, refused, past, priced = [], [], [], []
    req = SimpleNamespace(rid="weg2-0-3", init_next_round_input=lambda *a, **k: None)
    kw = {name: _NO for name in free}
    kw["self"] = _No(
        waiting_queue=[req],
        _weg2_x_defers=lambda r, h=None: defers,
        _weg2_x_refuses=lambda r, h=None: (priced.append(r), False)[1],
    )
    kw["_note_skip"] = lambda kind, rid: skips.append((kind, rid))
    kw["_x_refused"] = refused
    kw["_past"] = past
    ns["_slice"](**kw)
    return SimpleNamespace(skips=skips, refused=refused, past=past, priced=priced)


def test_c1i_a_deferred_request_never_REACHES_the_pricing_gate():
    """THE ROUND-5 LESSON APPLIED TO THIS FIX: an AST placement check proves
    the statement is PRESENT, never that it RUNS.  ``test_c1d`` is the cheap
    structural companion; this executes the real loop.

    Deferring must keep the request out of ``_weg2_x_refuses`` ALTOGETHER --
    not merely change what that method answers.  Pricing and then discarding
    the verdict would still emit an L9 line and still consume the pass's
    reduce, and a reader of the boot log could not tell a deferred request
    from an admitted one."""
    got = _run_gate_slice(defers=True)
    assert got.priced == [], (
        "law 4's pricing gate was ENTERED for a request whose store read is "
        "still in flight -- that is boot weg2sc3's uncached=8866 W31 exactly"
    )
    assert got.skips == [("weg2_x_defer", "weg2-0-3")], "named in the trace"
    assert got.refused == [], "a deferred request is not answered by name"
    assert got.past == [], "and does not continue into the rest of the pass"


def test_c1j_an_undeferred_request_still_reaches_the_pricing_gate():
    """The can-fail half of ``c1i``: with nothing pending the completion check
    must be inert, so law 4 is enforced exactly as before.  A defer that held
    everything would pass ``c1i`` and starve the group."""
    got = _run_gate_slice(defers=False)
    assert len(got.priced) == 1, "law 4 must still be enforced on D"
    assert got.skips == [] and got.past
    assert got.refused == []


# ================================ c2: the fact is the GROUP's, or not taken
def _min_reduce(payloads):
    return [min(col) for col in zip(*payloads)]


def test_c2a_pending_on_any_rank_is_pending_for_the_group():
    """THE REDUCE'S DIRECTION, which is the whole safety argument.  MIN over
    the pending arm means "pending on ANY rank" -- delay-never-force, the same
    direction as the #791b ballot's MIN == AND -- and the reduced value is the
    YOUNGEST timer in the group, the conservative age to price a bound
    against.

    The can-fail half is the second block: on these same three readings a
    RANK-LOCAL verdict splits the group two ways, and the split decision is
    the one that deletes the request from a queue."""
    canonical = thc.canonical_head_rids(["r"])
    locals_ = [{"r": 900}, {}, {"r": 300}]
    payloads = [thc.build_x_pending_payload(canonical, m) for m in locals_]
    reduced = _min_reduce(payloads)
    inputs = [
        thc.build_uniform_head_inputs(canonical, [0], None, True, reduced)
        for _ in range(3)
    ]
    seen = {thc.group_store_read_pending_ms(i, "r") for i in inputs}
    assert seen == {300}, "every rank reads ONE number, the youngest pending"
    verdicts = {thc.x_completion_verdict(
        thc.group_store_read_pending_ms(i, "r"), 5.0) for i in inputs}
    assert verdicts == {thc.X_DEFER}, "and therefore ONE verdict"
    rank_local = {
        thc.x_completion_verdict(m.get("r"), 5.0) for m in locals_
    }
    assert rank_local == {thc.X_DEFER, thc.X_PRICE}, (
        "CAN-FAIL: on the very same readings a rank-local term has rank 1 "
        "pricing (and W31-deleting) a request its peers are still fetching"
    )


def test_c2b_no_group_opinion_prices_rather_than_defers():
    """ABSTAIN-NEVER-DEFER, the mirror of the X gate's own
    abstain-never-refuse.  No vote taken, a rid outside the canonical head, a
    single rank: all of them price, which is exactly the behaviour that
    existed before this arm.  A defer on no evidence would wedge every boot
    that does not run the reduce."""
    req = _req(rid="r", ongoing=True)
    stub = _defer_stub([req])
    _stamp(stub, "r", 0.5)
    assert stub._weg2_x_defers(req, None) is False, "no verdict published"
    canonical = thc.canonical_head_rids(["other"])
    outside = thc.build_uniform_head_inputs(
        canonical,
        thc.build_head_order_payload(canonical, {"other": 0}),
        None, True,
        thc.build_x_pending_payload(canonical, {"other": 500}),
    )
    assert stub._weg2_x_defers(req, outside) is False, "rid outside the head"
    no_arm = thc.build_uniform_head_inputs(
        thc.canonical_head_rids(["r"]), [0], None, True
    )
    assert thc.group_store_read_pending_ms(no_arm, "r") is None, (
        "an EMPTY arm is 'no opinion', never 'nothing pending' -- the second "
        "would license pricing on a read nobody asked about"
    )
    assert stub._weg2_x_defers(req, no_arm) is False


def test_c2c_a_group_age_above_this_ranks_own_is_the_w37_stop():
    """RANKS NEVER DISAGREE.  MIN can only ever be <= a voting rank's own
    value, so a group age ABOVE it means the ranks indexed the canonical head
    differently -- a head-mapping split, not a completion question.  The group
    stops by name instead of deferring or pricing on a term it cannot trust,
    the same shape and the same W number as the X-TERM split."""
    req = _req(rid="r", ongoing=True)
    stub = _defer_stub([req])
    _stamp(stub, "r", 0.1)
    with pytest.raises(RuntimeError) as exc:
        stub._weg2_x_defers(req, _pending_head("r", 9000))
    assert "W37" in str(exc.value) and "X-COMPLETION SPLIT STOP" in str(exc.value)
    # and the legitimate direction is NOT a stop: a rank that is not pending
    # at all abstains from the check rather than raising on its peers' wait.
    landed = _req(rid="r", ongoing=False)
    calm = _defer_stub([landed])
    assert calm._weg2_x_defers(landed, _pending_head("r", 500)) is True


def test_c2d_the_arm_rides_the_existing_reduce_and_adds_no_collective():
    """MUST NOT 6.  The completion arm exists precisely because a second
    collective on the prefetch axis is a recorded fatal in this tree (#580).
    Asserted structurally: ``_update_uniform_pool_budget`` still takes exactly
    ONE ``all_reduce``, the pending payload is appended BEFORE it, and the
    slice is read back by a head index captured before the ballot."""
    src = inspect.getsource(Scheduler._update_uniform_pool_budget)
    tree = ast.parse(inspect.cleandoc(src))
    reduces = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "all_reduce"
    ]
    assert len(reduces) == 1, "one collective, and the arm may not add a second"
    builds = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "build_x_pending_payload"
    ]
    assert len(builds) == 1, "the arm must be voted into that one payload"
    assert builds[0].lineno < reduces[0].lineno, "and appended before it reduces"
    assert "_xpend_at = len(vals)" in src, (
        "read back by a captured head index, like the corridor width and the "
        "head block -- a negative index would silently start reading the ballot"
    )
    assert "_xpend_lens" in src and "build_uniform_head_inputs" in src


def _gloo_worker(rank, init_file, out_dir):
    """One TP rank of group D.  Rank 0's store read is 900 ms old, rank 1's
    has landed, rank 2's is 300 ms old -- three genuinely different local
    readings, which is the situation the arm exists for."""
    import torch
    import torch.distributed as dist

    res = {"rank": rank, "error": None}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=3
        )
        import sglang.srt.managers.tp_head_congruence as t

        canonical = t.canonical_head_rids(["weg2-0-3"])
        local = [{"weg2-0-3": 900}, {}, {"weg2-0-3": 300}][rank]
        payload = t.build_x_pending_payload(canonical, local)
        tt = torch.tensor(payload, dtype=torch.int64)
        dist.all_reduce(tt, op=dist.ReduceOp.MIN)
        inputs = t.build_uniform_head_inputs(
            canonical, [0], None, True, tt.tolist()
        )
        res["group_ms"] = t.group_store_read_pending_ms(inputs, "weg2-0-3")
        res["verdict"] = t.x_completion_verdict(res["group_ms"], 5.0)
        res["rank_local_verdict"] = t.x_completion_verdict(
            local.get("weg2-0-3"), 5.0
        )
        dist.barrier()
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"
    with open(os.path.join(out_dir, f"r{rank}.json"), "w") as fh:
        json.dump(res, fh)


def test_c2e_three_real_ranks_take_the_same_defer_verdict():
    """THE GROUP PROPERTY ON THREE REAL PROCESSES, because "the ranks agree"
    is not observable in one -- the same reason ``test_bb3`` spawns.  Group D
    is ``--tp-size 3``; the decision below this one deletes the request from a
    rank's ``waiting_queue`` and answers it 503, so a split here is permanent
    and silent off rank 0.

    The rank-local column is this test's can-fail half: it is genuinely
    two-valued on the same readings."""
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "store")
        mp.spawn(_gloo_worker, args=(init_file, tmp), nprocs=3, join=True)
        out = [json.load(open(os.path.join(tmp, f"r{r}.json"))) for r in range(3)]
    assert all(r["error"] is None for r in out), [r["error"] for r in out]
    assert {r["group_ms"] for r in out} == {300}, (
        "all three ranks read the group's YOUNGEST pending timer"
    )
    assert {r["verdict"] for r in out} == {thc.X_DEFER}, "one verdict, three ranks"
    assert len({r["rank_local_verdict"] for r in out}) == 2, (
        "CAN-FAIL: the rank-local term really does split these three ranks"
    )


# ========================================= c3: need priced from the realised
def test_c3a_need_is_realised_after_leg1_and_the_estimate_only_when_cold():
    """The front already HOLDS the realised count -- group P's leg 1 answers
    with the tokenizer's own ``prompt_tokens`` for this exact prompt -- and
    kept charging the arrival estimate anyway."""
    assert d_seat_need(15047, 8865) == (8865, NEED_REALISED)
    assert d_seat_need(15047, 0) == (15047, NEED_ESTIMATE)
    assert d_seat_need(0, 0) == (0, NEED_ESTIMATE)
    assert d_seat_need(-5, -5) == (0, NEED_ESTIMATE), "no negative charge"


def test_c3b_the_seat_wait_line_names_the_provenance_of_need(caplog):
    """The L-line's own convention: a number whose error bars are unknown is
    not readable.  ``need=8865 source=realised`` and ``need=15047
    source=estimate`` are the same field with different meanings."""
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=6,
              carrier_max_tokens=27466)
    f._d_seats_live = [SimpleNamespace(tokens=0, t_taken=0.0)]
    reading = {"t": time.time(), "available": 100, "limit": 27466, "occupied": 0}
    with caplog.at_level(logging.INFO, logger=front_mod.logger.name):
        assert f._d_token_budget_blocks("weg2-0-3", 15047, reading, 8865) is True
    assert "WEG2 D-SEAT-WAIT" in caplog.text
    assert "need=8865" in caplog.text and "source=realised" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=front_mod.logger.name):
        assert f._d_token_budget_blocks("weg2-0-9", 15047, reading, 0) is True
    assert "need=15047" in caplog.text and "source=estimate" in caplog.text


def test_c3c_available_clamps_at_zero_and_prints_the_overshoot(caplog):
    """``available=-1663`` on metal.  That is
    ``reading["available"] - charged_since_reading`` extrapolated past the
    reading it hangs on -- not a physical quantity, and a reader had no way to
    tell an exhausted pool from a stale anchor.  Two questions, two numbers."""
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=6,
              carrier_max_tokens=27466)
    t0 = time.time()
    f._d_seats_live = [SimpleNamespace(tokens=15047, t_taken=t0 + 1)]
    reading = {"t": t0, "available": 13384, "limit": 27466, "occupied": 0}
    with caplog.at_level(logging.INFO, logger=front_mod.logger.name):
        assert f._d_token_budget_blocks("weg2-0-4", 15308, reading, 0) is True
    text = caplog.text
    assert "available=0" in text, "clamped, never negative"
    assert "available=-" not in text
    assert "over_charged=1663" in text, (
        "and the overshoot kept as its own number rather than swallowed by a "
        "sign, so a stale anchor is distinguishable from an exhausted pool"
    )


def test_c3d_the_realised_price_seats_two_where_the_estimate_seated_one():
    """THE BOOT'S OWN ARITHMETIC, made falsifiable.  Effective concurrent
    seats were 2 of 6 with a 27,466-token limit and 8.6-9k prompts, and the
    reason was the left-hand side of the comparison: at ``need=15047`` two
    requests charge 30,094 and the second cannot be seated at all."""
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=6,
              carrier_max_tokens=27466)
    t0 = time.time()
    reading = {"t": t0, "available": 27466, "limit": 27466, "occupied": 0}
    # One request already seated at the ESTIMATE: the second is refused.
    f._d_seats_live = [SimpleNamespace(tokens=15047, t_taken=t0 + 1)]
    assert f._d_token_budget_blocks("b", 15308, reading, 0) is True
    # The same two requests at their REALISED extents fit with room to spare.
    f._d_seats_live = [SimpleNamespace(tokens=8865, t_taken=t0 + 1)]
    assert f._d_token_budget_blocks("b", 15308, reading, 8642) is False


def test_c3e_the_admitter_charges_the_same_number_the_gate_priced():
    """One request, one price.  If the gate consults the realised count and
    the seat is then charged the estimate, ``_d_charged_since`` re-introduces
    the 1.71x over-price one call later and the fix is cosmetic."""
    src = inspect.getsource(front_mod.Front.d_admitter)
    tree = ast.parse(textwrap.dedent(src))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_d_token_budget_blocks"
    ]
    assert len(calls) == 1
    assert any("leg1_prompt_tokens" in ast.dump(a) for a in calls[0].args), (
        "the gate must be handed the realised count"
    )
    seats = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "Seat"
    ]
    assert len(seats) == 1
    dumped = ast.dump(seats[0])
    assert "d_seat_need" in dumped and "leg1_prompt_tokens" in dumped, (
        "and the seat charged through the SAME derivation, or the gate and "
        "the charge price one request two ways"
    )
