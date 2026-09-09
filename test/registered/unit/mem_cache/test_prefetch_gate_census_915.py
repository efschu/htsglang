"""#915: a prefetch that was never attempted left no trace at all.

THE SECOND HALF OF THE SAME ZERO. #914 answered "why did the match refuse".
It does not answer the question that follows immediately: a walk that matches
nothing SHOULD fall through to an L3 storage prefetch, so "the match refused"
and "no prefetch was attempted" are two different failures and only the first
was instrumented.

MEASURED, 0826 window R7. Prefetch was attempted on 264 of 675 census-sampled
match walks (138 with completed_local=0, 126 with completed_local=4096). The
other 411 declined inside `UnifiedRadixCache.prefetch_from_storage` and left
NO counter and NO log line. The gate is a three-term conjunction:

    eligible = (locally_eligible
                and prefetch_length >= self.prefetch_threshold
                and not self.cache_controller.prefetch_rate_limited())

so 411 declines are three unrelated verdicts wearing one boolean, and the
remedies point at three different files:

  anchor        the caller's local gate, `last_host_node.backuped`
                (scheduler.py:4933, which admits the ROOT on purpose -- so a
                fully-refused match does NOT decline here, and anyone assuming
                the mamba refusal starves the prefetch is guessing)
  too_short     fewer than prefetch_threshold (256) new tokens to fetch
  rate_limited  prefetch_tokens_occupied >= prefetch_capacity_limit, and that
                limit is `0.5 * mem_pool_host.size` (cache_controller.py:729)
                -- which across a phase flip is not one number. #905 measured
                the two host tiers at 703472 rows (PP) and 30518 (TP), 23x
                apart, putting the TP-phase budget at ~15259 tokens: under
                four prefetches of the 4096 this window actually completed.

That last one is a HYPOTHESIS with an arithmetic fit, not a finding, and it is
written here as such. The counter is what turns it into either on the next
boot. This ticket deliberately changes no behaviour: it records the verdict the
code was already reaching.

WHY IT IS NOT ARMED BEHIND AN ENV FLAG, unlike the #904 match census. That one
builds an object and walks validators a second time, so it pays for itself only
when asked. This is one integer increment on a path that already builds a
RadixKey and takes a host lock. And a gate that counts only when someone
remembered to arm it cannot answer "was it ever tried" -- which is the whole
question.
"""

import logging
import unittest

from sglang.srt.mem_cache.match_refusal_census import (
    PREFETCH_GATE_COUNTS,
    format_prefetch_gate,
    note_prefetch_gate,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _CleanCounts(CustomTestCase):
    def setUp(self):
        PREFETCH_GATE_COUNTS.clear()

    def tearDown(self):
        PREFETCH_GATE_COUNTS.clear()


class TestTheThreeDeclinesAreSeparated(_CleanCounts):
    def test_each_reason_is_counted_under_its_own_name(self):
        note_prefetch_gate("anchor")
        note_prefetch_gate("too_short")
        note_prefetch_gate("rate_limited")
        note_prefetch_gate("rate_limited")
        self.assertEqual(PREFETCH_GATE_COUNTS["anchor"], 1)
        self.assertEqual(PREFETCH_GATE_COUNTS["too_short"], 1)
        self.assertEqual(PREFETCH_GATE_COUNTS["rate_limited"], 2)

    def test_an_attempt_is_counted_too_so_the_denominator_is_local(self):
        """#873: a denominator reconstructed from a different log is how a
        narrowed candidate set gets read as a decomposition."""
        note_prefetch_gate(None)
        note_prefetch_gate("anchor")
        self.assertEqual(PREFETCH_GATE_COUNTS["attempted"], 1)

    def test_the_parts_sum_to_every_call(self):
        for reason in (None, None, "anchor", "too_short", "rate_limited", None):
            note_prefetch_gate(reason)
        total = sum(
            v for k, v in PREFETCH_GATE_COUNTS.items() if not k.endswith("_tokens")
        )
        self.assertEqual(total, 6)

    def test_tokens_are_tracked_apart_from_counts(self):
        note_prefetch_gate("rate_limited", 4096)
        note_prefetch_gate("rate_limited", 4096)
        self.assertEqual(PREFETCH_GATE_COUNTS["rate_limited"], 2)
        self.assertEqual(PREFETCH_GATE_COUNTS["rate_limited_tokens"], 8192)

    def test_zero_tokens_does_not_invent_a_token_key(self):
        note_prefetch_gate("anchor", 0)
        self.assertNotIn("anchor_tokens", PREFETCH_GATE_COUNTS)


class TestTheInstrumentCanSayItDidNotMeasure(_CleanCounts):
    """#829/INDIKATOR-GESETZ: an instrument that cannot say 'I did not measure'
    is indistinguishable from one that measured nothing."""

    def test_no_observation_is_stated_not_implied_as_zero(self):
        self.assertEqual(format_prefetch_gate(), "[#915 prefetch-gate] no observation")

    def test_a_recorded_verdict_produces_a_greppable_line(self):
        note_prefetch_gate("rate_limited", 4096)
        line = format_prefetch_gate()
        self.assertIn("[#915 prefetch-gate]", line)
        self.assertIn("rate_limited=1", line)
        self.assertIn("rate_limited_tokens=4096", line)


class TestTheGateIsWiredAndOrdered(CustomTestCase):
    """PRESENT-AND-VERDRAHTET. A counter nothing calls is the middle of the
    three delivery states and the most expensive to mistake for either end."""

    def _src(self):
        import inspect

        from sglang.srt.mem_cache import unified_radix_cache

        return inspect.getsource(
            unified_radix_cache.UnifiedRadixCache.prefetch_from_storage
        )

    def test_the_gate_records_its_verdict(self):
        self.assertIn("_note_prefetch_gate(reason", self._src())

    def test_all_three_reasons_are_reachable_from_the_gate(self):
        src = self._src()
        for reason in ('"anchor"', '"too_short"', '"rate_limited"'):
            self.assertIn(reason, src)

    def test_eligibility_is_derived_from_the_reason_not_computed_twice(self):
        """Two expressions of one rule drift; #747 records these very match
        lineages doing it. `eligible` must BE the reason's absence."""
        self.assertIn("eligible = reason is None", self._src())

    def test_the_first_failing_term_is_named_not_all_of_them(self):
        """A request can trip several. Summing them would double-count exactly
        the way refused_tokens_by_component is documented to."""
        src = self._src()
        self.assertIn("if not locally_eligible:", src)
        self.assertIn("elif prefetch_length < self.prefetch_threshold:", src)
        self.assertIn("elif self.cache_controller.prefetch_rate_limited():", src)

    def test_the_rate_limit_check_is_still_called_at_most_once(self):
        """It reads a live counter; calling it twice per gate would be a second
        reading of a moving quantity, and the two could disagree.

        COMMENT LINES ARE STRIPPED FIRST, and that is not incidental. Counting
        occurrences in raw source counts the prose too -- this very function
        carries a pre-existing comment naming `prefetch_rate_limited()` at
        :2674, which made the naive assertion read 2 and fail for a reason that
        has nothing to do with the code. Matching on prose to reach a verdict
        about code is the #908 substring defect; a test may not commit it
        either.
        """
        code = "\n".join(
            line
            for line in self._src().splitlines()
            if not line.lstrip().startswith("#")
        )
        self.assertEqual(code.count("prefetch_rate_limited()"), 1)


class TestTheAttributionOrderIsTheExtendedOne(CustomTestCase):
    """#1068 slice 4 INVERTS the old 'the three terms are the same three as
    before': every exit behind the gate now has a term, and `gate_reason_since`
    attributes in ONE fixed order (first tripped wins) that the boot acceptance
    and the #969C verdict string both read."""

    def test_the_attribution_order_is_the_extended_one(self):
        from sglang.srt.mem_cache.match_refusal_census import (
            PREFETCH_DECLINE_ORDER,
        )

        self.assertEqual(
            PREFETCH_DECLINE_ORDER,
            (
                "anchor",
                "too_short",
                "rate_limited",
                "host_pool_exhausted",
                "host_alloc_failed",
                "anchor_pool_exhausted",
                "vote_negative",
                "alloc_failed_post_vote",
            ),
        )

    def test_the_first_tripped_term_in_order_is_the_verdict(self):
        from sglang.srt.mem_cache.match_refusal_census import (
            gate_reason_since,
            gate_snapshot,
        )

        PREFETCH_GATE_COUNTS.clear()
        before = gate_snapshot()
        note_prefetch_gate("alloc_failed_post_vote")
        note_prefetch_gate("anchor_pool_exhausted")
        self.assertEqual(gate_reason_since(before), "anchor_pool_exhausted")
        before = gate_snapshot()
        note_prefetch_gate("vote_negative")
        self.assertEqual(gate_reason_since(before), "vote_negative")
        before = gate_snapshot()
        note_prefetch_gate(None)
        self.assertEqual(gate_reason_since(before), "attempted_but_unregistered")
        before = gate_snapshot()
        self.assertEqual(gate_reason_since(before), "unreported")
        PREFETCH_GATE_COUNTS.clear()

    def test_the_symmetric_escape_is_untouched(self):
        """#580: under `symmetric` a locally ineligible rank must still enter
        the collective, or gloo aborts the peer that posted the vote alone."""
        self.assertIn(
            "if not eligible and not symmetric:", TestTheGateIsWiredAndOrdered()._src()
        )


# ---------------------------------------------------------------------------
# #1068 slice 4 (WEG1 spec 4.4): EVERY EXIT BEHIND THE GATE IS NAMED.
#
# The scheduler's own docstring counted SIX silent exits between the #915 gate
# and registration and answered them with an effect-based verdict. Slice 4
# gives each exit a term and ONE line (L1 refusal / L2 truncation), counts the
# scheduler's exits and the intake denominator, and speaks an unnamed exit as
# an ERROR line (L4) -- never a raise on the live intake path (G12).
#
# THE CLASS UNDER TEST IS THE SERVING ONE. The flip boot runs
# `UnifiedRadixCache` attached to `HybridCacheController`; the tree stand-in
# below is a bare instance of that class with only the collaborators
# `prefetch_from_storage` touches replaced, so the exits, counters and lines
# are proven on the method the boot executes, not on a copy.
# ---------------------------------------------------------------------------

import ast
import inspect
import textwrap
import types

TREE_LOGGER = "sglang.srt.mem_cache.unified_radix_cache"
SCHED_LOGGER = "sglang.srt.managers.scheduler"
POOL_ROWS = 366211
POOL_LIMIT = 329589


class _HostPool:
    """Host pool stand-in: an alloc succeeds iff the span fits `available`.
    `fragmented=True` reports room and still refuses every alloc -- the
    `host_alloc_failed` shape (room reported, alloc failed twice)."""

    def __init__(self, available: int, fragmented: bool = False):
        self.available = available
        self.fragmented = fragmented
        self.size = POOL_ROWS
        self.allocs = []

    def alloc(self, n: int):
        self.allocs.append(n)
        if self.fragmented or n > self.available:
            return None
        return list(range(n))

    def available_size(self) -> int:
        return self.available


class _Controller:
    def __init__(self, available: int, fragmented: bool = False):
        self.mem_pool_host = _HostPool(available, fragmented)
        self.prefetch_tokens_occupied = 0
        self.prefetch_capacity_limit = POOL_LIMIT
        self.released = []
        self.ops = []

    def prefetch_rate_limited(self) -> bool:
        return False

    def prefetch(
        self,
        request_id,
        host_indices,
        new_input_tokens,
        last_hash=None,
        prefix_keys=None,
        extra_pools=None,
    ):
        op = types.SimpleNamespace(
            request_id=request_id,
            host_indices=host_indices,
            mark_terminate=lambda: None,
        )
        self.ops.append(op)
        return op

    def append_host_mem_release(self, *args, **kwargs):
        self.released.append((args, kwargs))


class _AnchorlessComponent:
    """A component whose host anchor pool is exhausted: `[]`, the #1035 shape."""

    component_type = "mamba-stand-in"
    _mamba_pool_host = None

    def build_hicache_transfers(self, *args, **kwargs):
        return []


def _node():
    return types.SimpleNamespace(
        key=None,
        backuped=True,
        parent=None,
        get_last_hash_value=lambda: None,
        get_prefix_hash_values=lambda parent: None,
    )


def _serving_tree(
    available: int,
    *,
    fragmented: bool = False,
    symmetric: bool = False,
    vote=None,
    components=(),
    chunk: int = 4096,
):
    """A bare `UnifiedRadixCache` (the serving class) with only the
    collaborators `prefetch_from_storage` touches stubbed. Every method that
    runs is the real one on the real class."""
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    tree = UnifiedRadixCache.__new__(UnifiedRadixCache)
    tree.enable_storage = True
    tree.cache_controller = _Controller(available, fragmented)
    tree.is_eagle = False
    tree.page_size = 1
    tree.prefetch_threshold = 256
    tree.ongoing_prefetch = {}
    tree._retired_prefetch = []
    tree._retired_prefetch_attempts = {}
    tree._retired_prefetch_recompute = 0
    tree._components_tuple = tuple(components)
    tree._hicache_prefetch_symmetric = lambda: symmetric
    tree._all_reduce_attn_groups = vote or (lambda t, op, label="": None)
    tree.inc_host_lock_ref = lambda node: types.SimpleNamespace(
        to_dec_params=lambda: ("dec", node)
    )
    tree.dec_host_lock_ref = lambda node, params: None
    tree.evict_host = lambda n: 0
    tree._build_sidecar_transfers = lambda phase, kv_xfer, comp_xfers: []
    tree._prefetch_chunk_tokens = chunk
    return tree


def _lines(cm, marker: str):
    return [r.getMessage() for r in cm.records if marker in r.getMessage()]


class TestEveryExitBehindTheGateIsNamed(_CleanCounts):
    """T16/T17 and the two exits the spec names beside them."""

    def test_host_pool_exhausted_is_named(self):
        """T16: the pool has no room at all -> refused, counted, ONE line with
        every term, and NOT registered. RED on 228a66db32: the exit was a bare
        `return` and the verdict read 'attempted_but_unregistered'."""
        from sglang.srt.mem_cache.match_refusal_census import (
            gate_reason_since,
            gate_snapshot,
        )

        tree = _serving_tree(available=0)
        before = gate_snapshot()
        with self.assertLogs(TREE_LOGGER, level="WARNING") as cm:
            tree.prefetch_from_storage("rid-exhausted", _node(), list(range(14921)))
        self.assertEqual(PREFETCH_GATE_COUNTS.get("host_pool_exhausted"), 1)
        self.assertEqual(PREFETCH_GATE_COUNTS.get("host_pool_exhausted_tokens"), 14921)
        self.assertEqual(gate_reason_since(before), "host_pool_exhausted")
        lines = _lines(cm, "#915 PREFETCH REFUSED")
        self.assertEqual(len(lines), 1, lines)
        for term in (
            "reason=host_pool_exhausted",
            "rid=rid-exha",
            "need=14921",
            "available=0",
            "threshold=256",
            "occupied=0",
            f"limit={POOL_LIMIT}",
            "pool_id=",
            "epoch=",
            "phase=",
            "generation=",
        ):
            self.assertIn(term, lines[0])
        self.assertNotIn("rid-exhausted", tree.ongoing_prefetch)

    def test_truncation_is_named_with_loss(self):
        """T17: room for 5000 of 39364 -> the span is CUT, counted as
        host_pool_truncated with the lost tokens, spoken with lost= and
        over_bound=, and REGISTERED (it is not a refusal). RED: no line, no
        key."""
        tree = _serving_tree(available=5000)
        with self.assertLogs(TREE_LOGGER, level="WARNING") as cm:
            tree.prefetch_from_storage("rid-truncated", _node(), list(range(39364)))
        self.assertEqual(PREFETCH_GATE_COUNTS.get("host_pool_truncated"), 1)
        self.assertEqual(PREFETCH_GATE_COUNTS.get("host_pool_truncated_tokens"), 34364)
        self.assertEqual(PREFETCH_GATE_COUNTS.get("attempted"), 1)
        lines = _lines(cm, "#915 PREFETCH TRUNCATED")
        self.assertEqual(len(lines), 1, lines)
        for term in (
            "rid=rid-trun",
            "need=39364",
            "got=5000",
            "lost=34364",
            "chunk=4096",
            "over_bound=true",
            "available=5000",
            "pool_id=",
            "phase=",
            "generation=",
        ):
            self.assertIn(term, lines[0])
        self.assertEqual(_lines(cm, "#915 PREFETCH REFUSED"), [])
        self.assertIn("rid-truncated", tree.ongoing_prefetch)
        self.assertEqual(len(tree.ongoing_prefetch["rid-truncated"].prefetch_key), 5000)
        self.assertEqual(tree.cache_controller.prefetch_tokens_occupied, 5000)

    def test_a_truncation_inside_the_chunk_bound_says_so(self):
        tree = _serving_tree(available=39000)
        with self.assertLogs(TREE_LOGGER, level="WARNING") as cm:
            tree.prefetch_from_storage("rid-inbound", _node(), list(range(39364)))
        lines = _lines(cm, "#915 PREFETCH TRUNCATED")
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("lost=364", lines[0])
        self.assertIn("over_bound=false", lines[0])
        self.assertEqual(PREFETCH_GATE_COUNTS.get("host_pool_truncated_tokens"), 364)

    def test_host_alloc_failed_is_named(self):
        """Room reported, alloc refused anyway (a fragmented pool): the second
        exit of the truncation branch, by its own name."""
        from sglang.srt.mem_cache.match_refusal_census import (
            gate_reason_since,
            gate_snapshot,
        )

        tree = _serving_tree(available=5000, fragmented=True)
        before = gate_snapshot()
        with self.assertLogs(TREE_LOGGER, level="WARNING") as cm:
            tree.prefetch_from_storage("rid-fragment", _node(), list(range(39364)))
        self.assertEqual(PREFETCH_GATE_COUNTS.get("host_alloc_failed"), 1)
        self.assertEqual(gate_reason_since(before), "host_alloc_failed")
        lines = _lines(cm, "#915 PREFETCH REFUSED")
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("reason=host_alloc_failed", lines[0])
        self.assertIn("need=39364", lines[0])
        self.assertIn("available=5000", lines[0])
        self.assertNotIn("rid-fragment", tree.ongoing_prefetch)

    def test_the_1035_anchor_exit_counts_the_cause_and_the_exit(self):
        """#1035 (G1): the anchor-pool counter is incremented on EVERY
        occurrence (the WARNING stays rate-limited), and the exit this call
        then leaves through (alloc_failed_post_vote) counts itself; the
        verdict names the CAUSE first."""
        from sglang.srt.mem_cache.match_refusal_census import (
            gate_reason_since,
            gate_snapshot,
        )

        tree = _serving_tree(available=100000, components=(_AnchorlessComponent(),))
        before = gate_snapshot()
        with self.assertLogs(TREE_LOGGER, level="WARNING") as cm:
            tree.prefetch_from_storage("rid-anchor", _node(), list(range(4096)))
        self.assertEqual(PREFETCH_GATE_COUNTS.get("anchor_pool_exhausted"), 1)
        self.assertEqual(PREFETCH_GATE_COUNTS.get("anchor_pool_exhausted_tokens"), 4096)
        self.assertEqual(PREFETCH_GATE_COUNTS.get("alloc_failed_post_vote"), 1)
        self.assertEqual(gate_reason_since(before), "anchor_pool_exhausted")
        lines = _lines(cm, "#915 PREFETCH REFUSED")
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("reason=alloc_failed_post_vote", lines[0])
        self.assertEqual(len(_lines(cm, "#1035 PREFETCH DROPPED")), 1)
        self.assertNotIn("rid-anchor", tree.ongoing_prefetch)
        self.assertEqual(len(tree.cache_controller.released), 1)

    def test_a_negative_vote_is_named(self):
        """#580 symmetric form: a peer lowered the vote; this rank was ready.
        The exit is vote_negative, counted and spoken."""
        from sglang.srt.mem_cache.match_refusal_census import (
            gate_reason_since,
            gate_snapshot,
        )

        def _peer_declines(tensor, op, label=""):
            if label == "prefetch_participation_vote":
                tensor[2] = 0

        tree = _serving_tree(available=100000, symmetric=True, vote=_peer_declines)
        before = gate_snapshot()
        with self.assertLogs(TREE_LOGGER, level="WARNING") as cm:
            tree.prefetch_from_storage(
                "rid-vote", _node(), list(range(4096)), locally_eligible=True
            )
        self.assertEqual(PREFETCH_GATE_COUNTS.get("vote_negative"), 1)
        self.assertEqual(gate_reason_since(before), "vote_negative")
        lines = _lines(cm, "#915 PREFETCH REFUSED")
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("reason=vote_negative", lines[0])
        self.assertNotIn("rid-vote", tree.ongoing_prefetch)
        self.assertEqual(len(tree.cache_controller.released), 1)


class TestTheRatchetEveryReturnBehindTheGateIsNamed(CustomTestCase):
    """T19b, the AST ratchet: a `return` in `prefetch_from_storage` after the
    gate line that does not count a term in its own block is the seventh
    silent exit, and this test refuses it before a boot has to.

    An exit is NAMED when (a) its own block calls `_note_prefetch_gate(`
    before the return, or (b) it is the gate's own exit: the `if` that holds
    it is directly preceded by the gate line `_note_prefetch_gate(reason, ...)`
    (a second note there would double-count the gate's term). RED on
    228a66db32: four returns without a term (the truncation `else`, the
    post-truncation alloc failure, the negative vote, the post-vote alloc
    failure)."""

    @staticmethod
    def _is_note(node) -> bool:
        """A DIRECT statement `_note_prefetch_gate(...)`, never a note buried
        in a nested block of an earlier sibling. Slice 4 fix (review,
        non-blocking): the first form walked `ast.walk` over the siblings, so
        a bare `return` closing an `if` block counted as named whenever an
        earlier sibling's INNER block carried a note -- the nested case in
        `test_the_ratchet_can_fail` is that gap, red on af399f19c1."""
        return (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_note_prefetch_gate"
        )

    @classmethod
    def _gate_line(cls, fn) -> int:
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_note_prefetch_gate"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "reason"
            ):
                return node.lineno
        raise AssertionError("the #915 gate line `_note_prefetch_gate(reason, ...)` is gone")

    @classmethod
    def unnamed_returns(cls, src: str):
        fn = ast.parse(textwrap.dedent(src)).body[0]
        gate = cls._gate_line(fn)
        unnamed, after_gate = [], 0

        def walk(body, chain):
            nonlocal after_gate
            for i, stmt in enumerate(body):
                if isinstance(stmt, ast.Return) and stmt.lineno > gate:
                    after_gate += 1
                    named = any(cls._is_note(s) for s in body[:i])
                    if not named and chain:
                        pbody, pidx = chain[-1]
                        prev = pbody[pidx - 1] if pidx > 0 else None
                        named = (
                            prev is not None
                            and prev.lineno == gate
                            and cls._is_note(prev)
                        )
                    if not named:
                        unnamed.append(stmt.lineno)
                for field in ("body", "orelse", "finalbody"):
                    sub = getattr(stmt, field, None)
                    if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                        walk(sub, chain + [(body, i)])
                for handler in getattr(stmt, "handlers", []) or []:
                    walk(handler.body, chain + [(body, i)])

        walk(fn.body, [])
        return unnamed, after_gate

    def test_the_ratchet_can_fail(self):
        """DESK-WRITTEN-NEVER-EXECUTED guard: the checker must catch a bare
        return and accept the gate's own exit."""
        bad = """
def f(reason, x):
    _note_prefetch_gate(reason, 1)
    if x:
        return
    if x > 1:
        return
    _note_prefetch_gate("named", 1)
    return
"""
        unnamed, n = self.unnamed_returns(bad)
        self.assertEqual(n, 3)
        self.assertEqual(unnamed, [7], "the bare return on line 7 must be caught")
        # The nested shape: a note INSIDE an earlier sibling's block names
        # nothing outside that block. RED on af399f19c1 (walked as named).
        nested = """
def f(reason, x):
    _note_prefetch_gate(reason, 1)
    y = x
    if y:
        if y > 1:
            _note_prefetch_gate("inner", 1)
        return
"""
        unnamed, n = self.unnamed_returns(nested)
        self.assertEqual(n, 1)
        self.assertEqual(
            unnamed,
            [8],
            "a note inside an earlier sibling's block must not name line 8",
        )

    def test_every_return_behind_the_gate_counts_a_term(self):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        src = inspect.getsource(UnifiedRadixCache.prefetch_from_storage)
        unnamed, n = self.unnamed_returns(src)
        self.assertGreaterEqual(n, 5, "the exits behind the gate are gone?")
        self.assertEqual(
            unnamed,
            [],
            f"return(s) at function-relative line(s) {unnamed} of "
            "prefetch_from_storage count no #915 term: a silent exit",
        )


class _TreeCache:
    """Scheduler-side tree stand-in whose only honest signal is the ongoing
    set; `gate_reason` makes it record a #915 verdict the way the real gate
    does, `register` makes it register."""

    def __init__(self, *, register=False, gate_reason="absent", ongoing=True, probe=None):
        if ongoing:
            self.ongoing_prefetch = {}
        self._register = register
        self._gate_reason = gate_reason
        self.calls = []
        self.hicache_storage_pass_prefix_keys = False
        self.root_node = object()
        if probe is not None:
            self.cache_controller = types.SimpleNamespace(store_presence_pages=probe)

    def prefetch_from_storage(self, req_id, *a, **kw):
        self.calls.append(req_id)
        if self._gate_reason != "absent":
            note_prefetch_gate(self._gate_reason, 4096)
        if self._register:
            self.ongoing_prefetch[req_id] = object()


def _sched(tree, enable=True):
    from sglang.srt.managers.scheduler import Scheduler

    s = Scheduler.__new__(Scheduler)
    s.enable_hicache_storage = enable
    s.tree_cache = tree
    return s


def _req(rid: str, backuped: bool = True):
    return types.SimpleNamespace(
        rid=rid,
        init_next_round_input=lambda *a, **kw: None,
        last_host_node=types.SimpleNamespace(
            backuped=backuped,
            parent=None,
            get_last_hash_value=lambda: "h",
            get_prefix_hash_values=lambda parent: None,
        ),
        prefix_indices=[],
        host_hit_length=0,
        full_untruncated_fill_ids=list(range(600)),
        _compute_max_prefix_len=lambda n: n - 1,
    )


class TestTheIntakePartitionSums(_CleanCounts):
    """T18: every entry into `_prefetch_kvcache` is one `intake`, and every
    exit counts exactly one partition term, so intake == sum(partition) at
    every instant. RED on 228a66db32: the scheduler exits counted nothing."""

    def test_intake_partition_sums(self):
        from sglang.srt.mem_cache.match_refusal_census import (
            PREFETCH_INTAKE_PARTITION,
        )

        verdicts = []
        # storage_disabled
        verdicts.append(_sched(_TreeCache(), enable=False)._prefetch_kvcache(_req("r1")))
        # store_absent: not locally eligible, the store says no
        verdicts.append(
            _sched(_TreeCache(probe=lambda *a: False))._prefetch_kvcache(
                _req("r2", backuped=False)
            )
        )
        # anchor_no_vote: not locally eligible, no store to ask
        verdicts.append(_sched(_TreeCache())._prefetch_kvcache(_req("r3", backuped=False)))
        # unobservable: a tree without an ongoing set
        verdicts.append(_sched(_TreeCache(ongoing=False))._prefetch_kvcache(_req("r4")))
        # already_in_flight
        t = _TreeCache(register=True, gate_reason=None)
        t.ongoing_prefetch["r5"] = object()
        verdicts.append(_sched(t)._prefetch_kvcache(_req("r5")))
        # issued
        verdicts.append(
            _sched(_TreeCache(register=True, gate_reason=None))._prefetch_kvcache(_req("r6"))
        )
        # attempted_but_unregistered: the gate admitted, nothing registered
        verdicts.append(_sched(_TreeCache(gate_reason=None))._prefetch_kvcache(_req("r7")))
        # a named tree decline (rate_limited)
        verdicts.append(
            _sched(_TreeCache(gate_reason="rate_limited"))._prefetch_kvcache(_req("r8"))
        )
        # unreported: the tree recorded no verdict at all
        verdicts.append(_sched(_TreeCache())._prefetch_kvcache(_req("r9")))

        self.assertEqual(
            verdicts,
            [
                "declined:storage_disabled",
                "declined:store_absent",
                "declined:anchor_no_vote",
                "declined:unobservable",
                "declined:already_in_flight",
                "issued",
                "declined:attempted_but_unregistered",
                "declined:rate_limited",
                "declined:unreported",
            ],
        )
        self.assertEqual(PREFETCH_GATE_COUNTS.get("intake"), 9)
        for key in (
            "storage_disabled",
            "store_absent",
            "anchor_no_vote",
            "unobservable",
            "already_in_flight",
            "issued",
            "attempted_but_unregistered",
            "rate_limited",
            "unreported",
        ):
            self.assertEqual(PREFETCH_GATE_COUNTS.get(key), 1, key)
        self.assertEqual(
            PREFETCH_GATE_COUNTS["intake"],
            sum(PREFETCH_GATE_COUNTS.get(k, 0) for k in PREFETCH_INTAKE_PARTITION),
            f"intake != sum of the partition: {sorted(PREFETCH_GATE_COUNTS.items())}",
        )
        self.assertNotIn("anchor_pool_exhausted", PREFETCH_INTAKE_PARTITION)
        self.assertNotIn("host_pool_truncated", PREFETCH_INTAKE_PARTITION)


class TestAnUnnamedExitIsAnErrorLineNotARaise(_CleanCounts):
    """T19: the verdict 'attempted_but_unregistered' (the gate admitted, no
    named exit counted, nothing registered) is spoken as L4 -- logger.error,
    rate-limited with its count printed -- and NEVER raised (G12: this sits on
    every intake). RED on 228a66db32: no line, no key."""

    def test_an_unnamed_exit_is_an_error_line_not_a_raise(self):
        sched = _sched(_TreeCache(gate_reason=None))
        with self.assertLogs(SCHED_LOGGER, level="ERROR") as cm:
            verdict = sched._prefetch_kvcache(_req("rid-seventh-exit"))
        self.assertEqual(verdict, "declined:attempted_but_unregistered")
        lines = _lines(cm, "#915 PREFETCH UNREGISTERED")
        self.assertEqual(len(lines), 1, lines)
        for term in (
            "rid=rid-seve",
            "phase=",
            "generation=",
            "n=1",
            "verdict=attempted_but_unregistered",
            "seventh silent exit",
        ):
            self.assertIn(term, lines[0])
        self.assertEqual(cm.records[0].levelno, logging.ERROR)
        self.assertEqual(PREFETCH_GATE_COUNTS.get("attempted_but_unregistered"), 1)

    def test_the_error_line_is_rate_limited_and_prints_its_count(self):
        sched = _sched(_TreeCache(gate_reason=None))
        with self.assertLogs(SCHED_LOGGER, level="ERROR") as cm:
            for i in range(41):
                sched._prefetch_kvcache(_req(f"rid-{i}"))
        lines = _lines(cm, "#915 PREFETCH UNREGISTERED")
        self.assertEqual(len(lines), 40, "n<=40 spoken, the 41st suppressed")
        self.assertIn("n=40", lines[-1])
        self.assertEqual(PREFETCH_GATE_COUNTS.get("attempted_but_unregistered"), 41)
        self.assertEqual(PREFETCH_GATE_COUNTS.get("intake"), 41)



# ---------------------------------------------------------------------------
# #1068 slice 4 fix (spec A11.3): T20, bs > pool spans lands via evict_host.
# ---------------------------------------------------------------------------

SPAN = 4096


class _SpanPool(_HostPool):
    """T20 (spec A11.3): room is CONSUMED by an alloc and RETURNED only by
    `evict`, and `evict` frees the rows of COMPLETED spans only. That is the
    tree's lock law in one stub: a span in flight holds the host lock its
    registration took (`inc_host_lock_ref(last_host_node)`) until
    `check_prefetch_progress` drops it after PREFETCH-COMPLETE
    (unified_radix_cache.py: `dec_host_lock_ref(last_host_node,
    anchor_lock_params)` right after the two generation-stamped frees), and
    `evict_host` walks only unlocked host leaves."""

    def __init__(self, spans: int):
        super().__init__(available=spans * SPAN)
        self.live = []
        self.completed = set()
        self.evicts = []

    def alloc(self, n: int):
        self.allocs.append(n)
        if n > self.available:
            return None
        self.available -= n
        self.live.append(n)
        return list(range(n))

    def complete(self, i: int) -> None:
        self.completed.add(i)

    def evict(self, need: int) -> int:
        self.evicts.append(need)
        freed = 0
        for i, n in enumerate(self.live):
            if freed >= need:
                break
            if i in self.completed and n > 0:
                freed += n
                self.available += n
                self.live[i] = 0
        return freed


class TestExcessRequestsLandAfterEvictHost(_CleanCounts):
    """T20 (spec A11.3, slice 4): bs > pool spans. The (k+1)-th issued span
    lands WITHOUT truncation once an earlier span has completed: the
    truncation retry in `prefetch_from_storage` calls `evict_host(need)`
    before its second alloc, and a completed span's host rows are evictable
    because the tree drops its lock after PREFETCH-COMPLETE.

    CHARACTERISATION PIN, not red-first (spec A11.3 says so for this case);
    both halves exist on the parent 228a66db32. WHAT IT COVERS: the retry
    half only -- `prefetch_from_storage` calls `evict_host(need)` between a
    failed alloc and its second alloc, with exactly `need` (the alloc/evict
    call shapes asserted below). A later change that drops the evict before
    the retry, or truncates instead of retrying, fails here.
    WHAT IT DOES NOT COVER: the lock-release half. `_serving_tree` stubs
    `inc_host_lock_ref` / `dec_host_lock_ref` as lambdas,
    `check_prefetch_progress` never runs, and `pool.complete(0)` is the
    test's own hand-simulation of "r1's rows became evictable" -- so the
    release of the anchor lock after PREFETCH-COMPLETE
    (unified_radix_cache.py `check_prefetch_progress`,
    `dec_host_lock_ref(last_host_node, anchor_lock_params)` right after the
    two generation-stamped frees) is NOT exercised here, and a change that
    keeps loaded rows locked beyond the load PASSES this pin. The control
    case shows the pin can fail (desk-written-never-executed)."""

    def _tree_with(self, pool: _SpanPool):
        tree = _serving_tree(available=0)
        tree.cache_controller.mem_pool_host = pool
        tree.evict_host = pool.evict
        return tree

    def test_excess_requests_land_after_evict_host_not_truncated(self):
        pool = _SpanPool(spans=2)
        tree = self._tree_with(pool)
        tree.prefetch_from_storage("r1", _node(), list(range(SPAN)))
        tree.prefetch_from_storage("r2", _node(), list(range(SPAN)))
        self.assertEqual(pool.available, 0)
        pool.complete(0)  # r1 completed: its rows are no longer lock-protected
        tree.prefetch_from_storage("r3", _node(), list(range(SPAN)))
        for rid in ("r1", "r2", "r3"):
            self.assertIn(rid, tree.ongoing_prefetch)
        self.assertEqual(len(tree.ongoing_prefetch["r3"].prefetch_key), SPAN)
        self.assertEqual(PREFETCH_GATE_COUNTS.get("host_pool_truncated", 0), 0)
        self.assertEqual(PREFETCH_GATE_COUNTS.get("host_pool_exhausted", 0), 0)
        self.assertEqual(PREFETCH_GATE_COUNTS.get("attempted"), 3)
        # the third span: one failed alloc, ONE evict of exactly `need`, one
        # successful alloc -- the retry shape, not a truncation
        self.assertEqual(pool.allocs, [SPAN, SPAN, SPAN, SPAN])
        self.assertEqual(pool.evicts, [SPAN])
        self.assertEqual(tree.cache_controller.prefetch_tokens_occupied, 3 * SPAN)

    def test_an_uncompleted_span_stays_locked_and_the_excess_request_is_refused(self):
        """Control: with NO completed span nothing is evictable, so the
        third request is refused BY NAME (host_pool_exhausted: 0 rows of
        room is under the threshold), never silently and never registered."""
        pool = _SpanPool(spans=2)
        tree = self._tree_with(pool)
        tree.prefetch_from_storage("r1", _node(), list(range(SPAN)))
        tree.prefetch_from_storage("r2", _node(), list(range(SPAN)))
        with self.assertLogs(TREE_LOGGER, level="WARNING") as cm:
            tree.prefetch_from_storage("r3", _node(), list(range(SPAN)))
        self.assertNotIn("r3", tree.ongoing_prefetch)
        self.assertEqual(pool.evicts, [SPAN])
        self.assertEqual(PREFETCH_GATE_COUNTS.get("host_pool_exhausted"), 1)
        self.assertEqual(len(_lines(cm, "#915 PREFETCH REFUSED")), 1)
        self.assertEqual(PREFETCH_GATE_COUNTS.get("host_pool_truncated", 0), 0)

if __name__ == "__main__":
    unittest.main()
