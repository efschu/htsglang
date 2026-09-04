"""#1203 family A -- THE GROUP MUST NOT BRANCH ON A NUMBER ONE RANK COMPUTED.

Same root as #1158 (queue axis) and #1176 (witness axis), on three further
consumers. All three are evaluated in the TP phase, where `tp_cpu_group` has
world N and every rank builds its own batch, so a rank-local input decides a
branch whose two sides carry DIFFERENT COLLECTIVES.

A1  `phase_purity.seam_transport_premise_holds` is the crown consumer. Its
    read set is rank-local end to end: `store_witness` reads
    `prefetch_loaded_tokens_by_reqid`, whose reduced half exists only under
    `UnifiedRadixCache.tp_world_size > 1` -- and that field is written once at
    construction (`unified_radix_cache.py:522`) and never rebound at the
    cutover, so on the shipping form (`--tp-size 1 --pp-size 3`) it is 1 in
    BOTH phases while `scheduler.tp_cpu_group` is rebound to world 3
    (`phase_flip_runtime.py:3201-3202`). The divergent input is on this rig's
    metal: boot_855_weg1b9_1116175f6d_0904_164023 log:1900 (PP0 absent=67)
    against :1977 / :2067 (PP1/PP2 assembling=67) -- same 67 stems, same
    second, two different answers about presence. Outcome split:
    premise True -> a transport extend batch is built; premise False ->
    SEAM TRANSPORT REFUSED and, under strict:3 + drain_mode, `new_batch=None`.

A4  `unified_radix_cache.check_prefetch_progress` publishes the TRANSFER
    (`completed_tokens`) as this rank's completion, and PP0 MINs those into
    the group floor (`pp_prefetch_completion.group_completion_verdict`). A
    rank whose insert DECLINED the fetched tail retains less than it
    transferred, so the floor over-reports -- the corrupting direction, since
    the floor licenses a told prefix.

A5  three defer bounds in `phase_flip_runtime._execute_body` are counted on
    RANK-LOCAL counters and spent against a GROUP-UNANIMOUS abandon. Three
    ranks taking turns objecting never spend a budget between them: the
    411-abandon decode wedge, reached through the mechanism that exists to
    prevent it. The correct currency is written twelve hundred lines up
    (`_seam_abandons_in_a_row`, booked from the already-reduced fit verdict)
    and was applied to one of four siblings.
"""

import ast
import inspect
import types
import unittest

import torch

from sglang.srt.managers import phase_flip_runtime, phase_purity
from sglang.srt.managers.phase_purity import (
    SEAM_GRANT_CONSUMED_ATTR,
    SEAM_READMIT_ATTR,
    seam_transport_premise_holds,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache import unified_radix_cache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

STAMP = 6_008


def _req(rid="a1", *, stamp=STAMP, tokens=8192, seam_epoch=3):
    r = types.SimpleNamespace(
        rid=rid,
        cached_prompt_tokens_at_retract=stamp,
        cache_protected_len=0,
        origin_input_ids=list(range(tokens)),
        prefix_indices=torch.arange(0, dtype=torch.int64),
        host_hit_length=0,
        storage_hit_length=0,
    )
    setattr(r, SEAM_READMIT_ATTR, seam_epoch)
    setattr(r, SEAM_GRANT_CONSUMED_ATTR, False)
    return r


def _sched(reqs, *, outcomes=None, pending=(), group=None, phase="tp"):
    """The premise's whole read surface, plus the group slot under test.

    `root_node.children` non-empty and a host pool with a used row keep
    `seam_store_presence_refuted` from refuting, so the ONLY thing deciding
    the premise here is the witness -- which is the term A1 is about.
    """
    pool = types.SimpleNamespace(size=100)
    pool.available_size = lambda: 50
    tree = types.SimpleNamespace(
        root_node=types.SimpleNamespace(children={"x": object()}),
        cache_controller=types.SimpleNamespace(mem_pool_host=pool),
        enable_storage=True,
        ongoing_prefetch={rid: object() for rid in pending},
        prefetch_loaded_tokens_by_reqid=dict(outcomes or {}),
        prefetch_threshold=256,
        _prefetch_chunk_tokens=4096,
    )
    s = types.SimpleNamespace(
        tree_cache=tree,
        waiting_queue=list(reqs),
        phase_flip_runtime=types.SimpleNamespace(phase=phase),
        ps=types.SimpleNamespace(pp_rank=0, pp_size=3, tp_rank=0),
    )
    if group is not None:
        setattr(s, phase_purity.UNIFORM_SEAM_PREMISE_ATTR, group)
    return s


#: A request the local rank witnesses as restored: a registered, pending
#: prefetch is a "pending" witness, which the premise counts.
def _restored_locally():
    return _sched([_req()], pending=("a1",))


class A1_TheSeamPremiseIsAGroupFact(CustomTestCase):
    def test_the_local_premise_holds_on_a_witnessed_restore(self):
        """The fixture itself: without a group verdict this rank says yes.

        Can-fail floor for the two tests below -- if this were False they
        would pass for the wrong reason."""
        self.assertTrue(seam_transport_premise_holds(_restored_locally()))

    def test_a_peer_without_a_restore_refuses_for_the_whole_group(self):
        """THE CUT. This rank's witness says restore; the reduced AND says a
        peer's did not. The group must take the peer's answer, or the ranks
        enter different collectives -- log:1900 vs :1977, on metal."""
        s = _sched([_req()], pending=("a1",), group=0)
        self.assertFalse(
            seam_transport_premise_holds(s),
            "a rank whose peer saw no restore must not build a transport "
            "batch its peers will not build",
        )

    def test_a_unanimous_group_leaves_the_local_answer_alone(self):
        s = _sched([_req()], pending=("a1",), group=1)
        self.assertTrue(seam_transport_premise_holds(s))

    def test_an_unpublished_verdict_is_byte_identical_to_today(self):
        """No reduce ran this pass (PP loop, single-rank group, offload
        branch): the local answer stands, exactly as before the cut."""
        self.assertTrue(seam_transport_premise_holds(_restored_locally()))
        self.assertFalse(seam_transport_premise_holds(_sched([])))

    def test_a_group_yes_can_never_override_a_local_no(self):
        """THE DANGEROUS DIRECTION, pinned. A stale or wrong 1 must not
        license a rank that has no restore of its own: the gate is
        local AND group, never group alone."""
        s = _sched([_req(stamp=0)], group=1)
        self.assertFalse(seam_transport_premise_holds(s))


class A1_ThePayloadCarriesTheVote(CustomTestCase):
    """The vote rides the reduce that already runs, in its own head-indexed
    slot ahead of the tail-indexed ballot."""

    def _run(self, vote, peer_vote):
        payloads = []

        class _CaptureDist:
            ReduceOp = torch.distributed.ReduceOp

            @staticmethod
            def get_world_size(group):
                return 2

            @staticmethod
            def all_reduce(t, op=None, group=None):
                payloads.append(t.clone())

        class _ReplayDist(_CaptureDist):
            @staticmethod
            def all_reduce(t, op=None, group=None):
                t.copy_(torch.minimum(payloads[0], payloads[1]))

        from unittest import mock

        for v in (vote, peer_vote):
            s = _fake_reduce_scheduler(v)
            with mock.patch.object(torch, "distributed", _CaptureDist):
                s._update_uniform_pool_budget()
        self.assertEqual(payloads[0].numel(), payloads[1].numel())
        s = _fake_reduce_scheduler(vote)
        with mock.patch.object(torch, "distributed", _ReplayDist):
            s._update_uniform_pool_budget()
        return s

    def test_one_peer_voting_no_pulls_the_group_to_no(self):
        s = self._run(1, 0)
        self.assertEqual(getattr(s, phase_purity.UNIFORM_SEAM_PREMISE_ATTR), 0)

    def test_a_unanimous_yes_survives_the_reduce(self):
        s = self._run(1, 1)
        self.assertEqual(getattr(s, phase_purity.UNIFORM_SEAM_PREMISE_ATTR), 1)

    def test_a_single_rank_group_publishes_nothing(self):
        """No reduce, so no group verdict -- cleared rather than left
        standing, the discipline #823 W9b wrote for `_uniform_head_inputs`."""
        s = _fake_reduce_scheduler(1)
        s.tp_cpu_group = None
        s._update_uniform_pool_budget()
        self.assertIsNone(getattr(s, phase_purity.UNIFORM_SEAM_PREMISE_ATTR))


class A4_TheGroupFloorMeasuresRetentionNotTransfer(CustomTestCase):
    """AST, not execution: the write site sits deep inside
    `check_prefetch_progress`, which wants a cache controller, a host pool and
    a live storage backend. The assignment's RHS is the whole cut, so its
    identifiers are what this pins."""

    def _rhs(self):
        tree = ast.parse(inspect.getsource(unified_radix_cache))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            tgt = node.targets[0]
            if (
                isinstance(tgt, ast.Subscript)
                and isinstance(tgt.value, ast.Attribute)
                and tgt.value.attr == "_prefetch_completed_tokens"
            ):
                return node.value
        self.fail("the completion write site is gone; A4 needs re-anchoring")

    def test_the_published_completion_is_the_retained_prefix(self):
        names = {
            n.attr if isinstance(n, ast.Attribute) else n.id
            for n in ast.walk(self._rhs())
            if isinstance(n, (ast.Attribute, ast.Name))
        }
        self.assertIn("prefix_len", names)
        self.assertIn("loaded_from_storage", names)

    def test_the_transfer_count_is_not_what_is_published(self):
        """The corrupting direction: `completed_tokens` counts bytes moved,
        and a declined insert retains none of them, so a floor built from it
        licenses a told prefix no rank holds."""
        names = {
            n.id for n in ast.walk(self._rhs()) if isinstance(n, ast.Name)
        }
        self.assertNotIn("completed_tokens", names)
        self.assertNotIn("min_completed_tokens", names)


class A1_TheVoteIsPure(CustomTestCase):
    """REPAIR: the group vote runs on a cadence the instrument was not written
    for.

    ``_update_uniform_pool_budget`` builds its payload once per loop
    iteration, unconditionally. The CONSUMER of this premise returns at
    ``_active_phase(scheduler) != PHASE_TP`` before it asks. So a vote that
    announces drives the edge-triggered ``_seam_premise_refused_announced``
    latch from iterations the gate never reaches -- swallowing the gate's own
    first engagement in TP and emitting ``SEAM TRANSPORT REFUSED`` where no
    transport decision was being made.
    """

    def _sched_that_refuses(self):
        """A queued candidate with NO restore witness: the refusing branch,
        the one that announces."""
        return _sched([_req(stamp=0)])

    def test_the_vote_leaves_the_announcement_latch_alone(self):
        sched = self._sched_that_refuses()
        sched._seam_premise_refused_announced = False
        phase_purity.local_seam_premise_vote(sched)
        self.assertFalse(
            getattr(sched, "_seam_premise_refused_announced", False),
            "the vote latched the gate's announcement from an iteration the "
            "gate itself never reaches",
        )

    def test_the_vote_does_not_clear_a_latch_the_gate_set(self):
        sched = self._sched_that_refuses()
        sched._seam_premise_refused_announced = True
        phase_purity.local_seam_premise_vote(sched)
        self.assertTrue(
            getattr(sched, "_seam_premise_refused_announced", False),
            "the vote cleared a latch it did not set, so the gate's next real "
            "engagement would be announced as if it were the first",
        )

    def test_the_gate_still_announces(self):
        """Can-fail anchor: `announce=False` must be the VOTE's choice, not a
        new default that silences the consumer too."""
        sched = self._sched_that_refuses()
        sched._seam_premise_refused_announced = False
        phase_purity.seam_transport_premise_holds_locally(sched)
        self.assertTrue(
            getattr(sched, "_seam_premise_refused_announced", False),
            "the gate's own read must still announce exactly once per edge",
        )


class A5_TheDeferBudgetIsSpentByItsOwnCause(CustomTestCase):
    """REPAIR of #1203 A5. The premise was right, the currency was wrong.

    A5's premise stands and is not re-litigated here: a rank-local counter
    that ZEROES ITSELF the moment this rank's own objection clears can never
    reach its limit while three ranks take turns objecting, so the direction
    defers for ever -- the 411-abandon decode wedge reached through the
    mechanism that exists to prevent it.

    WHAT A5 SHIPPED INSTEAD WAS NOT A BOUND, IT WAS AN ESCALATION GATE.
    ``flip_host_headroom_verdict`` and ``flip_seam_budget_verdict`` do not
    merely stop deferring at the limit: they return ``(allow=True,
    escalated=True)`` -- "PROCEEDING WITH EYES OPEN" -- and the writeback arm
    proceeds with an incomplete #703 fence. Spending that budget in
    ``_seam_abandons_in_a_row``, which is incremented on EVERY reduced-fit
    abandon or frame divergence whatever caused it, means three unrelated
    abandons disarm all three guards: the next genuine host-RAM shortfall
    escalates past the #721 floor on its FIRST firing, having deferred zero
    times. That converts a refusal into the kernel-OOM kill #721 exists to
    defend against -- the corrupting direction, bought to avoid a wedge.

    THE REPAIR KEEPS BOTH. Each guard is bounded by ITS OWN consecutive
    objection count again, and that count is no longer zeroed when this rank's
    own objection clears -- only when the flip COMPLETES (the group's own
    reset point, whose comment already states the rule: "a rank that cleared
    while a peer did not has learnt nothing about the group") or when the
    guard's own escalation has spent it. Ranks taking turns therefore still
    accumulate to the limit and still escalate; an unrelated abandon still
    spends nothing.
    """

    BOOK = "self._seam_abandons_in_a_row.get(direction, 0)"

    def _body(self):
        tree = ast.parse(inspect.getsource(phase_flip_runtime))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_execute_body":
                return node
        self.fail("_execute_body is gone; A5 needs re-anchoring")

    def _call_arg(self, fn_name, index):
        for node in ast.walk(self._body()):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == fn_name
            ):
                return ast.unparse(node.args[index])
        self.fail(f"{fn_name} is not called in _execute_body any more")

    def test_the_host_headroom_bound_reads_its_own_defer_count(self):
        arg = self._call_arg("flip_host_headroom_verdict", 2)
        self.assertNotEqual(
            arg,
            self.BOOK,
            "the #721 host-RAM guard escalates at its limit; spending that "
            "limit on unrelated abandons makes the first genuine shortfall "
            "proceed into the OOM the guard exists to defend against",
        )
        self.assertIn(
            f"{arg} = int(getattr(self, '_host_ram_defers', 0) or 0)",
            ast.unparse(self._body()),
            "the bound reads a local that is not this guard's own counter",
        )

    def test_the_seam_budget_bound_reads_its_own_defer_count(self):
        arg = self._call_arg("flip_seam_budget_verdict", 1)
        self.assertNotEqual(arg, self.BOOK)
        self.assertIn(
            f"{arg} = int(getattr(self, '_seam_budget_defers', 0) or 0)",
            ast.unparse(self._body()),
            "the bound reads a local that is not this guard's own counter",
        )

    def test_the_writeback_bound_reads_its_own_defer_count(self):
        found = []
        for node in ast.walk(self._body()):
            if isinstance(node, ast.Compare) and any(
                isinstance(c, ast.Name) and c.id == "_WRITEBACK_DEFER_LIMIT"
                for c in node.comparators
            ):
                found.append(ast.unparse(node.left))
        self.assertTrue(found, "the writeback defer bound is gone")
        for left in found:
            self.assertNotEqual(
                left,
                self.BOOK,
                "past three unrelated abandons the FIRST writeback shortfall "
                "would proceed with an incomplete #703 fence",
            )
            self.assertIn(
                f"{left} = int(getattr(self, '_writeback_defers', 0) or 0)",
                ast.unparse(self._body()),
                "the bound reads a local that is not this guard's own counter",
            )

    def test_the_correct_form_is_still_where_it_was_copied_from(self):
        """Can-fail anchor: the seam-margin term is the one place where the
        group's abandon book IS the right currency (it bounds the abandon
        itself, not an escalation past a physical floor)."""
        src = inspect.getsource(phase_flip_runtime)
        self.assertIn("spent = " + self.BOOK, src)


class A5_ADeferBudgetIsNotZeroedByThisRankClearing(CustomTestCase):
    """The half of A5's premise that must survive the repair.

    ``flip_defer_budget_after`` is the whole reset policy, extracted so it can
    be exercised without a scheduler. Three ranks taking turns objecting must
    still accumulate; an unrelated abandon must still spend nothing.
    """

    def _after(self, **kw):
        return phase_flip_runtime.flip_defer_budget_after(**kw)

    def test_an_objection_spends_one(self):
        self.assertEqual(self._after(objected=True, escalated=False, prior=0), 1)
        self.assertEqual(self._after(objected=True, escalated=False, prior=2), 3)

    def test_this_rank_clearing_does_not_refund_the_budget(self):
        """THE A5 DEFECT, in one line. If this returns 0, three ranks taking
        turns objecting never reach the limit and the direction defers for
        ever -- which is what made A5 reach for the group's currency."""
        self.assertEqual(
            self._after(objected=False, escalated=False, prior=2),
            2,
            "a rank that cleared while a peer did not has learnt nothing "
            "about the group, so it may not refund its own budget",
        )

    def test_an_escalation_spends_the_whole_budget(self):
        self.assertEqual(self._after(objected=False, escalated=True, prior=3), 0)

    def test_a_taking_turns_run_still_reaches_the_limit(self):
        """Rank A objects on every other abandon; the budget still fills."""
        n = 0
        for lap in range(6):
            n = self._after(objected=(lap % 2 == 0), escalated=False, prior=n)
        self.assertGreaterEqual(
            n,
            phase_flip_runtime.FLIP_HOST_RAM_MAX_DEFERS,
            "the guard would never escalate and the direction would wedge",
        )

    def test_an_unrelated_abandon_run_spends_nothing(self):
        """THE REPAIRED PROPERTY. Six abandons this guard did not cause leave
        its budget whole, so the next genuine shortfall DEFERS."""
        n = 0
        for _ in range(6):
            n = self._after(objected=False, escalated=False, prior=n)
        self.assertEqual(n, 0)
        allow, escalated, _ = phase_flip_runtime.flip_host_headroom_verdict(0, 0, n)
        self.assertFalse(allow)
        self.assertFalse(escalated)

    def test_a_completed_flip_returns_every_budget_beside_the_abandon_book(self):
        """The group's own reset point clears all three, and that is the ONLY
        legitimate refund short of a guard's own escalation.

        Pinned by ADJACENCY to `self._seam_abandons_in_a_row[direction] = 0`,
        because the argument for refunding there is the argument already
        written at that line -- "a rank that cleared while a peer did not has
        learnt nothing about the group". A refund anywhere else is the defect.
        """
        src = inspect.getsource(phase_flip_runtime).splitlines()
        book = [
            i
            for i, ln in enumerate(src)
            if ln.strip() == "self._seam_abandons_in_a_row[direction] = 0"
        ]
        self.assertTrue(book, "the abandon book's own reset point is gone")
        for counter in (
            "_writeback_defers",
            "_host_ram_defers",
            "_seam_budget_defers",
        ):
            zeroes = [
                i
                for i, ln in enumerate(src)
                if ln.strip() == f"self.{counter} = 0"
            ]
            self.assertTrue(zeroes, f"{counter} is never refunded; it will wedge")
            for z in zeroes:
                self.assertTrue(
                    any(abs(z - b) <= 12 for b in book),
                    f"phase_flip_runtime.py:{z + 1} refunds {counter} away "
                    "from the completed-flip reset point -- a rank refunding "
                    "its own budget on a cleared vote is #1203 A5's wedge",
                )


def _fake_reduce_scheduler(vote):
    """Only the surface `_update_uniform_pool_budget` touches, with the
    per-rank local votes stubbed so the payload width is fixed and the ONLY
    slot that varies between the two fixture ranks is the seam vote."""
    class _S:
        _update_uniform_pool_budget = Scheduler._update_uniform_pool_budget
        _publish_uniform_evict_floor = Scheduler._publish_uniform_evict_floor
        _publish_uniform_host_floor = Scheduler._publish_uniform_host_floor
        _publish_uniform_mamba_floor = Scheduler._publish_uniform_mamba_floor
        _HOST_AVAIL_ABSENT = Scheduler._HOST_AVAIL_ABSENT
        _MAMBA_AVAIL_ABSENT = Scheduler._MAMBA_AVAIL_ABSENT

        def __init__(self, vote):
            self.kv_session_offload = None
            self.token_to_kv_pool_allocator = types.SimpleNamespace(
                available_size=lambda: 1000
            )
            self.tree_cache = None
            self.tp_cpu_group = object()
            self.server_args = types.SimpleNamespace(dcp_size=1)
            self.waiting_queue = []
            self.ps = types.SimpleNamespace(tp_rank=0, tp_size=2, pp_size=1)
            self._vote = vote
            self._uniform_floor_scope_reported = None

        def _local_host_avail(self):
            return Scheduler._HOST_AVAIL_ABSENT

        def _local_mamba_avail(self):
            return Scheduler._MAMBA_AVAIL_ABSENT

        def _local_corridor_width_ceiling(self):
            return 0

        def _local_head_prefix_matches(self):
            return [], {}

        def _local_admit_limit(self):
            return 0

        def _drain_prefetch_progress(self):
            return {}

        def _local_seam_premise_vote(self):
            return self._vote

    return _S(vote)


if __name__ == "__main__":
    unittest.main()
