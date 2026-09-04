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


class A5_TheDeferBudgetIsSpentInGroupCurrency(CustomTestCase):
    """The three bounds must read the reduced abandon book, exactly as the
    seam-margin term already does."""

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

    def test_the_host_headroom_bound_reads_the_group_book(self):
        self.assertEqual(self._call_arg("flip_host_headroom_verdict", 2), self.BOOK)

    def test_the_seam_budget_bound_reads_the_group_book(self):
        self.assertEqual(self._call_arg("flip_seam_budget_verdict", 1), self.BOOK)

    def test_the_writeback_bound_reads_the_group_book(self):
        found = []
        for node in ast.walk(self._body()):
            if isinstance(node, ast.Compare) and any(
                isinstance(c, ast.Name) and c.id == "_WRITEBACK_DEFER_LIMIT"
                for c in node.comparators
            ):
                found.append(ast.unparse(node.left))
        self.assertTrue(found, "the writeback defer bound is gone")
        for left in found:
            self.assertEqual(left, self.BOOK)

    def test_the_correct_form_is_still_where_it_was_copied_from(self):
        """Can-fail anchor: if the seam-margin term stops reading the book,
        the three tests above are copying a form that no longer exists."""
        src = inspect.getsource(phase_flip_runtime)
        self.assertIn("spent = " + self.BOOK, src)


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
