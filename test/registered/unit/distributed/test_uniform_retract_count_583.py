"""#583 (desync site 2): HOW MANY requests retraction pops is rank-uniform.

THE DEFECT THIS PINS CLOSED
---------------------------
#603 made the decision to ENTER retraction rank-uniform: the scheduler
MIN-reduces the pool headroom once per iteration
(``Scheduler._update_uniform_pool_budget``) and compares the replicated token
demand against the reduced value, so every rank agrees on *whether* the
decode batch is short.

It did not reach what happens next. Once inside
``ScheduleBatch.retract_decode``, the loop bound

    while first_iter or (not self.check_decode_mem(selected_indices=...)):

and the last-survivor test just below it

    if len(sorted_indices) <= 1 and not self.check_decode_mem(...)

both call ``check_decode_mem``, which -- before this fix -- compared the
RANK-LOCAL ``token_to_kv_pool_allocator.available_size()`` against the
replicated demand. Under uneven DCP/TP the ranks own weighted shares of the
token pool (the production rig boots [332656, 177832, 177832] tokens), so
each rank kept popping victims until ITS OWN pool was satisfied. The ranks
enter together and leave with DIFFERENT batches.

Two consequences, both collective-bearing:

  (a) different retracted counts -> different ``batch.reqs`` -> the ranks run
      forwards of different shapes;

  (b) a rank that pops down to the last survivor and still does not fit
      retracts that one too, so ``filter_batch`` leaves the batch EMPTY.
      ``update_running_batch`` then returns ``ret = None``, the ``if batch:``
      at the top of ``event_loop_overlap`` is False, and that rank SKIPS
      ``run_batch`` entirely and goes round to ``recv_requests`` -- while the
      other ranks enter the decode collective and wait for a peer that is
      never coming.

(b) is the crash signature observed on 2026-08-05 at 21:10 (and described in
#603's own docstring for the 19:41 abort): ranks 0/1 inside a BAR1
``all_to_all`` in a CUDA-graph replay, rank 2 in ``broadcast_pyobj`` under
``recv_requests``. The spin kernels take their abort path ~30 s later.

The fix hands ``retract_decode`` the SAME reduced value #603 already
computes, via ``ScheduleBatch.uniform_avail_floor``. Because that value is
the group MINIMUM it is <= every rank's own headroom, so the loop never
under-retracts (it cannot leave a rank short and OOM it) and every rank stops
on the same iteration. No collective is taken inside the loop.

Deliberately hermetic: no CUDA, no model, no real memory pool, no process
group. A bare ``ScheduleBatch`` with the collaborators retraction touches
faked out, and the REAL ``check_decode_mem`` / ``retract_decode`` under test.
"""

import types
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import schedule_batch as sb  # noqa: E402
from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 16

#: The three ranks of the crashing boot, scaled down: uneven pools, one
#: binding. The shape is what matters -- rank 2 owns the smallest share.
WORLD_AVAILS = [80, 48, 32]


def _make_req(rid, num_decoded):
    """Only the fields retraction and the token-demand estimate touch."""
    return types.SimpleNamespace(
        rid=rid,
        output_ids=[0] * num_decoded,
        origin_input_ids=[0] * 10,
        sampling_params=types.SimpleNamespace(max_new_tokens=200),
        to_finish=None,
        priority=None,
        solo_oom_count=0,
        # multiple of PAGE_SIZE -> this request needs one fresh page next step
        kv_committed_len=PAGE_SIZE,
    )


class _FakeAllocator:
    page_size = PAGE_SIZE

    def __init__(self, avail):
        self._avail = avail

    def available_size(self):
        return self._avail


def _make_batch(num_reqs, avail, *, uniform_floor=None):
    """A ScheduleBatch whose decode-mem decision is the REAL one.

    ``release_req`` / ``filter_batch`` are faked (they need real pools), but
    ``check_decode_mem``, ``new_tokens_required_next_decode`` and
    ``retract_decode`` are the production code under test.
    """
    batch = ScheduleBatch.__new__(ScheduleBatch)
    batch.reqs = [_make_req(f"r{i}", num_decoded=i + 1) for i in range(num_reqs)]
    batch.spec_algorithm = types.SimpleNamespace(is_none=lambda: True)
    batch.token_to_kv_pool_allocator = _FakeAllocator(avail)
    batch.tree_cache = object()
    batch.uniform_avail_floor = uniform_floor

    batch.released = []

    def _release_req(idx, remaining_req_count, server_args):
        batch.released.append(idx)

    def _filter_batch(keep_indices):
        batch.reqs = [batch.reqs[i] for i in keep_indices]

    batch.release_req = _release_req
    batch.filter_batch = _filter_batch
    return batch


def _run_retraction(num_reqs, avail, *, uniform_floor=None):
    """Returns (num_retracted, num_survivors). Eviction is stubbed out: it is
    a side effect on a real tree cache and frees nothing in this fixture."""
    server_args = types.SimpleNamespace(
        retraction_policy="priority",
        schedule_low_priority_values_first=False,
    )
    batch = _make_batch(num_reqs, avail, uniform_floor=uniform_floor)
    with mock.patch.object(sb, "evict_from_tree_cache", lambda *a, **k: None):
        retracted, _ratio, _aborts = batch.retract_decode(server_args)
    return len(retracted), len(batch.reqs)


class UniformRetractCountTest(unittest.TestCase):
    # -- (a) how many victims -----------------------------------------------

    def test_prefix_the_local_predicate_really_does_diverge(self):
        """THE FALSIFIER, first half: the fixture must be a genuinely
        divergent case, or the uniformity assertion below proves nothing.

        With the local pool driving the loop, each rank stops as soon as ITS
        OWN headroom covers the shrinking demand -- so the rank with the
        biggest pool retracts fewest and the binding rank retracts most.
        """
        counts = [_run_retraction(5, avail)[0] for avail in WORLD_AVAILS]
        self.assertEqual(
            len(set(counts)),
            3,
            msg=f"expected three different retraction counts, got {counts}",
        )
        # Monotone in pool size: more headroom -> fewer victims.
        self.assertEqual(counts, sorted(counts))

    def test_the_reduced_floor_makes_every_rank_retract_the_same_count(self):
        """THE FALSIFIER, second half: same three ranks, same demand, but the
        decision reads the group MINIMUM -- one answer."""
        floor = min(WORLD_AVAILS)
        counts = [
            _run_retraction(5, avail, uniform_floor=floor)[0] for avail in WORLD_AVAILS
        ]
        self.assertEqual(len(set(counts)), 1, msg=f"counts diverged: {counts}")
        # And it is the BINDING rank's answer, not the roomiest rank's: the
        # group can only decode what the smallest pool can hold.
        self.assertEqual(counts[0], _run_retraction(5, min(WORLD_AVAILS))[0])

    def test_survivor_sets_match_too(self):
        """Equal counts are not enough -- the ranks must keep the SAME
        requests, or the forwards still differ in shape."""
        floor = min(WORLD_AVAILS)
        survivors = [
            _run_retraction(5, avail, uniform_floor=floor)[1] for avail in WORLD_AVAILS
        ]
        self.assertEqual(len(set(survivors)), 1, msg=f"survivors: {survivors}")

    # -- (b) the batch-emptying case, i.e. the observed crash ---------------

    def test_prefix_only_the_binding_rank_empties_its_batch(self):
        """THE CRASH SHAPE, before the fix: ranks 0/1 keep a decode batch and
        enter ``run_batch``; the binding rank retracts even the last survivor,
        ends with an EMPTY batch, and skips ``run_batch`` altogether."""
        world = [4096, 4096, PAGE_SIZE - 1]  # rank 2 cannot fund even one page
        survivors = [_run_retraction(3, avail)[1] for avail in world]
        self.assertGreater(survivors[0], 0)
        self.assertGreater(survivors[1], 0)
        self.assertEqual(
            survivors[2],
            0,
            msg="fixture must actually empty the binding rank's batch",
        )

    def test_the_reduced_floor_keeps_the_batch_decision_uniform(self):
        """After the fix every rank reaches the same emptiness verdict, so
        ``if batch:`` at the top of the event loop is the same on all of
        them -- nobody runs a collective alone."""
        world = [4096, 4096, PAGE_SIZE - 1]
        floor = min(world)
        survivors = [
            _run_retraction(3, avail, uniform_floor=floor)[1] for avail in world
        ]
        self.assertEqual(len(set(survivors)), 1, msg=f"survivors: {survivors}")

    # -- the floor must not silently mean something else --------------------

    def test_absent_floor_falls_back_to_the_local_value(self):
        """Single rank and tests supply no floor; the local pool is then the
        correct source, and must not read as a stale zero."""
        batch = _make_batch(1, 777, uniform_floor=None)
        self.assertEqual(batch.decode_mem_avail(), 777)

    def test_a_supplied_floor_wins_over_the_local_value(self):
        batch = _make_batch(1, 777, uniform_floor=12)
        self.assertEqual(batch.decode_mem_avail(), 12)

    def test_the_gate_can_still_answer_yes(self):
        """A comfortable pool must NOT retract -- otherwise the uniformity
        tests above would pass against a predicate that always retracts."""
        num_retracted, survivors = _run_retraction(3, 4096, uniform_floor=4096)
        # The loop always pops one victim on entry (the caller only calls it
        # when the group agreed it is short), but it must stop immediately.
        self.assertEqual(num_retracted, 1)
        self.assertEqual(survivors, 2)


if __name__ == "__main__":
    unittest.main()
