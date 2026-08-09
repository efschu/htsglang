# SPDX-License-Identifier: Apache-2.0
"""#631 corpse I: one-sided speculative state across a phase seam.

THE DEATH THESE PIN. Production, 2026-08-09 20:31:48Z, epoch 8, PP0, one
pass after a tp_to_pp cutover that committed with two requests decoding
and a prefill pending:

    get_next_batch_to_run (scheduler.py:4368)
      -> running_batch.merge_batch(last_batch)   (schedule_batch.py:3399)
      -> self.spec_info.merge_batch(other.spec_info)
      -> eagle_info.py:271  len(spec_info.topk_index)
    AttributeError: 'NoneType' object has no attribute 'topk_index'

The carried TP decode batch kept a live ``EagleDraftInput``; the prefill
batch built after the cutover, in a phase with no drafter, had
``spec_info=None``. ``merge_batch`` branches on the truthiness of SELF
alone and then dereferences OTHER unconditionally, so the pair was fatal.

WHAT THE FIX IS, and therefore what these tests are aimed at. The seam is
the producer of the illegal pair, so the seam is where it is removed --
``clear_spec_info_for_unspeculated_phase`` on the TP->PP leg and
``arm_draft_bootstrap_all_reachable`` on the PP->TP leg. The guard inside
``merge_batch`` is ADDITIONAL and RAISES; it is not the fix. A test suite
that only pinned the guard would pass against a build that still produced
the illegal pair on every flip.

THE REACH IS THE WHOLE POINT. ``harvest_resident_batches`` -- the carry's
own harvest -- does not look at ``last_batch`` and drops empty batches.
``last_batch`` is exactly the handle that held the None side at 20:31:48.
So ``test_reach_covers_last_batch_which_the_resident_harvest_misses``
below is the test that would have caught the crash, and it is written to
fail if anyone narrows the reach back to the harvest.

Every behavioural pin is followed by a CAN-FAIL proof against a
deliberately unfixed input, so a refactor that turns a pin into a
tautology is visible.
"""

import types

import pytest
import torch

from sglang.srt.managers.phase_flip_draft_bootstrap import (
    _reachable_batches,
    clear_spec_info_for_unspeculated_phase,
)
from sglang.srt.managers.phase_flip_resident_carry import (
    harvest_resident_batches,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.eagle_info import EagleDraftInput


# --------------------------------------------------------------------------
# Fakes. The BATCH is real -- these tests exercise the real
# ScheduleBatch.merge_batch, because the dereference that killed PP0 lives
# in it and a stand-in would pin nothing.
# --------------------------------------------------------------------------


class FakeReq:
    def __init__(self, rid):
        self.rid = rid
        self.origin_input_ids = [1, 2, 3]
        self.output_ids = [4]
        self.finished_reason = None

    def finished(self):
        return False


class FakeSamplingInfo:
    def __init__(self):
        self.merged = 0

    def merge_batch(self, other):
        self.merged += 1


def draft_input(bs=1, hidden=8, topk=2):
    """A real EagleDraftInput shaped like one a TP decode round produces."""
    return EagleDraftInput(
        topk_p=torch.rand(bs, topk),
        topk_index=torch.randint(0, 100, (bs, topk)),
        hidden_states=torch.rand(bs, hidden),
        bonus_tokens=torch.randint(0, 100, (bs, 1)),
    )


def real_batch(rids, spec_info=None, spec_algorithm="EAGLE"):
    """A real ScheduleBatch carrying only what merge_batch touches.

    Built with ``__new__`` rather than the ctor on purpose: the ctor pulls
    in a model runner and a device pool, and none of that participates in
    the defect. The METHOD under test is the real one.
    """
    b = ScheduleBatch.__new__(ScheduleBatch)
    n = len(rids)
    b.reqs = [FakeReq(r) for r in rids]
    b.sampling_info = FakeSamplingInfo()
    b.model_config = types.SimpleNamespace(is_encoder_decoder=False)
    b.req_pool_indices = torch.arange(n, dtype=torch.int64)
    b.req_pool_indices_cpu = torch.arange(n, dtype=torch.int64)
    b.seq_lens = torch.full((n,), 16, dtype=torch.int64)
    b.orig_seq_lens = torch.full((n,), 16, dtype=torch.int64)
    b.seq_lens_cpu = torch.full((n,), 16, dtype=torch.int64)
    b.input_ids = torch.arange(n, dtype=torch.int64)
    b.out_cache_loc = None
    b.seq_lens_sum = n * 16
    b.return_logprob = False
    b.top_logprobs_nums = []
    b.token_ids_logprobs = []
    b.multimodal_inputs = None
    b.has_grammar = False
    b.return_hidden_states = False
    b.is_prefill_only = False
    b.encoder_lens = None
    b.encoder_lens_cpu = None
    b.mamba_track_indices = None
    b.mamba_track_mask = None
    b.mamba_track_seqlens = None
    b.spec_algorithm = spec_algorithm
    b.spec_info = spec_info
    return b


def scheduler_at_cutover(running_batch, last_batch=None, running_mbs=None):
    s = types.SimpleNamespace()
    s.running_batch = running_batch
    s.last_batch = last_batch
    s.cur_batch = None
    s.cur_batch_for_debug = None
    s.running_mbs = running_mbs or []
    return s


# --------------------------------------------------------------------------
# 1. THE CRASH ITSELF, reproduced through the real merge.
# --------------------------------------------------------------------------


def test_one_sided_spec_info_is_refused_by_name_not_by_attributeerror():
    """The 20:31:48 pair, merged. It must fail LOUDLY and say what it is.

    Before the fix this raised AttributeError from eagle_info.py:271, three
    frames below the batch that caused it and naming neither batch nor the
    seam. The replacement names both sides.
    """
    carried = real_batch(["87e74cda", "c099fc4f"], spec_info=draft_input(bs=2))
    fresh_pp_prefill = real_batch(["d38bd526"], spec_info=None)

    with pytest.raises(ValueError) as exc:
        carried.merge_batch(fresh_pp_prefill)

    msg = str(exc.value)
    assert "one-sided speculative state" in msg
    assert "EagleDraftInput" in msg
    assert "spec_info=None" in msg
    # It must not be mistakable for a generic merge failure: the message
    # has to point at the seam, because that is where the reader must go.
    assert "seam" in msg


def test_one_sided_spec_info_is_refused_in_the_mirror_direction_too():
    """self=None, other=set. The DANGEROUS one: it never crashed.

    ``if self.spec_info:`` is false, so the pre-existing code completed the
    merge and dropped other's draft state silently -- other's requests
    enter self.reqs and decode without the chain they were proposing on.
    Wrong tokens out of a healthy-looking server.
    """
    pp_running = real_batch(["a1"], spec_info=None)
    tp_leftover = real_batch(["b2"], spec_info=draft_input(bs=1))

    with pytest.raises(ValueError) as exc:
        pp_running.merge_batch(tp_leftover)
    assert "one-sided speculative state" in str(exc.value)


def test_can_fail_a_legal_pair_still_merges():
    """CAN-FAIL for both guards: they must not refuse legal pairs.

    Two speculating batches merge, and two non-speculating batches merge.
    If either raised, the guards above would be tautologies that also broke
    every normal spec decode round.
    """
    a = real_batch(["a1"], spec_info=draft_input(bs=1))
    b = real_batch(["b2"], spec_info=draft_input(bs=1))
    a.merge_batch(b)
    assert [r.rid for r in a.reqs] == ["a1", "b2"]
    assert len(a.spec_info.topk_index) == 2

    c = real_batch(["c3"], spec_info=None, spec_algorithm="NONE")
    d = real_batch(["d4"], spec_info=None, spec_algorithm="NONE")
    c.merge_batch(d)
    assert [r.rid for r in c.reqs] == ["c3", "d4"]
    assert c.spec_info is None


# --------------------------------------------------------------------------
# 2. THE SEAM. The fix proper: no TP draft state reachable in PP.
# --------------------------------------------------------------------------


def test_seam_clears_every_reachable_batch_then_the_real_merge_succeeds():
    """End to end over the real objects: run the seam, then the merge.

    This is the shape of the production sequence -- cutover leg, then the
    next ``get_next_batch_to_run`` merging the post-cutover prefill batch
    into the carried decode batch.
    """
    carried = real_batch(["87e74cda", "c099fc4f"], spec_info=draft_input(bs=2))
    sched = scheduler_at_cutover(carried, running_mbs=[carried])

    cleared, rids = clear_spec_info_for_unspeculated_phase(sched)

    assert cleared == 1
    assert set(rids) == {"87e74cda", "c099fc4f"}
    assert carried.spec_info is None

    # The batch the PP phase builds after the cutover, and the merge that
    # killed PP0. It must now go through.
    fresh_pp_prefill = real_batch(["d38bd526"], spec_info=None)
    carried.merge_batch(fresh_pp_prefill)
    assert [r.rid for r in carried.reqs] == ["87e74cda", "c099fc4f", "d38bd526"]
    assert carried.spec_info is None


def test_reach_covers_last_batch_which_the_resident_harvest_misses():
    """THE test that would have caught the crash.

    ``last_batch`` is the handle that held the fatal side. The carry's own
    ``harvest_resident_batches`` does not look at it -- deliberately, since
    it answers a different question -- so a seam built on that harvest
    leaves TP draft state reachable. This pins the two reaches apart, and
    fails if anyone unifies them in the wrong direction.
    """
    running = real_batch(["r1"], spec_info=draft_input(bs=1))
    last = real_batch(["l1"], spec_info=draft_input(bs=1))
    sched = scheduler_at_cutover(running, last_batch=last, running_mbs=[running])

    # The premise, asserted rather than assumed: the resident harvest does
    # NOT see last_batch.
    harvested = harvest_resident_batches(sched)
    assert running in harvested
    assert last not in harvested

    # The seam's reach does.
    assert last in _reachable_batches(sched)

    cleared, _ = clear_spec_info_for_unspeculated_phase(sched)
    assert cleared == 2
    assert running.spec_info is None
    assert last.spec_info is None


def test_can_fail_seam_built_on_the_resident_harvest_leaves_the_crash_live():
    """CAN-FAIL: the narrower reach, and the crash it leaves behind.

    This reproduces the pre-fix build by clearing only what the resident
    harvest returns, then performing the same merge. It must still refuse
    -- proving the passing test above is carried by the WIDER reach and not
    by the guard or by luck.
    """
    running = real_batch(["r1"], spec_info=draft_input(bs=1))
    last = real_batch(["l1"], spec_info=None)
    sched = scheduler_at_cutover(running, last_batch=last, running_mbs=[running])

    for batch in harvest_resident_batches(sched):
        batch.spec_info = None

    # running was cleared, so this particular pair is now legal...
    assert running.spec_info is None

    # ...but flip the roles the way epoch 8 actually had them -- the
    # carried TP batch reachable ONLY through last_batch -- and the narrow
    # reach misses it entirely.
    running2 = real_batch(["r2"], spec_info=None)
    last2 = real_batch(["l2"], spec_info=draft_input(bs=1))
    sched2 = scheduler_at_cutover(
        running2, last_batch=last2, running_mbs=[running2]
    )
    for batch in harvest_resident_batches(sched2):
        batch.spec_info = None
    assert last2.spec_info is not None, "narrow reach must miss last_batch"
    with pytest.raises(ValueError):
        running2.merge_batch(last2)


def test_reach_deduplicates_aliases_and_includes_empty_batches():
    """running_batch is routinely an ALIAS of a running_mbs slot.

    Counting it twice would misreport, and -- for any future reach that
    mutates rather than clears -- double-apply. Empty batches are included
    on purpose: an empty batch with a stale spec_info is still a live merge
    target once requests land in it.
    """
    shared = real_batch(["r1"], spec_info=draft_input(bs=1))
    empty = real_batch([], spec_info=draft_input(bs=1))
    sched = scheduler_at_cutover(shared, last_batch=empty, running_mbs=[shared])

    reach = _reachable_batches(sched)
    assert len([b for b in reach if b is shared]) == 1
    assert empty in reach

    cleared, rids = clear_spec_info_for_unspeculated_phase(sched)
    assert cleared == 2
    assert rids == ["r1"]  # the empty batch contributes no rid
    assert shared.spec_info is None and empty.spec_info is None


def test_seam_is_idempotent_and_silent_on_an_already_clean_phase():
    """The flip that commits on an idle non-speculating server pays nothing.

    That is the common case and it must not log or count anything.
    """
    sched = scheduler_at_cutover(real_batch(["r1"], spec_info=None))
    assert clear_spec_info_for_unspeculated_phase(sched) == (0, [])
    assert clear_spec_info_for_unspeculated_phase(sched) == (0, [])
