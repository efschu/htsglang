# SPDX-License-Identifier: Apache-2.0
"""#861 fix (b): a request never speculates over draft rows nothing wrote.

THE HOLE #856 OPENED. #631's cutover leg scrubs a CARRIED request's stale draft
rows and seeds it for one non-drafting round. #856 retired the carry: residents
are retracted at the seam and re-admitted in the next layout as a read-through
prefill, so the cutover leg finds an empty resident set and the scrub never
runs. What reaches the TP phase instead is a request whose TARGET prefix was
restored from HiCache and whose DRAFT rows hold the previous occupants' bytes.

Four pins:

  1. THE COUNTER. `BOOTSTRAP_ATTR` is widened from a flag to a debt, monotonic,
     truthiness-compatible with the pre-#861 `True`, and BOUNDED -- a mark that
     never discharges is a permanent throughput regression that reads as a slow
     rig rather than as a defect.
  2. THE TRIGGERS. Seam re-admission and a disarmed draft tier, and NOT a
     request whose whole context was just computed here.
  3. THE ORDER. Scrub, then mark, then stamp armed. A request marked but not
     scrubbed stops speculating and still holds garbage; scrubbed but not
     marked speculates off zeros with no seed.
  4. BOTH FAILURE DIRECTIONS LOUD: never silently spec-off forever, never
     spec-on with unwritten draft rows.
"""

import types

import pytest
import torch

from sglang.srt.managers.phase_flip_draft_bootstrap import (
    BOOTSTRAP_ATTR,
    COLD_ARMED_ATTR,
    DEFAULT_DRAFT_COLD_ROUNDS,
    MAX_DRAFT_COLD_ROUNDS,
    DraftBootstrapError,
    arm_draft_cold_for_admission,
    batch_needs_bootstrap,
    clear_bootstrap,
    draft_cold_reason,
    mark_draft_cold,
    rounds_owed,
)
from sglang.srt.managers.phase_purity import SEAM_READMIT_ATTR

N_SLOTS = 64


class FakeKVPool:
    def __init__(self):
        self.layer_num = 1
        self.start_layer = 0
        self._k = torch.full((N_SLOTS, 4), 7.0)
        self._v = torch.full((N_SLOTS, 4), 9.0)

    def get_key_buffer(self, layer_id):
        return self._k

    def get_value_buffer(self, layer_id):
        return self._v


def make_req(rid, req_pool_idx, prefix_len, seam=False):
    req = types.SimpleNamespace(
        rid=rid,
        req_pool_idx=req_pool_idx,
        prefix_indices=list(range(prefix_len)),
    )
    if seam:
        setattr(req, SEAM_READMIT_ATTR, 3)
    return req


def make_scheduler(pool, tier_armed):
    # Slot ids must be real rows of the draft pool: 4 requests x 16 slots each,
    # disjoint, all inside [0, N_SLOTS).
    req_to_token = torch.arange(N_SLOTS, dtype=torch.int64).reshape(4, N_SLOTS // 4)
    controller = types.SimpleNamespace(draft_tier_armed=lambda direction: tier_armed)
    return types.SimpleNamespace(
        draft_worker=types.SimpleNamespace(
            draft_worker=types.SimpleNamespace(
                draft_runner=types.SimpleNamespace(token_to_kv_pool=pool)
            )
        ),
        req_to_token_pool=types.SimpleNamespace(req_to_token=req_to_token),
        tree_cache=types.SimpleNamespace(cache_controller=controller),
    )


def batch_of(*reqs):
    return types.SimpleNamespace(reqs=list(reqs))


# ---------------------------------------------------------------- pin 1


def test_counter_is_monotonic_and_truthiness_compatible():
    req = types.SimpleNamespace(rid="a")
    setattr(req, BOOTSTRAP_ATTR, True)  # the pre-#861 value
    assert rounds_owed(req) == 1
    mark_draft_cold(req, 3)
    assert rounds_owed(req) == 3
    mark_draft_cold(req, 1)  # a second, weaker reason must not cancel the first
    assert rounds_owed(req) == 3


def test_clear_bootstrap_discharges_one_round_at_a_time():
    req = types.SimpleNamespace(rid="a")
    mark_draft_cold(req, 3)
    batch = batch_of(req)
    for expected in (2, 1, 0):
        assert clear_bootstrap(batch) == 1
        assert rounds_owed(req) == expected
        assert batch_needs_bootstrap(batch) is (expected > 0)
    assert clear_bootstrap(batch) == 0


def test_default_is_byte_identical_to_the_pre_861_clear():
    """One round in, one clear out -- the #631 call site is unchanged."""
    req = types.SimpleNamespace(rid="a")
    mark_draft_cold(req)
    assert DEFAULT_DRAFT_COLD_ROUNDS == 1
    assert clear_bootstrap(batch_of(req)) == 1
    assert rounds_owed(req) == 0


def test_an_unbounded_mark_is_refused_where_it_is_set():
    """NEVER SILENTLY SPEC-OFF FOREVER. Refused at the SET site, the only place
    the intent is still visible; discovering it at discharge would mean the
    request had already stopped speculating for an unknown number of rounds."""
    req = types.SimpleNamespace(rid="a")
    with pytest.raises(DraftBootstrapError, match="ceiling"):
        mark_draft_cold(req, MAX_DRAFT_COLD_ROUNDS + 1)
    assert rounds_owed(req) == 0


def test_an_unreadable_mark_is_refused_rather_than_ignored():
    req = types.SimpleNamespace(rid="a")
    setattr(req, BOOTSTRAP_ATTR, "forever")
    with pytest.raises(DraftBootstrapError, match="never discharges"):
        rounds_owed(req)


# ---------------------------------------------------------------- pin 2


def test_trigger_seam_readmission():
    sched = make_scheduler(FakeKVPool(), tier_armed=True)
    req = make_req("s", 0, prefix_len=10, seam=True)
    assert "seam re-admission" in draft_cold_reason(sched, req, True)


def test_trigger_disarmed_draft_tier():
    sched = make_scheduler(FakeKVPool(), tier_armed=False)
    req = make_req("h", 0, prefix_len=10)
    assert "disarmed" in draft_cold_reason(sched, req, False)


def test_a_freshly_computed_context_is_never_cold():
    """NEVER SPEC-OFF WHAT IS WARM. No cached prefix means
    `_draft_extend_for_prefill` wrote every draft row in this very batch."""
    sched = make_scheduler(FakeKVPool(), tier_armed=False)
    assert draft_cold_reason(sched, make_req("f", 0, prefix_len=0), False) is None


def test_a_warm_tier_and_no_seam_stamp_is_not_cold():
    sched = make_scheduler(FakeKVPool(), tier_armed=True)
    assert draft_cold_reason(sched, make_req("w", 0, prefix_len=10), True) is None


# ---------------------------------------------------------------- pin 3


def test_admission_scrubs_the_prefix_marks_and_stamps():
    pool = FakeKVPool()
    sched = make_scheduler(pool, tier_armed=False)
    req = make_req("c", 1, prefix_len=5)
    report = arm_draft_cold_for_admission(sched, batch_of(req))

    assert report["cold"] == 1 and report["rows"] == 5
    assert rounds_owed(req) == DEFAULT_DRAFT_COLD_ROUNDS
    assert getattr(req, COLD_ARMED_ATTR) is True

    rows = sched.req_to_token_pool.req_to_token[1, :5]
    assert torch.all(pool.get_key_buffer(0)[rows] == 0)
    assert torch.all(pool.get_value_buffer(0)[rows] == 0)


def test_the_extend_region_is_not_scrubbed():
    """CAN-FAIL for the prefix bound: the rows past the cached prefix are
    written by THIS batch's draft extend, and scrubbing them would erase the
    one part that is real."""
    pool = FakeKVPool()
    sched = make_scheduler(pool, tier_armed=False)
    req = make_req("c", 1, prefix_len=5)
    arm_draft_cold_for_admission(sched, batch_of(req))
    beyond = sched.req_to_token_pool.req_to_token[1, 5:10]
    assert torch.all(pool.get_key_buffer(0)[beyond] == 7.0)


def test_admission_is_idempotent_across_chunked_visits():
    """A chunked prefill reaches run_batch many times with the same request;
    re-scrubbing would erase rows the drafter has since written."""
    pool = FakeKVPool()
    sched = make_scheduler(pool, tier_armed=False)
    req = make_req("c", 1, prefix_len=5)
    assert arm_draft_cold_for_admission(sched, batch_of(req))["cold"] == 1
    pool._k[sched.req_to_token_pool.req_to_token[1, :5]] = 5.0  # drafter wrote
    assert arm_draft_cold_for_admission(sched, batch_of(req))["cold"] == 0
    assert torch.all(pool.get_key_buffer(0)[sched.req_to_token_pool.req_to_token[1, :5]] == 5.0)


def test_no_drafter_in_this_phase_is_a_free_no_op():
    """The PP phase of a flip instance takes this exit on every prefill batch."""
    sched = make_scheduler(FakeKVPool(), tier_armed=False)
    sched.draft_worker = None
    assert arm_draft_cold_for_admission(sched, batch_of(make_req("x", 0, 5))) == {
        "cold": 0,
        "rows": 0,
    }


def test_warm_batch_is_untouched():
    pool = FakeKVPool()
    sched = make_scheduler(pool, tier_armed=True)
    req = make_req("w", 1, prefix_len=5)
    assert arm_draft_cold_for_admission(sched, batch_of(req))["cold"] == 0
    assert rounds_owed(req) == 0
    assert torch.all(pool.get_key_buffer(0) == 7.0)


def test_mixed_batch_marks_only_the_cold_ones():
    sched = make_scheduler(FakeKVPool(), tier_armed=True)
    cold = make_req("c", 1, prefix_len=4, seam=True)
    warm = make_req("w", 2, prefix_len=4)
    fresh = make_req("f", 3, prefix_len=0)
    arm_draft_cold_for_admission(sched, batch_of(cold, warm, fresh))
    assert rounds_owed(cold) == 1
    assert rounds_owed(warm) == 0 and rounds_owed(fresh) == 0
    # The batch-level shape: ONE round costs the warm ones a speculation step
    # and keeps the cold one off a chain it does not have.
    assert batch_needs_bootstrap(batch_of(cold, warm, fresh)) is True
