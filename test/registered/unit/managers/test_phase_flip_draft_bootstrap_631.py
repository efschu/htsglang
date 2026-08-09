# SPDX-License-Identifier: Apache-2.0
"""#631: the draft-state bootstrap for requests carried into a speculating
TP phase.

WHAT THESE PIN, and what they deliberately do not.

They pin the three properties the metal failure turned on:

  1. the SCRUB covers exactly the committed slots of every carried request
     and not one row more (a row past the allocation is another request's
     slot, and writing it is corruption dressed as hygiene);
  2. the SEED is merge-safe -- ``EagleDraftInput.merge_batch`` takes
     ``len(self.topk_index)`` unconditionally, so a seed carrying only
     bonus_tokens crashes the first time a request admitted after the
     cutover merges into the carried batch;
  3. the MARK survives exactly one round and is discharged only after the
     draft_extend that replaces the seed with a real chain.

They do NOT pin acceptance rate. That is a metal quantity and belongs in
the boot log, not in a fake.

Each behavioural pin is followed by a CAN-FAIL proof: the same assertion
against a deliberately broken input, so a future refactor that turns the
pin into a tautology is visible. Mutated code must TERMINATE and fail, not
hang (a previous can-fail in this feature spun forever on a frozen clock).
"""

import types

import pytest
import torch

from sglang.srt.managers.phase_flip_draft_bootstrap import (
    BOOTSTRAP_ATTR,
    DraftBootstrapError,
    arm_draft_bootstrap,
    batch_needs_bootstrap,
    build_bootstrap_draft_input,
    clear_bootstrap,
    committed_slots,
    draft_kv_layer_ids,
    draft_kv_pool,
    retune_carried_batches_for_phase,
    scrub_draft_kv,
)

HIDDEN = 8
N_SLOTS = 64


class FakeKVPool:
    """A one-layer draft pool with distinct key and value buffers."""

    def __init__(self, layer_num=1, start_layer=0, alias_value=False):
        self.layer_num = layer_num
        self.start_layer = start_layer
        self._k = [
            torch.full((N_SLOTS, 4), float(7 + i)) for i in range(layer_num)
        ]
        self._v = (
            self._k
            if alias_value
            else [torch.full((N_SLOTS, 4), float(9 + i)) for i in range(layer_num)]
        )

    def get_key_buffer(self, layer_id):
        return self._k[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id):
        return self._v[layer_id - self.start_layer]


class FakeReq:
    def __init__(self, rid, req_pool_idx, origin, output):
        self.rid = rid
        self.req_pool_idx = req_pool_idx
        self.origin_input_ids = list(origin)
        self.output_ids = list(output)


class FakeBatch:
    def __init__(self, reqs, seq_lens):
        self.reqs = list(reqs)
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64)
        self.spec_info = None


class FakeReqToTokenPool:
    def __init__(self, n_reqs=4):
        # Row r holds slots r*10 .. r*10+9, so a scrub that runs one row
        # long lands in a slot that provably belongs to nobody in the test
        # and is still checkable.
        self.req_to_token = torch.stack(
            [torch.arange(r * 10, r * 10 + 10) for r in range(n_reqs)]
        )


def make_scheduler(draft_runner=True):
    sched = types.SimpleNamespace()
    sched.device = "cpu"
    sched.req_to_token_pool = FakeReqToTokenPool()
    pool = FakeKVPool()
    inner = types.SimpleNamespace(
        draft_runner=types.SimpleNamespace(
            token_to_kv_pool=pool,
            spec_algorithm=types.SimpleNamespace(is_standalone=lambda: False),
            model_config=types.SimpleNamespace(
                spec_hidden_size=HIDDEN, dtype=torch.float32
            ),
        )
    )
    sched.draft_worker = types.SimpleNamespace(draft_worker=inner, topk=1)
    if not draft_runner:
        sched.draft_worker = None
    return sched, pool


def make_batch():
    return FakeBatch(
        [
            FakeReq("a", 0, [11, 12, 13], [14]),
            FakeReq("b", 1, [21, 22], [23, 24]),
        ],
        seq_lens=[3, 4],
    )


# --------------------------------------------------------------------- #
# 1. The scrub covers exactly the committed slots
# --------------------------------------------------------------------- #


def test_committed_slots_uses_seq_lens_not_seqlen():
    sched, _ = make_scheduler()
    batch = make_batch()
    rows = committed_slots(sched, batch)
    assert [r.tolist() for r in rows] == [[0, 1, 2], [10, 11, 12, 13]]


def test_can_fail_committed_slots_would_overrun_on_req_seqlen():
    """Proof the pin above is not a tautology.

    ``req.seqlen`` is len(origin) + len(output) -- one MORE than the
    committed KV length, because the last sampled token has no row yet.
    Slot 3 / slot 14 are what a seqlen-based implementation would scrub,
    and slot 14 is outside request b's allocation entirely.
    """
    sched, _ = make_scheduler()
    batch = make_batch()
    seqlens = [
        len(r.origin_input_ids) + len(r.output_ids) for r in batch.reqs
    ]
    assert seqlens == [4, 4]
    overrun = [
        sched.req_to_token_pool.req_to_token[r.req_pool_idx, :n].tolist()
        for r, n in zip(batch.reqs, seqlens)
    ]
    assert overrun[0] == [0, 1, 2, 3]
    correct = [r.tolist() for r in committed_slots(sched, batch)]
    assert correct[0] != overrun[0]


def test_scrub_zeroes_only_the_carried_slots():
    sched, pool = make_scheduler()
    batch = make_batch()
    rows, layers = scrub_draft_kv(pool, committed_slots(sched, batch))
    assert rows == 7
    assert layers == [0]
    k = pool.get_key_buffer(0)
    v = pool.get_value_buffer(0)
    for slot in [0, 1, 2, 10, 11, 12, 13]:
        assert torch.all(k[slot] == 0), slot
        assert torch.all(v[slot] == 0), slot
    # Untouched neighbours, on both sides of both runs.
    for slot in [3, 4, 9, 14, 20, 63]:
        assert torch.all(k[slot] == 7.0), slot
        assert torch.all(v[slot] == 9.0), slot


def test_can_fail_scrub_detects_an_unscrubbed_slot():
    sched, pool = make_scheduler()
    batch = make_batch()
    # Scrub everything EXCEPT request b -- the shape of a bug that harvests
    # only the first microbatch slot.
    scrub_draft_kv(pool, committed_slots(sched, batch)[:1])
    k = pool.get_key_buffer(0)
    assert torch.all(k[0] == 0)
    assert not torch.all(k[10] == 0)


def test_scrub_skips_an_aliased_value_buffer():
    """MLA-style pools return one tensor for key and value."""
    pool = FakeKVPool(alias_value=True)
    sched, _ = make_scheduler()
    batch = make_batch()
    rows, _ = scrub_draft_kv(pool, committed_slots(sched, batch))
    assert rows == 7
    assert torch.all(pool.get_key_buffer(0)[0] == 0)


def test_scrub_covers_every_layer_of_a_multilayer_draft():
    pool = FakeKVPool(layer_num=3, start_layer=2)
    sched, _ = make_scheduler()
    batch = make_batch()
    scrub_draft_kv(pool, committed_slots(sched, batch))
    for layer_id in (2, 3, 4):
        assert torch.all(pool.get_key_buffer(layer_id)[0] == 0), layer_id


class FakeHybridPool:
    """The pool shape that actually killed boot 08:56:39Z.

    A Qwen3.6 layer stack mixes full attention with linear/GDN layers, so
    the pool holds KV for SOME layer ids only, carries no ``layer_num``,
    and its ``get_key_buffer`` raises for any id outside the mapping.
    """

    def __init__(self):
        # Model layer 5 and 11 are the full-attention ones; the inner pool
        # numbers them 0 and 1.
        self.full_attention_layer_id_mapping = {5: 0, 11: 1}
        self._k = [torch.full((N_SLOTS, 4), 7.0) for _ in range(2)]
        self._v = [torch.full((N_SLOTS, 4), 9.0) for _ in range(2)]

    def _inner(self, layer_id):
        if layer_id not in self.full_attention_layer_id_mapping:
            raise ValueError(f"{layer_id=} not in full attention layers")
        return self.full_attention_layer_id_mapping[layer_id]

    def get_key_buffer(self, layer_id):
        return self._k[self._inner(layer_id)]

    def get_value_buffer(self, layer_id):
        return self._v[self._inner(layer_id)]


def test_layer_ids_come_from_the_mapping_on_a_hybrid_draft_pool():
    assert draft_kv_layer_ids(FakeHybridPool()) == [5, 11]


def test_scrub_walks_a_hybrid_pool_s_real_layer_ids():
    pool = FakeHybridPool()
    sched, _ = make_scheduler()
    batch = make_batch()
    rows, layers = scrub_draft_kv(pool, committed_slots(sched, batch))
    assert rows == 7
    assert layers == [5, 11]
    for layer_id in (5, 11):
        assert torch.all(pool.get_key_buffer(layer_id)[0] == 0)
        assert torch.all(pool.get_key_buffer(layer_id)[3] == 7.0)


def test_can_fail_a_range_based_scrub_is_rejected_by_a_hybrid_pool():
    """Proof the mapping lookup is load-bearing, not defensive padding.

    This is exactly what the first metal boot did: derive a range from a
    count, then hand the pool an id it does not hold.
    """
    pool = FakeHybridPool()
    assert not hasattr(pool, "layer_num")
    with pytest.raises(ValueError, match="not in full attention layers"):
        pool.get_key_buffer(0)


def test_scrub_refuses_a_pool_whose_geometry_is_unreadable():
    pool = FakeKVPool()
    pool.layer_num = 0
    with pytest.raises(DraftBootstrapError, match="declares neither"):
        scrub_draft_kv(pool, [torch.tensor([0])])


def test_committed_slots_refuses_a_length_mismatch():
    sched, _ = make_scheduler()
    batch = make_batch()
    batch.seq_lens = torch.tensor([3], dtype=torch.int64)
    with pytest.raises(DraftBootstrapError, match="disagree"):
        committed_slots(sched, batch)


# --------------------------------------------------------------------- #
# 2. The seed is merge-safe
# --------------------------------------------------------------------- #


def test_seed_roots_the_verify_at_each_request_s_last_committed_token():
    sched, _ = make_scheduler()
    batch = make_batch()
    seed = build_bootstrap_draft_input(sched, batch, topk=1)
    # a's last output token is 14, b's is 24.
    assert seed.bonus_tokens.tolist() == [14, 24]
    assert seed.topk_index.shape == (2, 1)
    assert seed.topk_p.shape == (2, 1)


def test_seed_falls_back_to_the_prompt_tail_before_any_output():
    sched, _ = make_scheduler()
    batch = FakeBatch([FakeReq("c", 0, [31, 32], [])], seq_lens=[2])
    seed = build_bootstrap_draft_input(sched, batch, topk=1)
    assert seed.bonus_tokens.tolist() == [32]


def test_seed_refuses_a_request_with_no_tokens_at_all():
    sched, _ = make_scheduler()
    batch = FakeBatch([FakeReq("d", 0, [], [])], seq_lens=[0])
    with pytest.raises(DraftBootstrapError, match="last committed token"):
        build_bootstrap_draft_input(sched, batch, topk=1)


def test_seed_survives_a_merge_with_a_freshly_prefilled_request():
    """The pin that matters for a batch mixing carried and new requests."""
    from sglang.srt.speculative.eagle_info import EagleDraftInput

    sched, _ = make_scheduler()
    batch = make_batch()
    seed = build_bootstrap_draft_input(sched, batch, topk=1)
    fresh = EagleDraftInput(
        bonus_tokens=torch.tensor([99], dtype=torch.int64),
        hidden_states=torch.zeros((1, seed.hidden_states.shape[1])),
        topk_p=torch.zeros((1, 1)),
        topk_index=torch.tensor([[99]], dtype=torch.int64),
        num_tokens_per_req=1,
        num_tokens_for_logprob_per_req=1,
    )
    seed.merge_batch(fresh)
    assert seed.bonus_tokens.tolist() == [14, 24, 99]


def test_can_fail_a_bonus_tokens_only_seed_breaks_the_merge():
    """Proof that the seed's extra fields are load-bearing, not decoration.

    This is the exact shape ``spec_in_tick_bootstrap_seed`` builds -- valid
    there because that batch is a single request that is never merged.
    """
    from sglang.srt.speculative.eagle_info import EagleDraftInput

    thin = EagleDraftInput(
        bonus_tokens=torch.tensor([14], dtype=torch.int64),
        num_tokens_per_req=1,
        num_tokens_for_logprob_per_req=1,
    )
    fresh = EagleDraftInput(
        bonus_tokens=torch.tensor([99], dtype=torch.int64),
        topk_p=torch.zeros((1, 1)),
        topk_index=torch.tensor([[99]], dtype=torch.int64),
        num_tokens_per_req=1,
        num_tokens_for_logprob_per_req=1,
    )
    with pytest.raises(TypeError):
        thin.merge_batch(fresh)


# --------------------------------------------------------------------- #
# 3. The mark, and the arming leg as a whole
# --------------------------------------------------------------------- #


def test_arm_scrubs_seeds_and_marks_in_one_call():
    sched, pool = make_scheduler()
    batch = make_batch()
    report = arm_draft_bootstrap(sched, batch, sched.draft_worker)
    assert report == {"reqs": 2, "rows": 7, "layers": [0], "armed": True}
    assert torch.all(pool.get_key_buffer(0)[0] == 0)
    assert batch.spec_info is not None
    assert batch_needs_bootstrap(batch)


def test_arm_is_a_no_op_without_speculation():
    sched, _ = make_scheduler()
    batch = make_batch()
    report = arm_draft_bootstrap(sched, batch, None)
    assert report["armed"] is False
    assert batch.spec_info is None
    assert not batch_needs_bootstrap(batch)


def test_arm_is_a_no_op_on_an_idle_flip():
    """The commit that takes an idle server must not pay for any of this."""
    sched, pool = make_scheduler()
    empty = FakeBatch([], seq_lens=[])
    report = arm_draft_bootstrap(sched, empty, sched.draft_worker)
    assert report == {"reqs": 0, "rows": 0, "armed": False}
    assert torch.all(pool.get_key_buffer(0)[0] == 7.0)


def test_the_mark_is_per_request_and_a_mixed_batch_still_bootstraps():
    sched, _ = make_scheduler()
    carried = make_batch()
    arm_draft_bootstrap(sched, carried, sched.draft_worker)
    fresh = FakeReq("new", 3, [41], [42])
    carried.reqs.append(fresh)
    assert not getattr(fresh, BOOTSTRAP_ATTR, False)
    assert batch_needs_bootstrap(carried)


def test_clear_discharges_every_mark_once():
    sched, _ = make_scheduler()
    batch = make_batch()
    arm_draft_bootstrap(sched, batch, sched.draft_worker)
    assert clear_bootstrap(batch) == 2
    assert not batch_needs_bootstrap(batch)
    assert clear_bootstrap(batch) == 0


def test_can_fail_an_uncleared_mark_keeps_the_batch_out_of_speculation():
    """If the discharge is ever dropped, the instance stops speculating.

    That is the failure this test exists to make loud: it is silent in
    production (correct answers, no crash, throughput back at the
    non-speculating rate), which is exactly the kind that survives a
    release.
    """
    sched, _ = make_scheduler()
    batch = make_batch()
    arm_draft_bootstrap(sched, batch, sched.draft_worker)
    for _ in range(5):
        assert batch_needs_bootstrap(batch)  # never discharged
    assert clear_bootstrap(batch) == 2
    assert not batch_needs_bootstrap(batch)


# --------------------------------------------------------------------- #
# 4. The batch's own view of the phase
# --------------------------------------------------------------------- #


class FakeAlgo:
    def __init__(self, name):
        self.name = name

    def is_none(self):
        return self.name is None

    def __repr__(self):
        return f"FakeAlgo({self.name})"


NONE_ALGO = FakeAlgo(None)
NEXTN = FakeAlgo("NEXTN")


def test_retune_points_every_carried_batch_at_the_new_phase():
    """The PP phase BUILT these batches, so their own field says NONE."""
    sched, _ = make_scheduler()
    slot0, slot1 = make_batch(), make_batch()
    slot0.spec_algorithm = NONE_ALGO
    slot1.spec_algorithm = NONE_ALGO
    sched.running_mbs = [slot0, None, slot1]
    sched.running_batch = slot0  # the usual alias of one slot
    assert retune_carried_batches_for_phase(sched, NEXTN) == 2
    assert slot0.spec_algorithm is NEXTN
    assert slot1.spec_algorithm is NEXTN


def test_retune_is_idempotent_and_reports_nothing_to_do():
    sched, _ = make_scheduler()
    slot0 = make_batch()
    slot0.spec_algorithm = NEXTN
    sched.running_mbs = [slot0]
    sched.running_batch = slot0
    assert retune_carried_batches_for_phase(sched, NEXTN) == 0


def test_retune_disarms_on_the_return_leg():
    """TP->PP is the mirror hole: a speculating batch landing in a phase
    that has no draft worker at all."""
    sched, _ = make_scheduler()
    slot0 = make_batch()
    slot0.spec_algorithm = NEXTN
    sched.running_mbs = [slot0]
    sched.running_batch = None
    assert retune_carried_batches_for_phase(sched, NONE_ALGO) == 1
    assert slot0.spec_algorithm is NONE_ALGO


def test_can_fail_a_stale_batch_field_splits_the_decode_round():
    """Proof the retune is load-bearing rather than tidy.

    ``ScheduleBatch.prepare_for_decode`` hands decode preparation to the
    spec path ONLY when the BATCH's own spec_algorithm is non-NONE. A
    carried batch left at NONE prepares itself the plain way while the
    scheduler's worker runs the speculating path over it -- the two halves
    of one round disagreeing about the phase.
    """
    sched, _ = make_scheduler()
    slot0 = make_batch()
    slot0.spec_algorithm = NONE_ALGO
    sched.running_mbs = [slot0]
    sched.running_batch = None
    # Without the retune the batch would take the non-spec branch:
    assert slot0.spec_algorithm.is_none()
    retune_carried_batches_for_phase(sched, NEXTN)
    assert not slot0.spec_algorithm.is_none()


def test_retune_skips_empty_slots_without_touching_them():
    sched, _ = make_scheduler()
    empty = FakeBatch([], seq_lens=[])
    empty.spec_algorithm = NONE_ALGO
    sched.running_mbs = [empty, None]
    sched.running_batch = None
    assert retune_carried_batches_for_phase(sched, NEXTN) == 0
    assert empty.spec_algorithm is NONE_ALGO


def test_draft_kv_pool_reads_through_the_spec_worker():
    sched, pool = make_scheduler()
    assert draft_kv_pool(sched.draft_worker) is pool
    assert draft_kv_pool(None) is None
    assert draft_kv_pool(types.SimpleNamespace()) is None
