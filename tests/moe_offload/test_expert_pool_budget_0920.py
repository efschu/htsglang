# SPDX-License-Identifier: Apache-2.0
"""Tasks #14/#48: the expert pool's LRU rows and the KV pool come from the
budget, not from the hand.

FIXTURE PROVENANCE -- every number in :data:`FN8AJ_LOG` is a line boot fn8aj
printed on 2026-09-20
(``/spinning/evidence-665-f1/boot_fn_fn8aj_20260920T094757Z.server.log``),
quoted verbatim.  Hermetic: no CUDA, no GPU, no file I/O.

THREE BOOTS THIS PINS, and the term has to agree with all three:

* **fn8aj** ``SGLANG_MOE_SCRATCH_SLOTS=145,36,36``, ``--max-total-tokens
  270000``.  Ran; left ~2.4 GiB of KV over-provision per rank on the card.
* **fn8ak** ``175,60,60``.  Died before ``/health``:
  ``pool mode requires buffer_size == R+C (97 != 43+60)`` -- rank 2 owns 97.
* **fn8ak2** ``175,60,54``.  Booted (KV 1.215/2.031/2.293 GiB), then rank 0
  went OOM in the FIRST 8192-token prefill chunk: ``MoE offload gather fetch
  failed (OutOfMemoryError ... 254 MiB free ... 30.88 GiB in use)``.
* **fn8ak3** ``145,60,54`` -- the 5090 left alone, the 3080s raised.

The term must REFUSE the first two and ACCEPT the third, from fn8aj's census
alone.  That is what :func:`test_the_term_retrodicts_all_three_configuration_boots`
asserts.
"""

import math
from dataclasses import replace

import pytest

from sglang.srt.planner import expert_pool_budget as epb

GIB = epb.GIB
MIB = epb.MIB

FN8AJ_LOG = """\
[2026-09-20 09:49:29 TP2] [vram-census] pp0tp2 after load: model tensors on device 10.80 GiB = {experts 8.95, hyper_connection 0.63, linear_attn 0.38, other 0.25, embed_tokens 0.20, lm_head 0.20, moe_gate 0.12, shared_expert 0.05, ple 0.03, norm 0.00}; torch allocated 11.08 GiB, reserved 12.13 GiB
[2026-09-20 09:49:32 TP1] [vram-census] pp0tp1 after load: model tensors on device 10.34 GiB = {experts 8.50, hyper_connection 0.63, linear_attn 0.38, other 0.25, embed_tokens 0.20, lm_head 0.20, moe_gate 0.12, shared_expert 0.05, ple 0.03, norm 0.00}; torch allocated 10.64 GiB, reserved 11.74 GiB
[2026-09-20 09:50:07 TP0] [vram-census] pp0tp0 after load: model tensors on device 19.61 GiB = {experts 16.65, linear_attn 1.26, hyper_connection 0.63, other 0.38, embed_tokens 0.20, lm_head 0.20, shared_expert 0.14, moe_gate 0.12, ple 0.03, norm 0.00}; torch allocated 20.13 GiB, reserved 21.98 GiB
[2026-09-20 09:50:10 TP0] [vram-census] pp0tp0-draft after load: model tensors on device 1.78 GiB = {experts 0.81, embed_tokens 0.39, lm_head 0.39, other 0.14, hyper_connection 0.04, shared_expert 0.01, moe_gate 0.00, norm 0.00}; torch allocated 21.48 GiB, reserved 23.88 GiB
[2026-09-20 09:50:11 TP1] KV pool sizing: available_bytes=5081313280 (4.732 GiB), cell_size=14143, page_size=64 -> max_total_num_tokens=359232
[2026-09-20 09:50:11 TP0] KV pool sizing: available_bytes=5074821120 (4.726 GiB), cell_size=14143, page_size=64 -> max_total_num_tokens=358784
[2026-09-20 09:50:11 TP2] KV pool sizing: available_bytes=4659785728 (4.340 GiB), cell_size=14143, page_size=64 -> max_total_num_tokens=329472
[2026-09-20 09:50:11 TP0] [world_rank 0] KV budget posts (GiB): weights + runtime state=23.945, mamba state pool=0.034, speculative intermediate state=0.103, GGUF dequant scratch=0.000 | rest=4.726 | measured free=6.641 | unaccounted=+1.915
[2026-09-20 09:50:11 TP0] Uneven-DCP token sizing: rank 0 local capacity 269995 tokens / ratio 11 = unit 24545; min-reduced unit 24545 -> projected 785440 -> EFFECTIVE max_total_num_tokens 262151 (bound by hybrid mamba cap 262151; vector [11, 11, 10], hybrid mamba cap 262151).
[2026-09-20 09:50:11 TP2] Uneven-DCP token sizing: rank 2 local capacity 270000 tokens / ratio 10 = unit 27000; min-reduced unit 24545 -> projected 785440 -> EFFECTIVE max_total_num_tokens 262151 (bound by hybrid mamba cap 262151; vector [11, 11, 10], hybrid mamba cap 262151).
[2026-09-20 09:50:11 TP1] Uneven-DCP token sizing: rank 1 local capacity 269995 tokens / ratio 11 = unit 24545; min-reduced unit 24545 -> projected 785440 -> EFFECTIVE max_total_num_tokens 262151 (bound by hybrid mamba cap 262151; vector [11, 11, 10], hybrid mamba cap 262151).
[2026-09-20 09:50:13 TP1] MoE expert pool on layer 0: residents 39, LRU rows 28, staging 8, spill rows 66, tensors 4, prefetch off
[2026-09-20 09:50:13 TP2] MoE expert pool on layer 0: residents 43, LRU rows 28, staging 8, spill rows 54, tensors 4, prefetch off
[2026-09-20 09:50:13 TP0] MoE expert pool on layer 0: residents 2, LRU rows 137, staging 8, spill rows 311, tensors 4, prefetch off
[2026-09-20 09:50:48 TP1] [vram-peak] extend (8192 rows): allocator peak since pools 14.64 GiB, allocated now 12.94, reserved 18.19, card free 0.85 of 19.58 GiB -> transient headroom used = peak - allocated 1.69 GiB
[2026-09-20 09:50:48 TP2] [vram-peak] extend (8192 rows): allocator peak since pools 14.98 GiB, allocated now 13.29, reserved 18.08, card free 0.96 of 19.58 GiB -> transient headroom used = peak - allocated 1.69 GiB
[2026-09-20 09:50:48 TP0] [vram-peak] extend (8192 rows): allocator peak since pools 24.77 GiB, allocated now 23.09, reserved 29.82, card free 0.55 of 31.34 GiB -> transient headroom used = peak - allocated 1.69 GiB
[2026-09-20 09:55:26 TP1] [vram-peak] decode (3 rows): allocator peak since pools 14.79 GiB, allocated now 12.80, reserved 17.39, card free 1.63 of 19.58 GiB -> transient headroom used = peak - allocated 2.00 GiB
[2026-09-20 09:55:27 TP2] [vram-peak] decode (3 rows): allocator peak since pools 15.14 GiB, allocated now 13.15, reserved 16.92, card free 2.09 of 19.58 GiB -> transient headroom used = peak - allocated 2.00 GiB
[2026-09-20 09:55:27 TP0] [vram-peak] decode (3 rows): allocator peak since pools 25.41 GiB, allocated now 22.94, reserved 29.71, card free 0.65 of 31.34 GiB -> transient headroom used = peak - allocated 2.47 GiB
"""

#: User law (memory KONTEXT-262K-PFLICHT): Next Flash is always planned and
#: booted at 262144, never at a convenience length.
CTX = 262144


@pytest.fixture
def census():
    return epb.parse_boot_log(FN8AJ_LOG)


# --- 1. the census is read, never assumed ----------------------------------


def test_census_reads_every_rank_from_the_boots_own_instruments(census):
    assert [c.rank for c in census] == [0, 1, 2]
    r0, r1, r2 = census
    assert (r0.available_bytes, r0.cell_size, r0.page_size) == (5074821120, 14143, 64)
    assert (r0.dcp_ratio, r1.dcp_ratio, r2.dcp_ratio) == (11, 11, 10)
    assert (r0.kv_tokens_now, r1.kv_tokens_now, r2.kv_tokens_now) == (
        269995,
        269995,
        270000,
    )
    # C = LRU rows + staging; buffer = R + C; E_local = R + spill.
    assert (r0.scratch, r1.scratch, r2.scratch) == (145, 36, 36)
    assert (r0.buffer_rows, r1.buffer_rows, r2.buffer_rows) == (147, 75, 79)
    assert (r0.owned_experts, r1.owned_experts, r2.owned_experts) == (313, 105, 97)


def test_the_headroom_instrument_is_card_free_at_the_WORST_load_state(census):
    # rank 0 is free 0.55 GiB at extend and 0.65 at decode -> 0.55 is kept,
    # with the whole line it came from.
    assert [c.card_free_gib for c in census] == [0.55, 0.85, 0.96]
    assert census[0].peak_gib == 24.77 and census[0].allocated_gib == 23.09


def test_the_two_cheaper_transient_readings_are_reported_but_not_used(census):
    r0 = census[0]
    # what the [vram-peak] line itself calls the transient -- fn8ak2 was
    # planned on it and died.
    assert r0.peak_minus_allocated_gib == pytest.approx(1.68, abs=0.01)
    # the allocator cache high-water -- larger than rank 0's whole KV budget,
    # so a term subtracting it would refuse a boot that ran.
    assert r0.allocator_cache_gib == pytest.approx(6.73, abs=0.01)
    assert r0.allocator_cache_gib > r0.available_bytes / GIB


def test_draft_census_line_is_not_counted_as_arena(census):
    # rank 0's draft census says experts 0.81 GiB; the arena is 16.65.
    assert census[0].expert_tensor_gib == pytest.approx(16.65)


# --- 2. the row cost is measured -------------------------------------------


def test_row_bytes_agree_across_three_ranks_with_three_different_arenas(census):
    mib = [
        epb.row_bytes_from_census(c.expert_tensor_gib, c.buffer_rows) / MIB
        for c in census
    ]
    assert mib == pytest.approx([115.98, 116.05, 116.01], abs=0.02)
    # 0.06 % spread over R/C combinations 2/145, 39/36, 43/36 -> a measurement,
    # not a model. Cross-check against the nominal 2.45 MiB x 48 MoE layers:
    assert all(abs(m - 2.45 * 48) / m < 0.02 for m in mib)


def test_row_cost_refuses_to_guess_without_a_census():
    with pytest.raises(ValueError, match="refuses to guess"):
        epb.row_bytes_from_census(0.0, 147)


# --- 3. the KV need comes from the DCP unit, not from the rest -------------


def test_kv_need_is_the_dcp_unit_times_the_ratio_page_rounded_up():
    unit, local0 = epb.kv_tokens_for_rank(
        ctx_tokens=CTX, spec_tokens=0, dcp_ratios=(11, 11, 10), rank=0, page_size=64
    )
    _, local2 = epb.kv_tokens_for_rank(
        ctx_tokens=CTX, spec_tokens=0, dcp_ratios=(11, 11, 10), rank=2, page_size=64
    )
    assert unit == 8192 == math.ceil(CTX / 32)
    assert (local0, local2) == (90112, 81920)
    # 1.19 GiB at cell_size 14143 -- against the 3.56 GiB fn8aj allocated.
    assert local0 * 14143 / GIB == pytest.approx(1.187, abs=0.002)


def test_spec_rows_are_charged_to_the_world_not_once_per_rank():
    # 3 draft tokens x 64 concurrent seqs = 192 WORLD tokens. They are divided
    # by the ratio sum once (192/32 = 6 units), never added per rank -- a
    # per-rank addend would charge them len(ratios) times over.
    unit_a, local_a = epb.kv_tokens_for_rank(
        ctx_tokens=CTX, spec_tokens=0, dcp_ratios=(11, 11, 10), rank=0, page_size=64
    )
    unit_b, local_b = epb.kv_tokens_for_rank(
        ctx_tokens=CTX, spec_tokens=192, dcp_ratios=(11, 11, 10), rank=0, page_size=64
    )
    assert (unit_a, unit_b) == (8192, 8198)
    assert (local_a, local_b) == (90112, 90240)  # 8198 * 11 = 90178, page_up 64


def test_page_rounding_goes_UP_because_the_sizer_rounds_down():
    # a pool of 100 would be floored to 64 by pool_configurator.py:534.
    _, local = epb.kv_tokens_for_rank(
        ctx_tokens=100, spec_tokens=0, dcp_ratios=(1,), rank=0, page_size=64
    )
    assert local == 128


# --- 4. the latched [vram-peak] sample is NOT a maximum ---------------------
#
# fn8aj's log carries no measured prefill peak: maybe_log_vram_peak latched on
# the first extend of >= 2048 rows, which under chunked prefill is the first
# chunk of the first request.  A strict plan therefore REFUSES to size the
# pool from it -- which is the lesson fn8ak3 paid for.


def test_a_log_without_a_measured_prefill_peak_is_refused_by_default(census):
    with pytest.raises(epb.ExpertPoolBudgetRefused) as exc:
        epb.plan_expert_pool(census, ctx_tokens=CTX)
    assert exc.value.rank == 0
    assert "LATCHED sample" in str(exc.value)


def test_the_upper_bound_plan_is_available_but_says_so(census):
    plan = epb.plan_expert_pool(census, ctx_tokens=CTX, strict=False)
    assert plan.unit_tokens == 8192
    # user law: under uneven DCP quote the WORLD pool.
    assert plan.kv_tokens_world == 262144
    # --max-total-tokens is the WORLD number (fn8am), the pool cap is per rank.
    assert plan.max_total_tokens == 262144
    assert plan.pool_cap_tokens == (90112, 90112, 81920)
    # each rank holds its OWN share C * ratio_r / S, never the global number.
    assert [p.kv_gib for p in plan.ranks] == pytest.approx(
        [1.187, 1.187, 1.079], abs=0.002
    )
    assert [p.kv_credit_bytes / GIB for p in plan.ranks] == pytest.approx(
        [2.369, 2.369, 2.477], abs=0.002
    )
    assert [p.rows_affordable for p in plan.ranks] == [25, 28, 18]
    assert [p.scratch for p in plan.ranks] == [170, 64, 54]
    # and every rank carries the warning that these are not budgets.
    for rank in (0, 1, 2):
        assert any(
            n.startswith(f"rank{rank}: no measured prefill peak") for n in plan.notes
        )
    assert all("UPPER BOUND" in n for n in plan.notes if "no measured" in n)


def test_the_pool_identity_holds_on_every_rank_by_construction(census):
    plan = epb.plan_expert_pool(census, ctx_tokens=CTX, strict=False)
    for p in plan.ranks:
        assert p.buffer_rows <= p.owned_experts
        assert p.leftover_bytes >= 0.0


def test_rank0_buys_fewest_rows_despite_the_largest_budget(census):
    """The 5090 has the biggest card and the biggest KV credit and still gets
    the fewest rows -- because its card free at the sampled state is 0.55 GiB
    against the 3080s' 0.85/0.96."""
    plan = epb.plan_expert_pool(census, ctx_tokens=CTX, strict=False)
    r0, r1, r2 = plan.ranks
    assert r0.card_free_bytes < r1.card_free_bytes < r2.card_free_bytes
    assert r0.rows_affordable < r1.rows_affordable
    assert r0.bound_by == "budget"


def test_rank2_is_ownership_capped_and_the_surplus_is_named_for_the_ratio(census):
    plan = epb.plan_expert_pool(census, ctx_tokens=CTX, strict=False)
    r2 = plan.ranks[2]
    assert r2.bound_by.startswith("ownership")
    # 30 rows affordable, 18 of them ownable (97 - 43 - 36).
    assert r2.rows_affordable == 18 and r2.ownership_surplus_rows == 12
    assert r2.buffer_rows == 97 == r2.owned_experts
    assert any("--rank-moe-ratio" in n for n in plan.notes)


def test_env_replaces_both_hand_pins_and_books_the_transient(census):
    plan = epb.plan_expert_pool(census, ctx_tokens=CTX, strict=False)
    env = plan.env()
    assert env["SGLANG_MOE_SCRATCH_SLOTS"] == "170,64,54"
    # the knob that shrinks the pool is per rank ...
    assert env["SGLANG_KV_POOL_CAP_TOKENS"] == "90112,90112,81920"
    # ... and --max-total-tokens stays the GLOBAL context ceiling (fn8am).
    assert env["MAX_TOTAL_TOKENS"] == "262144"
    # User law: transients are BOOKED, explicitly, per rank. The value is the
    # residual that keeps the sizer from handing the rows' bytes back to KV.
    booked = [int(x) for x in env["SGLANG_KV_BUDGET_PREFILL_TRANSIENT_MIB"].split(",")]
    assert len(booked) == 3 and all(b > 0 for b in booked)
    for b, c, p in zip(booked, census, plan.ranks):
        assert b == pytest.approx(
            (c.available_bytes - p.kv_bytes - p.rows_affordable * p.row_bytes) / MIB,
            abs=1.0,
        )


def test_no_reserve_is_added_anywhere(census):
    """User law 2026-09-19, verbatim: reserves NEVER, 'nicht ein Byte'."""
    assert epb.DEFAULT_FLOOR_BYTES == 0
    plan = epb.plan_expert_pool(census, ctx_tokens=CTX, strict=False)
    for p in plan.ranks:
        assert p.headroom_bytes == pytest.approx(
            p.card_free_bytes + p.kv_credit_bytes, rel=1e-9
        )
        if p.bound_by == "budget":
            assert p.leftover_bytes < p.row_bytes


def test_a_named_floor_is_possible_but_never_default(census):
    tight = epb.plan_expert_pool(
        census, ctx_tokens=CTX, floor_bytes=int(1.0 * GIB), strict=False
    )
    loose = epb.plan_expert_pool(census, ctx_tokens=CTX, strict=False)
    assert tight.ranks[0].rows_affordable < loose.ranks[0].rows_affordable


# --- 5. the MEASURED prefill peak, and the chunk lever ---------------------
#
# fn8ak3 booted 145,60,54 -- the 5090 untouched, the 3080s +24/+18 rows -- and
# rank 2 went OOM in the needle prefill:
#
#   Tried to allocate 160 MiB, 73.5 MiB free, 19.36 GiB in use;
#   18.60 GiB allocated by PyTorch, 1.79 GiB in private pools (CUDA Graphs),
#   372 MiB reserved-unallocated
#
# with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True already set, so this is
# not fragmentation: at the real prefill peak the 3080 is simply full.  The
# 18.60 GiB is the number [vram-peak] should have printed and printed 14.98
# for.  Only rank 2's peak was measured, so only rank 2 is priced here.

MEASURED_PEAK_GIB = 18.60  # fn8ak3 rank 2, OOM line
GRAPH_PRIVATE_GIB = 1.79  # same line: CUDA-graph private pools, chunk-independent
FN8AK3_CHUNK = 8192


@pytest.fixture
def rank2_measured(census):
    return replace(
        census[2],
        prefill_peak_allocated_gib=MEASURED_PEAK_GIB,
        graph_private_gib=GRAPH_PRIVATE_GIB,
        prefill_chunk_tokens=FN8AK3_CHUNK,
    )


def _credit(c):
    return float(c.kv_bytes_now - 90112 * c.cell_size)


def test_the_measured_peak_is_3_6_GiB_above_what_the_latched_line_reported(
    rank2_measured,
):
    c = rank2_measured
    latched = c.peak_gib - c.allocated_gib  # 14.98 - 13.29
    real = c.prefill_peak_allocated_gib - c.allocated_gib  # 18.60 - 13.29
    assert latched == pytest.approx(1.69, abs=0.01)
    assert real == pytest.approx(5.31, abs=0.01)
    assert real - latched == pytest.approx(3.62, abs=0.02)


def test_at_chunk_8192_the_3080_has_no_headroom_at_all(rank2_measured):
    """The ~3.5 GiB that look idle on a 3080 ARE the prefill transient at
    CHUNK 8192.  They are free only in decode."""
    headroom, transient, note = epb.rank_headroom_bytes(
        rank2_measured,
        kv_credit_bytes=_credit(rank2_measured),
        prefill_chunk_tokens=8192,
    )
    assert note is None  # measured, so no upper-bound warning
    assert transient == pytest.approx(5.31, abs=0.01)
    # even after handing back 2.37 GiB of KV over-provision the rank is short.
    assert headroom < 0
    assert headroom / GIB == pytest.approx(-0.29, abs=0.03)


def test_halving_the_chunk_is_the_lever_that_actually_frees_rows(rank2_measured):
    at8192, t8192, _ = epb.rank_headroom_bytes(
        rank2_measured,
        kv_credit_bytes=_credit(rank2_measured),
        prefill_chunk_tokens=8192,
    )
    at4096, t4096, _ = epb.rank_headroom_bytes(
        rank2_measured,
        kv_credit_bytes=_credit(rank2_measured),
        prefill_chunk_tokens=4096,
    )
    # only the non-graph half scales: 1.79 fixed + (5.31 - 1.79)/2.
    assert t4096 == pytest.approx(GRAPH_PRIVATE_GIB + (t8192 - GRAPH_PRIVATE_GIB) / 2)
    assert t4096 == pytest.approx(3.55, abs=0.02)
    assert at8192 < 0 < at4096
    row = epb.row_bytes_from_census(
        rank2_measured.expert_tensor_gib, rank2_measured.buffer_rows
    )
    assert int(at4096 // row) == 12


def test_the_graph_private_pool_does_not_scale_with_the_chunk(rank2_measured):
    """1.79 GiB of CUDA-graph private pools are captured once and are the same
    at every chunk size, so a term that scaled the WHOLE transient would
    promise rows that halving CHUNK cannot deliver."""
    _, t, _ = epb.rank_headroom_bytes(
        rank2_measured, kv_credit_bytes=0.0, prefill_chunk_tokens=1
    )
    assert t > GRAPH_PRIVATE_GIB
    assert t == pytest.approx(GRAPH_PRIVATE_GIB, abs=0.01)


def test_a_measured_rank_needs_no_upper_bound_warning(rank2_measured):
    _, _, note = epb.rank_headroom_bytes(rank2_measured, kv_credit_bytes=0.0)
    assert note is None


# --- 6. the three configuration boots, retrodicted -------------------------


def test_the_term_retrodicts_all_three_configuration_boots(census):
    """From fn8aj's census alone, on the (optimistic) upper-bound reading."""
    # fn8ak 175,60,60 -- rank 2 died on the pool identity, rank 0 was over.
    bad = epb.verify_scratch_vector(census, (175, 60, 60), ctx_tokens=CTX, strict=False)
    assert len(bad) == 2
    assert "buffer_size == R+C (97 != 43+60)" in bad[1]
    assert bad[0].startswith("rank0:")

    # fn8ak2 175,60,54 -- booted, then rank 0 OOMed in the first prefill chunk.
    bad2 = epb.verify_scratch_vector(
        census, (175, 60, 54), ctx_tokens=CTX, strict=False
    )
    assert len(bad2) == 1 and bad2[0].startswith("rank0:")
    assert "5 row(s)" in bad2[0]

    # fn8ak3 145,60,54 -- accepted on the upper bound, and metal refuted it.
    # That is exactly why strict=True is the default: the upper bound is not a
    # budget, and this line is the standing reminder.
    assert (
        epb.verify_scratch_vector(census, (145, 60, 54), ctx_tokens=CTX, strict=False)
        == ()
    )


def test_the_measured_peak_would_have_refused_fn8ak3(rank2_measured):
    """The same rank 2, priced on the MEASURED peak instead of the latched
    sample, has negative headroom -- so its +18 rows are refused before the
    boot rather than discovered by an OOM."""
    headroom, _, _ = epb.rank_headroom_bytes(
        rank2_measured,
        kv_credit_bytes=_credit(rank2_measured),
        prefill_chunk_tokens=8192,
    )
    assert headroom < 0


def test_verify_rejects_a_vector_of_the_wrong_length(census):
    with pytest.raises(ValueError, match="2 entries for 3 ranks"):
        epb.verify_scratch_vector(census, (145, 36), ctx_tokens=CTX, strict=False)


# --- 7. refusal by name -----------------------------------------------------


def test_refusal_names_the_rank_and_the_shortfall(census):
    # give every rank a measured peak so the refusal below is the BUDGET one,
    # not the "no measured peak" one.
    starved = [
        replace(c, prefill_peak_allocated_gib=c.peak_gib, prefill_chunk_tokens=8192)
        for c in census
    ]
    # rank 1 keeps a pool it cannot shrink and has nothing free on the card.
    starved[1] = replace(starved[1], card_free_gib=0.0, kv_tokens_now=1024)
    with pytest.raises(epb.ExpertPoolBudgetRefused) as exc:
        epb.plan_expert_pool(tuple(starved), ctx_tokens=CTX)
    assert exc.value.rank == 1 and exc.value.shortfall_bytes > 0
    msg = str(exc.value)
    assert "rank1 is infeasible" in msg
    assert "KV pool credit" in msg and "card free at worst observed state" in msg
    assert "Nothing is clamped" in msg


def test_refusal_rather_than_a_silent_clamp_on_an_impossible_context(census):
    with pytest.raises(epb.ExpertPoolBudgetRefused) as exc:
        epb.plan_expert_pool(census, ctx_tokens=4_000_000, strict=False)
    assert exc.value.rank == 0


def test_a_kvless_rank_is_refused_not_priced(census):
    with pytest.raises(ValueError, match="KV-less stage"):
        replace(census[0], cell_size=0)


def test_a_self_inconsistent_census_is_refused(census):
    with pytest.raises(ValueError, match="exceeds owned experts"):
        replace(census[2], spill_rows=1)


def test_a_partial_log_is_not_a_census():
    trimmed = "\n".join(l for l in FN8AJ_LOG.splitlines() if "vram-peak" not in l)
    with pytest.raises(ValueError, match="missing"):
        epb.parse_boot_log(trimmed)


def test_a_log_without_the_dcp_vector_is_refused():
    trimmed = "\n".join(l for l in FN8AJ_LOG.splitlines() if "Uneven-DCP" not in l)
    with pytest.raises(ValueError, match="cannot be assumed"):
        epb.parse_boot_log(trimmed)


def test_ranks_never_uneins_on_the_page_size(census):
    mixed = list(census)
    mixed[1] = replace(mixed[1], page_size=32)
    with pytest.raises(ValueError, match="RAENGE NIE UNEINS"):
        epb.plan_expert_pool(tuple(mixed), ctx_tokens=CTX, strict=False)


# --- 8. the dry run ---------------------------------------------------------


def test_dry_run_prices_both_chunk_sizes():
    out = epb.dry_run(FN8AJ_LOG, ctx_tokens=CTX, strict=False)
    assert "=== prefill chunk 8192" in out and "=== prefill chunk 4096" in out
    assert "WORLD pool 262144 tokens" in out
    assert "SGLANG_MOE_SCRATCH_SLOTS=170,64,54" in out
    for rank in (0, 1, 2):
        assert f"rank{rank}:" in out
    assert "--rank-moe-ratio" in out


def test_dry_run_refuses_a_log_whose_peak_instrument_latched():
    out = epb.dry_run(FN8AJ_LOG, ctx_tokens=CTX)
    assert out.count("REFUSED") == len(epb.DRY_RUN_CHUNKS)
    assert "LATCHED sample" in out


def test_dry_run_works_on_a_grep_of_the_five_line_shapes():
    keep = (
        "vram-census",
        "vram-peak",
        "KV pool sizing",
        "Uneven-DCP",
        "MoE expert pool",
    )
    grepped = "\n".join(l for l in FN8AJ_LOG.splitlines() if any(k in l for k in keep))
    assert epb.dry_run(grepped, ctx_tokens=CTX, strict=False) == epb.dry_run(
        FN8AJ_LOG, ctx_tokens=CTX, strict=False
    )
