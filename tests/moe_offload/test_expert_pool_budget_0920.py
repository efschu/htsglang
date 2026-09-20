# SPDX-License-Identifier: Apache-2.0
"""Tasks #14/#48: the expert pool's LRU rows and the KV pool from the budget.

FIXTURE PROVENANCE -- :data:`FN8AM_LOG` is the five instrument shapes of boot
fn8am, 2026-09-20, quoted verbatim from
``/spinning/evidence-665-f1/boot_fn_fn8am_20260920T103851Z.server.log``.
Hermetic: no CUDA, no GPU, no file I/O.

WHAT THIS SUITE PINS, and it is mostly what the term must NOT claim:

* The pool is ALREADY compacted to ``(C // S + 1) * ratio_r``
  (``layers/dcp/owner.py:155``).  ``KV pool sizing`` and ``local capacity`` are
  CAPACITY; ``KV Cache is allocated ... #tokens`` is the allocation.  An
  earlier version of this term read the first two as a pool and invented a
  2.37 GiB "KV credit" per rank that does not exist.
* ``--max-total-tokens`` IS C, the world context budget, by design.  fn8am set
  it to one rank's physical row count (90816) and the 259415-token needle came
  back as "exceeds the maximum allowed length (90810 tokens)".
* The transient at CHUNK 8192 on a 3080 is ~4.4 GiB
  (``18.60`` allocated at fn8ak3's deep-prefill OOM minus ``11.85/12.25`` at
  rest minus the 18 rows that boot carried), not the ``1.92`` the
  ``[vram-peak]`` line calls "transient headroom used".
* Consequence: at CHUNK 8192 the rig has NO headroom for more rows once the
  pool is restored to the full 262144-token context.  Halving CHUNK buys
  ~10-11 rows per 3080.  Both are asserted.
"""

from dataclasses import replace

import pytest

from sglang.srt.planner import expert_pool_budget as epb

GIB = epb.GIB
MIB = epb.MIB

FN8AM_LOG = """\
[2026-09-20 10:41:05 TP0] KV pool sizing: available_bytes=5074821120 (4.726 GiB), cell_size=14143, page_size=64 -> max_total_num_tokens=358784
[2026-09-20 10:41:05 TP0] Uneven-DCP token sizing: rank 0 local capacity 90816 tokens / ratio 11 = unit 8256; min-reduced unit 8256 -> projected 264192 -> EFFECTIVE max_total_num_tokens 90816 (bound by --max-total-tokens user limit 90816; vector [11, 11, 10], hybrid mamba cap 262151).
[2026-09-20 10:41:05 TP1] KV pool sizing: available_bytes=5083410432 (4.734 GiB), cell_size=14143, page_size=64 -> max_total_num_tokens=359424
[2026-09-20 10:41:05 TP1] Uneven-DCP token sizing: rank 1 local capacity 90816 tokens / ratio 11 = unit 8256; min-reduced unit 8256 -> projected 264192 -> EFFECTIVE max_total_num_tokens 90816 (bound by --max-total-tokens user limit 90816; vector [11, 11, 10], hybrid mamba cap 262151).
[2026-09-20 10:41:05 TP2] KV pool sizing: available_bytes=4659785728 (4.340 GiB), cell_size=14143, page_size=64 -> max_total_num_tokens=329472
[2026-09-20 10:41:05 TP2] Uneven-DCP token sizing: rank 2 local capacity 90810 tokens / ratio 10 = unit 9081; min-reduced unit 8256 -> projected 264192 -> EFFECTIVE max_total_num_tokens 90816 (bound by --max-total-tokens user limit 90816; vector [11, 11, 10], hybrid mamba cap 262151).
[2026-09-20 10:41:06 TP0] KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 31229, K size: 0.18 GB, V size: 0.18 GB
[2026-09-20 10:41:06 TP0] KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 90816, K size: 0.02 GB, V size: 0.02 GB
[2026-09-20 10:41:06 TP0] [vram-census] pp0tp0 after pools: model tensors on device 19.61 GiB = {experts 16.65, linear_attn 1.26, hyper_connection 0.63, other 0.38, embed_tokens 0.20, lm_head 0.20, shared_expert 0.14, moe_gate 0.12, ple 0.03, norm 0.00}; torch allocated 21.41 GiB, reserved 24.60 GiB 
[2026-09-20 10:41:06 TP1] KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 31229, K size: 0.18 GB, V size: 0.18 GB
[2026-09-20 10:41:06 TP1] KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 90816, K size: 0.02 GB, V size: 0.02 GB
[2026-09-20 10:41:06 TP1] [vram-census] pp0tp1 after pools: model tensors on device 10.34 GiB = {experts 8.50, hyper_connection 0.63, linear_attn 0.38, other 0.25, embed_tokens 0.20, lm_head 0.20, moe_gate 0.12, shared_expert 0.05, ple 0.03, norm 0.00}; torch allocated 11.39 GiB, reserved 13.12 GiB 
[2026-09-20 10:41:06 TP2] KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 28390, K size: 0.16 GB, V size: 0.16 GB
[2026-09-20 10:41:06 TP2] KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 90816, K size: 0.02 GB, V size: 0.02 GB
[2026-09-20 10:41:06 TP2] [vram-census] pp0tp2 after pools: model tensors on device 10.80 GiB = {experts 8.95, hyper_connection 0.63, linear_attn 0.38, other 0.25, embed_tokens 0.20, lm_head 0.20, moe_gate 0.12, shared_expert 0.05, ple 0.03, norm 0.00}; torch allocated 11.79 GiB, reserved 13.47 GiB 
[2026-09-20 10:41:07 TP0] MoE expert pool on layer 0: residents 2, LRU rows 137, staging 8, spill rows 311, tensors 4, prefetch off
[2026-09-20 10:41:07 TP1] MoE expert pool on layer 0: residents 39, LRU rows 28, staging 8, spill rows 66, tensors 4, prefetch off
[2026-09-20 10:41:07 TP2] MoE expert pool on layer 0: residents 43, LRU rows 28, staging 8, spill rows 54, tensors 4, prefetch off
[2026-09-20 10:41:10 TP0] [vram-peak] high-water (1 rows): allocator peak since pools 21.91 GiB, allocated now 21.86, reserved 25.16, card free 5.21 of 31.34 GiB -> transient headroom used = peak - allocated 0.05 GiB
[2026-09-20 10:41:10 TP1] [vram-peak] high-water (1 rows): allocator peak since pools 11.89 GiB, allocated now 11.85, reserved 13.68, card free 5.37 of 19.58 GiB -> transient headroom used = peak - allocated 0.05 GiB
[2026-09-20 10:41:10 TP2] [vram-peak] high-water (1 rows): allocator peak since pools 12.29 GiB, allocated now 12.25, reserved 14.05, card free 5.00 of 19.58 GiB -> transient headroom used = peak - allocated 0.05 GiB
[2026-09-20 10:41:33 TP0] [vram-peak] high-water (141 rows): allocator peak since pools 22.24 GiB, allocated now 22.01, reserved 29.79, card free 0.58 of 31.34 GiB -> transient headroom used = peak - allocated 0.22 GiB
[2026-09-20 10:41:33 TP0] [vram-peak] high-water (141 rows): allocator peak since pools 22.24 GiB, allocated now 22.02, reserved 29.79, card free 0.58 of 31.34 GiB -> transient headroom used = peak - allocated 0.22 GiB
[2026-09-20 10:41:33 TP1] [vram-peak] high-water (141 rows): allocator peak since pools 11.92 GiB, allocated now 11.87, reserved 15.09, card free 3.96 of 19.58 GiB -> transient headroom used = peak - allocated 0.05 GiB
[2026-09-20 10:41:33 TP2] [vram-peak] high-water (141 rows): allocator peak since pools 12.31 GiB, allocated now 12.26, reserved 15.11, card free 3.94 of 19.58 GiB -> transient headroom used = peak - allocated 0.04 GiB
[2026-09-20 10:41:41 TP0] [vram-peak] extend (4800 rows): allocator peak since pools 23.38 GiB, allocated now 22.13, reserved 28.74, card free 1.63 of 31.34 GiB -> transient headroom used = peak - allocated 1.25 GiB
[2026-09-20 10:41:41 TP0] [vram-peak] extend (4800 rows): allocator peak since pools 23.38 GiB, allocated now 22.13, reserved 28.74, card free 1.63 of 31.34 GiB -> transient headroom used = peak - allocated 1.26 GiB
[2026-09-20 10:41:41 TP1] [vram-peak] extend (4800 rows): allocator peak since pools 13.16 GiB, allocated now 11.98, reserved 16.39, card free 2.64 of 19.58 GiB -> transient headroom used = peak - allocated 1.18 GiB
[2026-09-20 10:41:41 TP2] [vram-peak] extend (4800 rows): allocator peak since pools 13.55 GiB, allocated now 12.38, reserved 16.45, card free 2.59 of 19.58 GiB -> transient headroom used = peak - allocated 1.18 GiB
[2026-09-20 10:41:49 TP0] [vram-peak] decode (3 rows): allocator peak since pools 23.38 GiB, allocated now 22.10, reserved 30.20, card free 0.16 of 31.34 GiB -> transient headroom used = peak - allocated 1.28 GiB
[2026-09-20 10:41:49 TP1] [vram-peak] decode (3 rows): allocator peak since pools 13.16 GiB, allocated now 11.96, reserved 18.10, card free 0.92 of 19.58 GiB -> transient headroom used = peak - allocated 1.20 GiB
[2026-09-20 10:41:49 TP2] [vram-peak] decode (3 rows): allocator peak since pools 13.55 GiB, allocated now 12.35, reserved 18.44, card free 0.58 of 19.58 GiB -> transient headroom used = peak - allocated 1.20 GiB
[2026-09-20 10:41:59 TP0] [vram-peak] high-water (8192 rows): allocator peak since pools 24.13 GiB, allocated now 22.21, reserved 27.54, card free 2.82 of 31.34 GiB -> transient headroom used = peak - allocated 1.92 GiB
[2026-09-20 10:41:59 TP0] [vram-peak] high-water (8192 rows): allocator peak since pools 24.13 GiB, allocated now 22.21, reserved 27.54, card free 2.82 of 31.34 GiB -> transient headroom used = peak - allocated 1.93 GiB
[2026-09-20 10:41:59 TP1] [vram-peak] high-water (8192 rows): allocator peak since pools 13.99 GiB, allocated now 12.06, reserved 18.94, card free 0.08 of 19.58 GiB -> transient headroom used = peak - allocated 1.93 GiB
[2026-09-20 10:41:59 TP2] [vram-peak] high-water (8192 rows): allocator peak since pools 14.38 GiB, allocated now 12.46, reserved 17.60, card free 1.41 of 19.58 GiB -> transient headroom used = peak - allocated 1.92 GiB
[2026-09-20 10:41:59 TP2] [vram-peak] high-water (8192 rows): allocator peak since pools 14.38 GiB, allocated now 12.46, reserved 17.60, card free 1.41 of 19.58 GiB -> transient headroom used = peak - allocated 1.93 GiB
"""

#: User law KONTEXT-262K-PFLICHT: Next Flash is planned and booted at 262144.
CTX = 262144

#: fn8ak3's deep-prefill OOM on a 3080: "18.60 GiB allocated by PyTorch,
#: 1.79 GiB in private pools (CUDA Graphs)", at CHUNK 8192, with C=54 scratch
#: slots against fn8am's 36 -- so 18 of the rows in that number are arena, not
#: transient, and are subtracted.
FN8AK3_DEEP_PEAK_GIB = 18.60
FN8AK3_GRAPH_PRIVATE_GIB = 1.79
FN8AK3_EXTRA_ROWS = 18
CHUNK = 8192


def _line(lines, rank):
    """survey_ranks interleaves NOTE lines, so index by prefix, not position."""
    return next(l for l in lines if l.startswith(f"rank{rank}:"))


@pytest.fixture
def census():
    return epb.parse_boot_log(FN8AM_LOG)


@pytest.fixture
def measured(census):
    """fn8am's census with the 3080s' DEEP prefill peak filled in from fn8ak3.
    Rank 0 (the 5090) has no deep measurement anywhere, and stays None."""
    out = list(census)
    for r in (1, 2):
        out[r] = replace(
            out[r],
            prefill_peak_allocated_gib=FN8AK3_DEEP_PEAK_GIB,
            peak_boot_extra_rows=FN8AK3_EXTRA_ROWS,
            graph_private_gib=FN8AK3_GRAPH_PRIVATE_GIB,
            prefill_chunk_tokens=CHUNK,
        )
    return tuple(out)


# --- 1. the census reads ALLOCATION, not capacity --------------------------


def test_the_pool_row_count_is_the_allocation_line(census):
    # "KV pool sizing ... -> max_total_num_tokens=358784" is capacity;
    # "KV Cache is allocated ... #tokens: 31229" is the pool.
    assert [c.kv_rows_now for c in census] == [31229, 31229, 28390]
    assert [c.available_bytes for c in census] == [5074821120, 5083410432, 4659785728]


def test_a_big_draft_pool_is_not_mistaken_for_the_main_one(census):
    # fn8am allocates a 90816-token draft pool at 0.02 GB alongside the
    # 31229-token main pool at 0.18 GB. Token count would pick the wrong one.
    assert 90816 not in [c.kv_rows_now for c in census]


def test_the_compaction_rule_reproduces_both_boots():
    """(C // S + 1) * ratio_r, owner.py:155 -- the +1 is a ceil to a whole
    owner block and is not cosmetic."""
    for c_global, expected in (
        (90816, (31229, 31229, 28390)),
        (262151, (90123, 90123, 81930)),
    ):
        rows = tuple(
            epb.kv_rows_for_rank(
                ctx_tokens=c_global, spec_tokens=0, dcp_ratios=(11, 11, 10), rank=r
            )[1]
            for r in range(3)
        )
        assert rows == expected


def test_the_at_rest_sample_is_the_one_row_high_water(census):
    assert [c.card_free_after_pools_gib for c in census] == [5.21, 5.37, 5.00]
    assert [c.alloc_at_rest_gib for c in census] == [21.86, 11.85, 12.25]


def test_the_worst_card_free_is_the_minimum_over_every_peak_line(census):
    assert [c.card_free_worst_gib for c in census] == [0.16, 0.08, 0.58]


# --- 2. there is no KV credit ----------------------------------------------


def test_fn8aj_had_already_allocated_its_DCP_share_so_the_credit_was_zero():
    """The 2.37 GiB an earlier version of this term claimed never existed."""
    _, rows = epb.kv_rows_for_rank(
        ctx_tokens=262151, spec_tokens=0, dcp_ratios=(11, 11, 10), rank=0
    )
    assert rows == 90123  # exactly fn8aj's "KV Cache is allocated ... #tokens"
    _, need = epb.kv_rows_for_rank(
        ctx_tokens=CTX, spec_tokens=0, dcp_ratios=(11, 11, 10), rank=0
    )
    assert (rows - need) * 14143 / GIB == pytest.approx(0.0, abs=0.001)


def test_fn8ams_capped_pool_has_a_NEGATIVE_credit(census, measured):
    """Restoring the full context COSTS 0.78 GiB per ratio-11 rank -- the pool
    has to grow, so it funds no rows at all."""
    lines = epb.survey_ranks(
        measured, ctx_tokens=CTX, prefill_chunk_tokens=CHUNK, strict=False
    )
    assert "credit -0.776 GiB" in _line(lines, 1)
    assert "pool 31229 -> 90123 rows" in _line(lines, 1)


# --- 3. the transient, three readings --------------------------------------


def test_the_vram_peak_lines_own_reading_underbooks_by_more_than_2x(measured):
    c = measured[1]
    # what the line calls the transient at the 8192-row sample: 1.93 GiB.
    priced = (
        c.prefill_peak_allocated_gib
        - c.alloc_at_rest_gib
        - FN8AK3_EXTRA_ROWS
        * epb.row_bytes_from_census(c.expert_tensor_gib, c.buffer_rows)
        / GIB
    )
    assert priced == pytest.approx(4.71, abs=0.02)
    assert priced / 1.93 > 2.4


def test_the_card_side_reading_is_stricter_again_and_is_reported(measured):
    c = measured[1]
    # 5.37 -> 0.08 GiB of card free, while allocated moved only 11.85 -> 12.06.
    assert c.transient_card_gib == pytest.approx(5.29, abs=0.01)
    lines = epb.survey_ranks(
        measured, ctx_tokens=CTX, prefill_chunk_tokens=CHUNK, strict=False
    )
    assert any("never returns to the driver" in l for l in lines)


def test_only_the_non_graph_half_of_the_transient_scales_with_the_chunk(measured):
    c = measured[2]
    row = epb.row_bytes_from_census(c.expert_tensor_gib, c.buffer_rows)
    _, t8192, _ = epb.rank_headroom_bytes(
        c, kv_credit_bytes=0.0, row_bytes=row, prefill_chunk_tokens=8192, strict=False
    )
    _, t4096, _ = epb.rank_headroom_bytes(
        c, kv_credit_bytes=0.0, row_bytes=row, prefill_chunk_tokens=4096, strict=False
    )
    assert t4096 == pytest.approx(
        FN8AK3_GRAPH_PRIVATE_GIB + (t8192 - FN8AK3_GRAPH_PRIVATE_GIB) / 2, abs=0.01
    )


# --- 4. the answer ----------------------------------------------------------


def test_at_chunk_8192_there_is_no_headroom_on_any_rank(measured):
    """The finding, and it is a refusal: with the pool restored to 262144
    tokens the 3080s are 16-119 MiB short and the 5090 631 MiB short."""
    with pytest.raises(epb.ExpertPoolBudgetRefused) as exc:
        epb.plan_expert_pool(
            measured, ctx_tokens=CTX, prefill_chunk_tokens=CHUNK, strict=False
        )
    assert exc.value.rank == 0
    lines = epb.survey_ranks(
        measured, ctx_tokens=CTX, prefill_chunk_tokens=CHUNK, strict=False
    )
    assert all("SHORT by" in l for l in lines if l.startswith("rank"))


def test_halving_the_chunk_is_the_only_lever_that_buys_rows(measured):
    lines = epb.survey_ranks(
        measured, ctx_tokens=CTX, prefill_chunk_tokens=4096, strict=False
    )
    assert "+11 row(s) -> C 47" in _line(lines, 1)
    assert "+10 row(s) -> C 46" in _line(lines, 2)
    # the 5090 still has no deep measurement, so it still cannot be priced.
    assert "SHORT by" in _line(lines, 0)


def test_a_rank_without_a_deep_peak_is_refused_in_strict_mode(census):
    with pytest.raises(epb.ExpertPoolBudgetRefused):
        epb.plan_expert_pool(census, ctx_tokens=CTX, prefill_chunk_tokens=CHUNK)
    lines = epb.survey_ranks(census, ctx_tokens=CTX, prefill_chunk_tokens=CHUNK)
    assert all("REFUSED" in l for l in lines)


def test_the_world_pool_is_what_gets_quoted(measured):
    lines = epb.survey_ranks(
        measured, ctx_tokens=CTX, prefill_chunk_tokens=4096, strict=False
    )
    # per-rank figures are SHARES: 90123 + 90123 + 81930 = 262176 rows for a
    # 262144-token world context (the ceil to whole owner blocks).
    assert "-> 90123 rows" in _line(lines, 1)
    assert "-> 81930 rows" in _line(lines, 2)


# --- 5. the pool identity and the ownership cap ----------------------------


def test_the_pool_identity_is_never_broken(measured):
    lines = epb.survey_ranks(
        measured, ctx_tokens=CTX, prefill_chunk_tokens=4096, strict=False
    )
    # rank 2 owns 97 with R=43, C=36 -> 18 rows ownable; 10 are affordable, so
    # the budget binds here, not the ownership.
    assert "+10 row(s)" in _line(lines, 2)
    c = measured[2]
    assert c.owned_experts - c.residents - c.scratch == 18


def test_the_fn8ak_vector_still_breaks_the_identity(census):
    bad = epb.verify_scratch_vector(
        census, (175, 60, 60), ctx_tokens=CTX, prefill_chunk_tokens=CHUNK, strict=False
    )
    assert any("buffer_size == R+C (97 != 43+60)" in b for b in bad)


# --- 6. refusal by name -----------------------------------------------------


def test_refusal_names_every_post(measured):
    with pytest.raises(epb.ExpertPoolBudgetRefused) as exc:
        epb.plan_expert_pool(
            measured, ctx_tokens=CTX, prefill_chunk_tokens=CHUNK, strict=False
        )
    msg = str(exc.value)
    assert "card free after pools" in msg
    assert "KV pool credit (allocated now - DCP need)" in msg
    assert "prefill transient at this chunk" in msg
    assert "Nothing is clamped" in msg


def test_no_reserve_is_added_anywhere():
    assert epb.DEFAULT_FLOOR_BYTES == 0


def test_a_census_without_the_allocation_line_is_refused():
    trimmed = "\n".join(
        l for l in FN8AM_LOG.splitlines() if "KV Cache is allocated" not in l
    )
    with pytest.raises(ValueError, match="kv_rows_now"):
        epb.parse_boot_log(trimmed)


def test_a_log_without_the_dcp_vector_is_refused():
    trimmed = "\n".join(l for l in FN8AM_LOG.splitlines() if "Uneven-DCP" not in l)
    with pytest.raises(ValueError, match="cannot be assumed"):
        epb.parse_boot_log(trimmed)


def test_ranks_never_uneins_on_the_page_size(measured):
    mixed = list(measured)
    mixed[1] = replace(mixed[1], page_size=32)
    with pytest.raises(ValueError, match="RAENGE NIE UNEINS"):
        epb.plan_expert_pool(tuple(mixed), ctx_tokens=CTX, strict=False)


def test_row_cost_refuses_to_guess_without_a_census():
    with pytest.raises(ValueError, match="refuses to guess"):
        epb.row_bytes_from_census(0.0, 147)


def test_row_bytes_agree_across_three_different_arenas(census):
    mib = [
        epb.row_bytes_from_census(c.expert_tensor_gib, c.buffer_rows) / MIB
        for c in census
    ]
    assert mib == pytest.approx([115.98, 116.05, 116.01], abs=0.02)
    assert (max(mib) - min(mib)) / min(mib) < 0.001


# --- 7. the dry run ---------------------------------------------------------


def test_dry_run_prices_both_chunk_sizes_and_shows_every_rank():
    out = epb.dry_run(FN8AM_LOG, ctx_tokens=CTX, strict=False)
    assert "=== prefill chunk 8192" in out and "=== prefill chunk 4096" in out
    for rank in (0, 1, 2):
        assert f"rank{rank}:" in out
