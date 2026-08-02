# SPDX-License-Identifier: Apache-2.0
"""#452 -- the replay contract, and the volume the graph is obliged to move.

``test_capture_desync_port.py`` proved the ported step's index ARITHMETIC: it
lands the same rows in the same slots as the eager fetch. It did that with a
fresh cache and a single step per test, and the boot disagreed with it anyway
(B2). That is the lesson this file acts on: **test the replay contract, not the
math.** A CUDA graph replays a fixed kernel sequence over fixed buffers, so the
properties that matter are the ones a single-step test cannot see --

* a step must be a pure function of the CURRENT contents of the static input
  buffer, never of what the previous step left behind;
* the scratch region is persistent storage between replays, so every slot a
  routed id can address must be re-established by the captured kernels;
* the gather must write THROUGH the ``out=`` view into the resident buffer the
  GEMM reads, because the graph bakes that pointer at capture time;

-- and, separately, the one the boot measured:

* the captured gather's index tensor has a STATIC length, so the step moves the
  worst-case scratch set every time, whatever the routing turns out to be. That
  is not an implementation choice; it is what "capturable" means. It is also
  the whole of B4, and this file measures it in bytes at the desk.

Geometry: the tests use the production shape the boot ran at (E=72 local
experts, R=31 resident, C=6 scratch -- see
``/spinning/gpu-battery-results/2026-08-02_desync_graph_proof/`` boot log,
"31/72 experts resident + 6 scratch"), which the #443 suite did not cover; its
shapes leave slack between ``bs x top_k`` and C, and the boot ran with none.

Run:  python -m pytest tests/moe_offload/test_capture_replay_contract_452.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from test_capture_desync_port import (  # noqa: E402
    ATTRS,
    _StubCache,
)

#: The boot's per-rank geometry. TP1 (the clock rank) and TP2 both ran
#: 31/72 + 6; TP0 ran 55/114 + 6. The scratch region is 6 either way because
#: ``bs=1 x top_k=6`` is what a captured decode step routes.
V4_SHAPE = dict(E=72, R=31, C=6)
TOP_K = 6


def _routing(n_spill, *, cache, seed=0, bs=1):
    """A bs x top_k routing with EXACTLY ``n_spill`` distinct spill experts.

    Constructed rather than sampled: the boot's operating point is
    ``bs x top_k == C``, i.e. the case where a rejection sampler that wants
    "spill fits scratch" almost always rejects. The remaining slots are filled
    with resident experts, which is what the #82 expert-dim shard produces --
    ``forward_impl`` has already collapsed every foreign id onto a resident
    zero-pad expert before the offload sees the ids.
    """
    g = torch.Generator().manual_seed(seed)
    resident = sorted(cache.planner.resident_ids or range(cache.resident_count))
    cold = cache.cold_ids
    perm = torch.randperm(len(cold), generator=g).tolist()
    chosen = [cold[i] for i in perm[:n_spill]]
    slots = bs * TOP_K
    assert n_spill <= slots
    pad = [
        resident[int(torch.randint(0, len(resident), (1,), generator=g))]
        for _ in range(slots - n_spill)
    ]
    ids = chosen + pad
    return torch.tensor(ids, dtype=torch.long).reshape(bs, TOP_K)


def _fresh(seed=0, **over):
    kwargs = dict(V4_SHAPE)
    kwargs.update(over)
    return _StubCache(seed=seed, **kwargs)


def _assert_routed_rows_are_right(cache, ids, remapped):
    """Every routed id addresses the row of the expert it routed to."""
    flat_ids = ids.reshape(-1).tolist()
    flat_slots = remapped.reshape(-1).tolist()
    for expert, slot in zip(flat_ids, flat_slots):
        if expert < 0:
            assert slot == -1
            continue
        assert 0 <= slot < cache.resident_count + cache.scratch
        for attr in ATTRS:
            assert torch.equal(cache._resident[attr][slot], cache.full[attr][expert]), (
                f"expert {expert} -> slot {slot} ({attr})"
            )


# --------------------------------------------------------------------------
# A. the replay contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_spill", [0, 1, 3, 6])
def test_a_step_is_a_pure_function_of_the_current_routing(n_spill):
    """History must not be observable.

    A replayed graph runs the same kernels over buffers that still hold the
    previous step's contents. If any part of the ported step carried state
    across -- a scratch slot the gather skips, a counter that only a first call
    initialises -- then the Nth step would differ from the same step run first,
    and THAT is the shape of a wrong-rows bug that a single-step test cannot
    see.
    """
    fresh = _fresh(seed=3)
    aged = _fresh(seed=3)
    target = _routing(n_spill, cache=fresh, seed=100 + n_spill)

    # Age one cache with a run of unrelated routings, exactly as decode does.
    static_ids = torch.zeros_like(target)
    for step in range(8):
        static_ids.copy_(_routing((step % 6) + 1, cache=aged, seed=step))
        aged.prepare_capturable(static_ids)

    # ...then drive BOTH with the same routing, writing the static input buffer
    # in place, which is what the graph runner does before every replay.
    static_ids.copy_(target)
    remap_fresh = fresh.prepare_capturable(target.clone())
    remap_aged = aged.prepare_capturable(static_ids)

    assert torch.equal(remap_fresh, remap_aged)
    for attr in ATTRS:
        used = fresh.resident_count + fresh.scratch
        assert torch.equal(fresh._resident[attr][:used], aged._resident[attr][:used]), (
            attr
        )
    _assert_routed_rows_are_right(aged, target, remap_aged)


@pytest.mark.parametrize("n_spill", [1, 2, 6])
def test_a_poisoned_scratch_region_is_fully_re_established(n_spill):
    """The scratch region is persistent storage, so treat it as hostile.

    Between two replays nothing zeroes ``[R:R+C]``. Poisoning it with garbage
    before the step is the strongest available desk stand-in for "the graph
    pool handed this address to something else", and the step must not care.
    """
    cache = _fresh(seed=5)
    R, C = cache.resident_count, cache.scratch
    for attr in ATTRS:
        cache._resident[attr][R : R + C].fill_(-99)
    ids = _routing(n_spill, cache=cache, seed=7)
    remapped = cache.prepare_capturable(ids)
    _assert_routed_rows_are_right(cache, ids, remapped)


def test_the_poison_check_can_fail():
    """The counterfactual for the test above.

    A gather that wrote only the slots it "needed" -- the data-dependent
    optimisation a graph forbids, and the one anybody would reach for after
    reading B4 -- leaves the poison in every other slot. If the assertion above
    cannot see that, it is not testing anything.
    """
    cache = _fresh(seed=5)
    R, C = cache.resident_count, cache.scratch
    for attr in ATTRS:
        cache._resident[attr][R : R + C].fill_(-99)

    def _partial_gather(self, src_row):
        # Write ONE slot instead of all C: the "move only what changed" shape.
        for attr, pool_dev in self._cap_pool_dev.items():
            torch.index_select(
                pool_dev, 0, src_row[:1], out=self._cap_scratch_dst[attr][:1]
            )

    cache._issue_fetch_capturable = _partial_gather.__get__(cache, type(cache))
    ids = _routing(3, cache=cache, seed=7)
    remapped = cache.prepare_capturable(ids)
    with pytest.raises(AssertionError):
        _assert_routed_rows_are_right(cache, ids, remapped)


def test_the_gather_writes_through_the_out_view_into_the_resident_buffer():
    """The graph bakes the ``out=`` pointer, so it must be the GEMM's storage.

    ``_cap_scratch_dst[attr]`` is a slice of ``_resident[attr]`` captured once
    in ``install_capturable_buffers``. ``index_select(..., out=view)`` writes in
    place only while the view's shape matches the result; if torch ever resized
    it, the write would land in fresh storage and the captured graph would keep
    replaying into an orphan while the GEMM read stale weights -- silently, and
    only at serving scale. Pin the in-place property, and the pointer.
    """
    cache = _fresh(seed=11)
    R, C = cache.resident_count, cache.scratch
    before = {attr: cache._cap_scratch_dst[attr].data_ptr() for attr in ATTRS}
    parent = {attr: cache._resident[attr].data_ptr() for attr in ATTRS}
    ids = _routing(4, cache=cache, seed=12)
    cache.prepare_capturable(ids)
    for attr in ATTRS:
        assert cache._cap_scratch_dst[attr].data_ptr() == before[attr], attr
        assert cache._resident[attr].data_ptr() == parent[attr], attr
        # The view really is a window onto the resident buffer, not a copy.
        assert (
            cache._cap_scratch_dst[attr].data_ptr()
            == cache._resident[attr][R : R + C].data_ptr()
        ), attr


def test_the_out_view_check_can_fail():
    """Hand ``index_select`` a wrong-shaped ``out`` and watch what torch does.

    Measured here, not assumed, because the two outcomes need different
    remedies and neither is the intuitive "the write goes somewhere harmless":

    * the resize FITS the shared storage -> torch grows the view in place and
      the gather writes over whatever followed the slice in the resident
      buffer. In-bounds, wrong rows, no error;
    * the resize does NOT fit -> torch reallocates the SHARED storage, so the
      resident buffer's own ``data_ptr`` moves. Under a captured graph that is
      the worst of the two: the graph replays into the address it baked, which
      nothing owns any more.

    Either way the in-place property the test above pins is load-bearing, and
    the shape equality that guarantees it is not decorative.
    """
    C = V4_SHAPE["C"]
    attr = "w2_weight"

    # (1) resize fits: in place, and the neighbours are overwritten.
    cache = _fresh(seed=11)
    R = cache.resident_count
    too_small = cache._resident[attr][R : R + 1]
    ptr_before = cache._resident[attr].data_ptr()
    src_row = torch.zeros(C, dtype=torch.int32)
    with pytest.warns(UserWarning, match="output with one or more elements"):
        torch.index_select(cache._cap_pool_dev[attr], 0, src_row, out=too_small)
    assert too_small.shape[0] == C
    assert cache._resident[attr].data_ptr() == ptr_before
    # Rows R+1..R+C-1 were never asked for and hold the gathered row anyway.
    assert torch.equal(cache._resident[attr][R + 1], cache._pinned[attr][0])

    # (2) resize does not fit: the resident buffer's storage is REPLACED.
    cache = _fresh(seed=11)
    R = cache.resident_count
    at_the_end = cache._resident[attr][R + C - 1 : R + C]
    ptr_before = cache._resident[attr].data_ptr()
    with pytest.warns(UserWarning, match="output with one or more elements"):
        torch.index_select(cache._cap_pool_dev[attr], 0, src_row, out=at_the_end)
    assert at_the_end.shape[0] == C
    assert cache._resident[attr].data_ptr() != ptr_before


@pytest.mark.parametrize("n_spill", [0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("hot", [False, True])
def test_the_ported_step_matches_the_eager_one_at_the_boot_geometry(n_spill, hot):
    """The #443 equivalence claim, re-run at the shape the boot actually ran.

    Its own suite used E=64/R=32/C=8 with 32 routed slots, i.e. always slack
    between the routed slots and the scratch region. The boot ran
    ``bs x top_k == C == 6`` -- no slack at all, and the cumsum rank reaching
    exactly ``C-1`` on the last spill expert. If the index arithmetic had a
    boundary bug, this is where it would be.
    """
    from test_capture_desync_port import _hotset

    E, R = V4_SHAPE["E"], V4_SHAPE["R"]
    resident_slot, spill_pool_index = _hotset(E, R, n_spill) if hot else (None, None)
    over = dict(resident_slot=resident_slot, spill_pool_index=spill_pool_index)
    eager = _fresh(seed=17, **over)
    ported = _fresh(seed=17, **over)
    ids = _routing(n_spill, cache=eager, seed=200 + n_spill)

    remap_eager = eager.prepare_eager(ids)
    remap_ported = ported.prepare_capturable(ids)
    assert torch.equal(remap_eager, remap_ported)
    _assert_routed_rows_are_right(ported, ids, remap_ported)
    # And the eager path agrees about the rows, so neither is a mis-indexed
    # twin of the other.
    _assert_routed_rows_are_right(eager, ids, remap_eager)


# --------------------------------------------------------------------------
# B. B4: the volume a captured step is obliged to move
# --------------------------------------------------------------------------


def _gathered_bytes(cache, ids):
    """Bytes the captured gather reads out of the pinned pool for one step."""
    moved = 0
    real = torch.index_select

    def _counting(src, dim, index, out=None):
        nonlocal moved
        result = real(src, dim, index, out=out)
        moved += result.numel() * result.element_size()
        return result

    torch.index_select = _counting
    try:
        cache.prepare_capturable(ids)
    finally:
        torch.index_select = real
    return moved


def _per_expert_bytes(cache):
    return sum(
        cache._pinned[attr][0].numel() * cache._pinned[attr].element_size()
        for attr in ATTRS
    )


@pytest.mark.parametrize("n_spill", [0, 1, 2, 3, 4, 5, 6])
def test_the_captured_gather_moves_the_worst_case_whatever_the_routing(n_spill):
    """B4's mechanism, in bytes, at the desk.

    The eager fetch moves ``n_spill`` expert rows because it knows, on the
    host, which ones the step missed. The captured gather moves ``C`` rows
    because its index tensor has a static length -- a graph cannot ask how many
    it needs. At the boot's operating point that is 6 rows against a MEASURED
    mean of 1.03-1.50 rows per (layer, forward)
    (``expert_stats_eager.tp{0,1,2}ep0.json``, residency.fetches / forwards).
    """
    eager = _fresh(seed=23)
    ported = _fresh(seed=23)
    ids = _routing(n_spill, cache=eager, seed=300 + n_spill)
    per_expert = _per_expert_bytes(eager)

    eager.prepare_eager(ids)
    eager_bytes = eager.planner.stats.h2d_bytes
    ported_bytes = _gathered_bytes(ported, ids)

    assert eager_bytes == n_spill * per_expert
    assert ported_bytes == V4_SHAPE["C"] * per_expert  # invariant in n_spill


def test_the_volume_claim_can_fail():
    """The counterfactual: a data-dependent gather would break the invariant.

    Written out because the invariant above reads like a tautology until you
    see the alternative fail. ``src_row[:num_spill]`` is exactly the fix a
    reader reaches for after B4 -- and it is precisely the host read
    (``num_spill`` is a device scalar) that makes the step uncapturable again.
    That is why B4 is structural rather than a missed optimisation.
    """
    cache = _fresh(seed=23)
    ids = _routing(2, cache=cache, seed=302)
    per_expert = _per_expert_bytes(cache)

    def _data_dependent_gather(self, src_row):
        n = int((src_row >= 0).sum())  # a HOST read -- illegal under capture
        n = min(n, 2)
        for attr, pool_dev in self._cap_pool_dev.items():
            torch.index_select(
                pool_dev, 0, src_row[:n], out=self._cap_scratch_dst[attr][:n]
            )

    cache._issue_fetch_capturable = _data_dependent_gather.__get__(cache, type(cache))
    moved = _gathered_bytes(cache, ids)
    assert moved == 2 * per_expert
    assert moved != V4_SHAPE["C"] * per_expert


def test_the_measured_multiplier_follows_from_the_static_index_length():
    """Reproduce the boot's ratio from its own numbers, so it is checkable.

    Measured (``expert_stats_eager.tp1ep0.json``, the clock rank): 43 MoE
    layers, mean 8.446 MiB per expert row, 1.133 fetches per (layer, forward).
    The captured step's index tensor is ``int32[C]`` with C = 6.
    """
    layers = 43
    mib_per_expert = 8.446
    eager_fetches_per_layer = 1.133
    C = V4_SHAPE["C"]

    eager_gib = layers * eager_fetches_per_layer * mib_per_expert / 1024
    captured_gib = layers * C * mib_per_expert / 1024

    assert round(eager_gib, 3) == 0.402  # vs 0.398 GiB/token measured
    assert round(captured_gib, 3) == 2.128  # the worst case, every step
    assert round(captured_gib / eager_gib, 2) == 5.30  # vs 6.60x wall clock

    # The static length is the whole cause: the index the gather is given has
    # shape [C] no matter what the routing was.
    cache = _fresh(seed=29)
    from sglang.srt.layers.moe.expert_offload import prepare_capturable_remap

    for n_spill in range(0, C + 1):
        ids = _routing(n_spill, cache=cache, seed=400 + n_spill)
        _, src_row, _ = prepare_capturable_remap(
            ids,
            cache._cap_resident_slot_lut,
            cache._cap_is_spill,
            cache._cap_spill_pool_row_lut,
            cache.resident_count,
            C,
        )
        assert src_row.shape == (C,)


def test_the_captured_path_has_no_copy_stream_to_overlap_with():
    """The residual between 5.30x of volume and 6.60x of wall clock.

    The eager fetch runs its copies on a dedicated stream with fences either
    side (``_fetch``, expert_offload.py), so the H2D overlaps the previous
    wave's GEMM. The captured gather is issued on the CURRENT stream by
    construction -- its docstring makes program order the correctness argument
    for the routed apply reading a complete scratch -- so it serialises against
    the compute it feeds, and it is a zero-copy kernel read over PCIe rather
    than a copy-engine DMA. Pinned structurally, since neither is measurable
    without a card.
    """
    import inspect

    from sglang.srt.layers.moe.expert_offload import MoEExpertOffloadCache

    eager_src = inspect.getsource(MoEExpertOffloadCache._fetch)
    captured_src = inspect.getsource(MoEExpertOffloadCache._issue_fetch_capturable)
    assert "torch.cuda.stream(self._stream)" in eager_src
    assert "wait_stream" in eager_src
    assert "self._stream" not in captured_src
    assert "index_select" in captured_src


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
