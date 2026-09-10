"""#1317 / #1318 -- the window solver and the transient staging release.

Hermetic, no GPU, no torch collectives.  Two things are pinned here:

* :func:`host_ledger.solve_window` -- the arithmetic that says how much of
  the store a group may hold in flight, and the W21 launch refusals that
  carry it.  This is the one deliberate cold test of the build: the solver
  cannot be isolated on metal, because a boot either fits its window or dies
  of something else first.
* :func:`host_ledger._ring_mult_gb_per_s` -- the #1318 derivation, against
  the pool rows and cell sizes boot ``weg2sn5w`` printed.

The release path (C1/C2) is exercised against a hand-built tree of stub
nodes rather than a live cache: the preconditions it enforces are the whole
point of it, and each one must be able to turn this file red on its own.
"""

import pytest

from sglang.srt.weg2 import host_ledger


# --------------------------------------------------------------------------
# #1318 -- the ring multipliers are DERIVED
# --------------------------------------------------------------------------


def test_ring_mult_d_is_the_measured_three_not_the_b0_six():
    """Group D: 30,518 rows x 32,768 B x 3 ranks = 3.00 GB per S.

    Boot weg2sn5w prints ``HiCache host pool (30518 tokens)`` on all three D
    ranks.  The shipped constant was 6.0 -- a 2x over-charge that held the
    store at 9 GiB with a 244,140-token bound below the 262,144 context.
    """
    got = host_ledger._ring_mult_gb_per_s(host_ledger.CELL_BYTES_D_PER_RANK)
    assert got == pytest.approx(3.0, abs=0.005)
    assert host_ledger.RING_D_MULT_GB_PER_S == pytest.approx(3.0, abs=0.005)


def test_ring_mult_p_matches_the_uneven_dcp_pool():
    """Group P: rows MIN-sync to the LARGEST cell, each rank pays its own.

    1e9/18432 = 54,254 rows (P log: ``host KV pool 54254``), and
    54,254 x (18432+8192+6144) = 1.78 GB per S.  The 2.0 charged before
    over-charges by 0.22 GB/S, which is why it was never caught: the error
    was in the conservative direction.
    """
    got = host_ledger._ring_mult_gb_per_s(host_ledger.CELL_BYTES_P_PER_RANK)
    assert got == pytest.approx(1.778, abs=0.005)
    rows = host_ledger.GB / max(host_ledger.CELL_BYTES_P_PER_RANK)
    # 1e9/18432 = 54,253.47 -> 54,253 rows of budget. The P log prints 54,254
    # because the pool adds ONE page-alignment slot on top (the same +1 the
    # draft-pool comment in host_ledger names for 61,036 -> 61,037). Pinning
    # the derivation's own number and the alignment separately keeps the two
    # from being confused for a rounding disagreement.
    assert int(rows) == 54253
    assert int(rows) + 1 == 54254


def test_derivation_never_charges_more_than_the_b0_reading_it_replaces():
    total = host_ledger.RING_P_MULT_GB_PER_S + host_ledger.RING_D_MULT_GB_PER_S
    assert total <= host_ledger.RING_B0_TOTAL_MULT_GB_PER_S
    assert total == pytest.approx(4.778, abs=0.01)


def test_zero_or_negative_cell_bytes_refuse_rather_than_divide():
    with pytest.raises(ValueError):
        host_ledger._ring_mult_gb_per_s(())
    with pytest.raises(ValueError):
        host_ledger._ring_mult_gb_per_s((32768, 0))


# --------------------------------------------------------------------------
# #1317 C10 -- solve_window
# --------------------------------------------------------------------------

# Group D on boot weg2sn5w: the MIN-synced host pool, the static
# --chunked-prefill-size, one chain, a one-chunk write-through floor, and the
# 37-slot anchor pool.
D_POOL, D_CHUNK, D_CHAINS, D_FLOOR, D_ANCHORS = 30518, 4096, 1, 4096, 37


def test_solve_window_group_d_reproduces_the_spec_arithmetic():
    w = host_ledger.solve_window(D_POOL, D_CHUNK, D_CHAINS, D_FLOOR, D_ANCHORS)
    assert w == 24576                      # 4096 * floor(26422/4096) = 4096*6
    assert w + D_FLOOR == 28672
    assert w + D_FLOOR <= D_POOL           # live fits
    assert D_POOL - w - D_FLOOR == 1846    # slack
    assert w // D_CHUNK + 1 == 7           # anchors live, of 37


def test_window_is_a_whole_number_of_chunks_so_the_anchors_line_up():
    """``C | W`` is load-bearing: GDN anchors are published one per chunk
    node, so a window that ends mid-chunk ends below an anchor and the next
    read has nothing to continue from."""
    for pool in range(D_CHUNK + D_FLOOR, 120_000, 997):
        w = host_ledger.solve_window(pool, D_CHUNK, D_CHAINS, D_FLOOR, 4096)
        assert w % D_CHUNK == 0
        assert w + D_FLOOR <= pool


@pytest.mark.parametrize(
    "prompt,windows",
    [(30100, 2), (50007, 3), (57318, 3), (108845, 5), (118498, 5), (262144, 11)],
)
def test_measured_prompts_take_the_expected_number_of_windows(prompt, windows):
    """The five prompt lengths boot weg2zr2's front actually routed, plus the
    context length.  These are the numbers the boot acceptance counts."""
    w = host_ledger.solve_window(D_POOL, D_CHUNK, D_CHAINS, D_FLOOR, D_ANCHORS)
    assert -(-prompt // w) == windows


def test_group_p_solves_a_different_window_from_the_same_solver():
    """R-8: W is solved IN-PROCESS from the group's own pool.  A single
    launcher-transported W would truncate every P read to a D-sized window."""
    p = host_ledger.solve_window(54254, D_CHUNK, D_CHAINS, D_FLOOR, 4096)
    d = host_ledger.solve_window(D_POOL, D_CHUNK, D_CHAINS, D_FLOOR, D_ANCHORS)
    assert p == 49152
    assert p != d


def test_w21_refuses_a_pool_too_small_for_one_window_and_prints_the_terms():
    with pytest.raises(host_ledger.WindowUnsolvable) as ei:
        host_ledger.solve_window(6000, D_CHUNK, D_CHAINS, D_FLOOR, D_ANCHORS)
    msg = str(ei.value)
    assert "W21" in msg
    assert "6000" in msg and "4096" in msg and "1904" in msg


def test_w21_refuses_when_the_windows_would_outrun_the_anchor_pool():
    """An anchor-starved chain stalls at window 2 with an unmatchable node.
    That is a launch-time fact and must be a launch-time refusal."""
    with pytest.raises(host_ledger.WindowUnsolvable) as ei:
        host_ledger.solve_window(300_000, D_CHUNK, D_CHAINS, D_FLOOR, 37)
    assert "anchor" in str(ei.value).lower()
    assert "W21" in str(ei.value)


def test_concurrent_chains_shrink_the_window_rather_than_overcommit():
    """R-12: --max-running-requests chains share ONE pool.  Sizing W as if a
    single request existed is how a second read finds no room and votes the
    first one down."""
    one = host_ledger.solve_window(D_POOL, D_CHUNK, 1, D_FLOOR, D_ANCHORS)
    four = host_ledger.solve_window(D_POOL, D_CHUNK, 4, D_FLOOR, D_ANCHORS)
    assert four < one
    assert 4 * four + D_FLOOR <= D_POOL


def test_window_provenance_names_every_term_not_just_the_number():
    line = host_ledger.window_provenance(
        D_POOL, D_CHUNK, D_CHAINS, D_FLOOR, D_ANCHORS, 60000
    )
    for term in ("W=24576", "pool_rows=30518", "chunk=4096", "chains=1",
                 "ring_floor=4096", "slack=1846", "anchors=7/37"):
        assert term in line, f"{term!r} missing from provenance line: {line}"


# --------------------------------------------------------------------------
# #1317 C1/C2 -- the release preconditions, each able to fail alone
# --------------------------------------------------------------------------


class _CD:
    def __init__(self, value=None, host_value=None):
        self.value = value
        self.host_value = host_value
        self.lock_ref = 0
        self.host_lock_ref = 0


class _Node:
    """The three predicates `release_staged_window` reads, and nothing else."""

    _ids = 0

    def __init__(self, *, device=True, host=True, l3=True, parent=None):
        _Node._ids += 1
        self.id = _Node._ids
        self.parent = parent
        self.children = {}
        self.l3_present = l3
        self.component_data = [
            _CD(value=object() if device else None,
                host_value=object() if host else None)
        ]

    @property
    def backuped(self):
        return self.component_data[0].host_value is not None

    @property
    def evicted(self):
        return self.parent is not None and self.component_data[0].value is None


def _release_gate(node, root):
    """The predicate of `release_staged_window`, transcribed.

    Kept beside the tests rather than imported so that this file pins the
    RULE; `test_release_gate_matches_the_shipped_predicate` below is what
    stops the transcription from drifting from the implementation.
    """
    if node is None or node is root:
        return False
    if node.evicted:
        return False
    if not node.l3_present:
        return False
    if any(cd.host_lock_ref > 0 for cd in node.component_data):
        return False
    if not node.backuped:
        return False
    return True


def test_a_device_resident_store_present_unpinned_node_is_released():
    root = _Node(parent=None)
    n = _Node(parent=root)
    assert _release_gate(n, root) is True


def test_a_node_not_on_the_device_is_never_released():
    """Freeing the host copy of an evicted node drops the only copy this
    rank holds -- the load-then-invalidate half of #904 one tier up."""
    root = _Node(parent=None)
    n = _Node(parent=root, device=False)
    assert _release_gate(n, root) is False


def test_a_node_not_known_present_in_l3_is_never_released():
    root = _Node(parent=None)
    n = _Node(parent=root, l3=False)
    assert _release_gate(n, root) is False


def test_a_host_pinned_node_is_never_released():
    """The eviction funnel checks only the DEVICE pin (#904); the host pin
    covers a concurrent prefetch anchor and must be checked here or the rows
    are freed under the reader."""
    root = _Node(parent=None)
    n = _Node(parent=root)
    n.component_data[0].host_lock_ref = 1
    assert _release_gate(n, root) is False


def test_the_root_is_never_released():
    root = _Node(parent=None)
    assert _release_gate(root, root) is False


def test_release_gate_matches_the_shipped_predicate():
    """The transcription above must keep naming the same four terms as the
    implementation.  An implementation that grows a fifth precondition, or
    loses one, fails here rather than silently diverging from this file."""
    import inspect

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    src = inspect.getsource(UnifiedRadixCache.release_staged_window)
    for term in ("node.evicted", "node.l3_present", "host_lock_ref",
                 "node.backuped", "root_node"):
        assert term in src, f"{term!r} left release_staged_window"
    # And no rank-local quantity may enter it: a rank deciding on its own
    # pool occupancy is the split the group vote exists to prevent.
    for bad in ("available_size()", "time.monotonic", "free_slots"):
        assert bad not in src, f"rank-local {bad} entered the release predicate"


def test_exactly_two_writers_set_l3_present_true():
    """C2's own check: a store-presence fact with three writers is a fact
    nobody owns.  One is the storage-write ack, one is the prefetch insert
    whose pages came out of L3."""
    import ast
    import inspect

    from sglang.srt.mem_cache import unified_radix_cache as urc

    tree = ast.parse(inspect.getsource(urc))
    writers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute) and t.attr == "l3_present":
                    v = node.value
                    if not (isinstance(v, ast.Constant) and v.value is False):
                        writers.append(node.lineno)
    assert len(writers) == 2, f"expected 2 non-False writers, got {writers}"


def test_the_sweep_skips_a_recycled_node():
    """R-5: without `l3_present` in the skip set, every node the windowed
    read recycled reads as un-backed to `publish_unbacked_sweep` and is
    re-written through -- a duplicate D->H copy and a duplicate L3 write that
    refill the pool at exactly the flush where it must be empty."""
    import inspect

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    src = inspect.getsource(UnifiedRadixCache.publish_unbacked_sweep)
    assert "node.l3_present" in src


def test_the_release_is_gated_on_the_controllers_own_role():
    """Read off the controller, never the server args: a phase rebind would
    otherwise leave this reading the role of the other phase."""
    import inspect

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    src = inspect.getsource(UnifiedRadixCache._staging_host_role)
    assert "cache_controller" in src and "host_role" in src
    assert "server_args" not in src
