"""#514 falsifiers for two silent-wrongness findings in kv-session-offload.

ITEM A (#505-A2-03) -- `try_spill`'s "tail does not fit one host region"
decline compared a RANK-LOCAL quantity (`n_own`, this rank's owned rows of the
tail under the weighted / even DCP owner rule) against the REPLICATED
`region_tokens`, and returned False. Every other decline in that function
rests on replicated inputs and is annotated as such; this one made the SPILL
VERDICT itself rank-dependent -- some ranks fall back to stock retraction while
the others run the spill, which under this fork's doctrine ("RANK-UNIFORMITY:
divergence == NCCL hang, not a wrong number", kv_session_offload.py module
header) is a hang, not a wrong number. The prefill-spill twin RAISES on the
identical predicate.

The verdict has to be the ANY-rank verdict, and it needs no collective: the
`req_to_token` row carries GLOBAL slot ids and is replicated, and `cp_prefix`
is replicated, so every rank can compute EVERY rank's owned count locally --
the same trick `_restore` already uses (`owned_counts_weighted(residues,
self.cp_prefix)` over all ranks, kv_session_offload.py:4874-4914).

ITEM B (#505-A2-01) -- the spec-in-tick draft-read scratch carve-out did
`self.allocator.size -= slices` inside `try: ... except Exception: pass` and
then logged "reserved %d draft-read scratch slots ... allocator.size shrunk to
%d" unconditionally. On the hybrid composites this fork serves
(`UnifiedMambaTokenToKVPoolAllocator`, `UnifiedSWATokenToKVPoolAllocator`)
`size` is a COMPUTED property whose setter is literally `pass`, so the write
vanishes with no exception at all: the slots stay counted in the advertised
capacity while being permanently out of circulation, the stated invariant in
the comment ("so the SchedulerInvariantChecker stays balanced") is broken, and
the log claims success. SUCCESS CLAIMS ARE NOT EVIDENCE (CLAUDE.md): the write
is verified with an independent read-back probe and refuses by name when it did
not take.

CPU-only, no GPU, no server (`CUDA_VISIBLE_DEVICES=99`).
"""

import ast
import inspect
import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.managers.kv_session_offload import (
    KVSessionOffloadManager,
    owned_device_indices,
    spill_tail_rows_max_over_ranks,
)
from sglang.srt.mem_cache.multi_ended_allocator import (
    UnifiedMambaTokenToKVPoolAllocator,
    UnifiedSWATokenToKVPoolAllocator,
)

# Row geometry, shared with test_kvso_reclaim_decline_501.py so the two pin
# files describe the same `try_spill` call.
L = 128
OVERHANG = 4
BLOCK = 8
NEED = 64  # -> boundary 64, tail = row[64:128], 64 slots
POOL_ROWS = 4
POOL_COLS = 256


class _ProceededPastDecline(Exception):
    """Raised from the first side effect AFTER the decline point, so an arm's
    outcome is a clean two-valued classifier: `False` == declined,
    `_ProceededPastDecline` == this rank went on to spill."""


class _SpillReq:
    """Just the request surface `try_spill` reads up to its decline points."""

    def __init__(self, rid, arrival_seq, req_pool_idx, protected=0):
        self.rid = rid
        self.kv_arrival_seq = arrival_seq
        self.req_pool_idx = req_pool_idx
        self.is_fast_lane = False
        self.spill_class = None
        self.to_finish = None
        self.kv_committed_len = L
        self.kv_allocated_len = L + OVERHANG
        self.kv_overallocated_freed = False
        self.cache_protected_len = protected
        self.output_ids = []
        self.origin_input_ids = list(range(L))

    def finished(self):
        return False

    def _cache_commit_len(self):
        return self.kv_committed_len


def _manager(*, mode, S, cp_prefix, dcp_size, dcp_rank, region_tokens):
    """A `KVSessionOffloadManager` carrying only what `try_spill` touches on
    the way to (and just past) the region decline -- no GPU, no scheduler.
    Same bypass idiom as `_bare_manager()` in test_kv_session_offload_unit.py.

    Everything here except `dcp_rank`/`lo`/`hi` is REPLICATED: two managers
    that differ only in those three fields ARE the same iteration on two
    ranks."""
    mgr = KVSessionOffloadManager.__new__(KVSessionOffloadManager)
    mgr.spills = {}
    mgr._free_regions = [0]
    mgr._dest = None
    mgr._iter_ct = 0
    mgr._log = lambda *a, **k: None
    mgr.scheduler = SimpleNamespace(enable_overlap=False)
    mgr._budget_armed = False
    mgr._budget_cooldown = None
    mgr._budget_counters = SimpleNamespace(
        admission_declines=0,
        pendulum_blocked=0,
        pendulum_events=0,
        note_exhaustion=lambda reason: None,
    )
    mgr.budget_session_cap = lambda: 0
    mgr.mode = mode
    mgr.S = S
    mgr.cp_prefix = list(cp_prefix)
    mgr.dcp_size = dcp_size
    mgr.dcp_rank = dcp_rank
    mgr.lo = mgr.cp_prefix[dcp_rank]
    mgr.hi = mgr.cp_prefix[dcp_rank + 1]
    mgr.block_size = BLOCK
    mgr.region_tokens = region_tokens
    mgr.allocator = MagicMock()
    row = torch.arange(POOL_COLS, dtype=torch.int32) + 1000
    # #1040 C1.5: the manager reads `scheduler.req_to_token_pool` at use,
    # so the pool lives on the scheduler stand-in, not on the manager.
    mgr.scheduler.req_to_token_pool = SimpleNamespace(
        req_to_token=row.unsqueeze(0).repeat(POOL_ROWS, 1).contiguous()
    )

    def _boom(*a, **k):
        raise _ProceededPastDecline()

    # First side effect past the decline point (kv_session_offload.py:3492).
    mgr.backend = SimpleNamespace(_sess_open_slot=_boom)
    return mgr


def _batch():
    """Two running sessions; the youngest (index 1) is the spill victim."""
    oldest = _SpillReq("r-old", arrival_seq=1, req_pool_idx=0)
    victim = _SpillReq("r-young", arrival_seq=9, req_pool_idx=1)
    batch = SimpleNamespace(
        reqs=[oldest, victim],
        seq_lens_cpu=torch.tensor([L, L], dtype=torch.int64),
        spec_algorithm=None,  # -> spec_active False -> plain snapshot, no CUDA
        batch_is_full=False,
    )
    return batch, victim


def _verdict(mgr):
    """ "decline" | "spill" for one rank's `try_spill` on the shared batch."""
    batch, _ = _batch()
    try:
        out = mgr.try_spill(batch, fast_pressure=False, need=NEED)
    except _ProceededPastDecline:
        return "spill"
    assert out is False, "the fixture only reaches a decline or the marker"
    return "decline"


# -- ITEM A: the region decline must not diverge across ranks ---------------


def test_region_decline_is_rank_uniform_weighted():
    """Weighted uneven DCP, S=4, ratio [1, 3]: the SAME 64-slot tail is 16
    owned rows on rank 0 and 48 on rank 1. With a 32-row region the pre-fix
    predicate `n_own > region_tokens` says "fits" on rank 0 and "does not fit"
    on rank 1 -- rank 0 spills while rank 1 falls back to stock retraction."""
    ranks = [
        _manager(
            mode="weighted",
            S=4,
            cp_prefix=[0, 1, 4],
            dcp_size=2,
            dcp_rank=r,
            region_tokens=32,
        )
        for r in range(2)
    ]
    verdicts = [_verdict(m) for m in ranks]
    assert len(set(verdicts)) == 1, (
        f"rank-divergent spill verdict {verdicts}: some ranks spill while "
        "others take stock retraction -- the two paths issue different "
        "collectives (kv_session_offload.py module header: divergence == NCCL "
        "hang, not a wrong number)"
    )
    # ANY-rank semantics: rank 1 genuinely does not fit, so nobody spills.
    assert verdicts[0] == "decline"


def test_region_decline_is_rank_uniform_even():
    """Even (modulo) DCP, dcp_size=3: positions [64, 128) split 22/21/21, so a
    21-row region diverges on rank 0 alone. The even rule is only off by one
    across ranks -- that is still a divergence."""
    ranks = [
        _manager(
            mode="even",
            S=3,
            cp_prefix=[0, 1, 2, 3],
            dcp_size=3,
            dcp_rank=r,
            region_tokens=21,
        )
        for r in range(3)
    ]
    verdicts = [_verdict(m) for m in ranks]
    assert len(set(verdicts)) == 1, f"rank-divergent spill verdict {verdicts}"
    assert verdicts[0] == "decline"


def test_region_decline_still_spills_when_every_rank_fits():
    """Spread precondition for the two arms above: with a region that fits the
    WIDEST rank, every rank must still spill -- otherwise the arms could pass
    on a fixture that never spills at all."""
    ranks = [
        _manager(
            mode="weighted",
            S=4,
            cp_prefix=[0, 1, 4],
            dcp_size=2,
            dcp_rank=r,
            region_tokens=48,
        )
        for r in range(2)
    ]
    assert [_verdict(m) for m in ranks] == ["spill", "spill"]


@pytest.mark.parametrize("dcp_rank", [0, 1])
def test_single_rank_path_is_unchanged(dcp_rank):
    """Plain mode (dcp_size == 1) keeps exactly today's behaviour: the max over
    ranks IS this rank's count."""
    mgr = _manager(
        mode="plain", S=1, cp_prefix=[0, 1], dcp_size=1, dcp_rank=0, region_tokens=4
    )
    assert _verdict(mgr) == "decline"  # 64 owned rows > region 4
    mgr = _manager(
        mode="plain", S=1, cp_prefix=[0, 1], dcp_size=1, dcp_rank=0, region_tokens=64
    )
    assert _verdict(mgr) == "spill"


def test_rows_max_over_ranks_matches_the_real_owner_rule():
    """The pure helper must equal the max over the per-rank counts the REAL
    owner function produces -- checked against `owned_device_indices` itself,
    for both non-trivial modes and a scrambled (non-arange) row."""
    torch.manual_seed(514)
    boundary, seq_len = 17, 113
    seg = torch.randint(1000, 9000, (seq_len - boundary,), dtype=torch.int32)

    for cp_prefix, S in ([0, 1, 4], 4), ([0, 2, 5, 9], 9), ([0, 3, 6], 6):
        n = len(cp_prefix) - 1
        per_rank = [
            int(
                owned_device_indices(
                    seg,
                    mode="weighted",
                    S=S,
                    lo=cp_prefix[r],
                    hi=cp_prefix[r + 1],
                    dcp_size=n,
                    dcp_rank=r,
                    pos_offset=boundary,
                )[1].numel()
            )
            for r in range(n)
        ]
        assert spill_tail_rows_max_over_ranks(
            seg,
            mode="weighted",
            S=S,
            cp_prefix=cp_prefix,
            dcp_size=n,
            boundary=boundary,
            L=seq_len,
        ) == max(per_rank)

    for dcp_size in (2, 3, 4, 5):
        per_rank = [
            int(
                owned_device_indices(
                    seg,
                    mode="even",
                    S=dcp_size,
                    lo=r,
                    hi=r + 1,
                    dcp_size=dcp_size,
                    dcp_rank=r,
                    pos_offset=boundary,
                )[1].numel()
            )
            for r in range(dcp_size)
        ]
        assert spill_tail_rows_max_over_ranks(
            seg,
            mode="even",
            S=dcp_size,
            cp_prefix=list(range(dcp_size + 1)),
            dcp_size=dcp_size,
            boundary=boundary,
            L=seq_len,
        ) == max(per_rank)


# -- ITEM B: the draft-scratch carve-out may not silently no-op -------------


def _pass_setter_allocator(size_value):
    """A stand-in for the hybrid composites that carries the REAL `size`
    property object off `UnifiedMambaTokenToKVPoolAllocator` -- getter and
    no-op setter both. Faithful by construction: if the production property
    ever changes, this fixture changes with it."""

    class _Composite:
        size = UnifiedMambaTokenToKVPoolAllocator.size

        def __init__(self):
            self.full_attn_allocator = SimpleNamespace(
                schedulable_available_size=lambda: size_value,
                allocated_count=lambda: 0,
            )
            self.freed = []

        def free(self, idx):
            self.freed.append(idx)

    return _Composite()


class _PlainAllocator:
    """The non-composite case, where `size` is a real attribute and the
    carve-out DOES take effect. Spread precondition for the arms below."""

    def __init__(self, size):
        self.size = size
        self.freed = []

    def free(self, idx):
        self.freed.append(idx)


def test_composite_size_setter_absorbs_the_shrink_silently():
    """The state probe behind the hypothesis: on both hybrid composites the
    production statement `allocator.size -= n` changes nothing AND raises
    nothing. This is why the write needs a read-back and not a `try/except`."""
    for cls in (UnifiedMambaTokenToKVPoolAllocator, UnifiedSWATokenToKVPoolAllocator):
        setter = cls.size.fset
        assert setter is not None
        body = ast.parse(textwrap.dedent(inspect.getsource(setter))).body[0].body
        assert all(isinstance(s, (ast.Pass, ast.Expr)) for s in body), (
            f"{cls.__name__}.size setter is no longer a no-op absorber -- "
            "re-check the carve-out contract"
        )

    alloc = _pass_setter_allocator(4096)
    before = alloc.size
    alloc.size -= 256  # the exact production statement
    assert alloc.size == before, "fixture is not reproducing the silent absorb"


def _carve_manager(alloc, *, slices=256):
    mgr = KVSessionOffloadManager.__new__(KVSessionOffloadManager)
    mgr.allocator = alloc
    mgr.mtp_resident_slices = slices
    mgr.dcp_rank = 0
    mgr._draft_read_scratch = torch.arange(slices, dtype=torch.int64)
    return mgr


def test_carve_out_refuses_by_name_when_the_write_does_not_take():
    """A carve-out that cannot take effect must fail LOUDLY, naming the
    allocator -- never no-op and log success."""
    alloc = _pass_setter_allocator(4096)
    mgr = _carve_manager(alloc)
    with pytest.raises(ValueError) as exc:
        mgr._carve_out_draft_scratch()
    msg = str(exc.value)
    assert "_Composite" in msg, msg  # the allocator is named
    assert "4096" in msg and "256" in msg, msg
    # #501 house rule: a refusal leaves no partial state -- the reserved slots
    # go back to the allocator and the handle is cleared.
    assert len(alloc.freed) == 1
    assert mgr._draft_read_scratch is None


def test_carve_out_applies_on_a_plain_allocator():
    """Spread precondition: the same code path succeeds silently where the
    write DOES take, so the arm above is not passing on a broken instrument."""
    alloc = _PlainAllocator(4096)
    mgr = _carve_manager(alloc)
    mgr._carve_out_draft_scratch()
    assert alloc.size == 4096 - 256
    assert alloc.freed == []
    assert mgr._draft_read_scratch is not None


def test_carve_out_write_is_not_swallowed_and_is_verified():
    """Structural ratchet over the production source, independent of where the
    carve-out lives: the `allocator.size` write may not sit in a bare
    `except: pass`, and the success log may not be reachable without a
    read-back of `allocator.size` and a raise on mismatch."""
    src = textwrap.dedent(inspect.getsource(KVSessionOffloadManager))
    tree = ast.parse(src)

    writers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AugAssign))
        and any(
            isinstance(t, ast.Attribute) and t.attr == "size"
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    assert len(writers) == 1, (
        f"expected exactly one allocator.size write, got {writers}"
    )
    write = writers[0]

    # 1. not swallowed by a bare handler
    swallowers = [
        h
        for h in ast.walk(tree)
        if isinstance(h, ast.Try)
        and any(isinstance(s, ast.Pass) for eh in h.handlers for s in eh.body)
        and any(w is write for w in ast.walk(h))
    ]
    assert not swallowers, (
        "the allocator.size carve-out is wrapped in a `except: pass` -- on a "
        "computed-property allocator the write vanishes with no exception at "
        "all, so the handler hides nothing and the success log lies"
    )

    # 2. the enclosing function reads size back and raises on mismatch
    owner = None
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        if any(w is write for w in ast.walk(fn)):
            owner = fn if owner is None or fn.lineno > owner.lineno else owner
    assert owner is not None
    reads = [
        n.lineno
        for n in ast.walk(owner)
        if isinstance(n, ast.Attribute)
        and n.attr == "size"
        and isinstance(n.ctx, ast.Load)
    ]
    assert any(ln > write.lineno for ln in reads), (
        "no read-back of allocator.size after the carve-out write -- a "
        "framework's success message about state must be backed by an "
        "independent state probe (CLAUDE.md)"
    )
    assert any(isinstance(n, ast.Raise) for n in ast.walk(owner)), (
        "the carve-out never refuses: it can still no-op and log success"
    )
