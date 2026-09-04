"""#501 falsifier: a kv-session-offload spill DECLINE must leave no partial
state on the request.

`try_spill` reclaims the speculative draft overhang of its victim -- it frees
the device slots `[snap.free_from, kv_allocated_len)` and marks the request
with `kv_overallocated_freed = True`. That flag is a one-shot: the stock
`Req.pop_overallocated_kv_cache()` asserts it is still False
(`schedule_batch.py:1141`) and only the spill RESUME path clears it again.

Before #501 the reclaim was committed BEFORE three decline points that each
return False and leave the victim RUNNING in the batch:

  * the spill plan is empty (`partial_spill_plan` returns `want == 0`, which is
    what a fully protected row yields),
  * a #236 budget regler refuses the volume,
  * the owned tail does not fit one host region.

A declined spill therefore armed the assert on a live request, and the
request's eventual stock retraction killed the scheduler instead of falling
back cleanly. These tests drive all three declines and check the request is
left exactly as it was found, including that the REAL
`pop_overallocated_kv_cache` still runs on it afterwards.

CPU-only, no GPU, no server (`CUDA_VISIBLE_DEVICES=99`).
"""

import ast
import inspect
import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.kv_session_offload import KVSessionOffloadManager
from sglang.srt.managers.schedule_batch import Req

# Row geometry shared by the arms: committed length, draft overhang, block.
L = 128
OVERHANG = 4
BLOCK = 8
NEED = 64
POOL_ROWS = 4
POOL_COLS = 256


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
        # Stands in for the real Req helper (which reads global server args);
        # `pop_overallocated_kv_cache` below is the REAL production method.
        return self.kv_committed_len


def _manager(*, region_tokens=4096, budget_armed=False):
    """A `KVSessionOffloadManager` carrying only what `try_spill` touches on
    the way to a decline -- no GPU, no scheduler (bypasses `__init__`), same
    idiom as `_bare_manager()` in test_kv_session_offload_unit.py."""
    mgr = KVSessionOffloadManager.__new__(KVSessionOffloadManager)
    mgr.spills = {}
    mgr._free_regions = [0]
    mgr._dest = None
    mgr._iter_ct = 0
    mgr._log = lambda *a, **k: None
    mgr.scheduler = SimpleNamespace(enable_overlap=False)
    mgr._budget_armed = budget_armed
    mgr._budget_cooldown = None
    mgr._budget_counters = SimpleNamespace(
        admission_declines=0,
        pendulum_blocked=0,
        pendulum_events=0,
        note_exhaustion=lambda reason: None,
    )
    mgr.budget_session_cap = lambda: 0
    # plain owner rule: this rank owns the whole tail (dcp_size == 1)
    mgr.mode = "plain"
    mgr.S = 1
    mgr.cp_prefix = [0, 1]  # single owner class, as __init__ builds it
    mgr.lo = 0
    mgr.hi = 1
    mgr.dcp_size = 1
    mgr.dcp_rank = 0
    mgr.block_size = BLOCK
    mgr.region_tokens = region_tokens
    mgr.allocator = MagicMock()
    row = torch.arange(POOL_COLS, dtype=torch.int32) + 1000
    # #1040 C1.5: the manager reads `scheduler.req_to_token_pool` at use,
    # so the pool lives on the scheduler stand-in, not on the manager.
    mgr.scheduler.req_to_token_pool = SimpleNamespace(
        req_to_token=row.unsqueeze(0).repeat(POOL_ROWS, 1).contiguous()
    )
    return mgr


def _batch(victim_protected=0):
    """Two running sessions; the youngest (index 1) is the spill victim -- the
    oldest is tabu, so a single-request batch would decline for the wrong
    reason."""
    oldest = _SpillReq("r-old", arrival_seq=1, req_pool_idx=0)
    victim = _SpillReq(
        "r-young", arrival_seq=9, req_pool_idx=1, protected=victim_protected
    )
    batch = SimpleNamespace(
        reqs=[oldest, victim],
        seq_lens_cpu=torch.tensor([L, L], dtype=torch.int64),
        spec_algorithm=None,  # -> spec_active False -> plain snapshot, no CUDA
        batch_is_full=False,
    )
    return batch, victim


def _assert_untouched_and_poppable(mgr, victim):
    """The invariant every decline owes the caller: the victim is exactly as it
    was found, and the REAL one-shot `pop_overallocated_kv_cache` still runs on
    it."""
    freed_flag = victim.kv_overallocated_freed
    allocated = victim.kv_allocated_len

    # THE falsifier line. This is the production method, unmodified; before #501
    # a declined spill made it raise on a live request:
    #   AssertionError: Overallocated KV cache already freed,
    #   self.kv_committed_len=128, self.kv_allocated_len=128
    # (schedule_batch.py:1141) -- the scheduler death, instead of the named
    # fallback to stock retraction the decline promises.
    assert Req.pop_overallocated_kv_cache(victim) == (L, L + OVERHANG)
    assert victim.kv_overallocated_freed is True  # the pop is the sole setter

    assert freed_flag is False  # no flag stranded by the decline
    assert allocated == L + OVERHANG  # watermark untouched
    mgr.allocator.free.assert_not_called()  # no slots handed back
    assert mgr._free_regions == [0]  # no host region claimed either


def test_decline_on_empty_spill_plan_leaves_no_reclaim():
    """Decline 1: the whole row is protected, so `partial_spill_plan` wants 0
    tokens and there is nothing to spill."""
    mgr = _manager()
    batch, victim = _batch(victim_protected=L)

    assert mgr.try_spill(batch, fast_pressure=False, need=NEED) is False
    _assert_untouched_and_poppable(mgr, victim)


def test_decline_on_budget_regler_leaves_no_reclaim():
    """Decline 2: a #236 volume/rate regler refuses the spill volume."""
    mgr = _manager(budget_armed=True)
    mgr._budget_admission_check = lambda spill_tokens: "volume"
    batch, victim = _batch()

    assert mgr.try_spill(batch, fast_pressure=False, need=NEED) is False
    _assert_untouched_and_poppable(mgr, victim)


def test_decline_on_tail_exceeding_host_region_leaves_no_reclaim():
    """Decline 3: the owned tail needs more host rows than one region has."""
    mgr = _manager(region_tokens=4)  # tail is NEED == 64 owned rows
    batch, victim = _batch()

    assert mgr.try_spill(batch, fast_pressure=False, need=NEED) is False
    _assert_untouched_and_poppable(mgr, victim)


def test_reclaim_is_committed_after_every_decline_point():
    """Structural ratchet: no `return False` may follow the reclaim commit.

    The three arms above cover today's decline points; this pins the ORDERING
    itself, so a decline added later inside `try_spill` cannot silently move
    back in front of the commit."""
    src = textwrap.dedent(inspect.getsource(KVSessionOffloadManager.try_spill))
    fn = ast.parse(src).body[0]

    commit_lines = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "kv_overallocated_freed"
            for t in node.targets
        )
    ]
    assert len(commit_lines) == 1, "try_spill must have exactly one flag commit"

    free_calls = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "free"
    ]
    assert free_calls, "expected the allocator frees to be found"

    declines = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and node.value.value is False
    ]
    assert declines, "expected try_spill to have decline returns"

    assert max(declines) < commit_lines[0], (
        "a `return False` follows the draft-overhang reclaim commit -- a "
        "declined spill would strand kv_overallocated_freed on a live request "
        "(#501)"
    )
    assert max(declines) < min(free_calls), (
        "a `return False` follows an allocator free -- a declined spill would "
        "leak the victim's slots (#501)"
    )
