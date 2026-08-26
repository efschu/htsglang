"""A FAITHFUL census-scheduler double for the #631/#656 flip contract suite.

WHY THIS FILE EXISTS (#905). `#856` ("the flip carries no KV", 9fab2cc62e,
2026-08-24) made the cutover REFUSE rather than degrade: without a scheduler
that can retract its residents and drop its prefix tree, the seam raises
`SeamOrderError` instead of flipping. The production guard is correct. What it
also did, silently, was blind the flip contract suite -- 33 tests across six
files construct `PhaseFlipRuntime` directly and never bound a census scheduler,
because before `#856` there was nothing to bind. Bisected: `9fab2cc62e^` 130
passed / 0 failed, `9fab2cc62e` 98 / 32. From that day until `#905` the flip's
byte identity, seam ordering, consensus discipline and abort deferral were
unverified, and `#856`'s own W27/W30/W31 root defects were found on the rig
instead.

WHY IT IS NOT A `SimpleNamespace`. The cheapest thing that clears the guard is
an object with a `tree_cache` that has a `reset`. That would turn 33 reds into
33 greens that measure nothing -- the `#630` failure mode by name, where
unfaithful stubs modelled a deadline-ignoring wait and hid a livelock. The
`#856` seam is worse than average to stub, because its own history is a
sequence of defects that a convenient double would have PASSED:

    #825  reset() before retract orphans locked nodes -> dec_lock_ref walks
          off the top into None, three ranks down.
    W27   retract_all frees the resources but leaves the Req referenced in
          running_mbs/running_batch/last_batch -- the next consumer is handed
          a live request whose rows are gone.
    W27r  a bare tree.reset() is a BOOKKEEPING reset; it orphaned 152 rows per
          cycle on metal because nothing returned them to the allocator.
    W29   the suite's own tree double had `full_evictable_size_` (attribute)
          where production has `full_evictable_size()` (method), so the read
          returned 0, the eviction was skipped, and ten green tests survived
          the boot it killed.
    W31   the released list was computed and discarded: 78 requests retracted,
          zero completions, every client timing out at 600 s.

Each of those is a property of the MECHANISM, not of the call graph, so this
double models the mechanism:

  * `_SeamTreeNode` has a `parent`, and `reset()` installs a NEW root object.
    A parent walk started before the reset therefore runs off the top exactly
    as `#825` did -- reversing the seam order here CRASHES, it does not merely
    fail an assertion.
  * `full_evictable_size` is a METHOD (the W29 direction), `evict` returns rows
    to a row ledger, and `reset` alone returns nothing -- so a drop that skips
    the eviction leaks rows the ledger can name.
  * batches expose `filter_batch(keep_indices=...)` and carry a parallel
    per-request tensor, so a raw `.reqs` edit desynchronises them and is
    detectable.
  * `readmit_seam_residents` is the REAL `Scheduler` method, bound here
    unmodified (same technique as `test_seam_readmission_w31.py`).

WHAT RUNS FOR REAL when the seam calls into this double, i.e. what these 33
tests actually exercise rather than restate:

    phase_flip_runtime.build_cutover_release
    phase_flip_runtime.release_residents_for_cutover      (the order law)
    phase_flip_runtime.drop_prefix_tree_returning_rows    (evict-then-reset)
    phase_flip_runtime.tree_evictable_full_rows           (the W29 reader)
    phase_flip_runtime.consume_retracted_from_live_universe
    schedule_batch.retract_all -> release_req -> seam_copy_state
    mem_cache.common.release_kv_cache, evict_from_tree_cache
    scheduler.Scheduler.readmit_seam_residents

WHAT IS MODELLED, named so nobody cites this file as evidence of it: the
internals of `Req` (`reset_for_retract`, `offload_kv_cache`) and the radix
cache's node bookkeeping. Those have their own suites (`#783`, `#856`,
`test_tree_drop_returns_rows_856.py`, `test_seam_readmission_w31.py`); what is
modelled here is only enough of them for the seam ORDER to be falsifiable.

The double's own faithfulness is pinned by `test_seam_census_double_905.py`,
which plants the contract violations this file claims to catch.
"""

from __future__ import annotations

import types
from typing import List, Optional

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

#: Rows the fixture's resident requests hold, one block per request. Small and
#: fixed: the number is never the point, the accounting is.
ROWS_PER_RESIDENT = 4


def _ensure_global_server_args() -> None:
    """`release_kv_cache` reads PAGE SIZE and the spec algorithm off the
    GLOBAL server args, not off the ones it is handed.

    A scheduler without global server args cannot exist in production, so a
    double that leaves them unset does not model an instance -- it models a
    process that has not booted, and the retract path raises `ValueError:
    Global server args is not set yet!` there. Set idempotently and with the
    established `ServerArgs(model_path="dummy")` idiom (see
    `test_prefill_adder.py`, `test_chunked_commitment_701.py`), so a module
    that has already set richer ones keeps them.
    """
    from sglang.srt.runtime_context import get_server_args

    try:
        get_server_args()
    except Exception:  # noqa: BLE001 - "not set yet" is the only expected one
        # `page_size` is resolved during a real launch, not by the dataclass,
        # and `release_kv_cache` compares it to 1. Stated here rather than
        # left None so the fixture is a one-token-per-page instance -- the
        # simplest paging that still runs the real alignment arithmetic.
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=1)
        )


class RowLedger:
    """Who owns which device row, as a set the seam must return to empty.

    W27-retry's 152-rows-per-cycle leak is invisible to any double that only
    counts calls: `reset()` was called, `evict` was not, and every assertion
    about "the tree was dropped" still held. A ledger makes the leak the
    OBSERVABLE -- rows that belong to nobody after the seam are the defect.
    """

    def __init__(self, total: int = 4096):
        self.total = total
        self.free = set(range(total))
        self.held = {}  # owner tag -> set of rows

    def alloc(self, owner: str, n: int) -> List[int]:
        rows = sorted(self.free)[:n]
        if len(rows) < n:
            raise AssertionError("row ledger exhausted -- fixture is too small")
        self.free.difference_update(rows)
        self.held.setdefault(owner, set()).update(rows)
        return rows

    def release(self, owner: str, rows=None) -> int:
        owned = self.held.get(owner, set())
        give = owned if rows is None else (owned & set(rows))
        self.free.update(give)
        self.held[owner] = owned - give
        if not self.held[owner]:
            self.held.pop(owner, None)
        return len(give)

    def available_size(self) -> int:
        return len(self.free)

    def charged(self, owner: str) -> int:
        return len(self.held.get(owner, ()))

    def outstanding(self) -> int:
        return sum(len(v) for v in self.held.values())


class ReqSlotPool:
    """The request-slot pool, as a second accounting axis.

    Rows and request slots are different resources released by different legs
    (`cache_finished_req` returns the KV rows, `req_to_token_pool.free(req)`
    returns the slot). Collapsing them into one counter would let a leg that
    silently stopped running hide behind the other one's number.
    """

    def __init__(self, size: int = 64):
        self.size = size
        self.free_slots = set(range(size))
        self.mamba_pool = None
        self.req_to_token = None

    def alloc(self) -> int:
        idx = min(self.free_slots)
        self.free_slots.discard(idx)
        return idx

    def free(self, req) -> None:
        self.free_slots.add(int(req.req_pool_idx))

    def outstanding(self) -> int:
        return self.size - len(self.free_slots)


class _SeamTreeNode:
    """A radix node reduced to the two fields the #825 crash turns on."""

    __slots__ = ("id", "parent", "lock_ref", "rows")

    def __init__(self, node_id, parent=None, rows=()):
        self.id = node_id
        self.parent = parent
        self.lock_ref = 0
        self.rows = list(rows)


class FaithfulTreeCache:
    """The prefix-tree surface the #856 seam actually reads.

    Deliberately implements `full_evictable_size` as a METHOD. The W29 defect
    was a double that had the trailing-underscore ATTRIBUTE instead, which
    `tree_evictable_full_rows` cannot see, so the reader returned None/0, the
    drop skipped its eviction, and the suite stayed green through the boot it
    killed. Getting this backwards again is the single cheapest way to make
    this whole file decorative.
    """

    def __init__(
        self,
        ledger: RowLedger,
        req_to_token_pool: ReqSlotPool,
        *,
        cached_rows: int = 12,
    ):
        self.ledger = ledger
        self.token_to_kv_pool_allocator = ledger
        self.req_to_token_pool = req_to_token_pool
        self.root_node = _SeamTreeNode("root-0")
        self._gen = 0
        self.resets = 0
        self.evict_calls = 0
        self.cached_nodes: List[_SeamTreeNode] = []
        self.dec_lock_calls = 0
        self.cache_finished_calls = 0
        if cached_rows:
            rows = ledger.alloc("tree", cached_rows)
            node = _SeamTreeNode("cached-0", parent=self.root_node, rows=rows)
            self.cached_nodes.append(node)

    # -- BasePrefixCache surface the seam touches --------------------------
    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return True

    def supports_mamba(self) -> bool:
        return False

    def full_evictable_size(self) -> int:
        return sum(len(n.rows) for n in self.cached_nodes if n.lock_ref == 0)

    def evictable_size(self) -> int:
        return self.full_evictable_size()

    def evict(self, params: EvictParams) -> EvictResult:
        """Return rows to the allocator, leaf frontier first.

        THIS is the call that actually pays the allocator back; `reset` below
        does not. A drop that reaches only `reset` leaks exactly what this
        would have returned, and `RowLedger.orphaned` names it.
        """
        self.evict_calls += 1
        want = int(getattr(params, "num_tokens", 0) or 0)
        freed = 0
        for node in list(self.cached_nodes):
            if freed >= want:
                break
            if node.lock_ref != 0:
                # A locked node refuses (#841); the residue is visible to the
                # caller's re-read, which is the point of re-reading.
                continue
            freed += self.ledger.release("tree", node.rows)
            self.cached_nodes.remove(node)
        return EvictResult(num_tokens_evicted=freed)

    def reset(self):
        """A BOOKKEEPING reset: a NEW root, and not one row returned.

        Both halves are the production behaviour this seam had to be taught
        about. The new root is #825's mechanism (every node a parked request
        still holds now has a parent chain that never reaches the live root);
        returning nothing is W27-retry's.
        """
        self._gen += 1
        self.root_node = _SeamTreeNode(f"root-{self._gen}")
        self.resets += 1
        # The old tree is simply dereferenced. Rows its nodes held stay
        # charged to "tree" in the ledger and are now orphaned -- which is
        # precisely what a reset without an evict costs on metal.
        self.cached_nodes = []

    def dec_lock_ref(self, node, *args, **kwargs):
        """hi_mamba_radix_cache.py:1610, reduced to its walk.

        `while node != self.root_node: node = node.parent` -- and running off
        the top into None is the #825 crash verbatim.
        """
        self.dec_lock_calls += 1
        while node is not self.root_node:
            if node is None:
                raise AttributeError("'NoneType' object has no attribute 'id'")
            node.lock_ref -= 1
            node = node.parent
        return None

    def inc_lock_ref(self, node):
        while node is not None and node is not self.root_node:
            node.lock_ref += 1
            node = node.parent
        return None

    def cache_finished_req(self, req, is_insert: bool = True, **kwargs):
        """The one leg `release_kv_cache` reaches on the retract path.

        Releases the request's rows AND its tree lock ref, in that order --
        the lock release is what makes the following `reset()` safe.
        """
        self.cache_finished_calls += 1
        self.ledger.release(f"req:{req.rid}")
        req.rows = []
        if getattr(req, "last_node", None) is not None:
            self.dec_lock_ref(req.last_node)
            req.last_node = None


class FaithfulBatch:
    """A batch that carries per-request tensors alongside its request list.

    `consume_retracted_from_live_universe` deliberately uses
    `filter_batch(keep_indices=...)` rather than mutating `.reqs`, "because a
    batch carries per-request tensors alongside the list and a raw list edit
    desynchronises them". A double whose batch is a bare list cannot tell the
    two apart, so this one keeps a parallel column and `assert_in_sync()`
    is what a raw edit trips.
    """

    def __init__(self, reqs=()):
        self.reqs = list(reqs)
        self.per_req = [id(r) for r in self.reqs]

    def filter_batch(self, keep_indices=None, **kwargs):
        keep = list(keep_indices or [])
        self.reqs = [self.reqs[i] for i in keep]
        self.per_req = [self.per_req[i] for i in keep]

    def assert_in_sync(self):
        if [id(r) for r in self.reqs] != self.per_req:
            raise AssertionError(
                "batch request list and its per-request column desynchronised "
                "-- somebody edited .reqs instead of calling filter_batch"
            )

    def is_empty(self) -> bool:
        return not self.reqs


class FaithfulReq:
    """A resident request, at the surface the retract path reads.

    Its `Req` internals are MODELLED, not real (see the module docstring):
    what has to be faithful here is that it holds rows, holds a tree lock ref,
    and can say whether its prefill is complete -- because those are the three
    facts the seam's ordering law is about.
    """

    def __init__(self, rid: str, arrival: int, ledger: RowLedger, tree, slots):
        self.rid = rid
        self.kv_arrival_seq = arrival
        self.rows = ledger.alloc(f"req:{rid}", ROWS_PER_RESIDENT)
        self.last_node = _SeamTreeNode(f"node-{rid}", parent=tree.root_node)
        tree.inc_lock_ref(self.last_node)
        self.req_pool_idx = slots.alloc()
        self.mamba_pool_idx = None
        self.cache_protected_len = 0
        self.kv_committed_len = ROWS_PER_RESIDENT
        self.kv_spill_state = None
        self.skip_radix_cache_insert = False
        # A decode resident: prefill complete, so `seam_copy_state` COPIES
        # rather than declining. Declining would silently skip the one leg of
        # the retraction that runs while the rows still hold live bytes.
        self.origin_input_ids = list(range(ROWS_PER_RESIDENT))
        self.output_ids = [0]
        self.seqlen = ROWS_PER_RESIDENT + 1
        self.kv_allocated_len = ROWS_PER_RESIDENT
        self.extend_range = types.SimpleNamespace(end=ROWS_PER_RESIDENT)
        self.retraction_count = 0
        self.is_retracted = False
        self.offloaded_at_extent: Optional[int] = None
        self.rows_at_offload: Optional[int] = None
        self.time_stats = types.SimpleNamespace(
            set_wait_queue_entry_time=lambda: None,
            set_retract_time=lambda: None,
        )
        self.seam_readmit_epoch = None
        self.kv_cache_cpu = None
        self.kv_cache_cpu_extent = None

    def finished(self) -> bool:
        return False

    def pop_overallocated_kv_cache(self):
        """No overallocation: a decode resident's row is filled to its
        committed length, which is the state `release_kv_cache` asserts for
        anything but a speculative or thinking-strip build."""
        return (self.kv_committed_len, self.kv_committed_len)

    def pop_committed_kv_cache(self) -> int:
        return self.kv_committed_len

    def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        """The seam's state copy. Records the extent AND how many rows were
        still held, so "the copy ran while the rows still held live bytes"
        is an assertion rather than an assumption."""
        self.offloaded_at_extent = self.seqlen - 1
        self.rows_at_offload = len(self.rows)
        self.kv_cache_cpu = object()
        self.kv_cache_cpu_extent = self.seqlen - 1

    def reset_for_retract(self):
        self.retraction_count += 1
        self.is_retracted = True
        self.extend_range = None
        self.cached_prompt_tokens_at_retract = len(self.origin_input_ids)


class FaithfulCensusScheduler:
    """The scheduler the #856 seam is entitled to find bound at the cutover.

    One instance per rank. `readmit_seam_residents` is the REAL method, so
    the queue discipline these tests rely on (front of queue, as a block, in
    `kv_arrival_seq` order) is the shipped one and not a restatement.
    """

    readmit_seam_residents = Scheduler.readmit_seam_residents

    def __init__(self, *, n_residents: int = 3, rank: int = 0, cached_rows: int = 12):
        _ensure_global_server_args()
        self.ledger = RowLedger()
        self.req_to_token_pool = ReqSlotPool()
        self.tree_cache = FaithfulTreeCache(
            self.ledger, self.req_to_token_pool, cached_rows=cached_rows
        )
        self.server_args = types.SimpleNamespace(
            disaggregation_mode="null",
            enable_hierarchical_cache=False,
        )
        self.token_to_kv_pool_allocator = self.ledger
        self.hisparse_coordinator = None
        self.phase_flip_runtime = None
        self.waiting_queue: List[FaithfulReq] = []
        self.queued: List[tuple] = []
        self.rank = rank
        reqs = [
            FaithfulReq(
                f"r{rank}-{i}",
                i,
                self.ledger,
                self.tree_cache,
                self.req_to_token_pool,
            )
            for i in range(n_residents)
        ]
        # The resident set as production presents it: the PP event loop's
        # per-slot batches are the authority, `running_batch`/`last_batch`
        # mirror one of them. `_live_reqs` deduplicates by identity, and a
        # double that put each request in exactly one place would never
        # exercise that.
        self.running_mbs = [FaithfulBatch(reqs[:1]), FaithfulBatch(reqs[1:])]
        self.running_batch = FaithfulBatch(reqs[1:])
        self.last_batch = FaithfulBatch(reqs[:1])
        self.chunked_req = None
        self._residents = list(reqs)

    # -- the queue authority `readmit_seam_residents` reuses ----------------
    def _add_request_to_queue(self, req, is_retracted: bool = False):
        self.queued.append((req.rid, is_retracted))
        self.waiting_queue.append(req)

    # -- what the fixtures assert on ---------------------------------------
    @property
    def residents(self) -> List[FaithfulReq]:
        return list(self._residents)

    def live_req_count(self) -> int:
        from sglang.srt.managers.phase_flip_runtime import _live_reqs

        return len(_live_reqs(self))

    def batches(self) -> List[FaithfulBatch]:
        out = list(self.running_mbs)
        for b in (self.running_batch, self.last_batch):
            if b is not None:
                out.append(b)
        return out

    def assert_batches_in_sync(self):
        for b in self.batches():
            b.assert_in_sync()

    def orphaned_rows(self) -> int:
        """Rows the allocator has charged out that no owner still names.

        THE POOL-LEAK ARITHMETIC, verbatim in shape: on metal the detector
        prints `total=472864, available=126802, evictable=22, ..., withheld=
        345888` and the shortfall is the leak (152 rows per cycle, W27-retry).
        The same subtraction is done here against the two owners this fixture
        has, so a bookkeeping reset that drops the tree's nodes without
        returning their rows is VISIBLE rather than merely un-asserted.
        """
        gap = self.ledger.charged("tree") - sum(
            len(n.rows) for n in self.tree_cache.cached_nodes
        )
        for req in self._residents:
            gap += self.ledger.charged(f"req:{req.rid}") - len(req.rows)
        return gap

    def seam_carried_bytes(self) -> int:
        """Rows still charged to a request after the seam.

        The no-carry contract's one number: the flip must carry NO KV, so a
        resident holding rows on the far side of the cutover is a carry.
        """
        return sum(
            len(v) for k, v in self.ledger.held.items() if str(k).startswith("req:")
        )


def bind_census_schedulers(runtimes, **kwargs) -> List[FaithfulCensusScheduler]:
    """Bind one faithful census scheduler per rank runtime.

    Named rather than inlined so every fixture in the suite binds the SAME
    double; a per-file stub is how the surfaces drift apart again.
    """
    out = []
    for r, rt in enumerate(runtimes):
        sched = FaithfulCensusScheduler(rank=r, **kwargs)
        sched.phase_flip_runtime = rt
        rt._census_scheduler = sched
        out.append(sched)
    return out
