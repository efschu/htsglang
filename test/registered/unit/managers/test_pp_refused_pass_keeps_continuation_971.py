"""#971: a REFUSED PP pass must re-home the chunked continuation it borrowed.

THE SPECIMEN, boot 1 of window-flip-0828: a 512-void livelock that ran until
the #801-spin guard killed the boot.

    #944 UNRESOLVED told=8192 local=UNKNOWN      x507
    #797d own pass voided                       x0

THE CHAIN, and every link of it is shipped code:

  1. `_get_new_batch_prefill_raw` hands the chunked continuation over --
     `self.chunked_req = adder.add_chunked_req(self.chunked_req)`. On this
     request it is a FINAL extend: prefix 8192 of 8422 fill tokens, so the
     remaining 230 fits inside the 4096 chunk whole, and
     `PrefillAdder.add_chunked_req` takes the non-chunked branch --
     `set_extend_range(8192, 8422)`, `self.can_run_list.append(req)`,
     `return req if truncated else None` with `truncated` False. It returns
     None. `self.chunked_req` is now None and the ADDER holds the only
     reference to the request.
  2. The forwarded-schedule membership check then finds the decision naming a
     rid this rank's admission loop never reached, and raises
     `PPScheduleRefused`.
  3. `get_new_batch_prefill` catches it and calls
     `_pp_refuse_forwarded_schedule`, which voids the pass and returns None.
     The adder is a local of the frame being unwound; it is collected, and
     the request goes with it.

After that the request is in NONE of the four places `pp_request_locations`
enumerates -- not the waiting queue (a chunked continuation is never there),
not `self.chunked_req` (set to None in step 1), not the slot ring, not the
running batch. Every consumer that could have rescued it -- the #943b
re-issue candidate set, the dead-premise sweep, the #944 reconcile -- looks
it up by rid and finds nothing, forever.

THE INVARIANT THIS FILE PINS, stated as a class rather than as this bug:

    Ownership of a chunked continuation may pass from `self.chunked_req`
    into the pass-owned `adder.can_run_list` only if EVERY exit of that pass
    re-homes it.

The successful exit re-homes it into the batch. There are three
`PPScheduleRefused` raise sites in `_get_new_batch_prefill_raw` and all three
are BELOW the handover, so all three dropped it; all three funnel through the
ONE junction, `_pp_refuse_forwarded_schedule`, which is where the restore
belongs and where it now lives.

WHY THE TWO EXISTING RESTORES CANNOT REACH THIS PATH, which is the reason
this was not already fixed by #797b:

  * `_pp_void_own_batch` early-returns on `if batch is None: return False`
    ABOVE its own restore. A refused pass built no batch, so it returns at
    that line every time -- measured as `#797d own pass voided` = 0 across
    the entire boot.
  * `_pp_absorb_void_output`'s restore is keyed on an output expectation that
    a pass which formed nothing never published.

WHAT THIS HARNESS DRIVES FOR REAL, stated plainly because a green from a thin
harness is indistinguishable from a green product:

    REAL: `Scheduler.get_new_batch_prefill` (the try/except funnel),
          `Scheduler._pp_refuse_forwarded_schedule` (the junction under test),
          `PrefillAdder.add_chunked_req` (the actual ownership handover, on a
          real `PrefillAdder` against a real `Req`),
          `PPScheduleRefused`, `void_pp_admission_decision`,
          `pp_rehome_refused_chunked_req`, `_park_chunked_prefill_chunk`,
          `pp_request_locations`.

    STOOD IN: `_get_new_batch_prefill_raw` itself. Nothing in this tree drives
          that function -- it is ~700 lines and builds its own `PrefillAdder`,
          tree cache, pools and mamba budget from ~40 scheduler attributes, and
          a harness that large is likelier to manufacture a false green than to
          prove anything. The stand-in performs the two production steps that
          define this defect, in production order: the real handover, then the
          real refusal. `TheHarnessMatchesProduction` below pins that fidelity
          against the shipped source, so the stand-in cannot drift away from
          the function it stands in for without going red.
"""

import inspect
import types
import unittest
from array import array
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.pp_admission_congruence import (
    PPAdmissionDecision,
    PPAdmissionEntry,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.utils.common import Range

try:
    from sglang.srt.managers.schedule_policy import PPScheduleRefused
except ImportError:  # the exception lives beside the guard, not the policy
    from sglang.srt.managers.pp_admission_congruence import PPScheduleRefused


#: The specimen's geometry, verbatim from the boot log.
PREFIX_DONE = 8192
TOTAL_FILL = 8422
CHUNK_SIZE = 4096
#: 8422 - 8192 = 230, and 230 < 4096 -- so this is the FINAL extend, the
#: branch on which `add_chunked_req` returns None and keeps the request.
REMAINING = TOTAL_FILL - PREFIX_DONE

RID_CHUNKED = "chunked-continuation-971"
#: The rid the upstream's decision names and this rank's loop never reached.
#: Its presence is what makes the membership check refuse the pass.
RID_UNREACHED = "named-but-unadmitted-971"

MB_ID = 1
RING_SIZE = 3
WORLD = 3


def _real_req(rid=RID_CHUNKED, prefix_len=PREFIX_DONE, total=TOTAL_FILL):
    """A real `Req`, in the state the specimen's continuation was in.

    `Req.__new__` rather than a stand-in class: `add_chunked_req` calls the
    real `set_extend_range` and reads `full_untruncated_fill_ids` /
    `prefix_indices` to derive the chunk, and those derivations are exactly
    what the handover under test consists of. A stub would answer the
    question with the harness's own arithmetic instead of the product's.
    """
    req = Req.__new__(Req)
    req.rid = rid
    fill = list(range(total))
    req.origin_input_ids = array("q", fill)
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", fill)
    # A TENSOR of KV-pool slot pointers, as the real `Req` carries (#796: this
    # object's type is load-bearing -- the release path slices it).
    req.prefix_indices = torch.arange(prefix_len, dtype=torch.int64)
    req.req_pool_idx = 0
    # The PREVIOUS chunk's range, already computed and stashed. The handover
    # is what overwrites this with the never-to-be-run (8192, 8422).
    req.extend_range = Range(prefix_len - CHUNK_SIZE, prefix_len)
    req.inflight_middle_chunks = 1
    req.host_hit_length = 0
    req.swa_host_hit_length = 0
    req.cache_protected_len = prefix_len
    req.skip_radix_cache_insert = False
    req.last_node = "node"
    req.best_match_node = "node"
    req.swa_uuid_for_lock = None
    req.session = None
    req.return_logprob = False
    req.logprob_start_len = -1
    req.positional_embed_overrides = None
    req.extra_key = None
    req.mamba_pool_idx = None
    req.kv_spill_state = None
    req.retracted_stain = False
    req.born_spilled = False
    req.born_spilled_deep = False
    req.sampling_params = types.SimpleNamespace(max_new_tokens=16, ignore_eos=False)
    return req


def _real_adder(scheduled_extents):
    """A real `PrefillAdder` with the specimen's chunk width.

    Only the cache and allocator are mocked, and only for capacity answers --
    `add_chunked_req`'s arithmetic, its budget update and its return value are
    the product's own. Mirrors `test_pp_forwarded_schedule_791._adder`.
    """
    from sglang.srt.managers.schedule_policy import PrefillAdder
    from sglang.srt.mem_cache.base_prefix_cache import (
        DecLockRefResult,
        IncLockRefResult,
    )
    from sglang.srt.server_args import (
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    tree_cache = MagicMock()
    tree_cache.disable = False
    tree_cache.evictable_size.return_value = 1 << 20
    tree_cache.full_evictable_size.return_value = 1 << 20
    tree_cache.swa_evictable_size.return_value = 0
    tree_cache.inc_lock_ref.return_value = IncLockRefResult()
    tree_cache.dec_lock_ref.return_value = DecLockRefResult()
    tree_cache.is_tree_cache.return_value = False

    allocator = MagicMock()
    allocator.available_size.return_value = 1 << 20
    allocator.full_available_size.return_value = 1 << 20
    allocator.swa_available_size.return_value = 0

    running_batch = MagicMock()
    running_batch.reqs = []

    return PrefillAdder(
        page_size=1,
        tree_cache=tree_cache,
        token_to_kv_pool_allocator=allocator,
        running_batch=running_batch,
        new_token_ratio=1.0,
        rem_input_tokens=1 << 20,
        rem_chunk_tokens=CHUNK_SIZE,
        scheduled_extents=scheduled_extents,
    )


class _StubPool:
    """`req_to_token_pool`, sized to the specimen's request."""

    def __init__(self, rows=1, cols=TOTAL_FILL):
        self.req_to_token = torch.arange(rows * cols, dtype=torch.int64).view(rows, cols)
        self.freed_req = []

    def free(self, req):
        self.freed_req.append(req)


class _StubAllocator:
    """Records every KV release, so the accounting arm can name the range."""

    def __init__(self):
        self.freed = []

    def free(self, indices):
        self.freed.append(indices.clone())


def _holder(chunked_req, *, ring_shape="list", amended=True):
    """A scheduler stand-in carrying exactly what the refusal path reads.

    The #630/#757/#795/#797 pattern this directory uses throughout: a bare
    `SimpleNamespace` with the REAL methods bound onto it via
    `types.MethodType`, so the code under test is the shipped code.
    """
    pool = _StubPool()
    h = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=1, pp_size=WORLD),
        pp_group=types.SimpleNamespace(is_first_rank=False, is_last_rank=False),
        chunked_req=chunked_req,
        waiting_queue=[],
        running_batch=types.SimpleNamespace(reqs=[]),
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=_StubAllocator(),
        # The funnel's own prologue: falsy, so the delayer branch is skipped.
        prefill_delayer=None,
        # Set by the pass before admission, and never cleared by the refusal.
        _pp_output_expected_incoming=False,
        _pp_live_mb_id=MB_ID,
        _admission_decline_note=None,
        noted_expectations=[],
    )

    # #797b: the pre-admission snapshot, written every pass by
    # `_pp_note_chunked_req_before_admission` under the loop's own `mb_id`.
    # PRODUCTION SHAPE IS A LIST (`init_pp_loop_state`'s `[None] *
    # pp_loop_size`), and this fixture uses that shape deliberately: a dict
    # here would exercise a container production never builds.
    if ring_shape == "list":
        ring = [None] * RING_SIZE
        ring[MB_ID] = chunked_req
    elif ring_shape == "dict":
        ring = {MB_ID: chunked_req}
    else:
        ring = None
    h._pp_chunked_req_before_by_slot = ring

    if amended:
        h._pp_admission_amended_to_forward = PPAdmissionDecision(
            mb_id=MB_ID,
            entries=(
                PPAdmissionEntry(
                    rid=RID_UNREACHED,
                    prefix_len=0,
                    extend_len=64,
                    admitted=True,
                ),
            ),
        )
    else:
        h._pp_admission_amended_to_forward = None

    h._pp_note_output_expectation = lambda mb_id, expected, decision: (
        h.noted_expectations.append((mb_id, expected))
    )
    h._trace_pp_admission_verdict = lambda ret: None
    h._pp_refuse_forwarded_schedule = types.MethodType(
        Scheduler._pp_refuse_forwarded_schedule, h
    )
    h.get_new_batch_prefill = types.MethodType(Scheduler.get_new_batch_prefill, h)
    return h


def _refused_pass(h):
    """Drive one pass that hands the continuation over and is then refused.

    The stand-in for `_get_new_batch_prefill_raw` performs the two production
    steps this defect is made of, in production order and with production
    code:

      1. THE HANDOVER, `self.chunked_req = adder.add_chunked_req(...)` on a
         real `PrefillAdder`. The forwarded schedule does NOT name this rid,
         so the adder takes its rank-local branch, writes the never-to-be-run
         `extend_range(8192, 8422)`, appends the request to `can_run_list`
         and returns None -- exactly `schedule_policy.py`'s final-extend
         return.
      2. THE REFUSAL, the real `PPScheduleRefused` the membership check
         raises when the decision names a rid the loop did not reach.

    The adder is a local of THIS frame, exactly as it is a local of the
    production frame -- so when the exception unwinds it, the only surviving
    reference to the request is whatever the junction re-homed.
    """
    scheduled = {RID_UNREACHED: (0, 64)}
    captured = {}

    def _raw(*, prefill_delayer_single_pass, running_batch):
        adder = _real_adder(scheduled)
        h.chunked_req = adder.add_chunked_req(h.chunked_req)
        captured["handed_over"] = h.chunked_req is None
        captured["can_run_list"] = list(adder.can_run_list)
        raise PPScheduleRefused(
            f"#791 FORWARDED SCHEDULE UNEXECUTABLE: the decision names "
            f"1 request(s) and this rank's admission loop reached only 1; "
            f"missing rid(s)={RID_UNREACHED}."
        )

    h._get_new_batch_prefill_raw = _raw
    plan = h.get_new_batch_prefill(running_batch=h.running_batch)
    return plan, captured


def _locations(h):
    from sglang.srt.managers.scheduler_pp_mixin import pp_request_locations

    return pp_request_locations(h)


class TheHandoverReallyHappens(unittest.TestCase):
    """The premise of every arm below. If the handover did not occur, a green
    on "the request is still reachable" would be vacuous."""

    def test_the_final_extend_hands_the_request_to_the_adder_and_returns_none(self):
        req = _real_req()
        adder = _real_adder({RID_UNREACHED: (0, 64)})
        self.assertIsNone(
            adder.add_chunked_req(req),
            "the specimen's geometry is a FINAL extend -- 230 remaining "
            "inside a 4096 chunk -- so `add_chunked_req` returns None and "
            "`self.chunked_req` becomes None at the call site",
        )
        self.assertIn(
            req,
            adder.can_run_list,
            "and the ONLY surviving reference is the pass-owned adder",
        )
        self.assertEqual(req.extend_range.start, PREFIX_DONE)
        self.assertEqual(
            req.extend_range.end,
            TOTAL_FILL,
            "a chunk that no rank will ever run is now written on the request",
        )
        self.assertEqual(req.extend_range.end - req.extend_range.start, REMAINING)


class TheSiblingVoidSitesAreUnchanged(unittest.TestCase):
    """#971 widened `_park_chunked_prefill_chunk` with a `pass_allocated`
    keyword. Every pre-#971 caller undoes a pass whose batch WAS built and
    must keep both give-backs -- the default is what guarantees that, so the
    default is pinned here rather than assumed."""

    def _park(self, req, **kw):
        from sglang.srt.managers.scheduler_pp_mixin import (
            _park_chunked_prefill_chunk,
        )

        h = types.SimpleNamespace(
            req_to_token_pool=_StubPool(),
            token_to_kv_pool_allocator=_StubAllocator(),
        )
        parked = _park_chunked_prefill_chunk(h, req, **kw)
        return h, parked

    def test_the_default_still_releases_kv_and_gives_back_the_increment(self):
        req = _real_req()
        req.extend_range = Range(PREFIX_DONE, TOTAL_FILL)
        h, parked = self._park(req)
        self.assertTrue(parked)
        self.assertEqual(
            len(h.token_to_kv_pool_allocator.freed),
            1,
            "a BUILT pass allocated those rows and still owes them back",
        )
        self.assertEqual(h.token_to_kv_pool_allocator.freed[0].numel(), REMAINING)
        self.assertEqual(req.inflight_middle_chunks, 0, "increment given back")
        self.assertEqual(req.extend_range.end, PREFIX_DONE, "and parked")

    def test_pass_allocated_false_parks_the_geometry_and_nothing_else(self):
        req = _real_req()
        req.extend_range = Range(PREFIX_DONE, TOTAL_FILL)
        h, parked = self._park(req, pass_allocated=False)
        self.assertTrue(parked)
        self.assertEqual(h.token_to_kv_pool_allocator.freed, [])
        self.assertEqual(req.inflight_middle_chunks, 1)
        self.assertEqual(req.extend_range.end, PREFIX_DONE, "geometry still parked")

    def test_it_is_idempotent_so_a_later_void_site_cannot_double_free(self):
        """`_pp_void_own_batch` may run after the refusal in the same pass. On
        an already-parked request `end == start`, so `prepared` is False and
        neither give-back fires a second time."""
        req = _real_req()
        req.extend_range = Range(PREFIX_DONE, TOTAL_FILL)
        self._park(req, pass_allocated=False)
        h, _ = self._park(req)
        self.assertEqual(h.token_to_kv_pool_allocator.freed, [])
        self.assertEqual(req.inflight_middle_chunks, 1)


class ARefusedPassKeepsTheChunkedContinuation(unittest.TestCase):
    """ARM 1 -- THE LOSS, through the real funnel and the real junction."""

    def test_the_pass_is_refused_and_voided(self):
        """The harness really reaches the defect, before anything is asserted
        about the repair."""
        h = _holder(_real_req())
        plan, captured = _refused_pass(h)
        self.assertTrue(captured["handed_over"], "the handover really happened")
        self.assertIsNone(plan.batch_to_run, "a refused pass builds no batch")
        self.assertTrue(h._pp_admission_pass_voided, "and it voids the pass")
        self.assertEqual(h._pp_admission_incoming_schedule, {})

    def test_the_continuation_is_still_reachable_after_the_refusal(self):
        """THE #971 DEFECT, in one assertion.

        507 rounds of `#944 UNRESOLVED told=8192 local=UNKNOWN` are what this
        assertion failing looks like on metal: every consumer resolves
        requests by rid through `pp_request_locations`, and after the refusal
        the continuation answered in none of the four places.
        """
        req = _real_req()
        h = _holder(req)
        _refused_pass(h)
        got = _locations(h)
        self.assertIn(
            RID_CHUNKED,
            got,
            "THE #971 DEFECT: the refusal handed `self.chunked_req` to the "
            "adder and then discarded the adder. `_pp_void_own_batch`'s "
            "restore is unreachable (it early-returns on `batch is None`, "
            "which is why `#797d own pass voided` is 0 in the whole boot "
            "log) and `_pp_absorb_void_output`'s needs an output expectation "
            "this pass never made. Ownership passed out of scheduler-owned "
            "state and no exit gave it back.",
        )
        self.assertIs(got[RID_CHUNKED], req, "and it is the SAME object")

    def test_it_is_restored_to_scheduler_owned_state_not_merely_findable(self):
        """`self.chunked_req` specifically -- the slot ring alone is not a
        home. `add_chunked_req` re-admits the continuation from
        `self.chunked_req` directly, so a request reachable only through the
        ring is still one the next round will not continue."""
        req = _real_req()
        h = _holder(req)
        _refused_pass(h)
        self.assertIs(
            h.chunked_req,
            req,
            "the next round continues `self.chunked_req`; nothing else",
        )


class TheRestoredChunkIsParked(unittest.TestCase):
    """ARM 2 -- GEOMETRY. Restoring the request without parking its range
    would leave the next round caching a chunk that never executed."""

    def test_extend_range_survives_the_refusal(self):
        """instr19's crash is the other direction of this same field: a
        restore that nulls `extend_range` makes the next round dereference
        `self.chunked_req.extend_range.end` on None."""
        h = _holder(_real_req())
        _refused_pass(h)
        self.assertIsNotNone(h.chunked_req.extend_range)

    def test_the_never_run_chunk_is_parked_so_the_next_stash_is_a_no_op(self):
        """THE PARK, and why the restore is incomplete without it.

        `add_chunked_req` already wrote `extend_range=(8192, 8422)` for a
        chunk no rank will run. The next round's stash site is
        UNCONDITIONAL -- `if self.chunked_req.extend_range.end >
        len(self.chunked_req.prefix_indices): self.stash_chunked_request(...)`
        -- so an unparked restore caches a chunk that never executed. The
        parked shape is the one that site already documents as its own no-op.
        """
        h = _holder(_real_req())
        _refused_pass(h)
        req = h.chunked_req
        self.assertEqual(
            req.extend_range.end,
            len(req.prefix_indices),
            "parked: `extend_range.end == len(prefix_indices)`, which is "
            "exactly the state the stash site no-ops on",
        )
        self.assertFalse(
            req.extend_range.end > len(req.prefix_indices),
            "the next round's unconditional stash must not fire on a chunk "
            "that never ran",
        )


class TheRefusalDiscardsNoTokens(unittest.TestCase):
    """ARM 5 -- ACCOUNTING. Kein-Doppel-Prefill: this junction must cost zero
    prefilled tokens. The status quo cost all 8192."""

    def test_nothing_is_released_because_this_pass_allocated_nothing(self):
        """NO KV RELEASE AT ALL on this path, and that is the exact opposite
        of what the sibling void sites owe.

        Every pre-#971 caller of `_park_chunked_prefill_chunk` undoes a pass
        whose batch was BUILT: `prepare_for_extend` allocated the chunk's rows,
        so freeing `req_to_token[.., 8192:8422]` returns pages that pass really
        took. A REFUSED pass is raised above that allocation -- the membership
        check fires before the batch is ever constructed -- so those columns
        still hold STALE pointers belonging to whatever request last occupied
        them. Freeing them would be the "double free, not a leak" the park's
        own docstring warns about, committed against a third party.

        So the refusal path takes the geometry park and neither give-back.
        """
        h = _holder(_real_req())
        _refused_pass(h)
        self.assertEqual(
            h.token_to_kv_pool_allocator.freed,
            [],
            "a pass that allocated nothing may release nothing -- the "
            "[8192:8422] columns were never this pass's to give back",
        )
        self.assertEqual(
            h.req_to_token_pool.freed_req, [], "and no req-pool row is freed"
        )

    def test_the_prefix_the_request_keeps_is_the_full_cached_prefix(self):
        h = _holder(_real_req())
        _refused_pass(h)
        self.assertEqual(
            len(h.chunked_req.prefix_indices),
            PREFIX_DONE,
            "zero tokens lost: the continuation resumes from 8192, not 0",
        )

    def test_the_request_is_not_retracted_and_keeps_its_tree_handles(self):
        """A void is a PARK, not a retraction (#797b). `reset_for_retract`
        would throw away `prefix_indices` and `last_node` -- the handles every
        already-stashed chunk is held by."""
        h = _holder(_real_req())
        _refused_pass(h)
        self.assertEqual(h.chunked_req.last_node, "node")
        self.assertFalse(getattr(h.chunked_req, "is_retracted", False))

    def test_the_inflight_count_is_left_exactly_as_it_was(self):
        """NO give-back either, because no increment was taken.

        `self.chunked_req.inflight_middle_chunks += 1` lives in
        `_get_new_batch_prefill_raw` BELOW the membership raise, so a refused
        pass never took it. Decrementing here would be the mirror of the leak
        the park exists to prevent: the park's own docstring calls a leaked
        increment "a request that can never report finished", and a spurious
        give-back is a request that reports finished while a chunk is still
        owed. The count belongs to the round that really took it.
        """
        h = _holder(_real_req())
        _refused_pass(h)
        self.assertEqual(
            h.chunked_req.inflight_middle_chunks,
            1,
            "unchanged: this pass never incremented it, so it owes nothing",
        )


class TheRestoreIsWhatMakesTheGreen(unittest.TestCase):
    """ARM 3 -- THE CAN-FAIL PROOF, and the reason the arms above are worth
    reading.

    ONE return value is neutered, through `scheduler_pp_mixin`'s own module
    globals and in the idiom this directory already uses for
    `pp_void_keeps_request`: `pp_rehome_refused_chunked_req` becomes a no-op,
    which is EXACTLY the behaviour that shipped before it existed. Everything
    else still runs its own body -- the pre-admission snapshot is still
    written, the real adder still performs the handover, the real junction
    still voids the pass. A wholesale revert would fail somewhere in the
    harness and prove nothing.
    """

    def _blinded_pass(self, h):
        from sglang.srt.managers import scheduler_pp_mixin as m

        original = getattr(m, "pp_rehome_refused_chunked_req", None)
        self.assertIsNotNone(
            original,
            "#971's restore must be a module-global in `scheduler_pp_mixin`, "
            "so that a can-fail proof can neuter THAT ONE return value "
            "without reverting the harness with it",
        )
        m.pp_rehome_refused_chunked_req = lambda scheduler, mb_id: False
        try:
            return _refused_pass(h)
        finally:
            m.pp_rehome_refused_chunked_req = original

    def test_arm1_fails_when_only_the_restore_is_neutered(self):
        req = _real_req()
        h = _holder(req)
        self._blinded_pass(h)
        self.assertIsNone(
            h.chunked_req,
            "ARM 1's decisive assertion must FAIL without the restore -- "
            "otherwise its green is the harness's, not the product's",
        )

    def test_the_slot_ring_alone_is_not_a_home(self):
        """THE TWO FIXES ARE NOT SUBSTITUTES, and this measures the seam.

        With the restore neutered, the sibling fix (`pp_request_locations`
        reading the production LIST ring) still resolves the rid -- the
        pre-admission snapshot is in the ring, and that is a real gain: the
        #943b re-issue and the dead-premise sweep can now nominate the
        request where before they saw nothing.

        It is NOT the repair. `add_chunked_req` re-admits the continuation
        from `self.chunked_req` DIRECTLY, and the ring slot is overwritten
        every pass by `_pp_note_chunked_req_before_admission`. So a request
        reachable only through the ring is visible and still not continued --
        findable is not continuable. Only the restore puts it back where the
        next round will act on it.
        """
        req = _real_req()
        h = _holder(req)
        self._blinded_pass(h)
        self.assertIn(
            RID_CHUNKED,
            _locations(h),
            "the sibling fix alone makes the request VISIBLE",
        )
        self.assertIsNone(
            h.chunked_req,
            "and visible is not restored -- the next round continues "
            "`self.chunked_req`, which is still None",
        )

    def test_arm2_fails_when_only_the_restore_is_neutered(self):
        req = _real_req()
        h = _holder(req)
        self._blinded_pass(h)
        self.assertEqual(
            req.extend_range.end,
            TOTAL_FILL,
            "blinded, the never-run chunk stays unparked on the request and "
            "the next round's unconditional stash would cache it",
        )


class TheSlotRingIsReadInItsProductionShape(unittest.TestCase):
    """ARM 4 -- THE SIBLING, independently valuable.

    `pp_request_locations` read the slot ring with `.values()` inside `try/
    except AttributeError`. Production builds that ring as a LIST
    (`init_pp_loop_state`'s `[None] * pp_loop_size`), a list has no
    `.values()`, and the except swallowed the shape question -- so the slot
    place was silently empty for EVERY consumer of the helper: the dead-
    premise sweep, the ring probe, and scheduler.py's #943b re-issue
    candidate set. Three consumers saw three places where the docstring
    promises four.
    """

    def test_a_slot_only_request_resolves_against_the_production_list_ring(self):
        req = _real_req()
        h = _holder(None, ring_shape="list")
        h._pp_chunked_req_before_by_slot = [None] * RING_SIZE
        h._pp_chunked_req_before_by_slot[MB_ID] = req
        self.assertIn(
            RID_CHUNKED,
            _locations(h),
            "THE SILENT HALF: the ring is a LIST in production, `.values()` "
            "raised AttributeError on it, and the except read that as 'a "
            "stand-in that is not a mapping' rather than as the real shape",
        )

    def test_the_mapping_shape_still_resolves(self):
        """The fix answers the shape question; it does not trade one shape
        for the other."""
        req = _real_req()
        h = _holder(None, ring_shape="dict")
        h._pp_chunked_req_before_by_slot = {MB_ID: req}
        self.assertIn(RID_CHUNKED, _locations(h))

    def test_a_holder_with_no_ring_still_never_raises(self):
        """This mapping feeds `take_agreed_reissue`, a COLLECTIVE. A rank that
        raised while building its candidate set would skip an all_reduce its
        peers had already entered (#787)."""
        self.assertEqual(_locations(types.SimpleNamespace()), {})
        h = _holder(None, ring_shape="none")
        self.assertEqual(_locations(h), {})

    def test_a_non_iterable_stand_in_ring_still_never_raises(self):
        h = _holder(None, ring_shape="none")
        h._pp_chunked_req_before_by_slot = object()
        self.assertEqual(_locations(h), {})


class TheSkipCensusSurvivesTheRefusal(unittest.TestCase):
    """FIX 3 -- the instrument must be readable on the path it explains.

    `_admission_decline_note` was written only inside the `len(can_run_list)
    == 0` branch, which every `PPScheduleRefused` raise jumps over. So the
    #788 verdict trace printed `reason=-` for exactly the passes whose local
    narrowing caused the refusal -- the instrument was blind on its own
    subject.

    Asserted against the shipped SOURCE rather than by driving the function:
    the census is a local closure over a local `_skips` dict inside the
    ~700-line `_get_new_batch_prefill_raw`, and standing that whole function
    up to read one string back would be a harness far larger than the fact it
    checks.
    """

    def _raw_source(self):
        return inspect.getsource(Scheduler._get_new_batch_prefill_raw)

    def test_the_census_is_written_before_the_first_refusal_raises(self):
        src = self._raw_source()
        first_raise = src.index("raise PPScheduleRefused")
        # THE CENSUS WRITE SPECIFICALLY, not any touch of the attribute. The
        # function opens with `self._admission_decline_note = None`, a per-pass
        # RESET -- counting that as a write would make this test green on the
        # unfixed tree, which is the exact shape of a green for the wrong
        # reason. The census is identified by its payload (`loop_skips(` /
        # `loop=clean`) or by the closure that emits it.
        note_writes = [
            i
            for i in range(len(src))
            if src.startswith("_write_skip_census()", i)
            or src.startswith('"loop_skips("', i)
            or src.startswith('"loop=clean"', i)
        ]
        self.assertTrue(note_writes, "the census must be written somewhere")
        self.assertLess(
            min(note_writes),
            first_raise,
            "THE BLIND INSTRUMENT: every skip-census write sat BELOW the "
            "membership raise, in the `len(can_run_list) == 0` branch the "
            "refusal jumps over -- so a refused pass logged `reason=-`, on "
            "the one path the census exists to explain",
        )


class TheHarnessMatchesProduction(unittest.TestCase):
    """The stand-in's fidelity, pinned against the shipped source.

    `_refused_pass` stands in for `_get_new_batch_prefill_raw` with two steps:
    the handover, then the refusal. These assertions fail if production ever
    stops matching that shape -- which is what keeps the stand-in honest
    instead of merely convenient.
    """

    def _raw_source(self):
        return inspect.getsource(Scheduler._get_new_batch_prefill_raw)

    def test_production_hands_the_continuation_to_the_adder(self):
        self.assertIn(
            "self.chunked_req = adder.add_chunked_req(self.chunked_req)",
            self._raw_source(),
            "the handover the stand-in reproduces must still be the shipped "
            "one",
        )

    def test_every_refusal_raise_is_below_the_handover(self):
        """THE INVARIANT'S SCOPE. Each raise site below the handover is an
        exit that leaves the continuation adder-owned. If a raise ever moves
        ABOVE the handover this assertion goes red and the junction's
        obligation must be re-derived for it."""
        src = self._raw_source()
        handover = src.index("self.chunked_req = adder.add_chunked_req")
        raises = [
            i for i in range(len(src)) if src.startswith("raise PPScheduleRefused", i)
        ] + [i for i in range(len(src)) if src.startswith("raise schedule_refusal", i)]
        self.assertTrue(raises, "the refusal sites must still exist")
        for site in raises:
            self.assertGreater(
                site,
                handover,
                "a PPScheduleRefused raised ABOVE the handover would be an "
                "exit with no continuation to re-home -- a different case "
                "than the one the junction fixes",
            )

    def test_every_refusal_raise_is_above_the_inflight_increment(self):
        """THE PREMISE `pass_allocated=False` RESTS ON, pinned.

        The junction skips both of the park's give-backs because a refused
        pass never took them. That is true only while every refusal raises
        ABOVE `self.chunked_req.inflight_middle_chunks += 1` -- the marker for
        the tail of the function where the batch is formed and its KV
        allocated. If a raise site ever moves below it, the refused pass DOES
        owe the give-backs and `pass_allocated=False` becomes a leak. Red is
        the correct outcome then, not a green that quietly under-releases.
        """
        src = self._raw_source()
        increment = src.index("inflight_middle_chunks += 1")
        raises = [
            i for i in range(len(src)) if src.startswith("raise PPScheduleRefused", i)
        ] + [i for i in range(len(src)) if src.startswith("raise schedule_refusal", i)]
        self.assertTrue(raises)
        for site in raises:
            self.assertLess(
                site,
                increment,
                "a refusal raised BELOW the inflight increment would owe the "
                "give-backs `pass_allocated=False` skips",
            )

    def test_the_funnel_routes_every_refusal_through_the_one_junction(self):
        funnel = inspect.getsource(Scheduler.get_new_batch_prefill)
        self.assertIn("except PPScheduleRefused", funnel)
        self.assertIn(
            "self._pp_refuse_forwarded_schedule(refusal)",
            funnel,
            "ONE junction is what makes one restore sufficient for all three "
            "raise sites",
        )


if __name__ == "__main__":
    unittest.main()
