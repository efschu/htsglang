"""#971: a REFUSED PP pass must re-home the chunked continuation it borrowed.

#1153 WITHDRAWAL (2026-09-02), read this before the rest of the docstring:
the junction this file was written around, `_pp_refuse_forwarded_schedule`,
is DELETED. A follower's refusal of a forwarded schedule is a detected rank
disagreement, and a detected rank disagreement is a group STOP
(RAENGE-NIE-UNEINS) -- the funnel now re-raises `PPScheduleRefused` as a
RuntimeError (`#791 FORWARDED SCHEDULE UNEXECUTABLE STOP ...`, see
test_pp_forwarded_refusal_stop_1153) and the process ends the group. The
void the junction performed (`_pp_admission_pass_voided = True`, emptied
decision dicts) nulled ONE rank's slot while PP0's stayed set and nothing
carried it upstream; PP0 then consumed the last rank's next output under
this slot's label and stayed one output ahead for the rest of the boot
(boot_855_weg1b2, log 65000-65119). The #971 re-home was a repair inside
that compensation layer: with no process left to continue, there is nothing
to re-home, and the arms below that pinned the restore are INVERTED to pin
its absence -- each keeps its name and says what it now asserts. The
handover facts (arm 0), the sibling park sites and the skip census remain
true and remain pinned unchanged.

THE ORIGINAL #971 TEXT FOLLOWS, kept as the record of why the junction had
the shape it had.

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
        # #1153: the forwarded extents `_pp_scheduled_extents` reads, as
        # `_pp_reconcile_incoming_admission` would have set them.
        _pp_admission_incoming_schedule={RID_UNREACHED: (0, 64)},
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
    # #1153: the junction is deleted; the funnel raises through this instead.
    h._pp_forwarded_schedule_stop = types.MethodType(
        Scheduler._pp_forwarded_schedule_stop, h
    )
    h._pp_scheduled_extents = types.MethodType(Scheduler._pp_scheduled_extents, h)
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
    # #1153: the funnel no longer returns a plan for a refused pass; it
    # raises the group STOP. The exception is handed back where the plan
    # used to be, so every arm below reads the same tuple shape.
    try:
        h.get_new_batch_prefill(running_batch=h.running_batch)
    except RuntimeError as stop:
        return stop, captured
    raise AssertionError("#1153: a refused forwarded schedule must STOP")


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
    """ARM 1 -- INVERTED by #1153. Was: 'the pass is refused and voided, and
    the continuation is restored to `self.chunked_req`'. Now: the pass is
    refused and the group STOPS; nothing is voided and nothing is restored,
    because no rank continues from a pass the group could not agree on."""

    def test_the_pass_is_refused_and_voided(self):
        """INVERTED (#1153): refused and STOPPED, not voided."""
        h = _holder(_real_req())
        stop, captured = _refused_pass(h)
        self.assertTrue(captured["handed_over"], "the handover really happened")
        self.assertIsInstance(stop, RuntimeError)
        self.assertTrue(
            str(stop).startswith("#791 FORWARDED SCHEDULE UNEXECUTABLE STOP"),
            str(stop),
        )
        self.assertIn(f"told=[{RID_UNREACHED}]", str(stop))
        self.assertFalse(
            getattr(h, "_pp_admission_pass_voided", False),
            "#1153 withdrawal: the void flag is never written by the funnel",
        )
        self.assertNotEqual(
            h._pp_admission_incoming_schedule,
            {},
            "#1153 withdrawal: the decision dicts are not emptied",
        )

    def test_it_is_restored_to_scheduler_owned_state_not_merely_findable(self):
        """INVERTED (#1153): there is no restore. The handover left the
        continuation adder-owned and the STOP leaves it there -- the process
        ends; `self.chunked_req` is not re-homed by the funnel."""
        req = _real_req()
        h = _holder(req)
        _refused_pass(h)
        self.assertIsNone(
            h.chunked_req,
            "#1153: the funnel performs no re-home; a STOP has no next round",
        )


class TheRestoredChunkIsParked(unittest.TestCase):
    """ARM 2 -- INVERTED by #1153. Was: the restored request's geometry is
    parked so the next round does not cache a never-run chunk. Now: there is
    no next round and no restore, so the geometry is left exactly as the
    handover wrote it -- the STOP touches no request."""

    def test_extend_range_survives_the_refusal(self):
        """INVERTED (#1153): the funnel does not touch the request at all,
        so the never-run range written by the handover is still on it."""
        h = _holder(_real_req())
        _, captured = _refused_pass(h)
        (req,) = captured["can_run_list"]
        self.assertIsNotNone(req.extend_range)
        self.assertEqual(req.extend_range.start, PREFIX_DONE)
        self.assertEqual(req.extend_range.end, TOTAL_FILL)

    def test_the_never_run_chunk_is_parked_so_the_next_stash_is_a_no_op(self):
        """INVERTED (#1153): nothing is parked, because the park was part of
        the restore and the restore is withdrawn with the junction."""
        h = _holder(_real_req())
        _, captured = _refused_pass(h)
        (req,) = captured["can_run_list"]
        self.assertNotEqual(req.extend_range.end, PREFIX_DONE, "not parked")
        self.assertEqual(req.inflight_middle_chunks, 1, "not given back")


class TheRefusalDiscardsNoTokens(unittest.TestCase):
    """ARM 3 -- kept under #1153 with its meaning sharpened: a STOP releases
    NOTHING. The refused pass allocated nothing, so the funnel must not free
    KV rows, req slots or tree handles on its way out -- a release here would
    be a second give-back on another pass's pages, exactly the hazard the
    old `pass_allocated=False` park avoided, now avoided by doing nothing."""

    def test_nothing_is_released_because_this_pass_allocated_nothing(self):
        h = _holder(_real_req())
        _refused_pass(h)
        self.assertEqual(h.token_to_kv_pool_allocator.freed, [])
        self.assertEqual(h.req_to_token_pool.freed_req, [])

    def test_the_prefix_the_request_keeps_is_the_full_cached_prefix(self):
        h = _holder(_real_req())
        _, captured = _refused_pass(h)
        (req,) = captured["can_run_list"]
        self.assertEqual(len(req.prefix_indices), PREFIX_DONE)

    def test_the_request_is_not_retracted_and_keeps_its_tree_handles(self):
        h = _holder(_real_req())
        _, captured = _refused_pass(h)
        (req,) = captured["can_run_list"]
        self.assertEqual(req.last_node, "node")
        self.assertFalse(getattr(req, "is_retracted", False))

    def test_the_inflight_count_is_left_exactly_as_it_was(self):
        h = _holder(_real_req())
        _, captured = _refused_pass(h)
        (req,) = captured["can_run_list"]
        self.assertEqual(req.inflight_middle_chunks, 1)


class TheRestoreIsWhatMakesTheGreen(unittest.TestCase):
    """ARM 4 -- INVERTED by #1153. Was: neutering the restore turns arms 1
    and 2 red, proving the green came from the restore. Now: the restore
    does not exist in the shipped funnel, and THAT is what is pinned --
    a future re-insertion of any re-home into the refusal path is red here."""

    def test_arm1_fails_when_only_the_restore_is_neutered(self):
        """INVERTED (#1153): there is no restore to neuter. The funnel and
        the STOP builder reference no re-home and no void."""
        funnel = inspect.getsource(Scheduler.get_new_batch_prefill)
        stop = inspect.getsource(Scheduler._pp_forwarded_schedule_stop)
        for src in (funnel, stop):
            self.assertNotIn("pp_rehome_refused_chunked_req", src)
            self.assertNotIn("_park_chunked_prefill_chunk", src)
            self.assertNotIn("_pp_admission_pass_voided = True", src)
            self.assertNotIn("void_pp_admission_decision(", src)

    def test_the_slot_ring_alone_is_not_a_home(self):
        """INVERTED (#1153): the ring is not consulted by the funnel at all;
        a refused pass leaves the ring untouched and stops."""
        req = _real_req()
        h = _holder(req, ring_shape="list")
        before = list(h._pp_chunked_req_before_by_slot)
        _refused_pass(h)
        self.assertEqual(h._pp_chunked_req_before_by_slot, before)
        self.assertIsNone(h.chunked_req)

    def test_arm2_fails_when_only_the_restore_is_neutered(self):
        """INVERTED (#1153): the junction the neuter targeted is deleted
        from the shipped class."""
        self.assertFalse(hasattr(Scheduler, "_pp_refuse_forwarded_schedule"))


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
        """INVERTED (#1153): the one junction is now the one STOP. Every
        `PPScheduleRefused` raise site still funnels through the single
        `except`, and that `except` raises the group STOP."""
        funnel = inspect.getsource(Scheduler.get_new_batch_prefill)
        self.assertIn("except PPScheduleRefused", funnel)
        self.assertNotIn("self._pp_refuse_forwarded_schedule(refusal)", funnel)
        self.assertIn(
            "raise self._pp_forwarded_schedule_stop(refusal) from refusal",
            funnel,
            "ONE junction is what makes one STOP sufficient for all three raise sites",
        )


if __name__ == "__main__":
    unittest.main()
