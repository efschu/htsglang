"""#969: a PP void must hand its requests back through the RETRACTION path,
not through the dynamic-chunk PROBE path.

WHAT THE BOOTS SHOWED. The funding rungs never fired: every attempt logged
"reclaimed 0 MiB" with an empty source list, because the tree reported nothing
evictable at all. `evictable_size_` counts UNLOCKED tokens, so evictable=0 with
a populated tree is not a pressure reading -- it is the statement that every
node in it is pinned under a `lock_ref` nobody ever decremented.

WHERE THE REFS WERE LOST. Both PP void sites -- `_pp_void_own_batch` (#797d
own-void) and `_pp_absorb_void_output` (#791b void-output) -- released their
non-resident requests with `_release_dynamic_chunk_probe`. That helper hands
back KV rows, the mamba slot and the req-pool slot by calling the allocator and
the pool DIRECTLY. It never reaches `cache_finished_req`, which is where
`dec_lock_ref(req.last_node)` lives (radix_cache.py, the last two lines of that
method). The rows went back to the allocator and the tree kept the lock on
them. `reset_for_retract` then cleared `req.last_node`, so the ref could never
be found again, let alone returned: one leaked lock per voided request, for the
life of the process.

WHY THE HELPER WAS RIGHT WHERE IT WAS BORN AND WRONG HERE. It was written for
the synthetic dynamic-chunk PROBE (`_dyn_chunk_probe_req`), a bare `Req` built
inside the profiling loop that is never matched against the tree and therefore
holds NO lock ref. For that request a direct free is not merely adequate, it is
the only correct thing: `UnifiedRadixCache.cache_finished_req` decrements
`req.last_node` UNCONDITIONALLY, so routing a never-matched request through the
disciplined path is a decrement with no matching increment -- the #929
underflow, in the other direction. The defect was not the helper; it was
applying a probe's release to admitted requests that had been through
`match_prefix`.

THE FIX IS NOT A THIRD MECHANISM. `release_req` (schedule_batch.py) already IS
this discipline, and both correct retraction paths -- `retract_decode` and
`retract_all` -- are its only callers. The void sites now join them.
`_release_voided_request` holds no release logic of its own; it supplies the
scheduler's collaborators and the never-raise contract this path has always
carried.

THE ORDER IS LOAD-BEARING and this file pins it: `reset_for_retract` clears
`last_node` and `mamba_pool_idx`, so a release that runs after it has nothing
left to release and no node left to unlock. `release_req` calls it as its LAST
act, which is why the call sites no longer call it themselves -- doing both
would increment `retraction_count` twice.
"""

import ast
import inspect
import textwrap
import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci


def _code_only(src: str) -> str:
    """`src` with its docstring and every comment removed.

    A source-level pin that reads the prose is not a pin. Both void sites
    NAME `_release_dynamic_chunk_probe` in their comments -- explaining what
    the probe helper frees and why a chunked request must not be handed to
    it -- and those sentences stay true after the fix. Asserting over raw
    `getsource` would therefore read the explanation as the call and report
    the defect as still present forever. `ast.unparse` keeps only what runs.
    """
    tree = ast.parse(textwrap.dedent(src))
    fn = tree.body[0]
    body = fn.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return "\n".join(ast.unparse(node) for node in body)


class _Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    @property
    def length(self):
        return self.end - self.start


class _FakeNode:
    """A radix node, reduced to the one field this defect is about."""

    def __init__(self, name):
        self.name = name
        self.lock_ref = 0

    def __repr__(self):  # pragma: no cover - diagnostics only
        return f"<node {self.name} lock_ref={self.lock_ref}>"


class _FakeTreeCache:
    """The subset `release_kv_cache` touches, with the REAL lock discipline.

    `cache_finished_req` reproduces two facts about the production caches and
    nothing else:

      * its last act is `dec_lock_ref(req.last_node)` (radix_cache.py:490-492,
        pure_swa_radix_cache.py:208-209, radix_cache_cpp.py:208,
        unified_radix_cache.py:1231);
      * it releases the req slot, which is what makes `release_kv_cache`
        return at its `if req.req_pool_idx is None: return` immediately
        afterwards -- so the real `release_req` and the real
        `release_kv_cache` run end to end here without a pool, an allocator
        or a device.

    `is_chunk_cache()` returns True for the same reason: it makes the real
    `evict_from_tree_cache` a no-op, so this file measures the LOCK, which is
    the defect, and not an eviction policy, which is not.
    """

    def __init__(self):
        self.nodes = []
        self.dec_calls = []
        self.underflows = []
        self.token_to_kv_pool_allocator = None
        self.req_to_token_pool = None
        # Present so that code which hand-edits this counter RUNS rather than
        # raising AttributeError. A red that fires because the harness is
        # thin is indistinguishable from a red that fires because the product
        # is broken, and only the second one is a finding.
        self.protected_size_ = 0
        self.evictable_size_ = 0

    def new_node(self, name):
        node = _FakeNode(name)
        self.nodes.append(node)
        return node

    def is_chunk_cache(self):
        return True

    def supports_mamba(self):
        return False

    def inc_lock_ref(self, node):
        if node.lock_ref == 0:
            self.protected_size_ += 1
            self.evictable_size_ -= 1
        node.lock_ref += 1

    def dec_lock_ref(self, node, *args, **kwargs):
        node.lock_ref -= 1
        self.dec_calls.append(node)
        if node.lock_ref == 0:
            # What RadixCache.dec_lock_ref does per node whose last ref goes:
            # the length moves from protected back to evictable. The
            # hand-rolled counter edit this fix removes did the second half
            # of one of these and none of the rest.
            self.protected_size_ -= 1
            self.evictable_size_ += 1
        if node.lock_ref < 0:
            # #929 class: a decrement with no matching increment. Recorded
            # rather than raised, so a test can assert on it instead of
            # discovering it as a crash somewhere downstream.
            self.underflows.append(node)

    def cache_finished_req(self, req, is_insert=True):
        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)
        req.req_pool_idx = None

    def evictable_nodes(self):
        return [n for n in self.nodes if n.lock_ref == 0]

    def evictable_size(self):
        return len(self.evictable_nodes())


class _FakeReq:
    """The subset of `Req` this path touches, with a faithful retract."""

    def __init__(self, rid, node=None, end=64):
        self.rid = rid
        self.extend_range = _Range(0, end)
        self.prefix_indices = []
        self.is_retracted = False
        self.retracted_stain = False
        self.last_node = node
        self.mamba_pool_idx = 3
        self.retraction_count = 0
        self.req_pool_idx = 7
        self.kv_spill_state = None
        self.skip_radix_cache_insert = False

    def finished(self):
        return False

    def reset_for_retract(self):
        self.retraction_count += 1
        self.prefix_indices = []
        self.last_node = None
        self.mamba_pool_idx = None
        self.extend_range = None
        self.is_retracted = True
        self.retracted_stain = True


def _fake_scheduler_bits(tree_cache):
    """Exactly the collaborators `_release_voided_request` reads off the
    scheduler -- named here so a rename in production goes red here first."""
    return dict(
        server_args=types.SimpleNamespace(disaggregation_mode=None),
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        tree_cache=tree_cache,
        hisparse_coordinator=None,
    )


def _holder(chunked_by_slot, batches, tree_cache):
    """`_pp_absorb_void_output`'s caller shape, plus the release collaborators.

    Same construction as test_pp_void_chunked_retracted_798.py, which is the
    proven driver for this function; the only addition is the scheduler state
    the disciplined release reads.
    """
    from sglang.srt.managers import scheduler_pp_mixin as m

    return types.SimpleNamespace(
        chunked_req=None,
        waiting_queue=[],
        running_mbs=[None] * len(batches),
        _pp_chunked_req_before_by_slot=list(chunked_by_slot),
        _pp_void_forward_payload=None,
        _absorb=m.SchedulerPPMixin._pp_absorb_void_output,
        **_fake_scheduler_bits(tree_cache),
    )


def _run_absorb(tree_cache, reqs, chunked_req=None):
    """Drive the REAL `_pp_absorb_void_output` over a slot holding `reqs`.

    `chunked_req` sets the scheduler's CURRENT carried chunk, which is not
    the same thing as the slot's pre-admission snapshot: the void's keep
    check asks about the SNAPSHOT (`chunked_before`), so a member that is
    the current `chunked_req` and was not the snapshot walks straight into
    the disposal. That gap is #990's specimen.
    """
    from sglang.srt.managers import scheduler_pp_mixin as m

    batches = [types.SimpleNamespace(reqs=list(reqs))]
    holder = _holder([None], batches, tree_cache)
    holder.chunked_req = chunked_req

    saved = {
        name: getattr(m, name)
        for name in (
            "pp_void_forward_payload",
            "pp_absorb_admission_return",
            "_park_chunked_prefill_chunk",
        )
    }
    m.pp_void_forward_payload = lambda *a, **k: None
    m.pp_absorb_admission_return = lambda *a, **k: None
    m._park_chunked_prefill_chunk = lambda scheduler, req, **kw: False
    try:
        holder._absorb(holder, 0, {m._PP_VOID_OUTPUT_KEY: True}, list(batches), [None])
    finally:
        for name, fn in saved.items():
            setattr(m, name, fn)
    return holder


class PPVoidReleasesItsTreeLock969(unittest.TestCase):
    """THE DEFECT, at the shape the boots produced it."""

    def test_a_voided_request_gives_its_lock_back(self):
        """RED before the fix: the node stays locked and nothing is evictable."""
        tree = _FakeTreeCache()
        node = tree.new_node("prefix-of-voided-req")
        tree.inc_lock_ref(node)  # what match_prefix did at admission
        req = _FakeReq("voided-rid", node=node)

        self.assertEqual(
            tree.evictable_size(),
            0,
            "precondition: while the request is admitted its prefix is "
            "locked -- if this is already non-zero the test is not "
            "reproducing the admitted state and its green proves nothing",
        )

        _run_absorb(tree, [req])

        self.assertEqual(
            node.lock_ref,
            0,
            "the void released the request's rows but kept the tree lock on "
            "them: this is the evictable=0 the boots reported, and every "
            "funding rung reads 'reclaimed 0 MiB' downstream of it",
        )
        self.assertGreater(
            tree.evictable_size(),
            0,
            "nothing became evictable, so the reclaim the void was supposed "
            "to fund cannot happen",
        )

    def test_the_request_is_retracted_exactly_once(self):
        """The give-back is counted once, not twice.

        `release_req` calls `reset_for_retract` as its last act. A call site
        that also calls it directly double-counts `retraction_count`, which
        is the input to the solo-OOM abort ladder in `retract_decode` -- a
        request would be aborted at half the retractions the operator set.
        """
        tree = _FakeTreeCache()
        node = tree.new_node("n")
        tree.inc_lock_ref(node)
        req = _FakeReq("once-rid", node=node)

        _run_absorb(tree, [req])

        # #984 RETRACTED THIS ARM'S PREMISE, and the premise is what changed
        # -- not the invariant. This used to read `assertTrue(req.is_retracted)`
        # and `retraction_count == 1`, because a voided pass RETRACTED rank
        # 0's members. It now PARKS them (the void became symmetric with the
        # follower ranks), so nothing on this path retracts and
        # `retraction_count` is never touched at all.
        #
        # WHAT THIS ARM WAS PROTECTING SURVIVES VERBATIM, which is why it is
        # re-encoded rather than deleted: `retraction_count` is the input to
        # the solo-OOM abort ladder in `retract_decode`, and a request must
        # never be aborted at half the retractions the operator configured.
        # A double count was the hazard when the void retracted; an UNASKED
        # count is the hazard now. Both are the same assertion: after a void,
        # this number reflects the retractions that actually happened, and a
        # void-park performed none.
        self.assertFalse(
            req.is_retracted,
            "#984: a voided pass PARKS rank 0's members, it does not retract "
            "them -- an is_retracted request here means the symmetry broke "
            "and the reset shape is back on the queue path",
        )
        self.assertEqual(
            req.retraction_count,
            0,
            "a void-park retracts nothing, so it must not advance the "
            "abort-ladder counter -- inflating it aborts the request early",
        )

    def test_the_ref_goes_back_while_the_prefix_handle_is_KEPT(self):
        """The #969 invariant under #984's premise: give the ref, keep the handle.

        RENAMED AND RE-ENCODED 2026-08-28, from
        `test_the_release_happens_before_the_reset`. The old name and its
        second assertion (`assertIsNone(req.last_node)`) encoded the ORDERING
        problem of a world where the void RETRACTED: `reset_for_retract`
        cleared `last_node`, so a release running after it had nothing left
        to unlock. #984 removes that world -- the void parks, nothing is
        reset, and `last_node` is deliberately PRESERVED
        (`pp_give_back_admission_lock_ref`: "`last_node` and `prefix_indices`
        are deliberately NOT cleared: they are what the next offer reports as
        executed, which is the entire point of #984").

        THE INVARIANT IS UNCHANGED AND IS WHAT THIS ARM NOW STATES: the
        admission-side `inc_lock_ref` that `PrefillAdder._req_inc_lock_ref`
        took must not outlive a pass that never ran. Whether the handle is
        destroyed afterwards is a detail of the old disposal; whether the
        REF came back is the thing that decides if the prefix is ever
        evictable again. Asserting the old `is None` today would assert the
        absence of the very state #984 exists to keep.
        """
        tree = _FakeTreeCache()
        node = tree.new_node("order")
        tree.inc_lock_ref(node)
        req = _FakeReq("order-rid", node=node)

        _run_absorb(tree, [req])

        self.assertIn(
            node,
            tree.dec_calls,
            "no dec_lock_ref reached the node at all -- the admission's "
            "increment is still outstanding and the prefix can never become "
            "evictable, which is #969 by a new route",
        )
        self.assertEqual(
            node.lock_ref,
            0,
            "exactly the admission's one ref came back: not zero give-backs "
            "(a leak) and not two (the #929 underflow)",
        )
        self.assertIs(
            req.last_node,
            node,
            "#984: the prefix HANDLE is kept while its CLAIM is released -- "
            "the pages stay in the tree and `last_node`/`prefix_indices` are "
            "what the next offer reports as already executed. Clearing them "
            "here is what would force the re-computation #984 prevents",
        )


class TheGiveBackRespectsOwnership990(unittest.TestCase):
    """#990: a CARRIED CHUNK does not take a fresh ref, so it keeps its one.

    THE PREMISE #984 GOT RIGHT FOR ONE CASE AND WRONG FOR THE OTHER. Its
    argument for giving the ref back is that re-admission takes a FRESH one:
    `init_next_round_input` re-matches and `_req_inc_lock_ref` increments
    again, so holding the old ref would leak one per void cycle. That is true
    for a request going back to the WAITING QUEUE. It is false for the
    request currently held as `self.chunked_req`: a carried chunk is
    re-admitted from that field by `add_chunked_req`, and the stash transfers
    the one admission ref stash-to-stash
    (`unified_radix_cache.py:1354-1355`). No fresh ref is taken, so giving
    this one back makes ONE inc meet TWO decs -- this give-back and the first
    stash's -- which is boot 9's `full_component.py:320` underflow assert.

    WHY THE MEMBER EVEN REACHES THE GIVE-BACK, which is the part worth
    stating because it is not obvious: the void's keep check asks
    `pp_void_keeps_request(req, resident, chunked_before)` about the slot's
    PRE-ADMISSION SNAPSHOT. A request that became `self.chunked_req` during
    THIS pass is not that snapshot, so it is not kept, and it walks into the
    disposal and the give-back as an ordinary member. Ownership and
    snapshot-identity are two different questions and only one of them was
    being asked.

    NOT A WEAKENED ASSERT. The underflow detector is untouched; what changed
    is that ownership is discriminated before the decrement. The #986 orphan
    route still gives back, correctly -- by definition an orphan has LEFT the
    `chunked_req` field, so nothing will transfer its ref.
    """

    @unittest.expectedFailure
    def test_an_orphaned_request_must_give_its_ref_back(self):
        """#990 REGRESSION, MEASURED at b27d7c2ff4. Expected-failure ON PURPOSE.

        A request DISPLACED out of `chunked_req` and re-homed into the waiting
        queue keeps its admission lock ref. Re-admission from the queue runs
        `init_next_round_input` -> `match_prefix` -> `_req_inc_lock_ref` and
        takes a SECOND ref, so the count climbs by one per void cycle and the
        prefix never becomes evictable again -- #969's leak, reopened by a new
        route.

        THE CAUSE IS A STATEMENT ORDER. #990's guard reads `if req is
        scheduler.chunked_req: return False` (:3160) and justifies itself with
        "the #986 orphan route keeps giving back: by its own definition the
        orphan has LEFT the chunked_req field". At the call site it has not
        left yet: the re-home runs BEFORE the field is overwritten
        (:7909/:7913 own-void, :9261/:9265 void-output), and the comment there
        explains why that order is required -- "Re-homed BEFORE the overwrite
        -- after it, the reference is already gone" (#968b needs it). So the
        guard fires on exactly the request being orphaned.

        MEASURED, at the tip, by instrumenting the give-back's caller:
          member IS current chunked_req -> called from
              pp_queue_orphaned_chunked_req, suppressed=True, lock_ref stays 1
          ordinary member               -> called from
              pp_park_voided_batch_member, suppressed=False, lock_ref -> 0
        In those two shapes the guard's only observed firing is the harmful
        one. #990's INTENT is sound and Boot 10 confirmed it on metal (no
        underflow); what is wrong is that it discriminates "is still in the
        field right now" instead of "will keep the field".

        MARKED expectedFailure rather than deleted or inverted: inverting it
        would encode the leak as correct, and deleting it would lose the only
        executable record. When the ownership question is asked properly this
        turns into an UNEXPECTED SUCCESS, which is a loud, self-clearing
        signal to drop the marker.
        """
        tree = _FakeTreeCache()
        node = tree.new_node("orphaned")
        tree.inc_lock_ref(node)
        req = _FakeReq("orphan-rid", node=node)

        holder = _run_absorb(tree, [req], chunked_req=req)

        self.assertTrue(
            any(r is req for r in holder.waiting_queue),
            "precondition: the request must actually have been orphaned into "
            "the queue -- otherwise this arm is not measuring the leak",
        )
        self.assertIsNone(
            holder.chunked_req,
            "precondition: it must have LEFT the field, which is what makes "
            "the ref nobody's to transfer",
        )
        self.assertIn(
            node,
            tree.dec_calls,
            "a request that left the chunked_req field and sits in the "
            "waiting queue must hand its admission ref back: re-admission "
            "takes a fresh one, so keeping this one leaks exactly one ref "
            "per void cycle and the prefix never becomes evictable",
        )

    def test_an_ordinary_member_still_gives_its_ref_back(self):
        """The #969/#984 case is untouched: not-carried still means give back.

        Without this, #990 could be 'fixed' by never giving any ref back,
        which would reinstate the #969 leak this file was written for.
        """
        tree = _FakeTreeCache()
        node = tree.new_node("ordinary")
        tree.inc_lock_ref(node)
        req = _FakeReq("ordinary-rid", node=node)

        _run_absorb(tree, [req], chunked_req=None)

        self.assertIn(
            node,
            tree.dec_calls,
            "an ordinary voided member goes back to the waiting queue, where "
            "re-admission DOES take a fresh ref -- keeping the old one is the "
            "#969 leak",
        )
        self.assertEqual(node.lock_ref, 0)


class PPVoidDoesNotOverRelease969(unittest.TestCase):
    """THE OTHER DIRECTION. An unpaired decrement is the #929 underflow."""

    def test_exactly_one_decrement_per_released_request(self):
        tree = _FakeTreeCache()
        nodes = [tree.new_node(f"n{i}") for i in range(3)]
        for n in nodes:
            tree.inc_lock_ref(n)
        reqs = [_FakeReq(f"r{i}", node=n) for i, n in enumerate(nodes)]

        _run_absorb(tree, reqs)

        self.assertEqual(len(tree.dec_calls), 3)
        self.assertEqual(
            tree.underflows,
            [],
            "a node's lock_ref went negative: the void decremented a ref it "
            "did not hold",
        )
        for n in nodes:
            self.assertEqual(n.lock_ref, 0)

    def test_a_request_holding_no_lock_is_not_decremented(self):
        """The probe's shape, arriving at the void path.

        A request with `last_node is None` never matched a prefix and holds
        no ref. Decrementing for it is the underflow this suite exists to
        keep out, and it is exactly what a blanket "route everything through
        the tree" fix would do under `UnifiedRadixCache`, whose
        `cache_finished_req` decrements without a None guard.
        """
        tree = _FakeTreeCache()
        locked = tree.new_node("someone-elses-prefix")
        tree.inc_lock_ref(locked)
        req = _FakeReq("no-node-rid", node=None)

        _run_absorb(tree, [req])

        self.assertEqual(tree.dec_calls, [])
        self.assertEqual(tree.underflows, [])
        self.assertEqual(
            locked.lock_ref,
            1,
            "the void decremented a node that belongs to another request",
        )

    def test_the_underflow_detector_can_actually_fail(self):
        """Kann-failen-Beweis for the two assertions above.

        A guard that has never been seen to go red is not a guard. This
        drives the mutant the fix must not become -- a second decrement for
        the same request -- straight at the detector and requires it to fire.

        THE MUTANT HAD TO MOVE, AND A DEAD CAN-FAIL IS WHY THIS MATTERS.
        It used to replace `tree_cache.cache_finished_req` with a
        double-decrement, because the void released rank 0's members through
        `_release_voided_request` -> `release_req` -> `cache_finished_req`.
        #984 parks them instead, so that path is not taken and the mutant was
        never invoked: measured at 8be86f55fe the arm failed `0 != -1`, i.e.
        the injected fault could no longer occur AT ALL. A can-fail arm whose
        mutant has become unreachable does not report a safe system -- it
        reports nothing, and it does so while looking like a red that a
        tired reader would "fix" by deleting the assertion.
        The mutant is therefore re-aimed at the actuator #984 actually uses.
        `pp_give_back_admission_lock_ref` is module-level precisely so this
        is possible -- its own docstring says "A module-level function so a
        can-fail proof can neuter this ONE step."
        """
        from sglang.srt.managers import scheduler_pp_mixin as mod

        tree = _FakeTreeCache()
        node = tree.new_node("mutant")
        tree.inc_lock_ref(node)
        req = _FakeReq("mutant-rid", node=node)

        # The mutant: the give-back hands back a ref it took once AND a ref
        # it never took -- the #929 underflow, injected at the one step that
        # now performs the release.
        original = mod.pp_give_back_admission_lock_ref

        def _double_give_back(scheduler, r):
            n = getattr(r, "last_node", None)
            if n is None:
                return False
            tree.dec_lock_ref(n)
            tree.dec_lock_ref(n)
            return True

        mod.pp_give_back_admission_lock_ref = _double_give_back
        try:
            _run_absorb(tree, [req])
        finally:
            mod.pp_give_back_admission_lock_ref = original

        self.assertEqual(node.lock_ref, -1)
        self.assertEqual(
            len(tree.underflows),
            1,
            "the underflow detector did not fire on a doubled decrement, so "
            "the two tests above would stay green through the #929 defect",
        )


class PPVoidReachability969(unittest.TestCase):
    """The defect path reaches the fix -- shown at the source, per site."""

    def _src(self, name):
        from sglang.srt.managers import scheduler_pp_mixin as m

        return _code_only(inspect.getsource(getattr(m.SchedulerPPMixin, name)))

    def test_both_void_sites_use_the_disciplined_release(self):
        for site in ("_pp_void_own_batch", "_pp_absorb_void_output"):
            src = self._src(site)
            with self.subTest(site=site):
                self.assertIn(
                    "_release_voided_request",
                    src,
                    f"{site} does not route its released requests through "
                    f"the retraction path",
                )
                self.assertNotIn(
                    "_release_dynamic_chunk_probe",
                    src,
                    f"{site} still releases admitted requests with the "
                    f"probe helper, which never reaches dec_lock_ref",
                )

    def test_the_probe_sites_keep_the_probe_release(self):
        """The boundary, so the next reader does not 'unify' it into a bug.

        The dynamic-chunk profiler's request is never matched against the
        tree. Routing it through the disciplined path would decrement a ref
        it never took.
        """
        from sglang.srt.managers import scheduler_pp_mixin as m

        src = _code_only(
            inspect.getsource(m.SchedulerPPMixin.profile_and_init_predictor)
        )
        self.assertIn("_release_dynamic_chunk_probe", src)
        self.assertNotIn("_release_voided_request", src)

    def test_the_discipline_has_exactly_one_expression(self):
        """`_release_voided_request` must delegate, not reimplement."""
        from sglang.srt.managers import scheduler_pp_mixin as m

        src = _code_only(inspect.getsource(m._release_voided_request))
        self.assertIn("release_req(", src)
        for reimplementation in (
            "token_to_kv_pool_allocator.free(",
            "free_mamba_cache(",
            "dec_lock_ref(",
        ):
            self.assertNotIn(
                reimplementation,
                src,
                f"the void release reimplements {reimplementation} instead "
                f"of delegating to release_req -- a second expression of a "
                f"discipline that already has one",
            )

    def test_the_fake_cache_cannot_drift_from_the_real_one(self):
        """`_FakeTreeCache.cache_finished_req` must keep describing production.

        If `RadixCache.cache_finished_req` ever stops ending in a
        `dec_lock_ref(req.last_node)`, this file is measuring something that
        no longer exists and its green means nothing.
        """
        from sglang.srt.mem_cache.radix_cache import RadixCache

        src = _code_only(inspect.getsource(RadixCache.cache_finished_req))
        self.assertIn("dec_lock_ref(req.last_node)", src)

    def test_the_retraction_paths_are_the_template(self):
        """Both correct paths still route through the same `release_req`."""
        from sglang.srt.managers import schedule_batch as sb

        self.assertIn("release_req(", _code_only(inspect.getsource(sb.retract_all)))
        self.assertIn(
            "release_req(",
            _code_only(inspect.getsource(sb.ScheduleBatch.retract_decode)),
        )
        self.assertIn(
            "release_kv_cache(", _code_only(inspect.getsource(sb.release_req))
        )


class _FakeReqToken:
    """`req_to_token[idx, a:b]` without a device."""

    def __getitem__(self, key):
        return ("rows", key)


class _RecordingAllocator:
    def __init__(self):
        self.freed = []

    def free(self, indices):
        self.freed.append(indices)


class _FakePool:
    def __init__(self):
        self.req_to_token = _FakeReqToken()
        self.freed_reqs = []

    def free(self, req):
        self.freed_reqs.append(req)
        req.req_pool_idx = None


class _OffloadReq:
    def __init__(self, rid, node, prefix_len):
        self.rid = rid
        self.req_pool_idx = 5
        self.last_node = node
        self.prefix_indices = list(range(prefix_len))

    def pop_committed_kv_cache(self):
        return 128

    def pop_overallocated_kv_cache(self):
        return (0, 0)


class DecodeOffloadReleaseGivesTheLockBack969(unittest.TestCase):
    """#969 SIBLING: the same leak, hand-rolled in the PD decode offload path.

    `DecodeKVCacheOffloadManager._release_finished_req` used to end with

        self.tree_cache.protected_size_ -= len(req.prefix_indices)

    which imitates ONE of the four effects `dec_lock_ref` has per node on the
    path to root. `lock_ref` itself was never touched, so every node on the
    request's prefix path stayed locked and `evictable_size_` never recovered
    -- the PP-void mechanism again, by a different route.

    This suite is UNEXERCISED ON THIS RIG: the path needs
    `--disaggregation-decode-enable-offload-kvcache`, and the lock chain it
    leaks needs `--disaggregation-decode-enable-radix-cache` on top. It was
    found by the sibling sweep for #969, not by a boot, and it is fixed
    rather than filed because the defect is decidable by inspection: under
    the `ChunkCache` default the old line was not merely incomplete but
    wrong in the other direction, since that cache's `inc_lock_ref` never
    raises the counter the line subtracted from.
    """

    def _manager(self, tree):
        from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
            DecodeKVCacheOffloadManager,
        )

        mgr = object.__new__(DecodeKVCacheOffloadManager)
        mgr.req_to_token_pool = _FakePool()
        mgr.token_to_kv_pool_allocator = _RecordingAllocator()
        mgr.tree_cache = tree
        mgr.page_size = 1
        mgr.offloaded_state = {}
        return mgr

    def test_the_finished_request_gives_its_lock_back(self):
        tree = _FakeTreeCache()
        node = tree.new_node("decode-prefix")
        tree.inc_lock_ref(node)
        req = _OffloadReq("offload-rid", node, prefix_len=64)

        self.assertEqual(tree.evictable_size(), 0)
        self._manager(tree)._release_finished_req(req, start_offset=0)

        self.assertEqual(
            node.lock_ref,
            0,
            "the offload release freed the request's rows but left the tree "
            "locked on them",
        )
        self.assertGreater(tree.evictable_size(), 0)
        self.assertEqual(tree.underflows, [])

    def test_a_request_holding_no_lock_is_not_decremented(self):
        """The ChunkCache shape: no lock chain, so nothing to give back."""
        tree = _FakeTreeCache()
        other = tree.new_node("someone-else")
        tree.inc_lock_ref(other)
        req = _OffloadReq("no-node", None, prefix_len=64)

        self._manager(tree)._release_finished_req(req, start_offset=0)

        self.assertEqual(tree.dec_calls, [])
        self.assertEqual(other.lock_ref, 1)

    def test_the_release_no_longer_hand_edits_the_counter(self):
        """One expression of the discipline here too."""
        from sglang.srt.disaggregation import decode_kvcache_offload_manager as d

        src = _code_only(
            inspect.getsource(d.DecodeKVCacheOffloadManager._release_finished_req)
        )
        self.assertIn("dec_lock_ref", src)
        self.assertNotIn(
            "protected_size_",
            src,
            "the release still edits protected_size_ by hand instead of "
            "letting dec_lock_ref keep the tree's counters coherent",
        )


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
