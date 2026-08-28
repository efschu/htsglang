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


def _run_absorb(tree_cache, reqs):
    """Drive the REAL `_pp_absorb_void_output` over a slot holding `reqs`."""
    from sglang.srt.managers import scheduler_pp_mixin as m

    batches = [types.SimpleNamespace(reqs=list(reqs))]
    holder = _holder([None], batches, tree_cache)

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
    m._park_chunked_prefill_chunk = lambda scheduler, req: False
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

        self.assertTrue(req.is_retracted)
        self.assertEqual(
            req.retraction_count,
            1,
            "the voided request was reset more than once; the release path "
            "and the call site are both doing it",
        )

    def test_the_release_happens_before_the_reset(self):
        """`reset_for_retract` clears `last_node`; a release after it is blind."""
        tree = _FakeTreeCache()
        node = tree.new_node("order")
        tree.inc_lock_ref(node)
        req = _FakeReq("order-rid", node=node)

        _run_absorb(tree, [req])

        self.assertIn(
            node,
            tree.dec_calls,
            "no dec_lock_ref reached the node at all -- the release ran "
            "after reset_for_retract had already set last_node to None, so "
            "there was nothing left to unlock",
        )
        self.assertIsNone(req.last_node)


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
        """
        tree = _FakeTreeCache()
        node = tree.new_node("mutant")
        tree.inc_lock_ref(node)
        req = _FakeReq("mutant-rid", node=node)

        # The mutant: cache_finished_req decrements twice, i.e. the void
        # gives back a ref it took once and a ref it never took.
        def _double_dec(r, is_insert=True):
            if r.last_node is not None:
                tree.dec_lock_ref(r.last_node)
                tree.dec_lock_ref(r.last_node)
            r.req_pool_idx = None

        tree.cache_finished_req = _double_dec
        _run_absorb(tree, [req])

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
