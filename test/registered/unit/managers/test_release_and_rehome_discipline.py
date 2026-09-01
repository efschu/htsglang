"""The release-delegation discipline and the chunked-req re-home, pinned by
SUBJECT rather than by the mechanism that used to carry them.

WHY THIS FILE EXISTS, AND WHY IT IS NAMED LIKE THIS. Both pins below lived in
files whose subject was the #797/#791b VOID RELAY, and that relay was deleted
whole in #1072 (a message no rank originated, nobody relayed, one rank
absorbed). The pins went down with it -- but neither pin was ABOUT the relay:

  * `_release_voided_request` is the one disciplined release path, and it
    still has a live caller (`_pp_void_own_batch`).
  * `pp_rehome_displaced_chunked_req` still has a live caller
    (`_pp_void_own_batch`, scheduler_pp_mixin.py:9022) and the reset-shape
    clears, and lost ALL coverage when
    test_pp_continuation_cross_slot_rehome_968b.py went (28 of its 30 tests
    drove the deleted absorber transitively, so the file could not stay).

So the subject of each test below is the DISCIPLINE, never the void site that
happened to exercise it. A future deletion sweep on the PP ring family must
be able to run straight through this file.

RETRACTED, AND THE RETRACTION IS THE POINT (2026-09-01). The predecessor pin
`test_the_discipline_has_exactly_one_expression` asserted that the string
`free_mamba_cache(` appears NOWHERE in `_release_voided_request`, and it was
RED at the time its file was deleted. It was reported as a surviving defect.
IT IS NOT ONE. #993 deliberately added a post-failure give-back inside the
`except` handler, with a measured justification (boot 12, 7b855f63fc,
2026-08-28 21:46:05, rid 5708abdd57..., `release_req` raised "Committed KV
cache already freed", the swallow left a re-admissible request holding a
req-pool row, and `ReqToTokenPool.alloc` killed rank 0 on the memory_pool.py
:395 assert one second later). The stale half was the ASSERTION, not the
code: its subject was "the normal path delegates" but its implementation was
"this string never appears in this function", and a containment path added
later violated the letter while keeping the intent. Measured split at
ca0ee3acd4+#1072:

    try body (normal path)  release_req( True, free_mamba_cache( False,
                            token_to_kv_pool_allocator.free( False,
                            dec_lock_ref( False
    except handlers         release_req( False, free_mamba_cache( True,
                            pool.free( True

The pin below therefore asserts the discipline WHERE IT IS CLAIMED -- on the
normal path -- and asserts separately that the containment stays containment
(reachable only from a failed release, and never raising out of it). A
re-pinned red would have manufactured a defect and cost a boot.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci


def _release_try() -> ast.Try:
    """The one `try` of `_release_voided_request`: normal path vs containment."""
    from sglang.srt.managers import scheduler_pp_mixin as m

    src = textwrap.dedent(inspect.getsource(m._release_voided_request))
    fn = ast.parse(src).body[0]
    tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try)]
    assert tries, "the release path no longer has a try/except -- re-derive this pin"
    return tries[0], src


def _seg(nodes, src) -> str:
    return "\n".join(ast.get_source_segment(src, n) or "" for n in nodes)


# -- the delegation discipline -------------------------------------------


class ReleaseDelegationDiscipline(unittest.TestCase):
    """`release_req` IS the discipline; this path supplies collaborators."""

    def test_the_normal_path_delegates_and_holds_no_release_logic(self):
        """The claim the docstring makes ("NOT A THIRD MECHANISM"), checked
        where it is made: the try body. Reimplementing any of these three is
        the leak #969 closed -- pages returned to the allocator while the
        radix tree keeps its lock on them, one leaked lock per request for
        the life of the process, read from outside as `evictable_size_` = 0
        against a full tree."""
        node, src = _release_try()
        body = _seg(node.body, src)
        self.assertIn("release_req(", body)
        for reimplementation in (
            "token_to_kv_pool_allocator.free(",
            "free_mamba_cache(",
            "dec_lock_ref(",
            "pool.free(",
        ):
            self.assertNotIn(
                reimplementation,
                body,
                f"the normal release path reimplements {reimplementation} "
                f"instead of delegating to release_req -- a second expression "
                f"of a discipline that already has one (#969)",
            )

    def test_the_give_back_is_containment_and_stays_in_the_handler(self):
        """#993's give-back is a DIFFERENT premise, not a second expression:
        it runs only when `release_req` raised. If it ever migrates to the
        normal path it becomes exactly the reimplementation the test above
        forbids, so its location is the pin."""
        node, src = _release_try()
        handlers = _seg(node.handlers, src)
        self.assertIn("free_mamba_cache(", handlers)
        self.assertIn("pool.free(", handlers)
        self.assertNotIn(
            "release_req(",
            handlers,
            "the containment calls release_req again -- a failed release "
            "retried in its own failure handler",
        )

    def test_the_containment_never_raises(self):
        """The never-raise contract, structurally. An instrument that raises
        while cleaning up after a divergence turns one defect into two, and
        this handler runs on exactly that path."""
        node, src = _release_try()
        for handler in node.handlers:
            calls = [
                n
                for n in ast.walk(handler)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("free", "free_mamba_cache")
            ]
            for call in calls:
                guarded = any(
                    isinstance(anc, ast.Try) and anc is not node
                    for anc in ast.walk(handler)
                    if isinstance(anc, ast.Try)
                    and call.lineno >= anc.lineno
                    and call.lineno <= (anc.end_lineno or anc.lineno)
                )
                self.assertTrue(
                    guarded,
                    "a give-back call in the failure handler is not itself "
                    "wrapped -- cleanup may not raise",
                )

    def test_mamba_is_given_back_before_the_req_slot(self):
        """The ordering #993 names: `free_mamba_cache` reads
        `...mapping[req.req_pool_idx]` and `free` nulls that index, so the
        reverse order gives back a mamba slot keyed on a row that is gone."""
        node, src = _release_try()
        handlers = _seg(node.handlers, src)
        self.assertLess(
            handlers.index("free_mamba_cache("),
            handlers.index("pool.free("),
            "the req-pool row is handed back before the mamba slot that is "
            "keyed on it",
        )


# -- the re-home, which lost all coverage with the 968b file ---------------


class _Range:
    def __init__(self, end):
        self.end = end


class _Req:
    def __init__(self, rid, *, end=None, prefix=0, retracted=False):
        self.rid = rid
        self.extend_range = None if end is None else _Range(end)
        self.prefix_indices = list(range(prefix))
        self.is_retracted = retracted


class RehomeDisplacedChunkedReq(unittest.TestCase):
    """`pp_rehome_displaced_chunked_req`: the ONE `chunked_req` field is
    about to be overwritten, and whatever is in it must not simply vanish.

    Live caller at scheduler_pp_mixin.py:9022 (`_pp_void_own_batch`). The
    displacement class is #968b: back-to-back per-slot restores mean
    last-slot-wins, and any other slot's continuation used to be dropped out
    of the only place it lived.
    """

    def setUp(self):
        from sglang.srt.managers import scheduler_pp_mixin as m

        self.m = m
        self.parked = []
        self.queued = []
        self._saved = {
            n: getattr(m, n)
            for n in ("_park_chunked_prefill_chunk", "pp_queue_orphaned_chunked_req")
        }
        m._park_chunked_prefill_chunk = lambda scheduler, req, **kw: (
            self.parked.append(req.rid) or True
        )
        m.pp_queue_orphaned_chunked_req = lambda scheduler, req, **kw: (
            self.queued.append(req.rid) or True
        )

    def tearDown(self):
        for n, fn in self._saved.items():
            setattr(self.m, n, fn)

    def _call(self, current, incoming):
        sched = types.SimpleNamespace(chunked_req=current)
        return self.m.pp_rehome_displaced_chunked_req(
            sched, incoming, mb_id=0, route="test"
        )

    def test_no_occupant_is_not_a_displacement(self):
        self.assertIsNone(self._call(None, _Req("in")))
        self.assertEqual(self.queued, [])

    def test_the_incoming_request_does_not_displace_itself(self):
        """The field is about to be set to the request already in it."""
        req = _Req("same", end=4096, prefix=4096)
        self.assertIsNone(self._call(req, req))
        self.assertEqual(self.queued, [])

    def test_a_parked_shape_occupant_is_re_homed(self):
        """end == len(prefix_indices): the settled shape. It is queued and
        its rid is returned, so the caller's overwrite cannot drop it."""
        self.assertEqual(self._call(_Req("keep", end=4096, prefix=4096), _Req("in")), "keep")
        self.assertEqual(self.queued, ["keep"])
        self.assertEqual(self.parked, [])

    def test_an_unparked_occupant_is_parked_before_being_re_homed(self):
        """BOOT 5's THIRD EXIT, inverted (#968b-2, end=7939 prefix=4096). The
        park-shape equality used to make this re-home REFUSE the un-parked
        mid-plan occupant and return None, and the caller's next statement
        then nulled the live occupant: a drop. It must park the prepared-but-
        never-run geometry and fall through to the same queue junction."""
        self.assertEqual(self._call(_Req("mid", end=7939, prefix=4096), _Req("in")), "mid")
        self.assertEqual(self.parked, ["mid"])
        self.assertEqual(self.queued, ["mid"])

    def test_the_reset_shape_declines_and_says_so(self):
        """No extend_range, or already retracted: nothing to re-home. The
        decline is deliberate and LOG-ONLY (#987) -- pinned here so a future
        change to the behaviour is a decision rather than a side effect."""
        self.assertIsNone(self._call(_Req("gone", end=None), _Req("in")))
        self.assertIsNone(
            self._call(_Req("dead", end=4096, prefix=4096, retracted=True), _Req("in"))
        )
        self.assertEqual(self.queued, [])

    def test_a_refused_queue_junction_reports_no_rehome(self):
        """The rid is returned iff the request actually reached the queue."""
        self.m.pp_queue_orphaned_chunked_req = lambda scheduler, req, **kw: False
        self.assertIsNone(self._call(_Req("keep", end=4096, prefix=4096), _Req("in")))

    def test_the_live_caller_still_exists(self):
        """The reason this file exists: the function outlived the test file
        that covered it. If its last caller ever goes, this pin should be
        re-read as a deletion candidate rather than kept green forever."""
        src = inspect.getsource(self.m.SchedulerPPMixin._pp_void_own_batch)
        self.assertIn("pp_rehome_displaced_chunked_req", src)


register_cpu_ci(__file__)

if __name__ == "__main__":
    unittest.main()
