"""#828 -- a funding post is worth what the ACTUATOR hands over, never what a
snapshot promises. Two edges, both read off one boot record.

THE SPECIMEN, ``boot_827_review_0823_0910c.log`` 09:11:05 PP0, four lines from
the same second::

    PHASE-FLIP-SPILL KV shrink verdict (pp_to_tp): GRANTED 75.6% ...
      this rank: KV rung: current=473088 rows -> APPLIED shrink to 357801 rows
      (1801 MiB returned)
    BACKING-DIAL call: request=357801 prev_size=349973 uniform_backed_rows=473088
      reserved_backing_rows=471654 store_bound_rows=471655 page_size=1
      delta=+7828 branch=grow
    BACKING-DIAL grow done: prev_size=349973 -> size=357801
      uniform_backed_rows 473088 -> 473088 released_bytes=0
    KV-BACKING shrink to 357801 rows released NOTHING and the pool agrees:
      it returned claimed=0 bytes ... asked 115287 rows against a release
      granularity of 8192 rows
    PHASE-FLIP staging reclaim: driver free 1436 -> 1436 MiB (+0 returned),
      334 MiB still cached
    CORRIDOR-GUARD REFUSED on device 0: want 1746 MiB, free 1436 -> 1436 MiB,
      reclaimed 0 MiB from [nothing], arming floor 1255 MiB
    ... #770 FUNDING POSTS: want 1746 MiB, covered 1870 MiB from
      [allocator-cache[local] 334 MiB; draft-weights[rebalance] 0 MiB;
      kv-slack[rebalance] 1536 MiB], cause=funded, retry_is_pointless=False

The funder says 1870 MiB is covered. The gate refuses at 1436 MiB free. The
briefing read that as "the gate rechnet auf einem eigenen free und ignoriert
den Funder". IT DOES NOT. ``free 1436`` is the true driver free, and BOTH posts
the funder credits were actually DRAWN in that same second and delivered ZERO.
The contradiction is the funder's, not the gate's.

EDGE A -- THE LATCH THAT MAKES ``kv-slack`` UNDELIVERABLE FOREVER.

``_current_rows()`` (kv_backing_relief.py:1307) reads the COMMITTED BACKING
(``uniform_backed_rows`` = 473088) and is right to: #796 measured that reading
the average depth targets a span above the shallowest buffer and releases
nothing. But ``runtime_set_backing_tokens`` branched grow-vs-shrink on
``self.size`` (349973), which is the EXPOSED id space. Between those two
numbers sits a 123115-row band, and every target that lands inside it takes
the GROW branch::

    plan:   473088 -> 357801   (115287 rows, 1801 MiB, 14 whole granules deep)
    dial:   357801 > size 349973  ->  grow  ->  released_bytes=0

So the rung asks fourteen granules deep and the actuator answers with a grow.
This is the question ``kv_backing_relief.py:1367`` left open in writing --
"all 15 zero-byte shrinks in that boot asked at least three whole granules
deep, so granularity cannot account for any of them ... the open question is
why runtime_set_backing_rows reported zero released bytes at that depth". This
is the answer, and it is a LATCH: every zero-byte shrink still assigns
``self.size = n``, so size drifts DOWN while the backing stays at the boot
reservation, the band widens, and the next plan is even more certainly inside
it. Once the two have diverged the dial can never release again.

The fix separates the two axes the one branch had fused. Backing converges
against the BACKING; the exposed size is assigned either way; and rows that
the call RE-EXPOSES are still zeroed, because the flush identity
(``zero_kv_data_buffers``) does not care which branch re-exposed them. That
last clause is the danger direction and it has its own test: under the old
code request 357801 took the grow branch and zeroed [349973, 357801) as a side
effect. A fix that only re-routes the branch would silently stop zeroing rows
that carry another request's KV.

EDGE B -- ``allocator-cache 334 MiB`` IS A PROMISE THAT DID NOT ARRIVE.

The census prices that post at ``memory_reserved() - memory_allocated()``.
That figure counts fragmented segments ``empty_cache()`` cannot return, and
the same second measured the truth: ``driver free 1436 -> 1436 (+0 returned),
334 MiB still cached``. Law 2 of :mod:`funding_authority` already says a post
is credited by DELIVERED driver bytes and carries the derating machinery
(``derate_num`` / ``derate_den``) to do it -- ``authority_from_seam_snapshot``
simply never passed the measurement in, so every verdict trusted every post at
face value forever. Wiring the measured delivery through turns
``cause=funded`` into ``cause=phantom_capacity`` on this record, which is the
difference between a refusal that misleads and one that names its defect.

Hermetic: no CUDA, no NVML, no pool. CUDA_VISIBLE_DEVICES="".
"""

import unittest

import torch

from sglang.srt.managers.funding_authority import (
    CAUSE_FUNDED,
    CAUSE_PHANTOM,
    authority_from_seam_snapshot,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

MIB = 1024 * 1024

# -- boot_827_review_0823_0910c, 09:11:05 PP0 --------------------------------
PP0_SIZE = 349973  # BACKING-DIAL prev_size
PP0_BACKED = 473088  # BACKING-DIAL uniform_backed_rows, and the rung's `current`
PP0_TARGET = 357801  # the rung's agreed shrink target
PP0_ROW_BYTES = 16384  # "commit chunk 8 MiB across 16 buffers, 16 KiB per row"
PP0_RESERVED = 471654
PP0_STORE_BOUND = 471655


class _Owner:
    """Stands in for ``KvVmmBufferOwner``: maps and decommits whole rows."""

    def __init__(self, backed: int, row_bytes: int) -> None:
        self.backed = int(backed)
        self._row_bytes = int(row_bytes)
        self.finalized = []
        self.shrunk = []

    def finalize(self, n: int) -> None:
        self.finalized.append(int(n))
        self.backed = max(self.backed, int(n))

    def shrink(self, n: int) -> int:
        n = int(n)
        self.shrunk.append(n)
        released = max(0, self.backed - n) * self._row_bytes
        self.backed = min(self.backed, n)
        return released


class _Desc:
    """Row index for a buffer slice; identity is enough for the zero-range."""

    @staticmethod
    def _rows(n: int) -> int:
        return int(n)


class _StubPool:
    """The attribute surface ``runtime_set_backing_tokens`` actually reads."""

    page_size = 1

    def __init__(self, size: int, backed: int, row_bytes: int = PP0_ROW_BYTES) -> None:
        self.size = int(size)
        self._post_capture_owner = _Owner(backed, row_bytes)
        self._kv_buffer_descs = [_Desc()]
        # One row per element: the zero-range assertions read this directly.
        self.k_buffer = [torch.ones(max(int(backed), int(size)) + 16)]
        self.v_buffer = []

    @property
    def uniform_backed_rows(self) -> int:
        return self._post_capture_owner.backed

    @property
    def reserved_backing_rows(self) -> int:
        return PP0_RESERVED

    @property
    def store_bound_rows(self) -> int:
        return PP0_STORE_BOUND


def _dial(pool, n):
    """Drive the REAL entry point against the stub, with no CUDA anywhere."""
    return MHATokenToKVPool.runtime_set_backing_tokens(pool, n)


class TheDialMustConvergeTheBackingNotTheExposedSize(unittest.TestCase):
    def test_a_target_between_size_and_backing_releases_the_band(self):
        """THE SPECIMEN. 14 granules deep, and the dial answered with a grow."""
        pool = _StubPool(PP0_SIZE, PP0_BACKED)
        released = _dial(pool, PP0_TARGET)

        self.assertGreater(
            released,
            0,
            "the rung asked 115287 rows (1801 MiB) deep and the dial released "
            "nothing: the target sits between size and backing, which is the "
            "latch this ticket closes",
        )
        # The exact figure the flip accounting already CLAIMED in the log.
        self.assertEqual(released, (PP0_BACKED - PP0_TARGET) * PP0_ROW_BYTES)
        self.assertEqual(released // MIB, 1801)
        self.assertEqual(pool._post_capture_owner.shrunk, [PP0_TARGET])

    def test_the_call_leaves_size_and_backing_converged(self):
        """The anti-latch invariant: no band survives the call to widen."""
        pool = _StubPool(PP0_SIZE, PP0_BACKED)
        _dial(pool, PP0_TARGET)
        self.assertEqual(pool.size, PP0_TARGET)
        self.assertEqual(pool.uniform_backed_rows, PP0_TARGET)

    def test_rows_this_call_re_exposes_are_still_zeroed(self):
        """THE DANGER DIRECTION.

        Under the old code the grow branch zeroed [349973, 357801) on its way
        past. Those rows are exposed by this call and may hold another
        request's KV; a fix that only re-routes the branch would hand them out
        dirty. Assert the zeroing survives the re-route.
        """
        pool = _StubPool(PP0_SIZE, PP0_BACKED)
        buf = pool.k_buffer[0]
        _dial(pool, PP0_TARGET)

        lo, hi = PP0_SIZE + _StubPool.page_size, PP0_TARGET + _StubPool.page_size
        self.assertTrue(
            bool(torch.all(buf[lo:hi] == 0)),
            "rows re-exposed by this call were left dirty",
        )
        # And it must not have zeroed rows that were already exposed and live.
        self.assertTrue(bool(torch.all(buf[:lo] == 1)))

    def test_asking_for_the_size_it_already_has_still_returns_the_band(self):
        """The de-latch. ``request == size`` is not a no-op while a band exists.

        A pool that drifted to size 349973 under a 473088-row backing is
        holding 123115 rows -- 1923 MiB on PP0 -- that NOTHING exposes and
        nothing can ever claim. Under the old early return (``if n == prev:
        return 0``) that band was unreachable by construction: the only call
        that could have released it was the one the guard treated as no work.
        """
        pool = _StubPool(PP0_SIZE, PP0_BACKED)
        released = _dial(pool, PP0_SIZE)
        self.assertEqual(released, (PP0_BACKED - PP0_SIZE) * PP0_ROW_BYTES)
        self.assertEqual(released // MIB, 1923)
        self.assertEqual(pool.uniform_backed_rows, PP0_SIZE)

    def test_the_dial_never_leaves_more_exposed_than_backed(self):
        """#816's invariant, enforced on the way out of every branch.

        A pool whose exposed size already exceeds its backing is the state
        that killed four boots on 2026-08-22/23 (``index >= size + page_size``
        in ``set_kv_buffer``). A dial that mapped against the EXPOSED size
        would leave that state standing on a request between the two.
        """
        pool = _StubPool(200000, 100000)
        _dial(pool, 150000)
        self.assertEqual(pool.size, 150000)
        self.assertGreaterEqual(
            pool.uniform_backed_rows,
            pool.size,
            "the dial returned with more id space exposed than backing behind it",
        )

    def test_a_true_grow_above_the_backing_still_grows(self):
        pool = _StubPool(PP0_SIZE, PP0_BACKED)
        released = _dial(pool, PP0_BACKED + 4096)
        self.assertEqual(released, 0)
        self.assertEqual(pool._post_capture_owner.finalized, [PP0_BACKED + 4096])
        self.assertEqual(pool._post_capture_owner.shrunk, [])
        self.assertEqual(pool.size, PP0_BACKED + 4096)

    def test_a_shrink_below_both_is_unchanged(self):
        pool = _StubPool(PP0_SIZE, PP0_BACKED)
        released = _dial(pool, 200000)
        self.assertEqual(released, (PP0_BACKED - 200000) * PP0_ROW_BYTES)
        self.assertEqual(pool._post_capture_owner.shrunk, [200000])
        self.assertEqual(pool.size, 200000)

    def test_a_true_noop_still_moves_nothing(self):
        pool = _StubPool(PP0_BACKED, PP0_BACKED)
        self.assertEqual(_dial(pool, PP0_BACKED), 0)
        self.assertEqual(pool._post_capture_owner.shrunk, [])
        self.assertEqual(pool._post_capture_owner.finalized, [])

    def test_an_undiverged_pool_keeps_its_old_behaviour_exactly(self):
        """No band, no change: size == backing is the ordinary case."""
        pool = _StubPool(100000, 100000)
        self.assertEqual(_dial(pool, 120000), 0)  # grow
        self.assertEqual(pool.size, 120000)
        self.assertEqual(pool.uniform_backed_rows, 120000)
        released = _dial(pool, 90000)  # shrink
        self.assertEqual(released, 30000 * PP0_ROW_BYTES)
        self.assertEqual(pool.size, 90000)


class APostIsCreditedByWhatItDelivered(unittest.TestCase):
    def test_the_specimen_verdict_is_not_funded(self):
        """want 1746, promise 1870, delivered 0 -- this may not read 'funded'."""
        auth = authority_from_seam_snapshot(
            allocator_cache_bytes=334 * MIB,
            allocator_cache_delivered_bytes=0,
            kv_slack_rows=0,  # the latch: the dial could reach nothing
            row_bytes=PP0_ROW_BYTES,
            kv_granule_rows=8192,
        )
        v = auth.can_fund(1746 * MIB)
        self.assertFalse(v.ok)
        self.assertNotEqual(v.cause, CAUSE_FUNDED)
        self.assertEqual(v.cause, CAUSE_PHANTOM)
        self.assertEqual(v.covered_bytes, 0)
        self.assertIn("allocator-cache", v.describe())

    def test_a_reclaim_that_returned_nothing_credits_nothing(self):
        auth = authority_from_seam_snapshot(
            allocator_cache_bytes=334 * MIB,
            allocator_cache_delivered_bytes=0,
        )
        v = auth.can_fund(100 * MIB)
        drawn = {d.post: d.drawn_bytes for d in v.draws}
        self.assertEqual(drawn["allocator-cache"], 0)

    def test_a_partial_delivery_derates_in_proportion(self):
        auth = authority_from_seam_snapshot(
            allocator_cache_bytes=400 * MIB,
            allocator_cache_delivered_bytes=100 * MIB,
        )
        v = auth.can_fund(400 * MIB)
        drawn = {d.post: d.drawn_bytes for d in v.draws}
        self.assertEqual(drawn["allocator-cache"], 100 * MIB)

    def test_an_unobserved_reclaim_is_still_trusted_once(self):
        """Law 2's own rule: den == 0 means no draw seen; trust it once.

        This is also the backward-compatibility pin -- every existing caller
        omits the new argument and must keep its old verdict.
        """
        auth = authority_from_seam_snapshot(allocator_cache_bytes=334 * MIB)
        v = auth.can_fund(300 * MIB)
        self.assertTrue(v.ok)
        self.assertEqual(v.cause, CAUSE_FUNDED)

    def test_a_delivery_above_the_promise_never_inflates_the_post(self):
        """A ratio above 1 means the measurement is wrong, not the post."""
        auth = authority_from_seam_snapshot(
            allocator_cache_bytes=100 * MIB,
            allocator_cache_delivered_bytes=900 * MIB,
        )
        v = auth.can_fund(500 * MIB)
        drawn = {d.post: d.drawn_bytes for d in v.draws}
        self.assertEqual(drawn["allocator-cache"], 100 * MIB)


class TheSeamCensusSpendsTheMeasurement(unittest.TestCase):
    """The wiring edge. A law connected to nothing is the failure mode this
    corpus has shipped repeatedly -- ``_funding_post_census`` is guarded by a
    broad ``except`` so that a refusal can never crash while explaining itself,
    and that safety means a wrong attribute name goes SILENT and is
    indistinguishable in a log from a census that was never needed. So drive
    the real method.
    """

    class _Runtime:
        _census_scheduler = None
        _rank = 0

    def _census(self, **attrs):
        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        stub = self._Runtime()
        for k, v in attrs.items():
            setattr(stub, k, v)
        return PhaseFlipRuntime._funding_post_census(stub, 1746 * MIB)

    def test_the_measured_delivery_reaches_the_refusal_line(self):
        line = self._census(
            _last_cache_promised_bytes=334 * MIB,
            _last_cache_delivered_bytes=0,
        )
        self.assertIn("#770 FUNDING POSTS", line)
        self.assertIn("cause=phantom_capacity", line)
        self.assertNotIn("cause=funded", line)
        # The promise is still NAMED (law 1: a refusal reports what it
        # considered), it is simply not credited.
        self.assertIn("allocator-cache", line)

    def test_a_pass_that_never_reclaimed_is_priced_as_before(self):
        """No measurement recorded -> unobserved -> old behaviour exactly."""
        line = self._census()
        self.assertIn("#770 FUNDING POSTS", line)
        self.assertNotIn("cause=phantom_capacity", line)

    def test_the_census_still_never_raises(self):
        """A refusal that CRASHES while explaining itself is worse than one
        that cannot explain itself. Feed it a hostile attribute."""
        line = self._census(
            _last_cache_promised_bytes="not-an-int",
            _last_cache_delivered_bytes=0,
        )
        self.assertIsInstance(line, str)


if __name__ == "__main__":
    unittest.main()
