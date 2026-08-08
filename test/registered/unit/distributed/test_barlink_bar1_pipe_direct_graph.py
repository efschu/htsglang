"""The BAR1 pipe direct mode under CUDA graph replay -- arithmetic and protocol.

What the direct mode does: instead of parking the reduced result in a slot
that the receiver then copies out, the compute kernel writes it straight
into the peer's RESULT buffer. That buffer lives in the exported BAR1
window -- it is a ring slot, and the tensor ``all_reduce`` hands back is a
``from_blob`` over it. So whoever picks the slot picks the address of a
tensor the caller receives; the slot cannot be drawn inside the kernel, it
has to be fixed on the host before the kernel runs.

That is what blocked graph capture. A free-running host ring index is baked
into the graph, and several captures walk over the same slots: two graphs
end up sharing one BAR1 slot and, replayed alternately, hand each other's
numbers back. No crash.

Two pieces fix it, and they belong together:

1. **Ownership instead of rotation** (``result_slot_split``): the ring is split
   statically. Two slots keep rotating for eager calls, everything above is
   a pool from which each captured call site takes ONE slot and never
   returns it.
2. **A release handshake** (flag family 4, ``resultReady``): a reserved slot
   is rewritten on EVERY replay, so the spacing the two-slot eager ring
   provides by itself is gone. Every rank therefore publishes its
   **generation** when it enters a direct call -- "my result slot is free"
   -- and whoever wants to write into a foreign result slot waits for it.
   The generation counter lives in local VRAM and is advanced by the
   KERNEL, not the host: no host code runs on replay.

Everything here is CPU-only. No card, no window, no extension: the slot
arithmetic is a set of pure functions, the flag protocol is replayed as a
Python simulation against overwrite scenarios, and the kernel source is
checked as text. What a GPU still has to prove is in
docs/dev/INTEGRATION_R3_VALIDATION.md.
"""

import ast
import re
import unittest
from pathlib import Path
from unittest import mock

from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    BarlinkBar1Transport,
    bc_plan,
    fbase_a2a,
    flags_requirement,
    geometry,
    max_payload,
)
from sglang.srt.distributed.device_communicators.barlink_bar1_pipe_ext import (
    RESULT_READY_FAMILY,
    RESULT_EAGER_SLOTS,
    result_slot_split,
    result_eager_slot,
    result_graph_slot,
    result_ring_bytes,
    result_stride_bytes,
    pipe_fbase,
    pipe_flags_extra,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


_PIPE_EXT = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/distributed/device_communicators/barlink_bar1_pipe_ext.py"
)


# ===========================================================================
# 1. The split of the result ring
# ===========================================================================


class TestResultSlotSplit(CustomTestCase):
    def test_off_by_default_the_whole_ring_stays_eager(self):
        """The measured behaviour, byte for byte: no capture pool at all."""
        for ring in (0, 1, 2, 3, 8, 64):
            self.assertEqual(result_slot_split(ring, False), (ring, 0))

    def test_on_the_eager_side_keeps_exactly_two(self):
        for ring in (3, 4, 5, 8, 64):
            eager, graph = result_slot_split(ring, True)
            self.assertEqual(eager, RESULT_EAGER_SLOTS)
            self.assertEqual(graph, ring - RESULT_EAGER_SLOTS)

    def test_a_ring_too_small_yields_no_graph_slots_not_a_smaller_eager_side(self):
        """The eager side is never cut below two to make room for a graph.

        Two is the minimum at which round ``n`` does not write into the
        buffer the caller still holds from round ``n-1``. Trading it away
        for a capture slot would reintroduce exactly the silently
        overwritten result the ring exists to prevent.
        """
        for ring in (0, 1, 2):
            eager, graph = result_slot_split(ring, True)
            self.assertEqual(graph, 0)
            self.assertEqual(eager, ring)

    def test_the_two_sides_are_disjoint_and_cover_the_ring(self):
        for ring in range(0, 12):
            for fixed in (False, True):
                eager, graph = result_slot_split(ring, fixed)
                self.assertEqual(eager + graph, ring, (ring, fixed))
                self.assertGreaterEqual(eager, 0)
                self.assertGreaterEqual(graph, 0)


# ===========================================================================
# 2. Slot arithmetic
# ===========================================================================


class TestSlotArithmetic(CustomTestCase):
    def test_eager_rotates_monotonically_modulo_and_wraps(self):
        for slots in (2, 3, 5):
            i = -1
            seen = []
            for _ in range(4 * slots):
                i = result_eager_slot(i, slots)
                seen.append(i)
            self.assertEqual(seen[:slots], list(range(slots)))
            # Wrap: after `slots` steps the sequence repeats exactly.
            self.assertEqual(seen[slots : 2 * slots], list(range(slots)))
            self.assertTrue(all(0 <= x < slots for x in seen))

    def test_eager_reuse_distance_is_the_number_of_slots(self):
        """The property the lifetime check rests on, stated as a number.

        Slot ``i`` comes round again after exactly ``slots`` calls -- not
        sooner. That is what makes ``resultSlack = slots`` the right value
        for the handshake on the eager side.
        """
        for slots in (2, 3, 5):
            i = -1
            sequence = []
            for _ in range(6 * slots):
                i = result_eager_slot(i, slots)
                sequence.append(i)
            for k, slot in enumerate(sequence):
                later = sequence[k + 1 : k + slots]
                self.assertNotIn(slot, later)

    def test_eager_without_slots_is_an_error_not_a_zero(self):
        with self.assertRaises(ValueError):
            result_eager_slot(-1, 0)

    def test_graph_slots_are_handed_out_once_ascending_and_then_run_out(self):
        eager, graph = 2, 3
        assigned = []
        for k in range(graph):
            slot = result_graph_slot(k, eager, graph)
            self.assertIsNotNone(slot)
            assigned.append(slot)
        self.assertEqual(assigned, [2, 3, 4])
        self.assertEqual(len(set(assigned)), len(assigned))
        for k in range(graph, graph + 4):
            self.assertIsNone(result_graph_slot(k, eager, graph))

    def test_graph_slots_never_touch_the_eager_slots(self):
        for ring in range(3, 10):
            eager, graph = result_slot_split(ring, True)
            eager_set = set(range(eager))
            for k in range(graph):
                self.assertNotIn(result_graph_slot(k, eager, graph), eager_set)

    def test_an_empty_pool_yields_none_from_the_first_request_on(self):
        self.assertIsNone(result_graph_slot(0, 2, 0))


# ===========================================================================
# 3. The flag region
# ===========================================================================


class TestFlagsRegion(CustomTestCase):
    def test_five_families_now_and_the_fifth_is_result_ready(self):
        for world in (2, 3, 8):
            self.assertEqual(pipe_flags_extra(world), 5 * world * 256)
        self.assertEqual(RESULT_READY_FAMILY, 4)

    def test_the_pipe_base_did_not_move(self):
        """Growing the pipe families must not shift mesh, ring or a2a.

        The pipe rows sit BEHIND everything else, so their count enters
        ``flags_requirement`` but never ``pipe_fbase`` -- and never
        ``fbase_a2a``. A shift here would point sender and receiver at
        different rows, which is a wrong number rather than a crash.
        """
        for world in (2, 3, 8):
            for a2a in (False, True):
                self.assertEqual(
                    pipe_fbase(world, a2a),
                    (2 + 2 * (world - 1) + (1 if a2a else 0)) * world * 256,
                )
            self.assertEqual(fbase_a2a(world), (2 + 2 * (world - 1)) * world * 256)

    def test_every_family_row_is_disjoint_and_256_byte_aligned(self):
        world = 3
        base = pipe_fbase(world, True)
        lines = []
        for family in range(5):
            for rank in range(world):
                offset = base + (family * world + rank) * 256
                self.assertEqual(offset % 256, 0)
                lines.append(offset)
        self.assertEqual(len(set(lines)), len(lines))
        # And all of them inside the budget the transport allocates.
        budget = flags_requirement(world, True, True)
        self.assertLessEqual(max(lines) + 256, budget)

    def test_without_the_pipe_the_budget_is_unchanged(self):
        for world in (2, 3, 8):
            for a2a in (False, True):
                self.assertEqual(
                    flags_requirement(world, a2a, False),
                    flags_requirement(world, a2a, True) - pipe_flags_extra(world),
                )


# ===========================================================================
# 4. The release handshake, as a pure Python simulation
# ===========================================================================


class Violation(AssertionError):
    """A rank wrote over a slot generation the owner had not released."""


class Window:
    """A pure-Python replay of the ``resultReady`` protocol.

    Deliberately NOT a mock of the kernel: it models only the four things
    the protocol turns on -- the local generation counter, what each rank
    has published, what each rank has seen, and which generation currently
    occupies a rank's result slot. Everything else (payload, chunking,
    the RS/AG window) is irrelevant to the question and would only make a
    failure harder to read.

    The invariant under test: writing generation ``g`` into rank ``z``'s
    slot destroys the content of generation ``g - slack``. That content
    must already have been consumed, and it is consumed exactly when ``z``
    enters generation ``g - slack + 1`` -- because on ``z``'s stream every
    consumer of the previous result is ordered before its next call.
    """

    def __init__(self, world: int, slack: int, handshake: bool = True):
        self.world = world
        self.slack = slack
        self.handshake = handshake
        #: local, device-resident generation counter per rank
        self.gen = [0] * world
        #: seen[r][z] -- what r has read from z's row in r's own window
        self.seen = [[0] * world for _ in range(world)]
        #: which generation currently sits in each rank's result slot
        self.content = [0] * world
        #: highest generation each rank has finished consuming
        self.consumed = [0] * world
        self.waiting: set = set()

    # -- the three kernel steps ------------------------------------------
    def enter(self, r: int) -> None:
        """Kernel start: bump the generation and publish it to all peers.

        Entering generation ``g`` means every consumer of generation
        ``g-1``'s result has run -- same stream, so this is a fact, not an
        assumption.
        """
        self.gen[r] += 1
        self.consumed[r] = self.gen[r] - 1
        if not self.handshake:
            return
        for z in range(self.world):
            if z != r:
                self.seen[z][r] = self.gen[r]

    def may_write(self, r: int) -> bool:
        """The wait condition, evaluated once per call."""
        if not self.handshake:
            return True
        g = self.gen[r]
        return all(
            self.seen[r][z] + self.slack - 1 >= g for z in range(self.world) if z != r
        )

    def write(self, r: int) -> None:
        """Write generation ``gen[r]`` into every peer's result slot."""
        g = self.gen[r]
        for z in range(self.world):
            if z == r:
                continue
            ueberschrieben = g - self.slack
            if ueberschrieben > self.consumed[z]:
                raise Violation(
                    f"rank {r} writes generation {g} into rank {z}'s slot "
                    f"and thereby overwrites generation {ueberschrieben}, "
                    f"which {z} has only consumed up to {self.consumed[z]}"
                )
            self.content[z] = g

    def call(self, r: int) -> bool:
        """One whole call of rank ``r``. ``False`` = blocked on the wait."""
        if r in self.waiting:
            if not self.may_write(r):
                return False
            self.waiting.discard(r)
            self.write(r)
            return True
        self.enter(r)
        if not self.may_write(r):
            self.waiting.add(r)
            return False
        self.write(r)
        return True


class TestReleaseHandshake(CustomTestCase):
    def test_a_slow_peer_cannot_be_run_over(self):
        """One rank races ahead as far as the protocol lets it.

        With a reserved graph slot (``slack = 1``) that is not far: the
        writer has to see the peer enter the very same generation.
        """
        for world in (2, 3):
            for slack in (1, 2, 3):
                f = Window(world, slack)
                fast = 0
                blocked = 0
                for _ in range(50):
                    if not f.call(fast):
                        blocked += 1
                        # The slow rank does one call, then the fast one
                        # gets going again.
                        for z in range(1, world):
                            f.call(z)
                        f.call(fast)
                self.assertGreater(
                    blocked,
                    0,
                    f"slack={slack}: the wait condition never triggered, "
                    f"so this case tests nothing at all",
                )

    def test_without_the_handshake_a_reserved_slot_is_run_over(self):
        """The falsifier. Without it the test above proves nothing.

        ``slack = 1`` is the reserved graph slot: same slot on every
        replay. Drop the handshake and the fast rank overwrites a
        generation the peer has not consumed -- the silent wrong number
        this whole construction exists to prevent.
        """
        f = Window(2, slack=1, handshake=False)
        with self.assertRaises(Violation):
            for _ in range(5):
                f.call(0)

    def test_the_eager_ring_survives_without_the_handshake(self):
        """Why the default path needs no new traffic, stated as a test.

        With two eager slots the reuse distance is two calls, and the AG
        window already forces one call of spacing between ranks. The
        simulation shows the second call of slack is what carries it: at
        ``slack = 2`` a rank one call ahead is still safe.
        """
        f = Window(2, slack=2, handshake=False)
        for _ in range(20):
            f.call(0)
            f.call(1)
        self.assertEqual(f.gen, [20, 20])

    def test_wrap_around_of_the_generation_counter_is_not_a_special_case(self):
        """Generations are absolute and never reset -- like ``stepDev``.

        A per-round counter would leave a gap exactly at the round
        boundary. This checks the arithmetic stays consistent far from
        zero, which is where a comparison written as ``seen >= g -
        slack + 1`` would underflow if it were not kept on the left.
        """
        f = Window(3, slack=1)
        f.gen = [10**9] * 3
        f.consumed = [10**9] * 3
        for _ in range(30):
            open_set = set(range(3))
            for _attempt in range(10):
                for r in sorted(open_set):
                    if f.call(r):
                        open_set.discard(r)
                if not open_set:
                    break
            self.assertFalse(open_set, "deadlocked instead of finished")
        self.assertEqual(f.gen, [10**9 + 30] * 3)

    def test_a_rank_that_skips_the_handshake_deadlocks_rather_than_lies(self):
        """Group uniformity is a requirement, and this states its failure mode.

        If one rank ran the handshake and another did not, the first would
        wait for a flag that never comes. That is a hang -- loud, findable
        with py-spy -- and not a wrong number. Worth having written down:
        the decision to run the direct mode has to be group-uniform, which
        it is, because it follows from the SPMD call sequence alone.
        """
        f = Window(2, slack=1)
        f.enter(0)  # rank 0 enters, publishes
        # rank 1 never enters -> rank 0 must not be allowed to write
        self.assertFalse(f.may_write(0))


# ===========================================================================
# 5. Unequal windows: the ring size is set by the SMALLEST card
# ===========================================================================


class TestUnequalWindows(CustomTestCase):
    """3080 with a 256 MiB BAR against a 5090 with the full aperture.

    The ring is group-uniform -- every rank maps the same layout -- so the
    smallest window decides how many slots there can be. These are the
    numbers that decide whether the capture pool exists at all on this rig.
    """

    SMALL = 256 << 20
    LARGE = 8 << 30

    def test_more_ring_slots_cost_payload_monotonically(self):
        previous = None
        for ring in (2, 3, 4, 5, 8):
            n = max_payload(3, self.SMALL, True, True, ring)
            self.assertGreater(n, 0, f"ring={ring}")
            if previous is not None:
                self.assertLess(n, previous, f"ring={ring} costs nothing?")
            previous = n

    def test_the_small_window_still_carries_a_capture_pool(self):
        """The concrete question for this rig, answered with a number."""
        ring = 5
        n = max_payload(3, self.SMALL, True, True, ring)
        eager, graph = result_slot_split(ring, True)
        self.assertEqual((eager, graph), (2, 3))
        self.assertGreaterEqual(
            n,
            64 << 10,
            "a ring with a graph pool pushes the largest payload under "
            "64 KiB -- and then the path no longer carries the handover sizes",
        )

    def test_the_ring_cost_is_charged_against_the_mapped_length(self):
        for ring in (2, 5):
            n = max_payload(3, self.SMALL, True, True, ring)
            self.assertLessEqual(result_ring_bytes(n, ring), self.SMALL)

    def test_the_big_window_carries_strictly_more(self):
        self.assertGreater(
            max_payload(3, self.LARGE, True, True, 5),
            max_payload(3, self.SMALL, True, True, 5),
        )


# ===========================================================================
# 6. Ownership at the seam: _result_slot
# ===========================================================================


def _stub(**kw):
    """A transport instance without ``__init__``.

    Only the fields ``_result_slot`` reads. Anything not set here stays
    absent on purpose: a new condition reading a new field then fails
    loudly instead of being silently skipped.
    """
    t = BarlinkBar1Transport.__new__(BarlinkBar1Transport)
    t.pipe_direct = True
    t.pipe_direct_graph = False
    t._direct_graph_reported = False
    t._result_graph_empty_reported = False
    t._result_graph_assigned = 0
    t._result_i = -1
    t._result_alive = [None, None]
    t._result_last = [None, None]
    t._result_counter = 0
    t._result_eager_full = 0
    t._result_eager_full_reported = False
    t._result_eager_slots = 2
    t._result_graph_slots = 0
    t._own = (1 << 30, 0, 0)
    t._geo = {"off_result": 4096, "result_stride": 1 << 20, "result_ring": 2}
    t._pipe_ext = mock.Mock()
    t._pipe_ext.bar1_result_tensor.side_effect = lambda ptr, like: mock.Mock(
        name=f"result@{ptr}"
    )
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def _without_capture():
    return mock.patch(
        "sglang.srt.distributed.device_communicators.barlink.graph_capture_running",
        lambda: False,
    )


def _with_capture():
    return mock.patch(
        "sglang.srt.distributed.device_communicators.barlink.graph_capture_running",
        lambda: True,
    )


class TestResultSlotOwnership(CustomTestCase):
    def test_the_slot_comes_back_with_the_buffer_not_as_a_field(self):
        """No returned buffer whose ordering lives only in a comment.

        The slot belongs to exactly this tensor. As an object field it
        would be a shared buffer with an implicit ordering assumption --
        the defect family that has already hit this transport twice.
        """
        t = _stub()
        with _without_capture():
            got = t._result_slot(object())
        self.assertIsInstance(got, tuple)
        self.assertEqual(len(got), 3)
        _out, slot, slack = got
        self.assertEqual(slot, 0)
        self.assertEqual(slack, 0)  # graph_safe off -> handshake off

    def test_eager_rotation_is_unchanged_when_the_graph_mode_is_off(self):
        t = _stub()
        slots = []
        with _without_capture():
            for _ in range(5):
                _out, slot, slack = t._result_slot(object())
                slots.append(slot)
                self.assertEqual(slack, 0)
        self.assertEqual(slots, [0, 1, 0, 1, 0])

    def test_eager_gets_a_slack_only_when_the_graph_mode_is_on(self):
        t = _stub(pipe_direct_graph=True)
        with _without_capture():
            _out, _slot, slack = t._result_slot(object())
        self.assertEqual(slack, t._result_eager_slots)

    def test_capture_without_the_flag_still_refuses(self):
        t = _stub()
        with _with_capture():
            self.assertIsNone(t._result_slot(object()))

    def test_capture_reserves_one_slot_per_call_site_with_slack_one(self):
        t = _stub(
            pipe_direct_graph=True,
            _result_graph_slots=3,
            _geo={"off_result": 4096, "result_stride": 1 << 20, "result_ring": 5},
        )
        slots = []
        with _with_capture():
            for _ in range(3):
                _out, slot, slack = t._result_slot(object())
                slots.append(slot)
                self.assertEqual(slack, 1)
        self.assertEqual(slots, [2, 3, 4])
        self.assertEqual(len(set(slots)), 3)

    def test_a_reserved_slot_is_never_handed_out_again(self):
        t = _stub(
            pipe_direct_graph=True,
            _result_graph_slots=2,
            _geo={"off_result": 4096, "result_stride": 1 << 20, "result_ring": 4},
        )
        seen = []
        with _with_capture():
            for _ in range(2):
                _out, slot, _slack = t._result_slot(object())
                seen.append(slot)
        with _without_capture():
            for _ in range(6):
                _out, slot, _slack = t._result_slot(object())
                self.assertNotIn(slot, seen)

    def test_an_exhausted_pool_falls_back_to_direct_zero_not_to_a_shared_slot(self):
        t = _stub(
            pipe_direct_graph=True,
            _result_graph_slots=1,
            _geo={"off_result": 4096, "result_stride": 1 << 20, "result_ring": 3},
        )
        with _with_capture():
            self.assertIsNotNone(t._result_slot(object()))
            self.assertIsNone(t._result_slot(object()))
            self.assertIsNone(t._result_slot(object()))

    def test_the_pointer_follows_the_slot(self):
        t = _stub(
            pipe_direct_graph=True,
            _result_graph_slots=3,
            _geo={"off_result": 4096, "result_stride": 1 << 20, "result_ring": 5},
        )
        with _with_capture():
            t._result_slot(object())
            t._result_slot(object())
        pointers = [c.args[0] for c in t._pipe_ext.bar1_result_tensor.call_args_list]
        self.assertEqual(
            pointers,
            [(1 << 30) + 4096 + 2 * (1 << 20), (1 << 30) + 4096 + 3 * (1 << 20)],
        )

    def test_direct_off_short_circuits_everything(self):
        t = _stub(pipe_direct=False)
        with _without_capture():
            self.assertIsNone(t._result_slot(object()))
        with _with_capture():
            self.assertIsNone(t._result_slot(object()))


# ===========================================================================
# 7. Kernel source invariants
# ===========================================================================


def _cuda_src() -> str:
    tree = ast.parse(_PIPE_EXT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if getattr(node.targets[0], "id", "") == "_CUDA_SRC":
                return node.value.value
    raise AssertionError("_CUDA_SRC not found")


def _cpp_src() -> str:
    tree = ast.parse(_PIPE_EXT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if getattr(node.targets[0], "id", "") == "_CPP_SRC":
                return node.value.value
    raise AssertionError("_CPP_SRC not found")


def _without_comments(text: str) -> str:
    """Line and block comments blanked, line count preserved."""
    text = re.sub(
        r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S
    )
    return re.sub(r"//[^\n]*", "", text)


def _kernel_body(src: str) -> str:
    start_off = src.index("__global__ void bar1_mesh_pipe_kernel")
    end = src.index("// Host side")
    return _without_comments(src[start_off:end])


class TestKernelSourceText(CustomTestCase):
    def test_the_declaration_and_the_definition_agree(self):
        """A signature that drifts between .cpp and .cu links but misbinds.

        ``load_inline`` compiles both; a parameter added to one side only
        would either fail to link or -- worse, with implicit conversions --
        bind the wrong argument to the wrong slot.
        """
        for name in ("result_slack", "result_gen_dev"):
            self.assertIn(name, _cpp_src(), name)
            self.assertIn(name, _cuda_src(), name)

    def test_the_new_pointer_tables_are_only_read_while_staging(self):
        """Same rule as the mesh kernel: no dynamic indexing of ``A``.

        Kernel parameters live in constant bank 0, which has no dynamic
        indexing. One ``A.resultReadyTo[z]`` with a running ``z`` in the body
        makes nvcc copy the WHOLE parameter block into local memory, per
        thread -- measured on this codebase as STACK 64 on the mesh kernel.
        """
        body = _kernel_body(_cuda_src())
        position = body.index("__syncthreads();")
        after = body[position:]
        for field in ("resultReadyTo", "resultReadyFrom"):
            self.assertNotIn(
                f"A.{field}",
                after,
                f"A.{field} is indexed outside the staging block -- "
                f"that puts the parameter block into local memory",
            )

    def test_the_generation_counter_is_advanced_on_both_exits(self):
        """Abort path included -- a stalled counter hangs the next call.

        A rank that left the counter alone on abort while another advanced
        it would wait, on the next call, for a generation that never
        arrives.
        """
        body = _kernel_body(_cuda_src())
        self.assertEqual(body.count("A.resultGenDev = resultGen"), 2)

    def test_the_handshake_is_gated_and_cannot_run_without_the_direct_mode(self):
        src = _without_comments(_cuda_src())
        self.assertIn("(A.direct != 0) && (A.resultSlack > 0)", src)

    def test_the_wait_stands_before_the_direct_write(self):
        """Order is the content here, so it is asserted rather than assumed."""
        body = _kernel_body(_cuda_src())
        wait_pos = body.index("PIPE_WAIT_RESULT_FREE(resultGen)")
        write = body.index("writeV4(sResultTo[z] + dst, s)")
        self.assertLess(wait_pos, write)

    def test_the_publish_stands_before_the_loop(self):
        body = _kernel_body(_cuda_src())
        publish = body.index("writeU64(sReadyTo[z], resultGen)")
        loop = body.index("for (int i = 0; i < K + PP; ++i)")
        self.assertLess(publish, loop)

    def test_the_flag_family_index_is_the_same_number_on_both_sides(self):
        """Python and kernel must not carry two versions of the row offset."""
        self.assertIn(f"({RESULT_READY_FAMILY} * R + r) * 256u", _cuda_src())
        self.assertIn(f"({RESULT_READY_FAMILY} * R + q) * 256u", _cuda_src())


# ===========================================================================
# 8. Coexistence with broadcast
# ===========================================================================
#
# broadcast and the direct mode were built on separate branches and met in
# one region. They are the two users of this window that OWN bytes across
# calls, and they own them by two different rules:
#
#   * broadcast is a one-sender all_to_all. It writes into the a2a slots --
#     the same block, the same kernel, the same address formula
#     (``off_a2a + (par*(R-1) + p) * a2a_slot``), only with a table in
#     which exactly one rank has a non-zero send length. Its slots are
#     borrowed for the duration of a round and then handed back.
#   * the direct mode RESERVES result-ring slots (``off_result + i*stride``)
#     for the lifetime of a capture and never hands them back.
#
# A shared byte between the two is not a crash: a broadcast round would
# silently overwrite the result a replayed graph is about to hand back, and
# the caller would read plausible-looking numbers from the wrong collective.
# So it is asserted rather than assumed, in both the payload region and the
# flag region -- and each assertion is shown to have teeth by a mutant
# layout in which the two really do collide.


def _a2a_usage(geo: dict, world: int) -> list:
    """Every ``(start_off, length)`` the a2a kernel can write in one region.

    Straight from the address formula in ``barlink_bar1_ext.a2aDst``: two
    halves ``par``, ``R-1`` peer positions ``p``, one slot each. broadcast
    reaches exactly these, never more -- it is that kernel with a different
    table.
    """
    slot = int(geo["a2a_slot"])
    base = int(geo["off_a2a"])
    return [
        (base + (par * (world - 1) + p) * slot, slot)
        for par in (0, 1)
        for p in range(world - 1)
    ]


def _result_usage(geo: dict, slots) -> list:
    """``(start_off, length)`` of the given result-ring slots."""
    base = int(geo["off_result"])
    stride = int(geo["result_stride"])
    return [(base + int(i) * stride, stride) for i in slots]


def _collisions(left: list, right: list) -> list:
    """Every pair of intervals that shares at least one byte."""
    hits = []
    for a, la in left:
        for b, lb in right:
            if a < b + lb and b < a + la:
                hits.append(((a, la), (b, lb)))
    return hits


class TestBroadcastAlongsideTheResultRing(CustomTestCase):
    RINGS = (2, 3, 5, 8)
    WORLDS = (2, 3, 4, 8)
    PAYLOADS = (16 << 10, 512 << 10, 8 << 20)

    def _geo(self, world: int, max_bytes: int, ring: int) -> dict:
        geo = geometry(world, max_bytes, True, True, ring)
        self.assertGreaterEqual(geo["off_a2a"], 0)
        self.assertGreaterEqual(geo["off_result"], 0)
        return geo

    def test_no_reserved_slot_shares_a_byte_with_an_a2a_slot(self):
        """The whole point, over the grid of shapes this rig can produce."""
        for world in self.WORLDS:
            for max_bytes in self.PAYLOADS:
                for ring in self.RINGS:
                    geo = self._geo(world, max_bytes, ring)
                    eager, graph = result_slot_split(ring, True)
                    self.assertEqual(eager + graph, ring)
                    hits = _collisions(
                        _a2a_usage(geo, world),
                        _result_usage(geo, range(ring)),
                    )
                    self.assertEqual(
                        hits, [],
                        f"R={world}, max_bytes={max_bytes}, ring={ring}: "
                        f"a2a slot and result slot share bytes",
                    )

    def test_a_broadcast_of_any_size_stays_inside_one_a2a_slot_per_round(self):
        """The round decomposition is what bounds the footprint.

        ``bc_plan`` cuts the payload so that no round carries more than one
        slot. Without that bound the a2a block would be a lower bound on
        what broadcast touches, not the exact extent, and the disjointness
        above would say nothing about long payloads.
        """
        for world in self.WORLDS:
            for max_bytes in self.PAYLOADS:
                geo = self._geo(world, max_bytes, 5)
                slot = int(geo["a2a_slot"])
                for nbytes in (12, 128, slot - 1, slot,
                               slot + 1, 7 * slot + 3):
                    plan = bc_plan(nbytes, slot)
                    self.assertEqual(sum(length for _, length in plan), nbytes)
                    for _, length in plan:
                        self.assertLessEqual(length, slot)

    def test_a_capture_run_interleaved_with_broadcasts_keeps_every_owner(self):
        """The coexistence itself, walked as a ledger rather than argued.

        Three call sites are captured and hold their result slots for good;
        between them broadcasts of growing size run their rounds through the
        a2a halves. After every single write the ledger has to agree that
        the byte belongs to whoever wrote it.
        """
        world, max_bytes, ring = 3, 512 << 10, 5
        geo = self._geo(world, max_bytes, ring)
        eager, graph = result_slot_split(ring, True)
        owner: dict = {}

        def claim(intervals, who):
            for start_off, length in intervals:
                for page in range(start_off, start_off + length, 4096):
                    previous = owner.get(page)
                    if previous is not None and previous != who:
                        self.fail(
                            f"byte {page} belongs to {previous!r}, "
                            f"{who!r} writes into it"
                        )
                    owner[page] = who

        assigned = 0
        for call_site in range(3):
            slot = result_graph_slot(assigned, eager, graph)
            self.assertIsNotNone(slot)
            assigned += 1
            claim(_result_usage(geo, [slot]), f"graph{call_site}")
            # A broadcast between two captures -- and one after the last.
            for nbytes in (128, 1 << 20):
                for _ in bc_plan(nbytes, int(geo["a2a_slot"])):
                    claim(_a2a_usage(geo, world), "broadcast")
        # And the eager result slots, which rotate the whole time.
        previous = -1
        for _ in range(2 * eager + 1):
            previous = result_eager_slot(previous, eager)
            claim(_result_usage(geo, [previous]), f"eager{previous}")

    def test_the_result_ring_begins_behind_the_last_a2a_slot(self):
        """Disjoint because of the ORDER, and the order is the invariant.

        ``geometry`` counts sets: mesh, ring, a2a, pipe, and only then the
        result ring. Whoever adds a set has to add it to that count too --
        this is the assertion that notices when they do not.
        """
        for world in self.WORLDS:
            for ring in self.RINGS:
                geo = self._geo(world, 512 << 10, ring)
                slots = 2 * (world - 1)
                a2a_ende = geo["off_a2a"] + slots * geo["chunk_max"]
                pipe_ende = geo["off_pipe"] + slots * geo["chunk_max"]
                self.assertGreaterEqual(geo["off_result"], a2a_ende)
                self.assertGreaterEqual(geo["off_result"], pipe_ende)
                self.assertLessEqual(
                    geo["off_result"] + ring * geo["result_stride"],
                    geo["region_bytes"],
                )

    def test_a_layout_that_forgot_the_a2a_set_is_caught_by_the_same_check(self):
        """The falsifier: the disjointness check has to have teeth.

        The mutant is the plausible mistake, not an arbitrary one. A result
        ring appended by the layout as it stood before a2a existed -- mesh,
        ring, then whatever comes next -- lands on exactly the block
        broadcast writes into. ``geometry`` avoids that by counting sets;
        the check from the first test has to say so when the count is off.
        """
        world, max_bytes, ring = 3, 512 << 10, 5
        geo = self._geo(world, max_bytes, ring)
        slots = 2 * (world - 1)
        for forgotten, off_result in (
            ("a2a and pipe", 2 * slots * geo["chunk_max"]),
            ("a2a only", 3 * slots * geo["chunk_max"]),
        ):
            wrong = dict(geo)
            wrong["off_result"] = off_result
            self.assertNotEqual(wrong["off_result"], geo["off_result"], forgotten)
            # Where the a2a block sits does not change -- only the ring moved.
            hits = _collisions(
                _a2a_usage(geo, world), _result_usage(wrong, range(ring))
            )
            if forgotten == "a2a and pipe":
                self.assertTrue(
                    hits,
                    "the mutant does not collide with the a2a slots -- "
                    "then the test above checks nothing",
                )
            else:
                # Forgetting only a2a puts the ring on the PIPE block. Still
                # wrong, still caught -- by the order assertion, not by the
                # broadcast one. Both are needed; neither covers the other.
                self.assertEqual(hits, [])
                self.assertLess(
                    wrong["off_result"],
                    geo["off_pipe"] + slots * geo["chunk_max"],
                )

    def test_the_stride_is_what_separates_two_result_slots(self):
        """Not only against a2a: the ring's own slots must not overlap."""
        for max_bytes in self.PAYLOADS:
            geo = self._geo(3, max_bytes, 5)
            self.assertEqual(geo["result_stride"], result_stride_bytes(max_bytes))
            self.assertGreaterEqual(geo["result_stride"], max_bytes)
            self.assertEqual(
                _collisions(
                    _result_usage(geo, [0, 2, 4]),
                    _result_usage(geo, [1, 3]),
                ),
                [],
            )


class TestFlagCoexistence(CustomTestCase):
    """The second region the two features share: the flag rows."""

    def test_the_a2a_row_and_the_result_ready_row_are_never_the_same_line(self):
        for world in (2, 3, 4, 8):
            a2a = [fbase_a2a(world) + r * 256 for r in range(world)]
            result_rows = [
                pipe_fbase(world, True) + (RESULT_READY_FAMILY * world + r) * 256
                for r in range(world)
            ]
            self.assertEqual(set(a2a) & set(result_rows), set())
            # And resultReady sits behind a2a, not merely elsewhere: the pipe
            # families were appended so no existing row could move.
            self.assertGreater(min(result_rows), max(a2a))

    def test_every_result_ready_row_fits_in_the_budget_that_is_allocated(self):
        for world in (2, 3, 4, 8):
            budget = flags_requirement(world, True, True)
            last_row = (pipe_fbase(world, True)
                      + (RESULT_READY_FAMILY * world + world - 1) * 256)
            self.assertLessEqual(last_row + 256, budget)

    def test_a_four_family_budget_would_push_result_ready_out_of_the_region(self):
        """The falsifier for the flag side.

        Family 4 was added by the direct mode; the four older rows were left
        where they were. Had ``pipe_flags_extra`` stayed at ``4 R * 256``,
        the resultReady rows would run past the end of the allocated flag
        region -- into whatever the allocator put there. The check above has
        to notice that, otherwise it only restates the formula.
        """
        for world in (2, 3, 4, 8):
            # The budget as it would be with only four pipe families. #622
            # appended two acknowledgment banks BEHIND the pipe rows, so they
            # come off here as well -- they are not part of the four-family
            # region this falsifier is about.
            old = flags_requirement(world, True, True) - world * 256 - 2 * world * 256
            last_row = (pipe_fbase(world, True)
                      + (RESULT_READY_FAMILY * world + world - 1) * 256)
            self.assertGreater(last_row + 256, old)


if __name__ == "__main__":
    unittest.main()
