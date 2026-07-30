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
2. **A release handshake** (flag family 4, ``ergBereit``): a reserved slot
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

from sglang.srt.distributed.device_communicators.htccl_bar1 import (
    HTCCLBar1Transport,
    bc_plan,
    fbasis_a2a,
    flags_requirement,
    geometry,
    max_payload,
)
from sglang.srt.distributed.device_communicators.htccl_bar1_pipe_ext import (
    ERG_BEREIT_FAMILIE,
    ERG_EAGER_PLAETZE,
    result_slot_split,
    result_eager_slot,
    result_graph_slot,
    result_ring_bytes,
    result_stride_bytes,
    pipe_fbasis,
    pipe_flags_extra,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


_PIPE_EXT = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/distributed/device_communicators/htccl_bar1_pipe_ext.py"
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
            self.assertEqual(eager, ERG_EAGER_PLAETZE)
            self.assertEqual(graph, ring - ERG_EAGER_PLAETZE)

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
            for fest in (False, True):
                eager, graph = result_slot_split(ring, fest)
                self.assertEqual(eager + graph, ring, (ring, fest))
                self.assertGreaterEqual(eager, 0)
                self.assertGreaterEqual(graph, 0)


# ===========================================================================
# 2. Slot arithmetic
# ===========================================================================


class TestSlotArithmetic(CustomTestCase):
    def test_eager_rotates_monotonically_modulo_and_wraps(self):
        for plaetze in (2, 3, 5):
            i = -1
            seen = []
            for _ in range(4 * plaetze):
                i = result_eager_slot(i, plaetze)
                seen.append(i)
            self.assertEqual(seen[:plaetze], list(range(plaetze)))
            # Wrap: after `plaetze` steps the sequence repeats exactly.
            self.assertEqual(seen[plaetze : 2 * plaetze], list(range(plaetze)))
            self.assertTrue(all(0 <= x < plaetze for x in seen))

    def test_eager_reuse_distance_is_the_number_of_slots(self):
        """The property the lifetime check rests on, stated as a number.

        Slot ``i`` comes round again after exactly ``plaetze`` calls -- not
        sooner. That is what makes ``ergSlack = plaetze`` the right value
        for the handshake on the eager side.
        """
        for plaetze in (2, 3, 5):
            i = -1
            folge = []
            for _ in range(6 * plaetze):
                i = result_eager_slot(i, plaetze)
                folge.append(i)
            for k, platz in enumerate(folge):
                spaeter = folge[k + 1 : k + plaetze]
                self.assertNotIn(platz, spaeter)

    def test_eager_without_slots_is_an_error_not_a_zero(self):
        with self.assertRaises(ValueError):
            result_eager_slot(-1, 0)

    def test_graph_slots_are_handed_out_once_ascending_and_then_run_out(self):
        eager, graph = 2, 3
        vergeben = []
        for k in range(graph):
            platz = result_graph_slot(k, eager, graph)
            self.assertIsNotNone(platz)
            vergeben.append(platz)
        self.assertEqual(vergeben, [2, 3, 4])
        self.assertEqual(len(set(vergeben)), len(vergeben))
        for k in range(graph, graph + 4):
            self.assertIsNone(result_graph_slot(k, eager, graph))

    def test_graph_slots_never_touch_the_eager_slots(self):
        for ring in range(3, 10):
            eager, graph = result_slot_split(ring, True)
            eager_menge = set(range(eager))
            for k in range(graph):
                self.assertNotIn(result_graph_slot(k, eager, graph), eager_menge)

    def test_an_empty_pool_yields_none_from_the_first_request_on(self):
        self.assertIsNone(result_graph_slot(0, 2, 0))


# ===========================================================================
# 3. The flag region
# ===========================================================================


class TestFlagsRegion(CustomTestCase):
    def test_five_families_now_and_the_fifth_is_ergbereit(self):
        for welt in (2, 3, 8):
            self.assertEqual(pipe_flags_extra(welt), 5 * welt * 256)
        self.assertEqual(ERG_BEREIT_FAMILIE, 4)

    def test_the_pipe_base_did_not_move(self):
        """Growing the pipe families must not shift mesh, ring or a2a.

        The pipe rows sit BEHIND everything else, so their count enters
        ``flags_requirement`` but never ``pipe_fbasis`` -- and never
        ``fbasis_a2a``. A shift here would point sender and receiver at
        different rows, which is a wrong number rather than a crash.
        """
        for welt in (2, 3, 8):
            for a2a in (False, True):
                self.assertEqual(
                    pipe_fbasis(welt, a2a),
                    (2 + 2 * (welt - 1) + (1 if a2a else 0)) * welt * 256,
                )
            self.assertEqual(fbasis_a2a(welt), (2 + 2 * (welt - 1)) * welt * 256)

    def test_every_family_row_is_disjoint_and_256_byte_aligned(self):
        welt = 3
        basis = pipe_fbasis(welt, True)
        zeilen = []
        for familie in range(5):
            for rang in range(welt):
                versatz = basis + (familie * welt + rang) * 256
                self.assertEqual(versatz % 256, 0)
                zeilen.append(versatz)
        self.assertEqual(len(set(zeilen)), len(zeilen))
        # And all of them inside the budget the transport allocates.
        bedarf = flags_requirement(welt, True, True)
        self.assertLessEqual(max(zeilen) + 256, bedarf)

    def test_without_the_pipe_the_budget_is_unchanged(self):
        for welt in (2, 3, 8):
            for a2a in (False, True):
                self.assertEqual(
                    flags_requirement(welt, a2a, False),
                    flags_requirement(welt, a2a, True) - pipe_flags_extra(welt),
                )


# ===========================================================================
# 4. The release handshake, as a pure Python simulation
# ===========================================================================


class Violation(AssertionError):
    """A rank wrote over a slot generation the owner had not released."""


class Window:
    """A pure-Python replay of the ``ergBereit`` protocol.

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

    def __init__(self, welt: int, slack: int, handshake: bool = True):
        self.welt = welt
        self.slack = slack
        self.handshake = handshake
        #: local, device-resident generation counter per rank
        self.gen = [0] * welt
        #: seen[r][z] -- what r has read from z's row in r's own window
        self.seen = [[0] * welt for _ in range(welt)]
        #: which generation currently sits in each rank's result slot
        self.content = [0] * welt
        #: highest generation each rank has finished consuming
        self.consumed = [0] * welt
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
        for z in range(self.welt):
            if z != r:
                self.seen[z][r] = self.gen[r]

    def may_write(self, r: int) -> bool:
        """The wait condition, evaluated once per call."""
        if not self.handshake:
            return True
        g = self.gen[r]
        return all(
            self.seen[r][z] + self.slack - 1 >= g for z in range(self.welt) if z != r
        )

    def write(self, r: int) -> None:
        """Write generation ``gen[r]`` into every peer's result slot."""
        g = self.gen[r]
        for z in range(self.welt):
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
        for welt in (2, 3):
            for slack in (1, 2, 3):
                f = Window(welt, slack)
                fast = 0
                blocked = 0
                for _ in range(50):
                    if not f.call(fast):
                        blocked += 1
                        # The slow rank does one call, then the fast one
                        # gets going again.
                        for z in range(1, welt):
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
        """Generations are absolute and never reset -- like ``schrittDev``.

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

    KLEIN = 256 << 20
    GROSS = 8 << 30

    def test_more_ring_slots_cost_payload_monotonically(self):
        vorher = None
        for ring in (2, 3, 4, 5, 8):
            n = max_payload(3, self.KLEIN, True, True, ring)
            self.assertGreater(n, 0, f"ring={ring}")
            if vorher is not None:
                self.assertLess(n, vorher, f"ring={ring} kostet nichts?")
            vorher = n

    def test_the_small_window_still_carries_a_capture_pool(self):
        """The concrete question for this rig, answered with a number."""
        ring = 5
        n = max_payload(3, self.KLEIN, True, True, ring)
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
            n = max_payload(3, self.KLEIN, True, True, ring)
            self.assertLessEqual(result_ring_bytes(n, ring), self.KLEIN)

    def test_the_big_window_carries_strictly_more(self):
        self.assertGreater(
            max_payload(3, self.GROSS, True, True, 5),
            max_payload(3, self.KLEIN, True, True, 5),
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
    t = HTCCLBar1Transport.__new__(HTCCLBar1Transport)
    t.pipe_direkt = True
    t.pipe_direkt_graph = False
    t._direkt_graph_gemeldet = False
    t._erg_graph_leer_gemeldet = False
    t._erg_graph_vergeben = 0
    t._erg_i = -1
    t._erg_lebt = [None, None]
    t._erg_zuletzt = [None, None]
    t._erg_zaehler = 0
    t._erg_eager_voll = 0
    t._erg_eager_voll_gemeldet = False
    t._erg_eager_plaetze = 2
    t._erg_graph_plaetze = 0
    t._eigen = (1 << 30, 0, 0)
    t._geo = {"off_erg": 4096, "erg_stride": 1 << 20, "erg_ring": 2}
    t._pipe_ext = mock.Mock()
    t._pipe_ext.bar1_erg_tensor.side_effect = lambda ptr, muster: mock.Mock(
        name=f"erg@{ptr}"
    )
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def _ohne_erfassung():
    return mock.patch(
        "sglang.srt.distributed.device_communicators.htccl.graph_capture_running",
        lambda: False,
    )


def _mit_erfassung():
    return mock.patch(
        "sglang.srt.distributed.device_communicators.htccl.graph_capture_running",
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
        with _ohne_erfassung():
            erg = t._result_slot(object())
        self.assertIsInstance(erg, tuple)
        self.assertEqual(len(erg), 3)
        _out, platz, slack = erg
        self.assertEqual(platz, 0)
        self.assertEqual(slack, 0)  # graphfest aus -> Handschlag aus

    def test_eager_rotation_is_unchanged_when_the_graph_mode_is_off(self):
        t = _stub()
        plaetze = []
        with _ohne_erfassung():
            for _ in range(5):
                _out, platz, slack = t._result_slot(object())
                plaetze.append(platz)
                self.assertEqual(slack, 0)
        self.assertEqual(plaetze, [0, 1, 0, 1, 0])

    def test_eager_gets_a_slack_only_when_the_graph_mode_is_on(self):
        t = _stub(pipe_direkt_graph=True)
        with _ohne_erfassung():
            _out, _platz, slack = t._result_slot(object())
        self.assertEqual(slack, t._erg_eager_plaetze)

    def test_capture_without_the_flag_still_refuses(self):
        t = _stub()
        with _mit_erfassung():
            self.assertIsNone(t._result_slot(object()))

    def test_capture_reserves_one_slot_per_call_site_with_slack_one(self):
        t = _stub(
            pipe_direkt_graph=True,
            _erg_graph_plaetze=3,
            _geo={"off_erg": 4096, "erg_stride": 1 << 20, "erg_ring": 5},
        )
        plaetze = []
        with _mit_erfassung():
            for _ in range(3):
                _out, platz, slack = t._result_slot(object())
                plaetze.append(platz)
                self.assertEqual(slack, 1)
        self.assertEqual(plaetze, [2, 3, 4])
        self.assertEqual(len(set(plaetze)), 3)

    def test_a_reserved_slot_is_never_handed_out_again(self):
        t = _stub(
            pipe_direkt_graph=True,
            _erg_graph_plaetze=2,
            _geo={"off_erg": 4096, "erg_stride": 1 << 20, "erg_ring": 4},
        )
        seen = []
        with _mit_erfassung():
            for _ in range(2):
                _out, platz, _slack = t._result_slot(object())
                seen.append(platz)
        with _ohne_erfassung():
            for _ in range(6):
                _out, platz, _slack = t._result_slot(object())
                self.assertNotIn(platz, seen)

    def test_an_exhausted_pool_falls_back_to_direkt_zero_not_to_a_shared_slot(self):
        t = _stub(
            pipe_direkt_graph=True,
            _erg_graph_plaetze=1,
            _geo={"off_erg": 4096, "erg_stride": 1 << 20, "erg_ring": 3},
        )
        with _mit_erfassung():
            self.assertIsNotNone(t._result_slot(object()))
            self.assertIsNone(t._result_slot(object()))
            self.assertIsNone(t._result_slot(object()))

    def test_the_pointer_follows_the_slot(self):
        t = _stub(
            pipe_direkt_graph=True,
            _erg_graph_plaetze=3,
            _geo={"off_erg": 4096, "erg_stride": 1 << 20, "erg_ring": 5},
        )
        with _mit_erfassung():
            t._result_slot(object())
            t._result_slot(object())
        zeiger = [c.args[0] for c in t._pipe_ext.bar1_erg_tensor.call_args_list]
        self.assertEqual(
            zeiger,
            [(1 << 30) + 4096 + 2 * (1 << 20), (1 << 30) + 4096 + 3 * (1 << 20)],
        )

    def test_direkt_off_short_circuits_everything(self):
        t = _stub(pipe_direkt=False)
        with _ohne_erfassung():
            self.assertIsNone(t._result_slot(object()))
        with _mit_erfassung():
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
    raise AssertionError("_CUDA_SRC nicht gefunden")


def _cpp_src() -> str:
    tree = ast.parse(_PIPE_EXT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if getattr(node.targets[0], "id", "") == "_CPP_SRC":
                return node.value.value
    raise AssertionError("_CPP_SRC nicht gefunden")


def _ohne_kommentare(text: str) -> str:
    """Line and block comments blanked, line count preserved."""
    text = re.sub(
        r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S
    )
    return re.sub(r"//[^\n]*", "", text)


def _kernel_koerper(src: str) -> str:
    anfang = src.index("__global__ void bar1_netz_pipe_kernel")
    end = src.index("// Hostseite")
    return _ohne_kommentare(src[anfang:end])


class TestKernelSourceText(CustomTestCase):
    def test_the_declaration_and_the_definition_agree(self):
        """A signature that drifts between .cpp and .cu links but misbinds.

        ``load_inline`` compiles both; a parameter added to one side only
        would either fail to link or -- worse, with implicit conversions --
        bind the wrong argument to the wrong slot.
        """
        for name in ("erg_slack", "erg_gen_dev"):
            self.assertIn(name, _cpp_src(), name)
            self.assertIn(name, _cuda_src(), name)

    def test_the_new_pointer_tables_are_only_read_while_staging(self):
        """Same rule as the mesh kernel: no dynamic indexing of ``A``.

        Kernel parameters live in constant bank 0, which has no dynamic
        indexing. One ``A.ergBereitAn[z]`` with a running ``z`` in the body
        makes nvcc copy the WHOLE parameter block into local memory, per
        thread -- measured on this codebase as STACK 64 on the mesh kernel.
        """
        koerper = _kernel_koerper(_cuda_src())
        stelle = koerper.index("__syncthreads();")
        rumpf = koerper[stelle:]
        for field in ("ergBereitAn", "ergBereitVon"):
            self.assertNotIn(
                f"A.{field}",
                rumpf,
                f"A.{field} is indexed outside the staging block -- "
                f"that puts the parameter block into local memory",
            )

    def test_the_generation_counter_is_advanced_on_both_exits(self):
        """Abort path included -- a stalled counter hangs the next call.

        A rank that left the counter alone on abort while another advanced
        it would wait, on the next call, for a generation that never
        arrives.
        """
        koerper = _kernel_koerper(_cuda_src())
        self.assertEqual(koerper.count("A.ergGenDev = ergGen"), 2)

    def test_the_handshake_is_gated_and_cannot_run_without_the_direct_mode(self):
        src = _ohne_kommentare(_cuda_src())
        self.assertIn("(A.direkt != 0) && (A.ergSlack > 0)", src)

    def test_the_wait_stands_before_the_direct_write(self):
        """Order is the content here, so it is asserted rather than assumed."""
        koerper = _kernel_koerper(_cuda_src())
        warte = koerper.index("PIPE_WARTE_ERGFREI(ergGen)")
        write = koerper.index("schreibeV4(sErgAn[z] + ziel, s)")
        self.assertLess(warte, write)

    def test_the_publish_stands_before_the_loop(self):
        koerper = _kernel_koerper(_cuda_src())
        veroeffentlichen = koerper.index("schreibeU64(sBereitAn[z], ergGen)")
        schleife = koerper.index("for (int i = 0; i < K + PP; ++i)")
        self.assertLess(veroeffentlichen, schleife)

    def test_the_flag_family_index_is_the_same_number_on_both_sides(self):
        """Python and kernel must not carry two versions of the row offset."""
        self.assertIn(f"({ERG_BEREIT_FAMILIE} * R + r) * 256u", _cuda_src())
        self.assertIn(f"({ERG_BEREIT_FAMILIE} * R + q) * 256u", _cuda_src())


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
#     (``off_a2a + (par*(R-1) + p) * a2a_schlitz``), only with a table in
#     which exactly one rank has a non-zero send length. Its slots are
#     borrowed for the duration of a round and then handed back.
#   * the direct mode RESERVES result-ring slots (``off_erg + i*stride``)
#     for the lifetime of a capture and never hands them back.
#
# A shared byte between the two is not a crash: a broadcast round would
# silently overwrite the result a replayed graph is about to hand back, and
# the caller would read plausible-looking numbers from the wrong collective.
# So it is asserted rather than assumed, in both the payload region and the
# flag region -- and each assertion is shown to have teeth by a mutant
# layout in which the two really do collide.


def _a2a_belegung(geo: dict, welt: int) -> list:
    """Every ``(anfang, length)`` the a2a kernel can write in one region.

    Straight from the address formula in ``htccl_bar1_ext.a2aZiel``: two
    halves ``par``, ``R-1`` peer positions ``p``, one slot each. broadcast
    reaches exactly these, never more -- it is that kernel with a different
    table.
    """
    slot = int(geo["a2a_schlitz"])
    basis = int(geo["off_a2a"])
    return [
        (basis + (par * (welt - 1) + p) * slot, slot)
        for par in (0, 1)
        for p in range(welt - 1)
    ]


def _erg_belegung(geo: dict, plaetze) -> list:
    """``(anfang, length)`` of the given result-ring slots."""
    basis = int(geo["off_erg"])
    stride = int(geo["erg_stride"])
    return [(basis + int(i) * stride, stride) for i in plaetze]


def _kollisionen(links: list, rechts: list) -> list:
    """Every pair of intervals that shares at least one byte."""
    treffer = []
    for a, la in links:
        for b, lb in rechts:
            if a < b + lb and b < a + la:
                treffer.append(((a, la), (b, lb)))
    return treffer


class TestBroadcastAlongsideTheResultRing(CustomTestCase):
    RINGE = (2, 3, 5, 8)
    WELTEN = (2, 3, 4, 8)
    NUTZLASTEN = (16 << 10, 512 << 10, 8 << 20)

    def _geo(self, welt: int, max_bytes: int, ring: int) -> dict:
        geo = geometry(welt, max_bytes, True, True, ring)
        self.assertGreaterEqual(geo["off_a2a"], 0)
        self.assertGreaterEqual(geo["off_erg"], 0)
        return geo

    def test_no_reserved_slot_shares_a_byte_with_an_a2a_slot(self):
        """The whole point, over the grid of shapes this rig can produce."""
        for welt in self.WELTEN:
            for max_bytes in self.NUTZLASTEN:
                for ring in self.RINGE:
                    geo = self._geo(welt, max_bytes, ring)
                    eager, graph = result_slot_split(ring, True)
                    self.assertEqual(eager + graph, ring)
                    treffer = _kollisionen(
                        _a2a_belegung(geo, welt),
                        _erg_belegung(geo, range(ring)),
                    )
                    self.assertEqual(
                        treffer, [],
                        f"R={welt}, max_bytes={max_bytes}, ring={ring}: "
                        f"a2a slot and result slot share bytes",
                    )

    def test_a_broadcast_of_any_size_stays_inside_one_a2a_slot_per_round(self):
        """The round decomposition is what bounds the footprint.

        ``bc_plan`` cuts the payload so that no round carries more than one
        slot. Without that bound the a2a block would be a lower bound on
        what broadcast touches, not the exact extent, and the disjointness
        above would say nothing about long payloads.
        """
        for welt in self.WELTEN:
            for max_bytes in self.NUTZLASTEN:
                geo = self._geo(welt, max_bytes, 5)
                slot = int(geo["a2a_schlitz"])
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
        welt, max_bytes, ring = 3, 512 << 10, 5
        geo = self._geo(welt, max_bytes, ring)
        eager, graph = result_slot_split(ring, True)
        besitzer: dict = {}

        def belege(intervalle, wer):
            for anfang, length in intervalle:
                for seite in range(anfang, anfang + length, 4096):
                    voriger = besitzer.get(seite)
                    if voriger is not None and voriger != wer:
                        self.fail(
                            f"Byte {seite} gehoert {voriger!r}, "
                            f"{wer!r} schreibt hinein"
                        )
                    besitzer[seite] = wer

        vergeben = 0
        for aufrufstelle in range(3):
            platz = result_graph_slot(vergeben, eager, graph)
            self.assertIsNotNone(platz)
            vergeben += 1
            belege(_erg_belegung(geo, [platz]), f"graph{aufrufstelle}")
            # A broadcast between two captures -- and one after the last.
            for nbytes in (128, 1 << 20):
                for _ in bc_plan(nbytes, int(geo["a2a_schlitz"])):
                    belege(_a2a_belegung(geo, welt), "broadcast")
        # And the eager result slots, which rotate the whole time.
        voriger = -1
        for _ in range(2 * eager + 1):
            voriger = result_eager_slot(voriger, eager)
            belege(_erg_belegung(geo, [voriger]), f"eager{voriger}")

    def test_the_result_ring_begins_behind_the_last_a2a_slot(self):
        """Disjoint because of the ORDER, and the order is the invariant.

        ``geometry`` counts sets: mesh, ring, a2a, pipe, and only then the
        result ring. Whoever adds a set has to add it to that count too --
        this is the assertion that notices when they do not.
        """
        for welt in self.WELTEN:
            for ring in self.RINGE:
                geo = self._geo(welt, 512 << 10, ring)
                schlitze = 2 * (welt - 1)
                a2a_ende = geo["off_a2a"] + schlitze * geo["chunk_max"]
                pipe_ende = geo["off_pipe"] + schlitze * geo["chunk_max"]
                self.assertGreaterEqual(geo["off_erg"], a2a_ende)
                self.assertGreaterEqual(geo["off_erg"], pipe_ende)
                self.assertLessEqual(
                    geo["off_erg"] + ring * geo["erg_stride"],
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
        welt, max_bytes, ring = 3, 512 << 10, 5
        geo = self._geo(welt, max_bytes, ring)
        schlitze = 2 * (welt - 1)
        for vergessen, off_erg in (
            ("a2a and pipe", 2 * schlitze * geo["chunk_max"]),
            ("a2a only", 3 * schlitze * geo["chunk_max"]),
        ):
            falsch = dict(geo)
            falsch["off_erg"] = off_erg
            self.assertNotEqual(falsch["off_erg"], geo["off_erg"], vergessen)
            # Where the a2a block sits does not change -- only the ring moved.
            treffer = _kollisionen(
                _a2a_belegung(geo, welt), _erg_belegung(falsch, range(ring))
            )
            if vergessen == "a2a and pipe":
                self.assertTrue(
                    treffer,
                    "the mutant does not collide with the a2a slots -- "
                    "then the test above checks nothing",
                )
            else:
                # Forgetting only a2a puts the ring on the PIPE block. Still
                # wrong, still caught -- by the order assertion, not by the
                # broadcast one. Both are needed; neither covers the other.
                self.assertEqual(treffer, [])
                self.assertLess(
                    falsch["off_erg"],
                    geo["off_pipe"] + schlitze * geo["chunk_max"],
                )

    def test_the_stride_is_what_separates_two_result_slots(self):
        """Not only against a2a: the ring's own slots must not overlap."""
        for max_bytes in self.NUTZLASTEN:
            geo = self._geo(3, max_bytes, 5)
            self.assertEqual(geo["erg_stride"], result_stride_bytes(max_bytes))
            self.assertGreaterEqual(geo["erg_stride"], max_bytes)
            self.assertEqual(
                _kollisionen(
                    _erg_belegung(geo, [0, 2, 4]),
                    _erg_belegung(geo, [1, 3]),
                ),
                [],
            )


class TestFlagCoexistence(CustomTestCase):
    """The second region the two features share: the flag rows."""

    def test_the_a2a_row_and_the_ergbereit_row_are_never_the_same_line(self):
        for welt in (2, 3, 4, 8):
            a2a = [fbasis_a2a(welt) + r * 256 for r in range(welt)]
            erg = [
                pipe_fbasis(welt, True) + (ERG_BEREIT_FAMILIE * welt + r) * 256
                for r in range(welt)
            ]
            self.assertEqual(set(a2a) & set(erg), set())
            # And ergBereit sits behind a2a, not merely elsewhere: the pipe
            # families were appended so no existing row could move.
            self.assertGreater(min(erg), max(a2a))

    def test_every_ergbereit_row_fits_in_the_budget_that_is_allocated(self):
        for welt in (2, 3, 4, 8):
            bedarf = flags_requirement(welt, True, True)
            letzte = (pipe_fbasis(welt, True)
                      + (ERG_BEREIT_FAMILIE * welt + welt - 1) * 256)
            self.assertLessEqual(letzte + 256, bedarf)

    def test_a_four_family_budget_would_push_ergbereit_out_of_the_region(self):
        """The falsifier for the flag side.

        Family 4 was added by the direct mode; the four older rows were left
        where they were. Had ``pipe_flags_extra`` stayed at ``4 R * 256``,
        the ergBereit rows would run past the end of the allocated flag
        region -- into whatever the allocator put there. The check above has
        to notice that, otherwise it only restates the formula.
        """
        for welt in (2, 3, 4, 8):
            alt = flags_requirement(welt, True, True) - welt * 256
            letzte = (pipe_fbasis(welt, True)
                      + (ERG_BEREIT_FAMILIE * welt + welt - 1) * 256)
            self.assertGreater(letzte + 256, alt)


if __name__ == "__main__":
    unittest.main()
