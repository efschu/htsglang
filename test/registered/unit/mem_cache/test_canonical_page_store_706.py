"""#706: the canonical page as it is PERSISTED -- partial writes, completeness,
read-time cut.

The claim under test is the one that lets the KV key drop its geometry
suffixes: a page written by three PP stages, each holding only its own layers,
is byte-for-byte the page the TP decode phase would have written alone, and
either geometry cuts its own slice out of it from the SAME key.

Every test below is written so it can fail. The three traps are planted
explicitly rather than described:

* ``test_rank_local_index_trap_never_completes`` writes with rank-LOCAL layer
  ordinals -- the right number of bytes into the wrong slots, the exact silent
  shape the module docstring warns about -- and shows what saves the store: the
  windows overlap, the high slots stay empty, the page never completes, and a
  reader gets a MISS instead of a wrong prefix.
* ``test_incomplete_page_is_refused`` first shows that the bytes cannot tell an
  unwritten slot from a legitimately zero one (they are both zero), which is
  why the refusal has to come from the marker and not from the payload.
* ``test_page_written_by_pp_is_read_by_tp`` is the two-geometry collision
  itself: one key, one page, two different readers, no suffix.
"""

import os
import tempfile
import time
import unittest

import torch

from sglang.srt.mem_cache.canonical_kv_page import (
    CanonicalPageError,
    CanonicalPageSpec,
)
from sglang.srt.mem_cache.canonical_page_store import (
    CanonicalPageWindow,
    build_page_window,
    local_attention_layer_ids,
    marker_path,
    missing_slots,
    page_is_complete,
    part_path,
    read_slice,
    sweep_partials,
    window_for_layers,
    write_slice,
)
from sglang.test.test_utils import CustomTestCase

# The live checkpoint: 64 layers, 16 of them full attention at 3, 7, ... 63
# (uniform spacing 4), 2 * 4 kv_heads * 256 head_dim * 1 B = 2048 B per token
# per attention layer. Small enough to hold in a test, real enough that the
# slot arithmetic is the deployment's arithmetic.
ATTN_LAYER_IDS = list(range(3, 64, 4))
CELL = 64  # stand-in for 2048: the arithmetic is identical, the files are small
SPEC = CanonicalPageSpec(
    num_attn_layers=len(ATTN_LAYER_IDS), kv_bytes_per_token_per_attn_layer=CELL
)

# The deployed PP cut [28, 20, 16]: stage layer ranges [0,28), [28,48), [48,64),
# which own 7 / 5 / 4 of the 16 attention layers.
PP_CUT = [(0, 28), (28, 48), (48, 64)]


def _stage_layers(lo, hi):
    """This stage's attention layers, by their GLOBAL ids -- the same filter
    ``model_runner_kv_cache_mixin`` applies with the runner's stage bounds."""
    return [i for i in ATTN_LAYER_IDS if lo <= i < hi]


def _payload(window, tag):
    """Bytes whose provenance is checkable: every byte of slot ``s`` carries
    ``tag + s``, so a slice landing in the wrong slot shows up as a wrong tag
    rather than merely a wrong length."""
    buf = bytearray()
    for slot in window.slots:
        buf += bytes([(tag + slot) % 256]) * window.cell_bytes
    return torch.frombuffer(bytes(buf), dtype=torch.uint8).clone()


def _empty(window):
    return torch.zeros(window.byte_length, dtype=torch.uint8)


class TestCanonicalPageStore(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.page = os.path.join(self._tmp.name, "deadbeef_model.bin")
        self.addCleanup(self._tmp.cleanup)

    # -- window derivation ------------------------------------------------

    def test_stage_windows_partition_the_page(self):
        windows = [
            window_for_layers(SPEC, ATTN_LAYER_IDS, _stage_layers(lo, hi))
            for lo, hi in PP_CUT
        ]
        self.assertEqual([w.num_slots for w in windows], [7, 5, 4])
        self.assertEqual([w.first_slot for w in windows], [0, 7, 12])
        covered = [s for w in windows for s in w.slots]
        self.assertEqual(covered, list(range(SPEC.num_attn_layers)))

    def test_tp_phase_window_is_the_whole_page(self):
        window = window_for_layers(SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS)
        self.assertTrue(window.is_whole_page)
        self.assertEqual(window.byte_offset, 0)
        self.assertEqual(window.byte_length, SPEC.page_bytes)

    def test_non_contiguous_layers_are_refused(self):
        with self.assertRaises(CanonicalPageError):
            window_for_layers(SPEC, ATTN_LAYER_IDS, [3, 11, 15])

    def test_a_layer_with_no_slot_is_refused(self):
        # Layer 4 is a GDN layer: it has no KV and therefore no page slot.
        with self.assertRaises(CanonicalPageError):
            window_for_layers(SPEC, ATTN_LAYER_IDS, [4])

    def test_spec_disagreement_is_refused(self):
        # The spec IS the layout contract, so a model whose attention-layer
        # count differs from the page's slot count is a different format.
        with self.assertRaises(CanonicalPageError):
            window_for_layers(SPEC, ATTN_LAYER_IDS[:-1], [3, 7])

    # -- the assembly this whole design exists for ------------------------

    def test_page_written_by_pp_is_read_by_tp(self):
        """One key, three PP writers, one TP reader. No geometry anywhere."""
        windows = [
            window_for_layers(SPEC, ATTN_LAYER_IDS, _stage_layers(lo, hi))
            for lo, hi in PP_CUT
        ]
        results = [write_slice(self.page, w, _payload(w, 10)) for w in windows]
        self.assertEqual([r.completed for r in results], [False, False, True])
        self.assertTrue(page_is_complete(self.page))

        tp_window = window_for_layers(SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS)
        out = _empty(tp_window)
        self.assertTrue(read_slice(self.page, tp_window, out))
        self.assertTrue(torch.equal(out, _payload(tp_window, 10)))

    def test_page_written_by_tp_is_cut_by_each_pp_stage(self):
        """The reverse direction, which is the flip's other half: bytes written
        as one whole page must cut into exactly the slices the PP stages own."""
        tp_window = window_for_layers(SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS)
        self.assertTrue(
            write_slice(self.page, tp_window, _payload(tp_window, 10)).completed
        )
        for lo, hi in PP_CUT:
            window = window_for_layers(SPEC, ATTN_LAYER_IDS, _stage_layers(lo, hi))
            out = _empty(window)
            self.assertTrue(read_slice(self.page, window, out))
            self.assertTrue(torch.equal(out, _payload(window, 10)))

    def test_write_order_does_not_change_the_bytes(self):
        """Three stages finish in whatever order the schedulers give them; the
        page is content-addressed, so the order must not be observable."""
        pages = []
        for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
            path = os.path.join(self._tmp.name, f"order{order[0]}{order[1]}.bin")
            for idx in order:
                lo, hi = PP_CUT[idx]
                window = window_for_layers(SPEC, ATTN_LAYER_IDS, _stage_layers(lo, hi))
                write_slice(path, window, _payload(window, 10))
            with open(path, "rb") as f:
                pages.append(f.read())
        self.assertEqual(len(set(pages)), 1)
        self.assertEqual(len(pages[0]), SPEC.page_bytes)

    # -- can-fail: the incomplete page ------------------------------------

    def test_incomplete_page_is_refused(self):
        windows = [
            window_for_layers(SPEC, ATTN_LAYER_IDS, _stage_layers(lo, hi))
            for lo, hi in PP_CUT
        ]
        write_slice(self.page, windows[0], _payload(windows[0], 10))
        write_slice(self.page, windows[2], _payload(windows[2], 10))

        self.assertFalse(page_is_complete(self.page))
        self.assertEqual(missing_slots(self.page, SPEC), tuple(range(7, 12)))

        tp_window = window_for_layers(SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS)
        out = _empty(tp_window)
        self.assertFalse(read_slice(self.page, tp_window, out))
        # Nothing was handed back: the target is untouched, not half-filled.
        self.assertTrue(torch.equal(out, _empty(tp_window)))

        # THE REASON the refusal cannot come from the payload: on disk the
        # unwritten slots are indistinguishable from written zeros. Had the
        # reader been allowed at the partial file, it would have returned a
        # full-length page whose middle five slots are silently zero.
        with open(part_path(self.page), "rb") as f:
            partial = f.read()
        self.assertEqual(len(partial), SPEC.page_bytes)
        self.assertEqual(partial[7 * CELL : 12 * CELL], b"\x00" * (5 * CELL))
        self.assertNotEqual(partial[0:CELL], b"\x00" * CELL)

        # And completing it makes the same key readable, from the same file.
        self.assertTrue(
            write_slice(self.page, windows[1], _payload(windows[1], 10)).completed
        )
        self.assertTrue(read_slice(self.page, tp_window, out))
        self.assertTrue(torch.equal(out, _payload(tp_window, 10)))

    def test_a_stage_that_needs_only_its_own_slots_still_waits(self):
        """A PP stage could serve itself from slots that ARE present. It must
        not: a page missing another stage's layers is a prefix nobody can
        continue from, and half-populating the tier would make the flip's first
        request read a prefix its own layers cannot extend."""
        windows = [
            window_for_layers(SPEC, ATTN_LAYER_IDS, _stage_layers(lo, hi))
            for lo, hi in PP_CUT
        ]
        write_slice(self.page, windows[0], _payload(windows[0], 10))
        out = _empty(windows[0])
        self.assertFalse(read_slice(self.page, windows[0], out))

    # -- can-fail: the rank-local index trap ------------------------------

    def test_rank_local_index_trap_never_completes(self):
        """Address slots with rank-LOCAL ordinals and the writes are the right
        SIZE at the wrong OFFSET -- the silent failure. What catches it is the
        marker: the three windows overlap, the top slots are never written, and
        the page stays invisible for good."""
        correct, planted = [], []
        for lo, hi in PP_CUT:
            global_ids = _stage_layers(lo, hi)
            # The mistake: renumber this stage's layers from zero, which is
            # exactly what a rank-local pool ordering hands you. On this
            # checkpoint several of those ordinals ARE attention-layer ids, so
            # the lookup succeeds and returns a plausible window.
            local_ids = [i - lo for i in global_ids]
            local_ids = [i for i in local_ids if i in ATTN_LAYER_IDS]
            correct.append(window_for_layers(SPEC, ATTN_LAYER_IDS, global_ids))
            if local_ids:
                planted.append(window_for_layers(SPEC, ATTN_LAYER_IDS, local_ids))

        # The trap's signature, stated rather than assumed: same byte count,
        # different offset.
        self.assertTrue(
            any(p.first_slot != c.first_slot for p, c in zip(planted, correct))
        )

        for window in planted:
            write_slice(self.page, window, _payload(window, 10))
        self.assertFalse(page_is_complete(self.page))
        self.assertTrue(len(missing_slots(self.page, SPEC)) > 0)

        tp_window = window_for_layers(SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS)
        out = _empty(tp_window)
        self.assertFalse(read_slice(self.page, tp_window, out))

    def test_payload_of_the_wrong_width_is_refused(self):
        """A rank whose per-layer KV size differs from the spec is not writing
        the canonical form. Refuse it at the seam rather than pad or truncate --
        a wrong-width page is the one error a content-addressed store cannot
        detect later."""
        window = window_for_layers(SPEC, ATTN_LAYER_IDS, _stage_layers(*PP_CUT[0]))
        with self.assertRaises(CanonicalPageError):
            write_slice(
                self.page,
                window,
                torch.zeros(window.byte_length - 1, dtype=torch.uint8),
            )

    # -- restarts, races, stale state -------------------------------------

    def test_rewriting_a_stage_slice_is_idempotent(self):
        """A crashed run comes back and writes its slots again. That is not a
        double-write bug -- the bytes are content-addressed and identical."""
        window = window_for_layers(SPEC, ATTN_LAYER_IDS, _stage_layers(*PP_CUT[0]))
        first = write_slice(self.page, window, _payload(window, 10))
        second = write_slice(self.page, window, _payload(window, 10))
        self.assertFalse(first.completed)
        self.assertFalse(second.completed)
        self.assertEqual(first.missing, second.missing)

    def test_writing_into_a_complete_page_is_a_no_op(self):
        tp_window = window_for_layers(SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS)
        write_slice(self.page, tp_window, _payload(tp_window, 10))
        before = os.stat(self.page).st_mtime_ns
        result = write_slice(self.page, tp_window, _payload(tp_window, 99))
        self.assertTrue(result.already_complete)
        self.assertEqual(os.stat(self.page).st_mtime_ns, before)
        out = _empty(tp_window)
        read_slice(self.page, tp_window, out)
        self.assertTrue(torch.equal(out, _payload(tp_window, 10)))

    def test_marker_from_another_page_geometry_is_discarded(self):
        """A marker that describes a different page format cannot be
        reinterpreted -- the partial page it belongs to is dropped, loudly."""
        other = CanonicalPageSpec(
            num_attn_layers=SPEC.num_attn_layers,
            kv_bytes_per_token_per_attn_layer=CELL * 2,
        )
        wide = CanonicalPageWindow(spec=other, first_slot=0, num_slots=4)
        write_slice(self.page, wide, torch.ones(wide.byte_length, dtype=torch.uint8))
        self.assertFalse(page_is_complete(self.page))

        # Now the real geometry writes the same key. The stale partial must not
        # contribute a single byte to it.
        windows = [
            window_for_layers(SPEC, ATTN_LAYER_IDS, _stage_layers(lo, hi))
            for lo, hi in PP_CUT
        ]
        for window in windows:
            write_slice(self.page, window, _payload(window, 10))
        self.assertTrue(page_is_complete(self.page))
        tp_window = window_for_layers(SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS)
        out = _empty(tp_window)
        self.assertTrue(read_slice(self.page, tp_window, out))
        self.assertTrue(torch.equal(out, _payload(tp_window, 10)))

    def test_missing_slots_of_an_absent_page_is_every_slot(self):
        self.assertEqual(
            missing_slots(self.page, SPEC), tuple(range(SPEC.num_attn_layers))
        )

    def test_a_page_of_the_wrong_width_is_not_served(self):
        """Defence against a store written under a different page format that
        somehow reached this key: cut nothing out of a page whose width does not
        match the spec."""
        with open(self.page, "wb") as f:
            f.write(b"\x01" * (SPEC.page_bytes // 2))
        window = window_for_layers(SPEC, ATTN_LAYER_IDS, _stage_layers(*PP_CUT[0]))
        out = _empty(window)
        self.assertFalse(read_slice(self.page, window, out))


class _HybridPool:
    """Stands in for ``HybridLinearKVPool``: knows its layers by GLOBAL id."""

    def __init__(self, lo, hi):
        self.full_attention_layer_id_mapping = {
            layer: i for i, layer in enumerate(_stage_layers(lo, hi))
        }


class _BoundedPool:
    """A pool that knows only its global stage bounds."""

    def __init__(self, lo, hi):
        self.start_layer = lo
        self.end_layer = hi


class _CountOnlyPool:
    """The dangerous shape: a layer COUNT and a start_layer of 0, which is what
    every host pool reports on every stage (``memory_pool_host``)."""

    def __init__(self, layer_num):
        self.layer_num = layer_num
        self.start_layer = 0
        self.end_layer = layer_num


class _HostPool:
    def __init__(self, layer_num, cell=CELL):
        self.layer_num = layer_num
        self._bytes = layer_num * cell

    def get_dummy_flat_data_page(self):
        return torch.zeros(self._bytes, dtype=torch.uint8)


class TestWindowFromPools(CustomTestCase):
    """Deriving the window from the live pools -- where the rank-local trap
    would actually be introduced."""

    def test_hybrid_pool_gives_global_ids(self):
        for lo, hi in PP_CUT:
            self.assertEqual(
                local_attention_layer_ids(_HybridPool(lo, hi), ATTN_LAYER_IDS),
                _stage_layers(lo, hi),
            )

    def test_stage_bounds_above_zero_are_the_filter_the_runner_uses(self):
        for lo, hi in PP_CUT[1:]:
            self.assertEqual(
                local_attention_layer_ids(_BoundedPool(lo, hi), ATTN_LAYER_IDS),
                _stage_layers(lo, hi),
            )

    def test_zero_based_bounds_that_do_not_span_the_model_are_refused(self):
        """Stage 0's global range [0, 28) and a rank-local renumbering are the
        same two numbers. Unanswerable from the pool alone, so it is refused
        rather than resolved -- the pool that CAN answer is the hybrid one."""
        with self.assertRaises(CanonicalPageError):
            local_attention_layer_ids(_BoundedPool(*PP_CUT[0]), ATTN_LAYER_IDS)

    def test_bounds_spanning_the_whole_model_are_the_whole_page(self):
        self.assertEqual(
            local_attention_layer_ids(_BoundedPool(0, 64), ATTN_LAYER_IDS),
            ATTN_LAYER_IDS,
        )

    def test_a_pool_holding_every_attention_layer_owns_the_page(self):
        pool = _CountOnlyPool(len(ATTN_LAYER_IDS))
        self.assertEqual(
            local_attention_layer_ids(pool, ATTN_LAYER_IDS), ATTN_LAYER_IDS
        )

    def test_a_count_only_pool_of_a_pp_stage_is_refused(self):
        """The can-fail for the wiring. A middle PP stage holds 5 attention
        layers and reports start_layer 0 -- exactly what the HOST pool reports
        on every stage. Guessing from that count would hand stage 1 the slots
        of stage 0: right byte count, wrong slots, silent. Refuse instead."""
        stage1 = _CountOnlyPool(len(_stage_layers(*PP_CUT[1])))
        with self.assertRaises(CanonicalPageError):
            local_attention_layer_ids(stage1, ATTN_LAYER_IDS)
        # And the difference that would have been silent, stated:
        correct = window_for_layers(SPEC, ATTN_LAYER_IDS, _stage_layers(*PP_CUT[1]))
        guessed = CanonicalPageWindow(spec=SPEC, first_slot=0, num_slots=5)
        self.assertEqual(correct.byte_length, guessed.byte_length)
        self.assertNotEqual(correct.byte_offset, guessed.byte_offset)

    def test_build_page_window_sizes_the_cell_from_the_page_the_pool_writes(self):
        for lo, hi in PP_CUT:
            n = len(_stage_layers(lo, hi))
            window = build_page_window(
                ATTN_LAYER_IDS, _HybridPool(lo, hi), _HostPool(n)
            )
            self.assertEqual(window.num_slots, n)
            self.assertEqual(window.cell_bytes, CELL)
            self.assertEqual(window.byte_length, n * CELL)
            self.assertEqual(window.spec, SPEC)

    def test_every_stage_agrees_on_the_page_spec(self):
        """Spec equality IS the layout contract: three stages with three
        different windows must describe the SAME page, or one key would name
        different bytes depending on who wrote it."""
        specs = {
            build_page_window(
                ATTN_LAYER_IDS,
                _HybridPool(lo, hi),
                _HostPool(len(_stage_layers(lo, hi))),
            ).spec
            for lo, hi in PP_CUT
        }
        self.assertEqual(len(specs), 1)

    def test_a_rank_whose_page_is_not_layer_divisible_is_refused(self):
        pool = _HybridPool(*PP_CUT[0])
        host = _HostPool(7)
        host._bytes = 7 * CELL + 1
        with self.assertRaises(CanonicalPageError):
            build_page_window(ATTN_LAYER_IDS, pool, host)


class TestNoStoreOwnedBuffers(CustomTestCase):
    """#550 spec note: the store allocates no host memory of its own.

    Pinned by CONTRACT rather than by measurement, which is the part that can
    actually rot: both directions take caller-owned buffers, so there is
    nothing for the process-wide pinned budget to account for. A future change
    that made the store allocate its own staging buffer would break these, and
    that change is exactly the one that would need to register with
    ``check_and_register_pinned_post``.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.page = os.path.join(self._tmp.name, "cafe.bin")
        self.addCleanup(self._tmp.cleanup)
        self.window = window_for_layers(SPEC, ATTN_LAYER_IDS, ATTN_LAYER_IDS)

    def test_read_fills_the_callers_own_tensor(self):
        write_slice(self.page, self.window, _payload(self.window, 10))
        out = _empty(self.window)
        returned = read_slice(self.page, self.window, out)
        self.assertTrue(returned)
        # Filled in place: no buffer of the store's was created and copied.
        self.assertTrue(torch.equal(out, _payload(self.window, 10)))

    def test_write_consumes_the_callers_own_tensor(self):
        payload = _payload(self.window, 10)
        before = payload.data_ptr()
        write_slice(self.page, self.window, payload)
        self.assertEqual(payload.data_ptr(), before)


class TestPartialSweep(CustomTestCase):
    """Orphaned partials: invisible to readers, untracked by the LRU evictor
    (it walks ``.bin`` only), so nothing else would ever reap them."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        os.makedirs(os.path.join(self.root, "ca"), exist_ok=True)
        self.addCleanup(self._tmp.cleanup)

    def _write(self, name, age_s=0.0):
        path = os.path.join(self.root, "ca", name)
        with open(path, "wb") as f:
            f.write(b"\x00" * 16)
        if age_s:
            old = time.time() - age_s
            os.utime(path, (old, old))
        return path

    def test_old_orphans_are_reaped(self):
        part = self._write("cafe.bin.part706", age_s=7200)
        marker = self._write("cafe.bin.slots706", age_s=7200)
        self.assertEqual(sweep_partials(self.root, older_than_s=3600), 2)
        self.assertFalse(os.path.exists(part))
        self.assertFalse(os.path.exists(marker))

    def test_a_live_partial_is_left_alone(self):
        """The dangerous direction: reaping a partial another stage is still
        filling would silently undo its work and the page would never
        complete. Age is the only safe signal available."""
        part = self._write("cafe.bin.part706", age_s=5)
        self.assertEqual(sweep_partials(self.root, older_than_s=3600), 0)
        self.assertTrue(os.path.exists(part))

    def test_complete_pages_are_never_touched(self):
        page = self._write("cafe.bin", age_s=99999)
        self.assertEqual(sweep_partials(self.root, older_than_s=1), 1 - 1)
        self.assertTrue(os.path.exists(page))

    def test_sweeping_a_missing_directory_is_harmless(self):
        self.assertEqual(
            sweep_partials(os.path.join(self.root, "nope"), older_than_s=1), 0
        )

    # -- cross-boot retention (2026-08-28 boot-3 store wipe) -----------------
    # The attach-time sweep reaped all 16898 files boot 2 had deposited into
    # /tmp/hicache_flip0828 -- every one a .part706/.slots706 pair of REAL
    # prefill work, older than the 3600 s TTL only because the next boot came
    # 100 minutes later. Age alone cannot tell abandoned garbage from work the
    # next boot would resume; the marker can: it decodes iff it was written
    # under the geometry this attach is about to write.

    def _deposit(self, stem="cafe01", age_s=7200.0):
        """A GENUINE partial pair: one PP stage's real deposit, aged."""
        final = os.path.join(self.root, "ca", f"{stem}.bin")
        window = window_for_layers(SPEC, ATTN_LAYER_IDS, _stage_layers(0, 28))
        write_slice(final, window, _payload(window, tag=10))
        part = part_path(final)
        marker = marker_path(final)
        for path in (part, marker):
            old = os.stat(path).st_mtime - age_s
            os.utime(path, (old, old))
        return final, part, marker

    def test_resumable_same_format_partials_survive_any_age(self):
        """Computed work is never thrown away (Kein-Doppel-Prefill): a partial
        whose marker decodes against THIS attach's page geometry is resumable
        deposited work, whatever its age."""
        final, part, marker = self._deposit(age_s=7200)
        reaped = sweep_partials(
            self.root, older_than_s=3600, resumable_totals=(SPEC.page_bytes,)
        )
        self.assertEqual(reaped, 0)
        self.assertTrue(os.path.exists(part))
        self.assertTrue(os.path.exists(marker))
        # The retention has VALUE: the next boot's stages complete the page.
        for lo, hi in PP_CUT[1:]:
            window = window_for_layers(SPEC, ATTN_LAYER_IDS, _stage_layers(lo, hi))
            write_slice(final, window, _payload(window, tag=10))
        self.assertTrue(page_is_complete(final))

    def test_a_foreign_geometry_pair_is_still_reaped_and_named(self):
        """A marker for a DIFFERENT page geometry is a format transition, not
        resumable work: it can never complete under this attach (any new
        writer resets it in place), so the TTL reap stands -- but it must be
        LOUD, never a silent wipe."""
        final, part, marker = self._deposit(age_s=7200)
        with self.assertLogs(
            "sglang.srt.mem_cache.canonical_page_store", level="WARNING"
        ) as logs:
            reaped = sweep_partials(
                self.root,
                older_than_s=3600,
                resumable_totals=(SPEC.page_bytes * 2,),
            )
        self.assertEqual(reaped, 2)
        self.assertFalse(os.path.exists(part))
        self.assertFalse(os.path.exists(marker))
        self.assertIn("geometry", "\n".join(logs.output))

    def test_a_young_foreign_pair_is_left_to_its_ttl(self):
        """The TTL guard is not weakened in the other direction: a young pair
        is presumed live even when its geometry is foreign."""
        final, part, marker = self._deposit(age_s=5)
        reaped = sweep_partials(
            self.root, older_than_s=3600, resumable_totals=(SPEC.page_bytes * 2,)
        )
        self.assertEqual(reaped, 0)
        self.assertTrue(os.path.exists(part))
        self.assertTrue(os.path.exists(marker))

    def test_a_markerless_partial_is_not_resumable(self):
        """No marker = no recorded coverage: the next writer resets the file
        in place anyway, so keeping it retains nothing."""
        part = self._write("cafe.bin.part706", age_s=7200)
        reaped = sweep_partials(
            self.root, older_than_s=3600, resumable_totals=(SPEC.page_bytes,)
        )
        self.assertEqual(reaped, 1)
        self.assertFalse(os.path.exists(part))

    def test_an_orphan_marker_is_not_resumable(self):
        """A marker whose partial is gone records coverage of bytes that no
        longer exist: keeping it retains nothing."""
        final, part, marker = self._deposit(age_s=7200)
        os.unlink(part)
        reaped = sweep_partials(
            self.root, older_than_s=3600, resumable_totals=(SPEC.page_bytes,)
        )
        self.assertEqual(reaped, 1)
        self.assertFalse(os.path.exists(marker))


if __name__ == "__main__":
    unittest.main()
