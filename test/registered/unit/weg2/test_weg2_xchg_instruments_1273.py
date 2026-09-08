# SPDX-License-Identifier: Apache-2.0
"""#1273 slice S7: the remap instrument and the W55 residency line.

TWO THINGS, and both exist because a number that nobody can observe is a
number that gets asserted instead of measured.

* **The remap instrument.** ``resume`` pass 1 maps every allocation of a tag
  one at a time (``cu_mem_create`` + ``cuMemMap`` + ``cu_mem_set_access``), so
  its wall is proportional to an ALLOCATION COUNT that no log has ever carried.
  With the count absent, pass 1's time was charged to the copy rate: every
  ``GB/s`` on a ``WEG2-FLIP-TAG`` line had the wrong denominator, and ADDENDUM 3
  section 4's one unexplained remainder had nowhere to be attributed.  The
  weight exchange does not change that cost in either arm -- which is exactly
  why both arms must be able to measure it (spec risk R2).

* **W55.** The exchange's VRAM peak is spec section 5's arithmetic over a
  measured census, and spec section 9 point 7 says the honest thing about it:
  *"no cell is separately observable."*  So it is checked at LAUNCH, against
  live NVML, and it refuses -- where W32/W34/W49 already refuse, before either
  group starts, because a refusal that arrives three waves into a flip arrives
  after the source's pages are unmapped.

THE TWO TESTS THE SPEC NAMES BY NAME are
``test_flip_tag_line_carries_allocations_and_map_ms`` and
``test_launch_refuses_when_a_wave_peak_exceeds_the_card``; the rest are the
can-fail proofs around them.  A gate that cannot fail is not a gate, so the
refusal test is built from a census that overflows exactly ONE card in exactly
ONE direction and asserts the other five cases stay silent.

DIRECTION TOKENS.  ``d2p`` = bytes move D -> P = **D sleeps, P wakes**.  That is
``ring_table.CardRing.need_d2p_mib``'s own docstring and the plain reading of
the arrow.  The spec's example acceptance line prints the D-WAKES triple under
``(d2p)``, i.e. the opposite binding; this tree keeps ONE meaning for the token
and the drift is recorded in WEG2_BUILD_DECISIONS_0906 section 1ai-S7.
"""

import ast
import json
import os
import re
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.test_utils import CustomTestCase

# NO WIDE SKIP GUARD.  Every symbol this file reaches for is one this slice
# adds; an ImportError on it is the regression the file exists to catch, and a
# module-level skip would make the red-first proof impossible to obtain (the
# measured lesson of test_weg2_launcher_argv_1235's own header).
from sglang.srt.weg2 import ring_table


def _repo_root() -> str:
    here = os.path.abspath(__file__)
    for _ in range(8):
        here = os.path.dirname(here)
        if os.path.isdir(os.path.join(here, "python", "sglang", "srt", "weg2")):
            return here
    raise AssertionError("could not locate the repo root from this test file")


ROOT = _repo_root()
WEIGHT_UPDATER = os.path.join(
    ROOT, "python", "sglang", "srt", "managers", "scheduler_components",
    "weight_updater.py",
)
CORE_CPP = os.path.join(ROOT, "python", "sglang", "srt", "weg2", "tms_csrc", "core.cpp")
CORE_H = os.path.join(ROOT, "python", "sglang", "srt", "weg2", "tms_csrc", "core.h")
ENTRYPOINT_CPP = os.path.join(
    ROOT, "python", "sglang", "srt", "weg2", "tms_csrc", "entrypoint.cpp"
)

#: ``%s``, ``%d``, ``%.0f`` ... in a logging format string.
_SPEC = re.compile(r"%(?:\.\d+)?[sdfg]")


def _flip_tag_format(direction: str) -> str:
    """The ``WEG2-FLIP-TAG dir=<direction>`` format string, out of the source.

    Read from the AST rather than from a copy kept here, because a test that
    keeps its own copy of the emitter's format string stops testing the emitter
    the first time one of the two is edited.
    """
    tree = ast.parse(open(WEIGHT_UPDATER, encoding="utf-8").read())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            text = first.value
        elif isinstance(first, ast.BinOp) or isinstance(first, ast.JoinedStr):
            continue
        else:
            continue
        if text.startswith("WEG2-FLIP-TAG") and f"dir={direction} " in text:
            found.append((text, len(node.args) - 1))
    assert found, f"no WEG2-FLIP-TAG dir={direction} emitter found in {WEIGHT_UPDATER}"
    assert len(found) == 1, f"more than one dir={direction} emitter: {len(found)}"
    return found[0][0]


def _render(fmt: str) -> str:
    """Format ``fmt`` with a plausible value per conversion spec."""
    values = []
    for spec in _SPEC.findall(fmt):
        if spec.endswith("s"):
            values.append("X")
        elif spec.endswith("d"):
            values.append(7)
        else:
            values.append(1.5)
    # the fields whose VALUES the parser reads are positional; give the ones the
    # regex captures shapes it can actually accept
    return fmt % tuple(values)


class _FakeAdapter:
    """The two calls the emitter makes on the saver, and nothing else."""

    def __init__(self, stats_by_tag):
        self._stats = dict(stats_by_tag)
        self.resumed = []
        self.stats_calls = []

    def resume(self, tag):
        self.resumed.append(tag)

    def resume_stats(self, tag):
        self.stats_calls.append(tag)
        return self._stats.get(tag)


class _FakeSelf:
    """The three attributes the emitter reaches for on the scheduler."""

    def __init__(self, adapter):
        self.memory_saver_adapter = adapter

    def _weg2_group_name(self):
        return "P"

    def _weg2_rank(self):
        return 3


class _FakeLogger:
    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args)


def _emitter_sources():
    """The two S7 statements of the wake path, as EXECUTABLE source.

    ROUND-2 REVIEW F2 IS WHY THIS EXISTS.  Every emitter test in the first cut
    of this file was a source-text or AST grep: they asserted the format string
    and the presence of both argument expressions IN ANY ORDER, so swapping the
    last two arguments -- printing the copy cost under ``map_ms=`` and the map
    cost under ``copy_ms=`` -- left all 17 tests green.  That is the class-A
    instrument lie this slice's own docstrings invoke, one field over, and only
    a RENDERED line can see it.

    The statements are lifted out of the module's AST rather than copied here,
    for the same reason :func:`_flip_tag_format` reads the format string out of
    it: a test carrying its own copy of the emitter stops testing the emitter
    the first time one of the two is edited.
    """
    src = open(WEIGHT_UPDATER, encoding="utf-8").read()
    tree = ast.parse(src)
    stats, loops = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            seg = ast.get_source_segment(src, node)
            if seg and "memory_saver_adapter.resume_stats(tag)" in seg:
                stats.append((node, seg))
        elif isinstance(node, ast.For):
            seg = ast.get_source_segment(src, node)
            if seg and "WEG2-FLIP-TAG" in seg and "dir=h2d " in seg:
                loops.append((node, seg))
    assert len(stats) == 1, f"expected one resume_stats read, got {len(stats)}"
    assert loops, "no loop containing the dir=h2d emitter"
    # the INNERMOST enclosing loop: ast.walk also yields every loop around it
    node, seg = min(loops, key=lambda pair: len(pair[1]))
    import textwrap

    return (
        textwrap.dedent(" " * stats[0][0].col_offset + stats[0][1]),
        textwrap.dedent(" " * node.col_offset + seg),
    )


def _drive_emitter(stats_by_tag, per_tag):
    """Run the real statements over a fake saver; return the rendered lines.

    The values are chosen so that every S7 field carries a number that appears
    nowhere else on the line: an ordering defect shows up as the wrong number
    beside the right token, which is the only shape of it a reader can catch.
    """
    adapter = _FakeAdapter(stats_by_tag)
    logger = _FakeLogger()
    ns = {
        "self": _FakeSelf(adapter),
        "logger": logger,
        "weg2_map_stats": {},
        "weg2_per_tag": dict(per_tag),
        "card_uuid": "GPU-fake-card",
        "MIB_": 1024 * 1024,
        "TMS_RING_GRANULE_BYTES": 32 * 1024 * 1024,
        "WEG2_TAG_POPULATION_WEIGHTS": "weights",
    }
    stats_src, emit_src = _emitter_sources()
    for tag in per_tag:
        ns["tag"] = tag
        exec(compile(stats_src, WEIGHT_UPDATER, "exec"), ns)  # noqa: S102
    exec(compile(emit_src, WEIGHT_UPDATER, "exec"), ns)  # noqa: S102
    return logger.lines, adapter


class XchgInstrumentTest(CustomTestCase):
    # ---------------------------------------------------------------- S7 (1)

    def test_flip_tag_line_carries_allocations_and_map_ms(self):
        """The wake line carries the remap cost AND its denominator.

        Spec S7 acceptance: ``WEG2-FLIP-TAG ... allocations=<n> map_ms=<n>
        copy_ms=<n>``.  ``allocations`` is not decoration -- it is the
        denominator that makes ``map_ms`` interpretable at all, which is why
        the two are asserted together and why neither may appear alone.
        """
        fmt = _flip_tag_format("h2d")
        self.assertIn("allocations=%s", fmt)
        self.assertIn("map_ms=%s", fmt)
        self.assertIn("copy_ms=%s", fmt)
        # appended, not interleaved: the ring planner's parser anchors on the
        # fields before them
        self.assertTrue(
            fmt.rstrip().endswith("allocations=%s map_ms=%s copy_ms=%s"),
            f"the three S7 fields must close the line, got: {fmt!r}",
        )
        rendered = _render(fmt)
        self.assertRegex(rendered, r"allocations=\S+ map_ms=\S+ copy_ms=\S+")

    def test_ring_table_tag_regex_still_parses_the_extended_line(self):
        """Appending fields must not blind the ring planner (W37's whole class).

        The instrument that made MEASURED possible is the instrument that made
        every ring-era boot ineligible once its line format moved past the
        parser.  The same trap, one slice later.
        """
        rendered = _render(_flip_tag_format("h2d"))
        self.assertIsNotNone(
            ring_table._TAG_RE.search(rendered),
            f"ring_table._TAG_RE no longer parses the wake line: {rendered!r}",
        )

    def test_the_emitter_binds_each_field_to_its_own_number(self):
        """The RENDERED line, from the module's own statements over a fake saver.

        Round-2 review F2: the argument order was untested.  Swapping the last
        two arguments of the emitter prints the copy cost as ``map_ms`` and the
        map cost as ``copy_ms`` -- a line whose every token is right and whose
        two numbers are exchanged, published under the name of the measurement
        the whole slice exists to make.  Distinct values per field, asserted as
        one contiguous substring, is what makes that visible.
        """
        lines, adapter = _drive_emitter(
            {"w_mapped": {"allocations": 271, "map_ms": 111.1, "copy_ms": 222.2}},
            {"w_mapped": [3.0 * 1024 * 1024 * 1024, 1000.0]},
        )
        self.assertEqual(len(lines), 1)
        line = lines[0]
        self.assertIn("allocations=271 map_ms=111.1 copy_ms=222.2", line)
        self.assertIn("tag=w_mapped", line)
        self.assertIn("bytes=3072 MiB", line)
        # the fixture is exercised, not merely defined: the emitter's stats read
        # goes through the adapter, once, naming the tag it is about to print
        self.assertEqual(adapter.stats_calls, ["w_mapped"])

    def test_the_rendered_line_prints_na_for_an_absent_record(self):
        """The absence, rendered -- not asserted as source text.

        Denominator law, executed: a hook without the symbol yields None, and
        the line must carry ``n/a`` in all three fields rather than a zero that
        reads as "the remap was free".
        """
        lines, adapter = _drive_emitter({}, {"w_absent": [1024 * 1024, 5.0]})
        self.assertEqual(len(lines), 1)
        self.assertIn("allocations=n/a map_ms=n/a copy_ms=n/a", lines[0])
        self.assertEqual(adapter.stats_calls, ["w_absent"])

    def test_the_emitter_reads_the_stats_inside_the_resume_loop(self):
        """One record, so it must be read before the next tag overwrites it.

        The saver keeps ONE resume record.  Reading it after the loop would
        publish the LAST tag's map cost under every tag's name -- the class-A
        instrument lie, and the cheapest possible way to get this wrong.
        """
        src = open(WEIGHT_UPDATER, encoding="utf-8").read()
        resume_call = src.index("self.memory_saver_adapter.resume(tag)")
        stats_call = src.index("self.memory_saver_adapter.resume_stats(tag)")
        leg_done = src.index("weg2_leg_ms = (time.perf_counter() - t_w0) * 1000")
        self.assertLess(resume_call, stats_call)
        self.assertLess(
            stats_call, leg_done,
            "resume_stats is read after the resume loop closed: by then the "
            "record belongs to the last tag, not to the tag being printed",
        )

    def test_absent_instrument_prints_na_never_zero(self):
        """A hook without the symbol yields ``n/a``; 0 would read as 'free'.

        Denominator law: an absence and a measured zero are different findings,
        and here the zero is the exact claim under test (that the remap costs
        nothing).  Rendering the emitter's own argument expressions.
        """
        src = open(WEIGHT_UPDATER, encoding="utf-8").read()
        block = src[src.index("dir=h2d tag=%s bytes=%d MiB"):]
        block = block[: block.index(")\n")]
        self.assertIn('"n/a" if st is None else int(st["allocations"])', block)
        self.assertIn('"n/a" if st is None else "%.1f" % st["map_ms"]', block)
        self.assertIn('"n/a" if st is None else "%.1f" % st["copy_ms"]', block)

    def test_adapter_refuses_a_record_that_names_another_tag(self):
        """resume_stats(tag) is None unless the RECORD names that tag.

        The recorder and the reader are two calls.  Without the check, anything
        that resumes between them silently re-attributes the cost -- and this is
        the mutant the S7 record names on the danger direction.
        """
        import ctypes

        from sglang.srt.utils import torch_memory_saver_adapter as tmsa

        class _FakeSymbol:
            restype = None
            argtypes = None

            def __init__(self, tag, allocations, map_ms, copy_ms, seq=1):
                self.tag, self.n = tag.encode(), allocations
                self.map_ms, self.copy_ms, self.seq = map_ms, copy_ms, seq

            def __call__(self, buf, buflen, allocations, map_ms, copy_ms):
                buf.value = self.tag
                allocations._obj.value = self.n
                map_ms._obj.value = self.map_ms
                copy_ms._obj.value = self.copy_ms
                return self.seq

        adapter = tmsa._TorchMemorySaverAdapterReal()
        original = tmsa._weg2_ring_symbol
        try:
            tmsa._weg2_ring_symbol = lambda name: _FakeSymbol("weights_3", 137, 42.0, 0.5)
            got = adapter.resume_stats("weights_3")
            self.assertEqual(got, {"allocations": 137, "map_ms": 42.0, "copy_ms": 0.5})
            self.assertIsNone(
                adapter.resume_stats("weights_4"),
                "a record naming weights_3 was handed back for weights_4",
            )
            # seq == 0: nothing has ever been recorded in this process
            tmsa._weg2_ring_symbol = lambda name: _FakeSymbol("weights_3", 0, 0.0, 0.0, seq=0)
            self.assertIsNone(adapter.resume_stats("weights_3"))
            # no such symbol at all (stock wheel / pre-S7 .so)
            tmsa._weg2_ring_symbol = lambda name: None
            self.assertIsNone(adapter.resume_stats("weights_3"))
        finally:
            tmsa._weg2_ring_symbol = original
        # the fake receives ctypes.byref() objects; asserting the module is
        # importable here keeps that dependency explicit
        self.assertTrue(hasattr(ctypes, "byref"))

    # ---------------------------------------------------------------- C++ seam

    def test_core_cpp_records_between_the_passes_it_names(self):
        """The two timestamps bracket pass 1, and the record follows pass 3.

        Asserted on the SOURCE because this file is compiled into an ``.so``
        the desk cannot build: the placement IS the claim (``map_ms`` is pass 1
        alone), and a moved line would silently redefine the number.
        """
        src = open(CORE_CPP, encoding="utf-8").read()
        resume_at = src.index("void TorchMemorySaver::resume(")
        body = src[resume_at:]
        i_pass1 = body.index("// --- pass 1: map every allocation of the tag ---")
        i_t0 = body.index("const auto weg2_map_t0")
        i_pass2 = body.index("// --- pass 2: async H2D per granule ---")
        i_t1 = body.index("const auto weg2_map_t1")
        i_pass3 = body.index("// --- pass 3: ONE synchronisation for the whole tag ---")
        i_note = body.index("note_resume(tag, matched_ptrs.size(), weg2_map_t0, weg2_map_t1)")
        i_pass4 = body.index("// --- pass 4: give the host bytes back ---")
        self.assertLess(i_pass1, i_t0, "t0 must be taken at the head of pass 1")
        self.assertLess(i_t0, i_t1)
        self.assertLess(i_t1, i_pass2, "t1 must lie BETWEEN pass 1 and pass 2")
        self.assertLess(i_pass3, i_note, "the record must follow pass 3's sync")
        self.assertLess(i_note, i_pass4, "pass 4's release is in neither term")
        # the count is the denominator and comes from the matched set, never
        # from allocation_metadata_.size() (which spans every tag)
        self.assertIn("matched_ptrs.size()", body[i_note - 200 : i_note + 200])

    def test_entrypoint_exposes_the_resume_instrument(self):
        """A C entry with the tag written back, so the reader can verify it."""
        src = open(ENTRYPOINT_CPP, encoding="utf-8").read()
        self.assertIn("uint64_t tms_resume_stats(char* tag, size_t tag_len,", src)
        head = src.index('extern "C" {')
        self.assertGreater(src.index("tms_resume_stats"), head)
        core_h = open(CORE_H, encoding="utf-8").read()
        self.assertIn("uint64_t resume_stats(char* tag_out", core_h)
        self.assertIn("last_resume_seq_ = 0", core_h)

    # ---------------------------------------------------------------- S7 (2)

    def test_launch_refuses_when_a_wave_peak_exceeds_the_card(self):
        """W55 by name, naming card, direction, wave, peak, total and floor.

        The census below overflows card 1 in the direction where **P wakes**
        (``d2p``) and in no other case, so the test also proves the gate is
        silent on the five cases that fit -- a refusal that fires on everything
        is not a gate either.
        """
        from sglang.srt.weg2 import launcher, xchg_residency

        census = _sb4_census()
        # Card 1 is the tight one: 4136 MiB free at its P-wake peak.  Add
        # 4000 MiB of never-paused resident bytes there and that margin is
        # below the arming floor while every other case still clears it.
        census["cards"][_UUID[1]]["tags"]["D"]["weights_resident_probe"] = 4000
        lines = []
        with _census_file(census) as path:
            with self.assertRaises(xchg_residency.Weg2XchgResidencyUnarmable) as ctx:
                launcher.prepare_weight_exchange(
                    _cards(), lines.append, "exchange", path, "b.0", 0,
                )
        msg = str(ctx.exception)
        self.assertIn("W55 Weg2XchgResidencyUnarmable", msg)
        self.assertIn(_UUID[1], msg)
        self.assertIn("nvml1", msg)
        self.assertIn("dir=d2p", msg)
        self.assertIn("D sleeps, P wakes", msg)
        self.assertRegex(msg, r"wave=\d+")
        self.assertRegex(msg, r"predicted peak \d+ MiB")
        self.assertIn("NVML total 20480 MiB", msg)
        self.assertIn("arming floor 1229 MiB", msg)
        self.assertIn("image_S=", msg)
        self.assertIn("2x_dormant_proc_used=", msg)
        # ONE case, not six: the other two cards and the other direction fit
        self.assertNotIn("dir=p2d", msg)
        self.assertNotIn(_UUID[0], msg)
        self.assertNotIn(_UUID[2], msg)
        self.assertFalse(
            any(ln.startswith("WEG2-XCHG-ARMED") for ln in lines),
            "a refusing launch must not also print the arming line",
        )

    def test_w55_leaves_cli_as_the_named_line_and_exit_2(self):
        """The refusal must be ENROLLED in the handler, not merely promise it.

        ROUND-2 REVIEW F1.  W55 shipped as a bare ``RuntimeError``, a subclass
        of none of ``launcher.REFUSALS``' members, so ``cli()`` did not catch
        it: exit **1** with a raw traceback -- which any wrapper keying on the
        exit code reads as a crash rather than as the refusal it is -- and
        ``drop_admin_key_file()`` never ran, leaving the secret of a boot that
        never served behind (#1275 fix 2).  Three docstrings in this slice
        promised exit 2 while the code did not deliver it.

        This is boot weg2rg1's W34 defect one module later, and the guard it
        produced (``test_weg2_ring_ledger_1235`` ::
        ``test_every_refusal_of_this_module_leaves_cli_as_the_named_line_and_exit_2``)
        asserts on the CLASS for exactly that reason.  Here the assertion goes
        one step further and drives ``cli()`` itself, because a subclass check
        proves enrolment and only a call proves the handler runs.
        """
        import contextlib
        import io

        from sglang.srt.weg2 import launcher, xchg_residency

        self.assertTrue(
            issubclass(xchg_residency.Weg2XchgResidencyUnarmable, launcher.REFUSALS),
            "W55 is not enrolled in launcher.REFUSALS: it will exit 1",
        )
        self.assertTrue(
            issubclass(
                xchg_residency.Weg2XchgResidencyUnarmable,
                xchg_residency.Weg2XchgRefused,
            ),
            "a future exchange refusal must inherit the handler, not re-enrol",
        )

        def boom(argv=None):
            raise xchg_residency.Weg2XchgResidencyUnarmable(
                "W55 Weg2XchgResidencyUnarmable: synthetic, for the handler only"
            )

        real_main = launcher.main
        launcher.main = boom
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = launcher.cli([])
        finally:
            launcher.main = real_main
        out = buf.getvalue()
        self.assertEqual(rc, 2, out)
        self.assertIn("WEG2-LAUNCH REFUSED", out)
        self.assertIn("W55 Weg2XchgResidencyUnarmable", out)
        self.assertIn("admin key file", out)

    def test_sb4_census_reproduces_the_spec_residency_table(self):
        """Every cell of spec section 5.1 / 5.2, from the census, not from prose.

        This is the indicator check for the whole slice: if the solver does not
        reproduce the six peaks and six free-at-peak figures the spec derived by
        hand from boot weg2sb4's own census, then the number W55 refuses on is
        not the number the design was argued from.
        """
        from sglang.srt.weg2 import xchg_residency

        res = xchg_residency.solve(_cards(), _inline_census(_sb4_census()), 1229.0)
        self.assertEqual(res.refusals, [])
        # spec section 5.1, "D WAKES (P sleeps -> D wakes)" == this tree's p2d
        self.assertEqual(
            [res.peak_mib("p2d", u) for u in _UUID], [25510, 14305, 15550]
        )
        self.assertEqual(
            [res.free_at_peak_mib("p2d", u) for u in _UUID], [7097, 6175, 4930]
        )
        # spec section 5.2, "P WAKES (D sleeps -> P wakes)" == this tree's d2p
        self.assertEqual(
            [res.peak_mib("d2p", u) for u in _UUID], [23060, 16344, 15664]
        )
        self.assertEqual(
            [res.free_at_peak_mib("d2p", u) for u in _UUID], [9547, 4136, 4816]
        )
        self.assertEqual(res.wave1_ok(), (6, 6))
        self.assertEqual(res.waves, 3)

    def test_wave1_inequality_is_reported_per_card_and_direction(self):
        """Spec section 5.4's six cases, each a row, none an aggregate.

        ``wave1_ok=6/6`` has to name its denominator: six is 3 cards x 2
        directions, and a 5/6 must be readable as one case rather than as a
        percentage.
        """
        from sglang.srt.weg2 import xchg_residency

        res = xchg_residency.solve(_cards(), _inline_census(_sb4_census()), 1229.0)
        rows = res.wave1_rows()
        self.assertEqual(len(rows), 6)
        self.assertEqual({r.wave for r in rows}, {1})
        self.assertEqual(
            sorted((r.uuid, r.direction) for r in rows),
            sorted((u, d) for u in _UUID for d in ("d2p", "p2d")),
        )
        # spec section 5.4's own six sums, in the spec's own order
        self.assertEqual(
            {(r.direction, r.nvml_index): r.resident_mib for r in rows},
            {
                ("p2d", 0): 23194, ("p2d", 1): 14305, ("p2d", 2): 15269,
                ("d2p", 0): 20542, ("d2p", 1): 16138, ("d2p", 2): 14466,
            },
        )

    def test_armed_line_carries_the_spec_tokens(self):
        """The acceptance line, in the spec's token order, with real numbers."""
        from sglang.srt.weg2 import launcher

        lines = []
        with _census_file(_sb4_census()) as path:
            res = launcher.prepare_weight_exchange(
                _cards(), lines.append, "exchange", path, "b17.4", 0,
            )
        self.assertIsNotNone(res)
        armed = [ln for ln in lines if ln.startswith("WEG2-XCHG-ARMED")]
        self.assertEqual(len(armed), 1)
        line = armed[0]
        for token in (
            "epoch=b17.4", "waves=3", "wave1_ok=6/6", "floor_mib=1229",
            "region_mib=385", "ring_H_mib=0",
        ):
            self.assertIn(token, line)
        self.assertIn("peak_mib=23060/16344/15664 (d2p) 25510/14305/15550 (p2d)", line)
        self.assertIn("free_mib=9547/4136/4816,7097/6175/4930", line)
        self.assertIn("provenance: boot weg2sb4", line)
        # and one audit row per (card, direction), each spelling the arrow out
        checks = [ln for ln in lines if ln.startswith("WEG2-XCHG-CHECK")]
        self.assertEqual(len(checks), 6)
        self.assertTrue(all("wakes" in ln for ln in checks))

    def test_armed_line_says_whether_any_rank_behaviour_is_wired(self):
        """An acceptance line on a boot that exchanges nothing must say so.

        ROUND-2 REFUTER F2.  Until S6 propagates ``--weg2-weight-source`` into
        the two groups' argv, the request structs and the saver's region flag,
        a boot launched with ``exchange`` runs the RING path end to end and
        still prints this line.  A later grep of the boot log -- the only thing
        anyone reads -- could not tell that boot from an exchanging one, which
        is the same defect the neighbouring docstring refuses in words ("an
        unarmed gate must never be read as a passed one").
        """
        from sglang.srt.weg2 import launcher, xchg_residency

        lines = []
        with _census_file(_sb4_census()) as path:
            launcher.prepare_weight_exchange(
                _cards(), lines.append, "exchange", path, "b.1", 0,
            )
        armed = [ln for ln in lines if ln.startswith("WEG2-XCHG-ARMED")][0]
        self.assertIn("wired=no", armed)
        self.assertIn("reason=S7-gate-only", armed)
        self.assertFalse(launcher.XCHG_RANK_BEHAVIOUR_WIRED)
        # and the token follows the FACT, not the format string: flip the fact
        # and the same builder prints the other value
        res = xchg_residency.solve(_cards(), _inline_census(_sb4_census()), 1229.0)
        self.assertIn(
            "wired=yes reason=none",
            xchg_residency.armed_line(
                res, "b.1", 1, 0, True, launcher.XCHG_UNWIRED_REASON
            ),
        )

    def test_launch_refuses_when_the_peak_exceeds_the_board_itself(self):
        """Spec section 5.3's own criterion: the peak does not FIT, floor aside.

        ROUND-2 REVIEW F4.  The spec-named refusal test drives the ARMING FLOOR
        branch (peak 20344 of 20480, free 136), not a peak that exceeds the
        card, so section 5.3's actual inequality and the ``wave1_fits=NO``
        branch went untested: with the floor removed the gate would still have
        had to refuse here, and nothing proved it did.
        """
        from sglang.srt.weg2 import launcher, xchg_residency

        census = _sb4_census()
        # 12000 MiB of never-paused resident bytes on card 1 puts BOTH of its
        # directions over the 20480 MiB board itself, wave 1 included.
        census["cards"][_UUID[1]]["tags"]["D"]["weights_resident_probe"] = 12000
        lines = []
        with _census_file(census) as path:
            with self.assertRaises(xchg_residency.Weg2XchgResidencyUnarmable) as ctx:
                launcher.prepare_weight_exchange(
                    _cards(), lines.append, "exchange", path, "b.0", 0,
                )
        msg = str(ctx.exception)
        # 16344 + 12000 = 28344 against a 20480 MiB board: NEGATIVE free
        self.assertIn("leaves -7864 MiB", msg)
        self.assertIn("dir=d2p", msg)
        self.assertIn("dir=p2d", msg)
        self.assertNotIn(_UUID[0], msg)
        self.assertNotIn(_UUID[2], msg)
        checks = [
            ln for ln in lines if ln.startswith("WEG2-XCHG-CHECK") and _UUID[1] in ln
        ]
        self.assertEqual(len(checks), 2)
        self.assertTrue(all("wave1_fits=NO" in ln for ln in checks), checks)
        self.assertTrue(all("free_mib=-" in ln for ln in checks), checks)
        # and the wave-1 census is 4/6, not an aggregate that hides which two
        res = xchg_residency.solve(_cards(), _inline_census(census), 1229.0)
        self.assertEqual(res.wave1_ok(), (4, 6))

    def test_duplicate_or_nameless_cards_are_refused(self):
        """Two boards under one key are priced as one board.

        ROUND-2 REFUTER F13.  Every per-card figure is keyed by UUID, so a
        collision silently drops a card from the peak table while the arming
        line still reads as complete.
        """
        from sglang.srt.weg2 import xchg_residency

        cards = _cards()
        cards[2].uuid = cards[1].uuid
        res = xchg_residency.solve(cards, _inline_census(_sb4_census()), 1229.0)
        self.assertFalse(res.armed)
        self.assertTrue(any("more than one live card" in r for r in res.refusals))

        cards = _cards()
        cards[0].uuid = ""
        res = xchg_residency.solve(cards, _inline_census(_sb4_census()), 1229.0)
        self.assertFalse(res.armed)
        self.assertTrue(any("carry no UUID" in r for r in res.refusals))

    def test_region_mib_is_arithmetic_over_stated_layout_terms(self):
        """385 is computed from four named inputs, never written down."""
        from sglang.srt.weg2 import launcher, xchg_residency

        # the pair count is DERIVED from this boot's card list (round-2 review
        # F8): it was typed as 6, which is n*(n-1) evaluated on this rig and
        # silently wrong on any other card count
        self.assertEqual(launcher.xchg_region_pairs(len(_cards())), 6)
        self.assertEqual(launcher.xchg_region_pairs(4), 12)
        self.assertEqual(launcher.xchg_region_pairs(1), 0)
        self.assertEqual(
            xchg_residency.region_mib_from_layout(
                launcher.xchg_region_pairs(len(_cards())),
                launcher.XCHG_REGION_SLOTS_PER_PAIR,
                launcher.XCHG_REGION_SLOT_MIB,
                launcher.XCHG_REGION_HEADER_MIB,
            ),
            385,
        )
        # no NUMERIC literal 385 anywhere in either module: the prose may name
        # the number it derives (and does), the code may not carry it
        for path in (
            os.path.join(ROOT, "python", "sglang", "srt", "weg2", "xchg_residency.py"),
            os.path.join(ROOT, "python", "sglang", "srt", "weg2", "launcher.py"),
        ):
            tree = ast.parse(open(path, encoding="utf-8").read())
            self.assertFalse(
                [
                    n
                    for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and n.value == 385
                    and not isinstance(n.value, (str, bool))
                ],
                f"the region size is typed as a literal in {path}, not derived",
            )

    # ---------------------------------------------------------------- refusals

    def test_default_ring_arm_arms_and_logs_nothing(self):
        """Backward compatibility: the default path is untouched and silent."""
        from sglang.srt.weg2 import launcher

        lines = []
        self.assertIsNone(
            launcher.prepare_weight_exchange(
                _cards(), lines.append, "ring", "", "b.0", 35771,
            )
        )
        self.assertEqual(lines, [])

    def test_absent_census_under_exchange_refuses_by_name(self):
        """No census, no peak, no boot -- never a guessed table."""
        from sglang.srt.weg2 import launcher, xchg_residency

        with self.assertRaises(xchg_residency.Weg2XchgResidencyUnarmable) as ctx:
            launcher.prepare_weight_exchange(
                _cards(), [].append, "exchange", "", "b.0", 0,
            )
        self.assertIn("--weg2-xchg-census", str(ctx.exception))
        with self.assertRaises(xchg_residency.Weg2XchgResidencyUnarmable):
            launcher.prepare_weight_exchange(
                _cards(), [].append, "exchange", "/nonexistent/census.json", "b.0", 0,
            )

    def test_a_wave_partition_that_is_not_a_partition_is_refused(self):
        """A repeated tag is charged twice; a tag with no bytes is not a zero."""
        from sglang.srt.weg2 import xchg_residency

        blob = _sb4_census()
        blob["waves"] = [["w1", "w2"], ["w2"], ["w3"]]
        res = xchg_residency.solve(_cards(), _inline_census(blob), 1229.0)
        self.assertTrue(any("repeats ['w2']" in r for r in res.refusals))
        self.assertFalse(res.armed)

        blob = _sb4_census()
        blob["waves"] = [["w1"], ["w2"], ["w3"], ["w_absent"]]
        res = xchg_residency.solve(_cards(), _inline_census(blob), 1229.0)
        self.assertTrue(any("w_absent" in r for r in res.refusals))
        self.assertFalse(res.armed)

    def test_missing_census_card_refuses_rather_than_guesses(self):
        """A card with no measured row is a missing measurement, not a default."""
        from sglang.srt.weg2 import xchg_residency

        blob = _sb4_census()
        del blob["cards"][_UUID[2]]
        res = xchg_residency.solve(_cards(), _inline_census(blob), 1229.0)
        self.assertFalse(res.armed)
        self.assertTrue(any(_UUID[2] in r and "not in the census" in r for r in res.refusals))

    def test_w55_is_the_only_code_this_slice_allocates(self):
        """One code, one exception name -- the #1263 discipline, checked here too."""
        src = open(
            os.path.join(ROOT, "python", "sglang", "srt", "weg2", "xchg_residency.py"),
            encoding="utf-8",
        ).read()
        codes = set(re.findall(r"\b(W\d{1,2}[a-z]?)\s+(Weg2[A-Za-z0-9_]+)", src))
        self.assertEqual(codes, {("W55", "Weg2XchgResidencyUnarmable")})


# ---------------------------------------------------------------------------
# FIXTURES.  The only measured numbers in this slice live here, with the boot
# that produced them named.  Production code carries none of them.
# ---------------------------------------------------------------------------

#: Synthetic card UUIDs; the census is joined to live cards BY UUID, and using
#: obviously-fake ones is what makes a positional join fail loudly.
_UUID = ("GPU-c0-5090", "GPU-c1-3080x4", "GPU-c2-3080x8")


class _Card:
    """The four attributes :func:`xchg_residency.solve` reads off a card."""

    def __init__(self, uuid, nvml_index, name, total_mib):
        self.uuid, self.nvml_index, self.name, self.total_mib = (
            uuid, nvml_index, name, total_mib,
        )


def _cards():
    """This rig's three cards with their NVML totals (spec section 5.1)."""
    return [
        _Card(_UUID[0], 0, "NVIDIA GeForce RTX 5090", 32607),
        _Card(_UUID[1], 1, "NVIDIA GeForce RTX 3080", 20480),
        _Card(_UUID[2], 2, "NVIDIA GeForce RTX 3080", 20480),
    ]


def _sb4_census():
    """Boot weg2sb4's per-card census, in the launch file's own shape.

    PROVENANCE, spec section 5.0: P totals 13860 / 7548 / 8504 MiB and D totals
    13914 / 9680 / 9370, of which the never-paused MTP/NEXTN draft is
    1382 / 1311 / 1311 (section 4.1); the per-rank non-tag term is the MEASURED
    dormant ``proc_used`` 1820 / 1368 / 1422, counted twice per card.  The
    per-WAVE splits are the differences between the successive residency rows
    of section 5.1 / 5.2, i.e. the same table read column-wise -- three tags
    stand in for the three waves' tag sets because the residency arithmetic
    only ever sums a wave.
    """
    return {
        "provenance": (
            "boot weg2sb4 WEG2-FLIP-TAG census (spec WEG2_REUSE_SPEC_0908 "
            "section 5.0); dormant proc_used from the same boot's dormant sample"
        ),
        "waves": [["w1"], ["w2"], ["w3"]],
        "cards": {
            _UUID[0]: {
                "dormant_proc_used_mib": 1820,
                "dormant_source": "weg2sb4 dormant proc_used, card 0",
                "tags": {
                    "P": {"w1": 2988, "w2": 2916, "w3": 7956},
                    "D": {"w1": 4312, "w2": 4042, "w3": 4178, "weights_draft": 1382},
                },
            },
            _UUID[1]: {
                "dormant_proc_used_mib": 1368,
                "dormant_source": "weg2sb4 dormant proc_used, card 1",
                "tags": {
                    "P": {"w1": 3722, "w2": 2916, "w3": 910},
                    "D": {"w1": 2710, "w2": 2554, "w3": 3105, "weights_draft": 1311},
                },
            },
            _UUID[2]: {
                "dormant_proc_used_mib": 1422,
                "dormant_source": "weg2sb4 dormant proc_used, card 2",
                "tags": {
                    "P": {"w1": 2252, "w2": 2916, "w3": 3336},
                    "D": {"w1": 2610, "w2": 2444, "w3": 3005, "weights_draft": 1311},
                },
            },
        },
    }


def _inline_census(blob):
    """The same loader the launcher uses, on an in-memory blob."""
    import tempfile

    from sglang.srt.weg2 import xchg_residency

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(blob, fh)
        path = fh.name
    try:
        return xchg_residency.load_census(path)
    finally:
        os.unlink(path)


class _census_file:
    """Write a census blob to a temp file for the duration of a ``with``."""

    def __init__(self, blob):
        self._blob = blob
        self._path = ""

    def __enter__(self):
        import tempfile

        fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(self._blob, fh)
        fh.close()
        self._path = fh.name
        return self._path

    def __exit__(self, *exc):
        if self._path and os.path.exists(self._path):
            os.unlink(self._path)
        return False


if __name__ == "__main__":
    unittest.main()
