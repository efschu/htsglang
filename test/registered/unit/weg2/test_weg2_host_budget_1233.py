# SPDX-License-Identifier: Apache-2.0
"""Weg 2 (#1233, fix 5): the HOST budget's DENOMINATOR and its MOMENT.

Boot weg2dk5 (2026-09-07, tip 0a28c32d36) was killed by the host cgroup, not by
a device OOM and not by anything with a traceback: ``oom_kill`` 60 -> 66 in
/proc/vmstat (18 -> 24 in the cgroup-scoped column of the launcher's own memts,
same delta +6) while the front was waking group D's ``weights_2``.  Everything
in the D log -- the TCPStore reset, the gloo ``Connection closed by peer`` --
is the cascade of processes that were SIGKILLed, and reading that traceback as
the root is the trap the boot sets.  The origin is the counter.

Two accounting defects, one class, and this file pins both:

1. THE DENOMINATOR.  ``WEG2-HOST-LEDGER`` priced every term from /proc/meminfo
   and refused nothing: the kill fired at MemAvailable 23.96 GB, well above the
   #721 16 GiB floor, so the quantity the ledger measured was never the
   quantity that governs.  The reaper watches ``memory.current`` against the
   cgroup ceiling.
2. THE MOMENT.  The run term was ``one image + image/N`` -- the interleave's
   ENDPOINT.  The interleave's PEAK holds both images partially resident plus
   the wake's anon working set; measured over that boot's ten flips it is
   9.97 GiB against the 3.60 GiB the endpoint charged.  Same "peak is not
   residency" class fix 4 closed on the DEVICE axis, on the HOST axis.

And one deletion: ``HOST_HEADROOM_GIB`` (#1232) was a constant fitted to the
gap between the #721 floor and the level at which this box actually OOMs, i.e.
a compensation layer for defect 1.  Upstream-minimal law makes that a deletion
candidate, not a repair order.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no server, no cgroup read (every cgroup
fact is passed in, and the one filesystem test writes its own fake cgroup).
"""

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger
from sglang.test.test_utils import CustomTestCase

GIB = host_ledger.GIB

# ------------------------------------------------------- MEASURED, boot weg2dk5
#: The launch moment the ledger actually priced, from the boot's own instruments:
#: front log 21:05:49Z ``WEG2-HOST-LEDGER TERMS memtotal=118.05 memavail=103.58``
#: and the memts row of the same second (/spinning/gpu-arb/memts_weg2_weg2dk5.csv).
DK5_MEMTOTAL_B = 126_751_866_880          # 118.05 GiB (lxcfs MemTotal)
DK5_MEMAVAIL_B = 111_196_077_056          # 103.56 GiB
DK5_CG_CURRENT_B = 22_719_148_032         # 21.16 GiB, memory.current before group P
DK5_CHUNKS = 8                            # --weight-chunks 8
#: What the ledger CHOSE on that boot, from the CHOSEN line: S=1 M=1200, store 9 GiB.
DK5_STORE_CHOSEN_GIB = 9
#: The floor that boot ran with (launcher --store-min-gib default).
DK5_STORE_MIN_GIB = 8.0


def _choose(store_min_gib, *, with_cgroup=True, **over):
    """One seam both trees answer through, so red/green is BEHAVIOURAL.

    The pre-fix ``choose`` has no cgroup parameters at all; calling it with them
    would raise TypeError and the failure would say nothing about the budget.
    This passes them only when the signature has them, so the pre-fix tree is
    exercised exactly as it ran on weg2dk5 -- and fails these assertions on the
    numbers, not on a signature.
    """
    kw = dict(store_min_gib=store_min_gib, weight_chunks=DK5_CHUNKS)
    params = inspect.signature(host_ledger.choose).parameters
    if with_cgroup and "cg_current_bytes" in params:
        kw.update(
            cg_current_bytes=DK5_CG_CURRENT_B,
            cg_ceiling_bytes=DK5_MEMTOTAL_B,
            cg_ceiling_source="test: lxcfs MemTotal fallback",
            cg_oom_kill=18,
        )
    kw.update(over)
    return host_ledger.choose(DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, **kw)


class TestTheDenominator(CustomTestCase):
    def test_base_is_bound_by_the_cgroup_on_the_weg2dk5_box_and_names_it(self):
        arm = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200,
            weight_chunks=DK5_CHUNKS,
            cg_current_bytes=DK5_CG_CURRENT_B,
            cg_ceiling_bytes=DK5_MEMTOTAL_B,
        )
        # ceiling - current - cli = 118.05 - 21.16 - 10 = 86.89, tighter than
        # meminfo's 103.56 -- and the ledger must SAY which one bound.
        self.assertAlmostEqual(arm.terms["base_cgroup_gib"], 86.89, delta=0.05)
        self.assertAlmostEqual(arm.terms["base_meminfo_gib"], 103.56, delta=0.05)
        self.assertAlmostEqual(arm.terms["base_gib"], 86.89, delta=0.05)
        self.assertIn("cgroup", arm.terms["base_source"])

    def test_a_roomy_cgroup_leaves_meminfo_binding_and_says_so(self):
        arm = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200,
            weight_chunks=DK5_CHUNKS,
            cg_current_bytes=0,
            cg_ceiling_bytes=DK5_MEMTOTAL_B * 4,
        )
        self.assertAlmostEqual(arm.terms["base_gib"], 103.56, delta=0.05)
        self.assertIn("meminfo", arm.terms["base_source"])

    def test_no_cgroup_sample_is_named_not_priced_as_zero(self):
        arm = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200, weight_chunks=DK5_CHUNKS
        )
        self.assertIsNone(arm.terms["base_cgroup_gib"])
        self.assertIsNone(arm.terms["cg_current_gib"])
        self.assertIn("no cgroup sample", arm.terms["base_source"])
        self.assertAlmostEqual(arm.terms["base_gib"], 103.56, delta=0.05)

    def test_the_1232_headroom_compensation_constant_is_gone(self):
        # Deleted, not shrunk: it was fitted to the gap this denominator now
        # measures.  A tree that still carries it is still compensating.
        self.assertFalse(hasattr(host_ledger, "HOST_HEADROOM_GIB"))
        arm = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200, weight_chunks=DK5_CHUNKS
        )
        self.assertNotIn("host_headroom_gib", arm.terms)

    def test_resolve_ceiling_prefers_memory_max_and_names_the_fallback(self):
        ceiling, src = host_ledger.resolve_cg_ceiling(
            {"max": 42 * 2**30, "current": 1, "peak": 2, "oom_kill": 0},
            DK5_MEMTOTAL_B,
        )
        self.assertEqual(ceiling, 42 * 2**30)
        self.assertIn("memory.max", src)
        ceiling, src = host_ledger.resolve_cg_ceiling(
            {"max": None, "current": 1, "peak": 2, "oom_kill": 0}, DK5_MEMTOTAL_B
        )
        self.assertEqual(ceiling, DK5_MEMTOTAL_B)
        self.assertIn("FALLBACK", src)

    def test_read_cgroup_reads_a_real_tree_and_reports_max_as_none(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            for name, text in (
                ("memory.current", "123\n"),
                ("memory.peak", "456\n"),
                ("memory.max", "max\n"),
                ("memory.events", "low 0\nhigh 0\nmax 0\noom 0\noom_kill 24\n"),
            ):
                with open(os.path.join(d, name), "w") as f:
                    f.write(text)
            cg = host_ledger.read_cgroup(d)
        self.assertEqual(cg["current"], 123)
        self.assertEqual(cg["peak"], 456)
        self.assertIsNone(cg["max"])          # "max" is an ABSENT ceiling, not 0
        self.assertEqual(cg["oom_kill"], 24)

    def test_read_cgroup_of_a_missing_tree_is_all_none_not_all_zero(self):
        cg = host_ledger.read_cgroup("/nonexistent-cgroup-root-1233")
        self.assertEqual(
            [cg["current"], cg["peak"], cg["max"], cg["oom_kill"]], [None] * 4
        )


class TestTheMoment(CustomTestCase):
    def test_the_run_moment_charges_the_measured_peak_not_the_endpoint(self):
        arm = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200, weight_chunks=DK5_CHUNKS
        )
        endpoint = arm.terms["backup_resident_gib"] / DK5_CHUNKS
        self.assertAlmostEqual(endpoint, 3.60, delta=0.05)      # what it charged
        self.assertAlmostEqual(
            arm.terms["flip_transient_gib"], host_ledger.FLIP_HOST_TRANSIENT_GIB, delta=1e-9
        )
        self.assertGreater(arm.terms["flip_transient_gib"], endpoint)

    def test_the_endpoint_arithmetic_is_the_floor_of_the_term_never_the_term(self):
        # One chunk = the whole image: coarse chunking must not price BELOW its
        # own arithmetic just because the measured population used N=8.
        arm = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200, weight_chunks=1
        )
        self.assertAlmostEqual(
            arm.terms["flip_transient_gib"], arm.terms["backup_resident_gib"], delta=1e-9
        )

    def test_the_measured_transient_is_the_max_of_its_named_population(self):
        # The ten interleaves of weg2dk5, cg_current at the interleave peak minus
        # the quiet minimum before that flip's begin line.  A PEAK term takes the
        # max; the mean would have priced this flip at 5.8 GiB and passed again.
        population = [0.03, 9.20, 3.39, 6.12, 6.83, 9.97, 4.32, 5.21, 3.81, 8.58]
        self.assertAlmostEqual(
            host_ledger.FLIP_HOST_TRANSIENT_GIB, max(population), delta=0.005
        )
        # ... and the peak does NOT sit in the tail: the transient is per-flip
        # FIXED rather than accumulating across flips, which is the discriminator
        # BOOT_weg2dk5_0907.md fix shape (3) asked the NEXT boot to spend itself
        # on.  It is settled here from the series already in hand.
        self.assertLess(max(population[6:]), max(population))
        self.assertEqual(population.index(max(population)), 5)

    def test_zero_chunks_is_still_the_two_image_dr1_shape(self):
        arm = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200, weight_chunks=0
        )
        self.assertAlmostEqual(
            arm.terms["flip_transient_gib"], arm.terms["backup_d_gib"], delta=1e-9
        )


class TestWeg2dk5WouldHaveBeenNamedBeforeItBooted(CustomTestCase):
    def test_the_arm_that_died_is_refused_at_that_boots_own_store_floor(self):
        # THE RED-FIRST FACT.  On weg2dk5 this exact call FUNDED S=1/M=1200 with
        # a 9 GiB store and the box was reaped 9 minutes later.
        with self.assertRaises(host_ledger.Weg2HostLedgerRefused) as ei:
            _choose(DK5_STORE_MIN_GIB)
        text = str(ei.exception)
        self.assertIn("W20 Weg2HostLedgerRefused", text)
        # the refusal is ACTIONABLE: it names both levers and rules one out
        self.assertIn("--store-min-gib", text)
        self.assertIn("shrinking the store tmpfs is NOT a lever", text)

    def test_it_funds_a_smaller_store_and_that_store_is_smaller_than_the_one_that_died(self):
        arm, store, lines = _choose(4.0)
        self.assertGreaterEqual(store, 4.0)
        self.assertLess(store, DK5_STORE_CHOSEN_GIB)
        self.assertTrue(any("RUN-PEAK ADVISORY" in ln for ln in lines))

    def test_the_advisory_puts_the_chosen_arm_below_the_observed_reap_point(self):
        arm, store, lines = _choose(4.0)
        predicted = arm.predicted_run_peak_gib(store)
        watermark = host_ledger.OBSERVED_REAP_CURRENT_BYTES / GIB
        self.assertAlmostEqual(watermark, 95.93, delta=0.02)
        self.assertLess(predicted, watermark)
        advisory = [ln for ln in lines if "RUN-PEAK ADVISORY" in ln][0]
        self.assertIn("below the OBSERVED REAP POINT", advisory)

    def test_the_advisory_would_have_flagged_the_arm_that_actually_died(self):
        # weg2dk5's own choice: S=1 M=1200 with a 9 GiB store.
        arm = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200,
            weight_chunks=DK5_CHUNKS,
            cg_current_bytes=DK5_CG_CURRENT_B,
            cg_ceiling_bytes=DK5_MEMTOTAL_B,
        )
        predicted = arm.predicted_run_peak_gib(DK5_STORE_CHOSEN_GIB)
        watermark = host_ledger.OBSERVED_REAP_CURRENT_BYTES / GIB
        self.assertGreater(predicted, watermark)
        # and the measured death sits between the prediction and the endpoint
        # model that let it through: 95.93 actual, ~99 predicted (conservative).
        self.assertLess(predicted - watermark, 5.0)

    def test_the_prediction_is_absent_not_green_without_a_cgroup_sample(self):
        arm, store, lines = _choose(4.0, with_cgroup=False)
        self.assertIsNone(arm.predicted_run_peak_gib(store))
        advisory = [ln for ln in lines if "RUN-PEAK ADVISORY" in ln][0]
        self.assertIn("not computed", advisory)

    def test_the_terms_line_carries_the_cgroup_reading_and_the_oom_baseline(self):
        _arm, _store, lines = _choose(4.0)
        terms = [ln for ln in lines if "WEG2-HOST-LEDGER TERMS" in ln][0]
        for needle in (
            "memory.current=",
            "ceiling=",
            "oom_kill_baseline=18",
            "base_cgroup=",
            "bound by",
            "flip_transient",
            "MEASURED over the 10 interleaves of boot weg2dk5",
        ):
            self.assertIn(needle, terms)
        self.assertNotIn("host_headroom=", terms)


class TestTheLauncherCallSites(CustomTestCase):
    """The two seams that live inside ``launcher.main`` and cannot be called.

    ``main`` resolves NVML cards, mounts a tmpfs and spawns two servers, so the
    call sites themselves are unreachable hermetically -- and they are exactly
    where both defects sat.  These read the call site out of the AST instead:
    a structural check matched to the error class (a reverted argument), not a
    generic import smoke that cannot fail on it.
    """

    @staticmethod
    def _launcher_ast():
        import ast

        from sglang.srt.weg2 import launcher as L

        with open(L.__file__) as f:
            return ast.parse(f.read()), L

    def _calls(self, tree, name):
        import ast

        return [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", getattr(n.func, "attr", None)) == name
        ]

    def test_the_flip_order_map_is_built_from_the_derived_split(self):
        import ast

        tree, _L = self._launcher_ast()
        calls = self._calls(tree, "chunk_tag_cards")
        self.assertEqual(len(calls), 1, "one map, one call site")
        first = calls[0].args[0]
        self.assertIsInstance(first, ast.Name)
        self.assertEqual(first.id, "p_split")   # not the score vector

    def test_the_score_vector_is_read_only_by_argv_and_by_the_derivation(self):
        import ast

        tree, _L = self._launcher_ast()
        holders = set()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for n in ast.walk(fn):
                if isinstance(n, ast.Name) and n.id == "P_PP_STAGE_RATIO_SCORES":
                    holders.add(fn.name)
        self.assertEqual(holders, {"argv_p", "p_stage_layers", "main"})

    def test_the_ledger_call_site_passes_the_cgroup_denominator(self):
        tree, _L = self._launcher_ast()
        calls = self._calls(tree, "choose")
        self.assertTrue(calls)
        kwargs = {k.arg for c in calls for k in c.keywords}
        for name in ("cg_current_bytes", "cg_ceiling_bytes", "cg_oom_kill"):
            self.assertIn(name, kwargs)

    def test_neither_score_vector_survives_anywhere_as_a_bare_literal(self):
        # m3: ``argv_p`` going back to "8,4,4" while the constant moves is the
        # exact two-definitions defect fix 5 closed, and comparing argv against
        # the constant cannot see it -- both read (8, 4, 4) today.  The source
        # must carry the vector ONCE, as the constant.
        import ast

        from sglang.srt.weg2 import launcher as L

        tree, _ = self._launcher_ast()
        forbidden = {
            ",".join(str(n) for n in L.P_PP_STAGE_RATIO_SCORES),
            ",".join(str(n) for n in L.P_PP_ATTN_STAGE_RATIO_SCORES),
        }
        literals = {
            n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        self.assertEqual(literals & forbidden, set())
