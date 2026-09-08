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
    kw = dict(store_min_gib=store_min_gib, **RING_KW)
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


#: C19 (ring rebase 0908): the host weights term is the measured per-card ring
#: table (Sigma H / Sigma image_P), not a chunk count.  Same figures the s3s4
#: and ring_ledger suites pin.
DK5_RING_BYTES = 32964 * 1024 * 1024
DK5_RING_SPAN1_BYTES = 29912 * 1024 * 1024
RING_KW = dict(ring_bytes=DK5_RING_BYTES, ring_span1_bytes=DK5_RING_SPAN1_BYTES)

class TestTheDenominator(CustomTestCase):
    def test_base_is_bound_by_the_cgroup_on_the_weg2dk5_box_and_names_it(self):
        arm = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200,
            **RING_KW,
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
            **RING_KW,
            cg_current_bytes=0,
            cg_ceiling_bytes=DK5_MEMTOTAL_B * 4,
        )
        self.assertAlmostEqual(arm.terms["base_gib"], 103.56, delta=0.05)
        self.assertIn("meminfo", arm.terms["base_source"])

    def test_no_cgroup_sample_is_named_not_priced_as_zero(self):
        arm = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200, **RING_KW
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
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200, **RING_KW
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
    """RING REBASE 0908 -- three of this class's four tests are DELETED, not
    weakened, because the mechanism they asserted no longer exists.

    They pinned the run moment as ``one image + flip_transient``, with the
    ``image / weight_chunks`` endpoint as that term's floor:
      * test_the_run_moment_charges_the_measured_peak_not_the_endpoint
      * test_the_endpoint_arithmetic_is_the_floor_of_the_term_never_the_term
      * test_zero_chunks_is_still_the_two_image_dr1_shape
    C19's shared host ring removes the quantity all three measure: the region is
    preallocated at Sigma H and the legs copy THROUGH it, so ``price`` charges
    Sigma H ONCE and there is no transient, no chunk-in-flight and no
    ``weight_chunks`` knob left to be the floor of.  ``terms`` no longer carries
    ``backup_resident_gib`` / ``flip_transient_gib`` at all, so these could only
    have been kept by re-introducing the double-charge ring fix 1 finding 3
    removed.  The DR-1 two-image shape they also covered survives, priced from
    the ring table, in test_weg2_s3s4_refusals'
    ``test_w20_still_refuses_the_live_box_shape_under_the_DR1_two_image_shape``.

    What SURVIVES here is the one test that is about a MEASUREMENT rather than
    about that mechanism: FLIP_HOST_TRANSIENT_GIB is kept as the measured datum
    of the old form (the figure FLIPCOST A1-3 compares the ring against), and
    its provenance is still pinned below.
    """

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


class TestWeg2dk5WouldHaveBeenNamedBeforeItBooted(CustomTestCase):
    """RING REBASE 0908 -- the class's claim SURVIVES, re-derived, and it is
    worth stating why rather than editing four numbers quietly.

    weg2dk5 chose S=1 M=1200 with a 9 GiB store and was reaped nine minutes
    later.  Under fix 8 the ledger refused that boot OUTRIGHT: every arm was
    unfundable and every arm's predicted run peak sat above the reap point.
    On the ring neither of those is true any more, and the reason is the ring
    rather than a softened test: Sigma H (32.19 GiB) replaces the dormant image
    plus the flip transient (38.63 + 9.97 = 48.60), so every moment is 16.41 GiB
    cheaper -- far more than the 3.13 GiB by which dk5's arm was over the
    watermark.

    So the box is no longer refused outright.  But THE ARM THAT DIED IS STILL
    REFUSED, which is the sentence this class is named for: at M=1200 the LAUNCH
    moment is -0.42 GiB, so the ladder skips it (and M=2400 at -5.35) and hands
    out M=600 instead, whose predicted peak is 92.19 GiB against the 95.90 GiB
    watermark.  weg2dk5 would still have been named before it booted -- by the
    launch moment now instead of by a blanket refusal, and it would have been
    given a smaller, survivable arm rather than nothing.
    """

    #: The ladder at weg2dk5's own readings on the ring, from the printed lines.
    #: launch/run leftovers per arm; the first arm fundable at BOTH moments wins.
    DK5_LADDER_ON_THE_RING = {2400: (-5.35, 3.67), 1200: (-0.42, 8.60), 600: (2.05, 11.07)}

    def test_the_arm_that_died_is_still_refused_at_that_boots_own_store_floor(self):
        # THE RED-FIRST FACT, unchanged: on weg2dk5 this exact call FUNDED
        # S=1/M=1200 with a 9 GiB store and the box was reaped 9 minutes later.
        # On the ring the call still does not hand back that arm.
        arm, store, lines = _choose(DK5_STORE_MIN_GIB)
        self.assertEqual((arm.s_gb, arm.m_mib), (1, 600), "the ladder must fall past M=1200")
        self.assertEqual(store, 11.0)
        # and M=1200 -- the arm that died -- is refused by name, at the LAUNCH
        # moment, which is the term the ring does NOT make free (it charges
        # Sigma span1 = 29.21 GiB there plus the 12 GiB load transient).
        died = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200, **RING_KW,
            cg_current_bytes=DK5_CG_CURRENT_B, cg_ceiling_bytes=DK5_MEMTOTAL_B,
        )
        self.assertLess(died.launch_leftover_gib, 0.0)
        self.assertAlmostEqual(died.launch_leftover_gib, -0.42, delta=0.05)
        self.assertFalse(died.fundable_moments)
        ladder = [ln for ln in lines if "ARM S=1 M=1200" in ln][0]
        self.assertIn("refused", ladder)
        self.assertIn("launch moment", ladder)

    def test_the_ring_is_what_moved_that_verdict_and_by_exactly_how_much(self):
        # Fix 8 refused this box at every arm; the ring funds M=600.  The whole
        # difference is one term, and it is stated as arithmetic rather than as
        # a changed expectation: the run moment charged image + transient
        # (38.63 + 9.97 = 48.60 GiB) and now charges Sigma H (32.19) once.
        saving = (38.63 + 9.97) - 32.19
        self.assertAlmostEqual(saving, 16.41, delta=0.01)
        arm = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 600, **RING_KW,
            cg_current_bytes=DK5_CG_CURRENT_B, cg_ceiling_bytes=DK5_MEMTOTAL_B,
        )
        # fix 8's run leftover at this arm was 11.07 - 16.41 = -5.34, i.e. no arm.
        self.assertAlmostEqual(arm.run_leftover_gib, 11.07, delta=0.05)
        self.assertLess(arm.run_leftover_gib - saving, 0.0)
        self.assertAlmostEqual(arm.terms["host_ring_gib"], 32.19, delta=0.01)

    def test_the_whole_ladder_of_that_boot_is_now_below_the_observed_reap_point(self):
        # THE INVERTED VERDICT, re-derived rather than flipped.  Under fix 8
        # EVERY arm predicted a peak ABOVE the reap point; on the ring every arm
        # is BELOW it, because the ring took 16.41 GiB out of the run moment and
        # dk5's worst arm was only 3.13 GiB over.  The advisory still prints
        # beside the whole ladder, which is what made the verdict checkable.
        arm, store, lines = _choose(DK5_STORE_MIN_GIB)
        watermark = host_ledger.OBSERVED_REAP_CURRENT_BYTES / GIB
        self.assertAlmostEqual(watermark, 95.93, delta=0.02)
        for m, (_launch, run) in self.DK5_LADDER_ON_THE_RING.items():
            priced = host_ledger.price(
                DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, m, **RING_KW,
                cg_current_bytes=DK5_CG_CURRENT_B, cg_ceiling_bytes=DK5_MEMTOTAL_B,
            )
            self.assertAlmostEqual(priced.run_leftover_gib, run, delta=0.05)
            peak = priced.predicted_run_peak_gib(max(0.0, float(int(run))))
            self.assertLess(peak, watermark, f"M={m} predicts {peak:.2f}")
        self.assertIn("RUN-PEAK ADVISORY", "\n".join(lines))
        for m in (2400, 1200, 600):
            self.assertIn(f"M={m}", "\n".join(lines))

    def test_the_launch_moment_is_what_names_the_arm_that_actually_died(self):
        # weg2dk5's own choice: S=1 M=1200 with a 9 GiB store.  Under fix 8 the
        # RUN PEAK named it (3.13 GiB over the watermark, plus the measured image
        # and the run-moment origin).  On the ring the peak no longer does --
        # 92.66 GiB against 95.93 -- and that is correct, because the cost that
        # killed it is the one the ring removed.  The LAUNCH moment names it
        # instead, and the ledger is therefore still not handing out that arm.
        arm = host_ledger.price(
            DK5_MEMTOTAL_B, DK5_MEMAVAIL_B, 1, 1200, **RING_KW,
            cg_current_bytes=DK5_CG_CURRENT_B, cg_ceiling_bytes=DK5_MEMTOTAL_B,
        )
        watermark = host_ledger.OBSERVED_REAP_CURRENT_BYTES / GIB
        predicted = arm.predicted_run_peak_gib(DK5_STORE_CHOSEN_GIB)
        self.assertAlmostEqual(predicted, 92.66, delta=0.05)
        self.assertLess(predicted, watermark)
        # the margin the ring bought, at that arm and that store, is Sigma H's
        # saving against fix 8's image + transient: 92.66 + 16.41 = 109.07, which
        # is where fix 8 predicted this arm and why it refused.
        self.assertGreater(predicted + ((38.63 + 9.97) - 32.19), watermark)
        # and the arm is still refused -- by the launch moment.
        self.assertLess(arm.launch_leftover_gib, 0.0)
        self.assertFalse(arm.fundable_moments)

    def test_the_prediction_is_absent_not_green_without_a_cgroup_sample(self):
        arm, store, lines = _choose(4.0, with_cgroup=False)
        self.assertIsNone(arm.predicted_run_peak_gib(store))
        advisory = [ln for ln in lines if "RUN-PEAK ADVISORY" in ln][0]
        self.assertIn("not computed", advisory)

    def test_the_terms_line_carries_the_cgroup_reading_and_the_oom_baseline(self):
        try:
            _arm, _store, lines = _choose(4.0)
        except host_ledger.Weg2HostLedgerRefused as e:
            lines = str(e).splitlines()      # fix 8: the same lines, on refusal
        terms = [ln for ln in lines if "WEG2-HOST-LEDGER TERMS" in ln][0]
        for needle in (
            "memory.current=",
            "ceiling=",
            "oom_kill_baseline=18",
            "base_cgroup=",
            "bound by",
            # RING REBASE: the run moment's term is Sigma H, so the line names
            # that instead of the flip transient it no longer charges.
            "RUN MOMENT = the host weights term",
            "LAUNCH MOMENT = ring span 1",
            "image_P=38.63 GiB",
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
        # ``solve_p_cut`` JOINED THE SET on the merge train, and that is a
        # decision rather than a leak: merges 3/4 gave the launcher a P-cut
        # solver, whose unpinned INCUMBENT is by definition the shipped score
        # vector.  It arrived restating that vector as the bare literal
        # "32,18,14" -- the two-definitions defect this class exists to catch,
        # and the sibling assertion below did catch it -- so train fix 2 made
        # it read the constant instead.  The set stays EXACT: a fourth reader
        # still fails here.
        self.assertEqual(
            holders, {"argv_p", "p_stage_layers", "main", "solve_p_cut"}
        )

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
