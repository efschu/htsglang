"""AP1 1006 (27B port of nf-next-1006-50, desk/27b-hwgen-ap1-1006): the fixed per-class gates of the launcher are VALUE refusals, not walls.

Order 05.10. 20:03/20:45Z and Q-710 ("das muss generisch sein bzw. per flags oder env gesetzt werden"; flags
instead of code, the existing way: ``--force`` + ``refusals.refuse_value``): a boot tree with a 5090 and two
arbitrary sm_86 cards (3090, 3080-10G, A6000 ...) must not die on

  (a) W19 ``dc_measured_d_mib``  -- refused every card of a class outside {RTX5090, RTX3080}, with ``--force`` too
                                    (it raised ``Weg2LaunchRefused`` directly, not through ``refuse_value``);
  (b) the P-cut attention anchor -- ``attn_anchor_stage`` refused every inventory without an RTX3080 class stage;
  (c) the awake-overshoot vector -- a record written for three cards died as a bare ``IndexError`` at N = 4;
  (d) the pinned cut 29,11,8     -- priced infeasible by the planner on 3 x 3090 (rank 0 +1897 MiB, 1520).

NOW (a)/(b): without ``--force`` the refusals read and raise exactly as before (same text, plus the named borrow
offer); with ``--force`` the boot BORROWS the figure of the calibrated class of the SAME ARCH (sm_86 -> RTX3080,
sm_120 -> RTX5090; the anchor stage of the reference rig), UNMEASURED, once per card as ``FORCED-PAST
HW-UNCALIBRATED``. sm_89 has no calibrated twin: it stays refused (not a target, order 03.10. ~20:45Z).
(c) is a named refusal instead of an IndexError (not needed at N = 3). (d) a FORCED boot on a foreign inventory whose
pinned cut the planner prices infeasible drops the pin and lets the cut solver decide (listed as FORCED-PAST); every
other case re-raises the planner's refusal.

THE REFERENCE RIG (3080, 5090, 3080 = N = 3) IS PLAN-IDENTICAL: no entry in ``refusals.forced_list`` under
``--force``, the same W19 figures, the same anchor stage, the same plan fingerprint armed and unarmed (the golden of
test_hw_generic_launcher_1002 pins the unarmed one against the base).

hw_sim step 5d (AP1 merge with the NF line, 06.10.): the pre-boot simulator asks the same two gates; the class
TestHwSimSeesTheGates is the NF file's, verbatim, so the AP1 code is the same in both lines (the 27B file keeps its
own register cases on top).

Honesty: every borrowed figure is UNMEASURED on the card it is applied to; a green test here is the Vorab-gate, not
a boot ("HOCHRECHNUNG != MESSUNG").

GPU-free, NVML-free.
"""

import os
import sys
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.planner import pp_cut_launch as PCL
from sglang.srt.weg2 import card_identity as CI
from sglang.srt.weg2 import form as F
from sglang.srt.weg2 import hw_sim as HS
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import refusals as R
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-weg2-unit")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def card(i, name, mib, cc, uuid=None):
    return L.Card(i, uuid or f"GPU-ap1-{i:02d}", name, mib, reserved_mib=0, cc=cc)


C3080 = lambda i: card(i, "NVIDIA GeForce RTX 3080", 20480, (8, 6))        # noqa: E731
C3080_10G = lambda i: card(i, "NVIDIA GeForce RTX 3080", 10240, (8, 6))    # noqa: E731
C3090 = lambda i: card(i, "NVIDIA GeForce RTX 3090", 24576, (8, 6))        # noqa: E731
C5090 = lambda i: card(i, "NVIDIA GeForce RTX 5090", 32607, (12, 0))       # noqa: E731
C5070TI = lambda i: card(i, "NVIDIA GeForce RTX 5070 Ti", 16303, (12, 0))  # noqa: E731
C4090 = lambda i: card(i, "NVIDIA GeForce RTX 4090", 24564, (8, 9))        # noqa: E731


def ordered(*cards):
    return L.order_cards(list(cards))


class _Armed(unittest.TestCase):
    """Each test starts disarmed and leaves the process disarmed (the switch is process-global)."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        R.arm(False)
        os.environ.pop(R.ENV_FORCED_BOOT, None)
        self.addCleanup(R.arm, False)

    def force(self):
        R.arm(True)

    def forced_texts(self):
        return [d["text"] for d in R.forced_list() if d["code"] == "HW-UNCALIBRATED"]


class TestCardIdentityTwin(unittest.TestCase):
    def test_arch_twin_is_the_calibrated_class_of_the_same_arch(self):
        self.assertEqual(CI.arch_twin_class(C3090(0)), "RTX3080")
        self.assertEqual(CI.arch_twin_class(C3080_10G(0)), "RTX3080")
        self.assertEqual(CI.arch_twin_class(C5070TI(0)), "RTX5090")
        self.assertEqual(CI.arch_twin_class(C5090(0)), "RTX5090")

    def test_no_twin_for_sm89_or_an_unreported_arch(self):
        self.assertIsNone(CI.arch_twin_class(C4090(0)))
        self.assertIsNone(CI.arch_twin_class(L.Card(0, "u", "NVIDIA GeForce RTX 3090", 24576)))

    def test_a_twin_is_not_a_class(self):
        """Borrowing names a figure's source; it never makes the card calibrated."""
        self.assertIsNone(CI.calibration_class(C3090(0)))
        self.assertEqual(CI.class_label(C3090(0)), "RTX3090/24576MiB/sm86")


class TestW19Residue(_Armed):
    FOREIGN = {"3090": C3090, "3080-10G": C3080_10G, "5070Ti": C5070TI}

    def test_without_force_every_foreign_card_refuses_as_before(self):
        for name, mk in self.FOREIGN.items():
            for ws in ("exchange", "serving"):
                with self.assertRaises(L.Weg2LaunchRefused, msg=name) as cm:
                    L.dc_measured_d_mib(mk(0), ws)
                self.assertIn("HW-UNCALIBRATED", str(cm.exception))
                self.assertIn(CI.card_key(mk(0)), str(cm.exception))
                self.assertIn("Measure it, do not borrow it.", str(cm.exception))
        self.assertEqual(R.forced_list(), [])

    def test_force_borrows_the_arch_twin_figure_and_says_so(self):
        self.force()
        for ws in ("exchange", "serving"):
            self.assertEqual(L.dc_measured_d_mib(C3090(0), ws), L.dc_measured_d_mib(C3080(0), ws))
            self.assertEqual(L.dc_measured_d_mib(C3080_10G(0), ws), L.dc_measured_d_mib(C3080(0), ws))
            self.assertEqual(L.dc_measured_d_mib(C5070TI(0), ws), L.dc_measured_d_mib(C5090(1), ws))
        texts = self.forced_texts()
        self.assertEqual(len(texts), 3, "one entry per foreign card, however often the selector is read")
        self.assertTrue(any("class RTX3080 (same arch sm86)" in t and "RTX3090/24576MiB/sm86" in t for t in texts))
        self.assertTrue(any("class RTX5090 (same arch sm120)" in t for t in texts))
        for t in texts:
            self.assertIn("UNMEASURED here", t)
            self.assertIn("nvml0", t)

    def test_sm89_has_no_twin_and_stays_refused_with_force(self):
        self.force()
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            L.dc_measured_d_mib(C4090(0), "exchange")
        self.assertIn("HW-UNCALIBRATED", str(cm.exception))
        self.assertEqual(R.forced_list(), [])

    def test_a_board_without_arch_stays_refused_with_force(self):
        self.force()
        with self.assertRaises(L.Weg2LaunchRefused):
            L.dc_measured_d_mib(L.Card(3, "GPU-cccc", "NVIDIA GeForce GTX 780", 3072), "exchange")

    def test_reference_classes_are_untouched_with_and_without_force(self):
        ref = ordered(C3080(0), C5090(1), C3080(2))
        plain = {ws: [L.dc_measured_d_mib(c, ws) for c in ref] for ws in ("exchange", "serving")}
        self.force()
        armed = {ws: [L.dc_measured_d_mib(c, ws) for c in ref] for ws in ("exchange", "serving")}
        self.assertEqual(plain, armed)
        self.assertEqual(R.forced_list(), [])
        self.assertEqual(armed["serving"], [L.DC_MEASURED_D_5090_MIB, L.DC_MEASURED_D_3080_MIB, L.DC_MEASURED_D_3080_MIB])


class TestAttnAnchor(_Armed):
    def test_without_force_a_class_less_inventory_refuses_as_before(self):
        for inv in (ordered(C3090(0), C3090(1), C3090(2)), ordered(C5090(0), C3090(1), C3090(2)),
                    ordered(C5090(0), C5090(1), C5090(2))):
            with self.assertRaises(L.Weg2LaunchRefused) as cm:
                L.attn_anchor_stage(inv)
            self.assertIn("HW-UNCALIBRATED", str(cm.exception))
        self.assertEqual(R.forced_list(), [])

    def test_force_places_the_anchor_on_the_stage_the_reference_rig_measured_it_on(self):
        self.force()
        ref_stage = CI.REFERENCE_INVENTORY.index(L.ATTN_ANCHOR_CARD_CLASS)
        self.assertEqual(ref_stage, 1)
        for inv in (ordered(C3090(0), C3090(1), C3090(2)), ordered(C5090(0), C3090(1), C3090(2)),
                    ordered(C5090(0), C5090(1), C5090(2)), ordered(C5090(0), C5070TI(1), C5070TI(2))):
            self.assertEqual(L.attn_anchor_stage(inv), ref_stage)
        texts = self.forced_texts()
        self.assertEqual(len(texts), 4)
        self.assertTrue(all("UNMEASURED on this card" in t and "BORROWS it onto P stage 1" in t for t in texts))

    def test_the_borrowed_stage_never_leaves_the_stage_list(self):
        self.force()
        self.assertEqual(L.attn_anchor_stage(ordered(C3090(0), C3090(1))), 1)
        self.assertEqual(L.attn_anchor_stage([C3090(0)]), 0)

    def test_reference_and_subsets_keep_their_measured_stage_and_log_nothing(self):
        self.force()
        self.assertEqual(L.attn_anchor_stage(ordered(C3080(0), C5090(1), C3080(2))), 1)
        self.assertEqual(L.attn_anchor_stage(ordered(C3080(0), C5090(1))), 1)
        self.assertEqual(L.attn_anchor_stage(ordered(C5090(0), C5090(1), C3080(2))), 2)
        self.assertEqual(R.forced_list(), [])

    def test_the_log_sees_the_borrow_when_a_logger_is_given(self):
        self.force()
        lines = []
        L.attn_anchor_stage(ordered(C3090(0), C3090(1), C3090(2)), lines.append)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("FORCED-PAST HW-UNCALIBRATED"))


class TestOvershootGuard(_Armed):
    def _budgets(self, cards, overshoot):
        lines = []
        return L.budgets_from_dc(cards, {c.uuid: 1500 for c in cards}, lines.append, "P",
                                 overshoot_mib=overshoot, overshoot_provenance="record of three cards")

    def test_a_three_vector_for_four_cards_is_a_named_refusal_not_an_index_error(self):
        four = ordered(C5090(0), C3090(1), C3090(2), C3080(3))
        with self.assertRaises(L.Weg2LaunchRefused) as cm:
            self._budgets(four, [100, 200, 300])
        self.assertIn("HW-UNCALIBRATED", str(cm.exception))
        self.assertIn("3 entries for 4 cards", str(cm.exception))

    def test_a_vector_of_the_cards_count_or_longer_is_read_as_before(self):
        three = ordered(C3080(0), C5090(1), C3080(2))
        self.assertEqual(len(self._budgets(three, [100, 200, 300])), 3)
        two = ordered(C5090(0), C3080(1))
        self.assertEqual(self._budgets(two, [100, 200, 300]), self._budgets(two, [100, 200]))

    def test_no_vector_is_still_no_overshoot(self):
        three = ordered(C3080(0), C5090(1), C3080(2))
        self.assertEqual(len(self._budgets(three, None)), 3)


class TestPinnedCutFallback(_Armed):
    """(d): the pin the profile carries (nf.env ``--pp-stage-ratio 29,11,8 --pp-attn-stage-ratio 7,3,2``) is dropped only
    when the PLANNER prices it infeasible on a FOREIGN inventory in a FORCED boot; the solver then derives the cut."""

    def ns(self, *extra):
        m = HS.MODELS["NF"]
        return L.build_parser().parse_args(
            ["--tree", "/sim", "--tag", "sim", "--profile", m.profile,
             "--model", F.profile_row(m.profile).formats[m.weight_format].checkpoint, *m.argv, *extra])

    @staticmethod
    def solver(calls, fail_unpinned=False):
        def fake(ns, cards, budgets_p, model, log, **kw):
            calls.append((ns.pp_stage_ratio, ns.pp_attn_stage_ratio))
            if ns.pp_stage_ratio or fail_unpinned:
                raise PCL.PPCutRefused("W40 Weg2PPCutRefused: the pinned layer cut 29,11,8 cannot be priced")
            return "SOLVED"
        return fake

    def run_wrapped(self, ns, cards, calls, log=None, **kw):
        return L._unpin_foreign_cut(self.solver(calls, **kw))(ns, cards, [1, 2, 3], "m", log or (lambda s: None))

    def test_forced_foreign_inventory_drops_the_pin_and_lets_the_solver_decide(self):
        self.force()
        ns, calls, lines = self.ns(), [], []
        self.assertEqual(ns.pp_stage_ratio, "29,11,8")
        got = self.run_wrapped(ns, ordered(C3090(0), C3090(1), C3090(2)), calls, lines.append)
        self.assertEqual(got, "SOLVED")
        self.assertEqual(calls, [("29,11,8", "7,3,2"), (None, None)])
        self.assertEqual((ns.pp_stage_ratio, ns.pp_attn_stage_ratio), (None, None))
        (text,) = self.forced_texts()
        self.assertIn("--pp-stage-ratio 29,11,8", text)
        self.assertIn("DROPPED", text)
        self.assertIn("[RTX5090, RTX3080, RTX3080]", text)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("FORCED-PAST HW-UNCALIBRATED"))

    def test_without_force_the_planner_refusal_propagates_and_the_pin_stays(self):
        ns, calls = self.ns(), []
        with self.assertRaises(PCL.PPCutRefused):
            self.run_wrapped(ns, ordered(C3090(0), C3090(1), C3090(2)), calls)
        self.assertEqual(calls, [("29,11,8", "7,3,2")])
        self.assertEqual(ns.pp_stage_ratio, "29,11,8")
        self.assertEqual(R.forced_list(), [])

    def test_the_calibrated_inventory_keeps_its_pin_even_in_a_forced_boot(self):
        self.force()
        ns, calls = self.ns(), []
        with self.assertRaises(PCL.PPCutRefused):
            self.run_wrapped(ns, ordered(C3080(0), C5090(1), C3080(2)), calls)
        self.assertEqual(calls, [("29,11,8", "7,3,2")])
        self.assertEqual(R.forced_list(), [])

    def test_an_inventory_the_operator_named_keeps_its_pin(self):
        self.force()
        cards = ordered(C3090(0), C3090(1), C3090(2))
        ns, calls = self.ns("--profile-inventory", ",".join(CI.inventory_signature(cards))), []
        with self.assertRaises(PCL.PPCutRefused):
            self.run_wrapped(ns, cards, calls)
        self.assertEqual(calls, [("29,11,8", "7,3,2")])

    def test_no_pin_no_retry(self):
        self.force()
        ns, calls = self.ns(), []
        ns.pp_stage_ratio = ns.pp_attn_stage_ratio = None
        with self.assertRaises(PCL.PPCutRefused):
            self.run_wrapped(ns, ordered(C3090(0), C3090(1), C3090(2)), calls, fail_unpinned=True)
        self.assertEqual(len(calls), 1)

    def test_when_the_solver_finds_no_cut_either_its_refusal_is_the_answer(self):
        self.force()
        ns, calls = self.ns(), []
        with self.assertRaises(PCL.PPCutRefused):
            self.run_wrapped(ns, ordered(C3090(0), C3090(1), C3090(2)), calls, fail_unpinned=True)
        self.assertEqual(len(calls), 2)

    def test_other_errors_are_not_swallowed(self):
        self.force()

        def boom(ns, cards, budgets_p, model, log, **kw):
            raise L.Weg2LaunchRefused("W99 something else")
        with self.assertRaises(L.Weg2LaunchRefused):
            L._unpin_foreign_cut(boom)(self.ns(), ordered(C3090(0), C3090(1), C3090(2)), [1], "m", print)

    def test_drop_cut_pins_also_clears_extra_words(self):
        ns = self.ns("--extra-p=--pp-stage-ratio 1,2,3 --pp-attn-stage-ratio 1,1,1 --keep me")
        dropped = L._drop_cut_pins(ns)
        self.assertEqual(len(dropped), 4)
        self.assertEqual(ns.extra_p.split(), ["--keep", "me"])
        self.assertIsNone(ns.pp_stage_ratio)

    def test_solve_p_cut_is_the_wrapped_solver_and_keeps_its_source(self):
        import inspect
        self.assertTrue(hasattr(L.solve_p_cut, "__wrapped__"))
        self.assertIn("def solve_p_cut(", inspect.getsource(L.solve_p_cut))
        self.assertIn("cut = solve_p_cut(", inspect.getsource(L.main))


class TestHwSimSeesTheGates(_Armed):
    def cell(self, keys, model="NF", force=False):
        if force:
            R.arm(True)
        try:
            c = HS.simulate("t", keys, HS.MODELS[model])
            return c, R.forced_list()
        finally:
            R.arm(False)
            os.environ.pop(R.ENV_FORCED_BOOT, None)

    def test_a_5090_and_two_sm86_cards_run_under_force_with_the_borrows_listed(self):
        for inv in (["3090", "5090", "3090"], ["5090", "3080-10G", "3080-10G"], ["5090", "A6000", "A6000"]):
            c, forced = self.cell(inv, force=True)
            self.assertEqual((c.result, c.blockers), (HS.RUNS, []), inv)
            texts = [f["text"] for f in forced]
            self.assertTrue(any("W19 dormant-residue" in t and "BORROWS" in t for t in texts), inv)
            self.assertTrue(any("deep attention anchor" in t and "BORROWS" in t for t in texts), inv)

    def test_without_force_the_cell_is_the_same_uncalibrated_refusal_as_before(self):
        c, forced = self.cell(["3090", "5090", "3090"])
        self.assertEqual((c.result, c.code, c.blockers), (HS.REFUSED, "HW-UNCALIBRATED", ["UNCALIBRATED"]))
        self.assertEqual(forced, [])

    def test_sm89_is_still_refused_under_force_and_by_the_new_gate(self):
        c, _ = self.cell(["5090", "4090", "3090"], force=True)
        self.assertEqual(c.result, HS.REFUSED)
        self.assertIn("W19-RESIDUE", c.blockers)

    def test_the_reference_cell_is_green_and_forces_nothing_in_both_models(self):
        for model in ("NF", "27B-INT8"):
            c, forced = self.cell(list(HS.REFERENCE_RIG), model=model, force=True)
            self.assertEqual((c.result, c.blockers), (HS.RUNS, []), model)
            self.assertEqual(forced, [], model)

    def test_another_stage_count_does_not_price_the_anchor(self):
        """Two stages derive the family cost from the reference basis (AP3): the anchor is not asked there."""
        c, forced = self.cell(["5090", "3090"], force=True)
        self.assertNotIn("ATTN-ANCHOR", c.blockers)
        self.assertFalse(any("deep attention anchor" in f["text"] for f in forced))


class TestRegisterShape27B(_Armed):
    """The 27B launcher carries its OWN register truth (``ENTRYPOINT_FORCE_CODES`` / ``entrypoint_wired_codes`` /
    ``public_register`` with ``wired_entrypoint`` / ``wired_at`` / ``force_scope``). The port must not move it: the
    value refusals W19 / anchor / pin go through ``refuse_value("HW-UNCALIBRATED")``, which the register already
    lists as wired (inventory_check_line); only the ``source`` text of that one entry names the new sites."""

    def entry(self, code="HW-UNCALIBRATED"):
        reg = R.public_register(R.wired_codes(open(L.__file__).read()))
        return {e["code"]: e for e in reg}[code]

    def test_hw_uncalibrated_stays_wired_in_launcher_and_entrypoint(self):
        e = self.entry()
        self.assertTrue(e["forcebar"])
        self.assertTrue(e["wired"])
        self.assertTrue(e["wired_entrypoint"])
        self.assertEqual(e["wired_at"], "launcher+entrypoint")
        self.assertEqual(e["force_scope"], "forceable in the Docker start (entrypoint) and in the launcher")

    def test_the_entrypoint_force_codes_are_unchanged(self):
        self.assertEqual(R.ENTRYPOINT_FORCE_CODES,
                         ("HW-COUNT", "HW-UNCALIBRATED", "MEMAVAIL", "PROFIL-STATUS", "SHM", "STORE"))

    def test_the_register_names_the_three_new_sites(self):
        src = self.entry()["source"]
        for site in ("launcher.dc_measured_d_mib", "launcher.attn_anchor_stage", "launcher._unpin_foreign_cut"):
            self.assertIn(site, src)

    def test_sm89_is_not_forcebar_anywhere_the_class_is_still_hard_for_other_codes(self):
        """No code became forcebar or hard by this port: the register's class column is what the 27B had."""
        classes = {r.code: r.klass for r in R.REGISTER}
        self.assertEqual(classes["HW-UNCALIBRATED"], R.CLASS_VALUE)
        self.assertEqual(classes["HW-TOPOLOGY"], R.CLASS_HARD)

    def test_main_flushes_the_w19_borrow_into_the_log_right_after_the_dc_residue_line(self):
        """``dc_measured_d_mib`` has no logger: the FORCED-PAST lines of the W19 twin borrow reach the boot log only through
        ``refusals.flush(log)`` in main (mutant: flush removed -> the borrow is in forced_list but invisible in the log)."""
        import inspect
        src = inspect.getsource(L.main)
        i = src.index("reserve incl. {slack_mib} MiB slack")
        self.assertIn("refusals.flush(log)", src[i:i + 600])

    def test_no_new_flag_or_env_in_the_port(self):
        """Q-710: flags instead of code, no new configuration. The new helpers add no argparse option and no env name."""
        import inspect
        for fn in (L.dc_measured_d_mib, L.attn_anchor_stage, L._refuse_value_once, L._drop_cut_pins,
                   L._unpin_foreign_cut, CI.arch_twin_class):
            src = inspect.getsource(fn)
            self.assertNotIn("add_argument", src, fn.__name__)
            self.assertNotIn("os.environ", src, fn.__name__)
            self.assertNotIn("getenv", src, fn.__name__)


class TestReferencePlanIdentical(_Armed):
    """The reference rig plan is the same armed and unarmed: the golden of test_hw_generic_launcher_1002 pins the
    unarmed fingerprint against the base, this pins armed == unarmed (nothing a forced boot would borrow)."""

    def test_plan_fingerprint_armed_equals_unarmed(self):
        import json
        import hw_generic_rig_plan_fingerprint_1002 as FP
        plain = json.dumps(FP.fingerprint(), sort_keys=True, default=str)
        self.force()
        armed = json.dumps(FP.fingerprint(), sort_keys=True, default=str)
        self.assertEqual(plain, armed)
        self.assertEqual(R.forced_list(), [])

    def test_the_solve_wrapper_is_inert_on_the_reference(self):
        """The reference inventory returns the solver's value untouched, and a refusal untouched."""
        cards = ordered(C3080(0), C5090(1), C3080(2))
        ok = L._unpin_foreign_cut(lambda ns, c, b, m, log, **kw: "OK")
        self.assertEqual(ok(object(), cards, [1], "m", print), "OK")


if __name__ == "__main__":
    unittest.main()
