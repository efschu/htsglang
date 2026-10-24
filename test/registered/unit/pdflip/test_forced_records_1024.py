"""#227 FORCED-RECORDS (24.10.): which forced boots still write measured records.

Befund 24.10.: alle 45 Rotations-Boots seit 23.10. 05:18Z laufen mit HTSGLANG_FORCE=1 und gingen an genau zwei
Ablehnungen vorbei -- HW-COUNT [METAL-UNPROVEN] (2-Karten-Formen) und HW-UNCALIBRATED der PP-CUT-Kreuzungsbreite
(Switch x16, Tabelle x4/x8). Mit "forced = keine Records" (PROFIL-EDITOR S1) schrieb seitdem kein Boot ein Sample:
der 27B-Sidecar endet 22.10. 02:04Z, int4ar bucht den D-Ruherest als Konstante 2652 MiB gegen gemessene 1258 MiB.

Gepinnt:
  * Klassifikator: nur HW-COUNT mit dem EINZIGEN Blocker METAL-UNPROVEN und die Kreuzungsbreiten-HW-UNCALIBRATED
    (Vollabgleich, kein ``||``-Anhang) sind record-neutral; Inventar-Fremdheit, geborgte Rueckstaende/Census,
    Kapazitaetsurteile, Entrypoint-Schwellen nicht.
  * Ein erzwungener Launcher HAELT, was er vor dem Siegel schreibt; das Siegel schreibt (WRITES, mit ``forced_past``)
    oder verwirft (NONE). Kinder schreiben nur mit dem Siegel in ihrer Umgebung.
  * Entrypoint: ``HTSGLANG_FORCE=1`` ohne die Liste ``FLLIPER_PDFLIP_FORCED_PAST_EP`` = unbekannt = keine Records.
  * GEFAHRENRICHTUNG (Mutant): ein Record aus fremder Form darf kein kleineres Budget erzeugen -- der D-Ruherest bleibt
    dann die Konstante, nie die fremde (kleinere) Messung.
  * Verdrahtung: das Siegel steht in ``main`` vor dem Front-Start, das Entrypoint exportiert die Liste.
GPU-frei, NVML-frei.
"""

import json
import os
import re
import subprocess
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import host_ledger
from flliper.srt.pdflip import launcher as L
from flliper.srt.pdflip import refusals as R
from flliper.srt.pdflip import topology as T

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..", "..")
LAUNCHER_SRC = os.path.join(ROOT, "python", "flliper", "srt", "pdflip", "launcher.py")
ENTRYPOINT_SRC = os.path.join(ROOT, "docker", "pdflip-release", "entrypoint.sh")

#: verbatim from boot_weg2_dkr27bint4arrc4bar1int4ar10241254_c07b574410_1024_125438.front.log (FORCED-PAST lines 15/53)
INT4AR_HW_COUNT = (
    "HW-COUNT: 2 cards would be P = TP1 x PP2, D = TP2 x PP1 (host ordinal 0); not yet runnable, 1 blocker(s): "
    "[METAL-UNPROVEN] no release boot on 2 cards yet (proven: [3]) (pdflip/topology.py PROVEN_CARD_COUNTS) || "
    "visible: nvml0 NVIDIA GeForce RTX 3080 20480 MiB sm86 800 GB/s PCIe gen4 x16 BAR1 256 MiB class=RTX3080; "
    "nvml1 NVIDIA GeForce RTX 3080 20480 MiB sm86 800 GB/s PCIe gen4 x16 BAR1 256 MiB class=RTX3080")
INT4AR_CROSSING = (
    "HW-UNCALIBRATED: PP-CUT crossing prices are measured at PCIe widths x4, x8 only; ordinal 0 (RTX3080) x16 -> x8; "
    "ordinal 1 (RTX3080) x16 -> x8 -- --force prices these at the nearest measured width not faster than their own: "
    "UNMEASURED on this board (a crossing measurement of this rig writes the table)")

ENV_KEYS = (R.ENV_FORCED_BOOT, R.ENV_FORCED_RECORDS, R.ENV_FORCED_PAST_EP) + tuple(R.ENV_FORCE_REQUEST_NAMES)


def card(i, name, mib, cc, uuid=None):
    return L.Card(i, uuid or f"GPU-{i:04d}", name, mib, reserved_mib=0, cc=cc)


def two_3080():
    return [card(0, "NVIDIA GeForce RTX 3080", 20480, (8, 6), "GPU-aaaa"),
            card(1, "NVIDIA GeForce RTX 3080", 20480, (8, 6), "GPU-bbbb")]


def d_sample(tag, mib_by_uuid, form="exchange"):
    """A group-D dormant-image sample in the shape the front writes (sample_dormant_image)."""
    return {"group": "D", "at": "2026-10-24T12:57:05Z", "boot_tag": tag, "commit": "c07b574410",
            "rss_shmem_gib": 3.4, "weight_tags_gib": 27.15, "vram_residue_mib": dict(mib_by_uuid),
            "vram_residue_form": form, "vram_residue_capture_bs": 6}


class Base(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in ENV_KEYS}
        R.arm(False)
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "weg2_measured_record.json")
        self.lines = []

    def tearDown(self):
        R.arm(False)
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v

    def samples(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path) as fh:
            return json.load(fh)["samples"]

    def force(self, *passes):
        R.arm(True)
        for code, text in passes:
            R.refuse_value(code, text, L.PdFlipLaunchRefused, self.lines.append)


class Classifier(Base):
    def test_the_two_int4ar_passes_are_record_neutral_verbatim(self):
        self.assertEqual(R.records_class("HW-COUNT", INT4AR_HW_COUNT), "HW-COUNT/METAL-UNPROVEN")
        self.assertEqual(R.records_class("HW-UNCALIBRATED", INT4AR_CROSSING), "HW-UNCALIBRATED/pp-crossing-width")

    def test_lane_borrow_text_from_the_launcher_itself_is_neutral(self):
        self.force()
        cards = L.order_cards(two_3080())
        self.assertEqual(L.forced_lane_borrow(cards, [16, 16], self.lines.append), [8, 8])
        (got,) = R.forced_list()
        self.assertEqual(R.records_class(got["code"], got["text"]), "HW-UNCALIBRATED/pp-crossing-width")

    def test_metal_only_built_by_topology_format_is_neutral_any_further_blocker_is_not(self):
        metal = T.Blocker("METAL-UNPROVEN", "pdflip/topology.py PROVEN_CARD_COUNTS", "no release boot on 2 cards yet (proven: [3])")
        nvec = T.Blocker("RECORDS-NVEC", "pdflip/profile_records_data/qwen27b.json",
                         "3 measured records are 3-vectors of one inventory that cannot be derived for this one (a, b); "
                         "2 cards of this inventory need a calibration boot that writes their records (plan K1/P3a)")
        head = "HW-COUNT: 2 cards would be P = TP1 x PP2, D = TP2 x PP1 (host ordinal 0); not yet runnable, "
        self.assertEqual(R.records_class("HW-COUNT", head + "1 blocker(s): " + metal.text()), "HW-COUNT/METAL-UNPROVEN")
        both = head + "2 blocker(s): " + "; ".join(b.text() for b in (nvec, metal))
        self.assertIsNone(R.records_class("HW-COUNT", both))
        lying = head + "1 blocker(s): " + "; ".join(b.text() for b in (metal, nvec))   # count says 1, list has 2
        self.assertIsNone(R.records_class("HW-COUNT", lying))

    def test_every_real_topology_refusal_is_judged_by_its_own_blocker_list(self):
        """Over every N the planner can refuse with HW-COUNT (generic context and a profile context with
        positional vectors), the verdict follows the refusal's OWN blocker objects: neutral iff all METAL-UNPROVEN."""
        seen = {True: 0, False: 0}
        ctxs = (T.TopologyContext(),
                T.TopologyContext(vectors={"p_budget": 3}))     # a 3-vector on N != 3 cards: PROFILE-VECTORS
        for ctx in ctxs:
            for n in range(T.MIN_CARDS, T.MAX_CARDS_BAR1 + 1):
                try:
                    T.plan_topology(n, ctx)
                except T.TopologyRefused as exc:
                    if not str(exc).startswith(T.CODE_COUNT):
                        continue
                    metal_only = all(b.code == "METAL-UNPROVEN" for b in exc.blockers)
                    got = R.records_class("HW-COUNT", str(exc) + " || visible: x")
                    self.assertEqual(got is not None, metal_only, str(exc)[:300])
                    seen[metal_only] += 1
        self.assertGreater(seen[True], 0)
        self.assertGreater(seen[False], 0)

    def test_count_mismatch_of_order_cards_is_a_foreign_configuration(self):
        self.force()
        L.order_cards(two_3080()[:1] + [card(2, "NVIDIA GeForce RTX 5090", 32607, (12, 0))], 3)
        (got,) = R.forced_list()
        self.assertEqual(got["code"], "HW-COUNT")
        self.assertIsNone(R.records_class(got["code"], got["text"]))

    def test_foreign_inventory_is_not_neutral(self):
        self.force()
        three_090 = [card(i, "NVIDIA GeForce RTX 3090", 24576, (8, 6)) for i in range(3)]
        ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--profile", "nextflash"])
        L.inventory_check_line(ns, L.order_cards(three_090))
        (got,) = R.forced_list()
        self.assertEqual(got["code"], "HW-UNCALIBRATED")
        self.assertIsNone(R.records_class(got["code"], got["text"]))

    def test_crossing_joined_with_another_message_or_altered_is_not_neutral(self):
        self.assertIsNone(R.records_class("HW-UNCALIBRATED", INT4AR_CROSSING + " || HW-UNCALIBRATED: profile records"))
        self.assertIsNone(R.records_class("HW-UNCALIBRATED", "HW-UNCALIBRATED: profile 'qwen27b' positional records"))
        self.assertIsNone(R.records_class("HW-COUNT", INT4AR_CROSSING))          # right text, wrong code
        self.assertIsNone(R.records_class("HW-UNCALIBRATED", INT4AR_HW_COUNT))

    def test_capacity_borrow_and_threshold_codes_are_never_neutral(self):
        for code in ("D-BUDGET", "P-CARD", "WAKE-CREDIT", "HW-BORROWED", "HOST-DISK", "SHM", "STORE", "MEMAVAIL",
                     "PROFIL-STATUS", "PP-CUT"):
            self.assertIsNone(R.records_class(code, INT4AR_CROSSING), code)
            self.assertIsNone(R.records_class(code, INT4AR_HW_COUNT), code)


class LauncherSeal(Base):
    def test_unforced_boot_writes_at_once_without_stamp(self):
        host_ledger.append_measured_record(self.path, {"group": "D", "x": 1})
        self.assertEqual(self.samples(), [{"group": "D", "x": 1}])
        self.assertEqual(host_ledger.flush_held_records(self.lines.append), (0, 0))

    def test_neutral_forced_boot_holds_then_writes_stamped(self):
        self.force(("HW-COUNT", INT4AR_HW_COUNT), ("HW-UNCALIBRATED", INT4AR_CROSSING))
        self.assertTrue(R.records_pending())
        host_ledger.append_measured_record(self.path, {"group": "P", "x": 1})
        self.assertEqual(self.samples(), [])                        # held: the verdict is not final yet
        allow, why = R.seal_records()
        self.assertTrue(allow, why)
        self.assertIn("FORCED-RECORDS WRITES", R.records_verdict_line())
        self.assertEqual(host_ledger.flush_held_records(self.lines.append), (1, 0))
        self.assertEqual(self.samples(), [{"group": "P", "x": 1,
                                           "forced_past": ["HW-COUNT/METAL-UNPROVEN", "HW-UNCALIBRATED/pp-crossing-width"]}])
        host_ledger.append_measured_record(self.path, {"group": "D", "x": 2})   # after the seal: direct
        self.assertEqual(len(self.samples()), 2)

    def test_forced_boot_that_passed_nothing_writes(self):
        self.force()
        self.assertEqual(R.seal_records()[0], True)
        host_ledger.append_measured_record(self.path, {"group": "D"})
        self.assertEqual(self.samples(), [{"group": "D", "forced_past": []}])

    def test_a_capacity_pass_keeps_the_boot_record_free_and_drops_the_held(self):
        self.force(("HW-COUNT", INT4AR_HW_COUNT), ("D-BUDGET", "KV does not fit by 1840 MiB"))
        host_ledger.append_measured_record(self.path, {"group": "P"})
        allow, why = R.seal_records()
        self.assertFalse(allow)
        self.assertIn("D-BUDGET", why)
        line = R.records_verdict_line()
        self.assertTrue(line.startswith("FORCED-RECORDS NONE: this boot writes NO records"), line)
        self.assertEqual(host_ledger.flush_held_records(self.lines.append), (0, 1))
        host_ledger.append_measured_record(self.path, {"group": "D"})
        self.assertEqual(self.samples(), [])

    def test_a_pass_after_the_seal_withdraws_the_verdict_in_this_process(self):
        self.force(("HW-COUNT", INT4AR_HW_COUNT))
        self.assertTrue(R.seal_records()[0])
        R.refuse_value("P-CARD", "chunk does not fit", L.PdFlipLaunchRefused, self.lines.append)
        self.assertFalse(R.records_allowed())
        self.assertIn("AFTER the records seal", R.records_verdict_line())
        host_ledger.append_measured_record(self.path, {"group": "D"})
        self.assertEqual(self.samples(), [])

    def test_arm_resets_seal_held_and_drops_an_inherited_verdict(self):
        os.environ[R.ENV_FORCED_RECORDS] = "allow;HW-COUNT/METAL-UNPROVEN"
        self.force(("D-BUDGET", "x"))
        self.assertNotIn(R.ENV_FORCED_RECORDS, os.environ)
        host_ledger.append_measured_record(self.path, {"group": "D"})
        R.arm(True)
        self.assertEqual(R.take_held(), [])
        self.assertFalse(R.records_allowed())


class Entrypoint(Base):
    def test_force_request_without_the_list_is_unknown_and_writes_nothing(self):
        for name in R.ENV_FORCE_REQUEST_NAMES:
            os.environ[name] = "1"
            self.force(("HW-COUNT", INT4AR_HW_COUNT))
            allow, why = R.seal_records()
            self.assertFalse(allow, name)
            self.assertIn(R.ENV_FORCED_PAST_EP, why)
            os.environ.pop(name)

    def test_empty_list_counts_as_nothing_passed(self):
        os.environ[R.ENV_FORCE_REQUEST] = "1"
        os.environ[R.ENV_FORCED_PAST_EP] = ""
        self.force(("HW-COUNT", INT4AR_HW_COUNT), ("HW-UNCALIBRATED", INT4AR_CROSSING))
        self.assertTrue(R.seal_records()[0])

    def test_an_entrypoint_threshold_pass_keeps_records_off(self):
        os.environ[R.ENV_FORCE_REQUEST] = "1"
        os.environ[R.ENV_FORCED_PAST_EP] = "SHM\t/dev/shm = 8 GiB < 48 GiB\n"
        self.force(("HW-COUNT", INT4AR_HW_COUNT))
        allow, why = R.seal_records()
        self.assertFalse(allow)
        self.assertIn("SHM (entrypoint)", why)
        self.assertEqual(R.entrypoint_forced(), [("SHM", "/dev/shm = 8 GiB < 48 GiB")])

    def test_no_entrypoint_at_all_is_a_bare_launcher_call(self):
        self.assertEqual(R.entrypoint_forced(), [])

    def test_entrypoint_source_lists_every_pass_and_exports_it(self):
        src = open(ENTRYPOINT_SRC, encoding="utf-8").read()
        rv = next(ln for ln in src.splitlines() if ln.startswith("refuse_value()"))
        self.assertIn('forced_past_ep "$2" "$3"', rv)
        self.assertIn('say "FORCED-PAST HW-COUNT $msg"; forced_past_ep HW-COUNT "$msg"', src)
        self.assertIn('say "FORCED-PAST HW-UNCALIBRATED $msg"; forced_past_ep HW-UNCALIBRATED "$msg"', src)
        block = src[src.index("LAUNCH+=(--force)"):]
        self.assertIn('export FLLIPER_PDFLIP_FORCED_PAST_EP="$FORCED_PAST_EP"', block[:400])
        # the register's pin on the entrypoint codes is untouched by the new helper
        self.assertEqual(sorted(R.ENTRYPOINT_FORCE_CODES), R.entrypoint_wired_codes(src))

    def test_the_shell_helper_builds_the_list_the_launcher_parses(self):
        fn = next(ln for ln in open(ENTRYPOINT_SRC, encoding="utf-8").read().splitlines()
                  if ln.startswith("forced_past_ep()"))
        script = ('FORCED_PAST_EP=""\n' + fn + '\nforced_past_ep HW-COUNT "HW-COUNT: a\nb"\n'
                  'forced_past_ep SHM "low"\nprintf "%s" "$FORCED_PAST_EP"\n')
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
        self.assertEqual(R.entrypoint_forced({R.ENV_FORCED_PAST_EP: out, R.ENV_FORCE_REQUEST: "1"}),
                         [("HW-COUNT", "HW-COUNT: a b"), ("SHM", "low")])


class ChildProcess(Base):
    def test_child_without_seal_writes_nothing(self):
        os.environ[R.ENV_FORCED_BOOT] = "1"
        host_ledger.append_measured_record(self.path, {"group": "D"})
        self.assertEqual(self.samples(), [])
        self.assertFalse(R.records_pending())                       # a child never holds

    def test_child_with_the_launcher_seal_writes_stamped(self):
        self.force(("HW-UNCALIBRATED", INT4AR_CROSSING))
        R.seal_records()
        fenv = {R.ENV_FORCED_RECORDS: "stale"}
        R.export_records_verdict(fenv)
        self.assertEqual(fenv[R.ENV_FORCED_RECORDS], "allow;HW-UNCALIBRATED/pp-crossing-width")
        R.arm(False)                                                # become the front: no arm, inherited env only
        os.environ[R.ENV_FORCED_BOOT] = "1"
        os.environ[R.ENV_FORCED_RECORDS] = fenv[R.ENV_FORCED_RECORDS]
        host_ledger.append_measured_record(self.path, {"group": "D"})
        self.assertEqual(self.samples(), [{"group": "D", "forced_past": ["HW-UNCALIBRATED/pp-crossing-width"]}])

    def test_none_verdict_removes_the_seal_from_the_child_env(self):
        self.force(("D-BUDGET", "x"))
        R.seal_records()
        fenv = {R.ENV_FORCED_RECORDS: "allow;"}
        R.export_records_verdict(fenv)
        self.assertNotIn(R.ENV_FORCED_RECORDS, fenv)


class DangerDirection(Base):
    """A record from a FOREIGN form must never price a smaller D residue (-> bigger P budget -> OOM)."""

    def test_foreign_inventory_boot_record_never_lowers_the_residue(self):
        three_090 = [card(i, "NVIDIA GeForce RTX 3090", 24576, (8, 6), f"GPU-{i:04d}") for i in range(3)]
        ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--profile", "nextflash"])
        self.force()
        L.inventory_check_line(ns, L.order_cards(three_090))        # FORCED-PAST HW-UNCALIBRATED (foreign inventory)
        R.seal_records()
        fenv = {}
        R.export_records_verdict(fenv)
        R.arm(False)                                                # the front of that boot
        os.environ[R.ENV_FORCED_BOOT] = "1"
        os.environ.update(fenv)
        host_ledger.append_measured_record(self.path, d_sample("foreign", {c.uuid: 900 for c in three_090}))
        self.assertEqual(self.samples(), [])
        os.environ.pop(R.ENV_FORCED_BOOT, None)                     # the next (unforced) boot reads the sidecar
        rec = host_ledger.read_measured_record(self.path).get("D")
        got, why = L.dc_residue_from_record(rec, three_090, "exchange")
        self.assertIsNone(got, why)                                 # -> the constant, never 900+256

    def test_neutral_boot_record_prices_the_next_boot_from_the_measurement(self):
        cards = two_3080()
        self.force(("HW-COUNT", INT4AR_HW_COUNT), ("HW-UNCALIBRATED", INT4AR_CROSSING))
        R.seal_records()
        fenv = {}
        R.export_records_verdict(fenv)
        R.arm(False)
        os.environ[R.ENV_FORCED_BOOT] = "1"
        os.environ.update(fenv)
        host_ledger.append_measured_record(self.path, d_sample("dkr27bint4ar", {c.uuid: 1258 for c in cards}))
        for k in (R.ENV_FORCED_BOOT, R.ENV_FORCED_RECORDS):
            os.environ.pop(k, None)
        rec = host_ledger.read_measured_record(self.path).get("D")
        got, why = L.dc_residue_from_record(rec, cards, "exchange")
        self.assertEqual(got, {c.uuid: 1258 + L.DC_RECORD_MARGIN_MIB for c in cards}, why)
        self.assertEqual(rec["forced_past"], ["HW-COUNT/METAL-UNPROVEN", "HW-UNCALIBRATED/pp-crossing-width"])


class LauncherWiring(unittest.TestCase):
    def test_seal_flush_and_export_precede_the_front_spawn_after_every_refusal_site(self):
        src = open(LAUNCHER_SRC, encoding="utf-8").read()
        main = src[src.index("\ndef main("):src.index("\ndef xchg_census_is_reserve(")]
        seal = main.index("refusals.seal_records()")
        spawn = main.index("fp = subprocess.Popen(front_argv")
        self.assertLess(seal, main.index("host_ledger.flush_held_records(log)"))
        self.assertLess(main.index("refusals.export_records_verdict(fenv)"), spawn)
        self.assertLess(main.index("fenv = dict(os.environ)"), seal)
        # every refusal site main reaches directly or through its solvers runs before the seal
        for call in ("refuse_value(", "log_d_rank_vram_solve(", "log_wake_credit_solve(", "p_card_verdict(",
                     "inventory_check_line(", "topology_check_line(", "forced_lane_borrow(", "dc_measured_d_mib("):
            for m in re.finditer(re.escape(call), main):
                self.assertLess(m.start(), seal, f"{call} at offset {m.start()} after the seal")

    def test_help_and_header_name_the_narrowed_rule(self):
        src = open(LAUNCHER_SRC, encoding="utf-8").read()
        self.assertIn("unless every refusal it passed is a proof gap of its own form (#227", src)
        self.assertIn("FORCED-RECORDS entrypoint list", src)


if __name__ == "__main__":
    unittest.main()
