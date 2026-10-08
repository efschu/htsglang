"""Auftrag 1006: "Hardwareprofil messen" liefert die Werte, die das Dashboard zeigt.

Ende-zu-Ende ohne Karte: das ECHTE ``hardware_profile.py`` des Baums (per Dateipfad geladen, wie der Dienst es tut) baut aus
einem synthetischen Probe-Cache (so, wie der Messlauf ihn schreibt) das Profil, das echte ``hwprofil.js`` zeichnet es (node),
und die Zeilen SM-Zahl, L2-Größe, int8 W8A8, NVFP4 W4A8/W4A16/W4A4, H2D-/D2H-Latenz und die BAR1-Matrix stehen mit Zahl und
Quellenmarke 'gem.' da.  Wo eine Karte eine Zeile nicht kann (W4A4 auf den 3080, W4A8 auf der 5090), steht "nicht gemessen"
mit dem Grund, nie eine Zahl.  Rot auf der Basis: dort bleiben W4A4 und BAR1 "nicht gemessen", weil die Basis weder die Zeile
noch den Schritt kennt.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import hwprofil, server  # noqa: E402

STATIC = server.STATIC
REPO_PYTHON = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "python"))
U = {"3080a": "GPU-0000", "5090": "GPU-1111", "3080b": "GPU-2222"}
NOW = 1_790_000_000.0


def _node():
    for c in (shutil.which("node"), "/opt/node-v22.14.0-linux-x64/bin/node", shutil.which("bun")):
        if c and os.path.exists(c):
            return c
    return None


def _nvml():
    spec = [(0, U["3080a"], "NVIDIA GeForce RTX 3080", 20480, [8, 6]), (1, U["5090"], "NVIDIA GeForce RTX 5090", 32607, [12, 0]),
            (2, U["3080b"], "NVIDIA GeForce RTX 3080", 20480, [8, 6])]
    cards = [{"nvml_index": i, "uuid": u, "name": n, "total_mib": mib, "cc": cc, "bar1_total_mib": 256 if cc == [8, 6] else 32768,
              "pcie_max_gen": 4, "pcie_max_width": 16, "pcie_cur_gen": 1, "pcie_cur_width": 8, "mem_bus_width_bits": 320,
              "mem_clock_max_mhz": 9501, "sm_clock_max_mhz": 2100, "power_limit_w": 230.0, "power_default_w": 320.0,
              "pci_bus_id": "0000:0%d:00.0" % i} for i, u, n, mib, cc in spec]
    return cards, "595.58", []


def _card(uuid, name, cc, **kw):
    d = {"uuid": uuid, "name": name, "cuda_index": 0, "total_mib": 20480, "gemm_bf16_tflops": 60.0, "gemm_fp8_tflops": None,
         "fp8_note": "no fp8", "membw_read_gbs": 700.0, "membw_copy_gbs": 690.0, "membw_gemv_gbs": 650.0, "h2d_gbs": 6.5,
         "d2h_gbs": 6.5, "h2d_lat_us": 12.5, "d2h_lat_us": 14.0, "h2d_lat_min_us": 9.0, "d2h_lat_min_us": 10.0, "sm_count": 68,
         "l2_mib": 5.0, "compute_capability": cc, "gemm_int8_tflops": 180.0, "gemm_w4a8_int8_tflops": 62.0,
         "gemm_w4a16_tflops": 55.0, "lane_notes": {}, "sm_clock_mhz": 1900, "sm_clock_max_mhz": 2100, "temp_c": 60.0,
         "throttle_reasons": [], "pcie_gen_cur": 4, "pcie_width_cur": 4, "pcie_gen_max": 4, "pcie_width_max": 16}
    d.update(kw)
    return d


def _probe(bar1=True, nccl=False):
    no_fp4 = {"nvfp4_w4a4": "compute capability 8.6: no native FP4 tensor cores (needs 10.0+)"}
    cards = [_card(U["3080a"], "RTX 3080", "8.6", lane_notes=no_fp4), _card(U["3080b"], "RTX 3080", "8.6", lane_notes=no_fp4),
             _card(U["5090"], "RTX 5090", "12.0", sm_count=170, l2_mib=96.0, gemm_fp8_tflops=500.0, fp8_note="",
                   gemm_w4a8_int8_tflops=None, gemm_w4a4_tflops=910.0,
                   lane_notes={"nvfp4_w4a8": "compute capability 12.0: the W4A8 kernel is the sm_8x one"})]
    ids = list(U.values())
    ordered = [(a, b) for a in ids for b in ids if a != b]
    stage = [{"src_uuid": a, "dst_uuid": b, "bandwidth_gbs": 12.0 + i, "bandwidth_serial_gbs": 6.0 + i, "latency_us": 21.0 + i,
              "transport": "host staging (pinned)", "peer_access": False, "note": "staged"} for i, (a, b) in enumerate(ordered)]
    d = {"version": 1, "created": NOW - 60, "driver": "595.58", "torch_version": "2.11", "cuda_version": "13.0", "cards": cards, "pairs": stage}
    if nccl:
        d.update(nccl_attempted=True, nccl_reason="", nccl_pairs=[
            {"src_uuid": a, "dst_uuid": b, "bandwidth_gbs": 9.0 + i, "latency_us": 33.0 + i, "transport": "nccl send/recv (SHM/direct/direct)",
             "peer_access": False, "note": "NCCL chose: SHM/direct/direct"} for i, (a, b) in enumerate(ordered)])
    if bar1:
        ids = list(U.values())
        d.update(bar1_attempted=True, bar1_reason="", bar1_pairs=[
            {"src_uuid": a, "dst_uuid": b, "bandwidth_gbs": 4.0 + 0.5 * i, "latency_us": 8.0 + i, "transport": "bar1 (direct write into the destination's BAR1)",
             "peer_access": True, "note": "n"} for i, (a, b) in enumerate((a, b) for a in ids for b in ids if a != b)])
    return d


class TestMeasuredValuesReachTheRows(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)

    def _doc(self, bar1=True, nccl=False):
        with open(os.path.join(self.d, "card_probe-x.json"), "w") as f:
            json.dump(_probe(bar1, nccl), f)
        mod = hwprofil._load_module(REPO_PYTHON)
        return mod.build(cache_dir=self.d, nvml=_nvml(), now=NOW), mod

    def _render(self, doc):
        node = _node()
        if not node:
            self.skipTest("weder node noch bun vorhanden")
        script = "const H=require(process.argv[1]);process.stdout.write(H.render(JSON.parse(process.argv[2]),{now:%s}));" % int(NOW)
        out = subprocess.run([node, "-e", script, os.path.join(STATIC, "hwprofil.js"),
                              json.dumps({"ok": True, "profile": doc, "problems": [], "window": None, "job": {"state": "idle"}})],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout

    @staticmethod
    def _row(html, label):
        m = re.search(r"<tr><td>" + re.escape(label) + r"</td>(.*?)</tr>", html, re.S)
        assert m, "Row missing: " + label
        return m.group(1).split("</td>")[:-1]

    def test_every_asked_row_is_filled_with_a_number_and_the_gem_mark(self):
        doc, mod = self._doc()
        self.assertEqual(mod.validate(doc), [])
        h = self._render(doc)
        for label in ("SM count", "L2 size", "int8 W8A8", "NVFP4 W4A16 (Marlin)", "H2D latency", "D2H latency"):
            cells = self._row(h, label)
            self.assertEqual(len(cells), 3, label)
            for c in cells:
                self.assertNotIn("not measured", c, label)
                self.assertIn("<sup>meas.</sup>", c, label)

    def test_w4a4_is_filled_only_on_the_sm120_card_and_the_3080s_say_why(self):
        doc, _ = self._doc()
        h = self._render(doc)
        cells = self._row(h, "NVFP4 W4A4 (native)")
        by_ord = {c["ord"]: c["uuid"] for c in doc["cards"]}
        for ordinal, cell in enumerate(cells):
            if by_ord[ordinal] == U["5090"]:
                self.assertIn("910", cell)
                self.assertIn("<sup>meas.</sup>", cell)
            else:
                self.assertIn("not measured", cell)
                self.assertIn("no native FP4 tensor cores", cell)       # der Grund im Hover
                self.assertNotIn("<sup>meas.</sup>", cell)
        w4a8 = self._row(h, "NVFP4 W4A8 (int8 cores)")                   # auf der 5090 nicht gefragt, mit Grund
        for ordinal, cell in enumerate(w4a8):
            if by_ord[ordinal] == U["5090"]:
                self.assertIn("not measured", cell)
                self.assertIn("sm_8x", cell)
            else:
                self.assertIn("<sup>meas.</sup>", cell)

    @staticmethod
    def _d2d_rows(h):
        sec = h[h.index("Card to card (D2D) per ordered pair"):]
        sec = sec[:sec.index("</table>")]
        rows = re.findall(r"<tr><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td></tr>", sec, re.S)
        return sec, rows

    def test_the_d2d_table_has_the_three_ways_side_by_side_barlink_first(self):
        doc, _ = self._doc(bar1=True, nccl=True)
        sec, rows = self._d2d_rows(self._render(doc))
        heads = re.findall(r"<th>([^<]*)<br>", sec)
        self.assertEqual(len(heads), 3)
        self.assertIn("barlink BAR1 direct (operating path", heads[0])
        self.assertIn("NCCL", heads[1])
        self.assertIn("Host staging pinned (fallback, not the operating path)", heads[2])
        self.assertEqual(len(rows), 6)                                     # one row per ORDERED pair
        for _pair, bar1, nccl, stage in rows:
            self.assertIn("<sup>meas.</sup>", bar1)
            self.assertIn("<sup>meas.</sup>", nccl)
            self.assertIn("SHM/direct/direct", nccl)                        # the transport NCCL chose
            self.assertEqual(stage.count("<sup>meas.</sup>"), 3)             # pipelined, serial, latency
        self.assertNotIn("not measured", "".join(r[1] for r in rows))      # the headline column is all numbers

    def test_the_bar1_column_is_nicht_gemessen_while_the_staging_column_has_numbers_never_filled_from_it(self):
        # MUTANT guard: "staging as the headline"
        doc, _ = self._doc(bar1=False, nccl=False)
        sec, rows = self._d2d_rows(self._render(doc))
        self.assertEqual(len(rows), 6)
        for _pair, bar1, nccl, stage in rows:
            self.assertIn(">not measured<", bar1)
            self.assertNotIn("<sup>meas.</sup>", bar1)
            self.assertIn(">not measured<", nccl)
            self.assertEqual(stage.count("<sup>meas.</sup>"), 3)
        self.assertIn("NOT MEASURED", self._render(doc))

    def test_the_definitions_are_shown_and_no_faster_slower_or_capability_claim_is_in_the_page_or_the_script(self):
        doc, _ = self._doc(bar1=True, nccl=True)
        h = self._render(doc)
        self.assertIn("no delivery time at the receiver", h)
        self.assertIn("round trip / 2", h)
        with open(os.path.join(STATIC, "hwprofil.js"), encoding="utf-8") as fh:
            js = fh.read()
        for bad in ("schneller", "langsamer", "Fähigkeit des Links", "kann der Link", "faster than", "slower than"):
            self.assertNotIn(bad.lower(), h.lower())
            self.assertNotIn(bad.lower(), js.lower())

    def test_both_latencies_the_reference_list_and_the_stage0_title_are_in_the_page(self):
        doc, _ = self._doc(bar1=True, nccl=True)
        # give the two ways a second latency, and a stage-0 NCCL table
        for r in doc["d2d"]["pairs"]:
            for w in ("barlink_bar1", "nccl"):
                r[w]["lat_dev_us"] = {"v": 1.5, "src": "gemessen", "at": 1.0, "probe": "p", "unit": "µs", "note": "ohne Host-Sync"}
        doc["links"].append({"src": 0, "dst": 1, "transport": "nccl", "transport_label": "NCCL via host (stage-0 probe, 30.07.2026)",
                             "gbs": {"v": 5.1, "src": "gemessen", "at": 1.0, "probe": "hw_profile-x.json", "unit": "GB/s"},
                             "lat_us": {"v": None, "src": "nicht gemessen", "note": "keine Latenz"}})
        h = self._render(doc)
        sec, rows = self._d2d_rows(h)
        self.assertIn("µs without host sync per round", sec)
        for _pair, bar1, nccl, _stage in rows:
            self.assertEqual(bar1.count("<sup>meas.</sup>"), 3)          # rate, latency 1, latency 2
            self.assertIn("[", bar1)
        self.assertIn("Already measured reference values", h)
        self.assertIn("barlink_bar1.py:75-83", h)
        self.assertIn("27b-nvfp4-dual.env:122", h)
        self.assertIn("NCCL via host (stage-0 probe, 30.07.2026)", h)
        self.assertNotIn("nccl p2p", h.lower())
        self.assertNotIn("NCCL p2p", h)

    def test_the_link_rows_show_the_narrow_card_and_the_utilisation_against_the_current_width(self):
        doc, _ = self._doc()
        h = self._render(doc)
        cells = [re.sub(r"<[^>]+>", "", re.sub(r"<sup>.*?</sup>", "", c)) for c in self._row(h, "Link (generation x width)")]
        self.assertEqual(len(cells), 3)
        self.assertIn("Gen4 x4", cells[0])
        self.assertIn("max Gen4 x16", cells[0])
        self.assertEqual(self._row(h, "Theoretical per direction")[0].count("7.88"), 1)
        h2d = self._row(h, "H2D measured / theoretical")[0]
        self.assertIn("82.5", h2d)                                          # 6.5 / 7.88
        self.assertIn("<sup>est.</sup>", h2d)                             # a calculation, labelled as derived

    def test_the_hover_of_a_latency_names_median_and_minimum(self):
        doc, _ = self._doc()
        h = self._render(doc)
        cells = self._row(h, "H2D latency")
        self.assertIn("median", cells[0])
        self.assertIn("minimum of the sample 9.0", cells[0])


class TestWindowAndBudget(unittest.TestCase):
    def test_the_window_is_fifteen_minutes_and_the_child_cap_fits_inside_it(self):
        self.assertEqual(hwprofil.WINDOW, "15m")
        self.assertLessEqual(hwprofil.CHILD_CAP_S + hwprofil.END_MARGIN_S, 15 * 60)
        self.assertGreater(hwprofil.CHILD_CAP_S, 10 * 60 - 60)      # mehr als die alten 9 min: der kalte Lauf passt
        self.assertGreater(hwprofil.MIN_LEFT_S, hwprofil.END_MARGIN_S)


if __name__ == "__main__":
    unittest.main()
