"""Nutzer 06.10.2026: FLIPZEIT hat EINE Definition und EINE Berechnung (flipzeit.py).

    P>D = letzter P-Chunk fertig -> erstes Decode-Token erzeugt; D>P = letztes Decode-Token erzeugt -> erster
    Prefill-Chunk beginnt zu rechnen.  Ausnahme: kein Flip zaehlt, wenn kein Prefill oder Decode ansteht.

Anlass: Ueberblick und Verlauf zeigten verschiedene Werte fuer dieselbe Groesse (NF-Boot 10:29Z, Stand 11:07Z:
P>D n=16 / 2,58 s zuletzt vs. n=40; D>P zuletzt 4,67 s [ein noch offener, vorlaeufiger Flip] vs. 3,81 s).  Hier:
dieselbe Menge Flips gibt in Kachel Ueberblick, Kachel Verlauf, Diagramm und Boot-Liste dieselben Zahlen; ein
vorlaeufiger oder offener Flip geht nirgends ein; das Ende von D>P ist der Rechenbeginn auf PP0, nie der Dispatch."""

import os
import re
import shutil
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rigdash import activity, flipzeit, history, ipcboot, server, vmpush  # noqa: E402
from rigdash.tests import test_flipzeit_1002 as base  # noqa: E402

STATIC = os.path.join(os.path.dirname(__file__), "..", "static")
NOW = 1790990000.0


def _row(d, begin, total_ms, kind="ok", provisional=False, **kw):
    """One ipcboot.flip_views row (the keys the counting rule and the mark writer read)."""
    x = {"dir": d, "begin": begin, "kind": kind, "total_ms": total_ms, "provisional": provisional,
         "vorlauf_ms": 100.0, "layer_ms": 2000.0, "wake_kv_dc_ms": 10.0, "nachlauf_ms": 40.0,
         "rest_ms": (total_ms - 2150.0) if total_ms is not None else None,
         "leer_ms": None, "halt_ms": None, "park_ms": None, "vor_rest_ms": None}
    x.update(kw)
    return x


#: ONE flip history of an NF boot (begin = NOW - age): measured flips of both directions, a provisional D>P (the
#: newest, the "zuletzt 4,67 s" of the page), idle D>P flips with a full measured total (an idle layout swap),
#: an open flip and one with a missing endpoint
def _views(now=NOW):
    return [
        _row("P>D", now - 3000, 2500.0), _row("D>P", now - 2900, 3800.0),
        _row("P>D", now - 2000, 2600.0), _row("D>P", now - 1900, 5300.0, leer_ms=10.0, halt_ms=80.0, park_ms=0.0,
                                                  vor_rest_ms=10.0),
        _row("P>D", now - 1000, 2800.0), _row("P>D", now - 900, 3000.0),
        _row("D>P", now - 800, 37400.0, kind="leerlauf"),                    # idle swap: nobody waited
        _row("D>P", now - 700, 41000.0, kind="leerlauf"),
        _row("P>D", now - 600, None, kind="fehlt", missing="rankstats ..."),
        _row("D>P", now - 120, 4670.0, provisional=True),                    # D's log has not written its last rounds
        _row("P>D", now - 30, None, kind="offen"),
    ]


def _db_from_views(views, model="NF", now=NOW, extra_old=()):
    """The real writer (history.Recorder._mark_flip_views) over ``views`` -> a HistoryDB with its marks."""
    db = history.HistoryDB(None)
    rec = history.Recorder.__new__(history.Recorder)
    rec.db = db
    boots = list(extra_old) + list(views)
    with mock.patch.object(ipcboot, "flip_views", return_value=boots), \
            mock.patch.object(ipcboot, "timeline_view", return_value={"segs": []}), \
            mock.patch.object(ipcboot, "boot_start", return_value=now - 7200):
        rec._mark_flip_views("k", model, {"terminal": False}, SimpleNamespace(ring=[]), now)
    return db


def _numbers(tile):
    return {d: (tile[d]["n"], tile[d]["last_ms"], tile[d]["p50_ms"], tile[d]["p90_ms"], tile[d]["max_ms"])
            for d in flipzeit.DIRS}


class Counting(unittest.TestCase):
    def test_only_a_finished_measured_flip_counts(self):
        v = _views()
        self.assertEqual([x["dir"] + x["kind"] + ("p" if x["provisional"] else "") for x in v if flipzeit.counted(x)],
                         ["P>Dok", "D>Pok", "P>Dok", "D>Pok", "P>Dok", "P>Dok"])
        # the Leerlauf flip HAS a measured total and still does not count (user's exception)
        idle = next(x for x in v if x["kind"] == "leerlauf")
        self.assertIsNotNone(idle["total_ms"])
        self.assertFalse(flipzeit.counted(idle))
        # a provisional one has a total and does not count
        prov = next(x for x in v if x["provisional"])
        self.assertIsNotNone(prov["total_ms"])
        self.assertFalse(flipzeit.counted(prov))

    def test_provisional_is_never_zuletzt(self):
        """The newest D>P is provisional (4,67 s): the tile's 'zuletzt' is the newest FINISHED one, 5,3 s."""
        pts = flipzeit.from_views(_views())
        st = flipzeit.stats(pts, "D>P")
        self.assertEqual((st["n"], st["last_ms"]), (2, 5300.0))
        self.assertNotIn(4670.0, [p["ms"] for p in pts])

    def test_the_writer_writes_exactly_the_counted_flips_and_tallies_the_idle_ones(self):
        db = _db_from_views(_views())
        marks = db.marks("NF", NOW - 7200, NOW)
        t2t = [m for m in marks if m["kind"] == "flip_t2t"]
        self.assertEqual(sorted(m["v"] for m in t2t), [2500.0, 2600.0, 2800.0, 3000.0, 3800.0, 5300.0])
        skip = [m for m in marks if m["kind"] == "flip_skip"]
        self.assertEqual(sorted(m["label"] for m in skip), ["D>P leerlauf ipc", "D>P leerlauf ipc"])
        self.assertTrue(all(m["v"] is None for m in skip))

    def test_vm_points_use_the_same_rule(self):
        lines = vmpush.flip_view_points(_views(), "NF", "b", set())
        totals = sorted(float(l.split("} ")[1].split()[0]) for l in lines if 'part="total"' in l)
        self.assertEqual(totals, [2500.0, 2600.0, 2800.0, 3000.0, 3800.0, 5300.0])

    def test_quantile_is_nearest_rank(self):
        self.assertEqual(flipzeit.quantile([2500, 2600, 2800, 3000], 0.5), 2600)
        self.assertEqual(flipzeit.quantile([2500, 2600, 2800, 3000], 0.9), 3000)
        self.assertIsNone(flipzeit.quantile([], 0.5))

    def test_label_roundtrip_is_the_partition(self):
        x = _row("D>P", 5.0, 5300.0, leer_ms=10.0, halt_ms=80.0, park_ms=0.0, vor_rest_ms=10.0)
        lab = flipzeit.mark_label(x)
        self.assertEqual(lab, "D>P v=100 l=2000 w=10 n=40 r=3150 (leer=10 halt=80 park=0 vr=10)")
        p = flipzeit.parse_label(lab + " ipc")
        self.assertEqual(p["dir"], "D>P")
        self.assertEqual(sum(p["parts"][k] for k in flipzeit.PART_KEYS), 5300)       # Zerlegung teilt den Total vollstaendig
        self.assertIsNone(flipzeit.parse_label("garbage"))


class OneNumberOneSet(unittest.TestCase):
    """The same flips -> the same numbers in Ueberblick, Verlauf, Diagramm and Boot-Liste."""

    def setUp(self):
        # one old flip pair 2 h before NOW belongs to the SAME boot but not to the last hour
        old = [_row("P>D", NOW - 6500, 9000.0), _row("D>P", NOW - 6400, 9100.0)]
        self.db = _db_from_views(_views(), extra_old=old)

    def tiles(self):
        ueberblick = history.flip_tile(self.db, "NF", NOW - flipzeit.OVERVIEW_S, NOW,
                                       flipzeit.window_label(flipzeit.OVERVIEW_S))
        verlauf = history.view(self.db, None, "NF", "1h", now=NOW)
        boot = history.flip_tile(self.db, "NF", NOW - 7200, NOW, "ganzer Boot")
        return ueberblick, verlauf, boot

    def test_ueberblick_equals_verlauf_equals_diagram_for_the_same_window(self):
        ue, vl, _ = self.tiles()
        vt = vl["tiles"]["flip"]
        self.assertEqual(_numbers(ue), _numbers(vt))
        self.assertEqual(_numbers(ue)["P>D"], (4, 3000.0, 2600.0, 3000.0, 3000.0))
        self.assertEqual(_numbers(ue)["D>P"], (2, 5300.0, 3800.0, 5300.0, 5300.0))
        self.assertEqual(ue["window"]["label"], vt["window"]["label"])
        self.assertEqual(vt["window"]["label"], "letzte 60 min")
        # the diagram draws exactly the marks the tile counted: its own mean/p50/max over those points
        pts = [m for m in vl["marks"] if m["kind"] == "flip_t2t" and m["v"] is not None]
        for d in flipzeit.DIRS:
            vals = [m["v"] for m in pts if m["label"].startswith(d)]
            self.assertEqual(len(vals), vt[d]["n"])
            self.assertEqual(flipzeit.quantile(vals, 0.5), vt[d]["p50_ms"])
            self.assertEqual(max(vals), vt[d]["max_ms"])
            self.assertAlmostEqual(sum(vals) / len(vals), sum(p["ms"] for p in flipzeit.from_marks(
                [m for m in pts if m["label"].startswith(d)])) / len(vals))
        # the idle flips are named, not counted
        self.assertEqual(vt["D>P"]["idle_n"], 2)
        self.assertEqual(vt["P>D"]["idle_n"], 0)

    def test_boot_list_is_the_same_function_over_the_whole_boot(self):
        ue, vl, boot = self.tiles()
        self.assertEqual(_numbers(boot)["P>D"][0], 5)                    # + the old flip of the same boot
        self.assertEqual(boot["P>D"]["max_ms"], 9000.0)
        self.assertEqual(boot["window"]["label"], "ganzer Boot")
        # a Verlauf range that covers the boot gives the boot's numbers (same marks, same function)
        v24 = history.view(self.db, None, "NF", "24h", now=NOW)["tiles"]["flip"]
        self.assertEqual(_numbers(v24), _numbers(boot))
        self.assertEqual(v24["window"]["label"], "letzte 24 h")
        # ... and a different window is labelled as different, with its own numbers
        self.assertNotEqual(_numbers(ue), _numbers(boot))

    def test_zoom_window_is_labelled(self):
        z = history.view(self.db, None, "NF", "1h", now=NOW, lo_hi=(NOW - 3100, NOW - 1500))["tiles"]["flip"]
        self.assertEqual(z["window"]["label"], "gezoomter Ausschnitt (27 min)")
        self.assertEqual((z["P>D"]["n"], z["D>P"]["n"]), (2, 2))

    def test_server_attaches_both_windows_from_the_marks(self):
        app = SimpleNamespace(hist=self.db, flip_boot_cache={})
        ipc = {"dir": "/spinning/docker-acceptance/nf/state/x", "tag": "nf"}
        live = {"stem": "a", "ipc": ipc, "live": True, "first_t": NOW - 7200, "age_s": 1.0}
        done = {"stem": "b", "ipc": ipc, "live": False, "first_t": NOW - 7200, "last_log_t": NOW, "age_s": 5000.0}
        boots = [live, done]
        server.App.flip_zeit(app, boots, NOW)
        ue, _, boot = self.tiles()
        self.assertEqual(_numbers(live["flip_zeit"]), _numbers(ue))
        # the Ueberblick switch "seit Boot": a live boot gets the same function over [its start, now]
        self.assertEqual(live["flip_boot"]["window"]["label"], "seit Boot")
        self.assertEqual(_numbers(live["flip_boot"]), _numbers(self.tiles()[2]))
        self.assertEqual(live["flip_zeit"]["window"]["label"], "letzte 60 min")
        self.assertEqual(_numbers(done["flip_boot"]), _numbers(boot))
        self.assertEqual(server.lean_boot(done)["flip_boot"], done["flip_boot"])
        # the list row carries no ring figure of its own any more
        self.assertNotIn("flip_last", server.lean_boot(done))

    def test_overview_switch_is_stored_and_both_windows_are_labelled(self):
        html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
        for needle in ('localStorage.getItem("rigdash.flipWin")', 'localStorage.setItem("rigdash.flipWin"',
                       "function setFlipWin(", "flipWinOf = (b) =>", "seit Boot", "60 min"):
            self.assertIn(needle, html, needle)
        # no tile reads b.flip_zeit directly any more: the switch decides the window
        self.assertEqual(html.count("b.flip_zeit"), 3)           # flipWinOf + two comments

    def test_features_row_reads_the_same_tile(self):
        from rigdash import features
        ue, _, _ = self.tiles()
        val, src = features._cur_flip({"flip_zeit": ue, "flip_count": 8}, None, None)
        self.assertIn("P→D Median 2,60 s", val)
        self.assertIn("letzte 60 min", src)


class RedOnBase(unittest.TestCase):
    """The pre-06.10. rule, rebuilt from its parts, disagrees with the marks on the very same flips: the ring figure
    counted the provisional flip (n, 'zuletzt') and kept its own window; the mark writer skipped it.  (The mutants of
    the new rule are in the report: provisional counted, ring instead of range, dispatch instead of compute start.)"""

    def test_old_ring_rule_counted_the_provisional_flip_and_the_new_one_does_not(self):
        v = _views()
        old_ok = [x for x in v if x["dir"] == "D>P" and x["kind"] == "ok" and x["total_ms"] is not None]
        self.assertEqual((len(old_ok), old_ok[-1]["total_ms"]), (3, 4670.0))      # old flip_last: n=3, zuletzt 4,67 s
        new = flipzeit.stats(flipzeit.from_views(v), "D>P")
        self.assertEqual((new["n"], new["last_ms"]), (2, 5300.0))


class DefinitionEndpoints(unittest.TestCase):
    def test_dp_ends_at_the_compute_start_on_pp0_never_the_dispatch(self):
        """The front's pp_first_forward stamp (rank beacon forward_ct rose = first forward began) is the end; the leg-1
        dispatch (46,608) and P's last stage (53,23) are not."""
        ipc = base._ipc_dp("pp_first_forward", 47.21)
        ipc["flip_user_time"][0]["pp_last_start_ts"] = base.T + 53.23
        x = ipcboot.flip_views(base.SEGS, ipc, base.T + 60.0, base._ring(), d_rounds=base.D_ROUNDS + [(base.T + 58, base.T + 58.03)])[0]
        self.assertEqual(x["kind"], "ok")
        self.assertAlmostEqual(x["end"], base.T + 47.21, places=3)
        self.assertAlmostEqual(x["total_ms"], (47.21 - 44.096) * 1000, delta=1)
        self.assertNotAlmostEqual(x["end"], base.T + 46.608, delta=0.5)
        self.assertTrue(flipzeit.counted(x))

    def test_dp_without_waiter_is_an_idle_flip_and_not_counted(self):
        """flip_user_time.idle_flip = no waiter at the begin and no park (front_state_ipc.DpFlipClock.begin): nothing
        was pending for P -- the user's exception.  The row still has its measured total; it is simply not counted."""
        ipc = base._ipc_dp("pp_first_forward", 47.21)
        ipc["flip_user_time"][0]["idle_flip"] = True
        x = ipcboot.flip_views(base.SEGS, ipc, base.T + 60.0, base._ring(), d_rounds=base.D_ROUNDS + [(base.T + 58, base.T + 58.03)])[0]
        self.assertEqual(x["kind"], "leerlauf")
        self.assertIsNotNone(x["total_ms"])
        self.assertFalse(flipzeit.counted(x))
        self.assertEqual(flipzeit.from_views([x]), [])

    def test_pd_flip_the_front_marks_idle_is_not_counted(self):
        """Front marker flip_begin.idle_flip (front_state_ipc.pd_idle_flip: nothing waited for D at the begin, Nutzer
        06.10.): the P>D row keeps its measured total and is a Leerlauf flip.  Without the marker the same flip counts."""
        ring = base._ring_pd()
        plain = ipcboot.flip_views(base.SEGS_PD, base._ipc_pd(), base.T + 60.0, ring, d_rounds=None)[0]
        self.assertEqual(plain["kind"], "ok")
        self.assertTrue(flipzeit.counted(plain))
        idle_row = None
        for marker, kind in ((True, "leerlauf"), (False, "ok")):
            ipc = base._ipc_pd()
            for e in ipc["ipc_events"]:
                if e["type"] == "flip_begin":
                    e["data"]["idle_flip"] = marker
            x = ipcboot.flip_views(base.SEGS_PD, ipc, base.T + 60.0, ring, d_rounds=None)[0]
            self.assertEqual(x["kind"], kind)
            self.assertAlmostEqual(x["total_ms"], 2603, delta=1)          # measured either way
            self.assertEqual(flipzeit.counted(x), kind == "ok")
            idle_row = x if marker else idle_row
        db = _db_from_views([idle_row])
        self.assertEqual([m["kind"] for m in db.marks("NF", 0, base.T + 100)], ["flip_skip"])

    def test_real_provisional_row_is_not_counted(self):
        x = ipcboot.flip_views(base.SEGS, base._ipc_dp(), base.T + 60.0, base._ring(), d_rounds=base.D_ROUNDS, arrivals={})[0]
        self.assertTrue(x["provisional"] and x["kind"] == "ok")
        self.assertFalse(flipzeit.counted(x))
        self.assertEqual(vmpush.flip_view_points([x], "NF", "b", set()), [])
        self.assertEqual(flipzeit.from_views([x]), [])

    def test_open_and_missing_flips_are_not_counted_but_named(self):
        v = _views()
        dg = ipcboot.flip_diag(v)
        self.assertEqual(dg["P>D"]["missing_n"], 1)
        self.assertEqual(dg["P>D"]["missing"], "rankstats ...")            # the newest finished-or-missing P>D lacks its endpoint
        self.assertTrue(dg["P>D"]["open"])


class OneDefinitionInTheCode(unittest.TestCase):
    def _read(self, *p):
        with open(os.path.join(STATIC, *p), encoding="utf-8") as fh:
            return fh.read()

    def test_every_surface_carries_the_same_definition_text(self):
        for d, txt in flipzeit.DEFINITION.items():
            self.assertIn('"%s": "%s"' % (d, txt), self._read("index.html"), d)
            self.assertIn('"%s": "%s"' % (d, txt), self._read("grafik.js"), d)

    def test_no_second_definition_is_left(self):
        live_src = open(os.path.join(os.path.dirname(__file__), "..", "live.py"), encoding="utf-8").read()
        for needle in ("FIRST_TOKEN_HEADLINE_FOR_27B", "apply_ipc_first_work", "flip_times_view", "flip_total\" if"):
            self.assertNotIn(needle, live_src, needle)
        html = self._read("index.html")
        for needle in ("flip_last", "flip_times", "vm_boot || {}).flips", "Ende des letzten P-Prefill-Chunks"):
            self.assertNotIn(needle, html, needle)
        for needle in ("Decode-Ende → erster Prefill-Forward auf P (PP0)", "P-Chunk-Ende → erstes Decode-Token"):
            self.assertNotIn(needle, self._read("grafik.js"), needle)
        for name in ("flip_last", "flip_times_of", "_stats"):
            self.assertFalse(hasattr(ipcboot, name), name)
        self.assertFalse(hasattr(vmpush, "flip_stats_from"))

    def test_boot_view_has_no_figure_of_its_own(self):
        # the card payload (ipcboot.boot_view) neither builds flip_last nor flip_times any more
        src = open(os.path.join(os.path.dirname(__file__), "..", "ipcboot.py"), encoding="utf-8").read()
        self.assertNotIn('v["flip_last"]', src)
        self.assertNotIn('v["flip_times"]', src)

    @unittest.skipUnless(shutil.which("bun") or os.path.exists("/root/.bun/bin/bun"), "no JS runtime for a syntax check")
    def test_the_pages_script_still_parses(self):
        bun = shutil.which("bun") or "/root/.bun/bin/bun"
        tmp = os.environ.get("TMPDIR") or "/root/.claude/jobs/1ab4cd30/tmp"
        os.makedirs(tmp, exist_ok=True)
        inline = re.findall(r"<script>(.*?)</script>", self._read("index.html"), re.S)
        path = os.path.join(tmp, "flipzeit_1006_inline.js")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(inline))
        for f in (path, os.path.join(STATIC, "grafik.js")):
            r = subprocess.run([bun, "build", "--no-bundle", f], capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr[-400:])


if __name__ == "__main__":
    unittest.main()
