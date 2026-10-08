"""AP-D (Plan Profil-Planer 06.10.): der Trockenlauf des Profil-Editors auf dem Orakel-Weg und ``POST /api/profil/propose``.

Gepinnt:
  * ``ProfilEditor.dry_run`` fragt das Orakel (Launcher-Trockenlauf im Kindprozess); das Rueckgabeformat bleibt (ok, goes, verdict, rejections,
    notes, cards, force_note, reference), neu sind quelle, orakel (Ausgang, Profil-Hash, Cache) und verdikte.  Ein Absturz des Launchers ist ein
    Verdikt ORAKEL-ABSTURZ (nicht forcebar), nie ein Fehler der Route.  Kann das Orakel nicht fragen, gilt die Teilpruefung des Planer-Gates
    MIT Notiz.
  * Mit dem ECHTEN Orakel (Kindprozess, Launcher-Trockenlauf): Referenz-Rig = keine Verweigerung (wie das Gate heute); zwei Karten = PROFILE-VECTORS
    benannt; der Vorschlag traegt N-Vektoren, kein Verdikt nennt einen; das Startprofil ist ein flliper.server/1 mit Herkunft planer, Verdikt und Kanten
    je Wert; der zweite gleiche Aufruf kommt aus dem Cache.
  * ``force_verdict`` des Dashboards und ``propose_verdict.force_state_of`` des Planers lesen das Register gleich.
"""

import copy
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import kartenplan as K  # noqa: E402
from rigdash import profil as P  # noqa: E402
from rigdash import profil_oracle as ORA  # noqa: E402

FIXTURE_TREE = os.path.join(HERE, "fixtures", "kartenplan", "planner_tree", "python")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
REPO_PY = os.path.join(REPO_ROOT, "python")
REPO_CATALOG = os.path.join(os.path.dirname(HERE), "profil_data", "catalog.json")
PLANER_FIX = os.path.join(REPO_ROOT, "test", "registered", "unit", "weg2", "fixtures", "planer_1006")
REPLAY_REF = os.path.join(REPO_ROOT, "test", "registered", "unit", "weg2", "fixtures", "xchg_launch_replay_0911", "nvml_devices_1378.json")
CENSUS_27B = "/spinning/gpu-arb/weg2/census/xchg_census_weg2xsn246_27198a2711.json"
#: line probes (module / function exists in the tree under test, never a sha or a branch name): the Dual form (dual_green.py) marks the 27B launcher
#: line; the NF line re-stages the P-card reference at N != 3 (AP2 1006, ``launcher._restage_p_card_reference``), the 27B line refuses W167
DUAL_LINE = os.path.isfile(os.path.join(REPO_ROOT, "python", "sglang", "srt", "weg2", "dual_green.py"))
def _launcher_defines(name):
    try:
        with open(os.path.join(REPO_ROOT, "python", "sglang", "srt", "weg2", "launcher.py"), encoding="utf-8") as fh:
            return ("\ndef %s(" % name) in fh.read()
    except OSError:
        return False
RESTAGES_P_CARD = _launcher_defines("_restage_p_card_reference")

ENV = """\
# shellcheck shell=bash
PROFILE_NAME=demo
PROFILE_LINE=nf
PROFILE_FORMAT=int4-mixed
PROFILE_STATUS=experimentell
PROFILE_OWNER="the owner"
PROFILE_CARD_COUNT=3
PROFILE_INVENTORY=RTX5090,RTX3080,RTX3080
PROFILE_ARGS=(--model /m --p-bs 2 --pp-stage-ratio 29,11,8 --pp-attn-stage-ratio 8,4,4 --p-hostgap
              "--extra-p=--rank-moe-ratio 183,137,168" --env-p "SGLANG_MOE_SCRATCH_SLOTS=74,48,48")
profile_form_env() {
  _form SGLANG_WEG2_OWNED_BASE stated
}
"""

RIG = [{"card": "rtx3080-20", "pcie": {"gen": 4, "lanes": 4}}, {"card": "rtx5090-32", "pcie": {"gen": 5, "lanes": 8}},
       {"card": "rtx3080-20", "pcie": {"gen": 4, "lanes": 8}}]


class FakeOracle:
    """Stand-in for ``OracleService``: records the requests, answers with a canned document (or an error)."""

    def __init__(self, verdikt=None, error=None, propose=None):
        self.verdikt, self.error, self.propose, self.calls = verdikt, error, propose, []

    def ask(self, kind, req, parts):
        self.calls.append((kind, req, parts))
        if self.error:
            return {"ok": False, "error": self.error}
        if kind == "propose":
            return dict(copy.deepcopy(self.propose), ok=True, cached=False, cache_key="prop123")
        return {"ok": True, "verdikt": copy.deepcopy(self.verdikt), "cached": False, "cache_key": "abc123"}


def _v(code, **kw):
    d = {"code": code, "ebene": "lauf", "forcebar": True, "force_state": "force", "grund": code + " grund", "konsequenz": "k", "text": code + " text", "titel": code,
         "werte": [], "durchgelassen": True}
    d.update(kw)
    return d


def _doc(verdikte, ausgang="geht_mit_force", **kw):
    d = {"schema": "flliper.verdikt/1", "n": 3, "verdikte": verdikte, "ausgang": ausgang, "geht": ausgang == "geht", "geht_mit_force": ausgang != "absturz",
         "forced": [], "plan": {}, "zaehlung": {}, "profil": {"datei_sha256": "f" * 64, "eingabe_sha256": "e" * 64}, "argv_sha256": "a" * 64,
         "orakel": {"dauer_s": 1.5, "version": {"launcher": "x"}, "laeufe": 1, "notizen": ["note from the oracle"]}}
    d.update(kw)
    return d


def editor(tmp, oracle=None, hardware=None, release_dir=None, tree=FIXTURE_TREE):
    rel = release_dir or os.path.join(tmp, "rel")
    usr = os.path.join(tmp, "usr")
    if release_dir is None:
        os.makedirs(rel)
        with open(os.path.join(rel, "demo.env"), "w") as fh:
            fh.write(ENV)
    kp = K.Kartenplaner(tree=tree)
    return P.ProfilEditor(kartenplaner=kp, release_dir=rel, user_dir=usr, tree=tree, catalog_file=REPO_CATALOG, oracle=oracle, hardware=hardware)


class DryRunWithOracle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="apd_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, oracle, cards=RIG):
        ed = editor(self.tmp, oracle=oracle)
        return ed, ed.dry_run(ed.load("release", "demo")["doc"], cards)

    def test_the_return_format_is_kept_and_new_fields_are_added(self):
        orc = FakeOracle(_doc([]), )
        orc.verdikt = _doc([], ausgang="geht")
        _ed, d = self._run(orc)
        for k in ("ok", "goes", "verdict", "rejections", "notes", "cards", "force_note", "reference"):          # profil.py:655-669 of the base
            self.assertIn(k, d)
        self.assertEqual(d["quelle"], "orakel")
        self.assertEqual(d["orakel"]["ausgang"], "geht")
        self.assertEqual(d["orakel"]["profil"]["datei_sha256"], "f" * 64)
        self.assertEqual((d["orakel"]["cached"], d["orakel"]["cache_key"]), (False, "abc123"))
        self.assertEqual(d["verdikte"], [])
        self.assertEqual([q["code"] for q in d["rejections"]], ["PROFIL-STATUS"])          # entrypoint code, the synthetic profile is experimentell
        self.assertIn("note from the oracle", d["notes"])
        self.assertTrue(any("replica of the cards" in n for n in d["notes"]))
        self.assertEqual([c["index"] for c in d["cards"]], [0, 1, 2])

    def test_the_request_to_the_oracle_is_the_rendered_profile_on_the_chosen_cards(self):
        orc = FakeOracle(_doc([], ausgang="geht"))
        ed, d = self._run(orc)
        (kind, req, parts), = orc.calls
        self.assertEqual(kind, "verdikt")
        doc = ed.load("release", "demo")["doc"]
        pj, _ = ed.mods()
        self.assertEqual(req["basis"]["env_text"], pj.render_env(doc))
        self.assertEqual(len(req["inventar"]["cards"]), 3)
        self.assertEqual(parts["env_sha256"], ORA.sha256_text(pj.render_env(doc)))
        self.assertEqual([c["entry"]["id"] for c in req["inventar"]["cards"]], ["rtx3080-20", "rtx5090-32", "rtx3080-20"])
        self.assertTrue(any("synthetic" in n.lower() for n in d["notes"]))

    def test_forced_refusals_carry_class_and_force_state_of_the_register(self):
        hc = _v("HW-COUNT", text="HW-COUNT: 2 cards would be ...", launcher_code=None)
        hu = _v("HW-UNCALIBRATED", text="HW-UNCALIBRATED: profile 'x'")
        blk = _v("PROFILE-VECTORS", ebene="blocker", parent="HW-COUNT")
        _ed, d = self._run(FakeOracle(_doc([hc, blk, hu])), RIG[:2])
        by = {q["code"]: q for q in d["rejections"] if "verdikt" in q}            # the oracle's rejections (PROFILE_CARD_COUNT of the entrypoint is another HW-COUNT)
        self.assertTrue({"HW-COUNT", "HW-UNCALIBRATED"} <= set(by))
        self.assertNotIn("PROFILE-VECTORS", {q["code"] for q in d["rejections"]})   # a blocker is named inside HW-COUNT, it is not a rejection of its own
        for c in ("HW-COUNT", "HW-UNCALIBRATED"):
            q = by[c]
            self.assertEqual((q["klass"], q["force_state"], q["forcebar"]), ("wert", "force", True))
            self.assertIn("force overrides it", q["force"])
            self.assertTrue(q["why_class"])
            self.assertIn("verdikt", q)
        self.assertEqual([x["code"] for x in d["verdikte"]], ["HW-COUNT", "PROFILE-VECTORS", "HW-UNCALIBRATED"])      # all verdicts, blockers too

    def test_a_crash_of_the_launcher_is_a_verdict_not_a_route_error(self):
        crash = _v("ORAKEL-ABSTURZ", ebene="absturz", forcebar=None, force_state="blockiert", durchgelassen=False, exc_type="IndexError",
                   wo="<TREE>/python/sglang/srt/weg2/launcher.py:15326 in f", text="IndexError: list index out of range (<TREE>/python/sglang/srt/weg2/launcher.py:15326 in f)")
        _ed, d = self._run(FakeOracle(_doc([_v("HW-COUNT"), crash], ausgang="absturz")), RIG[:2] + RIG[:2])
        by = {q["code"]: q for q in d["rejections"]}
        a = by["ORAKEL-ABSTURZ"]
        self.assertEqual((a["klass"], a["forcebar"], a["force_state"]), ("nicht_forcebar", False, "blockiert"))
        self.assertIn("no", a["force"])
        self.assertIn("launcher.py:15326", a["source"])
        self.assertIn("IndexError", a["text"])
        self.assertIn("remain even with force", d["verdict"])
        self.assertEqual(d["orakel"]["ausgang"], "absturz")
        self.assertFalse(d["goes"])

    def test_a_refusal_with_a_register_code_keeps_the_registers_words(self):
        last = _v("LAUNCHER-UNKLASSIFIZIERT", forcebar=False, force_state="blockiert", durchgelassen=False, launcher_code="W19", text="W19 dormant-residue ...")
        _ed, d = self._run(FakeOracle(_doc([_v("HW-COUNT"), last], ausgang="verweigert")))
        q = {x["code"]: x for x in d["rejections"]}["LAUNCHER-UNKLASSIFIZIERT"]
        self.assertEqual((q["klass"], q["force_state"]), ("nicht_forcebar", "blockiert"))
        self.assertIn("W19", q["source"])

    def test_oracle_cannot_ask_falls_back_to_the_planner_gate_with_a_note(self):
        _ed, d = self._run(FakeOracle(error="Python of the sglang environment is missing: /nowhere"), RIG[:2])
        self.assertEqual(d["quelle"], "gate")
        self.assertNotIn("orakel", d)
        self.assertTrue({"HW-COUNT", "HW-UNCALIBRATED"} <= {q["code"] for q in d["rejections"]})
        self.assertTrue(any("Oracle (launcher dry run) not available" in n and "/nowhere" in n for n in d["notes"]), d["notes"])
        # the same rejections as an editor without any oracle
        ed2 = editor(self.tmp + "_x", oracle=None) if os.makedirs(self.tmp + "_x") is None else None
        try:
            d2 = ed2.dry_run(ed2.load("release", "demo")["doc"], RIG[:2])
        finally:
            shutil.rmtree(self.tmp + "_x", ignore_errors=True)
        self.assertEqual([q["code"] for q in d["rejections"]], [q["code"] for q in d2["rejections"]])
        self.assertEqual(d2["quelle"], "gate")

    def test_an_oracle_that_could_not_ask_is_not_a_verdict(self):
        fehler = _v("ORAKEL-FEHLER", ebene="orakel", forcebar=None, force_state="blockiert", text="OSError: scratch dir not writable")
        _ed, d = self._run(FakeOracle(_doc([fehler], ausgang="orakel_fehler")), RIG[:2])
        self.assertEqual(d["quelle"], "gate")
        self.assertNotIn("ORAKEL-FEHLER", [q["code"] for q in d["rejections"]])
        self.assertTrue(any("could not be asked" in n and "scratch dir" in n for n in d["notes"]))

    def test_the_card_count_of_the_profile_is_still_checked(self):
        """PROFILE_CARD_COUNT is the entrypoint's gate (the launcher does not read it): kept beside the oracle's verdicts."""
        _ed, d = self._run(FakeOracle(_doc([], ausgang="geht")), RIG[:2])
        hc = [q for q in d["rejections"] if q["code"] == "HW-COUNT"]
        self.assertEqual(len(hc), 1)
        self.assertIn("PROFILE_CARD_COUNT=3", hc[0]["source"])

    def test_the_rig_cards_go_with_their_real_uuids(self):
        """Chosen cards that are exactly the NVML cards of this rig (hardware profile) are asked with the real profile, not synthetic cards."""
        hw = _hw_profile(_rows(3))
        orc = FakeOracle(_doc([], ausgang="geht"))
        ed = editor(self.tmp, oracle=orc, hardware=lambda: {"ok": True, "profile": hw})
        d = ed.dry_run(ed.load("release", "demo")["doc"], RIG)
        (_k, req, _p), = orc.calls
        self.assertEqual(req["inventar"].get("hardware", {}).get("schema"), "flliper.hardware/1")
        self.assertNotIn("cards", req["inventar"])
        self.assertTrue(any("real UUIDs" in n for n in d["notes"]))
        # two other cards: synthetic
        orc2 = FakeOracle(_doc([], ausgang="geht"))
        ed2 = editor(self.tmp + "_y", oracle=orc2, hardware=lambda: {"ok": True, "profile": hw}) if os.makedirs(self.tmp + "_y") is None else None
        try:
            ed2.dry_run(ed2.load("release", "demo")["doc"], RIG[:2])
        finally:
            shutil.rmtree(self.tmp + "_y", ignore_errors=True)
        self.assertIn("cards", orc2.calls[0][1]["inventar"])


def _rows(n):
    with open(REPLAY_REF, encoding="utf-8") as fh:
        rows = json.load(fh)
    return sorted(rows, key=lambda r: r["index"])[:n]


def _hw_profile(rows):
    """A minimal ``flliper.hardware/1`` of replay rows (what ``replay_from_hardware_profile`` reads)."""
    cards = []
    for r in rows:
        cards.append({"nvml_index": r["index"], "uuid": r["uuid"], "name": r["name"], "pci_bus_id": r["pci_bus_id"],
                      "cc": [r["cc_major"], r["cc_minor"]], "vram_total_mib": {"v": r["total_bytes"] >> 20, "src": "NVML"},
                      "bar1_total_mib": {"v": None, "src": "not measured"}, "pcie": {"max_gen": {"v": None}, "max_width": {"v": None}}})
    return {"schema": "flliper.hardware/1", "cards": cards}


class OracleServiceCache(unittest.TestCase):
    """Cache je (Inventar, Form, Argv-Hash) + Stand der Quellen; Fehler werden nicht gemerkt."""

    def _svc(self, answers):
        svc = ORA.OracleService(FIXTURE_TREE, python="/nowhere/python")
        calls = []

        def fake_request(req):
            calls.append(req)
            return answers.pop(0)

        svc.request = fake_request
        return svc, calls

    def test_the_second_equal_question_is_a_cache_hit(self):
        svc, calls = self._svc([{"ok": True, "verdikt": {"n": 1}}])
        a = svc.ask("verdikt", {"x": 1}, {"inventar": ["a"], "env_sha256": "h1", "form": None})
        b = svc.ask("verdikt", {"x": 1}, {"inventar": ["a"], "env_sha256": "h1", "form": None})
        self.assertEqual((a["cached"], b["cached"]), (False, True))
        self.assertEqual(a["cache_key"], b["cache_key"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["what"], "verdikt")
        self.assertEqual(svc.cache_info()["treffer"], 1)

    def test_every_part_of_the_key_matters(self):
        base = {"inventar": ["a"], "env_sha256": "h1", "form": "flip"}
        svc, calls = self._svc([{"ok": True, "verdikt": {}} for _ in range(4)])
        svc.ask("verdikt", {}, base)
        svc.ask("verdikt", {}, dict(base, inventar=["a", "b"]))             # another inventory
        svc.ask("verdikt", {}, dict(base, form="tp"))                        # another form
        svc.ask("verdikt", {}, dict(base, env_sha256="h2"))                  # drift of the profile (plan 4c)
        self.assertEqual(len(calls), 4)

    def test_a_changed_source_of_the_oracle_is_another_key(self):
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, "sglang", "srt", "weg2")
            os.makedirs(w)
            for n in ORA.SOURCES:
                with open(os.path.join(w, n), "w") as fh:
                    fh.write("x")
            svc = ORA.OracleService(d, python="/nowhere/python")
            k1 = svc.key("verdikt", {"a": 1})
            with open(os.path.join(w, "launcher.py"), "w") as fh:
                fh.write("xy")
            self.assertNotEqual(k1, svc.key("verdikt", {"a": 1}))

    def test_errors_are_not_cached(self):
        svc, calls = self._svc([{"ok": False, "error": "tot"}, {"ok": True, "verdikt": {}}])
        self.assertFalse(svc.ask("verdikt", {}, {"p": 1})["ok"])
        self.assertTrue(svc.ask("verdikt", {}, {"p": 1})["ok"])
        self.assertEqual(len(calls), 2)

    def test_cache_is_bounded(self):
        svc, calls = self._svc([{"ok": True, "verdikt": {}} for _ in range(5)])
        svc.cache_size = 3
        for i in range(5):
            svc.ask("verdikt", {}, {"i": i})
        self.assertEqual(svc.cache_info()["eintraege"], 3)

    def test_unknown_kind_and_dead_worker_are_errors_not_exceptions(self):
        svc = ORA.OracleService(FIXTURE_TREE, python="/nowhere/python")
        self.assertFalse(svc.ask("start", {}, {})["ok"])
        res = svc.ask("verdikt", {}, {"p": 1})                                # no such python: the service says so
        self.assertFalse(res["ok"])
        self.assertIn("/nowhere/python", res["error"])


class OracleChildScope(unittest.TestCase):
    """Review AP-D 1: der Kindprozess laeuft hinter einem Befehlspraefix (eigener cgroup-Scope), faellt bei einem kaputten Praefix einmal zurueck."""

    WORKER = (
        "import json, os, sys\n"
        "print(json.dumps({'ok': True}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    r = json.loads(line)\n"
        "    print(json.dumps({'id': r['id'], 'ok': True, 'prefix_seen': os.environ.get('ORACLE_PREFIX_SEEN')}), flush=True)\n")

    def _worker(self):
        d = tempfile.mkdtemp(prefix="oracle-scope-")
        self.addCleanup(shutil.rmtree, d, True)
        w = os.path.join(d, "w.py")
        with open(w, "w") as fh:
            fh.write(self.WORKER)
        return w

    def _svc(self, prefix):
        svc = ORA.OracleService(FIXTURE_TREE, python=sys.executable, worker=self._worker(), start_timeout_s=20, timeout_s=20, prefix=prefix)
        self.addCleanup(svc._stop)
        return svc

    def test_the_child_starts_behind_the_prefix(self):
        svc = self._svc(["env", "ORACLE_PREFIX_SEEN=1"])
        # the child's env is scrubbed by _env(); the prefix process is the one that sets it for the command it execs
        r = svc.request({"what": "ping"})
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["prefix_seen"], "1")
        self.assertFalse(svc.prefix_fallback)

    def test_a_broken_prefix_falls_back_once_and_says_so(self):
        for prefix in (["/nonexistent/prefix-cmd"], ["false"]):
            svc = self._svc(prefix)
            r = svc.request({"what": "ping"})
            self.assertTrue(r["ok"], (prefix, r))
            self.assertIsNone(r["prefix_seen"])
            self.assertTrue(svc.prefix_fallback)

    def test_no_prefix_is_the_old_start(self):
        svc = self._svc(None)
        self.assertEqual(svc.prefix, [])
        self.assertTrue(svc.request({"what": "ping"})["ok"])
        self.assertFalse(svc.prefix_fallback)

    def test_the_default_prefix_is_an_own_scope_above_the_measured_peak(self):
        # peak RSS of a full NF dry run on the reference rig, measured 06.10. (review AP-D): 1789432 kB
        self.assertEqual(ORA.ORACLE_PREFIX[:3], ("systemd-run", "--scope", "-q"))
        limit = [a for a in ORA.ORACLE_PREFIX if a.startswith("MemoryMax=")][0]
        self.assertEqual(limit, "MemoryMax=4G")
        self.assertGreater(4 * 1024 * 1024, 1789432)
        with mock.patch("shutil.which", return_value=None):
            self.assertEqual(ORA.default_prefix(), [])
        with mock.patch("shutil.which", return_value="/usr/bin/systemd-run"):
            self.assertEqual(ORA.default_prefix()[0], "/usr/bin/systemd-run")


class ProposeRequests(unittest.TestCase):
    """The validation of ``ProfilEditor.propose`` (no child process)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="apd_")
        self.ed = editor(self.tmp, oracle=FakeOracle(_doc([])))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_form_ziele_basis_and_inventory_are_checked(self):
        for body, needle in (({"basis": {"kind": "release", "name": "demo"}, "form": "quad"}, "form must be flip, tp, dual or single"),
                             ({"basis": {"kind": "release", "name": "demo"}, "ziele": {"x": 1}}, "unknown goals"),
                             ({"basis": {"kind": "release", "name": "demo"}, "ziele": {"seats": 0}}, "Goal seats must be between"),
                             ({"basis": {"kind": "release", "name": "demo"}, "ziele": {"p_cut": "x"}}, "Goal p_cut must be one of"),
                             ({"basis": {"kind": "release", "name": "demo"}, "inventar": "alles"}, "inventar must be"),
                             ({"basis": {"kind": "wild", "name": "demo"}}, "basis.kind"),
                             ({"basis": {"kind": "release", "name": "../etc"}}, "invalid profile name"),
                             ({"basis": {"kind": "release", "name": "demo"}, "model_path": "/etc"}, "model root")):
            with self.assertRaises(P.ProfilError, msg=str(body)) as cm:
                self.ed.propose(body)
            self.assertIn(needle, str(cm.exception), body)

    def test_without_an_oracle_it_says_so(self):
        ed = editor(self.tmp + "_n", oracle=None) if os.makedirs(self.tmp + "_n") is None else None
        try:
            with self.assertRaises(P.ProfilError) as cm:
                ed.propose({"basis": {"kind": "release", "name": "demo"}})
        finally:
            shutil.rmtree(self.tmp + "_n", ignore_errors=True)
        self.assertIn("The oracle is not configured", str(cm.exception))

    def test_doc_keys_of_the_proposals_labels(self):
        dk = P.ProfilEditor.doc_key
        self.assertEqual(dk("--d-bs"), "flag:--d-bs")
        self.assertEqual(dk("--extra-d --rank-moe-ratio"), "extra:D:--rank-moe-ratio")
        self.assertEqual(dk("--extra-p --pp-stage-ratio"), "extra:P:--pp-stage-ratio")
        self.assertEqual(dk("--env-p SGLANG_MOE_SCRATCH_SLOTS"), "env:P:SGLANG_MOE_SCRATCH_SLOTS")
        self.assertEqual(dk("env SGLANG_WEG2_L15_MIB"), "export:SGLANG_WEG2_L15_MIB")
        self.assertIsNone(dk("--pp-stage-ratio (Seed)"))

    def test_the_oracle_error_comes_back_as_ok_false(self):
        ed = editor(self.tmp + "_e", oracle=FakeOracle(error="no model profile: not mounted")) if os.makedirs(self.tmp + "_e") is None else None
        try:
            r = ed.propose({"basis": {"kind": "release", "name": "demo"}, "inventar": RIG})
        finally:
            shutil.rmtree(self.tmp + "_e", ignore_errors=True)
        self.assertFalse(r["ok"])
        self.assertIn("not mounted", r["error"])


def _werte(*rows):
    out = []
    for key, alt, wert, policy, zustand in rows:
        out.append({"key": key, "group": "-", "policy": policy, "alt": alt, "wert": wert, "eintraege": 1, "zustand": zustand, "herkunft": "Herkunft von " + key,
                    "grund": "Grund von " + key, "in_argv": wert is not None, "geaendert": alt != wert})
    return out


class ProposeStartprofil(unittest.TestCase):
    """``propose()`` of the editor with a canned answer of the child: the Startprofil is the base profile + the values of the proposal."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="apd_")
        werte = _werte(("--p-bs", "2", "3", "seats", "vorgeschlagen"),
                       ("--extra-p --rank-moe-ratio", "183,137,168", "300,212", "moe_ratio", "unbelegt"),
                       ("--env-p SGLANG_MOE_SCRATCH_SLOTS", "74,48,48", "74,48", "scratch", "unbelegt"),
                       ("--pp-stage-ratio", "29,11,8", None, "cut", "unbelegt"),
                       ("--d-only", None, "", "form", "vorgeschlagen"),
                       ("--pp-stage-ratio (Seed)", None, "32,8", "cut_seed", "unbelegt"),
                       ("--d-bs", "6", "6", "knob", "vorgeschlagen"))
        self.answer = {"vorschlag": {"schema": "flliper.propose-a/1", "form": "flip", "n": 2, "werte": werte, "ziele": {"seats": 3, "kv_tokens": 262144},
                                     "cards": [], "inventory": {}, "seeds": {}, "fit": {"level": "ja"}, "unbelegt": ["x"], "hinweise": [], "blocker": [],
                                     "vektorlaengen": {"--extra-p --rank-moe-ratio": 2}, "vektoren_ok": True, "vektoren_falsch": {}, "basis": "demo.env"},
                       "verdikt": _doc([_v("HW-COUNT")], n=2, ausgang="geht_mit_force"),
                       "je_wert": {"--extra-p --rank-moe-ratio": [{"code": "UNBELEGT", "ebene": "wert", "forcebar": None, "force_state": "hinweis",
                                                                   "grund": "g", "konsequenz": "k"}]},
                       "launch": {"argv": ["--model", "/m"], "env": {}}}
        self.orc = FakeOracle(propose=self.answer)
        self.ed = editor(self.tmp, oracle=self.orc, hardware=lambda: {"ok": True, "profile": _hw_profile(_rows(2))})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _body(self, **kw):
        return dict({"basis": {"kind": "release", "name": "demo"}, "form": "flip", "inventar": "rig", "ziele": {"seats": 3}}, **kw)

    def test_the_startprofil_is_the_base_profile_with_the_proposals_values(self):
        r = self.ed.propose(self._body())
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["schema"], "flliper.propose-d/1")
        sp = r["startprofil"]
        self.assertEqual(sp["schema"], "flliper.server/1")
        self.assertTrue(sp["verifiziert"], sp["probleme"])
        rows = {x["key"]: x for x in sp["view"]["rows"]}
        self.assertEqual(rows["flag:--p-bs"]["value"], "3")
        self.assertEqual(rows["flag:--p-bs"]["origin"], "planer")
        self.assertEqual(rows["extra:P:--rank-moe-ratio"]["value"], "300,212")
        self.assertEqual(rows["env:P:SGLANG_MOE_SCRATCH_SLOTS"]["value"], "74,48")
        self.assertIn("flag:--d-only", rows)                                   # a flag the profile did not have
        self.assertNotIn("flag:--pp-stage-ratio", rows)                         # the value the proposal could not derive is removed, not invented
        self.assertEqual(rows["flag:--p-hostgap"]["origin"], "profil")          # untouched values keep their origin
        self.assertEqual(rows["flag:--model"]["origin"], "profil")
        self.assertEqual(r["nicht_uebernommen"], [])
        self.assertEqual(sp["doc"]["meta"]["vorschlag"]["form"], "flip")
        self.assertEqual(sp["doc"]["meta"]["planner"]["flag:--pp-stage-ratio"], "32,8")       # the seed: the planner's calculation, shown as planner-only
        self.assertTrue(any(x["key"] == "flag:--pp-stage-ratio" for x in sp["view"]["planner_only"]))
        self.assertEqual(sp["name"], "demo-vorschlag")
        pj, _ = self.ed.mods()
        self.assertEqual(pj.doc_id(sp["doc"]), sp["doc"]["id"])

    def test_every_value_carries_origin_verdicts_and_edges(self):
        r = self.ed.propose(self._body())
        by = {w["label"]: w for w in r["werte"]}
        w = by["--extra-p --rank-moe-ratio"]
        self.assertEqual((w["key"], w["zustand"], w["geaendert"]), ("extra:P:--rank-moe-ratio", "unbelegt", True))
        self.assertEqual(w["herkunft"], "Herkunft von --extra-p --rank-moe-ratio")
        self.assertEqual([v["code"] for v in w["verdikte"]], ["UNBELEGT"])
        self.assertTrue(w["kanten"], "the catalog's edges of --rank-moe-ratio")
        for e in w["kanten"]:
            self.assertIn("to", e)
        self.assertEqual(by["--d-bs"]["geaendert"], False)
        self.assertEqual(by["--pp-stage-ratio (Seed)"]["key"], None)
        row = {x["key"]: x for x in r["startprofil"]["view"]["rows"]}["extra:P:--rank-moe-ratio"]
        self.assertEqual(row["vorschlag"]["zustand"], "unbelegt")
        self.assertEqual(row["vorschlag"]["verdikte"][0]["code"], "UNBELEGT")
        self.assertEqual(row["vorschlag"]["kanten"], row["explain"]["depends"])

    def test_the_balken_request_is_the_h2_contract(self):
        for form, want in (("flip", "flip"), ("tp", "d_only")):
            r = self.ed.propose(self._body(form=form))
            self.assertEqual((r["balken"]["route"], r["balken"]["what"], r["balken"]["form"]), ("/api/profil/recompute", "phase_bars", want))
            self.assertEqual(r["balken"]["doc"]["schema"], "flliper.server/1")

    def test_the_request_to_the_child(self):
        self.ed.propose(self._body(ziele={"seats": 3, "kv_tokens": 200000}))
        (kind, req, parts), = self.orc.calls
        self.assertEqual(kind, "propose")
        self.assertTrue(req["basis"]["env_path"].endswith("demo.env"))
        self.assertEqual((req["form"], req["ziele"]), ("flip", {"seats": 3, "kv_tokens": 200000}))
        self.assertEqual(req["inventar"]["hardware"]["schema"], "flliper.hardware/1")
        self.assertEqual(len(parts["basis_sha256"]), 64)                        # plan 4c: the profile file's hash is part of the key
        self.assertEqual(parts["form"], "flip")
        self.assertEqual(req["model_path"], None)                               # the demo profile names no model: the child takes the basis'

    def test_a_user_profile_is_a_basis_too_by_its_rendered_text(self):
        r = self.ed.load("release", "demo")
        self.ed.save(r["doc"], "mein-demo")
        self.ed.propose(self._body(basis={"kind": "user", "name": "mein-demo"}))
        (_k, req, parts), = self.orc.calls
        self.assertIn("PROFILE_NAME=mein-demo", req["basis"]["env_text"])
        self.assertEqual(len(parts["basis_sha256"]), 64)

    def test_the_synthetic_inventory_is_a_catalog_card_list(self):
        # 5090 + 3090: not the NVML cards of this rig (5090 + 3080): datasheet cards with synthetic UUIDs
        self.ed.propose(self._body(inventar=[{"card": "rtx5090-32", "pcie": {"gen": 5, "lanes": 16}}, {"card": "rtx3090-24", "pcie": {"gen": 4, "lanes": 16}}]))
        (_k, req, parts), = self.orc.calls
        self.assertEqual([c["entry"]["id"] for c in req["inventar"]["cards"]], ["rtx5090-32", "rtx3090-24"])
        self.assertNotIn("hardware", req["inventar"])
        self.assertEqual(parts["inventar"][0][0], "katalog")
        # the rig's own two cards go with their real UUIDs
        self.orc.calls.clear()
        self.ed.propose(self._body(inventar=RIG[:2]))
        self.assertIn("hardware", self.orc.calls[0][1]["inventar"])
        self.assertEqual(self.orc.calls[0][2]["inventar"][0][0], "nvml")

    def test_user_paths_need_the_model_root_check(self):
        calls = []
        ed = P.ProfilEditor(kartenplaner=K.Kartenplaner(tree=FIXTURE_TREE), release_dir=os.path.join(self.tmp, "rel"), user_dir=os.path.join(self.tmp, "usr"),
                            tree=FIXTURE_TREE, catalog_file=REPO_CATALOG, oracle=self.orc, hardware=lambda: {"ok": True, "profile": _hw_profile(_rows(2))},
                            check_path=lambda p, what: calls.append((p, what)) or "/ok/" + p.strip("/"))
        ed.propose(self._body(model_path="/models/x"))
        self.assertEqual(calls, [("/models/x", "model_path")])
        self.assertEqual(self.orc.calls[-1][1]["model_path"], "/ok/models/x")
        ed.check_path = lambda p, what: (_ for _ in ()).throw(ValueError("path is not under a model root"))
        with self.assertRaises(P.ProfilError):
            ed.propose(self._body(model_path="/etc/passwd"))


class ProposeRoute(unittest.TestCase):
    def serve(self, edition="rig"):
        from http.server import ThreadingHTTPServer
        from types import SimpleNamespace
        import http.client  # noqa: F401
        import threading
        from rigdash import server as S

        tmp = tempfile.mkdtemp(prefix="apd_r_")
        self.addCleanup(shutil.rmtree, tmp, True)
        base = ProposeStartprofil("test_the_request_to_the_child")
        base.tmp = tmp
        base.setUp()
        self.addCleanup(base.tearDown)
        app = SimpleNamespace(edition=edition, profil=base.ed, version="t")
        srv = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(app))
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        self.addCleanup(srv.server_close)
        return srv.server_address[1], base

    def call(self, port, path, body, headers=None):
        import http.client

        c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        h = dict(headers or {}, **{"Content-Type": "application/json"})
        c.request("POST", path, body=json.dumps(body), headers=h)
        r = c.getresponse()
        return r.status, r.read().decode()

    def test_the_route(self):
        port, base = self.serve()
        st, txt = self.call(port, "/api/profil/propose", base._body())
        self.assertEqual(st, 200, txt[:300])
        js = json.loads(txt)
        self.assertEqual((js["ok"], js["schema"], js["startprofil"]["schema"]), (True, "flliper.propose-d/1", "flliper.server/1"))
        st, txt = self.call(port, "/api/profil/propose", base._body(form="quad"))
        self.assertEqual(st, 400)
        self.assertIn("form must be flip, tp, dual or single", json.loads(txt)["error"])
        st, txt = self.call(port, "/api/profil/propose", base._body(basis={"kind": "release", "name": "../x"}))
        self.assertEqual(st, 400)

    def test_lan_only_like_the_other_profile_routes(self):
        port, base = self.serve()
        self.assertEqual(self.call(port, "/api/profil/propose", base._body(), headers={"X-Forwarded-For": "1.2.3.4"})[0], 403)
        port2, base2 = self.serve("release")
        self.assertEqual(self.call(port2, "/api/profil/propose", base2._body())[0], 200)          # the editor is in the release edition too


@unittest.skipUnless(os.path.exists(CENSUS_27B) and os.path.isdir(REPO_PY), "the real oracle needs the rig box (census, models) and the planner tree")
class RealOracle(unittest.TestCase):
    """The REAL oracle: a child process with this checkout's sglang, the launcher dry run on a replayed inventory (16-18 s a run to the end)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="apd_real_")
        cls.snaps = {}
        ck = os.path.join(PLANER_FIX, "checkpoints")
        for n in sorted(os.listdir(ck)):
            if os.path.isfile(os.path.join(ck, n, "manifest.json")):
                cls.snaps[n] = os.path.join(ck, n)
        cls.oracle = ORA.OracleService(REPO_PY, python=sys.executable, request_extra={"snapshots": cls.snaps})
        cls.rel = os.path.join(cls.tmp, "rel")
        os.makedirs(cls.rel)
        for n in ("nf-int4-h6-abl.env", "27b-base.env"):
            shutil.copy(os.path.join(PLANER_FIX, "profiles", n), os.path.join(cls.rel, n))
        # the real planner tree for the editor (the dry run asks the oracle; profile_json/refusals load by path from it)
        cls.hw3 = _hw_profile(_rows(3))
        cls.hw2 = _hw_profile([r for r in _rows(3) if r["index"] in (1, 0)][:2])
        for i, c in enumerate(cls.hw2["cards"]):
            c["nvml_index"] = i

    @classmethod
    def tearDownClass(cls):
        cls.oracle.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _ed(self, hw):
        usr = os.path.join(self.tmp, "usr")
        return P.ProfilEditor(kartenplaner=K.Kartenplaner(tree=REPO_PY), release_dir=self.rel, user_dir=usr, tree=REPO_PY, catalog_file=REPO_CATALOG,
                              oracle=self.oracle, hardware=lambda: {"ok": True, "profile": hw})

    def test_1_reference_rig_has_no_refusal_and_equals_the_gate(self):
        ed = self._ed(self.hw3)
        doc = ed.load("release", "nf-int4-h6-abl")["doc"]
        t0 = time.time()
        d = ed.dry_run(doc, RIG)
        self.assertEqual(d["quelle"], "orakel", d["notes"])
        self.assertEqual(d["orakel"]["ausgang"], "geht", [(x["code"], x["grund"][:160]) for x in d.get("verdikte", [])])
        self.assertEqual(d["rejections"], [])
        self.assertTrue(d["goes"])
        self.assertEqual(d["orakel"]["profil"]["quelle"], "nf-int4-h6-abl.env")
        self.assertEqual(len(d["orakel"]["profil"]["eingabe_sha256"]), 64)
        self.assertEqual(len(d["orakel"]["profil_text_sha256"]), 64)
        self.assertEqual(d["orakel"]["profil_datei_sha256"], doc["meta"]["based_on"]["sha256"])        # plan 4c: the hash of the profile FILE it was loaded from
        self.assertTrue(d["orakel"]["plan"]["pp_cut"])
        # the same codes as the planner gate alone says today (an editor without oracle)
        gate = P.ProfilEditor(kartenplaner=K.Kartenplaner(tree=REPO_PY), release_dir=self.rel, user_dir=os.path.join(self.tmp, "u2"), tree=REPO_PY,
                              catalog_file=REPO_CATALOG).dry_run(doc, RIG)
        self.assertEqual(gate["quelle"], "gate")
        self.assertEqual([q["code"] for q in gate["rejections"]], [q["code"] for q in d["rejections"]])
        # the second equal question is answered from the cache
        t1 = time.time()
        d2 = ed.dry_run(doc, RIG)
        self.assertTrue(d2["orakel"]["cached"])
        self.assertEqual(d2["orakel"]["cache_key"], d["orakel"]["cache_key"])
        self.assertLess(time.time() - t1, 5.0)
        self.assertGreater(t1 - t0, 0.0)

    def test_2_two_cards_name_the_vectors_of_the_profile(self):
        ed = self._ed(self.hw2)
        doc = ed.load("release", "nf-int4-h6-abl")["doc"]
        two = [{"card": "rtx5090-32", "pcie": {"gen": 5, "lanes": 8}}, {"card": "rtx3080-20", "pcie": {"gen": 4, "lanes": 8}}]
        d = ed.dry_run(doc, two)
        self.assertEqual(d["quelle"], "orakel", d["notes"])
        codes = {q["code"] for q in d["rejections"]}
        self.assertIn("HW-COUNT", codes)
        pv = [x for x in d["verdikte"] if x["code"] == "PROFILE-VECTORS"]
        self.assertEqual(len(pv), 1, [(x["code"]) for x in d["verdikte"]])
        self.assertIn("--rank-moe-ratio", pv[0]["werte"])
        self.assertIn("SGLANG_MOE_SCRATCH_SLOTS", pv[0]["werte"])
        self.assertEqual(pv[0]["parent"], "HW-COUNT")
        self.assertNotEqual(d["orakel"]["ausgang"], "geht")

    def test_3_propose_for_two_cards_is_a_startprofil_with_origin_verdict_and_edges(self):
        ed = self._ed(self.hw2)
        # the release profile of the line: the NF launcher cannot plan 27b-base (measured 07.10. on 2e68b3f94b: SystemExit '--p-chunk-policy dynamic:
        # need 0 < min_tokens <= max_tokens, got 4096/2048'); the 27B model holds 196608 tokens on two cards, the NF default is its own
        body = {"basis": {"kind": "release", "name": "27b-base" if DUAL_LINE else "nf-int4-h6-abl"}, "form": "flip", "inventar": "rig",
                "ziele": {"kv_tokens": 196608} if DUAL_LINE else {}}
        r = ed.propose(body)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["schema"], "flliper.propose-d/1")
        self.assertEqual((r["form"], r["n"]), ("flip", 2))
        sp = r["startprofil"]
        self.assertEqual(sp["schema"], "flliper.server/1")
        self.assertEqual(sp["doc"]["schema"], "flliper.server/1")
        self.assertTrue(sp["verifiziert"], sp["probleme"])
        # vectors of N entries; no verdict names a vector
        self.assertTrue(r["vorschlag"]["vektoren_ok"], r["vorschlag"]["vektoren_falsch"])
        self.assertEqual([x for x in r["verdikt"]["verdikte"] if x["code"] == "PROFILE-VECTORS"], [])
        # 27B line: the planner's dry run goes with --force; NF line: the same once the launcher re-stages the P-card reference, else it ends with W167
        self.assertEqual(r["verdikt"]["ausgang"], "geht_mit_force" if (DUAL_LINE or RESTAGES_P_CARD) else "verweigert",
                         [(x["code"], x["grund"][:100]) for x in r["verdikt"]["verdikte"]])
        self.assertEqual(r["verdikt"]["n"], 2)
        self.assertEqual(len(r["verdikt"]["profil"]["datei_sha256"]), 64)
        self.assertEqual(r["basis"]["sha256"], r["verdikt"]["profil"]["datei_sha256"])
        # every changed value is in the doc (origin planer) and the row shows what the proposal said
        rows = {x["key"]: x for x in sp["view"]["rows"]}
        changed = [w for w in r["werte"] if w["geaendert"] and w["key"]]
        self.assertTrue(changed)
        for w in changed:
            if w["wert"] is None:
                self.assertNotIn(w["key"], rows, w["label"])
                continue
            self.assertIn(w["key"], rows, w["label"])
            self.assertEqual(rows[w["key"]]["value"], str(w["wert"]), w["label"])
            self.assertEqual(rows[w["key"]]["origin"], "planer", w["label"])
            self.assertEqual(rows[w["key"]]["vorschlag"]["zustand"], w["zustand"])
        for w in r["werte"]:
            for f in ("key", "label", "wert", "zustand", "herkunft", "grund", "verdikte", "kanten"):
                self.assertIn(f, w)
            for v in w["verdikte"]:
                for f in ("code", "forcebar", "force_state", "grund", "konsequenz"):
                    self.assertIn(f, v)
        self.assertTrue(any(w["kanten"] for w in r["werte"]), "the edge catalog gives at least one value its dependencies")
        self.assertEqual(r["balken"]["what"], "phase_bars")
        self.assertEqual(r["balken"]["form"], "flip")
        self.assertEqual(r["balken"]["doc"]["schema"], "flliper.server/1")
        # the second equal question is a cache hit
        t0 = time.time()
        r2 = ed.propose(body)
        self.assertTrue(r2["orakel"]["cached"])
        self.assertEqual(r2["verdikt"]["argv_sha256"], r["verdikt"]["argv_sha256"])
        self.assertLess(time.time() - t0, 5.0)
        # another goal is another question
        r3 = ed.propose(dict(body, ziele={"kv_tokens": 180000}))
        self.assertFalse(r3["orakel"]["cached"])

    def test_4_a_dead_child_is_a_note_and_the_gate_answers(self):
        dead = ORA.OracleService(REPO_PY, python="/nowhere/python")
        ed = P.ProfilEditor(kartenplaner=K.Kartenplaner(tree=REPO_PY), release_dir=self.rel, user_dir=os.path.join(self.tmp, "u3"), tree=REPO_PY,
                            catalog_file=REPO_CATALOG, oracle=dead)
        d = ed.dry_run(ed.load("release", "nf-int4-h6-abl")["doc"], RIG[:2])
        self.assertEqual(d["quelle"], "gate")
        self.assertTrue(any("Oracle (launcher dry run) not available" in n for n in d["notes"]))
        r = ed.propose({"basis": {"kind": "release", "name": "nf-int4-h6-abl"}, "inventar": RIG[:2]})
        self.assertFalse(r["ok"])
        self.assertIn("/nowhere/python", r["error"])


if __name__ == "__main__":
    unittest.main()
