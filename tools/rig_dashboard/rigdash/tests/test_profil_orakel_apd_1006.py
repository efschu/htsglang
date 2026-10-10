"""AP-D (Plan Profil-Planer 06.10.): der Trockenlauf des Profil-Editors auf dem Oracle-Weg und ``POST /api/profil/propose``.

Gepinnt:
  * ``ProfilEditor.dry_run`` fragt das Oracle (Launcher-Trockenlauf im Kindprozess); das Rueckgabeformat bleibt (ok, goes, verdict, rejections,
    notes, cards, force_note, reference), neu sind quelle, oracle (Ausgang, Profil-Hash, Cache) und verdikte.  Ein Absturz des Launchers ist ein
    Verdikt ORAKEL-ABSTURZ (nicht forcebar), nie ein Fehler der Route.  Kann das Oracle nicht fragen, gilt die Teilpruefung des Planer-Gates
    MIT Notiz.
  * Mit dem ECHTEN Oracle (Kindprozess, Launcher-Trockenlauf): Referenz-Rig = keine Verweigerung (wie das Gate heute); zwei Karten = PROFILE-VECTORS
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
from rigdash import profile_oracle as ORA  # noqa: E402

FIXTURE_TREE = os.path.join(HERE, "fixtures", "kartenplan", "planner_tree", "python")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
REPO_PY = os.path.join(REPO_ROOT, "python")
REPO_CATALOG = os.path.join(os.path.dirname(HERE), "profil_data", "catalog.json")
PLANER_FIX = os.path.join(REPO_ROOT, "test", "registered", "unit", "pdflip", "fixtures", "planer_1006")
REPLAY_REF = os.path.join(REPO_ROOT, "test", "registered", "unit", "pdflip", "fixtures", "xchg_launch_replay_0911", "nvml_devices_1378.json")
CENSUS_27B = "/spinning/gpu-arb/weg2/census/xchg_census_weg2xsn246_27198a2711.json"
#: line probes (module / function exists in the tree under test, never a sha or a branch name): the Dual form (dual_green.py) marks the 27B launcher
#: line; the NF line re-stages the P-card reference at N != 3 (AP2 1006, ``launcher._restage_p_card_reference``), the 27B line refuses W167
DUAL_LINE = os.path.isfile(os.path.join(REPO_ROOT, "python", "flliper", "srt", "pdflip", "dual_green.py"))
def _launcher_defines(name):
    try:
        with open(os.path.join(REPO_ROOT, "python", "flliper", "srt", "pdflip", "launcher.py"), encoding="utf-8") as fh:
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
              "--extra-p=--rank-moe-ratio 183,137,168" --env-p "FLLIPER_MOE_SCRATCH_SLOTS=74,48,48")
profile_form_env() {
  _form FLLIPER_PDFLIP_OWNED_BASE stated
}
"""

RIG = [{"card": "rtx3080-20", "pcie": {"gen": 4, "lanes": 4}}, {"card": "rtx5090-32", "pcie": {"gen": 5, "lanes": 8}},
       {"card": "rtx3080-20", "pcie": {"gen": 4, "lanes": 8}}]


class FakeOracle:
    """Stand-in for ``OracleService``: records the requests, answers with a canned document (or an error)."""

    def __init__(self, budget_verdict=None, error=None, propose=None):
        self.budget_verdict, self.error, self.propose, self.calls = budget_verdict, error, propose, []

    def ask(self, kind, req, parts):
        self.calls.append((kind, req, parts))
        if self.error:
            return {"ok": False, "error": self.error}
        if kind == "propose":
            return dict(copy.deepcopy(self.propose), ok=True, cached=False, cache_key="prop123")
        return {"ok": True, "verdict": copy.deepcopy(self.budget_verdict), "cached": False, "cache_key": "abc123"}


def _v(code, **kw):
    d = {"code": code, "level": "run", "forcebar": True, "force_state": "force", "reason": code + " reason", "consequence": "k", "text": code + " text", "title": code,
         "values": [], "durchgelassen": True}
    d.update(kw)
    return d


def _doc(rank_verdicts, outcome="ok_with_force", **kw):
    d = {"schema": "flliper.verdict/1", "n": 3, "verdikte": rank_verdicts, "outcome": outcome, "geht": outcome == "geht", "ok_with_force": outcome != "crash",
         "forced": [], "plan": {}, "zaehlung": {}, "profil": {"file_sha256": "f" * 64, "input_sha256": "e" * 64}, "argv_sha256": "a" * 64,
         "oracle": {"duration_s": 1.5, "version": {"launcher": "x"}, "runs": 1, "notizen": ["note from the oracle"]}}
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
        orc.budget_verdict = _doc([], outcome="geht")
        _ed, d = self._run(orc)
        for k in ("ok", "goes", "verdict", "rejections", "notes", "cards", "force_note", "reference"):          # profil.py:655-669 of the base
            self.assertIn(k, d)
        self.assertEqual(d["quelle"], "oracle")
        self.assertEqual(d["oracle"]["outcome"], "geht")
        self.assertEqual(d["oracle"]["profil"]["file_sha256"], "f" * 64)
        self.assertEqual((d["oracle"]["cached"], d["oracle"]["cache_key"]), (False, "abc123"))
        self.assertEqual(d["verdikte"], [])
        self.assertEqual([q["code"] for q in d["rejections"]], ["PROFIL-STATUS"])          # entrypoint code, the synthetic profile is experimentell
        self.assertIn("note from the oracle", d["notes"])
        self.assertTrue(any("replica of the cards" in n for n in d["notes"]))
        self.assertEqual([c["index"] for c in d["cards"]], [0, 1, 2])

    def test_the_request_to_the_oracle_is_the_rendered_profile_on_the_chosen_cards(self):
        orc = FakeOracle(_doc([], outcome="geht"))
        ed, d = self._run(orc)
        (kind, req, parts), = orc.calls
        self.assertEqual(kind, "verdict")
        doc = ed.load("release", "demo")["doc"]
        pj, _ = ed.mods()
        self.assertEqual(req["basis"]["env_text"], pj.render_env(doc))
        self.assertEqual(len(req["inventory"]["cards"]), 3)
        self.assertEqual(parts["env_sha256"], ORA.sha256_text(pj.render_env(doc)))
        self.assertEqual([c["entry"]["id"] for c in req["inventory"]["cards"]], ["rtx3080-20", "rtx5090-32", "rtx3080-20"])
        self.assertTrue(any("synthetic" in n.lower() for n in d["notes"]))

    def test_forced_refusals_carry_class_and_force_state_of_the_register(self):
        hc = _v("HW-COUNT", text="HW-COUNT: 2 cards would be ...", launcher_code=None)
        hu = _v("HW-UNCALIBRATED", text="HW-UNCALIBRATED: profile 'x'")
        blk = _v("PROFILE-VECTORS", level="blocker", parent="HW-COUNT")
        _ed, d = self._run(FakeOracle(_doc([hc, blk, hu])), RIG[:2])
        by = {q["code"]: q for q in d["rejections"] if "verdict" in q}            # the oracle's rejections (PROFILE_CARD_COUNT of the entrypoint is another HW-COUNT)
        self.assertTrue({"HW-COUNT", "HW-UNCALIBRATED"} <= set(by))
        self.assertNotIn("PROFILE-VECTORS", {q["code"] for q in d["rejections"]})   # a blocker is named inside HW-COUNT, it is not a rejection of its own
        for c in ("HW-COUNT", "HW-UNCALIBRATED"):
            q = by[c]
            self.assertEqual((q["klass"], q["force_state"], q["forcebar"]), ("value", "force", True))
            self.assertIn("force overrides it", q["force"])
            self.assertTrue(q["why_class"])
            self.assertIn("verdict", q)
        self.assertEqual([x["code"] for x in d["verdikte"]], ["HW-COUNT", "PROFILE-VECTORS", "HW-UNCALIBRATED"])      # all verdicts, blockers too

    def test_a_crash_of_the_launcher_is_a_verdict_not_a_route_error(self):
        crash = _v("ORAKEL-ABSTURZ", level="crash", forcebar=None, force_state="is_blocked", durchgelassen=False, exc_type="IndexError",
                   wo="<TREE>/python/flliper/srt/pdflip/launcher.py:15326 in f", text="IndexError: list index out of range (<TREE>/python/flliper/srt/pdflip/launcher.py:15326 in f)")
        _ed, d = self._run(FakeOracle(_doc([_v("HW-COUNT"), crash], outcome="crash")), RIG[:2] + RIG[:2])
        by = {q["code"]: q for q in d["rejections"]}
        a = by["ORAKEL-ABSTURZ"]
        self.assertEqual((a["klass"], a["forcebar"], a["force_state"]), ("nicht_forcebar", False, "is_blocked"))
        self.assertIn("no", a["force"])
        self.assertIn("launcher.py:15326", a["source"])
        self.assertIn("IndexError", a["text"])
        self.assertIn("remain even with force", d["verdict"])
        self.assertEqual(d["oracle"]["outcome"], "crash")
        self.assertFalse(d["goes"])

    def test_a_refusal_with_a_register_code_keeps_the_registers_words(self):
        last = _v("LAUNCHER-UNKLASSIFIZIERT", forcebar=False, force_state="is_blocked", durchgelassen=False, launcher_code="W19", text="W19 dormant-residue ...")
        _ed, d = self._run(FakeOracle(_doc([_v("HW-COUNT"), last], outcome="verweigert")))
        q = {x["code"]: x for x in d["rejections"]}["LAUNCHER-UNKLASSIFIZIERT"]
        self.assertEqual((q["klass"], q["force_state"]), ("nicht_forcebar", "is_blocked"))
        self.assertIn("W19", q["source"])

    def test_oracle_cannot_ask_falls_back_to_the_planner_gate_with_a_note(self):
        _ed, d = self._run(FakeOracle(error="Python of the flliper environment is missing: /nowhere"), RIG[:2])
        self.assertEqual(d["quelle"], "gate")
        self.assertNotIn("oracle", d)
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
        errors = _v("ORAKEL-FEHLER", level="oracle", forcebar=None, force_state="is_blocked", text="OSError: scratch dir not writable")
        _ed, d = self._run(FakeOracle(_doc([errors], outcome="oracle_error")), RIG[:2])
        self.assertEqual(d["quelle"], "gate")
        self.assertNotIn("ORAKEL-FEHLER", [q["code"] for q in d["rejections"]])
        self.assertTrue(any("could not be asked" in n and "scratch dir" in n for n in d["notes"]))

    def test_the_card_count_of_the_profile_is_still_checked(self):
        """PROFILE_CARD_COUNT is the entrypoint's gate (the launcher does not read it): kept beside the oracle's verdicts."""
        _ed, d = self._run(FakeOracle(_doc([], outcome="geht")), RIG[:2])
        hc = [q for q in d["rejections"] if q["code"] == "HW-COUNT"]
        self.assertEqual(len(hc), 1)
        self.assertIn("PROFILE_CARD_COUNT=3", hc[0]["source"])

    def test_the_rig_cards_go_with_their_real_uuids(self):
        """Chosen cards that are exactly the NVML cards of this rig (hardware profile) are asked with the real profile, not synthetic cards."""
        hw = _hw_profile(_rows(3))
        orc = FakeOracle(_doc([], outcome="geht"))
        ed = editor(self.tmp, oracle=orc, hardware=lambda: {"ok": True, "profile": hw})
        d = ed.dry_run(ed.load("release", "demo")["doc"], RIG)
        (_k, req, _p), = orc.calls
        self.assertEqual(req["inventory"].get("hardware", {}).get("schema"), "flliper.hardware/1")
        self.assertNotIn("cards", req["inventory"])
        self.assertTrue(any("real UUIDs" in n for n in d["notes"]))
        # two other cards: synthetic
        orc2 = FakeOracle(_doc([], outcome="geht"))
        ed2 = editor(self.tmp + "_y", oracle=orc2, hardware=lambda: {"ok": True, "profile": hw}) if os.makedirs(self.tmp + "_y") is None else None
        try:
            ed2.dry_run(ed2.load("release", "demo")["doc"], RIG[:2])
        finally:
            shutil.rmtree(self.tmp + "_y", ignore_errors=True)
        self.assertIn("cards", orc2.calls[0][1]["inventory"])


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
        svc, calls = self._svc([{"ok": True, "verdict": {"n": 1}}])
        a = svc.ask("verdict", {"x": 1}, {"inventory": ["a"], "env_sha256": "h1", "form": None})
        b = svc.ask("verdict", {"x": 1}, {"inventory": ["a"], "env_sha256": "h1", "form": None})
        self.assertEqual((a["cached"], b["cached"]), (False, True))
        self.assertEqual(a["cache_key"], b["cache_key"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["what"], "verdict")
        self.assertEqual(svc.cache_info()["treffer"], 1)

    def test_every_part_of_the_key_matters(self):
        base = {"inventory": ["a"], "env_sha256": "h1", "form": "flip"}
        svc, calls = self._svc([{"ok": True, "verdict": {}} for _ in range(4)])
        svc.ask("verdict", {}, base)
        svc.ask("verdict", {}, dict(base, inventory=["a", "b"]))             # another inventory
        svc.ask("verdict", {}, dict(base, form="tp"))                        # another form
        svc.ask("verdict", {}, dict(base, env_sha256="h2"))                  # drift of the profile (plan 4c)
        self.assertEqual(len(calls), 4)

    def test_a_changed_source_of_the_oracle_is_another_key(self):
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, "flliper", "srt", "pdflip")
            os.makedirs(w)
            for n in ORA.SOURCES:
                with open(os.path.join(w, n), "w") as fh:
                    fh.write("x")
            svc = ORA.OracleService(d, python="/nowhere/python")
            k1 = svc.key("verdict", {"a": 1})
            with open(os.path.join(w, "launcher.py"), "w") as fh:
                fh.write("xy")
            self.assertNotEqual(k1, svc.key("verdict", {"a": 1}))

    def test_errors_are_not_cached(self):
        svc, calls = self._svc([{"ok": False, "error": "tot"}, {"ok": True, "verdict": {}}])
        self.assertFalse(svc.ask("verdict", {}, {"p": 1})["ok"])
        self.assertTrue(svc.ask("verdict", {}, {"p": 1})["ok"])
        self.assertEqual(len(calls), 2)

    def test_cache_is_bounded(self):
        svc, calls = self._svc([{"ok": True, "verdict": {}} for _ in range(5)])
        svc.cache_size = 3
        for i in range(5):
            svc.ask("verdict", {}, {"i": i})
        self.assertEqual(svc.cache_info()["entries"], 3)

    def test_unknown_kind_and_dead_worker_are_errors_not_exceptions(self):
        svc = ORA.OracleService(FIXTURE_TREE, python="/nowhere/python")
        self.assertFalse(svc.ask("start", {}, {})["ok"])
        res = svc.ask("verdict", {}, {"p": 1})                                # no such python: the service says so
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
                             ({"basis": {"kind": "release", "name": "demo"}, "goals": {"x": 1}}, "unknown goals"),
                             ({"basis": {"kind": "release", "name": "demo"}, "goals": {"seats": 0}}, "Goal seats must be between"),
                             ({"basis": {"kind": "release", "name": "demo"}, "goals": {"p_cut": "x"}}, "Goal p_cut must be one of"),
                             ({"basis": {"kind": "release", "name": "demo"}, "inventory": "alles"}, "inventory must be"),
                             ({"basis": {"kind": "wild", "name": "demo"}}, "basis.kind"),
                             ({"basis": {"kind": "release", "name": "../etc"}}, "invalid profile name"),
                             ({"basis": {"kind": "release", "name": "demo"}, "model_path": "/etc"}, "model root")):
            with self.assertRaises(P.ProfileError, msg=str(body)) as cm:
                self.ed.propose(body)
            self.assertIn(needle, str(cm.exception), body)

    def test_without_an_oracle_it_says_so(self):
        ed = editor(self.tmp + "_n", oracle=None) if os.makedirs(self.tmp + "_n") is None else None
        try:
            with self.assertRaises(P.ProfileError) as cm:
                ed.propose({"basis": {"kind": "release", "name": "demo"}})
        finally:
            shutil.rmtree(self.tmp + "_n", ignore_errors=True)
        self.assertIn("The oracle is not configured", str(cm.exception))

    def test_doc_keys_of_the_proposals_labels(self):
        dk = P.ProfilEditor.doc_key
        self.assertEqual(dk("--d-bs"), "flag:--d-bs")
        self.assertEqual(dk("--extra-d --rank-moe-ratio"), "extra:D:--rank-moe-ratio")
        self.assertEqual(dk("--extra-p --pp-stage-ratio"), "extra:P:--pp-stage-ratio")
        self.assertEqual(dk("--env-p FLLIPER_MOE_SCRATCH_SLOTS"), "env:P:FLLIPER_MOE_SCRATCH_SLOTS")
        self.assertEqual(dk("env FLLIPER_PDFLIP_L15_MIB"), "export:FLLIPER_PDFLIP_L15_MIB")
        self.assertIsNone(dk("--pp-stage-ratio (Seed)"))

    def test_the_oracle_error_comes_back_as_ok_false(self):
        ed = editor(self.tmp + "_e", oracle=FakeOracle(error="no model profile: not mounted")) if os.makedirs(self.tmp + "_e") is None else None
        try:
            r = ed.propose({"basis": {"kind": "release", "name": "demo"}, "inventory": RIG})
        finally:
            shutil.rmtree(self.tmp + "_e", ignore_errors=True)
        self.assertFalse(r["ok"])
        self.assertIn("not mounted", r["error"])


def _values(*rows):
    out = []
    for key, alt, value, policy, state in rows:
        out.append({"key": key, "group": "-", "policy": policy, "alt": alt, "value": value, "entries": 1, "state": state, "source": "Herkunft von " + key,
                    "reason": "Grund von " + key, "in_argv": value is not None, "changed": alt != value})
    return out


class ProposeStartprofil(unittest.TestCase):
    """``propose()`` of the editor with a canned answer of the child: the Startprofil is the base profile + the values of the proposal."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="apd_")
        values = _values(("--p-bs", "2", "3", "seats", "vorgeschlagen"),
                       ("--extra-p --rank-moe-ratio", "183,137,168", "300,212", "moe_ratio", "unverified"),
                       ("--env-p FLLIPER_MOE_SCRATCH_SLOTS", "74,48,48", "74,48", "scratch", "unverified"),
                       ("--pp-stage-ratio", "29,11,8", None, "cut", "unverified"),
                       ("--d-only", None, "", "form", "vorgeschlagen"),
                       ("--pp-stage-ratio (Seed)", None, "32,8", "cut_seed", "unverified"),
                       ("--d-bs", "6", "6", "knob", "vorgeschlagen"))
        self.answer = {"proposal": {"schema": "flliper.propose-a/1", "form": "flip", "n": 2, "values": values, "goals": {"seats": 3, "kv_tokens": 262144},
                                     "cards": [], "inventory": {}, "seeds": {}, "fit": {"level": "ja"}, "unverified": ["x"], "notes": [], "blocker": [],
                                     "vector_lengths": {"--extra-p --rank-moe-ratio": 2}, "vectors_ok": True, "vectors_wrong": {}, "basis": "demo.env"},
                       "verdict": _doc([_v("HW-COUNT")], n=2, outcome="ok_with_force"),
                       "per_value": {"--extra-p --rank-moe-ratio": [{"code": "UNVERIFIED", "level": "value", "forcebar": None, "force_state": "note",
                                                                   "reason": "g", "consequence": "k"}]},
                       "launch": {"argv": ["--model", "/m"], "env": {}}}
        self.orc = FakeOracle(propose=self.answer)
        self.ed = editor(self.tmp, oracle=self.orc, hardware=lambda: {"ok": True, "profile": _hw_profile(_rows(2))})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _body(self, **kw):
        return dict({"basis": {"kind": "release", "name": "demo"}, "form": "flip", "inventory": "rig", "goals": {"seats": 3}}, **kw)

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
        self.assertEqual(rows["env:P:FLLIPER_MOE_SCRATCH_SLOTS"]["value"], "74,48")
        self.assertIn("flag:--d-only", rows)                                   # a flag the profile did not have
        self.assertNotIn("flag:--pp-stage-ratio", rows)                         # the value the proposal could not derive is removed, not invented
        self.assertEqual(rows["flag:--p-hostgap"]["origin"], "profil")          # untouched values keep their origin
        self.assertEqual(rows["flag:--model"]["origin"], "profil")
        self.assertEqual(r["nicht_uebernommen"], [])
        self.assertEqual(sp["doc"]["meta"]["proposal"]["form"], "flip")
        self.assertEqual(sp["doc"]["meta"]["planner"]["flag:--pp-stage-ratio"], "32,8")       # the seed: the planner's calculation, shown as planner-only
        self.assertTrue(any(x["key"] == "flag:--pp-stage-ratio" for x in sp["view"]["planner_only"]))
        self.assertEqual(sp["name"], "demo-proposal")
        pj, _ = self.ed.mods()
        self.assertEqual(pj.doc_id(sp["doc"]), sp["doc"]["id"])

    def test_every_value_carries_origin_verdicts_and_edges(self):
        r = self.ed.propose(self._body())
        by = {w["label"]: w for w in r["values"]}
        w = by["--extra-p --rank-moe-ratio"]
        self.assertEqual((w["key"], w["state"], w["changed"]), ("extra:P:--rank-moe-ratio", "unverified", True))
        self.assertEqual(w["source"], "Herkunft von --extra-p --rank-moe-ratio")
        self.assertEqual([v["code"] for v in w["verdikte"]], ["UNVERIFIED"])
        self.assertTrue(w["kanten"], "the catalog's edges of --rank-moe-ratio")
        for e in w["kanten"]:
            self.assertIn("to", e)
        self.assertEqual(by["--d-bs"]["changed"], False)
        self.assertEqual(by["--pp-stage-ratio (Seed)"]["key"], None)
        row = {x["key"]: x for x in r["startprofil"]["view"]["rows"]}["extra:P:--rank-moe-ratio"]
        self.assertEqual(row["proposal"]["state"], "unverified")
        self.assertEqual(row["proposal"]["verdikte"][0]["code"], "UNVERIFIED")
        self.assertEqual(row["proposal"]["kanten"], row["explain"]["depends"])

    def test_the_vision_section_of_the_proposal_reaches_the_page_data_only(self):
        """VISION-WEIGHTS AP4: ``proposal.vision`` (``flliper.vision-victim/1``) passes through as DATA; an answer without it adds no key."""
        self.assertNotIn("vision", self.ed.propose(self._body())["proposal"])
        sec = {"schema": "flliper.vision-victim/1", "aktiv": True, "opferart": "dense", "turm_mib": 878.8, "host_mib": 878.8,
               "vision_transient": {"name": "vision_transient", "label": "Vision transient", "mib": 878.8, "resident_mib": 0.0},
               "host_posten": {"name": "vision_victim_host", "mib": 878.8}, "verdict": {"code": "W105b", "stage": "ja"}}
        self.answer["proposal"]["vision"] = sec
        r = self.ed.propose(self._body())
        self.assertEqual(r["proposal"]["vision"], sec)

    def test_the_balken_request_is_the_h2_contract(self):
        for form, want in (("flip", "flip"), ("tp", "d_only")):
            r = self.ed.propose(self._body(form=form))
            self.assertEqual((r["bar"]["route"], r["bar"]["what"], r["bar"]["form"]), ("/api/profil/recompute", "phase_bars", want))
            self.assertEqual(r["bar"]["doc"]["schema"], "flliper.server/1")

    def test_the_request_to_the_child(self):
        self.ed.propose(self._body(goals={"seats": 3, "kv_tokens": 200000}))
        (kind, req, parts), = self.orc.calls
        self.assertEqual(kind, "propose")
        self.assertTrue(req["basis"]["env_path"].endswith("demo.env"))
        self.assertEqual((req["form"], req["goals"]), ("flip", {"seats": 3, "kv_tokens": 200000}))
        self.assertEqual(req["inventory"]["hardware"]["schema"], "flliper.hardware/1")
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
        self.ed.propose(self._body(inventory=[{"card": "rtx5090-32", "pcie": {"gen": 5, "lanes": 16}}, {"card": "rtx3090-24", "pcie": {"gen": 4, "lanes": 16}}]))
        (_k, req, parts), = self.orc.calls
        self.assertEqual([c["entry"]["id"] for c in req["inventory"]["cards"]], ["rtx5090-32", "rtx3090-24"])
        self.assertNotIn("hardware", req["inventory"])
        self.assertEqual(parts["inventory"][0][0], "catalog")
        # the rig's own two cards go with their real UUIDs
        self.orc.calls.clear()
        self.ed.propose(self._body(inventory=RIG[:2]))
        self.assertIn("hardware", self.orc.calls[0][1]["inventory"])
        self.assertEqual(self.orc.calls[0][2]["inventory"][0][0], "nvml")

    def test_user_paths_need_the_model_root_check(self):
        calls = []
        ed = P.ProfilEditor(kartenplaner=K.Kartenplaner(tree=FIXTURE_TREE), release_dir=os.path.join(self.tmp, "rel"), user_dir=os.path.join(self.tmp, "usr"),
                            tree=FIXTURE_TREE, catalog_file=REPO_CATALOG, oracle=self.orc, hardware=lambda: {"ok": True, "profile": _hw_profile(_rows(2))},
                            check_path=lambda p, what: calls.append((p, what)) or "/ok/" + p.strip("/"))
        ed.propose(self._body(model_path="/models/x"))
        self.assertEqual(calls, [("/models/x", "model_path")])
        self.assertEqual(self.orc.calls[-1][1]["model_path"], "/ok/models/x")
        ed.check_path = lambda p, what: (_ for _ in ()).throw(ValueError("path is not under a model root"))
        with self.assertRaises(P.ProfileError):
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
    """The REAL oracle: a child process with this checkout's flliper, the launcher dry run on a replayed inventory (16-18 s a run to the end)."""

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
        self.assertEqual(d["quelle"], "oracle", d["notes"])
        self.assertEqual(d["oracle"]["outcome"], "geht", [(x["code"], x["reason"][:160]) for x in d.get("verdikte", [])])
        self.assertEqual(d["rejections"], [])
        self.assertTrue(d["goes"])
        self.assertEqual(d["oracle"]["profil"]["quelle"], "nf-int4-h6-abl.env")
        self.assertEqual(len(d["oracle"]["profil"]["input_sha256"]), 64)
        self.assertEqual(len(d["oracle"]["profil_text_sha256"]), 64)
        self.assertEqual(d["oracle"]["profile_file_sha256"], doc["meta"]["based_on"]["sha256"])        # plan 4c: the hash of the profile FILE it was loaded from
        self.assertTrue(d["oracle"]["plan"]["pp_cut"])
        # the same codes as the planner gate alone says today (an editor without oracle)
        gate = P.ProfilEditor(kartenplaner=K.Kartenplaner(tree=REPO_PY), release_dir=self.rel, user_dir=os.path.join(self.tmp, "u2"), tree=REPO_PY,
                              catalog_file=REPO_CATALOG).dry_run(doc, RIG)
        self.assertEqual(gate["quelle"], "gate")
        self.assertEqual([q["code"] for q in gate["rejections"]], [q["code"] for q in d["rejections"]])
        # the second equal question is answered from the cache
        t1 = time.time()
        d2 = ed.dry_run(doc, RIG)
        self.assertTrue(d2["oracle"]["cached"])
        self.assertEqual(d2["oracle"]["cache_key"], d["oracle"]["cache_key"])
        self.assertLess(time.time() - t1, 5.0)
        self.assertGreater(t1 - t0, 0.0)

    def test_2_two_cards_name_the_vectors_of_the_profile(self):
        ed = self._ed(self.hw2)
        doc = ed.load("release", "nf-int4-h6-abl")["doc"]
        two = [{"card": "rtx5090-32", "pcie": {"gen": 5, "lanes": 8}}, {"card": "rtx3080-20", "pcie": {"gen": 4, "lanes": 8}}]
        d = ed.dry_run(doc, two)
        self.assertEqual(d["quelle"], "oracle", d["notes"])
        codes = {q["code"] for q in d["rejections"]}
        self.assertIn("HW-COUNT", codes)
        pv = [x for x in d["verdikte"] if x["code"] == "PROFILE-VECTORS"]
        self.assertEqual(len(pv), 1, [(x["code"]) for x in d["verdikte"]])
        self.assertIn("--rank-moe-ratio", pv[0]["values"])
        self.assertIn("FLLIPER_MOE_SCRATCH_SLOTS", pv[0]["values"])
        self.assertEqual(pv[0]["parent"], "HW-COUNT")
        self.assertNotEqual(d["oracle"]["outcome"], "geht")

    def test_3_propose_for_two_cards_is_a_startprofil_with_origin_verdict_and_edges(self):
        ed = self._ed(self.hw2)
        # the release profile of the line: the NF launcher cannot plan 27b-base (measured 07.10. on 2e68b3f94b: SystemExit '--p-chunk-policy dynamic:
        # need 0 < min_tokens <= max_tokens, got 4096/2048'); the 27B model holds 196608 tokens on two cards, the NF default is its own
        body = {"basis": {"kind": "release", "name": "27b-base" if DUAL_LINE else "nf-int4-h6-abl"}, "form": "flip", "inventory": "rig",
                "goals": {"kv_tokens": 196608} if DUAL_LINE else {}}
        r = ed.propose(body)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["schema"], "flliper.propose-d/1")
        self.assertEqual((r["form"], r["n"]), ("flip", 2))
        sp = r["startprofil"]
        self.assertEqual(sp["schema"], "flliper.server/1")
        self.assertEqual(sp["doc"]["schema"], "flliper.server/1")
        self.assertTrue(sp["verifiziert"], sp["probleme"])
        # vectors of N entries; no verdict names a vector
        self.assertTrue(r["proposal"]["vectors_ok"], r["proposal"]["vectors_wrong"])
        self.assertEqual([x for x in r["verdict"]["verdikte"] if x["code"] == "PROFILE-VECTORS"], [])
        # 27B line: the planner's dry run goes with --force; NF line: the same once the launcher re-stages the P-card reference, else it ends with W167
        self.assertEqual(r["verdict"]["outcome"], "ok_with_force" if (DUAL_LINE or RESTAGES_P_CARD) else "verweigert",
                         [(x["code"], x["reason"][:100]) for x in r["verdict"]["verdikte"]])
        self.assertEqual(r["verdict"]["n"], 2)
        self.assertEqual(len(r["verdict"]["profil"]["file_sha256"]), 64)
        self.assertEqual(r["basis"]["sha256"], r["verdict"]["profil"]["file_sha256"])
        # every changed value is in the doc (origin planer) and the row shows what the proposal said
        rows = {x["key"]: x for x in sp["view"]["rows"]}
        changed = [w for w in r["values"] if w["changed"] and w["key"]]
        self.assertTrue(changed)
        for w in changed:
            if w["value"] is None:
                self.assertNotIn(w["key"], rows, w["label"])
                continue
            self.assertIn(w["key"], rows, w["label"])
            self.assertEqual(rows[w["key"]]["value"], str(w["value"]), w["label"])
            self.assertEqual(rows[w["key"]]["origin"], "planer", w["label"])
            self.assertEqual(rows[w["key"]]["proposal"]["state"], w["state"])
        for w in r["values"]:
            for f in ("key", "label", "value", "state", "source", "reason", "verdikte", "kanten"):
                self.assertIn(f, w)
            for v in w["verdikte"]:
                for f in ("code", "forcebar", "force_state", "reason", "consequence"):
                    self.assertIn(f, v)
        self.assertTrue(any(w["kanten"] for w in r["values"]), "the edge catalog gives at least one value its dependencies")
        self.assertEqual(r["bar"]["what"], "phase_bars")
        self.assertEqual(r["bar"]["form"], "flip")
        self.assertEqual(r["bar"]["doc"]["schema"], "flliper.server/1")
        # the second equal question is a cache hit
        t0 = time.time()
        r2 = ed.propose(body)
        self.assertTrue(r2["oracle"]["cached"])
        self.assertEqual(r2["verdict"]["argv_sha256"], r["verdict"]["argv_sha256"])
        self.assertLess(time.time() - t0, 5.0)
        # another goal is another question
        r3 = ed.propose(dict(body, goals={"kv_tokens": 180000}))
        self.assertFalse(r3["oracle"]["cached"])

    def test_4_a_dead_child_is_a_note_and_the_gate_answers(self):
        dead = ORA.OracleService(REPO_PY, python="/nowhere/python")
        ed = P.ProfilEditor(kartenplaner=K.Kartenplaner(tree=REPO_PY), release_dir=self.rel, user_dir=os.path.join(self.tmp, "u3"), tree=REPO_PY,
                            catalog_file=REPO_CATALOG, oracle=dead)
        d = ed.dry_run(ed.load("release", "nf-int4-h6-abl")["doc"], RIG[:2])
        self.assertEqual(d["quelle"], "gate")
        self.assertTrue(any("Oracle (launcher dry run) not available" in n for n in d["notes"]))
        r = ed.propose({"basis": {"kind": "release", "name": "nf-int4-h6-abl"}, "inventory": RIG[:2]})
        self.assertFalse(r["ok"])
        self.assertIn("/nowhere/python", r["error"])


if __name__ == "__main__":
    unittest.main()
