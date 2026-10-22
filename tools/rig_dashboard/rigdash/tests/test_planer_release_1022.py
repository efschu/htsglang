"""PLANER-RELEASE 1022: the planner page for 2 / 3 / 4 identical cards (RTX 3090) with the three models and the three layouts, on REAL answers.

The fixtures under ``fixtures/planer_release_1022/`` are the unchanged answers of a local rigdash server (release edition, from this tree, port 18893) to
``planer-layouts`` and ``planer-plan`` for every cell of: N in 2, 3, 4  x  {27B INT8, 27B NVFP4, NF INT4}  x  {solo, flip, dual}.  Every ``plan_*`` file is a
launcher dry run through the oracle child (``intern.oracle``), none is a placeholder; a cell the page does not offer (``layouts_*``: ``possible`` false) carries its
reason and its calculation.  Only the start profile document (``intern.startprofil.doc``, 100 kB of .env rows) was cut out of the records.

Gepinnt:
  * every cell is a real dry run or a named reason (a disabled button with its calculation);
  * 3 x RTX 3090 with 27B gets a sensible proposal for every layout the model has (INT8: solo + flip, NVFP4: solo + flip + dual), no refusal;
  * 2 and 4 cards are calculated too (``SUPPORTED_N``) and say that no release boot proved them: layout chip ``dry_run``, the blocker METAL-UNPROVEN in the launcher
    notes; 3 cards say neither; 1 card and 5..8 cards are "not supported yet" with their reason;
  * the ceiling per seat that the cards refuse outright (27B INT8 on 2 x 24 GB, W40) is answered by the largest preset the launcher takes, and the page says so;
  * the German view: not one English sentence in any server text of any cell (the check is run over every recorded answer; its detector is itself tested);
  * the page shows all of that (node, pure render functions on the recorded answers).
"""

import copy
import json
import os
import re
import shutil
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import planer_ziel as Z  # noqa: E402

STATIC = os.path.join(os.path.dirname(HERE), "static")
FIX = os.path.join(HERE, "fixtures", "planer_release_1022")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RELEASE_DIR = os.path.join(REPO_ROOT, "docker", "flliper", "profiles_release")
NODE = shutil.which("node") or ("/opt/node-v22.14.0-linux-x64/bin/node" if os.path.exists("/opt/node-v22.14.0-linux-x64/bin/node") else None)

NS = (2, 3, 4, 5, 6, 7)
NS_FULL = (2, 3, 4)                                   # every possible layout of these counts has a recorded dry run; 5..7: INT8 solo + flip, NVFP4 flip + dual, NF solo + flip
MODELS = ("qwen38-27b-int8", "qwen38-27b-nvfp4", "qwen38-nf-int4")
LAYOUTS = ("solo", "flip", "dual")


def rec(name):
    with open(os.path.join(FIX, name), encoding="utf-8") as fh:
        return json.load(fh)


def has(name):
    return os.path.exists(os.path.join(FIX, name))


def lay(n, m):
    return {l["id"]: l for l in rec("layouts_%d_%s.json" % (n, m))["layouts"]}


def plan(n, m, l):
    return rec("plan_%d_%s_%s.json" % (n, m, l))


def cells():
    """(n, model, layout, layout row) of every OFFERED layout that has a recorded dry run."""
    for n in NS:
        for m in MODELS:
            for l, row in lay(n, m).items():
                if row["possible"] and has("plan_%d_%s_%s.json" % (n, m, l)):
                    yield n, m, l, row


# ------------------------------------------------------------------------------------------------------------------ the German detector
SKIP_KEYS = {"id", "key", "roh", "code", "schema", "intern", "form", "outcome", "status", "gruppe", "was", "proof"}
#: English function words that German does not use (``in``, ``per``, ``an``, ``so``, ``will`` are German words too and not in the list)
ENGLISH = set("the of and for with not is are at by this that but has have can from only when which its be if than into after before each one more then there these "
              "those it as or to up on does do".split())


def strings(o):
    """Every visible server string of an answer (structural keys left out)."""
    if isinstance(o, str):
        yield o
    elif isinstance(o, list):
        for x in o:
            yield from strings(x)
    elif isinstance(o, dict):
        for k, v in o.items():
            if k not in SKIP_KEYS:
                yield from strings(v)


def english_words(s):
    t = re.sub(r"--[a-z0-9-]+", "", s)                                  # command-line flags
    t = re.sub(r"\b[A-Za-z]*[0-9_][A-Za-z0-9_.\-]*\b", "", t)            # identifiers and figures with digits / underscores
    return [w for w in re.findall(r"[A-Za-z]{2,}", t) if w.lower() in ENGLISH]


class Cells(unittest.TestCase):
    def test_the_fixtures_cover_every_cell(self):
        for n in NS:
            for m in MODELS:
                self.assertTrue(has("layouts_%d_%s.json" % (n, m)), (n, m))
                for l, row in lay(n, m).items():
                    if row["possible"] and n in NS_FULL:
                        self.assertTrue(has("plan_%d_%s_%s.json" % (n, m, l)), (n, m, l))
        for n in (5, 6, 7):
            for m, l in (("qwen38-27b-int8", "solo"), ("qwen38-27b-int8", "flip"), ("qwen38-27b-nvfp4", "flip"), ("qwen38-27b-nvfp4", "dual"),
                         ("qwen38-nf-int4", "solo"), ("qwen38-nf-int4", "flip")):
                self.assertTrue(lay(n, m)[l]["possible"], (n, m, l))
                self.assertTrue(has("plan_%d_%s_%s.json" % (n, m, l)), (n, m, l))

    def test_every_cell_is_a_real_dry_run_or_a_named_reason(self):
        real = 0
        for n in NS:
            for m in MODELS:
                self.assertEqual(set(lay(n, m)), set(LAYOUTS))
                for l, row in lay(n, m).items():
                    if not row["possible"]:
                        self.assertTrue(row["why"], (n, m, l))              # a disabled button names its reason ...
                        self.assertTrue(row["calc"], (n, m, l))             # ... and shows its calculation
        for n, m, l, row in cells():
            p = plan(n, m, l)
            self.assertEqual(p["status"], "ok", (n, m, l))
            self.assertEqual(p["n"], n)
            self.assertIn("duration_s", p["intern"]["oracle"], (n, m, l))      # the oracle child (launcher dry run) answered
            self.assertTrue(p["verdikt"]["outcome"], (n, m, l))
            self.assertTrue(p["felder"], (n, m, l))
            real += 1
        self.assertEqual(real, 20 + 6 * 3)                                      # 6 + 7 + 7 cells of 2..4 cards and 6 each of 5, 6, 7 cards (nfblock 1022: NF solo + flip on every count)

    def test_three_cards_27b_gets_a_sensible_proposal_for_every_layout_the_model_has(self):
        for m, want in (("qwen38-27b-int8", ("solo", "flip")), ("qwen38-27b-nvfp4", ("solo", "flip", "dual"))):
            row = lay(3, m)
            for l in want:
                self.assertTrue(row[l]["possible"], (m, l))
                p = plan(3, m, l)
                self.assertEqual(p["status"], "ok")
                self.assertIn(p["verdikt"]["outcome"], ("geht", "ok_with_force"), (m, l))      # the launcher takes it (with force: the numbers are borrowed)
                self.assertNotIn("anpassung", p)                                                 # the default ceiling per seat fits
                rows = {r["key"]: r for g in p["felder"] for r in g["zeilen"]}
                self.assertEqual(int(rows["tokens"]["wert"]), Z.DEFAULT_TOKENS)
                self.assertEqual(len(rows["d_mem"]["je_karte"]), 3)
                if l != "solo":
                    self.assertEqual(len(rows["p_mem"]["je_karte"]), 3)
                    self.assertEqual(sum(int(x) for x in rows["p_layers"]["je_karte"]), 64) if "p_layers" in rows else None
                    self.assertGreaterEqual(int(rows["p_pool"]["wert"]), Z.DEFAULT_TOKENS)       # the pool holds at least one request at the ceiling
                self.assertIn("d_pool", rows)

    def test_the_card_count_decides_the_per_card_columns(self):
        for n, m, l, row in cells():
            if l == "solo":
                continue
            rows = {r["key"]: r for g in plan(n, m, l)["felder"] for r in g["zeilen"]}
            self.assertEqual(len(rows["p_mem"]["je_karte"]), n, (n, m, l))
            self.assertEqual(len(rows["d_mem"]["je_karte"]), n, (n, m, l))

    def test_unproven_counts_say_so_and_the_proven_one_does_not(self):
        for n, m, l, row in cells():
            codes = [i["code"] for i in plan(n, m, l)["verdikt"]["items"]]
            if n == 3:
                self.assertEqual(row["proof"], "metal")
                self.assertNotIn("METAL-UNPROVEN", codes, (n, m, l))
                self.assertNotIn("HW-COUNT", codes, (n, m, l))
            else:
                self.assertEqual(row["proof"], "dry_run")
                self.assertIn("METAL-UNPROVEN", " ".join(row["why"]))
                self.assertIn("METAL-UNPROVEN", codes, (n, m, l))
                self.assertIn("HW-COUNT", codes, (n, m, l))

    def test_nothing_is_offered_that_the_launcher_refuses_outright(self):
        # a layout that the page offers either passes (with force) or the page adapted the ceiling; a plain refusal of an offered layout is a defect of the page
        for n, m, l, row in cells():
            p = plan(n, m, l)
            self.assertNotIn(p["verdikt"]["outcome"], ("verweigert", "crash"), (n, m, l, [i["code"] for i in p["verdikt"]["items"] if i["blocked"]]))

    def test_27b_int8_on_two_cards_gets_the_largest_ceiling_the_launcher_takes(self):
        for l in ("solo", "flip"):
            p = plan(2, "qwen38-27b-int8", l)
            a = p["anpassung"]
            self.assertEqual((a["gefragt"], a["was"]), (Z.DEFAULT_TOKENS, "tokens"))
            self.assertTrue(a["erreicht"] and a["erreicht"] < a["gefragt"])
            rows = {r["key"]: r for g in p["felder"] for r in g["zeilen"]}
            self.assertEqual(int(rows["tokens"]["wert"]), a["erreicht"])                         # the figures are for the lowered ceiling ...
            self.assertIn(str(a["gefragt"]), a["text"])                                           # ... and the text says what was asked
            self.assertIn("W40", a["text"])

    def test_one_card_and_more_than_four_are_not_supported_with_their_reason(self):
        z = Z.ZielPlaner(None)
        for n in (1, 8):
            for m in MODELS:
                for l in z.layouts({"model": m, "cards": [{"card": "rtx3090-24", "count": n}]})["layouts"]:
                    self.assertIn(l["status"], ("n_unsupported", "blocked"), (n, m, l["id"]))             # (INT8 dual has no profile at all: blocked, with the count reason first)
                    self.assertFalse(l["possible"])
                    why = " ".join(l["why"])
                    self.assertIn("not supported yet", why)
                    self.assertIn("%d card" % n, why)

    def test_the_supported_counts_are_the_ones_ncards_proves_with_a_golden(self):
        import glob
        found = glob.glob(os.path.join(REPO_ROOT, "test", "registered", "unit", "*", "fixtures", "ncards_1021", "golden"))
        if not found:
            self.skipTest("NCARDS goldens not in this tree")
        for n in NS:
            for prof in ("27b", "27b-nvfp4", "27b-nvfp4-dual", "nf"):
                self.assertTrue(os.path.exists(os.path.join(found[0], "dryrun_%s-%dx3090.json" % (prof, n))), (prof, n))


# ------------------------------------------------------------------------------------------------------------------ the German view
class German(unittest.TestCase):
    def test_the_detector_finds_english_and_lets_german_through(self):
        self.assertTrue(english_words("the launcher refuses this combination"))
        self.assertTrue(english_words("Der Planer ist for 2 Karten geprüft"))
        self.assertFalse(english_words("Der Launcher verweigert diese Kombination (siehe Gründe)."))
        self.assertFalse(english_words("--max-kv-per-request: eine Decke je Request, D_AWAKE_REST_MIB, RTX3090/24576MiB/sm86 und 262144 Token"))

    def test_no_english_sentence_in_the_german_view_of_any_recorded_answer(self):
        checked, found = 0, []
        docs = [("layouts", n, m, rec("layouts_%d_%s.json" % (n, m))) for n in NS for m in MODELS] + [("plan", n, m, plan(n, m, l)) for n, m, l, _row in cells()]
        for _kind, n, m, doc in docs:
            for s in strings(Z.localize(copy.deepcopy(doc), "de")):
                checked += 1
                if english_words(s):
                    found.append((n, m, s[:160]))
        self.assertGreater(checked, 1500)                                                        # the check really walked the answers
        self.assertEqual(found, [], "\n".join(str(x) for x in found[:10]))

    def test_the_english_view_is_left_alone(self):
        doc = rec("plan_3_qwen38-27b-int8_flip.json")
        self.assertEqual(Z.localize(copy.deepcopy(doc), None), doc)
        self.assertEqual(Z.localize(copy.deepcopy(doc), "en"), doc)

    def test_structural_keys_keep_their_values(self):
        doc = rec("plan_2_qwen38-27b-int8_flip.json")
        de = Z.localize(copy.deepcopy(doc), "de")
        self.assertEqual((de["anpassung"]["was"], de["layout_info"]["proof"], de["status"], de["verdikt"]["outcome"]), ("tokens", "dry_run", "ok", doc["verdikt"]["outcome"]))
        self.assertEqual([i["code"] for i in de["verdikt"]["items"]], [i["code"] for i in doc["verdikt"]["items"]])

    def test_the_composed_derivation_sentences_are_translated_part_by_part(self):
        # review B2: the P-pool line, the X line and the expert line were English in the German view
        p = Z.localize(plan(3, "qwen38-27b-int8", "flip"), "de")
        rows = {r["key"]: r for g in p["felder"] for r in g["zeilen"]}
        self.assertIn("des Launcher-Laufs", rows["p_pool"]["warum"])
        self.assertIn("Untergrenze", rows["p_pool"]["warum"])
        self.assertIn("Obergrenze", rows["x"]["warum"])
        nf = Z.localize(plan(3, "qwen38-nf-int4", "flip"), "de")
        text = " ".join(r["warum"] for g in nf["felder"] for r in g["zeilen"])
        self.assertIn("residente Experten", text)
        self.assertEqual(english_words(text), [])

    def test_the_launcher_verdict_items_are_german(self):
        p = Z.localize(plan(2, "qwen38-27b-nvfp4", "flip"), "de")
        text = " ".join(i["text"] for i in p["verdikt"]["items"])
        self.assertIn("Karten wären", text)
        self.assertIn("noch kein Release-Boot", text)
        self.assertIn("Das Profil", text)
        self.assertEqual(english_words(text), [])

    def test_the_n_states_and_the_adaptation_are_german(self):
        z = Z.ZielPlaner(None)
        for n, frag in ((1, "Karte"), (8, "Karten")):
            lays = z.layouts({"model": "qwen38-27b-nvfp4", "cards": [{"card": "rtx3090-24", "count": n}], "lang": "de"})["layouts"]
            why = " ".join(lays[0]["why"])
            self.assertIn("noch nicht unterstützt", why)
            self.assertIn(frag, why)
            self.assertEqual(english_words(why), [], why)
        why = " ".join(z.layouts({"model": "qwen38-27b-nvfp4", "cards": [{"card": "rtx3090-24", "count": 2}], "lang": "de"})["layouts"][1]["why"])
        self.assertIn("METAL-UNPROVEN", why)
        self.assertEqual(english_words(why), [], why)
        a = Z.localize(plan(2, "qwen38-27b-int8", "flip"), "de")["anpassung"]["text"]
        self.assertIn("Token pro Sitz", a)
        self.assertEqual(english_words(a), [], a)

    def test_an_unknown_text_stays_english_and_is_never_blank(self):
        self.assertEqual(Z.localize({"warum": "a sentence nobody translated"}, "de"), {"warum": "a sentence nobody translated"})


# ------------------------------------------------------------------------------------------------------------------ the card-count states and the ceiling
def res_refused_at(tokens_from):
    """A canned propose answer of the 27B INT8 flip on 3 x 3090 in which the launcher refuses with W40 at ``tokens_from`` tokens and above."""
    with open(os.path.join(HERE, "fixtures", "planer_ziel_1021", "res_27b_int8_flip_3090.json"), encoding="utf-8") as fh:
        ok = json.load(fh)
    bad = copy.deepcopy(ok)
    bad["verdict"]["outcome"] = "verweigert"
    bad["verdict"]["verdikte"] = [{"level": "run", "code": "PP-CUT", "forcebar": True, "force_state": "is_blocked", "reason": "W40 PPCutRefused: no SERVABLE cut"}]
    return ok, bad


class StubEd:
    _hw_cards_match = staticmethod(lambda hw, cards: False)

    def __init__(self, ok, bad, limit):
        self.ok, self.bad, self.limit, self.asked = ok, bad, limit, []

    def _hardware_profile(self):
        return None

    def propose(self, req):
        t = (req.get("goals") or {}).get("kv_tokens") or Z.DEFAULT_TOKENS
        self.asked.append(t)
        return copy.deepcopy(self.bad if t > self.limit else self.ok)


X3090 = [{"card": "rtx3090-24", "count": 3}]


class Ceiling(unittest.TestCase):
    def test_a_refused_ceiling_is_retried_at_the_lower_presets_and_the_answer_says_so(self):
        ok, bad = res_refused_at(0)
        ed = StubEd(ok, bad, 100000)
        z = Z.ZielPlaner(ed)
        out = z.plan({"model": "qwen38-27b-int8", "layout": "flip", "cards": X3090})
        self.assertEqual(ed.asked, [262144, 131072, 65536])                                    # largest preset first, stops at the first the launcher takes
        self.assertEqual((out["anpassung"]["gefragt"], out["anpassung"]["erreicht"]), (262144, 65536))
        self.assertNotIn(out["verdikt"]["outcome"], ("verweigert", "crash"))

    def test_a_ceiling_the_cards_take_is_not_touched(self):
        ok, bad = res_refused_at(0)
        ed = StubEd(ok, bad, 262144)
        out = Z.ZielPlaner(ed).plan({"model": "qwen38-27b-int8", "layout": "flip", "cards": X3090})
        self.assertEqual(ed.asked, [262144])
        self.assertNotIn("anpassung", out)

    def test_nothing_that_fits_is_a_refusal_with_its_reason_not_an_invention(self):
        ok, bad = res_refused_at(0)
        ed = StubEd(ok, bad, 1000)
        out = Z.ZielPlaner(ed).plan({"model": "qwen38-27b-int8", "layout": "flip", "cards": X3090})
        self.assertEqual(ed.asked, [262144, 131072, 65536, 32768])
        self.assertIsNone(out["anpassung"]["erreicht"])
        self.assertEqual(out["verdikt"]["outcome"], "verweigert")

    def test_a_ceiling_the_user_set_below_the_presets_is_never_raised(self):
        ok, bad = res_refused_at(0)
        ed = StubEd(ok, bad, 1000)
        Z.ZielPlaner(ed).plan({"model": "qwen38-27b-int8", "layout": "flip", "cards": X3090, "tokens": 40000})
        self.assertEqual(ed.asked, [40000, 32768])

    def test_the_card_count_states_follow_the_topology_gate(self):
        class Kp:
            @staticmethod
            def _mods():
                return None, object()

        class Ed(StubEd):
            kp = Kp()

            def _topology_verdict(self, n, tp, notes):
                if n == 3:
                    return None
                if n == 4:
                    return "HW-COUNT: 4 cards would be P = TP1 x PP4 ...; not yet runnable, 2 blocker(s): [BAR1-WINDOWS] x; [METAL-UNPROVEN] y"
                return "HW-COUNT: %d cards would be ...; not yet runnable, 1 blocker(s): [METAL-UNPROVEN] no release boot" % n

        z = Z.ZielPlaner(Ed(*res_refused_at(0), 10 ** 9))
        st = z.n_states()
        self.assertEqual({n: st[n]["state"] for n in range(1, 9)}, {1: "off", 2: "unproven", 3: "proven", 4: "refused", **{n: ("unproven" if n in Z.SUPPORTED_N else "off") for n in range(5, 9)}})
        self.assertEqual(z.supported_n()[:2], (2, 3))
        why = " ".join(next(l for l in z.layouts({"model": "qwen38-27b-int8", "cards": [{"card": "rtx3090-24", "count": 4}]})["layouts"] if l["id"] == "flip")["why"])
        self.assertIn("BAR1-WINDOWS", why)                                                      # the gate's own words, not a guess

    def test_without_a_gate_the_constants_decide(self):
        st = Z.ZielPlaner(None).n_states()
        self.assertEqual({n: st[n]["state"] for n in range(1, 9)}, {1: "off", 2: "unproven", 3: "proven", 4: "unproven", **{n: ("unproven" if n in Z.SUPPORTED_N else "off") for n in range(5, 9)}})


class ReleaseBlock(unittest.TestCase):
    """A release profile for N identical RTX 3090 that says ``PROFILE_STATUS=blockiert-<code>`` is a disabled button with that reason: the page does not offer what its
    own profile says cannot start.  Since nfblock 1022 no NF profile is blocked (the launcher's D-EIGENTUM decides a cut form on 2 / 3 cards and keeps Form A on 4..7,
    where it refused D-BUDGET before); the mechanism is pinned on a copy of the profiles with one status set back to blocked."""

    def z(self, release_dir=RELEASE_DIR):
        ed = StubEd(*res_refused_at(0), 10 ** 9)
        ed.release_dir = release_dir
        return Z.ZielPlaner(ed)

    def layouts(self, model, card, n, release_dir=RELEASE_DIR, lang=None):
        body = {"model": model, "cards": [{"card": card, "count": n}]}
        if lang:
            body["lang"] = lang
        return {l["id"]: l for l in self.z(release_dir).layouts(body)["layouts"]}

    def blocked_copy(self, n):
        import tempfile

        d = tempfile.mkdtemp(prefix="nfblock-release-")
        self.addCleanup(shutil.rmtree, d, True)
        for name in os.listdir(RELEASE_DIR):
            if name.endswith(".env"):
                shutil.copy(os.path.join(RELEASE_DIR, name), d)
        path = os.path.join(d, "nf-%dx3090.env" % n)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(re.sub(r"^PROFILE_STATUS=\S+", "PROFILE_STATUS=blockiert-d-budget", text, flags=re.M))
        return d

    def test_no_nf_profile_is_blocked_any_more(self):
        for n in NS:
            with open(os.path.join(RELEASE_DIR, "nf-%dx3090.env" % n), encoding="utf-8") as fh:
                text = fh.read()
            self.assertIsNone(re.search(r"^PROFILE_STATUS=blockiert-", text, re.M), n)
            self.assertRegex(text, r"(?m)^PROFILE_STATUS=hochrechnung\b")

    def test_nf_and_the_27b_models_on_any_count_are_not_blocked_by_a_profile(self):
        for n in NS:
            for m in ("qwen38-nf-int4", "qwen38-27b-int8", "qwen38-27b-nvfp4"):
                for l in ("solo", "flip"):
                    self.assertNotIn("marked blocked", " ".join(self.layouts(m, "rtx3090-24", n)[l]["why"]), (n, m, l))
            for l in ("solo", "flip"):
                self.assertTrue(self.layouts("qwen38-nf-int4", "rtx3090-24", n)[l]["possible"], (n, l))

    def test_a_blocked_profile_is_a_disabled_button_with_the_profiles_reason(self):
        d = self.blocked_copy(4)
        for l in ("solo", "flip"):
            row = self.layouts("qwen38-nf-int4", "rtx3090-24", 4, d)[l]
            self.assertEqual((row["status"], row["possible"]), ("blocked", False), l)
            why = " ".join(row["why"])
            self.assertIn("marked blocked (D-BUDGET)", why)
            self.assertIn("D-EIGENTUM", why)
        self.assertTrue(self.layouts("qwen38-nf-int4", "rtx3090-24", 5, d)["flip"]["possible"])   # only the profile that says so

    def test_other_cards_are_not_judged_by_a_profile_for_3090s(self):
        d = self.blocked_copy(2)
        for l in ("solo", "flip"):
            self.assertNotIn("marked blocked", " ".join(self.layouts("qwen38-nf-int4", "rtx3080-20", 2, d)[l]["why"]))

    def test_the_reason_is_german_in_the_german_view(self):
        d = self.blocked_copy(4)
        why = " ".join(self.layouts("qwen38-nf-int4", "rtx3090-24", 4, d, lang="de")["flip"]["why"])
        self.assertIn("als gesperrt markiert (D-BUDGET)", why)
        self.assertEqual(english_words(why), [], why)


# ------------------------------------------------------------------------------------------------------------------ the page (node, pure render functions)
HARNESS = r"""
const STATIC = process.argv[1], CASE = JSON.parse(process.argv[2]);
if (CASE.lang === "de") global.RigI18n = { lang: () => "de", loc: () => "de-DE" };
const mod = require(STATIC + "/planer_ziel.js");
const out = {};
out.layouts = CASE.layouts ? mod.renderLayouts(CASE.layouts, "") : "";
out.plan = CASE.plan ? mod.renderPlan(CASE.plan) : "";
console.log(JSON.stringify(out));
"""


def render(**case):
    p = subprocess.run([NODE, "-e", HARNESS, STATIC, json.dumps(case)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


@unittest.skipUnless(NODE, "node missing")
class Page(unittest.TestCase):
    def test_two_and_four_cards_show_the_dry_run_chip_three_cards_do_not(self):
        for n, chip in ((2, True), (3, False), (4, True)):
            h = render(layouts=rec("layouts_%d_qwen38-27b-nvfp4.json" % n))["layouts"]
            self.assertEqual("possible, not booted on metal yet" in h, chip, n)
            self.assertIn("pz-lay", h)
        h = render(layouts=rec("layouts_2_qwen38-27b-nvfp4.json"))["layouts"]
        self.assertIn("METAL-UNPROVEN", h)                                                       # the reason is visible, not hidden in a tooltip

    def test_a_layout_the_cards_cannot_hold_is_a_disabled_button_with_its_calculation(self):
        h = render(layouts=rec("layouts_2_qwen38-27b-nvfp4.json"))["layouts"]
        i = h.index('data-id="dual"')
        self.assertIn("disabled", h[h.rindex("<button", 0, i):i + 80])
        self.assertIn("does not fit by the rough calculation", h)
        self.assertIn("needed ", h)

    def test_the_plan_of_four_cards_shows_four_columns_and_the_chip(self):
        h = render(plan=plan(4, "qwen38-27b-nvfp4", "flip"))["plan"]
        self.assertEqual(h.count('<span class="pz-pc">') >= 3, True)
        self.assertGreaterEqual(h.count("RTX 3090"), 4 * 3)                                      # four cards in each per-card row
        self.assertIn("possible, not booted on metal yet", h)
        self.assertIn("METAL-UNPROVEN", h)
        self.assertNotIn("internal-doc", h)

    def test_the_adaptation_of_the_ceiling_is_the_first_thing_under_the_heading(self):
        p = plan(2, "qwen38-27b-int8", "flip")
        h = render(plan=p)["plan"]
        self.assertLess(h.index("do not fit these cards"), h.index("dry run passes only with force"))
        self.assertIn("W40", h)

    def test_the_german_server_texts_reach_the_page_unchanged(self):
        p = Z.localize(plan(2, "qwen38-27b-int8", "flip"), "de")
        h = render(plan=p, lang="de")["plan"]
        self.assertIn("Token pro Sitz passen nicht auf diese Karten", h)
        self.assertIn('data-i18n="skip"', h)                                                      # the translation memory must not touch them again


# the wiring, on the recorded answers: the page asks the server (replayed from the fixtures) exactly as in the browser
WIRING = r"""
const fs = require("fs"), path = require("path");
const STATIC = process.argv[1], FIX = process.argv[2], CASE = JSON.parse(process.argv[3]);
const host = { innerHTML: "", _h: {}, addEventListener(t, f) { this._h[t] = f; }, querySelector() { return null; } };
global.window = global;
global.document = { getElementById(id) { return id === "pz-root" ? host : id === "tab-profil" ? { hidden: false } : null; }, activeElement: null, createElement() { return { click() {} }; } };
global.setTimeout = (f) => 0;
const calls = [];
const rec = (name) => JSON.parse(fs.readFileSync(path.join(FIX, name), "utf8"));
const CAT = { ok: true, schema: "flliper.planer-ziel/1", supported_n: [2, 3, 4], layouts: [], machine: null, defaults: { seats: 6, tokens: 262144 },
  goals: { seats: [1, 256], tokens: [1024, 8388608], token_presets: [32768, 65536, 131072, 262144] },
  models: [{ id: "qwen38-27b-nvfp4", name: "Qwen3.8-27B NVFP4", format: "nvfp4", layouts: ["solo", "flip", "dual"], weights_mib: 1 },
           { id: "qwen38-27b-int8", name: "Qwen3.8-27B INT8", format: "int8", layouts: ["solo", "flip"], weights_mib: 1 }],
  cards: [{ id: "rtx3090-24", label: "RTX 3090 24 GB", arch: "sm86", mib: 24576, preset: true, origin: "Datasheet", origin_label: "datasheet" }],
  shortcuts: [{ id: "3x3090", label: "3 x RTX 3090", cards: [{ card: "rtx3090-24", count: 3 }] }] };
global.fetch = async (url, opt) => {
  const u = String(url).replace(/^api\/profil\//, "").replace(/\?.*$/, ""), body = opt && opt.body ? JSON.parse(opt.body) : null;
  calls.push([u, body]);
  let j;
  if (u === "planer-katalog") j = CAT;
  else if (u === "planer-layouts") j = rec("layouts_" + body.cards.reduce((s, c) => s + c.count, 0) + "_" + body.model + ".json");
  else if (u === "planer-plan") j = rec("plan_" + body.cards.reduce((s, c) => s + c.count, 0) + "_" + body.model + "_" + body.layout + ".json");
  else j = { ok: false, error: "unknown " + u };
  return { ok: j.ok !== false, status: 200, text: async () => JSON.stringify(j) };
};
require(STATIC + "/planer_ziel.js");
const tick = () => new Promise((r) => setImmediate(r));
async function settle() { for (let i = 0; i < 10; i++) await tick(); }
const click = (attrs) => { const t = { dataset: attrs, disabled: false, closest() { return t; } }; host._h.click({ target: t }); };
const change = (k, v) => host._h.change({ target: { dataset: { k }, value: String(v) } });
(async () => {
  const out = { steps: {} };
  try {
    await settle();
    for (const step of CASE.script) {
      if (step.click) click(step.click);
      if (step.change) change(step.change[0], step.change[1]);
      await settle();
      out.steps[step.name] = host.innerHTML;
    }
    out.calls = JSON.parse(JSON.stringify(calls));
  } catch (e) { out.error = String(e && e.stack || e); }
  console.log(JSON.stringify(out));
})();
"""


def wire(script):
    p = subprocess.run([NODE, "-e", WIRING, STATIC, FIX, json.dumps({"script": script})], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[-2000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert "error" not in out, out["error"]
    return out


@unittest.skipUnless(NODE, "node missing")
class Wiring(unittest.TestCase):
    def test_one_less_card_gives_two_cards_and_a_flip_plan_for_two(self):
        o = wire([{"click": {"pz": "dec", "id": "rtx3090-24"}, "name": "two"}, {"click": {"pz": "layout", "id": "flip"}, "name": "plan"}])
        self.assertEqual([c[1]["cards"] for c in o["calls"] if c[0] == "planer-layouts"][-1], [{"card": "rtx3090-24", "count": 2}])
        self.assertIn("possible, not booted on metal yet", o["steps"]["two"])
        self.assertNotIn("not supported yet", o["steps"]["two"])
        h = o["steps"]["plan"]
        self.assertEqual([c[1]["layout"] for c in o["calls"] if c[0] == "planer-plan"], ["flip"])
        self.assertIn("2 cards", h)
        self.assertIn("Prefill split", h)
        self.assertIn("METAL-UNPROVEN", h)

    def test_one_more_card_gives_four_cards_and_a_dual_plan_for_four(self):
        o = wire([{"click": {"pz": "inc", "id": "rtx3090-24"}, "name": "four"}, {"click": {"pz": "layout", "id": "dual"}, "name": "plan"}])
        self.assertEqual([c[1]["cards"] for c in o["calls"] if c[0] == "planer-layouts"][-1], [{"card": "rtx3090-24", "count": 4}])
        h = o["steps"]["plan"]
        self.assertIn("4 cards", h)
        self.assertGreaterEqual(h.count("RTX 3090"), 4)
        self.assertEqual([c[1]["layout"] for c in o["calls"] if c[0] == "planer-plan"], ["dual"])

    def test_one_card_is_greyed_with_the_reason(self):
        for clicks, frag in (([{"pz": "dec", "id": "rtx3090-24"}] * 2, "1 card"),):
            o = wire([{"click": c, "name": "s%d" % i} for i, c in enumerate(clicks)])
            h = list(o["steps"].values())[-1]
            self.assertIn("not supported yet", h)
            self.assertIn(frag, h)
            self.assertEqual(len(re.findall(r'data-pz="layout"[^>]*disabled', h)), 3)                  # all three layout buttons are disabled
            self.assertEqual([c for c in o["calls"] if c[0] == "planer-plan"], [])                    # and no dry run is asked for


if __name__ == "__main__":
    unittest.main()
