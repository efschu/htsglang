"""Profil-Planer: Herkunft, Abweichung und Glossar aus dem Catalog erreichen die Seite (Nutzerauftrag 05.10., Catalog über zwei Bäume).

Gepinnt:
  * ``render_view`` hängt ``trees`` / ``abweichung`` / ``satz_quelle`` des Katalogeintrags an die Erklärung der Zeile, und NUR wenn der Eintrag sie hat
    (eine Zeile ohne diese Felder bekommt keine leeren Platzhalter: die Seite zeigt dann nichts an).
  * ``view["glossary"]`` ist das Glossar des Katalogs; ein Catalog ohne Glossar ergibt ein leeres Dict (kein Fehler, die Seite zeigt keinen Abschnitt).
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from rigdash import kartenplan as K  # noqa: E402
from rigdash import profil as P  # noqa: E402

FIXTURE_TREE = os.path.join(HERE, "fixtures", "kartenplan", "planner_tree", "python")
REPO_CATALOG = os.path.join(os.path.dirname(HERE), "profil_data", "catalog.json")
ENV = """\
PROFILE_NAME=demo
PROFILE_LINE=nf
PROFILE_STATUS=experimentell
PROFILE_ARGS=(--model /m --p-bs 2 --pp-stage-ratio 29,11,8 --pp-attn-stage-ratio 8,4,4)
"""


def _editor(tmp, extra_entries=None, glossary=None):
    with open(REPO_CATALOG, encoding="utf-8") as fh:
        cat = json.load(fh)
    # der ausgelieferte Katalog ist der Union-Katalog: jede Zeile hat baeume; der Test steuert die Herkunftsfelder selbst
    for e in cat["entries"].values():
        for field in P.ProfilEditor.ORIGIN_FIELDS:
            e.pop(field, None)
    for name, fields in (extra_entries or {}).items():
        cat["entries"][name].update(fields)
    if glossary is None:
        cat.pop("glossary", None)
    else:
        cat["glossary"] = glossary
    cat_path = os.path.join(tmp, "catalog.json")
    with open(cat_path, "w", encoding="utf-8") as fh:
        json.dump(cat, fh)
    rel = os.path.join(tmp, "rel")
    os.makedirs(rel)
    with open(os.path.join(rel, "demo.env"), "w", encoding="utf-8") as fh:
        fh.write(ENV)
    return P.ProfilEditor(kartenplaner=K.Kartenplaner(tree=FIXTURE_TREE), release_dir=rel, user_dir=os.path.join(tmp, "u"),
                          tree=FIXTURE_TREE, catalog_file=cat_path)


class Herkunft(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="herkunft1005_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_fields_reach_the_row_only_where_the_catalog_has_them(self):
        ed = _editor(self.tmp, {"--p-bs": {"trees": ["nf"], "satz_quelle": "NF-Sitz, Bericht 1504",
                                           "abweichung": {"27b": {"default": "1", "help": "a"}, "nf": {"default": "6", "help": "b"}}}},
                     glossary={"Park": "Anfrage wird angehalten"})
        rows = {r["key"]: r for r in ed.load("release", "demo")["view"]["rows"]}
        ex = rows["flag:--p-bs"]["explain"]
        self.assertEqual((ex["trees"], ex["satz_quelle"]), (["nf"], "NF-Sitz, Bericht 1504"))
        self.assertEqual(sorted(ex["abweichung"]), ["27b", "nf"])
        for key in ("flag:--pp-stage-ratio", "flag:--model"):
            for field in P.ProfilEditor.ORIGIN_FIELDS:
                self.assertNotIn(field, rows[key]["explain"], "%s hat %s ohne Katalogfeld" % (key, field))

    def test_glossary_is_passed_on_and_may_be_absent(self):
        with_g = _editor(self.tmp, glossary={"D": "die Decode-Karte"}).load("release", "demo")["view"]
        self.assertEqual(with_g["glossary"], {"D": "die Decode-Karte"})
        tmp2 = tempfile.mkdtemp(prefix="herkunft1005b_")
        self.addCleanup(shutil.rmtree, tmp2, True)
        without = _editor(tmp2).load("release", "demo")["view"]
        self.assertEqual(without["glossary"], {})


if __name__ == "__main__":
    unittest.main()
