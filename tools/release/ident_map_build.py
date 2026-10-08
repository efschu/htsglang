#!/usr/bin/env python3
"""ident_map_build.py -- identifier/key table proposal from ident_scan.py reports (F0-A, 07.10.2026; writes only the two output files).

usage: ident_map_build.py --scan scan_27b.json [--scan scan_nf.json ...] --out data/ident_map_1007.json --review data/ident_map_1007.review.json

Input: the reports of ident_scan.py (German words in machine positions of the modules added since 03.10.) of the release heads.
Output:
  --out     {old: new, ...} in the shape ident_fix.py takes (keys starting with `_` are ignored by every kit reader). Contains only
            entries that are (a) the planner seat's decision of 07.10. (PLANNER), or (b) built from the word lists alone: every German part of
            the word is in de_en_subwords.json or in GLOSS (the dashboard/catalog glossaries of desk/planer-english*) AND the word is not so
            common in the tree (tree_files <= COMMON_MAX) that a whole-word rewrite would also hit German prose in docstrings/log strings.
  --review  everything that is NOT in the table, with the reason: needs_translation (a German part nobody mapped), too_common,
            conflicts_with_tables (an entry of merged_0928.json / identfix_map.json / decided_0928.json with another target),
            dup_target (two old words, one new word), false_positive (lexicon artefacts, listed by hand), plus per-word spread.
Nothing here is applied: ident_precheck.py judges the table against a ref, release_rename.sh takes it via FIXMAP_EXTRA.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import english_audit as EA  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON_MAX = 30

#: ENTSCHEID des Planer-Sitzes 07.10. (task text of F0-A): JSON keys of flliper.propose-a/1, planer-ui/1, balken/1, verdikt/1 -> English.
#: `in_argv` and `blocker` stay (already English / API words).
PLANNER = {
    "werte": "values", "herkunft": "source", "verdikt": "verdict", "zustand": "state", "ausgang": "outcome", "grund": "reason",
    "formen": "forms", "abschnitte": "sections", "ziele": "goals", "je_rang": "per_rank", "vektoren": "vectors", "geaendert": "changed",
    "unbelegt": "unverified", "hinweise": "notes", "vektorlaengen": "vector_lengths", "eintraege": "entries", "wert": "value",
    "gleich_wie_profil": "same_as_profile", "kuratiert": "curated", "erklaert": "explained", "geerntet": "harvested",
    "unerklaert": "unexplained", "tauscht": "trades", "braucht": "requires", "schliesst_aus": "excludes",
    "abgeleitet_von": "derived_from", "skaliert_mit": "scales_with",
}

#: words from the glossaries of the dashboard translation (rigdash/GLOSSARY_EN.md, weg2/GLOSSARY_EN.md) that de_en_subwords.json lacks;
#: each is a one-word entry of those tables, nothing is guessed from variable names.
GLOSS = {
    "profil": "profile", "vorschlag": "proposal", "vorgeschlagen": "proposed", "orakel": "oracle", "baum": "tree", "baeume": "trees",
    "beleg": "evidence", "belege": "evidence_items", "konsequenz": "consequence", "anker": "anchor", "kante": "edge", "kanten": "edges",
    "hinweis": "note", "datenblatt": "datasheet", "geborgt": "borrowed", "verweigert": "refused", "ablehnung": "refusal",
    "treiber": "driver", "eingabe": "input", "passung": "fit", "pfad": "path", "balken": "bar", "ueberlauf": "overflow",
    "schnitt": "cut", "gewichte": "weights", "experten": "experts", "experte": "expert", "sitze": "seats", "kontext": "context",
    "lauf": "run", "laeufe": "runs", "katalog": "catalog", "glossar": "glossary", "titel": "title", "datei": "file",
    "inventar": "inventory", "einzelkarte": "single_card", "einzel": "single", "stufe": "stage", "klasse": "class",
    "speicher": "memory", "aufteilung": "split", "gemessen": "measured", "geschaetzt": "estimated", "gerechnet": "computed",
    "forcebar": "forceable", "ebene": "level",
    # function words of the glossaries (Part 1 general terms: kein Lauf = no run, mit --force = with --force, nicht gemessen = not measured)
    "geht": "ok", "mit": "with", "nicht": "not", "und": "and", "je": "per", "fuer": "for", "von": "from", "kein": "no", "keine": "no",
}
#: words whose meaning differs by context (merged_0928.json: belegt -> occupied_rows = a ledger row is taken; the planner modules and the glossaries:
#: belegt = verified). A compound containing one is never built from the lists: it goes to the review file.
AMBIGUOUS_PARTS = {"belegt"}
#: hand-set targets for compounds the part-wise build words unnaturally
MANUAL = {"passt_nicht": "does_not_fit", "geht_mit_force": "ok_with_force"}
#: English/technical parts that are not in the English word lists
ENGLISH_OK = {"mib", "gib", "ms", "us", "sha256", "nf", "hw", "tp", "pp", "kv", "gpu", "api", "sm", "id", "ok", "x", "d", "p", "dual", "bar1",
              "a", "an", "in", "is", "of", "to", "the", "for", "and", "on", "by", "as", "at", "it", "or", "vs", "re", "js", "ui", "py"}
#: German-looking tokens the lexicon flags that are not German identifiers (reviewed by hand; listed in the review file)
FALSE_POSITIVE = {"prange", "sargs", "durs", "soff", "refus", "hangt_an", "ruhe"}


def split_parts(word):
    if "_" in word or word.isupper() or word.islower():
        return word.split("_"), ("upper" if word.isupper() else "lower")
    return re.findall(r"[A-Z][a-z0-9]*|[a-z0-9]+", word), "camel"


def join_parts(parts, style):
    if style == "upper":
        return "_".join(p.upper() for p in parts)
    if style == "camel":
        return "".join(p[:1].upper() + p[1:] for p in parts)
    return "_".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--review", required=True)
    ap.add_argument("--dict-dir", default=EA.DEFAULT_DICT_DIR)
    a = ap.parse_args()
    lex = EA.Lexicon(a.dict_dir, "/spinning/htsglang", "upstream/main")
    sub = json.load(open(os.path.join(HERE, "data", "de_en_subwords.json")))
    sub = {k: v for k, v in sub.get("map", sub).items() if not k.startswith("_")}
    tables = {n: {k: v for k, v in json.load(open(os.path.join(HERE, "data", n))).items() if k != "_comment"}
              for n in ("merged_0928.json", "identfix_map.json", "decided_0928.json")}
    words = {}
    for f in a.scan:
        d = json.load(open(f))
        tag = os.path.basename(f).split(".")[0]
        for w, e in d["words"].items():
            x = words.setdefault(w, {"heads": [], "count": 0, "kinds": {}, "tree_files": 0, "js_files": 0, "files": {}})
            x["heads"].append(tag)
            x["count"] = max(x["count"], e["count"])
            x["tree_files"] = max(x["tree_files"], e["tree_files"])
            x["js_files"] = max(x["js_files"], e.get("js_files", 0))
            for k, v in e["kinds"].items():
                x["kinds"][k] = max(x["kinds"].get(k, 0), v)
            for p, n in e["files"].items():
                x["files"][p] = max(x["files"].get(p, 0), n)
    table, review = {}, {"needs_translation": {}, "too_common": {}, "test_names": {}, "conflicts_with_tables": {}, "false_positive": sorted(FALSE_POSITIVE), "dup_target": []}
    for w, x in sorted(words.items()):
        spread = {"heads": x["heads"], "count": x["count"], "kinds": x["kinds"], "tree_files": x["tree_files"], "js_files": x["js_files"],
                  "top_files": dict(sorted(x["files"].items(), key=lambda kv: -kv[1])[:3])}
        if w in FALSE_POSITIVE:
            continue
        if w.startswith("test_") and w not in PLANNER:       # sentence-like test names: translation workers, not a table
            review["test_names"][w] = {"count": x["count"], "top_files": spread["top_files"]}
            continue
        lw = w.lower()
        if w in PLANNER:
            target = PLANNER[w]
        elif lw in PLANNER and w != lw:
            parts, style = split_parts(w)
            target = join_parts(split_parts(PLANNER[lw])[0], style)
        else:
            parts, style = split_parts(w)
            out, missing = [], []
            if w in MANUAL:
                parts = []
                out = [MANUAL[w]]
            if any(p.lower() in AMBIGUOUS_PARTS for p in parts):
                review["needs_translation"][w] = dict(spread, unmapped_parts=sorted(p.lower() for p in parts if p.lower() in AMBIGUOUS_PARTS),
                                                      note="ambiguous part (occupied vs verified)")
                continue
            for p in parts:
                q = p.lower()
                if q in GLOSS:
                    out.append(GLOSS[q])
                elif q in sub:
                    out.append(sub[q])
                elif q in ENGLISH_OK or re.fullmatch(r"[0-9]+[a-z]?[0-9]*|[a-z][0-9]+[a-z]?[0-9]*", q) or (q in lex.en and q not in lex.de):
                    out.append(q)
                else:
                    missing.append(q)
            if missing:
                review["needs_translation"][w] = dict(spread, unmapped_parts=missing)
                continue
            target = join_parts([o for p in out for o in p.split("_")], style)
        clash = {n: t[w] for n, t in tables.items() if w in t and t[w] != target}
        same = [n for n, t in tables.items() if t.get(w) == target]
        if clash:
            review["conflicts_with_tables"][w] = dict(spread, proposed=target, existing=clash)
            if w not in PLANNER:
                continue
        elif same:
            continue                       # already in a kit table with this target
        if x["tree_files"] > COMMON_MAX and w not in PLANNER:
            review["too_common"][w] = dict(spread, proposed=target)
            continue
        if target == w:
            continue
        table[w] = target
    for w, t in PLANNER.items():           # every planner decision is in the table, scanned or not, new or already in a kit table
        if table.get(w) != t:
            if w in table:
                review["conflicts_with_tables"].setdefault(w, {})["note"] = "planner decision overrides the generated entry"
            table[w] = t
    byt = {}
    for k, v in table.items():
        byt.setdefault(v, []).append(k)
    review["dup_target"] = [{"new": v, "old": sorted(ks)} for v, ks in sorted(byt.items()) if len(ks) > 1]
    meta = {"_comment": "F0-A 07.10.2026: PROPOSAL (not applied). Planner-seat decision 07.10. (JSON keys of the planner schemas -> English) + words built from "
                        "de_en_subwords.json and the dashboard glossaries; generated by ident_map_build.py from ident_scan.py reports of the 27B head "
                        "07c20a35e5 and the NF head c5da548b7c. Judge with ident_precheck.py; apply with FIXMAP_EXTRA (release_rename.sh). "
                        "Review items (needs_translation, too_common, conflicts) are in ident_map_1007.review.json."}
    json.dump({**meta, **dict(sorted(table.items()))}, open(a.out, "w"), indent=1, ensure_ascii=False)
    review["summary"] = {"table_entries": len(table), "planner_entries": len([k for k in table if k in PLANNER]),
                         "needs_translation": len(review["needs_translation"]), "test_names": len(review["test_names"]), "too_common": len(review["too_common"]),
                         "conflicts_with_tables": len(review["conflicts_with_tables"]), "dup_target": len(review["dup_target"]),
                         "words_scanned": len(words)}
    json.dump(review, open(a.review, "w"), indent=1, ensure_ascii=False, sort_keys=False)
    print("ident_map_build: %s" % json.dumps(review["summary"]))


if __name__ == "__main__":
    main()
