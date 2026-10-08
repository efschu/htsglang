#!/usr/bin/env python3
"""english_audit.py -- inventory and CHECKED translation pipeline for
"weg2 -> pdflip" and "German -> English" in the fLLiper code tree.

The mechanical part (paths, identifiers, env names, flags, log markers) is done
by rename_to_flliper.py (`--weg2`, `--ident-map`).  This tool covers what a
regex cannot do safely -- natural-language text -- and makes it checkable:

  fetch-dicts   download the two word lists (pinned sha256) into --dict-dir
  inventory     read-only census over a git ref (+ external tool dirs):
                weg2 by category, German identifiers, German prose by kind,
                translation volume (lines, chars, ~tokens)
  extract       write translation units (JSONL) for German prose, one line per
                unit: file, kind, start/end position, exact source text.
                Units are sharded by file so several workers never touch one file.
  apply         write translated units back by position; every unit's original
                text must still match byte for byte, else the unit is refused.
  check         the acceptance gate for a translated file set (base ref vs dir):
                * Python: the token stream without COMMENT tokens is identical
                  except inside translatable strings (docstrings, log/raise/
                  warning/print arguments, argparse help=), and ast.dump with
                  exactly those string constants blanked is identical;
                * inside every changed string the MACHINE TOKENS are preserved
                  as a multiset: log markers (UPPER-DASH words), key= fields,
                  %-/{}-placeholders, #ticket refs, numbers, `backticked` names,
                  and every string of the MUST-KEEP list (markers our evaluators
                  match: host_acceptance, logindex, monitors);
                * non-Python text files: the part of each line before a comment
                  is identical.

Detection of German: word lists (German list minus developer English, where
developer English = the 20k most common English words plus every word used in
upstream SGLang's own code and docs), German function words, umlauts/ae-oe-ue.
It is a heuristic, so the inventory prints samples and per-kind counts, and the
gate above -- not the detector -- is what guarantees behaviour is unchanged.
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tokenize
from typing import Dict, Iterable, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rename_to_flliper as R  # noqa: E402  (scope, git IO helpers)

DICTS = {
    "de.txt": ("https://raw.githubusercontent.com/enz/german-wordlist/master/words", "20095928ca70974f"),
    "en20k.txt": ("https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt",
                  "4ed6e5336d7760d2"),
    # broad English list (over-inclusive: it also holds fenster, sitz, kette ...); used only as a
    # second filter, DE_EXTRA overrides it
    "en.txt": ("https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt", "3ed0c94610d8bcf7"),
}
DEFAULT_DICT_DIR = os.environ.get("FLLIPER_DICT_DIR", "/root/.claude/jobs/1ab4cd30/tmp/fl_dict")

# German function words that are NOT (common) English words
FUNCTION_WORDS = set("""
und oder nicht ist sind wird werden wurde wurden der das dem den des ein eine einen einem einer eines
kein keine keinen keinem keiner mit fuer für auf bei beim noch nur auch wenn dann aus nach ueber über
zu zum zur im vom von sich hat haben wie als bis jetzt schon weil damit dass daß muss müssen muessen
kann können koennen soll sollen sein seine seinen wir ich er ihr uns gegen ohne unter durch sonst aber
doch hier dort alle alles immer nie mehr weniger erst wieder sowie bereits dieser diese dieses diesem
diesen jede jeder jedes jeden nichts etwas welche welcher welches wo wann warum weshalb ob sodass
zwischen waehrend während statt trotz innerhalb ausserhalb außerhalb hinter neben vorher nachher
gleich selbst genau eigentlich natuerlich natürlich richtig falsch fertig wirklich bitte danke
""".split())
UMLAUT = re.compile(r"[äöüÄÖÜß]")
WORD = re.compile(r"[A-Za-zÄÖÜäöüß]+")
SUB = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])")

# developer-English words that sit in the German list but are English in code
ALLOW_EN = set("""
arena ledger census carrier corridor seam tier probe ring lane rank tag slot band credit debit gate
arm form park wake sleep flip draft anchor verdict ratchet netting bounce leg shadow residue ramp
latch vote fence knob bench kernel graph chunk batch spill tail head drain witness quorum cursor
bar barlink bar1 dmabuf mamba hicache nvfp nvfp4 marlin cutlass triton prefill decode admission
rest sems profs reliefs gens bandit joules persona sabotage emitter minima idem newcomer residuum
stand real
""".split())
# German words the lists miss (compounds, domain words seen in this tree)
DE_EXTRA = set("""
abnahme erstboot bestform bauplan riegel waechter wächter mieter nadel sitz zeile grenze pflicht naht
stau zweig befund messung beweis karte wecker schlaf linie huelle hülle gesetz kasse kette zensus platte
bauphase auswertung uebergabe übergabe weiche lauf laeufer läufer schalter zaehler zähler fenster anker
abgleich anlauf aufwach aufwachen bremse eimer faden feld fuellstand griff haken kopf kopie leiter
merker muster pfad pruefung prüfung puffer quelle schicht schieber schnitt schritt speicher
stapel takt teil uhr umbau vorrat waage wert zahl ziel zug zuteilung stufe stufen
""".split())


# --------------------------------------------------------------------------
# lexicon
# --------------------------------------------------------------------------
class Lexicon:
    def __init__(self, dict_dir: str, repo: str, upstream_ref: str):
        de_path = os.path.join(dict_dir, "de.txt")
        en_path = os.path.join(dict_dir, "en20k.txt")
        for p in (de_path, en_path):
            if not os.path.exists(p):
                sys.exit(f"missing {p}: run `english_audit.py fetch-dicts --dict-dir {dict_dir}`")
        self.de: Set[str] = set()
        with open(de_path, encoding="utf-8") as f:
            for w in f:
                w = w.strip().lower()
                if w:
                    self.de.add(w)
                    self.de.add(w.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss"))
        with open(en_path, encoding="utf-8") as f:
            self.en = {w.strip().lower() for w in f if w.strip()}
        broad = os.path.join(dict_dir, "en.txt")
        if os.path.exists(broad):
            with open(broad, encoding="utf-8") as f:
                self.en |= {w.strip().lower() for w in f if w.strip()}
        self.en |= self._upstream_vocab(dict_dir, repo, upstream_ref)
        self.en |= ALLOW_EN
        self.en -= DE_EXTRA

    @staticmethod
    def _upstream_vocab(dict_dir: str, repo: str, ref: str) -> Set[str]:
        sha = R._git(repo, "rev-parse", ref).decode().strip()
        cache = os.path.join(dict_dir, f"upstream_vocab-{sha[:12]}.txt")
        if os.path.exists(cache):
            with open(cache) as f:
                return {w.strip() for w in f if w.strip()}
        out = R._git(repo, "grep", "-I", "-h", "-o", "-P", "[A-Za-z]+", sha, "--",
                     "python", "test", "scripts", "benchmark", "docs", "README.md")
        c: collections.Counter = collections.Counter()
        for w in out.decode("utf-8", "replace").split():
            for p in SUB.findall(w):
                c[p.lower()] += 1
        vocab = {k for k, v in c.items() if v >= 2}
        with open(cache, "w") as f:
            f.write("\n".join(sorted(vocab)) + "\n")
        return vocab

    def german_sub(self, w: str) -> bool:
        w = w.lower()
        if w in DE_EXTRA:
            return True
        return len(w) >= 4 and w in self.de and w not in self.en

    def prose_score(self, text: str) -> Tuple[int, int, int]:
        fw = de = um = 0
        for t in WORD.findall(text):
            lt = t.lower()
            if lt in FUNCTION_WORDS:
                fw += 1
            elif self.german_sub(lt):
                de += 1
            if UMLAUT.search(t):
                um += 1
        return fw, de, um

    def is_german(self, text: str) -> bool:
        fw, de, um = self.prose_score(text)
        return fw >= 2 or (fw >= 1 and de >= 1) or de >= 2 or (um >= 1 and (fw + de) >= 1)

    def german_ident(self, name: str) -> List[str]:
        subs = SUB.findall(name)
        out = []
        for i, s in enumerate(subs):
            if s.lower() == "weg" and name[name.lower().find("weg") + 3: name.lower().find("weg") + 4] == "2":
                continue
            if self.german_sub(s):
                out.append(s.lower())
        return out


# --------------------------------------------------------------------------
# Python units: comments, docstrings, log/raise/help strings, other strings
# --------------------------------------------------------------------------
LOG_FUNCS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "fatal", "log",
             "print", "print_once", "info_once", "warning_once", "rank0_log", "rank0_print", "echo"}


def _call_name(f: ast.AST) -> str:
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def translatable_spans(tree: ast.AST) -> Dict[Tuple[int, int], str]:
    """(lineno, col) of string-expression starts that may be translated -> kind."""
    spans: Dict[Tuple[int, int], str] = {}

    def mark(node: ast.AST, kind: str) -> None:
        for n in ast.walk(node):
            if isinstance(n, (ast.Constant, ast.JoinedStr)) and (
                    isinstance(n, ast.JoinedStr) or isinstance(n.value, str)):
                spans.setdefault((n.lineno, n.col_offset), kind)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                spans[(body[0].value.lineno, body[0].value.col_offset)] = "docstring"
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in LOG_FUNCS or name.endswith(("Error", "Exception", "Warning")) or name == "warn":
                for a in node.args:
                    mark(a, "log")
                for k in node.keywords:
                    if k.arg in (None, "msg", "message"):
                        mark(k.value, "log")
            for k in node.keywords:
                if k.arg in ("help", "description", "epilog"):
                    mark(k.value, "help")
        elif isinstance(node, ast.Raise) and node.exc is not None:
            mark(node.exc, "log")
        elif isinstance(node, ast.Assert) and node.msg is not None:
            mark(node.msg, "log")
    return spans


def py_units(src: str) -> Tuple[List[dict], Optional[str]]:
    """All comment and string units of a Python source with their kind."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [], f"syntax: {e}"
    spans = translatable_spans(tree)
    units: List[dict] = []
    lines = src.splitlines(keepends=True)
    offs = [0]
    for l in lines:
        offs.append(offs[-1] + len(l))

    def off(pos: Tuple[int, int]) -> int:
        return offs[pos[0] - 1] + pos[1]

    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError) as e:
        return [], f"tokenize: {e}"
    i = 0
    fs_start = getattr(tokenize, "FSTRING_START", -1)
    fs_end = getattr(tokenize, "FSTRING_END", -2)
    while i < len(toks):
        t = toks[i]
        if t.type == tokenize.COMMENT:
            units.append({"kind": "comment", "start": t.start, "end": t.end, "text": t.string})
        elif t.type == tokenize.STRING or t.type == fs_start:
            start = t.start
            if t.type == fs_start:
                depth = 0
                while i < len(toks):
                    if toks[i].type == fs_start:
                        depth += 1
                    elif toks[i].type == fs_end:
                        depth -= 1
                        if depth == 0:
                            break
                    i += 1
            end = toks[i].end
            text = src[off(start):off(end)]
            units.append({"kind": spans.get(start, "string"), "start": start, "end": end, "text": text})
        i += 1
    return units, None


SH_COMMENT = re.compile(r"(^|\s)#(?![!{])")


def text_units(path: str, src: str) -> List[dict]:
    ext = os.path.splitext(path)[1]
    units = []
    for n, line in enumerate(src.splitlines(), 1):
        if ext in (".md", ".rst", ".txt"):
            if line.strip():
                units.append({"kind": "doc", "start": (n, 0), "end": (n, len(line)), "text": line})
            continue
        m = SH_COMMENT.search(line)
        if m:
            c = m.start() + len(m.group(1))
            units.append({"kind": "comment", "start": (n, c), "end": (n, len(line)), "text": line[c:]})
    return units


TEXT_EXT = {".sh", ".bash", ".env", ".toml", ".yml", ".yaml", ".cfg", ".ini", ".md", ".rst", ".txt", ".conf"}


def mark_german(units: List[dict], lex: "Lexicon") -> None:
    """Set u['german'].  Comment lines are judged per BLOCK (consecutive comment lines):
    a German sentence wrapped over three lines has a last line without two German
    signals of its own, and it still has to be translated with its block."""
    i = 0
    while i < len(units):
        u = units[i]
        if u["kind"] != "comment":
            body = u["text"]
            u["german"] = lex.is_german(body)
            i += 1
            continue
        j = i
        while j + 1 < len(units) and units[j + 1]["kind"] == "comment" and \
                units[j + 1]["start"][0] == units[j]["start"][0] + 1:
            j += 1
        block = " ".join(x["text"].lstrip("#: ") for x in units[i:j + 1])
        g = lex.is_german(block)
        for x in units[i:j + 1]:
            # a block-level verdict marks only lines that carry prose at all
            x["german"] = g and bool(WORD.search(x["text"]))
        i = j + 1


def file_units(path: str, src: str) -> Tuple[List[dict], Optional[str]]:
    if path.endswith(".py"):
        return py_units(src)
    if os.path.splitext(path)[1] in TEXT_EXT or os.path.basename(path).startswith("Dockerfile"):
        return text_units(path, src), None
    return [], None


# --------------------------------------------------------------------------
# weg2 classification
# --------------------------------------------------------------------------
W_ANY = re.compile(r"(?i)weg2")
W_KEEP = [
    ("boot-tag(keep)", re.compile(r"\bweg2[a-z]+\d\w*")),
    ("host-path(keep)", re.compile(r"/spinning/[\w./-]*weg2[\w./-]*|hicache-weg2[\w-]*|gpu-arb/weg2\b[\w./-]*")),
    ("image-tag(keep)", re.compile(r"cu1\d\d-weg2[\w.-]*")),
    ("doc-ref(keep)", re.compile(r"\bWEG2_[A-Z0-9_]*(?:SPEC|DECISIONS|PLAN|DESIGN)[\w-]*")),
]


def weg2_classify(path: str, text: str, c: collections.Counter, samples: Dict[str, List[str]],
                  env_names: Set[str], markers: Set[str], flags: Set[str]) -> None:
    keep: List[Tuple[int, int, str]] = []
    for name, rx in W_KEEP:
        for m in rx.finditer(text):
            keep.append((m.start(), m.end(), name))

    def add(cat: str, s: int) -> None:
        c[cat] += 1
        lst = samples.setdefault(cat, [])
        if len(lst) < 3:
            ls = text.rfind("\n", 0, s) + 1
            le = text.find("\n", s)
            lst.append(f"{path}: {text[ls:le if le >= 0 else None].strip()[:110]}")

    for m in W_ANY.finditer(text):
        s, e = m.span()
        k = next((n for a, b, n in keep if a <= s < b), None)
        if k:
            add(k, s)
            continue
        before, after = text[max(0, s - 12):s], text[e:e + 40]
        word = m.group(0)
        if re.search(r"(srt|scripts|unit|test)[./]$", before):
            add("module/package path", s)
        elif word == "WEG2" and after.startswith("-") and re.match(r"-[A-Z0-9]", after):
            add("log marker WEG2-*", s)
            markers.add("WEG2" + re.match(r"-[A-Z0-9-]+", after).group(0).rstrip("-"))
        elif word == "WEG2" and (after.startswith("_") or before.endswith("_")):
            add("env / constant WEG2_*", s)
            tok = re.search(r"\w*$", before).group(0) + "WEG2" + re.match(r"\w*", after).group(0)
            env_names.add(tok)
        elif word == "weg2" and before.endswith("--"):
            add("cli flag --weg2-*", s)
            flags.add("--weg2" + re.match(r"[\w-]*", after).group(0))
        elif word == "weg2" and before.endswith("/") and after.startswith("/") and not re.search(r"\w/$", before[:-1] + "/"):
            add("http route /weg2/*", s)
        elif word == "weg2" and before.endswith("/") and after.startswith("/"):
            add("http route /weg2/*", s) if re.search(r"[\"' ]/$", before) else add("module/package path", s)
        else:
            glued = (s > 0 and (text[s - 1].isalnum() or text[s - 1] == "_")) or \
                    (e < len(text) and (text[e].isalnum() or text[e] == "_"))
            add("identifier" if glued else "prose word", s)


# --------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------
def scan_file(path: str, text: str, lex: Lexicon, acc: dict, in_tree: bool) -> None:
    weg2_classify(path, text, acc["weg2"], acc["weg2_samples"], acc["weg2_env"], acc["weg2_markers"],
                  acc["weg2_flags"])
    if path.endswith(".py"):
        units, err = py_units(text)
        if err:
            acc["errors"].append(f"{path}: {err}")
        # identifiers: NAME tokens + identifier-like tokens in strings (env names, flags)
        try:
            for t in tokenize.generate_tokens(io.StringIO(text).readline):
                if t.type == tokenize.NAME:
                    g = lex.german_ident(t.string)
                    if g:
                        acc["de_ident"][t.string] += 1
                        acc["de_ident_files"].setdefault(t.string, path)
                        for w in g:
                            acc["de_sub"][w] += 1
        except (tokenize.TokenError, IndentationError):
            pass
    else:
        units, _ = file_units(path, text)
    for m in re.finditer(r"\b(?:SGLANG|HTSGLANG|WEG2)_[A-Z0-9_]+|--[a-z][a-z0-9-]+", text):
        g = lex.german_ident(m.group(0).replace("-", "_").lower())
        if g:
            acc["de_names"][m.group(0)] += 1
    fname = os.path.basename(path)
    if lex.german_ident(os.path.splitext(fname)[0]):
        acc["de_files"].append(path)
    mark_german(units, lex)
    for u in units:
        body = u["text"]
        if u["kind"] in ("comment",) and body.startswith("#"):
            body = body.lstrip("#")
        if not u["german"]:
            continue
        k = u["kind"]
        # volume = the German LINES of the unit (a long English docstring that quotes one
        # German sentence costs one line, not the docstring)
        glines = ([body] if k == "comment" else
                  [l for l in body.splitlines() if l.strip() and lex.is_german(l)] or [body])
        for l in glines:  # translation memory: identical lines are translated once
            acc["de_unique"].setdefault(hashlib.sha1(l.strip().encode()).hexdigest(), len(l.strip()))
        acc["de_units"][k] += 1
        acc["de_lines"][k] += len(glines)
        acc["de_chars"][k] += sum(len(l.strip()) for l in glines)
        acc["de_files_prose"].add(path)
        lst = acc["de_samples"].setdefault(k, [])
        if len(lst) < 4:
            lst.append(f"{path}:{u['start'][0]}: {u['text'].strip()[:120]}")


def new_acc() -> dict:
    return {"weg2": collections.Counter(), "weg2_samples": {}, "weg2_env": set(), "weg2_markers": set(),
            "weg2_flags": set(), "de_ident": collections.Counter(), "de_ident_files": {},
            "de_sub": collections.Counter(), "de_names": collections.Counter(), "de_files": [],
            "de_units": collections.Counter(), "de_lines": collections.Counter(),
            "de_chars": collections.Counter(), "de_files_prose": set(), "de_samples": {}, "errors": [], "de_unique": {},
            "files": 0, "weg2_paths": 0}


def summarize(acc: dict) -> dict:
    chars = sum(acc["de_chars"].values())
    return {
        "files_scanned": acc["files"],
        "weg2_by_category": dict(acc["weg2"].most_common()),
        "weg2_total": sum(acc["weg2"].values()),
        "weg2_paths": acc["weg2_paths"],
        "weg2_distinct_env": len(acc["weg2_env"]),
        "weg2_distinct_markers": len(acc["weg2_markers"]),
        "weg2_distinct_flags": sorted(acc["weg2_flags"]),
        "weg2_samples": acc["weg2_samples"],
        "german_identifiers_distinct": len(acc["de_ident"]),
        "german_identifiers_occurrences": sum(acc["de_ident"].values()),
        "german_identifiers_top": acc["de_ident"].most_common(60),
        "german_subwords_top": acc["de_sub"].most_common(80),
        "german_env_flag_names": acc["de_names"].most_common(40),
        "german_file_names": acc["de_files"][:60],
        "german_file_names_count": len(acc["de_files"]),
        "german_prose_units": dict(acc["de_units"]),
        "german_prose_lines": dict(acc["de_lines"]),
        "german_prose_chars": dict(acc["de_chars"]),
        "german_prose_est_tokens": round(chars / 3.5),
        "german_unique_lines": len(acc["de_unique"]),
        "german_unique_est_tokens": round(sum(acc["de_unique"].values()) / 3.5),
        "german_prose_files": len(acc["de_files_prose"]),
        "german_samples": acc["de_samples"],
        "parse_errors": acc["errors"][:20],
    }


def cmd_inventory(a: argparse.Namespace) -> int:
    lex = Lexicon(a.dict_dir, a.repo, a.upstream)
    tree = new_acc()
    for path, mode, data in R.iter_ref(a.repo, a.ref):
        if mode == "120000" or not R.in_path_scope(path) or R._glob(path, R.EXCLUDE_CONTENT):
            continue
        text = R.as_text(data)
        tree["files"] += 1
        if W_ANY.search(path):
            tree["weg2_paths"] += 1
        if text is None:
            continue
        scan_file(path, text, lex, tree, True)
    ext = {}
    for d in a.scan or []:
        acc = new_acc()
        for full in iter_external(d):
            try:
                with open(full, "rb") as f:
                    text = R.as_text(f.read())
            except OSError:
                continue
            if text is None:
                continue
            acc["files"] += 1
            if W_ANY.search(os.path.basename(full)):
                acc["weg2_paths"] += 1
            scan_file(os.path.relpath(full, d), text, lex, acc, False)
        ext[d] = summarize(acc)
    res = {"ref": a.ref, "sha": R._git(a.repo, "rev-parse", a.ref).decode().strip(),
           "upstream_vocab_ref": a.upstream, "tree": summarize(tree),
           "must_keep_markers_in_tree": sorted(tree["weg2_markers"]),
           "weg2_env_names_in_tree": sorted(tree["weg2_env"]), "external": ext}
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"written {a.json}")
    return 0


def iter_external(d: str) -> Iterable[str]:
    for dp, dns, fns in os.walk(d):
        dns[:] = sorted(x for x in dns if x not in R.EXT_SKIP_DIRS and not x.startswith(("wt-", "base-"))
                        and x not in ("scratch", "hfhome", "fl_dict")
                        and not os.path.exists(os.path.join(dp, x, ".git"))
                        and not os.path.isdir(os.path.join(dp, x, "python", "sglang"))
                        and not os.path.isdir(os.path.join(dp, x, "sglang")))
        for fn in sorted(fns):
            if R.EXT_OK.search(fn) and ".bak" not in fn and os.path.getsize(os.path.join(dp, fn)) < 2_000_000:
                yield os.path.join(dp, fn)


# --------------------------------------------------------------------------
# extract / apply / check
# --------------------------------------------------------------------------
TRANSLATABLE = {"comment", "docstring", "log", "help", "doc"}
#: never extracted for translation (FL4 26.09.): test DATA is evidence or upstream input, not prose -- fixture trees
#: (captured logs, PROVENIENZ notes, graph_check.txt dumps that tests parse) and the upstream English long-prompt file
#: that the German detector scores as one German line.
EXTRACT_EXCLUDE = ["**/fixtures/**", "python/*/test/long_prompt.txt"]


def cmd_extract(a: argparse.Namespace) -> int:
    lex = Lexicon(a.dict_dir, a.repo, a.upstream)
    root = os.path.abspath(a.root)
    stats = collections.Counter()
    per_file: Dict[str, List[dict]] = {}
    for path, mode, data in R.iter_root(root):
        if mode == "120000" or not R.in_scope(path) or R._glob(path, EXTRACT_EXCLUDE):
            continue
        text = R.as_text(data)
        if text is None:
            continue
        units, _ = file_units(path, text)
        mark_german(units, lex)
        for u in units:
            if u["kind"] not in TRANSLATABLE and not (a.include_strings and u["kind"] == "string"):
                continue
            if not u["german"]:
                continue
            rec = {"file": path, "kind": u["kind"], "start": u["start"], "end": u["end"],
                   "sha": hashlib.sha256(u["text"].encode()).hexdigest()[:16], "text": u["text"],
                   "translation": None}
            per_file.setdefault(path, []).append(rec)
            stats[u["kind"]] += 1
            stats["chars"] += len(u["text"])
    # whole file -> one shard (two workers never touch one file). --balanced: greedy by characters, largest file
    # first, ties by path (deterministic); default: path hash, as before.
    assign: Dict[str, int] = {}
    if a.balanced:
        load = [0] * a.shards
        for path in sorted(per_file, key=lambda p: (-sum(len(r["text"]) for r in per_file[p]), p)):
            i = min(range(a.shards), key=lambda k: (load[k], k))
            assign[path] = i
            load[i] += sum(len(r["text"]) for r in per_file[path])
    else:
        assign = {p: int(hashlib.sha256(p.encode()).hexdigest(), 16) % a.shards for p in per_file}
    shards = [open(os.path.join(a.out, f"units-{i:02d}.jsonl"), "w") for i in range(a.shards)]
    for path in sorted(per_file):
        for rec in per_file[path]:
            shards[assign[path]].write(json.dumps(rec, ensure_ascii=False) + "\n")
    for s in shards:
        s.close()
    stats["files"] = len(per_file)
    print(json.dumps(dict(stats), indent=1))
    return 0


def cmd_apply(a: argparse.Namespace) -> int:
    root = os.path.abspath(a.root)
    by_file: Dict[str, List[dict]] = collections.defaultdict(list)
    for p in a.units:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r.get("translation"):
                    by_file[r["file"]].append(r)
    refused = applied = 0
    for path in sorted(by_file):
        full = os.path.join(root, path)
        with open(full, encoding="utf-8") as f:
            src = f.read()
        lines = src.splitlines(keepends=True)
        offs = [0]
        for l in lines:
            offs.append(offs[-1] + len(l))
        edits = []
        for r in by_file[path]:
            s = offs[r["start"][0] - 1] + r["start"][1]
            e = offs[r["end"][0] - 1] + r["end"][1]
            if src[s:e] != r["text"]:
                refused += 1
                print(f"REFUSED (source moved): {path}:{r['start'][0]}")
                continue
            edits.append((s, e, r["translation"]))
        for s, e, t in sorted(edits, reverse=True):
            src = src[:s] + t + src[e:]
            applied += 1
        with open(full, "w", encoding="utf-8") as f:
            f.write(src)
    print(f"applied {applied}, refused {refused}")
    return 0 if not refused else 1


MACHINE = re.compile(
    r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+\b"      # log markers WEG2-FLIP-TAG, PDFLIP-SEQ
    r"|\b[A-Za-z_][\w.]*="                     # key= fields
    r"|%[-+ #0]*\d*(?:\.\d+)?[sdifrxXeEgGc%]"  # printf placeholders
    r"|\{[^{}\s]*\}"                            # format fields
    r"|#\d+[a-z]?"                              # ticket refs
    r"|`[^`\n]+`"                               # backticked code
    r"|\b\d+(?:\.\d+)?\b"                       # numbers
    r"|\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b"           # CONSTANT / ENV names (plain CAPS words are emphasis)
    r"|\b[A-Z]+\d[A-Z0-9]*\b"                   # W53, H91, BAR1
    r"|--[a-z][\w-]+"                           # CLI flags
    r"|\b[A-Za-z]\w*_\w+\b"                     # snake_case identifiers (page_size, max_total_tokens)
    r"|\b\w+(?:\.\w+)+\b"                        # dotted names / paths (x.y, launcher.py)
    r"|\b[a-z]+[A-Z]\w*\b"                       # camelCase identifiers
)


def machine_tokens(s: str) -> collections.Counter:
    return collections.Counter(MACHINE.findall(s))


def py_check(base: str, new: str, must_keep: List[str],
             allowed: Optional[Set[Tuple[int, int]]] = None) -> List[str]:
    """Behaviour-invariance of a translated Python file."""
    fails = []
    try:
        tb, tn = ast.parse(base), ast.parse(new)
    except SyntaxError as e:
        return [f"syntax error after translation: {e}"]
    ub, _ = py_units(base)
    un, _ = py_units(new)
    if len(ub) != len(un):
        return [f"comment/string unit count changed {len(ub)} -> {len(un)}"]
    for x, y in zip(ub, un):
        if x["text"] == y["text"]:
            continue
        listed = allowed is not None and tuple(x["start"]) in allowed
        if allowed is not None and not listed:
            fails.append(f"{x['kind']} changed at line {x['start'][0]} but not in the approved units")
            continue
        if x["kind"] not in TRANSLATABLE and not listed:
            fails.append(f"non-translatable {x['kind']} changed at line {x['start'][0]}: {x['text'][:60]!r}")
            continue
        if machine_tokens(x["text"]) != machine_tokens(y["text"]):
            d = (machine_tokens(x["text"]) - machine_tokens(y["text"])) + \
                (machine_tokens(y["text"]) - machine_tokens(x["text"]))
            fails.append(f"machine tokens changed at line {x['start'][0]}: {dict(d)}")
        for k in (must_keep if x["kind"] not in ("comment", "docstring", "doc") else ()):
            # evaluators match runtime output, not comments: must-keep binds log/help/string units
            if x["text"].count(k) != y["text"].count(k):
                fails.append(f"must-keep string {k!r} changed at line {x['start'][0]}")
    # AST with translatable string constants blanked must be identical
    extra = {p for p in (allowed or set())}

    def blanked(tree: ast.AST) -> str:
        spans = dict(translatable_spans(tree))
        for p in extra:  # approved kind=string units are blanked too
            spans.setdefault(p, "log")
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and \
                    spans.get((n.lineno, n.col_offset)) in TRANSLATABLE:
                n.value = ""
            if isinstance(n, ast.JoinedStr) and spans.get((n.lineno, n.col_offset)) in TRANSLATABLE:
                n.values = [v for v in n.values if isinstance(v, ast.FormattedValue)]
            if isinstance(n, ast.JoinedStr):  # literal parts inside a translatable f-string
                for v in n.values:
                    if isinstance(v, ast.Constant) and spans.get((n.lineno, n.col_offset)) in TRANSLATABLE:
                        v.value = ""
        return ast.dump(tree, include_attributes=False)
    if blanked(tb) != blanked(tn):
        fails.append("AST differs outside translatable strings")
    return fails


def text_check(base: str, new: str) -> List[str]:
    bl, nl = base.splitlines(), new.splitlines()
    if len(bl) != len(nl):
        return [f"line count changed {len(bl)} -> {len(nl)}"]
    fails = []
    for i, (x, y) in enumerate(zip(bl, nl), 1):
        if x == y:
            continue
        mx, my = SH_COMMENT.search(x), SH_COMMENT.search(y)
        cx = x[:mx.start() + len(mx.group(1))] if mx else x
        cy = y[:my.start() + len(my.group(1))] if my else y
        if cx != cy:
            fails.append(f"code part changed at line {i}")
        elif machine_tokens(x) != machine_tokens(y):
            fails.append(f"machine tokens changed in comment at line {i}")
    return fails


def cmd_check(a: argparse.Namespace) -> int:
    must_keep = []
    if a.must_keep:
        with open(a.must_keep) as f:
            must_keep = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    root = os.path.abspath(a.root)
    allowed_by_file: Optional[Dict[str, Set[Tuple[int, int]]]] = None
    if a.units:
        allowed_by_file = collections.defaultdict(set)
        for p in a.units:
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    allowed_by_file[r["file"]].add(tuple(r["start"]))
    base = {p: d for p, m, d in R.iter_ref(a.repo, a.base) if m != "120000"}
    fails: List[str] = []
    n = changed = 0
    for path, mode, data in R.iter_root(root):
        if path not in base or mode == "120000":
            continue
        n += 1
        if base[path] == data:
            continue
        changed += 1
        bt, nt = R.as_text(base[path]), R.as_text(data)
        if bt is None or nt is None:
            fails.append(f"{path}: binary changed")
            continue
        ext = os.path.splitext(path)[1]
        if path.endswith(".py"):
            fs = py_check(bt, nt, must_keep,
                          None if allowed_by_file is None else allowed_by_file.get(path, set()))
        elif ext in (".md", ".rst", ".txt"):
            fs = []  # prose documents: reviewed, machine tokens checked line by line
            if machine_tokens(bt) != machine_tokens(nt):
                fs.append("machine tokens changed in document")
        else:
            fs = text_check(bt, nt)
        fails += [f"{path}: {x}" for x in fs]
    print(json.dumps({"files_compared": n, "files_changed": changed, "failures": len(fails)}, indent=1))
    for f in fails[:60]:
        print("FAIL", f)
    print("VERDICT:", "PASS" if not fails else "FAIL")
    return 0 if not fails else 1


# --------------------------------------------------------------------------
# identifier proposals (German -> English), for rename_to_flliper.py --ident-map
# --------------------------------------------------------------------------
SHORT_DE = {"im": "in", "zu": "to", "und": "and", "mit": "with", "vom": "from", "von": "of", "aus": "from",
            "auf": "on", "bei": "at", "neu": "new", "ist": "is", "nur": "only", "je": "per", "pro": "per",
            "als": "as", "noch": "still", "wie": "how", "der": "the", "die": "the", "das": "the"}
PART = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+|_+")


def translate_ident(name: str, sub_map: Dict[str, str], lex: "Lexicon") -> Tuple[Optional[str], List[str]]:
    """Part-wise translation keeping the identifier's case style; (None, missing) if a German part
    has no entry."""
    parts = PART.findall(name)
    if "".join(parts) != name:
        return None, ["<unsplittable>"]
    out, missing = [], []
    for p in parts:
        lp = p.lower()
        if p.startswith("_") or p.isdigit():
            out.append(p)
            continue
        tr = sub_map.get(lp) if (lex.german_sub(lp) or lp in sub_map) else None
        if tr is None and lp in SHORT_DE:
            tr = SHORT_DE[lp]
        if tr is None:
            if lex.german_sub(lp):
                missing.append(lp)
            out.append(p)
            continue
        if p.isupper():
            tr = tr.upper()
        elif p[0].isupper():
            tr = "".join(w.capitalize() for w in tr.split("_")) if "_" in tr else tr.capitalize()
        out.append(tr)
    new = "".join(out)
    if missing:
        return None, missing
    return (new if new != name else None), []


def cmd_ident_proposals(a: argparse.Namespace) -> int:
    lex = Lexicon(a.dict_dir, a.repo, a.upstream)
    with open(a.subwords) as f:
        sub_map = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    names: collections.Counter = collections.Counter()
    all_names: Set[str] = set()
    files_of: Dict[str, Set[str]] = collections.defaultdict(set)   # name -> files
    names_in: Dict[str, Set[str]] = {}                              # file -> names
    string_words: collections.Counter = collections.Counter()
    for path, mode, data in R.iter_ref(a.repo, a.ref):
        if not path.endswith(".py") or not R.in_scope(path):
            continue
        text = R.as_text(data)
        if text is None:
            continue
        fnames: Set[str] = set()
        names_in[path] = fnames
        try:
            for t in tokenize.generate_tokens(io.StringIO(text).readline):
                if t.type == tokenize.NAME:
                    all_names.add(t.string)
                    fnames.add(t.string)
                    if lex.german_ident(t.string):
                        names[t.string] += 1
                        files_of[t.string].add(path)
                elif t.type == tokenize.STRING:
                    for w in re.findall(r"[A-Za-z_]\w*", t.string):
                        string_words[w] += 1
        except (tokenize.TokenError, IndentationError):
            pass
    auto, conflict, exposed, open_ = {}, {}, {}, {}
    targets: Dict[str, str] = {}
    for n, cnt in sorted(names.items()):
        parts = [p for p in PART.findall(n) if not p.startswith("_") and not p.isdigit()]
        sentence = len(parts) >= 5 or any(p.lower() in FUNCTION_WORDS for p in parts)
        new, missing = translate_ident(n, sub_map, lex)
        if missing or sentence:
            # sentence-like names (test names) and names with unmapped words go to the
            # REVIEWED table: translated like prose, applied mechanically afterwards
            open_[n] = {"count": cnt, "missing": missing, "sentence": sentence}
            continue
        if new is None:
            continue
        clash = [f for f in files_of[n] if new in names_in.get(f, ())]
        if clash or new in targets:
            conflict[n] = {"proposal": new, "clashes_in": clash[:3] or [f"proposal of {targets[new]}"]}
            continue
        if string_words.get(n):  # getattr("x"), record keys, kwargs by name: needs review
            exposed[n] = {"proposal": new, "string_occurrences": string_words[n]}
            continue
        targets[new] = n
        auto[n] = new
    res = {"ref": a.ref, "sha": R._git(a.repo, "rev-parse", a.ref).decode().strip(),
           "counts": {"german_identifiers": len(names), "auto": len(auto), "string_exposed": len(exposed),
                      "conflict": len(conflict), "open": len(open_)},
           "auto": auto, "string_exposed": exposed, "conflict": conflict, "open": open_}
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, sort_keys=False)
    with open(a.map, "w") as f:
        json.dump(auto, f, indent=1, sort_keys=True)
    print(json.dumps(res["counts"]))
    return 0


def cmd_fetch(a: argparse.Namespace) -> int:
    os.makedirs(a.dict_dir, exist_ok=True)
    for name, (url, sha) in DICTS.items():
        dst = os.path.join(a.dict_dir, name)
        subprocess.run(["curl", "-sSfL", "-o", dst, url], check=True)
        got = hashlib.sha256(open(dst, "rb").read()).hexdigest()[:16]
        print(name, got, "OK" if got == sha else f"MISMATCH (pinned {sha})")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo", default="/spinning/htsglang")
    common.add_argument("--dict-dir", default=DEFAULT_DICT_DIR)
    common.add_argument("--upstream", default="upstream/main", help="ref whose vocabulary counts as English")
    f = sp.add_parser("fetch-dicts", parents=[common])
    i = sp.add_parser("inventory", parents=[common])
    i.add_argument("--ref", required=True)
    i.add_argument("--scan", action="append")
    i.add_argument("--json", required=True)
    e = sp.add_parser("extract", parents=[common])
    e.add_argument("--root", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--shards", type=int, default=8)
    e.add_argument("--balanced", action="store_true", help="size-balanced shards (files stay whole)")
    e.add_argument("--include-strings", action="store_true",
                   help="also German strings outside log/raise/help (kind=string): each needs review; "
                        "check allows them only when listed in --units")
    p = sp.add_parser("apply", parents=[common])
    p.add_argument("--root", required=True)
    p.add_argument("units", nargs="+")
    c = sp.add_parser("check", parents=[common])
    c.add_argument("--root", required=True)
    c.add_argument("--base", required=True)
    c.add_argument("--must-keep")
    c.add_argument("--units", nargs="*", help="approved unit files: only these units may change")
    ip = sp.add_parser("ident-proposals", parents=[common])
    ip.add_argument("--ref", required=True)
    ip.add_argument("--subwords", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "data", "de_en_subwords.json"))
    ip.add_argument("--json", required=True, help="full report (auto/exposed/conflict/open)")
    ip.add_argument("--map", required=True, help="the auto map, input of rename_to_flliper.py --ident-map")
    a = ap.parse_args(argv)
    return {"fetch-dicts": cmd_fetch, "inventory": cmd_inventory, "extract": cmd_extract,
            "apply": cmd_apply, "check": cmd_check, "ident-proposals": cmd_ident_proposals}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
