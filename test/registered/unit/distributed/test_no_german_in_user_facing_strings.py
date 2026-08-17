# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""No German left in a string an operator can read (#381).

WHY A SCAN AND NOT A LINE ASSERTION
-----------------------------------
Tasks #295 and #358 translated this fork's German to English, strand by
strand: the HTCCL/barlink transport, then bar1ep. Strand-by-strand is how a
remnant survives -- the dual-group planner was in neither, so
``check_nesting`` went on refusing ratio pairs with "the Verband's weight
bytes" until a boot-matrix arm hit it in #349 sweep 2 and someone read the
message.

A test that pins the one message that was found would not have found it. This
one asks the question the sweeps were asking, over the whole tree, every run:
does any string an operator can READ still contain German?

TWO SCANNERS, TWO SCOPES, ONE VOCABULARY
----------------------------------------
``scan_tree`` is unchanged: user-facing STRINGS only. Comments and docstrings
stay out of ITS scope, and the test pinning that is still below.

``scan_tree_prose`` was added later and covers comments and docstrings, because
the standing user law is that all code is English -- not only the parts an
operator reads at 3am. The original scoping note argued that a German comment
"costs a reader nothing"; that was a judgement about cost, and the law is not a
judgement about cost. Both scanners share ``GERMAN_WORDS``, so a remnant found
by either teaches the other.

The prose sweep found eleven islands the string sweep could not see by
construction -- including one word of English prose in
``distributed/dual_group.py``, "the serving Verband", which several passes over
that file had walked straight past.

WHAT COUNTS AS USER-FACING
--------------------------
Strings that reach a person: the message of a raised exception, a logging
call, ``print``, and the ``help=``/``why=``/``reason=`` keywords that end up
in CLI help or a rendered report. That is where a wrong language costs
someone time; a German comment costs a reader nothing at 3am.

ADDING A WORD is the point. When the next remnant turns up, put it in
``GERMAN_WORDS`` rather than only fixing the site -- the list is what makes
the next one fail here instead of in someone's boot log.
"""

import ast
import io
import os
import re
import tokenize
import unittest
from typing import Dict, List, Tuple

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

SRT_ROOT = os.path.join(os.path.dirname(os.path.abspath(sglang.__file__)), "srt")

#: Words that are German and are not also English. Deliberately not a
#: dictionary: a broad list would fire on "gross size" (English, and correct
#: for a BAR1 aperture) and teach everyone to skip the failure.
GERMAN_WORDS: Tuple[str, ...] = (
    "Verband", "Verbands", "Verbaende",
    "Raenge", "Rang", "Karte", "Karten", "Gruppe", "Gruppen",
    "Fenster", "Schlitz", "Schlitze", "Beleg", "Belege",
    "Runde", "Runden", "Zeile", "Zeilen", "Spalte", "Spalten",
    "Puffer", "Traeger", "Knoten", "Speicher", "Anzahl",
    "Groesse", "Laenge", "Aufteilung", "Verhaeltnis", "Vorgabe",
    "Zustand", "Vorbehalt", "Gitter", "Halter", "Ladeform", "Quelle",
    "Freigabe", "Deckel", "Domaenen", "Staffel", "Planer",
    "nicht", "oder", "keine", "einen", "diese", "wurde", "muss",  # codespell:ignore
)
_WORD_RE = re.compile(r"\b(" + "|".join(GERMAN_WORDS) + r")\b")
_UMLAUT_RE = re.compile(r"[äöüÄÖÜß]")

#: Calls whose string arguments a person reads.
_USER_FACING_CALLS = frozenset(
    {"warning", "warning_once", "info", "error", "critical", "debug",
     "exception", "fatal", "print"}
)
#: Keywords that end up in CLI help or a rendered report.
_USER_FACING_KWARGS = frozenset({"help", "why", "reason", "detail", "msg", "message"})

#: Strings that are German ON PURPOSE. Each needs a reason, because an
#: allowlist without one is just a way to make this test stop working.
ALLOWED: Dict[str, str] = {
    # MGSM is the multilingual grade-school-math benchmark; its German split
    # carries a German instruction by definition. Translating it would change
    # what the benchmark measures.
    "test/simple_eval_mgsm.py": "MGSM's German prompt is the benchmark itself",
}


def _is_allowed(path: str) -> bool:
    return any(path.endswith(suffix) for suffix in ALLOWED)


def _literals(node: ast.AST) -> List[Tuple[int, str]]:
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            out.append((sub.lineno, sub.value))
    return out


def user_facing_strings(tree: ast.AST) -> List[Tuple[int, str]]:
    """Every string literal in ``tree`` that a person can end up reading.

    Docstrings and comments are excluded STRUCTURALLY rather than filtered: a
    docstring is a bare ``Expr`` statement and a comment is not in the AST at
    all, so neither can be an argument to a raise or to a user-facing call.
    An explicit filter here would be dead code that looks load-bearing --
    which is worse than no filter, because the test below would then appear to
    prove something the code never does.
    """
    out: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        found: List[Tuple[int, str]] = []
        if isinstance(node, ast.Raise) and node.exc is not None:
            found += _literals(node.exc)
        elif isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            if name in _USER_FACING_CALLS:
                for arg in node.args:
                    found += _literals(arg)
            for kw in node.keywords or []:
                if kw.arg in _USER_FACING_KWARGS:
                    found += _literals(kw.value)
        out += found
    return out


def scan_tree(root: str) -> List[Tuple[str, int, str, str]]:
    """(path, line, matched word, text) for every German user-facing string."""
    hits: List[Tuple[str, int, str, str]] = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            if _is_allowed(path):
                continue
            try:
                tree = ast.parse(open(path, encoding="utf-8").read())
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue
            for lineno, text in user_facing_strings(tree):
                m = _WORD_RE.search(text) or _UMLAUT_RE.search(text)
                if m:
                    hits.append(
                        (path, lineno, m.group(0), " ".join(text.split())[:120])
                    )
    return hits


#: Files whose German is the SUBJECT MATTER, not translation debt. Same rule as
#: ALLOWED above: a reason each, because an allowlist without one is just a way
#: to make a test stop working.
ALLOWED_PROSE: Dict[str, str] = {
    # The DE<->ES live translator. Its docstrings quote real German field
    # recordings (the sentence that produced a measured false positive) and a
    # verbatim user order about speaking German. Translating the evidence
    # would destroy what it is evidence OF.
    "srt/translator/session.py": "German field-recording evidence is the subject",
    # Discusses accent folding with linguistic examples -- si/si and uber/uber
    # must compare equal. The German word IS the example.
    "srt/translator/scoring.py": "German words are the linguistic examples here",
}


def _is_allowed_prose(path: str) -> bool:
    norm = path.replace(os.sep, "/")
    return any(norm.endswith(suffix) for suffix in ALLOWED_PROSE)


def prose_blocks(path: str) -> List[Tuple[int, str]]:
    """Every comment and docstring in ``path``.

    Comments are not in the AST at all, so they need the tokenizer; docstrings
    are, so they do not. Both are read: a sweep covering only one of them is
    how the #381 remnant survived in the first place.
    """
    out: List[Tuple[int, str]] = []
    try:
        src = open(path, encoding="utf-8").read()
    except (UnicodeDecodeError, OSError):
        return out
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                out.append((tok.start[0], tok.string))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        pass
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return out
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            doc = ast.get_docstring(node)
            if doc:
                out.append((getattr(node, "lineno", 1), doc))
    return out


def scan_tree_prose(root: str) -> List[Tuple[str, int, str, str]]:
    """(path, line, matched word, text) for every German comment or docstring."""
    hits: List[Tuple[str, int, str, str]] = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            if _is_allowed(path) or _is_allowed_prose(path):
                continue
            for lineno, text in prose_blocks(path):
                m = _WORD_RE.search(text) or _UMLAUT_RE.search(text)
                if m:
                    hits.append(
                        (path, lineno, m.group(0), " ".join(text.split())[:120])
                    )
    return hits


class TestNoGermanInCommentsOrDocstrings(CustomTestCase):
    """The widened scope: all code is English, comments included."""

    def test_the_srt_tree_prose_is_clean(self):
        hits = scan_tree_prose(SRT_ROOT)
        if hits:
            lines = "\n".join(f"  {p}:{ln}  [{w}]  {t}" for p, ln, w, t in sorted(hits))
            self.fail(
                f"{len(hits)} comment(s)/docstring(s) still contain German:\n"
                f"{lines}\nTranslate them in place. If the German IS the "
                "subject matter (a quoted recording, a linguistic example), "
                "add the file to ALLOWED_PROSE with the reason."
            )

    def test_the_prose_scanner_finds_a_planted_island(self):
        """The falsifier. A widened scan that cannot fail widens nothing."""
        import tempfile

        src = (
            '"""Modul-Docstring ueber die Gruppe."""\n'  # codespell:ignore
            "# ein deutscher Kommentar ueber Raenge\n"
            "X = 1\n"
        )
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "planted.py"), "w") as fh:
                fh.write(src)
            hits = scan_tree_prose(d)
        self.assertEqual(len(hits), 2, hits)
        self.assertEqual({h[2] for h in hits}, {"Gruppe", "Raenge"})

    def test_the_planted_island_is_invisible_to_the_string_scanner(self):
        """Proof the two scopes really differ, so neither can later be deleted
        on the grounds that the other covers it."""
        import tempfile

        src = "# ein deutscher Kommentar ueber Raenge\nX = 1\n"
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "planted.py"), "w") as fh:
                fh.write(src)
            self.assertEqual(scan_tree(d), [])
            self.assertEqual(len(scan_tree_prose(d)), 1)

    def test_every_prose_allowlist_entry_carries_a_reason(self):
        for path, reason in sorted(ALLOWED_PROSE.items()):
            with self.subTest(path=path):
                self.assertGreater(
                    len(reason), 20, f"{path}: allowlisted without a real reason"
                )

    def test_the_prose_allowlist_files_still_exist(self):
        """An allowlist entry outliving its file silently widens the exemption
        the next time that path is recreated."""
        root = os.path.dirname(SRT_ROOT)
        for path in sorted(ALLOWED_PROSE):
            with self.subTest(path=path):
                self.assertTrue(os.path.isfile(os.path.join(root, path)), path)


class TestNoGermanInUserFacingStrings(CustomTestCase):
    def test_the_srt_tree_is_clean(self):
        hits = scan_tree(SRT_ROOT)
        if hits:
            lines = "\n".join(
                f"  {p}:{ln}  [{w}]  {t}" for p, ln, w, t in sorted(hits)
            )
            self.fail(
                f"{len(hits)} user-facing string(s) still contain German:\n{lines}\n"
                "Translate them, or -- if the German is deliberate, as in a "
                "multilingual benchmark prompt -- add the file to ALLOWED with "
                "the reason."
            )

    def test_the_scanner_finds_a_planted_remnant(self):
        """The falsifier. A scan that cannot fail is worse than no scan.

        #381 exists because a strand-by-strand translation missed a file; a
        test that silently matched nothing would repeat that at one remove.
        """
        import tempfile

        src = (
            "import logging\n"
            "logger = logging.getLogger(__name__)\n"
            "def f():\n"
            '    """Ein deutscher Docstring, der nicht zaehlt."""\n'
            '    raise ValueError("the PD lane cannot share the Verband\'s bytes")\n'
        )
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "planted.py"), "w") as fh:
                fh.write(src)
            hits = scan_tree(d)
        self.assertEqual(len(hits), 1, hits)
        self.assertEqual(hits[0][2], "Verband")

    def test_the_scanner_ignores_comments_and_docstrings(self):
        """Out of scope by decision, not by accident -- pinned so a later
        widening is a deliberate act."""
        import tempfile

        src = (
            '"""Modul-Docstring: diese Gruppe ist nicht uebersetzt."""\n'  # codespell:ignore
            "# ein deutscher Kommentar ueber Raenge\n"
            "X = 1\n"
        )
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "prose.py"), "w") as fh:
                fh.write(src)
            self.assertEqual(scan_tree(d), [])

    def test_english_gross_is_not_a_false_positive(self):
        """"BAR1 gross size" is English and correct; a word list that fired on
        it would be ignored within a week."""
        import tempfile

        src = (
            "import logging\n"
            "logger = logging.getLogger(__name__)\n"
            'logger.warning("could not get BAR1 gross size from sysfs (%r)", 1)\n'
        )
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "ok.py"), "w") as fh:
                fh.write(src)
            self.assertEqual(scan_tree(d), [])


if __name__ == "__main__":
    unittest.main()
