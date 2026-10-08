"""PA 28.09. (release checklist item 2, RENAME_PLAN 8.15 step 3): the mechanical rename commit
(``rename_to_flliper.py apply --pdflip``) refuses a file in which two DIFFERENT words become the same
word -- e.g. a legacy env name next to a literal of its renamed spelling. On unified f5d9d30c04 the
probe stopped on three such files (pdflip/form.py docstring, this directory's
test_unify_prefix_switches_profile.py, and an ident-map pair in the launcher). This pin runs the
same collision rule (package/subsystem words only; the German->English identifier table lives in
the fLLiper repo) over every tree file that spells a renamed word literally, so a new one is caught
at the desk instead of at the rename.

The renamed words are written split here and built at run time, so this file itself stays a fixed
point of the rename (as name_compat / compat_shims do).
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
_OLD_PKG, _NEW_PKG = "sg" "lang", "fll" "iper"
_OLD_SUB, _NEW_SUB = "we" "g2", "pd" "flip"
# (legacy, renamed) in the order the tool applies them; the subsystem word glued to "Flip" is shortened
_PAIRS = (
    (_OLD_PKG.upper(), _NEW_PKG.upper()),
    (_OLD_PKG, _NEW_PKG),
    ("SG" "Lang", "Fll" "iper"),
    ("Sg" "lang", "Fll" "iper"),
    (_OLD_SUB.capitalize() + "Flip", "Pd" "Flip"),
    (_OLD_SUB.upper(), _NEW_SUB.upper()),
    (_OLD_SUB, _NEW_SUB),
    (_OLD_SUB.capitalize(), "Pd" "Flip"),
)
_WORD = re.compile(r"\w+")
_NEW_RX = re.compile("|".join(re.escape(n) for _, n in _PAIRS) + "|" + "fLL" "iper")
# files that spell both generations ON PURPOSE (rename_to_flliper.COLLISION_OK, FL4 26.09.)
_EXEMPT = {
    "test/registered/unit/pdflip/test_pdflip_name_compat_env_1b.py",
    "test/registered/unit/pdflip/test_pdflip_name_compat_readers_1a.py",
}


def _mapped(word):
    low = word.lower()
    if ("ht" + _OLD_PKG) in low:  # the product name keeps its letters (tool: (?<![Hh][Tt]))
        return word
    if word in ("SG" "Lang", "SG" "lang", "sGL" "ang"):  # free-standing brand -> the display name (tool)
        return "fLL" "iper"
    for old, new in _PAIRS:
        word = word.replace(old, new)
    return word


def collisions(text):
    """{renamed word: [distinct source words]} for words that merge under the rename."""
    seen = {}
    for w in set(_WORD.findall(text)):
        seen.setdefault(_mapped(w), set()).add(w)
    # the free brand word (prose) is not an identifier namespace (tool: same exception)
    return {k: sorted(v) for k, v in seen.items() if len(v) > 1 and k != "fLL" "iper"}


def _files_with_renamed_words():
    out = subprocess.run(
        ["git", "-C", str(ROOT), "grep", "-l", "-E", _NEW_RX.pattern, "--", "python", "test"],
        capture_output=True, text=True,
    )
    if out.returncode not in (0, 1):
        pytest.skip("git grep unavailable: %s" % out.stderr.strip()[:200])
    return [p for p in out.stdout.split() if p.endswith(".py")]


def test_rule_catches_a_legacy_name_next_to_its_renamed_literal():
    legacy = _OLD_PKG.upper() + "_" + _OLD_SUB.upper() + "_TOLD_PACED"
    text = 'env = {"%s": "0"}\nassert f("%s")\n' % (_mapped(legacy), legacy)
    assert collisions(text) == {_mapped(legacy): sorted([legacy, _mapped(legacy)])}
    assert collisions('x = "%s"' % legacy) == {}


def test_product_name_is_not_mapped():
    assert _mapped("ht" + _OLD_PKG) == "ht" + _OLD_PKG


def test_no_tree_file_merges_two_words_under_the_rename():
    bad = {}
    for rel in _files_with_renamed_words():
        if rel in _EXEMPT:
            continue
        c = collisions((ROOT / rel).read_text(errors="replace"))
        if c:
            bad[rel] = c
    assert not bad, "rename collisions (write the renamed spelling split or build it at run time): %s" % bad
