# SPDX-License-Identifier: Apache-2.0
"""The pool is planner-solved: no acceptance recipe may pin it.

USER ORDER 2026-08-21. ``--max-total-tokens 430000`` was removed from every
boot-chain script and argv recipe on this rig. This test locks the CLASS
rather than the value, because the value was never the defect: the flag sat
inert for weeks while the profiled capacity was 161378 and the runtime logged
"larger than the profiled value ... use the profiled value instead". The
moment #785 raised the profile above it, the SAME untouched flag silently
became the binding constraint and capped a 764512-token pool at 430000.

A pin that is inert until it is catastrophic is not a safe default, and
"nobody has passed a number that low" is not a property anyone can maintain.
Per the planner's sole authority over VRAM, the pool comes from the solve.
"""

import glob
import os

import pytest

RECIPE_DIRS = ("/spinning/evidence-665-f1",)
RECIPE_GLOBS = ("argv_*.txt",)
PIN = "--max-total-tokens"


def _recipes():
    found = []
    for directory in RECIPE_DIRS:
        if not os.path.isdir(directory):
            continue
        for pattern in RECIPE_GLOBS:
            for path in sorted(glob.glob(os.path.join(directory, pattern))):
                if _is_recipe(path):
                    found.append(path)
    return found


def _is_recipe(path):
    """An argv recipe CONTAINS flag lines. Notes files sitting under the same
    glob are prose about a boot, not a boot -- and one of them
    (argv_hc_interval8192.HOSTRAM-NOTES.txt) was touched after the order and
    would otherwise fail this guard for quoting the flag it documents."""
    try:
        with open(path) as handle:
            return any(line.startswith("--") for line in handle)
    except OSError:
        return False


#: The order's effective moment, as a unix timestamp (2026-08-21 16:00 UTC).
#:
#: WHY A DATE AND NOT AN EXEMPTION LIST. 45 of this host's 68 argv recipes
#: still carry the pin: the cleanup that removed it covered the boot-chain
#: SCRIPTS, not the recipe archive. An exemption list naming 45 files is not a
#: lock, it is a formality, and the next reader would add the 46th. What has
#: to be impossible is REINTRODUCING the pin -- so the rule is that every
#: recipe written from the order onward must be clean, and the archive stays
#: an archive. A sed-derived chain copies an old recipe into a new file, and
#: a new file is exactly what this catches.
ORDER_EPOCH = 1787328000

#: The recipes the acceptance boots use, checked regardless of age.
FULL_FORM = ("argv_fullform_785.txt", "argv_bal321814.txt")


def test_there_are_recipes_to_check_or_the_test_says_so():
    """A guard that silently checks nothing is worse than no guard."""
    if not _recipes():
        pytest.skip("no acceptance recipes on this host")


def test_no_acceptance_recipe_pins_the_pool():
    recipes = _recipes()
    if not recipes:
        pytest.skip("no acceptance recipes on this host")
    offenders = []
    for path in recipes:
        if os.path.getmtime(path) < ORDER_EPOCH:
            continue  # archive; see ORDER_EPOCH
        with open(path) as handle:
            if PIN in handle.read():
                offenders.append(os.path.basename(path))
    assert not offenders, (
        f"{PIN} appears in {offenders}, written after the order that removed "
        f"it. The pool is planner-solved; a pin caps it silently and becomes "
        f"visible only once the solve would have exceeded it -- which is "
        f"exactly how 764512 profiled tokens were handed on as 430000."
    )


def test_the_full_form_recipes_are_clean():
    """The recipes the acceptance boots actually use, checked at any age."""
    for name in FULL_FORM:
        path = os.path.join(RECIPE_DIRS[0], name)
        if not os.path.isfile(path):
            pytest.skip(f"{name} not present on this host")
        with open(path) as handle:
            assert PIN not in handle.read(), f"{name} pins the pool"


def test_the_guard_would_catch_a_reintroduced_pin(tmp_path, monkeypatch):
    """CAN-FAIL PROOF. A guard nobody has seen fail is a guard nobody has."""
    recipe = tmp_path / "argv_new_acceptance.txt"
    recipe.write_text("--tp-size\n1\n" + PIN + "\n430000\n")
    os.utime(recipe, (ORDER_EPOCH + 60, ORDER_EPOCH + 60))
    monkeypatch.setattr("test_no_pool_pin_in_recipes_785.RECIPE_DIRS", (str(tmp_path),))
    import pytest as _pytest

    with _pytest.raises(AssertionError, match="planner-solved"):
        test_no_acceptance_recipe_pins_the_pool()
