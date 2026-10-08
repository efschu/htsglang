# SPDX-License-Identifier: Apache-2.0
"""The OPT subsystem family crosses the rename: ``<LEGACY>_OPT_<OLD>_X`` <-> ``FLLIPER_OPT_PDFLIP_X``.

NF release montage (01.10.): the renamed tree reads ``FLLIPER_OPT_PDFLIP_D_SEAT_VRAM``,
``..._DRAFT_PARK_EXACT_PIN`` and ``..._TAIL_READ_MMAP``, but ``canonical_env`` folded the
legacy spelling only onto ``FLLIPER_OPT_<OLD>_*`` (the generic pair), a name nobody reads --
an override in the old spelling never arrived. Old tokens are split like in ``name_compat``
so the test means the same before and after the mechanical rename.
"""

import importlib.util

import pytest

from sglang.srt import compat_shims
from sglang.srt import name_compat as nc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

LEG_OPT = "SG" "LANG_" "OPT_" "WE" "G2_"
NEW_OPT = "FLLIPER_OPT_PDFLIP_"
DEAD = "FLLIPER_OPT_" "WE" "G2_"           # what the generic pair produced: read by nobody
NAMES = ("D_SEAT_VRAM", "DRAFT_PARK_EXACT_PIN", "TAIL_READ_MMAP")


def _load_as(pkg: str):
    spec = importlib.util.spec_from_file_location(pkg + ".srt.name_compat", nc.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_opt_pair_exists_and_precedes_the_generic_one():
    pairs = list(nc.ENV_PREFIX_PAIRS)
    assert (LEG_OPT, NEW_OPT) in pairs
    gen = pairs.index(("SG" "LANG_", "FLLIPER_"))
    assert pairs.index((LEG_OPT, NEW_OPT)) < gen, "first match wins: the OPT family before the generic"
    assert gen == len(pairs) - 1


@pytest.mark.parametrize("name", NAMES)
def test_legacy_spelling_reaches_the_renamed_tree(name):
    m = _load_as("fl" "liper")
    assert m.CANONICAL_SIDE == 1
    assert m.canonical_env_name(LEG_OPT + name) == NEW_OPT + name
    env = {LEG_OPT + name: "1", "PATH": "/bin"}
    m.canonical_env(env)
    assert env == {NEW_OPT + name: "1", "PATH": "/bin"}
    assert DEAD + name not in env


@pytest.mark.parametrize("name", NAMES)
def test_renamed_spelling_reaches_the_legacy_tree(name):
    m = _load_as("sg" "lang")
    assert m.CANONICAL_SIDE == 0
    assert m.canonical_env_name(NEW_OPT + name) == LEG_OPT + name
    env = {NEW_OPT + name: "0"}
    m.canonical_env(env)
    assert env == {LEG_OPT + name: "0"}


@pytest.mark.parametrize("mod_pkg", ["sg" "lang", "fl" "liper"])
def test_renamed_value_wins_when_both_spellings_are_set(mod_pkg):
    m = _load_as(mod_pkg)
    c = m.CANONICAL_SIDE
    for name in NAMES:
        env = {LEG_OPT + name: "legacy", NEW_OPT + name: "renamed"}
        m.canonical_env(env)
        # F0-C: the renamed spelling wins on either side of the rename; the name left is the one the tree reads
        assert env == {(LEG_OPT, NEW_OPT)[c] + name: "renamed"}


def test_other_process_reader_accepts_both_spellings():
    for name in NAMES:
        assert compat_shims.env_name_variants(LEG_OPT + name) == (LEG_OPT + name, NEW_OPT + name)
        assert compat_shims.env_name_variants(NEW_OPT + name) == (NEW_OPT + name, LEG_OPT + name)


def test_generic_and_subsystem_families_are_unchanged():
    m = _load_as("fl" "liper")
    sub = "SG" "LANG_" "WE" "G2_"
    assert m.canonical_env_name(sub + "GROUP") == "FLLIPER_PDFLIP_GROUP"
    assert m.canonical_env_name("SG" "LANG_" "HICACHE_X") == "FLLIPER_HICACHE_X"
    assert m.canonical_env_name("SG" "LANG_" "OPT_OTHER") == "FLLIPER_OPT_OTHER"
