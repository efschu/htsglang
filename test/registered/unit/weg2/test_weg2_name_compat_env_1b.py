# SPDX-License-Identifier: Apache-2.0
"""Rename step 1b: child environments carry ONE spelling per variable.

The rename moves env names to ``FLLIPER_*`` / ``FLLIPER_PDFLIP_*`` while
profiles and arms keep writing the legacy spelling for a while. The launcher
builds each group's env from ``dict(os.environ)`` and then POPS variables by
name (group identity, form, draft tier, PCIe duplex, ...), strips the
``PHASE_FLIP`` family and checks W116. A pop that removes one spelling while
another survives is undone in the child (its own import mirrors the survivor
back). ``name_compat.canonical_env`` folds every spelling onto the one the tree
reads before any of that happens.

All names below are built from ``name_compat``'s own prefixes, so the tests
mean the same before and after the mechanical rename.
"""

import importlib.util
import os
import re

import pytest

from sglang.srt import name_compat as nc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

(SUB_LEG, SUB_NEW), (GEN_LEG, GEN_NEW) = nc.ENV_PREFIX_PAIRS
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))


def canon(prefix_pair, rest):
    return prefix_pair[nc.CANONICAL_SIDE] + rest


def other(prefix_pair, rest):
    return prefix_pair[1 - nc.CANONICAL_SIDE] + rest


SUB = nc.ENV_PREFIX_PAIRS[0]
GEN = nc.ENV_PREFIX_PAIRS[1]


def _load_as(pkg: str):
    """name_compat loaded under another package name: the SAME file after the
    rename (``flliper.srt.name_compat``) -- the bridge's other direction."""
    spec = importlib.util.spec_from_file_location(pkg + ".srt.name_compat", nc.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# the function
# --------------------------------------------------------------------------


def test_prefix_table_matches_the_plan():
    assert SUB == ("SG" "LANG_" "WE" "G2_", "FLLIPER_PDFLIP_")
    assert GEN == ("SG" "LANG_", "FLLIPER_")
    # 0 before the mechanical rename, 1 after it (the same file bridges both ways)
    assert nc.CANONICAL_SIDE == (0 if nc.__name__.split(".")[0] == "sg" "lang" else 1)


def test_without_other_spellings_the_env_is_byte_identical():
    env = dict(os.environ)
    for k in list(env):  # this box's env, minus anything that is not canonical already
        if nc.canonical_env_name(k) != k or k in nc.FOREIGN_READERS:
            del env[k]
    env.update({canon(SUB, "GROUP"): "P", canon(GEN, "HICACHE_X"): "1", "SGL_ALIAS": "a",
                "HT" "SG" "LANG_PRODUCT": "b", "PATH": "/bin", canon(GEN, "RPF_N"): "8"})
    # a foreign reader (sgl-kernel) always keeps its legacy spelling beside the canonical one: after the
    # rename the fixed point carries both (before it, both are the same name)
    env[GEN[0] + "RPF_N"] = "8"
    before = list(env.items())
    out = nc.canonical_env(env)
    assert out is env
    assert list(env.items()) == before


@pytest.mark.parametrize("mod_pkg", ["sg" "lang", "flliper"])
def test_every_spelling_folds_onto_one_name(mod_pkg):
    m = _load_as(mod_pkg)
    s, g = m.ENV_PREFIX_PAIRS
    c = m.CANONICAL_SIDE
    assert c == (0 if mod_pkg == "sg" "lang" else 1)
    env = {s[1 - c] + "GROUP": "P", g[1 - c] + "HICACHE_X": "1", "PATH": "/bin"}
    m.canonical_env(env)
    assert env == {s[c] + "GROUP": "P", g[c] + "HICACHE_X": "1", "PATH": "/bin"}


@pytest.mark.parametrize("mod_pkg", ["sg" "lang", "flliper"])
def test_canonical_value_wins_and_no_name_is_doubled(mod_pkg):
    m = _load_as(mod_pkg)
    s, g = m.ENV_PREFIX_PAIRS
    c = m.CANONICAL_SIDE
    env = {s[0] + "FORM": "legacy", s[1] + "FORM": "renamed", g[0] + "X": "l", g[1] + "X": "r"}
    m.canonical_env(env)
    assert env == {s[c] + "FORM": ("legacy", "renamed")[c], g[c] + "X": ("l", "r")[c]}
    fams = {}
    for k in env:
        fams.setdefault(m.canonical_env_name(k), []).append(k)
    assert all(len(v) == 1 for v in fams.values()), fams


@pytest.mark.parametrize("mod_pkg", ["sg" "lang", "flliper"])
def test_foreign_readers_keep_the_legacy_spelling(mod_pkg):
    m = _load_as(mod_pkg)
    s, g = m.ENV_PREFIX_PAIRS
    for leg in sorted(m.FOREIGN_READERS):
        pair = s if leg.startswith(s[0]) else g
        rest = leg[len(pair[0]):]
        for set_as in (pair[0] + rest, pair[1] + rest):
            env = {set_as: "7"}
            m.canonical_env(env)
            assert env.get(pair[0] + rest) == "7", (leg, set_as)  # what the foreign code reads
            assert env.get(m.canonical_env_name(leg)) == "7"      # what the tree reads
            assert set(env) <= {pair[0] + rest, pair[1] + rest}
            assert set(env.values()) == {"7"}


def test_foreign_list_is_preserved_by_an_explicit_empty_keep():
    env = {other(GEN, "RPF_N"): "3"}
    nc.canonical_env(env, foreign_keep=frozenset())
    assert env == {canon(GEN, "RPF_N"): "3"}
    assert GEN[0] + "RPF_N" in nc.FOREIGN_READERS


def test_foreign_list_covers_every_legacy_getenv_outside_the_rename():
    """Every C/C++/CUDA ``getenv("<legacy>...")`` in the tree and in sgl-kernel
    (the rename leaves those files alone) is on the list."""
    rx = re.compile(r'getenv\(\s*"(%s[A-Z0-9_]+)"' % re.escape(GEN[0]))
    ext = (".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".inl")
    found = set()
    for top in ("python", "sgl-kernel"):
        for dp, _dn, fn in os.walk(os.path.join(ROOT, top)):
            for f in fn:
                if f.endswith(ext):
                    with open(os.path.join(dp, f), errors="replace") as fh:
                        found.update(rx.findall(fh.read()))
    if not found:
        pytest.skip("no C/C++ sources in this checkout")
    assert found <= nc.FOREIGN_READERS, sorted(found - nc.FOREIGN_READERS)


# --------------------------------------------------------------------------
# the callers
# --------------------------------------------------------------------------


def _launcher():
    from sglang.srt.weg2 import launcher

    return launcher


def _build(**kw):
    return _launcher().build_env(tree="/tmp/t", venv="/tmp/v", cvd="0", store_dir="/tmp/s",
                                 debug_hold=False, tag="probe", **kw)


def _popped_names():
    """The literal names ``build_env`` pops (``env.pop("<name>"...)``)."""
    import inspect

    src = inspect.getsource(_launcher().build_env)
    return sorted(set(re.findall(r'env\.pop\("([A-Z0-9_]+)"', src)))


def test_no_popped_name_is_a_foreign_reader():
    names = _popped_names()
    assert len(names) >= 5, names
    assert not [n for n in names if nc.canonical_env_name(n) in
                {nc.canonical_env_name(f) for f in nc.FOREIGN_READERS}]


def _build_with(monkeypatch, names, spelling):
    with monkeypatch.context() as mp:
        for n in names:
            pair = SUB if n.startswith(SUB[nc.CANONICAL_SIDE]) else GEN
            rest = n[len(pair[nc.CANONICAL_SIDE]):]
            mp.delenv(canon(pair, rest), raising=False)
            mp.delenv(other(pair, rest), raising=False)
            if spelling:
                # the draft tier is validated (auto|on|off, and off conflicts with a P draft producer)
                value = "auto" if n == _launcher().HICACHE_DRAFT_TIER_ENV else "inherited"
                mp.setenv(canon(pair, rest) if spelling == "canonical" else other(pair, rest), value)
        return _build(group="")


def test_a_pop_acts_on_either_spelling(monkeypatch):
    """Every name build_env pops, inherited once in the tree's spelling and
    once in the other one: the two child envs are identical, a name the
    launcher does not publish is gone in both, and no other spelling is left."""
    L = _launcher()
    names = _popped_names() + [L.HICACHE_DRAFT_TIER_ENV]
    none = _build_with(monkeypatch, names, None)
    can = _build_with(monkeypatch, names, "canonical")
    oth = _build_with(monkeypatch, names, "other")
    assert oth == can
    for n in names:
        if n not in none:
            assert n not in can, n
    assert not [k for k in oth if nc.canonical_env_name(k) != k and k not in nc.FOREIGN_READERS]


def test_group_identity_pop_with_the_other_spelling(monkeypatch):
    rest = "GROUP"
    monkeypatch.delenv(canon(SUB, rest), raising=False)
    monkeypatch.setenv(other(SUB, rest), "P")
    env = _build(group="")
    assert canon(SUB, rest) not in env and other(SUB, rest) not in env
    env = _build(group="D")
    assert env[canon(SUB, rest)] == "D"
    assert other(SUB, rest) not in env


def test_phase_flip_strip_catches_the_other_spelling(monkeypatch):
    monkeypatch.setenv(other(GEN, "PHASE_FLIP_SOMETHING"), "1")
    monkeypatch.setenv(canon(GEN, "PHASE_FLIP_OTHER"), "1")
    env = _build(group="P")
    assert not [k for k in env if "PHASE_FLIP" in k]


def test_operator_group_env_is_canonical_and_wins(monkeypatch):
    L = _launcher()
    monkeypatch.setenv(canon(GEN, "MOE_SCRATCH_SLOTS"), "1,1,1")
    spec = "%s=74,48,48;%s=on" % (other(GEN, "MOE_SCRATCH_SLOTS"), other(SUB, "SOME_KNOB"))
    extra = L.parse_group_env(spec)
    assert extra == {canon(GEN, "MOE_SCRATCH_SLOTS"): "74,48,48", canon(SUB, "SOME_KNOB"): "on"}
    env = _build(group="D", group_env_extra={other(GEN, "MOE_SCRATCH_SLOTS"): "74,48,48"})
    assert env[canon(GEN, "MOE_SCRATCH_SLOTS")] == "74,48,48"
    assert other(GEN, "MOE_SCRATCH_SLOTS") not in env


def test_w116_judges_the_canonical_name():
    from sglang.srt import flip_nextflash_groups as fg

    owned = sorted(fg.GROUP_OWNED_ENV)[0]
    rest = owned[len(GEN[nc.CANONICAL_SIDE]):]
    with pytest.raises(fg.Weg2FlipGroupEnvInherited):
        fg.build_group_env("P", {other(GEN, rest): "1"})
    with pytest.raises(fg.Weg2FlipGroupEnvInherited):
        fg.build_group_env("P", {}, extra={other(GEN, rest): "1"})
    plan = fg.build_group_env("P", {"PATH": "/bin", other(SUB, "X"): "1"})
    assert canon(SUB, "X") in plan.env and other(SUB, "X") not in plan.env


def test_vision_probe_env_pops_every_spelling(monkeypatch):
    from sglang.srt.weg2 import vision_stage_boot as vb

    monkeypatch.setenv(other(SUB, "TMS_PRELOAD_SO"), "/x.so")
    _argv, env = vb.context_probe_command(uuid="GPU-0")
    assert not [k for k in env if nc.canonical_env_name(k) == canon(SUB, "TMS_PRELOAD_SO")]


def test_shell_statements_for_the_entrypoint():
    """The container entrypoint cannot import the package: it runs the file as
    a script (``--shell``) and evals the lines. Run as a script, the file
    derives its package from its own path (``<pkg>/srt/name_compat.py``)."""
    import subprocess
    import sys

    base = {"PATH": os.environ.get("PATH", "/bin")}

    def run(extra):
        out = subprocess.run([sys.executable, nc.__file__, "--shell"], env={**base, **extra},
                             capture_output=True, text=True, check=True).stdout
        return [line for line in out.split("\n") if line]

    assert run({canon(SUB, "GROUP"): "P"}) == []
    lines = run({other(SUB, "GROUP"): "P", other(GEN, "RPF_N"): "3"})
    assert "unset " + other(SUB, "GROUP") in lines
    assert "export %s=P" % canon(SUB, "GROUP") in lines
    assert "export %s=3" % canon(GEN, "RPF_N") in lines
    assert nc.shell_statements({canon(GEN, "X"): "a b"}) == []
    assert nc.shell_statements({other(GEN, "X"): "a b"}) == [
        "unset " + other(GEN, "X"), "export %s='a b'" % canon(GEN, "X")]
