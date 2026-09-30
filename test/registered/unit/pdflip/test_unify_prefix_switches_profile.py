"""RG (operator 26.09.): the five prefix switches the 27B agent-load boot
dkr27brc10bar1agent09261821 (image rc11a, 26.09. 18:21-18:56Z) proved on
metal -- 108 requests, flips 0.23/request (before 1.35), wasted prefill ~7 %
(before 48 %), 0 group deaths, ENV-IM-RANG all five = 1 on P, D and the front
-- are MODEL PROFILE fields (pdflip/form.py PREFIX_SWITCHES), plus PF as a field
only (off, unproven on metal). qwen27b on; nextflash since HS 27.09. TK, PACED
and TW on (NF3), MZ and #49 off until the NF seat releases them. An explicitly
set env wins. P, D (build_env) and the front
(its env) get one value; MZ must be equal on P and D. No form: off.
"""

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import form as FM  # noqa: E402
from flliper.srt.pdflip import launcher as L  # noqa: E402
from flliper.srt.pdflip import profile_docker as PD  # noqa: E402

FIVE = (
    "FLLIPER_ANTHROPIC_INLINE_SYSTEM_IN_PLACE",  # MZ
    "FLLIPER_PDFLIP_TOLD_PROBE_TREE_KEY",  # TK
    "FLLIPER_PDFLIP_TOLD_PACED",  # PX2
    "FLLIPER_PDFLIP_P_TWIN_DEFER",  # TW
    "FLLIPER_PDFLIP_FRONT_SPAN_INFLIGHT",  # #49
)
PF = "FLLIPER_PDFLIP_TOLD_GROUP_FALLBACK"
ALL = FIVE + (PF,)
MZ = "FLLIPER_ANTHROPIC_INLINE_SYSTEM_IN_PLACE"
#: NF-TK (HS 27.09.): the told trio the nextflash row carries.
NF3 = ("FLLIPER_PDFLIP_TOLD_PROBE_TREE_KEY", "FLLIPER_PDFLIP_TOLD_PACED", "FLLIPER_PDFLIP_P_TWIN_DEFER")


def _nf_on(name):
    return name in NF3


def _form(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == "qwen27b"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return FM.PdFlipForm(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                       flip="family", vision="off", profile=profile, model="m")


@pytest.fixture
def clean(monkeypatch):
    from flliper.srt.name_compat import canonical_env_name

    for k in list(os.environ):
        if canonical_env_name(k) in ALL + ("FLLIPER_PDFLIP_TOLD_ABSOLUTE",):
            monkeypatch.delenv(k, raising=False)
    for k in (FM.FORM_ENV, "FLLIPER_PDFLIP_GROUP"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _build(group, profile, **kw):
    return L.build_env("/t", "/v", "0", "/s", False, "x", group=group, profile=profile,
                       boot_form=_form(profile) if profile else None, **kw)


# --- the registry ----------------------------------------------------------------------------


def test_the_registry_rows():
    q, n = FM.PROFILES["qwen27b"], FM.PROFILES["nextflash"]
    assert tuple(e for _f, e in FM.PREFIX_SWITCHES) == ALL
    for fld, env in FM.PREFIX_SWITCHES:
        assert getattr(q, fld) is (env != PF), env
        assert getattr(n, fld) is _nf_on(env), env
        assert FM.PROFILE_SWITCH_DEFAULTS["qwen27b"][env] is (env != PF), env
        assert FM.PROFILE_SWITCH_DEFAULTS["nextflash"][env] is _nf_on(env), env
    assert FM.PREFIX_SWITCH_P_EQ_D == (MZ,)


@pytest.mark.parametrize("profile,want", [("qwen27b", True), ("nextflash", False), (None, False)])
def test_state_per_profile(clean, profile, want):
    for name in FIVE:
        on, src = FM.prefix_switch_state(name, {}, profile)
        assert on is (want or (profile == "nextflash" and _nf_on(name))), name
        assert src == (f"profile {profile}" if profile else "no form (code default off)")
    assert FM.prefix_switch_state(PF, {}, "qwen27b") == (False, "profile qwen27b")


def test_state_reads_the_published_form(clean):
    env = {FM.FORM_ENV: _form("qwen27b").env_value()}
    assert FM.prefix_switch_state("FLLIPER_PDFLIP_TOLD_PACED", env) == (True, "profile qwen27b")
    env = {FM.FORM_ENV: _form("nextflash").env_value()}
    assert FM.prefix_switch_state("FLLIPER_PDFLIP_TOLD_PACED", env) == (True, "profile nextflash")
    assert FM.prefix_switch_state(MZ, env) == (False, "profile nextflash")


@pytest.mark.parametrize("profile", ["qwen27b", "nextflash", None])
@pytest.mark.parametrize("raw,want", [("1", True), ("0", False), ("on", True), ("false", False)])
def test_an_explicit_env_wins(profile, raw, want):
    for name in ALL:
        on, src = FM.prefix_switch_state(name, {name: raw}, profile)
        assert on is want and src == f"env {name}={raw}", (name, src)


def _other_spelling(name):
    # the name's spelling of the OTHER package generation (name_compat prefix pairs). Built at
    # run time, not written as a literal: a literal of the renamed spelling next to the legacy one
    # becomes the same word after the mechanical rename (rename_to_flliper.py refuses that as a
    # collision), and the test would then compare a name with itself.
    from flliper.srt.compat_shims import env_name_variants

    variants = env_name_variants(name)
    assert len(variants) == 2 and variants[1] != name, variants
    return variants[1]


def test_an_explicit_renamed_spelling_wins():
    # name_compat family: the other generation's spelling of a switch is the same switch
    paced = "FLLIPER_PDFLIP_TOLD_PACED"
    env = {_other_spelling(paced): "0", _other_spelling(MZ): "0"}
    assert FM.prefix_switch_state(paced, env, "qwen27b")[0] is False
    assert FM.prefix_switch_state(MZ, env, "qwen27b")[0] is False
    got = dict(env)
    FM.publish_prefix_switches(got, "qwen27b")
    assert paced not in got and MZ not in got


def test_blank_env_is_unset():
    assert FM.prefix_switch_state(MZ, {MZ: " "}, "qwen27b") == (True, "profile qwen27b")


def test_publish_writes_only_on_values():
    q = {}
    rows = FM.publish_prefix_switches(q, "qwen27b")
    assert q == {n: "1" for n in FIVE}
    assert [(n, on) for n, on, _ in rows] == [(n, n != PF) for n in ALL]
    nf = {"X": "y"}
    FM.publish_prefix_switches(nf, "nextflash")
    assert nf == dict({"X": "y"}, **{n: "1" for n in NF3})  # only the on rows are written
    none = {}
    FM.publish_prefix_switches(none, None)
    assert none == {}
    kept = {"FLLIPER_PDFLIP_TOLD_PACED": "0"}
    FM.publish_prefix_switches(kept, "qwen27b")
    assert kept["FLLIPER_PDFLIP_TOLD_PACED"] == "0"


def test_the_line_names_value_and_source():
    rows = FM.publish_prefix_switches({"FLLIPER_PDFLIP_P_TWIN_DEFER": "0"}, "qwen27b")
    line = FM.prefix_switches_line(rows)
    assert line.startswith("PDFLIP-PREFIX-SWITCHES ")
    assert "FLLIPER_PDFLIP_TOLD_PACED=1 (profile qwen27b)" in line
    assert "FLLIPER_PDFLIP_P_TWIN_DEFER=0 (env FLLIPER_PDFLIP_P_TWIN_DEFER=0)" in line
    assert f"{PF}=0 (profile qwen27b)" in line


# --- P, D and front get one value ------------------------------------------------------------


@pytest.mark.parametrize("group", ["P", "D"])
def test_build_env_27b_carries_the_five(clean, group):
    env = _build(group, "qwen27b")
    for n in FIVE:
        assert env.get(n) == "1", (group, n)
    assert PF not in env


@pytest.mark.parametrize("group", ["P", "D"])
def test_build_env_nf_differs_by_exactly_the_trio(clean, group):
    """nextflash: the group env differs from the same call with the prefix
    publish switched off by exactly TK, PACED and TW (NF-TK, HS 27.09.)."""
    with_rows = _build(group, "nextflash")
    for n in ALL:
        assert (with_rows.get(n) == "1") is _nf_on(n), n
    clean.setattr(FM, "publish_prefix_switches", lambda env, profile: [])
    without = _build(group, "nextflash")
    assert {k: v for k, v in with_rows.items() if k not in NF3} == without


def test_build_env_27b_differs_by_exactly_the_five(clean):
    with_rows = {"P": _build("P", "qwen27b"), "D": _build("D", "qwen27b")}
    clean.setattr(FM, "publish_prefix_switches", lambda env, profile: [])
    for g, env in with_rows.items():
        without = _build(g, "qwen27b")
        added = {k: v for k, v in env.items() if k not in without}
        assert added == {n: "1" for n in FIVE}, g
        assert {k: v for k, v in env.items() if k in without} == without, g


def test_desk_caller_without_form_publishes_nothing(clean):
    env = L.build_env("/t", "/v", "0", "/s", False, "x", group="P", profile="qwen27b")
    for n in ALL:
        assert n not in env


def test_explicit_env_wins_in_build_env(clean):
    clean.setenv("FLLIPER_PDFLIP_TOLD_PACED", "0")
    clean.setenv(PF, "1")
    for g in ("P", "D"):
        env = _build(g, "qwen27b")
        assert env["FLLIPER_PDFLIP_TOLD_PACED"] == "0"
        assert env[PF] == "1"
        assert env["FLLIPER_PDFLIP_P_TWIN_DEFER"] == "1"
    clean.setenv("FLLIPER_PDFLIP_TOLD_PACED", "1")
    assert _build("P", "nextflash")["FLLIPER_PDFLIP_TOLD_PACED"] == "1"


def test_env_p_env_d_apply_after_the_row(clean):
    env = _build("D", "qwen27b", group_env_extra={"FLLIPER_PDFLIP_P_TWIN_DEFER": "0"})
    assert env["FLLIPER_PDFLIP_P_TWIN_DEFER"] == "0"


@pytest.mark.parametrize("profile", ["qwen27b", "nextflash"])
def test_p_d_front_agree(clean, profile):
    """The front's env is dict(os.environ) plus the same publish call
    (launcher front spawn): P, D and the front hold one value per switch."""
    p, d = _build("P", profile), _build("D", profile)
    front = dict(os.environ)
    FM.publish_prefix_switches(front, profile)
    for n in ALL:
        assert p.get(n) == d.get(n) == front.get(n), (profile, n)
    assert (p.get(MZ) == "1") is (profile == "qwen27b")


def test_the_front_spawn_publishes_on_its_env():
    import inspect

    src = inspect.getsource(L)
    i = src.index('fenv["PYTHONUNBUFFERED"] = "1"')
    j = src.index("subprocess.Popen(front_argv, env=fenv")
    assert "pdflip_form.publish_prefix_switches(fenv, ns.pdflip_boot_form.profile)" in src[i:j]


# --- MZ: P == D --------------------------------------------------------------------------------


def test_mz_split_is_refused():
    base = {}
    FM.publish_prefix_switches(base, "qwen27b")
    assert FM.prefix_p_eq_d_mismatch(base, {}, {}) is None
    assert FM.prefix_p_eq_d_mismatch(base, {MZ: "1"}, {}) is None
    msg = FM.prefix_p_eq_d_mismatch(base, {}, {MZ: "0"})
    assert msg and "P=1 D=0" in msg and MZ in msg
    nf = {}
    FM.publish_prefix_switches(nf, "nextflash")
    assert FM.prefix_p_eq_d_mismatch(nf, {MZ: "1"}, {}) is not None
    assert FM.prefix_p_eq_d_mismatch(nf, {MZ: "1"}, {MZ: "true"}) is None
    # a non-MZ switch may differ per group (TW is PP0-only by design)
    assert FM.prefix_p_eq_d_mismatch(base, {}, {"FLLIPER_PDFLIP_P_TWIN_DEFER": "0"}) is None


def test_the_launcher_refuses_the_mz_split_and_names_the_line():
    import argparse

    ns = argparse.Namespace(profile="qwen27b", env_p="", env_d=f"{MZ}=0")
    with pytest.raises(L.PdFlipLaunchRefused, match="P=1 D=0"):
        L.prefix_switches_announce(ns, _form("qwen27b"), environ={})
    ns.env_d = ""
    line = L.prefix_switches_announce(ns, _form("qwen27b"), environ={})
    assert f"{MZ}=1 (profile qwen27b)" in line
    env = {MZ: "0"}
    line = L.prefix_switches_announce(ns, _form("qwen27b"), environ=env)
    assert f"{MZ}=0 (env {MZ}=0)" in line and env == {MZ: "0"}  # a copy, never written
    ns.env_p = f"{MZ}=1"
    with pytest.raises(L.PdFlipLaunchRefused, match="P=1 D=0"):
        L.prefix_switches_announce(ns, _form("nextflash"), environ={})


def test_main_announces_before_argparse_defaults():
    import inspect

    src = inspect.getsource(L.main)
    i = src.index("os.environ[pdflip_form.FORM_ENV] = boot_form.env_value()")
    j = src.index("apply_profile_arg_defaults(ns")
    assert "print(prefix_switches_announce(ns, boot_form), flush=True)" in src[i:j]


# --- profile_docker sees the fields --------------------------------------------------------------


def test_profile_docker_owns_the_switches():
    q = {f.key: f.value for f in PD.registry_facts("qwen27b", "int8")}
    n = {f.key: f.value for f in PD.registry_facts("nextflash", "int4-mixed")}
    for name in FIVE:
        assert q[f"_form {name}"] == "1" and n[f"_form {name}"] == ("1" if _nf_on(name) else "0")
    assert q[f"_form {PF}"] == "0" and n[f"_form {PF}"] == "0"
    # a profile that drops the lines is registry-only, one that states 0 is a DIFF
    rows = PD.compare(PD.registry_facts("qwen27b", "int8"), {})
    assert ("_form FLLIPER_PDFLIP_TOLD_PACED", "1", "-", "registry-only") in rows
    rows = PD.compare(PD.registry_facts("qwen27b", "int8"), {"_form FLLIPER_PDFLIP_TOLD_PACED": "0"})
    assert ("_form FLLIPER_PDFLIP_TOLD_PACED", "1", "0", "DIFF") in rows


# --- the rank-side readers follow the same row -----------------------------------------------


def _readers():
    from flliper.srt.environ import envs
    from flliper.srt.managers import pdflip_store_told as st
    from flliper.srt.managers import pdflip_told_fallback as fb
    from flliper.srt.pdflip import front as fr
    from flliper.srt.pdflip import p_twin_defer as tw

    return {
        MZ: lambda: bool(envs.FLLIPER_ANTHROPIC_INLINE_SYSTEM_IN_PLACE.get()),
        "FLLIPER_PDFLIP_TOLD_PROBE_TREE_KEY": st._tree_key_probe_armed,
        "FLLIPER_PDFLIP_TOLD_PACED": st._paced_env,
        "FLLIPER_PDFLIP_P_TWIN_DEFER": tw._env_on,
        "FLLIPER_PDFLIP_FRONT_SPAN_INFLIGHT": fr.front_span_inflight,
        PF: fb.env_on,
    }


@pytest.mark.parametrize("profile,want", [("qwen27b", True), ("nextflash", False), (None, False)])
def test_rank_readers_follow_the_profile(clean, profile, want):
    from flliper.srt.managers import pdflip_store_told as st

    if profile is not None:
        clean.setenv(FM.FORM_ENV, _form(profile).env_value())
    nf = profile == "nextflash"
    for name, read in _readers().items():
        assert read() is ((want and name != PF) or (nf and _nf_on(name))), (profile, name)
    # TK brings the absolute told along (its rank default follows TREE_KEY)
    assert st._absolute_armed() is (want or nf)


@pytest.mark.parametrize("profile", ["qwen27b", "nextflash", None])
@pytest.mark.parametrize("raw,want", [("1", True), ("0", False), ("on", True)])
def test_rank_readers_explicit_env_wins(clean, profile, raw, want):
    if profile is not None:
        clean.setenv(FM.FORM_ENV, _form(profile).env_value())
    for name in ALL:
        clean.setenv(name, raw)
    for name, read in _readers().items():
        if name in (MZ, "FLLIPER_PDFLIP_FRONT_SPAN_INFLIGHT") and raw == "on":
            continue  # EnvBool's own parser (environ.py) knows no "on": unchanged
        assert read() is want, (profile, name, raw)


def test_every_switch_has_one_profile_default_environ_entry():
    from flliper.srt import environ as env_mod

    for name in ALL:
        assert callable(getattr(env_mod.Envs, name).default), name
