"""H88-D (plan PLAN-H88-W4A8-1007 D4, A4): the persistent L3 identity under the W4A8 switch.

Rules of the NF seat (memory nf-h88-w4a8-und-1243-auftrag-1007):
 (a) W4A8 (MoE activations rounded to int8) changes the identity -- one full recompute, no reuse of old pages;
 (b) NOTHING else changes it: the running store ``...-abl-wxp-dc6a8c2062`` keeps its directory;
 (c) nobody sweeps, renames or invalidates the store, and ``generation`` (``L3_PERSIST_GENERATION`` = "706") is NOT raised.

Mechanism under test: the field ``moe_act`` is added to ``launcher.l3_persist_identity`` (launcher side, directory digest and
W57) and ``moe_act=int8`` to ``hicache_storage.compute_model_identity_hash`` (key suffix, PD handshake, L3 rank identity and
W165) ONLY while the switch (flag ``--moe-act-int8 on`` / env ``SGLANG_MOE_ACT_INT8=1``, registered by H88-E, read defensively
here) is on. Off, both are byte-identical.

Pins come from ``fixtures/h88d_1007/nf_abl_l3_identity.json`` (read-only copies of the live store's identity files, never the
live mounts). The abl profile's own ``--extra-p/--extra-d/--profile/--weg2-vision`` come from the AP0 launch snapshot
``fixtures/planer_1006/golden/launch_nf-int4-h6-abl.json``.
"""
from __future__ import annotations

import hashlib
import json
import os
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache import hicache_storage as HS  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = json.load(open(os.path.join(HERE, "fixtures", "h88d_1007", "nf_abl_l3_identity.json")))
LAUNCH = json.load(open(os.path.join(HERE, "fixtures", "planer_1006", "golden", "launch_nf-int4-h6-abl.json")))

LIVE_IDENT = FIX["identity"]
LIVE_DIGEST = "dc6a8c2062"  # the plan's pin (running boot), also FIX["digest10"]
LIVE_RANK_HASH = "9341197366e3616e"  # model_identity in the live L3_RANK_IDENTITY.<group>.json


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def _abl_derived_identity(tmp_path, **kw):
    """l3_persist_identity from the abl profile's real argv values, on a stub checkpoint dir that carries the live model's
    basename. Fields only the live mount knows (model_path, config sha, weights fingerprint) are then set to the pinned live
    values, so the result is what the launcher computes at the Metall boot -- derived parts (profile, form_kv, kv dtype,
    override shas, vision, generation) really come from the profile argv."""
    argv = LAUNCH["argv"]
    model_dir = tmp_path / os.path.basename(LIVE_IDENT["model_path"])
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "stub"}))
    (model_dir / "model-00001.safetensors").write_bytes(b"\0" * 16)
    args = dict(extra_p=_flag(argv, "--extra-p"), extra_d=_flag(argv, "--extra-d"), vision=_flag(argv, "--weg2-vision"),
                env_p=_flag(argv, "--env-p"), env_d=_flag(argv, "--env-d"), env={})
    args.update(kw)
    ident = L.l3_persist_identity(str(model_dir), _flag(argv, "--profile"), "", **args)
    for k in ("model_path", "model_config_sha1", "weights_fp"):
        ident[k] = LIVE_IDENT[k]
    return ident


@pytest.fixture(autouse=True)
def _no_switch_in_the_environment(monkeypatch):
    monkeypatch.delenv("SGLANG_MOE_ACT_INT8", raising=False)


# ---------------------------------------------------------------- (1) default identity = the running store
def test_pin_the_running_store_dict_hashes_to_dc6a8c2062():
    assert hashlib.sha1(json.dumps(LIVE_IDENT, sort_keys=True).encode()).hexdigest()[:10] == LIVE_DIGEST == FIX["digest10"]
    assert L.l3_persist_dir_name(LIVE_IDENT) == FIX["dir_name"]
    assert FIX["dir_name"].endswith("-abl-wxp-dc6a8c2062")


def test_release_profile_nf_int4_h6_abl_default_identity_is_dc6a8c2062(tmp_path):
    ident = _abl_derived_identity(tmp_path)
    assert "moe_act" not in ident
    assert ident == LIVE_IDENT, {k: (ident.get(k), LIVE_IDENT.get(k)) for k in set(ident) | set(LIVE_IDENT)
                                 if ident.get(k) != LIVE_IDENT.get(k)}
    assert L.l3_persist_dir_name(ident) == FIX["dir_name"]


def test_the_abl_profile_itself_does_not_switch_it_on():
    argv = LAUNCH["argv"]
    for name in ("--extra-p", "--extra-d", "--env-p", "--env-d"):
        assert "moe-act-int8" not in _flag(argv, name) and "MOE_ACT_INT8" not in _flag(argv, name), name
    assert not any(k.startswith("SGLANG_MOE_ACT") for k in LAUNCH["env"])
    assert not L.l3_moe_act_active(_flag(argv, "--extra-p"), _flag(argv, "--env-p"), None, {})
    assert not L.l3_moe_act_active(_flag(argv, "--extra-d"), _flag(argv, "--env-d"), None, {})


def test_other_inputs_that_must_not_move_the_identity_do_not(tmp_path):
    """Rule (b): a switch that is absent or OFF in any spelling, in any source, is the default identity."""
    base = L.l3_persist_dir_name(_abl_derived_identity(tmp_path))
    for i, kw in enumerate((
        dict(moe_act_int8="off"), dict(moe_act_int8=False), dict(moe_act_int8=""), dict(moe_act_int8=None),
        dict(moe_act_int8="0"), dict(env={"SGLANG_MOE_ACT_INT8": "0"}), dict(env={"SGLANG_MOE_ACT_INT8": "off"}),
        dict(env={"SGLANG_MOE_ACT_INT8": ""}), dict(env={}),
        # not the runtime's spelling: EnvBool does not know on/int8 (unparsable -> default OFF at runtime), the flag is == "on"
        dict(env={"SGLANG_MOE_ACT_INT8": "on"}), dict(env={"SGLANG_MOE_ACT_INT8": "int8"}),
        dict(moe_act_int8="1"), dict(moe_act_int8="true"), dict(moe_act_int8="int8"),
    )):
        sub = tmp_path / f"v{i}"
        sub.mkdir()
        assert L.l3_persist_dir_name(_abl_derived_identity(sub, **kw)) == base, kw


# ---------------------------------------------------------------- (2) switch on -> another identity
@pytest.mark.parametrize("how", ["extra-flag", "extra-flag-eq", "group-env", "ns-flag-on", "ns-flag-true", "process-env"])
def test_switch_on_gives_another_hash_in_every_source(tmp_path, how):
    argv = LAUNCH["argv"]
    ep, ed = _flag(argv, "--extra-p"), _flag(argv, "--extra-d")
    vp, vd = _flag(argv, "--env-p"), _flag(argv, "--env-d")
    model_dir = tmp_path / os.path.basename(LIVE_IDENT["model_path"])
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    kw = dict(vision="transient", env_p=vp, env_d=vd, env={})
    if how == "extra-flag":
        ep, ed = ep + " --moe-act-int8 on", ed + " --moe-act-int8 on"
    elif how == "extra-flag-eq":
        ep, ed = ep + " --moe-act-int8=on", ed + " --moe-act-int8=on"
    elif how == "group-env":
        kw["env_p"], kw["env_d"] = vp + ";SGLANG_MOE_ACT_INT8=1", vd + ";SGLANG_MOE_ACT_INT8=1"
    elif how == "ns-flag-on":
        kw["moe_act_int8"] = "on"
    elif how == "ns-flag-true":
        kw["moe_act_int8"] = True
    elif how == "process-env":
        kw["env"] = {"SGLANG_MOE_ACT_INT8": "1"}
    on = L.l3_persist_identity(str(model_dir), "nextflash", "", extra_p=ep, extra_d=ed, **kw)
    off = L.l3_persist_identity(str(model_dir), "nextflash", "", extra_p=_flag(argv, "--extra-p"),
                                extra_d=_flag(argv, "--extra-d"), vision="transient", env_p=vp, env_d=vd, env={})
    assert on["moe_act"] == "int8"
    assert "moe_act" not in off
    assert L.l3_persist_dir_name(on) != L.l3_persist_dir_name(off)
    assert {k: v for k, v in on.items() if k != "moe_act"} == off  # exactly one key more, nothing else moves


def test_switch_on_on_the_live_identity_is_not_dc6a8c2062_and_names_the_field(tmp_path):
    ident = _abl_derived_identity(tmp_path, moe_act_int8="on")
    assert ident["moe_act"] == "int8"
    assert L.l3_persist_dir_name(ident) != FIX["dir_name"]
    assert not L.l3_persist_dir_name(ident).endswith(LIVE_DIGEST)
    assert L.l3_persist_dir_name(ident).startswith("l3-nextflash-Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp-")
    assert ident == dict(LIVE_IDENT, moe_act="int8")


def test_p_and_d_asymmetric_switch_is_refused_by_name_never_a_third_identity(tmp_path):
    """Review finding 3: the rank key suffix is per group, so P on / D off would make D miss every page P wrote."""
    argv = LAUNCH["argv"]
    on = ";SGLANG_MOE_ACT_INT8=1"
    for n in ("p", "d", "pe", "de"):
        (tmp_path / n).mkdir()
    cases = {
        "p": dict(env_p=_flag(argv, "--env-p") + on),
        "d": dict(env_d=_flag(argv, "--env-d") + on),
        "pe": dict(extra_p=_flag(argv, "--extra-p") + " --moe-act-int8 on"),
        "de": dict(extra_d=_flag(argv, "--extra-d") + " --moe-act-int8=on"),
    }
    for n, kw in cases.items():
        with pytest.raises(SystemExit) as e:
            _abl_derived_identity(tmp_path / n, **kw)
        msg = str(e.value)
        assert "REFUSED" in msg and "P and D must agree" in msg and "SGLANG_MOE_ACT_INT8" in msg, n
        with pytest.raises(SystemExit):  # the boot-wide check main() runs for every boot (persistent L3 or not)
            L.l3_moe_act_resolve(kw.get("extra_p", _flag(argv, "--extra-p")), kw.get("extra_d", _flag(argv, "--extra-d")),
                                 kw.get("env_p", _flag(argv, "--env-p")), kw.get("env_d", _flag(argv, "--env-d")), None, {})
    # both groups on (any source) is the one valid "on"; neither is the default
    (tmp_path / "ok1").mkdir()
    (tmp_path / "ok2").mkdir()
    assert _abl_derived_identity(tmp_path / "ok1", moe_act_int8="on")["moe_act"] == "int8"
    both = _abl_derived_identity(tmp_path / "ok2", env_p=_flag(argv, "--env-p") + on, env_d=_flag(argv, "--env-d") + on)
    assert both["moe_act"] == "int8"
    # a d_only boot has no group P: only D's value counts, and a P-only switch is no asymmetry there
    assert L.l3_moe_act_resolve("", "--moe-act-int8 on", "", "", None, {}, d_only=True) is True
    assert L.l3_moe_act_resolve("--moe-act-int8 on", "", "", "", None, {}, d_only=True) is False
    assert L.l3_moe_act_resolve("", "", "", "", None, {}) is False


# ---------------------------------------------------------------- group env beats process env (fix round 2, finding 1)
def _runtime_group_env(process_env, group_spec):
    """What the group's rank really sees: build_env applies --env-p/--env-d LAST over the process env."""
    merged = dict(process_env)
    merged.update(L.parse_group_env(group_spec))
    return L.l3_moe_act_active.__globals__["_moe_act_switch"].env_value_on(merged.get("SGLANG_MOE_ACT_INT8"))


def test_group_env_decides_alone_when_it_carries_the_key():
    off, on = "SGLANG_MOE_ACT_INT8=0", "SGLANG_MOE_ACT_INT8=1"
    glob = {"SGLANG_MOE_ACT_INT8": "1"}
    assert L.l3_moe_act_active("", off, None, glob) is False   # global export 1, group says 0 -> the group runs OFF
    assert L.l3_moe_act_active("", on, None, {}) is True
    assert L.l3_moe_act_active("", on, None, {"SGLANG_MOE_ACT_INT8": "0"}) is True  # group 1 beats global 0
    assert L.l3_moe_act_active("", "FOO=1", None, glob) is True  # key absent in the group env -> process env counts
    assert L.l3_moe_act_active("", "", None, glob) is True
    # an unparsable group value is OFF at runtime and alone decides, however the process env reads
    assert L.l3_moe_act_active("", "SGLANG_MOE_ACT_INT8=int8", None, glob) is False
    # launcher == runtime merge for every combination
    for pv in (None, "0", "1", "y", "off"):
        for gv in (None, "0", "1", "y", "off", "int8"):
            penv = {} if pv is None else {"SGLANG_MOE_ACT_INT8": pv}
            spec = "" if gv is None else f"SGLANG_MOE_ACT_INT8={gv}"
            assert L.l3_moe_act_active("", spec, None, penv) is _runtime_group_env(penv, spec), (pv, gv)


def test_process_env_on_with_one_group_overridden_to_off_is_refused(tmp_path):
    argv = LAUNCH["argv"]
    glob = {"SGLANG_MOE_ACT_INT8": "1"}
    off = ";SGLANG_MOE_ACT_INT8=0"
    for n, (ep, ed) in {"p": (off, ""), "d": ("", off)}.items():
        (tmp_path / n).mkdir()
        kp, kd = _flag(argv, "--env-p") + ep, _flag(argv, "--env-d") + ed
        with pytest.raises(SystemExit) as e:
            L.l3_moe_act_resolve("", "", kp, kd, None, glob)
        assert "REFUSED" in str(e.value) and "P and D must agree" in str(e.value), n
        with pytest.raises(SystemExit):
            _abl_derived_identity(tmp_path / n, env_p=kp, env_d=kd, env=glob)


def test_process_env_on_and_both_groups_overridden_to_off_is_the_default_identity(tmp_path):
    argv = LAUNCH["argv"]
    glob = {"SGLANG_MOE_ACT_INT8": "1"}
    off = ";SGLANG_MOE_ACT_INT8=0"
    kp, kd = _flag(argv, "--env-p") + off, _flag(argv, "--env-d") + off
    assert L.l3_moe_act_resolve("", "", kp, kd, None, glob) is False
    ident = _abl_derived_identity(tmp_path, env_p=kp, env_d=kd, env=glob)
    assert "moe_act" not in ident and ident == LIVE_IDENT
    assert L.l3_persist_dir_name(ident) == FIX["dir_name"]


def test_process_env_on_and_both_groups_on_is_the_switch_identity(tmp_path):
    argv = LAUNCH["argv"]
    on = ";SGLANG_MOE_ACT_INT8=1"
    ident = _abl_derived_identity(tmp_path, env_p=_flag(argv, "--env-p") + on, env_d=_flag(argv, "--env-d") + on,
                                  env={"SGLANG_MOE_ACT_INT8": "1"})
    assert ident["moe_act"] == "int8"


# ---------------------------------------------------------------- one predicate = the runtime's reading (review finding 2)
SPELLINGS = ["1", "0", "true", "True", "TRUE", "false", "yes", "YES", "y", "Y", "no", "n", "N", "on", "off", "int8", "", " 1", "2"]


def _envbool(v):
    from sglang.srt.environ import EnvBool

    try:
        return EnvBool(False).parse(v)
    except ValueError:
        return False  # EnvField.get(): an unparsable value warns and falls back to the default (OFF)


@pytest.mark.parametrize("v", SPELLINGS)
def test_env_spelling_launcher_rank_and_runtime_envbool_agree(v, monkeypatch):
    runtime = _envbool(v)
    assert L.l3_moe_act_active("", f"SGLANG_MOE_ACT_INT8={v}", None, {}) is runtime or v == " 1"  # the spec parser strips
    assert L.l3_moe_act_active("", "", None, {"SGLANG_MOE_ACT_INT8": v}) is runtime
    monkeypatch.setenv("SGLANG_MOE_ACT_INT8", v)
    assert HS.moe_act_int8_active(_sa()) is runtime
    # y/Y are on at runtime: launcher identity AND rank identity must move, not stay default
    if runtime:
        assert HS.compute_model_identity_hash(_sa()) != LIVE_RANK_HASH


@pytest.mark.parametrize("v", ["on", "ON", "off", "1", "true", "y", "", "int8"])
def test_flag_spelling_launcher_and_rank_agree_with_the_runtime_eq_on(v):
    runtime = str(v).lower() == "on"  # moe_act_int8._requested: str(server_args.moe_act_int8).lower() == "on"
    assert L.l3_moe_act_active(f"--moe-act-int8 {v}" if v else "", "", None, {}) is runtime
    assert L.l3_moe_act_active("", "", v, {}) is runtime
    assert HS.moe_act_int8_active(_sa(moe_act_int8=v)) is runtime


def test_y_in_both_group_envs_moves_launcher_and_rank_identity_alike(tmp_path, monkeypatch):
    argv = LAUNCH["argv"]
    for y in ("y", "Y"):
        sub = tmp_path / y
        sub.mkdir()
        ident = _abl_derived_identity(sub, env_p=_flag(argv, "--env-p") + f";SGLANG_MOE_ACT_INT8={y}",
                                      env_d=_flag(argv, "--env-d") + f";SGLANG_MOE_ACT_INT8={y}")
        assert ident["moe_act"] == "int8"
        monkeypatch.setenv("SGLANG_MOE_ACT_INT8", y)
        assert HS.compute_model_identity_hash(_sa(), include_parallel_vectors=False) != LIVE_RANK_HASH


def test_the_two_predicates_are_one_module():
    from sglang.srt.environ import EnvBool  # noqa: F401
    from sglang.srt.weg2 import moe_act_switch as M

    assert HS._moe_act_switch is M and L._moe_act_switch is M
    assert [v for v in SPELLINGS if M.env_value_on(v)] == [v for v in SPELLINGS if _envbool(v) and v != " 1"]
    assert L.L3_MOE_ACT_ENV == HS.MOE_ACT_INT8_ENV == M.MOE_ACT_INT8_ENV == "SGLANG_MOE_ACT_INT8"
    assert M.MOE_ACT_INT8_FLAG == L.L3_MOE_ACT_FLAG == "--moe-act-int8"


def test_p_and_d_rank_identity_is_one_value_when_the_switch_is_symmetric(monkeypatch):
    """P and D ranks of one boot compute the key suffix from the same inputs -> the same hash (the launcher refuses the rest)."""
    monkeypatch.setenv("SGLANG_MOE_ACT_INT8", "y")
    assert HS.compute_model_identity_hash(_sa()) == HS.compute_model_identity_hash(_sa())


def test_a_malformed_group_env_is_off_here_the_refusal_is_where_it_is_parsed():
    assert L.l3_moe_act_active("", "NOT-A-PAIR", None, {}) is False


# ---------------------------------------------------------------- (3) generation unchanged (rule c)
def test_generation_stays_706_with_the_switch_on_and_off(tmp_path):
    assert L.L3_PERSIST_GENERATION == "706"
    assert LIVE_IDENT["generation"] == "706"
    assert _abl_derived_identity(tmp_path)["generation"] == "706"
    sub = tmp_path / "on"
    sub.mkdir()
    assert _abl_derived_identity(sub, moe_act_int8="on")["generation"] == "706"


def test_generation_is_the_only_place_706_is_set_and_the_source_does_not_touch_it():
    src = open(L.__file__).read()
    assert src.count('L3_PERSIST_GENERATION = "706"') == 1
    assert 'ident["generation"]' not in src and "L3_PERSIST_GENERATION =" in src


# ---------------------------------------------------------------- (4) rank-side identity
def _sa(**over):
    d = dict(FIX["rank_server_args"], rank_tp_ratio=None, rank_kv_ratio=None, json_model_override_args='{"language_model_only":true}')
    d.update(over)
    return types.SimpleNamespace(**d)


def test_rank_identity_default_equals_the_live_L3_RANK_IDENTITY_value(tmp_path):
    sa = _sa(model_path=str(tmp_path))  # stub dir: only model_identity is compared
    assert HS.compute_model_identity_hash(_sa(), include_parallel_vectors=False) == LIVE_RANK_HASH == FIX["rank_identity"]["model_identity"]
    assert HS.l3_rank_identity(sa)["model_identity"] != LIVE_RANK_HASH  # other model_path -> proves the path is an input
    # the recipe, written out independently of the function (byte-identical to the upstream recipe)
    parts = [FIX["rank_server_args"]["model_path"], "", "auto", "", "fp8_e4m3"]
    assert hashlib.sha256("|".join(parts).encode()).hexdigest()[:16] == LIVE_RANK_HASH
    assert hashlib.sha1(b'{"language_model_only":true}').hexdigest()[:16] == FIX["rank_identity"]["override_sha"]


def test_rank_identity_with_the_switch_differs_attr_and_env(monkeypatch):
    base_nov = HS.compute_model_identity_hash(_sa(), include_parallel_vectors=False)
    base_vec = HS.compute_model_identity_hash(_sa())
    assert base_nov == base_vec == LIVE_RANK_HASH  # no vectors set -> the two forms agree
    for attr in ("on", "ON", True):
        on = HS.compute_model_identity_hash(_sa(moe_act_int8=attr), include_parallel_vectors=False)
        assert on != base_nov, attr
        assert on == HS.compute_model_identity_hash(_sa(moe_act_int8=attr))
    # recipe with the extra part, written out independently
    parts = [FIX["rank_server_args"]["model_path"], "", "auto", "", "fp8_e4m3", "moe_act=int8"]
    assert HS.compute_model_identity_hash(_sa(moe_act_int8="on")) == hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    for off in ("off", "0", "", None, False, "1", "true", "int8"):
        assert HS.compute_model_identity_hash(_sa(moe_act_int8=off), include_parallel_vectors=False) == base_nov, off
    monkeypatch.setenv("SGLANG_MOE_ACT_INT8", "1")
    assert HS.compute_model_identity_hash(_sa(), include_parallel_vectors=False) != base_nov
    monkeypatch.setenv("SGLANG_MOE_ACT_INT8", "0")
    assert HS.compute_model_identity_hash(_sa(), include_parallel_vectors=False) == base_nov


def test_rank_identity_after_the_parallel_vectors_stays_default_byte_identical():
    """An uneven-TP store (vectors set) keeps its key with the switch off; the switch part is appended after them."""
    sa = _sa(rank_tp_ratio="183,137,168")
    plain = HS.compute_model_identity_hash(sa)
    parts = [FIX["rank_server_args"]["model_path"], "", "auto", "", "fp8_e4m3", "rank_tp_ratio=183,137,168"]
    assert plain == hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    assert HS.compute_model_identity_hash(_sa(rank_tp_ratio="183,137,168", moe_act_int8="on")) != plain
    # a MagicMock ServerArgs (attribute auto-exists) is NOT read as "on"
    from unittest import mock
    assert HS.moe_act_int8_active(mock.MagicMock()) is False


def test_rank_identity_record_moves_with_the_switch(tmp_path):
    (tmp_path / "model-00001.safetensors").write_bytes(b"\0" * 16)
    off = HS.l3_rank_identity(_sa(model_path=str(tmp_path)))
    on = HS.l3_rank_identity(_sa(model_path=str(tmp_path), moe_act_int8="on"))
    assert set(off) == set(on) == {"model_identity", "override_sha", "weights_fp"}  # same record shape
    assert off["model_identity"] != on["model_identity"]
    assert off["override_sha"] == on["override_sha"] and off["weights_fp"] == on["weights_fp"]


# ---------------------------------------------------------------- W57 / W165 paths
def test_w57_a_store_of_the_default_identity_refuses_a_switch_on_boot_and_back(tmp_path):
    d = str(tmp_path / "store")
    default = dict(LIVE_IDENT)
    on = dict(LIVE_IDENT, moe_act="int8")
    assert L.l3_persist_check_identity(d, default, dry=False) == "new"
    assert L.l3_persist_check_identity(d, default, dry=False) == "match"
    with pytest.raises(L.Weg2StoreDiskRefused) as e:
        L.l3_persist_check_identity(d, on, dry=False)
    assert "W57" in str(e.value) and "moe_act" in str(e.value)
    # the store itself was not touched by the refusal (rule c)
    assert json.load(open(os.path.join(d, L.L3_IDENTITY_FILE))) == default
    d2 = str(tmp_path / "store_on")
    assert L.l3_persist_check_identity(d2, on, dry=False) == "new"
    with pytest.raises(L.Weg2StoreDiskRefused):
        L.l3_persist_check_identity(d2, default, dry=False)


def test_w165_rank_identity_mismatch_names_the_switch(tmp_path, monkeypatch):
    from unittest import mock
    from sglang.srt.mem_cache.weg2_store_gates import Weg2L3IdentityMismatch

    (tmp_path / "L3_IDENTITY.json").write_text("{}")
    monkeypatch.setenv("SGLANG_WEG2_L3_PERSIST", "1")
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")

    def run(ident):
        fake = mock.MagicMock()
        fake.file_path = str(tmp_path)
        fake._l3p_on = HS.HiCacheFile._l3p_on
        fake._l3p_group = HS.HiCacheFile._l3p_group
        fake._l3p_persistent_dir = lambda: HS.HiCacheFile._l3p_persistent_dir(fake)
        cfg = mock.MagicMock()
        cfg.l3_rank_identity = ident
        HS.HiCacheFile._l3p_check_rank_identity(fake, cfg)

    off = dict(FIX["rank_identity"])
    on = dict(off, model_identity=HS.compute_model_identity_hash(_sa(moe_act_int8="on"), include_parallel_vectors=False))
    run(off)  # recorded
    run(off)  # match
    with pytest.raises(Weg2L3IdentityMismatch) as e:
        run(on)
    assert "W165" in str(e.value) and on["model_identity"] in str(e.value)


# ---------------------------------------------------------------- (5) snapshot, wiring
def test_identity_dict_snapshot_fixture_pins_every_key(tmp_path):
    """No other package may move the identity: the key set and every value of the default dict are the fixture's."""
    ident = _abl_derived_identity(tmp_path)
    assert sorted(ident) == sorted(FIX["identity"])
    assert json.dumps(ident, sort_keys=True) == json.dumps(FIX["identity"], sort_keys=True)
    # the optional keys exist only on their own conditions
    assert not {"rope", "rope_apply", "moe_act"} & set(ident)


def test_both_launcher_call_sites_pass_the_group_envs_and_the_flag():
    src = open(L.__file__).read()
    for var in ("_l3_idx_ident", "_l3_ident"):
        i = src.index(f"{var} = (l3_persist_identity(")
        call = src[i:i + 900]
        for needle in ('env_p=getattr(ns, "env_p", "") or ""', 'env_d=getattr(ns, "env_d", "") or ""',
                       'moe_act_int8=getattr(ns, "moe_act_int8", None)', 'd_only=bool(getattr(ns, "d_only", False))'):
            assert needle in call, (var, needle)


def test_main_refuses_an_asymmetric_switch_before_the_identity_for_every_boot():
    src = open(L.__file__).read()
    i = src.index("    l3_moe_act_resolve(getattr(ns, \"extra_p\"")
    assert i < src.index("    _l3_idx_ident = (l3_persist_identity(")
    assert "if l3_persist_enabled()" not in src[i - 200:i]  # unconditional: not behind the persistent-L3 gate


def test_the_resolve_main_runs_for_every_boot_is_a_noop_in_the_default(tmp_path):
    """Fix-round-2 finding (minor, kept and argued, not narrowed away): main() calls l3_moe_act_resolve for EVERY boot,
    also one without a persistent L3. The refusal is a config-error gate: it fires only when the operator switched W4A8
    on asymmetrically, a state that cannot exist unless they asked for it. In the DEFAULT (no source carries the switch)
    the call must change nothing -- here with the release profile nf-int4-h6-abl's REAL argv values from the AP0 launch
    fixture: returns False, no SystemExit, process env honored (absent key = off), and the identity dict stays the pinned
    default dc6a8c2062 (the dry-run golden test test_planer_referenz_n3_1006 pins the whole boot dump)."""
    argv = LAUNCH["argv"]
    args = (_flag(argv, "--extra-p"), _flag(argv, "--extra-d"), _flag(argv, "--env-p"), _flag(argv, "--env-d"))
    assert L.l3_moe_act_resolve(*args, None, {}) is False          # empty process env
    assert L.l3_moe_act_resolve(*args, None) is False              # os.environ, key removed by the autouse fixture
    assert "moe_act" not in _abl_derived_identity(tmp_path)        # and the identity does not grow the field
