"""Task #47 Scheibe 6a (20.09.): the weg2 launcher's ``--profile nextflash``.

Next Flash runs P = PP3 (tp 1 per stage) and D = Form A, whose rank_role
collapses DCP to 1 and REFUSES an inherited SGLANG_UNEVEN_DCP=1. The 27B
launcher wrote the uneven-DCP facts (EARLY_READ_FACTS) into BOTH groups' env
and the flag half into D's argv. Under the nextflash profile the env is
group-owned (flip_nextflash_groups GROUP_ENV_VALUES) and the flag half is
dropped; the qwen27b profile stays byte-identical."""

import types

from sglang.srt.weg2 import launcher as lc


def _common(group, profile):
    return lc.common_flags("m", 1, 1, "{}", 1000, group=group, profile=profile)


def test_qwen27b_profile_keeps_the_dcp_flag_half_for_d():
    assert "--uneven-dcp" in _common("D", lc.PROFILE_QWEN27B)
    assert "--uneven-dcp" not in _common("P", lc.PROFILE_QWEN27B)
    assert _common("D", lc.PROFILE_QWEN27B) == lc.common_flags("m", 1, 1, "{}", 1000, group="D")


def test_nextflash_profile_drops_the_dcp_flag_half_on_both_groups():
    for g in ("P", "D"):
        argv = _common(g, lc.PROFILE_NEXTFLASH)
        assert "--uneven-dcp" not in argv and "--uneven-dcp-weighted" not in argv


def test_nextflash_env_is_group_owned(monkeypatch, tmp_path):
    monkeypatch.delenv("SGLANG_UNEVEN_DCP", raising=False)
    monkeypatch.delenv("SGLANG_UNEVEN_DCP_WEIGHTED", raising=False)
    kw = dict(chunk_layers=0, chunk_count=0, tms_so="", transport="bar1", ring=None)
    env_p = lc.build_env(str(tmp_path), "venv", "0,1,2", str(tmp_path), False, "t", group="P",
                         profile=lc.PROFILE_NEXTFLASH, **kw)
    env_d = lc.build_env(str(tmp_path), "venv", "0,1,2", str(tmp_path), False, "t", group="D",
                         profile=lc.PROFILE_NEXTFLASH, **kw)
    env_d27 = lc.build_env(str(tmp_path), "venv", "0,1,2", str(tmp_path), False, "t", group="D", **kw)
    assert (env_p["SGLANG_UNEVEN_DCP"], env_p["SGLANG_UNEVEN_DCP_WEIGHTED"]) == ("1", "1")
    assert (env_d["SGLANG_UNEVEN_DCP"], env_d["SGLANG_UNEVEN_DCP_WEIGHTED"]) == ("0", "0")
    assert (env_d27["SGLANG_UNEVEN_DCP"], env_d27["SGLANG_UNEVEN_DCP_WEIGHTED"]) == ("1", "1")


def test_the_knobs_carry_the_profile_and_the_cli_offers_it():
    ns = types.SimpleNamespace(weg2_seam_digest=False, barlink_build_window_cap_s=1, pp_chain_recv_stall_s=1,
                               pp_occupant_horizon_s=1, match_refusal_census_every=1, arming_floor_solved=True,
                               hicache_bigram_keys=True, hicache_flush_publish_sweep=True, profile="nextflash",
                               weg2_vision="off", tag="t")
    assert lc._env_knobs(ns)["profile"] == "nextflash"
    import inspect

    src = inspect.getsource(lc)
    assert 'ap.add_argument("--profile", choices=list(PROFILES), default=PROFILE_QWEN27B' in src
