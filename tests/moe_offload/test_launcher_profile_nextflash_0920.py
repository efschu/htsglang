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


def test_group_env_extra_is_applied_last_and_parsed_strictly(tmp_path):
    import pytest as _pt

    assert lc.parse_group_env("") == {}
    assert lc.parse_group_env("A=1; SGLANG_MOE_SCRATCH_SLOTS=74,48,48") == {"A": "1", "SGLANG_MOE_SCRATCH_SLOTS": "74,48,48"}
    with _pt.raises(ValueError):
        lc.parse_group_env("NOEQUALS")
    kw = dict(chunk_layers=0, chunk_count=0, tms_so="", transport="bar1", ring=None)
    env = lc.build_env(str(tmp_path), "venv", "0,1,2", str(tmp_path), False, "t", group="P",
                       profile=lc.PROFILE_NEXTFLASH, group_env_extra={"SGLANG_MOE_SCRATCH_SLOTS": "32"}, **kw)
    assert env["SGLANG_MOE_SCRATCH_SLOTS"] == "32"


def test_resident_arm_sleeps_only_kv(monkeypatch):
    """fnFL2 v7 (20.09.): the launcher's post-READY sleep(P) must not name a
    weights tag on a resident boot (server W4 refuses -> W29 rank disagree)."""
    import types

    from sglang.srt.weg2 import launcher as L

    fam = ["weights_0", "weights_1", "weights"]
    assert L.flip_weights_tags(types.SimpleNamespace(flip_weights="resident"), fam) == []
    assert L.flip_weights_tags(types.SimpleNamespace(flip_weights="family"), fam) == fam
    assert L.flip_weights_tags(types.SimpleNamespace(), fam) == fam
    sent = []
    monkeypatch.setattr(L, "http", lambda m, url, body, timeout: sent.append(body) or (200, "{}"))
    L.sleep_group(30031, lambda *a, **k: None, "P", L.flip_weights_tags(types.SimpleNamespace(flip_weights="resident"), fam))
    assert sent == [{"tags": ["kv_cache"]}]


def test_w7_w10_counts_form_a_workers(tmp_path):
    """fnFL2 v18 (21.09.): D READY with one attention host and two expert
    workers logged kv x1 blob x1; the workers' own #706 line stands in."""
    from sglang.srt.weg2 import launcher as L

    log = tmp_path / "D.log"
    log.write_text(
        "[TP0] #706 canonical KV page active: slots [0, 12) of 12\n"
        "[TP0] #706 canonical GDN blob active: layers [0, 36) of 36\n"
        "[TP1] #706 canonical KV page: this rank is a Form A expert worker (no attention layer) -- no page window, null storage tier\n"
        "[TP2] #706 canonical KV page: this rank is a Form A expert worker (no attention layer) -- no page window, null storage tier\n"
    )
    assert L.canonical_marker_counts(str(log)) == (3, 3, 2)
    log.write_text("[PP0] #706 canonical KV page active\n[PP0] canonical GDN blob active\n")
    assert L.canonical_marker_counts(str(log)) == (1, 1, 0)
    # the marker text is the controller's own line, byte for byte
    from sglang.srt.managers import cache_controller as cc
    import inspect

    src = inspect.getsource(cc.HiCacheController._generate_storage_config)
    assert "this rank is a Form A expert worker " in src and "(no attention layer) -- no page window" in src
