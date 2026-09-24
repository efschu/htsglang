"""--d-replayssm-spec: group D's opt-in to the ReplaySSM spec ring (S6).

27B ReplaySSM package, slice S6 (REPLAYSSM_PLAN.md). Hermetic. Pinned:

* default OFF: argv_d, the D pricing fields and P's argv are what they were
  (an explicit 'off' ships the identical argv);
* ON: group D alone gets ``--enable-linear-replayssm-spec`` plus a ring length
  that is a power of two, >= 16 and >= the draft window -- and the launcher's
  D pricing (spec_plan_fields -> d_plan_inputs) carries the same length, ONE
  factory for the argv and the price;
* the shipped D argv parses in the server's own CLI with the flag set;
* the launcher line names both states, the ON one with its metal gate.
"""

import argparse
from types import SimpleNamespace

import pytest

import sglang.srt.weg2.launcher as L

RING_FLAG = "--enable-linear-replayssm-spec"


@pytest.fixture
def restore_form():
    saved = dict(L._SPEC_FORM)
    try:
        yield
    finally:
        L._SPEC_FORM.clear()
        L._SPEC_FORM.update(saved)


def _dflash(tmp_path, block=8, **kw):
    L.apply_spec_form(
        SimpleNamespace(
            spec_form="dflash",
            dflash_draft_path=str(tmp_path),
            dflash_block=block,
            dflash_window=2048,
            **kw,
        )
    )


def _argv_d():
    return L.argv_d(
        "py", L.MODEL_DEFAULT, [1, 1, 1], 1, 1, L.RING_FORM_SENTINEL_STORE_CFG, [], d_bs=2
    )


def _argv_p():
    return L.argv_p(
        "py", L.MODEL_DEFAULT, [1, 1, 1], 1, 1, L.RING_FORM_SENTINEL_STORE_CFG, [], p_bs=2
    )


def test_default_off_ships_the_old_argv(restore_form, tmp_path):
    _dflash(tmp_path)
    default_d = _argv_d()
    assert RING_FLAG not in default_d
    assert "linear_replayssm_spec_ring_len" not in L.spec_plan_fields()
    assert L.d_replayssm_spec_line().startswith("WEG2 D-REPLAYSSM-SPEC: off")
    _dflash(tmp_path, d_replayssm_spec="off")
    assert _argv_d() == default_d


def test_on_is_group_d_only_and_priced_alike(restore_form, tmp_path):
    _dflash(tmp_path, d_replayssm_spec="on")
    argv_d = _argv_d()
    i = argv_d.index(RING_FLAG)
    assert argv_d[i : i + 3] == [RING_FLAG, "--linear-replayssm-cache-len", "16"]
    # right after the speculative family, before anything --extra-d appends
    spec = L.spec_flags(producer=False)
    assert argv_d[i - len(spec) : i] == spec
    assert RING_FLAG not in _argv_p()
    assert L.spec_plan_fields()["linear_replayssm_spec_ring_len"] == 16
    pi = L.d_plan_inputs(L.MODEL_DEFAULT, 3, 2)
    assert pi.linear_replayssm_spec_ring_len == 16
    line = L.d_replayssm_spec_line()
    assert line.startswith("WEG2 D-REPLAYSSM-SPEC: on") and "METAL GATE" in line


def test_ring_length_follows_the_window(restore_form, tmp_path):
    _dflash(tmp_path, block=32, d_replayssm_spec="on")
    assert L.d_replayssm_spec_ring_len() == 32
    L.apply_spec_form(SimpleNamespace(spec_form="NEXTN", d_replayssm_spec="on"))
    assert L.d_replayssm_spec_ring_len() == 16
    assert L.spec_plan_fields()["linear_replayssm_spec_ring_len"] == 16


def test_the_shipped_d_argv_parses_with_the_ring(restore_form, tmp_path):
    from sglang.srt.server_args import ServerArgs

    _dflash(tmp_path, d_replayssm_spec="on")
    argv = _argv_d()
    assert argv[1:3] == ["-m", "sglang.launch_server"]
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    ns = parser.parse_args(argv[3:])
    assert ns.enable_linear_replayssm_spec is True
    assert ns.linear_replayssm_cache_len == 16
    assert ns.speculative_algorithm == "DFLASH"
    assert ns.enable_linear_replayssm is False


# --- NF line (H64): the Next-Flash form -------------------------------------
# Group D of the Next-Flash arm runs NEXTN/MTP with its depth out of --extra-d
# (boot_nf_bestform_0924.sh: --speculative-num-steps 3 --speculative-eagle-topk
# 1 --speculative-num-draft-tokens 4), appended after the constants' family
# (SPEC_NUM_DRAFT_TOKENS 3) -- argparse takes the last value.
NF_EXTRA_D = (
    "--max-running-requests 1 --mamba-ssm-dtype bfloat16 "
    "--rank-role host,worker,worker --rank-tp-ratio 1,0,0 "
    "--speculative-draft-placement solo --speculative-algorithm NEXTN "
    "--speculative-num-steps 3 --speculative-eagle-topk 1 "
    "--speculative-num-draft-tokens 4"
)


def _nextn(extra_d="", **kw):
    L.apply_spec_form(SimpleNamespace(spec_form="NEXTN", extra_d=extra_d, **kw))


def _argv_d_nf(extra_d):
    import shlex

    return L.argv_d(
        "py",
        L.MODEL_DEFAULT,
        [1, 1, 1],
        1,
        1,
        L.RING_FORM_SENTINEL_STORE_CFG,
        shlex.split(extra_d),
        d_bs=1,
    )


def test_nf_window_is_the_extra_d_depth(restore_form):
    _nextn(NF_EXTRA_D, d_replayssm_spec="on")
    assert L.d_verify_window() == 4
    assert L.d_replayssm_spec_ring_len() == 16
    # the last value wins, as in argparse; a window past 16 doubles the ring
    _nextn(
        "--speculative-num-draft-tokens 4 --speculative-num-draft-tokens 20",
        d_replayssm_spec="on",
    )
    assert L.d_verify_window() == 20
    assert L.d_replayssm_spec_ring_len() == 32
    # no depth in --extra-d: the constants' window stands
    _nextn("", d_replayssm_spec="on")
    assert L.d_verify_window() == int(L.SPEC_NUM_DRAFT_TOKENS)
    assert L.d_replayssm_spec_ring_len() == 16
    line = L.d_replayssm_spec_line()
    assert "MTP (NEXTN) acceptance length" in line and "DFLASH" not in line


def test_nf_arm_argv_d_off_identical_on_parses(restore_form):
    from sglang.srt.server_args import ServerArgs

    _nextn(NF_EXTRA_D)
    default_d = _argv_d_nf(NF_EXTRA_D)
    _nextn(NF_EXTRA_D, d_replayssm_spec="off")
    assert _argv_d_nf(NF_EXTRA_D) == default_d
    assert RING_FLAG not in default_d
    assert "linear_replayssm_spec_ring_len" not in L.spec_plan_fields()

    _nextn(NF_EXTRA_D, d_replayssm_spec="on")
    argv = _argv_d_nf(NF_EXTRA_D)
    i = argv.index(RING_FLAG)
    spec = L.spec_flags(producer=False)
    assert argv[i - len(spec) : i] == spec
    # everything else is the off argv, in order
    assert argv[:i] + argv[i + 3 :] == default_d
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    ns = parser.parse_args(argv[3:])
    assert ns.enable_linear_replayssm_spec is True
    assert ns.linear_replayssm_cache_len == 16
    assert ns.speculative_algorithm == "NEXTN"
    assert ns.speculative_num_draft_tokens == 4
    assert ns.speculative_eagle_topk == 1
    assert ns.max_running_requests == 1
