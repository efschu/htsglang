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
