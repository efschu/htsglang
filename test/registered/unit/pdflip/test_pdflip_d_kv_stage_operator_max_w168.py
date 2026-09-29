"""#251c / W168: an operator's FLLIPER_PDFLIP_D_KV_STAGE_MAX_BY_SEATS reaches the rank.

WHAT MUST HOLD (the A/B arm of the KV-stage price, 28.09.).
(1) --env-d naming FLLIPER_PDFLIP_D_KV_STAGE_MAX_BY_SEATS: the launcher writes THAT
    value into D's form (not its own default) and says so in one line with the
    extra waves per bs. Metal before the fix: profile ...-s0 (0,0,0,0,0,0) booted
    rc12z17s0 11:11Z with 'CAPTURE-FLOOR ... max_by_seats [2, 1, 1, 1, 0, 0]' --
    the launcher had written its default over the operator's value.
(2) A second solve pass (d_kv_stage_undo, then apply again) keeps the operator's
    value: the undo takes back only what the launcher wrote itself.
(3) A value the stage table cannot take (wrong length, stage out of range, not
    an integer list) is refused by name (W168 PdFlipDKvStageMaxRefused) -- also
    through d_seat_table_lines, whose informational except must not turn it
    into a quiet default form.
(4) Without the key everything is as before (the default form, no OPERATOR line).
(5) The rank reads the written value: the capture floor follows it (S1 at n=5/6
    lowers the bs5/6 floor).
"""
from __future__ import annotations

import os
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.planner import expert_residency as er  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402

FORM = er.SeatVramForm(temporal_slot_bytes=(48 * 128 * 128 * 2, 0, 0), gdn_layers=36,
                       expert_row_bytes=2534448, moe_layers=48, small_row_bytes=76848)
KEY = "FLLIPER_PDFLIP_D_KV_STAGE_MAX_BY_SEATS"


def _rows():
    """The H95c fixture of test_pdflip_d_kv_stage_launcher_251c (scratch 100 on TP0)."""

    def row(n, max_rows):
        return er.SeatTableRow(
            seats=n, ids_per_step=40 * n, waves=2, mamba_slots=7, host_mamba_mib=0.0,
            host_spec_mib=0.0, max_rows=max_rows, scratch_given=(100, 48, 48),
            waves_given=(1, 1, 1), fraction_given=(None, None, None),
            scratch_min=(None, None, None), fraction_max=(None, None, None), refusal=None)

    return er._seat_vram_columns(
        tuple(row(n, (136 - (16 * (n - 1)) // 5, 140, 141)) for n in range(1, 7)), FORM)


def _plan():
    return types.SimpleNamespace(fits=[types.SimpleNamespace(
        rank=0, kv_cell_bytes=14143, kv_tokens=262144, local_experts=193, staging_rows=12)])


def _apply(ns):
    from flliper.srt.pdflip import launcher as L

    return L.apply_d_kv_stage_form(ns, er, _rows(), FORM, _plan(), "D", verify_tokens=4,
                                   top_k=10)


def test_the_operators_max_by_seats_is_written_and_named():
    from flliper.srt.pdflip import launcher as L

    ns = types.SimpleNamespace(env_d="FLLIPER_MOE_SCRATCH_SLOTS=100,48,48;%s=2,1,1,1,1,1" % KEY)
    lines = _apply(ns)
    env = L.parse_group_env(ns.env_d)
    assert env[KEY] == "2,1,1,1,1,1"
    # the rest of the form is the launcher's, unchanged by the override
    assert env["FLLIPER_PDFLIP_D_KV_STAGE_TOKENS"] == "262144,393216,524288"
    assert env["FLLIPER_PDFLIP_D_KV_STAGE_ROWS"] == "33"
    assert env["FLLIPER_MOE_SCRATCH_SLOTS"] == "67,48,48"
    op = [ln for ln in lines if "OPERATOR" in ln]
    assert len(op) == 1
    assert "%s=2,1,1,1,1,1 statt der Voreinstellung 2,1,1,1,0,0" % KEY in op[0]
    assert "Zusatzwellen je bs [0, 0, 0, 0, 1, 1]" in op[0]
    assert "max_by_seats [2, 1, 1, 1, 1, 1]" in op[0]


def test_the_s2_probe_arm_is_written():
    from flliper.srt.pdflip import launcher as L

    ns = types.SimpleNamespace(env_d="FLLIPER_MOE_SCRATCH_SLOTS=100,48,48;%s=2,2,2,1,0,0" % KEY)
    lines = _apply(ns)
    assert L.parse_group_env(ns.env_d)[KEY] == "2,2,2,1,0,0"
    assert any("OPERATOR" in ln and "2,2,2,1,0,0" in ln for ln in lines)


def test_a_second_solve_pass_keeps_the_operators_value():
    from flliper.srt.pdflip import launcher as L

    before = "FLLIPER_MOE_SCRATCH_SLOTS=100,48,48;FLLIPER_PDFLIP_D_SEAT_EXPERT_ROWS=14,0,0;%s=2,1,1,1,1,1" % KEY
    ns = types.SimpleNamespace(env_d=before)
    _apply(ns)
    L.d_kv_stage_undo(ns)
    assert L.parse_group_env(ns.env_d) == L.parse_group_env(before)
    _apply(ns)
    assert L.parse_group_env(ns.env_d)[KEY] == "2,1,1,1,1,1"
    assert L.parse_group_env(ns.env_d)["FLLIPER_MOE_SCRATCH_SLOTS"] == "67,48,48"


@pytest.mark.parametrize("raw", ["2,1,1,1,1", "2,1,1,1,1,1,1", "3,1,1,1,1,1", "-1,1,1,1,1,1",
                                 "2,x,1,1,1,1"])
def test_a_value_the_table_cannot_take_is_refused_by_name(raw):
    from flliper.srt.pdflip import launcher as L

    told = "FLLIPER_MOE_SCRATCH_SLOTS=100,48,48;%s=%s" % (KEY, raw)
    ns = types.SimpleNamespace(env_d=told)
    with pytest.raises(L.PdFlipDKvStageMaxRefused, match="W168 PdFlipDKvStageMaxRefused"):
        _apply(ns)
    assert ns.env_d == told  # nothing written
    assert issubclass(L.PdFlipDKvStageMaxRefused, L.PdFlipLaunchRefused)


def test_the_refusal_is_not_swallowed_by_the_seat_table():
    from flliper.srt.pdflip import launcher as L

    src = open(L.__file__).read()
    k = src.index("def d_seat_table_lines(")
    body = src[k:src.index("\ndef ", k + 10)]
    assert body.index("except PdFlipDKvStageMaxRefused:") < body.index("except Exception as exc:")
    assert "raise" in body[body.index("except PdFlipDKvStageMaxRefused:"):
                          body.index("except Exception as exc:")]


def test_without_the_key_the_default_form_is_unchanged():
    from flliper.srt.pdflip import launcher as L

    ns = types.SimpleNamespace(env_d="FLLIPER_MOE_SCRATCH_SLOTS=100,48,48")
    lines = _apply(ns)
    assert L.parse_group_env(ns.env_d)[KEY] == "2,1,1,1,0,0"
    assert not any("OPERATOR" in ln for ln in lines)
    assert L.operator_kv_stage_max(None, 6, 3) is None
    assert L.operator_kv_stage_max(" ", 6, 3) is None


def _cells(extra=(13, 10, 8, 5, 1, 0), stage_rows=(0, 16, 32), S=33):
    return {(n, j): types.SimpleNamespace(extra_rows=extra[n - 1] + S - r)
            for n in range(1, 7) for j, r in enumerate(stage_rows)}


def test_the_rank_reads_the_written_value():
    from flliper.srt.pdflip import launcher as L

    ns = types.SimpleNamespace(env_d="FLLIPER_MOE_SCRATCH_SLOTS=100,48,48;%s=2,1,1,1,1,1" % KEY)
    _apply(ns)
    env = L.parse_group_env(ns.env_d)
    form = dsv.StageForm(tokens=tuple(int(t) for t in env["FLLIPER_PDFLIP_D_KV_STAGE_TOKENS"].split(",")),
                         rows_on=int(env["FLLIPER_PDFLIP_D_KV_STAGE_ROWS"]),
                         max_by_seats=dsv._ints(env[KEY]))
    assert form.max_stage(5) == 1 and form.max_stage(6) == 1
    # default floors (14, 22, 22, 22, 33, 33); S1 at n=5/6 lowers bs5/6 to the S1 rows
    assert dsv.capture_floors(form, _cells(), 6) == (14, 17, 17, 17, 17, 17)
