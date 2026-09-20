# SPDX-License-Identifier: Apache-2.0
"""Slice 5: seam C -- every per-request state carried or priced, or W115."""
from __future__ import annotations

import pytest

from sglang.srt.flip_nextflash_plan import Weg2FlipDraftStateOrphaned
from sglang.srt.flip_state_carry import (
    DRAFT_REBUILD_TOKENS,
    KIND_KV_FOLLOWING,
    KIND_RECURRENT,
    KIND_SPECULATIVE,
    NEXT_FLASH_STATES,
    PP3_PREFILL_TOK_S,
    StateFamily,
    solve_flip_state_carry,
)

_MB = 1024 * 1024
_LIVE = [f.name for f in NEXT_FLASH_STATES if f.disposition != "absent"]


# --------------------------------------------------------------------------
# The declared set
# --------------------------------------------------------------------------
def test_every_declared_family_carries_its_evidence():
    for f in NEXT_FLASH_STATES:
        assert f.why, f.name
        assert f.kind in (KIND_RECURRENT, KIND_SPECULATIVE, KIND_KV_FOLLOWING)


def test_the_gdn_families_are_carried_because_rebuilding_means_reprefilling():
    """GDN state is not token-addressable: the state IS the result of every
    token so far. At 262k, rebuilding means the whole prefill again."""
    carry = solve_flip_state_carry(_LIVE)
    gdn = [f for f in carry.families if f.name.startswith(("mamba_", "intermediate_"))]
    assert len(gdn) == 4
    assert all(f.disposition == "carried" for f in gdn)
    assert all(f.kind == KIND_RECURRENT for f in gdn)


def test_the_carried_mass_matches_the_measured_form_a_mamba_lines():
    """fnFA19:1029-1031: 0.01 + 0.11 + 0.32 + 0.01 = 0.45 GB, plus the QSA
    ring. Small enough to ride the same legs as seam A."""
    carry = solve_flip_state_carry(_LIVE)
    assert carry.carried_bytes / _MB == pytest.approx(458, abs=2)


def test_the_gdn_carry_is_a_gather_not_a_copy():
    """P spreads the state over three cards (fn7s: 0.06 on PP0, 0.02 on
    PP1/PP2), D holds it on one."""
    carry = solve_flip_state_carry(_LIVE)
    assert carry.gathered_from["mamba_ssm_state"] == ("PP0", "1", "2")


# --------------------------------------------------------------------------
# The solo draft: no partner, and that is fine ONLY once declared
# --------------------------------------------------------------------------
def test_the_solo_draft_has_no_p_side_owner_and_is_priced():
    draft = next(f for f in NEXT_FLASH_STATES if f.name == "mtp_solo_draft_state")
    assert draft.owner_p == ""
    assert draft.disposition == "rebuilt"
    assert draft.rebuild_tokens == DRAFT_REBUILD_TOKENS


def test_the_draft_rebuild_is_far_under_the_physics_floor():
    carry = solve_flip_state_carry(_LIVE)
    assert carry.rebuild_tokens == DRAFT_REBUILD_TOKENS
    assert carry.rebuild_seconds < 0.002  # ~1.5 ms at 2729 tok/s
    assert PP3_PREFILL_TOK_S == 2729.0


def test_declaring_the_solo_draft_as_carried_is_W115():
    """'carried' with an empty source is the dishonest disposition."""
    bad = tuple(
        f if f.name != "mtp_solo_draft_state"
        else StateFamily(f.name, f.kind, "", f.owner_d, 0, "carried", 0, f.why)
        for f in NEXT_FLASH_STATES
    )
    with pytest.raises(Weg2FlipDraftStateOrphaned) as exc:
        solve_flip_state_carry(_LIVE, families=bad)
    msg = str(exc.value)
    assert "no P-side owner" in msg
    assert "solo draft" in msg


def test_a_rebuild_priced_at_zero_is_W115():
    """The point of the refusal is not the cost, it is the silence."""
    bad = tuple(
        f if f.name != "mtp_solo_draft_state"
        else StateFamily(f.name, f.kind, "", f.owner_d, 0, "rebuilt", 0, f.why)
        for f in NEXT_FLASH_STATES
    )
    with pytest.raises(Weg2FlipDraftStateOrphaned) as exc:
        solve_flip_state_carry(_LIVE, families=bad)
    assert "undeclared cost" in str(exc.value)


# --------------------------------------------------------------------------
# Risk R5: the list is no longer a hand list
# --------------------------------------------------------------------------
def test_the_qsa_raw_key_ring_is_declared_now():
    """It was MISSING from the first hand list -- that is R5."""
    ring = next(
        f for f in NEXT_FLASH_STATES if f.name == "qsa_pending_raw_key_ring"
    )
    assert ring.kind == KIND_RECURRENT
    assert ring.disposition == "carried"
    assert "R5" in ring.why


def test_an_undeclared_live_family_is_W115_not_silently_absent():
    """The whole mechanism: feed the runner's ACTUAL inventory and an
    undeclared family falls out as a refusal."""
    with pytest.raises(Weg2FlipDraftStateOrphaned) as exc:
        solve_flip_state_carry(_LIVE + ["hyper_connection_mixer_buffer"])
    msg = str(exc.value)
    assert "W115 Weg2FlipDraftStateOrphaned" in msg
    assert "hyper_connection_mixer_buffer" in msg
    assert "risk R5" in msg
    assert "belonging to nobody" in msg


def test_a_declared_family_the_runner_does_not_hold_is_also_W115():
    """A carry is a claim about something; a stale claim is still a claim."""
    with pytest.raises(Weg2FlipDraftStateOrphaned) as exc:
        solve_flip_state_carry([n for n in _LIVE if n != "mamba_ssm_state"])
    assert "does not hold" in str(exc.value)
    assert "stale" in str(exc.value)


def test_an_absent_family_need_not_be_live():
    """The QSA compressed key state follows the KV; the runner does not hold
    it as a separate family and that must not be a refusal."""
    carry = solve_flip_state_carry(_LIVE)
    assert "qsa_compressed_key_state" not in [f.name for f in carry.families]


# --------------------------------------------------------------------------
# The double-count seam C must not commit
# --------------------------------------------------------------------------
def test_the_kv_following_qsa_state_carries_zero_bytes_here():
    """It is allocated IN the KV pool, so seam A already moves it. Counting
    it again would inflate seam C with seam A's mass."""
    for name in ("qsa_compressed_key_state", "qsa_rope_position_buffer"):
        f = next(x for x in NEXT_FLASH_STATES if x.name == name)
        assert f.kind == KIND_KV_FOLLOWING
        assert f.bytes_per_request == 0
        assert f.disposition == "absent"


def test_the_report_names_every_family_and_its_disposition():
    text = solve_flip_state_carry(_LIVE).report()
    assert "STATE CARRY (seam C)" in text
    assert "mtp_solo_draft_state" in text and "rebuilt (4 tok)" in text
    assert "qsa_pending_raw_key_ring" in text


def test_planning_the_declared_set_alone_works_at_the_desk():
    carry = solve_flip_state_carry(None)
    assert len(carry.families) == len(NEXT_FLASH_STATES)
