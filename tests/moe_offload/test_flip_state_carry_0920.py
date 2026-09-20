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


# --------------------------------------------------------------------------
# Wired at the cutover (slice 5, part 2)
# --------------------------------------------------------------------------
class _Sched:
    def __init__(self, families=None, raising=False):
        self._f = families
        self._raising = raising

    def next_flash_state_families(self):
        if self._raising:
            raise RuntimeError("inventory unavailable")
        return list(self._f)


def test_the_cutover_verifies_the_state_carry_on_the_decode_side():
    from sglang.srt.managers.phase_flip_runtime import _verify_flip_state_carry

    _verify_flip_state_carry(_Sched(_LIVE), tp_phase=True)  # no raise


def test_the_cutover_raises_W115_for_an_undeclared_live_family():
    from sglang.srt.managers.phase_flip_runtime import _verify_flip_state_carry

    sched = _Sched(_LIVE + ["hyper_connection_mixer_buffer"])
    with pytest.raises(Weg2FlipDraftStateOrphaned) as exc:
        _verify_flip_state_carry(sched, tp_phase=True)
    assert "hyper_connection_mixer_buffer" in str(exc.value)


def test_the_pp_side_is_not_checked_because_it_rebuilds_anyway():
    from sglang.srt.managers.phase_flip_runtime import _verify_flip_state_carry

    sched = _Sched(_LIVE + ["something_undeclared"])
    _verify_flip_state_carry(sched, tp_phase=False)  # no raise


def test_a_runner_without_an_inventory_stands_aside_rather_than_passing_empty():
    """An empty list would pass the completeness test for every declared
    family at once -- an ABSENCE must be reported as one."""
    from sglang.srt.managers.phase_flip_runtime import (
        _next_flash_live_state_families,
        _verify_flip_state_carry,
    )

    class Bare:
        pass

    assert _next_flash_live_state_families(Bare()) is None
    _verify_flip_state_carry(Bare(), tp_phase=True)  # no raise


def test_a_raising_inventory_is_an_absence_not_a_crash():
    from sglang.srt.managers.phase_flip_runtime import _next_flash_live_state_families

    assert _next_flash_live_state_families(_Sched(raising=True)) is None


def test_a_plain_sequence_inventory_also_works():
    from sglang.srt.managers.phase_flip_runtime import _next_flash_live_state_families

    class Seq:
        next_flash_state_families = tuple(_LIVE)

    assert _next_flash_live_state_families(Seq()) == _LIVE
