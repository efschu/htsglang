# SPDX-License-Identifier: Apache-2.0
"""#656: pool = VRAM - corridor - staging, with the staging term solved as a fixed point.

Boot E sized the pool to 683150 tokens, held the 1024 MiB corridor with zero
breaches, and served nothing: on rank1 the seam needed 464 MiB of staging and
only 444 MiB was spendable above the corridor floor. 20 MiB. Every ``tp_to_pp``
flip abandoned, and under strict phase purity that means prefill never runs.

These are the arithmetic properties of the fix, pinned without a GPU:

  * the reserve is the BINDING constraint, not a second copy of the corridor;
  * the pool-dependent part is solved, not approximated -- the staging term
    grows with the pool it is subtracted from;
  * a COLD record changes nothing, so the first boot of a configuration is
    byte-identical to today.
"""

import json
import os

import pytest

from sglang.srt.managers import phase_flip_seam_reserve as sr

MIB = 1 << 20

# Boot E, rank1 (/spinning/evidence-631/kvuniverse-r1/boot_e.log).
BOOT_E_HEADROOM = 1468 * MIB
BOOT_E_CORRIDOR = 1024 * MIB
BOOT_E_STAGING = 464 * MIB


# ---------------------------------------------------------------------------
# The reservation is the binding term, and only the binding term.
# ---------------------------------------------------------------------------


def test_the_boot_e_shortfall_is_exactly_recovered():
    need = sr.required_free_bytes(BOOT_E_HEADROOM, BOOT_E_CORRIDOR, BOOT_E_STAGING)
    assert need == BOOT_E_CORRIDOR + BOOT_E_STAGING
    assert (need - BOOT_E_HEADROOM) == 20 * MIB, (
        "the fix must reserve the 20 MiB that were missing and not a byte "
        "more -- over-reserving here is paid for in permanent pool"
    )


def test_the_corridor_is_not_reserved_twice():
    """A headroom that already covers corridor+seam must not shrink the pool."""
    roomy = 4096 * MIB
    assert sr.required_free_bytes(roomy, BOOT_E_CORRIDOR, BOOT_E_STAGING) == roomy


def test_no_seam_means_no_change():
    assert sr.required_free_bytes(BOOT_E_HEADROOM, BOOT_E_CORRIDOR, 0) == BOOT_E_HEADROOM


# ---------------------------------------------------------------------------
# The fixed point.
# ---------------------------------------------------------------------------

CELL = 32 * 1024  # 32 KiB per token per rank, the rig's measured cell
PER_ROW = 512.0  # wave-boundary slack per pool row


def test_the_floor_branch_is_exact():
    """While a*T <= F the floor binds and the pool loses exactly F bytes."""
    R = 20 * (1 << 30)
    T = sr.solve_pool_tokens(R, CELL, fixed_bytes=8 * (1 << 30), per_row_bytes=1.0)
    assert T == (R - 8 * (1 << 30)) // CELL
    assert T * 1.0 <= 8 * (1 << 30)


def test_the_slack_branch_is_self_consistent():
    """When the row term binds, T must satisfy the equation it came from."""
    R = 20 * (1 << 30)
    T = sr.solve_pool_tokens(R, CELL, fixed_bytes=0, per_row_bytes=PER_ROW)
    assert T * CELL + T * PER_ROW <= R
    assert (T + 1) * CELL + (T + 1) * PER_ROW > R


def test_the_branch_that_binds_is_the_one_chosen():
    """max(F, a*T), not F + a*T: reserving the sum costs pool for a peak that
    does not occur (the runtime's own _staging_bytes takes the max)."""
    R = 20 * (1 << 30)
    F, a = 464 * MIB, 368.0
    T = sr.solve_pool_tokens(R, CELL, F, a)
    summed = int((R - F) / (CELL + a))
    assert T > summed, "summing the two terms gives away pool the seam never uses"
    assert T * CELL + max(F, T * a) <= R


def test_one_iteration_lands_on_the_wrong_side():
    """Why this is solved rather than iterated once.

    A single iteration charges the per-row term at the NAIVE pool size -- a
    pool that will not exist once the term is charged -- so it over-subtracts
    and settles BELOW the fixed point. The direction is the safe one, which
    is exactly why it would never be noticed: it silently gives away pool on
    a ticket whose whole subject is pool that was silently given away.
    """
    R = 20 * (1 << 30)
    naive = R // CELL
    one_step = int((R - naive * PER_ROW) // CELL)
    solved = sr.solve_pool_tokens(R, CELL, 0, PER_ROW)
    assert one_step < solved < naive


def test_the_boot_e_budget_gives_up_exactly_the_missing_bytes():
    """End to end on boot E's rank1 numbers, with no row term.

    budget + headroom - corridor - F is the KV that may remain, and that is
    20 MiB less than the budget the boot actually took.
    """
    budget = 15 * (1 << 30)
    reserve = sr.SeamReserve(
        fixed_bytes=BOOT_E_STAGING,
        per_row_bytes=0.0,
        provenance=sr.PROVENANCE_STORED,
    )
    new_bytes, tokens = sr.seam_adjusted_budget_bytes(
        budget, BOOT_E_HEADROOM, BOOT_E_CORRIDOR, CELL, reserve
    )
    assert budget - new_bytes >= 20 * MIB
    assert budget - new_bytes < 20 * MIB + CELL, "and not a page more"
    assert tokens == new_bytes // CELL


def test_a_roomy_headroom_leaves_the_budget_untouched():
    """The seam must never GROW a budget, whatever the headroom is."""
    budget = 15 * (1 << 30)
    reserve = sr.SeamReserve(fixed_bytes=64 * MIB, provenance=sr.PROVENANCE_STORED)
    new_bytes, _ = sr.seam_adjusted_budget_bytes(
        budget, 8 * (1 << 30), BOOT_E_CORRIDOR, CELL, reserve
    )
    assert new_bytes == budget


def test_no_cell_means_abstain(tmp_path):
    """A configurator with no single per-token cell gets no invented one."""
    budget = 15 * (1 << 30)
    reserve = sr.SeamReserve(fixed_bytes=BOOT_E_STAGING, provenance=sr.PROVENANCE_STORED)
    assert sr.seam_adjusted_budget_bytes(
        budget, BOOT_E_HEADROOM, BOOT_E_CORRIDOR, 0, reserve
    ) == (budget, 0)


# ---------------------------------------------------------------------------
# The record: cold is today's behaviour, exactly.
# ---------------------------------------------------------------------------


class _Args:
    def __init__(self, path):
        self._measured_kv_budget_registry_path = path


@pytest.fixture(autouse=True)
def _clean_env():
    saved = {k: os.environ.pop(k, None) for k in (sr.ENV_ENABLE, sr.ENV_FIXED_MIB)}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


def test_a_cold_record_corrects_nothing(tmp_path):
    args = _Args(str(tmp_path / "kv_budget-abc.json"))
    rv = sr.read_seam_reserve(args, world_rank=1)
    assert rv.provenance == sr.PROVENANCE_COLD
    assert rv.fixed_bytes == 0 and rv.per_row_bytes == 0.0 and not rv.active
    budget = 15 * (1 << 30)
    assert sr.seam_adjusted_budget_bytes(
        budget, BOOT_E_HEADROOM, BOOT_E_CORRIDOR, CELL, rv
    ) == (budget, 0), "a first boot must size exactly as it does today"
    assert "COLD" in sr.describe(rv, "p") and "boot E" in sr.describe(rv, "p")


def test_the_record_round_trips_and_is_per_rank(tmp_path):
    args = _Args(str(tmp_path / "kv_budget-abc.json"))
    sr.write_seam_reserve(args, 1, BOOT_E_STAGING, 512.0, "arena tail 466 MiB")
    sr.write_seam_reserve(args, 2, 1436 * MIB, 512.0, "arena tail 1436 MiB")

    r1, r2 = sr.read_seam_reserve(args, 1), sr.read_seam_reserve(args, 2)
    assert r1.provenance == sr.PROVENANCE_STORED
    assert r1.fixed_bytes == BOOT_E_STAGING and r1.per_row_bytes == 512.0
    assert r2.fixed_bytes == 1436 * MIB, (
        "the arena tail differs per rank by ~1 GiB on this rig; one shared "
        "record would size every rank from one rank's seam"
    )
    assert r1.written_at and "466" in r1.detail


def test_a_malformed_record_sizes_cold_rather_than_raising(tmp_path):
    args = _Args(str(tmp_path / "kv_budget-abc.json"))
    path = sr.record_path(args, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("{not json")
    rv = sr.read_seam_reserve(args, 1)
    assert rv.provenance == sr.PROVENANCE_MALFORMED and rv.fixed_bytes == 0


def test_a_partial_write_cannot_be_read(tmp_path):
    """The write is atomic, so a boot that dies mid-write leaves the old record."""
    args = _Args(str(tmp_path / "kv_budget-abc.json"))
    sr.write_seam_reserve(args, 1, BOOT_E_STAGING, 512.0, "first")
    path = sr.record_path(args, 1)
    with open(path) as fh:
        assert json.load(fh)["fixed_bytes"] == BOOT_E_STAGING
    leftovers = [p for p in os.listdir(os.path.dirname(path)) if ".tmp" in p]
    assert not leftovers


def test_the_term_can_be_switched_off_as_a_value(tmp_path):
    args = _Args(str(tmp_path / "kv_budget-abc.json"))
    sr.write_seam_reserve(args, 1, BOOT_E_STAGING, 512.0, "measured")
    os.environ[sr.ENV_ENABLE] = "0"
    rv = sr.read_seam_reserve(args, 1)
    assert rv.provenance == sr.PROVENANCE_DISABLED and rv.fixed_bytes == 0


def test_an_operator_override_wins_over_the_record(tmp_path):
    args = _Args(str(tmp_path / "kv_budget-abc.json"))
    sr.write_seam_reserve(args, 1, BOOT_E_STAGING, 512.0, "measured")
    os.environ[sr.ENV_FIXED_MIB] = "900"
    rv = sr.read_seam_reserve(args, 1)
    assert rv.provenance == sr.PROVENANCE_OVERRIDE and rv.fixed_bytes == 900 * MIB
