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
# The solve is anchored on a MEASUREMENT, not on a model of the sizer.
# ---------------------------------------------------------------------------

CELL = 32 * 1024  # per-RANK per-token KV bytes
BOOT_G_HAVE = 8 * MIB  # rank1 measured 8 MiB spendable above the law
BOOT_G_T = 683150


def _reserve(fixed, per_row=0.0, have=BOOT_G_HAVE, t=BOOT_G_T):
    return sr.SeamReserve(
        fixed_bytes=fixed,
        per_row_bytes=per_row,
        have_bytes=have,
        id_space=t,
        provenance=sr.PROVENANCE_STORED,
    )


def test_the_boot_g_shortfall_becomes_a_pool_reduction():
    """rank1 measured: needs 484 MiB, had 8 MiB spendable, at T=683150.

    Every row given back returns one cell to the spendable pool, so the pool
    must shrink by (484-8) MiB / cell tokens -- and no further.
    """
    need = 484 * MIB
    allowed = sr.seam_allowed_tokens(CELL, _reserve(need))
    assert allowed == BOOT_G_T + (BOOT_G_HAVE - need) // CELL
    assert allowed < BOOT_G_T
    # ... and at that id space the seam is exactly funded.
    have_at = BOOT_G_HAVE + (BOOT_G_T - allowed) * CELL
    assert have_at >= need
    assert have_at - need < CELL, "and not a token more conservative"


def test_a_rank_with_room_to_spare_is_not_cut():
    """rank0 measured 2000 MiB spendable against a 455 MiB floor."""
    allowed = sr.seam_allowed_tokens(CELL, _reserve(455 * MIB, have=2000 * MIB))
    assert allowed > BOOT_G_T, "a funded rank must not lower the pool"


def test_the_budget_is_never_grown():
    budget = 15 * (1 << 30)
    new_bytes, allowed = sr.seam_adjusted_budget_bytes(
        budget, CELL, _reserve(455 * MIB, have=2000 * MIB)
    )
    assert new_bytes == budget and allowed > BOOT_G_T


def test_the_slack_branch_is_self_consistent():
    """When the row term binds, have(T) must still cover need(T) = a*T."""
    a = 2360.4  # rank0's measured B/token
    allowed = sr.seam_allowed_tokens(CELL, _reserve(0, per_row=a, have=2000 * MIB))
    have_at = BOOT_G_HAVE * 0 + 2000 * MIB + (BOOT_G_T - allowed) * CELL
    assert have_at >= a * allowed
    assert have_at < a * (allowed + 1) + CELL


def test_a_cold_or_incomplete_record_changes_nothing():
    budget = 15 * (1 << 30)
    for rv in (
        sr.SeamReserve(provenance=sr.PROVENANCE_COLD),
        # A record from before the measured position existed: no id space,
        # so the slope has no anchor and the correction must abstain.
        sr.SeamReserve(fixed_bytes=484 * MIB, provenance=sr.PROVENANCE_STORED),
    ):
        assert sr.seam_adjusted_budget_bytes(budget, CELL, rv) == (budget, 0)


def test_no_cell_means_abstain():
    budget = 15 * (1 << 30)
    assert sr.seam_adjusted_budget_bytes(budget, 0, _reserve(484 * MIB)) == (budget, 0)


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
    assert sr.seam_adjusted_budget_bytes(budget, CELL, rv) == (
        budget,
        0,
    ), "a first boot must size exactly as it does today"
    assert "COLD" in sr.describe(rv, "p") and "boot E" in sr.describe(rv, "p")


def test_the_record_round_trips_and_is_per_rank(tmp_path):
    args = _Args(str(tmp_path / "kv_budget-abc.json"))
    sr.write_seam_reserve(
        args, 1, BOOT_E_STAGING, 512.0, "arena tail 466 MiB", 8 * MIB, BOOT_G_T
    )
    sr.write_seam_reserve(
        args, 2, 1436 * MIB, 512.0, "arena tail 1436 MiB", 524 * MIB, BOOT_G_T
    )

    r1, r2 = sr.read_seam_reserve(args, 1), sr.read_seam_reserve(args, 2)
    assert r1.provenance == sr.PROVENANCE_STORED
    assert r1.fixed_bytes == BOOT_E_STAGING and r1.per_row_bytes == 512.0
    assert r1.have_bytes == 8 * MIB and r1.id_space == BOOT_G_T
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
    sr.write_seam_reserve(args, 1, BOOT_E_STAGING, 512.0, "first", 8 * MIB, BOOT_G_T)
    path = sr.record_path(args, 1)
    with open(path) as fh:
        assert json.load(fh)["fixed_bytes"] == BOOT_E_STAGING
    leftovers = [p for p in os.listdir(os.path.dirname(path)) if ".tmp" in p]
    assert not leftovers


def test_the_term_can_be_switched_off_as_a_value(tmp_path):
    args = _Args(str(tmp_path / "kv_budget-abc.json"))
    sr.write_seam_reserve(args, 1, BOOT_E_STAGING, 512.0, "measured", 8 * MIB, BOOT_G_T)
    os.environ[sr.ENV_ENABLE] = "0"
    rv = sr.read_seam_reserve(args, 1)
    assert rv.provenance == sr.PROVENANCE_DISABLED and rv.fixed_bytes == 0


def test_an_operator_override_wins_over_the_record(tmp_path):
    args = _Args(str(tmp_path / "kv_budget-abc.json"))
    sr.write_seam_reserve(args, 1, BOOT_E_STAGING, 512.0, "measured", 8 * MIB, BOOT_G_T)
    os.environ[sr.ENV_FIXED_MIB] = "900"
    rv = sr.read_seam_reserve(args, 1)
    assert rv.provenance == sr.PROVENANCE_OVERRIDE and rv.fixed_bytes == 900 * MIB


def test_every_symbol_the_callers_import_exists():
    """A boot died on ImportError because an edit removed measure_and_record
    while every unit test still passed -- the deleted function had no test
    that imported it BY THE NAME THE CALLER USES. These are those names, one
    per call site, so a refactor that drops one fails here instead of on
    metal."""
    import importlib

    m = importlib.import_module("sglang.srt.managers.phase_flip_seam_reserve")
    # scheduler._phase_flip_on_round
    assert callable(m.measure_and_record)
    # model_runner_kv_cache_mixin._seam_reserve / _seam_adjusted_budget
    assert callable(m.read_seam_reserve)
    assert callable(m.record_path)
    assert callable(m.describe)
    assert callable(m.seam_adjusted_budget_bytes)
    assert isinstance(m.LOG_PREFIX, str)
    # internals the measurement path needs
    assert callable(m.measure_at_rest)
    assert callable(m.write_seam_reserve)
    assert callable(m.seam_allowed_tokens)
