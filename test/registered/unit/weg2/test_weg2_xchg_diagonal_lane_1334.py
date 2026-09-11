# SPDX-License-Identifier: Apache-2.0
"""#1334 -- the DIAGONAL lane on the ``host`` arm, and the instrument that named it.

BOOT weg2xsn9 (`a0407794df`, 2026-09-11, record `BOOT_weg2xsn9_0911.md`) was the
first boot on which the on-card lane ever ARMED: the arm moved `ipc` -> `host`,
`why=oncard-not-drainable` went 18 -> 0 per group, the store-and-forward deposit
priced itself (`oncard_slot_mib=32 batches=1 slots=1 deposit=32 MiB, 3.84
GB/s`), and then **18 legs on P and 18 on D died with
`why=transport-failed:IndexError`**.

TWO DEFECTS, one boot, and this file is red on both at that SHA:

**(1) THE LOG COULD NOT SAY WHICH INDEX.**
`weight_exchange_shadow.py` recorded `transport-failed:{type(exc).__name__}` --
the type alone -- and `Traceback` was 0 in both groups because an observer never
raises. That is #1328's defect (fixed for the manifest and derivation arms after
it cost windows xsn5/xsn6/xsn7) surviving on the transport arm. `exc_note` now
puts the message and the raising SITE on the same field.

**(2) THE DIAGONAL HAS NO CARRIER ON THIS PATH.**
`weight_exchange_region.CROSS_PAIRS` is the six CROSS pairs and no diagonal, by
construction: `pair_id(c, c)` refuses with *"the on-card lane takes the IPC
bounce buffer and never a staging slot"*, and `sem_name` does
`src, dst = CROSS_PAIRS[int(pair)]`. On the `host` arm the on-card lane MUST be
store-and-forward, i.e. it needs a carrier and a handshake of its own -- so the
two halves of the design contradicted each other, and `ipc` could never reach
the contradiction. Operator ruling (plan AMENDMENT 5): the diagonal keeps its
own per-card file (`oncard_host_path`'s `oncard-<card>.bin`, "A SEPARATE FILE,
not a seventh staging pair") and gets its own per-card semaphores BESIDE the 24
cross ones -- no seventh cross pair, no widening of `CROSS_PAIRS` -- and any
diagonal request that reaches a cross-only site is refused BY NAME.

THE DOUBLES ARE THE EXECUTION SMOKE'S, imported: same fake card, same fake
device layer, same P-stage/D-shard models whose agreement is already pinned.
The only difference is the ARM.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402
from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402
from sglang.srt.weg2 import weight_exchange_transport as tp  # noqa: E402
from sglang.srt.weg2 import xchg_bounce as xb  # noqa: E402


# ===========================================================================
# (1) THE INSTRUMENT. Red-first: `exc_note` does not exist at a0407794df.
# ===========================================================================


def test_the_note_carries_the_message_and_the_raising_site():
    """A type alone is a hint; this is the finding."""
    def raiser():
        rows = [1]
        return rows[5]

    try:
        raiser()
    except BaseException as exc:  # noqa: BLE001
        note = sh.exc_note(exc)
    assert note.startswith("IndexError: "), note
    assert "list index out of range" in note, note
    # THE SITE, which is the half a type cannot carry: this file and the line
    # of the subscript, not of the `except`.
    assert "@ test_weg2_xchg_diagonal_lane_1334.py:" in note, note


def test_the_note_is_bounded_and_single_line():
    """It rides on an existing log line, so it may not wrap or run away."""
    try:
        raise ValueError("x " * 500 + "\n\nsecond paragraph")
    except BaseException as exc:  # noqa: BLE001
        note = sh.exc_note(exc, limit=64)
    assert len(note) <= 64, len(note)
    assert "\n" not in note and "  " not in note, repr(note)


def test_the_note_never_replaces_the_finding_with_its_own_failure():
    """An instrument that raises while describing a failure is worse than none."""
    class Hostile(Exception):
        def __str__(self):  # noqa: D105
            raise RuntimeError("this exception refuses to be printed")

    try:
        raise Hostile()
    except BaseException as exc:  # noqa: BLE001
        note = sh.exc_note(exc)
    assert note == "Hostile", note


def test_the_transport_arm_reports_through_the_note_and_not_the_bare_type():
    """The seam: the arm that cost boot weg2xsn9 its verdict.

    Source-level, because the arm is an `except BaseException` inside a 200-line
    function whose failure path needs a whole leg to drive -- and the property
    is exactly "which STRING is built there".
    """
    import inspect

    src = inspect.getsource(sh.shadow_transport)
    assert 'result.reason = f"transport-failed:{exc_note(exc)}"' in src, (
        "the transport arm must report message+site, never the bare type")
    assert 'transport-failed:{type(exc).__name__}' not in src


# ===========================================================================
# (2) THE DIAGONAL. Red-first: every one of these fails at a0407794df.
# ===========================================================================


def test_the_cross_pair_table_still_has_no_diagonal_and_says_so_by_name():
    """THE PREMISE, and it must not be "fixed" by widening CROSS_PAIRS.

    The operator ruling is explicit: no seventh cross pair. So `pair_id` keeps
    refusing the diagonal, and this test is what makes the refusal a contract
    rather than an accident of the tuple's contents.
    """
    assert len(xr.CROSS_PAIRS) == 6, xr.CROSS_PAIRS
    assert xr.N_PAIRS == 6
    for card in range(3):
        assert (card, card) not in xr.CROSS_PAIRS
        with pytest.raises(ValueError) as exc:
            xr.pair_id(card, card)
        assert "not a cross-card pair" in str(exc.value)


def test_a_diagonal_id_reaching_a_cross_only_site_is_REFUSED_BY_NAME():
    """MUTANT DIRECTION 1, and the boot's own killer.

    `sem_name` indexes `CROSS_PAIRS[int(pair)]`. Reached with anything that is
    not a cross-pair index it raised a bare `IndexError` -- which is what 36
    legs of weg2xsn9 reported, with no site and no traceback. A cross-only site
    must refuse by NAME so the next reader lands on the mechanism instead of on
    a subscript.
    """
    boot = "s4bdiag"
    for bad in (xr.N_PAIRS, xr.N_PAIRS + 5, -1):
        with pytest.raises(xr.Weg2XchgDiagonalHasNoCrossPair) as exc:
            xr.sem_name(boot, bad, 0, "empty")
        msg = str(exc.value)
        assert "W15" in msg, msg
        assert "diagonal" in msg.lower(), msg
        assert str(bad) in msg, msg
    # The six real pairs are untouched.
    for pair in range(xr.N_PAIRS):
        name = xr.sem_name(boot, pair, 0, "empty")
        src, dst = xr.CROSS_PAIRS[pair]
        assert name.endswith(f"-{src}-{dst}-0-empty"), name


def test_the_diagonal_has_its_OWN_semaphores_named_by_CARD():
    """Beside the 24, never inside them: 3 cards x 2 slots x {empty, full}."""
    boot = "s4bdiag"
    cross = xr.all_sem_names(boot)
    assert len(cross) == 24, len(cross)
    diag = xr.all_diagonal_sem_names(boot)
    assert len(diag) == 3 * xr.SLOTS_PER_PAIR * 2, diag
    # DISJOINT NAMES, or `create_semaphores`' unlink-before-create would
    # destroy a cross handshake while arming a diagonal one.
    assert not (set(cross) & set(diag)), set(cross) & set(diag)
    for card in range(3):
        for slot in range(xr.SLOTS_PER_PAIR):
            for kind in ("empty", "full"):
                n = xr.diagonal_sem_name(boot, card, slot, kind)
                assert n in diag
                assert f"-card{card}-{slot}-{kind}" in n, n
    # The census the launcher arms is the union, and it is complete.
    every = xr.all_region_sem_names(boot)
    assert set(every) == set(cross) | set(diag)
    assert len(every) == 24 + 3 * xr.SLOTS_PER_PAIR * 2


def test_the_diagonal_carrier_is_sized_from_the_PUBLISHED_slot_by_its_ONE_owner():
    """MUTANT DIRECTION 2: sized up, or sized from the module default.

    RENAMED AND RE-AIMED (#1333), not deleted: the claim is unchanged -- the
    deposit is sized from the ONE published value the ARM line charged as
    staging, `--weg2-xchg-oncard-slot-mib` x `SLOTS_PER_PAIR`, not the 32 MiB
    the leg happened to plan and not the module's 64 -- but it is now asked of
    the function that OWNS that arithmetic and that the product actually reads
    (`xchg_bounce.staging_bytes_per_card`, via
    `weight_updater._weg2_shadow_host_budget`).  It used to be asked of
    `tp.diagonal_carrier_bytes`, a second copy of the same number with zero
    production call sites, now deleted.  The MiB round-trip half of the old
    assertion moved to where the product performs it -- `validate_oncard_slot_mib`
    at the launcher's three flag sites and at `ONCARD_SLOT_BYTES_MAX` -- because
    that is the only path by which a slot reaches the charge.
    """
    for slot_mib in (128, 64, 32):
        nbytes = xb.staging_bytes_per_card(slot_mib * xr.MIB)
        assert nbytes == xr.SLOTS_PER_PAIR * slot_mib * xr.MIB, (slot_mib, nbytes)
    # THE PUBLISHED SLOT IS A VALIDATED MiB VALUE BEFORE IT GETS HERE, and a
    # non-MiB or out-of-range one is refused at that gate rather than at the
    # multiplication.
    assert tp.ONCARD_SLOT_BYTES_MAX % xr.MIB == 0
    with pytest.raises(tp.Weg2XchgOncardSlotRefused):
        tp.validate_oncard_slot_mib(0)
    with pytest.raises(tp.Weg2XchgOncardSlotRefused):
        tp.validate_oncard_slot_mib(7)
    # The deleted second authority stays deleted.
    assert not hasattr(tp, "diagonal_carrier_bytes")


def test_more_batches_than_the_charge_is_refused_BY_NAME_and_never_sized_up():
    """MUTANT DIRECTION 3 -- with the refusal NAME corrected (#1333).

    The old form of this test asserted `DEPOSIT_REASON_BATCHES` for a 3-batch
    plan and got it only because it passed `slots_max=xr.SLOTS_PER_PAIR`
    explicitly.  No production call site passes `slots_max`
    (`weight_exchange_shadow`'s leg planner is the only one), so the default
    `ONCARD_SLOTS_MAX` = 8 applies and a 3-batch plan is refused as
    `DEPOSIT_REASON_UNFUNDED` instead -- graded against the charge, which is the
    lever that actually binds.  The RULING is untouched (never sized up, always
    refused by name); what is corrected is which word fires, because
    `ONCARD_SLOTS_MAX` is the transport's row area and not the ledger's charge.
    """
    slot = 128 * xr.MIB
    budget = xb.staging_bytes_per_card(slot)
    graded = dict(slot_bytes=slot, budget_bytes=budget, mode=tp.ONCARD_MODE_HOST)
    assert tp.deposit_refusal_reason(batches=2, slots=2, **graded) == ""
    assert tp.deposit_refusal_reason(batches=3, slots=3, **graded) == \
        tp.DEPOSIT_REASON_UNFUNDED
    # BATCHES is still reachable -- above the ROW AREA's own max, the shape it
    # names -- so the correction narrows the claim rather than removing it.
    assert tp.deposit_refusal_reason(
        batches=tp.ONCARD_SLOTS_MAX + 1, slots=tp.ONCARD_SLOTS_MAX + 1,
        **graded) == tp.DEPOSIT_REASON_BATCHES
    # And the grading of the budget is UNCHANGED: slots x slot_bytes must fit
    # the budget it was handed.
    assert tp.deposit_refusal_reason(
        batches=2, slots=2, slot_bytes=slot, budget_bytes=budget - 1,
        mode=tp.ONCARD_MODE_HOST) == tp.DEPOSIT_REASON_UNFUNDED


# ===========================================================================
# (2b) THE RUNTIME HALF: the leg the product actually runs on the host arm.
# ===========================================================================

from .test_weg2_shadow_execution_smoke_1329 import (  # noqa: E402
    OTHER_ROWS,
    D_ROW,
    P_ROW,
    _plans,
    _write_agreed_pieces,
    armed,      # noqa: F401 -- imported FIXTURES, registered by import
    boot,       # noqa: F401
    chunked,    # noqa: F401
    region,     # noqa: F401
)
from .test_weg2_xchg_shadow_1273 import byte_sum  # noqa: E402
from .test_weg2_xchg_transport_1273 import (  # noqa: E402
    FakeDeviceOps,
    SLOT,
    WAVE,
)


def _store_forward_leg(armed_region, boot_nonce, tmp_path, *, monkeypatch,
                       inject_map=False):
    """One leg with the PRODUCT's own settings for the host arm.

    `oncard_drainable=False` is what the adapter sets
    (`weight_updater._weg2_shadow_hook`), and under S6 it MEANS
    store-and-forward; the execution smoke leaves it True and therefore never
    reaches the deposit at all -- which is why the smoke was green on both arms
    while boot weg2xsn9 died 36 times.

    `inject_map` (#1335) IS THE SECOND HALF OF THAT LESSON, and its default is
    the one that matters.  This double used to pass
    `uuid_of_card=("u0","u1","u2")` unconditionally -- the ONE keyword the
    adapter does not pass -- so it supplied what the product never supplied and
    could not fail in the product's direction: boots weg2xsn9 AND weg2xsn10
    both died on that map being empty while this very function was green.  The
    default now MIRRORS THE ADAPTER (the keyword is omitted, the region
    produces the map from `CUDA_VISIBLE_DEVICES`), and injecting a map is the
    explicit opt-in for a test that wants a specific one.
    """
    import threading

    monkeypatch.setenv(tp.ENV_ONCARD_MODE, tp.ONCARD_MODE_HOST)
    assert sh.resolve_shadow_oncard_mode() == tp.ONCARD_MODE_HOST
    p_plan, d_plan = _plans(armed_region, boot_nonce)
    epoch = f"{boot_nonce}.1"
    subset = sh.select_subset(p_plan.descs, leg=0, rotation=tuple(p_plan.classes))
    for row in OTHER_ROWS:
        sh.write_shadow_vote(armed_region, row, leg=0, vote=True,
                             classes_hash=subset.hash, need_mib=0,
                             plan_digest=p_plan.facts.digest,
                             piece_digest=p_plan.piece_digest)
    xr.create_semaphores(boot_nonce)
    sems = tp.SemSet(boot_nonce)
    p_ops = FakeDeviceOps(str(tmp_path / "card0"), rank=0)
    d_ops = FakeDeviceOps(str(tmp_path / "card0"), rank=0)
    p_lines, d_lines = [], []

    def inputs(hook, row, peer_row, gate_rows):
        extra = {"uuid_of_card": ("u0", "u1", "u2")} if inject_map else {}
        return sh.ShadowLegInputs(
            leg=0, epoch=epoch, direction="P->D", hook=hook, rank=0, row=row,
            peer_row=peer_row, device=0, card_uuid="u0",
            **extra, wave=WAVE, free_mib=8192,
            gate_rows=tuple(gate_rows),
            # THE PRODUCT'S TWO SETTINGS, verbatim from the adapter.
            oncard_drainable=False,
            host_bounce_budget_bytes=xr.SLOTS_PER_PAIR * 128 * xr.MIB,
        )

    try:
        d_ops.raw_malloc(0, 2 << 20)
        _write_agreed_pieces(p_ops, d_ops, same=True)

        def source():
            sh.run_leg_hook(
                inputs(sh.HOOK_SOURCE, P_ROW, D_ROW, (P_ROW,) + OTHER_ROWS[:2]),
                log=p_lines.append, plan=p_plan, region=armed_region, sems=sems,
                ops=p_ops, slot_bytes=SLOT, oncard_slot_bytes=SLOT,
                stripe_bytes=1 << 20, budget_s=20.0, gate_budget_s=20.0,
                hook_budget_s=60.0)

        t = threading.Thread(target=source, name="sf-source")
        t.start()
        result = sh.run_leg_hook(
            inputs(sh.HOOK_DESTINATION, D_ROW, P_ROW, tuple(range(xr.N_RANKS))),
            log=d_lines.append, plan=d_plan, region=armed_region, sems=sems,
            ops=d_ops, sum_bytes=byte_sum(d_ops), slot_bytes=SLOT,
            oncard_slot_bytes=SLOT, stripe_bytes=1 << 20, budget_s=20.0,
            gate_budget_s=20.0, hook_budget_s=60.0)
        t.join(120)
        assert not t.is_alive()
        return result, p_lines, d_lines
    finally:
        p_ops.close()
        d_ops.close()
        sems.close()
        xr.unlink_semaphores(boot_nonce)


def test_the_store_forward_leg_reaches_the_deposit_and_does_not_die_on_an_index(
        armed, chunked, boot, tmp_path, monkeypatch):
    """THE BOOT'S OWN SHAPE, at the desk. Red at a0407794df.

    weg2xsn9 measured `why=transport-failed:IndexError` on 18 legs per group
    here. What this asserts is the FIX's property and not merely "no
    IndexError": the leg either RUNS, or it refuses with a NAMED reason -- an
    unnamed subscript failure is what cost that window.
    """
    result, _p, d_lines = _store_forward_leg(
        armed, boot, tmp_path, monkeypatch=monkeypatch)
    assert result is not None
    reason = str(result.reason or "")
    assert "IndexError" not in reason, (reason, result.counters.errors)
    assert "list index out of range" not in " ".join(result.counters.errors)
    if not result.ran:
        # A refusal is acceptable -- BY NAME, with a W-code, never a subscript.
        assert reason.startswith(("no-", "oncard-", "deposit-", "hop-")) or "W" in reason, reason
    else:
        assert result.pieces > 0, result.line()
