# SPDX-License-Identifier: Apache-2.0
"""#1335 -- the card->uuid map: a required input that had NO PRODUCER.

BOOTS weg2xsn9 (`a0407794df`) and weg2xsn10 (`57f13aa72b`) both died on every
leg of every flip with the same subscript, and #1334's `exc_note` is what
finally named it:

    why=transport-failed:IndexError: tuple index out of range
      @ weight_exchange_transport.py:3116

THE ROOT, closed by the WRITER-ENUMERATION question rather than by a third
window.  Every site of the name in the whole tree, before this commit:

    weight_exchange_shadow.py:3411    uuid_of_card: Tuple[str, ...] = ()   DECLARATION
    weight_exchange_shadow.py:1798    uuid_of_card: Sequence[str],         parameter
    weight_exchange_shadow.py:2051    uuid_of_card=uuid_of_card,           pass-through
    weight_exchange_shadow.py:4051    uuid_of_card=tuple(inputs.uuid_of_card)   the ONLY reader
    weight_exchange_transport.py:2978 uuid_of_card: Sequence[str],         parameter
    weight_exchange_transport.py:3116 uuid_of_card[src_card], ...          the INDEX

**No writer.**  The one product constructor, `weight_updater.py:1881`, passes
`card_uuid=` (this rank's own, from NVML) and never `uuid_of_card=`, so the
field took its declared default `()` on every leg and the subscript was an
`IndexError` UNCONDITIONALLY.  A required input declared with an unusable
default is the #606 class, and the answer is to remove the default, not to
fill it in behind the caller's back.

**AND THE VALUE IS A LABEL.**  Its only consumer is
`PairStats(src_card, dst_card, uuid_of_card[src_card], uuid_of_card[dst_card])`
-- the `src_uuid`/`dst_uuid` of the per-pair acceptance line (R10, "keyed by
SOURCE and DESTINATION uuid, not aggregated").  So a purely descriptive lookup,
evaluated inside the data path, aborted the entire payload.  Both halves are
fixed here: the map gets a producer, and an unusable map is refused BY NAME
BEFORE a single thread starts.

**THE PRODUCER IS A MISSING READ, NOT MISSING DATA.**  `launcher.py:8834` is
`cvd = ",".join(c.uuid for c in cards)` and `build_env` gives that string to
EVERY rank of BOTH groups; `weight_updater.py:1494-1498` states the reason
verbatim ("all three card uuids, one string, both groups ... exactly so that
`weight_exchange_region`'s 'rank n of either group runs on cards[n]' holds"),
and `weight_exchange_region.py:120-121` documents the same invariant from the
other side.  So the map IS `CUDA_VISIBLE_DEVICES.split(",")` in card order, and
the module that owns the card ordering is the one place it may be read.

**WHY THIS FILE EXISTS ON TOP OF #1334's.**
`test_weg2_xchg_diagonal_lane_1334.py::test_the_store_forward_leg_reaches_the_deposit_and_does_not_die_on_an_index`
drives the product's own store-and-forward path and was GREEN at the SHA both
boots died on -- because its double INJECTS `uuid_of_card=("u0","u1","u2")`,
which is precisely the keyword the adapter does not pass.  A double that
supplies what the product never supplies cannot fail in the product's
direction.  The reproduction below therefore does the one thing that double did
not: **it omits the keyword the adapter omits.**
"""

from __future__ import annotations

import inspect
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402
from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402
from sglang.srt.weg2 import weight_exchange_transport as tp  # noqa: E402

from .test_weg2_shadow_execution_smoke_1329 import (  # noqa: E402
    armed,      # noqa: F401 -- imported FIXTURES, registered by import
    boot,       # noqa: F401
    chunked,    # noqa: F401
    region,     # noqa: F401
)
from .test_weg2_xchg_diagonal_lane_1334 import _store_forward_leg  # noqa: E402
from .test_weg2_xchg_transport_1273 import (  # noqa: E402
    FakeDeviceOps,
    SLOT,
    WAVE,
    dev_ptr,
    flat_desc,
    pattern,
    write,
)

#: The three uuids the doubles' fake cards carry.  Deliberately NOT in NVML's
#: ``GPU-…`` shape: the producer does not police the string form of a uuid (the
#: fork does not own NVML's spelling), and the identity check below catches an
#: ordinal-valued ``CUDA_VISIBLE_DEVICES`` anyway, because ``"0" != "u0"``.
FAKE_MAP = ("u0", "u1", "u2")


# ===========================================================================
# (1) THE PRODUCER.  One reader of the launcher's own string, in the module
#     that already owns "rank n runs on cards[n]".
# ===========================================================================


def test_the_region_produces_the_map_from_the_launchers_cvd_in_card_order():
    """Red at 57f13aa72b: `xr.uuid_of_card` does not exist."""
    got = xr.uuid_of_card(env="GPU-aaa,GPU-bbb,GPU-ccc")
    assert got == ("GPU-aaa", "GPU-bbb", "GPU-ccc"), got
    # CARD ORDER IS THE CONTRACT, not a convenience: the launcher builds the
    # string as ",".join(c.uuid for c in cards), so index == card ordinal.
    assert got[1] == "GPU-bbb"
    # Whitespace a shell may have left is stripped, never treated as a uuid.
    assert xr.uuid_of_card(env=" GPU-aaa , GPU-bbb ,GPU-ccc ") == got


@pytest.mark.parametrize("bad,why", [
    ("", "unset"),
    ("   ", "unset"),
    ("GPU-aaa,GPU-bbb", "count"),
    ("GPU-aaa,GPU-bbb,GPU-ccc,GPU-ddd", "count"),
    ("GPU-aaa,,GPU-ccc", "blank"),
])
def test_the_producer_refuses_by_name_and_never_guesses(bad, why):
    """No default, no padding, no truncation -- a NAMED refusal.

    Red at 57f13aa72b on the missing symbol.  The danger direction is a
    producer that pads a short list or silently drops a fourth entry: either
    one hands the transport a map whose index is not the card ordinal, which is
    a WRONG LABEL on the acceptance line and, worse, a wrong identity check.
    """
    with pytest.raises(xr.Weg2XchgCardUuidMapUnusable) as exc:
        xr.uuid_of_card(env=bad)
    text = str(exc.value)
    assert text.startswith("W23 Weg2XchgCardUuidMapUnusable"), text
    assert why in text, text
    # The refusal quotes WHAT IT READ -- a refusal that hides its input cannot
    # be acted on (the #1328 lesson, one arm further down).
    assert repr(bad) in text or "unset" in text, text


def test_the_producer_reads_the_environment_when_no_string_is_injected(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(FAKE_MAP))
    assert xr.uuid_of_card() == FAKE_MAP
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(xr.Weg2XchgCardUuidMapUnusable):
        xr.uuid_of_card()


def test_the_env_name_is_the_launchers_and_is_not_retyped():
    """One spelling of the variable, so the two ends cannot drift."""
    src = inspect.getsource(xr.uuid_of_card)
    assert "CUDA_VISIBLE_DEVICES" in src
    assert src.count("CUDA_VISIBLE_DEVICES") == 1, \
        "the variable is named once; a second literal is a second producer"


# ===========================================================================
# (2) THE TRANSPORT.  An unusable map is a NAMED refusal before any thread,
#     never a subscript halfway through a payload.
# ===========================================================================


def _one_leg(region_, boot_nonce, ops, *, uuid_of_card, card_uuid="u0", rank=0):
    payload = pattern(0x21, SLOT)
    src, dst = dev_ptr(0, 0x10000), dev_ptr(0, 0x80000)
    write(ops, src, payload)
    descs = [flat_desc(0, 0, len(payload), src_ptr=src, dst_ptr=dst,
                       name="oncard")]
    xr.create_semaphores(boot_nonce)
    sems = tp.SemSet(boot_nonce)
    try:
        return tp.run_leg(region_, sems, ops, row=0, rank=rank, device=0,
                          card_uuid=card_uuid, uuid_of_card=uuid_of_card,
                          descs=descs, is_source=True,
                          oncard_mode=tp.ONCARD_MODE_HOST, peer_row=3,
                          wave=WAVE, log=lambda _s: None,
                          vote_failure=lambda _e: None, budget_s=2.0,
                          slot_bytes=SLOT, oncard_slot_bytes=SLOT,
                          oncard_slots=5, oncard_store_forward=True)
    finally:
        sems.close()
        xr.unlink_semaphores(boot_nonce)


def test_run_leg_refuses_by_name_instead_of_indexerror_on_an_empty_map(
        region, boot, tmp_path):
    """THE BOOT'S EXACT INPUT.  Red at 57f13aa72b with an `IndexError`.

    `()` is what `ShadowLegInputs`' default handed the transport on every leg
    of weg2xsn9 and weg2xsn10.  An empty map must be refused by its own name;
    the whole cost of those two windows was that it was not.
    """
    ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        with pytest.raises(xr.Weg2XchgCardUuidMapUnusable) as exc:
            _one_leg(region, boot, ops, uuid_of_card=())
        text = str(exc.value)
        assert text.startswith("W23 Weg2XchgCardUuidMapUnusable"), text
        assert "count" in text, text
        # The two boots' own site is named, so a reader of this refusal does
        # not have to rediscover where the label was indexed.
        assert "PairStats" in text or "label" in text, text
    finally:
        ops.close()


def test_run_leg_refuses_when_the_map_disagrees_with_this_ranks_own_card(
        region, boot, tmp_path):
    """RANKS NEVER DISAGREE: a disagreement is a STOP, not a compensation.

    The map says card `rank` is `uuid_of_card[rank]`; NVML told the adapter
    this rank's card is `card_uuid`.  If those two differ, the region's card
    ordering and the adapter's identity are talking about different cards --
    the exact danger `weight_updater.py:1499-1505` calls "danger (a) in its
    plainest form" -- and a leg that proceeds would price and label another
    rank's card.  This check also subsumes an ORDINAL-valued
    `CUDA_VISIBLE_DEVICES` ("0,1,2"), because "0" is not an NVML uuid.
    """
    ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        with pytest.raises(xr.Weg2XchgCardUuidMapUnusable) as exc:
            _one_leg(region, boot, ops, uuid_of_card=("uX", "u1", "u2"),
                     card_uuid="u0", rank=0)
        text = str(exc.value)
        assert text.startswith("W23 Weg2XchgCardUuidMapUnusable"), text
        assert "disagree" in text, text
        assert "'uX'" in text and "'u0'" in text, text
    finally:
        ops.close()


def test_the_refusal_happens_before_a_single_byte_moves(region, boot, tmp_path):
    """A LABEL MAY NOT ABORT A HALF-DONE TRANSFER.

    The second half of the defect: the lookup sat inside the data path, so a
    bad label could in principle kill a leg after some pairs had copied.  The
    check belongs at the door -- proven against the fake device layer's own
    `issued` counter, which counts every `memcpy*_async` the transport asked
    for.  That counter is non-zero on the GOOD-map leg above, so this
    assertion can fail.
    """
    ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        before = int(ops.issued)
        with pytest.raises(xr.Weg2XchgCardUuidMapUnusable):
            _one_leg(region, boot, ops, uuid_of_card=())
        assert int(ops.issued) == before, \
            "the map was validated after the transport had already issued work"
    finally:
        ops.close()


def test_a_good_map_still_labels_the_pair_line_with_both_uuids(
        region, boot, tmp_path):
    """The label must still BE the label -- R10's line is keyed by uuid."""
    ops = FakeDeviceOps(str(tmp_path / "d"), rank=0)
    try:
        result = _one_leg(region, boot, ops, uuid_of_card=FAKE_MAP)
        assert result is not None
        for stats in result.pairs:
            assert stats.src_uuid == FAKE_MAP[stats.src_card], stats
            assert stats.dst_uuid == FAKE_MAP[stats.dst_card], stats
        # THE DENOMINATOR for the before-a-byte-moves test above: a usable map
        # lets this leg issue real copies, so a zero there is a property of the
        # refusal and not of a leg that never does anything.
        assert int(ops.issued) > 0, "this leg issued no copy at all"
    finally:
        ops.close()


def test_run_leg_keeps_the_map_as_a_required_parameter():
    """The transport does NOT reach for the environment on its own.

    One authority for "ask the env" -- the shadow hook -- and one for "is this
    map usable" -- here.  A default on `run_leg` would make the transport a
    second reader of `CUDA_VISIBLE_DEVICES`, which is how two producers of one
    fact start (`ein-Job-ein-Mover`).  This is also pinned by
    `test_weg2_xchg_transport_1273`'s required-parameter set; stated twice on
    purpose, because that test would go green if the name were merely renamed.
    """
    sig = inspect.signature(tp.run_leg)
    assert sig.parameters["uuid_of_card"].default is inspect.Parameter.empty
    assert "CUDA_VISIBLE_DEVICES" not in inspect.getsource(tp.run_leg)


# ===========================================================================
# (3) THE SHADOW HOOK.  The field loses its unusable default; `None` means
#     "the region is the authority", which is a contract and not a fallback.
# ===========================================================================


def test_the_leg_inputs_field_has_no_unusable_default():
    """Red at 57f13aa72b: the default is `()`, which can only ever raise."""
    field = sh.ShadowLegInputs.__dataclass_fields__["uuid_of_card"]
    assert field.default is None, field.default
    # An explicitly-passed empty tuple is still an ERROR, not a synonym for
    # None: `()` means "I have a map and it is empty", which is the boot's own
    # broken state, and it must keep refusing.
    assert xr.uuid_of_card.__doc__


def test_the_hook_resolves_the_map_from_the_one_producer():
    src = inspect.getsource(sh.run_leg_hook)
    assert "xr.uuid_of_card()" in src, \
        "the hook must ask the region, not build a map of its own"
    # The GUARD, not the absence of a conversion: `tuple(inputs.uuid_of_card)`
    # is right for a caller that HAS a map and a TypeError for `None`, so what
    # this pins is that the None case is taken first.
    assert "inputs.uuid_of_card is None" in src, \
        "the None case must be decided before the tuple() conversion"


def test_the_adapter_omits_the_keyword_and_that_is_why_none_must_resolve():
    """THE PRODUCT PATH, pinned at its source, so this cannot silently return.

    `weight_updater.py` is another seat's file this cycle, so the contract is
    asserted and not edited: the adapter constructs `ShadowLegInputs` WITHOUT
    `uuid_of_card=`, therefore the field's default is the only thing that can
    supply the map, therefore that default must resolve to the authority.  If a
    later change makes the adapter pass the map explicitly, this test fails and
    says which half moved -- which is the point.
    """
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    src = inspect.getsource(wu)
    head = src.split("ShadowLegInputs(", 1)
    assert len(head) == 2, "the adapter must construct ShadowLegInputs"
    body = head[1].split("\n            )", 1)[0]
    assert "card_uuid=card_uuid" in body, body
    assert "uuid_of_card=" not in body, (
        "the adapter now passes the map; the None-resolves-to-the-region "
        "contract in run_leg_hook is no longer the only producer -- reconcile "
        "the two before deleting this assertion")


# ===========================================================================
# (4) THE REGRESSION, in the product's own shape: omit what the adapter omits.
# ===========================================================================


def test_the_store_forward_leg_runs_with_the_map_the_adapter_actually_supplies(
        armed, chunked, boot, tmp_path, monkeypatch):
    """THE TWO BOOTS' WALL, AT THE DESK.  Red at 57f13aa72b.

    #1334's own store-and-forward test injects `uuid_of_card=("u0","u1","u2")`
    into `ShadowLegInputs` and was green while both boots died; this one leaves
    the keyword out exactly as `weight_updater.py:1881` does, and supplies the
    map the only way the product does -- through the launcher's
    `CUDA_VISIBLE_DEVICES`.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(FAKE_MAP))
    result, _p, _d = _store_forward_leg(
        armed, boot, tmp_path, monkeypatch=monkeypatch, inject_map=False)
    assert result is not None
    reason = str(result.reason or "")
    errors = " ".join(result.counters.errors)
    assert "IndexError" not in reason, (reason, result.counters.errors)
    assert "tuple index out of range" not in errors, errors
    assert "W23" not in reason, (reason, "the map came from the env and was usable")


def test_the_same_leg_refuses_by_name_when_the_launcher_gave_no_map(
        armed, chunked, boot, tmp_path, monkeypatch):
    """And the hermetic case: no `CUDA_VISIBLE_DEVICES`, a NAMED refusal.

    This is the shape a rank would hit if the launcher ever stopped exporting
    the string.  It must be a W-coded refusal naming the variable, never 36
    subscripts.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    result, _p, _d = _store_forward_leg(
        armed, boot, tmp_path, monkeypatch=monkeypatch, inject_map=False)
    assert result is not None
    reason = str(result.reason or "")
    errors = " ".join(result.counters.errors)
    assert "IndexError" not in reason, (reason, errors)
    assert "Weg2XchgCardUuidMapUnusable" in reason or "W23" in reason or \
        "Weg2XchgCardUuidMapUnusable" in errors, (reason, errors)


def test_the_1334_double_now_mirrors_the_adapter_by_default():
    """The fixture defect itself, pinned.

    A double whose default supplies a keyword the product never supplies is an
    instrument that cannot fail in the product's direction -- measured, at the
    cost of two windows.  `_store_forward_leg` therefore defaults to the
    adapter's shape, and injecting the map is the OPT-IN.
    """
    sig = inspect.signature(_store_forward_leg)
    assert "inject_map" in sig.parameters
    assert sig.parameters["inject_map"].default is False
