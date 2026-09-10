# SPDX-License-Identifier: Apache-2.0
"""#1273 S6-BOUNCE -- EXECUTION SMOKE for BOTH authoritative paths, driven
from the PRODUCT call sites.

Design of record: ``/spinning/gpu-arb/weg2/WEG2_REUSE_SPEC_0908.md`` section 10
plus PLAN_S6_BOUNCE_0911 AMENDMENT 2 (buffer per GROUP, a layer assembled
complete) and AMENDMENT 3 (the sizing census, which superseded section 10.2's
mean).  User law: ``memory/gewichtsaustausch-ziel-kein-dauer-hostram.md``.

WHY THIS FILE EXISTS AT ALL, in one measured sentence: the shadow slice was
built, reviewed and booted three times before anyone noticed it had never
reached its transport (#1329, ``ran=no why=no-plan`` on 24/24 legs), because
every test drove the module FUNCTIONS and none drove the mixin METHODS the
product actually calls.  So the rule of #1329's smoke is the rule here:
**every step is taken through the method the scheduler calls.**

RED ON THE BASE (``f03e405e7d``): no ``weight_exchange_bounce`` module and no
``_weg2_xchg_*`` method, so the import fails and the file is one collection
error.  That red is an ABSENCE red; what the green run adds is a byte-exact
comparison, per descriptor, of every destination row against the source row it
must equal -- through the product's own call sites.

THE GEOMETRY MODELS THE WIDEST LAYER, NOT THE MEAN ONE (AMENDMENT 3).
Measured over this checkpoint's 18 safetensors headers, ``layers.0`` is
756,323,776 B = 721 MiB against a 386.9 MiB mean -- and it is a DIFFERENT
SHAPE, not merely a bigger one, because it carries the linear-attention family
(``in_proj_qkv``/``z``/``a``/``b``, ``conv1d``, ``A_log``, ``dt_bias``) on top
of the MLP triple.  A double that modelled only the MLP triple would test the
wrong form, so layer 0 here carries the GDN family too, including the two 1-D
tensors whose shape an assembler most easily gets wrong.  ``lm_head`` is
modelled as well, at 4x the widest layer, because it is the case the
AMENDMENT 3 decide is about: an UNLAYERED class must BAND, not refuse.

VERIFICATION IS GENERIC AND PER-DESCRIPTOR.  Every destination row is compared
against the source row at that descriptor's own offsets, so a new class added
to the plan is covered without a hand-written expectation -- and a
hand-written expectation is exactly how a class quietly stops being checked.
The source pattern has a 251-byte period against a 64-byte row, so any wrong
offset differs in nearly every byte rather than by luck.

Classes below are NAMESPACES FOR READING ONLY -- pytest collects the
module-level ``test_*`` functions, exactly as #1329's smoke does.  A class with
test methods collects nothing under this repo's pytest config, which is the
instrument-that-cannot-fail shape this campaign has already paid for.
"""

from __future__ import annotations

import ctypes
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402

#: ABSENCE RED on the base: this import is what B3 makes resolvable.  A plain
#: import rather than an ``importorskip`` on purpose -- a skip would read as
#: green in every summary, and the whole point of the red run is a number that
#: is not zero.
from sglang.srt.weg2 import weight_exchange_bounce as bx  # noqa: E402
from sglang.srt.weg2 import xchg_bounce as xb  # noqa: E402

from .test_weg2_xchg_transport_1273 import (  # noqa: E402
    FakeDeviceOps,
    _fresh_boot,
    dev_ptr,
)

# ---------------------------------------------------------------------------
# THE ADDRESS LAYOUT.  One layer per STEP, laid out so no two classes overlap
# (they did in the first draft, and an overlapping double proves nothing: the
# destination of one class is the source of another).
# ---------------------------------------------------------------------------

COLS = 32
ITEMSIZE = 2
ROW = COLS * ITEMSIZE                      # 64 B, the indivisible run
N_DST = 3
N_LAYERS = 4

QKV_ROWS_FULL = 24
QKV_ROWS_SHARD = QKV_ROWS_FULL // N_DST    # 8
DOWN_ROWS = 8

P_BASE = 0x400000
D_BASE = 0x600000
STEP = 0x8000
QKV_OFF = 0x0000                           # 24 rows -> 0x600
DOWN_OFF = 0x1000                          #  8 rows -> 0x200
GDN_OFF = 0x2000                           # 5 x 24 rows -> 0x1E00
GDN_1D_OFF = 0x4000                        # 2 x 1 row
LM_HEAD_OFF = 0x60000

GDN_FAMILY = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "conv1d")
GDN_1D = ("A_log", "dt_bias")

#: 4x the widest layer, i.e. 4 bands -- ``lm_head``'s measured ratio
#: (2425 MiB against a 721 MiB layer).
LM_HEAD_ROWS = 696

TAG = "weights.chunk.0"

#: Group-wide bytes, by hand, so the constant and the plan cannot drift --
#: ``test_the_slot_is_the_widest_layer`` asserts they agree.
LAYER0_BYTES = (N_DST * QKV_ROWS_SHARD * ROW          # qkv
                + N_DST * DOWN_ROWS * ROW             # down_proj
                + len(GDN_FAMILY) * N_DST * QKV_ROWS_SHARD * ROW
                + len(GDN_1D) * N_DST * ROW)          # 11136
PLAIN_LAYER_BYTES = N_DST * QKV_ROWS_SHARD * ROW + N_DST * DOWN_ROWS * ROW
LM_HEAD_BYTES = LM_HEAD_ROWS * ROW                    # 44544 = 4 x LAYER0

#: THE PRODUCTION SHAPE: one depth-slot holds the widest LAYER complete
#: (AMENDMENT 2), and unlayered classes band (the AMENDMENT 3 decide).
SLOT_BYTES = LAYER0_BYTES
DEPTH = 2

#: The MEAN over the layered units, and a slot strictly between it and the
#: widest layer.  Both exist only so the mean/max distinction is testable: a
#: refusal graded on either fires below the mean, so only this window tells
#: them apart.  Measured equivalent on the real checkpoint: mean 386.9 MiB,
#: widest 721 MiB.
MEAN_LAYER_BYTES = (LAYER0_BYTES + (N_LAYERS - 1) * PLAIN_LAYER_BYTES) // N_LAYERS
SLOT_BETWEEN = (MEAN_LAYER_BYTES + LAYER0_BYTES) // 2


def _p_ptr(layer: int, off: int) -> int:
    return dev_ptr(0, P_BASE + layer * STEP + off)


def _d_ptr(rank: int, layer: int, off: int) -> int:
    return dev_ptr(rank, D_BASE + layer * STEP + off)


def _sharded(layer: int, name: str, off: int, rows_full: int):
    """One source, N_DST destinations, each taking a DIFFERENT row band.

    The row map is the whole danger of path (b): a destination that copies from
    the wrong source offset gets plausible bytes of the wrong shard, and no
    length check anywhere would notice.
    """
    shard = rows_full // N_DST
    return [
        wx.XchgDesc(
            tag=TAG, src_rank=0, dst_rank=rank, param_name=name,
            kind=wx.STRIDED2D, nbytes=shard * ROW,
            rows=shard, run_bytes=ROW, spitch=ROW, dpitch=ROW,
            src_off=rank * shard * ROW, dst_off=0,
            src_ptr=_p_ptr(layer, off), dst_ptr=_d_ptr(rank, layer, off),
        )
        for rank in range(N_DST)
    ]


def _replicated(layer: int, name: str, off: int, rows: int):
    """Byte-identical on every card -- the class path (a) may carry."""
    return [
        wx.XchgDesc(
            tag=TAG, src_rank=0, dst_rank=rank, param_name=name,
            kind=wx.STRIDED2D, nbytes=rows * ROW,
            rows=rows, run_bytes=ROW, spitch=ROW, dpitch=ROW,
            src_off=0, dst_off=0,
            src_ptr=_p_ptr(layer, off), dst_ptr=_d_ptr(rank, layer, off),
        )
        for rank in range(N_DST)
    ]


def _layer_descs(layer: int, *, gdn: bool):
    out = _sharded(layer, f"model.layers.{layer}.self_attn.qkv_proj.weight",
                   QKV_OFF, QKV_ROWS_FULL)
    out += _replicated(layer, f"model.layers.{layer}.mlp.down_proj.weight",
                       DOWN_OFF, DOWN_ROWS)
    if not gdn:
        return out
    # The linear-attention family: what makes layer 0 the WIDEST and a
    # different SHAPE (AMENDMENT 3).
    for i, fam in enumerate(GDN_FAMILY):
        out += _sharded(layer, f"model.layers.{layer}.linear_attn.{fam}.weight",
                        GDN_OFF + i * QKV_ROWS_FULL * ROW, QKV_ROWS_FULL)
    # The 1-D tensors: one row, replicated.  Rows-only shape, no column split.
    for i, fam in enumerate(GDN_1D):
        out += _replicated(layer, f"model.layers.{layer}.linear_attn.{fam}",
                           GDN_1D_OFF + i * ROW, 1)
    return out


def _lm_head_descs():
    """An UNLAYERED class WIDER than one layer -- the decide's own case."""
    shard = LM_HEAD_ROWS // N_DST
    return [
        wx.XchgDesc(
            tag=TAG, src_rank=0, dst_rank=rank, param_name="lm_head.weight",
            kind=wx.STRIDED2D, nbytes=shard * ROW,
            rows=shard, run_bytes=ROW, spitch=ROW, dpitch=ROW,
            src_off=rank * shard * ROW, dst_off=0,
            src_ptr=dev_ptr(0, P_BASE + LM_HEAD_OFF),
            dst_ptr=dev_ptr(rank, D_BASE + LM_HEAD_OFF),
        )
        for rank in range(N_DST)
    ]


def _all_descs(*, lm_head: bool = False, n_layers: int = N_LAYERS):
    """The plan.  ``layers.0`` carries the GDN family, so it is the WIDEST."""
    out = []
    for layer in range(n_layers):
        out.extend(_layer_descs(layer, gdn=(layer == 0)))
    if lm_head:
        out.extend(_lm_head_descs())
    return out


# ---------------------------------------------------------------------------
# Seeding and GENERIC verification.
# ---------------------------------------------------------------------------

#: A 251-byte period against a 64-byte row: 251 is prime and coprime with 64,
#: so a shift by any small multiple of a row changes nearly every byte.  A
#: constant fill is the instrument that cannot fail -- every wrong offset would
#: still compare equal.
_PATTERN = bytes(((i * 97 + 13) % 256) for i in range(251))


def _seed_source(ops: FakeDeviceOps) -> None:
    """Tile the whole source card with the pattern, keyed on ABSOLUTE offset."""
    from .test_weg2_xchg_transport_1273 import FAKE_DEV_BYTES

    base = ops.real(dev_ptr(0, 0))
    buf = (_PATTERN * (FAKE_DEV_BYTES // len(_PATTERN) + 1))[:FAKE_DEV_BYTES]
    ctypes.memmove(base, buf, FAKE_DEV_BYTES)


def _mismatched_rows(ops: FakeDeviceOps, descs) -> list:
    """Every (param, dst_rank, row) whose destination bytes are wrong.

    PER DESCRIPTOR, from the descriptor's OWN offsets, so any class in the plan
    is covered without a bespoke expectation.  This is the byte proof; the
    leg's own ``verdict`` is an accounting verdict and deliberately claims less.
    """
    bad = []
    for d in descs:
        for r in range(int(d.rows)):
            src = ctypes.string_at(
                ops.real(int(d.src_ptr)) + int(d.src_off) + r * int(d.spitch),
                int(d.run_bytes))
            dst = ctypes.string_at(
                ops.real(int(d.dst_ptr)) + int(d.dst_off) + r * int(d.dpitch),
                int(d.run_bytes))
            if src != dst:
                bad.append((d.param_name, d.dst_rank, r))
    return bad


# ---------------------------------------------------------------------------
# Doubles the product methods need, and NOTHING more.
# ---------------------------------------------------------------------------


class _FakeRunner:
    def __init__(self, model=None) -> None:
        self.model = model


class _FakeWorker:
    def __init__(self, model=None) -> None:
        self.model_runner = _FakeRunner(model)


def _manager():
    """The product object, constructed with the six fields it requires."""
    return wu.SchedulerWeightUpdaterManager(
        tp_worker=_FakeWorker(), draft_worker=None, tp_cpu_group=None,
        memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
        is_fully_idle=lambda *a, **k: True,
    )


@pytest.fixture()
def boot():
    return _fresh_boot()


@pytest.fixture()
def armed(monkeypatch, boot):
    """``--weg2-weight-source exchange``, i.e. the arm that owns the bytes."""
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    monkeypatch.setenv(xr.ENV_REGION_BOOT, boot)
    assert wx.exchange_armed() is True
    return boot


@pytest.fixture()
def seeded(tmp_path):
    ops = FakeDeviceOps(str(tmp_path), 0)
    _seed_source(ops)
    return ops


def _run_bounce(mgr, ops, boot_nonce, root, *, descs=None, depth=DEPTH,
                slot_bytes=SLOT_BYTES, terms=None):
    """One call, through the PRODUCT method."""
    return mgr._weg2_xchg_bounce_leg(
        descs=_all_descs() if descs is None else descs,
        ops=ops, boot_nonce=boot_nonce, slot_bytes=slot_bytes, depth=depth,
        terms=terms, shm_root=root,
    )


# ===========================================================================
# STEP 1 -- THE UNIT MAP.  What one deposit/collect pair moves.
# ===========================================================================


class TheUnitMap:
    """Namespace only; the collected tests are the functions below."""


def test_units_group_by_layer_not_by_parameter():
    """A unit is a LAYER, group-wide: every descriptor of every rank."""
    units = bx.plan_units(_all_descs())
    assert len(units) == N_LAYERS
    assert all(u.is_layer for u in units)
    # Layer 0 carries the GDN family on top of the MLP triple, so it has more
    # descriptors than the others -- the AMENDMENT 3 shape, asserted.
    assert len(units[0].descs) > len(units[1].descs)


def test_the_slot_is_the_widest_layer_and_that_is_layer_zero():
    """AMENDMENT 3's finding, in this double's own arithmetic.

    The hand-written constant and the plan's own sum must agree, or the
    fixture and the code under test are describing different geometries.
    """
    units = bx.plan_units(_all_descs())
    assert units[0].nbytes == LAYER0_BYTES
    assert units[1].nbytes == PLAIN_LAYER_BYTES
    assert LAYER0_BYTES > PLAIN_LAYER_BYTES
    widest = bx.widest_unit(units)
    assert widest.key[1] == "layers.0"
    assert widest.nbytes == SLOT_BYTES


def test_an_unlayered_class_is_not_a_layer():
    """The AMENDMENT 3 decide rests on this distinction, so it is pinned."""
    units = {u.key[1]: u for u in bx.plan_units(_all_descs(lm_head=True))}
    assert units["layers.0"].is_layer is True
    assert units["lm_head.weight"].is_layer is False
    assert units["lm_head.weight"].nbytes == LM_HEAD_BYTES
    assert LM_HEAD_BYTES > SLOT_BYTES        # 4x, as measured


def test_a_run_wider_than_the_slot_is_refused():
    """The batcher's floor: no cut of a 2-D copy may go below one ROW.

    Measured 17408 B on the real checkpoint, so at any sane slot this cannot
    fire -- which is exactly why it must be tested rather than trusted.
    Delegated to the transport's own W68 so there is one producer of the
    verdict.
    """
    descs = _all_descs()
    bx.refuse_if_slot_short(SLOT_BYTES, descs)
    # NOTE, measured while writing this test: ``Weg2XchgPlanDisagree`` exists
    # TWICE -- ``weight_exchange.py:201`` and ``weight_exchange_region.py:260``
    # -- as two unrelated ``RuntimeError`` subclasses both documented as W68.
    # The batcher raises the REGION's, so that is the one asserted here.  A
    # test that caught the other would have gone green on the wrong identity;
    # unifying them is not this slice's change.
    with pytest.raises(xr.Weg2XchgPlanDisagree) as e:
        bx.refuse_if_slot_short(ROW - 1, descs)
    assert "run_bytes" in str(e.value)


def test_the_widest_run_is_reported_in_bytes():
    assert bx.widest_run(_all_descs()) == ROW


# ===========================================================================
# STEP 2 -- PATH (b) THROUGH THE PRODUCT CALL SITES, byte-exact.
# ===========================================================================


class PathBAssemblesAndEveryCardTakesItsSlice:
    """Namespace only; the collected tests are the functions below."""


def test_every_destination_row_equals_its_source_row(tmp_path, armed, seeded):
    """The whole slice, end to end, through the manager's own method.

    Source deposits every unit; each of three destinations collects ITS OWN
    band; then EVERY destination row is compared against the source row it
    must equal, per descriptor, including the GDN family and its two 1-D
    tensors.  This is the byte proof.
    """
    result = _run_bounce(_manager(), seeded, armed, str(tmp_path))

    assert result.units == N_LAYERS
    assert result.deposited_bytes == result.collected_bytes
    assert result.verdict == "MATCH"
    assert _mismatched_rows(seeded, _all_descs()) == []


def test_an_unlayered_class_wider_than_the_slot_bands_and_stays_exact(
        tmp_path, armed, seeded):
    """THE AMENDMENT 3 DECIDE, as the case it was decided for.

    ``lm_head`` is 4x the widest layer.  It must NOT refuse -- refusing would
    force a slot sized on ``max(layer, unlayered)``, which measures 4850 MiB
    of buffer, a 5.11 GiB bounce total and 8.4x instead of AMENDMENT 3's own
    1826 MiB and 24x.  It must band, stay byte-exact, and SAY on the line that
    it banded, so "which classes are not resident whole" is answerable from
    the log.
    """
    descs = _all_descs(lm_head=True)
    result = _run_bounce(_manager(), seeded, armed, str(tmp_path), descs=descs)

    assert result.verdict == "MATCH"
    assert _mismatched_rows(seeded, descs) == []
    assert result.banded == ("lm_head.weight",)
    assert "banded_units=lm_head.weight" in result.line()
    # 4 bands for lm_head, one for each layer: the ratio the census measured.
    assert result.bands == N_LAYERS + LM_HEAD_BYTES // SLOT_BYTES


def test_a_layer_above_the_slot_refuses_but_an_unlayered_class_bands(
        tmp_path, armed, seeded):
    """The decide's two halves, in one test, because they are one decision.

    A LAYER above the slot is a refusal (AMENDMENT 2: assembled complete); an
    unlayered class above the slot bands.  A build that treated both the same
    way -- either refusing both or banding both -- passes half of this test
    and fails the other half.
    """
    # Half one: the unlayered class alone is fine at this slot.
    bx.refuse_if_plan_exceeds_slot(SLOT_BYTES, _all_descs(lm_head=True))

    # Half two: a slot BETWEEN THE MEAN AND THE WIDEST layer must refuse.
    #
    # THE SLOT IS BETWEEN THEM ON PURPOSE, and it is a measured lesson: with a
    # slot below the MEAN (PLAIN_LAYER_BYTES, the first draft) a refusal graded
    # on the mean and a refusal graded on the max BOTH fire, so a mutant that
    # replaced the widest with the mean passed all twenty tests.  Only a slot
    # the mean clears and the widest does not separates the two -- which is
    # section 10.5's whole sentence ("the bound is the WIDEST layer, not the
    # mean") and the direction xchg_bounce's own mutants cover on the arm side.
    assert MEAN_LAYER_BYTES < SLOT_BETWEEN < LAYER0_BYTES
    with pytest.raises(xb.Weg2XchgBounceUnderCovered) as e:
        bx.refuse_if_plan_exceeds_slot(SLOT_BETWEEN, _all_descs())
    msg = str(e.value)
    assert "W71" in msg
    assert "layers.0" in msg
    assert "never the mean" in msg

    # And the leg must not start on such a slot: the refusal has to arrive
    # BEFORE the buffer is mapped, which is W71's whole reason for existing.
    before = bx.registered_bounce_bytes()
    with pytest.raises(xb.Weg2XchgBounceUnderCovered):
        _run_bounce(_manager(), seeded, armed, str(tmp_path),
                    slot_bytes=SLOT_BETWEEN)
    assert bx.registered_bounce_bytes() == before


def test_host_residency_does_not_grow_with_the_number_of_layers(
        tmp_path, armed, seeded):
    """THE USER LAW, as the only assertion that states it scale-free.

    "die 27gb an layerbytes muessen nicht mehr dauerhaft im systemram gehalten
    werden": the host bytes held at any instant must be the BUFFER, so
    doubling the layer count must double the bytes MOVED and leave the host
    residency untouched.

    Stated this way rather than as ``peak < deposited``, which was the first
    draft and was measured wrong: in a four-layer miniature whose widest layer
    is 3.6x the others, two slots legitimately exceed the whole image, so that
    comparison failed on a correct build.  On the real checkpoint it holds
    trivially (1442 MiB against 27.52 GiB) -- which is exactly why it was the
    wrong assertion to rest the law on: it would have passed for the wrong
    reason there and told us nothing.
    """
    four = _run_bounce(_manager(), seeded, armed, str(tmp_path))
    eight = _run_bounce(_manager(), seeded, armed, str(tmp_path),
                        descs=_all_descs(n_layers=2 * N_LAYERS))

    assert eight.units == 2 * four.units
    assert eight.deposited_bytes > four.deposited_bytes
    # The whole law, in one line: twice the image, the same host residency.
    assert eight.host_bytes_peak == four.host_bytes_peak == DEPTH * SLOT_BYTES


def test_a_unit_is_one_band_when_the_slot_covers_the_widest_layer(
        tmp_path, armed, seeded):
    """AMENDMENT 2 as an observable: COMPLETE assembly means bands == units.

    The user law says a layer is assembled "vollstaendig".  Under a slot that
    covers the widest layer that is an identity, not an aspiration, and the
    line prints both numbers so a config where it stops holding is visible in
    the log instead of being argued.
    """
    result = _run_bounce(_manager(), seeded, armed, str(tmp_path))
    assert result.bands == result.units == N_LAYERS
    assert result.banded == ()


def test_the_line_prints_the_expression_and_not_only_the_total(
        tmp_path, armed, seeded):
    """Section 10.2: "the arming line must PRINT the expression's terms"."""
    result = _run_bounce(_manager(), seeded, armed, str(tmp_path))
    line = result.line()
    assert line.startswith("WEG2-XCHG-BOUNCE-LEG ")
    for field in ("units=", "bands=", "widest_unit_bytes=", "slot_bytes=",
                  "depth=", "bounce_total_bytes=", "widest_run_bytes=",
                  "banded=", "overlap="):
        assert field in line, field


def test_the_leg_geometry_comes_from_the_arm_term_not_from_this_module():
    """Section 10.8's "one reader, or they drift", as a test.

    ``bounce_terms`` sets ``buffer_bytes = widest_layer_bytes * depth``, so the
    per-slot width IS the widest layer.  ``leg_geometry`` must return exactly
    that -- a module that recomputed a size would be the second ledger.

    THE WIDEST AND THE MEAN ARE DELIBERATELY DIFFERENT HERE, and that is a
    measured lesson rather than a flourish: with them equal, a mutant that
    returned ``terms.mean_layer_bytes`` from ``leg_geometry`` passed every
    test in this file.  The fixture must separate the two numbers or this test
    cannot see the one defect it is for -- the same mean-instead-of-max
    direction ``xchg_bounce``'s own mutants cover on the arm side.
    """
    terms = xb.bounce_terms(
        bytes_per_direction=PLAIN_LAYER_BYTES * N_LAYERS, n_layers=N_LAYERS,
        widest_layer_bytes=LAYER0_BYTES, pairs=6, depth=DEPTH,
        slot_bytes=xb.SLOT_BYTES_DEFAULT,
    )
    assert terms.mean_layer_bytes == PLAIN_LAYER_BYTES
    assert terms.mean_layer_bytes != terms.widest_layer_bytes
    assert bx.leg_geometry(terms) == (LAYER0_BYTES, DEPTH)
    assert terms.covers_widest_layer is True


def test_the_leg_runs_on_the_arm_term_alone(tmp_path, armed, seeded):
    """The normal way in: the ARM's priced decision, no hand-set sizes.

    This is the call shape the launcher will use, so it must be the one with
    an executing test -- a size passed only by tests is a size the product
    never exercises.
    """
    terms = xb.bounce_terms(
        bytes_per_direction=PLAIN_LAYER_BYTES * N_LAYERS, n_layers=N_LAYERS,
        widest_layer_bytes=LAYER0_BYTES, pairs=6, depth=DEPTH,
        slot_bytes=xb.SLOT_BYTES_DEFAULT,
    )
    result = _manager()._weg2_xchg_bounce_leg(
        descs=_all_descs(), ops=seeded, boot_nonce=armed, terms=terms,
        shm_root=str(tmp_path),
    )
    assert result.slot_bytes == LAYER0_BYTES
    assert result.depth == DEPTH
    assert result.verdict == "MATCH"
    assert _mismatched_rows(seeded, _all_descs()) == []


def test_the_leg_refuses_to_invent_a_size(tmp_path, armed, seeded):
    """Neither a term nor an explicit pair is a refusal, never a default.

    A default here would be a second sizing authority, which is the
    two-ledgers defect; and it would be the one that runs when the launcher
    forgets to pass the measured figure.
    """
    with pytest.raises(ValueError) as e:
        _manager()._weg2_xchg_bounce_leg(
            descs=_all_descs(), ops=seeded, boot_nonce=armed,
            shm_root=str(tmp_path),
        )
    assert "one owner" in str(e.value)


# ===========================================================================
# STEP 3 -- PATH (a) IS AUTHORITATIVE, not an observer.
# ===========================================================================


class PathAIsTheAuthorityForTheBytesItCarries:
    """Namespace only; the collected tests are the functions below."""


def test_path_a_changes_the_destinations_live_storage(tmp_path, armed, seeded):
    """The shadow re-targets descriptors into its own buffer; the authority
    does NOT -- it hands the transport the ORIGINAL destination pointers.

    That one difference is path (a)'s whole authority question, so the
    assertion is on the destination's LIVE address, not on a report.
    """
    descs = _replicated(0, "model.layers.0.mlp.down_proj.weight",
                        DOWN_OFF, DOWN_ROWS)
    result = _manager()._weg2_xchg_agreed_leg(
        descs=descs, ops=seeded, boot_nonce=armed, shm_root=str(tmp_path),
    )
    assert result.verdict == "MATCH"
    assert _mismatched_rows(seeded, descs) == []


def test_path_a_carries_only_the_agreed_names(tmp_path):
    """The filter, and it is a filter rather than a transport.

    Agreement is the manifest reconciliation's verdict, not something this
    module recomputes, so the only thing under test is that the descriptor set
    is narrowed to the names it was given.
    """
    descs = _all_descs()
    agreed = bx.agreed_descs(descs, ["model.layers.0.mlp.down_proj.weight"])
    assert agreed
    assert {d.param_name for d in agreed} == {
        "model.layers.0.mlp.down_proj.weight"}
    assert len(agreed) < len(descs)


# ===========================================================================
# THE CAN-FAIL CONTROLS.  A digest that cannot go red is not an instrument.
# ===========================================================================


class TheMutantsGoRed:
    """Namespace only; the collected tests are the functions below."""


def test_a_slice_from_the_wrong_offset_is_caught(tmp_path, armed, seeded):
    """THE defect class of path (b) (spec section 2.4 paid for it twice).

    Every sharded destination takes row band 0 instead of its own.  Lengths,
    totals and the deposit all stay correct, so only a comparison against the
    RIGHT rows can see it -- and it must, for the GDN family as well as for
    qkv, which is why the check is per descriptor.
    """
    descs = _all_descs()
    broken = [d.replace(src_off=0) if d.src_off else d for d in descs]
    _run_bounce(_manager(), seeded, armed, str(tmp_path), descs=broken)
    bad = _mismatched_rows(seeded, descs)
    assert bad, "a wrong source offset produced identical bytes -- the " \
                "pattern or the comparison cannot fail"
    # Ranks 1 and 2 of every sharded class are wrong; rank 0's band IS band 0.
    assert {r for _, r, _ in bad} == {1, 2}
    assert {"qkv_proj" in p or "linear_attn" in p for p, _, _ in bad} == {True}


def test_a_buffer_that_is_never_released_is_caught(tmp_path, armed, seeded):
    """The leg must give the host bytes back, or the law is unmet.

    ``pinned_host_budget`` is the #550 single owner of "may this pinned host
    buffer exist", so a leg that returns without releasing its post charges
    the next admission for bytes nobody holds.  The assertion is on the
    registry, not on a flag the leg sets about itself.
    """
    before = bx.registered_bounce_bytes()
    _run_bounce(_manager(), seeded, armed, str(tmp_path))
    assert bx.registered_bounce_bytes() == before


def test_depth_one_cannot_overlap_and_says_so(tmp_path, armed, seeded):
    """Section 10.4's falsifiable budget, as a verdict rather than prose.

    Depth 1 is a correct transfer and a DEGRADED one: nothing can be
    assembled while the previous unit drains, so the x4 link idles between
    units.  The leg must report ``overlap=none`` instead of claiming ok -- a
    build that printed ``overlap=ok`` for depth 1 would satisfy the acceptance
    line while failing the thing it measures.
    """
    result = _run_bounce(_manager(), seeded, armed, str(tmp_path), depth=1)
    assert result.verdict == "MATCH"
    assert "overlap=none" in result.line()
    assert _mismatched_rows(seeded, _all_descs()) == []


def test_a_destination_with_no_source_is_refused(tmp_path, armed, seeded):
    """W74: assembly stages a source, it does not create one.

    A descriptor whose ``src_ptr`` is absent would otherwise be staged from
    address 0 or skipped, and the destination would be served whatever the
    remap left in those pages -- undefined weights, silently.
    """
    descs = _all_descs()
    holed = [descs[0].replace(src_ptr=None)] + list(descs[1:])
    with pytest.raises(wx.Weg2XchgSourceMissing) as e:
        _run_bounce(_manager(), seeded, armed, str(tmp_path), descs=holed)
    assert "W74" in str(e.value)


# ===========================================================================
# STEP 4 -- WHAT "STAGING" MEANS, and the three answers the tree gives.
#
# NO REFUSAL IS BUILT HERE, deliberately, and an earlier commit of mine that
# built one was WRONG and is reverted: it graded `terms.staging_bytes` against
# `xr.DATA_BYTES` -- the CROSS region -- while the launcher feeds
# `pairs=xr.N_CARDS`, which makes the term model the PER-CARD ON-CARD lane.
# Two different payloads that both happen to equal 384 MiB today, so the check
# passed by the very coincidence it was written to catch. The lesson is the
# one the operator gave and I then broke: read the code, not the prose.
#
# So this is a MEASUREMENT recorded as a test, not a verdict. It pins the three
# numbers so that whoever resolves the question (operator + boot seat 2, whose
# accounting this is) has the arithmetic fixed and versioned rather than
# re-derived, and so that any change to one of the three shows up here.
# ===========================================================================


class WhatStagingMeans:
    """Namespace only; the collected tests are the functions below."""


def test_the_three_readings_of_the_staging_term_are_pinned():
    """Three payloads, one label, and 384 MiB reached twice by accident.

    (i)   the CROSS region, which is what section 10.2's "6 directed pairs x
          2 slots" describes:  xr.N_PAIRS x xr.SLOTS_PER_PAIR x xr.SLOT_BYTES.
    (ii)  what the launcher actually CHARGES since #1332 B1b
          (`launcher.py` -> `xchg_bounce_terms_for_arm` ->
          `checkpoint_census.widest_layer_terms(pairs=xr.N_CARDS)`):
          3 x xb.SLOTS_PER_PAIR x xb.SLOT_BYTES_DEFAULT.
    (iii) the ON-CARD host deposit's own worst case, which is what
          `host_ledger.xchg_bounce_bytes()` charged BEFORE B1b and what
          `weight_exchange_transport.py` still documents as charged "at both
          the launch and the run moment":
          (xr.N_RANKS // 2) x tp.ONCARD_SLOTS_MAX x tp.ONCARD_SLOT_BYTES_MAX.

    (i) and (ii) are both 384 MiB at today's constants and are NOT the same
    bytes.  (iii) is 3072 MiB, i.e. 2688 MiB more than the ledger now funds
    for the payload it names.  Which of the three the term is meant to be is an
    accounting question for the launcher's owner, not for this module -- this
    test only refuses to let the three drift silently.
    """
    from sglang.srt.weg2 import weight_exchange_transport as tp_

    cross = xr.N_PAIRS * xr.SLOTS_PER_PAIR * xr.SLOT_BYTES
    charged = xr.N_CARDS * xb.SLOTS_PER_PAIR * xb.SLOT_BYTES_DEFAULT
    oncard_worst = ((xr.N_RANKS // 2) * tp_.ONCARD_SLOTS_MAX
                    * tp_.ONCARD_SLOT_BYTES_MAX)

    assert cross == 384 * xr.MIB, cross
    assert charged == 384 * xr.MIB, charged
    assert oncard_worst == 3072 * xr.MIB, oncard_worst
    # THE COINCIDENCE, asserted so it cannot pass unnoticed: equal totals from
    # different factorisations.  If step 3 raises xr.SLOT_BYTES to 64 this
    # goes red, which is exactly when someone must look.
    assert cross == charged
    assert (xr.N_PAIRS, xr.SLOT_BYTES) != (xr.N_CARDS, xb.SLOT_BYTES_DEFAULT)
    # And the gap the ledger no longer funds for the on-card deposit.
    assert oncard_worst - charged == 2688 * xr.MIB
