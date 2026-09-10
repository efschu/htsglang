# SPDX-License-Identifier: Apache-2.0
"""#1273 S6-BOUNCE -- EXECUTION SMOKE for BOTH authoritative paths, driven
from the PRODUCT call sites.

Design of record: ``/spinning/gpu-arb/weg2/WEG2_REUSE_SPEC_0908.md`` section 10,
plan ``/spinning/gpu-arb/weg2/PLAN_S6_BOUNCE_0911.md`` steps B2/B3/B4.
User law: ``memory/gewichtsaustausch-ziel-kein-dauer-hostram.md`` -- the dormant
weight image must not live in system RAM between flips, and a layer that is not
byte-identical between P and D is assembled COMPLETELY in a bounded host bounce
buffer from which every card takes its slice.

WHY THIS FILE EXISTS AT ALL, in one measured sentence: the shadow slice was
built, reviewed and booted three times before anyone noticed it had never
reached its transport (#1329, ``ran=no why=no-plan`` on 24/24 legs), because
every test drove the module FUNCTIONS and none drove the mixin METHODS the
product actually calls.  So the rule of #1329's smoke is the rule here:
**every step is taken through the method the scheduler calls.**

RED ON THE BASE (``f03e405e7d``) BY CONSTRUCTION, and that is the point of the
file rather than a property of it: on the base there is no
``weight_exchange_bounce`` module and no ``_weg2_xchg_*`` method on the
manager, so the import at the top fails and the whole file is one collection
error.  That red is an ABSENCE red -- it proves the tests bind to something
that does not yet exist -- and it is only worth what the GREEN run adds: a
byte-exact comparison per destination slice, taken through the product's own
call sites.

THE TWO PATHS, and the reason they are not one:

* **path (a)** -- pieces both ends hold BYTE-IDENTICALLY move card-to-card
  through the existing double-buffered staging lane.  Measured on boot
  weg2xsn8 this set is 4.90 MiB against a 27.52 GiB image, so it is real and
  byte-negligible; the test proves it is AUTHORITATIVE (the destination's LIVE
  storage is what changes), not that it is large.
* **path (b)** -- everything else.  P is PP (whole layers on one stage) and D
  is TP (row-sliced), so ``qkv_proj`` is whole on the source and a third of
  it on each destination and there is no byte-identical piece to move.  The
  source deposits the unit into the bounce buffer; each destination collects
  ITS OWN ROWS.  This carries ~99.98 % of the bytes.

STATED DEVIATION FROM SECTION 10, with its measurement (census
``/spinning/gpu-arb/weg2/tools/layer_unit_census.py`` over
``Qwen3.8-27B-INT8-gdncov-vocabembed``, safetensors headers only):
section 10.2 sizes the slot at 433.9 MiB = 27.12 GiB / 64, which is the MEAN
layer, while section 10.5 bounds the refusal at the WIDEST layer.  Measured,
the widest decoder layer is 721.3 MiB and the widest unit overall is
``lm_head.weight`` at 2425.0 MiB, so those two sentences cannot both hold for a
whole-unit slot.  The streaming unit here is therefore a ROW BAND cut to the
slot -- which is ``tp.batch_descs`` with a layer-sized slot, not a new
mechanism -- so section 10.2's size survives exactly and the hard bound becomes
the widest indivisible RUN, measured at 17408 B.
``test_the_slot_refuses_only_what_no_cut_can_fit`` is that bound's own test.

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

from .test_weg2_xchg_transport_1273 import (  # noqa: E402
    FakeDeviceOps,
    _fresh_boot,
    dev_ptr,
    write,
)

# ---------------------------------------------------------------------------
# THE GEOMETRY, in miniature, and asymmetric ON PURPOSE.
#
# One "layer" of two parameters, at fake device addresses the fake device layer
# can resolve.  ``qkv_proj`` is the section 10.3 case: 24 rows whole on the
# source, 8 rows on each of three destinations.  ``down_proj`` is replicated:
# the same 8 rows on every card, which is what path (a) may carry.
# ---------------------------------------------------------------------------

COLS = 32
ITEMSIZE = 2
ROW = COLS * ITEMSIZE                      # 64 B, the indivisible run
QKV_ROWS_FULL = 24
QKV_ROWS_SHARD = QKV_ROWS_FULL // 3        # 8
DOWN_ROWS = 8

N_LAYERS = 4
N_DST = 3
P_BASE = 0x400000
D_BASE = 0x600000
STEP = 0x4000
QKV_OFF = 0x0
DOWN_OFF = 0x2000

#: The slot is deliberately SMALLER than a unit, so the band cut is exercised
#: rather than assumed: one layer group-wide is 3072 B and the slot is 1 KiB.
SLOT_BYTES = 1024
DEPTH = 2

TAG = "weights.chunk.0"


def _p_ptr(layer: int, off: int) -> int:
    return dev_ptr(0, P_BASE + layer * STEP + off)


def _d_ptr(rank: int, layer: int, off: int) -> int:
    return dev_ptr(rank, D_BASE + layer * STEP + off)


def _qkv_descs(layer: int):
    """One source, three destinations, each taking a DIFFERENT row band.

    The row map is the whole danger of path (b): a destination that copies from
    the wrong source offset gets plausible bytes of the wrong shard, and no
    length check anywhere would notice.  ``src_off`` is therefore
    ``rank * QKV_ROWS_SHARD * ROW`` and the mutant that flips it to 0 is in
    this file (``test_a_slice_from_the_wrong_offset_is_caught``).
    """
    return [
        wx.XchgDesc(
            tag=TAG, src_rank=0, dst_rank=rank,
            param_name=f"model.layers.{layer}.self_attn.qkv_proj.weight",
            kind=wx.STRIDED2D,
            nbytes=QKV_ROWS_SHARD * ROW,
            rows=QKV_ROWS_SHARD, run_bytes=ROW,
            spitch=ROW, dpitch=ROW,
            src_off=rank * QKV_ROWS_SHARD * ROW, dst_off=0,
            src_ptr=_p_ptr(layer, QKV_OFF),
            dst_ptr=_d_ptr(rank, layer, QKV_OFF),
        )
        for rank in range(N_DST)
    ]


def _down_descs(layer: int):
    """Replicated: byte-identical on every card, so path (a) may carry it."""
    return [
        wx.XchgDesc(
            tag=TAG, src_rank=0, dst_rank=rank,
            param_name=f"model.layers.{layer}.mlp.down_proj.weight",
            kind=wx.STRIDED2D,
            nbytes=DOWN_ROWS * ROW,
            rows=DOWN_ROWS, run_bytes=ROW,
            spitch=ROW, dpitch=ROW,
            src_off=0, dst_off=0,
            src_ptr=_p_ptr(layer, DOWN_OFF),
            dst_ptr=_d_ptr(rank, layer, DOWN_OFF),
        )
        for rank in range(N_DST)
    ]


def _all_descs():
    out = []
    for layer in range(N_LAYERS):
        out.extend(_qkv_descs(layer))
        out.extend(_down_descs(layer))
    return out


def _seed_source(ops: FakeDeviceOps) -> None:
    """Fill the source card with a per-(layer,row) pattern, never a constant.

    A constant pattern is the instrument that cannot fail: every wrong offset
    would still compare equal.  The bytes at row ``r`` of layer ``l`` carry
    both indices, so a slice taken from the wrong band differs in EVERY byte.
    """
    for layer in range(N_LAYERS):
        for row in range(QKV_ROWS_FULL):
            write(ops, _p_ptr(layer, QKV_OFF) + row * ROW,
                  bytes([(0x40 + layer) & 0xFF, (0x80 + row) & 0xFF]) * (ROW // 2))
        for row in range(DOWN_ROWS):
            write(ops, _p_ptr(layer, DOWN_OFF) + row * ROW,
                  bytes([(0x10 + layer) & 0xFF, (0x20 + row) & 0xFF]) * (ROW // 2))


def _expected_qkv(layer: int, rank: int) -> bytes:
    out = bytearray()
    for row in range(rank * QKV_ROWS_SHARD, (rank + 1) * QKV_ROWS_SHARD):
        out += bytes([(0x40 + layer) & 0xFF, (0x80 + row) & 0xFF]) * (ROW // 2)
    return bytes(out)


def _expected_down(layer: int) -> bytes:
    out = bytearray()
    for row in range(DOWN_ROWS):
        out += bytes([(0x10 + layer) & 0xFF, (0x20 + row) & 0xFF]) * (ROW // 2)
    return bytes(out)


def _read(ops: FakeDeviceOps, ptr: int, nbytes: int) -> bytes:
    return ctypes.string_at(ops.real(ptr), nbytes)


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


def _run_bounce(mgr, ops, boot_nonce, root, *, descs=None, depth=DEPTH,
                slot_bytes=SLOT_BYTES):
    """One call, through the PRODUCT method, with this file's one geometry."""
    return mgr._weg2_xchg_bounce_leg(
        descs=_all_descs() if descs is None else descs,
        ops=ops, boot_nonce=boot_nonce, slot_bytes=slot_bytes, depth=depth,
        shm_root=root,
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
    # Each layer carries three qkv slices and three down_proj copies.
    assert all(len(u.descs) == 6 for u in units), [len(u.descs) for u in units]


def test_a_unit_carries_the_bytes_of_every_destination():
    units = bx.plan_units(_all_descs())
    want = N_DST * QKV_ROWS_SHARD * ROW + N_DST * DOWN_ROWS * ROW
    assert units[0].nbytes == want


def test_the_widest_unit_is_named_not_averaged():
    """Section 10.5's bound, and it must name the unit, not a mean.

    The census that motivates this: the widest decoder layer measured
    721.3 MiB against a 368.9 MiB mean (1.95x), and ``lm_head`` 2425.0 MiB.
    """
    descs = _all_descs() + _qkv_descs(0) * 2   # layer 0 made artificially wide
    widest = bx.widest_unit(bx.plan_units(descs))
    assert widest.key[1] == "layers.0"
    assert widest.nbytes > bx.plan_units(_all_descs())[1].nbytes


def test_the_slot_refuses_only_what_no_cut_can_fit():
    """The HARD bound is the widest RUN, not the widest unit.

    With the band cut a slot smaller than a unit is correct, so a widest-unit
    refusal would refuse a working configuration.  What no cut can survive is a
    single indivisible run above the slot -- measured 17408 B on the real
    checkpoint -- and that is what refuses.
    """
    descs = _all_descs()
    bx.refuse_if_slot_short(SLOT_BYTES, descs)           # a unit > slot: fine
    with pytest.raises(wx.Weg2XchgPlanDisagree) as e:
        bx.refuse_if_slot_short(ROW - 1, descs)          # a RUN > slot: refused
    assert "run_bytes" in str(e.value)


def test_the_widest_run_is_reported_in_bytes():
    assert bx.widest_run(_all_descs()) == ROW


# ===========================================================================
# STEP 2 -- PATH (b) THROUGH THE PRODUCT CALL SITES, byte-exact.
# ===========================================================================


class PathBAssemblesAndEveryCardTakesItsSlice:
    """Namespace only; the collected tests are the functions below."""


def test_the_deposit_and_the_collect_move_the_right_rows(tmp_path, armed):
    """The whole slice, end to end, through the manager's own method.

    Source deposits every unit; each of three destinations collects ITS OWN
    band; every destination byte is then compared against the source rows it
    must equal.  This is the MATCH digest of the acceptance line.
    """
    ops = FakeDeviceOps(str(tmp_path), 0)
    _seed_source(ops)
    result = _run_bounce(_manager(), ops, armed, str(tmp_path))

    assert result.units == N_LAYERS
    assert result.deposited_bytes == result.collected_bytes
    assert result.verdict == "MATCH"
    for layer in range(N_LAYERS):
        for rank in range(N_DST):
            got = _read(ops, _d_ptr(rank, layer, QKV_OFF), QKV_ROWS_SHARD * ROW)
            assert got == _expected_qkv(layer, rank), (layer, rank)
            got = _read(ops, _d_ptr(rank, layer, DOWN_OFF), DOWN_ROWS * ROW)
            assert got == _expected_down(layer), (layer, rank)


def test_the_buffer_is_reused_and_never_grows_with_the_image(tmp_path, armed):
    """The user law, as an assertion: host residency is the BUFFER.

    Four layers move through a two-slot buffer, so the host bytes held at any
    instant are ``depth * slot_bytes`` and not the image.  A build that
    allocated per unit would pass every byte comparison above and fail exactly
    here, which is why this is a separate test.
    """
    ops = FakeDeviceOps(str(tmp_path), 0)
    _seed_source(ops)
    result = _run_bounce(_manager(), ops, armed, str(tmp_path))
    assert result.host_bytes_peak == DEPTH * SLOT_BYTES
    assert result.host_bytes_peak < result.deposited_bytes


def test_the_line_prints_the_expression_and_not_only_the_total(tmp_path, armed):
    """Section 10.2: "the arming line must PRINT the expression's terms"."""
    ops = FakeDeviceOps(str(tmp_path), 0)
    _seed_source(ops)
    result = _run_bounce(_manager(), ops, armed, str(tmp_path))
    line = result.line()
    assert "WEG2-XCHG-BOUNCE" in line
    for field in ("units=", "widest_unit_bytes=", "slot_bytes=", "depth=",
                  "bounce_total_bytes=", "widest_run_bytes=", "overlap="):
        assert field in line, field


# ===========================================================================
# STEP 3 -- PATH (a) IS AUTHORITATIVE, not an observer.
# ===========================================================================


class PathAIsTheAuthorityForTheBytesItCarries:
    """Namespace only; the collected tests are the functions below."""


def test_path_a_changes_the_destinations_live_storage(tmp_path, armed):
    """The shadow re-targets descriptors into its own buffer; the authority
    does NOT -- it hands the transport the ORIGINAL destination pointers.

    That one difference is path (a)'s whole authority question, so the
    assertion is on the destination's LIVE address, not on a report.
    """
    ops = FakeDeviceOps(str(tmp_path), 0)
    _seed_source(ops)
    result = _manager()._weg2_xchg_agreed_leg(
        descs=_down_descs(0), ops=ops, boot_nonce=armed,
        shm_root=str(tmp_path),
    )
    assert result.verdict == "MATCH"
    for rank in range(N_DST):
        assert _read(ops, _d_ptr(rank, 0, DOWN_OFF),
                     DOWN_ROWS * ROW) == _expected_down(0)


# ===========================================================================
# THE CAN-FAIL CONTROLS.  A digest that cannot go red is not an instrument.
# ===========================================================================


class TheMutantsGoRed:
    """Namespace only; the collected tests are the functions below."""


def test_a_slice_from_the_wrong_offset_is_caught(tmp_path, armed):
    """THE defect class of path (b) (section 2.4 paid for it twice).

    Every destination takes row band 0 instead of its own.  Lengths, totals
    and the deposit all stay correct, so only a byte comparison against the
    RIGHT rows can see it -- and it must.
    """
    ops = FakeDeviceOps(str(tmp_path), 0)
    _seed_source(ops)
    broken = [d.replace(src_off=0) if "qkv" in d.param_name else d
              for d in _all_descs()]
    _run_bounce(_manager(), ops, armed, str(tmp_path), descs=broken)
    wrong = [
        (layer, rank)
        for layer in range(N_LAYERS) for rank in (1, 2)
        if _read(ops, _d_ptr(rank, layer, QKV_OFF), QKV_ROWS_SHARD * ROW)
        != _expected_qkv(layer, rank)
    ]
    assert len(wrong) == N_LAYERS * 2, wrong


def test_a_buffer_that_is_never_released_is_caught(tmp_path, armed):
    """The leg must give the host bytes back, or the law is unmet.

    ``pinned_host_budget`` is the #550 single owner of "may this pinned host
    buffer exist", so a leg that returns without releasing its post charges
    the next admission for bytes nobody holds.  The assertion is on the
    registry, not on a flag the leg sets about itself.
    """
    ops = FakeDeviceOps(str(tmp_path), 0)
    _seed_source(ops)
    before = bx.registered_bounce_bytes()
    _run_bounce(_manager(), ops, armed, str(tmp_path))
    assert bx.registered_bounce_bytes() == before


def test_depth_one_cannot_overlap_and_says_so(tmp_path, armed):
    """Section 10.4's falsifiable budget, as a verdict rather than prose.

    Depth 1 is a correct transfer and a DEGRADED one: nothing can be
    assembled while the previous unit drains, so the x4 link idles between
    units.  The leg must report ``overlap=none`` instead of claiming ok -- a
    build that printed ``overlap=ok`` for depth 1 would satisfy the acceptance
    line while failing the thing it measures.
    """
    ops = FakeDeviceOps(str(tmp_path), 0)
    _seed_source(ops)
    result = _run_bounce(_manager(), ops, armed, str(tmp_path), depth=1)
    assert result.verdict == "MATCH"
    assert "overlap=none" in result.line()


def test_the_refusal_bound_may_not_be_the_mean():
    """Section 10.5, in the direction the census made concrete.

    Layer 0 is made three times the others, so mean < layer 0.  A refusal that
    graded a slot against the MEAN would accept a slot no cut of layer 0 can
    survive.  The bound under test is the widest RUN, and the mean may not
    appear in the decision at all.
    """
    descs = _all_descs() + _qkv_descs(0) * 2
    units = bx.plan_units(descs)
    mean = sum(u.nbytes for u in units) / len(units)
    widest = bx.widest_unit(units).nbytes
    assert widest > mean
    # A slot between the mean and the widest unit is ACCEPTED, because the band
    # cut makes it workable -- this is the assertion that fails if someone
    # reinstates a widest-unit refusal.
    bx.refuse_if_slot_short(int(mean), descs)
