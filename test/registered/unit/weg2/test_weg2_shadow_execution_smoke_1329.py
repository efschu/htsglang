# SPDX-License-Identifier: Apache-2.0
"""#1329 EXECUTION SMOKE -- the shadow path, driven from its PRODUCT call sites.

THE MEASURED GAP THIS FILE CLOSES.  Three boots (weg2xsn5, weg2xsn6, weg2xsn7)
were spent on a shadow that never once reached its transport: ``ran=no
why=no-plan`` on 24/24, 24/24 and 3/3 legs, ``WEG2-XCHG-PLAN`` 0 on P and 0 on
D every time.  The root, named only by #1328's note on the third boot, was a
single write:

    manifest=manifest-failed:AttributeError:
      'SchedulerWeightUpdaterManager' object has no attribute
      '_weg2_shadow_region_cache' @ weight_updater.py:

``SchedulerWeightUpdaterManager`` is ``@dataclass(kw_only=True, slots=True)``,
so the lazy ``self._weg2_shadow_region_cache = region`` raised ON THE WRITE.
Every existing test of this slice was green throughout, because they all drive
the FREE FUNCTIONS (``sh.derive_card_manifest``, ``sh.derive_leg_plan``,
``sh.shadow_transport``, ``sh.run_leg_hook``) directly and the mixin's own
methods -- the only callers the product has -- had no executing test at all.
#1329's pin is an AST sweep, which catches the next undeclared attribute but
would not have caught a manifest that never agrees or a plan that never
carries a piece.

SO THIS FILE IS THE CONSTRUCTION SMOKE, and its rule is: **every step is taken
through the method the scheduler calls**, never through the module function
underneath it.  It drives, on ONE fake card, with no CUDA and no NVML:

    _weg2_shadow_region()   -> a real region, opened from the two env vars
    _weg2_shadow_manifest() -> published, reconciled, MANIFEST_AGREED
    _weg2_shadow_plan()     -> a LegPlan with oncard_pieces > 0
    run_leg_hook(plan=...)  -> the WEG2-XCHG-PLAN line, the ipc lane, and
                               the destination's compare -> verdict=MATCH

RED ON THE BASE, and that is the point of the file rather than a property of
it: on ``376ae2a475`` (the SHA boot weg2xsn7 ran) the first assertion of
``TheRegionOpensThroughTheProductAccessor`` raises the boot's own
``AttributeError``, and every test below it fails the same way -- exactly the
24/24 the metal measured.  On ``672f984b1f`` all of it runs.

TWO DEVIATIONS FROM THE PRODUCT, both stated rather than hidden:

* the two ends run CONCURRENTLY in one process over the ipc lane, as every S4
  and S5 transport test does.  In the product the source hook runs on the
  sleeping group before its pause and the destination hook after the waking
  group's resume, and that store-and-forward shape is a ``host``-arm property
  (``weight_exchange_transport.py`` ``DEPOSIT_REASON_IPC``); it is not what
  this smoke is for.
* the four ranks that are not this card's pair vote through
  ``write_shadow_vote`` instead of running their own leg.  The gate is
  rank-uniform, so without them no leg opens; what they vote is derived from
  THIS run's plan (``facts.digest``, the subset's ``classes_hash``), never a
  literal.

THE CAN-FAIL CONTROL IS IN THE SAME FILE (``a_different_restore_is_MISMATCH``):
a digest that cannot go red is the instrument-that-cannot-fail shape this
campaign has already paid for three times.
"""

from __future__ import annotations

import os
import threading

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402
from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402
from sglang.srt.weg2 import weight_exchange_transport as tp  # noqa: E402

from .test_weg2_xchg_shadow_1273 import (  # noqa: E402 -- after the env guard
    _FakeModel,
    _FakeParam,
    byte_sum,
)
from .test_weg2_xchg_transport_1273 import (  # noqa: E402
    FakeDeviceOps,
    SLOT,
    WAVE,
    _fresh_boot,
    dev_ptr,
    write,
)

# ---------------------------------------------------------------------------
# The two layouts, in miniature, AT FAKE DEVICE ADDRESSES.
#
# The addresses are what makes this file different from
# test_weg2_xchg_manifest_1311.py: there the models only had to be WALKED, so
# any pointer did; here the derived descriptors are handed to the transport,
# so every pointer must be resolvable by the fake device layer.  Both models
# live in the upper half of the 8 MiB fake card, clear of the bump allocator
# the lane's own bounce and shadow buffers come out of.
# ---------------------------------------------------------------------------

LAYERS = 16
COLS = 32
ITEMSIZE = 2
#: A contiguous TAIL, which is what a PP cut produces (boot weg2xsn5's P rank 1
#: began at ``model.layers.42``).
P_STAGE = range(8, LAYERS)
#: TP shards of EVERY layer, the D form.  ``qkv_proj`` is sharded and must be
#: EXCLUDED by the agreement; ``down_proj`` is replicated and must survive.
D_SHARDS = 3

P_BASE = 0x400000
D_BASE = 0x600000
STEP = 0x4000
QKV_OFF = 0x0
DOWN_OFF = 0x2000
EMBED_OFF = 0x50000

QKV_ROWS = 64
DOWN_ROWS = 32
DOWN_BYTES = DOWN_ROWS * COLS * ITEMSIZE


def _qkv_ptr(base: int, layer: int) -> int:
    return dev_ptr(0, base + layer * STEP + QKV_OFF)


def _down_ptr(base: int, layer: int) -> int:
    return dev_ptr(0, base + layer * STEP + DOWN_OFF)


def _p_stage_model() -> _FakeModel:
    out = []
    for layer in P_STAGE:
        out.append((f"model.layers.{layer}.self_attn.qkv_proj.weight",
                    _FakeParam(QKV_ROWS, COLS, itemsize=ITEMSIZE,
                               ptr=_qkv_ptr(P_BASE, layer))))
        out.append((f"model.layers.{layer}.mlp.down_proj.weight",
                    _FakeParam(DOWN_ROWS, COLS, itemsize=ITEMSIZE,
                               ptr=_down_ptr(P_BASE, layer))))
    out.append(("model.embed_tokens.weight",
                _FakeParam(128, COLS, itemsize=ITEMSIZE,
                           ptr=dev_ptr(0, P_BASE + EMBED_OFF))))
    return _FakeModel(out)


def _d_shard_model() -> _FakeModel:
    out = []
    for layer in range(LAYERS):
        out.append((f"model.layers.{layer}.self_attn.qkv_proj.weight",
                    _FakeParam(QKV_ROWS // D_SHARDS, COLS, itemsize=ITEMSIZE,
                               ptr=_qkv_ptr(D_BASE, layer))))
        out.append((f"model.layers.{layer}.mlp.down_proj.weight",
                    _FakeParam(DOWN_ROWS, COLS, itemsize=ITEMSIZE,
                               ptr=_down_ptr(D_BASE, layer))))
    out.append(("model.embed_tokens.weight",
                _FakeParam(128, COLS, itemsize=ITEMSIZE,
                           ptr=dev_ptr(0, D_BASE + EMBED_OFF))))
    return _FakeModel(out)


# ---------------------------------------------------------------------------
# The doubles the mixin needs, and NOTHING more.
# ---------------------------------------------------------------------------


class _FakeRunner:
    """What ``self.tp_worker.model_runner`` has to answer.

    ``wx.RunnerShape.of`` cannot classify it, so ``weights_region_tag_for``
    raises inside the adapter's own try and the region tag falls back to the
    weights default -- the same path a runner of an unclassified shape takes in
    the product, and the reason both adapter methods wrap that call.
    """

    def __init__(self, model) -> None:
        self.model = model


class _FakeWorker:
    def __init__(self, model) -> None:
        self.model_runner = _FakeRunner(model)


def _manager(model):
    """The product object, constructed with the six fields it requires."""
    return wu.SchedulerWeightUpdaterManager(
        tp_worker=_FakeWorker(model), draft_worker=None, tp_cpu_group=None,
        memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
        is_fully_idle=lambda *a, **k: True,
    )


@pytest.fixture()
def chunked(monkeypatch):
    """The ring's chunk geometry, armed the way the launcher arms it."""
    from sglang.srt.managers import weg2_memory_saver as ms

    monkeypatch.setenv(ms.WEIGHT_CHUNK_ENV_LAYERS, "8")
    monkeypatch.setenv(ms.WEIGHT_CHUNK_ENV_COUNT, "4")
    return ms


@pytest.fixture()
def boot():
    return _fresh_boot()


@pytest.fixture()
def region(tmp_path, boot):
    r = xr.XchgRegion.create(boot, shm_root=str(tmp_path))
    r.begin_flip(f"{boot}.1")
    yield r
    r.close()


@pytest.fixture()
def armed(monkeypatch, region, boot):
    """The arm and the two env vars the ADAPTER reads -- no second channel.

    ``_weg2_shadow_region`` opens the region from exactly these two variables,
    which is why the smoke can drive it without a launcher.
    """
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, sh.WEIGHT_SOURCE_SHADOW)
    monkeypatch.setenv(xr.ENV_REGION_PATH, region.path)
    monkeypatch.setenv(xr.ENV_REGION_BOOT, boot)
    assert sh.shadow_armed() is True
    return region


P_ROW = xr.rank_row("P", 0)
D_ROW = xr.rank_row("D", 0)
#: The four rows that are not this card's co-located pair.  The gate is
#: rank-uniform, so a leg with these silent never opens.
OTHER_ROWS = tuple(r for r in range(xr.N_RANKS) if r not in (P_ROW, D_ROW))


# ===========================================================================
# STEP 1 -- the write that raised 24/24 on three boots.
# ===========================================================================


class TheRegionOpensThroughTheProductAccessor:
    """Namespaced only for reading; pytest collects the functions below."""


def test_the_region_accessor_opens_the_real_region_and_caches_the_handle(armed):
    """THE FIRST BYTE OF THE WHOLE PATH, and on 376ae2a475 it raises.

    Not ``getattr``-shaped: the READ was always safe, the WRITE was the defect,
    and a test that only read would have stayed green through all three boots.
    """
    m = _manager(_p_stage_model())
    got = m._weg2_shadow_region()
    assert got is not None, (
        "the adapter could not open the region the env vars name -- on "
        "376ae2a475 this line raises AttributeError instead, which is the "
        "boot's own manifest-failed")
    assert got.boot_hash == armed.boot_hash
    # THE CACHE IS THE ASSERTION, not a side note: the mutant that deletes the
    # write leaves this identity unbound and the region re-opened per leg.
    assert m._weg2_shadow_region_cache is got


def test_the_region_is_opened_once_and_the_second_leg_reuses_it(armed,
                                                               monkeypatch):
    """MUTANT DIRECTION: remove the cache write and this counts two opens.

    A per-leg re-open is not merely wasteful -- the region file is per BOOT and
    merely re-stamped per flip, so a handle re-opened inside a leg is a new
    mapping of the same bytes on every one of a boot's flips.
    """
    opens: list = []
    real_open = xr.XchgRegion.open

    def counting_open(path, *, expect_boot):
        opens.append(str(path))
        return real_open(path, expect_boot=expect_boot)

    monkeypatch.setattr(xr.XchgRegion, "open", staticmethod(counting_open))
    m = _manager(_p_stage_model())
    first = m._weg2_shadow_region()
    second = m._weg2_shadow_region()
    assert first is second
    assert len(opens) == 1, opens


# ===========================================================================
# STEP 2 -- the manifest, through the adapter, to MANIFEST_AGREED.
# ===========================================================================


def _agree_through_the_adapter(p_mgr, d_mgr, *, epoch, legs=2):
    """The two-flip handshake, driven through the PRODUCT method.

    The first hook of each end cannot agree -- the peer has not published yet
    -- and the second does.  That two-leg cost is a real property of the fix
    (``test_weg2_xchg_manifest_1311.py`` pins it at the function level); here it
    is paid through ``_weg2_shadow_manifest``, which is the half that raised.
    """
    out = []
    for leg in range(legs):
        p = p_mgr._weg2_shadow_manifest("P", "D", 0, leg=leg, epoch=epoch)
        d = d_mgr._weg2_shadow_manifest("D", "P", 0, leg=leg, epoch=epoch)
        out.append((p, d))
    return out[-1]


def test_the_manifest_agrees_through_the_adapter_and_excludes_the_shard(
        armed, chunked, boot):
    """Both ends AGREED, and the agreement is the pair's real intersection."""
    p_mgr, d_mgr = _manager(_p_stage_model()), _manager(_d_shard_model())
    (p_agreed, p_state), (d_agreed, d_state) = _agree_through_the_adapter(
        p_mgr, d_mgr, epoch=f"{boot}.1")
    assert p_state == sh.MANIFEST_AGREED, p_state
    assert d_state == sh.MANIFEST_AGREED, d_state
    assert p_agreed is not None and d_agreed is not None
    assert p_agreed.keys == d_agreed.keys
    assert p_agreed.count > 0
    # The sharded qkv_proj must NOT be agreed (64 rows on P, 21 on D); the
    # replicated down_proj must survive.  Without this the "agreement" could be
    # name equality, which is what would put a stage-holder's bytes into a
    # shard-holder's storage.
    kept_rows = {k[2] for k in p_agreed.keys}
    assert QKV_ROWS not in kept_rows, sorted(kept_rows)
    assert DOWN_ROWS in kept_rows, sorted(kept_rows)
    # ONLY the overlap of the layer range: D's layers 0..7 are gone as well.
    assert p_agreed.count == len(P_STAGE) + 1, p_agreed.count


def test_the_manifest_is_derived_once_and_the_second_leg_reuses_it(
        armed, chunked, boot, monkeypatch):
    """MUTANT DIRECTION: remove the manifest cache write -> two derivations.

    The manifest is a BOOT constant; the reconciliation is per leg.  A per-leg
    re-derivation walks every parameter of the model inside a flip leg, which
    is the cost the cache exists to remove.
    """
    derivations: list = []
    real = sh.derive_card_manifest

    def counting(**kw):
        derivations.append(kw.get("rank"))
        return real(**kw)

    monkeypatch.setattr(sh, "derive_card_manifest", counting)
    p_mgr, d_mgr = _manager(_p_stage_model()), _manager(_d_shard_model())
    _agree_through_the_adapter(p_mgr, d_mgr, epoch=f"{boot}.1", legs=3)
    assert p_mgr._weg2_shadow_manifest_cache is not None
    assert d_mgr._weg2_shadow_manifest_cache is not None
    assert len(derivations) == 2, (
        f"one derivation per PROCESS is the contract, three legs were driven: "
        f"{derivations}")


# ===========================================================================
# STEP 3 -- the plan, through the adapter, with pieces.
# ===========================================================================


def _plans(armed_region, boot, *, chunked_ms=None):
    p_mgr, d_mgr = _manager(_p_stage_model()), _manager(_d_shard_model())
    (p_agreed, p_state), (d_agreed, d_state) = _agree_through_the_adapter(
        p_mgr, d_mgr, epoch=f"{boot}.1")
    assert p_state == d_state == sh.MANIFEST_AGREED, (p_state, d_state)
    p_plan, p_reason = p_mgr._weg2_shadow_plan(sh.HOOK_SOURCE, "P", 0,
                                               agreed=p_agreed,
                                               require_agreement=True)
    d_plan, d_reason = d_mgr._weg2_shadow_plan(sh.HOOK_DESTINATION, "D", 0,
                                               agreed=d_agreed,
                                               require_agreement=True)
    assert p_plan is not None, p_reason
    assert d_plan is not None, d_reason
    return p_plan, d_plan


def test_the_plan_comes_out_of_the_adapter_with_pieces_and_one_digest(
        armed, chunked, boot):
    """``WEG2-XCHG-PLAN`` read 0 on P and 0 on D for three boots.

    Here both ends produce one, from the adapter, with a non-zero piece count
    and the SAME piece digest -- which is the field the gate votes (S6 fix F2)
    and the one that was rank-local when boot weg2xsn5 refused 7 of 8 legs.
    """
    p_plan, d_plan = _plans(armed, boot)
    assert p_plan.oncard_pieces > 0, p_plan.line()
    assert d_plan.oncard_pieces > 0, d_plan.line()
    assert p_plan.oncard_pieces == d_plan.oncard_pieces
    assert p_plan.piece_digest == d_plan.piece_digest
    assert p_plan.piece_digest != 0, "two ends that both refused also 'agree'"
    assert p_plan.agreed_state == d_plan.agreed_state == sh.MANIFEST_AGREED
    # The provenance line the product logs once per leg, before anything opens.
    line = p_plan.line()
    assert line.startswith(sh.PLAN_LINE_PREFIX), line
    assert f"oncard_pieces={p_plan.oncard_pieces}" in line, line
    # The facts digest is what the six rows vote as plan_digest; it is derived
    # from the launcher's environment, so the two ends must read it equal.
    assert p_plan.facts.digest == d_plan.facts.digest


# ===========================================================================
# STEP 4 -- THE SMOKE: the leg runs, the lane moves bytes, the digest MATCHES.
# ===========================================================================


def _inputs(hook, *, row, peer_row, epoch, gate_rows):
    return sh.ShadowLegInputs(
        leg=0, epoch=epoch, direction="P->D", hook=hook, rank=0, row=row,
        peer_row=peer_row, device=0, card_uuid="u0",
        uuid_of_card=("u0", "u1", "u2"), wave=WAVE, free_mib=8192,
        gate_rows=tuple(gate_rows), oncard_drainable=True,
    )


def _write_agreed_pieces(p_ops, d_ops, *, same: bool):
    """The bytes under the agreed pieces, on BOTH ends of the card.

    ``same=True`` is the ring having restored what the exchange would move;
    ``same=False`` is the can-fail control.  Written per layer with a distinct
    pattern so a compare that read one piece and reported all of them would be
    visible.

    MEASURED TRAP, hit while writing this file, and it is the reason the
    difference is ASSERTED here rather than assumed: the first version of the
    control wrote ``(b + 1) & 0xFF`` over every byte and the compare still said
    ``verdict=MATCH``.  That is not a defect in the compare -- the stripe
    checksum is an EXACT BYTE SUM (``uint8_checksum``'s contract), the pattern
    ``(0x40 + layer*7 + i*13) & 0xFF`` hits each of the 256 byte values exactly
    eight times per 2048-byte piece because 13 is coprime with 256, and a
    wrapping +1 over a uniform multiset is a PERMUTATION of it, so the sum is
    invariant.  A control whose "different" bytes have the same checksum proves
    nothing, and it reads exactly like a passing control.  So the destination
    differs in ONE byte per piece and the two sums are compared in Python
    before the leg is allowed to run.
    """
    for layer in P_STAGE:
        payload = bytes(((0x40 + layer * 7 + i * 13) & 0xFF)
                        for i in range(DOWN_BYTES))
        write(p_ops, _down_ptr(P_BASE, layer), payload)
        if same:
            write(d_ops, _down_ptr(D_BASE, layer), payload)
            continue
        head = 0x00 if payload[0] != 0x00 else 0x01
        restored = bytes([head]) + payload[1:]
        assert sum(restored) != sum(payload), (
            "the control's two images have the same byte sum, so the compare "
            "cannot tell them apart -- see this function's docstring")
        write(d_ops, _down_ptr(D_BASE, layer), restored)


def _run_the_leg(armed_region, boot, tmp_path, *, same_bytes: bool):
    """Both hooks of one leg, through ``run_leg_hook``, on one fake card."""
    p_plan, d_plan = _plans(armed_region, boot)
    epoch = f"{boot}.1"
    # THE SUBSET AND THE DIGESTS THE FOUR OTHER ROWS MUST VOTE, derived from
    # this run's plan exactly as ``run_leg_hook`` derives them.
    subset = sh.select_subset(p_plan.descs, leg=0,
                              rotation=tuple(p_plan.classes))
    assert subset.descs, ("the rotation picked a class with no descriptors, so "
                          "the leg would refuse before the lane")
    for row in OTHER_ROWS:
        sh.write_shadow_vote(armed_region, row, leg=0, vote=True,
                             classes_hash=subset.hash, need_mib=0,
                             plan_digest=p_plan.facts.digest,
                             piece_digest=p_plan.piece_digest)

    xr.create_semaphores(boot)
    sems = tp.SemSet(boot)
    p_ops = FakeDeviceOps(str(tmp_path / "card0"), rank=0)
    d_ops = FakeDeviceOps(str(tmp_path / "card0"), rank=0)
    p_lines: list = []
    d_lines: list = []
    try:
        # A HARNESS PROPERTY: the two ops share one card's storage (that is
        # what makes the on-card lane real here) but each has its own bump
        # allocator from 0, so the source's bounce and the destination's shadow
        # buffer would alias.  On metal two processes' cudaMalloc cannot.
        d_ops.raw_malloc(0, 2 << 20)
        _write_agreed_pieces(p_ops, d_ops, same=same_bytes)

        def source():
            sh.run_leg_hook(
                _inputs(sh.HOOK_SOURCE, row=P_ROW, peer_row=D_ROW,
                        epoch=epoch, gate_rows=(P_ROW,) + OTHER_ROWS[:2]),
                log=p_lines.append, plan=p_plan, region=armed_region,
                sems=sems, ops=p_ops, slot_bytes=SLOT, oncard_slot_bytes=SLOT,
                stripe_bytes=1 << 20, budget_s=20.0, gate_budget_s=20.0,
                hook_budget_s=60.0)

        thread = threading.Thread(target=source, name="shadow-source")
        thread.start()
        result = sh.run_leg_hook(
            _inputs(sh.HOOK_DESTINATION, row=D_ROW, peer_row=P_ROW,
                    epoch=epoch, gate_rows=tuple(range(xr.N_RANKS))),
            log=d_lines.append, plan=d_plan, region=armed_region, sems=sems,
            ops=d_ops, sum_bytes=byte_sum(d_ops), slot_bytes=SLOT,
            oncard_slot_bytes=SLOT, stripe_bytes=1 << 20, budget_s=20.0,
            gate_budget_s=20.0, hook_budget_s=60.0)
        thread.join(120)
        assert not thread.is_alive(), "the source hook never returned"
        return result, p_lines, d_lines, p_plan
    finally:
        p_ops.close()
        d_ops.close()
        sems.close()
        xr.unlink_semaphores(boot)


def test_the_shadow_leg_runs_to_the_first_byte_and_the_digest_matches(
        armed, chunked, boot, tmp_path):
    """THE SMOKE.  What three boots never reached, reached at the desk.

    Graded on the same four things boot order XSN8 grades criterion (a) on:
    no ``W79 ... reason=no-plan``, a ``WEG2-XCHG-PLAN`` line with pieces on the
    leg, ``ran`` true with ``pieces>0``, and a compare that produced a verdict.
    """
    result, p_lines, d_lines, plan = _run_the_leg(
        armed, boot, tmp_path, same_bytes=True)

    lines = p_lines + d_lines
    # (1) the refusal that stood on every leg of every boot is ABSENT.
    assert not [ln for ln in lines if "reason=no-plan" in ln], [
        ln for ln in lines if "reason=no-plan" in ln]
    # (2) the plan line the boot record reads FIRST, on both ends.
    assert [ln for ln in p_lines if ln.startswith(sh.PLAN_LINE_PREFIX)], p_lines
    assert [ln for ln in d_lines if ln.startswith(sh.PLAN_LINE_PREFIX)], d_lines
    # (3) the leg RAN, over pieces, and the transport moved them.
    assert result is not None
    assert result.ran, (result.reason, result.counters.errors)
    assert result.pieces > 0, result.line()
    # (4) the compare happened and MATCHED -- the digest, not merely a copy.
    assert result.counters.stripes >= 1, result.line()
    assert result.counters.match >= 1, result.line()
    assert result.counters.mismatch == 0, result.line()
    assert result.counters.verdict == sh.MATCH, result.line()
    line = result.line()
    for token in ("WEG2-XCHG-SHADOW ", "ran=yes", "verdict=MATCH",
                  "xchg_ms=", "issue_ms=", "slot_wait_ms=", "gate_skew_ms="):
        assert token in line, (token, line)
    # The bytes really travelled: the shadow buffer holds the SOURCE's image.
    assert result.counters.errors == [], result.counters.errors


def test_a_different_restore_is_MISMATCH_so_the_digest_can_go_red(
        armed, chunked, boot, tmp_path):
    """THE CAN-FAIL CONTROL for the test above.

    Same leg, same lane, one byte per piece different on the destination.  A
    green MATCH here would mean the compare reads nothing -- the shape that
    turned ``mismatch=0`` into a vacuous pass on xsn6 (``ran=no`` on 24 legs,
    ``mismatch=0`` recorded as VACUOUS in the record for exactly this reason).
    """
    result, _p, d_lines, _plan = _run_the_leg(
        armed, boot, tmp_path, same_bytes=False)
    assert result is not None and result.ran, (result.reason if result else None)
    assert result.counters.mismatch >= 1, result.line()
    assert result.counters.verdict == sh.MISMATCH, result.line()
    assert any(sh.MISMATCH_MARKER in ln for ln in d_lines), d_lines[-3:]
    # ZERO AUTHORITY: a mismatch is counted and logged, never raised.
    assert result.counters.errors == [], result.counters.errors
