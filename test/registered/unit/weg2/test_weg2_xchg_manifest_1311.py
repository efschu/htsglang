"""#1311 S6b -- THE CARD MANIFEST, and the two blockers boot weg2xsn5 measured.

RED-FIRST, AND THE SHAPES ARE THE BOOT'S OWN, not invented ones.  Boot
weg2xsn5 (2026-09-09, ``BOOT_weg2xsn5_0909.md``) drove eight flips of the xchg
shadow on the serve-next5 base and produced NO verdict at all: ``ran=no`` on 48
of 48 legs.  Two independent causes, and this file reproduces both.

**FINDING 1.**  ``W80 Weg2XchgShadowPlanDiverged scope=oncard-peer
field=piece_digest`` on 7 of 8 legs, 21 lines = 7 legs x 3 ranks, with the two
values STABLE per co-located pair across every leg (row0 ``0x369957d09cd846fa``
vs row3 ``0xb34760593777eac7``, and so on).  The same boot's PARAM-CENSUS says
why: group D carries **1249** parameters on every rank (TP, every layer sliced)
while group P carries **935 / 488 / 490** (PP stage subsets -- rank 1 starts at
``model.layers.42``, the 42,11,11 cut).  The plan lines agreed: ``oncard_pieces``
was 1200 on the D ranks and 902/481/479 on the P ranks.  Sets of DIFFERENT
CARDINALITY cannot hash equal, so the gate could not open on any leg.

The models below are that census in miniature: one card, a P end holding a
CONTIGUOUS TAIL of the layers whole, a D end holding EVERY layer sliced.

**FINDING 2.**  ``WEG2-XCHG-SHADOW-ONCARD-REFUSED`` 24 times, every one
``hook=source``, independent of W80.  Its root is the ARM, and the obvious
reading -- "the store-and-forward deposit was never wired" -- is WRONG: the leg
path derives ``store_forward = not inputs.oncard_drainable``, so the product has
always asked for it.  Every one of those refusals came through the other half of
the condition, ``DEPOSIT_REASON_IPC``: an exported VRAM bounce is freed with the
leg that exported it, so the deposit is a ``host``-arm shape by construction and
weg2xsn5 ran ``--weg2-xchg-oncard ipc``.  What was defective was the LINE, which
blamed the hook placement the deposit exists to defeat and named no arm.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402
from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402

from .test_weg2_xchg_shadow_1273 import (  # noqa: E402 -- after the env guard
    _FakeModel,
    _FakeParam,
    _fresh_boot,
)


# ---------------------------------------------------------------------------
# The boot's two layouts, in miniature.
# ---------------------------------------------------------------------------

LAYERS = 32
COLS = 32
#: The P end's stage, as a CONTIGUOUS TAIL, which is what a PP cut produces and
#: what boot weg2xsn5's ``first=model.layers.42.input_layernorm.weight`` on P
#: rank 1 shows.  A random subset would test a different fact.
P_STAGE = range(20, LAYERS)


def _p_stage_model(base_ptr=0x10000000):
    """Group P: WHOLE layers over a SUBSET of layer indices, plus the head."""
    out = []
    for layer in P_STAGE:
        out.append((f"model.layers.{layer}.self_attn.qkv_proj.weight",
                    _FakeParam(64, COLS, ptr=base_ptr + layer * 0x10000)))
        out.append((f"model.layers.{layer}.mlp.down_proj.weight",
                    _FakeParam(32, COLS,
                               ptr=base_ptr + layer * 0x10000 + 0x8000)))
    out.append(("model.embed_tokens.weight",
                _FakeParam(128, COLS, ptr=base_ptr + 0x900000)))
    return _FakeModel(out)


def _d_shard_model(base_ptr=0x20000000, *, shards=3):
    """Group D: SLICES of EVERY layer.

    ``qkv_proj`` is sharded (64 rows -> 64/shards), ``down_proj`` is replicated
    at full extent.  That asymmetry is the point: the pair's intersection must
    keep the pieces both ends hold IDENTICALLY and drop the ones they do not,
    and a model where nothing is sharded could not tell the two apart.
    """
    out = []
    for layer in range(LAYERS):
        out.append((f"model.layers.{layer}.self_attn.qkv_proj.weight",
                    _FakeParam(64 // shards, COLS,
                               ptr=base_ptr + layer * 0x10000)))
        out.append((f"model.layers.{layer}.mlp.down_proj.weight",
                    _FakeParam(32, COLS,
                               ptr=base_ptr + layer * 0x10000 + 0x8000)))
    out.append(("model.embed_tokens.weight",
                _FakeParam(128, COLS, ptr=base_ptr + 0x900000)))
    return _FakeModel(out)


@pytest.fixture()
def chunked(monkeypatch):
    """The ring's own chunk geometry, armed the way the launcher arms it."""
    from sglang.srt.managers import weg2_memory_saver as ms

    monkeypatch.setenv(ms.WEIGHT_CHUNK_ENV_LAYERS, "8")
    monkeypatch.setenv(ms.WEIGHT_CHUNK_ENV_COUNT, "4")
    return ms


@pytest.fixture()
def region(tmp_path):
    boot = _fresh_boot()
    r = xr.XchgRegion.create(boot, shm_root=str(tmp_path))
    r.begin_flip(f"{boot}.1")
    yield r
    r.close()


P_ROW = xr.rank_row("P", 0)
D_ROW = xr.rank_row("D", 0)


def _derive(model, *, group, rank=0, agreed=None, require_agreement=False,
            **kw):
    return sh.derive_leg_plan(
        hook=("source" if group == "P" else "destination"),
        group=group, peer=("D" if group == "P" else "P"), rank=rank,
        model=model, agreed=agreed, require_agreement=require_agreement, **kw)


def _manifest(model, rank=0):
    entries, reason = sh.derive_card_manifest(rank=rank, model=model)
    assert entries is not None, reason
    return entries


def _agree_both(region):
    """Drive the two-flip handshake to completion, as a boot does.

    The first reconcile of each end is the boot's FIRST hook and cannot agree
    (the peer has not published); the second agrees.  Written out rather than
    hidden in a helper flag, because the two-flip cost is a real property of
    this fix and the tests below assert it directly.
    """
    p_entries = _manifest(_p_stage_model())
    d_entries = _manifest(_d_shard_model())
    for _ in range(2):
        p_agreed, p_state = sh.reconcile_card_manifest(
            region, row=P_ROW, peer_row=D_ROW, entries=p_entries)
        d_agreed, d_state = sh.reconcile_card_manifest(
            region, row=D_ROW, peer_row=P_ROW, entries=d_entries)
    return (p_agreed, p_state), (d_agreed, d_state)


# ---------------------------------------------------------------------------
# DENOMINATOR FIRST: the two doubles really are the boot's two layouts.
# ---------------------------------------------------------------------------

def test_the_two_models_reproduce_the_boots_own_param_census(chunked):
    """A scan that matched nothing would pass every assertion below.

    The boot's census is P 935/488/490 against D 1249 -- P holds FEWER
    parameters over a CONTIGUOUS layer tail, D holds MORE over every layer.
    If the doubles did not have that shape, the divergence this file exists to
    reproduce would be an artifact of the fixture.
    """
    p = _manifest(_p_stage_model())
    d = _manifest(_d_shard_model())
    assert len(p) < len(d), (len(p), len(d))
    assert len(d) == 2 * LAYERS + 1, len(d)
    assert len(p) == 2 * len(P_STAGE) + 1, len(p)


# ---------------------------------------------------------------------------
# FINDING 1 -- the divergence, and the convergence.
# ---------------------------------------------------------------------------

def test_the_rank_local_digests_diverge_exactly_as_the_boot_measured(chunked):
    """THE CONTROL, and it must keep passing after the fix.

    Without a peer-agreed manifest each end hashes its OWN plan, and the two
    ends of one card produce different numbers -- which is what boot weg2xsn5
    printed 21 times.  Asserting it here is what makes the convergence test
    below a measurement rather than a tautology: if these two agreed by
    accident of the fixture, the next test would prove nothing.
    """
    p_plan, p_reason = _derive(_p_stage_model(), group="P")
    d_plan, d_reason = _derive(_d_shard_model(), group="D")
    assert p_plan is not None, p_reason
    assert d_plan is not None, d_reason
    assert p_plan.piece_digest != d_plan.piece_digest
    # The boot's own tell: DIFFERENT CARDINALITY, which no hash can reconcile.
    assert p_plan.oncard_pieces != d_plan.oncard_pieces, (
        p_plan.oncard_pieces, d_plan.oncard_pieces)
    assert p_plan.agreed_state == sh.MANIFEST_NOT_ASKED


def test_the_agreed_manifest_makes_both_ends_derive_the_same_digest(
        chunked, region):
    """THE FIX.  Two asymmetric ends, one number, and it is not zero."""
    (p_agreed, p_state), (d_agreed, d_state) = _agree_both(region)
    assert p_state == sh.MANIFEST_AGREED, p_state
    assert d_state == sh.MANIFEST_AGREED, d_state
    p_plan, p_reason = _derive(_p_stage_model(), group="P", agreed=p_agreed,
                               require_agreement=True)
    d_plan, d_reason = _derive(_d_shard_model(), group="D", agreed=d_agreed,
                               require_agreement=True)
    assert p_plan is not None, p_reason
    assert d_plan is not None, d_reason
    assert p_plan.piece_digest == d_plan.piece_digest
    # NOT ZERO, and that check is load-bearing: two ends that both refused
    # would also "agree", and an empty agreement hashes to a fixed number.
    assert p_plan.piece_digest != 0
    assert p_plan.agreed_count > 0
    assert p_plan.agreed_state == d_plan.agreed_state == sh.MANIFEST_AGREED


def test_the_agreement_keeps_only_pieces_both_ends_hold_identically(
        chunked, region):
    """Name equality is NOT enough, and this is why the extents are in the key.

    ``qkv_proj`` exists on both ends under the same name with DIFFERENT row
    counts (P holds the whole layer, D holds a TP shard).  Agreeing on the name
    would put a stage-holder's bytes into a shard-holder's storage.  It must be
    excluded; ``down_proj``, which both hold whole, must survive.
    """
    (p_agreed, _), (d_agreed, _) = _agree_both(region)
    assert p_agreed.keys == d_agreed.keys
    kept_rows = {k[2] for k in p_agreed.keys}
    assert 64 not in kept_rows, "the sharded qkv_proj must not be agreed"
    assert 32 in kept_rows, "the replicated down_proj must be agreed"
    # ONLY THE OVERLAP OF THE LAYER RANGE, so D's layers 0..19 are gone too.
    assert p_agreed.count == len(P_STAGE) + 1, p_agreed.count
    assert p_agreed.mine > p_agreed.count < p_agreed.theirs


def test_both_cardinalities_reach_the_plan_line_not_only_the_survivor(
        chunked, region):
    """How much each side was EXCLUDED is the number the next boot is graded on."""
    (p_agreed, _), _ = _agree_both(region)
    plan, reason = _derive(_p_stage_model(), group="P", agreed=p_agreed,
                           require_agreement=True)
    assert plan is not None, reason
    line = plan.line()
    assert f"manifest={sh.MANIFEST_AGREED}" in line
    assert f"agreed={p_agreed.count}" in line
    assert f"manifest_mine={p_agreed.mine}" in line
    assert f"manifest_theirs={p_agreed.theirs}" in line


def test_the_narrowed_plan_still_transports_and_still_rotates(chunked, region):
    """A converged plan that moved nothing would be a gate that cannot fail."""
    (p_agreed, _), _ = _agree_both(region)
    plan, reason = _derive(_p_stage_model(), group="P", agreed=p_agreed,
                           require_agreement=True)
    assert plan is not None, reason
    assert plan.descs, "the agreed plan must still describe bytes"
    assert plan.oncard_pieces > 0
    assert plan.nbytes > 0
    assert plan.classes, "the rotation must still have classes to rotate over"


# ---------------------------------------------------------------------------
# The gate itself: W80 must go quiet on a converged pair and LOUD on a real one.
# ---------------------------------------------------------------------------

def _gate(region, row, peer_row, *, piece_digest, classes_hash=1,
          expect_rows=None):
    seen = []
    return sh.shadow_gate(
        region, row, leg=1, vote=True, classes_hash=classes_hash, need_mib=1,
        log=seen.append, budget_s=0.05, poll_s=0.001,
        expect_rows=(expect_rows if expect_rows is not None
                     else (P_ROW, D_ROW)),
        plan_digest=7, piece_digest=piece_digest, peer_row=peer_row), seen


def test_the_gate_no_longer_reports_w80_for_a_converged_pair(chunked, region):
    """The destination hook is the only end that reads the peer row, so it is
    the only end that could print W80 -- and it must not, once both ends carry
    the agreed digest."""
    (p_agreed, _), (d_agreed, _) = _agree_both(region)
    assert p_agreed.digest == d_agreed.digest
    _gate(region, P_ROW, D_ROW, piece_digest=p_agreed.digest)
    verdict, seen = _gate(region, D_ROW, P_ROW, piece_digest=d_agreed.digest)
    assert verdict.run, verdict.reason
    assert not [m for m in seen if sh.PLAN_DIVERGED_MARKER in m], seen


def test_a_pair_that_really_diverges_still_raises_w80(chunked, region):
    """THE DANGER DIRECTION.  A fix that made W80 unreachable would be worse
    than the defect: the gate exists to catch two ranks reading different boot
    configurations, and that fault has not gone away."""
    (p_agreed, _), (d_agreed, _) = _agree_both(region)
    _gate(region, P_ROW, D_ROW, piece_digest=p_agreed.digest)
    verdict, seen = _gate(region, D_ROW, P_ROW,
                          piece_digest=d_agreed.digest ^ 0xDEAD)
    assert not verdict.run
    assert verdict.reason == "plan-diverged-oncard-peer"
    assert [m for m in seen if sh.PLAN_DIVERGED_MARKER in m
            and "field=piece_digest" in m], seen


# ---------------------------------------------------------------------------
# The two "not yet" states, which are ordinary boot startup and not a fault.
# ---------------------------------------------------------------------------

def test_the_first_hook_of_a_boot_refuses_by_name_and_never_diverges(
        chunked, region):
    """THE ASYMMETRY THIS FIX MUST NOT REPLACE ONE DIVERGENCE WITH.

    At the boot's first hook the source has published and the destination has
    not.  If the destination then used the intersection while the source
    published a zero, the gate would report W80 for pure startup order.  The
    mutual-readiness flag makes both ends switch on together: both refuse by
    name, both carry ``piece_digest == 0``, and the table is uniform.
    """
    p_entries = _manifest(_p_stage_model())
    d_entries = _manifest(_d_shard_model())
    p_agreed, p_state = sh.reconcile_card_manifest(
        region, row=P_ROW, peer_row=D_ROW, entries=p_entries)
    assert (p_agreed, p_state) == (None, sh.MANIFEST_PEER_ABSENT)
    d_agreed, d_state = sh.reconcile_card_manifest(
        region, row=D_ROW, peer_row=P_ROW, entries=d_entries)
    # The peer's row EXISTS by now, so "absent" would be the wrong word -- but
    # it says peer_seen=0, i.e. it is about to vote a zero for this leg.
    assert (d_agreed, d_state) == (None, sh.MANIFEST_PEER_UNREADY)
    for model, group, agreed in ((_p_stage_model(), "P", p_agreed),
                                 (_d_shard_model(), "D", d_agreed)):
        plan, reason = _derive(model, group=group, agreed=agreed,
                               require_agreement=True)
        assert plan is None
        assert reason.startswith("manifest-unagreed:"), reason


def test_the_handshake_clears_on_the_second_flip_and_stays_clear(
        chunked, region):
    """Bounded and named: exactly one flip of each direction is refused."""
    p_entries = _manifest(_p_stage_model())
    d_entries = _manifest(_d_shard_model())
    states = []
    for _ in range(3):
        states.append(sh.reconcile_card_manifest(
            region, row=P_ROW, peer_row=D_ROW, entries=p_entries)[1])
        states.append(sh.reconcile_card_manifest(
            region, row=D_ROW, peer_row=P_ROW, entries=d_entries)[1])
    assert states == [sh.MANIFEST_PEER_ABSENT, sh.MANIFEST_PEER_UNREADY,
                      sh.MANIFEST_AGREED, sh.MANIFEST_AGREED,
                      sh.MANIFEST_AGREED, sh.MANIFEST_AGREED], states


def test_a_row_from_another_boot_reads_as_absent_never_as_empty(
        chunked, tmp_path):
    """An empty agreement is the dangerous one: it hashes to a fixed number
    both ends would match, which is the instrument-that-cannot-go-red shape."""
    boot_a, boot_b = _fresh_boot(), _fresh_boot()
    ra = xr.XchgRegion.create(boot_a, shm_root=str(tmp_path / "a"))
    try:
        sh.write_card_manifest(ra, D_ROW, _manifest(_d_shard_model()),
                               peer_seen=True)
        entries, seen, state = sh.read_card_manifest(ra, D_ROW)
        assert state == "" and seen and entries
        # Same bytes, a region that names another boot: the mapping is stale.
        ra.boot_hash = xr.epoch_hash(boot_b)
        entries, seen, state = sh.read_card_manifest(ra, D_ROW)
        assert (entries, seen, state) == ((), False, sh.MANIFEST_PEER_ABSENT)
    finally:
        ra.close()


def test_an_unsealed_row_reads_as_absent(chunked, region):
    """A half-written row is not a signal -- the same discipline as every other
    row in this region."""
    import ctypes

    sh.write_card_manifest(region, D_ROW, _manifest(_d_shard_model()),
                           peer_seen=True)
    view = region.dir_view()
    off = sh._manifest_row_off(D_ROW) + sh.MANIFEST_HEADER_STRUCT.size
    ctypes.memset(ctypes.addressof(view) + off, 0xA5, 8)
    assert sh.read_card_manifest(region, D_ROW)[2] == sh.MANIFEST_PEER_ABSENT


# ---------------------------------------------------------------------------
# W83 -- the overflow refuses, and never truncates.
# ---------------------------------------------------------------------------

def test_an_oversized_inventory_refuses_by_name_and_does_not_truncate(
        chunked, region):
    """THE DANGER DIRECTION FOR THE MANIFEST ITSELF.

    Two ends that both truncated at the same cap would intersect two truncated
    sets, agree, and shadow a subset neither of them chose -- with every
    instrument green.  So the cap is a refusal with a W-code, and the row must
    be left untouched.
    """
    too_many = tuple(sh.manifest_entry(f"p{i}", "c", 1, 1, 2)
                     for i in range(sh.MANIFEST_MAX_ENTRIES + 1))
    with pytest.raises(sh.Weg2XchgManifestOverflow) as caught:
        sh.write_card_manifest(region, P_ROW, too_many, peer_seen=False)
    assert sh.MANIFEST_OVERFLOW_MARKER in str(caught.value)
    assert f"entries={len(too_many)}" in str(caught.value)
    assert sh.read_card_manifest(region, P_ROW)[2] == sh.MANIFEST_PEER_ABSENT
    agreed, state = sh.reconcile_card_manifest(
        region, row=P_ROW, peer_row=D_ROW, entries=too_many)
    assert (agreed, state) == (None, sh.MANIFEST_OVERFLOW)
    line = sh.manifest_state_message(
        state=state, rank=0, row=P_ROW, peer_row=D_ROW, leg=1, epoch="e",
        mine=len(too_many))
    assert line.startswith(sh.MANIFEST_OVERFLOW_MARKER)


def test_the_manifest_area_fits_the_dir_area_it_was_carved_from(chunked):
    """Asserted rather than asserted in prose: ``dir_view`` is exactly
    ``DATA_OFF - DIR_OFF`` long, so an arithmetic error here would raise deep
    inside a flip leg."""
    assert sh.MANIFEST_AREA_OFF == sh.SHADOW_AREA_END
    assert sh.MANIFEST_AREA_END <= xr.DATA_OFF - xr.DIR_OFF
    assert (sh.MANIFEST_HEADER_STRUCT.size
            + sh.MANIFEST_MAX_ENTRIES * sh.MANIFEST_ENTRY_STRUCT.size
            + sh.MANIFEST_SEAL_BYTES) <= sh.MANIFEST_ROW_BYTES


# ---------------------------------------------------------------------------
# ONE PRODUCER for the identity, checkable rather than claimed.
# ---------------------------------------------------------------------------

def test_the_published_identity_and_the_planned_identity_have_one_producer(
        chunked):
    """The set the pair agrees over and the set the plan is narrowed by must be
    the same set, or the fix reintroduces the drift it removed."""
    model = _d_shard_model()
    plan, reason = _derive(model, group="D")
    assert plan is not None, reason
    assert plan.manifest == _manifest(model)
    assert len(plan.manifest) == len(set(plan.manifest))


def test_the_identity_is_stable_across_processes_never_the_builtin_hash(
        chunked):
    """PYTHONHASHSEED randomises ``hash()`` per process, and six ranks are six
    processes: a builtin-hashed manifest would intersect to nothing."""
    import inspect

    # THE BODY, NOT THE DOCSTRING -- which names ``hash()`` in order to forbid
    # it, and a source-level grep that could not tell the two apart would fail
    # on the comment that documents the rule.
    import ast as _ast

    body = _ast.unparse(_ast.parse(inspect.getsource(sh.manifest_entry).strip()))
    body = body.split('"""', 2)[-1] if '"""' in body else body
    assert "xr.epoch_hash" in body
    assert "hash(" not in body.replace("epoch_hash(", "")
    a = sh.manifest_entry("model.layers.3.mlp.down_proj.weight", "down_proj",
                          32, 32, 2)
    b = sh.manifest_entry("model.layers.3.mlp.down_proj.weight", "down_proj",
                          32, 32, 2)
    assert a == b
    assert a != sh.manifest_entry("model.layers.3.mlp.down_proj.weight",
                                  "down_proj", 64, 32, 2)


def test_the_digest_is_order_free(chunked, region):
    """Two processes build their sets in different orders and must still
    produce the same number."""
    (p_agreed, _), _ = _agree_both(region)
    keys = list(p_agreed.keys)
    assert sh.agreed_piece_digest(keys) == sh.agreed_piece_digest(
        list(reversed(keys)))


# ---------------------------------------------------------------------------
# THE SIBLING SWEEP: classes_hash is voted too, and it was rank-local.
# ---------------------------------------------------------------------------

def test_the_voted_class_set_now_comes_from_the_agreed_inventory(
        chunked, region):
    """``classes_hash`` is the OTHER field the gate compares, and it was derived
    from this rank's own inventory: on boot weg2xsn5 the six ranks agreed only
    because a PP stage subset and a TP slice of every layer happen to yield the
    same class NAMES.  That is an accident of naming, not a property of the
    derivation.  Deriving it from the agreed inventory makes it structural.
    """
    (p_agreed, _), (d_agreed, _) = _agree_both(region)
    p_plan, _ = _derive(_p_stage_model(), group="P", agreed=p_agreed,
                        require_agreement=True)
    d_plan, _ = _derive(_d_shard_model(), group="D", agreed=d_agreed,
                        require_agreement=True)
    assert p_plan.classes == d_plan.classes
    # The sharded class is gone from BOTH rotations, not only from one.
    assert "qkv_proj" not in p_plan.classes, p_plan.classes


# ---------------------------------------------------------------------------
# FINDING 2 -- the source hook's refusal, and the lane that was never wired.
# ---------------------------------------------------------------------------

def test_the_product_already_asks_for_the_deposit_so_the_arm_is_the_root():
    """THE ROOT OF THE 24 REFUSALS IS THE ARM, not a missing argument.

    It is worth pinning because the obvious reading is wrong and cost one round
    here.  ``shadow_transport`` refuses with ``if not oncard_store_forward or
    reason == DEPOSIT_REASON_IPC``, and ``ShadowLegInputs`` carries no
    ``oncard_store_forward`` field at all -- the leg path DERIVES it from the
    placement flag the adapter does set: ``store_forward = not
    inputs.oncard_drainable``.  So the deposit was always asked for on the
    product path, and every one of boot weg2xsn5's refusals came through the
    second half of that condition: the ``ipc`` arm.
    """
    import inspect

    from sglang.srt.managers.scheduler_components import weight_updater as wu

    src = inspect.getsource(wu)
    head = src.split("ShadowLegInputs(", 1)
    assert len(head) == 2, "the adapter must construct ShadowLegInputs"
    body = head[1].split("\n            )", 1)[0]
    assert "oncard_drainable=False" in body
    # ``ShadowLegInputs`` must NOT grow a store-forward field behind the leg
    # path's derivation -- two producers for one decision is how they drift.
    assert not hasattr(sh.ShadowLegInputs, "oncard_store_forward")
    leg = inspect.getsource(sh.run_leg_hook)
    assert "store_forward = not bool(inputs.oncard_drainable)" in leg


def test_the_ipc_arm_refusal_names_the_arm_and_not_only_the_placement():
    """The old sentence blamed the PLACEMENT and boot weg2xsn5 printed it 24
    times on an ``ipc`` boot.  The placement is true but is no longer the
    operative fact: the deposit exists precisely to run with no concurrent
    peer, and the action that changes this line is ``--weg2-xchg-oncard host``.
    """
    from sglang.srt.weg2 import weight_exchange_transport as tp

    asked = sh.oncard_not_drainable_message(
        rank=2, row=5, peer_row=2, leg=0, epoch="e", is_source=True, descs=48,
        mode=tp.ONCARD_MODE_IPC, asked=True)
    assert asked.startswith(sh.ONCARD_NOT_DRAINABLE_PREFIX)
    assert "deposit_asked=yes" in asked
    assert f"oncard_mode={tp.ONCARD_MODE_IPC}" in asked
    assert "--weg2-xchg-oncard host" in asked
    # The blameless original survives for a caller that never asked.
    unasked = sh.oncard_not_drainable_message(
        rank=2, row=5, peer_row=2, leg=0, epoch="e", is_source=True, descs=48,
        mode=tp.ONCARD_MODE_IPC, asked=False)
    assert "deposit_asked=no" in unasked
    assert "--weg2-xchg-oncard host" not in unasked


def test_the_host_arm_can_now_fund_the_deposit_the_ipc_arm_cannot(chunked):
    """A HERMETIC DOUBLE OF THE HOOK ORDER's consequence.

    ``deposit_refusal_reason`` is the one decision between "this lane runs with
    no concurrent peer" and "it is refused by name", and it is the function the
    24 refusals came through.  The ``ipc`` arm cannot carry a deposit by
    construction (an exported VRAM bounce is freed with its leg); the ``host``
    arm can, once the slots hold every batch and the #1269 ledger charged it.
    """
    from sglang.srt.weg2 import weight_exchange_transport as tp

    assert tp.deposit_refusal_reason(
        batches=2, slots=4, slot_bytes=1 << 20, budget_bytes=1 << 30,
        mode=tp.ONCARD_MODE_IPC) == tp.DEPOSIT_REASON_IPC
    assert tp.deposit_refusal_reason(
        batches=2, slots=4, slot_bytes=1 << 20, budget_bytes=1 << 30,
        mode=tp.ONCARD_MODE_HOST) == ""
    # AND THE TWO WAYS THE HOST ARM STILL REFUSES, both by name.
    assert tp.deposit_refusal_reason(
        batches=8, slots=4, slot_bytes=1 << 20, budget_bytes=1 << 30,
        mode=tp.ONCARD_MODE_HOST) == tp.DEPOSIT_REASON_BATCHES
    assert tp.deposit_refusal_reason(
        batches=2, slots=4, slot_bytes=1 << 20, budget_bytes=0,
        mode=tp.ONCARD_MODE_HOST) == tp.DEPOSIT_REASON_UNFUNDED
