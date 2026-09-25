# SPDX-License-Identifier: Apache-2.0
"""#1397 wiring, DESK10's half (weight_updater.py only -- xchg_bounce.py /
weight_exchange_bounce.py are DESK11's; this file drives their PRODUCT
through the real call site, it does not re-test their mechanism).

THE GAP, per DESK11's own DESIGN_option3_band_credit_0914.md section 6.3
and section 8.3: Option 3's whole mechanism (`CrossSlotRendezvous.
post_band_drained`/`wait_band_drained`, `run_bounce_leg`'s `_band_credit_leg`
branch, `BounceTerms.band_credit`/`n_cross_lanes`) is DESK-BEWIESEN and
BYTE-PROVEN in `test_weg2_band_credit_1397.py` -- but every one of those
calls drives `weight_exchange_bounce.run_bounce_leg` DIRECTLY. Nothing
proves `weight_updater.py`'s own production callers (`_weg2_xchg_bounce_leg`,
reached from `_weg2_xchg_deposit_before_sleep` / `_weg2_xchg_inject_weights`)
ever REACH the `band_credit=True` branch at all -- "a lever nobody pulls is
not built" (#1367/#1375 class), user order 2026-09-14 on discovering this
exact gap: "ja dann - bauen!".

VERIFIED BEFORE WRITING A LINE (Prior-Art-Gate, direct source read, no
edit needed to confirm): `weight_updater.py` has EXACTLY two `BounceTerms`
construction sites, both `xb.read_published_terms()` (:1069, :1204) -- never
a local `xb.bounce_terms(...)` call that could drop a field. `_TERM_FIELDS`
(xchg_bounce.py) already carries `band_credit`/`n_cross_lanes` through
`publish_terms`/`read_published_terms`, so a `terms` object obtained through
either of those two real call sites carries whatever the publisher (the
launcher, DESK9's #1395, not this file) stated -- UNCHANGED, unfiltered --
all the way to `bx.run_bounce_leg(..., terms=terms, ...)` inside
`_weg2_xchg_bounce_leg`'s phased/cross-pair branch (weight_updater.py
~:3990-4070), which ALREADY constructs `rv = bx.CrossSlotRendezvous(sems,
slots, pair=pair)` for a genuine cross pair (`pair is not None`) -- exactly
`_band_credit_leg`'s second precondition. Confirmed directly with DESK11
(cross-session, a5686619029fc8141): no code gap in the threading itself:
the un-pulled lever is that NOTHING exercises this chain end to end through
weight_updater.py's own call site, and the launcher (#1395, DESK9, not this
file) does not publish `band_credit=1` yet on any boot.

THIS FILE THEREFORE BUILDS, WITHIN THE weight_updater.py FILE BOUNDARY:

  1. THE POSITIVE EXECUTION SMOKE, #1391's own discipline (two THREADS on
     the REAL leg function, not a reimplementation): a genuine cross pair
     (rank 0 -> rank 1), real POSIX semaphores, real `FakeDeviceOps` host
     pages, `terms.band_credit=True` with `n_bands > cross_lane_slots`
     (forces at least one real wrap -- band-credit's own reuse must fire
     for this to succeed at all), driven through `_weg2_xchg_bounce_leg`
     ON BOTH SIDES concurrently. Proven POSITIVELY, not merely "did not
     raise": the allocated lane buffer's file size on tmpfs is measured
     and shown to equal the SMALL `cross_lane_slots`-based size, and a
     SEPARATE run of the identical descriptors with `band_credit=False`
     is shown to allocate the BIG tag-sized buffer instead -- the same
     comparison DESK11's own suite makes, but through THIS file's wrapper.

  2. THE #1397 REFUSAL'S REACHABILITY FROM THE REAL CALL PATH: an
     inconsistent publication (`band_credit=1`, `n_cross_lanes>0`,
     `max_tag_bytes=0`) reaches `bounce_terms()`'s own `ValueError`
     (xchg_bounce.py) THROUGH `_weg2_xchg_inject_weights()` -- the
     production wake-side entry point -- not through a direct
     `xb.bounce_terms(...)` test call. The chain: `_weg2_xchg_inject_
     weights()` -> `xb.read_published_terms()` -> `bounce_terms(**kw)` ->
     raise, uncaught by anything in between (verified: neither call site
     wraps `read_published_terms()` in a `try`).

  3. THE DANGER-DIRECTION MUTANT (coordinator order): with that same
     refusal PATCHED AWAY (simulating "the guard was never built"), the
     SAME inconsistent term is shown to produce the exact OOM-direction
     mismatch the guard exists to prevent -- the LEDGER's own price
     (`terms.total_bytes`, which takes the SMALL band-credit branch the
     moment `band_credit and n_cross_lanes_priced > 0`, `max_tag_bytes`
     NOT a precondition of THAT branch) prices the cross share small,
     while `run_bounce_leg`'s actual allocator (`leg_slot_bytes`/`lane_
     slots`, whose OWN precondition IS `max_tag_bytes > 0` --
     `_option1_leg`, and `_band_credit_leg` requires `_option1_leg` first)
     falls back to the widest-layer floor -- BIG. Priced small, allocated
     big, is the #1385-round-3 mismatch class by another name; proven here
     with the REAL product functions (`xb.bounce_terms`,
     `wb.leg_slot_bytes`, `wb.leg_geometry`, `BounceTerms.lane_slots`),
     reached through `weight_updater.py`'s own real terms-acquisition call
     site, never reimplemented.

Not this file's job (DESK9/#1395): publishing `band_credit=1` from a real
launcher argv. Not this file's job (DESK11): the mechanism inside
`weight_exchange_bounce.py`/`xchg_bounce.py`, already proven elsewhere.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_weg2_xchg_transport_1273 import FakeDeviceOps  # noqa: E402

from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from sglang.srt.managers.weg2_memory_saver import Weg2WakeRefused  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_bounce as wb  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402
from sglang.srt.weg2 import weight_exchange_transport as tp  # noqa: E402
from sglang.srt.weg2 import xchg_bounce as xb  # noqa: E402

Manager = wu.SchedulerWeightUpdaterManager

TAG = "weights_0"
SRC_RANK, DST_RANK = 0, 1


# ---------------------------------------------------------------------------
# Fakes -- the same shapes test_weg2_undrained_lane_refusal_1391.py /
# test_weg2_1394_1369_wake_source_gap.py drive the product with.
# ---------------------------------------------------------------------------


class _FakeServerArgs:
    def __init__(self):
        self.enable_memory_saver = True
        self.enable_weights_cpu_backup = True
        self.enable_draft_weights_cpu_backup = False
        self.speculative_draft_model_path = None
        self.model_path = "/models/main"


class _FakeRunner:
    def __init__(self):
        self.model = None
        self.model_config = None


class _FakeWorker:
    def __init__(self):
        self.model_runner = _FakeRunner()


def _manager(monkeypatch, *, group, rank, device):
    monkeypatch.setattr(Manager, "_weg2_server_args",
                        lambda self: _FakeServerArgs(), raising=True)
    monkeypatch.setattr(Manager, "_weg2_group_name", lambda self: group,
                        raising=True)
    monkeypatch.setattr(Manager, "_weg2_rank", lambda self: rank, raising=True)
    monkeypatch.setattr(Manager, "_weg2_device_index", lambda self: device,
                        raising=True)
    return Manager(
        tp_worker=_FakeWorker(), draft_worker=None, tp_cpu_group=None,
        memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
        is_fully_idle=lambda *a, **k: True,
    )


def _lane_descs_filter(descs):
    """2026-09-15: the sequential transport derives each lane's unit list
    from the manifests (`_weg2_seq_lane_descs`); this harness has no
    manifests, only explicit descs -- hand them to the leg filtered the way
    the derivation would (by tag, and by lane: the on-card diagonal is
    src == dst == card, a cross lane is one of CROSS_PAIRS)."""
    from sglang.srt.weg2 import weight_exchange_region as _xr

    def _f(self, *, hook, group, rank, pair, card, tag, log=None):
        out = []
        for d in descs:
            if tag is not None and str(getattr(d, "tag", "")) != str(tag):
                continue
            if pair is None:
                if int(d.src_rank) == int(d.dst_rank) == int(card):
                    out.append(d)
            elif (int(d.src_rank), int(d.dst_rank)) == tuple(
                    _xr.CROSS_PAIRS[int(pair)]):
                out.append(d)
        return out
    return _f


def _cross_descs(n_bands, nbytes, src_ops, dst_ops, *, kind_flat_only=True):
    """`n_bands` descriptors on the SAME cross pair (rank 0 -> rank 1),
    real host pages behind each pointer via `FakeDeviceOps`."""
    from sglang.srt.weg2 import weight_exchange as wxm

    out = []
    for i in range(n_bands):
        sp = src_ops.raw_malloc(0, nbytes)
        dp = dst_ops.raw_malloc(0, nbytes)
        ctypes.memmove(src_ops.real(sp), bytes([(i * 17 + 3) & 0xFF] * nbytes),
                       nbytes)
        ctypes.memset(dst_ops.real(dp), 0xEE, nbytes)  # poison, must not survive
        out.append(wxm.XchgDesc(
            tag=TAG, src_rank=SRC_RANK, dst_rank=DST_RANK, param_name=f"p{i}",
            src_ptr=sp, dst_ptr=dp, kind=wxm.FLAT, nbytes=nbytes, rows=1,
            run_bytes=nbytes, spitch=0, dpitch=0, src_off=0, dst_off=0))
    return out


def _band_credit_terms(*, n_bands, slot_bytes, band_credit):
    """Mirrors DESK11's own `_band_credit_terms` helper
    (test_weg2_band_credit_1397.py) but public here since this file drives
    a different call site."""
    return xb.bounce_terms(
        bytes_per_direction=slot_bytes * n_bands, n_layers=1,
        widest_layer_bytes=slot_bytes * n_bands, pairs=1, depth=1,
        slot_bytes=slot_bytes, n_lanes=1,
        max_tag_bytes=slot_bytes * n_bands,  # Option 1 precondition
        band_credit=band_credit,
        n_cross_lanes=(1 if band_credit else 0),
    )


@pytest.fixture()
def real_region(tmp_path):
    """A real semaphore set + a real region dir, boot-scoped."""
    nonce = f"desk10-1397-{os.getpid()}"
    xr.unlink_semaphores(nonce)
    xr.create_semaphores(nonce)
    try:
        yield nonce, tp.SemSet(nonce), str(tmp_path)
    finally:
        xr.unlink_semaphores(nonce)


# ===========================================================================
# 1. THE POSITIVE EXECUTION SMOKE -- two threads, the real leg function,
#    band_credit=True reached from weight_updater.py's own call site.
# ===========================================================================


def _run_cross_leg_pair(monkeypatch, real_region, *, n_bands, slot_bytes,
                        band_credit, tmp_path, lane_suffix):
    """Runs ONE deposit + ONE collect leg, concurrently, through the REAL
    `_weg2_xchg_bounce_leg` -- both sides of a genuine cross pair. Returns
    (ok, error, dst_ops, dst_ptrs) so the caller can check bytes and/or the
    allocated file size."""
    nonce, sems, shm_root = real_region
    terms = _band_credit_terms(n_bands=n_bands, slot_bytes=slot_bytes,
                                band_credit=band_credit)
    src_ops = FakeDeviceOps(str(tmp_path), rank=SRC_RANK)
    dst_ops = FakeDeviceOps(str(tmp_path), rank=DST_RANK)
    descs = _cross_descs(n_bands, slot_bytes, src_ops, dst_ops)
    monkeypatch.setattr(Manager, "_weg2_seq_lane_descs",
                        _lane_descs_filter(descs), raising=True)

    m_src = _manager(monkeypatch, group="P", rank=SRC_RANK, device=SRC_RANK)
    m_dst = _manager(monkeypatch, group="D", rank=DST_RANK, device=DST_RANK)

    errors = {}

    def _deposit():
        try:
            m_src._weg2_xchg_bounce_leg(
                descs=descs, ops=src_ops, boot_nonce=nonce,
                terms=terms, mode=wx.INJECT_AUTHORITATIVE,
                shm_root=shm_root,
                device=SRC_RANK, hook="source", region=None, sems=sems,
                tag=TAG, rank=SRC_RANK)
        except BaseException as exc:  # noqa: BLE001 -- surfaced via `errors`
            errors["deposit"] = exc

    def _collect():
        try:
            m_dst._weg2_xchg_bounce_leg(
                descs=descs, ops=dst_ops, boot_nonce=nonce,
                terms=terms, mode=wx.INJECT_AUTHORITATIVE,
                shm_root=shm_root,
                device=DST_RANK, hook="authoritative", region=None, sems=sems,
                tag=TAG, rank=DST_RANK)
        except BaseException as exc:  # noqa: BLE001
            errors["collect"] = exc

    t_dep = threading.Thread(target=_deposit, name=f"deposit-{lane_suffix}")
    t_col = threading.Thread(target=_collect, name=f"collect-{lane_suffix}")
    t_col.start()
    t_dep.start()
    t_dep.join(timeout=15.0)
    t_col.join(timeout=15.0)
    assert not t_dep.is_alive(), "deposit thread hung -- band credit never freed a slot"
    assert not t_col.is_alive(), "collect thread hung"
    try:
        return errors, dst_ops, descs
    finally:
        src_ops.close()
        # dst_ops closed by the caller after reading bytes/checking the file


def test_band_credit_true_reaches_run_bounce_leg_through_the_real_callsite(
        monkeypatch, real_region, tmp_path):
    """THE LEVER IS PULLED: `n_bands=5` through a term whose
    `cross_lane_slots` is 1 (depth=1, `slot_bytes`-sized Option 1 unit) --
    if `_band_credit_leg` never activated (the wiring gap this file exists
    to close), `_weg2_xchg_bounce_leg`'s cross path would fall back to
    `terms.lane_slots` (5, no wrap needed) OR -- if band_credit's plumbing
    silently dropped somewhere between `read_published_terms` and
    `run_bounce_leg` -- would still not need a real credit wait, so a
    hanging test here is itself the strongest possible signal that the
    reuse mechanism engaged. The byte check below is the SECOND, POSITIVE
    proof: it is not enough that this returns, the bytes AND the buffer
    size must be right too (see the next test for the size half)."""
    n_bands, slot_bytes = 5, 4096
    terms_probe = _band_credit_terms(n_bands=n_bands, slot_bytes=slot_bytes,
                                     band_credit=True)
    assert terms_probe.cross_lane_slots < n_bands, (
        "the test must force at least one real wrap or it proves nothing "
        "about reuse")

    errors, dst_ops, descs = _run_cross_leg_pair(
        monkeypatch, real_region, n_bands=n_bands, slot_bytes=slot_bytes,
        band_credit=True, tmp_path=tmp_path, lane_suffix="bc")
    try:
        assert errors == {}, f"leg(s) raised: {errors}"
        for i, d in enumerate(descs):
            got = ctypes.string_at(dst_ops.real(d.dst_ptr), slot_bytes)
            assert got == bytes([(i * 17 + 3) & 0xFF] * slot_bytes), (
                f"band {i} arrived corrupted or was never written -- a slot "
                f"was reused before its own drain credit, or band_credit "
                f"never actually engaged")
    finally:
        dst_ops.close()


def test_band_credit_true_allocates_the_small_buffer_not_the_tag_sized_one(
        monkeypatch, real_region, tmp_path, caplog):
    """THE SIZE PROOF, measured, on the form that ships since 2026-09-15:
    the SEQUENTIAL transport stages a lane in ONE buffer sized to the
    lane's OWN byte sum (`_lane_bytes`, weg2xsn56), and `band_credit` is a
    lever of the retired band form (`LayerBounce`, `bounce.bin.<lane>`) --
    it reaches the leg unchanged and changes nothing about that size. The
    SAME descriptor population, run with band_credit=True and False
    through the IDENTICAL `_weg2_xchg_bounce_leg` call, must map the SAME
    buffer, and that buffer must be the lane's sum -- never tag-sized
    (`terms.lane_slots x slot_bytes`) and never the priced single slot."""
    import logging
    n_bands, slot_bytes = 5, 4096
    want = n_bands * slot_bytes

    def _mapped_bytes(records, phase):
        out = []
        for r in records:
            msg = r.getMessage()
            if msg.startswith("WEG2-SEQ mapped lane=p0") and f"phase={phase}" in msg:
                out.append(int(msg.split("bytes=")[1].split()[0]))
        return out

    with caplog.at_level(logging.INFO):
        errors_bc, dst_bc, _ = _run_cross_leg_pair(
            monkeypatch, real_region, n_bands=n_bands, slot_bytes=slot_bytes,
            band_credit=True, tmp_path=tmp_path, lane_suffix="small")
    assert errors_bc == {}, f"band_credit=True leg raised: {errors_bc}"
    dst_bc.close()
    small = _mapped_bytes(caplog.records, "deposit")
    caplog.clear()
    nonce2 = f"{real_region[0]}-tagsized"
    xr.create_semaphores(nonce2)
    try:
        region2 = (nonce2, tp.SemSet(nonce2), real_region[2])
        with caplog.at_level(logging.INFO):
            errors_big, dst_big, _ = _run_cross_leg_pair(
                monkeypatch, region2, n_bands=n_bands, slot_bytes=slot_bytes,
                band_credit=False, tmp_path=tmp_path, lane_suffix="big")
        assert errors_big == {}, f"band_credit=False leg raised: {errors_big}"
        dst_big.close()
        big = _mapped_bytes(caplog.records, "deposit")
    finally:
        xr.unlink_semaphores(nonce2)
    assert small == [want] and big == [want], (
        f"the lane's buffer must be the lane's own sum ({want} B) under "
        f"either term: band_credit=True mapped {small}, False mapped {big}")

def _publish_inconsistent_term(monkeypatch, *, max_tag_bytes):
    """Publishes a term whose fields `read_published_terms()` will accept
    syntactically (every `_TERM_FIELDS` key present, all ints) but which
    `bounce_terms()` itself refuses to reconstruct: `band_credit=1` with
    `n_cross_lanes>0` and `max_tag_bytes<=0`."""
    fields = dict(
        bytes_per_direction=4096 * 5, n_layers=1, widest_layer_bytes=4096 * 5,
        pairs=1, depth=1, slot_bytes=4096, n_lanes=1,
        max_tag_bytes=max_tag_bytes, lanes_concurrent=0,
        band_credit=1, n_cross_lanes=1,
        # #1464b (9f625b91c4) added this field to `_TERM_FIELDS`; without it
        # `read_published_terms` refuses the term as partial BEFORE
        # `bounce_terms` runs, so the #1397 refusal below was never reached
        # (and the consistent-term guard passed on the wrong error).
        price_lane_cap=0,
    )
    # The docstring's promise ("every `_TERM_FIELDS` key present") as a check:
    # the next field added to the published term fails HERE, by name, instead
    # of silently turning this fixture into a partial-term test.
    missing = [f for f in xb._TERM_FIELDS if f not in fields]
    assert not missing, (
        f"fixture misses published term field(s) {missing}: "
        "read_published_terms would refuse the partial term first")
    raw = ",".join(f"{k}={v}" for k, v in fields.items()
                   if k in xb._TERM_FIELDS)
    monkeypatch.setenv(xb.ENV_BOUNCE_TERMS, raw)


def test_the_1397_refusal_is_reachable_from_the_real_wake_callsite(monkeypatch):
    """RED-FIRST shape proven the other direction: this is the CHAIN, named
    end to end. `_weg2_xchg_inject_weights()` (weight_updater.py) ->
    `xb.read_published_terms()` -> `bounce_terms(**kw)` -> `ValueError`
    ("band_credit=True with n_cross_lanes stated needs max_tag_bytes > 0"),
    uncaught by anything between the production entry point and the raise --
    verified by NOT wrapping the call below in anything that could swallow
    it, exactly as the two real production call sites do not either."""
    m = _manager(monkeypatch, group="D", rank=0, device=0)
    _publish_inconsistent_term(monkeypatch, max_tag_bytes=0)

    with pytest.raises(ValueError) as exc:
        m._weg2_xchg_inject_weights(tag=TAG)
    msg = str(exc.value)
    assert "band_credit=True" in msg
    assert "max_tag_bytes" in msg


def test_a_consistent_term_passes_the_same_gate_cleanly(monkeypatch):
    """THE REGRESSION GUARD beside the refusal above: the SAME chain, with
    `max_tag_bytes>0` (Option 1 active, the refusal's own precondition
    satisfied), must NOT raise at the `read_published_terms()` step -- it
    must reach `_weg2_xchg_inject_from_peer` and fail there instead (for an
    unrelated, expected reason: this fake manager has no real device/plan),
    proving the #1397 gate itself is narrow and not a blanket refusal on
    every `band_credit=1` publication."""
    m = _manager(monkeypatch, group="D", rank=0, device=0)
    _publish_inconsistent_term(monkeypatch, max_tag_bytes=4096 * 5)

    with pytest.raises(Exception) as exc:
        m._weg2_xchg_inject_weights(tag=TAG)
    # Must NOT be the #1397 ValueError this time.
    assert "band_credit=True with n_cross_lanes" not in str(exc.value)
    # ... and not the partial-term refusal either: that one fires BEFORE the
    # #1397 gate, so a pass on it would prove nothing about the gate.
    assert "is missing" not in str(exc.value), str(exc.value)


# ===========================================================================
# 3. THE DANGER-DIRECTION MUTANT (coordinator order): the guard removed,
#    the ledger prices small while the real allocator falls back to big.
# ===========================================================================


def test_M_without_the_1397_guard_the_ledger_prices_small_while_the_allocator_goes_big(
        monkeypatch):
    """THE MUTANT: patch `xb.bounce_terms` to skip its own #1397 refusal
    (as if it had never been built -- exactly this ticket's own "vorhanden
    aber unbrauchbar" starting point, one guard further back). The SAME
    inconsistent publication used above now constructs successfully through
    `_weg2_xchg_inject_weights()`'s real call chain -- and the resulting
    `terms` object is shown, with the PRODUCT's OWN sizing functions (never
    reimplemented here), to price the cross share SMALL
    (`terms.total_bytes`'s band_credit branch has no `max_tag_bytes`
    precondition) while `run_bounce_leg`'s actual allocator would use the
    BIG widest-layer floor (`leg_slot_bytes`/`lane_slots`, whose PRECONDITION
    -- `_option1_leg`, i.e. `max_tag_bytes>0` -- `_band_credit_leg` requires
    -- is exactly what this inconsistent term violates). Priced small,
    allocated big: the #1385-round-3 mismatch class (measured 5.6x on boot
    weg2xsn31/4), OOM direction, reached through the real callsite. This
    test PASSES only because it demonstrates the DANGEROUS, mutated
    behaviour on purpose -- proving the guard in
    test_the_1397_refusal_is_reachable_from_the_real_wake_callsite is what
    stands between this and a boot that silently underprices its own
    cross-lane host charge."""
    import sglang.srt.weg2.xchg_bounce as xb_mod

    real_bounce_terms = xb_mod.bounce_terms

    def _mutated_bounce_terms(**kw):
        # THE MUTATION: strip the #1397 guard's own precondition by
        # forcing `max_tag_bytes` non-zero ONLY for the refusal check,
        # then constructing the REAL (unmutated) dataclass with the
        # CALLER's original (inconsistent) value -- i.e. exactly what
        # "the guard was never written" would look like: every other
        # line of `bounce_terms` runs unchanged.
        import sglang.srt.weg2.xchg_bounce as m

        n_layers = int(kw.get("n_layers", 1))
        bytes_per_direction = int(kw.get("bytes_per_direction", 1))
        widest_layer_bytes = int(kw.get("widest_layer_bytes", 1))
        mean_layer = -(-bytes_per_direction // n_layers)
        depth = int(kw.get("depth", 1))
        pairs = int(kw.get("pairs", 0))
        slot_bytes = int(kw.get("slot_bytes", 1))
        return m.BounceTerms(
            bytes_per_direction=bytes_per_direction, n_layers=n_layers,
            widest_layer_bytes=widest_layer_bytes, depth=depth, pairs=pairs,
            slot_bytes=slot_bytes, mean_layer_bytes=mean_layer,
            buffer_bytes=m.assemble_buffer_bytes(widest_layer_bytes, depth),
            staging_bytes=pairs * m.SLOTS_PER_PAIR * slot_bytes,
            n_lanes=max(1, int(kw.get("n_lanes", 1))),
            max_tag_bytes=max(0, int(kw.get("max_tag_bytes", 0))),
            lanes_concurrent=max(0, int(kw.get("lanes_concurrent", 0))),
            band_credit=bool(kw.get("band_credit", False)),
            n_cross_lanes=max(0, int(kw.get("n_cross_lanes", 0))),
        )

    monkeypatch.setattr(xb_mod, "bounce_terms", _mutated_bounce_terms,
                        raising=True)
    m = _manager(monkeypatch, group="D", rank=0, device=0)
    _publish_inconsistent_term(monkeypatch, max_tag_bytes=0)

    # With the guard mutated away, the wake callsite's OWN read of the
    # publication now SUCCEEDS where it should have refused -- proven by
    # reaching the NEXT step (`_weg2_xchg_inject_from_peer`, which fails for
    # an unrelated fake-manager reason) instead of the #1397 ValueError.
    with pytest.raises(Exception) as exc:
        m._weg2_xchg_inject_weights(tag=TAG)
    assert "band_credit=True with n_cross_lanes" not in str(exc.value), (
        "the mutant should have let the inconsistent term through -- if "
        "this fires, the mutation did not actually bypass the guard")

    # Recover the terms object the mutated construction produced (same
    # inputs, direct call) to measure the mismatch with the PRODUCT's own
    # functions -- never reimplemented.
    terms = _mutated_bounce_terms(
        bytes_per_direction=4096 * 5, n_layers=1, widest_layer_bytes=4096 * 5,
        pairs=1, depth=1, slot_bytes=4096, n_lanes=1, max_tag_bytes=0,
        band_credit=True, n_cross_lanes=1,
    )
    assert terms.n_cross_lanes_priced > 0, (
        "the term must actually take the band-credit pricing branch or "
        "there is no mismatch to demonstrate")
    priced_cross_bytes = terms.cross_lane_buffer_bytes
    # `_option1_leg` is False (max_tag_bytes<=0), so `_band_credit_leg` in
    # `run_bounce_leg` would be False too (its own precondition) and the
    # REAL allocator falls back to `leg_geometry`'s widest-layer floor --
    # the product's own function, not reimplemented here.
    allocated_slot_bytes = wb.leg_slot_bytes(terms)
    assert allocated_slot_bytes == terms.widest_layer_bytes, (
        "this mutant's whole point is that Option 1 is absent, so "
        "leg_slot_bytes must fall back to the widest-layer floor")
    assert priced_cross_bytes < allocated_slot_bytes, (
        f"priced {priced_cross_bytes} B (band-credit branch) vs. allocated "
        f"{allocated_slot_bytes} B/slot (widest-layer fallback) -- the "
        f"mismatch this mutant must demonstrate did not materialise")


if __name__ == "__main__":
    import unittest

    unittest.main()
