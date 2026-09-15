"""#1378 xsn55 -- THE DIRECTION KEY of the sequential lane derivation.

MEASURED ROOT (boot weg2xsn53 / tip 0ecd75efd3, logs
``boot_weg2_weg2xsn53_0ecd75efd3_0915_015240.{P,D}.log``):

``_weg2_xchg_bounce_leg`` has no ``group`` parameter; inside it ``group`` is
the LANE's desc list (``for pair, group in _lanes.items()``), and it handed
that list to ``_weg2_seq_lane_descs(group=...)``, which feeds it to
``wx.leg_direction(hook, group)``.  ``str(<desc list>)`` is neither ``"P"``
nor ``"D"``, so the direction was answered from the HOOK ALONE -- correct for
group D by coincidence, MIRRORED for group P.  The two P-group walls of that
boot are the same defect:

* P rank 1's wake (hook=authoritative) planned ``pp_to_tp``; its lane (0,1)
  carried P rank 0's 102 layer-32..38 descs -- the manifest-true count, on the
  wrong side -- and P rank 1's own address book correctly answered None for
  every one of them: "102 of 102 descs have no address".  Reproduced at the
  desk against the boot's own manifests: ``plan_from_join(direction="pp_to_tp")``
  lane (0,1) tag=weights_4 -> 102 descs, first
  ``model.layers.32.input_layernorm.weight``; the SAME lane under
  ``tp_to_pp`` -> 6 descs of layer 39, which P rank 1 DOES hold.
* P rank 2's wake printed the direction in its own refusal:
  "NO desc for lane src=0 dst=2 tag='weights_6' (direction=pp_to_tp)" -- a
  P-group importing leg may never plan pp_to_tp.

The three diagonal lane lines of the same boot (P log: c0/c1/c2 descs
114/12/57) match the mirrored plan's diagonal cells for weights_0/weights_4/
weights_6 (114/12/57, measured on the same manifests), which is why the
diagonal resolved while the cross legs refused: on a diagonal,
``dst_card == my_rank`` holds for BOTH readings of the rank number, so the
wrong direction is invisible there.  That is exactly the half a
pointer-profile or ownership audit cannot see, and the reason this test pins
the KEY rather than the counts.

These tests are hermetic: no CUDA, no device, no boot.
"""

import pytest

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_region as xr
from sglang.srt.weg2 import xchg_manifest as xm


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeModel:
    """A model whose only job is to exist for ``_weg2_model_for_group``."""

    is_draft_worker = False
    is_phase_flip_tp_stack = False
    is_dual_group_lane = False


class _FakeRunner:
    model = _FakeModel()
    server_args = None


class _FakeWorker:
    model_runner = _FakeRunner()


class _FakeTensor:
    def __init__(self, ptr):
        self._ptr = int(ptr)

    def data_ptr(self):
        return self._ptr


def _manager(monkeypatch, *, group, rank):
    """The product object, with the identity the leg derivation must read.

    The group name goes through the SAME source the product uses
    (``weg2_group_name``, the ``SGLANG_WEG2_GROUP`` env the launcher
    publishes), cache reset included -- the cache is process-lifetime by
    design, so a test that wants a group must clear it, exactly as
    ``test_weg2_dormant_vram_1y`` does.
    """
    from sglang.srt.managers import weg2_memory_saver as ms

    mgr = SchedulerWeightUpdaterManager(
        tp_worker=_FakeWorker(), draft_worker=None, tp_cpu_group=None,
        memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
        is_fully_idle=lambda *a, **k: True,
    )
    monkeypatch.setenv(ms.WEG2_GROUP_ENV, group)
    monkeypatch.setattr(ms, "_WEG2_GROUP_NAME", None)
    # The manager is a slots dataclass: the identity methods are patched on
    # the CLASS (monkeypatch restores them), the way the 1385 test patches
    # the derivation itself.
    monkeypatch.setattr(SchedulerWeightUpdaterManager, "_weg2_rank",
                        lambda self: int(rank))
    monkeypatch.setattr(
        SchedulerWeightUpdaterManager, "_weg2_rank_param_table",
        lambda self: {("weights", "model.layers.32.input_layernorm.weight"):
                      _FakeTensor(0x1000)})
    return mgr


@pytest.fixture()
def fresh_group_cache():
    """Leave the process-lifetime group cache as it was found."""
    from sglang.srt.managers import weg2_memory_saver as ms

    saved = getattr(ms, "_WEG2_GROUP_NAME", None)
    yield
    ms._WEG2_GROUP_NAME = saved


# ---------------------------------------------------------------------------
# 1. THE CALLER: the leg hands down the group NAME, not the lane's desc list
# ---------------------------------------------------------------------------


def test_the_leg_hands_down_the_group_name_not_the_lane_desc_list(
        monkeypatch, fresh_group_cache):
    """The defect site.  ``_weg2_xchg_bounce_leg``'s ``group`` local is the
    lane's desc list; the derivation must still receive a NAME.

    Drives the REAL leg with a capturing stub of the derivation, exactly the
    way ``test_weg2_xchg_lanes_concurrent_1385`` drives it (same helpers,
    same reason: the plan is not what this test prices).  A mutant that
    reverts the call to ``group=group`` -- the lane list -- is killed by the
    second assertion, because the captured value is then a ``list``.
    """
    import tempfile

    from sglang.srt.weg2 import weight_exchange_bounce as bx
    from sglang.srt.weg2 import weight_exchange_transport as tp
    from sglang.srt.weg2 import xchg_bounce as xb

    from .test_weg2_xchg_bounce_execution_smoke_1273 import (
        DEPTH, LAYER0_BYTES, N_LAYERS, PLAIN_LAYER_BYTES, TAG, _all_descs,
        _manager as _smoke_manager, as_single_hook_descs,
    )
    from .test_weg2_xchg_transport_1273 import FakeDeviceOps, _fresh_boot

    root = tempfile.mkdtemp()
    nonce = _fresh_boot()
    xr.create_semaphores(nonce)
    try:
        descs_all = _all_descs()
        src_descs = [d for d in as_single_hook_descs(descs_all, is_source=True)
                     if d.tag == TAG]
        ops = FakeDeviceOps(root, 0)
        mgr = _smoke_manager()
        # The group name goes through the product's own source (the launcher's
        # env), and that source caches for the process lifetime -- so the env
        # AND the cache are set, exactly as test_weg2_dormant_vram_1y does.
        from sglang.srt.managers import weg2_memory_saver as ms

        monkeypatch.setenv(ms.WEG2_GROUP_ENV, "P")
        monkeypatch.setattr(ms, "_WEG2_GROUP_NAME", None)

        terms = xb.bounce_terms(
            bytes_per_direction=PLAIN_LAYER_BYTES * N_LAYERS,
            n_layers=N_LAYERS, widest_layer_bytes=LAYER0_BYTES, pairs=6,
            depth=DEPTH, slot_bytes=512, max_tag_bytes=LAYER0_BYTES * 2,
            n_lanes=3)

        captured = {}
        real = SchedulerWeightUpdaterManager._weg2_seq_lane_descs

        def _capture(self, **kw):
            captured.update(kw)
            return []

        monkeypatch.setattr(SchedulerWeightUpdaterManager,
                            "_weg2_seq_lane_descs", _capture)
        try:
            mgr._weg2_xchg_bounce_leg(
                descs=src_descs, ops=ops, boot_nonce=nonce,
                terms=terms, mode=wx.INJECT_AUTHORITATIVE, device=0,
                hook="source", sems=tp.SemSet(nonce), tag=TAG, shm_root=root,
            )
        finally:
            SchedulerWeightUpdaterManager._weg2_seq_lane_descs = real

        assert captured, ("the leg never reached the lane derivation, so "
                          "this test asserts nothing")
        assert captured.get("group") == "P", (
            f"the leg handed down {type(captured.get('group')).__name__} "
            f"where the group NAME belongs: {captured.get('group')!r}.  That "
            f"is the weg2xsn53 direction swap: leg_direction() then answers "
            f"from the hook alone and group P's legs plan mirrored")
        assert isinstance(captured.get("group"), str)
    finally:
        xr.unlink_semaphores(nonce)


# ---------------------------------------------------------------------------
# 2. THE CLASS GUARD: a non-name is refused, never planned
# ---------------------------------------------------------------------------


def test_the_derivation_refuses_a_group_that_is_not_a_name(
        monkeypatch, fresh_group_cache):
    """Closes the class: any future caller that hands down a desc list, a
    lane key or a plan dies HERE, by name, instead of planning a mirrored
    direction.  Kills a mutant that deletes the guard."""
    mgr = _manager(monkeypatch, group="P", rank=1)
    lane_list = ["not", "a", "group", "name"]
    with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
        mgr._weg2_seq_lane_descs(hook="authoritative", group=lane_list,
                                 rank=1, pair=0, card=None, tag="weights_4")
    msg = str(exc.value)
    assert "GROUP NAME" in msg
    assert "list" in msg, "the refusal must name what it was handed"
    # and it must refuse BEFORE any manifest read: no manifests are needed to
    # detect a wrong-typed key, so none may be consulted.
    monkeypatch.setattr(xm, "manifests_for_boot",
                        lambda **kw: (_ for _ in ()).throw(
                            AssertionError("manifests read before the "
                                           "group-name guard")))


# ---------------------------------------------------------------------------
# 3. THE DIRECTION: derived from (hook, group NAME), matching the contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hook,group,rank,pair,want", [
    # (hook, group) -> direction, the table `leg_direction` documents.
    # The pair is chosen so THIS rank owns no end of the lane: the derivation
    # then returns [] without needing real manifests, and the captured
    # direction is the only thing under test.
    ("authoritative", "P", 1, 1, "tp_to_pp"),   # P rank 1's wake: D -> P
    ("authoritative", "D", 1, 3, "pp_to_tp"),   # D rank 1's wake: P -> D
    ("source", "P", 1, 0, "pp_to_tp"),          # P rank 1's deposit: P -> D
    ("source", "D", 1, 0, "tp_to_pp"),          # D rank 1's deposit: D -> P
    ("destination", "P", 1, 1, "tp_to_pp"),
    ("destination", "D", 1, 3, "pp_to_tp"),
])
def test_the_lane_derivation_derives_the_direction_from_hook_and_group_name(
        monkeypatch, fresh_group_cache, hook, group, rank, pair, want):
    """The 2x2 (+destination) direction table, pinned at the derivation.

    ``leg_plan_from_join`` documents the same table as "THE DIRECTION IS
    DERIVED, NOT PASSED" -- this test proves the lane derivation answers the
    SAME thing once its ``group`` really is a name, which is what makes the
    lane list the same list the caller's own plan narrowed.
    """
    mgr = _manager(monkeypatch, group=group, rank=rank)
    captured = {}

    def _fake_plan_from_join(join, *, direction="", waves=None,
                            src_addr=None, dst_addr=None):
        captured["direction"] = direction

        class _Plan:
            descs = ()

        return _Plan()

    monkeypatch.setattr(xm, "manifests_for_boot",
                        lambda **kw: ((object(),), ""))
    monkeypatch.setattr(xm, "join_manifests", lambda *a, **kw: object())
    monkeypatch.setattr(xm, "plan_from_join", _fake_plan_from_join)

    out = mgr._weg2_seq_lane_descs(hook=hook, group=group, rank=rank,
                                   pair=pair, card=None, tag=None)
    assert out == []
    assert captured["direction"] == want, (
        f"leg_direction({hook!r}, {group!r}) answered "
        f"{captured['direction']!r}, wanted {want!r} -- a mirrored direction "
        f"here is the weg2xsn53 102-of-102 wall")


# ---------------------------------------------------------------------------
# 4. THE AUDIT RUNS: the lane audit's own names resolve and its cache is a field
# ---------------------------------------------------------------------------


def test_the_lane_audit_executes_and_counts_this_ranks_own_names(
        monkeypatch, fresh_group_cache):
    """EXECUTED, not read: the first version of the audit (8bd23a9a72) held
    three defects no import smoke catches, and every one of them was fatal or
    lying on the first lane of the next boot. Measured by running it:

    * ``region_of_tag`` was a BARE name at the ``owned_n`` comprehension and
      is not imported at module level -> ``NameError`` on the first lane;
    * ``xm`` was a bare name inside ``_weg2_owned_name_keys`` -> NameError,
      swallowed by its own ``except BaseException`` into an EMPTY key set, so
      the instrument would have printed ``owned=0`` for every lane;
    * the cache was written lazily onto a ``slots=True`` dataclass ->
      ``AttributeError: object has no attribute
      '_weg2_owned_name_keys_cache'``, measured.

    This test drives the REAL audit against a real manifest file and asserts
    the count is the rank's OWN piece count, non-zero.
    """
    import dataclasses
    import json
    import tempfile
    from types import SimpleNamespace

    from sglang.srt.managers import weg2_memory_saver as ms

    mgr = _manager(monkeypatch, group="P", rank=0)

    # a slots class cannot carry a lazily written attribute, so the cache
    # must be a DECLARED field -- the same ratchet the three fields above
    # document.
    assert any(f.name == "_weg2_owned_name_keys_cache"
               for f in dataclasses.fields(mgr)), (
        "the audit's cache is not a declared field; on a slots dataclass a "
        "lazy write raises AttributeError, which is the measured xsn54 "
        "defect")

    piece = SimpleNamespace(
        param_name="model.layers.32.input_layernorm.weight", tag="weights_4")
    manifest = SimpleNamespace(group="P", rank=0, pieces=[piece])
    monkeypatch.setattr(xm, "manifests_for_boot",
                        lambda **kw: ((manifest,), ""))
    monkeypatch.setattr(SchedulerWeightUpdaterManager, "_weg2_rank",
                        lambda self: 0)

    keys = mgr._weg2_owned_name_keys()
    assert ("weights", "model.layers.32.input_layernorm.weight") in keys, (
        f"the audit answered {keys!r} for a manifest that carries the name -- "
        f"an empty or absent answer here is the lying-instrument shape, not a "
        f"measurement")
    assert len(keys) == 1
    # the cache is read from the FIELD on the second call (no re-read).
    assert mgr._weg2_owned_name_keys() == keys


def test_the_audit_matches_rank_zero(monkeypatch, fresh_group_cache):
    """Rank 0 is a real rank and the widest holder on every PP boot.

    ``self._weg2_rank() or -1`` turned the real 0 into -1, so rank 0 matched
    no manifest and its audit answered ``owned=0`` on every lane -- the same
    falsy-zero class ``_weg2_rank``'s own docstring warns about ("``-1`` may
    not become 0: rank 0 is a real row another rank owns", inverted here:
    0 may not become -1 either).
    """
    from types import SimpleNamespace

    mgr = _manager(monkeypatch, group="P", rank=0)
    piece = SimpleNamespace(param_name="model.layers.0.mlp.down_proj.weight",
                            tag="weights_0")
    manifest = SimpleNamespace(group="P", rank=0, pieces=[piece])
    monkeypatch.setattr(xm, "manifests_for_boot",
                        lambda **kw: ((manifest,), ""))
    monkeypatch.setattr(SchedulerWeightUpdaterManager, "_weg2_rank",
                        lambda self: 0)
    keys = mgr._weg2_owned_name_keys()
    assert ("weights", "model.layers.0.mlp.down_proj.weight") in keys, (
        f"rank 0's own manifest answered {keys!r} -- the falsy-zero default "
        f"ate the rank, so the audit would have printed owned=0 on every "
        f"lane of the rank that holds the most")

    # and an UNKNOWN identity still matches nothing: -1 is not a rank.
    mgr_unknown = _manager(monkeypatch, group="P", rank=-1)
    monkeypatch.setattr(SchedulerWeightUpdaterManager, "_weg2_rank",
                        lambda self: -1)
    assert mgr_unknown._weg2_owned_name_keys() == set()


def test_the_audit_counts_each_ranks_own_keys_not_the_whole_boot(
        monkeypatch, fresh_group_cache):
    """Counted per (rank, tag), never summed over the boot: rank 1's keys
    must not include rank 0's, which is the "over three ranks summed" shape
    that produced two dead roots earlier in this ticket."""
    from types import SimpleNamespace

    mgr = _manager(monkeypatch, group="P", rank=1)
    mine = SimpleNamespace(param_name="model.layers.39.mlp.down_proj.weight",
                           tag="weights_4")
    theirs = SimpleNamespace(
        param_name="model.layers.32.input_layernorm.weight", tag="weights_4")
    manifests = (SimpleNamespace(group="P", rank=0, pieces=[theirs]),
                 SimpleNamespace(group="P", rank=1, pieces=[mine]),
                 SimpleNamespace(group="D", rank=1, pieces=[mine]))
    monkeypatch.setattr(xm, "manifests_for_boot",
                        lambda **kw: (manifests, ""))
    monkeypatch.setattr(SchedulerWeightUpdaterManager, "_weg2_rank",
                        lambda self: 1)
    keys = mgr._weg2_owned_name_keys()
    assert keys == {("weights", "model.layers.39.mlp.down_proj.weight")}, (
        f"the audit for rank 1 answered {keys!r}: it must carry THIS rank's "
        f"own (region, name) keys only -- not P rank 0's, not group D's")
