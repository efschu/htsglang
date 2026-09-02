"""#847/W33: the WRITER for `scheduler.phase_flip_host_pools`.

MECHANISMUS VORHANDEN, AKTUATOR FEHLT -- the whole #718 rebind chain already
existed and was already wired: `rebind_for_cutover` is called at the cutover,
the #719 generation stamp and `coherence_check` are built, and
`phase_pools_for` knows exactly what it wants. It wanted
`scheduler.phase_flip_host_pools[phase]`, and across the entire tree that name
appeared ONLY in its own docstring and its own refusal message. Nothing ever
wrote it.

W32 measured the consequence end to end (SPECIMEN_w32_policy_purity_copy_
pulls_back_to_pp.log): no host pool -> RebindRefused -> the rebind never arms
-> `bound_phase()` stays "pp" -> `device_tier_disarmed("load")` is True for the
whole TP phase -> `HiCacheController.load()` returns None -> zero tokens reach
the device. The one transport prefill logged `#cached-token: 0` on what should
have been a perfect disk hit, and the specimen carries 6 `#718
hicache-phase-guard` warnings beside `phase_flip_rebind_hicache=False`.

REFUSAL CONVERSION, NOT GUARD DELETION (#847). The guard still refuses for a
genuinely absent or mis-shaped pool -- that is the can-fail direction below.
What changes is that the pool can now EXIST and is PRICED: a small staging pin
(#810) whose bytes are a named HOST-LEDGER post (#721), where the post shrinks
and the floor never does.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import types
import unittest

from sglang.srt.managers.phase_flip_boot import build_phase_flip_host_pools
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.test.test_utils import CustomTestCase


#: The Boot-2 pp pool (WEG1_BUILD_SPEC_0901 section 5): 366211 rows of
#: 16384 B on a stage holding 8 attention layers -> 2048 B per layer.
PP_ROWS = 366211
PP_CELL = 16384
PP_LAYERS = 8
PER_LAYER = PP_CELL // PP_LAYERS
TP_LAYERS = 16
TP_CELL = PER_LAYER * TP_LAYERS


class _PPKVHost:
    """The pp phase's anchor KV host pool (HostKVCache shape: size, cell,
    device_pool). This is the pool the tp pin is row-coupled TO."""

    def __init__(self, size=PP_ROWS, size_per_token=PP_CELL, layer_num=PP_LAYERS):
        self.size = size
        self.size_per_token = size_per_token
        self.device_pool = types.SimpleNamespace(layer_num=layer_num)
        self.layout = "layer_first"
        self.page_size = 1
        self.device = "cpu"


class _PPHost:
    """The live tier is a HostPoolGroup COMPOSITE: anchor_entry + entry_map."""

    def __init__(self, kv=None, mamba_host=None):
        kv = kv if kv is not None else _PPKVHost()
        self.anchor_entry = types.SimpleNamespace(
            name=PoolName.KV, host_pool=kv, device_pool=kv.device_pool
        )
        self.entry_map = {PoolName.KV: self.anchor_entry}
        if mamba_host is not None:
            self.entry_map[PoolName.MAMBA] = types.SimpleNamespace(
                name=PoolName.MAMBA, host_pool=mamba_host
            )
        self.size = kv.size


class _BuiltKVHost:
    """What the (faked) assembler returns for the tp pin."""

    def __init__(self, kv_pool, ratio, size_gb, cell):
        self.device_pool = kv_pool
        self.ratio = ratio
        self.size_gb = size_gb
        self.size_per_token = cell
        self.layer_num = kv_pool.layer_num
        # EXACTLY pool_host/base.py:140-147 for host_size > 0: floor to whole
        # cells, then one page of alignment. This is the faithful double: the
        # coupling formula must round-trip through THIS arithmetic.
        self.size = int(size_gb * 1e9 // cell) + 1


class _FakeBuildKVHostPool:
    """Stands in for hybrid_pool_assembler.build_kv_host_pool and records."""

    def __init__(self, per_layer=PER_LAYER, force_size=None):
        self.per_layer = per_layer
        self.force_size = force_size
        self.last = None
        self.calls = []

    def __call__(self, *, kv_pool, page_size, server_args, use_mla):
        cell = self.per_layer * int(kv_pool.layer_num)
        built = _BuiltKVHost(kv_pool, server_args.hicache_ratio, server_args.hicache_size, cell)
        if self.force_size is not None:
            built.size = self.force_size
        self.last = built
        self.calls.append(dict(kv_pool=kv_pool, server_args=server_args, use_mla=use_mla))
        return built


class _DevicePool:
    layer_num = TP_LAYERS

    def __init__(self, cell=8192):
        self._cell = cell

    def get_kv_size_per_token(self):
        return self._cell


def _sched(*, rebind=True, host=True, tp_pool=True, cell=8192, pp_host=None,
           max_running_requests=8, chunked_prefill_size=4096, mamba_mib=0):
    sa = types.SimpleNamespace(
        phase_flip_rebind_hicache=rebind,
        chunked_prefill_size=chunked_prefill_size,
        max_running_requests=max_running_requests,
        page_size=1,
        hicache_mem_layout="layer_first",
        hicache_storage_backend="file",
        hicache_host_role="staging",
        hicache_size=6,
        hicache_mamba_host_mib=mamba_mib,
    )
    if pp_host is None:
        pp_host = _PPHost()
    tree = types.SimpleNamespace(token_to_kv_pool_host=pp_host if host else None)
    stacks = types.SimpleNamespace(
        tp_worker=types.SimpleNamespace(
            model_runner=types.SimpleNamespace(
                token_to_kv_pool=_DevicePool(cell) if tp_pool else None
            )
        )
    )
    return types.SimpleNamespace(
        server_args=sa, tree_cache=tree, phase_flip_stacks=stacks
    )


def _build_patched(sched, builder=None):
    """Assembly needs real pools; patch the NAMED primitives instead.

    Patching the builders the writer is required to use is itself the
    assertion that it uses them -- a writer that went back to cloning
    `type(pp_host)` would ignore these patches and fail here.
    """
    import unittest.mock as mock

    builder = builder if builder is not None else _FakeBuildKVHostPool()
    with (
        mock.patch(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler.build_kv_host_pool",
            new=builder,
        ),
        mock.patch(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler.build_pool_entry"
        ) as entry,
        mock.patch("sglang.srt.mem_cache.memory_pool_host.HostPoolGroup") as grp,
    ):
        entry.side_effect = lambda **kw: types.SimpleNamespace(**kw)
        grp.side_effect = lambda entries: types.SimpleNamespace(
            entries=entries,
            entry_map={e.name: e for e in entries},
            device_pool=sched.phase_flip_stacks.tp_worker.model_runner.token_to_kv_pool,
        )
        pools = build_phase_flip_host_pools(sched)
    return pools, builder


#: The Boot-2 anchor pool (WEG1_BUILD_SPEC_0901 section 5): 2400 MiB at
#: 37.41 MiB per slot on PP0 buys 64 slots; the group MIN plus the one page
#: of pool_host/base.py:146-147 is 65, and the tp anchor half must land on
#: exactly that count. 20 device slots is the boot form's mamba pool.
ANCHOR_MIB = 2400
ANCHOR_SLOTS = 65
ANCHOR_PER_SLOT = int(37.41 * 2**20)
DEVICE_MAMBA_SLOTS = 20


class _FakeMambaPoolHost:
    """Stands in for memory_pool_host.MambaPoolHost and RECORDS its call.

    The recorded kwargs are the pin: the tp anchor half must be built from
    the SAME knob the pp anchor pool takes (--hicache-mamba-host-mib), with
    ratio 1.0 and host_size 0 (the fallback form below the knob). What the
    fake returns is the shape the post-build check reads: size (slots) and
    size_per_token (bytes per slot).
    """

    def __init__(self, size=ANCHOR_SLOTS, per_slot=ANCHOR_PER_SLOT):
        self.size = size
        self.per_slot = per_slot
        self.calls = []

    def __call__(self, device_pool, host_to_device_ratio, host_size, **kw):
        self.calls.append(
            dict(device_pool=device_pool, ratio=host_to_device_ratio, host_size=host_size, **kw)
        )
        return types.SimpleNamespace(size=self.size, size_per_token=self.per_slot)


def _hybrid_sched(
    *,
    device_slots=DEVICE_MAMBA_SLOTS,
    pp_anchor_slots=ANCHOR_SLOTS,
    mamba_mib=ANCHOR_MIB,
    handles=True,
):
    """A pp tier describing KV AND MAMBA, and a TP stack exposing the mamba
    handles `_hybrid_pin_entries` mirrors: req_to_token_pool.mamba_pool /
    mamba_map / mamba_allocator, and the device pool's own
    full_attention_layer_id_mapping. `handles=False` is the #871 path where
    the TP side cannot supply them."""
    pp_host = _PPHost(mamba_host=types.SimpleNamespace(size=pp_anchor_slots))
    s = _sched(pp_host=pp_host, mamba_mib=mamba_mib)
    runner = s.phase_flip_stacks.tp_worker.model_runner
    runner.token_to_kv_pool.full_attention_layer_id_mapping = {
        i: i for i in range(TP_LAYERS)
    }
    if handles:
        runner.req_to_token_pool = types.SimpleNamespace(
            mamba_pool=types.SimpleNamespace(size=device_slots),
            mamba_map={TP_LAYERS + i: i for i in range(48)},
            mamba_allocator=types.SimpleNamespace(
                alloc=lambda *a, **k: None, free=lambda *a, **k: None
            ),
        )
    return s


def _build_hybrid(sched, mamba=None):
    """`_build_patched` with the anchor-pool constructor replaced by a
    recording fake; returns (pools, kv_builder, mamba_fake)."""
    import unittest.mock as mock

    mamba = mamba if mamba is not None else _FakeMambaPoolHost()
    with mock.patch("sglang.srt.mem_cache.memory_pool_host.MambaPoolHost", new=mamba):
        pools, builder = _build_patched(sched)
    return pools, builder, mamba


class TestTheDefaultBootIsUntouched(CustomTestCase):
    def test_the_flag_off_allocates_nothing(self):
        # Every boot that does not ask for the rebind must be byte-identical.
        self.assertEqual(build_phase_flip_host_pools(_sched(rebind=False)), {})


class TestTheWriterBuildsBothPhases(CustomTestCase):
    def _patched(self, sched):
        pools, _ = _build_patched(sched)
        return pools

    def test_both_phases_are_registered(self):
        pools = self._patched(_sched())
        self.assertEqual(sorted(pools), ["pp", "tp"])

    def test_the_pp_entry_is_the_tier_the_boot_already_built(self):
        # The rebind needs a HANDLE per phase, not a second pp pool.
        s = _sched()
        pools = build_phase_flip_host_pools(s)
        self.assertIs(pools["pp"], s.tree_cache.token_to_kv_pool_host)

    def test_the_tp_pin_is_allocated_from_the_tp_device_pool(self):
        # DESIGN_706 C1: a host pool is allocated FROM its device pool, which
        # is why this cannot be derived after the fact and must run at boot.
        s = _sched()
        pools = self._patched(s)
        self.assertIs(
            pools["tp"].device_pool,
            s.phase_flip_stacks.tp_worker.model_runner.token_to_kv_pool,
        )


class TestItIsAStagingPinNotAMirror(CustomTestCase):
    """#810: a `retention` tier is sized as a RATIO of the device pool. The tp
    pin is never ratio-sized: since #1068 its row count is COUPLED to the pp
    pool that --hicache-size built, so both phase pools hold the same rows.
    """

    def test_the_ratio_is_not_used(self):
        _, builder = _build_patched(_sched())
        sa_pin = builder.calls[-1]["server_args"]
        self.assertEqual(sa_pin.hicache_ratio, 0, "a RATIO is the mirror-shaped answer")
        self.assertGreater(sa_pin.hicache_size, 0, "an explicit size instead")


class TestTheTpPinIsRowCoupledToThePpPool(CustomTestCase):
    """#1068 WEG 1 slice 1 (WEG1_BUILD_SPEC_0901 section 4.1, G13).

    Both phase pools are built from ONE absolute knob, --hicache-size: the pp
    pool directly, the tp pin by ROW COUNT. The pin's GB figure is derived so
    that pool_host/base.py:140-147 (floor to whole cells, plus one page) lands
    on EXACTLY the pp row count -- never one row more or less, because a
    phase pin whose rows differ from the synced pp pool would let one rank
    refuse a prefetch its peers register (RAENGE-NIE-UNEINS).
    """

    def test_the_tp_pin_rows_equal_the_pp_pool_rows(self):
        # T1. pp: 366211 rows x 16384 B on 8 layers; tp: 16 layers -> 32768 B.
        pools, builder = _build_patched(_sched())
        sa_pin = builder.calls[-1]["server_args"]
        self.assertEqual(sa_pin.hicache_ratio, 0)
        # size_gb_tp = ((pp_rows - 1) * cell_tp + cell_tp // 2) / 1e9: the
        # midpoint between (pp_rows - 1) and pp_rows whole cells, so the
        # floor division in base.py is robust to float error and lands on
        # pp_rows - 1, and the +1 page makes it pp_rows exactly.
        expected_gb = ((PP_ROWS - 1) * TP_CELL + TP_CELL // 2) / 1e9
        self.assertAlmostEqual(sa_pin.hicache_size, expected_gb, places=9)
        self.assertAlmostEqual(sa_pin.hicache_size, 11.999985664, places=9)
        self.assertEqual(builder.last.size, PP_ROWS, "rows must round-trip exactly")
        self.assertIn("tp", pools)

    def test_the_coupling_holds_for_other_layer_counts(self):
        # The formula is not a fitted constant: any tp layer count round-trips.
        for layers in (24, 7, 5, 4):
            s = _sched()
            s.phase_flip_stacks.tp_worker.model_runner.token_to_kv_pool.layer_num = layers
            _, builder = _build_patched(s)
            self.assertEqual(builder.last.size, PP_ROWS, f"layers={layers}")

    def test_row_mismatch_is_a_raise_not_a_soft_refusal(self):
        # T2. A builder that lands one row short is a phase pin that would
        # disagree with its peers' pp pool: refuse the BOOT, never log-and-go.
        builder = _FakeBuildKVHostPool(force_size=PP_ROWS - 1)
        with self.assertRaises(RuntimeError) as cm:
            _build_patched(_sched(), builder=builder)
        msg = str(cm.exception)
        self.assertIn("#1068 TP PIN ROW MISMATCH", msg)
        self.assertIn(f"tp_rows={PP_ROWS - 1}", msg)
        self.assertIn(f"pp_rows={PP_ROWS}", msg)
        self.assertIn(f"cell_tp={TP_CELL}", msg)
        self.assertIn(f"cell_pp={PP_CELL}", msg)

    def test_the_pin_below_one_wave_is_refused(self):
        # T3 (G10). 30518 rows cannot hold max_running_requests x chunk =
        # 8 x 4096 = 32768 in flight; the old max(1, ...) built it silently.
        s = _sched(pp_host=_PPHost(_PPKVHost(size=30518)))
        with self.assertRaises(ValueError) as cm:
            _build_patched(s)
        msg = str(cm.exception)
        self.assertIn("pp_rows=30518", msg)
        self.assertIn("max_running_requests=8", msg)
        self.assertIn("chunked_prefill_size=4096", msg)
        self.assertIn("--hicache-size", msg)

    def test_a_degenerate_wave_skips_the_floor_by_name(self):
        # chunked_prefill_size=-1 (chunking off) leaves the wave size
        # undefined, so the floor cannot be applied. The exit is NAMED in the
        # log rather than silently skipped: 30518 rows would be refused under
        # any positive chunk, and the operator must be able to see why not.
        s = _sched(pp_host=_PPHost(_PPKVHost(size=30518)), chunked_prefill_size=-1)
        with self.assertLogs("sglang.srt.managers.phase_flip_boot", level="WARNING") as logs:
            pools, _ = _build_patched(s)
        self.assertIn("tp", pools)
        self.assertTrue(
            any(
                "#1068 ONE-WAVE FLOOR SKIPPED" in m
                and "chunked_prefill_size=-1" in m
                and "pp_rows=30518" in m
                for m in logs.output
            ),
            logs.output,
        )

    def test_a_cell_the_pp_layers_do_not_divide_is_refused(self):
        # per_layer * layer_num must reproduce the pp cell exactly, or the
        # tp cell derived from it is a guess (R7 of the spec).
        s = _sched(pp_host=_PPHost(_PPKVHost(size_per_token=16383)))
        with self.assertRaises(ValueError) as cm:
            _build_patched(s)
        self.assertIn("16383", str(cm.exception))

    def test_a_pp_tier_without_a_readable_shape_refuses_softly(self):
        # A stand-in with no rows/cell is not a live composite; the rebind
        # refuses at the first cutover exactly as before (#847 contract).
        s = _sched(pp_host=types.SimpleNamespace())
        pools, _ = _build_patched(s)
        self.assertEqual(sorted(pools), ["pp"])


class TestTheTpPinAnchorHalfIsCoupledToThePpAnchor(CustomTestCase):
    """#1068 WEG 1 slice 1 (WEG1_BUILD_SPEC_0901 section 4.1, the MAMBA half
    of L12) -- review findings R1/R2 of 2026-09-02.

    On a hybrid model the bound tier carries KV AND MAMBA and the tp pin
    mirrors both (#871). The anchor half is sized per SLOT from the same
    knob the pp anchor pool takes, --hicache-mamba-host-mib, MIN-synced in
    MambaPoolHost -- so both phases hold the same slot count by construction
    -- and the builder HOLDS them to it: a pin whose anchor slots differ from
    the pp anchor pool is RAENGE-NIE-UNEINS and a boot refusal, and an anchor
    pool that cannot hold device_slots + max_running_requests + 1 is a
    too-small knob. Before these tests neither raise had a test, and the two
    mutants "mismatch raise off" and "knob nulled" both survived the suite.
    """

    def test_the_anchor_half_takes_the_same_knob_as_the_pp_anchor_pool(self):
        # R2. The knob IS the coupling: without it MambaPoolHost falls back to
        # the ratio path (host_size=0 -> sync_fixed_hicache_size returns early,
        # no collective, no group MIN), and "one budget, both phases" has no
        # carrier. The fallback form around it stays exactly as it was.
        pools, _, mamba = _build_hybrid(_hybrid_sched())
        self.assertEqual(len(mamba.calls), 1)
        call = mamba.calls[-1]
        self.assertEqual(call["anchor_host_mib"], ANCHOR_MIB)
        self.assertEqual(call["ratio"], 1.0, "fallback form untouched")
        self.assertEqual(call["host_size"], 0, "fallback form untouched")
        self.assertEqual(call["layout"], "layer_first")
        self.assertIn("tp", pools)
        self.assertEqual(set(pools["tp"].entry_map), {PoolName.KV, PoolName.MAMBA})

    def test_anchor_slot_mismatch_is_a_raise_not_a_soft_refusal(self):
        # R1. tp anchor 129 slots against pp anchor 65: the two phases would
        # hold different ceilings -- refuse the BOOT, never log-and-go.
        with self.assertRaises(RuntimeError) as cm:
            _build_hybrid(_hybrid_sched(), mamba=_FakeMambaPoolHost(size=129))
        msg = str(cm.exception)
        self.assertIn("#1068 TP PIN ANCHOR SLOT MISMATCH", msg)
        self.assertIn("tp_slots=129", msg)
        self.assertIn(f"pp_slots={ANCHOR_SLOTS}", msg)
        self.assertIn("RAENGE-NIE-UNEINS", msg)
        self.assertIn("--hicache-mamba-host-mib", msg)

    def test_an_anchor_pool_below_the_floor_is_refused(self):
        # 65 == 65 agrees, but 60 device slots + 8 in flight + 1 = 69 > 65.
        with self.assertRaises(ValueError) as cm:
            _build_hybrid(_hybrid_sched(device_slots=60))
        msg = str(cm.exception)
        self.assertIn("--hicache-mamba-host-mib too small", msg)
        self.assertIn(f"{ANCHOR_SLOTS} slots", msg)
        self.assertIn("device_slots 60", msg)
        self.assertIn("max_running_requests 8", msg)
        self.assertIn("= 69", msg)

    def test_the_floor_is_inclusive(self):
        # device_slots + max_running_requests + 1 == slots exactly is enough.
        pools, _, _ = _build_hybrid(_hybrid_sched(device_slots=ANCHOR_SLOTS - 8 - 1))
        self.assertIn("tp", pools)

    def test_the_mismatch_is_named_before_the_floor(self):
        # 10 slots is BOTH a mismatch (vs 65) and below the floor (20+8+1=29):
        # the disagreement is the graver refusal (RAENGE-NIE-UNEINS) and wins.
        with self.assertRaises(RuntimeError) as cm:
            _build_hybrid(_hybrid_sched(), mamba=_FakeMambaPoolHost(size=10))
        self.assertIn("#1068 TP PIN ANCHOR SLOT MISMATCH", str(cm.exception))

    def test_a_tp_stack_without_mamba_handles_keeps_the_kv_only_pin(self):
        # #871 contract: the hybrid half cannot be mirrored -> named reason,
        # KV-only pin registered, check_pool_coverage keeps refusing.
        pools, _, mamba = _build_hybrid(_hybrid_sched(handles=False))
        self.assertEqual(mamba.calls, [], "no anchor pool without the handles")
        self.assertIn("tp", pools)
        self.assertEqual(set(pools["tp"].entry_map), {PoolName.KV})


class TestTheRefusalIsCONVERTEDNotDeleted(CustomTestCase):
    """CAN-FAIL. A genuinely absent or mis-shaped pool must STILL refuse."""

    def test_no_host_tier_at_all_yields_nothing_to_bind(self):
        self.assertEqual(build_phase_flip_host_pools(_sched(host=False)), {})

    def test_no_tp_device_pool_leaves_the_tp_phase_unbound(self):
        pools = build_phase_flip_host_pools(_sched(tp_pool=False))
        self.assertEqual(sorted(pools), ["pp"])
        self.assertNotIn("tp", pools, "and the cutover must then refuse")

    def test_the_guard_still_raises_when_the_phase_is_unbound(self):
        # THE REAL GUARD, not a restatement. This is the W32 failure exactly,
        # and it must remain reachable -- the point of #847 is that the pool
        # can now exist, never that the check was removed.
        from sglang.srt.mem_cache.hicache_phase_binding import (
            RebindRefused,
            phase_pools_for,
        )

        s = _sched(tp_pool=False)
        s.phase_flip_host_pools = build_phase_flip_host_pools(s)
        with self.assertRaises(RebindRefused) as caught:
            phase_pools_for(s, "tp")
        self.assertIn("host pool", str(caught.exception))

    def test_a_constructor_that_refuses_does_not_take_down_the_boot(self):
        # A pin whose ALLOCATION fails (the assembler raises) is the #847
        # soft refusal: logged, pp handle kept, rebind refuses at the cutover.
        class _Boom:
            def __call__(self, **kw):
                raise RuntimeError("mis-shaped pool")

        pools, _ = _build_patched(_sched(), builder=_Boom())
        self.assertEqual(sorted(pools), ["pp"], "refuse loudly, boot anyway")


class TestTheBootWiresIt(CustomTestCase):
    def test_the_scheduler_calls_the_writer(self):
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        # W33 arm 1: the wiring must sit AFTER `self.tree_cache` is assigned,
        # not inside `init_model_worker`. It needs THREE inputs -- both device
        # pools AND the host tier -- and the host tier hangs off `tree_cache`,
        # which is assigned after that method returns. Placed too early the
        # writer runs, finds no host tier, and refuses; measured on metal.
        src = inspect.getsource(Scheduler.__init__)
        self.assertNotIn(
            "build_phase_flip_host_pools",
            inspect.getsource(Scheduler.init_model_worker),
            "too early: the host tier does not exist yet there",
        )
        self.assertIn("build_phase_flip_host_pools", src)
        self.assertIn("phase_flip_host_pools", src)

    def test_the_ledger_post_is_named(self):
        import inspect

        from sglang.srt.managers import phase_flip_boot

        src = inspect.getsource(phase_flip_boot.build_phase_flip_host_pools)
        self.assertIn("HOST-LEDGER POST", src)
        self.assertIn("floor", src.lower())


if __name__ == "__main__":
    unittest.main()


class TestTheHostTierAccessorKnowsTheLiveTree(CustomTestCase):
    """W33 arm 2 -- the W29 defect, repeated one strand later by me.

    The writer read `getattr(tree, "token_to_kv_pool_host", None)`. That
    attribute belongs to `HiRadixCache`; the tree this box runs is
    `UnifiedRadixCache`, which does not have it and reaches the host tier
    through `cache_controller.mem_pool_host`. The read returned None on the
    live tree, the writer logged its own "no HiCache host tier" refusal on
    every rank, and the rebind could not arm -- 6 `#718 hicache-phase-guard`
    warnings, the W32 read-through miss reproduced with its fix installed.

    Identical in shape to W29's `full_evictable_size_`: an attribute three
    tree types have and `UnifiedRadixCache` does not, with `getattr(..., 0)`
    turning the absence into a value that silently disabled the mechanism.
    Same tree class, same silent default. Hence a NAMED accessor plus this
    drift-detector, which tests the REAL classes rather than a double.
    """

    def test_the_direct_attribute_route(self):
        from sglang.srt.managers.phase_flip_boot import host_tier_of

        pool = object()
        self.assertIs(
            host_tier_of(types.SimpleNamespace(token_to_kv_pool_host=pool)), pool
        )

    def test_the_cache_controller_route_the_live_tree_uses(self):
        from sglang.srt.managers.phase_flip_boot import host_tier_of

        pool = object()
        tree = types.SimpleNamespace(
            token_to_kv_pool_host=None,
            cache_controller=types.SimpleNamespace(mem_pool_host=pool),
        )
        self.assertIs(host_tier_of(tree), pool)

    def test_absent_is_absent_not_a_route_i_forgot_to_look_at(self):
        from sglang.srt.managers.phase_flip_boot import host_tier_of

        self.assertIsNone(host_tier_of(types.SimpleNamespace()))
        self.assertIsNone(host_tier_of(None))

    def test_the_live_tree_class_really_lacks_the_direct_attribute(self):
        # THE DRIFT-DETECTOR, against the REAL class. If UnifiedRadixCache ever
        # grows `token_to_kv_pool_host`, the first route starts working and
        # this test says so -- rather than the accessor quietly depending on a
        # route that only some trees have, which is the whole defect.
        import inspect

        from sglang.srt.mem_cache import unified_radix_cache

        src = inspect.getsource(unified_radix_cache)
        self.assertNotIn(
            "self.token_to_kv_pool_host",
            src,
            "UnifiedRadixCache reaches the host tier via cache_controller; if "
            "that changed, revisit host_tier_of",
        )

    def test_the_writer_uses_the_named_accessor(self):
        import inspect

        from sglang.srt.managers import phase_flip_boot

        src = inspect.getsource(phase_flip_boot.build_phase_flip_host_pools)
        self.assertIn("host_tier_of(tree)", src)
        self.assertNotIn('getattr(tree, "token_to_kv_pool_host"', src)


class TestTheDevicePoolIsUnwrappedLikeTheConsumerDoes(CustomTestCase):
    """W34 arm 2: `the TP device pool exposes no layer_num`.

    `phase_pools_for` does `inner = getattr(device_pool, "full_kv_pool",
    device_pool)` before building the PhasePools that `check_shapes` reads, so
    the pool whose `layer_num` must match is the INNER one. The writer read
    the wrapper and refused itself; the pool that had the attribute was one
    dereference away.

    Building from the wrapper would ALSO have been a latent mismatch even if
    the attribute had existed, because the shape check compares the inner
    pool. So the writer must unwrap exactly as its consumer does -- pinned
    here, and in the source, so the two cannot drift.
    """

    def test_a_wrapped_pool_is_unwrapped(self):
        inner = _DevicePool()
        inner.layer_num = 24
        wrapper = types.SimpleNamespace(full_kv_pool=inner)
        s = _sched()
        s.phase_flip_stacks.tp_worker.model_runner.token_to_kv_pool = wrapper

        pools, builder = _build_patched(s)
        self.assertIn("tp", pools, "the inner pool has layer_num=24")
        # and the host pool must be built FROM the inner pool, not the
        # wrapper -- otherwise check_shapes compares two different things
        self.assertIs(builder.calls[-1]["kv_pool"], inner)

    def test_the_writer_unwraps_the_same_field_the_consumer_does(self):
        import inspect

        from sglang.srt.managers import phase_flip_boot
        from sglang.srt.mem_cache import hicache_phase_binding

        writer = inspect.getsource(phase_flip_boot.build_phase_flip_host_pools)
        consumer = inspect.getsource(hicache_phase_binding.phase_pools_for)
        self.assertIn("full_kv_pool", writer)
        self.assertIn(
            "full_kv_pool",
            consumer,
            "if the consumer stops unwrapping, the writer must stop too",
        )


class TestTheCompositeCanStateItsLayerCount(CustomTestCase):
    """The defect that predates this work: `check_shapes` compares
    `host_pool.layer_num`, and `HostPoolGroup` delegated six properties to its
    anchor but not that one -- so on every boot whose host tier is a composite
    (this fork's live shape) the check could only read None and refuse."""

    def test_the_group_delegates_layer_num_to_its_anchor(self):
        from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup

        self.assertTrue(
            isinstance(getattr(HostPoolGroup, "layer_num", None), property),
            "check_shapes reads host_pool.layer_num and refuses on None",
        )

    def test_it_sits_with_its_neighbours(self):
        # start_layer/end_layer/dtype already delegate; layer_num is the same
        # kind of question and must not be answered differently.
        import inspect

        from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup

        src = inspect.getsource(HostPoolGroup)
        self.assertIn("self.anchor_entry.host_pool.layer_num", src)
