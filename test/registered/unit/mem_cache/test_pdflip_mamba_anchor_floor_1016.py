"""MAMBA-ANCHOR-FLOOR 1016: a fixed ``--hicache-mamba-host-mib`` under the code's own floor is RAISED to it,
on every construction site of the anchor host pool (assembler on P and D, the phase-flip rebind).

THE DEFECT (live, boot 1016_103428). Every NF profile set ``--hicache-mamba-host-mib 300`` in --extra-p and
--extra-d. 300 MiB bought D-TP0 6 host anchor slots (the law: device_slots 38 + max_running_requests 6 + 1 =
45) and P 9 (32 + 6 + 1 = 39). The only check of that law (phase_flip_boot.py, '--hicache-mamba-host-mib too
small') runs on the P rebind alone, so on D nothing named it: anchors were lost at the D sleep flush, X refusals,
mini P prefills, flip ping-pong. The ledger had chosen M=600 and the argv overruled it (argv beats planner).

THE FIX. ``MambaPoolHost`` computes the floor itself (``mamba_anchor_slot_floor``) and raises a fixed budget
that buys fewer slots, with the named line ``MAMBA-ANCHOR-FLOOR RAISED``; host RAM only warns (user rule 06.10.).
Rank-uniform: the fixed branches take the group MAX of the floor (``sync_fixed_hicache_floor``) in the group the
count was MIN-synced in; a rank without GDN state bids 0; the rebind matches the pp pool's count.

Hermetic like test_mamba_host_mib_synced_1068: allocation and registration stubbed, the collectives are fakes.

    CUDA_VISIBLE_DEVICES='' python -m pytest test/registered/unit/mem_cache/test_pdflip_mamba_anchor_floor_1016.py -q
"""

import ast
import os
import types
import unittest
from unittest import mock

import torch

from flliper.srt.mem_cache import memory_pool_host as mph
from flliper.srt.mem_cache.memory_pool_host import MambaPoolHost, mamba_anchor_slot_floor

MIB = 1024**2
#: per-slot bytes measured live: 300 MiB bought 6 slots on D-TP0, 9 on P (16.10.)
PER_SLOT_D = int(50 * MIB)
PER_SLOT_P = int(33 * MIB)
SRT = os.path.join(os.path.dirname(mph.__file__), "..")


def _device_pool(slots):
    cache = types.SimpleNamespace(
        conv=[torch.zeros(1, 1, 4, dtype=torch.float16)],
        temporal=torch.zeros(1, 1, 4, dtype=torch.float16),
    )
    return types.SimpleNamespace(num_mamba_layers=1, mamba_cache=cache, device="cpu", size=slots)


class _Sync:
    """Stand-ins for the two group reductions: MIN of the count (one rank alone = identity, or a fixed group
    value), MAX of the floor (a fixed group value models a peer with a larger device pool)."""

    def __init__(self, group_min=None, group_floor=None):
        self.group_min, self.group_floor = group_min, group_floor
        self.min_calls, self.floor_calls = [], []

    def min(self, size, host_size):
        self.min_calls.append((int(size), int(host_size)))
        return int(size) if self.group_min is None else min(int(size), self.group_min)

    def floor(self, floor, host_size):
        self.floor_calls.append((int(floor), int(host_size)))
        return int(floor) if self.group_floor is None else max(int(floor), self.group_floor)


def _build(per_slot, device_slots, sync=None, **kwargs):
    sync = sync or _Sync()
    with (
        mock.patch.object(MambaPoolHost, "get_size_per_token", return_value=per_slot),
        mock.patch.object(MambaPoolHost, "init_kv_buffer", lambda self: None),
        mock.patch.object(MambaPoolHost, "_init_write_back_staging_buffers", lambda self: None),
        mock.patch.object(MambaPoolHost, "clear", lambda self: None),
        mock.patch.object(mph, "check_and_register_pinned_post", lambda **kw: None),
        mock.patch.object(mph, "sync_fixed_hicache_size", sync.min),
        mock.patch.object(mph, "sync_fixed_hicache_floor", sync.floor),
    ):
        return MambaPoolHost(_device_pool(device_slots), 1.0, 0, layout="layer_first", **kwargs), sync


class TheLiveCaseIsRaised(unittest.TestCase):
    def test_d_tp0_300_mib_is_raised_to_45_slots(self):
        """RED before the fix: 300 MiB on D-TP0 buys 6 slots (+1 page row = 7); the floor is 38 + 6 + 1 = 45."""
        pool, _ = _build(PER_SLOT_D, 38, anchor_host_mib=300, max_running_requests=6)
        self.assertEqual(pool.anchor_synced_slots, 45)
        self.assertEqual(pool.size, 46)  # + the one page row every host pool adds

    def test_p_300_mib_is_raised_to_39_slots(self):
        pool, _ = _build(PER_SLOT_P, 32, anchor_host_mib=300, max_running_requests=6)
        self.assertEqual(pool.anchor_synced_slots, 39)

    def test_the_raise_is_a_named_warning_never_a_refusal(self):
        with self.assertLogs("flliper.srt.mem_cache.memory_pool_host", level="WARNING") as logs:
            _build(PER_SLOT_D, 38, anchor_host_mib=300, max_running_requests=6)
        line = [x for x in logs.output if "MAMBA-ANCHOR-FLOOR RAISED" in x]
        self.assertEqual(len(line), 1, logs.output)
        for term in ("bought 6 host anchor slots", "device_slots 38", "max_running_requests 6", "-> 45 slots", "never refuses"):
            self.assertIn(term, line[0])

    def test_a_budget_over_the_floor_is_untouched(self):
        """The corrected host profile (D 2600 MiB = 52 slots at 50 MiB) is not raised and logs no raise."""
        with self.assertNoLogs("flliper.srt.mem_cache.memory_pool_host", level="WARNING"):
            pool, _ = _build(PER_SLOT_D, 38, anchor_host_mib=2600, max_running_requests=6)
        self.assertEqual(pool.anchor_synced_slots, 2600 * MIB // PER_SLOT_D)


class TheFloorIsGroupUniform(unittest.TestCase):
    def test_the_fixed_branch_takes_the_group_max_of_the_floor(self):
        """A peer with a larger device pool (floor 47) lifts this rank (floor 45) to 47: one count per group."""
        sync = _Sync(group_floor=47)
        pool, sync = _build(PER_SLOT_D, 38, sync=sync, anchor_host_mib=300, max_running_requests=6)
        self.assertEqual(sync.floor_calls, [(45, 300)])
        self.assertEqual(pool.anchor_synced_slots, 47)

    def test_the_gb_branch_is_floored_and_synced_too(self):
        sync = _Sync()
        with mock.patch.object(mph, "fixed_host_pool_tokens", return_value=5):
            with (
                mock.patch.object(MambaPoolHost, "get_size_per_token", return_value=PER_SLOT_D),
                mock.patch.object(MambaPoolHost, "init_kv_buffer", lambda self: None),
                mock.patch.object(MambaPoolHost, "_init_write_back_staging_buffers", lambda self: None),
                mock.patch.object(MambaPoolHost, "clear", lambda self: None),
                mock.patch.object(mph, "check_and_register_pinned_post", lambda **kw: None),
                mock.patch.object(mph, "sync_fixed_hicache_size", sync.min),
                mock.patch.object(mph, "sync_fixed_hicache_floor", sync.floor),
            ):
                pool = MambaPoolHost(_device_pool(38), 1.0, 1, layout="layer_first", max_running_requests=6)
        self.assertEqual(sync.floor_calls, [(45, 1)])
        self.assertEqual(pool.anchor_synced_slots, 45)

    def test_a_rank_without_gdn_state_bids_zero(self):
        """A Form A worker (0 B/slot) must not lift the group with a floor it does not hold."""
        sync = _Sync()
        _build(0, 38, sync=sync, anchor_host_mib=300, max_running_requests=6)
        self.assertEqual(sync.floor_calls, [(0, 300)])

    def test_the_ratio_branch_is_floored_without_a_new_collective(self):
        sync = _Sync()
        pool, sync = _build(PER_SLOT_D, 38, sync=sync, anchor_host_mib=0, max_running_requests=6)
        self.assertEqual((sync.min_calls, sync.floor_calls), ([], []))
        self.assertEqual(pool.anchor_synced_slots, 45)

    def test_without_max_running_requests_nothing_changes_and_it_is_named(self):
        sync = _Sync()
        with self.assertLogs("flliper.srt.mem_cache.memory_pool_host", level="INFO") as logs:
            pool, sync = _build(PER_SLOT_D, 38, sync=sync, anchor_host_mib=300)
        self.assertEqual(sync.floor_calls, [])
        self.assertEqual(pool.anchor_synced_slots, 6)
        prov = [x for x in logs.output if "#1035 ANCHOR-POOL PROVENANCE" in x]
        self.assertIn("floor=skipped(max_running_requests unknown)", prov[0])

    def test_the_rebind_matches_the_pp_pool_count(self):
        """The phase-flip pin must hold the pp pool's count (#1068 TP PIN ANCHOR SLOT MISMATCH)."""
        pool, _ = _build(PER_SLOT_D, 30, anchor_host_mib=300, max_running_requests=6, slot_floor_extra=45)
        self.assertEqual(pool.anchor_synced_slots, 45)

    def test_the_law(self):
        self.assertEqual(mamba_anchor_slot_floor(38, 6), 45)
        self.assertEqual(mamba_anchor_slot_floor(32, 6), 39)
        self.assertEqual(mamba_anchor_slot_floor(38, 0), 0)


def _calls_with_kw(path, func_suffix, kw):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith(func_suffix):
            out.append(any(k.arg == kw for k in n.keywords))
    return out


class EveryBootPathPassesTheFloorTerms(unittest.TestCase):
    """The pool can only floor what it is told: both construction sites (assembler = P and D, Form A and PP;
    phase-flip rebind) must pass max_running_requests, the rebind also the pp pool's count."""

    def test_the_assembler_passes_max_running_requests(self):
        path = os.path.join(SRT, "mem_cache", "hybrid_cache", "hybrid_pool_assembler.py")
        self.assertEqual(_calls_with_kw(path, "_MambaCls", "max_running_requests"), [True])

    def test_the_rebind_passes_max_running_requests_and_the_pp_count(self):
        path = os.path.join(SRT, "managers", "phase_flip_boot.py")
        self.assertEqual(_calls_with_kw(path, "_MambaCls", "max_running_requests"), [True])
        self.assertEqual(_calls_with_kw(path, "_MambaCls", "slot_floor_extra"), [True])
        self.assertEqual(_calls_with_kw(path, "_hybrid_pin_entries", "pp_anchor_slots"), [True])


class _ArenaForm(MambaPoolHost):
    """the arena class's marker only (ArenaMambaPoolHost.arena_read = True)"""
    arena_read = True


def _build_cls(cls, per_slot, device_slots, sync=None, **kwargs):
    sync = sync or _Sync()
    with (
        mock.patch.object(MambaPoolHost, "get_size_per_token", return_value=per_slot),
        mock.patch.object(MambaPoolHost, "init_kv_buffer", lambda self: None),
        mock.patch.object(MambaPoolHost, "_init_write_back_staging_buffers", lambda self: None),
        mock.patch.object(mph, "check_and_register_pinned_post", lambda **kw: None),
        mock.patch.object(mph, "sync_fixed_hicache_size", sync.min),
        mock.patch.object(mph, "sync_fixed_hicache_floor", sync.floor),
    ):
        return cls(_device_pool(device_slots), 1.0, 0, layout="layer_first", **kwargs), sync


class TheArenaFormFloorsOnlyTheBypassRows(unittest.TestCase):
    """Planer 16.10. (host 2.4 GB MemAvailable): on the arena form the staging rows carry only bypass reads and
    transfers without an arena id; the device + running + 1 law belongs to the arena (launcher). Floor
    max_running + 1, named."""

    def test_300_mib_on_the_arena_form_is_raised_to_7_not_45(self):
        with self.assertLogs("flliper.srt.mem_cache.memory_pool_host", level="INFO") as logs:
            pool, sync = _build_cls(_ArenaForm, PER_SLOT_D, 38, anchor_host_mib=300, max_running_requests=6)
        self.assertEqual(sync.floor_calls, [(7, 300)])
        self.assertEqual(pool.anchor_synced_slots, 7)
        raised = [x for x in logs.output if "MAMBA-ANCHOR-FLOOR RAISED" in x]
        self.assertEqual(len(raised), 1, logs.output)
        self.assertIn("arena form: staging = max_running_requests 6 + 1 bypass rows = 7", raised[0])
        prov = [x for x in logs.output if "#1035 ANCHOR-POOL PROVENANCE" in x]
        self.assertIn("floor=7 (arena form: bypass rows)", prov[0])

    def test_the_host_profile_900_mib_is_kept(self):
        pool, _ = _build_cls(_ArenaForm, int(56.11 * MIB), 38, anchor_host_mib=900, max_running_requests=6)
        self.assertEqual(pool.anchor_synced_slots, 900 * MIB // int(56.11 * MIB))  # 16

    def test_the_plain_pool_keeps_the_full_law(self):
        pool, _ = _build_cls(MambaPoolHost, PER_SLOT_D, 38, anchor_host_mib=300, max_running_requests=6)
        self.assertEqual(pool.anchor_synced_slots, 45)


class TheStagingHighWaterMark(unittest.TestCase):
    """MAMBA-STAGING-HWM: rows in use at once, highest row, allocations and refusals per path, logged at a new
    high-water mark, periodically and at reset."""

    def _pool(self):
        from flliper.srt.mem_cache.pool_host import arena_mamba_pool as amp
        from flliper.srt.pdflip import sleep_staging
        with mock.patch.object(sleep_staging, "register_mamba_pool", lambda p: None):
            pool, _ = _build_cls(amp.ArenaMambaPoolHost, PER_SLOT_D, 38, anchor_host_mib=900, max_running_requests=6)
        return pool, amp

    def test_paths_hwm_and_reset(self):
        pool, amp = self._pool()
        log = "flliper.srt.mem_cache.pool_host.arena_mamba_pool"
        with self.assertLogs(log, level="INFO") as logs:
            a = pool.alloc(2)                                   # transfer without an arena id
            pool._pdflip_flush_tag = True
            b = pool.alloc(1)                                   # the flush
            pool._pdflip_flush_tag = False
            with mock.patch.object(amp, "anchor_bypass_on", lambda env=None: True):
                c = pool._bypass_row()                          # a bypass read
            pool.free(a)
            st = pool._stg_state()
            self.assertEqual((st["hwm"], len(st["in_use"])), (4, 2))
            self.assertEqual(st["alloc"], {"bypass": 1, "transfer": 2, "flush": 1})
            self.assertEqual(st["max_row"], max(int(x) for x in list(a) + list(b) + list(c)))
            pool.clear()
        lines = [x for x in logs.output if "MAMBA-STAGING-HWM" in x]
        self.assertTrue(any("new-hwm path=bypass" in x and "hwm=4" in x for x in lines), lines)
        self.assertTrue(any(" reset " in x and "alloc bypass=1 transfer=2 flush=1" in x for x in lines), lines)
        self.assertEqual(len(pool._stg_state()["in_use"]), 0)

    def test_a_refusal_is_counted_per_path(self):
        pool, amp = self._pool()
        self.assertIsNone(pool.alloc(pool.size + 5))
        self.assertEqual(pool._stg_state()["refused"]["transfer"], 1)

    def test_the_radix_cache_tags_the_flush_on_the_pool_it_hands_out(self):
        src = open(os.path.join(SRT, "mem_cache", "unified_radix_cache.py"), encoding="utf-8").read()
        self.assertIn('mp._pdflip_flush_tag = bool(getattr(self, "_pdflip_flush_spill", False)); return mp', src)


if __name__ == "__main__":
    unittest.main()
