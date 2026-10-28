# SPDX-License-Identifier: Apache-2.0
"""D-POOL-LEND (flip form, 27.10.): the KV pool tail is card free memory while ONE prefill runs.

DANGER DIRECTIONS guarded here (a mock driver that models ``tms_set_spans``: an extent wholly inside the new
plan keeps its pages, the others are unmapped, the gaps map fresh, a card without room answers rc=2):
* gate: only group D of the FLIP form -- never the dual layout (its D-KV stage owns the pool), never the seat
  form, never group P, never with the switch off;
* the lattice cut maps the WHOLE pool (boot = top), every extent is one cell, the tensors keep their size;
* a release goes down to T on the lattice, the cap first, the freed bytes show on the card; the return maps
  exactly them and the card reads what it read before (frei_nachher = frei_vorher);
* a live id above T is never released under (the floor decides), the compaction runs only when the used ids
  would fit;
* a second pending prefill returns the tail at once; SETTLE decode rounds return it (hysteresis);
* a card that cannot map the return is rolled back on EVERY rank (GROW-SHORT), degraded not dead;
* D never sleeps lent (plan back to the full lattice without mapping, cap lifted, rows unbounded);
* the dual's gates are untouched (compact armed under dual; low-first N default from the dual switch).
"""
from __future__ import annotations

import inspect
import logging
import os
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from flliper.srt.pdflip import d_pool_lend as L
from flliper.srt.pdflip import d_seat_vram as SV
from flliper.srt.pdflip import dual_d_compact as DC
from flliper.srt.pdflip import dual_d_kv_stage as DK
from flliper.srt.pdflip import dual_p_kv_stage as PK
from flliper.srt.pdflip.d_seat_vram import AllocInfo
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

G = 2 << 20
TOP = 65536                       # global tokens
ROW = 256                         # bytes per row of one K or V tensor
STEP = 16384
CARD = 400 << 20                  # the mock card's total bytes
FLIP = {"FLLIPER_PDFLIP_GROUP": "D"}


class MockDriver:
    """``tms_set_spans`` / ``tms_alloc_info`` by their documented semantics (tms_csrc/core.cpp set_spans)."""

    available = True

    def __init__(self, card_total: int = CARD):
        self.size, self.ext, self.plans, self.calls = {}, {}, {}, []
        self.card_total = card_total

    def add(self, t, size: int) -> None:
        self.size[t.data_ptr()] = size
        self.ext[t.data_ptr()] = [(0, size)]

    def mapped_all(self) -> int:
        return sum(b - a for e in self.ext.values() for a, b in e)

    def card_free(self) -> int:
        return self.card_total - self.mapped_all() - (30 << 20)

    def info(self, ptr):
        if ptr not in self.size:
            return None
        return AllocInfo(size=self.size[ptr], mapped=sum(b - a for a, b in self.ext[ptr]), planned=0, active=True)

    def set_spans(self, ptr, spans, now):
        plan = [(int(a), int(b)) for a, b in spans]
        self.calls.append((ptr, tuple(plan), bool(now)))
        self.plans[ptr] = plan
        if not now:
            return 0
        want = plan or [(0, self.size[ptr])]
        kept = [e for e in self.ext[ptr] if any(lo <= e[0] and e[1] <= hi for lo, hi in want)]
        self.ext[ptr] = kept
        for lo, hi in want:
            for a, b in self._gaps(lo, hi, kept):
                if b - a > self.card_free():
                    return 2                                 # cuMemCreate out of memory
                self.ext[ptr] = sorted(self.ext[ptr] + [(a, b)])
        return 0

    @staticmethod
    def _gaps(lo, hi, kept):
        out, cur = [], lo
        for a, b in sorted(e for e in kept if e[0] >= lo and e[1] <= hi):
            if a > cur:
                out.append((cur, a))
            cur = b
        if cur < hi:
            out.append((cur, hi))
        return out


class FakePool:
    page_size = 1

    def __init__(self, drv: MockDriver, layers: int = 2):
        self.size = TOP
        self.stage_rows = []
        rows = TOP + self.page_size
        self.k_buffer = [torch.ones(rows, ROW // 4) for _ in range(layers)]
        self.v_buffer = [torch.ones(rows, ROW // 4) for _ in range(layers)]
        for t in self.k_buffer + self.v_buffer:
            drv.add(t, -(-rows * ROW // G) * G)

    def set_stage_backed_rows(self, rows):
        self.stage_rows.append(rows)


class FakeAllocator:
    """Free ids as the token allocator keeps them; ``engage_cap`` below withholds the ids above the cap."""

    page_size = 1

    def __init__(self, live=()):
        self.size = TOP
        live = set(live)
        self.free_pages = torch.tensor([i for i in range(1, TOP + 1) if i not in live], dtype=torch.int64)
        self.release_pages = torch.empty(0, dtype=torch.int64)
        self.withheld = torch.empty(0, dtype=torch.int64)
        self.caps = []

    @property
    def _pdflip_kv_stage_cap(self):
        return types.SimpleNamespace(_withheld=self.withheld)       # what max_live_id reads of the real KvRowCap

    def engage_cap(self, tokens):
        self.caps.append(int(tokens))
        both = torch.cat([self.free_pages, self.withheld])
        over = both > tokens if tokens < self.size else torch.zeros_like(both, dtype=torch.bool)
        self.free_pages, self.withheld = both[~over], both[over]

    def free(self, ids):
        self.free_pages = torch.cat([self.free_pages, torch.tensor(ids, dtype=torch.int64)])


def make(live=(), card_total: int = CARD, **kn):
    drv = MockDriver(card_total)
    pool = FakePool(drv)
    alloc = FakeAllocator(live)
    step = kn.get("step", STEP)
    knobs = L.Knobs(step=step, settle=kn.get("settle", 3), min_rows=kn.get("min_rows", 4096),
                    recheck=kn.get("recheck", 2), low_first=0)
    born, extents, reason = L._cut_pools([pool], top=TOP, spans=drv, granule=G, step=step)
    assert reason == "", reason
    trims = []
    actor = L.PoolLend(born, allocator=alloc, pools=[pool], page_size=1, granule=G, top_tokens=TOP, knobs=knobs,
                       rank=1, spans=drv, engage_cap=lambda a, tok, pg: a.engage_cap(tok), sync=lambda: None,
                       free_fn=drv.card_free, trim_fn=lambda: trims.append(1))
    L._ACTOR = actor
    return drv, pool, alloc, actor, extents, trims


def req(n_tokens: int):
    return types.SimpleNamespace(rid="r%d" % n_tokens, origin_input_ids=[0] * n_tokens, output_ids=[],
                                 prefix_indices=None)


def sched(waiting=(), chunked=None, running=0):
    return types.SimpleNamespace(
        chunked_req=chunked, waiting_queue=list(waiting), running_batch=types.SimpleNamespace(reqs=[1] * running),
        chunked_prefill_size=4096,
        server_args=types.SimpleNamespace(max_running_requests=6, chunked_prefill_size=4096,
                                          speculative_num_draft_tokens=0),
        _pdflip_group_min_ints=lambda vals: [int(v) for v in vals])


class Gate(CustomTestCase):
    def test_only_group_d_of_the_flip_form(self):
        env = {"FLLIPER_PDFLIP_GROUP": "D"}
        self.assertTrue(L.armed(env))
        self.assertFalse(L.armed({"FLLIPER_PDFLIP_GROUP": "P"}))
        self.assertFalse(L.armed({}))
        self.assertFalse(L.armed({**env, "FLLIPER_PDFLIP_DUAL_LAYOUT": "1"}))      # the dual D-KV stage owns it
        with mock.patch.object(SV, "armed", return_value=True):                      # the seat/stage form
            self.assertFalse(L.armed(env))

    def test_switch_off_is_off(self):
        from flliper.srt.environ import envs

        with envs.FLLIPER_PDFLIP_D_POOL_LEND.override(False):
            self.assertFalse(L.armed({"FLLIPER_PDFLIP_GROUP": "D"}))

    def test_dual_compact_gate_is_widened_not_changed(self):
        dual = {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "D", DK.MAX_TOKENS_ENV: "65536"}
        self.assertTrue(DC.armed(dual))
        self.assertTrue(DC.armed({"FLLIPER_PDFLIP_GROUP": "D"}))                    # the flip form's pool lend
        self.assertFalse(DC.armed({"FLLIPER_PDFLIP_GROUP": "P"}))
        self.assertFalse(DC.armed({"FLLIPER_PDFLIP_GROUP": "D", DC.SWITCH_ENV: "0"}))
        self.assertFalse(DC.armed({"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "D",
                                   DK.MAX_TOKENS_ENV: "0"}))                         # the dual rule itself

    def test_dual_low_first_keeps_its_own_switch(self):
        alloc = types.SimpleNamespace(set_low_first=lambda n: calls.append(n))
        calls = []
        with mock.patch.object(DK, "low_first_every", return_value=0):
            self.assertEqual(DK._arm_low_first(alloc), 0)                            # dual default: off
        self.assertEqual(calls, [])


class Pure(CustomTestCase):
    def test_need_is_on_the_lattice(self):
        self.assertEqual(L.need_tokens(live=0, incoming=1, air=0, step=STEP), STEP)
        self.assertEqual(L.need_tokens(live=STEP, incoming=0, air=0, step=STEP), STEP)
        self.assertEqual(L.need_tokens(live=STEP, incoming=1, air=0, step=STEP), 2 * STEP)

    def test_size_pool_release_needs_two_cells(self):
        kw = dict(top=TOP, step=STEP)
        self.assertEqual(L.size_pool(mapped=TOP, need=2 * STEP, **kw).kind, "release")
        self.assertEqual(L.size_pool(mapped=TOP, need=2 * STEP, **kw).target, 2 * STEP)
        self.assertEqual(L.size_pool(mapped=TOP, need=3 * STEP, **kw).kind, "hold")     # one cell is no lend
        self.assertEqual(L.size_pool(mapped=TOP, need=TOP + 5, **kw).kind, "hold")

    def test_size_pool_lent_only_grows(self):
        kw = dict(top=TOP, step=STEP)
        lent = L.size_pool(mapped=2 * STEP, need=3 * STEP, **kw)
        self.assertEqual((lent.kind, lent.target), ("grow", 3 * STEP))
        self.assertEqual(L.size_pool(mapped=2 * STEP, need=STEP, **kw).kind, "hold")     # never a second shrink
        self.assertEqual(L.size_pool(mapped=2 * STEP, need=TOP + 1, **kw).target, TOP)   # past the top = all back

    def test_auto_step_keeps_the_int8_lattice_and_gives_the_small_int4ar_pool_room(self):
        """Derived property: the fixed step 32768 left the int4ar pool (126178 rows per rank) without a release for a
        live floor above 60642 rows (a release needs MIN_RELEASE_CELLS cells under the top); the auto step keeps the
        INT8 drei pool (270000 rows, boot dkr27bbar1drei10272126) on 32768 and cuts the int4ar pool to ~11 cells."""
        self.assertEqual(L.resolve_step(0, 270000), 32768)
        self.assertEqual(L.resolve_step(0, 126178), 12288)
        self.assertEqual(L.resolve_step(16384, 126178), 16384)          # a set value is taken as set
        self.assertEqual(L.resolve_step(0, 20000), L.STEP_MIN)          # a tiny pool never goes under the floor
        top = 126178
        need32k = L.need_tokens(live=50000, incoming=12000, air=2048, step=32768)
        self.assertEqual(L.size_pool(mapped=top, top=top, step=32768, need=need32k).kind, "hold")
        step = L.resolve_step(0, top)
        need = L.need_tokens(live=50000, incoming=12000, air=2048, step=step)
        move = L.size_pool(mapped=top, top=top, step=step, need=need)
        self.assertEqual((move.kind, move.target), ("release", need))
        self.assertGreaterEqual(top - need, 2 * step)

    def test_intent_table(self):
        kw = dict(incoming=100000, busy=True, streak=0, settle=3, min_rows=4096)
        self.assertEqual(L.intent(lent=True, pend=2, **kw).kind, "back")
        self.assertEqual(L.intent(lent=False, pend=2, **kw).kind, "none")
        self.assertEqual(L.intent(lent=False, pend=1, **kw).kind, "check")
        self.assertEqual(L.intent(lent=False, pend=1, **{**kw, "incoming": 100}).kind, "none")   # too short
        self.assertEqual(L.intent(lent=True, pend=1, **{**kw, "incoming": 100}).kind, "check")   # lent: size it
        self.assertEqual(L.intent(lent=True, pend=0, **{**kw, "streak": 2}).kind, "check")
        self.assertEqual(L.intent(lent=True, pend=0, **{**kw, "streak": 3}).kind, "back")        # hysteresis
        self.assertEqual(L.intent(lent=True, pend=0, **{**kw, "streak": 99, "busy": False}).kind, "none")  # one idle pass
        self.assertEqual(L.intent(lent=False, pend=0, **kw).kind, "none")

    def test_an_idle_lent_d_gives_the_tail_back_after_settle_idle_passes(self):
        # boot int4arxd 28.10. 01:12Z: lent to T=28672 after an 8k prompt that finished at once (nothing running),
        # the front held the next 32k prompt back on the lent pool's deficit and D saw no pending prefill: both
        # waited until the timeout. Lent and idle for ``settle`` passes in a row -> back. One idle pass is not
        # enough (the pass right after a prefill runs before the finished request joins the running batch).
        kw = dict(incoming=0, streak=0, settle=3, min_rows=4096, busy=False)
        self.assertEqual(L.intent(lent=True, pend=0, idle=2, **kw).kind, "none")
        self.assertEqual(L.intent(lent=True, pend=0, idle=3, **kw).kind, "back")
        self.assertEqual(L.intent(lent=False, pend=0, idle=99, **kw).kind, "none")              # nothing lent
        self.assertEqual(L.intent(lent=True, pend=1, idle=99, **{**kw, "incoming": 100}).kind, "check")  # mid-prefill

    def test_streak_counts_busy_idle_keeps_pending_resets(self):
        self.assertEqual(L.next_streak(streak=2, pend=0, busy=True), 3)
        self.assertEqual(L.next_streak(streak=2, pend=0, busy=False), 2)
        self.assertEqual(L.next_streak(streak=2, pend=1, busy=True), 0)


class Cut(CustomTestCase):
    def test_boot_is_top_every_extent_is_one_cell(self):
        drv, pool, _alloc, actor, extents, _ = make()
        self.assertEqual(len(actor.born), 4)
        for t in pool.k_buffer + pool.v_buffer:
            ext = drv.ext[t.data_ptr()]
            self.assertGreaterEqual(len(ext), TOP // STEP)                                  # one per cell
            self.assertEqual(len(ext), len(drv.plans[t.data_ptr()]))                        # = the plan's ranges
            self.assertEqual((ext[0][0], ext[-1][1]), (0, drv.size[t.data_ptr()]))          # the whole pool
            self.assertTrue(all(ext[i][1] == ext[i + 1][0] for i in range(len(ext) - 1)))   # no hole
            self.assertEqual(int(t.sum()), 0)                                               # zeroed after the re-map
        self.assertEqual(extents, sum(len(e) for e in drv.ext.values()))
        self.assertEqual(actor.mapped_tokens, TOP)
        self.assertFalse(actor.lent)

    def test_owner_block_geometry_maps_the_top_onto_all_rows(self):
        S, ratio = 64, 20
        rows = (TOP // S + 1) * ratio + 1
        t = torch.zeros(rows, ROW // 4)
        pool = types.SimpleNamespace(size=rows - 1, page_size=1)
        with mock.patch.object(SV, "kv_owner_block", return_value=(S, ratio)):
            geom = L._tensor_geom(t, pool, TOP, -(-rows * ROW // G) * G, "k0")
        self.assertIsNotNone(geom)
        self.assertEqual(geom.slots_for(TOP), rows)
        self.assertLess(geom.slots_for(TOP // 2), geom.slots_for(TOP))
        with mock.patch.object(SV, "kv_owner_block", return_value=(0, 0)):                  # not this layout
            self.assertIsNone(L._tensor_geom(t, pool, TOP, 1 << 22, "k0"))

    def test_not_a_saver_allocation_arms_nothing(self):
        drv = MockDriver()
        pool = FakePool(drv)
        drv.size.pop(pool.k_buffer[1].data_ptr())
        born, extents, reason = L._cut_pools([pool], top=TOP, spans=drv, granule=G, step=STEP)
        self.assertEqual((born, extents), ([], 0))
        self.assertIn("not a saver allocation", reason)


class Moves(CustomTestCase):
    def test_release_frees_the_cells_above_t_and_return_reads_the_same_card(self):
        drv, pool, alloc, actor, _, _ = make()
        free0, mapped0 = drv.card_free(), drv.mapped_all()
        with self.assertLogs(L.logger, level=logging.INFO) as cm:
            actor.release_to(2 * STEP, note="test")
            freed = drv.card_free() - free0
            self.assertTrue(actor.grow_to(TOP, why="test"))
        self.assertEqual(freed, actor.bytes_for(TOP) - actor.bytes_for(2 * STEP))   # exactly the cells above T
        self.assertGreaterEqual(freed, 4 * (TOP - 2 * STEP - STEP) * ROW)            # and that is the tail
        self.assertEqual(drv.card_free(), free0)                               # frei_nachher = frei_vorher
        self.assertEqual(drv.mapped_all(), mapped0)
        self.assertEqual(alloc.caps[0], 2 * STEP)                              # the cap before the unmap
        self.assertEqual(alloc.caps[-1], TOP)
        self.assertEqual(pool.stage_rows[-1], None)                            # full again: no stage bound
        self.assertEqual(int(alloc.withheld.numel()), 0)
        rel = next(m for m in cm.output if " release T=" in m)
        ret = next(m for m in cm.output if " return T=" in m)
        self.assertIn("rank=1 release T=%d (from %d)" % (2 * STEP, TOP), rel)
        self.assertIn("card_free_mib before=", rel)
        self.assertIn("pre_lend_free_mib=%d delta_mib=0" % (free0 // L.MIB), ret)

    def test_release_keeps_the_cells_it_keeps(self):
        drv, pool, _alloc, actor, _, _ = make()
        t = pool.k_buffer[0]
        low = list(drv.ext[t.data_ptr()][:2])
        actor.release_to(2 * STEP, note="test")
        self.assertEqual(drv.ext[t.data_ptr()][:2], low)                       # same extents: bytes kept
        self.assertLess(drv.info(t.data_ptr()).mapped, drv.size[t.data_ptr()])   # the tail is gone
        self.assertEqual(pool.stage_rows[-1], actor.born[0][1].slots_for(2 * STEP))

    def test_release_under_a_live_id_stops_by_name(self):
        _drv, _pool, alloc, actor, _, _ = make(live=[TOP - 5])
        with self.assertRaises(PK.PdFlipDualKvCapBreach):
            actor.release_to(2 * STEP, note="test")
        self.assertEqual(actor.mapped_tokens, TOP)

    def test_grow_short_rolls_back_on_every_rank(self):
        drv, pool, alloc, actor, _, trims = make()
        actor.release_to(2 * STEP, note="test")
        drv.card_total = drv.mapped_all() + (30 << 20) + (3 << 20)              # the card lost the room meanwhile
        mapped = drv.mapped_all()
        with self.assertLogs(L.logger, level=logging.WARNING) as cm:
            ok = actor.grow_to(TOP, why="test")
        self.assertFalse(ok)
        self.assertEqual(trims, [1])                                            # empty_cache before the map
        self.assertEqual(actor.mapped_tokens, 2 * STEP)
        self.assertEqual(drv.mapped_all(), mapped)                              # what this rank got is unmapped
        self.assertEqual(alloc.caps[-1], 2 * STEP)                              # cap back at T
        self.assertEqual(actor.next_round, actor.rounds + L.GROW_SHORT_BACKOFF_ROUNDS)
        self.assertIn("GROW-SHORT", cm.output[0])

    def test_another_ranks_short_card_rolls_this_rank_back_too(self):
        drv, _pool, alloc, actor, _, _ = make()
        actor.release_to(2 * STEP, note="test")
        actor.gmin = lambda vals: [0]                                           # the group verdict: not all mapped
        mapped = drv.mapped_all()
        self.assertFalse(actor.grow_to(TOP, why="test"))
        self.assertEqual((actor.mapped_tokens, drv.mapped_all()), (2 * STEP, mapped))

    def test_quiesce_puts_the_plan_back_without_mapping(self):
        drv, pool, alloc, actor, _, _ = make()
        n = len(drv.calls)
        self.assertFalse(actor.quiesce())                                       # not lent: nothing to do
        self.assertEqual(len(drv.calls), n)
        actor.release_to(2 * STEP, note="test")
        calls = len(drv.calls)
        mapped = drv.mapped_all()
        self.assertTrue(actor.quiesce())
        self.assertEqual(drv.mapped_all(), mapped)                              # nothing mapped here
        later = drv.calls[calls:]
        self.assertEqual(len(later), 4)
        self.assertTrue(all(not now for _p, _plan, now in later))               # plan only
        for ptr, plan, _now in later:
            self.assertEqual((plan[0][0], plan[-1][1]), (0, drv.size[ptr]))      # the full lattice
        self.assertEqual((actor.mapped_tokens, pool.stage_rows[-1], alloc.caps[-1]), (TOP, None, TOP))
        self.assertEqual(len(drv.calls), calls + 4)                             # and nothing else was called


class Round(CustomTestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {**FLIP, DC.SWITCH_ENV: "0"})
        self._env.start()
        self.addCleanup(self._env.stop)
        self.addCleanup(setattr, L, "_ACTOR", None)

    def test_a_sole_prefill_releases_before_the_width_vote(self):
        drv, pool, alloc, actor, _, _ = make()
        free0 = drv.card_free()
        s = sched(waiting=[req(30000)])
        self.assertEqual(L.round_start(s), "release")
        need = L.need_tokens(live=0, incoming=30000, air=L._air(s, 2), step=STEP)
        self.assertEqual(actor.mapped_tokens, need)
        self.assertGreater(drv.card_free(), free0)
        self.assertEqual(alloc.caps[0], need)

    def test_a_short_prefill_lends_nothing(self):
        drv, _pool, _alloc, actor, _, _ = make()
        self.assertIsNone(L.round_start(sched(waiting=[req(1000)])))
        self.assertFalse(actor.lent)

    def test_a_second_prefill_returns_the_tail_at_once(self):
        drv, _pool, alloc, actor, _, _ = make()
        free0 = drv.card_free()
        s = sched(waiting=[req(30000)])
        L.round_start(s)
        s.waiting_queue.append(req(2000))
        self.assertEqual(L.round_start(s), "back")
        self.assertFalse(actor.lent)
        self.assertEqual(drv.card_free(), free0)

    def test_settled_decode_rounds_return_it_not_before(self):
        drv, _pool, _alloc, actor, _, _ = make()
        s = sched(waiting=[req(30000)])
        L.round_start(s)
        s.waiting_queue.clear()
        s.running_batch.reqs.append(1)
        out = [L.round_start(s) for _ in range(3)]
        self.assertEqual(out[:2], [None, None])
        self.assertEqual(out[2], "back")
        self.assertFalse(actor.lent)

    def test_idle_loop_passes_keep_the_lend(self):
        _drv, _pool, _alloc, actor, _, _ = make()
        s = sched(waiting=[req(30000)])
        L.round_start(s)
        s.waiting_queue.clear()
        for _ in range(actor.knobs.settle - 1):
            self.assertIsNone(L.round_start(s))                                  # nothing running: no hysteresis
        self.assertTrue(actor.lent)

    def test_a_long_idle_gives_the_lend_back(self):
        # the same idle D, ``settle`` passes later: the tail returns (int4arxd 28.10.: the front waited on it)
        _drv, _pool, _alloc, actor, _, _ = make()
        s = sched(waiting=[req(30000)])
        L.round_start(s)
        s.waiting_queue.clear()
        outs = [L.round_start(s) for _ in range(actor.knobs.settle + 2)]
        self.assertIn("back", outs)
        self.assertFalse(actor.lent)

    def test_the_need_outgrowing_t_grows_the_pool(self):
        drv, _pool, alloc, actor, _, _ = make()
        s = sched(waiting=[req(30000)])
        L.round_start(s)
        t0 = actor.mapped_tokens
        alloc.free_pages = alloc.free_pages[alloc.free_pages != t0 - 100]        # decode rows reached T - 100
        s.waiting_queue.clear()
        s.running_batch.reqs.append(1)
        outs = [L.round_start(s) for _ in range(4)]
        self.assertIn("grow", outs)
        self.assertGreater(actor.mapped_tokens, t0)

    def test_a_live_row_high_up_holds_the_release_and_says_so(self):
        drv, _pool, _alloc, actor, _, _ = make(live=[TOP - 3])
        with self.assertLogs(L.logger, level=logging.INFO) as cm:
            self.assertIsNone(L.round_start(sched(waiting=[req(30000)])))
        self.assertFalse(actor.lent)
        self.assertIn("skip live=%d" % (TOP - 3), cm.output[0])

    def test_compaction_runs_when_the_used_ids_would_fit_and_the_release_follows(self):
        drv, _pool, alloc, actor, _, _ = make(live=[TOP - 3], step=8192)
        def moved(*_a, **_k):                       # the compaction: the live row goes from TOP-3 down to id 1
            alloc.free_pages = torch.cat([alloc.free_pages[alloc.free_pages != 1], torch.tensor([TOP - 3])])
            return 1

        with mock.patch.dict(os.environ, {DC.SWITCH_ENV: "1"}), \
                mock.patch.object(DC, "run", side_effect=moved) as run:
            self.assertEqual(L.round_start(sched(waiting=[req(5000)])), "release")
        kw = run.call_args.kwargs
        self.assertEqual(kw["floor"], TOP - 3)
        self.assertEqual(kw["target"], L.compact_target(used_ids=1, step=8192))
        self.assertTrue(actor.lent)
        self.assertLessEqual(actor.mapped_tokens, 8192 * 2)                      # sized on the NEW floor

    def test_a_refused_compaction_backs_off_and_releases_nothing(self):
        drv, _pool, _alloc, actor, _, _ = make(live=[TOP - 3], step=8192)
        with mock.patch.dict(os.environ, {DC.SWITCH_ENV: "1"}), mock.patch.object(DC, "run", return_value=None):
            self.assertIsNone(L.round_start(sched(waiting=[req(5000)])))
        self.assertFalse(actor.lent)
        self.assertEqual(actor.refusals, 1)
        self.assertEqual(actor.next_round, actor.rounds + actor.knobs.recheck)

    def test_no_compaction_when_even_the_used_ids_leave_no_cell(self):
        drv, _pool, _alloc, actor, _, _ = make(live=[TOP - 3])                      # 20000 rows + one cell of room
        with mock.patch.dict(os.environ, {DC.SWITCH_ENV: "1"}), mock.patch.object(DC, "run") as run:
            self.assertIsNone(L.round_start(sched(waiting=[req(20000)])))
        run.assert_not_called()

    def test_no_actor_no_effect(self):
        L._ACTOR = None
        self.assertIsNone(L.round_start(sched(waiting=[req(30000)])))
        self.assertFalse(L.quiesce_for_sleep(sched()))


class Wiring(CustomTestCase):
    def test_hooks_are_wired_in_order(self):
        from flliper.srt.managers import scheduler as SC
        from flliper.srt.managers.scheduler_components import weight_updater as WU
        from flliper.srt.model_executor import model_runner_kv_cache_mixin as MX

        src = inspect.getsource(SC.Scheduler.get_next_batch_to_run)
        self.assertLess(src.index("_pdflip_d_transient_lend.round_start(self)"),
                        src.index("_pdflip_d_pool_lend.round_start(self)"))
        self.assertLess(src.index("_pdflip_d_pool_lend.round_start(self)"), src.index("_ddk.tick(self)"))
        self.assertIn("_dpl.attach(self)", inspect.getsource(MX))
        wu = inspect.getsource(WU)
        hook = wu.index("_dpl.quiesce_for_sleep(self.scheduler)")
        pause = wu.index("self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)", hook)
        self.assertLess(pause - hook, 200)                                           # right before the kv pause
        flush = wu.rindex('_kvsub.mark("flush")', 0, hook)
        self.assertNotIn(".pause(", wu[flush:hook])                                  # after the flush, no pause between
