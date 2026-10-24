# SPDX-License-Identifier: Apache-2.0
"""#217 ADMIT-PEAK 24.10. (desk/27b-admit-peak-1024): what the D admission peak A IS, the fix that shrinks it, and the
record that keeps pricing the MEASURED peak until a boot measures the smaller one.

ROOT (measured on the D logs, fixture ``admit_peak_1024/stage_lines_D.json.gz``: every PDFLIP-ARENA-PAGE-LOAD /
-STATE-LOAD / SLEEP-STAGING line of nine 27B boots, with the trims around each logged load):

- KV page stage (arena_pool._load_pages_all_layers): a fresh ``torch.empty((min(8192, pages), 32 KiB))`` = 256 MiB per
  load; the trims hand its segment back, so every admission cudaMallocs it again. The forward trim after a load with
  >= 8192 rank pages releases 278..282 MiB in 54 of 62 int4ar passes (never less than 256).
- mamba state stage (arena_mamba_pool._load_states_all_layers): ``256 MiB // row`` slots, live until the sleep, re-made
  by the first load after every wake (the handback admission): 224.4 int4ar / 187 drei / 233.8 duo at the sleeps;
  ``A - released`` of such a pass is the stage + at most 7.6 MiB.
- A_peak 504..522 = the two stages + 22..32 MiB.

FIX: 64-MiB KV device stage ("cpu" mode keeps its 256-MiB pinned stage), one mamba row per block in "dma" mode with a
vectorised split (two strided index_copy_ per block, no per-layer temporaries) -- same bytes in the same rows.

RECORD: D_ADMIT_NEED_MIB stays 414 (measured under the 256-MiB stages, budget_rest.ADMIT_STAGE_BASIS); the #217
projection (A_peak 147.6, need 40) is a HOCHRECHNUNG and is never booked; only a census whose boots all ran the #217
stages may replace the record (budget_rest.admit_record_may_replace).

Danger directions pinned here:
- OOM: the projection booked as the record, the replace gate opened for old-geometry boots or boots without a load
  line, the gate's #217 thresholds widened to the old geometry.
- correctness: the vectorised split landing other bytes or other rows than the per-layer loop.
- regression of the fix: the KV device stage back at 256 MiB, the mamba dma block back at 256 MiB // row; the "cpu"
  mode losing its 256-MiB pinned stage (throughput, xsn325).
"""

import gzip
import json
import os
import types
import unittest.mock as mock

from flliper.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.managers import corridor_guard as CG  # noqa: E402
from flliper.srt.mem_cache.pool_host import arena_mamba_pool as AMP  # noqa: E402
from flliper.srt.mem_cache.pool_host import arena_pool as AP  # noqa: E402
from flliper.srt.pdflip import budget_rest as BR  # noqa: E402
from flliper.srt.pdflip import form as F  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "admit_peak_1024",
                   "stage_lines_D.json.gz")
MIB = 1 << 20
INT4AR_ROW = 39223296          # int4ar mamba state row per rank (PDFLIP-ARENA-STATE-LOAD own_bytes / slots)
DUO_ROW = 53932032             # NVFP4 duo TP1, the largest row of the three forms


def _fixture():
    with gzip.open(FIX, "rt") as fh:
        return json.load(fh)


def _int4ar_parts():
    return [q for b, t in sorted(_fixture().items()) if b.startswith("dkr27bint4ar") for q in BR.admit_peak_parts(t)]


def _env(**kw):
    clean = {k: v for k, v in os.environ.items()
             if k not in (AP.ARENA_PAGE_LOAD_BLOCK_ENV, AP.ENV_ADMIT_PEAK_STAGES, AMP.ENV_STATE_LOAD_BLOCK_BYTES,
                          AMP.ENV_STATE_SPLIT_VEC, "FLLIPER_PDFLIP_ARENA_PAGE_LOAD_MODE")}
    clean.update(kw)
    return mock.patch.dict(os.environ, clean, clear=True)


class TestRootIsTheTwoStages(CustomTestCase):
    def test_every_record_boot_ran_the_256_mib_stages(self):
        fx = _fixture()
        for boot in BR.BOOTS_ADMIT_NEED:
            b = BR.admit_stage_basis(fx[boot])
            self.assertEqual(set(b.kv_stage_mib), {256.0}, boot)
            self.assertEqual(set(b.state_blocks), {6}, boot)
            self.assertLessEqual(max(b.mamba_stage_mib), 6 * INT4AR_ROW / MIB + 1e-6, boot)
        self.assertEqual(dict(BR.ADMIT_STAGE_BASIS), {"kv_page_stage_mib": 256, "mamba_state_block_rows": 6})
        # the mamba stage at the sleeps: 6 rows on int4ar, 10 x 18.7 on drei TP1/2, 233.8 on duo
        self.assertAlmostEqual(max(BR.admit_stage_basis(fx["dkr27bint4arrc4bar1int4ar10241254"]).mamba_stage_mib),
                               224.4, places=1)
        self.assertAlmostEqual(max(BR.admit_stage_basis(fx["dkr27bnvfp4duorebarbar1duo10241008"]).mamba_stage_mib),
                               233.8, places=1)

    def test_the_forward_trim_returns_the_kv_page_stage(self):
        big = [q for q in _int4ar_parts() if q.kv_stage_mib >= 255.9]
        self.assertEqual(len(big), 62)
        self.assertGreaterEqual(min(q.released for q in big), 256.0)
        self.assertEqual(sum(276 <= q.released <= 284 for q in big), 54)

    def test_a_minus_released_after_a_wake_is_the_live_mamba_stage(self):
        aft = [q for q in _int4ar_parts() if q.after_sleep and q.mamba_stage_mib > 0]
        self.assertEqual(len(aft), 51)
        self.assertLessEqual(max(q.admission - q.released - q.mamba_stage_mib for q in aft), 8.0)
        six = [q for q in aft if abs(q.mamba_stage_mib - 6 * INT4AR_ROW / MIB) < 0.1]
        self.assertTrue(six)
        self.assertGreaterEqual(max(q.admission - q.released for q in six), 6 * INT4AR_ROW / MIB)

    def test_the_peak_is_the_two_stages_plus_small_blocks(self):
        parts = _int4ar_parts()
        a_max = max(q.admission for q in parts)
        self.assertEqual(a_max, 510.0)
        self.assertGreaterEqual(a_max, 256.0 + 6 * INT4AR_ROW / MIB)
        self.assertLessEqual(a_max, sum(v for _, v in BR.ADMIT_PEAK_PARTS))
        self.assertAlmostEqual(sum(v for _, v in BR.ADMIT_PEAK_PARTS), 522.0, places=6)  # the record's A_peak (duo)


class TestKvPageStage(CustomTestCase):
    def test_device_modes_take_the_64_mib_stage(self):
        with _env():
            for mode in (None, "kernel", "dma"):
                self.assertEqual(AP._arena_page_load_block(32768, mode), 2048)
                self.assertEqual(AP._arena_page_load_block(0, mode), 2048)
                b = AP._arena_page_load_block(786432, mode)
                self.assertEqual(b, 85)
                self.assertLessEqual(b * 786432, 64 * MIB)
            self.assertEqual(AP._arena_page_load_block(32768) * 32768, AP._PAGE_LOAD_DEVICE_STAGE_BYTES)
            self.assertEqual(AP._PAGE_LOAD_DEVICE_STAGE_BYTES, 64 * MIB)

    def test_cpu_mode_keeps_the_256_mib_pinned_stage(self):
        with _env():
            self.assertEqual(AP._arena_page_load_block(32768, "cpu"), 8192)
            self.assertEqual(AP._arena_page_load_block(786432, "cpu"), 341)

    def test_switch_off_and_explicit_block(self):
        with _env(**{AP.ENV_ADMIT_PEAK_STAGES: "0"}):
            self.assertEqual(AP._arena_page_load_block(32768, "kernel"), 8192)
            self.assertEqual(AP._arena_page_load_block(786432, "dma"), 341)
        with _env(**{AP.ARENA_PAGE_LOAD_BLOCK_ENV: "4096"}):
            self.assertEqual(AP._arena_page_load_block(32768, "kernel"), 4096)
            self.assertEqual(AP._arena_page_load_block(32768, "cpu"), 4096)

    def test_the_page_load_sizes_its_stage_for_the_mode_it_runs(self):
        seen = []

        class _Stop(Exception):
            pass

        def spy(pb, mode=None):
            seen.append((pb, mode))
            raise _Stop()

        p = object.__new__(AP.ArenaMHAHostPool)
        p._page_bytes = 32768
        p._page_mode = None
        dev = types.SimpleNamespace(k_buffer=[torch.zeros(1)])
        with _env(), mock.patch.object(AP, "_arena_page_load_block", spy):
            with self.assertRaises(_Stop):
                p._load_pages_all_layers(dev, torch.tensor([0]), torch.tensor([0]))
        with _env(FLLIPER_PDFLIP_ARENA_PAGE_LOAD_MODE="dma"), mock.patch.object(AP, "_arena_page_load_block", spy):
            with self.assertRaises(_Stop):
                p._load_pages_all_layers(dev, torch.tensor([0]), torch.tensor([0]))
        # CPU box: "kernel" falls back to "cpu" (as the load itself does); an explicit dma stays dma
        self.assertEqual(seen, [(32768, "cpu"), (32768, "dma")])


# ---- mamba state stage --------------------------------------------------------------------------------------------

L_, A_, T_SHAPE, WIDTH, CONV_SHAPE, E = 3, 8, (2, 4), 3, (6, 3), 2
T_ROW = 8 * E
SPLIT = (2, 3, 1)


def _mamba_pool(page_bytes=None, foreign=True):
    """A synthetic arena slot: per layer [foreign][temporal][foreign][conv q][conv k][foreign][conv v] -- the foreign
    bytes are other ranks' extents; the loader must land only this rank's."""
    t_ext, c_ext, off = [], [], 0
    for _l in range(L_):
        off += 5 if foreign else 0
        t_ext.append((off, T_ROW)); off += T_ROW
        off += 3 if foreign else 0
        segs = []
        for j, n_j in enumerate(SPLIT):
            if j == 2:
                off += 2 if foreign else 0
            segs.append((off, n_j * WIDTH * E)); off += n_j * WIDTH * E
        c_ext.append(segs)
    slot_bytes = off
    g = torch.Generator().manual_seed(217)
    blob = torch.randint(0, 255, (A_, slot_bytes), dtype=torch.uint8, generator=g)
    pool = AMP.ArenaMambaPoolHost.__new__(AMP.ArenaMambaPoolHost)
    pool._slot_view = blob
    pool._page_bytes = slot_bytes if page_bytes is None else page_bytes
    pool._state_stage = None
    pool.temporal_dtype = torch.bfloat16
    pool.conv_dtype = torch.bfloat16
    pool._layout = {"L": L_, "t_shape": T_SHAPE, "conv_shape": CONV_SHAPE, "width": WIDTH, "t_ext": t_ext,
                    "c_ext": c_ext}
    return pool, blob, t_ext, c_ext


def _dev(rows=16, stacked=True):
    if stacked:
        t = torch.zeros((L_, rows) + T_SHAPE, dtype=torch.bfloat16)
        c = torch.zeros((L_, rows) + CONV_SHAPE, dtype=torch.bfloat16)
    else:
        t = [torch.zeros((rows,) + T_SHAPE, dtype=torch.bfloat16) for _ in range(L_)]
        c = [torch.zeros((rows,) + CONV_SHAPE, dtype=torch.bfloat16) for _ in range(L_)]
    return types.SimpleNamespace(mamba_cache=types.SimpleNamespace(temporal=t, conv=[c]))


def _bytes(t):
    return t.contiguous().view(torch.uint8)


def _assert_landed(tc, dp, blob, t_ext, c_ext, slots, didx):
    for l in range(L_):
        for s, d in zip(slots.tolist(), didx.tolist()):
            off, ln = t_ext[l]
            tc.assertTrue(torch.equal(_bytes(dp.mamba_cache.temporal[l][d]).reshape(-1), blob[s, off:off + ln]))
            ch0 = 0
            for (off_j, ln_j) in c_ext[l]:
                n_j = ln_j // (WIDTH * E)
                tc.assertTrue(torch.equal(_bytes(dp.mamba_cache.conv[0][l][d, ch0:ch0 + n_j]).reshape(-1),
                                          blob[s, off_j:off_j + ln_j]))
                ch0 += n_j


class TestMambaStateStage(CustomTestCase):
    def test_the_dma_block_is_one_row(self):
        with _env():
            self.assertEqual(AMP._state_load_block_rows(INT4AR_ROW, dma=True), 1)
            self.assertEqual(AMP._state_load_block_rows(DUO_ROW, dma=True), 1)
            self.assertEqual(AMP._state_load_block_rows(INT4AR_ROW, dma=False), 6)   # cpu: the pinned 256-MiB stage
        with _env(**{AP.ENV_ADMIT_PEAK_STAGES: "0"}):
            self.assertEqual(AMP._state_load_block_rows(INT4AR_ROW, dma=True), 6)    # the record's geometry
        with _env(**{AMP.ENV_STATE_LOAD_BLOCK_BYTES: str(128 * MIB)}):
            self.assertEqual(AMP._state_load_block_rows(INT4AR_ROW, dma=True), 3)    # explicit bytes: old meaning

    def test_the_vectorised_split_lands_the_loops_bytes(self):
        slots = torch.tensor([5, 1, 7, 2]); didx = torch.tensor([3, 9, 0, 12])
        with _env():
            pool, blob, t_ext, c_ext = _mamba_pool()
            vec = _dev()
            pool._load_states_all_layers(vec, slots, didx)
            self.assertIsNotNone(pool._state_split_vec(vec, pool._compact[0], pool._compact[1]))
        with _env(**{AMP.ENV_STATE_SPLIT_VEC: "0"}):
            pool2, _, _, _ = _mamba_pool()
            loop = _dev()
            pool2._load_states_all_layers(loop, slots, didx)
        _assert_landed(self, vec, blob, t_ext, c_ext, slots, didx)
        self.assertTrue(torch.equal(_bytes(vec.mamba_cache.temporal), _bytes(loop.mamba_cache.temporal)))
        self.assertTrue(torch.equal(_bytes(vec.mamba_cache.conv[0]), _bytes(loop.mamba_cache.conv[0])))
        untouched = [r for r in range(16) if r not in didx.tolist()]
        self.assertFalse(_bytes(vec.mamba_cache.temporal[:, untouched]).any())
        self.assertFalse(_bytes(vec.mamba_cache.conv[0][:, untouched]).any())

    def test_dma_mode_reuses_one_row_and_lands_every_slot(self):
        slots = torch.tensor([6, 0, 3, 4, 2]); didx = torch.tensor([1, 14, 7, 2, 11])
        with _env():
            pool, blob, t_ext, c_ext = _mamba_pool(page_bytes=1 << 20)   # > 32 KiB: "dma" (the 78-MiB blob's mode)
            dp = _dev()
            pool._load_states_all_layers(dp, slots, didx)
        self.assertEqual(tuple(pool._state_dev_stage.shape), (1, pool._compact[1]))
        _assert_landed(self, dp, blob, t_ext, c_ext, slots, didx)

    def test_unknown_shapes_keep_the_per_layer_loop(self):
        slots = torch.tensor([5, 1]); didx = torch.tensor([3, 9])
        with _env():
            pool, blob, t_ext, c_ext = _mamba_pool()
            listed = _dev(stacked=False)                       # per-layer lists: no (L, S, ...) tensor
            pool._load_states_all_layers(listed, slots, didx)
            self.assertIsNone(pool._state_split_vec(listed, pool._compact[0], pool._compact[1]))
            _assert_landed(self, listed, blob, t_ext, c_ext, slots, didx)
            comp, row = pool._compact
            stacked = _dev()
            bad = list(comp)
            bad[4] = (bad[4][0], bad[4][1], bad[4][2] + 2)     # layer 1's temporal extent longer than layer 0's
            self.assertIsNone(pool._state_split_vec(stacked, bad, row))
            wrong = _dev()
            wrong.mamba_cache.temporal = wrong.mamba_cache.temporal.to(torch.float16)  # another dtype
            self.assertIsNone(pool._state_split_vec(wrong, comp, row))
            self.assertIsNone(pool._state_split_vec(stacked, comp, row + 1))


# ---- the record -----------------------------------------------------------------------------------------------------

def _line_page(block, mode="kernel", pages=20000, page=32768):
    return (f"[2026-10-25 01:00:00 TP1] PDFLIP-ARENA-PAGE-LOAD n=1 rows={pages} pages={pages} block={block} "
            f"bytes={pages * page} mode={mode} (whole pages, layers split on device)")


def _line_state(block, mode="dma"):
    return (f"[2026-10-25 01:00:00 TP1] PDFLIP-ARENA-STATE-LOAD n=1 slots=4 own_bytes={4 * INT4AR_ROW} block={block} "
            f"mode={mode} stage_bytes={INT4AR_ROW} split=vec (this rank's extents, 192 copies per slot)")


class TestRecordStaysMeasured(CustomTestCase):
    def test_the_record_stays_414_until_a_217_census(self):
        rec = F.PROFILES[F.PROFILE_QWEN27B].constants[BR.ADMIT_RECORD]
        self.assertEqual(list(rec.value), [414, 414, 414])
        fx = _fixture()
        ok, lines = BR.admit_record_may_replace([(b, BR.admit_stage_basis(fx[b])) for b in BR.BOOTS_ADMIT_NEED])
        self.assertFalse(ok)
        self.assertIn("NO -- the record stays", lines[-1])
        self.assertIn("#217", rec.provenance)
        self.assertIn("HOCHRECHNUNG", rec.provenance)

    def test_only_an_all_217_census_may_replace(self):
        new = BR.admit_stage_basis("\n".join([_line_page(2048), _line_state(1)]))
        old = BR.admit_stage_basis("\n".join([_line_page(8192), _line_state(6)]))
        mixed_kv = BR.admit_stage_basis("\n".join([_line_page(2048), _line_page(8192), _line_state(1)]))
        mixed_st = BR.admit_stage_basis("\n".join([_line_page(2048), _line_state(1), _line_state(6)]))
        no_state = BR.admit_stage_basis(_line_page(2048))
        nothing = BR.admit_stage_basis("")
        self.assertTrue(BR.admit_record_may_replace([("a", new), ("b", new)])[0])
        for bad in (old, mixed_kv, mixed_st, no_state, nothing):
            self.assertFalse(BR.admit_record_may_replace([("a", new), ("b", bad)])[0])
        self.assertFalse(BR.admit_record_may_replace([])[0])
        # an owner-lane line (NF) is not the whole-page stage and never counts as one
        lane = BR.admit_stage_basis(_line_page(2048).replace("mode=kernel", "mode=dma lane_dma=2d"))
        self.assertEqual(lane.kv_stage_mib, ())

    def test_the_projection_is_a_hochrechnung_and_never_booked(self):
        a, need, label = BR.admit_peak_projection()
        self.assertAlmostEqual(a, 147.6, places=6)
        self.assertEqual(need, 40)
        self.assertIn("HOCHRECHNUNG", label)
        rec = F.PROFILES[F.PROFILE_QWEN27B].constants[BR.ADMIT_RECORD]
        self.assertNotEqual(int(rec.value[0]), need)
        booked = BR.admit_deltas([2079, 2079], [0, 0], [float(rec.value[0])] * 2)
        self.assertEqual(booked, [416, 416])
        # the 32-MiB instant of the record boots is held by the booked post, not by the projection
        self.assertGreaterEqual(32 + booked[0], CG.NEAR_OOM_MIB)
        self.assertLess(32 + -(-need // 8) * 8, CG.NEAR_OOM_MIB)
