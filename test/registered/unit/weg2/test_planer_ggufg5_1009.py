"""NF-GGUF AP G5 (2026-10-09): planner / launcher per LAYER, profile nf-gguf, catalog edges, golden plan_nf_gguf_n3.

Plan deskq/PLAN-GGUF-NF-1009.md section 0 (7, 9, 10) and section 2 row G5.  Parts

* ``TestLayerRows``        the per-layer expert-row carrier (``planner/expert_layer_rows.py``): the mean as its float value (an
                           unchanged reader keeps its number), the layer vector for a reader that prices a stage SLICE; equal
                           layers never get a vector (every INT4/AWQ/FP8 checkpoint: the exact scalar of before).  The numbers
                           of ``deskq/gguf-nf/calc.py`` (PP0 +144, PP1 +71, PP2 -350 MiB against the x177 budgets) are recomputed.
* ``TestStageModels``      the fraction solve, the pool model, the seat form and the draft post per stage slice; the scalar path
                           is the pre-G5 formula bit for bit.
* ``TestTermsFromHeaders`` ``pp_cut.checkpoint_weight_terms`` of a GGUF: per-layer vector, routed experts counted from the STACKED
                           tensors, census WITHOUT a sibling config.json; a safetensors checkpoint gets no vector.
* ``TestDraftPostGguf``    a GGUF draft file is priced from its header.
* ``TestProfile``          ``tools/release/profconv/nf-gguf.env``: the GGUF changes of nf.env, one by one; the committed launch JSON.
* ``TestCatalog``          the edges K132/K133 (curated text for ``--tokenizer-path`` waits for the union catalog rebuild, G7).
* ``TestDryRunGolden``     the launcher dry run of nf-gguf on the reference rig (RTX 5090 + 2x RTX 3080) from the committed header
                           snapshots == ``golden/nf/plan_nf_gguf_n3.txt`` (0 diff lines); box-bound like every dry-run golden of
                           this directory (it carries the census / evidence files of this box and SKIPS, with the reason, where
                           they are absent).

GPU-free, NVML-free, Docker-free.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import pathlib
import struct
import tempfile
import unittest

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    from sglang.srt.planner import expert_layer_rows as ELR
    from sglang.srt.planner import expert_residency as ER
    from sglang.srt.planner import pp_cut
    from sglang.srt.weg2 import draft_post as DP
    from sglang.srt.weg2 import launcher
    from sglang.srt.weg2 import propose_oracle as O
except Exception as exc:  # pragma: no cover - no weg2 launcher in this build
    pytest.skip(f"weg2 launcher unavailable: {exc}", allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = str(pathlib.Path(launcher.__file__).resolve().parents[4])
FIX = os.path.join(HERE, "fixtures", "planer_1006")
GOLDEN = os.path.join(FIX, "golden")
CKPT = os.path.join(FIX, "checkpoints")
REPLAY_REF = os.path.join(HERE, "fixtures", "xchg_launch_replay_0911", "nvml_devices_1378.json")
PROFILE = os.path.join(TREE, "tools", "release", "profconv", "nf-gguf.env")
SIBLING_SNAPSHOT = os.path.join(CKPT, "Qwen3.8-Flash-Next-GGUF-unsloth-sibling")
MTP_SNAPSHOT = os.path.join(CKPT, "MTP")
MIB = float(1 << 20)

#: the three expert row classes of the unsloth UD-IQ4_XS header, bytes per expert and layer (deskq/gguf-nf/calc.py)
CLASS_A, CLASS_B, CLASS_C = 2329600, 3148800, 3481600
LAYERS_B = (4, 30, 46, 47)
LAYER_C = 2
N_EXPERTS = 512


def _real_vector():
    """Expert bytes of EACH of the 48 layers of the real header (x 512 experts)."""
    return [float((CLASS_C if i == LAYER_C else CLASS_B if i in LAYERS_B else CLASS_A) * N_EXPERTS) for i in range(48)]


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ---------------------------------------------------------------------------
# the carrier
# ---------------------------------------------------------------------------

class TestLayerRows(unittest.TestCase):
    def test_the_float_value_is_the_mean_and_the_vector_rides_along(self):
        r = ELR.LayerRows([1.0, 1.0, 1.0, 5.0])
        self.assertIsInstance(r, float)
        self.assertEqual(float(r), 2.0)
        self.assertEqual(r.per_layer, (1.0, 1.0, 1.0, 5.0))
        self.assertEqual(r * 2, 4.0)                       # arithmetic of an unchanged reader: the mean, a plain float
        self.assertNotIsInstance(r * 2, ELR.LayerRows)
        self.assertEqual(float(ELR.LayerRows([1.0, 3.0], mean=7.0)), 7.0)   # an explicit mean (the terms' own) wins

    def test_equal_layers_never_get_a_vector(self):
        got = ELR.rows_from_layer_bytes([4.0, 4.0, 4.0])
        self.assertIs(type(got), float)
        self.assertEqual(got, (4.0 + 4.0 + 4.0) / 3)
        self.assertIs(type(ELR.rows_from_layer_bytes([4.0, 4.0], scale=0.5)), float)
        self.assertTrue(ELR.is_layer_vector(ELR.rows_from_layer_bytes([4.0, 5.0])))
        self.assertEqual(ELR.rows_from_layer_bytes([]), 0.0)

    def test_scalar_stage_total_is_the_old_product_exactly(self):
        row = 2.3097 * 1.1
        for n in (29, 11, 8):
            self.assertEqual(ELR.stage_total(row, (29, 11, 8), (29, 11, 8).index(n)), n * row)
        self.assertEqual(ELR.stage_mean(row, (29, 11, 8), 1), row)

    def test_the_slice_of_a_contiguous_cut(self):
        r = ELR.LayerRows([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(ELR.stage_slice((2, 1, 3), 0), (0, 2))
        self.assertEqual(ELR.stage_slice((2, 1, 3), 2), (3, 6))
        self.assertEqual([ELR.stage_total(r, (2, 1, 3), s) for s in range(3)], [3.0, 3.0, 15.0])
        self.assertEqual(ELR.stage_means(r, (2, 1, 3)), (1.5, 3.0, 5.0))
        # a cut that names more layers than the vector holds falls back to the mean, never to an IndexError
        self.assertEqual(ELR.stage_total(r, (4, 4), 1), 4 * float(r))

    def test_slot_of_keeps_the_vector_and_the_mean(self):
        r = ELR.LayerRows([512.0, 1024.0])
        s = ELR.slot_of(r, 512)
        self.assertIsInstance(s, ELR.LayerRows)
        self.assertEqual(s.per_layer, (1.0, 2.0))
        self.assertEqual(float(s), 768.0 / 512)
        self.assertEqual(ELR.slot_of(1024.0, 512), 2.0)
        self.assertIs(type(ELR.slot_of(1024.0, 512)), float)

    def test_the_plan_numbers_are_recomputed_from_the_header_classes(self):
        """calc.py (the Ist-Bericht, section 4): the Mittelwert error per P stage at the x177 budgets is +144 / +71 / -350 MiB."""
        vec = ELR.LayerRows(_real_vector())
        self.assertEqual(sorted(set(vec.per_layer)), [CLASS_A * 512.0, CLASS_B * 512.0, CLASS_C * 512.0])
        self.assertEqual(sum(1 for v in vec.per_layer if v == CLASS_A * 512.0), 43)
        self.assertAlmostEqual(float(vec) / MIB, 1182.5520833, places=4)
        cut = (29, 11, 8)
        want = (144, 71, -350)
        for s, (gib, w) in enumerate(zip((13.83, 9.35, 7.70), want)):
            tot = ELR.stage_total(vec, cut, s)
            err = (cut[s] * float(vec) - tot) * (gib * 1024 * MIB) / tot / MIB
            self.assertEqual(round(err), w, (s, err))
        # the same facts without the budgets: PP2 (layers 40..47, two class-B layers) holds MORE expert bytes than 8 mean layers
        self.assertGreater(ELR.stage_total(vec, cut, 2), 8 * float(vec))
        self.assertLess(ELR.stage_total(vec, cut, 0), 29 * float(vec))


# ---------------------------------------------------------------------------
# stage models
# ---------------------------------------------------------------------------

BUDGET = (28240.0, 17680.0, 17168.0)
COUNTS, ATTN = (29, 11, 8), (7, 3, 2)


class TestStageModels(unittest.TestCase):
    def test_the_fraction_solve_scalar_path_is_the_pre_g5_formula(self):
        slot = 2.31
        got = ER.solve_stage_fraction_by_buffer_rule(
            budgets_mib=list(BUDGET), stage_layers=list(COUNTS), dense_layer_mib=82.0, slot_mib=slot,
            num_experts=512, scratch_rows=[32, 32, 32])
        for s, (b, n) in enumerate(zip(BUDGET, COUNTS)):
            max_rows = int(math.floor((b - n * 82.0) / (n * slot)))
            self.assertEqual(got[s], ER.largest_fraction_for_rows(local_experts=512, scratch_rows=32, max_rows=max_rows))

    def test_the_fraction_solve_prices_the_slice_a_stage_holds(self):
        per_mib = ELR.LayerRows([v / MIB for v in _real_vector()])
        slot = ELR.slot_of(per_mib, 512)
        tight = [14000.0, 9000.0, 5200.0]                   # below the (E-2)/E cap on every stage, so the price decides
        kw = dict(budgets_mib=tight, stage_layers=list(COUNTS), dense_layer_mib=82.0, num_experts=512, scratch_rows=[32, 32, 32])
        mean_fr = ER.solve_stage_fraction_by_buffer_rule(slot_mib=float(slot), **kw)
        slice_fr = ER.solve_stage_fraction_by_buffer_rule(slot_mib=slot, **kw)
        self.assertGreater(slice_fr[0], mean_fr[0])        # PP0 holds class-A layers only: cheaper than the mean says
        self.assertLess(slice_fr[2], mean_fr[2])           # PP2 holds two class-B layers: dearer
        self.assertLess(max(slice_fr + mean_fr), 0.99)

    def _model(self, **kw):
        return pp_cut.PhasePoolModel(
            free_mib=BUDGET, weight_mib_per_layer=61.6, weight_mib_per_layer_by_stage=(539.87, 927.21, 1047.11),
            kv_mib_per_token_per_attn_layer=1024.0 / MIB, arming_floor_mib=(1229.0,) * 3,
            stage_fixed_mib=(1584.0, 527.1, 1065.4), activation_reserve_mib=1024.0, corridor_holdback_mib=0.0,
            mamba_mib_per_linear_layer_per_slot=1.5602, mamba_slots=32, **kw)

    def test_the_pool_model_without_a_vector_is_the_scalar_model(self):
        m = self._model()
        for s, n in enumerate(COUNTS):
            self.assertEqual(m.stage_weight_mib(s, n, sum(COUNTS[:s])), m.layer_mib(s) * n)
        free = pp_cut._stage_free_after_residency(COUNTS, ATTN, m)
        self.assertEqual(free, pp_cut._stage_free_after_residency(COUNTS, ATTN, self._model(expert_layer_mib_by_layer=())))

    def test_the_pool_model_corrects_each_stage_by_its_slice(self):
        vec = tuple(v / MIB for v in _real_vector())
        mean = sum(vec) / len(vec)
        fac = (0.336, 0.65, 0.726)                          # resident fraction + LRU rows / experts per stage
        plain = self._model()
        m = self._model(expert_layer_mib_by_layer=vec, expert_resident_factor_by_stage=fac, expert_layer_mib_mean=mean)
        free_plain = pp_cut._stage_free_after_residency(COUNTS, ATTN, plain)
        free_vec = pp_cut._stage_free_after_residency(COUNTS, ATTN, m)
        pos = 0
        for s, n in enumerate(COUNTS):
            slice_mib = sum(vec[pos:pos + n])
            self.assertAlmostEqual(free_plain[s] - free_vec[s], fac[s] * (slice_mib - n * mean), places=6)
            pos += n
        self.assertLess(free_vec[2], free_plain[2])         # PP2 pays MORE than the mean price said (the -350 MiB)
        self.assertGreater(free_vec[0], free_plain[0])
        # a vector without its factors, or without a start (a swing slab): the scalar price, never a half model
        self.assertEqual(self._model(expert_layer_mib_by_layer=vec).stage_weight_mib(0, 29, 0), plain.layer_mib(0) * 29)
        self.assertEqual(m.stage_weight_mib(0, 29, None), plain.layer_mib(0) * 29)

    def test_the_seat_form_carries_unequal_rows_only_when_they_are_unequal(self):
        text = {"linear_num_value_heads": 48, "linear_value_head_dim": 128, "linear_key_head_dim": 128,
                "layer_types": ["linear_attention"] * 3 + ["full_attention"], "moe_intermediate_size": 640, "hidden_size": 2560}
        packed = 3 * 640 * 2560 // 2
        int4_row = packed + 76800
        a = ER.seat_vram_form(text, ssm_dtype="bfloat16", rank_tp_ratio="1,0,0", n_ranks=3, expert_row_bytes=int4_row,
                              moe_layers=4)
        b = ER.seat_vram_form(text, ssm_dtype="bfloat16", rank_tp_ratio="1,0,0", n_ranks=3, expert_row_bytes=int4_row,
                              moe_layers=4, expert_row_bytes_by_layer=[float(int4_row)] * 4)
        self.assertEqual(a, b)                              # equal layers: the old form, field for field
        self.assertEqual(a.row_bytes_by_layer, ())
        self.assertEqual(a.small_row_bytes, 76800)
        rows = [CLASS_A, CLASS_A, CLASS_C, CLASS_A]
        c = ER.seat_vram_form(text, ssm_dtype="bfloat16", rank_tp_ratio="1,0,0", n_ranks=3,
                              expert_row_bytes=sum(rows) / 4, moe_layers=4, expert_row_bytes_by_layer=[float(x) for x in rows])
        self.assertEqual(c.row_bytes_by_layer, tuple(rows))
        self.assertEqual(c.small_row_bytes, 0)              # a GGUF row has no scale tensor: the INT4 split does not apply

    def test_the_draft_post_prices_the_last_stage_slice(self):
        """The rows the draft's freed MiB buy on the LAST P stage: ``layers x row`` of ITS slice (``raise_for_draft_post``
        hands ``stage_mean`` of the row vector to ``expert_rows_for``); a scalar row is unchanged."""
        per_mib = ELR.LayerRows([v / MIB for v in _real_vector()])
        row = ELR.slot_of(per_mib, 512)
        freed = 3334.0
        rows_mean, _ = DP.expert_rows_for(freed, 8, float(row), 512, 0.39)
        rows_slice, _ = DP.expert_rows_for(freed, 8, ELR.stage_mean(row, COUNTS, 2), 512, 0.39)
        self.assertEqual(rows_mean, 180)
        self.assertEqual(rows_slice, 172)                   # the golden's "+172 rows (172 per layer x 8 layers x 2.417 MiB)"
        self.assertEqual(ELR.stage_mean(2.31, COUNTS, 2), 2.31)

    def test_raise_for_draft_post_uses_the_slice_row(self):
        import gguf
        import numpy as np

        per_mib = ELR.LayerRows([v / MIB for v in _real_vector()])
        row = ELR.slot_of(per_mib, 512)

        class _Card:
            nvml_index = 2

        with tempfile.TemporaryDirectory(prefix="g5-rfd-") as td:
            path = os.path.join(td, "draft.gguf")
            w = gguf.GGUFWriter(path, "qwen4exp")
            w.add_block_count(1)
            w.add_tensor("blk.0.nextn.eh_proj.weight", np.zeros((int(3200 * MIB) // 4,), dtype=np.float32))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()
            new, post, why = DP.raise_for_draft_post(fracs=[0.324, 0.637, 0.39], stage_layers=list(COUNTS), row_mib=row,
                                                     num_experts=512, draft_path=path, cards=[_Card()] * 3)
        self.assertEqual(why, "")
        self.assertAlmostEqual(post.row_mib, ELR.stage_mean(row, COUNTS, 2), places=9)
        self.assertEqual(post.layers, 8)
        self.assertEqual(new[:2], [0.324, 0.637])

    def test_the_launcher_helper_returns_plain_floats_for_equal_layers(self):
        class _T:
            expert_layer_weight_bytes = 1239995733.3333333
            num_experts = 512
            expert_layer_weight_bytes_by_layer = tuple([1239995733.3333333] * 48)

        row, layer = launcher.p_expert_rows_for_stage_models(_T, pp_cut)
        self.assertIs(type(row), float)
        self.assertIs(type(layer), float)
        self.assertEqual(row, (_T.expert_layer_weight_bytes / max(1, _T.num_experts)) / pp_cut.MIB)
        self.assertEqual(layer, _T.expert_layer_weight_bytes / pp_cut.MIB)

        class _G(_T):
            expert_layer_weight_bytes = sum(_real_vector()) / 48
            expert_layer_weight_bytes_by_layer = tuple(_real_vector())

        row, layer = launcher.p_expert_rows_for_stage_models(_G, pp_cut)
        self.assertIsInstance(layer, ELR.LayerRows)
        self.assertAlmostEqual(float(layer), _G.expert_layer_weight_bytes / pp_cut.MIB, places=9)
        self.assertEqual(len(row.per_layer), 48)
        self.assertAlmostEqual(float(row), float(layer) / 512, places=9)


# ---------------------------------------------------------------------------
# terms from headers
# ---------------------------------------------------------------------------

def _load_g1():
    """The tiny qwen4exp GGUF builder of the G1 test (same branch): all 47 roles, Q8_0 everywhere."""
    path = os.path.join(HERE, "..", "model_loader", "test_gguf_qwen4exp_g1_1009.py")
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location("g5_g1_builder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G1 = _load_g1()


def _clear_header_caches():
    from sglang.srt.model_loader import gguf_shards
    from sglang.srt.weg2 import gguf_census, host_ledger

    getattr(host_ledger, "_GGUF_HEADER_FACTS", {}).clear()
    gguf_shards._RESOLVED_CACHE.clear()
    gguf_census._CENSUS_CACHE.clear()


@unittest.skipIf(G1 is None, "the G1 test module (tiny qwen4exp GGUF builder) is not in this tree")
class TestTermsFromHeaders(unittest.TestCase):
    def setUp(self):
        _clear_header_caches()
        self.td = tempfile.TemporaryDirectory(prefix="g5-terms-")
        self.addCleanup(self.td.cleanup)
        self.addCleanup(_clear_header_caches)

    def _tiny(self, *, config=True, unequal=False):
        import numpy as np

        d = os.path.join(self.td.name, "cfg" if config else "nocfg")
        os.makedirs(d)
        path = os.path.join(d, "m-00001-of-00001.gguf")
        if unequal:
            name = "blk.2.ffn_down_exps.weight"
            G1.build_tiny(path, drop=(name,),
                          extra=((name, np.zeros((G1.N_EXP, G1.H, G1.FF), dtype=np.float32)),))
        else:
            G1.build_tiny(path)
        if config:
            G1._write_cfg(d)
        return path

    def test_equal_layers_get_the_scalar_and_a_constant_vector(self):
        path = self._tiny()
        terms = pp_cut.checkpoint_weight_terms(path)
        self.assertEqual(terms.n_layers, G1.N_LAYERS)
        self.assertEqual(terms.num_experts, G1.N_EXP)       # counted from the STACKED ffn_*_exps tensors
        vec = terms.expert_layer_weight_bytes_by_layer
        self.assertEqual(len(vec), G1.N_LAYERS)
        self.assertEqual(len(set(vec)), 1)
        self.assertIs(type(pp_cut.expert_layer_rows(terms)), float)
        self.assertEqual(pp_cut.expert_layer_rows(terms), terms.expert_layer_weight_bytes)

    def test_unequal_layers_get_the_vector_and_the_mean(self):
        path = self._tiny(unequal=True)
        terms = pp_cut.checkpoint_weight_terms(path)
        vec = terms.expert_layer_weight_bytes_by_layer
        self.assertEqual(len(vec), G1.N_LAYERS)
        self.assertGreater(vec[2], vec[1])
        self.assertEqual(vec[0], vec[1])
        rows = pp_cut.expert_layer_rows(terms)
        self.assertIsInstance(rows, ELR.LayerRows)
        self.assertEqual(rows.per_layer, tuple(vec))
        self.assertEqual(float(rows), terms.expert_layer_weight_bytes)
        self.assertAlmostEqual(float(rows), sum(vec) / len(vec), places=6)

    def test_the_census_needs_no_sibling_config(self):
        """The unsloth export is .gguf parts and NO config.json: the family map is built from the header's own geometry."""
        from sglang.srt.weg2 import gguf_census

        path = self._tiny(config=False)
        c = gguf_census.gguf_tensor_census(path)
        self.assertEqual((c.family, c.arch, c.backbone_depth), ("qwen4exp", "qwen4exp", G1.N_LAYERS))
        self.assertEqual(c.expert_count(), G1.N_EXP)
        terms = pp_cut.checkpoint_weight_terms(path)
        self.assertEqual((terms.n_layers, terms.num_experts), (G1.N_LAYERS, G1.N_EXP))

    def test_a_sibling_config_still_wins_and_a_foreign_arch_without_one_is_named(self):
        from sglang.srt.weg2 import gguf_census

        path = self._tiny()
        c = gguf_census.gguf_tensor_census(path)
        self.assertEqual(c.family, "qwen4exp")
        with self.assertRaises(gguf_census.GgufCensusUnavailable):
            gguf_census._header_text_config(os.path.join(self.td.name, "does-not-exist.gguf"))

    def test_a_safetensors_checkpoint_keeps_its_scalar(self):
        d = os.path.join(self.td.name, "st")
        os.makedirs(d)
        hdr, off = {}, 0
        for layer in range(3):
            for e in range(2):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    n = 1000
                    hdr["model.layers.%d.mlp.experts.%d.%s.weight" % (layer, e, proj)] = {
                        "dtype": "U8", "shape": [n], "data_offsets": [off, off + n]}
                    off += n
            hdr["model.layers.%d.self_attn.q_proj.weight" % layer] = {"dtype": "U8", "shape": [10], "data_offsets": [off, off + 10]}
            off += 10
        raw = json.dumps(hdr).encode()
        with open(os.path.join(d, "model.safetensors"), "wb") as fh:
            fh.write(struct.pack("<Q", len(raw)) + raw)
            fh.truncate(8 + len(raw) + off)
        terms = pp_cut.checkpoint_weight_terms(d)
        self.assertEqual((terms.n_layers, terms.num_experts), (3, 2))
        self.assertEqual(terms.expert_layer_weight_bytes, 6000.0)
        self.assertEqual(terms.expert_layer_weight_bytes_by_layer, (6000.0, 6000.0, 6000.0))
        self.assertIs(type(pp_cut.expert_layer_rows(terms)), float)


_REAL_FILES = os.path.isfile(os.path.join(SIBLING_SNAPSHOT, "manifest.json"))


@unittest.skipUnless(_REAL_FILES, "header snapshot of the unsloth UD-IQ4_XS parts is not committed")
class TestRealHeaderSnapshot(unittest.TestCase):
    """The REAL unsloth Qwen3.8-Flash-Next UD-IQ4_XS header (3 parts, 1224 tensors), rebuilt from the committed snapshot as a
    sparse stub WITHOUT its config.json: the census, the per-layer vector and the planner terms read it."""

    @classmethod
    def setUpClass(cls):
        _clear_header_caches()
        cls.td = tempfile.TemporaryDirectory(prefix="g5-real-")
        stub = O.materialize_checkpoint(SIBLING_SNAPSHOT, cls.td.name)
        os.remove(os.path.join(stub, "config.json"))
        cls.model = os.path.join(stub, "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf")
        cls.terms = pp_cut.checkpoint_weight_terms(cls.model)

    @classmethod
    def tearDownClass(cls):
        _clear_header_caches()
        cls.td.cleanup()

    def test_the_planner_sees_48_layers_and_512_experts(self):
        t = self.terms
        self.assertEqual((t.n_layers, t.num_experts), (48, 512))
        self.assertEqual(t.attention_layer_indices, (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47))

    def test_the_vector_is_the_three_row_classes_of_calc_py(self):
        vec = self.terms.expert_layer_weight_bytes_by_layer
        self.assertEqual(vec, tuple(_real_vector()))
        self.assertEqual({v for v in vec}, {CLASS_A * 512.0, CLASS_B * 512.0, CLASS_C * 512.0})
        self.assertAlmostEqual(self.terms.expert_layer_weight_bytes / MIB, 1182.5520833, places=4)
        rows = pp_cut.expert_layer_rows(self.terms)
        self.assertIsInstance(rows, ELR.LayerRows)
        # 110,864 MiB over 48 layers per expert (Ist-Bericht 4) against 116,016 MiB for INT4
        self.assertAlmostEqual(sum(v / 512 for v in vec) / MIB, 110.864, places=3)

    def test_the_fit_profile_comes_from_the_header_not_from_the_config_formulas(self):
        """``hw_fit.derive_profile_from_header``: per-layer expert MiB of the three classes (all 512 experts of a layer), the
        GGUF format, the PLE table as disk; the planner's own ``fit_profile_from_model`` (the model profile ``estimate``) reads
        the same per-layer values: the fit bound and ``propose()`` agree on weight_format gguf."""
        from sglang.srt.weg2 import hw_fit as HF
        from sglang.srt.weg2 import model_profile as MP
        from sglang.srt.weg2 import propose_rules as R

        p = HF.derive_profile_from_header(self.model, profile="nextflash")
        self.assertEqual((p.weight_format, p.n_layers, p.attn_layers, p.n_experts, p.is_moe), ("gguf", 48, 12, 512, True))
        self.assertEqual(sorted(set(p.layer_expert_mib)), [1137.5, 1537.5, 1700.0])
        self.assertEqual(sum(1 for v in p.layer_expert_mib if v == 1137.5), 43)
        self.assertEqual(tuple(round(v / MIB, 4) for v in _real_vector()), p.layer_expert_mib)
        self.assertGreater(p.disk_mib, 20000.0)
        self.assertEqual(p.draft_mib, 0.0)
        # the planner's model profile: the same bytes, and the format named gguf
        q = R.fit_profile_from_model(MP.estimate(self.model, allow_config_only=False))
        self.assertEqual(q.weight_format, "gguf")
        self.assertEqual(q.layer_expert_mib, p.layer_expert_mib)
        self.assertEqual(q.layer_dense_mib, p.layer_dense_mib)

    def test_the_ple_table_stays_off_the_device_and_the_mtp_block_is_not_a_layer(self):
        self.assertGreater(self.terms.ple_layer_weight_bytes, 0.0)
        self.assertEqual(self.terms.replicated_breakdown, {})   # the unsloth parts carry no MTP block and no vision tensors


# ---------------------------------------------------------------------------
# draft post
# ---------------------------------------------------------------------------

class TestDraftPostGguf(unittest.TestCase):
    def test_a_tiny_gguf_draft_is_priced_from_its_header_with_the_hf_exclusions(self):
        import gguf
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="g5-draft-") as td:
            path = os.path.join(td, "draft.gguf")
            w = gguf.GGUFWriter(path, "qwen4exp")
            w.add_block_count(1)
            for name, n in (("token_embd.weight", 64), ("output.weight", 32), ("blk.0.nextn.eh_proj.weight", 16)):
                w.add_tensor(name, np.zeros((n,), dtype=np.float32))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()
            allb = (64 + 32 + 16) * 4 / MIB
            self.assertAlmostEqual(DP.checkpoint_tensor_mib(path), allb, places=9)
            self.assertAlmostEqual(DP.checkpoint_tensor_mib(path, exclude=("lm_head",)), (64 + 16) * 4 / MIB, places=9)
            self.assertAlmostEqual(DP.checkpoint_tensor_mib(path, exclude=("embed_tokens", "lm_head")), 16 * 4 / MIB, places=9)
            w2, tr = DP.p_draft_post_mib(path)
            self.assertAlmostEqual(w2, (64 + 16) * 4 / MIB + DP.DRAFT_RUNNER_BUFFER_MIB, places=9)
            self.assertAlmostEqual(DP.d_draft_host_mib(path, share_embed=True), 16 * 4 / MIB + DP.DRAFT_RUNNER_BUFFER_MIB, places=9)
            # a GGUF whose header cannot be read is ABSENT (None), never 0
            bad = os.path.join(td, "bad.gguf")
            with open(bad, "wb") as fh:
                fh.write(b"not a gguf")
            self.assertIsNone(DP.checkpoint_tensor_mib(bad))

    @unittest.skipUnless(os.path.isfile(os.path.join(MTP_SNAPSHOT, "manifest.json")), "MTP header snapshot not committed")
    def test_the_shared_mtp_file_of_the_unsloth_export(self):
        """mtp-...-shared-Q8_0.gguf: 32 tensors, 2,585 GiB (Ist-Bericht 3); it carries neither embedding nor head (shared with
        the target), so the P and D exclusions cost it nothing."""
        with tempfile.TemporaryDirectory(prefix="g5-mtp-") as td:
            stub = O.materialize_checkpoint(MTP_SNAPSHOT, td)
            f = os.path.join(stub, "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf")
            mib = DP.checkpoint_tensor_mib(f)
            self.assertAlmostEqual(mib, 2647.0390625, places=6)
            self.assertEqual(DP.checkpoint_tensor_mib(f, exclude=("embed_tokens", "lm_head")), mib)
            self.assertAlmostEqual(DP.p_draft_post_mib(f)[0], mib + DP.DRAFT_RUNNER_BUFFER_MIB, places=6)


# ---------------------------------------------------------------------------
# the profile
# ---------------------------------------------------------------------------

def _argv_value(argv, flag):
    i = len(argv) - 1 - argv[::-1].index(flag)
    return argv[i + 1]


class TestProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.li = O.profile_launch_input(PROFILE, asset_dirs=())

    def test_no_placeholder_and_the_status_is_experimental(self):
        v = self.li.vars
        self.assertEqual(v["PROFILE_NAME"], "nf-gguf")
        self.assertEqual(v["PROFILE_FORMAT"], "gguf")
        self.assertEqual(v["PROFILE_PLACEHOLDER"], "0")
        self.assertEqual(v["PROFILE_STATUS"], "experimentell")
        self.assertEqual(v["PROFILE_LINE"], "nf")
        self.assertNotIn("PLATZHALTER", _read(PROFILE).split("PROFILE_NAME=")[0].split("\n", 2)[1])

    def test_the_model_is_part_one_of_the_gguf_set_in_the_sibling_directory(self):
        a = self.li.argv
        model = _argv_value(a, "--model")
        self.assertTrue(model.endswith("Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"), model)
        self.assertEqual(self.li.model, model)
        self.assertEqual(_argv_value(a, "--tokenizer-path"), os.path.dirname(model))
        self.assertEqual(self.li.vars["PROFILE_SIBLING"], os.path.dirname(model))
        self.assertEqual(a.count("--model"), 1)

    def test_vision_is_off_and_the_census_is_foreign(self):
        a = self.li.argv
        self.assertEqual(_argv_value(a, "--pdflip-vision"), "off")
        self.assertEqual(a.count("--pdflip-vision"), 1)
        self.assertIn("--pdflip-xchg-census-foreign", a)
        self.assertEqual(a.index("--pdflip-xchg-census-foreign"), a.index("--pdflip-xchg-census") + 2)

    def test_the_draft_is_the_shared_mtp_gguf_everywhere_and_the_int4_draft_is_gone(self):
        a = self.li.argv
        draft = self.li.draft
        self.assertTrue(draft.endswith("MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"), draft)
        for flag in ("--extra-p", "--extra-d"):
            extra = _argv_value(a, flag)
            self.assertIn("--speculative-draft-model-path " + draft, extra)
            self.assertNotIn("albucino", extra)
            self.assertNotIn(".safetensors", extra)
        self.assertNotIn("albucino", " ".join(a))

    def test_store_dir_is_its_own_and_hc_mixer_int8_is_off_exactly_once(self):
        a = self.li.argv
        for flag in ("--env-p", "--env-d"):
            e = _argv_value(a, flag)
            self.assertIn("SGLANG_MOE_EXPERT_STORE_DIR=/mnt/nf-experts/gguf", e.replace("FLLIPER_", "SGLANG_"))
            self.assertNotIn("fnFL2", e)
        self.assertEqual(self.li.env["FLLIPER_HC_MIXER_INT8"], "0")
        self.assertEqual(sum(1 for k in self.li.env if k.endswith("HC_MIXER_INT8")), 1)

    def test_the_rest_of_the_form_is_nf_env(self):
        nf = O.profile_launch_input(os.path.join(os.path.dirname(PROFILE), "nf.env"), asset_dirs=())
        want = {"--pp-stage-ratio", "--pp-attn-stage-ratio", "--pp-cut-expert-device-fraction", "--pp-cut-expert-lru-rows",
                "--rank-moe-ratio"}
        for f in ("--pp-stage-ratio", "--pp-attn-stage-ratio", "--pp-cut-expert-device-fraction", "--pp-cut-expert-lru-rows"):
            self.assertEqual(_argv_value(self.li.argv, f), _argv_value(nf.argv, f), f)
        self.assertIn("--rank-moe-ratio 183,137,168", _argv_value(self.li.argv, "--extra-d"))
        self.assertEqual(want & {"--pp-stage-ratio"}, {"--pp-stage-ratio"})
        # the HW-GENERIC lines of nf.env (Preflight-Gate aus NVML-Eigenschaften)
        for k in ("PROFILE_CARD_COUNT", "PROFILE_INVENTORY", "PROFILE_GPUS", "PROFILE_LINE"):
            self.assertEqual(self.li.vars[k], nf.vars[k], k)
        # nothing that does not exist in the tree is named
        self.assertNotIn("--moe-act-int8", " ".join(self.li.argv))
        # P eager until G6 is proven at the metal (nf.env already says it)
        self.assertIn("--disable-cuda-graph", _argv_value(self.li.argv, "--extra-p"))

    def test_the_required_paths_name_the_gguf_files_not_safetensors(self):
        raw = _read(PROFILE)
        self.assertIn('PROFILE_REQUIRED_PATHS=(', raw)
        block = raw.split("PROFILE_REQUIRED_PATHS=(")[1].split(")")[0]
        for need in ('"$PROFILE_MODEL"', '"$PROFILE_TOKENIZER/config.json"', '"$PROFILE_TOKENIZER/tokenizer.json"', '"$PROFILE_DRAFT"'):
            self.assertIn(need, block)
        self.assertNotIn("safetensors", block)

    def test_the_committed_launch_json_is_this_profile(self):
        """Profile snapshot: argv and environment as the entrypoint hands them to the launcher (``propose_oracle launch``)."""
        want = json.loads(_read(os.path.join(GOLDEN, "nf", "launch_nf-gguf.json")))
        got = O.launch_input_doc(O.profile_launch_input(PROFILE, asset_dirs=()))
        self.assertEqual(got, want, "regenerate: python -m sglang.srt.weg2.propose_oracle launch --profile %s --out ..." % PROFILE)

    def test_the_sibling_script_builds_the_directory_the_profile_names(self):
        import subprocess

        script = os.path.join(TREE, "tools", "release", "make_nf_gguf_sibling.sh")
        self.assertTrue(os.access(script, os.X_OK))
        with tempfile.TemporaryDirectory(prefix="g5-sib-") as td:
            parts, orig, dest = os.path.join(td, "parts"), os.path.join(td, "orig"), os.path.join(td, "out")
            os.makedirs(parts)
            os.makedirs(orig)
            for i in (1, 2, 3):
                open(os.path.join(parts, "m-0000%d-of-00003.gguf" % i), "wb").close()
            for n in ("config.json", "tokenizer.json", "tokenizer_config.json"):
                with open(os.path.join(orig, n), "w") as fh:
                    fh.write("{}")
            r = subprocess.run(["bash", script, dest], capture_output=True, text=True,
                               env={**os.environ, "NF_GGUF_PARTS_DIR": parts, "NF_GGUF_CONFIG_SRC": orig})
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(sorted(os.listdir(dest)), ["config.json", "m-00001-of-00003.gguf", "m-00002-of-00003.gguf",
                                                       "m-00003-of-00003.gguf", "tokenizer.json", "tokenizer_config.json"])
            self.assertTrue(os.path.islink(os.path.join(dest, "m-00001-of-00003.gguf")))
            r2 = subprocess.run(["bash", script, os.path.join(td, "o2")], capture_output=True, text=True,
                                env={**os.environ, "NF_GGUF_PARTS_DIR": os.path.join(td, "nope"), "NF_GGUF_CONFIG_SRC": orig})
            self.assertNotEqual(r2.returncode, 0)


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------

class TestCatalog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        weg2 = os.path.join(TREE, "python", "sglang", "srt", "weg2")
        with open(os.path.join(weg2, "kantenkatalog_1004.json"), encoding="utf-8") as fh:
            cls.edges = {k["id"]: k for k in json.load(fh)["kanten"]}
        spec = importlib.util.spec_from_file_location("g5_curated", os.path.join(weg2, "profile_catalog_curated.py"))
        cls.cu = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cu)
        spec = importlib.util.spec_from_file_location("g5_catalog", os.path.join(weg2, "profile_catalog.py"))
        cls.pc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.pc)

    def test_gguf_needs_a_tokenizer_path_and_vision_off(self):
        for eid, nach in (("K132", "--tokenizer-path"), ("K133", "--weg2-vision")):
            k = self.edges[eid]
            self.assertEqual((k["von"], k["wert"], k["nach"], k["rel"]), ("--load-format", "gguf", nach, "braucht"))
            self.assertTrue(k["beleg"]["datei"].startswith("python/sglang/srt/weg2/launcher.py"))
        self.assertIn("W163", self.edges["K132"]["satz"])

    def test_the_anchors_resolve_in_this_tree(self):
        res = self.pc.resolve_edge_belege([self.edges["K132"], self.edges["K133"]], TREE, "nf")
        for eid in ("K132", "K133"):
            self.assertIn(res[eid]["status"], self.pc.ANKER_OK, res[eid])

    def test_no_edge_names_a_flag_the_tree_does_not_have(self):
        """--moe-act-int8 (H88) is not in this base: no edge may name it (the catalog test would fail, and an anchor cannot exist)."""
        self.assertFalse([k for k in self.edges.values() if "moe-act-int8" in (k["von"], k["nach"])])

    def test_the_curated_core_is_untouched_by_g5(self):
        """The shipped catalog.json carries the curated count (119): a new curated entry would need the union rebuild over BOTH
        code trees (the integration step G7 does it); G5 adds edges only."""
        self.assertNotIn("--tokenizer-path", self.cu.CURATED)


# ---------------------------------------------------------------------------
# the dry run
# ---------------------------------------------------------------------------

def _snapshots():
    out = {}
    for n in sorted(os.listdir(CKPT)):
        if os.path.isfile(os.path.join(CKPT, n, "manifest.json")):
            out[O.read_snapshot_manifest(os.path.join(CKPT, n))["name"]] = os.path.join(CKPT, n)
    return out


def _inputs_present():
    try:
        li = O.profile_launch_input(PROFILE)
    except Exception:
        return False
    return not li.unresolved_paths and os.path.isfile(os.path.join(GOLDEN, "nf", "plan_nf_gguf_n3.txt"))


_LINE = O.launcher_line()


@unittest.skipUnless(_inputs_present(), "the census / evidence files the nf profile names are not on this box")
@unittest.skipUnless(_LINE == O.LINE_NF, "the golden is of the NF launcher line (golden/nf/)")
class TestDryRunGolden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prun = O.run_profile(PROFILE, O.read_replay(REPLAY_REF), tree=TREE, snapshots=_snapshots())
        cls.dump = cls.prun.result.dump()

    def test_the_dump_equals_the_golden(self):
        res = self.prun.result
        self.assertIsNone(res.exc_type, "%s: %s" % (res.exc_type, res.exc_msg[:300]))
        self.assertEqual(res.rc, 0)
        want = _read(os.path.join(GOLDEN, "nf", "plan_nf_gguf_n3.txt"))
        d = O.diff_lines(want, self.dump)
        self.assertEqual(d, [], "plan diff vs plan_nf_gguf_n3.txt: %d lines\n%s" % (len(d), "\n".join(x[:240] for x in d[:12])))
        self.assertEqual(self.prun.result.forced, [])

    def test_the_stub_stood_in_for_model_tokenizer_and_draft_and_nothing_else_was_changed(self):
        notes = " | ".join(self.prun.notes)
        self.assertIn("model:", notes)
        self.assertIn("tokenizer:", notes)
        self.assertIn("draft:", notes)

    def test_the_planner_prices_the_cut_per_layer(self):
        lines = [ln for ln in self.dump.splitlines() if "PP-CUT EXPERT ROWS JE LAYER (G5)" in ln]
        self.assertEqual(len(lines), 1, lines)
        ln = lines[0]
        self.assertIn("1137.5000 x43", ln)
        self.assertIn("1537.5000 x4", ln)
        self.assertIn("1700.0000 x1", ln)
        self.assertIn("Stufen-Schnitt [29, 11, 8]", ln)
        # the stage 2 slice is dearer than the mean price (the -350 MiB), stage 0 cheaper
        import re

        per = [int(x) for x in re.search(r"Experten je Stufe \[([^\]]*)\]", ln).group(1).replace("'", "").split(",")]
        mean = [int(x) for x in re.search(r"Mittelwert-Preis \[([^\]]*)\]", ln).group(1).replace("'", "").split(",")]
        self.assertGreater(per[2], mean[2])
        self.assertLess(per[0], mean[0])

    def test_vision_is_off_and_the_draft_is_the_gguf_file(self):
        self.assertIn("vision=off", self.dump)
        self.assertNotIn("vision=transient", self.dump)
        self.assertIn("mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", self.dump)
        self.assertIn("--no-enable-multimodal", self.dump)

    def test_no_reader_looked_for_a_config_json_inside_the_gguf_file(self):
        self.assertNotIn("NotADirectoryError", self.dump)
        self.assertNotIn("unreadable (", self.dump.split("WEG2-FORM arch", 1)[1].split("\n", 1)[0])

    def test_the_golden_provenance_names_the_inputs(self):
        side = json.loads(_read(os.path.join(GOLDEN, "nf", "plan_nf_gguf_n3.provenance.json")))
        self.assertEqual(side["golden"], "plan_nf_gguf_n3.txt")
        self.assertEqual(side["profile"]["sha256"], _sha(PROFILE))
        self.assertEqual(side["profile_base"]["sha256"], _sha(os.path.join(os.path.dirname(PROFILE), "nf.env")))
        snaps = _snapshots()
        for name in side["checkpoints"]:
            self.assertIn(name, snaps)


class TestGivenFlagWords(unittest.TestCase):
    """Review 1 (G5 fix round 1): the canonical spelling of the "flag given" words is GGUF-only (R1)."""

    OTHER = "--pdflip-vision"

    def _ns(self, model, vision="off"):
        import argparse

        return argparse.Namespace(profile="nextflash", model=model, weg2_vision=vision, teardown=False)

    def _other_spelling(self):
        from sglang.srt.compat_shims import canonical_flags

        w = canonical_flags([self.OTHER, "off"])
        return w[0] != self.OTHER   # this tree maps --pdflip-* onto --weg2-*; else the case does not exist

    def test_gguf_launch_reads_the_other_spelling_as_given(self):
        if not self._other_spelling():
            self.skipTest("running tree spells the flag --pdflip-*")
        with tempfile.TemporaryDirectory() as td:
            gg = os.path.join(td, "x-00001-of-00003.gguf")   # check_gguf_file wants the FILE (suffix)
            open(gg, "wb").close()
            words = launcher.given_flag_words(self._ns(gg), [self.OTHER, "off"])
            ns = self._ns(gg)
            line = launcher.apply_profile_vision_default(ns, words)
        self.assertTrue(launcher.weg2_form.flag_given(words, "--weg2-vision"))
        self.assertIsNone(line)
        self.assertEqual(ns.weg2_vision, "off")

    def test_non_gguf_launch_keeps_the_raw_words_and_the_registry_default(self):
        # nf-nvfp4-d (NVFP4/Marlin): byte-identical to the pre-G5 base, the row's transient default still applies
        argv = [self.OTHER, "off", "--profile", "nextflash"]
        for model in ("/m/Qwen3.8-NVFP4", "", "/m/dir-with.gguf.d"):
            words = launcher.given_flag_words(self._ns(model), argv)
            self.assertEqual(words, argv)
        ns = self._ns("/m/Qwen3.8-NVFP4")
        line = launcher.apply_profile_vision_default(ns, launcher.given_flag_words(ns, argv))
        self.assertTrue(line.startswith(launcher.VISION_DEFAULT_MARKER))
        self.assertEqual(ns.weg2_vision, "transient")


if __name__ == "__main__":
    unittest.main()
