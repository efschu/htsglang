"""AP-B Modellprofil-Lücken (Planer-Workflow 06.10.2026): was ``propose()`` (Plan §1.2 K1-K4) vom Modellprofil ``flliper.model/1`` braucht
und vorher fehlte.  Je Lücke ein Test (Lücken-Tabelle: ``deskq/done/planer-apB-luecken-1006.md``):

* L6  KV-Gleitfenster (``kv.sliding``): der DFlash2-Draft (2048 / 5 Layer) wird nicht mit vollem Kontext gerechnet;
* L7  Decode-Bytes je Token (``decode``): dicht = alle Layer + lm_head, MoE = Nicht-Experten + top_k/n der Experten + lm_head;
* L9  DFlash2-Draft als eigenes Profil (Art, ``dflash_config``, Draft-KV-Zelle, Einbettung/lm_head/Rest) -- echter Tensorkopf als Fixture;
* L10 NEXTN-Draft: EIGENE Layerzahl (1 MTP-Layer, nicht die 48 des Ziels aus der kopierten config.json);
* L11 Draft-Abzüge gleich ``draft_post.checkpoint_tensor_mib`` (P ohne lm_head, D ohne embed_tokens + lm_head);
* L13 GGUF-Kopf als Profilabschnitt ``gguf``;
* L14 GGUF ohne config.json: Geometrie aus den Kopfschlüsseln, gegen die echten Configs (27B, NF) geprüft;
* L15 geteilter GGUF-Satz, fehlender Teil wird benannt verweigert;
* L16 ``probe`` / ``estimate_or_state``: "nicht gemountet" und Verwandte als Zustand.
"""

import gzip
import json
import math
import os
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import model_profile as MP  # noqa: E402
from test_profil_s3_modell_1003 import DT, TestSyntheticMoE, census_dir, with_config, write_shard  # noqa: E402

FX = os.path.join(HERE, "fixtures", "planer_apb_1006")
FX_S3 = os.path.join(HERE, "fixtures", "profil_s3_1003")
MC = "/spinning/llm_stuff/club-3090/models-cache/"
MIB = float(1 << 20)


def dflash2_dir(tmp, layers=5):
    """Der echte DFlash2-Draft: config.json und das Tensorverzeichnis (81 Tensoren, Namen/Dtype/Formen) des echten Kopfes."""
    d = os.path.join(tmp, "dflash2")
    os.makedirs(d)
    with open(os.path.join(FX, "dflash2_draft", "config.json")) as src, open(os.path.join(d, "config.json"), "w") as dst:
        dst.write(src.read())
    with gzip.open(os.path.join(FX, "dflash2_draft", "tensors.json.gz"), "rt") as fh:
        rows = json.load(fh)["tensors"]
    write_shard(os.path.join(d, "model.safetensors"), {r[0]: (r[1], r[2]) for r in rows})
    return d, rows


def gguf_file(path, kv, tensors):
    """GGUF-Datei NUR mit Kopf: ``kv`` = {Schlüssel: (Werttyp, Wert)} (4 u32, 6 f32, 8 string, 9 u32-Feld), ``tensors`` = [(Name, Dims, ggml-Typ)]."""

    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kv))
    for k, (vt, v) in kv.items():
        out += s(k) + struct.pack("<I", vt)
        if vt == 4:
            out += struct.pack("<I", v)
        elif vt == 6:
            out += struct.pack("<f", v)
        elif vt == 8:
            out += s(v)
        elif vt == 9:
            out += struct.pack("<I", 4) + struct.pack("<Q", len(v)) + b"".join(struct.pack("<I", x) for x in v)
        else:
            raise ValueError(vt)
    for name, dims, gt in tensors:
        out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<I", gt) + struct.pack("<Q", 0)
    with open(path, "wb") as fh:
        fh.write(out)


def qwen35_kv(block_count=9, nextn=1, file_type=30):
    """Ein kleines qwen35-GDN-Hybridmodell (Geometrie klein, Schlüssel wie der echte 27B-Kopf)."""
    a = "qwen35."
    kv = {"general.architecture": (8, "qwen35"), "general.file_type": (4, file_type),
          a + "block_count": (4, block_count), a + "context_length": (4, 4096), a + "embedding_length": (4, 256),
          a + "feed_forward_length": (4, 512), a + "attention.head_count": (4, 4), a + "attention.head_count_kv": (4, 2),
          a + "attention.key_length": (4, 64), a + "attention.value_length": (4, 64), a + "full_attention_interval": (4, 4),
          a + "ssm.conv_kernel": (4, 4), a + "ssm.state_size": (4, 16), a + "ssm.group_count": (4, 2), a + "ssm.time_step_rank": (4, 4),
          a + "ssm.inner_size": (4, 64), a + "rope.freq_base": (6, 10000000.0), a + "rope.dimension_count": (4, 16),
          a + "rope.dimension_sections": (9, [11, 11, 10, 0])}
    if nextn:
        kv[a + "nextn_predict_layers"] = (4, nextn)
    return kv


def qwen35_tensors(n_layers=8, with_mtp=True):
    t = [("token_embd.weight", (256, 512), 30), ("output.weight", (256, 512), 30)]
    for i in range(n_layers):
        if (i + 1) % 4 == 0:
            t.append(("blk.%d.attn_q.weight" % i, (256, 2 * 4 * 64), 30))      # Ausgabe-Gate: doppelt so breit
            t.append(("blk.%d.attn_k.weight" % i, (256, 2 * 64), 30))
        else:
            t.append(("blk.%d.attn_qkv.weight" % i, (256, 512), 30))
            t.append(("blk.%d.ssm_a" % i, (256,), 0))
        t.append(("blk.%d.ffn_up.weight" % i, (256, 512), 30))
    if with_mtp:
        t.append(("blk.%d.attn_q.weight" % n_layers, (256, 2 * 4 * 64), 30))
    return t


class TestKvSliding(unittest.TestCase):
    """L6: das Gleitfenster steht im Profil, damit der Draft-KV nicht mit vollem Kontext gerechnet wird."""

    def test_dflash2_draft_has_window_2048_on_5_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, _ = dflash2_dir(tmp)
            est = MP.estimate(d)
        sl = est["kv"]["sliding"]
        self.assertEqual((sl["window_tokens"]["v"], sl["layers"]["v"]), (2048, 5))
        self.assertEqual((sl["window_tokens"]["src"], sl["layers"]["src"]), ("config", "config"))
        self.assertEqual(est["kv"]["attn_layers"]["v"], 5)

    def test_full_attention_models_have_no_sliding_entry(self):
        for name in ("qwen27b_int8_vocabembed", "nextflash_nvfp4"):
            with tempfile.TemporaryDirectory() as tmp:
                est = MP.estimate(with_config(tmp, name))
            self.assertNotIn("sliding", est["kv"], name)

    def test_window_without_explicit_layer_types_is_not_guessed(self):
        self.assertIsNone(MP.sliding_info({"sliding_window": 4096, "num_hidden_layers": 4}))
        self.assertIsNone(MP.sliding_info({"layer_types": ["sliding_attention"], "sliding_window": None}))
        self.assertEqual(MP.sliding_info({"layer_types": ["sliding_attention", "full_attention", "sliding_attention"], "sliding_window": 512}), (512, 2))


class TestDecodeBytes(unittest.TestCase):
    """L7: Gewichtsbytes, die EIN Decode-Token liest."""

    def test_dense_27b_is_every_layer_plus_lm_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, _ = census_dir(tmp)
            est = MP.estimate(d)
        w, dec = est["weights"], est["decode"]
        self.assertEqual(dec["bytes_per_token"]["v"], w["layers_bytes_nonexpert"]["v"] + w["lm_head_bytes"]["v"])
        self.assertEqual(dec["experts_active_bytes"]["v"], 0)
        self.assertEqual(dec["bytes_per_token"]["src"], "Index")
        self.assertGreater(dec["bytes_per_token"]["v"], 0.9 * w["total_bytes"]["v"] * 0.8)      # der Großteil des Modells wird je Token gelesen

    def test_moe_reads_top_k_of_n_experts(self):
        with tempfile.TemporaryDirectory() as tmp:
            TestSyntheticMoE().build(tmp)
            est = MP.estimate(tmp)
        w, dec = est["weights"], est["decode"]
        self.assertEqual(dec["experts_active_bytes"]["v"], w["layers_bytes_expert"]["v"] * 2 // 4)        # top_k 2 von 4
        self.assertEqual(dec["bytes_per_token"]["v"],
                         w["layers_bytes_nonexpert"]["v"] + w["layers_bytes_expert"]["v"] * 2 // 4 + w["lm_head_bytes"]["v"])

    def test_config_only_moe_is_an_estimate_and_scales_with_top_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            est = MP.estimate(with_config(tmp, "nextflash_nvfp4"))
        w, dec = est["weights"], est["decode"]
        self.assertEqual(dec["bytes_per_token"]["src"], "geschätzt")
        self.assertAlmostEqual(dec["experts_active_bytes"]["v"] / w["layers_bytes_expert"]["v"], 10 / 512, places=5)
        self.assertLess(dec["bytes_per_token"]["v"], w["total_bytes"]["v"] / 8)

    def test_tied_embedding_is_read_as_lm_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"num_hidden_layers": 1, "hidden_size": 16, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
                   "intermediate_size": 32, "vocab_size": 64, "tie_word_embeddings": True, "dtype": "bfloat16"}
            with open(os.path.join(tmp, "config.json"), "w") as fh:
                json.dump(cfg, fh)
            write_shard(os.path.join(tmp, "m.safetensors"), {
                "model.language_model.embed_tokens.weight": ("BF16", (64, 16)),
                "model.language_model.layers.0.self_attn.q_proj.weight": ("BF16", (16, 16)),
                "model.language_model.layers.0.mlp.up_proj.weight": ("BF16", (32, 16))})
            est = MP.estimate(tmp)
        self.assertEqual(est["decode"]["lm_head_bytes"]["v"], 64 * 16 * 2)


class TestDraftProfile(unittest.TestCase):
    """L9/L10/L11: der getrennte Draft als eigenes Profil."""

    def test_dflash2_draft_from_the_real_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, rows = dflash2_dir(tmp)
            dr = MP.estimate_draft(d)
            full = MP.estimate(d)
        total = sum(int(MP.SAFETENSORS_DTYPE_BYTES[r[1]] * math.prod(r[2])) for r in rows)
        self.assertEqual(dr["kind"]["v"], "dflash2")
        self.assertEqual(dr["n_layers"]["v"], 5)
        self.assertEqual(dr["total_bytes"]["v"], total)
        self.assertEqual(total, 3848808960)                          # Kopfsumme des echten Verzeichnisses (Probe 06.10.2026)
        self.assertEqual((dr["embed_bytes"]["v"], dr["lm_head_bytes"]["v"]), (0, 0))   # der Draft teilt Einbettung/lm_head mit dem Ziel
        self.assertEqual(dr["bytes_without_lm_head"]["v"], total)
        kv = dr["kv"]
        self.assertEqual((kv["attn_layers"]["v"], kv["kv_heads"]["v"], kv["head_dim"]["v"]), (5, 8, 128))
        self.assertEqual(kv["cell_bytes_per_attn_layer_token"]["auto"]["v"], 4096.0)
        self.assertEqual(kv["cell_bytes_per_attn_layer_token"]["fp8_e4m3"]["v"], 2176.0)
        self.assertEqual((kv["sliding"]["window_tokens"]["v"], kv["sliding"]["layers"]["v"]), (2048, 5))
        self.assertEqual(dr["dflash"]["target_layer_ids"]["v"], [5, 19, 33, 47, 61])
        self.assertEqual(dr["dflash"]["block_size"]["v"], 8)
        # das Verzeichnis ist auch als vollwertiges flliper.model/1 lesbar: gleiche Summe, 5 Layer
        self.assertEqual(full["weights"]["total_bytes"]["v"], total)
        self.assertEqual(full["arch"]["n_layers"]["v"], 5)

    def nextn_dir(self, tmp):
        """Verzeichnis wie ``albucino-mtp-int4-g32/runtime/mtp-int4-g32``: die config.json ist die des ZIELS (NF, 48 Layer), die
        Gewichte sind Einbettung + lm_head + EIN ``mtp.layers.0``."""
        d = os.path.join(tmp, "nextn")
        os.makedirs(d)
        with open(os.path.join(FX_S3, "nextflash_nvfp4", "config.json")) as src, open(os.path.join(d, "config.json"), "w") as dst:
            dst.write(src.read())
        t = {"lm_head.weight": ("BF16", (640, 128)), "model.language_model.embed_tokens.weight": ("BF16", (640, 128)),
             "mtp.fc_embedding.weight": ("BF16", (128, 128)), "mtp.layers.0.self_attn.q_proj.weight": ("BF16", (256, 128)),
             "mtp.layers.0.mlp.shared_expert.up_proj.weight": ("BF16", (32, 128))}
        write_shard(os.path.join(d, "mtp-dense.safetensors"), t)
        return d, t

    def test_nextn_draft_counts_its_own_layers_not_the_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, t = self.nextn_dir(tmp)
            dr = MP.estimate_draft(d)
        sz = {n: int(DT[dt] * math.prod(s)) for n, (dt, s) in t.items()}
        self.assertEqual(dr["kind"]["v"], "nextn")
        self.assertEqual(dr["n_layers"]["v"], 1)                    # NICHT 48 (num_hidden_layers des Ziels)
        self.assertEqual(dr["n_layers"]["src"], "Index")
        self.assertFalse(dr["own_backbone"]["v"])
        self.assertEqual(dr["mtp_layers_found"]["v"], 1)
        self.assertEqual(dr["embed_bytes"]["v"], sz["model.language_model.embed_tokens.weight"])
        self.assertEqual(dr["lm_head_bytes"]["v"], sz["lm_head.weight"])
        self.assertEqual(dr["total_bytes"]["v"], sum(sz.values()))
        self.assertEqual(dr["kv"]["attn_layers"]["v"], 1)
        self.assertEqual(dr["kv"]["cell_bytes_per_attn_layer_token"]["fp8_e4m3"]["v"], 1088.0)      # NF fp8: 2 KV-Köpfe x 256
        self.assertNotIn("sliding", dr["kv"])

    def test_draft_deductions_equal_the_launchers_draft_post(self):
        draft_post = __import__("sglang.srt.weg2.draft_post", fromlist=["x"])
        with tempfile.TemporaryDirectory() as tmp:
            d, _ = self.nextn_dir(tmp)
            dr = MP.estimate_draft(d)
            p_mib = draft_post.checkpoint_tensor_mib(d, exclude=draft_post.P_SHARED_WITH_TARGET)
            d_mib = draft_post.checkpoint_tensor_mib(d, exclude=draft_post.D_SHARED_WITH_TARGET)
        self.assertEqual(dr["bytes_without_lm_head"]["v"] / MIB, p_mib)
        self.assertEqual(dr["bytes_without_embed_lm_head"]["v"] / MIB, d_mib)
        self.assertLess(d_mib, p_mib)

    def test_unknown_kind_is_named_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as fh:
                json.dump({"architectures": ["SomethingElse"], "num_hidden_layers": 2, "hidden_size": 16, "num_attention_heads": 2,
                           "num_key_value_heads": 1, "head_dim": 8}, fh)
            write_shard(os.path.join(tmp, "m.safetensors"), {"layers.0.self_attn.q_proj.weight": ("BF16", (16, 16))})
            dr = MP.estimate_draft(tmp)
        self.assertEqual(dr["kind"]["v"], "unbekannt")
        self.assertEqual(dr["n_layers"]["v"], 2)

    def test_estimate_embeds_the_enriched_external_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, _ = dflash2_dir(tmp)
            with open(os.path.join(tmp, "t.json"), "w"):
                pass
            tgt = with_config(tmp, "qwen27b_int8_vocabembed")
            est = MP.estimate(tgt, draft_path=d)
        ext = est["draft"]["external"]
        self.assertEqual(ext["kind"]["v"], "dflash2")
        self.assertEqual(ext["kv"]["sliding"]["window_tokens"]["v"], 2048)

    @unittest.skipUnless(os.path.isfile(MC + "Qwen3.8-27B-DFlash2/model.safetensors"), "Rig-Draft nicht gemountet")
    def test_on_the_rig_dflash2_checkpoint(self):
        dr = MP.estimate_draft(MC + "Qwen3.8-27B-DFlash2")
        self.assertEqual(dr["total_bytes"]["v"], 3848808960)
        self.assertEqual(dr["kind"]["v"], "dflash2")


class TestProbe(unittest.TestCase):
    """L16: "nicht gemountet" und Verwandte als Zustand, nie als Ausnahme."""

    def test_missing_path_is_not_mounted(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = MP.probe(os.path.join(tmp, "gibt-es-nicht"))
        self.assertEqual((r["state"], r["estimable"]), ("not_mounted", False))
        self.assertIn("not mounted", r["reason"])

    def test_empty_directory_is_an_empty_mountpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = MP.probe(tmp)
        self.assertEqual((r["state"], r["estimable"]), ("empty", False))
        self.assertIn("mount point", r["reason"])

    def test_directory_with_only_subdirectories_names_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "UD-IQ4_XS"))
            os.makedirs(os.path.join(tmp, "MTP"))
            r = MP.probe(tmp)
        self.assertEqual(r["state"], "no_model_files")
        self.assertEqual(r["subdirs"], ["MTP", "UD-IQ4_XS"])
        self.assertIn("MTP", r["reason"])

    def test_config_only_index_only_complete_and_no_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = with_config(tmp, "qwen27b_int8_vocabembed")
            self.assertEqual(MP.probe(d)["state"], "config_only")
            with open(os.path.join(d, "model.safetensors.index.json"), "w") as fh:
                json.dump({"metadata": {"total_size": 1}, "weight_map": {}}, fh)
            self.assertEqual(MP.probe(d)["state"], "index_only")
            write_shard(os.path.join(d, "m.safetensors"), {"lm_head.weight": ("BF16", (2, 2))})
            r = MP.probe(d)
            self.assertEqual((r["state"], r["estimable"], r["safetensors"]), ("complete", True, 1))
            os.remove(os.path.join(d, "config.json"))
            r = MP.probe(d)
            self.assertEqual((r["state"], r["estimable"]), ("no_config", False))

    def test_estimate_or_state_returns_the_state_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = MP.estimate_or_state(tmp)
            self.assertFalse(r["ok"])
            self.assertEqual(r["state"], "empty")
            self.assertNotIn("profile", r)
            d = with_config(tmp, "qwen27b_int8_vocabembed")
            ok = MP.estimate_or_state(d)
        self.assertTrue(ok["ok"])
        self.assertEqual(ok["state"], "config_only")
        self.assertEqual(ok["profile"]["schema"], "flliper.model/1")
        self.assertEqual(ok["profile"]["weights"]["total_bytes"]["src"], "geschätzt")

    def test_unreadable_header_becomes_a_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = with_config(tmp, "qwen27b_int8_vocabembed")
            with open(os.path.join(d, "a.safetensors"), "wb") as fh:
                fh.write(struct.pack("<Q", 1000) + b"{}")
            r = MP.estimate_or_state(d)
        self.assertEqual((r["ok"], r["state"]), (False, "unreadable"))
        self.assertIn("truncated", r["reason"])


class TestGgufWithoutConfig(unittest.TestCase):
    """L13/L14: der GGUF-Kopf trägt die Geometrie, wenn keine config.json daneben liegt."""

    def build(self, tmp, **kw):
        path = os.path.join(tmp, "m.gguf")
        gguf_file(path, qwen35_kv(**kw), qwen35_tensors())
        return path

    def test_geometry_comes_from_the_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            est = MP.estimate(self.build(tmp))
        a = est["arch"]
        self.assertEqual(est["config_source"]["v"], "gguf")
        self.assertIsNone(est["config_path"])
        self.assertEqual((a["n_layers"]["v"], a["hidden"]["v"], a["heads_q"]["v"], a["heads_kv"]["v"], a["head_dim"]["v"], a["vocab"]["v"]),
                         (8, 256, 4, 2, 64, 512))
        self.assertEqual(a["layer_counts"]["v"], {"attn": 2, "gdn": 6, "mamba": 0})
        self.assertEqual(est["context"]["max_position_embeddings"]["v"], 4096)
        self.assertEqual(est["context"]["rope"]["mrope_section"]["v"], [11, 11, 10])
        self.assertEqual(est["context"]["rope"]["partial_rotary_factor"]["v"], 0.25)
        self.assertEqual(est["state"]["linear_layers"]["v"], 6)
        # GDN-Zustand aus ssm.*: 4 Wertköpfe x 16 x 16 Elemente, Kern 4
        self.assertEqual(est["state"]["ssm_bytes"]["v"], 4 * 16 * 16 * 4)
        self.assertEqual(est["state"]["conv_bytes"]["v"], (2 * 2 * 16 + 4 * 16) * 3 * 2)
        self.assertEqual(est["kv"]["variants"]["fp8_e4m3"]["cell_bytes_per_attn_layer_token"]["v"], 2 * (64 + 64) * 1 + 2 * 64 * 2 / 16)
        # Der Backbone (8) trennt den MTP-Block (Layer 8) ab
        self.assertEqual(est["draft"]["mtp_layers"]["v"], 1)
        self.assertGreater(est["weights"]["mtp_bytes"]["v"], 0)
        self.assertTrue(any("GGUF header" in w for w in est["warnings"]))

    def test_head_facts_are_a_profile_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            est = MP.estimate(self.build(tmp, file_type=30))
            est18 = MP.estimate(self.build(tmp, file_type=18))
            est99 = MP.estimate(self.build(tmp, file_type=99))
        g = est["gguf"]
        self.assertEqual((g["architecture"]["v"], g["block_count"]["v"], g["nextn_predict_layers"]["v"], g["file_type"]["v"]),
                         ("qwen35", 9, 1, "IQ4_XS"))
        self.assertEqual(est18["gguf"]["file_type"]["v"], "Q6_K")
        self.assertEqual(est99["gguf"]["file_type"]["v"], "ftype 99")             # unbekannter Wert: nie benannt geraten
        self.assertEqual(g["architecture"]["src"], "Index")

    def test_output_gate_is_read_from_the_q_projection_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            td, kv, _ = MP.scan_gguf_set(self.build(tmp))
        cfg, _ = MP.config_from_gguf(kv, td)
        self.assertTrue(cfg["text_config"]["attn_output_gate"])

    def test_non_qwen_ssm_is_flagged_not_mapped(self):
        kv = {"general.architecture": "foo", "foo.block_count": 4, "foo.embedding_length": 64, "foo.attention.head_count": 2,
              "foo.ssm.inner_size": 128, "foo.ssm.time_step_rank": 4}
        cfg, notes = MP.config_from_gguf(kv, MP.TensorDir("gguf", [], {}, 0))
        self.assertNotIn("linear_num_value_heads", cfg["text_config"])
        self.assertTrue(any("not mapped" in n for n in notes))

    def test_missing_architecture_or_depth_is_refused(self):
        td = MP.TensorDir("gguf", [], {}, 0)
        with self.assertRaises(MP.ModelProfileError):
            MP.config_from_gguf({}, td)
        with self.assertRaises(MP.ModelProfileError):
            MP.config_from_gguf({"general.architecture": "qwen35"}, td)

    def test_a_config_json_beside_the_file_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.build(tmp)
            with open(os.path.join(tmp, "config.json"), "w") as fh:
                json.dump({"num_hidden_layers": 8, "hidden_size": 256, "full_attention_interval": 4, "num_attention_heads": 4,
                           "num_key_value_heads": 2, "head_dim": 64}, fh)
            est = MP.estimate(os.path.join(tmp, "m.gguf"))
        self.assertNotIn("config_source", est)
        self.assertIsNotNone(est["config_path"])

    @unittest.skipUnless(os.path.isfile(MC + "Qwen3.8-27B-GGUF-unsloth/Qwen3.8-27B-UD-IQ4_XS.gguf"), "Rig-GGUF nicht gemountet")
    def test_head_geometry_equals_the_real_27b_config(self):
        d = MC + "Qwen3.8-27B-GGUF-unsloth/"
        td, kv, _ = MP.scan_gguf_set(d + "Qwen3.8-27B-UD-IQ4_XS.gguf")
        derived = MP.config_from_gguf(kv, td)[0]["text_config"]
        with open(d + "config.json") as fh:
            real = MP.text_config(json.load(fh))
        for k in ("num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size",
                  "max_position_embeddings", "full_attention_interval", "linear_num_key_heads", "linear_key_head_dim",
                  "linear_num_value_heads", "linear_value_head_dim", "linear_conv_kernel_dim", "vocab_size", "mtp_num_hidden_layers",
                  "partial_rotary_factor", "attn_output_gate"):
            self.assertEqual(derived[k], real[k], k)

    @unittest.skipUnless(os.path.isfile(MC + "Qwen3.8-Flash-Next-GGUF-unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"), "Rig-GGUF nicht gemountet")
    def test_head_geometry_equals_the_nf_config_fixture(self):
        td, kv, _ = MP.scan_gguf_set(MC + "Qwen3.8-Flash-Next-GGUF-unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf")
        derived = MP.config_from_gguf(kv, td)[0]["text_config"]
        with open(os.path.join(FX_S3, "nextflash_nvfp4", "config.json")) as fh:
            real = MP.text_config(json.load(fh))
        for k in ("num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "max_position_embeddings",
                  "full_attention_interval", "linear_num_key_heads", "linear_key_head_dim", "linear_num_value_heads",
                  "linear_value_head_dim", "num_experts", "num_experts_per_tok", "moe_intermediate_size",
                  "shared_expert_intermediate_size", "hc_count", "hc_lowrank", "indexer_n_heads", "indexer_head_dim", "indexer_budget",
                  "mtp_num_hidden_layers", "partial_rotary_factor", "vocab_size"):
            self.assertEqual(derived[k], real[k], k)


class TestSplitGguf(unittest.TestCase):
    """L15: ein geteilter GGUF-Satz (Kopf nur in Teil 1) wird als EIN Modell gelesen; ein fehlender Teil wird benannt verweigert."""

    def build(self, tmp):
        tensors = qwen35_tensors()
        parts = [tensors[:5], tensors[5:11], tensors[11:]]
        names = []
        for i, chunk in enumerate(parts, 1):
            name = "Model-UD-IQ4_XS-%05d-of-00003.gguf" % i
            kv = qwen35_kv() if i == 1 else {"split.no": (4, i - 1), "split.count": (4, 3)}
            if i == 1:
                kv["split.count"] = (4, 3)
            gguf_file(os.path.join(tmp, name), kv, chunk)
            names.append(name)
        return names, tensors

    def expected_total(self, tensors):
        total = 0
        for name, dims, gt in tensors:
            tname, blck, tsize = MP.GGML_TYPES[gt]
            total += math.prod(dims) // blck * tsize
        return total

    def test_all_parts_are_one_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            names, tensors = self.build(tmp)
            est_dir = MP.estimate(tmp)
            est_part = MP.estimate(os.path.join(tmp, names[1]))                 # ein mittlerer Teil nennt den Satz
            state = MP.probe(tmp)["state"]
        for est in (est_dir, est_part):
            self.assertEqual(est["weights"]["total_bytes"]["v"], self.expected_total(tensors))
            self.assertEqual(est["gguf"]["files"]["v"], 3)
            self.assertEqual(est["arch"]["n_layers"]["v"], 8)
        self.assertEqual(state, "complete")

    def test_a_missing_part_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            names, _ = self.build(tmp)
            os.remove(os.path.join(tmp, names[2]))
            with self.assertRaises(MP.ModelProfileError) as cm:
                MP.estimate(tmp)
            r = MP.estimate_or_state(tmp)
            pr = MP.probe(tmp)
        self.assertIn("Model-UD-IQ4_XS-00003-of-00003.gguf", str(cm.exception))
        self.assertEqual((r["ok"], pr["state"], pr["estimable"]), (False, "gguf_incomplete", False))

    def test_gguf_sets_groups_parts_and_keeps_singles_apart(self):
        sets = MP.gguf_sets(["a-00001-of-00002.gguf", "a-00002-of-00002.gguf", "mmproj.gguf", "b-00001-of-00003.gguf"])
        self.assertEqual(sorted(sets), ["a/00002", "b/00003", "mmproj.gguf"])
        self.assertEqual(len(sets["a/00002"]), 2)

    def test_two_different_sets_need_the_file_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            for n in ("a.gguf", "mmproj.gguf"):
                open(os.path.join(tmp, n), "wb").close()
            self.assertEqual(MP.probe(tmp)["state"], "ambiguous")
            with self.assertRaises(MP.ModelProfileError):
                MP.estimate(tmp)


if __name__ == "__main__":
    unittest.main()
