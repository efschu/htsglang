"""PROFIL-EDITOR S3 (Auftrag 960, 03.10.2026): das Modellprofil ``flliper.model/1``, am Desk geschätzt.

Gepinnt (GPU-frei, ohne Rig-Pfade; die Config-Dateien der echten Checkpoints liegen als Fixture, die 27B-INT8-Kopfzeilen als
komprimiertes Tensorverzeichnis):

* ``model_profile.py`` ist reine Standardbibliothek (das Dashboard lädt es per Dateipfad), und jeder Zahlenwert trägt seine
  Quelle (``config`` | ``Index`` | ``geschätzt`` | ``stat``);
* Gewichtsbytes aus den Safetensors-Köpfen: Tensorsumme + Köpfe == Dateigröße der Shards, BYTEGENAU (27B INT8, 1608 Tensoren);
* Gewichtsbytes allein aus Config + Quantisierung (kein Tensorverzeichnis): 27B INT8 / NF NVFP4 / NF INT4-mixed je <= 1 % neben
  der Kopfsumme (am Metall-Checkpoint gemessen: +0,03 / -0,34 / +0,23 %);
* Stufen-Summen gegen die ``Load weight end ... mem usage``-Zeilen des Boots dkr27browauthoritybar1fs10031930 (<= 2,5 %);
* KV-Zelle 1088 B (NF fp8, Metall fnFL2w123), KV-Seitenbytes 32768 (27B, STORE_CENSUS_KV_PAGE_BYTES), Mamba-Zustand
  1,5586 MiB je Linear-Layer und Slot (Records 1,5588 / 1,5602), Extend-Rate == ``extend_trim.derived_rate_mib`` (Q-694b);
* GGUF-Kopf (ggml-Typen IQ4_XS, Q6_K, F32 byte-genau), Fehlerfälle benannt;
* die abgeleitete ``form.ModelProfile``-Zeile: alle 8 Modellfakten, die Kennung, ``experts.store/swap`` und die Formatzeile
  GLEICH der Handzeile ``qwen27b`` / ``nextflash``, jede andere Abweichung mit Grund; jedes ModelProfile-Feld ist eingeordnet.
"""

import dataclasses
import gzip
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "fixtures", "profil_s3_1003")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
MP_FILE = os.path.join(REPO, "python", "flliper", "srt", "pdflip", "model_profile.py")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import form as F  # noqa: E402
from flliper.srt.pdflip import model_profile as MP  # noqa: E402

GIB = float(1 << 30)
MC = "/spinning/llm_stuff/club-3090/models-cache/"
ALLOWED_SRC = {"config", "Index", "geschätzt", "stat"}
DT = MP.SAFETENSORS_DTYPE_BYTES


def write_shard(path, tensors):
    """Eine Safetensors-Datei NUR mit Kopf (keine Nutzlast): ``tensors`` = {name: (dtype, shape)}."""
    header, off = {}, 0
    for name, (dt, shape) in tensors.items():
        n = 1
        for d in shape:
            n *= d
        nb = int(n * DT[dt])
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + nb]}
        off += nb
    blob = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)) + blob)
    return 8 + len(blob)


def fixture_dir(name):
    return os.path.join(FX, name)


def with_config(tmp, name):
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(fixture_dir(name), "config.json")) as src, open(os.path.join(d, "config.json"), "w") as dst:
        dst.write(src.read())
    return d


def walk_leaves(obj, path=""):
    """Jedes Dict mit ``v`` muss ``src`` tragen."""
    if isinstance(obj, dict):
        if "v" in obj:
            yield path, obj
        else:
            for k, v in obj.items():
                yield from walk_leaves(v, path + "/" + str(k))


def census_dir(tmp, name="qwen27b_int8_vocabembed", shards=2):
    """Das echte 27B-INT8-Tensorverzeichnis (1608 Tensoren) als Header-only-Shards."""
    with gzip.open(os.path.join(fixture_dir(name), "tensors.json.gz"), "rt") as fh:
        meta = json.load(fh)
    d = with_config(tmp, name)
    rows = meta["tensors"]
    per = (len(rows) + shards - 1) // shards
    for i in range(shards):
        chunk = rows[i * per:(i + 1) * per]
        write_shard(os.path.join(d, "model-%05d-of-%05d.safetensors" % (i + 1, shards)), {r[0]: (r[2], r[3]) for r in chunk})
    return d, meta


class TestPureStdlib(unittest.TestCase):
    def test_module_loads_without_flliper_or_torch(self):
        code = ("import importlib.util, sys\n"
                "spec = importlib.util.spec_from_file_location('mp', %r)\n"
                "mod = importlib.util.module_from_spec(spec); sys.modules['mp'] = mod; spec.loader.exec_module(mod)\n"
                "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'flliper', 'numpy', 'safetensors'))\n"
                "assert not bad, bad\n"
                "assert mod.SCHEMA == 'flliper.model/1'\n" % MP_FILE)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=""))
        self.assertEqual(r.returncode, 0, r.stderr)


class TestHeaderPath27B(unittest.TestCase):
    """Das echte 27B-INT8-Tensorverzeichnis (Köpfe von Qwen3.8-27B-INT8-gdncov-vocabembed)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir, cls.meta = census_dir(cls.tmp.name)
        cls.est = MP.estimate(cls.dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_tensor_sum_plus_headers_is_the_shard_size_to_the_byte(self):
        w = self.est["weights"]
        total = sum(r[1] for r in self.meta["tensors"])
        self.assertEqual(w["total_bytes"]["v"], total)
        # Dateigröße der echten Shards == Tensorsumme + Köpfe: die Schätzung hat 0 Byte Abweichung zur Platte
        self.assertEqual(self.meta["disk_bytes"], total + self.meta["header_bytes"])
        self.assertEqual(w["total_bytes"]["src"], "Index")

    def test_families_layers_and_roles(self):
        a = self.est["arch"]
        self.assertEqual(a["n_layers"]["v"], 64)
        self.assertEqual(a["layer_counts"]["v"], {"attn": 16, "gdn": 48, "mamba": 0})
        self.assertEqual(a["layer_families"]["v"][:4], ["gdn", "gdn", "gdn", "attn"])
        self.assertEqual(a["family"]["v"], "dense")
        self.assertTrue(a["hybrid"]["v"])
        self.assertEqual(a["attention"]["v"], "full")
        fam = self.est["weights"]["per_family_mean_bytes"]
        self.assertEqual(fam["attn"]["v"], 372384768)
        self.assertEqual(fam["gdn"]["v"], 383939008)
        self.assertEqual(self.est["format"]["v"], "int8")
        self.assertEqual(self.est["format"]["src"], "Index")

    def test_nonlayer_posts(self):
        w = self.est["weights"]
        self.assertEqual(w["embed_bytes"]["v"], 1271895040)        # vocabembed: die Einbettung ist int8
        self.assertEqual(w["lm_head_bytes"]["v"], 2542796800)
        self.assertEqual(w["visual_bytes"]["v"], 921460192)
        self.assertEqual(w["mtp_bytes"]["v"], 424854528)
        self.assertEqual(w["layers_bytes_expert"]["v"], 0)

    def test_stage_sums_against_the_boot_log(self):
        """dkr27browauthoritybar1fs10031930 P.log, 19:31: 'Load weight end ... mem usage' PP0/PP1/PP2 bei Schnitt 41,12,11."""
        measured = [15.86, 4.34, 6.41]
        got = [x / GIB for x in MP.stage_weight_bytes(self.est, [41, 12, 11])]
        for g, m in zip(got, measured):
            self.assertLess(abs(g - m) / m, 0.025, (got, measured))

    def test_every_number_carries_its_source(self):
        n = 0
        for path, leaf in walk_leaves(self.est):
            n += 1
            self.assertIn(leaf["src"], ALLOWED_SRC, path)
        self.assertGreater(n, 40)
        json.dumps(self.est)

    def test_id_is_independent_of_the_path(self):
        with tempfile.TemporaryDirectory() as t2:
            d2, _ = census_dir(t2, shards=3)
            other = MP.estimate(d2)
        # drei statt zwei Shards, anderer Pfad: gleiches Modell, gleicher Hash (Dateizahl/-größen sind nicht Inhalt)
        a = {k: v for k, v in self.est["weights"].items() if k != "disk_bytes"}
        b = {k: v for k, v in other["weights"].items() if k != "disk_bytes"}
        a.pop("header_bytes", None), b.pop("header_bytes", None)
        self.assertEqual(a, b)


class TestConfigOnly(unittest.TestCase):
    """Nur ``config.json``: die Gewichtsbytes aus Geometrie und Quantisierungsformat (kein Tensorverzeichnis)."""

    #: Kopfsummen der echten Checkpoints (Safetensors-Köpfe, 03.10.2026)
    REAL = {"qwen27b_int8_vocabembed": (29548245472, "int8"),
            "nextflash_nvfp4": (132639846394, "nvfp4"),
            "nextflash_int4mixed": (175229362136, "int4-mixed")}

    def test_totals_within_one_percent_of_the_headers(self):
        for name, (real, fmt) in self.REAL.items():
            with tempfile.TemporaryDirectory() as tmp:
                est = MP.estimate(with_config(tmp, name))
            w = est["weights"]
            self.assertEqual(est["weights_source"]["v"], "config", name)
            self.assertEqual(w["total_bytes"]["src"], "geschätzt", name)
            dev = (w["total_bytes"]["v"] - real) / real
            self.assertLess(abs(dev), 0.01, "%s %+.3f%%" % (name, 100 * dev))
            self.assertEqual(est["format"]["v"], fmt, name)

    def test_nf_experts_are_priced_to_the_byte_per_expert(self):
        with tempfile.TemporaryDirectory() as tmp:
            est = MP.estimate(with_config(tmp, "nextflash_nvfp4"))
        # 48 Layer x 512 Experten; die Kopfsumme der Expertentensoren ist 67948314624 B (Gate+Up+Down samt Skalen)
        self.assertEqual(est["experts"]["n"]["v"], 512)
        self.assertEqual(est["experts"]["top_k"]["v"], 10)
        self.assertEqual(est["experts"]["moe_layers"]["v"], 48)
        self.assertLess(abs(est["experts"]["total_bytes"]["v"] - 67948314624) / 67948314624, 0.0005)

    def test_missing_shards_but_an_index_total_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = with_config(tmp, "qwen27b_int8_vocabembed")
            with open(os.path.join(d, "model.safetensors.index.json"), "w") as fh:
                json.dump({"metadata": {"total_size": 29548245472}, "weight_map": {}}, fh)
            est = MP.estimate(d)
        self.assertEqual(est["weights"]["total_bytes"]["v"], 29548245472)
        self.assertEqual(est["weights"]["total_bytes"]["src"], "Index")
        self.assertIn("total_bytes_formula", est["weights"])

    def test_config_only_is_refusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = with_config(tmp, "qwen27b_int8_vocabembed")
            with self.assertRaises(MP.ModelProfileError):
                MP.estimate(d, allow_config_only=False)


class TestGeometryTerms(unittest.TestCase):
    def cfg(self, name):
        with open(os.path.join(fixture_dir(name), "config.json")) as fh:
            return json.load(fh)

    def est(self, name):
        with tempfile.TemporaryDirectory() as tmp:
            return MP.estimate(with_config(tmp, name))

    def test_kv_cell_nf_is_1088_bytes_per_attention_layer_fp8(self):
        """Metall fnFL2w123: 1024 + 64 = 1088 B; mal 12 Attention-Layer = 12288 (fa_kv_token_cell_bytes, rc12r)."""
        kv = self.est("nextflash_nvfp4")["kv"]
        v = kv["variants"]["fp8_e4m3"]
        self.assertEqual(v["cell_bytes_per_attn_layer_token"]["v"], 1088.0)
        self.assertEqual(v["payload_bytes"]["v"], 1024.0)
        self.assertEqual(v["payload_per_token_all_attn_layers"]["v"], 12288.0)
        self.assertEqual(kv["attn_layers"]["v"], 12)
        self.assertEqual(self.est("nextflash_nvfp4")["arch"]["attention"]["v"], "qsa")

    def test_kv_27b_payload_is_the_store_census_page_bytes(self):
        """STORE_CENSUS_KV_PAGE_BYTES 32768 (qwen27b-Records): 16 Attention-Layer x 4 KV-Köpfe x 512 x 1 B (fp8)."""
        with open(os.path.join(REPO, "python", "flliper", "srt", "pdflip", "profile_records_data", "qwen27b.json")) as fh:
            rec = {r["name"]: r["value"] for r in json.load(fh)["records"]}
        kv = self.est("qwen27b_int8_vocabembed")["kv"]
        self.assertEqual(kv["variants"]["fp8_e4m3"]["payload_per_token_all_attn_layers"]["v"], rec["STORE_CENSUS_KV_PAGE_BYTES"])
        self.assertEqual(kv["variants"]["auto"]["cell_bytes_per_attn_layer_token"]["v"], 4096.0)   # bf16, kein Skalenpuffer
        self.assertEqual(kv["chosen"]["v"], "auto")

    def test_kv_dtype_choice_selects_the_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = with_config(tmp, "qwen27b_int8_vocabembed")
            est = MP.estimate(d, kv_dtype="fp8_e4m3")
        self.assertEqual(est["kv"]["chosen"]["v"], "fp8_e4m3")
        self.assertEqual(est["kv"]["cell_bytes_per_attn_layer_token"]["v"], 2176.0)

    def test_mamba_state_matches_the_p_records(self):
        """P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT: 27B 1,5588, NF 1,5602 -- das Rig fährt --mamba-ssm-dtype bfloat16."""
        for name, rec in (("qwen27b_int8_vocabembed", 1.5588), ("nextflash_nvfp4", 1.5602)):
            st = self.est(name)["state"]
            self.assertEqual(st["variants_mib"]["float32"], 3.0586, name)
            self.assertLess(abs(st["variants_mib"]["bfloat16"] - rec), 0.002, name)
            self.assertEqual(st["ssm_dtype"]["v"], "float32")          # die Config sagt float32, der Standard folgt der Config
        with tempfile.TemporaryDirectory() as tmp:
            est = MP.estimate(with_config(tmp, "qwen27b_int8_vocabembed"), mamba_ssm_dtype="bfloat16")
        self.assertEqual(est["state"]["per_linear_layer_per_slot_mib"]["v"], 1.5586)
        self.assertEqual(est["state"]["linear_layers"]["v"], 48)

    def test_extend_rate_is_the_q694b_formula(self):
        """0,4422 MiB/Zeile für Qwen3.8-27B (Q-694b 1de15e82ca, extend_trim.derived_rate_mib); NF 0,4622 (identisch gegengerechnet)."""
        self.assertEqual(MP.extend_rate_mib_per_row(self.cfg("qwen27b_int8_vocabembed")), 0.4422)
        self.assertEqual(MP.extend_rate_mib_per_row(self.cfg("nextflash_nvfp4")), 0.4622)
        self.assertIsNone(MP.extend_rate_mib_per_row({}))
        e = self.est("qwen27b_int8_vocabembed")["activation"]["extend_rate_mib_per_row"]
        self.assertEqual((e["v"], e["src"]), (0.4422, "geschätzt"))

    def test_context_and_rope(self):
        c = self.est("qwen27b_int8_vocabembed")["context"]
        self.assertEqual(c["max_position_embeddings"]["v"], 262144)
        self.assertEqual(c["rope"]["type"]["v"], "default")
        self.assertEqual(c["rope"]["theta"]["v"], 10000000)
        self.assertEqual(c["rope"]["mrope_section"]["v"], [11, 11, 10])
        self.assertNotIn("rope_extended_tokens", c)

    def test_yarn_extension_is_priced_from_the_factor(self):
        t = {"num_hidden_layers": 2, "hidden_size": 8, "max_position_embeddings": 4096,
             "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096}}
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as fh:
                json.dump(t, fh)
            est = MP.estimate(tmp)
        self.assertEqual(est["context"]["rope_extended_tokens"]["v"], 16384)

    def test_expert_buffer_fraction_is_the_planner_formula(self):
        """NF y8: fraction 0,33 -> 169 resident + 32 Scratch = 201 von 512; 0,701 -> 391; 0,9956 -> 512 (510 + 2)."""
        self.assertEqual(MP.expert_buffer_fraction(512, 0.33, 32) * 512, 201)
        self.assertEqual(MP.expert_buffer_fraction(512, 0.701, 32) * 512, 391)
        self.assertEqual(MP.expert_buffer_fraction(512, 0.995605, 2) * 512, 512)
        self.assertEqual(MP.expert_buffer_fraction(512, 1.0, 32), 1.0)
        self.assertEqual(MP.expert_buffer_fraction(0, 0.5, 32), 1.0)


class TestClassifier(unittest.TestCase):
    def test_hf_names(self):
        c = MP.classify
        self.assertEqual(c("mtp.layers.0.mlp.up_proj.weight"), ("mtp", None, ""))
        self.assertEqual(c("model.visual.blocks.3.attn.qkv.weight"), ("visual", None, ""))
        self.assertEqual(c("lm_head.weight")[0], "lm_head")
        self.assertEqual(c("model.language_model.embed_tokens.weight")[0], "embed")
        self.assertEqual(c("model.language_model.layers.7.self_attn.q_proj.weight"), ("layer", 7, "attn"))
        self.assertEqual(c("model.language_model.layers.7.linear_attn.in_proj_qkv.weight"), ("layer", 7, "gdn"))
        self.assertEqual(c("model.language_model.layers.7.mlp.experts.12.up_proj.weight"), ("expert", 7, ""))
        self.assertEqual(c("model.language_model.layers.7.mlp.gate.weight"), ("layer", 7, "router"))
        self.assertEqual(c("model.language_model.layers.7.mlp.shared_expert.up_proj.weight"), ("layer", 7, "shared"))
        self.assertEqual(c("model.language_model.layers.7.mlp.down_proj.weight"), ("layer", 7, "mlp"))
        self.assertEqual(c("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_9.weight")[0], "ngram")
        self.assertEqual(c("model.language_model.layers.1.ple.key_proj.weight")[0], "ple")
        self.assertEqual(c("model.language_model.layers.7.attn_hyper_connection.hc_norm.weight"), ("layer", 7, "hc"))
        self.assertEqual(c("model.norm.weight")[0], "other")

    def test_layer_families(self):
        self.assertEqual(MP.layer_families({"num_hidden_layers": 8, "full_attention_interval": 4}),
                         ("gdn", "gdn", "gdn", "attn", "gdn", "gdn", "gdn", "attn"))
        self.assertEqual(MP.layer_families({"num_hidden_layers": 3}), ("attn", "attn", "attn"))
        self.assertEqual(MP.layer_families({"num_hidden_layers": 2, "layer_types": ["mamba", "full_attention"],
                                            "mamba_num_heads": 4}), ("mamba", "attn"))
        with self.assertRaises(MP.ModelProfileError):
            MP.layer_families({"num_hidden_layers": 3, "layer_types": ["full_attention"]})
        with self.assertRaises(MP.ModelProfileError):
            MP.layer_families({})


class TestSyntheticMoE(unittest.TestCase):
    """Ein kleines hybrides MoE-Modell, die erwarteten Bytes aus der Tensorliste selbst gerechnet."""

    CFG = {"architectures": ["TinyMoe"], "model_type": "tiny",
           "text_config": {"model_type": "tiny_text", "num_hidden_layers": 4, "full_attention_interval": 2, "hidden_size": 16,
                           "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8, "num_experts": 4,
                           "num_experts_per_tok": 2, "moe_intermediate_size": 8, "shared_expert_intermediate_size": 8,
                           "vocab_size": 32, "max_position_embeddings": 1024, "dtype": "bfloat16", "mtp_num_hidden_layers": 1,
                           "linear_num_key_heads": 2, "linear_key_head_dim": 4, "linear_num_value_heads": 4,
                           "linear_value_head_dim": 4, "linear_conv_kernel_dim": 4, "mamba_ssm_dtype": "float32"}}

    def build(self, tmp):
        with open(os.path.join(tmp, "config.json"), "w") as fh:
            json.dump(self.CFG, fh)
        L = "model.language_model.layers.%d."
        t = {"model.language_model.embed_tokens.weight": ("BF16", (32, 16)), "lm_head.weight": ("BF16", (32, 16))}
        for i in range(4):
            p = L % i
            t[p + "input_layernorm.weight"] = ("BF16", (16,))
            if i % 2 == 1:
                t[p + "self_attn.q_proj.weight"] = ("BF16", (16, 16))
                t[p + "self_attn.o_proj.weight"] = ("BF16", (16, 16))
            else:
                t[p + "linear_attn.in_proj_qkv.weight"] = ("BF16", (32, 16))
                t[p + "linear_attn.out_proj.weight"] = ("BF16", (16, 16))
            t[p + "mlp.gate.weight"] = ("BF16", (4, 16))
            t[p + "mlp.shared_expert.up_proj.weight"] = ("BF16", (8, 16))
            for e in range(4):
                for nm in ("gate_proj", "up_proj", "down_proj"):
                    t[p + "mlp.experts.%d.%s.weight" % (e, nm)] = ("BF16", (8, 16))
        t["mtp.fc.weight"] = ("BF16", (16, 32))
        t["model.visual.blocks.0.attn.qkv.weight"] = ("BF16", (16, 16))
        t["model.norm.weight"] = ("BF16", (16,))
        write_shard(os.path.join(tmp, "model-00001-of-00001.safetensors"), t)
        return t

    def test_estimate_counts_what_the_tensors_say(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = self.build(tmp)
            est = MP.estimate(tmp)
        sz = {n: int(DT[d] * __import__("math").prod(s)) for n, (d, s) in t.items()}
        exp_bytes = sum(v for n, v in sz.items() if ".mlp.experts." in n)
        w = est["weights"]
        self.assertEqual(w["total_bytes"]["v"], sum(sz.values()))
        self.assertEqual(w["layers_bytes_expert"]["v"], exp_bytes)
        self.assertEqual(w["embed_bytes"]["v"], sz["model.language_model.embed_tokens.weight"])
        self.assertEqual(w["mtp_bytes"]["v"], sz["mtp.fc.weight"])
        self.assertEqual(w["visual_bytes"]["v"], sz["model.visual.blocks.0.attn.qkv.weight"])
        self.assertEqual(w["other_bytes"]["v"], sz["model.norm.weight"])
        e = est["experts"]
        self.assertEqual((e["n"]["v"], e["top_k"]["v"], e["moe_layers"]["v"]), (4, 2, 4))
        self.assertEqual(e["bytes_per_expert"]["v"], 3 * 8 * 16 * 2)          # Gate+Up+Down, bf16
        self.assertEqual(est["arch"]["family"]["v"], "moe")
        self.assertEqual(est["arch"]["layer_families"]["v"], ["gdn", "attn", "gdn", "attn"])
        self.assertEqual(est["format"]["v"], "bf16")
        self.assertEqual(est["draft"]["mtp_layers"]["v"], 1)
        self.assertTrue(est["draft"]["mtp_tensors_found"]["v"])
        # Stufen: Experten skalieren, der Rest nicht; Einbettung vorn, lm_head hinten
        st = MP.stage_weight_bytes(est, [2, 2], expert_fractions=[0.5, 1.0])
        lb, le = w["layer_bytes"]["v"], w["layer_expert_bytes"]["v"]
        self.assertEqual(st[0], sum(lb[:2]) + 0.5 * sum(le[:2]) + w["embed_bytes"]["v"])
        self.assertEqual(st[1], sum(lb[2:]) + sum(le[2:]) + w["lm_head_bytes"]["v"])
        with self.assertRaises(ValueError):
            MP.stage_weight_bytes(est, [2, 1])

    def test_config_only_agrees_with_the_tensors_on_the_linear_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.build(tmp)
            est = MP.estimate(tmp)
            cw = MP.estimate_weights_from_config(self.CFG)
        # Experten und Einbettung rechnet die Formel genau; die Mixer sind im Spielzeug bewusst anders geformt
        self.assertEqual(sum(cw["layer_expert_bytes"]), est["weights"]["layers_bytes_expert"]["v"])
        self.assertEqual(cw["embed_bytes"], est["weights"]["embed_bytes"]["v"])

    def test_truncated_header_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as fh:
                json.dump(self.CFG, fh)
            with open(os.path.join(tmp, "a.safetensors"), "wb") as fh:
                fh.write(struct.pack("<Q", 1000) + b"{}")
            with self.assertRaises(MP.ModelProfileError) as cm:
                MP.estimate(tmp)
        self.assertIn("truncated", str(cm.exception))

    def test_unknown_dtype_is_flagged_not_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as fh:
                json.dump({"num_hidden_layers": 1, "hidden_size": 4}, fh)
            blob = json.dumps({"model.layers.0.mlp.up_proj.weight": {"dtype": "WEIRD", "shape": [2, 2], "data_offsets": [0, 8]}}).encode()
            with open(os.path.join(tmp, "m.safetensors"), "wb") as fh:
                fh.write(struct.pack("<Q", len(blob)) + blob)
            est = MP.estimate(tmp)
        self.assertTrue(any("WEIRD" in w for w in est["warnings"]))

    def test_no_config_and_no_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MP.ModelProfileError):
                MP.estimate(tmp)
            with self.assertRaises(MP.ModelProfileError):
                MP.estimate(os.path.join(tmp, "gibt-es-nicht"))


class TestGguf(unittest.TestCase):
    @staticmethod
    def gguf(path, kv, tensors):
        def s(x):
            b = x.encode()
            return struct.pack("<Q", len(b)) + b

        out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kv))
        for k, (vt, v) in kv.items():
            out += s(k) + struct.pack("<I", vt)
            out += struct.pack("<I", v) if vt == 4 else s(v)
        for name, dims, gt in tensors:
            out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<I", gt) + struct.pack("<Q", 0)
        with open(path, "wb") as fh:
            fh.write(out)

    def test_ggml_bytes_are_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"num_hidden_layers": 2, "hidden_size": 256, "full_attention_interval": 2, "vocab_size": 8,
                   "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 128}
            with open(os.path.join(tmp, "config.json"), "w") as fh:
                json.dump(cfg, fh)
            path = os.path.join(tmp, "m.gguf")
            self.gguf(path, {"general.architecture": (8, "qwen35"), "qwen35.block_count": (4, 2)},
                      [("token_embd.weight", (256, 8), 14),           # Q6_K: 256-Elemente-Blöcke zu 210 B
                       ("blk.0.ssm_a", (256,), 0),                     # F32
                       ("blk.0.ffn_up.weight", (256, 512), 23),        # IQ4_XS: 136 B je 256
                       ("blk.1.attn_q.weight", (256, 256), 23),
                       ("output.weight", (256, 8), 23)])
            est = MP.estimate(path)
            self.assertEqual(MP.estimate(tmp)["weights"]["total_bytes"]["v"], est["weights"]["total_bytes"]["v"])   # Verzeichnis mit einer .gguf
        w = est["weights"]
        self.assertEqual(est["format"]["v"], "gguf")
        self.assertEqual(w["embed_bytes"]["v"], 256 * 8 // 256 * 210)
        self.assertEqual(w["lm_head_bytes"]["v"], 256 * 8 // 256 * 136)
        self.assertEqual(w["layer_bytes"]["v"], [256 * 4 + 256 * 512 // 256 * 136, 256 * 256 // 256 * 136])
        self.assertEqual(w["total_bytes"]["v"], 8 * 210 + 1024 + 512 * 136 + 256 * 136 + 8 * 136)
        self.assertEqual(est["arch"]["layer_families"]["v"], ["gdn", "attn"])

    def test_not_a_gguf_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as fh:
                json.dump({"num_hidden_layers": 1}, fh)
            path = os.path.join(tmp, "x.gguf")
            with open(path, "wb") as fh:
                fh.write(b"NOPE" + b"\0" * 32)
            with self.assertRaises(MP.ModelProfileError):
                MP.estimate(path)

    def test_several_ggufs_need_the_file_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as fh:
                json.dump({"num_hidden_layers": 1}, fh)
            for n in ("a.gguf", "b.gguf"):
                open(os.path.join(tmp, n), "wb").close()
            with self.assertRaises(MP.ModelProfileError) as cm:
                MP.estimate(tmp)
        self.assertIn("a.gguf", str(cm.exception))


class TestRegistryDerivation(unittest.TestCase):
    """Die abgeleitete ``form.ModelProfile``-Zeile gegen die Handzeilen ``qwen27b`` und ``nextflash``."""

    CASES = (("qwen27b", "qwen27b_int8_vocabembed", "int8", MC + "Qwen3.8-27B-INT8-gdncov-vocabembed"),
             ("nextflash", "nextflash_int4mixed", "int4-mixed", MC + "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"))

    def derived(self, rid, fx, ckpt):
        with tempfile.TemporaryDirectory() as tmp:
            est = MP.estimate(with_config(tmp, fx))
        return MP.to_model_profile(MP.derive_registry_fields(est, row_id=rid, checkpoint=ckpt)), est

    def test_every_model_profile_field_is_classified(self):
        names = {f.name for f in dataclasses.fields(F.ModelProfile)}
        # FIELD_CLASS ist die Vereinigung der Zeilen-Felder beider Linien (27B-Linie hat Felder, die NF nicht führt, und umgekehrt):
        # jedes Feld DIESER Dataclass muss eingeordnet sein, ein Feld nur der anderen Linie ist kein Fehler
        self.assertLessEqual(names, set(MP.FIELD_CLASS), "ein neues ModelProfile-Feld muss in model_profile.FIELD_CLASS eingeordnet werden")
        self.assertEqual({c for c, _ in MP.FIELD_CLASS.values()}, {"fact", "name", "experts", "format", "policy"})
        self.assertEqual(sorted(n for n, (c, _) in MP.FIELD_CLASS.items() if c == "fact"),
                         ["arch", "attn", "context_tokens", "d_layout", "kv_dtype", "page_size", "ple", "replayssm"])

    def test_facts_name_experts_and_format_equal_the_hand_rows(self):
        for rid, fx, fmt, ckpt in self.CASES:
            row, est = self.derived(rid, fx, ckpt)
            self.assertEqual(est["format"]["v"], fmt)
            rows = MP.compare_with_registry(row, F.PROFILES[rid])
            self.assertEqual(len(rows), len(dataclasses.fields(F.ModelProfile)))
            for r in rows:
                if r["class"] != "policy":
                    self.assertTrue(r["equal"], "%s.%s: %r != %r" % (rid, r["field"], r["derived"], r["hand"]))
            self.assertEqual(sum(1 for r in rows if r["class"] == "fact"), 8)

    def test_every_deviation_has_its_reason(self):
        for rid, fx, fmt, ckpt in self.CASES:
            row, _ = self.derived(rid, fx, ckpt)
            n_diff = 0
            for r in MP.compare_with_registry(row, F.PROFILES[rid]):
                if not r["equal"] or r["partial"]:
                    n_diff += 1
                    self.assertTrue(r["reason"].strip(), "%s.%s weicht ohne Grund ab" % (rid, r["field"]))
            self.assertGreater(n_diff, 20)              # die Betriebsschalter der Handzeilen sind das Betriebswissen

    def test_the_generic_row_is_conservative_and_valid(self):
        for rid, fx, fmt, ckpt in self.CASES:
            row, _ = self.derived(rid + "-generisch", fx, ckpt)
            self.assertEqual(row.draft.kind, "none")
            self.assertEqual(row.p_draft, "none")
            present = {f.name for f in dataclasses.fields(F.ModelProfile)}
            switches = [fld for fld in ("agent_span", "standard_form", "inline_system_in_place", "told_paced", "p_twin_defer", "d_hostgap_levers",
                                        "d_release_fixes", "p_row_authority", "mamba_carrier_hold", "repack_outside_pool") if fld in present]
            self.assertGreaterEqual(len(switches), 7)       # die Linie führt die Mehrzahl dieser Schalter; fehlende gehören der anderen Linie
            for fld in switches:
                self.assertFalse(getattr(row, fld), fld)
            self.assertEqual(row.constants, {})
            self.assertEqual(row.vision, "off")
            sw = row.switch_defaults()
            self.assertFalse(sw["FLLIPER_PDFLIP_ENABLE_AGENT_SPAN"])
            self.assertEqual(row.expect["draft"], ("none",))
            self.assertIn(row.formats[fmt].name, row.formats)

    def test_register_never_overwrites_and_leaves_the_hand_rows_alone(self):
        before = {k: F.PROFILES[k] for k in F.PROFILES}
        row, _ = self.derived("generisch-test-1003", "nextflash_int4mixed", MC + "x")
        try:
            MP.register_profile(row)
            self.assertIn("generisch-test-1003", F.PROFILES)
            self.assertIn("generisch-test-1003", F.PROFILE_EXPECT)
            self.assertIn("generisch-test-1003", F.PROFILE_SWITCH_DEFAULTS)
            with self.assertRaises(MP.ModelProfileError):
                MP.register_profile(row)
            with self.assertRaises(MP.ModelProfileError):
                MP.register_profile(dataclasses.replace(row, id="qwen27b"))
        finally:
            F.PROFILES.pop("generisch-test-1003", None)
            F.PROFILE_EXPECT.pop("generisch-test-1003", None)
            F.PROFILE_SWITCH_DEFAULTS.pop("generisch-test-1003", None)
        for k, v in before.items():
            self.assertIs(F.PROFILES[k], v)

    def test_a_dense_and_a_moe_model_get_their_axes_from_the_estimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(os.path.exists(tmp), True)
            TestSyntheticMoE().build(tmp)
            est = MP.estimate(tmp)
        f = MP.derive_registry_fields(est, row_id="tiny", checkpoint="/x/tiny")
        self.assertEqual((f["arch"], f["experts"]["store"], f["attn"], f["d_layout"], f["page_size"], f["kv_dtype"]),
                         ("moe", "offload", "full", "paged_dcp", 1, "auto"))
        self.assertEqual(f["expect"]["experts"], ("resident", "offload"))
        self.assertTrue(f["replayssm"])
        self.assertEqual(f["context_tokens"], 1024)
        MP.to_model_profile(f)


class TestAgreesWithThePlannerBuildingBlocks(unittest.TestCase):
    """Die Schätzung ersetzt keinen Baustein des Planers, sie fasst sie zusammen: gleiche Zahlen wie
    ``pp_cut.checkpoint_weight_terms`` und ``pp_cut.layer_families_from_config`` auf demselben Verzeichnis."""

    def test_weight_terms_equal_checkpoint_weight_terms_dense_27b(self):
        from flliper.srt.planner import pp_cut

        with tempfile.TemporaryDirectory() as tmp:
            d, _ = census_dir(tmp)
            est = MP.estimate(d)
            ct = pp_cut.checkpoint_weight_terms(d)
        w = est["weights"]
        self.assertAlmostEqual(ct.attn_layer_weight_bytes, sum(b for b, f in zip(w["layer_bytes"]["v"], est["arch"]["layer_families"]["v"]) if f == "attn") / 16, places=3)
        self.assertEqual(int(ct.attn_layer_weight_bytes), w["per_family_mean_bytes"]["attn"]["v"])
        self.assertEqual(int(ct.linear_layer_weight_bytes), w["per_family_mean_bytes"]["gdn"]["v"])
        self.assertEqual(ct.embedding_weight_bytes, w["embed_bytes"]["v"])
        self.assertEqual(ct.lm_head_weight_bytes, w["lm_head_bytes"]["v"])
        self.assertEqual(ct.replicated_weight_bytes, w["visual_bytes"]["v"] + w["mtp_bytes"]["v"])
        self.assertEqual(ct.n_layers, est["arch"]["n_layers"]["v"])
        self.assertEqual(len(ct.attention_layer_indices), est["arch"]["layer_counts"]["v"]["attn"])

    def test_weight_terms_equal_checkpoint_weight_terms_moe(self):
        from flliper.srt.planner import pp_cut

        with tempfile.TemporaryDirectory() as tmp:
            TestSyntheticMoE().build(tmp)
            est = MP.estimate(tmp)
            ct = pp_cut.checkpoint_weight_terms(tmp)
        w = est["weights"]
        self.assertEqual(ct.num_experts, est["experts"]["n"]["v"])
        self.assertEqual(ct.expert_layer_weight_bytes, est["experts"]["bytes_per_moe_layer"]["v"])
        self.assertEqual(ct.expert_layer_weight_bytes * 4, w["layers_bytes_expert"]["v"])

    def test_layer_families_equal_layer_families_from_config(self):
        from flliper.srt.planner import pp_cut

        for name in ("qwen27b_int8_vocabembed", "nextflash_nvfp4"):
            with open(os.path.join(fixture_dir(name), "config.json")) as fh:
                cfg = json.load(fh)
            theirs = pp_cut.layer_families_from_config(cfg)
            mine = MP.layer_families(MP.text_config(cfg))
            self.assertEqual(len(theirs), len(mine))
            self.assertEqual(["full_attention" if m == "attn" else "linear_attention" for m in mine], list(theirs), name)


@unittest.skipUnless(os.path.isfile(MC + "Qwen3.8-27B-INT8-gdncov/model-00001-of-00018.safetensors"), "Rig-Checkpoint nicht gemountet")
class TestOnTheRigCheckpoint(unittest.TestCase):
    def test_header_sum_plus_headers_equals_the_shards_on_disk(self):
        est = MP.estimate(MC + "Qwen3.8-27B-INT8-gdncov")
        w = est["weights"]
        self.assertEqual(w["total_bytes"]["v"] + w["header_bytes"]["v"], w["disk_bytes"]["v"])


if __name__ == "__main__":
    unittest.main()
