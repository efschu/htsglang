# SPDX-License-Identifier: Apache-2.0
"""H87: a D-only boot has no flip arm, and no record of another model prices it.

Befund (container rc2.1c2, NVFP4 --d-only, boot dkrnfdnvfp4dbar109251235,
launcher.log Z.126-133): W87/W20 "no arm funds a flip".

* Ursache 1: the host ledger priced a FLIP arm for a boot that starts group D
  alone -- the 27B ratchet series 8.70 GiB plus a dormant group's heaps
  (3 x 2.36 GiB) -> run leftover -2.82 GiB against the 76 GiB cgroup ceiling.
* Ursache 2: with no record of this model in the container's sidecar the run
  origin fell to the dk7 residual 24.50 GiB -- a Qwen3.8-27B-INT8 boot
  (weg2dk7), priced for a Qwen3.8-Flash-Next-NVFP4 boot without a word.

The identity every check here uses is the MEMORY FOOTPRINT
(``weg2_form.footprint_key``): config minus provenance keys, quantisation
sidecars, (name, dtype, shape) of every tensor from the safetensors headers --
never the weight values. An abliterated derivative (identical headers) is the
same footprint; a re-quantisation under the same name is not.

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, fake /proc/meminfo + cgroup, synthetic
checkpoints (a config and a few safetensors headers, a handful of bytes).
"""

import json
import os
import shutil
import struct
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import form as weg2_form
from sglang.srt.weg2 import host_ledger, launcher
from sglang.srt.weg2 import wake_credit as wc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

GIB = host_ledger.GIB

# ------------------------------------------- MEASURED, dkrnfdnvfp4dbar109251235
# launcher.log Z.127 (WEG2-HOST-LEDGER TERMS): memtotal 125.70, memavail 86.10,
# cgroup memory.current 1.14 GiB of which reclaimable 0.37, ceiling 76.00
# [memory.max]; Z.132: anchors 0.40 rings 0.22 arena 23.75 (staging 0.05 GB,
# anchor 100 MiB), heaps 17.23; Z.96 d_draft_host 1587 MiB; ring absent (Z.103).
C_MEMTOTAL_KB = int(125.70 * (1 << 20))
C_MEMAVAIL_KB = int(86.10 * (1 << 20))
C_CG_CURRENT_B = int(1.14 * GIB)
C_RECLAIMABLE_B = int(0.37 * GIB)
C_CEILING_B = 76 * (1 << 30)
C_ARENA = dict(arena_gib=23.75, staging_gb=0.05, anchor_mib=100)
C_D_DRAFT_HOST_GIB = 1587 / 1024.0
DK7_ORIGIN_GIB = 24.50

NF_NVFP4 = "Qwen3.8-Flash-Next-NVFP4-nvidia"
#: the checkpoint of the built-in ledger references (host_ledger.REFERENCE_MODEL)
REF_27B = "Qwen3.8-27B-INT8-gdncov-vocabembed"
MODELS_CACHE = "/spinning/llm_stuff/club-3090/models-cache"


def _fake_host(tmp: str, ceiling: str = str(C_CEILING_B)) -> tuple:
    meminfo = os.path.join(tmp, "meminfo")
    with open(meminfo, "w") as f:
        f.write(f"MemTotal:       {C_MEMTOTAL_KB} kB\nMemFree:        1000000 kB\n"
                f"MemAvailable:   {C_MEMAVAIL_KB} kB\nShmem:          0 kB\n"
                f"SwapTotal:      0 kB\n")
    cg = os.path.join(tmp, "cgroup")
    os.makedirs(cg, exist_ok=True)
    for name, val in (("memory.current", C_CG_CURRENT_B), ("memory.peak", C_CG_CURRENT_B),
                      ("memory.max", ceiling)):
        with open(os.path.join(cg, name), "w") as f:
            f.write(f"{val}\n")
    with open(os.path.join(cg, "memory.events"), "w") as f:
        f.write("low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n")
    with open(os.path.join(cg, "memory.stat"), "w") as f:
        f.write(f"anon {C_CG_CURRENT_B - C_RECLAIMABLE_B}\nfile {C_RECLAIMABLE_B}\n"
                f"shmem 0\nunevictable 0\nslab_reclaimable 0\n")
    return meminfo, cg


# ---------------------------------------------------------- synthetic checkpoints


def _write_safetensors(path: str, tensors: dict) -> None:
    """A valid safetensors header (+ zero payload) for ``{name: (dtype, shape)}``."""
    hdr, off = {"__metadata__": {"format": "pt"}}, 0
    size = {"I32": 4, "BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}
    for name, (dt, shape) in sorted(tensors.items()):
        n = size[dt]
        for d in shape:
            n *= d
        hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    raw = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + b"\x00" * off)


BASE_TENSORS = {
    "model-00001-of-00002.safetensors": {
        "model.layers.0.mlp.experts.0.down_proj.weight_packed": ("I32", (8, 4)),
        "model.layers.0.mlp.experts.0.down_proj.weight_scale": ("BF16", (8, 2)),
    },
    "model-00002-of-00002.safetensors": {
        "lm_head.weight": ("BF16", (16, 8)),
    },
}
BASE_CONFIG = {"architectures": ["Qwen3_8ForCausalLM"], "_name_or_path": "/base",
               "transformers_version": "4.57.0",
               "quantization_config": {"bits": 4, "group_size": 32},
               "text_config": {"num_hidden_layers": 1, "hidden_size": 8}}


def _make_ckpt(root: str, name: str, config=None, tensors=None, payload=b"") -> str:
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config if config is not None else BASE_CONFIG, f)
    tensors = tensors if tensors is not None else BASE_TENSORS
    weight_map = {}
    for shard, ts in tensors.items():
        _write_safetensors(os.path.join(d, shard), ts)
        if payload:
            with open(os.path.join(d, shard), "ab") as f:
                f.write(payload)
        weight_map.update({t: shard for t in ts})
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    return d


class _Tmp(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="h87_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ------------------------------------------------------------------- Ursache 1


class TestDOnlyHasNoFlipArm(_Tmp):
    """--d-only: no flip ratchet, no dormant group, no P host pools."""

    def _choose(self, **kw):
        meminfo, cg = _fake_host(self.tmp)
        return host_ledger.choose(
            *[host_ledger.read_meminfo(meminfo)[k] for k in ("MemTotal", "MemAvailable")],
            arms=[(1, 600)], ring_absent_by_design=True, s_gb_d=4,
            cg_current_bytes=C_CG_CURRENT_B, reclaimable_bytes=C_RECLAIMABLE_B,
            slab_reclaimable_bytes=0, cg_ceiling_bytes=C_CEILING_B,
            d_draft_host_gib=C_D_DRAFT_HOST_GIB, **C_ARENA, **kw)

    def test_the_container_refusal_reproduces_as_a_flip_arm(self):
        """Control: the rc2.1c2 box refuses a FLIP arm at run=-2.82 GiB with the
        27B series 8.70 GiB charged -- the refusal of Z.126, bit for bit."""
        series = host_ledger.resolve_flip_ratchet_gib({})
        self.assertAlmostEqual(series.charged_gib, 8.697, places=2)
        with self.assertRaises(host_ledger.Weg2HostLedgerRefused) as cm:
            self._choose(flip_ratchet=series)
        self.assertIn("run moment (-2.8", str(cm.exception))

    def test_d_only_prices_one_awake_group_and_no_flip(self):
        arm, _h, lines = self._choose(flip_ratchet=host_ledger.d_only_flip_ratchet(),
                                      d_only=True)
        t = arm.terms
        self.assertEqual(t["flip_ratchet_gib"], 0.0)
        self.assertEqual(t["flip_ratchet_flips_priced"], 0)
        self.assertAlmostEqual(t["heaps_gib"], 3 * host_ledger.HEAP_AWAKE_GIB)
        self.assertEqual(t["draft_host_p_gib"], 0.0)
        self.assertTrue(t["d_only"])
        self.assertGreater(arm.run_leftover_gib, 0.0)
        self.assertTrue(any(host_ledger.D_ONLY_TAG in ln for ln in lines), lines)

    def test_the_launcher_seam_prices_d_only_without_a_flip(self):
        """Through the ONE ledger call site, from files: d_only reaches the arm."""
        meminfo, cg = _fake_host(self.tmp)
        model = _make_ckpt(self.tmp, NF_NVFP4)
        orig = launcher._weg2_arena_ledger_terms
        launcher._weg2_arena_ledger_terms = lambda _m: dict(C_ARENA)
        try:
            arm, _h, lines, _cg = launcher.choose_host_ledger(
                0, 0, d_draft_host_gib=C_D_DRAFT_HOST_GIB, meminfo_path=meminfo,
                cgroup_root=cg, record_path=os.path.join(self.tmp, "none.json"),
                pin_m_mib=600, s_gb_d=4, model_dir=model,
                weights_cpu_backup_armed=False, d_only=True)
        finally:
            launcher._weg2_arena_ledger_terms = orig
        self.assertEqual(arm.terms["flip_ratchet_gib"], 0.0)
        self.assertIn(host_ledger.D_ONLY_TAG, arm.terms["flip_ratchet_source"])
        self.assertGreater(arm.run_leftover_gib, 0.0)

    def test_main_hands_its_d_only_flag_to_the_ledger(self):
        import inspect

        src = inspect.getsource(launcher.main)
        self.assertIn('d_only=bool(getattr(ns, "d_only", False))', src)


# ------------------------------------------------------------------- Ursache 2


class TestRunOriginOnlyFromTheSameModel(_Tmp):

    def test_no_record_and_a_foreign_model_takes_the_launch_reading_by_name(self):
        origin, src = host_ledger.run_origin_gib(
            0.77, {}, reference_model_ok=False, reference_model_why="NVFP4 != 27B")
        self.assertAlmostEqual(origin, 0.77)
        self.assertIn(host_ledger.FOREIGN_REFERENCE_TAG, src)
        self.assertIn("FALLBACK", src)
        self.assertIn(host_ledger.REFERENCE_MODEL, src)
        self.assertIn("NVFP4 != 27B", src)

    def test_the_reference_model_and_a_legacy_caller_keep_dk7(self):
        for ok in (None, True):
            origin, src = host_ledger.run_origin_gib(0.77, {}, reference_model_ok=ok)
            self.assertAlmostEqual(origin, host_ledger.dk7_run_residual_gib())
            self.assertAlmostEqual(origin, DK7_ORIGIN_GIB, places=1)
            self.assertNotIn(host_ledger.FOREIGN_REFERENCE_TAG, src)

    def test_a_same_model_record_still_wins_over_the_fallback(self):
        rec = {"P": {"run_residual_gib": 5.0, "boot_tag": "own", "interleaved": False}}
        origin, src = host_ledger.run_origin_gib(0.77, rec, reference_model_ok=False)
        self.assertAlmostEqual(origin, 5.0)
        self.assertNotIn("FALLBACK", src)

    def _seam(self, model: str):
        meminfo, cg = _fake_host(self.tmp, ceiling="max")
        orig = launcher._weg2_arena_ledger_terms
        launcher._weg2_arena_ledger_terms = lambda _m: dict(C_ARENA)
        try:
            return launcher.choose_host_ledger(
                0, 0, d_draft_host_gib=C_D_DRAFT_HOST_GIB, meminfo_path=meminfo,
                cgroup_root=cg, record_path=os.path.join(self.tmp, "none.json"),
                pin_m_mib=600, s_gb_d=4, model_dir=model,
                weights_cpu_backup_armed=False)
        finally:
            launcher._weg2_arena_ledger_terms = orig

    def test_a_foreign_checkpoint_is_not_priced_from_the_27b_references(self):
        """The NVFP4 flip boot with an empty sidecar: origin = launch reading,
        the ratchet series NAMED as a foreign stand-in, both on printed lines."""
        arm, _h, lines, _cg = self._seam(_make_ckpt(self.tmp, NF_NVFP4))
        t = arm.terms
        self.assertAlmostEqual(t["run_origin_gib"], (C_CG_CURRENT_B - C_RECLAIMABLE_B) / GIB,
                               places=2)
        self.assertIn(host_ledger.FOREIGN_REFERENCE_TAG, t["run_origin_source"])
        self.assertIn(host_ledger.FOREIGN_REFERENCE_TAG, t["flip_ratchet_source"])
        self.assertIn(host_ledger.FOREIGN_REFERENCE_TAG, t["image_p_source"])
        text = "\n".join(lines)
        self.assertIn("do NOT apply to this boot", text)

    @unittest.skipUnless(os.path.isdir(os.path.join(MODELS_CACHE, REF_27B)),
                         "reference checkpoint not on this box")
    def test_the_reference_checkpoint_keeps_its_own_references(self):
        arm, _h, lines, _cg = self._seam(os.path.join(MODELS_CACHE, REF_27B))
        self.assertAlmostEqual(arm.terms["run_origin_gib"], DK7_ORIGIN_GIB, places=1)
        self.assertNotIn(host_ledger.FOREIGN_REFERENCE_TAG, arm.terms["flip_ratchet_source"])
        self.assertIn("APPLY to this boot", "\n".join(lines))


# --------------------------------------------------------------- the key itself


class TestFootprintKey(_Tmp):

    def test_an_abliterated_derivative_is_the_same_footprint(self):
        base = _make_ckpt(self.tmp, "Base")
        cfg = dict(BASE_CONFIG, _name_or_path="/somewhere/else", transformers_version="4.99")
        abl = _make_ckpt(self.tmp, "Base-abl-wxp", config=cfg, payload=b"\x01" * 64)
        a, _ = weg2_form.footprint_key(base)
        b, _ = weg2_form.footprint_key(abl)
        self.assertIsNotNone(a)
        self.assertEqual(a, b)
        ok, why = weg2_form.reference_model_verdict(abl, "Base")
        self.assertTrue(ok, why)
        self.assertIn("DERIVATIVE", why)

    def test_a_requantisation_under_the_same_names_is_not(self):
        """g32 -> g128: same tensor NAMES and same text_config, so the names-only
        checkpoint_digest collides; the footprint does not."""
        base = _make_ckpt(self.tmp, "Base")
        t2 = {s: dict(ts) for s, ts in BASE_TENSORS.items()}
        t2["model-00001-of-00002.safetensors"][
            "model.layers.0.mlp.experts.0.down_proj.weight_scale"] = ("BF16", (8, 1))
        cfg = dict(BASE_CONFIG, quantization_config={"bits": 4, "group_size": 128})
        req = _make_ckpt(self.tmp, "Base-g128", config=cfg, tensors=t2)
        self.assertEqual(host_ledger.checkpoint_digest(base)[0],
                         host_ledger.checkpoint_digest(req)[0])
        self.assertNotEqual(weg2_form.footprint_key(base)[0], weg2_form.footprint_key(req)[0])

    def test_dtype_and_shape_and_quant_each_change_the_key(self):
        base = weg2_form.footprint_key(_make_ckpt(self.tmp, "Base"))[0]
        t_dtype = {s: dict(ts) for s, ts in BASE_TENSORS.items()}
        t_dtype["model-00002-of-00002.safetensors"]["lm_head.weight"] = ("F32", (16, 8))
        t_shape = {s: dict(ts) for s, ts in BASE_TENSORS.items()}
        t_shape["model-00002-of-00002.safetensors"]["lm_head.weight"] = ("BF16", (32, 8))
        keys = {
            weg2_form.footprint_key(_make_ckpt(self.tmp, "D", tensors=t_dtype))[0],
            weg2_form.footprint_key(_make_ckpt(self.tmp, "S", tensors=t_shape))[0],
            weg2_form.footprint_key(_make_ckpt(
                self.tmp, "Q", config=dict(BASE_CONFIG, quantization_config={"bits": 8})))[0],
        }
        self.assertEqual(len(keys), 3)
        self.assertNotIn(base, keys)

    def test_an_incomplete_checkpoint_is_unknown_not_same(self):
        d = _make_ckpt(self.tmp, "Base-abl-wxp")
        os.remove(os.path.join(d, "model-00002-of-00002.safetensors"))
        key, why = weg2_form.footprint_key(d)
        self.assertIsNone(key)
        self.assertIn("incomplete", why)
        _make_ckpt(self.tmp, "Base")
        self.assertFalse(weg2_form.reference_model_verdict(d, "Base")[0])

    def test_same_name_other_footprint_is_foreign(self):
        """A pinned reference under its own NAME but with other content (a
        directory re-filled with another quantisation) is not the reference."""
        pinned = weg2_form.footprint_key(_make_ckpt(os.path.join(self.tmp, "a"), "Model"))[0]
        refilled = _make_ckpt(self.tmp, "Model",
                              config=dict(BASE_CONFIG, quantization_config={"bits": 8}))
        saved = dict(weg2_form.REFERENCE_FOOTPRINTS)
        weg2_form.REFERENCE_FOOTPRINTS["Model"] = pinned
        try:
            ok, why = weg2_form.reference_model_verdict(refilled, "Model")
        finally:
            weg2_form.REFERENCE_FOOTPRINTS.clear()
            weg2_form.REFERENCE_FOOTPRINTS.update(saved)
        self.assertFalse(ok, why)
        self.assertIn("!=", why)

    def test_pinned_reference_footprints_match_the_live_checkpoints(self):
        seen = 0
        for name, key in weg2_form.REFERENCE_FOOTPRINTS.items():
            d = os.path.join(MODELS_CACHE, name)
            if not os.path.isdir(d):
                continue
            seen += 1
            self.assertEqual(weg2_form.footprint_key(d)[0], key, name)
        if not seen:
            self.skipTest("no reference checkpoint on this box")


# ------------------------------------------------- record + wake-credit selection


class TestRecordSelectionByFootprint(_Tmp):

    def _evidence(self, tag: str, model_path: str) -> str:
        ev = os.path.join(self.tmp, "evidence")
        os.makedirs(ev, exist_ok=True)
        with open(os.path.join(ev, f"boot_weg2_{tag}_0123456789_0925_120000.front.log"), "w") as f:
            f.write(f"group P argv: python -m sglang --model-path {model_path} "
                    f"--speculative-algorithm NEXTN\n")
        return ev

    def test_a_derivative_reads_its_base_models_records(self):
        base = _make_ckpt(self.tmp, "Base")
        abl = _make_ckpt(self.tmp, "Base-abl-wxp", payload=b"\x07" * 16)
        other = _make_ckpt(self.tmp, "Other",
                           config=dict(BASE_CONFIG, quantization_config={"bits": 8}))
        ev = self._evidence("baseboot", base)
        sample = {"boot_tag": "baseboot", "group": "P"}
        self.assertTrue(weg2_form.same_model_sample(abl, ev)(sample))
        self.assertTrue(weg2_form.same_model_sample(base, ev)(sample))
        self.assertFalse(weg2_form.same_model_sample(other, ev)(sample))

    def test_the_wake_credit_reference_follows_the_footprint(self):
        """A derivative keeps its base's reference; a foreign footprint under the
        reference key ENTFAELLT and says H87."""
        from sglang.srt.planner import expert_residency as er

        _make_ckpt(self.tmp, "Synth-Base")
        abl = _make_ckpt(self.tmp, "Synth-Base-abl", payload=b"\x02" * 8)
        other = _make_ckpt(self.tmp, "Synth-Other",
                           config=dict(BASE_CONFIG, quantization_config={"bits": 8}))
        split, chunk, n_layers, p_card = (29, 11, 8), 3, 48, (1, 0, 2)
        key = {"model": "Synth-Base", "p_split": split, "chunk_layers": chunk,
               "p_card": p_card, "d_ratio": "183,137,168"}
        kw = dict(p_split=split, chunk_layers=chunk, n_layers=n_layers, p_card=p_card,
                  d_ratio="183,137,168",
                  p_rows=[er.buffer_rows(local_experts=512, fraction=f, scratch_rows=32)
                          for f in (0.26, 0.45, 0.39)],
                  d_rows=[er.buffer_rows(local_experts=e, fraction=f, scratch_rows=s)
                          for e, f, s in zip((193, 145, 177), (0.207, 0.55, 0.45),
                                             (44, 48, 48))],
                  slot_mib=1297637376 / 512 / (1 << 20), label="D", reorder=True,
                  double_staging=False, reference_key=key)
        got = wc.plan_wake_credit(model=abl, **kw)
        self.assertNotIn("ENTFAELLT", got.lines[0])
        self.assertIn("H87", got.lines[0])
        bad = wc.plan_wake_credit(model=other, **kw)
        self.assertIn("ENTFAELLT", bad.lines[0])
        self.assertIn("H87", bad.lines[0])


NF_INT4 = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"


class TestCalibrationReferencesFollowTheFootprint(_Tmp):
    """The NF calibration references (#114/H41 prefill transient, D-rank and
    Karte-D references) were keyed by the directory NAME: an abliterated
    derivative lost them silently. With the INT4 reference's pinned footprint
    pointed at a synthetic base, a synthetic derivative must inherit them and a
    foreign footprint must not."""

    def setUp(self):
        super().setUp()
        base = _make_ckpt(self.tmp, "Synth-NF")
        self.abl = _make_ckpt(self.tmp, NF_INT4 + "-abl-wxp", payload=b"\x05" * 32)
        self.other = _make_ckpt(self.tmp, "Synth-Other",
                                config=dict(BASE_CONFIG, quantization_config={"bits": 8}))
        self._saved = dict(weg2_form.REFERENCE_FOOTPRINTS)
        weg2_form.REFERENCE_FOOTPRINTS[NF_INT4] = weg2_form.footprint_key(base)[0]

    def tearDown(self):
        weg2_form.REFERENCE_FOOTPRINTS.clear()
        weg2_form.REFERENCE_FOOTPRINTS.update(self._saved)
        launcher._CALIBRATION_FOOTPRINT_ALIAS.clear()
        super().tearDown()

    def _form(self, model_name: str):
        return weg2_form.Weg2Form(arch="moe", experts="offload", draft="mtp", p_draft="none",
                                  kv="qsa_forma", flip="family", vision="off",
                                  profile="nextflash", model=model_name)

    def test_the_prefill_transient_is_published_for_a_derivative(self):
        line = launcher.note_calibration_footprint_alias(self.abl)
        self.assertIsNotNone(line)
        self.assertIn("H87 CALIBRATION-IDENTITY", line)
        ok, why = launcher.p_prefill_transient_for(self._form(weg2_form.model_key(self.abl)))
        self.assertTrue(ok, why)
        self.assertIn("same memory footprint", why)

    def test_a_foreign_footprint_stays_unpublished_by_name(self):
        self.assertIsNone(launcher.note_calibration_footprint_alias(self.other))
        ok, why = launcher.p_prefill_transient_for(self._form("Synth-Other"))
        self.assertFalse(ok)
        self.assertIn("NOT published", why)

    def test_the_d_rank_and_card_references_apply_to_a_derivative(self):
        from sglang.srt.planner import expert_residency as er

        kw = dict(rank_tp_ratio="1,0,0", n_ranks=3, n_layers=48,
                  slot_bytes=1297637376 / 512)
        ref, why = er._reference_for(model_path=self.abl, reference_logs="", **kw)
        self.assertIsNotNone(ref, why)
        ref, why = er._reference_for(model_path=self.other, reference_logs="", **kw)
        self.assertIsNone(ref)
        card, why = er._card_reference_for(model_path=self.abl, card_reference_logs="", **kw)
        self.assertIsNotNone(card, why)
        card, why = er._card_reference_for(model_path=self.other, card_reference_logs="", **kw)
        self.assertIsNone(card)


if __name__ == "__main__":
    unittest.main()
