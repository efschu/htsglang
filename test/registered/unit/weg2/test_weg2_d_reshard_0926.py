"""--d-reshard (27B, 26.09.): D weight shards per load class.

Pinned without a GPU (design: /spinning/gpu-arb/docs/DYN_D_RESHARD.md):
  * NUMERICALLY EQUAL -- a column-parallel gate/up + row-parallel down MLP run
    under two different MLP family vectors gives the same output as the whole
    MLP (fp32: atol 2e-5 / rtol 1e-5, only the summation order of the
    all-reduce differs; bf16: max relative error 2e-2 of the output norm, the
    bf16 rounding of the three partial sums), and re-slicing the whole weight
    (what every P->D flip does) gives bit-identical shards for either vector;
  * RANKS NEVER DISAGREE -- the leader decides one row per wake epoch, every
    follower adopts it and derives the same tiling of the intermediate
    dimension; a missing, foreign, replayed or unknown row is a crash-stop;
  * THE MODEL -- the two-point fit reproduces both measured operating points,
    the 'dec' preset beats today's D at every decode class, a single preset
    costs no KV (capacity sum invariant), a second preset's spread lands on
    the rank that grows;
  * PROFILE -- nextflash (Form A) and unknown architectures are refused under
    wake/live, never ignored;
  * OFF IS IDENTICAL -- default 'off': no env, no argv, no line; a single
    preset ships exactly one --rank-mlp-ratio; more presets and 'live' refuse.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import torch

from sglang.srt.distributed.utils import partition_units
from sglang.srt.weg2 import d_reshard as D
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

INT8 = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"


def _shards(units_total: int, elems: int, vec):
    sizes = [u * elems for u in partition_units(units_total, list(vec))]
    offs, acc = [], 0
    for s in sizes:
        offs.append((acc, s))
        acc += s
    return offs


def _tp_mlp(x, gate, up, down, offs, dtype):
    """Emulated TP: rank r holds rows [o, o+s) of gate/up and cols of down;
    the row-parallel partials are summed (the all-reduce)."""
    parts = []
    for o, s in offs:
        g = x.to(dtype) @ gate[o:o + s].to(dtype).T
        u = x.to(dtype) @ up[o:o + s].to(dtype).T
        h = torch.nn.functional.silu(g) * u
        parts.append((h @ down[:, o:o + s].to(dtype).T).float())
    out = parts[0]
    for p in parts[1:]:
        out = out + p
    return out


class TestNumericEquality(unittest.TestCase):
    H, I, UNITS = 64, 256, 16  # 16 units of 16 elements (the INT8 MLP unit)

    def setUp(self):
        g = torch.Generator().manual_seed(7)
        self.x = torch.randn(5, self.H, generator=g)
        self.gate = torch.randn(self.I, self.H, generator=g) / 8
        self.up = torch.randn(self.I, self.H, generator=g) / 8
        self.down = torch.randn(self.H, self.I, generator=g) / 16
        self.full = (torch.nn.functional.silu(self.x @ self.gate.T) * (self.x @ self.up.T)) @ self.down.T

    def test_fp32_two_vectors_equal_whole(self):
        elems = self.I // self.UNITS
        a = _tp_mlp(self.x, self.gate, self.up, self.down, _shards(self.UNITS, elems, (58, 25, 25)), torch.float32)
        b = _tp_mlp(self.x, self.gate, self.up, self.down, _shards(self.UNITS, elems, (11, 3, 2)), torch.float32)
        torch.testing.assert_close(a, self.full, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(b, self.full, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-5)

    def test_bf16_tolerance(self):
        elems = self.I // self.UNITS
        ref = self.full
        for vec in ((58, 25, 25), (11, 3, 2), (6, 5, 5)):
            out = _tp_mlp(self.x, self.gate, self.up, self.down, _shards(self.UNITS, elems, vec), torch.bfloat16)
            rel = float((out - ref).norm() / ref.norm())
            self.assertLess(rel, 2e-2, vec)

    def test_reslice_from_whole_is_bit_exact(self):
        """The flip writes every D shard from the whole P tensor: preset B's
        shards re-cut from the whole equal the direct cut, byte for byte."""
        elems = self.I // self.UNITS
        a = _shards(self.UNITS, elems, (58, 25, 25))
        b = _shards(self.UNITS, elems, (11, 3, 2))
        whole = torch.cat([self.gate[o:o + s] for o, s in a])  # reassemble from preset A
        self.assertTrue(torch.equal(whole, self.gate))
        for o, s in b:
            self.assertTrue(torch.equal(whole[o:o + s], self.gate[o:o + s]))

    def test_shard_map_tiles_intermediate(self):
        for fmt in ("int8", "nvfp4"):
            g = D.rc9_geometry(fmt)
            spec = D.ReshardSpec("wake", D.RC9_BASE, D.rc9_presets(fmt, ("dec", "pf", "rc9")), "dec", 0.02, fmt)
            for p in spec.presets:
                m = D.shard_map(g, spec, p.name)["mlp.intermediate"]
                self.assertEqual(m[0][0], 0)
                for (o1, s1), (o2, _) in zip(m, m[1:]):
                    self.assertEqual(o1 + s1, o2)
                self.assertEqual(m[-1][0] + m[-1][1], g.inter)
                self.assertTrue(all(s % 16 == 0 for _, s in m))


class TestRanksAgree(unittest.TestCase):
    def _mk(self, policy="wake", names=("dec", "pf"), min_gain=0.01):
        g, cal = D.rc9_geometry("int8"), D.rc9_calib("int8")
        spec = D.ReshardSpec(policy, D.RC9_BASE, D.rc9_presets("int8", names), names[0], min_gain, "int8")
        ts = D.token_share(D.RC9_TOKEN_VECTOR["int8"])
        return g, spec, D.LeaderCursor(g, cal, spec, ts), [D.FollowerCursor(g, spec, r) for r in range(3)]

    def test_all_followers_adopt_one_row(self):
        g, spec, lead, fol = self._mk()
        loads = [D.LoadClass("decode", 1, 32768), D.LoadClass("prefill", 1, 4096),
                 D.LoadClass("decode", 6, 131072), D.LoadClass("decode", 1, 2048)]
        seen = []
        for e, ld in enumerate(loads):
            row = lead.decide(e, ld)
            cuts = [f.adopt(row) for f in fol]
            self.assertEqual({f.current for f in fol}, {row.preset})
            self.assertEqual(cuts, D.shard_map(g, spec, row.preset)["mlp.intermediate"])
            seen.append(row.preset)
        # the D-prefill epoch takes the concentrated preset (INT8 +1.7 % > 1 %
        # hysteresis), the next bs=1 epoch goes back
        self.assertEqual((seen[0], seen[1], seen[3]), ("dec", "pf", "dec"))

    def test_divergence_is_crash_stop(self):
        g, spec, lead, fol = self._mk()
        row = lead.decide(0, D.LoadClass("decode", 1, 2048))
        fol[0].adopt(row)
        with self.assertRaises(D.ReshardDivergence):
            fol[0].adopt(row)                                   # replay
        with self.assertRaises(D.ReshardDivergence):
            fol[1].adopt(None)                                  # missing
        with self.assertRaises(D.ReshardDivergence):
            fol[1].adopt(D.ReshardRow(1, "dec", "deadbeefdeadbeef"))   # foreign spec
        with self.assertRaises(D.ReshardDivergence):
            fol[2].adopt(D.ReshardRow(1, "nope", spec.digest()))       # unknown preset
        with self.assertRaises(D.ReshardDivergence):
            lead.decide(0, D.LoadClass("decode", 1, 2048))      # leader epoch not advancing

    def test_off_always_boot(self):
        _, spec, lead, fol = self._mk(policy="off")
        for e in range(3):
            row = lead.decide(e, D.LoadClass("prefill", 1, 4096))
            self.assertEqual(row.preset, spec.boot)

    def test_spec_json_roundtrip_digest(self):
        _, spec, _, _ = self._mk()
        back = D.ReshardSpec.from_json(spec.to_json())
        self.assertEqual(back, spec)
        self.assertEqual(back.digest(), spec.digest())


class TestModel(unittest.TestCase):
    def test_fit_reproduces_both_points(self):
        g, cal = D.rc9_geometry("int8"), D.rc9_calib("int8")
        ld = D.LoadClass("decode", 1, 0)
        a = D.decode_compute_ms(g, cal, D.Vector(D.RC9_BASE), ld, (1 / 3,) * 3)
        b = D.decode_compute_ms(g, cal, D.Vector(D.XSN420_BASE), ld, (1 / 3,) * 3)
        self.assertAlmostEqual(a[0], 18.3, places=6)
        self.assertAlmostEqual(b[0], 16.3, places=6)
        self.assertAlmostEqual((a[1] + a[2]) / 2, 21.85, places=6)
        self.assertAlmostEqual((b[1] + b[2]) / 2, 23.4, places=6)

    def test_dec_beats_rc9_every_decode_class(self):
        for fmt in ("int8", "nvfp4"):
            g, cal = D.rc9_geometry(fmt), D.rc9_calib(fmt)
            dec = D.rc9_presets(fmt, ("dec",))[0].vector(D.RC9_BASE)
            ts = D.token_share(D.RC9_TOKEN_VECTOR[fmt])
            for bs in (1, 2, 4, 6):
                for ctx in (2048, 32768, 131072):
                    ld = D.LoadClass("decode", bs, ctx)
                    self.assertLess(D.round_ms(g, cal, dec, ld, ts), D.round_ms(g, cal, D.Vector(D.RC9_BASE), ld, ts))

    def test_single_preset_costs_no_kv_second_costs_rank0(self):
        for fmt in ("int8", "nvfp4"):
            one = D.ReshardSpec("wake", D.RC9_BASE, D.rc9_presets(fmt, ("dec",)), "dec", 0.02, fmt)
            self.assertEqual(sum(D.boot_capacity(fmt, one)), sum(D.RC9_KV_CAPACITY[fmt]))
            two = D.ReshardSpec("wake", D.RC9_BASE, D.rc9_presets(fmt, ("dec", "pf")), "dec", 0.02, fmt)
            sp = D.spread_bytes(D.rc9_geometry(fmt), D.RC9_BASE, two.presets, two.presets[0])
            self.assertGreater(sp[0], 1e9)
            self.assertEqual(sp[1:], [0.0, 0.0])
            self.assertLess(sum(D.boot_capacity(fmt, two)), sum(D.RC9_KV_CAPACITY[fmt]))

    def test_hysteresis_keeps_current(self):
        g, cal = D.rc9_geometry("int8"), D.rc9_calib("int8")
        ps = D.rc9_presets("int8", ("dec", "pf"))
        ts = D.token_share(D.RC9_TOKEN_VECTOR["int8"])
        ld = D.LoadClass("decode", 4, 32768)  # dec and pf within a few % here
        self.assertEqual(D.choose(g, cal, D.RC9_BASE, ps, ld, ts, "pf", min_gain=0.10), "pf")
        self.assertEqual(D.choose(g, cal, D.RC9_BASE, ps, ld, ts, "pf", min_gain=0.0), "dec")

    def test_validate_refuses_bad_presets(self):
        g = D.rc9_geometry("int8")
        with self.assertRaises(D.ReshardError):
            D.ReshardSpec("wake", D.RC9_BASE, (D.Preset("a", (1, 2)),), "a").validate(g)
        with self.assertRaises(D.ReshardError):
            D.ReshardSpec("wake", D.RC9_BASE, D.rc9_presets("int8", ("dec",)), "pf").validate(g)
        with self.assertRaises(D.ReshardError):
            D.parse_presets("x=1:2:zz", "int8")


class TestProfile(unittest.TestCase):
    def test_nf_refused_27b_supported(self):
        self.assertEqual(D.PROFILE_SUPPORT["nextflash"], D.UNSUPPORTED)
        self.assertEqual(D.PROFILE_SUPPORT["qwen27b"], D.SUPPORTED)
        for pol in ("wake", "live"):
            with self.assertRaises(D.ReshardProfileRefused):
                D.check_profile("nextflash", pol)
            with self.assertRaises(D.ReshardProfileRefused):
                D.check_profile(None, pol)
            D.check_profile("qwen27b", pol)
        D.check_profile("nextflash", "off")  # off never refuses

    def test_profile_of_config(self):
        self.assertEqual(D.profile_of_config({"text_config": {"model_type": "qwen3_5_text"}}), "qwen27b")
        self.assertEqual(D.profile_of_config({"model_type": "qwen3_next", "num_experts": 512}), "nextflash")
        self.assertIsNone(D.profile_of_config({"model_type": "llama"}))


class TestLauncher(unittest.TestCase):
    def setUp(self):
        from sglang.srt.weg2 import launcher as L
        self.L = L
        self.tmp = tempfile.mkdtemp()

    def _model(self, cfg):
        d = tempfile.mkdtemp(dir=self.tmp)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg, f)
        return d

    def _ns(self, *extra, model=None):
        base = ["--tree", "/t", "--tag", "t"] + (["--model", model] if model else [])
        return self.L.build_parser().parse_args(base + list(extra))

    def _int8(self):
        return self._model({"text_config": dict(D._QWEN38_27B),
                            "quantization_config": {"quant_method": "compressed-tensors"}})

    def test_default_off_identical(self):
        ns = self._ns()
        self.assertEqual(ns.d_reshard, "off")
        self.L.apply_d_reshard(ns)
        self.assertEqual(self.L.d_reshard_env(), {})
        self.assertEqual(self.L.d_reshard_argv(("--rank-tp-ratio", "58,25,25"), []), [])
        self.assertEqual(self.L.d_reshard_lines(), ())

    def test_nf_profile_refused(self):
        nf = self._model({"model_type": "qwen3_next", "num_experts": 512,
                          "quantization_config": {"quant_method": "compressed-tensors"}})
        with self.assertRaises(SystemExit) as cm:
            self.L.apply_d_reshard(self._ns("--d-reshard", "wake", model=nf))
        self.assertIn("nextflash", str(cm.exception.code))
        self.assertIn("REFUSED", str(cm.exception.code))
        self.assertEqual(self.L.d_reshard_env(), {})

    def test_single_preset_ships_mlp_ratio(self):
        self.L.apply_d_reshard(self._ns("--d-reshard", "wake", model=self._int8()))
        argv = self.L.d_reshard_argv(("--rank-tp-ratio", "58,25,25"), [])
        self.assertEqual(argv, ["--rank-mlp-ratio", "707,191,190"])
        env = self.L.d_reshard_env()
        self.assertEqual(env[D.POLICY_ENV], "wake")
        self.assertEqual(D.ReshardSpec.from_json(env[D.SPEC_ENV]).presets[0].mlp, (707, 191, 190))
        self.assertTrue(any("D-RESHARD armed" in x for x in self.L.d_reshard_lines()))
        with self.assertRaises(SystemExit):   # presets built on another base vector
            self.L.d_reshard_argv(("--rank-tp-ratio", "2,1,1"), [])
        with self.assertRaises(SystemExit):   # two authors of one vector
            self.L.d_reshard_argv(("--rank-tp-ratio", "58,25,25"), ["--rank-mlp-ratio", "1,1,1"])
        self.L.apply_d_reshard(self._ns())    # back to off: nothing left behind
        self.assertEqual(self.L.d_reshard_env(), {})

    def test_multi_preset_and_live_refuse(self):
        m = self._int8()
        for extra in (("--d-reshard", "wake", "--d-reshard-presets", "dec,pf"), ("--d-reshard", "live")):
            with self.assertRaises(SystemExit) as cm:
                self.L.apply_d_reshard(self._ns(*extra, model=m))
            msg = str(cm.exception.code)
            self.assertIn("D-RESHARD plan", msg)
            self.assertIn("REFUSED", msg)
            self.assertEqual(self.L.d_reshard_env(), {})

    def test_unknown_format_refused(self):
        m = self._model({"text_config": dict(D._QWEN38_27B)})
        with self.assertRaises(SystemExit) as cm:
            self.L.apply_d_reshard(self._ns("--d-reshard", "wake", model=m))
        self.assertIn("no D cost model", str(cm.exception.code))


class TestDcpSpeedAdvisory(unittest.TestCase):
    """Under uneven DCP the KV restart hint is replaced by the speed table."""

    def test_table_rows_and_wake_vector(self):
        for qm, units, tv, wake in (("compressed-tensors", 1088, (26, 19, 19), "707,191,190"),
                                    ("modelopt", 136, (14, 9, 9), "98,19,19")):
            cur = partition_units(units, [58, 25, 25])
            lines = D.speed_advisory_lines(D._QWEN38_27B, qm, (58, 25, 25), cur, units, tv)
            self.assertIn("KV sum is conserved", lines[0])
            self.assertIn(f"--d-reshard wake takes mlp={wake}", lines[0])  # = the launcher's one preset
            rows = [x for x in lines if "D-SPEED" in x]
            self.assertEqual(len(rows), 4 * 3 + 1)
            self.assertTrue(any("prefill-4k" in x for x in rows))
            for r in rows:
                gain = float(r.split("gain=")[1].split("%")[0])
                wake_gain = float(r.split("wake mlp")[1].split("%")[0])
                self.assertGreaterEqual(gain, wake_gain - 1e-9)   # the class optimum is never worse
            fmt = D.advisory_format(qm)
            self.assertEqual(",".join(map(str, D.rc9_presets(fmt, ("dec",))[0].mlp)), wake)

    def test_running_optimum_gains_zero(self):
        cur = D.rc9_presets("int8", ("dec",))[0].mlp
        lines = D.speed_advisory_lines(D._QWEN38_27B, "compressed-tensors", (58, 25, 25), cur, 1088, (26, 19, 19))
        row = [x for x in lines if "decode-bs1-2k" in x][0]
        self.assertIn("gain= +0.0%", row)

    def test_unmodelled_boots_say_why(self):
        for args in (("gguf", (58, 25, 25)), ("compressed-tensors", (2, 1)),):
            qm, base = args
            cur = partition_units(68, list(base))
            lines = D.speed_advisory_lines(D._QWEN38_27B, qm, base, cur, 68, None)
            self.assertEqual(len(lines), 1)
            self.assertIn("No speed table", lines[0])
        other = dict(D._QWEN38_27B, hidden_size=4096)
        lines = D.speed_advisory_lines(other, "compressed-tensors", (58, 25, 25),
                                       partition_units(1088, [58, 25, 25]), 1088, None)
        self.assertIn("No speed table", lines[0])

    def test_predicate_rank_uniform(self):
        self.assertTrue(D.uneven_dcp_token_split(3, 3, True, False))
        self.assertTrue(D.uneven_dcp_token_split(3, 3, False, True))
        self.assertFalse(D.uneven_dcp_token_split(1, 3, True, True))   # no DCP: old KV hint stays
        self.assertFalse(D.uneven_dcp_token_split(3, 3, False, False))
        self.assertFalse(D.uneven_dcp_token_split(1, 1, True, True))

    def test_quant_method_of(self):
        class Cfg:
            quantization_config = {"quant_method": "modelopt"}
        self.assertEqual(D.quant_method_of(Cfg()), "modelopt")
        self.assertEqual(D.quant_method_of(Cfg(), "gguf"), "gguf")

    def test_runner_branch_skips_kv_hint_collective(self):
        """The mixin under uneven DCP: speed table on rank 0, True on every rank,
        never the all_gather of the KV hint; without DCP it falls through."""
        import types
        from sglang.srt.distributed import utils as U
        from sglang.srt.model_executor import model_runner_kv_cache_mixin as M

        class Lin(torch.nn.Module):
            tp_family, tp_units = "mlp", 1088

        model = torch.nn.Sequential(Lin())
        sa = types.SimpleNamespace(uneven_dcp=True, quantization=None,
                                   uneven_memory_budgets_active=lambda: True)
        mc = types.SimpleNamespace(hf_text_config=dict(D._QWEN38_27B),
                                   hf_config=types.SimpleNamespace(
                                       quantization_config={"quant_method": "compressed-tensors"}))
        logged = []
        orig_info = M.logger.info
        M.logger.info = lambda msg, *a: logged.append(msg % a if a else msg)
        try:
            with U.scoped_tp_partition_ratios([58, 25, 25]):
                for rank in (0, 1):
                    r = types.SimpleNamespace(server_args=sa, dcp_size=3, tp_size=3, tp_rank=rank,
                                              is_draft_worker=False, model=model, model_config=mc)
                    self.assertTrue(M._dcp_speed_advisory(r))
                nodcp = types.SimpleNamespace(server_args=sa, dcp_size=1, tp_size=3, tp_rank=0,
                                              is_draft_worker=False, model=model, model_config=mc)
                self.assertFalse(M._dcp_speed_advisory(nodcp))
        finally:
            M.logger.info = orig_info
        self.assertTrue(any("D-SPEED decode-bs1-32k" in x for x in logged))
        self.assertEqual(sum("KV sum is conserved" in x for x in logged), 1)   # rank 0 only


if __name__ == "__main__":
    unittest.main()
