"""Release table row 27 (26.09.): the power limit as a field of the P stage
model, the chunk-model JSON, the D speed table and every boot record.

Pinned without a GPU (NVML mocked throughout):
  * the NVML reader (mW -> W, SM clock max, an unanswered query is None, the
    replay seam with and without the new fields);
  * the rank's POWER-LIMIT boot line (UUID mask -> the right card, NVML
    failure never raises) and its parser (the evaluators' contract);
  * the check: > 5 % = loud MISMATCH, <= 5 % = ok, an old JSON without the
    field = a warning (never a refusal), unknown running limit = a warning;
  * the scaling law (linear / power / off) on COMPUTE terms only (widths
    below 256 and host-launch floors unscaled), labelled as a model;
  * the stage model and the pchunk export carry power_limit_w; an old model
    exports byte-identically;
  * the launcher: 'fixed' touches no NVML, 'dynamic' without the flag ships a
    byte-identical spec (only lines added), with the flag it rescales; the
    boot record gets the four fields per card;
  * the D speed table names its limit on every row and warns on a mismatch.
"""

from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from unittest import mock

from sglang.srt.registry import nvml as N
from sglang.srt.weg2 import d_reshard as D
from sglang.srt.weg2 import p_chunk_policy as P
from sglang.srt.weg2 import p_stage_model as M
from sglang.srt.weg2 import power_limit as PL
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

try:
    from sglang.srt.weg2 import launcher as L
except Exception:  # pragma: no cover
    L = None

DATA = os.path.join(os.path.dirname(M.__file__), "p_stage_model_data")
U5090 = "GPU-aaaaaaaa-5090"
U3080A = "GPU-bbbbbbbb-3080a"
U3080B = "GPU-cccccccc-3080b"


class _FakeNvml(types.SimpleNamespace):
    """A pynvml stand-in: three cards in NVML order 3080, 5090, 3080."""

    NVML_CLOCK_SM = 1

    def __init__(self, limits=(230, 400, 230), fail=()):
        super().__init__()
        self.cards = [("NVIDIA GeForce RTX 3080", U3080A, limits[0], 320, 2100),
                      ("NVIDIA GeForce RTX 5090", U5090, limits[1], 600, 3090),
                      ("NVIDIA GeForce RTX 3080", U3080B, limits[2], 320, 2100)]
        self.fail = set(fail)

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetCount(self):
        return len(self.cards)

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetUUID(self, h):
        return self.cards[h][1].encode()

    def nvmlDeviceGetName(self, h):
        return self.cards[h][0]

    def _maybe(self, what):
        if what in self.fail:
            raise RuntimeError(f"NVML_ERROR_NOT_SUPPORTED {what}")

    def nvmlDeviceGetEnforcedPowerLimit(self, h):
        self._maybe("enforced")
        return int(self.cards[h][2] * 1000)

    def nvmlDeviceGetPowerManagementDefaultLimit(self, h):
        self._maybe("default")
        return int(self.cards[h][3] * 1000) - 25000 * (h == 1)

    def nvmlDeviceGetPowerManagementLimitConstraints(self, h):
        self._maybe("constraints")
        return (100000, int(self.cards[h][3] * 1000))

    def nvmlDeviceGetMaxClockInfo(self, h, clock):
        self._maybe("clock")
        assert clock == self.NVML_CLOCK_SM
        return self.cards[h][4]


def _patch_nvml(fake):
    return mock.patch.object(N, "_pynvml", return_value=fake)


class TestNvmlReader(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {N.ENV_NVML_REPLAY: ""})
        self._env.start()

    def tearDown(self):
        self._env.stop()

    def test_snapshot_converts_and_keys_by_uuid(self):
        with _patch_nvml(_FakeNvml()):
            snap = N.power_snapshot()
            self.assertEqual([p.uuid for p in snap], [U3080A, U5090, U3080B])
            p = N.power_info_for_uuid(U5090)
        self.assertEqual((p.power_limit_w, p.power_limit_default_w, p.power_limit_max_w, p.sm_clock_max_mhz),
                         (400.0, 575.0, 600.0, 3090))
        self.assertEqual(p.index, 1)
        self.assertEqual(PL.card_class(p.name), "RTX5090")

    def test_unanswered_query_is_none_not_zero(self):
        with _patch_nvml(_FakeNvml(fail={"enforced", "clock"})):
            p = N.power_info_for_uuid(U3080A)
        self.assertIsNone(p.power_limit_w)
        self.assertIsNone(p.sm_clock_max_mhz)
        self.assertEqual(p.power_limit_max_w, 320.0)

    def test_unknown_uuid_raises(self):
        with _patch_nvml(_FakeNvml()):
            with self.assertRaises(N.DeviceNotFoundError):
                N.power_info_for_uuid("GPU-nope")

    def test_replay_with_and_without_fields(self):
        rows = [{"index": 0, "uuid": U3080A, "name": "NVIDIA GeForce RTX 3080", "total_bytes": 1},
                {"index": 1, "uuid": U5090, "name": "NVIDIA GeForce RTX 5090", "total_bytes": 1,
                 "power_limit_w": 400, "sm_clock_max_mhz": 3090}]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(rows, fh)
        try:
            with mock.patch.dict(os.environ, {N.ENV_NVML_REPLAY: fh.name}):
                snap = N.power_snapshot()
        finally:
            os.unlink(fh.name)
        self.assertIsNone(snap[0].power_limit_w)   # an old recording does not know the limit
        self.assertEqual((snap[1].power_limit_w, snap[1].sm_clock_max_mhz), (400.0, 3090))


class TestBootLine(unittest.TestCase):
    def test_rank_line_resolves_the_card_through_the_uuid_mask(self):
        env = {"CUDA_VISIBLE_DEVICES": f"{U5090},{U3080A},{U3080B}", N.ENV_NVML_REPLAY: ""}
        with mock.patch.dict(os.environ, env), _patch_nvml(_FakeNvml()):
            l0 = PL.rank_boot_line(0, 0, 0)
            l1 = PL.rank_boot_line(0, 1, 1)
        self.assertIn("card=RTX5090 power_limit_w=400 ", l0)
        self.assertIn("sm_clock_max_mhz=3090", l0)
        self.assertIn(f"uuid={U5090}", l0)
        self.assertIn("card=RTX3080 power_limit_w=230 ", l1)
        self.assertTrue(l0.startswith("POWER-LIMIT rank tp_rank=0 pp_rank=0 gpu_id=0 "))

    def test_rank_line_never_raises(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": U5090, N.ENV_NVML_REPLAY: ""}), \
                mock.patch.object(N, "_pynvml", side_effect=N.NvmlUnavailableError("no driver")):
            ln = PL.rank_boot_line(0, 0, 0)
        self.assertIn("power_limit_w=NA unavailable reason=NvmlUnavailableError", ln)

    def test_parse_roundtrip_with_prefix_and_without(self):
        info = N.PowerInfo(U5090, "NVIDIA GeForce RTX 5090", 1, 400.0, 575.0, 600.0, 3090)
        lines = [
            "[2026-09-26 12:00:00 PP0] " + PL.boot_line({"tp_rank": 0, "pp_rank": 0, "gpu_id": 0}, info),
            "[2026-09-26 12:00:01 TP2] " + PL.boot_line({"tp_rank": 2, "pp_rank": 0, "gpu_id": 2}, None, "x"),
            PL.boot_line({"tp_rank": 1, "pp_rank": 0, "gpu_id": 1}, info),
            "[2026-09-26 12:00:02 PP1] unrelated line power_limit_w=999",
        ]
        got = PL.parse_boot_lines(lines)
        self.assertEqual(set(got), {"PP0", "TP2", "tp1pp0"})
        self.assertEqual(got["PP0"]["power_limit_w"], 400.0)
        self.assertEqual(got["PP0"]["sm_clock_max_mhz"], 3090.0)
        self.assertEqual(got["PP0"]["card"], "RTX5090")
        self.assertIsNone(got["TP2"]["power_limit_w"])


class TestCheckAndLaw(unittest.TestCase):
    def test_factor_laws(self):
        self.assertEqual(PL.compute_factor(400, 600, "off"), 1.0)
        self.assertAlmostEqual(PL.compute_factor(400, 600, "linear"), 400 / 600)
        self.assertAlmostEqual(PL.compute_factor(400, 600, "power"), (400 / 600) ** (1 / 3))
        self.assertAlmostEqual(PL.compute_factor(400, 600, "power", 0.5), (400 / 600) ** 0.5)
        self.assertEqual(PL.compute_factor(None, 600, "linear"), 1.0)
        with self.assertRaises(PL.PowerLimitError):
            PL.exponent_of("cubic")
        with self.assertRaises(PL.PowerLimitError):
            PL.exponent_of("power", 0.0)

    def test_tolerance_old_json_and_unknown(self):
        vs = PL.check(["a", "b", "c", "d"], [400, 230, None, 230], [410, 260, 400, None], "linear")
        self.assertFalse(vs[0].mismatch)             # +2.5 %: inside 5 %
        self.assertEqual(vs[0].factor, 1.0)
        self.assertTrue(vs[1].mismatch)              # +13 %
        self.assertAlmostEqual(vs[1].factor, 230 / 260)
        self.assertEqual((vs[2].factor, vs[3].factor), (1.0, 1.0))
        lines = PL.verdict_lines("t", vs, "linear")
        self.assertIn("ok t a", lines[0])
        self.assertIn("MISMATCH !!! t b", lines[1])
        self.assertIn("x0.8846", lines[1])
        self.assertIn("old JSON", lines[2])
        self.assertIn("not refused", lines[2])
        self.assertIn("running limit UNKNOWN", lines[3])
        self.assertIn(PL.MODEL_LABEL, lines[-1])
        # without a law: loud, but nothing rescaled and no SCALE line
        off = PL.check(["b"], [230], [260])
        self.assertEqual(off[0].factor, 1.0)
        self.assertIn("NOT rescaled (flag off)", PL.verdict_lines("t", off)[0])
        self.assertEqual(len(PL.verdict_lines("t", off)), 1)

    def test_scale_stage_model_compute_only(self):
        st = P.StageModel(((16, 7.0), (256, 30.0), (512, 60.0)), 0.003, 63.0, "PP0",
                          eager_points=((512, 70.0),), attn_points=((512, 0.004),),
                          eager_attn_points=((128, 0.001), (512, 0.005)))
        sc = PL.scale_stage_model(st, 0.5)
        self.assertEqual(sc.points, ((16, 7.0), (256, 15.0), (512, 30.0)))   # 16: bandwidth, unscaled
        self.assertEqual(sc.eager_points, ((512, 35.0),))
        self.assertAlmostEqual(sc.attn_ms_per_tok_1k, 0.0015)
        self.assertEqual(sc.attn_points, ((512, 0.002),))
        self.assertEqual(sc.eager_attn_points, ((128, 0.001), (512, 0.0025)))
        self.assertEqual(sc.eager_floor_ms, 63.0)                                # host launch
        self.assertIs(PL.scale_stage_model(st, 1.0), st)

    def test_stage_limits_from_json_shapes(self):
        self.assertIsNone(PL.stage_limits_from_json({"stages": []}, 3))
        self.assertEqual(PL.stage_limits_from_json({"power_limit_w": [400, 230, 230]}, 3), [400.0, 230.0, 230.0])
        self.assertEqual(PL.stage_limits_from_json({"power_limit_w": {"PP0": 400, "PP2": 230}}, 3),
                         [400.0, None, 230.0])
        self.assertEqual(PL.stage_limits_from_json(
            {"power_limit_w": {"RTX5090": 400, "RTX3080": 230},
             "power_limit_cards": ["RTX5090", "RTX3080", "RTX3080"]}, 3), [400.0, 230.0, 230.0])
        with self.assertRaises(PL.PowerLimitError):
            PL.stage_limits_from_json({"power_limit_w": [400]}, 3)


class TestStageModel(unittest.TestCase):
    def _old(self):
        with open(os.path.join(DATA, "27b_int8_rc9j.json")) as fh:
            d = json.load(fh)
        d.pop("power_limit_w")
        d.pop("power_limit_source")
        return d

    def test_committed_models_carry_the_field(self):
        for fmt, cut in (("int8", "41-12-11"), ("int8", "42-11-11"), ("nvfp4", "49-8-7")):
            m = M.load_model(os.path.join(DATA, f"27b_{fmt}_rc9j.json"))
            self.assertEqual(m.power_limit_w, {"RTX5090": 400.0, "RTX3080": 230.0})
            self.assertEqual(m.stage_power_limits(), [400.0, 230.0, 230.0])
            with open(os.path.join(DATA, f"27b_{fmt}_rc9j_cut{cut}.pchunk.json")) as fh:
                pc = json.load(fh)
            self.assertEqual(pc["power_limit_w"], [400.0, 230.0, 230.0])
            doc = M.pchunk_json(m, tuple(int(x) for x in cut.split("-")))
            self.assertEqual((doc["power_limit_w"], doc["stages"]), (pc["power_limit_w"], pc["stages"]))

    def test_old_model_loads_and_exports_byte_identically(self):
        old = self._old()
        m = M.LayerCostModel.from_json(old)
        self.assertEqual(m.power_limit_w, {})
        self.assertNotIn("power_limit_w", m.to_json())
        doc = M.pchunk_json(m, (41, 12, 11))
        self.assertEqual(set(doc), {"stages", "source"})
        # the check on an old model warns per card and changes nothing
        m2, lines = M.check_power(m, {"RTX5090": 400.0, "RTX3080": 230.0}, "linear")
        self.assertIs(m2, m)
        self.assertEqual(sum("old JSON" in x for x in lines), 2)

    def test_mismatch_rescales_gemm_and_attn_under_a_law_only(self):
        m = M.load_model(os.path.join(DATA, "27b_int8_rc9j.json"))
        cur = {"RTX5090": 600.0, "RTX3080": 230.0}
        same, lines = M.check_power(m, cur, "off")
        self.assertIs(same, m)
        self.assertTrue(any("MISMATCH !!!" in x and "RTX5090" in x for x in lines))
        sc, lines = M.check_power(m, cur, "linear")
        f = 400.0 / 600.0
        g0, g1 = m.cards["RTX5090"].gemm(1024, M.MODE_EAGER), sc.cards["RTX5090"].gemm(1024, M.MODE_EAGER)
        self.assertAlmostEqual(g1, g0 * f, places=3)
        self.assertAlmostEqual(sc.cards["RTX5090"].attn_coeff(512, M.MODE_GRAPH),
                               m.cards["RTX5090"].attn_coeff(512, M.MODE_GRAPH) * f, places=6)
        self.assertEqual(sc.cards["RTX3080"], m.cards["RTX3080"])            # inside tolerance
        self.assertEqual(sc.power_limit_w, {"RTX5090": 600.0, "RTX3080": 230.0})
        self.assertIn(PL.MODEL_LABEL, sc.source)
        self.assertTrue(any(x.startswith("POWER-LIMIT SCALE") for x in lines))
        # the rescaled cut moves toward the faster card, as the physics says
        self.assertLess(sc.stage_ms((41, 12, 11), 0, 1024, 0, M.MODE_EAGER),
                        m.stage_ms((41, 12, 11), 0, 1024, 0, M.MODE_EAGER))

    def test_cli_names_the_limit_on_stderr(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "o.json")
            with mock.patch("sys.stderr") as err:
                M.main([os.path.join(DATA, "27b_nvfp4_rc9j.json"), "--cut", "49,8,7", "--out", out,
                        "--current-power-w", "RTX5090=575,RTX3080=320", "--power-scale", "power"])
            text = "".join(c.args[0] for c in err.write.call_args_list)
            with open(out) as fh:
                doc = json.load(fh)
        self.assertIn("MISMATCH !!!", text)
        self.assertEqual(doc["power_limit_w"], [575.0, 320.0, 320.0])
        self.assertIn("powerscale:law=power", doc["source"])


@unittest.skipIf(L is None, "weg2 launcher unavailable")
class TestLauncher(unittest.TestCase):
    def setUp(self):
        self._g = dict(L._P_PREFILL_GRAPH)
        self._c = dict(L._P_CHUNK)
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        L._P_PREFILL_GRAPH.clear()
        L._P_PREFILL_GRAPH.update(self._g)
        L._P_CHUNK.clear()
        L._P_CHUNK.update(self._c)
        self._tmp.cleanup()

    def _apply(self, *extra, current=(400.0, 230.0, 230.0)):
        ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t", *extra])
        with mock.patch.object(L, "p_stage_power_current",
                               return_value=(list(current), ["PP0", "PP1", "PP2"])) as cur:
            L.apply_p_prefill_graph(ns)
            L.apply_p_chunk_policy(ns)
        return ns, cur

    def _json(self, obj):
        path = os.path.join(self._tmp.name, "m.json")
        with open(path, "w") as fh:
            json.dump(obj, fh)
        return path

    def test_parser_default_off(self):
        ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
        self.assertEqual((ns.p_power_scale, ns.p_power_scale_exponent), ("off", None))

    def test_fixed_reads_no_nvml(self):
        _, cur = self._apply("--p-prefill-graph", "512")
        cur.assert_not_called()
        self.assertEqual((L.p_chunk_power_lines(), L.p_chunk_policy_env()), ([], {}))

    def test_dynamic_without_flag_ships_the_same_spec(self):
        with open(os.path.join(DATA, "27b_int8_rc9j_cut41-12-11.pchunk.json")) as fh:
            d = json.load(fh)
        new = self._json(d)
        self._apply("--p-prefill-graph", "512", "--p-chunk-policy", "dynamic", "--p-chunk-model", new,
                    current=(575.0, 320.0, 320.0))
        spec_new = L.p_chunk_policy_env()[P.SPEC_ENV]
        lines = L.p_chunk_power_lines()
        self.assertEqual(sum("MISMATCH !!!" in x for x in lines), 3)
        self.assertFalse(any("SCALE" in x for x in lines))
        for k in ("power_limit_w", "power_limit_cards", "power_limit_source"):
            d.pop(k)
        old = self._json(d)
        self._apply("--p-prefill-graph", "512", "--p-chunk-policy", "dynamic", "--p-chunk-model", old)
        self.assertEqual(L.p_chunk_policy_env()[P.SPEC_ENV], spec_new)     # byte-identical spec
        self.assertEqual(sum("old JSON" in x for x in L.p_chunk_power_lines()), 3)

    def test_dynamic_with_flag_rescales_only_mismatching_stages(self):
        path = os.path.join(DATA, "27b_int8_rc9j_cut41-12-11.pchunk.json")
        self._apply("--p-prefill-graph", "512", "--p-chunk-policy", "dynamic", "--p-chunk-model", path)
        base = P.PolicySpec.from_json(L.p_chunk_policy_env()[P.SPEC_ENV])
        self._apply("--p-prefill-graph", "512", "--p-chunk-policy", "dynamic", "--p-chunk-model", path,
                    "--p-power-scale", "linear", current=(600.0, 230.0, 230.0))
        spec = P.PolicySpec.from_json(L.p_chunk_policy_env()[P.SPEC_ENV])
        self.assertAlmostEqual(spec.stages[0].base_ms(1024, True), base.stages[0].base_ms(1024, True) * 400 / 600,
                               places=3)
        self.assertEqual(spec.stages[1], base.stages[1])
        self.assertEqual(spec.stages[0].eager_floor_ms, base.stages[0].eager_floor_ms)
        self.assertIn("+powerscale:law=linear,PP0x0.6667", spec.source)
        self.assertTrue(any("SCALE" in x for x in L.p_chunk_power_lines()))
        with self.assertRaises(SystemExit):
            self._apply("--p-prefill-graph", "512", "--p-chunk-policy", "dynamic",
                        "--p-power-scale", "power", "--p-power-scale-exponent", "5")

    def test_model_sources_know_their_limit(self):
        self.assertEqual(L.p_chunk_model_power_limits("builtin-int8", 3)[0], [400.0, 230.0, 230.0])
        log = os.path.join(self._tmp.name, "x.P.log")
        info5 = N.PowerInfo(U5090, "NVIDIA GeForce RTX 5090", 1, 450.0)
        info3 = N.PowerInfo(U3080A, "NVIDIA GeForce RTX 3080", 0, 250.0)
        with open(log, "w") as fh:
            fh.write("[2026-09-26 12:00:00 PP0] " + PL.boot_line({"tp_rank": 0, "pp_rank": 0, "gpu_id": 0}, info5)
                     + "\n[2026-09-26 12:00:00 PP1] " + PL.boot_line({"tp_rank": 0, "pp_rank": 1, "gpu_id": 1}, info3)
                     + "\n[2026-09-26 12:00:01 PP0] #PGAP pp_rank=0 fwd=1\n")
        vals, src = L.p_chunk_model_power_limits("fit:" + log, 3)
        self.assertEqual(vals, [450.0, 250.0, None])
        self.assertIn("POWER-LIMIT", src)
        with open(log, "w") as fh:
            fh.write("[2026-09-26 12:00:01 PP0] #PGAP pp_rank=0 fwd=1\n")
        self.assertIsNone(L.p_chunk_model_power_limits("fit:" + log, 3)[0])
        self.assertIsNone(L.p_chunk_model_power_limits(self._json({"stages": []}), 3)[0])

    def test_stage_current_maps_pp_to_ordered_cards(self):
        cards = [L.Card(0, U3080A, "NVIDIA GeForce RTX 3080", 20480), L.Card(1, U5090, "NVIDIA GeForce RTX 5090", 32607),
                 L.Card(2, U3080B, "NVIDIA GeForce RTX 3080", 20480)]
        with mock.patch.object(L, "resolve_cards", return_value=cards), \
                mock.patch.object(PL, "read_current",
                                  return_value={U5090: (400.0, "RTX5090"), U3080A: (230.0, "RTX3080"),
                                                U3080B: (231.0, "RTX3080")}):
            vals, labels = L.p_stage_power_current(3)
        self.assertEqual(vals, [400.0, 230.0, 231.0])
        self.assertEqual(labels[0], "PP0(nvml1 RTX5090)")
        with mock.patch.object(L, "resolve_cards", side_effect=RuntimeError("no nvml")):
            self.assertEqual(L.p_stage_power_current(3)[0], [None, None, None])

    def test_boot_record_gets_the_fields(self):
        cards = [L.Card(1, U5090, "NVIDIA GeForce RTX 5090", 32607), L.Card(0, U3080A, "NVIDIA GeForce RTX 3080", 20480)]
        recs = [c.__dict__ for c in cards]
        logged = []
        with mock.patch.dict(os.environ, {N.ENV_NVML_REPLAY: ""}), _patch_nvml(_FakeNvml()):
            L.record_card_power(recs, cards, logged.append)
        self.assertEqual((recs[0]["power_limit_w"], recs[0]["sm_clock_max_mhz"]), (400.0, 3090))
        self.assertEqual(recs[1]["power_limit_w"], 230.0)
        self.assertNotIn("power_limit_w", cards[0].__dict__)       # the Card itself is not grown
        self.assertIn("power_limit_w=400", logged[0])
        recs = [c.__dict__ for c in cards]
        logged = []
        with mock.patch.dict(os.environ, {N.ENV_NVML_REPLAY: ""}), \
                mock.patch.object(N, "_pynvml", side_effect=N.NvmlUnavailableError("x")):
            L.record_card_power(recs, cards, logged.append)
        self.assertIsNone(recs[0]["power_limit_w"])
        self.assertIn("power_limit_w=NA", logged[0])


class TestDSpeed(unittest.TestCase):
    def _lines(self, current=None):
        from sglang.srt.distributed.utils import partition_units

        cur = partition_units(1088, [58, 25, 25])
        return D.speed_advisory_lines(D._QWEN38_27B, "compressed-tensors", (58, 25, 25), cur, 1088,
                                      (26, 19, 19), current_power_w=current)

    def test_calib_and_every_row_name_the_limit(self):
        self.assertEqual(D.rc9_calib("int8").power_limit_w, (400.0, 230.0, 230.0))
        self.assertEqual(D.rc9_calib("nvfp4").power_limit_w, (400.0, 230.0, 230.0))
        lines = self._lines()
        self.assertIn("valid under power limit 400/230/230W per rank (calibration), cards run unknown", lines[0])
        rows = [x for x in lines if "D-SPEED" in x]
        self.assertEqual(len(rows), 13)
        self.assertTrue(all(r.endswith("pl_w=400/230/230W") for r in rows))

    def test_mismatch_is_loud_and_rows_unchanged(self):
        same = self._lines((400.0, 230.0, 230.0))
        self.assertIn("cards run 400/230/230W", same[0])
        self.assertFalse(any("MISMATCH" in x for x in same))
        lines = self._lines((575.0, 230.0, 230.0))
        self.assertEqual(sum("MISMATCH !!!" in x for x in lines), 1)
        self.assertEqual(len([x for x in lines if "D-SPEED" in x]), 13)

    def test_current_power_of_ranks_never_raises(self):
        env = {"CUDA_VISIBLE_DEVICES": f"{U5090},{U3080A},{U3080B}", N.ENV_NVML_REPLAY: ""}
        with mock.patch.dict(os.environ, env), _patch_nvml(_FakeNvml()):
            self.assertEqual(D.current_power_w_of_ranks(0, 3), [400.0, 230.0, 230.0])
        with mock.patch.object(N, "_pynvml", side_effect=N.NvmlUnavailableError("x")):
            self.assertIsNone(D.current_power_w_of_ranks(0, 3))


if __name__ == "__main__":
    unittest.main()
