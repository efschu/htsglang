# SPDX-License-Identifier: Apache-2.0
"""Q-694b: the D extend chunk-cap rate is hardware-generic -- measured per rank, the record is a start.

User question 03.10. ~19:10Z ("und du baust jetzt nicht die ganze zeit systemspezifische fixes?"):
Q-694 (fef69192fe) armed the rc12g width vote on the 27B flip line from ``D_EXTEND_CAP_PER_ROW_MIB``
0.3091 -- a number measured on THIS rig for 27B INT8. Without that record (another model, other
cards) nothing was written, the vote stayed unarmed and the y8va OOM class (X-GATE admit ->
D-direct 4096-row extend at < 600 MiB free -> OOM in barlink_bar1 all_reduce) stayed open silently.

Now: no record -> a start rate derived from the model geometry (``EXTEND-RATE source=derived``);
record -> start value (``EXTEND-RATE source=record``); every D rank measures its own extend
transient per row and votes with ``max(start, measured x 1.15)`` (ratchet up only, per rank).
Next Flash (its own ``D_EXTEND_GROWTH_PER_ROW_MIB`` via the #145 ledger), the P0 arm with its
record and the dual layout stay byte-identical.
"""
import json
import os
import tempfile
import types
import unittest
from unittest import mock

try:
    from flliper.test.ci.ci_register import register_cpu_ci
except ImportError:  # pragma: no cover

    def register_cpu_ci(*args, **kwargs):
        return None


from flliper.srt.environ import envs
from flliper.srt.pdflip import extend_trim as ET
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

MIB = 1 << 20
RECORD = 0.3091                 # D_EXTEND_CAP_PER_ROW_MIB (qwen27b), the Q-694 start value
MEASURED_RATE_MAX = 0.2690      # max allocated transient/row over 642 D extends >= 2048 rows (40 boots)
RATE_ENV = "FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB"
MEASURE_ENV = "FLLIPER_PDFLIP_EXTEND_RATE_MEASURE"
TRIM_ENV_D = "FLLIPER_PDFLIP_EXTEND_TRIM_MIB=1200,0,0"

#: the text geometry of Qwen3.8-27B (config.json of every 27B checkpoint on the rig)
QWEN27B_TEXT = {
    "hidden_size": 5120, "intermediate_size": 17408, "num_attention_heads": 24,
    "num_key_value_heads": 4, "head_dim": 256, "attn_output_gate": True,
    "linear_num_key_heads": 16, "linear_key_head_dim": 128, "linear_num_value_heads": 48,
    "linear_value_head_dim": 128, "num_hidden_layers": 64, "dtype": "bfloat16",
}
#: (6*5120 + 3*17408 + 2*max(24*256*2 + 2*4*256, 2*16*128 + 2*48*128 + 2*48)) * 4 B
DERIVED_27B = 0.4422


def _model_dir(text=QWEN27B_TEXT):
    d = tempfile.mkdtemp(prefix="q694b_")
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
                   "text_config": dict(text)}, f)
    return d


def _cards():
    from flliper.srt.pdflip import launcher as L

    return [L.Card(1, "u1", "RTX 5090", 32607), L.Card(0, "u0", "RTX 3080", 20480),
            L.Card(2, "u2", "RTX 3080", 20480)]


def _ns(profile="qwen27b", env_d=TRIM_ENV_D, model="/nonexistent/Qwen3.8-27B-INT8-gdncov", **kw):
    base = dict(profile=profile, model=model, extra_d="", env_d=env_d,
                d_foreign_context_mib="", d_nontorch_mib="", d_reserve_mib="",
                d_residency_reference_logs="", d_card_reference_logs="",
                wake_credit_reference_logs="", dual_layout=False, dual_share=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _solve(ns):
    from flliper.srt.pdflip import launcher as L

    lines = []
    L.log_d_rank_vram_solve(ns, _cards(), [27792, 17384, 17168], lines.append, "D")
    return lines


def _no_cap_record():
    """The 27B profile as another rig / model would see it: no D_EXTEND_CAP_PER_ROW_MIB."""
    from flliper.srt.pdflip import launcher as L

    real = L._pconst

    def fake(name, profile=None):
        if name == L.D_EXTEND_CAP_RATE_RECORD:
            raise KeyError(name)
        return real(name, profile)

    return mock.patch.object(L, "_pconst", side_effect=fake)


class NoRecordStillArmsTheCap(CustomTestCase):
    def test_no_record_writes_the_derived_rate_and_arms_the_measurement(self):
        """RED on 76f5bbde19: no record -> env_d stayed the trim alone, no vote anywhere."""
        ns = _ns(model=_model_dir())
        with _no_cap_record():
            lines = _solve(ns)
        self.assertIn(f"{RATE_ENV}={DERIVED_27B}", ns.env_d)
        self.assertIn(f"{MEASURE_ENV}=1", ns.env_d)
        self.assertTrue(ns.env_d.startswith(TRIM_ENV_D + ";"))
        self.assertTrue(any("EXTEND-RATE source=derived" in ln and "measure=on" in ln for ln in lines), lines)
        self.assertTrue(any("Q694 EXTEND-CAP-FLIP" in ln and "Modellgeometrie" in ln for ln in lines), lines)

    def test_no_record_no_geometry_names_the_unarmed_cap(self):
        ns = _ns(model="/nonexistent/model-without-config")
        with _no_cap_record():
            lines = _solve(ns)
        self.assertEqual(ns.env_d, TRIM_ENV_D)
        self.assertTrue(any("EXTEND-RATE source=none" in ln and "NICHT scharf" in ln for ln in lines), lines)

    def test_derived_second_pass_is_idempotent_and_operator_wins(self):
        ns = _ns(model=_model_dir())
        with _no_cap_record():
            _solve(ns)
            first = ns.env_d
            lines = _solve(ns)
        self.assertEqual(ns.env_d, first)
        self.assertFalse(any("Vorrang" in ln for ln in lines if "EXTEND" in ln), lines)
        ns = _ns(model=_model_dir(), env_d=TRIM_ENV_D + f";{RATE_ENV}=0.6;{MEASURE_ENV}=0")
        with _no_cap_record():
            lines = _solve(ns)
        self.assertIn(f"{RATE_ENV}=0.6", ns.env_d)
        self.assertIn(f"{MEASURE_ENV}=0", ns.env_d)
        self.assertTrue(any("EXTEND-RATE source=derived" in ln and "measure=off" in ln for ln in lines))

    def test_p0_arm_without_record_derives_too(self):
        from flliper.srt.pdflip import launcher as L

        ns = _ns(model=_model_dir(), env_d="FLLIPER_PDFLIP_TORCH_CACHE_CAP=1")
        lines = []
        with _no_cap_record():
            self.assertEqual(L._d_extend_cap_rate_env(ns, lines.append, "D"), f"{DERIVED_27B}")
        self.assertIn(f"{MEASURE_ENV}=1", ns.env_d)
        self.assertTrue(any("EXTEND-CAP rows_cap aus min(card_free, torch_cap - reserved)" in ln
                            for ln in lines))


class RecordIsTheStart(CustomTestCase):
    def test_flip_record_arms_the_measurement(self):
        ns = _ns()
        lines = _solve(ns)
        self.assertIn(f"{RATE_ENV}={RECORD},{RECORD},{RECORD}", ns.env_d)
        self.assertIn(f"{MEASURE_ENV}=1", ns.env_d)
        self.assertTrue(any("EXTEND-RATE source=record" in ln for ln in lines), lines)

    def test_p0_arm_with_record_is_byte_identical(self):
        from flliper.srt.pdflip import launcher as L

        ns = _ns(env_d="FLLIPER_PDFLIP_TORCH_CACHE_CAP=1")
        lines = []
        self.assertEqual(L._d_extend_cap_rate_env(ns, lines.append, "D"), f"{RECORD},{RECORD},{RECORD}")
        self.assertEqual(ns.env_d, f"FLLIPER_PDFLIP_TORCH_CACHE_CAP=1;{RATE_ENV}={RECORD},{RECORD},{RECORD}")
        self.assertFalse(any("EXTEND-RATE" in ln for ln in lines))

    def test_derived_start_is_above_the_reference_record(self):
        """On the reference rig the geometry start must not undercut what the record (and every
        measured extend) needs -- else a fresh rig would start below the y8va death's rate."""
        self.assertEqual(ET.derived_rate_mib({"text_config": QWEN27B_TEXT}), DERIVED_27B)
        self.assertGreaterEqual(DERIVED_27B, RECORD)
        self.assertGreaterEqual(RECORD, MEASURED_RATE_MAX)


class NextFlashAndDualUnchanged(CustomTestCase):
    def test_nextflash_writes_nothing_even_with_a_readable_geometry(self):
        from flliper.srt.pdflip import launcher as L

        for capped in (True, False):
            ns = _ns(profile="nextflash", env_d="", model=_model_dir())
            lines = []
            self.assertIsNone(L._d_extend_cap_rate_env(ns, lines.append, "D", capped=capped))
            self.assertEqual((ns.env_d, lines), ("", []))

    def test_dual_writes_nothing_without_record_either(self):
        for kw in ({"dual_layout": True}, {"dual_share": True}):
            ns = _ns(model=_model_dir(), **kw)
            with _no_cap_record():
                lines = _solve(ns)
            self.assertEqual(ns.env_d, TRIM_ENV_D, kw)
            self.assertFalse(any("EXTEND-RATE" in ln or "Q694" in ln for ln in lines), kw)


class _Mode:
    def __init__(self, extend=True, verify=False):
        self._e, self._v = extend, verify

    def is_extend(self):
        return self._e

    def is_target_verify(self):
        return self._v


class _Ids:
    def __init__(self, n):
        self.shape = (n,)


class _Cuda:
    """A rank's allocator: peak/allocated/reserved in MiB, card free in MiB."""

    def __init__(self, free=529, allocated=27627, reserved=28000, peak=27700):
        self.free, self.allocated, self.reserved, self.peak = free, allocated, reserved, peak
        self.emptied = 0

    def is_current_stream_capturing(self):
        return False

    def mem_get_info(self, *a):
        return self.free * MIB, 32088 * MIB

    def memory_reserved(self, *a):
        return self.reserved * MIB

    def memory_stats(self):
        return {"allocated_bytes.all.peak": self.peak * MIB,
                "allocated_bytes.all.current": self.allocated * MIB,
                "reserved_bytes.all.current": self.reserved * MIB}

    def synchronize(self):
        pass

    def empty_cache(self):
        self.emptied += 1

    def run_extend(self, transient, growth=0):
        self.peak = max(self.peak, self.allocated + transient)
        self.reserved += growth
        self.free -= growth


class _Runner:
    is_draft_worker = False
    tp_rank = 0


class RankMeasuresItsOwnRate(CustomTestCase):
    def setUp(self):
        ET.reset_for_tests()

    def tearDown(self):
        ET.reset_for_tests()

    def _extend(self, cuda, runner, rows, transient, growth=0):
        worker = types.SimpleNamespace(is_draft_worker=False, model_runner=runner)
        ET.measure_open(worker, types.SimpleNamespace(forward_mode=_Mode()), cuda)
        cuda.run_extend(transient, growth)
        return ET.measure_close(runner, types.SimpleNamespace(forward_mode=_Mode(), input_ids=_Ids(rows)),
                                cuda, rank=0)

    def test_measured_rate_raises_the_vote_ratchet_up_only(self):
        runner, cuda = _Runner(), _Cuda()
        with envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE.override(True), \
                envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(f"{RECORD},{RECORD},{RECORD}"):
            # a deeper prefix than the record's boots: 1600 MiB over 4096 rows = 0.3907/row
            line = self._extend(cuda, runner, 4096, 1600)
            self.assertIn("EXTEND-RATE source=measured rank=0 rows=4096", line)
            self.assertAlmostEqual(ET.measured_rate(), 0.3907)
            self.assertAlmostEqual(ET.effective_rate(RECORD), 0.4494)    # ceil4(0.3907 x 1.15)
            cuda.peak = cuda.allocated                                     # the window re-based
            self.assertIsNone(self._extend(cuda, runner, 4096, 800))      # lower: no ratchet down
            self.assertAlmostEqual(ET.measured_rate(), 0.3907)
            cuda.free = 529
            vote = ET.width_vote(cuda, 0, 4096, 1, True)
        self.assertEqual(vote, 509)                                        # floor(229 / 0.4494)
        self.assertLess(vote, 740)                                         # the record alone: 740

    def test_below_the_record_the_record_binds(self):
        runner, cuda = _Runner(), _Cuda()
        with envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE.override(True), \
                envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(f"{RECORD}"):
            self._extend(cuda, runner, 4096, 700)                          # 0.1709/row
            self.assertEqual(ET.effective_rate(RECORD), RECORD)
            self.assertEqual(ET.width_vote(cuda, 0, 4096, 1, True), 740)

    def test_short_extend_is_not_priced(self):
        runner, cuda = _Runner(), _Cuda()
        with envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE.override(True):
            self.assertIsNone(self._extend(cuda, runner, ET.GROWTH_PER_ROW_MIN_ROWS - 64, 1500))
            self.assertIsNone(ET.measured_rate())

    def test_peak_not_raised_prices_the_reserved_growth(self):
        runner, cuda = _Runner(), _Cuda(peak=40000)                        # an older higher peak
        with envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE.override(True):
            line = self._extend(cuda, runner, 4096, 900, growth=1200)
        self.assertIn("transient_mib=0 reserved_growth_mib=1200", line)
        self.assertAlmostEqual(ET.measured_rate(), 0.293)

    def test_draft_runner_never_closes_the_target_reading(self):
        runner, cuda = _Runner(), _Cuda()
        draft = types.SimpleNamespace(is_draft_worker=True)
        with envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE.override(True):
            worker = types.SimpleNamespace(is_draft_worker=False, model_runner=runner)
            ET.measure_open(worker, types.SimpleNamespace(forward_mode=_Mode()), cuda)
            cuda.run_extend(1600)
            fb = types.SimpleNamespace(forward_mode=_Mode(), input_ids=_Ids(4096))
            self.assertIsNone(ET.measure_close(draft, fb, cuda, rank=0))
            self.assertIsNotNone(ET.measure_close(runner, fb, cuda, rank=0))

    def test_off_is_byte_identical(self):
        runner, cuda = _Runner(), _Cuda()
        with envs.FLLIPER_PDFLIP_EXTEND_GROWTH_PER_ROW_MIB.override(f"{RECORD}"), \
                envs.FLLIPER_PDFLIP_EXTEND_TRIM_MIB.override("1200"):
            worker = types.SimpleNamespace(is_draft_worker=False, model_runner=runner, tp_rank=0)
            batch = types.SimpleNamespace(forward_mode=_Mode())
            with mock.patch.object(ET, "measure_open") as mo:
                ET.before_extend(worker, batch)
            mo.assert_not_called()
            ET._CACHE["measured"] = 0.9                                     # ignored while off
            self.assertEqual(ET.effective_rate(RECORD), RECORD)
            with self.assertLogs(ET.logger, level="INFO") as cm:
                self.assertEqual(ET.width_vote(cuda, 0, 4096, 1, True), 740)
        self.assertFalse(any("rate_src" in m for m in cm.output), cm.output)

    def test_open_reads_after_the_trim(self):
        runner, cuda = _Runner(), _Cuda(free=265, reserved=28000)
        seen = []

        def opened(worker, batch, cuda=None):
            seen.append(("open", fake.emptied))

        fake = cuda
        with envs.FLLIPER_PDFLIP_EXTEND_RATE_MEASURE.override(True), \
                envs.FLLIPER_PDFLIP_EXTEND_TRIM_MIB.override("1200"), \
                mock.patch("torch.cuda", cuda, create=True), \
                mock.patch.object(ET, "measure_open", side_effect=opened):
            worker = types.SimpleNamespace(is_draft_worker=False, model_runner=runner, tp_rank=0)
            ET.before_extend(worker, types.SimpleNamespace(forward_mode=_Mode()))
        self.assertEqual(seen, [("open", 1)])

    def test_forward_end_hook_closes_before_the_window_rebases(self):
        from flliper.srt.model_executor import vram_family_census as VFC
        from flliper.srt.model_executor import vram_peak_window as VPW

        order = []
        ET._CACHE["pending"] = (None, 0, 0, 0)
        with mock.patch.object(VFC, "_maybe_log_vram_peak_since_pools", return_value=None), \
                mock.patch.object(ET, "measure_close", side_effect=lambda *a, **k: order.append("rate")), \
                mock.patch.object(VPW, "on_forward_end", side_effect=lambda *a, **k: order.append("window")):
            VFC.maybe_log_vram_peak(_Runner(), types.SimpleNamespace(forward_mode=_Mode()), cuda=_Cuda())
        self.assertEqual(order, ["rate", "window"])


if __name__ == "__main__":
    unittest.main()
