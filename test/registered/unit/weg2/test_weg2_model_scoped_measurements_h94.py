# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ==============================================================================
"""H94: a measurement is charged only to the checkpoint it was taken on.

THE DEFECT (inventory 26.09., UNIFY_PLAN risk (c), boot fnFL2x178). Three
Qwen3.8-27B measurements shaped the Next-Flash launch:

* ``P_OVERSHOOT_MIB = [920, 0, 512]`` (27B boot weg2ls2b2) was subtracted from
  NF's P budget -- ``budget P ... RTX 5090: 28208 MiB = ... - measured_awake_
  overshoot 920 (boot weg2ls2b2)``; P argv ``--rank-gpu-memory-mib
  28208,17840,17168``;
* ``D_OVERSHOOT_MIB = [489, 0, 0]`` (27B boot weg2ls4b1) from NF's D budget;
* ``resolve_x`` seeded X from the NEWEST front log on the rig, whatever it
  booted, and fell back to a 27B pair (weg2zr2);
* ``solve_p_cut`` would pick an UNPINNED NF cut from the 27B per-layer rates
  (``MEASURED_MS_PER_LAYER`` bsscale, ``ATTN_ANCHOR_MS``).

THE FIX. Each is resolved per checkpoint: the overshoot from the NF boots' own
H55 windows (record -> built-in reference of this checkpoint -> named
UNMEASURED-FALLBACK 0), X from this checkpoint's front logs only (named
fallback = floor), the cut rates refused unpinned and named when pinned. The
27B keeps every number byte-identical.
"""

import inspect
import json
import os
import tempfile
import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(__file__)

NF = "/models/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
NF_NVFP4 = "/models/Qwen3.8-Flash-Next-NVFP4-nvidia"
B27_INT8 = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"
B27_FP8 = "/models/Qwen3.8-27B-FP8"
OTHER = "/models/Some-Other-Model-GGUF"
UUIDS = ("GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d",
         "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7",
         "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4")
NVML = (1, 0, 2)
NAMES = ("NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 3080", "NVIDIA GeForce RTX 3080")


def _cards():
    return [types.SimpleNamespace(uuid=u, nvml_index=n, name=nm)
            for u, n, nm in zip(UUIDS, NVML, NAMES)]


def _tmpdir(test) -> str:
    t = tempfile.TemporaryDirectory()
    test.addCleanup(t.cleanup)
    return t.name


# ---------------------------------------------------------------------------
# 1. the launcher no longer charges the 27B literal on every boot (RED on base)
# ---------------------------------------------------------------------------


class MainChargesTheCheckpointsOwnOvershoot(CustomTestCase):
    def test_main_no_longer_passes_the_27b_literals(self):
        """Base: ``overshoot_mib=P_OVERSHOOT_MIB, overshoot_provenance="boot
        weg2ls2b2"`` and ``overshoot_mib=D_OVERSHOOT_MIB`` in main() -- charged
        whatever the checkpoint."""
        from sglang.srt.weg2 import launcher

        src = inspect.getsource(launcher.main)
        self.assertNotIn("overshoot_mib=P_OVERSHOOT_MIB", src)
        self.assertNotIn("overshoot_mib=D_OVERSHOOT_MIB", src)
        self.assertNotIn('overshoot_provenance="boot weg2ls2b2"', src)
        self.assertEqual(src.count('awake_overshoot_for("P"'), 1)
        # the real D pass AND the dry-run's D expectation (the gate must print
        # and charge what the boot will charge)
        self.assertEqual(src.count('awake_overshoot_for("D"'), 2)
        dry = src[src.index('"D(dry, expectation)", '):]
        self.assertIn("overshoot_mib=list(_overshoot_d.mib)", dry[:400])

    def test_nf_p_overshoot_is_nf_measured_not_the_27b_vector(self):
        from sglang.srt.weg2 import launcher

        lines = []
        res = launcher.awake_overshoot_for("P", _cards(), NF, lines.append,
                                           record_path=os.path.join(_tmpdir(self), "none.json"))
        self.assertNotEqual(list(res.mib), [920, 0, 512])
        self.assertEqual(res.mib, (920, 0, 0))
        self.assertEqual(res.source, "BUILTIN")
        self.assertNotIn("weg2ls2b2", res.provenance)
        self.assertIn("fnFL2h91v1", res.provenance)
        self.assertEqual(len(lines), 1)
        self.assertIn("WEG2-OVERSHOOT group=P", lines[0])
        self.assertIn("source=BUILTIN", lines[0])

    def test_the_27b_keeps_every_number_byte_identical(self):
        from sglang.srt.weg2 import awake_overshoot as ao
        from sglang.srt.weg2 import launcher

        self.assertEqual(launcher.P_OVERSHOOT_MIB, [920, 0, 512])
        self.assertEqual(launcher.D_OVERSHOOT_MIB, [489, 0, 0])
        for model in (B27_INT8, B27_FP8):
            p = ao.resolve("P", _cards(), model)
            d = ao.resolve("D", _cards(), model)
            self.assertEqual((p.mib, d.mib), ((920, 0, 512), (489, 0, 0)), model)
            self.assertIn("boot weg2ls2b2", p.provenance)
            self.assertIn("boot weg2ls4b1", d.provenance)


# ---------------------------------------------------------------------------
# 2. resolution order: RECORD -> BUILTIN -> named FALLBACK, never foreign
# ---------------------------------------------------------------------------


def _record(model, group, by_uuid, at, tag="fnFL2test"):
    return {"kind": "awake_overshoot", "group": group, "model": os.path.basename(model),
            "boot_tag": tag, "commit": "abc", "at": at, "overshoot_by_uuid": dict(by_uuid)}


class ResolutionNeverTakesAnotherCheckpointsNumber(CustomTestCase):
    def _sidecar(self, *entries) -> str:
        path = os.path.join(_tmpdir(self), "weg2_measured_record.json")
        with open(path, "w") as f:
            json.dump({"samples": [
                # a dormant-image entry of the same sidecar must not confuse the reader
                {"group": "P", "rss_shmem_gib": 1.0, "boot_tag": "x", "at": "2026-09-26T09:00:00Z"},
                *entries]}, f)
        return path

    def test_unknown_checkpoint_gets_the_named_fallback_zero(self):
        from sglang.srt.weg2 import awake_overshoot as ao

        for g in ("P", "D"):
            res = ao.resolve(g, _cards(), OTHER)
            self.assertEqual(res.mib, (0, 0, 0))
            self.assertEqual(res.source, "UNMEASURED-FALLBACK")
            self.assertIn("awake_overshoot_record", res.provenance)
            self.assertIn("UNMEASURED-FALLBACK", res.line(_cards()))

    def test_a_newer_record_of_another_checkpoint_is_never_charged(self):
        from sglang.srt.weg2 import awake_overshoot as ao

        path = self._sidecar(
            _record(B27_INT8, "P", dict(zip(UUIDS, (4000, 4000, 4000))), "2026-09-26T10:00:00Z"),
            _record(NF, "P", dict(zip(UUIDS, (700, 0, 0))), "2026-09-26T08:00:00Z"),
        )
        res = ao.resolve("P", _cards(), NF, record_path=path)
        self.assertEqual((res.source, res.mib), ("RECORD", (700, 0, 0)))
        # and an unknown checkpoint finds neither
        self.assertEqual(ao.resolve("P", _cards(), OTHER, record_path=path).mib, (0, 0, 0))

    def test_newest_same_checkpoint_record_wins_over_the_builtin(self):
        from sglang.srt.weg2 import awake_overshoot as ao

        path = self._sidecar(
            _record(NF, "D", dict(zip(UUIDS, (100, 0, 0))), "2026-09-26T08:00:00Z"),
            _record(NF, "D", dict(zip(UUIDS, (300, 10, 20))), "2026-09-26T09:00:00Z"),
        )
        res = ao.resolve("D", _cards(), NF, record_path=path)
        self.assertEqual(res.mib, (300, 10, 20))
        self.assertIn("fnFL2test", res.provenance)

    def test_accept_filter_rejects_a_record_of_another_form(self):
        from sglang.srt.weg2 import awake_overshoot as ao

        path = self._sidecar(_record(NF, "P", dict(zip(UUIDS, (1, 2, 3))), "2026-09-26T08:00:00Z"))
        res = ao.resolve("P", _cards(), NF, record_path=path, accept=lambda e: False)
        self.assertEqual(res.source, "BUILTIN")
        self.assertIn("skipped", " ".join(res.notes))

    def test_a_record_that_misses_a_card_is_not_used(self):
        from sglang.srt.weg2 import awake_overshoot as ao

        path = self._sidecar(_record(NF, "P", {UUIDS[0]: 5, UUIDS[1]: 5}, "2026-09-26T08:00:00Z"))
        res = ao.resolve("P", _cards(), NF, record_path=path)
        self.assertEqual(res.source, "BUILTIN")
        self.assertIn(UUIDS[2], " ".join(res.notes))

    def test_nvfp4_is_its_own_checkpoint(self):
        from sglang.srt.weg2 import awake_overshoot as ao

        res = ao.resolve("P", _cards(), NF_NVFP4)
        self.assertEqual(res.source, "BUILTIN")
        self.assertIn("fnNV4f4", res.provenance)


# ---------------------------------------------------------------------------
# 3. the derivation rule, bound to the NF boots' own numbers
# ---------------------------------------------------------------------------


class TheDerivationRule(CustomTestCase):
    def _m(self, charged, floor, free, cache=0, retries=0, ordinal=0):
        from sglang.srt.weg2 import awake_overshoot as ao

        return ao.CardMeasurement(uuid=UUIDS[ordinal], ordinal=ordinal, name=NAMES[ordinal],
                                  charged_mib=charged, floor_mib=floor, free_min_mib=free,
                                  cache_at_min_mib=cache, retries=retries, windows=1)

    def test_three_named_cases(self):
        from sglang.srt.weg2 import awake_overshoot as ao

        d = ao.derive_card(self._m(512, 858, 2372))
        self.assertEqual((d.verdict, d.overshoot_mib, d.unclaimed_slack_mib), ("SLACK", 0, 1002))
        d = ao.derive_card(self._m(920, 1055, 1500))
        self.assertEqual((d.verdict, d.overshoot_mib), ("SLACK", 475))
        # the cache held the shortfall: a raise would ratchet, the charge is kept
        d = ao.derive_card(self._m(920, 1055, 179, cache=6936, retries=6))
        self.assertEqual((d.verdict, d.overshoot_mib), ("SATURATED", 920))
        d = ao.derive_card(self._m(0, 1095, 20, cache=10, retries=1))
        self.assertEqual((d.verdict, d.overshoot_mib), ("SATURATED", 0))
        # a real shortfall with nothing cached raises by exactly the shortfall
        d = ao.derive_card(self._m(100, 800, 500, cache=50))
        self.assertEqual((d.verdict, d.overshoot_mib), ("SHORT", 400))

    def test_the_nf_builtin_is_what_the_rule_derives_from_fnFL2h91v1(self):
        """The shipped NF reference is bound to the measured h91v1 numbers
        (tool output on /spinning/evidence-665-f1, 26.09.)."""
        from sglang.srt.weg2 import awake_overshoot as ao

        p = [ao.derive_card(self._m(920, 1055, 179, 6936, 6, 0)).overshoot_mib,
             ao.derive_card(self._m(0, 1095, 20, 4714, 2, 1)).overshoot_mib,
             ao.derive_card(self._m(512, 858, 2372, 4647, 0, 2)).overshoot_mib]
        d = [ao.derive_card(self._m(489, 767, 73, 2289, 244, 0)).overshoot_mib,
             ao.derive_card(self._m(0, 700, 56, 3853, 32, 1)).overshoot_mib,
             ao.derive_card(self._m(0, 701, 4, 2706, 94, 2)).overshoot_mib]
        self.assertEqual(tuple(p), ao.reference_for("P", NF).mib)
        self.assertEqual(tuple(d), ao.reference_for("D", NF).mib)


# ---------------------------------------------------------------------------
# 4. the tool: one boot's logs -> record, refusing a partial one
# ---------------------------------------------------------------------------

_TS = "[2026-09-26T07:49:04Z] WEG2-LAUNCH "


def _front_text(model=os.path.basename(NF), with_floor_d=True) -> str:
    rows = [
        _TS + "=== WEG2 BOOT tag=fnFL2unit tree=/t @ 39fd662d9e (clean) stamp=0926 dry=False",
        _TS + f"WEG2-FORM arch=moe experts=offload draft=mtp p_draft=none kv=qsa_forma "
              f"flip=family vision=off profile=nextflash model={model} (sources: x)",
        _TS + "WEG2-USER-RESERVE PROVENANCE (#1257c, line shape #1342): " + "; ".join(
            f"argv[{i}] -> ordinal={i} nvml{n} name='{nm}' uuid={u} reserve_mib=0"
            for i, (u, n, nm) in enumerate(zip(UUIDS, NVML, NAMES))),
        _TS + "budget P group=P ordinal=0 nvml_idx=1 NVIDIA GeForce RTX 5090: 28208 MiB = total 32607 - corridor 1459 (floor 1055 source=MEASURED-P reserve=0 + awake_overshoot 404) - dormant_other 2018 - measured_awake_overshoot 920 (boot weg2ls2b2) MiB",
        _TS + f"CORRIDOR-FLOOR card={UUIDS[0]} group=P floor=1055 verdict_floor=1055 ceiling=1266",
        _TS + "budget P group=P ordinal=1 nvml_idx=0 NVIDIA GeForce RTX 3080: 17840 MiB = total 20480 - corridor 1499 (floor 1095 source=MEASURED-P reserve=0 + awake_overshoot 404) - dormant_other 1140 MiB",
        _TS + f"CORRIDOR-FLOOR card={UUIDS[1]} group=P floor=1095 verdict_floor=1095 ceiling=1314",
        _TS + "budget P group=P ordinal=2 nvml_idx=2 NVIDIA GeForce RTX 3080: 17168 MiB = total 20480 - corridor 1262 (floor 858 source=MEASURED-P reserve=0 + awake_overshoot 404) - dormant_other 1538 - measured_awake_overshoot 512 (boot weg2ls2b2) MiB",
        _TS + f"CORRIDOR-FLOOR card={UUIDS[2]} group=P floor=858 verdict_floor=858 ceiling=1030",
    ]
    for i, (n, nm) in enumerate(zip(NVML, NAMES)):
        rows.append(_TS + f"budget D group=D ordinal={i} nvml_idx={n} {nm}: 18000 MiB = total 20480 - corridor 1104 (floor 700 source=MEASURED-D reserve=0 + awake_overshoot 404) - dormant_other 712"
                    + (" - measured_awake_overshoot 489 (boot weg2ls4b1)" if i == 0 else "") + " MiB")
        if with_floor_d:
            rows.append(_TS + f"CORRIDOR-FLOOR card={UUIDS[i]} group=D floor=700 verdict_floor=700 ceiling=840")
    return "\n".join(rows) + "\n"


def _peak(rank, phase, free, alloc, res, retries=0) -> str:
    return (f"[2026-09-26 07:56:25 PP{rank}] WEG2-VRAM-PEAK rank={rank} phase={phase} rows=16384 n=1 "
            f"t0_unix_ms=1 t_unix_ms=2 window_ms=1 peak_allocated_mib={alloc} peak_reserved_mib={res} "
            f"start_allocated_mib=1 transient_mib=1 allocated_mib={alloc} reserved_mib={res} "
            f"card_free_start_mib=1 card_free_mib={free} card_total_mib=32088 "
            f"alloc_retries={retries} ooms=0 alloc_retries_total=0\n")


class TheToolDerivesOneBootsRecord(CustomTestCase):
    def _boot(self, front=None):
        d = _tmpdir(self)
        base = os.path.join(d, "boot_weg2_fnFL2unit_39fd662d9e_0926_074852")
        with open(base + ".front.log", "w") as f:
            f.write(front if front is not None else _front_text())
        with open(base + ".P.log", "w") as f:
            f.write(_peak(0, "idle", 5, 1, 1))          # not a load window: ignored
            f.write(_peak(0, "chunk", 179, 21811, 28747, retries=6))
            f.write(_peak(1, "chunk", 20, 13786, 18500, retries=2))
            f.write(_peak(2, "chunk", 2372, 12000, 16647))
        with open(base + ".D.log", "w") as f:
            f.write(_peak(0, "round", 73, 27000, 29289, retries=5))
            f.write(_peak(1, "chunk", 1500, 15000, 15100))
            f.write(_peak(2, "round", 900, 16000, 16100))
        return base, os.path.join(d, "sidecar.json")

    def test_record_appended_and_resolved_back(self):
        from sglang.srt.weg2 import awake_overshoot as ao
        from sglang.srt.weg2.tools import awake_overshoot_record as tool

        base, sidecar = self._boot()
        self.assertEqual(tool.main(["--front", base + ".front.log", "--append", sidecar]), 0)
        p = ao.resolve("P", _cards(), NF, record_path=sidecar)
        d = ao.resolve("D", _cards(), NF, record_path=sidecar)
        self.assertEqual((p.source, p.mib), ("RECORD", (920, 0, 0)))
        # D: ord0 saturated (retries) keeps 489; ord1/ord2 slack, 0
        self.assertEqual((d.source, d.mib), ("RECORD", (489, 0, 0)))
        with open(sidecar) as f:
            rec = json.load(f)["samples"][0]
        self.assertEqual(rec["model"], os.path.basename(NF))
        self.assertEqual(rec["boot_tag"], "fnFL2unit")
        self.assertEqual([c["verdict"] for c in rec["cards"]], ["SATURATED", "SATURATED", "SLACK"])
        # the record never reaches another checkpoint
        self.assertEqual(ao.resolve("P", _cards(), B27_INT8, record_path=sidecar).mib, (920, 0, 512))

    def test_a_card_without_its_floor_refuses_the_group(self):
        from sglang.srt.weg2.tools import awake_overshoot_record as tool

        base, sidecar = self._boot(_front_text(with_floor_d=False))
        self.assertEqual(tool.main(["--front", base + ".front.log", "--append", sidecar]), 2)
        with open(sidecar) as f:
            groups = [e["group"] for e in json.load(f)["samples"]]
        self.assertEqual(groups, ["P"])


# ---------------------------------------------------------------------------
# 5. resolve_x: only this checkpoint's front logs, named fallback (RED on base)
# ---------------------------------------------------------------------------


def _front_log(model_line: str, flip_s=2.9, r_d=1138.0, r_p=4628.0) -> str:
    return "\n".join([
        model_line,
        f"[t] INFO weg2.front: WEG2-FLIP epoch=1 flip_total={int(flip_s * 1000)} ms",
        "[t] INFO weg2.front: WEG2-SERVED group=D leg=2 rid=a "
        f"uncached={int(r_d * 10)} verdict=single_prefill wall=10.0s",
        "[t] INFO weg2.front: WEG2-SERVED group=P leg=1 rid=b "
        f"prompt_tokens={int(r_p * 10)} cached_tokens=0 wall=1.0s",
        "[t] INFO weg2.front: WEG2 P-DRAIN epoch=1 prefilled=8 drain_s=10.0",
        "",
    ])


_27B_ARGV = f"[t] WEG2-LAUNCH group P argv: python -m sglang.launch_server --model-path {B27_INT8} --tp-size 1"
_NF_FORM = (f"[t] WEG2-LAUNCH WEG2-FORM arch=moe experts=offload draft=mtp p_draft=none "
            f"kv=qsa_forma flip=family vision=off profile=nextflash model={os.path.basename(NF)}")


class ResolveXSeedsOnlyFromThisCheckpoint(CustomTestCase):
    def _dir(self, *logs):
        d = _tmpdir(self)
        t = 1_700_000_000
        for name, body in logs:           # oldest first
            path = os.path.join(d, f"boot_weg2_{name}_abc1234_0926_000000.front.log")
            with open(path, "w") as f:
                f.write(body)
            t += 60
            os.utime(path, (t, t))
        return d

    def test_a_27b_front_log_never_seeds_nf_x(self):
        from sglang.srt.weg2 import form as weg2_form
        from sglang.srt.weg2.launcher import resolve_x

        d = self._dir(("weg2rc7", _front_log(_27B_ARGV)))
        seed = resolve_x(None, d, 4096, accept=weg2_form.same_model_log(NF), model=NF)
        self.assertEqual(seed.tokens, 4096)
        self.assertFalse(seed.measured)
        self.assertIn("UNMEASURED-FALLBACK", seed.provenance)
        self.assertIn("1 of another checkpoint", seed.provenance)
        self.assertNotIn("weg2rc7", seed.provenance)

    def test_the_own_checkpoint_log_is_used_past_a_newer_foreign_one(self):
        from sglang.srt.weg2 import form as weg2_form
        from sglang.srt.weg2.launcher import resolve_x

        d = self._dir(("fnFL2own", _front_log(_NF_FORM)),
                      ("weg2rc7", _front_log(_27B_ARGV, flip_s=13.0)))
        seed = resolve_x(None, d, 4096, accept=weg2_form.same_model_log(NF), model=NF)
        self.assertTrue(seed.measured)
        self.assertIn("fnFL2own", seed.provenance)
        self.assertIn("and 1 of another checkpoint", seed.provenance)

    def test_the_27b_still_gets_its_recorded_pair(self):
        from sglang.srt.weg2 import form as weg2_form
        from sglang.srt.weg2.launcher import resolve_x

        d = self._dir(("fnFL2own", _front_log(_NF_FORM)))
        seed = resolve_x(None, d, 4096, accept=weg2_form.same_model_log(B27_INT8), model=B27_INT8)
        self.assertIn("recorded PRE-BARLINK", seed.provenance)
        self.assertEqual(seed.tokens, resolve_x(None, _tmpdir(self), 4096).tokens)

    def test_main_passes_the_calibration_filter(self):
        from sglang.srt.weg2 import launcher

        src = inspect.getsource(launcher.main)
        start = src.index("x_seed = resolve_x(")
        call = src[start:src.index("x_tokens, x_provenance", start)]
        self.assertIn("accept=calib_log_accept", call)


# ---------------------------------------------------------------------------
# 6. the P-cut rates (RED on base)
# ---------------------------------------------------------------------------


def _cut_ns(**kw):
    from sglang.srt.weg2 import launcher

    base = dict(
        weg2_boot_form=object(),
        pp_cut_measured_ms_per_layer=launcher.MEASURED_MS_PER_LAYER,
        pp_cut_attn_anchor_ms=launcher.ATTN_ANCHOR_MS,
        pp_stage_ratio=None, pp_attn_stage_ratio=None, pp_layer_set=None,
        pp_cut_expert_device_fraction="",
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


class CutRatesAreThisCheckpoints(CustomTestCase):
    def test_foreign_cut_rates_scope(self):
        from sglang.srt.weg2 import launcher

        self.assertIsNotNone(launcher.foreign_cut_rates(_cut_ns(), NF))
        self.assertIsNone(launcher.foreign_cut_rates(_cut_ns(), B27_INT8))
        self.assertIsNone(launcher.foreign_cut_rates(_cut_ns(), B27_FP8))
        self.assertIsNone(launcher.foreign_cut_rates(_cut_ns(weg2_boot_form=None), NF))
        self.assertIsNone(launcher.foreign_cut_rates(
            _cut_ns(pp_cut_measured_ms_per_layer="133.4,346.4,287.9"), NF))

    def _model_dir(self, name) -> str:
        d = os.path.join(_tmpdir(self), name)
        os.makedirs(d)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"text_config": {
                "num_hidden_layers": 8, "hidden_size": 64,
                "layer_types": ["linear_attention", "full_attention"] * 4}}, f)
        return d

    def test_unpinned_solve_on_foreign_rates_is_refused_by_name(self):
        from sglang.srt.weg2 import launcher

        model = self._model_dir(os.path.basename(OTHER))
        with self.assertRaises(launcher.Weg2LaunchRefused) as cm:
            launcher.solve_p_cut(_cut_ns(), _cards(), [1000, 1000, 1000], model, lambda _l: None)
        self.assertIn("W40 Weg2PPCutRefused", str(cm.exception))
        self.assertIn("FOREIGN-MODEL", str(cm.exception))
        # a layer cut without its attention split is not a pin: the split is solved from the rates
        with self.assertRaises(launcher.Weg2LaunchRefused):
            launcher.solve_p_cut(_cut_ns(pp_stage_ratio="4,2,2"), _cards(), [1000, 1000, 1000],
                                 model, lambda _l: None)

    def test_pinned_solve_names_the_foreign_rates(self):
        from sglang.srt.weg2 import launcher

        model = self._model_dir(os.path.basename(OTHER))
        lines = []
        try:
            launcher.solve_p_cut(_cut_ns(pp_stage_ratio="4,2,2", pp_attn_stage_ratio="2,1,1"),
                                 _cards(), [1000, 1000, 1000], model, lines.append)
        except Exception as e:  # the stub checkpoint has no weights; the line comes first
            self.assertNotIn("FOREIGN-MODEL", str(e))
        self.assertTrue(any("PP-CUT RATES FOREIGN-MODEL" in ln for ln in lines), lines)


if __name__ == "__main__":
    unittest.main()
