# SPDX-License-Identifier: Apache-2.0
"""#233 -- the corridor pointer is keyed by MODEL IDENTITY, not by group alone.

FUND #232 (24.10., deskq/done/nf-korridor-digest-1024.md): the running NF boot
priced its D corridor at 767/701/700 MiB, stamped ``MEASURED-D`` with
``actuates=yes`` -- the 27B INT8 floors measured by boot ``weg2sb5h`` on 09.09.
Root, two halves:

1. ``corridor_guard._published_digest`` read ``{"groups": {"D": digest}}`` --
   keyed by GROUP, with no model or architecture term -- and the NF cache was a
   ``cp -a`` copy of the 27B one, so the NF read the 27B pointer;
2. the only writer published on a footprint HIT, so a boot without a record of
   its own (NF: ``Qwen4ExpForConditionalGeneration`` has none) never replaced
   the foreign pointer: the promised "self-correction from the second boot"
   never ran.

DANGER DIRECTION: OOM -- a foreign record with a LOWER floor must never price
this boot; without an own record the stated law (1024, or
``FLLIPER_CORRIDOR_LAW_FLOOR_MIB``) holds, named.

HERMETIC: no NVML, no torch device, no GPU, no boot, no real checkpoint. The
rig fingerprint is pinned, footprints are written into a tmp store through the
shipped writer, and the boot's architectures are pinned per test (the registry
path of a checkpoint is not on a desk box).
"""

from __future__ import annotations

import ast
import json
import os
import tempfile
import types
import unittest
from unittest import mock

from flliper.srt.managers import corridor_guard as cg
from flliper.srt.mem_ledger import activation as act
from flliper.srt.pdflip import form as pdflip_form

HW = "a191a0712717"

#: D cards of the rig and the 27B INT8 D floors from
#: phase_footprint-a191a0712717-cc0ac5caf91e.json (#232 section 1).
C5090 = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
C3080A = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"
C3080B = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
D27_PEAKS = {C5090: 767, C3080A: 701, C3080B: 700}

ARCH_27B = ("Qwen3_5ForConditionalGeneration",)
ARCH_NF = ("Qwen4ExpForConditionalGeneration",)
CKPT_27B = "Qwen3.8-27B-INT8-gdncov-vocabembed"
CKPT_NF = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp"

#: The 27B D profile of the 09.09. seed (chunk 4096, tp 3, pp 1, fp8_e4m3,
#: spec 3, decode_max_bs 24) and an NF D profile.
PROF_27B_D = act.ActivationProfile(ARCH_27B, 4096, 3, 1, "fp8_e4m3", 3, 24)
PROF_NF_D = act.ActivationProfile(ARCH_NF, 4096, 3, 1, "fp8_e4m3", 3, 6)


def _form_env(arch, experts, draft, p_draft, kv, profile, model) -> str:
    return pdflip_form.PdFlipForm(
        arch=arch, experts=experts, draft=draft, p_draft=p_draft, kv=kv,
        flip="family", vision="transient", profile=profile, model=model,
    ).env_value()


FORM_NF = _form_env("moe", "offload", "mtp", "none", "qsa_forma_dcp", "nextflash", CKPT_NF)
FORM_27B = _form_env("dense", "none", "dflash", "cold", "paged_dcp", "qwen27b", CKPT_27B)


def setUpModule():
    cg._RIG_FP_CACHE[:] = [HW, None]


def tearDownModule():
    cg._RIG_FP_CACHE.clear()


def _save(cache_dir, profile, peaks) -> str:
    """A footprint file through the SHIPPED writer (it records ``profile``)."""
    fps = {
        u: act.PhaseFootprint(
            activation_mib=int(m), capture_mib=0,
            provenance=act.FootprintProvenance.MEASURED_PEAK,
            source="test #233: probe_activation.py ingest", card_uuid=u,
            profile_digest=act.profile_key(profile),
        )
        for u, m in peaks.items()
    }
    return act.save_footprints(fps, hw_fingerprint=HW, profile=profile, cache_dir=cache_dir)


def _pointer() -> dict:
    with open(cg._floor_digest_path()) as fh:
        return json.load(fh)


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        env = {cg.FLOOR_DIGEST_FILE_ENV: os.path.join(self.cache, "corridor_floor_digest.json")}
        self._env = mock.patch.dict(os.environ, env)
        self._env.start()
        self.addCleanup(self._env.stop)
        for k in (cg.LAW_ENV, cg.GROUP_ENV, cg.RECORD_IDENTITY_ENV, pdflip_form.FORM_ENV):
            os.environ.pop(k, None)
        self._arch_saved = dict(cg._ARCH_CACHE)
        cg._ARCH_CACHE.clear()
        self.addCleanup(self._restore_arch)
        cg._RECORD_NOTED.clear()
        cg._BOOT_PUBLISHED.clear()
        self.d27 = act.profile_key(PROF_27B_D)
        self.dnf = act.profile_key(PROF_NF_D)

    def _restore_arch(self):
        cg._ARCH_CACHE.clear()
        cg._ARCH_CACHE.update(self._arch_saved)

    def _boot(self, form_env, archs, why="test: checkpoint config"):
        """This process is a boot of ``form_env`` whose config names ``archs``."""
        os.environ[pdflip_form.FORM_ENV] = form_env
        form = pdflip_form.parse_form(form_env)
        cg._ARCH_CACHE[(form.profile, form.model)] = (tuple(sorted(archs)), why)

    def _legacy_27b_pointer(self):
        """The NF rig's state on 24.10.: the 27B pointer and footprint, cp -a."""
        _save(self.cache, PROF_27B_D, D27_PEAKS)
        path = cg._floor_digest_path()
        with open(path, "w") as fh:
            json.dump({"hw_fingerprint": HW, "groups": {"D": self.d27}}, fh)
        return path

    def _floor(self, uuid=C5090, group="D"):
        return cg.corridor_floor_mib(uuid, group=group, cache_dir=self.cache)


class TheReproNfReadsThe27bRecord(_Base):
    """THE REPRO of #232. Pre-#233 every card below read MEASURED-D 767/701/700."""

    def test_nf_boot_refuses_the_27b_legacy_record_and_takes_the_law(self):
        self._legacy_27b_pointer()
        self._boot(FORM_NF, ARCH_NF)
        for uuid in D27_PEAKS:
            f = self._floor(uuid)
            self.assertEqual(f.source, cg.FLOOR_SOURCE_FALLBACK, f.line)
            self.assertEqual(f.transient_mib, cg.CORRIDOR_LAW_MIB, f.line)
            self.assertFalse(f.measured, f.line)
            self.assertIn(cg.RECORD_FOREIGN_TAG, f.provenance)
            self.assertIn("Qwen3_5ForConditionalGeneration", f.provenance)
            self.assertIn("Qwen4ExpForConditionalGeneration", f.provenance)
            self.assertIn(f"falling back to the stated law {cg.CORRIDOR_LAW_MIB}", f.provenance)

    def test_danger_direction_no_card_gets_a_floor_below_the_law(self):
        self._legacy_27b_pointer()
        self._boot(FORM_NF, ARCH_NF)
        for uuid, low in D27_PEAKS.items():
            self.assertGreaterEqual(self._floor(uuid).mib, cg.CORRIDOR_LAW_MIB)
            self.assertGreater(self._floor(uuid).mib, low)

    def test_unreadable_architectures_make_it_unknown_origin_not_own(self):
        self._legacy_27b_pointer()
        self._boot(FORM_NF, ())  # the container cannot read the checkpoint config
        cg._ARCH_CACHE[("nextflash", CKPT_NF)] = ((), "checkpoint config unreadable (test)")
        f = self._floor()
        self.assertEqual(f.source, cg.FLOOR_SOURCE_FALLBACK, f.line)
        self.assertIn(cg.RECORD_UNKNOWN_TAG, f.provenance)
        self.assertFalse(f.measured, f.line)

    def test_the_hand_set_law_still_wins_named(self):
        self._legacy_27b_pointer()
        self._boot(FORM_NF, ARCH_NF)
        os.environ[cg.LAW_ENV] = "1104"
        f = self._floor()
        self.assertEqual(f.source, cg.FLOOR_SOURCE_ENV)
        self.assertEqual(f.mib, 1104)

    def test_the_refusal_is_logged_once_with_its_stable_tag(self):
        self._legacy_27b_pointer()
        self._boot(FORM_NF, ARCH_NF)
        with self.assertLogs(cg.logger, level="WARNING") as cm:
            self._floor(C5090)
            self._floor(C5090)
        hits = [r for r in cm.output if cg.RECORD_FOREIGN_TAG in r]
        self.assertEqual(len(hits), 1, cm.output)

    def test_another_rig_stays_the_verdict_where_it_was(self):
        """REFUTER FIX 4 runs first: a foreign record on another rig still
        reads 'another rig' (the planner goldens carry that sentence)."""
        self._legacy_27b_pointer()
        self._boot(FORM_NF, ARCH_NF)
        saved = list(cg._RIG_FP_CACHE)
        self.addCleanup(lambda: cg._RIG_FP_CACHE.__setitem__(slice(None), saved))
        cg._RIG_FP_CACHE[:] = ["0deb25e84b18", None]
        f = self._floor()
        self.assertEqual(f.source, cg.FLOOR_SOURCE_FALLBACK)
        self.assertIn("another rig", f.provenance)


class LegacyRecordsMigrateOnlyForTheirOwnArchitecture(_Base):
    """Altbestand: a record without identity is unknown origin, EXCEPT where
    the footprint it names states this boot's architectures in its own
    content. Without that rule the first post-#233 27B boot would lose its
    P floors 1055/1095 (below them: the law 1024 is LOWER on two cards)."""

    def test_27b_boot_keeps_its_legacy_record(self):
        self._legacy_27b_pointer()
        self._boot(FORM_27B, ARCH_27B)
        for uuid, mib in D27_PEAKS.items():
            f = self._floor(uuid)
            self.assertEqual(f.source, "MEASURED-D", f.line)
            self.assertEqual(f.transient_mib, mib)
            self.assertIn(cg.RECORD_LEGACY_TAG, f.provenance)

    def test_a_legacy_footprint_without_profile_content_is_unknown_origin(self):
        path = act.footprint_cache_path(HW, "olddigest", self.cache)
        with open(path, "w") as fh:
            json.dump({
                "version": act.PHASE_FOOTPRINT_VERSION, "hw_fingerprint": HW,
                "profile_digest": "olddigest", "profile": {},
                "cards": {C5090: act.PhaseFootprint(
                    600, 0, act.FootprintProvenance.MEASURED_PEAK, "old",
                    C5090, "olddigest").to_json()},
            }, fh)
        with open(cg._floor_digest_path(), "w") as fh:
            json.dump({"hw_fingerprint": HW, "groups": {"D": "olddigest"}}, fh)
        self._boot(FORM_27B, ARCH_27B)
        f = self._floor()
        self.assertEqual(f.source, cg.FLOOR_SOURCE_FALLBACK, f.line)
        self.assertIn(cg.RECORD_UNKNOWN_TAG, f.provenance)

    def test_without_a_published_form_the_pre_233_read_stands_byte_identical(self):
        """A process with no form (desk, non-pdflip) has no identity to
        compare; it keeps the pre-#233 read and adds nothing to the line
        (``TheFingerprintOfThePointerIsChecked`` pins 'nothing extra')."""
        self._legacy_27b_pointer()
        f = self._floor()
        self.assertEqual(f.source, "MEASURED-D")
        self.assertNotIn("#233", f.provenance)


class OwnAndForeignIdentityRecords(_Base):
    def test_own_record_is_used(self):
        _save(self.cache, PROF_27B_D, D27_PEAKS)
        self._boot(FORM_27B, ARCH_27B)
        self.assertTrue(cg.publish_floor_profile("D", PROF_27B_D, HW))
        f = self._floor()
        self.assertEqual(f.source, "MEASURED-D", f.line)
        self.assertEqual(f.transient_mib, 767)
        self.assertIn(cg.RECORD_OWN_TAG, f.provenance)
        rec = _pointer()["records"][f"{CKPT_27B}|dense"]
        self.assertEqual(rec["groups"], {"D": self.d27})
        self.assertEqual(rec["identity"]["architectures"], list(ARCH_27B))

    def test_another_identitys_record_is_refused_by_name(self):
        _save(self.cache, PROF_27B_D, D27_PEAKS)
        self._boot(FORM_27B, ARCH_27B)
        cg.publish_floor_profile("D", PROF_27B_D, HW)
        self._boot(FORM_NF, ARCH_NF)
        f = self._floor()
        self.assertEqual(f.source, cg.FLOOR_SOURCE_FALLBACK, f.line)
        self.assertIn(cg.RECORD_FOREIGN_TAG, f.provenance)
        self.assertIn(f"{CKPT_27B}|dense", f.provenance)

    def test_same_checkpoint_name_other_architecture_is_foreign(self):
        """A directory re-populated with another model keeps its NAME
        (``model_key`` doc): the arch term of the key must still separate it.
        THE MUTANT 'architecture removed from the key' reads MEASURED here."""
        _save(self.cache, PROF_27B_D, D27_PEAKS)
        self._boot(_form_env("dense", "none", "dflash", "cold", "paged_dcp", "qwen27b", "reused-dir"), ARCH_27B)
        cg.publish_floor_profile("D", PROF_27B_D, HW)
        self._boot(_form_env("moe", "offload", "mtp", "none", "qsa_forma_dcp", "nextflash", "reused-dir"), ARCH_NF)
        f = self._floor()
        self.assertEqual(f.source, cg.FLOOR_SOURCE_FALLBACK, f.line)
        self.assertIn(cg.RECORD_FOREIGN_TAG, f.provenance)

    def test_same_name_other_arch_is_foreign_even_with_an_unreadable_config(self):
        """The arch TERM OF THE KEY carries the separation on its own when the
        architectures cross-check cannot run (config unreadable). THE MUTANT
        'architecture removed from the key' prices this boot at 767 here."""
        _save(self.cache, PROF_27B_D, D27_PEAKS)
        self._boot(_form_env("dense", "none", "dflash", "cold", "paged_dcp", "qwen27b", "reused-dir"), ARCH_27B)
        cg.publish_floor_profile("D", PROF_27B_D, HW)
        self._boot(_form_env("moe", "offload", "mtp", "none", "qsa_forma_dcp", "nextflash", "reused-dir"), ())
        cg._ARCH_CACHE[("nextflash", "reused-dir")] = ((), "checkpoint config unreadable (test)")
        f = self._floor()
        self.assertEqual(f.source, cg.FLOOR_SOURCE_FALLBACK, f.line)
        self.assertGreaterEqual(f.mib, cg.CORRIDOR_LAW_MIB)

    def test_own_key_with_contradicting_architectures_is_foreign(self):
        _save(self.cache, PROF_27B_D, D27_PEAKS)
        self._boot(FORM_27B, ARCH_27B)
        cg.publish_floor_profile("D", PROF_27B_D, HW)
        cg._ARCH_CACHE[("qwen27b", CKPT_27B)] = (ARCH_NF, "test: re-populated dir")
        f = self._floor()
        self.assertEqual(f.source, cg.FLOOR_SOURCE_FALLBACK, f.line)
        self.assertIn(cg.RECORD_FOREIGN_TAG, f.provenance)

    def test_the_legacy_map_is_never_rewritten_by_an_identity_writer(self):
        self._legacy_27b_pointer()
        self._boot(FORM_NF, ARCH_NF)
        cg.publish_floor_profile("D", PROF_NF_D, HW)
        self.assertEqual(_pointer()["groups"], {"D": self.d27})

    def test_collision_is_judged_inside_this_identity(self):
        self._boot(FORM_27B, ARCH_27B)
        cg.publish_floor_profile("D", PROF_27B_D, HW)
        cg.publish_floor_profile("P", PROF_27B_D, HW)
        self.assertEqual(cg.digest_group_collision(self.d27, "D"), "P")
        self._boot(FORM_NF, ARCH_NF)
        self.assertIsNone(cg.digest_group_collision(self.d27, "D"))


class PublishOnAMissSelfCorrects(_Base):
    """(a): a boot without a record of its own writes one, so the next read is
    its own identity -> no footprint -> the named law, never the 27B record."""

    def test_nf_publishes_on_a_miss_and_then_reads_its_own_entry(self):
        self._legacy_27b_pointer()
        self._boot(FORM_NF, ARCH_NF)
        os.environ[cg.GROUP_ENV] = "D"
        self.assertTrue(cg.publish_boot_floor_profile(None, profile=PROF_NF_D, footprint_hit=False))
        rec = _pointer()["records"][f"{CKPT_NF}|moe"]
        self.assertEqual(rec["groups"], {"D": self.dnf})
        f = self._floor()
        self.assertEqual(f.source, cg.FLOOR_SOURCE_FALLBACK, f.line)
        self.assertIn(cg.RECORD_OWN_TAG, f.provenance)
        self.assertIn("no phase footprint", f.provenance)

    def test_outside_weg2_nothing_is_published(self):
        self._boot(FORM_NF, ARCH_NF)
        self.assertIsNone(cg.publish_boot_floor_profile(None, profile=PROF_NF_D))
        self.assertFalse(os.path.exists(cg._floor_digest_path()))

    def test_activation_reserve_mb_publishes_on_a_footprint_miss(self):
        """MUTANT 'publish only on a hit' (the pre-#233 server_args) is red here."""
        from flliper.srt.server_args import ServerArgs

        self._boot(FORM_NF, ARCH_NF)
        os.environ[cg.GROUP_ENV] = "D"
        stub = types.SimpleNamespace(
            model_path=os.path.join(self.cache, "no-such-model"),
            trust_remote_code=False, revision=None, model_config_parser=None,
            chunked_prefill_size=4096, tp_size=3, pp_size=1,
            kv_cache_dtype="fp8_e4m3", speculative_num_draft_tokens=3,
            cuda_graph_config=types.SimpleNamespace(decode=types.SimpleNamespace(max_bs=6)),
            mamba_pre_capture_reserve_mb=lambda gpu_mem: 123.0,
        )
        with mock.patch("flliper.srt.mem_ledger.activation.resolve_phase_footprint", return_value=None), \
                mock.patch("flliper.srt.mem_ledger.calibration.live_fingerprint", return_value=None):
            got = ServerArgs.activation_reserve_mb(stub, 1000.0, card_uuid=C5090)
        self.assertEqual(got, 123.0)
        recs = _pointer().get("records") or {}
        self.assertIn(f"{CKPT_NF}|moe", recs)
        self.assertIn("D", recs[f"{CKPT_NF}|moe"]["groups"])

    def test_every_group_rank_reaches_the_writer(self):
        """The writer sits where every target runner passes (alloc_memory_pool)."""
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
        path = os.path.join(root, "python", "flliper", "srt", "model_executor", "model_runner.py")
        tree = ast.parse(open(path).read())
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "alloc_memory_pool"
        )
        calls = {
            getattr(c.func, "id", getattr(c.func, "attr", ""))
            for c in ast.walk(fn) if isinstance(c, ast.Call)
        }
        self.assertIn("publish_boot_floor_profile", calls)


class TheSwitchIsOnByDefault(_Base):
    def test_default_on(self):
        self.assertTrue(cg.record_identity_enabled())

    def test_explicit_off_restores_the_pre_233_read_and_says_so(self):
        self._legacy_27b_pointer()
        self._boot(FORM_NF, ARCH_NF)
        os.environ[cg.RECORD_IDENTITY_ENV] = "0"
        f = self._floor()
        self.assertEqual(f.source, "MEASURED-D")
        self.assertIn("switched off", f.provenance)


class BootArchitecturesComeFromTheRegistryCheckpoint(_Base):
    def test_config_architectures_of_the_registry_path(self):
        ck = os.path.join(self.cache, "ckpt")
        os.makedirs(ck)
        with open(os.path.join(ck, "config.json"), "w") as fh:
            json.dump({"architectures": list(ARCH_NF)}, fh)
        row = types.SimpleNamespace(formats={"int4": types.SimpleNamespace(checkpoint=ck)})
        with mock.patch.object(pdflip_form, "format_of", return_value="int4"), \
                mock.patch.object(pdflip_form, "profile_row", return_value=row):
            got, why = cg.boot_architectures({"profile": "nextflash", "checkpoint": "x"})
        self.assertEqual(got, ARCH_NF)
        self.assertIn("config.json", why)

    def test_an_unlisted_checkpoint_is_unreadable_not_a_match(self):
        with mock.patch.object(pdflip_form, "format_of", return_value=""):
            got, why = cg.boot_architectures({"profile": "nextflash", "checkpoint": "y"})
        self.assertEqual(got, ())
        self.assertIn("not a registry format", why)


if __name__ == "__main__":
    unittest.main()
