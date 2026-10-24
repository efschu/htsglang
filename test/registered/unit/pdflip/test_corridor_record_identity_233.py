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

REVIEW AUFLAGEN (24.10., deskq/done/review-corridor-arch-233-1024.md):

* MUSS 1 -- ONE pointer state per boot: the launcher freezes its resolution
  before the first floor read; a writer that publishes mid-boot never moves a
  reader of the same boot (``OnePointerStatePerBoot``; MUTANT "reader reads
  live" is red there);
* MUSS 2 -- the recipe is in the key (``TheRecipeIsInTheKey``; MUTANT "recipe
  removed from the key" is red) and a footprint measured over less than one
  chunk is never MEASURED (``ShortBootFootprintsAreNeverMeasured``; MUTANT
  "short-boot record accepted" is red);
* SOLL 1 (publish failure is a WARNING), SOLL 5 (two writers under flock, one
  write per group and process).

HERMETIC: no NVML, no torch device, no GPU, no boot, no real checkpoint. The
rig fingerprint is pinned, footprints are written into a tmp store through the
shipped writer, and the boot's architectures are pinned per test (the registry
path of a checkpoint is not on a desk box).
"""

from __future__ import annotations

import ast
import json
import multiprocessing
import os
import shlex
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

#: Group argvs (footprint-relevant part plus noise the recipe must ignore).
#: The 27B seed D recipe matches the cc0ac5caf91e footprint content; today's
#: 27B D runs DFLASH spec 8 (#233 report "27B line"); the NF D runs MTP.
ARGV_27B_SEED_D = [
    "py", "-m", "flliper.launch_server", "--model-path", "/m/27b",
    "--chunked-prefill-size", "4096", "--tp-size", "3", "--pp-size", "1",
    "--kv-cache-dtype", "fp8_e4m3", "--speculative-algorithm", "NEXTN",
    "--speculative-num-draft-tokens", "3", "--max-running-requests", "24",
    "--rank-gpu-memory-mib", "26800,17656,17880", "--port", "30001",
]
ARGV_27B_TODAY_D = [
    "py", "-m", "flliper.launch_server", "--chunked-prefill-size", "4096",
    "--tp-size", "3", "--pp-size", "1", "--kv-cache-dtype", "fp8_e4m3",
    "--speculative-algorithm", "DFLASH", "--speculative-num-draft-tokens", "8",
    "--max-running-requests", "32",
]
ARGV_NF_D = [
    "py", "-m", "flliper.launch_server", "--chunked-prefill-size", "4096",
    "--tp-size", "3", "--pp-size", "1", "--kv-cache-dtype", "fp8_e4m3",
    "--speculative-algorithm", "NEXTN", "--speculative-num-draft-tokens", "3",
    "--max-running-requests", "6",
]


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


def _save(cache_dir, profile, peaks, source="test #233: probe_activation.py ingest") -> str:
    """A footprint file through the SHIPPED writer (it records ``profile``)."""
    fps = {
        u: act.PhaseFootprint(
            activation_mib=int(m), capture_mib=0,
            provenance=act.FootprintProvenance.MEASURED_PEAK,
            source=source, card_uuid=u,
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
        for k in (cg.LAW_ENV, cg.GROUP_ENV, cg.RECORD_IDENTITY_ENV, pdflip_form.FORM_ENV,
                  cg.POINTER_FROZEN_ENV, cg.RECIPE_ENV, cg.RECIPE_SELF_ENV):
            os.environ.pop(k, None)
        self._arch_saved = dict(cg._ARCH_CACHE)
        cg._ARCH_CACHE.clear()
        self.addCleanup(self._restore_arch)
        cg._RECORD_NOTED.clear()
        cg._BOOT_PUBLISHED.clear()
        cg._POINTER_READ_NOTED.clear()
        cg._PUBLISH_FAILED_NOTED.clear()
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

    def _arm(self, group, argv, *, self_recipe=True):
        """The launcher published ``group``'s recipe of ``argv`` (RECIPE_ENV);
        with ``self_recipe`` this process is also a rank of that group
        (RECIPE_SELF_ENV, armed by launch_group). Returns the recipe hash."""
        rec = cg.recipe_from_argv(argv)
        cur = json.loads(os.environ.get(cg.RECIPE_ENV) or "{}")
        cur[group] = rec
        os.environ[cg.RECIPE_ENV] = json.dumps(cur)
        if self_recipe:
            os.environ[cg.RECIPE_SELF_ENV] = str(rec["hash"])
        return str(rec["hash"])

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
        """... when its recipe is the one the legacy footprint was measured
        under (MUSS 2: the recipe is read from the footprint's own content)."""
        self._legacy_27b_pointer()
        self._boot(FORM_27B, ARCH_27B)
        self._arm("D", ARGV_27B_SEED_D)
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
    def setUp(self):
        super().setUp()
        self.r27 = self._arm("D", ARGV_27B_SEED_D)

    def test_own_record_is_used(self):
        _save(self.cache, PROF_27B_D, D27_PEAKS)
        self._boot(FORM_27B, ARCH_27B)
        self.assertTrue(cg.publish_floor_profile("D", PROF_27B_D, HW))
        f = self._floor()
        self.assertEqual(f.source, "MEASURED-D", f.line)
        self.assertEqual(f.transient_mib, 767)
        self.assertIn(cg.RECORD_OWN_TAG, f.provenance)
        rec = _pointer()["records"][f"{CKPT_27B}|dense|{self.r27}"]
        self.assertEqual(rec["identity"]["recipe"], self.r27)
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
        self.assertIn(f"{CKPT_27B}|dense|{self.r27}", f.provenance)

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
        self._arm("P", ARGV_27B_SEED_D)  # P and D share recipe AND digest here
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
        rnf = self._arm("D", ARGV_NF_D)
        os.environ[cg.GROUP_ENV] = "D"
        self.assertTrue(cg.publish_boot_floor_profile(None, profile=PROF_NF_D, footprint_hit=False))
        rec = _pointer()["records"][f"{CKPT_NF}|moe|{rnf}"]
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
        rnf = self._arm("D", ARGV_NF_D)
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
        self.assertIn(f"{CKPT_NF}|moe|{rnf}", recs)
        self.assertIn("D", recs[f"{CKPT_NF}|moe|{rnf}"]["groups"])

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


class OnePointerStatePerBoot(_Base):
    """MUSS 1: the launcher freezes the pointer resolution before the first
    floor read; a rank that publishes mid-boot (alloc_memory_pool) writes for
    the NEXT boot and never moves a reader of THIS boot. MUTANT "reader reads
    live" (``_published_digest_for_boot`` ignoring the frozen state) is red in
    the first two tests: the 27B-Altbestand path of the review (launcher 767 ->
    publish -> rank law 1024) and its mirror (launcher law -> a low own record
    appears mid-boot -> rank 39 MiB)."""

    def test_a_mid_boot_publish_does_not_move_the_rank_readers(self):
        self._legacy_27b_pointer()
        self._boot(FORM_27B, ARCH_27B)
        self._arm("D", ARGV_27B_SEED_D)
        lines = cg.freeze_boot_pointer(cache_dir=self.cache)
        self.assertTrue(any(cg.POINTER_FROZEN_TAG in l and "group=D source=FROZEN" in l for l in lines), lines)
        launcher = {u: self._floor(u) for u in D27_PEAKS}
        # The ranks publish their own (new) digest under their key, mid-boot:
        # after the publish the LIVE pointer resolves to the own record, whose
        # footprint does not exist -> the law 1024 (the review's two floors).
        os.environ[cg.GROUP_ENV] = "D"
        prof_now = act.ActivationProfile(ARCH_27B, 4096, 3, 1, "fp8_e4m3", 8, 32)
        self.assertTrue(cg.publish_boot_floor_profile(None, profile=prof_now, footprint_hit=False))
        self.assertIn(act.profile_key(prof_now), json.dumps(_pointer()["records"]))
        for u, f0 in launcher.items():
            f1 = self._floor(u)
            self.assertEqual((f1.mib, f1.source), (f0.mib, f0.source), f1.line)
            self.assertEqual(f1.transient_mib, D27_PEAKS[u])
            self.assertIn(cg.RECORD_LEGACY_TAG, f1.provenance)

    def test_a_low_record_appearing_mid_boot_never_prices_this_boot(self):
        self._boot(FORM_27B, ARCH_27B)
        r = self._arm("D", ARGV_27B_SEED_D)
        cg.freeze_boot_pointer(cache_dir=self.cache)  # nothing published yet: the law
        low = {u: 39 for u in D27_PEAKS}
        _save(self.cache, PROF_27B_D, low)
        cg.publish_floor_profile("D", PROF_27B_D, HW, recipe=r)
        for u in D27_PEAKS:
            f = self._floor(u)
            self.assertEqual(f.source, cg.FLOOR_SOURCE_FALLBACK, f.line)
            self.assertGreaterEqual(f.mib, cg.CORRIDOR_LAW_MIB)

    def test_the_next_boot_reads_what_this_boot_wrote(self):
        """The writer keeps working -- for the NEXT boot (a new freeze)."""
        self._boot(FORM_27B, ARCH_27B)
        r = self._arm("D", ARGV_27B_SEED_D)
        cg.freeze_boot_pointer(cache_dir=self.cache)
        _save(self.cache, PROF_27B_D, D27_PEAKS)
        cg.publish_floor_profile("D", PROF_27B_D, HW, recipe=r)
        self.assertEqual(self._floor().source, cg.FLOOR_SOURCE_FALLBACK)
        cg.clear_boot_pointer_env()
        self._arm("D", ARGV_27B_SEED_D)
        cg.freeze_boot_pointer(cache_dir=self.cache)
        f = self._floor()
        self.assertEqual((f.source, f.transient_mib), ("MEASURED-D", 767), f.line)

    def test_the_read_line_names_its_source(self):
        self._legacy_27b_pointer()
        self._boot(FORM_NF, ARCH_NF)
        self._arm("D", ARGV_NF_D)
        cg.freeze_boot_pointer(cache_dir=self.cache)
        with self.assertLogs(cg.logger, level="INFO") as cm:
            self._floor()
            self._floor(group="P")
        reads = [r for r in cm.output if cg.POINTER_READ_TAG in r]
        self.assertTrue(any("group=D source=FOREIGN" in r for r in reads), reads)
        self.assertTrue(any("group=P source=LAW" in r for r in reads), reads)
        self.assertEqual(len([r for r in reads if "group=D" in r]), 1, reads)
        cg.clear_boot_pointer_env()
        with self.assertLogs(cg.logger, level="INFO") as cm:
            self._floor()
        self.assertTrue(any(cg.POINTER_READ_TAG in r and "source=LIVE" in r for r in cm.output), cm.output)

    def test_an_unreadable_frozen_state_is_the_law_not_a_live_read(self):
        self._boot(FORM_27B, ARCH_27B)
        r = self._arm("D", ARGV_27B_SEED_D)
        _save(self.cache, PROF_27B_D, D27_PEAKS)
        cg.publish_floor_profile("D", PROF_27B_D, HW, recipe=r)
        os.environ[cg.POINTER_FROZEN_ENV] = "{not json"
        f = self._floor()
        self.assertEqual(f.source, cg.FLOOR_SOURCE_FALLBACK, f.line)
        self.assertIn(cg.POINTER_FROZEN_ENV, f.provenance)

    def test_the_collision_is_judged_on_the_frozen_state(self):
        self._boot(FORM_27B, ARCH_27B)
        r = self._arm("D", ARGV_27B_SEED_D)
        self._arm("P", ARGV_27B_SEED_D)
        cg.freeze_boot_pointer(cache_dir=self.cache)
        cg.publish_floor_profile("D", PROF_27B_D, HW, recipe=r)
        cg.publish_floor_profile("P", PROF_27B_D, HW, recipe=r)
        self.assertIsNone(cg.digest_group_collision(self.d27, "D"))

    def test_launch_group_arms_the_frozen_state_and_the_real_recipe(self):
        self._boot(FORM_27B, ARCH_27B)
        self._arm("D", ARGV_27B_SEED_D, self_recipe=False)
        cg.freeze_boot_pointer(cache_dir=self.cache)
        env = {cg.POINTER_FROZEN_ENV: "stale", cg.RECIPE_SELF_ENV: "stale"}
        lines = cg.arm_group_recipe_env("D", ARGV_27B_SEED_D, env)
        self.assertEqual(env[cg.POINTER_FROZEN_ENV], os.environ[cg.POINTER_FROZEN_ENV])
        self.assertEqual(env[cg.RECIPE_ENV], os.environ[cg.RECIPE_ENV])
        self.assertEqual(env[cg.RECIPE_SELF_ENV], cg.recipe_from_argv(ARGV_27B_SEED_D)["hash"])
        self.assertTrue(any(cg.RECIPE_TAG in l and "agree" in l for l in lines), lines)

    def test_the_launcher_freezes_before_its_first_floor_read(self):
        """Source order in ``launcher.main``: clear the inherited state, publish
        the recipe and freeze -- all before the first ``budgets_from_dc``."""
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
        src = open(os.path.join(root, "python", "flliper", "srt", "pdflip", "launcher.py")).read()
        tree = ast.parse(src)
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")

        def first_line(name):
            return min(
                c.lineno for c in ast.walk(main) if isinstance(c, ast.Call)
                and getattr(c.func, "attr", getattr(c.func, "id", "")) == name
            )
        self.assertLess(first_line("clear_boot_pointer_env"), first_line("freeze_boot_pointer"))
        self.assertLess(first_line("corridor_recipe_probe"), first_line("freeze_boot_pointer"))
        self.assertLess(first_line("freeze_boot_pointer"), first_line("budgets_from_dc"))
        lg = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "launch_group")
        self.assertIn("arm_group_recipe_env", {getattr(c.func, "attr", "") for c in ast.walk(lg) if isinstance(c, ast.Call)})
        # build_env pops the frozen state (the group gets it ONLY from launch_group)
        be = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_env")
        attrs = {getattr(a, "attr", "") for a in ast.walk(be)}
        self.assertTrue({"POINTER_FROZEN_ENV", "RECIPE_ENV", "RECIPE_SELF_ENV"} <= attrs, attrs & {"POINTER_FROZEN_ENV"})


class TheRecipeIsInTheKey(_Base):
    """MUSS 2: ``checkpoint|arch|recipe``. A boot of another recipe finds no
    record of its own -> the law, named. MUTANT "recipe removed from the key"
    (``record_identity_key`` ignoring ``recipe``) prices boot B with recipe A's
    LOW footprint in ``test_another_recipe_never_reads_this_recipes_footprint``."""

    def test_another_recipe_never_reads_this_recipes_footprint(self):
        low = {u: 39 for u in D27_PEAKS}
        prof_a = act.ActivationProfile(ARCH_27B, 512, 3, 1, "fp8_e4m3", 0, 24)
        _save(self.cache, prof_a, low)
        self._boot(FORM_27B, ARCH_27B)
        ra = self._arm("D", ["--chunked-prefill-size", "512", "--tp-size", "3", "--pp-size", "1",
                             "--kv-cache-dtype", "fp8_e4m3"])
        cg.publish_floor_profile("D", prof_a, HW)
        cg.freeze_boot_pointer(cache_dir=self.cache)
        self.assertEqual(self._floor().transient_mib, 39)  # recipe A reads its own
        cg.clear_boot_pointer_env()
        cg._RECORD_NOTED.clear()
        rb = self._arm("D", ARGV_27B_TODAY_D)
        self.assertNotEqual(ra, rb)
        cg.freeze_boot_pointer(cache_dir=self.cache)
        for u in D27_PEAKS:
            f = self._floor(u)
            self.assertEqual(f.source, cg.FLOOR_SOURCE_FALLBACK, f.line)
            self.assertGreaterEqual(f.mib, cg.CORRIDOR_LAW_MIB, f.line)
            self.assertIn(cg.RECORD_OTHER_RECIPE_TAG, f.provenance)
            self.assertIn(ra, f.provenance)

    def test_the_recipe_reads_only_footprint_flags_and_argparse_order(self):
        a = cg.recipe_from_argv(ARGV_27B_SEED_D)
        noisy = ARGV_27B_SEED_D[:-4] + ["--rank-gpu-memory-mib", "1,2,3", "--port", "9",
                                         "--mem-fraction-static", "0.5", "--max-total-tokens", "77",
                                         "--pp-stage-ratio", "1,1,1"]
        self.assertEqual(cg.recipe_from_argv(noisy)["hash"], a["hash"])
        i = ARGV_27B_SEED_D.index("--tp-size")
        eq = ARGV_27B_SEED_D[:i] + ARGV_27B_SEED_D[i + 2:] + ["--tensor-parallel-size=3"]
        self.assertEqual(cg.recipe_from_argv(eq)["hash"], a["hash"])
        later = ARGV_27B_SEED_D + ["--chunked-prefill-size", "2048"]
        self.assertEqual(cg.recipe_from_argv(later)["items"]["--chunked-prefill-size"], "2048")
        self.assertNotEqual(cg.recipe_from_argv(ARGV_27B_SEED_D + ["--enable-mixed-chunk"])["hash"], a["hash"])
        self.assertEqual(
            cg.recipe_relevant_tokens(["--rank-gpu-memory-mib", "9", "--chunked-prefill-size", "8",
                                       "--disable-cuda-graph", "--kv-cache-dtype=fp8"]),
            ["--chunked-prefill-size", "8", "--disable-cuda-graph", "--kv-cache-dtype=fp8"])

    def test_the_27b_legacy_seed_is_another_recipe_for_todays_27b_d(self):
        """Altbestand: determinable from the footprint's own profile; today's
        27B D (DFLASH spec 8) is not the seed recipe (spec 3) -> the law."""
        self._legacy_27b_pointer()
        self._boot(FORM_27B, ARCH_27B)
        self._arm("D", ARGV_27B_TODAY_D)
        f = self._floor()
        self.assertEqual(f.source, cg.FLOOR_SOURCE_FALLBACK, f.line)
        self.assertIn(cg.RECORD_OTHER_RECIPE_TAG, f.provenance)
        self.assertIn("--speculative-num-draft-tokens", f.provenance)

    def test_an_undeterminable_legacy_recipe_is_unknown_origin(self):
        self._legacy_27b_pointer()
        self._boot(FORM_27B, ARCH_27B)
        self._arm("D", [t for t in ARGV_27B_SEED_D if t not in ("--kv-cache-dtype", "fp8_e4m3")])
        f = self._floor()
        self.assertEqual(f.source, cg.FLOOR_SOURCE_FALLBACK, f.line)
        self.assertIn(cg.RECORD_UNKNOWN_TAG, f.provenance)
        self.assertIn("--kv-cache-dtype", f.provenance)

    def test_without_a_published_recipe_nothing_is_own(self):
        _save(self.cache, PROF_27B_D, D27_PEAKS)
        self._boot(FORM_27B, ARCH_27B)
        cg.publish_floor_profile("D", PROF_27B_D, HW, recipe="r1")
        f = self._floor()
        self.assertEqual(f.source, cg.FLOOR_SOURCE_FALLBACK, f.line)
        self.assertIn(cg.RECORD_OTHER_RECIPE_TAG, f.provenance)

    def test_a_probe_argv_drift_is_named_and_the_real_recipe_is_the_key(self):
        self._arm("D", ARGV_27B_SEED_D, self_recipe=False)
        env = {}
        lines = cg.arm_group_recipe_env("D", ARGV_27B_TODAY_D, env)
        self.assertEqual(env[cg.RECIPE_SELF_ENV], cg.recipe_from_argv(ARGV_27B_TODAY_D)["hash"])
        drift = [l for l in lines if cg.RECIPE_DRIFT_TAG in l]
        self.assertEqual(len(drift), 1, lines)
        self.assertIn("--speculative-num-draft-tokens", drift[0])


class ShortBootFootprintsAreNeverMeasured(_Base):
    """MUSS 2, second half: minimum evidence = the stated prefill >= one chunk
    of the footprint's own profile. MUTANT "short-boot record accepted" (the
    THIN gate removed) reads MEASURED-D 39 in the first test."""

    PROF_P512 = act.ActivationProfile(ARCH_27B, 512, 3, 1, "fp8_e4m3", 0, 24)

    def _own(self, profile, peaks, source):
        _save(self.cache, profile, peaks, source=source)
        self._boot(FORM_27B, ARCH_27B)
        r = self._arm("D", ARGV_27B_SEED_D)
        cg.publish_floor_profile("D", profile, HW, recipe=r)

    def test_a_122_token_footprint_is_refused_by_name(self):
        self._own(self.PROF_P512, {u: 39 for u in D27_PEAKS},
                  "probe_activation.py: ... reserved peak 27794 MiB; prefill of 122 tokens")
        for u in D27_PEAKS:
            f = self._floor(u)
            self.assertEqual(f.source, cg.FLOOR_SOURCE_FALLBACK, f.line)
            self.assertFalse(f.actuates, f.line)
            self.assertIn(cg.RECORD_THIN_TAG, f.provenance)
            self.assertIn("122", f.provenance)

    def test_a_full_chunk_footprint_is_measured(self):
        self._own(PROF_27B_D, D27_PEAKS, "probe_activation.py: ...; prefill of 4096 tokens")
        f = self._floor()
        self.assertEqual((f.source, f.transient_mib), ("MEASURED-D", 767), f.line)

    def test_an_unknown_count_is_refused(self):
        self._own(PROF_27B_D, D27_PEAKS, "probe_activation.py: ...; prefill of ? tokens")
        self.assertIn(cg.RECORD_THIN_TAG, self._floor().provenance)

    def test_the_ingest_warns_on_a_short_prefill(self):
        import importlib.util
        import io
        from contextlib import redirect_stdout

        from flliper.srt.mem_ledger import activation_probe as ap

        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
        spec = importlib.util.spec_from_file_location(
            "probe_activation_233", os.path.join(root, "scripts", "vram_ledger", "probe_activation.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        dumps = os.path.join(self.cache, "dumps")
        os.makedirs(dumps)
        ap.write_footprint_dump(
            rank=0, card_uuid=C5090, hw_fingerprint=HW,
            profile_canonical=self.PROF_P512.canonical(),
            activation_peak_bytes=(16000 + 39) << 20, capture_bytes=35 << 20,
            reserved_peak_bytes=(16000 + 80) << 20, prefill_tokens=122,
            dump_dir=dumps, peak_floor_bytes=16000 << 20, group="D",
            boot_token_override="t233")
        buf = io.StringIO()
        with redirect_stdout(buf):
            m.ingest(dumps, os.path.join(self.cache, "store"), boot_token="t233")
        self.assertIn(cg.RECORD_THIN_TAG, buf.getvalue())


def _concurrent_writer(path, group, recipe, n):
    os.environ[cg.FLOOR_DIGEST_FILE_ENV] = path
    for i in range(n):
        prof = act.ActivationProfile(ARCH_27B, 4096, 3 if group == "D" else 1, 1 if group == "D" else 3,
                                     "fp8_e4m3", 3, 24)
        cg.publish_floor_profile(group, prof, HW, identity={"checkpoint": CKPT_27B, "arch": "dense"},
                                 recipe=recipe)


class PublishIsSafeAndLoud(_Base):
    def test_two_groups_two_writers_each_under_flock_keep_both_groups(self):
        """SOLL 5: P and D, two processes each, at the same time."""
        path = cg._floor_digest_path()
        ctx = multiprocessing.get_context("fork")
        procs = [ctx.Process(target=_concurrent_writer, args=(path, g, "r" + g, 25))
                 for g in ("P", "D", "P", "D")]
        for p in procs:
            p.start()
        for p in procs:
            p.join(60)
            self.assertEqual(p.exitcode, 0)
        recs = _pointer()["records"]
        self.assertIn("P", recs[f"{CKPT_27B}|dense|rP"]["groups"])
        self.assertIn("D", recs[f"{CKPT_27B}|dense|rD"]["groups"])

    def test_one_write_per_group_and_process_a_second_digest_is_refused(self):
        self._boot(FORM_27B, ARCH_27B)
        self._arm("D", ARGV_27B_SEED_D)
        os.environ[cg.GROUP_ENV] = "D"
        self.assertTrue(cg.publish_boot_floor_profile(None, profile=PROF_27B_D))
        other = act.ActivationProfile(ARCH_27B, 2048, 3, 1, "fp8_e4m3", 4, 24)
        with self.assertLogs(cg.logger, level="WARNING") as cm:
            self.assertIsNone(cg.publish_boot_floor_profile(None, profile=other))
        self.assertTrue(any("SECOND DIGEST" in r for r in cm.output), cm.output)
        groups = [r["groups"] for r in _pointer()["records"].values()]
        self.assertEqual(groups, [{"D": self.d27}])

    def test_a_publish_failure_is_a_warning_once(self):
        """SOLL 1: never DEBUG-silent -- the self-correction hangs on it."""
        blocker = os.path.join(self.cache, "a-file")
        open(blocker, "w").close()
        os.environ[cg.FLOOR_DIGEST_FILE_ENV] = os.path.join(blocker, "pointer.json")
        self._boot(FORM_27B, ARCH_27B)
        with self.assertLogs(cg.logger, level="WARNING") as cm:
            self.assertIsNone(cg.publish_floor_profile("D", PROF_27B_D, HW, recipe="r"))
            self.assertIsNone(cg.publish_floor_profile("D", PROF_27B_D, HW, recipe="r"))
        failed = [r for r in cm.output if cg.RECORD_PUBLISH_TAG in r and "FAILED" in r]
        self.assertEqual(len(failed), 1, cm.output)


class TheLaunchersRecipeProbe(_Base):
    """The launcher's pre-budget recipe probe (``corridor_recipe_probe``): read
    off the SAME argv builders with sentinel budgets. It must not move with the
    budget values (the budget is priced from the floor) and must see the
    operator's recipe flags in ``--extra-*``."""

    def _ns(self, *extra):
        from flliper.srt.pdflip import launcher as L

        return L, L.build_parser().parse_args(["--tree", "/t", "--tag", "t233", *extra])

    def test_the_probe_names_both_groups_and_ignores_the_budget_values(self):
        L, ns = self._ns()
        rec = L.corridor_recipe_probe(ns, [1, 2, 3], int(ns.p_bs), int(ns.d_bs), False)
        self.assertEqual(sorted(rec), ["D", "P"])
        self.assertEqual(rec["D"]["items"].get("--tp-size"), "3")
        real_d = L.argv_d("py", ns.model, [26800, 17656, 17880], 7, 512, L.RING_FORM_SENTINEL_STORE_CFG,
                          shlex.split(ns.extra_d), d_bs=int(ns.d_bs))
        self.assertEqual(cg.recipe_from_argv(real_d)["hash"], rec["D"]["hash"])

    def test_the_probe_sees_an_operator_recipe_flag_and_not_a_budget_flag(self):
        L, ns0 = self._ns()
        base = L.corridor_recipe_probe(ns0, [1, 2, 3], int(ns0.p_bs), int(ns0.d_bs), False)
        _, ns1 = self._ns("--extra-d", "--chunked-prefill-size 1024 --rank-gpu-memory-mib 99999,99999,99999")
        got = L.corridor_recipe_probe(ns1, [1, 2, 3], int(ns1.p_bs), int(ns1.d_bs), False)
        self.assertEqual(got["D"]["items"]["--chunked-prefill-size"], "1024")
        self.assertNotEqual(got["D"]["hash"], base["D"]["hash"])


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
