# SPDX-License-Identifier: Apache-2.0
"""#1395 -- the boot belongs in the phase-footprint dump's identity, the
sibling gap in #1292's own mechanism.

Coordinator's finding: the xsn31/8 phase-footprint dump EXISTS in the boot
log's own proof of a write (03:21:36Z, PP0, legs=0) and does NOT exist on
disk any more. Root cause: ``activation_probe.dump_filename(rank, group)``
carries no boot identity at all -- #1292 folded the Weg-2 GROUP into the
filename (fixing P-over-D within one boot) but never the BOOT itself, so a
second boot of the identical form (same group, same rank, same profile
digest -- #1292's own W18 collision check cannot see this axis, since two
boots of one form share one digest by construction) silently overwrites the
first boot's dump the moment ``write_footprint_dump`` runs again.

THE FIX, mirroring ``weg2.lane_coverage``'s own #1395 shape (extended, not
rebuilt) one file over:

1. ``activation_probe.boot_token()`` -- reads ``SGLANG_WEG2_BOOT_TOKEN``,
   published UNCONDITIONALLY by ``weg2/launcher.py build_env``
   (``weg2_boot_token(ns)``, memoised on the launcher's OWN namespace so
   BOTH groups' ranks -- separate OS processes -- inherit the IDENTICAL
   value; see that function's docstring for why a per-process fallback
   alone breaks the #1292 "P and D share one directory" reading).
2. ``write_footprint_dump`` now writes under
   ``<directory>/<_boot_subdir(token)>/<dump_filename(rank, group)>``, never
   flat -- two boots of the same form land in two different subdirectories,
   never the same file.
3. ``BOOT_COLLISION_CODE`` (W109) -- defense in depth, the SAME shape as
   #1292's own W18 one axis over: a write that would still overwrite an
   EXISTING file carrying a DIFFERENT ``boot_token`` is refused by name.
4. ``scripts/vram_ledger/probe_activation.py load_dumps``/``ingest`` gain
   ``boot_token=``: given, they read EXACTLY that boot's subdirectory or
   report a NAMED ABSENCE -- never another boot's dumps sitting in a
   sibling subdirectory or the flat root. THE DANGER DIRECTION NAMED BY THE
   COORDINATOR DIRECTLY: "ein Leser, der beim Fehlen des angefragten Boots
   still den neuesten fremden Dump liefert, ist teurer als eine Absenz" --
   pinned below as the file's own central mutant.

Hermetic: no driver, no CUDA, no torch device -- exactly like
``test_phase_footprint_group_collision_1292.py``, whose fixtures and helper
shape this file reuses rather than reinventing.
"""

import glob
import importlib.util
import json
import os

from sglang.srt.mem_ledger import activation_probe as ap
from sglang.srt.mem_ledger.activation import ActivationProfile
from sglang.test.test_utils import CustomTestCase

_HERE = os.path.abspath(__file__)
_ROOT = _HERE
for _ in range(5):
    _ROOT = os.path.dirname(_ROOT)
SCRIPT = os.path.join(_ROOT, "scripts", "vram_ledger", "probe_activation.py")
assert os.path.isfile(SCRIPT), SCRIPT

FP = "a191a0712717"

PROFILE = ActivationProfile(
    architectures=("Qwen3_8ForCausalLM",),
    chunked_prefill_size=4096,
    tp_size=1,
    pp_size=3,
)


def load_script():
    spec = importlib.util.spec_from_file_location("probe_activation_1395", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(rank, group, boot_token, activation_mib, dump_dir, uuid="GPU-x"):
    return ap.write_footprint_dump(
        rank=rank,
        card_uuid=uuid,
        hw_fingerprint=FP,
        profile_canonical=PROFILE.canonical(),
        activation_peak_bytes=(16000 + activation_mib) << 20,
        capture_bytes=0,
        dump_dir=str(dump_dir),
        group=group,
        boot_token_override=boot_token,
    )


class TheBootSubdirSanitisesTheToken(CustomTestCase):
    def test_colon_and_slash_become_underscore(self):
        self.assertEqual(ap._boot_subdir("xsn31:1000:111"), "xsn31_1000_111")
        self.assertEqual(ap._boot_subdir("a/b:c"), "a_b_c")

    def test_empty_token_gets_a_named_fallback_not_the_bare_directory(self):
        self.assertEqual(ap._boot_subdir(""), "no-boot-token")


class TheBootTokenIsSharedByBothGroups(CustomTestCase):
    """The exact defect a per-process token would reintroduce."""

    def setUp(self):
        self._env_backup = os.environ.pop(ap.BOOT_TOKEN_ENV, None)
        ap._boot_token_cache = None

    def tearDown(self):
        if self._env_backup is not None:
            os.environ[ap.BOOT_TOKEN_ENV] = self._env_backup
        else:
            os.environ.pop(ap.BOOT_TOKEN_ENV, None)
        ap._boot_token_cache = None

    def test_reads_the_launcher_published_token_when_present(self):
        os.environ[ap.BOOT_TOKEN_ENV] = "weg2xsn31:1000:9999"
        self.assertEqual(ap.boot_token(), "weg2xsn31:1000:9999")

    def test_falls_back_to_a_derived_token_when_absent(self):
        os.environ.pop(ap.BOOT_TOKEN_ENV, None)
        tok = ap.boot_token()
        self.assertTrue(tok)
        self.assertEqual(tok.count(":"), 2)

    def test_mutant_a_per_process_token_would_separate_P_and_D(self, tmp_path=None):
        """MUTANT, reproduced by hand: if boot_token() derived its own value
        from THIS PROCESS's own start time/PID instead of reading
        BOOT_TOKEN_ENV, P and D -- two separate OS processes for the
        identical real boot -- would get two DIFFERENT tokens and their
        dumps would land in two different subdirectories, exactly
        reintroducing the "P and D share one directory" break this fix
        exists to prevent. Demonstrated directly: two per-process tokens
        (different pid) are unequal even when everything else matches.
        """
        mutant_p = f"notag:1000:{111}"
        mutant_d = f"notag:1000:{222}"
        self.assertNotEqual(
            mutant_p, mutant_d,
            "two per-process tokens for the SAME boot moment differ purely "
            "by pid -- this is the mutant's own failure mode",
        )
        # The real function, with the launcher's token published, gives P
        # and D (simulated as two independent calls after popping any
        # process-level cache) the IDENTICAL value instead.
        os.environ[ap.BOOT_TOKEN_ENV] = "weg2xsn31:1000:9999"
        ap._boot_token_cache = None
        token_as_seen_by_p = ap.boot_token()
        ap._boot_token_cache = None  # simulate a second, independent process
        token_as_seen_by_d = ap.boot_token()
        self.assertEqual(token_as_seen_by_p, token_as_seen_by_d)


class TwoBootsOfTheSameFormNeverCollideOnDisk(CustomTestCase):
    def test_two_writes_different_tokens_survive_both(self, ):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p1 = _write(0, "P", "xsn31:1:100", 900, d)
            p2 = _write(0, "P", "xsn31:2:200", 950, d)
            self.assertIsNotNone(p1)
            self.assertIsNotNone(p2)
            self.assertNotEqual(p1, p2)
            self.assertTrue(os.path.exists(p1))
            self.assertTrue(os.path.exists(p2))
            with open(p1) as f:
                d1 = json.load(f)
            with open(p2) as f:
                d2 = json.load(f)
            self.assertEqual(d1["activation_peak_bytes"], (16000 + 900) << 20)
            self.assertEqual(d2["activation_peak_bytes"], (16000 + 950) << 20)
            self.assertEqual(d1["boot_token"], "xsn31:1:100")
            self.assertEqual(d2["boot_token"], "xsn31:2:200")

    def test_pflicht_mutant_the_old_flat_writer_would_have_destroyed_boot_1(self):
        """MUTANT: reproduce the PRE-#1395 write (flat, no subdirectory) by
        hand -- exactly what activation_probe.write_footprint_dump did
        before this fix. The second boot's write clobbers the first boot's
        file outright, because #1292's own collision check compares PROFILE
        DIGEST only, and two boots of the SAME form share one digest by
        construction.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, ap.dump_filename(0, "P"))
            payload_boot_1 = {"boot": "1", "activation_peak_bytes": 900}
            with open(path, "w") as f:
                json.dump(payload_boot_1, f)
            # the mutant: unconditional os.replace, no boot axis at all
            payload_boot_2 = {"boot": "2", "activation_peak_bytes": 950}
            with open(path, "w") as f:
                json.dump(payload_boot_2, f)
            with open(path) as f:
                survivor = json.load(f)
            self.assertEqual(
                survivor["boot"], "2",
                "the mutant destroys boot 1's dump -- exactly the xsn31/8 "
                "loss this fix exists to prevent",
            )
        # The REAL function, same scenario, must not do this (proven above).


class TheDefenseInDepthRefusesARealCollision(CustomTestCase):
    def test_same_directory_name_different_token_is_refused_W109(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p1 = _write(0, "P", "a:1:1", 900, d)
            self.assertIsNotNone(p1)
            with open(p1) as f:
                original = json.load(f)
            # Tamper the on-disk boot_token so a DIFFERENT token sanitises to
            # the SAME directory name as this write's own token would not --
            # simulating the residual collision (two tokens -> one sanitised
            # name) the defense-in-depth exists for, since the normal path
            # (different tokens -> different _boot_subdir output) cannot be
            # driven to collide by construction.
            tampered = dict(original)
            tampered["boot_token"] = "a:1:1-DIFFERENT"
            with open(p1, "w") as f:
                json.dump(tampered, f)
            p2 = _write(0, "P", "a:1:1", 950, d)
            self.assertIsNone(p2, "must REFUSE, not overwrite a dump whose "
                                  "recorded boot_token disagrees")
            with open(p1) as f:
                still_there = json.load(f)
            self.assertEqual(still_there["boot_token"], "a:1:1-DIFFERENT",
                             "the refused write must leave the file untouched")


class TheReaderRefusesToSubstituteAForeignBoot(CustomTestCase):
    """THE COORDINATOR'S OWN DANGER DIRECTION, verbatim: "ein Leser, der beim
    Fehlen des angefragten Boots still den neuesten fremden Dump liefert,
    ist teurer als eine Absenz (genau die Form, die heute einen
    Kalibrierfall gekostet hat)."
    """

    def test_asking_for_one_boot_never_returns_another_boots_dump(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            _write(0, "P", "xsn31:5:500", 900, d)
            _write(0, "P", "xsn31:6:600", 950, d)
            m = load_script()
            got = m.load_dumps(str(d), boot_token="xsn31:5:500")
            self.assertEqual(len(got), 1)
            self.assertEqual(got[0]["activation_peak_bytes"], (16000 + 900) << 20)

    def test_asking_for_a_missing_boot_returns_EMPTY_not_a_substitute(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            _write(0, "P", "xsn31:7:700", 900, d)  # a DIFFERENT, real boot
            m = load_script()
            got = m.load_dumps(str(d), boot_token="xsn31:8:800")
            self.assertEqual(
                got, [],
                "boot xsn31:8 has no dump here -- the reader must report "
                "this as an absence, never hand back xsn31:7's dump",
            )

    def test_pflicht_mutant_a_reader_that_globs_the_newest_subdir_is_the_danger(
        self,
    ):
        """MUTANT, reproduced by hand: a reader that, on a miss, falls back
        to "the newest boot subdirectory found" instead of reporting an
        absence. This is EXACTLY what cost #1389 a calibration case today
        (per the coordinator's order) and is more expensive than an honest
        miss: it looks like data, grades against the wrong boot, and no
        counter or log line distinguishes it from a correct read.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            _write(0, "P", "xsn31:5:500", 900, d)
            _write(0, "P", "xsn31:6:600", 950, d)  # the "newest" wrong boot

            def mutant_load(dump_dir, boot_token):
                sub = os.path.join(dump_dir, ap._boot_subdir(boot_token))
                matches = sorted(glob.glob(
                    os.path.join(sub, "phase_footprint_*rank*.json")))
                if not matches:
                    # THE MUTANT: silently substitute the newest OTHER
                    # subdirectory instead of returning [].
                    subs = sorted(
                        p for p in glob.glob(os.path.join(dump_dir, "*"))
                        if os.path.isdir(p)
                    )
                    if subs:
                        matches = sorted(glob.glob(
                            os.path.join(subs[-1], "phase_footprint_*rank*.json")))
                out = []
                for p in matches:
                    with open(p) as f:
                        out.append(json.load(f))
                return out

            mutant_result = mutant_load(str(d), "xsn31:9:900")  # a THIRD, missing boot
            self.assertNotEqual(
                mutant_result, [],
                "the mutant DOES substitute a foreign boot's dump -- this "
                "is the exact defect class the real load_dumps refuses",
            )

            m = load_script()
            real_result = m.load_dumps(str(d), boot_token="xsn31:9:900")
            self.assertEqual(
                real_result, [],
                "the REAL load_dumps must return EMPTY for the same miss, "
                "never the mutant's substituted foreign dump",
            )
