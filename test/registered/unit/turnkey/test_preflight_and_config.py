# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Unit tests for turnkey config parsing, preflight and plan staleness.

Hermetic: no GPU, no driver, no socket, no systemd. Every probe is injected.

The FALSIFIER these tests exist for is
``test_every_failure_mode_is_reachable_and_named``: a preflight is only worth
having if each of its refusals can actually fire, and each fires under exactly
its own condition. A check that cannot be triggered in a test is a check that
will first run in production.

The second falsifier is ``test_cuda_visible_devices_is_never_an_index``: the
device-order trap (AUDIT_331) is that NVML index, CUDA ordinal and PCI order
are three different orderings of one set of cards. Emitting a bare integer
anywhere in the boot path reintroduces it.
"""

import os
import tempfile
import unittest

from sglang.srt.turnkey import config as C
from sglang.srt.turnkey import plan as PL
from sglang.srt.turnkey import preflight as PF
from sglang.srt.turnkey import refusal as RF
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

U1 = "GPU-11111111-1111-1111-1111-111111111111"
U2 = "GPU-22222222-2222-2222-2222-222222222222"

#: A REAL temporary checkout, because the repo-stability check deliberately
#: reads the filesystem rather than a probe: it is the one check whose whole
#: subject IS the on-disk shape of a git checkout, and faking that through an
#: injection seam would test the fake. A temp dir keeps it hermetic.
_TMP = tempfile.TemporaryDirectory()
REPO = os.path.join(_TMP.name, "repo")
os.makedirs(os.path.join(REPO, ".git"), exist_ok=True)


def tearDownModule():
    _TMP.cleanup()


BASE = """
[stack]
name = "t"
repo = "%s"
venv = "%s/.venv"
log_dir = "/var/log/t"

[env]
SGLANG_X = "1"

[[cards]]
uuid = "%s"
label = "a"
[[cards]]
uuid = "%s"
label = "b"

[wheel]
dist = "sglang-kernel"
version = "0.4.4"
must_import = ["sgl_kernel"]

[serving.ship]
port = 30030
argv = ["/bin/python", "-m", "sglang.launch_server"]
cards = [1, 0]
boot_log = "/var/log/t/ship.log"
""" % (REPO, REPO, U1, U2)


def cfg(extra=""):
    return C.loads(BASE + extra)


def probes(**over):
    base = dict(
        cards=lambda: [PF.CardObs(U1, "RTX 3080", 20 << 30, 20 << 30),
                       PF.CardObs(U2, "RTX 5090", 32 << 30, 32 << 30)],
        procs_on=lambda u: {},
        mem_available_bytes=lambda: 100 << 30,
        disk_free_bytes=lambda p: 100 << 30,
        port_busy=lambda p: False,
        path_exists=lambda p: True,
        probe_import=lambda m, a: PF.ImportObs("/repo/x.py", "0.4.4", True),
        # One provider = no #384 shadow. The clean machine has exactly one
        # distribution owning the import name; two is the refusal, even when
        # the import itself still answers correctly.
        dist_providers=lambda pkg: [PF.DistObs("sglang-kernel", "0.4.4", 74)],
    )
    base.update(over)
    return PF.Probes(**base)


class TestConfig(CustomTestCase):
    def test_parses(self):
        c = cfg()
        self.assertEqual(c.name, "t")
        self.assertEqual([x.name for x in c.serving], ["ship"])

    def test_cuda_visible_devices_is_never_an_index(self):
        """THE device-order falsifier. Cards are addressed by UUID, and the
        string handed to CUDA is UUIDs in RANK order -- not NVML order, not
        CUDA ordinal order, and never a bare integer."""
        c = cfg()
        cvd = c.visible_devices(c.lane("ship"))
        # cards = [1, 0] -> the SECOND configured card first.
        self.assertEqual(cvd, f"{U2},{U1}")
        for token in cvd.split(","):
            self.assertTrue(token.startswith("GPU-"), token)
            self.assertFalse(token.isdigit())

    def test_index_as_card_uuid_is_refused(self):
        with self.assertRaises(RF.RefusalError) as e:
            C.loads(BASE.replace(f'uuid = "{U1}"', 'uuid = "0"'))
        self.assertEqual(e.exception.refusal.name, RF.REFUSE_CONFIG_INCOMPLETE)

    def test_shared_boot_log_between_lanes_is_refused(self):
        """#375 defect 3: interleaved logs prove nothing about either lane."""
        with self.assertRaises(RF.RefusalError) as e:
            cfg("""
[serving.second]
port = 30031
argv = ["/bin/python"]
cards = [0]
boot_log = "/var/log/t/ship.log"
""")
        self.assertEqual(e.exception.refusal.name, RF.REFUSE_LOG_PATH_SHARED)

    def test_duplicate_port_is_refused(self):
        with self.assertRaises(RF.RefusalError):
            cfg("""
[serving.second]
port = 30030
argv = ["/bin/python"]
cards = [0]
boot_log = "/var/log/t/second.log"
""")

    def test_protected_port_cannot_be_required_free(self):
        """30099 is the local router; expecting it free would refuse forever."""
        with self.assertRaises(RF.RefusalError) as e:
            cfg("\n[preflight]\ncheck_ports = [30099]\n")
        self.assertEqual(e.exception.refusal.name, RF.REFUSE_CONFIG_INCOMPLETE)

    def test_card_index_out_of_range_refuses_at_parse_time(self):
        with self.assertRaises(RF.RefusalError):
            C.loads(BASE.replace("cards = [1, 0]", "cards = [1, 7]"))

    def test_unparsable_toml(self):
        with self.assertRaises(RF.RefusalError) as e:
            C.loads("not = = toml")
        self.assertEqual(e.exception.refusal.name, RF.REFUSE_CONFIG_UNPARSABLE)

    def test_pythonpath_defaults_to_the_configured_repo(self):
        """The worktree-PYTHONPATH trap: a lane launched against the wrong
        tree silently runs another build."""
        c = cfg()
        env = c.env_for(c.lane("ship"))
        self.assertEqual(env["PYTHONPATH"], os.path.join(REPO, "python"))


class TestRepoStability(CustomTestCase):
    def test_worktree_refused_unless_acknowledged(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            real = os.path.join(d, "real.git")
            os.makedirs(real)
            wt = os.path.join(d, "wt")
            os.makedirs(wt)
            with open(os.path.join(wt, ".git"), "w") as fh:
                fh.write(f"gitdir: {real}\n")
            with self.assertRaises(RF.RefusalError) as e:
                C.assert_repo_stable(wt, allow_worktree=False)
            self.assertEqual(e.exception.refusal.name,
                             RF.REFUSE_REPO_IS_WORKTREE)
            # Acknowledged: accepted.
            C.assert_repo_stable(wt, allow_worktree=True)

    def test_broken_worktree_pointer_refused_even_when_acknowledged(self):
        """An acknowledgement cannot repair a checkout that is already gone."""
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            wt = os.path.join(d, "wt")
            os.makedirs(wt)
            with open(os.path.join(wt, ".git"), "w") as fh:
                fh.write(f"gitdir: {d}/vanished\n")
            with self.assertRaises(RF.RefusalError):
                C.assert_repo_stable(wt, allow_worktree=True)


class TestPreflight(CustomTestCase):
    def test_clean_machine_has_no_refusals(self):
        self.assertEqual(PF.run_all(cfg(), probes()), [])

    def test_every_failure_mode_is_reachable_and_named(self):
        """THE falsifier: each check fires, under its own condition only."""
        cases = [
            (RF.REFUSE_WHEEL_SHADOW,
             dict(probe_import=lambda m, a: PF.ImportObs("/x", "0.4.4", False))),
            (RF.REFUSE_WHEEL_SHADOW,
             dict(probe_import=lambda m, a: PF.ImportObs("/x", "0.3.21", True))),
            # The #384 standing-reinstall block. Note what the probes say:
            # the import still resolves to the fork version WITH the arm, so
            # every other wheel check passes. The installation is one pip
            # invocation from silently losing it, and that is the refusal.
            (RF.REFUSE_WHEEL_DIST_SHADOW,
             dict(dist_providers=lambda pkg: [
                 PF.DistObs("sglang-kernel", "0.4.4", 74),
                 PF.DistObs("sgl-kernel", "0.3.21", 69)])),
            (RF.REFUSE_CARD_UNKNOWN_UUID,
             dict(cards=lambda: [PF.CardObs(U1, "RTX 3080", 1, 1)])),
            (RF.REFUSE_CARD_CENSUS, dict(cards=lambda: (_ for _ in ()).throw(
                RuntimeError("nvml gone")))),
            (RF.REFUSE_CARD_BUSY,
             dict(cards=lambda: [PF.CardObs(U1, "RTX 3080", 20 << 30, 20 << 30),
                                 PF.CardObs(U2, "RTX 5090", 32 << 30, 1 << 30)],
                  procs_on=lambda u: {4242: 30 << 30})),
            (RF.REFUSE_HOST_HEADROOM, dict(mem_available_bytes=lambda: 1 << 30)),
            (RF.REFUSE_PORT_BUSY, dict(port_busy=lambda p: True)),
            (RF.REFUSE_PATH_MISSING, dict(path_exists=lambda p: False)),
        ]
        c = cfg("\n[preflight]\nhost_headroom_gib = 15\n")
        for want, over in cases:
            names = [r.name for r in PF.run_all(c, probes(**over))]
            self.assertIn(want, names, f"{want} unreachable with {over.keys()}")

    def test_disk_headroom_refusal(self):
        c = cfg('\n[preflight.disk_free_gib]\n"/spinning" = 500\n')
        names = [r.name for r in PF.run_all(c, probes())]
        self.assertIn(RF.REFUSE_DISK_HEADROOM, names)

    def test_busy_card_refusal_names_the_pid_not_a_pattern(self):
        """Orphan cleanup is BY PID. `pkill -f sglang` also matches the
        router on :30099, whose liveness is a standing law here."""
        c = cfg()
        over = dict(
            cards=lambda: [PF.CardObs(U1, "RTX 3080", 20 << 30, 20 << 30),
                           PF.CardObs(U2, "RTX 5090", 32 << 30, 1 << 30)],
            procs_on=lambda u: {4242: 30 << 30})
        r = [x for x in PF.run_all(c, probes(**over))
             if x.name == RF.REFUSE_CARD_BUSY][0]
        self.assertIn("4242", r.observed)
        self.assertIn("never pkill", r.remedy)

    def test_protected_port_is_not_probed(self):
        asked = []
        c = cfg()
        PF.run_all(c, probes(port_busy=lambda p: (asked.append(p), False)[1]))
        self.assertNotIn(30099, asked)

    def test_all_refusals_reported_not_just_the_first(self):
        c = cfg()
        rs = PF.run_all(c, probes(port_busy=lambda p: True,
                                  mem_available_bytes=lambda: 1))
        self.assertGreaterEqual(len(rs), 2)

    def test_expect_name_mismatch_catches_a_swapped_card(self):
        c = C.loads(BASE.replace('label = "b"',
                                 'label = "b"\nexpect_name = "5090"'))
        over = dict(cards=lambda: [PF.CardObs(U1, "RTX 3080", 1 << 30, 1 << 30),
                                   PF.CardObs(U2, "RTX 4090", 1 << 30, 1 << 30)])
        names = [r.name for r in PF.run_all(c, probes(**over))]
        self.assertIn(RF.REFUSE_CARD_CENSUS, names)


class TestRefusalVocabulary(CustomTestCase):
    def test_unregistered_name_is_itself_an_error(self):
        with self.assertRaises(ValueError):
            RF.Refusal(name="REFUSE_MADE_UP", subject="x", observed="y",
                       expected="z")

    def test_line_is_greppable_and_carries_the_numbers(self):
        r = RF.refuse(RF.REFUSE_HOST_HEADROOM, "MemAvailable", "2 GiB",
                      ">= 15 GiB", remedy="free some")
        line = r.line()
        self.assertTrue(line.startswith("REFUSE_HOST_HEADROOM "))
        for part in ("subject=", "observed=", "expected=", "remedy="):
            self.assertIn(part, line)


class TestPlanStaleness(CustomTestCase):
    def _fp(self, **kw):
        base = dict(cards=[(U1, 20480), (U2, 32607)],
                    argv=["/bin/python", "-m", "x"],
                    model_path="", wheel_version="0.4.4")
        base.update(kw)
        return PL.fingerprint_of(base["cards"], base["argv"],
                                 model_path=base["model_path"],
                                 wheel_version=base["wheel_version"])

    def test_matching_fingerprint_is_not_stale(self):
        fp = self._fp()
        pin = PL.PinnedPlan(fingerprint=fp)
        self.assertIsNone(PL.check_staleness(pin, fp))

    def test_a_swapped_card_refuses(self):
        pin = PL.PinnedPlan(fingerprint=self._fp())
        now = self._fp(cards=[(U1, 20480), ("GPU-3333", 24576)])
        r = PL.check_staleness(pin, now)
        self.assertIsNotNone(r)
        self.assertEqual(r.name, RF.REFUSE_PLAN_STALE)
        self.assertIn("cards", r.observed)

    def test_changed_argv_refuses(self):
        """A plan solved for one context length says nothing about another."""
        pin = PL.PinnedPlan(fingerprint=self._fp())
        now = self._fp(argv=["/bin/python", "-m", "x", "--context-length", "9"])
        r = PL.check_staleness(pin, now)
        self.assertIsNotNone(r)
        self.assertIn("argv_digest", r.observed)

    def test_changed_wheel_refuses(self):
        pin = PL.PinnedPlan(fingerprint=self._fp())
        r = PL.check_staleness(pin, self._fp(wheel_version="0.3.21"))
        self.assertIsNotNone(r)
        self.assertIn("wheel_version", r.observed)

    def test_stale_plan_never_silently_resolves(self):
        """The point of the pin: refuse rather than adapt."""
        pin = PL.PinnedPlan(fingerprint=self._fp(),
                            launch_flags=("--max-total-tokens", "620000"))
        r = PL.check_staleness(pin, self._fp(cards=[(U1, 20480)]))
        self.assertIsNotNone(r)
        self.assertIn("re-pin", r.remedy)

    def test_age_check_only_when_configured(self):
        import time
        old = PL.PinnedPlan(fingerprint=self._fp(),
                            solved_at=time.time() - 40 * 86400)
        self.assertIsNone(PL.check_staleness(old, self._fp(), 0))
        self.assertIsNotNone(PL.check_staleness(old, self._fp(), 30))

    def test_missing_plan_file_refuses_by_name(self):
        p, r = PL.load_pinned("/nonexistent/plan.json")
        self.assertIsNone(p)
        self.assertEqual(r.name, RF.REFUSE_PLAN_MISSING)

    def test_roundtrip(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.json")
            pin = PL.PinnedPlan(fingerprint=self._fp(),
                                launch_flags=("--a", "1"), solver="test")
            PL.save_pinned(path, pin)
            got, r = PL.load_pinned(path)
            self.assertIsNone(r)
            self.assertEqual(got.fingerprint, pin.fingerprint)
            self.assertEqual(got.launch_flags, ("--a", "1"))


class TestCardBusyIsForeignUsageNotCarveOut(CustomTestCase):
    """#656: preflight refused an IDLE machine.

    VAL-R4 hit ``REFUSE_CARD_BUSY subject=RTX 5090 observed=521 MiB in use
    (no compute pids) expected=<= 512 MiB`` on a rig with nothing running.
    521 MiB is the 5090's NVML driver carve-out; the shipped
    ``card_busy_mib = 512`` sat just under it. The remedy -- "stop the named
    pids BY PID" -- was unactionable, because the message itself said there
    were none.

    The carve-out is not foreign occupancy. It is measured, named and
    budgeted everywhere else in this tree (``registry/nvml.py`` reports it as
    ``reserved_bytes``; ``mem_ledger`` books it as
    ``TERM_NVML_CARVE_OUT``; ``uneven_perf.py`` records the measured
    magnitudes, 425 MiB on a 3080 and 518 on a 5090). Only preflight folded
    it into ``used`` and compared the sum against a flat constant -- which is
    why the threshold had to be hand-raised to 600 to let the machine boot.

    The fix derives the quantity under test from the measured carve-out
    instead of guessing a constant above it: what must stay small is
    ``total - free - reserved``, the FOREIGN part.
    """

    #: The measured values from the VAL-R4 refusal, in bytes.
    _5090_TOTAL = 32 << 30
    _5090_CARVE = 518 * (1 << 20)
    _5090_FREE_IDLE = _5090_TOTAL - 521 * (1 << 20)

    def _idle_cards(self):
        """Both cards idle: nothing but each driver's own carve-out."""
        return [
            PF.CardObs(U1, "RTX 3080", 20 << 30,
                       (20 << 30) - 425 * (1 << 20),
                       reserved_bytes=425 * (1 << 20)),
            PF.CardObs(U2, "RTX 5090", self._5090_TOTAL,
                       self._5090_FREE_IDLE,
                       reserved_bytes=self._5090_CARVE),
        ]

    def test_an_idle_machine_is_not_a_busy_card(self):
        """THE fix test. Fails before the fix with REFUSE_CARD_BUSY on the
        5090, at the shipped threshold of 512 -- exactly VAL-R4's refusal."""
        c = cfg()  # card_busy_mib defaults to 512
        names = [r.name for r in
                 PF.run_all(c, probes(cards=self._idle_cards,
                                      procs_on=lambda u: {}))]
        # The whole list, not just the absence of CARD_BUSY: asserting only
        # the absence would pass for the WRONG reason before the fix, because
        # the unknown ``reserved_bytes`` argument raises inside the card
        # probe and the census guard converts it into REFUSE_CARD_CENSUS --
        # at which point "not busy" is trivially true. An idle machine must
        # produce no refusal at all.
        self.assertEqual(names, [])

    def test_the_threshold_measures_foreign_bytes_alone(self):
        """#751: THE BOUNDARY CELL where the old and new computation
        disagree, pinned in both directions with the shipped threshold 512.

        foreign=500 with carve-out 518: total-minus-free reads 1018 MiB
        (the old computation refuses), but only 500 MiB belong to a tenant
        -- under the threshold, must PASS. foreign=600, same carve-out:
        over the threshold by its own weight, must REFUSE. Together these
        pin that the allowance prices GENUINE foreign bytes and never the
        hardware constant beneath them."""
        c = cfg()  # card_busy_mib defaults to 512
        for foreign_mib, want_busy in ((500, False), (600, True)):
            with self.subTest(foreign_mib=foreign_mib):
                cards = self._idle_cards()
                free = (
                    self._5090_TOTAL
                    - self._5090_CARVE
                    - foreign_mib * (1 << 20)
                )
                cards[1] = PF.CardObs(
                    U2,
                    "RTX 5090",
                    self._5090_TOTAL,
                    free,
                    reserved_bytes=self._5090_CARVE,
                )
                names = [
                    r.name
                    for r in PF.run_all(
                        c,
                        probes(
                            cards=lambda: cards,
                            procs_on=lambda u: {777: foreign_mib << 20},
                        ),
                    )
                ]
                if want_busy:
                    self.assertIn(RF.REFUSE_CARD_BUSY, names)
                else:
                    self.assertNotIn(RF.REFUSE_CARD_BUSY, names)

    def test_a_genuinely_occupied_card_still_refuses(self):
        """The check must not be defanged: a real tenant is still foreign
        occupancy even after its card's carve-out is discounted."""
        c = cfg()
        cards = self._idle_cards()
        cards[1] = PF.CardObs(U2, "RTX 5090", self._5090_TOTAL,
                              1 << 30, reserved_bytes=self._5090_CARVE)
        r = [x for x in PF.run_all(c, probes(cards=lambda: cards,
                                             procs_on=lambda u: {4242: 30 << 30}))
             if x.name == RF.REFUSE_CARD_BUSY]
        self.assertTrue(r, "a 30 GiB tenant must still refuse")
        self.assertIn("4242", r[0].observed)

    def test_the_refusal_separates_foreign_bytes_from_the_carve_out(self):
        """A reader must be able to tell the two apart in the message. The
        old text reported one number that silently contained both."""
        c = cfg()
        cards = self._idle_cards()
        cards[1] = PF.CardObs(U2, "RTX 5090", self._5090_TOTAL,
                              1 << 30, reserved_bytes=self._5090_CARVE)
        r = [x for x in PF.run_all(c, probes(cards=lambda: cards,
                                             procs_on=lambda u: {4242: 30 << 30}))
             if x.name == RF.REFUSE_CARD_BUSY][0]
        self.assertIn("carve-out", r.observed)
        self.assertIn("518", r.observed)

    def test_a_card_reporting_no_carve_out_behaves_as_before(self):
        """``reserved_bytes`` defaults to 0, and NVML's own fallback path
        returns 0 when it cannot read the v2 struct. An unknown carve-out
        must not silently widen the threshold -- it degrades to the old,
        conservative answer."""
        c = cfg()
        cards = [PF.CardObs(U1, "RTX 3080", 20 << 30, 20 << 30),
                 PF.CardObs(U2, "RTX 5090", 32 << 30, (32 << 30) - (1 << 30))]
        names = [r.name for r in PF.run_all(c, probes(cards=lambda: cards))]
        self.assertIn(RF.REFUSE_CARD_BUSY, names)


if __name__ == "__main__":
    unittest.main()
