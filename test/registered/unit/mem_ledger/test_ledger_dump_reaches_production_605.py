"""#605: the modeled ledger must be dumped on the path production actually takes.

FIRST REAL EXECUTION, 2026-08-05 21:11: production booted the recorder tree,
wrote all three ranks' marks correctly, and produced NO modeled ledger. The
dump had been placed in ``enforce_boot_contract``, which is reached only from
``_vram_ledger_non_kv_per_gpu`` -- inside ``if vram_ledger_enabled()``.
Production runs ``--rank-auto-reserve-mib auto`` with ``enable_vram_ledger``
False, so it takes ``ledger_full_demand_per_gpu`` instead and never calls the
contract at all. server_args.py says so in its own comment: "the gated ledger
path has always added it (_vram_ledger_non_kv_per_gpu), this one did not, and
production runs THIS one".

The lesson is not "that was the wrong caller" but "a caller was the wrong kind
of place". The dump now sits in ``build_card_ledgers``, the single function
that constructs a ledger, so no reserve path, flagset or future caller can
route around it.

A SECOND defect rode along and would have wasted the next boot too: the ledger
is built in the LAUNCHER during argument resolution, while the marks are
written by the RANKS. boot_id derived from the parent process gives the ranks
the launcher (right) and the launcher its shell (wrong), so the two halves
would have been filed under different ids and reconciliation would have matched
nothing even with the dump in the right place.
"""

import json
import os
import tempfile
import unittest

from sglang.srt.mem_ledger.engine import CardFacts, DemandInputs, build_card_ledgers
from sglang.srt.mem_ledger.flight_recorder import (
    BOOT_ID_ENV,
    DIR_ENV,
    dump_ledger,
    publish_boot_id,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _inputs(ranks=2):
    """Shaped like the reference rig's, so the constructor takes the same
    branches production takes. The numbers are fixture values, not rig facts."""
    return DemandInputs(
        weight_mib_per_rank=[0] * ranks,
        activation_mib_per_rank=[1766.0] * ranks,
        capture_tokens_per_rank=[96] * ranks,
        capture_mib_per_rank=[640.0] * ranks,
        phase_footprint_source_per_rank=["[upper_bound] fixture"] * ranks,
        phase_footprint_fingerprint="a191a0712717",
        mamba_pool_mib_per_rank=[512.0] * ranks,
        chunked_prefill_size=2048,
        max_running_requests=4,
    )


def _cards():
    return [
        CardFacts(
            gpu_id=0, uuid="GPU-a", name="RTX 3080", total_mib=20480, reserved_mib=425
        ),
        CardFacts(
            gpu_id=1, uuid="GPU-b", name="RTX 5090", total_mib=32607, reserved_mib=518
        ),
    ]


class TestDumpSitsAtTheConstructor(unittest.TestCase):
    """Execution, not inspection: call the real constructor and look for the
    file. A source-level assertion would have passed for the shipped defect
    too, because the call WAS present -- just somewhere production never went.
    """

    def setUp(self):
        self.env = dict(os.environ)
        self.dir = tempfile.mkdtemp()
        os.environ[DIR_ENV] = self.dir
        os.environ[BOOT_ID_ENV] = "bootTEST"
        import sglang.srt.mem_ledger.flight_recorder as fr

        fr._boot_id = "bootTEST"
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.env)))

    def test_building_a_ledger_writes_it_no_contract_involved(self):
        ledgers = build_card_ledgers(
            _inputs(),
            cards=_cards(),
            rank_gpu_id=[0, 1],
            user_reserve_mib={0: 1024, 1: 1024},
        )
        self.assertTrue(ledgers)
        path = os.path.join(self.dir, "ledger_bootTEST.json")
        self.assertTrue(os.path.exists(path), "no ledger written by the constructor")
        payload = json.load(open(path))
        self.assertEqual(payload["boot_id"], "bootTEST")
        self.assertEqual(len(payload["cards"]), 2)
        labels = [c["card"] for c in payload["cards"]]
        self.assertTrue(any("RTX 3080" in x for x in labels), labels)
        self.assertTrue(any("RTX 5090" in x for x in labels), labels)
        # The terms must survive the round trip; a dump of empty cards would
        # satisfy every assertion above and reconcile against nothing.
        self.assertTrue(all(c["terms"] for c in payload["cards"]), payload)

    def test_a_rebuild_overwrites_and_counts(self):
        """Argument resolution builds the ledger repeatedly while deriving the
        reserve; the last build is the one closest to what the boot runs."""
        for _ in range(3):
            build_card_ledgers(
                _inputs(),
                cards=_cards(),
                rank_gpu_id=[0, 1],
                user_reserve_mib={0: 1024, 1: 1024},
            )
        payload = json.load(open(os.path.join(self.dir, "ledger_bootTEST.json")))
        self.assertGreaterEqual(payload["build_index"], 3)

    def test_unarmed_writes_nothing(self):
        os.environ.pop(DIR_ENV)
        build_card_ledgers(
            _inputs(),
            cards=_cards(),
            rank_gpu_id=[0, 1],
            user_reserve_mib={0: 1024, 1: 1024},
        )
        self.assertEqual(os.listdir(self.dir), [])

    def test_an_unwritable_directory_never_fails_the_boot(self):
        os.environ[DIR_ENV] = "/proc/nonexistent/flight"
        ledgers = build_card_ledgers(
            _inputs(),
            cards=_cards(),
            rank_gpu_id=[0, 1],
            user_reserve_mib={0: 1024, 1: 1024},
        )
        self.assertTrue(ledgers, "a dump failure must not cost the caller its ledger")


class TestLauncherAndRanksAgree(unittest.TestCase):
    def setUp(self):
        self.env = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.env)))
        os.environ.pop(BOOT_ID_ENV, None)

    def test_the_launcher_publishes_and_a_rank_inherits_it(self):
        import sglang.srt.mem_ledger.flight_recorder as fr

        fr._boot_id = None
        launcher = publish_boot_id()
        self.assertEqual(os.environ[BOOT_ID_ENV], launcher)
        fr._boot_id = None  # a freshly spawned rank, inheriting the environment
        self.assertEqual(fr.boot_id(), launcher)

    def test_publishing_twice_does_not_rename_the_boot(self):
        import sglang.srt.mem_ledger.flight_recorder as fr

        fr._boot_id = None
        first = publish_boot_id()
        fr._boot_id = None
        self.assertEqual(publish_boot_id(), first)

    def test_the_ledger_and_the_marks_land_under_one_id(self):
        """The end-to-end property the reconciliation depends on."""
        import sglang.srt.mem_ledger.flight_recorder as fr

        with tempfile.TemporaryDirectory() as d:
            os.environ[DIR_ENV] = d
            fr._boot_id = None
            publish_boot_id()  # launcher
            build_card_ledgers(
                _inputs(),
                cards=_cards(),
                rank_gpu_id=[0, 1],
                user_reserve_mib={0: 1024, 1: 1024},
            )
            fr._boot_id = None  # rank process
            fr.mark("process_start", rank=0, directory=d)
            marks = fr.read_marks(d)
            ledgers = [f for f in os.listdir(d) if f.startswith("ledger_")]
            self.assertEqual(len(ledgers), 1)
            self.assertEqual(
                ledgers[0], f"ledger_{marks[os.getpid()][0]['boot_id']}.json"
            )


class TestNoFlagGate(unittest.TestCase):
    def test_the_dump_is_not_reachable_only_through_the_contract(self):
        """The shipped defect, pinned at the source: enforce_boot_contract is
        behind vram_ledger_enabled(), so nothing needed for Stage 1 may live
        there alone."""
        import sglang.srt.mem_ledger.contract as contract

        source = open(contract.__file__).read()
        self.assertNotIn("dump_ledger", source)
        self.assertNotIn("flight_recorder", source)

    def test_dump_ledger_is_callable_with_an_empty_ledger_list(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ[DIR_ENV] = d
            try:
                self.assertIsNotNone(dump_ledger([]))
            finally:
                os.environ.pop(DIR_ENV, None)


if __name__ == "__main__":
    unittest.main()
