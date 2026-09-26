"""RC7: a missing ``memory.current`` is a NAMED ledger refusal, never a formatted ``None``.

Docker host acceptance, 2026-09-25 07:21Z: the container ran with ``-v /sys:/sys`` and read the HOST's root
cgroup -- ``memory.stat`` present, ``memory.current`` absent. ``read_cgroup_pressure`` took its sum FALLBACK
(``nonreclaim_gib`` set, ``current_gib`` None) and ``watermark_provenance`` died on ``f"{None:.2f}"`` with a
TypeError inside the launcher's host ledger. The contract now, both directions:

* memory.current MISSING: the ledger (``choose`` -> ``watermark_provenance(..., refuse_unreadable=True)``)
  raises W20 Weg2HostLedgerRefused naming the file; every other caller (the front's periodic line) prints
  that it is unreadable -- no number is formatted in either case;
* memory.current PRESENT: the line is byte-for-byte the old one, and the ledger does not refuse.
"""

from __future__ import annotations

import inspect
import os
import tempfile
import unittest
from unittest import mock

from sglang.srt.weg2 import host_ledger as hl
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

GIB = 1024 ** 3


def _host_root_like_cgroup() -> str:
    """The container's view of the HOST's root cgroup: memory.stat, no memory.current."""
    root = tempfile.mkdtemp(prefix="rc7_cg_")
    with open(os.path.join(root, "memory.stat"), "w") as f:
        f.write(f"anon {20 * GIB}\nshmem {5 * GIB}\nslab_unreclaimable {GIB}\nunevictable 0\nfile {9 * GIB}\n")
    return root


class MissingMemoryCurrent(CustomTestCase):
    def setUp(self):
        super().setUp()
        root = _host_root_like_cgroup()
        self.live = hl.read_cgroup_pressure(root)
        # the exact precondition of the 07:21Z TypeError
        self.assertIsNone(self.live["current_gib"])
        self.assertIsNotNone(self.live["nonreclaim_gib"])

    def test_the_ledger_refuses_by_name(self):
        with mock.patch.object(hl, "read_cgroup_pressure", lambda *a, **k: dict(self.live)):
            with self.assertRaises(hl.Weg2HostLedgerRefused) as cm:
                hl.watermark_provenance(refuse_unreadable=True)
        msg = str(cm.exception)
        self.assertIn("W20 Weg2HostLedgerRefused", msg)
        self.assertIn("memory.current is unreadable", msg)

    def test_the_front_line_formats_nothing(self):
        with mock.patch.object(hl, "read_cgroup_pressure", lambda *a, **k: dict(self.live)):
            line = hl.watermark_provenance()
        self.assertIn("LIVE memory.current unreadable", line)
        self.assertNotIn("raw_current=", line)
        self.assertNotIn("nonreclaim=", line)

    def test_the_ledger_asks_for_the_refusal(self):
        src = inspect.getsource(hl.choose)
        self.assertIn("watermark_provenance(margin, watermark_gib, refuse_unreadable=True)", src)


class PresentMemoryCurrent(CustomTestCase):
    LIVE = {
        "current_gib": 50.0, "nonreclaim_gib": 30.0, "file_reclaimable_gib": 20.0,
        "anon_gib": 25.0, "shmem_gib": 4.0, "source": "memory.current - inactive_file - active_file",
    }

    def test_the_line_is_unchanged_and_the_ledger_does_not_refuse(self):
        with mock.patch.object(hl, "read_cgroup_pressure", lambda *a, **k: dict(self.LIVE)):
            line = hl.watermark_provenance(refuse_unreadable=True)
            front_line = hl.watermark_provenance()
        want = (" LIVE nonreclaim=30.00 raw_current=50.00 file_reclaimable=20.00 GiB "
                "[memory.current - inactive_file - active_file]")
        self.assertTrue(line.endswith(want), line[-160:])
        self.assertEqual(line, front_line)

    def test_memory_stat_unreadable_keeps_its_old_line(self):
        live = {"current_gib": 50.0, "nonreclaim_gib": None, "file_reclaimable_gib": None,
                "anon_gib": None, "shmem_gib": None,
                "source": "memory.stat unreadable -- no non-reclaimable reading"}
        with mock.patch.object(hl, "read_cgroup_pressure", lambda *a, **k: dict(live)):
            line = hl.watermark_provenance(refuse_unreadable=True)
        self.assertTrue(line.endswith(" LIVE unreadable [memory.stat unreadable -- no non-reclaimable reading]"),
                        line[-120:])


if __name__ == "__main__":
    unittest.main()
