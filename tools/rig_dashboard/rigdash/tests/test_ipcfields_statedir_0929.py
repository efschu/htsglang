"""The rank files of a boot with a state dir are read from ``<state dir>/rankstate/<G>/``.

Host 29.09. ~20:00Z: /spinning/docker-acceptance/{27b,nf}/state/current/rankstate/{P,D}/ held the
RankState files of both lines (host_acceptance_v2.sh mounts the state dir, the launcher names
SGLANG_WEG2_RANK_STATE_DIR=/var/lib/htsglang/state/rankstate/<G>), while live.py looked only for
``<log>.rankstate`` next to the group logs -- so KV/seats stayed "aus Log (Übergang)".
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import ipcfields  # noqa: E402


def _write(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh)


class StateDirRankFilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sdir = os.path.join(self.tmp.name, "27bbf-boot-x")
        for g, ranks in (("D", ("D.tp0pp0", "D.tp1pp0", "D.tp2pp0")), ("P", ("P.tp0pp0", "P.tp0pp1"))):
            for r in ranks:
                _write(os.path.join(self.sdir, "rankstate", g, r + ".json"),
                       {"schema": 2, "kv": {"holds_kv": True, "kv_tokens": 1000}, "seats": 4})
        _write(os.path.join(self.sdir, "rankstate", "D", "D.tp0pp0.rankstats"),
               {"schema": ipcfields.RANKSTATS_SCHEMA, "tokens": {"prefill_total": 7}})

    def test_state_dir_rank_files_are_read(self):
        dirs = ipcfields.rank_dirs({}, {"dir": self.sdir})
        self.assertEqual([os.path.basename(d) for d in dirs], ["D", "P"])
        r = ipcfields.read_rank_files(dirs)
        self.assertEqual(sorted(r["rankstate"]), ["D.tp0pp0", "D.tp1pp0", "D.tp2pp0", "P.tp0pp0", "P.tp0pp1"])
        self.assertEqual(sorted(r["rankstats"]), ["D.tp0pp0"])

    def test_log_side_dirs_stay_and_are_not_doubled(self):
        log = os.path.join(self.tmp.name, "boot.D.log")
        os.makedirs(log + ".rankstate")
        dirs = ipcfields.rank_dirs({"D": {"path": log}}, {"dir": self.sdir})
        self.assertEqual(dirs[0], log + ".rankstate")
        self.assertEqual(len(dirs), 3)
        self.assertEqual(ipcfields.rank_dirs({"D": {"path": log}}, None), [log + ".rankstate"])

    def test_no_ipc_or_no_state_dir(self):
        self.assertEqual(ipcfields.rank_dirs({}, None), [])
        self.assertEqual(ipcfields.rank_dirs({}, {"dir": os.path.join(self.tmp.name, "nope")}), [])
        self.assertEqual(ipcfields.rank_dirs({}, {}), [])

    def test_live_reads_through_rank_dirs(self):
        src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "live.py")).read()
        self.assertIn('ipcfields.rank_dirs(v.get("files"), v["ipc"])', src)


if __name__ == "__main__":
    unittest.main()
