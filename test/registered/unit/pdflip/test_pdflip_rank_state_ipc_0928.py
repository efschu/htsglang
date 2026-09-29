"""IPC Phase 1 (user 28.09. ~20:45Z: "über logfiles?"): W7/W10 on RankState.

The launcher half of W7/W10 counted '#706 canonical KV page active' /
'canonical GDN blob active' / the Form A worker line in a group log. rc12z29d
(28.09. 20:29Z) booted D clean under the uneven-DCP cut and was refused as
'kv x1 blob x1': the two workers own token rows (#239 F14) and print
'#239 F14 KV-WORKER-WINDOW' instead. These tests pin the replacement:

  * every rank writes a versioned RankState record (pdflip/rank_state.py),
    a rank owning zero token rows included (27B requirement 3);
  * the launcher grades the records, not the log; the log count still runs
    and a disagreement is reported ('IPC MISMATCH'), never decided on;
  * a missing rank, a stale record of an earlier launch and a record of
    another schema are named refusals.

No GPU, no launch.
"""

import json
import os
import tempfile
import unittest

from flliper.srt.pdflip import rank_state as rs

#: rc12z29d, D log: the three lines as the three D ranks printed them under
#: the cut (TP0 host page + blob, TP1/TP2 F14 workers). Shapes only.
_RC12Z29D_D_LOG = (
    "[TP0] #706 canonical KV page active: slots [0, 16) of 16, 4096 B per slot; ...\n"
    "[TP0] #706 canonical GDN blob active: layers [0, 48) of 48, 1 of 1 blob bytes on this rank, 1 extent(s).\n"
    "[TP1] #239 F14 KV-WORKER-WINDOW: Form A worker owns token rows (64, 40, 52) of every 64-token page; KV page window only (no mamba/QSA/draft).\n"
    "[TP2] #239 F14 KV-WORKER-WINDOW: Form A worker owns token rows (64, 52, 64) of every 64-token page; KV page window only (no mamba/QSA/draft).\n"
)


def _d_rank(tp, worker, kv_built, rows=None):
    return rs.build_rank_state(
        group="D", tp_rank=tp, tp_size=3, pp_rank=0, pp_size=1,
        form_a_worker=worker, canonical_on=True, canonical_kv_built=kv_built,
        canonical_blob_built=not worker, has_mamba_pool=True, page_size=64,
        owner_ctx=rows, seq=1,
    )


def _p_rank(pp, blob=True):
    return rs.build_rank_state(
        group="P", tp_rank=0, tp_size=1, pp_rank=pp, pp_size=3,
        form_a_worker=False, canonical_on=True, canonical_kv_built=True,
        canonical_blob_built=blob, has_mamba_pool=True, page_size=64, owner_ctx=None, seq=1,
    )


class _Log:
    def __init__(self):
        self.lines = []

    def __call__(self, s):
        self.lines.append(s)


class TestRankStateRecord(unittest.TestCase):
    def test_roundtrip_and_schema(self):
        s = _d_rank(1, True, True, rows=(64, 40, 52))
        self.assertEqual(rs.RankState.from_json(s.to_json()), s)
        d = json.loads(s.to_json())
        d["schema"] = rs.RANK_STATE_SCHEMA + 1
        with self.assertRaises(rs.RankStateSchemaError):
            rs.RankState.from_json(json.dumps(d))
        d["schema"] = rs.RANK_STATE_SCHEMA
        d["surprise"] = 1
        with self.assertRaises(rs.RankStateSchemaError):
            rs.RankState.from_json(json.dumps(d))

    def test_mapping_roles_and_rows(self):
        w = _d_rank(1, True, True, rows=(64, 40, 52))
        self.assertEqual((w.role, w.kv_page_applicable, w.kv_page_active, w.gdn_blob_applicable, w.kv_rows_per_page),
                         (rs.ROLE_FORM_A_WORKER, True, True, False, 12))
        # 27B requirement 3: a worker owning NO rows under the cut still
        # reports, with rows 0 -- not by silence.
        z = _d_rank(2, True, True, rows=(64, 64, 64))
        self.assertEqual((z.kv_rows_per_page, z.kv_page_applicable), (0, False))
        n = _d_rank(1, True, False)  # worker without a cut: null storage tier
        self.assertEqual((n.kv_page_applicable, n.kv_page_active, n.kv_rows_per_page), (False, False, 64))
        h = _d_rank(0, False, True, rows=(64, 0, 40))
        self.assertEqual((h.role, h.kv_page_applicable, h.kv_page_active, h.gdn_blob_applicable, h.gdn_blob_active),
                         (rs.ROLE_ATTN_HOST, True, True, True, True))
        # A dense rank (no mamba pool, the W7 witness) owes no blob.
        dense = rs.build_rank_state(
            group="P", tp_rank=0, tp_size=1, pp_rank=0, pp_size=3, form_a_worker=False,
            canonical_on=True, canonical_kv_built=True, canonical_blob_built=False,
            has_mamba_pool=False, page_size=64, owner_ctx=None, seq=1)
        self.assertEqual((dense.gdn_blob_applicable, dense.gdn_blob_active), (False, False))
        self.assertEqual(_p_rank(1).role, rs.ROLE_STAGE)

    def test_write_is_atomic_and_clear_removes(self):
        with tempfile.TemporaryDirectory() as d:
            sd = os.path.join(d, "g.log.rankstate")
            p = rs.write_rank_state(_p_rank(0), sd)
            self.assertTrue(p.endswith("P.tp0pp0.json"))
            self.assertEqual([n for n in os.listdir(sd) if ".tmp." in n], [])
            states, bad = rs.read_group_states(sd)
            self.assertEqual((len(states), bad), (1, []))
            self.assertEqual(rs.clear_rank_state_dir(sd), 1)
            self.assertEqual(rs.read_group_states(sd), ([], []))
        self.assertIsNone(rs.write_rank_state(_p_rank(0), None))


class TestGradeCanonical(unittest.TestCase):
    def test_rc12z29d_cut_form_passes(self):
        states = [_d_rank(0, False, True, rows=(64, 0, 40)),
                  _d_rank(1, True, True, rows=(64, 40, 52)),
                  _d_rank(2, True, True, rows=(64, 52, 64))]
        v = rs.grade_canonical(states, 3, group="D")
        self.assertTrue(v.ok, v.line())
        self.assertEqual((v.n_kv, v.n_blob, v.n_worker), (3, 3, 2))

    def test_worker_without_cut_passes(self):
        states = [_d_rank(0, False, True), _d_rank(1, True, False), _d_rank(2, True, False)]
        self.assertTrue(rs.grade_canonical(states, 3, group="D").ok)

    def test_missing_rank_is_named(self):
        v = rs.grade_canonical([_p_rank(0), _p_rank(2)], 3, group="P")
        self.assertFalse(v.ok)
        self.assertEqual(v.missing, ["tp0pp1"])
        self.assertIn("MISSING tp0pp1", v.line())

    def test_no_records_refuses(self):
        v = rs.grade_canonical([], 3, group="P")
        self.assertFalse(v.ok)
        self.assertTrue(v.missing)

    def test_host_without_blob_refuses(self):
        v = rs.grade_canonical([_p_rank(0), _p_rank(1, blob=False), _p_rank(2)], 3, group="P")
        self.assertFalse(v.ok)
        self.assertEqual(v.n_blob, 2)
        self.assertTrue(any("tp0pp1" in r and "GDN blob applicable, not active" in r for r in v.reasons))

    def test_format_not_armed_refuses(self):
        off = [rs.build_rank_state(
            group="P", tp_rank=0, tp_size=1, pp_rank=p, pp_size=3, form_a_worker=False,
            canonical_on=False, canonical_kv_built=False, canonical_blob_built=False,
            has_mamba_pool=True, page_size=64, owner_ctx=None, seq=1) for p in range(3)]
        v = rs.grade_canonical(off, 3, group="P")
        self.assertFalse(v.ok)
        self.assertEqual((v.n_kv, v.n_blob), (0, 0))

    def test_wrong_group_and_size_refuse(self):
        v = rs.grade_canonical([_p_rank(0), _p_rank(1), _p_rank(2)], 3, group="D")
        self.assertFalse(v.ok)
        states = [_d_rank(0, False, True), _d_rank(1, True, False)]
        v = rs.grade_canonical(states, 2, group="D")
        self.assertFalse(v.ok)
        self.assertTrue(any("launcher expects 2" in r for r in v.reasons))


class TestLauncherGate(unittest.TestCase):
    """The launcher half decides on the records; the log count only reports."""

    def setUp(self):
        from flliper.srt.pdflip import launcher

        self.launcher = launcher
        self.tmp = tempfile.TemporaryDirectory()
        self.log_path = os.path.join(self.tmp.name, "boot.D.log")

    def tearDown(self):
        self.tmp.cleanup()

    def _spec(self, name="D"):
        return self.launcher.GroupSpec(name, 0, ["true"], self.log_path, {})

    def _write(self, states):
        sd = rs.rank_state_dir_for_log(self.log_path)
        for s in states:
            rs.write_rank_state(s, sd)

    def test_rc12z29d_d_group_is_not_refused(self):
        with open(self.log_path, "w") as f:
            f.write(_RC12Z29D_D_LOG)
        # rc12z30f: with a33ee80394 the log count knows the F14 line too
        # (1+2, 1+2); the records of the same three ranks pass and decide.
        n_kv, n_blob, _ = self.launcher.canonical_marker_counts(self.log_path)
        self.assertEqual((n_kv, n_blob), (3, 3))
        self._write([_d_rank(0, False, True, rows=(64, 0, 40)),
                     _d_rank(1, True, True, rows=(64, 40, 52)),
                     _d_rank(2, True, True, rows=(64, 52, 64))])
        log = _Log()
        self.launcher.canonical_state_gate(self._spec(), 3, log)
        self.assertFalse(any(l.startswith("IPC MISMATCH W7/W10 group D") for l in log.lines), log.lines)
        self.assertTrue(any("RankState" in l and "kv x3 blob x3" in l for l in log.lines), log.lines)

    def test_row_owning_worker_that_reports_no_kv_is_refused(self):
        # a33ee80394's point under the records: the argv gives tp1 token rows,
        # the rank reports the KV page as not applicable -> refused, not 'n/a'
        # a Form A worker that built no window reports kv as not applicable
        # (build_rank_state) -- the records alone would grade it 'n/a' and pass
        self._write([_d_rank(0, False, True, rows=(64, 0, 40)),
                     _d_rank(1, True, False, rows=(64, 40, 52)),
                     _d_rank(2, True, True, rows=(64, 52, 64))])
        self.launcher.canonical_state_gate(self._spec(), 3, _Log())
        with self.assertRaises(self.launcher.PdFlipLaunchRefused) as cm:
            self.launcher.canonical_state_gate(self._spec(), 3, _Log(), kv_owner_ranks=[1, 2])
        self.assertIn("tp1 owns token rows", str(cm.exception))

    def test_missing_rank_refuses_even_when_log_counts_three(self):
        with open(self.log_path, "w") as f:
            f.write("#706 canonical KV page active\ncanonical GDN blob active\n" * 3)
        self._write([_p_rank(0), _p_rank(1)])
        with self.assertRaises(self.launcher.PdFlipLaunchRefused) as cm:
            self.launcher.canonical_state_gate(self._spec("P"), 3, _Log())
        self.assertIn("MISSING tp0pp2", str(cm.exception))
        self.assertIn("W10 PdFlipCanonicalPageMissing", str(cm.exception))

    def test_foreign_schema_record_refuses(self):
        self._write([_p_rank(0), _p_rank(1), _p_rank(2)])
        sd = rs.rank_state_dir_for_log(self.log_path)
        with open(os.path.join(sd, "P.tp0pp9.json"), "w") as f:
            f.write(json.dumps({"schema": 99}))
        with self.assertRaises(self.launcher.PdFlipLaunchRefused) as cm:
            self.launcher.canonical_state_gate(self._spec("P"), 3, _Log())
        self.assertIn("unreadable record", str(cm.exception))

    def test_launch_group_names_dir_and_drops_stale_records(self):
        # A record of an earlier launch into the same log path must never be
        # graded as this launch's.
        self._write([_p_rank(0), _p_rank(1), _p_rank(2)])
        spec = self._spec("P")
        log = _Log()
        self.launcher.launch_group(spec, self.tmp.name, log, dry=False)
        spec.proc.wait(timeout=30)
        sd = rs.rank_state_dir_for_log(self.log_path)
        self.assertEqual(spec.env[rs.RANK_STATE_ENV], sd)
        self.assertEqual(rs.read_group_states(sd), ([], []))
        self.assertTrue(any("3 record(s) of an earlier launch removed" in l for l in log.lines), log.lines)


class TestCacheControllerPublishes(unittest.TestCase):
    """The producer seam: the controller's facts land in the record."""

    def test_publish_writes_one_record_per_attach(self):
        from flliper.srt.environ import envs
        from flliper.srt.managers.cache_controller import pdflip_publish_rank_state

        class _Ctl:
            tp_rank, tp_size, pp_rank, pp_size, page_size = 2, 3, 0, 1, 64

        with tempfile.TemporaryDirectory() as d:
            ctl = _Ctl()
            old = os.environ.get("FLLIPER_PDFLIP_GROUP")
            os.environ["FLLIPER_PDFLIP_GROUP"] = "D"
            try:
                with envs.FLLIPER_PDFLIP_RANK_STATE_DIR.override(d):
                    for _ in range(2):
                        pdflip_publish_rank_state(
                            ctl, form_a_worker=True, canonical_on=True,
                            canonical_kv_built=True, canonical_blob_built=False,
                            has_mamba_pool=True, owner_ctx=(64, 52, 64),
                        )
            finally:
                if old is None:
                    os.environ.pop("FLLIPER_PDFLIP_GROUP", None)
                else:
                    os.environ["FLLIPER_PDFLIP_GROUP"] = old
            states, bad = rs.read_group_states(d)
            self.assertEqual(bad, [])
            self.assertEqual(len(states), 1)
            s = states[0]
            self.assertEqual((s.group, s.rank_key, s.role, s.kv_page_active, s.gdn_blob_applicable, s.kv_rows_per_page, s.seq),
                             ("D", "tp2pp0", rs.ROLE_FORM_A_WORKER, True, False, 12, 2))

    def test_no_dir_no_record(self):
        from flliper.srt.managers.cache_controller import pdflip_publish_rank_state

        class _Ctl:
            tp_rank, tp_size, pp_rank, pp_size, page_size = 0, 1, 0, 3, 64

        pdflip_publish_rank_state(
            _Ctl(), form_a_worker=False, canonical_on=True, canonical_kv_built=True,
            canonical_blob_built=True, has_mamba_pool=True, owner_ctx=None,
        )


if __name__ == "__main__":
    unittest.main()
