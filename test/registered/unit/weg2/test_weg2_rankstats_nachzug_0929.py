"""DASHBOARD-AUS-IPC Nachzug (29.09.): A14, C5, C2.full_token_usage, E2.mamba_tok,
E2.prefetch.timeout leave the log (operator to the producer seat: "Drei Felder
bleiben aus dem Log, weil kein Produzent sie schreibt").

  * A14 -- the scheduler's death path records the stop in the rank's rankstats
    file (``stops``), synchronously; the front publishes each stop once as the
    event ``rank_stop`` (group, rank, code) when a group's health verdict turns
    failing/held/dead.
  * C5  -- RankState carries ``kv {holds_kv, kv_tokens, share}`` and ``seats``
    from the scheduler's own counters; a change rewrites the last record once.
  * C2/E2 -- full_token_usage from the pool stats the batch line prints,
    mamba_tok from the MAMBA-HOST-RESUME acceptances, prefetch.timeout from the
    #1157 reaps; ``deferred`` is the census' own ``deferred`` key.

RED on 9969bd336b: no ``note_stop``/``stops``, no ``kv``/``seats`` on RankState,
no ``note_capacity``, no ``publish_rank_stops``, no ``full_token_usage``/
``timeout``, mamba_tok stays None, deferred reads defer_refused.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler_components import metrics_reporter as mr_mod
from sglang.srt.mem_cache import match_refusal_census as mrc
from sglang.srt.weg2 import front_state_ipc as fsi
from sglang.srt.weg2 import rank_state as rs
from sglang.srt.weg2 import rankstats
from sglang.srt.weg2 import state_file as sf
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _mr():
    return SimpleNamespace(prefill_tokens_total=0, gen_tokens_total=0,
                           spec_total_num_accept_tokens=0, spec_total_num_forward_ct=0,
                           rank_prefill_log=mr_mod.RankPrefillLog(), decode_round_log=None,
                           accept_len_ewma=None, accept_rate_ewma=None, last_cuda_graph=None,
                           last_running_reqs=None, last_pending_tokens=None,
                           last_full_token_usage=0.4271)


def _sched(**kw):
    s = SimpleNamespace(forward_ct=0, metrics_reporter=_mr(), waiting_queue=[],
                        running_batch=SimpleNamespace(reqs=[]), max_total_num_tokens=262144,
                        max_running_requests=6, tree_cache=SimpleNamespace(_1157_reaped_n=3))
    s.__dict__.update(kw)
    return s


def _d_rank(tp, worker, rows=None):
    return rs.build_rank_state(
        group="D", tp_rank=tp, tp_size=3, pp_rank=0, pp_size=1, form_a_worker=worker,
        canonical_on=True, canonical_kv_built=True, canonical_blob_built=not worker,
        has_mamba_pool=True, page_size=64, owner_ctx=rows, seq=1)


class _Base(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        rs.reset_capacity()

    def tearDown(self):
        rankstats._CURRENT = None
        rs.reset_capacity()
        self.tmp.cleanup()

    def _rank(self, group="D", tp=0, sched=None):
        sched = sched or _sched()
        st = rankstats.RankStats(state_dir=self.dir, group=group, tp_rank=tp, pp_rank=0,
                                 read_counters=lambda: rankstats.scheduler_counters(sched),
                                 period=3600.0)
        rankstats._CURRENT = st
        return st


class TestA14StopRecord(_Base):
    def test_stop_entry_names_w_code_and_ticket(self):
        e = rankstats.stop_entry(RuntimeError("Weg2X: #1233 W27 PP WIDTH DIVERGENCE REFUSED"), now=5.0)
        self.assertEqual((e["t"], e["code"], e["exc"], e["ticket"], e["reason"]),
                         (5.0, "W27_RuntimeError", "RuntimeError", "#1233", "scheduler_exception"))
        self.assertEqual(rankstats.stop_entry(ValueError("boom"))["code"], "ValueError")

    def test_note_stop_writes_the_file_at_once(self):
        st = self._rank()
        self.assertIsNotNone(rankstats.note_stop(RuntimeError("CUDA out of memory")))
        with open(st.path) as f:
            rec = json.load(f)
        self.assertEqual(rec["stops"]["n"], 1)
        self.assertEqual(rec["stops"]["last"][0]["exc"], "RuntimeError")

    def test_switch_off_records_nothing(self):
        rankstats._CURRENT = None
        self.assertIsNone(rankstats.note_stop(RuntimeError("x")))

    def test_death_path_records_before_hold_and_sigquit(self):
        from sglang.srt.managers import scheduler

        src = inspect.getsource(scheduler.run_scheduler_process)
        i_stop = src.index("note_stop(scheduler_exc)")
        self.assertLess(src.index("Scheduler hit an exception"), i_stop)
        self.assertLess(i_stop, src.index("debug_hold.maybe_hold("))
        self.assertLess(i_stop, src.rindex("SIGQUIT"))


class TestA14FrontEvent(_Base):
    def _boot(self, bid="b1"):
        root = os.path.join(self.dir, "state")
        os.makedirs(root, exist_ok=True)
        d = sf.init(root, bid, "boot", {})
        sf.transition(d, None, fields={"groups.D": {"state": "serving", "rankstate_dir": self.dir}},
                      writer="launcher")
        return d

    def test_stop_becomes_one_rank_stop_event(self):
        d = self._boot()
        self._rank(tp=1)
        rankstats.note_stop(RuntimeError("Weg2Y: W35 credit"))
        seen = set()
        self.assertEqual(fsi.publish_rank_stops(d, "D", seen), 1)
        self.assertEqual(fsi.publish_rank_stops(d, "D", seen), 0)  # once, not per verdict
        ev = [e for e in sf.events(d) if e["type"] == "rank_stop"]
        self.assertEqual(len(ev), 1)
        self.assertEqual((ev[0]["group"], ev[0]["rank"], ev[0]["code"]), ("D", "tp1pp0", "W35_RuntimeError"))
        self.assertEqual(ev[0]["data"]["reason"], "scheduler_exception")
        self.assertIn("t", ev[0]["data"])

    def test_no_state_or_no_rank_dir_publishes_nothing(self):
        self.assertEqual(fsi.publish_rank_stops("", "D", set()), 0)
        root = os.path.join(self.dir, "s2")
        os.makedirs(root)
        d = sf.init(root, "b2", "boot", {})
        self.assertEqual(fsi.publish_rank_stops(d, "D", set()), 0)

    def test_front_publishes_on_failing_held_dead(self):
        from sglang.srt.weg2 import front

        self.assertIn("publish_rank_stops", inspect.getsource(front))
        self.assertEqual(fsi.RANK_STOP_VERDICTS, ("failing", "held", "dead"))


class TestC5Capacity(_Base):
    def test_record_carries_kv_and_seats_after_note(self):
        rs.note_capacity(kv_tokens=262144, seats=6)
        rs.write_rank_state(_d_rank(1, True, rows=(64, 40, 52)), self.dir)
        (back,), bad = rs.read_group_states(self.dir)
        self.assertEqual(bad, [])
        self.assertEqual(back.seats, 6)
        self.assertEqual(back.kv, {"holds_kv": True, "kv_tokens": 262144, "share": round(12 / 64, 4)})

    def test_change_rewrites_the_last_record_once(self):
        rs.write_rank_state(_d_rank(0, False), self.dir)
        self.assertIsNotNone(rs.note_capacity(kv_tokens=100000, seats=4))
        self.assertIsNone(rs.note_capacity(kv_tokens=100000, seats=4))  # unchanged: no write
        (back,), _ = rs.read_group_states(self.dir)
        self.assertEqual((back.kv["kv_tokens"], back.seats), (100000, 4))
        rs.note_capacity(kv_tokens=120000, seats=4)
        (back,), _ = rs.read_group_states(self.dir)
        self.assertEqual(back.kv["kv_tokens"], 120000)

    def test_zero_row_worker_holds_no_kv(self):
        z = _d_rank(2, True, rows=(64, 64, 64))
        self.assertFalse(rs.capacity_fields(z, 262144, 6)["kv"]["holds_kv"])

    def test_timer_syncs_the_scheduler_counters(self):
        rs.write_rank_state(_d_rank(0, False), self.dir)
        st = self._rank(sched=_sched(max_total_num_tokens=77777, max_running_requests=3))
        st.write_once()
        st.sync_capacity()
        (back,), _ = rs.read_group_states(self.dir)
        self.assertEqual((back.kv["kv_tokens"], back.seats), (77777, 3))

    def test_schema1_record_without_kv_still_reads(self):
        d = json.loads(_d_rank(0, False).to_json())
        d["schema"] = 1
        for k in ("vram", "kv", "seats"):
            d.pop(k, None)
        self.assertIsNone(rs.RankState.from_json(json.dumps(d)).kv)

    def test_scheduler_names_capacity_after_init(self):
        from sglang.srt.managers import scheduler

        self.assertIn("note_capacity(", inspect.getsource(scheduler.run_scheduler_process))


class TestC2E2Counters(_Base):
    def test_full_token_usage_timeout_mamba_tok_cap(self):
        from sglang.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent

        had = "_host_resume_tok" in MambaComponent.__dict__
        old = MambaComponent.__dict__.get("_host_resume_tok")
        MambaComponent._host_resume_tok = 4480
        try:
            c = rankstats.scheduler_counters(_sched())
        finally:
            if had:
                MambaComponent._host_resume_tok = old
            else:
                del MambaComponent._host_resume_tok
        self.assertEqual(c["sched"]["full_token_usage"], 0.4271)
        self.assertEqual(c["cache"]["mamba_tok"], 4480)
        self.assertEqual(c["cache"]["prefetch"]["timeout"], 3)
        self.assertEqual(c["cap"], {"kv_tokens": 262144, "seats": 6})

    def test_deferred_is_the_census_deferred_key(self):
        saved = dict(mrc.PREFETCH_GATE_COUNTS)
        try:
            mrc.PREFETCH_GATE_COUNTS.clear()
            mrc.PREFETCH_GATE_COUNTS.update({"deferred": 5, "defer_refused": 2, "too_short": 3,
                                             "vote_negative": 1, "anchor_pool_exhausted": 7,
                                             "too_short_tokens": 999})
            p = rankstats.scheduler_counters(_sched())["cache"]["prefetch"]
        finally:
            mrc.PREFETCH_GATE_COUNTS.clear()
            mrc.PREFETCH_GATE_COUNTS.update(saved)
        self.assertEqual((p["deferred"], p["defer_refused"], p["refused"]), (5, 2, 4))

    def test_counter_sites_are_single_increments(self):
        from sglang.srt.mem_cache import unified_radix_cache
        from sglang.srt.mem_cache.unified_cache_components import mamba_component

        self.assertIn("_host_resume_tok", inspect.getsource(mamba_component))
        self.assertIn("_1157_reaped_n", inspect.getsource(unified_radix_cache))
        self.assertEqual(inspect.getsource(mr_mod).count(
            "self.last_full_token_usage = pool_stats.full_token_usage"), 2)
        # the counter site imports nothing: no rankstats, no file I/O in the match walk
        tree = ast.parse(inspect.getsource(mamba_component))
        mods = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names} | {getattr(n, "module", None) or "" for n in ast.walk(tree)
                                     if isinstance(n, ast.ImportFrom)}
        self.assertFalse([m for m in mods if "rankstats" in m])


if __name__ == "__main__":
    unittest.main()
