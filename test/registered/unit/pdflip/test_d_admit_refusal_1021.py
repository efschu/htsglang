"""D-ADMIT-REFUSED (pdflip/d_admit_refusal.py): every request D's admission does not take is named.

Metal rc27 (D log ..._893d09fa3f_1021_020616.D.log, 02:15:44-02:15:56): pdflip-2-30 (kurz, u=181,
prefix on D's device at 176960) was refused 11 times in 12.3 s -- each pass ``X-GATE verdict=admit``
then only ``#924D station=give_back ... site=admission_revert``. The H105 WAIT line was throttled on a
process-global counter (65..127 in that window), the host's pre-lock lifetime gate writes no line, and
the skip census reached the log only on post-wake passes. SOLL v2 R7: under X, a refusal is a defect --
it has to carry its gate and its numbers.

Pinned here:
* every refusing return of ``PrefillAdder.add_one_req`` stamps its gate (no raw ``return AddReqResult``);
* the lifetime gate stamps price/budget, the H105 group verdict stamps itself when this rank's gate let
  the request through, the twin pass names itself;
* the not-added line carries gate, group verdict with the decider's numbers, the budget terms, the X
  gate's uncached/limit and ``soll=R7`` under X; throttled per rid; an after-line names the refusals;
* the scheduler calls it at the #991 give-back site, before the give-back;
* the H105 WAIT line is throttled per rid (a fresh rid's refusals print although the global count is
  past 20 and not a multiple of 64).
"""

import inspect
import logging
import os
import re
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers.schedule_batch import Req  # noqa: E402
from flliper.srt.managers.schedule_policy import AddReqResult, PrefillAdder  # noqa: E402
from flliper.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult  # noqa: E402
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler  # noqa: E402
from flliper.srt.pdflip import d_admit_refusal as DAR  # noqa: E402
from flliper.srt.managers import tp_match_floor as TMF  # noqa: E402

D_ENV = {"FLLIPER_PDFLIP_GROUP": "D"}
PAGE = 64
MAX_NEW = 4096
RID = "pdflip-2-30"


def _tree_cache():
    tc = MagicMock()
    tc.supports_mamba.return_value = False
    tc.evictable_size.return_value = 0
    tc.full_evictable_size.return_value = 0
    tc.swa_evictable_size.return_value = 0
    tc.disable = False
    tc.uniform_avail_floor = None
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()
    return tc


def _allocator(available):
    a = MagicMock()
    a.available_size.return_value = available
    a.full_available_size.return_value = available
    a.swa_available_size.return_value = 0
    return a


def _adder(available, n_running=0):
    running = MagicMock()
    running.reqs = [SimpleNamespace(rid=f"run-{i}", output_ids=[],
                                    sampling_params=SimpleNamespace(max_new_tokens=MAX_NEW))
                    for i in range(n_running)]
    return PrefillAdder(
        page_size=PAGE, tree_cache=_tree_cache(), token_to_kv_pool_allocator=_allocator(available),
        running_batch=running, new_token_ratio=1.0, rem_input_tokens=10**9, rem_chunk_tokens=None,
        num_mixed_decode_tokens=0, priority_scheduling_preemption_threshold=0,
    )


def _req(fill, rid=RID):
    req = MagicMock(spec=Req)
    req.rid = rid
    req.priority = 0
    req.prefix_indices = []
    req.full_untruncated_fill_ids = list(range(fill))
    req.output_ids = []
    req.sampling_params = SimpleNamespace(max_new_tokens=MAX_NEW, ignore_eos=False)
    req.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    req.retracted_stain = False
    req.finished.return_value = False
    req.needs_host_load_back.return_value = False
    req.host_hit_length = 0
    req.last_node = MagicMock()
    req.born_spilled = False
    req.born_spilled_deep = False
    return req


def _sched():
    return SimpleNamespace()


def _follow(code="NO_TOKEN", price=4341, budget=1234):
    def follow(req, gate, price_=None, budget_=None):
        follow.last_verdict = (str(req.rid), code, price, budget)
        return None if code == "ADMIT" else AddReqResult[code]

    follow.host_decides_load_back = False
    follow.last_verdict = None
    return follow


class TheAdderNamesItsGate(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        DAR.STATS.clear()

    def test_no_raw_refusing_return_is_left_in_add_one_req(self):
        src = inspect.getsource(PrefillAdder.add_one_req)
        lines = src.splitlines()
        # a raw refusing return is allowed only right behind its own stamp (the
        # #282 no-room wait keeps its literal return, pinned by its own test)
        raw = [ln.strip() for i, ln in enumerate(lines)
               if re.search(r"return\s+AddReqResult\.", ln)
               and not any("_d_admit_refusal.stamp(" in p for p in lines[max(0, i - 12):i])]
        self.assertEqual(raw, [], "a refusing return without a gate name")
        bare = [ln.strip() for ln in src.splitlines() if re.fullmatch(r"\s*return _gate\s*", ln)]
        self.assertEqual(bare, [], "a gate verdict returned without its name")

    def test_the_lifetime_gate_stamps_price_and_budget(self):
        # 181 uncached + max_new 4096 + page 64 = 4341 against 3000: refused
        adder = _adder(3000)
        req = _req(181)
        self.assertEqual(adder.add_one_req(req, truncation_align_size=None), AddReqResult.NO_TOKEN)
        rid, gate, code, nums = getattr(adder, DAR.STAMP_ATTR)
        self.assertEqual((rid, gate, code), (RID, "lifetime", "NO_TOKEN"))
        self.assertEqual(nums["price"], 181 + MAX_NEW + PAGE)
        self.assertEqual(nums["budget"], 3000)
        self.assertEqual(nums["extend"], 181)

    def test_an_admitted_request_leaves_no_stamp_of_an_earlier_refusal(self):
        adder = _adder(3000)
        adder.add_one_req(_req(181), truncation_align_size=None)
        adder.token_to_kv_pool_allocator.available_size.return_value = 10**6
        res = adder.add_one_req(_req(181, rid="pdflip-2-31"), truncation_align_size=None)
        self.assertEqual(res, AddReqResult.CONTINUE)
        self.assertIsNone(getattr(adder, DAR.STAMP_ATTR))

    def test_the_group_verdict_names_itself_when_this_rank_would_admit(self):
        adder = _adder(10**6)
        adder.form_a_admission_follow = _follow("NO_TOKEN", 4341, 1234)
        req = _req(181)
        self.assertEqual(adder.add_one_req(req, truncation_align_size=None), AddReqResult.NO_TOKEN)
        self.assertEqual(getattr(adder, DAR.STAMP_ATTR)[1], "h105_group_verdict")

    def test_the_local_gate_keeps_its_name_through_the_group_verdict(self):
        adder = _adder(3000)
        adder.form_a_admission_follow = _follow("NO_TOKEN", 4341, 3000)
        adder.add_one_req(_req(181), truncation_align_size=None)
        self.assertEqual(getattr(adder, DAR.STAMP_ATTR)[1], "lifetime")


class TheNotAddedLine(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        DAR.STATS.clear()

    def _refused(self, sched, adder, req, now):
        res = adder.add_one_req(req, truncation_align_size=None)
        return DAR.note_not_added(sched, adder, req, res, site="admission_revert",
                                  gave_back="slot", now=now, env=D_ENV)

    def test_the_specimen_kurz_under_x_is_named_with_gate_numbers_and_r7(self):
        adder = _adder(10**6)
        adder.form_a_admission_follow = _follow("NO_TOKEN", 4341, 1234)
        req = _req(181)
        setattr(req, DAR.XGATE_ATTR, (181, 6464))
        line = self._refused(_sched(), adder, req, now=100.0)
        self.assertIsNotNone(line)
        for part in ("rid=pdflip-2-30", "res=NO_TOKEN", "gate=h105_group_verdict", "group=NO_TOKEN",
                     "decider_price=4341", "decider_budget=1234", "uncached=181", "limit=6464",
                     "under_x=1", "soll=R7", "site=admission_revert", "gave_back=slot",
                     "refusals=1", "available=", "evictable=", "offset="):
            self.assertIn(part, line)

    def test_a_refusal_above_x_carries_no_r7_mark(self):
        adder = _adder(3000)
        req = _req(181)
        setattr(req, DAR.XGATE_ATTR, (20000, 6464))
        line = self._refused(_sched(), adder, req, now=1.0)
        self.assertIn("under_x=0", line)
        self.assertNotIn("soll=R7", line)

    def test_throttle_is_per_rid_and_the_after_line_counts(self):
        sched = _sched()
        printed = []
        for k in range(1, 12):  # the specimen's 11 refusals
            adder = _adder(3000)
            printed.append(self._refused(sched, adder, _req(181), now=100.0 + k) is not None)
        # 1, 2, 3, 4, 8 print; the others are counted
        self.assertEqual([i + 1 for i, p in enumerate(printed) if p], [1, 2, 3, 4, 8])
        # a second rid prints its first refusal although the first one is at 11
        other = self._refused(sched, _adder(3000), _req(181, rid="pdflip-2-32"), now=120.0)
        self.assertIn("refusals=1", other)
        line = DAR.note_admitted(sched, _req(181), now=112.3, env=D_ENV)
        self.assertIn("rid=pdflip-2-30 refusals=11 waited_s=11.3", line)
        self.assertIn("lifetime:11", line)
        self.assertIsNone(DAR.note_admitted(sched, _req(181), now=113.0, env=D_ENV))
        self.assertEqual(DAR.STATS["lifetime"], 12)

    def test_a_gate_change_prints_at_once(self):
        sched = _sched()
        for k in range(1, 6):
            self._refused(sched, _adder(3000), _req(181), now=float(k))
        adder = _adder(10**6)
        adder.form_a_admission_follow = _follow("NO_TOKEN", 4341, 1234)
        line = self._refused(sched, adder, _req(181), now=7.0)   # k=6, not due by count
        self.assertIsNotNone(line)
        self.assertIn("gate=h105_group_verdict", line)

    def test_off_group_d_and_switched_off_nothing_is_printed(self):
        adder = _adder(3000)
        req = _req(181)
        res = adder.add_one_req(req, truncation_align_size=None)
        self.assertIsNone(DAR.note_not_added(_sched(), adder, req, res, site="x", env={}))
        self.assertIsNone(DAR.note_not_added(
            _sched(), adder, req, res, site="x",
            env={"FLLIPER_PDFLIP_GROUP": "D", DAR.ENV: "0"}))

    def test_an_unreadable_adder_never_raises(self):
        line = DAR.note_not_added(_sched(), object(), _req(1), AddReqResult.OTHER, site="s", env=D_ENV)
        self.assertIn("gate=unlabeled", line)
        self.assertIn("unreadable", line)


class TheSchedulerNamesBeforeTheGiveBack(unittest.TestCase):
    def test_note_not_added_sits_before_the_admission_revert_give_back(self):
        from flliper.srt.managers import scheduler as S

        src = open(S.__file__).read()
        i_note = src.index("_d_admit_refusal.note_not_added(")
        i_give = src.index('release_admission_acquired_mamba_slot(\n                                req, self.tree_cache, site="admission_revert"')
        i_not_added = src.rindex("if not added:", 0, i_note)
        self.assertLess(i_not_added, i_note)
        self.assertLess(i_note, i_give)
        self.assertEqual(src.count("_d_admit_refusal.note_admitted(self, req)"), 2)  # added on both result paths
        self.assertIn("_d_admit_refusal.XGATE_ATTR", src)


class TheH105WaitLineIsThrottledPerRid(unittest.TestCase):
    def setUp(self):
        TMF._ADMISSION_WAIT.clear()

    def test_a_fresh_rids_refusals_print_while_the_global_count_is_quiet(self):
        TMF._STATS["admission_refused"] = 64  # the specimen: n=64 at 02:15:05, quiet until 128
        with self.assertLogs(TMF.logger, level="INFO") as cap:
            for _ in range(11):
                TMF._note_admission_wait(RID, "NO_TOKEN", 4341, 1234, "ADMIT", 4341, 9999)
            TMF._note_admission_wait(RID, TMF.ADMISSION_ADMIT, 4341, 81672, "ADMIT", 4341, 81672)
        waits = [r for r in cap.output if "ADMISSION WAIT rid=pdflip-2-30" in r]
        self.assertEqual(len(waits), 5)  # refusals 1, 2, 3, 4, 8
        self.assertTrue(any("AFTER-WAIT rid=pdflip-2-30 refusals=11" in r for r in cap.output))

    def test_a_change_of_the_hosts_code_prints(self):
        TMF._STATS["admission_refused"] = 100
        for _ in range(4):
            TMF._note_admission_wait(RID, "NO_TOKEN", 1, 1, "ADMIT", 1, 1)
        with self.assertLogs(TMF.logger, level="INFO") as cap:
            TMF._note_admission_wait(RID, "OTHER", 1, 1, "ADMIT", 1, 1)  # k=5, code changed
        self.assertTrue(any("host=OTHER" in r for r in cap.output))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
