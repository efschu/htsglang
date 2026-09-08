"""T3 / T7 / T8 + C14 + the refusal form -- the flip-cost slice (C9-C17, C21).

Every class here names the mutation that turns it red, because a test whose own
failure mode is unstated is a claim, not a check:

* T3 (C9/C10/C11) -- serialise the two legs in ``Front.flip``       -> the
  overlap assertion fails; drop the per-tag report from the W4 text -> the tag
  assertion fails; let L5 print ``interleave`` as sleep+wake        -> the
  "NOT sleep+wake" assertion fails.
* T7 (C15) -- delete the ok-bit gather, or move it ABOVE the monitored barrier
  -> the three-rank case either never raises or hangs past its bound.
* T8 (C12/C13) -- key the lock on the uuid alone, or split it on a card below
  R17's gate -> the exclusion table changes on exactly one row each.
* C14 -- let the waiting rank proceed on an unfunded tag, or wait past the
  peer's ``leg_complete`` -> the credit case stops refusing by name.
* the refusal form -- let any arm of the launcher leave as a bare exception
  -> ``cli`` no longer returns 2 with the one named line.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import shutil
import tempfile
import time
import unittest

from sglang.srt.managers import weg2_memory_saver as ms
from sglang.srt.weg2 import front as front_mod

MIB = 1024 * 1024


# ---------------------------------------------------------------------------
# T3 -- the driver ordering
# ---------------------------------------------------------------------------


class _Call:
    def __init__(self, group: str, path: str, body):
        self.group = group
        self.path = path
        self.tags = tuple((body or {}).get("tags", ()))
        self.t_in = time.perf_counter()
        self.t_out = 0.0


class GatheredLegsTest(unittest.TestCase):
    """T3: one gathered pair per family, kv strictly outside, W4 names the tags."""

    FAMILY_MS = 0.05

    def _front(self):
        f = front_mod.Front(
            prefill="http://p", decode="http://d", awake="D", tag="t3",
            store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
            weight_chunks=2,
        )
        f.calls = []
        f.stops = []
        f.rpc_result = {}
        f.body_for = {}

        async def rpc(g, path, body, timeout):
            call = _Call(g.name, path, body)
            f.calls.append(call)
            if path == "/flush_cache":
                call.t_out = time.perf_counter()
                return 200, "{}"
            await asyncio.sleep(self.FAMILY_MS if len(call.tags) > 1 else 0.001)
            call.t_out = time.perf_counter()
            key = (g.name, path, call.tags)
            code = f.rpc_result.get((g.name, len(call.tags)), 200)
            return code, f.body_for.get(key, f._default_body(call.tags))

        def _default_body(tags):
            import json

            return json.dumps({
                "per_tag": {t: [1024 * MIB, 11.0] for t in tags},
                "critical_path": "rank=2 card=GPU-x ms=42",
            })

        f._default_body = _default_body
        f.rpc = rpc
        f.do_stop = lambda name, detail: f.stops.append((name, detail))
        return f

    def _flip(self, f):
        asyncio.run(f.flip("D", "P"))

    def test_the_two_weights_legs_are_in_flight_at_the_same_time(self):
        f = self._front()
        self._flip(f)
        legs = [c for c in f.calls if len(c.tags) > 1]
        self.assertEqual(len(legs), 2, [c.path for c in f.calls])
        s, w = legs
        self.assertLess(s.t_in, w.t_out)
        self.assertLess(w.t_in, s.t_out)
        overlap = min(s.t_out, w.t_out) - max(s.t_in, w.t_in)
        self.assertGreater(
            overlap, 0.0,
            "SERIALISED: the whole point of C9 is that W's per-tag releases fund "
            "S's acquires, which they cannot do if the second RPC is only sent "
            "after the first has returned",
        )

    def test_one_rpc_per_group_per_leg_not_one_per_tag(self):
        f = self._front()
        self._flip(f)
        weights = [c for c in f.calls if c.path.endswith("memory_occupation")
                   and len(c.tags) > 1]
        self.assertEqual(len(weights), 2)
        self.assertEqual(set(weights[0].tags), set(f.weights_tags))
        self.assertEqual(set(weights[1].tags), set(f.weights_tags))

    def test_kv_release_is_strictly_first_and_kv_resume_strictly_last(self):
        f = self._front()
        self._flip(f)
        occ = [c for c in f.calls if c.path.endswith("memory_occupation")]
        first, last = occ[0], occ[-1]
        self.assertEqual((first.group, first.path, first.tags),
                         ("D", "/release_memory_occupation", ("kv_cache",)))
        self.assertEqual((last.group, last.path, last.tags),
                         ("P", "/resume_memory_occupation", ("kv_cache",)))
        for leg in [c for c in occ if len(c.tags) > 1]:
            self.assertGreaterEqual(leg.t_in, first.t_out)
            self.assertLessEqual(leg.t_out, last.t_in)

    def test_a_failing_leg_names_what_each_group_completed(self):
        # C10.  Both legs were in flight, so both groups' completion state is
        # part of the fault; "HTTP 500" alone cannot say which half is parked.
        f = self._front()
        f.rpc_result[("P", len(f.weights_tags))] = 500
        self._flip(f)
        self.assertEqual(len(f.stops), 1, f.stops)
        name, detail = f.stops[0]
        self.assertEqual(name, "W4 Weg2WakeRefused")
        for tag in f.weights_tags:
            self.assertIn(tag, detail)
        self.assertIn("D completed", detail)
        self.assertIn("P completed", detail)
        self.assertIn("BOTH sides", detail)

    def test_an_answer_without_a_per_tag_report_says_so_instead_of_claiming_none(self):
        f = self._front()
        f.body_for = {}
        f._default_body = lambda tags: "not json at all"
        f.rpc_result[("P", len(f.weights_tags))] = 500
        self._flip(f)
        _, detail = f.stops[0]
        self.assertIn("not JSON", detail)
        self.assertIn("[]", detail)

    def test_l5_says_interleave_is_not_sleep_plus_wake_and_prints_the_overlap(self):
        f = self._front()
        self._flip(f)
        rec = f.flip_log[-1]
        self.assertIn("legs_wall_ms", rec)
        self.assertGreater(rec["overlap_ms"], 0)
        self.assertLess(
            rec["legs_wall_ms"], rec["sleep_leg_ms"] + rec["wake_leg_ms"],
            "the gather's wall time must be less than the sum of its legs, or "
            "nothing overlapped",
        )
        self.assertIn("rank=", rec["critical_path"])
        self.assertTrue(rec["critical_path"].startswith(("sleep/", "wake/")))

    def test_the_flip_leg_form_constant_and_the_code_agree(self):
        self.assertEqual(front_mod.FLIP_LEG_FORM, "interleave")


# ---------------------------------------------------------------------------
# T8 -- the direction-split lock key
# ---------------------------------------------------------------------------


class DirectionKeyTest(unittest.TestCase):
    """C12/C13 + A1-4: per card, from the measured ratio, or not at all."""

    OK = "GPU-5090"        # step-0 measured 1.759
    NULL = "GPU-3080-x4"   # step-0 measured 1.316 -- DUPLEX-NULL
    RATIOS = {OK: 1.759, NULL: 1.316}

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="weg2dirkey-")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def _key(self, uuid, direction):
        return os.path.basename(
            ms.pcie_lock_path(uuid, lock_dir=self.dir, direction=direction,
                              ratios=self.RATIOS)
        )

    def test_a_card_above_the_gate_splits_and_the_two_keys_differ(self):
        d2h, h2d = self._key(self.OK, "d2h"), self._key(self.OK, "h2d")
        self.assertNotEqual(d2h, h2d)
        self.assertTrue(d2h.endswith(".d2h.lock"))
        self.assertTrue(h2d.endswith(".h2d.lock"))

    def test_the_x4_card_keeps_one_key_and_still_serialises(self):
        self.assertEqual(self._key(self.NULL, "d2h"), self._key(self.NULL, "h2d"))
        self.assertNotIn("d2h", self._key(self.NULL, "d2h"))

    def test_same_direction_pairs_always_share_a_key(self):
        for uuid in (self.OK, self.NULL):
            self.assertEqual(self._key(uuid, "d2h"), self._key(uuid, "d2h"))

    def test_the_exclusion_table_is_exactly_the_measured_one(self):
        # rows: (card, dir A, dir B) -> do they exclude each other?
        table = {
            (self.OK, "d2h", "d2h"): True,
            (self.OK, "h2d", "h2d"): True,
            (self.OK, "d2h", "h2d"): False,
            (self.NULL, "d2h", "d2h"): True,
            (self.NULL, "h2d", "h2d"): True,
            (self.NULL, "d2h", "h2d"): True,
        }
        for (uuid, a, b), excludes in table.items():
            self.assertEqual(self._key(uuid, a) == self._key(uuid, b), excludes,
                             f"{uuid} {a} vs {b}")

    def test_an_unmeasured_card_never_splits(self):
        self.assertEqual(self._key("GPU-unknown", "d2h"), self._key("GPU-unknown", "h2d"))

    def test_two_holders_of_one_key_really_do_exclude(self):
        # The keys are only meaningful if the lock behind them works: one held,
        # the other must time out rather than proceed.
        with ms.pcie_transfer_lock(nvml_uuid=self.NULL, lock_dir=self.dir,
                                   direction="d2h", timeout_s=0.2, label="a",
                                   ratios=self.RATIOS):
            with self.assertRaises(ms.Weg2PcieLockTimeout):
                with ms.pcie_transfer_lock(nvml_uuid=self.NULL, lock_dir=self.dir,
                                           direction="h2d", timeout_s=0.2, label="b",
                                           ratios=self.RATIOS):
                    pass

    def test_opposite_directions_on_a_split_card_do_not_exclude(self):
        with ms.pcie_transfer_lock(nvml_uuid=self.OK, lock_dir=self.dir,
                                   direction="d2h", timeout_s=0.2, label="a",
                                   ratios=self.RATIOS):
            with ms.pcie_transfer_lock(nvml_uuid=self.OK, lock_dir=self.dir,
                                       direction="h2d", timeout_s=0.2, label="b",
                                       ratios=self.RATIOS):
                pass

    def test_a_bad_direction_is_refused_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            ms.pcie_lock_path(self.OK, lock_dir=self.dir, direction="both")

    def test_the_published_table_is_parsed_and_an_unparsable_row_is_dropped(self):
        # An unparsable RATIO is still dropped -- a defaulted ratio is the hand
        # number spec 10.3 forbids.  What is no longer permissive is the
        # DECISION: FIX 2 round 2 makes a published string that names no
        # decision a named refusal, so the rows here carry theirs.
        parsed = ms.duplex_ratios(
            "format=v2,GPU-a=1.759:split,GPU-b=nonsense:split,,GPU-c=1.2:split")
        self.assertEqual(parsed, {"GPU-a": 1.759, "GPU-c": 1.2})
        self.assertEqual(
            ms.duplex_splits(
                "format=v2,GPU-a=1.759:split,GPU-b=nonsense:split,,GPU-c=1.2:split"),
            {"GPU-a": True, "GPU-b": True, "GPU-c": True},
            "a card whose ratio nobody could read still has a decision")


# ---------------------------------------------------------------------------
# C14 -- the VRAM credit
# ---------------------------------------------------------------------------


class VramCreditTest(unittest.TestCase):
    """S publishes, W waits, and every ending is named."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="weg2credit-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.uuid = "GPU-c14"
        self.credit = ms.VramCredit(self.uuid, credit_dir=self.dir)

    def test_the_publisher_is_monotone_within_a_leg(self):
        self.credit.begin_leg(1)
        self.assertEqual(self.credit.publish("weights_0", 100 * MIB), 100 * MIB)
        self.assertEqual(self.credit.publish("weights_1", 50 * MIB), 150 * MIB)
        self.assertEqual(self.credit.read()["credit_bytes"], 150 * MIB)

    def test_a_new_leg_resets_the_counter_and_that_is_the_only_reset(self):
        self.credit.begin_leg(1)
        self.credit.publish("weights_0", 100 * MIB)
        self.credit.begin_leg(2)
        self.assertEqual(self.credit.read()["credit_bytes"], 0)
        self.assertEqual(self.credit.read()["epoch"], "2",
                         "the stamp is a TOKEN, not a number -- see credit_epoch")

    def test_a_previous_legs_TERMINAL_state_is_not_read_as_this_flips_funding(self):
        # FIX 1 round 1, finding 4.  The epoch was WRITE-ONLY: begin_leg stamped
        # it and read()/wait_for() never looked.  C9 issues both legs in ONE
        # asyncio.gather, so there is no happens-before between S's begin_leg
        # and W's first wait_for; the per-card file is shared by the two
        # co-located ranks and alternates writers, so at flip N+1 W typically
        # read flip N's terminal state -- leg_complete with a whole image of
        # credit -- and resumed on device bytes nobody had freed.  That is the
        # rank-local CUDA OOM of spec 10.5, silently licensed by the guard that
        # exists to prevent it.
        self.credit.begin_leg(7)
        self.credit.publish("weights_0", 10 * 1024 * MIB)
        self.credit.leg_complete()
        with self.assertRaises(ms.Weg2VramCreditRefused) as cm:
            self.credit.wait_for(2 * 1024 * MIB, budget_s=0.3, tag="weights_0",
                                 free_bytes_now=0, epoch=8)
        self.assertIn("EXPIRED", str(cm.exception),
                      "a stale leg_complete must not refuse instantly either -- "
                      "it says nothing about the leg now in flight")
        self.assertIn("epoch 7", str(cm.exception))
        self.assertIn("not this flip's 8", str(cm.exception))

    def test_a_previous_BOOTs_terminal_credit_is_not_read_as_this_boots(self):
        # FIX 2 round 2, finding 1 -- THE CROSS-BOOT HALF, and a REGRESSION of
        # the property the grandparent had.  FIX 1 dated the counter with
        # front.Front.self.epoch, a counter initialised to 0 at every front
        # start; the counter FILE lives in /dev/shm and nothing unlinked it, so
        # boot A that ran K flips left epoch K-1 on disk with leg_complete and a
        # whole image of credit, and boot B's flip K-1 read it as ITS OWN.  Spec
        # 9.1 demands >= 4 flips per direction, so every boot walks through the
        # range its predecessor left behind.  Measured on this rig 2026-09-08:
        # three terminal files from boot weg2rg2, one carrying 13912 MiB.
        # int(time.time()) -- what FIX 1 replaced -- never collided; a per-front
        # counter always will.  The epoch now names the BOOT and the flip.
        boot_a = ms.VramCredit(self.uuid, credit_dir=self.dir)
        boot_a.begin_leg(ms.credit_epoch("1757300000", 7))
        boot_a.publish("weights_0", 13 * 1024 * MIB)
        boot_a.leg_complete()

        boot_b = ms.VramCredit(self.uuid, credit_dir=self.dir)
        with self.assertRaises(ms.Weg2VramCreditRefused) as cm:
            boot_b.wait_for(2 * 1024 * MIB, budget_s=0.3, tag="weights_0",
                            free_bytes_now=0,
                            epoch=ms.credit_epoch("1757399999", 7))
        self.assertIn("EXPIRED", str(cm.exception),
                      "a previous BOOT's terminal state must be waited past for "
                      "the same reason a previous flip's is -- it says nothing "
                      "about the leg now in flight")
        self.assertIn("1757300000.7", str(cm.exception))

    def test_the_credit_epoch_names_the_boot_AND_the_flip(self):
        # The flip index alone cannot date a file that outlives the front, and
        # an old file's bare integer must not compare equal to a composed token.
        self.assertNotEqual(ms.credit_epoch("1757300000", 7),
                            ms.credit_epoch("1757399999", 7))
        self.assertNotEqual(ms.credit_epoch("1757300000", 7), "7")
        # boot weg2rg2's leftovers on this rig carry a bare integer stamp.
        legacy = ms.VramCredit(self.uuid, credit_dir=self.dir)
        legacy.begin_leg(12)
        legacy.publish("weights_0", 13 * 1024 * MIB)
        legacy.leg_complete()
        with self.assertRaises(ms.Weg2VramCreditRefused):
            legacy.wait_for(2 * 1024 * MIB, budget_s=0.3, tag="weights_0",
                            free_bytes_now=0,
                            epoch=ms.credit_epoch("1757399999", 12))

    def test_teardown_unlinks_the_credit_counters_it_created(self):
        # Two halves, because neither closes the other: the epoch makes a
        # leftover harmless (including after a CRASH, which teardown never
        # runs after), and teardown makes it absent.
        from sglang.srt.weg2 import launcher

        credit = ms.VramCredit(self.uuid, credit_dir=self.dir)
        credit.begin_leg(ms.credit_epoch("1757300000", 1))
        self.assertTrue(os.path.exists(credit.path))
        removed = launcher.remove_vram_credit_counters(
            self.dir, boot_nonce="1757300000")
        self.assertEqual(len(removed), 1, removed)
        self.assertFalse(os.path.exists(credit.path))

    def test_teardown_unlinks_only_THIS_boots_counters(self):
        # FIX 3 round 3.  The predecessor unlinked EVERY .weg2-vram-credit-* in
        # the directory -- so a teardown of boot A takes a LIVE boot B's counter
        # mid-flip, and B then waits on a file that has gone.  The nonce the
        # same commit introduced is what identifies a boot; it is used here.
        from sglang.srt.weg2 import launcher

        mine = ms.VramCredit("GPU-mine", credit_dir=self.dir)
        mine.begin_leg(ms.credit_epoch("1757300000", 3))
        theirs = ms.VramCredit("GPU-theirs", credit_dir=self.dir)
        theirs.begin_leg(ms.credit_epoch("1757399999", 3))

        removed = launcher.remove_vram_credit_counters(
            self.dir, boot_nonce="1757300000")
        self.assertEqual(len(removed), 1, removed)
        self.assertFalse(os.path.exists(mine.path))
        self.assertTrue(os.path.exists(theirs.path),
                        "another boot's counter must survive this boot's teardown")
        # And an UNKNOWN nonce removes nothing at all: "every counter on the
        # box" is the defect, never the fallback for not knowing which boot.
        self.assertEqual(launcher.remove_vram_credit_counters(self.dir), [])
        self.assertTrue(os.path.exists(theirs.path))

    def test_launch_sweeps_DEAD_epoch_counters_and_never_a_live_holder(self):
        # FIX 3 round 3.  Teardown is exactly what a CRASHED boot does not run,
        # which is how three weg2rg2 counters were still on this rig when
        # weg2rg3 launched.  The epoch makes them harmless; the launch sweep
        # makes them absent -- but a counter whose publisher is ALIVE belongs to
        # a boot in flight and is named, not removed (presence_sweep's rule).
        from sglang.srt.weg2 import launcher

        dead = ms.VramCredit("GPU-dead", credit_dir=self.dir)
        dead.begin_leg(ms.credit_epoch("1757300000", 1), pid=999999)
        live = ms.VramCredit("GPU-live", credit_dir=self.dir)
        live.begin_leg(ms.credit_epoch("1757399999", 1), pid=os.getpid())

        lines = []
        removed = launcher.sweep_dead_credit_counters(lines.append, self.dir)
        self.assertEqual(len(removed), 1, removed)
        self.assertFalse(os.path.exists(dead.path))
        self.assertTrue(os.path.exists(live.path),
                        "a counter whose publisher is alive is a boot in flight")
        self.assertTrue(any("LIVE publisher" in ln for ln in lines), lines)
        # DRY-RUN removes nothing and says what it would have taken.
        dead2 = ms.VramCredit("GPU-dead2", credit_dir=self.dir)
        dead2.begin_leg(ms.credit_epoch("1757300000", 2), pid=999999)
        lines = []
        self.assertEqual(
            launcher.sweep_dead_credit_counters(lines.append, self.dir, dry=True),
            [])
        self.assertTrue(os.path.exists(dead2.path))
        self.assertTrue(any("DRY-RUN" in ln for ln in lines), lines)

    def test_a_stale_full_credit_does_not_licence_a_resume(self):
        self.credit.begin_leg(7)
        self.credit.publish("weights_0", 10 * 1024 * MIB)
        with self.assertRaises(ms.Weg2VramCreditRefused):
            self.credit.wait_for(MIB, budget_s=0.3, tag="weights_0",
                                 free_bytes_now=0, epoch=8)
        # ... and THIS flip's credit does
        self.credit.begin_leg(8)
        self.credit.publish("weights_0", 4 * MIB)
        rec = self.credit.wait_for(MIB, budget_s=1.0, tag="weights_0",
                                   free_bytes_now=0, epoch=8)
        self.assertIn("funded", rec["reason"])

    def test_the_flip_epoch_rides_on_both_legs_of_the_gathered_pair(self):
        # The epoch has to be the FLIP's, and the front is the only thing that
        # owns one.  Asserted on the source so the two RPCs cannot drift apart.
        import inspect

        from sglang.srt.weg2 import front

        src = inspect.getsource(front.Front.flip)
        self.assertIn('"/release_memory_occupation",\n                           '
                      '{"tags": family, "epoch": flip_epoch}', src)
        self.assertIn('"/resume_memory_occupation",\n                           '
                      '{"tags": family, "epoch": flip_epoch}', src)
        # FIX 2 round 2: and the token both legs carry names the BOOT as well as
        # the flip, because the counter file outlives the boot.
        self.assertIn("flip_epoch = credit_epoch(self.boot_epoch, self.epoch)", src)

    def test_a_leg_without_a_flip_epoch_arms_no_credit_rather_than_reading_one(self):
        # A counter that cannot be dated cannot be told from the previous
        # flip's, so the honest answer is to consult none -- not to read it and
        # hope.  The methods are called unbound on a stub: the manager is a
        # slots dataclass and the only state these two touch is these two hooks.
        from sglang.srt.managers.scheduler_components import weight_updater as wu

        class _Rank:
            def _weg2_fence_is_armed(self):
                return True

            def _weg2_card_uuid(self):
                return "GPU-c14"

        rank = _Rank()
        self.assertIsNone(
            wu.SchedulerWeightUpdaterManager._weg2_open_credit_for_leg(rank, None))
        self.assertEqual(
            wu.SchedulerWeightUpdaterManager._weg2_credit_reader(rank, None),
            (None, None))

    def test_the_census_population_is_the_savers_own_backup_metadata(self):
        # FIX 1 round 1, finding 2.  ``tag_bytes`` cannot answer "which tags
        # hold HOST bytes" -- it counts a tag's device bytes whether or not they
        # are ever copied out, so a census built from it over a tag list would
        # charge kv_cache (paused WITHOUT cpu backup, R20) into the ring.  The
        # population is the saver's metadata, and where the running hook cannot
        # answer, the line says weights-family and the planner reads a BOUND.
        from sglang.srt.managers.scheduler_components import weight_updater as wu

        census = wu.SchedulerWeightUpdaterManager._weg2_backup_census

        class _Adapter:
            def __init__(self, answer):
                self.answer = answer

            def backed_up_tag_bytes(self):
                return self.answer

        class _Rank:
            def __init__(self, adapter):
                self.memory_saver_adapter = adapter

        rows, population = census(
            _Rank(_Adapter({"weights_0": 100, "cuda_graph": 25})),
            ["weights_0"], {"weights_0": 100})
        self.assertEqual(rows, {"weights_0": 100, "cuda_graph": 25},
                         "the cuda_graph pause is inside the same RPC and its "
                         "backup is part of the dormant image")
        self.assertEqual(population, "all-backed-up-tags")

        for answer in (None, {}):
            rows, population = census(
                _Rank(_Adapter(answer)), ["weights_0"], {"weights_0": 100})
            self.assertEqual(rows, {"weights_0": 100})
            self.assertEqual(population, "weights-family",
                             "an absence must degrade the CLAIM, not be hidden")

    def test_the_ring_acquire_budget_is_strictly_inside_the_lock_timeout(self):
        # Review nb3, PREDICTED then OBSERVED: on boot weg2rg2 both 120 s
        # budgets expired in the SAME SECOND for one fault, so which named
        # refusal reached the operator was a race -- and the one that carries
        # the arithmetic is the ring's.  The acquire runs INSIDE the PCIe lock,
        # so the inner wait must expire first.
        import re

        header = open(os.path.join(
            os.path.dirname(ms.__file__), "..", "weg2", "tms_csrc", "host_ring.h"
        )).read()
        m = re.search(r"TMS_RING_ACQUIRE_BUDGET_S\s*=\s*([\d.]+)", header)
        self.assertIsNotNone(m, "the budget must stay a named constant")
        self.assertLess(float(m.group(1)), ms.DEFAULT_PCIE_LOCK_TIMEOUT_S,
                        "equal budgets make the refusal a race (nb3, observed)")

    def test_a_funded_wait_returns_and_names_why(self):
        self.credit.begin_leg(1)
        self.credit.publish("weights_0", 100 * MIB)
        rec = self.credit.wait_for(80 * MIB, budget_s=1.0, tag="weights_0",
                                   free_bytes_now=0)
        self.assertIn("funded", rec["reason"])

    def test_a_card_that_is_not_short_waits_for_nobody(self):
        # This is what makes a boot with no co-located peer cost nothing: with
        # the bytes already free there is no shortfall for a peer to fund, and
        # the terminating predicate is irrelevant.
        rec = self.credit.wait_for(80 * MIB, budget_s=1.0, tag="weights_0",
                                   free_bytes_now=100 * MIB)
        self.assertEqual(rec["waited_s"], 0.0)
        self.assertIn("already holds the bytes", rec["reason"])

    def test_the_peers_leg_completing_short_is_a_named_refusal_not_a_longer_wait(self):
        self.credit.begin_leg(1)
        self.credit.publish("weights_0", 10 * MIB)
        self.credit.leg_complete()
        t0 = time.perf_counter()
        with self.assertRaises(ms.Weg2VramCreditRefused) as cm:
            self.credit.wait_for(80 * MIB, budget_s=30.0, tag="weights_0",
                                 free_bytes_now=0)
        self.assertLess(time.perf_counter() - t0, 5.0,
                        "the predicate is the peer's leg, not the budget")
        msg = str(cm.exception)
        self.assertIn("W35 Weg2VramCreditRefused", msg)
        self.assertIn("W30", msg, "spec section 6's name must be findable too")
        self.assertIn("peer_leg_complete=True", msg)
        self.assertIn("card=GPU-c14", msg)
        self.assertIn("tag=weights_0", msg)

    def test_an_expired_budget_is_the_same_named_refusal_with_the_other_reason(self):
        self.credit.begin_leg(1)
        with self.assertRaises(ms.Weg2VramCreditRefused) as cm:
            self.credit.wait_for(80 * MIB, budget_s=0.05, tag="weights_1",
                                 free_bytes_now=0)
        self.assertIn("EXPIRED", str(cm.exception))
        self.assertIn("peer_leg_complete=False", str(cm.exception))

    def test_a_torn_counter_reads_as_no_credit_never_as_credit(self):
        with open(self.credit.path, "w") as f:
            f.write("{not json")
        self.assertEqual(self.credit.read(), {})
        with self.assertRaises(ms.Weg2VramCreditRefused):
            self.credit.wait_for(1 * MIB, budget_s=0.05, tag="t", free_bytes_now=0)

    def test_the_credit_and_the_pcie_lock_name_the_same_card(self):
        self.assertIn("GPU-c14", os.path.basename(self.credit.path))
        self.assertNotEqual(
            os.path.basename(self.credit.path),
            os.path.basename(ms.pcie_lock_path("GPU-c14", lock_dir=self.dir)),
            "three mechanisms, three prefixes: a path in a log is never guessed at",
        )

    def test_no_new_timeout_constant_lives_in_this_module(self):
        # Spec section 10.9.  The bound is the CALLER's, passed in; a default
        # here would be the second timeout constant the spec forbids.
        import inspect

        sig = inspect.signature(ms.VramCredit.wait_for)
        self.assertIs(sig.parameters["budget_s"].default, inspect.Parameter.empty)


# ---------------------------------------------------------------------------
# T7 -- the group STOP, on a real three-rank gloo group
# ---------------------------------------------------------------------------


def _fence_worker(rank: int, world: int, init_file: str, failing_rank: int, out):
    """One rank of the T7 group.  Module level so ``spawn`` can import it."""
    try:
        import torch
        import torch.distributed as dist
        from types import SimpleNamespace

        from sglang.srt.managers.scheduler_components.weight_updater import (
            SchedulerWeightUpdaterManager,
        )

        dist.init_process_group(
            backend="gloo", init_method=f"file://{init_file}",
            rank=rank, world_size=world,
        )
        group = dist.new_group(backend="gloo")
        fake = SimpleNamespace(
            scheduler=SimpleNamespace(world_group=SimpleNamespace(cpu_group=group)),
            _weg2_card_uuid=lambda: f"GPU-{rank}",
        )
        # Bind BOTH halves: the public fence is the re-entry marker shim and it
        # calls the implementation, so a fake that only carries one of them
        # tests a different function than the tree runs.
        fake._weg2_group_fence_impl = (
            SchedulerWeightUpdaterManager._weg2_group_fence_impl.__get__(
                fake, SchedulerWeightUpdaterManager
            )
        )
        fence = SchedulerWeightUpdaterManager._weg2_group_fence.__get__(
            fake, SchedulerWeightUpdaterManager
        )
        t0 = time.perf_counter()
        try:
            report = fence(
                "release tags=['weights_0']",
                ok=(rank != failing_rank),
                failure="" if rank != failing_rank else "boom on this rank",
                per_tag={"weights_0": [1024.0, 10.0 + rank]},
                leg_ms=100.0 + rank,
            )
            out.put((rank, "ok", time.perf_counter() - t0, report))
        except Exception as exc:  # noqa: BLE001
            out.put((rank, type(exc).__name__ + ": " + str(exc),
                     time.perf_counter() - t0, None))
        finally:
            dist.destroy_process_group()
        _ = torch
    except Exception as exc:  # noqa: BLE001
        out.put((rank, "SETUP " + type(exc).__name__ + ": " + str(exc), 0.0, None))


def _run_fence(failing_rank: int, world: int = 3):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    d = tempfile.mkdtemp(prefix="weg2fence-")
    init_file = os.path.join(d, "store")
    procs = [
        ctx.Process(target=_fence_worker, args=(r, world, init_file, failing_rank, q))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    results = []
    try:
        for _ in range(world):
            results.append(q.get(timeout=180))
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        shutil.rmtree(d, ignore_errors=True)
    return sorted(results)


@unittest.skipUnless(os.environ.get("WEG2_T7", "1") == "1", "T7 disabled")
class RankDisagreementTest(unittest.TestCase):
    """T7 / C15: one rank's False stops all three, inside the fence, fast."""

    def test_one_rank_voting_false_raises_W29_on_every_rank(self):
        results = _run_fence(failing_rank=1)
        self.assertEqual(len(results), 3, results)
        for rank, verdict, elapsed, _ in results:
            self.assertIn("Weg2FlipRankDisagree", verdict,
                          f"rank {rank} did not stop: {verdict}")
            self.assertIn("W29 Weg2FlipRankDisagree", verdict)
            self.assertIn("rank 1", verdict, "the failing rank must be NAMED")
            self.assertIn("boom on this rank", verdict)
            self.assertLess(elapsed, 60.0,
                            "the ok-bit sits after the bounded barrier, so a "
                            "disagreement can never cost the 120 s budget")

    def test_all_ranks_ok_returns_the_reduction_the_answer_is_built_from(self):
        results = _run_fence(failing_rank=-1)
        self.assertEqual(len(results), 3, results)
        for rank, verdict, _, report in results:
            self.assertEqual(verdict, "ok", f"rank {rank}: {verdict}")
            self.assertIn("weights_0", report["per_tag"])
            # per tag, the MAXIMUM over ranks: the leg is done when its slowest
            # holder is done.  Ranks contributed 10, 11, 12 ms.
            self.assertEqual(report["per_tag"]["weights_0"][1], 12.0)
            # the critical path is the slowest rank, named with its card
            self.assertIn("rank=2", report["critical_path"])
            self.assertIn("card=GPU-2", report["critical_path"])


# ---------------------------------------------------------------------------
# the refusal FORM -- every W-code leaves as the named line and exit 2
# ---------------------------------------------------------------------------


class RefusalFormTest(unittest.TestCase):
    """Boot weg2rg1 saw W34 as a raw traceback with exit 1.  Never again."""

    def _cli_rc(self, exc):
        from sglang.srt.weg2 import launcher

        real = launcher.main
        launcher.main = lambda argv=None: (_ for _ in ()).throw(exc)
        try:
            return launcher.cli([])
        finally:
            launcher.main = real

    def test_every_named_refusal_returns_2_from_cli(self):
        from sglang.srt.weg2 import host_ledger, launcher, ring_table

        for exc in (
            ring_table.Weg2RingCreditRefused("W32 x"),
            ring_table.Weg2RingNeedsInterleave("W34 x"),
            launcher.Weg2RingFormUnproven("W33 x"),
            host_ledger.Weg2HostLedgerRefused("W20 x"),
            launcher.Weg2LaunchRefused("W? x"),
        ):
            self.assertEqual(self._cli_rc(exc), 2, type(exc).__name__)

    def test_the_refusal_prints_the_one_named_line(self):
        import contextlib
        import io

        from sglang.srt.weg2 import ring_table

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self._cli_rc(ring_table.Weg2RingNeedsInterleave("W34 the reason"))
        self.assertEqual(rc, 2)
        self.assertIn("WEG2-LAUNCH REFUSED: W34 the reason", buf.getvalue())

    def test_an_unnamed_exception_is_still_a_crash_and_not_dressed_as_a_refusal(self):
        with self.assertRaises(ValueError):
            self._cli_rc(ValueError("a real bug"))


if __name__ == "__main__":
    unittest.main()
