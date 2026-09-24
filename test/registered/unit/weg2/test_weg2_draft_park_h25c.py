"""H25c: D's draft sleeps in pinned system RAM while P runs.

Pinned, CPU-only with fakes:

1. ORDER at the sleep: every byte copied BEFORE the pause (the pause unmaps
   the pages the copy reads -- the campaign (a) fault class); at the wake the
   resume BEFORE the copy (writing into a paused VA is the x33/x34 SIGSEGV).
2. POPULATION: storages the draft shares with its TARGET (H1b: embed/head are
   the target's tensors, region ``weights``) are never parked or written back;
   views are one storage; META tensors (solo-shadow drafts on TP1/TP2) hold no
   bytes.
3. ONE host image per process: the second park reuses it.
4. LEDGER: the post ``d_draft_host`` is charged at both moments, and W128
   refuses a D draft the ledger cannot price.
5. WIRING: park at the first weights RPC before the family loop, unpark behind
   the legs, join before the wake's fence.
"""

import inspect
import unittest
from types import SimpleNamespace

import pytest
import torch

try:
    from sglang.srt.weg2 import draft_park as dpk
    from sglang.srt.weg2 import host_ledger
    from sglang.srt.weg2 import launcher as L
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

H25_OWN_DRAFT_FORM = True


class _Draft(torch.nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.w = torch.nn.Parameter(torch.arange(16, dtype=torch.float32), requires_grad=False)
        self.register_buffer("cos_sin", torch.full((8,), 3.0))
        self.embed = shared                  # the TARGET's tensor (H1b share)
        self.w_view = self.w.data[4:8]       # a view: same storage as w
        self.meta = torch.empty(4, device="meta")


class TestPopulation(unittest.TestCase):
    def test_target_shared_views_and_meta_are_not_parked(self):
        target = torch.nn.Module()
        target.embed = torch.nn.Parameter(torch.ones(32), requires_grad=False)
        pop = dpk.park_population(_Draft(target.embed), target)
        names = sorted(n for n, _v in pop)
        self.assertEqual(names, ["cos_sin", "w"])
        self.assertEqual(sum(int(v.numel()) for _n, v in pop), 16 * 4 + 8 * 4)


class TestOrder(unittest.TestCase):
    def test_copy_then_pause_resume_then_copy_and_bytes_round_trip(self):
        target = torch.nn.Module()
        target.embed = torch.nn.Parameter(torch.ones(32), requires_grad=False)
        draft = _Draft(target.embed)
        events = []
        park = dpk.DraftHostPark(pin=False)
        pop = dpk.park_population(draft, target)

        def pause(tag):
            events.append(("pause", tag, float(park.host.view(torch.float32)[:16].sum())))
            # the saver's unmap: the device bytes are gone
            draft.w.data.zero_()
            draft.cos_sin.zero_()
            target.embed.data.fill_(7.0)   # the exchange rewrites the target meanwhile

        rec = park.park(pop, tag="weights_draft", pause=pause, sync=lambda: events.append(("sync",)))
        # the host image was complete when the pause ran (sum 0..15 = 120)
        self.assertEqual(events[0], ("sync",))
        self.assertEqual(events[1], ("pause", "weights_draft", 120.0))
        self.assertEqual(rec.nbytes, 16 * 4 + 8 * 4)
        self.assertIn("WEG2-DRAFT-PARK tag=weights_draft bytes=96", rec.line())

        def resume(tag):
            events.append(("resume", tag, float(draft.w.sum())))

        park.unpark_start(tag="weights_draft", resume=resume)
        self.assertEqual(events[2], ("resume", "weights_draft", 0.0))  # before the copy
        line = park.join(overlap="scratch+rearm")
        self.assertTrue(torch.equal(draft.w, torch.arange(16, dtype=torch.float32)))
        self.assertTrue(torch.equal(draft.cos_sin, torch.full((8,), 3.0)))
        # the shared target tensor is NOT written back (it is the exchange's)
        self.assertTrue(torch.equal(target.embed, torch.full((32,), 7.0)))
        self.assertIn("WEG2-DRAFT-UNPARK tag=weights_draft bytes=96", line)
        self.assertIn("overlap=scratch+rearm", line)

    def test_second_park_reuses_the_image_and_double_park_refuses(self):
        draft = _Draft(torch.ones(2))
        park = dpk.DraftHostPark(pin=False)
        pop = dpk.park_population(draft, None)
        r1 = park.park(pop, tag="t", pause=lambda t: None, sync=lambda: None)
        host = park.host
        with self.assertRaises(RuntimeError):
            park.park(pop, tag="t", pause=lambda t: None, sync=lambda: None)
        park.unpark_start(tag="t", resume=lambda t: None)
        park.join(overlap="")
        r2 = park.park(dpk.park_population(draft, None), tag="t",
                       pause=lambda t: None, sync=lambda: None)
        self.assertIs(park.host, host)
        self.assertGreaterEqual(r1.alloc_ms, 0.0)
        self.assertEqual(r2.alloc_ms, -1.0)
        self.assertIn("reused", r2.line())


class TestLedger(unittest.TestCase):
    def test_post_is_charged_at_both_moments(self):
        images = host_ledger.ImageTerms(p_gib=0.0, d_gib=0.0, p_source="t", d_source="t",
                                        p_measured=False, d_measured=False,
                                        extra_p_gib=0.0, extra_d_gib=0.0)
        base = host_ledger.charge_terms(1, 600, 3, images)
        withp = host_ledger.charge_terms(1, 600, 3, images, d_draft_host_gib=1587 / 1024.0)
        self.assertEqual(base["d_draft_host_gib"], 0.0)
        self.assertAlmostEqual(host_ledger._boot_charges_gib(withp)
                               - host_ledger._boot_charges_gib(base), 1587 / 1024.0)
        self.assertAlmostEqual(host_ledger._run_moment_charges_gib(withp)
                               - host_ledger._run_moment_charges_gib(base), 1587 / 1024.0)

    def test_choose_and_price_carry_the_keyword(self):
        self.assertIn("d_draft_host_gib", inspect.signature(host_ledger.choose).parameters)
        self.assertIn("d_draft_host_gib", inspect.signature(host_ledger.price).parameters)
        src = inspect.getsource(host_ledger.choose)
        self.assertGreaterEqual(src.count("d_draft_host_gib=d_draft_host_gib"), 2)


class TestParkTermAndW128(unittest.TestCase):
    def _ns(self, extra_d, source="exchange"):
        return SimpleNamespace(weg2_weight_source=source, flip_weights="family",
                               extra_d=extra_d, env_d="")

    def test_unpriceable_placement_refuses_w128(self):
        ns = self._ns("--speculative-algorithm NEXTN --speculative-draft-placement split "
                      "--speculative-draft-model-path /nonexistent")
        with self.assertRaisesRegex(L.Weg2LaunchRefused, "W128"):
            L.d_draft_park_term(ns, False)

    def test_no_term_where_nothing_parks(self):
        ns = self._ns("--speculative-algorithm NEXTN --speculative-draft-placement split")
        self.assertEqual(L.d_draft_park_term(ns, True)[0], 0.0)
        self.assertEqual(L.d_draft_park_term(self._ns("", "ring"), False)[0], 0.0)
        self.assertEqual(L.d_draft_park_term(self._ns("--max-total-tokens 1"), False)[0], 0.0)

    def test_the_term_reaches_the_one_ledger_call(self):
        src = inspect.getsource(L.main)
        self.assertIn("d_draft_host_gib=d_draft_host_mib / 1024.0", src)
        self.assertLess(src.index("d_draft_park_term(ns, draft_on_p)"),
                        src.index("choose_host_ledger("))


class TestWiring(unittest.TestCase):
    """Order on the rank, read off the source (the RPCs need a device)."""

    def setUp(self):
        from sglang.srt.managers.scheduler_components import weight_updater as wu

        cls = wu.SchedulerWeightUpdaterManager
        self.rel = inspect.getsource(cls.release_memory_occupation)
        self.res = inspect.getsource(cls.resume_memory_occupation)

    def test_park_before_the_family_pauses(self):
        i = self.rel.index("self._weg2_park_draft_at_sleep(credit)")
        self.assertLess(i, self.rel.index("self.memory_saver_adapter.pause(tag)"))
        self.assertIn("if not family_paused_before:", self.rel[i - 120:i])

    def test_unpark_behind_the_legs_and_joined_at_the_admission(self):
        s = self.res.index("self._weg2_unpark_draft_start(")
        self.assertLess(self.res.index('_weg2_ph("leg_collects")'), s)
        # H31b: the TARGET is rearmed before the unpark (its closing sync must
        # not wait for the 1.5 GB copy), the DRAFT's own tensors behind it
        self.assertLess(self.res.index("for _m in _early:"), s)
        self.assertLess(s, self.res.index("_scratch += self._weg2_zero_local_scratch(_late)"))
        self.assertLess(s, self.res.index("for _m in _late:"))
        # joined at the last instant before a request can reach the verifier:
        # inside the admission block, BEFORE the DORMANT flag drops
        j = self.res.index('self._weg2_unpark_draft_join(_weg2_ph_l, where="admit")')
        self.assertLess(self.res.index("def _weg2_kv_clear_part():"), j)
        self.assertLess(j, self.res.index("scheduler.weg2_dormant = False"))

    def test_a_pending_unpark_is_joined_before_the_next_park(self):
        from sglang.srt.managers.scheduler_components import weight_updater as wu

        src = inspect.getsource(wu.SchedulerWeightUpdaterManager._weg2_park_draft_at_sleep)
        self.assertLess(src.index('_weg2_unpark_draft_join([], where="sleep")'),
                        src.index(".park("))


if __name__ == "__main__":
    unittest.main()
