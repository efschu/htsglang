# SPDX-License-Identifier: Apache-2.0
"""WEG2-ALLOC-OVERHANG (desk/27b-d-alloc-overhang-0929): the 27B D rest split into
its posts, and the P0 torch cache cap (5f33ec18836a, weg2/torch_cache_cap.py)
made to bind on the 27B so the general allocator cache goes to the KV pool.

The measured rest D books since desk/27b-no-reserve-0929 (``D_AWAKE_REST_BOOKED_MIB``
3191 / 2079 / 2067 MiB) is, at the tightest instant of w109290020 on the 5090::

    over 435 (peak_allocated 27955 - budget 27520) + cache 1325 (peak_reserved
    29280 - 27955: private_free 428 + general 897) + other 1431 (non-torch + P growth)

P0 exists but never bound on the 27B: its cap is the D verdict's ``verfuegbar``
minus the corridor floor, and the 27B verdict books no foreign/non-torch term
(``NICHT GEBUCHT: --d-foreign-context-mib, --d-nontorch-mib``), so the cap is the
NVML total minus 819 -- above what torch can ever reach on the card. Here the cap
stands on the physical line from the records, and the budget books the CAPPED
rest (other + over + kept cache): the general cache is torch's to release.

Fixture lines are verbatim from /spinning/docker-acceptance/27b/evidence (boots
w109290020 bb82fbcb68 and w109281851 85386b1df1).
"""

import dataclasses
import os
import types
import unittest.mock as mock

from sglang.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import budget_rest as BR  # noqa: E402
from sglang.srt.weg2 import corridor_budget as CB  # noqa: E402
from sglang.srt.weg2 import form as F  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402
from sglang.srt.weg2 import torch_cache_cap as TCC  # noqa: E402

U5090 = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
U0 = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
U2 = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"

BUDGET_D = (
    "[{ts}] WEG2-LAUNCH budget D group=D ordinal=0 nvml_idx=1 NVIDIA GeForce RTX 5090: 27520 MiB = "
    "total 32607 - corridor 2971 (floor 2567 source=MEASURED-D reserve=1800 + awake_overshoot 404) - "
    "dormant_other 1104 - driver_carve 518 (NVML reserved) - measured_awake_overshoot 489 (boot "
    "weg2ls4b1) MiB\n"
    "[{ts}] WEG2-LAUNCH budget D group=D ordinal=1 nvml_idx=0 NVIDIA GeForce RTX 3080: 17384 MiB = "
    "total 20480 - corridor 2504 (floor 2100 source=MEASURED-D reserve=1400 + awake_overshoot 404) - "
    "dormant_other 588 MiB\n"
    "[{ts}] WEG2-LAUNCH budget D group=D ordinal=2 nvml_idx=2 NVIDIA GeForce RTX 3080: 17160 MiB = "
    "total 20480 - corridor 2505 (floor 2101 source=MEASURED-D reserve=1400 + awake_overshoot 404) - "
    "dormant_other 814 MiB\n"
)

_GP_TAIL = (" -- captured = Segmente privater Pools (Graph/Tag), private_free = davon frei und fuer "
            "empty_cache unerreichbar")

#: w109290020: the awake line with the largest card_total - card_free - reserved per rank,
#: the post-capture pool line per rank, and the tightest instant (the chunk of 3971 rows)
D_W109290020 = "\n".join([
    "[2026-09-29 00:23:07 TP0] WEG2-GRAPH-POOL rank=0 phase=post-capture captured_mib=26782 private_free_mib=428 reserved_after_mib=27706 allocated_mib=27133 peak_mib=27220 allocator_cache_mib=573 general_cache_mib=145 card_free_mib=1963 card_total_mib=32088 cap_mib=29669 headroom_mib=2021" + _GP_TAIL,
    "[2026-09-29 00:23:07 TP1] WEG2-GRAPH-POOL rank=1 phase=post-capture captured_mib=16434 private_free_mib=584 reserved_after_mib=17356 allocated_mib=16628 peak_mib=16629 allocator_cache_mib=728 general_cache_mib=143 card_free_mib=1394 card_total_mib=20055 cap_mib=18750 headroom_mib=1536" + _GP_TAIL,
    "[2026-09-29 00:23:07 TP2] WEG2-GRAPH-POOL rank=2 phase=post-capture captured_mib=16050 private_free_mib=584 reserved_after_mib=16972 allocated_mib=16244 peak_mib=16245 allocator_cache_mib=728 general_cache_mib=143 card_free_mib=1552 card_total_mib=20055 cap_mib=18524 headroom_mib=1694" + _GP_TAIL,
    "[2026-09-29 00:26:56 TP0] WEG2-GRAPH-POOL rank=0 phase=extend captured_mib=26782 private_free_mib=428 reserved_after_mib=29280 allocated_mib=27510 peak_mib=27955 allocator_cache_mib=1770 general_cache_mib=1342 card_free_mib=273 card_total_mib=32088 cap_mib=29553 headroom_mib=1170" + _GP_TAIL,
    "[2026-09-29 00:26:56 TP0] WEG2-VRAM-PEAK rank=0 phase=chunk rows=3971 n=1 t0_unix_ms=1790641613876 t_unix_ms=1790641616770 window_ms=2894 peak_allocated_mib=27955 peak_reserved_mib=29280 start_allocated_mib=27163 transient_mib=792 allocated_mib=27510 reserved_mib=29280 card_free_start_mib=1729 card_free_mib=273 card_total_mib=32088 alloc_retries=0 ooms=0 alloc_retries_total=0",
    "[2026-09-29 00:52:45 TP0] WEG2-VRAM-PEAK rank=0 phase=chunk rows=444 n=1 t0_unix_ms=1790643163239 t_unix_ms=1790643165465 window_ms=2226 peak_allocated_mib=27716 peak_reserved_mib=28414 start_allocated_mib=27222 transient_mib=494 allocated_mib=27478 reserved_mib=28172 card_free_start_mib=1545 card_free_mib=1275 card_total_mib=32088 alloc_retries=0 ooms=0 alloc_retries_total=0",
    "[2026-09-29 00:42:51 TP1] WEG2-VRAM-PEAK rank=1 phase=chunk rows=152 n=1 t0_unix_ms=1790642570167 t_unix_ms=1790642571723 window_ms=1557 peak_allocated_mib=17054 peak_reserved_mib=17930 start_allocated_mib=16712 transient_mib=341 allocated_mib=16800 reserved_mib=17930 card_free_start_mib=1068 card_free_mib=684 card_total_mib=20055 alloc_retries=0 ooms=0 alloc_retries_total=3",
    "[2026-09-29 00:51:52 TP2] WEG2-VRAM-PEAK rank=2 phase=chunk rows=308 n=1 t0_unix_ms=1790643109564 t_unix_ms=1790643112302 window_ms=2738 peak_allocated_mib=16688 peak_reserved_mib=17588 start_allocated_mib=16309 transient_mib=378 allocated_mib=16443 reserved_mib=17588 card_free_start_mib=1322 card_free_mib=860 card_total_mib=20055 alloc_retries=0 ooms=0 alloc_retries_total=1",
    # a flip leg while D wakes: not an awake (chunk/round) line -- never read as a post
    "[2026-09-29 00:26:53 TP0] WEG2-VRAM-PEAK rank=0 phase=flip leg=resume rpc_ms=1540 rpc=ok n=0 t0_unix_ms=1 t_unix_ms=2 window_ms=1540 peak_allocated_mib=27163 peak_reserved_mib=27758 start_allocated_mib=27163 transient_mib=0 allocated_mib=27163 reserved_mib=27758 card_free_start_mib=11487 card_free_mib=12825 card_total_mib=32088 alloc_retries=0 ooms=0 alloc_retries_total=0",
])
#: w109281851: the awake line with the largest peak_allocated per rank + post-capture
D_W109281851 = "\n".join([
    "[2026-09-28 18:53:32 TP0] WEG2-GRAPH-POOL rank=0 phase=post-capture captured_mib=26782 private_free_mib=428 reserved_after_mib=27708 allocated_mib=27134 peak_mib=27220 allocator_cache_mib=574 general_cache_mib=146 card_free_mib=1961 card_total_mib=32088 cap_mib=29669 headroom_mib=2021" + _GP_TAIL,
    "[2026-09-28 18:53:32 TP1] WEG2-GRAPH-POOL rank=1 phase=post-capture captured_mib=16434 private_free_mib=584 reserved_after_mib=17358 allocated_mib=16630 peak_mib=16632 allocator_cache_mib=728 general_cache_mib=144 card_free_mib=1392 card_total_mib=20055 cap_mib=18750 headroom_mib=1534" + _GP_TAIL,
    "[2026-09-28 18:53:32 TP2] WEG2-GRAPH-POOL rank=2 phase=post-capture captured_mib=16050 private_free_mib=584 reserved_after_mib=16974 allocated_mib=16246 peak_mib=16248 allocator_cache_mib=728 general_cache_mib=144 card_free_mib=1550 card_total_mib=20055 cap_mib=18524 headroom_mib=1692" + _GP_TAIL,
    "[2026-09-28 18:58:11 TP0] WEG2-VRAM-PEAK rank=0 phase=chunk rows=3992 n=1 t0_unix_ms=1790621888744 t_unix_ms=1790621891446 window_ms=2702 peak_allocated_mib=28084 peak_reserved_mib=29098 start_allocated_mib=27628 transient_mib=456 allocated_mib=27635 reserved_mib=29098 card_free_start_mib=411 card_free_mib=375 card_total_mib=32088 alloc_retries=0 ooms=0 alloc_retries_total=0",
    "[2026-09-28 18:58:11 TP1] WEG2-VRAM-PEAK rank=1 phase=chunk rows=3992 n=1 t0_unix_ms=1790621888744 t_unix_ms=1790621891661 window_ms=2917 peak_allocated_mib=17273 peak_reserved_mib=18578 start_allocated_mib=16960 transient_mib=313 allocated_mib=16967 reserved_mib=18276 card_free_start_mib=130 card_free_mib=392 card_total_mib=20055 alloc_retries=1 ooms=0 alloc_retries_total=2",
    "[2026-09-28 18:58:11 TP2] WEG2-VRAM-PEAK rank=2 phase=chunk rows=3992 n=1 t0_unix_ms=1790621888745 t_unix_ms=1790621891447 window_ms=2702 peak_allocated_mib=16889 peak_reserved_mib=18250 start_allocated_mib=16576 transient_mib=313 allocated_mib=16583 reserved_mib=18250 card_free_start_mib=414 card_free_mib=218 card_total_mib=20055 alloc_retries=0 ooms=0 alloc_retries_total=1",
])

BOOTS = (
    ("dkr27browauthoritybar1w109290020", BUDGET_D.format(ts="2026-09-29T00:22:14Z"), D_W109290020),
    ("dkr27browauthoritybar1w109281851", BUDGET_D.format(ts="2026-09-28T18:52:33Z"), D_W109281851),
)

DORMANT_P = {U5090: 1104, U0: 588, U2: 814}
RESERVE = {U5090: 1800, U0: 1400, U2: 1400}
#: desk/27b-no-reserve-0929 on the w109290020 real D pass (its own test's number)
UNCAPPED_D = [27792, 17384, 17168]
#: w109290020 per-rank profiled KV capacity at budgets 27520/17384/17160 (D.log
#: 'Uneven DCP ... per-rank profiled capacity'), cell_size 32768 on every rank
W1_BUDGETS = [27520, 17384, 17160]
W1_CAPACITY = [272994, 252097, 244929]
TOKENS_PER_MIB = (1 << 20) // 32768


def _cards():
    return L.order_cards([
        L.Card(nvml_index=0, uuid=U0, name="NVIDIA GeForce RTX 3080", total_mib=20480, reserved_mib=425),
        L.Card(nvml_index=1, uuid=U5090, name="NVIDIA GeForce RTX 5090", total_mib=32607, reserved_mib=518),
        L.Card(nvml_index=2, uuid=U2, name="NVIDIA GeForce RTX 3080", total_mib=20480, reserved_mib=425),
    ])


def _d_pass(profile, lines, capped):
    cards = _cards()
    over, over_prov = L.d_overshoot_record(profile)
    kw = dict(L.booked_rest_kwargs(cards, profile, "D", lines.append, capped=capped))
    return cards, L.budgets_from_dc(
        cards, dict(DORMANT_P), lines.append, "D", overshoot_mib=over, overshoot_provenance=over_prov,
        user_reserve_by_card=dict(RESERVE), charge_driver_carve=L.budget_charges_driver_carve(profile),
        driver_carve_min_total_mib=L.driver_carve_min_total_mib(profile), **kw)


def _ns(profile, env_d=""):
    return types.SimpleNamespace(profile=profile, env_d=env_d)


class TheRestSplitsIntoItsPosts(CustomTestCase):
    def test_the_tightest_instant_is_over_plus_cache_plus_other(self):
        """5090, w109290020 chunk: 3191 = 435 + 1325 + 1431 (verbatim line)."""
        rest = BR.boot_rests(BOOTS[0][1], BOOTS[0][2], "D")[0]
        self.assertEqual(rest.rest, 3191)
        over, cache, other = 27955 - 27520, 29280 - 27955, 32088 - 273 - 29280 - 1104
        self.assertEqual((over, cache, other), (435, 1325, 1431))
        self.assertEqual(over + cache + other, rest.rest)
        # the cache: private_free 428 (no empty_cache reaches it) + general 897
        self.assertEqual(cache - 428, 897)

    def test_the_records_are_that_measurement(self):
        others, capped, lines = BR.capped_record_from_boots(BOOTS, "D", 3)
        self.assertEqual(others, [1537, 853, 793])
        # other + over (w109281851: 28084-27520, 17273-17384, 16889-17160) + keep
        self.assertEqual(capped, [1537 + 564 + 574, 853 - 111 + 728, 793 - 271 + 728])
        row = F.PROFILES[F.PROFILE_QWEN27B]
        self.assertEqual(list(row.constants[BR.other_record_name("D")].value), others)
        self.assertEqual(list(row.constants[BR.capped_record_name("D")].value), capped)
        self.assertTrue(any("capped rest 2675 = other 1537 + over 564 + keep 574" in ln for ln in lines))

    def test_flip_legs_are_not_awake_posts(self):
        posts = BR.awake_posts(D_W109290020)
        self.assertEqual(posts[0].other_total, 32088 - 1275 - 28172)  # not the flip leg's
        self.assertEqual(posts[0].keep, 428 + 145)  # the FIRST post-capture line, not extend

    def test_a_missing_post_is_unmeasured_never_zero(self):
        _, capped, lines = BR.capped_record_from_boots(BOOTS[:1], "D", 4)
        self.assertIsNone(capped[3])
        self.assertTrue(any("ordinal=3" in ln and "UNMEASURED" in ln for ln in lines))


class P0NeverBoundOnThe27B(CustomTestCase):
    def test_the_verdict_cap_sits_above_what_torch_can_reach(self):
        """Why the switch did nothing for the 27B: its verdict books no foreign /
        non-torch term, so verfuegbar = NVML total and the cap = total - 819."""
        from sglang.srt.planner import pp_cut as PC

        vs = PC.d_rank_budget_verdict(
            budgets_mib=[27792.0, 17384.0, 17168.0], card_total_mib=[32607.0, 20480.0, 20480.0],
            foreign_context_mib=[0.0] * 3, nontorch_mib=[0.0] * 3, reserve_mib_by_rank=None,
            corridor_floor_mib=819.0)
        caps = [int(x) for x in TCC.launcher_caps(vs, 819.0).split(",")]
        self.assertEqual(caps, [31788, 19661, 19661])
        # torch's physical reach at w109290020's tightest instant: reserved + card_free
        reach = [29280 + 273, 18548 + 118, 18350 + 116]
        self.assertTrue(all(c > r for c, r in zip(caps, reach)))

    def test_the_record_cap_stands_on_the_physical_line(self):
        cards, budgets = _d_pass(F.PROFILE_QWEN27B, [], capped=True)
        caps = BR.torch_caps(budgets, [2675, 1470, 1250], [1537, 853, 793])
        for c, cap, dc in zip(cards, caps, (1104, 588, 814)):
            # total - carve - dormant_other - OTHER, down to the budget's 8-MiB grain
            line = c.total_mib - c.reserved_mib - dc - {U5090: 1537, U0: 853, U2: 793}[c.uuid]
            self.assertLessEqual(cap, line)
            self.assertGreater(cap, line - 8)
        # budget + over + keep: torch keeps its measured overhang and kept cache
        self.assertEqual(caps, [b + o + k for b, o, k in zip(budgets, (564, -111, -271), (574, 728, 728))])


class TheCappedRestGoesToTheKvPool(CustomTestCase):
    def test_budget_rises_by_the_released_cache(self):
        lines = []
        _, capped = _d_pass(F.PROFILE_QWEN27B, lines, capped=True)
        _, uncapped = _d_pass(F.PROFILE_QWEN27B, [], capped=False)
        self.assertEqual(uncapped, UNCAPPED_D)  # the no-reserve pass, unchanged
        self.assertEqual(capped, [28304, 17992, 17984])
        self.assertEqual([a - b for a, b in zip(capped, uncapped)], [512, 608, 816])
        blines = [ln for ln in lines if ln.startswith("budget D")]
        self.assertTrue(blines and all("D_AWAKE_REST_CAPPED_MIB" in ln for ln in blines))

    def test_the_world_kv_pool_grows(self):
        """THE KV-GESAMTTEST: KV sizing is budget - used_by_me on this path
        (model_runner_kv_cache_mixin, absolute-budget branch), so every budget MiB
        is 32 tokens of the rank's capacity; the runtime's own vector solver
        (partition_units, corridor_budget._resolved_world_pool) re-solves the
        ownership. It reproduces w109290020's installed 759616 exactly."""
        def caps(b):
            return [c + (x - y) * TOKENS_PER_MIB for c, x, y in zip(W1_CAPACITY, b, W1_BUDGETS)]

        self.assertEqual(CB._resolved_world_pool(caps(W1_BUDGETS)), (759616, None))
        _, uncapped = _d_pass(F.PROFILE_QWEN27B, [], capped=False)
        _, capped = _d_pass(F.PROFILE_QWEN27B, [], capped=True)
        before, _ = CB._resolved_world_pool(caps(uncapped))
        after, _ = CB._resolved_world_pool(caps(capped))
        self.assertEqual((before, after), (768256, 827584))
        self.assertEqual(after - before, 59328)


class TheSwitchIsTheRegistryRow(CustomTestCase):
    def test_qwen27b_row_is_off_until_the_cell(self):
        self.assertFalse(F.PROFILES[F.PROFILE_QWEN27B].torch_cache_cap)
        ns = _ns(F.PROFILE_QWEN27B)
        self.assertIsNone(L.apply_profile_torch_cache_cap_default(ns))
        self.assertEqual(ns.env_d, "")
        self.assertFalse(L.torch_cache_cap_armed(ns))

    def test_row_on_writes_the_switch_into_env_d(self):
        row = F.PROFILES[F.PROFILE_QWEN27B]
        with mock.patch.dict(F.PROFILES, {F.PROFILE_QWEN27B: dataclasses.replace(row, torch_cache_cap=True)}):
            ns = _ns(F.PROFILE_QWEN27B, "SGLANG_WEG2_EXTEND_TRIM_MIB=1200,0,0")
            line = L.apply_profile_torch_cache_cap_default(ns)
            self.assertIn(BR.OVERHANG_MARKER, line)
            self.assertEqual(ns.env_d, "SGLANG_WEG2_EXTEND_TRIM_MIB=1200,0,0;SGLANG_WEG2_TORCH_CACHE_CAP=1")
            self.assertTrue(L.torch_cache_cap_armed(ns))
            off = _ns(F.PROFILE_QWEN27B, "SGLANG_WEG2_TORCH_CACHE_CAP=0")
            self.assertIsNone(L.apply_profile_torch_cache_cap_default(off))
            self.assertFalse(L.torch_cache_cap_armed(off))

    def test_the_cell_arms_it_through_env_d(self):
        """27b-row-authority-p0 (the cell's profile) names it in --env-d."""
        ns = _ns(F.PROFILE_QWEN27B, "SGLANG_WEG2_EXTEND_TRIM_MIB=1200,0,0;SGLANG_WEG2_TORCH_CACHE_CAP=1")
        self.assertTrue(L.torch_cache_cap_armed(ns))
        cards = _cards()
        _, budgets = _d_pass(F.PROFILE_QWEN27B, [], capped=L.torch_cache_cap_armed(ns))
        lines = []
        caps = L.d_record_torch_caps(ns, cards, budgets, lines.append, "D")
        self.assertEqual(caps, [28304 + 564 + 574, 17992 - 111 + 728, 17984 - 271 + 728])
        self.assertTrue(any("rang0 28304 + 2675 - 1537 = 29442" in ln for ln in lines))
        # the rank side: fraction of torch's own total (card_total_mib 32088 on the 5090)
        with mock.patch.dict(os.environ, {TCC.ENV: "1", TCC.MIB_ENV: ",".join(map(str, caps)),
                                          "SGLANG_WEG2_GROUP": "D"}):
            calls = []
            cuda = types.SimpleNamespace(
                mem_get_info=lambda i: (0, 32088 << 20),
                set_per_process_memory_fraction=lambda f, i: calls.append((f, i)))
            frac = TCC.arm(0, 0, 1, torch_mod=types.SimpleNamespace(cuda=cuda))
        self.assertAlmostEqual(frac, 29442 / 32088, places=6)
        self.assertEqual(calls, [(frac, 1)])

    def test_d_only_and_unarmed_keep_the_verdict_cap(self):
        ns = _ns(F.PROFILE_QWEN27B, "SGLANG_WEG2_TORCH_CACHE_CAP=1")
        self.assertIsNone(L.d_record_torch_caps(ns, _cards(), UNCAPPED_D, [].append, L.D_ONLY_LABEL))
        # unarmed (rest = the uncapped record): the cap is still the physical line
        caps = L.d_record_torch_caps(_ns(F.PROFILE_QWEN27B), _cards(), UNCAPPED_D, [].append, "D")
        self.assertEqual(caps, [27792 + 3191 - 1537, 17384 + 2079 - 853, 17168 + 2067 - 793])


class NextFlashStaysByteIdentical(CustomTestCase):
    def test_the_nf_row_is_off_and_carries_no_record(self):
        nf = F.PROFILES[F.PROFILE_NEXTFLASH]
        self.assertFalse(nf.torch_cache_cap)
        for name in (BR.other_record_name("D"), BR.capped_record_name("D")):
            self.assertNotIn(name, nf.constants)
        self.assertIsNone(L.apply_profile_torch_cache_cap_default(_ns(F.PROFILE_NEXTFLASH)))

    def test_nf_budget_and_cap_paths_are_untouched(self):
        a, b = [], []
        _, capped = _d_pass(F.PROFILE_NEXTFLASH, a, capped=True)
        _, plain = _d_pass(F.PROFILE_NEXTFLASH, b, capped=False)
        self.assertEqual(capped, plain)
        self.assertEqual(a, b)
        ns = _ns(F.PROFILE_NEXTFLASH, "SGLANG_WEG2_TORCH_CACHE_CAP=1")
        self.assertIsNone(L.d_record_torch_caps(ns, _cards(), plain, [].append, "D"))


if __name__ == "__main__":
    import unittest

    unittest.main()
