# SPDX-License-Identifier: Apache-2.0
"""rc12c (NF Dauerlauf, D TP0 OOM 27.09. 01:22:42Z): the D form never followed its budget.

Three images, three budgets (29624 / 29144 / 28288 MiB), one death size
(D per-process NVML 29.14-29.27 GiB). The runtime sizes only the KV rest from
the budget and clamps it to the 262144-token context in all three boots, so
the budget cut reached no allocation. The record meant to price the excess
(``D_OVERSHOOT_MIB`` = peak - BUDGET) grew by exactly what the budget lost --
a ratchet -- and the #145 solve's budget path ("Rest nach KV 1202 -> PASST",
fixed post from fnFL2x151/x158) disagreed with its own card path ("Decode frei
-1096 -> KORRIDOR GERISSEN").

The fix, tested here:
* ``D_AWAKE_REST_MIB`` / ``D_FIXED_MIB`` measured against the BOOKED FORM
  (weg2/d_awake_rest.py) -- budget-independent, with a fixpoint;
* the D budget books the rest instead of the builtin 404 and the ratchet;
* the #145 card is priced from the budget's own terms (``DCardLedger``) and
  says what the budget says, for every scratch;
* the expert scratch follows the booked budget (``d_scratch_cap``).
The OOM lines below are verbatim from the four D logs
(/spinning/docker-acceptance/nf/evidence/boot_weg2_dkrnfh91bar1dauer0926224{9},
09262302, 09270007, 09270103).
"""

import os

from sglang.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.planner import expert_residency as ER  # noqa: E402
from sglang.srt.weg2 import d_awake_rest as AR  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402

MIB = 1024 * 1024
SLOT = 2.417 * MIB  # "48 Layer x 2.417 MiB/Zeile" (front.log #145 line)
LAYER_ROW = 48 * SLOT / MIB

POSTS = ("[2026-09-27 01:07:46 TP0] [world_rank 0] KV budget posts (GiB): weights + runtime "
         "state={wr:.3f}, mamba state pool=2.082, speculative intermediate state=0.131, "
         "prefill activation reserve=1.000, GGUF dequant scratch=0.000 | rest=4.455 | "
         "measured free=9.284 | unaccounted=+4.829")
SIZING = ("[2026-09-27 01:07:46 TP0] KV pool sizing: available_bytes=4783497216 (4.455 GiB), "
          "cell_size=14143, page_size=64 -> max_total_num_tokens=338176")
TOKENS = ("[2026-09-27 01:08:03 TP0] max_total_num_tokens=262144, chunked_prefill_size=4096, "
          "max_prefill_tokens=16384, max_running_requests=6, context_len=262144, "
          "available_gpu_mem=2.35 GB")
BUFFER = ("[2026-09-27 01:07:41 TP0] MoE expert-offload active on layer 47: 12/193 experts "
          "resident + {s} scratch (buffer={b}, fraction=0.060)")
GATHER = ("[2026-09-27 01:22:13 TP0] MoE offload gather fetch failed (OutOfMemoryError: CUDA out "
          "of memory. {oom}. Of the allocated memory 29.36 GiB is allocated by PyTorch)")
HEAD = "[2026-09-27 01:22:42 TP0] Scheduler hit an exception: Traceback (most recent call last):"
TRACE = "torch.OutOfMemoryError: CUDA out of memory. {oom}. Of the allocated memory ..."


def _oom(ask, free_mib, proc_gib, mine_gib, cap_gib=31.34):
    return (f"Tried to allocate {ask:.2f} MiB. GPU 0 has a total capacity of {cap_gib:.2f} GiB of "
            f"which {free_mib:.2f} MiB is free. Process 438 has {proc_gib:.2f} GiB memory in use. "
            f"Including non-PyTorch memory, this process has {mine_gib:.2f} GiB memory in use")


def _log(ooms, *, wr=19.957, scratch=100, buffer=112, traceback_last=True):
    lines = [BUFFER.format(s=scratch, b=buffer), POSTS.format(wr=wr), SIZING, TOKENS]
    for i, o in enumerate(ooms):
        if traceback_last and i == len(ooms) - 1:
            lines += [HEAD, "  File \"x.py\", line 1, in f", TRACE.format(oom=o)]
        else:
            lines.append(GATHER.format(oom=o))
    return "\n".join(lines)


#: the four Dauerlauf boots: (tag, budget TP0, its OOM lines verbatim)
BOOTS = (
    ("dkrnfh91bar1dauer09262249", 29624, [_oom(158, 132.75, 1.76, 29.14), _oom(80, 50.75, 1.69, 29.27)]),
    ("dkrnfh91bar1dauer09262302", 29624, [_oom(158, 108.75, 1.76, 29.12)]),
    ("dkrnfh91bar1dauer09270007", 29144, [_oom(80, 68.75, 1.76, 29.19)]),
    ("dkrnfh91bar1dauer09270103", 28288, [_oom(158, 134.75, 1.64, 29.22), _oom(144, 108.75, 1.64, 29.24)]),
)


class TheRestIsMeasuredAgainstTheBookedForm(CustomTestCase):
    def test_rc12c_rest_and_fixed_post(self):
        fixed, rest, lines = AR.records([(t, _log(o)) for t, _b, o in BOOTS[3:]], n_ranks=3,
                                        n_layers=48, layer_row_mib=LAYER_ROW)
        # demand 32092 - 108.75 - 1679.4 + 144 = 30448; posts 27262 -> 3186
        self.assertEqual(rest, [3186, None, None], lines)
        self.assertEqual(fixed[0], round(19.957 * 1024 - 112 * LAYER_ROW))
        self.assertIn("unattributed 362", lines[-1])

    def test_the_rest_does_not_follow_the_budget_the_overshoot_did(self):
        rests, overs = [], []
        for tag, budget, ooms in BOOTS:
            b = AR.boot_rest(tag, _log(ooms), n_layers=48)
            v, s = b.rest[0]
            rests.append(v)
            overs.append(s.mine + s.ask - budget)  # the withdrawn D_OVERSHOOT_MIB formula
        self.assertLess(max(rests) - min(rests), 200, rests)       # 3039..3186
        self.assertGreater(max(overs) - min(overs), 1000, overs)   # 353..1798: the ratchet

    def test_the_shipped_record_is_the_logs_maximum(self):
        fixed, rest, _ = AR.records([(t, _log(o)) for t, _b, o in BOOTS], n_ranks=3,
                                    n_layers=48, layer_row_mib=LAYER_ROW)
        self.assertEqual(list(L._pconst("D_AWAKE_REST_MIB", "nextflash")), rest)
        self.assertEqual(list(L._pconst("D_FIXED_MIB", "nextflash"))[0], fixed[0])

    def test_no_edge_no_record(self):
        with self.assertRaises(AR.AwakeRestError):
            AR.records([("x", _log([]))], n_ranks=3, n_layers=48, layer_row_mib=LAYER_ROW)


# --- the rig at rc12c (front.log budget lines + WEG2-DORMANT-SERVED record) ------------

def _cards():
    return L.order_cards([
        L.Card(nvml_index=0, uuid="GPU-5c648f96", name="NVIDIA GeForce RTX 3080",
               total_mib=20480, reserved_mib=425),
        L.Card(nvml_index=1, uuid="GPU-31d7ef41", name="NVIDIA GeForce RTX 5090",
               total_mib=32607, reserved_mib=518),
        L.Card(nvml_index=2, uuid="GPU-62dbbae1", name="NVIDIA GeForce RTX 3080",
               total_mib=20480, reserved_mib=425),
    ])


DORMANT = {"GPU-31d7ef41": 1320, "GPU-5c648f96": 712, "GPU-62dbbae1": 696}
FLOORS = (767, 700, 701)
P_ASLEEP_5090 = 1320 + 482
FRACTIONS = (0.06, 0.51, 0.48)
RATIOS = (183, 137, 168)


def _floors(cards, label, reserve=None):
    class _F:
        def __init__(self, mib):
            self.mib, self.source, self.reserve_mib = mib, "MEASURED-D", 0
            self.line, self.actuates = f"floor {mib}", True
    return {c.uuid: _F(FLOORS[i]) for i, c in enumerate(cards)}


def _d_pass(profile, lines, terms=None, rest=True):
    """The launcher's real D pass with the rc12c terms (floors pinned to the
    rig's MEASURED-D 767/700/701 instead of this desk's fallback)."""
    from unittest import mock

    cards = _cards()
    g, gp = L.served_dormant_growth(cards, profile)
    r, rp = L.d_awake_rest(cards, profile) if rest else (None, "")
    o, op = L.d_overshoot_record(profile)
    with mock.patch.object(L.corridor_budget, "floors_for_cards", _floors):
        b = L.budgets_from_dc(cards, dict(DORMANT), lines.append, "D", overshoot_mib=o,
                              overshoot_provenance=op, dormant_growth_mib=g,
                              dormant_growth_provenance=gp,
                              charge_driver_carve=L.budget_charges_driver_carve(profile),
                              awake_rest_mib=r, awake_rest_provenance=rp, terms_out=terms)
    return cards, b


def _reference(fixed=(7442.0, 1036.0, 914.0)):
    # the rc12c solve's own posts: seat-rebooked mamba 2134.5 / ReplaySSM spec 133.9 on TP0
    return ER.DRankReference(
        source="rc12c", model="Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist",
        rank_tp_ratio="1,0,0", fixed_mib=tuple(fixed), mamba_mib=(2134.5, 0.0, 0.0),
        spec_mib=(133.9, 0.0, 0.0), activation_mib=(1024.0, 1024.0, 1024.0),
        kv_cell_bytes=(14143, 768, 768), draft_host_rank=0, draft_vocab_held=False,
        dense_repack_outside_pool=True, max_running=6)


def _fits(budgets, scratch, fixed=(7442.0, 1036.0, 914.0)):
    return ER.solve_d_rank_residency(
        budgets_mib=[float(b) for b in budgets], fractions=FRACTIONS, ratios=RATIOS,
        scratch_rows=list(scratch), staging_rows=12, num_experts=512, pad_rows=1, n_layers=48,
        slot_bytes=SLOT, reference=_reference(fixed), vocab_mib=2425.0, share_embed=True,
        kv_tokens=262144)


class TheDBudgetBooksTheRestNotTheRatchet(CustomTestCase):
    def test_nf_budget_line(self):
        lines = []
        _cards_, b = _d_pass("nextflash", lines)
        b0 = [l for l in lines if "ordinal=0 " in l][0]
        self.assertIn("awake_rest 3186 (D_AWAKE_REST_MIB", b0)
        self.assertNotIn("awake_overshoot 404", b0)
        self.assertNotIn("measured_awake_overshoot", b0)
        # 32607 - 767 - 1320 - 482 - 518 - 3186 = 26334 -> 26328
        self.assertEqual(b[0], 26328)
        for i in (1, 2):  # no rest measured on the 3080s: the builtin 404 stays
            bi = [l for l in lines if f"ordinal={i} " in l][0]
            self.assertIn("+ awake_overshoot 404)", bi)
        self.assertEqual(b[1], 17664)  # rc12c front.log, unchanged

    def test_27b_budget_is_byte_identical(self):
        lines, plain = [], []
        cards, b = _d_pass("qwen27b", lines, terms=[])
        from unittest import mock

        with mock.patch.object(L.corridor_budget, "floors_for_cards", _floors):
            ref = L.budgets_from_dc(cards, dict(DORMANT), plain.append, "D",
                                    overshoot_mib=[489, 0, 0], overshoot_provenance="boot weg2ls4b1")
        self.assertEqual(b, ref)
        self.assertEqual(lines, plain)
        self.assertEqual(L.d_awake_rest(cards, "qwen27b"), (None, ""))
        self.assertEqual(L.d_fixed_record("qwen27b"), (None, ""))


class TheBudgetAndTheCardSayTheSame(CustomTestCase):
    """Two sides of one seam: for every scratch, the #145 budget path and the
    card path (from the budget's own terms) give the same verdict and edge."""

    def _both(self, scratch):
        terms = []
        _c, budgets = _d_pass("nextflash", [], terms=terms)
        led = L.d_card_ledger(terms, budgets, "D")
        fits = _fits(budgets, scratch)
        cards = ER.solve_d_card_ledger(fits=fits, ledger=led)
        return budgets, fits, cards

    def test_every_scratch_agrees(self):
        for s0 in range(40, 141, 3):
            _b, fits, cards = self._both((s0, 48, 48))
            for f, c in zip(fits, cards):
                self.assertEqual(f.verdict == "PASST", c.headroom_mib >= c.near_oom_mib,
                                 (s0, f.rank, f.verdict, c.headroom_mib))
                self.assertEqual(f.ceiling_max_rows, c.ceiling_max_rows, (s0, f.rank))

    def test_the_old_pair_disagreed_at_rc12c(self):
        # budget path with the stale fixed post and the rc12c budget ...
        old = _fits((28288, 17664, 17864), (100, 48, 48), fixed=(7264.2, 1044.0, 921.9))
        self.assertEqual(old[0].verdict, "PASST")
        # ... card path from fnFL2x151/x158: decode free below the band floor
        card = ER.solve_d_card(fits=old, reference=ER.D_CARD_REFERENCE_FNFL2_H39, vocab_mib=2425.0,
                               share_embed=True, near_oom_mib=400.0, band_floor_mib=819.0)
        self.assertLess(card[0].free_decode_mib, card[0].band_floor_mib)


class TheScratchFollowsTheBookedBudget(CustomTestCase):
    def test_rc12c_scratch_100_is_capped_to_the_edge(self):
        _c, budgets = _d_pass("nextflash", [])
        fits = _fits(budgets, (100, 48, 48))
        self.assertEqual(fits[0].verdict, "262K VERFEHLT")
        capped, lines = L.d_scratch_cap(fits, [100, 48, 48])
        self.assertEqual(capped, [91, 48, 48], lines)
        self.assertIn("SCRATCH-DECKEL rang0: SGLANG_MOE_SCRATCH_SLOTS 100 -> 91", lines[0])
        again = _fits(budgets, capped)
        self.assertTrue(all(f.verdict == "PASST" for f in again))

    def test_worst_case_peak_leaves_the_corridor_floor(self):
        """TP0 after the cap: posts (KV 262144 fixed) + measured rest, against
        the torch-visible 5090 minus P asleep (1320 + 482) -> >= floor 767."""
        _c, budgets = _d_pass("nextflash", [])
        f = _fits(budgets, (91, 48, 48))[0]
        posts = f.budget_mib - f.kv_rest_mib
        peak = posts + 3186
        free = (32607 - 518) - P_ASLEEP_5090 - peak
        self.assertGreaterEqual(free, 767, (posts, peak, free))
        self.assertEqual(f.buffer_rows, 103)

    def test_the_cap_never_raises(self):
        _c, budgets = _d_pass("nextflash", [])
        fits = _fits(budgets, (60, 48, 48))
        capped, lines = L.d_scratch_cap(fits, [60, 48, 48])
        self.assertEqual((capped, lines), ([60, 48, 48], []))

    def test_set_group_env(self):
        s = "SGLANG_A=1;SGLANG_MOE_SCRATCH_SLOTS=100,48,48;SGLANG_B=x,y"
        self.assertEqual(L.set_group_env(s, "SGLANG_MOE_SCRATCH_SLOTS", "91,48,48"),
                         "SGLANG_A=1;SGLANG_MOE_SCRATCH_SLOTS=91,48,48;SGLANG_B=x,y")
        self.assertEqual(L.set_group_env("SGLANG_A=1", "K", "v"), "SGLANG_A=1;K=v")


class TheRecordHasAFixpoint(CustomTestCase):
    """Boot on the record's budget, measure again: the second iteration
    changes nothing (the withdrawn peak - budget record moved by the whole
    budget change instead)."""

    R_TRUE = 3186.0

    def _boot_log(self, scratch):
        rows = 12 + scratch
        wr_mib = 7442.0 + rows * LAYER_ROW
        posts = wr_mib + 2.082 * 1024 + 0.131 * 1024 + 1024 + 262144 * 14143 / MIB
        demand = posts + self.R_TRUE
        cap = 31.34 * 1024
        ask = 144.0
        others = 1.64 * 1024  # P PP0 asleep, as printed (2 decimals of GiB)
        free = cap - others + ask - demand
        return _log([_oom(ask, free, 1.64, 29.0)], wr=wr_mib / 1024, scratch=scratch,
                     buffer=rows), demand

    def test_second_iteration_changes_nothing(self):
        rec = []
        overs = []
        scratch = 100
        budget = 28288  # rc12c, the first boot
        for _it in range(3):
            text, demand = self._boot_log(scratch)
            fixed, rest, _ = AR.records([("it", text)], n_ranks=3, n_layers=48,
                                        layer_row_mib=LAYER_ROW)
            rec.append((fixed[0], rest[0]))
            overs.append(round(demand - budget))
            # the next boot: budget from the record, scratch from the planner edge
            budget = (int(32607 - 767 - 1320 - 482 - 518 - rest[0]) // 8) * 8
            fits = _fits((budget, 17664, 17864), (scratch, 48, 48), fixed=(fixed[0], 1036.0, 914.0))
            scratch = L.d_scratch_cap(fits, [scratch, 48, 48])[0][0]
        for a, b in zip(rec, rec[1:]):
            self.assertLessEqual(abs(a[0] - b[0]), 1, rec)
            self.assertLessEqual(abs(a[1] - b[1]), 1, rec)
        self.assertEqual(scratch, 91)
        self.assertGreater(abs(overs[0] - overs[1]), 500, overs)  # peak - budget: moved ~900, no fixpoint


class TheGatherGoesThroughOneRing(CustomTestCase):
    def test_byte_identical_and_one_ring(self):
        import torch

        from sglang.srt.layers.moe import expert_offload as EO

        EO._GATHER_RINGS.clear()
        g = torch.Generator().manual_seed(0)
        pool = torch.randint(0, 255, (40, 3, 5), generator=g, dtype=torch.int32)
        for n in (1, 7, 16, 23, 30):
            rows = torch.randperm(40, generator=g)[:n]
            slots = torch.randperm(30, generator=g)[:n]
            a = torch.zeros(30, 3, 5, dtype=torch.int32)
            b = torch.zeros(30, 3, 5, dtype=torch.int32)
            a.index_copy_(0, slots, torch.index_select(pool, 0, rows))
            EO.gather_rows_into(b, slots, pool, rows, ring_rows=EO.GATHER_RING_ROWS)
            self.assertTrue(torch.equal(a, b), n)
        self.assertEqual(len(EO._GATHER_RINGS), 1)
        ring = next(iter(EO._GATHER_RINGS.values()))[0]
        self.assertEqual(tuple(ring.shape), (EO.GATHER_RING_ROWS, 3, 5))


if __name__ == "__main__":
    import unittest

    unittest.main()
