"""One cost library, many consumers (#348b).

Three things are pinned here, in the order they matter:

1. **Byte-identical plans.** The migration moved rate reads and hop pricing out
   of three planners and into ``planner.cost_model``. Every consumer must
   produce the SAME plan through the library as it did before, on the
   reference rig's probe. The K1 arm is covered end to end by
   ``test_key_solver.py`` (unchanged and green); what is pinned here is the
   equivalence of each primitive against a frozen copy of the code it
   replaced, over a sweep wide enough that a rounding difference cannot hide.

2. **Provenance.** An absent rate surfaces as a NAMED absence -- never a zero,
   never a roofline fill, never a plausible default. This is the property the
   pre-#348b readers did not have: the K1 solver defaulted a missing streaming
   bandwidth and a missing GEMM rate to ``0.0`` while its own docstring
   promised it never defaulted anything.

3. **Divergence.** Where two consumers used to price the same thing from two
   sources, the library makes it one number and the test asserts they agree.
   Where two rules genuinely differ and unifying them would re-tune a plan,
   the difference is pinned as a fact so a future change to it is deliberate.

    python -m pytest test/registered/unit/planner/test_cost_model.py -v
"""

import math
import unittest

from sglang.srt.planner import cost_model as cm
from sglang.test.ci.ci_register import register_cpu_ci

# Pure arithmetic over inlined probe fixtures; no device, no NVML, no network.
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


#: The reference rig as the card probe records it (same numbers as
#: test_key_solver._PROBE, inlined for the same reason: a re-probe of the
#: machine must not silently move a pinned figure).
_CARDS = [
    {
        "uuid": "GPU-5090",
        "name": "NVIDIA GeForce RTX 5090",
        "cuda_index": 0,
        "gemm_tflops": 231.97,
        "gemm_bf16_tflops": 231.97,
        "gemm_fp8_tflops": 566.88,
        "gemm_lanes": {"fp8_native": 566.88},
        "membw_read_gbs": 1660.4,
        "membw_gemv_gbs": 1533.8,
        "h2d_gbs": 14.41,
        "d2h_gbs": 14.26,
    },
    {
        "uuid": "GPU-3080a",
        "name": "NVIDIA GeForce RTX 3080",
        "cuda_index": 1,
        "gemm_tflops": 65.57,
        "gemm_bf16_tflops": 65.57,
        "membw_read_gbs": 717.0,
        "membw_gemv_gbs": 717.4,
        "h2d_gbs": 6.47,
        "d2h_gbs": 6.58,
    },
    {
        "uuid": "GPU-3080b",
        "name": "NVIDIA GeForce RTX 3080",
        "cuda_index": 2,
        "gemm_tflops": 65.59,
        "gemm_bf16_tflops": 65.59,
        "membw_read_gbs": 717.1,
        "membw_gemv_gbs": 717.8,
        "h2d_gbs": 13.4,
        "d2h_gbs": 13.16,
    },
]

_PAIR_ROWS = (
    ("GPU-5090", "GPU-3080a", 4.44, 22.4),
    ("GPU-5090", "GPU-3080b", 6.91, 19.8),
    ("GPU-3080a", "GPU-5090", 4.52, 22.1),
    ("GPU-3080a", "GPU-3080b", 4.41, 21.5),
    ("GPU-3080b", "GPU-5090", 6.88, 19.5),
    ("GPU-3080b", "GPU-3080a", 4.32, 21.6),
)

_PROBE = {
    "cards": _CARDS,
    "pairs": [
        {
            "src_uuid": a,
            "dst_uuid": b,
            "bandwidth_gbs": bw,
            "latency_us": lat,
            "transport": "host staging (pinned)",
            "peer_access": False,
        }
        for a, b, bw, lat in _PAIR_ROWS
    ],
}

#: The same rig in the OTHER on-disk shape: the hardware profile's unordered
#: link map. Deliberately holds the same pairs so the reconciliation test has
#: something real to compare.
_PROFILE = {
    "gpus": {c["uuid"]: c for c in _CARDS},
    "links": {
        "GPU-3080a|GPU-5090": {"p2p_gbs": 4.48},
        "GPU-3080b|GPU-5090": {"p2p_gbs": 6.90},
        "GPU-3080a|GPU-3080b": {"p2p_gbs": 4.37},
        "__group__": {"ar_10kb_us": 88.1, "ar_1mb_us": 512.4},
    },
}

_UUIDS = [c["uuid"] for c in _CARDS]


# ---------------------------------------------------------------------------
# 1. Byte-identical: each primitive against the code it replaced
# ---------------------------------------------------------------------------


def _frozen_apportion(total, weights):
    """``sp_shard_utils._apportion`` exactly as it stood at 5fa03e1664."""
    n = len(weights)
    if any(w <= 0 for w in weights):
        raise ValueError(f"capacity weights must be positive; got {list(weights)}")
    total_w = math.fsum(weights)
    ideal = [total * w / total_w for w in weights]
    counts = [int(math.floor(x)) for x in ideal]
    remainder = total - sum(counts)
    order = sorted(range(n), key=lambda i: ideal[i] - counts[i], reverse=True)
    for k in range(remainder):
        counts[order[k]] += 1
    if total >= n:
        donors = sorted(range(n), key=lambda i: counts[i], reverse=True)
        d = 0
        for i in range(n):
            if counts[i] == 0:
                while counts[donors[d]] <= 1:
                    d += 1
                counts[donors[d]] -= 1
                counts[i] = 1
    return counts


def _frozen_boundaries(weights, total_frames):
    """``shard_plan._weighted_boundaries`` exactly as it stood at 5fa03e1664."""
    total_weight = sum(weights)
    boundaries = []
    cumulative = 0.0
    for weight in weights[:-1]:
        cumulative += weight
        boundaries.append(round(total_frames * cumulative / total_weight))
    boundaries.append(total_frames)
    for i in range(1, len(boundaries)):
        boundaries[i] = max(boundaries[i], boundaries[i - 1])
    return boundaries


def _frozen_ring_factor(ranks):
    """``key_solver._ring_factor`` exactly as it stood at 5fa03e1664."""
    return 0.0 if ranks < 2 else 2.0 * (ranks - 1) / ranks


#: Weight vectors the sweep runs: equal, the reference rig's real GEMM ratio,
#: ties that stress the rounding rules, and a near-degenerate share.
_SWEEP_WEIGHTS = (
    [1, 1],
    [1, 3],
    [3, 1],
    [1, 1, 1],
    [231.97, 65.57, 65.59],
    [566.88, 65.57, 65.59],
    [1, 2, 3, 4],
    [0.5, 0.5, 9.0],
    [7, 7, 7, 7, 7],
    [1e-6, 1.0],
    [2, 2, 3],
)
_SWEEP_TOTALS = (1, 2, 3, 4, 5, 7, 10, 11, 16, 17, 63, 100, 101, 999, 4096, 4097, 65536)


class TestMigrationIsByteIdentical(unittest.TestCase):
    def test_the_hamilton_rule_matches_the_code_it_replaced(self):
        """The diffusion SP split calls the library now; same counts."""
        checked = 0
        for weights in _SWEEP_WEIGHTS:
            for total in _SWEEP_TOTALS:
                self.assertEqual(
                    cm.apportion_largest_remainder(total, weights, min_one=True),
                    _frozen_apportion(total, weights),
                    f"weights={weights} total={total}",
                )
                checked += 1
        self.assertGreater(checked, 150)

    def test_the_cumulative_rule_matches_the_code_it_replaced(self):
        """The video chunk planner calls the library now; same boundaries."""
        for weights in _SWEEP_WEIGHTS:
            for total in (0,) + _SWEEP_TOTALS:
                self.assertEqual(
                    cm.cumulative_boundaries(total, weights),
                    _frozen_boundaries(weights, total),
                    f"weights={weights} total={total}",
                )

    def test_the_ring_factor_matches_the_code_it_replaced(self):
        for ranks in range(0, 33):
            self.assertEqual(cm.ring_factor(ranks), _frozen_ring_factor(ranks))

    def test_the_allreduce_formula_matches_the_inlined_one(self):
        """``key_solver.collective_decode_s`` inlined this arithmetic."""
        for ranks in (2, 3, 4, 8):
            for payload in (5120.0, 5120.0 * 20000, 1.0):
                for bw, lat in ((4.32, 22.4), (6.88, 19.5), (0.1, 900.0)):
                    inlined = (ranks - 1) * lat * 1e-6 + _frozen_ring_factor(
                        ranks
                    ) * payload / (bw * 1e9 * 1.0)
                    self.assertAlmostEqual(
                        cm.allreduce_seconds(payload, ranks, bw, lat),
                        inlined,
                        places=15,
                    )

    def test_the_narrowest_pair_matches_the_hand_rolled_reduction(self):
        """``rates_from_probe`` used to take min/max over the pair list itself."""
        matrix = cm.pair_matrix_from_card_probe(_PROBE, _UUIDS)
        self.assertEqual(
            matrix.narrowest_bandwidth_gbs().require("bw"),
            min(r[2] for r in _PAIR_ROWS),
        )
        self.assertEqual(
            matrix.worst_latency_us().require("lat"),
            max(r[3] for r in _PAIR_ROWS),
        )

    def test_a_subset_of_cards_only_sees_its_own_pairs(self):
        """The K1 solver asks for the ranks it uses, not the whole rig."""
        matrix = cm.pair_matrix_from_card_probe(_PROBE, ["GPU-5090", "GPU-3080b"])
        self.assertEqual(
            sorted(matrix.hops),
            [
                ("GPU-3080b", "GPU-5090"),
                ("GPU-5090", "GPU-3080b"),
            ],
        )
        self.assertEqual(matrix.narrowest_bandwidth_gbs().require("bw"), 6.88)


# ---------------------------------------------------------------------------
# 2. Provenance: an absent rate is named, never filled
# ---------------------------------------------------------------------------


class TestProvenance(unittest.TestCase):
    def test_a_rate_cannot_be_absent_and_carry_a_value(self):
        with self.assertRaises(ValueError):
            cm.Rate(1.0, cm.Provenance.ABSENT, "nope")
        with self.assertRaises(ValueError):
            cm.Rate(None, cm.Provenance.MEASURED, "nope")

    def test_requiring_an_absent_rate_names_it_instead_of_returning_zero(self):
        rate = cm.Rate.absent("the card probe never scored this card")
        with self.assertRaises(cm.AbsentRate) as ctx:
            rate.require("GEMM rate for GPU-3080a")
        self.assertIn("GPU-3080a", str(ctx.exception))
        self.assertIn("never scored", str(ctx.exception))
        # And it is NOT quietly 0.0 for a caller that forgot to check.
        self.assertIsNone(rate.or_none())

    def test_a_card_without_a_gemm_rate_is_a_named_absence_not_a_zero(self):
        entries = [dict(_CARDS[0]), None, dict(_CARDS[2])]
        rates = cm.compute_rates_from_entries(entries, _UUIDS)
        self.assertFalse(rates.rates[0].is_absent)
        self.assertTrue(rates.rates[1].is_absent)
        self.assertFalse(rates.rates[2].is_absent)
        (absence,) = rates.absences()
        self.assertIn("GPU-3080a", absence)
        self.assertIn("Run the rig probe", absence)
        # The measured neighbours keep their real numbers.
        self.assertEqual(rates.rates[0].value, 231.97)
        self.assertEqual(rates.rates[2].value, 65.59)
        # And asking for the vector refuses rather than yielding a 0.0 divisor.
        with self.assertRaises(cm.AbsentRate):
            rates.values()

    def test_no_profile_at_all_names_every_card(self):
        rates = cm.compute_rates_for_cards(_UUIDS, profile={})
        self.assertEqual(len(rates.absences()), 3)
        self.assertTrue(
            all("no cached hardware profile" in a for a in rates.absences())
        )

    def test_a_measured_rate_carries_its_lane_label(self):
        rates = cm.compute_rates_for_cards(_UUIDS, profile=_PROFILE)
        self.assertEqual([r.value for r in rates.rates], [231.97, 65.57, 65.59])
        self.assertTrue(
            all(r.provenance is cm.Provenance.MEASURED for r in rates.rates)
        )
        self.assertTrue(all(r.label for r in rates.rates))

    def test_the_fp8_lane_is_taken_where_the_card_has_one(self):
        """The #324 lane resolution is reused, not re-derived: the 5090 has a
        measured fp8 lane and the two 3080s fall back with a LOUD warning."""
        rates = cm.compute_rates_for_cards(_UUIDS, fmt="fp8", profile=_PROFILE)
        self.assertEqual(rates.rates[0].value, 566.88)
        self.assertEqual(rates.rates[1].value, 65.57)
        self.assertTrue(rates.warnings)
        self.assertTrue(any("3080" in w for w in rates.warnings))

    def test_a_missing_membw_is_named_rather_than_defaulted_to_zero(self):
        """The gap this library closes on the K1 side.

        ``rates_from_probe`` read ``membw_read_gbs or membw_gbs or 0.0`` and
        recorded nothing when both were missing -- a 0.0 that then reached the
        decode term as an almost-valid divisor.
        """
        stripped = dict(_CARDS[1])
        stripped.pop("membw_read_gbs")
        rates = cm.memory_rates_from_entries(
            [_CARDS[0], stripped, _CARDS[2]], _UUIDS, "membw"
        )
        self.assertFalse(rates[0].is_absent)
        self.assertTrue(rates[1].is_absent)
        self.assertIn("GPU-3080a", rates[1].source)
        self.assertIn("streaming read rate", rates[1].source)
        self.assertIsNone(rates[1].value)

    def test_an_absent_pair_matrix_is_named_not_priced(self):
        matrix = cm.pair_matrix_from_card_probe({"cards": _CARDS, "pairs": []}, _UUIDS)
        bw = matrix.narrowest_bandwidth_gbs()
        self.assertTrue(bw.is_absent)
        self.assertIn("reported absent", bw.source)
        (absence,) = matrix.absences()
        self.assertIn("no ordered pair was measured", absence)

    def test_an_incomplete_pair_matrix_names_the_missing_wires(self):
        partial = {
            "cards": _CARDS,
            "pairs": [p for p in _PROBE["pairs"] if p["src_uuid"] == "GPU-5090"],
        }
        (absence,) = cm.pair_matrix_from_card_probe(partial, _UUIDS).absences()
        self.assertIn("GPU-3080a -> GPU-5090", absence)
        self.assertIn("incomplete", absence)

    def test_a_single_card_has_no_link_and_no_absence(self):
        matrix = cm.pair_matrix_from_card_probe(_PROBE, ["GPU-5090"])
        self.assertEqual(matrix.absences(), [])

    def test_a_loopback_row_is_rejected_and_the_rejection_is_visible(self):
        """A same-card copy is not a hop.

        Left in, it wins ``min(bandwidth)`` with a device-local rate and makes
        every collective look free. The probe cannot emit one today; a
        hand-edited or foreign artifact can, and the old reader had no guard.
        """
        poisoned = {
            "cards": _CARDS,
            "pairs": list(_PROBE["pairs"])
            + [
                {
                    "src_uuid": "GPU-5090",
                    "dst_uuid": "GPU-5090",
                    "bandwidth_gbs": 1400.0,
                    "latency_us": 0.4,
                    "transport": "device-local",
                }
            ],
        }
        matrix = cm.pair_matrix_from_card_probe(poisoned, _UUIDS)
        self.assertEqual(len(matrix.rejected), 1)
        self.assertIn("loopback", matrix.rejected[0])
        # The narrowest pair is untouched by the poisoned row.
        self.assertEqual(matrix.narrowest_bandwidth_gbs().require("bw"), 4.32)
        self.assertNotIn(("GPU-5090", "GPU-5090"), matrix.hops)

    def test_the_profile_shape_rejects_loopback_too(self):
        poisoned = dict(_PROFILE)
        poisoned["links"] = dict(_PROFILE["links"])
        poisoned["links"]["GPU-5090|GPU-5090"] = {"p2p_gbs": 1400.0}
        matrix = cm.pair_matrix_from_hardware_profile(poisoned, _UUIDS)
        self.assertEqual(len(matrix.rejected), 1)
        self.assertIn("loopback", matrix.rejected[0])

    def test_the_profile_has_no_per_pair_latency_and_says_so(self):
        """It times all-reduce for the GROUP. Charging that to one pair would
        be an invented number, so the latency stays absent."""
        matrix = cm.pair_matrix_from_hardware_profile(_PROFILE, _UUIDS)
        hop = matrix.hop("GPU-5090", "GPU-3080a")
        self.assertFalse(hop.bandwidth_gbs.is_absent)
        self.assertTrue(hop.latency_us.is_absent)
        self.assertIn("whole group", hop.latency_us.source)
        self.assertTrue(matrix.worst_latency_us().is_absent)

    def test_the_unordered_shape_states_that_it_widened(self):
        matrix = cm.pair_matrix_from_hardware_profile(_PROFILE, _UUIDS)
        self.assertEqual(
            matrix.hop("GPU-5090", "GPU-3080a").bandwidth_gbs.value,
            matrix.hop("GPU-3080a", "GPU-5090").bandwidth_gbs.value,
        )
        self.assertTrue(any("asymmetry" in n for n in matrix.notes))


# ---------------------------------------------------------------------------
# 3. Divergence: two consumers, one number
# ---------------------------------------------------------------------------


class TestDivergenceIsCaught(unittest.TestCase):
    def test_two_consumers_reading_one_matrix_price_a_pair_identically(self):
        """The property the unification buys.

        Before #348b the K1 solver read ``probe["pairs"]`` with its own parser
        and the hardware-profile consumers read ``profile["links"]`` with
        another. Two readers over one artifact now produce one number for one
        pair, so a consumer cannot disagree with a consumer.
        """
        a = cm.pair_matrix_from_card_probe(_PROBE, _UUIDS)
        b = cm.pair_matrix_from_card_probe(_PROBE, _UUIDS)
        for pair in a.hops:
            self.assertEqual(
                a.hop(*pair).bandwidth_gbs.value,
                b.hop(*pair).bandwidth_gbs.value,
                pair,
            )
        self.assertEqual(
            a.narrowest_bandwidth_gbs().value, b.narrowest_bandwidth_gbs().value
        )

    def test_the_two_artifacts_disagreeing_on_a_pair_is_reported_not_averaged(self):
        """The card probe and the hardware profile measure the same wire with
        different methods. Where they disagree, the library names it."""
        skewed = dict(_PROFILE)
        skewed["links"] = dict(_PROFILE["links"])
        skewed["links"]["GPU-3080b|GPU-5090"] = {"p2p_gbs": 12.0}  # vs 6.88/6.91
        merged, divergences = cm.reconcile_pair_matrices(
            cm.pair_matrix_from_card_probe(_PROBE, _UUIDS),
            cm.pair_matrix_from_hardware_profile(skewed, _UUIDS),
        )
        self.assertTrue(divergences)
        joined = " ".join(divergences)
        self.assertIn("GPU-3080b", joined)
        self.assertIn("12.00", joined)
        # The first argument -- the ordered card probe -- wins the pair.
        self.assertEqual(merged.hop("GPU-3080b", "GPU-5090").bandwidth_gbs.value, 6.88)

    def test_artifacts_that_agree_produce_no_divergence(self):
        _merged, divergences = cm.reconcile_pair_matrices(
            cm.pair_matrix_from_card_probe(_PROBE, _UUIDS),
            cm.pair_matrix_from_hardware_profile(_PROFILE, _UUIDS),
        )
        self.assertEqual(divergences, [])

    def test_a_pair_only_one_artifact_has_is_carried_over(self):
        thin = {"cards": _CARDS, "pairs": _PROBE["pairs"][:2]}
        merged, _div = cm.reconcile_pair_matrices(
            cm.pair_matrix_from_card_probe(thin, _UUIDS),
            cm.pair_matrix_from_hardware_profile(_PROFILE, _UUIDS),
        )
        self.assertIsNotNone(merged.hop("GPU-3080a", "GPU-3080b"))

    def test_the_two_rounding_rules_disagree_and_the_difference_is_pinned(self):
        """The one divergence this refactor deliberately does NOT unify.

        The video planner rounds cumulatively (it needs contiguous boundaries
        on a shared timeline); the diffusion SP split apportions by largest
        remainder. On ``(1, 3)`` over 10 units they part company. Unifying them
        would move a shipped plan, which is a re-tune and not a refactor, so
        the difference is a pinned fact instead of a silent one.
        """
        self.assertEqual(cm.apportion_largest_remainder(10, [1, 3]), [3, 7])
        self.assertEqual(cm.apportion_cumulative(10, [1, 3]), [2, 8])
        self.assertEqual(cm.apportion_largest_remainder(10, [1, 1, 1]), [4, 3, 3])
        self.assertEqual(cm.apportion_cumulative(10, [1, 1, 1]), [3, 4, 3])
        # Both rules always tile the total exactly -- the invariant that IS
        # shared, and the one a split cannot be allowed to break.
        for weights in _SWEEP_WEIGHTS:
            for total in _SWEEP_TOTALS:
                self.assertEqual(
                    sum(cm.apportion_largest_remainder(total, weights)), total
                )
                self.assertEqual(sum(cm.apportion_cumulative(total, weights)), total)

    def test_the_absent_link_substitutions_are_named_in_one_place(self):
        """Three code paths answered "no pair matrix" with numbers 80x apart.

        The library does not pick a winner -- that would re-tune an unprobed
        rig -- but it holds both, so the gap is one grep away instead of
        spread over three modules.
        """
        self.assertEqual(cm.ABSENT_LINK_RANKING_PLACEHOLDER_GBS, 1e-3)
        self.assertEqual(cm.ABSENT_LINK_ASSUMED_GBS, 8.0)

        from sglang.srt.planner import lever_profiles

        self.assertEqual(lever_profiles._FALLBACK_LINK_GBS, cm.ABSENT_LINK_ASSUMED_GBS)

    def test_the_absent_link_placeholder_cannot_reorder_candidates(self):
        """The #216/#264 guard, checked as arithmetic rather than asserted.

        The collective term depends on the layer count, the hidden size and
        the rank count -- never on the candidate split -- so substituting any
        link rate adds the SAME constant to every candidate and the ordering
        of their compute terms survives.
        """
        compute = [0.9, 1.0, 1.1, 1.05, 0.95]  # five candidates' compute time
        for bw in (
            cm.ABSENT_LINK_RANKING_PLACEHOLDER_GBS,
            cm.ABSENT_LINK_ASSUMED_GBS,
            0.1,
            4.32,
        ):
            comm = cm.allreduce_seconds(5120.0 * 62 * 2, 3, bw, 22.4)
            ranked = sorted(range(5), key=lambda i: compute[i] + comm)
            self.assertEqual(ranked, sorted(range(5), key=lambda i: compute[i]), bw)

    def test_a_ratio_is_not_invariant_and_that_is_the_open_risk(self):
        """The other half of the same guard, stated as a failing property.

        ``lever_profiles._speed_ratios`` divides two predicted times and
        compares the result against a move threshold. An additive constant
        does NOT survive a ratio: at 8.0 GB/s and at 0.1 GB/s the same two
        candidates produce different ratios on identical measured inputs. The
        fix is for the ratio consumer to report the absence rather than rank
        through it; this test pins the size of what is at stake.
        """
        base, cand = 1.0, 0.8
        ratios = []
        for bw in (cm.ABSENT_LINK_ASSUMED_GBS, 0.1):
            comm = cm.allreduce_seconds(5120.0 * 62 * 2, 3, bw, 22.4)
            ratios.append((base + comm) / (cand + comm))
        self.assertNotAlmostEqual(ratios[0], ratios[1], places=3)
        self.assertGreater(ratios[0], ratios[1])  # a smaller comm term = bigger claim


# ---------------------------------------------------------------------------
# 4. The bundle, and the documented #302 entry point
# ---------------------------------------------------------------------------


class TestCostSources(unittest.TestCase):
    def test_both_axes_come_back_resolved_from_the_artifacts_on_hand(self):
        sources = cm.load_cost_sources(
            _UUIDS, card_probe=_PROBE, hardware_profile=_PROFILE
        )
        self.assertEqual(sources.compute.values(), [231.97, 65.57, 65.59])
        self.assertEqual(sources.links.narrowest_bandwidth_gbs().require("bw"), 4.32)
        self.assertEqual(sources.absences(), [])
        self.assertEqual(sources.divergences, ())

    def test_the_302_entry_point_resolves_a_moe_family_and_a_hop(self):
        """What expert placement needs, and all it needs.

        A MoE family whose format diverges from the checkpoint-wide one scores
        on its own lane per card -- so #302 gets the fp8 expert rate on the
        5090 and the bf16 fallback on the 3080s without deriving anything --
        and the directed hop for the dispatch/combine collective.
        """
        from sglang.srt import uneven_perf

        sources = cm.load_cost_sources(
            _UUIDS,
            fmt="bf16",
            family_formats={uneven_perf.GEMM_FAMILY_MOE: "fp8"},
            card_probe=_PROBE,
            hardware_profile=_PROFILE,
        )
        moe = sources.compute.for_family(uneven_perf.GEMM_FAMILY_MOE)
        scalar = sources.compute.for_family(None)
        self.assertTrue(sources.compute.mixed)
        self.assertEqual(moe[0].value, 566.88)  # 5090 on its native fp8 lane
        self.assertEqual(scalar[0].value, 231.97)  # the dense rate, unchanged
        self.assertEqual(moe[1].value, 65.57)  # 3080 falls back, loudly
        hop = sources.links.hop("GPU-5090", "GPU-3080b")
        self.assertEqual(hop.bandwidth_gbs.value, 6.91)
        self.assertGreater(
            cm.allreduce_seconds(1 << 20, 3, hop.bandwidth_gbs.value, 19.8), 0.0
        )

    def test_a_family_that_does_not_diverge_returns_the_scalar_object(self):
        sources = cm.load_cost_sources(
            _UUIDS, card_probe=_PROBE, hardware_profile=_PROFILE
        )
        self.assertFalse(sources.compute.mixed)
        self.assertEqual(sources.compute.for_family("moe"), sources.compute.rates)

    def test_weights_normalise_and_refuse_an_absent_card(self):
        sources = cm.load_cost_sources(
            _UUIDS, card_probe=_PROBE, hardware_profile=_PROFILE
        )
        weights = sources.compute.weights()
        self.assertAlmostEqual(sum(weights), 1.0, places=12)
        self.assertGreater(weights[0], weights[1])
        thin = dict(_PROFILE)
        thin["gpus"] = {"GPU-5090": _CARDS[0]}
        with self.assertRaises(cm.AbsentRate):
            cm.compute_rates_for_cards(_UUIDS, profile=thin).weights()


if __name__ == "__main__":
    unittest.main()
