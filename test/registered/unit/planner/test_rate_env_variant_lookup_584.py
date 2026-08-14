"""#584 -- dating a rate stored under a CAPACITY-DISAMBIGUATED profile name.

FOUND ON METAL, R13 act window. `card_rate_pass --run` measured all three
cards; `--show` then reported:

    RTX 3080 20GB: gemm 50.97 TFLOPS  membw 716.2 GB/s  [UNKNOWN] the rate was
    measured at driver 595.58.03, power limit 200 W, and NVML reports no
    current environment for this card to compare against
    RTX 5090:      gemm 203.42 TFLOPS ...                [FRESH]

Both rates were taken in the same pass, seconds apart, on cards NVML can see.
The 3080's could not be dated, and re-running the pass could never fix it.

THE CROSS-BRANCH SEAM. Two things merged in the same batch:

* `fix/card-library-guards` resolves a card by CAPACITY, because the driver
  calls both the 10 GB and the 20 GB RTX 3080 ``NVIDIA GeForce RTX 3080`` and
  the 20 GB cards were silently resolving onto the 10240 MiB seed entry. So
  this rig's 3080 profile is named ``RTX 3080 20GB``.
* `rate_env` dates a stored rate by looking its card NAME up in a table of
  live environments keyed by the raw NVML name -- ``RTX 3080``.

Exact-key lookup, two different names for one card: permanent UNKNOWN for
every capacity-disambiguated profile. The 5090 escaped only because nothing
collides with it, which is why the defect looked like a card-specific quirk.

THE FIX is not a new rule. ``CardLibrary.variants`` already states the
relation -- an entry is a candidate when its key equals the request, EXTENDS
it, or is extended BY it -- and the lookup now uses that relation instead of
equality. One rule, held in one place.

Hermetic: no NVML, no cards, the live table is injected.
"""

import unittest

from sglang.srt.planner.rate_env import RateEnv, check_card_rate_freshness
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_DRIVER = "595.58.03"


def _env(power_w, driver=_DRIVER):
    return RateEnv(driver_version=driver, power_limit_mw=int(power_w * 1000))


def _token(power_w, driver=_DRIVER):
    return _env(power_w, driver).token


class TestCapacitySuffixedProfileCanBeDated(CustomTestCase):
    """The metal case, reproduced hermetically."""

    def test_the_20gb_profile_is_dated_against_the_plain_nvml_name(self):
        verdict = check_card_rate_freshness(
            "RTX 3080 20GB",
            _token(200),
            by_name={"rtx 3080": [_env(200)]},
        )
        self.assertEqual(verdict.state, "fresh", verdict.reason)

    def test_the_reverse_direction_works_too(self):
        """A config file may name the variant the driver never does."""
        verdict = check_card_rate_freshness(
            "RTX 3080",
            _token(200),
            by_name={"rtx 3080 20gb": [_env(200)]},
        )
        self.assertEqual(verdict.state, "fresh", verdict.reason)

    def test_the_uncollided_card_still_works(self):
        """The 5090 was never broken; the fix must not break it."""
        verdict = check_card_rate_freshness(
            "RTX 5090", _token(400), by_name={"rtx 5090": [_env(400)]}
        )
        self.assertEqual(verdict.state, "fresh", verdict.reason)


class TestTheLooseningDoesNotSwallowRealVerdicts(CustomTestCase):
    """A prefix match that reports FRESH too eagerly is worse than UNKNOWN:
    it certifies a rate measured under a power limit the rig no longer runs."""

    def test_a_changed_power_limit_is_still_STALE_through_the_variant_match(self):
        verdict = check_card_rate_freshness(
            "RTX 3080 20GB",
            _token(320),
            by_name={"rtx 3080": [_env(200)]},
        )
        self.assertEqual(verdict.state, "stale", verdict.reason)
        self.assertIn("320", verdict.reason)
        self.assertIn("200", verdict.reason)

    def test_a_changed_driver_is_still_STALE(self):
        verdict = check_card_rate_freshness(
            "RTX 3080 20GB",
            _token(200, driver="590.00.00"),
            by_name={"rtx 3080": [_env(200)]},
        )
        self.assertEqual(verdict.state, "stale", verdict.reason)

    def test_a_DIFFERENT_card_model_does_not_match(self):
        """``RTX 3080`` must not be dated against an ``RTX 3090`` -- the
        relation is a token-boundary prefix, not a substring."""
        verdict = check_card_rate_freshness(
            "RTX 3080 20GB", _token(200), by_name={"rtx 3090": [_env(200)]}
        )
        self.assertEqual(verdict.state, "unknown", verdict.reason)

    def test_a_longer_model_number_is_not_a_variant(self):
        verdict = check_card_rate_freshness(
            "RTX 308", _token(200), by_name={"rtx 3080": [_env(200)]}
        )
        self.assertEqual(verdict.state, "unknown", verdict.reason)

    def test_an_absent_card_is_still_UNKNOWN(self):
        verdict = check_card_rate_freshness(
            "RTX 3080 20GB", _token(200), by_name={"rtx 5090": [_env(400)]}
        )
        self.assertEqual(verdict.state, "unknown", verdict.reason)

    def test_an_unfingerprinted_rate_is_still_UNKNOWN(self):
        """The predates-fingerprinting verdict must survive the fix -- it is
        what told this shift the m584 rates had to be re-measured."""
        verdict = check_card_rate_freshness(
            "RTX 3080 20GB", None, by_name={"rtx 3080": [_env(200)]}
        )
        self.assertEqual(verdict.state, "unknown", verdict.reason)
        self.assertIn("predates rate fingerprinting", verdict.reason)


class TestTwoCardsOfOneModelAtDifferentLimits(CustomTestCase):
    def test_matching_ANY_live_env_of_the_variant_is_enough(self):
        """rate_env's own documented rule: the rate was taken on one of these
        cards, and which one is a question the name cannot answer."""
        verdict = check_card_rate_freshness(
            "RTX 3080 20GB",
            _token(200),
            by_name={"rtx 3080": [_env(320), _env(200)]},
        )
        self.assertEqual(verdict.state, "fresh", verdict.reason)

    def test_matching_none_of_them_is_STALE(self):
        verdict = check_card_rate_freshness(
            "RTX 3080 20GB",
            _token(150),
            by_name={"rtx 3080": [_env(320), _env(200)]},
        )
        self.assertEqual(verdict.state, "stale", verdict.reason)


if __name__ == "__main__":
    unittest.main()
