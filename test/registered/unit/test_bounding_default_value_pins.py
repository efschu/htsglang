"""Value pins for shipped bounding defaults (#514, from audit #505 axis C).

See ``docs/dev/CONVENTION_bounding_defaults.md``. The rule these implement:
a numeric default that exists to BOUND something ships with a test that fails
when the value changes, so the number has to be argued rather than inherited.

Audit #505 enumerated 106 fork-added bounding defaults and found ZERO with such
a test. The pattern it found instead reads the default and derives the
assertion from it, which passes for every possible value -- that one is
corrected at its own site, ``test_retract_decode_fcfs.py``, because the point
is easiest to see next to the guard test it used to be mistaken for.

This file is the reference implementation for the top of that backlog. It is
deliberately NOT a sweep: pinning all 106 in one pass would produce 106
unargued literals, which is the same defect with a green tick on it. New
postens are added here as they are touched.

What a green run here does and does not mean, restated because the distinction
is the whole point of the convention: it means these defaults are DELIBERATE.
It does not mean they are correct, and it does not mean they BIND at the served
geometry. Where a posten is known not to bind, that is asserted explicitly
rather than left as an absence.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class BoundingDefaultValuePinTest(CustomTestCase):
    def test_measured_kv_budget_safety_margin(self):
        """#505-C-03. Evidence tier: DESK, and contradicted by a measurement
        recorded in its own consumer -- pinned so that stays visible.

        ``model_runner_kv_cache_mixin.py:809-815`` records: the draft-solo host
        carries dual-prefill / draft-append transients that scale with prompt
        length, measured 2026-07-22 at ~1 GiB for 10k prefill and ~2-3.5 GiB at
        50k, while shadow ranks served everything with ~1.6 GiB. Every one of
        those numbers is above 400 MiB, on every rank. The surrounding comment
        concedes the point: "the only ASSUMED number left is the safety margin
        itself" (:786-788).

        The value is NOT changed here. Deriving the right one is a per-rank-role
        question (#505-C-03) and needs the measurement, not a desk edit -- and
        the feature is opt-in (SGLANG_MEASURED_KV_BUDGET defaults False), which
        is why it is a backlog item rather than an incident. The pin exists so
        the next person to touch this number meets the contradiction.
        """
        self.assertEqual(
            str(envs.SGLANG_MEASURED_KV_BUDGET_SAFETY_MIB.get()),
            "400",
            "SGLANG_MEASURED_KV_BUDGET_SAFETY_MIB changed. It is a DESK value "
            "that the measurement at model_runner_kv_cache_mixin.py:809-815 "
            "already contradicts (10k prefill ~1 GiB, 50k ~2-3.5 GiB on the "
            "draft-solo host). Deriving it per rank role is #505-C-03",
        )

    def test_gguf_stream_trim_watermark_is_off_and_therefore_inert(self):
        """#505-C-02. The honest pin: this one is armed at 0.0, i.e. OFF, so
        it bounds nothing at any geometry.

        The measurement that motivated it sits four lines above the flag in
        ``environ.py`` -- a stream window in which memory.current moved
        88 -> 102 GiB inside 15 s, on a swapless box that streams GGUF weights
        in the standing recipe. So this is the #449 shape with the sign
        flipped: not a bound too high to bind, a bound never armed.

        Asserted as INERT rather than silently left alone, so that arming it
        becomes a deliberate act with a red test attached. The consumer agrees
        the off state is real: gguf_shards.py:477 "Off unless ... is set; when
        off, nothing".
        """
        self.assertEqual(
            envs.SGLANG_GGUF_STREAM_TRIM_SOFT_GIB.get(),
            0.0,
            "SGLANG_GGUF_STREAM_TRIM_SOFT_GIB is no longer 0.0. Arming it is "
            "the right direction (#505-C-02) but the value needs the measured "
            "stream slew behind it, not a desk number -- update this pin with "
            "the measurement in the same change",
        )

    def test_gguf_stream_trim_headroom_is_off_and_therefore_inert(self):
        """#537. Evidence tier: none, deliberately -- the term ships at 0.0.

        The #537 fix raises the trim's target to the UNRECLAIMABLE floor
        (``anon`` + the pinned host pool, which the kernel files under ``file``
        where ``memory.current`` cannot tell it from page cache). That floor is
        a physical statement and needs no calibration. This headroom is the one
        policy term on top of it -- room for the loader's own read-ahead inside
        the budget -- and picking it needs a load-time page-cache measurement
        that only a GPU window can take.

        It therefore ships INERT: at 0.0 the trim may still drive page cache
        down to the floor exactly as it did before #537, and the fix's
        behaviour comes entirely from the floor. Arming it is a deliberate act
        with a red test attached, and the measurement belongs in this message
        when it exists (``gguf_shards.ProgressCoupledTrim._effective_target``).
        """
        self.assertEqual(
            envs.SGLANG_GGUF_STREAM_TRIM_HEADROOM_GIB.get(),
            0.0,
            "SGLANG_GGUF_STREAM_TRIM_HEADROOM_GIB is no longer 0.0. It is the "
            "only policy term in the #537 trim budget; arming it needs the "
            "measured load-time read-ahead working set, not a desk number -- "
            "record the measurement in this pin in the same change",
        )

    def test_retract_solo_oom_retry_budget_is_pinned_at_its_own_site(self):
        """#505-C-05's named anti-pattern. The pin lives next to the guard test
        in ``test/registered/unit/managers/test_retract_decode_fcfs.py`` rather
        than here, because that is where the reader who would change it looks.
        This test only asserts the two have not drifted apart.
        """
        self.assertEqual(envs.SGLANG_RETRACT_SOLO_OOM_MAX_RETRIES.get(), 8)


if __name__ == "__main__":
    unittest.main()
