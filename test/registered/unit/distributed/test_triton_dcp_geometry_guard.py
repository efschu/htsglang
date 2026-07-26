"""The Triton backend's DCP geometry gate: loud at boot, never silently wrong.

Two independent rejections live in ``reject_unsupported_dcp_geometry``:

1. **No uneven-DCP wiring.** The Triton DCP path implements exactly one owner
   rule -- the EVEN modulo one -- on the write side (``out_cache_loc //
   dcp_size`` + ``positions % dcp_size == dcp_rank``) and on the read side
   (``get_dcp_lens``), and it never gathers the kv heads before writing
   (flashinfer's ``_dcp_write_gather``). The fork's uneven-DCP pool is a
   different layout: a virtual block of ``sum(token ratios)`` slots, every slot
   declared to hold the FULL replicated kv-head set. Measured on that
   combination: CJK mojibake in an English greedy completion.

   The predecessor guard SKIPPED itself on exactly that config
   (``not uneven_dcp_active(...)``), so the geometry it should have rejected
   hardest was the one it waved through. ``test_the_old_exemption_waved_through
   _the_measured_garbage`` pins that regression directly.

2. **Even DCP without kv-head replication** (the pre-existing rule, kept
   byte-identical): ``tp_size // total_kv_heads >= dcp_size``.

CPU only: the rule is a pure function of the geometry, so it is called here
directly -- no device, no process group, no ModelRunner.
"""

import unittest

from sglang.srt.layers.attention.triton_backend import (
    reject_unsupported_dcp_geometry,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _call(dcp, tp, kv, *, plan=False, tokens=False, weightless=False):
    reject_unsupported_dcp_geometry(
        dcp,
        tp,
        kv,
        uneven_plan=plan,
        weighted_tokens=tokens,
        weightless_kv=weightless,
    )


class TestTritonDcpGeometryGuard(CustomTestCase):
    # ---------------------------------------------------------------- uneven

    def test_the_old_exemption_waved_through_the_measured_garbage(self):
        """THE falsifier for this fix.

        The measured mojibake config is uneven-DCP + a dense (qwen2) class on
        triton: a --rank-tp-ratio plan, a non-uniform token vector, dcp == tp.
        The old condition was ``if dcp_size > 1 and not uneven_dcp_active(...)``
        -- i.e. a token vector DISABLED the whole guard. This asserts both
        halves: the old predicate skips, the new rule rejects.
        """
        dcp, tp, kv = 3, 3, 8
        weighted_tokens = True

        # what the old code did: guard body not entered at all
        old_guard_runs = dcp > 1 and not weighted_tokens
        self.assertFalse(
            old_guard_runs,
            "premise of this test is wrong: the old guard did run here",
        )

        with self.assertRaises(ValueError) as ctx:
            _call(dcp, tp, kv, plan=True, tokens=True)
        msg = str(ctx.exception)
        self.assertIn("WEIGHTED owner rule", msg)
        self.assertIn("flashinfer", msg)

    def test_uneven_plan_without_a_token_vector_is_rejected_too(self):
        """The silent hole the replication arithmetic could not see.

        tp=4 / kv=2 / dcp=2 has replicas = 2 >= 2, so the even-DCP arithmetic
        PASSES it. With a --rank-tp-ratio plan installed the pool is still
        token-sharded with replicated kv heads, which triton neither writes nor
        reads correctly -- and its q-head shards are unequal on top. Without
        this branch the config boots and serves wrong tokens.
        """
        # the arithmetic alone accepts it ...
        _call(2, 4, 2)
        # ... the plan makes it unserviceable
        with self.assertRaises(ValueError) as ctx:
            _call(2, 4, 2, plan=True)
        self.assertIn("--rank-tp-ratio", str(ctx.exception))
        self.assertIn("REPLICATED kv heads", str(ctx.exception))

    def test_weightless_kv_fastlane_is_rejected(self):
        """Head counts [all, 0, 0] are not an equal split either.

        Geometry chosen so the replication arithmetic ACCEPTS it (tp=4, kv=2,
        dcp=2 -> replicas 2 >= 2); without the fast-lane branch nothing would
        stop it.
        """
        _call(2, 4, 2)  # same geometry, fast lane off -> accepted
        with self.assertRaises(ValueError) as ctx:
            _call(2, 4, 2, weightless=True)
        self.assertIn("weightless", str(ctx.exception))

    def test_every_uneven_reason_is_named_when_several_apply(self):
        """A config that trips all three must name all three, so the operator
        does not fix one and re-run into the next."""
        with self.assertRaises(ValueError) as ctx:
            _call(3, 3, 8, plan=True, tokens=True, weightless=True)
        msg = str(ctx.exception)
        for fragment in ("--rank-tp-ratio", "WEIGHTED owner rule", "weightless"):
            self.assertIn(fragment, msg)

    # ------------------------------------------------- even-DCP replication

    def test_replication_arithmetic_on_the_four_measured_cases(self):
        """Unchanged from the predecessor guard, re-pinned behaviourally
        (it used to be asserted only as a source regex)."""
        cases = [
            # (attn_tp, total_kv, dcp, must_reject)
            (2, 2, 2, True),  # Qwen2.5-1.5B TP=2/DCP=2 -> measured mojibake
            (2, 8, 2, True),  # Qwen3-0.6B              -> measured mojibake
            (4, 2, 2, False),  # replicas 2 >= 2        -> measured coherent
            (8, 4, 2, False),  # the path's origin (Qwen3.5-397B)
        ]
        for attn_tp, kv, dcp, must_reject in cases:
            with self.subTest(tp=attn_tp, kv=kv, dcp=dcp):
                if must_reject:
                    with self.assertRaises(ValueError) as ctx:
                        _call(dcp, attn_tp, kv)
                    # names the numbers, not just "unsupported"
                    self.assertIn(f"{attn_tp} // {kv}", str(ctx.exception))
                else:
                    _call(dcp, attn_tp, kv)

    def test_zero_kv_heads_does_not_divide_by_zero(self):
        with self.assertRaises(ValueError):
            _call(2, 2, 0)

    # ------------------------------------------------------- default path

    def test_dcp_off_is_inert_under_every_flag_combination(self):
        """dcp_size <= 1 is the default path: nothing may raise there, not even
        with a plan, a token vector and the fast lane all set."""
        for dcp in (0, 1):
            for plan in (False, True):
                for tokens in (False, True):
                    for weightless in (False, True):
                        with self.subTest(
                            dcp=dcp,
                            plan=plan,
                            tokens=tokens,
                            weightless=weightless,
                        ):
                            _call(
                                dcp,
                                3,
                                8,
                                plan=plan,
                                tokens=tokens,
                                weightless=weightless,
                            )

    def test_the_constructor_still_calls_the_rule(self):
        """A guard nobody calls is not a guard. Pins the call site so a
        refactor cannot orphan the rule while every test above stays green.
        """
        import pathlib

        import sglang.srt.layers.attention.triton_backend as tb

        src = pathlib.Path(tb.__file__).read_text()
        self.assertIn("reject_unsupported_dcp_geometry(", src)
        # constructor call site, with all three uneven inputs wired in
        self.assertIn("uneven_plan=uneven_dcp_kv_replicated(self.dcp_size)", src)
        self.assertIn("weighted_tokens=uneven_dcp_active(self.dcp_size)", src)
        self.assertIn("weightless_kv=weightless_kv_active()", src)


if __name__ == "__main__":
    unittest.main()
