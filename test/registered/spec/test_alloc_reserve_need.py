"""Task #486: the per-decode KV reserve must equal the DERIVED need.

Two falsifier directions, both required:

  (a) OVER-reservation. The reserve must be exactly ``write footprint +
      commit lag``. The blanket ``2 * get_alloc_len_per_decode()`` that this
      replaces is ``W + W``; every row below whose ``pre_fix`` differs from
      ``expect`` is red against that old form.

  (b) UNDER-reservation. A synthetic worst-case verify burst -- the in-flight
      verify commits its maximum, then this step's draft+verify write their
      maximum -- must still land inside the reserve. Anyone who later shaves
      the formula below the need trips ``test_shaved_reserve_is_caught``,
      which proves this direction can actually fail.

Desk test: pure arithmetic over ServerArgs, no device.
"""

import unittest

from sglang.srt.mem_cache.common import (
    get_alloc_len_per_decode,
    get_alloc_reserve_per_decode,
    get_commit_lag_per_decode,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _sa(
    *,
    algorithm=None,
    steps=None,
    topk=None,
    draft_tokens=None,
    page_size=1,
    overlap=True,
) -> ServerArgs:
    """A ServerArgs carrying only the fields the reserve derivation reads.

    Built field-by-field rather than through the CLI so the table stays a
    statement about the formula, not about arg-resolution side effects.
    ``max_speculative_num_draft_tokens`` is a cached_property -- nothing may
    read it before the fields below are in place.
    """
    sa = ServerArgs(model_path="dummy")
    sa.speculative_algorithm = algorithm
    sa.speculative_num_steps = steps
    sa.speculative_eagle_topk = topk
    sa.speculative_num_draft_tokens = draft_tokens
    sa.speculative_adaptive = False
    sa.speculative_cross_algorithm = False
    sa.page_size = page_size
    sa.disable_overlap_schedule = not overlap
    return sa


# name -> (kwargs, expected reserve, reserve the old blanket 2x produced)
#
# W = get_alloc_len_per_decode, L = get_commit_lag_per_decode, expect = W + L.
_CASES = {
    # --- our NEXTN production recipe: chain draft, topk=1, D=steps+1 --------
    # W = max(3*1, 4) = 4, L = 4  ->  8. The old 2*W was ALSO 8: on this shape
    # the blanket 2x coincides with the derived need. Asserted, not assumed --
    # this row is the honest "no saving on the production path" statement and
    # it must stay green, i.e. the fix must not shave the production reserve.
    "nextn_topk1_s3_d4_overlap": (
        dict(algorithm="EAGLE", steps=3, topk=1, draft_tokens=4),
        8,
        8,
    ),
    # Same recipe with overlap off: kv_committed_len is exact at prepare time
    # (process_batch_result runs before the next prepare), so L collapses to 0.
    "nextn_topk1_s3_d4_no_overlap": (
        dict(algorithm="EAGLE", steps=3, topk=1, draft_tokens=4, overlap=False),
        4,
        8,
    ),
    # Deeper chain than the verify width: W is driven by the draft writes
    # (steps), L by the accept run (D). The two terms genuinely differ.
    "chain_steps_exceed_draft_tokens": (
        dict(algorithm="EAGLE", steps=6, topk=1, draft_tokens=4),
        10,
        12,
    ),
    # --- topk>1 trees: W = topk*steps, L = D -------------------------------
    "tree_topk4_s3_d8_page1": (
        dict(algorithm="EAGLE", steps=3, topk=4, draft_tokens=8),
        20,
        24,
    ),
    # page>1 + topk>1: per-branch page-aligned draft footprint. Largest gap.
    # W = ceil((64-1+3)/64)*64*4 = 512, L = 8.
    "tree_topk4_s3_d8_page64": (
        dict(algorithm="EAGLE", steps=3, topk=4, draft_tokens=8, page_size=64),
        520,
        1024,
    ),
    # --- no spec: W = 1, L = 1. The 2x was right here too, by coincidence. --
    "no_spec_overlap": (dict(), 2, 2),
    "no_spec_no_overlap": (dict(overlap=False), 1, 2),
}


class TestDerivedReserveIsNotABlanketDouble(CustomTestCase):
    """Direction (a): no over-reservation."""

    def test_reserve_equals_write_footprint_plus_commit_lag(self):
        for name, (kwargs, expect, _pre_fix) in _CASES.items():
            with self.subTest(case=name):
                sa = _sa(**kwargs)
                write_footprint = get_alloc_len_per_decode(sa)
                commit_lag = get_commit_lag_per_decode(sa)
                self.assertEqual(
                    get_alloc_reserve_per_decode(sa),
                    write_footprint + commit_lag,
                    f"{name}: reserve must be the sum of the two derived terms",
                )
                self.assertEqual(get_alloc_reserve_per_decode(sa), expect, name)

    def test_blanket_double_is_the_thing_being_replaced(self):
        """Every table row's ``pre_fix`` is what ``2 * W`` yields.

        Rows where pre_fix > expect are red against the old form: that is the
        over-reservation this task removes. Rows where they are equal are the
        honest coincidences (W == L) and are called out as such.
        """
        tightened = []
        for name, (kwargs, expect, pre_fix) in _CASES.items():
            with self.subTest(case=name):
                sa = _sa(**kwargs)
                self.assertEqual(
                    2 * get_alloc_len_per_decode(sa),
                    pre_fix,
                    f"{name}: table's pre_fix must be the old blanket 2x",
                )
                self.assertLessEqual(
                    get_alloc_reserve_per_decode(sa),
                    pre_fix,
                    f"{name}: the fix must never reserve MORE than the old 2x",
                )
                if pre_fix > expect:
                    tightened.append(name)
        self.assertIn("tree_topk4_s3_d8_page64", tightened)
        self.assertIn("nextn_topk1_s3_d4_no_overlap", tightened)

    def test_commit_lag_uses_the_widest_rung_not_the_active_one(self):
        """Adaptive / cross-algo: the in-flight verify may be a WIDER rung.

        The lag ceiling must follow ``max_speculative_num_draft_tokens``, so a
        narrow active rung cannot under-reserve against a wide in-flight one.
        """
        sa = _sa(algorithm="EAGLE", steps=3, topk=1, draft_tokens=4)
        sa.speculative_adaptive = True
        sa.adaptive_max_candidate_steps = 7
        self.assertEqual(sa.max_speculative_num_draft_tokens, 8)
        self.assertEqual(get_commit_lag_per_decode(sa), 8)
        self.assertGreaterEqual(get_alloc_reserve_per_decode(sa), 8 + 8)


def _worst_case_trace(sa, num_steps=64):
    """Replay the overlap watermark recurrence under a worst-case accept run.

    Mirrors the scheduler ordering established in ``event_loop_overlap``:
    ``prepare_for_decode`` (which sets the watermark from the host-visible
    ``kv_committed_len``) runs BEFORE ``run_batch``'s ``resolve_seq_lens_cpu``
    and before ``pop_and_process`` -- so at prepare time the host is one verify
    behind the device.

    Yields ``(allocated, device_write_end)`` per step.
    """
    reserve = get_alloc_reserve_per_decode(sa)
    write_footprint = get_alloc_len_per_decode(sa)
    max_accept = get_commit_lag_per_decode(sa)

    committed_host = 100  # post-prefill; any base works
    committed_device = committed_host
    allocated = committed_host

    for _ in range(num_steps):
        # prepare_for_decode: watermark from the STALE host committed length.
        allocated = max(allocated, committed_host + reserve)
        # the in-flight verify lands: the device is now this far ahead.
        committed_device = committed_host + max_accept
        # this step's draft+verify write from the device length outward.
        yield allocated, committed_device + write_footprint
        # ...and the host catches up by exactly that verify.
        committed_host = committed_device


class TestReserveCoversWorstCaseVerifyBurst(CustomTestCase):
    """Direction (b): no UNDER-reservation. Correctness first."""

    def test_worst_case_burst_fits_the_reserve(self):
        for name, (kwargs, _expect, _pre_fix) in _CASES.items():
            with self.subTest(case=name):
                sa = _sa(**kwargs)
                for step, (allocated, write_end) in enumerate(_worst_case_trace(sa)):
                    self.assertGreaterEqual(
                        allocated,
                        write_end,
                        f"{name} step {step}: verify would write past "
                        f"kv_allocated_len ({write_end} > {allocated}) -- the "
                        f"reserve UNDER-reserves. Do not shave it.",
                    )

    def test_shaved_reserve_is_caught(self):
        """The can-fail proof: one token less than the derived need is unsafe.

        Without this, the guard above would pass vacuously for any formula that
        happens to be generous. Every shape must be genuinely tight.
        """
        for name, (kwargs, _expect, _pre_fix) in _CASES.items():
            with self.subTest(case=name):
                sa = _sa(**kwargs)
                reserve = get_alloc_reserve_per_decode(sa) - 1
                write_footprint = get_alloc_len_per_decode(sa)
                max_accept = get_commit_lag_per_decode(sa)

                committed_host = 100
                allocated = committed_host
                violated = False
                for _ in range(8):
                    allocated = max(allocated, committed_host + reserve)
                    committed_device = committed_host + max_accept
                    if allocated < committed_device + write_footprint:
                        violated = True
                        break
                    committed_host = committed_device
                self.assertTrue(
                    violated,
                    f"{name}: shaving the reserve by one token was NOT caught -- "
                    f"the under-reservation falsifier is not actually binding",
                )

    def test_dropping_the_commit_lag_term_under_reserves(self):
        """Upstream #32574's shape (1x, no lag term) must be rejected here.

        In this tree ``batch.seq_lens_cpu`` is resolved in ``run_batch``, i.e.
        after ``prepare_for_decode``, so it carries the same one-verify
        staleness as ``kv_committed_len``; a 1x reserve therefore under-reserves
        by the accept run whenever overlap is on.
        """
        for name, (kwargs, _expect, _pre_fix) in _CASES.items():
            if not kwargs.get("overlap", True):
                continue
            with self.subTest(case=name):
                sa = _sa(**kwargs)
                reserve = get_alloc_len_per_decode(sa)  # the 1x form
                write_footprint = get_alloc_len_per_decode(sa)
                max_accept = get_commit_lag_per_decode(sa)
                allocated = max(100, 100 + reserve)
                self.assertLess(
                    allocated,
                    100 + max_accept + write_footprint,
                    f"{name}: 1x would have been safe here -- re-derive before "
                    f"claiming upstream #32574 under-reserves on this shape",
                )


if __name__ == "__main__":
    unittest.main()
