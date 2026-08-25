"""#783 half 1: the CUTOVER retraction copies state; ordinary retraction does not.

THE TRIGGER IS EXACTLY ONE SITE, and this file exists to keep it that way,
because the host-cost calculation depends on it.

`retract_all` has exactly two callers in the tree:

    ScheduleBatch.retract_all (schedule_batch.py:3067)
        <- scheduler.py:11766, the DECODE-PRESSURE path
    build_cutover_release._retract (phase_flip_runtime.py:1421)
        <- the #856 cutover, and nothing else

The copy belongs to the second and must never fire on the first. The budget
computed before the boot -- 0.585 GiB per cutover, ~57 MiB/s aggregate
device->host -- is priced at the FLIP CADENCE (5.7 flips/min measured in
W37-H arm A). Pressure retractions have an entirely different and
load-dependent rate, so a copy that also fired there would not make the
estimate conservative, it would make it wrong.

"UNGATED BY DESIGN" MEANS NO FLAG. It does not mean no condition. The
condition is: *this retraction is the #856 cutover's*, expressed as a
parameter that only the seam sets, defaulting to False for every other caller.
No server arg, no env var, nothing a deployment can flip.

FIXTURE SHAPE, FOLLOWING REPO PRECEDENT. `release_req` needs real pools --
`test_uniform_retract_count_583.py:105` says so in as many words and fakes it
for exactly this reason. So this file pins the two things that do NOT need
pools: that `retract_all` THREADS the flag to `release_req`, and that only the
cutover sets it. That `Req.offload_kv_cache` then copies the mamba state is
pinned separately, on real code, in test_unified_mamba_cpu_copy_783.

WHY A COPY AND NOT AN INSERT. W37-H arm B tried to persist at this same
instant by INSERTING into the tree, i.e. by transferring row ownership, and
died 33 s in with `pool memory leak detected!` -- 22 rows claimed twice,
because `readmit_seam_residents` brings the population straight back and it
resumes on rows the tree had taken. A copy leaves ownership untouched, which
is what `test_seam_ownership_ledger_783` pins.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest

import torch

from sglang.test.test_utils import CustomTestCase

MAMBA_SLOT = torch.tensor([3], dtype=torch.int64)


class _Recorder:
    """Captures what `retract_all` hands to `release_req`, which is the only
    thing this file needs to see."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)


def _run(monkey, **kw):
    import sglang.srt.managers.schedule_batch as sb

    rec = _Recorder()
    orig = sb.release_req
    sb.release_req = rec
    try:
        sb.retract_all(
            reqs=[object()],
            server_args=types.SimpleNamespace(disaggregation_mode="null"),
            req_to_token_pool=object(),
            token_to_kv_pool_allocator=object(),
            tree_cache=object(),
            hisparse_coordinator=None,
            **kw,
        )
    finally:
        sb.release_req = orig
    return rec


class TestTheCutoverFlagIsThreaded(CustomTestCase):
    def test_cutover_retraction_passes_copy_state(self):
        """RED until half 1 lands."""
        rec = _run(None, copy_state=True)
        self.assertEqual(len(rec.calls), 1)
        self.assertTrue(
            rec.calls[0].get("copy_state"),
            "the #856 cutover must ask release_req to copy the state before "
            "releasing; otherwise the next phase has no GDN state to resume",
        )

    def test_the_seam_is_the_only_caller_that_sets_it(self):
        """STRUCTURAL, and it is what keeps the host budget honest.

        `retract_all` has exactly two callers. Only the cutover may set the
        flag; the decode-pressure path must not, because its rate is
        load-dependent and outside the flip-cadence budget."""
        import inspect

        from sglang.srt.managers import phase_flip_runtime

        cutover = inspect.getsource(phase_flip_runtime.build_cutover_release)
        self.assertIn("copy_state=True", cutover)

        import sglang.srt.managers.schedule_batch as sb

        pressure = inspect.getsource(sb.ScheduleBatch.retract_all)
        self.assertNotIn("copy_state=True", pressure)


class TestOrdinaryRetractionIsUnchanged(CustomTestCase):
    """The controls. Both pass TODAY and must keep passing."""

    def test_default_is_no_copy(self):
        rec = _run(None)
        self.assertEqual(len(rec.calls), 1)
        self.assertFalse(
            rec.calls[0].get("copy_state", False),
            "a decode-pressure retraction must not pay a device->host copy",
        )

    def test_retract_all_still_returns_its_reqs(self):
        import sglang.srt.managers.schedule_batch as sb

        rec = _Recorder()
        orig = sb.release_req
        sb.release_req = rec
        try:
            reqs = [object(), object()]
            out = sb.retract_all(
                reqs=reqs,
                server_args=types.SimpleNamespace(disaggregation_mode="null"),
                req_to_token_pool=object(),
                token_to_kv_pool_allocator=object(),
                tree_cache=object(),
                hisparse_coordinator=None,
            )
        finally:
            sb.release_req = orig
        self.assertEqual(out, reqs)
        self.assertEqual(len(rec.calls), 2)


if __name__ == "__main__":
    unittest.main()
