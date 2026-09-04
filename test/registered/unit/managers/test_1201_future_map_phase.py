# SPDX-License-Identifier: Apache-2.0
"""#1201 B3 -- the FutureMap is stamped with the BOOT phase and never rebuilt.

CLASS.  Same class as the request-pool holders (test_1201_phase_stamped_handles):
an object built once, at boot, from state that the cutover replaces.  The
FutureMap is the fourth holder of ``ReqToTokenPool`` -- and it carries a SECOND
phase-stamped field the pool holders do not have, ``spec_algo``.

THE TWO STAMPS.

  * ``spec_algo``.  ``Scheduler.__init__`` (scheduler.py:820-823) deliberately
    swaps a phase-flip instance's configured algorithm to NONE, because the boot
    phase is PP prefill and there is no draft worker there (#631).
    ``Scheduler.init_overlap`` then builds the FutureMap out of that NONE
    (scheduler.py:2257, ``self.spec_algorithm.create_future_map(...)``).  The
    cutover swaps the scheduler back to the configured algorithm
    (phase_flip_runtime.py:3412 ``scheduler.spec_algorithm = want_spec_algo``)
    but the FutureMap keeps its boot copy for the life of the process.
  * the request pool.  ``ConfidenceRelay.pool`` (overlap_utils.py:339) and
    ``req_pool_size`` / ``max_context_len`` come from whichever pool existed at
    boot -- the PP one.  #1201's first cut moved three holders and left this
    one, registered as an open gap in ``cutover_participants``.

WHY IT IS LIVE, NOT LATENT.  ``resolve_forward_inputs`` is reached from the
NON-overlap spec branch (scheduler.py:13174-13177), outside the
``if self.enable_overlap:`` at :13081 -- and the standing boot form runs
``disable_overlap_schedule=True`` with ``speculative_algorithm='EAGLE'``.
``overlap_utils.py:107`` tests ``future_map.spec_algo.is_none()``, reads the
frozen NONE and gathers ``output_tokens_buf[batch.req_pool_indices]`` into
``batch.input_ids`` on a batch whose input ids the spec worker owns.

THE DANGEROUS DIRECTION IS SILENT.  The only thing that could shout is
``_assert_nonneg_and_invalidate``, gated on ``SGLANG_IS_IN_CI``
(overlap_utils.py:72), which is OFF on the rig.  So the failure is not a raise:
it is a decode round fed input ids nobody meant to relay.  The test that matters
most here is therefore the one asserting the WRONG BRANCH IS NOT TAKEN, not one
asserting a guard fires.

ORDERING IS PART OF THE CUT.  The rebuild has to run after BOTH swaps: the
spec-algorithm swap at phase_flip_runtime.py:3412 and the request-pool rebind at
:3746.  A rebuild placed between them would carry the incoming algorithm and the
OUTGOING pool -- half a fix that still reads the wrong tensor.  One test pins
that source order.

Hermetic: CPU tensors, no accelerator, no scheduler process.
"""

from __future__ import annotations

import types
import unittest

import torch

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.test_utils import CustomTestCase

NONE = SpeculativeAlgorithm.from_string(None)
EAGLE = SpeculativeAlgorithm.from_string("EAGLE")


def _pool(size=4, ctx=8):
    return ReqToTokenPool(
        size=size, max_context_len=ctx, device="cpu", enable_memory_saver=False
    )


def _backend(needs_cpu_seq_lens=True):
    return types.SimpleNamespace(needs_cpu_seq_lens=needs_cpu_seq_lens)


def _scheduler(pool, algo, draft_worker=None, tp_backend=None):
    """Exactly the surface ``Scheduler.init_overlap`` reads to build the map."""
    return types.SimpleNamespace(
        device=torch.device("cpu"),
        spec_algorithm=algo,
        req_to_token_pool=pool,
        draft_worker=draft_worker,
        tp_worker=types.SimpleNamespace(
            model_runner=types.SimpleNamespace(
                attn_backend=tp_backend if tp_backend is not None else _backend()
            )
        ),
        server_args=types.SimpleNamespace(
            enable_two_batch_overlap=False,
            speculative_algorithm="EAGLE",
        ),
    )


def _decode_batch(indices, algo):
    """A non-overlap spec decode batch: the shape scheduler.py:13174 resolves."""
    return types.SimpleNamespace(
        prefill_input_ids_cpu=None,
        mix_running_indices=None,
        input_ids=None,
        req_pool_indices=torch.tensor(indices, dtype=torch.int64),
        device=torch.device("cpu"),
        enable_overlap=False,
        spec_algorithm=algo,
    )


class TestFutureMapIsBuiltForThePhase(CustomTestCase):
    """The builder exists, is callable outside ``Scheduler``, and reads NOW."""

    def test_the_construction_is_extractable_from_init_overlap(self):
        from sglang.srt.managers.overlap_utils import build_future_map

        fm = build_future_map(_scheduler(_pool(), EAGLE))
        self.assertIs(fm.spec_algo, EAGLE)

    def test_the_map_carries_the_schedulers_algorithm_at_call_time(self):
        from sglang.srt.managers.overlap_utils import build_future_map

        sched = _scheduler(_pool(), NONE)
        self.assertTrue(build_future_map(sched).spec_algo.is_none())
        sched.spec_algorithm = EAGLE
        self.assertFalse(build_future_map(sched).spec_algo.is_none())

    def test_the_map_names_the_schedulers_pool_at_call_time(self):
        from sglang.srt.managers.overlap_utils import build_future_map

        pp_pool, tp_pool = _pool(size=4), _pool(size=4)
        sched = _scheduler(pp_pool, NONE)
        self.assertIs(build_future_map(sched).confidence_relay.pool, pp_pool)
        sched.req_to_token_pool = tp_pool
        self.assertIs(build_future_map(sched).confidence_relay.pool, tp_pool)

    def test_the_draft_workers_backends_decide_when_one_exists(self):
        """Parity with scheduler.py:2248-2254: the draft worker's declared
        backends win over the target's, and a worker without the override falls
        back to target-only."""
        from sglang.srt.managers.overlap_utils import build_future_map

        draft = types.SimpleNamespace(
            spec_v2_attn_backends=(_backend(needs_cpu_seq_lens=False),)
        )
        sched = _scheduler(_pool(), EAGLE, draft_worker=draft, tp_backend=_backend(True))
        self.assertFalse(build_future_map(sched).needs_cpu_seq_lens)

        unaudited = types.SimpleNamespace()
        sched2 = _scheduler(
            _pool(), EAGLE, draft_worker=unaudited, tp_backend=_backend(True)
        )
        self.assertTrue(build_future_map(sched2).needs_cpu_seq_lens)


class TestThePpMapIsNotTheMapTheTpPhaseConsults(CustomTestCase):

    def test_the_cutover_produces_a_different_object(self):
        from sglang.srt.managers.overlap_utils import build_future_map

        pp_pool, tp_pool = _pool(), _pool()
        sched = _scheduler(pp_pool, NONE)
        boot_map = build_future_map(sched)

        # what the cutover does, in the order phase_flip_runtime does it:
        # spec swap (:3412) then request-pool rebind (:3746).
        sched.spec_algorithm = EAGLE
        sched.req_to_token_pool = tp_pool
        tp_map = build_future_map(sched)

        self.assertIsNot(tp_map, boot_map)
        self.assertTrue(boot_map.spec_algo.is_none())
        self.assertFalse(tp_map.spec_algo.is_none())
        self.assertIs(tp_map.confidence_relay.pool, tp_pool)

    def test_the_frozen_none_takes_the_branch_it_must_not(self):
        """THE DANGEROUS DIRECTION, on the live non-overlap spec path.

        With the boot map, ``resolve_forward_inputs`` believes the server is
        non-speculative and relays last iteration's sampled token into
        ``batch.input_ids``.  With the phase's own map it leaves them alone --
        the V2 worker owns them.  Nothing raises in either case.
        """
        from sglang.srt.managers.overlap_utils import (
            build_future_map,
            resolve_forward_inputs,
        )

        pp_pool, tp_pool = _pool(), _pool()
        sched = _scheduler(pp_pool, NONE)
        boot_map = build_future_map(sched)
        boot_map.output_tokens_buf.fill_(7)

        frozen_batch = _decode_batch([1, 2], EAGLE)
        resolve_forward_inputs(frozen_batch, boot_map)
        self.assertIsNotNone(
            frozen_batch.input_ids,
            "premise check: the frozen-NONE map must take the gather branch, "
            "or this test proves nothing about the fix",
        )

        sched.spec_algorithm = EAGLE
        sched.req_to_token_pool = tp_pool
        tp_map = build_future_map(sched)
        tp_map.output_tokens_buf.fill_(7)

        phase_batch = _decode_batch([1, 2], EAGLE)
        resolve_forward_inputs(phase_batch, tp_map)
        self.assertIsNone(
            phase_batch.input_ids,
            "a map stamped with the phase's own algorithm must leave the spec "
            "worker's input ids alone",
        )

    def test_a_non_spec_batch_still_gathers_through_the_phase_map(self):
        """Default-path parity: a genuinely non-speculative phase must keep
        relaying.  The cut must not turn the gather off, only stop it from
        firing under a stale stamp."""
        from sglang.srt.managers.overlap_utils import (
            build_future_map,
            resolve_forward_inputs,
        )

        sched = _scheduler(_pool(), NONE)
        fm = build_future_map(sched)
        fm.output_tokens_buf.fill_(7)
        batch = _decode_batch([1, 2], NONE)
        resolve_forward_inputs(batch, fm)
        self.assertIsNotNone(batch.input_ids)
        self.assertTrue(bool((batch.input_ids == 7).all()))


class TestTheIdentityAssertion(CustomTestCase):
    """The registry's second obligation: a probe that the hook actually ran."""

    def test_a_stale_algorithm_stamp_is_refused(self):
        from sglang.srt.managers.overlap_utils import (
            FutureMapPhaseMismatch,
            assert_future_map_identity,
            build_future_map,
        )

        sched = _scheduler(_pool(), NONE)
        sched.future_map = build_future_map(sched)
        sched.spec_algorithm = EAGLE  # the cutover swapped; nothing rebuilt
        with self.assertRaises(FutureMapPhaseMismatch):
            assert_future_map_identity(sched)

    def test_a_stale_pool_stamp_is_refused(self):
        from sglang.srt.managers.overlap_utils import (
            FutureMapPhaseMismatch,
            assert_future_map_identity,
            build_future_map,
        )

        sched = _scheduler(_pool(), EAGLE)
        sched.future_map = build_future_map(sched)
        sched.req_to_token_pool = _pool()  # the rebind moved; nothing rebuilt
        with self.assertRaises(FutureMapPhaseMismatch):
            assert_future_map_identity(sched)

    def test_a_rebuilt_map_passes(self):
        from sglang.srt.managers.overlap_utils import (
            assert_future_map_identity,
            build_future_map,
        )

        sched = _scheduler(_pool(), NONE)
        sched.future_map = build_future_map(sched)
        sched.spec_algorithm = EAGLE
        sched.req_to_token_pool = _pool()
        sched.future_map = build_future_map(sched)
        assert_future_map_identity(sched)

    def test_the_stamp_arms_cannot_fire_where_the_probe_actually_runs(self):
        """RECORDED, not asserted away: at the seam both stamp arms are
        construction invariants.

        ``build_future_map`` hands ``scheduler.spec_algorithm`` in as the
        stamp and ``scheduler.req_to_token_pool`` straight into the relay, and
        the probe runs 57 lines after that rebuild with nothing in between
        touching either field. The two tests above hand-mutate a map that no
        code path produces, which is why they are green while the probe fires
        on nothing. This pins the fact so the next reader does not mistake the
        green for coverage -- and it is the reason `previous` exists.
        """
        from sglang.srt.managers.overlap_utils import (
            FutureMapPhaseMismatch,
            assert_future_map_identity,
            build_future_map,
        )

        fires = 0
        for algo in (NONE, EAGLE):
            for _ in range(2):
                sched = _scheduler(_pool(), algo)
                sched.future_map = build_future_map(sched)
                try:
                    assert_future_map_identity(sched)
                except FutureMapPhaseMismatch:  # pragma: no cover - the point
                    fires += 1
        self.assertEqual(0, fires)

    def test_a_rebuild_that_did_not_replace_the_map_is_refused(self):
        """THE ARM THAT CAN FAIL AT THE SEAM.

        A memoised ``build_future_map``, a deleted assignment, or a rebuild
        moved out of the cutover all leave the OUTGOING phase's object in
        place -- and the stamp arms cannot see any of them, because a map
        built from the scheduler's fields always matches the scheduler's
        fields. The object identity is the only seam-time evidence.
        """
        from sglang.srt.managers.overlap_utils import (
            FutureMapPhaseMismatch,
            assert_future_map_identity,
            build_future_map,
        )

        sched = _scheduler(_pool(), NONE)
        sched.future_map = build_future_map(sched)
        stale = sched.future_map
        sched.spec_algorithm = EAGLE
        sched.req_to_token_pool = _pool()
        # the cutover swapped both handles and the rebuild did NOT run
        with self.assertRaises(FutureMapPhaseMismatch):
            assert_future_map_identity(sched, previous=stale)
        # and the real rebuild passes the same probe
        sched.future_map = build_future_map(sched)
        assert_future_map_identity(sched, previous=stale)

    def test_the_seam_hands_the_probe_the_outgoing_map(self):
        """Can-fail anchor: without `previous=_old_map` at the call site the
        arm above is unreachable in production."""
        import pathlib

        src = pathlib.Path(
            "/spinning/wt-weg1/python/sglang/srt/managers/phase_flip_runtime.py"
        ).read_text()
        self.assertIn(
            "assert_future_map_identity(scheduler, previous=_old_map)",
            src,
            "the seam calls the probe without the only argument that can fail",
        )

    def test_a_scheduler_without_a_future_map_is_not_an_error(self):
        """Default-path parity, duck-typed like assert_req_pool_identity: an
        instance that never built one is not a divergence."""
        from sglang.srt.managers.overlap_utils import assert_future_map_identity

        assert_future_map_identity(types.SimpleNamespace())
        assert_future_map_identity(types.SimpleNamespace(future_map=None))


class TestTheSeamDeclaresIt(CustomTestCase):
    """#859: what this list forgets, a boot finds."""

    def test_the_holder_is_no_longer_an_open_gap(self):
        from sglang.srt.managers.cutover_participants import REGISTRY

        entry = next(p for p in REGISTRY if p.name == "future_map_req_pool_holder")
        self.assertIsNotNone(entry.hook, "the rebuild is the hook")
        self.assertIsNotNone(entry.probe, "the identity assertion is the probe")
        self.assertIsNone(entry.gap)

    def test_the_future_map_is_declared_mutated_state(self):
        from sglang.srt.managers.cutover_participants import (
            MUTATED_STATE,
            ReadWindow,
        )

        self.assertEqual(
            MUTATED_STATE.get("future_map"), ReadWindow.OUTSIDE_CUTOVER
        )

    def test_the_rebuild_runs_after_both_swaps(self):
        """ORDERING.  The map must be built after the spec swap AND after the
        request-pool rebind, or it carries the outgoing pool.  Source order is
        the only desk-checkable proof of that."""
        import pathlib

        src = pathlib.Path(
            "/spinning/wt-weg1/python/sglang/srt/managers/phase_flip_runtime.py"
        ).read_text()
        swap = src.index("scheduler.spec_algorithm = want_spec_algo")
        rebind = src.index('rebind_req_pool_for_cutover(scheduler, "tp" if tp_phase')
        build = src.index("scheduler.future_map = build_future_map(scheduler)")
        self.assertLess(swap, build, "built before the algorithm was swapped")
        self.assertLess(rebind, build, "built before the pool was rebound")


if __name__ == "__main__":
    unittest.main()
