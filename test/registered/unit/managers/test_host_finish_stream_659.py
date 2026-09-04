"""C28 (#659): a session that finishes ON HOST must still reach the client.

Red-first falsifier for the alias booked as C28 in
``docs/dev/631/CONTRADICTIONS_REGISTER.md`` and measured on metal by
successor 44 (``/spinning/evidence-631/s44/WEDGE_AFTER_UNPARK.txt``): a
spilled session parked, unparked, finished on host -- and the HTTP caller
never received a byte, blocking forever against a scheduler at
``#running-req: 0``.

THE MECHANISM. ``process_batch_result_decode`` iterates ``batch.reqs``; for a
request with ``kv_spill_state == "host"`` the loop calls
``kv_session_offload.release_finished_spilled_req(req)``, which calls
``slot.batch.filter_batch()``. On a spill tick ``batch`` IS ``slot.batch``
(one object, two names), ``filter_batch`` REBINDS ``self.reqs`` (both its
branches rebind -- ``schedule_batch.py`` empty-branch and list-comprehension
branch alike), and the ``stream_output(batch.reqs, ...)`` at the end of the
method then reads the NEW list, which no longer holds the finished request.
The request deleted its own completion from the list that was about to carry
it.

These tests drive the REAL ``SchedulerBatchResultProcessor.
process_batch_result_decode`` over the REAL ``KVSessionOffloadManager.
release_finished_spilled_req`` and the REAL ``ScheduleBatch.filter_batch``.
Only concerns unrelated to the alias (logprobs, mamba, reasoning tokens,
metrics) are stubbed. On unfixed code test 1 and test 2 FAIL with an empty /
truncated streamer list -- that is the red.
"""

import types
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


class _RecordingStreamer:
    """Stands in for the scheduler's output streamer and records exactly what
    the terminal emit would have carried to the detokenizer."""

    def __init__(self):
        self.calls = []

    def stream_output(self, reqs, return_logprob, *args, **kwargs):
        # list() so a later rebind/mutation cannot rewrite what we recorded.
        self.calls.append((list(reqs), return_logprob))


class _FakeReq:
    """The subset of ``Req`` the decode result path touches."""

    def __init__(self, rid, rpi, *, finished, spilled, return_logprob=False):
        self.rid = rid
        self.req_pool_idx = rpi
        self._finished = finished
        self.kv_spill_state = "host" if spilled else None
        self.kv_spill_boundary = 0  # no device head -> no allocator traffic
        self.cache_protected_len = 0
        self.last_node = None  # no tree lock
        self.mamba_pool_idx = None  # no mamba state
        self.output_ids = []
        self.return_logprob = return_logprob
        self.return_hidden_states = False
        self.grammar = None
        self.multimodal_inputs = None
        self.session = None
        self.is_retracted = False
        self.finished_reason = None
        self.time_stats = MagicMock()

    def finished(self):
        return self._finished

    def update_finish_state(self, new_accept_len):
        # The real one promotes to_finish / stop conditions; this fixture
        # decides finishedness up front, so the state is already correct.
        return None


def _make_batch(reqs, *, return_logprob=False):
    """A real ScheduleBatch carrying a real ``filter_batch``.

    Built with ``__new__`` on purpose: ``filter_batch`` with no arguments only
    reads ``self.reqs`` (and the lockstep sentinel, unarmed here), so the
    fields a forward pass would need are irrelevant and constructing them
    would need a GPU.
    """
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    batch = ScheduleBatch.__new__(ScheduleBatch)
    batch.reqs = list(reqs)
    batch.return_logprob = return_logprob
    batch.spec_algorithm = SpeculativeAlgorithm.NONE
    return batch


class _RebindingBatch:
    """Stands in for ScheduleBatch in the PARTIAL-survivor case only.

    The real ``filter_batch`` survivor branch rebuilds CUDA tensors
    (``keep_indices`` is moved to ``self.device``), so it cannot run on this
    CPU lane. This class reproduces the two rebinds the real one performs and
    nothing else. ``test_filter_batch_rebinds_are_still_what_this_file_assumes``
    fails the moment those rebinds change, so the stand-in cannot quietly drift
    away from the code it stands for.

    In production ``slot.batch`` is a bs=1 batch, so the real path takes the
    empty branch -- which test 1 exercises against the REAL ScheduleBatch.
    """

    def __init__(self, reqs, return_logprob):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        self.reqs = list(reqs)
        self.return_logprob = return_logprob
        self.spec_algorithm = SpeculativeAlgorithm.NONE

    def filter_batch(self):
        keep = [i for i in range(len(self.reqs)) if not self.reqs[i].finished()]
        self.reqs = [self.reqs[i] for i in keep]
        self.return_logprob = any(r.return_logprob for r in self.reqs)


def _make_manager(batch):
    """A KVSessionOffloadManager with just the attributes the host-finish
    release path touches, and its slot aliased to ``batch`` -- which is the
    spill-tick shape: ``maybe_take_tick`` hands the scheduler the persistent
    batch, so ``slot.batch is batch``."""
    from sglang.srt.managers.kv_session_offload import (
        KVSessionOffloadManager,
        RestoreHysteresis,
        SpillSlot,
        WaveBackController,
    )

    mgr = KVSessionOffloadManager.__new__(KVSessionOffloadManager)
    mgr.spills = {}
    mgr._free_regions = []
    mgr.backend = MagicMock()
    mgr.backend._sess_copy_stream = None  # CPU path: no stream ordering
    mgr.scheduler = types.SimpleNamespace(
        running_batch=types.SimpleNamespace(batch_is_full=True)
    )
    mgr._fast_lane_enabled = False
    mgr._iter_ct = 0
    mgr.tick_controller = None
    mgr._log = lambda *a, **k: None
    mgr.allocator = MagicMock()
    mgr.tree_cache = MagicMock()

    def _free(r):
        r.req_pool_idx = None

    # #1040 C1.5: the manager reads `scheduler.req_to_token_pool` at use,
    # so the pool lives on the scheduler stand-in, not on the manager.
    mgr.scheduler.req_to_token_pool = MagicMock()
    mgr.req_to_token_pool.free.side_effect = _free
    return mgr


def _install_slot(mgr, req, batch, region=0):
    from sglang.srt.managers.kv_session_offload import (
        RestoreHysteresis,
        SpillSlot,
        WaveBackController,
    )

    slot = SpillSlot(
        req=req,
        region=region,
        spill_iter=0,
        wave=WaveBackController(8, 1),
        hysteresis=RestoreHysteresis(1),
    )
    slot.batch = batch  # THE ALIAS: slot.batch is the batch being processed
    mgr.spills[req.req_pool_idx] = slot
    return slot


def _make_processor(mgr, streamer):
    """The REAL processor class, subclassed only to neutralise concerns that
    are orthogonal to the alias (logprobs, mamba, reasoning tokens, weightless
    lane). It is a frozen, SLOTTED dataclass, so per-instance method stubs are
    impossible -- the subclass is the only seam."""
    from sglang.srt.managers.scheduler_components.batch_result_processor import (
        DisaggregationMode,
        SchedulerBatchResultProcessor,
    )

    class _ProbeProcessor(SchedulerBatchResultProcessor):
        def _is_weightless_worker(self):
            return False

        def _normalize_decode_outputs(self, **kw):
            return kw["next_token_ids"], None

        def _maybe_update_reasoning_tokens(self, *a, **k):
            return None

        def _mamba_prefix_cache_update(self, *a, **k):
            return None

        def _maybe_collect_routed_experts(self, *a, **k):
            return None

        def _maybe_collect_indexer_topk(self, *a, **k):
            return None

        def _apply_decode_logprobs(self, **kw):
            return None

    return _ProbeProcessor(
        is_generation=True,
        disaggregation_mode=DisaggregationMode.NULL,
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=types.SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False,
            enable_metrics=False,
            enable_hisparse=False,
        ),
        model_config=types.SimpleNamespace(is_encoder_decoder=False),
        token_to_kv_pool_allocator=MagicMock(),
        tree_cache=MagicMock(),
        hisparse_coordinator=None,
        req_to_token_pool=MagicMock(),
        decode_offload_manager=None,
        metrics_collector=MagicMock(),
        metrics_reporter=types.SimpleNamespace(
            num_generated_tokens=0,
            gen_tokens_total=0,
            forward_ct_decode=0,
            report_decode_stats=lambda *a, **k: None,
            update_spec_metrics=lambda *a, **k: None,
        ),
        draft_worker=None,
        model_worker=types.SimpleNamespace(),
        logprob_result_processor=MagicMock(),
        output_streamer=streamer,
        abort_request=lambda *a, **k: None,
        kv_session_offload=mgr,
    )


def _make_result(n_reqs):
    return types.SimpleNamespace(
        copy_done=None,
        routed_experts_output=None,
        indexer_topk_output=None,
        logits_output=None,
        next_token_ids=[[100 + i] for i in range(n_reqs)],
        can_run_cuda_graph=False,
        num_correct_drafts=0,
        speculative_num_draft_tokens=None,
        num_block_accept_tokens=None,
        num_cap_tokens=None,
    )


# ---------------------------------------------------------------------------
# the falsifier
# ---------------------------------------------------------------------------


def test_filter_batch_rebinds_are_still_what_this_file_assumes():
    """Attribution guard for _RebindingBatch: if the real filter_batch stops
    rebinding `reqs` / `return_logprob`, or stops keying on `finished()`, the
    stand-in used by the sibling test is no longer a stand-in for anything and
    this test says so before that test can pass on a false model."""
    import inspect

    from sglang.srt.managers.schedule_batch import ScheduleBatch

    src = inspect.getsource(ScheduleBatch.filter_batch)
    assert "not self.reqs[i].finished()" in src, "keep-criterion changed"
    assert "self.reqs = []" in src, "empty branch no longer rebinds"
    assert "self.reqs = [self.reqs[i] for i in keep_indices]" in src, (
        "survivor branch no longer rebinds"
    )
    assert "self.return_logprob = any(" in src, "return_logprob no longer recomputed"


def test_host_finish_reaches_the_streamer_when_it_is_the_only_req():
    """THE RED TEST (C28). One spilled request, finishing on host, on a
    spill-tick batch aliased to its slot. ``filter_batch`` empties the list;
    the streamer must still receive the request, or the caller hangs forever
    exactly as ``rid=s44-sat-3`` did on metal."""
    req = _FakeReq("s45-host-finish", rpi=5, finished=True, spilled=True)
    batch = _make_batch([req])
    mgr = _make_manager(batch)
    _install_slot(mgr, req, batch)
    streamer = _RecordingStreamer()
    proc = _make_processor(mgr, streamer)

    proc.process_batch_result_decode(batch, _make_result(1))

    # the release really did run (this is the aliasing precondition, not a
    # side assertion: without it the test would pass vacuously)
    assert batch.reqs == [], "precondition: filter_batch must have emptied the alias"
    assert mgr.spills == {}, "precondition: the slot must have been closed"

    assert streamer.calls, "the streamer was never called at all"
    streamed, _ = streamer.calls[-1]
    assert req in streamed, (
        "C28: the host-finished session was dropped from the list the streamer "
        "reads -- no terminal chunk is emitted and the HTTP caller blocks forever"
    )


def test_host_finish_reaches_the_streamer_alongside_a_surviving_sibling():
    """The same defect WITHOUT the empty-list special case. Two requests: one
    finishes on host, one keeps decoding. ``filter_batch`` rebinds ``reqs`` to
    the survivor alone, so the finished one is dropped from the emit even
    though the batch is not empty -- proving the bug is the rebind, not the
    emptiness. ``filter_batch`` also rebinds ``return_logprob`` from the
    SURVIVING requests, so a logprob request that finishes loses its flag."""
    done = _FakeReq("s45-done", rpi=5, finished=True, spilled=True, return_logprob=True)
    live = _FakeReq("s45-live", rpi=6, finished=False, spilled=False)
    batch = _RebindingBatch([done, live], return_logprob=True)
    mgr = _make_manager(batch)
    _install_slot(mgr, done, batch)
    streamer = _RecordingStreamer()
    proc = _make_processor(mgr, streamer)

    proc.process_batch_result_decode(batch, _make_result(2))

    assert batch.reqs == [live], "precondition: filter_batch kept only the survivor"

    assert streamer.calls, "the streamer was never called at all"
    streamed, return_logprob = streamer.calls[-1]
    assert done in streamed, (
        "C28: the host-finished session was dropped from a NON-EMPTY batch's "
        "emit -- the defect is filter_batch's rebind, not the empty case"
    )
    assert live in streamed, "the surviving request must still stream normally"
    assert return_logprob is True, (
        "C28 corollary: filter_batch rebinds return_logprob from the survivors, "
        "so a finished logprob request would be streamed without its logprobs"
    )


def test_plain_decode_path_streams_exactly_the_batch_unchanged():
    """Backward compatibility: with no spilled session in the batch nothing
    filters, and the streamer must receive precisely the batch's own list --
    the default path stays byte-identical. (The stock FINISHED-request path is
    left to the rest of the suite; driving it here would pull the whole
    release_kv_cache / radix machinery into a test about one list.)"""
    a = _FakeReq("s45-a", rpi=1, finished=False, spilled=False)
    b = _FakeReq("s45-b", rpi=2, finished=False, spilled=False)
    batch = _make_batch([a, b])
    mgr = _make_manager(batch)
    streamer = _RecordingStreamer()
    proc = _make_processor(mgr, streamer)

    proc.process_batch_result_decode(batch, _make_result(2))

    assert batch.reqs == [a, b], "nothing may filter on the default path"
    streamed, _ = streamer.calls[-1]
    assert streamed == [a, b]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
