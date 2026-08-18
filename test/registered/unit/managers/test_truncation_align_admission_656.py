"""C30: a chunk budget below the prefill truncation alignment admits nothing.

CPU-only; no server, no GPU. Run:
  python -m pytest test/registered/unit/managers/test_truncation_align_admission_656.py -q

THE DEFECT. ``PrefillAdder.add_one_req``'s chunked branch aligns the chunk it
is about to take and refuses outright when the whole chunk budget is smaller
than one alignment unit. ``rem_chunk_tokens`` is bounded above by
``--chunked-prefill-size``, so if that flag is below the alignment size the
branch returns ``OTHER`` for every request longer than the budget, forever.
The scheduler's admission loop ``break``s on any non-CONTINUE verdict, so one
such request at the head of the FCFS queue blocks everything behind it.

The instance therefore BOOTS, reports ready, serves its warmup prefills (short
enough to take the non-chunked branch) and then admits NOTHING. Measured:
zero ``Decode batch`` lines across an entire boot, an 8-token ``/generate``
hung 55 s, ``/health`` timing out while ``/get_model_info`` answered
instantly, and the collective census frozen at an IDENTICAL count on both
ranks -- no crash, no collective hang, no rank divergence, no log line.

It was first seen under ``--enable-deterministic-inference`` +
``--enable-kv-session-offload`` and booked as an exclusion between those two
flags. It is neither. kv-session-offload's only role is that it forces the
flashinfer backend; the refusing predicate is a third variable,
``--chunked-prefill-size``, and the same two flags with a large enough chunk
budget serve normally. The alignment also has a SECOND source that needs no
deterministic inference at all -- ``--mamba-checkpoint-interval`` -- so a
refusal worded as "these two flags are incompatible" would both reject
working configurations and leave the real trap armed elsewhere.
"""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

# `truncation_align_admission_error` is imported INSIDE the two guard tests,
# not here: on a tree without it a module-level import would fail collection
# and take the mechanism test down with it. The mechanism test must be able to
# RUN on the unfixed tree -- that it passes there is the evidence that the
# trap is real and that nothing was refusing it.
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefResult,
    IncLockRefResult,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def _adder(rem_chunk_tokens, page_size=1):
    """A PrefillAdder with a real budget and a chunk budget of the given size."""
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    tree_cache = MagicMock()
    tree_cache.full_evictable_size.return_value = 0
    tree_cache.swa_evictable_size.return_value = 0
    tree_cache.evictable_size.return_value = 0
    tree_cache.disable = False
    tree_cache.inc_lock_ref.return_value = IncLockRefResult()
    tree_cache.dec_lock_ref.return_value = DecLockRefResult()

    allocator = MagicMock()
    allocator.full_available_size.return_value = 1_000_000
    allocator.swa_available_size.return_value = 1_000_000
    allocator.available_size.return_value = 1_000_000

    running_batch = MagicMock()
    running_batch.reqs = []

    return PrefillAdder(
        page_size=page_size,
        tree_cache=tree_cache,
        token_to_kv_pool_allocator=allocator,
        running_batch=running_batch,
        new_token_ratio=1.0,
        rem_input_tokens=1_000_000,
        rem_chunk_tokens=rem_chunk_tokens,
        num_mixed_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
    )


def _req(n_tokens):
    """A fresh request of `n_tokens` prompt tokens with no cached prefix."""
    from sglang.srt.managers.schedule_batch import Req

    req = MagicMock(spec=Req)
    req.rid = "c30"
    req.priority = 0
    req.prefix_indices = []
    req.full_untruncated_fill_ids = list(range(n_tokens))
    req.fill_ids = list(range(n_tokens))
    req.origin_input_ids = list(range(n_tokens))
    req.output_ids = []
    req.host_hit_length = 0
    req.swa_host_hit_length = 0
    req.sampling_params = SimpleNamespace(max_new_tokens=8, ignore_eos=False)
    req.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    req.retracted_stain = False
    req.born_spilled = False
    req.born_spilled_deep = False
    req.last_node = None
    req.finished.return_value = False
    req.needs_host_load_back.return_value = False
    return req


def test_a_chunk_budget_below_the_alignment_refuses_every_long_request():
    """THE MECHANISM, driven through the real add_one_req.

    True on both trees -- this is upstream behaviour, and it stays here as the
    regression guard that proves the boot-time refusal added for C30 is
    guarding something real rather than an imagined failure. A gate whose
    failure mode was never executed is a gate nobody can trust.
    """

    def _admitted(align):
        """Did the request make it onto can_run_list?

        The RETURN value cannot answer this: a chunked admission consumes the
        whole chunk budget, so `budget_state()` reports OTHER on the way out
        of a SUCCESSFUL admission just as the alignment refusal does. What the
        scheduler actually acts on is the list -- `can_run_list` empty is
        literally the wedge condition (`if len(can_run_list) == 0: return
        None`), so that is what this measures.
        """
        adder = _adder(rem_chunk_tokens=256)
        adder.add_one_req(
            _req(2000), has_chunked_req=False, truncation_align_size=align
        )
        return len(adder.can_run_list) > 0

    # 256-token chunk budget against a 4096-token alignment: the exact shape
    # of the wedged boot. Nothing reaches can_run_list, so the scheduler
    # builds no batch -- forever, since neither the budget nor the alignment
    # ever changes.
    assert not _admitted(4096), (
        "the request was admitted at a 4096-token alignment on a 256-token "
        "chunk budget; this test no longer pins the wedge it was written for"
    )

    # THE CONTROL, one variable moved: the identical request and the identical
    # budget, with an alignment the budget can satisfy, IS admitted. So the
    # refusal above belongs to the alignment and not to the fixture.
    assert _admitted(256), (
        "the control was refused too -- the fixture, not the alignment, is "
        "deciding these verdicts"
    )

    # ...and with no alignment at all, as on every non-deterministic boot.
    assert _admitted(None)


def test_truncation_align_admission_error_names_the_numbers():
    """The guard itself: it must fire on the wedging config and stay silent on
    every configuration that can actually admit."""
    from sglang.srt.managers.schedule_policy import truncation_align_admission_error

    def _err(*a, **k):
        return truncation_align_admission_error(*a, **k)[0]

    def _warn(*a, **k):
        return truncation_align_admission_error(*a, **k)[1]

    # the configuration that wedged the instance
    err = _err(256, 1, 4096)
    assert err is not None, (
        "a 256-token chunk budget against a 4096-token alignment was accepted; "
        "that configuration boots, reports ready and then admits nothing"
    )
    for needle in ("256", "4096", "chunked-prefill-size", "admit nothing"):
        assert needle in err, f"error does not name {needle!r}: {err}"

    # the same alignment with a budget that satisfies it
    assert _err(4096, 1, 4096) is None
    assert _err(8192, 1, 4096) is None

    # page alignment is part of the budget: 4100 aligns DOWN to 4096 at
    # page_size 8 -> 4096, still fine; but 4095 does not reach 4096
    assert _err(4100, 8, 4096) is None
    assert _err(4095, 1, 4096) is not None

    # inert wherever the trap is unreachable: no alignment, or chunked
    # prefill switched off (rem_chunk_tokens is None -> the branch never runs)
    assert _err(256, 1, None) is None
    assert _err(256, 1, 0) is None
    assert _err(None, 1, 4096) is None
    assert _err(-1, 1, 4096) is None
    assert _err(0, 1, 4096) is None

    # the sources are quoted back so the operator knows which flag set the
    # alignment -- there are two and they are not interchangeable
    err = _err(256, 1, 4096, ("--mamba-checkpoint-interval=4096",))
    assert "--mamba-checkpoint-interval=4096" in err

    # DYNAMIC CHUNKING: the predictor's floor is base//4, so a static
    # budget that satisfies the alignment can still dip below it at
    # runtime. Warned, not refused -- it is conditional on runtime
    # behaviour and refusing would reject configs that mostly work.
    assert _err(8192, 1, 4096, dynamic_chunking=True) is None
    w = _warn(8192, 1, 4096, dynamic_chunking=True)
    assert w is not None and "2048" in w and "16384" in w
    # a budget of 4x the alignment cannot dip below it -> silent
    assert _warn(16384, 1, 4096, dynamic_chunking=True) is None
    # and with dynamic chunking OFF the same budget says nothing
    assert _warn(8192, 1, 4096) is None


def test_the_scheduler_refuses_at_boot_for_both_alignment_sources():
    """The wiring, driven through the REAL init_deterministic_inference_config.

    Both contributors are covered, because either alone arms the trap:
    deterministic inference on flashinfer/triton, and the mamba checkpoint
    grid. The check sits after the lcm, which is the only point where the
    final alignment is known -- server_args cannot see it without restating
    the lcm.
    """
    from sglang.srt.managers.scheduler import Scheduler

    def _run(**kw):
        sa = SimpleNamespace(
            enable_deterministic_inference=False,
            attention_backend="flashinfer",
            mamba_checkpoint_interval=None,
            chunked_prefill_size=256,
            page_size=1,
        )
        for k, v in kw.items():
            setattr(sa, k, v)
        s = object.__new__(Scheduler)
        s.server_args = sa
        Scheduler.init_deterministic_inference_config(s)
        return s.truncation_align_size

    # SOURCE 1: deterministic inference on flashinfer, chunk budget 256.
    # This is the boot that wedged.
    try:
        _run(enable_deterministic_inference=True)
        raise AssertionError(
            "the scheduler accepted --enable-deterministic-inference on "
            "flashinfer with --chunked-prefill-size 256: that instance boots, "
            "reports ready and then admits nothing at all"
        )
    except ValueError as e:
        assert "4096" in str(e) and "256" in str(e)
        assert "deterministic" in str(e)

    # ...and the SAME two flags with a chunk budget that satisfies the
    # alignment must still boot. The refusal is about the budget, not about
    # deterministic inference, and wording it as a flag-pair exclusion would
    # reject this working configuration.
    assert _run(enable_deterministic_inference=True, chunked_prefill_size=4096) == 4096

    # SOURCE 2, REWRITTEN BY #750: the mamba checkpoint grid ALONE can no
    # longer arm the wedge at all. A divisible interval above the chunk
    # budget (4096 = 16 x 256) is a sparse grid that is NOT folded into the
    # alignment -- every 16th full chunk end anchors, the ends between are
    # not cached, and the truncation alignment stays untouched, so C30 has
    # nothing to refuse. (An interval AT or BELOW the budget folds, but its
    # fold equals the interval and therefore always fits the budget; a
    # non-divisible sparse interval is refused earlier, at server_args
    # validation, for collapsing the anchor cadence.) The trap now needs a
    # SECOND alignment source -- which is exactly the lcm case below.
    assert _run(mamba_checkpoint_interval=4096) is None

    # BOTH sources: the alignment is their lcm, and the guard sees the lcm
    # rather than either input. 4096 and 768 -> 12288, which a 4096-token
    # chunk budget does NOT satisfy even though it satisfies each alone.
    assert math.lcm(4096, 768) == 12288
    try:
        _run(
            enable_deterministic_inference=True,
            mamba_checkpoint_interval=768,
            chunked_prefill_size=4096,
        )
        raise AssertionError(
            "the lcm of the two alignment sources was not checked: each input "
            "fits the 4096-token budget on its own, their lcm 12288 does not"
        )
    except ValueError as e:
        assert "12288" in str(e)

    # the ordinary boot: no alignment from either source, nothing refused
    assert _run() is None
    assert _run(chunked_prefill_size=-1) is None
    # deterministic on a backend with no alignment requirement
    assert _run(enable_deterministic_inference=True, attention_backend="fa3") is None
