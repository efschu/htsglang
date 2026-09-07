"""Weg-2 S3+S4: the refusals that guard CORRECTNESS, each with a can-it-fail
proof (speed mode: ONLY these, no battery).

W3  Weg2DrainWitnessDisagreement -- either direction is a STOP.
W16 Weg2DoublePrefillExceeded    -- one re-route, then refuse.
W17 '/health 200 is not a serving fact'.
W20 Weg2HostLedgerRefused        -- the launcher's ledger refusal.
W25 Weg2DormantRefused           -- the dormant admission seam (S1 killer K2).

Hermetic: CUDA_VISIBLE_DEVICES="" and no server.
"""

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger
from sglang.srt.weg2.front import (
    SpanLRU,
    double_prefill_verdict,
    fairness_reached,
    health_is_serving_fact,
    price_remainder,
    usage_of,
    witness_verdict,
)

GIB = 2**30


# ---------------------------------------------------------------- W3
def test_w3_fires_when_front_drained_but_rank_not_idle():
    assert witness_verdict(0, False) is not None


def test_w3_fires_in_the_reverse_direction():
    assert witness_verdict(3, True) is not None


def test_w3_silent_when_both_agree():
    assert witness_verdict(0, True) is None
    assert witness_verdict(2, False) is None


#: #1233 shipped X: one prefill chunk. Since #1234 slice A the bound is the
#: --tp-prefill-max-tokens FLAG, passed in explicitly, so these cases keep
#: asserting the same arithmetic against the same number.
ONE_CHUNK = 4096


# ---------------------------------------------------------------- W16
def test_w16_serve_within_one_chunk():
    assert double_prefill_verdict(10000, 10000 - ONE_CHUNK, 0, ONE_CHUNK) == "serve"


def test_w16_reroute_once_then_refuse():
    assert double_prefill_verdict(10000, 0, 0, ONE_CHUNK) == "reroute"
    assert double_prefill_verdict(10000, 0, 1, ONE_CHUNK) == "W16"


def test_w16_reads_cached_tokens_not_loaded():
    # the #1176 defect: a body with cached_tokens in prompt_tokens_details
    pt, ct, comp = usage_of({"usage": {"prompt_tokens": 9000, "completion_tokens": 5,
                                       "prompt_tokens_details": {"cached_tokens": 8000}}})
    assert (pt, ct, comp) == (9000, 8000, 5)
    assert double_prefill_verdict(pt, ct, 0, ONE_CHUNK) == "serve"


# ---------------------------------------------------------------- W17
def test_w17_http_200_alone_is_not_a_serving_fact():
    assert not health_is_serving_fact(True, False)
    assert health_is_serving_fact(True, True)
    assert not health_is_serving_fact(False, True)


# ---------------------------------------------------------------- W20
def test_w20_refuses_when_no_arm_funds_the_store():
    memtotal = 118 * GIB
    memavail = 60 * GIB  # a box with 60 GiB available cannot fund two backups
    with pytest.raises(host_ledger.Weg2HostLedgerRefused) as ei:
        host_ledger.choose(memtotal, memavail, store_min_gib=4.0)
    assert "W20 Weg2HostLedgerRefused" in str(ei.value)
    assert "ARM S=1 M=600" in str(ei.value)  # the whole ladder is printed


def test_w20_refuses_the_live_box_shape_with_the_1232_headroom_and_prints_every_term():
    """boot weg2ls1b2 (2026-09-07): with the #721 floor alone the ledger funded
    S=1/M=1200/store 5 GiB and the box OOM-killed at group D's first sleep.
    With the #1232 headroom charged the same box REFUSES by name."""
    memtotal = 118 * GIB
    memavail = 107 * GIB
    with pytest.raises(host_ledger.Weg2HostLedgerRefused) as ei:
        host_ledger.choose(memtotal, memavail, store_min_gib=4.0)
    text = str(ei.value)
    for term in ("heaps=", "backup_P=", "backup_D=", "load_transient=", "anchors@2400=", "rings=", "floor=", "host_headroom="):
        assert term in text


def test_w20_funds_a_box_with_enough_ram_and_prints_every_term():
    memtotal = 160 * GIB
    memavail = 150 * GIB
    arm, store, lines = host_ledger.choose(memtotal, memavail, store_min_gib=4.0)
    assert arm.s_gb == 1 and store >= 4
    assert arm.launch_leftover_gib >= 0 and arm.run_leftover_gib >= 0


def test_w20_both_moments_are_priced_differently():
    arm = host_ledger.price(118 * GIB, 107 * GIB, 1, 1200)
    # launch charges LOAD_TRANSIENT without D's backup; run charges both backups
    assert arm.launch_leftover_gib != arm.run_leftover_gib


# ---------------------------------------------------------------- pricing / fairness
def test_span_unknown_is_priced_at_full_prompt():
    spans = SpanLRU()
    remainder, est, known = price_remainder("x" * 3000, spans)
    assert not known and remainder == est


def test_span_known_reduces_the_remainder():
    spans = SpanLRU()
    prefix = "system:you are helpful\n" * 200
    spans.record(prefix, 1000)
    remainder, est, known = price_remainder(prefix + "user:hi\n", spans)
    assert known and remainder < est and remainder <= ONE_CHUNK


def test_fairness_bound_reaches_at_w():
    assert not fairness_reached(100.0, 140.0, 45.0)
    assert fairness_reached(100.0, 145.0, 45.0)
    assert not fairness_reached(None, 1e9, 45.0)


# ---------------------------------------------------------------- W25 dormant seam
def test_w25_dormant_refusal_names_the_seam_and_rid():
    from sglang.srt.managers.weg2_memory_saver import (
        DORMANT_REFUSAL_MARKER,
        Weg2DormantRefused,
        dormant_refusal_message,
    )

    msg = dormant_refusal_message(rid="abc", context="generate")
    assert DORMANT_REFUSAL_MARKER in msg and "abc" in msg and "generate" in msg
    assert issubclass(Weg2DormantRefused, RuntimeError)


def test_w25_seam_is_the_first_statement_of_both_admission_handlers():
    """Can-it-fail: the guard must precede any pool access.  Read the source
    of the two handlers and assert the dormant check is their first executable
    statement -- a guard moved below a Req() construction would still pass an
    import smoke and kill the group on metal."""
    import ast
    import inspect

    from sglang.srt.managers.scheduler import Scheduler

    for name in ("handle_generate_request", "handle_embedding_request"):
        src = inspect.getsource(getattr(Scheduler, name))
        tree = ast.parse("class _X:\n" + "\n".join("    " + l for l in src.splitlines()))
        fn = tree.body[0].body[0]
        first = fn.body[0]
        assert isinstance(first, ast.If), name
        assert "weg2_dormant" in ast.unparse(first.test), name
        assert any(isinstance(n, ast.Return) for n in first.body), name


def test_w25_flag_is_set_after_pause_and_cleared_after_resume():
    import inspect

    from sglang.srt.managers.scheduler_components import weight_updater as wu

    rel = inspect.getsource(wu.SchedulerWeightUpdaterManager.release_memory_occupation)
    res = inspect.getsource(wu.SchedulerWeightUpdaterManager.resume_memory_occupation)
    i_pause = rel.index("pause(GPU_MEMORY_TYPE_KV_CACHE)")
    i_set = rel.index("weg2_dormant = True")
    assert i_set > i_pause
    i_res = res.index("resume(GPU_MEMORY_TYPE_KV_CACHE)")
    i_clr = res.index("weg2_dormant = False")
    assert i_clr > i_res


# ---------------------------------------------------------------- boot weg2ls1b1 killer
def test_mamba_window_ratio_accepts_the_flag_parsers_list_shape():
    """weg2ls1b1 07:05:58Z: --rank-tp-ratio auto resolves to a LIST and the
    window ladder did str(list).split(',') -> int('[3725') ValueError, all
    three TP ranks dead before READY."""
    from sglang.srt.managers.cache_controller import parse_tp_ratio_vector

    assert parse_tp_ratio_vector([3725, 2264, 2259]) == [3725, 2264, 2259]
    assert parse_tp_ratio_vector("2,1,1") == [2, 1, 1]
    assert parse_tp_ratio_vector("auto") is None
    assert parse_tp_ratio_vector(None) is None
    assert parse_tp_ratio_vector([]) is None
