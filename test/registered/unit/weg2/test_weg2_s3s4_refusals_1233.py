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
    CHUNK_TOKENS,
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


# ---------------------------------------------------------------- W16
def test_w16_serve_within_one_chunk():
    assert double_prefill_verdict(10000, 10000 - CHUNK_TOKENS, 0) == "serve"


def test_w16_reroute_once_then_refuse():
    assert double_prefill_verdict(10000, 0, 0) == "reroute"
    assert double_prefill_verdict(10000, 0, 1) == "W16"


def test_w16_reads_cached_tokens_not_loaded():
    # the #1176 defect: a body with cached_tokens in prompt_tokens_details
    # usage_of grew a fourth term (`priced`, #1233 zero-remainder 1j finding 3)
    # and this pin was not carried with it -- it has been failing to unpack
    # since, i.e. red at bc31554f90 and at 53bd804e2e, before fix 4 touched
    # anything.
    pt, ct, comp, priced = usage_of({"usage": {"prompt_tokens": 9000, "completion_tokens": 5,
                                               "prompt_tokens_details": {"cached_tokens": 8000}}})
    assert (pt, ct, comp, priced) == (9000, 8000, 5, True)
    assert double_prefill_verdict(pt, ct, 0) == "serve"


# ---------------------------------------------------------------- W17
def test_w17_http_200_alone_is_not_a_serving_fact():
    assert not health_is_serving_fact(True, False)
    assert health_is_serving_fact(True, True)
    assert not health_is_serving_fact(False, True)


# ---------------------------------------------------------------- W20
#: C19 (2026-09-07): the ledger's host weights term is no longer a constant in
#: the module -- it is the previous boot's own per-card table, so every caller
#: passes it in.  These are boot weg2zr2's numbers (WEG2_FLIPCOST_SPEC_0907
#: section 2, reproduced from that boot's lines by
#: test_weg2_ring_ledger_1235.RealBootProvenanceTest): Sigma H 32 964 MiB,
#: Sigma image_P 29 912 MiB.  This file tests the LEDGER; the provenance claim
#: itself is that other file's job.
RING_BYTES = 32964 * 1024 * 1024
RING_SPAN1_BYTES = 29912 * 1024 * 1024
RING = dict(ring_bytes=RING_BYTES, ring_span1_bytes=RING_SPAN1_BYTES,
            ring_provenance="boot weg2zr2 (spec section 2 table)")


def test_w20_refuses_when_no_arm_funds_the_store():
    memtotal = 118 * GIB
    memavail = 60 * GIB  # a box with 60 GiB available cannot fund the ring
    with pytest.raises(host_ledger.Weg2HostLedgerRefused) as ei:
        host_ledger.choose(memtotal, memavail, store_min_gib=4.0, **RING)
    assert "W20 Weg2HostLedgerRefused" in str(ei.value)
    assert "ARM S=1 M=600" in str(ei.value)  # the whole ladder is printed


def test_w20_refuses_a_boot_with_no_measured_ring_table_rather_than_guessing():
    """C19/R22: with BACKUP_P_BYTES and BACKUP_D_BYTES deleted there is no
    constant left to price the flip with, and the planner does not invent one."""
    with pytest.raises(host_ledger.Weg2HostLedgerRefused) as ei:
        host_ledger.choose(160 * GIB, 150 * GIB, store_min_gib=4.0)
    assert "no measured source" in str(ei.value)


#: The DR-1 shape this test was written against, now priced from the SAME
#: measured table instead of from the deleted constants: BOTH full images
#: host-resident at one instant, Sigma image_P + Sigma image_D = 29 912 +
#: 32 964 MiB = 61.4 GiB (spec R10, "the DR-1 shape refused in record 1h").
#: The old ledger expressed it as ``weight_chunks=0`` -> backup_P + backup_D.
DR1_BYTES = (29912 + 32964) * 1024 * 1024


def test_w20_still_refuses_the_live_box_shape_under_the_DR1_two_image_shape():
    """boot weg2ls1b2 (2026-09-07): with the #721 floor alone the ledger funded
    S=1/M=1200/store 5 GiB and the box OOM-killed at group D's first sleep.
    With the #1232 headroom charged the same box REFUSES by name.

    C19 keeps this regression intact by pricing the same shape from the
    measured table rather than from the deleted constants: two resident images
    = 61.4 GiB, which this box does not fund at any arm."""
    memtotal = 118 * GIB
    memavail = 107 * GIB
    with pytest.raises(host_ledger.Weg2HostLedgerRefused) as ei:
        host_ledger.choose(memtotal, memavail, store_min_gib=4.0,
                           ring_bytes=DR1_BYTES,
                           ring_span1_bytes=RING_SPAN1_BYTES,
                           ring_provenance="boot weg2zr2, DR-1 two-image shape")
    text = str(ei.value)
    # fix 8: the MEASURED dormant image (`image_P=`/`image_D=`) prints beside the
    # #809 census sums (`weight_tags_P=`) it used to be confused with -- and on
    # the ring both are PROVENANCE, not charges: the charge is Sigma H.
    for term in ("heaps=", "RUN MOMENT = the host weights term", "LAUNCH MOMENT = ring span 1",
                 "image_P=", "image_D=", "weight_tags_P=", "run_origin=",
                 "load_transient=", "anchors@2400=", "rings=", "floor=",
                 "memory.current=", "base_cgroup="):
        assert term in text
    # And the deleted constants may not come back through the printed line.
    assert "backup_P=" not in text and "backup_D=" not in text
    # fix 5's deletions are deletions on this branch too: the #1232 headroom is
    # not a charged term (it survives only as the prose naming its own removal),
    # and the flip transient is not a term beside Sigma H.
    assert "host_headroom=" not in text
    assert "flip_transient=" not in text


def test_the_cgroup_denominator_binds_and_names_itself(capsys):
    """fix 5 (boot weg2dk5), carried onto the ring: the reaper watches
    memory.current, so a cgroup sample tighter than meminfo must BIND and the
    printed line must say which reading bound it."""
    arm, _store, lines = host_ledger.choose(
        118 * GIB, 107 * GIB, store_min_gib=4.0,
        cg_current_bytes=int(60 * GIB), cg_ceiling_bytes=int(118 * GIB),
        cg_ceiling_source="memory.max", cg_oom_kill=18, **RING)
    text = "\n".join(lines)
    assert "memory.current=" in text and "base_cgroup=" in text
    assert "oom_kill_baseline=18" in text
    assert arm.terms["base_source"].startswith("cgroup")


def test_the_ring_is_what_makes_that_same_box_fundable():
    """The whole point of C1-C8, priced: the shared region charges Sigma H once
    (32.19 GiB) where DR-1 charged both images (61.4)."""
    arm, store, lines = host_ledger.choose(118 * GIB, 107 * GIB, store_min_gib=4.0, **RING)
    assert arm.launch_leftover_gib >= 0 and arm.run_leftover_gib >= 0 and store >= 4
    assert round(arm.terms["host_ring_gib"], 2) == 32.19


def test_w20_funds_a_box_with_enough_ram_and_prints_every_term():
    memtotal = 160 * GIB
    memavail = 150 * GIB
    arm, store, lines = host_ledger.choose(memtotal, memavail, store_min_gib=4.0, **RING)
    assert arm.s_gb == 1 and store >= 4
    assert arm.launch_leftover_gib >= 0 and arm.run_leftover_gib >= 0


def test_w20_both_moments_are_priced_differently():
    arm = host_ledger.price(118 * GIB, 107 * GIB, 1, 1200,
                            ring_bytes=RING_BYTES, ring_span1_bytes=RING_SPAN1_BYTES)
    # launch charges span 1 plus LOAD_TRANSIENT (R7); run charges the whole ring
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
    assert known and remainder < est and remainder <= CHUNK_TOKENS


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
