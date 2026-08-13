#!/usr/bin/env bash
# #631 phase-flip test family -- THE canonical list (operator request
# 2026-08-08: two independent under-collections happened from ad-hoc
# globs; the family spans three directories, one glob does NOT cover it).
# CPU-only desk run: correct PYTHONPATH (worktree, not the serving tree)
# and no GPU (CUDA_VISIBLE_DEVICES=99).
set -euo pipefail

WT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/spinning/htsglang-gpu/.venv/bin/python}"

FAMILY=(
    test/registered/scheduler/test_phase_flip_plan.py
    test/registered/scheduler/test_phase_flip_runtime.py
    test/registered/scheduler/test_phase_flip_boot.py
    test/registered/scheduler/test_phase_flip_protocol.py
    test/registered/scheduler/test_phase_flip_round_cadence_631.py
    test/registered/scheduler/test_phase_flip_resident_carry.py
    # Fourth under-collection, caught by MERGE-R7 §7 and fixed here. The
    # kv-universe branch (#656) added six test files and did not extend this
    # list, so the family total sat at 1116 looking green while the seam
    # sizer that REPLACED the 620000-token quarantine constant -- the whole
    # point of that branch -- was collected by nothing. MERGE-R7 ran them as
    # a private arm; a private arm is not a canonical list, and the next
    # shift would not have known to run it.
    test/registered/scheduler/test_phase_flip_seam_reserve.py
    test/registered/scheduler/test_phase_policy_arm_outcome.py
    test/registered/scheduler/test_seam_fingerprint_and_margin_656.py
    test/registered/scheduler/test_gdn_flip_plan.py
    test/registered/scheduler/test_gdn_flip_mover.py
    test/registered/scheduler/test_weights_arena.py
    test/registered/scheduler/test_kv_reshard.py
    test/registered/scheduler/test_step6_harness.py
    test/registered/unit/managers/test_regime_act.py
    test/registered/unit/managers/test_phase_policy.py
    test/registered/unit/managers/test_phase_flip_counters.py
    test/registered/unit/managers/test_pp_chain_receiver.py
    test/registered/unit/managers/test_pp_proxy_stamp_631.py
    test/registered/unit/managers/test_pp_flip_slot_hold_631.py
    test/registered/unit/managers/test_pp_slot_last_batch_631.py
    test/registered/unit/managers/test_phase_flip_draft_bootstrap_631.py
    test/registered/unit/managers/test_phase_flip_spec_seam_631.py
    test/registered/unit/managers/test_phase_purity_631.py
    test/registered/unit/managers/test_phase_flip_prefill_hold_631.py
    test/registered/unit/managers/test_phase_flip_output_trace_631.py
    test/registered/unit/managers/test_phase_flip_decode_relay_631.py
    test/registered/unit/managers/test_spec_verify_width_631.py
    test/registered/unit/managers/test_kv_arena_reclaim_631.py
    test/registered/unit/managers/test_phase_flip_seam_census_631.py
    test/registered/unit/managers/test_truncation_align_admission_656.py
    test/registered/unit/managers/test_phase_flip_spill_depth_631.py
    test/registered/unit/managers/test_phase_flip_draft_carrier_631.py
    test/registered/unit/managers/test_corridor_guard_631.py
    test/registered/unit/managers/test_corridor_law_preempt_656.py
    test/registered/unit/managers/test_kv_backing_relief_631.py
    test/registered/unit/managers/test_kv_backing_collective_631.py
    test/registered/unit/managers/test_kv_admission_floor_631.py
    test/registered/unit/managers/test_corridor_even_fill_631.py
    test/registered/unit/managers/test_corridor_admission_631.py
    test/registered/unit/managers/test_kvso_flip_contract_631.py
    test/registered/unit/managers/test_dynamic_chunk_engagement_631.py
    test/registered/unit/managers/test_arena_high_water_631.py
    test/registered/unit/managers/test_phase_flip_corridor_gate_631.py
    test/registered/unit/managers/test_seam_entry_margin_631.py
    test/registered/unit/managers/test_seam_abandon_cap_631.py
    test/registered/unit/managers/test_phase_flip_arena_tail_631.py
    test/registered/unit/managers/test_kv_arena_handle_retention_631.py
    test/registered/unit/managers/test_spec_counter_wire_631.py
    test/registered/unit/managers/test_corridor_steering_657.py
    # FIFTH under-collection, avoided rather than caught: the #656
    # remediation shift added these two and extended the list in the same
    # commit. The first is register C22's root cause (the flip's wire frame
    # is never agreed) and carries the red arm that reproduces the metal
    # signature on a CPU desk; the second pins the corridor law and the
    # gate's arming floor as ONE declared pair, and stops a private 1024
    # literal creeping back into an instrument that audits the gate.
    test/registered/scheduler/test_flip_frame_agreement_656.py
    test/registered/unit/managers/test_corridor_threshold_pair_656.py
    # Third under-collection, caught 2026-08-10: six tests were added to
    # this file and the family total did not move, because the list is
    # explicit and nobody extended it. The file guards spec item 8's flag
    # and the eager-runner regression that took all three ranks down, so
    # it belongs in the sweep, not only in a hand-run.
    test/registered/unit/managers/test_draft_cuda_graph_removal.py
    test/registered/unit/managers/test_spec_mamba_commit_width_631.py
    test/registered/unit/server_args/test_phase_flip_args.py
    test/registered/unit/distributed/test_phase_flip_groups.py
    test/registered/unit/distributed/test_census_wire_domain_631.py
    test/registered/unit/layers/test_causal_conv1d_bounds_631.py
    test/registered/unit/managers/test_phase_flip_staging_reserve_631.py
    test/registered/unit/managers/test_phase_flip_live_slots_no_pool_idx_631.py
    test/registered/unit/mem_cache/test_gdn_cap_flip_two_stacks_631.py
    # Added by successor 25. The streaming file was NEVER collected by this
    # runner despite holding the strongest peak-residency pins in the tree
    # (TestMoverLiveSetIsBounded, TestStagingFormulaMatchesReality) -- the
    # suite reported green while those assertions never executed. That is
    # the third under-collection this header warns about, so it is fixed
    # here rather than noted.
    test/registered/unit/managers/test_phase_flip_mover_streaming_631.py
    test/registered/unit/managers/test_kv_arena_span_ops_631.py
    test/registered/unit/mem_cache/test_kv_pool_row_span_631.py
    # Added by successor 45 with the C28 fix (#659). Red-first: it fails on
    # the pre-fix tree with the host-finished session missing from the emit.
    test/registered/unit/managers/test_host_finish_stream_659.py
    # The other half of the MERGE-R7 §7 under-collection: three #656 files
    # under test/srt/. They are reached by the same test/conftest.py as
    # everything above, so the runner's cd does not change how they collect.
    #
    # test_phase_flip_serving_proof_gate.py needs a VISIBLE DEVICE and carries
    # its own module-level skipif for that reason (measured: 7/7 with a
    # device, 4 failed / 3 passed without). It is listed here so the family
    # names it, and skips itself under this runner's CUDA_VISIBLE_DEVICES=99
    # rather than reporting red -- the runner's CPU-only contract is that a
    # green run means something, and a permanently-red entry destroys that.
    # Running it FOR REAL requires a device arm inside a GPU window.
    test/srt/test_phase_flip_serving_proof_gate.py
    test/srt/test_rope_lazy_cache.py
    test/srt/test_yarn_rope_cache_growth.py
)

cd "$WT/test/registered/scheduler"
PYTHONPATH="$WT/python" CUDA_VISIBLE_DEVICES=99 \
    "$PY" -m pytest "${FAMILY[@]/#/$WT/}" "$@"
