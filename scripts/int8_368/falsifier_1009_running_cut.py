from sglang.srt.server_args import ServerArgs
from sglang.srt.planner import pp_cut
from sglang.srt.planner.pp_cut_calibration import load_census_calibration, with_arena_split_state
sa = ServerArgs(model_path="/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed",
    trust_remote_code=True, tp_size=1, pp_size=3, device="cuda",
    rank_gpu_id=[0,1,2], rank_gpu_memory_mib=[31800,18800,19800],
    kv_cache_dtype="fp8_e4m3", context_length=262144, chunked_prefill_size=4096,
    page_size=1, pp_layer_ratio=[32,18,14], enable_phase_flip=True,
    phase_flip_tp_vector="32,16,16", uneven_token_vector="29,19,16",
    uneven_dcp=True, uneven_dcp_weighted=True, disable_overlap_schedule=True,
    enable_hierarchical_cache=True, hicache_ratio=1.5, barlink=True,
    barlink_transport="bar1", barlink_bar1_window_mib="24,PP_0=96,FLIP_TP_0=48,FLIP_DCP_0=32")
D="/spinning/evidence-665-f1/census-855-v2"
kinds=sa.declared_layer_kinds(); depth=sa.declared_num_hidden_layers()
families=tuple(pp_cut.LAYER_FAMILY_ATTENTION if k else pp_cut.LAYER_FAMILY_LINEAR for k in kinds)
pt=int(sa.max_total_tokens or 0); kvb=sa._pp_cut_kv_bytes_per_token_per_attn_layer(); sh=sa._pp_cut_token_shares()
cal=load_census_calibration(D)
cal=with_arena_split_state(cal,census_dir=D,kv_bytes_per_token_per_attn_layer=kvb,pool_tokens=pt or depth,tp_token_shares=sh)
rates=sa._pp_cut_card_rates(cal.gpu_names); budg=sa._pp_cut_budgets(cal.total_visible_mib)
tr=sa._pp_cut_transients(cal,D); sm=sa._pp_cut_seam_staging(tr,D)
ranks=tuple(pp_cut.RankResources(label=f"stage{i}-{rates[i][0]}",gemm_tflops=rates[i][1],attn_bw_gbs=rates[i][2],
    budget_mib=budg[i],fixed_overhead_mib=cal.residual_mib[i],transient_by_load_state=tr[i],seam_staging_mib=sm[i]) for i in range(3))
fl=sa._pp_cut_flops_per_token()
def mk(arena_pool):
    return pp_cut.PPCutInputs(layer_families=families,attn_layer_weight_bytes=cal.attn_layer_mib*pp_cut.MIB,
    linear_layer_weight_bytes=cal.linear_layer_mib*pp_cut.MIB,embedding_weight_bytes=cal.embedding_mib*pp_cut.MIB,
    lm_head_weight_bytes=cal.lm_head_mib*pp_cut.MIB,replicated_weight_bytes=cal.replicated_mib*pp_cut.MIB,
    state_bytes_per_linear_layer=cal.state_per_linear_mib*pp_cut.MIB,attn_layer_flops_per_token=fl["attn"],
    linear_layer_flops_per_token=fl["linear"],attn_core_flops_per_token_pair=fl["core"],
    kv_bytes_per_token_per_attn_layer=kvb,kv_depth_tokens=int(sa.chunked_prefill_size or 2048)*64,
    prefill_chunk_tokens=int(sa.chunked_prefill_size or 2048),ranks=ranks,kv_pool_tokens=arena_pool,tp_token_shares=sh)
inp=mk(pt)
print(f"RESULT arena_tokens={inp.arena_tokens} (kv_pool_tokens={inp.kv_pool_tokens}, kv_depth_tokens={inp.kv_depth_tokens})")
print(f"RESULT budget_mib  ={[round(r.budget_mib,1) for r in inp.ranks]}")
print(f"RESULT residual_mib={[round(r.fixed_overhead_mib,1) for r in inp.ranks]}")
print(f"RESULT seam_mib    ={[round(r.seam_staging_mib,1) for r in inp.ranks]}")
print(f"RESULT worst_trans ={[round(r.worst_transient_mib,1) for r in inp.ranks]}")
print("RESULT === RUNNING CUT [32,18,14] under the solver's own model:")
for c in pp_cut.stage_costs((32,18,14), inp):
    print(f"RESULT   {c.rank}: resident={c.resident_mib:9.1f} headroom={c.headroom_mib:9.1f} feasible={c.feasible}")
