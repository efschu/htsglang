"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

from flliper.srt.pdflip import DEFAULT_D_BS
from flliper.srt.pdflip import DEFAULT_P_BS
from flliper.srt.pdflip import topology as _topo
from flliper.srt.pdflip import vote_fixed_live as _vfl
from flliper.srt.pdflip.launcher_d_leaf import D_BARLINK_BAR1_WINDOW_MIB
from flliper.srt.pdflip.launcher_d_leaf import VISION_OFF
from flliper.srt.pdflip.launcher_d_leaf import _adopt_load_format_flag
from flliper.srt.pdflip.launcher_d_leaf import admin_key_flag
from flliper.srt.pdflip.launcher_d_mid import CONTEXT_LENGTH_TOKENS
from flliper.srt.pdflip.launcher_d_p_r2_L0 import P_CHUNKED_PREFILL_TOKENS
from flliper.srt.pdflip.launcher_d_p_r2_L0 import P_PREFILL_TRANSIENT_SUPPORT
from flliper.srt.pdflip.launcher_d_p_r2_L0 import p_activation_record
from flliper.srt.pdflip.launcher_d_p_r2_L1 import BARLINK_BAR1_CAP_CYCLES
from flliper.srt.pdflip.launcher_d_p_r2_L1 import COLLECTIVE_CENSUS_INTERVAL
from flliper.srt.pdflip.launcher_d_p_r2_L1 import RANDOM_SEED
from flliper.srt.pdflip.launcher_d_p_r2_L1 import common_flags
from flliper.srt.pdflip.launcher_d_p_r2_L1 import d_window_pool_fields
from flliper.srt.pdflip.launcher_d_p_r2_L2 import argv_p
from flliper.srt.pdflip.launcher_d_p_r2_L2 import d_extend_vote_fixed_live_form
from flliper.srt.pdflip.launcher_d_p_r2_L2 import d_replayssm_spec_flags
from flliper.srt.pdflip.launcher_d_p_r2_L2 import p_prefill_transient_vector_mib
from flliper.srt.pdflip.launcher_d_p_r2_L2 import spec_plan_fields
from flliper.srt.pdflip.launcher_p_l3_leaf import P_PREFILL_TRANSIENT_CALIBRATION_MODELS
from flliper.srt.pdflip.launcher_p_l3_leaf import _CALIBRATION_FOOTPRINT_ALIAS
from flliper.srt.pdflip.launcher_p_l3_leaf import _group_fact_value
from flliper.srt.pdflip.launcher_p_l3_leaf import _pconst
from flliper.srt.pdflip.launcher_p_mid import KV_CACHE_DTYPE
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import spec_flags
from flliper.srt.pdflip.launcher_refuse_proc_leaf import PROFILE_QWEN27B
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _profile_early_read
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _published_profile
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _refuse_if_extra_drops_transient_lmo
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _refuse_if_extra_raises_budget
from flliper.srt.pdflip.launcher_types import Card
from flliper.srt.pdflip.launcher_xchg_vision_leaf import vote_fixed_live_root

PORT_D = 30032
#: #1236 store on disk: the store parameter of ``argv_p``/``argv_d``/``common_flags``
#: is the RENDERED extra-config JSON (a str), so the form-key sentinel is an
#: impossible JSON string, never a float -- boot weg2sb5h died at
#: ring_table.p_form_key on a float that reached ``_flag_pairs`` (stale call
#: sites after the signature change).  The flag is in FORM_KEY_EXCLUDED_FLAGS,
#: so the value never enters the key; it only has to be a str.
RING_FORM_SENTINEL_STORE_CFG = '{"form-key-sentinel":"not-a-store"}'
#: The `prefill activation reserve` post, MiB per rank. 1.000 GiB on every
#: rank of both boots. It had NO FIELD in the pool model before #1286.
P_PREFILL_ACTIVATION_RESERVE_MIB = 1024.0
#: The runtime's residual post for exactly this transient
#: (``model_runner_kv_cache_mixin.PREFILL_TRANSIENT_ENV``): a comma vector,
#: MiB per rank, that the sizer subtracts from ``rest`` before the KV pool.
#: Named here rather than imported so the launcher stays torch-free; the
#: unit test pins the two names to each other.
P_PREFILL_TRANSIENT_ENV = "FLLIPER_KV_BUDGET_PREFILL_TRANSIENT_MIB"
#: PDFLIP-EXTEND-CAP: allocated transient per D extend row under the P0 torch cache cap
D_EXTEND_CAP_RATE_RECORD = "D_EXTEND_CAP_PER_ROW_MIB"

def p_prefill_transient_for(boot_form) -> Tuple[bool, str]:
    """``(publish?, provenance)`` of the #114 transient for this boot's form.

    No form (a desk caller of ``build_env``) publishes exactly as before.
    """
    if boot_form is None:
        return True, "no PDFLIP-FORM (desk caller): published as before"
    _alias = _CALIBRATION_FOOTPRINT_ALIAS.get(boot_form.model)
    if boot_form.model in P_PREFILL_TRANSIENT_CALIBRATION_MODELS or (
            _alias and _alias[0] in P_PREFILL_TRANSIENT_CALIBRATION_MODELS):
        return True, (
            f"published: chunk {P_CHUNKED_PREFILL_TOKENS} -> "
            f"{p_prefill_transient_vector_mib(P_CHUNKED_PREFILL_TOKENS)} MiB per P stage, "
            f"MEASURED on {_alias[0] if _alias and boot_form.model not in P_PREFILL_TRANSIENT_CALIBRATION_MODELS else boot_form.model} (support points "
            f"{list(P_PREFILL_TRANSIENT_SUPPORT.chunks)}, linear between, H41)"
            + (f"; H87: this boot's checkpoint {boot_form.model} has the same memory "
               f"footprint ({_alias[1]})"
               if _alias and boot_form.model not in P_PREFILL_TRANSIENT_CALIBRATION_MODELS
               else "")
        )
    return False, (
        f"NOT published: the support points "
        f"{list(P_PREFILL_TRANSIENT_SUPPORT.chunks)} were measured on "
        f"{', '.join(P_PREFILL_TRANSIENT_CALIBRATION_MODELS)}; this boot runs "
        f"{boot_form.model or '?'}, on which no prefill transient was measured. "
        f"Its P ranks size the pool without that post, as every boot of this "
        f"model before #114 did (27B line xsn411: PP0 277606 tokens; xsn417 with "
        f"the foreign slopes: 246155 < 262144). An operator-exported "
        f"{P_PREFILL_TRANSIENT_ENV} still wins."
    )

def p_prefill_activation_reserve_mib(pinned: Optional[float], chunk_tokens: int) -> float:
    """The pool model's `prefill activation reserve` post for ``chunk_tokens``.

    ``pinned`` (``--pp-cut-activation-reserve-mib``) wins when the operator set
    it. Otherwise the LARGER of the reference boots' 1024 MiB and the measured
    transient of the binding stage (max over stages): a chunk at or below the
    reference width prices byte-identically to #1286, a wider chunk prices what
    ``[vram-peak]`` measured. The pool model carries ONE post for every rank,
    and the max over stages is the safe way to be wrong -- an over-charge
    under-prices a pool, an under-charge OOMs a forward (fnFL2x121).

    H41: BELOW the smallest support point (512) the transient is at most the
    measured 133 MiB (it grows with the chunk), so the 1024 MiB reference post
    dominates without extrapolating; ABOVE the largest the chunk is REFUSED
    (W131) -- the width x121/x122 died of was exactly an unmeasured one.
    """
    if pinned is not None:
        return float(pinned)
    if int(chunk_tokens) < P_PREFILL_TRANSIENT_SUPPORT.chunks[0]:
        return float(P_PREFILL_ACTIVATION_RESERVE_MIB)
    return max(
        float(P_PREFILL_ACTIVATION_RESERVE_MIB),
        max(p_prefill_transient_vector_mib(chunk_tokens)),
    )

def p_prefill_activation_reserve_by_stage_mib(
    pinned: Optional[float], chunk_tokens: int, profile: Optional[str] = None
) -> Tuple[float, ...]:
    """#242: the pool model's activation post PER STAGE -- ``()`` (the scalar
    of :func:`p_prefill_activation_reserve_mib` on every stage, 27B
    byte-identical) unless the profile carries ``P_ACTIVATION_MIB``. Then each
    stage books its own measured transient, at least the reference 1024 MiB,
    exactly what the runtime books per rank from ``P_PREFILL_TRANSIENT_ENV``
    (the PP-POOL-JOIN of the two is then a join of equal posts). A pinned
    ``--pp-cut-activation-reserve-mib`` pins every stage."""
    if pinned is not None or p_activation_record(profile)[0] is None:
        return ()
    if int(chunk_tokens) < P_PREFILL_TRANSIENT_SUPPORT.chunks[0]:
        return ()
    return tuple(
        max(float(P_PREFILL_ACTIVATION_RESERVE_MIB), float(v))
        for v in p_prefill_transient_vector_mib(chunk_tokens, profile)
    )

def argv_d(
    py: str,
    model: str,
    budgets: List[int],
    s_gb: int,
    m_mib: int,
    store_cfg: str,
    extra: List[str],
    d_bs: int = DEFAULT_D_BS,
    max_kv_per_request: int = CONTEXT_LENGTH_TOKENS,
    x_tokens: int = 0,
    num_continuous_decode_steps: int = 1,
    disable_overlap: bool = False,
    tp_ratio_flags: Sequence[str] = ("--rank-tp-ratio", "auto"),
    token_vector_flags: Sequence[str] = (),
    random_seed: int = RANDOM_SEED,
    barlink_cap_cycles: int = BARLINK_BAR1_CAP_CYCLES,
    census_interval: int = COLLECTIVE_CENSUS_INTERVAL,
    disable_cuda_graph: bool = False,
    admin_api_key: Optional[str] = None,
    # #1386: APPENDED LAST, deliberately -- `argv_d` has no `*` keyword-only
    # marker, so a new parameter anywhere else would be the exact positional
    # shift class #1356 already paid for once in `argv_p`. Every existing
    # caller passes admin_api_key by keyword or stops before it; none can
    # collide with this.
    hicache_disabled: bool = False,
    d_write_policy: str = "write_back",
    # #1369: APPENDED LAST for the same reason as `hicache_disabled` two
    # lines up -- `argv_d` has no `*` marker.
    weights_cpu_backup: bool = True,
    profile: str = PROFILE_QWEN27B,
    # #108: kommt als WERT vom Launcher, nie aus der Env dieses Prozesses.
    d_adopt: bool = False,
    # PDFLIP VISION (D side, 24.09., 27B f09dc0d6e4; H125 on NF): APPENDED LAST
    # (argv_d has no `*` marker). Under `transient` D tokenizes images like P
    # (language_model_only instead of --no-enable-multimodal): the same pad
    # ids reach P's stored pages and the decode gets the mrope delta. D still
    # has no tower and arms no stage (FLLIPER_PDFLIP_VISION stays P-only,
    # build_env); an image inside D's own extent is refused by name at its
    # admission (W123). `off` (default) is byte-identical to every argv
    # before this parameter existed.
    vision: str = VISION_OFF,
    # HW-P1c 1003: APPENDED LAST (argv_d has no `*` marker). Default = the
    # shipped N = 3 window, so every existing caller's argv is byte-identical.
    window_mib: str = D_BARLINK_BAR1_WINDOW_MIB,
) -> List[str]:
    _refuse_if_extra_raises_budget(budgets, list(extra or ()), "D")
    _refuse_if_extra_drops_transient_lmo(vision, extra, "D")
    return [py, "-m", "flliper.launch_server"] + common_flags(
        model, s_gb, m_mib, store_cfg, max_kv_per_request, d_write_policy, "D",
        random_seed, barlink_cap_cycles, census_interval,
        hicache_disabled=hicache_disabled,
        weights_cpu_backup=weights_cpu_backup,
        profile=profile,
        vision=vision,
        n_cards=len(budgets),
    ) + (
        ["--disable-overlap-schedule"] if disable_overlap else []
    ) + (
        # #1241b CONTROL ARM, NOT A TUNING KNOB. Default off, and off means
        # this list is empty -- the shipped argv is byte-identical to the one
        # before this flag existed. Passing it CHANGES THE MEASURED FORM
        # (full-perf validation runs with graphs and spec), so a number taken
        # under it describes the eager form and nothing else.
        ["--disable-cuda-graph"] if disable_cuda_graph else []
    ) + [
        # C1/K2: D's own bs, independent of P's by construction.
        "--max-running-requests", str(d_bs),
        # C11/K5: law 4 enforced where the uncached extent is REAL -- after
        # match_prefix, on extend_input_len, inside get_new_batch_prefill.
        # The front prices with len(text)/3.0 and never re-checks, so a
        # front-side estimate alone let a 19,401-token uncached extent
        # through a 4,096-token grant (weg2zr2 rid pdflip-4-8). 0 = off.
        "--tp-prefill-max-tokens", str(x_tokens),
        # THE STRATEGY IS STATED, NOT INHERITED (FIX 1r/1). Leaving this at
        # 'auto' made D's mamba radix-cache strategy a SIDE EFFECT of the
        # absent disable flag: _mamba_radix_cache_resolution reads
        # `wants_overlap = not view.disable_overlap_schedule`
        # (arg_groups/overrides.py:1225-1239), so switching the overlap
        # schedule on also switched no_buffer -> extra_buffer, and
        # mamba_pool_floor.mamba_ping_pong_slots then charges 2 state slots
        # per running request instead of 0 -- DOUBLING D's device mamba floor,
        # by `2 x --max-running-requests` extra slots per rank, out of the FIXED
        # --rank-gpu-memory-mib budgets. Stated PER RUNNING REQUEST and not as
        # the product, because the product moves with `d_bs` and a comment
        # cannot: the old wording read "16 -> 32 slots at
        # --max-running-requests 8" and was stale the moment the shipped
        # default moved to DEFAULT_D_BS (user order 2026-09-09). The live
        # product is computed, printed and asserted by d_overlap_cost_line /
        # d_mamba_ping_pong_cost, which read the RESOLVED bs -- never this
        # comment. Stated here so the argv is an honest
        # statement of what D runs, and priced in the SCHEDULER: log line
        # (d_overlap_cost_line) so the launcher cannot move D's device
        # residency without the number appearing. The value follows the
        # overlap choice exactly -- it is not a second knob.
        "--mamba-radix-cache-strategy", "no_buffer" if disable_overlap else "extra_buffer",
        # HW-GENERIC 1002 S2 (enabling): D = TP<N> (PP1) over every card;
        # from the topology, "--tp-size 3" on the N=3 release.
        "--tp-size", str(_topo.release_topology(len(budgets)).d_tp), "--pp-size", "1",
        # OVERLAP SCHEDULE ON for D -- by ABSENCE of the disable flag, which
        # is the only way to have it: there is no --enable-overlap-schedule.
        # Every gate that forces it off was checked against THIS argv and
        # none applies: pp_size>1 (D is 1), --enable-pdmux (not passed,
        # server_args.py:19553), device cpu/mps (cuda), sparse-head
        # embeddings and dllm (neither), and the hybrid-mamba resolution
        # (arg_groups/overrides._mamba_radix_cache_resolution), which picks
        # 'extra_buffer' and LEAVES overlap on for Qwen3_5ForConditional-
        # Generation on linear_attn_backend=triton -- that arch is in
        # _MAMBA_EXTRA_BUFFER_ARCHS. If a gate ever does refuse, it is not to
        # be weakened: the launcher's --d-disable-overlap-schedule puts the
        # flag back and logs W41.
        #
        # #1030's justification is PP-only and does not reach D
        # (BSSCALE_0907.md D4).
        "--num-continuous-decode-steps", str(int(num_continuous_decode_steps)),
        "--rank-gpu-memory-mib", ",".join(str(b) for b in budgets),
    ] + list(tp_ratio_flags) + [
        # THE WEIGHT OBJECTIVE IS STATED, NOT ASSUMED (#1017). A per-rank MiB
        # LIST under TP requires the uneven-TP ratio, and 'auto' derives the
        # weights from that list -- but 'auto' is the CAPACITY-FIRST default
        # (server_args.py:838), so shipping it as a literal meant every Weg-2
        # boot chose an objective without saying so. The choice now arrives
        # from d_tp_ratio_decision as --d-tp-objective, priced on the launch
        # line; the default is still 'auto' because the maxkv law makes
        # capacity the default objective, not because nothing was decided.
    ] + spec_flags(producer=False) + d_replayssm_spec_flags() + list(token_vector_flags) + [
        # NO TOKEN VECTOR BY DEFAULT (#1032). What stood here was
        # `--uneven-token-vector 29,19,16 --uneven-token-vector-role seed`, the
        # emitted value of RETRACTED investigation #602. See
        # RETRACTED_SEED_VECTOR_1032 for what boot weg2rg6 measured the runtime
        # doing with it (superseding it to 17,7,8, every rank, every boot) and
        # for why "the install landed" is a reason to stop shipping it rather
        # than a reason it was harmless. An operator vector arrives through
        # d_token_vector_decision, which refuses a retracted one BY TICKET.
        # #1234 C1 -- dcp:0 goes 24 -> 40 MiB. MEASURED, and this one
        # number is the whole regression fix.
        #
        # WHY 24 WAS WRONG. At a 24-MiB window max_payload yields
        # chunk_max = 2 093 056 B, exactly one 4096-byte page below 2 MiB,
        # so 16 rounds carry 100 466 688 B = 4088 tokens. The dcp
        # attention-out combine at --chunked-prefill-size 4096 sends
        # 4096 x 24576 = 100 663 296 B. Miss: 196 608 B, 0.196 %, EIGHT
        # TOKENS -- and every one of the 16 calls per prefill chunk ran on
        # the host-staged gloo plane at 0.68 GB/s instead of bar1's 3.19.
        #
        # WHY 40 AND NOT MORE. The binding gate is not "sum of declared <=
        # 224" but NVML free minus RESERVE_MIB_DEFAULT (32), evaluated when
        # each group builds (barlink_matrix_transport.py:330, :357-364);
        # measured at dcp:0 build time on boot weg2ab1: 74 - 32 = 42 MiB.
        # DCP_0=48 was REFUSED on metal (Bar1WindowRefused, boot weg2ab1).
        # At 40 MiB chunk_max = 3 493 888, so the 96-MiB all_reduce plans to
        # 10 rounds and the 22-MiB q_full all_gather to 7; measured BAR1
        # Used 224/256 MiB per 3080, i.e. the aperture is EXHAUSTED, which
        # is the argument for the derived round bound rather than for a
        # bigger window next time.
        #
        # WHY NOT 25 (the arithmetic minimum for 16 rounds). Because the
        # window also buys all_gather rounds: 12 -> 7 across the same step.
        # ARM 1 of the A/B boot measured w40 at 2361.4 ms of wait per full
        # chunk against the baseline's 4755.7 and NCCL's 3307.6.
        "--barlink-bar1-window-mib", window_mib,
        # #1234 C5: group D is the one place where a silent host-staged
        # 4.7x is a known boot killer, and with the window above no declared
        # class is anywhere near the round budget. The library default stays
        # 'warn'; opting in is a deployment decision, made here.
        #
        # WHAT THIS DOES AND DOES NOT STOP (FIX 1). Only the two size-driven
        # refusal kinds -- 'round' and 'oversize', barlink.py's
        # UNCOVERED_REFUSAL_STOPS -- stop the group. A sub-min_bytes or
        # misaligned collective anywhere in D's process still takes the
        # priced warn path and its small, correct gloo answer; killing a boot
        # for one of those would be an abort wider than the reason above.
        # Under the DERIVED bound the 'round' half is currently inert on this
        # rig (the crossover is scale-invariant and has no root here -- see
        # barlink_bar1.round_budget); it binds when a cap is pinned, which is
        # exactly how boot weg2zr2 failed, and 'oversize' is the reachable
        # half that guards the window this line sits next to.
        "--barlink-uncovered-class", "refuse",
        "--port", str(PORT_D),
    ] + admin_key_flag(admin_api_key) + _adopt_load_format_flag(d_adopt) + extra

def d_extend_vote_fixed_live_verdict(profile: Optional[str], cards: Optional[Sequence[Card]],
                                     model: Optional[str]):
    """VOTE-FIXED-LIVE: ``(digest, StoreVerdict | None, why)`` -- the form's persisted readings priced at the
    profile's START rate (``D_EXTEND_CAP_PER_ROW_MIB``, its smallest value: the larger, safe F; none = rate 0)."""
    dig, why = d_extend_vote_fixed_live_form(profile, cards, model)
    if dig is None:
        return None, None, why
    try:
        rates = [float(x) for x in _pconst(D_EXTEND_CAP_RATE_RECORD, profile) if x is not None]
        rate = min(rates) if rates else 0.0
    except Exception:  # noqa: BLE001 -- no start rate: F = the largest cost (rate 0)
        rate = 0.0
    try:
        v = _vfl.aggregate(_vfl.load_docs(vote_fixed_live_root(), dig), rate)
    except Exception as exc:  # noqa: BLE001 -- an unreadable store is no store
        return dig, None, f"store unreadable ({type(exc).__name__})"
    return dig, v, why

def _p_page_size(model: str, p_bs: int = DEFAULT_P_BS) -> int:
    """``--page-size`` as this launcher actually passes it to group P.

    #1286 F7. The pool model floors twice, as the sizer does
    (``pool_configurator.calculate_pool_sizes``), and the page is the second
    floor. Read off the argv for the same reason ``_max_running_requests`` is:
    the flag lives in ``common_flags`` today and the model must follow it if it
    moves, rather than carrying a second copy of the value. No cut on the
    probe argv (H68c, see ``_max_running_requests``).
    """
    flags = argv_p("py", model, [1, 1, 1], 1, 1, RING_FORM_SENTINEL_STORE_CFG, [], p_bs=int(p_bs),
                   stage_ratio="", attn_stage_ratio="")
    try:
        return int(flags[flags.index("--page-size") + 1])
    except ValueError:
        # Absent means the runtime's own default, which is 1. Not an error:
        # the launcher is allowed to stop stating a flag whose default it wants.
        return 1

def d_plan_inputs(model: str, tp_size: int, d_bs: int,
                  overhead_mib_by_rank: Optional[Sequence[float]] = None):
    """The runtime ``PlanInputs`` describing THIS boot's group D.

    ONE FACTORY (refuter MF-6). Every field is read from the module constant
    the argv builder emits, so the vector this launcher PRICES and the argv it
    SHIPS cannot describe two different configurations. Both callers -- the
    #1017 weight line and the #1241 operating-point rows -- go through here.
    """
    from flliper.srt.uneven_perf import PlanInputs

    return PlanInputs(
        tp_size=int(tp_size),
        model_path=model,
        kv_cache_dtype=KV_CACHE_DTYPE,
        max_running_requests=int(d_bs),
        **spec_plan_fields(),
        # 27B line (24.09.): the GROUP's facts, which this launcher process
        # does not carry in its own environment -- the SSM dtype (priced fp32
        # before: the planner read FLLIPER_MAMBA_SSM_DTYPE off the launcher's
        # env, where it is never set) and the DFLASH window pool (cell 32768 +
        # constant draft reserve instead of 34816 per token).
        # UNIFY S7: the group fact only where the profile ships the early-read
        # flag half (27B TP-D); elsewhere None = the pre-existing env read.
        mamba_ssm_dtype=(_group_fact_value("D", "--mamba-ssm-dtype")
                         if _profile_early_read(_published_profile()) else None),
        **d_window_pool_fields(),
        overhead_mib_by_rank=(list(overhead_mib_by_rank)
                              if overhead_mib_by_rank is not None else None),
    )
