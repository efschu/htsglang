"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Callable
from typing import List
from typing import Sequence
from typing import Tuple

from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip.launcher_d_leaf import D_REPLAYSSM_SPEC_DEFAULT
from flliper.srt.pdflip.launcher_d_mid import D_TOKEN_PLACEMENT_DEFAULT
from flliper.srt.pdflip.launcher_d_p_r2_L0 import STORE_SIDECAR_FACTOR
from flliper.srt.pdflip.launcher_p_l3_leaf import P_CHUNK_POLICY_DEFAULT
from flliper.srt.pdflip.launcher_p_l3_leaf import P_PREFILL_GRAPH_DEFAULT
from flliper.srt.pdflip.launcher_p_l3_leaf import _p_chunk_dynamic_runnable
from flliper.srt.pdflip.launcher_p_l3_leaf import _pconst
from flliper.srt.pdflip.launcher_prepare_sweep_r2_L0_1 import store_sidecar_factor_of
from flliper.srt.pdflip.launcher_refuse_proc_leaf import PROFILE_QWEN27B
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _PROFILE_27B
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _profile_row_or_27b
from flliper.srt.pdflip.launcher_refuse_proc_leaf import _row_format_default

#: MEASURED per-stage prefill cost, ms per layer, boot `bsscale`
#: (/spinning/gpu-arb/weg2/BSSCALE_0907.md, tip 37c884b0b0, table under
#: "The headline of Table A"): per full 4096-token chunk at bs6 PP0 259.1 ms
#: over 32 layers = 8.10, PP1 632.9 over 18 = 35.16, PP2 470.2 over 14 =
#: 33.59, with stage 0 on the 5090 and stages 1-2 on the two 3080s. The two
#: 3080 figures differ by 4.7 %, which is inside the +-10 % per-rank spread
#: that measurement's own A/A repeat established. Used as the ANCHOR for the
#: card-rate library's ratios, and as the whole cost model when no measured
#: library exists.
MEASURED_MS_PER_LAYER = _PROFILE_27B.constant("MEASURED_MS_PER_LAYER")
#: Per-stage NON-LAYER weight + runtime state, MiB: the embedding on stage 0,
#: lm_head plus the MTP/draft head on the last stage, the replicated payload
#: every stage carries, and the per-rank runtime state. Recovered as
#: `weights_and_runtime_state_GiB * 1024 - MEAN_LAYER_MIB * n_layers`:
#:   weg2sb5f  2348.2 / 1100.0 / 3515.6 MiB
#:   weg2rg6   2335.5 / 1111.2 / 3520.1 MiB
#: The two agree to <= 12.7 MiB (<= 0.55 % of the term, <= 600 tokens on the
#: binding stage), which is the residual this constant carries.
P_PP_STAGE_FIXED_MIB = _PROFILE_27B.constant("P_PP_STAGE_FIXED_MIB")
#: Device GDN state per linear layer per sequence SLOT, MiB. Recovered from
#: `mamba state pool / (linear_layers * 20 slots)` on six rank readings:
#: 31.185/31.23/31.16 (weg2sb5f) and 31.19/31.16/31.13 (weg2rg6) MiB per
#: linear layer, i.e. 1.5588 MiB per linear layer per slot. Cross-checked
#: against the boot's own `per_req=51.43 MiB` at 33 linear layers.
P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT = _PROFILE_27B.constant("P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT")
#: UNIFY S3: argparse defaults that come from the booted MODEL'S REGISTRY ROW
#: (pdflip/form.py PROFILES) -- the ONE mechanism for "the row is the default of
#: this CLI flag". Two kinds:
#:  * MEASURED constants of one model: the parser's default is the 27B row (the
#:    launcher's default --profile);
#:  * BEST-FORM fields (user rule 29.09. ~10:15Z: a switch proven on metal is
#:    default-on in the code, not only in the docker profile): the parser's
#:    default stays the code default (off / fixed / capacity), the row turns it
#:    on -- ``replayssm`` -> --d-replayssm-spec.
#: After parsing, an UNSET flag takes the booted profile's row. A given flag,
#: or a value that is no longer the parser default (argparse abbreviations),
#: wins. Each entry: (flag, dest, of(profile, ns) -> the row's value).
PROFILE_ARG_DEFAULTS: Tuple[Tuple[str, str, Callable[[str, object], object]], ...] = (
    ("--pp-cut-measured-ms-per-layer", "pp_cut_measured_ms_per_layer",
     lambda p, ns: _pconst("MEASURED_MS_PER_LAYER", p)),
    ("--pp-cut-stage-fixed-mib", "pp_cut_stage_fixed_mib",
     lambda p, ns: _pconst("P_PP_STAGE_FIXED_MIB", p)),
    ("--pp-cut-mamba-mib-per-linear-layer-per-slot", "pp_cut_mamba_mib_per_linear_layer_per_slot",
     lambda p, ns: float(_pconst("P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT", p))),
    ("--store-sidecar-factor", "store_sidecar_factor", lambda p, ns: store_sidecar_factor_of(p)),
    # H64 (27B xsn436: D-KV +21/+13 %, not slower, user 24.09. ~20:47Z "ja
    # einschalten"; NF x172/x174: TP0 -396 MiB, 257k needle MATCH): both rows
    # carry replayssm=True, every profile passed --d-replayssm-spec on.
    ("--d-replayssm-spec", "d_replayssm_spec",
     lambda p, ns: "on" if _profile_row_or_27b(p).replayssm else "off"),
    # P chunk policy: the row's chunk.policy on its chunk.default_formats
    # (qwen27b dynamic on INT8, chunkab rc9j 128k -13.6 %) -- and only when
    # the flags make that form RUNNABLE (_p_chunk_dynamic_runnable: the rc9j
    # form ran with --p-prefill-graph 512); NF row fixed.
    ("--p-prefill-graph", "p_prefill_graph",
     lambda p, ns: _row_format_default(p, ns, int(_profile_row_or_27b(p).p_prefill_graph),
                                       _profile_row_or_27b(p).p_prefill_graph_formats,
                                       P_PREFILL_GRAPH_DEFAULT)),
    ("--p-chunk-policy", "p_chunk_policy",
     lambda p, ns: (_row_format_default(p, ns, _profile_row_or_27b(p).chunk.policy,
                                        _profile_row_or_27b(p).chunk.default_formats,
                                        P_CHUNK_POLICY_DEFAULT)
                    if _p_chunk_dynamic_runnable(ns) else P_CHUNK_POLICY_DEFAULT)),
    # LS6 (30.09., metal z30y5m dkr27browauthoritybar1fs09301128: 15/15, needle
    # MATCH): the row's P forms. ORDER: --p-prefill-graph BEFORE --p-chunk-policy,
    # whose runnability check reads the graph bucket.
    ("--p-trim-end-anchor", "p_trim_end_anchor",
     lambda p, ns: _profile_row_or_27b(p).end_anchor == "trim"),
    ("--p-host-overlap", "p_host_overlap",
     lambda p, ns: bool(_profile_row_or_27b(p).p_host_overlap)),
    # 27B row 24b: bandwidth on the formats the row measured it on (INT8).
    ("--d-token-placement", "d_token_placement",
     lambda p, ns: _row_format_default(p, ns, _profile_row_or_27b(p).d_token_placement,
                                       _profile_row_or_27b(p).d_token_placement_formats,
                                       D_TOKEN_PLACEMENT_DEFAULT)),
)

def apply_profile_arg_defaults(ns, argv_words: Sequence[str]) -> List[str]:
    """Rewrite the unset :data:`PROFILE_ARG_DEFAULTS` to ``ns.profile``'s row;
    returns the dests that changed (for the 27B its best-form fields; for NF
    since #242 its own stage-fixed post and mamba rate, MEASURED_MS_PER_LAYER
    still borrowed, and its best-form fields)."""
    parser_default = {
        "pp_cut_measured_ms_per_layer": MEASURED_MS_PER_LAYER,
        "pp_cut_stage_fixed_mib": P_PP_STAGE_FIXED_MIB,
        "pp_cut_mamba_mib_per_linear_layer_per_slot": P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT,
        "store_sidecar_factor": STORE_SIDECAR_FACTOR,
        "d_replayssm_spec": D_REPLAYSSM_SPEC_DEFAULT,
        "p_chunk_policy": P_CHUNK_POLICY_DEFAULT,
        "d_token_placement": D_TOKEN_PLACEMENT_DEFAULT,
        "p_prefill_graph": P_PREFILL_GRAPH_DEFAULT,
        "p_trim_end_anchor": False,
        "p_host_overlap": False,
    }
    profile = getattr(ns, "profile", None) or PROFILE_QWEN27B
    changed: List[str] = []
    # PPCUT-INT4AR 24.10.: an operator-stated stage-fixed vector is THEIR
    # measurement of THIS checkpoint; solve_p_cut moves only the record default.
    ns.pp_cut_stage_fixed_given = bool(
        pdflip_form.flag_given(argv_words, "--pp-cut-stage-fixed-mib"))
    for flag, dest, of in PROFILE_ARG_DEFAULTS:
        # A store_true flag's "--no-" twin (LS6) counts as given, too.
        if (pdflip_form.flag_given(argv_words, flag)
                or pdflip_form.flag_given(argv_words, "--no-" + flag[2:])
                or not hasattr(ns, dest)):
            continue
        if getattr(ns, dest) != parser_default[dest]:
            continue
        want = of(profile, ns)
        if want != getattr(ns, dest):
            setattr(ns, dest, want)
            changed.append(dest)
    return changed
