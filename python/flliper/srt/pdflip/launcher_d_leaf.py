"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
import os
import re
import sys

from flliper.srt.compat_shims import name_counterparts as _name_counterparts
from flliper.srt.environ import envs
from flliper.srt.pdflip import bar1_windows as bar1_windows_mod
from flliper.srt.pdflip import budget_rest as _budget_rest
from flliper.srt.pdflip import card_identity
from flliper.srt.pdflip import corridor_budget
from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip import host_ledger
from flliper.srt.pdflip import topology as _topo
from flliper.srt.pdflip import weight_exchange_region
from flliper.srt.pdflip.launcher_types import Card
from flliper.srt.pdflip.launcher_types import DOperatingPointRow
from flliper.srt.pdflip.launcher_types import DTokenVectorDecision
from flliper.srt.pdflip.launcher_types import EarlyReadFact
from flliper.srt.pdflip.launcher_types import MixedRateUnits
from flliper.srt.pdflip.launcher_types import PdFlipLaunchRefused
from flliper.srt.registry import nvml as nvml_registry
import json
import math
import shlex

#: Spec section 1.6 V1 (arm B, graphs resident) derived upper bounds for the
#: dormant residue of a rank: 5090 1,848 MiB, 3080 1,442 MiB -- EXPECTATIONS
#: (record 1d/1f B6), printed beside the measurement, used only for group P's
#: budget (D has not slept yet when P is sized) and graded by W19 at D's
#: first sleep by the front.  Plus D's open BAR1 windows (deviation above).
DC_EXPECT_5090_MIB = 1848
DC_EXPECT_3080_MIB = 1442
#: HW-GENERIC 1002: the two expectations above BY CALIBRATION CLASS
#: (pdflip/card_identity.py) -- the selector, so no reader picks one by a name
#: substring. A class absent here has no expectation: printed UNCALIBRATED.
DC_EXPECT_MIB_BY_CLASS = {"RTX5090": DC_EXPECT_5090_MIB, "RTX3080": DC_EXPECT_3080_MIB}
#: #1444: margin on top of a RECORDED residue.  The front measured group D's
#: residue on this form at 1616-1686 MiB (5090) and 1256-1356 MiB (3080) across
#: the flips of one boot (weg2xsn203), a 100 MiB spread; 256 covers it twice.
#: The constant it replaces over-reserved 1300-1400 MiB per card, because it
#: was read at boot weg2xsn14 when the weights_draft tag was still RESIDENT --
#: the tag has travelled with the exchange since, and P's budget paid for
#: bytes that are no longer there.
DC_RECORD_MARGIN_MIB = 256
#: #227 R1: the named log marker of a boot whose D maximum rests on fewer than this many D sleeps.
DC_MAX_MIN_SLEEPS = 2
DC_ONE_SLEEP_MARK = "DC-ONE-SLEEP-BOOT"
#: ``=0`` prices the xsn14 constant again (the pre-#1444 form).
DC_RECORD_ENV = "FLLIPER_PDFLIP_DC_D_RECORD"
#: How many of the newest group-P boots (same calibration identity, same weight
#: form, every card measured) the expectation takes the per-card MAXIMUM over.
#: Measured scatter of this record under ONE P form (Next Flash, 28./29.09.,
#: profile family -st/-st-cut/-odx, 30 boots): nvml1 1286-1414, nvml0 680-764,
#: nvml2 666-1012 MiB; about one boot in four carries a +250..+350 MiB outlier
#: on one card (09281447 nvml2 1012, 09290450/0500 nvml2 924). Eight boots are
#: the last ~2-6 h of the serving loop: an outlier is inside the window with
#: ~90 % (1 - 0.75^8), and the window stays within today's P form (the older
#: form of 27.09., 1850/1130/1120, is a different P and ages out). A maximum
#: of measurements, not a reserve: no margin is added on top.
P_DORMANT_EXPECT_N = 8
D_RESERVE_RECORD_ENV = "FLLIPER_PDFLIP_D_RESERVE_RECORD_MAX"
#: --spec-form (PLAN_DFLASH2_P_0917). THE SPECULATIVE FORM OF THE BOOT, ONE
#: WRITER (the same MF-6 argument as SPEC_ALGORITHM above): argv_p, argv_d,
#: d_plan_inputs and the D environment all read it here. ``NEXTN`` is the
#: shipping form (the constants above, byte-identical); ``DFLASH`` puts the
#: external DFlash2 draft on both groups -- P as the draft-KV producer
#: (dflash_draft_kv_producer), D as the proposer with the compact draft
#: cache and the window pool.
SPEC_FORM_DEFAULT = "NEXTN"
DFLASH_DRAFT_PATH_DEFAULT = (
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-DFlash2-W8-lued"
)
DFLASH_BLOCK_DEFAULT = 8
DFLASH_WINDOW_DEFAULT = 2048
#: --dflash-produce-on-p (27B user decision 2026-09-24, verbatim: "ne wir lassen
#: den draft produzenten auf P weg. wir lassen seine layer kalt dort liegen im
#: vram, oder im systemram. entscheidung ist gefallen. P ohne draft rechnen").
#: ``off`` = the standard form under --spec-form DFLASH: group P still loads
#: the DFlash2 draft on its last PP stage (cold-resident bytes: same VRAM, same
#: planner cut, same exchange census and flip) but computes nothing with it.
#: ``on`` is the old producer form, A/B only. Measured reason (boot xsn410, P
#: log): DRAFT-KV-PRODUCE took 19.5 s of a 21 s prefill span per 98k prompt.
#: The switch rides group P's ENVIRONMENT only (spec_form_env("P")) -- argv_p,
#: the P FORM key, the census and the ledger are byte-identical under both.
#: UNIFY S6: it is the ``cold|compute`` half of the ``p_draft`` axis
#: (pdflip/form.py, the one writer); only read under DFLASH.
DFLASH_PRODUCE_ON_P_DEFAULT = "off"
#: --d-replayssm-spec (27B ReplaySSM package, S6). ``on`` gives group D
#: ``--enable-linear-replayssm-spec``: its GDN target verify runs the compact
#: spec ring instead of per-draft intermediate states, and the freed
#: "speculative intermediate state" post goes to D's KV pool (27B desk, not an
#: NF value: -0.41 GiB post / +0.41 GiB KV per rank at 18/6 heads, mrr 2). The
#: CODE default (no registry row) is off -- off leaves argv_d and the D pricing
#: byte-identical; the booted profile's row (``ModelProfile.replayssm``, both
#: rows True since the metal: 27B xsn436, NF x172/x174) sets it for an unset
#: flag (apply_profile_arg_defaults, 29.09.). P never gets it: P runs no
#: target verify.
D_REPLAYSSM_SPEC_DEFAULT = "off"
_SPEC_FORM: Dict[str, object] = {
    "form": SPEC_FORM_DEFAULT,
    "draft_path": DFLASH_DRAFT_PATH_DEFAULT,
    "block": DFLASH_BLOCK_DEFAULT,
    "window": DFLASH_WINDOW_DEFAULT,
    "produce_on_p": DFLASH_PRODUCE_ON_P_DEFAULT == "on",
    # HICACHE-DRAFT-TIER (NF pick of the 27B line's 9eef037e8d): "does group
    # P carry the draft-KV producer" as RESOLVED by main (H25 resolve_draft_on_p,
    # and HiCache enabled). None until then: p_group_has_draft_producer reads
    # FLLIPER_PDFLIP_DRAFT_ON_P (the H25 authority, default 0), never the CLI
    # default of --draft-kv-on-p (on, overruled by the order since H25).
    "draft_kv_on_p": None,
    "d_replayssm_spec": D_REPLAYSSM_SPEC_DEFAULT == "on",
    # NF line (H64): --extra-d's --speculative-num-draft-tokens (d_verify_window).
    "d_extra_draft_tokens": None,
}
#: --d-reshard (27B, user order 26.09. ~08:40Z/08:45Z: dynamic D resharding per
#: load class, "nur auf TP1<, nicht fuers nf"). Rule and model live in
#: pdflip/d_reshard.py (pure); design/evidence /spinning/gpu-arb/docs/DYN_D_RESHARD.md.
#: 'off' (default) changes NOTHING: no env, no line, D argv byte-identical.
#: 'wake' with ONE preset is the static MLP family vector (--rank-mlp-ratio on
#: group D; the flip plan reads the loaders' own widths, xchg_manifest.family_ratios)
#: and needs no executor. 'wake' with >= 2 presets and 'live' print the spec and
#: the desk plan, then REFUSE: the rank-side executor (per-preset MLP views, graph
#: sets, per-preset manifest rows, the planner's spread posten) is not wired.
#: Any profile other than qwen27b (NF Form A) is refused, never ignored.
D_RESHARD_DEFAULT = "off"
_D_RESHARD: Dict[str, object] = {"policy": D_RESHARD_DEFAULT, "spec": None, "lines": ()}
#: --d-gc-freeze (SB 26.09., agent FR: three single-rank D-round stalls of ~0.5 s,
#: candidate a Python gen-2 collection).  'on' = group D's schedulers call
#: gc.freeze() once every runner is up (after the CUDA-graph capture), so the
#: boot-time objects leave the collector's generations.  'off' (default) = no env.
D_GC_FREEZE_DEFAULT = "off"
_D_TOKEN_PLACEMENT: Dict[str, object] = {"spec": None}
#: --d-kv-evict-for-placement (27B row 24c, user 26.09.: "fertige requests
#: oder vorherige requests muessen dann natuerlich den vram kvcache frei machen
#: um die bestverteilung zu ermoeglichen"). pdflip/d_kv_evict.py, DYN_D_RESHARD
#: sec. 14: cold, L2-backed radix leaves are demoted copy-free (device free,
#: node keeps its arena rows) so --d-token-placement bandwidth holds its
#: shares at high fill; the group agrees in one fixed-shape MIN collective.
#: 'off' (default) = no env, argv byte-identical.
D_KV_EVICT_DEFAULT = "off"
_D_KV_EVICT: Dict[str, object] = {"spec": None}
#: 19.09. (xsn393, Punkt 5 D-Kapazitaet): the DFLASH draft on D runs SOLO by
#: default -- the host rank holds it whole, the other ranks a meta shadow
#: (measured: decode bs6 58 ms vs 61 ms replicated, KV 259k/193k/208k vs
#: 290k/132k/141k tokens per rank, needle MATCH). `split` restores the
#: replicated per-rank draft (A/B only).
ENV_DFLASH_PLACEMENT = "FLLIPER_PDFLIP_DFLASH_PLACEMENT"
#: #251d: the operator's switch -- the stage follows the wake's demand alone
D_KV_STAGE_BY_DEMAND_KEY = "FLLIPER_PDFLIP_D_KV_STAGE_BY_DEMAND"
#: 29.09.: the lowest KV stage and the step of the stages below the booked S0
D_KV_STAGE_FLOOR_KEY = "FLLIPER_PDFLIP_D_KV_STAGE_FLOOR_TOKENS"
#: #1271: the UNIT a rate is expressed in. X* divides one rate by another, so
#: two rates of different units produce a number with no meaning -- and that is
#: not hypothetical, it is what shipped.
#:
#: ``group_throughput``  -- tokens the GROUP moved / the wall it was busy.
#: ``request_latency``   -- one request's tokens / that request's own wall,
#:                          which under concurrency includes the time it spent
#:                          queued behind its peers and is therefore NOT a rate
#:                          of the group.
#: ``compute_honest``    -- tokens / rank gpu-ms (compute+wait), the per-rank
#:                          instrument both groups carry.
RATE_UNIT_GROUP_THROUGHPUT = "group_throughput"
#: USER ORDER 2026-09-09, verbatim: "der decode bs6 soll mit bs6 (nicht mehr
#: bs4) der standard werden". Group D's default objective moves from the
#: capacity-first `maxkv` split to the bs=6 OPERATING POINT.
#:
#: WHAT THE ORDER TRADES, from dec2c's own A/B rather than from an argument:
#: the bs=6 vector [42,11,11] measured **+10.3 % at bs4 for -1.4 % pool**. So
#: the maxkv law is not repealed -- it is outranked at this one position,
#: because the ranking a boot should follow is the ranking AT THE OPERATING
#: POINT IT RUNS, and D runs decode.
#:
#: `maxkv` stays fully selectable via --d-tp-objective and its argv is
#: byte-identical to the pre-#1241 form; only which arm you get by saying
#: nothing has changed. --d-bs is NOT touched here (separate branch).
D_TP_OBJECTIVE_DEFAULT = "decode-bs6"
#: THE THREE FACTS, ONCE. See :class:`EarlyReadFact` for why each is stated in
#: both currencies and why that is not second bookkeeping.
EARLY_READ_FACTS: Tuple[EarlyReadFact, ...] = (
    EarlyReadFact(
        env_key="FLLIPER_UNEVEN_DCP",
        env_value="1",
        flag=("--uneven-dcp",),
        groups="D",
        reader="server_args.py:9631 _handle_dcp_validation / :12025 "
               "_handle_uneven_tp, both before :7186 _handle_environment_variables",
    ),
    EarlyReadFact(
        env_key="FLLIPER_UNEVEN_DCP_WEIGHTED",
        env_value="1",
        flag=("--uneven-dcp-weighted",),
        groups="D",
        reader="server_args.py:9632 _handle_dcp_validation / :10517 "
               "uneven_weighted_dcp_enabled, both before :7186",
    ),
    EarlyReadFact(
        env_key="FLLIPER_MAMBA_SSM_DTYPE",
        env_value="bfloat16",
        flag=("--mamba-ssm-dtype", "bfloat16"),
        groups="both",
        reader="uneven_perf.py:4648, reached from _handle_uneven_tp at "
               "server_args.py:6976 -- before the flag publishes itself at :7186",
    ),
)
#: Group P's BAR1 windows. The provenance is the SAME arithmetic group D's
#: window carries (#1234 C1) and it lived only in state.deviations: the two
#: groups' windows are sized to fit TOGETHER -- P 24+96 and D 16+32+40 = 208 of
#: the 224 MiB usable per 3080, measured Used 224/256 including the RM
#: carve-out. It is cited here so P's window is not the one number in this file
#: a reader has to go looking for.
P_BARLINK_BAR1_WINDOW_MIB = "24,PP_0=96"
#: DUAL-TP3PP3 (27B line): group P's BAR1 windows while BOTH groups are awake
#: (metal 30.09. boot f9fch3: with P at the flip default 24+96, D's dcp window
#: was refused on both 3080s). The NF release ships no dual layout, but the
#: constant keeps :func:`apply_bar1_windows`' default check well-defined when
#: the dual flags are set explicitly.
DUAL_P_BARLINK_BAR1_WINDOW_MIB = "16,PP_0=64"
#: Group D's BAR1 windows (#1234 C1, the derivation lives at argv_d): the
#: default of --d-barlink-bar1-window-mib (HW-P1c: derived per N by
#: pdflip/bar1_windows.py; this value is the N = 3 / 256 MiB BAR measurement).
D_BARLINK_BAR1_WINDOW_MIB = "16,TP_0=32,DCP_0=40"
#: #239: the marker of the refusal and of the dry-run line.
KV_TOKEN_CUT_MARKER = "#239 KV-TOKEN-SCHNITT"
#: VRAM loop P1c (28.09., user: "macht der Planer es besser und die Werte werden
#: nicht uebernommen?" -- yes: FRACTION-SOLVE D computed DECKE 0.062/0.586/0.491
#: against the profile's 0.060/0.510/0.480 in every NF boot, ~1.3 GiB idle on
#: nv0). The planner OWNS FR_D: its edge wins; a profile value binds only as a
#: named cap (``FLLIPER_PDFLIP_FR_D_CAP`` in --env-d). Default ON.
FR_D_CEILING_ENV = "FLLIPER_PDFLIP_FR_D_CEILING"
FR_D_CAP_ENV = "FLLIPER_PDFLIP_FR_D_CAP"
#: P0 (torch cache cap) must be armed in the SAME D group: the edge leaves ~0 MiB
#: after KV, and without a bounded cache the allocator swing (0.7-3.5 GiB,
#: PDFLIP-VRAM-PEAK rc12z26) takes that memory back at the next peak.
TORCH_CACHE_CAP_ENV = "FLLIPER_PDFLIP_TORCH_CACHE_CAP"
FR_D_CEILING_MARKER = "FR-D-DECKE (VRAM-Loop P1c)"
#: #1356 THE VISION KNOB. `off` (default) boots P and D TEXT-ONLY; `resident`
#: keeps the tower loaded on both groups, which is what every boot up to and
#: including weg2xsn27 did. `resident` is kept for the day the tower becomes a
#: 7th rank transient between the flips -- that is #1356 proper and it is NOT
#: this slice.
#:
#: WHY IT IS A DEFAULT AND NOT AN OPT-IN: weg2xsn27 loaded the tower on all six
#: ranks and replicated it threefold on P -- 879 MiB per P card in the manifest
#: and 2.63 GiB across the host ring, measured. No Weg-2 form serves images,
#: and the front refuses them by name, so the bytes bought nothing.
VISION_OFF = "off"
VISION_RESIDENT = "resident"
#: Task #58 (user order 2026-09-20): the tower is in NO layout resident. An
#: image request loads it onto one card before the P prefill really starts,
#: runs it, and takes it down again.
#:
#: THE ARGV IS NOT `--no-enable-multimodal` FOR THIS ONE, and that is the
#: whole difference. `off` passes that flag, which also switches off the
#: TOKENIZER's image path (`model_config.py:573 is_multimodal`), so no
#: `mm_items` are ever built and there is nothing for a stage to encode.
#: `transient` instead sets the hf-config override `language_model_only`,
#: which `vision_tower_forced_off` (`models/qwen3_vl.py:1230`) also honours:
#: the tokenizer keeps processing images, `self.visual` stays None
#: (`:1286`), `skip_vision_weight` (`:1243`) drops the checkpoint's tower
#: tensors, and `weight_name_needed` now stops them being READ at all.
#: Measured in fn8ag2 (docstring `qwen3_vl.py:1216-1220`).
VISION_TRANSIENT = "transient"
VISION_CHOICES = (VISION_OFF, VISION_RESIDENT, VISION_TRANSIENT)
#: P-COVERS-D-SESSION (user law 02.10. ~07:55Z): the boot-refusal marker.
P_COVERS_D_SESSION_MARK = "P-COVERS-D-SESSION"
_DRAFTER_RE = re.compile(r"HiCache draft KV registered: .*drafter=([0-9a-f]{16}|None)")
_LAYOUT_RE = re.compile(r"#706 canonical DRAFT page active: .*layout=v(\d+) drafter=([0-9a-f]{16}|None)")
#: SCHALTER-HALBPORT 1002: the line naming a --pdflip-vision the registry row set
VISION_DEFAULT_MARKER = "PDFLIP-VISION-DEFAULT"
#: #1032 THE VECTOR THIS LAUNCHER MUST NEVER SHIP AGAIN, and the boot that
#: showed what the runtime does with it.
#:
#: ``--uneven-token-vector 29,19,16 --uneven-token-vector-role seed`` sat on
#: group D's argv. 29,19,16 is the emitted value of RETRACTED investigation
#: #602 (planner/retracted.py REGISTER), so every Weg-2 boot printed the #797
#: PROVENANCE warning and then spent a supersession undoing it.
#:
#: WHAT THE LOG ACTUALLY SHOWS, because the fix must not be justified against
#: a claim the evidence refutes (boot weg2rg6, D log
#: /spinning/evidence-665-f1/boot_weg2_weg2rg6_7f88b1c75d_0908_070324.D.log):
#: the seed did NOT reach the pools. All three "Uneven-DCP token sizing" lines
#: read ratios 17 / 7 / 8, i.e. partition_units(64, measured P_r) gcd-reduced
#: from the measured per-rank capacities 364718 / 154980 / 170464; `vector [29`
#: does not appear on a single sizing line. The install LANDED and superseded
#: the seed, exactly as role='seed' promises.
#:
#: SO THE DEFECT IS NOT "D SERVED ON THE RETRACTED SPLIT" -- it is that a
#: retracted lineage was shipped at all, and that the runtime spends a
#: supersession every boot undoing a number the launcher had no business
#: stating. Normalised to one sum, the seed 29,19,16 against the measured
#: 34,14,16 is +17 % on rank 0 and -26 % on rank 1: had the install NOT landed,
#: that is the split D would have served.
#:
#: The fix is therefore to state nothing. With no vector on the argv the
#: runtime derives its own starting point from THIS boot's budgets
#: (distributed/utils.py:1135, partition_units(64, budget-minus-checkpoint))
#: and the measured install replaces it after profiling -- the same install
#: that already ran, now without withdrawn evidence upstream of it.
#:
#: WHAT IS GIVEN UP, named rather than left to be discovered: no seed means
#: nothing arms ``note_seed_awaiting_supersession``, so the
#: assert_seed_superseded gate does not hold this boot to an install. That
#: gate exists because a SEED carries foreign lineage; the budget-derived
#: fallback carries this boot's own numbers, so an install that does not
#: happen leaves D on its own honest estimate rather than on a withdrawn one.
#: (And the gate's disarm keys on the VERDICT boolean, not on a changed vector
#: -- distributed/utils.py:306 says so in its own docstring -- so "the gate did
#: not fire" was never proof of an install either. The sizing lines are.)
RETRACTED_SEED_VECTOR_1032 = (29, 19, 16)
#: The one line a stage-fit RECORD is read back from (group P's cards in stage
#: order). Parsed by :func:`read_card_power_record`; change both together.
CARD_POWER_MARKER = "PP-CUT CARD-POWER (the record for --pp-cut-stage-fit):"
#: #1453: the KV arena never shrinks below this (8 GiB = 262144 slots of 32 KiB,
#: one 262k-token request), and the excess is rounded UP to the next 0.25 GiB.
ARENA_FROM_LEDGER_FLOOR_GIB = 8.0

def dc_expect_mib(card) -> Optional[int]:
    """The spec-1.6 dormant-residue EXPECTATION of ``card``'s calibration
    class, or None (uncalibrated class: no expectation, never the 3080's)."""
    return DC_EXPECT_MIB_BY_CLASS.get(card_identity.calibration_class(card) or "")

def d_residue_capture_bs(sample: Dict[str, object], line_id) -> Optional[int]:
    """The capture set (group D's ``--max-running-requests``) a D dormant
    sample was measured under: the sample's own field, or -- a sample written
    before the field existed -- the ``group D argv:`` line of its boot's
    front log (the log the line identity already judges it by)."""
    v = sample.get("vram_residue_capture_bs")
    if v is not None:
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    reader = getattr(line_id, "d_max_running_requests", None)
    return reader(sample.get("boot_tag")) if callable(reader) else None

def dc_residue_from_record(
    rec: Optional[Dict[str, object]], cards: Sequence[Card], weight_source: str
) -> Tuple[Optional[Dict[str, int]], str]:
    """Group D's dormant residue per card from the PREVIOUS boot's record.

    ``(per-uuid MiB incl. :data:`DC_RECORD_MARGIN_MIB`, provenance)`` when the
    newest group-D dormant-image record carries a ``vram_residue_mib`` for
    EVERY card of this boot and was measured under the SAME weight form;
    ``(None, why)`` otherwise, and the caller prices the constant.  A record
    from another form is not this form's residue (the whole reason the xsn14
    constant went stale), so the form gate is exact, not fuzzy.
    """
    if os.environ.get(DC_RECORD_ENV, "1").strip() == "0":
        return None, f"{DC_RECORD_ENV}=0 -> constant"
    if not isinstance(rec, dict):
        return None, "no group-D dormant-image record in the sidecar -> constant"
    tag = str(rec.get("boot_tag", "?"))
    form = str(rec.get("vram_residue_form", "") or "")
    if form != str(weight_source):
        return None, (f"record of boot {tag} measured form {form!r}, this boot is "
                      f"{str(weight_source)!r} -> constant")
    vals = rec.get("vram_residue_mib") or {}
    if not isinstance(vals, dict):
        return None, f"record of boot {tag} carries no vram_residue_mib -> constant"
    out: Dict[str, int] = {}
    parts = []
    for c in cards:
        v = vals.get(c.uuid)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
            return None, (f"record of boot {tag} carries no residue for card "
                          f"{c.uuid} ({c.name}) -> constant")
        out[c.uuid] = int(v) + DC_RECORD_MARGIN_MIB
        parts.append(f"nvml{c.nvml_index} {c.name} {int(v)}+{DC_RECORD_MARGIN_MIB}={out[c.uuid]}")
    return out, (f"boot {tag} at {rec.get('at', '?')} form {form!r}: "
                 + ", ".join(parts) + " MiB")

def dc_residue_max_from_records(rows: Sequence[Dict[str, object]], cards: Sequence[Card], weight_source: str,
                                n: Optional[int] = None) -> Tuple[Optional[Dict[str, int]], str]:
    """#227 MUSS-1: group D's dormant residue per card = the MAXIMUM over ALL D sleeps of the newest ``n``
    (default :data:`P_DORMANT_EXPECT_N`) boots whose samples name EXACTLY this boot's cards under the same
    weight form -- bare, no margin ("a maximum of measurements, not a reserve"; the caller adds the slack).

    A boot measured on the #227 rule stamps ``vram_residue_kind`` (``first_sleep`` + one
    ``max_over_sleeps`` sample per new maximum, front ``_note_d_residue_reading``); its value is the maximum
    over its samples. A LEGACY boot (first sleep only, no kind) contributes first sleep +
    :data:`DC_RECORD_MARGIN_MIB` (the #1444 form: its later sleeps were never recorded -- measured growth
    +244..+484 MiB, so the bare first sleep would book too little). ``(None, why)`` when no exact-card-set
    boot exists, or when none of the selected boots carries the kind (all legacy: the caller keeps the
    pre-#227 newest-sample form, byte for byte).

    R1: a measured boot's samples carry ``vram_residue_sleeps`` (how many D sleeps they rest on). A boot with
    FEWER than :data:`DC_MAX_MIN_SLEEPS` has not reached its maximum yet (it ended early; the growth after the
    first sleep was measured up to +484 MiB), so it contributes first sleep + :data:`DC_RECORD_MARGIN_MIB` like a
    legacy boot, named :data:`DC_ONE_SLEEP_MARK`. A measured boot WITHOUT the counter (written by a build
    between the first #227 head and R1 -- none ran on metal) keeps the kind-based rule: bare."""
    n = P_DORMANT_EXPECT_N if n is None else int(n)
    want = {c.uuid for c in cards}
    by_boot: Dict[str, List[Dict[str, object]]] = {}
    for r in rows or ():
        if not isinstance(r, dict):
            continue
        vals = r.get("vram_residue_mib")
        if not isinstance(vals, dict) or set(vals) != want:
            continue
        if str(r.get("vram_residue_form", "") or "") != str(weight_source):
            continue
        if any(not isinstance(vals[u], (int, float)) or isinstance(vals[u], bool) or vals[u] <= 0 for u in want):
            continue
        by_boot.setdefault(str(r.get("boot_tag", "?")), []).append(r)
    if not by_boot:
        return None, f"no group-D sample of exactly this card set under form {str(weight_source)!r}"
    boots = sorted(by_boot.items(), key=lambda kv: max(str(x.get("at", "")) for x in kv[1]), reverse=True)[:n]
    if not any(x.get("vram_residue_kind") for _t, rs in boots for x in rs):
        return None, "only legacy first-sleep samples (no vram_residue_kind) -> pre-#227 newest-sample form"
    out: Dict[str, int] = {}
    parts = []
    for tag, rs in boots:
        measured = any(x.get("vram_residue_kind") for x in rs)
        counted = [int(x["vram_residue_sleeps"]) for x in rs
                   if isinstance(x.get("vram_residue_sleeps"), int) and not isinstance(x.get("vram_residue_sleeps"), bool)]
        one_sleep = bool(measured and counted and max(counted) < DC_MAX_MIN_SLEEPS)
        add = DC_RECORD_MARGIN_MIB if (not measured or one_sleep) else 0
        for u in want:
            v = max(int(x["vram_residue_mib"][u]) for x in rs) + add
            out[u] = max(out.get(u, 0), v)
        kind = ("max_over_sleeps" if any(x.get("vram_residue_kind") == "max_over_sleeps" for x in rs)
                else f"{DC_ONE_SLEEP_MARK} first sleep +{DC_RECORD_MARGIN_MIB}: {max(counted)} D sleep seen, "
                     f"the maximum is not reached yet" if one_sleep
                else "first_sleep" if measured else f"LEGACY first sleep +{DC_RECORD_MARGIN_MIB}")
        parts.append(f"{tag} [{kind}]")
    return out, (f"MAX over all D sleeps of the newest {len(boots)} of {len(by_boot)} boot(s) of exactly this "
                 f"card set, form {str(weight_source)!r}, bare: "
                 + ", ".join(f"nvml{c.nvml_index} {c.name} {out[c.uuid]}" for c in cards)
                 + " MiB; boots " + ", ".join(parts))

def dormant_max_from_records(
    recs: Optional[Sequence[Dict[str, object]]], cards: Sequence[Card], weight_source: str,
    n: int = P_DORMANT_EXPECT_N, group: str = "P",
) -> Tuple[Optional[Dict[str, int]], str]:
    """A group's MEASURED dormant VRAM residue per card (``group`` names it in
    the provenance; the caller reads that group's records): the per-card MAXIMUM
    over the newest ``n`` group-P dormant-image records (one per boot) that
    the front stamped at P's first sleep (``vram_residue_mib``, the same
    field #1444 reads for group D), measured under the SAME weight form and
    carrying a positive value for EVERY card of this boot.

    ``(per-uuid MiB, provenance naming n, the boots and the maximum)``;
    ``(None, why)`` when no such record exists (UNMEASURED -- the caller names
    its fallback). A record of another form or missing a card is skipped by
    name, never merged partially. Priced BARE, no margin: the real D pass
    after P's sleep charges the launcher's own reading of the same residue
    bare too (``budgets_from_dc(cards, dc_p, ...)``), and the served growth of
    P's dormant image has its own term (``served_dormant_growth``). This is
    the one reader for "D's budget after P's sleep" before P has slept -- the
    expectation passes and an early D start read it here, not from a second
    source."""
    if not recs:
        return None, f"no group-{group} dormant-image record in the sidecar"
    by_boot: Dict[str, Dict[str, object]] = {}
    skipped = []
    for rec in sorted((r for r in recs if isinstance(r, dict)),
                      key=lambda r: str(r.get("at", "")), reverse=True):
        tag = str(rec.get("boot_tag", "?"))
        if tag in by_boot:
            continue
        form = str(rec.get("vram_residue_form", "") or "")
        vals = rec.get("vram_residue_mib") or {}
        if form != str(weight_source) or not isinstance(vals, dict):
            skipped.append(f"{tag} (form {form!r})")
            continue
        miss = [c for c in cards if not isinstance(vals.get(c.uuid), (int, float))
                or isinstance(vals.get(c.uuid), bool) or vals.get(c.uuid) <= 0]
        if miss:
            skipped.append(f"{tag} (no residue for nvml{miss[0].nvml_index})")
            continue
        by_boot[tag] = rec
        if len(by_boot) >= int(n):
            break
    if not by_boot:
        return None, (f"no group-{group} record measured under form {str(weight_source)!r} on "
                      f"every card (skipped: {', '.join(skipped[:4]) or 'none'})")
    out: Dict[str, int] = {}
    parts = []
    for c in cards:
        vals = [int(r["vram_residue_mib"][c.uuid]) for r in by_boot.values()]
        out[c.uuid] = max(vals)
        parts.append(f"nvml{c.nvml_index} {out[c.uuid]} (min {min(vals)})")
    return out, (f"MAX over the newest {len(by_boot)} of N={int(n)} group-{group} records form "
                 f"{str(weight_source)!r}: " + ", ".join(parts) + " MiB; boots "
                 + ", ".join(by_boot))

def d_reserve_from_records_or_census(
    cards: Sequence[Card], dc_expect_d: Mapping[str, int], census_reserve: Mapping[str, int],
    d_max: Optional[Mapping[str, int]], slack_mib: int, record_priced: bool,
) -> Tuple[Dict[str, int], List[str]]:
    """Group D's dormant-residue RESERVE per card on the census-priced profile
    (Next Flash): what P's budget subtracts, the front's ``--dc-reserve`` and
    the W19 bolt at D's first sleep grade.

    With D records of this identity and form (``d_max`` = per-card maximum over
    the newest boots, :func:`dormant_max_from_records`): that maximum +
    :data:`DC_RECORD_MARGIN_MIB` + slack -- the same margin the single #1444
    record carried. The census is only reported. Before (still without
    records): per card the larger of the newest record and the xchg census.
    rc12z30r3: census 1140/1538 (nvml0/2) against D records 612-680, so P's
    budget carried ~460/860 MiB per 3080 for a D residue that was never there
    -- VRAM that belongs to P's resident experts. The fnFL2x111 lesson (an
    OLDER census undercut a NEWER record -> W19) holds: a maximum of recent
    same-form measurements, never a single low sample.
    ``FLLIPER_PDFLIP_D_RESERVE_RECORD_MAX=0`` keeps the old rule."""
    out = dict(dc_expect_d)
    won: List[str] = []
    use_max = (d_max is not None
               and os.environ.get(D_RESERVE_RECORD_ENV, "1").strip() != "0")
    for c in cards:
        if c.uuid not in census_reserve:
            continue
        census = int(census_reserve[c.uuid])
        if use_max:
            out[c.uuid] = int(d_max[c.uuid]) + DC_RECORD_MARGIN_MIB + int(slack_mib)
            won.append(f"nvml{c.nvml_index}=record-max({out[c.uuid]}; census {census} reported)")
            continue
        record = int(dc_expect_d[c.uuid]) if record_priced else 0
        if record > census:
            out[c.uuid] = record
            won.append(f"nvml{c.nvml_index}=record({record})>census({census})")
        else:
            out[c.uuid] = census
            won.append(f"nvml{c.nvml_index}=census({census})>=record({record})")
    return out, won

def d_expect_check_lines(cards: Sequence[Card], expected: Optional[Mapping[str, int]],
                         dc_p: Mapping[str, int]) -> List[str]:
    """The real pass, after P's sleep: P's measured residue against what the
    expectation priced, per card. ``[]`` when no expectation was priced here.
    UNDER means the pinned D form gets less than it was solved for -- the
    solve that follows caps the scratch or refuses (W122) by name."""
    if not expected:
        return []
    parts = []
    for c in cards:
        e, m = int(expected.get(c.uuid, 0)), int(dc_p.get(c.uuid, 0))
        parts.append(f"nvml{c.nvml_index} P real {m} vs expectation {e} MiB "
                     f"({'UNDER by ' + str(m - e) if m > e else 'ok, ' + str(e - m) + ' spare'})")
    return ["D-EXPECT CHECK (after P's sleep): " + "; ".join(parts)]

def _csv(values: Sequence[int]) -> str:
    """One vector, one rendering: the CSV form the two count flags take.

    FIX 2.  ``argv_p`` used to carry ``"32,18,14"`` / ``"8,4,4"`` as bare
    default arguments -- a THIRD statement of the constants above, which the
    #1233 fix 5 one-definition property had already closed once and which
    ``test_neither_score_vector_survives_anywhere_as_a_bare_literal`` was
    written to catch.  It caught it; the reading was wrong, not the guard.
    """
    return ",".join(str(int(v)) for v in values)

def d_reshard_depth(raw) -> Optional[int]:
    """--d-reshard-depth: 'robust' -> None (the measured ladder 10k..240k, bs1/bs2),
    else the expected per-request depth in tokens ('131072', '128k')."""
    r = str(raw or "robust").strip().lower()
    if r == "robust":
        return None
    mult = 1024 if r.endswith("k") else 1
    n = int(float(r[:-1] if r.endswith("k") else r) * mult)
    if n <= 0:
        raise ValueError(f"--d-reshard-depth {raw!r}: expected 'robust' or a positive token count")
    return n

def _d_reshard_seg_placement(ns) -> Tuple[str, Optional[Tuple[float, ...]]]:
    """The token placement the segment objective prices (the one this boot runs)."""
    if str(getattr(ns, "d_token_placement", "capacity") or "capacity") != "bandwidth":
        return "capacity", None
    raw = str(getattr(ns, "d_token_shares", "auto") or "auto")
    return "bandwidth", (None if raw == "auto" else tuple(float(x) for x in raw.split(",")))

def d_reshard_lines() -> Tuple[str, ...]:
    return tuple(_D_RESHARD.get("lines") or ())

def d_reshard_argv(d_ratio_flags: Sequence[str], extra_d: Sequence[str]) -> List[str]:
    """Group D's extra argv: [] under 'off'; under a single-preset 'wake' the MLP
    family vector. Refuses a base vector the presets were not built for, and a
    hand-written --rank-mlp-ratio next to it (two authors of one vector)."""
    spec = _D_RESHARD.get("spec")
    if spec is None:
        return []
    flags = list(d_ratio_flags)
    base = ",".join(str(x) for x in spec.base)
    if "--rank-tp-ratio" not in flags or flags[flags.index("--rank-tp-ratio") + 1] != base:
        raise SystemExit(f"--d-reshard {spec.policy}: REFUSED -- group D's weight vector is "
                         f"{flags} but the presets are built on --rank-tp-ratio {base}")
    if any(str(a).startswith("--rank-mlp-ratio") for a in extra_d):
        raise SystemExit("--d-reshard: REFUSED -- --extra-d already carries --rank-mlp-ratio")
    return ["--rank-mlp-ratio", ",".join(str(u) for u in spec.presets[0].mlp)]

def d_reshard_env() -> Dict[str, str]:
    """Group D's environment for the reshard; {} under 'off'."""
    from flliper.srt.pdflip import d_reshard as _dr

    spec = _D_RESHARD.get("spec")
    if spec is None:
        return {}
    env = {_dr.POLICY_ENV: spec.policy, _dr.SPEC_ENV: spec.to_json()}
    if spec.objective != _dr.OBJECTIVE_SUM:
        env[_dr.OBJECTIVE_ENV] = spec.objective  # the D-SPEED table on group D follows it
    return env

def d_gc_env(ns) -> Dict[str, str]:
    from flliper.srt.pdflip import gc_instrument as _gci

    if str(getattr(ns, "d_gc_freeze", D_GC_FREEZE_DEFAULT) or D_GC_FREEZE_DEFAULT) != "on":
        return {}
    return {_gci.FREEZE_ENV: "1"}

def d_token_placement_env() -> Dict[str, str]:
    """Group D's env for the token placement; {} under 'capacity'."""
    from flliper.srt.pdflip import d_token_placement as _dtp

    spec = _D_TOKEN_PLACEMENT.get("spec")
    return {} if spec is None else {_dtp.ENV: spec.to_json()}

def apply_d_kv_evict(ns) -> None:
    from flliper.srt.pdflip import d_kv_evict as _dke

    mode = str(getattr(ns, "d_kv_evict_for_placement", D_KV_EVICT_DEFAULT) or D_KV_EVICT_DEFAULT)
    _D_KV_EVICT["spec"] = None
    if mode not in ("off", "on"):
        raise SystemExit(f"--d-kv-evict-for-placement {mode!r}: one of ['off', 'on']")
    if mode == "off":
        return
    if _D_TOKEN_PLACEMENT.get("spec") is None:
        raise SystemExit("--d-kv-evict-for-placement on: REFUSED without --d-token-placement bandwidth "
                         "(it evicts FOR the bandwidth placement; under 'capacity' there is nothing to hold)")
    try:
        spec = _dke.EvictSpec(min_idle_s=float(getattr(ns, "d_kv_evict_min_idle_s", 20.0)),
                              publish_max=int(getattr(ns, "d_kv_evict_publish_max", 8)))
        spec.validate()
    except (ValueError, _dke.EvictError) as exc:
        raise SystemExit(f"--d-kv-evict-for-placement on: {exc}")
    _D_KV_EVICT["spec"] = spec

def d_kv_evict_env() -> Dict[str, str]:
    """Group D's env for the placement eviction; {} under 'off'."""
    from flliper.srt.pdflip import d_kv_evict as _dke

    spec = _D_KV_EVICT.get("spec")
    return {} if spec is None else {_dke.ENV: spec.to_json()}

def dflash_placement() -> str:
    # 19.09. (user order): D shards the draft across the TP ranks ("split",
    # the server default); "solo" (draft whole on one rank, xsn393) is the A/B
    # opt-in, never the standard -- a replicated or solo draft is refused on D.
    v = (os.environ.get(ENV_DFLASH_PLACEMENT, "split") or "split").strip().lower()
    return "solo" if v == "solo" else "split"

def dflash_produce_on_p() -> bool:
    """--dflash-produce-on-p as installed by :func:`apply_spec_form`."""
    return bool(_SPEC_FORM.get("produce_on_p", DFLASH_PRODUCE_ON_P_DEFAULT == "on"))

def d_replayssm_spec() -> bool:
    """--d-replayssm-spec as installed by :func:`apply_spec_form`."""
    return bool(_SPEC_FORM.get("d_replayssm_spec", D_REPLAYSSM_SPEC_DEFAULT == "on"))

def _argv_int_list(extra: str, flag: str) -> Optional[List[int]]:
    """The integers after the LAST ``flag`` of an --extra-* string (argparse
    ``nargs='+'``: up to the next ``--``), or None when the flag is absent."""
    if not extra or not flag:
        return None
    try:
        parts = shlex.split(str(extra))
    except ValueError:
        return None
    found: Optional[List[int]] = None
    for i, tok in enumerate(parts):
        vals: List[str] = []
        if tok == flag:
            j = i + 1
            while j < len(parts) and not parts[j].startswith("--"):
                vals.append(parts[j])
                j += 1
        elif tok.startswith(flag + "="):
            vals = [v for v in tok[len(flag) + 1:].replace(",", " ").split() if v]
        else:
            continue
        try:
            found = [int(v) for v in vals]
        except ValueError:
            found = None
    return found

def d_seat_vram_armed(env_d) -> bool:
    """H95c: FLLIPER_OPT_PDFLIP_D_SEAT_VRAM truthy in the D group's env."""
    return str((env_d or {}).get("FLLIPER_OPT_PDFLIP_D_SEAT_VRAM", "")).strip().lower() in (
        "1", "true", "yes", "on")

def d_kv_stage_floor_tokens(env: Mapping[str, str]) -> int:
    """29.09. (Nutzer 12:35Z, Grundgesetz): the floor stage below S0 in tokens --
    stages floor, 2 x floor, ... below the booked S0 (0 = none, byte-identical).
    The D group's env wins over the environ default."""
    raw = (env or {}).get(D_KV_STAGE_FLOOR_KEY)
    if raw is None:
        from flliper.srt.environ import envs

        return max(0, int(envs.FLLIPER_PDFLIP_D_KV_STAGE_FLOOR_TOKENS.get() or 0))
    try:
        return max(0, int(float(str(raw).strip() or 0)))
    except ValueError:
        return 0

def d_kv_stage_by_demand(env: Mapping[str, str]) -> bool:
    """#251d: FLLIPER_PDFLIP_D_KV_STAGE_BY_DEMAND in the D group's env; a D env that
    does not name it gets the rank's own default (environ.py, ON since 29.09.),
    so the launcher prices the form the rank will run."""
    raw = (env or {}).get(D_KV_STAGE_BY_DEMAND_KEY)
    if raw is None:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_D_KV_STAGE_BY_DEMAND.get())
    return str(raw).strip().lower() in ("1", "true", "yes", "on")

def er_trim_cell(fit, host_rank: int) -> int:
    """The planner's trim cell of a fit (0 without one)."""
    if fit is None:
        return 0
    from flliper.srt.planner import expert_residency as er

    return int(er.kv_stage_trim_cell(fit, host_rank))

def _drop_group_env(spec: str, key: str) -> str:
    return ";".join(x for x in str(spec or "").split(";")
                    if x.strip() and x.split("=", 1)[0].strip() != key)

def d_seat_graph_mib(ns) -> Optional[List[float]]:
    """--d-seat-graph-mib: the measured decode-graph cost per extra seat, one
    value for every rank or one per rank; None = not given."""
    raw = str(getattr(ns, "d_seat_graph_mib", "") or "").strip()
    if not raw:
        return None
    return [float(x) for x in raw.split(",") if x.strip()]

def draft_on_p_line(draft_on_p: bool, provenance: str, stripped: Sequence[str]) -> str:
    """The one line naming which H25 form ran and where D's draft lives."""
    if draft_on_p:
        return (f"PDFLIP DRAFT-ON-P: on ({provenance}) -- group P carries the MTP head as "
                "draft-KV producer and as the exchange partner of D's draft")
    return (
        f"PDFLIP DRAFT-ON-P: off ({provenance}) -- group P boots with NO draft "
        f"(stripped from --extra-p: {list(stripped) or 'nothing'}); weights_draft leaves "
        "the family, D parks its draft in pinned system RAM while P runs "
        "(PDFLIP-DRAFT-PARK/UNPARK) and starts every flipped request DRAFT-COLD"
    )

def draft_kv_off_line() -> str:
    """The one line the launcher prints when the producer is switched OFF.

    An A-B arm that reads like the default is how a measurement gets
    attributed to the wrong tree.  This says, in the boot's own log, exactly
    which of the two forms ran and what it costs.
    """
    return (
        "PDFLIP DRAFT-KV-ON-P: off -- group P boots without the MTP head; "
        "D reads KV+Mamba from the store, draft state is cold after every "
        "flip (user order 2026-09-07 stands; this form is the rg6 baseline)"
    )

def derive_x_star(flip_s: float, r_d: float, r_p: float, floor_tokens: int,
                  *, unit_d: str = RATE_UNIT_GROUP_THROUGHPUT,
                  unit_p: str = RATE_UNIT_GROUP_THROUGHPUT) -> int:
    """X* = 2*flip_s / (1/r_D - 1/r_P), floored (spec section 0, K5, A1-4).

    THE BREAK-EVEN of the round trip: below X* it is cheaper for D to
    prefill the request itself than to pay two flips so P can do it faster.
    ONE number, TWO consumers -- the per-request bound of law 4 and C7's
    aggregate departure latch -- so there is no second hand number.

    Refuses (ValueError) rather than guessing when the inputs cannot
    produce a break-even: a D that is not slower than P has none.
    """
    # #1271 (a): REFUSE MIXED UNITS BY NAME, before any arithmetic.
    #
    # THE SHIPPED DEFECT, measured. `r_P` was
    # `(prompt_tokens - cached_tokens) / wall` off the front's own
    # `PDFLIP-SERVED group=P leg=1` line (launcher.py:445-447, appended at
    # :485-489) -- a PER-REQUEST wall, taken while the front runs the drain at
    # `p_concurrency=8`, so it counts the time a leg sat behind its peers.
    # `r_D` came from `verdict=single_prefill` legs, i.e. concurrency ONE.
    # The formula therefore divided a concurrent-latency rate by a
    # single-request rate and called the quotient a break-even.
    #
    # What that cost: on sb1's front log the SAME boot yields r_P = 1964 tok/s
    # under the shipped `unc >= 4096` filter (23 samples), 528 tok/s over all
    # 79 legs, and ~3.1k tok/s as drain-aggregate. rg6 gave 3096 on 8 samples.
    # r_P did not halve between boots -- the SAMPLE changed: the filter admits
    # only large-uncached legs, which are the cold, least-queued ones, and as
    # more of a drain's later legs qualify the median falls. P's compute-honest
    # rate was ~6594-8325 tok/s across both boots, i.e. flat.
    if unit_d != unit_p:
        raise MixedRateUnits(
            f"X* would divide r_D measured as {unit_d!r} by r_P measured as "
            f"{unit_p!r}. These are not the same quantity: a request_latency "
            f"rate under concurrency counts queueing behind peers, a "
            f"group_throughput rate does not, and a compute_honest rate counts "
            f"neither. 1/r_D - 1/r_P is only a time-per-token difference when "
            f"both sides are the SAME measurement. Refusing rather than "
            f"returning a number with no unit (#1271)."
        )
    if flip_s <= 0 or r_d <= 0 or r_p <= 0:
        raise ValueError(f"non-positive input: flip_s={flip_s} r_d={r_d} r_p={r_p}")
    denom = (1.0 / r_d) - (1.0 / r_p)
    if denom <= 0:
        raise ValueError(
            f"r_D {r_d:.0f} >= r_P {r_p:.0f}: no break-even exists, the round trip never pays"
        )
    return max(int(floor_tokens), int(2.0 * flip_s / denom))

def d_hold_active(d_hold_s: Optional[float]) -> bool:
    """RC2 review (L4): is ``--d-hold-s`` ON? Unset OR 0 is OFF, like
    ``--d-short-drain-tokens 0`` and as the >= 0 refusal already promised
    ("0 / unset = off"). A 0 s hold is not a hold: it used to release a held
    backlog the moment D went free -- a different policy under the same flag.
    ONE definition for the argv and the IDLE POLICY line, so they agree."""
    return d_hold_s is not None and float(d_hold_s) > 0

def early_read_flags(group: str) -> List[str]:
    """The argv half of :data:`EARLY_READ_FACTS` for one group."""
    out: List[str] = []
    for fact in EARLY_READ_FACTS:
        if fact.groups in ("both", group):
            out.extend(fact.flag)
    return out

def early_read_provenance() -> str:
    """One line naming every doubly-stated fact and the reader that needs it."""
    return "PDFLIP EARLY-READ ENV (#1235): " + "; ".join(
        "%s=%s == %s [groups %s, pre-promotion reader %s]"
        % (f.env_key, f.env_value, " ".join(f.flag), f.groups, f.reader)
        for f in EARLY_READ_FACTS
    )

def d_kv_token_cut(ns):
    """#239: what ``--d-kv-token-cut`` hands the D solve: ``None`` (off),
    ``expert_residency.KV_TOKEN_CUT_MAXMIN`` / ``KV_TOKEN_CUT_JOINT`` (S2b), or
    the stated ratio vector."""
    from flliper.srt.planner import expert_residency as _er

    raw = str(getattr(ns, "d_kv_token_cut", pdflip_form.KV_TOKEN_CUT_OFF)
              or pdflip_form.KV_TOKEN_CUT_OFF).strip()
    if raw == pdflip_form.KV_TOKEN_CUT_OFF:
        return None
    if raw in (_er.KV_TOKEN_CUT_MAXMIN, _er.KV_TOKEN_CUT_JOINT, _er.KV_TOKEN_CUT_OWNED):
        return raw
    if raw.startswith(_er.KV_TOKEN_CUT_OWNED_FORCED_PREFIX):
        # #239 M1b: shares forced, ownership + FR_D solved by the planner
        try:
            _er.owned_cut_request(raw)
        except ValueError as exc:
            raise PdFlipLaunchRefused(f"{KV_TOKEN_CUT_MARKER}: {exc}") from None
        return raw
    try:
        vec = [float(x) for x in raw.split(",") if x.strip()]
    except ValueError:
        vec = []
    if not vec or any(x < 0 for x in vec) or sum(vec) <= 0:
        raise PdFlipLaunchRefused(
            f"{KV_TOKEN_CUT_MARKER}: --d-kv-token-cut {raw!r} is neither "
            f"{pdflip_form.KV_TOKEN_CUT_OFF!r}, {_er.KV_TOKEN_CUT_MAXMIN!r}, "
            f"{_er.KV_TOKEN_CUT_JOINT!r}, {_er.KV_TOKEN_CUT_OWNED!r} nor a ratio "
            f"vector of the D ranks")
    return tuple(vec)

def d_seat_table_form(plan, kv_cut_kw: dict, fractions):
    """#239 S3f: what the H95 seat table solves at every seat count. Under an
    owned cut (``owned`` / ``owned:<v>``) the planner has SOLVED ownership,
    cut and FR_D once; the seat table recomputes exactly that form (the cut
    as a fixed vector, FR_D as solved) and never re-solves the ownership --
    a second owned solve starts from the already published vector as its x1
    base and refuses every row (dry run 28.09. 17:50Z, 'keine tragbare Form
    ... Eigentum ab 215,113,160'). Any other cut: unchanged."""
    from flliper.srt.planner import expert_residency as _er

    if (not kv_cut_kw or not _er.owned_cut_request(kv_cut_kw.get("kv_token_shares"))[0]
            or not getattr(plan, "solved_owner_ratio", None)):
        return dict(kv_cut_kw), [float(x) for x in fractions]
    out = dict(kv_cut_kw)
    out["kv_token_shares"] = tuple(float(x) for x in plan.kv_token_cut)
    fr = plan.solved_fractions or fractions
    return out, [float(x) for x in fr]

def d_fr_ceiling_adopt(plan, fractions, env_d: Mapping[str, str], owned: bool,
                       map_built: bool = False
                       ) -> Tuple[Optional[List[float]], List[str]]:
    """The fraction group D boots with when the planner's edge is higher than
    the stated FR_D, or ``(None, lines)`` when the stated one stays.

    Never under an owned cut (#239 S3f publishes FR_D itself) and never over a
    fraction the planner already solved (S2b). Raises only: a rank whose edge
    is below its stated FR is the W122/scratch-cap path's business, not this
    one's. Every rank is named, adopted or not.

    rc12z30b: never after the Platztausch map is built (``map_built``) -- the
    map read FR_D from --extra-d before P started, and a raised FR_D after it
    is a D form the map does not describe (the rc12z29b death class)."""
    given = [float(x) for x in fractions]
    if str(env_d.get(FR_D_CEILING_ENV, "1")).strip() == "0":
        return None, [f"{FR_D_CEILING_MARKER}: {FR_D_CEILING_ENV}=0 -- gegeben {given} bleibt"]
    if owned or getattr(plan, "solved_fractions", ()) or getattr(plan, "solved_owner_ratio", ()):
        return None, []
    fits = list(getattr(plan, "fits", ()) or ())
    if len(fits) != len(given) or any(f.ceiling_fraction is None for f in fits):
        return None, [f"{FR_D_CEILING_MARKER}: keine Kante auf jedem Rang -- gegeben {given} bleibt"]
    ceil = [float(f.ceiling_fraction) for f in fits]
    cap_raw = str(env_d.get(FR_D_CAP_ENV, "")).strip()
    cap: Optional[List[float]] = None
    if cap_raw:
        try:
            cap = [float(x) for x in cap_raw.split(",")]
        except ValueError:
            cap = None
        if cap is None or len(cap) != len(given):
            return None, [f"{FR_D_CEILING_MARKER}: {FR_D_CAP_ENV}={cap_raw!r} unlesbar -- "
                          f"gegeben {given} bleibt"]
    new = []
    lines = []
    for r, (g, c) in enumerate(zip(given, ceil)):
        target = min(c, cap[r]) if cap is not None else c
        val = max(g, target)
        new.append(val)
        why = ("Decke" if cap is None or c <= cap[r] else f"Decke {c:.3f} gekappt auf "
               f"{FR_D_CAP_ENV} {cap[r]:.3f}")
        lines.append(f"{FR_D_CEILING_MARKER} rang{r}: gegeben {g:.3f} -> {val:.3f} ({why}"
                     + (", Decke unter gegeben: bleibt" if c < g else "") + ")")
    if all(abs(a - b) < 5e-4 for a, b in zip(new, given)):
        return None, lines + [f"{FR_D_CEILING_MARKER}: Decke = gegeben, nichts zu uebernehmen"]
    if str(env_d.get(TORCH_CACHE_CAP_ENV, "0")).strip() != "1":
        return None, lines + [f"{FR_D_CEILING_MARKER}: Decke NICHT uebernommen: P0 aus "
                              f"({TORCH_CACHE_CAP_ENV}!=1 in --env-d) -- ohne gedeckelten "
                              f"Allokator-Cache nimmt der Cache-Swing den Platz an der naechsten "
                              f"Peak zurueck; gegeben {given} bleibt"]
    if map_built:
        return None, lines + [f"{FR_D_CEILING_MARKER}: Decke NICHT uebernommen: die "
                              f"Platztausch-Karte ist schon aus FR_D {given} gebaut -- "
                              f"eine hoehere FR_D danach waere eine Form, die die Karte "
                              f"nicht beschreibt"]
    return new, lines

def d_token_cut_vector(plan_cut, stated) -> Optional[Tuple[int, ...]]:
    """#239 S3a: the integer token vector D runs, or None (no cut).

    ``plan_cut`` is the planner's solved cut (``maxmin``/``joint``, already in
    /``KV_TOKEN_SHARE_GRID``); a stated ratio vector (``stated``) goes onto the
    same grid by largest remainder, zeros kept -- a 0 is a layout here.
    """
    from flliper.srt.distributed.utils import partition_units
    from flliper.srt.planner import expert_residency as _er

    if plan_cut:
        return tuple(int(x) for x in plan_cut)
    if not isinstance(stated, tuple) or not stated:
        return None
    return tuple(
        partition_units(
            _er.KV_TOKEN_SHARE_GRID,
            [int(round(float(x) * 10**6)) for x in stated],
            allow_zero=True,
        )
    )

def apply_bar1_windows(ns, cards: Sequence["Card"], log=None) -> Optional[str]:
    """HW-P1c 1003 (BAR1-WINDOW): the group windows for THIS card count and the
    MEASURED BAR1 of the cards (pdflip/bar1_windows.py). Only the untouched
    defaults (and the dual default) are replaced; an operator's explicit
    window is kept. N = 3 on 256 MiB BARs: the shipped windows, no line."""
    n = len(cards)
    if n in _topo.PROVEN_CARD_COUNTS:
        return None   # the proven layout is not re-judged: the shipped windows stand byte for byte
    dual = bool(getattr(ns, "dual_layout", False) or getattr(ns, "dual_share", False))
    p_cur = str(getattr(ns, "p_barlink_bar1_window_mib", P_BARLINK_BAR1_WINDOW_MIB))
    d_cur = str(getattr(ns, "d_barlink_bar1_window_mib", D_BARLINK_BAR1_WINDOW_MIB))
    p_default = p_cur in (P_BARLINK_BAR1_WINDOW_MIB, DUAL_P_BARLINK_BAR1_WINDOW_MIB)
    d_default = d_cur == D_BARLINK_BAR1_WINDOW_MIB
    pl = bar1_windows_mod.plan(
        n, [getattr(c, "bar1_total_mib", None) for c in cards], dual=dual,
        p_base=p_cur if not p_default else (DUAL_P_BARLINK_BAR1_WINDOW_MIB if dual else None),
        d_base=d_cur if not d_default else None)
    if not pl.ok:
        raise PdFlipLaunchRefused(f"BAR1-WINDOW: {pl.why}")
    changed = (pl.p != p_cur and p_default) or (pl.d != d_cur and d_default)
    if p_default:
        ns.p_barlink_bar1_window_mib = pl.p
    if d_default:
        ns.d_barlink_bar1_window_mib = pl.d
    if changed or pl.usable_mib is None and n != _topo.PROVEN_CARD_COUNTS[0]:
        line = pl.line()
        if log is not None:
            log(line)
        return line
    return None

def configure_xchg_region(n_cards: int, log=None) -> None:
    """HW-P1c 1003 (XCHG-REGION): lay the host exchange region out for ``n_cards``
    in THIS process (the ranks learn N from the env var ``prepare_region``
    publishes). N = 3: no-op."""
    if int(n_cards) == weight_exchange_region.DEFAULT_N_CARDS:
        weight_exchange_region.configure(weight_exchange_region.DEFAULT_N_CARDS)
        return
    prev = weight_exchange_region.configure(int(n_cards))
    if log is not None:
        log(f"PDFLIP-XCHG-REGION geometry for {int(n_cards)} cards (was {prev}): "
            f"{weight_exchange_region.N_PAIRS} directed cross pairs, "
            f"{weight_exchange_region.REGION_BYTES / (1 << 20):.0f} MiB, data offset "
            f"{weight_exchange_region.DATA_OFF // (1 << 20)} MiB; ranks read "
            f"{weight_exchange_region.ENV_N_CARDS}")

def _derive_csv(policy: str, text: str, calibrated: Sequence[str], live: Sequence[str]) -> Optional[str]:
    """A 'a,b,c' vector under a derivation policy, as 'a,b' text; None when not derivable."""
    from flliper.srt.pdflip import inventory_view as _iv

    got = _iv.derive_vector(policy, str(text), tuple(calibrated), tuple(live))
    return None if got is None else (got if isinstance(got, str) else ",".join(str(x) for x in got))

def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def d_only_bind_argv(front_host: str) -> List[str]:
    """#239 M1s (rc12z30g D-only, 28.09.): with --d-only there is NO front, so
    group D on :30032 IS the client surface. common_flags binds every group to
    127.0.0.1 (the groups sit behind the front), and inside the container the
    Docker port forward cannot reach a loopback-only socket -- host_acceptance
    serve_d waited forever and the probes had to go in by nsenter. D-only binds
    where the front would have: --front-host (default 0.0.0.0, user order
    2026-09-10). Appended after common_flags, so it is the last --host argparse
    reads; a flip boot never calls this."""
    return ["--host", str(front_host)]

def admin_key_flag(admin_api_key: Optional[str]) -> List[str]:
    """#1275: BOTH groups get the key, or the capability is half-present.

    The front talks to P and D with the same header, and the resize lever is
    wanted on whichever group holds the store -- so a key on one group only
    would make the front's flip RPCs succeed against one peer and 401 against
    the other, which is a worse state than having no key at all. One helper,
    two call sites, so the two argv builders cannot drift apart.

    Empty list when unkeyed, which keeps every pre-#1275 argv byte-identical.
    """
    # #1361 [22-fix4] ONE TOKEN, `--flag=value`. The two-token form hands the
    # value to argparse as a separate word, and a value that starts with '-'
    # is then read as an option: boot weg2xsn25's first launch (065608) died
    # on `argument --admin-api-key: expected one argument` because that mint
    # began with a hyphen (1 in 64). `mint()` no longer produces such a key,
    # and this form makes the argv safe for ANY key, including one an operator
    # supplies by hand, which never passes through `mint()` at all.
    # The P form key does not move: `ring_table._flag_pairs` normalises
    # `--flag value` to `--flag=value` before hashing, and the flag is in
    # FORM_KEY_EXCLUDED_FLAGS either way (verified, both spellings).
    return [f"--admin-api-key={admin_api_key}"] if admin_api_key else []

def _d_adopt_armed(ns) -> bool:
    """#108: EINE Aufloesung des ``--pdflip-d-adopt``-Flags im Launcher.

    Beide ``argv_d``-Bauer und der Env-Schreiber lesen DIESE Funktion. Zwei
    Ableitungen derselben Entscheidung sind der Weg, auf dem zwei Antworten
    entstehen (#91/#106/#107 -- dreimal dieselbe Klasse an einem Tag).
    """
    return str(getattr(ns, "pdflip_d_adopt", "off")).strip().lower() == "on"

def _adopt_load_format_flag(armed: bool) -> List[str]:
    """#108: unter Adoption laedt D mit ``dummy``, nicht von Platte.

    ``armed`` KOMMT VOM AUFRUFER, ES WIRD KEINE ENV GELESEN -- und das ist
    der ganze Fix aus fnFL2w52. Die erste Fassung rief
    ``adopt.adopt_armed()`` ohne ``explicit``; im LAUNCHER-Prozess ist
    ``FLLIPER_PDFLIP_D_ADOPT`` aber nicht gesetzt (sie geht nur an die
    Kinder), also gab die Funktion IMMER ``[]`` zurueck -- waehrend der
    Launcher danebenschrieb "D startet mit --load-format dummy". Im argv
    des Boots stand der Flag null mal, D las 288 Presplit-Layer von
    Platte, und die Zeile war eine BEHAUPTUNG. Der Docstring von
    ``adopt_armed`` benennt genau diese Falle ("ein Launcher-Aufrufer ohne
    explicit liest still seine eigene ungesetzte Umgebung"); ein
    Parameter kann man nicht vergessen zu setzen, eine Env schon.

    ``DummyModelLoader`` ruft ``initialize_dummy_weights`` UND danach
    ``process_weights_after_loading`` je Modul -- Repack, Presplit und
    Pool-Geometrie entstehen also identisch zum Plattenweg. Nur die Zahlen
    sind Zufall, und genau die bringt der Erstflip.

    Der Flag geht ans ENDE der Launcher-Liste, VOR ``extra``: so kann ein
    Arm ihn mit seinem eigenen ``--load-format`` ueberstimmen.
    """
    if not armed:
        return []
    return ["--load-format", "dummy"]

def _credit_counter_rows(credit_dir: str = "") -> List[Tuple[str, str, str, int]]:
    """``(path, name, boot half of the epoch, publisher pid)`` per counter file.

    The boot half is everything before the first ``.`` of the stored epoch token
    (:func:`pdflip_memory_saver.credit_epoch` composes ``<boot>.<flip>``).  A
    LEGACY file whose stamp is a bare integer has no boot half and gets ``""``,
    which matches no boot -- the conservative direction, since it belongs to a
    boot that predates the token.
    """
    from flliper.srt.managers import pdflip_memory_saver

    directory = credit_dir or os.environ.get(
        pdflip_memory_saver.VRAM_CREDIT_DIR_ENV,
        os.environ.get(pdflip_memory_saver.PCIE_LOCK_DIR_ENV,
                       pdflip_memory_saver.DEFAULT_PCIE_LOCK_DIR),
    )
    prefix = "." + pdflip_memory_saver.VRAM_CREDIT_PREFIX
    # rename transition (compat_shims): counters of boots under the other name are this line's too
    prefixes = (prefix,) + _name_counterparts((prefix,))
    rows: List[Tuple[str, str, str, int]] = []
    if not os.path.isdir(directory):
        return rows
    for name in sorted(os.listdir(directory)):
        if not name.startswith(prefixes):
            continue
        path = os.path.join(directory, name)
        epoch, pid = "", 0
        try:
            with open(path, errors="replace") as fh:
                state = json.loads(fh.read() or "{}")
            if isinstance(state, dict):
                epoch = str(state.get("epoch", ""))
                pid = int(state.get("publisher_pid", 0) or 0)
        except (OSError, ValueError, TypeError):
            # A torn or foreign file names no boot and no holder.  It is swept
            # as dead residue at launch, never removed as "this boot's".
            epoch, pid = "", 0
        rows.append((path, name, epoch.split(".")[0] if "." in epoch else "", pid))
    return rows

def _argv_scalar(extra: str, flag: str):
    """Der LETZTE Wert hinter ``flag`` in einer ``--extra-*``-Zeichenkette.

    Der letzte, nicht der erste: argparse nimmt bei doppelter Flagge den
    letzten, und genau diese Doppelung hat am 22.09. zehn Flags still
    umgedreht (Memory ``argv-doppelung-arm-schlaegt-planer``). Wer hier den
    ersten naehme, laese einen Wert, mit dem der Boot nicht startet.

    ``None`` wenn die Flagge fehlt oder ohne Wert dasteht.
    """
    if not extra or not flag:
        return None
    try:
        parts = shlex.split(str(extra))
    except ValueError:
        return None
    hit = None
    for i, tok in enumerate(parts):
        if tok == flag and i + 1 < len(parts):
            hit = parts[i + 1]
        elif tok.startswith(flag + "="):
            hit = tok[len(flag) + 1:]
    return hit

def carrier_covers_d_session(carrier_max: int, d_session: int) -> Tuple[bool, str]:
    """LAW 02.10. (second half): the carrier (D's host staging read, the
    window loop's transit) must hold D's maximum session too -- below it the
    front's W52 would refuse a prompt D can hold. Returns ``(ok, line)``."""
    c, d = int(carrier_max), int(d_session)
    line = f"{P_COVERS_D_SESSION_MARK} carrier_max={c} D session={d}"
    if c > 0 and c < d:
        return False, (line + " REFUSED: the carrier is below D's session -- raise D's L2 "
                       "(it is derived from --max-kv-per-request) or lower --max-kv-per-request")
    return True, line + " ok"

def _argv_vector(extra: str, flag: str):
    """Der Komma-Vektor hinter ``flag`` in einer ``--extra-*``-Zeichenkette.

    #106. Gibt eine Liste von STRINGS zurueck, nicht von Zahlen: sie wird
    woertlich in `FLLIPER_MOE_EXPERT_STORE_GEOMETRY` weitergereicht, und der
    Leser (`expert_store.shared_geometry`) parst selbst und verwirft
    Unlesbares. Hier zu casten hiesse, denselben Zweifel zweimal zu
    beantworten -- und die zweite Antwort koennte von der ersten abweichen.

    ``None`` wenn die Flagge fehlt oder ohne Wert dasteht; eine leere Liste
    kann nie entstehen, weil sie vom Aufrufer nicht von "fehlt" zu
    unterscheiden waere.
    """
    if not extra or not flag:
        return None
    try:
        parts = shlex.split(str(extra))
    except ValueError:
        return None
    for i, tok in enumerate(parts):
        raw = None
        if tok == flag and i + 1 < len(parts):
            raw = parts[i + 1]
        elif tok.startswith(flag + "="):
            raw = tok[len(flag) + 1:]
        if raw is None:
            continue
        vec = [x.strip() for x in raw.split(",") if x.strip()]
        return vec or None
    return None

def effective_reserve_by_card(
    cards: Sequence["Card"],
    user_reserve_by_card: Optional[Dict[str, int]],
    booked_rest_mib: Optional[Sequence[Optional[int]]] = None,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """The reserve the launcher ACTUALLY BOOKS per card for ONE group, and what it masked.

    ``(effective, masked)``, both ``{card_uuid: MiB}``. On a card whose awake rest is a
    BOOKED record (``booked_rest_mib[i] is not None``) the launcher charges no user
    reserve: the measured rest is what consumed it ("reserve 0 BY DESIGN ... would book a
    forbidden corridor reserve", ``booked_rest_kwargs``; keine-korridor-reserve-nie), so the
    effective value is 0 and the stated value goes to ``masked``. On every other card the
    effective value is the operator's. ONE function for ``budgets_from_dc`` (its
    ``_reserve_eff``) and for the rank env (#236 review MUSS-1), so the launcher's budget
    and a rank's floor cannot price different reserves for the same card and group.
    """
    booked = [None] * len(cards) if booked_rest_mib is None else list(booked_rest_mib)
    if len(booked) != len(cards):
        raise PdFlipLaunchRefused(
            f"#236: {len(booked)} booked rests for {len(cards)} cards -- a partial "
            "vector never prices a card it never saw")
    effective: Dict[str, int] = {}
    masked: Dict[str, int] = {}
    for i, c in enumerate(cards):
        stated = int((user_reserve_by_card or {}).get(c.uuid, 0))
        if booked[i] is not None:
            effective[c.uuid] = 0
            if stated > 0:
                masked[c.uuid] = stated
        else:
            effective[c.uuid] = stated
    return effective, masked

def check_drafter_identity(log_p: str, log_d: str) -> Dict[str, object]:
    """W10 (#1233): the drafter identity and canonical draft layout of both
    groups, read from their logs. ``match`` is False on ANY disagreement or
    absence -- a group that registered no drafter at all is the pre-fix
    shape, not a pass."""
    out: Dict[str, object] = {}
    for name, path in (("P", log_p), ("D", log_d)):
        ids, layouts, n = set(), set(), 0
        try:
            with open(path, errors="replace") as f:
                for line in f:
                    m = _DRAFTER_RE.search(line)
                    if m:
                        ids.add(m.group(1))
                        n += 1
                    m = _LAYOUT_RE.search(line)
                    if m:
                        layouts.add(f"v{m.group(1)}")
        except OSError:
            pass
        out[name] = ",".join(sorted(ids)) if ids else None
        out[f"layout_{name}"] = ",".join(sorted(layouts)) if layouts else None
        out[f"n_{name}"] = n
    out["match"] = bool(
        out["P"] and out["D"] and out["P"] == out["D"] and "," not in str(out["P"])
        and out["layout_P"] and out["layout_P"] == out["layout_D"]
    )
    return out

def boot_state_dir() -> Optional[str]:
    """IPC §2.2: the boot's state/<boot_id>/ directory (PDFLIP_STATE_DIR, mounted by
    the arm), or None -- then the launcher writes no state and the rank records
    stay next to the group logs."""
    return envs.PDFLIP_STATE_DIR.get() or None

def count_marker(path: str, marker: str) -> int:
    n = 0
    try:
        with open(path, errors="replace") as f:
            for line in f:
                if marker in line:
                    n += 1
    except FileNotFoundError:
        pass
    return n

def census_entry_for_pricing(record_path: str, key: str, *, model: str,
                             evidence_dir: str) -> Tuple[Optional[Dict[str, object]], str]:
    """29.09. NF1d: the host census entry the ledger prices from, ``(entry, line)``.

    A record that predates ``shm_total_max_gib`` (the peak of the SUM at one
    instant) gets it rebuilt from this key's own memts
    (host_census.backfill_shm_total) -- only for a checkpoint that is NOT the
    reference: the 27B record stays byte for byte, and no model named (a desk
    caller) touches nothing."""
    from flliper.srt.pdflip import host_census as _hc

    if model and pdflip_form.reference_model_verdict(model, host_ledger.REFERENCE_MODEL)[0] is False:
        entry, line = _hc.backfill_shm_total(record_path, key, evidence_dir)
        return (dict(entry) if entry is not None else None), line
    return _hc.load_record(record_path).get(key), ""

def budget_rest_from_records(profile: Optional[str] = None) -> bool:
    """VRAM-GRUNDGESETZ 29.09.: the profile row's ``budget_rest_from_records``
    (qwen27b on, nextflash off) unless ``FLLIPER_PDFLIP_BUDGET_REST_RECORD`` is set."""
    row = pdflip_form.profile_row(profile if profile is not None else pdflip_form.DEFAULT_PROFILE)
    return _budget_rest.switch_on(bool(getattr(row, "budget_rest_from_records", False)))

def d_card_vote_lines(cards: Sequence[Card], group: str = "D", fixed: Optional[Sequence[float]] = None):
    """THE LINE OF THE CARD, ONE READER for the launcher (PDFLIP-VOTE-LINE-BOOK and, #201 TRIM-LAW, the flip arm's
    trim threshold): per D ordinal ``i`` (``cards[i]`` = the NVML-identified card of D rank ``i``, its ``uuid`` from
    ``card_identity``, never torch order) the corridor floor (``corridor_budget.floors_for_cards``, reserve 0 -- the
    rank's own read ``corridor_floor_for_current_device`` gets no user reserve either, and a reserve in a line is
    forbidden: keine-korridor-reserve-nie), the vote line ``extend_trim.line_for_floor_mib(floor, fixed[i] if
    measured)`` -- the formula :func:`extend_trim.corridor_line_mib` evaluates on the rank -- and whether that line is
    USABLE: the floor is MEASURED and its footprint store is of THIS form (review S3,
    ``budget_rest.footprint_form_mismatch``: the launcher reads the LAST published D digest). Returns
    ``(floors by uuid, lines, usable, why-not-of-this-form)``."""
    from flliper.srt.pdflip.extend_trim import line_for_floor_mib

    n = len(cards)
    # reserve 0 BY DESIGN: the booked path charges no user reserve, and a reserve here
    # would book a forbidden corridor reserve as "line" (keine-korridor-reserve-nie)
    floors = corridor_budget.floors_for_cards(cards, group, 0)
    act = [bool(floors[c.uuid].measured) for c in cards]
    lines = [line_for_floor_mib(floors[c.uuid].mib, fixed[i] if (fixed is not None and act[i]) else None)
             for i, c in enumerate(cards)]
    form_why: List[Optional[str]] = [None] * n
    uuids = [c.uuid for c in cards]
    _stores: Dict[str, object] = {}
    for i, c in enumerate(cards):
        if not act[i]:
            continue
        path = str(floors[c.uuid].provenance or "").split("#", 1)[0].strip()
        if path not in _stores:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    _stores[path] = json.load(fh) or {}
            except Exception as exc:  # noqa: BLE001 -- an unreadable store books nothing
                _stores[path] = f"footprint store {path or '-'} unreadable ({type(exc).__name__})"
        st = _stores[path]
        why = st if isinstance(st, str) else _budget_rest.footprint_form_mismatch(st, uuids)
        if why is not None:
            act[i] = False
            form_why[i] = why
    return floors, lines, act, form_why

def apply_profile_vision_default(ns, argv_words: Sequence[str]) -> Optional[str]:
    """SCHALTER-HALBPORT 1002: an UNSET ``--pdflip-vision`` takes the registry
    row's ``vision`` where the row says so (``vision_arg_default``: nextflash
    ``transient``, as every NF boot ran it, nf-int4.env:229). qwen27b states
    nothing here (its profiles give the flag), so its parser default ``off``
    stands. Applied BEFORE the form is resolved (the vision axis reads the
    flag). A given flag always wins. Returns the line naming the default, or
    None."""
    if getattr(ns, "teardown", False) or pdflip_form.flag_given(argv_words, "--pdflip-vision"):
        return None
    if str(getattr(ns, "pdflip_vision", VISION_OFF) or VISION_OFF) != VISION_OFF:
        return None
    row = pdflip_form.profile_row(getattr(ns, "profile", None))
    if row is None or not getattr(row, "vision_arg_default", False):
        return None
    want = str(row.vision or VISION_OFF)
    if want == VISION_OFF or want not in VISION_CHOICES:
        return None
    ns.pdflip_vision = want
    return (f"{VISION_DEFAULT_MARKER} registry {row.id}: --pdflip-vision {want} "
            "(the row's vision; a given --pdflip-vision wins)")

def d_owned_miss_ms_rank(ns, *, env_d: Mapping[str, str], n: int
                         ) -> Tuple[Optional[Tuple[float, ...]], str]:
    """01.10. (owned cut from the profiles): the cost per missed expert row
    of EVERY D rank from its own paired records (each card at its own link:
    NF D TP1 = NVML0 3080 x4, TP2 = NVML2 3080 x8 -- the (host, worker) pair
    of :func:`d_owned_miss_ms` pools both 3080s into one worker cost).
    ``(None, "")`` without a full window on every rank."""
    from flliper.srt.layers.moe import pool_miss_cost as _miss_cost
    from flliper.srt.planner import expert_residency as _er

    model = str(getattr(ns, "model", "") or "")
    root = str(env_d.get("FLLIPER_PDFLIP_OWNED_MISS_RECORD", "") or "").strip()
    if not root:
        return None, ""
    got = _er.owned_miss_per_rank_from_records(
        _er.read_owned_miss_rank_records(_miss_cost.record_dirs_for(root, model)),
        n=int(n), model=model)
    if got is None:
        return None, ""
    return got

def d_owned_extend_h2d(cards: Sequence["Card"]) -> Tuple[Optional[List[float]], str]:
    """S54 (19.10.): the measured pinned H2D GB/s of every D rank's card, in
    rank order, from the card probe (``rigmon.card_probe``, the artifact the
    cold tier and the planner's cost model already price links from -- read
    by UUID, never by enumeration order). ``(None, why)`` when a card has no
    measured rate: the owned solve's extend term then stays off, named."""
    try:
        from flliper.srt.layers.moe.expert_offload import _card_probe_h2d_table

        table = _card_probe_h2d_table()
    except Exception as exc:  # noqa: BLE001 - an unreadable probe is an absence
        return None, f"Kartenprobe nicht lesbar ({type(exc).__name__})"
    out, parts = [], []
    for c in cards:
        g = table.get(str(getattr(c, "uuid", "")))
        if g is None:
            return None, f"nvml{getattr(c, 'nvml_index', '?')} ohne gemessenen H2D in der Kartenprobe"
        out.append(float(g))
        parts.append(f"nvml{getattr(c, 'nvml_index', '?')} {g:.2f}")
    return out, "Kartenprobe H2D GB/s " + " / ".join(parts)

def d_owned_heat_records(ns, *, env_d: Mapping[str, str]) -> Tuple[List[dict], str]:
    """01.10. (owned cut from the profiles): the #276 heat records of group
    D's ``FLLIPER_DEBUG_MOE_HEAT`` directory -- the owned solve's non-resident
    demand (``plan_d_residency`` maps them to a routed-lane share per global
    expert). The directory is the line's per-checkpoint heat directory (the
    profile names it); ``([], "")`` when it is unset."""
    import glob
    import json as _json

    root = str(env_d.get("FLLIPER_DEBUG_MOE_HEAT", "") or "").strip()
    if not root or not os.path.isdir(root):
        return [], ""
    recs = []
    for path in sorted(glob.glob(os.path.join(root, "moe_heat_*.json"))):
        try:
            with open(path) as fh:
                recs.append(_json.load(fh))
        except (OSError, ValueError):
            continue
    return recs, root

def d_card_ledger(terms: Sequence[Dict[str, object]], budgets_d: Sequence[int], label: str):
    """rc12c: the #145 card ledger from the SAME terms ``budgets_from_dc``
    charged, plus what its measured corridor pass took off on top."""
    import msgspec

    from flliper.srt.planner import expert_residency as _er

    led = _er.DCardLedger(
        source=f"budgets_from_dc pass {label}",
        total_mib=tuple(float(t["total"]) for t in terms),
        carve_mib=tuple(float(t["carve"]) for t in terms),
        dormant_mib=tuple(float(t["dormant"]) for t in terms),
        floor_mib=tuple(float(t["floor"]) for t in terms),
        awake_mib=tuple(float(t["awake"]) for t in terms),
        awake_source=tuple(str(t["awake_source"]) for t in terms),
    )
    extra = tuple(max(0.0, led.card_budget_mib(i) - float(b)) for i, b in enumerate(budgets_d))
    return msgspec.structs.replace(led, corridor_pass_mib=extra)

def d_scratch_cap(fits, scratch: Sequence[int]) -> Tuple[List[int], List[str]]:
    """rc12c: THE CONSUMER FOLLOWS THE BOOKED BUDGET. Where the #145 balance of
    a rank misses (262k KV no longer fits beside its expert buffer), its
    FLLIPER_MOE_SCRATCH_SLOTS is lowered to the planner's edge at the given
    fraction -- never raised, never below the two rows the runtime needs (a
    rank without an edge keeps its value and the refusal names it)."""
    from flliper.srt.planner import expert_residency as _er

    out = [int(s) for s in scratch]
    lines: List[str] = []
    for f in fits:
        if f.verdict == "PASST":
            continue
        rows, s_edge = _er.scratch_edge(
            local_experts=f.local_experts, fraction=f.fraction, max_rows=f.ceiling_max_rows)
        if s_edge is None or s_edge >= out[f.rank]:
            continue
        lines.append(
            f"SCRATCH-DECKEL rang{f.rank}: FLLIPER_MOE_SCRATCH_SLOTS {out[f.rank]} -> {s_edge} "
            f"(Planer-Kante bei f {f.fraction:.3f}: <= {rows} Zeilen x {f.n_layers * f.slot_mib:.1f} "
            f"MiB; Budget {f.budget_mib:.0f} MiB gebucht, Verdikt vorher {f.verdict}) -- der "
            f"Experten-Puffer folgt dem gebuchten Budget, nicht dem freien Speicher")
        out[f.rank] = int(s_edge)
    return out, lines

def d_extend_trim_env(ledger, fits, growth_mib: Optional[Sequence[Optional[float]]] = None) -> str:
    """rc12e: ``FLLIPER_PDFLIP_EXTEND_TRIM_MIB`` from the card ledger's floor and
    the solved form's booked activation, per rank; '' when a rank is missing.
    rc12f: with ``growth_mib`` (``D_EXTEND_GROWTH_MIB``) the extend's measured
    reserved growth replaces the activation where it is larger."""
    from flliper.srt.pdflip import extend_trim as _et

    act = {int(f.rank): float(f.activation_mib) for f in fits}
    n = len(ledger.floor_mib)
    if any(r not in act for r in range(n)):
        return ""
    if growth_mib is not None and len(growth_mib) != n:
        growth_mib = None
    return _et.launcher_thresholds(ledger.floor_mib, [act[r] for r in range(n)], growth_mib)

def d_lend_floor_env(ledger) -> str:
    """D-TRANSIENT-LEND: ``FLLIPER_PDFLIP_D_LEND_FLOOR_MIB`` = the card ledger's
    near-OOM floor per rank (MiB, rounded up)."""
    return ",".join(str(int(math.ceil(float(f)))) for f in ledger.floor_mib)

def budget_charges_driver_carve(profile: Optional[str] = None) -> bool:
    """The profile row's ``budget_charges_driver_carve`` (rc12b OOM); False
    for an unknown profile, so nothing changes where no row says so."""
    row = pdflip_form.profile_row(profile if profile is not None else pdflip_form.DEFAULT_PROFILE)
    return bool(getattr(row, "budget_charges_driver_carve", False))

def driver_carve_charged(card, min_total_mib: int) -> bool:
    """Is the driver carve booked on ``card`` (with the profile's carve switch on)?

    The ``min_total_mib`` scope (27B b1 death) is a MEASURED decision about
    the reference rig's classes: the 27B corridor pass was SATISFIED on the
    20 GB RTX 3080s with their carve already in predicted_free, and bound on
    the 5090. HW-GENERIC 1002: it therefore holds for those calibration
    classes only; a card of any other class books its NVML-reported carve
    (a measured, unallocatable term -- not a margin), because no measurement
    says its corridor already absorbs it."""
    if card_identity.calibration_class(card) is None:
        return True
    return int(card.total_mib) >= int(min_total_mib or 0)

def driver_carve_min_total_mib(profile: Optional[str] = None) -> int:
    """The profile row's ``driver_carve_min_total_mib`` (27B b1 death): the
    carve is charged only on cards with at least this NVML total; 0 (and an
    unknown profile) = every card."""
    row = pdflip_form.profile_row(profile if profile is not None else pdflip_form.DEFAULT_PROFILE)
    return int(getattr(row, "driver_carve_min_total_mib", 0) or 0)

def d_row_attn_units(pcm, weights: Sequence[int]) -> List[int]:
    """The attention UNIT vector a D operating-point row prices ``weights``
    with -- the runtime widens the grid to the q-heads when the o-group count
    is below the world size (uneven_perf.py:4520-4528). ONE implementation for
    the row and for the calibration that must reproduce the row exactly."""
    from flliper.srt.distributed.utils import partition_units

    n_ranks = len(weights)
    a_units = int(pcm.attn_units)
    grid = a_units if a_units >= n_ranks else max(int(pcm.q_heads), n_ranks)
    return list(partition_units(grid, list(weights)))

def _d_argv_of_front_log(front_log: str) -> List[str]:
    try:
        with open(front_log, errors="replace") as f:
            for line in f:
                if "group D argv:" in line:
                    return shlex.split(line.split("group D argv:", 1)[1])
    except OSError:
        pass
    return []

def _attn_axis_for(weights: Sequence[int], plan_flags) -> str:
    """Which axis the attention family's COMPUTE rides for one D vector (#1293).

    ``"token"`` exactly when the boot form is replicated-KV uneven DCP -- the
    serving form: every rank keeps the FULL replicated kv-head set and runs the
    attention core over its own TOKEN shard, so the compute follows the token
    vector, which is continuous (#492: "attention grid-PINNED was FALSE").
    ``"head"`` otherwise (even DCP, no DCP, or a uniform plan), where the
    kv-head unit grid genuinely carries the family.

    THE PREDICATE IS #503's SHARED ONE, NOT A THIRD SPELLING.
    ``plan_uneven_dcp_kv_replicated`` (distributed/utils.py, placed next to the
    runtime's ``uneven_dcp_kv_replicated`` by #503 so the two cannot drift)
    answers this exact question at plan time from the flags about to be
    emitted. A ``plan_flags`` with no DCP evidence (``dcp_size`` unset and no
    KV token vector) answers "head" for ANY weights -- a non-DCP form must
    never silently take the token axis, because there the token vector does
    not exist and the head grid is the real carrier.
    """
    from flliper.srt.distributed.utils import plan_uneven_dcp_kv_replicated

    return (
        "token"
        if plan_uneven_dcp_kv_replicated(plan_flags, list(weights))
        else "head"
    )

def _axis_floor_ranks(units: int, weights: Sequence[int]) -> Tuple[int, ...]:
    """Ranks whose proportional share of ``units`` is BELOW one whole unit.

    ``partition_units`` guarantees every rank >= 1 unit, so each of these is
    pinned at the partitioner's floor and the realised split stops carrying the
    weight ratio on that axis. This is the one REACHABLE failure of a unit grid
    -- a rank owning ZERO units cannot happen -- and naming the ranks rather
    than answering yes/no is what lets the caller re-normalise against the grid
    that binds and say WHICH rank moved.
    """
    total = sum(int(w) for w in weights)
    if total <= 0 or int(units) <= 0:
        return tuple()
    return tuple(
        r
        for r, w in enumerate(weights)
        if float(int(w)) / float(total) * float(int(units)) < 1.0
    )

def d_operating_point_line(
    rows: Sequence[DOperatingPointRow], refusals: Sequence[str], shipped: str
) -> str:
    """All three vectors, their price and their pool, on ONE line.

    The trade is only readable if the alternatives are printed next to what
    shipped, in the same units, from the same model, on the same boot.
    """
    if not rows:
        return (
            "PDFLIP D-OPERATING-POINTS UNPRICED (shipped=%s): %s"
            % (shipped, " | ".join(refusals) or "no reason given")
        )
    parts = []
    for row in rows:
        priced = (
            "%.4f %s" % (row.round_ms, row.round_unit)
            if row.round_ms is not None
            else row.round_unit
        )
        # #1293: which axis the attention compute rides, named per row, plus
        # the token partition when that axis carries it and the floor-binding
        # statement when any grid binds. `attn_axis=head` is the pre-DCP
        # reading of the attn heads printed before it; `attn_axis=token`
        # says the heads are the PROJECTION split and the compute follows
        # `token_units`.
        axis_bit = " attn_axis=%s" % row.attn_axis
        if row.attn_token_units:
            axis_bit += " token_units %s" % (list(row.attn_token_units),)
        if row.axis_note:
            axis_bit += " [%s]" % row.axis_note
        parts.append(
            "%s%s: weights %s from %s %s -> attn %s%s GDN %s, round %s, world pool "
            "%s, funded ctx %s, feasible %s"
            % (
                row.position,
                " [SHIPPED]" if row.position == shipped else "",
                list(row.weights),
                row.score_name,
                [round(x, 1) for x in row.scores],
                list(row.attn_heads),
                axis_bit,
                list(row.gdn_heads),
                priced,
                row.world_pool_tokens
                if row.world_pool_tokens is not None
                else "UNPRICED",
                # BOTH, because they answer different questions: the world
                # pool is the sum over ranks, the funded context is bounded by
                # the SMALLEST rank (min(sum P, 64 min P)) and is the quantity
                # a carrier bound is about. A vector that collapses one rank
                # moves the second far more than the first.
                row.funded_ctx_tokens
                if row.funded_ctx_tokens is not None
                else "UNPRICED",
                "UNPRICED" if row.feasible is None else row.feasible,
            )
        )
    tail = (" REFUSED: " + " | ".join(refusals)) if refusals else ""
    note = next((r.note for r in rows if r.note), "")
    return (
        "PDFLIP D-OPERATING-POINTS (#1241 slice 2, the operating-point axis of "
        "the #1017 solve; shipped=%s, default %s since the user order of "
        "2026-09-09 -- 'maxkv' is still selectable and its argv is still "
        "byte-identical to the pre-#1241 form). %s. %s%s"
        % (shipped, D_TP_OBJECTIVE_DEFAULT, " || ".join(parts), note, tail)
    )

def d_token_vector_decision(
    vector: Optional[str],
    role: str,
    provenance: Optional[str],
) -> DTokenVectorDecision:
    """Group D's token-vector argv, and the refusal that keeps #602 out.

    Default (``vector`` unset): NOTHING is shipped. The line says which branch
    the boot is on and where the vector will come from instead.

    Operator-supplied: the value is checked against the runtime's OWN
    retraction register (``planner.retracted.find_retracted_token_vector``, the
    same authority ``_refuse_retracted_token_vector`` uses) BEFORE the window
    is spent. The runtime refuses a retracted PIN and only warns about a
    retracted SEED, because at that point a seed is still going to be
    superseded; here neither is accepted, because a launcher that types a
    retracted number has already made the mistake the register exists to
    catch, and the desk is the cheap place to say so.
    """
    role = str(role or "pin")
    if not vector:
        return DTokenVectorDecision(
            flags=(),
            line=(
                "PDFLIP D-TOKEN-VECTOR: none shipped (default). SOURCE = "
                "estimate(budget): group D derives its own starting vector "
                "from THIS boot's --rank-gpu-memory-mib "
                "(distributed/utils.py resolve_cp_token_ratios, the "
                "partition_units(64, budget - weights - 1536 MiB) rung). "
                "OBJECTIVE = the profiled per-rank capacity optimum: the "
                "runtime supersedes that estimate IN-PROCESS after profiling "
                "and prints the pool it moved -- 'installed measured KV-token "
                "ownership vector [..] (pre-boot estimate was [..]), "
                "max_total_num_tokens X -> ~Y'. That install is armed by the "
                "ROLE, and #1270 is why this sentence can be believed again: "
                "an undeclared vector now reads role='estimate', not 'pin'. "
                "Boot weg2sb1 is the counter-example -- with the #1032 seed "
                "gone nothing declared a vector, the estimate [30,17,17] "
                "inherited 'pin', the install was suppressed and D served "
                "574,336 tokens against rg6's 681,856 (-15.8 %%) while the "
                "measured [17,7,8] was printed as an unusable restart hint. "
                "The retracted #602 seed %s this launcher used to ship stays "
                "GONE (#1032): the seed was never the right way to arm the "
                "install, it was only the way that happened to work, and it "
                "carried a foreign lineage to hold a boot to."
                % (",".join(str(v) for v in RETRACTED_SEED_VECTOR_1032),)
            ),
        )

    try:
        parsed = [int(x) for x in str(vector).split(",") if x.strip() != ""]
    except ValueError:
        raise PdFlipLaunchRefused(
            "W46 PdFlipTokenVectorRefused: --d-uneven-token-vector %r is not a "
            "comma-separated integer vector." % (vector,)
        )
    if not parsed or any(v <= 0 for v in parsed):
        raise PdFlipLaunchRefused(
            "W46 PdFlipTokenVectorRefused: --d-uneven-token-vector %r must be "
            "positive integers, one per DCP rank." % (vector,)
        )

    from flliper.srt.planner.retracted import (
        find_retracted_token_vector,
        token_vector_refusal_text,
    )

    entry = find_retracted_token_vector(parsed, provenance)
    if entry is not None:
        raise PdFlipLaunchRefused(
            "W46 PdFlipTokenVectorRefused: "
            + token_vector_refusal_text(
                entry,
                parsed,
                "the Weg-2 launcher refuses to ship it in EITHER role. The "
                "runtime warns about a retracted role='seed' and refuses only "
                "a 'pin' (distributed/utils.py:704); this launcher refuses "
                "both, because a boot window is the expensive place to learn "
                "that a number was withdrawn",
            )
        )

    flags = (
        "--uneven-token-vector",
        ",".join(str(v) for v in parsed),
        "--uneven-token-vector-role",
        role,
    )
    if provenance:
        flags = flags + ("--uneven-token-vector-provenance", str(provenance))
    return DTokenVectorDecision(
        flags=flags,
        line=(
            "PDFLIP D-TOKEN-VECTOR: %s role=%s provenance=%s -- OPERATOR "
            "ADVISORY, checked against planner.retracted's register at the "
            "desk and not found withdrawn. role='seed' still promises "
            "supersession by this boot's measured per-rank capacity; 'pin' "
            "asserts the value against that measurement. Default is to ship "
            "nothing at all."
            % (
                ",".join(str(v) for v in parsed),
                role,
                provenance or "(undeclared -- the value match is what caught "
                "#602, so state it if you know it)",
            )
        ),
    )

def card_power_limits_w(cards: Sequence[Card]) -> List[Optional[float]]:
    """ENFORCED power limit per CUDA ordinal, W, from NVML; ``None`` if unknown.

    27B line (user 24.09. ~18:55Z: the cards run power-limited -- 5090 400 W of
    600, 3080 230 W of 320 -- and the limits may be raised). A per-layer cost
    measured at one limit is not the cost at another, so every boot prints
    its limits (:func:`card_power_line`) and a stage fit is checked against
    them (:func:`stage_fit_family_cost`)."""
    out: List[Optional[float]] = []
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        return [None for _ in cards]
    try:
        for c in cards:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(int(c.nvml_index))
                out.append(float(pynvml.nvmlDeviceGetEnforcedPowerLimit(h)) / 1000.0)
            except Exception:
                out.append(None)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return out

def card_power_line(cards: Sequence[Card], limits_w: Sequence[Optional[float]]) -> str:
    return CARD_POWER_MARKER + " " + " ; ".join(
        "stage %d card='%s' uuid=%s limit_w=%s"
        % (r, c.name, c.uuid, "unknown" if w is None else "%.0f" % w)
        for r, (c, w) in enumerate(zip(cards, limits_w))
    )

def chunked_prefill_size_of(argv: Sequence[str]) -> int:
    """The chunk this launcher's own argv states, read back off it.

    Same rule as ``--max-running-requests`` in the pool model: a second copy of
    the number here would drift the day the flag moves, and the depth price is
    linear in it.
    """
    argv = list(argv)
    if "--chunked-prefill-size" not in argv:
        raise PdFlipLaunchRefused(
            "W42 PdFlipDepthUnfunded: group P's argv states no "
            "--chunked-prefill-size, so one in-flight microbatch has no "
            "priceable size and the depth cannot be funded against the pool. "
            "Refusing rather than assuming a chunk."
        )
    return int(argv[argv.index("--chunked-prefill-size") + 1])

def _csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]

def _csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]

def _arm_run_peak_gib(arm) -> Optional[float]:
    """The arm's priced run peak, or None when the arm cannot say."""
    try:
        v = arm.predicted_run_peak_gib()
        return None if v is None else float(v)
    except Exception:  # noqa: BLE001
        return None

def arena_from_ledger(cur_gib: float, run_peak_gib: Optional[float], guard_gib,
                      *, enabled: bool = True) -> Tuple[Optional[float], str]:
    """``(new arena GiB or None, why)`` -- pure.  None means: leave it."""
    if not enabled:
        return None, "off (FLLIPER_PDFLIP_ARENA_FROM_LEDGER=0 or hicache disabled); arena %.2f GiB stays" % cur_gib
    if guard_gib is None:
        return None, "no --host-riegel-gib on this boot; arena %.2f GiB stays" % cur_gib
    if run_peak_gib is None:
        return None, "the arm carries no run peak; arena %.2f GiB stays" % cur_gib
    over = float(run_peak_gib) - float(guard_gib)
    if over <= 0:
        return None, ("run_peak %.2f <= riegel %.2f GiB (headroom %.2f); arena %.2f GiB stays"
                      % (run_peak_gib, guard_gib, -over, cur_gib))
    give = math.ceil(over * 4) / 4
    new = max(ARENA_FROM_LEDGER_FLOOR_GIB, float(cur_gib) - give)
    if new >= float(cur_gib):
        return None, ("run_peak %.2f is %.2f over the riegel %.2f but the arena is at its floor %.2f GiB"
                      % (run_peak_gib, over, guard_gib, cur_gib))
    return new, ("run_peak %.2f GiB is %.2f over the riegel %.2f -> arena %.2f -> %.2f GiB "
                 "(%d KV slots of 32 KiB), re-pricing"
                 % (run_peak_gib, over, guard_gib, cur_gib, new, int(new * (1 << 30) // 32768)))

def d_verdict_measured_terms(fremd, nicht_torch, card_terms, other_record, n: int, record_name: str):
    """27B port fix 21.10. (rc29b INT8: the #145 verdict printed REST 3096 MiB on D TP1, the card held 1332 MiB free at
    idle): ``(fremd, nichttorch, booked_from)`` for the D verdict. A flag given (``fremd`` / ``nicht_torch`` not None) is
    kept. Unset, they are booked from what this pass and the records MEASURED -- never guessed: fremd = the budget's
    ``dormant`` term (P's dormant footprint the budget charged), nichttorch = the budget's ``carve`` term (NVML reserved)
    + the record ``D_TORCH_CAP_OTHER_MIB`` (D's non-torch part: context/driver/communicator/lmem, max over the awake
    VRAM-PEAK lines of 8 boots). Without terms or record a term stays None (the verdict names it NICHT GEBUCHT)."""
    booked: List[str] = []
    v_fremd, v_nt = fremd, nicht_torch
    terms = list(card_terms) if card_terms is not None else None
    if terms is None or len(terms) != n:
        return v_fremd, v_nt, booked
    if v_fremd is None:
        v_fremd = [float(t.get("dormant", 0) or 0) for t in terms]
        booked.append("fremd=dormant_other (budget term)")
    if v_nt is None and other_record is not None and len(list(other_record)) == n \
            and all(o is not None for o in other_record):
        v_nt = [float(t.get("carve", 0) or 0) + float(o) for t, o in zip(terms, other_record)]
        booked.append("nichttorch=driver_carve (budget term) + RECORD %s" % record_name)
    return v_fremd, v_nt, booked

def build_expert_karte(total, ratios, fr_pp, fr_tp, p_stage_layers, mirror: bool = False):
    """(karte, p_layer_stage) -- the Version-2 (Platztausch) map when the
    realised P layer split matches FR_P, else the Version-1 map with
    p_layer_stage None. The ONE build both the map writer and the ledger's
    dry sizing call."""
    from flliper.srt.layers.moe import expert_map as _em

    _stages = [int(x) for x in (p_stage_layers or []) if int(x) > 0]
    if _stages and len(_stages) == len(fr_pp) and not mirror:
        _layer_stage = [s for s, n in enumerate(_stages) for _ in range(n)]
        return _em.build_nested(
            total=total,
            ratios=[int(float(x)) for x in ratios],
            fr_pp=[float(x) for x in fr_pp],
            fr_tp=[float(x) for x in fr_tp],
            p_layer_stage=_layer_stage,
            pad_tp=1,
        ), _layer_stage
    return _em.build(total=total,
                     ratios=[int(float(x)) for x in ratios],
                     fr_pp=[float(x) for x in fr_pp],
                     fr_tp=[float(x) for x in fr_tp],
                     mirror=mirror), None

def bs_source(flag: str, argv: Optional[Sequence[str]] = None) -> str:
    """``"flag"`` if this boot was TOLD its bs, ``"default"`` if it inherited it.

    Every boot record must be able to answer "was this operating point chosen
    for this boot, or is it the shipped provisional default?" without a second
    document -- the user order of 2026-09-09 calls the pair explicitly
    re-measurable, so a record that shows only the VALUE cannot later be sorted
    into "measured arm" and "inherited arm".

    Read off the argv that was PARSED, not off a comparison with the default:
    an operator who passes ``--d-bs 4`` explicitly while the default is also 4
    has still chosen it, and a value-comparison would silently call that boot a
    default boot. Both spellings argparse accepts are matched (``--d-bs 4`` and
    ``--d-bs=4``).
    """
    words = list(sys.argv[1:] if argv is None else argv)
    return "flag" if any(w == flag or w.startswith(flag + "=") for w in words) else "default"

def d_expect_from_p_records(profile: Optional[str]) -> bool:
    """Does D's EXPECTATION budget (before P's first sleep) price P's measured
    dormant residue from the records? The registry row decides
    (``ModelProfile.d_expect_from_p_records``); independent of the reserve
    question :func:`xchg_census_is_reserve` answers. No/unknown profile = the
    launcher default (``--profile`` qwen27b)."""
    row = pdflip_form.profile_row(profile or pdflip_form.DEFAULT_PROFILE)
    return bool(getattr(row, "d_expect_from_p_records", False))

def _cards_empty(cards: list[dict]) -> bool:
    """True iff NVML reports zero tenant-held MiB on every card this boot
    claimed (``BootState.cards``, i.e. ``Card.__dict__`` entries).

    Reuses the SAME reader :func:`teardown` already prints from
    (``nvml_registry.memory_snapshot`` / ``tenant_used_mib``, the
    carve-out-honest figure, #539) so this check and that print can never
    disagree about what "empty" means. Conservative on any read failure or
    on a card missing from the snapshot: an unread/unmatched card proves
    nothing, so this returns False rather than claiming an unverified
    guarantee -- the opposite mistake (claiming empty when it is not) is the
    one #1248 exists to stop repeating.
    """
    try:
        snap = nvml_registry.memory_snapshot()
    except Exception:  # noqa: BLE001 - unread NVML must read as "not proven empty", never crash the refusal path
        return False
    used_by_index = {d.index: m.tenant_used_mib for d, m in snap}
    for c in cards:
        idx = c.get("nvml_index") if isinstance(c, dict) else getattr(c, "nvml_index", None)
        if idx not in used_by_index or used_by_index[idx] != 0:
            return False
    return True
