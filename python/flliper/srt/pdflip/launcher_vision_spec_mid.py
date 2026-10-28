"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
import os

from flliper.srt.managers import corridor_guard
from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip import host_ledger
from flliper.srt.pdflip.launcher_d_leaf import TORCH_CACHE_CAP_ENV
from flliper.srt.pdflip.launcher_d_leaf import VISION_OFF
from flliper.srt.pdflip.launcher_d_leaf import VISION_TRANSIENT
from flliper.srt.pdflip.launcher_d_leaf import _csv_ints
from flliper.srt.pdflip.launcher_d_leaf import bs_source
from flliper.srt.pdflip.launcher_d_leaf import effective_reserve_by_card
from flliper.srt.pdflip.launcher_p_l3_leaf import _gcd_reduce
from flliper.srt.pdflip.launcher_p_l3_leaf import model_is_gguf_file
from flliper.srt.pdflip.launcher_p_l3_leaf import parse_group_env
from flliper.srt.pdflip.launcher_types import PdFlipLaunchRefused
from flliper.srt.pdflip.launcher_xchg_vision_leaf import VISION_PLACE_AUTO
from flliper.srt.pdflip.launcher_xchg_vision_leaf import VISION_SOURCE_DISK
from flliper.srt.pdflip.launcher_xchg_vision_leaf import strip_barlink_flags
from flliper.srt.pdflip.launcher_xchg_vision_leaf import user_reserve_env_value
from flliper.srt.pdflip.launcher_xchg_vision_leaf import vision_stage_knobs_refusal

#: VISION-SYNC LAW (user 02.10. ~08:00Z, both lines; ported from the NF line c651892375 launcher.py:24175-24205 by VISION-WEIGHTS AP4): the
#: switches that would let a vision stage outlive its pass (pdflip/vision_rank_runner: the async stage and its PP0 row term). Either one ON is
#: refused at launch, by name. On the 27B line the runner's default is the async stage (vision_rank_runner.vision_async_on, default ON): only an
#: EXPLICIT on is refused here, an unset switch is the old default path and stays unchanged; ``--pdflip-vision-place weights`` is synchronous
#: always (vision_rank_runner.arm_victims refuses the async switch at arming too).
VISION_ASYNC_SWITCHES: Tuple[str, ...] = (
    "FLLIPER_PDFLIP_VISION_ASYNC",
    "FLLIPER_PDFLIP_P_ROW_VISION_ASYNC",
)
VISION_SYNC_LAW = (
    "VISION-SYNC LAW (user 02.10. ~08:00Z, both lines): vision runs ONLY synchronously, BEFORE "
    "the real prefill; spare VRAM goes to MoE experts (NF) and to the L15 cache (27B), never "
    "to a vision tower that sits there across passes")
#: VISION-WEIGHTS AP4 (plan PLAN-VISION-GEWICHTE-VERDRAENGEN-1009 section 8): the places that are guaranteed broken in the dual layout. The tower
#: of kvtail needs free KV-tail pages and the dual P pool is born trimmed to 0 tokens (pdflip/dual_p_kv_stage.py:1-25), so ``reserve_tail_pages``
#: returns None (pdflip/vision_rank_stage.py:117-129); ``auto`` resolves to kvtail first. ``weights`` is the place of the dual.
VISION_PLACES_BROKEN_IN_DUAL: Tuple[str, ...] = ("kvtail", VISION_PLACE_AUTO)
W111_VISION_PLACE_DUAL = "W111 PdFlipVisionArmRefused"

def transport_argv(argv: List[str], transport: str) -> List[str]:
    """``argv`` as the chosen transport needs it. One seam, both groups."""
    return argv if str(transport) != "nccl" else strip_barlink_flags(argv)

def _vision_knob(ns, attr: str, default: str) -> str:
    """H125: one knob, refused by name when it cannot act (see
    vision_stage_knobs_refusal)."""
    val = str(getattr(ns, attr, default) or default)
    why = vision_stage_knobs_refusal(
        str(getattr(ns, "pdflip_vision", VISION_OFF)),
        str(getattr(ns, "pdflip_vision_source", VISION_SOURCE_DISK) or VISION_SOURCE_DISK),
        str(getattr(ns, "pdflip_vision_place", VISION_PLACE_AUTO) or VISION_PLACE_AUTO),
    )
    if why:
        raise PdFlipLaunchRefused(why)
    return val

def user_reserve_rank_env(
    user_reserve_by_card: Optional[Dict[str, int]],
    cards: Optional[Sequence["Card"]] = None,
    booked_rest_mib: Optional[Sequence[Optional[int]]] = None,
) -> Dict[str, str]:
    """The env entries that carry the operator's reserve to the ranks of ONE group.

    Empty unless the operator stated a reserve on at least one card:
    ``parse_user_reserve`` returns ``{uuid: 0}`` for every card when the flag is
    absent, and publishing that would turn "no reserve stated" into "reserve 0
    stated" on a rank, which silently beats the rank's own
    ``--rank-user-reserve-mib`` scalar fallback (``reserve_for_card``).

    With ``cards`` the value is the group's EFFECTIVE reserve
    (:func:`effective_reserve_by_card`, the same function ``budgets_from_dc`` prices
    with): a booked-rest card gets 0 and is named in
    ``corridor_guard.USER_RESERVE_MASKED_ENV`` (log only). Without ``cards`` (a group
    with no booked rest, e.g. P, and desk callers) the operator's value is published as
    stated.
    """
    if not user_reserve_by_card or not any(int(v) > 0 for v in user_reserve_by_card.values()):
        return {}
    if cards is None:
        if booked_rest_mib is not None:
            raise ValueError("a booked-rest vector needs the cards it belongs to")
        effective, masked = {u: int(v) for u, v in user_reserve_by_card.items()}, {}
    else:
        effective, masked = effective_reserve_by_card(cards, user_reserve_by_card, booked_rest_mib)
    out = {corridor_guard.USER_RESERVE_ENV: user_reserve_env_value(effective)}
    if masked:
        out[corridor_guard.USER_RESERVE_MASKED_ENV] = user_reserve_env_value(masked)
    return out

def tokenizer_path_decision(
    model: str, explicit: str = ""
) -> Tuple[Optional[str], Optional[str]]:
    """``(the --tokenizer-path both groups get, or None; a log line or None)``.

    * given: used on BOTH groups, refused (W163) unless it is a directory with
      one of the server's own sibling tokenizer files -- a typo dies here, not
      in two groups' tokenizer managers;
    * not given, not a GGUF file: ``(None, None)`` -- the argv is unchanged;
    * not given, a GGUF file whose tokenizer the server can find itself
      (transformers' GGUF route for a non-bespoke arch, or sibling files next
      to the .gguf): ``(None, line)``;
    * not given, a bespoke-arch GGUF (qwen35, qwen35moe, gemma4, ...) with no
      tokenizer next to it: W163 now, instead of the same ValueError in both
      groups' tokenizer managers after the cards are taken. Nothing is guessed:
      the fix is named, not applied (the unsloth Qwen3.8-27B GGUF directory
      ships no tokenizer; the FP8/INT8 checkpoints of the same model carry the
      same tokenizer.json).
    """
    from pathlib import Path

    from flliper.srt.utils.hf_transformers.tokenizer import (
        _SIBLING_TOKENIZER_FILES,
        _has_sibling_tokenizer,
    )

    wanted = ", ".join(_SIBLING_TOKENIZER_FILES)
    explicit = str(explicit or "").strip()
    if explicit:
        if not (os.path.isdir(explicit) and _has_sibling_tokenizer(Path(explicit))):
            raise PdFlipLaunchRefused(
                f"W163 PdFlipTokenizerRefused: --tokenizer-path {explicit!r} is not "
                f"a directory holding any of {wanted}."
            )
        return explicit, f"PDFLIP TOKENIZER --tokenizer-path {explicit} (both groups)"
    if not model_is_gguf_file(model):
        return None, None
    from flliper.srt.model_loader.gguf_registry import sibling_config_gguf_archs

    parent = os.path.dirname(os.path.abspath(model))
    # _peek_bespoke_gguf_arch's derivation (general.architecture of the metadata
    # part, in the registry's sibling-config set) over the header this launch
    # reads once anyway -- its own call would parse the GGUF again (8.5 s).
    header_arch = host_ledger.gguf_header_facts(model).arch
    arch = header_arch if header_arch in sibling_config_gguf_archs() else None
    if arch is None:
        return None, (
            f"PDFLIP TOKENIZER {model}: read by transformers from the GGUF itself "
            "(not a sibling-config arch)"
        )
    if _has_sibling_tokenizer(Path(parent)):
        return None, f"PDFLIP TOKENIZER {model}: the sibling files in {parent} (arch {arch})"
    raise PdFlipLaunchRefused(
        f"W163 PdFlipTokenizerRefused: {model} is a {arch} GGUF, whose tokenizer "
        f"the server reads ONLY from sibling files, and {parent} holds none of "
        f"{wanted}. Pass --tokenizer-path <dir> with this model's tokenizer "
        "(the safetensors checkpoint of the same model), or place the files "
        "next to the .gguf."
    )

def shipped_layer_split(
    stage_ratio: str,
    attn_stage_ratio: str,
    layer_set: str,
    is_full_attention: Sequence[bool],
    n_stages: int,
) -> List[int]:
    """The per-stage LAYER COUNTS group P's ARGV will actually run.

    The counterpart of :func:`p_stage_layers`, and the reason it exists: that
    function's own docstring says it reads "the SAME two score vectors
    ``argv_p`` passes", and since the merge train brought :func:`solve_p_cut`
    that premise is no longer true by construction -- argv_p passes whatever
    the solver returned.  A premise that used to hold and now has to be CHECKED
    is a guard, not a comment, so the split is derived from the argv strings
    themselves and compared (``main``, W46).

    Both argv forms are resolved through the runtime's OWN parser for that
    form -- ``parse_pp_layer_sets`` for a layer SET, ``derive_pp_layer_split``
    for the score pair -- never by counting the score vector, which is the
    conflation :func:`p_stage_layers` exists to undo.  ``[]`` when the argv
    states no cut at all; an empty answer is never compared as agreement.
    """
    if layer_set:
        from flliper.srt.distributed.utils import parse_pp_layer_sets

        return [
            len(x)
            for x in parse_pp_layer_sets(
                layer_set, len(list(is_full_attention)), int(n_stages), allow_gapped=True
            )
        ]
    if not stage_ratio:
        return []
    from flliper.srt.distributed.utils import derive_pp_layer_split

    return list(
        derive_pp_layer_split(
            _csv_ints(stage_ratio),
            is_full_attention=list(is_full_attention),
            attn_scores=_csv_ints(attn_stage_ratio) if attn_stage_ratio else None,
        )
    )

def torch_cache_cap_armed(ns) -> bool:
    """P0 (pdflip/torch_cache_cap.py) armed for group D: ``FLLIPER_PDFLIP_TORCH_CACHE_CAP=1``
    in --env-d -- after :func:`apply_profile_torch_cache_cap_default` that is
    the registry row's default unless --env-d named the switch itself."""
    return str(parse_group_env(getattr(ns, "env_d", "") or "").get(TORCH_CACHE_CAP_ENV, "0")
               ).strip() == "1"

def _weights_from_scores(scores: Sequence[float]) -> Tuple[int, ...]:
    """A ratio vector proportional to a measured per-rank rate.

    ``--rank-tp-ratio`` takes a comma-separated integer ratio
    (server_args.py:467 ``_parse_rank_tp_ratio``), so the rates are scaled to
    integers and gcd-reduced. x1000 because this rig's slowest and fastest
    measured rates differ by ~4x and rounding at x1 would quantise a 3.99x
    ratio to 4:1 -- the vector must carry the measurement, not a rounding of it.
    """
    lo = min(float(s) for s in scores)
    return _gcd_reduce([int(round(float(s) / max(lo, 1e-9) * 1000.0)) for s in scores])

def vision_async_refusal(ns, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """VISION-SYNC LAW: the refusal line when :data:`VISION_ASYNC_SWITCHES` are
    set ON in the launcher's own environment (which build_env and the front
    inherit) or in ``--env-p`` / ``--env-d``; None when every one is unset or
    off (an explicit 0/false/no/off is the law's own value and passes). Any
    spelling of a switch's rename family counts."""
    env = dict(os.environ if environ is None else environ)
    scopes = (
        ("launcher env", env),
        ("--env-p", parse_group_env(str(getattr(ns, "env_p", "") or ""))),
        ("--env-d", parse_group_env(str(getattr(ns, "env_d", "") or ""))),
    )
    hits: List[str] = []
    for where, scope in scopes:
        for name in VISION_ASYNC_SWITCHES:
            raw = pdflip_form._explicit_env(scope, name)
            if raw is not None and raw.lower() not in ("0", "false", "no", "off"):
                hits.append(f"{where} {name}={raw}")
    if not hits:
        return None
    return ("PDFLIP VISION-ASYNC refused: %s -- %s. Set it 0 (27B line: an UNSET FLLIPER_PDFLIP_VISION_ASYNC is the async stage, "
            "vision_rank_runner.vision_async_on default ON; NF line: unset = sync)" % (", ".join(hits), VISION_SYNC_LAW))

def vision_place_dual_refusal(ns, argv: Optional[Sequence[str]] = None) -> Optional[str]:
    """VISION-WEIGHTS AP4: ``--pdflip-vision-place kvtail|auto`` told on the command line together with ``--dual-layout`` (and the transient tower) is
    refused by name -- the dual P pool has no KV-tail page to lend, so the first image would be refused at request time (W105). Only a place that
    was TOLD counts (``bs_source``: the argv, not a comparison with the default): a profile that does not name the place keeps the old default path
    byte for byte (text-only serving works, the golden dry runs are unchanged); the place for the dual is ``weights``. None = consistent."""
    if not bool(getattr(ns, "dual_layout", False)):
        return None
    if str(getattr(ns, "pdflip_vision", VISION_OFF)) != VISION_TRANSIENT:
        return None
    place = str(getattr(ns, "pdflip_vision_place", VISION_PLACE_AUTO) or VISION_PLACE_AUTO)
    if place not in VISION_PLACES_BROKEN_IN_DUAL or bs_source("--pdflip-vision-place", argv) != "flag":
        return None
    return (f"{W111_VISION_PLACE_DUAL}: --pdflip-vision-place {place} with --dual-layout: the dual P pool is born trimmed to 0 tokens "
            "(pdflip/dual_p_kv_stage.py), so the tower has no KV-tail page to lend and the first image request would be refused (W105). "
            "Use --pdflip-vision-place weights (the tower borrows weight memory of PP0, only the hull parts that are not D's shard) "
            "or leave --pdflip-vision-place unset")

def vision_tower_dir(ns) -> str:
    """VISION-GGUF (09.10.): the directory the transient tower is read from -- ``--model``, or for a GGUF FILE ``--tokenizer-path``
    (``planner.vision_stage_load.tower_source_dir``, the rank's rule; raises ``VisionStageLoadRefused`` when that holds no sharded
    checkpoint). ``--tokenizer-path`` is read only for a GGUF model."""
    from flliper.srt.planner.vision_stage_load import tower_source_dir

    model = str(ns.model)
    if not model_is_gguf_file(model):
        return model
    return tower_source_dir(model_path=model, tokenizer_path=str(ns.tokenizer_path or ""))
