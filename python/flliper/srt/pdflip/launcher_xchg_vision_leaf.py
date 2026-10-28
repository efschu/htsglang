"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
import os
import re

from flliper.srt.environ import envs
from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip import host_ledger
from flliper.srt.pdflip import ring_table
from flliper.srt.pdflip import vote_fixed_live as _vfl
from flliper.srt.pdflip import weight_exchange_transport
from flliper.srt.pdflip.launcher_d_leaf import VISION_CHOICES
from flliper.srt.pdflip.launcher_d_leaf import VISION_OFF
from flliper.srt.pdflip.launcher_d_leaf import VISION_RESIDENT
from flliper.srt.pdflip.launcher_d_leaf import VISION_TRANSIENT
from flliper.srt.pdflip.launcher_p_l3_leaf import FP4_NATIVE_MIXED_ARGV
from flliper.srt.pdflip.launcher_p_l3_leaf import FP4_UNIFORM_MARLIN_ARGV
from flliper.srt.pdflip.launcher_refuse_proc_leaf import VISION_TRANSIENT_OVERRIDE
from flliper.srt.pdflip.launcher_refuse_proc_leaf import WEIGHT_SOURCE_ARMED
from flliper.srt.pdflip.launcher_refuse_proc_leaf import WEIGHT_SOURCE_DEFAULT
from flliper.srt.pdflip.launcher_types import PdFlipLaunchRefused
import json
import shlex

#: Flags that only make sense while barlink owns the group's collectives.
#: Each takes a value except the bare --barlink itself.
BARLINK_FLAGS_WITH_VALUE = (
    "--barlink-transport",
    "--barlink-bar1-window-mib",
    "--barlink-bar1-cap-cycles",
    "--barlink-uncovered-class",
)
BARLINK_FLAGS_BARE = ("--barlink",)
#: 27B RC7-X (e28c450a0d): read by group D's scheduler (``_pdflip_store_short_tail_x``).
STORE_SHORT_TAIL_X_ENV = "FLLIPER_PDFLIP_STORE_SHORT_TAIL_X"
#: H125 (NF, user order 2026-09-24 10:15Z): where the transient tower's bytes
#: come from (`disk` = the checkpoint shard, O_DIRECT, the 27B stage; `ram` =
#: a tmpfs image of the tower extent, staged once by P's PP0) and where it
#: sits on the card (`auto` = the KV tail, else free VRAM; `kvtail`; `free`).
#: The defaults publish NOTHING: the P env of a transient boot is the 27B one,
#: and a text-only boot is byte-identical to every boot before H125. The
#: variable names are the rank's (`pdflip.vision_rank_stage`), one spelling.
VISION_SOURCE_DISK = "disk"
VISION_SOURCE_RAM = "ram"
#: user order 27.10.: ONE standing pinned host copy of the tower, an H2D per image;
#: the RANK's default under `--pdflip-vision-place weights` (FLLIPER_PDFLIP_VISION_PINNED=0
#: = off), so the launcher default stays `disk` and publishes nothing
VISION_SOURCE_PINNED = "pinned"
VISION_PLACE_AUTO = "auto"
#: The D-side vector every production boot runs, and the DEFAULT of
#: ``choose_host_ledger``'s ``d_vector`` parameter. Named once because the
#: lane record is keyed on it: two spellings of this string are two cut keys,
#: and a boot that prices under one key and publishes under the other charges
#: for lanes it does not create (or creates lanes it did not charge).
XCHG_D_VECTOR_DEFAULT = "tp3"
#: --tokenizer-path (27B line G2, 2026-09-25): the tokenizer BOTH groups load.
#: None, the default, adds nothing to either argv (the server reads the
#: tokenizer from --model-path, byte-identical to every argv before the flag).
#: Installed once by :func:`apply_tokenizer_path`, read by every
#: :func:`common_flags` call, so P and D can never tokenize differently.
_TOKENIZER_PATH: Dict[str, Optional[str]] = {"path": None}
#: #1248: first W-code substring in a refusal message, for the one-line
#: summary. Not every PdFlipLaunchRefused carries one (build_env's
#: PYTORCH_CUDA_ALLOC_CONF refusal does not) -- covers the "Wnn" and "Wnnb"
#: forms actually used in this file (e.g. W11b PdFlipDraftBuildUnaccounted).
_W_CODE_RE = re.compile(r"\bW\d+[a-z]?\b")

def strip_barlink_flags(argv: List[str]) -> List[str]:
    """``argv`` without the barlink family -- the NCCL development mode.

    Removal, not substitution: without --barlink the group takes the stock
    flliper dispatch, which builds the PyNccl communicator barlink suppresses.
    Written as a filter over the ONE argv builder rather than as a second
    argv builder, so the two arms cannot drift apart in anything except the
    transport.
    """
    out: List[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token in BARLINK_FLAGS_WITH_VALUE:
            skip = True
            continue
        if token in BARLINK_FLAGS_BARE:
            continue
        out.append(token)
    return out

def strip_speculative_flags(tokens: Sequence[str]) -> Tuple[List[str], List[str]]:
    """``(kept, stripped)``: every ``--speculative-*`` token of an extra argv,
    with its value when it takes one.

    H25: no draft on P means ZERO ``--speculative-*`` tokens on group P
    (#1264 property 1), and ``ring_table.p_carries_drafter`` reads ANY such
    token as "P carries a drafter".  The Next-Flash arm passes
    ``--speculative-draft-model-path`` through ``--extra-p``; without this P
    would still load the MTP head (and a draft path with no algorithm).  A
    value is a following token that is not itself a flag; ``--flag=value`` is
    one token.
    """
    kept: List[str] = []
    stripped: List[str] = []
    toks = [str(t) for t in tokens]
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--speculative-"):
            stripped.append(t)
            if "=" not in t and i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                stripped.append(toks[i + 1])
                i += 1
        else:
            kept.append(t)
        i += 1
    return kept, stripped

def store_short_tail_env(x_tokens: int, x_d_guard: int, x_split: Optional[bool] = None) -> Dict[str, str]:
    """27B RC7-X: group D's env addition when its W50 riegel stands above the
    launch X: the #1324/#1471 store-short tail keeps pricing a stalled read's
    remainder against the LAUNCH X (a recovery prefill on D halts every running
    decode, like a SHORT grant, which the front bounds by X_busy). Empty when
    the ceiling is unset -- D's env byte for byte as before.

    UNIFY S7: part of the profile's busy/idle split (``x_split``,
    FLLIPER_PDFLIP_X_IDLE_REGRANT; qwen27b on, nextflash off) -- off returns {}.
    ``x_split`` None = read the switch (the published form's profile)."""
    if x_split is None:
        x_split = bool(envs.FLLIPER_PDFLIP_X_IDLE_REGRANT.get())
    if not x_split or int(x_d_guard) <= int(x_tokens):
        return {}
    return {STORE_SHORT_TAIL_X_ENV: str(int(x_tokens))}

def xchg_region_pairs(n_cards: int) -> int:
    """DIRECTED cross-card pairs on ``n_cards`` cards: ``n * (n - 1)``.

    Round-2 review F8: this was typed as ``6``, which is that arithmetic
    evaluated on THIS rig and nowhere stated as arithmetic -- on any other card
    count the region size would have been silently wrong while every test that
    grades it still passed, because the tests were handed the same 6.  The
    count comes from the boot's own card list now.
    """
    n = int(n_cards)
    return n * (n - 1) if n > 1 else 0

def _vector_len(value: str) -> Optional[int]:
    """Entry count of a per-card vector value; None for a scalar (no comma:
    one value for every card, e.g. ``FLLIPER_MOE_SCRATCH_SLOTS=32``) or an
    unexpanded shell variable."""
    v = str(value or "").strip().strip("'\"")
    if not v or v.startswith("$") or "," not in v:
        return None
    return len(v.split(","))

def _tree_bytes(path: str) -> Tuple[int, int, int]:
    """(allocated bytes, apparent bytes, entry count) of a file or tree.

    WHICH BYTE COUNT, stated because the two differ here by 0.53 GiB and only
    one of them is host RAM: ``st_blocks * 512`` is what the tmpfs actually
    OCCUPIES (and what ``df`` and the cgroup's ``shmem`` count), ``st_size`` is
    the apparent size ``ls`` shows.  Measured on the real orphan
    ``/dev/shm/hicache-weg2-fix973`` (2026-09-08): apparent 4,042,810,432 B =
    3.77 GiB, allocated 3,471,384,576 B = 3.23 GiB -- and 3.23 GiB is the figure
    boot weg2dk7 attributed and the figure a sweep frees.  Reporting the
    apparent size as "freed" would over-claim by half a GiB.
    """
    def _pair(p: str) -> Tuple[int, int]:
        st = os.lstat(p)
        return st.st_blocks * 512, st.st_size

    try:
        alloc, apparent = _pair(path)
    except OSError:
        return 0, 0, 0
    if not os.path.isdir(path) or os.path.islink(path):
        return alloc, apparent, 1
    alloc, apparent, n = 0, 0, 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                a, b = _pair(os.path.join(root, name))
            except OSError:
                continue
            alloc += a
            apparent += b
            n += 1
    return alloc, apparent, n

def vision_on_d_of(ns) -> bool:
    """The launcher's ONE decision whether group D encodes images (user order 27.10.: flip layouts yes, dual layout no,
    ``FLLIPER_PDFLIP_VISION_D=0`` off). The front reads the same function (``front.vision_on_d``) with the same inputs."""
    from flliper.srt.pdflip.front import vision_on_d

    dual = bool(getattr(ns, "dual_layout", False) or getattr(ns, "dual_share", False))
    return vision_on_d(str(getattr(ns, "pdflip_vision", VISION_OFF)), dual)

def vision_stage_knobs_refusal(vision: str, source: str, place: str) -> str:
    """A source/place that is not the default only means something under
    `--pdflip-vision transient`; anywhere else it is refused by name (a knob
    that silently does nothing is a boot nobody can read)."""
    if vision == VISION_TRANSIENT:
        return ""
    if source != VISION_SOURCE_DISK or place != VISION_PLACE_AUTO:
        return (
            f"W111 PdFlipVisionArmRefused: --pdflip-vision-source {source} / "
            f"--pdflip-vision-place {place} were given, but --pdflip-vision is "
            f"{vision!r} -- they act only on the transient stage"
        )
    return ""

def vision_source_host_line(model: str, source: str) -> str:
    """The host-RAM price of `--pdflip-vision-source ram`, named at the launch:
    the tower extent (header read only) lives in a tmpfs image for the life
    of P's PP0 (shmem, charged to memory.current). Empty for `disk`."""
    if source not in (VISION_SOURCE_RAM, VISION_SOURCE_PINNED):
        return ""
    try:
        from flliper.srt.planner.vision_stage_load import find_tower_shard, tower_extent

        ext = tower_extent(find_tower_shard(model))
        mib = f"{(ext.file_last_end - ext.file_first_offset) / (1 << 20):.1f} MiB"
    except Exception as exc:  # noqa: BLE001 -- unmeasured is said, never guessed
        mib = f"size unread ({type(exc).__name__}: {exc})"
    if source == VISION_SOURCE_PINNED:
        return (
            f"W102 PdFlipVisionStage SOURCE=pinned: P's PP0 stages the tower extent into ONE pinned host "
            f"copy once ({mib}), kept for the life of the boot (pinned, counted in memory.current); "
            "every image then loads from it by H2D instead of the disk"
        )
    return (
        f"W102 PdFlipVisionStage SOURCE=ram: P's PP0 stages the tower extent into a tmpfs "
        f"image once ({mib}), held in host RAM for the life of the boot (shmem, counted in "
        "memory.current, NOT in the host ledger's terms); every image then loads from RAM "
        "instead of the disk"
    )

def vision_model_flags(vision: str):
    """The model-side argv this vision form needs. PURE -- enumerable at a
    desk, which is where the three forms are pinned against each other."""
    if vision == VISION_OFF:
        return ["--no-enable-multimodal"]
    if vision == VISION_TRANSIENT:
        return ["--json-model-override-args", VISION_TRANSIENT_OVERRIDE]
    if vision == VISION_RESIDENT:
        return []
    raise ValueError(
        f"unknown --pdflip-vision {vision!r}; expected one of "
        f"{list(VISION_CHOICES)}. Refusing rather than defaulting: a boot "
        "whose tower state nobody stated is a boot that can return plausible "
        "wrong text for an image request."
    )

def w38_armed_line(argv_of_p: Sequence[str]) -> str:
    """State group P's carrierless-PP arm AT LAUNCH, once.

    RECONCILED 2026-09-08 (pdflip train, W38 vs #1245): there is ONE arm for the
    carrierless PP fact now, and it is #1245's ``undistributable`` drop --
    ``clear_state_aligned_extent_undistributable`` at scheduler.py:12157, gated
    on ``pp_row_carrier_present(self)`` exactly as W38 was.  W38's own gate at
    ``Scheduler._prefetch_kvcache`` is DELETED: it refused the store READ
    outright and was desk-only, while #1245 is BOOT-PROVEN (weg2rg6, 141 group-P
    prefill passes, 20 leg-1 legs, three real drops, W27 genuine 0 over that
    population).  So group P DOES read the store again; what it may not do is
    ADOPT a host hit that landed on this rank alone.

    Read off the argv this launcher is about to run rather than asserted from
    memory: ``--pp-size N`` with N > 1 is the predicate's own precondition, so
    if that flag ever changes the line changes with it instead of lying.
    """
    pp = 1
    for i, a in enumerate(argv_of_p):
        if a == "--pp-size" and i + 1 < len(argv_of_p):
            try:
                pp = int(argv_of_p[i + 1])
            except ValueError:
                pp = 1
    if pp <= 1:
        return ("PDFLIP #1245 NOT ARMED: group P is launched with --pp-size %d, so no rank can hold a "
                "fact it cannot tell its peers and nothing is dropped" % pp)
    return ("PDFLIP #1245 ARMED (W38 RETIRED INTO IT): group P is launched --pp-size %d and the no-flip "
            "PP form carries no #631 row carrier, so an asynchronously-completed host hit that landed "
            "on ONE rank is DROPPED at admission (#1245 UNDISTRIBUTABLE LOAD-BACK DROPPED) instead of "
            "moving that rank's prefix_indices alone -- the W27 width divergence that killed boots "
            "weg2sc1 and weg2rg3. THE STORE READ ITSELF IS NOT REFUSED any more: W38 "
            "PdFlipCarrierlessPpStoreRead disarmed it entirely and was the desk-only half of the same "
            "fact; the two arms were reconciled to the boot-proven one. WHAT IT COSTS: only the "
            "extents actually dropped, counted by that log line, not P's whole L3 prefix reuse. "
            "Measured per drain epoch by the front's 'PDFLIP P-PREFIX-REUSE' line; the write-through "
            "and D's own store read are untouched, and the drop lifts itself the moment a carrier "
            "exists (#968 PP0-authoritative materialisation is the named remedy)." % pp)

def yarn_context_tokens(override_json) -> Optional[int]:
    """YaRN x2 (27.09.): the context a YaRN rope override stretches the model
    to -- ``original_max_position_embeddings x factor`` of a ``rope_type``
    ``yarn`` block in ``--json-model-override-args`` (in ``text_config`` or at
    the top level, ``rope_parameters`` or ``rope_scaling``); None without one.
    The runtime derives the same number (``get_context_length``)."""
    if not override_json:
        return None
    try:
        args = json.loads(override_json) if isinstance(override_json, str) else dict(override_json)
    except (TypeError, ValueError):
        return None
    for scope in (args.get("text_config"), args):
        if not isinstance(scope, dict):
            continue
        for key in ("rope_parameters", "rope_scaling"):
            rope = scope.get(key)
            if (isinstance(rope, dict) and rope.get("rope_type", rope.get("type")) == "yarn"
                    and rope.get("original_max_position_embeddings") and rope.get("factor")):
                return int(int(rope["original_max_position_embeddings"]) * float(rope["factor"]))
    return None

def xchg_form_argv(argv: Sequence[str], weight_source: str) -> List[str]:
    """Group P's form argv, PLUS the xchg token when this boot arms the exchange.

    B4e, and ``ring_table`` wrote the requirement at the name it defined for
    it: *"when the xchg slice lands on the line its launcher must add the same
    token to the argv it hands ``solve`` (one line, ``p_argv +
    [XCHG_FORM_TOKEN]``), or its own boots will rank a serving source as their
    form's twin -- the mirror of this defect."*  The slice has landed and the
    token was missing, so #1305 item 4 -- written to keep an xchg source away
    from a boot that does NOT arm the exchange -- was excluding every boot of
    this form from the selection of a boot that DOES.

    MEASURED cost of the omission, 2026-09-11: B4c's census took group P's
    per-card rows from a boot that ran the PP layer split ``[42, 11, 11]``
    while the armed form runs ``[39, 13, 12]``, because every boot that ran the
    armed split carries ``PDFLIP-XCHG-REGION`` and was excluded.  P's rows are a
    LAYER PLACEMENT, so that is not a rounding difference.

    THE PREDICATE IS ``WEIGHT_SOURCE_ARMED``, not a hand-written comparison,
    and that is the whole design of it: the same derived tuple decides whether
    ``prepare_xchg_env`` publishes the region at all, so the MARKER this boot
    emits and the TOKEN it claims cannot disagree, and a future arm carries
    both by construction.  A hand comparison here is exactly how ``exchange``
    was missed at step 7.
    """
    out = list(argv)
    if str(weight_source) in WEIGHT_SOURCE_ARMED:
        out.append(ring_table.XCHG_FORM_TOKEN)
    return out

def user_reserve_env_value(user_reserve_by_card: Dict[str, int]) -> str:
    """``{card_uuid: MiB}`` -> the value of ``corridor_guard.USER_RESERVE_ENV``.

    #236: ONE serialiser for the front's ``fenv`` and every group's env, so the
    three consumers (front, P ranks, D ranks) cannot be handed different text
    for the same mapping.
    """
    return json.dumps({u: int(v) for u, v in user_reserve_by_card.items()})

def xchg_bounce_arm_pins_host(weight_source: str, oncard_mode: str) -> bool:
    """Does THIS ARM pin host bytes for the exchange's carrier?  A BOOLEAN.

    ONE PRODUCER OF THE PREDICATE, and after AMENDMENT 5 it produces ONLY the
    predicate.  It used to answer "how many bytes" as well
    (`xchg_bounce_charge_bytes`, returning the retired
    `host_ledger.xchg_bounce_bytes` ceiling), which made it a second producer
    of a SIZE that `xchg_bounce.bounce_terms` already owns -- and the two
    disagreed by 4x on one payload, because the ceiling counted
    `ONCARD_SLOTS_MAX 8` slots where the term prices `SLOTS_PER_PAIR 2`.  The
    size now has exactly one owner and this call answers only the arm question.

    BOTH ARMS DECIDE IT.  `--pdflip-weight-source ring` creates no region at all;
    `--pdflip-xchg-oncard ipc` creates one but exports a VRAM bounce that is
    freed with its own leg, so no bounce FILE exists and no host byte is
    pinned.  Host bytes exist on `shadow`/`exchange` + `host` and nowhere else,
    and this is the same arm string `prepare_xchg_env` publishes to the
    ranks -- so the arm a rank enforces and the arm that was paid for cannot be
    two readings.

    It is a function rather than an expression inside `choose_host_ledger`
    because the property under test -- "the charge fires on exactly the arm
    that can allocate" -- must be provable without a host to read.
    """
    if str(weight_source) == WEIGHT_SOURCE_DEFAULT:
        return False
    return str(oncard_mode) == weight_exchange_transport.ONCARD_MODE_HOST

def xchg_lane_count(stage_ratio, legs, d_vector: str = XCHG_D_VECTOR_DEFAULT):
    """``(n_lanes, provenance)`` for THIS boot's cut -- the one producer.

    #1368: the ledger charges the bounce and the launcher publishes its INPUTS
    to the ranks, and until now each built the cut key at its own call site.
    The rank-side guard compares the published ``n_lanes`` against the lanes it
    really enumerates and refuses when the price is below them, so the two
    sites disagreeing is not a cosmetic drift -- it is a refused boot, and the
    refusal would name the ledger rather than the publication that diverged.
    """
    return host_ledger.resolve_xchg_lanes(
        host_ledger.xchg_cut_key(str(stage_ratio or ""), str(d_vector), str(legs)))

def tokenizer_path_flags() -> List[str]:
    """The installed tokenizer as argv words: ``[]`` when none was installed."""
    path = _TOKENIZER_PATH.get("path")
    return ["--tokenizer-path", str(path)] if path else []

def vote_fixed_live_root() -> str:
    """VOTE-FIXED-LIVE: the store root (``FLLIPER_PDFLIP_VOTE_FIXED_DIR``, else the rig cache dir of the footprint
    and card-probe files); the launcher hands the resolved path to the ranks, so both read one directory."""
    got = str(os.environ.get(_vfl.ENV_DIR, "") or "").strip()
    if got:
        return got
    try:
        from flliper.srt.rigmon.card_probe import CACHE_DIR
    except Exception:  # noqa: BLE001 -- rigmon import shape
        CACHE_DIR = os.path.expanduser("~/.cache/flliper")
    return CACHE_DIR

def _without_budget_flag(extra: List[str]) -> List[str]:
    """``extra`` without ``--rank-gpu-memory-mib`` (both spellings)."""
    out: List[str] = []
    skip = False
    for tok in extra:
        if skip:
            skip = False
            continue
        if tok == "--rank-gpu-memory-mib":
            skip = True
            continue
        if tok.startswith("--rank-gpu-memory-mib="):
            continue
        out.append(tok)
    return out

def uniform_marlin_argv(
    quant_method: str, uniform_marlin: bool, fp4_native_mixed: bool = False
) -> List[str]:
    """The server argv BOTH groups get for --fp8-uniform-marlin: the NVFP4
    Marlin backend for a ModelOpt checkpoint, nothing otherwise (FP8 is forced
    through the environment). With --fp4-native-mixed the NVFP4 linears get
    the native-mixed backend instead (one native layout, kernel per rank); the
    FP8 linears stay on Marlin."""
    if uniform_marlin and str(quant_method or "").lower() == "modelopt":
        return list(FP4_NATIVE_MIXED_ARGV if fp4_native_mixed else FP4_UNIFORM_MARLIN_ARGV)
    return []

def with_uniform_marlin_argv(extra: str, argv: Sequence[str]) -> str:
    """``--extra-p`` / ``--extra-d`` with :func:`uniform_marlin_argv` appended;
    unchanged when it is empty or already carries the same backend. A different
    --fp4-gemm-backend in the extras is REFUSED (W160): the flag promises one
    layout on every rank and an operator-pinned backend would break it."""
    if not argv:
        return extra
    toks = shlex.split(extra or "")
    flag, value = argv[0], argv[1]
    for i, tok in enumerate(toks):
        got = None
        if tok == flag and i + 1 < len(toks):
            got = toks[i + 1]
        elif tok.startswith(flag + "="):
            got = tok.split("=", 1)[1]
        if got is None:
            continue
        if got == value:
            return extra
        raise PdFlipLaunchRefused(
            "W160 PdFlipFp8LayoutRefused: --fp8-uniform-marlin puts every NVFP4 linear "
            "on Marlin (%s %s) on BOTH groups, but the group extras pin %s %s -- one "
            "tensor would have two byte layouts across the flip" % (flag, value, flag, got))
    return " ".join(shlex.quote(t) for t in toks + list(argv))

def turn_anchor_env(token: Optional[int]) -> Dict[str, str]:
    """``--turn-anchor-token``: group P's environment for the TURN ANCHOR
    (pdflip/turn_anchor.py; {} when off -- the byte-identity guarantee). Group
    D reads nothing new: the anchor is a regular store node at a page."""
    if token is None:
        return {}
    if int(token) <= 0:
        raise SystemExit(f"--turn-anchor-token must be a positive token id, got {token}")
    return {"FLLIPER_PDFLIP_TURN_ANCHOR_TOKEN": str(int(token))}

def xchg_census_is_reserve(profile: Optional[str]) -> bool:
    """Does ``profile``'s exchange-form census REPLACE the group-D constant?

    The registry row decides (``ModelProfile.d_residue_census``): Next Flash
    yes (cb1575e94e), the 27B no -- its line keeps the weg2xsn14 constant.
    No/unknown profile = the launcher default (``--profile`` qwen27b)."""
    row = pdflip_form.profile_row(profile or pdflip_form.DEFAULT_PROFILE)
    return bool(getattr(row, "d_residue_census", False))

def _w_code_of(e: BaseException) -> str:
    """Best-effort W-code out of a refusal's message; "W?" when absent
    rather than inventing one."""
    m = _W_CODE_RE.search(str(e))
    return m.group(0) if m else "W?"
