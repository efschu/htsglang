"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
import functools
import os
import re

from flliper.srt.compat_shims import env_name_variants as _env_name_variants
from flliper.srt.name_compat import canonical_env
from flliper.srt.pdflip import DEFAULT_PP_ORDERED_CUT
from flliper.srt.pdflip import card_identity
from flliper.srt.pdflip import form as pdflip_form
from flliper.srt.pdflip import pp_ordered_cut_for as pdflip_pp_ordered_cut_for
from flliper.srt.pdflip import refusals
from flliper.srt.pdflip import ring_table
from flliper.srt.pdflip import weight_exchange
from flliper.srt.pdflip import x_curves as pdflip_x_curves
from flliper.srt.pdflip.launcher_d_leaf import CARD_POWER_MARKER
from flliper.srt.pdflip.launcher_d_leaf import KV_TOKEN_CUT_MARKER
from flliper.srt.pdflip.launcher_d_leaf import VISION_TRANSIENT
from flliper.srt.pdflip.launcher_d_leaf import _SPEC_FORM
from flliper.srt.pdflip.launcher_p_l3_leaf import CHUNKED_PREFILL_TOKENS
from flliper.srt.pdflip.launcher_p_l3_leaf import CODE_CARDS
from flliper.srt.pdflip.launcher_p_l3_leaf import WEIGHT_SOURCE_EXCHANGE
from flliper.srt.pdflip.launcher_p_l3_leaf import _import_duplex_gate
from flliper.srt.pdflip.launcher_types import BubbleMeasurement
from flliper.srt.pdflip.launcher_types import Card
from flliper.srt.pdflip.launcher_types import PdFlipLaunchRefused
from flliper.srt.pdflip.launcher_types import PdFlipTokenCutNotWired
import json
import shlex
import subprocess

#: UNIFY S3: the 27B row of the model-profile registry (pdflip/form.py PROFILES).
#: The module-level measured constants below are ALIASES of this row -- the
#: launcher's default ``--profile`` -- kept for their readers and pins; every
#: boot-time reader goes through :func:`_pconst` (the published form's profile),
#: so a Next-Flash boot reads the nextflash row.
_PROFILE_27B = pdflip_form.PROFILES[pdflip_form.PROFILE_QWEN27B]
DC_RESERVE_SLACK_MIB = 64
#: #1234 C6 -- the DEVELOPMENT transport switch, and the one number it moves.
#:
#: The user's order of 2026-09-07 put development on NCCL until barlink
#: covers the Weg-2 message classes again. NCCL is a LAUNCHER-level mode, not
#: a tier inside barlink: a barlink-owned group never constructs a PyNccl
#: communicator (parallel_state.should_build_pynccl), and building one would
#: change the flip path, which is out of bounds. Dropping --barlink instead
#: leaves the stock flliper dispatch in place, which does build one.
#:
#: MEASURED (A/B boot weg2ab0, ARM 0): D's dormant residue on the 5090 rises
#: 2230 -> 2310 MiB under NCCL, because libnccl's buffers are not
#: memory-saver-tagged and therefore survive the sleep. That tripped W19
#: DormantResidueRefused against DC_MEASURED_D_5090_MIB 2228 +
#: DC_RESERVE_SLACK_MIB 64 = 2292 and killed the run. The extra slack lives
#: HERE, behind the switch, and not in the constant -- under 'bar1' nothing
#: about the flip path changes, which is the entire point.
DC_RESERVE_SLACK_NCCL_MIB = 192
#: H25 (Nutzer-Order 24.09. 08:25Z, "Draft auf P streichen"): the operator's
#: input for "does group P carry the MTP head". The launcher flag
#: ``--draft-kv-on-p`` and this variable are TWO SPELLINGS OF ONE FACT (the
#: draft-KV producer IS the draft on P); :func:`resolve_draft_on_p` reconciles
#: them once, the launcher writes the resolved value back into
#: ``ns.draft_kv_on_p`` and into its own environment (front and both groups
#: inherit it through ``build_env``), and
#: ``pdflip_memory_saver.draft_tag_in_family`` reads it on every rank.
DRAFT_ON_P_ENV = "FLLIPER_PDFLIP_DRAFT_ON_P"
DRAFT_ON_P_CONFLICT = "W127 PdFlipDraftOnPConflict"
#: RC7-X (27B): the name a negative --x-busy-tokens is refused with at launch.
X_BUSY_NEGATIVE_NAME = "W155 PdFlipXBusyNegative"
#: RC2 review (L1): the name a --d-short-drain-tokens above X is refused with.
SHORT_DRAIN_ABOVE_X_NAME = "W153 PdFlipShortDrainAboveX"
#: #1273: where a waking group's weight bytes come from. ``ring`` is today's
#: path, byte for byte, and stays the default until S6 has booted the other
#: one; ``exchange`` moves them card-to-card and is what W71 prices.
#: ``shadow`` (#1273 S5) is the third value and it is RING-AUTHORITATIVE: the
#: weights still come from the host ring, byte for byte, and the exchange runs
#: beside them into buffers nothing reads so its bytes can be compared against
#: the ring's.  It is the only arm that can prove the exchange on silicon
#: without the exchange being able to break a flip.
WEIGHT_SOURCE_CHOICES = ("ring", "exchange", "shadow")
WEIGHT_SOURCE_DEFAULT = "ring"
#: The arms that NEED the region armed and published to the ranks -- every
#: choice except the default.  DERIVED and not a second literal: a future arm
#: added to `WEIGHT_SOURCE_CHOICES` is armed by construction instead of
#: silently publishing nothing, which is precisely the defect step 7 closed for
#: `exchange` (it had been absent from a hand-written comparison since S5).
WEIGHT_SOURCE_ARMED = tuple(a for a in WEIGHT_SOURCE_CHOICES
                            if a != WEIGHT_SOURCE_DEFAULT)
#: #239 S3a: the D argv that carries the planner's token cut to the runtime
#: (server_args._resolve_form_a_dcp -> rank_role.resolve_dcp_under_host_kv).
D_TOKEN_CUT_FLAGS = (
    "--uneven-token-vector",
    "--uneven-token-vector-role",
    "--uneven-token-vector-provenance",
)
#: The provenance the runtime's #797 register sees: the planner, this boot.
D_TOKEN_CUT_PROVENANCE = "planner-239-token-cut"
#: The boot token's env key as ``KEY=`` bytes, this tree's spelling first, then the other
#: generation's.  A rank's own environment is folded onto one spelling by the package hook, so a
#: server of the other generation carries ONLY the other one (see :func:`_proc_boot_tag`).
BOOT_TOKEN_ENV_KEYS: Tuple[bytes, ...] = tuple(
    k.encode() + b"=" for k in _env_name_variants("FLLIPER_PDFLIP_BOOT_TOKEN")
)
#: `--json-model-override-args` payload for the transient form. One constant
#: so the launcher and the tests cannot drift.
VISION_TRANSIENT_OVERRIDE = '{"language_model_only": true}'
#: Task #47 Scheibe 6a (20.09.): the checkpoint PROFILE the two groups are built
#: for. ``qwen27b`` is every boot before this flag existed (byte-identical argv
#: and env). ``nextflash`` = Qwen3.8 Next Flash: P = PP3 (tp 1 per stage), D =
#: Form A (host,worker,worker), whose rank_role collapses DCP to 1 and REFUSES an
#: inherited FLLIPER_UNEVEN_DCP=1 -- so the uneven-DCP facts (EARLY_READ_FACTS,
#: env AND flag half) are owned per group here (flip_nextflash_groups
#: GROUP_ENV_VALUES: P 1/1, D 0/0) and the flag half is dropped on both.
#: UNIFY S3: the names and their rows live in the registry (pdflip/form.py
#: PROFILES); what differs per profile is a table row there, never an ``if``
#: on the name here.
PROFILE_QWEN27B = pdflip_form.PROFILE_QWEN27B
MM_SKIP_HASH_ENV = "FLLIPER_MM_SKIP_COMPUTE_HASH"
#: R17's gate, read from the module that owns the key so the launcher's printed
#: verdict and the rank's actual key can never be computed from two numbers.
DUPLEX_GATE = _import_duplex_gate()
#: The launch check's body, run BY THE TREE THE RANKS IMPORT (FIX 3 round 3,
#: finding 2).  It sets the duplex variable under the NAME THAT TREE USES -- not
#: the launcher's -- so a tree that renamed it is a failed probe rather than a
#: silently unconfigured one, and it asks for the KEY, which is the only artefact
#: the launcher and the rank must agree on.  ``CUDA_VISIBLE_DEVICES=''`` in the
#: caller's env means no context is bought to answer.
_KEY_PROBE = (
    "import json, os, sys\n"
    "from flliper.srt.managers import pdflip_memory_saver as s\n"
    "os.environ[s.PCIE_DUPLEX_ENV] = sys.argv[1]\n"
    "out = {}\n"
    "for u in json.loads(sys.argv[2]):\n"
    "    try:\n"
    "        keys = {s.pcie_lock_path(u, direction=d) for d in s.PCIE_DIRECTIONS}\n"
    "    except TypeError:\n"
    "        keys = {s.pcie_lock_path(u)}\n"
    "    out[u] = [len(keys), len(s.PCIE_DIRECTIONS)]\n"
    "sys.stdout.write('PDFLIP-KEYS ' + json.dumps(out))\n"
)
#: B4e (ii): the refusal is W48's, and NO new code is minted for it.  W48
#: ``PdFlipRingFormMismatch`` IS "form key X != source Y" -- the launcher already
#: raises that class for its own un-priceable argv -- and the W-code census
#: keys on (code, NAME) pairs, so a second name under W48 would be a new
#: collision while a new number would be a second spelling of one event.
PdFlipXchgFormSourceMissing = ring_table.PdFlipRingFormMismatch
#: AP4 1526: ``--pdflip-xchg-census-map`` of THIS launch, set once by ``main`` (the
#: three census readers below take no ``ns``; their call texts are pinned by
#: desk tests).  "" = no operator mapping.
_XCHG_CENSUS_MAP: List[str] = [""]
#: The one line ``read_pp_bubble`` understands, emitted per window per rank by
#: ``scheduler_components/pp_bubble.py:summary_line``. Anchored on the whole
#: field name including its ``=``: a bare number would match milliseconds
#: elsewhere in the line, which is the bare-ticket-number trap in miniature.
_BUBBLE_RE = re.compile(
    r"PP-BUBBLE rank=(?P<rank>\d+) .*?"
    r" n=(?P<n>\d+) \(n_gaps=(?P<n_gaps>\d+),"
    r" forward_ms=(?P<forward>[0-9.]+),"
    r" bubble_ms=(?P<bubble>[0-9.]+),"
    r" starved_ms=(?P<starved>[0-9.]+);"
)
#: The runtime's transport for an explicit per-stage layer SET. It is READ by
#: the ranks and WRITTEN by exactly one thing: this launcher, from
#: ``--pp-layer-set`` or from the solver's own chosen map. An INHERITED value
#: is refused (W44), never honoured -- see :func:`refuse_inherited_layer_set`.
PP_LAYER_SET_ENV = "FLLIPER_PP_LAYER_SET"
_PREFILL_RE = re.compile(
    r"\b(?P<rank>PP\d+)\] Prefill batch,.*?#new-token: (?P<new>\d+), "
    r"#cached-token: (?P<cached>\d+)"
)
_CARD_POWER_RE = re.compile(r"stage (\d+) card='([^']*)' uuid=(\S+) limit_w=(\S+)")

def pick_d_residue_record(rows: Sequence[Dict[str, object]], cards: Sequence[Card]
                          ) -> Tuple[Optional[Dict[str, object]], str]:
    """#227: which group-D dormant-image sample prices THIS boot's D residue (``rows`` oldest first, already
    through the calibration identity). The newest whose ``vram_residue_mib`` names EXACTLY this boot's cards
    (same card set = same D topology); else the newest that covers every card (the pre-#227 answer: a sample
    of a bigger card set); else the newest at all, so :func:`dc_residue_from_record` names what is missing.
    ``(None, why)`` without any sample."""
    want = {c.uuid for c in cards}
    if not rows:
        return None, "no sample"

    def _keys(r):
        v = r.get("vram_residue_mib")
        return set(v) if isinstance(v, dict) else set()

    for how, ok in (("exact card set", lambda k: k == want), ("covering card set", lambda k: want <= k)):
        hit = [r for r in rows if ok(_keys(r))]
        if hit:
            r = max(hit, key=lambda e: str(e.get("at", "")))
            return r, f"{how}: boot {r.get('boot_tag', '?')}"
    r = max(rows, key=lambda e: str(e.get("at", "")))
    return r, f"newest, no sample covers this card set: boot {r.get('boot_tag', '?')}"

def _refuse_value_once(code: str, text: str, log: Optional["Log"] = None) -> None:
    """AP1 1006: ``refusals.refuse_value(code, text, PdFlipLaunchRefused)`` -- the
    SAME (code, text) is remembered once per forced boot (the W19 selector is read
    by several passes per card; AP4's ``load_xchg_census_for_cards`` dedups the
    same way). Without ``--force`` it raises every time, exactly as before."""
    if refusals.forced_boot() and (code, text) in {
            (d["code"], d["text"]) for d in refusals.forced_list()}:
        return
    refusals.refuse_value(code, text, PdFlipLaunchRefused, log=log)

def reserve_slack_mib(transport: str) -> int:
    """Dormant-residue slack for group D under this transport, in MiB.

    ONE definition, so the NCCL arm's +128 MiB cannot drift away from the
    reason it exists. Under 'bar1' this is the unchanged
    :data:`DC_RESERVE_SLACK_MIB`, and the flip path's arithmetic is exactly
    what it was.
    """
    return (DC_RESERVE_SLACK_NCCL_MIB if str(transport) == "nccl"
            else DC_RESERVE_SLACK_MIB)

def spec_form_is_dflash() -> bool:
    return str(_SPEC_FORM["form"]) == "DFLASH"

def rank_wave_floor(demand: Sequence[int], scratch_after: Sequence[int],
                    capture_on: Sequence[int], waves: int) -> Tuple[List[str], List[int]]:
    """#239 S3g floor, ONE formula for the planner and the rank: the captured
    step of the cap's seats routes D_r = min(ids, E - R) ids over
    C_r = LRU + staging (the scratch AFTER the stage rows moved into the seat
    rows) + capture_on[r] (the seat/stage rows the capture counts ON: the
    rank's capture floor where it has KV cells, #251c; 0 where it has none --
    'seat rows ... OFF until a D phase funds them'), in at most ``waves``
    waves: D_r <= waves x C_r on EVERY rank. Returns (one line per rank with
    its air ``waves x C - D``, the ranks that break it)."""
    lines, bad = [], []
    for r, (d, c0, on) in enumerate(zip(demand, scratch_after, capture_on)):
        c = int(c0) + int(on)
        air = int(waves) * c - int(d)
        lines.append("rang%d D %d <= %d x (%d + %d) = %d, Luft %d%s"
                     % (r, int(d), int(waves), int(c0), int(on), int(waves) * c, air,
                        "" if air >= 0 else " -- BRICHT"))
        if air < 0:
            bad.append(r)
    return lines, bad

def _rank_vec(raw: Optional[str], n: int) -> List[int]:
    vals = [int(float(x)) for x in str(raw or "").split(",") if x.strip()]
    if len(vals) == 1:
        vals = vals * n
    return (vals + [0] * n)[:n] if vals else [0] * n

def resolve_draft_on_p(flag_value: str, flag_explicit: bool,
                       env_set: bool, env_value: bool,
                       profile_default_on: bool = False) -> Tuple[bool, str]:
    """``(draft_on_p, provenance)`` for group P, from flag and env.

    The ENV is the authority since H25 (its default is False, the user
    order); an explicit flag that contradicts an EXPLICIT env is refused by
    name (W127) -- an A/B arm whose two inputs say opposite things is the boot
    that gets attributed to the wrong form.  An explicit ``--draft-kv-on-p
    on`` against the UNSET env (every arm since #1264 passes the flag) does
    not refuse: it is overruled by the order and the line says so, because the
    producer it asks for has had no reader since fnFL2x63 (H1: PUBLISH-SWEEP
    draft_issued=0, D runs DRAFT-COLD every flip).

    UNIFY S6: THE ONE RULE for "does group P carry a draft producer", called
    by the ONE writer of the ``p_draft`` axis (pdflip/form.resolve_form, injected
    like parse_group_env). ``profile_default_on`` is the booted profile's row
    (``p_draft != none``): False for nextflash (H25 above, unchanged), True for
    qwen27b (``p_draft=cold``, 27B 2026-09-24: the DFlash producer stays built
    on P, computes nothing) -- there the env is still the explicit authority,
    and the flag is honoured instead of overruled.
    """
    flag_on = str(flag_value) == "on"
    if env_set:
        if flag_explicit and flag_on != bool(env_value):
            raise PdFlipLaunchRefused(
                f"{DRAFT_ON_P_CONFLICT}: --draft-kv-on-p {flag_value} contradicts "
                f"{DRAFT_ON_P_ENV}={'1' if env_value else '0'}. Both name whether "
                "group P carries the MTP draft head; drop one of them."
            )
        return bool(env_value), f"env {DRAFT_ON_P_ENV}={'1' if env_value else '0'}"
    if profile_default_on:
        if flag_explicit:
            return flag_on, f"--draft-kv-on-p {flag_value} (profile default: producer on P)"
        return True, "profile default: producer on P (p_draft cold|compute)"
    if flag_explicit and flag_on:
        return False, (f"default {DRAFT_ON_P_ENV}=0 (Nutzer-Order 24.09.) OVERRULES "
                       f"--draft-kv-on-p {flag_value}: the producer has no reader "
                       "since fnFL2x63; set the env to 1 to keep the draft on P")
    return False, f"default {DRAFT_ON_P_ENV}=0 (Nutzer-Order 24.09.)"

def resolve_x_ceiling(ceiling_flag: Optional[int], x_tokens: int,
                      x_busy: Optional[int] = None) -> Tuple[int, int, str]:
    """H84: ``(D's --tp-prefill-max-tokens, the front's --x-ceiling-tokens or 0, line)``.

    UNIFY S4: ONE form of the NF H84 and the 27B RC7-X ceiling (the same
    intent, doubly built; the 27B had already taken the H84 names). D refuses
    by W50 on its OWN ``--tp-prefill-max-tokens``; the front routes on its LIVE
    X. Without the flag both stay the start X, byte-identical to before. With
    it, D's riegel is raised to the ceiling and the front is told the same
    number, so the live X may rise up to exactly what D accepts and never above
    it -- raising only the front's X sent 5-9k to D, D answered W50 and the
    request went round through P. A ceiling below the start X is lifted to it:
    D's riegel may never sit below what the front already routes to D. Group P
    carries no ``--tp-prefill-max-tokens`` and is not touched.

    ``x_busy`` (27B ``--x-busy-tokens``): the floor of the front's X-SOLO band
    -- up to it D prefills while it decodes others, above it only as a
    singleton. None = the front's default (the start X, the H84 band floor);
    a negative value is refused (W155); 0 = no D prefill while D decodes
    others.
    """
    if x_busy is not None and int(x_busy) < 0:
        raise SystemExit(
            f"{X_BUSY_NEGATIVE_NAME}: --x-busy-tokens {x_busy} < 0. It bounds the uncached "
            f"tokens D prefills while it decodes other requests; 0 = none, unset = the start X.")
    asked = int(ceiling_flag or 0)
    x = int(x_tokens)
    busy = "" if x_busy is None else f"; --x-busy-tokens {int(x_busy)} (X-SOLO band floor)"
    if asked <= 0:
        return x, 0, (f"X CEILING: off (--x-ceiling-tokens 0) -- group D --tp-prefill-max-tokens {x} "
                      f"= the front's start X, which is also the ceiling of the front's live X{busy}")
    ceiling = max(asked, x)
    lifted = "" if ceiling == asked else f" (asked {asked} < start X {x}: lifted to the start X)"
    return ceiling, ceiling, (
        f"X CEILING: --x-ceiling-tokens {ceiling}{lifted} -- group D --tp-prefill-max-tokens "
        f"{ceiling} (its W50 riegel; D prefills in --chunked-prefill-size {CHUNKED_PREFILL_TOKENS} "
        f"chunks); front --tp-prefill-max-tokens {x} (start X, unchanged) --x-ceiling-tokens "
        f"{ceiling}: the live X re-solves within [{CHUNKED_PREFILL_TOKENS}, {ceiling}] and a request "
        f"between {x if x_busy is None else min(int(x_busy), x)} and the live X goes to D only as a "
        f"singleton (PDFLIP X-SOLO); group P unchanged{busy}")

def resolve_x_mode_launch(ns, cards, x_tokens: int) -> "pdflip_x_curves.XModeLaunch":
    """X-CURVES 1006: ``--x-mode`` / ``--x-curves`` / ``--x-curves-beyond``
    resolved ONCE, before D's riegel: the words checked, the curve file loaded
    and checked against THIS boot's checkpoint, form and live cards. A
    refusal is a SystemExit naming its W-code (W190..W197), like W155 beside
    it -- main() keeps its zero try/except law. No flag = the live mode and
    every argv / line byte-identical. The dual layout refuses every one of the
    flags (W197): no X rework for Dual."""
    boot_form = ns.pdflip_boot_form
    try:
        return pdflip_x_curves.resolve_launch(
            mode=ns.x_mode, curves_path=ns.x_curves, beyond=ns.x_curves_beyond,
            ceiling_flag=ns.x_ceiling_tokens, x_tokens=int(x_tokens),
            model=pdflip_form.model_key(ns.model),
            form=pdflip_x_curves.form_axes_of(boot_form) if boot_form is not None else "",
            hardware=",".join(card_identity.inventory_signature(cards)),
            dual=bool(getattr(ns, "dual_layout", False)))
    except pdflip_x_curves.XCurvesRefused as e:
        raise SystemExit(str(e)) from e

def refuse_short_drain_above_x(n_tokens: int, x_tokens: int, x_provenance: str) -> None:
    """LAW 4 SUMMED OVER A DRAIN, at launch (RC2 review, L1).

    ``--d-short-drain-tokens N`` (27B idle policy b) hands a queued SHORT-only
    backlog of at most N uncached tokens to group D in ONE drain. Each request
    is <= X by its own route verdict, but the SUM is what D prefills at once,
    and law 4 (user veto 2026-09-10: D never prefills above X) holds for that
    sum too. X here is the value this launch hands the front and group D:
    :func:`resolve_x` has already applied the front floor to the flag
    (``max(floor, --tp-prefill-max-tokens)``) or derived it. N > X can never
    be served as asked, so it is refused by name instead of being quietly cut.
    The front keeps its own riegel beside this one -- every drain is capped at
    ``min(N, X in force)`` -- because the live X re-solve moves X during a
    boot, which no launch-time check can see. 0 = off, never refused.
    """
    n, x = int(n_tokens or 0), int(x_tokens)
    if n > 0 and n > x:
        raise SystemExit(
            f"{SHORT_DRAIN_ABOVE_X_NAME}: --d-short-drain-tokens {n} exceeds X={x} "
            f"({x_provenance}). One drain would hand group D {n} uncached tokens to "
            f"prefill at once; law 4 caps what D prefills at X, summed over the drain "
            f"too. Pass --d-short-drain-tokens <= {x}, or 0 for off."
        )

def resolve_pool_floor(
    override: Optional[int],
    n_stages: Optional[int] = None,
) -> Tuple[Optional[int], Optional[Tuple[int, ...]], str]:
    """The P-cut POOL FLOOR's SOURCE and its RULE LINE -- before the solve.

    Same shape as :func:`resolve_x` for the same reason: a number that can come
    from two places must publish WHICH, or the boot log cannot be read back.
    Three states, and the middle one is the one a reader gets wrong:

    * flag ABSENT -> ``(None, DEFAULT_PP_ORDERED_CUT, rule)``: the floor is NOT
      a number yet.  It is READ OFF THIS BOOT'S OWN FRONTIER by the solver as
      the priced pool of the ordered cut (#1305,
      ``pp_cut_launch.derive_pool_floor_from_cut``), and the solver refuses by
      name (``W67``) when that cut is not on the frontier.  The numeric line
      is therefore logged AFTER the solve (:func:`pool_floor_line`); this
      function only states the rule, so a refusal inside the solve still has
      the source on the record above it.  The previous form -- a constant
      448,027 typed here as the midpoint of an interval read off ONE boot's
      frontier -- shipped 42,11,11 under the order's name on boot weg2sn5pre
      when the census budgets moved the frontier; the operator ruled the
      constant out (record SECTION 1bg).
    * flag <= 0 -> ``(None, None, ...)``, ``source=flag``.  OFF, exactly:
      ``None`` is what the solver saw before #1286b existed, so
      ``--pp-solve-pool-floor 0`` restores the unfloored makespan byte for
      byte rather than approximately.  A floor of literally zero would be
      indistinguishable in effect but would print ``pool_floor=0``, and an
      instrument that says a floor is armed when none is, is the class this
      rig calls instrument-text-lies.
    * flag > 0 -> ``(that value, None, ...)``, ``source=flag``.  The operator
      outranks the derivation; nothing warns, because choosing the number is
      the flag's job.

    The cut itself stays SOLVED either way -- this only bounds the set
    ``--pp-solve-objective`` ranks over.
    """
    cut_s = ",".join(str(n) for n in DEFAULT_PP_ORDERED_CUT)
    if override is None and n_stages is not None and pdflip_pp_ordered_cut_for(n_stages) is None:
        # HW-P1c 1003 (PP-CUT-FLOOR): the ordered cut is a user order for THREE
        # stages; a P group of another size has no order and no floor.
        return None, None, (
            f"pool_floor=none source=n-stages ({int(n_stages)} stages) -- the shipped default "
            f"(the floor of the ordered cut {cut_s}, user order 2026-09-09) is a "
            f"{len(DEFAULT_PP_ORDERED_CUT)}-stage cut and names no floor for {int(n_stages)}; the "
            f"cut is SOLVED over the unfloored makespan (planner/pp_cut.py), as with "
            f"--pp-solve-pool-floor 0. --pp-solve-pool-floor N still overrides"
        )
    if override is None:
        return None, tuple(int(n) for n in DEFAULT_PP_ORDERED_CUT), (
            f"source=default-from-ordered-cut {cut_s} -- the SHIPPED default, user "
            f"order 2026-09-09 verbatim: '39,13,12 mit bs2 im pp layout soll standard "
            f"werden vorerst'. The floor is READ OFF THIS BOOT'S FRONTIER by the solve "
            f"below as the priced pool of {cut_s} (#1305: no constant, no interval, "
            f"no midpoint -- a number typed here drifted off a moved frontier on boot "
            f"weg2sn5pre); the cut stays solver-chosen, the floor only bounds the "
            f"pool. Not on the frontier = W67 refusal, never a neighbour. "
            f"--pp-solve-pool-floor N overrides, 0 turns it off"
        )
    if int(override) <= 0:
        return None, None, (
            f"pool_floor=none source=flag (--pp-solve-pool-floor {int(override)}) -- OFF by "
            f"operator override; the objective ranks over the unfloored feasible set, "
            f"which is the pre-#1286b behaviour exactly. The shipped default (the floor "
            f"of the ordered cut {cut_s}, user order 2026-09-09) is NOT in force"
        )
    return int(override), None, (
        f"pool_floor={int(override)} source=flag (--pp-solve-pool-floor) -- operator "
        f"override; the shipped default (the floor of the ordered cut {cut_s}, user "
        f"order 2026-09-09, '39,13,12 ... soll standard werden vorerst') is NOT in force"
    )

def pool_floor_line(decision) -> str:
    """THE ``PP-CUT POOL FLOOR:`` line, from the decision, AFTER the solve.

    Pure, so a desk test can render it.  Three sources, one format:
    ``pool_floor=<n|none> source=<flag|default-from-ordered-cut <cut>|none>``.
    Printed after the solve because under the default the NUMBER exists only
    once the frontier does (#1305); the rule was printed before the solve by
    :func:`resolve_pool_floor` so a refusal inside the solve is attributable.
    """
    src = str(getattr(decision, "pool_floor_source", "none") or "none")
    floor = getattr(decision, "pool_floor", None)
    cut = getattr(decision, "pool_floor_cut", None)
    if src == "default-from-ordered-cut" and cut is not None:
        cut_s = ",".join(str(n) for n in cut)
        return (
            f"PP-CUT POOL FLOOR: pool_floor={int(floor)} "
            f"source=default-from-ordered-cut {cut_s} (frontier of this boot) -- "
            f"user order 2026-09-09: '39,13,12 mit bs2 im pp layout soll standard "
            f"werden vorerst'; the floor is the priced pool of {cut_s} on the "
            f"PP-CUT FRONTIER line of THIS boot, so 'fastest cut that clears it' "
            f"selects {cut_s} exactly (#1305)"
        )
    if src == "flag":
        return (
            f"PP-CUT POOL FLOOR: pool_floor={int(floor)} source=flag "
            f"(--pp-solve-pool-floor, operator override; the ordered-cut default "
            f"is NOT in force)"
        )
    return (
        "PP-CUT POOL FLOOR: pool_floor=none source=none (--pp-solve-pool-floor 0: "
        "OFF by operator override; the objective ranks over the unfloored feasible "
        "set, the pre-#1286b behaviour exactly)"
    )

def pp_cut_chunk_mix(log_path: Optional[str], fallback_width: int, fallback_prefix: int,
                     graph_bucket: int):
    """(chunk mix, provenance): the predecessor boot's PP0 chunks as
    (width, prefix bin, mode, count); without a readable log ONE cell, the
    configured chunk at the design prefix, named as such."""
    from flliper.srt.pdflip import p_stage_model as _psm

    if log_path:
        try:
            with open(log_path, errors="replace") as fh:
                mix = _psm.chunk_mix_from_samples(_psm.samples_from_rank_lines(fh))
        except OSError as exc:
            mix, why = (), "unreadable (%s)" % exc
        else:
            why = "no paired PP0 rank line"
        if mix:
            return mix, "MEASURED chunk mix of %s (%d PP0 chunks, %d cells)" % (
                log_path, int(sum(n for *_x, n in mix)), len(mix))
    else:
        why = "no predecessor P log"
    mode = _psm.MODE_GRAPH if (graph_bucket and int(fallback_width) <= int(graph_bucket)) else _psm.MODE_EAGER
    return ((int(fallback_width), int(fallback_prefix), mode, 1.0),), (
        "FALLBACK chunk mix: %s -- one cell, the configured chunk %d at the design prefix %d (%s)"
        % (why, int(fallback_width), int(fallback_prefix), mode))

def refuse_unwired_token_cut(ns, boot_form) -> None:
    """#239 S2/S3e: a real boot of ``kv=qsa_forma_dcp`` stops here, by name,
    while a seam it stands on (``rank_role.TOKEN_CUT_SEAMS``) is unwired. The
    refusal names the seams from the register, so it lifts when -- and only
    when -- the last of them is marked built."""
    if boot_form is None or boot_form.kv != "qsa_forma_dcp" or getattr(ns, "dry_run", False):
        return
    from flliper.srt import rank_role

    missing = rank_role.unwired_token_cut_seams()
    if missing == ("F14",) and getattr(ns, "pdflip_disable_hicache", False):
        # #239 S4a: F14 is the worker's bytes in the HOST TIER and the store;
        # a boot without either (--pdflip-disable-hicache: no L2, no L3, #1386)
        # has no such bytes, so the runtime half is complete for it.
        print(f"{KV_TOKEN_CUT_MARKER} F14-FREI (no Host-Tier): --d-kv-token-cut "
              f"{ns.d_kv_token_cut} bootet mit --pdflip-disable-hicache; F14 (Host-Tier/"
              "Store des KV-Workers) ist offen und wird hier nicht betreten.", flush=True)
        return
    if not missing:
        return
    raise PdFlipTokenCutNotWired(
        f"{KV_TOKEN_CUT_MARKER}: --d-kv-token-cut {ns.d_kv_token_cut} (form kv=qsa_forma_dcp) "
        f"is planned but not wired -- Form A seam(s) {', '.join(missing)} unwired "
        f"({'; '.join(rank_role.SEAMS[m].what for m in missing)}; #239 S3d/S3e). "
        "--dry-run prints the planned cut; a real boot needs every seam built.")

def publish_d_token_cut(ns, vector, label: str) -> Optional[str]:
    """#239 S3a: ship the token cut to group D (``ns.extra_d``), or refuse.

    An operator ``--d-uneven-token-vector`` next to a planned cut is a second
    statement of the same fact; which one the runtime would read is argparse
    order, so it is refused by name. Returns the log line, None when there is
    no cut to ship.
    """
    if vector is None:
        return None
    if getattr(ns, "d_uneven_token_vector", None):
        raise PdFlipLaunchRefused(
            f"{KV_TOKEN_CUT_MARKER}: --d-uneven-token-vector "
            f"{ns.d_uneven_token_vector} and the planned token cut "
            f"{list(vector)} (--d-kv-token-cut {ns.d_kv_token_cut}) both state "
            "D's token ownership. Drop one: a stated vector goes through "
            "--d-kv-token-cut, which prices it.")
    toks = shlex.split(str(getattr(ns, "extra_d", "") or ""))
    out: List[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in D_TOKEN_CUT_FLAGS:
            i += 2
            continue
        if any(t.startswith(f + "=") for f in D_TOKEN_CUT_FLAGS):
            i += 1
            continue
        out.append(t)
        i += 1
    out += [
        D_TOKEN_CUT_FLAGS[0], ",".join(str(int(v)) for v in vector),
        D_TOKEN_CUT_FLAGS[1], "pin",
        D_TOKEN_CUT_FLAGS[2], D_TOKEN_CUT_PROVENANCE,
    ]
    ns.extra_d = shlex.join(out)
    return (
        f"{KV_TOKEN_CUT_MARKER} {label} (#239 S3a) an D: --uneven-token-vector "
        f"{','.join(str(int(v)) for v in vector)} role=pin provenance="
        f"{D_TOKEN_CUT_PROVENANCE} -- der Planer-Schnitt ist die Laufzeit-Form "
        "(dcp = Raenge, jeder Rang haelt die kv-Koepfe seiner Token)")

def select_cards(cards: Sequence["Card"], selection: Optional[Sequence[int]]) -> List["Card"]:
    """The cards whose NVML index is in ``selection`` (all when None), in
    NVML order; an index NVML did not report is refused BY NAME."""
    cards = list(cards)
    if selection is None:
        return cards
    have = {int(c.nvml_index) for c in cards}
    missing = [i for i in selection if int(i) not in have]
    if missing:
        raise PdFlipLaunchRefused(
            f"{CODE_CARDS}: --cards {','.join(str(i) for i in selection)}: NVML index "
            f"{','.join(str(i) for i in missing)} not reported (NVML sees "
            f"{','.join(str(i) for i in sorted(have)) or 'no card'})")
    want = {int(i) for i in selection}
    return [c for c in cards if int(c.nvml_index) in want]

def record_card_power(records: List[Dict], cards: List[Card], log) -> None:
    """Release table row 27: the power limit and SM clock ceiling of every
    card go into the boot record (``state.cards[i]``: power_limit_w,
    power_limit_default_w, power_limit_max_w, sm_clock_max_mhz; None = NVML
    did not answer) and one ``POWER-LIMIT rank`` line per card into the
    launcher log. Never raises: a missing answer is recorded as such."""
    from flliper.srt.registry import nvml as _nvml
    from flliper.srt.pdflip import power_limit as _pl

    try:
        by_uuid = {p.uuid: p for p in _nvml.power_snapshot()}
        why = ""
    except Exception as exc:  # noqa: BLE001
        by_uuid, why = {}, f"{type(exc).__name__}: {exc}"
    for i, c in enumerate(cards[:len(records)]):
        p = by_uuid.get(c.uuid)
        # a COPY of the record: the Card object itself is not grown
        records[i] = dict(records[i], **{
            k: (getattr(p, k) if p is not None else None)
            for k in ("power_limit_w", "power_limit_default_w", "power_limit_max_w", "sm_clock_max_mhz")})
        log(_pl.boot_line({"launcher_card": i}, p, why or "card not in the NVML power snapshot"))

def records_inventory(profile: str) -> Tuple[Tuple[str, ...], str]:
    """(the inventory the profile's MEASURED records were taken on, its source)."""
    from flliper.srt.pdflip import profile_records as _pr

    inv = _pr.inventory_of(str(profile))
    if inv is not None:
        return inv, f"profile_records_data/{profile}.json inventory"
    return card_identity.REFERENCE_INVENTORY, (
        f"profile {profile!r} records declare no inventory: every record of this tree "
        "was measured on the reference rig")

def session_pids(sid: int) -> set:
    out = subprocess.run(["ps", "-eo", "pid,sid"], capture_output=True, text=True).stdout
    pids = set()
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) == sid:
            pids.add(int(parts[0]))
    return pids

def shm_holder_pids(path: str, proc_root: str = "/proc") -> List[int]:
    """Every LIVE pid that maps or holds open a file at or under ``path``.

    Read out of ``/proc/<pid>/maps`` and ``/proc/<pid>/fd`` rather than from
    ``fuser``: ``fuser`` is one exit code for a whole argv (the #1217 check
    could not say WHICH file was held, or by whom), it is not always installed,
    and it cannot be handed a fake tree by a test.  The pid list this returns is
    printed in the refusal, so a live holder is actionable instead of a boolean.

    A path that appears with the kernel's ``(deleted)`` suffix still counts: an
    unlinked-but-mapped region still occupies the memory this sweep exists to
    free.  Matching is on the full path with a boundary, never a substring, so
    ``hicache-weg2-fix973`` never matches ``hicache-weg2-fix9730``.
    """
    prefix = os.path.realpath(path)
    holders: List[int] = []

    def _hit(candidate: str) -> bool:
        p = candidate.strip()
        if p.endswith(" (deleted)"):
            p = p[: -len(" (deleted)")]
        return p == prefix or p.startswith(prefix + "/")

    try:
        entries = os.listdir(proc_root)
    except OSError:
        return holders
    for entry in sorted(entries, key=lambda e: int(e) if e.isdigit() else 0):
        if not entry.isdigit():
            continue
        found = False
        try:
            with open(f"{proc_root}/{entry}/maps") as f:
                for line in f:
                    parts = line.split(maxsplit=5)
                    if len(parts) == 6 and _hit(parts[5]):
                        found = True
                        break
        except OSError:
            pass
        if not found:
            fd_dir = f"{proc_root}/{entry}/fd"
            try:
                for fd in os.listdir(fd_dir):
                    try:
                        if _hit(os.readlink(f"{fd_dir}/{fd}")):
                            found = True
                            break
                    except OSError:
                        continue
            except OSError:
                pass
        if found:
            holders.append(int(entry))
    return holders

def _proc_argv(pid: str, proc_root: str) -> List[str]:
    try:
        with open(f"{proc_root}/{pid}/cmdline", "rb") as f:
            raw = f.read()
    except OSError:
        return []
    return [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]

def _proc_ppid(pid: int, proc_root: str) -> int:
    try:
        with open(f"{proc_root}/{pid}/stat") as f:
            # comm may hold spaces and ')': the fields after the LAST ')' are
            # "state ppid ...".
            return int(f.read().rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return 0

def _proc_has_cuda_context(pid: str, proc_root: str) -> bool:
    """A CUDA context holds ``/dev/nvidia*`` open (fd) and mapped (maps)."""
    fd_dir = f"{proc_root}/{pid}/fd"
    try:
        for fd in os.listdir(fd_dir):
            try:
                if os.readlink(f"{fd_dir}/{fd}").startswith("/dev/nvidia"):
                    return True
            except OSError:
                continue
    except OSError:
        pass
    try:
        with open(f"{proc_root}/{pid}/maps") as f:
            for line in f:
                if "/dev/nvidia" in line:
                    return True
    except OSError:
        pass
    return False

def _proc_boot_tag(pid: str, proc_root: str) -> Optional[str]:
    """The Weg-2 boot tag a server was started under, from its own
    ``FLLIPER_PDFLIP_BOOT_TOKEN`` (``<tag>:<epoch>:<launcher pid>``, published by
    :func:`build_env` for both groups).  ``None`` = no token (a stock server of
    another operator) or unreadable -- both count as a FOREIGN tag.

    Rename transition: either spelling of the key (:data:`BOOT_TOKEN_ENV_KEYS`),
    this tree's first when a process somehow carries both.  A server of the
    other generation with a token is judged by its TAG like one of ours, not
    as "no token"."""
    try:
        with open(f"{proc_root}/{pid}/environ", "rb") as f:
            raw = f.read()
    except OSError:
        return None
    found: Dict[bytes, bytes] = {}
    for kv in raw.split(b"\0"):
        for key in BOOT_TOKEN_ENV_KEYS:
            if kv.startswith(key) and key not in found:
                found[key] = kv[len(key):]
    for key in BOOT_TOKEN_ENV_KEYS:
        if key in found:
            return found[key].decode("utf-8", "replace").split(":", 1)[0]
    return None

def split_preflight_refusals(text: str) -> Tuple[List[str], List[str]]:
    """``(ram_lines, other_lines)``: the ``REFUSED:`` lines of ``host_ledger_preflight.sh``
    split by cause (HOSTWARN2, 13.10.2026).

    The script (outside the tree, ``devtools/host_ledger_preflight.sh``) has two verdicts
    and ONE exit code: ``REFUSED: MemAvailable ...`` (host RAM under its floor) and
    ``REFUSED: only N GiB free on /`` (DISK under 100 GiB). The user rule 06.10. covers host
    RAM only, so the launcher reads the script's own log and lets only the RAM lines through
    as warnings; a disk line stays a refusal. The split is by the script's own wording
    (``MemAvailable``), not by guess: a ``REFUSED:`` line that does not name it is ``other``.
    """
    ram: List[str] = []
    other: List[str] = []
    for ln in text.splitlines():
        if not ln.startswith("REFUSED:"):
            continue
        (ram if "MemAvailable" in ln else other).append(ln.strip())
    return ram, other

def store_extra_config(max_size_bytes: int, min_free_bytes: int) -> str:
    """The backend's own knobs, in BYTES, with no unit rounding in between.

    ``max_size_scope: shared`` is kept from the tmpfs form and is what makes
    this a DIRECTORY cap: every writer rank builds its own evictor over the
    same directory, and ``LRUFileEvictor._load_config`` divides a shared cap by
    the writer count so the sum of the ranks' caps is the number passed here.
    """
    return json.dumps(
        {
            "max_size": str(int(max_size_bytes)),
            "min_free_space": str(int(min_free_bytes)),
            "max_size_scope": "shared",
        },
        separators=(",", ":"),
    )

def read_arc_state(
    param_path: str = "/sys/module/zfs/parameters/zfs_arc_max",
    kstat_path: str = "/proc/spl/kstat/zfs/arcstats",
) -> Dict[str, Optional[int]]:
    """ZFS ARC facts in BYTES: ``arc_max`` (the module parameter), ``c_max``
    and ``size`` (live, from arcstats).  ``None`` -- never 0 -- for anything
    unreadable, because 0 is a MEANINGFUL value of ``zfs_arc_max`` (it means
    "unset, use the built-in default") and must not be forged by an absence.
    """
    out: Dict[str, Optional[int]] = {"arc_max": None, "c_max": None, "size": None}
    try:
        with open(param_path) as f:
            out["arc_max"] = int(f.read().strip())
    except (OSError, ValueError):
        pass
    try:
        with open(kstat_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3 and parts[0] in ("c_max", "size"):
                    out[parts[0]] = int(parts[2])
    except (OSError, ValueError):
        pass
    return out

def _profile_early_read(profile: Optional[str]) -> bool:
    """Does ``profile`` put the #1235 early-read facts' FLAG half in the argv
    (ModelProfile.early_read_flags; 27B yes, NF no)? Unknown profile: no."""
    row = pdflip_form.profile_row(profile)
    return bool(row is not None and row.early_read_flags)

def _refuse_if_extra_raises_budget(
    budgets: List[int], extra: List[str], group: str
) -> None:
    """W100: ``--extra-*`` darf ein Kartenbudget SENKEN, nie HEBEN.

    WARUM DAS EIN EIGENER RIEGEL IST (fnFL2w51, 22.09.). Der Launcher
    rechnet je Karte ``total - corridor - dormant_residue`` und schreibt das
    Ergebnis als ``--rank-gpu-memory-mib`` in die argv. Danach haengt er
    ``extra`` an -- und wenn der Arm dort dieselbe Flag noch einmal setzt,
    gewinnt sie, weil argparse den LETZTEN Wert nimmt. Still, ohne eine
    Zeile.

    In w51 stand im argv von D zweimal ``--rank-gpu-memory-mib``:

        28240,16672,16672   der Launcher (Korridor, predicted_free -230)
        29900,18500,18500   der Arm (Handzahl aus v70-v75)

    Der Arm gewann, D-TP0 lud das Zielmodell bis auf 1,30 GB avail, und der
    SOLO-Draft danach fand keinen Platz mehr:

        PDFLIP-TAG-POOL tag=weights_draft pool=new
        [torch_memory_saver.cpp] CUresult error: 2 (out of memory)
                                 func=cu_mem_create line=194

    Die Richtung ist das Entscheidende, nicht die Doppelung: ein Arm, der
    ein Budget SENKT, nimmt sich Luft und das ist immer sicher. Ein Arm, der
    es HEBT, ueberzieht eine Karte, die der Launcher bereits bis an ihre
    Grenze gerechnet hat -- und der Posten, der dann faellt, ist der
    letztgeladene (hier der Draft), nicht der, den jemand gewaehlt haette.

    Kein Opt-out: ist der Launcher-Wert zu niedrig, gehoert SEINE Rechnung
    korrigiert ([[planner-alleinzustaendigkeit-vram]]), nicht ueberstimmt.
    """
    if not extra:
        return
    _values: Optional[List[int]] = None
    for _i, _tok in enumerate(extra):
        if _tok == "--rank-gpu-memory-mib" and _i + 1 < len(extra):
            _roh = extra[_i + 1]
        elif _tok.startswith("--rank-gpu-memory-mib="):
            _roh = _tok.split("=", 1)[1]
        else:
            continue
        try:
            _values = [int(x) for x in str(_roh).strip().strip('"\'').split(",") if x != ""]
        except ValueError:
            # Unlesbar ist NICHT harmlos: dieselbe Flag steht dann trotzdem
            # hinter der des Launchers und gewinnt. Benennen statt schweigen.
            raise RuntimeError(
                f"W100 PdFlipExtraBudgetRaise: --extra-{group.lower()} setzt "
                f"--rank-gpu-memory-mib auf {_roh!r}, das nicht als "
                f"MiB-Liste lesbar ist -- es ueberschreibt trotzdem die "
                f"Launcher-Budgets {budgets}."
            )
    if _values is None:
        return
    if len(_values) != len(budgets):
        raise RuntimeError(
            f"W100 PdFlipExtraBudgetRaise: --extra-{group.lower()} setzt "
            f"--rank-gpu-memory-mib mit {len(_values)} Eintraegen "
            f"({_values}), der Launcher rechnet mit {len(budgets)} "
            f"({budgets}). Die Listen muessen dieselbe Flotte meinen."
        )
    _hoch = [
        (i, int(b), int(e)) for i, (b, e) in enumerate(zip(budgets, _values)) if e > b
    ]
    if _hoch:
        _text = "; ".join(
            f"Rang {i}: Launcher {b} MiB -> extra {e} MiB (+{e - b})"
            for i, b, e in _hoch
        )
        raise RuntimeError(
            f"W100 PdFlipExtraBudgetRaise: --extra-{group.lower()} HEBT "
            f"Kartenbudgets ueber die Launcher-Rechnung -- {_text}. Der "
            f"Launcher-Wert ist total minus Korridor minus dormantes "
            f"Residuum; darueber ueberzieht die Karte und der zuletzt "
            f"geladene Posten faellt (fnFL2w51: der Solo-Draft, "
            f"cu_mem_create CUresult 2). Senken ist erlaubt, heben nicht: "
            f"ist die Launcher-Rechnung zu tight, gehoert SIE korrigiert."
        )

def _refuse_if_extra_drops_transient_lmo(vision: str, extra, group: str) -> None:
    """EXTRA is appended LAST and argparse keeps the last occurrence of
    ``--json-model-override-args``, so an EXTRA override is the one the group
    gets. Under ``--pdflip-vision transient`` it must carry
    ``language_model_only`` itself, or the group would build the tower it must
    not have -- refused by name instead of lost silently (27B f09dc0d6e4,
    extended to P on NF by H125: the NF arm's EXTRA_P and EXTRA_D both carry
    ``{"language_model_only":true}``, which passes)."""
    _extra = list(extra or ())
    if vision != VISION_TRANSIENT or "--json-model-override-args" not in _extra:
        return
    _i = len(_extra) - 1 - _extra[::-1].index("--json-model-override-args")
    try:
        _lmo = bool(json.loads(_extra[_i + 1]).get("language_model_only"))
    except (IndexError, ValueError, AttributeError):
        _lmo = False
    if not _lmo:
        raise PdFlipLaunchRefused(
            f"W111 PdFlipVisionArmRefused: --extra-{group.lower()} carries its own "
            "--json-model-override-args without language_model_only, and "
            "argparse keeps only the last one -- --pdflip-vision transient's "
            f"{VISION_TRANSIENT_OVERRIDE} would be lost and {group} would build a "
            'tower. Add "language_model_only": true to the EXTRA override.'
        )

def _refuse_if_transient_mm_hash_unstable(vision: str, env, group: str) -> None:
    """H125d: under ``--pdflip-vision transient`` an image's radix/L3 key is its
    ``pad_value`` = ``MM_PAD_SHIFT_VALUE + hash``, computed from the pixel
    values in P's AND in D's tokenizer (``MultimodalDataItem.set_pad_value``).
    D reads P's pages only if both hashes agree. ``FLLIPER_MM_SKIP_COMPUTE_HASH``
    replaces the content hash with ``uuid4()`` -- a different key per process,
    so D never matches the image and refuses it (W123) on every request.
    Refused by name at the launch instead."""
    if vision != VISION_TRANSIENT:
        return
    raw = str((env or {}).get(MM_SKIP_HASH_ENV, "") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        raise PdFlipLaunchRefused(
            f"W111 PdFlipVisionArmRefused: {MM_SKIP_HASH_ENV}={raw} in group {group}'s "
            "environment -- --pdflip-vision transient keys an image by its content "
            "hash on P and D; a random per-process hash makes D's image prefix "
            "unmatchable (W123 on every image). Unset it for this boot."
        )

def _split_decisions(
    card_uuids: Sequence[str],
    ratios: Optional[Dict[str, float]] = None,
    leg_form: str = "interleave",
    override: Optional[bool] = None,
) -> Dict[str, bool]:
    """Per card: does its PCIe key SPLIT the two directions?  Decided ONCE, here.

    FIX 1 round 1, and boot weg2rg2 is the reason it is a decision at all rather
    than a reading of R17's ratio.  R17 asks whether the split BUYS enough
    (concurrent aggregate >= 1.5x serial); weg2rg2 asked whether the single key
    is SAFE under C9's gathered legs, and the answer is no.  Each leg holds the
    per-card key across its whole tag loop, so on a one-key card the pair is
    serialised again and R5's premise -- W's per-tag releases fund S's acquires
    -- is false there: the sleep leg blocks in ``acquire`` at free = 0 holding
    the key the wake leg needs.  Measured cost of that on nvml0: 120.189 s, W31,
    W29 on three ranks, group-fatal W4.

    So: under GATHERED legs every card splits, whatever its ratio.  That is not
    free-lunch reasoning -- the x4 card's 1.316 is concurrent OVER serial, so
    concurrency is 32 % faster there too; R17's gate only ever said the split is
    not worth a mechanism of its own, and A1-4 says that card gets no BENEFIT
    from the split, not that it must not have one.  Under a front that does NOT
    gather (no state of this tree), the ratio gate stands and a card below it
    keeps one key -- which is correct there, because a serial front never has
    two legs in flight to deadlock.

    ``override`` is the test injection point: ``True`` = force every key to
    split, ``False`` = force every key single (the weg2rg2 state).
    """
    if override is not None:
        return {uuid: bool(override) for uuid in card_uuids}
    if leg_form == "interleave":
        return {uuid: True for uuid in card_uuids}
    table = ratios or {}
    return {
        uuid: table.get(uuid) is not None and table[uuid] >= DUPLEX_GATE
        for uuid in card_uuids
    }

def _resolve_keys_in_rank_tree(
    card_uuids: Sequence[str], published: str, py: str, tree: str
) -> Tuple[Dict[str, Tuple[int, int]], str]:
    """``{uuid: (keys THAT tree builds, directions THAT tree has)}``, or ``({}, why)``.

    BOTH halves come from the rank tree, never one from here: a tree with a
    different :data:`PCIE_DIRECTIONS` would otherwise be compared against this
    launcher's count and read as split when it is not.

    One interpreter start per boot, in the environment ``build_env`` gives the
    ranks (``PYTHONPATH=<tree>/python``), because a stale tree on that path is
    precisely the divergence this check exists to see and precisely the one an
    in-process import cannot show.
    """
    # Rename 1b: one spelling per variable before the pop below (name_compat).
    env = canonical_env(dict(os.environ))
    env["PYTHONPATH"] = f"{tree}/python"
    env["CUDA_VISIBLE_DEVICES"] = ""
    from flliper.srt.managers import pdflip_memory_saver as _s

    env.pop(_s.PCIE_DUPLEX_ENV, None)  # the probe sets the TREE's own name
    try:
        proc = subprocess.run(
            [py, "-c", _KEY_PROBE, published, json.dumps(list(card_uuids))],
            capture_output=True, text=True, env=env, cwd=tree, timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    out = proc.stdout or ""
    marker = "PDFLIP-KEYS "
    if proc.returncode != 0 or marker not in out:
        tail = " | ".join((proc.stderr or "").strip().splitlines()[-3:])
        return {}, f"rc={proc.returncode} stderr={tail[:400]!r}"
    try:
        raw = json.loads(out[out.index(marker) + len(marker):].strip())
        return {str(k): (int(v[0]), int(v[1])) for k, v in raw.items()}, ""
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return {}, f"unparsable probe answer ({type(exc).__name__}: {exc})"

def _split_fraction_text(text) -> list:
    """Ein Fraction-Flag des LAUNCHERS als Liste (#130).

    `--pp-cut-expert-device-fraction 0.367,0.75,0.95` kommt als String am
    Namespace an, nicht als argv-Fragment -- `_argv_vector` findet es
    dort nie. Leerer/unlesbarer Wert -> leere Liste, der Aufrufer faellt
    dann auf seine zweite Quelle zurueck.
    """
    t = str(text or "").strip()
    if not t:
        return []
    out = []
    for piece in t.replace(";", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            out.append(float(piece))
        except ValueError:
            return []
    return out

def _rope_cache_cols(cfg: dict) -> int:
    """Columns of the cos/sin cache ``get_rope`` builds for this config: the
    rotary dim = head_dim x partial_rotary_factor (text sub-config first, the
    v5 ``rope_parameters`` or the top level for the factor). 0 = unreadable."""
    text = cfg.get("text_config") or cfg
    head_dim = text.get("head_dim") or (
        int(text["hidden_size"]) // int(text["num_attention_heads"])
        if text.get("hidden_size") and text.get("num_attention_heads") else 0)
    rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
    factor = rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 1.0))
    try:
        return int(int(head_dim) * float(factor if factor is not None else 1.0))
    except (TypeError, ValueError):
        return 0

def refuse_unless_same_form_source(
    table: Optional[ring_table.RingTable], weight_source: str
) -> Optional[str]:
    """Under an armed arm, a table solved from ANOTHER form is a refusal.

    The token above stops #1305 item 4 from excluding this form's own boots --
    and by doing so it also removes the thing that used to make the absence
    LOUD.  With the token on, ``solve`` simply falls through to the form-DISTANCE
    ranking and returns the closest twin, silently, which is the foreign-split
    defect one layer down: the ring's residual, the ledger's charge and (via
    ``pdflip/xchg_census.py``) group P's per-card placement would all be a
    measurement of a form this boot does not run.

    So the launcher refuses, and PRINTS THE CLOSEST TWIN IT REJECTED -- the
    selected boot IS the closest twin, because the distance ranking put it
    first -- together with both form keys and the term-level difference, so the
    next reader knows which boot to produce rather than which number to lower.

    THREE NON-CASES, each deliberate:
      * ``table is None``: ``prepare_host_ring`` has already refused (R22/W20)
        for a reason of its own; a second refusal here would bury it.
      * the DEFAULT arm: a serving boot has no same-form requirement -- that is
        #1305 item 4's whole subject, and acceptance (iii) is that its
        selection does not move.
      * ``form_same is None``: the form gate never ran, and an unarmed gate is
        never read as a passed one (``solve``'s own words).  It refuses.
    """
    if table is None or str(weight_source) not in WEIGHT_SOURCE_ARMED:
        return None
    if getattr(table, "form_same", None) is True:
        return None
    unarmed = getattr(table, "form_same", None) is None
    # #1367 THE FOURTH NON-CASE: THE FIRST ARMED BOOT OF A FORM.
    # The armed arm puts XCHG_FORM_TOKEN into group P's argv and the token is
    # IN the form key, so an armed boot's key can never equal an unarmed
    # boot's. With no same-form candidate in existence, the sentence this
    # refusal ended with -- "Boot the arm once to produce the first same-form
    # source" -- names the boot it is refusing: a remedy that cannot be
    # carried out. Measured: xsn30 dry run rc=2 on all three groups, armed key
    # 0a15b55459fb against the unarmed 2b66740bedf9 of xsn29.
    #
    # The operator's ruling (2026-09-13) is narrow and is NOT a rebuild of the
    # key: the exchange arm is a different weight statement and keeps its own
    # key. Absence of a same-form source is the BOOTSTRAP STATE -- the same
    # reading #1362 gave the model digest and the lane cut -- and it takes the
    # THIRD W48 shape that already exists for the default arm: the source's
    # weight statement is discarded and re-derived from THIS boot's argv
    # (ring_table.py:2737), so only the non-weight residual is carried.
    #
    # WHAT IS NOT RELAXED: a same-form source that EXISTS and did not win is a
    # real mismatch and still stops the boot below -- which is why this reads
    # the inventory `solve` ranked, and not merely `form_same`.
    same_form_sources = tuple(getattr(table, "same_form_candidates", ()) or ())
    # #1374 W9: PRESENT-BUT-UNUSABLE IS NOT A WITNESS. `same_form_candidates`
    # now carries only the stems that survived `solve`'s own checks; the ones it
    # ranked and then eliminated come with their reason, and they are NAMED in
    # the bootstrap line so the zero above is explained rather than merely
    # asserted. Boot weg2xsn31 is what this costs otherwise: weg2xsn30 died at
    # wall 8 before D ever slept, so it is same-form (0a15b55459fb) and carries
    # `no sleep-pass lines for D` -- counted as a witness, it refused its own
    # successor, and `--ring-table-boot` could not help because the form gate
    # runs after the stem choice and meets the same missing D lines. Every boot
    # that dies at a wall would lock the next one out.
    unusable = tuple(getattr(table, "same_form_unusable", ()) or ())
    if not unarmed and not same_form_sources:
        _excluded = ("; same-form stems present but unusable: "
                     + ", ".join(f"{st} reason={why}" for st, why in unusable)
                     if unusable else "")
        return (
            f"PDFLIP-RING W48 BOOTSTRAP armed first boot -- form key "
            f"{table.form_key} has NO boot yet, so there is no same-form source "
            f"to be missing. The closest twin {table.boot} (form key "
            f"{table.source_form_key}) is carried for its NON-WEIGHT residual "
            f"only; this boot's weight statement is source=derived, re-derived "
            f"from its own argv, and the exchange region's residual comes from "
            f"the published terms rather than from a foreign form. The next "
            f"armed boot of this form finds this one and is no longer a "
            f"bootstrap. A USABLE same-form source that disagrees still "
            f"refuses: {table.form_diff}{_excluded}"
        )
    raise PdFlipXchgFormSourceMissing(
        f"W48 PdFlipRingFormMismatch: --pdflip-weight-source {weight_source} arms the "
        f"exchange, so this boot's group-P form carries {ring_table.XCHG_FORM_TOKEN} "
        f"and its ring table must be solved from a boot of the SAME form.  "
        + ("The form gate did not run at all, so no source can be called same-form. "
           if unarmed else
           f"The closest twin the ranking could offer is boot {table.boot}, whose "
           f"form key is {table.source_form_key} against this boot's "
           f"{table.form_key}: {table.form_diff}.  ")
        + (f"A USABLE same-form source EXISTS and did not win, so this is a "
           f"real mismatch and not a first boot: "
           f"{', '.join(same_form_sources)}.  "
           if same_form_sources else "")
        + (f"(Same-form stems that were ranked and then ELIMINATED do not "
           f"count as witnesses and are not the reason for this refusal: "
           f"{', '.join(f'{st} reason={why}' for st, why in unusable)}.)  "
           if unusable else "")
        + "There is NO fallback: a chunk tag is a LAYER BAND, so a source of "
        "another PP form states group P's bytes on the wrong cards, and the "
        "ring's residual, the ledger's charge and the exchange census would "
        "each be a measurement of a form this boot does not run (MEASURED "
        "2026-09-11: source split [42, 11, 11] against this form's "
        "[39, 13, 12]).  Boot the arm once to produce the first same-form "
        "source, or pin one with --ring-table-boot."
    )

def set_xchg_census_map(text: str) -> None:
    _XCHG_CENSUS_MAP[0] = str(text or "")

def refuse_double_coverage_arm(env, *, xchg_coverage_diff: bool) -> None:
    """PRE-SPAWN refusal when both coverage instruments would be armed.

    `lane_coverage.arm` refuses by name when `FLLIPER_SEAM_COVERAGE_DIR` is
    set -- but it does so INSIDE each rank, after the spawn. A leftover value
    in the operator's shell therefore burned the whole window: six ranks
    refused, the boot ran with no instrument, and the refusal sat in the rank
    logs where the ingest report never looks (review S-3). The launcher pops
    its OWN variable and never that one, so nothing upstream catches it
    either.

    Measured reason this matters at all: a second `Coverage.start()` raises
    nothing and blinds the first collector for its duration, so "both armed"
    does not fail loudly on its own -- it produces two dumps that quietly
    describe something neither instrument claims.
    """
    if not xchg_coverage_diff:
        return
    other = env.get("FLLIPER_SEAM_COVERAGE_DIR")
    if other:
        raise SystemExit(
            "REFUSING TO BOOT: --xchg-coverage-diff was passed while "
            f"FLLIPER_SEAM_COVERAGE_DIR={other!r} is set in the environment. "
            "Two coverage tracers in one process measure neither -- the "
            "second start() silently blinds the first, and a blind window "
            "reads as a never-executed seam. Unset FLLIPER_SEAM_COVERAGE_DIR "
            "or drop --xchg-coverage-diff. Refused here, before the spawn, "
            "because the per-rank refusal costs the whole window."
        )

def served_model_name(model_path: str, override: str = "") -> str:
    """#1362: the name a boot serves under, DERIVED from the model it loads.

    ``--served-model-name`` was the literal ``Qwen3.8-27B`` for every boot this
    launcher ever made.  It is the name the front routes on, the load drivers
    address and every probe asserts against, so on the 4B-FP8 transition
    vehicle the boot would have answered under the 27B's name -- a silent
    identity swap in the one field whose whole job is identity.

    The basename of the checkpoint directory is the default; an explicit
    override wins and is the reason the flag exists (a HuggingFace snapshot
    path ends in a commit hash, which is a correct identity and a terrible
    name).
    """
    if override:
        return str(override)
    p = str(model_path or "").rstrip("/")
    base = os.path.basename(p)
    # A snapshot directory is named by its commit; walk up to the repo name,
    # which is what a human and a load driver both mean by "the model".
    if base and len(base) >= 32 and all(c in "0123456789abcdef" for c in base):
        parts = [x for x in p.split("/") if x]
        for cand in reversed(parts[:-1]):
            if cand not in ("snapshots", "blobs", "refs"):
                return cand.replace("models--", "").replace("--", "/")
    return base or "unknown-model"

def ring_absent_by_design(weight_source: str, inject_mode: str) -> bool:
    """Is there NO host weights ring in this boot -- by design, not by accident?

    B4f, and boot weg2xsn13's S6I shadow dry-run is why it is a function.  The
    predecessor read ring absence off the WEIGHT SOURCE alone
    (``weight_source == "exchange"``) and the dry-run stopped at rc=1 before a
    card was touched, inside the ledger::

        ValueError: ring_absent_by_design with a non-zero ring
        (ring_bytes=46133149696, ring_span1_bytes=44559237120)

    THE LEDGER WAS RIGHT.  Its guard exists for exactly that -- *"a
    contradiction resolved silently is how a charged term becomes invisible"* --
    and the same run had armed 21715+9758+12529 MiB = 42.97 GiB of ring and
    handed it those very bytes.

    RING ABSENCE IS A PROPERTY OF THE INJECT ARM, NOT OF THE WEIGHT SOURCE.
    #1327's comment ("under ``exchange`` the launcher publishes no
    ``TMS_HOST_RING_*``") is true only where the exchange is the AUTHORITY.
    Under ``--pdflip-xchg-inject shadow`` -- the DEFAULT, and the form the S6I
    order grades -- the refill is still the authority and the ring is still
    armed: that order's item (b) requires ``host_weights=`` UNCHANGED and calls
    a ``host_weights=0.00`` there a FAIL.  So ``exchange`` + ``shadow`` is an
    ordered, real form in which the ring is PRESENT, and it is precisely the
    combination the old predicate could not express.

    Hence the conjunction, in ONE named place so the declaration and the arm
    strings cannot be spelled differently at two call sites, and written as an
    equality against the recognised constants rather than a negation -- an
    unrecognised inject mode must not zero a charged ring, and
    ``weight_exchange.inject_mode`` already answers ``shadow`` for anything it
    does not recognise.
    """
    return (
        str(weight_source) == WEIGHT_SOURCE_EXCHANGE
        and str(inject_mode) == weight_exchange.INJECT_AUTHORITATIVE
    )

def pinned_reserve_line(ledger_gib, ledger_src: str = "") -> str:
    """29.09.: the PDFLIP-HOST PINNED-RESERVE line -- the reserve the ranks'
    pinned-host checks will ACTUALLY use (read back through
    pinned_host_reserve(), after the export) with its source: env (explicit
    FLLIPER_PINNED_HOST_RESERVE_GIB), ledger (the host ledger's measured
    margin) or native (10 GiB, no finite cgroup); plus the number the
    ledger's PINNED WALL was priced with, so a mismatch is visible."""
    from flliper.srt.mem_cache import pinned_host_budget as _phb

    gib, kind, detail = _phb.pinned_host_reserve_effective()
    wall = "none (no finite memory.max)" if ledger_gib is None else f"{float(ledger_gib):.2f} GiB ({ledger_src})"
    return (f"PDFLIP-HOST PINNED-RESERVE effective={gib:.2f} GiB source={kind} ({detail}) "
            f"ledger_wall_reserve={wall}")

def _proc_start_ticks(pid: int) -> int:
    """/proc/<pid>/stat field 22 (starttime), 0 when unreadable. Split after the
    LAST ')' -- comm may carry spaces and parentheses."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            rest = f.read().rsplit(b")", 1)[1].split()
        return int(rest[19])
    except (OSError, IndexError, ValueError):
        return 0

def _proc_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""

def set_group_env(spec: str, key: str, value: str) -> str:
    """``spec`` ('KEY=VAL;...') with ``key`` set to ``value`` (appended when
    absent); every other entry keeps its text and its place."""
    items = [x for x in str(spec or "").split(";") if x.strip()]
    done = False
    for i, item in enumerate(items):
        if "=" in item and item.split("=", 1)[0].strip() == key:
            items[i] = f"{key}={value}"
            done = True
    if not done:
        items.append(f"{key}={value}")
    return ";".join(items)

def _read_driver_text() -> Optional[str]:
    try:
        with open("/proc/driver/nvidia/version") as f:
            return f.read(4096)
    except OSError:
        return None

def _published_profile() -> Optional[str]:
    """The profile id of the pdflip form this launcher published (None = none)."""
    f = pdflip_form.current_form()
    return f.profile if f is not None else None

def read_pp_bubble(path: str) -> Optional[BubbleMeasurement]:
    """The BINDING stage's bubble totals from one group-P log, or None.

    The binding stage is the one with the largest forward total: under a
    pipelined prefill the makespan is that stage's time, so its idle is the
    idle that costs throughput. Returns None when the file does not exist or
    carries no PP-BUBBLE line -- absence of the instrument, never a measured
    zero (#892 / the indicator law: a tool says "I found nothing", never
    "there is nothing").
    """
    per_rank: Dict[int, List[float]] = {}
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                m = _BUBBLE_RE.search(line)
                if m is None:
                    continue
                acc = per_rank.setdefault(int(m.group("rank")), [0.0] * 4 + [0.0])
                acc[0] += float(m.group("forward"))
                acc[1] += float(m.group("bubble"))
                acc[2] += float(m.group("starved"))
                acc[3] += float(m.group("n"))
                acc[4] += 1.0
    except OSError:
        return None
    if not per_rank:
        return None
    rank = max(per_rank, key=lambda r: per_rank[r][0])
    fwd, bub, starved, n_fwd, windows = per_rank[rank]
    return BubbleMeasurement(
        source=path,
        rank=rank,
        windows=int(windows),
        forward_ms=fwd,
        bubble_ms=bub,
        starved_ms=starved,
        n_forwards=int(n_fwd),
    )

def refuse_inherited_layer_set(env: Mapping[str, str]) -> None:
    """W44: a layer set inherited from the environment is REFUSED, never used.

    ``build_env`` starts from ``os.environ``, so before this refusal an
    exported ``FLLIPER_PP_LAYER_SET`` reached group P's ranks without passing
    through the launcher at all: the solver would rank and print one layout
    while the boot ran another, and the PP-CUT provenance line would describe
    a layout that never existed. That is the #505(a) silent-substitution class
    with a whole stage map as the substituted object.

    ONE KNOB, ONE READER (upstream-minimal). The flag ``--pp-layer-set`` is the
    only way to configure the map, and the launcher is the only writer of the
    variable. Not a twin of the env: its REPLACEMENT as the interface, with the
    variable demoted to the wire that carries the decision to the ranks.
    """
    raw = str(env.get(PP_LAYER_SET_ENV, "") or "").strip()
    if not raw:
        return
    raise PdFlipLaunchRefused(
        "W44 PdFlipLayerSetEnvRefused: %s=%r is set in this launcher's own "
        "environment. It is no longer an input: --pp-layer-set is, and the "
        "launcher is the only writer of the variable (it carries the solved "
        "or pinned map to group P's ranks). Honouring an inherited value "
        "would let the boot run a stage map the PP-CUT provenance line does "
        "not describe. Unset it and pass --pp-layer-set %s instead."
        % (PP_LAYER_SET_ENV, raw, raw)
    )

def read_mean_prefill_prefix(path: str) -> Optional[Tuple[float, int, str]]:
    """Mean PREFIX DEPTH over one boot's prefill census. ``(mean, n, rank)``.

    THE PREFIX IS ACCUMULATED, NOT READ. No line states it: a ``Prefill batch``
    line carries ``#new-token`` (this chunk) and ``#cached-token`` (the
    prefix-cache hit), never "how much of this request is already computed".
    But group P admits ONE chunked request per pass, so a request is a maximal
    run of consecutive lines on one rank and the prefix at chunk *j* is the sum
    of ``#new-token`` over chunks *0..j-1* plus that run's cache hit. A chunk
    SHORTER than the run's own maximum ends the request -- the last chunk of a
    prompt, or the zero-remainder 1-token end anchor -- and the accumulator
    resets.

    ONE RANK ONLY, and it is named in the answer: every stage prints the same
    batch, so summing across ranks would triple the denominator without adding
    a single measurement.

    ``None`` when the file does not exist or carries no census -- absence of
    the instrument, never a measured zero (#892).
    """
    rows: Dict[str, List[Tuple[int, int]]] = {}
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                m = _PREFILL_RE.search(line)
                if m is None:
                    continue
                rows.setdefault(m.group("rank"), []).append(
                    (int(m.group("new")), int(m.group("cached")))
                )
    except OSError:
        return None
    if not rows:
        return None
    rank = sorted(rows)[0]
    seq = rows[rank]
    full = max(n for n, _ in seq)
    prefixes: List[int] = []
    acc = 0
    for new, cached in seq:
        prefixes.append(acc + cached)
        if new < full:
            acc = 0
        else:
            acc += new
    return (sum(prefixes) / float(len(prefixes)), len(prefixes), rank)

@functools.lru_cache(maxsize=1)
def _running_power_by_class():
    return pdflip_form.running_power_by_class()

def _profile_row_or_27b(profile: Optional[str]):
    return pdflip_form.profile_row(profile) or _PROFILE_27B

def _row_format_default(profile, ns, row_value, formats, code_default):
    """A best-form value the row measured on some checkpoint FORMATS only: the
    row's value when the booted checkpoint's registry format
    (pdflip_form.format_of, by calibration identity) is one of them, else the
    code default."""
    fmt = pdflip_form.format_of(profile, str(getattr(ns, "model", "") or ""))
    return row_value if fmt and fmt in formats else code_default

def profile_arg_defaults_line(ns, changed: Sequence[str]) -> Optional[str]:
    """The launcher's one line naming every CLI default the registry row set
    (None when the row changed nothing)."""
    if not changed:
        return None
    parts = ", ".join(f"{d}={getattr(ns, d, None)!r}" for d in changed)
    return (f"PDFLIP-PROFILE-ARG-DEFAULTS (registry row {getattr(ns, 'profile', '') or PROFILE_QWEN27B}; "
            f"unset flags only, a given flag wins): {parts}")

def read_card_power_record(front_log: str) -> Optional[List[Tuple[str, Optional[float]]]]:
    """``[(card name, limit W or None), ...]`` in stage order from a boot's
    front log, or ``None`` when the boot printed no record (every boot before
    this line existed)."""
    try:
        with open(front_log, "r", errors="replace") as fh:
            for line in fh:
                if CARD_POWER_MARKER not in line:
                    continue
                rows = sorted(
                    (int(s), name, None if lim == "unknown" else float(lim))
                    for s, name, _uuid, lim in _CARD_POWER_RE.findall(line)
                )
                if rows:
                    return [(name, lim) for _s, name, lim in rows]
    except OSError:
        return None
    return None

def shipped_line(
    decision,
    chosen,
    ship_why,
    incumbent_row,
    makespan_row,
    *,
    incumbent_fallback: str,
    floor_source: str,
) -> str:
    """THE ``PP-CUT SHIPPED`` line. Pure, so it can be RENDERED by a test.

    Split out for the same reason :func:`incumbent_candidate` was: this line is
    the boot's whole statement of which priced row it pays for, and until
    #1286b the only proof it still rendered was a boot. It carries twelve
    substitutions across a %-format, which is exactly the failure class an edit
    to it produces -- a wrong arity raises TypeError at emit time, i.e. after
    the weights are loaded, i.e. a spent window. Now a desk test builds a
    decision and renders it.

    ``incumbent_fallback`` is passed IN rather than built from
    ``P_PP_STAGE_RATIO_SCORES`` here, and that is not style. #1233 pins the
    census of functions that read the incumbent score vector, precisely because
    an extra reader is how a constant drifts back into the argv path; this
    formatter has no business joining that list, so the one string it would
    need is handed to it by the caller that legitimately holds the vector.

    #1286b renamed ``pool_tokens=`` to ``chosen_pool=`` and added
    ``pool_floor=``: a pool figure whose constraint is not printed beside it is
    the shape that let weg2sb5f's +64.1 %% sit in plain sight across two logs.
    ``pool_tokens=`` survives on the ``PP-CUT solver:`` table rows and on
    ``PP-POOL-JOIN``, and no parser on this rig reads it off THIS line.

    ``floor_source`` is REQUIRED and keyword-only (2026-09-09) for the same
    reason ``pool_floor=`` was added beside ``chosen_pool=``: once a floor has
    a SHIPPED DEFAULT, the number alone no longer says whether the boot was
    obeying a standing user order or an operator's one-off ``--pp-solve-pool-
    floor``.  Those two produce the same ``pool_floor=578199`` and want
    opposite responses from a reader, so the line prints which.  No default
    value here on purpose: a caller that forgets it must fail loudly at the
    call site rather than quietly publish an unattributed floor.
    """
    return (
        "PP-CUT SHIPPED: layers=%s attn=%s chosen_pool=%d pool_floor=%s "
        "pool_floor_source=%s makespan_ms=%.1f -- %s. "
        "The two objectives this boot did NOT take stay priced beside it and are "
        "therefore not paid by accident: incumbent %s pool %s makespan %s, "
        "pool-maximal (kv-floor) %s pool %d makespan %.1f, makespan-optimal %s "
        "pool %d makespan %.1f (--pp-solve-objective incumbent|maxkv|makespan "
        "ships them)."
        % (
            ",".join(str(n) for n in chosen.layers),
            ",".join(str(a) for a in chosen.attn),
            int(chosen.pool_tokens),
            "none" if decision.pool_floor is None else str(int(decision.pool_floor)),
            floor_source,
            chosen.makespan_ms,
            ship_why,
            incumbent_row.fmt() if incumbent_row is not None else incumbent_fallback,
            "%d" % int(incumbent_row.pool_tokens) if incumbent_row is not None
            else "n/a (NOT RANKED by this solve)",
            "%.1f" % incumbent_row.makespan_ms if incumbent_row is not None
            else "n/a",
            decision.kv_floor.fmt(),
            int(decision.kv_floor.pool_tokens),
            decision.kv_floor.makespan_ms,
            makespan_row.fmt(),
            int(makespan_row.pool_tokens),
            makespan_row.makespan_ms,
        )
    )

def publish_store_identity(model: str, map_path: str, log) -> str:
    """H2c: die Identitaet des geteilten Expertenstores fuer diesen Boot.

    Ein Sentinel ``*.written.json`` sagte bis hier nur "diese Plaetze sind
    geschrieben" -- nicht mit welchem Checkpoint und welcher Slot-Karte. Der
    Store ueberlebt den Prozess; ein Boot mit anderem Checkpoint oder anderen
    FR/``--rank-moe-ratio`` haette mit D-Store-Adopt (H2) fremde Bytes als
    eigene Experten gelesen. Deshalb war H2 im Profil aus
    (ENTSCHEIDUNGEN-0927, 28.09.: "Expertenstore-Identitaet ist VORBEDINGUNG").

    Ohne Karte ``""``: dann gibt es keine Slot-Zuordnung, und store_adopt
    bleibt inaktiv. Ein Fehler beim Rechnen ist ebenfalls ``""`` -- das ist
    die sichere Richtung (D liest vom Checkpoint), nie eine halbe Identitaet.
    """
    if not map_path:
        log("H2c STORE-IDENTITY ENTFAELLT: no experts card -> D-Store-Adopt inactive")
        return ""
    from flliper.srt.layers.moe import expert_store as _es

    try:
        ident = _es.compute_identity(model, map_path)
    except OSError as exc:
        log("H2c STORE-IDENTITY ENTFAELLT: %s: %s -> D-Store-Adopt inactive"
            % (type(exc).__name__, exc))
        return ""
    log("H2c STORE-IDENTITY id=%s model=%s map=%s (beide Gruppen; ein Sentinel "
        "ohne diese Identitaet belegt nichts)" % (ident or "-", model, map_path))
    return ident

def _same_vector(a, b) -> bool:
    try:
        fa, fb = [float(x) for x in a], [float(x) for x in b]
    except (TypeError, ValueError):
        return False
    return len(fa) == len(fb) and all(abs(x - y) <= 1e-9 for x, y in zip(fa, fb))

def repoint_store_geometry_at_pinned_form(xchg_env: Dict[str, str], form: dict,
                                          total: int, log) -> None:
    """#239 rc12z29c-Blocker 2: the geometry published from --extra-d BEFORE
    the pre-map D solve (#106 STORE-GEOMETRY, #134 bands, the V1 map) carries
    the stated vectors -- rc12z29c's D log still printed '#96 STORE-SLOTS
    rank=2 ... ratios=[183, 137, 168] fracs=[0.06, 0.51, 0.48]'. With the
    Platztausch map that line is only a diagnostic (the map is the slot
    authority), but a run whose map fails falls back to exactly these values.
    So every geometry published from the vectors follows the pinned form."""
    from flliper.srt.layers.moe import expert_map as _em

    ratios, fracs = list(form["ratios"]), list(form["fractions"])
    if "FLLIPER_MOE_EXPERT_STORE_GEOMETRY" in xchg_env:
        was = xchg_env["FLLIPER_MOE_EXPERT_STORE_GEOMETRY"]
        xchg_env["FLLIPER_MOE_EXPERT_STORE_GEOMETRY"] = f"{','.join(ratios)}|{','.join(fracs)}"
        log(f"PDFLIP-STORE-GEOMETRY shared={xchg_env['FLLIPER_MOE_EXPERT_STORE_GEOMETRY']} "
            f"source=KARTE-FORM (#239 rc12z29c; vorher {was} aus dem genannten Vektor)")
    if "FLLIPER_PDFLIP_EXPERT_BAND_SIZE" in xchg_env:
        _bs, _bc = _em.band_geometry([int(float(x)) for x in ratios], int(total))
        if _bs > 0:
            xchg_env["FLLIPER_PDFLIP_EXPERT_BAND_SIZE"] = str(_bs)
            xchg_env["FLLIPER_PDFLIP_EXPERT_BANDS"] = str(_bc)
        else:
            xchg_env.pop("FLLIPER_PDFLIP_EXPERT_BAND_SIZE", None)
            xchg_env.pop("FLLIPER_PDFLIP_EXPERT_BANDS", None)
        log(f"PDFLIP-EXPERT-BAND size={_bs} count={_bc} from the KARTE-FORM (#239 rc12z29c)")
    if xchg_env.pop(_em.MAP_ENV, None) is not None:
        log("PDFLIP-EXPERT-MAP (V1, aus dem genannten Vektor) rejected (#239 rc12z29c): "
            "die Karte kommt aus publish_expert_map und der gepinnten Form")

def _refuse_unbuilt_platztausch_buffers(emap: dict, *,
                                        chunk_layers: Optional[int]) -> None:
    """W120 VOR dem ersten Byte: die Karte gibt einem Rang einen
    Platztausch-Puffer, den er beim Laden nicht baut.

    fnFL2x100 (FR_P 0.45/0.95/1.0): P-Stufe 2 hielt alle 512 Experten, baute
    darum keinen Puffer und publizierte fuer Layer 40-47 den nackten Stapel;
    der Join liess D's 32 Puffer (3,69 GB) als "destination-only" fallen, und
    D TP1 starb im ersten Schlaf-Leg an W106 auf weights_14 -- nach 5 Minuten
    Laden, READY und Kontroll-Decode. Das ist aus der Karte allein ablesbar,
    also refused der Dry-Run es, mit Stufe/Rang, Layern und Tags.
    """
    from flliper.srt.layers.moe import expert_map as _em

    unbuilt = _em.unbuilt_platztausch_buffers(emap)
    if not unbuilt:
        return
    row_list = []
    for u in unbuilt:
        wer = (f"P-Stufe {u.index}" if u.phase == _em.PHASE_PP
               else f"D-Rang {u.index}")
        tags = ("?" if not chunk_layers else ",".join(
            f"weights_{t}" for t in sorted({l // int(chunk_layers)
                                             for l in u.layers})))
        row_list.append(
            f"{wer}: {u.resident} von {u.experts} Experten resident -> kein "
            f"Puffer (Layer {u.layers[0] if u.layers else '?'}-"
            f"{u.layers[-1] if u.layers else '?'}, Tags {tags}); groesste "
            f"Fraction mit Puffer {u.max_fraction:.3f}")
    raise PdFlipLaunchRefused(
        "W120 PdFlipPlatztauschBufferUnbuilt: " + " | ".join(row_list)
        + ". Ein Rang mit weniger als 2 Scratch-Zeilen baut keinen "
        "Platztausch-Puffer (plan_load_time_staging: R >= E; scratch_slot_count: "
        "E-R < 2) und publiziert den nackten Expertenstapel unter einem anderen "
        "Namen -- der Flip-Join findet fuer diese Layer kein Gegenstueck "
        "(x100: W106 auf D TP1 weights_14, 120 s spaeter W29, Gruppe tot). "
        "Bei der genannten Fraction umfasst der Puffer weiterhin jede Zeile.")

def publish_p_fractions(ns, fracs: Sequence[float]) -> None:
    """FR_P an die DREI Stellen, an denen es lebt (arm_fnFL2.sh: "FR_P steht
    an DREI Stellen"; dieselben wie ``apply_p_draft_post``)."""
    from flliper.srt.pdflip import draft_post as _dp

    ns.pp_cut_expert_device_fraction = ",".join(_dp._fmt(x) for x in fracs)
    ns.extra_p = _dp.replace_vector_flag(ns.extra_p, "--rank-moe-resident-fraction", fracs)
    ns.env_p = _dp.replace_env_vector(getattr(ns, "env_p", ""),
                                      "FLLIPER_MOE_RESIDENT_EXPERT_FRACTION", fracs)

def standard_form_resolved(ns) -> Tuple[bool, str]:
    if pdflip_form.profile_row(getattr(ns, "profile", None)) is None and not str(
            os.environ.get(pdflip_form.STANDARD_FORM_ENV, "")).strip():
        return False, "unknown profile"
    return pdflip_form.standard_form_state(os.environ, getattr(ns, "profile", None))
