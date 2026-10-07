"""AP-E: the DUAL form of ``propose()`` (plan PLAN-PROFIL-PLANER-1006 section 3 row AP-E, R3).

DUAL = group P (PP<N>) and group D (TP<N>) awake AT THE SAME TIME on the SAME cards (``launcher.resolve_dual_layout``,
``--dual-share``): D boots as the union owner of every card, P holds only the part of its stage that is not D's shard, one KV pool
per card is shared (``--dual-unified-kv``).  Tree: 27B only (R2).

``propose(..., form="dual")`` runs the common stage-A machinery of ``propose.py`` (card order, per-class measured values, scalars)
and then calls :func:`apply_dual`, which owns what is Dual-specific:

* **P budget per card** (``--extra-p ... --rank-gpu-memory-mib``): what group P may use on the card.  D is sized from it
  (``launcher.dual_share_planned_dc``: ``D budget = card - P budget - --dual-p-overhead-mib - awake rest``), so every MiB of the
  P budget is a MiB less for D.  The rule is a SHIFT from the profile's own calibration (the "dual1h method" of
  ``profiles/27b-nvfp4-dual1h.env``): ``budget = model terms(new layout) + residual`` where the residual is what the profile's own
  budget holds beyond the same model terms at the profile's layout (graphs, activations, slack: not derivable, only measured).
  Where a card has no twin class in the profile the residual is 0 and the value is ``unbelegt``.
* **P layer cut** (``--pp-stage-ratio`` / ``--pp-attn-stage-ratio``): carried, or searched against the KV obligation (below).
* **KV obligation (Pflichtwert of the Dual form, plan 4c)**: P must carry ``kv_tokens`` (default 262144) of prefill context:
  ``--max-kv-per-request = kv_tokens`` (P-COVERS-D-SESSION), ``--dual-p-kv-max-tokens = round_up(kv_tokens + 1 page, 4096)`` (the
  level ``dual_p_kv_stage.group_grant`` asks), and per card the shared pool must hold ``FA layers of the stage x cell x level``.
  The pool floor of the cut solver is NOT a flag here: the launcher derives ``cap + chunk`` itself (``launcher.py`` PP-CUT POOL
  FLOOR RULE, p_bs=1) and a typed ``--pp-solve-pool-floor 262144`` would LOWER it below ``262144 + chunk`` (the 27B seat's own
  note, ``done/dual-schnitt-262k-1006.md`` section 4).  The pool per card is calibrated by ONE measured boot (``POOL_REF``): it is
  computed where the live cards are that boot's classes and N, else "nicht gerechnet".
* **Dual fit: planner calculation, not hw_fit.**  ``hw_fit`` does not model Dual (``hw_fit.py`` prints "Dual ... NOT modelled").  The
  coupling "P budget + overhead + D rest + D weights incl. draft + D Mamba pool <= card" is computed here, per card, from the model terms and the
  records (``D_AWAKE_REST_BOOKED_MIB``); the result is a NECESSARY condition, the launcher dry run (stage B) decides.

Two modes (``dual["modus"]``):

* ``profil``  the live inventory IS the profile's inventory and no goal asks for a rule: every value of the profile is CARRIED
  byte for byte (plan 1.3 A1, diff 0).  The KV obligation is then only JUDGED against those values.
* ``regel``   another inventory, ``ziele["force_rules"]``, a ``ziele["kv_tokens"]`` other than the default 262144, or ``ziele["dual_cut"]``: the
              Dual values are derived.

Goals read here (all optional): ``kv_tokens`` (obligation), ``dual_cut`` (pin the P cut: the budgets, the pool and the obligation follow
from it), ``dual_p_boot_tokens`` (KV tokens each P stage contributes at boot; the
reference boot's 95771 by default, ``unbelegt`` elsewhere), ``dual_pool_reserve_mib`` (margin the cut search keeps per card, default 0 =
D at rest).  STDLIB ONLY at module level.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.weg2 import dual_layout_plan as DL
from sglang.srt.weg2 import propose_rules as R

SCHEMA = "flliper.propose-e/1"
LABEL = "Dual fit: planner calculation, not hw_fit"
MIB = 1 << 20
#: the level grid of the P pool grants (``dual_p_kv_stage.STEP_TOKENS_DEFAULT``, env ``SGLANG_WEG2_DUAL_P_KV_STEP_TOKENS``)
STEP_TOKENS = 4096
#: the P pool page of this form (``PP-POOL-JOIN ... page_size=1`` in the reference dry run): the grant asks ``tokens + page``
PAGE_TOKENS = 1
#: budgets are written in steps of 10 MiB (the style of ``dual1h`` / ``dual-schnitt-262k-1006``: 6607 -> 6610)
BUDGET_ROUND_MIB = 10

#: D's INSTALLED weight vector in the Dual boot when the profile arms ``--d-reshard`` (``launcher.py`` ``ReshardSpec(policy, RC9_BASE, ...)``): D boots
#: at ``d_reshard.RC9_BASE`` (MLP units per rank), and P binds the part of its stage that is NOT this shard (``dual_share_env``: D owns the image
#: and publishes its installed vectors).  MEASURED: ``dual_w64.py`` (boot a3t5js: D TP0 weights 12.020 GiB = 12308 MiB) and the D log read by
#: ``deskq/done/dual-schnitt-262k-1006.md`` section 3 ("D-Gewichte [58,25,25]").  The ``--d-reshard`` preset ``dec`` (host share 0.65 / 0.72) is a
#: state AFTER a wake reshard and is NOT what the Dual-Passung rests on.  A COPY (``d_reshard`` imports the distributed stack);
#: ``test_planer_ape_dual_1006`` pins it equal.  The planer uses this ONE vector for P-private weights AND D's weights AND D's Mamba pool of a card
#: (one state of D per sum; the review of fix round 1: two different D shards in one line fit no real state).
INSTALLED_D_BASE = (58, 25, 25)
#: D's weights on card 0 of the reference boot (``dual_w64.py`` docstring: 12.020 GiB): the planer's model of the installed vector is checked against it
D_WEIGHTS_K0_MEASURED_MIB = 12308

#: THE measured calibration point of the Dual form: boot b9p (05.10., 27B NVFP4, 5090 + 2x 3080, cut 45,10,9, P budgets 8740,3000,3500,
#: ``--dual-p-overhead-mib 2500``, P mamba 8 slots): the card KV ledger's pool per card (``LEDGER-PHYS ... budget=`` 3332374528 /
#: 6408896512 / 6123683840 B) and the P boot KV of 95771 tokens per stage.  Source: ``deskq/done/1959-dual-p-262k.md`` section A,
#: ``deskq/done/dual-schnitt-262k-1006.md`` section 2 and the header of ``profiles/27b-nvfp4-dual1h.env``.
POOL_REF: Dict[str, Any] = {
    "classes": ("RTX5090", "RTX3080", "RTX3080"), "fmt": "nvfp4", "n_layers": 64,
    "cut": (45, 10, 9), "pool_mib": (3178, 6112, 5840), "slots": 8, "overhead_mib": 2500, "boot_tokens": 95771,
    "quelle": "Boot b9p 05.10. LEDGER-PHYS (deskq/done/1959-dual-p-262k.md section A; dual-schnitt-262k-1006.md section 2)",
}

#: group P's own state words used in the records (shared with ``propose``)
VORGESCHLAGEN, UNBELEGT = R.VORGESCHLAGEN, R.UNBELEGT


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def level_tokens(kv_tokens: int, step: int = STEP_TOKENS, page: int = PAGE_TOKENS) -> int:
    """The P pool level a prompt of ``kv_tokens`` asks: ``round_up(tokens + page, step)`` (``dual_p_kv_stage.group_grant``)."""
    t = int(kv_tokens) + int(page)
    return ((t + step - 1) // step) * step


def role_index(k: int, n: int, n_ref: int) -> int:
    """The entry of a reference vector of ``n_ref`` entries that card ``k`` of ``n`` reads (first / last, the middle one for the rest: the
    rule of ``hw_fit.role_value`` / ``inventory_view.ROLE``)."""
    if n_ref == n:
        return k
    if k == 0:
        return 0
    if k == n - 1:
        return n_ref - 1
    return 1 if n_ref > 2 else n_ref - 1


def round_budget(x: float) -> int:
    return int(round(x / BUDGET_ROUND_MIB)) * BUDGET_ROUND_MIB


def _ints(text: Any) -> Optional[List[int]]:
    v = R.parse_csv(text)
    if v is None:
        return None
    try:
        return [int(float(x)) for x in v]
    except ValueError:
        return None


def _int(text: Any) -> Optional[int]:
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return None


def fmt_of(modell: Mapping[str, Any]) -> str:
    """The checkpoint format key (``nvfp4`` / ``int8`` / ``fp8`` ...) of a ``flliper.model/1`` profile."""
    f = modell.get("format")
    f = f.get("v") if isinstance(f, Mapping) else f
    return str(f or "").lower()


def kv_payload_bytes(modell: Mapping[str, Any], fp: Any, kv_dtype: str) -> Tuple[float, str]:
    """Bytes of ONE token's KV in ONE full-attention layer as the Dual P pool counts them (payload only: ``cell_bytes=22528`` = 11 layers x
    2048), plus where the number comes from."""
    var = ((modell.get("kv") or {}).get("variants") or {}).get("fp8_e4m3" if kv_dtype.startswith("fp8") else "auto") or {}
    node = var.get("payload_bytes")
    v = node.get("v") if isinstance(node, Mapping) else node
    if v:
        return float(v), "Model profile (kv payload_bytes)"
    return float(fp.kv_bytes_per_token_per_attn_layer[kv_dtype]), "hw_fit cell (with scale buffer, unverified)"


def d_shares(fmt: str, totals: Sequence[float], reshard_wake: bool) -> Tuple[List[float], str]:
    """D's weight share per card (ONE vector for every term of a card: P-private weights, D's weights, D's Mamba pool): the installed
    vector ``INSTALLED_D_BASE`` when the profile reshards (``--d-reshard`` armed, D boots at RC9_BASE, three ranks), else proportional to the
    card sizes (the capacity-first rule; the launcher solves the real vector by ``--d-tp-objective``: planer assumption, unbelegt)."""
    n = len(totals)
    if reshard_wake and n == len(INSTALLED_D_BASE):
        tot = float(sum(INSTALLED_D_BASE))
        return [x / tot for x in INSTALLED_D_BASE], (
            "D weight share: installed D vector %s (RC9_BASE, --d-reshard boots D there; measured: D weights [58,25,25] in the dual boot, card 0 %d MiB); the preset 'dec' after a wake is another state and not calculated. ONE vector for P-private weights, D weights and D Mamba of a card" % (",".join(str(x) for x in INSTALLED_D_BASE), D_WEIGHTS_K0_MEASURED_MIB))
    tot = float(sum(totals))
    return [t / tot for t in totals], ("D weight share: proportional to the card size (planner assumption, unverified: the launcher solves the vector according to --d-tp-objective). ONE vector for P-private weights, D weights and D Mamba of a card")


def model_bytes(fp: Any) -> DL.ModelBytes:
    """The ``dual_layout_plan`` byte model of a ``hw_fit.FitProfile`` (one mixer entry per layer: the total, no family split)."""
    layers = tuple(DL.LayerBytes("attn" if f == "attn" else "gdn", int(round(m * MIB)), 0, 0)
                   for f, m in zip(fp.layer_families, fp.layer_dense_mib))
    return DL.ModelBytes(layers, int(round(fp.embed_mib * MIB)), int(round(fp.lm_head_mib * MIB)), int(round(fp.draft_mib * MIB)),
                         int(round(fp.visual_mib * MIB)))


def stage_terms(m: DL.ModelBytes, cut: Sequence[int], shares: Sequence[float], *, slots: int, slot_mib: float, cell_b: float,
                boot_tokens: int, vision_in_p: bool, embed_replicated: bool = True) -> List[Dict[str, float]]:
    """Per P stage / card, in MiB: ``priv`` (what P holds that D's shard does not: ``dual_layout_plan`` pp_only), ``mam`` (the stage's
    GDN layers x slots x slot size), ``kv`` (its full-attention layers x ``boot_tokens`` x cell), ``d_total`` / ``d_draft`` (D's shard of the
    whole model / of the draft on this card), layer counts."""
    n = len(cut)
    lay = DL.DualLayout(d_card=tuple(range(n)), p_card=tuple(range(n)), p_cut=tuple(int(x) for x in cut), mixer=tuple(shares),
                        mlp=tuple(shares), vocab=tuple(shares), draft=tuple(shares), draft_stage=None, vision_in_p=bool(vision_in_p),
                        d_embed_replicated=bool(embed_replicated))
    rows = DL.plan(m, lay, {i: 1 << 50 for i in range(n)}, {i: 0 for i in range(n)})
    tot = float(sum(shares))
    out = []
    for r in rows:
        lo, hi = r.layers
        gdn, fa = m.gdn_layers(lo, hi), m.fa_layers(lo, hi)
        out.append({"priv": r.pp_only / MIB, "d_total": r.d_total / MIB, "d_draft": m.draft * (shares[r.d_rank] / tot) / MIB,
                    "fa": fa, "gdn": gdn, "mam": gdn * slot_mib * slots, "kv": fa * cell_b * boot_tokens / MIB})
    return out


def compositions(total: int, parts: int):
    """Every split of ``total`` layers into ``parts`` stages of at least one layer."""
    if parts == 1:
        if total >= 1:
            yield (total,)
        return
    for a in range(1, total - parts + 2):
        for rest in compositions(total - a, parts - 1):
            yield (a,) + rest


# ---------------------------------------------------------------------------
# records: replace what the common machinery recorded for a key
# ---------------------------------------------------------------------------

def _drop(rec: Any, labels: Sequence[str]) -> None:
    ls = set(labels)
    rec.values[:] = [v for v in rec.values if v["key"] not in ls]
    rec.unbelegt[:] = [u for u in rec.unbelegt if u.split(": ", 1)[0] not in ls]


def _record(rec: Any, label: str, *, group: str, old: Optional[str], new: Optional[str], state: str, herkunft: str, grund: str,
            in_argv: bool = True, policy: str = "dual") -> None:
    _drop(rec, [label])
    rec.add(label, group=group, old=old, new=new, state=state, herkunft=herkunft, grund=grund, in_argv=in_argv, policy=policy)


# ---------------------------------------------------------------------------
# the proposal
# ---------------------------------------------------------------------------

def apply_dual(*, la: Any, la0: Any, rec: Any, cards: Sequence[Mapping[str, Any]], fp: Any, modell: Mapping[str, Any], z: Mapping[str, Any],
               bname: str, binv: Optional[Sequence[str]], same_inv: bool, live_cls: Sequence[str], kv_tokens: int, kv_dtype: str,
               rate: Sequence[float], rate_src: Sequence[str], layers: Sequence[int], attn: Sequence[int], posts: Any, d_slots: int, carried: Any,
               inv_txt: str, get: Any, set_: Any, slot_label: Any) -> Dict[str, Any]:
    """Make the common proposal Dual: records, rule values (mode ``regel``) and the Dual-Passung.  Edits ``la`` and ``rec`` in place and
    returns the ``dual`` section of the proposal (``SCHEMA``).  ``la0`` is the profile exactly as it came (the common machinery may have
    re-derived values in ``la`` already): every value of the profile's own calibration is read from it."""
    n = len(cards)
    totals = [float(c["total_mib"]) for c in cards]
    fmt = fmt_of(modell)
    from sglang.srt.weg2 import propose as _P

    # a kv_tokens goal other than the default obligation asks for the rule (the default value typed in changes nothing, as in the Flip forms)
    explicit = bool(z.get("kv_tokens")) and int(z["kv_tokens"]) != _P.KV_TOKENS_DEFAULT
    regel = (not same_inv) or bool(z.get("force_rules")) or explicit or bool(z.get("dual_cut"))
    modus = "regel" if regel else "profil"
    annahmen: List[str] = []
    unb = rec.unbelegt

    # the Flip-host arithmetic of the common machinery (draft solo on rank 0, D context fallback) does not describe the Dual: its own
    # rule below replaces it
    rec.unbelegt[:] = [u for u in rec.unbelegt if not u.startswith("D context per card:")]
    rec.hinweise[:] = [h for h in rec.hinweise if not h.startswith("Draft: ")]

    # --- is the basis a Dual profile? ----------------------------------------------------------------------------------------
    if "--dual-share" not in la.t and "--dual-layout" not in la.t:
        regel, modus = True, "regel"
        for flag, val, why in (("--dual-share", None, "both groups awake on the same cards (implies --dual-layout)"),
                               ("--dual-unified-kv", "on", "ONE KV pool per card for P and D (obligation of the dual form for the P KV stages)")):
            if val is None:
                la.t.append(flag)
            else:
                la.set_flag(flag, val)
            _record(rec, flag, group="-", old=None, new=val or "", state=VORGESCHLAGEN, herkunft="Form dual (the base profile is no dual profile)",
                    grund=why, policy="dual_form")
        rec.hinweise.append("The base profile %s is no dual profile: the planner adds only the obligatory flags of the form; MPS, priority, sleep and controls of the dual profile are missing and not invented" % bname)

    # --- the inputs of the model ------------------------------------------------------------------------------------------------
    cut_slot = ("flag", "-", "--pp-stage-ratio") if la0.get_flag("--pp-stage-ratio") is not None else ("extra", "p", "--pp-stage-ratio")
    attn_slot = ("flag", "-", "--pp-attn-stage-ratio") if la0.get_flag("--pp-attn-stage-ratio") is not None else ("extra", "p", "--pp-attn-stage-ratio")
    bud_old = la0.extra_get("p", "--rank-gpu-memory-mib")
    cut_old = get(la0, *cut_slot)
    attn_old = get(la0, *attn_slot)
    ovh_old = _int(la0.get_flag("--dual-p-overhead-mib"))
    top_old = _int(la0.get_flag("--dual-p-kv-max-tokens"))
    cap_old = _int(la0.get_flag("--max-kv-per-request"))
    dkv_old = _int(la0.get_flag("--dual-d-kv-max-tokens"))
    slots_old = _int(la0.extra_get("p", "--max-mamba-cache-size"))
    chunk = _int(la0.get_flag("--p-chunk-max"))
    p_bs = _int(la0.get_flag("--p-bs"))
    slots = slots_old if slots_old is not None else R.mamba_slots_p(1)
    ovh = ovh_old if ovh_old is not None else 1500                    # launcher default of --dual-p-overhead-mib (argparse)
    cell_b, cell_src = kv_payload_bytes(modell, fp, kv_dtype)
    slot_mib = posts.scalar("P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT")
    slot_src = "Record P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT"
    if slot_mib is None:
        slot_mib, slot_src = float(fp.mamba_mib_per_slot_per_linear_layer), "Model profile (Mamba state per slot)"
    vision = str(la0.get_flag("--weg2-vision") or "")
    vision_in_p = vision != "transient"
    reshard_wake = str(la0.get_flag("--d-reshard") or "") == "wake"
    boot_tokens = int(z.get("dual_p_boot_tokens") or POOL_REF["boot_tokens"])
    reserve_mib = float(z.get("dual_pool_reserve_mib") or 0.0)
    m = model_bytes(fp)
    sh_new, sh_txt = d_shares(fmt, totals, reshard_wake)
    annahmen.append(sh_txt)
    annahmen.append("P-private weights = D shard difference from the layer sizes of the model profile (dual_layout_plan, D holds the embedding block whole); Mamba slot %.4f MiB (%s); KV cell %.0f B per layer and token (%s)" % (slot_mib, slot_src, cell_b, cell_src))
    if not vision_in_p:
        annahmen.append("Vision is transient (--weg2-vision transient): no vision block in P")
    anchored_tokens = boot_tokens == POOL_REF["boot_tokens"]
    if not anchored_tokens:
        annahmen.append("Boot KV per P stage %d tokens (goal dual_p_boot_tokens), not the measured value of the reference boot" % boot_tokens)

    def terms(cut: Sequence[int], shares: Sequence[float], slots_: int) -> List[Dict[str, float]]:
        return stage_terms(m, cut, shares, slots=slots_, slot_mib=slot_mib, cell_b=cell_b, boot_tokens=boot_tokens,
                           vision_in_p=vision_in_p)

    # --- the profile's own calibration: residual of its P budget over the same terms at its own layout -----------------------------
    cut_ref = _ints(cut_old)
    bud_ref = _ints(bud_old)
    resid: List[Optional[float]] = [None] * n
    resid_note = "no calibration: the profile names no P budgets / no cut"
    if cut_ref and bud_ref and binv and len(cut_ref) == len(bud_ref) == len(binv) and sum(cut_ref) == fp.n_layers:
        n_ref = len(binv)
        by_cls: Dict[str, float] = {}
        for c in cards:
            by_cls.setdefault(c["class"], float(c["total_mib"]))
        ref_tot = [by_cls.get(c) for c in binv]
        preset = reshard_wake and n_ref == len(INSTALLED_D_BASE)
        if preset or all(t is not None for t in ref_tot):
            sh_ref, _ = d_shares(fmt, [t or 0.0 for t in ref_tot], reshard_wake)
            t_ref = terms(cut_ref, sh_ref, slots_old if slots_old is not None else slots)
            e_ref = [bud_ref[r] - (t_ref[r]["priv"] + t_ref[r]["mam"] + t_ref[r]["kv"]) for r in range(n_ref)]
            for k in range(n):
                r = role_index(k, n, n_ref)
                if live_cls[k] == binv[r]:
                    resid[k] = e_ref[r]
            resid_note = "Calibration residue of profile %s per role (budget minus model terms at the profile cut): %s MiB" % (
                bname, ", ".join("%.0f" % x for x in e_ref))
    annahmen.append(resid_note)

    # --- D's side of the coupling: awake rest (record) and D's own weights (the installed shard, draft included) ----------------
    aw = posts.vec("D_AWAKE_REST_BOOKED_MIB")
    if aw is None:
        annahmen.append("D_AWAKE_REST_BOOKED_MIB: no record for the profile: 0 MiB calculated (unverified)")
        unb.append("D_AWAKE_REST_BOOKED_MIB: no record -- the rest items of D are not calculated in the dual fit")
    ruhe = [0.0 if aw is None else _role(aw, k, n) for k in range(n)]
    # D's weights and Mamba pool are the shard of the SAME installed vector the P-private weights are the difference to (``sh_new``): one state of D
    # per card sum.  The D weights do not depend on P's cut (the cut only moves what P holds beyond them): any cut gives the same d_total.
    d_w = [t["d_total"] for t in terms([int(x) for x in layers], sh_new, slots)]
    # D's Mamba pool (d_slots slots x the linear layers) is paid from D's budget: split like D's weights
    d_mam = [fp.linear_layers * slot_mib * d_slots * sh_new[k] for k in range(n)]

    # --- the KV obligation: flags and the level ----------------------------------------------------------------------------------
    lvl = level_tokens(kv_tokens)
    pool_ref_ok = (tuple(live_cls) == POOL_REF["classes"] and n == 3 and fmt == POOL_REF["fmt"] and fp.n_layers == POOL_REF["n_layers"])

    ref_pool_terms: List[Any] = []

    def pool_of(cut: Sequence[int], shares: Sequence[float], slots_: int, ovh_: int) -> Optional[List[float]]:
        """The card KV ledger pool per card at ``cut``: the reference boot's pool shifted by the P-private, mamba and overhead terms."""
        if not pool_ref_ok:
            return None
        if not ref_pool_terms:
            sh_ref, _ = d_shares(fmt, totals, reshard_wake)
            ref_pool_terms.append(terms(POOL_REF["cut"], sh_ref, POOL_REF["slots"]))
        t_r = ref_pool_terms[0]
        t_n = terms(cut, shares, slots_)
        return [POOL_REF["pool_mib"][k] + (t_r[k]["priv"] + t_r[k]["mam"] + POOL_REF["overhead_mib"])
                - (t_n[k]["priv"] + t_n[k]["mam"] + ovh_) for k in range(3)]

    def need_of(cut: Sequence[int]) -> List[float]:
        fa = [m.fa_layers(lo, lo + c) for lo, c in zip((sum(cut[:i]) for i in range(len(cut))), cut)]
        return [f * cell_b * lvl / MIB for f in fa]

    # --- rule mode: derive the Dual values --------------------------------------------------------------------------------------
    cut_new: List[int] = [int(x) for x in (cut_ref or [])]
    search_note = ""
    if regel:
        cut_new = [int(x) for x in layers]
        origin_cut = "Rule: layer cut proportional to the GEMM rate (flip rule of the form), dual obligation per card in the verdict"
        state_cut = UNBELEGT if str(rate_src[0]).startswith(("Datenblatt", "unbelegt", "datasheet", "unverified")) else VORGESCHLAGEN
        pin = z.get("dual_cut")
        if pin:
            pv = _ints(pin) if not isinstance(pin, (list, tuple)) else [int(x) for x in pin]
            if not pv or len(pv) != n or sum(pv) != fp.n_layers or min(pv) < 1:
                raise _P.ProposeError("dual_cut %r: %d stages of at least one layer summing to %d needed" % (pin, n, fp.n_layers))
            cut_new = list(pv)
            origin_cut, state_cut = "Goal dual_cut: the dual cut set by the user", VORGESCHLAGEN
            search_note = "Cut set by the user (goal dual_cut); P budgets, pool and obligation follow from it"
        elif pool_ref_ok and anchored_tokens:
            best = None
            for cand in compositions(fp.n_layers, n):
                pool = pool_of(cand, sh_new, slots, ovh)
                need = need_of(cand)
                margin = min(pool[k] - need[k] for k in range(n))
                if margin < reserve_mib:
                    continue
                t = terms(cand, sh_new, slots)
                bud = [round_budget(t[k]["priv"] + t[k]["mam"] + t[k]["kv"] + (resid[k] or 0.0)) for k in range(n)]
                if any(bud[k] + ovh + ruhe[k] + d_w[k] + d_mam[k] > totals[k] for k in range(n)):
                    continue                  # physics: P budget + overhead + D rest + D weights + D Mamba pool must fit the card
                mk = max(cand[k] / float(rate[k] or 1.0) for k in range(n))
                key = (round(mk, 6), -margin)
                if best is None or key < best[0]:
                    best = (key, cand, margin)
            if best is not None:
                cut_new = list(best[1])
                search_note = ("Cut searched: the fastest (smallest stage time by GEMM rate) at which the pool of each card carries level %d (smallest rest %.0f MiB, reserve %.0f MiB)" % (lvl, best[2], reserve_mib))
                origin_cut = "Rule dual: fastest cut whose pool per card carries the KV obligation (calibration: reference boot b9p)"
                state_cut = UNBELEGT
            else:
                search_note = ("no cut carries level %d on all three cards (reserve %.0f MiB): the seed by GEMM rate stays, the verdict names the shortfall" % (lvl, reserve_mib))
        elif pool_ref_ok:
            search_note = "Cut search is skipped: the calibration of the pool applies to %d boot tokens per stage, here %d" % (POOL_REF["boot_tokens"], boot_tokens)
        else:
            search_note = ("Cut search is skipped: the pool per card is calibrated only for the cards and the model of the reference boot (%s) -- seed by GEMM rate" % ",".join(POOL_REF["classes"]))
        attn_new = R.attn_counts(fp.layer_families, cut_new)
        rec.hinweise.append("Dual cut: " + search_note)
        tn = terms(cut_new, sh_new, slots)
        bud_new = []
        unb_card = []
        for k in range(n):
            if resid[k] is None:
                unb_card.append(k)
            bud_new.append(round_budget(tn[k]["priv"] + tn[k]["mam"] + tn[k]["kv"] + (resid[k] or 0.0)))
        ct, at = R.csv(cut_new), R.csv(attn_new)
        old_c, old_a = cut_old, attn_old
        set_(la, *cut_slot, ct)
        set_(la, *attn_slot, at)
        la.extra_set("p", "--rank-gpu-memory-mib", R.csv(bud_new))
        bud_txt = R.csv(bud_new)
        cstate = state_cut if not unb_card else UNBELEGT
        _drop(rec, ["--pp-stage-ratio (Seed)"])
        _record(rec, slot_label(*cut_slot), group=cut_slot[1], old=old_c, new=ct, state=cstate, herkunft=origin_cut, grund=search_note,
                policy="dual_cut")
        _record(rec, slot_label(*attn_slot), group=attn_slot[1], old=old_a, new=at, state=cstate,
                herkunft="Rule: full-attention layers per stage from the layer cut", grund="exactly from the layer families of the model profile",
                policy="dual_cut_attn")
        _record(rec, "--extra-p --rank-gpu-memory-mib", group="p", old=bud_old, new=bud_txt, state=UNBELEGT,
                herkunft="Rule dual (shift calculation): model terms at the new cut + calibration residue of profile %s" % bname,
                grund="P budget per card = private weights of the stage + Mamba slots + boot KV (%d tokens) + calibration residue (graphs, activation, margin)%s; D is sized from the rest; unverified on the hardware (planner calculation)" % (
                          boot_tokens, "" if not unb_card else " [card(s) %s without a twin class in the profile: calibration residue 0]" % ",".join(map(str, unb_card))),
                policy="dual_budget")
        for k in unb_card:
            unb.append("P budget card %d (%s): no calibration residue -- no twin class in profile %s (graphs/activation/margin unmeasured)" % (
                k, live_cls[k], bname))
        cut_cur, bud_cur = cut_new, bud_new
    else:
        cut_cur = [int(x) for x in cut_ref] if cut_ref and len(cut_ref) == n else cut_new
        bud_cur = list(bud_ref) if bud_ref and len(bud_ref) == n else []

    # --- the scalar Dual values: level, cap, overhead, D pool ---------------------------------------------------------------------
    cap_new, top_new = int(kv_tokens), lvl
    if regel:
        if cap_old != cap_new:
            la.set_flag("--max-kv-per-request", str(cap_new))
        _record(rec, "--max-kv-per-request", group="-", old=None if cap_old is None else str(cap_old), new=str(cap_new), state=VORGESCHLAGEN,
                herkunft="Obligation of the dual form: KV obligation %d tokens (goal kv_tokens)" % kv_tokens,
                grund="P-COVERS-D-SESSION: the D session (--max-kv-per-request) must not exceed P's cap; P carries %d tokens of prefill context" % kv_tokens,
                policy="dual_pflicht")
        if top_old != top_new:
            la.set_flag("--dual-p-kv-max-tokens", str(top_new))
        _record(rec, "--dual-p-kv-max-tokens", group="-", old=None if top_old is None else str(top_old), new=str(top_new), state=VORGESCHLAGEN,
                herkunft="Obligation of the dual form: level round_up(%d + %d side, %d)" % (kv_tokens, PAGE_TOKENS, STEP_TOKENS),
                grund="virtual size of the P pool = largest stage a grant can request (dual_p_kv_stage.group_grant: want = min(top, round_up(tokens, step)))", policy="dual_pflicht")
        cap_cur, top_cur = cap_new, top_new
    else:
        cap_cur, top_cur = cap_old, top_old
        for lab, old, why in (("--max-kv-per-request", cap_old, "Per-request cap of the profile (P-COVERS-D-SESSION: P and D session equal)"),
                              ("--dual-p-kv-max-tokens", top_old, "virtual size of the P pool: the largest stage of a grant")):
            if old is not None:
                st, hk, gr = carried()
                _record(rec, lab, group="-", old=str(old), new=str(old), state=st, herkunft=hk, grund=why + "; " + gr, policy="dual_pflicht")
    # scalars that no hardware decides: carried with their provenance
    for lab, old, why in (("--dual-p-overhead-mib", ovh_old, "what P holds per card outside its budget (context, allocator, workspaces); D is sized from budget + overhead; measured only on the 5090 with 1024 chunks"),
                          ("--dual-d-kv-max-tokens", dkv_old, "virtual size of the D pool (global)")):
        if old is not None:
            st, hk, gr = carried()
            _record(rec, lab, group="-", old=str(old), new=str(old), state=st, herkunft=hk, grund=why + "; " + gr, policy="dual_knob")
    if ovh_old is None:
        rec.hinweise.append("--dual-p-overhead-mib is missing in the profile: the launcher default 1500 applies (meant for MPS, profile 27b-nvfp4-dual: 2500 measured)")
    # the Dual regulators and switches of the profile (names read from weg2/dual_green.py and dual_share.py), shown, not decided
    for lab, key in (("--dual-priority", "--dual-priority"), ("--dual-share-actuators", "--dual-share-actuators"),
                     ("--dual-green-ladder", "--dual-green-ladder"), ("--dual-p-sm-pct", "--dual-p-sm-pct"), ("--dual-p-duty", "--dual-p-duty")):
        old = la0.get_flag(key)
        if old is not None:
            st, hk, gr = carried()
            _record(rec, lab, group="-", old=old, new=old, state=st, herkunft=hk, grund="Control of the dual form (no card value); " + gr, policy="dual_knob")
    for name in ("SGLANG_WEG2_DUAL_SHARE_GREEN_TABLE", "SGLANG_WEG2_DUAL_SHARE_STARVE_AGE_S", "SGLANG_WEG2_DUAL_SHARE_STARVE_MAX_RUNG",
                 "SGLANG_WEG2_DUAL_GRANT_RETRY_MS"):
        if name in la0.env:
            st, hk, gr = carried()
            _record(rec, "env " + name, group="-", old=la0.env[name], new=la0.env[name], state=st, herkunft=hk,
                    grund="Environment variable of the dual form (name from weg2/dual_green.py / dual_share.py / dual_p_kv_stage.py); " + gr, policy="dual_env")
    # draft: D shards it over all cards (it is part of D's weights); P holds none in the Dual (it never flips)
    dk_old = la0.get_flag("--draft-kv-on-p")
    if regel and dk_old != "off":
        la.set_flag("--draft-kv-on-p", "off")
        _record(rec, "--draft-kv-on-p", group="-", old=dk_old, new="off", state=VORGESCHLAGEN, herkunft="Rule dual: the draft runs in D, P holds none",
                grund="the draft on P exists only for the flip (cold bytes); the dual never flips: saves weights and KV cell on the last stage",
                policy="dual_draft")
    elif dk_old is not None and dk_old != "off":
        rec.hinweise.append("--draft-kv-on-p %s in the dual profile: the draft on P is meant only for the flip; the dual never flips" % dk_old)
    # the pool floor in effect (the launcher derives it; see the module docstring)
    chunk_v = chunk if chunk is not None else 0
    floor_now = ((cap_cur or 0) + chunk_v) if (p_bs == 1 and la.get_flag("--pp-solve-pool-floor") is None) else None
    floor_need = int(kv_tokens) + chunk_v
    pf_txt = ("Pool floor of the cut solver in force: %s (launcher rule cap + chunk, p_bs=1); obligation of the dual form: %d (= %d + %d). The flag is not set: a typed %d would LOWER the floor below cap + chunk" % (
                  "%d = %s + %d" % (floor_now, cap_cur, chunk_v) if floor_now is not None else "set by the profile or not derivable",
                  floor_need, kv_tokens, chunk_v, kv_tokens))
    _record(rec, "--pp-solve-pool-floor", group="-", old=la.get_flag("--pp-solve-pool-floor"), new=la.get_flag("--pp-solve-pool-floor"),
            state=VORGESCHLAGEN, herkunft="Obligatory value of the dual form (plan 4c): KV obligation %d tokens" % kv_tokens, grund=pf_txt,
            in_argv=la.get_flag("--pp-solve-pool-floor") is not None, policy="dual_pflicht")

    # --- Dual-Passung: P budget + overhead + D rest + D weights (incl. draft) <= card ---------------------------------------------
    cur_slots = slots
    out_cards: List[Dict[str, Any]] = []
    level = "ja"
    first = ""
    sh_cur = sh_new
    tc = terms(cut_cur, sh_cur, cur_slots) if cut_cur and len(cut_cur) == n and sum(cut_cur) == fp.n_layers else None
    # D's own weights: its installed shard (the vector of the P-private terms above), draft included
    td = tc
    pool_cur = pool_of(cut_cur, sh_cur, cur_slots, ovh) if tc is not None else None
    need_cur = need_of(cut_cur) if tc is not None else None
    for k in range(n):
        row: Dict[str, Any] = {"ordinal": k, "name": cards[k].get("name"), "klasse": live_cls[k], "karte_mib": int(totals[k])}
        p = bud_cur[k] if bud_cur and k < len(bud_cur) else None
        row["p_budget_mib"] = p
        row["overhead_mib"] = ovh
        row["d_ruhe_mib"] = None if aw is None else ruhe[k]
        if tc is not None:
            row["p_privat_mib"] = round(tc[k]["priv"], 1)
            row["p_mamba_mib"] = round(tc[k]["mam"], 1)
            row["stufe_schichten"] = cut_cur[k]
            row["stufe_attn"] = int(tc[k]["fa"])
            row["d_gewichte_mib"] = round(td[k]["d_total"], 1)
            row["d_draft_mib"] = round(td[k]["d_draft"], 1)
            row["d_mamba_mib"] = round(d_mam[k], 1)
        if p is not None:
            row["d_budget_mib"] = int(totals[k] - p - ovh - ruhe[k])
            row["summe_mib"] = round(p + ovh + ruhe[k] + (row.get("d_gewichte_mib") or 0.0) + d_mam[k], 1)
            row["rest_mib"] = round(totals[k] - row["summe_mib"], 1)
            row["ok"] = row["rest_mib"] >= 0
            if tc is not None and p < tc[k]["priv"] + tc[k]["mam"]:
                row["ok"] = False
                row["grund"] = "the P budget (%d MiB) does not carry the private weights and Mamba slots of the stage (%.0f MiB)" % (
                    p, tc[k]["priv"] + tc[k]["mam"])
            if not row["ok"]:
                level = "nein"
                if not first:
                    first = row.get("grund") or ("Card %d (%s): P budget %d + overhead %d + rest %.0f + D weights %.0f + D Mamba %.0f = %.0f MiB > card %d MiB" % (
                        k, live_cls[k], p, ovh, ruhe[k], row.get("d_gewichte_mib") or 0.0, d_mam[k], row["summe_mib"], totals[k]))
        if pool_cur is not None:
            row["pool_mib"] = round(pool_cur[k], 1)
        if need_cur is not None:
            row["bedarf_mib"] = round(need_cur[k], 1)
            if pool_cur is not None:
                row["pool_rest_mib"] = round(pool_cur[k] - need_cur[k], 1)
        out_cards.append(row)

    # --- the obligation: verdict per card ----------------------------------------------------------------------------------------
    pf_grund: List[str] = []
    if top_cur is None or top_cur < lvl:
        pf_grund.append("--dual-p-kv-max-tokens %s < level %d: a grant for %d tokens is never approved" % (top_cur, lvl, kv_tokens))
    if cap_cur is None or cap_cur < kv_tokens:
        pf_grund.append("--max-kv-per-request %s < %d: the cap of the request is below the KV obligation" % (cap_cur, kv_tokens))
    if floor_now is not None and floor_now < floor_need:
        pf_grund.append("Pool floor %d < %d (cap + chunk of the obligation)" % (floor_now, floor_need))
    pool_known = pool_cur is not None and need_cur is not None
    if pool_known:
        for k, r in enumerate(out_cards):
            if r["pool_rest_mib"] < 0:
                pf_grund.append("Card %d (%s): pool %.0f MiB < need %.0f MiB (shortfall %.0f MiB, %d attention layers of the stage)" % (
                    k, live_cls[k], r["pool_mib"], r["bedarf_mib"], -r["pool_rest_mib"], int(out_cards[k].get("stufe_attn") or 0)))
        erfuellt: Optional[bool] = not pf_grund
    else:
        erfuellt = False if pf_grund else None
    if not pool_known:
        pf_grund.append("Pool per card not calculated: the calibration (boot b9p) applies only to %s and %s" % (",".join(POOL_REF["classes"]), POOL_REF["fmt"]))
    pflicht = {"kv_tokens": int(kv_tokens), "level_tokens": lvl, "step": STEP_TOKENS, "top": top_cur, "cap": cap_cur, "pool_floor_in_kraft": floor_now,
               "pool_floor_pflicht": floor_need, "erfuellt": erfuellt, "pool_geeicht": bool(pool_known), "grund": pf_grund,
               "eichung": POOL_REF["quelle"]}

    # --- verdicts -------------------------------------------------------------------------------------------------------------------
    verdikte: List[Dict[str, Any]] = []
    rest_min = min((r["rest_mib"] for r in out_cards if "rest_mib" in r), default=None)
    if not out_cards or any(r.get("p_budget_mib") is None for r in out_cards):
        level = "ungeprueft"
        first = "no P budget per card in the proposal: the fit is not calculated"
    txt = "%s: %s%s%s" % (LABEL, level, (" (smallest rest %.0f MiB)" % rest_min) if rest_min is not None and level != "ungeprueft" else "",
                          ("; " + first) if first else "")
    verdikte.append({"code": "DUAL-PASSUNG", "stufe": level, "text": txt, "etikett": LABEL, "rest_mib": rest_min})
    if erfuellt is True:
        ptxt, pstufe = "KV obligation %d tokens (level %d): met on all cards" % (kv_tokens, lvl), "ja"
    elif erfuellt is False:
        ptxt, pstufe = "KV obligation %d tokens (level %d): NOT met: %s" % (kv_tokens, lvl, "; ".join(pf_grund)), "nein"
    else:
        ptxt, pstufe = "KV obligation %d tokens (level %d): not calculated: %s" % (kv_tokens, lvl, "; ".join(pf_grund)), "ungeprueft"
    verdikte.append({"code": "DUAL-PFLICHT", "stufe": pstufe, "text": "%s: %s" % (LABEL, ptxt), "etikett": LABEL, "rest_mib": None})
    if erfuellt is False:
        rec.hinweise.append("Dual: " + ptxt)
    for a in annahmen:
        if a.startswith(("keine Eichung", "no calibration")) or "Zwilling" in a or "twin" in a:
            unb.append("Dual calibration residue: " + a)
    if regel:
        rec.hinweise.append("Dual: the launcher dry run may report W64 (Weg2TpOperatingPointInfeasible) with these P budgets if the evidence directory holds no measured dual D log of this model with the same D weight vector (dual_w64.find_dual_d_measurement): then the more pessimistic model of the launcher applies (2304 MiB reserve per rank), not this calculation. The run says that, not the planner.")
    if modus == "profil" and erfuellt is False:
        rec.hinweise.append("Dual: the profile carries its values unchanged (diff 0); it does not meet the KV obligation. With the goal force_rules (or a kv_tokens goal other than %d) the planner derives cut, P budgets, cap and level by the rule." % _P.KV_TOKENS_DEFAULT)
    return {"schema": SCHEMA, "etikett": LABEL, "modus": modus, "kopplung": "P budget + overhead + D rest + D weights (with draft) + D Mamba <= card",
            "karten": out_cards, "passung": {"stufe": level, "erste": first, "rest_min_mib": rest_min}, "pflicht": pflicht,
            "regeln": {"schnitt": cut_cur, "attn": R.attn_counts(fp.layer_families, cut_cur) if tc is not None else None, "budgets": bud_cur,
                       "level": lvl, "boot_tokens": boot_tokens, "uebernommen": not regel, "suche": search_note},
            "draft": {"d": "shared by D across all cards (part of the D weights)", "p": "none (--draft-kv-on-p off)", "mib": round(float(fp.draft_mib), 1)},
            "annahmen": annahmen, "verdikte": verdikte}


def _role(vec: Sequence[float], k: int, n: int) -> float:
    """``vec`` entry of card ``k`` of ``n`` (a record vector of another length is read by role)."""
    return float(vec[role_index(k, n, len(vec))])
