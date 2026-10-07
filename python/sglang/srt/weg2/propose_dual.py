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
* **Dual-Passung: Planer-Rechnung, nicht hw_fit.**  ``hw_fit`` does not model Dual (``hw_fit.py`` prints "Dual ... NOT modelled").  The
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
LABEL = "Dual-Passung: Planer-Rechnung, nicht hw_fit"
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
    "quelle": "Boot b9p 05.10. LEDGER-PHYS (deskq/done/1959-dual-p-262k.md Abschnitt A; dual-schnitt-262k-1006.md Abschnitt 2)",
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
        return float(v), "Modellprofil (kv payload_bytes)"
    return float(fp.kv_bytes_per_token_per_attn_layer[kv_dtype]), "hw_fit-Zelle (mit Skalenpuffer, unbelegt)"


def d_shares(fmt: str, totals: Sequence[float], reshard_wake: bool) -> Tuple[List[float], str]:
    """D's weight share per card (ONE vector for every term of a card: P-private weights, D's weights, D's Mamba pool): the installed
    vector ``INSTALLED_D_BASE`` when the profile reshards (``--d-reshard`` armed, D boots at RC9_BASE, three ranks), else proportional to the
    card sizes (the capacity-first rule; the launcher solves the real vector by ``--d-tp-objective``: planer assumption, unbelegt)."""
    n = len(totals)
    if reshard_wake and n == len(INSTALLED_D_BASE):
        tot = float(sum(INSTALLED_D_BASE))
        return [x / tot for x in INSTALLED_D_BASE], (
            "D-Gewichtsanteil: installierter D-Vektor %s (RC9_BASE, --d-reshard bootet D dort; gemessen: D-Gewichte [58,25,25] im Dual-Boot, "
            "Karte 0 %d MiB); der Preset 'dec' nach einem Wake ist ein anderer Zustand und nicht gerechnet. EIN Vektor fuer P-private Gewichte, "
            "D-Gewichte und D-Mamba einer Karte" % (",".join(str(x) for x in INSTALLED_D_BASE), D_WEIGHTS_K0_MEASURED_MIB))
    tot = float(sum(totals))
    return [t / tot for t in totals], ("D-Gewichtsanteil: proportional zur Kartengroesse (Planer-Annahme, unbelegt: der Launcher loest den Vektor "
                                       "nach --d-tp-objective). EIN Vektor fuer P-private Gewichte, D-Gewichte und D-Mamba einer Karte")


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
    rec.unbelegt[:] = [u for u in rec.unbelegt if not u.startswith("D-Kontext je Karte:")]
    rec.hinweise[:] = [h for h in rec.hinweise if not h.startswith("Draft: ")]

    # --- is the basis a Dual profile? ----------------------------------------------------------------------------------------
    if "--dual-share" not in la.t and "--dual-layout" not in la.t:
        regel, modus = True, "regel"
        for flag, val, why in (("--dual-share", None, "beide Gruppen wach auf denselben Karten (impliziert --dual-layout)"),
                               ("--dual-unified-kv", "on", "EIN KV-Pool je Karte fuer P und D (Pflicht der Dual-Form fuer die P-KV-Stufen)")):
            if val is None:
                la.t.append(flag)
            else:
                la.set_flag(flag, val)
            _record(rec, flag, group="-", old=None, new=val or "", state=VORGESCHLAGEN, herkunft="Form Dual (das Basisprofil ist kein Dual-Profil)",
                    grund=why, policy="dual_form")
        rec.hinweise.append("Das Basisprofil %s ist kein Dual-Profil: der Planer ergaenzt nur die Pflichtflags der Form; MPS, Prioritaet, "
                            "Schlaf und Regler des Dual-Profils fehlen und sind nicht erfunden" % bname)

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
        slot_mib, slot_src = float(fp.mamba_mib_per_slot_per_linear_layer), "Modellprofil (Mamba-Zustand je Slot)"
    vision = str(la0.get_flag("--weg2-vision") or "")
    vision_in_p = vision != "transient"
    reshard_wake = str(la0.get_flag("--d-reshard") or "") == "wake"
    boot_tokens = int(z.get("dual_p_boot_tokens") or POOL_REF["boot_tokens"])
    reserve_mib = float(z.get("dual_pool_reserve_mib") or 0.0)
    m = model_bytes(fp)
    sh_new, sh_txt = d_shares(fmt, totals, reshard_wake)
    annahmen.append(sh_txt)
    annahmen.append("P-private Gewichte = D-Shard-Differenz aus den Schichtgroessen des Modellprofils (dual_layout_plan, D haelt den "
                    "Embedding-Block ganz); Mamba-Slot %.4f MiB (%s); KV-Zelle %.0f B je Schicht und Token (%s)" % (slot_mib, slot_src, cell_b, cell_src))
    if not vision_in_p:
        annahmen.append("Vision ist transient (--weg2-vision transient): kein Vision-Block in P")
    anchored_tokens = boot_tokens == POOL_REF["boot_tokens"]
    if not anchored_tokens:
        annahmen.append("Boot-KV je P-Stufe %d Token (Ziel dual_p_boot_tokens), nicht der gemessene Wert des Referenzboots" % boot_tokens)

    def terms(cut: Sequence[int], shares: Sequence[float], slots_: int) -> List[Dict[str, float]]:
        return stage_terms(m, cut, shares, slots=slots_, slot_mib=slot_mib, cell_b=cell_b, boot_tokens=boot_tokens,
                           vision_in_p=vision_in_p)

    # --- the profile's own calibration: residual of its P budget over the same terms at its own layout -----------------------------
    cut_ref = _ints(cut_old)
    bud_ref = _ints(bud_old)
    resid: List[Optional[float]] = [None] * n
    resid_note = "keine Eichung: das Profil nennt keine P-Budgets / keinen Schnitt"
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
            resid_note = "Eichrest des Profils %s je Rolle (Budget minus Modellterme beim Profil-Schnitt): %s MiB" % (
                bname, ", ".join("%.0f" % x for x in e_ref))
    annahmen.append(resid_note)

    # --- D's side of the coupling: awake rest (record) and D's own weights (the installed shard, draft included) ----------------
    aw = posts.vec("D_AWAKE_REST_BOOKED_MIB")
    if aw is None:
        annahmen.append("D_AWAKE_REST_BOOKED_MIB: kein Record fuer das Profil: 0 MiB gerechnet (unbelegt)")
        unb.append("D_AWAKE_REST_BOOKED_MIB: kein Record -- die Ruhe-Posten von D sind in der Dual-Passung nicht gerechnet")
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
        origin_cut = "Regel: Schicht-Schnitt proportional zur GEMM-Rate (Flip-Regel der Form), Dual-Pflicht je Karte im Verdikt"
        state_cut = UNBELEGT if str(rate_src[0]).startswith(("Datenblatt", "unbelegt")) else VORGESCHLAGEN
        pin = z.get("dual_cut")
        if pin:
            pv = _ints(pin) if not isinstance(pin, (list, tuple)) else [int(x) for x in pin]
            if not pv or len(pv) != n or sum(pv) != fp.n_layers or min(pv) < 1:
                raise _P.ProposeError("dual_cut %r: %d stages of at least one layer summing to %d needed" % (pin, n, fp.n_layers))
            cut_new = list(pv)
            origin_cut, state_cut = "Ziel dual_cut: der vom Anwender gesetzte Dual-Schnitt", VORGESCHLAGEN
            search_note = "Schnitt vom Anwender gesetzt (Ziel dual_cut); P-Budgets, Pool und Pflicht folgen aus ihm"
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
                search_note = ("Schnitt gesucht: der schnellste (kleinste Stufenzeit nach GEMM-Rate), bei dem der Pool jeder Karte das Level %d "
                               "traegt (kleinster Rest %.0f MiB, Reserve %.0f MiB)" % (lvl, best[2], reserve_mib))
                origin_cut = "Regel Dual: schnellster Schnitt, dessen Pool je Karte die KV-Pflicht traegt (Eichung: Referenzboot b9p)"
                state_cut = UNBELEGT
            else:
                search_note = ("kein Schnitt traegt das Level %d auf allen drei Karten (Reserve %.0f MiB): der Seed nach GEMM-Rate bleibt, "
                               "das Verdikt nennt den Fehlbetrag" % (lvl, reserve_mib))
        elif pool_ref_ok:
            search_note = "Schnittsuche entfaellt: die Eichung des Pools gilt fuer %d Boot-Token je Stufe, hier %d" % (POOL_REF["boot_tokens"], boot_tokens)
        else:
            search_note = ("Schnittsuche entfaellt: der Pool je Karte ist nur fuer die Karten und das Modell des Referenzboots geeicht "
                           "(%s) -- Seed nach GEMM-Rate" % ",".join(POOL_REF["classes"]))
        attn_new = R.attn_counts(fp.layer_families, cut_new)
        rec.hinweise.append("Dual-Schnitt: " + search_note)
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
                herkunft="Regel: Voll-Attention-Schichten je Stufe aus dem Schichtschnitt", grund="exakt aus den Layer-Familien des Modellprofils",
                policy="dual_cut_attn")
        _record(rec, "--extra-p --rank-gpu-memory-mib", group="p", old=bud_old, new=bud_txt, state=UNBELEGT,
                herkunft="Regel Dual (Verschiebungsrechnung): Modellterme beim neuen Schnitt + Eichrest des Profils %s" % bname,
                grund="P-Budget je Karte = private Gewichte der Stufe + Mamba-Slots + Boot-KV (%d Token) + Eichrest (Graphen, Aktivierung, Spielraum)%s; "
                      "D wird aus dem Rest bemessen; am Metall unbelegt (Planer-Rechnung)" % (
                          boot_tokens, "" if not unb_card else " [Karte(n) %s ohne Zwillingsklasse im Profil: Eichrest 0]" % ",".join(map(str, unb_card))),
                policy="dual_budget")
        for k in unb_card:
            unb.append("P-Budget Karte %d (%s): kein Eichrest -- keine Zwillingsklasse im Profil %s (Graphen/Aktivierung/Spielraum ungemessen)" % (
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
                herkunft="Pflicht der Dual-Form: KV-Pflicht %d Token (Ziel kv_tokens)" % kv_tokens,
                grund="P-COVERS-D-SESSION: die D-Sitzung (--max-kv-per-request) darf P's Deckel nicht uebersteigen; P traegt %d Token Prefill-Kontext" % kv_tokens,
                policy="dual_pflicht")
        if top_old != top_new:
            la.set_flag("--dual-p-kv-max-tokens", str(top_new))
        _record(rec, "--dual-p-kv-max-tokens", group="-", old=None if top_old is None else str(top_old), new=str(top_new), state=VORGESCHLAGEN,
                herkunft="Pflicht der Dual-Form: Level round_up(%d + %d Seite, %d)" % (kv_tokens, PAGE_TOKENS, STEP_TOKENS),
                grund="virtuelle Groesse des P-Pools = groesste Stufe, die ein Grant anfordern kann (dual_p_kv_stage.group_grant: want = min(top, "
                      "round_up(tokens, step)))", policy="dual_pflicht")
        cap_cur, top_cur = cap_new, top_new
    else:
        cap_cur, top_cur = cap_old, top_old
        for lab, old, why in (("--max-kv-per-request", cap_old, "Pro-Request-Deckel des Profils (P-COVERS-D-SESSION: P und D-Sitzung gleich)"),
                              ("--dual-p-kv-max-tokens", top_old, "virtuelle Groesse des P-Pools: die groesste Stufe eines Grants")):
            if old is not None:
                st, hk, gr = carried()
                _record(rec, lab, group="-", old=str(old), new=str(old), state=st, herkunft=hk, grund=why + "; " + gr, policy="dual_pflicht")
    # scalars that no hardware decides: carried with their provenance
    for lab, old, why in (("--dual-p-overhead-mib", ovh_old, "was P je Karte ausserhalb seines Budgets haelt (Kontext, Allokator, Workspaces); D wird aus "
                                                                 "Budget + Overhead bemessen; gemessen nur auf der 5090 bei 1024er Chunks"),
                          ("--dual-d-kv-max-tokens", dkv_old, "virtuelle Groesse des D-Pools (global)")):
        if old is not None:
            st, hk, gr = carried()
            _record(rec, lab, group="-", old=str(old), new=str(old), state=st, herkunft=hk, grund=why + "; " + gr, policy="dual_knob")
    if ovh_old is None:
        rec.hinweise.append("--dual-p-overhead-mib fehlt im Profil: der Launcher-Standard 1500 gilt (fuer MPS gedacht, Profil 27b-nvfp4-dual: 2500 gemessen)")
    # the Dual regulators and switches of the profile (names read from weg2/dual_green.py and dual_share.py), shown, not decided
    for lab, key in (("--dual-priority", "--dual-priority"), ("--dual-share-actuators", "--dual-share-actuators"),
                     ("--dual-green-ladder", "--dual-green-ladder"), ("--dual-p-sm-pct", "--dual-p-sm-pct"), ("--dual-p-duty", "--dual-p-duty")):
        old = la0.get_flag(key)
        if old is not None:
            st, hk, gr = carried()
            _record(rec, lab, group="-", old=old, new=old, state=st, herkunft=hk, grund="Regler der Dual-Form (kein Kartenwert); " + gr, policy="dual_knob")
    for name in ("SGLANG_WEG2_DUAL_SHARE_GREEN_TABLE", "SGLANG_WEG2_DUAL_SHARE_STARVE_AGE_S", "SGLANG_WEG2_DUAL_SHARE_STARVE_MAX_RUNG",
                 "SGLANG_WEG2_DUAL_GRANT_RETRY_MS"):
        if name in la0.env:
            st, hk, gr = carried()
            _record(rec, "env " + name, group="-", old=la0.env[name], new=la0.env[name], state=st, herkunft=hk,
                    grund="Umgebungsvariable der Dual-Form (Name aus weg2/dual_green.py / dual_share.py / dual_p_kv_stage.py); " + gr, policy="dual_env")
    # draft: D shards it over all cards (it is part of D's weights); P holds none in the Dual (it never flips)
    dk_old = la0.get_flag("--draft-kv-on-p")
    if regel and dk_old != "off":
        la.set_flag("--draft-kv-on-p", "off")
        _record(rec, "--draft-kv-on-p", group="-", old=dk_old, new="off", state=VORGESCHLAGEN, herkunft="Regel Dual: der Draft laeuft in D, P haelt keinen",
                grund="der Draft auf P existiert nur fuer den Flip (kalte Bytes); das Dual flippt nie: spart Gewichte und KV-Zelle auf der letzten Stufe",
                policy="dual_draft")
    elif dk_old is not None and dk_old != "off":
        rec.hinweise.append("--draft-kv-on-p %s im Dual-Profil: der Draft auf P ist nur fuer den Flip gedacht; das Dual flippt nie" % dk_old)
    # the pool floor in effect (the launcher derives it; see the module docstring)
    chunk_v = chunk if chunk is not None else 0
    floor_now = ((cap_cur or 0) + chunk_v) if (p_bs == 1 and la.get_flag("--pp-solve-pool-floor") is None) else None
    floor_need = int(kv_tokens) + chunk_v
    pf_txt = ("Pool-Floor des Schnitt-Loesers in Kraft: %s (Launcher-Regel cap + chunk, p_bs=1); Pflicht der Dual-Form: %d (= %d + %d). Das Flag wird "
              "nicht gesetzt: ein getipptes %d wuerde den Floor unter cap + chunk SENKEN" % (
                  "%d = %s + %d" % (floor_now, cap_cur, chunk_v) if floor_now is not None else "vom Profil gesetzt oder nicht ableitbar",
                  floor_need, kv_tokens, chunk_v, kv_tokens))
    _record(rec, "--pp-solve-pool-floor", group="-", old=la.get_flag("--pp-solve-pool-floor"), new=la.get_flag("--pp-solve-pool-floor"),
            state=VORGESCHLAGEN, herkunft="Pflichtwert der Dual-Form (Plan 4c): KV-Pflicht %d Token" % kv_tokens, grund=pf_txt,
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
                row["grund"] = "das P-Budget (%d MiB) traegt die privaten Gewichte und Mamba-Slots der Stufe (%.0f MiB) nicht" % (
                    p, tc[k]["priv"] + tc[k]["mam"])
            if not row["ok"]:
                level = "nein"
                if not first:
                    first = row.get("grund") or ("Karte %d (%s): P-Budget %d + Overhead %d + Ruhe %.0f + D-Gewichte %.0f + D-Mamba %.0f = %.0f MiB > Karte %d MiB" % (
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
        pf_grund.append("--dual-p-kv-max-tokens %s < Level %d: ein Grant fuer %d Token wird nie bewilligt" % (top_cur, lvl, kv_tokens))
    if cap_cur is None or cap_cur < kv_tokens:
        pf_grund.append("--max-kv-per-request %s < %d: der Deckel der Anfrage liegt unter der KV-Pflicht" % (cap_cur, kv_tokens))
    if floor_now is not None and floor_now < floor_need:
        pf_grund.append("Pool-Floor %d < %d (cap + chunk der Pflicht)" % (floor_now, floor_need))
    pool_known = pool_cur is not None and need_cur is not None
    if pool_known:
        for k, r in enumerate(out_cards):
            if r["pool_rest_mib"] < 0:
                pf_grund.append("Karte %d (%s): Pool %.0f MiB < Bedarf %.0f MiB (Fehlbetrag %.0f MiB, %d Attention-Schichten der Stufe)" % (
                    k, live_cls[k], r["pool_mib"], r["bedarf_mib"], -r["pool_rest_mib"], int(out_cards[k].get("stufe_attn") or 0)))
        erfuellt: Optional[bool] = not pf_grund
    else:
        erfuellt = False if pf_grund else None
    if not pool_known:
        pf_grund.append("Pool je Karte nicht gerechnet: die Eichung (Boot b9p) gilt nur fuer %s und %s" % (",".join(POOL_REF["classes"]), POOL_REF["fmt"]))
    pflicht = {"kv_tokens": int(kv_tokens), "level_tokens": lvl, "step": STEP_TOKENS, "top": top_cur, "cap": cap_cur, "pool_floor_in_kraft": floor_now,
               "pool_floor_pflicht": floor_need, "erfuellt": erfuellt, "pool_geeicht": bool(pool_known), "grund": pf_grund,
               "eichung": POOL_REF["quelle"]}

    # --- verdicts -------------------------------------------------------------------------------------------------------------------
    verdikte: List[Dict[str, Any]] = []
    rest_min = min((r["rest_mib"] for r in out_cards if "rest_mib" in r), default=None)
    if not out_cards or any(r.get("p_budget_mib") is None for r in out_cards):
        level = "ungeprueft"
        first = "kein P-Budget je Karte im Vorschlag: die Passung ist nicht gerechnet"
    txt = "%s: %s%s%s" % (LABEL, level, (" (kleinster Rest %.0f MiB)" % rest_min) if rest_min is not None and level != "ungeprueft" else "",
                          ("; " + first) if first else "")
    verdikte.append({"code": "DUAL-PASSUNG", "stufe": level, "text": txt, "etikett": LABEL, "rest_mib": rest_min})
    if erfuellt is True:
        ptxt, pstufe = "KV-Pflicht %d Token (Level %d): erfuellt auf allen Karten" % (kv_tokens, lvl), "ja"
    elif erfuellt is False:
        ptxt, pstufe = "KV-Pflicht %d Token (Level %d): NICHT erfuellt: %s" % (kv_tokens, lvl, "; ".join(pf_grund)), "nein"
    else:
        ptxt, pstufe = "KV-Pflicht %d Token (Level %d): nicht gerechnet: %s" % (kv_tokens, lvl, "; ".join(pf_grund)), "ungeprueft"
    verdikte.append({"code": "DUAL-PFLICHT", "stufe": pstufe, "text": "%s: %s" % (LABEL, ptxt), "etikett": LABEL, "rest_mib": None})
    if erfuellt is False:
        rec.hinweise.append("Dual: " + ptxt)
    for a in annahmen:
        if a.startswith("keine Eichung") or "Zwilling" in a:
            unb.append("Dual-Eichrest: " + a)
    if regel:
        rec.hinweise.append("Dual: der Launcher-Trockenlauf kann bei diesen P-Budgets W64 (Weg2TpOperatingPointInfeasible) melden, wenn im Evidence-Verzeichnis "
                            "kein gemessenes Dual-D-Log dieses Modells mit demselben D-Gewichtsvektor liegt (dual_w64.find_dual_d_measurement): dann gilt "
                            "das pessimistischere Modell des Launchers (2304 MiB Reserve je Rang), nicht diese Rechnung. Das sagt der Lauf, nicht der Planer.")
    if modus == "profil" and erfuellt is False:
        rec.hinweise.append("Dual: das Profil traegt seine Werte unveraendert (Diff 0); die KV-Pflicht erfuellt es nicht. Mit dem Ziel force_rules (oder einem "
                            "anderen kv_tokens-Ziel als %d) leitet der Planer Schnitt, P-Budgets, Deckel und Level nach der Regel ab." % _P.KV_TOKENS_DEFAULT)
    return {"schema": SCHEMA, "etikett": LABEL, "modus": modus, "kopplung": "P-Budget + Overhead + D-Ruhe + D-Gewichte (mit Draft) + D-Mamba <= Karte",
            "karten": out_cards, "passung": {"stufe": level, "erste": first, "rest_min_mib": rest_min}, "pflicht": pflicht,
            "regeln": {"schnitt": cut_cur, "attn": R.attn_counts(fp.layer_families, cut_cur) if tc is not None else None, "budgets": bud_cur,
                       "level": lvl, "boot_tokens": boot_tokens, "uebernommen": not regel, "suche": search_note},
            "draft": {"d": "von D ueber alle Karten geteilt (Teil der D-Gewichte)", "p": "keiner (--draft-kv-on-p off)", "mib": round(float(fp.draft_mib), 1)},
            "annahmen": annahmen, "verdikte": verdikte}


def _role(vec: Sequence[float], k: int, n: int) -> float:
    """``vec`` entry of card ``k`` of ``n`` (a record vector of another length is read by role)."""
    return float(vec[role_index(k, n, len(vec))])
