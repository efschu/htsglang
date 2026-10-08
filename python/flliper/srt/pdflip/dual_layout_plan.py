"""DUAL-TP3PP3 (F26): the byte plan of the double layout, per card.

User idea 2026-09-29 ~20:00Z: D (TP3, uneven, shards by a weight vector) and
P (PP3, whole layers per stage) are resident on the cards AT THE SAME TIME.
P reuses the part of its stage's layers that D already holds on the same card
and only adds the rest.  Variant #2 (~20:45Z) names the three parts:

* SHARED   -- bytes both layouts need on this card: D's TP shard of the layers
              of THIS card's P stage (plus replicated tensors of those layers,
              plus D's vocab shard of embed/lm_head where P keeps them here);
* PP_ONLY  -- what P needs in addition (the "diff"): the other ranks' shards of
              the stage's layers, and the rest of embed/lm_head/draft/vision
              that P keeps on this card;
* TP_ONLY  -- what D needs in addition: D's shard of the layers that live in
              OTHER P stages.  Variant #2 parks exactly this in host RAM when no
              decode is pending, so P can widen over the freed VRAM.

This module is PURE (stdlib only), every model number handed in.  It is the
planner input the user asked for (memory rang-form-gewichte-kv-per-flag-0928):
the D weight vector, the P layer cut and the context target are ARGUMENTS,
never literals; ``solve_cut_equal_context`` proposes a cut, the flag decides.

Byte model (checked against the checkpoint header by the tests):

* a layer = ``mixer`` (GDN linear_attn or full self_attn) + ``mlp`` + ``rep``
  (replicated: norms).  D splits ``mixer`` by the mixer vector and ``mlp`` by
  the mlp vector, holds ``rep`` on every rank.
* ``embed`` / ``lm_head`` are vocab-parallel in D (vocab vector), whole in P
  on the first / last stage.
* ``draft`` (DFlash2) is TP-sharded in D (draft vector, default = mlp vector);
  P keeps it whole on ``draft_stage`` (or not at all: ``draft_stage=None``).
* ``vision`` stays with P stage 0 when ``vision_in_p`` (D never holds it).

A ratio vector is a RATIO, not a fraction (memory rank-ratios-sind-
verhaeltnis): ``share_r = v_r / sum(v)``; bytes are split proportionally, not
rounded to units -- this is a PLAN, the loader's unit rounding is a few MiB.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

MIB = 1 << 20


class DualPlanError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class LayerBytes:
    kind: str  # "gdn" | "attn"
    mixer: int
    mlp: int
    rep: int = 0

    @property
    def total(self) -> int:
        return self.mixer + self.mlp + self.rep


@dataclasses.dataclass(frozen=True)
class ModelBytes:
    layers: Tuple[LayerBytes, ...]
    embed: int
    lm_head: int
    draft: int = 0
    vision: int = 0
    #: KV bytes per token per FULL-ATTENTION layer (all heads): 2*kv*hd*dtype.
    kv_bytes_per_token_layer: int = 2 * 4 * 256
    #: GDN recurrent state per request per GDN layer (all heads), bytes.
    gdn_state_bytes_per_layer: int = 0

    @property
    def layer_total(self) -> int:
        return sum(l.total for l in self.layers)

    def fa_layers(self, lo: int, hi: int) -> int:
        return sum(1 for l in self.layers[lo:hi] if l.kind == "attn")

    def gdn_layers(self, lo: int, hi: int) -> int:
        return sum(1 for l in self.layers[lo:hi] if l.kind == "gdn")


def model_bytes_from_families(
    fam: Mapping[int, Mapping[str, int]],
    *,
    embed: int,
    lm_head: int,
    draft: int = 0,
    vision: int = 0,
    kv_bytes_per_token_layer: int = 2 * 4 * 256,
    gdn_state_bytes_per_layer: int = 0,
) -> ModelBytes:
    """Build from per-layer family byte sums as read off a safetensors header
    (keys ``linear_attn`` / ``self_attn`` / ``mlp`` / anything else = rep)."""
    layers = []
    for i in sorted(int(k) for k in fam):
        f = fam[i] if i in fam else fam[str(i)]
        kind = "attn" if f.get("self_attn") else "gdn"
        mixer = int(f.get("self_attn", 0)) + int(f.get("linear_attn", 0))
        mlp = int(f.get("mlp", 0))
        rep = sum(int(v) for k, v in f.items() if k not in ("self_attn", "linear_attn", "mlp"))
        layers.append(LayerBytes(kind, mixer, mlp, rep))
    return ModelBytes(tuple(layers), int(embed), int(lm_head), int(draft), int(vision),
                      int(kv_bytes_per_token_layer), int(gdn_state_bytes_per_layer))


def _shares(vec: Sequence[float]) -> Tuple[float, ...]:
    if not vec or any(v < 0 for v in vec) or sum(vec) <= 0:
        raise DualPlanError(f"ratio vector must be non-negative with a positive sum, got {list(vec)}")
    s = float(sum(vec))
    return tuple(float(v) / s for v in vec)


@dataclasses.dataclass(frozen=True)
class DualLayout:
    """The two layouts over the same cards.

    ``d_card[r]``   -- card index of D rank r (TP order).
    ``p_card[s]``   -- card index of P stage s (pipeline order).
    ``p_cut[s]``    -- number of layers of stage s (sum = num layers).
    ``mixer`` / ``mlp`` / ``vocab`` / ``draft`` -- D ratio vectors in TP order.
    """

    d_card: Tuple[int, ...]
    p_card: Tuple[int, ...]
    p_cut: Tuple[int, ...]
    mixer: Tuple[float, ...]
    mlp: Tuple[float, ...]
    vocab: Tuple[float, ...]
    draft: Optional[Tuple[float, ...]] = None
    draft_stage: Optional[int] = -1  # -1 = last stage, None = P holds no draft
    vision_in_p: bool = True
    #: D keeps the input embedding WHOLE on every rank (27B NVFP4: the
    #: launcher's own D model prices embed 2425 MiB per rank). Then P's
    #: stage-0 embed is shared in full, not by D's vocab share.
    d_embed_replicated: bool = False

    def validate(self, model: ModelBytes) -> None:
        n = len(self.d_card)
        if sorted(self.d_card) != sorted(self.p_card) or len(set(self.d_card)) != n:
            raise DualPlanError(
                f"D ranks {list(self.d_card)} and P stages {list(self.p_card)} must cover the same "
                "cards, one D rank and one P stage per card")
        if len(self.p_cut) != n or sum(self.p_cut) != len(model.layers) or min(self.p_cut) < 0:
            raise DualPlanError(f"P cut {list(self.p_cut)} must have {n} non-negative entries summing to "
                                f"{len(model.layers)} layers")
        for name in ("mixer", "mlp", "vocab"):
            if len(getattr(self, name)) != n:
                raise DualPlanError(f"D {name} vector needs {n} entries")
        if self.draft is not None and len(self.draft) != n:
            raise DualPlanError(f"D draft vector needs {n} entries")

    def stage_range(self, s: int) -> Tuple[int, int]:
        lo = sum(self.p_cut[:s])
        return lo, lo + self.p_cut[s]


@dataclasses.dataclass(frozen=True)
class CardPlan:
    card: int
    d_rank: int
    p_stage: int
    layers: Tuple[int, int]
    shared: int
    pp_only: int
    tp_only: int
    d_total: int
    p_total: int
    capacity: int
    overhead: int
    #: False = stage 1a: both layouts hold their FULL weight sets (nothing is
    #: shared yet); the shared part is then paid twice.
    share: bool = True

    @property
    def weights(self) -> int:
        w = self.shared + self.pp_only + self.tp_only
        return w if self.share else w + self.shared

    @property
    def context(self) -> int:
        return self.capacity - self.weights - self.overhead

    def as_row(self) -> Dict[str, object]:
        return {
            "card": self.card, "d_rank": self.d_rank, "p_stage": self.p_stage,
            "layers": list(self.layers),
            "shared_mib": round(self.shared / MIB), "pp_only_mib": round(self.pp_only / MIB),
            "tp_only_mib": round(self.tp_only / MIB), "d_total_mib": round(self.d_total / MIB),
            "p_total_mib": round(self.p_total / MIB), "overhead_mib": round(self.overhead / MIB),
            "capacity_mib": round(self.capacity / MIB), "context_mib": round(self.context / MIB),
        }


def plan(model: ModelBytes, lay: DualLayout, capacity: Mapping[int, int],
         overhead: Mapping[int, int], share: bool = True) -> List[CardPlan]:
    """Per card: shared / pp_only / tp_only bytes and what is left for context.

    ``capacity[card]`` is what torch can allocate on the card (bytes);
    ``overhead[card]`` is everything that is neither weights nor context
    (both processes' CUDA contexts, workspaces, graphs, activations) -- handed
    in, measured elsewhere; this module never invents it.
    """
    lay.validate(model)
    mixer, mlp, vocab = _shares(lay.mixer), _shares(lay.mlp), _shares(lay.vocab)
    draft_sh = _shares(lay.draft) if lay.draft is not None else mlp
    n = len(lay.d_card)
    last = n - 1
    draft_stage = None if lay.draft_stage is None else (last if lay.draft_stage == -1 else lay.draft_stage)
    rows = []
    for card in lay.d_card:
        r = lay.d_card.index(card)
        s = lay.p_card.index(card)
        lo, hi = lay.stage_range(s)
        # D's shard on this card.
        d_layers = sum(l.mixer * mixer[r] + l.mlp * mlp[r] + l.rep for l in model.layers)
        d_embed = model.embed if lay.d_embed_replicated else model.embed * vocab[r]
        d_total = d_layers + d_embed + model.lm_head * vocab[r] + model.draft * draft_sh[r]
        # P's stage on this card.
        p_layers = sum(l.total for l in model.layers[lo:hi])
        p_total = p_layers
        p_total += model.embed if s == 0 else 0
        p_total += model.lm_head if s == last else 0
        p_total += model.draft if draft_stage == s else 0
        p_total += model.vision if (lay.vision_in_p and s == 0) else 0
        # SHARED: D's shard of exactly the tensors P also holds here.
        shared = sum(l.mixer * mixer[r] + l.mlp * mlp[r] + l.rep for l in model.layers[lo:hi])
        shared += d_embed if s == 0 else 0
        shared += model.lm_head * vocab[r] if s == last else 0
        shared += model.draft * draft_sh[r] if draft_stage == s else 0
        rows.append(CardPlan(
            card=card, d_rank=r, p_stage=s, layers=(lo, hi),
            shared=int(round(shared)), pp_only=int(round(p_total - shared)),
            tp_only=int(round(d_total - shared)), d_total=int(round(d_total)),
            p_total=int(round(p_total)), capacity=int(capacity[card]), overhead=int(overhead[card]),
            share=bool(share)))
    return rows


def check_physics(rows: Sequence[CardPlan], model: ModelBytes, lay: DualLayout) -> List[str]:
    """Two-way checks (memory planer-ergebnis-gegen-physik-pruefen-0928).
    Returns the list of violations (empty = consistent)."""
    bad = []
    tot_p = sum(r.p_total for r in rows)
    want_p = model.layer_total + model.embed + model.lm_head
    want_p += model.draft if lay.draft_stage is not None else 0
    want_p += model.vision if lay.vision_in_p else 0
    if abs(tot_p - want_p) > len(rows):
        bad.append(f"P stages hold {tot_p} B, the model has {want_p} B")
    tot_d = sum(r.d_total for r in rows)
    rep = sum(l.rep for l in model.layers) * (len(rows) - 1)
    rep += model.embed * (len(rows) - 1) if lay.d_embed_replicated else 0
    want_d = model.layer_total + model.embed + model.lm_head + model.draft + rep
    if abs(tot_d - want_d) > len(rows):
        bad.append(f"D shards hold {tot_d} B, the model (+replicas) has {want_d} B")
    for r in rows:
        if r.shared < 0 or r.pp_only < 0 or r.tp_only < 0:
            bad.append(f"card {r.card}: negative part {r.as_row()}")
        # each part is rounded on its own: allow the two rounding bytes
        if abs(r.shared + r.pp_only - r.p_total) > 2 or abs(r.shared + r.tp_only - r.d_total) > 2:
            bad.append(f"card {r.card}: parts do not add up {r.as_row()}")
    return bad


def solve_cut_equal_context(model: ModelBytes, lay: DualLayout, capacity: Mapping[int, int],
                            overhead: Mapping[int, int],
                            target: Optional[Mapping[int, float]] = None,
                            share: bool = True) -> Tuple[Tuple[int, ...], List[CardPlan]]:
    """The P cut whose per-card context is closest to ``target`` shares
    (default: proportional to capacity, i.e. "gleicher Kontextanteil").
    Exhaustive over all cuts with every stage >= 1 layer (64 layers, 3 stages:
    ~1900 candidates, pure arithmetic).  Objective: minimise the largest
    relative deviation of context share from target share; ties -> the larger
    context sum."""
    n = len(lay.d_card)
    L = len(model.layers)
    cards = list(lay.p_card)
    if target is None:
        tot = sum(capacity[c] for c in cards)
        target = {c: capacity[c] / tot for c in cards}
    best = None
    def cuts(k, left):
        if k == 1:
            if left >= 1:
                yield (left,)
            return
        for a in range(1, left - k + 2):
            for rest in cuts(k - 1, left - a):
                yield (a,) + rest
    for cut in cuts(n, L):
        cand = dataclasses.replace(lay, p_cut=cut)
        rows = plan(model, cand, capacity, overhead, share=share)
        ctx = {r.card: r.context for r in rows}
        tot = sum(ctx.values())
        if tot <= 0 or min(ctx.values()) <= 0:
            continue
        dev = max(abs(ctx[c] / tot - target[c]) / target[c] for c in cards)
        key = (round(dev, 4), -tot)
        if best is None or key < best[0]:
            best = (key, cut, rows)
    if best is None:
        raise DualPlanError("no cut leaves positive context on every card")
    return best[1], best[2]


def d_tokens(rows: Sequence[CardPlan], model: ModelBytes, d_share_of_context: float = 1.0) -> int:
    """KV tokens D can hold if it gets ``d_share_of_context`` of every card's
    context region and its DCP token vector follows the context (uneven DCP):
    all full-attention layers, every head, per token, spread over the cards."""
    per_tok = model.kv_bytes_per_token_layer * model.fa_layers(0, len(model.layers))
    return int(sum(max(0, r.context) for r in rows) * d_share_of_context // per_tok)


def p_tokens(rows: Sequence[CardPlan], model: ModelBytes, p_share_of_context: float = 1.0) -> int:
    """KV tokens P can hold: stage s keeps its OWN full-attention layers for
    every token, so the stage with the least context per FA layer binds."""
    best = None
    for r in rows:
        fa = model.fa_layers(*r.layers)
        if fa == 0:
            continue
        t = int(max(0, r.context) * p_share_of_context // (fa * model.kv_bytes_per_token_layer))
        best = t if best is None else min(best, t)
    return best or 0


def format_table(rows: Sequence[CardPlan]) -> str:
    cols = ("card", "d_rank", "p_stage", "layers", "shared_mib", "pp_only_mib", "tp_only_mib",
            "overhead_mib", "context_mib", "capacity_mib")
    out = [" ".join(f"{c:>12s}" for c in cols)]
    for r in rows:
        d = r.as_row()
        out.append(" ".join(f"{str(d[c]):>12s}" for c in cols))
    return "\n".join(out)


def p_kv_need(rows: Sequence[CardPlan], model: ModelBytes, prompt_tokens: int) -> Dict[int, int]:
    """Bytes of P KV each card needs to prefill ONE prompt of ``prompt_tokens``:
    stage s keeps its own full-attention layers for every token."""
    return {r.card: model.fa_layers(*r.layers) * model.kv_bytes_per_token_layer * int(prompt_tokens)
            for r in rows}


def p_max_prompt(rows: Sequence[CardPlan], model: ModelBytes, d_min: Mapping[int, int]) -> int:
    """The longest single prompt P can hold when D keeps at least ``d_min``
    bytes of context on every card (the card with the most FA layers per
    free byte binds)."""
    best = None
    for r in rows:
        fa = model.fa_layers(*r.layers)
        if fa == 0:
            continue
        free = max(0, r.context - int(d_min.get(r.card, 0)))
        t = free // (fa * model.kv_bytes_per_token_layer)
        best = t if best is None else min(best, t)
    return int(best or 0)


def d_tokens_after_p(rows: Sequence[CardPlan], model: ModelBytes, prompt_tokens: int) -> int:
    """D's KV tokens (uneven DCP, token vector follows capacity) with P holding
    KV for one ``prompt_tokens`` prompt; a card where P's need exceeds the
    context contributes nothing (the caller sees it in p_kv_need)."""
    need = p_kv_need(rows, model, prompt_tokens)
    per_tok = model.kv_bytes_per_token_layer * model.fa_layers(0, len(model.layers))
    return int(sum(max(0, r.context - need[r.card]) for r in rows) // per_tok)
