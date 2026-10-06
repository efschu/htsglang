"""AP-C rules of the profile planner: the ARITHMETIC behind ``weg2/propose.py`` (plan PLAN-PROFIL-PLANER-1006 section 1.2 K1-K5).

Pure functions on plain numbers and dicts, no I/O, no NVML, no torch, no launcher import.  Every function returns the value
AND the words that say where it came from -- ``propose.py`` stores both per value (``vorgeschlagen`` / ``unbelegt`` plus a
sentence), so a figure the planner computed from a datasheet is never shown as a measurement (HOCHRECHNUNG != MESSUNG).

What lives here, by criterion of the plan (section 1.2):

* **K5 card order**  :func:`order_cards` -- ``card_identity.order_key`` (biggest NVML total, nameplate bandwidth, NVML index);
  rank 0 = attention host.
* **K2 seeds**       :func:`rate_table` (GEMM rates: measured > card library > datasheet peak, ONE basis for all cards),
  :func:`split_layers` (P cut proportional to the compute rate, clamped to the memory capacity of each stage),
  :func:`attn_counts` (the attention layers each stage owns, exact from the family list), :func:`class_rekey` (a per-card
  MEASURED vector of the profile onto another inventory: class maximum, arch twin as BORROWED, never an average),
  :func:`scale_by_seats` (a seat-dependent vector).
* **K1 fit**         :func:`fit_profile_from_model` + :func:`stage_budgets` -- the ``hw_fit`` terms (weights + KV + mamba +
  record posts against ``total - residue`` per card), fed from the ``flliper.model/1`` profile instead of a typed table.
* **K3 draft**       :func:`draft_placement` -- ``solo`` on rank 0 when draft + dense weights + the KV obligation fit rank 0,
  else ``split`` and a hint.
* **K4 rest rules**  :func:`fr_p` (MoE: whatever a P stage has left goes to resident experts), :func:`form_a_d` (MoE: the
  Form A solve ``form_a_plan.solve_form_a``: host + expert workers), :func:`dense_d_shares` (dense: TP shares proportional to
  the VRAM budget), :func:`mamba_slots_p` / :func:`mamba_slots_d` (the "Sitze gleichzeitig" regulator).

Nothing here chooses a value the launcher solves itself at boot (the exact layer cut under ``--pp-solve-objective``, the
D rank VRAM fractions, the KV cut): those the launcher prints in its plan (stage B, the oracle).  A seed that is not applied
is shown, not hidden.

STDLIB ONLY at module level; ``card_identity`` / ``hw_fit`` / ``form_a_plan`` / ``planner.card_library`` are imported inside
the functions that need them (all four are stdlib/msgspec-only, none imports torch).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

MIB = 1 << 20

#: HW-GENERIC 1002: the launcher flags (dest names) and the --extra-p/-d, --env-p/-d tokens that carry a POSITIONAL per-card
#: vector (``weg2/launcher.py:5964-5975``).  A COPY on purpose (the launcher cannot be imported from a stdlib-only planner);
#: ``test_planer_apc_propose_1006`` pins it equal to the launcher's tuples, so a launcher edit turns the test red.
POSITIONAL_VECTOR_FLAGS = (
    "d_foreign_context_mib", "d_nontorch_mib", "pp_stage_ratio", "pp_attn_stage_ratio",
    "pp_cut_expert_device_fraction", "pp_cut_expert_lru_rows", "user_reserve_mib",
    "d_reserve_mib", "pp_cut_reserve_mib", "d_reshard_presets", "p_barlink_bar1_window_mib",
)
POSITIONAL_VECTOR_TOKENS = (
    "--rank-role", "--rank-tp-ratio", "--rank-moe-ratio", "--rank-moe-resident-fraction",
    "--rank-user-reserve-mib", "--rank-gpu-memory-mib", "--pp-stage-ratio",
    "--pp-attn-stage-ratio", "SGLANG_MOE_SCRATCH_SLOTS=", "SGLANG_MOE_RESIDENT_EXPERT_FRACTION=",
    "SGLANG_WEG2_L15_MIB=", "SGLANG_WEG2_EXTEND_TRIM_MIB=",
)

#: state words of a value
VORGESCHLAGEN = "vorgeschlagen"
UNBELEGT = "unbelegt"

#: the largest resident expert fraction of a MoE stage: ``(E - 2) / E`` ("groesste Offload-Fraction", PP-CUT FRACTION-SOLVE)
FR_MAX_EXPERT_MARGIN = 2

#: arch -> reference class a card of that arch BORROWS its measured per-card values from (``hw_fit.RESIDUE_ARCH_TWIN``)
ARCH_TWIN = {"sm120": "RTX5090", "sm86": "RTX3080", "sm89": "RTX3080"}

#: PCIe nominal payload rate per lane in GB/s by generation (datasheet; encoding overhead included) -- only the RATIO of
#: two cards matters (the Form A spill split), never the absolute value
PCIE_GBS_PER_LANE = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938}

#: mamba slots a P stage holds: ``mamba floor 4 per seat + retention 8`` (launcher line ``PP-CUT budget posts``:
#: "mamba floor 6 seats x 4 = 24, retention budget 8" -> 32 slots at 6 seats)
P_MAMBA_SLOTS_PER_SEAT, P_MAMBA_RETENTION = 4, 8
#: D mamba slots of the reference dry run: 38 at 6 seats (``hw_fit.D_MAMBA_SLOTS_DEFAULT``); scaled linearly (one point)
D_MAMBA_SLOTS_REF, D_MAMBA_SEATS_REF = 38, 6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def largest_remainder(total: int, weights: Sequence[float]) -> List[int]:
    """``total`` split over ``weights`` by largest remainder, ties to the LOWER index; a zero weight gets zero."""
    wsum = float(sum(weights))
    if wsum <= 0 or total < 0:
        raise ValueError("cannot split %r over weights summing to %r" % (total, wsum))
    quotas = [total * w / wsum for w in weights]
    sizes = [int(q) for q in quotas]
    order = sorted(range(len(weights)), key=lambda r: (-(quotas[r] - int(quotas[r])), r))
    for k in range(total - sum(sizes)):
        sizes[order[k % len(order)]] += 1
    return sizes


def csv(values: Sequence[Any]) -> str:
    """``[1, 2.0, 3]`` -> ``"1,2,3"`` (integral floats without ``.0``)."""
    out = []
    for v in values:
        out.append(str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v))
    return ",".join(out)


def parse_csv(text: Any) -> Optional[List[str]]:
    """The entries of a ``"a,b,c"`` vector, None when ``text`` is not a vector of two or more entries."""
    if isinstance(text, (list, tuple)):
        return [str(x) for x in text]
    if isinstance(text, str) and "," in text:
        return [x.strip() for x in text.split(",")]
    return None


def num(x: Any) -> float:
    return float(x)


# ---------------------------------------------------------------------------
# K5: card order and identity
# ---------------------------------------------------------------------------

def order_cards(cards: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The cards in CUDA-ordinal order (rank 0 / PP0 / TP0 first) by ``card_identity.order_key``, each as a new dict with
    ``ordinal``, ``class`` (the calibration label, else the card key), ``calibrated`` and ``arch`` added."""
    from sglang.srt.weg2 import card_identity as ci

    ordered = sorted(cards, key=ci.order_key)
    out = []
    for i, c in enumerate(ordered):
        p = ci.props_of(c)
        d = dict(c)
        d.update({"ordinal": i, "class": ci.class_label(c), "calibrated": ci.calibration_class(c) is not None,
                  "arch": "sm%d%d" % tuple(p.cc) if p.cc else "sm?", "total_mib": int(p.total_mib)})
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# K2: rates and the P cut seed
# ---------------------------------------------------------------------------

def rate_table(cards: Sequence[Mapping[str, Any]], *, measured: Optional[Mapping[str, float]] = None,
               library: Any = None) -> Tuple[List[float], List[str], str]:
    """GEMM rate per card (TFLOP/s) on ONE basis: ``(rates, source per card, basis text)``.

    Order of trust, per card: the card's own measured node (``card["tflops"]`` with ``tflops_src == "gemessen"``) > the
    ``measured`` mapping (card name or ``"<name> <GB>GB"`` -> TFLOP/s) > the measured ``gemm_tflops`` of the card library row
    (``library`` or, when None, ``card_rate_pass.load_measured_library()`` = ``card_library.json``) > the datasheet FP16/BF16
    peak of the card library seed (``peak_gemm_tflops_fp16``).  A measured achieved rate and a datasheet PEAK are not the same
    quantity (5090: 203 achieved of 419 peak): the table is MEASURED only when EVERY card has a measured rate; otherwise ALL
    cards are priced on the datasheet peak and the basis says ``Datenblatt/unbelegt``.  A card with neither falls back to its
    VRAM size as a weight and is named ``unbelegt (keine Rate)``.
    """
    from sglang.srt.weg2 import card_identity as ci

    lib = library
    if lib is None:
        # the measured library of the rig first (card_rate_pass.load_measured_library: None = no pass was run), the seed-only
        # CardLibrary only as the datasheet source below
        try:
            from sglang.srt.planner.card_rate_pass import load_measured_library
            lib = load_measured_library()
        except Exception:  # noqa: BLE001 - no loader / unreadable file: the seed library below names the gap
            lib = None
        if lib is None:
            try:
                from sglang.srt.planner.card_library import CardLibrary
                lib = CardLibrary()
            except Exception:  # noqa: BLE001 - no library: the VRAM fallback below names the gap
                lib = None
    meas: List[Optional[float]] = []
    src: List[str] = []
    for c in cards:
        v, s = None, ""
        if c.get("tflops") is not None and str(c.get("tflops_src", "")) == "gemessen":
            v, s = float(c["tflops"]), "gemessen (Hardwareprofil)"
        elif measured:
            nm = ci.model_name(c.get("name", ""))
            for key in ("%s %dGB" % (nm, round(int(c["total_mib"]) / 1024.0)), nm):
                if measured.get(key):
                    v, s = float(measured[key]), "gemessen (%s)" % key
                    break
        if v is None and lib is not None:
            # the MEASURED GEMM rate of the library row (spec.gemm_tflops, written by card_rate_pass), not the datasheet peak
            try:
                spec = lib.resolve(str(c.get("name", "")), int(c["total_mib"]))
                g = getattr(spec, "gemm_tflops", None)
                if g:
                    v, s = float(g), "gemessen (card_library)"
            except Exception:  # noqa: BLE001 - UncalibratedCard / CardCapacityMismatch: no library row for this card
                pass
        meas.append(v)
        src.append(s)
    if all(v is not None for v in meas) and meas:
        return [float(v) for v in meas], src, "gemessen: GEMM-Rate je Karte"
    # datasheet basis for ALL cards
    peaks: List[Optional[float]] = []
    for c in cards:
        p = None
        if lib is not None:
            try:
                spec = lib.resolve(str(c.get("name", "")), int(c["total_mib"]))
                p = spec.peak_gemm_tflops_fp16
            except Exception:  # noqa: BLE001 - UncalibratedCard / CardCapacityMismatch: no datasheet row
                p = None
        if p is None and c.get("peak_tflops"):
            p = float(c["peak_tflops"])
        peaks.append(p)
    if all(p is not None for p in peaks) and peaks:
        return [float(p) for p in peaks], ["Datenblatt/unbelegt (FP16-Spitze)"] * len(peaks), (
            "Datenblatt/unbelegt: FP16-Spitzenrate je Karte (%s)" % (
                "nicht jede Karte hat eine Messrate" if any(m is not None for m in meas) else "keine Messrate"))
    return ([float(c["total_mib"]) for c in cards], ["unbelegt (keine Rate)"] * len(cards),
            "unbelegt: keine GEMM-Rate und keine Datenblattspitze, die VRAM-Groesse ersetzt die Rate")


def split_layers(n_layers: int, weights: Sequence[float], caps: Sequence[Optional[int]]) -> Tuple[List[int], List[str]]:
    """The P layer cut seed: ``n_layers`` over the stages proportional to ``weights`` (the compute rates), every stage at
    least 1 layer and at most ``caps[i]`` (the memory capacity in layers; None = no cap).  Water-filling: a stage over its cap
    is fixed at the cap and the rest is split over the others.  Returns ``(layers, notes)``; ``notes`` names an infeasible
    case (``caps`` add up to less than ``n_layers``: the surplus is put on the stage with the most spare cap anyway, and the
    note says so -- the verdict FIT is ``nein`` there)."""
    k = len(weights)
    notes: List[str] = []
    capv = [max(1, int(c)) if c is not None else n_layers for c in caps]
    fixed: Dict[int, int] = {}
    free = list(range(k))
    remaining = n_layers
    for _ in range(2 * k + 2):
        if not free:
            break
        w = [max(1e-9, float(weights[i])) for i in free]
        alloc = largest_remainder(max(0, remaining), w)
        # a stage over its cap is fixed at the cap; a stage that rounds to 0 is fixed at the 1 layer every stage holds
        over = [(i, capv[i]) for i, a in zip(free, alloc) if a > capv[i]]
        under = [(i, 1) for i, a in zip(free, alloc) if a < 1 and (i, capv[i]) not in over]
        if not over and not under:
            for i, a in zip(free, alloc):
                fixed[i] = a
            free = []
            break
        for i, v in over + under:
            fixed[i] = v
            remaining -= v
            free.remove(i)
    layers = [fixed.get(i, 1) for i in range(k)]
    diff = n_layers - sum(layers)
    if diff != 0:
        notes.append("memory capacity of the stages (%s layers) does not hold %d layers: surplus %d placed on the stage "
                     "with the most spare capacity" % (csv([c if c < n_layers else "-" for c in capv]), n_layers, diff))
        spare = sorted(range(k), key=lambda i: (-(capv[i] - layers[i]), i))
        layers[spare[0]] += diff
    return layers, notes


def attn_counts(families: Sequence[str], layers: Sequence[int]) -> List[int]:
    """The full-attention layers of each stage of the contiguous cut ``layers`` over the family list (``"attn"`` / ``"gdn"``)."""
    out, pos = [], 0
    for n in layers:
        out.append(sum(1 for f in families[pos:pos + n] if f == "attn"))
        pos += n
    return out


# ---------------------------------------------------------------------------
# a per-card MEASURED vector of the profile, put onto another inventory
# ---------------------------------------------------------------------------

def class_rekey(values: Sequence[str], calibrated: Sequence[str], live: Sequence[Mapping[str, Any]]
                ) -> Tuple[List[str], List[str]]:
    """The vector ``values`` (one entry per card of the CALIBRATED inventory, class labels in card order) for the live cards.

    Per live card, in this order: the MAXIMUM over the calibrated cards of its OWN class (the conservative direction of
    ``inventory_view.CLASS_MAX``: books more, never an average, never another class); else the class of its arch twin
    (:data:`ARCH_TWIN`, ``HW-BORROWED``), the maximum over THAT class; else the maximum of every entry (``unbelegt``).
    Returns ``(new values, source per card)``; the source is ``"Klasse X"``, ``"geborgt: <twin> (unbelegt)"`` or
    ``"unbelegt: Maximum aller Karten"``."""
    def key(x: str) -> float:
        return float(x)

    out: List[str] = []
    srcs: List[str] = []
    for c in live:
        cls = c.get("class")
        same = [values[i] for i, k in enumerate(calibrated) if k == cls]
        if same:
            out.append(max(same, key=key))
            srcs.append("Klasse %s" % cls)
            continue
        twin = ARCH_TWIN.get(str(c.get("arch")))
        tw = [values[i] for i, k in enumerate(calibrated) if k == twin]
        if twin and tw:
            out.append(max(tw, key=key))
            srcs.append("geborgt: %s (unbelegt)" % twin)
            continue
        out.append(max(values, key=key))
        srcs.append("unbelegt: Maximum aller Karten")
    return out, srcs


def role_rekey(values: Sequence[str], n: int) -> List[str]:
    """A per-STAGE/rank-role vector (``inventory_view.ROLE``): first, last, and the middle one repeated; a length-N input is
    returned as it is."""
    vals = list(values)
    if len(vals) == n:
        return vals
    if n < 2 or len(vals) < 2:
        return [vals[0]] * n
    mid = vals[1] if len(vals) > 2 else vals[-1]
    return [vals[0]] + [mid] * (n - 2) + [vals[-1]]


def scale_by_seats(values: Sequence[str], seats: int, seats_ref: int) -> List[str]:
    """A seat-dependent vector (``SGLANG_MOE_SCRATCH_SLOTS``) linear in the seat count, rounded up, at least 1."""
    if not seats_ref or seats == seats_ref:
        return list(values)
    return [str(max(1, int(math.ceil(float(v) * seats / seats_ref)))) for v in values]


# ---------------------------------------------------------------------------
# K1: the fit terms (hw_fit) from the model profile
# ---------------------------------------------------------------------------

def _v(node: Any, default: Any = None) -> Any:
    """Value of a ``flliper.model/1`` node ``{"v":..,"src":..}`` or a plain scalar."""
    if isinstance(node, Mapping) and "v" in node:
        return node["v"]
    return default if node is None else node


def fit_profile_from_model(modell: Mapping[str, Any], draft: Optional[Mapping[str, Any]] = None, *,
                           name: str = "propose") -> Any:
    """A ``hw_fit.FitProfile`` made of the ``flliper.model/1`` profile (weights per layer from the safetensors headers,
    experts, KV cell, mamba state) and the optional separate draft profile (``model_profile.estimate_draft``): the draft
    counts with the bytes the D host keeps (``bytes_without_embed_lm_head``: D shares the target's embedding and head)."""
    from sglang.srt.weg2 import hw_fit

    arch = modell["arch"]
    w = modell["weights"]
    fams = tuple("attn" if f == "attn" else "gdn" for f in _v(arch["layer_families"]))
    dense = tuple(round(b / MIB, 4) for b in _v(w["layer_bytes"]))
    expert = tuple(round(b / MIB, 4) for b in _v(w.get("layer_expert_bytes"), [0] * len(dense)))
    n_exp = int(_v((modell.get("experts") or {}).get("n"), 0) or 0)
    kvv = (modell.get("kv") or {}).get("variants") or {}
    kv: Dict[str, float] = {}
    for dst, srcname in (("fp8_e4m3", "fp8_e4m3"), ("bf16", "auto")):
        cell = _v(((kvv.get(srcname) or {}).get("cell_bytes_per_attn_layer_token")))
        if cell is not None:
            kv[dst] = float(cell)
    if "bf16" not in kv and "fp8_e4m3" in kv:
        kv["bf16"] = kv["fp8_e4m3"] * 2.0
    st = modell.get("state") or {}
    draft_mib = 0.0
    if draft and draft.get("bytes_without_embed_lm_head"):
        draft_mib = float(_v(draft["bytes_without_embed_lm_head"])) / MIB
    elif draft and draft.get("total_bytes"):
        draft_mib = float(_v(draft["total_bytes"])) / MIB
    elif (modell.get("draft") or {}).get("mtp_bytes"):
        draft_mib = float(_v(modell["draft"]["mtp_bytes"])) / MIB
    ple = (w.get("ple") or {})
    disk = float(sum(_v(ple.get("per_layer_bytes"), []) or [0])) + float(_v(ple.get("ngram_table_bytes"), 0) or 0)
    return hw_fit.FitProfile(
        profile=name, weight_format=str(_v(modell.get("format"), "")), derived_from=str(modell.get("config_path", "")),
        n_layers=len(fams), layer_families=fams, layer_dense_mib=dense, layer_expert_mib=expert, n_experts=n_exp,
        embed_mib=round(float(_v(w.get("embed_bytes"), 0)) / MIB, 4),
        lm_head_mib=round(float(_v(w.get("lm_head_bytes"), 0)) / MIB, 4), draft_mib=round(draft_mib, 4),
        visual_mib=round(float(_v(w.get("visual_bytes"), 0)) / MIB, 4), disk_mib=round(disk / MIB, 4),
        kv_bytes_per_token_per_attn_layer=kv,
        mamba_mib_per_slot_per_linear_layer=float(_v(st.get("per_linear_layer_per_slot_mib"), 0.0) or 0.0),
        extend_rate_mib_per_row=_v((modell.get("activation") or {}).get("extend_rate_mib_per_row")))


def stage_budgets(p: Any, fit_cards: Sequence[Any], asm: Any, argv: Sequence[str], *, records_profile: str,
                  verdict: Any) -> Dict[str, Any]:
    """The P stage terms of the ``hw_fit`` bound, per card in launcher order (the SAME arithmetic as ``hw_fit._check_p``,
    kept visible here because the cut seed needs the pieces, not only the verdict):

    ``avail``  MiB the stage may fill = total - residue(class) - fixed(role) - activation(role) (records of
               ``records_profile``; a record of another profile is marked BORROWED in ``verdict.marks``),
    ``cost``   per-layer cost list at the LRU-row floor (dense + LRU expert rows + the layer's KV + mamba slots),
    ``mean``   the mean of ``cost``, ``caps`` the layers each stage holds at that mean."""
    from sglang.srt.weg2 import hw_fit

    n = len(fit_cards)
    posts = hw_fit._Posts(records_profile, verdict)           # noqa: SLF001 - the one reader of the record posts
    fixed = posts.vec("P_PP_STAGE_FIXED_MIB")
    act = posts.vec("P_ACTIVATION_MIB")
    costs = hw_fit.layer_costs_mib(p, asm, argv, mamba=True, slot_mib=hw_fit.mamba_slot_mib(p, posts))
    mean = sum(costs) / len(costs)
    avail, caps = [], []
    for i, c in enumerate(fit_cards):
        a = c.total_mib - hw_fit.residue_mib(c, asm, verdict)
        a -= hw_fit.role_value(fixed, i, n) + hw_fit.role_value(act, i, n)
        avail.append(a)
        caps.append(int(max(0.0, a) // mean) if mean > 0 else None)
    return {"avail": avail, "cost": costs, "mean": mean, "caps": caps, "posts": posts}


# ---------------------------------------------------------------------------
# K4 (MoE): the resident expert fraction a P stage can afford
# ---------------------------------------------------------------------------

def fr_p(p: Any, layers: Sequence[int], avail: Sequence[float], costs: Sequence[float]) -> Tuple[List[float], List[str]]:
    """``--pp-cut-expert-device-fraction`` per P stage: after the stage's layers cost their FLOOR (dense + LRU rows + KV +
    mamba, ``costs``) the rest of ``avail`` buys resident experts; ``FR = spare / (the stage's expert MiB)``, clamped to
    ``[0, (E-2)/E]`` (the launcher's largest offload fraction) and rounded DOWN to 3 decimals (never promise a row that is
    not paid).  Dense models have no expert fraction: returns ``([], [])``."""
    if not p.is_moe or not p.n_experts:
        return [], []
    fmax = (p.n_experts - FR_MAX_EXPERT_MARGIN) / float(p.n_experts)
    out, notes, pos = [], [], 0
    for i, n in enumerate(layers):
        floor = sum(costs[pos:pos + n])
        exp = sum(p.layer_expert_mib[pos:pos + n])
        pos += n
        spare = avail[i] - floor
        f = 0.0 if exp <= 0 else max(0.0, min(fmax, spare / exp))
        out.append(math.floor(f * 1000.0) / 1000.0)
        if spare < 0:
            notes.append("stage %d: the layers' floor (%.0f MiB) exceeds what the card offers (%.0f MiB): FR 0" % (
                i, floor, avail[i]))
    return out, notes


def mamba_slots_p(seats: int) -> int:
    """Mamba slots one P stage holds for ``seats`` decode seats (``4 per seat + 8 retention``)."""
    return P_MAMBA_SLOTS_PER_SEAT * int(seats) + P_MAMBA_RETENTION


def mamba_slots_d(seats: int) -> int:
    """Mamba slots D holds for ``seats`` (reference 38 at 6 seats, linear: ONE measured point -> unbelegt)."""
    return max(1, int(round(D_MAMBA_SLOTS_REF * int(seats) / float(D_MAMBA_SEATS_REF))))


# ---------------------------------------------------------------------------
# K4/K2 (MoE): the Form A decode layout
# ---------------------------------------------------------------------------

def link_rate(card: Mapping[str, Any]) -> Tuple[float, str]:
    """Relative host->card transfer rate of a card (the weight of the Form A spill split): the MEASURED h2d rate when the
    hardware profile has one, else the NOMINAL PCIe payload (generation x lanes, datasheet), else 1.0.  Returns ``(rate,
    source)``; only ratios between cards are used."""
    if card.get("h2d_gbs"):
        return float(card["h2d_gbs"]), "gemessen"
    g, w = card.get("pcie_max_gen"), card.get("pcie_max_width")
    if g and w and int(g) in PCIE_GBS_PER_LANE:
        return PCIE_GBS_PER_LANE[int(g)] * int(w), "Datenblatt (PCIe-Nennwert, unbelegt)"
    return 1.0, "unbelegt (keine PCIe-Angabe, gleiche Rate angenommen)"


def draft_placement(p: Any, host_budget_mib: float, host_fixed_mib: float, kv_mib: float, mamba_mib: float,
                    draft_mib: float) -> Dict[str, Any]:
    """K3: ``solo`` (``--speculative-draft-placement solo``: the draft runs unsharded on rank 0, the attention host) when
    draft + dense weights + the KV obligation + mamba fit the host's budget, else ``split`` (the launcher default: the draft
    sharded over all ranks; Form A REFUSES it, ``server_args.py:7889``) with the missing MiB.  Returns ``{placement,
    need_mib, budget_mib, margin_mib, why}``."""
    need = host_fixed_mib + draft_mib + kv_mib + mamba_mib
    margin = host_budget_mib - need
    return {"placement": "solo" if margin >= 0 else "split", "need_mib": need, "budget_mib": host_budget_mib,
            "margin_mib": margin,
            "why": ("draft %.0f + dense %.0f + KV %.0f + mamba %.0f = %.0f MiB %s the host budget %.0f MiB (margin %.0f)" % (
                draft_mib, host_fixed_mib, kv_mib, mamba_mib, need, "fit" if margin >= 0 else "exceed", host_budget_mib,
                margin))}


def form_a_d(p: Any, cards: Sequence[Mapping[str, Any]], budgets_mib: Sequence[int], reserves_mib: Sequence[int], *,
             kv_tokens: int, kv_cell_bytes: float, draft_mib: float, d_mamba_slots: int, mamba_slot_mib: float
             ) -> Dict[str, Any]:
    """The Form A D layout (``form_a_plan.solve_form_a``): rank 0 = attention host with the dense weights, the undivided KV,
    the draft and the GDN states; every other rank is a pure expert worker.  ``budgets_mib`` = what each rank may use
    (``--rank-gpu-memory-mib`` semantics: total minus the D context), ``reserves_mib`` the user reserve.

    The runtime posts of the plan (host runtime 3.20 GiB, worker runtime 1.45, corridor 1.90, dispatch buffer 0.20, router
    0.12, speculative state 0.15) are the measured posts of boot fn8ah (``MeasuredPosts.from_fn8ah``): BORROWED for every
    model and card, named in ``unbelegt``.  Returns ``{ok, error, role, tp_ratio, moe_ratio, fr, capacity, owned, unbelegt}``."""
    from sglang.srt import form_a_plan as FA

    n = len(cards)
    ref = FA.MeasuredPosts.from_fn8ah(context_tokens=int(kv_tokens))
    dense_gib = (p.resident_dense_mib()) / 1024.0
    attn = p.attn_layers
    posts = FA.MeasuredPosts(
        dense_gib=dense_gib, kv_bytes_per_token=int(round(attn * kv_cell_bytes)), context_tokens=int(kv_tokens),
        draft_gib=draft_mib / 1024.0, gdn_state_gib=p.linear_layers * mamba_slot_mib * d_mamba_slots / 1024.0,
        spec_state_gib=ref.spec_state_gib, host_runtime_gib=ref.host_runtime_gib, worker_runtime_gib=ref.worker_runtime_gib,
        corridor_gib=ref.corridor_gib, dispatch_buffer_gib=ref.dispatch_buffer_gib, worker_router_gib=ref.worker_router_gib,
        source="model profile (dense/KV/draft/gdn) + boot fn8ah runtime posts (borrowed)")
    exp_layers = sum(1 for e in p.layer_expert_mib if e > 0)
    exp_mib = (sum(p.layer_expert_mib) / exp_layers / p.n_experts) if exp_layers and p.n_experts else 0.0
    geom = FA.ExpertGeometry(num_experts=p.n_experts, num_layers=exp_layers,
                             expert_bytes=int(round(exp_mib * MIB)), pad_experts_per_rank=1,
                             source="model profile (bytes per expert from the safetensors headers)")
    cb = [FA.CardBudget(rank=i, name=str(c.get("name", "")), nameplate_mib=int(c["total_mib"]), budget_mib=int(budgets_mib[i]),
                        reserve_mib=int(reserves_mib[i]), link_gib_s=link_rate(c)[0], role="host" if i == 0 else "worker")
          for i, c in enumerate(cards)]
    unb = ["runtime posts of Form A (host %.2f / worker %.2f GiB, corridor %.2f, dispatch %.2f, router %.2f, spec state %.2f) "
           "are the measured posts of boot fn8ah: BORROWED, unbelegt for this model and these cards" % (
               ref.host_runtime_gib, ref.worker_runtime_gib, ref.corridor_gib, ref.dispatch_buffer_gib, ref.worker_router_gib,
               ref.spec_state_gib)]
    try:
        plan = FA.solve_form_a(cb, posts, geom)
    except FA.FormAInfeasible as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc), "unbelegt": unb}
    return {"ok": True, "error": "", "role": ["host"] + ["worker"] * (n - 1), "tp_ratio": [1] + [0] * (n - 1),
            "moe_ratio": [int(o) for o in plan.owned],
            "fr": [round(math.floor(f * 1000.0) / 1000.0, 3) for f in plan.resident_fraction],
            "capacity": [int(c) for c in plan.capacity], "owned": [int(o) for o in plan.owned],
            "host_breakdown": dict(plan.host_breakdown), "unbelegt": unb}


# ---------------------------------------------------------------------------
# K4 (dense): TP shares of the decode group
# ---------------------------------------------------------------------------

def dense_d_shares(budgets_mib: Sequence[float]) -> List[float]:
    """The tensor-parallel D shares of a dense model: proportional to each card's VRAM budget (``--d-tp-objective maxkv``
    asks the launcher for exactly this family of splits; the launcher solves the exact ratios at boot)."""
    s = float(sum(budgets_mib))
    return [round(float(b) / s, 4) for b in budgets_mib]
