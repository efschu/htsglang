"""HW-GENERIC 1002: the reference rig's PLAN FINGERPRINT -- every launcher
decision the hardware-generalisation touched, computed for this rig's
inventory (nvml0 RTX 3080 20480 MiB sm86, nvml1 RTX 5090 32607 MiB sm120,
nvml2 RTX 3080 20480 MiB sm86) under every release profile row
(nextflash = nf-int4; qwen27b = 27b, 27b-row-authority-cut43,
27b-nvfp4-dual1i -- the four release forms differ by argv, the
card-dependent selectors by profile row and weight source only).

Runs UNCHANGED on the base tree (3fe878018d) and on the generic tree: it
only calls functions both trees have, and replicates the base's inline
selectors (anchor stage, DC_EXPECT) where the generic tree made a function
of them. ``python hw_generic_rig_plan_fingerprint_1002.py <out.json>`` with
PYTHONPATH=<tree>/python writes the fingerprint; the golden file
``fixtures/hw_generic_rig_plan_1002.json`` was written on the base tree and
``test_hw_generic_rig_plan_identity_1002.py`` compares the current tree to it.

GPU-free and NVML-free: the cards are hand-built (no live NVML read; the
power/current-limit readers are not part of the fingerprint because they
read the live machine).
"""

import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

PROFILES = ("nextflash", "qwen27b")
WEIGHT_SOURCES = ("exchange", "serving")
#: the D pass inputs of test_pdflip_driver_carve_rc12b (rc12b record)
FRESH = {"GPU-31d7ef41": 1320, "GPU-5c648f96": 712, "GPU-62dbbae1": 696}
#: FLLIPER_PDFLIP_L15_MIB of the 27B L15 boots (c1=7616,c2=1792)
L15_ENV = {"FLLIPER_PDFLIP_L15": "1", "FLLIPER_PDFLIP_L15_MIB": "c1=7616,c2=1792"}


def rig_cards(L):
    rows = [
        (0, "GPU-5c648f96", "NVIDIA GeForce RTX 3080", 20480, 425, (8, 6)),
        (1, "GPU-31d7ef41", "NVIDIA GeForce RTX 5090", 32607, 518, (12, 0)),
        (2, "GPU-62dbbae1", "NVIDIA GeForce RTX 3080", 20480, 425, (8, 6)),
    ]
    out = []
    for idx, uuid, name, total, res, cc in rows:
        try:
            out.append(L.Card(nvml_index=idx, uuid=uuid, name=name, total_mib=total,
                              reserved_mib=res, cc=cc))
        except TypeError:  # base tree: Card has no cc field
            out.append(L.Card(nvml_index=idx, uuid=uuid, name=name, total_mib=total,
                              reserved_mib=res))
    return out


def fingerprint():
    from flliper.srt.pdflip import form as F
    from flliper.srt.pdflip import launcher as L
    try:  # the NF y7 line carries no L15 (27B release feature) -- recorded as absent
        from flliper.srt.pdflip import l15_plan
    except ImportError:
        l15_plan = None
    from flliper.srt.pdflip import xchg_census as XC

    fp = {}
    cards = L.order_cards(rig_cards(L))
    fp["order_nvml"] = [c.nvml_index for c in cards]
    fp["order_uuid"] = [c.uuid for c in cards]
    fp["cvd"] = ",".join(c.uuid for c in cards)

    # W19 dormant residue (default profile row) per weight source
    fp["dc_measured_d_mib"] = {ws: [L.dc_measured_d_mib(c, ws) for c in cards] for ws in WEIGHT_SOURCES}
    # DC_EXPECT (base: inline `"5090" in c.name` selector)
    if hasattr(L, "dc_expect_mib"):
        fp["dc_expect_mib"] = [L.dc_expect_mib(c) for c in cards]
    else:
        fp["dc_expect_mib"] = [L.DC_EXPECT_5090_MIB if "5090" in c.name else L.DC_EXPECT_3080_MIB
                               for c in cards]
    # P-cut deep-attention anchor stage (base: inline expression)
    if hasattr(L, "attn_anchor_stage"):
        fp["attn_anchor_stage"] = L.attn_anchor_stage(cards)
    else:
        fp["attn_anchor_stage"] = next((i for i, c in enumerate(cards) if "5090" not in c.name),
                                       len(cards) - 1)
    # xchg census: the boot-named constant fallback per card
    fp["xchg_dormant_for_card"] = [list(XC.dormant_for_card(c, {}, "stem")) for c in cards]
    # P chunk model calibration power limits (builtin record, by stage class)
    fp["p_chunk_model_power_limits"] = list(L.p_chunk_model_power_limits("builtin-int8", 3))
    fp["p_stage_card_classes"] = list(L.P_STAGE_CARD_CLASSES)
    # planner presets: sm86 detection and the reference-rig calibration gate
    from flliper.srt.planner import flags as PF

    gpus = [{"name": c.name, "total_mib": c.total_mib, "memory_mib": c.total_mib} for c in cards]
    fp["planner_rig_has_sm86"] = bool(PF.rig_has_sm86(gpus))
    fp["planner_match_calibration"] = {str(q): PF._match_calibration(gpus, q) for q in ("fp8", "awq", None)}

    for prof in PROFILES:
        pf = fp.setdefault(prof, {})
        pf["budget_charges_driver_carve"] = bool(L.budget_charges_driver_carve(prof))
        pf["driver_carve_min_total_mib"] = int(L.driver_carve_min_total_mib(prof))
        g, prov = L.served_dormant_growth(cards, prof)
        lines = []
        budgets = L.budgets_from_dc(
            cards, dict(FRESH), lines.append, "D",
            overshoot_mib=[489, 0, 0], overshoot_provenance="boot weg2ls4b1",
            dormant_growth_mib=g, dormant_growth_provenance=prov,
            charge_driver_carve=L.budget_charges_driver_carve(prof),
            driver_carve_min_total_mib=L.driver_carve_min_total_mib(prof))
        pf["budgets_d"] = list(budgets)
        pf["budget_lines"] = [ln for ln in lines if ln.startswith("budget ")]
        rest = L.d_awake_rest(cards, prof)
        pf["d_awake_rest"] = [list(rest[0]) if rest[0] is not None else None, str(rest[1])]
        try:
            if l15_plan is None:
                raise ImportError("pdflip.l15_plan not in this tree")
            posts = l15_plan.resolve_posts(prof, list(budgets), [None] * len(cards), dict(L15_ENV))
            pf["l15_posts"] = [[p.card, p.mib, p.src, p.experts_rows_traded] for p in posts]
        except Exception as exc:  # noqa: BLE001 - recorded, compared like a value
            pf["l15_posts"] = f"{type(exc).__name__}: {exc}"
        ckpt = F.profile_row("qwen27b").formats["int8"].checkpoint
        if not hasattr(L, "resolve_pp_cut_stage_model"):
            # NF y7 line: PP-COST stage model (27B release 01.10.) not in this tree
            pf["pp_cut_stage_model_auto"] = "absent"
            continue
        try:
            m, why = L.resolve_pp_cut_stage_model("auto", prof, ckpt,
                                                  inventory=("RTX5090", "RTX3080", "RTX3080"))
        except TypeError:  # base tree: no inventory parameter (no card dependence)
            m, why = L.resolve_pp_cut_stage_model("auto", prof, ckpt)
        # the DECISION and, when a model is picked, its provenance; the
        # not-picked reason text names the inventory key on the generic tree
        # (a log text, not a plan input)
        pf["pp_cut_stage_model_auto"] = [m is not None, why if m is not None else "-"]
    return fp


if __name__ == "__main__":
    out = fingerprint()
    text = json.dumps(out, indent=1, sort_keys=True, default=str)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as fh:
            fh.write(text + "\n")
    else:
        print(text)
