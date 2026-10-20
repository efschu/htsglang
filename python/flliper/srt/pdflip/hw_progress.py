"""HW-AP0 1525: the PROGRESS METER of the hardware-generic work -- ONE command
that prints, for every example configuration x every release model, whether the
configuration FITS by calculation and what the FIRST refusal is.

    python -m flliper.srt.pdflip.hw_progress            # (or tools/hw_progress.py)
    python -m flliper.srt.pdflip.hw_progress --models NF,27B-INT8 --json out.json

Per cell:

* ``passt``  -- the :mod:`hw_fit` verdict computed from the model PROFILE
  (layers, KV heads x head dim, quant bytes, MoE/dense, draft) + the profile's
  records + the launch argv: ``ja`` / ``tight`` (only the mamba slots push it
  over) / ``nein``. ``Profil fehlt`` when the model has no fit profile.
* ``erste Sperre`` -- the first refusal of the launcher's own pre-spawn
  hardware path (:func:`hw_sim.simulate`: arch gate, topology/count,
  calibration inventory, vectors, fit), or ``läuft``.

User orders 05.10.: the work must make every combination of sm86 and sm120
cards plan and start ("2-4x3090, 3070+5070+3090+5090") AND is not tied to one
model ("nicht das int8 oder das nf ... gguf ... nvfp4 nur im flip modus ohne
dual"). A model is a ROW of this table, not a branch of the code: add a model
by a profile (``fit_profiles_data/``) and a :data:`hw_sim.MODELS` row.

The rig can test only the two REFERENCE rows (NF on 5090 + 2x3080, 27B INT8 on
5090 + 3080); everything else is a calculation and the cards marked
HW-BORROWED are not backed by any measurement. The meter counts how many cells
no longer refuse -- the goal of the work; it never claims a configuration
RUNS (HOCHRECHNUNG != MESSUNG).

CPU only, no GPU, no NVML.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import msgspec

#: (label, catalog keys in NVML order, ``--cards`` selection or None, reference?)
Config = Tuple[str, Tuple[str, ...], Optional[Tuple[int, ...]], bool]

EXAMPLE_CONFIGS: Tuple[Config, ...] = (
    ("REF 5090+2x3080 (N=3)", ("3080-20G", "5090", "3080-20G"), None, True),
    ("REF 27B 5090+3080 (N=2)", ("3080-20G", "5090", "3080-20G"), (1, 0), True),
    ("REF 3080+3080 (N=2)", ("3080-20G", "5090", "3080-20G"), (0, 2), True),
    ("2x3090", ("3090",) * 2, None, False),
    ("3x3090", ("3090",) * 3, None, False),
    ("4x3090", ("3090",) * 4, None, False),
    ("2x5090", ("5090",) * 2, None, False),
    ("5090+2x3090", ("5090", "3090", "3090"), None, False),
    ("3070+5070+3090+5090", ("3070", "5070", "3090", "5090"), None, False),
    ("5070Ti+3090", ("5070Ti", "3090"), None, False),
    ("2x4090", ("4090",) * 2, None, False),
    ("3x3080-10G", ("3080-10G",) * 3, None, False),
    ("3x3070", ("3070",) * 3, None, False),
    ("3x5070Ti", ("5070Ti",) * 3, None, False),
)

#: models of the meter, in table order (keys of :data:`hw_sim.MODELS`).
PROGRESS_MODELS: Tuple[str, ...] = ("NF", "27B-INT8", "27B-FP8", "27B-NVFP4", "27B-GGUF", "27B-NVFP4-DUAL")

MODEL_TITLES: Dict[str, str] = {
    "NF": "NF-int4 (Qwen3.8-Flash-Next, MoE, Draft, Flip)",
    "27B-INT8": "27B INT8 (dense, Draft, Flip)",
    "27B-FP8": "27B FP8 (dense, Flip)",
    "27B-NVFP4": "27B NVFP4 (dense, Flip OHNE Dual)",
    "27B-GGUF": "27B GGUF IQ4_XS (dense, Flip)",
    "27B-NVFP4-DUAL": "27B NVFP4 DUAL (P+D gleichzeitig)",
}


class Row(msgspec.Struct, kw_only=True):
    model: str
    config: str
    n: int
    reference: bool
    fit: str                      # ja | knapp | nein | Profil fehlt
    fit_first: str = ""
    result: str = ""              # läuft | verweigert
    code: str = ""
    first_block: str = ""
    blockers: List[str] = []
    borrowed: List[str] = []      # cards / posts that no measurement backs
    unmodelled: List[str] = []    # posts the model's records do not carry / modes not modelled


def _card_flags(keys: Sequence[str], selection: Optional[Sequence[int]]) -> List[str]:
    from flliper.srt.pdflip import hw_sim as HS

    live = [keys[i] for i in selection] if selection is not None else list(keys)
    out: List[str] = []
    for k in dict.fromkeys(live):
        c = HS.CATALOG[k]
        if c.borrowed:
            out.append(f"{k}: HW-BORROWED/unverified")
        if "sm89" in c.evidence:
            out.append(f"{k}: sm89 am Metall nie gemessen")
    return out


def rows_for(models: Sequence[str] = PROGRESS_MODELS, configs: Sequence[Config] = EXAMPLE_CONFIGS,
             table: Optional[Mapping[str, object]] = None, asm=None) -> List[Row]:
    from flliper.srt.pdflip import hw_sim as HS

    table = table or HS.MODELS
    out: List[Row] = []
    for mk in models:
        for label, keys, sel, ref in configs:
            c = HS.simulate(label, keys, table[mk], sel, fit_asm=asm)
            fit = c.fit
            borrowed = _card_flags(keys, sel)
            unmodelled: List[str] = []
            if fit is not None:
                borrowed += [m.replace("HW-BORROWED/unverified: ", "") for m in fit["marks"]
                             if "BORROWED" in m and "layer geometry" not in m]
                unmodelled = [m for m in fit["marks"] if "no record" in m or "NOT modelled" in m or "unreadable" in m
                              or "UNVERIFIED" in m]
            out.append(Row(
                model=mk, config=label, n=c.n_cards, reference=ref,
                fit="Profil fehlt" if fit is None else str(fit["level"]),
                fit_first="" if fit is None else str(fit["first"]),
                result=c.result, code=c.code, first_block=(c.blockers[0] if c.blockers else ""),
                blockers=list(c.blockers), borrowed=borrowed, unmodelled=unmodelled))
    return out


def golden_of(rows: Sequence[Row]) -> Dict[str, List[object]]:
    """The comparable core: ``config | model`` -> [fit, result, code, first blocker]."""
    return {f"{r.config} | {r.model}": [r.fit, r.result, r.code, r.first_block] for r in rows}


def first_refusal(r: Row) -> str:
    if r.result == "läuft":
        return "läuft"
    return f"{r.code} ({r.first_block})" if r.first_block and r.first_block != r.code else r.code


def profile_line(model_key: str, table: Optional[Mapping[str, object]] = None) -> str:
    from flliper.srt.pdflip import hw_fit as HF
    from flliper.srt.pdflip import hw_sim as HS

    m = (table or HS.MODELS)[model_key]
    p = HF.load_profile(m.profile)
    title = MODEL_TITLES.get(model_key, model_key)
    if p is None:
        return f"{title}: Profil fehlt (kein fit_profiles_data/{m.profile}.json)"
    scaled = m.ckpt_mib is not None and m.weight_format != p.weight_format
    moe = f"{p.n_experts} Experten (Host-Store, FR=0-Untergrenze)" if p.is_moe else "dicht"
    kv = p.kv_bytes_per_token_per_attn_layer[HF.KV_DTYPE_DEFAULT]
    return (f"{title}: {p.n_layers} Layer ({p.attn_layers} Attention, KV {kv:.0f} B/Token/Layer fp8), {moe}, "
            f"Draft {p.draft_mib:.0f} MiB, Platte-only {p.disk_mib:.0f} MiB, Format {m.weight_format}"
            + (f" [HW-BORROWED/unverified: Geometrie des {p.weight_format}-Profils, Bytes auf ckpt {m.ckpt_mib} MiB skaliert]"
               if scaled else f" [Profil aus {p.derived_from.split(' ')[0]}]")
            + (", Dual" if "--dual-share" in m.argv else ", Flip"))


def render(rows: Sequence[Row], table: Optional[Mapping[str, object]] = None) -> str:
    out: List[str] = []
    for mk in dict.fromkeys(r.model for r in rows):
        sub = [r for r in rows if r.model == mk]
        out.append(f"### {profile_line(mk, table)}")
        unmod = list(dict.fromkeys(m for r in sub for m in r.unmodelled))
        if unmod:
            out.append("Nicht modelliert / ohne Posten: " + "; ".join(unmod))
        out.append("| Konfiguration | N | passt rechnerisch | erste Sperre | weitere Sperren | HW-BORROWED / unverified |")
        out.append("|---|---|---|---|---|---|")
        for r in sub:
            rest = [b for b in r.blockers if b != r.first_block]
            passt = r.fit if not r.fit_first or r.fit == "ja" else f"{r.fit}: {r.fit_first}"
            out.append(f"| {r.config} | {r.n} | {passt} | {first_refusal(r)} | {', '.join(rest) or '-'} | "
                       f"{'; '.join(r.borrowed) or '-'} |")
        out.append("")
        out.append(model_summary(sub))
        out.append("")
    out.append(total_summary(rows))
    return "\n".join(out)


def model_summary(sub: Sequence[Row]) -> str:
    n = len(sub)
    fits = sum(1 for r in sub if r.fit in ("ja", "tight"))
    free = sum(1 for r in sub if r.result == "läuft")
    return f"FORTSCHRITT {sub[0].model}: passt rechnerisch {fits}/{n}, ohne Sperre (laeuft) {free}/{n}"


def total_summary(rows: Sequence[Row]) -> str:
    n = len(rows)
    fits = sum(1 for r in rows if r.fit in ("ja", "tight"))
    free = sum(1 for r in rows if r.result == "läuft")
    return f"HW-PROGRESS cells={n} passt={fits} laeuft={free} refused={n - free}"


def _asm_from(ns) -> "object":
    from flliper.srt.pdflip import hw_fit as HF

    residue: Dict[str, int] = {}
    for item in ns.p_residue_mib or []:
        k, _, v = item.partition("=")
        residue[k.strip()] = int(v)
    return HF.Assumptions(kv_tokens=ns.kv_tokens, kv_dtype=ns.kv_dtype, p_mamba_slots=ns.p_mamba_slots,
                          d_mamba_slots=ns.d_mamba_slots, lru_rows=ns.lru_rows, residue_mib=residue)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from flliper.srt.pdflip import hw_fit as HF
    from flliper.srt.pdflip import hw_sim as HS

    ap = argparse.ArgumentParser(
        prog="python -m flliper.srt.pdflip.hw_progress",
        description="HW-AP0: fits-by-calculation + first refusal for every example configuration x release "
                    "model (CPU only).")
    ap.add_argument("--models", default=",".join(PROGRESS_MODELS), help="models: " + ", ".join(HS.MODELS))
    ap.add_argument("--configs", default="", help="only these configuration labels (comma separated, exact)")
    ap.add_argument("--profiles-dir", default="", help="source each model's argv from the real release profiles")
    ap.add_argument("--json", default="", help="write the rows to this file")
    ap.add_argument("--write-golden", default="", help="write golden_of(rows) to this file")
    ap.add_argument("--kv-tokens", type=int, default=HF.KV_TOKENS_DEFAULT, help="KV tokens the plan must hold")
    ap.add_argument("--kv-dtype", default=HF.KV_DTYPE_DEFAULT, choices=["fp8_e4m3", "bf16"])
    ap.add_argument("--p-mamba-slots", type=int, default=HF.P_MAMBA_SLOTS_DEFAULT)
    ap.add_argument("--d-mamba-slots", type=int, default=HF.D_MAMBA_SLOTS_DEFAULT)
    ap.add_argument("--lru-rows", type=int, default=HF.LRU_ROWS_DEFAULT,
                    help="MoE LRU rows per layer when the launch argv names none")
    ap.add_argument("--p-residue-mib", action="append", metavar="CLASS=MIB",
                    help="override the P stage residue of a card class (e.g. RTX5090=3847)")
    ns = ap.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    models = [m.strip() for m in ns.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in HS.MODELS]
    if unknown:
        raise SystemExit(f"hw_progress: unknown model(s) {unknown}; models: {', '.join(HS.MODELS)}")
    configs = EXAMPLE_CONFIGS
    if ns.configs:
        want = [c.strip() for c in ns.configs.split(",") if c.strip()]
        configs = tuple(c for c in EXAMPLE_CONFIGS if c[0] in want)
        if not configs:
            raise SystemExit(f"hw_progress: no such configuration; known: {', '.join(c[0] for c in EXAMPLE_CONFIGS)}")
    table = HS.models_from_profiles(ns.profiles_dir) if ns.profiles_dir else dict(HS.MODELS)
    rows = rows_for(models, configs, table, _asm_from(ns))
    print(render(rows, table))
    if ns.json:
        with open(ns.json, "w") as fh:
            json.dump(msgspec.to_builtins(rows), fh, indent=1, ensure_ascii=False)
    if ns.write_golden:
        with open(ns.write_golden, "w") as fh:
            json.dump(golden_of(rows), fh, indent=1, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
