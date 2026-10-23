"""PLANER-ZIELBILD 1021: generates ``rigdash/planer_data/steckbrief.json`` -- the model fact sheet the target-picture planner (``rigdash/planer_ziel.py``)
reads.  One row per model: the numbers come from the planner's own model profile (its ``model_profile`` module, ``flliper.model/1``: config.json +
safetensors headers), never typed by hand.  Run with the Python of the flliper environment (the dashboard process itself imports no flliper)::

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<tree>/python python tools/rig_dashboard/kartenplan_build/steckbrief_gen.py [--out FILE]

A model directory that is empty on this machine (an unmounted model) is read from the checkpoint fixture of the planner tests
(the planner-test fixture ``planer_1006/checkpoints``: config + safetensors headers of the real checkpoints); the row names which source it used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MC = "/spinning/llm_stuff/club-3090/models-cache/"
FIXTURES = os.path.join(TREE, "test", "registered", "unit", "pdflip", "fixtures", "planer_1006", "checkpoints")
MIB = float(1 << 20)

#: model id -> (checkpoint dir, draft dir or None).  The ids are the ones of ``rigdash/planer_ziel.py`` MODELS.
SOURCES = {
    "qwen38-27b-int8": ("Qwen3.8-27B-INT8-gdncov-vocabembed", "Qwen3.8-27B-DFlash2-W8-lued"),
    "qwen38-27b-nvfp4": ("Qwen3.8-27B-NVFP4-RadixArk", "Qwen3.8-27B-DFlash2-NVFP4-RTNcal"),
    "qwen38-nf-int4": ("Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp", None),
    # planer-n2fix2 1023 (A4): Huihui-Qwen3.8-27B-Abliterated INT4 W4A16 AutoRound, losslessly repacked to compressed-tensors
    # (tools/quant/autoround_gptq_to_ct.py on desk/27b-int4ar-1023); the W8 draft as in profiles_release/27b-int4ar.env
    "qwen38-27b-int4ar": ("Huihui-Qwen3.8-27B-Abliterated-INT4-W4A16-AutoRound-ct", "Qwen3.8-27B-DFlash2-W8-lued"),
}


def _v(node, default=None):
    return node.get("v", default) if isinstance(node, dict) else default


def _readable(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))


def _quant(path: str) -> dict:
    """planer-n2fix2 1023 (A4): the checkpoint's quantization as its config.json states it (no interpretation): method, the weight scheme
    of every config group (type, bits, group size, symmetric, strategy, act-order) and the module patterns it leaves unquantized."""
    with open(os.path.join(path, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    q = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config") or {}
    out = {"method": q.get("quant_method")}
    groups = []
    for g in (q.get("config_groups") or {}).values():
        w = g.get("weights") or {}
        groups.append({k: w.get(k) for k in ("type", "num_bits", "group_size", "symmetric", "strategy", "actorder")})
    if groups:
        out["weight_groups"] = groups
    if q.get("quant_algo"):
        out["algo"] = q.get("quant_algo")
    if q.get("ignore"):
        ign = list(q["ignore"])
        out["unquantized"] = ign[:12]                   # NF lists every vision module by name (hundreds): the first ones + the count
        if len(ign) > 12:
            out["unquantized_n"] = len(ign)
    if q.get("_converted_from"):
        out["converted_from"] = dict(q["_converted_from"])
    return out


def _tensor_mib(path: str, needle: str) -> float:
    """MiB of the tensors whose name contains ``needle``, from the safetensors HEADERS (no weight byte read) -- read with the planner's own
    header reader (``model_profile._read_safetensors_header``), the same bytes the model profile prices."""
    import glob

    from flliper.srt.pdflip import model_profile as MP

    tot = 0
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        for name, t in MP._read_safetensors_header(f).items():
            if not name.startswith("__") and needle in name:
                a, b = t["data_offsets"]
                tot += b - a
    return round(tot / MIB, 1)


def row(model_id: str, tmp: str) -> dict:
    from flliper.srt.pdflip import model_profile as MP
    from flliper.srt.pdflip import propose_oracle as O

    name, draft = SOURCES[model_id]
    src, path = "model directory", MC + name
    if not _readable(path):
        O.materialize_checkpoint(os.path.join(FIXTURES, name), tmp)
        src, path = "checkpoint fixture of the planner tests (config + safetensors headers)", os.path.join(tmp, name)
    est = MP.estimate_or_state(path)
    if not est.get("ok"):
        raise SystemExit("model profile of %s failed: %s" % (name, est.get("error")))
    p = est["profile"]
    a, w, kv, st, ex, dec = p["arch"], p["weights"], p["kv"], p["state"], p["experts"], p["decode"]
    row = {
        "id": model_id,
        "checkpoint": name,
        "source": src,
        "config_sha": p.get("config_sha"),
        "n_layers": _v(a.get("n_layers")),
        "attn_layers": _v(kv.get("attn_layers")),
        "linear_layers": _v(st.get("linear_layers")),
        "weights_total_mib": round(_v(w["total_bytes"]) / MIB),
        "weights_dense_mib": round(_v(w["layers_bytes_nonexpert"]) / MIB),
        "lm_head_mib": round(_v(dec["lm_head_bytes"]) / MIB),
        "kv_bytes_per_token_fp8": _v(kv["variants"]["fp8_e4m3"]["bytes_per_token_all_attn_layers"]),
        "state_mib_per_seat_bf16": round(st["variants_mib"]["bfloat16"] * _v(st["linear_layers"]), 1),
        "experts": None,
    }
    row["quant"] = _quant(path)
    row["vision_tower_mib"] = _tensor_mib(path, "visual")
    row["mtp_mib"] = round(_v(p.get("draft", {}).get("mtp_bytes"), 0) / MIB, 1) if _v(p.get("draft", {}).get("mtp_bytes")) else None
    if (_v(ex.get("n")) or 0) > 0:
        row["experts"] = {"n": _v(ex["n"]), "top_k": _v(ex.get("top_k")), "moe_layers": _v(ex.get("moe_layers")),
                          "total_mib": round(_v(ex["total_bytes"]) / MIB), "bytes_per_expert": _v(ex.get("bytes_per_expert"))}
    d_mib = None
    if draft:
        dpath = MC + draft
        if not _readable(dpath) and os.path.isdir(os.path.join(FIXTURES, draft)):
            O.materialize_checkpoint(os.path.join(FIXTURES, draft), tmp)
            dpath = os.path.join(tmp, draft)
        if _readable(dpath):
            d = MP.estimate_draft(dpath)
            d_mib = round(_v(d["total_bytes"]) / MIB)
            row["draft_checkpoint"] = draft
    elif _v(p.get("draft", {}).get("mtp_bytes")):
        d_mib = round(_v(p["draft"]["mtp_bytes"]) / MIB)
        row["draft_checkpoint"] = "MTP layers inside the checkpoint"
    row["draft_mib"] = d_mib
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "rigdash", "planer_data", "steckbrief.json"))
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="steckbrief-") as tmp:
        rows = [row(m, tmp) for m in SOURCES]
    doc = {"schema": "flliper.steckbrief/1",
           "quelle": "the planner's model profile (flliper.model/1) over config.json + safetensors headers; generated by tools/rig_dashboard/kartenplan_build/steckbrief_gen.py",
           "einheit": "MiB (2^20 bytes) unless a field name says bytes",
           "models": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("wrote", os.path.abspath(args.out), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
