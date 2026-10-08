#!/usr/bin/env python3
"""PROFIL-EDITOR S3 (Auftrag 960): Abgleich der Modellschätzung gegen die vorhandenen Zeilen und gemessenen Größen.

    tools/profil_s3_abgleich.py [--root /spinning/llm_stuff/club-3090/models-cache]

Liest nur Kopfzeilen (Safetensors-/GGUF-Köpfe, config.json) und die Records/Handzeilen des Baums; keine GPU, kein Gewicht.
Ausgabe: Markdown-Tabellen (A Köpfe gegen Dateien, B nur Config gegen Köpfe, C Stufensummen gegen Boot-Log, D Geometrieterme
gegen Records, E Registry-Ableitung gegen die Handzeilen).  ``--root`` darf ein Verzeichnis aus Kopf-Schnappschüssen sein
(``<root>/<modell>/`` mit abgeschnittenen Shards und ``_ondisk.json`` = {Datei: Größe}).
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import form as F  # noqa: E402
from flliper.srt.pdflip import model_profile as MP  # noqa: E402

GIB = float(1 << 30)
MC = "/spinning/llm_stuff/club-3090/models-cache/"

#: (Etikett, Verzeichnis unter --root, Registry-Zeile oder None)
MODELS = (
    ("27B INT8 gdncov-vocabembed", "Qwen3.8-27B-INT8-gdncov-vocabembed", "qwen27b"),
    ("27B INT8 gdncov", "Qwen3.8-27B-INT8-gdncov", None),
    ("27B FP8", "Qwen3.8-27B-FP8", None),
    ("27B NVFP4 RadixArk", "Qwen3.8-27B-NVFP4-RadixArk", None),
    ("27B GGUF UD-IQ4_XS", "Qwen3.8-27B-GGUF-unsloth__Qwen3.8-27B-UD-IQ4_XS.gguf", None),
    ("NF INT4-mixed Minachist", "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist", "nextflash"),
    ("NF INT4-mixed abl-wxp", "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp", None),
    ("NF NVFP4 nvidia", "Qwen3.8-Flash-Next-NVFP4-nvidia", None),
)
#: Gesamtbytes der Checkpoints, die der Dashboard-Kartenplaner per stat aufgezeichnet hat (kartenplan_data/*.json ``weights``)
STAT_RECORDED = {"Qwen3.8-27B-FP8": 30866866928, "Qwen3.8-27B-NVFP4-RadixArk": 21921697280,
                 "Qwen3.8-27B-GGUF-unsloth__Qwen3.8-27B-UD-IQ4_XS.gguf": 14252845984}
#: ``Load weight end ... mem usage=X GB`` (X in GiB) der Hauptmodell-Zeilen je PP-Stufe
BOOT_STAGES = (
    ("27B INT8 gdncov-vocabembed, Schnitt 41,12,11", "Qwen3.8-27B-INT8-gdncov-vocabembed", [41, 12, 11], None, [15.86, 4.34, 6.41],
     "boot_weg2_dkr27browauthoritybar1fs10031930_76f5bbde19_1003_193056.P.log 19:31:32-39"),
    ("NF INT4-mixed abl, Schnitt 29,11,8, Expertenzeilen 201/391/512", "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp",
     [29, 11, 8], [201, 391, 512], [16.91, 11.13, 11.02],
     "boot_weg2_dkrnfint4h6ablbar1dauer10031352_5444cf8cde_1003_135248.P.log 13:53:50-13:54:20"),
)


def pct(a, b):
    return "%+.2f %%" % (100.0 * (a - b) / b) if b else "n/a"


def load(root, d):
    p = os.path.join(root, d)
    if os.path.isdir(p) or os.path.isfile(p):
        return p
    return None


def ondisk_shards(path):
    f = os.path.join(path, "_ondisk.json")
    if os.path.isfile(f):                                    # Kopf-Schnappschuss: die wahren Größen stehen daneben
        with open(f) as fh:
            ond = json.load(fh)
        return sum(v for k, v in ond.items() if k.endswith((".safetensors", ".gguf")))
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=MC)
    a = ap.parse_args(argv)
    est = {}
    print("## A. Gewichtsbytes aus den Köpfen gegen die Dateien auf der Platte\n")
    print("| Modell | Format | Tensorsumme (Index) | + Köpfe | Shards auf Platte | Abweichung | stat (Kartenplaner-Record) |")
    print("|---|---|---:|---:|---:|---:|---|")
    for label, d, _ in MODELS:
        p = load(a.root, d)
        if not p:
            print("| %s | nicht gefunden | | | | | |" % label)
            continue
        e = MP.estimate(p)
        est[d] = e
        w = e["weights"]
        hdr = w.get("header_bytes", {}).get("v", 0)
        disk = ondisk_shards(p) or w.get("disk_bytes", {}).get("v")
        tot = w["total_bytes"]["v"]
        if e["weights_source"]["v"] == "gguf":
            hdr = 0
        st = STAT_RECORDED.get(d)
        print("| %s | %s | %d | %d | %s | %s | %s |" % (label, e["format"]["v"], tot, hdr, disk, pct(tot + hdr, disk) if disk else "n/a",
                                                       ("%d (%s)" % (st, pct(tot + hdr, st)) if st else "-")))
    print("\nGGUF: der Kopf trägt Metadaten (Tokenizer) vor den Tensordaten; die Abweichung dort ist dieser Kopf (kein Gewicht).\n")

    print("## B. Nur Config + Quantisierung (kein Tensorverzeichnis) gegen die Köpfe\n")
    print("| Modell | Format (Config) | Formel | Köpfe | Abweichung | Formel: Layer ohne Experten / Experten / Einbettung+lm_head / Sicht / MTP |")
    print("|---|---|---:|---:|---:|---|")
    for label, d, _ in MODELS:
        if d not in est:
            continue
        cfg, _ = MP.load_config(load(a.root, d))
        cw = MP.estimate_weights_from_config(cfg)
        real = est[d]["weights"]["total_bytes"]["v"]
        print("| %s | %s | %d | %d | %s | %.2f / %.2f / %.2f / %.2f / %.2f GiB |" % (
            label, cw["format"], cw["total_bytes"], real, pct(cw["total_bytes"], real), sum(cw["layer_bytes"]) / GIB,
            sum(cw["layer_expert_bytes"]) / GIB, (cw["embed_bytes"] + cw["lm_head_bytes"]) / GIB, cw["visual_bytes"] / GIB, cw["mtp_bytes"] / GIB))
    print("\nGGUF hat keine Quantisierungsangabe in der Config (UD-Mix je Tensor): nur der Kopf bepreist es.\n")

    print("## C. Stufensummen gegen `Load weight end ... mem usage` (GiB)\n")
    print("| Lauf | Stufe | Schätzung | Boot-Log | Abweichung |")
    print("|---|---|---:|---:|---:|")
    for label, d, cut, rows, meas, src in BOOT_STAGES:
        if d not in est:
            continue
        fr = None
        if rows:
            E = est[d]["experts"]["n"]["v"]
            fr = [r / float(E) for r in rows]
        st = MP.stage_weight_bytes(est[d], cut, expert_fractions=fr)
        for i, (s, m) in enumerate(zip(st, meas)):
            print("| %s | PP%d | %.2f | %.2f | %s |" % (label if i == 0 else "", i, s / GIB, m, pct(s / GIB, m)))
        print("| _Quelle: %s_ | | | | |" % src)
    for label, d, e_kind in (("DFlash2-W8 (27B-Draft, PP2)", "Qwen3.8-27B-DFlash2-W8-lued", None),):
        p = load(a.root, d)
        if p:
            dr = MP.estimate_draft(p)
            print("| %s | Draft | %.2f | 2.14 | %s |" % (label, dr["total_bytes"]["v"] / GIB, pct(dr["total_bytes"]["v"] / GIB, 2.14)))
    print()

    print("## D. Geometrieterme gegen Records und Metall\n")
    print("| Größe | Schätzung | Referenz | Abweichung | Quelle der Referenz |")
    print("|---|---:|---:|---:|---|")
    e27 = est.get("Qwen3.8-27B-INT8-gdncov-vocabembed")
    enf = est.get("Qwen3.8-Flash-Next-NVFP4-nvidia")
    rec27 = {r["name"]: r["value"] for r in json.load(open(os.path.join(os.path.dirname(HERE), "python/flliper/srt/pdflip/profile_records_data/qwen27b.json")))["records"]}
    recnf = {r["name"]: r["value"] for r in json.load(open(os.path.join(os.path.dirname(HERE), "python/flliper/srt/pdflip/profile_records_data/nextflash.json")))["records"]}
    if e27:
        v = e27["kv"]["variants"]["fp8_e4m3"]["payload_per_token_all_attn_layers"]["v"]
        print("| 27B KV-Seitenbytes je Token (16 Attn-Layer, fp8) | %d B | %d B | %s | STORE_CENSUS_KV_PAGE_BYTES (weg2sb5g) |" % (v, rec27["STORE_CENSUS_KV_PAGE_BYTES"], pct(v, rec27["STORE_CENSUS_KV_PAGE_BYTES"])))
        m = e27["state"]["variants_mib"]["bfloat16"]
        print("| 27B Mamba-Zustand je Linear-Layer und Slot (ssm bf16) | %.4f MiB | %.4f MiB | %s | P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT (weg2sb5f+weg2rg6) |" % (m, rec27["P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT"], pct(m, rec27["P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT"])))
        r = e27["activation"]["extend_rate_mib_per_row"]["v"]
        print("| 27B Extend-Rate je Zeile (Startwert) | %.4f MiB | %.4f MiB (gemessenes Maximum 0,2690) | %s | D_EXTEND_CAP_PER_ROW_MIB; der Start ist absichtlich konservativ (Q-694b) |" % (r, rec27["D_EXTEND_CAP_PER_ROW_MIB"][0], pct(r, rec27["D_EXTEND_CAP_PER_ROW_MIB"][0])))
    if enf:
        c = enf["kv"]["variants"]["fp8_e4m3"]["cell_bytes_per_attn_layer_token"]["v"]
        print("| NF KV-Zelle je Attn-Layer und Token (fp8) | %d B | 1088 B | %s | fnFL2w123, drei emittierte Zellen 8704/4352/3264 |" % (c, pct(c, 1088)))
        m = enf["state"]["variants_mib"]["bfloat16"]
        print("| NF Mamba-Zustand je Linear-Layer und Slot (ssm bf16) | %.4f MiB | %.4f MiB | %s | P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT (dkrnfh91...) |" % (m, recnf["P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT"], pct(m, recnf["P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT"])))
        r = enf["activation"]["extend_rate_mib_per_row"]["v"]
        print("| NF Extend-Rate je Zeile (Startwert) | %.4f MiB | %.4f MiB (reservierter Zuwachs, nicht gleiche Größe) | %s | D_EXTEND_GROWTH_PER_ROW_MIB rc12g |" % (r, recnf["D_EXTEND_GROWTH_PER_ROW_MIB"][0], pct(r, recnf["D_EXTEND_GROWTH_PER_ROW_MIB"][0])))
        n = enf["experts"]
        print("| NF Experten (n, top_k) | %d, %d | 512, 10 | exakt | Boot-Log `512 Experten`, ServerArgs |" % (n["n"]["v"], n["top_k"]["v"]))
    print()

    print("## E. Registry-Zeile aus dem Schätzprofil gegen die Handzeile\n")
    for label, d, rid in MODELS:
        if not rid or d not in est:
            continue
        fields = MP.derive_registry_fields(est[d], row_id=rid, checkpoint=MC + d)
        rows = MP.compare_with_registry(MP.to_model_profile(fields), F.PROFILES[rid])
        by = {}
        for r in rows:
            by.setdefault(r["class"], []).append(r)
        print("### %s -> `%s` (%d Felder)\n" % (label, rid, len(rows)))
        print("| Klasse | gleich | gleich im Modellteil, Rest weicht ab | abweichend |")
        print("|---|---:|---:|---:|")
        for cls in ("fact", "name", "experts", "format", "policy"):
            rs = by.get(cls, [])
            print("| %s | %d | %d | %d |" % (cls, sum(r["equal"] and not r["partial"] for r in rs), sum(r["equal"] and r["partial"] for r in rs),
                                          sum(not r["equal"] for r in rs)))
        print()
        print("| Feld | Klasse | erzeugt | Handzeile | Grund |")
        print("|---|---|---|---|---|")
        for r in rows:
            if r["class"] in ("fact", "name"):
                print("| %s | %s | %s | %s | gleich |" % (r["field"], r["class"], json.dumps(r["derived"]), json.dumps(r["hand"])))
            elif r["class"] in ("experts", "format"):
                print("| %s | %s | store/swap bzw. Name+Pfad gleich | %s | %s |" % (
                    r["field"], r["class"], "Rest weicht ab" if r["partial"] else "gleich", r["reason"] or "gleich"))
        pol = [r for r in rows if r["class"] == "policy" and not r["equal"]]
        print("\nPolicy-Felder, in denen die Handzeile vom konservativen Standard abweicht (%d):\n" % len(pol))
        for r in pol:
            print("* `%s`: erzeugt %s, Hand %s -- %s" % (r["field"], json.dumps(r["derived"])[:50], json.dumps(r["hand"])[:60], r["reason"]))
        print()


if __name__ == "__main__":
    sys.exit(main())
