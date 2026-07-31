#!/usr/bin/env python3
"""Reads the window's raw files into verdicts, against the 2026-07-31 anchors.

Every expectation this checks is stated in scripts/nvfp4/v4_boot_proof.sh or in
the families slice report, and every anchor number comes from the 06:40 NVFP4
beleg's own JSONL -- read out of that file here rather than retyped, so an
anchor cannot drift by transcription.
"""

from __future__ import annotations

import json
import pathlib
import re

OUT = pathlib.Path("/spinning/gpu-battery-results/2026-07-31_332_fam_beleg")
ANCHOR = pathlib.Path("/spinning/gpu-battery-results/2026-07-31_nvfp4_beleg")


def load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def by_arm(rows: list[dict], arm: str, **kw) -> dict | None:
    for r in rows:
        if r.get("arm") != arm:
            continue
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def main() -> None:
    anchors_p = load_jsonl(ANCHOR / "punkte.jsonl")
    anchors_d = load_jsonl(ANCHOR / "decode_punkte.jsonl")
    mine_p = load_jsonl(OUT / "punkte.jsonl")
    mine_d = load_jsonl(OUT / "decode_punkte.jsonl")

    print("=" * 72)
    print("ANCHORS (2026-07-31 06:40 window)")
    for arm in ("v4_solo_5090", "fp8_tp3_anchor"):
        p = by_arm(anchors_p, arm)
        if p:
            pf = p["prefill"]
            print(f"  {arm:16} prefill {pf['prefill_tok_s']:9.1f} tok/s  "
                  f"p50 {pf['latenz_ms_p50']:8.1f} ms")
        for bs in (1, 8):
            d = by_arm(anchors_d, arm, bs=bs)
            if d:
                tok_s = d["klient_tok_s"]
                streams = d["klient_stroeme"]
                print(f"  {arm:16} decode bs={bs}: {tok_s:8.2f} tok/s  "
                      f"{1000.0 * streams / tok_s:6.2f} ms/step")

    print("=" * 72)
    print("THIS WINDOW")
    for arm in ("v4_tp3_nextn", "v4_solo_res2048"):
        p = by_arm(mine_p, arm)
        if p:
            pf = p["prefill"]
            print(f"  {arm:16} prefill {pf['prefill_tok_s']:9.1f} tok/s  "
                  f"p50 {pf['latenz_ms_p50']:8.1f} ms  reqs={pf['requests']}")
        else:
            print(f"  {arm:16} prefill: NO POINT")
        for bs in (1, 8):
            d = by_arm(mine_d, arm, bs=bs)
            if d:
                tok_s = d["klient_tok_s"]
                streams = d["klient_stroeme"]
                acc = (d.get("probe_accept_bs1") or {}).get("spec_accept_length")
                print(f"  {arm:16} decode bs={bs}: {tok_s:8.2f} tok/s  "
                      f"{1000.0 * streams / tok_s:6.2f} ms/step  "
                      f"accept_len={acc}")

    print("=" * 72)
    print("POSTEN 1 / 2  (readouts)")
    for tag in ("v4_tp3_nextn", "v4_solo_res2048"):
        f = OUT / "proofs" / f"{tag}.readout.txt"
        if not f.exists():
            print(f"  {tag}: no readout (arm did not reach teardown)")
            continue
        head = {}
        for line in f.read_text().splitlines():
            m = re.match(r"^([A-Z_]+)=(\d+)$", line)
            if m:
                head[m.group(1)] = int(m.group(2))
        print(f"  {tag}: {head}")

    print("=" * 72)
    print("SIZING / CORRIDOR")
    for tag in ("v4_tp3_nextn", "v4_solo_res2048"):
        log = OUT / "logs" / f"{tag}.server.log"
        if not log.exists():
            print(f"  {tag}: no server log")
            continue
        text = log.read_text(errors="replace")
        for pat, label in (
            (r"Load weight end\..*?mem usage=([\d.]+) GB", "weights GB"),
            (r"KV Cache is allocated.*?#tokens: (\d+)", "KV tokens"),
            (r"max_total_num_tokens=(\d+)", "max_total_num_tokens"),
            (r"available_gpu_mem=([\d.]+) GB", "free GB after boot"),
            (r"Reserve-based sizing \(#332\)[^\n]*", "reserve line"),
        ):
            hits = re.findall(pat, text)
            if hits:
                uniq = sorted(set(hits))[:4]
                print(f"  {tag}: {label} = {uniq}")

    print("=" * 72)
    print("COHERENCE")
    for tag in ("v4_tp3_nextn", "v4_solo_res2048"):
        rows = load_jsonl(OUT / f"coherence_{tag}.jsonl")
        if not rows:
            print(f"  {tag}: none")
            continue
        ok = sum(1 for r in rows if (r.get("checks") or {}).get("coherent"))
        print(f"  {tag}: {ok}/{len(rows)} coherent")

    print("=" * 72)
    print("FAMILY FOLLOW-UPS")
    for name in ("dense2_noradix", "fp82_lowbudget"):
        d = OUT / name
        gate = d / "gate.txt"
        print(f"  {name}: gate={'yes' if gate.exists() else 'no'}")
        if gate.exists():
            for line in gate.read_text().splitlines()[-6:]:
                print(f"      {line}")


if __name__ == "__main__":
    main()
