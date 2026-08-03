#!/usr/bin/env python3
"""Predict the DSV4 C4-indexer prefill transient, both arms, before the boot.

Prints what the A/B of the next GPU window must see. Uses the PRODUCTION
formula (`layers.attention.dsv4.indexer.indexer_prefill_scratch_bytes`), not a
restatement of it, so a prediction that disagrees with the run is evidence about
the model rather than about this script.

No GPU, no CUDA call, no server. Run it with CUDA_VISIBLE_DEVICES=99.

    python3 scripts/dev/493_indexer_transient/predict.py \
        --rows 256 --span 8196 --seq-chunk 2048

Defaults are the window-3 geometry (2026-08-03, DeepSeek-V4-Flash TP=3 on the
club-3090 rig): --chunked-prefill-size 256, C4 span 8196 (the compress_ratio-4
image of a 32768-token prompt), SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK=2048.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"
    ),
)

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.attention.dsv4.indexer import (  # noqa: E402
    _indexer_logits_chunk_rows,
    _indexer_logits_output_bytes,
    _indexer_logits_step_bytes,
    indexer_prefill_scratch_bytes,
)

MIB = 1024 * 1024


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=256, help="--chunked-prefill-size")
    ap.add_argument(
        "--span", type=int, default=8196, help="C4 indexer span (positions)"
    )
    ap.add_argument("--seq-chunk", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=64, help="index_n_heads")
    ap.add_argument("--head-dim", type=int, default=128, help="index_head_dim")
    ap.add_argument(
        "--steady-free-mib",
        type=int,
        default=873,
        help="steady free VRAM of the tightest rank; window 3 measured 873",
    )
    ap.add_argument("--corridor-mib", type=int, default=400)
    args = ap.parse_args()

    per_row = _indexer_logits_step_bytes(
        chunk_seq=args.seq_chunk, num_heads=args.heads, head_dim=args.head_dim
    )
    out_mib = _indexer_logits_output_bytes(args.rows, args.span) / MIB

    print(
        f"geometry: rows={args.rows} span={args.span} seq_chunk={args.seq_chunk} "
        f"heads={args.heads} head_dim={args.head_dim}"
    )
    print(f"  per query row : {per_row / MIB:.4f} MiB")
    print(
        f"  logits output : {out_mib:.1f} MiB  (unchunkable -- it is the return value)"
    )
    print()
    print(
        f"{'budget MiB':>12} {'chunk_rows':>11} {'steps':>6} {'peak MiB':>10} "
        f"{'free@peak':>10} {'corridor':>9}"
    )
    arms = {}
    for budget in (0, 2048, envs.SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB.get()):
        with envs.SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB.override(budget):
            with envs.SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK.override(args.seq_chunk):
                rows = _indexer_logits_chunk_rows(
                    chunk_seq=args.seq_chunk,
                    num_heads=args.heads,
                    head_dim=args.head_dim,
                    num_rows=args.rows,
                )
                peak = (
                    indexer_prefill_scratch_bytes(
                        num_rows=args.rows,
                        max_seq_len=args.span,
                        num_heads=args.heads,
                        head_dim=args.head_dim,
                    )
                    / MIB
                )
        arms[budget] = peak
        steps = -(-args.rows // rows) if rows else 0
        free = args.steady_free_mib - peak
        verdict = "OK" if free >= args.corridor_mib else "BREACH"
        label = "0 (off)" if budget == 0 else str(budget)
        print(
            f"{label:>12} {rows:>11} {steps:>6} {peak:>10.1f} {free:>10.1f} "
            f"{verdict:>9}"
        )

    off = arms[0]
    on = arms[envs.SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB.get()]
    print()
    print(
        f"PREDICTED A/B DELTA (budget off -> shipped default): {off - on:.1f} MiB "
        f"per rank of peak allocated VRAM."
    )
    print("This is what forward_peak's `peak_bytes_max` must move by for the")
    print("attribution to hold. If it does not move by roughly this much, the")
    print("corridor breach is NOT the indexer transient and #493 is wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
