"""Offline planner solve: which MLP unit vector does auto-performance
consider PREFILL-optimal for a given checkpoint?

No GPU is touched (CUDA_VISIBLE_DEVICES=99): ServerArgs parsing runs the
VRAM-auto base split and then apply_auto_performance, which logs the whole
decision block -- including, when every concentrated candidate is rejected,
the best FORFEITED candidate. That forfeited vector is the planner-solved
prefill optimum; the gates that reject it are context/decode trades, not a
statement that the vector is wrong for prefill.

Usage:
  python solve_vector.py <model-path> [extra ServerArgs flags...]
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> int:
    model_path = sys.argv[1]
    extra = sys.argv[2:]

    from sglang.srt.server_args import ServerArgs, prepare_server_args

    argv = [
        "--model-path",
        model_path,
        "--tp-size",
        "3",
        "--rank-gpu-id",
        "0,1,2",
        "--rank-tp-ratio",
        "auto-performance",
        "--rank-auto-reserve-mib",
        "3000,2700,2700",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--context-length",
        "32768",
        "--trust-remote-code",
        "--max-running-requests",
        "16",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--enable-metrics",
    ] + extra

    sa: ServerArgs = prepare_server_args(argv)
    print("=== RESOLVED ===")
    print("rank_tp_ratio  :", sa.rank_tp_ratio)
    print("rank_mlp_ratio :", sa.rank_mlp_ratio)
    return 0


if __name__ == "__main__":
    sys.exit(main())
