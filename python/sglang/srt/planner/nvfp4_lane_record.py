# SPDX-License-Identifier: Apache-2.0
"""Evidence register (Backlog #38 L7): MEASURED NVFP4 GEMM lane rates on the
reference rig, for ``--fp4-gemm-backend native-mixed``.

The NVFP4 lanes have no planner probe, so the boot solver
(``uneven_perf.rank_gemm_scores``) looks the DETECTED card up here and uses a
value only when the rig profile did not measure the lane itself. Like the other
evidence registers in this package it names the rig it measured; the solver
never does.

Source: window zcx7pv (2026-09-25 ~12:40Z, cards 1+2), raw data
/spinning/evidence-665-f1/n4b_bench_0925_1239/b5090_all.json and
b3080_lanes.json; CUDA graph, weights rotated past L2, the serving apply paths
(apply_nat = fp4_quantize + fork CUTLASS sm_120a; apply_mar =
apply_fp4_marlin_linear). Rate = the FLOP-weighted MLP of one 27B layer at M=512
(the P chunk): 2*512*3*5120*17408 FLOP over (gate_up 34816x5120 + down
5120x17408):

  RTX 5090  native  220.55 + 119.38 us -> 805.4 TFLOPS   (power limit 400 W)
  RTX 5090  Marlin  874.28 + 414.68 us -> 212.4 TFLOPS
  RTX 3080  Marlin 3228.47 + 1711.67 us ->  55.4 TFLOPS  (power limit 230 W)

The power limit is part of the record (p-schnitt-anpassbar-powerlimit): another
limit is another record -- re-measure, never scale.
"""

from __future__ import annotations

from typing import Dict

SOURCE = "record zcx7pv 2026-09-25 M=512 (5090 400 W, 3080 230 W)"

#: card-name substring -> {lane: TFLOPS}
LANES: Dict[str, Dict[str, float]] = {
    "RTX 5090": {"nvfp4_native": 805.4, "nvfp4_marlin": 212.4},
    "RTX 3080": {"nvfp4_marlin": 55.4},
}


def lanes_for(card_name: str) -> Dict[str, float]:
    """The measured record for the detected card, or {} (never a guess)."""
    name = str(card_name or "")
    for key, lanes in LANES.items():
        if key in name:
            return dict(lanes)
    return {}
