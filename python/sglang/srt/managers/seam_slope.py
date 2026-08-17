# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#685: the per-rank seam slope, DERIVED from the layout rather than measured.

Deliberately dependency-free: this module imports nothing from sglang, so it
can be read (and cherry-picked) on its own by the funding path without pulling
in a solver or a cost model.

WHAT THE SLOPE IS
-----------------
``phase_flip_runtime._staging_bytes`` (:4403) peaks at
``incoming + max(outgoing, local)``, and the incoming leg (:4487-4497) is

    dst.row_nbytes(layer) * rows      summed over tr.recv_layers[peer]

Nothing else in that formula scales with rows-per-layer. So a rank's per-ROW
seam slope is exactly the number of full-attention layers it must RECEIVE at a
pp->tp cutover, times the KV cell.

HOW MANY LAYERS A RANK RECEIVES
-------------------------------
The pp->tp cutover reconciles two different ownerships of the same KV. In the
PP layout rank ``r`` holds its own ``attention_held_r`` layers for every token;
in the TP layout it holds ``tp_share_r`` of the token axis for ALL layers. The
difference is what has to cross the wire::

    received_r = max(0, tp_share_r * n_attention_total - attention_held_r)

``max(0, ...)`` because a rank whose PP holding EXCEEDS its TP share sheds KV
rather than acquiring it, and shedding costs no incoming staging.

WHY THIS IS NOT A RANK-0 PATHOLOGY
----------------------------------
On the reference rig (``--phase-flip-tp-vector 32,16,16``, cut ``28,20,16``,
attention ``[7,5,4]``, 16 full-attention layers)::

    rank 0:  0.500 * 16 - 7 = +1 layer   -> 1 * 2326.7 = 2326.7 B/row
    rank 1:  0.250 * 16 - 5 = -1 layer   -> sheds, receives nothing
    rank 2:  0.250 * 16 - 4 =  0 layers  -> neutral, receives nothing

against measured 2360.1 / 424.1 / 547.6 B/row. Rank 0 lands within 1.4 %; the
other two predict ZERO and their entire measured slope is the baseline every
rank pays regardless (checksums, the one-layer streaming window, allocator
grain). The oft-quoted "rank 0 is 5.6x its peers" is one received layer against
a baseline. Rank 0 is simply the only rank this flip asks to ACQUIRE KV,
because the flip hands it half the token axis while the cut gave it 7 of 16
attention layers.

WHY A FROZEN TRIPLE CANNOT BE CARRIED ACROSS CUTS
-------------------------------------------------
``received_r`` moves in WHOLE LAYER steps, so a measured slope vector is valid
only for the cut and flip vector it was measured at. Move one attention layer
off rank 2 and it starts paying a full cell it did not pay before; move one off
rank 0 and its slope RISES, because it still needs half the token axis and now
holds less of it. Any consumer that carries a fixed triple across a re-cut is
pricing the seam of a layout it is not proposing.

WHAT THIS MEANS FOR FUNDING
---------------------------
A rank that receives nothing is being taxed for bytes that never move. On the
shipping cut that is ranks 1 and 2: their measured slope is baseline only, and
funding a per-token seam for them reserves against a transfer that does not
happen.

SCOPE
-----
This module does not read or write ``phase_flip_seam_reserve.SHIP_PIN``. That
pin's ``basis_per_row_bytes`` is a frozen PROVENANCE record, kept fixed so a
registered test can detect drift against it; the live sizing path reads the
per-boot measured record instead. Deriving a slope here neither replaces nor
invalidates either of those.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

__all__ = ["derive_seam_slope_bytes_per_token", "received_attention_layers"]


def received_attention_layers(
    flip_tp_vector: Sequence[float],
    attention_counts: Sequence[int],
    n_attention_total: int,
) -> Tuple[float, ...]:
    """Full-attention layers each rank must RECEIVE at a pp->tp cutover.

    Fractional by construction: the flip vector need not divide the attention
    count evenly, and a rank that receives half a layer's rows stages half a
    layer's bytes. Rounding here would quantise a real cost.
    """
    n = len(attention_counts)
    if len(flip_tp_vector) != n:
        raise ValueError(
            f"seam slope: {len(flip_tp_vector)} flip weights for {n} stages."
        )
    if n_attention_total < 0:
        raise ValueError(f"seam slope: n_attention_total is {n_attention_total}.")
    total = float(sum(float(w) for w in flip_tp_vector))
    if total <= 0.0:
        raise ValueError("seam slope: the flip vector sums to zero.")
    out: List[float] = []
    for rank in range(n):
        share = float(flip_tp_vector[rank]) / total
        # Shedding costs no INCOMING staging, so the deficit clamps at zero.
        out.append(
            max(0.0, share * float(n_attention_total) - float(attention_counts[rank]))
        )
    return tuple(out)


def derive_seam_slope_bytes_per_token(
    flip_tp_vector: Sequence[float],
    attention_counts: Sequence[int],
    kv_bytes_per_token_per_attn_layer: float,
    n_attention_total: int,
    baseline_bytes_per_token: Sequence[float] = (),
) -> Tuple[float, ...]:
    """Per-rank seam slope in bytes per KV row.

    ``baseline_bytes_per_token`` is the residual every rank pays whatever it
    receives (checksums, the one-layer streaming window, allocator grain).
    Supply the measured values; it defaults to zero, which makes the return
    value purely the transfer term.
    """
    if kv_bytes_per_token_per_attn_layer < 0.0:
        raise ValueError(
            "seam slope: kv_bytes_per_token_per_attn_layer is "
            f"{kv_bytes_per_token_per_attn_layer}."
        )
    received = received_attention_layers(
        flip_tp_vector, attention_counts, n_attention_total
    )
    base = list(baseline_bytes_per_token) + [0.0] * (
        len(received) - len(baseline_bytes_per_token)
    )
    return tuple(
        r * float(kv_bytes_per_token_per_attn_layer) + float(b)
        for r, b in zip(received, base)
    )
