"""Wave-aware KV split for FlashInfer FA2 prefill (27B line, group P, 2026-09-24).

Default OFF. Nothing here runs unless ``SGLANG_FI_PREFILL_WAVE_SPLIT=1`` is in
the rank's environment (launcher ``--p-deep-split-from N``); unset, the eager
prefill plan and the prefill graph's eligibility are byte-identical to a tree
without this module.

WHAT IT FIXES, MEASURED. Boots weg2xsn426 (eager) and weg2xsn428 (full prefill
graph), 512-token P chunks, PP3 5090/3080/3080, cut 42,11,11 / attn 10,3,3.
Every ``#PGAP`` gpu_fwd_ms joined to its chunk's prefix depth (``#969N ADMIT``
extend + rid), least squares ``a + b * (prefix + C/2) / 1000`` per rank,
residual sd 1.2-3.6 ms over 764-1667 chunks per rank:

    per 512 tokens            depth-independent a     attention slope b (ms / 1k)
    PP0 5090  C=512           41.2 (426) 41.9 (428)   1.424 (426) 1.432 (428)
    PP0 5090  C=4096          41.9 (423) 41.3 (422)   0.876 (423) 0.875 (422)
    PP1 3080  C=512           33.1 (426) 33.8 (428)   1.029 (426) 1.038 (428)
    PP1 3080  C=4096          40.9 (423) 41.3 (422)   0.992 (423) 1.026 (422)

The GEMM/GDN part does not lose at 512 (a is chunk-size neutral on the 5090);
the ATTENTION slope does, x1.63 on the 5090 and x1.02 on the 3080. Cause, from
flashinfer's own scheduler (data/include/flashinfer/attention/scheduler.cuh,
utils.cuh, flashinfer 0.6.14): head_dim 256 -> ``FA2DetermineCtaTileQ`` = 64
rows, packed rows = 512 x gqa 6 -> 48 q tiles, x 4 KV heads = 192 CTAs. Its KV
split search (``PrefillBinarySearchKVChunkSize``) only splits while
``q_tiles * chunks <= 2 * num_sm / num_kv_heads`` = 85 on the 5090 -- 48 tiles
x 2 chunks is already 96, so it never splits. 192 CTAs on 170 SMs run in 2
rounds (192/340 = 56 %); on a 3080 (68 SMs) in 3 rounds (94 %). The per-SM
round model below predicts the 512/4096 slope ratio as 1.60 (5090) and 1.04
(3080), measured 1.63 and 1.02-1.04.

THE KNOB flashinfer already has: ``plan(fixed_split_size=S)`` (the
deterministic-inference path passes it). With S = ceil(kv / n) every q tile
becomes n work items; n = 5 gives 960 CTAs = 6 rounds on 170 SMs (94 %).
:func:`choose_prefill_kv_split` picks n per plan from the round model, the
float workspace (flashinfer allocates ``num_qo_heads * work_items * cta_tile_q
* (head_dim + 1) * 4`` bytes of partials -- n <= 5 fits the default 384 MiB for
this geometry) and a minimum gain; on a 3080 it finds no gain and returns no
split, so one environment serves all three stages.

WHY NOT INSIDE THE PREFILL GRAPH. In graph mode flashinfer fixes the grid at
``max(2 * num_sm / num_kv_heads, q_tiles)`` = 85 work items per KV head on the
5090, and a fixed split that exceeds it is refused ("new batch size should not
exceed padded batch size"). So the split only runs eager. The target form
(operator, 24.09.): shallow chunks replay the graph, deep chunks run eager
with the split. The graph's side is Agent H's depth threshold
(3e6ce57ce4: ``SGLANG_PREFILL_GRAPH_MAX_PREFIX`` / launcher
``--p-prefill-graph-max-prefix N``, prefix > N runs eager, census reason
``deep_split``); this module is the eager side and splits a chunk whose prefix
is >= ``SGLANG_FI_PREFILL_WAVE_SPLIT_FROM_PREFIX``. Set both to the same N.
The crossover from the same fits: eager PP0 is host-launch bound at ~57 ms per
512 chunk for 42 layers (xsn426 launch spans), the graph costs 41.9 + 1.432 x
(prefix + 256) / 1000 ms, so eager + split wins from prefix ~10.2k on.

THE SAME LINE AS THE DETERMINISTIC TILE. The value reaches flashinfer through
the extend plan's existing ``fixed_split_size`` argument (flashinfer_backend,
where ``prefill_split_tile_size`` / SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE
goes; flashinfer #1675), never over it. It is computed per plan rather than a
fixed tile because flashinfer 0.6.14 reserves the split partials for
``num_qo_heads * work_items * cta_tile_q * head_dim * 4`` bytes (the GQA
over-reservation that upstream #5177, merged 2026-09-17, removes; not in
0.6.14): 75.5 MB per KV chunk for this geometry, so a fixed tile of, say,
13k tokens is n=5 at a 64k prefix (fits 384 MiB) and n=10 at 128k (755 MB,
"Buffer overflow when allocating memory for batch_prefill_tmp_v"). The chooser
caps n by the workspace instead.

UPSTREAM STATE CHECKED 2026-09-24 against 0.6.14: flashinfer PrefillPlan
still assumes ``num_blocks_per_sm = 2`` (scheduler.cuh:717, main identical) --
the root of the 192-CTA non-split. #5502 (split-KV row stride from the plan,
wrong output with qo_len > 1 when the plan has trailing EMPTY pages; open) does
not apply here: page_size 1, exact lengths, every page full. #4356 (tmp_v
typed DTypeO; open): 0.6.14 writes bf16 partials, so a split result is NOT
bit-identical to the unsplit one -- expect up to ~1 bf16 ulp on a share of the
elements (upstream #5166 measured 2063/4096 differing values for split decode
with output-dtype partials). The GPU check is
test/registered/unit/layers/test_fi_prefill_wave_split_gpu_0924.py.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Dict, Optional, Sequence

#: "1" arms the wave-aware split in the EAGER prefill plan (flashinfer_backend).
WAVE_SPLIT_ENV = "SGLANG_FI_PREFILL_WAVE_SPLIT"
#: Minimum PREFIX (tokens already computed for the request) before a split is
#: considered; below it the attention is a small share of the forward and the
#: plan stays stock. Default 0 when the switch is on.
WAVE_SPLIT_FROM_PREFIX_ENV = "SGLANG_FI_PREFILL_WAVE_SPLIT_FROM_PREFIX"

#: Largest n tried. The float workspace bounds it earlier on this geometry.
MAX_KV_CHUNKS = 8
#: A KV chunk below this many tokens is not worth its merge and Q reload.
MIN_KV_CHUNK_TOKENS = 2048
#: Predicted attention time must fall by at least this fraction.
MIN_GAIN = 0.10
#: Per-CTA fixed work in KV-token equivalents: the 64 x 256 bf16 Q tile load
#: (32 KiB ~ 64 fp8 K+V tokens) plus the fp32 partial written for the merge
#: (64 KiB ~ 128 tokens). Keeps the model from preferring tiny chunks.
CTA_OVERHEAD_TOKENS = 192
#: Headroom kept free in the float workspace (the allocator aligns each
#: region to 16 bytes and the tmp_s region follows tmp_v).
WORKSPACE_FILL_CAP = 0.95


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(str(os.environ.get(name, "") or default).strip())
    except ValueError:
        return default


def wave_split_on() -> bool:
    return os.environ.get(WAVE_SPLIT_ENV, "") == "1"


def wave_split_from_prefix() -> int:
    return max(0, _env_int(WAVE_SPLIT_FROM_PREFIX_ENV, 0))


def launcher_env_p_deep_split(from_prefix: int) -> Dict[str, str]:
    """Group P's environment for ``--p-deep-split-from N``; {} when N <= 0, so
    the default environment stays byte-identical. ONE place for the two
    names, read by the launcher and pinned by the tests. The graph's own
    threshold is Agent H's SGLANG_PREFILL_GRAPH_MAX_PREFIX, set by his flag."""
    n = int(from_prefix or 0)
    if n <= 0:
        return {}
    return {
        WAVE_SPLIT_ENV: "1",
        WAVE_SPLIT_FROM_PREFIX_ENV: str(n),
    }


def fa2_cta_tile_q(avg_packed_qo_len: int, head_dim: int, sm_major: int = 8) -> int:
    """Mirror of flashinfer ``FA2DetermineCtaTileQ`` (utils.cuh, 0.6.14)."""
    if head_dim >= 512:
        return 16 if avg_packed_qo_len <= 32 else 32
    if avg_packed_qo_len > 64 and head_dim < 256:
        return 128
    if sm_major >= 8:
        return 64 if avg_packed_qo_len > 16 else 16
    return 64


@dataclasses.dataclass(frozen=True)
class SplitChoice:
    """One plan's decision. ``fixed_split_size`` is None = stock plan."""

    fixed_split_size: Optional[int]
    chunks: int
    q_tiles: int
    cta_tile_q: int
    ctas_stock: int
    rounds_stock: int
    ctas_split: int
    rounds_split: int
    predicted_ratio: float
    reason: str

    def line(self) -> str:
        return (
            "FI-WAVE-SPLIT %s chunks=%d fixed_split_size=%s q_tiles=%d cta_tile_q=%d "
            "ctas %d->%d rounds %d->%d predicted_attn_ratio=%.3f"
            % (
                self.reason,
                self.chunks,
                self.fixed_split_size,
                self.q_tiles,
                self.cta_tile_q,
                self.ctas_stock,
                self.ctas_split,
                self.rounds_stock,
                self.rounds_split,
                self.predicted_ratio,
            )
        )


def choose_prefill_kv_split(
    qo_lens: Sequence[int],
    kv_lens: Sequence[int],
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_sm: int,
    float_workspace_bytes: int,
    page_size: int = 1,
    min_prefix_tokens: int = 0,
    max_chunks: int = MAX_KV_CHUNKS,
    min_chunk_tokens: int = MIN_KV_CHUNK_TOKENS,
    min_gain: float = MIN_GAIN,
    sm_major: int = 8,
) -> SplitChoice:
    """Pick the KV chunk count for one prefill plan by the per-SM round model.

    ``qo_lens`` are the new tokens per request, ``kv_lens`` the KV length the
    PAGED kernel reads per request (prefix + new tokens when the plan is
    causal over both; prefix only under the ragged/paged pair). Cost of a
    split into chunks of S tokens: ``ceil(CTAs(S) / num_sm) * (S +
    CTA_OVERHEAD_TOKENS)`` -- every SM runs its share of equal-length CTAs, the
    last round sets the makespan (the model that reproduced the measured 512 vs
    4096 slopes on both cards). Returns the stock plan unless a split is
    predicted to save ``min_gain`` and fits the float workspace.
    """
    qo = [int(x) for x in qo_lens]
    kv = [max(1, int(x)) for x in kv_lens]
    if not qo or len(qo) != len(kv) or min(qo) <= 0:
        return SplitChoice(None, 1, 0, 0, 0, 0, 0, 0, 1.0, "skip:bad_lens")
    if num_kv_heads <= 0 or num_qo_heads % num_kv_heads or num_sm <= 0:
        return SplitChoice(None, 1, 0, 0, 0, 0, 0, 0, 1.0, "skip:bad_geometry")
    gqa = num_qo_heads // num_kv_heads
    packed = [q * gqa for q in qo]
    tile = fa2_cta_tile_q(sum(packed) // len(packed), head_dim, sm_major)
    tiles = [math.ceil(p / tile) for p in packed]
    q_tiles = sum(tiles)
    kv_max = max(kv)
    prefix_max = max(k - q for k, q in zip(kv, qo))

    def ctas(chunk: int) -> int:
        return sum(t * math.ceil(k / chunk) for t, k in zip(tiles, kv)) * num_kv_heads

    def cost(chunk: int) -> float:
        return math.ceil(ctas(chunk) / num_sm) * float(min(chunk, kv_max) + CTA_OVERHEAD_TOKENS)

    ctas1 = ctas(kv_max)
    rounds1 = math.ceil(ctas1 / num_sm)
    stock = SplitChoice(None, 1, q_tiles, tile, ctas1, rounds1, ctas1, rounds1, 1.0, "stock")
    if prefix_max < int(min_prefix_tokens):
        return dataclasses.replace(stock, reason="stock:shallow")
    base = cost(kv_max)
    best = None
    for n in range(2, int(max_chunks) + 1):
        chunk = math.ceil(kv_max / n)
        if chunk < int(min_chunk_tokens):
            break
        # Every KV chunk must reach into the causal range of the request's
        # first new row (its last chunk starts at or before kv - qo), so no
        # work item is an all-masked CTA. Holds by construction while chunk
        # >= MIN_KV_CHUNK_TOKENS > qo; checked, not assumed.
        if any(k - (math.ceil(k / chunk) - 1) * chunk < q for k, q in zip(kv, qo)):
            continue
        work_items = sum(t * math.ceil(k / chunk) for t, k in zip(tiles, kv))
        need = num_qo_heads * work_items * tile * (head_dim + 1) * 4
        if need > WORKSPACE_FILL_CAP * float(float_workspace_bytes):
            break
        c = cost(chunk)
        if best is None or c < best[0] - 1e-9:
            best = (c, n, chunk)
    if best is None or best[0] > (1.0 - float(min_gain)) * base:
        return dataclasses.replace(stock, reason="stock:no_gain")
    c, n, chunk = best
    ps = max(1, int(page_size))
    split_pages = math.ceil(chunk / ps)
    ctas_n = ctas(split_pages * ps)
    return SplitChoice(
        fixed_split_size=split_pages,
        chunks=n,
        q_tiles=q_tiles,
        cta_tile_q=tile,
        ctas_stock=ctas1,
        rounds_stock=rounds1,
        ctas_split=ctas_n,
        rounds_split=math.ceil(ctas_n / num_sm),
        predicted_ratio=c / base,
        reason="split",
    )
