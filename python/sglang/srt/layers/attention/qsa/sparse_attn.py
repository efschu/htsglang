"""Validated sparse GQA operators migrated from the QSA reference branch."""

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_H20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (1024, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]
_L20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (128, (64, 4, 2)),
    (512, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]


def _get_best_config(total_q: int):
    table = _H20_CONFIGS if "H20" in torch.cuda.get_device_name(0) else _L20_CONFIGS
    return next(cfg for limit, cfg in table if total_q <= limit)


# fnFL2 H58: the table above is keyed on the device NAME -- H20, else L20 --
# so the rig's RTX 3080 (sm86) and RTX 5090 (sm120) both take the L20 row,
# and above 512 query rows (every P prefix chunk) that is (BLOCK_N 16,
# 1 warp, 2 stages). Compiled for head_dim 256 (Triton cache, cuobjdump) that
# build holds REG 255 with STACK 1104-1120 B per thread on sm86 and 992-1000 B
# on sm120 -- a spilling kernel -- in 16.4 KB of shared memory per 32-thread
# CTA: 5 CTAs = 5 of 48 warp slots per SM on both cards. The same kernel at
# (32, 8, 2), the table's own <=32-row entry, does not spill: REG 211-226 /
# STACK 0 on sm86 (1 CTA = 8 warps per SM), REG 111 / STACK 0 on sm120
# (2 CTAs). On the P stages the rows path of a 16k prefix chunk costs 428-435
# ms per full-attention layer on the 3080s against 57 ms on the 5090 (x162
# ATTN-TIMING-PREFILL, x136 FWD-TIMING-PREFILL alike), 7.5x, while the
# prefix-free prefill kernel of the same backend is 3.3x apart. This override
# makes the launch an A/B at the metal, per arch, without touching the
# default: SGLANG_FORCE_QSA_ROWS_CONFIG, grammar
#   [smXX:]LIMIT=BLOCK_N/WARPS/STAGES[,LIMIT=...][;[smYY:]...]
# LIMIT is the largest total_q the entry serves ("inf" closes a group);
# e.g. "sm86:inf=32/8/2" moves only the 3080 stages to the spill-free build.
def parse_rows_config(raw: str, arch: int):
    """The (limit, (block_n, warps, stages)) table the override names for
    ``arch`` (e.g. 86, 120), or None when it names none for this arch."""
    chosen = None
    for group in (g.strip() for g in str(raw or "").split(";")):
        if not group:
            continue
        target = None
        if group.startswith("sm") and ":" in group:
            head, group = group.split(":", 1)
            target = int(head[2:])
        if target is not None and target != int(arch):
            continue
        table = []
        for entry in (e.strip() for e in group.split(",")):
            if not entry:
                continue
            limit_s, cfg_s = entry.split("=", 1)
            block_n, warps, stages = (int(x) for x in cfg_s.split("/"))
            if block_n < 16 or block_n & (block_n - 1):
                raise ValueError(f"QSA rows config {entry!r}: BLOCK_N must be a power of two >= 16")
            if warps not in (1, 2, 4, 8, 16) or stages < 1:
                raise ValueError(f"QSA rows config {entry!r}: warps in 1/2/4/8/16, stages >= 1")
            limit = float("inf") if limit_s.strip() == "inf" else int(limit_s)
            table.append((limit, (block_n, warps, stages)))
        if not table or table[-1][0] != float("inf"):
            raise ValueError(f"QSA rows config group {group!r} must end with an 'inf=' entry")
        # an arch-specific group wins over a generic one, whatever the order
        if chosen is None or target is not None:
            chosen = table
    return chosen


_ROWS_CONFIG_CACHE: dict = {}

# H101 (rc9p, 26.09.): the L20 row above 512 query rows, (16, 1, 2), is a
# SPILLING build on sm120, and since the CUDA-13 image (rc2.1d, 8f1db7a6e5,
# PTX 9.0) a much bigger one: the container's Triton cache holds the D build
# (fp8 exp2 decode, USE_COUNTS=False) at REG 128 / STACK 2320 B per thread
# (cuobjdump -res-usage, entry SEVCQXIANJ...; the same source under PTX 8.8
# was REG 255 / STACK 992 B). The driver backs the stack for every resident
# thread: 2320 B x 170 SM x 1536 = 578 MiB (agent09252021, the next
# WEG2-SLEEP-LMEM: 'lmem 578->0 MiB ... NVML -580 MiB measured'), against the
# 1024 B = 255 MiB the context holds. The FIRST launch of that form on a rank
# grows the reservation inside cuLaunchKernel -- rc9p D-TP0 died there
# ('Triton Error [CUDA]: out of memory', X-direct extend of 3585 rows on a
# 23744-token prefix, 22 min into the boot), with no planner post for it.
# Group P never met it: its arm sets SGLANG_FORCE_QSA_ROWS_CONFIG=inf=64/8/2
# (NF_ENV_P_FORM), group D had no such line and kept the table.
# The sm120 default therefore replaces ONLY that entry with the build P runs
# for its 16k chunks: (64, 8, 2) is REG 152 / STACK 0 on sm120 under PTX 9.0
# (UZIGNZUEW4..., the very D build, fp8 exp2, USE_COUNTS=False) and REG
# 111-152 / STACK 0 under PTX 8.8. Every band up to 512 rows (decode, verify,
# short extends) keeps its build, so their SASS is unchanged. sm86 keeps the
# table (its D ranks do not attend under Form A; its P ranks run the arm's
# override). An env table for this arch still wins over this default.
_SM120_ROWS_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (128, (64, 4, 2)),
    (512, (32, 4, 2)),
    (float("inf"), (64, 8, 2)),
]
_ARCH_ROWS_DEFAULTS = {120: _SM120_ROWS_CONFIGS}


def _device_arch() -> int:
    """sm number of the current device (e.g. 86, 120); torch caches the
    device properties, so this is a dictionary read per launch."""
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def rows_table_for_arch(arch: int):
    """The rows kernel's default (limit, (block_n, warps, stages)) table for
    ``arch`` when no env override names it: H101's spill-free sm120 table, else
    ``None`` (= the device-name keyed H20/L20 table of ``_get_best_config``)."""
    return _ARCH_ROWS_DEFAULTS.get(int(arch))


def _get_rows_config(total_q: int):
    """``_get_best_config`` for the rows kernel, unless
    SGLANG_FORCE_QSA_ROWS_CONFIG names a table for this device's arch, or the
    arch has a measured spill-free default (H101, sm120)."""
    from sglang.srt.environ import envs

    raw = envs.SGLANG_FORCE_QSA_ROWS_CONFIG.get()
    if raw:
        major, minor = torch.cuda.get_device_capability()
        key = (raw, major * 10 + minor)
        if key not in _ROWS_CONFIG_CACHE:
            _ROWS_CONFIG_CACHE[key] = parse_rows_config(raw, key[1])
        table = _ROWS_CONFIG_CACHE[key]
        if table is not None:
            return next(cfg for limit, cfg in table if total_q <= limit)
    table = rows_table_for_arch(_device_arch())
    if table is not None:
        return next(cfg for limit, cfg in table if total_q <= limit)
    return _get_best_config(total_q)


def rows_launch_forms(arch: Optional[int] = None):
    """Every distinct (block_n, warps, stages) the rows launch can pick on this
    device, each with the smallest total_q that selects it -- the set a boot
    prewarm has to load (H101). Walks the bands of the effective table."""
    if arch is None:
        major, minor = torch.cuda.get_device_capability()
        arch = major * 10 + minor
    from sglang.srt.environ import envs

    raw = envs.SGLANG_FORCE_QSA_ROWS_CONFIG.get()
    table = parse_rows_config(raw, arch) if raw else None
    if table is None:
        table = rows_table_for_arch(arch)
    if table is None:
        table = _H20_CONFIGS if "H20" in torch.cuda.get_device_name(0) else _L20_CONFIGS
    forms = []
    lower = 1
    for limit, cfg in table:
        if cfg not in (f for _q, f in forms):
            forms.append((lower, cfg))
        if limit == float("inf"):
            break
        lower = int(limit) + 1
    return forms


# fnFL2 H65 (F2 of H58): the prefix-free prefill kernel (_sparse_gqa_prefill,
# every prompt's first chunk and every short prefill) takes the same L20 row:
# (16, 1, 2) above 512 rows. Offline compiled for head_dim 256 and the boot's
# top-k width 2051 that build is REG 255 (STACK 24 B sm86 / 32 B sm120) in
# 24.6 KB smem per 1-warp CTA -> 3 CTAs = 3 warps per SM on sm86 AND sm120, a
# latency-bound kernel (x166: 59 ms per layer on the 3080s, 17 on the 5090,
# ~8x above its issue bound); (32, 8, 2) is REG 80-89 / STACK 0 in 42 KB ->
# 2 CTAs = 16 warps per SM on both. SGLANG_WEG2_QSA_PREFILL_CONFIG takes
# the SGLANG_FORCE_QSA_ROWS_CONFIG grammar for this launch only; empty = table.
_PREFILL_CONFIG_CACHE: dict = {}


def _get_prefill_config(total_q: int):
    """``_get_best_config`` for the prefix-free prefill kernel, unless
    SGLANG_WEG2_QSA_PREFILL_CONFIG names a table for this device's arch."""
    from sglang.srt.environ import envs

    raw = envs.SGLANG_WEG2_QSA_PREFILL_CONFIG.get()
    if raw:
        major, minor = torch.cuda.get_device_capability()
        key = (raw, major * 10 + minor)
        if key not in _PREFILL_CONFIG_CACHE:
            _PREFILL_CONFIG_CACHE[key] = parse_rows_config(raw, key[1])
        table = _PREFILL_CONFIG_CACHE[key]
        if table is not None:
            return next(cfg for limit, cfg in table if total_q <= limit)
    return _get_best_config(total_q)


# fnFL2 H65: which in-kernel fp8 decode the rows kernel runs (FP8_DECODE_*,
# see the decode helpers above _sparse_attn_rows_fwd).
# SGLANG_WEG2_QSA_FP8_DECODE, grammar  [smXX:]MODE[;[smYY:]MODE]  with
# MODE = exp2 | bits | ptx; an arch group wins over a generic one, whatever
# the order; an arch no group names keeps exp2 (the default -- the kernel
# then compiles to the same SASS as before H65). "ptx" needs sm80+
# (fma.rn.bf16x2) and is refused below.
FP8_DECODE_EXP2 = 0
FP8_DECODE_BITS = 1
FP8_DECODE_PTX = 2
_FP8_DECODE_MODES = {"exp2": FP8_DECODE_EXP2, "bits": FP8_DECODE_BITS, "ptx": FP8_DECODE_PTX}
_FP8_DECODE_NAMES = {v: k for k, v in _FP8_DECODE_MODES.items()}


def parse_fp8_decode(raw: str, arch: int) -> int:
    """The FP8_DECODE constexpr SGLANG_WEG2_QSA_FP8_DECODE names for ``arch``
    (e.g. 86, 120); 0 (exp2) when it names none."""
    generic = chosen = None
    for group in (g.strip() for g in str(raw or "").split(";")):
        if not group:
            continue
        target = None
        if group.startswith("sm") and ":" in group:
            head, group = group.split(":", 1)
            target = int(head[2:])
        mode = group.strip().lower()
        if mode not in _FP8_DECODE_MODES:
            raise ValueError(
                f"SGLANG_WEG2_QSA_FP8_DECODE group {group!r}: mode must be one of "
                f"{sorted(_FP8_DECODE_MODES)}"
            )
        if target is None:
            generic = _FP8_DECODE_MODES[mode]
        elif target == int(arch):
            chosen = _FP8_DECODE_MODES[mode]
    value = chosen if chosen is not None else generic if generic is not None else FP8_DECODE_EXP2
    if value == FP8_DECODE_PTX and int(arch) < 80:
        raise ValueError(
            f"SGLANG_WEG2_QSA_FP8_DECODE={raw!r}: 'ptx' needs sm80+ "
            f"(fma.rn.bf16x2), this device is sm{int(arch)}"
        )
    return value


_FP8_DECODE_CACHE: dict = {}


def _rows_fp8_decode() -> int:
    """FP8_DECODE for this device: 0 (exp2) unless SGLANG_WEG2_QSA_FP8_DECODE
    names a mode for its arch."""
    from sglang.srt.environ import envs

    raw = envs.SGLANG_WEG2_QSA_FP8_DECODE.get()
    if not raw:
        return FP8_DECODE_EXP2
    major, minor = torch.cuda.get_device_capability()
    key = (raw, major * 10 + minor)
    if key not in _FP8_DECODE_CACHE:
        _FP8_DECODE_CACHE[key] = parse_fp8_decode(raw, key[1])
    return _FP8_DECODE_CACHE[key]


_ROWS_LAUNCH_SEEN: set = set()


def _note_rows_launch(total_q, block_n, warps, stages, kv_fp8, fp8_decode) -> None:
    """One QSA-ROWS-LAUNCH line per process and launch form: proof in the
    rank's own log that the arm's launch config and fp8 decode arrived
    (an env var in the launcher is not an env var in the rank)."""
    form = (block_n, warps, stages, bool(kv_fp8), int(fp8_decode))
    if form in _ROWS_LAUNCH_SEEN:
        return
    _ROWS_LAUNCH_SEEN.add(form)
    from sglang.srt.environ import envs

    try:
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm{major * 10 + minor}"
    except Exception:  # noqa: BLE001 -- a log line never fails a launch
        arch = "sm?"
    logger.info(
        "QSA-ROWS-LAUNCH arch=%s kv=%s decode=%s cfg=%d/%d/%d first_total_q=%d "
        "table=%s (SGLANG_FORCE_QSA_ROWS_CONFIG=%r SGLANG_WEG2_QSA_FP8_DECODE=%r; one line "
        "per launch form)",
        arch,
        "fp8" if kv_fp8 else "16bit",
        _FP8_DECODE_NAMES.get(int(fp8_decode), str(fp8_decode)) if kv_fp8 else "-",
        block_n,
        warps,
        stages,
        int(total_q),
        _rows_table_source(),
        envs.SGLANG_FORCE_QSA_ROWS_CONFIG.get(),
        envs.SGLANG_WEG2_QSA_FP8_DECODE.get(),
    )


def _rows_table_source() -> str:
    """Which table the rows launch reads on this device: ``env`` (an override
    names this arch), ``h101-sm120`` (the spill-free arch default), else the
    device-name keyed ``h20`` / ``l20``. For the QSA-ROWS-LAUNCH line only."""
    from sglang.srt.environ import envs

    try:
        arch = _device_arch()
        raw = envs.SGLANG_FORCE_QSA_ROWS_CONFIG.get()
        if raw and parse_rows_config(raw, arch) is not None:
            return "env"
        if rows_table_for_arch(arch) is not None:
            return f"h101-sm{arch}"
        return "h20" if "H20" in torch.cuda.get_device_name(0) else "l20"
    except Exception:  # noqa: BLE001 -- a log line never fails a launch
        return "?"


#: H101: the specializations the rows launches of this process use -- q heads /
#: head_dim / dtype, the rows width K (``sr_m`` is a constexpr, so the exact
#: width is part of the build) and the KV pool tensors (weak references: their
#: strides and dtype are build keys too). One entry per distinct key, recorded
#: at the launches the boot already makes (the target's AND the draft's decode
#: graph capture), so the boot prewarm (qsa/rows_prewarm.py) builds exactly the
#: variants serving will launch. ``done`` closes the recording after the
#: prewarm: serving launches then pay one dict read.
_ROWS_PREWARM_SIG: dict = {"sigs": {}, "done": False}
_ROWS_PREWARM_SIG_MAX = 8


def _note_rows_prewarm_sig(q, k_pool, v_pool, rows) -> None:
    import weakref

    try:
        sigs = _ROWS_PREWARM_SIG["sigs"]
        key = (
            int(q.shape[1]), int(q.shape[2]), str(q.dtype), int(rows.shape[-1]),
            str(k_pool.dtype), tuple(k_pool.stride()), tuple(v_pool.stride()),
        )
        if key in sigs or len(sigs) >= _ROWS_PREWARM_SIG_MAX:
            return
        sigs[key] = {
            "heads": key[0],
            "head_dim": key[1],
            "dtype": q.dtype,
            "k": key[3],
            "k_pool": weakref.ref(k_pool),
            "v_pool": weakref.ref(v_pool),
        }
    except Exception:  # noqa: BLE001 -- a record never fails a launch
        pass


def rows_prewarm_signatures():
    """The recorded launch signatures (list of dicts with live pool tensors);
    entries whose pool tensors are gone are left out."""
    out = []
    for sig in list(_ROWS_PREWARM_SIG["sigs"].values()):
        k_pool, v_pool = sig["k_pool"](), sig["v_pool"]()
        if k_pool is not None and v_pool is not None:
            out.append(dict(sig, k_pool=k_pool, v_pool=v_pool))
    return out


def close_rows_prewarm_recording() -> None:
    _ROWS_PREWARM_SIG["done"] = True


@triton.jit
def _sparse_gqa_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_seqlens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    seq_start = tl.load(cu_seqlens + batch).to(tl.int64)
    seq_end = tl.load(cu_seqlens + batch + 1).to(tl.int64)
    query_relative = tl.program_id(0).to(tl.int64)
    query = seq_start + query_relative
    if query >= seq_end:
        return

    row_topk = tl.minimum(topk, query_relative + 1)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    head_start = group * GROUP_SIZE
    q_values = tl.load(
        q
        + query * sq_m
        + (head_start + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + seq_start * sk_n + group * sk_h
    v_base = v + seq_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (head_start + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton(q, k, v, max_seqlen_k, indices, cu_seqlens, scale):
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_prefill_config(total_q)
    out = torch.empty_like(q)
    _sparse_gqa_prefill[(max_seqlen_k, (cu_seqlens.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_seqlens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _sparse_gqa_chunk_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch)
    q_end = tl.load(cu_q + batch + 1)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return
    k_start = tl.load(cu_k + batch).to(tl.int64)
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    visible = query_relative + kv_len - (q_end - q_start) + 1
    row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q
        + query * sq_m
        + (group * GROUP_SIZE + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + k_start * sk_n + group * sk_h
    v_base = v + k_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        # The chunk-prefill K/V tensors are gathered from the KV pool and can
        # therefore carry the FP8 storage dtype, which Triton's dot rejects
        # (`Unsupported rhs dtype fp8e4nv`). Convert to Q's dtype; the QSA
        # backend writes the pool without per-tensor k/v scales, so this is a
        # plain cast (no-op for BF16 pools).
        keys = keys.to(q_values.dtype)
        values = values.to(q_values.dtype)
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (group * GROUP_SIZE + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton_ck(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    max_q = int((cu_q[1:] - cu_q[:-1]).max().item())
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    out = torch.empty_like(q)
    _sparse_gqa_chunk_prefill[(max_q, (cu_q.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _fa2_valid_counts(
    seq_lens,
    indices,
    counts,
    topk: tl.constexpr,
    stride_i: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_TOPK)
    length = tl.load(seq_lens + row)
    positions = tl.load(
        indices + row * stride_i + cols,
        mask=cols < topk,
        other=-1,
    )
    valid = (positions >= 0) & (positions < length)
    tl.store(counts + row, tl.sum(valid.to(tl.int32), axis=0))


@triton.jit
def _fa2_prefix_sum(counts, cu_k, batch, BLOCK_B: tl.constexpr):
    rows = tl.arange(0, BLOCK_B)
    valid_rows = rows < batch
    row_counts = tl.load(counts + rows, mask=valid_rows, other=0)
    tl.store(cu_k, 0)
    tl.store(cu_k + rows + 1, tl.cumsum(row_counts, 0), mask=valid_rows)


def qwen_sparse_fa2_cu_seqlens_triton(
    seq_lens, indices, counts, cu_k, batch, topk, block_b: Optional[int] = None
):
    block_b = block_b or triton.next_power_of_2(batch)
    # One request per program: Triton caps a tile at 1M elements,
    # which [next_pow2(topk), next_pow2(batch)] exceeds at topk=2051, batch=512.
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )
    # Prefix sum is only over the batch dimension and remains a small 1-D
    # tensor, including during CUDA graph capture.
    _fa2_prefix_sum[(1,)](
        counts,
        cu_k,
        batch,
        BLOCK_B=block_b,
        num_warps=8,
    )


@triton.jit
def _compact_kv(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    pad_cols,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ZERO_FILL: tl.constexpr,
    KV_FP8: tl.constexpr = False,
):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    )
    # 64-bit element offsets: slot * heads * dim exceeds int32 once the pool holds
    # more than 2^31 / (heads * dim) tokens (~4.2M for 2 x 256), which an FP8 pool
    # on one GPU does reach.
    src = slots.to(tl.int64)[:, None] * heads * dim + head * dim + dims[None, :]
    dst = (
        (pack_start + cols).to(tl.int64)[:, None] * heads * dim
        + head * dim
        + dims[None, :]
    )
    load_mask = valid[:, None] & (dims[None, :] < dim)
    if ZERO_FILL:
        # Strided (page-aligned) packing: the paged decode kernel reads whole pages,
        # so every slot in [valid_count, pad_cols) must hold zeros, never stale bytes.
        # `valid_count` here is the row's page-aligned stride, not its valid count, so
        # the store covers the full region while the load stays limited to valid rows.
        store_mask = (cols < pad_cols)[:, None] & (dims[None, :] < dim)
    else:
        store_mask = load_mask
    # Dequantize while gathering: the scratch is allocated in the query dtype, so an
    # FP8 pool is read as fp8 and stored as bf16. The QSA backend writes the pool
    # without per-tensor k/v scales (see set_kv_buffer calls in
    # qwen_sparse_attn_backend.py), so no scale is applied here either.
    out_dtype = out_k.dtype.element_ty
    # fn5d 2026-09-16 (PP=3, 3080 stage): Triton refuses fp8e4nv below sm89, so
    # an FP8 pool arrives as uint8 bytes (KV_FP8) and is decoded here -- the
    # same helper the DCP rows kernel uses (WP3b).
    if KV_FP8:
        k_vals = _fp8_e4m3_bytes_to_f32(tl.load(k + src, mask=load_mask, other=0)).to(out_dtype)
        v_vals = _fp8_e4m3_bytes_to_f32(tl.load(v + src, mask=load_mask, other=0)).to(out_dtype)
    else:
        k_vals = tl.load(k + src, mask=load_mask, other=0.0).to(out_dtype)
        v_vals = tl.load(v + src, mask=load_mask, other=0.0).to(out_dtype)
    tl.store(out_k + dst, k_vals, mask=store_mask)
    tl.store(out_v + dst, v_vals, mask=store_mask)


def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
    """Valid-count pass alone, without the packed cu_seqlens prefix sum."""
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )


def qwen_sparse_kv_extraction_compact_triton(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    batch,
    topk,
    zero_fill_cols: int = 0,
):
    """Gather the selected K/V rows into ``out_k``/``out_v``.

    ``zero_fill_cols`` > 0 selects the strided (page-aligned) layout used by the paged
    decode kernel: row ``b`` owns ``[cu_k[b], cu_k[b] + zero_fill_cols)`` and every slot
    past its valid rows is zero-filled. Paged kernels read whole pages and multiply the
    masked probabilities into V, so stale or uninitialized bytes there (NaN/Inf bit
    patterns) would otherwise leak into the output. ``0`` keeps the compact layout for
    the varlen fallback, whose rows are packed back-to-back.

    ``out_k``/``out_v`` may use a wider dtype than the pool (bf16 scratch for an FP8
    pool); rows are converted while gathering.

    Both layouts assume the valid entries of each ``indices`` row are contiguous at
    the front (``expand_qsa_block_indices`` sorts them that way): ``valid_count`` is a
    count, not a mask, so a ``-1`` in the middle of a row would shift the packing.
    """
    _, heads, dim = k.shape
    block_topk = 16
    zero_fill = zero_fill_cols > 0
    num_cols = zero_fill_cols if zero_fill else topk
    kv_fp8 = k.dtype == torch.float8_e4m3fn
    if kv_fp8:
        # 1 byte per element either way; decoded in-kernel (fn5d, sm86 stages)
        k, v = k.view(torch.uint8), v.view(torch.uint8)
    _compact_kv[(batch, heads, triton.cdiv(num_cols, block_topk))](
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        topk,
        heads,
        dim,
        req_to_token.stride(0),
        indices.stride(0),
        num_cols,
        BLOCK_TOPK=block_topk,
        BLOCK_D=triton.next_power_of_2(dim),
        ZERO_FILL=zero_fill,
        KV_FP8=kv_fp8,
        num_warps=8,
    )


@triton.jit
def _fp8_e4m3_bytes_to_f32(x):
    """float8_e4m3fn stored as uint8 -> float32 by bit arithmetic. Triton
    refuses the fp8e4nv type below sm89 (the 3080 ranks, fn1w boot
    2026-09-16), so the pool bytes are loaded as uint8 and decoded here on
    every architecture alike: sign(1) exp(4, bias 7) mant(3); exp==0 is
    subnormal (2^-6 * m/8); 0x7F/0xFF (exp 15, mant 7) is NaN."""
    xi = x.to(tl.int32)
    sign = (xi >> 7) & 1
    exp = (xi >> 3) & 0xF
    man = xi & 0x7
    normal = tl.where(
        exp > 0, tl.math.exp2((exp - 7).to(tl.float32)) * (1.0 + man.to(tl.float32) / 8.0), man.to(tl.float32) / 512.0
    )
    val = tl.where(sign == 1, -normal, normal)
    nan = (exp == 15) & (man == 7)
    return tl.where(nan, float("nan"), val)


# fnFL2 H65: the rows kernel decodes EVERY selected K/V byte once per
# (query, kv head) program -- a 16k prefix chunk decodes each prefix row
# thousands of times -- and the exp2 decode above costs ~23 SASS instructions
# per element (MUFU.EX2, two I2F, FMULs, ISETP/SEL chains). Offline compiled
# (Triton 3.6, cuobjdump), it is ~95 % of the loop of every spill-free launch
# config on sm86 AND sm120 (~370 of ~390 warp instructions per key). The two
# decodes below give the SAME bits for all 254 non-NaN codes (0x80 -> +0.0
# like the exp2 decode, see below; the two NaN codes stay NaN) with far
# fewer instructions; SGLANG_WEG2_QSA_FP8_DECODE picks one
# (_rows_fp8_decode, FP8_DECODE_*), default = the exp2 decode above.


@triton.jit
def _fp8_e4m3_bytes_to_f32_bits(x):
    """Same value as ``_fp8_e4m3_bytes_to_f32`` without exp2 or int->float:
    a normal code (exp field e >= 1) IS the fp32 word
    ``(e + 120) << 23 | m << 20`` = ``((x & 0x7F) << 20) + (120 << 23)``;
    for e == 0 that word is 2^-7 + m 2^-10, and ``2 f - 2^-6`` = m 2^-9 is the
    subnormal, exactly (both operands normal, no denormal arithmetic);
    0x7F/0xFF stay NaN, the sign comes last -- with the same ``-f`` as the
    exp2 decode, which Triton lowers as ``0 - f``: 0x80 is +0.0 in both
    (torch says -0.0; no dot can tell, every dot adds onto a +0 accumulator)."""
    xi = x.to(tl.int32)
    t = xi & 0x7F
    f = ((t << 20) + 0x3C000000).to(tl.float32, bitcast=True)
    f = tl.where(t < 8, f * 2.0 - 0.015625, f)
    f = tl.where(t == 0x7F, float("nan"), f)
    return tl.where(xi > 0x7F, -f, f)


@triton.jit
def _fp8_e4m3_bytes_to_bf16_ptx(x):
    """Four fp8 bytes -> four bf16 per inline-PTX instance (sm80+), the
    packed form of the same value: each byte goes to the high byte of a
    16-bit half (prmt), ``(half >> 4) & 0x07F0`` puts exp/mantissa where bf16
    keeps them (bias still 120 short), the sign bit is OR-ed back (lop3) and
    ``fma.rn.bf16x2(s, 2^120, +0.0)`` rebiases -- exact for normals and for the
    subnormals (bf16 denormal inputs, no .ftz on bf16). The +0.0 addend turns
    0x80 into +0.0, bit for bit what the exp2 decode gives (see
    _fp8_e4m3_bytes_to_f32_bits). The NaN codes (magnitude half 0x07F0) come
    out of the fma as +-480; ``set.eq`` on that half forces 0x7FC0 into them,
    so they stay NaN. 7 instructions per two elements instead of ~46."""
    return tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b32 a<2>, m<2>, s<2>, v<2>, n<2>, k120, knan, kzero;
        mov.b32 k120, 0x7B807B80;
        mov.b32 knan, 0x07F007F0;
        mov.b32 kzero, 0;
        prmt.b32 a0, 0, $2, 0x5040;
        prmt.b32 a1, 0, $2, 0x7060;
        shr.b32 m0, a0, 4;
        shr.b32 m1, a1, 4;
        and.b32 m0, m0, 0x07F007F0;
        and.b32 m1, m1, 0x07F007F0;
        lop3.b32 s0, a0, 0x80008000, m0, 0xEA;
        lop3.b32 s1, a1, 0x80008000, m1, 0xEA;
        fma.rn.bf16x2 v0, s0, k120, kzero;
        fma.rn.bf16x2 v1, s1, k120, kzero;
        set.eq.u32.f16x2 n0, m0, knan;
        set.eq.u32.f16x2 n1, m1, knan;
        lop3.b32 $0, n0, 0x7FC07FC0, v0, 0xEA;
        lop3.b32 $1, n1, 0x7FC07FC0, v1, 0xEA;
        }
        """,
        constraints="=r,=r,r",
        args=[x],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _sparse_attn_rows_fwd(
    q,
    k,
    v,
    out,
    lse,
    rows,
    scale,
    topk,
    counts,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    sl_m: tl.constexpr,
    sl_h: tl.constexpr,
    sr_m: tl.constexpr,
    sr_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KV_FP8: tl.constexpr,
    USE_COUNTS: tl.constexpr,
    FP8_DECODE: tl.constexpr = 0,
):
    """WP3b: sparse GQA over ABSOLUTE K/V rows, with the LSE.

    ``rows[query, :]`` are row indices straight into ``k``/``v`` (the KV pool
    as this rank stores it), ``-1`` = not attended. That is what makes one
    kernel serve every mode: prefill with a prefix, decode and the
    speculative paged modes hand it the rows they selected; under uneven DCP
    the rows are this rank's OWNED subset of the selection and the natural-log
    ``lse`` this kernel emits is what the group's LSE merge combines. A query
    whose rows are all ``-1`` (this rank owns none of its keys) gets out=0 and
    lse=-inf, which the merge weighs as exp(-inf)=0.
    """
    query = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    head_start = group * GROUP_SIZE
    q_values = tl.load(
        q + query * sq_m + (head_start + offs_h[:, None]) * sq_h + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + group * sk_h
    v_base = v + group * sv_h
    row_ptr = rows + query * sr_m
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    # Owned-row compaction (Task #42, 19.09.): under DCP every rank holds only
    # its OWN subset of a query's top-k rows (the rest are -1). Looping to
    # ``topk`` and masking the foreign lanes still runs the tile math for every
    # lane, so each rank paid the whole query x head x top-k work regardless of
    # its share -- the Ampere ranks at ~210 ms per layer against 33 ms on the
    # Blackwell rank, and the chunk waited for them. With the rows sorted so
    # the owned ones lead, ``counts[query]`` bounds the loop to the owned rows.
    if USE_COUNTS:
        limit = tl.load(counts + query).to(tl.int32)
    else:
        limit = topk
    for start in range(0, limit, BLOCK_N):
        current = start + offs_n
        row = tl.load(row_ptr + current * sr_n, mask=current < topk, other=-1).to(tl.int64)
        valid = row >= 0
        safe_row = tl.where(valid, row, 0)
        keys_raw = tl.load(
            k_base + safe_row[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0,
        )
        values_raw = tl.load(
            v_base + safe_row[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0,
        )
        if KV_FP8:
            if FP8_DECODE == 2:
                keys = _fp8_e4m3_bytes_to_bf16_ptx(keys_raw).to(q_values.dtype)
                values = _fp8_e4m3_bytes_to_bf16_ptx(values_raw).to(q_values.dtype)
            elif FP8_DECODE == 1:
                keys = _fp8_e4m3_bytes_to_f32_bits(keys_raw).to(q_values.dtype)
                values = _fp8_e4m3_bytes_to_f32_bits(values_raw).to(q_values.dtype)
            else:
                keys = _fp8_e4m3_bytes_to_f32(keys_raw).to(q_values.dtype)
                values = _fp8_e4m3_bytes_to_f32(values_raw).to(q_values.dtype)
        else:
            keys = keys_raw.to(q_values.dtype)
            values = values_raw.to(q_values.dtype)
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        # A block with no valid key leaves next_max at -inf; exp2(-inf - -inf)
        # is nan, so the running max is only moved by finite scores.
        next_max = tl.where(next_max == -float("inf"), max_value, next_max)
        alpha = tl.where(
            max_value == -float("inf"), 1.0, tl.math.exp2(max_value - next_max)
        )
        probabilities = tl.where(
            next_max[:, None] == -float("inf"),
            0.0,
            tl.math.exp2(scores - next_max[:, None]),
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    has_keys = normalizer > 0
    safe_norm = tl.where(has_keys, normalizer, 1.0)
    output = accumulator / safe_norm[:, None]
    tl.store(
        out + query * so_m + (head_start + offs_h[:, None]) * so_h + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )
    lse_value = tl.where(
        has_keys, (max_value + tl.math.log2(safe_norm)) * 0.6931471805599453, -float("inf")
    )
    tl.store(
        lse + query * sl_m + (head_start + offs_h) * sl_h,
        lse_value,
        mask=offs_h < GROUP_SIZE,
    )


def compact_owned_rows(rows):
    """Sort each query's rows so the owned ones (>= 0) lead and the -1 lanes
    trail, and count the owned ones: (rows_sorted [Tq, topk] int32,
    counts [Tq] int32). Attention is permutation-invariant over the rows, so
    the order change is exact up to fp32 summation order."""
    rows = rows.to(torch.int32)
    if rows.numel() == 0:
        return rows.contiguous(), torch.zeros((rows.shape[0],), dtype=torch.int32, device=rows.device)
    rows_sorted, _ = torch.sort(rows, dim=-1, descending=True)
    counts = (rows_sorted >= 0).sum(dim=-1, dtype=torch.int32)
    return rows_sorted.contiguous(), counts.contiguous()


def sparse_attn_rows_triton(q, k_pool, v_pool, rows, scale, row_counts=None):
    """(out [Tq, Hq, D] in q.dtype, lse [Tq, Hq] fp32 natural log) for the
    selected absolute rows; see ``_sparse_attn_rows_fwd``.

    ``row_counts`` ([Tq] int32, from ``compact_owned_rows``) bounds every
    query's loop to its leading valid rows; without it the loop runs to
    ``topk`` and masks."""
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k_pool.shape[1]
    group_size = num_q_heads // num_kv_heads
    rows = rows.to(torch.int32).contiguous()
    use_counts = row_counts is not None
    if use_counts:
        row_counts = row_counts.to(torch.int32).contiguous()
        if row_counts.shape[0] != rows.shape[0]:
            raise ValueError("row_counts must have one entry per query row")
    else:
        row_counts = rows  # unused pointer; USE_COUNTS=False never reads it
    out = torch.empty_like(q)
    lse = torch.empty((total_q, num_q_heads), dtype=torch.float32, device=q.device)
    if total_q == 0:
        return out, lse
    if not _ROWS_PREWARM_SIG["done"]:
        _note_rows_prewarm_sig(q, k_pool, v_pool, rows)
    kv_fp8 = k_pool.dtype == torch.float8_e4m3fn
    if kv_fp8:
        # Same strides (1 byte per element either way); decoded in-kernel.
        k_pool, v_pool = k_pool.view(torch.uint8), v_pool.view(torch.uint8)
    elif k_pool.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"sparse rows kernel: unsupported KV dtype {k_pool.dtype}")
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_rows_config(total_q)
    # fnFL2 H65: the fp8 decode only exists for an fp8 pool; a 16-bit pool
    # keeps FP8_DECODE=0 so it never compiles a second variant.
    fp8_decode = _rows_fp8_decode() if kv_fp8 else FP8_DECODE_EXP2
    _note_rows_launch(total_q, block_n, warps, stages, kv_fp8, fp8_decode)
    _sparse_attn_rows_fwd[(total_q, num_kv_heads)](
        q,
        k_pool,
        v_pool,
        out,
        lse,
        rows,
        scale,
        rows.shape[-1],
        row_counts,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_pool.stride(0),
        k_pool.stride(1),
        k_pool.stride(2),
        v_pool.stride(0),
        v_pool.stride(1),
        v_pool.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        rows.stride(0),
        rows.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        KV_FP8=kv_fp8,
        USE_COUNTS=use_counts,
        FP8_DECODE=fp8_decode,
        num_warps=warps,
        num_stages=stages,
    )
    return out, lse


def fp8_e4m3_bytes_to_f32_reference(x: torch.Tensor) -> torch.Tensor:
    """Torch twin of the in-kernel decode, for the pin against torch's own
    float8_e4m3fn conversion (all 256 codes)."""
    xi = x.to(torch.int32)
    sign, exp, man = (xi >> 7) & 1, (xi >> 3) & 0xF, xi & 0x7
    normal = torch.where(
        exp > 0, torch.exp2((exp - 7).float()) * (1.0 + man.float() / 8.0), man.float() / 512.0
    )
    val = torch.where(sign == 1, -normal, normal)
    return torch.where((exp == 15) & (man == 7), torch.full_like(val, float("nan")), val)


def sparse_attn_rows_reference(q, k_pool, v_pool, rows, scale):
    """Device-agnostic twin of ``sparse_attn_rows_triton`` (same contract,
    fp32 math): the pin for the kernel and for the LSE merge."""
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k_pool.shape[1]
    repeats = num_q_heads // num_kv_heads
    out = torch.zeros((total_q, num_q_heads, head_dim), dtype=torch.float32, device=q.device)
    lse = torch.full((total_q, num_q_heads), -float("inf"), dtype=torch.float32, device=q.device)
    for t in range(total_q):
        valid = rows[t] >= 0
        if not bool(valid.any()):
            continue
        sel = rows[t][valid].long()
        keys = k_pool[sel].float()  # [n, Hkv, D]
        values = v_pool[sel].float()
        for h in range(num_q_heads):
            g = h // repeats
            scores = keys[:, g, :] @ q[t, h].float() * scale  # [n]
            m = scores.max()
            p = torch.exp(scores - m)
            l = p.sum()
            out[t, h] = (p @ values[:, g, :]) / l
            lse[t, h] = m + torch.log(l)
    return out.to(q.dtype), lse


def merge_partial_attention(outs, lses):
    """The group LSE merge (the math of ``cp_lse_ag_out_ar_mha_uneven``) over a
    list of per-rank partials: ``sum_r exp(lse_r - lse) * out_r`` with
    ``lse = logsumexp_r(lse_r)``. A rank with lse=-inf contributes nothing."""
    lses = torch.stack([l.float() for l in lses])  # [R, T, H]
    global_lse = torch.logsumexp(lses, dim=0)
    total = None
    for out, l in zip(outs, lses):
        w = torch.exp(l - global_lse).unsqueeze(-1)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        part = torch.nan_to_num(out.float(), nan=0.0) * w
        total = part if total is None else total + part
    return total, global_lse


__all__ = [
    "qwen_sparse_fa2_cu_seqlens_triton",
    "qwen_sparse_valid_counts_triton",
    "qwen_sparse_kv_extraction_compact_triton",
    "sparse_gqa_fwd_interface_triton",
    "sparse_gqa_fwd_interface_triton_ck",
    "sparse_attn_rows_triton",
    "sparse_attn_rows_reference",
    "merge_partial_attention",
]
