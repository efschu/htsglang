#!/usr/bin/env python3
"""Fixed-cost footprint of a DSV4F GGUF quant tier, from the tensor table only.

Why this exists (task #478): the question "does UD-Q3_K_XL fit on this rig?"
was previously answered by scaling the measured RAM of the UD-IQ3_XXS boot by
the ratio of the on-disk sizes. That is wrong in the direction that matters,
because the two byte classes land in two different places:

  * NON-EXPERT tensors (attention, norms, embeddings, shared experts, the MLA
    projections) are ALWAYS device-resident. They come off the top of VRAM and
    no offload knob can move them.
  * EXPERT tensors (``*_exps``) are the only class the #77/#123 offload tier
    can place. A fraction stays resident in VRAM, the remainder lives in the
    host cold pool.

So the binding constraint is not "total checkpoint bytes vs total RAM"; it is
"non-expert bytes vs VRAM left after KV and activations", and only the residual
lands in host RAM. This reads the GGUF tensor table (header only, no tensor
data is touched) and prints both classes for each quant tier, then solves for
the resident expert fraction that the rig can actually carry.

Header-only: we parse the tensor info block ourselves rather than materialising
anything, so running this against a 120 GiB checkpoint costs a few hundred KiB
of page cache and no measurable time.

Hermetic: touches no GPU. Run under CUDA_VISIBLE_DEVICES=99.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# GGUF header parsing (tensor table only)
# ---------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"

# GGUF metadata value type ids -> struct format / handling.
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16 = 0, 1, 2, 3
_T_UINT32, _T_INT32, _T_FLOAT32, _T_BOOL = 4, 5, 6, 7
_T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_FMT = {
    _T_UINT8: "<B",
    _T_INT8: "<b",
    _T_UINT16: "<H",
    _T_INT16: "<h",
    _T_UINT32: "<I",
    _T_INT32: "<i",
    _T_FLOAT32: "<f",
    _T_BOOL: "<?",
    _T_UINT64: "<Q",
    _T_INT64: "<q",
    _T_FLOAT64: "<d",
}

# ggml type id -> (block size in elements, bytes per block).
# Only the types that actually occur in these checkpoints need to be right;
# an unknown id is reported by name rather than silently costed as zero.
GGML_TYPES: Dict[int, Tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    39: ("MXFP4", 32, 17),
}


class GgufParseError(RuntimeError):
    pass


class _Reader:
    def __init__(self, fh):
        self.fh = fh

    def raw(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise GgufParseError(f"short read: wanted {n}, got {len(b)}")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self.raw(n).decode("utf-8", errors="replace")

    def skip_value(self, vtype: int) -> None:
        """Skip one metadata value. We only need the tensor table, but the
        metadata block sits in front of it and must be walked exactly."""
        if vtype == _T_STRING:
            self.raw(self.u64())
        elif vtype == _T_ARRAY:
            elem_type = self.u32()
            count = self.u64()
            if elem_type == _T_STRING:
                for _ in range(count):
                    self.raw(self.u64())
            elif elem_type == _T_ARRAY:
                raise GgufParseError("nested arrays are not supported")
            else:
                fmt = _SCALAR_FMT.get(elem_type)
                if fmt is None:
                    raise GgufParseError(f"unknown array element type {elem_type}")
                self.raw(struct.calcsize(fmt) * count)
        else:
            fmt = _SCALAR_FMT.get(vtype)
            if fmt is None:
                raise GgufParseError(f"unknown metadata value type {vtype}")
            self.raw(struct.calcsize(fmt))


@dataclass
class TensorInfo:
    name: str
    dims: Tuple[int, ...]
    ggml_type: int

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.ggml_type, (f"UNKNOWN_{self.ggml_type}", 0, 0))[0]

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def n_bytes(self) -> int:
        entry = GGML_TYPES.get(self.ggml_type)
        if entry is None:
            raise GgufParseError(
                f"tensor {self.name!r} has unknown ggml type id {self.ggml_type}; "
                "add it to GGML_TYPES rather than costing it as zero"
            )
        _, block_elems, block_bytes = entry
        n = self.n_elements
        if n % block_elems:
            raise GgufParseError(
                f"tensor {self.name!r}: {n} elements is not a multiple of the "
                f"{entry[0]} block size {block_elems}"
            )
        return (n // block_elems) * block_bytes


def read_tensor_table(path: str) -> List[TensorInfo]:
    """Parse one GGUF shard's tensor table. Header bytes only."""
    with open(path, "rb") as fh:
        r = _Reader(fh)
        if r.raw(4) != GGUF_MAGIC:
            raise GgufParseError(f"{path}: not a GGUF file")
        version = r.u32()
        if version != 3:
            raise GgufParseError(f"{path}: unsupported GGUF version {version}")
        tensor_count = r.u64()
        kv_count = r.u64()
        for _ in range(kv_count):
            r.string()  # key
            r.skip_value(r.u32())
        out: List[TensorInfo] = []
        for _ in range(tensor_count):
            name = r.string()
            n_dims = r.u32()
            dims = tuple(r.u64() for _ in range(n_dims))
            ggml_type = r.u32()
            r.u64()  # offset within the data block; not needed
            out.append(TensorInfo(name=name, dims=dims, ggml_type=ggml_type))
        return out


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def is_expert_tensor(name: str) -> bool:
    """Is this tensor part of the offloadable routed-expert stack?

    The GGUF naming convention marks the expert-major stacked tensors with the
    ``_exps`` suffix (``blk.N.ffn_{gate,up,down}_exps.weight``). Shared experts
    (``ffn_*_shexp``) are deliberately NOT in this class: they run on every
    token, are never evicted, and therefore cost VRAM like attention does.
    """
    return "_exps" in name


@dataclass
class QuantFootprint:
    label: str
    path: str
    shards: List[str] = field(default_factory=list)
    expert_bytes: int = 0
    nonexpert_bytes: int = 0
    expert_tensors: int = 0
    nonexpert_tensors: int = 0
    file_bytes: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    nonexpert_by_type: Dict[str, int] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return self.expert_bytes + self.nonexpert_bytes


GIB = float(1 << 30)


def gib(n: int) -> float:
    return n / GIB


# MXFP4 (ggml type 39) has no kernel on this fork, so the loader repacks it
# losslessly to Q5_0 inside gguf_quant_weights_iterator. The lattice maps
# exactly, but the block grows from 17 to 22 bytes -- see
# model_loader/gguf_mxfp4_repack.py, which states the factor by name: "The
# price is bytes: 22 per block instead of 17, a factor 22/17 = 1.294 on the
# repacked tensors only. That is real RAM and real VRAM."
# So a footprint taken from the on-disk tensor table UNDERSTATES what the
# checkpoint costs once loaded, by 29.4% of its MXFP4 bytes. For a tier that
# is mostly MXFP4 this is the difference between fits and does not fit.
MXFP4_REPACK_FACTOR = 22.0 / 17.0


def apply_mxfp4_repack(fp: QuantFootprint) -> QuantFootprint:
    """Restate a footprint as the loader will actually hold it in memory."""
    mx = fp.by_type.get("MXFP4", 0)
    if not mx:
        return fp
    mx_nonexpert = fp.nonexpert_by_type.get("MXFP4", 0)
    mx_expert = mx - mx_nonexpert
    grow = MXFP4_REPACK_FACTOR - 1.0
    fp.expert_bytes += int(mx_expert * grow)
    fp.nonexpert_bytes += int(mx_nonexpert * grow)
    fp.by_type.pop("MXFP4")
    fp.by_type["Q5_0(from MXFP4)"] = int(mx * MXFP4_REPACK_FACTOR)
    if mx_nonexpert:
        fp.nonexpert_by_type.pop("MXFP4")
        fp.nonexpert_by_type["Q5_0(from MXFP4)"] = int(
            mx_nonexpert * MXFP4_REPACK_FACTOR
        )
    fp.label += "+repack"
    return fp


def scan_quant(label: str, first_shard: str) -> QuantFootprint:
    """Scan every shard of a split GGUF, given the first one."""
    directory = os.path.dirname(first_shard)
    base = os.path.basename(first_shard)
    if "-00001-of-" not in base:
        shards = [first_shard]
    else:
        stem, _, tail = base.partition("-00001-of-")
        total = int(tail.split(".")[0])
        shards = [
            os.path.join(directory, f"{stem}-{i:05d}-of-{total:05d}.gguf")
            for i in range(1, total + 1)
        ]
    fp = QuantFootprint(label=label, path=first_shard, shards=shards)
    for shard in shards:
        if not os.path.exists(shard):
            raise GgufParseError(f"missing shard {shard}")
        fp.file_bytes += os.path.getsize(shard)
        for t in read_tensor_table(shard):
            n = t.n_bytes
            fp.by_type[t.type_name] = fp.by_type.get(t.type_name, 0) + n
            if is_expert_tensor(t.name):
                fp.expert_bytes += n
                fp.expert_tensors += 1
            else:
                fp.nonexpert_bytes += n
                fp.nonexpert_tensors += 1
                fp.nonexpert_by_type[t.type_name] = (
                    fp.nonexpert_by_type.get(t.type_name, 0) + n
                )
    return fp


# ---------------------------------------------------------------------------
# the rig model
# ---------------------------------------------------------------------------


@dataclass
class Rig:
    """Measured properties of this rig. Every number here is a measurement or
    a documented reservation, never a guess."""

    # NVML totals, MiB. Filled at runtime by --nvml or from --card-total.
    card_total_mib: List[float]
    card_names: List[str]
    ram_total_gib: float
    # Non-weight VRAM per card that the #417 window measured: torch peak minus
    # the weight stack, i.e. KV cache + activations + CUDA context + workspace.
    # Measured window 5: 5090 1.460 GiB non-torch + KV/activation headroom,
    # 3080s 0.640 GiB non-torch. We carry the reservation the boot recipe uses.
    reserve_mib: List[float]
    corridor_mib: float = 400.0  # #493: free VRAM floor on ALL cards


def solve_resident_fraction(
    fp: QuantFootprint,
    rig: Rig,
    tp_ratio: List[float],
    kv_and_activation_gib: List[float],
    extra_vram_gib: List[float] | None = None,
) -> Dict[str, object]:
    """How much of the expert stack fits in VRAM, and what spills to host?

    The split follows the boot recipe's uneven TP: rank r carries
    ``tp_ratio[r]`` of every sharded tensor class. Non-expert bytes are
    unavoidable; whatever VRAM is left after them, the reservation, the
    corridor and the KV/activation term is what resident experts may use.
    """
    ratios = [r / sum(tp_ratio) for r in tp_ratio]
    # Per-rank VRAM taken by something other than this model's weights: a
    # co-resident speculative draft head (#470 solo placement), graph pools,
    # anything else that must be paid before resident experts get what is left.
    extra = list(extra_vram_gib or [0.0] * len(ratios))
    per_rank = []
    total_resident = 0.0
    total_spill = 0.0
    feasible = True
    for r, share in enumerate(ratios):
        total_mib = rig.card_total_mib[r]
        nonexpert_gib = gib(fp.nonexpert_bytes) * share
        expert_gib = gib(fp.expert_bytes) * share
        budget_gib = (
            total_mib - rig.reserve_mib[r] - rig.corridor_mib
        ) / 1024.0 - kv_and_activation_gib[r]
        for_experts = budget_gib - nonexpert_gib - extra[r]
        if for_experts < 0:
            feasible = False
            resident = 0.0
        else:
            resident = min(for_experts, expert_gib)
        spill = expert_gib - resident
        frac = (resident / expert_gib) if expert_gib else 1.0
        total_resident += resident
        total_spill += spill
        per_rank.append(
            {
                "rank": r,
                "card": rig.card_names[r],
                "share": round(share, 4),
                "extra_vram_gib": round(extra[r], 3),
                "card_total_gib": round(total_mib / 1024.0, 3),
                "nonexpert_gib": round(nonexpert_gib, 3),
                "expert_gib": round(expert_gib, 3),
                "vram_for_experts_gib": round(for_experts, 3),
                "resident_gib": round(resident, 3),
                "spill_to_host_gib": round(spill, 3),
                "resident_fraction": round(frac, 4),
                "vram_overcommit_gib": round(-for_experts, 3) if for_experts < 0 else 0.0,
            }
        )
    return {
        "quant": fp.label,
        "checkpoint_gib": round(gib(fp.total_bytes), 3),
        "file_gib": round(gib(fp.file_bytes), 3),
        "expert_gib": round(gib(fp.expert_bytes), 3),
        "nonexpert_gib": round(gib(fp.nonexpert_bytes), 3),
        "expert_tensors": fp.expert_tensors,
        "nonexpert_tensors": fp.nonexpert_tensors,
        "by_type_gib": {k: round(gib(v), 3) for k, v in sorted(fp.by_type.items())},
        "nonexpert_by_type_gib": {
            k: round(gib(v), 3) for k, v in sorted(fp.nonexpert_by_type.items())
        },
        "per_rank": per_rank,
        "host_cold_pool_gib": round(total_spill, 3),
        "vram_resident_experts_gib": round(total_resident, 3),
        "vram_feasible": feasible,
        "resident_fraction_vector": ",".join(
            f"{p['resident_fraction']:.3f}" for p in per_rank
        ),
        # The host cold pool is a pinned/anonymous allocation
        # (expert_offload.py:1430 `pool.pin_memory()`). With SwapTotal 0 the
        # kernel cannot reclaim it -- gguf_shards.py:479-486 states that only
        # page cache can be taken. So this term, not the checkpoint size, is
        # what must fit in RAM alongside the runtime.
        "ram_headroom_gib": round(rig.ram_total_gib - total_spill, 3),
    }


def read_nvml() -> Tuple[List[float], List[str]]:
    import pynvml

    pynvml.nvmlInit()
    totals, names = [], []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        totals.append(mem.total / (1 << 20))
        name = pynvml.nvmlDeviceGetName(h)
        names.append(name.decode() if isinstance(name, bytes) else name)
    pynvml.nvmlShutdown()
    return totals, names


def read_ram_total_gib() -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / (1 << 20)
    raise RuntimeError("MemTotal not found in /proc/meminfo")


def selftest() -> int:
    """Can-discriminate check: the classifier and the coster must separate
    known-different inputs, or their verdicts do not count."""
    ok = True

    def check(label: str, cond: bool) -> None:
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        ok = ok and cond

    check("expert tensor recognised", is_expert_tensor("blk.7.ffn_down_exps.weight"))
    check(
        "shared expert is NOT offloadable",
        not is_expert_tensor("blk.7.ffn_down_shexp.weight"),
    )
    check("attention tensor not expert", not is_expert_tensor("blk.7.attn_q.weight"))
    check("embedding not expert", not is_expert_tensor("token_embd.weight"))

    q4k = TensorInfo("t", (256, 4), 12)
    check("Q4_K costing", q4k.n_bytes == 4 * 144)
    f32 = TensorInfo("t", (10,), 0)
    check("F32 costing", f32.n_bytes == 40)
    iq3 = TensorInfo("t", (256,), 18)
    check("IQ3_XXS costing", iq3.n_bytes == 98)
    check("distinct types cost differently", q4k.n_bytes != TensorInfo("t", (256, 4), 14).n_bytes)

    try:
        TensorInfo("t", (256,), 9999).n_bytes
        check("unknown ggml type raises", False)
    except GgufParseError:
        check("unknown ggml type raises", True)
    try:
        TensorInfo("t", (100,), 12).n_bytes
        check("non-multiple of block size raises", False)
    except GgufParseError:
        check("non-multiple of block size raises", True)

    # The solver must call an over-subscribed card infeasible, and a roomy one
    # feasible -- otherwise it cannot discriminate and its verdict is worthless.
    rig_small = Rig([4096.0], ["tiny"], 8.0, [1000.0])
    fp = QuantFootprint("t", "-")
    fp.nonexpert_bytes = 20 * (1 << 30)
    fp.expert_bytes = 10 * (1 << 30)
    check(
        "over-subscribed VRAM reported infeasible",
        not solve_resident_fraction(fp, rig_small, [1.0], [0.0])["vram_feasible"],
    )
    rig_big = Rig([131072.0], ["huge"], 256.0, [1000.0])
    check(
        "roomy VRAM reported feasible",
        solve_resident_fraction(fp, rig_big, [1.0], [0.0])["vram_feasible"],
    )
    fp_mx = QuantFootprint("mx", "-")
    fp_mx.expert_bytes = 17 * 1000
    fp_mx.by_type["MXFP4"] = 17 * 1000
    apply_mxfp4_repack(fp_mx)
    check("MXFP4 repack grows experts by 22/17", fp_mx.expert_bytes == 22 * 1000)
    fp_no = QuantFootprint("no", "-")
    fp_no.expert_bytes = 17 * 1000
    fp_no.by_type["Q4_K"] = 17 * 1000
    apply_mxfp4_repack(fp_no)
    check("no MXFP4 means no growth", fp_no.expert_bytes == 17 * 1000)

    rig_mid = Rig([32000.0], ["mid"], 104.0, [1000.0])
    fp2 = QuantFootprint("t2", "-")
    fp2.nonexpert_bytes = 3 * (1 << 30)
    fp2.expert_bytes = 40 * (1 << 30)
    base = solve_resident_fraction(fp2, rig_mid, [1.0], [2.0])
    cut = solve_resident_fraction(fp2, rig_mid, [1.0], [2.0], [10.0])
    check(
        "a co-resident draft head cuts residency",
        cut["per_rank"][0]["resident_gib"] < base["per_rank"][0]["resident_gib"],
    )
    check(
        "and moves exactly that much to the host pool",
        abs(
            (cut["host_cold_pool_gib"] - base["host_cold_pool_gib"]) - 10.0
        ) < 0.01,
    )

    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument(
        "--quant",
        action="append",
        default=[],
        metavar="LABEL=FIRST_SHARD",
        help="a quant tier to scan, e.g. q3kxl=/path/...-00001-of-00004.gguf",
    )
    ap.add_argument("--tp-ratio", default="0.438,0.281,0.281")
    ap.add_argument(
        "--rank-gpu-id",
        default=None,
        help="rank->NVML index map, e.g. 1,0,2. Resolve by UUID, never assume.",
    )
    ap.add_argument(
        "--kv-activation-gib",
        default="2.0,1.2,1.2",
        help="per-rank non-weight VRAM at forward peak (KV + activations + workspace)",
    )
    ap.add_argument("--reserve-mib", default="2200,1400,1400")
    ap.add_argument("--json-out", default=None)
    ap.add_argument(
        "--extra-vram-gib",
        default=None,
        help="per-rank VRAM taken by a non-weight consumer, e.g. a co-resident "
        "draft head: '10.12,0,0'",
    )
    ap.add_argument(
        "--repack-mxfp4",
        action="store_true",
        help="restate MXFP4 bytes as the Q5_0 the loader repacks them into",
    )
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.quant:
        ap.error("need at least one --quant LABEL=PATH (or --selftest)")

    totals, names = read_nvml()
    if args.rank_gpu_id:
        # rank r runs on NVML index rank_gpu_id[r]. The rig model is indexed by
        # RANK, so permute. Note the index space: --rank-gpu-id is CUDA-indexed
        # (server_args.py:8476-8477), and CUDA order (FASTEST_FIRST) is NOT NVML
        # order on this rig -- the 5090 is cuda:0 but nvidia-smi index 1. This
        # argument takes NVML indices because that is what read_nvml() returns,
        # so pass the NVML index, not the value you would give --rank-gpu-id.
        order = [int(x) for x in args.rank_gpu_id.split(",")]
        totals = [totals[i] for i in order]
        names = [names[i] for i in order]
    ram = read_ram_total_gib()
    tp_ratio = [float(x) for x in args.tp_ratio.split(",")]
    kv_act = [float(x) for x in args.kv_activation_gib.split(",")]
    reserve = [float(x) for x in args.reserve_mib.split(",")]
    rig = Rig(
        card_total_mib=totals,
        card_names=names,
        ram_total_gib=ram,
        reserve_mib=reserve,
    )

    report: Dict[str, object] = {
        "rig": {
            "cards": [
                {"index": i, "name": names[i], "total_gib": round(totals[i] / 1024, 3)}
                for i in range(len(totals))
            ],
            "ram_total_gib": round(ram, 3),
            "tp_ratio": tp_ratio,
            "kv_activation_gib": kv_act,
            "reserve_mib": reserve,
            "corridor_mib": rig.corridor_mib,
        },
        "quants": [],
    }
    for spec in args.quant:
        label, _, path = spec.partition("=")
        fp = scan_quant(label, path)
        if args.repack_mxfp4:
            fp = apply_mxfp4_repack(fp)
        extra = (
            [float(x) for x in args.extra_vram_gib.split(",")]
            if args.extra_vram_gib
            else None
        )
        report["quants"].append(
            solve_resident_fraction(fp, rig, tp_ratio, kv_act, extra)
        )

    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
