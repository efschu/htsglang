#!/usr/bin/env python
"""#651: read a GGUF's tensor inventory from the HEADER alone.

`gguf.GGUFReader` memory-maps and reshapes every tensor's DATA, so it cannot
read a file that is still downloading, and it is needlessly expensive on a
21 GiB file when the question is only "which tensors, of which type". The
header answers that and sits entirely in the first few MiB.

Written for one specific question on this checkpoint: blk.40 is the NEXTN/MTP
draft block, and its ffn_gate_inp / ffn_gate_inp_shexp router gates are stored
BF16 rather than F32. That is the whole reason the #647 defect bites here (the
shared iterator renames every non-F32 `.weight` to `.qweight`, the dense gate
lands on a parameter no module owns, and the draft router stays uninitialized
while the model stays fluent). Confirming the dtypes on the real file turns
that from a claim in a commit message into a measured property.

    python gguf_header.py <file.gguf> [name-substring ...]
"""

from __future__ import annotations

import struct
import sys

# ggml_type enum. Only the ones this fork can meet are named; anything else is
# reported by number rather than guessed at.
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16",
}

# GGUF metadata value types, needed only to SKIP the kv block correctly.
_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class _R:
    def __init__(self, f):
        self.f = f

    def raw(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("header truncated -- not enough bytes downloaded yet")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self) -> str:
        return self.raw(self.u64()).decode("utf-8", "replace")

    def skip_value(self, vt: int) -> None:
        if vt in _FIXED:
            self.raw(_FIXED[vt])
        elif vt == 8:  # string
            self.string()
        elif vt == 9:  # array
            et = self.u32()
            n = self.u64()
            if et in _FIXED:
                self.raw(_FIXED[et] * n)
            elif et == 8:
                for _ in range(n):
                    self.string()
            else:
                raise ValueError(f"unsupported array element type {et}")
        else:
            raise ValueError(f"unsupported value type {vt}")


def read_header(path: str):
    with open(path, "rb") as f:
        r = _R(f)
        if r.raw(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        kv = {}
        for _ in range(n_kv):
            key = r.string()
            vt = r.u32()
            if vt == 8:
                kv[key] = r.string()
            elif vt in (4, 5, 10, 11):
                raw = r.raw(_FIXED[vt])
                kv[key] = int.from_bytes(raw, "little", signed=vt in (5, 11))
            else:
                r.skip_value(vt)
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.u32()
            dims = tuple(r.u64() for _ in range(nd))
            ttype = r.u32()
            r.u64()  # offset
            tensors.append((name, GGML_TYPES.get(ttype, f"type{ttype}"), dims))
    return version, kv, tensors


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    filters = sys.argv[2:]
    version, kv, tensors = read_header(path)

    print(f"gguf v{version}, {len(tensors)} tensors")
    for k in ("general.architecture", "general.name"):
        if k in kv:
            print(f"  {k}: {kv[k]}")
    for k, v in sorted(kv.items()):
        if "block_count" in k or "expert" in k or "nextn" in k.lower():
            print(f"  {k}: {v}")

    types: dict[str, int] = {}
    for _, t, _ in tensors:
        types[t] = types.get(t, 0) + 1
    print("  type histogram:", ", ".join(f"{t}={n}" for t, n in sorted(types.items())))

    # The point of the exercise: every non-F32 tensor that is NOT a quantized
    # weight is a rename candidate, and on this file that set should be exactly
    # the two MTP router gates.
    nonf32_dense = [
        (n, t, d) for n, t, d in tensors if t in ("BF16", "F16") and "_exps" not in n
    ]
    print(f"\n  non-F32 float (rename-hazard) tensors: {len(nonf32_dense)}")
    for n, t, d in nonf32_dense:
        print(f"    {n:38s} {t:6s} {d}")

    if filters:
        print()
        for f in filters:
            sel = [(n, t, d) for n, t, d in tensors if f in n]
            print(f"  '{f}': {len(sel)} tensor(s)")
            for n, t, d in sorted(sel):
                print(f"    {n:38s} {t:6s} {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
