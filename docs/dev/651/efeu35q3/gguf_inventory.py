#!/usr/bin/env python
"""efeu-TP14 / Qwen3.8-35B-A3B Q3_K_M: header-only GGUF inventory.

Answers, from the header alone (works on a partially downloaded file):
  * the architecture metadata (every scalar kv, arrays by length), so the
    "same arch as Qwen3.6-35B-A3B?" question is a diff of two printouts;
  * the tensor-type histogram;
  * per ROLE (blk.N. prefix stripped) which ggml types occur on which layers,
    flagging the types that are unsafe on gfx1103 by the #651 laws:
        Q6_K   -> nondeterministically WRONG (dequant / MMVQ / moe_a8)  => requant Q8_0
        Q5_K   -> rare per-launch errors (~0.05 %/launch)               => decide
        IQ*    -> MMQ catastrophically broken (non-finite)              => requant
        BF16/F16 dense -> #647 rename hazard (router gates)

    python gguf_inventory.py <file.gguf> [--json out.json]
"""

from __future__ import annotations

import json
import re
import struct
import sys

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16",
}
# value types: 0 u8 1 i8 2 u16 3 i16 4 u32 5 i32 6 f32 7 bool 8 str 9 arr
# 10 u64 11 i64 12 f64
_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
        7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

UNSAFE = {
    "Q6_K": "UNSAFE gfx1103: nondet WRONG -> requant Q8_0",
    "Q5_K": "RISK gfx1103: rare per-launch errors",
    "BF16": "rename-hazard (#647) if dense",
    "F16": "rename-hazard (#647) if dense",
}


class _R:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("header truncated")
        return b

    def val(self, vt):
        if vt in _FMT:
            fmt = _FMT[vt]
            return struct.unpack(fmt, self.raw(struct.calcsize(fmt)))[0]
        if vt == 8:
            return self.raw(struct.unpack("<Q", self.raw(8))[0]).decode("utf-8", "replace")
        if vt == 9:
            et = struct.unpack("<I", self.raw(4))[0]
            n = struct.unpack("<Q", self.raw(8))[0]
            items = [self.val(et) for _ in range(n)]
            if n > 16:
                return {"__array_len__": n, "head": items[:4] if et != 8 else None}
            return items
        raise ValueError(f"value type {vt}")


def read_header(path):
    with open(path, "rb") as f:
        r = _R(f)
        if r.raw(4) != b"GGUF":
            raise SystemExit("not GGUF")
        version = struct.unpack("<I", r.raw(4))[0]
        n_t = struct.unpack("<Q", r.raw(8))[0]
        n_kv = struct.unpack("<Q", r.raw(8))[0]
        kv = {}
        for _ in range(n_kv):
            k = r.val(8)
            vt = struct.unpack("<I", r.raw(4))[0]
            kv[k] = r.val(vt)
        tensors = []
        for _ in range(n_t):
            name = r.val(8)
            nd = struct.unpack("<I", r.raw(4))[0]
            dims = tuple(struct.unpack("<Q", r.raw(8))[0] for _ in range(nd))
            tt = struct.unpack("<I", r.raw(4))[0]
            r.raw(8)
            tensors.append((name, GGML_TYPES.get(tt, f"type{tt}"), dims))
    return version, kv, tensors


def main():
    path = sys.argv[1]
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    version, kv, tensors = read_header(path)
    print(f"gguf v{version}, {len(tensors)} tensors, {len(kv)} kv")
    for k in sorted(kv):
        if k.startswith("tokenizer.") and isinstance(kv[k], dict):
            continue
        v = kv[k]
        if isinstance(v, str) and len(v) > 120:
            v = v[:120] + f"...(+{len(kv[k]) - 120})"
        print(f"  {k} = {v}")
    hist = {}
    for _, t, _ in tensors:
        hist[t] = hist.get(t, 0) + 1
    print("type histogram:", ", ".join(f"{t}={n}" for t, n in sorted(hist.items())))
    roles = {}
    for n, t, d in tensors:
        m = re.match(r"blk\.(\d+)\.(.*)", n)
        layer, role = (int(m.group(1)), m.group(2)) if m else (None, n)
        roles.setdefault(role, {}).setdefault(t, []).append(layer)
    print("per-role types (layers):")
    for role in sorted(roles):
        for t, layers in sorted(roles[role].items()):
            ls = [x for x in layers if x is not None]
            where = (f"{len(ls)} layers" + (f" {ls}" if len(ls) <= 12 else "")) if ls else "global"
            flag = UNSAFE.get(t, "")
            if t.startswith("IQ"):
                flag = "UNSAFE gfx1103: IQ MMQ non-finite -> requant"
            print(f"  {role:34s} {t:7s} {where} {flag}")
    if out_json:
        json.dump({"version": version, "kv": kv, "hist": hist,
                   "tensors": [[n, t, list(d)] for n, t, d in tensors]},
                  open(out_json, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
