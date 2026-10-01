"""efeu-TP14 (01.10. 22:59): load GGUF modules whose fused splits mix F32 and a
quantized type -- APEX-I-MiniPlus-V2.1 stores ssm_alpha F32 and ssm_beta Q3_K,
which fuse into linear_attn.in_proj_ba. The adapter refused ("mixes an F32
split with a quantized sibling ... cannot be loaded either dense or
quantized").

Fix: build such a module DENSE (as for an all-F32 module) and dequantize the
quantized sibling(s) on read (gguf.quants.dequantize, float32, CPU) so the
stream hands it over as a plain `.weight`. in_proj_ba is tiny (2 x 32 rows x
2048 per GDN layer), so the dense cost is nil and the result is exactly the
quantized tensor's value.

  * gguf_adapter_base.unquantized_module_prefixes: instead of raising, record
    the quantized gguf tensor names of the mixed module in
    weight_utils.GGUF_DEQUANT_ON_READ and keep the module in the dense set.
  * weight_utils.gguf_quant_weights_iterator: names in GGUF_DEQUANT_ON_READ
    emit no `.qweight_type` and are yielded dequantized as `.weight`.
"""
import ast
import os
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/efeu35q3/sglang_src/python"
AB = os.path.join(ROOT, "sglang/srt/model_loader/gguf_adapter_base.py")
WU = os.path.join(ROOT, "sglang/srt/model_loader/weight_utils.py")


def sub(s, old, new, label):
    assert s.count(old) == 1, f"{label}: anchor found {s.count(old)}x"
    return s.replace(old, new, 1)


a = open(AB).read()
assert "GGUF_DEQUANT_ON_READ" not in a, "already patched"
a = sub(a, """        prefixes = set()
        module_split_types: Dict[str, set] = {}
        for gname, hf in name_map.items():""", """        prefixes = set()
        module_split_types: Dict[str, set] = {}
        module_split_names: Dict[str, list] = {}
        for gname, hf in name_map.items():""", "split names init")
a = sub(a, """            module_split_types.setdefault(base, set()).add(types.get(gname))
""", """            module_split_types.setdefault(base, set()).add(types.get(gname))
            module_split_names.setdefault(base, []).append(gname)
""", "split names collect")
a = sub(a, """            if any(t is not None and t not in unq_types for t in tset):
                raise RuntimeError(
                    f"{self.FAMILY} GGUF: module {base!r} mixes an F32 split "
                    f"with a quantized sibling "
                    f"({sorted(t.name for t in tset if t is not None)}); "
                    "this cannot be loaded either dense or quantized. "
                    "Re-quantize the F32 tensor (e.g. to Q8_0/F16)."
                )
""", """            if any(t is not None and t not in unq_types for t in tset):
                # efeu-TP14: build the module dense and dequantize the
                # quantized sibling(s) on read (weight_utils.GGUF_DEQUANT_ON_READ).
                from sglang.srt.model_loader import weight_utils as _wu

                quant = [
                    g for g in module_split_names.get(base, [])
                    if types.get(g) is not None and types.get(g) not in unq_types
                ]
                _wu.GGUF_DEQUANT_ON_READ.update(quant)
                logger.info(
                    "%s GGUF: module %r mixes F32 with %s; loading it dense, "
                    "dequantizing %d quantized split(s) on read",
                    self.FAMILY, base,
                    sorted(t.name for t in tset if t is not None), len(quant),
                )
""", "raise -> dense")
ast.parse(a)
if "\nlogger = " not in a:
    raise SystemExit("gguf_adapter_base has no module logger")

w = open(WU).read()
assert "GGUF_DEQUANT_ON_READ" not in w
w = sub(w, """def gguf_is_dense_unquantized_target(name: str, weight_type) -> bool:""",
        """#: efeu-TP14: gguf tensor names that must be streamed DEQUANTIZED as a dense
#: `.weight` (filled by GGUFAdapterBase.unquantized_module_prefixes for fused
#: modules that mix an F32 split with a quantized one).
GGUF_DEQUANT_ON_READ: set = set()


def gguf_is_dense_unquantized_target(name: str, weight_type) -> bool:""", "global")
w = sub(w, """            if weight_type.name != "F32" and not gguf_is_dense_unquantized_target(
                name, weight_type
            ):""", """            if (
                weight_type.name != "F32"
                and tensor_name not in GGUF_DEQUANT_ON_READ
                and not gguf_is_dense_unquantized_target(name, weight_type)
            ):""", "pass1")
w = sub(w, """                if gguf_is_dense_unquantized_target(name, weight_type):""",
        """                if tensor_name in GGUF_DEQUANT_ON_READ:
                    # efeu-TP14: quantized split of a dense (mixed F32) module
                    param = torch.from_numpy(
                        gguf.quants.dequantize(weight, source_type).astype("float32")
                    )
                elif gguf_is_dense_unquantized_target(name, weight_type):""", "pass2")
ast.parse(w)

if "--dry" in sys.argv:
    print("dry run ok")
    sys.exit(0)
for p, t in ((AB, a), (WU, w)):
    if not os.path.exists(p + ".orig-efeu-mixed"):
        shutil.copy(p, p + ".orig-efeu-mixed")
    open(p + ".new", "w").write(t)
    os.replace(p + ".new", p)
print("mixed F32/quantized fused modules: dense + dequant-on-read (active at next start)")
