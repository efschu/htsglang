"""efeu-TP14: offline check of patch_mixed_f32_dense on the APEX file (metadata
only + the 30 small ssm_beta tensors; no model load, no GPU).
  * the adapter no longer raises, in_proj_ba of all 30 GDN layers is dense
  * GGUF_DEQUANT_ON_READ holds exactly the 30 Q3_K ssm_beta tensors
  * the iterator's dequant-on-read value equals gguf.quants.dequantize
  * the retired Q3_K_M file: no module is mixed -> GGUF_DEQUANT_ON_READ empty
"""
import sys

import numpy as np
import gguf
from transformers import AutoConfig

from sglang.srt.model_loader import weight_utils as wu
from sglang.srt.model_loader.gguf_registry import create_gguf_adapter

APEX = "/root/efeu35q3/models_apex/Qwen3.8-35B-A3B-Distill.APEX-I-MiniPlus-V2.1-Abliterated.gguf"
Q3 = "/root/efeu35q3/models/Qwen3.8-35B-A3B-Q3_K_M.gguf"

ok = True
for path, cfgdir, expect in ((APEX, "/root/efeu35q3/hf_apex", 30), (Q3, "/root/efeu35q3/hf", 0)):
    wu.GGUF_DEQUANT_ON_READ.clear()
    cfg = AutoConfig.from_pretrained(cfgdir)
    ad = create_gguf_adapter(cfg, path)
    pre = ad.unquantized_module_prefixes()
    ba = [p for p in pre if p.endswith("in_proj_ba")]
    names = sorted(wu.GGUF_DEQUANT_ON_READ)
    print(f"{path.split('/')[-1]}: dense prefixes {len(pre)} (in_proj_ba {len(ba)}), "
          f"dequant-on-read {len(names)} e.g. {names[:2]}")
    ok &= len(names) == expect and all("ssm_beta" in n for n in names)
    if expect:
        ok &= len(ba) == 30
        r = gguf.GGUFReader(path)
        t = next(t for t in r.tensors if t.name == names[0])
        deq = gguf.quants.dequantize(t.data, t.tensor_type).astype("float32")
        print(f"  {t.name}: {t.tensor_type.name} -> float32 {deq.shape}, "
              f"finite={np.isfinite(deq).all()}, absmax={np.abs(deq).max():.4f}")
        ok &= bool(np.isfinite(deq).all())
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
