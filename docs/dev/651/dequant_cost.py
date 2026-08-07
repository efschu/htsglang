import re, collections
from gguf import GGUFReader
P="/spinning/llm_stuff/club-3090/models-cache/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
r=GGUFReader(P); MIB=1024*1024
q=0; dense=0; per_layer_q=collections.Counter(); per_layer_d=collections.Counter()
for t in r.tensors:
    ne=1
    for d in t.shape: ne*=int(d)
    q+=int(t.n_bytes); dense+=ne*2
    m=re.match(r"^blk\.(\d+)\.", t.name)
    if m and int(m.group(1))<40:
        per_layer_q[int(m.group(1))]+=int(t.n_bytes); per_layer_d[int(m.group(1))]+=ne*2
print(f"whole checkpoint quantized : {q/MIB:9.0f} MiB")
print(f"whole checkpoint as bf16   : {dense/MIB:9.0f} MiB   -> expansion {dense/q:.2f}x")
lq=sum(per_layer_q.values())/40; ld=sum(per_layer_d.values())/40
print(f"\nper decoder layer quantized: {lq/MIB:9.1f} MiB")
print(f"per decoder layer as bf16  : {ld/MIB:9.1f} MiB   -> expansion {ld/lq:.2f}x")
print("\nRAM cost of putting N layers on a CPU stage (must be dense bf16):")
for n in (1,2,4,8,16,24):
    print(f"  {n:2d} CPU layers: {n*ld/MIB:8.0f} MiB dense  (vs {n*lq/MIB:7.0f} MiB if it could stay quantized, +{n*(ld-lq)/MIB:7.0f} MiB)")
