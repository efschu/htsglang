import re, collections
from gguf import GGUFReader
P="/spinning/llm_stuff/club-3090/models-cache/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
r=GGUFReader(P); MIB=1024*1024
per_layer=collections.Counter(); nonlayer=0; mtp=0
for t in r.tensors:
    m=re.match(r"^blk\.(\d+)\.", t.name)
    if not m: nonlayer+=int(t.n_bytes); continue
    L=int(m.group(1))
    if L==40: mtp+=int(t.n_bytes)
    else: per_layer[L]+=int(t.n_bytes)
vals=[v/MIB for v in per_layer.values()]
gdn=[per_layer[i]/MIB for i in range(40) if (i+1)%4!=0]
att=[per_layer[i]/MIB for i in range(40) if (i+1)%4==0]
print(f"40 decoder layers: total {sum(vals):.0f} MiB, mean {sum(vals)/40:.1f} MiB/layer")
print(f"  GDN layers   (30): mean {sum(gdn)/len(gdn):.1f} MiB  min {min(gdn):.1f} max {max(gdn):.1f}")
print(f"  attn layers  (10): mean {sum(att)/len(att):.1f} MiB  min {min(att):.1f} max {max(att):.1f}")
print(f"non-layer (vocab+final norm): {nonlayer/MIB:.0f} MiB")
print(f"MTP block blk.40           : {mtp/MIB:.0f} MiB")
print()
print("How many decoder layers fit in a GPU memory carve-out (weights only,")
print("before KV / activations / context):")
mean=sum(vals)/40
for gib in (4,6,8,10,12,16,20,24):
    mib=gib*1024
    print(f"  {gib:2d} GiB -> {int(mib//mean):2d} of 40 layers   (remainder {40-int(mib//mean):2d} on CPU)")
print()
print("Compute-weighted balance: stage times equal when L_gpu/L_cpu = R_gpu/R_cpu")
for ratio in (5,10,20,50,100):
    lg = 40*ratio/(ratio+1); lc = 40/(ratio+1)
    print(f"  R_gpu/R_cpu = {ratio:3d}x -> balanced split {lg:5.1f} GPU / {lc:4.1f} CPU layers")
