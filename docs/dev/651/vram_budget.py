import re, collections
from gguf import GGUFReader
P="/spinning/llm_stuff/club-3090/models-cache/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
r=GGUFReader(P)
MIB=1024*1024

def nbytes(t):
    return int(t.n_bytes)

cat=collections.Counter()
for t in r.tensors:
    n=t.name
    m=re.match(r"^blk\.(\d+)\.(.*)$", n)
    if not m:
        cat["vocab (token_embd/output) + final norm"]+=nbytes(t); continue
    L=int(m.group(1)); rest=m.group(2)
    if L==40:
        cat["MTP draft block (blk.40)"]+=nbytes(t); continue
    if "_exps" in rest:      cat["routed experts (TP-sharded on intermediate)"]+=nbytes(t)
    elif "shexp" in rest:    cat["shared expert (TP-sharded)"]+=nbytes(t)
    elif "gate_inp" in rest: cat["router gates (replicated)"]+=nbytes(t)
    elif rest.startswith("ssm") or rest.startswith("attn_gate"): cat["GDN linear-attn (TP-sharded)"]+=nbytes(t)
    elif rest.startswith("attn"): cat["full attention (TP-sharded)"]+=nbytes(t)
    else: cat["norms (replicated)"]+=nbytes(t)

tot=sum(cat.values())
print(f"{'CATEGORY':52s} {'MiB':>10s}  {'%':>6s}")
for k,v in sorted(cat.items(), key=lambda x:-x[1]):
    print(f"{k:52s} {v/MIB:10.1f}  {100*v/tot:5.1f}%")
print(f"{'TOTAL':52s} {tot/MIB:10.1f}")
print(f"\nfile size on disk: {22853663008/MIB:.1f} MiB")

# ---- KV cache -------------------------------------------------------------
# full attention layers only; hybrid model, full_attention_interval=4
n_layers=40; interval=4
full_layers=[i for i in range(n_layers) if (i+1)%interval==0]
kv_heads=2; head_dim=256
print(f"\nfull-attention layers: {len(full_layers)} of {n_layers} -> {full_layers}")
for name,b in (("fp16/bf16",2),("fp8_e4m3",1)):
    per_tok = 2*kv_heads*head_dim*b*len(full_layers)   # K and V
    print(f"  KV per token, ALL ranks, {name:9s}: {per_tok/1024:8.2f} KiB   "
          f"({per_tok*8192/MIB:7.1f} MiB @8k ctx, {per_tok*262144/MIB:8.1f} MiB @262k)")

# ---- GDN / mamba state ----------------------------------------------------
lin_layers=n_layers-len(full_layers)
k_heads=16; v_heads=32; k_dim=128; v_dim=128; conv_k=4
key_dim=k_heads*k_dim; value_dim=v_heads*v_dim; conv_dim=2*key_dim+value_dim
ssm_elems=v_heads*k_dim*v_dim
print(f"\nGDN layers: {lin_layers}   conv_dim={conv_dim}  ssm elems/layer={ssm_elems}")
for name,b in (("float32",4),("bfloat16",2)):
    ssm=ssm_elems*b*lin_layers
    conv=conv_dim*conv_k*2*lin_layers
    print(f"  GDN state per SEQUENCE, {name:8s}: ssm {ssm/MIB:7.1f} MiB + conv {conv/MIB:5.1f} MiB "
          f"= {(ssm+conv)/MIB:7.1f} MiB")

# ---- vision tower ---------------------------------------------------------
d,h,inter,merge_out=27,1152,4304,2048
per=h*h*3 + h*h + h*inter + inter*h
vis=per*d + 3*16*16*2*h + 2304*h + (h*4)*merge_out + merge_out*merge_out
print(f"\nvision tower (built but NEVER used on this text-only file):")
print(f"  ~{vis/1e6:.0f}M params -> {vis*2/MIB:.0f} MiB bf16 PER RANK")
