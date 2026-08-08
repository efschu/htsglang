"""#651 9.7: is there a COLD TAIL in expert routing on this checkpoint?

Reads the expert-distribution recorder dump and answers the one question the
MoE disk-spill design depends on: can a minority of experts be moved to NVMe
without most tokens touching one?

The number that matters is not "how skewed is the distribution" in the
abstract, but: if we spill the coldest experts holding X GiB, what fraction of
expert LOOKUPS miss to disk? That is what sets the per-token I/O cost.
"""
import sys

import torch

path = sys.argv[1]
d = torch.load(path, map_location="cpu", weights_only=False)
print("top-level type:", type(d).__name__)
if isinstance(d, dict):
    for k, v in d.items():
        shape = getattr(v, "shape", None)
        print(f"  {k}: {type(v).__name__} shape={shape}")

# The stat mode stores a per-(layer, expert) count matrix somewhere in here.
def find_counts(obj):
    if isinstance(obj, torch.Tensor) and obj.dim() >= 2 and obj.numel() > 16:
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            r = find_counts(v)
            if r is not None:
                return r
    if isinstance(obj, (list, tuple)):
        for v in obj:
            r = find_counts(v)
            if r is not None:
                return r
    return None

counts = find_counts(d)
if counts is None:
    print("no count matrix found")
    sys.exit(1)

counts = counts.float()
print("counts shape:", tuple(counts.shape), "total:", counts.sum().item())

# Collapse to (layer, expert) if there is a leading pass/token axis.
while counts.dim() > 2:
    counts = counts.sum(dim=0)
n_layers, n_experts = counts.shape
print(f"layers={n_layers} experts_per_layer={n_experts}")

total = counts.sum().item()
if total <= 0:
    print("no routing recorded")
    sys.exit(1)

# Per layer, sort experts coldest-first and report what spilling the coldest
# fraction would cost in missed lookups.
print()
print("  spill%   miss%   (share of expert LOOKUPS that would hit disk)")
flat = counts.flatten()
order = torch.argsort(flat)          # coldest first
sorted_counts = flat[order]
cum = torch.cumsum(sorted_counts, 0) / total
n = flat.numel()
for frac in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
    idx = int(n * frac) - 1
    if idx < 0:
        continue
    print(f"   {frac*100:5.0f}   {cum[idx].item()*100:6.2f}")

# A flat distribution would give miss% == spill%. Report the ratio.
idx10 = int(n * 0.10) - 1
ratio = (cum[idx10].item()) / 0.10 if idx10 >= 0 else float("nan")
print()
print(f"coldest-10% lookup share = {cum[idx10].item()*100:.2f}%  "
      f"(flat would be 10%); skew ratio = {ratio:.3f}")
zero = int((flat == 0).sum().item())
print(f"experts never routed to: {zero} / {n} ({zero/n*100:.1f}%)")
