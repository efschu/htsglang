#!/usr/bin/env python3
"""
Generator for the topology diagrams referenced by ../TOPOLOGIES.md.

Every diagram is a self-contained SVG: inline shapes + text only, no external
fonts, no external images, no <image href> to remote hosts. GPU boxes are scaled
by VRAM (box height = VRAM_GB * PXGB) and partitioned into labelled regions with
a fixed, consistent colour meaning across every diagram (see COLORS / LEGEND).

Region sizes are illustrative partitionings of a card's VRAM, not measured
allocations, unless a number is quoted directly from FEATURES_VS_UPSTREAM.md.
The prose in TOPOLOGIES.md states which is which.
"""

import os

OUT = os.path.dirname(os.path.abspath(__file__))
PXGB = 4.4            # pixels per GB of VRAM (vertical scale for GPU boxes)
BOXW = 150           # GPU box width
SLOT = 205           # horizontal slot per GPU (box + gap)
LEFT = 40            # left margin before first card

# Fixed colour meaning, used identically in every diagram.
COLORS = {
    "weights":   "#3b6fb0",  # model-weight shard (TP)
    "kv":        "#4a9d5b",  # KV cache (on-GPU)
    "resident":  "#e08a2b",  # resident experts (MoE, on-GPU)
    "scratch":   "#f2c07a",  # expert scratch / prefetch staging buffer
    "spill":     "#c0504d",  # experts spilled to pinned host RAM
    "hostkv":    "#3f9fa0",  # host-staged / transferred KV
    "free":      "#d3dae1",  # unused / reserve headroom
    "ctx":       "#98a2ab",  # CUDA context + framework overhead
}
# Human labels for the legend, in a stable order.
LEGEND = [
    ("weights",  "model-weight shard"),
    ("kv",       "KV cache (on-GPU)"),
    ("resident", "resident experts (MoE)"),
    ("scratch",  "expert scratch / prefetch"),
    ("spill",    "experts spilled to host RAM"),
    ("hostkv",   "host-staged KV"),
    ("free",     "free / reserve headroom"),
    ("ctx",      "CUDA context + overhead"),
]

FONT = 'font-family="DejaVu Sans, Verdana, Arial, sans-serif"'


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def text(x, y, s, size=11, anchor="start", weight="normal", color="#1a1a1a"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" {FONT} font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" '
            f'fill="{color}">{esc(s)}</text>')


def rect(x, y, w, h, fill, stroke="#2b2b2b", sw=1.0, rx=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def line(x1, y1, x2, y2, stroke="#555", sw=1.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def gpu_column(x, top, name, vram_gb, regions, caption=None):
    """Draw one GPU box scaled to vram_gb, stacked top-down with regions.

    regions: list of (colorkey, gb, label). Their sum should be <= vram_gb;
    the remainder is drawn as 'free'. A gb of 0 is skipped.
    """
    out = []
    h = vram_gb * PXGB
    # Card title (two lines above the box).
    out.append(text(x + BOXW / 2, top - 20, name, size=12.5, anchor="middle", weight="bold"))
    out.append(text(x + BOXW / 2, top - 7, f"{vram_gb} GB VRAM", size=10.5, anchor="middle", color="#444"))
    # Outer box.
    out.append(rect(x, top, BOXW, h, "none", stroke="#222", sw=1.6))
    used = sum(g for _, g, _ in regions)
    filled = list(regions)
    if used < vram_gb - 0.05:
        filled = filled + [("free", vram_gb - used, "free")]
    cy = top
    for key, gb, label in filled:
        if gb <= 0:
            continue
        rh = gb * PXGB
        out.append(rect(x, cy, BOXW, rh, COLORS[key]))
        if label and rh >= 15:
            tc = "#ffffff" if key in ("weights", "kv", "resident", "spill", "hostkv") else "#1a1a1a"
            out.append(text(x + BOXW / 2, cy + rh / 2 + 3.6, f"{label}", size=10, anchor="middle", color=tc))
        elif label:
            # too thin to hold text: side label with a leader
            out.append(line(x + BOXW, cy + rh / 2, x + BOXW + 8, cy + rh / 2, stroke="#888", sw=0.8))
            out.append(text(x + BOXW + 11, cy + rh / 2 + 3.4, f"{label}", size=9, anchor="start", color="#333"))
        cy += rh
    if caption:
        out.append(text(x + BOXW / 2, top + h + 15, caption, size=9.5, anchor="middle", color="#333"))
    return "".join(out), top + h


def host_ram_bar(x, y, w, regions, title="Pinned host RAM (DDR)"):
    """Horizontal host-RAM bar partitioned left-to-right by (colorkey, frac, label)."""
    out = []
    barh = 34
    out.append(text(x, y - 6, title, size=10.5, anchor="start", weight="bold", color="#333"))
    out.append(rect(x, y, w, barh, "none", stroke="#222", sw=1.4))
    cx = x
    for key, frac, label in regions:
        rw = w * frac
        out.append(rect(cx, y, rw, barh, COLORS[key]))
        tc = "#ffffff" if key in ("spill", "hostkv") else "#1a1a1a"
        if rw >= 40:
            out.append(text(cx + rw / 2, y + barh / 2 + 3.6, label, size=9.5, anchor="middle", color=tc))
        cx += rw
    return "".join(out), y + barh


def legend(x, y, keys):
    out = []
    out.append(text(x, y, "Legend (colour meaning is identical in every diagram)",
                    size=9.5, anchor="start", weight="bold", color="#333"))
    cx, cy = x, y + 12
    per_row = 4
    col_w = 150
    for i, k in enumerate(keys):
        label = dict(LEGEND)[k]
        col = i % per_row
        row = i // per_row
        px = x + col * col_w
        py = cy + row * 16
        out.append(rect(px, py - 9, 12, 12, COLORS[k], stroke="#333", sw=0.8, rx=1))
        out.append(text(px + 17, py + 1, label, size=9, anchor="start", color="#333"))
    rows = (len(keys) + per_row - 1) // per_row
    return "".join(out), cy + rows * 16


def pcie_link(x1, y1, x2, y2, label, kind="x16"):
    """Interconnect line with a label. kind sets the visual weight/dash."""
    styles = {
        "x16": (2.4, None, "#5a5a5a"),
        "x4":  (1.4, "4 3", "#8a5a2b"),
        "nvlink": (4.0, None, "#227a3a"),
        "bus": (2.0, None, "#5a5a5a"),
    }
    sw, dash, col = styles[kind]
    out = [line(x1, y1, x2, y2, stroke=col, sw=sw, dash=dash)]
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    out.append(text(mx, my - 4, label, size=8.5, anchor="middle", color=col))
    return "".join(out)


def svg(width, height, body, title, subtitle=None):
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" font-family="sans-serif">')
    bg = rect(0, 0, width, height, "#ffffff", stroke="none", sw=0, rx=0)
    t = text(20, 24, title, size=15, anchor="start", weight="bold")
    s = text(20, 41, subtitle, size=10.5, anchor="start", color="#555") if subtitle else ""
    return f'{head}{bg}{t}{s}{body}</svg>'


def write(name, content):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        f.write(content)
    print("wrote", p)


# ---------------------------------------------------------------------------
# Scenario 1 — two mismatched GPUs, PCIe, no NVLink -> uneven Tensor Parallel
# ---------------------------------------------------------------------------
def s1_uneven_tp():
    W, H = 680, 430
    body = []
    topY = 110
    # Fork side: proportional shards. 5090 carries ~62%, 3080 ~38% (32:20 -> 8:5).
    x0 = LEFT
    c, _ = gpu_column(x0, topY, "RTX 5090", 32,
                      [("weights", 13, "weight shard (rank0)"),
                       ("kv", 13, "KV cache"),
                       ("ctx", 2, "ctx")],
                      caption="rank 0  •  ratio 8")
    body.append(c)
    x1 = LEFT + SLOT
    c, _ = gpu_column(x1, topY + (32 - 20) * PXGB, "RTX 3080", 20,
                      [("weights", 8, "weight shard (rank1)"),
                       ("kv", 8, "KV cache"),
                       ("ctx", 2, "ctx")],
                      caption="rank 1  •  ratio 5")
    body.append(c)
    # interconnect between the two cards
    body.append(pcie_link(x0 + BOXW, topY + 60, x1, topY + 60,
                          "TP collective — NCCL over PCIe host-staging (no NVLink)", "x16"))
    body.append(text(20, 60, "--rank-tp-ratio 8,5   (shards sized to each card, not to the smallest)",
                     size=10.5, color="#333"))
    body.append(text(20, 76, "Stock sglang: even TP only — both ranks capped to the 3080's shard; "
                             "~12 GB of the 5090 goes unused (grey).",
                     size=10.5, color="#8a2b2b"))
    # ghost of wasted 5090 region under stock, drawn faintly to the right
    lx = LEFT + 2 * SLOT + 10
    body.append(text(lx + 60, topY - 20, "stock even-TP", size=11, anchor="middle", weight="bold", color="#8a2b2b"))
    body.append(rect(lx, topY, 120, 32 * PXGB, "none", stroke="#8a2b2b", sw=1.4, dash="5 3"))
    body.append(rect(lx, topY, 120, 8 * PXGB, COLORS["weights"]))
    body.append(text(lx + 60, topY + 8 * PXGB / 2 + 3.6, "weights (capped)", size=9, anchor="middle", color="#fff"))
    body.append(rect(lx, topY + 8 * PXGB, 120, 8 * PXGB, COLORS["kv"]))
    body.append(text(lx + 60, topY + 8 * PXGB + 8 * PXGB / 2 + 3.6, "KV (capped)", size=9, anchor="middle", color="#fff"))
    body.append(rect(lx, topY + 16 * PXGB, 120, 16 * PXGB, "none", stroke="#8a2b2b", sw=1.0, dash="4 3"))
    body.append(text(lx + 60, topY + 24 * PXGB, "WASTED", size=11, anchor="middle", weight="bold", color="#8a2b2b"))
    body.append(text(lx + 60, topY + 24 * PXGB + 14, "~12 GB idle", size=9, anchor="middle", color="#8a2b2b"))
    leg, ly = legend(20, H - 46, ["weights", "kv", "ctx", "free"])
    body.append(leg)
    write("01-uneven-tp.svg",
          svg(W, H, "".join(body),
              "1 — Two mismatched GPUs, PCIe, no NVLink → uneven Tensor Parallelism",
              "RTX 5090 32 GB + RTX 3080 20 GB. Decisive setting: --rank-tp-ratio (proportional shards)."))


# ---------------------------------------------------------------------------
# Scenario 2 — three mismatched GPUs -> uneven-TP auto + uneven-DCP token KV
# ---------------------------------------------------------------------------
def s2_uneven_dcp():
    W, H = 680, 440
    body = []
    topY = 120
    cards = [("RTX 5090", 32, 0, "rank 0"), ("RTX 3080", 20, 1, "rank 1"), ("RTX 3080", 20, 2, "rank 2")]
    for i, (nm, vram, r, cap) in enumerate(cards):
        x = LEFT + i * SLOT
        wsh = 11 if vram == 32 else 7
        kv = vram - wsh - 2
        c, _ = gpu_column(x, topY + (32 - vram) * PXGB, nm, vram,
                          [("weights", wsh, "weight shard"),
                           ("kv", kv, "KV token-shard"),
                           ("ctx", 2, "ctx")],
                          caption=f"{cap}  •  KV owns a proportional token slice")
        body.append(c)
        if i > 0:
            xp = LEFT + (i - 1) * SLOT
            body.append(pcie_link(xp + BOXW, topY + 150, x, topY + 150, "DCP LSE-merge (PCIe)", "x16"))
    body.append(text(20, 58, "--rank-tp-ratio auto  +  uneven-DCP token sharding "
                             "(KV split along the TOKEN axis, not the KV-head axis).",
                     size=10.5, color="#333"))
    body.append(text(20, 74, "Fills every card; the big card is not throttled to the small ones. "
                             "Doc: ≈+2.5-3x KV context vs a naive equal split.",
                     size=10.5, color="#2b6b3a"))
    body.append(text(20, 90, "Stock sglang: even TP + head-axis KV — needs equal shards and "
                             "num_kv_heads divisible by rank count; wastes the 5090.",
                     size=10.5, color="#8a2b2b"))
    leg, _ = legend(20, H - 46, ["weights", "kv", "ctx", "free"])
    body.append(leg)
    write("02-uneven-dcp.svg",
          svg(W, H, "".join(body),
              "2 — Three mismatched cards → uneven-TP auto + uneven-DCP (token-sharded KV)",
              "1×5090 + 2×3080. Decisive setting: --rank-tp-ratio auto (sets DCP=TP), token-axis KV."))


# ---------------------------------------------------------------------------
# Scenario 3 — low-KV-head model, TP > num_kv_heads (replicated KV)
# ---------------------------------------------------------------------------
def s3_tp_gt_kvheads():
    W, H = 680, 450
    body = []
    topY = 130
    cards = [("RTX 5090", 32, "rank 0"), ("RTX 3080", 20, "rank 1"), ("RTX 3080", 20, "rank 2")]
    for i, (nm, vram, cap) in enumerate(cards):
        x = LEFT + i * SLOT
        wsh = 11 if vram == 32 else 7
        kvrep = 2  # small replicated KV-head slice
        kvtok = vram - wsh - kvrep - 2
        c, _ = gpu_column(x, topY + (32 - vram) * PXGB, nm, vram,
                          [("weights", wsh, "weight shard"),
                           ("kv", kvtok, "KV token-shard"),
                           ("scratch", kvrep, "replicated KV heads"),
                           ("ctx", 2, "ctx")],
                          caption=cap)
        body.append(c)
    body.append(text(20, 58, "Model has 2 GQA KV heads; TP=3. Cannot head-shard (2 heads < 3 ranks).",
                     size=10.5, color="#333"))
    body.append(text(20, 74, "Fork: replicate the few KV heads across ranks + token-shard the KV "
                             "(LSE-merge). Query heads still sharded normally.",
                     size=10.5, color="#2b6b3a"))
    body.append(text(20, 90, "The small salmon slice is the only duplicated part (single-digit % KV "
                             "overhead per the doc); the bulk of KV is still split.",
                     size=10.5, color="#333"))
    body.append(text(20, 106, "Stock sglang: head-axis TP is structurally impossible here — "
                             "TP is capped at num_kv_heads (=2), the 3rd card cannot join.",
                     size=10.5, color="#8a2b2b"))
    leg, _ = legend(20, H - 44, ["weights", "kv", "scratch", "ctx", "free"])
    body.append(leg)
    write("03-tp-gt-kvheads.svg",
          svg(W, H, "".join(body),
              "3 — Low-KV-head model, TP=3 → TP > num_kv_heads (replicated KV + token-shard)",
              "1×5090 + 2×3080, model with 2 KV heads. Decisive setting: replicated-KV path (§1/§9)."))


# ---------------------------------------------------------------------------
# Scenario 4 — MoE larger than VRAM per card -> per-expert offload + uneven-TP
# ---------------------------------------------------------------------------
def s4_moe_offload():
    W, H = 640, 500
    body = []
    topY = 120
    cards = [("RTX 5090", 32, "rank 0"), ("RTX 3080", 20, "rank 1"), ("RTX 3080", 20, "rank 2")]
    for i, (nm, vram, cap) in enumerate(cards):
        x = LEFT + i * SLOT
        wsh = 6 if vram == 32 else 4
        res = 8 if vram == 32 else 5   # resident experts
        scr = 2
        kv = vram - wsh - res - scr - 2
        c, _ = gpu_column(x, topY + (32 - vram) * PXGB, nm, vram,
                          [("weights", wsh, "attn+shared weights"),
                           ("resident", res, "resident experts"),
                           ("scratch", scr, "prefetch buf"),
                           ("kv", kv, "KV cache"),
                           ("ctx", 2, "ctx")],
                          caption=cap)
        body.append(c)
    # host RAM bar with the spilled experts
    hb, hy = host_ram_bar(LEFT, topY + 32 * PXGB + 55, 3 * SLOT - (SLOT - BOXW),
                          [("spill", 0.72, "cold experts (pinned) — fetched per token-wave"),
                           ("free", 0.28, "free")])
    body.append(hb)
    # links from each card down to host bar
    for i in range(3):
        x = LEFT + i * SLOT + BOXW / 2
        body.append(pcie_link(x, topY + 32 * PXGB + 20, x, topY + 32 * PXGB + 53, "PCIe", "x16"))
    body.append(text(20, 58, "SGLANG_MOE_RESIDENT_EXPERT_FRACTION<1 [in progress]: a fixed set of experts "
                             "stays resident, the rest spill to pinned host RAM.",
                     size=10.5, color="#333"))
    body.append(text(20, 74, "Wave-over-tokens prefetch → byte-identical (doc #120: ≈+0.15% ppl, "
                             "15/15 batteries). Cost is throughput (decode ≈1.4×), not quality.",
                     size=10.5, color="#2b6b3a"))
    body.append(text(20, 90, "Stock sglang: only --cpu-offload-gb (generic, layer-granular, not "
                             "quant/MoE-aware, slow) or EP (needs all experts to fit aggregate VRAM).",
                     size=10.5, color="#8a2b2b"))
    leg, _ = legend(20, H - 40, ["weights", "resident", "scratch", "kv", "spill", "ctx"])
    body.append(leg)
    write("04-moe-expert-offload.svg",
          svg(W, H, "".join(body),
              "4 — MoE with more experts than fit in VRAM → per-expert host offload + uneven-TP",
              "35B-A3B on 1×5090 + 2×3080. Decisive setting: resident-fraction expert offload (§6, [in progress])."))


# ---------------------------------------------------------------------------
# Scenario 5 — model too big for aggregate VRAM -> host-spill (122B story)
# ---------------------------------------------------------------------------
def s5_122b():
    W, H = 680, 520
    body = []
    topY = 110
    x = LEFT
    c, _ = gpu_column(x, topY, "RTX 5090", 32,
                      [("weights", 6, "attn + shared (dense) weights"),
                       ("resident", 12, "resident experts (hot tier)"),
                       ("scratch", 3, "prefetch + cushion"),
                       ("kv", 9, "KV cache"),
                       ("ctx", 2, "ctx")],
                      caption="single card holds the full model as TP=1")
    body.append(c)
    hb, hy = host_ram_bar(LEFT, topY + 32 * PXGB + 60, W - 2 * LEFT,
                          [("spill", 0.80, "~61 GB cold experts (pinned host RAM)"),
                           ("free", 0.20, "free")],
                          title="Pinned host RAM (DDR) — the experts that do not fit VRAM")
    body.append(hb)
    body.append(pcie_link(x + BOXW / 2, topY + 32 * PXGB + 20, x + BOXW / 2,
                          topY + 32 * PXGB + 58, "PCIe — streamed at load AND per token-wave", "x16"))
    body.append(text(20, 58, "Head-rank load-time MoE offload [in progress]: materialize only resident+cushion "
                             "experts on the GPU, stream the cold tier straight to host RAM at load.",
                     size=10.5, color="#333"))
    body.append(text(20, 74, "Boots what used to OOM at load. Mechanism validated on 35B-A3B; the full "
                             "122B-A10B Int4 run (~61 GB experts) is [planned] (download-gated).",
                     size=10.5, color="#2b6b3a"))
    body.append(text(20, 90, "Stock sglang: cannot run — a 122B model does not fit 32 GB, EP needs the "
                             "experts to fit aggregate VRAM, generic offload OOMs during load.",
                     size=10.5, color="#8a2b2b"))
    leg, _ = legend(20, H - 40, ["weights", "resident", "scratch", "kv", "spill", "ctx"])
    body.append(leg)
    write("05-122b-host-spill.svg",
          svg(W, H, "".join(body),
              "5 — Model too big for any single card → load-time expert offload to host RAM",
              "122B-A10B Int4 on one RTX 5090 32 GB. Decisive setting: load-time MoE offload (§6, mechanism [in progress] / 122B run [planned])."))


# ---------------------------------------------------------------------------
# Scenario 6 — long-context priority -> weightless-KV fast lane
# ---------------------------------------------------------------------------
def s6_weightless():
    W, H = 640, 470
    body = []
    topY = 120
    # fast card: full model TP=1
    x0 = LEFT
    c, _ = gpu_column(x0, topY, "RTX 5090", 32,
                      [("weights", 14, "FULL model (collective-free TP=1)"),
                       ("kv", 16, "KV cache"),
                       ("ctx", 2, "ctx")],
                      caption="Q/K/V producer + attention dispatcher")
    body.append(c)
    # two weightless workers: no weights, only KV
    for i in (1, 2):
        x = LEFT + i * SLOT
        c, _ = gpu_column(x, topY + (32 - 20) * PXGB, "RTX 3080", 20,
                          [("weights", 0.4, ""),
                           ("kv", 17.6, "KV token-shard ONLY"),
                           ("ctx", 2, "ctx")],
                          caption="weightless KV worker  •  ~14 GB freed")
        body.append(c)
        body.append(pcie_link(x0 + BOXW if i == 1 else LEFT + (i - 1) * SLOT + BOXW,
                              topY + 150, x, topY + 150, "attention KV collective (PCIe)", "x16"))
    body.append(text(20, 58, "Weightless-KV Fast Lane (Variant C, stages B1+B2a — landed, eager-only): "
                             "the fast card holds the full model, the slow cards hold only KV.",
                     size=10.5, color="#333"))
    body.append(text(20, 74, "Slow-card weight VRAM → ≈0 (thin blue sliver); ≈14 GB freed per worker → "
                             "doc: ≈4× context on the 27B test model. Extend Δ=0 vs full-TP=1.",
                     size=10.5, color="#2b6b3a"))
    body.append(text(20, 90, "Stock sglang: every rank must hold layer weights — the slow cards spend "
                             "VRAM on weight replicas/shards instead of pure KV headroom.",
                     size=10.5, color="#8a2b2b"))
    leg, _ = legend(20, H - 40, ["weights", "kv", "ctx", "free"])
    body.append(leg)
    write("06-weightless-kv-lane.svg",
          svg(W, H, "".join(body),
              "6 — Long-context priority → weightless-KV fast lane (dedicate slow cards to KV)",
              "1×5090 (full model) + 2×3080 (KV-only). Decisive setting: Variant C weightless lane (§10, landed)."))


# ---------------------------------------------------------------------------
# Scenario 7 — more ranks than cards -> multi-rank co-location (TP=5 on 3)
# ---------------------------------------------------------------------------
def s7_colocation():
    W, H = 640, 470
    body = []
    topY = 130
    # 5090 hosts 2 ranks (co-located)
    x0 = LEFT
    c, _ = gpu_column(x0, topY, "RTX 5090", 32,
                      [("weights", 6, "rank 0 shard"),
                       ("kv", 8, "rank 0 KV"),
                       ("weights", 6, "rank 1 shard"),
                       ("kv", 8, "rank 1 KV"),
                       ("ctx", 3, "2× ctx")],
                      caption="2 co-located ranks (2 processes, same CUDA_VISIBLE_DEVICES)")
    body.append(c)
    for i, r in ((1, 2), (2, 3)):
        x = LEFT + i * SLOT
        c, _ = gpu_column(x, topY + (32 - 20) * PXGB, "RTX 3080", 20,
                          [("weights", 6, f"rank {r} shard"),
                           ("kv", 12, f"rank {r} KV"),
                           ("ctx", 2, "ctx")],
                          caption=f"rank {r}")
        body.append(c)
    body.append(text(20, 58, "--rank-gpu-id 0,0,1,2 (duplicate → co-location). TP=5 emulated on 3 cards; "
                             "NCCL multi-rank auto-set; physical-impossibility check enforced.",
                     size=10.5, color="#333"))
    body.append(text(20, 74, "Used to prove replicated-KV at TP=5 (#62) without owning 5 GPUs. "
                             "Honest caveat: co-located ranks share silicon — capability, not bandwidth.",
                     size=10.5, color="#2b6b3a"))
    body.append(text(20, 90, "Stock sglang: TP is bounded by the physical GPU count — you cannot place "
                             "two ranks on one card, so TP=5 on 3 cards is impossible.",
                     size=10.5, color="#8a2b2b"))
    body.append(text(20, 106, "The 5090 box shows TWO stacked (weights+KV) blocks — two independent ranks "
                             "sharing one physical card.", size=10, color="#333"))
    leg, _ = legend(20, H - 40, ["weights", "kv", "ctx", "free"])
    body.append(leg)
    write("07-multi-rank-colocation.svg",
          svg(W, H, "".join(body),
              "7 — More ranks than cards → multi-rank co-location (TP=5 on 3 physical GPUs)",
              "1×5090 (2 ranks) + 2×3080 (1 rank each). Decisive setting: --rank-gpu-id duplicates (§9)."))


# ---------------------------------------------------------------------------
# Scenario 8 — slow PCIe x4 link -> PD-disaggregation placement
# ---------------------------------------------------------------------------
def s8_pd_disagg():
    W, H = 640, 470
    body = []
    topY = 125
    # prefill card: solo, TP=1
    x0 = LEFT
    c, _ = gpu_column(x0, topY, "RTX 5090 (PCIe x16)", 32,
                      [("weights", 14, "full weights — PREFILL role"),
                       ("kv", 14, "prefill KV (handed off)"),
                       ("ctx", 2, "ctx")],
                      caption="solo prefill TP=1  •  zero cross-GPU traffic")
    body.append(c)
    # decode cards behind x4
    for i, r in ((1, 0), (2, 1)):
        x = LEFT + i * SLOT
        c, _ = gpu_column(x, topY + (32 - 20) * PXGB, "RTX 3080 (PCIe x4)", 20,
                          [("weights", 7, "decode weight shard"),
                           ("hostkv", 3, "handed-off KV in"),
                           ("kv", 8, "decode KV"),
                           ("ctx", 2, "ctx")],
                          caption=f"decode rank {r}  (TP=2 uneven+DCP)")
        body.append(c)
    body.append(pcie_link(x0 + BOXW, topY + 40, LEFT + SLOT, topY + 40,
                          "KV handoff (mooncake_tcp loopback)", "x4"))
    body.append(pcie_link(LEFT + SLOT + BOXW, topY + 165, LEFT + 2 * SLOT, topY + 165,
                          "decode collective", "x4"))
    body.append(text(20, 58, "One decode card sits behind a slow PCIe x4 link (dashed brown). "
                             "Put PREFILL on the fast x16 card so it runs alone, zero cross-GPU comm.",
                     size=10.5, color="#333"))
    body.append(text(20, 74, "Doc: ≈2-5× faster TTFT; decode stays distributed (negligible ≈-2% long ctx). "
                             "Crash-robust handoff, tears down to 0 MiB.",
                     size=10.5, color="#2b6b3a"))
    body.append(text(20, 90, "Stock sglang: a single TP group forces every prefill collective over the "
                             "x4 link too — the slow lane throttles time-to-first-token.",
                     size=10.5, color="#8a2b2b"))
    leg, _ = legend(20, H - 40, ["weights", "kv", "hostkv", "ctx", "free"])
    body.append(leg)
    write("08-pd-disagg-slow-pcie.svg",
          svg(W, H, "".join(body),
              "8 — A slow PCIe x4 link in the rig → PD-disaggregation placement",
              "5090 on x16 (prefill), 3080s on x4 (decode). Decisive setting: single-node PD-disagg (§2)."))


# ---------------------------------------------------------------------------
# Scenario 9 — 8-GPU mixed fleet combining several capabilities
# ---------------------------------------------------------------------------
def s9_fleet():
    W, H = 1180, 520
    body = []
    topY = 150
    fleet = [
        ("RTX 5090", 32, "x16", [("weights", 8, "weights"), ("resident", 8, "resident exp"),
                                  ("kv", 14, "KV"), ("ctx", 2, "ctx")], "prefill + hot experts"),
        ("RTX 4090", 24, "x16", [("weights", 7, "weights"), ("resident", 5, "resident exp"),
                                  ("kv", 10, "KV"), ("ctx", 2, "ctx")], "decode rank"),
        ("RTX 4090", 24, "x16", [("weights", 7, "weights"), ("resident", 5, "resident exp"),
                                  ("kv", 10, "KV"), ("ctx", 2, "ctx")], "decode rank"),
        ("RTX 3090", 24, "x8", [("weights", 7, "weights"), ("kv", 15, "KV"), ("ctx", 2, "ctx")], "decode rank"),
        ("RTX 3080", 20, "x8", [("weights", 6, "weights"), ("kv", 12, "KV"), ("ctx", 2, "ctx")], "decode rank"),
        ("RTX 3080", 20, "x4", [("weights", 0.4, ""), ("kv", 17.6, "KV token-shard ONLY"),
                                 ("ctx", 2, "ctx")], "weightless KV worker"),
        ("RTX 3080", 20, "x4", [("weights", 0.4, ""), ("kv", 17.6, "KV token-shard ONLY"),
                                 ("ctx", 2, "ctx")], "weightless KV worker"),
        ("RTX 2080Ti", 11, "x4", [("weights", 3, "weights"), ("kv", 6, "KV"), ("ctx", 2, "ctx")], "decode rank"),
    ]
    for i, (nm, vram, link, regs, cap) in enumerate(fleet):
        x = LEFT + i * (BOXW - 10 + 6)  # tighter packing for 8 cards
        pass
    # tighter slot for 8 cards
    SLOT8 = (W - 2 * LEFT) / 8
    boxw8 = SLOT8 - 20
    def col8(x, top, name, vram, regions, caption, link):
        out = []
        h = vram * PXGB
        out.append(text(x + boxw8 / 2, top - 18, name, size=10.5, anchor="middle", weight="bold"))
        out.append(text(x + boxw8 / 2, top - 6, f"{vram} GB • {link}", size=8.5, anchor="middle", color="#555"))
        out.append(rect(x, top, boxw8, h, "none", stroke="#222", sw=1.4))
        used = sum(g for _, g, _ in regions)
        filled = regions + ([("free", vram - used, "free")] if used < vram - 0.05 else [])
        cy = top
        for key, gb, label in filled:
            if gb <= 0:
                continue
            rh = gb * PXGB
            out.append(rect(x, cy, boxw8, rh, COLORS[key]))
            if rh >= 13 and label:
                tc = "#fff" if key in ("weights", "kv", "resident", "spill", "hostkv") else "#1a1a1a"
                out.append(text(x + boxw8 / 2, cy + rh / 2 + 3, label, size=8, anchor="middle", color=tc))
            cy += rh
        out.append(text(x + boxw8 / 2, top + h + 13, caption, size=8, anchor="middle", color="#333"))
        return "".join(out)
    for i, (nm, vram, link, regs, cap) in enumerate(fleet):
        x = LEFT + i * SLOT8 + 10
        body.append(col8(x, topY + (32 - vram) * PXGB, nm, vram, regs, cap, link))
    # host RAM spill bar
    hb, _ = host_ram_bar(LEFT, topY + 32 * PXGB + 45, W - 2 * LEFT,
                         [("spill", 0.6, "cold MoE experts (pinned host RAM)"), ("free", 0.4, "free")])
    body.append(hb)
    body.append(text(20, 60, "One TP group across 8 mixed cards combines: uneven-TP (proportional shards), "
                             "uneven-DCP (token-KV), per-expert host offload,", size=10.5, color="#333"))
    body.append(text(20, 76, "two weightless-KV workers on the x4 cards (pure KV headroom), and PD-style "
                             "placement (prefill on the fast x16 5090).", size=10.5, color="#333"))
    body.append(text(20, 92, "Stock sglang: even TP forces the whole group down to the 11 GB 2080Ti's shard "
                             "(or excludes it); no token-KV, no weightless workers, no quant-aware offload.",
                     size=10.5, color="#8a2b2b"))
    leg, _ = legend(20, H - 40, ["weights", "kv", "resident", "hostkv", "spill", "ctx", "free"])
    body.append(leg)
    write("09-eight-gpu-fleet.svg",
          svg(W, H, "".join(body),
              "9 — Eight-GPU mixed fleet → several capabilities combined in one TP group",
              "5090+2×4090+3090+3×3080+2080Ti, mixed PCIe x16/x8/x4. Composite of §1/§6/§9/§10."))


if __name__ == "__main__":
    s1_uneven_tp()
    s2_uneven_dcp()
    s3_tp_gt_kvheads()
    s4_moe_offload()
    s5_122b()
    s6_weightless()
    s7_colocation()
    s8_pd_disagg()
    s9_fleet()
    print("done")
