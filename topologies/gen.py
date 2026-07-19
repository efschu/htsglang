#!/usr/bin/env python3
"""
Generator for the topology diagrams referenced by ../TOPOLOGIES.md.

Every diagram is a self-contained SVG: inline shapes + text only, no external
fonts, no external images, no <image href> to remote hosts. GPU boxes are scaled
by VRAM (box height = VRAM_GB * PXGB) and partitioned into labelled regions with
a fixed, consistent colour meaning across every diagram (see COLORS / LEGEND).

Text layout is auto-flowed: the title, subtitle and header sentences are wrapped
to the canvas width and stacked, and the GPU cards are placed below that header
block with a clear gap, so nothing clips the right edge or collides with a card
label. Inter-card connector tags are kept short and live in the empty column
between two cards; card->host-RAM links run down the card's right edge, clear of
the centred caption.

Region sizes inside a card are illustrative partitionings of that card's VRAM,
not measured allocations, unless a number is quoted from FEATURES_VS_UPSTREAM.md.
"""

import os

OUT = os.path.dirname(os.path.abspath(__file__))
PXGB = 4.4            # pixels per GB of VRAM (vertical scale for GPU boxes)
BOXW = 150           # GPU box width
SLOT = 205           # horizontal slot per GPU (box + gap)
LEFT = 40            # left margin before first card
MAXVRAM = 32         # tallest card in the fleet (sets bottom-align baseline)

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
WHITE_TEXT = ("weights", "kv", "resident", "spill", "hostkv")

FONT = 'font-family="DejaVu Sans, Verdana, Arial, sans-serif"'


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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


def wrap(s, maxw, size):
    """Greedy word-wrap using a conservative per-char width estimate."""
    charw = size * 0.60  # over-estimate so lines break early (no clipping)
    words = s.split()
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if not cur or len(t) * charw <= maxw:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def flow(x, y, s, maxw, size=10.5, color="#1a1a1a", weight="normal", lh=None):
    """Draw wrapped text starting at baseline y; return (svg, next_baseline_y)."""
    lh = lh or (size + 4.5)
    out = []
    cy = y
    for ln in wrap(s, maxw, size):
        out.append(text(x, cy, ln, size=size, color=color, weight=weight))
        cy += lh
    return "".join(out), cy


def gpu_column(x, top, name, vram_gb, regions, caption=None, sub=None):
    """One GPU box scaled to vram_gb, regions stacked top-down.

    regions: list of (colorkey, gb, label); remainder drawn as 'free'.
    Returns (svg, box_bottom_y).
    """
    out = []
    h = vram_gb * PXGB
    out.append(text(x + BOXW / 2, top - 21, name, size=12, anchor="middle", weight="bold"))
    lbl2 = f"{vram_gb} GB" + (f" • {sub}" if sub else " VRAM")
    out.append(text(x + BOXW / 2, top - 8, lbl2, size=10, anchor="middle", color="#444"))
    out.append(rect(x, top, BOXW, h, "none", stroke="#222", sw=1.6))
    used = sum(g for _, g, _ in regions)
    filled = list(regions)
    if used < vram_gb - 0.05:
        filled.append(("free", vram_gb - used, "free"))
    cy = top
    for key, gb, label in filled:
        if gb <= 0:
            continue
        rh = gb * PXGB
        out.append(rect(x, cy, BOXW, rh, COLORS[key]))
        tc = "#ffffff" if key in WHITE_TEXT else "#1a1a1a"
        if label and rh >= 15:
            out.append(text(x + BOXW / 2, cy + rh / 2 + 3.6, label, size=10, anchor="middle", color=tc))
        elif label and rh >= 11:
            out.append(text(x + BOXW / 2, cy + rh / 2 + 3.2, label, size=8.5, anchor="middle", color=tc))
        elif label:
            out.append(line(x + BOXW, cy + rh / 2, x + BOXW + 8, cy + rh / 2, stroke="#888", sw=0.8))
            out.append(text(x + BOXW + 11, cy + rh / 2 + 3.2, label, size=8.5, anchor="start", color="#333"))
        cy += rh
    if caption:
        out.append(text(x + BOXW / 2, top + h + 15, caption, size=9, anchor="middle", color="#333"))
    return "".join(out), top + h


def host_ram_bar(x, y, w, regions, title):
    out = []
    barh = 34
    tl, _ = flow(x, y - 7, title, w, size=10.5, color="#333", weight="bold")
    out.append(tl)
    out.append(rect(x, y, w, barh, "none", stroke="#222", sw=1.4))
    cx = x
    for key, frac, label in regions:
        rw = w * frac
        out.append(rect(cx, y, rw, barh, COLORS[key]))
        tc = "#ffffff" if key in WHITE_TEXT else "#1a1a1a"
        if rw >= 45:
            out.append(text(cx + rw / 2, y + barh / 2 + 3.6, label, size=9.5, anchor="middle", color=tc))
        cx += rw
    return "".join(out), y + barh


def gap_tag(x_left_box_right, x_right_box_left, y, label, kind="x16"):
    """Short connector drawn in the empty column between two cards."""
    styles = {"x16": (2.2, None, "#5a5a5a"), "x4": (1.5, "4 3", "#8a5a2b"),
              "nvlink": (4.0, None, "#227a3a")}
    sw, dash, col = styles[kind]
    mx = (x_left_box_right + x_right_box_left) / 2
    out = [line(x_left_box_right, y, x_right_box_left, y, stroke=col, sw=sw, dash=dash)]
    out.append(text(mx, y - 5, label, size=8.5, anchor="middle", color=col))
    return "".join(out)


def down_link(x, y1, y2, kind="x16", label=None):
    """Vertical card->host link, drawn in an empty column; optional top label."""
    styles = {"x16": (2.2, None, "#5a5a5a"), "x4": (1.5, "4 3", "#8a5a2b")}
    sw, dash, col = styles[kind]
    out = [line(x, y1, x, y2, stroke=col, sw=sw, dash=dash)]
    if label:
        out.append(text(x, y1 + 11, label, size=8.5, anchor="middle", color=col))
    return "".join(out)


def legend(x, y, keys, width):
    out = [text(x, y, "Legend (colour meaning is identical in every diagram)",
                size=9.5, anchor="start", weight="bold", color="#333")]
    per_row = 4 if width >= 620 else 3
    col_w = min(190, (width - 2 * x) / per_row)
    cy = y + 15
    for i, k in enumerate(keys):
        label = dict(LEGEND)[k]
        col, row = i % per_row, i // per_row
        px, py = x + col * col_w, cy + row * 16
        out.append(rect(px, py - 9, 12, 12, COLORS[k], stroke="#333", sw=0.8, rx=1))
        out.append(text(px + 17, py + 1, label, size=9, anchor="start", color="#333"))
    rows = (len(keys) + per_row - 1) // per_row
    return "".join(out), cy + rows * 16


def compose(name, W, title, subtitle, sentences, draw, legend_keys):
    """Stack title/subtitle/header (auto-wrapped), then cards, then legend."""
    body = []
    maxw = W - 40
    hx = 20
    s, y = flow(hx, 26, title, maxw, size=14, color="#111", weight="bold", lh=19)
    body.append(s)
    s, y = flow(hx, y + 3, subtitle, maxw, size=10.5, color="#555", lh=15)
    body.append(s)
    for txt, col in sentences:
        s, y = flow(hx, y + 4, txt, maxw, size=10.5, color=col, lh=15)
        body.append(s)
    topY = y + 30  # clear gap before the first card's name label (at topY-21)
    dbody, content_bottom = draw(topY, W)
    body.append(dbody)
    leg, legend_end = legend(20, content_bottom + 30, legend_keys, W)
    body.append(leg)
    H = int(legend_end + 18)
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'width="{W}" height="{H}" font-family="sans-serif">')
    bg = rect(0, 0, W, H, "#ffffff", stroke="none", sw=0, rx=0)
    svg = f'{head}{bg}{"".join(body)}</svg>'
    with open(os.path.join(OUT, name), "w") as f:
        f.write(svg)
    print("wrote", name, f"({W}x{H})")


def cardbottom(topY):
    return topY + MAXVRAM * PXGB


# ---------------------------------------------------------------------------
def s1():
    def draw(topY, W):
        b = []
        cb = cardbottom(topY)
        x0 = LEFT
        c, _ = gpu_column(x0, topY, "RTX 5090", 32,
                          [("weights", 13, "weight shard (rank0)"),
                           ("kv", 13, "KV cache"), ("ctx", 2, "ctx")],
                          caption="rank 0  •  ratio 8")
        b.append(c)
        x1 = LEFT + SLOT
        c, _ = gpu_column(x1, topY + (32 - 20) * PXGB, "RTX 3080", 20,
                          [("weights", 8, "weight shard (rank1)"),
                           ("kv", 8, "KV cache"), ("ctx", 2, "ctx")],
                          caption="rank 1  •  ratio 5")
        b.append(c)
        b.append(gap_tag(x0 + BOXW, x1, topY + 95, "PCIe x16", "x16"))
        # stock even-TP ghost
        lx = LEFT + 2 * SLOT + 20
        b.append(text(lx + 60, topY - 21, "stock even-TP", size=11.5, anchor="middle", weight="bold", color="#8a2b2b"))
        b.append(rect(lx, topY, 120, 32 * PXGB, "none", stroke="#8a2b2b", sw=1.4, dash="5 3"))
        b.append(rect(lx, topY, 120, 8 * PXGB, COLORS["weights"]))
        b.append(text(lx + 60, topY + 8 * PXGB / 2 + 3.6, "weights (capped)", size=9, anchor="middle", color="#fff"))
        b.append(rect(lx, topY + 8 * PXGB, 120, 8 * PXGB, COLORS["kv"]))
        b.append(text(lx + 60, topY + 12 * PXGB + 3.6, "KV (capped)", size=9, anchor="middle", color="#fff"))
        b.append(rect(lx, topY + 16 * PXGB, 120, 16 * PXGB, "none", stroke="#8a2b2b", sw=1.0, dash="4 3"))
        b.append(text(lx + 60, topY + 23 * PXGB, "WASTED", size=11, anchor="middle", weight="bold", color="#8a2b2b"))
        b.append(text(lx + 60, topY + 23 * PXGB + 14, "~12 GB idle", size=9, anchor="middle", color="#8a2b2b"))
        return "".join(b), cb + 22
    compose("01-uneven-tp.svg", 700,
            "1 — Two mismatched GPUs, PCIe, no NVLink → uneven Tensor Parallelism",
            "RTX 5090 32 GB + RTX 3080 20 GB. Decisive setting: --rank-tp-ratio (proportional shards).",
            [("--rank-tp-ratio 8,5: shards sized to each card, not to the smallest. Cards linked by "
              "NCCL over PCIe host-staging (no NVLink).", "#333"),
             ("Stock sglang: even TP only — both ranks capped to the 3080's shard; ~12 GB of the 5090 "
              "goes unused (dashed box at right).", "#8a2b2b")],
            draw, ["weights", "kv", "ctx", "free"])


def s2():
    def draw(topY, W):
        b = []
        cb = cardbottom(topY)
        cards = [("RTX 5090", 32, "rank 0"), ("RTX 3080", 20, "rank 1"), ("RTX 3080", 20, "rank 2")]
        for i, (nm, vram, cap) in enumerate(cards):
            x = LEFT + i * SLOT
            wsh = 11 if vram == 32 else 7
            kv = vram - wsh - 2
            c, _ = gpu_column(x, topY + (32 - vram) * PXGB, nm, vram,
                              [("weights", wsh, "weight shard"),
                               ("kv", kv, "KV token-shard"), ("ctx", 2, "ctx")],
                              caption=cap)
            b.append(c)
            if i > 0:
                xp = LEFT + (i - 1) * SLOT
                b.append(gap_tag(xp + BOXW, x, topY + 100, "DCP", "x16"))
        return "".join(b), cb + 22
    compose("02-uneven-dcp.svg", 700,
            "2 — Three mismatched cards → uneven-TP auto + uneven-DCP",
            "1×5090 + 2×3080. Decisive setting: --rank-tp-ratio auto (sets DCP=TP), token-axis KV.",
            [("--rank-tp-ratio auto + uneven-DCP token sharding: KV is split along the TOKEN axis, not "
              "the KV-head axis. Inter-card links are DCP LSE-merges over PCIe.", "#333"),
             ("Fills every card; the big card is not throttled to the small ones. Doc: ≈+2.5-3x KV "
              "context vs a naive equal split.", "#2b6b3a"),
             ("Stock sglang: even TP + head-axis KV — needs equal shards and num_kv_heads divisible by "
              "rank count; wastes the 5090.", "#8a2b2b")],
            draw, ["weights", "kv", "ctx", "free"])


def s3():
    def draw(topY, W):
        b = []
        cb = cardbottom(topY)
        cards = [("RTX 5090", 32, "rank 0"), ("RTX 3080", 20, "rank 1"), ("RTX 3080", 20, "rank 2")]
        for i, (nm, vram, cap) in enumerate(cards):
            x = LEFT + i * SLOT
            wsh = 11 if vram == 32 else 7
            kvrep = 3
            kvtok = vram - wsh - kvrep - 2
            c, _ = gpu_column(x, topY + (32 - vram) * PXGB, nm, vram,
                              [("weights", wsh, "weight shard"),
                               ("kv", kvtok, "KV token-shard"),
                               ("scratch", kvrep, "replicated KV"),
                               ("ctx", 2, "ctx")], caption=cap)
            b.append(c)
            if i > 0:
                xp = LEFT + (i - 1) * SLOT
                b.append(gap_tag(xp + BOXW, x, topY + 100, "DCP", "x16"))
        return "".join(b), cb + 22
    compose("03-tp-gt-kvheads.svg", 700,
            "3 — Fewer KV heads than ranks → TP > num_kv_heads (replicated KV)",
            "1×5090 + 2×3080, model with 2 KV heads, TP=3. Decisive setting: replicated-KV path (§1/§9).",
            [("The model has 2 GQA KV heads; TP=3 cannot head-shard (2 heads < 3 ranks). Fork: replicate "
              "the few KV heads across ranks + token-shard the KV (LSE-merge). Query heads still sharded.", "#333"),
             ("The salmon slice is the only duplicated part (single-digit-% KV overhead per the doc); the "
              "bulk of KV is still split.", "#2b6b3a"),
             ("Stock sglang: head-axis TP is structurally impossible here — TP is capped at num_kv_heads "
              "(=2), the 3rd card cannot join.", "#8a2b2b")],
            draw, ["weights", "kv", "scratch", "ctx", "free"])


def s4():
    def draw(topY, W):
        b = []
        cb = cardbottom(topY)
        cards = [("RTX 5090", 32, "rank 0"), ("RTX 3080", 20, "rank 1"), ("RTX 3080", 20, "rank 2")]
        for i, (nm, vram, cap) in enumerate(cards):
            x = LEFT + i * SLOT
            wsh = 6 if vram == 32 else 4
            res = 8 if vram == 32 else 5
            scr = 3
            kv = vram - wsh - res - scr - 2
            c, _ = gpu_column(x, topY + (32 - vram) * PXGB, nm, vram,
                              [("weights", wsh, "attn+shared wts"),
                               ("resident", res, "resident experts"),
                               ("scratch", scr, "prefetch"),
                               ("kv", kv, "KV cache"), ("ctx", 2, "ctx")], caption=cap)
            b.append(c)
        # PCIe links drawn in the empty gap columns between cards
        for gx in (LEFT + BOXW + (SLOT - BOXW) / 2, LEFT + SLOT + BOXW + (SLOT - BOXW) / 2):
            b.append(down_link(gx, cb, cb + 44, "x16", "PCIe"))
        hb, hy = host_ram_bar(LEFT, cb + 44, 3 * SLOT - (SLOT - BOXW),
                              [("spill", 0.72, "cold experts (pinned) — fetched per token-wave"),
                               ("free", 0.28, "free")],
                              "Pinned host RAM (DDR)")
        b.append(hb)
        return "".join(b), hy
    compose("04-moe-expert-offload.svg", 660,
            "4 — MoE with more experts than fit VRAM → per-expert host offload + uneven-TP",
            "35B-A3B on 1×5090 + 2×3080. Decisive setting: resident-fraction expert offload (§6, [in progress]).",
            [("SGLANG_MOE_RESIDENT_EXPERT_FRACTION<1 [in progress]: a fixed set of experts stays resident "
              "(amber), the rest spill to pinned host RAM (red) and are prefetched per forward.", "#333"),
             ("Wave-over-tokens prefetch → byte-identical (doc #120: ≈+0.15% ppl, 15/15 batteries). Cost is "
              "throughput (decode ≈1.4×), not quality.", "#2b6b3a"),
             ("Stock sglang: only --cpu-offload-gb (generic, layer-granular, not quant/MoE-aware, slow) or "
              "EP (needs all experts to fit aggregate VRAM).", "#8a2b2b")],
            draw, ["weights", "resident", "scratch", "kv", "spill", "ctx"])


def s5():
    def draw(topY, W):
        b = []
        cb = cardbottom(topY)
        x = LEFT
        c, _ = gpu_column(x, topY, "RTX 5090", 32,
                          [("weights", 6, "attn + shared (dense) wts"),
                           ("resident", 12, "resident experts (hot tier)"),
                           ("scratch", 3, "prefetch + cushion"),
                           ("kv", 9, "KV cache"), ("ctx", 2, "ctx")],
                          caption="full model, TP=1")
        b.append(c)
        b.append(down_link(x + BOXW + 30, cb, cb + 46, "x16", "PCIe"))
        hb, hy = host_ram_bar(LEFT, cb + 46, W - 2 * LEFT,
                              [("spill", 0.80, "~61 GB cold experts (pinned host RAM)"),
                               ("free", 0.20, "free")],
                              "Pinned host RAM (DDR)")
        b.append(hb)
        return "".join(b), hy
    compose("05-122b-host-spill.svg", 660,
            "5 — Model too big for any single card → load-time expert offload to host RAM",
            "122B-A10B Int4 on one RTX 5090 32 GB. Decisive setting: load-time MoE offload (§6, mechanism [in progress] / 122B run [planned]).",
            [("Head-rank load-time MoE offload [in progress]: materialize only resident+cushion experts on "
              "the GPU, stream the cold tier straight to host RAM at load — instead of materializing all "
              "experts then slicing (which OOMs at load).", "#333"),
             ("Boots what used to OOM at load. Mechanism validated on 35B-A3B; the full 122B-A10B Int4 run "
              "(~61 GB experts) is [planned] (download-gated).", "#2b6b3a"),
             ("Stock sglang: cannot run — a 122B model does not fit 32 GB, EP needs experts to fit aggregate "
              "VRAM, generic offload OOMs during load.", "#8a2b2b")],
            draw, ["weights", "resident", "scratch", "kv", "spill", "ctx"])


def s6():
    def draw(topY, W):
        b = []
        cb = cardbottom(topY)
        x0 = LEFT
        c, _ = gpu_column(x0, topY, "RTX 5090", 32,
                          [("weights", 14, "FULL model (TP=1)"),
                           ("kv", 16, "KV cache"), ("ctx", 2, "ctx")],
                          caption="Q/K/V producer + attn dispatcher")
        b.append(c)
        for i in (1, 2):
            x = LEFT + i * SLOT
            c, _ = gpu_column(x, topY + (32 - 20) * PXGB, "RTX 3080", 20,
                              [("weights", 0.4, ""),
                               ("kv", 17.6, "KV token-shard ONLY"), ("ctx", 2, "ctx")],
                              caption="weightless KV worker • ~14 GB freed")
            b.append(c)
            xp = LEFT + (i - 1) * SLOT
            b.append(gap_tag(xp + BOXW, x, topY + 100, "KV", "x16"))
        return "".join(b), cb + 22
    compose("06-weightless-kv-lane.svg", 700,
            "6 — Long-context priority → weightless-KV fast lane",
            "1×5090 (full model) + 2×3080 (KV-only). Decisive setting: Variant C weightless lane (§10, landed).",
            [("Weightless-KV Fast Lane (Variant C, stages B1+B2a — landed, eager-only): the fast card holds "
              "the full model as collective-free TP=1; the slow cards hold ONLY a KV token-shard and run a "
              "stripped attention-only forward — no layer weights (thin blue sliver ≈0).", "#333"),
             ("≈14 GB freed per worker → doc: ≈4× context on the 27B test model. Extend Δ=0 vs full-TP=1; "
              "decode differs only by benign kernel fp-order.", "#2b6b3a"),
             ("Stock sglang: every rank must hold layer weights — the slow cards spend VRAM on weight "
              "shards/replicas instead of pure KV headroom.", "#8a2b2b")],
            draw, ["weights", "kv", "ctx", "free"])


def s7():
    def draw(topY, W):
        b = []
        cb = cardbottom(topY)
        x0 = LEFT
        c, _ = gpu_column(x0, topY, "RTX 5090", 32,
                          [("weights", 6, "rank 0 shard"), ("kv", 8, "rank 0 KV"),
                           ("weights", 6, "rank 1 shard"), ("kv", 8, "rank 1 KV"),
                           ("ctx", 3, "2× ctx")],
                          caption="2 co-located ranks (2 processes)")
        b.append(c)
        for i, r in ((1, 2), (2, 3)):
            x = LEFT + i * SLOT
            c, _ = gpu_column(x, topY + (32 - 20) * PXGB, "RTX 3080", 20,
                              [("weights", 6, f"rank {r} shard"),
                               ("kv", 12, f"rank {r} KV"), ("ctx", 2, "ctx")],
                              caption=f"rank {r}")
            b.append(c)
            xp = LEFT + (i - 1) * SLOT
            b.append(gap_tag(xp + BOXW, x, topY + 100, "NCCL", "x16"))
        return "".join(b), cb + 22
    compose("07-multi-rank-colocation.svg", 700,
            "7 — More ranks than cards → multi-rank co-location (TP=5 on 3 GPUs)",
            "1×5090 (2 ranks) + 2×3080 (1 rank each). Decisive setting: --rank-gpu-id duplicates (§9).",
            [("--rank-gpu-id 0,0,1,2 (duplicate → co-location): two ranks run on the 5090 as two processes; "
              "NCCL multi-rank auto-set; the physical-impossibility check enforces (ranks × MiB) ≤ NVML "
              "total. The 5090 shows TWO stacked (weights+KV) blocks — two real ranks on one card.", "#333"),
             ("Used to prove replicated-KV at TP=5 (#62) without owning 5 GPUs. Honest caveat: co-located "
              "ranks share silicon — capability, not extra bandwidth.", "#2b6b3a"),
             ("Stock sglang: TP is bounded by the physical GPU count — you cannot place two ranks on one "
              "card, so TP=5 on 3 cards is impossible.", "#8a2b2b")],
            draw, ["weights", "kv", "ctx", "free"])


def s8():
    def draw(topY, W):
        b = []
        cb = cardbottom(topY)
        x0 = LEFT
        c, _ = gpu_column(x0, topY, "RTX 5090", 32,
                          [("weights", 14, "full weights — PREFILL"),
                           ("kv", 14, "prefill KV (handed off)"), ("ctx", 2, "ctx")],
                          caption="solo prefill TP=1 • zero cross-GPU traffic", sub="PCIe x16")
        b.append(c)
        for i, r in ((1, 0), (2, 1)):
            x = LEFT + i * SLOT
            c, _ = gpu_column(x, topY + (32 - 20) * PXGB, "RTX 3080", 20,
                              [("weights", 7, "decode wt shard"),
                               ("hostkv", 4, "handed KV in"),
                               ("kv", 7, "decode KV"), ("ctx", 2, "ctx")],
                              caption=f"decode rank {r} (TP=2 uneven+DCP)", sub="PCIe x4")
            b.append(c)
        b.append(gap_tag(x0 + BOXW, LEFT + SLOT, topY + 90, "x4 KV in", "x4"))
        b.append(gap_tag(LEFT + SLOT + BOXW, LEFT + 2 * SLOT, topY + 120, "x4", "x4"))
        return "".join(b), cb + 22
    compose("08-pd-disagg-slow-pcie.svg", 700,
            "8 — A slow PCIe x4 link in the rig → PD-disaggregation placement",
            "5090 on x16 (prefill), 3080s on x4 (decode). Decisive setting: single-node PD-disagg (§2).",
            [("One decode card sits behind a slow PCIe x4 link (dashed brown). Put PREFILL on the fast x16 "
              "card so it runs alone, zero cross-GPU comm; the KV handoff uses mooncake_tcp loopback (teal "
              "region on the decode cards).", "#333"),
             ("Doc: ≈2-5× faster TTFT; decode stays distributed (negligible ≈-2% long ctx). Crash-robust "
              "handoff, tears down to 0 MiB.", "#2b6b3a"),
             ("Stock sglang: a single TP group forces every prefill collective over the x4 link too — the "
              "slow lane throttles time-to-first-token.", "#8a2b2b")],
            draw, ["weights", "kv", "hostkv", "ctx", "free"])


def s9():
    def draw(topY, W):
        b = []
        SLOT8 = (W - 2 * LEFT) / 8
        boxw8 = SLOT8 - 22
        cb = topY + 32 * PXGB
        fleet = [
            ("RTX 5090", 32, "x16", [("weights", 8, "wts"), ("resident", 8, "resident exp"),
                                     ("kv", 14, "KV"), ("ctx", 2, "")], "prefill + hot exp"),
            ("RTX 4090", 24, "x16", [("weights", 7, "wts"), ("resident", 5, "resident"),
                                     ("kv", 10, "KV"), ("ctx", 2, "")], "decode rank"),
            ("RTX 4090", 24, "x16", [("weights", 7, "wts"), ("resident", 5, "resident"),
                                     ("kv", 10, "KV"), ("ctx", 2, "")], "decode rank"),
            ("RTX 3090", 24, "x8", [("weights", 7, "wts"), ("kv", 15, "KV"), ("ctx", 2, "")], "decode rank"),
            ("RTX 3080", 20, "x8", [("weights", 6, "wts"), ("kv", 12, "KV"), ("ctx", 2, "")], "decode rank"),
            ("RTX 3080", 20, "x4", [("weights", 0.4, ""), ("kv", 17.6, "KV only"), ("ctx", 2, "")], "weightless KV"),
            ("RTX 3080", 20, "x4", [("weights", 0.4, ""), ("kv", 17.6, "KV only"), ("ctx", 2, "")], "weightless KV"),
            ("RTX 2080Ti", 11, "x4", [("weights", 3, "wts"), ("kv", 6, "KV"), ("ctx", 2, "")], "decode rank"),
        ]

        def col8(x, top, nm, vram, regions, cap, link):
            o = []
            h = vram * PXGB
            o.append(text(x + boxw8 / 2, top - 19, nm, size=10, anchor="middle", weight="bold"))
            o.append(text(x + boxw8 / 2, top - 7, f"{vram}GB • {link}", size=8.5, anchor="middle", color="#555"))
            o.append(rect(x, top, boxw8, h, "none", stroke="#222", sw=1.3))
            used = sum(g for _, g, _ in regions)
            filled = regions + ([("free", vram - used, "free")] if used < vram - 0.05 else [])
            cy = top
            for key, gb, lab in filled:
                if gb <= 0:
                    continue
                rh = gb * PXGB
                o.append(rect(x, cy, boxw8, rh, COLORS[key]))
                if lab and rh >= 12:
                    tc = "#fff" if key in WHITE_TEXT else "#1a1a1a"
                    o.append(text(x + boxw8 / 2, cy + rh / 2 + 3, lab, size=8, anchor="middle", color=tc))
                cy += rh
            o.append(text(x + boxw8 / 2, top + h + 12, cap, size=8, anchor="middle", color="#333"))
            return "".join(o)
        for i, (nm, vram, link, regs, cap) in enumerate(fleet):
            x = LEFT + i * SLOT8 + 11
            b.append(col8(x, topY + (32 - vram) * PXGB, nm, vram, regs, cap, link))
        hb, hy = host_ram_bar(LEFT, cb + 40, W - 2 * LEFT,
                              [("spill", 0.6, "cold MoE experts (pinned host RAM), streamed over PCIe"),
                               ("free", 0.4, "free")],
                              "Pinned host RAM (DDR)")
        b.append(hb)
        return "".join(b), hy
    compose("09-eight-gpu-fleet.svg", 1180,
            "9 — Eight-GPU mixed fleet → several capabilities combined in one TP group",
            "5090 + 2×4090 + 3090 + 3×3080 + 2080Ti, mixed PCIe x16/x8/x4, no NVLink. Composite of §1/§6/§9/§10.",
            [("One TP group across 8 mixed cards combines: uneven-TP (proportional shards), uneven-DCP "
              "(token-KV), per-expert host offload (bottom bar), two weightless-KV workers on the x4 cards "
              "(pure KV headroom), and PD-style placement (prefill on the fast x16 5090).", "#333"),
             ("Stock sglang: even TP forces the whole group down to the 11 GB 2080Ti's shard (or excludes "
              "it); no token-KV, no weightless workers, no quant-aware offload — most of the fleet's VRAM is "
              "wasted or unusable for one model.", "#8a2b2b")],
            draw, ["weights", "kv", "resident", "hostkv", "spill", "ctx", "free"])


if __name__ == "__main__":
    for fn in (s1, s2, s3, s4, s5, s6, s7, s8, s9):
        fn()
    print("done")
