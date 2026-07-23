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
    # granular (per-layer / per-head) palette, used by diagrams 10+:
    "gdn":       "#7d5ba6",  # GDN / linear-attention layer (recurrent state)
    "fullattn":  "#2f6f8f",  # full-attention layer (holds KV cache)
    "state":     "#c9b8e0",  # GDN recurrent state (fixed size)
    "mtp":       "#c65b9b",  # MTP / NEXTN draft head
    "bad":       "#e9d7d7",  # upstream: unused region / a split its constraints do not admit here
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
    ("gdn",      "GDN / linear-attn layer"),
    ("fullattn", "full-attention layer (KV)"),
    ("state",    "GDN recurrent state"),
    ("mtp",      "MTP / NEXTN draft head"),
    ("bad",      "upstream: unused / not admitted here"),
]
WHITE_TEXT = ("weights", "kv", "resident", "spill", "hostkv", "gdn", "fullattn", "mtp")

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


def defs():
    """Shared arrowhead markers (self-contained, no external refs)."""
    ms = [("arr", "#333"), ("arrP", "#8a5a2b"), ("arrG", "#227a3a"),
          ("arrR", "#c0504d"), ("arrB", "#3b6fb0"), ("arrV", "#7d5ba6")]
    s = "<defs>"
    for mid, col in ms:
        s += (f'<marker id="{mid}" markerWidth="9" markerHeight="9" refX="6.5" '
              f'refY="3" orient="auto" markerUnits="userSpaceOnUse">'
              f'<path d="M0,0 L7,3 L0,6 Z" fill="{col}"/></marker>')
    return s + "</defs>"


def arrow(x1, y1, x2, y2, color="#333", sw=1.7, dash=None, marker="arr"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#{marker})"/>')


def panel_label(x, y, label, kind):
    """Pill badge marking a panel as 'htsglang' (ours) or 'upstream sglang' (neutral)."""
    col = "#1d6b34" if kind == "ours" else "#4a5568"
    fill = "#e7f2ea" if kind == "ours" else "#eef1f4"
    w = len(label) * 7.2 + 22
    return (rect(x, y - 15, w, 21, fill, stroke=col, sw=1.3, rx=5)
            + text(x + 11, y, label, size=11.5, weight="bold", color=col))


def vdivider(x, y1, y2):
    return line(x, y1, x, y2, stroke="#bbb", sw=1.2, dash="3 4")


def chip(x, y, w, h, key, label="", size=8, tc=None, stroke="#2b2b2b", sw=0.8):
    tc = tc or ("#fff" if key in WHITE_TEXT else "#1a1a1a")
    o = rect(x, y, w, h, COLORS[key], stroke=stroke, sw=sw, rx=1.5)
    if label and h >= 11:
        o += text(x + w / 2, y + h / 2 + size * 0.35, label, size=size, anchor="middle", color=tc)
    return o


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
    svg = f'{head}{defs()}{bg}{"".join(body)}</svg>'
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
        b.append(text(lx + 60, topY - 21, "upstream even-TP", size=11.5, anchor="middle", weight="bold", color="#4a5568"))
        b.append(rect(lx, topY, 120, 32 * PXGB, "none", stroke="#4a5568", sw=1.4, dash="5 3"))
        b.append(rect(lx, topY, 120, 8 * PXGB, COLORS["weights"]))
        b.append(text(lx + 60, topY + 8 * PXGB / 2 + 3.6, "weights (3080-sized)", size=9, anchor="middle", color="#fff"))
        b.append(rect(lx, topY + 8 * PXGB, 120, 8 * PXGB, COLORS["kv"]))
        b.append(text(lx + 60, topY + 12 * PXGB + 3.6, "KV (3080-sized)", size=9, anchor="middle", color="#fff"))
        b.append(rect(lx, topY + 16 * PXGB, 120, 16 * PXGB, COLORS["bad"], stroke="#4a5568", sw=1.0, dash="4 3"))
        b.append(text(lx + 60, topY + 23 * PXGB, "~12 GB", size=11, anchor="middle", weight="bold", color="#4a5568"))
        b.append(text(lx + 60, topY + 23 * PXGB + 14, "unused on the 5090", size=9, anchor="middle", color="#4a5568"))
        return "".join(b), cb + 22
    compose("01-uneven-tp.svg", 700,
            "1 — Two mismatched GPUs, PCIe, no NVLink → uneven Tensor Parallelism",
            "RTX 5090 32 GB + RTX 3080 20 GB. Decisive setting: --rank-tp-ratio (proportional shards).",
            [("--rank-tp-ratio 8,5: shards sized to each card, not to the smallest. Cards linked by "
              "NCCL over PCIe host-staging (no NVLink).", "#333"),
             ("Upstream sglang: even TP sizes both ranks to the 3080's shard; ~12 GB of the 5090 is left "
              "unused (dashed box at right).", "#555")],
            draw, ["weights", "kv", "ctx", "free", "bad"])


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
             ("Fills every card; the big card is sized to its own VRAM. Measured ≈+2.3-2.8× token-axis "
              "KV context vs an equal head-axis split (2.27-2.81× on the 27B uneven-DCP runs).", "#2b6b3a"),
             ("Upstream sglang: even TP + head-axis KV uses equal shards and requires num_kv_heads "
              "divisible by the rank count; the 5090 is sized to the 3080's shard.", "#555")],
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
             ("Upstream sglang: head-axis TP is bounded by num_kv_heads (=2); a 3rd rank is not covered "
              "by the head split.", "#555")],
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
              "(amber), the rest spill to pinned host RAM (red) and are prefetched per token-wave.", "#333"),
             ("FP8 path is quality-neutral (#120: ≈+0.15% ppl within FP8 reduction-order noise, 15/15 "
              "batteries, needle 100%) and self-deterministic — not bit-identical, diverging only at "
              "near-ties (sub-ULP). The cost is throughput, not quality (offload is slower than no-offload; "
              "exact FP8 decode factor is config-dependent, not separately benchmarked).", "#2b6b3a"),
             ("Upstream sglang: the generic --cpu-offload-gb path (layer-granular, dequantizes on CPU) or "
              "Expert Parallelism (experts must fit aggregate VRAM).", "#555")],
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
              "the GPU and stream the cold tier straight to host RAM at load — rather than materializing all "
              "experts on the GPU and then slicing.", "#333"),
             ("Boots a model whose weights exceed a single card's VRAM at load. Mechanism validated on "
              "35B-A3B; the full 122B-A10B Int4 run (~61 GB experts) is [planned] (download-gated).", "#2b6b3a"),
             ("Upstream sglang: no load-time host-spill path — a 122B model exceeds 32 GB; Expert Parallelism "
              "requires experts to fit aggregate VRAM, and the generic offload materializes on the GPU "
              "during load.", "#555")],
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
            "6 — Long-context priority → weightless-KV lane",
            "1×5090 (full model) + 2×3080 (KV-only). Decisive setting: Variant C weightless lane (§10, landed).",
            [("Weightless-KV lane (Variant C, landed; CUDA-graph decode landed #133/#136): the fast card holds "
              "the full model as collective-free TP=1; the slow cards hold ONLY a KV token-shard and run a "
              "stripped attention-only forward — no layer weights (thin blue sliver ≈0).", "#333"),
             ("≈14 GB of weights freed per worker (measured) becomes KV headroom; the host-spill KV tier was "
              "measured to 262k tokens on the 27B test model (#134). Prefill is bit-identical to full-TP=1; "
              "decode/extend is argmax-identical (decode-class, not bit-0). Resulting context multiple depends "
              "on the KV budget, not separately benchmarked.", "#2b6b3a"),
             ("Upstream sglang: every rank holds its layer-weight shard; each card's VRAM is shared between "
              "weights and KV.", "#555")],
            draw, ["weights", "kv", "ctx", "free"])


def s7():
    def draw(topY, W):
        b = []
        cb = cardbottom(topY)
        x0 = LEFT
        # 5090 hosts THREE co-located ranks (TP=5 layout 0,0,0,1,2), ~7 GB budget each.
        c, _ = gpu_column(x0, topY, "RTX 5090", 32,
                          [("weights", 3, "rank0 shard"), ("kv", 4, "rank0 KV"),
                           ("weights", 3, "rank1 shard"), ("kv", 4, "rank1 KV"),
                           ("weights", 3, "rank2 shard"), ("kv", 4, "rank2 KV"),
                           ("ctx", 4, "3× ctx (MPS)")],
                          caption="3 co-located ranks • MPS time-slice")
        b.append(c)
        # MPS time-slicing marker: one card, three processes share the SMs in turn.
        b.append(text(x0 + BOXW / 2, cb + 34, "one card, 3 processes → MPS time-slices the SMs",
                      size=8.5, anchor="middle", color="#8a5a2b"))
        for i, r in ((1, 3), (2, 4)):
            x = LEFT + i * SLOT
            c, _ = gpu_column(x, topY + (32 - 20) * PXGB, "RTX 3080", 20,
                              [("weights", 5, f"rank{r} shard"),
                               ("kv", 13, f"rank{r} KV (~17 GB)"), ("ctx", 2, "ctx")],
                              caption=f"rank {r} • 1 rank / card")
            b.append(c)
            xp = LEFT + (i - 1) * SLOT
            b.append(gap_tag(xp + BOXW, x, topY + 100, "NCCL x5", "x16"))
        return "".join(b), cb + 44
    compose("07-multi-rank-colocation.svg", 700,
            "7 — More ranks than cards → multi-rank co-location (TP=5 on 3 GPUs, booted)",
            "1×5090 (3 ranks) + 2×3080 (1 rank each). Decisive setting: --rank-gpu-id maps ranks to physical GPUs (§9).",
            [("--rank-gpu-id 0,0,0,1,2 --rank-tp-ratio auto: three ranks run on the 5090 as three processes "
              "(~7 GB budget each), one rank on each 3080. NCCL multi-rank auto-set; the physical-impossibility "
              "check enforces Σ(co-located budgets) ≤ NVML total. The 5090 shows THREE stacked (weights+KV) "
              "blocks — three real ranks time-slicing one card via MPS.", "#333"),
             ("Booted + validated (NCCL 2.30.7 side-loaded, MPS on): all 5 ranks build the communicator, "
              "capture target-verify + draft-decode + draft-extend graphs, serve a coherent needle from ~15k "
              "context, bit-identical across two boots. A simpler TP=4 (2 ranks on the 5090) is the same "
              "mechanism with one fewer co-located rank. Decode tok/s is not 5-card-representative — the 3 "
              "ranks time-slice one card.", "#2b6b3a"),
             ("Upstream sglang: TP is bounded by the physical GPU count — one rank per physical card, so a "
              "TP=5 layout is expressed over 5 physical GPUs.", "#555")],
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
             ("Faster TTFT is expected because prefill avoids the x4-lane collectives; decode stays "
              "distributed and largely unchanged. Handoff is crash-robust and tears down to 0 MiB. (The "
              "TTFT factor is an estimate, not benchmarked on this rig.)", "#2b6b3a"),
             ("Upstream sglang: a single fused TP group runs every prefill collective over every link in "
              "the group, including the x4 lane.", "#555")],
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
             ("Upstream sglang: even TP sizes the whole group to the 11 GB 2080Ti's shard (or excludes it); "
              "the token-KV, weightless-worker and quant-aware host-offload paths are fork-specific, so the "
              "larger cards are sized to the smallest card's shard.", "#555")],
            draw, ["weights", "kv", "resident", "hostkv", "spill", "ctx", "free"])


# ===========================================================================
# GRANULAR diagrams for the measured 122B-A10B-Int4 TP=3 end-state.
# Config (offload_handoff.md): Qwen3.5-122B-A10B-GPTQ-Int4, 48 layers, hybrid-GDN
# ~3:1, 256 experts/layer top-8, 2 GQA KV heads, 32 Q heads, FFN intermediate
# 1024, GPTQ group_size 128. tp=3 dcp=3, --rank-tp-ratio auto -> [28,19,19]:
# rank0=5090(cuda:0, 28000 MiB budget), rank1/2=3080(19000). q 16/8/8, FFN
# 512/256/256 => 4/2/2 GPTQ groups, 2 KV heads REPLICATED on all ranks, KV
# token-sharded 3 ways by DCP. Offload f=0.25 -> 64 resident + 16 scratch
# (buffer 80) + 176 host-spilled per layer x48. Per-rank GPU 25.5/16.6/15.4 GB,
# host floor 24.4 GB, eager (--disable-cuda-graph), 6.97 tok/s, MTP on rank0.
# ===========================================================================
R0T, R1T, R2T = "#cfe0f0", "#d4ecd9", "#f5e3c7"   # rank ownership tints
RANKTINT = (R0T, R1T, R2T)


def swatch_note(x, y, key, s, maxw, size=9.5):
    o = rect(x, y - 10, 13, 13, COLORS[key], stroke="#333", sw=0.8, rx=2)
    t, ny = flow(x + 20, y, s, maxw - 20, size=size, color="#333")
    return o + t, ny


def g10():
    """Shared model structure: the 48-layer hybrid-GDN stack + MTP head."""
    def draw(topY, W):
        b = []
        x = 90
        cellh = 6.4
        n = 48
        wcol = 150
        top = topY + 34
        # MTP head above the stack
        b.append(chip(x, top - 26, wcol, 17, "mtp", "MTP / NEXTN draft head"))
        b.append(arrow(x + wcol, top - 17, x + wcol + 26, top - 17, "#c65b9b", marker="arr"))
        b.append(text(x + wcol + 30, top - 21, "1 draft layer, lives on rank0 (5090);", size=9.5, color="#333"))
        b.append(text(x + wcol + 30, top - 8, "own small KV reservation (see KV-layout diagram).", size=9.5, color="#333"))
        # layer stack, L47 at top -> L0 at bottom
        for idx in range(n):
            layer = n - 1 - idx
            y = top + idx * cellh
            key = "fullattn" if layer % 4 == 3 else "gdn"
            b.append(rect(x, y, wcol, cellh, COLORS[key], stroke="#ffffff", sw=0.4, rx=0))
        b.append(rect(x, top, wcol, n * cellh, "none", stroke="#222", sw=1.5))
        b.append(text(x - 8, top + 7, "L47", size=8.5, anchor="end", color="#555"))
        b.append(text(x - 8, top + n * cellh - 1, "L0", size=8.5, anchor="end", color="#555"))
        b.append(text(x + wcol / 2, top + n * cellh + 16, "48 decoder layers", size=10.5, anchor="middle", weight="bold"))
        b.append(text(x + wcol / 2, top + n * cellh + 30, "hybrid-GDN, ~3:1 (GDN : full-attn)", size=9, anchor="middle", color="#555"))
        # callouts with arrows to a representative full-attn row and gdn row
        rx = x + wcol
        cxr = 330
        fy = top + (n - 1 - 43) * cellh + cellh / 2   # layer 43 = full-attn
        gy = top + (n - 1 - 42) * cellh + cellh / 2   # layer 42 = gdn
        b.append(arrow(rx + 2, fy, cxr - 8, top + 18, "#2f6f8f"))
        b.append(arrow(rx + 2, gy, cxr - 8, top + 92, "#7d5ba6"))
        note1, y1 = swatch_note(cxr, top + 14, "fullattn",
                                "full-attention layer (12 of 48): holds a growing KV cache — 2 GQA KV "
                                "heads, 32 Q heads. KV grows per token, so it is TOKEN-sharded across "
                                "ranks by uneven-DCP (dcp-size 3).", W - cxr - 30)
        b.append(note1)
        note2, y2 = swatch_note(cxr, top + 96, "gdn",
                                "GDN / linear-attention layer (36 of 48): holds a fixed-size recurrent "
                                "state, not a per-token KV cache — its VRAM does not grow with sequence "
                                "length.", W - cxr - 30)
        b.append(note2)
        note3, y3 = flow(cxr, y2 + 14,
                         "Model: Qwen3.5-122B-A10B-GPTQ-Int4 — 256 experts/layer, top-8 routing, ~10B "
                         "active. This structure is identical under htsglang and upstream sglang; the "
                         "difference is entirely in HOW each layer is split across the 3 mismatched cards "
                         "(next diagrams).", W - cxr - 30, size=9.5, color="#333")
        b.append(note3)
        bottom = max(top + n * cellh + 34, y3)
        return "".join(b), bottom
    compose("10-model-layer-stack.svg", 940,
            "10 — The model both stacks run: 48-layer hybrid-GDN, 256-expert MoE",
            "Qwen3.5-122B-A10B-GPTQ-Int4. Shared structure; the fork's work is in the split, shown next.",
            [("Full-attention layers carry the KV cache (2 KV heads); GDN layers carry a fixed recurrent "
              "state. The MTP/NEXTN draft head is one extra layer on rank0 (§3, shipped).", "#333")],
            draw, ["fullattn", "gdn", "mtp", "state"])


def rank_col(x, top, wcol, title, sub, cells, tint, foot=None):
    """A rank column: header + vertically stacked labelled tensor cells."""
    o = [text(x + wcol / 2, top - 20, title, size=11, anchor="middle", weight="bold"),
         text(x + wcol / 2, top - 8, sub, size=9, anchor="middle", color="#555")]
    cy = top
    for key, h, label, tc in cells:
        if key in COLORS:
            o.append(rect(x, cy, wcol, h, COLORS[key], stroke="#2b2b2b", sw=1.0))
        else:
            o.append(rect(x, cy, wcol, h, key, stroke="#2b2b2b", sw=1.0))
        col = tc or ("#fff" if key in WHITE_TEXT else "#1a1a1a")
        o.append(text(x + wcol / 2, cy + h / 2 + 3.4, label, size=9, anchor="middle", color=col))
        cy += h + 3
    o.append(rect(x - 4, top - 4, wcol + 8, cy - top + 4, "none", stroke=tint, sw=2.0, rx=4))
    if foot:
        o.append(text(x + wcol / 2, cy + 12, foot, size=8.5, anchor="middle", color="#333"))
        cy += 16
    return "".join(o), cy


def g11():
    """Per-layer tensor split across 3 ranks: uneven-TP vs stock even-TP=3."""
    def draw(topY, W):
        b = []
        mid = W / 2
        b.append(vdivider(mid, topY - 4, topY + 330))
        b.append(panel_label(30, topY - 2, "htsglang — uneven-TP=3, ratio [28,19,19]", "ours"))
        b.append(panel_label(mid + 20, topY - 2, "upstream sglang — even-TP=3", "stock"))
        top = topY + 44
        wcol = 132
        # --- ours: three uneven rank columns ---
        ours = [
            ("rank0 — 5090", "cuda:0 · 28000 MiB", [
                ("weights", 58, "Q: 16 heads", None),
                ("kv", 24, "KV: 2 heads (repl.)", None),
                ("weights", 46, "FFN 512 = 4 grp", None),
                ("resident", 22, "256 experts →offload", None)], "#1d6b34"),
            ("rank1 — 3080", "cuda:1 · 19000 MiB", [
                ("weights", 30, "Q: 8", None),
                ("kv", 24, "KV: 2 (repl.)", None),
                ("weights", 24, "FFN 256 = 2 grp", None),
                ("resident", 22, "experts →offload", None)], "#1d6b34"),
            ("rank2 — 3080", "cuda:2 · 19000 MiB", [
                ("weights", 30, "Q: 8", None),
                ("kv", 24, "KV: 2 (repl.)", None),
                ("weights", 24, "FFN 256 = 2 grp", None),
                ("resident", 22, "experts →offload", None)], "#1d6b34"),
        ]
        xs = [40, 40 + 165, 40 + 330]
        maxb = top
        for x, (t, s, cells, tint) in zip(xs, ours):
            c, cb = rank_col(x, top, wcol, t, s, cells, tint)
            b.append(c)
            maxb = max(maxb, cb)
        b.append(text(40, maxb + 16, "Every dimension divides cleanly: 32 Q = 16+8+8, 1024 FFN = 512+256+256,",
                      size=9.5, color="#1d6b34"))
        b.append(text(40, maxb + 30, "each FFN shard is a whole multiple of group_size 128 (4/2/2 groups). 2 KV "
                      "heads < 3 ranks, so KV is replicated + token-sharded (next).", size=9.5, color="#1d6b34"))
        # --- stock: three EQUAL columns with failing arithmetic ---
        sx = [mid + 40, mid + 40 + 150, mid + 40 + 300]
        for i, x in enumerate(sx):
            c, _ = rank_col(x, top, 120, f"rank{i}", "even 1/3 share", [
                ("bad", 34, "Q 32/3 = 10.7", "#4a5568"),
                ("bad", 26, "KV 2/3 = 0.7", "#4a5568"),
                ("bad", 40, "FFN 1024/3 = 341.3", "#4a5568"),
                ("bad", 22, "not ÷128", "#4a5568")], "#4a5568")
            b.append(c)
        b.append(text(mid + 40, maxb + 16, "Even split does not divide these dims: 32, 1024 and 2 are not multiples of 3,",
                      size=9.5, color="#4a5568"))
        b.append(text(mid + 40, maxb + 30, "and FFN 341.3 is not a multiple of group_size 128 → the GPTQ "
                      "group-alignment check does not admit an even TP=3 here.", size=9.5, color="#4a5568"))
        return "".join(b), maxb + 36
    compose("11-tp-tensor-split.svg", 1320,
            "11 — One layer's tensors across 3 mismatched cards: uneven-TP vs even-TP",
            "Decisive setting: --rank-tp-ratio auto (§1, shipped). Cell heights are proportional to each rank's share.",
            [("htsglang sizes each rank's Q-head / FFN / expert shard to its VRAM budget (16/8/8 Q, "
              "512/256/256 FFN = 4/2/2 GPTQ groups). Upstream even-TP=3 does not divide these dimensions: "
              "32, 1024 and the 2 KV heads are not divisible by 3, and 341.3 is not group-128-aligned.", "#333")],
            draw, ["weights", "kv", "resident", "bad"])


def g12():
    """Attention under 2 KV heads: replicate + DCP token-shard + LSE-merge."""
    def draw(topY, W):
        b = []
        mid = W / 2
        b.append(vdivider(mid, topY - 4, topY + 360))
        b.append(panel_label(30, topY - 2, "htsglang — TP > num_kv_heads: replicate KV + DCP token-shard", "ours"))
        b.append(panel_label(mid + 20, topY - 2, "upstream sglang — head-sharded KV", "stock"))
        top = topY + 40
        # sequence bar split into 3 rank token-ranges
        sx, sw_ = 40, 520
        fr = [0.42, 0.29, 0.29]
        labels = ["rank0: tokens 0..t0", "rank1: t0..t1", "rank2: t1..T"]
        cx = sx
        b.append(text(sx, top - 6, "sequence KV (per full-attn layer), split along the TOKEN axis:", size=9.5, color="#333"))
        for f, lab, tint in zip(fr, labels, RANKTINT):
            w = sw_ * f
            b.append(rect(cx, top, w, 26, tint, stroke="#2b2b2b", sw=1.0))
            b.append(text(cx + w / 2, top + 16, lab, size=8.5, anchor="middle", color="#1a1a1a"))
            cx += w
        # new-token Q broadcast
        qy = top + 58
        b.append(chip(40, qy, 150, 20, "weights", "Q (new token), 32 heads"))
        # three rank compute boxes
        ry = qy + 52
        rxs = [40, 220, 400]
        for i, x in enumerate(rxs):
            b.append(rect(x, ry, 150, 74, RANKTINT[i], stroke="#2b2b2b", sw=1.2, rx=3))
            b.append(text(x + 75, ry + 14, f"rank{i}", size=9.5, anchor="middle", weight="bold"))
            b.append(chip(x + 8, ry + 22, 60, 16, "kv", "2 KV", size=8))
            b.append(text(x + 105, ry + 33, f"Q {[16,8,8][i]} heads", size=8.5, anchor="middle", color="#333"))
            b.append(text(x + 75, ry + 52, "partial attn", size=8.5, anchor="middle", color="#1a1a1a"))
            b.append(text(x + 75, ry + 65, "+ local LSE", size=8.5, anchor="middle", color="#555"))
            # Q broadcast arrow
            b.append(arrow(115, qy + 20, x + 75, ry - 2, "#3b6fb0", dash="4 3", marker="arrB"))
        # LSE-merge
        my = ry + 100
        b.append(rect(180, my, 200, 26, COLORS["kv"], stroke="#2b2b2b", sw=1.2, rx=3))
        b.append(text(280, my + 16, "LSE-merge → attention out", size=9.5, anchor="middle", color="#fff"))
        for x in rxs:
            b.append(arrow(x + 75, ry + 74, 280, my - 2, "#227a3a", marker="arrG"))
        b.append(text(40, my + 48, "Each rank holds BOTH KV heads (replicated) but only its token-slice of the",
                      size=9.5, color="#1d6b34"))
        b.append(text(40, my + 62, "cache. Q is broadcast; ranks attend their slice; LSE-merge stitches the",
                      size=9.5, color="#1d6b34"))
        b.append(text(40, my + 76, "partial softmaxes. Aggregate KV capacity scales with ranks (§1, shipped).",
                      size=9.5, color="#1d6b34"))
        # --- stock panel ---
        sx2 = mid + 40
        b.append(text(sx2, top - 6, "upstream shards KV by HEAD — one KV head per rank:", size=9.5, color="#333"))
        b.append(chip(sx2, top, 90, 40, "kv", "KV head 0", size=9))
        b.append(chip(sx2 + 110, top, 90, 40, "kv", "KV head 1", size=9))
        b.append(arrow(sx2 + 45, top + 44, sx2 + 30, ry - 6, "#227a3a", marker="arrG"))
        b.append(arrow(sx2 + 155, top + 44, sx2 + 150, ry - 6, "#227a3a", marker="arrG"))
        for i in range(3):
            x = sx2 + i * 150
            ok = i < 2
            b.append(rect(x, ry, 130, 60, RANKTINT[i] if ok else COLORS["bad"], stroke="#2b2b2b", sw=1.2, rx=3))
            b.append(text(x + 65, ry + 20, f"rank{i}", size=9.5, anchor="middle", weight="bold"))
            b.append(text(x + 65, ry + 40, f"KV head {i}" if ok else "no head-axis KV",
                          size=9.5, anchor="middle", color="#1a1a1a" if ok else "#4a5568", weight="normal" if ok else "bold"))
        b.append(text(sx2, my, "2 KV heads, 3 ranks: rank2 is not covered by the head split.", size=9.5, color="#4a5568", weight="bold"))
        b.append(text(sx2, my + 15, "Head-axis KV requires num_kv_heads ≥ tp (2 < 3 here).", size=9.5, color="#4a5568"))
        b.append(text(sx2, my + 30, "Replicating the whole KV on every rank is possible and stores", size=9.5, color="#4a5568"))
        b.append(text(sx2, my + 44, "the full cache on each card (no token-capacity gain from more cards).", size=9.5, color="#4a5568"))
        return "".join(b), my + 90
    compose("12-attention-dcp.svg", 1320,
            "12 — 2 KV heads across 3 ranks: the DCP token-shard (the crux)",
            "Decisive setting: replicated-KV + uneven-DCP (§1/§9, shipped). Why 2 KV heads on 3 cards is the hard case.",
            [("The 122B has only 2 GQA KV heads but runs TP=3. It cannot head-shard (2<3), so the fork "
              "REPLICATES the 2 KV heads on every rank and shards the KV cache by TOKEN (dcp-size 3): each "
              "rank attends its token-slice, then a Q-broadcast + LSE-merge stitches the partial softmaxes.", "#333")],
            draw, ["weights", "kv", "bad"])


def g13():
    """Per-token KV-cache memory arrangement across ranks + MTP reservation."""
    def draw(topY, W):
        b = []
        mid = W / 2
        b.append(vdivider(mid, topY - 4, topY + 320))
        b.append(panel_label(30, topY - 2, "htsglang — token-sharded KV (per full-attn layer)", "ours"))
        b.append(panel_label(mid + 20, topY - 2, "upstream sglang — head-sharded / replicated KV", "stock"))
        top = topY + 46
        gx, gw = 40, 470
        rows = 12                     # the 12 full-attn layers
        rh = 12
        fr = [0.42, 0.29, 0.29]
        # column bands
        cx = gx
        for f, tint, lab in zip(fr, RANKTINT, ["rank0 tokens 0..t0", "rank1 t0..t1", "rank2 t1..T"]):
            w = gw * f
            b.append(rect(cx, top, w, rows * rh, tint, stroke="#ffffff", sw=0))
            b.append(text(cx + w / 2, top - 5, lab, size=8.5, anchor="middle", color="#333"))
            cx += w
        # row grid lines + labels
        for r in range(rows + 1):
            b.append(line(gx, top + r * rh, gx + gw, top + r * rh, stroke="#c8ccd0", sw=0.6))
        b.append(rect(gx, top, gw, rows * rh, "none", stroke="#222", sw=1.4))
        b.append(text(gx - 6, top + 9, "L47", size=8, anchor="end", color="#555"))
        b.append(text(gx - 6, top + rows * rh - 2, "L3", size=8, anchor="end", color="#555"))
        b.append(text(gx + gw / 2, top + rows * rh + 15, "12 full-attention layers × (2 KV heads) — each rank owns a TOKEN band",
                      size=9, anchor="middle", color="#333"))
        # MTP strip on rank0
        mtpy = top + rows * rh + 26
        b.append(chip(gx, mtpy, gw * fr[0], 16, "mtp", "MTP draft KV (rank0)", size=8))
        # GDN state block (separate, small, per rank)
        sy = mtpy + 30
        b.append(text(gx, sy - 4, "36 GDN layers: fixed recurrent state (no per-token growth)", size=9, color="#333"))
        for i, f in enumerate(fr):
            b.append(chip(gx + sum(fr[:i]) * gw, sy, gw * f, 16, "state", "GDN state", size=8))
        b.append(text(gx, sy + 42, "Per-rank KV memory = 12 layers × 2 KV heads × its token-band. Growing the",
                      size=9.5, color="#1d6b34"))
        b.append(text(gx, sy + 56, "context adds tokens split 3 ways → aggregate KV capacity scales with ranks.",
                      size=9.5, color="#1d6b34"))
        # --- stock panel ---
        sx2 = mid + 40
        b.append(text(sx2, top - 5, "Option A — head-shard (upstream default):", size=9, color="#333"))
        b.append(rect(sx2, top, 240, rows * rh, COLORS["bad"], stroke="#222", sw=1.2))
        b.append(text(sx2 + 120, top + rows * rh / 2, "2 KV heads / 3 ranks", size=10, anchor="middle", weight="bold", color="#4a5568"))
        b.append(text(sx2, top + rows * rh + 20, "Option B — replicate KV on every rank:", size=9, color="#333"))
        for i in range(3):
            b.append(rect(sx2 + i * 130, top + rows * rh + 28, 120, 34, R0T, stroke="#222", sw=1.0))
            b.append(text(sx2 + i * 130 + 60, top + rows * rh + 49, "ALL tokens", size=8.5, anchor="middle", color="#1a1a1a"))
        b.append(text(sx2, top + rows * rh + 84, "Every rank stores the WHOLE sequence's KV → 1× capacity,",
                      size=9.5, color="#4a5568"))
        b.append(text(sx2, top + rows * rh + 98, "no token-capacity gain from more cards. Even-TP dims: see diagram 11.",
                      size=9.5, color="#4a5568"))
        return "".join(b), sy + 66
    compose("13-kv-token-layout.svg", 1200,
            "13 — KV-cache arrangement: which rank holds which tokens, and where MTP sits",
            "Decisive setting: uneven-DCP token sharding (§1, shipped). GDN layers hold state, not KV.",
            [("Only the 12 full-attention layers hold a KV cache; the fork splits it by TOKEN band across "
              "ranks (rank0 gets the largest band, matching its shard). The MTP draft head keeps its own "
              "small KV band on rank0. The 36 GDN layers hold a fixed recurrent state instead.", "#333")],
            draw, ["fullattn", "state", "mtp"])


def g14():
    """Per-layer MoE expert-offload flow (f=0.25) vs stock (no offload)."""
    def draw(topY, W):
        b = []
        mid = W / 2
        b.append(vdivider(mid, topY - 4, topY + 360))
        b.append(panel_label(30, topY - 2, "htsglang — per-expert host offload, f=0.25 (§6, [in progress])", "ours"))
        b.append(panel_label(mid + 20, topY - 2, "upstream sglang — no per-expert host spill", "stock"))
        top = topY + 44
        # GPU strip: resident + scratch chips
        gx = 40
        b.append(text(gx, top - 6, "GPU — 80 expert slots resident (per layer):", size=9.5, color="#333"))
        cw, gap = 11, 2
        # 64 resident chips in a 16-wide grid
        for k in range(64):
            r, cc = divmod(k, 16)
            b.append(chip(gx + cc * (cw + gap), top + r * (cw + gap), cw, cw, "resident"))
        rx = gx + 16 * (cw + gap) + 14
        b.append(text(gx + 8 * (cw + gap), top + 4 * (cw + gap) + 8, "64 resident (slots 0..63)", size=8.5, anchor="middle", color="#333"))
        # 16 scratch chips
        for k in range(16):
            r, cc = divmod(k, 8)
            b.append(chip(rx + cc * (cw + gap), top + r * (cw + gap), cw, cw, "scratch"))
        b.append(text(rx + 4 * (cw + gap), top + 2 * (cw + gap) + 8, "16 scratch (64..79)", size=8.5, anchor="middle", color="#333"))
        # host strip: 176 spill
        hy = top + 96
        b.append(text(gx, hy - 6, "Pinned host RAM — 176 spilled experts (per layer × 48 = host floor 24.4 GB):", size=9.5, color="#333"))
        for k in range(176):
            r, cc = divmod(k, 44)
            b.append(chip(gx + cc * (cw + gap), hy + r * (cw + gap), cw, cw, "spill", stroke="#a03b38", sw=0.5))
        # flow steps
        fy = hy + 92
        steps = [
            "1. router picks top-8 experts for the token-wave",
            "2. resident hits → read on GPU directly",
            "3. spill misses → fetch H2D over PCIe into scratch slots",
            "4. resident + fetched → grouped GEMM → output tokens",
        ]
        ty = fy
        for s in steps:
            b.append(text(gx, ty, s, size=9.5, color="#1a1a1a"))
            ty += 15
        # arrow: host -> scratch (PCIe), scratch -> GEMM
        b.append(arrow(gx + 22 * (cw + gap), hy + 8, rx + 3 * (cw + gap), top + 20, "#8a5a2b", marker="arrP"))
        b.append(text(gx + 300, hy - 20, "PCIe H2D fetch", size=8.5, color="#8a5a2b"))
        b.append(rect(gx + 250, ty + 6, 150, 24, COLORS["weights"], stroke="#222", sw=1.1, rx=3))
        b.append(text(gx + 325, ty + 22, "grouped GEMM", size=9, anchor="middle", color="#fff"))
        b.append(arrow(rx + 3 * (cw + gap), top + 30, gx + 320, ty + 4, "#3b6fb0", marker="arrB"))
        # eager banner
        eby = ty + 40
        b.append(rect(gx, eby, mid - gx - 30, 34, "#fdf3e3", stroke="#c88a2b", sw=1.2, rx=4))
        b.append(text(gx + 10, eby + 14, "EAGER (--disable-cuda-graph): the expert set is data-dependent per wave,",
                      size=9, color="#8a5a2b"))
        b.append(text(gx + 10, eby + 27, "so the decode path is not CUDA-graph-capturable. Graph-capture for offload: [planned].",
                      size=9, color="#8a5a2b"))
        # --- stock panel ---
        sx2 = mid + 40
        b.append(text(sx2, top - 6, "Option A — keep all 256 experts resident in VRAM:", size=9.5, color="#333"))
        for k in range(256):
            r, cc = divmod(k, 32)
            over = r >= 5
            b.append(chip(sx2 + cc * (cw - 2 + gap), top + r * (cw - 2 + gap), cw - 2, cw - 2,
                          "spill" if over else "resident", stroke="#888", sw=0.4))
        b.append(rect(sx2 - 4, top - 4, 32 * (cw - 2 + gap) + 4, 5 * (cw - 2 + gap) + 2, "none", stroke="#1d6b34", sw=1.6, rx=3))
        b.append(text(sx2 + 16 * (cw - 2 + gap), top + 5 * (cw - 2 + gap) + 12, "↑ within aggregate VRAM ·  ↓ beyond it", size=8.5, anchor="middle", color="#4a5568"))
        oy = top + 8 * (cw - 2 + gap) + 26
        b.append(text(sx2, oy, "256 experts × 48 layers ≈ 65 GB vs 72 GB aggregate; per-card context +", size=9.5, color="#4a5568"))
        b.append(text(sx2, oy + 14, "KV must also fit, and even-TP=3 does not divide these dims (diagram 11).", size=9.5, color="#4a5568"))
        b.append(text(sx2, oy + 36, "Option B — --cpu-offload-gb: generic, LAYER-granular, dequantizes on", size=9.5, color="#4a5568"))
        b.append(text(sx2, oy + 50, "CPU, not per-expert / not quant-aware; no per-wave prefetch.", size=9.5, color="#4a5568"))
        return "".join(b), max(eby + 40, oy + 60)
    compose("14-expert-offload-flow.svg", 1340,
            "14 — Per-layer MoE expert offload: 256 experts → 64 resident + 16 scratch + 176 host",
            "Decisive setting: SGLANG_MOE_RESIDENT_EXPERT_FRACTION<1 (§6, [in progress]). f=0.25 on the measured run.",
            [("Only ceil(f·256)=64 experts stay resident per layer, plus a 16-slot scratch buffer; the other "
              "176 live in pinned host RAM. Per token-wave the top-8 are gathered — resident ones read "
              "directly, spilled ones are fetched H2D over PCIe into scratch, then the grouped GEMM runs. "
              "Wave-over-tokens keeps it coherent (§6). Upstream has no per-expert, quant-aware host spill.", "#333")],
            draw, ["resident", "scratch", "spill", "weights"])


def g15():
    """Full 122B TP=3 end-state rig map (measured) vs stock 'cannot run'."""
    def draw(topY, W):
        b = []
        mid = W / 2
        b.append(panel_label(30, topY - 2, "htsglang — 122B-A10B-Int4, TP=3, f=0.25 (measured: 6.97 tok/s)", "ours"))
        b.append(panel_label(mid + 20, topY - 2, "upstream sglang — same 3 cards", "stock"))
        top = topY + 52
        PX = 4.0
        cards = [("RTX 5090", 32.6, "cuda:0 · rank0 · 25.5 GB used",
                  [("weights", 3.0, "Q16+KV2+FFN512 shard"),
                   ("resident", 16.0, "64+16 experts/layer ×48"),
                   ("kv", 5.0, "KV token-band (rank0)"),
                   ("mtp", 1.0, "MTP head")]),
                 ("RTX 3080", 20.48, "cuda:1 · rank1 · 16.6 GB used",
                  [("weights", 2.2, "Q8+KV2+FFN256"),
                   ("resident", 11.0, "experts/layer ×48"),
                   ("kv", 3.4, "KV token-band")]),
                 ("RTX 3080", 20.48, "cuda:2 · rank2 · 15.4 GB used",
                  [("weights", 2.2, "Q8+KV2+FFN256"),
                   ("resident", 10.0, "experts/layer ×48"),
                   ("kv", 3.2, "KV token-band")])]
        xs = [40, 210, 380]
        base = top + 32.6 * PX
        for (nm, vram, sub, regs), x in zip(cards, xs):
            th = top + (32.6 - vram) * PX
            b.append(text(x + 70, th - 20, nm, size=11, anchor="middle", weight="bold"))
            b.append(text(x + 70, th - 8, sub, size=8, anchor="middle", color="#555"))
            b.append(rect(x, th, 140, vram * PX, "none", stroke="#222", sw=1.5))
            cy = th
            for key, gb, lab in regs:
                b.append(rect(x, cy, 140, gb * PX, COLORS[key]))
                tc = "#fff" if key in WHITE_TEXT else "#1a1a1a"
                if gb * PX >= 12:
                    b.append(text(x + 70, cy + gb * PX / 2 + 3, lab, size=8, anchor="middle", color=tc))
                cy += gb * PX
            b.append(rect(x, cy, 140, base - cy, COLORS["free"]))
            if base - cy >= 12:
                b.append(text(x + 70, cy + (base - cy) / 2 + 3, "free", size=8, anchor="middle", color="#1a1a1a"))
        # host RAM bar
        hby = base + 40
        for i, x in enumerate(xs):
            b.append(down_link(x + 110, base, hby, "x16", "PCIe" if i == 1 else None))
        hb, hy = host_ram_bar(40, hby, 480,
                              [("spill", 0.66, "176 spilled experts/layer ×48 (pinned)"), ("free", 0.34, "host floor 24.4 GB")],
                              "Pinned host RAM (DDR)")
        b.append(hb)
        # eager + tok/s banner
        b.append(rect(40, hy + 14, 480, 30, "#fdf3e3", stroke="#c88a2b", sw=1.1, rx=4))
        b.append(text(50, hy + 33, "decode: EAGER (--disable-cuda-graph, offload is data-dependent) · 6.97 tok/s · MTP on rank0",
                      size=9, color="#8a5a2b"))
        # --- stock panel: 3 cards, walls ---
        sx = [mid + 40, mid + 210, mid + 380]
        for (nm, vram, sub, _), x in zip(cards, sx):
            th = top + (32.6 - vram) * PX
            b.append(text(x + 70, th - 20, nm, size=11, anchor="middle", weight="bold"))
            b.append(rect(x, th, 140, vram * PX, COLORS["bad"], stroke="#222", sw=1.5))
            b.append(text(x + 70, th + vram * PX / 2, "even TP=3", size=9, anchor="middle", color="#4a5568", weight="bold"))
            b.append(text(x + 70, th + vram * PX / 2 + 14, "not admitted", size=9, anchor="middle", color="#4a5568"))
        wy = base + 34
        for i, s in enumerate([
                "even-TP=3: 32 Q / 1024 FFN / 2 KV not ÷3, FFN not ÷128 → not admitted (diagram 11)",
                "head-sharded KV: requires num_kv_heads(2) ≥ tp(3) (diagram 12)",
                "no per-expert host spill: 256 experts ×48 ≈ 65 GB vs 72 GB aggregate; KV must also fit (diagram 14)",
                "even-TP=2 fallback sizes both ranks to the 3080 shard → ~12 GB of the 5090 unused, full model does not fit"]):
            b.append(text(mid + 40, wy + i * 15, "• " + s, size=9, color="#4a5568"))
        return "".join(b), max(hy + 50, wy + 4 * 15)
    compose("15-endstate-rig-map.svg", 1320,
            "15 — 122B-A10B-Int4 TP=3 end-state: the whole rig, measured",
            "5090 + 2×3080, PCIe, no NVLink. Everything from diagrams 10-14 combined on the real config.",
            [("The measured end-state: uneven-TP [28,19,19], 2 KV heads replicated + DCP token-shard, "
              "per-layer expert offload f=0.25 (64+16 resident, 176 host), MTP on rank0, eager decode. "
              "Per-rank GPU 25.5 / 16.6 / 15.4 GB; host floor 24.4 GB; 6.97 tok/s. On the upstream path a "
              "TP=3 split on this hardware meets the structural constraints listed at right.", "#333")],
            draw, ["weights", "resident", "kv", "mtp", "spill", "free", "bad"])


def g16():
    """Smaller concrete offload examples: solo 1-card 122B (f=0.15) + 35B TP=3."""
    def draw(topY, W):
        b = []
        mid = W / 2
        b.append(vdivider(mid, topY - 4, topY + 300))
        b.append(panel_label(30, topY - 2, "solo 1-card 122B — f=0.15 (39 resident)", "ours"))
        b.append(panel_label(mid + 20, topY - 2, "35B-A3B TP=3 — 2/1/1 groups, 11.83 tok/s", "ours"))
        PX = 4.0
        top = topY + 50
        # solo card
        x = 60
        b.append(text(x + 70, top - 18, "RTX 5090 (solo, TP=1)", size=10.5, anchor="middle", weight="bold"))
        b.append(rect(x, top, 140, 32.6 * PX, "none", stroke="#222", sw=1.5))
        cy = top
        for key, gb, lab in [("weights", 4, "attn+shared wts"), ("resident", 12, "39 resident experts/layer"),
                             ("scratch", 2, "scratch"), ("kv", 6, "KV cache"), ("mtp", 1.2, "MTP")]:
            b.append(rect(x, cy, 140, gb * PX, COLORS[key]))
            tc = "#fff" if key in WHITE_TEXT else "#1a1a1a"
            if gb * PX >= 12:
                b.append(text(x + 70, cy + gb * PX / 2 + 3, lab, size=8, anchor="middle", color=tc))
            cy += gb * PX
        b.append(rect(x, cy, 140, top + 32.6 * PX - cy, COLORS["free"]))
        b.append(down_link(x + 170, top + 32.6 * PX - 40, top + 32.6 * PX + 34, "x16", "PCIe"))
        hb, hy = host_ram_bar(x, top + 32.6 * PX + 34, 320,
                              [("spill", 0.78, "208 spilled experts/layer (pinned)"), ("free", 0.22, "floor ~28 GB")],
                              "Pinned host RAM (DDR)")
        b.append(hb)
        b.append(text(x, hy + 20, "Same offload mechanism at TP=1: fewer resident (f=0.15 → 39),", size=9, color="#333"))
        b.append(text(x, hy + 33, "more host spill. Boots the 122B on a single 32 GB card (§6).", size=9, color="#333"))
        # 35B TP=3 mini map
        sx = mid + 40
        for i, (nm, vram, grp) in enumerate([("5090", 32.6, "FFN 2 grp"), ("3080", 20.48, "1 grp"), ("3080", 20.48, "1 grp")]):
            xx = sx + i * 150
            th = top + (32.6 - vram) * PX
            b.append(text(xx + 60, th - 16, nm, size=10, anchor="middle", weight="bold"))
            b.append(rect(xx, th, 120, vram * PX, "none", stroke="#222", sw=1.3))
            cy = th
            for key, gb, lab in [("weights", 2.5, "wts · " + grp), ("resident", 5, "experts"), ("kv", 4, "KV band")]:
                b.append(rect(xx, cy, 120, gb * PX, COLORS[key]))
                tc = "#fff" if key in WHITE_TEXT else "#1a1a1a"
                if gb * PX >= 12:
                    b.append(text(xx + 60, cy + gb * PX / 2 + 3, lab, size=7.5, anchor="middle", color=tc))
                cy += gb * PX
            b.append(rect(xx, cy, 120, top + 32.6 * PX - cy, COLORS["free"]))
        b.append(text(sx, top + 32.6 * PX + 24, "35B-A3B-AWQ: uneven-TP=3 + DCP + offload, mixed-arch. Stage(a)", size=9, color="#333"))
        b.append(text(sx, top + 32.6 * PX + 37, "32/32 token-identical to TP=1; self-det 5/5; 11.83 tok/s (§6/§7).", size=9, color="#333"))
        return "".join(b), hy + 44
    compose("16-solo-and-35b.svg", 1200,
            "16 — Smaller concrete examples: solo-card 122B and 35B TP=3",
            "The same offload + uneven-TP machinery at other scales (§6/§7, [in progress]/shipped).",
            [("Solo 122B on one 5090 uses f=0.15 (39 resident experts/layer, host floor ~28 GB). The 35B-A3B "
              "runs uneven-TP=3 (2/1/1 groups) with offload and was validated 32/32 token-identical to TP=1 "
              "under uneven-TP+DCP.", "#333")],
            draw, ["weights", "resident", "scratch", "kv", "mtp", "spill", "free"])


# ===========================================================================
# Part 3 — runtime capabilities (speculative routing, memory, scheduling).
# These explain the mechanism of a landed/experimental fork feature rather than
# a GPU-placement topology. Region colours stay consistent with the atlas.
# ===========================================================================
DFLASH_FILL, DFLASH_STROKE, DFLASH_TXT = "#e4dcf1", "#7d5ba6", "#4a3b66"


def g17():
    """Adaptive drafter routing: NEXTN/DFLASH dual residence, policy vs bandit, ctx-gate."""
    def draw(topY, W):
        b = []
        top = topY + 6
        # 1) dual residence
        b.append(text(40, top, "Both drafters resident at once (cross_algo_worker, dual residence):",
                      size=10.5, weight="bold", color="#111"))
        y = top + 10
        b.append(chip(40, y, 150, 30, "mtp", "NEXTN / MTP", size=9.5))
        b.append(rect(205, y, 150, 30, DFLASH_FILL, stroke=DFLASH_STROKE, sw=1.4, rx=2))
        b.append(text(280, y + 19, "DFLASH", size=9.5, anchor="middle", color=DFLASH_TXT, weight="bold"))
        b.append(rect(370, y, 130, 30, "#eef1f4", stroke="#4a5568", sw=1.1, rx=2))
        b.append(text(435, y + 13, "solo-draft pool", size=8.5, anchor="middle", color="#333"))
        b.append(text(435, y + 24, "sized to threshold (→160 MiB)", size=7.6, anchor="middle", color="#555"))
        b.append(text(520, y + 13, "the inactive drafter holds ≈0 VRAM", size=9, color=DFLASH_STROKE))
        b.append(text(520, y + 25, "(VMM tag-alias, #93/#102)", size=9, color=DFLASH_STROKE))
        # 2) router
        ry = y + 52
        b.append(rect(40, ry, W - 80, 30, "#f3f6fa", stroke="#3b6fb0", sw=1.3, rx=4))
        b.append(text(50, ry + 19, "Per-round-boundary router — picks ONE drafter for the next batch; the choice is made by one of two modes:",
                      size=9.5, color="#1a1a1a"))
        # 3) two mode cards
        cy = ry + 48
        cardw = (W - 80 - 30) / 2
        # policy card (recommended default)
        b.append(rect(40, cy, cardw, 132, "#eef5ef", stroke="#1d6b34", sw=1.4, rx=5))
        b.append(text(52, cy + 17, "policy  —  recommended default", size=10, weight="bold", color="#1d6b34"))
        b.append(text(52, cy + 32, "deterministic ctx → rung table (--speculative-drafter-policy)", size=8.6, color="#333"))
        rows = [("ctx < 4096", "DFLASH, k=16", DFLASH_FILL, DFLASH_STROKE),
                ("ctx ≥ 4096", "NEXTN, k* (analytic)", COLORS["mtp"], "#8a3b6b")]
        for i, (cond, act, fill, st) in enumerate(rows):
            yy = cy + 44 + i * 26
            b.append(text(60, yy + 15, cond, size=9, color="#1a1a1a"))
            b.append(text(150, yy + 15, "→", size=11, color="#555"))
            b.append(rect(172, yy + 2, 168, 20, fill, stroke=st, sw=1.0, rx=2))
            b.append(text(256, yy + 16, act, size=8.6, anchor="middle",
                          color="#fff" if fill == COLORS["mtp"] else DFLASH_TXT))
        b.append(text(52, cy + 108, "k* = argmax_k  E[accept]/round_s  from per-depth accept EMA.", size=8.4, color="#333"))
        b.append(text(52, cy + 122, "Fixed switch point = drafter training-ctx; probing not needed.", size=8.4, color="#333"))
        # bandit card (opt-in)
        bx = 40 + cardw + 30
        b.append(rect(bx, cy, cardw, 132, "#eef1f4", stroke="#4a5568", sw=1.4, rx=5))
        b.append(text(bx + 12, cy + 17, "auto / bandit  —  opt-in (--speculative-cross-algorithm)", size=10, weight="bold", color="#4a5568"))
        b.append(text(bx + 12, cy + 33, "acceptance-driven, for unknown drafters or content-split 4-8k loads", size=8.6, color="#333"))
        b.append(rect(bx + 12, cy + 42, cardw - 24, 24, "#ffffff", stroke="#4a5568", sw=1.0, rx=3))
        b.append(text(bx + 20, cy + 58, "score = EMA[accept-tokens / round] ÷ EMA[round seconds]", size=8.6, color="#1a1a1a"))
        b.append(text(bx + 12, cy + 82, "rank-0 decides, gloo-broadcasts every 16 rounds; dwell 64,", size=8.4, color="#333"))
        b.append(text(bx + 12, cy + 95, "dead-zone 6%. Costs a small steady-state probe overhead.", size=8.4, color="#333"))
        b.append(text(bx + 12, cy + 116, "Prior art: BanditSpec (arXiv 2505.15141, ICML'25).", size=8.4, color="#555"))
        # 4) ctx gate
        gy = cy + 150
        b.append(rect(40, gy, W - 80, 30, "#fdf3e3", stroke="#c88a2b", sw=1.2, rx=4))
        b.append(text(50, gy + 19, "Context-length gate (--speculative-cross-algorithm-ctx-gate, from the drafter training config, ~8k): "
                      "above the gate DFLASH is ineligible and is not probed.", size=9, color="#8a5a2b"))
        # 5) honest claim
        hn, hy = flow(40, gy + 48, "Honest claim: robustness / no-regret across mixed streams — not a peak speedup. "
                      "A switching mode carries ≈+5.7% systemic overhead vs a single static drafter "
                      "(warm-keep + dual capture), so the win is confined to streams that actually change "
                      "regime. Feature status: work in progress (§5).", W - 80, size=9.5, color="#333")
        b.append(hn)
        return "".join(b), hy
    compose("17-adaptive-drafter-routing.svg", 1120,
            "17 — Adaptive drafter routing: switch draft algorithm at round boundaries",
            "Two draft algorithms resident at once; one chosen per batch. Decisive setting: drafter routing (§5, work in progress).",
            [("NEXTN/MTP and DFLASH are both loaded (the inactive one held at ≈0 VRAM via VMM tag-aliasing). "
              "A per-round router selects one, either by a deterministic ctx→rung policy table (recommended "
              "default) or an acceptance-driven bandit (opt-in), with a context-length gate that keeps DFLASH "
              "to its trained range. Upstream adaptive spec-decode adapts k / num-draft-tokens for a single "
              "drafter; switching between draft algorithms is the fork addition.", "#333")],
            draw, ["mtp"])


def g18():
    """Session KV spill: newest session's KV offloaded to host, keeps decoding."""
    def draw(topY, W):
        b = []
        top = topY + 10
        # device lane: session KV blocks, oldest..newest
        b.append(text(40, top, "Device VRAM — full-attention KV, one shard per active session (FCFS):",
                      size=10, weight="bold", color="#111"))
        ly = top + 12
        sess = [("S1 oldest", 0.9), ("S2", 0.9), ("S3", 0.9), ("S4 newest", 0.55)]
        x = 40
        for i, (nm, sc) in enumerate(sess):
            w = 118
            h = 46
            newest = i == len(sess) - 1
            b.append(rect(x, ly, w, h, COLORS["kv"], stroke="#2b2b2b", sw=1.3 if not newest else 1.6,
                          dash="4 3" if newest else None))
            b.append(text(x + w / 2, ly + 20, nm, size=9, anchor="middle", color="#fff", weight="bold"))
            b.append(text(x + w / 2, ly + 34, "KV shard", size=8, anchor="middle", color="#fff"))
            x += w + 12
        # GDN resident chip
        b.append(chip(x + 8, ly, 96, 46, "state", "GDN state", size=8.5))
        b.append(text(x + 56, ly + 60, "always resident", size=8, anchor="middle", color="#4a3b66"))
        # overflow marker + spill arrow (newest -> host)
        newest_x = 40 + 3 * (118 + 12)
        b.append(text(newest_x + 59, ly - 2, "VRAM overflow (after tree eviction)", size=8.2, anchor="middle", color="#c0504d"))
        # host lane
        hy0 = ly + 96
        hb, hy = host_ram_bar(40, hy0, W - 80,
                              [("hostkv", 0.30, "S4 KV (host-streamed, still decoding)"),
                               ("free", 0.70, "host KV tier")],
                              "Host RAM (DDR) — spilled session KV")
        b.append(hb)
        b.append(arrow(newest_x + 59, ly + 46, 40 + (W - 80) * 0.15, hy0, "#c0504d", marker="arrR"))
        b.append(text(newest_x + 120, ly + 74, "spill newest first (FCFS victim)", size=8.4, color="#c0504d"))
        b.append(arrow(40 + (W - 80) * 0.30, hy0, newest_x + 59, ly + 48, "#3f9fa0", dash="4 3", marker="arr"))
        b.append(text(40 + (W - 80) * 0.42, hy0 + 18, "FIFO restore when device capacity frees (~0.4 s)", size=8.4, color="#227a3a"))
        # mechanism notes
        n1, ny = flow(40, hy + 26,
                      "Mechanism: on KV overflow the NEWEST session's full-attention KV shard is moved to host "
                      "RAM (block-LSE attention + double-buffer prefetch, reused from the weightless lane) and "
                      "that session keeps DECODING from host, in a separate eager bs=1 tick — never mixed into "
                      "the device CUDA-graph batch. The oldest session stays fully device-resident until it "
                      "finishes (strict FCFS). Only KV spills; GDN/Mamba state is always resident. Fast-lane "
                      "requests take precedence (restore is held while a fast request waits).",
                      W - 80, size=9.3, color="#333")
        b.append(n1)
        # measured vs modeled
        my = ny + 8
        b.append(rect(40, my, (W - 80) / 2 - 10, 78, "#eef5ef", stroke="#1d6b34", sw=1.3, rx=4))
        b.append(text(52, my + 16, "S1 — measured (landed)", size=9.5, weight="bold", color="#1d6b34"))
        for i, s in enumerate(["zero-overhead when unused +0.16% (<1% bar)",
                                "host decode 8.1 tok/s @1k ctx; restore 0.4 s",
                                "determinism 50/50 exact host-vs-device tokens"]):
            b.append(text(52, my + 32 + i * 14, "• " + s, size=8.4, color="#1a1a1a"))
        mx = 40 + (W - 80) / 2 + 10
        b.append(rect(mx, my, (W - 80) / 2 - 10, 78, "#eef1f4", stroke="#4a5568", sw=1.3, rx=4))
        b.append(text(mx + 12, my + 16, "design model — modeled, not benchmarked (needs S2)", size=9.2, weight="bold", color="#4a5568"))
        for i, s in enumerate(["12 KiB/token fp8, x4 link, DCP R=3 + overlap:",
                                "32k ≈63 · 64k ≈31 · 128k ≈16 · 262k ≈7.6 tok/s",
                                "worthwhile only with uneven DCP (262k ≈3.8 without)"]):
            b.append(text(mx + 12, my + 32 + i * 14, "• " + s, size=8.4, color="#1a1a1a"))
        return "".join(b), my + 92
    compose("18-session-kv-spill.svg", 1120,
            "18 — Session KV spill: overflow the newest session to host RAM, keep decoding",
            "Per-session KV offload under VRAM pressure. Decisive setting: --enable-kv-session-offload (§20, experimental S1).",
            [("When device KV overflows, the newest active session's KV shard is offloaded to host RAM and that "
              "session keeps decoding via host-streamed attention, instead of being paused. Victim order is "
              "strict FCFS (oldest stays resident) with fast-lane precedence; sessions restore FIFO when "
              "capacity frees; only KV spills, GDN state stays resident. Upstream retracts/recomputes or swaps "
              "a request (paused, not decoded from host).", "#333")],
            draw, ["kv", "hostkv", "state", "free"])


def g19():
    """Fast-lane priority scheduling: preempt a tagged request, reserved floor, aging."""
    def draw(topY, W):
        b = []
        top = topY + 10
        colw = 250
        # running batch before
        b.append(text(40, top, "Running batch (slot budget)", size=10, weight="bold", color="#111"))
        by = top + 12
        slots = [("normal", "kv"), ("normal", "kv"), ("normal", "kv"),
                 ("reserved", "free"), ("reserved", "free")]
        sw_ = 44
        for i, (kind, key) in enumerate(slots):
            xx = 40 + i * (sw_ + 6)
            b.append(rect(xx, by, sw_, 40, COLORS[key], stroke="#2b2b2b", sw=1.1, rx=2))
            b.append(text(xx + sw_ / 2, by + 18, "req" if kind == "normal" else "resv", size=8, anchor="middle",
                          color="#fff" if key == "kv" else "#1a1a1a"))
            if kind == "reserved":
                b.append(text(xx + sw_ / 2, by + 31, "heavy", size=7, anchor="middle", color="#1a1a1a"))
        b.append(text(40, by + 56, "reserved-heavy-slots: a floor kept for latency-priority work", size=8.6, color="#333"))
        # fast request arrives
        fy = by + 78
        b.append(rect(40, fy, 150, 28, "#c65b9b", stroke="#2b2b2b", sw=1.3, rx=3))
        b.append(text(115, fy + 18, "\"lane\":\"fast\" request", size=9, anchor="middle", color="#fff", weight="bold"))
        b.append(arrow(195, fy + 14, 300, fy + 14, "#c65b9b", marker="arr"))
        b.append(text(360, fy + 6, "preempts INTO the running batch immediately", size=9, color="#c65b9b"))
        b.append(text(360, fy + 19, "(takes a reserved-heavy slot; no queue wait)", size=8.6, color="#555"))
        # aging
        ay = fy + 44
        b.append(text(40, ay, "heavy-aging (--fast-lane-heavy-aging-ms): a long-waiting heavy request ages up in "
                      "priority so fast traffic cannot starve it.", size=9, color="#333"))
        # interaction with spill
        iy = ay + 24
        b.append(rect(40, iy, W - 80, 46, "#fdf3e3", stroke="#c88a2b", sw=1.2, rx=4))
        b.append(text(52, iy + 18, "Interaction with session KV spill (§20): fast-lane outranks FCFS — admitting a fast request "
                      "can spill a normal", size=9, color="#8a5a2b"))
        b.append(text(52, iy + 33, "session's KV to host to make room, and a spilled session's restore is held while a fast "
                      "request is still waiting.", size=9, color="#8a5a2b"))
        # default off
        b.append(text(40, iy + 68, "Default off (--enable-fast-lane opt-in): the default scheduling path is unchanged.",
                      size=9.3, color="#1d6b34"))
        return "".join(b), iy + 80
    compose("19-fast-lane-priority.svg", 1000,
            "19 — Fast-lane priority scheduling: preempt a tagged request into the batch",
            "Opt-in latency-priority class with a starvation guard. Decisive setting: --enable-fast-lane (§16, implemented).",
            [("A request tagged \"lane\":\"fast\" preempts into the running batch at once, taking one of a "
              "reserved floor of heavy slots; heavy-aging raises long-waiting heavy requests so fast traffic "
              "cannot starve them. It composes with session KV spill (a fast request can spill a normal "
              "session to host). Default off — the default path is unchanged. Upstream has priority scheduling "
              "the fork builds on; this reserved-floor fast-lane class is the fork addition.", "#333")],
            draw, ["kv", "free"])


def g20():
    """Measured VRAM budget: components measured, KV is the remainder; corridor rule."""
    def draw(topY, W):
        b = []
        top = topY + 20
        PX = 5.2
        # one rank card, component stack (measured) with KV as remainder
        x = 60
        cardw = 150
        total = 20.0   # a 3080 rank, illustrative absolute MiB budget
        b.append(text(x + cardw / 2, top - 22, "one rank (3080)", size=11, anchor="middle", weight="bold"))
        b.append(text(x + cardw / 2, top - 9, "--rank-gpu-memory-mib (absolute)", size=8.5, anchor="middle", color="#444"))
        b.append(rect(x, top, cardw, total * PX, "none", stroke="#222", sw=1.6))
        comps = [("ctx", 1.4, "CUDA context"),
                 ("weights", 2.4, "weight shard"),
                 ("resident", 3.4, "resident experts"),
                 ("mtp", 0.7, "solo-draft pool"),
                 ("state", 0.8, "GDN state"),
                 ("free", 1.0, "graphs / workspace")]
        cy = top
        last_leader_y = top - 20
        for key, gb, lab in comps:
            b.append(rect(x, cy, cardw, gb * PX, COLORS[key]))
            tc = "#fff" if key in WHITE_TEXT else "#1a1a1a"
            mid_y = cy + gb * PX / 2
            if gb * PX >= 13:
                b.append(text(x + cardw / 2, mid_y + 3, lab, size=8, anchor="middle", color=tc))
            else:
                ly = max(mid_y, last_leader_y + 12)   # stagger thin-band leader labels
                last_leader_y = ly
                b.append(line(x + cardw, mid_y, x + cardw + 8, ly, stroke="#888", sw=0.7))
                b.append(text(x + cardw + 11, ly + 3, lab, size=8, anchor="start", color="#333"))
            cy += gb * PX
        # KV = measured remainder
        kvh = top + total * PX - cy - 0.5 * PX
        b.append(rect(x, cy, cardw, kvh, COLORS["kv"]))
        b.append(text(x + cardw / 2, cy + kvh / 2 - 4, "KV cache", size=9, anchor="middle", color="#fff", weight="bold"))
        b.append(text(x + cardw / 2, cy + kvh / 2 + 9, "= measured remainder", size=7.6, anchor="middle", color="#fff"))
        # safety rest at the bottom
        sy = top + total * PX - 0.5 * PX
        b.append(rect(x, sy, cardw, 0.5 * PX, "#d3dae1", stroke="#888", sw=0.6))
        b.append(line(x + cardw, sy + 1, x + cardw + 8, sy + 1, stroke="#888", sw=0.7))
        b.append(text(x + cardw + 11, sy + 4, "safety rest ≥ 400 MiB", size=8, color="#333"))
        # right column explanation
        rx = x + cardw + 130
        n1, y1 = flow(rx, top + 6,
                      "Every component above is read from the measured component registry after boot + one "
                      "short request (so pools and CUDA graphs are really allocated) — no hand-guessed values. "
                      "KV cache is sized as what is LEFT after those measured components, within the absolute "
                      "per-rank --rank-gpu-memory-mib ceiling (an absolute MiB budget, not a fraction of total "
                      "or free VRAM).", W - rx - 30, size=9.4, color="#333")
        b.append(n1)
        n2, y2 = flow(rx, y1 + 12,
                      "Two-boot convergence: the boot logs a per-rank KV-split hint vector; feeding it back on "
                      "restart self-calibrates the split so the KV remainder lands on the safety rest.",
                      W - rx - 30, size=9.4, color="#333")
        b.append(n2)
        # corridor box
        cby = y2 + 14
        b.append(rect(rx, cby, W - rx - 30, 62, "#eef5ef", stroke="#1d6b34", sw=1.3, rx=4))
        b.append(text(rx + 12, cby + 17, "Corridor rule (evaluated per card, Option A):", size=9.3, weight="bold", color="#1d6b34"))
        b.append(text(rx + 12, cby + 33, "• fail if nvml_free < 400 MiB (absolute floor)", size=8.8, color="#1a1a1a"))
        b.append(text(rx + 12, cby + 48, "• fail if (nvml_free − measured transients) > 1.5 GiB (net waste)", size=8.8, color="#1a1a1a"))
        bottom = max(top + total * PX + 24, cby + 74)
        b.append(text(x, top + total * PX + 18, "upstream: fraction-based mem-fraction-static /", size=8.6, color="#555"))
        b.append(text(x, top + total * PX + 30, "gpu-memory-utilization; no per-rank absolute MiB budget.", size=8.6, color="#555"))
        return "".join(b), max(bottom, top + total * PX + 36)
    compose("20-measured-vram-budget.svg", 1040,
            "20 — Measured VRAM budget: components measured, KV is the remainder",
            "Per-rank absolute MiB budget from measured usage. Decisive setting: --rank-gpu-memory-mib + component registry (§10, implemented).",
            [("Each rank gets an absolute MiB budget (not a fraction). The per-component usage — CUDA context, "
              "weight shard, resident experts, solo-draft pool, GDN state, graph/workspace pools — is measured "
              "from a component registry after boot; the KV cache is sized as the measured remainder, and a "
              "logged split-hint vector converges over two boots. A corridor rule fails a card that has < 400 "
              "MiB free or > 1.5 GiB of net measured waste. Upstream sizes memory by a global fraction.", "#333")],
            draw, ["weights", "kv", "resident", "ctx", "mtp", "state", "free"])


if __name__ == "__main__":
    for fn in (s1, s2, s3, s4, s5, s6, s7, s8, s9,
               g10, g11, g12, g13, g14, g15, g16,
               g17, g18, g19, g20):
        fn()
    print("done")
