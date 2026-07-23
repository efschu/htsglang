#!/usr/bin/env python3
"""
Generator for the topology diagrams referenced by ../TOPOLOGIES.md.

Every diagram is a self-contained SVG: inline shapes + text only, no external
fonts, no external images, no <image href> to remote hosts.

The core of this document is, PER FORK FEATURE, ONE side-by-side VRAM/RAM
diagram:

  LEFT  — the fork feature on the reference rig (1x RTX 5090 32 GB + 2x RTX 3080
          20 GB, PCIe, no NVLink, no P2P), drawn GRANULARLY per card into the
          measured component segments (weights, KV, GDN state, experts, draft
          pools, graphs, CUDA context, free) plus a host-RAM bar.
  RIGHT — the hypothetical NORMAL / homogeneous UPSTREAM config for the same
          workload: N IDENTICAL cards, even TP = card count, with the divisibility
          reason for that TP. The whole right panel is illustrative / ESTIMATED.

Visual truth convention (used in every diagram, explained in the evidence key):
  * MEASURED           -> solid fill.
  * ESTIMATED          -> solid fill + diagonal hatch overlay + dashed outline.
  * UNKNOWN / not captured -> empty box, dashed grey outline, labelled "not captured".
The entire upstream (right) panel is ESTIMATED by construction (it is hypothetical
hardware), so every right-panel segment is hatched.

Numbers are taken only from the truth-checked data tables; where a segment was
never measured it is drawn as "not captured", never invented.
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
    # granular (per-layer / per-head) palette:
    "gdn":       "#7d5ba6",  # GDN / linear-attention layer (recurrent state)
    "fullattn":  "#2f6f8f",  # full-attention layer (holds KV cache)
    "state":     "#c9b8e0",  # GDN recurrent state (fixed size)
    "mtp":       "#c65b9b",  # MTP / NEXTN draft head / solo-draft pool
    "bad":       "#e9d7d7",  # upstream: region a split's constraints do not admit here
}
LEGEND = [
    ("weights",  "model-weight shard"),
    ("kv",       "KV cache (on-GPU)"),
    ("resident", "resident experts (MoE)"),
    ("scratch",  "expert scratch / prefetch"),
    ("spill",    "experts spilled to host RAM"),
    ("hostkv",   "host-staged / spilled KV"),
    ("free",     "free / reserve headroom"),
    ("ctx",      "CUDA context + overhead"),
    ("gdn",      "GDN / linear-attn layer"),
    ("fullattn", "full-attention layer (KV)"),
    ("state",    "GDN recurrent state"),
    ("mtp",      "MTP draft head / draft pool"),
    ("bad",      "upstream: not admitted here"),
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


# ---------------------------------------------------------------------------
# Evidence-aware primitives (the MEASURED / ESTIMATED / UNKNOWN convention).
# ---------------------------------------------------------------------------
def evseg(x, y, w, h, key, ev, label="", size=8.5, lead=None):
    """One evidence-tagged rectangle. ev in {measured, estimated, unknown}.

    lead: if a label does not fit inside a thin band, draw it to the right of
    x0 at height 'lead' with a leader line (used only in single-panel diagrams).
    """
    o = []
    if ev == "unknown":
        o.append(rect(x, y, w, h, "#f4f5f6", stroke="#9aa3ac", sw=1.0, dash="3 3"))
        tc = "#5a636c"
    else:
        o.append(rect(x, y, w, h, COLORS[key], stroke="#2b2b2b", sw=1.0,
                      dash="4 2" if ev == "estimated" else None))
        if ev == "estimated":
            o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                     f'fill="url(#hatch)" stroke="none"/>')
        tc = "#ffffff" if key in WHITE_TEXT else "#1a1a1a"
    if label:
        if h >= 14:
            o.append(text(x + w / 2, y + h / 2 + 3.2, label, size=size, anchor="middle", color=tc))
        elif h >= 9.5:
            o.append(text(x + w / 2, y + h / 2 + 2.6, label, size=min(size, 7.4), anchor="middle", color=tc))
    return "".join(o)


def stack_card(x, base_y, name, sub, vram, segs, PX, cardw=140, force_ev=None,
               free_label="free", emph=None):
    """A GPU card, bottom-aligned to base_y, height = vram*PX.

    segs: list of (key, gb, label, ev). ev in {measured, estimated, unknown}.
    force_ev: if set, overrides every segment's evidence (upstream panels -> estimated).
    emph: if set to a colour key, that segment is FOREGROUNDED (bold accent border)
      while the OTHER of {weights, kv} is de-emphasised (muted grey, kept to scale)
      so the reader sees which axis this diagram is about.
    Remainder up to vram is drawn as 'free'.
    """
    th = base_y - vram * PX
    nsz = 10.5 if cardw >= 105 else (9.2 if cardw >= 82 else 8.2)
    o = [text(x + cardw / 2, th - 19, name, size=nsz, anchor="middle", weight="bold"),
         text(x + cardw / 2, th - 7, sub, size=min(8.4, nsz - 1.6), anchor="middle", color="#555")]
    used = 0.0
    cy = th
    for key, gb, label, ev in segs:
        e = force_ev or ev
        h = gb * PX
        muted = emph is not None and key in ("weights", "kv") and key != emph
        if muted:
            # de-emphasised background segment (kept to scale, muted grey)
            o.append(rect(x, cy, cardw, h, "#c3cad0", stroke="#9aa3ac", sw=1.0,
                          dash="4 2" if e == "estimated" else None))
            if e == "estimated":
                o.append(f'<rect x="{x:.1f}" y="{cy:.1f}" width="{cardw:.1f}" '
                         f'height="{h:.1f}" fill="url(#hatch)" stroke="none"/>')
            if label and h >= 13:
                o.append(text(x + cardw / 2, cy + h / 2 + 3.2, label, size=8.2, anchor="middle", color="#5a636c"))
            elif label and h >= 9.5:
                o.append(text(x + cardw / 2, cy + h / 2 + 2.6, label, size=7.2, anchor="middle", color="#5a636c"))
        else:
            o.append(evseg(x, cy, cardw, h, key, e, label))
            if emph is not None and key == emph:
                o.append(rect(x, cy, cardw, h, "none", stroke="#12303a", sw=2.6))
        cy += h
        used += gb
    if used < vram - 0.03:
        fh = (vram - used) * PX
        fev = "estimated" if force_ev == "estimated" else "measured"
        o.append(evseg(x, cy, cardw, fh, "free", fev, free_label if fh >= 13 else ""))
    o.append(rect(x, th, cardw, vram * PX, "none", stroke="#222", sw=1.6))
    return "".join(o), th


def host_bar_ev(x, y, w, title, segs):
    """Horizontal host-RAM bar; segs = (key, frac, label, ev)."""
    barh = 32
    o = [text(x, y - 6, title, size=10, weight="bold", color="#333")]
    o.append(rect(x, y, w, barh, "none", stroke="#222", sw=1.4))
    cx = x
    for key, frac, label, ev in segs:
        rw = w * frac
        o.append(evseg(cx, y, rw, barh, key, ev, label if rw >= 55 else "", size=9))
        cx += rw
    return "".join(o), y + barh


def evidence_key(x, y):
    o = [text(x, y, "Evidence:", size=9.5, weight="bold", color="#333")]
    bx = x + 62
    o.append(rect(bx, y - 9, 13, 13, COLORS["kv"], stroke="#333", sw=0.8, rx=1))
    o.append(text(bx + 18, y + 1, "MEASURED (solid)", size=9, color="#333"))
    bx2 = bx + 150
    o.append(rect(bx2, y - 9, 13, 13, COLORS["kv"], stroke="#2b2b2b", sw=1.0, dash="4 2", rx=1))
    o.append(f'<rect x="{bx2}" y="{y-9}" width="13" height="13" fill="url(#hatch)" stroke="none"/>')
    o.append(text(bx2 + 18, y + 1, "ESTIMATED (hatched + dashed)", size=9, color="#333"))
    bx3 = bx2 + 232
    o.append(rect(bx3, y - 9, 13, 13, "#f4f5f6", stroke="#9aa3ac", sw=1.0, dash="3 3", rx=1))
    o.append(text(bx3 + 18, y + 1, "UNKNOWN — not captured (empty)", size=9, color="#333"))
    return "".join(o), y + 6


def core_note(x, y, w, s):
    txt, ny = flow(x + 14, y + 18, "Core — " + s, w - 28, size=10.4,
                   color="#12303a", weight="bold", lh=15)
    h = ny - y
    box = rect(x, y, w, h, "#edf4f6", stroke="#2f6f8f", sw=1.3, rx=6)
    return box + txt, y + h


def place(x0, x1, n, maxcw=140):
    """Evenly place n cards, centred, in [x0, x1]; returns (xs, cardw)."""
    gap = 14
    area = x1 - x0
    cw = min(maxcw, (area - (n - 1) * gap) / n)
    total = n * cw + (n - 1) * gap
    sx = x0 + (area - total) / 2
    return [sx + i * (cw + gap) for i in range(n)], cw


def gpu_column(x, top, name, vram_gb, regions, caption=None, sub=None):
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
        cy += rh
    if caption:
        out.append(text(x + BOXW / 2, top + h + 15, caption, size=9, anchor="middle", color="#333"))
    return "".join(out), top + h


def defs():
    """Shared arrowhead markers + the hatch pattern (self-contained)."""
    ms = [("arr", "#333"), ("arrP", "#8a5a2b"), ("arrG", "#227a3a"),
          ("arrR", "#c0504d"), ("arrB", "#3b6fb0"), ("arrV", "#7d5ba6")]
    s = "<defs>"
    s += ('<pattern id="hatch" patternUnits="userSpaceOnUse" width="7" height="7" '
          'patternTransform="rotate(45)">'
          '<line x1="0" y1="0" x2="0" y2="7" stroke="#ffffff" stroke-width="2" '
          'stroke-opacity="0.5"/></pattern>')
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
    w = len(label) * 6.9 + 22
    return (rect(x, y - 15, w, 21, fill, stroke=col, sw=1.3, rx=5)
            + text(x + 11, y, label, size=11, weight="bold", color=col))


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
    col_w = min(210, (width - 2 * x) / per_row)
    cy = y + 15
    for i, k in enumerate(keys):
        label = dict(LEGEND)[k]
        col, row = i % per_row, i // per_row
        px, py = x + col * col_w, cy + row * 16
        out.append(rect(px, py - 9, 12, 12, COLORS[k], stroke="#333", sw=0.8, rx=1))
        out.append(text(px + 17, py + 1, label, size=9, anchor="start", color="#333"))
    rows = (len(keys) + per_row - 1) // per_row
    return "".join(out), cy + rows * 16


def compose(name, W, title, subtitle, sentences, draw, legend_keys, evidence=False):
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
    topY = y + 30
    dbody, content_bottom = draw(topY, W)
    body.append(dbody)
    leg, legend_end = legend(20, content_bottom + 28, legend_keys, W)
    body.append(leg)
    end = legend_end
    if evidence:
        ek, end = evidence_key(20, legend_end + 16)
        body.append(ek)
    H = int(end + 18)
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'width="{W}" height="{H}" font-family="sans-serif">')
    bg = rect(0, 0, W, H, "#ffffff", stroke="none", sw=0, rx=0)
    svg = f'{head}{defs()}{bg}{"".join(body)}</svg>'
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(svg)
    print("wrote", name, f"({W}x{H})")


def cardbottom(topY):
    return topY + MAXVRAM * PXGB


# ===========================================================================
# The 10 per-feature SIDE-BY-SIDE VRAM/RAM diagrams (Parts 1-2, cool-first).
# Data comes only from the truth-checked tables; missing segments are drawn as
# "not captured". Left = fork on the rig; right = hypothetical homogeneous
# upstream config (all right-panel segments ESTIMATED).
# ===========================================================================
def feature(spec):
    W = spec.get("W", 1320)
    PX = spec.get("PX", 4.0)
    maxv = spec.get("maxv", 32.6)

    def draw(topY, W):
        b = []
        mid = W / 2
        top = topY + 60
        base = top + maxv * PX
        b.append(panel_label(30, topY - 2, spec["left_label"], "ours"))
        b.append(panel_label(mid + 16, topY - 2, spec["right_label"], "stock"))
        # left cards
        lcards = spec["left_cards"]
        lxs, lcw = place(40, mid - 26, len(lcards),
                         maxcw=spec.get("left_cw", 140))
        emph = spec.get("emph")
        for (nm, sub, vram, segs), x in zip(lcards, lxs):
            c, _ = stack_card(x, base, nm, sub, vram, segs, PX, cardw=lcw,
                              free_label=spec.get("left_free", "free"), emph=emph)
            b.append(c)
        # right cards (upstream, all estimated)
        rcards = spec["right_cards"]
        rxs, rcw = place(mid + 26, W - 30, len(rcards),
                         maxcw=spec.get("right_cw", 140))
        for (nm, sub, vram, segs), x in zip(rcards, rxs):
            c, _ = stack_card(x, base, nm, sub, vram, segs, PX, cardw=rcw,
                              force_ev="estimated",
                              free_label=spec.get("right_free", "free"), emph=emph)
            b.append(c)
        b.append(vdivider(mid, topY - 4, base + 18))
        yL = base + 30
        yR = base + 30
        # host bars
        if spec.get("host"):
            hb, yL = host_bar_ev(40, yL + 8, mid - 26 - 40, spec["host"][0], spec["host"][1])
            b.append(hb)
            yL += 6
        if spec.get("right_host"):
            hb, yR = host_bar_ev(mid + 26, yR + 8, W - 30 - (mid + 26),
                                 spec["right_host"][0], spec["right_host"][1])
            b.append(hb)
            yR += 6
        # notes
        for s, col in spec.get("left_notes", []):
            sv, yL = flow(40, yL + 13, s, mid - 26 - 40, size=9.3, color=col)
            b.append(sv)
        for s, col in spec.get("right_notes", []):
            sv, yR = flow(mid + 26, yR + 13, s, W - 30 - (mid + 26), size=9.3, color=col)
            b.append(sv)
        cy = max(yL, yR) + 14
        cn, cy = core_note(40, cy, W - 70, spec["core"])
        b.append(cn)
        return "".join(b), cy
    compose(spec["name"], W, spec["title"], spec["subtitle"], spec["sentences"],
            draw, spec["legend_keys"], evidence=True)


M = "measured"
E = "estimated"
U = "unknown"

FEATURES = [
    # -- 1. Uneven DCP -----------------------------------------------------
    {
        "name": "01-uneven-dcp.svg",
        "title": "1 — Uneven DCP = the KV / TOKEN axis: KV token-bands follow card VRAM, decoupled from the weight split",
        "subtitle": "Read this diagram for the KV bands (foreground); the weight shards are the constant grey background here. Fork on the rig vs a homogeneous upstream config for the same 27B workload.",
        "sentences": [
            ("Left (measured): FP8-27B TP=3 + uneven-DCP, --rank-gpu-id 0,1,2 --rank-tp-ratio auto "
             "--rank-gpu-memory-mib 28591,16464,16464. The MESSAGE is the green KV bands: KV is split along "
             "the TOKEN axis (374310 / 212109 / 212109 tokens) sized to each card's VRAM budget "
             "(28591:16464:16464 ≈ 1.74:1:1) — DECOUPLED from the weight ratio (12.7:8.0:8.0 ≈ 1.59:1:1, grey). "
             "Aggregate context grows with the cards you already own: 735k tokens, 2.81x the hand-budget start.", "#333"),
            ("This is a DIFFERENT axis from Uneven TP (§2): §2 sizes the WEIGHT shards to card compute; §1 sizes "
             "the KV TOKENS to card VRAM. On the rig both run together — --rank-tp-ratio auto sets the uneven "
             "TP weight split AND the uneven-DCP KV token split at once — so a rank's KV band is not tied to its "
             "weight shard.", "#1d6b34"),
            ("Right (illustrative): the natural homogeneous upstream config is 2 identical 24 GB cards at even "
             "TP=2 (4 KV heads split 2/rank). TP=3 is not legal here — 4 KV heads are not divisible by 3. "
             "Upstream head-shards KV, so aggregate KV does not grow with added cards the way the token split does.", "#555"),
        ],
        "left_label": "htsglang — uneven-DCP: KV / TOKEN axis (KV foregrounded)",
        "right_label": "upstream — 2x identical 24 GB, even TP=2",
        "emph": "kv",
        "left_cards": [
            ("RTX 5090", "rank0 · 32.6 GB", 32.6, [
                ("weights", 12.7, "weights (from TP split)", M),
                ("kv", 11.42, "KV 374k tok ← card VRAM", M),
                ("gdn", 0.55, "GDN", M),
                ("free", 0.41, "", M),  # graphs
                ("ctx", 0.80, "", M)]),
            ("RTX 3080", "rank1 · 20.5 GB", 20.5, [
                ("weights", 8.0, "weights (TP split)", M),
                ("kv", 6.48, "KV 212k tok", M),
                ("gdn", 0.47, "GDN", M),
                ("ctx", 0.78, "", M)]),
            ("RTX 3080", "rank2 · 20.5 GB", 20.5, [
                ("weights", 8.0, "weights (TP split)", M),
                ("kv", 6.48, "KV 212k tok", M),
                ("gdn", 0.47, "GDN", M),
                ("ctx", 0.78, "", M)]),
        ],
        "right_cards": [
            ("identical 24 GB", "rank0 · even TP=2", 24, [
                ("weights", 12.5, "weights ½", E),
                ("kv", 8.5, "KV head-shard (2 of 4)", E),
                ("ctx", 1.5, "", E)]),
            ("identical 24 GB", "rank1 · even TP=2", 24, [
                ("weights", 12.5, "weights ½", E),
                ("kv", 8.5, "KV head-shard (2 of 4)", E),
                ("ctx", 1.5, "", E)]),
        ],
        "host": ("Host RAM (DDR) — no spill in this mode",
                 [("free", 1.0, "not captured (low, no host tier)", U)]),
        "left_notes": [
            ("KV bands (green, foreground) + weights/GDN/ctx are MEASURED per rank; the weight shards are drawn "
             "GREY here because in this diagram they are the constant background, not the point. The KV token "
             "ratio tracks the MiB budget (1.74:1:1), NOT the weight ratio (1.59:1:1) — that gap is the "
             "decoupling. Exact host-RAM floor for this non-spill config was not captured.", "#1d6b34"),
        ],
        "right_notes": [
            ("Head-axis KV stores every token per head on each card, so aggregate KV does not grow "
             "with added cards the way the token-axis split does. Assumed card size named; not measured.", "#4a5568"),
        ],
        "core": ("the fork splits KV along the TOKEN axis, sized to each card's VRAM and DECOUPLED from the weight "
                 "split, so aggregate context scales with the mismatched cards you already own; upstream even-TP "
                 "head-shards KV and needs identical cards whose KV-head count divides the rank count. (Contrast "
                 "§2, which is about the WEIGHT axis; on the rig both run together.)"),
        "legend_keys": ["weights", "kv", "gdn", "ctx", "free"],
    },
    # -- 2. Uneven TP ------------------------------------------------------
    {
        "name": "02-uneven-tp.svg",
        "title": "2 — Uneven TP = the WEIGHT axis: size each weight shard to the card's compute (ratio 2:1:1, Q-heads 12/6/6)",
        "subtitle": "Read this diagram for the weight shards (foreground); the KV here is just the leftover remainder (grey). Fork on the rig vs a homogeneous upstream even-TP config for the same 27B workload.",
        "sentences": [
            ("Left (measured): 27B TP=3, --rank-tp-ratio 2,1,1 --rank-gpu-memory-mib 26000,15000,15000, one "
             "rank per GPU. The MESSAGE is the blue weight shards: Q heads split 12 / 6 / 6, the 5090 carries "
             "the 2x shard — measured weight shards 12.7 / 8.0 / 8.0 GiB (ratio 2 : 1 : 1, sized to card "
             "COMPUTE). KV here is only the leftover remainder (grey), not the point.", "#333"),
            ("This is a DIFFERENT axis from Uneven DCP (§1): §2 sizes the WEIGHT shards to card compute; §1 "
             "sizes the KV TOKENS to card VRAM. On the rig both run together — --rank-tp-ratio auto sets the "
             "weight split AND the KV token split at once — so do not read the two diagrams as the same thing.", "#1d6b34"),
            ("Right (illustrative): upstream even-TP gives every rank an IDENTICAL shard, so it wants N equal "
             "cards (2x 24 GB shown). On mixed cards it would size every rank to the smallest and strand the "
             "surplus of the larger one. TP=3 is illegal for this model (4 KV heads).", "#555"),
        ],
        "left_label": "htsglang — uneven-TP: WEIGHT axis, ratio 2:1:1 (weights foregrounded)",
        "right_label": "upstream — 2x identical 24 GB, even TP=2",
        "emph": "weights",
        "left_cards": [
            ("RTX 5090", "rank0 · ratio 2 · Q 12", 32.6, [
                ("weights", 12.7, "weight shard 2x (Q 12/24)", M),
                ("gdn", 0.55, "GDN", M),
                ("ctx", 0.80, "", M),
                ("kv", 14.5, "KV (just the remainder)", E)]),
            ("RTX 3080", "rank1 · ratio 1 · Q 6", 20.5, [
                ("weights", 8.0, "weight shard (Q 6/24)", M),
                ("gdn", 0.47, "GDN", M),
                ("ctx", 0.78, "", M),
                ("kv", 8.8, "KV (remainder)", E)]),
            ("RTX 3080", "rank2 · ratio 1 · Q 6", 20.5, [
                ("weights", 8.0, "weight shard (Q 6/24)", M),
                ("gdn", 0.47, "GDN", M),
                ("ctx", 0.78, "", M),
                ("kv", 8.8, "KV (remainder)", E)]),
        ],
        "right_cards": [
            ("identical 24 GB", "rank0 · even shard", 24, [
                ("weights", 12.5, "weights (equal)", E),
                ("kv", 9.5, "KV", E),
                ("ctx", 1.5, "", E)]),
            ("identical 24 GB", "rank1 · even shard", 24, [
                ("weights", 12.5, "weights (equal)", E),
                ("kv", 9.5, "KV", E),
                ("ctx", 1.5, "", E)]),
        ],
        "left_notes": [
            ("Weight shards (blue, foreground) + GDN + ctx are MEASURED; the weight ratio 2:1:1 (Q-heads 12/6/6) "
             "is the message — shards sized to card COMPUTE. KV is drawn GREY because here it is only the "
             "measured remainder (exact per-rank GiB not separately dumped -> ESTIMATED). Earlier 68 / 97 tok/s "
             "figures used a contaminated bench (pre-2026-07-22) and are withdrawn, not shown as fact.", "#1d6b34"),
        ],
        "right_notes": [
            ("Even-TP is clean ON identical cards; the contrast is the hardware premise (N equal cards), "
             "not the per-card layout. Assumed card size named; not measured.", "#4a5568"),
        ],
        "core": ("the fork sizes each WEIGHT shard to the specific card's compute (ratio 2:1:1), using a 32 GB + "
                 "2x20 GB set as-is; upstream even-TP gives identical shards and therefore wants N equal cards "
                 "(and strands a bigger card if mixed). (Contrast §1, which is about the KV/TOKEN axis; on the "
                 "rig both run together.)"),
        "legend_keys": ["weights", "kv", "gdn", "ctx", "free"],
    },
    # -- 3. Adaptive drafter routing --------------------------------------
    {
        "name": "03-adaptive-drafter.svg",
        "title": "3 — Adaptive drafter routing (NEXTN ↔ DFLASH): dual residence, itemised per-rung cost",
        "subtitle": "Fork on the rig (TP=3 uneven) vs a homogeneous upstream TP=2 running one fixed drafter.",
        "sentences": [
            ("Left (measured): both drafters resident behind --speculative-cross-algorithm. The solo-draft "
             "pool costs 4.58 GiB on rank0 (5090); the per-k rung graphs/state are itemised "
             "DFLASH_k16 662 / EAGLE_k3 634 / EAGLE_k2 554 MiB. Draft graphs push rank0 graphs to 7.44 GiB.", "#333"),
            ("Right (illustrative): the homogeneous upstream reference is 2 identical 24 GB cards at even "
             "TP=2 running ONE FIXED drafter (e.g. NEXTN k=3) — a single draft residence, no per-k rung "
             "tags, no dual-residence and no runtime routing (upstream cannot switch draft algorithm). Less "
             "draft VRAM, but no runtime choice.", "#555"),
        ],
        "left_label": "htsglang — TP=3 uneven, cross-algo drafter resident (measured)",
        "right_label": "upstream — 2x identical 24 GB, even TP=2, one fixed drafter",
        "left_cw": 132,
        "right_cw": 150,
        "left_cards": [
            ("RTX 5090", "rank0 · 28.5 GB used", 32.6, [
                ("weights", 13.09, "weights", M),
                ("kv", 1.37, "KV pool", M),
                ("gdn", 1.96, "GDN state", M),
                ("mtp", 4.58, "solo-draft pool", M),
                ("free", 7.44, "graphs incl draft", M)]),
            ("RTX 3080", "rank1 · 18.6 GB", 20.5, [
                ("weights", 8.03, "weights", M),
                ("kv", 6.41, "KV pool", M),
                ("gdn", 1.43, "GDN", M),
                ("free", 2.48, "graphs", M)]),
            ("RTX 3080", "rank2 · 18.4 GB", 20.5, [
                ("weights", 7.68, "weights", M),
                ("kv", 6.87, "KV pool", M),
                ("gdn", 1.13, "GDN", M),
                ("free", 2.48, "graphs", M)]),
        ],
        "right_cards": [
            ("identical 24 GB", "rank0 · even TP=2", 24, [
                ("weights", 12.5, "weights ½", E),
                ("mtp", 0.5, "1 fixed drafter (NEXTN k=3)", E),
                ("kv", 9.0, "KV", E),
                ("ctx", 1.3, "", E)]),
            ("identical 24 GB", "rank1 · even TP=2", 24, [
                ("weights", 12.5, "weights ½", E),
                ("kv", 9.5, "KV", E),
                ("ctx", 1.3, "", E)]),
        ],
        "host": ("Host RAM (DDR) — solo-draft placement",
                 [("hostkv", 0.28, "~2–3 GB draft (embed/lm_head + draft KV)", M),
                  ("free", 0.72, "", M)]),
        "left_notes": [
            ("Rung tags itemised (MEASURED): DFLASH_k16 662 · EAGLE_k3 634 · EAGLE_k2 554 MiB. "
             "KV budget cost of the cross-gate: ~282k vs ~524k tokens without it (MEASURED).", "#1d6b34"),
            ("Honest claim: robustness / no-regret on mixed streams, NOT a peak speedup — switching costs "
             "~+5.7% systemic vs a single static drafter (MEASURED).", "#333"),
        ],
        "right_notes": [
            ("One fixed drafter, single residence — no cross-algo pool, no rung tags, no runtime routing. "
             "Assumed 24 GB card; whole upstream side estimated (not measured).", "#4a5568"),
        ],
        "core": ("the fork holds BOTH drafters resident plus the routing machinery (solo-draft pool 4.58 GiB + "
                 "per-k rung tags) — more draft VRAM, but runtime-adaptive; homogeneous upstream TP=2 holds ONE "
                 "fixed drafter — less draft VRAM, but no runtime choice. A capability / VRAM trade-off, not a "
                 "verdict (upstream is not worse, just fixed)."),
        "legend_keys": ["weights", "kv", "gdn", "mtp", "hostkv", "free"],
    },
    # -- 4. Session KV spill ----------------------------------------------
    {
        "name": "04-session-kv-spill.svg",
        "title": "4 — Session KV spill: overflow the newest session to host RAM and keep decoding",
        "subtitle": "Fork on the rig vs upstream behaviour under KV pressure (any homogeneous TP).",
        "sentences": [
            ("Left (measured S1): on device-KV overflow the NEWEST session's full-attention KV shard is "
             "pushed to host and keeps decoding (block-LSE attention, eager bs=1 tick). GDN/Mamba state "
             "always stays resident. Zero-overhead when unused +0.16%; host decode 8.1 tok/s @1k ctx; "
             "restore ~0.4 s; determinism 50/50 exact host-vs-device.", "#333"),
            ("Right (illustrative): upstream has no per-session host-KV decode — under pressure it "
             "retracts/recomputes or pauses the request. The device KV pool is a hard ceiling; there is no "
             "host KV tier to overflow into.", "#555"),
        ],
        "left_label": "htsglang — device KV + host overflow tier (S1, measured)",
        "right_label": "upstream — device KV is the hard ceiling",
        "left_cards": [
            ("RTX 5090", "device · rank0", 32.6, [
                ("weights", 12.7, "weights", M),
                ("gdn", 0.55, "GDN (always resident)", M),
                ("kv", 11.0, "resident session KV", M),
                ("free", 4.0, "band freed by spill", M)]),
        ],
        "right_cards": [
            ("identical card", "device · KV ceiling", 24, [
                ("weights", 12.5, "weights", E),
                ("gdn", 0.5, "state", E),
                ("kv", 9.0, "session KV (hard ceiling)", E),
                ("bad", 1.0, "overflow → pause/recompute", E)]),
        ],
        "host": ("Host RAM (DDR) — spilled-session KV tier",
                 [("hostkv", 0.32, "spilled session KV (32 KiB/tok x tokens, est)", E),
                  ("free", 0.68, "host KV tier", M)]),
        "right_host": ("Host RAM — no host KV decode path",
                       [("free", 1.0, "not captured (no host tier)", U)]),
        "left_notes": [
            ("Host KV bytes = 32 KiB/token x spilled tokens (ESTIMATED from the measured cell size). "
             "Long-context curve (32k~63 ... 262k~7.6 tok/s) is MODELED, not benchmarked (needs S2); "
             "worthwhile only with uneven DCP active.", "#1d6b34"),
            ("The more important number — how the DEVICE-RESIDENT (non-spilled) session runs DURING a spill "
             "(measured, ctx ~1.6k): 10.4 / 13.4 / 19.9 / 26.0 tok/s at tick-interval 1 / 2 / 4 / 8 (pre-spill "
             "~40); the spilled session ~7–8 tok/s. Isolation target met only from tick 4 up, VIOLATED at tick "
             "1/2 — the eager spill tick still blocks the shared cadence (open item; see mechanism diagram 12).", "#333"),
        ],
        "right_notes": [
            ("Behaviour, not a VRAM split: the request is paused, not decoded from host.", "#4a5568"),
        ],
        "core": ("the fork lets an overflowing session keep decoding from system RAM instead of being "
                 "paused/rejected — a capacity/behaviour difference (device-KV ceiling vs host-tier overflow), "
                 "not a speed claim."),
        "legend_keys": ["weights", "kv", "gdn", "hostkv", "bad", "free"],
    },
    # -- 5. Multi-rank co-location (TP=5) ---------------------------------
    {
        "name": "05-tp5-colocation.svg",
        "title": "5 — Multi-rank co-location: run MORE TP-ranks than physical GPUs by sharing a GPU via MPS",
        "subtitle": "TP=5 is a standard sglang capability; the fork contribution is co-locating ranks so it runs on 3 cards, not 5.",
        "sentences": [
            ("Attribution: TP=5 is a STANDARD sglang capability (any TP degree — normally 5 physical cards, one "
             "rank each). That is NOT the fork's feature. The fork contribution is MULTI-RANK CO-LOCATION — "
             "running more TP-ranks than physical GPUs by letting several ranks share a GPU via MPS (+ NCCL "
             ">= 2.30 for the co-located communicator).", "#333"),
            ("Left (measured budgets): with co-location, a standard TP=5 config was TESTED on just 3 physical "
             "cards — --tp 5 --rank-gpu-id 0,0,0,1,2 --rank-auto-reserve-mib 11500,11500,11500,3500,3500, MPS on: "
             "three ranks time-slice the 5090 (~7 GB budget each) + one rank per 3080. This EMULATES TP=5, it is "
             "NOT a 5-card perf equivalent (decode tok/s deliberately not 5-card-representative). Per-rank "
             "weight/KV/GDN split inside each budget was not dumped.", "#333"),
            ("Right (illustrative): a standard TP=5 needs 5 physical identical cards, one rank each; this 3-card "
             "box cannot express TP=5 without co-location. That 5-cards-vs-3-cards is the honest contrast.", "#555"),
        ],
        "left_label": "htsglang — co-location: standard TP=5 emulated on 3 cards via MPS",
        "right_label": "upstream — standard TP=5 = 5x identical cards, 1 rank each",
        "left_cw": 140,
        "right_cw": 96,
        "left_cards": [
            ("RTX 5090", "3 ranks · MPS", 32.6, [
                ("free", 7.0, "rank0 budget ~7 GB — split not captured", U),
                ("free", 7.0, "rank1 budget ~7 GB — not captured", U),
                ("free", 7.0, "rank2 budget ~7 GB — not captured", U),
                ("ctx", 2.5, "MPS/ctx", E)]),
            ("RTX 3080", "rank3", 20.5, [
                ("free", 17.0, "rank3 budget ~17 GB — split not captured", U)]),
            ("RTX 3080", "rank4", 20.5, [
                ("free", 17.0, "rank4 budget ~17 GB — split not captured", U)]),
        ],
        "right_cards": [
            ("identical", f"rank{i}", 16, [("weights", 3.0, "shard", E), ("kv", 9.0, "KV", E), ("ctx", 1.2, "", E)])
            for i in range(5)
        ],
        "host": ("Host RAM (DDR) — GGUF file cache + MPS",
                 [("free", 1.0, "not captured", U)]),
        "left_notes": [
            ("Budgets are MEASURED (7/7/7 on the 5090, 17/17 on the 3080s); the weight/KV/GDN breakdown "
             "within each rank was not registry-dumped. Coherent, needle from ~15k ctx, bit-identical "
             "across two boots. Decode tok/s is deliberately NOT 5-card-representative (3 ranks share one card).", "#1d6b34"),
            ("Fork delta beyond co-location: the uneven-TP + kv-boundary-aware auto-split (#116) lets a "
             "co-located UNEVEN TP=5 boot even when num_kv_heads < tp (it constrains the per-rank Q-head split "
             "to whole KV-head groups, fixing the #105 Q-split straddle).", "#1d6b34"),
            ("Models are dense-27B-GGUF and 35B-A3B-GGUF. The GGUF quant itself (format + K-quant / MMQ / "
             "MMVQ kernels) is ggml/llama.cpp via upstream; the fork delta is only the uneven-TP adaptation "
             "(256-superblock alignment, MLP coarsening, MMQ-OOB fix under expert sharding, Qwen3.5/3.6+Gemma-4 "
             "arch adapters) and the MMVQ↔MMQ crossover tuning.", "#333"),
        ],
        "right_notes": [
            ("Standard TP=5 = 5 equal cards (one rank each); the honest contrast to co-location on 3 cards. "
             "Illustrative sizes.", "#4a5568"),
        ],
        "core": ("TP=5 is standard sglang (normally 5 physical cards); the fork contribution is co-locating ranks "
                 "so several share one GPU via MPS — here EMULATING/testing a TP=5 config on just 3 cards — plus "
                 "the uneven + kv-boundary-aware split that lets a co-located uneven TP=5 boot. A capacity / "
                 "emulation / testability difference, not a throughput advantage and not a claim to have invented TP=5."),
        "legend_keys": ["weights", "kv", "ctx", "free"],
    },
    # -- 6. Weightless-KV lane --------------------------------------------
    {
        "name": "06-weightless-kv-lane.svg",
        "title": "6 — Weightless-KV lane: free the workers of weights so their freed VRAM becomes device KV",
        "subtitle": "Fork on the rig (head holds all weights, workers become device-KV donors) vs upstream, where every rank splits VRAM between weights and KV.",
        "sentences": [
            ("Left (measured, PRIMARY): --weightless-kv-fastlane, TP=3 + DCP. rank0 (5090) is the HEAD — it holds "
             "ALL layer weights (TP=1). rank1/2 (3080) are WEIGHTLESS meta-device workers with ZERO layer weights "
             "(~14 GiB freed EACH, MEASURED); that freed VRAM instead carries DEVICE KV — KV token-shards + "
             "KV-heads — and the workers compute the attention over it. The slow cards become on-device KV "
             "DONORS: pooled device KV across the freed workers is the capacity win.", "#333"),
            ("Secondary (extreme context only): an OPTIONAL host-KV tier sits ON TOP, used only for context that "
             "exceeds the pooled device KV — up to 262k tokens proven (~12.6 GiB pinned host, #134 B1/B2, "
             "needle-at-midpoint retrieved). It is an extension, not the primary store; the 262k extreme config "
             "deliberately shrinks the device pool and pushes most KV to host (40000 device / 64000 host slots), "
             "whereas in the normal case the freed worker VRAM holds the device KV.", "#8a5a2b"),
            ("Right (illustrative): a homogeneous upstream even-TP holds the layer weights on EVERY rank, so each "
             "identical card splits its VRAM between weights and KV — there are no weightless workers, so per-card "
             "context is bounded without more/bigger equal cards.", "#555"),
        ],
        "left_label": "htsglang — head holds weights, workers become device-KV donors (measured)",
        "right_label": "upstream — N identical cards, weights on every rank",
        "left_cards": [
            ("RTX 5090", "HEAD · 22.8 GB used", 32.6, [
                ("weights", 17.0, "ALL layer weights (TP=1)", M),
                ("kv", 4.0, "head KV", E),
                ("ctx", 1.8, "", E)]),
            ("RTX 3080", "weightless worker", 20.5, [
                ("weights", 0.3, "0 layer weights (meta-device)", M),
                ("kv", 16.0, "device KV (freed VRAM)", M),
                ("ctx", 0.7, "", E)]),
            ("RTX 3080", "weightless worker", 20.5, [
                ("weights", 0.3, "0 layer weights (meta-device)", M),
                ("kv", 16.0, "device KV (freed VRAM)", M),
                ("ctx", 0.7, "", E)]),
        ],
        "right_cards": [
            ("identical 24 GB", "rank0 · even TP", 24, [
                ("weights", 12.5, "weights (every rank)", E),
                ("kv", 9.5, "KV", E),
                ("ctx", 1.5, "", E)]),
            ("identical 24 GB", "rank1 · even TP", 24, [
                ("weights", 12.5, "weights (every rank)", E),
                ("kv", 9.5, "KV", E),
                ("ctx", 1.5, "", E)]),
        ],
        "host": ("Host RAM (DDR) — OPTIONAL extreme-context tier (only beyond pooled device KV)",
                 [("hostkv", 0.42, "~12.6 GB pinned — beyond device pool → 262k (#134)", M),
                  ("free", 0.58, "", M)]),
        "left_notes": [
            ("PRIMARY win = pooled DEVICE KV on the freed workers (0 layer weights, ~14 GiB freed each, MEASURED; "
             "head 22.8 / worker 3.7 GiB VRAM in the 262k run). The host tier is a SECONDARY extension for context "
             "beyond that; it stages only ~C/dcp_size per rank (3.5x host-RAM saving), no unverified context "
             "multiplier claimed.", "#1d6b34"),
            ("Throughput is interconnect-bound, NOT a \"fast\" claim: ~25 tok/s @8k · ~7 @28k · ~1.5 @262k eager "
             "(graph+prefetch raises exact-rung to ~26-29). After #136a/#136b the PCIe wall is mostly hidden, so "
             "the deep-context floor is now block-attention compute + per-layer collectives on the slow workers, "
             "not H2D bandwidth.", "#333"),
        ],
        "right_notes": [
            ("Weights compete with KV on every identical card; no weightless workers. Illustrative sizes.", "#4a5568"),
        ],
        "core": ("the fork frees the worker cards of ALL weights (~14 GiB each) so their VRAM becomes pooled DEVICE "
                 "KV — the slow cards turn into KV donors that also compute attention; an OPTIONAL host tier extends "
                 "context to a proven 262k only beyond the pooled device KV. Upstream spends every identical card's "
                 "VRAM on both weights and KV. A capacity feature — throughput is interconnect-bound on this "
                 "no-NVLink rig, not a speed win."),
        "legend_keys": ["weights", "kv", "hostkv", "ctx", "free"],
    },
    # -- 7. MoE expert offload --------------------------------------------
    {
        "name": "07-moe-expert-offload.svg",
        "title": "7 — MoE expert offload: run a 122B on 3 mismatched cards by spilling cold experts to host RAM",
        "subtitle": "Fork on the rig vs a realistic homogeneous upstream config that also offloads weights to host RAM.",
        "sentences": [
            ("Left (measured): Qwen3.5-122B-A10B-GPTQ-Int4, TP=3, --rank-tp-ratio auto, "
             "SGLANG_MOE_RESIDENT_EXPERT_FRACTION=0.25. Per-rank GPU 25.5 / 16.6 / 15.4 GiB; 64 resident + 16 "
             "scratch experts/layer stay on-GPU, 176/layer spill to pinned host RAM (host floor 24.4 GiB). "
             "Throughput 6.97 (eager) → 10.61 (graph) → 16.34 tok/s (graph+hotset).", "#333"),
            ("Right (illustrative): upstream runs the same 122B on a realistic homogeneous config too — 2 "
             "identical RTX 3090 (24 GB each, even TP=2) plus --cpu-offload-gb, keeping part of the weights "
             "on-device and the rest in system RAM. The whole right panel is estimated (stock cpu-offload was "
             "not benched here).", "#555"),
        ],
        "left_label": "htsglang — 122B TP=3, expert offload f=0.25 (measured)",
        "right_label": "upstream — 2x RTX 3090 24 GB, TP=2 + --cpu-offload-gb",
        "left_cw": 130,
        "right_cw": 150,
        "left_cards": [
            ("RTX 5090", "rank0 · 25.5 GB", 32.6, [
                ("weights", 3.0, "shard", M),
                ("resident", 16.0, "64+16 experts/layer x48", M),
                ("kv", 5.0, "KV band", M),
                ("mtp", 1.0, "MTP", M)]),
            ("RTX 3080", "rank1 · 16.6 GB", 20.5, [
                ("weights", 2.2, "shard", M),
                ("resident", 11.0, "experts/layer", M),
                ("kv", 3.4, "KV band", M)]),
            ("RTX 3080", "rank2 · 15.4 GB", 20.5, [
                ("weights", 2.2, "shard", M),
                ("resident", 10.0, "experts/layer", M),
                ("kv", 3.2, "KV band", M)]),
        ],
        "right_cards": [
            ("RTX 3090", "rank0 · TP=2 + offload", 24, [
                ("weights", 14.0, "on-device layer-group weights", E),
                ("kv", 7.5, "KV", E),
                ("ctx", 2.0, "", E)]),
            ("RTX 3090", "rank1 · TP=2 + offload", 24, [
                ("weights", 14.0, "on-device layer-group weights", E),
                ("kv", 7.5, "KV", E),
                ("ctx", 2.0, "", E)]),
        ],
        "host": ("Host RAM (DDR) — spilled experts (measured floor)",
                 [("spill", 0.30, "176 experts/layer x48 (pinned) — floor 24.4 GB", M),
                  ("free", 0.70, "of 108 GB (no swap)", M)]),
        "right_host": ("Host RAM (DDR) — --cpu-offload-gb: offloaded weights",
                       [("weights", 0.55, "offloaded layer-group weights — streamed EVERY forward", E),
                        ("free", 0.45, "reserved (--cpu-offload-gb N)", E)]),
        "left_notes": [
            ("Per-rank GPU + host floor MEASURED. Not bit-identical to no-offload (marlin ~1e-2 argmax at "
             "near-ties); bar is coherence + self-determinism (5/5).", "#1d6b34"),
        ],
        "right_notes": [
            ("--cpu-offload-gb (server_args.py: \"How many GBs of RAM to reserve for CPU offloading\") is a "
             "GENERIC per-layer-weight offload: the offloaded weights are streamed back in EVERY forward, "
             "regardless of which experts a token routes. Assumed 2x 3090; stock cpu-offload not benched here.", "#4a5568"),
        ],
        "core": ("both run a 122B on modest cards by using host RAM; the mechanism differs — the fork offload is "
                 "EXPERT-GRANULAR (per token-wave it fetches only the routed top-K experts from host), whereas "
                 "--cpu-offload-gb streams a fixed offloaded weight fraction back on every forward. At the same "
                 "VRAM budget the fork moves less data per token; upstream moves the full offloaded fraction per "
                 "forward — a capability / data-volume difference, stated neutrally."),
        "legend_keys": ["weights", "resident", "kv", "mtp", "spill", "free"],
    },
    # -- 8. Measured VRAM budget ------------------------------------------
    {
        "name": "08-measured-vram-budget.svg",
        "title": "8 — Measured VRAM budget: an absolute per-rank MiB budget with a per-segment registry",
        "subtitle": "Fork on the rig vs upstream's single global fraction across identical cards.",
        "sentences": [
            ("Left (measured): --rank-gpu-memory-mib gives each rank an ABSOLUTE MiB budget (not a fraction). "
             "Every component — weights, KV pool, GDN state, draft pool, graphs, CUDA context, fragmentation, "
             "required-free — is read from a measured registry after boot + one warm request; KV is the "
             "measured remainder. Two-boot self-calibration via a logged split-hint vector.", "#333"),
            ("Right (illustrative): upstream sizes memory by ONE global fraction (mem-fraction-static / "
             "gpu-memory-utilization, e.g. 0.806) applied uniformly, with no per-rank absolute MiB budget and "
             "no measured per-segment registry. On identical cards a single fraction is a natural fit.", "#555"),
        ],
        "left_label": "htsglang — measured per-rank registry (rank0, no-spec boot)",
        "right_label": "upstream — global fraction 0.806",
        "left_cw": 150,
        "right_cw": 150,
        "left_cards": [
            ("RTX 5090", "rank0 · absolute MiB budget", 32.6, [
                ("ctx", 0.80, "CUDA ctx", M),
                ("weights", 12.70, "param+buffer weights", M),
                ("gdn", 0.55, "GDN/aux", M),
                ("free", 0.41, "graphs/ws", M),
                ("kv", 11.5, "KV = measured remainder", M),
                ("free", 0.11, "frag", M),
                ("free", 3.00, "required-free (safety)", M)]),
        ],
        "right_cards": [
            ("identical card", "0.806 fraction", 24, [
                ("free", 0.6, "reserved (1 - fraction)", E),
                ("weights", 12.5, "weights", E),
                ("kv", 9.0, "KV (within fraction)", E)]),
        ],
        "left_notes": [
            ("Entire registry MEASURED per rank. Corridor rule (Option A): fail a card if nvml_free < 400 MiB "
             "(floor) or nvml_free − measured transients > 1536 MiB (net waste). Self-calibration vector across "
             "boots: C 215488 → 282560 → 484160 → 524160.", "#1d6b34"),
        ],
        "right_notes": [
            ("One fraction, no per-segment readout. Natural on identical cards. Not measured.", "#4a5568"),
        ],
        "core": ("the fork gives each rank an absolute measured MiB budget with a per-segment registry and a "
                 "corridor rule; upstream uses one global fraction across identical cards — an "
                 "observability / capability difference, not a speed claim."),
        "legend_keys": ["weights", "kv", "gdn", "ctx", "free"],
    },
    # -- 9. Fast-lane priority scheduling ---------------------------------
    {
        "name": "09-fast-lane.svg",
        "title": "9 — Fast-lane scheduling: a fairness / anti-starvation layer on the priority path",
        "subtitle": "Fork on the rig vs upstream's continuous-priority scheduling. A scheduling behaviour, not a hardware-capacity split.",
        "sentences": [
            ("Left: --enable-fast-lane adds a BINARY two-tier lane on top of the priority path (the \"lane\":\"fast\" "
             "tag sets a fixed high fast_lane_priority, not a manual integer). Its delta over generic priority is "
             "two anti-starvation guarantees: (1) RESERVED HEAVY SLOTS (--fast-lane-reserved-heavy-slots) — at "
             "least N normal (\"heavy\") requests are never preempted below the reserved floor, so sustained fast "
             "load cannot fully starve normal requests; (2) HEAVY AGING (--fast-lane-heavy-aging-ms) — a normal "
             "request waiting past the window is promoted ahead of the fast tier, so a stream of fast requests "
             "cannot block a waiting normal one indefinitely. Default OFF.", "#333"),
            ("Right (illustrative): upstream priority scheduling sorts the waiting queue by a continuous integer "
             "priority and preempts a running request when priority_diff exceeds "
             "priority_scheduling_preemption_threshold — a general mechanism, with no reserved floor for the "
             "preempted and no aging. No distinct VRAM segment beyond the normal slot pool.", "#555"),
        ],
        "left_label": "htsglang — reserved floor + heavy-aging on the priority path",
        "right_label": "upstream — continuous-priority + preemption threshold",
        "left_cw": 150,
        "right_cw": 150,
        "left_cards": [
            ("RTX 5090", "rank0 · slot/KV pool", 32.6, [
                ("weights", 12.7, "weights", M),
                ("gdn", 0.55, "GDN", M),
                ("kv", 11.0, "normal KV slot pool", M),
                ("free", 2.5, "reserved heavy-slot floor (bytes est)", E)]),
        ],
        "right_cards": [
            ("identical card", "normal slot pool", 24, [
                ("weights", 12.5, "weights", E),
                ("kv", 9.0, "KV slot pool (no reserved class)", E),
                ("ctx", 1.2, "", E)]),
        ],
        "left_notes": [
            ("Integration with session KV spill: a fast request can spill a normal session's KV to host RATHER "
             "than queue, and a fast request is never itself spilled. The reserved-floor byte cost was not "
             "registry-dumped — drawn ESTIMATED; the guarantees are behavioural (see mechanism diagram 13).", "#1d6b34"),
        ],
        "right_notes": [
            ("enable_priority_scheduling + priority_scheduling_preemption_threshold verified in "
             "schedule_policy.py. No reserved floor, no aging, no spill coupling. Not measured.", "#4a5568"),
        ],
        "core": ("upstream supplies the priority axis (continuous integer priority + a preemption threshold); the "
                 "fork's fast-lane adds a fairness / anti-starvation layer on top — a reserved floor of heavy slots "
                 "and heavy-aging that guarantee progress for preempted / waiting normal requests — plus coupling to "
                 "session KV spill. Stated as what each side does, not a ranking."),
        "legend_keys": ["weights", "kv", "gdn", "free"],
    },
    # -- 10. PD-disaggregation --------------------------------------------
    {
        "name": "10-pd-disagg.svg",
        "title": "10 — PD-disaggregation (EXPERIMENTAL / WIP): pin prefill to the fast x16 card so its collectives skip the x4 lane",
        "subtitle": "Fork on the rig vs upstream. Implemented but NOT perf-/VRAM-benchmarked — TTFT ESTIMATED, combined per-card VRAM not captured.",
        "sentences": [
            ("Left: the prefill instance runs solo TP=1 on the fast x16 5090 (zero cross-GPU traffic); the decode "
             "instance runs uneven-TP=3 + DCP on the x4/x8 cards; KV is handed off via mooncake_tcp loopback. "
             "Both instances are CUDA-graph-covered by default (prefill = breakable graph, decode = full graph, "
             "MEASURED). Two instances = two weight copies; the combined per-card split was not dumped.", "#333"),
            ("Status: EXPERIMENTAL / work in progress — implemented (local_proxy.py, pd_disaggregation_hook.py; "
             "#99 M1/M2) but NOT perf- or VRAM-benchmarked: the TTFT factor is ESTIMATED and the combined "
             "per-card VRAM is UNKNOWN (not captured), consistent with how other WIP features are marked here.", "#8a5a2b"),
            ("Right (illustrative): upstream runs PD across identical cards (a prefill pool + a decode pool of "
             "equal GPUs) or a single fused TP group. On identical cards there is no x4 lane to route around, so "
             "the fork's specific advantage (isolating the slow lane) is rig-specific.", "#555"),
        ],
        "left_label": "htsglang — prefill solo on 5090 + decode TP=3 on 3080s (WIP)",
        "right_label": "upstream — identical PD fleet / fused TP",
        "left_cw": 140,
        "right_cw": 140,
        "left_cards": [
            ("RTX 5090", "prefill TP=1 · x16", 32.6, [
                ("weights", 25.0, "full weights fp8 (prefill)", E),
                ("kv", 3.0, "prefill KV (handed off)", E),
                ("free", 2.0, "decode rank0 co-resident — not captured", U)]),
            ("RTX 3080", "decode rank0 · x4", 20.5, [
                ("weights", 8.0, "decode shard", E),
                ("hostkv", 3.0, "handed KV in", E),
                ("kv", 6.0, "decode KV", E)]),
            ("RTX 3080", "decode rank1 · x8", 20.5, [
                ("weights", 8.0, "decode shard", E),
                ("hostkv", 3.0, "handed KV in", E),
                ("kv", 6.0, "decode KV", E)]),
        ],
        "right_cards": [
            ("identical", "prefill pool", 24, [("weights", 12.5, "weights", E), ("kv", 9.0, "KV", E)]),
            ("identical", "decode pool", 24, [("weights", 12.5, "weights", E), ("kv", 9.0, "KV", E)]),
        ],
        "left_notes": [
            ("Graph coverage MEASURED; the two-weight-copy combined per-card VRAM was never registry-dumped "
             "(\"not captured\"). Faster TTFT is EXPECTED (prefill avoids the x4-lane collectives) but the TTFT "
             "factor is an ESTIMATE, not benchmarked on this no-P2P/no-NVLink rig.", "#1d6b34"),
            ("Distributed decode is a CAPACITY choice, not a throughput win: decode is latency-sensitive "
             "(per-layer collectives EVERY step), so spreading it across the mixed cards makes every decode step "
             "pay cross-card collective latency — on this rig (no P2P/NVLink, all PHB, one 3080 on x4) a hard "
             "floor that can only be hidden, the very slow lane PD keeps prefill off. If model+KV fit the fast "
             "card alone, decode SOLO is faster (skips all cross-card collectives); if not (large model / large "
             "KV context), decode-KV MUST span the cards via uneven-DCP, the collective cost being the price of "
             "fitting. If the decode TP=3 instance also lands on the 5090, two model copies (prefill + decode) "
             "coexist there — the unknown two-copy VRAM.", "#333"),
        ],
        "right_notes": [
            ("On identical cards there is no x4 bottleneck to isolate. Illustrative sizes.", "#4a5568"),
        ],
        "core": ("PD pins prefill to the fast x16 card so its collectives skip the x4 lane; whether decode runs "
                 "solo on the fast card or is distributed across the mixed cards is a CAPACITY / PLACEMENT choice, "
                 "not a throughput win — distributed decode pays per-step cross-card collective latency that is "
                 "interconnect-bound on this no-P2P/no-NVLink rig. Experimental / not benchmarked."),
        "legend_keys": ["weights", "kv", "hostkv", "free"],
    },
]


# ===========================================================================
# Reframed 8-GPU fleet: explicitly ILLUSTRATIVE / NOT MEASURED, capacity only.
# ===========================================================================
def eight_gpu():
    W = 1180

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
                # whole fleet is illustrative -> hatched estimated fill
                o.append(evseg(x, cy, boxw8, rh, key, "estimated", lab if rh >= 12 else "", size=8))
                cy += rh
            o.append(text(x + boxw8 / 2, top + h + 12, cap, size=8, anchor="middle", color="#333"))
            return "".join(o)
        for i, (nm, vram, link, regs, cap) in enumerate(fleet):
            x = LEFT + i * SLOT8 + 11
            b.append(col8(x, topY + (32 - vram) * PXGB, nm, vram, regs, cap, link))
        hb, hy = host_bar_ev(LEFT, cb + 40, W - 2 * LEFT,
                             "Pinned host RAM (DDR) — illustrative",
                             [("spill", 0.6, "cold MoE experts (pinned), streamed over PCIe", "estimated"),
                              ("free", 0.4, "free", "estimated")])
        b.append(hb)
        cn, cy = core_note(LEFT, hy + 16, W - 2 * LEFT,
                           "capacity composition only. Per-layer-TP collectives over PCIe without P2P/NVLink are "
                           "bandwidth-bound — more cards / slower links means LESS throughput, not more. This whole "
                           "figure is a hypothetical 8-GPU illustration; this rig has only 3 GPUs and none of it was measured.")
        b.append(cn)
        return "".join(b), cy
    compose("15-eight-gpu-fleet.svg", W,
            "Appendix — Eight-GPU mixed fleet: how the capabilities COMPOSE (illustrative, not measured)",
            "5090 + 2x4090 + 3090 + 3x3080 + 2080Ti, mixed PCIe x16/x8/x4, no NVLink. Hypothetical — this rig has 3 GPUs.",
            [("The settings compose on CAPACITY: uneven-TP shards, uneven-DCP token-KV, two weightless-KV workers "
              "on the x4 cards, per-expert host offload (bottom bar), and PD-style prefill placement on the fast "
              "x16 card. The entire figure is ESTIMATED (hatched): it was never built or measured. Upstream even-TP "
              "would size the whole group to the 11 GB card's shard (or exclude it).", "#333")],
            draw, ["weights", "kv", "resident", "spill", "ctx", "free"], evidence=True)


# ===========================================================================
# Part 3 - runtime-mechanism diagrams (kept intact from the prior revision):
# adaptive drafter routing, session KV spill, fast-lane, measured VRAM budget.
# These explain the MECHANISM of a feature rather than a placement/VRAM split.
# ===========================================================================
DFLASH_FILL, DFLASH_STROKE, DFLASH_TXT = "#e4dcf1", "#7d5ba6", "#4a3b66"


def m11():
    """Adaptive drafter routing: NEXTN/DFLASH dual residence, policy vs bandit, ctx-gate."""
    def draw(topY, W):
        b = []
        top = topY + 6
        b.append(text(40, top, "Both drafters resident at once (cross_algo_worker, dual residence):",
                      size=10.5, weight="bold", color="#111"))
        y = top + 10
        b.append(chip(40, y, 150, 30, "mtp", "NEXTN / MTP", size=9.5))
        b.append(rect(205, y, 150, 30, DFLASH_FILL, stroke=DFLASH_STROKE, sw=1.4, rx=2))
        b.append(text(280, y + 19, "DFLASH", size=9.5, anchor="middle", color=DFLASH_TXT, weight="bold"))
        b.append(rect(370, y, 130, 30, "#eef1f4", stroke="#4a5568", sw=1.1, rx=2))
        b.append(text(435, y + 13, "solo-draft pool", size=8.5, anchor="middle", color="#333"))
        b.append(text(435, y + 24, "4.58 GiB on rank0 (measured)", size=7.6, anchor="middle", color="#555"))
        b.append(text(520, y + 13, "the inactive drafter holds ≈0 VRAM", size=9, color=DFLASH_STROKE))
        b.append(text(520, y + 25, "(VMM tag-alias, #93/#102)", size=9, color=DFLASH_STROKE))
        ry = y + 52
        b.append(rect(40, ry, W - 80, 30, "#f3f6fa", stroke="#3b6fb0", sw=1.3, rx=4))
        b.append(text(50, ry + 19, "Per-round-boundary router — picks ONE drafter for the next batch; the choice is made by one of two modes:",
                      size=9.5, color="#1a1a1a"))
        cy = ry + 48
        cardw = (W - 80 - 30) / 2
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
        bx = 40 + cardw + 30
        b.append(rect(bx, cy, cardw, 132, "#eef1f4", stroke="#4a5568", sw=1.4, rx=5))
        b.append(text(bx + 12, cy + 17, "auto / bandit  —  opt-in (--speculative-cross-algorithm)", size=10, weight="bold", color="#4a5568"))
        b.append(text(bx + 12, cy + 33, "acceptance-driven, for unknown drafters or content-split 4-8k loads", size=8.6, color="#333"))
        b.append(rect(bx + 12, cy + 42, cardw - 24, 24, "#ffffff", stroke="#4a5568", sw=1.0, rx=3))
        b.append(text(bx + 20, cy + 58, "score = EMA[accept-tokens / round] ÷ EMA[round seconds]", size=8.6, color="#1a1a1a"))
        b.append(text(bx + 12, cy + 82, "rank-0 decides, gloo-broadcasts every 16 rounds; dwell 64,", size=8.4, color="#333"))
        b.append(text(bx + 12, cy + 95, "dead-zone 6%. Costs a small steady-state probe overhead.", size=8.4, color="#333"))
        b.append(text(bx + 12, cy + 116, "Prior art: BanditSpec (arXiv 2505.15141, ICML'25).", size=8.4, color="#555"))
        gy = cy + 150
        b.append(rect(40, gy, W - 80, 30, "#fdf3e3", stroke="#c88a2b", sw=1.2, rx=4))
        b.append(text(50, gy + 19, "Context-length gate (--speculative-cross-algorithm-ctx-gate, from the drafter training config, ~8k): "
                      "above the gate DFLASH is ineligible and is not probed.", size=9, color="#8a5a2b"))
        hn, hy = flow(40, gy + 48, "Honest claim: robustness / no-regret across mixed streams — not a peak speedup. "
                      "A switching mode carries ≈+5.7% systemic overhead vs a single static drafter (measured), so the "
                      "win is confined to streams that actually change regime. Feature status: work in progress (§5).",
                      W - 80, size=9.5, color="#333")
        b.append(hn)
        return "".join(b), hy
    compose("11-adaptive-drafter-routing.svg", 1120,
            "11 (mechanism) — Adaptive drafter routing: switch draft algorithm at round boundaries",
            "Two draft algorithms resident at once; one chosen per batch. Decisive setting: drafter routing (§5, work in progress).",
            [("NEXTN/MTP and DFLASH are both loaded (the inactive one held at ≈0 VRAM via VMM tag-aliasing). "
              "A per-round router selects one, either by a deterministic ctx→rung policy table (recommended "
              "default) or an acceptance-driven bandit (opt-in), with a context-length gate that keeps DFLASH "
              "to its trained range. Upstream adaptive spec-decode adapts k / num-draft-tokens for a single "
              "drafter; switching between draft algorithms is the fork addition.", "#333")],
            draw, ["mtp"])


def m12():
    """Session KV spill mechanism."""
    def draw(topY, W):
        b = []
        top = topY + 10
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
        b.append(chip(x + 8, ly, 96, 46, "state", "GDN state", size=8.5))
        b.append(text(x + 56, ly + 60, "always resident", size=8, anchor="middle", color="#4a3b66"))
        newest_x = 40 + 3 * (118 + 12)
        b.append(text(newest_x + 59, ly - 2, "VRAM overflow (after tree eviction)", size=8.2, anchor="middle", color="#c0504d"))
        hy0 = ly + 96
        hb, hy = host_bar_ev(40, hy0, W - 80, "Host RAM (DDR) — spilled session KV",
                             [("hostkv", 0.30, "S4 KV (host-streamed, still decoding)", "measured"),
                              ("free", 0.70, "host KV tier", "measured")])
        b.append(hb)
        b.append(arrow(newest_x + 59, ly + 46, 40 + (W - 80) * 0.15, hy0, "#c0504d", marker="arrR"))
        b.append(text(newest_x + 120, ly + 74, "spill newest first (FCFS victim)", size=8.4, color="#c0504d"))
        b.append(arrow(40 + (W - 80) * 0.30, hy0, newest_x + 59, ly + 48, "#3f9fa0", dash="4 3", marker="arr"))
        b.append(text(40 + (W - 80) * 0.42, hy0 + 18, "FIFO restore when device capacity frees (~0.4 s)", size=8.4, color="#227a3a"))
        n1, ny = flow(40, hy + 26,
                      "Mechanism: on KV overflow the NEWEST session's full-attention KV shard is moved to host "
                      "RAM (block-LSE attention + double-buffer prefetch) and that session keeps DECODING from "
                      "host, in a separate eager bs=1 tick — never mixed into the device CUDA-graph batch. The "
                      "oldest session stays fully device-resident until it finishes (strict FCFS). Only KV spills; "
                      "GDN/Mamba state is always resident. Fast-lane requests take precedence (restore is held "
                      "while a fast request waits).", W - 80, size=9.3, color="#333")
        b.append(n1)
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
        for i, s in enumerate(["32 KiB/token fp8, x4 link, DCP R=3 + overlap:",
                                "32k ≈63 · 64k ≈31 · 128k ≈16 · 262k ≈7.6 tok/s",
                                "worthwhile only with uneven DCP (262k ≈3.8 without)"]):
            b.append(text(mx + 12, my + 32 + i * 14, "• " + s, size=8.4, color="#1a1a1a"))
        # Isolation matrix — how the DEVICE-RESIDENT (non-spilled) session runs during a spill.
        iy = my + 108
        b.append(text(40, iy, "During a spill (isolation matrix, ctx ~1.6k, MEASURED): does the DEVICE-RESIDENT (non-spilled) session stay fast?",
                      size=10, weight="bold", color="#111"))
        cx0, cy0 = 62, iy + 16
        chh, chw = 96, 300
        vmax = 44.0
        b.append(line(cx0, cy0, cx0, cy0 + chh, stroke="#888", sw=0.8))
        b.append(line(cx0, cy0 + chh, cx0 + chw, cy0 + chh, stroke="#888", sw=0.8))
        b.append(text(cx0 - 6, cy0 + 8, "tok/s", size=7.5, anchor="end", color="#888"))
        b.append(text(cx0 - 6, cy0 + chh, "0", size=7.5, anchor="end", color="#888"))
        refy = cy0 + chh - (40 / vmax) * chh
        b.append(line(cx0, refy, cx0 + chw, refy, stroke="#4a5568", sw=1.2, dash="5 3"))
        b.append(text(cx0 + chw + 8, refy + 3, "pre-spill both sessions ~40 tok/s", size=8.4, color="#4a5568"))
        spy = cy0 + chh - (7.5 / vmax) * chh
        b.append(line(cx0, spy, cx0 + chw, spy, stroke="#c0504d", sw=1.2, dash="2 2"))
        b.append(text(cx0 + chw + 8, spy + 3, "spilled session ~7–8 tok/s (all ticks)", size=8.4, color="#c0504d"))
        ticks = [("tick 1", 10.4), ("tick 2", 13.4), ("tick 4", 19.9), ("tick 8", 26.0)]
        status = ["target violated", "target violated", "target met", "target met"]
        bw = 40
        for i, (lab, val) in enumerate(ticks):
            bx = cx0 + 22 + i * (bw + 30)
            bh = (val / vmax) * chh
            byy = cy0 + chh - bh
            b.append(rect(bx, byy, bw, bh, COLORS["kv"], stroke="#2b2b2b", sw=1.0))
            b.append(text(bx + bw / 2, byy - 4, f"{val}", size=8.6, anchor="middle", color="#1a1a1a"))
            b.append(text(bx + bw / 2, cy0 + chh + 12, lab, size=8.4, anchor="middle", color="#333"))
            col = "#c0504d" if i < 2 else "#1d6b34"
            b.append(text(bx + bw / 2, cy0 + chh + 23, status[i], size=7.4, anchor="middle", color=col))
        ny0 = cy0 + chh + 42
        hn2, ny1 = flow(40, ny0, "Honest read: the device-resident session is still dragged along by the spill — the "
                        "isolation target (device loses at most its 1/N tick share) holds only from tick-interval 4 "
                        "upward; at tick 1/2 it is VIOLATED (device falls from ~40 to ~10–13 tok/s) because the spill "
                        "step still runs eager and blocks the shared scheduler tick.", W - 80, size=9.3, color="#333")
        b.append(hn2)
        pby = ny1 + 6
        b.append(rect(40, pby, W - 80, 34, "#f4f0fa", stroke="#7d5ba6", sw=1.2, rx=4))
        b.append(text(52, pby + 14, "Planned (Step 5 — NOT YET built, not measured): a bs=1 spill CUDA-graph takes the eager spill tick out of the shared",
                      size=9, color="#4a3b66"))
        b.append(text(52, pby + 27, "cadence, so even tick-interval 1 should barely touch the device-resident session.",
                      size=9, color="#4a3b66"))
        return "".join(b), pby + 40
    compose("12-session-kv-spill.svg", 1120,
            "12 (mechanism) — Session KV spill: overflow the newest session to host RAM, keep decoding",
            "Per-session KV offload under VRAM pressure. Decisive setting: --enable-kv-session-offload (§20, experimental S1).",
            [("When device KV overflows, the newest active session's KV shard is offloaded to host RAM and that "
              "session keeps decoding via host-streamed attention, instead of being paused. Victim order is "
              "strict FCFS (oldest stays resident) with fast-lane precedence; sessions restore FIFO when "
              "capacity frees; only KV spills, GDN state stays resident. Upstream retracts/recomputes or swaps "
              "a request (paused, not decoded from host).", "#333")],
            draw, ["kv", "hostkv", "state", "free"])


def m13():
    """Fast-lane priority scheduling mechanism."""
    def draw(topY, W):
        b = []
        top = topY + 10
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
        b.append(text(40, by + 56, "Guarantee 1 — RESERVED HEAVY SLOTS (--fast-lane-reserved-heavy-slots): at least N "
                      "normal (\"heavy\") requests are never preempted below this floor", size=8.6, color="#1d6b34"))
        b.append(text(40, by + 69, "(schedule_policy.py: max_heavy_preemptible = num_heavy_running − reserved_slots; preemption "
                      "stops at the floor) → sustained fast load cannot fully starve normal requests.", size=8.4, color="#333"))
        fy = by + 92
        b.append(rect(40, fy, 150, 28, "#c65b9b", stroke="#2b2b2b", sw=1.3, rx=3))
        b.append(text(115, fy + 18, "\"lane\":\"fast\" request", size=9, anchor="middle", color="#fff", weight="bold"))
        b.append(arrow(195, fy + 14, 300, fy + 14, "#c65b9b", marker="arr"))
        b.append(text(360, fy + 6, "binary opt-in: sets a fixed high fast_lane_priority (no manual integer)", size=9, color="#c65b9b"))
        b.append(text(360, fy + 19, "and preempts into the running batch, down to the reserved floor.", size=8.6, color="#555"))
        ay = fy + 44
        b.append(text(40, ay, "Guarantee 2 — HEAVY AGING (--fast-lane-heavy-aging-ms): a normal request waiting past the "
                      "window is promoted to fast_lane_priority−1", size=9, color="#1d6b34"))
        b.append(text(40, ay + 13, "and jumps ahead of the fast tier → a stream of fast requests cannot block a waiting normal one "
                      "indefinitely.", size=8.6, color="#333"))
        iy = ay + 34
        b.append(rect(40, iy, W - 80, 46, "#fdf3e3", stroke="#c88a2b", sw=1.2, rx=4))
        b.append(text(52, iy + 18, "Coupling with session KV spill (§20): a fast request can spill a normal session's KV to host "
                      "RATHER than queue,", size=9, color="#8a5a2b"))
        b.append(text(52, iy + 33, "a spilled session's restore is held while a fast request waits, and a fast request is never itself "
                      "spilled.", size=9, color="#8a5a2b"))
        uy = iy + 60
        b.append(rect(40, uy, W - 80, 44, "#eef1f4", stroke="#4a5568", sw=1.2, rx=4))
        b.append(text(52, uy + 17, "Upstream baseline (schedule_policy.py, verified): priority scheduling sorts the waiting queue by a "
                      "continuous integer priority and", size=9, color="#4a5568"))
        b.append(text(52, uy + 31, "preempts a running request when priority_diff > priority_scheduling_preemption_threshold — general, "
                      "with no reserved floor for the preempted and no aging.", size=9, color="#4a5568"))
        b.append(text(40, uy + 62, "Default off (--enable-fast-lane opt-in): the default scheduling path is unchanged.",
                      size=9.3, color="#1d6b34"))
        return "".join(b), uy + 74
    compose("13-fast-lane-priority.svg", 1060,
            "13 (mechanism) — Fast-lane: a fairness / anti-starvation layer on the priority path",
            "Opt-in binary lane with two anti-starvation guarantees. Decisive setting: --enable-fast-lane (§16, implemented).",
            [("Upstream supplies the priority axis (continuous integer priority + a preemption threshold). The "
              "fork's fast-lane is a binary opt-in lane ON that path whose delta is two anti-starvation guarantees "
              "generic priority does not have — a reserved floor of heavy slots that are never preempted, and "
              "heavy-aging that promotes a long-waiting normal request ahead of the fast tier — plus coupling to "
              "session KV spill. Stated as what each side does, not a ranking.", "#333")],
            draw, ["kv", "free"])


def m14():
    """Measured VRAM budget mechanism: components measured, KV is the remainder; corridor rule."""
    def draw(topY, W):
        b = []
        top = topY + 20
        PX = 5.2
        x = 60
        cardw = 150
        total = 20.0
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
                ly = max(mid_y, last_leader_y + 12)
                last_leader_y = ly
                b.append(line(x + cardw, mid_y, x + cardw + 8, ly, stroke="#888", sw=0.7))
                b.append(text(x + cardw + 11, ly + 3, lab, size=8, anchor="start", color="#333"))
            cy += gb * PX
        kvh = top + total * PX - cy - 0.5 * PX
        b.append(rect(x, cy, cardw, kvh, COLORS["kv"]))
        b.append(text(x + cardw / 2, cy + kvh / 2 - 4, "KV cache", size=9, anchor="middle", color="#fff", weight="bold"))
        b.append(text(x + cardw / 2, cy + kvh / 2 + 9, "= measured remainder", size=7.6, anchor="middle", color="#fff"))
        sy = top + total * PX - 0.5 * PX
        b.append(rect(x, sy, cardw, 0.5 * PX, "#d3dae1", stroke="#888", sw=0.6))
        b.append(line(x + cardw, sy + 1, x + cardw + 8, sy + 1, stroke="#888", sw=0.7))
        b.append(text(x + cardw + 11, sy + 4, "safety rest ≥ 400 MiB", size=8, color="#333"))
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
        cby = y2 + 14
        b.append(rect(rx, cby, W - rx - 30, 62, "#eef5ef", stroke="#1d6b34", sw=1.3, rx=4))
        b.append(text(rx + 12, cby + 17, "Corridor rule (evaluated per card, Option A):", size=9.3, weight="bold", color="#1d6b34"))
        b.append(text(rx + 12, cby + 33, "• fail if nvml_free < 400 MiB (absolute floor)", size=8.8, color="#1a1a1a"))
        b.append(text(rx + 12, cby + 48, "• fail if (nvml_free − measured transients) > 1.5 GiB (net waste)", size=8.8, color="#1a1a1a"))
        bottom = max(top + total * PX + 24, cby + 74)
        b.append(text(x, top + total * PX + 18, "upstream: fraction-based mem-fraction-static /", size=8.6, color="#555"))
        b.append(text(x, top + total * PX + 30, "gpu-memory-utilization; no per-rank absolute MiB budget.", size=8.6, color="#555"))
        return "".join(b), max(bottom, top + total * PX + 36)
    compose("14-measured-vram-budget.svg", 1040,
            "14 (mechanism) — Measured VRAM budget: components measured, KV is the remainder",
            "Per-rank absolute MiB budget from measured usage. Decisive setting: --rank-gpu-memory-mib + component registry (§10, implemented).",
            [("Each rank gets an absolute MiB budget (not a fraction). The per-component usage — CUDA context, "
              "weight shard, resident experts, solo-draft pool, GDN state, graph/workspace pools — is measured "
              "from a component registry after boot; the KV cache is sized as the measured remainder, and a "
              "logged split-hint vector converges over two boots. A corridor rule fails a card that has < 400 "
              "MiB free or > 1.5 GiB of net measured waste. Upstream sizes memory by a global fraction.", "#333")],
            draw, ["weights", "kv", "resident", "ctx", "mtp", "state", "free"])


if __name__ == "__main__":
    for spec in FEATURES:
        feature(spec)
    eight_gpu()
    for fn in (m11, m12, m13, m14):
        fn()
    print("done")
