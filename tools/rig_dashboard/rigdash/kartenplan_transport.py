"""Kartenplaner (Item 510): PCIe-Anbindung je Karte und Transportwahl (barlink BAR1 / NCCL).

Eingabe je Karte: Generation 3/4/5, Lanes x1/x4/x8/x16, Resizable BAR an/aus, "über Chipsatz" ja/nein.
Alle vier Angaben fließen in die Wahl und in ihre Begründung ein.

Belegte Rig-Fakten (Quelle in ``FACTS``):
  * barlink BAR1 ist der Transport der Bestform; PP-Aktivierungen laufen trotzdem über NCCL (NF_PROFILE.md 5.1).
  * Gruppenfenster auf den 3080 belegen "rund 168 von 256 MiB" BAR1 (NF_PROFILE.md 5.2); Weg-2-Fenster
    PP_0=96 + D 16+32+24 (HW-GENERISCH K3).  Weniger als 168 MiB BAR1 -> Fenster passen nicht.
  * Rig-BAR1: 5090 32768 MiB, 3080 je 256 MiB, EnableResizableBar 0.
  * Host braucht den gepatchten nvidia-open 595.58.03 + dmabuf_holder (Host-Tatsache, nicht je Karte).
  * Alle Rig-Karten hängen an der CPU (PHB); ein Chipsatz-Slot wurde nie am Metall geprüft.

Link-Bandbreite: brutto, Datenblatt (Gen3 8, Gen4 16, Gen5 32 GT/s je Lane, 128b/130b); real etwas weniger.
"""

from __future__ import annotations

from typing import Dict, List, Optional

#: GB/s je Lane und Richtung, brutto nach 128b/130b-Kodierung (Datenblatt, theoretisch)
LANE_GBS = {3: 0.985, 4: 1.969, 5: 3.938}
GENS = (3, 4, 5)
LANES = (1, 4, 8, 16)

#: Mindest-BAR1 für die weg2-Gruppenfenster (MiB): 96 (P PP_0) + 16 + 32 + 24 (D) = 168
BAR1_WINDOW_NEED_MIB = 168
#: BAR1 bei ausgeschaltetem Resizable BAR: das Rig misst 256 MiB auf den 3080
BAR1_NO_REBAR_MIB = 256

FACTS = {
    "barlink_default": "NF_PROFILE.md 5.1: collectives and flip lanes via barlink BAR1; PP send/recv via NCCL",
    "window_need": "NF_PROFILE.md 5.2: group window about 168 of 256 MiB; HW-GENERISCH K3: BAR1 < ~168 MiB does not fit",
    "rig_bar1": "NF_PROFILE.md 5.2: 5090 32768 MiB, 3080 256 MiB each, EnableResizableBar 0 (24.09.)",
    "rig_widths": "User-confirmed 17.08.: 5090 x8, one 3080 x8, one 3080 x4 (memory rig-interconnect-p2p)",
    "host": "Host needs the patched nvidia-open 595.58.03, RegistryDwords RMSmallBarP2PPeerBar1=1;PeerMappingOverride=1, dmabuf_holder",
    "nccl": "NF_PROFILE.md 5.1: pure NCCL operation (--transport nccl) is a development switch, never booted for NF",
}


def link_gbs(gen: int, lanes: int) -> float:
    return round(LANE_GBS[int(gen)] * int(lanes), 2)


def bar1_mib(vram_mib: int, rebar: bool) -> dict:
    """BAR1-Größe.  Aus: 256 MiB (am Rig an den 3080 gemessen).  An: kleinste Zweierpotenz >= VRAM
    (abgeleitet; am Rig nur die 5090 mit 32768 MiB bestätigt)."""
    if not rebar:
        return {"mib": BAR1_NO_REBAR_MIB, "src": "Resizable BAR off: 256 MiB (measured on the rig 3080s, NF_PROFILE.md 5.2)"}
    p = 1
    while p < int(vram_mib):
        p <<= 1
    return {"mib": p, "src": "Resizable BAR on: power of two >= VRAM (derived; rig confirmation only 5090 = 32768 MiB)"}


def effective_link(card_native: dict, slot: dict) -> dict:
    """Kleinster gemeinsamer Nenner aus Karte und Slot (Gen und Lanes getrennt)."""
    gen = min(int(card_native["gen"]), int(slot["gen"]))
    lanes = min(int(card_native["lanes"]), int(slot["lanes"]))
    return {"gen": gen, "lanes": lanes, "gbs": link_gbs(gen, lanes)}


def normalize_slot(raw: Optional[dict]) -> dict:
    raw = raw or {}
    gen = int(raw.get("gen", 4))
    lanes = int(raw.get("lanes", 16))
    if gen not in GENS:
        raise ValueError("PCIe generation must be 3, 4 or 5, not %r" % gen)
    if lanes not in LANES:
        raise ValueError("PCIe lanes must be 1, 4, 8 or 16, not %r" % lanes)
    return {"gen": gen, "lanes": lanes, "rebar": bool(raw.get("rebar", False)),
            "chipset": bool(raw.get("chipset", False))}


def per_card_link(cat_entry: dict, slot: dict) -> dict:
    s = normalize_slot(slot)
    eff = effective_link(cat_entry["pcie_native"], s)
    bar = bar1_mib(cat_entry["usable_mib"], s["rebar"])
    return {"slot": s, "effective": eff, "bar1_mib": bar["mib"], "bar1_src": bar["src"],
            "limited_by": ("Slot" if (s["gen"] < cat_entry["pcie_native"]["gen"]
                                      or s["lanes"] < cat_entry["pcie_native"]["lanes"]) else "card"),
            "chipset": s["chipset"]}


def choose_transport(links: List[dict], labels: List[str], *, host_patched: bool = True) -> dict:
    """Transport für die weg2-Kollektive und den Flip.  ``links`` = Ausgaben von ``per_card_link``.

    Ergebnis: ``transport`` ('bar1' | 'nccl' | 'none'), ``confidence`` ('verified' | 'unchecked' | 'development'),
    ``reasons`` (jede Angabe, die in die Wahl einging), ``warnings``, ``card_notes`` (je Karte)."""
    reasons: List[str] = []
    warns: List[str] = []
    notes: List[str] = []
    n = len(links)
    if n < 2:
        return {"transport": "none", "confidence": "verified",
                "reasons": ["One card: no card traffic, neither barlink nor NCCL needed. The weg2 flip needs two groups on the same cards and at least 2 cards (topology.MIN_CARDS)."],
                "warnings": [], "card_notes": ["%s: %s" % (labels[0], "no peer")] if labels else []}
    small = [(labels[i], l["bar1_mib"]) for i, l in enumerate(links) if l["bar1_mib"] < BAR1_WINDOW_NEED_MIB]
    chip = [labels[i] for i, l in enumerate(links) if l["chipset"]]
    narrow = [(labels[i], l["effective"]) for i, l in enumerate(links) if l["effective"]["lanes"] <= 4]
    for i, l in enumerate(links):
        e = l["effective"]
        notes.append("%s: PCIe Gen%d x%d = %.1f GB/s gross (%s limited), BAR1 %d MiB%s%s"
                     % (labels[i], e["gen"], e["lanes"], e["gbs"], l["limited_by"], l["bar1_mib"],
                        " (Resizable BAR %s)" % ("on" if l["slot"]["rebar"] else "off"),
                        ", via chipset" if l["chipset"] else ""))
    if not host_patched:
        reasons.append("The host does NOT have the patched nvidia-open 595.58.03 with RMSmallBarP2PPeerBar1=1/PeerMappingOverride=1 and dmabuf_holder: barlink BAR1 refused (source: " + FACTS["host"] + ").")
        reasons.append("NCCL remains (no GPUDirect P2P on GeForce: host staging). " + FACTS["nccl"] + ".")
        return {"transport": "nccl", "confidence": "development", "reasons": reasons,
                "warnings": ["NCCL operation of the weg2 line is a development switch and never booted for NF."],
                "card_notes": notes}
    if small:
        reasons.append("BAR1 too small for the weg2 group windows (needed %d MiB, source: %s): %s."
                       % (BAR1_WINDOW_NEED_MIB, FACTS["window_need"], ", ".join("%s %d MiB" % s for s in small)))
        reasons.append("Hence NCCL instead of barlink BAR1.")
        return {"transport": "nccl", "confidence": "development", "reasons": reasons,
                "warnings": ["NCCL operation is a development switch (never booted, " + FACTS["nccl"] + ")."],
                "card_notes": notes}
    reasons.append("barlink BAR1: patched host driver assumed (assumption of the page: yes), all cards have BAR1 >= %d MiB (smallest: %d MiB). This is the best form of the rig (%s)."
                   % (BAR1_WINDOW_NEED_MIB, min(l["bar1_mib"] for l in links), FACTS["barlink_default"]))
    reasons.append("PP activations between the P stages run via NCCL (send/recv) regardless of this.")
    conf = "verified"
    if any(not l["slot"]["rebar"] for l in links):
        reasons.append("Resizable BAR off on %s: BAR1 = 256 MiB, enough for the windows (booted like this on the rig); a larger BAR1 brings no verified speedup." % ", ".join(labels[i] for i, l in enumerate(links) if not l["slot"]["rebar"]))
    if chip:
        conf = "unchecked"
        warns.append("Connected via chipset: %s. The DMA path card to card via the chipset uplink (DMI) is not checked on the hardware (all rig cards hang on the CPU, PHB); the uplink is shared and narrower than a CPU slot." % ", ".join(chip))
        reasons.append("Because of the chipset connection barlink BAR1 stays 'unchecked'; fallback would be NCCL (development switch).")
    if narrow:
        warns.append("Narrow link (<= x4): %s. On the rig one 3080 runs at x4; the planner puts the stage with the fewest attention layers there (placement rule #704b). Flip prices (pull rate 6.5 GB/s 'x4-3080 edge', HW-GENERISCH K4) are measured only for the rig." % ", ".join("%s x%d" % (a, e["lanes"]) for a, e in narrow))
    slow = min(links, key=lambda l: l["effective"]["gbs"])
    warns.append("Narrowest link: %.1f GB/s gross; it determines the flip weight exchange and ring restore (theoretical, not measured)." % slow["effective"]["gbs"])
    return {"transport": "bar1", "confidence": conf, "reasons": reasons, "warnings": warns, "card_notes": notes}
