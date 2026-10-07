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
    "barlink_default": "NF_PROFILE.md 5.1: Kollektive und Flip-Lanes über barlink BAR1; PP-send/recv über NCCL",
    "window_need": "NF_PROFILE.md 5.2: Gruppenfenster rund 168 von 256 MiB; HW-GENERISCH K3: BAR1 < ~168 MiB passt nicht",
    "rig_bar1": "NF_PROFILE.md 5.2: 5090 32768 MiB, 3080 je 256 MiB, EnableResizableBar 0 (24.09.)",
    "rig_widths": "Nutzer-bestätigt 17.08.: 5090 x8, eine 3080 x8, eine 3080 x4 (memory rig-interconnect-p2p)",
    "host": "Host braucht gepatchten nvidia-open 595.58.03, RegistryDwords RMSmallBarP2PPeerBar1=1;PeerMappingOverride=1, dmabuf_holder",
    "nccl": "NF_PROFILE.md 5.1: reiner NCCL-Betrieb (--transport nccl) ist Entwicklungsschalter, für NF nie gebootet",
}


def link_gbs(gen: int, lanes: int) -> float:
    return round(LANE_GBS[int(gen)] * int(lanes), 2)


def bar1_mib(vram_mib: int, rebar: bool) -> dict:
    """BAR1-Größe.  Aus: 256 MiB (am Rig an den 3080 gemessen).  An: kleinste Zweierpotenz >= VRAM
    (abgeleitet; am Rig nur die 5090 mit 32768 MiB bestätigt)."""
    if not rebar:
        return {"mib": BAR1_NO_REBAR_MIB, "src": "Resizable BAR aus: 256 MiB (gemessen an den Rig-3080, NF_PROFILE.md 5.2)"}
    p = 1
    while p < int(vram_mib):
        p <<= 1
    return {"mib": p, "src": "Resizable BAR an: Zweierpotenz >= VRAM (abgeleitet; Rig-Bestätigung nur 5090 = 32768 MiB)"}


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
        raise ValueError("PCIe-Generation muss 3, 4 oder 5 sein, nicht %r" % gen)
    if lanes not in LANES:
        raise ValueError("PCIe-Lanes müssen 1, 4, 8 oder 16 sein, nicht %r" % lanes)
    return {"gen": gen, "lanes": lanes, "rebar": bool(raw.get("rebar", False)),
            "chipset": bool(raw.get("chipset", False))}


def per_card_link(cat_entry: dict, slot: dict) -> dict:
    s = normalize_slot(slot)
    eff = effective_link(cat_entry["pcie_native"], s)
    bar = bar1_mib(cat_entry["usable_mib"], s["rebar"])
    return {"slot": s, "effective": eff, "bar1_mib": bar["mib"], "bar1_src": bar["src"],
            "limited_by": ("Slot" if (s["gen"] < cat_entry["pcie_native"]["gen"]
                                      or s["lanes"] < cat_entry["pcie_native"]["lanes"]) else "Karte"),
            "chipset": s["chipset"]}


def choose_transport(links: List[dict], labels: List[str], *, host_patched: bool = True) -> dict:
    """Transport für die weg2-Kollektive und den Flip.  ``links`` = Ausgaben von ``per_card_link``.

    Ergebnis: ``transport`` ('bar1' | 'nccl' | 'keiner'), ``confidence`` ('belegt' | 'ungeprüft' | 'entwicklung'),
    ``reasons`` (jede Angabe, die in die Wahl einging), ``warnings``, ``card_notes`` (je Karte)."""
    reasons: List[str] = []
    warns: List[str] = []
    notes: List[str] = []
    n = len(links)
    if n < 2:
        return {"transport": "keiner", "confidence": "belegt",
                "reasons": ["Eine Karte: kein Kartenverkehr, weder barlink noch NCCL nötig. Der weg2-Flip braucht zwei Gruppen auf denselben Karten "
                            "und mindestens 2 Karten (topology.MIN_CARDS)."],
                "warnings": [], "card_notes": ["%s: %s" % (labels[0], "keine Gegenstelle")] if labels else []}
    small = [(labels[i], l["bar1_mib"]) for i, l in enumerate(links) if l["bar1_mib"] < BAR1_WINDOW_NEED_MIB]
    chip = [labels[i] for i, l in enumerate(links) if l["chipset"]]
    narrow = [(labels[i], l["effective"]) for i, l in enumerate(links) if l["effective"]["lanes"] <= 4]
    for i, l in enumerate(links):
        e = l["effective"]
        notes.append("%s: PCIe Gen%d x%d = %.1f GB/s brutto (%s begrenzt), BAR1 %d MiB%s%s"
                     % (labels[i], e["gen"], e["lanes"], e["gbs"], l["limited_by"], l["bar1_mib"],
                        " (Resizable BAR %s)" % ("an" if l["slot"]["rebar"] else "aus"),
                        ", über Chipsatz" if l["chipset"] else ""))
    if not host_patched:
        reasons.append("Der Host hat NICHT den gepatchten nvidia-open 595.58.03 mit RMSmallBarP2PPeerBar1=1/PeerMappingOverride=1 und "
                       "dmabuf_holder: barlink BAR1 verweigert (Quelle: " + FACTS["host"] + ").")
        reasons.append("Bleibt NCCL (kein GPUDirect-P2P auf GeForce: Host-Staging). " + FACTS["nccl"] + ".")
        return {"transport": "nccl", "confidence": "entwicklung", "reasons": reasons,
                "warnings": ["NCCL-Betrieb der weg2-Linie ist ein Entwicklungsschalter und für NF nie gebootet."],
                "card_notes": notes}
    if small:
        reasons.append("BAR1 zu klein für die weg2-Gruppenfenster (nötig %d MiB, Quelle: %s): %s."
                       % (BAR1_WINDOW_NEED_MIB, FACTS["window_need"], ", ".join("%s %d MiB" % s for s in small)))
        reasons.append("Darum NCCL statt barlink BAR1.")
        return {"transport": "nccl", "confidence": "entwicklung", "reasons": reasons,
                "warnings": ["NCCL-Betrieb ist ein Entwicklungsschalter (nie gebootet, " + FACTS["nccl"] + ")."],
                "card_notes": notes}
    reasons.append("barlink BAR1: gepatchter Host-Treiber vorausgesetzt (Annahme der Seite: ja), alle Karten haben BAR1 >= %d MiB "
                   "(kleinste: %d MiB). Das ist die Bestform des Rigs (%s)."
                   % (BAR1_WINDOW_NEED_MIB, min(l["bar1_mib"] for l in links), FACTS["barlink_default"]))
    reasons.append("PP-Aktivierungen zwischen den P-Stufen laufen unabhängig davon über NCCL (send/recv).")
    conf = "belegt"
    if any(not l["slot"]["rebar"] for l in links):
        reasons.append("Resizable BAR aus auf %s: BAR1 = 256 MiB, genug für die Fenster (am Rig so gebootet); größere BAR1 bringt "
                       "keine belegte Beschleunigung." % ", ".join(labels[i] for i, l in enumerate(links) if not l["slot"]["rebar"]))
    if chip:
        conf = "ungeprüft"
        warns.append("Über Chipsatz angebunden: %s. Der DMA-Weg Karte-zu-Karte über den Chipsatz-Uplink (DMI) ist am Metall nicht geprüft "
                     "(alle Rig-Karten hängen an der CPU, PHB); der Uplink ist geteilt und schmaler als ein CPU-Slot." % ", ".join(chip))
        reasons.append("Wegen Chipsatz-Anbindung bleibt barlink BAR1 'ungeprüft'; Rückfall wäre NCCL (Entwicklungsschalter).")
    if narrow:
        warns.append("Schmale Anbindung (<= x4): %s. Am Rig läuft eine 3080 auf x4; der Planer legt dort die Stufe mit den wenigsten "
                     "Attention-Layern hin (Platzierungsregel #704b). Flip-Preise (Pull-Rate 6,5 GB/s 'x4-3080-Kante', HW-GENERISCH K4) "
                     "sind nur für das Rig gemessen." % ", ".join("%s x%d" % (a, e["lanes"]) for a, e in narrow))
    slow = min(links, key=lambda l: l["effective"]["gbs"])
    warns.append("Engste Anbindung: %.1f GB/s brutto; sie bestimmt Flip-Gewichtstausch und Ring-Restore (theoretisch, nicht gemessen)." % slow["effective"]["gbs"])
    return {"transport": "bar1", "confidence": conf, "reasons": reasons, "warnings": warns, "card_notes": notes}
