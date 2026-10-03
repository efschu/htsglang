"""Kartenplaner (Item 510): Katalog der Grafikkarten und der wählbaren Modelle/Profile.

Reine Daten plus kleine Prüffunktionen, nur Standardbibliothek.  Jede Zahl trägt
ihre Quelle:

  * ``NVML-Record``  = am Rig gemessen (vram_plan.json ``cards`` der echten Boots,
                       PCIe-Breiten vom Nutzer bestätigt 17.08., BAR1 aus NF_PROFILE.md 5.2)
  * ``Datenblatt``   = Herstellerangabe (Nennwert; NVML meldet bei fremden Karten
                       einige MiB weniger, das ist NICHT gemessen)
  * ``abgeleitet``   = aus den obigen Angaben gerechnet, Rechenweg im Feld ``src``

Nie ein geratener Wert ohne dieses Etikett.  Turing (sm75) ist als deaktivierter
Eintrag vorbereitet (Nutzer 03.10.: erst nach sm75-Port in den Katalog).
"""

from __future__ import annotations

from typing import Dict, List, Optional

MIB_PER_GB = 1024

#: Referenz-Rig, in NVML-Reihenfolge (vram_plan.json cards, Boot 27bbf-boot-20261003T090641Z-30ca)
RIG_NVML = {
    "RTX 3080 20GB": {"total_mib": 20480, "driver_reserved_mib": 425},
    "RTX 5090 32GB": {"total_mib": 32607, "driver_reserved_mib": 518},
}

# arch -> (cc, Bezeichnung, bf16, fp8_nativ, nvfp4_nativ)
ARCHS: Dict[str, dict] = {
    "sm120": {"cc": (12, 0), "bf16": True, "fp8_native": True, "nvfp4_native": True},
    "sm89": {"cc": (8, 9), "bf16": True, "fp8_native": True, "nvfp4_native": False},
    "sm86": {"cc": (8, 6), "bf16": True, "fp8_native": False, "nvfp4_native": False},
    "sm75": {"cc": (7, 5), "bf16": False, "fp8_native": False, "nvfp4_native": False},
}


def _card(cid, name, gb, arch, bw, pcie_gen, pcie_lanes, *, usable=None, usable_src=None,
          bw_src="Datenblatt", pcie_src="Datenblatt", variant="", enabled=True, off_reason="",
          note="", bus_bits=None, nvml_name=None):
    usable_mib = usable if usable is not None else gb * MIB_PER_GB
    return {
        "id": cid, "name": name, "variant": variant, "vram_gb": gb,
        "usable_mib": usable_mib,
        "usable_src": usable_src or "Datenblatt (Nennwert GB x 1024; NVML meldet fremde Karten leicht darunter, ungemessen)",
        "arch": arch, "cc": list(ARCHS[arch]["cc"]),
        "mem_bw_gbs": bw, "mem_bw_src": bw_src,
        "bus_bits": bus_bits,
        "pcie_native": {"gen": pcie_gen, "lanes": pcie_lanes, "src": pcie_src},
        "nvml_name": nvml_name or ("NVIDIA GeForce " + name),
        "enabled": enabled, "off_reason": off_reason, "note": note,
        "driver_reserved_mib": None,
    }


def _build_catalog() -> List[dict]:
    c: List[dict] = []
    c.append(_card("rtx5090-32", "RTX 5090", 32, "sm120", 1792, 5, 16, usable=32607,
                   usable_src="NVML-Record (Rig, vram_plan.json cards, nvml1)", bus_bits=512,
                   note="Referenzkarte des Rigs (TP0/PP0, Form-A-Host)"))
    c.append(_card("rtx5080-16", "RTX 5080", 16, "sm120", 960, 5, 16, bus_bits=256))
    c.append(_card("rtx5070ti-16", "RTX 5070 Ti", 16, "sm120", 896, 5, 16, bus_bits=256))
    c.append(_card("rtx5070-12", "RTX 5070", 12, "sm120", 672, 5, 16, bus_bits=192))
    c.append(_card("rtx5060ti-16", "RTX 5060 Ti", 16, "sm120", 448, 5, 8, variant="16 GB", bus_bits=128))
    c.append(_card("rtx5060ti-8", "RTX 5060 Ti", 8, "sm120", 448, 5, 8, variant="8 GB", bus_bits=128))

    c.append(_card("rtx4090-24", "RTX 4090", 24, "sm89", 1008, 4, 16, bus_bits=384))
    c.append(_card("rtx4080s-16", "RTX 4080 SUPER", 16, "sm89", 736, 4, 16, bus_bits=256))
    c.append(_card("rtx4080-16", "RTX 4080", 16, "sm89", 716.8, 4, 16, bus_bits=256))
    c.append(_card("rtx4070tis-16", "RTX 4070 Ti SUPER", 16, "sm89", 672, 4, 16, bus_bits=256))
    c.append(_card("rtx4070ti-12", "RTX 4070 Ti", 12, "sm89", 504, 4, 16, bus_bits=192))
    c.append(_card("rtx4070s-12", "RTX 4070 SUPER", 12, "sm89", 504, 4, 16, bus_bits=192))
    c.append(_card("rtx4070-12", "RTX 4070", 12, "sm89", 504, 4, 16, bus_bits=192))
    c.append(_card("rtx4060ti-16", "RTX 4060 Ti", 16, "sm89", 288, 4, 8, variant="16 GB", bus_bits=128))
    c.append(_card("rtx4060ti-8", "RTX 4060 Ti", 8, "sm89", 288, 4, 8, variant="8 GB", bus_bits=128))

    c.append(_card("rtx3090ti-24", "RTX 3090 Ti", 24, "sm86", 1008, 4, 16, bus_bits=384))
    c.append(_card("rtx3090-24", "RTX 3090", 24, "sm86", 936, 4, 16, bus_bits=384))
    c.append(_card("rtx3080ti-12", "RTX 3080 Ti", 12, "sm86", 912, 4, 16, bus_bits=384))
    c.append(_card("rtx3080-12", "RTX 3080", 12, "sm86", 912, 4, 16, variant="12 GB", bus_bits=384))
    c.append(_card("rtx3080-10", "RTX 3080", 10, "sm86", 760, 4, 16, variant="10 GB", bus_bits=320))
    c.append(_card("rtx3080-20", "RTX 3080", 20, "sm86", 760, 4, 16, variant="20 GB (Umbau)", usable=20480,
                   usable_src="NVML-Record (Rig, vram_plan.json cards, nvml0/nvml2)",
                   bw_src="Datenblatt der 10-GB-Karte (gleiche 320-Bit-Platine; Umbau-Karte am Rig nicht separat gemessen)",
                   bus_bits=320, note="Umbau-Karte, wie unser Rig (2 Stück)"))
    # Turing: vorbereitet, aber deaktiviert (Nutzer 03.10.)
    c.append(_card("rtx2080ti-11", "RTX 2080 Ti", 11, "sm75", 616, 3, 16, variant="11 GB", bus_bits=352,
                   enabled=False, off_reason="sm75 (Turing) ist noch nicht im Image/Planer: kein bf16, kein FP8, kein NVFP4; "
                   "Image baut nur sm86/sm120. Eintrag vorbereitet, aktiv erst nach dem sm75-Port."))
    c.append(_card("rtx2080ti-22", "RTX 2080 Ti", 22, "sm75", 616, 3, 16, variant="22 GB (Umbau)", bus_bits=352,
                   enabled=False, off_reason="sm75 (Turing) ist noch nicht im Image/Planer: kein bf16, kein FP8, kein NVFP4; Image baut nur sm86/sm120. Eintrag vorbereitet, aktiv erst nach dem sm75-Port."))
    for e in c:
        rig = e["usable_src"].startswith("NVML-Record")
        e["driver_reserved_mib"] = (518 if e["id"] == "rtx5090-32" else 425) if rig else None
        e["measured_on_rig"] = rig
    return c


CATALOG: List[dict] = _build_catalog()
_BY_ID = {c["id"]: c for c in CATALOG}


def card(card_id: str) -> Optional[dict]:
    return _BY_ID.get(str(card_id))


def label(entry: dict) -> str:
    v = entry.get("variant") or ""
    return entry["name"] + (" " + v if v else (" %d GB" % entry["vram_gb"]))


def catalog_public(include_disabled: bool = False) -> List[dict]:
    """Katalog für die Seite.  Deaktivierte (Turing) nur auf Wunsch."""
    out = []
    for e in CATALOG:
        if not e["enabled"] and not include_disabled:
            continue
        d = dict(e)
        d["label"] = label(e)
        out.append(d)
    return out


# --------------------------------------------------------------------------- Profile
#: Modelle/Profile des Kartenplaners (Nutzer-Entscheide 03.10.).  ``record`` nennt die
#: Planer-Aufzeichnung (kartenplan_data/<record>.json); ``release_profile`` die Release-Datei
#: unter docker/profiles_release, ``boot_profile`` das Profil des Referenz-Boots.
PROFILES: List[dict] = [
    {"id": "27b-int8", "label": "27B INT8 (gdncov)", "line": "27b", "format": "int8", "flip": True,
     "release_profile": "27b", "boot_profile": "27b-row-authority-cut43", "record": "27b-int8",
     "note": "Release-Form: P = PP3 (Prefill), D = TP3 (Decode), Flip; Draft DFlash2"},
    {"id": "nf-int4-abl", "label": "NF INT4 (abl)", "line": "nf", "format": "int4", "flip": True,
     "release_profile": "nf-int4", "boot_profile": "nf-int4-h6-abl", "record": "nf-int4-abl",
     "note": "Qwen3.8-Flash-Next (MoE, Experten im Host, MTP-Draft), Flip"},
    {"id": "27b-nvfp4-dual", "label": "27B NVFP4 Dual", "line": "27b", "format": "nvfp4", "flip": True,
     "release_profile": "27b-nvfp4-dual", "boot_profile": "27b-nvfp4-dual1m-psleep", "record": "27b-nvfp4-dual",
     "note": "Dual-Form: P und D im Wechsel, NVFP4-Gewichte"},
    {"id": "27b-fp8", "label": "27B FP8", "line": "27b", "format": "fp8", "flip": True,
     "release_profile": "27b-fp8", "boot_profile": "27b-fp8", "record": "27b-fp8",
     "note": "Metallbeleg nur aus dem Docker-Boot 26.09. (vor dem IPC-Zustand), daher ohne state.json"},
    {"id": "27b-gguf-iq4xs", "label": "27B GGUF UD-IQ4_XS", "line": "27b", "format": "gguf", "flip": True,
     "release_profile": "27b-gguf", "boot_profile": "27b-gguf", "record": "27b-gguf-iq4xs",
     "note": "Einzige am Metall gebootete GGUF-Variante des 27B; Q8_K_XL liegt auf Platte, wurde nie gebootet",
     "variants": [{"id": "UD-IQ4_XS", "file": "Qwen3.8-27B-UD-IQ4_XS.gguf",
                   "path": "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-GGUF-unsloth/Qwen3.8-27B-UD-IQ4_XS.gguf",
                   "size_bytes": 14252845984, "size_src": "ls -l, gemessen",
                   "boots": ["weg2rc5gg (25.09., RC5)", "weg2rc7gg2 (25.09.)",
                             "dkr27bggufbar109251311 / ...1743 / ...1825 (Docker, 25.09.)",
                             "dkr27bggufbar1final09260238 (Docker, 26.09., serving)"]}]},
]
_PROFILE_BY_ID = {p["id"]: p for p in PROFILES}


def profile(profile_id: str) -> Optional[dict]:
    return _PROFILE_BY_ID.get(str(profile_id))


# --------------------------------------------------------------------------- Lauffähigkeit je Karte/Format
def arch_status(cc: List[int], fmt: str) -> dict:
    """Ob eine Architektur dieses Image/dieses Format tragen kann.  ``level``:
    ``metall`` (am Rig gebootet), ``gesperrt`` (Planer/Image verweigert), ``ungebaut``."""
    tup = tuple(cc)
    if tup in ((8, 6), (12, 0)):
        return {"level": "metall", "ok": True,
                "why": "sm%d%d: im Release-Image gebaut (sgl-kernel-Wheel 86;120a, JIT-Prebuild 8.6,12.0); am Rig gebootet." % tup}
    if tup == (8, 9):
        return {"level": "ungebaut", "ok": False,
                "why": "sm89: das Release-Image enthält keinen sm89-Code (Wheel 86;120a); der Arch-Gate des Launchers verweigert "
                       "cc 8.9 (HW-ARCH, NF-Linie 7a8f4087a9). Der Port ist gelandet (Item 230: NF ec9e28cfa1, 27B-Teilstand 4c7eda3016; "
                       "FP8 läuft dort über den benannten Marlin-Rückfall FP8-SM89-FALLBACK), der sm89-Image-Bau ist nur gestaged "
                       "(Item 270, /spinning/gpu-arb/docker/sm89-1002/BUILD_SM89.md), ungebaut und am Metall ungetestet."}
    if tup == (7, 5):
        return {"level": "gesperrt", "ok": False,
                "why": "sm75 (Turing): kein bf16, kein FP8, kein NVFP4; das Image baut nur für sm86/sm120. Eintrag deaktiviert, bis sm75 hinzugefügt ist."}
    return {"level": "gesperrt", "ok": False, "why": "Architektur sm%d%d nicht im Image." % tup}
