"""Kartenplaner (Item 510): Catalog der Grafikkarten und der wählbaren Modelle/Profile.

Reine Data plus kleine Prüffunktionen, nur Standardbibliothek.  Jede Zahl trägt
ihre Quelle:

  * ``NVML-Record``  = am Rig gemessen (vram_plan.json ``cards`` der echten Boots,
                       PCIe-Breiten vom Nutzer bestätigt 17.08., BAR1 aus NF_PROFILE.md 5.2)
  * ``Datasheet``   = Herstellerangabe (Nennwert; NVML meldet bei fremden Karten
                       einige MiB weniger, das ist NICHT gemessen)
  * ``abgeleitet``   = aus den obigen Angaben gerechnet, Rechenweg im Feld ``src``

Nie ein geratener Wert ohne dieses Etikett.

Vorbelegt (``preset``, Plan Profil-Planer 06.10. R5): nur RTX 5090, RTX 3080 20 GB und RTX 3090.  Die übrigen Karten
bleiben im Catalog (Datasheet, ohne Messraten) und stehen in der Oberfläche eingeklappt.  Jeder Eintrag trägt seine
``origin`` (wie ``planner/card_library.py`` ``CardSpec.source``: ein Eintrag, der nicht beweisen kann, dass er gemessen
wurde, ist nicht gemessen):

  * ``measured_on_rig``   = die Speichergröße stammt aus einem NVML-Record des Rigs
  * ``Datasheet``        = Herstellerangabe, am Rig nicht gemessen
  * ``borrowed-unverified`` = ein Wert ist von einer anderen Variante borrowed und nicht belegt

``origin_fields`` nennt die Herkunft je Feld (vram, mem_bw, pcie); ein geborgtes Feld macht den Eintrag nicht zum Rig-Eintrag.  Turing (sm75) ist als deaktivierter
Eintrag vorbereitet (Nutzer 03.10.: erst nach sm75-Port in den Catalog).
"""

from __future__ import annotations

from typing import Dict, List, Optional

MIB_PER_GB = 1024

ORIGIN_MEASURED = "measured_on_rig"
ORIGIN_DATASHEET = "Datasheet"
ORIGIN_BORROWED = "borrowed-unverified"
ORIGIN_LABELS = {
    ORIGIN_MEASURED: "measured on the rig",
    ORIGIN_DATASHEET: "datasheet (manufacturer value, not measured)",
    ORIGIN_BORROWED: "borrowed from another variant, unverified",
}

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
          bw_src="Datasheet", pcie_src="Datasheet", variant="", enabled=True, off_reason="",
          note="", bus_bits=None, nvml_name=None, preset=False, borrowed=()):
    usable_mib = usable if usable is not None else gb * MIB_PER_GB
    return {
        "id": cid, "name": name, "variant": variant, "vram_gb": gb,
        "usable_mib": usable_mib,
        "usable_src": usable_src or "datasheet (nominal GB x 1024; NVML reports foreign cards slightly below, unmeasured)",
        "arch": arch, "cc": list(ARCHS[arch]["cc"]),
        "mem_bw_gbs": bw, "mem_bw_src": bw_src,
        "bus_bits": bus_bits,
        "pcie_native": {"gen": pcie_gen, "lanes": pcie_lanes, "src": pcie_src},
        "nvml_name": nvml_name or ("NVIDIA GeForce " + name),
        "enabled": enabled, "off_reason": off_reason, "note": note,
        "driver_reserved_mib": None,
        "preset": bool(preset), "borrowed": sorted(borrowed),
    }


def _build_catalog() -> List[dict]:
    c: List[dict] = []
    c.append(_card("rtx5090-32", "RTX 5090", 32, "sm120", 1792, 5, 16, usable=32607,
                   usable_src="NVML-Record (Rig, vram_plan.json cards, nvml1)", bus_bits=512, preset=True,
                   note="Reference card of the rig (TP0/PP0, form A host)"))
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
    c.append(_card("rtx3090-24", "RTX 3090", 24, "sm86", 936, 4, 16, bus_bits=384, preset=True))
    # Datenblatt: 8 GB GDDR6, 256 Bit, 448 GB/s, PCIe 4.0 x16 (GA104); Compute Capability 8.6 wie die übrigen 30er-Karten (Ampere), nicht am Rig gemessen
    c.append(_card("rtx3070-8", "RTX 3070", 8, "sm86", 448, 4, 16, bus_bits=256))
    c.append(_card("rtx3080ti-12", "RTX 3080 Ti", 12, "sm86", 912, 4, 16, bus_bits=384))
    c.append(_card("rtx3080-12", "RTX 3080", 12, "sm86", 912, 4, 16, variant="12 GB", bus_bits=384))
    c.append(_card("rtx3080-10", "RTX 3080", 10, "sm86", 760, 4, 16, variant="10 GB", bus_bits=320))
    c.append(_card("rtx3080-20", "RTX 3080", 20, "sm86", 760, 4, 16, variant="20 GB (modded)", usable=20480,
                   usable_src="NVML-Record (Rig, vram_plan.json cards, nvml0/nvml2)",
                   bw_src="datasheet of the 10 GB card (same 320-bit board; modded card not measured separately on the rig)",
                   bus_bits=320, note="Modded card, like our rig (2 pieces)", preset=True, borrowed=("mem_bw",)))
    # Turing: vorbereitet, aber deaktiviert (Nutzer 03.10.)
    c.append(_card("rtx2080ti-11", "RTX 2080 Ti", 11, "sm75", 616, 3, 16, variant="11 GB", bus_bits=352,
                   enabled=False, off_reason="sm75 (Turing) is not in the image/planner yet: no bf16, no FP8, no NVFP4; "
                   "the image builds only sm86/sm120. Entry prepared, active only after the sm75 port."))
    c.append(_card("rtx2080ti-22", "RTX 2080 Ti", 22, "sm75", 616, 3, 16, variant="22 GB (modded)", bus_bits=352,
                   enabled=False, off_reason="sm75 (Turing) is not in the image/planner yet: no bf16, no FP8, no NVFP4; the image builds only sm86/sm120. Entry prepared, active only after the sm75 port."))
    for e in c:
        rig = e["usable_src"].startswith("NVML-Record")
        e["driver_reserved_mib"] = (518 if e["id"] == "rtx5090-32" else 425) if rig else None
        e["measured_on_rig"] = rig
        # Herkunft je Feld und des Eintrags (Quelle: usable_src/bw_src/pcie_src der Eintragung, ``borrowed`` ausdrücklich)
        fields = {"vram": ORIGIN_MEASURED if rig else ORIGIN_DATASHEET,
                  "mem_bw": ORIGIN_BORROWED if "mem_bw" in e["borrowed"] else ORIGIN_DATASHEET,
                  "pcie": ORIGIN_DATASHEET if e["pcie_native"]["src"] == "Datasheet" else ORIGIN_MEASURED}
        e["origin_fields"] = fields
        e["origin"] = ORIGIN_MEASURED if rig else (ORIGIN_BORROWED if ORIGIN_BORROWED in fields.values() else ORIGIN_DATASHEET)
    return c


CATALOG: List[dict] = _build_catalog()
_BY_ID = {c["id"]: c for c in CATALOG}


def card(card_id: str) -> Optional[dict]:
    return _BY_ID.get(str(card_id))


def label(entry: dict) -> str:
    v = entry.get("variant") or ""
    return entry["name"] + (" " + v if v else (" %d GB" % entry["vram_gb"]))


def catalog_public(include_disabled: bool = False) -> List[dict]:
    """Catalog für die Seite.  Deaktivierte (Turing) nur auf Wunsch."""
    out = []
    for e in CATALOG:
        if not e["enabled"] and not include_disabled:
            continue
        d = dict(e)
        d["label"] = label(e)
        d["origin_label"] = ORIGIN_LABELS[e["origin"]]
        out.append(d)
    return out


def presets() -> List[dict]:
    """Die vorbelegten Karten (RTX 5090, RTX 3080 20 GB, RTX 3090): der sichtbare Teil des Katalogs."""
    return [e for e in CATALOG if e["preset"]]


def _norm_name(text) -> str:
    return " ".join(str(text or "").lower().split())


def match_nvml(name: str, total_mib: Optional[int] = None, cc=None) -> Optional[dict]:
    """Der Katalogeintrag zu einer NVML-Karte (Name, Größe, Compute Capability) oder ``None``.

    Der NVML-Name muss gleich sein (Groß/Klein und Leerraum egal), ebenso die cc, wenn sie bekannt ist.  Mehrere Einträge
    gleichen Namens (RTX 3080 mit 10/12/20 GB) trennt die Größe: der mit der nächsten ``usable_mib`` gewinnt, aber nur
    innerhalb von 3 % (fremde Karten melden NVML-seitig einige MiB unter Nennwert).  Passt keiner: ``None``, nie geraten."""
    cands = [e for e in CATALOG if _norm_name(e["nvml_name"]) == _norm_name(name)
             and (cc is None or list(cc) == e["cc"])]
    if not cands:
        return None
    if total_mib is None:
        return cands[0] if len(cands) == 1 else None
    best = min(cands, key=lambda e: abs(e["usable_mib"] - total_mib))
    return best if abs(best["usable_mib"] - total_mib) <= 0.03 * best["usable_mib"] else None


def datasheet_of(row: dict) -> dict:
    """Datasheet-Angaben für das Hardwareprofil (``hardware_profile.build(datasheet=...)``): Nennbandbreite und
    Katalogkarte einer NVML-Zeile ``{name, total_mib, cc}``, oder ``{}``.  Die Herkunft steht im Notiztext des Knotens."""
    e = match_nvml(row.get("name"), row.get("total_mib"), row.get("cc"))
    if e is None:
        return {}
    why = e["mem_bw_src"] + " [catalog kartenplan_catalog.py: %s]" % e["id"]
    if ORIGIN_BORROWED in e["origin_fields"].values() and e["origin_fields"]["mem_bw"] == ORIGIN_BORROWED:
        why = "BORROWED, unverified: " + why
    return {"mem_bw_gbs": e["mem_bw_gbs"], "bw_note": why,
            "catalog": {"id": e["id"], "label": label(e), "preset": e["preset"], "origin": e["origin"],
                        "origin_label": ORIGIN_LABELS[e["origin"]], "origin_fields": dict(e["origin_fields"])}}


# --------------------------------------------------------------------------- Profile
#: Modelle/Profile des Kartenplaners (Nutzer-Entscheide 03.10.).  ``record`` nennt die
#: Planer-Aufzeichnung (kartenplan_data/<record>.json); ``release_profile`` die Release-Datei
#: unter docker/profiles_release, ``boot_profile`` das Profil des Referenz-Boots.
PROFILES: List[dict] = [
    {"id": "27b-int8", "label": "27B INT8 (gdncov)", "line": "27b", "format": "int8", "flip": True,
     "release_profile": "27b", "boot_profile": "27b-row-authority-cut43", "record": "27b-int8",
     "note": "Release form: P = PP3 (prefill), D = TP3 (decode), flip; draft DFlash2"},
    {"id": "nf-int4-abl", "label": "NF INT4 (abl)", "line": "nf", "format": "int4", "flip": True,
     "release_profile": "nf-int4", "boot_profile": "nf-int4-h6-abl", "record": "nf-int4-abl",
     "note": "Qwen3.8-Flash-Next (MoE, experts in the host, MTP draft), flip"},
    {"id": "27b-nvfp4-dual", "label": "27B NVFP4 Dual", "line": "27b", "format": "nvfp4", "flip": True,
     "release_profile": "27b-nvfp4-dual", "boot_profile": "27b-nvfp4-dual1m-psleep", "record": "27b-nvfp4-dual",
     "note": "Dual form: P and D alternating, NVFP4 weights"},
    {"id": "27b-fp8", "label": "27B FP8", "line": "27b", "format": "fp8", "flip": True,
     "release_profile": "27b-fp8", "boot_profile": "27b-fp8", "record": "27b-fp8",
     "note": "Hardware evidence only from the Docker boot of 26.09. (before the IPC state), hence without state.json"},
    {"id": "27b-gguf-iq4xs", "label": "27B GGUF UD-IQ4_XS", "line": "27b", "format": "gguf", "flip": True,
     "release_profile": "27b-gguf", "boot_profile": "27b-gguf", "record": "27b-gguf-iq4xs",
     "note": "The only GGUF variant of the 27B booted on the hardware; Q8_K_XL is on disk, was never booted",
     "variants": [{"id": "UD-IQ4_XS", "file": "Qwen3.8-27B-UD-IQ4_XS.gguf",
                   "path": "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-GGUF-unsloth/Qwen3.8-27B-UD-IQ4_XS.gguf",
                   "size_bytes": 14252845984, "size_src": "ls -l, measured",
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
                "why": "sm%d%d: built in the release image (sgl-kernel wheel 86;120a, JIT prebuild 8.6,12.0); booted on the rig." % tup}
    if tup == (8, 9):
        return {"level": "ungebaut", "ok": False,
                "why": "sm89: the release image contains no sm89 code (wheel 86;120a); the arch gate of the launcher refuses cc 8.9 (HW-ARCH, NF line 7a8f4087a9). The port has landed (item 230: NF ec9e28cfa1, 27B partial state 4c7eda3016; FP8 runs there via the named Marlin fallback FP8-SM89-FALLBACK), the sm89 image build is only staged (item 270, /spinning/gpu-arb/docker/sm89-1002/BUILD_SM89.md), unbuilt and untested on the hardware."}
    if tup == (7, 5):
        return {"level": "gesperrt", "ok": False,
                "why": "sm75 (Turing): no bf16, no FP8, no NVFP4; the image builds only for sm86/sm120. Entry disabled until sm75 is added."}
    return {"level": "gesperrt", "ok": False, "why": "Architecture sm%d%d not in the image." % tup}
